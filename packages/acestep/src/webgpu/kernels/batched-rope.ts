import type { AceModelProfileId } from "../capabilities.js";

const WORKGROUP_SIZE = 256;
const MAX_DISPATCH_DIMENSION = 65_535;
const MAX_U32 = 0xffff_ffff;

export interface AceBatchedRopeShape {
  readonly batch: number;
  readonly heads: number;
  readonly tokens: number;
  readonly headDimension: number;
}

export interface AceBatchedRopePlan extends AceBatchedRopeShape {
  readonly scalarCount: number;
  readonly tableScalarCount: number;
  readonly workgroupsX: number;
  readonly workgroupsY: number;
}

export interface AceBatchedRopeBindings {
  readonly input: GPUBufferBinding;
  /** FP32 `[batch,tokens,headDimension]`, Qwen duplicated-half form. */
  readonly cosine: GPUBufferBinding;
  readonly sine: GPUBufferBinding;
  readonly output: GPUBufferBinding;
}

export interface AceBatchedRopeDispatch {
  readonly label: string;
  readonly plan: AceBatchedRopePlan;
  encode(pass: GPUComputePassEncoder): void;
}

/** Qwen RoPE with independent position IDs for every batch row. */
export class AceCorrectnessBatchedRopeKernel {
  private readonly compiled = new Map<string, Promise<GPUComputePipeline>>();
  private destroyed = false;

  private constructor(
    private readonly device: GPUDevice,
    readonly modelProfile: AceModelProfileId,
  ) {}

  static create(
    device: GPUDevice,
    modelProfile: AceModelProfileId,
  ): AceCorrectnessBatchedRopeKernel {
    if (modelProfile === "raw-fp16" && !device.features.has("shader-f16")) {
      throw new Error("ACE batched RoPE requires WebGPU shader-f16 for raw FP16");
    }
    if (modelProfile !== "reference-bf16" && modelProfile !== "raw-fp16") {
      throw new TypeError(`Unknown ACE batched RoPE model profile ${String(modelProfile)}`);
    }
    if (
      device.limits.maxComputeInvocationsPerWorkgroup < WORKGROUP_SIZE ||
      device.limits.maxComputeWorkgroupSizeX < WORKGROUP_SIZE
    ) {
      throw new Error("ACE batched RoPE requires 256 compute lanes");
    }
    return new AceCorrectnessBatchedRopeKernel(device, modelProfile);
  }

  async createDispatch(
    label: string,
    shape: AceBatchedRopeShape,
    bindings: AceBatchedRopeBindings,
  ): Promise<AceBatchedRopeDispatch> {
    if (this.destroyed) throw new Error("ACE batched RoPE kernel was destroyed");
    const plan = planAceBatchedRope(shape);
    const elementBytes = this.modelProfile === "raw-fp16" ? 2 : 4;
    requireBindingBytes(bindings.input, plan.scalarCount * elementBytes, `${label} input`);
    requireBindingBytes(bindings.output, plan.scalarCount * elementBytes, `${label} output`);
    requireBindingBytes(bindings.cosine, plan.tableScalarCount * 4, `${label} cosine`);
    requireBindingBytes(bindings.sine, plan.tableScalarCount * 4, `${label} sine`);
    requireDisjointOutput(label, bindings.output, [
      bindings.input,
      bindings.cosine,
      bindings.sine,
    ]);
    const pipeline = await this.pipelineFor(plan);
    if (this.destroyed) {
      throw new Error("ACE batched RoPE kernel was destroyed while compiling");
    }
    const bindGroup = this.device.createBindGroup({
      label: `${label}-bindings`,
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: bindings.input },
        { binding: 1, resource: bindings.cosine },
        { binding: 2, resource: bindings.sine },
        { binding: 3, resource: bindings.output },
      ],
    });
    return Object.freeze({
      label,
      plan,
      encode(pass: GPUComputePassEncoder): void {
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(plan.workgroupsX, plan.workgroupsY, 1);
      },
    });
  }

  destroy(): void {
    this.destroyed = true;
    this.compiled.clear();
  }

  private pipelineFor(plan: AceBatchedRopePlan): Promise<GPUComputePipeline> {
    const key = `${plan.batch}x${plan.heads}x${plan.tokens}x${plan.headDimension}`;
    const existing = this.compiled.get(key);
    if (existing !== undefined) return existing;
    const label = `ace-correctness-batched-rope-${this.modelProfile}-${key}`;
    const module = this.device.createShaderModule({
      label,
      code: aceCorrectnessBatchedRopeWgsl(this.modelProfile, plan),
    });
    const created = this.device.createComputePipelineAsync({
      label,
      layout: "auto",
      compute: { module, entryPoint: "main" },
    });
    this.compiled.set(key, created);
    void created.catch(() => {
      if (this.compiled.get(key) === created) this.compiled.delete(key);
    });
    return created;
  }
}

export function planAceBatchedRope(shape: AceBatchedRopeShape): AceBatchedRopePlan {
  for (const [name, value] of Object.entries(shape)) {
    if (!Number.isSafeInteger(value) || value <= 0 || value > MAX_U32) {
      throw new RangeError(`ACE batched RoPE ${name} must be a positive U32 integer`);
    }
  }
  if (shape.headDimension % 2 !== 0) {
    throw new RangeError("ACE batched RoPE headDimension must be even");
  }
  const scalarCount = checkedProduct([
    shape.batch,
    shape.heads,
    shape.tokens,
    shape.headDimension,
  ]);
  const tableScalarCount = checkedProduct([
    shape.batch,
    shape.tokens,
    shape.headDimension,
  ]);
  if (scalarCount > MAX_U32 || tableScalarCount > MAX_U32) {
    throw new RangeError("ACE batched RoPE indexing exceeds U32");
  }
  const totalWorkgroups = Math.ceil(scalarCount / WORKGROUP_SIZE);
  const workgroupsX = Math.min(totalWorkgroups, MAX_DISPATCH_DIMENSION);
  const workgroupsY = Math.ceil(totalWorkgroups / workgroupsX);
  if (workgroupsY > MAX_DISPATCH_DIMENSION) {
    throw new RangeError("ACE batched RoPE exceeds the portable 2D dispatch domain");
  }
  return Object.freeze({
    ...shape,
    scalarCount,
    tableScalarCount,
    workgroupsX,
    workgroupsY,
  });
}

export function aceCorrectnessBatchedRopeWgsl(
  modelProfile: AceModelProfileId,
  shape: AceBatchedRopeShape,
): string {
  const plan = planAceBatchedRope(shape);
  if (modelProfile !== "reference-bf16" && modelProfile !== "raw-fp16") {
    throw new TypeError(`Unknown ACE batched RoPE model profile ${String(modelProfile)}`);
  }
  const f16 = modelProfile === "raw-fp16";
  return /* wgsl */ `${f16 ? "enable f16;" : ""}
const SCALAR_COUNT: u32 = ${plan.scalarCount}u;
const HEADS: u32 = ${plan.heads}u;
const TOKENS: u32 = ${plan.tokens}u;
const HEAD_DIMENSION: u32 = ${plan.headDimension}u;
const HALF_HEAD: u32 = ${plan.headDimension / 2}u;
const DISPATCH_X: u32 = ${plan.workgroupsX}u;

@group(0) @binding(0) var<storage, read> input: array<${f16 ? "f16" : "f32"}>;
@group(0) @binding(1) var<storage, read> cosine: array<f32>;
@group(0) @binding(2) var<storage, read> sine: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<${f16 ? "f16" : "f32"}>;

@compute @workgroup_size(${WORKGROUP_SIZE}, 1, 1)
fn main(
  @builtin(workgroup_id) workgroup_id: vec3<u32>,
  @builtin(local_invocation_index) lane: u32,
) {
  let workgroup = workgroup_id.y * DISPATCH_X + workgroup_id.x;
  let index = workgroup * ${WORKGROUP_SIZE}u + lane;
  if (index >= SCALAR_COUNT) { return; }

  let dimension = index % HEAD_DIMENSION;
  let token = (index / HEAD_DIMENSION) % TOKENS;
  let head = (index / (HEAD_DIMENSION * TOKENS)) % HEADS;
  let batch = index / (HEAD_DIMENSION * TOKENS * HEADS);
  let rotated_dimension = select(
    dimension + HALF_HEAD,
    dimension - HALF_HEAD,
    dimension >= HALF_HEAD,
  );
  let rotated_index = index - dimension + rotated_dimension;
  let table_index = (batch * TOKENS + token) * HEAD_DIMENSION + dimension;
  let rotated_sign = select(1.0, -1.0, dimension < HALF_HEAD);
  let value = f32(input[index]) * cosine[table_index] +
    rotated_sign * f32(input[rotated_index]) * sine[table_index];
  output[index] = ${f16 ? "f16(value)" : "value"};
}
`;
}

function requireBindingBytes(
  binding: GPUBufferBinding,
  required: number,
  label: string,
): void {
  const offset = binding.offset ?? 0;
  const available = binding.size ?? binding.buffer.size - offset;
  if (
    !Number.isSafeInteger(offset) ||
    offset < 0 ||
    !Number.isSafeInteger(available) ||
    available < required ||
    offset + available > binding.buffer.size
  ) {
    throw new RangeError(`${label} binding does not expose ${required} bytes`);
  }
}

function requireDisjointOutput(
  label: string,
  output: GPUBufferBinding,
  inputs: readonly GPUBufferBinding[],
): void {
  const outputRange = bindingRange(output);
  for (const input of inputs) {
    if (input.buffer !== output.buffer) continue;
    const inputRange = bindingRange(input);
    if (outputRange.start < inputRange.end && inputRange.start < outputRange.end) {
      throw new RangeError(`${label} output overlaps an input`);
    }
  }
}

function bindingRange(binding: GPUBufferBinding): { readonly start: number; readonly end: number } {
  const start = binding.offset ?? 0;
  const size = binding.size ?? binding.buffer.size - start;
  return { start, end: start + size };
}

function checkedProduct(values: readonly number[]): number {
  let product = 1;
  for (const value of values) {
    product *= value;
    if (!Number.isSafeInteger(product)) {
      throw new RangeError("ACE batched RoPE scalar count is not a safe integer");
    }
  }
  return product;
}
