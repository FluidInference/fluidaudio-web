import type { AceModelProfileId } from "../capabilities.js";

const REDUCTION_LANES = 256;
const MAX_DISPATCH_DIMENSION = 65_535;

/** Benchmark-only OPT-0075 identity; production remains on WG256. */
export const ACE_OPT_0075_WIDTH128_RMSNORM_KERNEL_ID =
  "ace-opt-0075-reference-bf16-rmsnorm-width128-wg128-v1";
export const ACE_OPT_0075_WIDTH128_RMSNORM_LANES = 128;

export interface AceRmsNormShape {
  readonly rows: number;
  readonly width: number;
  readonly epsilon: number;
}

export interface AceRmsNormPlan extends AceRmsNormShape {
  readonly elements: number;
  readonly workgroupsX: number;
  readonly workgroupsY: number;
}

export interface AceRmsNormBindings {
  readonly input: GPUBufferBinding;
  readonly weight: GPUBufferBinding;
  readonly output: GPUBufferBinding;
}

export interface AceRmsNormDispatch {
  readonly label: string;
  readonly plan: AceRmsNormPlan;
  encode(pass: GPUComputePassEncoder): void;
}

/**
 * FP32-statistics RMSNorm matching the pinned Qwen/ACE inference contract.
 * One workgroup owns one logical row; no partial statistics leave the GPU.
 */
export class AceCorrectnessRmsNormKernel {
  private readonly compiled = new Map<string, Promise<GPUComputePipeline>>();
  private destroyed = false;

  private constructor(
    private readonly device: GPUDevice,
    readonly modelProfile: AceModelProfileId,
  ) {}

  static create(
    device: GPUDevice,
    modelProfile: AceModelProfileId,
  ): AceCorrectnessRmsNormKernel {
    if (modelProfile === "raw-fp16" && !device.features.has("shader-f16")) {
      throw new Error("ACE raw-FP16 RMSNorm requires WebGPU shader-f16");
    }
    if (modelProfile !== "reference-bf16" && modelProfile !== "raw-fp16") {
      throw new TypeError(`Unknown ACE RMSNorm model profile ${String(modelProfile)}`);
    }
    if (
      device.limits.maxComputeInvocationsPerWorkgroup < REDUCTION_LANES ||
      device.limits.maxComputeWorkgroupSizeX < REDUCTION_LANES ||
      device.limits.maxComputeWorkgroupStorageSize <
        REDUCTION_LANES * Float32Array.BYTES_PER_ELEMENT
    ) {
      throw new Error("ACE RMSNorm requires 256 lanes and 1024 bytes of workgroup storage");
    }
    return new AceCorrectnessRmsNormKernel(device, modelProfile);
  }

  async createDispatch(
    label: string,
    shape: AceRmsNormShape,
    bindings: AceRmsNormBindings,
  ): Promise<AceRmsNormDispatch> {
    if (this.destroyed) throw new Error("ACE RMSNorm kernel was destroyed");
    const plan = planAceRmsNorm(shape);
    const elementBytes = this.modelProfile === "reference-bf16" ? 4 : 2;
    requireBindingBytes(bindings.input, plan.elements * elementBytes, `${label} input`);
    requireBindingBytes(bindings.output, plan.elements * elementBytes, `${label} output`);
    requireBindingBytes(
      bindings.weight,
      this.modelProfile === "reference-bf16"
        ? Math.ceil(plan.width / 2) * 4
        : plan.width * 2,
      `${label} weight`,
    );
    const pipeline = await this.pipelineFor(plan);
    if (this.destroyed) {
      throw new Error("ACE RMSNorm kernel was destroyed while compiling");
    }
    const bindGroup = this.device.createBindGroup({
      label: `${label}-bindings`,
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: bindings.input },
        { binding: 1, resource: bindings.weight },
        { binding: 2, resource: bindings.output },
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

  private pipelineFor(plan: AceRmsNormPlan): Promise<GPUComputePipeline> {
    const key = `${plan.rows}x${plan.width}:${plan.epsilon}`;
    const existing = this.compiled.get(key);
    if (existing !== undefined) return existing;
    const label = `ace-correctness-rmsnorm-${this.modelProfile}-${plan.rows}x${plan.width}`;
    const module = this.device.createShaderModule({
      label,
      code: aceCorrectnessRmsNormWgsl(this.modelProfile, plan),
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

/**
 * Benchmark-only width-128 owner. It preserves the WG256 reduction arithmetic
 * by retaining the inert stride-128 +0 locally before the shared stride-64
 * tree, while never launching the 128 lanes that only supplied those zeros.
 */
export class AceOpt0075Width128RmsNormKernel {
  private readonly compiled = new Map<string, Promise<GPUComputePipeline>>();
  private destroyed = false;

  private constructor(private readonly device: GPUDevice) {}

  static create(device: GPUDevice): AceOpt0075Width128RmsNormKernel {
    if (
      device.limits.maxComputeInvocationsPerWorkgroup <
        ACE_OPT_0075_WIDTH128_RMSNORM_LANES ||
      device.limits.maxComputeWorkgroupSizeX <
        ACE_OPT_0075_WIDTH128_RMSNORM_LANES ||
      device.limits.maxComputeWorkgroupStorageSize <
        ACE_OPT_0075_WIDTH128_RMSNORM_LANES *
          Float32Array.BYTES_PER_ELEMENT
    ) {
      throw new Error(
        "OPT-0075 width-128 RMSNorm requires 128 lanes and 512 bytes of workgroup storage",
      );
    }
    return new AceOpt0075Width128RmsNormKernel(device);
  }

  async createDispatch(
    label: string,
    shape: AceRmsNormShape,
    bindings: AceRmsNormBindings,
  ): Promise<AceRmsNormDispatch> {
    if (this.destroyed) {
      throw new Error("OPT-0075 width-128 RMSNorm kernel was destroyed");
    }
    const plan = planAceOpt0075Width128RmsNorm(shape);
    requireBindingBytes(bindings.input, plan.elements * 4, `${label} input`);
    requireBindingBytes(bindings.output, plan.elements * 4, `${label} output`);
    requireBindingBytes(
      bindings.weight,
      Math.ceil(plan.width / 2) * 4,
      `${label} weight`,
    );
    const pipeline = await this.pipelineFor(plan);
    if (this.destroyed) {
      throw new Error(
        "OPT-0075 width-128 RMSNorm kernel was destroyed while compiling",
      );
    }
    const bindGroup = this.device.createBindGroup({
      label: `${label}-opt-0075-bindings`,
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: bindings.input },
        { binding: 1, resource: bindings.weight },
        { binding: 2, resource: bindings.output },
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
    if (this.destroyed) return;
    this.destroyed = true;
    this.compiled.clear();
  }

  private pipelineFor(plan: AceRmsNormPlan): Promise<GPUComputePipeline> {
    const key = `${plan.rows}x${plan.width}:${plan.epsilon}`;
    const existing = this.compiled.get(key);
    if (existing !== undefined) return existing;
    const created = compileAceOpt0075Width128RmsNorm(this.device, plan);
    this.compiled.set(key, created);
    void created.catch(() => {
      if (this.compiled.get(key) === created) this.compiled.delete(key);
    });
    return created;
  }
}

export function planAceRmsNorm(shape: AceRmsNormShape): AceRmsNormPlan {
  const { rows, width, epsilon } = shape;
  if (!Number.isSafeInteger(rows) || rows <= 0) {
    throw new RangeError("ACE RMSNorm rows must be a positive safe integer");
  }
  if (!Number.isSafeInteger(width) || width <= 0) {
    throw new RangeError("ACE RMSNorm width must be a positive safe integer");
  }
  if (typeof epsilon !== "number" || !Number.isFinite(epsilon) || epsilon <= 0) {
    throw new RangeError("ACE RMSNorm epsilon must be positive and finite");
  }
  const elements = rows * width;
  if (!Number.isSafeInteger(elements)) {
    throw new RangeError("ACE RMSNorm element count is not a safe integer");
  }
  const workgroupsX = Math.min(rows, MAX_DISPATCH_DIMENSION);
  const workgroupsY = Math.ceil(rows / workgroupsX);
  if (workgroupsY > MAX_DISPATCH_DIMENSION) {
    throw new RangeError("ACE RMSNorm exceeds the portable 2D dispatch domain");
  }
  return Object.freeze({ rows, width, epsilon, elements, workgroupsX, workgroupsY });
}

export function planAceOpt0075Width128RmsNorm(
  shape: AceRmsNormShape,
): AceRmsNormPlan {
  const plan = planAceRmsNorm(shape);
  if (plan.width !== ACE_OPT_0075_WIDTH128_RMSNORM_LANES) {
    throw new RangeError("OPT-0075 RMSNorm accepts only width 128");
  }
  return plan;
}

export function aceCorrectnessRmsNormWgsl(
  modelProfile: AceModelProfileId,
  shape: AceRmsNormShape,
): string {
  const plan = planAceRmsNorm(shape);
  const preamble = modelProfile === "raw-fp16" ? "enable f16;\n" : "";
  const inputType = modelProfile === "raw-fp16" ? "f16" : "f32";
  const weightType = modelProfile === "raw-fp16" ? "f16" : "u32";
  const outputType = inputType;
  const loadInput = modelProfile === "raw-fp16"
    ? "f32(input[index])"
    : "input[index]";
  const storeOutput = modelProfile === "raw-fp16"
    ? "output[index] = f16(value * inverse_rms) * weight[column];"
    : "output[index] = value * inverse_rms * load_weight(column);";
  const weightLoader = modelProfile === "reference-bf16"
    ? `
fn load_weight(index: u32) -> f32 {
  let pair = weight[index >> 1u];
  let bits16 = select(pair >> 16u, pair & 0xffffu, (index & 1u) == 0u);
  return bitcast<f32>(bits16 << 16u);
}
`
    : "";
  if (modelProfile !== "reference-bf16" && modelProfile !== "raw-fp16") {
    throw new TypeError(`Unknown ACE RMSNorm model profile ${String(modelProfile)}`);
  }
  return /* wgsl */ `${preamble}
const ROWS: u32 = ${plan.rows}u;
const WIDTH: u32 = ${plan.width}u;
const DISPATCH_X: u32 = ${plan.workgroupsX}u;
const EPSILON: f32 = ${plan.epsilon};

@group(0) @binding(0) var<storage, read> input: array<${inputType}>;
@group(0) @binding(1) var<storage, read> weight: array<${weightType}>;
@group(0) @binding(2) var<storage, read_write> output: array<${outputType}>;
var<workgroup> partial_squares: array<f32, ${REDUCTION_LANES}>;
${weightLoader}
@compute @workgroup_size(${REDUCTION_LANES}, 1, 1)
fn main(
  @builtin(workgroup_id) workgroup_id: vec3<u32>,
  @builtin(local_invocation_index) lane: u32,
) {
  let row = workgroup_id.y * DISPATCH_X + workgroup_id.x;
  if (row >= ROWS) { return; }
  let row_base = row * WIDTH;
  var square_sum = 0.0;
  for (var column = lane; column < WIDTH; column += ${REDUCTION_LANES}u) {
    let value = ${loadInput.replaceAll("index", "row_base + column")};
    square_sum = square_sum + value * value;
  }
  partial_squares[lane] = square_sum;
  workgroupBarrier();

  var stride = ${REDUCTION_LANES / 2}u;
  while (stride > 0u) {
    if (lane < stride) {
      partial_squares[lane] = partial_squares[lane] + partial_squares[lane + stride];
    }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  workgroupBarrier();
  let inverse_rms = inverseSqrt(partial_squares[0] / f32(WIDTH) + EPSILON);
  for (var column = lane; column < WIDTH; column += ${REDUCTION_LANES}u) {
    let index = row_base + column;
    let value = ${loadInput};
    ${storeOutput}
  }
}
`;
}

export function aceOpt0075Width128RmsNormWgsl(
  shape: AceRmsNormShape,
): string {
  const plan = planAceOpt0075Width128RmsNorm(shape);
  return /* wgsl */ `
const ROWS: u32 = ${plan.rows}u;
const WIDTH: u32 = ${plan.width}u;
const DISPATCH_X: u32 = ${plan.workgroupsX}u;
const EPSILON: f32 = ${plan.epsilon};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
var<workgroup> partial_squares: array<f32, ${ACE_OPT_0075_WIDTH128_RMSNORM_LANES}>;

fn load_weight(index: u32) -> f32 {
  let pair = weight[index >> 1u];
  let bits16 = select(pair >> 16u, pair & 0xffffu, (index & 1u) == 0u);
  return bitcast<f32>(bits16 << 16u);
}

@compute @workgroup_size(${ACE_OPT_0075_WIDTH128_RMSNORM_LANES}, 1, 1)
fn main(
  @builtin(workgroup_id) workgroup_id: vec3<u32>,
  @builtin(local_invocation_index) lane: u32,
) {
  let row = workgroup_id.y * DISPATCH_X + workgroup_id.x;
  if (row >= ROWS) { return; }
  let row_base = row * WIDTH;
  var square_sum = 0.0;
  for (var column = lane; column < WIDTH;
    column += ${ACE_OPT_0075_WIDTH128_RMSNORM_LANES}u) {
    let value = input[row_base + column];
    square_sum = square_sum + value * value;
  }
  // Preserve the WG256 stride-128 addition against its empty upper lane.
  square_sum = square_sum + 0.0;
  partial_squares[lane] = square_sum;
  workgroupBarrier();

  var stride = ${ACE_OPT_0075_WIDTH128_RMSNORM_LANES / 2}u;
  while (stride > 0u) {
    if (lane < stride) {
      partial_squares[lane] = partial_squares[lane] +
        partial_squares[lane + stride];
    }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  workgroupBarrier();
  let inverse_rms = inverseSqrt(partial_squares[0] / f32(WIDTH) + EPSILON);
  for (var column = lane; column < WIDTH;
    column += ${ACE_OPT_0075_WIDTH128_RMSNORM_LANES}u) {
    let index = row_base + column;
    let value = input[index];
    output[index] = value * inverse_rms * load_weight(column);
  }
}
`;
}

async function compileAceOpt0075Width128RmsNorm(
  device: GPUDevice,
  plan: AceRmsNormPlan,
): Promise<GPUComputePipeline> {
  const label = `ace-opt-0075-rmsnorm-${plan.rows}x${plan.width}`;
  const module = device.createShaderModule({
    label,
    code: aceOpt0075Width128RmsNormWgsl(plan),
  });
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((message) => message.type === "error");
  if (errors.length !== 0) {
    throw new Error(
      `${label} WGSL compilation failed:\n` + errors.map((message) =>
        `${message.lineNum}:${message.linePos} ${message.message}`
      ).join("\n"),
    );
  }
  return await device.createComputePipelineAsync({
    label,
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });
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
