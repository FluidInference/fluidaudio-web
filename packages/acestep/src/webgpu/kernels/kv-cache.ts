import type { AceModelProfileId } from "../capabilities.js";

const KV_CACHE_WRITE_LANES = 64;
const MAX_DISPATCH_DIMENSION = 65_535;
const MAX_U32 = 0xffff_ffff;

export interface AceKvCacheWriteShape {
  readonly batch: number;
  readonly keyValueHeads: number;
  readonly appendTokens: number;
  readonly cacheCapacity: number;
  readonly headDimension: number;
}

export interface AceKvCacheWritePlan extends AceKvCacheWriteShape {
  readonly sourceElements: number;
  readonly cacheElements: number;
  readonly sourceValidityElements: number;
  readonly cacheValidityElements: number;
  readonly workgroupsX: number;
  readonly workgroupsY: number;
}

export interface AceKvCacheWriteBindings {
  /** `[batch, keyValueHeads, appendTokens, headDimension]`. */
  readonly sourceKey: GPUBufferBinding;
  readonly sourceValue: GPUBufferBinding;
  /** U32 `[batch, appendTokens]`; exactly `1` means visible to attention. */
  readonly sourceValidity: GPUBufferBinding;
  /** `[batch, keyValueHeads, cacheCapacity, headDimension]`. */
  readonly cacheKey: GPUBufferBinding;
  readonly cacheValue: GPUBufferBinding;
  /** U32 `[batch, cacheCapacity]`, updated for every admitted source token. */
  readonly cacheValidity: GPUBufferBinding;
  /** U32 `[batch]`, the physical destination start for each row. */
  readonly rowStartPositions: GPUBufferBinding;
  /** U32 `[batch]`; written as `1` for an admitted row and `0` on range failure. */
  readonly writeStatus: GPUBufferBinding;
}

export interface AceKvCacheWriteDispatch {
  readonly label: string;
  readonly plan: AceKvCacheWritePlan;
  encode(pass: GPUComputePassEncoder): void;
}

/**
 * Untuned planner KV-cache writer for correctness work.
 *
 * Source and cache use the exact `[B, KVH, T, D]` layout consumed by
 * `AceCorrectnessAttentionKernel`. Each row has one physical start position,
 * making a multi-token write contiguous and preventing duplicate-destination
 * races by construction. The shader validates the complete row range before
 * writing any K, V, or validity element. The graph owner must still avoid
 * submitting this write while an earlier attention quantum reads the aliases.
 */
export class AceCorrectnessKvCacheWriteKernel {
  private readonly compiled = new Map<string, Promise<GPUComputePipeline>>();
  private destroyed = false;

  private constructor(
    private readonly device: GPUDevice,
    readonly modelProfile: AceModelProfileId,
  ) {}

  static create(
    device: GPUDevice,
    modelProfile: AceModelProfileId,
  ): AceCorrectnessKvCacheWriteKernel {
    if (modelProfile === "raw-fp16" && !device.features.has("shader-f16")) {
      throw new Error("ACE raw-FP16 KV-cache writing requires WebGPU shader-f16");
    }
    if (modelProfile !== "reference-bf16" && modelProfile !== "raw-fp16") {
      throw new TypeError(`Unknown ACE KV-cache model profile ${String(modelProfile)}`);
    }
    if (
      device.limits.maxComputeInvocationsPerWorkgroup < KV_CACHE_WRITE_LANES ||
      device.limits.maxComputeWorkgroupSizeX < KV_CACHE_WRITE_LANES
    ) {
      throw new Error("ACE KV-cache writing requires a 64-lane compute workgroup");
    }
    return new AceCorrectnessKvCacheWriteKernel(device, modelProfile);
  }

  async createDispatch(
    label: string,
    shape: AceKvCacheWriteShape,
    bindings: AceKvCacheWriteBindings,
  ): Promise<AceKvCacheWriteDispatch> {
    if (this.destroyed) throw new Error("ACE KV-cache writer was destroyed");
    const plan = planAceKvCacheWrite(shape);
    const elementBytes = this.modelProfile === "raw-fp16" ? 2 : 4;
    const logicalBindings: readonly LogicalBinding[] = [
      logicalBinding(bindings.sourceKey, plan.sourceElements * elementBytes, `${label} source key`),
      logicalBinding(
        bindings.sourceValue,
        plan.sourceElements * elementBytes,
        `${label} source value`,
      ),
      logicalBinding(
        bindings.sourceValidity,
        plan.sourceValidityElements * 4,
        `${label} source validity`,
      ),
      logicalBinding(bindings.cacheKey, plan.cacheElements * elementBytes, `${label} cache key`),
      logicalBinding(
        bindings.cacheValue,
        plan.cacheElements * elementBytes,
        `${label} cache value`,
      ),
      logicalBinding(
        bindings.cacheValidity,
        plan.cacheValidityElements * 4,
        `${label} cache validity`,
      ),
      logicalBinding(bindings.rowStartPositions, plan.batch * 4, `${label} row starts`),
      logicalBinding(bindings.writeStatus, plan.batch * 4, `${label} write status`),
    ];
    requireDisjointLogicalBindings(logicalBindings);

    const pipeline = await this.pipelineFor(plan);
    if (this.destroyed) {
      throw new Error("ACE KV-cache writer was destroyed while compiling");
    }
    const bindGroup = this.device.createBindGroup({
      label: `${label}-bindings`,
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: bindings.sourceKey },
        { binding: 1, resource: bindings.sourceValue },
        { binding: 2, resource: bindings.sourceValidity },
        { binding: 3, resource: bindings.cacheKey },
        { binding: 4, resource: bindings.cacheValue },
        { binding: 5, resource: bindings.cacheValidity },
        { binding: 6, resource: bindings.rowStartPositions },
        { binding: 7, resource: bindings.writeStatus },
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

  private pipelineFor(plan: AceKvCacheWritePlan): Promise<GPUComputePipeline> {
    const key = [
      plan.batch,
      plan.keyValueHeads,
      plan.appendTokens,
      plan.cacheCapacity,
      plan.headDimension,
    ].join("x");
    const existing = this.compiled.get(key);
    if (existing !== undefined) return existing;
    const label = `ace-correctness-kv-cache-write-${this.modelProfile}-${key}`;
    const module = this.device.createShaderModule({
      label,
      code: aceCorrectnessKvCacheWriteWgsl(this.modelProfile, plan),
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

export function planAceKvCacheWrite(shape: AceKvCacheWriteShape): AceKvCacheWritePlan {
  const { batch, keyValueHeads, appendTokens, cacheCapacity, headDimension } = shape;
  for (const [name, value] of Object.entries({
    batch,
    keyValueHeads,
    appendTokens,
    cacheCapacity,
    headDimension,
  })) {
    if (!Number.isSafeInteger(value) || value <= 0 || value > MAX_U32) {
      throw new RangeError(`ACE KV-cache ${name} must be a positive U32 integer`);
    }
  }
  if (appendTokens > cacheCapacity) {
    throw new RangeError("ACE KV-cache appendTokens cannot exceed cacheCapacity");
  }
  const sourceElements = checkedProduct(
    checkedProduct(checkedProduct(batch, keyValueHeads), appendTokens),
    headDimension,
  );
  const cacheElements = checkedProduct(
    checkedProduct(checkedProduct(batch, keyValueHeads), cacheCapacity),
    headDimension,
  );
  const sourceValidityElements = checkedProduct(batch, appendTokens);
  const cacheValidityElements = checkedProduct(batch, cacheCapacity);
  if (
    sourceElements > MAX_U32 ||
    cacheElements > MAX_U32 ||
    sourceValidityElements > MAX_U32 ||
    cacheValidityElements > MAX_U32
  ) {
    throw new RangeError("ACE KV-cache tensor indexing exceeds the U32 shader domain");
  }
  const workgroups = Math.ceil(sourceElements / KV_CACHE_WRITE_LANES);
  const workgroupsX = Math.min(workgroups, MAX_DISPATCH_DIMENSION);
  const workgroupsY = Math.ceil(workgroups / workgroupsX);
  if (workgroupsY > MAX_DISPATCH_DIMENSION) {
    throw new RangeError("ACE KV-cache write exceeds the portable 2D dispatch domain");
  }
  return Object.freeze({
    batch,
    keyValueHeads,
    appendTokens,
    cacheCapacity,
    headDimension,
    sourceElements,
    cacheElements,
    sourceValidityElements,
    cacheValidityElements,
    workgroupsX,
    workgroupsY,
  });
}

export function aceCorrectnessKvCacheWriteWgsl(
  modelProfile: AceModelProfileId,
  shape: AceKvCacheWriteShape,
): string {
  const plan = planAceKvCacheWrite(shape);
  if (modelProfile !== "reference-bf16" && modelProfile !== "raw-fp16") {
    throw new TypeError(`Unknown ACE KV-cache model profile ${String(modelProfile)}`);
  }
  const scalar = modelProfile === "raw-fp16" ? "f16" : "f32";
  return /* wgsl */ `${modelProfile === "raw-fp16" ? "enable f16;" : ""}
const KV_HEADS: u32 = ${plan.keyValueHeads}u;
const APPEND_TOKENS: u32 = ${plan.appendTokens}u;
const CACHE_CAPACITY: u32 = ${plan.cacheCapacity}u;
const HEAD_DIMENSION: u32 = ${plan.headDimension}u;
const SOURCE_ELEMENTS: u32 = ${plan.sourceElements}u;
const SOURCE_ELEMENTS_PER_BATCH: u32 = ${plan.keyValueHeads * plan.appendTokens * plan.headDimension}u;
const DISPATCH_X: u32 = ${plan.workgroupsX}u;

@group(0) @binding(0) var<storage, read> source_key: array<${scalar}>;
@group(0) @binding(1) var<storage, read> source_value: array<${scalar}>;
@group(0) @binding(2) var<storage, read> source_validity: array<u32>;
@group(0) @binding(3) var<storage, read_write> cache_key: array<${scalar}>;
@group(0) @binding(4) var<storage, read_write> cache_value: array<${scalar}>;
@group(0) @binding(5) var<storage, read_write> cache_validity: array<u32>;
@group(0) @binding(6) var<storage, read> row_start_positions: array<u32>;
@group(0) @binding(7) var<storage, read_write> write_status: array<u32>;

@compute @workgroup_size(${KV_CACHE_WRITE_LANES}, 1, 1)
fn main(
  @builtin(workgroup_id) workgroup_id: vec3<u32>,
  @builtin(local_invocation_index) lane: u32,
) {
  let flat_workgroup = workgroup_id.y * DISPATCH_X + workgroup_id.x;
  let source_index = flat_workgroup * ${KV_CACHE_WRITE_LANES}u + lane;
  if (source_index >= SOURCE_ELEMENTS) { return; }

  let batch = source_index / SOURCE_ELEMENTS_PER_BATCH;
  let within_batch = source_index % SOURCE_ELEMENTS_PER_BATCH;
  let elements_per_head = APPEND_TOKENS * HEAD_DIMENSION;
  let kv_head = within_batch / elements_per_head;
  let within_head = within_batch % elements_per_head;
  let append_token = within_head / HEAD_DIMENSION;
  let dimension = within_head % HEAD_DIMENSION;
  let row_start = row_start_positions[batch];
  let last_allowed_start = CACHE_CAPACITY - APPEND_TOKENS;
  let row_range_is_valid = row_start <= last_allowed_start;

  if (within_batch == 0u) {
    write_status[batch] = select(0u, 1u, row_range_is_valid);
  }
  if (!row_range_is_valid) { return; }

  let destination_token = row_start + append_token;
  let destination_index =
    ((batch * KV_HEADS + kv_head) * CACHE_CAPACITY + destination_token) *
      HEAD_DIMENSION + dimension;
  cache_key[destination_index] = source_key[source_index];
  cache_value[destination_index] = source_value[source_index];
  if (kv_head == 0u && dimension == 0u) {
    let source_mask = source_validity[batch * APPEND_TOKENS + append_token];
    cache_validity[batch * CACHE_CAPACITY + destination_token] =
      select(0u, 1u, source_mask == 1u);
  }
}
`;
}

interface LogicalBinding {
  readonly buffer: GPUBuffer;
  readonly start: number;
  readonly end: number;
  readonly label: string;
}

function logicalBinding(
  binding: GPUBufferBinding,
  requiredBytes: number,
  label: string,
): LogicalBinding {
  const offset = binding.offset ?? 0;
  const available = binding.size ?? binding.buffer.size - offset;
  if (
    !Number.isSafeInteger(offset) ||
    offset < 0 ||
    !Number.isSafeInteger(available) ||
    available < requiredBytes ||
    offset + available > binding.buffer.size
  ) {
    throw new RangeError(`${label} binding does not expose ${requiredBytes} bytes`);
  }
  return Object.freeze({
    buffer: binding.buffer,
    start: offset,
    end: offset + requiredBytes,
    label,
  });
}

function requireDisjointLogicalBindings(bindings: readonly LogicalBinding[]): void {
  for (let leftIndex = 0; leftIndex < bindings.length; leftIndex += 1) {
    const left = bindings[leftIndex]!;
    for (let rightIndex = leftIndex + 1; rightIndex < bindings.length; rightIndex += 1) {
      const right = bindings[rightIndex]!;
      if (
        left.buffer === right.buffer &&
        left.start < right.end &&
        right.start < left.end
      ) {
        throw new RangeError(`${left.label} overlaps ${right.label}`);
      }
    }
  }
}

function checkedProduct(left: number, right: number): number {
  const product = left * right;
  if (!Number.isSafeInteger(product)) {
    throw new RangeError("ACE KV-cache element count is not a safe integer");
  }
  return product;
}
