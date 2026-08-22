import {
  ACE_OPT_0025_VAE_K1_SUBGROUP_SIZE,
  ACE_OPT_0025_VAE_K1_TILE_COLUMNS,
  ACE_OPT_0025_VAE_K1_TILE_INNER,
  ACE_OPT_0025_VAE_K1_TILE_ROWS,
  ACE_OPT_0025_VAE_K1_WORKGROUP_SIZE,
  planAceOpt0025VaeK1SubgroupGemm,
  type AceOpt0025VaeK1SubgroupGemmPlan,
} from "./vae-k1-fp16-subgroup-gemm.js";
import type { AceVaeConv1dShape } from "./vae-primitives.js";

export const ACE_OPT_0063_VAE_K1_ADD_SNAKE_EXACT_FUSED_KERNEL_ID =
  "opt-0063-vae-k1-add-snake-exact-fused-v1" as const;
export const ACE_OPT_0063_VAE_K1_ADD_SNAKE_STORAGE_BINDINGS = 8;

const F16_BYTES = 2;
const STORAGE_ALIGNMENT = 4;
const MAX_DISPATCH_DIMENSION = 65_535;
const SUBGROUPS_PER_WORKGROUP =
  ACE_OPT_0025_VAE_K1_WORKGROUP_SIZE / ACE_OPT_0025_VAE_K1_SUBGROUP_SIZE;
const ROWS_PER_SUBGROUP =
  ACE_OPT_0025_VAE_K1_TILE_ROWS / SUBGROUPS_PER_WORKGROUP;
const COLUMNS_PER_LANE =
  ACE_OPT_0025_VAE_K1_TILE_COLUMNS / ACE_OPT_0025_VAE_K1_SUBGROUP_SIZE;

export interface AceOpt0063VaeK1AddSnakePlan {
  readonly shape: AceVaeConv1dShape;
  readonly k1: AceOpt0025VaeK1SubgroupGemmPlan;
  readonly elements: number;
  readonly activationStorageBytes: number;
  readonly activationBindingBytes: number;
  readonly parameterStorageBytes: number;
  readonly parameterBindingBytes: number;
  readonly workgroupCount: number;
  readonly storageBindingCount:
    typeof ACE_OPT_0063_VAE_K1_ADD_SNAKE_STORAGE_BINDINGS;
  readonly formerBoundaries: readonly [
    "producer-f32-to-f16",
    "add-f32-to-f16",
    "snake-f16-to-f32",
    "snake-f32-to-f16",
  ];
}

export interface AceOpt0063VaeK1AddSnakeBindings {
  /** FP16 activation entering the unchanged OPT-0025 K1 producer. */
  readonly input: GPUBufferBinding;
  /** Converter-native OPT-0025 tile-major FP16 K1 weight. */
  readonly packedWeight: GPUBufferBinding;
  /** FP16 K1 bias. */
  readonly bias: GPUBufferBinding;
  /** FP16 residual skip, the left operand of the production Add. */
  readonly skip: GPUBufferBinding;
  /** FP16 successor-Snake log alpha. */
  readonly alpha: GPUBufferBinding;
  /** FP16 successor-Snake log beta. */
  readonly beta: GPUBufferBinding;
  /** Diagnostic/raw residual value at the former Add storage boundary. */
  readonly addOutput: GPUBufferBinding;
  /** FP16 successor-Snake output. */
  readonly snakeOutput: GPUBufferBinding;
}

export interface AceOpt0063VaeK1AddSnakeDispatch {
  readonly label: string;
  readonly kernelId:
    typeof ACE_OPT_0063_VAE_K1_ADD_SNAKE_EXACT_FUSED_KERNEL_ID;
  readonly plan: AceOpt0063VaeK1AddSnakePlan;
  encode(pass: GPUComputePassEncoder): void;
}

export interface AceOpt0063VaeK1AddSnakeOwner {
  readonly workgroup: number;
  readonly localInvocation: number;
  readonly subgroup: number;
  readonly subgroupLane: number;
  readonly rowBase: number;
  readonly columnBase: number;
  readonly outputIndices: readonly number[];
}

interface CompiledKernel {
  readonly pipeline: GPUComputePipeline;
  readonly bindGroupLayout: GPUBindGroupLayout;
}

/** Benchmark-only owner for the first bounded OPT-0063 screen. */
export class AceOpt0063VaeK1AddSnakeExactFusedKernel {
  private readonly pipelines = new Map<string, Promise<CompiledKernel>>();
  private destroyed = false;

  private constructor(private readonly device: GPUDevice) {}

  static create(
    device: GPUDevice,
    capability: Readonly<{
      subgroupMinSize?: number;
      subgroupMaxSize?: number;
    }>,
  ): AceOpt0063VaeK1AddSnakeExactFusedKernel {
    if (
      !device.features.has("shader-f16") ||
      !device.features.has("subgroups") ||
      capability.subgroupMinSize !== ACE_OPT_0025_VAE_K1_SUBGROUP_SIZE ||
      capability.subgroupMaxSize !== ACE_OPT_0025_VAE_K1_SUBGROUP_SIZE
    ) {
      throw new Error(
        "OPT-0063 fused K1/Add/Snake requires shader-f16 and fixed 32-lane subgroups",
      );
    }
    if (
      device.limits.maxComputeInvocationsPerWorkgroup <
        ACE_OPT_0025_VAE_K1_WORKGROUP_SIZE ||
      device.limits.maxComputeWorkgroupSizeX <
        ACE_OPT_0025_VAE_K1_WORKGROUP_SIZE ||
      device.limits.maxStorageBuffersPerShaderStage <
        ACE_OPT_0063_VAE_K1_ADD_SNAKE_STORAGE_BINDINGS
    ) {
      throw new Error(
        "OPT-0063 fused K1/Add/Snake requires WG128 and eight storage bindings",
      );
    }
    return new AceOpt0063VaeK1AddSnakeExactFusedKernel(device);
  }

  async createDispatch(
    label: string,
    shape: AceVaeConv1dShape,
    bindings: AceOpt0063VaeK1AddSnakeBindings,
  ): Promise<AceOpt0063VaeK1AddSnakeDispatch> {
    this.requireLive();
    const plan = planAceOpt0063VaeK1AddSnake(shape);
    const normalized = normalizeBindings(this.device, plan, bindings, label);
    const compiled = await this.pipelineFor(plan);
    this.requireLive();
    const bindGroup = this.device.createBindGroup({
      label: `${label}-opt-0063-bindings`,
      layout: compiled.bindGroupLayout,
      entries: normalized.map((resource, binding) => ({ binding, resource })),
    });
    const owner = this;
    return Object.freeze({
      label,
      kernelId: ACE_OPT_0063_VAE_K1_ADD_SNAKE_EXACT_FUSED_KERNEL_ID,
      plan,
      encode(pass: GPUComputePassEncoder): void {
        owner.requireLive();
        pass.setPipeline(compiled.pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(plan.workgroupCount, 1, 1);
      },
    });
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.pipelines.clear();
  }

  private pipelineFor(plan: AceOpt0063VaeK1AddSnakePlan): Promise<CompiledKernel> {
    const key = `${plan.k1.rows}x${plan.k1.inner}x${plan.k1.columns}`;
    const existing = this.pipelines.get(key);
    if (existing !== undefined) return existing;
    const created = compileKernel(this.device, plan);
    this.pipelines.set(key, created);
    void created.catch(() => {
      if (this.pipelines.get(key) === created) this.pipelines.delete(key);
    });
    return created;
  }

  private requireLive(): void {
    if (this.destroyed) {
      throw new Error("OPT-0063 fused K1/Add/Snake kernel was destroyed");
    }
  }
}

export function planAceOpt0063VaeK1AddSnake(
  shape: AceVaeConv1dShape,
): AceOpt0063VaeK1AddSnakePlan {
  const k1 = planAceOpt0025VaeK1SubgroupGemm(shape);
  if (k1.workgroupCount > MAX_DISPATCH_DIMENSION) {
    throw new RangeError(
      "OPT-0063 isolated fused screen requires at most 65,535 workgroups",
    );
  }
  const activationStorageBytes = checkedProduct(
    k1.outputElements,
    F16_BYTES,
    "activation storage bytes",
  );
  const parameterStorageBytes = checkedProduct(
    k1.columns,
    F16_BYTES,
    "parameter storage bytes",
  );
  return Object.freeze({
    shape: Object.freeze({ ...shape }),
    k1,
    elements: k1.outputElements,
    activationStorageBytes,
    activationBindingBytes: alignStorage(activationStorageBytes),
    parameterStorageBytes,
    parameterBindingBytes: alignStorage(parameterStorageBytes),
    workgroupCount: k1.workgroupCount,
    storageBindingCount: ACE_OPT_0063_VAE_K1_ADD_SNAKE_STORAGE_BINDINGS,
    formerBoundaries: Object.freeze([
      "producer-f32-to-f16",
      "add-f32-to-f16",
      "snake-f16-to-f32",
      "snake-f32-to-f16",
    ] as const),
  });
}

/** Exact packed-weight coordinate used by the copied OPT-0025 K1 loop. */
export function aceOpt0063VaeK1PackedWeightIndex(
  plan: AceOpt0063VaeK1AddSnakePlan,
  inner: number,
  column: number,
): number {
  requireIndex(inner, plan.k1.inner, "inner");
  requireIndex(column, plan.k1.columns, "column");
  const columnTile = Math.floor(column / ACE_OPT_0025_VAE_K1_TILE_COLUMNS);
  const columnInTile = column % ACE_OPT_0025_VAE_K1_TILE_COLUMNS;
  const innerTile = Math.floor(inner / ACE_OPT_0025_VAE_K1_TILE_INNER);
  const innerInTile = inner % ACE_OPT_0025_VAE_K1_TILE_INNER;
  return (
    ((columnTile * plan.k1.innerTiles + innerTile) *
      ACE_OPT_0025_VAE_K1_TILE_INNER + innerInTile) *
      ACE_OPT_0025_VAE_K1_TILE_COLUMNS + columnInTile
  );
}

/** Pure ownership seam for exhaustive no-overlap/no-hole tests. */
export function planAceOpt0063VaeK1AddSnakeOwner(
  plan: AceOpt0063VaeK1AddSnakePlan,
  workgroup: number,
  localInvocation: number,
): AceOpt0063VaeK1AddSnakeOwner {
  requireIndex(workgroup, plan.workgroupCount, "workgroup");
  requireIndex(
    localInvocation,
    ACE_OPT_0025_VAE_K1_WORKGROUP_SIZE,
    "local invocation",
  );
  const subgroup = Math.floor(
    localInvocation / ACE_OPT_0025_VAE_K1_SUBGROUP_SIZE,
  );
  const subgroupLane =
    localInvocation % ACE_OPT_0025_VAE_K1_SUBGROUP_SIZE;
  const rowTile = Math.floor(workgroup / plan.k1.columnTiles);
  const columnTile = workgroup % plan.k1.columnTiles;
  const rowBase =
    rowTile * ACE_OPT_0025_VAE_K1_TILE_ROWS + subgroup * ROWS_PER_SUBGROUP;
  const columnBase =
    columnTile * ACE_OPT_0025_VAE_K1_TILE_COLUMNS +
    subgroupLane * COLUMNS_PER_LANE;
  const outputIndices: number[] = [];
  for (let rowInOwner = 0; rowInOwner < ROWS_PER_SUBGROUP; rowInOwner += 1) {
    const row = rowBase + rowInOwner;
    if (row >= plan.k1.rows) continue;
    for (let columnInOwner = 0;
      columnInOwner < COLUMNS_PER_LANE;
      columnInOwner += 1) {
      outputIndices.push(
        row * plan.k1.columns + columnBase + columnInOwner,
      );
    }
  }
  return Object.freeze({
    workgroup,
    localInvocation,
    subgroup,
    subgroupLane,
    rowBase,
    columnBase,
    outputIndices: Object.freeze(outputIndices),
  });
}

export function aceOpt0063VaeK1AddSnakeExactFusedWgsl(
  shape: AceVaeConv1dShape,
): string {
  const plan = planAceOpt0063VaeK1AddSnake(shape);
  const accumulatorDeclarations = Array.from(
    { length: ROWS_PER_SUBGROUP },
    (_, row) => `  var acc${row} = load_bias(column);`,
  ).join("\n");
  const contractions = Array.from(
    { length: ROWS_PER_SUBGROUP },
    (_, row) => /* wgsl */ `
      let a${row} = subgroupBroadcast(lane_a, ${row}u);
      acc${row} = acc${row} + vec4<f32>(a${row}) * b;`,
  ).join("\n");
  const stores = Array.from(
    { length: ROWS_PER_SUBGROUP },
    (_, row) => /* wgsl */ `
  {
    let row = row_base + ${row}u;
    if (row < ROWS) {
      let output_base = row * COLUMNS + column;
      store_exact_chain(output_base, column, acc${row}.x);
      store_exact_chain(output_base + 1u, column + 1u, acc${row}.y);
      store_exact_chain(output_base + 2u, column + 2u, acc${row}.z);
      store_exact_chain(output_base + 3u, column + 3u, acc${row}.w);
    }
  }`,
  ).join("\n");

  return /* wgsl */ `
// kernel-id: ${ACE_OPT_0063_VAE_K1_ADD_SNAKE_EXACT_FUSED_KERNEL_ID}
enable f16;
enable subgroups;

const ROWS: u32 = ${plan.k1.rows}u;
const INNER: u32 = ${plan.k1.inner}u;
const COLUMNS: u32 = ${plan.k1.columns}u;
const COLUMN_TILES: u32 = ${plan.k1.columnTiles}u;
const INNER_TILES: u32 = ${plan.k1.innerTiles}u;

@group(0) @binding(0) var<storage, read> input: array<f16>;
@group(0) @binding(1) var<storage, read> packed_weight: array<f16>;
@group(0) @binding(2) var<storage, read> bias: array<f16>;
@group(0) @binding(3) var<storage, read> skip: array<f16>;
@group(0) @binding(4) var<storage, read> alpha: array<f16>;
@group(0) @binding(5) var<storage, read> beta: array<f16>;
@group(0) @binding(6) var<storage, read_write> add_output: array<f16>;
@group(0) @binding(7) var<storage, read_write> snake_output: array<f16>;

fn load_bias(column: u32) -> vec4<f32> {
  return vec4<f32>(
    f32(bias[column]),
    f32(bias[column + 1u]),
    f32(bias[column + 2u]),
    f32(bias[column + 3u]),
  );
}

fn store_exact_chain(index: u32, channel: u32, producer_accumulator: f32) {
  // Former K1 storage boundary: this typed value must not remain FP32.
  let producer_rounded: f16 = f16(producer_accumulator);
  // Preserve production Add operand order and its former FP16 store boundary.
  let left_operand: f32 = f32(skip[index]);
  let right_operand: f32 = f32(producer_rounded);
  let sum: f32 = left_operand + right_operand;
  let add_rounded: f16 = f16(sum);
  add_output[index] = add_rounded;
  // Preserve the complete production Snake FP32 nonlinear island verbatim.
  let value: f32 = f32(add_rounded);
  let alpha_log_scale: f32 = f32(alpha[channel]);
  let beta_log_scale: f32 = f32(beta[channel]);
  let alpha_value: f32 = exp(alpha_log_scale);
  let beta_value: f32 = exp(beta_log_scale);
  let periodic: f32 = sin(alpha_value * value);
  let reciprocal_beta: f32 = 1.0 / (beta_value + 1e-9);
  let result: f32 =
    value + reciprocal_beta * periodic * periodic;
  snake_output[index] = f16(result);
}

@compute @workgroup_size(${ACE_OPT_0025_VAE_K1_WORKGROUP_SIZE}, 1, 1)
fn main(
  @builtin(subgroup_invocation_id) subgroup_lane: u32,
  @builtin(subgroup_id) subgroup: u32,
  @builtin(subgroup_size) subgroup_size: u32,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  if (subgroup_size != ${ACE_OPT_0025_VAE_K1_SUBGROUP_SIZE}u) { return; }
  let row_tile = group.x / COLUMN_TILES;
  let column_tile = group.x % COLUMN_TILES;
  let row_base =
    row_tile * ${ACE_OPT_0025_VAE_K1_TILE_ROWS}u +
    subgroup * ${ROWS_PER_SUBGROUP}u;
  let column =
    column_tile * ${ACE_OPT_0025_VAE_K1_TILE_COLUMNS}u +
    subgroup_lane * ${COLUMNS_PER_LANE}u;
${accumulatorDeclarations}

  // This is the unchanged OPT-0025 increasing-Cin FP32 producer loop.
  for (var inner_tile = 0u; inner_tile < INNER_TILES; inner_tile += 1u) {
    let tile_base =
      (column_tile * INNER_TILES + inner_tile) *
      ${ACE_OPT_0025_VAE_K1_TILE_INNER * ACE_OPT_0025_VAE_K1_TILE_COLUMNS}u;
    for (
      var inner_in_tile = 0u;
      inner_in_tile < ${ACE_OPT_0025_VAE_K1_TILE_INNER}u;
      inner_in_tile += 1u
    ) {
      let inner =
        inner_tile * ${ACE_OPT_0025_VAE_K1_TILE_INNER}u + inner_in_tile;
      var lane_a = 0.0;
      let lane_row = row_base + subgroup_lane;
      if (subgroup_lane < ${ROWS_PER_SUBGROUP}u && lane_row < ROWS) {
        lane_a = f32(input[lane_row * INNER + inner]);
      }
      let weight_base =
        tile_base + inner_in_tile * ${ACE_OPT_0025_VAE_K1_TILE_COLUMNS}u +
        subgroup_lane * ${COLUMNS_PER_LANE}u;
      let b = vec4<f32>(
        f32(packed_weight[weight_base]),
        f32(packed_weight[weight_base + 1u]),
        f32(packed_weight[weight_base + 2u]),
        f32(packed_weight[weight_base + 3u]),
      );
${contractions}
    }
  }
${stores}
}
`;
}

async function compileKernel(
  device: GPUDevice,
  plan: AceOpt0063VaeK1AddSnakePlan,
): Promise<CompiledKernel> {
  const label = `opt-0063-k1-add-snake-${plan.k1.rows}x${plan.k1.inner}`;
  const module = device.createShaderModule({
    label,
    code: aceOpt0063VaeK1AddSnakeExactFusedWgsl(plan.shape),
  });
  const compilation = await module.getCompilationInfo();
  const errors = compilation.messages.filter(({ type }) => type === "error");
  if (errors.length > 0) {
    throw new Error(
      `OPT-0063 fused WGSL failed: ${errors.map(({ lineNum, linePos, message }) =>
        `${lineNum}:${linePos} ${message}`
      ).join("; ")}`,
    );
  }
  const pipeline = await device.createComputePipelineAsync({
    label,
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });
  return Object.freeze({
    pipeline,
    bindGroupLayout: pipeline.getBindGroupLayout(0),
  });
}

function normalizeBindings(
  device: GPUDevice,
  plan: AceOpt0063VaeK1AddSnakePlan,
  bindings: AceOpt0063VaeK1AddSnakeBindings,
  label: string,
): readonly GPUBufferBinding[] {
  const values = Object.freeze([
    normalizeBinding(device, bindings.input, plan.k1.inputBytes, `${label} input`),
    normalizeBinding(
      device,
      bindings.packedWeight,
      plan.k1.weightBytes,
      `${label} packed weight`,
    ),
    normalizeBinding(device, bindings.bias, plan.k1.biasBytes, `${label} bias`),
    normalizeBinding(
      device,
      bindings.skip,
      plan.activationBindingBytes,
      `${label} skip`,
    ),
    normalizeBinding(
      device,
      bindings.alpha,
      plan.parameterBindingBytes,
      `${label} alpha`,
    ),
    normalizeBinding(
      device,
      bindings.beta,
      plan.parameterBindingBytes,
      `${label} beta`,
    ),
    normalizeBinding(
      device,
      bindings.addOutput,
      plan.activationBindingBytes,
      `${label} Add output`,
    ),
    normalizeBinding(
      device,
      bindings.snakeOutput,
      plan.activationBindingBytes,
      `${label} Snake output`,
    ),
  ]);
  for (let left = 0; left < values.length; left += 1) {
    for (let right = left + 1; right < values.length; right += 1) {
      if (bindingsOverlap(values[left]!, values[right]!)) {
        throw new RangeError(`${label} storage bindings ${left}/${right} alias`);
      }
    }
  }
  return values;
}

function normalizeBinding(
  device: GPUDevice,
  binding: GPUBufferBinding,
  requiredBytes: number,
  label: string,
): GPUBufferBinding {
  const offset = Number(binding.offset ?? 0);
  const size = Number(binding.size ?? (binding.buffer.size - offset));
  const alignment = Number(device.limits.minStorageBufferOffsetAlignment);
  const maximumStorage = Number(device.limits.maxStorageBufferBindingSize);
  if (
    !Number.isSafeInteger(offset) || offset < 0 ||
    !Number.isSafeInteger(size) || size < requiredBytes ||
    !Number.isSafeInteger(alignment) || alignment < 1 ||
    offset % alignment !== 0 ||
    offset + size > binding.buffer.size ||
    requiredBytes > maximumStorage
  ) {
    throw new RangeError(`${label} does not satisfy its exact storage binding`);
  }
  return Object.freeze({ buffer: binding.buffer, offset, size });
}

function bindingsOverlap(left: GPUBufferBinding, right: GPUBufferBinding): boolean {
  if (left.buffer !== right.buffer) return false;
  const leftOffset = Number(left.offset ?? 0);
  const leftEnd = leftOffset + Number(left.size ?? (left.buffer.size - leftOffset));
  const rightOffset = Number(right.offset ?? 0);
  const rightEnd = rightOffset + Number(right.size ?? (right.buffer.size - rightOffset));
  return leftOffset < rightEnd && rightOffset < leftEnd;
}

function alignStorage(bytes: number): number {
  return Math.ceil(bytes / STORAGE_ALIGNMENT) * STORAGE_ALIGNMENT;
}

function requireIndex(value: number, upperBound: number, label: string): void {
  if (!Number.isSafeInteger(value) || value < 0 || value >= upperBound) {
    throw new RangeError(`OPT-0063 ${label} is outside its domain`);
  }
}

function checkedProduct(left: number, right: number, label: string): number {
  const product = left * right;
  if (!Number.isSafeInteger(product) || product > 0xffff_ffff) {
    throw new RangeError(`OPT-0063 ${label} exceeds the WGSL u32 domain`);
  }
  return product;
}
