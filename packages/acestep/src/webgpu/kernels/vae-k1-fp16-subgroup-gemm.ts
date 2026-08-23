import type {
  AceVaeConv1dShape,
  AceVaeOutputRangeBinding,
} from "./vae-primitives.js";

export const ACE_OPT_0025_VAE_K1_SUBGROUP_GEMM_KERNEL_ID =
  "opt-0025-vae-k1-fp16-fixed32-subgroup-gemm-v1" as const;
export const ACE_OPT_0025_VAE_K1_SUBGROUP_SIZE = 32;
export const ACE_OPT_0025_VAE_K1_TILE_ROWS = 32;
export const ACE_OPT_0025_VAE_K1_TILE_COLUMNS = 128;
export const ACE_OPT_0025_VAE_K1_TILE_INNER = 32;
export const ACE_OPT_0025_VAE_K1_WORKGROUP_SIZE = 128;

const SUBGROUPS_PER_WORKGROUP =
  ACE_OPT_0025_VAE_K1_WORKGROUP_SIZE /
  ACE_OPT_0025_VAE_K1_SUBGROUP_SIZE;
const ROWS_PER_SUBGROUP =
  ACE_OPT_0025_VAE_K1_TILE_ROWS / SUBGROUPS_PER_WORKGROUP;
const COLUMNS_PER_LANE =
  ACE_OPT_0025_VAE_K1_TILE_COLUMNS /
  ACE_OPT_0025_VAE_K1_SUBGROUP_SIZE;
const MAX_DISPATCH_DIMENSION = 65_535;
const F16_BYTES = 2;

export interface AceOpt0025VaeK1SubgroupGemmPlan {
  readonly shape: AceVaeConv1dShape;
  readonly rows: number;
  readonly inner: number;
  readonly columns: number;
  readonly rowTiles: number;
  readonly columnTiles: number;
  readonly innerTiles: number;
  readonly workgroupCount: number;
  readonly inputElements: number;
  readonly weightElements: number;
  readonly outputElements: number;
  readonly inputBytes: number;
  readonly weightBytes: number;
  readonly biasBytes: number;
  readonly outputBytes: number;
  readonly packedWeightStorageShape: readonly [number, number, 32, 128];
}

export interface AceOpt0025VaeK1SubgroupGemmBindings {
  /** FP16 activation in frame-major NLC order. */
  readonly input: GPUBufferBinding;
  /** Tile-major FP16 `[Cout/128,Cin/32,32,128]`. */
  readonly packedWeight: GPUBufferBinding;
  /** FP16 `[Cout]`. */
  readonly bias: GPUBufferBinding;
  /** FP16 frame-major NLC output. */
  readonly output: GPUBufferBinding;
}

export interface AceOpt0025VaeK1SubgroupGemmDispatch {
  readonly label: string;
  readonly kernelId: typeof ACE_OPT_0025_VAE_K1_SUBGROUP_GEMM_KERNEL_ID;
  readonly plan: AceOpt0025VaeK1SubgroupGemmPlan;
  encode(pass: GPUComputePassEncoder): void;
}

export interface AceOpt0025VaeK1SubgroupGemmRangePlan {
  readonly base: number;
  readonly count: number;
  readonly firstRow: number;
  readonly rowCount: number;
  readonly workgroupCount: number;
}

export interface AceOpt0025VaeK1SubgroupGemmRangeDispatchPlan {
  readonly mapping: "flat-x" | "column-x-row-y";
  readonly workgroupsX: number;
  readonly workgroupsY: number;
  readonly workgroupsZ: 1;
}

interface CompiledKernel {
  readonly pipeline: GPUComputePipeline;
  readonly bindGroupLayout: GPUBindGroupLayout;
}

/** Exact K1 mapping shared by the isolated benchmark and OPT-0028 runtime. */
export class AceOpt0025VaeK1SubgroupGemmKernel {
  private readonly pipelines = new Map<string, Promise<CompiledKernel>>();
  private readonly bindGroups = new Map<string, GPUBindGroup>();
  private readonly bufferIds = new WeakMap<GPUBuffer, number>();
  private nextBufferId = 0;
  private destroyed = false;

  private constructor(private readonly device: GPUDevice) {}

  static create(
    device: GPUDevice,
    capability: Readonly<{
      subgroupMinSize?: number;
      subgroupMaxSize?: number;
    }>,
  ): AceOpt0025VaeK1SubgroupGemmKernel {
    if (
      !device.features.has("shader-f16") ||
      !device.features.has("subgroups") ||
      capability.subgroupMinSize !== ACE_OPT_0025_VAE_K1_SUBGROUP_SIZE ||
      capability.subgroupMaxSize !== ACE_OPT_0025_VAE_K1_SUBGROUP_SIZE
    ) {
      throw new Error(
        "OPT-0025 VAE K1 GEMM requires shader-f16 and fixed 32-lane subgroups",
      );
    }
    if (
      device.limits.maxComputeInvocationsPerWorkgroup <
        ACE_OPT_0025_VAE_K1_WORKGROUP_SIZE ||
      device.limits.maxComputeWorkgroupSizeX <
        ACE_OPT_0025_VAE_K1_WORKGROUP_SIZE
    ) {
      throw new Error("OPT-0025 VAE K1 GEMM requires a 128-lane workgroup");
    }
    return new AceOpt0025VaeK1SubgroupGemmKernel(device);
  }

  async createDispatch(
    label: string,
    shape: AceVaeConv1dShape,
    bindings: AceOpt0025VaeK1SubgroupGemmBindings,
  ): Promise<AceOpt0025VaeK1SubgroupGemmDispatch> {
    this.requireLive();
    const plan = planAceOpt0025VaeK1SubgroupGemm(shape);
    if (plan.workgroupCount > MAX_DISPATCH_DIMENSION) {
      throw new RangeError(
        "OPT-0025 K1 complete operation requires cooperative range dispatches",
      );
    }
    requireBindingBytes(bindings.input, plan.inputBytes, `${label} input`);
    requireBindingBytes(
      bindings.packedWeight,
      plan.weightBytes,
      `${label} packed weight`,
    );
    requireBindingBytes(bindings.bias, plan.biasBytes, `${label} bias`);
    requireBindingBytes(bindings.output, plan.outputBytes, `${label} output`);
    const compiled = await this.pipelineFor(plan);
    this.requireLive();
    const bindGroup = this.device.createBindGroup({
      label: `${label}-opt-0025-bindings`,
      layout: compiled.bindGroupLayout,
      entries: [
        { binding: 0, resource: bindings.input },
        { binding: 1, resource: bindings.packedWeight },
        { binding: 2, resource: bindings.bias },
        { binding: 3, resource: bindings.output },
      ],
    });
    const owner = this;
    return Object.freeze({
      label,
      kernelId: ACE_OPT_0025_VAE_K1_SUBGROUP_GEMM_KERNEL_ID,
      plan,
      encode(pass: GPUComputePassEncoder): void {
        owner.requireLive();
        pass.setPipeline(compiled.pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(plan.workgroupCount, 1, 1);
      },
    });
  }

  /** Production range seam: one exact row-aligned cooperative quantum. */
  async createRangeDispatch(
    label: string,
    shape: AceVaeConv1dShape,
    bindings: AceOpt0025VaeK1SubgroupGemmBindings,
    range: AceVaeOutputRangeBinding,
  ): Promise<AceOpt0025VaeK1SubgroupGemmDispatch> {
    this.requireLive();
    const plan = planAceOpt0025VaeK1SubgroupGemm(shape);
    const outputRange = planAceOpt0025VaeK1SubgroupGemmRange(plan, range);
    const dispatchPlan = planAceOpt0025VaeK1SubgroupGemmRangeDispatch(
      plan,
      outputRange,
    );
    requireBindingBytes(bindings.input, plan.inputBytes, `${label} input`);
    requireBindingBytes(
      bindings.packedWeight,
      plan.weightBytes,
      `${label} packed weight`,
    );
    requireBindingBytes(bindings.bias, plan.biasBytes, `${label} bias`);
    requireBindingBytes(bindings.output, plan.outputBytes, `${label} output`);
    requireBindingBytes(range.control, 16, `${label} range control`);
    const controlOffset = Number(range.control.offset ?? 0);
    const alignment = Number(this.device.limits.minUniformBufferOffsetAlignment);
    if (
      !Number.isSafeInteger(alignment) || alignment < 4 ||
      !Number.isSafeInteger(controlOffset) || controlOffset < 0 ||
      controlOffset % alignment !== 0 || controlOffset > 0xffff_ffff
    ) {
      throw new RangeError(`${label} range control is not dynamically aligned`);
    }
    const compiled = await this.rangePipelineFor(plan);
    this.requireLive();
    const resources = [
      bindings.input,
      bindings.packedWeight,
      bindings.bias,
      bindings.output,
      Object.freeze({ buffer: range.control.buffer, offset: 0, size: 16 }),
    ] as const;
    const bindGroupKey = `range:${plan.rows}x${plan.inner}x${plan.columns}:` +
      resources.map((binding) => this.bindingKey(binding)).join("|");
    let bindGroup = this.bindGroups.get(bindGroupKey);
    if (bindGroup === undefined) {
      bindGroup = this.device.createBindGroup({
        label: `${label}-opt-0025-range-bindings`,
        layout: compiled.bindGroupLayout,
        entries: resources.map((resource, binding) => ({ binding, resource })),
      });
      this.bindGroups.set(bindGroupKey, bindGroup);
    }
    const owner = this;
    return Object.freeze({
      label,
      kernelId: ACE_OPT_0025_VAE_K1_SUBGROUP_GEMM_KERNEL_ID,
      plan,
      encode(pass: GPUComputePassEncoder): void {
        owner.requireLive();
        pass.setPipeline(compiled.pipeline);
        pass.setBindGroup(0, bindGroup, [controlOffset]);
        pass.dispatchWorkgroups(
          dispatchPlan.workgroupsX,
          dispatchPlan.workgroupsY,
          dispatchPlan.workgroupsZ,
        );
      },
    });
  }

  destroy(): void {
    this.destroyed = true;
    this.bindGroups.clear();
    this.pipelines.clear();
  }

  private pipelineFor(
    plan: AceOpt0025VaeK1SubgroupGemmPlan,
  ): Promise<CompiledKernel> {
    const key = `${plan.rows}x${plan.inner}x${plan.columns}`;
    const existing = this.pipelines.get(key);
    if (existing !== undefined) return existing;
    const created = compileKernel(this.device, plan);
    this.pipelines.set(key, created);
    void created.catch(() => {
      if (this.pipelines.get(key) === created) this.pipelines.delete(key);
    });
    return created;
  }

  private rangePipelineFor(
    plan: AceOpt0025VaeK1SubgroupGemmPlan,
  ): Promise<CompiledKernel> {
    const key = `range:${plan.rows}x${plan.inner}x${plan.columns}`;
    const existing = this.pipelines.get(key);
    if (existing !== undefined) return existing;
    const created = compileRangeKernel(this.device, plan);
    this.pipelines.set(key, created);
    void created.catch(() => {
      if (this.pipelines.get(key) === created) this.pipelines.delete(key);
    });
    return created;
  }

  private bindingKey(binding: GPUBufferBinding): string {
    let id = this.bufferIds.get(binding.buffer);
    if (id === undefined) {
      id = this.nextBufferId++;
      this.bufferIds.set(binding.buffer, id);
    }
    return `${id}:${binding.offset ?? 0}:${binding.size ?? -1}`;
  }

  private requireLive(): void {
    if (this.destroyed) {
      throw new Error("OPT-0025 VAE K1 GEMM kernel was destroyed");
    }
  }
}

export function planAceOpt0025VaeK1SubgroupGemm(
  shape: AceVaeConv1dShape,
): AceOpt0025VaeK1SubgroupGemmPlan {
  for (const [name, value] of Object.entries(shape)) {
    const invalidBoundary = name === "padding" ? value < 0 : value <= 0;
    if (!Number.isSafeInteger(value) || invalidBoundary) {
      throw new RangeError(`OPT-0025 VAE K1 ${name} is invalid`);
    }
  }
  if (
    shape.batch !== 1 ||
    shape.kernelSize !== 1 ||
    shape.stride !== 1 ||
    shape.dilation !== 1 ||
    shape.padding !== 0 ||
    shape.inputChannels !== shape.outputChannels
  ) {
    throw new RangeError(
      "OPT-0025 requires a batch-one, square, unpadded K1 Conv1D",
    );
  }
  if (shape.inputChannels % ACE_OPT_0025_VAE_K1_TILE_INNER !== 0) {
    throw new RangeError("OPT-0025 K1 Cin must be divisible by 32");
  }
  if (shape.outputChannels % ACE_OPT_0025_VAE_K1_TILE_COLUMNS !== 0) {
    throw new RangeError("OPT-0025 K1 Cout must be divisible by 128");
  }
  const rows = shape.inputFrames;
  const inner = shape.inputChannels;
  const columns = shape.outputChannels;
  const rowTiles = Math.ceil(rows / ACE_OPT_0025_VAE_K1_TILE_ROWS);
  const columnTiles = columns / ACE_OPT_0025_VAE_K1_TILE_COLUMNS;
  const innerTiles = inner / ACE_OPT_0025_VAE_K1_TILE_INNER;
  const workgroupCount = checkedProduct(rowTiles, columnTiles, "workgroups");
  const inputElements = checkedProduct(rows, inner, "input elements");
  const weightElements = checkedProduct(inner, columns, "weight elements");
  const outputElements = checkedProduct(rows, columns, "output elements");
  return Object.freeze({
    shape: Object.freeze({ ...shape }),
    rows,
    inner,
    columns,
    rowTiles,
    columnTiles,
    innerTiles,
    workgroupCount,
    inputElements,
    weightElements,
    outputElements,
    inputBytes: checkedProduct(inputElements, F16_BYTES, "input bytes"),
    weightBytes: checkedProduct(weightElements, F16_BYTES, "weight bytes"),
    biasBytes: checkedProduct(columns, F16_BYTES, "bias bytes"),
    outputBytes: checkedProduct(outputElements, F16_BYTES, "output bytes"),
    packedWeightStorageShape: Object.freeze([
      columnTiles,
      innerTiles,
      ACE_OPT_0025_VAE_K1_TILE_INNER,
      ACE_OPT_0025_VAE_K1_TILE_COLUMNS,
    ]) as readonly [number, number, 32, 128],
  });
}

export function planAceOpt0025VaeK1SubgroupGemmRange(
  plan: AceOpt0025VaeK1SubgroupGemmPlan,
  range: Readonly<{ base: number; count: number }>,
): AceOpt0025VaeK1SubgroupGemmRangePlan {
  if (
    !Number.isSafeInteger(range.base) || range.base < 0 ||
    !Number.isSafeInteger(range.count) || range.count <= 0 ||
    range.base + range.count > plan.outputElements ||
    range.base % plan.columns !== 0 || range.count % plan.columns !== 0
  ) {
    throw new RangeError("OPT-0025 K1 range must contain complete in-bounds rows");
  }
  const firstRow = range.base / plan.columns;
  const rowCount = range.count / plan.columns;
  const workgroupCount = checkedProduct(
    Math.ceil(rowCount / ACE_OPT_0025_VAE_K1_TILE_ROWS),
    plan.columnTiles,
    "range workgroups",
  );
  const result = Object.freeze({
    base: range.base,
    count: range.count,
    firstRow,
    rowCount,
    workgroupCount,
  });
  planAceOpt0025VaeK1SubgroupGemmRangeDispatch(plan, result);
  return result;
}

/**
 * Keep the accepted C512 flattened dispatch byte-for-byte, but use WebGPU's
 * independent X/Y limits for larger shapes. The range control supplies the
 * row origin, so every cooperative quantum remains an exact, disjoint slice.
 */
export function planAceOpt0025VaeK1SubgroupGemmRangeDispatch(
  plan: AceOpt0025VaeK1SubgroupGemmPlan,
  range: AceOpt0025VaeK1SubgroupGemmRangePlan,
): AceOpt0025VaeK1SubgroupGemmRangeDispatchPlan {
  if (plan.workgroupCount <= MAX_DISPATCH_DIMENSION) {
    if (range.workgroupCount > MAX_DISPATCH_DIMENSION) {
      throw new RangeError("OPT-0025 K1 range exceeds one dispatch dimension");
    }
    return Object.freeze({
      mapping: "flat-x",
      workgroupsX: range.workgroupCount,
      workgroupsY: 1,
      workgroupsZ: 1,
    });
  }
  if (
    range.workgroupCount % plan.columnTiles !== 0 ||
    plan.columnTiles > MAX_DISPATCH_DIMENSION
  ) {
    throw new RangeError("OPT-0025 K1 range has invalid 2D dispatch geometry");
  }
  const rowTiles = range.workgroupCount / plan.columnTiles;
  if (rowTiles > MAX_DISPATCH_DIMENSION) {
    throw new RangeError("OPT-0025 K1 range exceeds the 2D dispatch limits");
  }
  return Object.freeze({
    mapping: "column-x-row-y",
    workgroupsX: plan.columnTiles,
    workgroupsY: rowTiles,
    workgroupsZ: 1,
  });
}

/** Packs converter-native K1 `[Cout,1,Cin]` FP16 bits into direct GEMM tiles. */
export function packAceOpt0025VaeK1WeightU16(
  nativeWeight: Uint16Array,
  inputChannels: number,
  outputChannels: number,
): Uint16Array {
  const shape = {
    batch: 1,
    inputFrames: 1,
    inputChannels,
    outputChannels,
    kernelSize: 1,
    stride: 1,
    dilation: 1,
    padding: 0,
  } satisfies AceVaeConv1dShape;
  const plan = planAceOpt0025VaeK1SubgroupGemm(shape);
  if (nativeWeight.length !== plan.weightElements) {
    throw new RangeError("OPT-0025 native K1 weight length changed");
  }
  const packed = new Uint16Array(nativeWeight.length);
  for (let columnTile = 0; columnTile < plan.columnTiles; columnTile += 1) {
    for (let innerTile = 0; innerTile < plan.innerTiles; innerTile += 1) {
      for (
        let innerInTile = 0;
        innerInTile < ACE_OPT_0025_VAE_K1_TILE_INNER;
        innerInTile += 1
      ) {
        const inner =
          innerTile * ACE_OPT_0025_VAE_K1_TILE_INNER + innerInTile;
        const packedBase = (
          (columnTile * plan.innerTiles + innerTile) *
            ACE_OPT_0025_VAE_K1_TILE_INNER +
          innerInTile
        ) * ACE_OPT_0025_VAE_K1_TILE_COLUMNS;
        for (
          let columnInTile = 0;
          columnInTile < ACE_OPT_0025_VAE_K1_TILE_COLUMNS;
          columnInTile += 1
        ) {
          const column =
            columnTile * ACE_OPT_0025_VAE_K1_TILE_COLUMNS + columnInTile;
          packed[packedBase + columnInTile] =
            nativeWeight[column * inputChannels + inner]!;
        }
      }
    }
  }
  return packed;
}

export function aceOpt0025VaeK1SubgroupGemmWgsl(
  shape: AceVaeConv1dShape,
): string {
  const plan = planAceOpt0025VaeK1SubgroupGemm(shape);
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
      output[output_base] = f16(acc${row}.x);
      output[output_base + 1u] = f16(acc${row}.y);
      output[output_base + 2u] = f16(acc${row}.z);
      output[output_base + 3u] = f16(acc${row}.w);
    }
  }`,
  ).join("\n");

  return /* wgsl */ `
enable f16;
enable subgroups;

const ROWS: u32 = ${plan.rows}u;
const INNER: u32 = ${plan.inner}u;
const COLUMNS: u32 = ${plan.columns}u;
const COLUMN_TILES: u32 = ${plan.columnTiles}u;
const INNER_TILES: u32 = ${plan.innerTiles}u;

@group(0) @binding(0) var<storage, read> input: array<f16>;
@group(0) @binding(1) var<storage, read> packed_weight: array<f16>;
@group(0) @binding(2) var<storage, read> bias: array<f16>;
@group(0) @binding(3) var<storage, read_write> output: array<f16>;

fn load_bias(column: u32) -> vec4<f32> {
  return vec4<f32>(
    f32(bias[column]),
    f32(bias[column + 1u]),
    f32(bias[column + 2u]),
    f32(bias[column + 3u]),
  );
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

  // The nested tile loops visit logical Cin exactly as 0, 1, ..., INNER - 1.
  // Bias is already in each FP32 accumulator, matching production reduction order.
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

/** Same arithmetic as the measured kernel, with a dynamic complete-row range. */
export function aceOpt0025VaeK1SubgroupGemmRangeWgsl(
  shape: AceVaeConv1dShape,
): string {
  const plan = planAceOpt0025VaeK1SubgroupGemm(shape);
  const flatSource = aceOpt0025VaeK1SubgroupGemmWgsl(shape);
  const source = plan.workgroupCount <= MAX_DISPATCH_DIMENSION
    ? flatSource
    : flatSource.replace(
        `let row_tile = group.x / COLUMN_TILES;
  let column_tile = group.x % COLUMN_TILES;`,
        `let row_tile = group.y;
  let column_tile = group.x;`,
      );
  return source
    .replace(
      "@group(0) @binding(3) var<storage, read_write> output: array<f16>;",
      `@group(0) @binding(3) var<storage, read_write> output: array<f16>;

struct OutputRangeParameters {
  first_output: u32,
  output_count: u32,
  _padding0: u32,
  _padding1: u32,
}
@group(0) @binding(4) var<uniform> output_range: OutputRangeParameters;`,
    )
    .replace(
      "  let row_base =",
      `  let first_row = output_range.first_output / COLUMNS;
  let range_row_count = output_range.output_count / COLUMNS;
  let row_base =`,
    )
    .replace(
      `row_tile * ${ACE_OPT_0025_VAE_K1_TILE_ROWS}u +`,
      `first_row + row_tile * ${ACE_OPT_0025_VAE_K1_TILE_ROWS}u +`,
    )
    .replaceAll("row < ROWS", "row < first_row + range_row_count")
    .replaceAll("lane_row < ROWS", "lane_row < first_row + range_row_count");
}

async function compileKernel(
  device: GPUDevice,
  plan: AceOpt0025VaeK1SubgroupGemmPlan,
): Promise<CompiledKernel> {
  const label = `opt-0025-vae-k1-${plan.rows}x${plan.inner}`;
  const module = device.createShaderModule({
    label,
    code: aceOpt0025VaeK1SubgroupGemmWgsl(plan.shape),
  });
  const compilation = await module.getCompilationInfo();
  const errors = compilation.messages.filter(({ type }) => type === "error");
  if (errors.length > 0) {
    throw new Error(
      `OPT-0025 VAE K1 WGSL failed: ${errors.map(({ lineNum, linePos, message }) =>
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

async function compileRangeKernel(
  device: GPUDevice,
  plan: AceOpt0025VaeK1SubgroupGemmPlan,
): Promise<CompiledKernel> {
  const label = `opt-0025-vae-k1-range-${plan.rows}x${plan.inner}`;
  const module = device.createShaderModule({
    label,
    code: aceOpt0025VaeK1SubgroupGemmRangeWgsl(plan.shape),
  });
  const compilation = await module.getCompilationInfo();
  const errors = compilation.messages.filter(({ type }) => type === "error");
  if (errors.length > 0) {
    throw new Error(
      `OPT-0025 VAE K1 range WGSL failed: ${errors.map(({ lineNum, linePos, message }) =>
        `${lineNum}:${linePos} ${message}`
      ).join("; ")}`,
    );
  }
  const bindGroupLayout = device.createBindGroupLayout({
    label: `${label}-bindings`,
    entries: [
      ...[plan.inputBytes, plan.weightBytes, plan.biasBytes, plan.outputBytes]
        .map((minBindingSize, binding) => ({
          binding,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: binding === 3 ? "storage" as const : "read-only-storage" as const,
            minBindingSize,
          },
        })),
      {
        binding: 4,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: "uniform" as const,
          hasDynamicOffset: true,
          minBindingSize: 16,
        },
      },
    ],
  });
  const pipeline = await device.createComputePipelineAsync({
    label,
    layout: device.createPipelineLayout({
      label: `${label}-layout`,
      bindGroupLayouts: [bindGroupLayout],
    }),
    compute: { module, entryPoint: "main" },
  });
  return Object.freeze({ pipeline, bindGroupLayout });
}

function requireBindingBytes(
  binding: GPUBufferBinding,
  requiredBytes: number,
  label: string,
): void {
  const offset = Number(binding.offset ?? 0);
  const size = Number(binding.size ?? (binding.buffer.size - offset));
  if (
    !Number.isSafeInteger(offset) || offset < 0 ||
    !Number.isSafeInteger(size) || size < requiredBytes ||
    offset + requiredBytes > binding.buffer.size
  ) {
    throw new RangeError(`${label} is smaller than ${requiredBytes} bytes`);
  }
}

function checkedProduct(left: number, right: number, label: string): number {
  const product = left * right;
  if (!Number.isSafeInteger(product) || product > 0xffff_ffff) {
    throw new RangeError(`OPT-0025 ${label} exceeds the WGSL u32 domain`);
  }
  return product;
}
