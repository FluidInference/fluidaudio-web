import {
  ACE_GEMM_MAX_MULTIPLY_ADDS_PER_RANGE,
  ACE_GEMM_MAX_OUTPUTS_PER_RANGE,
  type AceGemmBufferBindings,
  type AceGemmOutputRange,
  type AceGemmShape,
} from "../src/webgpu/kernels/gemm.js";
import {
  requireAceBindingBytes,
  requireAceDisjointOutput,
} from "../src/webgpu/kernels/correctness-utils.js";
import { createAceScopedBuffers } from
  "../src/webgpu/scoped-buffer-allocation.js";

export const ACE_OPT_0003_SUBGROUP_SIZE = 32;
export const ACE_OPT_0003_SUBGROUP_GEMM_TILE_ROWS = 32;
export const ACE_OPT_0003_SUBGROUP_GEMM_TILE_COLUMNS = 128;
export const ACE_OPT_0003_SUBGROUP_GEMM_TILE_INNER = 32;
export const ACE_OPT_0003_SUBGROUP_GEMM_WORKGROUP_SIZE = 128;
/** Direct-B has no workgroup-memory staging. */
export const ACE_OPT_0003_SUBGROUP_GEMM_WORKGROUP_BYTES = 0;

const SUBGROUPS_PER_WORKGROUP =
  ACE_OPT_0003_SUBGROUP_GEMM_WORKGROUP_SIZE /
  ACE_OPT_0003_SUBGROUP_SIZE;
const ROWS_PER_SUBGROUP =
  ACE_OPT_0003_SUBGROUP_GEMM_TILE_ROWS /
  SUBGROUPS_PER_WORKGROUP;
const COLUMNS_PER_LANE =
  ACE_OPT_0003_SUBGROUP_GEMM_TILE_COLUMNS /
  ACE_OPT_0003_SUBGROUP_SIZE;
const BF16_WORDS_PER_TILE_INNER_ROW =
  ACE_OPT_0003_SUBGROUP_GEMM_TILE_COLUMNS / 2;
const BF16_WORDS_PER_TILE =
  ACE_OPT_0003_SUBGROUP_GEMM_TILE_INNER *
  BF16_WORDS_PER_TILE_INNER_ROW;
const MAX_DISPATCH_DIMENSION = 65_535;
const OUTPUT_RANGE_PARAMETER_BYTES = 16;
const MINIMUM_UNIFORM_STRIDE = 256;
const MAX_WGSL_U32 = 0xffff_ffff;

export interface AceOpt0003FixedSubgroupCapability {
  readonly subgroupMinSize?: number;
  readonly subgroupMaxSize?: number;
}

export interface AceOpt0003SubgroupGemmPlan extends AceGemmShape {
  readonly tileRows: typeof ACE_OPT_0003_SUBGROUP_GEMM_TILE_ROWS;
  readonly tileColumns: typeof ACE_OPT_0003_SUBGROUP_GEMM_TILE_COLUMNS;
  readonly tileInner: typeof ACE_OPT_0003_SUBGROUP_GEMM_TILE_INNER;
  readonly workgroupSize: typeof ACE_OPT_0003_SUBGROUP_GEMM_WORKGROUP_SIZE;
  readonly subgroupSize: typeof ACE_OPT_0003_SUBGROUP_SIZE;
  readonly rowTiles: number;
  readonly columnTiles: number;
  readonly innerTiles: number;
  readonly workgroupCount: number;
  readonly activationElements: number;
  readonly weightElements: number;
  readonly packedWeightWords: number;
  readonly outputElements: number;
  readonly outputRangeCount: number;
  readonly outputRanges: readonly AceGemmOutputRange[];
  /** Physical scalar shape; the Uint32 storage packs its final dimension by two. */
  readonly packedWeightStorageShape: readonly [number, number, 32, 128];
}

export interface AceOpt0003SubgroupGemmBindings
  extends AceGemmBufferBindings {
  /** Direct tile-major packed BF16 `[N/128,K/32,32,128]`. */
  readonly weight: GPUBufferBinding;
  /** Optional logical `[N]` packed BF16 bias. */
  readonly bias?: GPUBufferBinding;
}

export interface AceOpt0003SubgroupGemmDispatch {
  readonly label: string;
  readonly plan: AceOpt0003SubgroupGemmPlan;
  readonly rangeCount: number;
  encodeRange(pass: GPUComputePassEncoder, rangeIndex: number): void;
  encode(pass: GPUComputePassEncoder): void;
}

interface CompiledAceOpt0003SubgroupGemm {
  readonly pipeline: GPUComputePipeline;
  readonly bindGroupLayout: GPUBindGroupLayout;
  readonly outputRangeParameters: GPUBuffer;
  readonly outputRangeParameterStride: number;
  destroy(): void;
}

/**
 * Benchmark-only fixed-32 subgroup candidate for the OPT-0003 A/B gate.
 *
 * It deliberately accepts only the converter-candidate geometry. A fixed-size
 * capability report is mandatory because WebGPU's `subgroups` feature alone
 * does not prove that `subgroupBroadcast` source lanes 0 through 7 exist.
 */
export class AceOpt0003SubgroupGemmKernel {
  private readonly compiled = new Map<
    string,
    Promise<CompiledAceOpt0003SubgroupGemm>
  >();
  private destroyed = false;

  private constructor(private readonly device: GPUDevice) {}

  static create(
    device: GPUDevice,
    capability: AceOpt0003FixedSubgroupCapability,
  ): AceOpt0003SubgroupGemmKernel {
    const subgroupMinSize = capability.subgroupMinSize;
    const subgroupMaxSize = capability.subgroupMaxSize;
    if (
      !device.features.has("subgroups") ||
      subgroupMinSize !== ACE_OPT_0003_SUBGROUP_SIZE ||
      subgroupMaxSize !== ACE_OPT_0003_SUBGROUP_SIZE
    ) {
      throw new Error(
        "OPT-0003 subgroup GEMM requires reported fixed 32-lane subgroups",
      );
    }
    if (
      device.limits.maxComputeInvocationsPerWorkgroup <
        ACE_OPT_0003_SUBGROUP_GEMM_WORKGROUP_SIZE ||
      device.limits.maxComputeWorkgroupSizeX <
        ACE_OPT_0003_SUBGROUP_GEMM_WORKGROUP_SIZE
    ) {
      throw new Error(
        `OPT-0003 subgroup GEMM requires WG${ACE_OPT_0003_SUBGROUP_GEMM_WORKGROUP_SIZE}`,
      );
    }
    return new AceOpt0003SubgroupGemmKernel(device);
  }

  async createDispatch(
    label: string,
    shape: AceGemmShape,
    bindings: AceOpt0003SubgroupGemmBindings,
  ): Promise<AceOpt0003SubgroupGemmDispatch> {
    if (this.destroyed) {
      throw new Error("OPT-0003 subgroup GEMM kernel was destroyed");
    }
    const plan = planAceOpt0003SubgroupGemm(shape);
    const activationBytes = checkedBytes(
      plan.activationElements,
      Float32Array.BYTES_PER_ELEMENT,
      "activation",
    );
    const weightBytes = checkedBytes(
      plan.packedWeightWords,
      Uint32Array.BYTES_PER_ELEMENT,
      "packed weight",
    );
    const outputBytes = checkedBytes(
      plan.outputElements,
      Float32Array.BYTES_PER_ELEMENT,
      "output",
    );
    const biasBytes = checkedBytes(
      plan.columns / 2,
      Uint32Array.BYTES_PER_ELEMENT,
      "bias",
    );
    requireAceBindingBytes(bindings.activation, activationBytes, `${label} activation`);
    requireAceBindingBytes(bindings.weight, weightBytes, `${label} weight`);
    requireAceBindingBytes(bindings.output, outputBytes, `${label} output`);
    if (bindings.bias !== undefined) {
      requireAceBindingBytes(bindings.bias, biasBytes, `${label} bias`);
    }
    requireAceDisjointOutput(
      exactBinding(bindings.output, outputBytes),
      [
        exactBinding(bindings.activation, activationBytes),
        exactBinding(bindings.weight, weightBytes),
        ...(bindings.bias === undefined
          ? []
          : [exactBinding(bindings.bias, biasBytes)]),
      ],
      label,
    );

    const compiled = await this.pipelineFor(plan, bindings.bias !== undefined);
    if (this.destroyed) {
      throw new Error(
        "OPT-0003 subgroup GEMM kernel was destroyed while compiling",
      );
    }
    const rangeBinding = bindings.bias === undefined ? 3 : 4;
    const bindGroups = plan.outputRanges.map((_, rangeIndex) =>
      this.device.createBindGroup({
        label: `${label}-opt-0003-range-${rangeIndex}-bindings`,
        layout: compiled.bindGroupLayout,
        entries: [
          { binding: 0, resource: bindings.activation },
          { binding: 1, resource: bindings.weight },
          { binding: 2, resource: bindings.output },
          ...(bindings.bias === undefined
            ? []
            : [{ binding: 3, resource: bindings.bias }]),
          {
            binding: rangeBinding,
            resource: {
              buffer: compiled.outputRangeParameters,
              offset: rangeIndex * compiled.outputRangeParameterStride,
              size: OUTPUT_RANGE_PARAMETER_BYTES,
            },
          },
        ],
      }),
    );

    return Object.freeze({
      label,
      plan,
      rangeCount: plan.outputRangeCount,
      encodeRange(pass: GPUComputePassEncoder, rangeIndex: number): void {
        const range = plan.outputRanges[rangeIndex];
        if (range === undefined) {
          throw new RangeError(
            `${label} OPT-0003 range ${rangeIndex} is outside [0, ${plan.outputRangeCount})`,
          );
        }
        pass.setPipeline(compiled.pipeline);
        pass.setBindGroup(0, bindGroups[rangeIndex]!);
        pass.dispatchWorkgroups(range.workgroupCount, 1, 1);
      },
      encode(pass: GPUComputePassEncoder): void {
        pass.setPipeline(compiled.pipeline);
        for (let index = 0; index < plan.outputRanges.length; index += 1) {
          const range = plan.outputRanges[index]!;
          pass.setBindGroup(0, bindGroups[index]!);
          pass.dispatchWorkgroups(range.workgroupCount, 1, 1);
        }
      },
    });
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    for (const compiled of this.compiled.values()) {
      void compiled.then(
        (resources) => resources.destroy(),
        () => undefined,
      );
    }
    this.compiled.clear();
  }

  private pipelineFor(
    shape: AceGemmShape,
    hasBias: boolean,
  ): Promise<CompiledAceOpt0003SubgroupGemm> {
    const key =
      `${shape.rows}x${shape.inner}x${shape.columns}:` +
      (hasBias ? "bias" : "no-bias");
    const existing = this.compiled.get(key);
    if (existing !== undefined) return existing;
    const created = compileAceOpt0003SubgroupGemm(
      this.device,
      shape,
      hasBias,
    );
    this.compiled.set(key, created);
    void created.catch(() => {
      if (this.compiled.get(key) === created) this.compiled.delete(key);
    });
    return created;
  }
}

export function planAceOpt0003SubgroupGemm(
  shape: AceGemmShape,
): AceOpt0003SubgroupGemmPlan {
  const { rows, inner, columns } = shape;
  requirePositiveSafeInteger(rows, "rows");
  requirePositiveSafeInteger(inner, "inner");
  requirePositiveSafeInteger(columns, "columns");
  if (inner % ACE_OPT_0003_SUBGROUP_GEMM_TILE_INNER !== 0) {
    throw new RangeError("OPT-0003 subgroup GEMM requires K divisible by 32");
  }
  if (columns % ACE_OPT_0003_SUBGROUP_GEMM_TILE_COLUMNS !== 0) {
    throw new RangeError("OPT-0003 subgroup GEMM requires N divisible by 128");
  }

  const activationElements = checkedProduct(rows, inner, "activation");
  const weightElements = checkedProduct(columns, inner, "weight");
  const outputElements = checkedProduct(rows, columns, "output");
  requireWgslIndexable(activationElements, "activation");
  requireWgslIndexable(weightElements, "weight");
  requireWgslIndexable(outputElements, "output");
  const packedWeightWords = weightElements / 2;
  const rowTiles = Math.ceil(rows / ACE_OPT_0003_SUBGROUP_GEMM_TILE_ROWS);
  const columnTiles = columns / ACE_OPT_0003_SUBGROUP_GEMM_TILE_COLUMNS;
  const innerTiles = inner / ACE_OPT_0003_SUBGROUP_GEMM_TILE_INNER;
  const workgroupCount = checkedProduct(rowTiles, columnTiles, "workgroups");
  requireWgslIndexable(workgroupCount, "workgroups");
  const outputsPerWorkgroup =
    ACE_OPT_0003_SUBGROUP_GEMM_TILE_ROWS *
    ACE_OPT_0003_SUBGROUP_GEMM_TILE_COLUMNS;
  const multiplyAddsPerWorkgroup = checkedProduct(
    outputsPerWorkgroup,
    inner,
    "workgroup multiply-adds",
  );
  if (multiplyAddsPerWorkgroup > ACE_GEMM_MAX_MULTIPLY_ADDS_PER_RANGE) {
    throw new RangeError(
      "OPT-0003 K exceeds one bounded subgroup output tile",
    );
  }
  const workgroupsPerRange = Math.min(
    MAX_DISPATCH_DIMENSION,
    Math.floor(ACE_GEMM_MAX_OUTPUTS_PER_RANGE / outputsPerWorkgroup),
    Math.floor(
      ACE_GEMM_MAX_MULTIPLY_ADDS_PER_RANGE /
        multiplyAddsPerWorkgroup,
    ),
  );
  if (workgroupsPerRange <= 0) {
    throw new RangeError("OPT-0003 subgroup GEMM cannot form a bounded range");
  }

  const outputRanges: AceGemmOutputRange[] = [];
  let emittedOutputs = 0;
  for (
    let firstWorkgroup = 0;
    firstWorkgroup < workgroupCount;
    firstWorkgroup += workgroupsPerRange
  ) {
    const rangeWorkgroups = Math.min(
      workgroupsPerRange,
      workgroupCount - firstWorkgroup,
    );
    const outputCount = rangeOutputCount(
      rows,
      columnTiles,
      firstWorkgroup,
      rangeWorkgroups,
    );
    outputRanges.push(Object.freeze({
      firstOutput: emittedOutputs,
      outputCount,
      firstWorkgroup,
      workgroupCount: rangeWorkgroups,
      // Tail rows execute zero-padded contraction slots, so budget the tile.
      multiplyAdds: checkedProduct(
        rangeWorkgroups,
        multiplyAddsPerWorkgroup,
        "range multiply-adds",
      ),
    }));
    emittedOutputs += outputCount;
  }
  if (emittedOutputs !== outputElements) {
    throw new Error("OPT-0003 subgroup GEMM range planner lost outputs");
  }

  return Object.freeze({
    rows,
    inner,
    columns,
    tileRows: ACE_OPT_0003_SUBGROUP_GEMM_TILE_ROWS,
    tileColumns: ACE_OPT_0003_SUBGROUP_GEMM_TILE_COLUMNS,
    tileInner: ACE_OPT_0003_SUBGROUP_GEMM_TILE_INNER,
    workgroupSize: ACE_OPT_0003_SUBGROUP_GEMM_WORKGROUP_SIZE,
    subgroupSize: ACE_OPT_0003_SUBGROUP_SIZE,
    rowTiles,
    columnTiles,
    innerTiles,
    workgroupCount,
    activationElements,
    weightElements,
    packedWeightWords,
    outputElements,
    outputRangeCount: outputRanges.length,
    outputRanges: Object.freeze(outputRanges),
    packedWeightStorageShape: Object.freeze([
      columnTiles,
      innerTiles,
      ACE_OPT_0003_SUBGROUP_GEMM_TILE_INNER,
      ACE_OPT_0003_SUBGROUP_GEMM_TILE_COLUMNS,
    ]) as readonly [number, number, 32, 128],
  });
}

/**
 * Losslessly transpose logical packed-BF16 `[N,K]` bits into direct physical
 * `[N/128,K/32,32,128]` scalar order. The returned u32 pairs pack adjacent N.
 */
export function packAceOpt0003SubgroupBf16Weight(
  rowMajorPackedU32: Uint32Array,
  shape: AceGemmShape,
): Uint32Array {
  const plan = planAceOpt0003SubgroupGemm(shape);
  requireExactPackedWords(
    rowMajorPackedU32,
    plan.packedWeightWords,
    "logical [N,K]",
  );
  const packed = new Uint32Array(plan.packedWeightWords);
  for (let columnTile = 0; columnTile < plan.columnTiles; columnTile += 1) {
    for (let innerTile = 0; innerTile < plan.innerTiles; innerTile += 1) {
      const tileWordBase =
        (columnTile * plan.innerTiles + innerTile) *
        BF16_WORDS_PER_TILE;
      for (
        let innerInTile = 0;
        innerInTile < ACE_OPT_0003_SUBGROUP_GEMM_TILE_INNER;
        innerInTile += 1
      ) {
        const logicalInner =
          innerTile * ACE_OPT_0003_SUBGROUP_GEMM_TILE_INNER + innerInTile;
        const physicalRowWordBase =
          tileWordBase + innerInTile * BF16_WORDS_PER_TILE_INNER_ROW;
        for (
          let columnPair = 0;
          columnPair < BF16_WORDS_PER_TILE_INNER_ROW;
          columnPair += 1
        ) {
          const evenColumn =
            columnTile * ACE_OPT_0003_SUBGROUP_GEMM_TILE_COLUMNS +
            columnPair * 2;
          const low = packedBf16ScalarBits(
            rowMajorPackedU32,
            evenColumn * plan.inner + logicalInner,
          );
          const high = packedBf16ScalarBits(
            rowMajorPackedU32,
            (evenColumn + 1) * plan.inner + logicalInner,
          );
          packed[physicalRowWordBase + columnPair] =
            (low | (high << 16)) >>> 0;
        }
      }
    }
  }
  return packed;
}

/** Inverse diagnostic helper used to prove that prepacking changes no BF16 bit. */
export function unpackAceOpt0003SubgroupBf16Weight(
  tileMajorPackedU32: Uint32Array,
  shape: AceGemmShape,
): Uint32Array {
  const plan = planAceOpt0003SubgroupGemm(shape);
  requireExactPackedWords(
    tileMajorPackedU32,
    plan.packedWeightWords,
    "physical [N/128,K/32,32,128]",
  );
  const logical = new Uint32Array(plan.packedWeightWords);
  for (let columnTile = 0; columnTile < plan.columnTiles; columnTile += 1) {
    for (let innerTile = 0; innerTile < plan.innerTiles; innerTile += 1) {
      const tileWordBase =
        (columnTile * plan.innerTiles + innerTile) *
        BF16_WORDS_PER_TILE;
      for (
        let innerInTile = 0;
        innerInTile < ACE_OPT_0003_SUBGROUP_GEMM_TILE_INNER;
        innerInTile += 1
      ) {
        const logicalInner =
          innerTile * ACE_OPT_0003_SUBGROUP_GEMM_TILE_INNER + innerInTile;
        const physicalRowWordBase =
          tileWordBase + innerInTile * BF16_WORDS_PER_TILE_INNER_ROW;
        for (
          let columnPair = 0;
          columnPair < BF16_WORDS_PER_TILE_INNER_ROW;
          columnPair += 1
        ) {
          const pair = tileMajorPackedU32[physicalRowWordBase + columnPair]!;
          const evenColumn =
            columnTile * ACE_OPT_0003_SUBGROUP_GEMM_TILE_COLUMNS +
            columnPair * 2;
          setPackedBf16ScalarBits(
            logical,
            evenColumn * plan.inner + logicalInner,
            pair & 0xffff,
          );
          setPackedBf16ScalarBits(
            logical,
            (evenColumn + 1) * plan.inner + logicalInner,
            pair >>> 16,
          );
        }
      }
    }
  }
  return logical;
}

export function aceOpt0003SubgroupGemmWgsl(
  shape: AceGemmShape,
  hasBias: boolean,
): string {
  const plan = planAceOpt0003SubgroupGemm(shape);
  const biasDeclaration = hasBias
    ? "@group(0) @binding(3) var<storage, read> bias: array<u32>;"
    : "";
  const outputRangeBinding = hasBias ? 4 : 3;
  const biasAdd = hasBias
    ? "value = value + load_bias_vec4(column);"
    : "";
  const biasLoader = hasBias
    ? /* wgsl */ `
fn load_bias_vec4(first_scalar: u32) -> vec4<f32> {
  let first_pair = bias[first_scalar >> 1u];
  let second_pair = bias[(first_scalar >> 1u) + 1u];
  return vec4<f32>(
    decode_bf16_low(first_pair),
    decode_bf16_high(first_pair),
    decode_bf16_low(second_pair),
    decode_bf16_high(second_pair),
  );
}`
    : "";
  const accumulatorDeclarations = Array.from(
    { length: ROWS_PER_SUBGROUP },
    (_, row) => `  var acc${row} = vec4<f32>(0.0);`,
  ).join("\n");
  const broadcastsAndContractions = Array.from(
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
      var value = acc${row};
      ${biasAdd}
      output[row * COLUMNS + column] = value.x;
      output[row * COLUMNS + column + 1u] = value.y;
      output[row * COLUMNS + column + 2u] = value.z;
      output[row * COLUMNS + column + 3u] = value.w;
    }
  }`,
  ).join("\n");

  return /* wgsl */ `
enable subgroups;

const ROWS: u32 = ${plan.rows}u;
const INNER: u32 = ${plan.inner}u;
const COLUMNS: u32 = ${plan.columns}u;
const COLUMN_TILES: u32 = ${plan.columnTiles}u;
const INNER_TILES: u32 = ${plan.innerTiles}u;

@group(0) @binding(0) var<storage, read> activation: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
${biasDeclaration}

struct OutputRangeParameters {
  first_workgroup: u32,
  _padding0: u32,
  _padding1: u32,
  _padding2: u32,
}
@group(0) @binding(${outputRangeBinding}) var<uniform>
  output_range: OutputRangeParameters;

fn decode_bf16_low(pair: u32) -> f32 {
  return bitcast<f32>((pair & 0xffffu) << 16u);
}

fn decode_bf16_high(pair: u32) -> f32 {
  return bitcast<f32>(pair & 0xffff0000u);
}

${biasLoader}

@compute @workgroup_size(${ACE_OPT_0003_SUBGROUP_GEMM_WORKGROUP_SIZE}, 1, 1)
fn main(
  @builtin(subgroup_invocation_id) subgroup_lane: u32,
  @builtin(subgroup_id) subgroup: u32,
  @builtin(subgroup_size) subgroup_size: u32,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  if (subgroup_size != ${ACE_OPT_0003_SUBGROUP_SIZE}u) {
    return;
  }
  let linear_group = output_range.first_workgroup + group.x;
  let row_tile = linear_group / COLUMN_TILES;
  let column_tile = linear_group % COLUMN_TILES;
  let row_base =
    row_tile * ${ACE_OPT_0003_SUBGROUP_GEMM_TILE_ROWS}u +
    subgroup * ${ROWS_PER_SUBGROUP}u;
  let column =
    column_tile * ${ACE_OPT_0003_SUBGROUP_GEMM_TILE_COLUMNS}u +
    subgroup_lane * ${COLUMNS_PER_LANE}u;
${accumulatorDeclarations}

  // This nesting visits logical K as 0, 1, ..., INNER - 1 for every output.
  for (
    var inner_tile = 0u;
    inner_tile < INNER_TILES;
    inner_tile += 1u
  ) {
    let weight_tile_base =
      (column_tile * INNER_TILES + inner_tile) *
      ${BF16_WORDS_PER_TILE}u;
    for (
      var inner_in_tile = 0u;
      inner_in_tile < ${ACE_OPT_0003_SUBGROUP_GEMM_TILE_INNER}u;
      inner_in_tile += 1u
    ) {
      let inner =
        inner_tile * ${ACE_OPT_0003_SUBGROUP_GEMM_TILE_INNER}u +
        inner_in_tile;
      var lane_a = 0.0;
      let lane_row = row_base + subgroup_lane;
      if (subgroup_lane < ${ROWS_PER_SUBGROUP}u && lane_row < ROWS) {
        lane_a = activation[lane_row * INNER + inner];
      }
      let weight_word =
        weight_tile_base +
        inner_in_tile * ${BF16_WORDS_PER_TILE_INNER_ROW}u +
        subgroup_lane * 2u;
      let first_pair = weight[weight_word];
      let second_pair = weight[weight_word + 1u];
      let b = vec4<f32>(
        decode_bf16_low(first_pair),
        decode_bf16_high(first_pair),
        decode_bf16_low(second_pair),
        decode_bf16_high(second_pair),
      );
${broadcastsAndContractions}
    }
  }
${stores}
}
`;
}

async function compileAceOpt0003SubgroupGemm(
  device: GPUDevice,
  shape: AceGemmShape,
  hasBias: boolean,
): Promise<CompiledAceOpt0003SubgroupGemm> {
  const plan = planAceOpt0003SubgroupGemm(shape);
  const label =
    `ace-opt-0003-subgroup-gemm-${shape.rows}x${shape.inner}x${shape.columns}` +
    (hasBias ? "-bias" : "");
  const module = device.createShaderModule({
    label,
    code: aceOpt0003SubgroupGemmWgsl(shape, hasBias),
  });
  const pipeline = await device.createComputePipelineAsync({
    label,
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });
  const reportedAlignment = device.limits.minUniformBufferOffsetAlignment;
  const outputRangeParameterStride = Math.max(
    MINIMUM_UNIFORM_STRIDE,
    reportedAlignment,
  );
  if (
    !Number.isSafeInteger(outputRangeParameterStride) ||
    outputRangeParameterStride <= 0
  ) {
    throw new Error("OPT-0003 device reported an invalid uniform alignment");
  }
  const parameterBytes = checkedProduct(
    Math.max(1, plan.outputRangeCount),
    outputRangeParameterStride,
    "range parameter bytes",
  );
  const allocated = await createAceScopedBuffers(
    device,
    [{
      label: `${label}-output-range-parameters`,
      size: parameterBytes,
      usage: GPUBufferUsage.UNIFORM,
      mappedAtCreation: true,
    }],
    `${label} output range parameters`,
  );
  const outputRangeParameters = allocated[0];
  if (outputRangeParameters === undefined) {
    throw new Error(`${label} output range allocation returned no buffer`);
  }
  try {
    const mapped = outputRangeParameters.getMappedRange();
    for (let index = 0; index < plan.outputRanges.length; index += 1) {
      new Uint32Array(
        mapped,
        index * outputRangeParameterStride,
        OUTPUT_RANGE_PARAMETER_BYTES / Uint32Array.BYTES_PER_ELEMENT,
      )[0] = plan.outputRanges[index]!.firstWorkgroup;
    }
    outputRangeParameters.unmap();
    return Object.freeze({
      pipeline,
      bindGroupLayout: pipeline.getBindGroupLayout(0),
      outputRangeParameters,
      outputRangeParameterStride,
      destroy(): void {
        outputRangeParameters.destroy();
      },
    });
  } catch (error) {
    outputRangeParameters.destroy();
    throw error;
  }
}

function requireExactPackedWords(
  values: Uint32Array,
  expectedWords: number,
  label: string,
): void {
  if (!(values instanceof Uint32Array) || values.length !== expectedWords) {
    throw new RangeError(
      `OPT-0003 ${label} weight requires exactly ${expectedWords} packed BF16 words`,
    );
  }
}

function packedBf16ScalarBits(values: Uint32Array, index: number): number {
  const pair = values[index >> 1]!;
  return (index & 1) === 0 ? pair & 0xffff : pair >>> 16;
}

function setPackedBf16ScalarBits(
  values: Uint32Array,
  index: number,
  bits: number,
): void {
  const word = index >> 1;
  const previous = values[word]!;
  values[word] = (index & 1) === 0
    ? ((previous & 0xffff0000) | (bits & 0xffff)) >>> 0
    : ((previous & 0xffff) | ((bits & 0xffff) << 16)) >>> 0;
}

function rangeOutputCount(
  rows: number,
  columnTiles: number,
  firstWorkgroup: number,
  workgroupCount: number,
): number {
  let outputs = 0;
  const end = firstWorkgroup + workgroupCount;
  for (let workgroup = firstWorkgroup; workgroup < end; workgroup += 1) {
    const rowTile = Math.floor(workgroup / columnTiles);
    const activeRows = Math.min(
      ACE_OPT_0003_SUBGROUP_GEMM_TILE_ROWS,
      rows - rowTile * ACE_OPT_0003_SUBGROUP_GEMM_TILE_ROWS,
    );
    outputs += activeRows * ACE_OPT_0003_SUBGROUP_GEMM_TILE_COLUMNS;
  }
  return outputs;
}

function exactBinding(
  binding: GPUBufferBinding,
  requiredBytes: number,
): GPUBufferBinding {
  return {
    buffer: binding.buffer,
    offset: binding.offset ?? 0,
    size: requiredBytes,
  };
}

function requirePositiveSafeInteger(value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new RangeError(
      `OPT-0003 subgroup GEMM ${label} must be a positive safe integer`,
    );
  }
}

function checkedProduct(left: number, right: number, label: string): number {
  const product = left * right;
  if (!Number.isSafeInteger(product)) {
    throw new RangeError(
      `OPT-0003 subgroup GEMM ${label} is not a safe integer`,
    );
  }
  return product;
}

function checkedBytes(elements: number, bytesPerElement: number, label: string): number {
  return checkedProduct(elements, bytesPerElement, `${label} bytes`);
}

function requireWgslIndexable(elements: number, label: string): void {
  if (elements > MAX_WGSL_U32) {
    throw new RangeError(
      `OPT-0003 subgroup GEMM ${label} exceeds WGSL's u32 indexing domain`,
    );
  }
}
