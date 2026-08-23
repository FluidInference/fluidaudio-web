import { ACE_DIT_DENSE_FP16_TILE_LAYOUT } from "../../model/manifest.js";
import {
  checkedAceProduct,
  requireAceBindingBytes,
  requireAceDisjointOutput,
} from "./correctness-utils.js";
import {
  ACE_OPT_0009_DENSE_SUBGROUP_SIZE,
  ACE_OPT_0009_DENSE_TILE_COLUMNS,
  ACE_OPT_0009_DENSE_TILE_INNER,
  ACE_OPT_0009_DENSE_TILE_ROWS,
  ACE_OPT_0009_DENSE_WORKGROUP_SIZE,
  planAceOpt0009DenseGemm,
  type AceOpt0009DenseGemmPlan,
} from "./dit-dense-fp16.js";
import type {
  AceGemmBufferBindings,
  AceGemmDispatch,
  AceGemmKernel,
  AceGemmShape,
} from "./gemm.js";

/**
 * OPT-0088 portable identity for the production rev7-oracle dense package.
 * This kernel exists so browsers without the WebGPU `subgroups` feature
 * (Safari, Firefox) can run the hosted production DiT dense path;
 * `shader-f16` remains required.
 */
export const ACE_OPT_0088_DENSE_PORTABLE_KERNEL_ID =
  "opt-0088-dense-fp16-fp32-portable-v1";
/** Identical packed-B payload; no repack is ever required to swap kernels. */
export const ACE_OPT_0088_DENSE_PORTABLE_WEIGHT_LAYOUT =
  ACE_DIT_DENSE_FP16_TILE_LAYOUT;

/**
 * The former 32-lane subgroup becomes a 32-lane slice of the WG128 workgroup
 * addressed as `local_invocation_index / 32` (slice) and `% 32` (lane). All
 * geometry constants are reused from the measured OPT-0009 owner so the two
 * kernels stay tile-for-tile interchangeable.
 */
const SLICE_LANES = ACE_OPT_0009_DENSE_SUBGROUP_SIZE;
const SLICES_PER_WORKGROUP =
  ACE_OPT_0009_DENSE_WORKGROUP_SIZE / SLICE_LANES;
const ROWS_PER_SLICE =
  ACE_OPT_0009_DENSE_TILE_ROWS / SLICES_PER_WORKGROUP;
const SCALARS_PER_LANE =
  ACE_OPT_0009_DENSE_TILE_COLUMNS / SLICE_LANES;
const STAGED_SCALARS = SLICES_PER_WORKGROUP * ROWS_PER_SLICE;
const STAGED_SCALAR_BYTES = 2;

/** 4 slices x 8 staged f16 activation scalars = 64 bytes. */
export const ACE_OPT_0088_DENSE_PORTABLE_WORKGROUP_STORAGE_BYTES =
  STAGED_SCALARS * STAGED_SCALAR_BYTES;

export interface AceOpt0088DensePortableDispatch extends AceGemmDispatch {
  readonly kernelId: typeof ACE_OPT_0088_DENSE_PORTABLE_KERNEL_ID;
  readonly weightLayout: typeof ACE_OPT_0088_DENSE_PORTABLE_WEIGHT_LAYOUT;
  readonly plan: AceOpt0009DenseGemmPlan;
  readonly rangeCount: 1;
}

/**
 * Portable workgroup-memory port of the measured OPT-0009 N256/K32 owner.
 *
 * Declared numerical relationship: bit-identical output to the OPT-0009
 * kernel for identical inputs. OPT-0009's lane broadcast is pure data
 * movement, so this kernel replaces it with a staging array indexed by
 * (slice, staged row) plus `workgroupBarrier()`; the values every invocation
 * observes are unchanged. Every arithmetic token is preserved from OPT-0009:
 * the same activation f32-to-f16 rounding point, the same packed-B
 * `unpack_f16x4` decode, exactly one widening of each staged f16 activation
 * and each vec4<f16> weight vector into vec4<f32> per contraction, the same
 * FP32 accumulation order over ascending K, and the same vec4<f32> store
 * expressions.
 *
 * Device requirements: `shader-f16` only. This kernel must never require or
 * inspect the `subgroups` feature or any subgroup size capability.
 */
export class AceOpt0088DensePortableKernel implements AceGemmKernel {
  private readonly compiled = new Map<string, Promise<GPUComputePipeline>>();
  private destroyed = false;

  private constructor(private readonly device: GPUDevice) {}

  /**
   * Signature-compatible with the OPT-0009 owner so backend wiring can swap
   * on a backend string. The subgroup capability hints are intentionally
   * ignored: this kernel is valid on devices with no subgroup support at all.
   */
  static create(
    device: GPUDevice,
    _capability: Readonly<{
      subgroupMinSize?: number;
      subgroupMaxSize?: number;
    }> = {},
  ): AceOpt0088DensePortableKernel {
    if (!device.features.has("shader-f16")) {
      throw new Error(
        "OPT-0088 portable dense GEMM requires WebGPU shader-f16",
      );
    }
    if (
      device.limits.maxComputeInvocationsPerWorkgroup <
        ACE_OPT_0009_DENSE_WORKGROUP_SIZE ||
      device.limits.maxComputeWorkgroupSizeX <
        ACE_OPT_0009_DENSE_WORKGROUP_SIZE
    ) {
      throw new Error(
        `OPT-0088 portable dense GEMM requires WG${ACE_OPT_0009_DENSE_WORKGROUP_SIZE}`,
      );
    }
    if (
      device.limits.maxComputeWorkgroupStorageSize <
        ACE_OPT_0088_DENSE_PORTABLE_WORKGROUP_STORAGE_BYTES
    ) {
      throw new Error(
        `OPT-0088 portable dense GEMM requires ${ACE_OPT_0088_DENSE_PORTABLE_WORKGROUP_STORAGE_BYTES} workgroup-storage bytes`,
      );
    }
    return new AceOpt0088DensePortableKernel(device);
  }

  async createDispatch(
    label: string,
    shape: AceGemmShape,
    bindings: AceGemmBufferBindings,
  ): Promise<AceOpt0088DensePortableDispatch> {
    if (this.destroyed) {
      throw new Error("OPT-0088 portable dense GEMM kernel was destroyed");
    }
    if (bindings.bias !== undefined) {
      throw new Error("OPT-0088 repeated-layer dense GEMMs do not accept bias");
    }
    const plan = planAceOpt0009DenseGemm(shape);
    const activationBytes = checkedAceProduct(
      [plan.activationElements, 4],
      `${label} activation bytes`,
    );
    const weightBytes = checkedAceProduct(
      [plan.weightElements, 2],
      `${label} weight bytes`,
    );
    const outputBytes = checkedAceProduct(
      [plan.outputElements, 4],
      `${label} output bytes`,
    );
    requireAceBindingBytes(bindings.activation, activationBytes, `${label} activation`);
    requireAceBindingBytes(bindings.weight, weightBytes, `${label} weight`);
    requireAceBindingBytes(bindings.output, outputBytes, `${label} output`);
    requireAceDisjointOutput(
      exactBinding(bindings.output, outputBytes),
      [
        exactBinding(bindings.activation, activationBytes),
        exactBinding(bindings.weight, weightBytes),
      ],
      label,
    );
    const pipeline = await this.pipelineFor(shape);
    if (this.destroyed) {
      throw new Error(
        "OPT-0088 portable dense GEMM kernel was destroyed while compiling",
      );
    }
    const bindGroup = this.device.createBindGroup({
      label: `${label}-opt-0088-dense-bindings`,
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: exactBinding(bindings.activation, activationBytes) },
        { binding: 1, resource: exactBinding(bindings.weight, weightBytes) },
        { binding: 2, resource: exactBinding(bindings.output, outputBytes) },
      ],
    });
    return Object.freeze({
      label,
      kernelId: ACE_OPT_0088_DENSE_PORTABLE_KERNEL_ID,
      weightLayout: ACE_OPT_0088_DENSE_PORTABLE_WEIGHT_LAYOUT,
      plan,
      rangeCount: 1 as const,
      encodeRange(pass: GPUComputePassEncoder, rangeIndex: number): void {
        if (rangeIndex !== 0) {
          throw new RangeError(`${label} OPT-0088 dense range must be zero`);
        }
        encodeDispatch(pass, pipeline, bindGroup, plan);
      },
      encode(pass: GPUComputePassEncoder): void {
        encodeDispatch(pass, pipeline, bindGroup, plan);
      },
    });
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.compiled.clear();
  }

  private pipelineFor(shape: AceGemmShape): Promise<GPUComputePipeline> {
    const key = `${shape.rows}x${shape.inner}x${shape.columns}`;
    const existing = this.compiled.get(key);
    if (existing !== undefined) return existing;
    const created = compileAceOpt0088DensePortable(this.device, shape);
    this.compiled.set(key, created);
    void created.catch(() => {
      if (this.compiled.get(key) === created) this.compiled.delete(key);
    });
    return created;
  }
}

/**
 * Generate the portable WGSL. The staged reads replace the OPT-0009 lane
 * broadcast; the declaration, contraction, and store sections are emitted
 * with the same token sequences as the OPT-0009 generator so per-output
 * rounding and accumulation order stay bit-identical. The generated module
 * must never contain the substring "subgroup".
 */
export function aceOpt0088DensePortableWgsl(shape: AceGemmShape): string {
  const plan = planAceOpt0009DenseGemm(shape);
  const declarations = Array.from(
    { length: ROWS_PER_SLICE },
    (_, row) =>
      `  var acc${row}_0 = vec4<f32>(0.0);\n` +
      `  var acc${row}_1 = vec4<f32>(0.0);`,
  ).join("\n");
  const stagedReads = Array.from(
    { length: ROWS_PER_SLICE },
    (_, row) => `      let a${row} = staged_a[staged_base + ${row}u];`,
  ).join("\n");
  const contractions = Array.from(
    { length: ROWS_PER_SLICE },
    (_, row) =>
      `      acc${row}_0 = acc${row}_0 + vec4<f32>(f32(a${row})) * vec4<f32>(b0);\n` +
      `      acc${row}_1 = acc${row}_1 + vec4<f32>(f32(a${row})) * vec4<f32>(b1);`,
  ).join("\n");
  const stores = Array.from(
    { length: ROWS_PER_SLICE },
    (_, row) => /* wgsl */ `
  {
    let row = row_base + ${row}u;
    if (row < ROWS) {
      let vector_base = row * (COLUMNS / 4u) + column_base / 4u;
      output[vector_base] = acc${row}_0;
      output[vector_base + 1u] = acc${row}_1;
    }
  }`,
  ).join("\n");
  return /* wgsl */ `
// kernel-id: ${ACE_OPT_0088_DENSE_PORTABLE_KERNEL_ID}
// arithmetic-identity: bit-identical to the OPT-0009 N256/K32 owner; only
// the broadcast-activation transport changed (lane shuffle -> shared memory).
enable f16;

const ROWS = ${plan.rows}u;
const INNER = ${plan.inner}u;
const COLUMNS = ${plan.columns}u;
const INNER_TILES = ${plan.innerTiles}u;

@group(0) @binding(0) var<storage, read> activation: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> output: array<vec4<f32>>;

fn unpack_f16x4(low: u32, high: u32) -> vec4<f16> {
  let low_pair = unpack2x16float(low);
  let high_pair = unpack2x16float(high);
  return vec4<f16>(
    f16(low_pair.x), f16(low_pair.y), f16(high_pair.x), f16(high_pair.y)
  );
}

// One staged FP16 activation scalar per (slice, owned row): 4 x 8 f16.
var<workgroup> staged_a: array<f16, ${STAGED_SCALARS}>;

@compute @workgroup_size(${ACE_OPT_0009_DENSE_WORKGROUP_SIZE}, 1, 1)
fn main(
  @builtin(local_invocation_index) local_index: u32,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  // Workgroup-uniform guard: barriers below stay uniformly executed.
  if (
    group.x >= ${plan.columnTiles}u ||
    group.y >= ${plan.rowTiles}u ||
    group.z != 0u
  ) {
    return;
  }
  let slice = local_index / ${SLICE_LANES}u;
  let lane = local_index % ${SLICE_LANES}u;
  let staged_base = slice * ${ROWS_PER_SLICE}u;
  let row_base =
    group.y * ${ACE_OPT_0009_DENSE_TILE_ROWS}u +
    slice * ${ROWS_PER_SLICE}u;
  let column_base =
    group.x * ${ACE_OPT_0009_DENSE_TILE_COLUMNS}u +
    lane * ${SCALARS_PER_LANE}u;
${declarations}

  for (var inner_tile = 0u; inner_tile < INNER_TILES; inner_tile += 1u) {
    let weight_tile_base =
      (group.x * INNER_TILES + inner_tile) *
      ${ACE_OPT_0009_DENSE_TILE_INNER * SLICE_LANES}u;
    for (
      var inner_in_tile = 0u;
      inner_in_tile < ${ACE_OPT_0009_DENSE_TILE_INNER}u;
      inner_in_tile += 1u
    ) {
      let inner = inner_tile * ${ACE_OPT_0009_DENSE_TILE_INNER}u + inner_in_tile;
      var lane_a = 0.0h;
      let lane_row = row_base + lane;
      if (lane < ${ROWS_PER_SLICE}u && lane_row < ROWS) {
        lane_a = f16(activation[lane_row * INNER + inner]);
      }
      if (lane < ${ROWS_PER_SLICE}u) {
        staged_a[staged_base + lane] = lane_a;
      }
      let packed_b = weight[
        weight_tile_base +
        inner_in_tile * ${SLICE_LANES}u +
        lane
      ];
      let b0 = unpack_f16x4(packed_b.x, packed_b.y);
      let b1 = unpack_f16x4(packed_b.z, packed_b.w);
      workgroupBarrier();
${stagedReads}
${contractions}
      workgroupBarrier();
    }
  }
${stores}
}
`;
}

async function compileAceOpt0088DensePortable(
  device: GPUDevice,
  shape: AceGemmShape,
): Promise<GPUComputePipeline> {
  const label =
    `ace-opt-0088-dense-portable-${shape.rows}x${shape.inner}x${shape.columns}`;
  const module = device.createShaderModule({
    label,
    code: aceOpt0088DensePortableWgsl(shape),
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

function encodeDispatch(
  pass: GPUComputePassEncoder,
  pipeline: GPUComputePipeline,
  bindGroup: GPUBindGroup,
  plan: AceOpt0009DenseGemmPlan,
): void {
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(plan.columnTiles, plan.rowTiles, 1);
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
