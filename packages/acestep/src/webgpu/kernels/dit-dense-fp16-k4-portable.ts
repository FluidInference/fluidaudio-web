import {
  checkedAceProduct,
  requireAceBindingBytes,
  requireAceDisjointOutput,
} from "./correctness-utils.js";
import {
  ACE_OPT_0032_DENSE_K4_PARTIALS_WEIGHT_LAYOUT,
  ACE_OPT_0032_DENSE_SUBGROUP_SIZE,
  ACE_OPT_0032_DENSE_TILE_COLUMNS,
  ACE_OPT_0032_DENSE_TILE_INNER,
  ACE_OPT_0032_DENSE_TILE_ROWS,
  ACE_OPT_0032_DENSE_WORKGROUP_SIZE,
  planAceOpt0032DenseK4Partials,
  type AceOpt0032DenseK4PartialsPlan,
} from "./dit-dense-fp16-k4-partials.js";
import type {
  AceGemmBufferBindings,
  AceGemmShape,
} from "./gemm.js";

/**
 * OPT-0088 portable identity. This kernel exists so browsers without the
 * WebGPU `subgroups` feature (Safari, Firefox) can run the production DiT
 * dense path; `shader-f16` remains required.
 */
export const ACE_OPT_0088_DENSE_K4_PORTABLE_KERNEL_ID =
  "opt-0088-dense-k4-fp16-portable-v1";
/** Identical packed-B payload; no repack is ever required to swap kernels. */
export const ACE_OPT_0088_DENSE_K4_PORTABLE_WEIGHT_LAYOUT =
  ACE_OPT_0032_DENSE_K4_PARTIALS_WEIGHT_LAYOUT;

/**
 * The former 32-lane subgroup becomes a 32-lane slice of the WG128 workgroup
 * addressed as `local_invocation_index / 32` (slice) and `% 32` (lane). All
 * geometry constants are reused from the measured OPT-0032 owner so the two
 * kernels stay tile-for-tile interchangeable.
 */
const SLICE_LANES = ACE_OPT_0032_DENSE_SUBGROUP_SIZE;
const SLICES_PER_WORKGROUP =
  ACE_OPT_0032_DENSE_WORKGROUP_SIZE / SLICE_LANES;
const ROWS_PER_SLICE =
  ACE_OPT_0032_DENSE_TILE_ROWS / SLICES_PER_WORKGROUP;
const COLUMNS_PER_LANE =
  ACE_OPT_0032_DENSE_TILE_COLUMNS / SLICE_LANES;
const STAGED_VECTORS = SLICES_PER_WORKGROUP * ROWS_PER_SLICE;
const STAGED_VECTOR_BYTES = 8;

/** 4 slices x 8 staged vec4<f16> activation vectors = 256 bytes. */
export const ACE_OPT_0088_DENSE_K4_PORTABLE_WORKGROUP_STORAGE_BYTES =
  STAGED_VECTORS * STAGED_VECTOR_BYTES;

export interface AceOpt0088DenseK4PortableDispatch {
  readonly label: string;
  readonly kernelId: typeof ACE_OPT_0088_DENSE_K4_PORTABLE_KERNEL_ID;
  readonly weightLayout:
    typeof ACE_OPT_0088_DENSE_K4_PORTABLE_WEIGHT_LAYOUT;
  readonly plan: AceOpt0032DenseK4PartialsPlan;
  readonly rangeCount: 1;
  encodeRange(pass: GPUComputePassEncoder, rangeIndex: number): void;
  encode(pass: GPUComputePassEncoder): void;
}

/**
 * Portable workgroup-memory port of the measured OPT-0032 K4-partial owner.
 *
 * Declared numerical relationship: bit-identical output to the OPT-0032
 * kernel for identical inputs. OPT-0032's lane broadcast is pure data
 * movement, so this kernel replaces it with a staging array indexed by
 * (slice, staged row) plus `workgroupBarrier()`; the values every invocation
 * observes are unchanged. Every arithmetic token is preserved from OPT-0032:
 * the same activation f32-to-f16 rounding points, the same four native FP16
 * K4 dots per staged row in ascending K order, exactly one widening of each
 * f16 partial vector into vec4<f32>, the same FP32 accumulation order, and
 * the same vec4<f32> store expression.
 *
 * Device requirements: `shader-f16` only. This kernel must never require or
 * inspect the `subgroups` feature or any subgroup size capability.
 */
export class AceOpt0088DenseK4PortableKernel {
  private readonly compiled = new Map<string, Promise<GPUComputePipeline>>();
  private destroyed = false;

  private constructor(private readonly device: GPUDevice) {}

  /**
   * Signature-compatible with the OPT-0032 owner so backend wiring can swap
   * on a backend string. The subgroup capability hints are intentionally
   * ignored: this kernel is valid on devices with no subgroup support at all.
   */
  static create(
    device: GPUDevice,
    _capability: Readonly<{
      subgroupMinSize?: number;
      subgroupMaxSize?: number;
    }> = {},
  ): AceOpt0088DenseK4PortableKernel {
    if (!device.features.has("shader-f16")) {
      throw new Error(
        "OPT-0088 portable dense K4 partials require WebGPU shader-f16",
      );
    }
    if (
      device.limits.maxComputeInvocationsPerWorkgroup <
        ACE_OPT_0032_DENSE_WORKGROUP_SIZE ||
      device.limits.maxComputeWorkgroupSizeX <
        ACE_OPT_0032_DENSE_WORKGROUP_SIZE
    ) {
      throw new Error(
        `OPT-0088 portable dense K4 partials require WG${ACE_OPT_0032_DENSE_WORKGROUP_SIZE}`,
      );
    }
    if (
      device.limits.maxComputeWorkgroupStorageSize <
        ACE_OPT_0088_DENSE_K4_PORTABLE_WORKGROUP_STORAGE_BYTES
    ) {
      throw new Error(
        `OPT-0088 portable dense K4 partials require ${ACE_OPT_0088_DENSE_K4_PORTABLE_WORKGROUP_STORAGE_BYTES} workgroup-storage bytes`,
      );
    }
    return new AceOpt0088DenseK4PortableKernel(device);
  }

  async createDispatch(
    label: string,
    shape: AceGemmShape,
    bindings: AceGemmBufferBindings,
  ): Promise<AceOpt0088DenseK4PortableDispatch> {
    if (this.destroyed) {
      throw new Error("OPT-0088 portable dense K4 kernel was destroyed");
    }
    if (bindings.bias !== undefined) {
      throw new Error("OPT-0088 repeated-layer dense GEMMs do not accept bias");
    }
    const plan = planAceOpt0032DenseK4Partials(shape);
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
        "OPT-0088 portable dense K4 kernel was destroyed while compiling",
      );
    }
    const bindGroup = this.device.createBindGroup({
      label: `${label}-opt-0088-bindings`,
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: exactBinding(bindings.activation, activationBytes) },
        { binding: 1, resource: exactBinding(bindings.weight, weightBytes) },
        { binding: 2, resource: exactBinding(bindings.output, outputBytes) },
      ],
    });
    return Object.freeze({
      label,
      kernelId: ACE_OPT_0088_DENSE_K4_PORTABLE_KERNEL_ID,
      weightLayout: ACE_OPT_0088_DENSE_K4_PORTABLE_WEIGHT_LAYOUT,
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
    const created = compileAceOpt0088DenseK4Portable(this.device, shape);
    this.compiled.set(key, created);
    void created.catch(() => {
      if (this.compiled.get(key) === created) this.compiled.delete(key);
    });
    return created;
  }
}

/**
 * Generate the portable WGSL. The staged reads replace the OPT-0032 lane
 * shuffle; the declaration, contraction, and store sections are emitted with
 * the same token sequences as the OPT-0032 generator so per-output rounding
 * and accumulation order stay bit-identical. The generated module must never
 * contain the substring "subgroup".
 */
export function aceOpt0088DenseK4PortableWgsl(shape: AceGemmShape): string {
  const plan = planAceOpt0032DenseK4Partials(shape);
  const declarations = Array.from(
    { length: ROWS_PER_SLICE },
    (_, row) => `  var acc${row} = vec4<f32>(0.0);`,
  ).join("\n");
  const stagedReads = Array.from(
    { length: ROWS_PER_SLICE },
    (_, row) => `    let a${row} = staged_a[staged_base + ${row}u];`,
  ).join("\n");
  const contractions = Array.from(
    { length: ROWS_PER_SLICE },
    (_, row) => /* wgsl */ `
    let partial${row} = vec4<f16>(
      dot(a${row}, b0),
      dot(a${row}, b1),
      dot(a${row}, b2),
      dot(a${row}, b3)
    );
    acc${row} = acc${row} + vec4<f32>(partial${row});`,
  ).join("\n");
  const stores = Array.from(
    { length: ROWS_PER_SLICE },
    (_, row) => /* wgsl */ `
  {
    let row = row_base + ${row}u;
    if (row < ROWS) {
      output[row * (COLUMNS / 4u) + column_vector] = acc${row};
    }
  }`,
  ).join("\n");
  return /* wgsl */ `
// kernel-id: ${ACE_OPT_0088_DENSE_K4_PORTABLE_KERNEL_ID}
// arithmetic-identity: bit-identical to the OPT-0032 K4-partial owner; only
// the staged-activation transport changed (lane shuffle -> shared memory).
enable f16;

const ROWS = ${plan.rows}u;
const INNER = ${plan.inner}u;
const COLUMNS = ${plan.columns}u;
const INNER_K4_GROUPS = ${plan.innerK4Groups}u;

@group(0) @binding(0) var<storage, read> activation: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<vec4<f16>>;
@group(0) @binding(2) var<storage, read_write> output: array<vec4<f32>>;

// One staged K4 activation vector per (slice, owned row): 4 x 8 vec4<f16>.
var<workgroup> staged_a: array<vec4<f16>, ${STAGED_VECTORS}>;

@compute @workgroup_size(${ACE_OPT_0032_DENSE_WORKGROUP_SIZE}, 1, 1)
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
    group.y * ${ACE_OPT_0032_DENSE_TILE_ROWS}u +
    slice * ${ROWS_PER_SLICE}u;
  let column_vector =
    group.x * ${SLICE_LANES}u + lane;
${declarations}

  for (var inner_k4 = 0u; inner_k4 < INNER_K4_GROUPS; inner_k4 += 1u) {
    let inner_base = inner_k4 * ${ACE_OPT_0032_DENSE_TILE_INNER}u;
    var lane_a = vec4<f16>(0.0h);
    let lane_row = row_base + lane;
    if (lane < ${ROWS_PER_SLICE}u && lane_row < ROWS) {
      let activation_base = lane_row * INNER + inner_base;
      lane_a = vec4<f16>(
        f16(activation[activation_base]),
        f16(activation[activation_base + 1u]),
        f16(activation[activation_base + 2u]),
        f16(activation[activation_base + 3u])
      );
    }
    if (lane < ${ROWS_PER_SLICE}u) {
      staged_a[staged_base + lane] = lane_a;
    }
    let weight_base =
      ((group.x * INNER_K4_GROUPS + inner_k4) *
      ${COLUMNS_PER_LANE}u) * ${SLICE_LANES}u +
      lane;
    let b0 = weight[weight_base];
    let b1 = weight[weight_base + ${SLICE_LANES}u];
    let b2 = weight[weight_base + ${2 * SLICE_LANES}u];
    let b3 = weight[weight_base + ${3 * SLICE_LANES}u];
    workgroupBarrier();
${stagedReads}
${contractions}
    workgroupBarrier();
  }
${stores}
}
`;
}

async function compileAceOpt0088DenseK4Portable(
  device: GPUDevice,
  shape: AceGemmShape,
): Promise<GPUComputePipeline> {
  const label =
    `ace-opt-0088-dense-k4-portable-${shape.rows}x${shape.inner}x${shape.columns}`;
  const module = device.createShaderModule({
    label,
    code: aceOpt0088DenseK4PortableWgsl(shape),
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
  plan: AceOpt0032DenseK4PartialsPlan,
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
