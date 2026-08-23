import type { AceModelProfileId } from "../capabilities.js";
import {
  ACE_DIT_DENSE_K4_FP16_LAYOUT,
  ACE_DIT_DENSE_FP16_TILE_LAYOUT,
  ACE_DIT_GEMM_TILE_LAYOUT,
} from "../../model/manifest.js";
import { createAceScopedBuffers } from "../scoped-buffer-allocation.js";
import { requireAceDisjointOutput } from "./correctness-utils.js";

const SCALAR_WORKGROUP_COLUMNS = 8;
const SCALAR_WORKGROUP_ROWS = 8;
const TILED_WORKGROUP_SIZE = 128;
const TILED_COLUMNS = 128;
const TILED_COLUMN_VECTORS = TILED_COLUMNS / 4;
const TILED_ROW_GROUPS = TILED_WORKGROUP_SIZE / TILED_COLUMN_VECTORS;
const TILED_ROWS_PER_LANE = 4;
const TILED_ROWS = TILED_ROW_GROUPS * TILED_ROWS_PER_LANE;
const TILED_INNER = 16;
const TILED_A_VALUES = TILED_ROWS * TILED_INNER;
const TILED_B_VALUES = TILED_INNER * TILED_COLUMNS;
const DIT_GEMM_WEIGHT_TILE_COLUMNS = 128;
const DIT_GEMM_WEIGHT_TILE_INNER = 32;
export const ACE_DIT_GEMM_WEIGHT_LAYOUT =
  ACE_DIT_GEMM_TILE_LAYOUT;
export type AceGemmWeightLayout =
  | "source-row-major"
  | typeof ACE_DIT_GEMM_WEIGHT_LAYOUT
  | typeof ACE_DIT_DENSE_FP16_TILE_LAYOUT
  | typeof ACE_DIT_DENSE_K4_FP16_LAYOUT;
export const ACE_TILED_GEMM_WORKGROUP_BYTES =
  (TILED_A_VALUES + TILED_B_VALUES) * Float32Array.BYTES_PER_ELEMENT;
/**
 * Provisional M3 Stage-1 responsiveness ceiling. The exact-shape Chrome
 * harness must keep every drained range below 100 ms before this can grow.
 */
export const ACE_GEMM_MAX_OUTPUTS_PER_RANGE = 8 * 1024 * 1024;
export const ACE_GEMM_MAX_MULTIPLY_ADDS_PER_RANGE = 2 * 1024 * 1024 * 1024;
const MAX_DISPATCH_DIMENSION = 65_535;
const STORAGE_UNIFORM_ALIGNMENT = 256;
const OUTPUT_RANGE_PARAMETER_BYTES = 16;

export interface AceGemmShape {
  readonly rows: number;
  readonly inner: number;
  readonly columns: number;
}

export interface AceGemmPlan extends AceGemmShape {
  readonly workgroupsX: number;
  readonly workgroupsY: number;
  readonly activationElements: number;
  readonly weightElements: number;
  readonly outputElements: number;
}

export interface AceGemmOutputRange {
  readonly firstOutput: number;
  readonly outputCount: number;
  readonly firstWorkgroup: number;
  readonly workgroupCount: number;
  readonly multiplyAdds: number;
}

export interface AceCooperativeGemmPlan extends AceGemmPlan {
  readonly tileRows: number;
  readonly tileColumns: number;
  readonly tileInner: number;
  readonly workgroupSize: number;
  readonly outputRangeCount: number;
  readonly outputRanges: readonly AceGemmOutputRange[];
}

export interface AceTiledGemmPlan extends AceCooperativeGemmPlan {
  readonly tileRows: typeof TILED_ROWS;
  readonly tileColumns: typeof TILED_COLUMNS;
  readonly tileInner: typeof TILED_INNER;
  readonly workgroupSize: typeof TILED_WORKGROUP_SIZE;
}

export interface AceGemmBufferBindings {
  readonly activation: GPUBufferBinding;
  /** Weight in the kernel's declared physical layout. */
  readonly weight: GPUBufferBinding;
  readonly output: GPUBufferBinding;
  /** Packed BF16 or FP16, matching the selected model profile. */
  readonly bias?: GPUBufferBinding;
}

export interface AceGemmDispatch {
  readonly label: string;
  readonly weightLayout: AceGemmWeightLayout;
  readonly plan: AceCooperativeGemmPlan;
  /** One bounded output-domain dispatch, in monotonically increasing order. */
  readonly rangeCount: number;
  encodeRange(pass: GPUComputePassEncoder, rangeIndex: number): void;
  /** Correctness-harness convenience. Production graph owners use ranges. */
  encode(pass: GPUComputePassEncoder): void;
}

export interface AceGemmKernel {
  createDispatch(
    label: string,
    shape: AceGemmShape,
    bindings: AceGemmBufferBindings,
  ): Promise<AceGemmDispatch>;
  destroy(): void;
}

export interface AceGpuEncodeQuantum {
  readonly id: string;
  readonly primitiveCount: number;
  /** Known scheduled GEMM work; absent for non-GEMM primitives. */
  readonly scheduledMultiplyAdds?: number;
  encode(pass: GPUComputePassEncoder): void;
}

/** Immutable compile-time description of one encoded primitive/range. */
export interface AceGpuEncodeSequenceMemberDescriptor {
  readonly id: string;
  readonly label: string;
  readonly rangeIndex: number | null;
  readonly primitiveCount: number;
  readonly scheduledMultiplyAdds: number;
}

/**
 * Immutable physical-command membership. Production profiling builds these
 * before submission; the ordinary encode path does not materialize them.
 */
export interface AceGpuEncodeSequenceQuantumDescriptor {
  readonly id: string;
  readonly primitiveCount: number;
  readonly scheduledMultiplyAdds: number;
  readonly members: readonly AceGpuEncodeSequenceMemberDescriptor[];
}

/** Compact, index-addressed cooperative work without one closure per range. */
export interface AceGpuEncodeSequence {
  readonly quantumCount: number;
  describeQuantum(quantumIndex: number): AceGpuEncodeSequenceQuantumDescriptor;
  encodeQuantum(pass: GPUComputePassEncoder, quantumIndex: number): void;
}

/**
 * Expand a primitive into command-buffer-sized encoders. Composite runtimes
 * concatenate these arrays in dependency order; the FIFO owner submits one
 * returned quantum at a time. This is the sole production path that preserves
 * GEMM output-range cooperation through higher-level graph composition.
 */
export function acePrimitiveCooperativeQuanta(
  dispatch: Readonly<{
      readonly label: string;
      readonly plan?: Pick<AceCooperativeGemmPlan, "outputRanges">;
      readonly rangeCount: number;
      encodeRange(pass: GPUComputePassEncoder, rangeIndex: number): void;
    }> |
    Readonly<{
      readonly label: string;
      encode(pass: GPUComputePassEncoder): void;
    }>,
): readonly AceGpuEncodeQuantum[] {
  if ("rangeCount" in dispatch && "encodeRange" in dispatch) {
    return Object.freeze(Array.from(
      { length: dispatch.rangeCount },
      (_, rangeIndex) => Object.freeze({
        id: `${dispatch.label}-range-${rangeIndex}`,
        primitiveCount: 1,
        ...(dispatch.plan?.outputRanges[rangeIndex] === undefined
          ? {}
          : {
              scheduledMultiplyAdds:
                dispatch.plan.outputRanges[rangeIndex]!.multiplyAdds,
            }),
        encode(pass: GPUComputePassEncoder): void {
          dispatch.encodeRange(pass, rangeIndex);
        },
      }),
    ));
  }
  return Object.freeze([Object.freeze({
    id: dispatch.label,
    primitiveCount: 1,
    encode(pass: GPUComputePassEncoder): void {
      dispatch.encode(pass);
    },
  })]);
}

export function aceCompositeCooperativeQuanta(
  dispatches: readonly (
    Pick<
      AceGemmDispatch,
      "label" | "plan" | "rangeCount" | "encodeRange"
    > |
    Readonly<{
      readonly label: string;
      encode(pass: GPUComputePassEncoder): void;
    }>
  )[],
): readonly AceGpuEncodeQuantum[] {
  const primitives = dispatches.flatMap(acePrimitiveCooperativeQuanta);
  const packed: AceGpuEncodeQuantum[] = [];
  let members: AceGpuEncodeQuantum[] = [];
  let scheduledMultiplyAdds = 0;
  const flush = (): void => {
    if (members.length === 0) return;
    const captured = Object.freeze(members);
    const work = scheduledMultiplyAdds;
    packed.push(Object.freeze({
      id: captured.length === 1
        ? captured[0]!.id
        : `${captured[0]!.id}..${captured.at(-1)!.id}`,
      primitiveCount: captured.reduce(
        (total, member) => total + member.primitiveCount,
        0,
      ),
      ...(work === 0 ? {} : { scheduledMultiplyAdds: work }),
      encode(pass: GPUComputePassEncoder): void {
        for (const member of captured) member.encode(pass);
      },
    }));
    members = [];
    scheduledMultiplyAdds = 0;
  };
  for (const primitive of primitives) {
    const work = primitive.scheduledMultiplyAdds ?? 0;
    if (
      work > 0 &&
      scheduledMultiplyAdds > 0 &&
      scheduledMultiplyAdds + work > ACE_GEMM_MAX_MULTIPLY_ADDS_PER_RANGE
    ) {
      flush();
    }
    members.push(primitive);
    scheduledMultiplyAdds += work;
  }
  flush();
  return Object.freeze(packed);
}

/**
 * Compact equivalent of `aceCompositeCooperativeQuanta`. It retains only the
 * source dispatch list and rescans that short list for the requested quantum,
 * avoiding thousands of long-lived range closures in large DiT graphs.
 */
export function aceCompositeCooperativeSequence(
  dispatches: readonly (
    Pick<
      AceGemmDispatch,
      "label" | "plan" | "rangeCount" | "encodeRange"
    > |
    Readonly<{
      readonly label: string;
      encode(pass: GPUComputePassEncoder): void;
    }>
  )[],
): AceGpuEncodeSequence {
  if (dispatches.length === 0) {
    throw new RangeError("ACE cooperative sequence requires a primitive");
  }
  const quantumCount = countCompositeDispatchQuanta(dispatches);
  return Object.freeze({
    quantumCount,
    describeQuantum(
      quantumIndex: number,
    ): AceGpuEncodeSequenceQuantumDescriptor {
      requireCompositeQuantumIndex(quantumIndex, quantumCount);
      const members: AceGpuEncodeSequenceMemberDescriptor[] = [];
      let scheduledMultiplyAdds = 0;
      let primitiveCount = 0;
      visitCompositeQuantum(
        dispatches,
        quantumIndex,
        (_encode, work, label, rangeIndex) => {
          const id = rangeIndex === null
            ? label
            : `${label}-range-${rangeIndex}`;
          members.push(Object.freeze({
            id,
            label,
            rangeIndex,
            primitiveCount: 1,
            scheduledMultiplyAdds: work,
          }));
          primitiveCount += 1;
          scheduledMultiplyAdds += work;
        },
      );
      if (members.length === 0) {
        throw new Error(
          `ACE cooperative quantum ${quantumIndex} described no work`,
        );
      }
      const first = members[0]!.id;
      const last = members.at(-1)!.id;
      return Object.freeze({
        id: members.length === 1 ? first : `${first}..${last}`,
        primitiveCount,
        scheduledMultiplyAdds,
        members: Object.freeze(members),
      });
    },
    encodeQuantum(
      pass: GPUComputePassEncoder,
      quantumIndex: number,
    ): void {
      if (
        !Number.isSafeInteger(quantumIndex) ||
        quantumIndex < 0 ||
        quantumIndex >= quantumCount
      ) {
        throw new RangeError(
          `ACE cooperative quantum ${quantumIndex} is outside [0, ${quantumCount})`,
        );
      }
      let group = 0;
      let groupWork = 0;
      let encoded = false;
      visitCompositeDispatchItems(dispatches, (encode, work) => {
        if (
          work > 0 &&
          groupWork > 0 &&
          groupWork + work > ACE_GEMM_MAX_MULTIPLY_ADDS_PER_RANGE
        ) {
          group += 1;
          groupWork = 0;
        }
        if (group === quantumIndex) {
          encode(pass);
          encoded = true;
        }
        groupWork += work;
      });
      if (!encoded) {
        throw new Error(`ACE cooperative quantum ${quantumIndex} encoded no work`);
      }
    },
  });
}

/** Count the same sequential work-budget packing without compiled dispatches. */
export function planAceCompositeCooperativeQuantumCount(
  gemmPlans: readonly Readonly<Pick<
    AceCooperativeGemmPlan,
    "outputRanges"
  >>[],
): number {
  if (gemmPlans.length === 0) return 1;
  let count = 1;
  let scheduledMultiplyAdds = 0;
  for (const plan of gemmPlans) {
    for (const range of plan.outputRanges) {
      if (
        scheduledMultiplyAdds > 0 &&
        scheduledMultiplyAdds + range.multiplyAdds >
          ACE_GEMM_MAX_MULTIPLY_ADDS_PER_RANGE
      ) {
        count += 1;
        scheduledMultiplyAdds = 0;
      }
      scheduledMultiplyAdds += range.multiplyAdds;
    }
  }
  return count;
}

function countCompositeDispatchQuanta(
  dispatches: Parameters<typeof aceCompositeCooperativeSequence>[0],
): number {
  let count = 1;
  let scheduledMultiplyAdds = 0;
  visitCompositeDispatchItems(dispatches, (_encode, work) => {
    if (
      work > 0 &&
      scheduledMultiplyAdds > 0 &&
      scheduledMultiplyAdds + work > ACE_GEMM_MAX_MULTIPLY_ADDS_PER_RANGE
    ) {
      count += 1;
      scheduledMultiplyAdds = 0;
    }
    scheduledMultiplyAdds += work;
  });
  return count;
}

function requireCompositeQuantumIndex(
  quantumIndex: number,
  quantumCount: number,
): void {
  if (
    !Number.isSafeInteger(quantumIndex) ||
    quantumIndex < 0 ||
    quantumIndex >= quantumCount
  ) {
    throw new RangeError(
      `ACE cooperative quantum ${quantumIndex} is outside [0, ${quantumCount})`,
    );
  }
}

function visitCompositeQuantum(
  dispatches: Parameters<typeof aceCompositeCooperativeSequence>[0],
  quantumIndex: number,
  visit: (
    encode: (pass: GPUComputePassEncoder) => void,
    scheduledMultiplyAdds: number,
    label: string,
    rangeIndex: number | null,
  ) => void,
): void {
  let group = 0;
  let groupWork = 0;
  visitCompositeDispatchItems(
    dispatches,
    (encode, work, label, rangeIndex) => {
      if (
        work > 0 &&
        groupWork > 0 &&
        groupWork + work > ACE_GEMM_MAX_MULTIPLY_ADDS_PER_RANGE
      ) {
        group += 1;
        groupWork = 0;
      }
      if (group === quantumIndex) {
        visit(encode, work, label, rangeIndex);
      }
      groupWork += work;
    },
  );
}

function visitCompositeDispatchItems(
  dispatches: Parameters<typeof aceCompositeCooperativeSequence>[0],
  visit: (
    encode: (pass: GPUComputePassEncoder) => void,
    scheduledMultiplyAdds: number,
    label: string,
    rangeIndex: number | null,
  ) => void,
): void {
  for (const dispatch of dispatches) {
    if ("rangeCount" in dispatch && "encodeRange" in dispatch) {
      for (let rangeIndex = 0; rangeIndex < dispatch.rangeCount; rangeIndex += 1) {
        const work = dispatch.plan.outputRanges[rangeIndex]?.multiplyAdds;
        if (work === undefined) {
          throw new Error(`${dispatch.label} omitted GEMM range ${rangeIndex}`);
        }
        visit(
          (pass) => dispatch.encodeRange(pass, rangeIndex),
          work,
          dispatch.label,
          rangeIndex,
        );
      }
    } else {
      visit((pass) => dispatch.encode(pass), 0, dispatch.label, null);
    }
  }
}

interface CompiledGemm {
  readonly pipeline: GPUComputePipeline;
  readonly bindGroupLayout: GPUBindGroupLayout;
  readonly outputRangeParameters: GPUBuffer;
  destroy(): void;
}

/**
 * Portable workgroup-memory GEMM preserving the correctness contraction.
 *
 * Each lane owns four adjacent output columns for four strided rows.
 * Workgroup memory stages A and B from the selected physical weight layout,
 * but no lane shares a partial sum: every scalar accumulator still contracts
 * logical K from zero through K-1. This preserves the reference profile's
 * FP32 `sum + a * b` expression and the raw profile's per-step f16 rounding.
 * It deliberately avoids `fma`, K-parallel reduction, and subgroups.
 */
export class AceCorrectnessGemmKernel implements AceGemmKernel {
  readonly modelProfile: AceModelProfileId;
  readonly weightLayout: AceGemmWeightLayout;

  private readonly compiled = new Map<string, Promise<CompiledGemm>>();
  private destroyed = false;

  private constructor(
    private readonly device: GPUDevice,
    modelProfile: AceModelProfileId,
    weightLayout: AceGemmWeightLayout,
  ) {
    this.modelProfile = modelProfile;
    this.weightLayout = weightLayout;
  }

  static create(
    device: GPUDevice,
    modelProfile: AceModelProfileId,
    weightLayout: AceGemmWeightLayout = "source-row-major",
  ): AceCorrectnessGemmKernel {
    if (modelProfile === "raw-fp16" && !device.features.has("shader-f16")) {
      throw new Error("ACE raw-FP16 GEMM requires WebGPU shader-f16");
    }
    if (modelProfile !== "reference-bf16" && modelProfile !== "raw-fp16") {
      throw new TypeError(`Unknown ACE GEMM model profile ${String(modelProfile)}`);
    }
    requireAceGemmWeightLayout(weightLayout);
    if (
      device.limits.maxComputeInvocationsPerWorkgroup <
        TILED_WORKGROUP_SIZE ||
      device.limits.maxComputeWorkgroupSizeX < TILED_WORKGROUP_SIZE ||
      device.limits.maxComputeWorkgroupStorageSize <
        ACE_TILED_GEMM_WORKGROUP_BYTES
    ) {
      throw new Error(
        `ACE tiled GEMM requires WG${TILED_WORKGROUP_SIZE} and ` +
          `${ACE_TILED_GEMM_WORKGROUP_BYTES} bytes of workgroup storage`,
      );
    }
    return new AceCorrectnessGemmKernel(device, modelProfile, weightLayout);
  }

  async createDispatch(
    label: string,
    shape: AceGemmShape,
    bindings: AceGemmBufferBindings,
  ): Promise<AceGemmDispatch> {
    if (this.destroyed) {
      throw new Error("ACE correctness GEMM kernel was destroyed");
    }
    const plan = planAceTiledGemm(shape);
    requireAceGemmWeightLayoutShape(this.weightLayout, plan);
    requireBindingBytes(
      bindings.activation,
      activationBytes(this.modelProfile, plan.activationElements),
      `${label} activation`,
    );
    requireBindingBytes(
      bindings.weight,
      storageBytes(this.modelProfile, plan.weightElements),
      `${label} weight`,
    );
    requireBindingBytes(
      bindings.output,
      activationBytes(this.modelProfile, plan.outputElements),
      `${label} output`,
    );
    if (bindings.bias !== undefined) {
      requireBindingBytes(
        bindings.bias,
        storageBytes(this.modelProfile, plan.columns),
        `${label} bias`,
      );
    }
    requireAceDisjointOutput(
      exactBindingRange(bindings.output, activationBytes(
        this.modelProfile,
        plan.outputElements,
      )),
      [
        exactBindingRange(bindings.activation, activationBytes(
          this.modelProfile,
          plan.activationElements,
        )),
        exactBindingRange(bindings.weight, storageBytes(
          this.modelProfile,
          plan.weightElements,
        )),
        ...(bindings.bias === undefined
          ? []
          : [exactBindingRange(bindings.bias, storageBytes(
              this.modelProfile,
              plan.columns,
            ))]),
      ],
      label,
    );
    const compiled = await this.pipelineFor(plan, bindings.bias !== undefined);
    if (this.destroyed) {
      throw new Error("ACE correctness GEMM kernel was destroyed while compiling");
    }
    const rangeBinding = bindings.bias === undefined ? 3 : 4;
    const bindGroups = plan.outputRanges.map((_, rangeIndex) =>
      this.device.createBindGroup({
        label: `${label}-${this.weightLayout}-range-${rangeIndex}-bindings`,
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
              offset: rangeIndex * STORAGE_UNIFORM_ALIGNMENT,
              size: OUTPUT_RANGE_PARAMETER_BYTES,
            },
          },
        ],
      })
    );
    return Object.freeze({
      label,
      weightLayout: this.weightLayout,
      plan,
      rangeCount: plan.outputRangeCount,
      encodeRange(pass: GPUComputePassEncoder, rangeIndex: number): void {
        const range = plan.outputRanges[rangeIndex];
        if (range === undefined) {
          throw new RangeError(
            `${label} GEMM range ${rangeIndex} is outside [0, ${plan.outputRangeCount})`,
          );
        }
        pass.setPipeline(compiled.pipeline);
        pass.setBindGroup(0, bindGroups[rangeIndex]!);
        pass.dispatchWorkgroups(
          range.workgroupCount,
          1,
        );
      },
      encode(pass: GPUComputePassEncoder): void {
        pass.setPipeline(compiled.pipeline);
        for (let index = 0; index < plan.outputRanges.length; index += 1) {
          const range = plan.outputRanges[index]!;
          pass.setBindGroup(0, bindGroups[index]!);
          pass.dispatchWorkgroups(
            range.workgroupCount,
            1,
          );
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
  ): Promise<CompiledGemm> {
    const key =
      `${this.weightLayout}:${shape.rows}x${shape.inner}x${shape.columns}:` +
      (hasBias ? "bias" : "no-bias");
    const existing = this.compiled.get(key);
    if (existing !== undefined) return existing;
    const created = compileAceGemm(
      this.device,
      this.modelProfile,
      this.weightLayout,
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

export function planAceGemm(shape: AceGemmShape): AceGemmPlan {
  const { rows, inner, columns } = shape;
  for (const [name, value] of Object.entries({ rows, inner, columns })) {
    if (!Number.isSafeInteger(value) || value <= 0) {
      throw new RangeError(`ACE GEMM ${name} must be a positive safe integer`);
    }
  }
  const workgroupsX = Math.ceil(columns / SCALAR_WORKGROUP_COLUMNS);
  const workgroupsY = Math.ceil(rows / SCALAR_WORKGROUP_ROWS);
  if (
    workgroupsX > MAX_DISPATCH_DIMENSION ||
    workgroupsY > MAX_DISPATCH_DIMENSION
  ) {
    throw new RangeError("ACE GEMM exceeds the portable 2D dispatch domain");
  }
  const activationElements = checkedProduct(rows, inner, "activation");
  const weightElements = checkedProduct(columns, inner, "weight");
  const outputElements = checkedProduct(rows, columns, "output");
  return Object.freeze({
    rows,
    inner,
    columns,
    workgroupsX,
    workgroupsY,
    activationElements,
    weightElements,
    outputElements,
  });
}

export function planAceTiledGemm(shape: AceGemmShape): AceTiledGemmPlan {
  const scalar = planAceGemm(shape);
  const rowTiles = Math.ceil(scalar.rows / TILED_ROWS);
  const columnTiles = Math.ceil(scalar.columns / TILED_COLUMNS);
  const outputsPerWorkgroup = TILED_ROWS * TILED_COLUMNS;
  const workgroupCount = rowTiles * columnTiles;
  const multiplyAddsPerWorkgroup = checkedProduct(
    outputsPerWorkgroup,
    scalar.inner,
    "workgroup work",
  );
  if (multiplyAddsPerWorkgroup > ACE_GEMM_MAX_MULTIPLY_ADDS_PER_RANGE) {
    throw new RangeError(
      "ACE tiled GEMM inner dimension exceeds one cooperative output tile",
    );
  }
  const workgroupsPerRange = Math.min(
    MAX_DISPATCH_DIMENSION,
    Math.floor(ACE_GEMM_MAX_OUTPUTS_PER_RANGE / outputsPerWorkgroup),
    Math.floor(
      ACE_GEMM_MAX_MULTIPLY_ADDS_PER_RANGE / multiplyAddsPerWorkgroup,
    ),
  );
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
    const firstOutput = emittedOutputs;
    const outputCount = rangeOutputCount(
      scalar.rows,
      scalar.columns,
      columnTiles,
      firstWorkgroup,
      rangeWorkgroups,
    );
    outputRanges.push(Object.freeze({
      firstOutput,
      outputCount,
      firstWorkgroup,
      workgroupCount: rangeWorkgroups,
      // Tail lanes still execute the K loop with zero-padded staged inputs.
      multiplyAdds: checkedProduct(
        rangeWorkgroups,
        multiplyAddsPerWorkgroup,
        "range work",
      ),
    }));
    emittedOutputs += outputCount;
  }
  if (outputRanges.some(({ workgroupCount: count }) =>
    count > MAX_DISPATCH_DIMENSION
  )) {
    throw new RangeError("ACE tiled GEMM exceeds the portable dispatch domain");
  }
  return Object.freeze({
    ...scalar,
    tileRows: TILED_ROWS,
    tileColumns: TILED_COLUMNS,
    tileInner: TILED_INNER,
    workgroupSize: TILED_WORKGROUP_SIZE,
    outputRangeCount: outputRanges.length,
    outputRanges: Object.freeze(outputRanges),
  });
}

/** Map a logical `[N,K]` weight coordinate to its physical scalar offset. */
export function aceGemmWeightScalarIndex(
  weightLayout: AceGemmWeightLayout,
  shape: Pick<AceGemmShape, "inner" | "columns">,
  column: number,
  inner: number,
): number {
  requirePositiveSafeInteger(shape.inner, "inner");
  requirePositiveSafeInteger(shape.columns, "columns");
  requireAceGemmWeightLayoutShape(weightLayout, shape);
  if (
    !Number.isSafeInteger(column) ||
    column < 0 ||
    column >= shape.columns
  ) {
    throw new RangeError(
      `ACE GEMM weight column ${String(column)} is outside [0, ${shape.columns})`,
    );
  }
  if (!Number.isSafeInteger(inner) || inner < 0 || inner >= shape.inner) {
    throw new RangeError(
      `ACE GEMM weight inner ${String(inner)} is outside [0, ${shape.inner})`,
    );
  }
  checkedProduct(shape.columns, shape.inner, "weight");
  if (weightLayout === "source-row-major") {
    return column * shape.inner + inner;
  }
  const innerTiles = shape.inner / DIT_GEMM_WEIGHT_TILE_INNER;
  const physicalScalar = (
    ((
      Math.floor(column / DIT_GEMM_WEIGHT_TILE_COLUMNS) * innerTiles +
      Math.floor(inner / DIT_GEMM_WEIGHT_TILE_INNER)
    ) * DIT_GEMM_WEIGHT_TILE_INNER +
      inner % DIT_GEMM_WEIGHT_TILE_INNER) *
      DIT_GEMM_WEIGHT_TILE_COLUMNS +
    column % DIT_GEMM_WEIGHT_TILE_COLUMNS
  );
  if (!Number.isSafeInteger(physicalScalar)) {
    throw new RangeError("ACE GEMM physical weight index is not a safe integer");
  }
  return physicalScalar;
}

export function aceCorrectnessGemmWgsl(
  modelProfile: AceModelProfileId,
  shape: AceGemmShape,
  hasBias: boolean,
  weightLayout: AceGemmWeightLayout = "source-row-major",
): string {
  const plan = planAceTiledGemm(shape);
  requireAceGemmWeightLayoutShape(weightLayout, plan);
  if (modelProfile === "reference-bf16") {
    return referenceBf16Wgsl(plan, hasBias, weightLayout);
  }
  if (modelProfile === "raw-fp16") {
    return rawFp16Wgsl(plan, hasBias, weightLayout);
  }
  throw new TypeError(`Unknown ACE GEMM model profile ${String(modelProfile)}`);
}

/**
 * Diagnostic-only pre-tiling shader used to prove the Stage-1 tiled kernel did
 * not change the profile's per-output contraction sequence. Production graph
 * construction never instantiates this scalar oracle.
 */
export function aceScalarGemmOracleWgsl(
  modelProfile: AceModelProfileId,
  shape: AceGemmShape,
  hasBias: boolean,
): string {
  const plan = planAceGemm(shape);
  if (modelProfile === "reference-bf16") {
    return scalarReferenceBf16Wgsl(plan, hasBias);
  }
  if (modelProfile === "raw-fp16") {
    return scalarRawFp16Wgsl(plan, hasBias);
  }
  throw new TypeError(`Unknown ACE scalar GEMM model profile ${String(modelProfile)}`);
}

async function compileAceGemm(
  device: GPUDevice,
  modelProfile: AceModelProfileId,
  weightLayout: AceGemmWeightLayout,
  shape: AceGemmShape,
  hasBias: boolean,
): Promise<CompiledGemm> {
  const label =
    `ace-correctness-gemm-${modelProfile}-${weightLayout}-` +
    `${shape.rows}x${shape.inner}x${shape.columns}` +
    (hasBias ? "-bias" : "-no-bias");
  const module = device.createShaderModule({
    label,
    code: aceCorrectnessGemmWgsl(
      modelProfile,
      shape,
      hasBias,
      weightLayout,
    ),
  });
  const pipeline = await device.createComputePipelineAsync({
    label,
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });
  const ranges = planAceTiledGemm(shape).outputRanges;
  const allocated = await createAceScopedBuffers(
    device,
    [{
      label: `${label}-output-range-parameters`,
      size: Math.max(
        STORAGE_UNIFORM_ALIGNMENT,
        ranges.length * STORAGE_UNIFORM_ALIGNMENT,
      ),
      usage: GPUBufferUsage.UNIFORM,
      mappedAtCreation: true,
    }],
    `${label} output range parameters`,
  );
  const outputRangeParameters = allocated[0];
  if (outputRangeParameters === undefined) {
    throw new Error(`${label} output range parameter allocation returned no buffer`);
  }
  try {
    const mapped = outputRangeParameters.getMappedRange();
    for (let index = 0; index < ranges.length; index += 1) {
      new Uint32Array(
        mapped,
        index * STORAGE_UNIFORM_ALIGNMENT,
        OUTPUT_RANGE_PARAMETER_BYTES / Uint32Array.BYTES_PER_ELEMENT,
      )[0] = ranges[index]!.firstWorkgroup;
    }
    outputRangeParameters.unmap();
    return Object.freeze({
      pipeline,
      bindGroupLayout: pipeline.getBindGroupLayout(0),
      outputRangeParameters,
      destroy(): void {
        outputRangeParameters.destroy();
      },
    });
  } catch (error) {
    outputRangeParameters.destroy();
    throw error;
  }
}

function referenceBf16Wgsl(
  shape: AceGemmPlan,
  hasBias: boolean,
  weightLayout: AceGemmWeightLayout,
): string {
  const biasDeclaration = hasBias
    ? "@group(0) @binding(3) var<storage, read> bias: array<u32>;"
    : "";
  const biasLoader = hasBias
    ? `fn load_bias(index: u32) -> f32 {
  return decode_bf16(bias[index >> 1u], index);
}`
    : "";
  const biasAdd = hasBias ? "sum = sum + load_bias(column);" : "";
  const outputRangeBinding = hasBias ? 4 : 3;
  const weightIndex = gemmWeightIndexWgsl(weightLayout);
  return /* wgsl */ `
// ACE GEMM weight layout: ${weightLayout}
const ROWS: u32 = ${shape.rows}u;
const INNER: u32 = ${shape.inner}u;
const COLUMNS: u32 = ${shape.columns}u;

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

fn decode_bf16(pair: u32, index: u32) -> f32 {
  let bits16 = select(pair >> 16u, pair & 0xffffu, (index & 1u) == 0u);
  return bitcast<f32>(bits16 << 16u);
}

fn load_weight(index: u32) -> f32 {
  return decode_bf16(weight[index >> 1u], index);
}

${weightIndex}

${biasLoader}

var<workgroup> activation_tile: array<f32, ${TILED_A_VALUES}>;
var<workgroup> weight_tile: array<f32, ${TILED_B_VALUES}>;

@compute @workgroup_size(${TILED_WORKGROUP_SIZE}, 1, 1)
fn main(
  @builtin(local_invocation_index) lane: u32,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  let linear_group = output_range.first_workgroup + group.x;
  let groups_per_row_tile = (COLUMNS + ${TILED_COLUMNS - 1}u) / ${TILED_COLUMNS}u;
  let row_base = (linear_group / groups_per_row_tile) * ${TILED_ROWS}u;
  let column_base = (linear_group % groups_per_row_tile) * ${TILED_COLUMNS}u;
  let local_column_vector = lane % ${TILED_COLUMN_VECTORS}u;
  let local_row_group = lane / ${TILED_COLUMN_VECTORS}u;
  let local_row_base = local_row_group;
  let column = column_base + local_column_vector * 4u;

  var acc0 = vec4<f32>(0.0);
  var acc1 = vec4<f32>(0.0);
  var acc2 = vec4<f32>(0.0);
  var acc3 = vec4<f32>(0.0);

  for (var k_base = 0u; k_base < INNER; k_base += ${TILED_INNER}u) {
    for (var item = lane; item < ${TILED_A_VALUES}u; item += ${TILED_WORKGROUP_SIZE}u) {
      let tile_row = item / ${TILED_INNER}u;
      let tile_inner = item % ${TILED_INNER}u;
      let source_row = row_base + tile_row;
      let source_inner = k_base + tile_inner;
      if (source_row < ROWS && source_inner < INNER) {
        activation_tile[item] = activation[source_row * INNER + source_inner];
      } else {
        activation_tile[item] = 0.0;
      }
    }
    for (var item = lane; item < ${TILED_B_VALUES}u; item += ${TILED_WORKGROUP_SIZE}u) {
      let tile_column = item / ${TILED_INNER}u;
      let tile_inner = item % ${TILED_INNER}u;
      let source_inner = k_base + tile_inner;
      let source_column = column_base + tile_column;
      if (source_column < COLUMNS && source_inner < INNER) {
        weight_tile[tile_inner * ${TILED_COLUMNS}u + tile_column] =
          load_weight(weight_scalar_index(source_column, source_inner));
      } else {
        weight_tile[tile_inner * ${TILED_COLUMNS}u + tile_column] = 0.0;
      }
    }
    workgroupBarrier();

    for (var tile_inner = 0u; tile_inner < ${TILED_INNER}u; tile_inner += 1u) {
      let b_base = tile_inner * ${TILED_COLUMNS}u + local_column_vector * 4u;
      let b = vec4<f32>(
        weight_tile[b_base],
        weight_tile[b_base + 1u],
        weight_tile[b_base + 2u],
        weight_tile[b_base + 3u],
      );
      let a_index = local_row_base * ${TILED_INNER}u + tile_inner;
      acc0 = acc0 + vec4<f32>(activation_tile[a_index]) * b;
      acc1 = acc1 + vec4<f32>(activation_tile[a_index + ${TILED_ROW_GROUPS * TILED_INNER}u]) * b;
      acc2 = acc2 + vec4<f32>(activation_tile[a_index + ${2 * TILED_ROW_GROUPS * TILED_INNER}u]) * b;
      acc3 = acc3 + vec4<f32>(activation_tile[a_index + ${3 * TILED_ROW_GROUPS * TILED_INNER}u]) * b;
    }
    workgroupBarrier();
  }

  for (var owned_row = 0u; owned_row < ${TILED_ROWS_PER_LANE}u; owned_row += 1u) {
    let row = row_base + local_row_base + owned_row * ${TILED_ROW_GROUPS}u;
    var value = select(select(acc0, acc1, owned_row == 1u), select(acc2, acc3, owned_row == 3u), owned_row >= 2u);
    if (row < ROWS) {
      for (var component = 0u; component < 4u; component += 1u) {
        let output_column = column + component;
        if (output_column < COLUMNS) {
          var sum = value[component];
          ${biasAdd.replaceAll("column", "output_column")}
          output[row * COLUMNS + output_column] = sum;
        }
      }
    }
  }
}
`;
}

function rawFp16Wgsl(
  shape: AceGemmPlan,
  hasBias: boolean,
  weightLayout: AceGemmWeightLayout,
): string {
  const biasDeclaration = hasBias
    ? "@group(0) @binding(3) var<storage, read> bias: array<f16>;"
    : "";
  const biasAdd = hasBias ? "sum = sum + bias[column];" : "";
  const outputRangeBinding = hasBias ? 4 : 3;
  const weightIndex = gemmWeightIndexWgsl(weightLayout);
  return /* wgsl */ `
enable f16;

// ACE GEMM weight layout: ${weightLayout}
const ROWS: u32 = ${shape.rows}u;
const INNER: u32 = ${shape.inner}u;
const COLUMNS: u32 = ${shape.columns}u;

@group(0) @binding(0) var<storage, read> activation: array<f16>;
@group(0) @binding(1) var<storage, read> weight: array<f16>;
@group(0) @binding(2) var<storage, read_write> output: array<f16>;
${biasDeclaration}

struct OutputRangeParameters {
  first_workgroup: u32,
  _padding0: u32,
  _padding1: u32,
  _padding2: u32,
}
@group(0) @binding(${outputRangeBinding}) var<uniform>
  output_range: OutputRangeParameters;

${weightIndex}

var<workgroup> activation_tile: array<f16, ${TILED_A_VALUES}>;
var<workgroup> weight_tile: array<f16, ${TILED_B_VALUES}>;

@compute @workgroup_size(${TILED_WORKGROUP_SIZE}, 1, 1)
fn main(
  @builtin(local_invocation_index) lane: u32,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  let linear_group = output_range.first_workgroup + group.x;
  let groups_per_row_tile = (COLUMNS + ${TILED_COLUMNS - 1}u) / ${TILED_COLUMNS}u;
  let row_base = (linear_group / groups_per_row_tile) * ${TILED_ROWS}u;
  let column_base = (linear_group % groups_per_row_tile) * ${TILED_COLUMNS}u;
  let local_column_vector = lane % ${TILED_COLUMN_VECTORS}u;
  let local_row_group = lane / ${TILED_COLUMN_VECTORS}u;
  let local_row_base = local_row_group;
  let column = column_base + local_column_vector * 4u;

  var acc0 = vec4<f16>(0.0h);
  var acc1 = vec4<f16>(0.0h);
  var acc2 = vec4<f16>(0.0h);
  var acc3 = vec4<f16>(0.0h);

  for (var k_base = 0u; k_base < INNER; k_base += ${TILED_INNER}u) {
    for (var item = lane; item < ${TILED_A_VALUES}u; item += ${TILED_WORKGROUP_SIZE}u) {
      let tile_row = item / ${TILED_INNER}u;
      let tile_inner = item % ${TILED_INNER}u;
      let source_row = row_base + tile_row;
      let source_inner = k_base + tile_inner;
      if (source_row < ROWS && source_inner < INNER) {
        activation_tile[item] = activation[source_row * INNER + source_inner];
      } else {
        activation_tile[item] = 0.0h;
      }
    }
    for (var item = lane; item < ${TILED_B_VALUES}u; item += ${TILED_WORKGROUP_SIZE}u) {
      let tile_column = item / ${TILED_INNER}u;
      let tile_inner = item % ${TILED_INNER}u;
      let source_inner = k_base + tile_inner;
      let source_column = column_base + tile_column;
      if (source_column < COLUMNS && source_inner < INNER) {
        weight_tile[tile_inner * ${TILED_COLUMNS}u + tile_column] =
          weight[weight_scalar_index(source_column, source_inner)];
      } else {
        weight_tile[tile_inner * ${TILED_COLUMNS}u + tile_column] = 0.0h;
      }
    }
    workgroupBarrier();

    for (var tile_inner = 0u; tile_inner < ${TILED_INNER}u; tile_inner += 1u) {
      let b_base = tile_inner * ${TILED_COLUMNS}u + local_column_vector * 4u;
      let b = vec4<f16>(
        weight_tile[b_base],
        weight_tile[b_base + 1u],
        weight_tile[b_base + 2u],
        weight_tile[b_base + 3u],
      );
      let a_index = local_row_base * ${TILED_INNER}u + tile_inner;
      acc0 = acc0 + vec4<f16>(activation_tile[a_index]) * b;
      acc1 = acc1 + vec4<f16>(activation_tile[a_index + ${TILED_ROW_GROUPS * TILED_INNER}u]) * b;
      acc2 = acc2 + vec4<f16>(activation_tile[a_index + ${2 * TILED_ROW_GROUPS * TILED_INNER}u]) * b;
      acc3 = acc3 + vec4<f16>(activation_tile[a_index + ${3 * TILED_ROW_GROUPS * TILED_INNER}u]) * b;
    }
    workgroupBarrier();
  }

  for (var owned_row = 0u; owned_row < ${TILED_ROWS_PER_LANE}u; owned_row += 1u) {
    let row = row_base + local_row_base + owned_row * ${TILED_ROW_GROUPS}u;
    var value = select(select(acc0, acc1, owned_row == 1u), select(acc2, acc3, owned_row == 3u), owned_row >= 2u);
    if (row < ROWS) {
      for (var component = 0u; component < 4u; component += 1u) {
        let output_column = column + component;
        if (output_column < COLUMNS) {
          var sum = value[component];
          ${biasAdd.replaceAll("column", "output_column")}
          output[row * COLUMNS + output_column] = sum;
        }
      }
    }
  }
}
`;
}

function scalarReferenceBf16Wgsl(
  shape: AceGemmPlan,
  hasBias: boolean,
): string {
  const biasDeclaration = hasBias
    ? "@group(0) @binding(3) var<storage, read> bias: array<u32>;"
    : "";
  const biasLoader = hasBias
    ? `fn load_bias(index: u32) -> f32 {
  return decode_bf16(bias[index >> 1u], index);
}`
    : "";
  const biasAdd = hasBias ? "sum = sum + load_bias(column);" : "";
  return /* wgsl */ `
const ROWS: u32 = ${shape.rows}u;
const INNER: u32 = ${shape.inner}u;
const COLUMNS: u32 = ${shape.columns}u;

@group(0) @binding(0) var<storage, read> activation: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
${biasDeclaration}

fn decode_bf16(pair: u32, index: u32) -> f32 {
  let bits16 = select(pair >> 16u, pair & 0xffffu, (index & 1u) == 0u);
  return bitcast<f32>(bits16 << 16u);
}

fn load_weight(index: u32) -> f32 {
  return decode_bf16(weight[index >> 1u], index);
}

${biasLoader}

@compute @workgroup_size(${SCALAR_WORKGROUP_COLUMNS}, ${SCALAR_WORKGROUP_ROWS}, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let column = global_id.x;
  let row = global_id.y;
  if (row >= ROWS || column >= COLUMNS) { return; }

  var sum = 0.0;
  let activation_base = row * INNER;
  let weight_base = column * INNER;
  for (var inner = 0u; inner < INNER; inner += 1u) {
    sum = sum + activation[activation_base + inner] *
      load_weight(weight_base + inner);
  }
  ${biasAdd}
  output[row * COLUMNS + column] = sum;
}
`;
}

function scalarRawFp16Wgsl(
  shape: AceGemmPlan,
  hasBias: boolean,
): string {
  const biasDeclaration = hasBias
    ? "@group(0) @binding(3) var<storage, read> bias: array<f16>;"
    : "";
  const biasAdd = hasBias ? "sum = sum + bias[column];" : "";
  return /* wgsl */ `
enable f16;

const ROWS: u32 = ${shape.rows}u;
const INNER: u32 = ${shape.inner}u;
const COLUMNS: u32 = ${shape.columns}u;

@group(0) @binding(0) var<storage, read> activation: array<f16>;
@group(0) @binding(1) var<storage, read> weight: array<f16>;
@group(0) @binding(2) var<storage, read_write> output: array<f16>;
${biasDeclaration}

@compute @workgroup_size(${SCALAR_WORKGROUP_COLUMNS}, ${SCALAR_WORKGROUP_ROWS}, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let column = global_id.x;
  let row = global_id.y;
  if (row >= ROWS || column >= COLUMNS) { return; }

  var sum = 0.0h;
  let activation_base = row * INNER;
  let weight_base = column * INNER;
  for (var inner = 0u; inner < INNER; inner += 1u) {
    sum = sum + activation[activation_base + inner] *
      weight[weight_base + inner];
  }
  ${biasAdd}
  output[row * COLUMNS + column] = sum;
}
`;
}

function storageBytes(
  modelProfile: AceModelProfileId,
  logicalElements: number,
): number {
  if (modelProfile === "raw-fp16") return logicalElements * 2;
  return Math.ceil(logicalElements / 2) * Uint32Array.BYTES_PER_ELEMENT;
}

function activationBytes(
  modelProfile: AceModelProfileId,
  logicalElements: number,
): number {
  return logicalElements * (modelProfile === "raw-fp16" ? 2 : 4);
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
    throw new RangeError(
      `${label} binding exposes ${String(available)} bytes; ${required} required`,
    );
  }
}

function exactBindingRange(
  binding: GPUBufferBinding,
  requiredBytes: number,
): GPUBufferBinding {
  return {
    buffer: binding.buffer,
    offset: binding.offset ?? 0,
    size: requiredBytes,
  };
}

function requireAceGemmWeightLayout(
  weightLayout: AceGemmWeightLayout,
): void {
  if (
    weightLayout !== "source-row-major" &&
    weightLayout !== ACE_DIT_GEMM_WEIGHT_LAYOUT
  ) {
    throw new TypeError(
      `Unknown ACE GEMM weight layout ${String(weightLayout)}`,
    );
  }
}

function requireAceGemmWeightLayoutShape(
  weightLayout: AceGemmWeightLayout,
  shape: Pick<AceGemmShape, "inner" | "columns">,
): void {
  requireAceGemmWeightLayout(weightLayout);
  if (weightLayout === "source-row-major") return;
  if (shape.columns % DIT_GEMM_WEIGHT_TILE_COLUMNS !== 0) {
    throw new RangeError(
      `ACE ${weightLayout} GEMM requires N divisible by ` +
        DIT_GEMM_WEIGHT_TILE_COLUMNS,
    );
  }
  if (shape.inner % DIT_GEMM_WEIGHT_TILE_INNER !== 0) {
    throw new RangeError(
      `ACE ${weightLayout} GEMM requires K divisible by ` +
        DIT_GEMM_WEIGHT_TILE_INNER,
    );
  }
}

function gemmWeightIndexWgsl(weightLayout: AceGemmWeightLayout): string {
  requireAceGemmWeightLayout(weightLayout);
  if (weightLayout === "source-row-major") {
    return /* wgsl */ `fn weight_scalar_index(column: u32, inner: u32) -> u32 {
  return column * INNER + inner;
}`;
  }
  return /* wgsl */ `fn weight_scalar_index(column: u32, inner: u32) -> u32 {
  return ((((column / 128u) * (INNER / 32u) + (inner / 32u)) * 32u +
    (inner % 32u)) * 128u + (column % 128u));
}`;
}

function requirePositiveSafeInteger(value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new RangeError(`ACE GEMM ${label} must be a positive safe integer`);
  }
}

function checkedProduct(left: number, right: number, label: string): number {
  const product = left * right;
  if (!Number.isSafeInteger(product)) {
    throw new RangeError(`ACE GEMM ${label} element count is not a safe integer`);
  }
  return product;
}

function rangeOutputCount(
  rows: number,
  columns: number,
  columnTiles: number,
  firstWorkgroup: number,
  workgroupCount: number,
): number {
  let outputs = 0;
  const end = firstWorkgroup + workgroupCount;
  for (let workgroup = firstWorkgroup; workgroup < end; workgroup += 1) {
    const rowTile = Math.floor(workgroup / columnTiles);
    const columnTile = workgroup % columnTiles;
    const activeRows = Math.min(TILED_ROWS, rows - rowTile * TILED_ROWS);
    const activeColumns = Math.min(
      TILED_COLUMNS,
      columns - columnTile * TILED_COLUMNS,
    );
    outputs += activeRows * activeColumns;
  }
  return outputs;
}
