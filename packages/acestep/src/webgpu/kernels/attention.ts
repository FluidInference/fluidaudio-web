import type { AceModelProfileId } from "../capabilities.js";
import { createAceScopedBuffers } from "../scoped-buffer-allocation.js";
import { requireAceDisjointOutput } from "./correctness-utils.js";
import { ACE_GEMM_MAX_MULTIPLY_ADDS_PER_RANGE } from "./gemm.js";

const ATTENTION_LANES = 128;
export const ACE_FIXED32_ATTENTION_SUBGROUP_SIZE = 32;
export const ACE_FIXED32_ATTENTION_WORKGROUP_SIZE = 256;
export const ACE_FIXED32_ATTENTION_QUERY_TOKENS_PER_TILE = 4;
export const ACE_FIXED32_ATTENTION_QUERIES_PER_WORKGROUP = 8;
/** Generalized query8 is measured only at the direct-target M2250 shape. */
export const ACE_FIXED32_ATTENTION_GENERALIZED_MIN_QUERY_TOKENS = 2_250;
export const ACE_FIXED32_ATTENTION_WORKGROUP_BYTES =
  2 * ATTENTION_LANES * Float32Array.BYTES_PER_ELEMENT;
const RANGE_PARAMETER_BYTES = 16;
const MINIMUM_UNIFORM_STRIDE = 256;
const MAX_DISPATCH_DIMENSION = 65_535;
const MAX_U32 = 0xffff_ffff;

export type AceAttentionMode = "full" | "sliding" | "causal";
export type AceAttentionKeyValidity = "none" | "causal-per-key";

export interface AceAttentionShape {
  readonly batch: number;
  readonly queryHeads: number;
  readonly keyValueHeads: number;
  readonly queryTokens: number;
  readonly keyValueTokens: number;
  readonly headDimension: number;
  readonly mode: AceAttentionMode;
  /** Required only for sliding attention; inclusive distance in tokens. */
  readonly slidingRadius?: number;
  /** Absolute cache position of local query token zero; causal mode only. */
  readonly queryPositionOffset?: number;
  /**
   * Planner-only arbitrary key masking. When enabled, `queryPositionOffset`
   * is forbidden and physical query positions come from a required binding.
   * Full/sliding DiT attention deliberately cannot opt into this mode.
   */
  readonly keyValidity?: AceAttentionKeyValidity;
}

export interface AceAttentionPlan extends AceAttentionShape {
  readonly keyValidity: AceAttentionKeyValidity;
  readonly queryHeadsPerKeyValueHead: number;
  readonly queryElements: number;
  readonly keyValueElements: number;
  readonly keyValidityElements: number;
  readonly queryPositionElements: number;
  readonly outputElements: number;
  readonly workgroupsX: number;
  readonly workgroupsY: number;
}

export interface AceFixed32AttentionCapability {
  readonly subgroupMinSize?: number;
  readonly subgroupMaxSize?: number;
}

export type AceAttentionRuntimeConfiguration =
  | Readonly<{ backend: "portable" }>
  | Readonly<{
      backend: "fixed32-subgroup-query8";
      capability: AceFixed32AttentionCapability;
    }>
  | Readonly<{
      backend: "opt-0062-fixed32-quad-query32-full-self";
      runtimeProfileId: "opt-0062-fixed32-quad-query32-full-self-v1";
      capability: AceFixed32AttentionCapability;
    }>
  | Readonly<{
      backend: "opt-0070-fixed32-quad-query32-full-self-production";
      runtimeProfileId:
        "opt-0070-fixed32-quad-query32-full-self-production-v1";
      capability: AceFixed32AttentionCapability;
      /** Exact graph-plan self-attention token count. */
      expectedQueryTokens: number;
      /** Exact graph-plan conditioning token count. */
      expectedConditionTokens: number;
    }>;

export interface AceAttentionOutputRange {
  readonly firstOutput: number;
  readonly outputCount: number;
  readonly firstWorkgroup: number;
  readonly workgroupCount: number;
  /** Maximum QK multiply-adds; consumed by the shared cooperative scheduler. */
  readonly multiplyAdds: number;
}

/**
 * Query8 plan. The historical name is retained because the original optimized
 * shape was square full attention; the same tile now also covers production
 * non-square full and sliding attention.
 */
export interface AceFixed32TiledFullAttentionPlan extends AceAttentionPlan {
  readonly backend: "fixed32-subgroup-query8";
  readonly subgroupSize: typeof ACE_FIXED32_ATTENTION_SUBGROUP_SIZE;
  readonly workgroupSize: typeof ACE_FIXED32_ATTENTION_WORKGROUP_SIZE;
  readonly workgroupStorageBytes: typeof ACE_FIXED32_ATTENTION_WORKGROUP_BYTES;
  readonly queryTokensPerTile: typeof ACE_FIXED32_ATTENTION_QUERY_TOKENS_PER_TILE;
  readonly queriesPerWorkgroup: typeof ACE_FIXED32_ATTENTION_QUERIES_PER_WORKGROUP;
  readonly queryTokenTiles: number;
  readonly workgroupCount: number;
  readonly outputRangeCount: number;
  readonly outputRanges: readonly AceAttentionOutputRange[];
  readonly portableKeyValueScalarLoads: number;
  readonly tiledKeyValueScalarLoads: number;
  readonly keyValueLoadReduction: number;
  readonly portableBarriersPerKey: 10;
  readonly tiledBarriersPerKey: 2;
}

export interface AceAttentionBindings {
  /** `[batch, queryHeads, queryTokens, headDimension]`. */
  readonly query: GPUBufferBinding;
  /** `[batch, keyValueHeads, keyValueTokens, headDimension]`. */
  readonly key: GPUBufferBinding;
  readonly value: GPUBufferBinding;
  /** U32 `[batch, 2]`: valid query length followed by valid key length. */
  readonly validLengths: GPUBufferBinding;
  readonly output: GPUBufferBinding;
  /**
   * U32 `[batch, keyValueTokens]`; exactly `1` admits a key. Required only
   * with `keyValidity: "causal-per-key"` and rejected otherwise.
   */
  readonly keyValidity?: GPUBufferBinding;
  /**
   * U32 `[batch, queryTokens]` physical cache positions. Required together
   * with `keyValidity` so left padding does not renumber causal positions.
   */
  readonly queryPositions?: GPUBufferBinding;
  /**
   * OPT-0062 correctness-only oracle. The production graph omits this object;
   * when present, the quad owner runs query8 into `oracleOutput` and records an
   * exhaustive raw-U32 comparison in the route's counter slice.
   */
  readonly opt0062Identity?: Readonly<{
    readonly oracleOutput: GPUBufferBinding;
    readonly counters: GPUBufferBinding;
    readonly routeIndex: number;
  }>;
}

export interface AcePortableAttentionDispatch {
  readonly label: string;
  readonly backend: "portable";
  readonly plan: AceAttentionPlan;
  encode(pass: GPUComputePassEncoder): void;
}

export interface AceFixed32TiledFullAttentionDispatch {
  readonly label: string;
  readonly backend: "fixed32-subgroup-query8";
  readonly plan: AceFixed32TiledFullAttentionPlan;
  readonly rangeCount: number;
  encodeRange(pass: GPUComputePassEncoder, rangeIndex: number): void;
  encode(pass: GPUComputePassEncoder): void;
}

export interface AceOpt0062QuadQueryAttentionDispatch {
  readonly label: string;
  readonly backend: "opt-0062-fixed32-quad-query32-full-self";
  readonly kernelId: "opt-0062-fixed32-quad-query32-full-self-v1";
  readonly plan: Readonly<{
    readonly outputRanges: readonly AceAttentionOutputRange[];
    readonly outputRangeCount: number;
    readonly workgroupCount: number;
    readonly outputElements: number;
  }>;
  readonly rangeCount: number;
  encodeRange(pass: GPUComputePassEncoder, rangeIndex: number): void;
  encode(pass: GPUComputePassEncoder): void;
}

export type AceAttentionDispatch =
  | AcePortableAttentionDispatch
  | AceFixed32TiledFullAttentionDispatch
  | AceOpt0062QuadQueryAttentionDispatch;

interface CompiledFixed32TiledFullAttention {
  readonly pipeline: GPUComputePipeline;
  readonly bindGroupLayout: GPUBindGroupLayout;
  readonly rangeParameters: GPUBuffer;
  readonly rangeParameterStride: number;
  destroy(): void;
}

/**
 * Score-buffer-free GQA attention with FP32 online-softmax state.
 *
 * A 128-lane workgroup owns one query/head. Each key is reduced across the
 * head dimension and immediately incorporated into the online weighted sum,
 * so memory is O(output), never O(tokens squared).
 */
export class AceCorrectnessAttentionKernel {
  private readonly compiledPortable = new Map<
    string,
    Promise<GPUComputePipeline>
  >();
  private readonly compiledFixed32 = new Map<
    string,
    Promise<CompiledFixed32TiledFullAttention>
  >();
  private destroyed = false;

  private constructor(
    private readonly device: GPUDevice,
    readonly modelProfile: AceModelProfileId,
    readonly configuration: AceAttentionRuntimeConfiguration,
  ) {}

  static create(
    device: GPUDevice,
    modelProfile: AceModelProfileId,
    configuration: AceAttentionRuntimeConfiguration = { backend: "portable" },
  ): AceCorrectnessAttentionKernel {
    if (modelProfile === "raw-fp16" && !device.features.has("shader-f16")) {
      throw new Error("ACE raw-FP16 attention requires WebGPU shader-f16");
    }
    if (modelProfile !== "reference-bf16" && modelProfile !== "raw-fp16") {
      throw new TypeError(`Unknown ACE attention model profile ${String(modelProfile)}`);
    }
    if (
      device.limits.maxComputeInvocationsPerWorkgroup < ATTENTION_LANES ||
      device.limits.maxComputeWorkgroupSizeX < ATTENTION_LANES ||
      device.limits.maxComputeWorkgroupStorageSize <
        (ATTENTION_LANES + 4) * Float32Array.BYTES_PER_ELEMENT
    ) {
      throw new Error("ACE attention requires 128 lanes and 528 bytes of workgroup storage");
    }
    if (configuration.backend === "fixed32-subgroup-query8") {
      if (
        modelProfile !== "reference-bf16" ||
        !device.features.has("subgroups") ||
        configuration.capability.subgroupMinSize !==
          ACE_FIXED32_ATTENTION_SUBGROUP_SIZE ||
        configuration.capability.subgroupMaxSize !==
          ACE_FIXED32_ATTENTION_SUBGROUP_SIZE
      ) {
        throw new Error(
          "ACE fixed-32 tiled attention requires reference-bf16 and reported fixed 32-lane subgroups",
        );
      }
      if (
        device.limits.maxComputeInvocationsPerWorkgroup <
          ACE_FIXED32_ATTENTION_WORKGROUP_SIZE ||
        device.limits.maxComputeWorkgroupSizeX <
          ACE_FIXED32_ATTENTION_WORKGROUP_SIZE ||
        device.limits.maxComputeWorkgroupStorageSize <
          ACE_FIXED32_ATTENTION_WORKGROUP_BYTES
      ) {
        throw new Error(
          `ACE fixed-32 tiled attention requires WG${ACE_FIXED32_ATTENTION_WORKGROUP_SIZE} and ` +
            `${ACE_FIXED32_ATTENTION_WORKGROUP_BYTES} bytes of workgroup storage`,
        );
      }
    } else if (configuration.backend !== "portable") {
      throw new TypeError(
        `Unknown ACE attention backend ${String((configuration as { backend?: unknown }).backend)}`,
      );
    }
    return new AceCorrectnessAttentionKernel(
      device,
      modelProfile,
      Object.freeze(configuration),
    );
  }

  async createDispatch(
    label: string,
    shape: AceAttentionShape,
    bindings: AceAttentionBindings,
  ): Promise<AceAttentionDispatch> {
    if (this.destroyed) throw new Error("ACE attention kernel was destroyed");
    const plan = planAceAttention(shape);
    const elementBytes = this.modelProfile === "raw-fp16" ? 2 : 4;
    requireBindingBytes(bindings.query, plan.queryElements * elementBytes, `${label} query`);
    requireBindingBytes(bindings.key, plan.keyValueElements * elementBytes, `${label} key`);
    requireBindingBytes(bindings.value, plan.keyValueElements * elementBytes, `${label} value`);
    requireBindingBytes(bindings.output, plan.outputElements * elementBytes, `${label} output`);
    requireBindingBytes(bindings.validLengths, plan.batch * 2 * 4, `${label} lengths`);
    if (plan.keyValidity === "causal-per-key") {
      if (bindings.keyValidity === undefined || bindings.queryPositions === undefined) {
        throw new TypeError(
          `${label} causal-per-key attention requires keyValidity and queryPositions bindings`,
        );
      }
      requireBindingBytes(
        bindings.keyValidity,
        plan.keyValidityElements * 4,
        `${label} key validity`,
      );
      requireBindingBytes(
        bindings.queryPositions,
        plan.queryPositionElements * 4,
        `${label} query positions`,
      );
    } else if (bindings.keyValidity !== undefined || bindings.queryPositions !== undefined) {
      throw new TypeError(
        `${label} attention mask bindings require causal-per-key attention`,
      );
    }
    requireAceDisjointOutput(
      bindings.output,
      [
        bindings.query,
        bindings.key,
        bindings.value,
        bindings.validLengths,
        ...(bindings.keyValidity === undefined ? [] : [bindings.keyValidity]),
        ...(bindings.queryPositions === undefined
          ? []
          : [bindings.queryPositions]),
      ],
      label,
    );
    if (
      this.configuration.backend === "fixed32-subgroup-query8" &&
      isAceFixed32TiledFullAttentionShape(plan)
    ) {
      return await this.createFixed32Dispatch(label, plan, bindings);
    }
    const pipeline = await this.portablePipelineFor(plan);
    if (this.destroyed) {
      throw new Error("ACE attention kernel was destroyed while compiling");
    }
    const entries: GPUBindGroupEntry[] = [
      { binding: 0, resource: bindings.query },
      { binding: 1, resource: bindings.key },
      { binding: 2, resource: bindings.value },
      { binding: 3, resource: bindings.validLengths },
      { binding: 4, resource: bindings.output },
    ];
    if (plan.keyValidity === "causal-per-key") {
      entries.push(
        { binding: 5, resource: bindings.keyValidity! },
        { binding: 6, resource: bindings.queryPositions! },
      );
    }
    const bindGroup = this.device.createBindGroup({
      label: `${label}-bindings`,
      layout: pipeline.getBindGroupLayout(0),
      entries,
    });
    return Object.freeze({
      label,
      backend: "portable" as const,
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
    for (const compiled of this.compiledFixed32.values()) {
      void compiled.then(
        (resources) => resources.destroy(),
        () => undefined,
      );
    }
    this.compiledPortable.clear();
    this.compiledFixed32.clear();
  }

  private async createFixed32Dispatch(
    label: string,
    shape: AceAttentionShape,
    bindings: AceAttentionBindings,
  ): Promise<AceFixed32TiledFullAttentionDispatch> {
    const plan = planAceFixed32TiledFullAttention(shape);
    const compiled = await this.fixed32PipelineFor(plan);
    if (this.destroyed) {
      throw new Error("ACE attention kernel was destroyed while compiling");
    }
    const commonEntries: GPUBindGroupEntry[] = [
      { binding: 0, resource: bindings.query },
      { binding: 1, resource: bindings.key },
      { binding: 2, resource: bindings.value },
      { binding: 3, resource: bindings.validLengths },
      { binding: 4, resource: bindings.output },
    ];
    const bindGroups = plan.outputRanges.map((_, rangeIndex) =>
      this.device.createBindGroup({
        label: `${label}-fixed32-query8-range-${rangeIndex}-bindings`,
        layout: compiled.bindGroupLayout,
        entries: [
          ...commonEntries,
          {
            binding: 7,
            resource: {
              buffer: compiled.rangeParameters,
              offset: rangeIndex * compiled.rangeParameterStride,
              size: RANGE_PARAMETER_BYTES,
            },
          },
        ],
      })
    );
    return Object.freeze({
      label,
      backend: "fixed32-subgroup-query8" as const,
      plan,
      rangeCount: plan.outputRangeCount,
      encodeRange(pass: GPUComputePassEncoder, rangeIndex: number): void {
        const range = plan.outputRanges[rangeIndex];
        if (range === undefined) {
          throw new RangeError(
            `${label} attention range ${rangeIndex} is outside [0, ${plan.outputRangeCount})`,
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

  private portablePipelineFor(plan: AceAttentionPlan): Promise<GPUComputePipeline> {
    const key = [
      plan.batch,
      plan.queryHeads,
      plan.keyValueHeads,
      plan.queryTokens,
      plan.keyValueTokens,
      plan.headDimension,
      plan.mode,
      plan.slidingRadius ?? "none",
      plan.queryPositionOffset ?? "none",
      plan.keyValidity,
    ].join("x");
    const existing = this.compiledPortable.get(key);
    if (existing !== undefined) return existing;
    const label = `ace-correctness-attention-${this.modelProfile}-${key}`;
    const module = this.device.createShaderModule({
      label,
      code: aceCorrectnessAttentionWgsl(this.modelProfile, plan),
    });
    const created = this.device.createComputePipelineAsync({
      label,
      layout: "auto",
      compute: { module, entryPoint: "main" },
    });
    this.compiledPortable.set(key, created);
    void created.catch(() => {
      if (this.compiledPortable.get(key) === created) {
        this.compiledPortable.delete(key);
      }
    });
    return created;
  }

  private fixed32PipelineFor(
    plan: AceFixed32TiledFullAttentionPlan,
  ): Promise<CompiledFixed32TiledFullAttention> {
    const key = [
      plan.batch,
      plan.queryHeads,
      plan.keyValueHeads,
      plan.queryTokens,
      plan.keyValueTokens,
      plan.headDimension,
      plan.mode,
      plan.slidingRadius ?? "none",
    ].join("x");
    const existing = this.compiledFixed32.get(key);
    if (existing !== undefined) return existing;
    const created = compileAceFixed32TiledFullAttention(this.device, plan);
    this.compiledFixed32.set(key, created);
    void created.catch(() => {
      if (this.compiledFixed32.get(key) === created) {
        this.compiledFixed32.delete(key);
      }
    });
    return created;
  }
}

export function planAceAttention(shape: AceAttentionShape): AceAttentionPlan {
  const {
    batch,
    queryHeads,
    keyValueHeads,
    queryTokens,
    keyValueTokens,
    headDimension,
    mode,
    slidingRadius,
    queryPositionOffset,
    keyValidity,
  } = shape;
  for (const [name, value] of Object.entries({
    batch,
    queryHeads,
    keyValueHeads,
    queryTokens,
    keyValueTokens,
    headDimension,
  })) {
    if (!Number.isSafeInteger(value) || value <= 0 || value > MAX_U32) {
      throw new RangeError(`ACE attention ${name} must be a positive safe integer`);
    }
  }
  if (headDimension > ATTENTION_LANES) {
    throw new RangeError(`ACE attention headDimension cannot exceed ${ATTENTION_LANES}`);
  }
  if (queryHeads % keyValueHeads !== 0) {
    throw new RangeError("ACE attention query heads must be divisible by KV heads");
  }
  if (mode !== "full" && mode !== "sliding" && mode !== "causal") {
    throw new TypeError(`Unknown ACE attention mode ${String(mode)}`);
  }
  const normalizedKeyValidity = keyValidity ?? "none";
  if (normalizedKeyValidity !== "none" && normalizedKeyValidity !== "causal-per-key") {
    throw new TypeError(`Unknown ACE attention key-validity mode ${String(keyValidity)}`);
  }
  if (normalizedKeyValidity === "causal-per-key" && mode !== "causal") {
    throw new RangeError("ACE causal-per-key validity is available only in causal mode");
  }
  if (
    (mode === "sliding" &&
      (!Number.isSafeInteger(slidingRadius) ||
        slidingRadius === undefined ||
        slidingRadius < 0 ||
        slidingRadius > MAX_U32)) ||
    (mode !== "sliding" && slidingRadius !== undefined)
  ) {
    throw new RangeError("ACE slidingRadius must be a non-negative integer only in sliding mode");
  }
  if (normalizedKeyValidity === "causal-per-key") {
    if (queryPositionOffset !== undefined) {
      throw new RangeError(
        "ACE causal-per-key attention takes bound physical positions, not queryPositionOffset",
      );
    }
  } else if (
    (mode === "causal" &&
      (!Number.isSafeInteger(queryPositionOffset) ||
        queryPositionOffset === undefined ||
        queryPositionOffset < 0 ||
        queryPositionOffset + queryTokens > keyValueTokens)) ||
    (mode !== "causal" && queryPositionOffset !== undefined)
  ) {
    throw new RangeError(
      "ACE queryPositionOffset must place every local causal query inside the KV cache",
    );
  }
  const queryElements = checkedProduct(
    checkedProduct(checkedProduct(batch, queryHeads), queryTokens),
    headDimension,
  );
  const keyValueElements = checkedProduct(
    checkedProduct(checkedProduct(batch, keyValueHeads), keyValueTokens),
    headDimension,
  );
  const keyValidityElements = normalizedKeyValidity === "causal-per-key"
    ? checkedProduct(batch, keyValueTokens)
    : 0;
  const queryPositionElements = normalizedKeyValidity === "causal-per-key"
    ? checkedProduct(batch, queryTokens)
    : 0;
  if (
    queryElements > MAX_U32 ||
    keyValueElements > MAX_U32 ||
    keyValidityElements > MAX_U32 ||
    queryPositionElements > MAX_U32
  ) {
    throw new RangeError("ACE attention tensor indexing exceeds the U32 shader domain");
  }
  const totalQueries = checkedProduct(checkedProduct(batch, queryHeads), queryTokens);
  const workgroupsX = Math.min(totalQueries, MAX_DISPATCH_DIMENSION);
  const workgroupsY = Math.ceil(totalQueries / workgroupsX);
  if (workgroupsY > MAX_DISPATCH_DIMENSION) {
    throw new RangeError("ACE attention exceeds the portable 2D dispatch domain");
  }
  return Object.freeze({
    batch,
    queryHeads,
    keyValueHeads,
    queryTokens,
    keyValueTokens,
    headDimension,
    mode,
    ...(slidingRadius === undefined ? {} : { slidingRadius }),
    ...(queryPositionOffset === undefined ? {} : { queryPositionOffset }),
    keyValidity: normalizedKeyValidity,
    queryHeadsPerKeyValueHead: queryHeads / keyValueHeads,
    queryElements,
    keyValueElements,
    keyValidityElements,
    queryPositionElements,
    outputElements: queryElements,
    workgroupsX,
    workgroupsY,
  });
}

export function isAceFixed32TiledFullAttentionShape(
  shape: AceAttentionShape,
): boolean {
  const plan = planAceAttention(shape);
  const originalSquareFullShape =
    plan.mode === "full" && plan.queryTokens === plan.keyValueTokens;
  const longNonSquareFullShape =
    plan.mode === "full" &&
    plan.queryTokens !== plan.keyValueTokens &&
    plan.queryTokens >= ACE_FIXED32_ATTENTION_GENERALIZED_MIN_QUERY_TOKENS;
  const longSquareSlidingShape =
    plan.mode === "sliding" &&
    plan.queryTokens === plan.keyValueTokens &&
    plan.queryTokens >= ACE_FIXED32_ATTENTION_GENERALIZED_MIN_QUERY_TOKENS &&
    plan.slidingRadius! <= MAX_U32 - plan.queryTokens;
  return (
    (originalSquareFullShape ||
      longNonSquareFullShape ||
      longSquareSlidingShape) &&
    plan.keyValidity === "none" &&
    plan.queryHeadsPerKeyValueHead === 2 &&
    plan.headDimension === ATTENTION_LANES
  );
}

export function planAceFixed32TiledFullAttention(
  shape: AceAttentionShape,
): AceFixed32TiledFullAttentionPlan {
  const portable = planAceAttention(shape);
  if (!isAceFixed32TiledFullAttentionShape(portable)) {
    throw new RangeError(
      "ACE fixed-32 tiled attention requires square unmasked full GQA2, or at least 2250 query tokens for unmasked non-square full or square sliding GQA2, with head dimension 128",
    );
  }
  const queryTokenTiles = Math.ceil(
    portable.queryTokens / ACE_FIXED32_ATTENTION_QUERY_TOKENS_PER_TILE,
  );
  const workgroupCount = checkedProduct(
    checkedProduct(portable.batch, portable.keyValueHeads),
    queryTokenTiles,
  );
  const maximumKeysPerTile = portable.mode === "sliding"
    ? Math.min(
        portable.keyValueTokens,
        2 * portable.slidingRadius! +
          ACE_FIXED32_ATTENTION_QUERY_TOKENS_PER_TILE,
      )
    : portable.keyValueTokens;
  const maximumMultiplyAddsPerWorkgroup = checkedProduct(
    checkedProduct(
      ACE_FIXED32_ATTENTION_QUERIES_PER_WORKGROUP,
      maximumKeysPerTile,
    ),
    portable.headDimension,
  );
  if (maximumMultiplyAddsPerWorkgroup > ACE_GEMM_MAX_MULTIPLY_ADDS_PER_RANGE) {
    throw new RangeError(
      "ACE fixed-32 attention one-workgroup QK work exceeds the cooperative range budget",
    );
  }
  const workgroupsPerRange = Math.max(1, Math.min(
    MAX_DISPATCH_DIMENSION,
    Math.floor(
      ACE_GEMM_MAX_MULTIPLY_ADDS_PER_RANGE /
        maximumMultiplyAddsPerWorkgroup,
    ),
  ));
  const outputRanges: AceAttentionOutputRange[] = [];
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
    const queryTokens =
      tiledQueryTokensBeforeWorkgroup(
        firstWorkgroup + rangeWorkgroups,
        queryTokenTiles,
        portable.queryTokens,
      ) -
      tiledQueryTokensBeforeWorkgroup(
        firstWorkgroup,
        queryTokenTiles,
        portable.queryTokens,
      );
    const outputCount = checkedProduct(
      checkedProduct(queryTokens, portable.queryHeadsPerKeyValueHead),
      portable.headDimension,
    );
    const multiplyAdds = checkedProduct(outputCount, maximumKeysPerTile);
    outputRanges.push(Object.freeze({
      firstOutput: emittedOutputs,
      outputCount,
      firstWorkgroup,
      workgroupCount: rangeWorkgroups,
      multiplyAdds,
    }));
    emittedOutputs += outputCount;
  }
  if (emittedOutputs !== portable.outputElements) {
    throw new Error("ACE fixed-32 attention range accounting lost output elements");
  }
  const portableKeyVisits = portable.mode === "sliding"
    ? slidingQueryKeyVisits(
        portable.queryTokens,
        portable.keyValueTokens,
        portable.slidingRadius!,
      )
    : checkedProduct(portable.queryTokens, portable.keyValueTokens);
  const tiledKeyVisits = portable.mode === "sliding"
    ? slidingTileUnionKeyVisits(
        portable.queryTokens,
        portable.keyValueTokens,
        portable.slidingRadius!,
      )
    : checkedProduct(queryTokenTiles, portable.keyValueTokens);
  const portableKeyValueScalarLoads = checkedProduct(
    checkedProduct(
      checkedProduct(portable.batch, portable.queryHeads),
      portableKeyVisits,
    ),
    portable.headDimension * 2,
  );
  const tiledKeyValueScalarLoads = checkedProduct(
    checkedProduct(
      checkedProduct(portable.batch, portable.keyValueHeads),
      tiledKeyVisits,
    ),
    portable.headDimension * 2,
  );
  return Object.freeze({
    ...portable,
    backend: "fixed32-subgroup-query8" as const,
    subgroupSize: ACE_FIXED32_ATTENTION_SUBGROUP_SIZE,
    workgroupSize: ACE_FIXED32_ATTENTION_WORKGROUP_SIZE,
    workgroupStorageBytes: ACE_FIXED32_ATTENTION_WORKGROUP_BYTES,
    queryTokensPerTile: ACE_FIXED32_ATTENTION_QUERY_TOKENS_PER_TILE,
    queriesPerWorkgroup: ACE_FIXED32_ATTENTION_QUERIES_PER_WORKGROUP,
    queryTokenTiles,
    workgroupCount,
    outputRangeCount: outputRanges.length,
    outputRanges: Object.freeze(outputRanges),
    portableKeyValueScalarLoads,
    tiledKeyValueScalarLoads,
    keyValueLoadReduction:
      portableKeyValueScalarLoads / tiledKeyValueScalarLoads,
    portableBarriersPerKey: 10 as const,
    tiledBarriersPerKey: 2 as const,
  });
}

export function aceCorrectnessAttentionWgsl(
  modelProfile: AceModelProfileId,
  shape: AceAttentionShape,
): string {
  const plan = planAceAttention(shape);
  if (modelProfile !== "reference-bf16" && modelProfile !== "raw-fp16") {
    throw new TypeError(`Unknown ACE attention model profile ${String(modelProfile)}`);
  }
  const f16 = modelProfile === "raw-fp16";
  const scale = 1 / Math.sqrt(plan.headDimension);
  const keyBounds = plan.mode === "sliding"
    ? `
  let key_start = select(0u, query_token - SLIDING_RADIUS, query_token > SLIDING_RADIUS);
  let key_end = min(valid_key_tokens, query_token + SLIDING_RADIUS + 1u);`
    : plan.mode === "causal" && plan.keyValidity === "causal-per-key"
      ? `
  let physical_query_position = query_positions[batch * QUERY_TOKENS + query_token];
  let physical_query_is_valid = physical_query_position < valid_key_tokens;
  let key_start = 0u;
  let key_end = select(
    0u,
    min(valid_key_tokens, physical_query_position + 1u),
    physical_query_is_valid,
  );`
      : plan.mode === "causal"
      ? `
  let key_start = 0u;
  let key_end = min(valid_key_tokens, QUERY_POSITION_OFFSET + query_token + 1u);`
      : `
  let key_start = 0u;
  let key_end = valid_key_tokens;`;
  return /* wgsl */ `${f16 ? "enable f16;" : ""}
const QUERY_HEADS: u32 = ${plan.queryHeads}u;
const KV_HEADS: u32 = ${plan.keyValueHeads}u;
const QUERY_TOKENS: u32 = ${plan.queryTokens}u;
const KV_TOKENS: u32 = ${plan.keyValueTokens}u;
const HEAD_DIMENSION: u32 = ${plan.headDimension}u;
const HEADS_PER_KV: u32 = ${plan.queryHeadsPerKeyValueHead}u;
const DISPATCH_X: u32 = ${plan.workgroupsX}u;
const TOTAL_QUERIES: u32 = ${plan.batch * plan.queryHeads * plan.queryTokens}u;
const ATTENTION_SCALE: f32 = ${scale};
const SLIDING_RADIUS: u32 = ${plan.slidingRadius ?? 0}u;
const QUERY_POSITION_OFFSET: u32 = ${plan.queryPositionOffset ?? 0}u;

@group(0) @binding(0) var<storage, read> query: array<${f16 ? "f16" : "f32"}>;
@group(0) @binding(1) var<storage, read> key: array<${f16 ? "f16" : "f32"}>;
@group(0) @binding(2) var<storage, read> value: array<${f16 ? "f16" : "f32"}>;
@group(0) @binding(3) var<storage, read> valid_lengths: array<u32>;
@group(0) @binding(4) var<storage, read_write> output: array<${f16 ? "f16" : "f32"}>;
${plan.keyValidity === "causal-per-key" ? `@group(0) @binding(5) var<storage, read> key_validity: array<u32>;
@group(0) @binding(6) var<storage, read> query_positions: array<u32>;` : ""}

var<workgroup> dot_partial: array<f32, ${ATTENTION_LANES}>;
var<workgroup> online_max: f32;
var<workgroup> online_denominator: f32;
var<workgroup> online_alpha: f32;
var<workgroup> online_beta: f32;

@compute @workgroup_size(${ATTENTION_LANES}, 1, 1)
fn main(
  @builtin(workgroup_id) workgroup_id: vec3<u32>,
  @builtin(local_invocation_index) lane: u32,
) {
  let flat_query = workgroup_id.y * DISPATCH_X + workgroup_id.x;
  if (flat_query >= TOTAL_QUERIES) { return; }
  let queries_per_batch = QUERY_HEADS * QUERY_TOKENS;
  let batch = flat_query / queries_per_batch;
  let within_batch = flat_query % queries_per_batch;
  let query_head = within_batch / QUERY_TOKENS;
  let query_token = within_batch % QUERY_TOKENS;
  let valid_query_tokens = min(valid_lengths[batch * 2u], QUERY_TOKENS);
  let valid_key_tokens = min(valid_lengths[batch * 2u + 1u], KV_TOKENS);
  let output_index = flat_query * HEAD_DIMENSION + lane;

  if (query_token >= valid_query_tokens) {
    if (lane < HEAD_DIMENSION) { output[output_index] = ${f16 ? "0.0h" : "0.0"}; }
    return;
  }
${keyBounds}
  if (key_start >= key_end) {
    if (lane < HEAD_DIMENSION) { output[output_index] = ${f16 ? "0.0h" : "0.0"}; }
    return;
  }

  let kv_head = query_head / HEADS_PER_KV;
  let query_base = flat_query * HEAD_DIMENSION;
  let kv_batch_base = (batch * KV_HEADS + kv_head) * KV_TOKENS * HEAD_DIMENSION;
  var query_value = 0.0;
  if (lane < HEAD_DIMENSION) {
    query_value = f32(query[query_base + lane]);
  }
  var weighted_value = 0.0;
  if (lane == 0u) {
    // Literal divide-by-zero is rejected as an abstract-float constant by
    // WGSL validators. This finite f32 sentinel is far below any finite
    // QK-RMSNorm score (bounded by head dimension before scaling).
    online_max = -3.4028234663852886e38;
    online_denominator = 0.0;
  }
  workgroupBarrier();

  for (var key_token = key_start; key_token < key_end; key_token += 1u) {
    let kv_index = kv_batch_base + key_token * HEAD_DIMENSION + lane;
    dot_partial[lane] = 0.0;
    if (lane < HEAD_DIMENSION) {
      dot_partial[lane] = query_value * f32(key[kv_index]);
    }
    workgroupBarrier();
    var stride = ${ATTENTION_LANES / 2}u;
    while (stride > 0u) {
      if (lane < stride) {
        dot_partial[lane] = dot_partial[lane] + dot_partial[lane + stride];
      }
      workgroupBarrier();
      stride = stride >> 1u;
    }
    if (lane == 0u) {
      online_alpha = 1.0;
      online_beta = 0.0;
      let key_is_valid = ${plan.keyValidity === "causal-per-key"
        ? "key_validity[batch * KV_TOKENS + key_token] == 1u"
        : "true"};
      if (key_is_valid) {
        let score = dot_partial[0] * ATTENTION_SCALE;
        let new_max = max(online_max, score);
        online_alpha = exp(online_max - new_max);
        online_beta = exp(score - new_max);
        online_denominator = online_denominator * online_alpha + online_beta;
        online_max = new_max;
      }
    }
    workgroupBarrier();
    if (lane < HEAD_DIMENSION) {
      weighted_value = weighted_value * online_alpha +
        online_beta * f32(value[kv_index]);
    }
    workgroupBarrier();
  }
  if (lane < HEAD_DIMENSION) {
    if (online_denominator > 0.0) {
      output[output_index] = ${f16
        ? "f16(weighted_value / online_denominator)"
        : "weighted_value / online_denominator"};
    } else {
      output[output_index] = ${f16 ? "0.0h" : "0.0"};
    }
  }
}
`;
}

/**
 * Fixed-32 subgroup query8 attention for the long DiT path.
 *
 * One 256-lane workgroup owns four adjacent query tokens for both query heads
 * that share a KV head. Lanes 0..127 stage each K/V row once; eight subgroups
 * then consume that tile for eight independent FP32 online-softmax streams.
 * Full attention admits every staged key. Sliding attention stages the union
 * of the four local windows and admits a key only in the subgroups whose query
 * window contains it. Every query therefore retains ascending key order and
 * the same online update expression as the portable oracle. The QK reduction
 * tree is subgroup-native, so promotion requires numerical and listening
 * gates rather than portable bit identity.
 */
export function aceFixed32TiledFullAttentionWgsl(
  shape: AceAttentionShape,
): string {
  const plan = planAceFixed32TiledFullAttention(shape);
  const scale = 1 / Math.sqrt(plan.headDimension);
  const tileKeyBounds = plan.mode === "sliding"
    ? `
  let tile_first_query = query_tile * QUERY_TOKENS_PER_TILE;
  let tile_last_query = min(
    valid_query_tokens - 1u,
    tile_first_query + QUERY_TOKENS_PER_TILE - 1u,
  );
  let tile_key_start = select(
    0u,
    tile_first_query - SLIDING_RADIUS,
    tile_first_query > SLIDING_RADIUS,
  );
  let tile_key_end = min(
    valid_key_tokens,
    tile_last_query + SLIDING_RADIUS + 1u,
  );
  let query_key_start = select(
    0u,
    query_token - SLIDING_RADIUS,
    query_token > SLIDING_RADIUS,
  );
  let query_key_end = min(
    valid_key_tokens,
    query_token + SLIDING_RADIUS + 1u,
  );`
    : "";
  const slidingRadiusConstant = plan.mode === "sliding"
    ? `const SLIDING_RADIUS: u32 = ${plan.slidingRadius!}u;\n`
    : "";
  const keyLoopHeader = plan.mode === "sliding"
    ? `for (
    var key_token = tile_key_start;
    key_token < tile_key_end;
    key_token += 1u
  ) {`
    : "for (var key_token = 0u; key_token < valid_key_tokens; key_token += 1u) {";
  const updateGuardStart = plan.mode === "sliding"
    ? "    if (query_is_valid && key_token >= query_key_start && key_token < query_key_end) {\n"
    : "";
  const updateGuardEnd = plan.mode === "sliding" ? "    }\n" : "";
  return /* wgsl */ `
enable subgroups;

const QUERY_HEADS: u32 = ${plan.queryHeads}u;
const KV_HEADS: u32 = ${plan.keyValueHeads}u;
const QUERY_TOKENS: u32 = ${plan.queryTokens}u;
const KV_TOKENS: u32 = ${plan.keyValueTokens}u;
const HEAD_DIMENSION: u32 = ${plan.headDimension}u;
const HEADS_PER_KV: u32 = ${plan.queryHeadsPerKeyValueHead}u;
const QUERY_TOKENS_PER_TILE: u32 = ${plan.queryTokensPerTile}u;
const QUERY_TOKEN_TILES: u32 = ${plan.queryTokenTiles}u;
const TOTAL_WORKGROUPS: u32 = ${plan.workgroupCount}u;
const ATTENTION_SCALE: f32 = ${scale};
${slidingRadiusConstant}

@group(0) @binding(0) var<storage, read> query: array<f32>;
@group(0) @binding(1) var<storage, read> key: array<f32>;
@group(0) @binding(2) var<storage, read> value: array<f32>;
@group(0) @binding(3) var<storage, read> valid_lengths: array<u32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

struct AttentionRangeParameters {
  first_workgroup: u32,
  _padding0: u32,
  _padding1: u32,
  _padding2: u32,
}
@group(0) @binding(7) var<uniform>
  attention_range: AttentionRangeParameters;

var<workgroup> key_tile: array<f32, ${ATTENTION_LANES}>;
var<workgroup> value_tile: array<f32, ${ATTENTION_LANES}>;

@compute @workgroup_size(${ACE_FIXED32_ATTENTION_WORKGROUP_SIZE}, 1, 1)
fn main(
  @builtin(local_invocation_index) local_lane: u32,
  @builtin(subgroup_invocation_id) subgroup_lane: u32,
  @builtin(subgroup_id) subgroup: u32,
  @builtin(subgroup_size) subgroup_size: u32,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  if (subgroup_size != ${ACE_FIXED32_ATTENTION_SUBGROUP_SIZE}u) {
    return;
  }
  let linear_group = attention_range.first_workgroup + group.x;
  if (linear_group >= TOTAL_WORKGROUPS) {
    return;
  }
  let groups_per_batch = KV_HEADS * QUERY_TOKEN_TILES;
  let batch = linear_group / groups_per_batch;
  let within_batch = linear_group % groups_per_batch;
  let kv_head = within_batch / QUERY_TOKEN_TILES;
  let query_tile = within_batch % QUERY_TOKEN_TILES;
  let query_head =
    kv_head * HEADS_PER_KV + subgroup / QUERY_TOKENS_PER_TILE;
  let query_token =
    query_tile * QUERY_TOKENS_PER_TILE + subgroup % QUERY_TOKENS_PER_TILE;
  let valid_query_tokens = min(valid_lengths[batch * 2u], QUERY_TOKENS);
  let valid_key_tokens = min(valid_lengths[batch * 2u + 1u], KV_TOKENS);
  let query_is_in_tensor = query_token < QUERY_TOKENS;
  let query_is_valid = query_token < valid_query_tokens;
  let query_base =
    ((batch * QUERY_HEADS + query_head) * QUERY_TOKENS + query_token) *
    HEAD_DIMENSION;

  var query_values: array<f32, 4>;
  var weighted_values: array<f32, 4>;
  for (var chunk = 0u; chunk < 4u; chunk += 1u) {
    let dimension = subgroup_lane + chunk * ${ACE_FIXED32_ATTENTION_SUBGROUP_SIZE}u;
    query_values[chunk] = 0.0;
    weighted_values[chunk] = 0.0;
    if (query_is_valid) {
      query_values[chunk] = query[query_base + dimension];
    }
  }

  if (query_tile * QUERY_TOKENS_PER_TILE >= valid_query_tokens ||
      valid_key_tokens == 0u) {
    if (query_is_in_tensor) {
      for (var chunk = 0u; chunk < 4u; chunk += 1u) {
        let dimension = subgroup_lane + chunk * ${ACE_FIXED32_ATTENTION_SUBGROUP_SIZE}u;
        output[query_base + dimension] = 0.0;
      }
    }
    return;
  }
${tileKeyBounds}

  let kv_batch_base =
    (batch * KV_HEADS + kv_head) * KV_TOKENS * HEAD_DIMENSION;
  var online_max = -3.4028234663852886e38;
  var online_denominator = 0.0;

  ${keyLoopHeader}
    if (local_lane < HEAD_DIMENSION) {
      let kv_index = kv_batch_base + key_token * HEAD_DIMENSION + local_lane;
      key_tile[local_lane] = key[kv_index];
      value_tile[local_lane] = value[kv_index];
    }
    workgroupBarrier();

    var dot_partial = 0.0;
    for (var chunk = 0u; chunk < 4u; chunk += 1u) {
      let dimension = subgroup_lane + chunk * ${ACE_FIXED32_ATTENTION_SUBGROUP_SIZE}u;
      dot_partial = dot_partial + query_values[chunk] * key_tile[dimension];
    }
    let score = subgroupAdd(dot_partial) * ATTENTION_SCALE;
${updateGuardStart}
      let new_max = max(online_max, score);
      let online_alpha = exp(online_max - new_max);
      let online_beta = exp(score - new_max);
      online_denominator =
        online_denominator * online_alpha + online_beta;
      online_max = new_max;
      for (var chunk = 0u; chunk < 4u; chunk += 1u) {
        let dimension = subgroup_lane + chunk * ${ACE_FIXED32_ATTENTION_SUBGROUP_SIZE}u;
        weighted_values[chunk] = weighted_values[chunk] * online_alpha +
          online_beta * value_tile[dimension];
      }
${updateGuardEnd}
    workgroupBarrier();
  }

  if (query_is_in_tensor) {
    for (var chunk = 0u; chunk < 4u; chunk += 1u) {
      let dimension = subgroup_lane + chunk * ${ACE_FIXED32_ATTENTION_SUBGROUP_SIZE}u;
      output[query_base + dimension] = select(
        0.0,
        weighted_values[chunk] / online_denominator,
        query_is_valid && online_denominator > 0.0,
      );
    }
  }
}
`;
}

async function compileAceFixed32TiledFullAttention(
  device: GPUDevice,
  plan: AceFixed32TiledFullAttentionPlan,
): Promise<CompiledFixed32TiledFullAttention> {
  const label =
    `ace-fixed32-query8-${plan.mode}-attention-` +
    `${plan.batch}x${plan.queryHeads}x${plan.keyValueHeads}x` +
    `${plan.queryTokens}x${plan.keyValueTokens}x${plan.headDimension}x` +
    `${plan.slidingRadius ?? "none"}`;
  const module = device.createShaderModule({
    label,
    code: aceFixed32TiledFullAttentionWgsl(plan),
  });
  const pipeline = await device.createComputePipelineAsync({
    label,
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });
  const rangeParameterStride = Math.max(
    MINIMUM_UNIFORM_STRIDE,
    device.limits.minUniformBufferOffsetAlignment ?? MINIMUM_UNIFORM_STRIDE,
  );
  const allocated = await createAceScopedBuffers(
    device,
    [{
      label: `${label}-range-parameters`,
      size: Math.max(
        rangeParameterStride,
        plan.outputRangeCount * rangeParameterStride,
      ),
      usage: GPUBufferUsage.UNIFORM,
      mappedAtCreation: true,
    }],
    `${label} range parameters`,
  );
  const rangeParameters = allocated[0];
  if (rangeParameters === undefined) {
    throw new Error(`${label} range parameter allocation returned no buffer`);
  }
  try {
    const mapped = rangeParameters.getMappedRange();
    for (let index = 0; index < plan.outputRanges.length; index += 1) {
      new Uint32Array(
        mapped,
        index * rangeParameterStride,
        RANGE_PARAMETER_BYTES / Uint32Array.BYTES_PER_ELEMENT,
      )[0] = plan.outputRanges[index]!.firstWorkgroup;
    }
    rangeParameters.unmap();
    return Object.freeze({
      pipeline,
      bindGroupLayout: pipeline.getBindGroupLayout(0),
      rangeParameters,
      rangeParameterStride,
      destroy(): void {
        rangeParameters.destroy();
      },
    });
  } catch (error) {
    rangeParameters.destroy();
    throw error;
  }
}

function slidingQueryKeyVisits(
  queryTokens: number,
  keyValueTokens: number,
  slidingRadius: number,
): number {
  let visits = 0;
  for (let queryToken = 0; queryToken < queryTokens; queryToken += 1) {
    const keyStart = Math.max(0, queryToken - slidingRadius);
    const keyEnd = Math.min(
      keyValueTokens,
      queryToken + slidingRadius + 1,
    );
    visits = checkedAdd(
      visits,
      Math.max(0, keyEnd - keyStart),
      "sliding query key visits",
    );
  }
  return visits;
}

function slidingTileUnionKeyVisits(
  queryTokens: number,
  keyValueTokens: number,
  slidingRadius: number,
): number {
  let visits = 0;
  for (
    let firstQuery = 0;
    firstQuery < queryTokens;
    firstQuery += ACE_FIXED32_ATTENTION_QUERY_TOKENS_PER_TILE
  ) {
    const lastQuery = Math.min(
      queryTokens - 1,
      firstQuery + ACE_FIXED32_ATTENTION_QUERY_TOKENS_PER_TILE - 1,
    );
    const keyStart = Math.max(0, firstQuery - slidingRadius);
    const keyEnd = Math.min(
      keyValueTokens,
      lastQuery + slidingRadius + 1,
    );
    visits = checkedAdd(
      visits,
      Math.max(0, keyEnd - keyStart),
      "sliding tile-union key visits",
    );
  }
  return visits;
}

function tiledQueryTokensBeforeWorkgroup(
  workgroup: number,
  queryTokenTiles: number,
  queryTokens: number,
): number {
  const completeHeads = Math.floor(workgroup / queryTokenTiles);
  const partialTiles = workgroup % queryTokenTiles;
  return (
    completeHeads * queryTokens +
    Math.min(
      partialTiles * ACE_FIXED32_ATTENTION_QUERY_TOKENS_PER_TILE,
      queryTokens,
    )
  );
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

function checkedProduct(left: number, right: number): number {
  const product = left * right;
  if (!Number.isSafeInteger(product)) {
    throw new RangeError("ACE attention element count is not a safe integer");
  }
  return product;
}

function checkedAdd(left: number, right: number, label: string): number {
  const sum = left + right;
  if (!Number.isSafeInteger(sum)) {
    throw new RangeError(`ACE attention ${label} is not a safe integer`);
  }
  return sum;
}
