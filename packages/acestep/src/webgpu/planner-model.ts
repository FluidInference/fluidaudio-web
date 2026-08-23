import type { AceGpuLogicalTensor } from "../model/gpu-tensors.js";
import type {
  AcePackageManifest,
  AcePackageTensorRecord,
} from "../model/manifest.js";
import type {
  AcePlannerDecodeBatch,
  AcePlannerLogitRange,
  AcePlannerPrefillBatch,
} from "../runtime/planner.js";
import type { AceModelProfileId } from "./capabilities.js";
import {
  AceCorrectnessEmbeddingKernel,
  type AceEmbeddingDispatch,
  type AceEmbeddingShardBinding,
} from "./kernels/embedding.js";
import {
  aceActivationBytes,
  acePackedWeightBytes,
  checkedAceProduct,
  checkedAceSum,
  requireAceBindingBytes,
  requireAceDisjointOutput,
  requirePositiveSafeInteger,
} from "./kernels/correctness-utils.js";
import {
  aceCompositeCooperativeQuanta,
  acePrimitiveCooperativeQuanta,
  type AceGpuEncodeQuantum,
  type AceCooperativeGemmPlan,
  type AceGemmDispatch,
  planAceTiledGemm,
} from "./kernels/gemm.js";
import {
  AceCorrectnessTensorCopyKernel,
  type AceCopyDispatch,
  type AceGatherRowsPlan,
} from "./kernels/tensor-copy.js";
import {
  ACE_PLANNER_QWEN3_CONFIG,
  AceCorrectnessQwen3Runtime,
  createAceQwen3RopeTables,
  planAceQwen3Block,
  type AceQwen3BlockDispatch,
  type AceQwen3BlockScratch,
  type AceQwen3BlockWeights,
  type AceQwen3Config,
  type AceQwen3RopeTables,
} from "./qwen3.js";
import {
  AceOpt0087PlannerDenseOwner,
  type AceOpt0087PlannerDenseArm,
  type AceOpt0087PlannerDenseRequest,
  type AceOpt0087PlannerDenseSelection,
} from "./planner-dense-owner.js";

const U32_BYTES = Uint32Array.BYTES_PER_ELEMENT;

export const ACE_PLANNER_MODEL_REFERENCE_PROVENANCE = Object.freeze({
  aceSourceRevision: "6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0",
  plannerSnapshotRevision: "148d8ea0225bdab342ee1ae3a354275ccd60ca80",
  qwenImplementation: "transformers-4.57.6-Qwen3ForCausalLM",
  cacheLayout: "batch-kv-head-physical-token-head-dimension",
  logitsSelection: "last-physical-token-per-left-padded-row",
  tiedOutput: "model.embed_tokens.weight-is-lm-head-weight",
} as const);

export const ACE_PLANNER_MODEL_CACHE_CONTRACT = Object.freeze({
  phaseCapacity: "exact-requested-capacity",
  maximumCapacityIsNotAnAllocationDefault: 40_960,
  prefillStartsFresh: true,
  prefillCacheValidityInitialization: "zero-before-first-quantum",
  cacheValidity: "one-shared-physical-key-mask-rewritten-identically-by-each-layer",
  keyValueCache: "one-disjoint-key-and-value-pair-per-layer",
  cooperativeOrder:
    "embedding,28-cache-writing-layers,final-norm,last-row-gather,tied-lm-head",
} as const);

export interface AcePlannerLayerTensorNames {
  readonly inputLayerNorm: string;
  readonly queryProjection: string;
  readonly keyProjection: string;
  readonly valueProjection: string;
  readonly queryNorm: string;
  readonly keyNorm: string;
  readonly outputProjection: string;
  readonly postAttentionLayerNorm: string;
  readonly gateProjection: string;
  readonly upProjection: string;
  readonly downProjection: string;
}

export const ACE_PLANNER_SHARED_TENSOR_NAMES = Object.freeze({
  embedding: "planner.model.embed_tokens.weight",
  finalNorm: "planner.model.norm.weight",
} as const);

export function acePlannerLayerTensorNames(
  layer: number,
): AcePlannerLayerTensorNames {
  if (
    !Number.isSafeInteger(layer) ||
    layer < 0 ||
    layer >= ACE_PLANNER_QWEN3_CONFIG.layerCount
  ) {
    throw new RangeError(
      `ACE planner layer must be in [0, ${ACE_PLANNER_QWEN3_CONFIG.layerCount - 1}]`,
    );
  }
  const prefix = `planner.model.layers.${layer}`;
  return Object.freeze({
    inputLayerNorm: `${prefix}.input_layernorm.weight`,
    queryProjection: `${prefix}.self_attn.q_proj.weight`,
    keyProjection: `${prefix}.self_attn.k_proj.weight`,
    valueProjection: `${prefix}.self_attn.v_proj.weight`,
    queryNorm: `${prefix}.self_attn.q_norm.weight`,
    keyNorm: `${prefix}.self_attn.k_norm.weight`,
    outputProjection: `${prefix}.self_attn.o_proj.weight`,
    postAttentionLayerNorm: `${prefix}.post_attention_layernorm.weight`,
    gateProjection: `${prefix}.mlp.gate_proj.weight`,
    upProjection: `${prefix}.mlp.up_proj.weight`,
    downProjection: `${prefix}.mlp.down_proj.weight`,
  });
}

export const ACE_PLANNER_LAYER_TENSOR_NAMES: readonly AcePlannerLayerTensorNames[] =
  Object.freeze(Array.from(
    { length: ACE_PLANNER_QWEN3_CONFIG.layerCount },
    (_, layer) => acePlannerLayerTensorNames(layer),
  ));

export const ACE_PLANNER_LOGICAL_TENSOR_NAMES: readonly string[] = Object.freeze([
  ACE_PLANNER_SHARED_TENSOR_NAMES.embedding,
  ...ACE_PLANNER_LAYER_TENSOR_NAMES.flatMap((layer) => Object.values(layer)),
  ACE_PLANNER_SHARED_TENSOR_NAMES.finalNorm,
]);

export const ACE_PLANNER_EMBEDDING_ROW_PARTS = Object.freeze([
  Object.freeze({ firstRow: 0, rowCount: 49_152 }),
  Object.freeze({ firstRow: 49_152, rowCount: 49_152 }),
  Object.freeze({ firstRow: 98_304, rowCount: 49_152 }),
  Object.freeze({ firstRow: 147_456, rowCount: 49_152 }),
  Object.freeze({ firstRow: 196_608, rowCount: 20_596 }),
]);

export interface AcePlannerHeadSlice {
  readonly shardIndex: number;
  readonly globalFirstRow: number;
  readonly localFirstRow: number;
  readonly rowCount: number;
}

/** Intersect one ascending logit interval with the physical embedding shards. */
export function planAcePlannerHeadSlices(
  logitRange?: AcePlannerLogitRange,
): readonly AcePlannerHeadSlice[] {
  return planHeadSlices(
    ACE_PLANNER_QWEN3_CONFIG.vocabularySize,
    ACE_PLANNER_EMBEDDING_ROW_PARTS,
    logitRange,
  );
}

function planHeadSlices(
  vocabularySize: number,
  shards: readonly Readonly<{ firstRow: number; rowCount: number }>[],
  logitRange?: AcePlannerLogitRange,
): readonly AcePlannerHeadSlice[] {
  const firstTokenId = logitRange?.firstTokenId ?? 0;
  const tokenCount = logitRange?.tokenCount ?? vocabularySize;
  if (
    !Number.isSafeInteger(firstTokenId) ||
    !Number.isSafeInteger(tokenCount) ||
    firstTokenId < 0 ||
    tokenCount <= 0 ||
    firstTokenId + tokenCount > vocabularySize
  ) {
    throw new RangeError("ACE planner logit range is outside the vocabulary");
  }
  const lastTokenId = firstTokenId + tokenCount;
  const slices: AcePlannerHeadSlice[] = [];
  for (let shardIndex = 0; shardIndex < shards.length; shardIndex += 1) {
    const shard = shards[shardIndex]!;
    const first = Math.max(firstTokenId, shard.firstRow);
    const last = Math.min(lastTokenId, shard.firstRow + shard.rowCount);
    if (last <= first) continue;
    slices.push(Object.freeze({
      shardIndex,
      globalFirstRow: first,
      localFirstRow: first - shard.firstRow,
      rowCount: last - first,
    }));
  }
  const plannedRows = slices.reduce((total, slice) => total + slice.rowCount, 0);
  if (plannedRows !== tokenCount) {
    throw new Error("ACE planner head slices do not cover the requested range");
  }
  return Object.freeze(slices);
}

export const ACE_PLANNER_PHYSICAL_TENSOR_NAMES: readonly string[] = Object.freeze([
  ...ACE_PLANNER_EMBEDDING_ROW_PARTS.map(({ firstRow, rowCount }) =>
    `${ACE_PLANNER_SHARED_TENSOR_NAMES.embedding}.rows-${padRow(firstRow)}-${padRow(firstRow + rowCount)}`
  ),
  ...ACE_PLANNER_LAYER_TENSOR_NAMES.flatMap((layer) => Object.values(layer)),
  ACE_PLANNER_SHARED_TENSOR_NAMES.finalNorm,
]);

export interface AcePlannerTensorResolver {
  logicalTensor(logicalTensor: string): AceGpuLogicalTensor;
  binding(logicalTensor: string): GPUBufferBinding;
}

export interface AcePlannerModelWeights {
  /** The only embedding/LM-head storage; no separate output weight is accepted. */
  readonly embedding: readonly AceEmbeddingShardBinding[];
  readonly layers: readonly AceQwen3BlockWeights[];
  readonly finalNorm: GPUBufferBinding;
}

export type AcePlannerModelBatch = AcePlannerPrefillBatch | AcePlannerDecodeBatch;

export interface AceQwen3PlannerShape {
  readonly kind: "prefill" | "decode";
  readonly batch: 1 | 2;
  readonly tokens: number;
  /** Required explicitly and used verbatim for every layer cache. */
  readonly cacheCapacity: number;
}

export interface AcePlannerModelPlan extends AceQwen3PlannerShape {
  readonly rows: number;
  readonly layerCount: number;
  readonly hiddenElements: number;
  readonly queryElements: number;
  readonly keyValueElements: number;
  readonly intermediateElements: number;
  readonly blockScratchElements: number;
  readonly transientActivationElements: number;
  readonly transientActivationBytes: number;
  readonly logitsElements: number;
  readonly logitsBytes: number;
  readonly kvCacheElementsPerLayer: number;
  readonly kvCacheBytesPerLayer: number;
  readonly kvCacheElements: number;
  readonly kvCacheBytes: number;
  readonly totalActivationElements: number;
  readonly totalActivationBytes: number;
  readonly cacheValidityBytes: number;
  readonly invocationControlBytes: number;
  readonly totalNonWeightBytes: number;
}

export interface AcePlannerModelControlData extends AceQwen3RopeTables {
  readonly validLengths: Uint32Array;
  readonly queryPositions: Uint32Array;
  readonly sourceValidity: Uint32Array;
  readonly rowStartPositions: Uint32Array;
  readonly lastPhysicalRowIndices: Uint32Array;
  readonly clearCacheBeforeDispatch: boolean;
}

export interface AcePlannerLayerCacheBindings {
  readonly key: GPUBufferBinding;
  readonly value: GPUBufferBinding;
}

export interface AcePlannerModelBindings {
  readonly tokenIds: GPUBufferBinding;
  readonly weights: AcePlannerModelWeights;
  readonly controls: Readonly<{
    readonly validLengths: GPUBufferBinding;
    readonly queryPositions: GPUBufferBinding;
    readonly sourceValidity: GPUBufferBinding;
    readonly rowStartPositions: GPUBufferBinding;
    readonly cosine: GPUBufferBinding;
    readonly sine: GPUBufferBinding;
    readonly lastPhysicalRowIndices: GPUBufferBinding;
    /** Shared across layers because physical validity is layer-independent. */
    readonly cacheValidity: GPUBufferBinding;
    /** Shared overwrite target; each CPU-validated layer append writes the same status. */
    readonly writeStatus: GPUBufferBinding;
  }>;
  readonly cache: Readonly<{
    readonly layers: readonly AcePlannerLayerCacheBindings[];
  }>;
  readonly scratch: Readonly<{
    readonly embedded: GPUBufferBinding;
    readonly block: AceQwen3BlockScratch;
    readonly layerOutputs: readonly [GPUBufferBinding, GPUBufferBinding];
    readonly normalizedSequence: GPUBufferBinding;
    readonly lastHiddenRows: GPUBufferBinding;
  }>;
  /**
   * One output per embedding row shard, each laid out `[batch,rowCount]`.
   * Keeping shard outputs independently aligned avoids an invalid unaligned
   * second CFG-row binding while never assembling an LM-head weight.
   */
  readonly logits: readonly GPUBufferBinding[];
}

export type AcePlannerModelQuantumKind =
  | "embedding"
  | "layer"
  | "final-norm"
  | "last-row-gather"
  | "tied-lm-head";

export interface AcePlannerModelQuantum {
  readonly id: string;
  readonly kind: AcePlannerModelQuantumKind;
  readonly layer: number | null;
  readonly primitiveCount: number;
  readonly logicalId?: string;
  encode(pass: GPUComputePassEncoder): void;
}

export interface AcePlannerModelDispatch {
  readonly label: string;
  readonly plan: AcePlannerModelPlan;
  readonly logitRange: AcePlannerLogitRange | null;
  readonly headSlices: readonly AcePlannerHeadSlice[];
  readonly primitiveCount: number;
  readonly quanta: readonly AcePlannerModelQuantum[];
  /** @internal Exact clone-free selector evidence for the OPT-0087 gate. */
  readonly opt0087DenseSelections?: readonly AceOpt0087PlannerDenseSelection[];
  /** @internal Exact physical head slices encoded by each preserved A quantum. */
  readonly opt0087HeadQuantumSliceFirstRows?: readonly (readonly number[])[];
  /** Harness convenience. Production submits and drains each quantum in FIFO order. */
  encode(pass: GPUComputePassEncoder): void;
}

/**
 * Complete cached Qwen3ForCausalLM composer without device or arena ownership.
 * The caller owns authenticated planner weights and exact phase-sized buffers.
 */
export class AceCorrectnessPlannerModelRuntime {
  private readonly embedding: AceCorrectnessEmbeddingKernel;
  private readonly qwen: AceCorrectnessQwen3Runtime;
  private readonly copy: AceCorrectnessTensorCopyKernel;
  private readonly tiedGemm: AceOpt0087PlannerDenseOwner;
  private destroyed = false;

  private constructor(
    readonly modelProfile: AceModelProfileId,
    embedding: AceCorrectnessEmbeddingKernel,
    qwen: AceCorrectnessQwen3Runtime,
    copy: AceCorrectnessTensorCopyKernel,
    tiedGemm: AceOpt0087PlannerDenseOwner,
  ) {
    this.embedding = embedding;
    this.qwen = qwen;
    this.copy = copy;
    this.tiedGemm = tiedGemm;
  }

  static create(
    device: GPUDevice,
    modelProfile: AceModelProfileId,
  ): AceCorrectnessPlannerModelRuntime {
    return AceCorrectnessPlannerModelRuntime.createOwned(
      device,
      modelProfile,
      false,
    );
  }

  /** @internal One authenticated weight owner with both frozen OPT-0087 arms. */
  static createForOpt0087(
    device: GPUDevice,
    modelProfile: AceModelProfileId,
  ): AceCorrectnessPlannerModelRuntime {
    return AceCorrectnessPlannerModelRuntime.createOwned(
      device,
      modelProfile,
      true,
    );
  }

  private static createOwned(
    device: GPUDevice,
    modelProfile: AceModelProfileId,
    pairedOpt0087: boolean,
  ): AceCorrectnessPlannerModelRuntime {
    let embedding: AceCorrectnessEmbeddingKernel | undefined;
    let qwen: AceCorrectnessQwen3Runtime | undefined;
    let copy: AceCorrectnessTensorCopyKernel | undefined;
    let tiedGemm: AceOpt0087PlannerDenseOwner | undefined;
    try {
      embedding = AceCorrectnessEmbeddingKernel.create(device, modelProfile);
      qwen = pairedOpt0087
        ? AceCorrectnessQwen3Runtime.createForOpt0087Planner(
            device,
            modelProfile,
          )
        : AceCorrectnessQwen3Runtime.create(device, modelProfile);
      copy = AceCorrectnessTensorCopyKernel.create(device, modelProfile);
      tiedGemm = pairedOpt0087
        ? AceOpt0087PlannerDenseOwner.createPairedForOpt0087(
            device,
            modelProfile,
          )
        : AceOpt0087PlannerDenseOwner.createGeneric(device, modelProfile);
      return new AceCorrectnessPlannerModelRuntime(
        modelProfile,
        embedding,
        qwen,
        copy,
        tiedGemm,
      );
    } catch (error) {
      embedding?.destroy();
      qwen?.destroy();
      copy?.destroy();
      tiedGemm?.destroy();
      throw error;
    }
  }

  async createPlannerDispatch(
    label: string,
    batch: AcePlannerModelBatch,
    bindings: AcePlannerModelBindings,
    logitRange?: AcePlannerLogitRange,
  ): Promise<AcePlannerModelDispatch> {
    validateAcePlannerModelBatch(batch);
    return this.createQwen3PlannerDispatch(label, ACE_PLANNER_QWEN3_CONFIG, {
      kind: batch.kind,
      batch: batch.rows as 1 | 2,
      tokens: batch.tokens,
      cacheCapacity: batch.cacheCapacity,
    }, bindings, logitRange);
  }

  /** @internal Explicit A/B invocation; ordinary production cannot name an arm. */
  async createPlannerDispatchForOpt0087(
    label: string,
    arm: AceOpt0087PlannerDenseArm,
    batch: AcePlannerModelBatch,
    bindings: AcePlannerModelBindings,
    logitRange?: AcePlannerLogitRange,
  ): Promise<AcePlannerModelDispatch> {
    validateAcePlannerModelBatch(batch);
    return this.createQwen3PlannerDispatchForOpt0087(
      label,
      arm,
      ACE_PLANNER_QWEN3_CONFIG,
      {
        kind: batch.kind,
        batch: batch.rows as 1 | 2,
        tokens: batch.tokens,
        cacheCapacity: batch.cacheCapacity,
      },
      bindings,
      logitRange,
    );
  }

  /** Generic miniature entry point used to validate complete graph wiring. */
  async createQwen3PlannerDispatch(
    label: string,
    config: AceQwen3Config,
    shape: AceQwen3PlannerShape,
    bindings: AcePlannerModelBindings,
    logitRange?: AcePlannerLogitRange,
  ): Promise<AcePlannerModelDispatch> {
    return this.createQwen3PlannerDispatchOwned(
      label,
      config,
      shape,
      bindings,
      logitRange,
    );
  }

  /** @internal Miniature/package harness entry over the same paired owner. */
  async createQwen3PlannerDispatchForOpt0087(
    label: string,
    arm: AceOpt0087PlannerDenseArm,
    config: AceQwen3Config,
    shape: AceQwen3PlannerShape,
    bindings: AcePlannerModelBindings,
    logitRange?: AcePlannerLogitRange,
  ): Promise<AcePlannerModelDispatch> {
    return this.createQwen3PlannerDispatchOwned(
      label,
      config,
      shape,
      bindings,
      logitRange,
      arm,
    );
  }

  private async createQwen3PlannerDispatchOwned(
    label: string,
    config: AceQwen3Config,
    shape: AceQwen3PlannerShape,
    bindings: AcePlannerModelBindings,
    logitRange?: AcePlannerLogitRange,
    opt0087Arm?: AceOpt0087PlannerDenseArm,
  ): Promise<AcePlannerModelDispatch> {
    this.requireLive();
    const plan = planAceQwen3PlannerModel(this.modelProfile, config, shape);
    if (bindings.weights.layers.length !== config.layerCount) {
      throw new RangeError(
        `${label} has ${bindings.weights.layers.length} layer weights; ` +
          `${config.layerCount} required`,
      );
    }
    if (bindings.cache.layers.length !== config.layerCount) {
      throw new RangeError(
        `${label} has ${bindings.cache.layers.length} layer caches; ` +
          `${config.layerCount} required`,
      );
    }
    if (bindings.scratch.layerOutputs.length !== 2) {
      throw new RangeError(`${label} requires two layer-output ping-pong bindings`);
    }
    validatePlannerLogitBindings(
      label,
      this.modelProfile,
      config,
      shape.batch,
      bindings.weights.embedding,
      bindings.scratch.lastHiddenRows,
      bindings.logits,
    );
    const opt0087DenseSelections: AceOpt0087PlannerDenseSelection[] | undefined =
      opt0087Arm === undefined ? undefined : [];
    const captureOpt0087Selection = (
      selection: AceOpt0087PlannerDenseSelection,
    ): void => {
      if (opt0087DenseSelections === undefined) {
        throw new Error("OPT-0087 selection escaped an ordinary planner dispatch");
      }
      opt0087DenseSelections.push(selection);
    };

    const embeddingDispatch = await this.embedding.createDispatch(
      `${label}-embed-tokens`,
      {
        tokenCount: plan.rows,
        width: config.hiddenSize,
        vocabularySize: config.vocabularySize,
      },
      {
        tokenIds: bindings.tokenIds,
        shards: bindings.weights.embedding,
        output: bindings.scratch.embedded,
      },
    );
    const layerDispatches = await Promise.all(
      Array.from({ length: config.layerCount }, async (_, layer) => {
        const input = layer === 0
          ? bindings.scratch.embedded
          : bindings.scratch.layerOutputs[(layer - 1) & 1]!;
        const output = bindings.scratch.layerOutputs[layer & 1]!;
        const cache = bindings.cache.layers[layer]!;
        const blockLabel = `${label}-layer-${layer}`;
        const blockShape = {
          batch: shape.batch,
          tokens: shape.tokens,
          attention: {
            kind: "cached" as const,
            cacheCapacity: shape.cacheCapacity,
          },
        };
        const blockBindings = {
          input,
          output,
          weights: bindings.weights.layers[layer]!,
          scratch: bindings.scratch.block,
          attention: {
            kind: "cached" as const,
            validLengths: bindings.controls.validLengths,
            queryPositions: bindings.controls.queryPositions,
            sourceValidity: bindings.controls.sourceValidity,
            cacheKey: cache.key,
            cacheValue: cache.value,
            cacheValidity: bindings.controls.cacheValidity,
            rowStartPositions: bindings.controls.rowStartPositions,
            writeStatus: bindings.controls.writeStatus,
            cosine: bindings.controls.cosine,
            sine: bindings.controls.sine,
          },
        };
        return opt0087Arm === undefined
          ? this.qwen.createBlockDispatch(
              blockLabel,
              config,
              blockShape,
              blockBindings,
            )
          : this.qwen.createPlannerBlockDispatchForOpt0087(
              blockLabel,
              config,
              blockShape,
              blockBindings,
              {
                arm: opt0087Arm,
                kind: shape.kind,
                onSelection: captureOpt0087Selection,
              },
            );
      }),
    );
    const finalNorm = await this.qwen.createFinalNormDispatch(
      `${label}-final-norm`,
      config,
      plan.rows,
      {
        input: bindings.scratch.layerOutputs[(config.layerCount - 1) & 1]!,
        weight: bindings.weights.finalNorm,
        output: bindings.scratch.normalizedSequence,
      },
    );
    const gather = await this.copy.createGatherRowsDispatch(
      `${label}-last-physical-rows`,
      {
        outer: shape.batch,
        sourceRows: shape.tokens,
        outputRows: 1,
        width: config.hiddenSize,
      },
      {
        input: bindings.scratch.normalizedSequence,
        indices: bindings.controls.lastPhysicalRowIndices,
        output: bindings.scratch.lastHiddenRows,
      },
    );
    const headSlices = planHeadSlices(
      config.vocabularySize,
      bindings.weights.embedding,
      logitRange,
    );
    const headRequests = aceOpt0087PlannerHeadDenseRequests(
      label,
      this.modelProfile,
      config,
      shape,
      bindings,
      headSlices,
    );
    const heads = await Promise.all(headRequests.map(async (request) => {
      if (opt0087Arm === undefined) {
        return this.tiedGemm.createGenericDispatch(
          request.label,
          request.shape,
          request.bindings,
        );
      }
      const { dispatch } = await this.tiedGemm.createDispatchForOpt0087({
        ...request,
        invocation: {
          owner: config.id === ACE_PLANNER_QWEN3_CONFIG.id
            ? "planner"
            : "non-planner",
          kind: shape.kind,
          batch: shape.batch,
          tokens: shape.tokens,
          requestedArm: opt0087Arm,
        },
      }, captureOpt0087Selection);
      return dispatch;
    }));
    this.requireLive(" while compiling");

    // B executes its real direct kernels, but the cooperative partition is
    // deliberately computed from A's padded plans. This keeps the accepted
    // two-quantum full head (and every compact one-quantum head) identical.
    const headTopologyPlans = opt0087Arm === undefined
      ? undefined
      : headRequests.map(({ shape: headShape }) => planAceTiledGemm(headShape));
    const headDispatch = sequenceDispatch(heads, headTopologyPlans);
    let headSliceCursor = 0;
    const opt0087HeadQuantumSliceFirstRows = opt0087Arm === undefined
      ? undefined
      : Object.freeze(headDispatch.cooperativeQuanta.map((quantum) => {
          const firstRows = headSlices.slice(
            headSliceCursor,
            headSliceCursor + quantum.primitiveCount,
          ).map(({ globalFirstRow }) => globalFirstRow);
          headSliceCursor += quantum.primitiveCount;
          return Object.freeze(firstRows);
        }));
    if (
      opt0087HeadQuantumSliceFirstRows !== undefined &&
      headSliceCursor !== headSlices.length
    ) {
      throw new Error("OPT-0087 tied-head quantum slices did not reconcile");
    }

    const quanta: AcePlannerModelQuantum[] = [
      ...modelQuanta(
        `${label}-embedding`,
        "embedding",
        null,
        bindings.weights.embedding.length,
        embeddingDispatch,
      ),
      ...layerDispatches.flatMap((dispatch, layer) => modelQuanta(
        `${label}-layer-${layer}`,
        "layer",
        layer,
        dispatch.primitiveCount,
        dispatch,
      )),
      ...modelQuanta(`${label}-final-norm`, "final-norm", null, 1, finalNorm),
      ...modelQuanta(`${label}-last-row-gather`, "last-row-gather", null, 1, gather),
      ...modelQuanta(
        `${label}-tied-lm-head`,
        "tied-lm-head",
        null,
        heads.length,
        headDispatch,
      ),
    ];
    const primitiveCount =
      1 + layerDispatches.reduce(
        (total, dispatch) => total + dispatch.primitiveCount,
        0,
      ) + 1 + 1 + heads.length;
    return Object.freeze({
      label,
      plan,
      logitRange: logitRange === undefined
        ? null
        : Object.freeze({ ...logitRange }),
      headSlices,
      primitiveCount,
      quanta: Object.freeze(quanta),
      ...(opt0087DenseSelections === undefined
        ? {}
        : {
            opt0087DenseSelections: Object.freeze(opt0087DenseSelections),
          }),
      ...(opt0087HeadQuantumSliceFirstRows === undefined
        ? {}
        : { opt0087HeadQuantumSliceFirstRows }),
      encode(pass: GPUComputePassEncoder): void {
        for (const quantum of quanta) quantum.encode(pass);
      },
    });
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.embedding.destroy();
    this.qwen.destroy();
    this.copy.destroy();
    this.tiedGemm.destroy();
  }

  private requireLive(suffix = ""): void {
    if (this.destroyed) {
      throw new Error(`ACE planner-model runtime was destroyed${suffix}`);
    }
  }
}

function plannerHeadWeightSlice(
  modelProfile: AceModelProfileId,
  hiddenSize: number,
  owner: GPUBufferBinding,
  slice: AcePlannerHeadSlice,
): GPUBufferBinding {
  const scalarOffset = slice.localFirstRow === 0
    ? 0
    : checkedAceProduct(
        [slice.localFirstRow, hiddenSize],
        "ACE planner head weight slice scalar offset",
      );
  if (modelProfile === "reference-bf16" && scalarOffset % 2 !== 0) {
    throw new RangeError(
      "ACE planner BF16 head weight slice starts inside a packed word",
    );
  }
  const byteOffset = scalarOffset === 0
    ? 0
    : modelProfile === "reference-bf16"
    ? checkedAceProduct(
        [scalarOffset / 2, 4],
        "ACE planner head weight slice byte offset",
      )
    : checkedAceProduct(
        [scalarOffset, 2],
        "ACE planner head weight slice byte offset",
      );
  const byteLength = acePackedWeightBytes(
    modelProfile,
    checkedAceProduct(
      [slice.rowCount, hiddenSize],
      "ACE planner head weight slice elements",
    ),
  );
  const ownerOffset = owner.offset ?? 0;
  const ownerBytes = owner.size ?? owner.buffer.size - ownerOffset;
  if (byteOffset + byteLength > ownerBytes) {
    throw new RangeError("ACE planner head weight slice exceeds its shard");
  }
  const offset = ownerOffset + byteOffset;
  if (offset % 256 !== 0) {
    throw new RangeError("ACE planner head weight slice is not binding aligned");
  }
  return Object.freeze({ buffer: owner.buffer, offset, size: byteLength });
}

/** @internal Tied-head request without its per-invocation A/B selector facts. */
export type AceOpt0087PlannerHeadDenseRequest = Omit<
  AceOpt0087PlannerDenseRequest,
  "invocation"
> & Readonly<{ readonly role: "tied-lm-head" }>;

/**
 * @internal
 * Materialize every physical tied-head slice directly over its authenticated
 * embedding shard. The returned binding is a view, never a copied weight.
 */
export function aceOpt0087PlannerHeadDenseRequests(
  label: string,
  modelProfile: AceModelProfileId,
  config: AceQwen3Config,
  shape: AceQwen3PlannerShape,
  bindings: AcePlannerModelBindings,
  slices: readonly AcePlannerHeadSlice[],
): readonly AceOpt0087PlannerHeadDenseRequest[] {
  return Object.freeze(slices.map((slice) => Object.freeze({
    label: `${label}-lm-head-rows-${slice.globalFirstRow}`,
    role: "tied-lm-head" as const,
    shape: Object.freeze({
      rows: shape.batch,
      inner: config.hiddenSize,
      columns: slice.rowCount,
    }),
    bindings: Object.freeze({
      activation: bindings.scratch.lastHiddenRows,
      weight: plannerHeadWeightSlice(
        modelProfile,
        config.hiddenSize,
        bindings.weights.embedding[slice.shardIndex]!.weight,
        slice,
      ),
      output: bindings.logits[slice.shardIndex]!,
    }),
  })));
}

export function planAcePlannerModel(
  modelProfile: AceModelProfileId,
  batch: AcePlannerModelBatch,
): AcePlannerModelPlan {
  validateAcePlannerModelBatch(batch);
  return planAceQwen3PlannerModel(modelProfile, ACE_PLANNER_QWEN3_CONFIG, {
    kind: batch.kind,
    batch: batch.rows as 1 | 2,
    tokens: batch.tokens,
    cacheCapacity: batch.cacheCapacity,
  });
}

export function planAceQwen3PlannerModel(
  modelProfile: AceModelProfileId,
  config: AceQwen3Config,
  shape: AceQwen3PlannerShape,
): AcePlannerModelPlan {
  if (shape.batch !== 1 && shape.batch !== 2) {
    throw new RangeError("ACE planner model requires one row or two CFG rows");
  }
  if (shape.kind !== "prefill" && shape.kind !== "decode") {
    throw new TypeError(`Unknown ACE planner model kind ${String(shape.kind)}`);
  }
  if (shape.kind === "decode" && shape.tokens !== 1) {
    throw new RangeError("ACE planner decode graph requires exactly one token per row");
  }
  const block = planAceQwen3Block(config, {
    batch: shape.batch,
    tokens: shape.tokens,
    attention: { kind: "cached", cacheCapacity: shape.cacheCapacity },
  });
  const blockScratchElements = checkedSum([
    5 * block.hiddenElements,
    6 * block.queryElements,
    6 * block.keyValueElements,
    3 * block.intermediateElements,
  ], "ACE planner block scratch elements");
  const lastHiddenElements = checkedAceProduct(
    [shape.batch, config.hiddenSize],
    "ACE planner last hidden rows",
  );
  const transientActivationElements = checkedSum([
    blockScratchElements,
    // Embedding, two layer ping-pong ranges, and full final-norm output.
    4 * block.hiddenElements,
    lastHiddenElements,
  ], "ACE planner transient activation elements");
  const logitsElements = checkedAceProduct(
    [shape.batch, config.vocabularySize],
    "ACE planner logits",
  );
  const oneCacheTensorElements = checkedAceProduct([
    shape.batch,
    config.keyValueHeads,
    shape.cacheCapacity,
    config.headDimension,
  ], "ACE planner cache tensor");
  const kvCacheElementsPerLayer = checkedProductAllowZero(
    oneCacheTensorElements,
    2,
    "ACE planner per-layer KV cache",
  );
  const kvCacheElements = checkedProductAllowZero(
    kvCacheElementsPerLayer,
    config.layerCount,
    "ACE planner complete KV cache",
  );
  const totalActivationElements = checkedSum([
    transientActivationElements,
    logitsElements,
    kvCacheElements,
  ], "ACE planner total activation elements");
  const cacheValidityBytes = checkedProductAllowZero(
    checkedAceProduct(
      [shape.batch, shape.cacheCapacity],
      "ACE planner cache validity",
    ),
    U32_BYTES,
    "ACE planner cache validity bytes",
  );
  const ropeElements = checkedAceProduct(
    [shape.batch, shape.tokens, config.headDimension],
    "ACE planner RoPE controls",
  );
  const invocationU32Elements = checkedSum([
    block.rows, // token IDs
    shape.batch * 2, // valid lengths
    block.rows, // physical query positions
    block.rows, // source validity
    shape.batch, // row starts
    shape.batch, // write status
    shape.batch, // last-row gather indices
  ], "ACE planner invocation U32 controls");
  const invocationControlBytes = checkedSum([
    checkedProductAllowZero(
      invocationU32Elements,
      U32_BYTES,
      "ACE planner invocation U32 bytes",
    ),
    checkedProductAllowZero(
      ropeElements,
      Float32Array.BYTES_PER_ELEMENT * 2,
      "ACE planner RoPE bytes",
    ),
  ], "ACE planner invocation control bytes");
  const transientActivationBytes = aceActivationBytes(
    modelProfile,
    transientActivationElements,
  );
  const logitsBytes = aceActivationBytes(modelProfile, logitsElements);
  const kvCacheBytesPerLayer = aceActivationBytes(
    modelProfile,
    kvCacheElementsPerLayer,
  );
  const kvCacheBytes = aceActivationBytes(modelProfile, kvCacheElements);
  const totalActivationBytes = aceActivationBytes(
    modelProfile,
    totalActivationElements,
  );
  return Object.freeze({
    ...shape,
    rows: block.rows,
    layerCount: config.layerCount,
    hiddenElements: block.hiddenElements,
    queryElements: block.queryElements,
    keyValueElements: block.keyValueElements,
    intermediateElements: block.intermediateElements,
    blockScratchElements,
    transientActivationElements,
    transientActivationBytes,
    logitsElements,
    logitsBytes,
    kvCacheElementsPerLayer,
    kvCacheBytesPerLayer,
    kvCacheElements,
    kvCacheBytes,
    totalActivationElements,
    totalActivationBytes,
    cacheValidityBytes,
    invocationControlBytes,
    totalNonWeightBytes: checkedSum([
      totalActivationBytes,
      cacheValidityBytes,
      invocationControlBytes,
    ], "ACE planner non-weight bytes"),
  });
}

/** Clone the exact CPU controls that the graph owner must upload for this call. */
export function createAcePlannerModelControlData(
  batch: AcePlannerModelBatch,
): AcePlannerModelControlData {
  validateAcePlannerModelBatch(batch);
  const rope = createAceQwen3RopeTables([...batch.rotaryPositionIds], {
    batch: batch.rows,
    tokens: batch.tokens,
    headDimension: ACE_PLANNER_QWEN3_CONFIG.headDimension,
    ropeTheta: ACE_PLANNER_QWEN3_CONFIG.ropeTheta,
    maximumPositionEmbeddings:
      ACE_PLANNER_QWEN3_CONFIG.maximumPositionEmbeddings,
  });
  const lastPhysicalRowIndices = new Uint32Array(batch.rows);
  lastPhysicalRowIndices.fill(batch.tokens - 1);
  return Object.freeze({
    validLengths: batch.causal.validLengths.slice(),
    queryPositions: batch.causal.queryPositions.slice(),
    sourceValidity: batch.causal.sourceValidity.slice(),
    rowStartPositions: batch.causal.rowStartPositions.slice(),
    lastPhysicalRowIndices,
    clearCacheBeforeDispatch: batch.kind === "prefill",
    cosine: rope.cosine,
    sine: rope.sine,
  });
}

/** Resolve all 310 logical / 314 physical authenticated planner tensors. */
export function resolveAcePlannerModelWeights(
  resolver: AcePlannerTensorResolver,
  modelProfile: AceModelProfileId,
): AcePlannerModelWeights {
  const config = ACE_PLANNER_QWEN3_CONFIG;
  const embedding = resolver.logicalTensor(
    ACE_PLANNER_SHARED_TENSOR_NAMES.embedding,
  );
  requireLogicalShape(embedding, [config.vocabularySize, config.hiddenSize]);
  if (embedding.parts.length !== ACE_PLANNER_EMBEDDING_ROW_PARTS.length) {
    throw new Error(
      `ACE planner embedding has ${embedding.parts.length} parts; ` +
        `${ACE_PLANNER_EMBEDDING_ROW_PARTS.length} required`,
    );
  }
  const embeddingBindings = embedding.parts.map((part, index) => {
    const expected = ACE_PLANNER_EMBEDDING_ROW_PARTS[index]!;
    requireProfileTensor(part.tensor, modelProfile, true);
    if (
      part.tensor.partStart !== expected.firstRow ||
      part.tensor.partEnd !== expected.firstRow + expected.rowCount
    ) {
      throw new Error(`ACE planner embedding row shard ${index} changed`);
    }
    return Object.freeze({
      firstRow: expected.firstRow,
      rowCount: expected.rowCount,
      weight: part.binding,
    });
  });
  const layers = ACE_PLANNER_LAYER_TENSOR_NAMES.map((names) => Object.freeze({
    inputLayerNorm: resolveUnsharded(
      resolver, names.inputLayerNorm, [config.hiddenSize], modelProfile,
    ),
    queryProjection: resolveUnsharded(resolver, names.queryProjection, [
      config.queryHeads * config.headDimension,
      config.hiddenSize,
    ], modelProfile),
    keyProjection: resolveUnsharded(resolver, names.keyProjection, [
      config.keyValueHeads * config.headDimension,
      config.hiddenSize,
    ], modelProfile),
    valueProjection: resolveUnsharded(resolver, names.valueProjection, [
      config.keyValueHeads * config.headDimension,
      config.hiddenSize,
    ], modelProfile),
    queryNorm: resolveUnsharded(
      resolver, names.queryNorm, [config.headDimension], modelProfile,
    ),
    keyNorm: resolveUnsharded(
      resolver, names.keyNorm, [config.headDimension], modelProfile,
    ),
    outputProjection: resolveUnsharded(resolver, names.outputProjection, [
      config.hiddenSize,
      config.queryHeads * config.headDimension,
    ], modelProfile),
    postAttentionLayerNorm: resolveUnsharded(
      resolver, names.postAttentionLayerNorm, [config.hiddenSize], modelProfile,
    ),
    gateProjection: resolveUnsharded(resolver, names.gateProjection, [
      config.intermediateSize,
      config.hiddenSize,
    ], modelProfile),
    upProjection: resolveUnsharded(resolver, names.upProjection, [
      config.intermediateSize,
      config.hiddenSize,
    ], modelProfile),
    downProjection: resolveUnsharded(resolver, names.downProjection, [
      config.hiddenSize,
      config.intermediateSize,
    ], modelProfile),
  }));
  return Object.freeze({
    embedding: Object.freeze(embeddingBindings),
    layers: Object.freeze(layers),
    finalNorm: resolveUnsharded(
      resolver,
      ACE_PLANNER_SHARED_TENSOR_NAMES.finalNorm,
      [config.hiddenSize],
      modelProfile,
    ),
  });
}

/** Fail closed on missing, hidden, reshaped, or differently encoded planner data. */
export function validateAcePlannerManifestInventory(
  manifest: AcePackageManifest,
): void {
  const modelProfile: AceModelProfileId = manifest.profile === "reference"
    ? "reference-bf16"
    : "raw-fp16";
  const physicalEntries = Object.entries(manifest.tensors).filter(
    ([name, tensor]) =>
      tensor.phase === "planner" ||
      name.startsWith("planner.") ||
      tensor.logicalTensor.startsWith("planner."),
  );
  requireExactInventory(
    physicalEntries.map(([name]) => name).sort(),
    [...ACE_PLANNER_PHYSICAL_TENSOR_NAMES].sort(),
    `${manifest.profile} physical planner`,
  );
  const logical = [...new Set(physicalEntries.map(([, tensor]) =>
    tensor.logicalTensor
  ))].sort();
  requireExactInventory(
    logical,
    [...ACE_PLANNER_LOGICAL_TENSOR_NAMES].sort(),
    `${manifest.profile} logical planner`,
  );

  const expectedShapes = plannerExpectedShapes();
  for (const name of ACE_PLANNER_LOGICAL_TENSOR_NAMES) {
    const expectedShape = expectedShapes.get(name)!;
    const parts = physicalEntries
      .map(([, tensor]) => tensor)
      .filter((tensor) => tensor.logicalTensor === name)
      .sort((left, right) => left.partStart - right.partStart);
    const expectedParts = name === ACE_PLANNER_SHARED_TENSOR_NAMES.embedding
      ? ACE_PLANNER_EMBEDDING_ROW_PARTS
      : [{ firstRow: 0, rowCount: expectedShape[0]! }];
    if (parts.length !== expectedParts.length) {
      throw new Error(`ACE planner tensor ${name} physical part count changed`);
    }
    for (let index = 0; index < parts.length; index += 1) {
      const part = parts[index]!;
      const expected = expectedParts[index]!;
      requireTensorShape(part, expectedShape, name);
      requireProfileTensor(
        part,
        modelProfile,
        name === ACE_PLANNER_SHARED_TENSOR_NAMES.embedding,
      );
      if (
        part.partStart !== expected.firstRow ||
        part.partEnd !== expected.firstRow + expected.rowCount
      ) {
        throw new Error(`ACE planner tensor ${name} part ${index} range changed`);
      }
      const expectedStorageShape = modelProfile === "reference-bf16"
        ? [Math.ceil(
            expected.rowCount * expectedShape.slice(1).reduce(
              (product, dimension) => product * dimension,
              1,
            ) / 2,
          )]
        : [expected.rowCount, ...expectedShape.slice(1)];
      if (!sameShape(part.storageShape, expectedStorageShape)) {
        throw new Error(`ACE planner tensor ${name} part ${index} storage shape changed`);
      }
    }
  }
}

export function validateAcePlannerModelBatch(batch: AcePlannerModelBatch): void {
  if (batch.kind !== "prefill" && batch.kind !== "decode") {
    throw new TypeError(
      `Unknown ACE planner batch kind ${String((batch as { kind?: unknown }).kind)}`,
    );
  }
  if (batch.rows !== 1 && batch.rows !== 2) {
    throw new RangeError("ACE planner batch must contain one or two rows");
  }
  requirePositiveSafeInteger(batch.tokens, "ACE planner batch tokens");
  requirePositiveSafeInteger(batch.cacheCapacity, "ACE planner batch cacheCapacity");
  if (
    batch.tokens > batch.cacheCapacity ||
    batch.cacheCapacity > ACE_PLANNER_QWEN3_CONFIG.maximumPositionEmbeddings
  ) {
    throw new RangeError("ACE planner batch cache geometry is outside the model limit");
  }
  if (batch.conditionalRow !== 0) {
    throw new RangeError("ACE planner conditional row must be physical row zero");
  }
  if (batch.unconditionalRow !== (batch.rows === 2 ? 1 : null)) {
    throw new RangeError("ACE planner unconditional row does not match its physical CFG row");
  }
  const rows = checkedAceProduct(
    [batch.rows, batch.tokens],
    "ACE planner batch physical rows",
  );
  requireLength(batch.inputIds, rows, "ACE planner input IDs");
  requireLength(batch.rotaryPositionIds, rows, "ACE planner rotary positions");
  requireLength(batch.causal.queryPositions, rows, "ACE planner causal query positions");
  requireLength(batch.causal.sourceValidity, rows, "ACE planner source validity");
  requireLength(batch.causal.validLengths, batch.rows * 2, "ACE planner valid lengths");
  requireLength(batch.causal.rowStartPositions, batch.rows, "ACE planner row starts");
  for (let index = 0; index < rows; index += 1) {
    const tokenId = batch.inputIds[index]!;
    if (tokenId >= ACE_PLANNER_QWEN3_CONFIG.vocabularySize) {
      throw new RangeError(`ACE planner inputIds[${index}] exceeds the vocabulary`);
    }
    if (
      batch.rotaryPositionIds[index] !== batch.causal.queryPositions[index]
    ) {
      throw new Error(`ACE planner rotary position ${index} is not the physical cache slot`);
    }
    const validity = batch.causal.sourceValidity[index]!;
    if (validity !== 0 && validity !== 1) {
      throw new RangeError(`ACE planner source validity ${index} is not binary`);
    }
  }
  if (batch.kind === "prefill") {
    requireLength(batch.keyValidity, rows, "ACE planner prefill key validity");
    if (batch.tokens >= batch.cacheCapacity) {
      throw new RangeError("ACE planner prefill must leave a cache slot for decoding");
    }
    for (let row = 0; row < batch.rows; row += 1) {
      requireBatchRowControls(batch, row, 0, batch.tokens);
    }
    requireEqualArrays(
      batch.keyValidity,
      batch.causal.sourceValidity,
      "ACE planner left-padding validity",
    );
  } else {
    if (batch.tokens !== 1) {
      throw new RangeError("ACE planner decode must append exactly one token per row");
    }
    if (
      !Number.isSafeInteger(batch.cachedTokensBeforeAppend) ||
      batch.cachedTokensBeforeAppend <= 0 ||
      batch.cachedTokensBeforeAppend >= batch.cacheCapacity
    ) {
      throw new RangeError("ACE planner decode cache position is invalid");
    }
    for (let row = 0; row < batch.rows; row += 1) {
      requireBatchRowControls(
        batch,
        row,
        batch.cachedTokensBeforeAppend,
        batch.cachedTokensBeforeAppend + 1,
      );
      if (batch.causal.sourceValidity[row] !== 1) {
        throw new Error("ACE planner decode source token must be valid");
      }
      if (batch.inputIds[row] !== batch.inputIds[0]) {
        throw new Error("ACE planner CFG decode rows must append the same token");
      }
    }
  }
}

function requireBatchRowControls(
  batch: AcePlannerModelBatch,
  row: number,
  expectedStart: number,
  expectedValidLength: number,
): void {
  if (batch.causal.rowStartPositions[row] !== expectedStart) {
    throw new Error(`ACE planner row ${row} cache start changed`);
  }
  if (
    batch.causal.validLengths[row * 2] !== batch.tokens ||
    batch.causal.validLengths[row * 2 + 1] !== expectedValidLength
  ) {
    throw new Error(`ACE planner row ${row} causal lengths changed`);
  }
  for (let token = 0; token < batch.tokens; token += 1) {
    const physicalIndex = row * batch.tokens + token;
    if (batch.causal.queryPositions[physicalIndex] !== expectedStart + token) {
      throw new Error(`ACE planner row ${row} query positions are not physical slots`);
    }
  }
}

function validatePlannerLogitBindings(
  label: string,
  modelProfile: AceModelProfileId,
  config: AceQwen3Config,
  batch: number,
  embedding: readonly AceEmbeddingShardBinding[],
  lastHiddenRows: GPUBufferBinding,
  logits: readonly GPUBufferBinding[],
): void {
  if (logits.length !== embedding.length) {
    throw new RangeError(
      `${label} has ${logits.length} logit shards; ${embedding.length} required`,
    );
  }
  requireAceBindingBytes(
    lastHiddenRows,
    aceActivationBytes(modelProfile, batch * config.hiddenSize),
    `${label} last hidden rows`,
  );
  const weights = embedding.map(({ weight }) => weight);
  for (let index = 0; index < logits.length; index += 1) {
    const output = logits[index]!;
    requireAceBindingBytes(
      output,
      aceActivationBytes(modelProfile, batch * embedding[index]!.rowCount),
      `${label} logits shard ${index}`,
    );
    requireAceDisjointOutput(
      output,
      [
        lastHiddenRows,
        ...weights,
        ...logits.filter((_, other) => other !== index),
      ],
      `${label} logits shard ${index}`,
    );
  }
}

function sequenceDispatch(
  dispatches: readonly AceGemmDispatch[],
  controlTopologyPlans?: readonly AceCooperativeGemmPlan[],
): Readonly<{
  readonly label: string;
  readonly cooperativeQuanta: readonly AceGpuEncodeQuantum[];
  encode(pass: GPUComputePassEncoder): void;
}> {
  if (
    controlTopologyPlans !== undefined &&
    controlTopologyPlans.length !== dispatches.length
  ) {
    throw new RangeError("OPT-0087 tied-head topology plan count changed");
  }
  const cooperativeDispatches = controlTopologyPlans === undefined
    ? dispatches
    : dispatches.map((dispatch, index) => {
        const plan = controlTopologyPlans[index]!;
        if (plan.outputRangeCount !== dispatch.rangeCount) {
          throw new Error(
            `${dispatch.label} OPT-0087 candidate/control range count changed`,
          );
        }
        return Object.freeze({
          label: dispatch.label,
          plan,
          rangeCount: dispatch.rangeCount,
          encodeRange(pass: GPUComputePassEncoder, rangeIndex: number): void {
            dispatch.encodeRange(pass, rangeIndex);
          },
        });
      });
  const cooperativeQuanta = Object.freeze(
    aceCompositeCooperativeQuanta(cooperativeDispatches),
  );
  return Object.freeze({
    label: dispatches[0]?.label ?? "empty-gemm-sequence",
    cooperativeQuanta,
    encode(pass: GPUComputePassEncoder): void {
      for (const dispatch of dispatches) dispatch.encode(pass);
    },
  });
}

function modelQuanta(
  id: string,
  kind: AcePlannerModelQuantumKind,
  layer: number | null,
  primitiveCount: number,
  dispatch: Pick<
    AceEmbeddingDispatch | AceQwen3BlockDispatch | AceGemmDispatch |
      AceCopyDispatch<AceGatherRowsPlan>,
    "encode"
  > & Readonly<{
    readonly label?: string;
    readonly plan?: unknown;
    readonly rangeCount?: number;
    readonly encodeRange?: AceGemmDispatch["encodeRange"];
    readonly cooperativeQuanta?: readonly AceGpuEncodeQuantum[];
  }>,
): readonly AcePlannerModelQuantum[] {
  const expanded = dispatch.cooperativeQuanta ??
    (dispatch.label !== undefined && dispatch.rangeCount !== undefined &&
        dispatch.encodeRange !== undefined
      ? acePrimitiveCooperativeQuanta(dispatch as AceGemmDispatch)
      : Object.freeze([Object.freeze({
          id,
          primitiveCount,
          encode(pass: GPUComputePassEncoder): void {
            dispatch.encode(pass);
          },
        })]));
  return Object.freeze(expanded.map((quantum, index) => Object.freeze({
    id: expanded.length === 1 ? id : `${id}-part-${index}`,
    logicalId: id,
    kind,
    layer,
    primitiveCount: expanded.length === 1 ? primitiveCount : quantum.primitiveCount,
    encode(pass: GPUComputePassEncoder): void {
      quantum.encode(pass);
    },
  })));
}

function plannerExpectedShapes(): ReadonlyMap<string, readonly number[]> {
  const config = ACE_PLANNER_QWEN3_CONFIG;
  const shapes = new Map<string, readonly number[]>([
    [ACE_PLANNER_SHARED_TENSOR_NAMES.embedding, [
      config.vocabularySize,
      config.hiddenSize,
    ]],
    [ACE_PLANNER_SHARED_TENSOR_NAMES.finalNorm, [config.hiddenSize]],
  ]);
  for (const names of ACE_PLANNER_LAYER_TENSOR_NAMES) {
    shapes.set(names.inputLayerNorm, [config.hiddenSize]);
    shapes.set(names.queryProjection, [
      config.queryHeads * config.headDimension,
      config.hiddenSize,
    ]);
    shapes.set(names.keyProjection, [
      config.keyValueHeads * config.headDimension,
      config.hiddenSize,
    ]);
    shapes.set(names.valueProjection, [
      config.keyValueHeads * config.headDimension,
      config.hiddenSize,
    ]);
    shapes.set(names.queryNorm, [config.headDimension]);
    shapes.set(names.keyNorm, [config.headDimension]);
    shapes.set(names.outputProjection, [
      config.hiddenSize,
      config.queryHeads * config.headDimension,
    ]);
    shapes.set(names.postAttentionLayerNorm, [config.hiddenSize]);
    shapes.set(names.gateProjection, [config.intermediateSize, config.hiddenSize]);
    shapes.set(names.upProjection, [config.intermediateSize, config.hiddenSize]);
    shapes.set(names.downProjection, [config.hiddenSize, config.intermediateSize]);
  }
  return shapes;
}

function resolveUnsharded(
  resolver: AcePlannerTensorResolver,
  name: string,
  expectedShape: readonly number[],
  modelProfile: AceModelProfileId,
): GPUBufferBinding {
  const logical = resolver.logicalTensor(name);
  requireLogicalShape(logical, expectedShape);
  if (logical.parts.length !== 1) {
    throw new Error(`ACE planner tensor ${name} unexpectedly has ${logical.parts.length} parts`);
  }
  requireProfileTensor(logical.parts[0]!.tensor, modelProfile, false);
  return resolver.binding(name);
}

function requireLogicalShape(
  logical: AceGpuLogicalTensor,
  expected: readonly number[],
): void {
  if (!sameShape(logical.logicalShape, expected)) {
    throw new Error(
      `ACE planner tensor ${logical.logicalTensor} has shape ` +
        `[${logical.logicalShape.join(",")}], expected [${expected.join(",")}]`,
    );
  }
}

function requireTensorShape(
  tensor: AcePackageTensorRecord,
  expected: readonly number[],
  name: string,
): void {
  if (!sameShape(tensor.logicalShape, expected)) {
    throw new Error(`ACE planner tensor ${name} logical shape changed`);
  }
}

function requireProfileTensor(
  tensor: AcePackageTensorRecord,
  modelProfile: AceModelProfileId,
  rowSharded: boolean,
): void {
  const expectedDtype = modelProfile === "reference-bf16"
    ? "uint32-bf16-pairs"
    : "float16";
  const expectedLayout = rowSharded
    ? modelProfile === "reference-bf16"
      ? "row-shard-axis0-bf16-pairs-lsb-u32"
      : "row-shard-axis0"
    : modelProfile === "reference-bf16"
      ? "source-row-major-bf16-pairs-lsb-u32"
      : "source-row-major";
  if (
    tensor.phase !== "planner" ||
    tensor.lifetime !== "planner" ||
    tensor.dtype !== expectedDtype ||
    tensor.layout !== expectedLayout
  ) {
    throw new Error(
      `ACE planner tensor ${tensor.logicalTensor} violates ${modelProfile} storage`,
    );
  }
}

function requireExactInventory(
  actual: readonly string[],
  expected: readonly string[],
  label: string,
): void {
  if (
    actual.length === expected.length &&
    actual.every((value, index) => value === expected[index])
  ) return;
  const actualSet = new Set(actual);
  const expectedSet = new Set(expected);
  const missing = expected.filter((name) => !actualSet.has(name));
  const hidden = actual.filter((name) => !expectedSet.has(name));
  throw new Error(
    `ACE ${label} inventory changed; missing=[${missing.join(",")}], ` +
      `hidden=[${hidden.join(",")}]`,
  );
}

function requireLength(
  value: ArrayLike<number>,
  expected: number,
  label: string,
): void {
  if (value.length !== expected) {
    throw new RangeError(`${label} has ${value.length} entries; ${expected} required`);
  }
}

function requireEqualArrays(
  left: ArrayLike<number>,
  right: ArrayLike<number>,
  label: string,
): void {
  requireLength(left, right.length, label);
  for (let index = 0; index < right.length; index += 1) {
    if (left[index] !== right[index]) {
      throw new Error(`${label} differs at ${index}`);
    }
  }
}

function sameShape(left: readonly number[], right: readonly number[]): boolean {
  return left.length === right.length &&
    left.every((value, index) => value === right[index]);
}

function checkedSum(values: readonly number[], label: string): number {
  let result = 0;
  for (const value of values) {
    result = checkedAceSum(result, value, label);
  }
  return result;
}

function checkedProductAllowZero(
  left: number,
  right: number,
  label: string,
): number {
  if (
    !Number.isSafeInteger(left) ||
    left < 0 ||
    !Number.isSafeInteger(right) ||
    right < 0
  ) {
    throw new RangeError(`${label} contains an invalid factor`);
  }
  const result = left * right;
  if (!Number.isSafeInteger(result)) {
    throw new RangeError(`${label} exceeds safe integer arithmetic`);
  }
  return result;
}

function padRow(row: number): string {
  return row.toString().padStart(6, "0");
}
