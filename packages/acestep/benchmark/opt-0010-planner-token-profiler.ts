import {
  ACE_COOPERATIVE_GPU_IDLE_MILLISECONDS,
} from "../src/runtime/scheduler.js";
import {
  ACE_PLANNER_SEMANTIC_CODE_COUNT,
  ACE_PLANNER_SEMANTIC_FIRST_TOKEN_ID,
} from "../src/runtime/planner.js";
import { ACE_QWEN_IM_END_TOKEN_ID } from "../src/tokenizer/index.js";
import {
  ACE_PLANNER_EMBEDDING_ROW_PARTS,
  type AcePlannerModelQuantumKind,
} from "../src/webgpu/planner-model.js";
import { planAceAttention } from "../src/webgpu/kernels/attention.js";
import { planAceBatchedRope } from "../src/webgpu/kernels/batched-rope.js";
import { planAceEmbedding } from "../src/webgpu/kernels/embedding.js";
import {
  ACE_GEMM_MAX_MULTIPLY_ADDS_PER_RANGE,
  planAceTiledGemm,
} from "../src/webgpu/kernels/gemm.js";
import { planAceKvCacheWrite } from "../src/webgpu/kernels/kv-cache.js";
import { planAceRmsNorm } from "../src/webgpu/kernels/rmsnorm.js";
import { planAceGatherRows } from "../src/webgpu/kernels/tensor-copy.js";
import {
  planAceHeadTransform,
  planAceTransformerTensor,
} from "../src/webgpu/kernels/transformer-plumbing.js";
import { ACE_PLANNER_QWEN3_CONFIG } from "../src/webgpu/qwen3.js";

const FP16_BYTES = Uint16Array.BYTES_PER_ELEMENT;
const U32_BYTES = Uint32Array.BYTES_PER_ELEMENT;
const STORAGE_ALIGNMENT = 256;
const LAYER_PRIMITIVE_COUNT = 22;
const PLANNER_MODEL_DISPATCH_PRIMITIVE_COUNT =
  1 +
  ACE_PLANNER_QWEN3_CONFIG.layerCount * LAYER_PRIMITIVE_COUNT +
  1 +
  1 +
  ACE_PLANNER_EMBEDDING_ROW_PARTS.length;

export const ACE_OPT_0010_PRODUCTION_SOURCE_IDENTITIES = Object.freeze({
  "src/model/graph-contract.ts":
    "ad6de30e8c089f1b0d1dd86951518150904bf1761a7285b2f5f1a2e4304aa116",
  "src/runtime/scheduler.ts":
    "a3c319b7bf7f8d4141c1a0814072d1b9f2a9b50a54b54ef82290a40ed49698c7",
  "src/webgpu/planner-model.ts":
    "673a5a3d45a749c12337cc6186212084a4b05c6ffde7f02b782a7e033113cce6",
  "src/webgpu/planner-executor.ts":
    "f802db8c35701278619c0ad5c13e7543a224662d561d1db1d6e3a71271032786",
  "src/webgpu/qwen3.ts":
    "bb982aeb380c52ea842183b1649a6674203c18a421c17ea07f68d7668309158a",
  "src/webgpu/kernels/gemm.ts":
    "8922c2cebe36186b7cbf173e69126dd45a01fbc79a2cb2f84c34090880268928",
  "src/runtime/planner.ts":
    "35c1effda1a52e6a73fabfb467a2ac7fd015197a4a6e494d360f9db78b7950b1",
  "src/tokenizer/qwen-bpe.ts":
    "5a83fa16a178621e8b9de457800564e1c4b832bc70e39d627b39cd0678064737",
  "src/tokenizer/index.ts":
    "18a9815b18f79248933087bf8c04b6f3ae653fae80fbb0e58004ce9ae38a7419",
  "src/webgpu/arena.ts":
    "11ae7e88b9b6f2ca1529e265c6e20e2576684c99396d25ee2d76d361cd0cd093",
  "src/webgpu/kernels/correctness-utils.ts":
    "f727d347e666e4f4ebdedc7acedcc6b1cdde371850b9fc83a2c214cb83290018",
  "src/webgpu/kernels/attention.ts":
    "824d86bca3cdd4be4de425a482eb516ac8fa318148d1324a750f7b7c8e8035b2",
  "src/webgpu/kernels/batched-rope.ts":
    "05301c6d020ab720c004b6a85fad76aaee725ac96df3fba6ff8c1ac57d81377b",
  "src/webgpu/kernels/embedding.ts":
    "049750864bd389978c6b90575b7f91e18fb046a884d5092db66b06527fe161d4",
  "src/webgpu/kernels/kv-cache.ts":
    "a08e5c94396771a22ebb92f90124fec71721792c31c25751dcdc57bfdabd3a79",
  "src/webgpu/kernels/rmsnorm.ts":
    "f49ace6e346c9e13ea48be9632e505a42f678b8e9e7427cd2a1f1e9ded525b5d",
  "src/webgpu/kernels/tensor-copy.ts":
    "b00963d03be8c4d29e6c2a0f504efa513b841e8641e193a8e43a43df5b7ac193",
  "src/webgpu/kernels/transformer-plumbing.ts":
    "baeb6a3931b1593d6b3a9396e27ea527a4f2083cb91571410d7ebac12053e92f",
  "src/webgpu/scoped-buffer-allocation.ts":
    "599cab5d15b441c516c75d2b4c632e35116e2b4c3c2b115415874c8119c8b7c0",
} as const);

export type AceOpt0010PlannerTokenMode = "cot-m1" | "semantic-m2";

export interface AceOpt0010PlannerProductionQuantumTag {
  readonly id: string;
  readonly logicalId: string;
  readonly kind: AcePlannerModelQuantumKind;
  readonly layer: number | null;
  /** Expanded physical dispatches encoded by this production quantum. */
  readonly primitiveCount: number;
}

export interface AceOpt0010PlannerPhysicalDispatch {
  readonly indexInQuantum: number;
  /** Canonical pipeline kind and exact shape, authenticated by source hashes. */
  readonly pipelineIdentity: string;
  /** Exact production dispatch label/range identity in encode order. */
  readonly dispatchIdentity: string;
  readonly workgroups: readonly [number, number, number];
}

export interface AceOpt0010PlannerQuantumDescriptor {
  readonly index: number;
  readonly id: string;
  readonly kind: AcePlannerModelQuantumKind;
  readonly layer: number | null;
  readonly primitiveCount: number;
  readonly productionQuantum: AceOpt0010PlannerProductionQuantumTag;
  readonly physicalDispatches: readonly AceOpt0010PlannerPhysicalDispatch[];
  readonly tiedHeadShards: readonly number[];
  readonly logicalWeightBytes: number;
  readonly logicalGemmActivationBytes: number;
  readonly logicalMultiplyAdds: number;
  readonly scheduledMultiplyAdds: number;
}

export interface AceOpt0010PlannerReadbackShard {
  readonly shardIndex: number;
  readonly globalFirstRow: number;
  readonly localFirstRow: 0;
  readonly rowCount: number;
  readonly byteOffset: number;
  readonly byteLength: number;
}

export interface AceOpt0010PlannerReadbackPlan {
  readonly rows: 1 | 2;
  readonly shards: readonly AceOpt0010PlannerReadbackShard[];
  readonly rawLogitBytes: number;
  readonly writeStatusByteOffset: number;
  readonly writeStatusByteLength: number;
  readonly bufferBytes: number;
  readonly copyCommands: readonly AceOpt0010PlannerCopyCommand[];
}

export interface AceOpt0010PlannerCopyCommand {
  readonly index: number;
  readonly sourceBufferLabel: string;
  readonly shardIndex: number | null;
  readonly sourceOffset: number;
  readonly destinationBufferLabel: "ace-planner-logit-readback";
  readonly destinationOffset: number;
  readonly copiedBytes: number;
}

export interface AceOpt0010HeadOutputRangePlan {
  readonly index: number;
  readonly firstOutput: number;
  readonly outputCount: number;
  readonly firstWorkgroup: number;
  readonly workgroupCount: number;
  readonly multiplyAdds: number;
}

export interface AceOpt0010HeadSlicePlan {
  readonly shardIndex: number;
  readonly globalFirstRow: number;
  readonly localFirstRow: number;
  readonly rowCount: number;
  readonly rows: 2;
  readonly inner: number;
  readonly columns: number;
  readonly logicalWeightTrafficBytes: number;
  readonly gemmActivationBytes: number;
  readonly rawLogitBytes: number;
  readonly logicalMultiplyAdds: number;
  readonly logicalFloatingPointOperations: number;
  readonly scheduledMultiplyAdds: number;
  readonly scheduledFloatingPointOperations: number;
  readonly outputRanges: readonly AceOpt0010HeadOutputRangePlan[];
}

export interface AceOpt0010HeadMetrics {
  /** Per decoded token; the tied embedding remains fully resident. */
  readonly logicalWeightTrafficBytes: number;
  readonly gemmActivationBytes: number;
  readonly rawLogitBytes: number;
  readonly logicalMultiplyAdds: number;
  readonly logicalFloatingPointOperations: number;
  readonly scheduledMultiplyAdds: number;
  readonly scheduledFloatingPointOperations: number;
}

export interface AceOpt0010SemanticHeadOpportunity {
  readonly firstTokenId: number;
  readonly tokenCount: number;
  readonly fullHead: AceOpt0010HeadMetrics;
  readonly restrictedCodeHead: AceOpt0010HeadMetrics & Readonly<{
    readonly intersections: readonly AceOpt0010HeadSlicePlan[];
  }>;
  readonly avoidablePerToken: AceOpt0010HeadMetrics;
  readonly terminalEos: Readonly<{
    readonly tokenId: number;
    readonly shardIndex: number;
    readonly globalFirstRow: number;
    readonly localFirstRow: number;
    readonly rowCount: 1;
    readonly optionalHeadRow: AceOpt0010HeadSlicePlan;
    readonly terminalOnly: true;
    readonly includedInRestrictedCodeHead: false;
    readonly tiedEmbeddingRemainsFullyResident: true;
  }>;
}

export interface AceOpt0010PlannerAttentionAttribution {
  readonly queryRows: 1 | 2;
  readonly queryHeads: number;
  readonly keyValueHeads: number;
  readonly headDimension: number;
  readonly validKeyValueTokens: number;
  readonly scheduledKeyValueCapacity: number;
  readonly perLayer: Readonly<{
    readonly queryElements: number;
    readonly logicalKeyValuePairElements: number;
    readonly scheduledKeyValuePairElements: number;
    readonly logicalKeyValuePairBytes: number;
    readonly scheduledKeyValuePairBytes: number;
    readonly logicalKeyValidityBytes: number;
    readonly scheduledKeyValidityBytes: number;
    readonly logicalMultiplyAdds: number;
    readonly scheduledMultiplyAdds: number;
    readonly logicalFloatingPointOperations: number;
    readonly scheduledFloatingPointOperations: number;
  }>;
  readonly allLayers: Readonly<{
    readonly logicalKeyValuePairBytes: number;
    readonly scheduledKeyValuePairBytes: number;
    readonly logicalKeyValidityBytes: number;
    readonly scheduledKeyValidityBytes: number;
    readonly logicalMultiplyAdds: number;
    readonly scheduledMultiplyAdds: number;
    readonly logicalFloatingPointOperations: number;
    readonly scheduledFloatingPointOperations: number;
  }>;
}

export interface AceOpt0010PlannerTokenAttribution {
  readonly schemaVersion: 1;
  readonly kind: "ace-opt-0010-planner-token-attribution";
  readonly modelProfile: "raw-fp16";
  readonly mode: AceOpt0010PlannerTokenMode;
  readonly rows: 1 | 2;
  readonly cachedTokensBeforeAppend: number;
  /** Logically valid after this decode append. */
  readonly validAttentionKeyValueTokens: number;
  /** Exact phase allocation and compiled attention stride. */
  readonly scheduledAttentionKeyValueCapacity: number;
  readonly attention: AceOpt0010PlannerAttentionAttribution;
  readonly productionSourceIdentities:
    typeof ACE_OPT_0010_PRODUCTION_SOURCE_IDENTITIES;
  readonly quanta: readonly AceOpt0010PlannerQuantumDescriptor[];
  readonly readback: AceOpt0010PlannerReadbackPlan;
  readonly semanticHeadOpportunity: AceOpt0010SemanticHeadOpportunity | null;
  readonly totals: Readonly<{
    readonly modelQuantumCount: number;
    /** `AcePlannerModelDispatch.primitiveCount` from production composition. */
    readonly modelDispatchPrimitiveCount: number;
    /** Sum of expanded production quantum primitive tags. */
    readonly modelPhysicalPrimitiveDispatchCount: number;
    readonly modelCommandBufferCount: number;
    readonly readbackCommandBufferCount: 1;
    readonly commandBufferCount: number;
    readonly queueDrainCount: number;
    readonly completedIdleCount: number;
    readonly requestedIdleMilliseconds: number;
    readonly residentWeightBytes: number;
    readonly logicalWeightBytes: number;
    readonly logicalGemmActivationBytes: number;
    readonly logicalMultiplyAdds: number;
    readonly scheduledMultiplyAdds: number;
  }>;
}

export interface AceOpt0010PlannerModelProgressPayload {
  readonly phaseKind: "decode";
  readonly completedCommandBuffers: number;
  readonly totalCommandBuffers: number;
  readonly queueDrains: number;
  readonly cooperativeIdleMs: number;
  readonly stage: "model";
  readonly quantum: AceOpt0010PlannerProductionQuantumTag;
  readonly peakAccountedGpuBytes: number;
  readonly cumulativeQueueDrains: number;
  readonly cumulativeCooperativeIdleMs: number;
}

export interface AceOpt0010PlannerReadbackProgressPayload {
  readonly phaseKind: "decode";
  readonly completedCommandBuffers: number;
  readonly totalCommandBuffers: number;
  readonly queueDrains: number;
  readonly cooperativeIdleMs: number;
  readonly stage: "readback";
  readonly quantum: null;
  readonly peakAccountedGpuBytes: number;
  readonly cumulativeQueueDrains: number;
  readonly cumulativeCooperativeIdleMs: number;
}

export interface AceOpt0010PlannerQuantumTrace {
  readonly index: number;
  readonly productionQuantum: AceOpt0010PlannerProductionQuantumTag;
  readonly progress: AceOpt0010PlannerModelProgressPayload;
  readonly physicalDispatches: readonly AceOpt0010PlannerPhysicalDispatch[];
  readonly encodeStartedAt: number;
  readonly encodeEndedAt: number;
  readonly submitStartedAt: number;
  readonly submitReturnedAt: number;
  readonly drainStartedAt: number;
  readonly drainEndedAt: number;
  readonly idleStartedAt: number;
  readonly progressReportedAt: number;
  readonly idleEndedAt: number;
  readonly nextEncodeStartedAt: number;
  readonly commandBufferCount: number;
  readonly submissionCount: number;
  readonly queueDrainCount: number;
  readonly completedIdleCount: number;
  readonly requestedIdleMilliseconds: number;
}

export interface AceOpt0010PlannerReadbackTrace {
  readonly progress: AceOpt0010PlannerReadbackProgressPayload;
  readonly encodeStartedAt: number;
  readonly encodeEndedAt: number;
  readonly submitStartedAt: number;
  readonly submitReturnedAt: number;
  readonly drainStartedAt: number;
  readonly drainEndedAt: number;
  readonly idleStartedAt: number;
  readonly progressReportedAt: number;
  readonly mapStartedAt: number;
  readonly mapEndedAt: number;
  readonly reconstructStartedAt: number;
  readonly reconstructEndedAt: number;
  readonly idleEndedAt: number;
  readonly invocationResolvedAt: number;
  readonly commandBufferCount: number;
  readonly submissionCount: number;
  readonly queueDrainCount: number;
  readonly completedIdleCount: number;
  readonly requestedIdleMilliseconds: number;
  readonly copyCommands: readonly AceOpt0010PlannerCopyCommand[];
}

export interface AceOpt0010PlannerTokenTrace {
  readonly quanta: readonly AceOpt0010PlannerQuantumTrace[];
  readonly readback: AceOpt0010PlannerReadbackTrace;
  readonly constraintStartedAt: number;
  readonly constraintEndedAt: number;
  readonly samplingStartedAt: number;
  readonly samplingEndedAt: number;
}

export interface AceOpt0010PlannerValidatedTrace {
  readonly schemaVersion: 1;
  readonly kind: "ace-opt-0010-planner-validated-trace";
  readonly quantumTimings: readonly Readonly<{
    readonly index: number;
    readonly id: string;
    readonly kind: AcePlannerModelQuantumKind;
    readonly layer: number | null;
    readonly encodeMilliseconds: number;
    readonly submitCallMilliseconds: number;
    readonly submitThroughDrainMilliseconds: number;
    readonly drainWaitMilliseconds: number;
    readonly actualIdleMilliseconds: number;
    readonly wallMilliseconds: number;
  }>[];
  readonly readback: Readonly<{
    readonly encodeMilliseconds: number;
    readonly submitThroughDrainMilliseconds: number;
    readonly mapMilliseconds: number;
    readonly reconstructMilliseconds: number;
    readonly actualIdleMilliseconds: number;
    readonly wallMilliseconds: number;
  }>;
  readonly constraintMilliseconds: number;
  readonly samplingMilliseconds: number;
  readonly timedWindowMilliseconds: number;
  readonly commandBufferCount: number;
  readonly queueDrainCount: number;
  readonly completedIdleCount: number;
}

export interface AceOpt0010PlannerTraceSummary {
  readonly schemaVersion: 1;
  readonly kind: "ace-opt-0010-planner-trace-summary";
  readonly mode: AceOpt0010PlannerTokenMode;
  readonly totals: AceOpt0010PlannerValidatedTrace;
  readonly byKind: readonly Readonly<{
    readonly kind: AcePlannerModelQuantumKind;
    readonly quantumCount: number;
    readonly primitiveCount: number;
    readonly wallMilliseconds: number;
    readonly submitThroughDrainMilliseconds: number;
    readonly actualIdleMilliseconds: number;
  }>[];
}

interface GemmShape {
  readonly inner: number;
  readonly columns: number;
}

const LAYER_GEMMS: readonly GemmShape[] = Object.freeze([
  { inner: 1_024, columns: 2_048 },
  { inner: 1_024, columns: 1_024 },
  { inner: 1_024, columns: 1_024 },
  { inner: 2_048, columns: 1_024 },
  { inner: 1_024, columns: 3_072 },
  { inner: 1_024, columns: 3_072 },
  { inner: 3_072, columns: 1_024 },
]);

export function createAceOpt0010PlannerTokenAttribution(
  mode: AceOpt0010PlannerTokenMode,
  cachedTokensBeforeAppend: number,
  cacheCapacity: number,
): AceOpt0010PlannerTokenAttribution {
  if (mode !== "cot-m1" && mode !== "semantic-m2") {
    throw new TypeError("ACE OPT-0010 has an unknown token mode");
  }
  if (!Number.isSafeInteger(cachedTokensBeforeAppend) ||
      cachedTokensBeforeAppend < 0) {
    throw new RangeError(
      "ACE OPT-0010 cachedTokensBeforeAppend must be a non-negative integer",
    );
  }
  if (
    !Number.isSafeInteger(cacheCapacity) ||
    cacheCapacity <= cachedTokensBeforeAppend ||
    cacheCapacity > ACE_PLANNER_QWEN3_CONFIG.maximumPositionEmbeddings
  ) {
    throw new RangeError(
      "ACE OPT-0010 cacheCapacity must exceed cachedTokensBeforeAppend and " +
        `be at most ${ACE_PLANNER_QWEN3_CONFIG.maximumPositionEmbeddings}`,
    );
  }
  const rows = mode === "cot-m1" ? 1 as const : 2 as const;
  const validAttentionKeyValueTokens = cachedTokensBeforeAppend + 1;
  const productionLabel =
    `ace-planner-decode-${rows}x1-capacity-${cacheCapacity}`;
  const layerWork = gemmTotals(rows, LAYER_GEMMS);
  const layerWeightBytes = layerWork.weightBytes +
    (2 * ACE_PLANNER_QWEN3_CONFIG.hiddenSize +
      2 * ACE_PLANNER_QWEN3_CONFIG.headDimension) * FP16_BYTES;
  const quanta: AceOpt0010PlannerQuantumDescriptor[] = [];
  const push = (
    id: string,
    kind: AcePlannerModelQuantumKind,
    layer: number | null,
    productionId: string,
    productionLogicalId: string,
    physicalDispatches: readonly AceOpt0010PlannerPhysicalDispatch[],
    tiedHeadShards: readonly number[],
    logicalWeightBytes: number,
    logicalGemmActivationBytes: number,
    logicalMultiplyAdds: number,
    scheduledMultiplyAdds: number,
  ): void => {
    const primitiveCount = physicalDispatches.length;
    quanta.push(Object.freeze({
      index: quanta.length,
      id,
      kind,
      layer,
      primitiveCount,
      productionQuantum: Object.freeze({
        id: productionId,
        logicalId: productionLogicalId,
        kind,
        layer,
        primitiveCount,
      }),
      physicalDispatches: Object.freeze(physicalDispatches),
      tiedHeadShards: Object.freeze([...tiedHeadShards]),
      logicalWeightBytes,
      logicalGemmActivationBytes,
      logicalMultiplyAdds,
      scheduledMultiplyAdds,
    }));
  };
  push(
    "embedding",
    "embedding",
    null,
    `${productionLabel}-embedding`,
    `${productionLabel}-embedding`,
    createEmbeddingPhysicalDispatches(rows, productionLabel),
    [],
    rows * ACE_PLANNER_QWEN3_CONFIG.hiddenSize * FP16_BYTES,
    0,
    0,
    0,
  );
  for (let layer = 0; layer < ACE_PLANNER_QWEN3_CONFIG.layerCount; layer += 1) {
    const physicalDispatches = createLayerPhysicalDispatches(
      rows,
      cacheCapacity,
      productionLabel,
      layer,
    );
    if (layerWork.scheduledMultiplyAdds > ACE_GEMM_MAX_MULTIPLY_ADDS_PER_RANGE) {
      throw new Error("ACE OPT-0010 layer no longer fits one production quantum");
    }
    push(
      "layer-" + layer,
      "layer",
      layer,
      `${productionLabel}-layer-${layer}`,
      `${productionLabel}-layer-${layer}`,
      physicalDispatches,
      [],
      layerWeightBytes,
      layerWork.activationBytes,
      layerWork.logicalMultiplyAdds,
      layerWork.scheduledMultiplyAdds,
    );
  }
  push(
    "final-norm",
    "final-norm",
    null,
    `${productionLabel}-final-norm`,
    `${productionLabel}-final-norm`,
    createFinalNormPhysicalDispatches(rows, productionLabel),
    [],
    ACE_PLANNER_QWEN3_CONFIG.hiddenSize * FP16_BYTES,
    0,
    0,
    0,
  );
  push(
    "last-row-gather",
    "last-row-gather",
    null,
    `${productionLabel}-last-row-gather`,
    `${productionLabel}-last-row-gather`,
    createGatherPhysicalDispatches(rows, productionLabel),
    [],
    0,
    0,
    0,
    0,
  );

  const headGroups = createHeadPhysicalDispatchGroups(rows, productionLabel);
  for (let groupIndex = 0; groupIndex < headGroups.length; groupIndex += 1) {
    const group = headGroups[groupIndex]!;
    const head = headTotals(rows, group.shardIndices);
    push(
      `tied-lm-head-${groupIndex}`,
      "tied-lm-head",
      null,
      `${productionLabel}-tied-lm-head-part-${groupIndex}`,
      `${productionLabel}-tied-lm-head`,
      group.dispatches,
      group.shardIndices,
      head.weightBytes,
      head.activationBytes,
      head.logicalMultiplyAdds,
      head.scheduledMultiplyAdds,
    );
  }

  const readback = createReadbackPlan(rows);
  const fullHeadWeightBytes =
    ACE_PLANNER_QWEN3_CONFIG.vocabularySize *
    ACE_PLANNER_QWEN3_CONFIG.hiddenSize * FP16_BYTES;
  const semanticShardIntersections = ACE_PLANNER_EMBEDDING_ROW_PARTS.flatMap(
    (shard, shardIndex) => {
    const first = Math.max(shard.firstRow, ACE_PLANNER_SEMANTIC_FIRST_TOKEN_ID);
    const last = Math.min(
      shard.firstRow + shard.rowCount,
      ACE_PLANNER_SEMANTIC_FIRST_TOKEN_ID + ACE_PLANNER_SEMANTIC_CODE_COUNT,
    );
      return last <= first
        ? []
        : [createHeadSlicePlan(
            shardIndex,
            first,
            first - shard.firstRow,
            last - first,
          )];
    },
  );
  const residentWeightBytes =
    fullHeadWeightBytes +
    ACE_PLANNER_QWEN3_CONFIG.layerCount * layerWeightBytes +
    ACE_PLANNER_QWEN3_CONFIG.hiddenSize * FP16_BYTES;
  const modelPhysicalPrimitiveDispatchCount =
    sum(quanta.map((q) => q.primitiveCount));
  const commandBufferCount = quanta.length + 1;
  const attention = createAttentionAttribution(
    rows,
    validAttentionKeyValueTokens,
    cacheCapacity,
  );
  const eosShardIndex = ACE_PLANNER_EMBEDDING_ROW_PARTS.findIndex((shard) =>
    ACE_QWEN_IM_END_TOKEN_ID >= shard.firstRow &&
    ACE_QWEN_IM_END_TOKEN_ID < shard.firstRow + shard.rowCount);
  if (eosShardIndex < 0) {
    throw new Error("ACE OPT-0010 terminal EOS row is outside planner shards");
  }
  const eosShard = ACE_PLANNER_EMBEDDING_ROW_PARTS[eosShardIndex]!;
  const semanticHeadOpportunity = mode === "semantic-m2"
    ? createSemanticHeadOpportunity(
        readback,
        semanticShardIntersections,
        eosShardIndex,
        eosShard,
      )
    : null;
  return Object.freeze({
    schemaVersion: 1,
    kind: "ace-opt-0010-planner-token-attribution",
    modelProfile: "raw-fp16",
    mode,
    rows,
    cachedTokensBeforeAppend,
    validAttentionKeyValueTokens,
    scheduledAttentionKeyValueCapacity: cacheCapacity,
    attention,
    productionSourceIdentities: ACE_OPT_0010_PRODUCTION_SOURCE_IDENTITIES,
    quanta: Object.freeze(quanta),
    readback,
    semanticHeadOpportunity,
    totals: Object.freeze({
      modelQuantumCount: quanta.length,
      modelDispatchPrimitiveCount: PLANNER_MODEL_DISPATCH_PRIMITIVE_COUNT,
      modelPhysicalPrimitiveDispatchCount,
      modelCommandBufferCount: quanta.length,
      readbackCommandBufferCount: 1 as const,
      commandBufferCount,
      queueDrainCount: commandBufferCount,
      completedIdleCount: commandBufferCount,
      requestedIdleMilliseconds:
        commandBufferCount * ACE_COOPERATIVE_GPU_IDLE_MILLISECONDS,
      residentWeightBytes,
      logicalWeightBytes: sum(quanta.map((q) => q.logicalWeightBytes)),
      logicalGemmActivationBytes:
        sum(quanta.map((q) => q.logicalGemmActivationBytes)),
      logicalMultiplyAdds: sum(quanta.map((q) => q.logicalMultiplyAdds)),
      scheduledMultiplyAdds: sum(quanta.map((q) => q.scheduledMultiplyAdds)),
    }),
  });
}

export function validateAceOpt0010PlannerTokenTrace(
  attribution: AceOpt0010PlannerTokenAttribution,
  trace: AceOpt0010PlannerTokenTrace,
): AceOpt0010PlannerValidatedTrace {
  if (trace.quanta.length !== attribution.quanta.length) {
    throw new Error("ACE OPT-0010 model quantum count changed");
  }
  let cumulativeQueueDrainBase: number | undefined;
  let cumulativeIdleBase: number | undefined;
  let peakAccountedGpuBytes: number | undefined;
  const quantumTimings = trace.quanta.map((actual, index) => {
    const expected = attribution.quanta[index]!;
    if (actual.index !== index) {
      throw new Error("ACE OPT-0010 quantum " + index + " identity changed");
    }
    requireProductionQuantumTag(
      actual.productionQuantum,
      expected.productionQuantum,
      `quantum ${index}`,
    );
    if (
      actual.physicalDispatches.length !== expected.physicalDispatches.length ||
      actual.physicalDispatches.some((dispatch, dispatchIndex) =>
        !samePhysicalDispatch(
          dispatch,
          expected.physicalDispatches[dispatchIndex]!,
        ))
    ) {
      throw new Error(
        `ACE OPT-0010 quantum ${index} physical dispatch sequence changed`,
      );
    }
    const progress = actual.progress;
    if (progress.phaseKind !== "decode" || progress.stage !== "model") {
      throw new Error(`ACE OPT-0010 quantum ${index} progress stage changed`);
    }
    requireProductionQuantumTag(
      progress.quantum,
      expected.productionQuantum,
      `quantum ${index} progress`,
    );
    const ordinal = index + 1;
    requireCount(
      progress.completedCommandBuffers,
      ordinal,
      `quantum ${index} completed command buffers`,
    );
    requireCount(
      progress.totalCommandBuffers,
      attribution.totals.commandBufferCount,
      `quantum ${index} total command buffers`,
    );
    requireCount(progress.queueDrains, ordinal, `quantum ${index} progress drains`);
    requireCount(
      progress.cooperativeIdleMs,
      ordinal * ACE_COOPERATIVE_GPU_IDLE_MILLISECONDS,
      `quantum ${index} progress idle`,
    );
    requireNonNegativeSafeInteger(
      progress.peakAccountedGpuBytes,
      `quantum ${index} peak accounted GPU bytes`,
    );
    requireNonNegativeSafeInteger(
      progress.cumulativeQueueDrains,
      `quantum ${index} cumulative drains`,
    );
    requireNonNegativeSafeInteger(
      progress.cumulativeCooperativeIdleMs,
      `quantum ${index} cumulative idle`,
    );
    if (index === 0) {
      cumulativeQueueDrainBase = progress.cumulativeQueueDrains - ordinal;
      cumulativeIdleBase =
        progress.cumulativeCooperativeIdleMs -
        ordinal * ACE_COOPERATIVE_GPU_IDLE_MILLISECONDS;
      peakAccountedGpuBytes = progress.peakAccountedGpuBytes;
      requireNonNegativeSafeInteger(
        cumulativeQueueDrainBase,
        "cumulative drain base",
      );
      requireNonNegativeSafeInteger(cumulativeIdleBase, "cumulative idle base");
    }
    requireCount(
      progress.cumulativeQueueDrains,
      cumulativeQueueDrainBase! + ordinal,
      `quantum ${index} cumulative drains`,
    );
    requireCount(
      progress.cumulativeCooperativeIdleMs,
      cumulativeIdleBase! +
        ordinal * ACE_COOPERATIVE_GPU_IDLE_MILLISECONDS,
      `quantum ${index} cumulative idle`,
    );
    requireCount(
      progress.peakAccountedGpuBytes,
      peakAccountedGpuBytes!,
      `quantum ${index} peak accounted GPU bytes`,
    );
    requireUnitCounts(actual, "quantum " + index);
    requireTimeline([
      actual.encodeStartedAt,
      actual.encodeEndedAt,
      actual.submitStartedAt,
      actual.submitReturnedAt,
      actual.drainStartedAt,
      actual.drainEndedAt,
      actual.idleStartedAt,
      actual.progressReportedAt,
      actual.idleEndedAt,
      actual.nextEncodeStartedAt,
    ], "quantum " + index);
    requirePositiveInterval(
      actual.idleStartedAt,
      actual.idleEndedAt,
      `quantum ${index} idle`,
    );
    const expectedNext = index + 1 < trace.quanta.length
      ? trace.quanta[index + 1]!.encodeStartedAt
      : trace.readback.encodeStartedAt;
    requireSame(actual.nextEncodeStartedAt, expectedNext, "next encode");
    return Object.freeze({
      index,
      id: expected.id,
      kind: expected.kind,
      layer: expected.layer,
      encodeMilliseconds: actual.encodeEndedAt - actual.encodeStartedAt,
      submitCallMilliseconds:
        actual.submitReturnedAt - actual.submitStartedAt,
      submitThroughDrainMilliseconds:
        actual.drainEndedAt - actual.submitStartedAt,
      drainWaitMilliseconds: actual.drainEndedAt - actual.drainStartedAt,
      actualIdleMilliseconds: actual.idleEndedAt - actual.idleStartedAt,
      wallMilliseconds:
        actual.nextEncodeStartedAt - actual.encodeStartedAt,
    });
  });

  const readback = trace.readback;
  requireUnitCounts(readback, "readback");
  if (
    readback.copyCommands.length !== attribution.readback.copyCommands.length ||
    readback.copyCommands.some((actual, index) =>
      !sameCopyCommand(actual, attribution.readback.copyCommands[index]!))
  ) {
    throw new Error("ACE OPT-0010 readback copy command sequence changed");
  }
  const readbackProgress = readback.progress;
  if (
    readbackProgress.phaseKind !== "decode" ||
    readbackProgress.stage !== "readback" ||
    readbackProgress.quantum !== null
  ) {
    throw new Error("ACE OPT-0010 readback progress stage changed");
  }
  const readbackOrdinal = attribution.totals.commandBufferCount;
  requireCount(
    readbackProgress.completedCommandBuffers,
    readbackOrdinal,
    "readback completed command buffers",
  );
  requireCount(
    readbackProgress.totalCommandBuffers,
    readbackOrdinal,
    "readback total command buffers",
  );
  requireCount(readbackProgress.queueDrains, readbackOrdinal, "readback progress drains");
  requireCount(
    readbackProgress.cooperativeIdleMs,
    readbackOrdinal * ACE_COOPERATIVE_GPU_IDLE_MILLISECONDS,
    "readback progress idle",
  );
  requireCount(
    readbackProgress.cumulativeQueueDrains,
    cumulativeQueueDrainBase! + readbackOrdinal,
    "readback cumulative drains",
  );
  requireCount(
    readbackProgress.cumulativeCooperativeIdleMs,
    cumulativeIdleBase! +
      readbackOrdinal * ACE_COOPERATIVE_GPU_IDLE_MILLISECONDS,
    "readback cumulative idle",
  );
  requireCount(
    readbackProgress.peakAccountedGpuBytes,
    peakAccountedGpuBytes!,
    "readback peak accounted GPU bytes",
  );
  requireTimeline([
    readback.encodeStartedAt,
    readback.encodeEndedAt,
    readback.submitStartedAt,
    readback.submitReturnedAt,
    readback.drainStartedAt,
    readback.drainEndedAt,
    readback.idleStartedAt,
    readback.progressReportedAt,
    readback.mapStartedAt,
    readback.mapEndedAt,
    readback.reconstructStartedAt,
    readback.reconstructEndedAt,
  ], "readback primary");
  requireTimeline([
    readback.idleStartedAt,
    readback.progressReportedAt,
    readback.idleEndedAt,
    readback.invocationResolvedAt,
    trace.constraintStartedAt,
    trace.constraintEndedAt,
    trace.samplingStartedAt,
    trace.samplingEndedAt,
  ], "readback idle and CPU");
  requirePositiveInterval(
    readback.idleStartedAt,
    readback.idleEndedAt,
    "readback idle",
  );
  if (readback.invocationResolvedAt < readback.reconstructEndedAt) {
    throw new Error("ACE OPT-0010 invocation resolved before reconstruction");
  }

  const commandBufferCount = sum(trace.quanta.map((q) =>
    q.commandBufferCount)) + readback.commandBufferCount;
  const queueDrainCount = sum(trace.quanta.map((q) =>
    q.queueDrainCount)) + readback.queueDrainCount;
  const completedIdleCount = sum(trace.quanta.map((q) =>
    q.completedIdleCount)) + readback.completedIdleCount;
  requireCount(
    commandBufferCount,
    attribution.totals.commandBufferCount,
    "command buffers",
  );
  requireCount(queueDrainCount, attribution.totals.queueDrainCount, "drains");
  requireCount(
    completedIdleCount,
    attribution.totals.completedIdleCount,
    "completed idles",
  );
  return Object.freeze({
    schemaVersion: 1,
    kind: "ace-opt-0010-planner-validated-trace",
    quantumTimings: Object.freeze(quantumTimings),
    readback: Object.freeze({
      encodeMilliseconds: readback.encodeEndedAt - readback.encodeStartedAt,
      submitThroughDrainMilliseconds:
        readback.drainEndedAt - readback.submitStartedAt,
      mapMilliseconds: readback.mapEndedAt - readback.mapStartedAt,
      reconstructMilliseconds:
        readback.reconstructEndedAt - readback.reconstructStartedAt,
      actualIdleMilliseconds: readback.idleEndedAt - readback.idleStartedAt,
      wallMilliseconds:
        readback.invocationResolvedAt - readback.encodeStartedAt,
    }),
    constraintMilliseconds:
      trace.constraintEndedAt - trace.constraintStartedAt,
    samplingMilliseconds: trace.samplingEndedAt - trace.samplingStartedAt,
    timedWindowMilliseconds:
      trace.samplingEndedAt - trace.quanta[0]!.encodeStartedAt,
    commandBufferCount,
    queueDrainCount,
    completedIdleCount,
  });
}

export function summarizeAceOpt0010PlannerTokenTrace(
  attribution: AceOpt0010PlannerTokenAttribution,
  trace: AceOpt0010PlannerTokenTrace,
): AceOpt0010PlannerTraceSummary {
  const validated = validateAceOpt0010PlannerTokenTrace(attribution, trace);
  const order: readonly AcePlannerModelQuantumKind[] = [
    "embedding",
    "layer",
    "final-norm",
    "last-row-gather",
    "tied-lm-head",
  ];
  const byKind = order.map((kind) => {
    const timings = validated.quantumTimings.filter((entry) =>
      entry.kind === kind);
    const descriptors = attribution.quanta.filter((entry) =>
      entry.kind === kind);
    return Object.freeze({
      kind,
      quantumCount: timings.length,
      primitiveCount: sum(descriptors.map((entry) => entry.primitiveCount)),
      wallMilliseconds: sum(timings.map((entry) => entry.wallMilliseconds)),
      submitThroughDrainMilliseconds:
        sum(timings.map((entry) => entry.submitThroughDrainMilliseconds)),
      actualIdleMilliseconds:
        sum(timings.map((entry) => entry.actualIdleMilliseconds)),
    });
  });
  return Object.freeze({
    schemaVersion: 1,
    kind: "ace-opt-0010-planner-trace-summary",
    mode: attribution.mode,
    totals: validated,
    byKind: Object.freeze(byKind),
  });
}

type PhysicalDispatchSeed = Omit<
  AceOpt0010PlannerPhysicalDispatch,
  "indexInQuantum"
>;

function finalizePhysicalDispatches(
  seeds: readonly PhysicalDispatchSeed[],
): readonly AceOpt0010PlannerPhysicalDispatch[] {
  return Object.freeze(seeds.map((seed, indexInQuantum) => Object.freeze({
    indexInQuantum,
    ...seed,
    workgroups: Object.freeze([...seed.workgroups]) as readonly [
      number,
      number,
      number,
    ],
  })));
}

function physical(
  pipelineIdentity: string,
  dispatchIdentity: string,
  workgroupsX: number,
  workgroupsY: number,
): PhysicalDispatchSeed {
  return Object.freeze({
    pipelineIdentity,
    dispatchIdentity,
    workgroups: Object.freeze([workgroupsX, workgroupsY, 1] as const),
  });
}

function createEmbeddingPhysicalDispatches(
  rows: 1 | 2,
  productionLabel: string,
): readonly AceOpt0010PlannerPhysicalDispatch[] {
  const plan = planAceEmbedding({
    tokenCount: rows,
    width: ACE_PLANNER_QWEN3_CONFIG.hiddenSize,
    vocabularySize: ACE_PLANNER_QWEN3_CONFIG.vocabularySize,
  }, ACE_PLANNER_EMBEDDING_ROW_PARTS);
  return finalizePhysicalDispatches(plan.shards.map((shard) => {
    const suffix = `rows-${shard.firstRow}-${shard.firstRow + shard.rowCount}`;
    return physical(
      `embedding/raw-fp16/${rows}x${plan.width}/${suffix}`,
      `${productionLabel}-embed-tokens-${suffix}`,
      plan.workgroupsX,
      plan.workgroupsY,
    );
  }));
}

function createLayerPhysicalDispatches(
  rows: 1 | 2,
  cacheCapacity: number,
  productionLabel: string,
  layer: number,
): readonly AceOpt0010PlannerPhysicalDispatch[] {
  const config = ACE_PLANNER_QWEN3_CONFIG;
  const label = `${productionLabel}-layer-${layer}`;
  const seeds: PhysicalDispatchSeed[] = [];
  const rms = (suffix: string, rmsRows: number, width: number): void => {
    const plan = planAceRmsNorm({ rows: rmsRows, width, epsilon: 1e-6 });
    seeds.push(physical(
      `rmsnorm/raw-fp16/${rmsRows}x${width}/epsilon-0.000001`,
      `${label}-${suffix}`,
      plan.workgroupsX,
      plan.workgroupsY,
    ));
  };
  const gemm = (suffix: string, inner: number, columns: number): void => {
    seeds.push(...createGemmPhysicalSeeds(
      `${label}-${suffix}`,
      rows,
      inner,
      columns,
    ));
  };
  const headTransform = (
    suffix: string,
    operation: "split-heads" | "merge-heads",
    heads: number,
  ): void => {
    const plan = planAceHeadTransform({
      batch: rows,
      tokens: 1,
      heads,
      headDimension: config.headDimension,
    });
    seeds.push(physical(
      `head-transform/raw-fp16/${operation}/${rows}x1x${heads}x${config.headDimension}`,
      `${label}-${suffix}`,
      plan.workgroupsX,
      plan.workgroupsY,
    ));
  };
  const rope = (suffix: string, heads: number): void => {
    const plan = planAceBatchedRope({
      batch: rows,
      heads,
      tokens: 1,
      headDimension: config.headDimension,
    });
    seeds.push(physical(
      `batched-rope/raw-fp16/${rows}x${heads}x1x${config.headDimension}`,
      `${label}-${suffix}`,
      plan.workgroupsX,
      plan.workgroupsY,
    ));
  };
  const tensor = (
    suffix: string,
    operation: "residual-add" | "swiglu",
    width: number,
  ): void => {
    const plan = planAceTransformerTensor({ batch: rows, tokens: 1, width });
    seeds.push(physical(
      `transformer/raw-fp16/${operation}/${rows}x1x${width}`,
      `${label}-${suffix}`,
      plan.workgroupsX,
      plan.workgroupsY,
    ));
  };

  rms("input-norm", rows, config.hiddenSize);
  gemm("query-projection", config.hiddenSize, config.queryHeads * config.headDimension);
  gemm("key-projection", config.hiddenSize, config.keyValueHeads * config.headDimension);
  gemm("value-projection", config.hiddenSize, config.keyValueHeads * config.headDimension);
  headTransform("split-query-heads", "split-heads", config.queryHeads);
  headTransform("split-key-heads", "split-heads", config.keyValueHeads);
  headTransform("split-value-heads", "split-heads", config.keyValueHeads);
  rms("query-norm", rows * config.queryHeads, config.headDimension);
  rms("key-norm", rows * config.keyValueHeads, config.headDimension);
  rope("query-rope", config.queryHeads);
  rope("key-rope", config.keyValueHeads);
  const cache = planAceKvCacheWrite({
    batch: rows,
    keyValueHeads: config.keyValueHeads,
    appendTokens: 1,
    cacheCapacity,
    headDimension: config.headDimension,
  });
  seeds.push(physical(
    `kv-cache-write/raw-fp16/${rows}x${config.keyValueHeads}x1x${cacheCapacity}x${config.headDimension}`,
    `${label}-cache-append`,
    cache.workgroupsX,
    cache.workgroupsY,
  ));
  const attention = planAceAttention({
    batch: rows,
    queryHeads: config.queryHeads,
    keyValueHeads: config.keyValueHeads,
    queryTokens: 1,
    keyValueTokens: cacheCapacity,
    headDimension: config.headDimension,
    mode: "causal",
    keyValidity: "causal-per-key",
  });
  seeds.push(physical(
    `attention/raw-fp16/${rows}x${config.queryHeads}x${config.keyValueHeads}x1x${cacheCapacity}x${config.headDimension}/causal-per-key`,
    `${label}-causal-attention`,
    attention.workgroupsX,
    attention.workgroupsY,
  ));
  headTransform("merge-attention-heads", "merge-heads", config.queryHeads);
  gemm("attention-output-projection", config.queryHeads * config.headDimension, config.hiddenSize);
  tensor("attention-residual", "residual-add", config.hiddenSize);
  rms("post-attention-norm", rows, config.hiddenSize);
  gemm("gate-projection", config.hiddenSize, config.intermediateSize);
  gemm("up-projection", config.hiddenSize, config.intermediateSize);
  tensor("swiglu", "swiglu", config.intermediateSize);
  gemm("down-projection", config.intermediateSize, config.hiddenSize);
  tensor("mlp-residual", "residual-add", config.hiddenSize);
  if (seeds.length !== LAYER_PRIMITIVE_COUNT) {
    throw new Error(
      `ACE OPT-0010 derived ${seeds.length} layer dispatches; expected ${LAYER_PRIMITIVE_COUNT}`,
    );
  }
  return finalizePhysicalDispatches(seeds);
}

function createGemmPhysicalSeeds(
  label: string,
  rows: number,
  inner: number,
  columns: number,
): readonly PhysicalDispatchSeed[] {
  const plan = planAceTiledGemm({ rows, inner, columns });
  const pipelineIdentity =
    `gemm/raw-fp16/source-row-major/${rows}x${inner}x${columns}/no-bias`;
  return Object.freeze(plan.outputRanges.map((range, rangeIndex) => physical(
    pipelineIdentity,
    `${label}-range-${rangeIndex}`,
    range.workgroupCount,
    1,
  )));
}

function createFinalNormPhysicalDispatches(
  rows: 1 | 2,
  productionLabel: string,
): readonly AceOpt0010PlannerPhysicalDispatch[] {
  const plan = planAceRmsNorm({
    rows,
    width: ACE_PLANNER_QWEN3_CONFIG.hiddenSize,
    epsilon: ACE_PLANNER_QWEN3_CONFIG.rmsNormEpsilon,
  });
  return finalizePhysicalDispatches([physical(
    `rmsnorm/raw-fp16/${rows}x${plan.width}/epsilon-0.000001`,
    `${productionLabel}-final-norm`,
    plan.workgroupsX,
    plan.workgroupsY,
  )]);
}

function createGatherPhysicalDispatches(
  rows: 1 | 2,
  productionLabel: string,
): readonly AceOpt0010PlannerPhysicalDispatch[] {
  const plan = planAceGatherRows({
    outer: rows,
    sourceRows: 1,
    outputRows: 1,
    width: ACE_PLANNER_QWEN3_CONFIG.hiddenSize,
  });
  return finalizePhysicalDispatches([physical(
    `gather-rows/raw-fp16/${rows}x1x1x${plan.width}`,
    `${productionLabel}-last-physical-rows`,
    plan.workgroupsX,
    plan.workgroupsY,
  )]);
}

function createHeadPhysicalDispatchGroups(
  rows: 1 | 2,
  productionLabel: string,
): readonly Readonly<{
  readonly shardIndices: readonly number[];
  readonly dispatches: readonly AceOpt0010PlannerPhysicalDispatch[];
}>[] {
  const groups: Array<{
    shardIndices: number[];
    seeds: PhysicalDispatchSeed[];
    scheduledMultiplyAdds: number;
  }> = [];
  let current: {
    shardIndices: number[];
    seeds: PhysicalDispatchSeed[];
    scheduledMultiplyAdds: number;
  } = { shardIndices: [], seeds: [], scheduledMultiplyAdds: 0 };
  const flush = (): void => {
    if (current.shardIndices.length > 0) groups.push(current);
    current = { shardIndices: [], seeds: [], scheduledMultiplyAdds: 0 };
  };
  for (let shardIndex = 0; shardIndex < ACE_PLANNER_EMBEDDING_ROW_PARTS.length; shardIndex += 1) {
    const shard = ACE_PLANNER_EMBEDDING_ROW_PARTS[shardIndex]!;
    const plan = planAceTiledGemm({
      rows,
      inner: ACE_PLANNER_QWEN3_CONFIG.hiddenSize,
      columns: shard.rowCount,
    });
    if (plan.outputRanges.length !== 1) {
      throw new Error("ACE OPT-0010 tied-head shard no longer has one range");
    }
    const work = plan.outputRanges[0]!.multiplyAdds;
    if (
      current.scheduledMultiplyAdds > 0 &&
      current.scheduledMultiplyAdds + work >
        ACE_GEMM_MAX_MULTIPLY_ADDS_PER_RANGE
    ) flush();
    current.shardIndices.push(shardIndex);
    current.seeds.push(...createGemmPhysicalSeeds(
      `${productionLabel}-lm-head-rows-${shard.firstRow}`,
      rows,
      ACE_PLANNER_QWEN3_CONFIG.hiddenSize,
      shard.rowCount,
    ));
    current.scheduledMultiplyAdds += work;
  }
  flush();
  return Object.freeze(groups.map((group) => Object.freeze({
    shardIndices: Object.freeze(group.shardIndices),
    dispatches: finalizePhysicalDispatches(group.seeds),
  })));
}

function createHeadSlicePlan(
  shardIndex: number,
  globalFirstRow: number,
  localFirstRow: number,
  rowCount: number,
): AceOpt0010HeadSlicePlan {
  const rows = 2 as const;
  const inner = ACE_PLANNER_QWEN3_CONFIG.hiddenSize;
  const plan = planAceTiledGemm({ rows, inner, columns: rowCount });
  const logicalMultiplyAdds = rows * inner * rowCount;
  const scheduledMultiplyAdds = sum(
    plan.outputRanges.map((range) => range.multiplyAdds),
  );
  return Object.freeze({
    shardIndex,
    globalFirstRow,
    localFirstRow,
    rowCount,
    rows,
    inner,
    columns: rowCount,
    logicalWeightTrafficBytes: inner * rowCount * FP16_BYTES,
    gemmActivationBytes: rows * (inner + rowCount) * FP16_BYTES,
    rawLogitBytes: rows * rowCount * FP16_BYTES,
    logicalMultiplyAdds,
    logicalFloatingPointOperations: logicalMultiplyAdds * 2,
    scheduledMultiplyAdds,
    scheduledFloatingPointOperations: scheduledMultiplyAdds * 2,
    outputRanges: Object.freeze(plan.outputRanges.map((range, index) =>
      Object.freeze({ index, ...range }))),
  });
}

function headMetrics(
  slices: readonly AceOpt0010HeadSlicePlan[],
): AceOpt0010HeadMetrics {
  return Object.freeze({
    logicalWeightTrafficBytes:
      sum(slices.map((slice) => slice.logicalWeightTrafficBytes)),
    gemmActivationBytes: sum(slices.map((slice) => slice.gemmActivationBytes)),
    rawLogitBytes: sum(slices.map((slice) => slice.rawLogitBytes)),
    logicalMultiplyAdds: sum(slices.map((slice) => slice.logicalMultiplyAdds)),
    logicalFloatingPointOperations:
      sum(slices.map((slice) => slice.logicalFloatingPointOperations)),
    scheduledMultiplyAdds:
      sum(slices.map((slice) => slice.scheduledMultiplyAdds)),
    scheduledFloatingPointOperations:
      sum(slices.map((slice) => slice.scheduledFloatingPointOperations)),
  });
}

function subtractHeadMetrics(
  full: AceOpt0010HeadMetrics,
  restricted: AceOpt0010HeadMetrics,
): AceOpt0010HeadMetrics {
  return Object.freeze({
    logicalWeightTrafficBytes:
      full.logicalWeightTrafficBytes - restricted.logicalWeightTrafficBytes,
    gemmActivationBytes:
      full.gemmActivationBytes - restricted.gemmActivationBytes,
    rawLogitBytes: full.rawLogitBytes - restricted.rawLogitBytes,
    logicalMultiplyAdds: full.logicalMultiplyAdds - restricted.logicalMultiplyAdds,
    logicalFloatingPointOperations:
      full.logicalFloatingPointOperations -
      restricted.logicalFloatingPointOperations,
    scheduledMultiplyAdds:
      full.scheduledMultiplyAdds - restricted.scheduledMultiplyAdds,
    scheduledFloatingPointOperations:
      full.scheduledFloatingPointOperations -
      restricted.scheduledFloatingPointOperations,
  });
}

function createSemanticHeadOpportunity(
  readback: AceOpt0010PlannerReadbackPlan,
  restrictedSlices: readonly AceOpt0010HeadSlicePlan[],
  eosShardIndex: number,
  eosShard: Readonly<{ firstRow: number }>,
): AceOpt0010SemanticHeadOpportunity {
  const fullSlices = ACE_PLANNER_EMBEDDING_ROW_PARTS.map((shard, shardIndex) =>
    createHeadSlicePlan(shardIndex, shard.firstRow, 0, shard.rowCount));
  const fullHead = headMetrics(fullSlices);
  const restrictedMetrics = headMetrics(restrictedSlices);
  if (fullHead.rawLogitBytes !== readback.rawLogitBytes) {
    throw new Error("ACE OPT-0010 full-head metrics disagree with readback");
  }
  const eos = createHeadSlicePlan(
    eosShardIndex,
    ACE_QWEN_IM_END_TOKEN_ID,
    ACE_QWEN_IM_END_TOKEN_ID - eosShard.firstRow,
    1,
  );
  return Object.freeze({
    firstTokenId: ACE_PLANNER_SEMANTIC_FIRST_TOKEN_ID,
    tokenCount: ACE_PLANNER_SEMANTIC_CODE_COUNT,
    fullHead,
    restrictedCodeHead: Object.freeze({
      ...restrictedMetrics,
      intersections: Object.freeze(restrictedSlices),
    }),
    avoidablePerToken: subtractHeadMetrics(fullHead, restrictedMetrics),
    terminalEos: Object.freeze({
      tokenId: ACE_QWEN_IM_END_TOKEN_ID,
      shardIndex: eosShardIndex,
      globalFirstRow: ACE_QWEN_IM_END_TOKEN_ID,
      localFirstRow: ACE_QWEN_IM_END_TOKEN_ID - eosShard.firstRow,
      rowCount: 1 as const,
      optionalHeadRow: eos,
      terminalOnly: true as const,
      includedInRestrictedCodeHead: false as const,
      tiedEmbeddingRemainsFullyResident: true as const,
    }),
  });
}

function gemmTotals(rows: 1 | 2, shapes: readonly GemmShape[]): Readonly<{
  weightBytes: number;
  activationBytes: number;
  logicalMultiplyAdds: number;
  scheduledMultiplyAdds: number;
}> {
  let weightBytes = 0;
  let activationBytes = 0;
  let logicalMultiplyAdds = 0;
  let scheduledMultiplyAdds = 0;
  for (const shape of shapes) {
    const plan = planAceTiledGemm({ rows, ...shape });
    weightBytes += shape.inner * shape.columns * FP16_BYTES;
    activationBytes +=
      rows * (shape.inner + shape.columns) * FP16_BYTES;
    logicalMultiplyAdds += rows * shape.inner * shape.columns;
    scheduledMultiplyAdds += sum(plan.outputRanges.map((range) =>
      range.multiplyAdds));
  }
  return Object.freeze({
    weightBytes,
    activationBytes,
    logicalMultiplyAdds,
    scheduledMultiplyAdds,
  });
}

function createAttentionAttribution(
  rows: 1 | 2,
  validKeyValueTokens: number,
  scheduledKeyValueCapacity: number,
): AceOpt0010PlannerAttentionAttribution {
  const queryElements =
    rows *
    ACE_PLANNER_QWEN3_CONFIG.queryHeads *
    ACE_PLANNER_QWEN3_CONFIG.headDimension;
  const keyValuePairElementsPerToken =
    rows *
    ACE_PLANNER_QWEN3_CONFIG.keyValueHeads *
    ACE_PLANNER_QWEN3_CONFIG.headDimension *
    2;
  const multiplyAddsPerToken =
    rows *
    ACE_PLANNER_QWEN3_CONFIG.queryHeads *
    ACE_PLANNER_QWEN3_CONFIG.headDimension *
    2;
  const perLayer = Object.freeze({
    queryElements,
    logicalKeyValuePairElements:
      keyValuePairElementsPerToken * validKeyValueTokens,
    scheduledKeyValuePairElements:
      keyValuePairElementsPerToken * scheduledKeyValueCapacity,
    logicalKeyValuePairBytes:
      keyValuePairElementsPerToken * validKeyValueTokens * FP16_BYTES,
    scheduledKeyValuePairBytes:
      keyValuePairElementsPerToken * scheduledKeyValueCapacity * FP16_BYTES,
    logicalKeyValidityBytes: rows * validKeyValueTokens * U32_BYTES,
    scheduledKeyValidityBytes:
      rows * scheduledKeyValueCapacity * U32_BYTES,
    logicalMultiplyAdds: multiplyAddsPerToken * validKeyValueTokens,
    scheduledMultiplyAdds:
      multiplyAddsPerToken * scheduledKeyValueCapacity,
    logicalFloatingPointOperations:
      multiplyAddsPerToken * validKeyValueTokens * 2,
    scheduledFloatingPointOperations:
      multiplyAddsPerToken * scheduledKeyValueCapacity * 2,
  });
  const layerCount = ACE_PLANNER_QWEN3_CONFIG.layerCount;
  return Object.freeze({
    queryRows: rows,
    queryHeads: ACE_PLANNER_QWEN3_CONFIG.queryHeads,
    keyValueHeads: ACE_PLANNER_QWEN3_CONFIG.keyValueHeads,
    headDimension: ACE_PLANNER_QWEN3_CONFIG.headDimension,
    validKeyValueTokens,
    scheduledKeyValueCapacity,
    perLayer,
    allLayers: Object.freeze({
      logicalKeyValuePairBytes:
        perLayer.logicalKeyValuePairBytes * layerCount,
      scheduledKeyValuePairBytes:
        perLayer.scheduledKeyValuePairBytes * layerCount,
      logicalKeyValidityBytes:
        perLayer.logicalKeyValidityBytes * layerCount,
      scheduledKeyValidityBytes:
        perLayer.scheduledKeyValidityBytes * layerCount,
      logicalMultiplyAdds: perLayer.logicalMultiplyAdds * layerCount,
      scheduledMultiplyAdds: perLayer.scheduledMultiplyAdds * layerCount,
      logicalFloatingPointOperations:
        perLayer.logicalFloatingPointOperations * layerCount,
      scheduledFloatingPointOperations:
        perLayer.scheduledFloatingPointOperations * layerCount,
    }),
  });
}

function headTotals(
  rows: 1 | 2,
  shardIndices: readonly number[],
): ReturnType<typeof gemmTotals> {
  return gemmTotals(rows, shardIndices.map((index) => ({
    inner: ACE_PLANNER_QWEN3_CONFIG.hiddenSize,
    columns: ACE_PLANNER_EMBEDDING_ROW_PARTS[index]!.rowCount,
  })));
}

function createReadbackPlan(rows: 1 | 2): AceOpt0010PlannerReadbackPlan {
  let cursor = 0;
  const shards = ACE_PLANNER_EMBEDDING_ROW_PARTS.map((part, shardIndex) => {
    cursor = align(cursor, STORAGE_ALIGNMENT);
    const byteLength = rows * part.rowCount * FP16_BYTES;
    const shard = Object.freeze({
      shardIndex,
      globalFirstRow: part.firstRow,
      localFirstRow: 0 as const,
      rowCount: part.rowCount,
      byteOffset: cursor,
      byteLength,
    });
    cursor += align(byteLength, 4);
    return shard;
  });
  cursor = align(cursor, STORAGE_ALIGNMENT);
  const writeStatusByteOffset = cursor;
  const writeStatusByteLength = rows * U32_BYTES;
  cursor += writeStatusByteLength;
  const copyCommands: AceOpt0010PlannerCopyCommand[] = shards.map(
    (shard, index) => Object.freeze({
      index,
      sourceBufferLabel: `logits-${shard.shardIndex}`,
      shardIndex: shard.shardIndex,
      sourceOffset: 0,
      destinationBufferLabel: "ace-planner-logit-readback" as const,
      destinationOffset: shard.byteOffset,
      copiedBytes: align(shard.byteLength, 4),
    }),
  );
  copyCommands.push(Object.freeze({
    index: copyCommands.length,
    sourceBufferLabel: "write-status",
    shardIndex: null,
    sourceOffset: 0,
    destinationBufferLabel: "ace-planner-logit-readback" as const,
    destinationOffset: writeStatusByteOffset,
    copiedBytes: writeStatusByteLength,
  }));
  return Object.freeze({
    rows,
    shards: Object.freeze(shards),
    rawLogitBytes:
      rows * ACE_PLANNER_QWEN3_CONFIG.vocabularySize * FP16_BYTES,
    writeStatusByteOffset,
    writeStatusByteLength,
    bufferBytes: align(cursor, STORAGE_ALIGNMENT),
    copyCommands: Object.freeze(copyCommands),
  });
}

function requireProductionQuantumTag(
  actual: AceOpt0010PlannerProductionQuantumTag,
  expected: AceOpt0010PlannerProductionQuantumTag,
  label: string,
): void {
  if (
    actual.id !== expected.id ||
    actual.logicalId !== expected.logicalId ||
    actual.kind !== expected.kind ||
    actual.layer !== expected.layer ||
    actual.primitiveCount !== expected.primitiveCount
  ) {
    throw new Error(`ACE OPT-0010 ${label} production tag changed`);
  }
}

function sameCopyCommand(
  actual: AceOpt0010PlannerCopyCommand,
  expected: AceOpt0010PlannerCopyCommand,
): boolean {
  return actual.index === expected.index &&
    actual.sourceBufferLabel === expected.sourceBufferLabel &&
    actual.shardIndex === expected.shardIndex &&
    actual.sourceOffset === expected.sourceOffset &&
    actual.destinationBufferLabel === expected.destinationBufferLabel &&
    actual.destinationOffset === expected.destinationOffset &&
    actual.copiedBytes === expected.copiedBytes;
}

function samePhysicalDispatch(
  actual: AceOpt0010PlannerPhysicalDispatch,
  expected: AceOpt0010PlannerPhysicalDispatch,
): boolean {
  return actual.indexInQuantum === expected.indexInQuantum &&
    actual.pipelineIdentity === expected.pipelineIdentity &&
    actual.dispatchIdentity === expected.dispatchIdentity &&
    actual.workgroups.length === 3 &&
    actual.workgroups.every((value, index) => value === expected.workgroups[index]);
}

function requireUnitCounts(
  value: Readonly<{
    commandBufferCount: number;
    submissionCount: number;
    queueDrainCount: number;
    completedIdleCount: number;
    requestedIdleMilliseconds: number;
  }>,
  label: string,
): void {
  for (const [name, actual] of Object.entries({
    commandBufferCount: value.commandBufferCount,
    submissionCount: value.submissionCount,
    queueDrainCount: value.queueDrainCount,
    completedIdleCount: value.completedIdleCount,
  })) {
    requireCount(actual, 1, label + " " + name);
  }
  requireCount(
    value.requestedIdleMilliseconds,
    ACE_COOPERATIVE_GPU_IDLE_MILLISECONDS,
    label + " requested idle",
  );
}

function requireTimeline(values: readonly number[], label: string): void {
  let previous = -Infinity;
  for (const value of values) {
    if (!Number.isFinite(value) || value < 0 || value < previous) {
      throw new RangeError("ACE OPT-0010 " + label + " timeline is invalid");
    }
    previous = value;
  }
}

function requirePositiveInterval(
  startedAt: number,
  endedAt: number,
  label: string,
): void {
  if (!Number.isFinite(startedAt) || !Number.isFinite(endedAt) || endedAt <= startedAt) {
    throw new RangeError(`ACE OPT-0010 ${label} duration must be positive`);
  }
}

function requireCount(actual: number, expected: number, label: string): void {
  if (!Number.isSafeInteger(actual) || actual !== expected) {
    throw new Error(
      "ACE OPT-0010 " + label + " is " + actual + "; expected " + expected,
    );
  }
}

function requireNonNegativeSafeInteger(actual: number, label: string): void {
  if (!Number.isSafeInteger(actual) || actual < 0) {
    throw new Error(`ACE OPT-0010 ${label} is not a non-negative safe integer`);
  }
}

function requireSame(actual: number, expected: number, label: string): void {
  const scale = Math.max(1, Math.abs(actual), Math.abs(expected));
  if (Math.abs(actual - expected) > scale * 1e-12) {
    throw new Error("ACE OPT-0010 " + label + " timestamp changed");
  }
}

function sum(values: readonly number[]): number {
  return values.reduce((total, value) => total + value, 0);
}

function align(value: number, alignment: number): number {
  return Math.ceil(value / alignment) * alignment;
}
