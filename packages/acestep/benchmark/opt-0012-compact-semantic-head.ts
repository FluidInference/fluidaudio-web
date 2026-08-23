import {
  ACE_PACKAGE_ALIGNMENT_BYTES,
  type AceTensorLayout,
} from "../src/model/manifest.js";
import {
  ACE_PLANNER_SEMANTIC_CODE_COUNT,
  ACE_PLANNER_SEMANTIC_FIRST_TOKEN_ID,
} from "../src/runtime/planner.js";
import {
  ACE_PLANNER_SOFTMAX_ACCEPTANCE,
  applyAcePlannerRepetitionPenalty,
  applyAcePlannerTopK,
  combineAcePlannerCfgLogits,
  createAcePlannerBrowserSamplingWeights,
  createAcePlannerBrowserTopPKeep,
  maskAcePlannerLogits,
  type AcePlannerAllowedTokens,
  type AcePlannerSamplingParameters,
} from "../src/runtime/planner-sampling.js";
import {
  aceCategoricalTokenFromWord,
  aceRandomWord,
  type AceSeed,
} from "../src/runtime/seed.js";
import { ACE_QWEN_IM_END_TOKEN_ID } from "../src/tokenizer/qwen-bpe.js";
import {
  ACE_PLANNER_EMBEDDING_ROW_PARTS,
  ACE_PLANNER_PHYSICAL_TENSOR_NAMES,
} from "../src/webgpu/planner-model.js";
import {
  planAceTiledGemm,
  type AceGemmOutputRange,
  type AceGemmWeightLayout,
} from "../src/webgpu/kernels/gemm.js";
import { ACE_PLANNER_QWEN3_CONFIG } from "../src/webgpu/qwen3.js";

const FP16_BYTES = Uint16Array.BYTES_PER_ELEMENT;
const U32_BYTES = Uint32Array.BYTES_PER_ELEMENT;
const COPY_ALIGNMENT_BYTES = 4;
const CFG_ROWS = 2;
const CONDITIONAL_ROW = 0;
const UNCONDITIONAL_ROW = 1;
const RAW_FP16_PROFILE = "raw-fp16";
const RAW_FP16_MANIFEST_LAYOUT: AceTensorLayout = "row-shard-axis0";
const SOURCE_ROW_MAJOR_LAYOUT: AceGemmWeightLayout = "source-row-major";
const FP16_DTYPE = "float16";
const SHARD_MAJOR_ROW_MAJOR = "shard-major-row-major";
const CONTRACT_ID = "opt-0012-compact-semantic-head-v1";
const AUTHENTICATED_VOCABULARY_SIZE = 217_204;
const AUTHENTICATED_HIDDEN_SIZE = 1_024;
const AUTHENTICATED_SEMANTIC_FIRST_TOKEN_ID = 151_669;
const AUTHENTICATED_SEMANTIC_CODE_COUNT = 64_000;
const AUTHENTICATED_EOS_TOKEN_ID = 151_645;

const AUTHENTICATED_TIED_SHARD_PARTITION = Object.freeze([
  Object.freeze({ firstRow: 0, rowCount: 49_152 }),
  Object.freeze({ firstRow: 49_152, rowCount: 49_152 }),
  Object.freeze({ firstRow: 98_304, rowCount: 49_152 }),
  Object.freeze({ firstRow: 147_456, rowCount: 49_152 }),
  Object.freeze({ firstRow: 196_608, rowCount: 20_596 }),
]);

const AUTHENTICATED_TIED_TENSOR_NAMES = Object.freeze([
  "planner.model.embed_tokens.weight.rows-000000-049152",
  "planner.model.embed_tokens.weight.rows-049152-098304",
  "planner.model.embed_tokens.weight.rows-098304-147456",
  "planner.model.embed_tokens.weight.rows-147456-196608",
  "planner.model.embed_tokens.weight.rows-196608-217204",
]);

/** Frozen identities for every production/transitive source used by this core. */
export const ACE_OPT_0012_CORE_SOURCE_IDENTITIES = Object.freeze({
  "src/model/manifest.ts":
    "77213ed8c096d0e2b49cd4aeb7c0eb96f63c12671173d7d83b7b6eb6331916b1",
  "src/runtime/planner.ts":
    "35c1effda1a52e6a73fabfb467a2ac7fd015197a4a6e494d360f9db78b7950b1",
  "src/runtime/planner-sampling.ts":
    "67055acfbb96e10682092e5d0ccfa9a5d822fd708a091fe46f7f47458226d0f3",
  "src/runtime/seed.ts":
    "157f349808aff5b4e64eb82e36f7b3ace7483845c78336e94ee6707557ee557e",
  "src/tokenizer/qwen-bpe.ts":
    "5a83fa16a178621e8b9de457800564e1c4b832bc70e39d627b39cd0678064737",
  "src/webgpu/planner-model.ts":
    "673a5a3d45a749c12337cc6186212084a4b05c6ffde7f02b782a7e033113cce6",
  "src/webgpu/kernels/gemm.ts":
    "8922c2cebe36186b7cbf173e69126dd45a01fbc79a2cb2f84c34090880268928",
  "src/webgpu/kernels/correctness-utils.ts":
    "f727d347e666e4f4ebdedc7acedcc6b1cdde371850b9fc83a2c214cb83290018",
  "src/webgpu/qwen3.ts":
    "bb982aeb380c52ea842183b1649a6674203c18a421c17ea07f68d7668309158a",
});

export const ACE_OPT_0012_NEGATIVE_INFINITY_U32 = 0xff80_0000;

export const ACE_OPT_0012_SAMPLING_PARAMETERS:
Readonly<AcePlannerSamplingParameters> = Object.freeze({
  temperature: 0.85,
  guidanceScale: 2,
  topK: 0,
  topP: 0.9,
  repetitionPenalty: 1,
});

export const ACE_OPT_0012_PRE_CFG_ALLOWED_TOKENS:
Readonly<AcePlannerAllowedTokens> = Object.freeze({
  kind: "range",
  firstTokenId: AUTHENTICATED_SEMANTIC_FIRST_TOKEN_ID,
  tokenCount: AUTHENTICATED_SEMANTIC_CODE_COUNT,
  additionalTokenIds: Object.freeze([AUTHENTICATED_EOS_TOKEN_ID]),
});

export const ACE_OPT_0012_REGULAR_ALLOWED_TOKENS:
Readonly<AcePlannerAllowedTokens> = Object.freeze({
  kind: "range",
  firstTokenId: AUTHENTICATED_SEMANTIC_FIRST_TOKEN_ID,
  tokenCount: AUTHENTICATED_SEMANTIC_CODE_COUNT,
});

export const ACE_OPT_0012_FORCED_EOS_ALLOWED_TOKENS:
Readonly<AcePlannerAllowedTokens> = Object.freeze({
  kind: "ids",
  tokenIds: Object.freeze([AUTHENTICATED_EOS_TOKEN_ID]),
});

export type AceOpt0012SemanticState = "regular-code" | "forced-eos";

export interface AceOpt0012SourceShardBinding {
  readonly shardIndex: number;
  readonly tensorName: string;
  readonly firstRow: number;
  readonly rowCount: number;
  /** Offset and length of the complete tensor part within its GPU buffer. */
  readonly bindingByteOffset: number;
  readonly bindingByteLength: number;
  readonly bufferByteLength: number;
  /** Manifest storage is an axis-0 part; each bound part is source-row-major. */
  readonly manifestLayout: AceTensorLayout;
  readonly kernelWeightLayout: AceGemmWeightLayout;
  readonly dtype: string;
}

export const ACE_OPT_0012_RAW_FP16_TIED_WEIGHT_SHARDS:
readonly AceOpt0012SourceShardBinding[] = Object.freeze([
  authenticatedSourceShard(0, 0, 49_152, 100_663_296),
  authenticatedSourceShard(1, 49_152, 49_152, 100_663_296),
  authenticatedSourceShard(2, 98_304, 49_152, 100_663_296),
  authenticatedSourceShard(3, 147_456, 49_152, 100_663_296),
  authenticatedSourceShard(4, 196_608, 20_596, 42_180_608),
]);

export interface AceOpt0012PlanRequest {
  readonly phase: string;
  readonly state: string;
  readonly modelProfile: string;
  readonly rows: number;
  readonly conditionalRow: number;
  readonly unconditionalRow: number;
  readonly vocabularySize: number;
  readonly hiddenSize: number;
  readonly manifestWeightLayout: string;
  readonly kernelWeightLayout: string;
  readonly weightDtype: string;
  readonly outputDtype: string;
  readonly physicalReadbackLayout: string;
  readonly softmaxOracleId: string;
  readonly sampling: AcePlannerSamplingParameters;
  readonly preCfgAllowedTokens: AcePlannerAllowedTokens;
  readonly allowedTokens: AcePlannerAllowedTokens;
  readonly acceptedRegularCodeCount: number;
  readonly requestedRegularCodeCount: number;
  readonly tiedWeightShards: readonly AceOpt0012SourceShardBinding[];
}

export interface AceOpt0012RowIntersection {
  readonly shardIndex: number;
  readonly globalFirstRow: number;
  readonly globalLastRow: number;
  readonly localFirstRow: number;
  readonly rowCount: number;
}

export interface AceOpt0012HeadSlice extends AceOpt0012RowIntersection {
  readonly sourceTensorName: string;
  readonly sourceBindingByteOffset: number;
  readonly sourceBindingByteLength: number;
  readonly sourceBindingByteEnd: number;
  readonly sourceOwnerByteOffset: number;
  readonly sourceOwnerByteLength: number;
  readonly sourceOwnerByteEnd: number;
  readonly logicalWeightTrafficBytes: number;
  readonly logicalMultiplyAdds: number;
  readonly scheduledMultiplyAdds: number;
  readonly workgroupCount: number;
  readonly outputRanges: readonly Readonly<AceGemmOutputRange>[];
  readonly rawLogitBytes: number;
}

export interface AceOpt0012ReadbackCopy {
  readonly index: number;
  readonly kind: "logits" | "write-status";
  readonly shardIndex: number | null;
  readonly sourceByteOffset: number;
  readonly destinationByteOffset: number;
  readonly byteLength: number;
}

export interface AceOpt0012LogicalSpan {
  readonly shardIndex: number;
  readonly physicalRow: 0 | 1;
  readonly sourceByteOffset: number;
  readonly byteLength: number;
  readonly destinationCandidateOffset: number;
  readonly candidateCount: number;
  readonly globalFirstTokenId: number;
}

export interface AceOpt0012CompactReadbackPlan {
  readonly physicalLayout: typeof SHARD_MAJOR_ROW_MAJOR;
  readonly copies: readonly AceOpt0012ReadbackCopy[];
  readonly logicalSpans: readonly AceOpt0012LogicalSpan[];
  readonly rawLogitBytes: number;
  readonly writeStatusByteOffset: number;
  readonly writeStatusByteLength: number;
  readonly usedBytes: number;
  readonly alignmentPaddingBytes: number;
  readonly allocationBytes: number;
}

export interface AceOpt0012CompactSemanticHeadPlan {
  readonly contractId: typeof CONTRACT_ID;
  readonly phase: "semantic-m2";
  readonly state: AceOpt0012SemanticState;
  readonly modelProfile: typeof RAW_FP16_PROFILE;
  readonly rows: typeof CFG_ROWS;
  readonly conditionalRow: typeof CONDITIONAL_ROW;
  readonly unconditionalRow: typeof UNCONDITIONAL_ROW;
  readonly vocabularySize: number;
  readonly hiddenSize: number;
  readonly firstCandidateTokenId: number;
  readonly candidateCount: number;
  readonly lastCandidateTokenId: number;
  readonly sampling: Readonly<AcePlannerSamplingParameters>;
  readonly intersections: readonly AceOpt0012RowIntersection[];
  readonly headSlices: readonly AceOpt0012HeadSlice[];
  readonly logicalWeightTrafficBytes: number;
  readonly logicalMultiplyAdds: number;
  readonly scheduledMultiplyAdds: number;
  readonly workgroupCount: number;
  readonly readback: AceOpt0012CompactReadbackPlan;
}

export interface AceOpt0012DecodedCompactLogits {
  readonly conditionalLogits: Float32Array;
  readonly unconditionalLogits: Float32Array;
  readonly writeStatus: Uint32Array;
}

export interface AceOpt0012CompactSampleInput {
  readonly plan: AceOpt0012CompactSemanticHeadPlan;
  readonly conditionalLogits: ArrayLike<number>;
  readonly unconditionalLogits: ArrayLike<number>;
  readonly seenTokenIds: readonly number[];
  readonly parameters: AcePlannerSamplingParameters;
  readonly word: number;
}

export interface AceOpt0012CompactSamplingTrace {
  readonly conditionalAllowedLogits: Float32Array;
  readonly unconditionalAllowedLogits: Float32Array;
  readonly cfgLogits: Float32Array;
  readonly finalMaskedLogits: Float32Array;
  readonly repetitionPenalizedLogits: Float32Array;
  readonly topKLogits: Float32Array;
  readonly topKGlobalTokenIds: readonly number[];
  readonly topPKeep: Uint8Array;
  readonly topPGlobalTokenIds: readonly number[];
  readonly topPLogits: Float32Array;
  readonly temperatureScaledLogits: Float32Array;
  readonly weights: Float32Array;
  readonly selectedCandidateIndex: number;
  readonly tokenId: number;
  readonly word: number;
  readonly positiveCandidateCount: number;
}

export interface AceOpt0012CompactCursorSample
extends AceOpt0012CompactSamplingTrace {
  readonly drawIndex: bigint;
  readonly drawEnd: bigint;
}

const ISSUED_PLANS = new WeakSet<object>();

/** Create the complete explicit request expected by the benchmark-only arm. */
export function createAceOpt0012PlanRequest(
  state: AceOpt0012SemanticState,
  acceptedRegularCodeCount = state === "forced-eos" ? 150 : 0,
  requestedRegularCodeCount = 150,
): AceOpt0012PlanRequest {
  return {
    phase: "semantic-m2",
    state,
    modelProfile: RAW_FP16_PROFILE,
    rows: CFG_ROWS,
    conditionalRow: CONDITIONAL_ROW,
    unconditionalRow: UNCONDITIONAL_ROW,
    vocabularySize: AUTHENTICATED_VOCABULARY_SIZE,
    hiddenSize: AUTHENTICATED_HIDDEN_SIZE,
    manifestWeightLayout: RAW_FP16_MANIFEST_LAYOUT,
    kernelWeightLayout: SOURCE_ROW_MAJOR_LAYOUT,
    weightDtype: FP16_DTYPE,
    outputDtype: FP16_DTYPE,
    physicalReadbackLayout: SHARD_MAJOR_ROW_MAJOR,
    softmaxOracleId: ACE_PLANNER_SOFTMAX_ACCEPTANCE.productionOracleId,
    sampling: ACE_OPT_0012_SAMPLING_PARAMETERS,
    preCfgAllowedTokens: ACE_OPT_0012_PRE_CFG_ALLOWED_TOKENS,
    allowedTokens: state === "regular-code"
      ? ACE_OPT_0012_REGULAR_ALLOWED_TOKENS
      : ACE_OPT_0012_FORCED_EOS_ALLOWED_TOKENS,
    acceptedRegularCodeCount,
    requestedRegularCodeCount,
    tiedWeightShards: ACE_OPT_0012_RAW_FP16_TIED_WEIGHT_SHARDS,
  };
}

/**
 * Fail-closed static plan for only the two authenticated M2 semantic states.
 * No full-head fallback is represented by this API.
 */
export function createAceOpt0012CompactSemanticHeadPlan(
  request: AceOpt0012PlanRequest,
): AceOpt0012CompactSemanticHeadPlan {
  validateProductionGeometry();
  validateEligibility(request);
  const state = request.state as AceOpt0012SemanticState;
  const firstCandidateTokenId = state === "regular-code"
    ? AUTHENTICATED_SEMANTIC_FIRST_TOKEN_ID
    : AUTHENTICATED_EOS_TOKEN_ID;
  const candidateCount = state === "regular-code"
    ? AUTHENTICATED_SEMANTIC_CODE_COUNT
    : 1;
  const intersections = deriveIntersections(
    firstCandidateTokenId,
    candidateCount,
  );
  const headSlices = Object.freeze(intersections.map((intersection) =>
    createHeadSlice(intersection, request.tiedWeightShards[intersection.shardIndex]!)));
  const readback = createReadbackPlan(
    firstCandidateTokenId,
    candidateCount,
    intersections,
  );
  const plan = Object.freeze({
    contractId: CONTRACT_ID,
    phase: "semantic-m2" as const,
    state,
    modelProfile: RAW_FP16_PROFILE,
    rows: CFG_ROWS,
    conditionalRow: CONDITIONAL_ROW,
    unconditionalRow: UNCONDITIONAL_ROW,
    vocabularySize: AUTHENTICATED_VOCABULARY_SIZE,
    hiddenSize: AUTHENTICATED_HIDDEN_SIZE,
    firstCandidateTokenId,
    candidateCount,
    lastCandidateTokenId: firstCandidateTokenId + candidateCount - 1,
    sampling: ACE_OPT_0012_SAMPLING_PARAMETERS,
    intersections,
    headSlices,
    logicalWeightTrafficBytes: sum(headSlices.map((slice) =>
      slice.logicalWeightTrafficBytes)),
    logicalMultiplyAdds: sum(headSlices.map((slice) =>
      slice.logicalMultiplyAdds)),
    scheduledMultiplyAdds: sum(headSlices.map((slice) =>
      slice.scheduledMultiplyAdds)),
    workgroupCount: sum(headSlices.map((slice) => slice.workgroupCount)),
    readback,
  });
  ISSUED_PLANS.add(plan);
  return plan;
}

/** Decode shard-major FP16 bytes into conditional/unconditional candidate rows. */
export function decodeAceOpt0012CompactFp16Readback(
  mappedBytes: ArrayBuffer,
  plan: AceOpt0012CompactSemanticHeadPlan,
): AceOpt0012DecodedCompactLogits {
  requireIssuedPlan(plan);
  if (!(mappedBytes instanceof ArrayBuffer) ||
      mappedBytes.byteLength !== plan.readback.allocationBytes) {
    throw new RangeError(
      `OPT-0012 mapped readback must be exactly ${plan.readback.allocationBytes} bytes`,
    );
  }
  const status = new Uint32Array(
    mappedBytes,
    plan.readback.writeStatusByteOffset,
    CFG_ROWS,
  );
  for (let row = 0; row < status.length; row += 1) {
    if (status[row] !== 1) {
      throw new Error(`OPT-0012 cache append failed for physical row ${row}`);
    }
  }
  const rows = [
    new Float32Array(plan.candidateCount),
    new Float32Array(plan.candidateCount),
  ] as const;
  const rowWords = [
    new Uint32Array(
      rows[CONDITIONAL_ROW].buffer,
      rows[CONDITIONAL_ROW].byteOffset,
      rows[CONDITIONAL_ROW].length,
    ),
    new Uint32Array(
      rows[UNCONDITIONAL_ROW].buffer,
      rows[UNCONDITIONAL_ROW].byteOffset,
      rows[UNCONDITIONAL_ROW].length,
    ),
  ] as const;
  for (const span of plan.readback.logicalSpans) {
    const source = new Uint16Array(
      mappedBytes,
      span.sourceByteOffset,
      span.candidateCount,
    );
    const destination = rowWords[span.physicalRow];
    for (let index = 0; index < span.candidateCount; index += 1) {
      destination[span.destinationCandidateOffset + index] =
        aceOpt0012Float16BitsToFloat32U32(source[index]!);
    }
  }
  return Object.freeze({
    conditionalLogits: rows[CONDITIONAL_ROW],
    unconditionalLogits: rows[UNCONDITIONAL_ROW],
    writeStatus: status.slice(),
  });
}

/** Reconstruct arm B's two full vectors with exact binary32 negative infinity. */
export function reconstructAceOpt0012FullPlannerLogits(
  decoded: AceOpt0012DecodedCompactLogits,
  plan: AceOpt0012CompactSemanticHeadPlan,
): readonly [Float32Array, Float32Array] {
  requireIssuedPlan(plan);
  requireCompactRows(decoded.conditionalLogits, decoded.unconditionalLogits, plan);
  if (
    !(decoded.writeStatus instanceof Uint32Array) ||
    decoded.writeStatus.length !== CFG_ROWS ||
    decoded.writeStatus[0] !== 1 ||
    decoded.writeStatus[1] !== 1
  ) {
    throw new Error("OPT-0012 reconstruction requires two successful status words");
  }
  const result = [
    new Float32Array(plan.vocabularySize),
    new Float32Array(plan.vocabularySize),
  ] as const;
  for (const row of result) row.fill(Number.NEGATIVE_INFINITY);
  for (let row = 0; row < CFG_ROWS; row += 1) {
    const source = row === CONDITIONAL_ROW
      ? decoded.conditionalLogits
      : decoded.unconditionalLogits;
    result[row]!.set(source, plan.firstCandidateTokenId);
  }
  return Object.freeze(result);
}

/**
 * Exact candidate-domain proof implementation. Parameters may vary for
 * adversarial equivalence tests; the planned candidate path below admits only
 * the allocation's frozen production settings.
 */
export function traceAceOpt0012CompactBrowserV1Sampling(
  input: AceOpt0012CompactSampleInput,
): AceOpt0012CompactSamplingTrace {
  const plan = input.plan;
  requireIssuedPlan(plan);
  requireCompactRows(input.conditionalLogits, input.unconditionalLogits, plan);
  const conditionalAllowedLogits = maskAcePlannerLogits(
    input.conditionalLogits,
    { kind: "all" },
  );
  const unconditionalAllowedLogits = maskAcePlannerLogits(
    input.unconditionalLogits,
    { kind: "all" },
  );
  const cfgLogits = combineAcePlannerCfgLogits(
    conditionalAllowedLogits,
    unconditionalAllowedLogits,
    input.parameters.guidanceScale,
  );
  const finalMaskedLogits = maskAcePlannerLogits(cfgLogits, { kind: "all" });
  const localSeenTokenIds = mapSeenTokenIds(input.seenTokenIds, plan);
  const repetitionPenalizedLogits = applyAcePlannerRepetitionPenalty(
    finalMaskedLogits,
    localSeenTokenIds,
    input.parameters.repetitionPenalty,
  );
  const compactTopK = requireTopK(input.parameters.topK, plan.vocabularySize) >=
      plan.candidateCount
    ? 0
    : input.parameters.topK;
  const topKLogits = applyAcePlannerTopK(
    repetitionPenalizedLogits,
    compactTopK,
  );
  const topKGlobalTokenIds = globalFiniteTokenIds(topKLogits, plan);
  const topPKeep = createAcePlannerBrowserTopPKeep(
    topKLogits,
    input.parameters.topP,
  );
  const topPLogits = topKLogits.slice();
  for (let index = 0; index < topPLogits.length; index += 1) {
    if (topPKeep[index] === 0) topPLogits[index] = Number.NEGATIVE_INFINITY;
  }
  const topPGlobalTokenIds = globalFiniteTokenIds(topPLogits, plan);
  const temperatureScaledLogits = scaleTemperatureForTrace(
    topPLogits,
    input.parameters.temperature,
  );
  const weights = createAcePlannerBrowserSamplingWeights(
    topPLogits,
    input.parameters.temperature,
  );
  const selectedCandidateIndex = aceCategoricalTokenFromWord(
    weights,
    input.word,
  );
  let positiveCandidateCount = 0;
  for (const weight of weights) {
    if (weight > 0) positiveCandidateCount += 1;
  }
  return Object.freeze({
    conditionalAllowedLogits,
    unconditionalAllowedLogits,
    cfgLogits,
    finalMaskedLogits,
    repetitionPenalizedLogits,
    topKLogits,
    topKGlobalTokenIds,
    topPKeep,
    topPGlobalTokenIds,
    topPLogits,
    temperatureScaledLogits,
    weights,
    selectedCandidateIndex,
    tokenId: plan.firstCandidateTokenId + selectedCandidateIndex,
    word: input.word >>> 0,
    positiveCandidateCount,
  });
}

/** The benchmark candidate entry point: frozen settings only, no fallback. */
export function sampleAceOpt0012PlannedCompactToken(
  input: AceOpt0012CompactSampleInput,
): AceOpt0012CompactSamplingTrace {
  requireExactSamplingParameters(input.parameters);
  return traceAceOpt0012CompactBrowserV1Sampling(input);
}

/** Continuous Philox cursor with the same commit-after-success boundary. */
export class AceOpt0012CompactSamplingCursor {
  readonly seed: AceSeed;
  private nextDrawIndex: bigint;

  constructor(seed: AceSeed, firstDrawIndex: number | bigint = 0) {
    this.seed = seed;
    this.nextDrawIndex = requireNonNegativeBigInt(
      firstDrawIndex,
      "OPT-0012 first draw index",
    );
    // Authenticate canonical seed syntax before exposing the cursor.
    aceRandomWord(seed, "planner-sampling", this.nextDrawIndex);
  }

  get consumed(): bigint {
    return this.nextDrawIndex;
  }

  sample(
    input: Omit<AceOpt0012CompactSampleInput, "parameters" | "word">,
  ): AceOpt0012CompactCursorSample {
    const drawIndex = this.nextDrawIndex;
    const word = aceRandomWord(this.seed, "planner-sampling", drawIndex);
    const sampled = sampleAceOpt0012PlannedCompactToken({
      ...input,
      parameters: ACE_OPT_0012_SAMPLING_PARAMETERS,
      word,
    });
    this.nextDrawIndex += 1n;
    return Object.freeze({
      ...sampled,
      drawIndex,
      drawEnd: this.nextDrawIndex,
    });
  }
}

/**
 * Exact allocation-free IEEE-754 binary16 to binary32 word expansion.
 * Every NaN is deterministically quieted while its sign and payload are kept.
 */
export function aceOpt0012Float16BitsToFloat32U32(bits: number): number {
  if (!Number.isInteger(bits) || bits < 0 || bits > 0xffff) {
    throw new RangeError("OPT-0012 FP16 bits must be an unsigned 16-bit integer");
  }
  const sign = (bits & 0x8000) << 16;
  let exponent = (bits >>> 10) & 0x1f;
  let mantissa = bits & 0x03ff;
  let output: number;
  if (exponent === 0) {
    if (mantissa === 0) {
      output = sign;
    } else {
      exponent = 1;
      while ((mantissa & 0x0400) === 0) {
        mantissa <<= 1;
        exponent -= 1;
      }
      mantissa &= 0x03ff;
      output = sign | ((exponent + 112) << 23) | (mantissa << 13);
    }
  } else if (exponent === 0x1f) {
    output = sign | 0x7f80_0000 | (mantissa << 13);
    if (mantissa !== 0) output |= 0x0040_0000;
  } else {
    output = sign | ((exponent + 112) << 23) | (mantissa << 13);
  }
  return output >>> 0;
}

function validateProductionGeometry(): void {
  if (
    ACE_PLANNER_QWEN3_CONFIG.vocabularySize !== AUTHENTICATED_VOCABULARY_SIZE ||
    ACE_PLANNER_QWEN3_CONFIG.hiddenSize !== AUTHENTICATED_HIDDEN_SIZE ||
    ACE_PLANNER_EMBEDDING_ROW_PARTS.length !==
      AUTHENTICATED_TIED_SHARD_PARTITION.length ||
    ACE_PLANNER_PHYSICAL_TENSOR_NAMES.length !== 314
  ) {
    throw new Error("OPT-0012 production planner geometry changed");
  }
  for (let index = 0; index < AUTHENTICATED_TIED_SHARD_PARTITION.length;
    index += 1) {
    const actual = ACE_PLANNER_EMBEDDING_ROW_PARTS[index]!;
    const expected = AUTHENTICATED_TIED_SHARD_PARTITION[index]!;
    if (
      actual.firstRow !== expected.firstRow ||
      actual.rowCount !== expected.rowCount ||
      ACE_PLANNER_PHYSICAL_TENSOR_NAMES[index] !==
        AUTHENTICATED_TIED_TENSOR_NAMES[index]
    ) {
      throw new Error(`OPT-0012 production tied-weight shard ${index} changed`);
    }
  }
  if (
    ACE_PLANNER_SEMANTIC_FIRST_TOKEN_ID !==
      AUTHENTICATED_SEMANTIC_FIRST_TOKEN_ID ||
    ACE_PLANNER_SEMANTIC_CODE_COUNT !== AUTHENTICATED_SEMANTIC_CODE_COUNT ||
    ACE_QWEN_IM_END_TOKEN_ID !== AUTHENTICATED_EOS_TOKEN_ID
  ) {
    throw new Error("OPT-0012 semantic vocabulary identity changed");
  }
}

function validateEligibility(request: AceOpt0012PlanRequest): void {
  if (request.phase !== "semantic-m2") {
    throw new Error("OPT-0012 is eligible only for semantic M2");
  }
  if (request.state !== "regular-code" && request.state !== "forced-eos") {
    throw new Error("OPT-0012 has an unknown semantic FSM state");
  }
  if (
    request.modelProfile !== RAW_FP16_PROFILE ||
    request.rows !== CFG_ROWS ||
    request.conditionalRow !== CONDITIONAL_ROW ||
    request.unconditionalRow !== UNCONDITIONAL_ROW ||
    request.vocabularySize !== AUTHENTICATED_VOCABULARY_SIZE ||
    request.hiddenSize !== AUTHENTICATED_HIDDEN_SIZE
  ) {
    throw new Error("OPT-0012 M2/profile/CFG geometry changed");
  }
  if (
    request.manifestWeightLayout !== RAW_FP16_MANIFEST_LAYOUT ||
    request.kernelWeightLayout !== SOURCE_ROW_MAJOR_LAYOUT ||
    request.weightDtype !== FP16_DTYPE ||
    request.outputDtype !== FP16_DTYPE ||
    request.physicalReadbackLayout !== SHARD_MAJOR_ROW_MAJOR
  ) {
    throw new Error("OPT-0012 raw-FP16 source-row-major layout changed");
  }
  if (
    request.softmaxOracleId !==
      ACE_PLANNER_SOFTMAX_ACCEPTANCE.productionOracleId
  ) {
    throw new Error("OPT-0012 requires the accepted browser-v1 sampler");
  }
  requireExactSamplingParameters(request.sampling);
  requireAllowedTokensEqual(
    request.preCfgAllowedTokens,
    ACE_OPT_0012_PRE_CFG_ALLOWED_TOKENS,
    "pre-CFG constraint",
  );
  requireAllowedTokensEqual(
    request.allowedTokens,
    request.state === "regular-code"
      ? ACE_OPT_0012_REGULAR_ALLOWED_TOKENS
      : ACE_OPT_0012_FORCED_EOS_ALLOWED_TOKENS,
    "final constraint",
  );
  if (
    !Number.isSafeInteger(request.requestedRegularCodeCount) ||
    request.requestedRegularCodeCount <= 0 ||
    !Number.isSafeInteger(request.acceptedRegularCodeCount) ||
    request.acceptedRegularCodeCount < 0 ||
    (request.state === "regular-code"
      ? request.acceptedRegularCodeCount >= request.requestedRegularCodeCount
      : request.acceptedRegularCodeCount !== request.requestedRegularCodeCount)
  ) {
    throw new Error("OPT-0012 semantic FSM count/state is ineligible");
  }
  validateSourceShards(request.tiedWeightShards);
}

function validateSourceShards(
  actual: readonly AceOpt0012SourceShardBinding[],
): void {
  if (actual.length !== ACE_OPT_0012_RAW_FP16_TIED_WEIGHT_SHARDS.length) {
    throw new Error("OPT-0012 tied-weight shard count changed");
  }
  for (let index = 0; index < actual.length; index += 1) {
    const shard = actual[index]!;
    const expected = ACE_OPT_0012_RAW_FP16_TIED_WEIGHT_SHARDS[index]!;
    if (
      shard.shardIndex !== expected.shardIndex ||
      shard.tensorName !== expected.tensorName ||
      shard.firstRow !== expected.firstRow ||
      shard.rowCount !== expected.rowCount ||
      shard.bindingByteOffset !== expected.bindingByteOffset ||
      shard.bindingByteLength !== expected.bindingByteLength ||
      shard.manifestLayout !== expected.manifestLayout ||
      shard.kernelWeightLayout !== expected.kernelWeightLayout ||
      shard.dtype !== expected.dtype
    ) {
      throw new Error(`OPT-0012 tied-weight shard ${index} changed`);
    }
    if (
      !Number.isSafeInteger(shard.bufferByteLength) ||
      shard.bufferByteLength < 0 ||
      shard.bindingByteOffset + shard.bindingByteLength > shard.bufferByteLength
    ) {
      throw new RangeError(`OPT-0012 tied-weight shard ${index} exceeds its buffer`);
    }
  }
}

function deriveIntersections(
  firstTokenId: number,
  tokenCount: number,
): readonly AceOpt0012RowIntersection[] {
  const rangeEnd = firstTokenId + tokenCount;
  const intersections = ACE_PLANNER_EMBEDDING_ROW_PARTS.flatMap(
    (shard, shardIndex) => {
      const first = Math.max(firstTokenId, shard.firstRow);
      const end = Math.min(rangeEnd, shard.firstRow + shard.rowCount);
      return first >= end
        ? []
        : [Object.freeze({
            shardIndex,
            globalFirstRow: first,
            globalLastRow: end - 1,
            localFirstRow: first - shard.firstRow,
            rowCount: end - first,
          })];
    },
  );
  let cursor = firstTokenId;
  for (const intersection of intersections) {
    if (intersection.globalFirstRow !== cursor) {
      throw new Error("OPT-0012 candidate intersections contain a gap or overlap");
    }
    cursor += intersection.rowCount;
  }
  if (cursor !== rangeEnd) {
    throw new Error("OPT-0012 candidate intersections do not cover the domain");
  }
  return Object.freeze(intersections);
}

function createHeadSlice(
  intersection: AceOpt0012RowIntersection,
  source: AceOpt0012SourceShardBinding,
): AceOpt0012HeadSlice {
  const rowBytes = ACE_PLANNER_QWEN3_CONFIG.hiddenSize * FP16_BYTES;
  const sourceBindingByteOffset = source.bindingByteOffset +
    intersection.localFirstRow * rowBytes;
  const sourceBindingByteLength = intersection.rowCount * rowBytes;
  const sourceBindingByteEnd = sourceBindingByteOffset + sourceBindingByteLength;
  const sourceOwnerByteEnd = source.bindingByteOffset + source.bindingByteLength;
  if (
    sourceBindingByteOffset < source.bindingByteOffset ||
    sourceBindingByteEnd > sourceOwnerByteEnd ||
    sourceBindingByteEnd > source.bufferByteLength
  ) {
    throw new RangeError(
      `OPT-0012 shard ${intersection.shardIndex} slice exceeds its source binding`,
    );
  }
  const gemm = planAceTiledGemm({
    rows: CFG_ROWS,
    inner: ACE_PLANNER_QWEN3_CONFIG.hiddenSize,
    columns: intersection.rowCount,
  });
  const outputRanges = Object.freeze(gemm.outputRanges.map((range) =>
    Object.freeze({ ...range })));
  return Object.freeze({
    ...intersection,
    sourceTensorName: source.tensorName,
    sourceBindingByteOffset,
    sourceBindingByteLength,
    sourceBindingByteEnd,
    sourceOwnerByteOffset: source.bindingByteOffset,
    sourceOwnerByteLength: source.bindingByteLength,
    sourceOwnerByteEnd,
    logicalWeightTrafficBytes: sourceBindingByteLength,
    logicalMultiplyAdds: CFG_ROWS * ACE_PLANNER_QWEN3_CONFIG.hiddenSize *
      intersection.rowCount,
    scheduledMultiplyAdds: sum(outputRanges.map((range) => range.multiplyAdds)),
    workgroupCount: sum(outputRanges.map((range) => range.workgroupCount)),
    outputRanges,
    rawLogitBytes: CFG_ROWS * intersection.rowCount * FP16_BYTES,
  });
}

function createReadbackPlan(
  firstCandidateTokenId: number,
  candidateCount: number,
  intersections: readonly AceOpt0012RowIntersection[],
): AceOpt0012CompactReadbackPlan {
  let cursor = 0;
  const copies: AceOpt0012ReadbackCopy[] = [];
  const logicalSpans: AceOpt0012LogicalSpan[] = [];
  for (const intersection of intersections) {
    const blockBytes = CFG_ROWS * intersection.rowCount * FP16_BYTES;
    if (cursor % COPY_ALIGNMENT_BYTES !== 0 ||
        blockBytes % COPY_ALIGNMENT_BYTES !== 0) {
      throw new Error("OPT-0012 compact logit copy lost four-byte alignment");
    }
    copies.push(Object.freeze({
      index: copies.length,
      kind: "logits" as const,
      shardIndex: intersection.shardIndex,
      sourceByteOffset: 0,
      destinationByteOffset: cursor,
      byteLength: blockBytes,
    }));
    for (let physicalRow = 0; physicalRow < CFG_ROWS; physicalRow += 1) {
      logicalSpans.push(Object.freeze({
        shardIndex: intersection.shardIndex,
        physicalRow: physicalRow as 0 | 1,
        sourceByteOffset:
          cursor + physicalRow * intersection.rowCount * FP16_BYTES,
        byteLength: intersection.rowCount * FP16_BYTES,
        destinationCandidateOffset:
          intersection.globalFirstRow - firstCandidateTokenId,
        candidateCount: intersection.rowCount,
        globalFirstTokenId: intersection.globalFirstRow,
      }));
    }
    cursor += blockBytes;
  }
  const rawLogitBytes = cursor;
  const writeStatusByteOffset = cursor;
  const writeStatusByteLength = CFG_ROWS * U32_BYTES;
  copies.push(Object.freeze({
    index: copies.length,
    kind: "write-status" as const,
    shardIndex: null,
    sourceByteOffset: 0,
    destinationByteOffset: writeStatusByteOffset,
    byteLength: writeStatusByteLength,
  }));
  const usedBytes = writeStatusByteOffset + writeStatusByteLength;
  const allocationBytes = align(usedBytes, ACE_PACKAGE_ALIGNMENT_BYTES);
  if (rawLogitBytes !== CFG_ROWS * candidateCount * FP16_BYTES) {
    throw new Error("OPT-0012 compact readback does not cover every candidate row");
  }
  return Object.freeze({
    physicalLayout: SHARD_MAJOR_ROW_MAJOR,
    copies: Object.freeze(copies),
    logicalSpans: Object.freeze(logicalSpans),
    rawLogitBytes,
    writeStatusByteOffset,
    writeStatusByteLength,
    usedBytes,
    alignmentPaddingBytes: allocationBytes - usedBytes,
    allocationBytes,
  });
}

function requireIssuedPlan(plan: AceOpt0012CompactSemanticHeadPlan): void {
  if (
    typeof plan !== "object" ||
    plan === null ||
    !ISSUED_PLANS.has(plan) ||
    !Object.isFrozen(plan) ||
    plan.contractId !== CONTRACT_ID
  ) {
    throw new Error("OPT-0012 requires an issued immutable compact-head plan");
  }
}

function requireCompactRows(
  conditional: ArrayLike<number>,
  unconditional: ArrayLike<number>,
  plan: AceOpt0012CompactSemanticHeadPlan,
): void {
  if (
    !Number.isSafeInteger(conditional.length) ||
    conditional.length !== plan.candidateCount ||
    unconditional.length !== plan.candidateCount
  ) {
    throw new RangeError(
      `OPT-0012 compact rows must each contain ${plan.candidateCount} candidates`,
    );
  }
}

function mapSeenTokenIds(
  tokenIds: readonly number[],
  plan: AceOpt0012CompactSemanticHeadPlan,
): number[] {
  const result: number[] = [];
  for (const tokenId of tokenIds) {
    if (
      !Number.isSafeInteger(tokenId) ||
      tokenId < 0 ||
      tokenId >= plan.vocabularySize
    ) {
      throw new RangeError(
        `OPT-0012 seen token ${String(tokenId)} is outside the vocabulary`,
      );
    }
    const local = tokenId - plan.firstCandidateTokenId;
    if (local >= 0 && local < plan.candidateCount) result.push(local);
  }
  return result;
}

function globalFiniteTokenIds(
  logits: Float32Array,
  plan: AceOpt0012CompactSemanticHeadPlan,
): readonly number[] {
  const result: number[] = [];
  for (let index = 0; index < logits.length; index += 1) {
    if (Number.isFinite(logits[index])) {
      result.push(plan.firstCandidateTokenId + index);
    }
  }
  return Object.freeze(result);
}

function scaleTemperatureForTrace(
  logits: Float32Array,
  temperature: number,
): Float32Array {
  if (!Number.isFinite(temperature) || temperature <= 0) {
    throw new RangeError("OPT-0012 temperature must be positive and finite");
  }
  const roundedTemperature = Math.fround(temperature);
  const result = logits.slice();
  for (let index = 0; index < result.length; index += 1) {
    if (result[index] !== Number.NEGATIVE_INFINITY) {
      result[index] = Math.fround(result[index]! / roundedTemperature);
    }
  }
  return result;
}

function requireTopK(topK: number, vocabularySize: number): number {
  if (!Number.isSafeInteger(topK) || topK < 0 || topK > vocabularySize) {
    throw new RangeError("OPT-0012 topK must be between zero and vocabulary size");
  }
  return topK;
}

function requireExactSamplingParameters(
  actual: AcePlannerSamplingParameters,
): void {
  const expected = ACE_OPT_0012_SAMPLING_PARAMETERS;
  if (
    actual.temperature !== expected.temperature ||
    actual.guidanceScale !== expected.guidanceScale ||
    actual.topK !== expected.topK ||
    actual.topP !== expected.topP ||
    actual.repetitionPenalty !== expected.repetitionPenalty
  ) {
    throw new Error("OPT-0012 sampling settings changed");
  }
}

function requireAllowedTokensEqual(
  actual: AcePlannerAllowedTokens,
  expected: AcePlannerAllowedTokens,
  label: string,
): void {
  if (actual.kind !== expected.kind) {
    throw new Error(`OPT-0012 ${label} kind changed`);
  }
  if (actual.kind === "all" || expected.kind === "all") return;
  if (actual.kind === "ids" && expected.kind === "ids") {
    if (!equalNumbers(actual.tokenIds, expected.tokenIds)) {
      throw new Error(`OPT-0012 ${label} IDs changed`);
    }
    return;
  }
  if (actual.kind === "range" && expected.kind === "range") {
    if (
      actual.firstTokenId !== expected.firstTokenId ||
      actual.tokenCount !== expected.tokenCount ||
      !equalNumbers(
        actual.additionalTokenIds ?? [],
        expected.additionalTokenIds ?? [],
      )
    ) {
      throw new Error(`OPT-0012 ${label} range changed`);
    }
    return;
  }
  throw new Error(`OPT-0012 ${label} shape changed`);
}

function equalNumbers(
  left: readonly number[],
  right: readonly number[],
): boolean {
  return left.length === right.length &&
    left.every((value, index) => value === right[index]);
}

function requireNonNegativeBigInt(
  value: number | bigint,
  label: string,
): bigint {
  if (typeof value === "number") {
    if (!Number.isSafeInteger(value) || value < 0) {
      throw new RangeError(`${label} must be a non-negative safe integer`);
    }
    return BigInt(value);
  }
  if (value < 0n) throw new RangeError(`${label} must be non-negative`);
  return value;
}

function authenticatedSourceShard(
  shardIndex: number,
  firstRow: number,
  rowCount: number,
  bindingByteLength: number,
): AceOpt0012SourceShardBinding {
  return Object.freeze({
    shardIndex,
    tensorName: AUTHENTICATED_TIED_TENSOR_NAMES[shardIndex]!,
    firstRow,
    rowCount,
    bindingByteOffset: 0,
    bindingByteLength,
    bufferByteLength: bindingByteLength,
    manifestLayout: RAW_FP16_MANIFEST_LAYOUT,
    kernelWeightLayout: SOURCE_ROW_MAJOR_LAYOUT,
    dtype: FP16_DTYPE,
  });
}

function sum(values: readonly number[]): number {
  let result = 0;
  for (const value of values) {
    result += value;
    if (!Number.isSafeInteger(result)) {
      throw new RangeError("OPT-0012 accounting exceeds safe integers");
    }
  }
  return result;
}

function align(value: number, alignment: number): number {
  if (
    !Number.isSafeInteger(value) ||
    value < 0 ||
    !Number.isSafeInteger(alignment) ||
    alignment <= 0
  ) {
    throw new RangeError("OPT-0012 alignment input is invalid");
  }
  const result = Math.ceil(value / alignment) * alignment;
  if (!Number.isSafeInteger(result)) {
    throw new RangeError("OPT-0012 alignment exceeds safe integers");
  }
  return result;
}
