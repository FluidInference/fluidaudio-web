import {
  ACE_MAX_DURATION_SECONDS,
  ACE_MIN_DURATION_SECONDS,
  ACE_SAMPLE_RATE_HZ,
} from "../api.js";
import {
  ACE_PLANNER_SOURCE_REVISION,
  ACE_SOURCE_REVISION,
} from "../runtime/diagnostics.js";

export const ACE_LATENT_RATE_HZ = 25;
export const ACE_SEMANTIC_RATE_HZ = 5;
export const ACE_VAE_TEMPORAL_STRIDE = 1_920;
export const ACE_DIT_PATCH_SIZE = 2;
export const ACE_AUDIO_ACOUSTIC_CHANNELS = 64;
export const ACE_CONTEXT_CHANNELS = 128;
export const ACE_DIT_INPUT_CHANNELS = 192;

export const ACE_DIT_LAYER_TYPES = Object.freeze(
  Array.from({ length: 24 }, (_, index) =>
    index % 2 === 0 ? "sliding_attention" : "full_attention",
  ),
) as readonly ("sliding_attention" | "full_attention")[];

/**
 * Immutable geometry for the two pinned model revisions.
 *
 * Stage 1 supports one audited graph, so incompatible manifests must fail
 * instead of silently selecting a nearby Transformers architecture.
 */
export const ACE_GRAPH_CONTRACT = Object.freeze({
  id: "ace-step-1.5-turbo-graph-v1",
  source: Object.freeze({
    aceRepositoryRevision: ACE_SOURCE_REVISION,
    plannerModelRevision: ACE_PLANNER_SOURCE_REVISION,
  }),
  timing: Object.freeze({
    sampleRateHz: ACE_SAMPLE_RATE_HZ,
    latentRateHz: ACE_LATENT_RATE_HZ,
    semanticRateHz: ACE_SEMANTIC_RATE_HZ,
    vaeTemporalStride: ACE_VAE_TEMPORAL_STRIDE,
  }),
  dit: Object.freeze({
    hiddenSize: 2_048,
    intermediateSize: 6_144,
    layerCount: 24,
    attentionHeads: 16,
    keyValueHeads: 8,
    headDimension: 128,
    queryHeadsPerKeyValueHead: 2,
    rmsNormEpsilon: 1e-6,
    attentionBias: false,
    inputChannels: ACE_DIT_INPUT_CHANNELS,
    audioChannels: ACE_AUDIO_ACOUSTIC_CHANNELS,
    contextChannels: ACE_CONTEXT_CHANNELS,
    patchSize: ACE_DIT_PATCH_SIZE,
    slidingWindowRadius: 128,
    maximumSlidingKeys: 257,
    ropeTheta: 1_000_000,
    maximumPositionEmbeddings: 32_768,
    layerTypes: ACE_DIT_LAYER_TYPES,
    denoisingEvaluations: 8,
  }),
  conditioner: Object.freeze({
    hiddenSize: 2_048,
    intermediateSize: 6_144,
    textInputSize: 1_024,
    timbreInputSize: 64,
    lyricLayerCount: 8,
    timbreLayerCount: 4,
    poolerLayerCount: 2,
    poolWindowSize: 5,
    fsqDimension: 2_048,
    fsqLevels: Object.freeze([8, 8, 8, 5, 5, 5] as const),
    fsqQuantizerCount: 1,
  }),
  textEncoder: Object.freeze({
    hiddenSize: 1_024,
    intermediateSize: 3_072,
    layerCount: 28,
    attentionHeads: 16,
    keyValueHeads: 8,
    headDimension: 128,
    vocabularySize: 151_669,
    maximumPositionEmbeddings: 32_768,
    ropeTheta: 1_000_000,
  }),
  planner: Object.freeze({
    hiddenSize: 1_024,
    intermediateSize: 3_072,
    layerCount: 28,
    attentionHeads: 16,
    keyValueHeads: 8,
    headDimension: 128,
    vocabularySize: 217_204,
    maximumPositionEmbeddings: 40_960,
    ropeTheta: 1_000_000,
  }),
  vae: Object.freeze({
    audioChannels: 2,
    decoderInputChannels: 64,
    decoderChannels: 128,
    channelMultiples: Object.freeze([1, 2, 4, 8, 16] as const),
    downsamplingRatios: Object.freeze([2, 4, 4, 6, 10] as const),
    samplingRateHz: ACE_SAMPLE_RATE_HZ,
  }),
  tensorPayloadBytes: Object.freeze({
    planner: 1_325_768_704,
    textEncoder: 1_191_553_024,
    conditionEncoder: 1_216_735_232,
    semanticDetokenizer: 210_023_552,
    fsqOutputProjection: 28_672,
    dit: 3_150_917_760,
    vae: 337_583_104,
    silenceLatent: 3_840_000,
    total: 7_436_450_048,
  }),
} as const);

export interface AceDurationGraphShape {
  readonly durationSeconds: number;
  readonly latentFrames: number;
  readonly paddedLatentFrames: number;
  readonly latentPaddingFrames: 0 | 1;
  readonly ditTokens: number;
  readonly semanticCodes: number;
  readonly audioFramesPerChannel: number;
  readonly audioFloat32Bytes: number;
}

/** Resolve every duration-dependent production shape before allocating. */
export function deriveAceDurationGraphShape(
  durationSeconds: number,
): AceDurationGraphShape {
  if (
    !Number.isSafeInteger(durationSeconds) ||
    durationSeconds < ACE_MIN_DURATION_SECONDS ||
    durationSeconds > ACE_MAX_DURATION_SECONDS
  ) {
    throw new RangeError(
      `ACE duration must be an integer from ${ACE_MIN_DURATION_SECONDS} through ${ACE_MAX_DURATION_SECONDS} seconds`,
    );
  }

  const latentFrames = durationSeconds * ACE_LATENT_RATE_HZ;
  const latentPaddingFrames = (latentFrames % ACE_DIT_PATCH_SIZE) as 0 | 1;
  const paddedLatentFrames = latentFrames + latentPaddingFrames;
  const audioFramesPerChannel = latentFrames * ACE_VAE_TEMPORAL_STRIDE;

  return Object.freeze({
    durationSeconds,
    latentFrames,
    paddedLatentFrames,
    latentPaddingFrames,
    ditTokens: paddedLatentFrames / ACE_DIT_PATCH_SIZE,
    semanticCodes: durationSeconds * ACE_SEMANTIC_RATE_HZ,
    audioFramesPerChannel,
    audioFloat32Bytes: audioFramesPerChannel * 2 * Float32Array.BYTES_PER_ELEMENT,
  });
}

/** Fail fast if an edited constant makes the frozen graph self-inconsistent. */
export function assertAceGraphContract(): void {
  const { dit, vae, tensorPayloadBytes } = ACE_GRAPH_CONTRACT;
  if (dit.attentionHeads * dit.headDimension !== dit.hiddenSize) {
    throw new Error("ACE query-head geometry does not reconstruct hidden size");
  }
  if (
    dit.keyValueHeads * dit.queryHeadsPerKeyValueHead !== dit.attentionHeads
  ) {
    throw new Error("ACE grouped-query attention geometry is inconsistent");
  }
  if (dit.audioChannels + dit.contextChannels !== dit.inputChannels) {
    throw new Error("ACE DiT input channel geometry is inconsistent");
  }
  if (
    ACE_DIT_LAYER_TYPES.length !== dit.layerCount ||
    ACE_DIT_LAYER_TYPES.filter((type) => type === "full_attention").length !== 12 ||
    ACE_DIT_LAYER_TYPES.filter((type) => type === "sliding_attention").length !== 12
  ) {
    throw new Error("ACE alternating DiT layer contract is inconsistent");
  }
  const vaeStride = vae.downsamplingRatios.reduce(
    (product, ratio) => product * ratio,
    1,
  );
  if (
    vaeStride !== ACE_VAE_TEMPORAL_STRIDE ||
    ACE_SAMPLE_RATE_HZ / vaeStride !== ACE_LATENT_RATE_HZ
  ) {
    throw new Error("ACE VAE temporal geometry is inconsistent");
  }
  const componentTotal =
    tensorPayloadBytes.planner +
    tensorPayloadBytes.textEncoder +
    tensorPayloadBytes.conditionEncoder +
    tensorPayloadBytes.semanticDetokenizer +
    tensorPayloadBytes.fsqOutputProjection +
    tensorPayloadBytes.dit +
    tensorPayloadBytes.vae +
    tensorPayloadBytes.silenceLatent;
  if (componentTotal !== tensorPayloadBytes.total) {
    throw new Error("ACE tensor payload accounting is inconsistent");
  }
}
