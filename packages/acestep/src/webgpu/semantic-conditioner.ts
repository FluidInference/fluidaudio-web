import type { AceModelProfileId } from "./capabilities.js";
import {
  AceCorrectnessEncoderRuntime,
  aceEncoderLayerAttentionMode,
  planAceEncoderBlock,
  type AceEncoderBlockScratch,
  type AceEncoderBlockWeights,
  type AceEncoderConfig,
} from "./ace-encoder.js";
import {
  AceCorrectnessConditionLayoutKernel,
  ACE_CONTEXT_CHANNELS,
} from "./kernels/condition-layout.js";
import {
  ACE_FSQ_CODE_DIMENSION,
  AceCorrectnessFsqDecodeKernel,
} from "./kernels/fsq-decode.js";
import { AceCorrectnessDetokenizerExpandKernel } from "./kernels/detokenizer-expand.js";
import {
  acePrimitiveCooperativeQuanta,
  AceCorrectnessGemmKernel,
  type AceGemmDispatch,
} from "./kernels/gemm.js";
import {
  AceCorrectnessTensorCopyKernel,
  type AceCopyDispatch,
} from "./kernels/tensor-copy.js";

export const ACE_CONDITION_HIDDEN_SIZE = 2_048;
export const ACE_CONDITION_INTERMEDIATE_SIZE = 6_144;
export const ACE_CONDITION_QUERY_HEADS = 16;
export const ACE_CONDITION_KEY_VALUE_HEADS = 8;
export const ACE_CONDITION_HEAD_DIMENSION = 128;
export const ACE_CONDITION_ROPE_THETA = 1_000_000;
export const ACE_CONDITION_RMS_NORM_EPSILON = 1e-6;
export const ACE_CONDITION_SLIDING_RADIUS = 128;
export const ACE_TEXT_EMBEDDING_WIDTH = 1_024;
export const ACE_AUDIO_LATENT_CHANNELS = 64;
export const ACE_SEMANTIC_POOL_WIDTH = 5;
export const ACE_DETOKENIZER_LAYER_COUNT = 2;
export const ACE_LYRIC_ENCODER_LAYER_COUNT = 8;
export const ACE_TIMBRE_ENCODER_LAYER_COUNT = 4;
export const ACE_MAX_LYRIC_TOKENS = 2_048;
export const ACE_MAX_TEXT_TOKENS = 256;
export const ACE_NO_REFERENCE_TIMBRE_FRAMES = 750;

export const ACE_CONDITION_ENCODER_CONFIG: Readonly<AceEncoderConfig> =
  Object.freeze({
    hiddenSize: ACE_CONDITION_HIDDEN_SIZE,
    intermediateSize: ACE_CONDITION_INTERMEDIATE_SIZE,
    queryHeads: ACE_CONDITION_QUERY_HEADS,
    keyValueHeads: ACE_CONDITION_KEY_VALUE_HEADS,
    headDimension: ACE_CONDITION_HEAD_DIMENSION,
    maximumPositionEmbeddings: 32_768,
    ropeTheta: ACE_CONDITION_ROPE_THETA,
    rmsNormEpsilon: ACE_CONDITION_RMS_NORM_EPSILON,
    slidingRadius: ACE_CONDITION_SLIDING_RADIUS,
    attentionBias: false,
    hiddenActivation: "silu",
  });

/** Canonical browser-package bindings outside the repeated encoder layers. */
export const ACE_SEMANTIC_TENSOR_NAMES = Object.freeze({
  fsqProjectOut: "ace.tokenizer.quantizer.project_out.weight",
  fsqProjectOutBias: "ace.tokenizer.quantizer.project_out.bias",
  detokenizerInputProjection: "ace.detokenizer.embed_tokens.weight",
  detokenizerInputProjectionBias: "ace.detokenizer.embed_tokens.bias",
  specialTokens: "ace.detokenizer.special_tokens",
  finalNorm: "ace.detokenizer.norm.weight",
  acousticProjection: "ace.detokenizer.proj_out.weight",
  acousticProjectionBias: "ace.detokenizer.proj_out.bias",
} as const);

export const ACE_DIRECT_CONDITIONER_TENSOR_NAMES = Object.freeze({
  textProjection: "ace.encoder.text_projector.weight",
  lyricInputProjection: "ace.encoder.lyric_encoder.embed_tokens.weight",
  lyricInputProjectionBias: "ace.encoder.lyric_encoder.embed_tokens.bias",
  lyricFinalNorm: "ace.encoder.lyric_encoder.norm.weight",
  timbreInputProjection: "ace.encoder.timbre_encoder.embed_tokens.weight",
  timbreInputProjectionBias: "ace.encoder.timbre_encoder.embed_tokens.bias",
  timbreFinalNorm: "ace.encoder.timbre_encoder.norm.weight",
  silenceSource: "constants.silence_latent",
} as const);

export interface AceGraphStage {
  readonly id: string;
  readonly outputShape: readonly number[];
  readonly weightLifetime: "semantic" | "conditioner" | "constant";
}

export interface AceSemanticDecodeShape {
  readonly batch: number;
  /** Parsed audio-code count at 5 Hz per item. */
  readonly codeTokens: number;
}

export interface AceSemanticDecodePlan extends AceSemanticDecodeShape {
  readonly codeCount: number;
  readonly outputFrames: number;
  readonly outputElements: number;
  readonly detokenizerBatch: number;
  readonly detokenizerTokens: typeof ACE_SEMANTIC_POOL_WIDTH;
  readonly layerAttentionModes: readonly ["sliding", "full"];
  /** Logical subsystem stages before GEMM output-range expansion. */
  readonly quantumCount: 8;
  readonly stages: readonly AceGraphStage[];
}

export interface AceDirectConditionerShape {
  readonly batch: number;
  readonly textTokens: number;
  readonly lyricTokens: number;
  readonly latentFrames: number;
}

/**
 * Selects only the final 64-channel source context. The learned
 * lyric/timbre/text conditioner graph is identical in both modes.
 *
 * This is the pinned upstream `is_covers=true` plus
 * `precomputed_lm_hints_25Hz` branch.
 */
export type AceConditionerSourceMode =
  | Readonly<{ readonly kind: "direct-silence" }>
  | Readonly<{
      readonly kind: "planner-semantic-cover";
      readonly semanticCodeCount: number;
      readonly semanticFrames: number;
    }>;

export type AceConditionerDispatchSource =
  | Readonly<{ readonly kind: "direct-silence" }>
  | Readonly<{
      readonly kind: "planner-semantic-cover";
      readonly semanticCodeCount: number;
      readonly semanticFrames: number;
      /** Unpadded semantic-detokenizer output `[batch,semanticFrames,64]`. */
      readonly semanticHints: GPUBufferBinding;
    }>;

export interface AceDirectConditionerPlan extends AceDirectConditionerShape {
  readonly plannerEnabled: boolean;
  readonly semanticCodeCount: number;
  readonly referenceMode: "silence-750";
  readonly sourceSelection:
    | "silence-direct-no-cover"
    | "semantic-precomputed-lm-hints-cover";
  readonly textRows: number;
  readonly lyricRows: number;
  readonly timbreRows: number;
  readonly firstPackedTokens: number;
  readonly conditionTokens: number;
  readonly conditionElements: number;
  readonly contextChannels: typeof ACE_CONTEXT_CHANNELS;
  readonly contextElements: number;
  readonly lyricLayerAttentionModes: readonly string[];
  readonly timbreLayerAttentionModes: readonly string[];
  /** Logical stages before GEMM output-range expansion. */
  readonly quantumCount: 23;
  readonly stages: readonly AceGraphStage[];
}

export interface AceEncoderLayerTensorNames {
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

export interface AceSemanticDecodeBindings {
  readonly codeIds: GPUBufferBinding;
  readonly output: GPUBufferBinding;
  readonly weights: Readonly<{
    readonly fsqProjectOut: GPUBufferBinding;
    readonly fsqProjectOutBias: GPUBufferBinding;
    readonly detokenizerInputProjection: GPUBufferBinding;
    readonly detokenizerInputProjectionBias: GPUBufferBinding;
    readonly specialTokens: GPUBufferBinding;
    readonly layers: readonly AceEncoderBlockWeights[];
    readonly finalNorm: GPUBufferBinding;
    readonly acousticProjection: GPUBufferBinding;
    readonly acousticProjectionBias: GPUBufferBinding;
  }>;
  readonly controls: Readonly<{
    readonly validationStatus: GPUBufferBinding;
    readonly validLengths: GPUBufferBinding;
    readonly cosine: GPUBufferBinding;
    readonly sine: GPUBufferBinding;
  }>;
  readonly scratch: Readonly<{
    readonly fsqScalars: GPUBufferBinding;
    readonly quantized: GPUBufferBinding;
    readonly embeddedCodes: GPUBufferBinding;
    readonly patchInput: GPUBufferBinding;
    /** May be shared between the two sequential layers. */
    readonly block: AceEncoderBlockScratch;
    /** Two outputs; ping-pong bindings are accepted. */
    readonly layerOutputs: readonly GPUBufferBinding[];
    readonly normalized: GPUBufferBinding;
  }>;
}

export interface AceSemanticDecodeDispatch {
  readonly label: string;
  readonly plan: AceSemanticDecodePlan;
  /** FIFO Stage-1 units; encoder blocks remain intact within one quantum. */
  readonly quanta: readonly AceConditionerQuantum[];
  readonly primitiveCount: number;
  /** Correctness-harness convenience; production submits `quanta`. */
  encode(pass: GPUComputePassEncoder): void;
}

export interface AceConditionerQuantum {
  readonly id: string;
  readonly primitiveCount: number;
  encode(pass: GPUComputePassEncoder): void;
}

export interface AceDirectConditionerBindings {
  readonly textHiddenStates: GPUBufferBinding;
  readonly lyricHiddenStates: GPUBufferBinding;
  readonly textMask: GPUBufferBinding;
  readonly lyricMask: GPUBufferBinding;
  /** Authenticated FP32 `[1,64,15000]` package constant. */
  readonly silenceSource: GPUBufferBinding;
  /** U32 `[batch,latentFrames]`; direct text-to-music is all ones. */
  readonly chunkMask: GPUBufferBinding;
  readonly output: Readonly<{
    readonly conditionHiddenStates: GPUBufferBinding;
    readonly conditionMask: GPUBufferBinding;
    readonly contextLatents: GPUBufferBinding;
  }>;
  readonly weights: Readonly<{
    readonly textProjection: GPUBufferBinding;
    readonly lyricInputProjection: GPUBufferBinding;
    readonly lyricInputProjectionBias: GPUBufferBinding;
    readonly lyricLayers: readonly AceEncoderBlockWeights[];
    readonly lyricFinalNorm: GPUBufferBinding;
    readonly timbreInputProjection: GPUBufferBinding;
    readonly timbreInputProjectionBias: GPUBufferBinding;
    readonly timbreLayers: readonly AceEncoderBlockWeights[];
    readonly timbreFinalNorm: GPUBufferBinding;
  }>;
  readonly controls: Readonly<{
    readonly lyricValidLengths: GPUBufferBinding;
    readonly lyricCosine: GPUBufferBinding;
    readonly lyricSine: GPUBufferBinding;
    readonly timbreValidLengths: GPUBufferBinding;
    readonly timbreCosine: GPUBufferBinding;
    readonly timbreSine: GPUBufferBinding;
    /** U32 `[batch]`, all zero: first row from each 750-frame sequence. */
    readonly timbreFirstRowIndices: GPUBufferBinding;
    /** U32 `[batch]`, all one: exactly one no-reference timbre token. */
    readonly timbreMask: GPUBufferBinding;
  }>;
  readonly scratch: Readonly<{
    readonly textProjected: GPUBufferBinding;
    readonly lyricProjected: GPUBufferBinding;
    readonly lyricBlock: AceEncoderBlockScratch;
    readonly lyricLayerOutputs: readonly GPUBufferBinding[];
    readonly lyricEncoded: GPUBufferBinding;
    readonly timbreSource: GPUBufferBinding;
    readonly timbreProjected: GPUBufferBinding;
    readonly timbreBlock: AceEncoderBlockScratch;
    readonly timbreLayerOutputs: readonly GPUBufferBinding[];
    readonly timbreNormalized: GPUBufferBinding;
    readonly timbreToken: GPUBufferBinding;
    readonly firstPackIndices: GPUBufferBinding;
    readonly firstPacked: GPUBufferBinding;
    readonly firstPackedMask: GPUBufferBinding;
    readonly secondPackIndices: GPUBufferBinding;
    readonly sourceLatents: GPUBufferBinding;
  }>;
}

export interface AceDirectConditionerDispatch {
  readonly label: string;
  readonly plan: AceDirectConditionerPlan;
  readonly quanta: readonly AceConditionerQuantum[];
  readonly primitiveCount: number;
  /** Correctness-harness convenience; production submits `quanta`. */
  encode(pass: GPUComputePassEncoder): void;
}

type CompositeDispatch =
  | AceGemmDispatch
  | AceCopyDispatch<unknown>
  | Readonly<{
      readonly label: string;
      readonly cooperativeQuanta?: readonly AceConditionerQuantum[];
      encode(pass: GPUComputePassEncoder): void;
    }>;

function createConditionerQuanta(
  label: string,
  dispatches: readonly CompositeDispatch[],
  primitiveCounts: readonly number[],
): readonly AceConditionerQuantum[] {
  if (dispatches.length !== primitiveCounts.length) {
    throw new Error(`${label} conditioner quantum accounting differs from dispatches`);
  }
  return Object.freeze(dispatches.flatMap((dispatch, index) => {
    const primitiveCount = primitiveCounts[index]!;
    if (!Number.isSafeInteger(primitiveCount) || primitiveCount <= 0) {
      throw new RangeError(`${label} quantum ${index} has an invalid primitive count`);
    }
    if ("cooperativeQuanta" in dispatch) {
      return dispatch.cooperativeQuanta as readonly AceConditionerQuantum[];
    }
    const quanta = acePrimitiveCooperativeQuanta(dispatch);
    return quanta.map((quantum, rangeIndex) => Object.freeze({
      ...quantum,
      id: quanta.length === 1
        ? `${label}-quantum-${index}`
        : `${label}-quantum-${index}-range-${rangeIndex}`,
    }));
  }));
}

/** Real Stage 1 subsystem composition; the FIFO owner supplies every buffer. */
export class AceCorrectnessSemanticConditionerRuntime {
  private readonly fsq: AceCorrectnessFsqDecodeKernel;
  private readonly detokenizerExpand: AceCorrectnessDetokenizerExpandKernel;
  private readonly gemm: AceCorrectnessGemmKernel;
  private readonly copy: AceCorrectnessTensorCopyKernel;
  private readonly conditionLayout: AceCorrectnessConditionLayoutKernel;
  private readonly encoder: AceCorrectnessEncoderRuntime;
  private destroyed = false;

  private constructor(
    readonly modelProfile: AceModelProfileId,
    fsq: AceCorrectnessFsqDecodeKernel,
    detokenizerExpand: AceCorrectnessDetokenizerExpandKernel,
    gemm: AceCorrectnessGemmKernel,
    copy: AceCorrectnessTensorCopyKernel,
    conditionLayout: AceCorrectnessConditionLayoutKernel,
    encoder: AceCorrectnessEncoderRuntime,
  ) {
    this.fsq = fsq;
    this.detokenizerExpand = detokenizerExpand;
    this.gemm = gemm;
    this.copy = copy;
    this.conditionLayout = conditionLayout;
    this.encoder = encoder;
  }

  static create(
    device: GPUDevice,
    modelProfile: AceModelProfileId,
  ): AceCorrectnessSemanticConditionerRuntime {
    let fsq: AceCorrectnessFsqDecodeKernel | undefined;
    let detokenizerExpand: AceCorrectnessDetokenizerExpandKernel | undefined;
    let gemm: AceCorrectnessGemmKernel | undefined;
    let copy: AceCorrectnessTensorCopyKernel | undefined;
    let conditionLayout: AceCorrectnessConditionLayoutKernel | undefined;
    let encoder: AceCorrectnessEncoderRuntime | undefined;
    try {
      fsq = AceCorrectnessFsqDecodeKernel.create(device, modelProfile);
      detokenizerExpand = AceCorrectnessDetokenizerExpandKernel.create(
        device,
        modelProfile,
      );
      gemm = AceCorrectnessGemmKernel.create(device, modelProfile);
      copy = AceCorrectnessTensorCopyKernel.create(device, modelProfile);
      conditionLayout = AceCorrectnessConditionLayoutKernel.create(
        device,
        modelProfile,
      );
      encoder = AceCorrectnessEncoderRuntime.create(device, modelProfile);
      return new AceCorrectnessSemanticConditionerRuntime(
        modelProfile,
        fsq,
        detokenizerExpand,
        gemm,
        copy,
        conditionLayout,
        encoder,
      );
    } catch (error) {
      fsq?.destroy();
      detokenizerExpand?.destroy();
      gemm?.destroy();
      copy?.destroy();
      conditionLayout?.destroy();
      encoder?.destroy();
      throw error;
    }
  }

  async createSemanticDecodeDispatch(
    label: string,
    shape: AceSemanticDecodeShape,
    bindings: AceSemanticDecodeBindings,
  ): Promise<AceSemanticDecodeDispatch> {
    this.requireLive();
    const plan = planAceSemanticDecode(shape);
    requireCount(bindings.weights.layers, ACE_DETOKENIZER_LAYER_COUNT, `${label} weights`);
    requireCount(bindings.scratch.layerOutputs, ACE_DETOKENIZER_LAYER_COUNT, `${label} outputs`);
    validateAceSemanticDecodeBindingAliases(label, bindings);

    const dispatches: CompositeDispatch[] = [];
    const quantumPrimitiveCounts: number[] = [];
    dispatches.push(await this.fsq.createDispatch(`${label}-fsq-decode`, {
      codeCount: plan.codeCount,
    }, {
      codeIds: bindings.codeIds,
      output: bindings.scratch.fsqScalars,
      validationStatus: bindings.controls.validationStatus,
    }));
    quantumPrimitiveCounts.push(1);
    dispatches.push(await this.gemm.createDispatch(`${label}-fsq-project-out`, {
      rows: plan.codeCount,
      inner: ACE_FSQ_CODE_DIMENSION,
      columns: ACE_CONDITION_HIDDEN_SIZE,
    }, {
      activation: bindings.scratch.fsqScalars,
      weight: bindings.weights.fsqProjectOut,
      bias: bindings.weights.fsqProjectOutBias,
      output: bindings.scratch.quantized,
    }));
    quantumPrimitiveCounts.push(1);
    dispatches.push(await this.gemm.createDispatch(`${label}-detokenizer-input`, {
      rows: plan.codeCount,
      inner: ACE_CONDITION_HIDDEN_SIZE,
      columns: ACE_CONDITION_HIDDEN_SIZE,
    }, {
      activation: bindings.scratch.quantized,
      weight: bindings.weights.detokenizerInputProjection,
      bias: bindings.weights.detokenizerInputProjectionBias,
      output: bindings.scratch.embeddedCodes,
    }));
    quantumPrimitiveCounts.push(1);
    dispatches.push(await this.detokenizerExpand.createDispatch(
      `${label}-repeat-add-specials`,
      { codeCount: plan.codeCount, width: ACE_CONDITION_HIDDEN_SIZE },
      {
        embeddedCodes: bindings.scratch.embeddedCodes,
        specialTokens: bindings.weights.specialTokens,
        output: bindings.scratch.patchInput,
      },
    ));
    quantumPrimitiveCounts.push(1);

    let layerInput = bindings.scratch.patchInput;
    let blockPrimitiveCount = 0;
    for (let layer = 0; layer < ACE_DETOKENIZER_LAYER_COUNT; layer += 1) {
      const block = await this.encoder.createBlockDispatch(
        `${label}-layer-${layer}`,
        ACE_CONDITION_ENCODER_CONFIG,
        {
          batch: plan.codeCount,
          tokens: ACE_SEMANTIC_POOL_WIDTH,
          attentionMode: aceEncoderLayerAttentionMode(layer),
        },
        {
          input: layerInput,
          output: bindings.scratch.layerOutputs[layer]!,
          weights: bindings.weights.layers[layer]!,
          scratch: bindings.scratch.block,
          validLengths: bindings.controls.validLengths,
          cosine: bindings.controls.cosine,
          sine: bindings.controls.sine,
        },
      );
      dispatches.push(block);
      quantumPrimitiveCounts.push(block.primitiveCount);
      blockPrimitiveCount += block.primitiveCount;
      layerInput = bindings.scratch.layerOutputs[layer]!;
    }
    dispatches.push(await this.encoder.createFinalNormDispatch(
      `${label}-final-norm`,
      ACE_CONDITION_ENCODER_CONFIG,
      plan.codeCount * ACE_SEMANTIC_POOL_WIDTH,
      {
        input: layerInput,
        weight: bindings.weights.finalNorm,
        output: bindings.scratch.normalized,
      },
    ));
    quantumPrimitiveCounts.push(1);
    dispatches.push(await this.gemm.createDispatch(`${label}-acoustic-output`, {
      rows: plan.codeCount * ACE_SEMANTIC_POOL_WIDTH,
      inner: ACE_CONDITION_HIDDEN_SIZE,
      columns: ACE_AUDIO_LATENT_CHANNELS,
    }, {
      activation: bindings.scratch.normalized,
      weight: bindings.weights.acousticProjection,
      bias: bindings.weights.acousticProjectionBias,
      output: bindings.output,
    }));
    quantumPrimitiveCounts.push(1);
    this.requireLive(" while compiling");
    const quanta = createConditionerQuanta(label, dispatches, quantumPrimitiveCounts);
    if (quanta.length < plan.quantumCount) {
      throw new Error(`${label} semantic quantum expansion lost logical work`);
    }
    const primitiveCount =
      dispatches.length - ACE_DETOKENIZER_LAYER_COUNT + blockPrimitiveCount;
    if (!Number.isSafeInteger(primitiveCount) || primitiveCount <= 0) {
      throw new Error(`${label} semantic primitive accounting changed`);
    }
    return Object.freeze({
      label,
      plan,
      primitiveCount,
      quanta,
      encode(pass: GPUComputePassEncoder): void {
        for (const quantum of quanta) quantum.encode(pass);
      },
    });
  }

  async createDirectConditionerDispatch(
    label: string,
    shape: AceDirectConditionerShape,
    bindings: AceDirectConditionerBindings,
  ): Promise<AceDirectConditionerDispatch> {
    return await this.createConditionerDispatch(
      label,
      shape,
      { kind: "direct-silence" },
      bindings,
    );
  }

  /**
   * Shared learned conditioner graph for direct and planner-generated music.
   * Planner callers own preparation of the exact-size semantic source binding;
   * this composer never silently falls back to silence for a cover request.
   */
  async createConditionerDispatch(
    label: string,
    shape: AceDirectConditionerShape,
    sourceMode: AceConditionerDispatchSource,
    bindings: AceDirectConditionerBindings,
  ): Promise<AceDirectConditionerDispatch> {
    this.requireLive();
    const plan = planAceConditioner(shape, sourceMode);
    requireCount(bindings.weights.lyricLayers, ACE_LYRIC_ENCODER_LAYER_COUNT, `${label} lyric weights`);
    requireCount(bindings.scratch.lyricLayerOutputs, ACE_LYRIC_ENCODER_LAYER_COUNT, `${label} lyric outputs`);
    requireCount(bindings.weights.timbreLayers, ACE_TIMBRE_ENCODER_LAYER_COUNT, `${label} timbre weights`);
    requireCount(bindings.scratch.timbreLayerOutputs, ACE_TIMBRE_ENCODER_LAYER_COUNT, `${label} timbre outputs`);
    validateAceConditionerBindingAliases(label, sourceMode, bindings);

    const dispatches: CompositeDispatch[] = [];
    const quantumPrimitiveCounts: number[] = [];
    let blockPrimitiveCount = 0;
    if (sourceMode.kind === "planner-semantic-cover") {
      dispatches.push(await this.conditionLayout.createSemanticSourceDispatch(
        `${label}-planner-semantic-source`,
        {
          batch: plan.batch,
          semanticFrames: sourceMode.semanticFrames,
          outputFrames: plan.latentFrames,
        },
        {
          semanticHints: sourceMode.semanticHints,
          silenceSource: bindings.silenceSource,
          output: bindings.scratch.sourceLatents,
        },
      ));
      quantumPrimitiveCounts.push(1);
    }
    dispatches.push(await this.gemm.createDispatch(`${label}-text-projector`, {
      rows: plan.textRows,
      inner: ACE_TEXT_EMBEDDING_WIDTH,
      columns: ACE_CONDITION_HIDDEN_SIZE,
    }, {
      activation: bindings.textHiddenStates,
      weight: bindings.weights.textProjection,
      output: bindings.scratch.textProjected,
    }));
    quantumPrimitiveCounts.push(1);
    dispatches.push(await this.gemm.createDispatch(`${label}-lyric-input`, {
      rows: plan.lyricRows,
      inner: ACE_TEXT_EMBEDDING_WIDTH,
      columns: ACE_CONDITION_HIDDEN_SIZE,
    }, {
      activation: bindings.lyricHiddenStates,
      weight: bindings.weights.lyricInputProjection,
      bias: bindings.weights.lyricInputProjectionBias,
      output: bindings.scratch.lyricProjected,
    }));
    quantumPrimitiveCounts.push(1);
    let lyricState = bindings.scratch.lyricProjected;
    for (let layer = 0; layer < ACE_LYRIC_ENCODER_LAYER_COUNT; layer += 1) {
      const block = await this.encoder.createBlockDispatch(
        `${label}-lyric-layer-${layer}`,
        ACE_CONDITION_ENCODER_CONFIG,
        {
          batch: plan.batch,
          tokens: plan.lyricTokens,
          attentionMode: aceEncoderLayerAttentionMode(layer),
        },
        {
          input: lyricState,
          output: bindings.scratch.lyricLayerOutputs[layer]!,
          weights: bindings.weights.lyricLayers[layer]!,
          scratch: bindings.scratch.lyricBlock,
          validLengths: bindings.controls.lyricValidLengths,
          cosine: bindings.controls.lyricCosine,
          sine: bindings.controls.lyricSine,
        },
      );
      dispatches.push(block);
      quantumPrimitiveCounts.push(block.primitiveCount);
      blockPrimitiveCount += block.primitiveCount;
      lyricState = bindings.scratch.lyricLayerOutputs[layer]!;
    }
    dispatches.push(await this.encoder.createFinalNormDispatch(
      `${label}-lyric-final-norm`,
      ACE_CONDITION_ENCODER_CONFIG,
      plan.lyricRows,
      {
        input: lyricState,
        weight: bindings.weights.lyricFinalNorm,
        output: bindings.scratch.lyricEncoded,
      },
    ));
    quantumPrimitiveCounts.push(1);

    dispatches.push(await this.conditionLayout.createSilenceExpandDispatch(
      `${label}-no-reference-timbre-source`,
      { batch: plan.batch, frames: ACE_NO_REFERENCE_TIMBRE_FRAMES },
      {
        source: bindings.silenceSource,
        output: bindings.scratch.timbreSource,
      },
    ));
    quantumPrimitiveCounts.push(1);
    dispatches.push(await this.gemm.createDispatch(`${label}-timbre-input`, {
      rows: plan.timbreRows,
      inner: ACE_AUDIO_LATENT_CHANNELS,
      columns: ACE_CONDITION_HIDDEN_SIZE,
    }, {
      activation: bindings.scratch.timbreSource,
      weight: bindings.weights.timbreInputProjection,
      bias: bindings.weights.timbreInputProjectionBias,
      output: bindings.scratch.timbreProjected,
    }));
    quantumPrimitiveCounts.push(1);
    let timbreState = bindings.scratch.timbreProjected;
    for (let layer = 0; layer < ACE_TIMBRE_ENCODER_LAYER_COUNT; layer += 1) {
      const block = await this.encoder.createBlockDispatch(
        `${label}-timbre-layer-${layer}`,
        ACE_CONDITION_ENCODER_CONFIG,
        {
          batch: plan.batch,
          tokens: ACE_NO_REFERENCE_TIMBRE_FRAMES,
          attentionMode: aceEncoderLayerAttentionMode(layer),
        },
        {
          input: timbreState,
          output: bindings.scratch.timbreLayerOutputs[layer]!,
          weights: bindings.weights.timbreLayers[layer]!,
          scratch: bindings.scratch.timbreBlock,
          validLengths: bindings.controls.timbreValidLengths,
          cosine: bindings.controls.timbreCosine,
          sine: bindings.controls.timbreSine,
        },
      );
      dispatches.push(block);
      quantumPrimitiveCounts.push(block.primitiveCount);
      blockPrimitiveCount += block.primitiveCount;
      timbreState = bindings.scratch.timbreLayerOutputs[layer]!;
    }
    dispatches.push(await this.encoder.createFinalNormDispatch(
      `${label}-timbre-final-norm`,
      ACE_CONDITION_ENCODER_CONFIG,
      plan.timbreRows,
      {
        input: timbreState,
        weight: bindings.weights.timbreFinalNorm,
        output: bindings.scratch.timbreNormalized,
      },
    ));
    quantumPrimitiveCounts.push(1);
    dispatches.push(await this.copy.createGatherRowsDispatch(
      `${label}-timbre-first-row`,
      {
        outer: plan.batch,
        sourceRows: ACE_NO_REFERENCE_TIMBRE_FRAMES,
        outputRows: 1,
        width: ACE_CONDITION_HIDDEN_SIZE,
      },
      {
        input: bindings.scratch.timbreNormalized,
        indices: bindings.controls.timbreFirstRowIndices,
        output: bindings.scratch.timbreToken,
      },
    ));
    quantumPrimitiveCounts.push(1);

    dispatches.push(await this.copy.createStablePackDispatch(
      `${label}-pack-lyric-timbre`,
      {
        batch: plan.batch,
        leftLength: plan.lyricTokens,
        rightLength: 1,
        width: ACE_CONDITION_HIDDEN_SIZE,
      },
      {
        left: bindings.scratch.lyricEncoded,
        right: bindings.scratch.timbreToken,
        leftMask: bindings.lyricMask,
        rightMask: bindings.controls.timbreMask,
        indicesScratch: bindings.scratch.firstPackIndices,
        output: bindings.scratch.firstPacked,
        outputMask: bindings.scratch.firstPackedMask,
      },
    ));
    quantumPrimitiveCounts.push(2);
    dispatches.push(await this.copy.createStablePackDispatch(
      `${label}-pack-text`,
      {
        batch: plan.batch,
        leftLength: plan.firstPackedTokens,
        rightLength: plan.textTokens,
        width: ACE_CONDITION_HIDDEN_SIZE,
      },
      {
        left: bindings.scratch.firstPacked,
        right: bindings.scratch.textProjected,
        leftMask: bindings.scratch.firstPackedMask,
        rightMask: bindings.textMask,
        indicesScratch: bindings.scratch.secondPackIndices,
        output: bindings.output.conditionHiddenStates,
        outputMask: bindings.output.conditionMask,
      },
    ));
    quantumPrimitiveCounts.push(2);
    if (sourceMode.kind === "direct-silence") {
      dispatches.push(await this.conditionLayout.createSilenceExpandDispatch(
        `${label}-direct-source-latents`,
        { batch: plan.batch, frames: plan.latentFrames },
        {
          source: bindings.silenceSource,
          output: bindings.scratch.sourceLatents,
        },
      ));
      quantumPrimitiveCounts.push(1);
    }
    dispatches.push(await this.conditionLayout.createDirectContextDispatch(
      `${label}-${sourceMode.kind === "direct-silence" ? "direct" : "planner-cover"}-context`,
      { batch: plan.batch, frames: plan.latentFrames },
      {
        sourceLatents: bindings.scratch.sourceLatents,
        chunkMask: bindings.chunkMask,
        output: bindings.output.contextLatents,
      },
    ));
    quantumPrimitiveCounts.push(1);
    this.requireLive(" while compiling");
    const blockCount = ACE_LYRIC_ENCODER_LAYER_COUNT + ACE_TIMBRE_ENCODER_LAYER_COUNT;
    const quanta = createConditionerQuanta(label, dispatches, quantumPrimitiveCounts);
    if (quanta.length < plan.quantumCount) {
      throw new Error(`${label} direct-conditioner quantum expansion lost logical work`);
    }
    const primitiveCount = dispatches.length - blockCount + blockPrimitiveCount + 2;
    if (!Number.isSafeInteger(primitiveCount) || primitiveCount <= 0) {
      throw new Error(`${label} direct-conditioner primitive accounting changed`);
    }
    return Object.freeze({
      label,
      plan,
      primitiveCount,
      quanta,
      encode(pass: GPUComputePassEncoder): void {
        for (const quantum of quanta) quantum.encode(pass);
      },
    });
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.fsq.destroy();
    this.detokenizerExpand.destroy();
    this.gemm.destroy();
    this.copy.destroy();
    this.conditionLayout.destroy();
    this.encoder.destroy();
  }

  private requireLive(suffix = ""): void {
    if (this.destroyed) {
      throw new Error(`ACE semantic/conditioner runtime was destroyed${suffix}`);
    }
  }
}

interface NamedBinding {
  readonly name: string;
  readonly binding: GPUBufferBinding;
}

/** Rejects cross-dispatch read/write aliases before any GPU work is compiled. */
export function validateAceSemanticDecodeBindingAliases(
  label: string,
  bindings: AceSemanticDecodeBindings,
): void {
  requireCount(
    bindings.weights.layers,
    ACE_DETOKENIZER_LAYER_COUNT,
    `${label} weights`,
  );
  requireCount(
    bindings.scratch.layerOutputs,
    ACE_DETOKENIZER_LAYER_COUNT,
    `${label} outputs`,
  );
  const writable: NamedBinding[] = [
    named("output", bindings.output),
    named("validation status", bindings.controls.validationStatus),
    named("scratch fsq scalars", bindings.scratch.fsqScalars),
    named("scratch quantized", bindings.scratch.quantized),
    named("scratch embedded codes", bindings.scratch.embeddedCodes),
    named("scratch patch input", bindings.scratch.patchInput),
    named("scratch normalized", bindings.scratch.normalized),
    ...namedRecord("block scratch", bindings.scratch.block),
  ];
  const readonlyBindings: NamedBinding[] = [
    named("code IDs", bindings.codeIds),
    named("FSQ project-out weight", bindings.weights.fsqProjectOut),
    named("FSQ project-out bias", bindings.weights.fsqProjectOutBias),
    named(
      "detokenizer input weight",
      bindings.weights.detokenizerInputProjection,
    ),
    named(
      "detokenizer input bias",
      bindings.weights.detokenizerInputProjectionBias,
    ),
    named("detokenizer special tokens", bindings.weights.specialTokens),
    ...bindings.weights.layers.flatMap((layer, index) =>
      namedRecord(`layer ${index} weight`, layer)
    ),
    named("final norm weight", bindings.weights.finalNorm),
    named("acoustic projection weight", bindings.weights.acousticProjection),
    named("acoustic projection bias", bindings.weights.acousticProjectionBias),
    named("valid lengths", bindings.controls.validLengths),
    named("cosine", bindings.controls.cosine),
    named("sine", bindings.controls.sine),
  ];
  requireGraphBindingAliases(
    label,
    writable,
    readonlyBindings,
    [{ name: "detokenizer layer output", bindings: bindings.scratch.layerOutputs }],
  );
}

/**
 * Validates the complete planner-disabled direct graph. Layer-state ranges may
 * be reused only as an exact A/B/A ping-pong after the intervening layer has
 * consumed the old value; every other graph write stays disjoint.
 */
export function validateAceDirectConditionerBindingAliases(
  label: string,
  bindings: AceDirectConditionerBindings,
): void {
  validateAceConditionerBindingAliases(
    label,
    { kind: "direct-silence" },
    bindings,
  );
}

/** Fail-closed graph alias validation for either exact source route. */
export function validateAceConditionerBindingAliases(
  label: string,
  sourceMode: AceConditionerDispatchSource,
  bindings: AceDirectConditionerBindings,
): void {
  validateAceConditionerSourceMode(sourceMode);
  requireCount(
    bindings.weights.lyricLayers,
    ACE_LYRIC_ENCODER_LAYER_COUNT,
    `${label} lyric weights`,
  );
  requireCount(
    bindings.scratch.lyricLayerOutputs,
    ACE_LYRIC_ENCODER_LAYER_COUNT,
    `${label} lyric outputs`,
  );
  requireCount(
    bindings.weights.timbreLayers,
    ACE_TIMBRE_ENCODER_LAYER_COUNT,
    `${label} timbre weights`,
  );
  requireCount(
    bindings.scratch.timbreLayerOutputs,
    ACE_TIMBRE_ENCODER_LAYER_COUNT,
    `${label} timbre outputs`,
  );
  const writable: NamedBinding[] = [
    named("condition hidden-state output", bindings.output.conditionHiddenStates),
    named("condition mask output", bindings.output.conditionMask),
    named("context output", bindings.output.contextLatents),
    named("scratch text projected", bindings.scratch.textProjected),
    named("scratch lyric projected", bindings.scratch.lyricProjected),
    ...namedRecord("lyric block scratch", bindings.scratch.lyricBlock),
    named("scratch lyric encoded", bindings.scratch.lyricEncoded),
    named("scratch timbre source", bindings.scratch.timbreSource),
    named("scratch timbre projected", bindings.scratch.timbreProjected),
    ...namedRecord("timbre block scratch", bindings.scratch.timbreBlock),
    named("scratch timbre normalized", bindings.scratch.timbreNormalized),
    named("scratch timbre token", bindings.scratch.timbreToken),
    named("scratch first-pack indices", bindings.scratch.firstPackIndices),
    named("scratch first packed", bindings.scratch.firstPacked),
    named("scratch first packed mask", bindings.scratch.firstPackedMask),
    named("scratch second-pack indices", bindings.scratch.secondPackIndices),
    named("scratch source latents", bindings.scratch.sourceLatents),
  ];
  const readonlyBindings: NamedBinding[] = [
    named("text hidden states", bindings.textHiddenStates),
    named("lyric hidden states", bindings.lyricHiddenStates),
    named("text mask", bindings.textMask),
    named("lyric mask", bindings.lyricMask),
    named("silence source", bindings.silenceSource),
    named("chunk mask", bindings.chunkMask),
    ...(sourceMode.kind === "planner-semantic-cover"
      ? [named("semantic detokenizer hints", sourceMode.semanticHints)]
      : []),
    named("text projection weight", bindings.weights.textProjection),
    named("lyric input weight", bindings.weights.lyricInputProjection),
    named("lyric input bias", bindings.weights.lyricInputProjectionBias),
    ...bindings.weights.lyricLayers.flatMap((layer, index) =>
      namedRecord(`lyric layer ${index} weight`, layer)
    ),
    named("lyric final norm weight", bindings.weights.lyricFinalNorm),
    named("timbre input weight", bindings.weights.timbreInputProjection),
    named("timbre input bias", bindings.weights.timbreInputProjectionBias),
    ...bindings.weights.timbreLayers.flatMap((layer, index) =>
      namedRecord(`timbre layer ${index} weight`, layer)
    ),
    named("timbre final norm weight", bindings.weights.timbreFinalNorm),
    named("lyric valid lengths", bindings.controls.lyricValidLengths),
    named("lyric cosine", bindings.controls.lyricCosine),
    named("lyric sine", bindings.controls.lyricSine),
    named("timbre valid lengths", bindings.controls.timbreValidLengths),
    named("timbre cosine", bindings.controls.timbreCosine),
    named("timbre sine", bindings.controls.timbreSine),
    named("timbre first-row indices", bindings.controls.timbreFirstRowIndices),
    named("timbre mask", bindings.controls.timbreMask),
  ];
  requireGraphBindingAliases(label, writable, readonlyBindings, [
    { name: "lyric layer output", bindings: bindings.scratch.lyricLayerOutputs },
    { name: "timbre layer output", bindings: bindings.scratch.timbreLayerOutputs },
  ]);
}

export function planAceSemanticDecode(
  shape: AceSemanticDecodeShape,
): AceSemanticDecodePlan {
  requirePositiveInteger(shape.batch, "ACE semantic batch");
  requirePositiveInteger(shape.codeTokens, "ACE semantic codeTokens");
  const codeCount = checkedProduct(
    [shape.batch, shape.codeTokens],
    "ACE semantic code count",
  );
  const outputFrames = checkedProduct(
    [shape.codeTokens, ACE_SEMANTIC_POOL_WIDTH],
    "ACE semantic output frames",
  );
  const outputElements = checkedProduct(
    [shape.batch, outputFrames, ACE_AUDIO_LATENT_CHANNELS],
    "ACE semantic output",
  );
  // Validates every block activation against the U32 shader domain.
  for (let layer = 0; layer < ACE_DETOKENIZER_LAYER_COUNT; layer += 1) {
    planAceEncoderBlock(ACE_CONDITION_ENCODER_CONFIG, {
      batch: codeCount,
      tokens: ACE_SEMANTIC_POOL_WIDTH,
      attentionMode: aceEncoderLayerAttentionMode(layer),
    });
  }
  const stages: readonly AceGraphStage[] = Object.freeze([
    stage("fsq.inverse", [shape.batch, shape.codeTokens, ACE_FSQ_CODE_DIMENSION], "semantic"),
    stage("fsq.project_out", [shape.batch, shape.codeTokens, ACE_CONDITION_HIDDEN_SIZE], "semantic"),
    stage("detokenizer.embed_tokens", [shape.batch, shape.codeTokens, ACE_CONDITION_HIDDEN_SIZE], "semantic"),
    stage("detokenizer.repeat_add_special_tokens", [codeCount, ACE_SEMANTIC_POOL_WIDTH, ACE_CONDITION_HIDDEN_SIZE], "semantic"),
    stage("detokenizer.layer.0.sliding", [codeCount, ACE_SEMANTIC_POOL_WIDTH, ACE_CONDITION_HIDDEN_SIZE], "semantic"),
    stage("detokenizer.layer.1.full", [codeCount, ACE_SEMANTIC_POOL_WIDTH, ACE_CONDITION_HIDDEN_SIZE], "semantic"),
    stage("detokenizer.norm", [codeCount, ACE_SEMANTIC_POOL_WIDTH, ACE_CONDITION_HIDDEN_SIZE], "semantic"),
    stage("detokenizer.proj_out", [shape.batch, outputFrames, ACE_AUDIO_LATENT_CHANNELS], "semantic"),
  ]);
  return Object.freeze({
    ...shape,
    codeCount,
    outputFrames,
    outputElements,
    detokenizerBatch: codeCount,
    detokenizerTokens: ACE_SEMANTIC_POOL_WIDTH,
    layerAttentionModes: Object.freeze(["sliding", "full"] as const),
    quantumCount: 8,
    stages,
  });
}

export function planAceDirectConditioner(
  shape: AceDirectConditionerShape,
): AceDirectConditionerPlan {
  return planAceConditioner(shape, { kind: "direct-silence" });
}

/** Plan the common conditioner while pinning the upstream source selection. */
export function planAceConditioner(
  shape: AceDirectConditionerShape,
  sourceMode: AceConditionerSourceMode,
): AceDirectConditionerPlan {
  validateAceConditionerSourceMode(sourceMode);
  requirePositiveInteger(shape.batch, "ACE direct conditioner batch");
  requirePositiveInteger(shape.textTokens, "ACE direct conditioner textTokens");
  requirePositiveInteger(shape.lyricTokens, "ACE direct conditioner lyricTokens");
  requirePositiveInteger(shape.latentFrames, "ACE direct conditioner latentFrames");
  if (shape.textTokens > ACE_MAX_TEXT_TOKENS) {
    throw new RangeError(`ACE direct textTokens exceed ${ACE_MAX_TEXT_TOKENS}`);
  }
  if (shape.lyricTokens > ACE_MAX_LYRIC_TOKENS) {
    throw new RangeError(`ACE direct lyricTokens exceed ${ACE_MAX_LYRIC_TOKENS}`);
  }
  const textRows = checkedProduct([shape.batch, shape.textTokens], "ACE text rows");
  const lyricRows = checkedProduct([shape.batch, shape.lyricTokens], "ACE lyric rows");
  const timbreRows = checkedProduct(
    [shape.batch, ACE_NO_REFERENCE_TIMBRE_FRAMES],
    "ACE timbre rows",
  );
  const firstPackedTokens = shape.lyricTokens + 1;
  const conditionTokens = firstPackedTokens + shape.textTokens;
  const conditionElements = checkedProduct(
    [shape.batch, conditionTokens, ACE_CONDITION_HIDDEN_SIZE],
    "ACE packed condition",
  );
  const contextElements = checkedProduct(
    [shape.batch, shape.latentFrames, ACE_CONTEXT_CHANNELS],
    "ACE direct context",
  );
  const lyricLayerAttentionModes = Object.freeze(
    Array.from(
      { length: ACE_LYRIC_ENCODER_LAYER_COUNT },
      (_, layer) => aceEncoderLayerAttentionMode(layer),
    ),
  );
  const timbreLayerAttentionModes = Object.freeze(
    Array.from(
      { length: ACE_TIMBRE_ENCODER_LAYER_COUNT },
      (_, layer) => aceEncoderLayerAttentionMode(layer),
    ),
  );
  for (const mode of lyricLayerAttentionModes) {
    planAceEncoderBlock(ACE_CONDITION_ENCODER_CONFIG, {
      batch: shape.batch,
      tokens: shape.lyricTokens,
      attentionMode: mode,
    });
  }
  for (const mode of timbreLayerAttentionModes) {
    planAceEncoderBlock(ACE_CONDITION_ENCODER_CONFIG, {
      batch: shape.batch,
      tokens: ACE_NO_REFERENCE_TIMBRE_FRAMES,
      attentionMode: mode,
    });
  }
  const stages: readonly AceGraphStage[] = Object.freeze([
    stage("condition.text_projector", [shape.batch, shape.textTokens, ACE_CONDITION_HIDDEN_SIZE], "conditioner"),
    stage("condition.lyric_input_projection", [shape.batch, shape.lyricTokens, ACE_CONDITION_HIDDEN_SIZE], "conditioner"),
    ...lyricLayerAttentionModes.map((mode, layer) =>
      stage(`condition.lyric_layer.${layer}.${mode}`, [shape.batch, shape.lyricTokens, ACE_CONDITION_HIDDEN_SIZE], "conditioner")
    ),
    stage("condition.lyric_norm", [shape.batch, shape.lyricTokens, ACE_CONDITION_HIDDEN_SIZE], "conditioner"),
    stage("condition.no_reference_silence", [shape.batch, ACE_NO_REFERENCE_TIMBRE_FRAMES, ACE_AUDIO_LATENT_CHANNELS], "constant"),
    stage("condition.timbre_input_projection", [shape.batch, ACE_NO_REFERENCE_TIMBRE_FRAMES, ACE_CONDITION_HIDDEN_SIZE], "conditioner"),
    ...timbreLayerAttentionModes.map((mode, layer) =>
      stage(`condition.timbre_layer.${layer}.${mode}`, [shape.batch, ACE_NO_REFERENCE_TIMBRE_FRAMES, ACE_CONDITION_HIDDEN_SIZE], "conditioner")
    ),
    stage("condition.timbre_norm_first_row", [shape.batch, 1, ACE_CONDITION_HIDDEN_SIZE], "conditioner"),
    stage("condition.pack_lyric_timbre_text", [shape.batch, conditionTokens, ACE_CONDITION_HIDDEN_SIZE], "conditioner"),
    stage(
      sourceMode.kind === "direct-silence"
        ? "condition.direct_context"
        : "condition.planner_semantic_context",
      [shape.batch, shape.latentFrames, ACE_CONTEXT_CHANNELS],
      "constant",
    ),
  ]);
  const plannerEnabled = sourceMode.kind === "planner-semantic-cover";
  return Object.freeze({
    ...shape,
    plannerEnabled,
    semanticCodeCount: plannerEnabled ? sourceMode.semanticCodeCount : 0,
    referenceMode: "silence-750",
    sourceSelection: plannerEnabled
      ? "semantic-precomputed-lm-hints-cover"
      : "silence-direct-no-cover",
    textRows,
    lyricRows,
    timbreRows,
    firstPackedTokens,
    conditionTokens,
    conditionElements,
    contextChannels: ACE_CONTEXT_CHANNELS,
    contextElements,
    lyricLayerAttentionModes,
    timbreLayerAttentionModes,
    quantumCount: 23,
    stages,
  });
}

function validateAceConditionerSourceMode(
  sourceMode: AceConditionerSourceMode,
): void {
  if (sourceMode.kind === "direct-silence") return;
  if (sourceMode.kind !== "planner-semantic-cover") {
    throw new TypeError(
      `Unknown ACE conditioner source mode ${String((sourceMode as { kind?: unknown }).kind)}`,
    );
  }
  requirePositiveInteger(
    sourceMode.semanticCodeCount,
    "ACE planner conditioner semanticCodeCount",
  );
  requirePositiveInteger(
    sourceMode.semanticFrames,
    "ACE planner conditioner semanticFrames",
  );
}

/** Exact upstream repeat/add before `(B*T,5,H)` detokenizer attention. */
export function expandAceDetokenizerPatchesCpu(
  embeddedCodes: Float32Array,
  specialTokens: Float32Array,
  codeCount: number,
  width: number,
): Float32Array {
  requirePositiveInteger(codeCount, "ACE detokenizer CPU codeCount");
  requirePositiveInteger(width, "ACE detokenizer CPU width");
  if (embeddedCodes.length !== codeCount * width) {
    throw new RangeError("ACE detokenizer CPU embedded-code shape mismatch");
  }
  if (specialTokens.length !== ACE_SEMANTIC_POOL_WIDTH * width) {
    throw new RangeError("ACE detokenizer CPU special-token shape mismatch");
  }
  const output = new Float32Array(
    codeCount * ACE_SEMANTIC_POOL_WIDTH * width,
  );
  for (let code = 0; code < codeCount; code += 1) {
    for (let patch = 0; patch < ACE_SEMANTIC_POOL_WIDTH; patch += 1) {
      for (let column = 0; column < width; column += 1) {
        output[(code * ACE_SEMANTIC_POOL_WIDTH + patch) * width + column] =
          Math.fround(
            embeddedCodes[code * width + column]! +
              specialTokens[patch * width + column]!,
          );
      }
    }
  }
  return output;
}

export interface AcePackedSequencesCpu {
  readonly hiddenStates: Float32Array;
  readonly mask: Uint32Array;
  readonly sourceIndices: Uint32Array;
}

/** Stable concatenate/valid-first partition matching upstream `pack_sequences`. */
export function packAceSequencesCpu(
  left: Float32Array,
  right: Float32Array,
  leftMask: readonly number[],
  rightMask: readonly number[],
  batch: number,
  leftLength: number,
  rightLength: number,
  width: number,
): AcePackedSequencesCpu {
  for (const [name, value] of Object.entries({
    batch,
    leftLength,
    rightLength,
    width,
  })) {
    requirePositiveInteger(value, `ACE CPU pack ${name}`);
  }
  const leftRows = batch * leftLength;
  const rightRows = batch * rightLength;
  if (left.length !== leftRows * width || right.length !== rightRows * width) {
    throw new RangeError("ACE CPU pack hidden-state shape mismatch");
  }
  if (leftMask.length !== leftRows || rightMask.length !== rightRows) {
    throw new RangeError("ACE CPU pack mask shape mismatch");
  }
  const packedLength = leftLength + rightLength;
  const hiddenStates = new Float32Array(batch * packedLength * width);
  const mask = new Uint32Array(batch * packedLength);
  const sourceIndices = new Uint32Array(batch * packedLength);
  for (let item = 0; item < batch; item += 1) {
    const valid: number[] = [];
    const invalid: number[] = [];
    for (let row = 0; row < packedLength; row += 1) {
      const value = row < leftLength
        ? leftMask[item * leftLength + row]
        : rightMask[item * rightLength + row - leftLength];
      if (value !== 0 && value !== 1) {
        throw new RangeError("ACE CPU pack masks must contain only zero or one");
      }
      (value === 1 ? valid : invalid).push(row);
    }
    const order = [...valid, ...invalid];
    for (let packedRow = 0; packedRow < packedLength; packedRow += 1) {
      const sourceRow = order[packedRow]!;
      sourceIndices[item * packedLength + packedRow] = sourceRow;
      mask[item * packedLength + packedRow] = packedRow < valid.length ? 1 : 0;
      const destinationBase = (item * packedLength + packedRow) * width;
      if (sourceRow < leftLength) {
        const sourceBase = (item * leftLength + sourceRow) * width;
        hiddenStates.set(left.subarray(sourceBase, sourceBase + width), destinationBase);
      } else {
        const rightRow = sourceRow - leftLength;
        const sourceBase = (item * rightLength + rightRow) * width;
        hiddenStates.set(right.subarray(sourceBase, sourceBase + width), destinationBase);
      }
    }
  }
  return Object.freeze({ hiddenStates, mask, sourceIndices });
}

export function createAceNoReferenceTimbreControls(batch: number): Readonly<{
  firstRowIndices: Uint32Array;
  mask: Uint32Array;
  orderMask: Uint32Array;
}> {
  requirePositiveInteger(batch, "ACE no-reference timbre batch");
  const firstRowIndices = new Uint32Array(batch);
  const mask = new Uint32Array(batch);
  const orderMask = new Uint32Array(batch);
  mask.fill(1);
  for (let item = 0; item < batch; item += 1) orderMask[item] = item;
  return Object.freeze({ firstRowIndices, mask, orderMask });
}

/**
 * Pinned v1 direct text-to-music has no repaint span. Upstream builds a bool
 * all-ones mask; assigning the default `auto` sentinel `2.0` to that bool
 * tensor remains `true`, so the value reaching `prepare_condition` is one.
 */
export function createAceDirectV1ChunkMask(
  batch: number,
  latentFrames: number,
): Uint32Array {
  requirePositiveInteger(batch, "ACE direct chunk-mask batch");
  requirePositiveInteger(latentFrames, "ACE direct chunk-mask latentFrames");
  const output = new Uint32Array(
    checkedProduct([batch, latentFrames], "ACE direct chunk mask"),
  );
  output.fill(1);
  return output;
}

export function aceEncoderLayerTensorNames(
  prefix: string,
  layer: number,
): AceEncoderLayerTensorNames {
  if (prefix.length === 0) throw new TypeError("ACE tensor prefix must be non-empty");
  if (!Number.isSafeInteger(layer) || layer < 0) {
    throw new RangeError("ACE tensor layer must be a non-negative integer");
  }
  const base = `${prefix}.layers.${layer}`;
  return Object.freeze({
    inputLayerNorm: `${base}.input_layernorm.weight`,
    queryProjection: `${base}.self_attn.q_proj.weight`,
    keyProjection: `${base}.self_attn.k_proj.weight`,
    valueProjection: `${base}.self_attn.v_proj.weight`,
    queryNorm: `${base}.self_attn.q_norm.weight`,
    keyNorm: `${base}.self_attn.k_norm.weight`,
    outputProjection: `${base}.self_attn.o_proj.weight`,
    postAttentionLayerNorm: `${base}.post_attention_layernorm.weight`,
    gateProjection: `${base}.mlp.gate_proj.weight`,
    upProjection: `${base}.mlp.up_proj.weight`,
    downProjection: `${base}.mlp.down_proj.weight`,
  });
}

function stage(
  id: string,
  outputShape: readonly number[],
  weightLifetime: AceGraphStage["weightLifetime"],
): AceGraphStage {
  return Object.freeze({ id, outputShape: Object.freeze([...outputShape]), weightLifetime });
}

function requireCount(
  values: readonly unknown[],
  expected: number,
  label: string,
): void {
  if (values.length !== expected) {
    throw new RangeError(`${label} has ${values.length} entries; ${expected} required`);
  }
}

function requirePositiveInteger(value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new RangeError(`${label} must be a positive safe integer`);
  }
}

function checkedProduct(values: readonly number[], label: string): number {
  let result = 1;
  for (const value of values) {
    result *= value;
    if (!Number.isSafeInteger(result)) {
      throw new RangeError(`${label} is not a safe integer`);
    }
  }
  return result;
}

function named(name: string, binding: GPUBufferBinding): NamedBinding {
  return { name, binding };
}

function namedRecord(
  prefix: string,
  bindings: object,
): NamedBinding[] {
  return Object.entries(bindings).map(([name, binding]) =>
    named(`${prefix} ${name}`, binding as GPUBufferBinding)
  );
}

function requireGraphBindingAliases(
  label: string,
  writable: readonly NamedBinding[],
  readonlyBindings: readonly NamedBinding[],
  layerGroups: readonly Readonly<{
    name: string;
    bindings: readonly GPUBufferBinding[];
  }>[],
): void {
  const layers = layerGroups.map((group) => ({
    name: group.name,
    entries: group.bindings.map((binding, index) =>
      named(`${group.name} ${index}`, binding)
    ),
  }));
  for (const entry of [
    ...writable,
    ...readonlyBindings,
    ...layers.flatMap((group) => group.entries),
  ]) {
    bindingRange(entry.binding, `${label} ${entry.name}`);
  }

  for (let leftIndex = 0; leftIndex < writable.length; leftIndex += 1) {
    const left = writable[leftIndex]!;
    for (let rightIndex = leftIndex + 1; rightIndex < writable.length; rightIndex += 1) {
      requireNamedDisjoint(label, left, writable[rightIndex]!, "writable");
    }
    for (const right of readonlyBindings) {
      requireNamedDisjoint(label, left, right, "read-only");
    }
  }

  for (let groupIndex = 0; groupIndex < layers.length; groupIndex += 1) {
    const group = layers[groupIndex]!;
    for (let leftIndex = 0; leftIndex < group.entries.length; leftIndex += 1) {
      const left = group.entries[leftIndex]!;
      for (const right of writable) {
        requireNamedDisjoint(label, left, right, "writable");
      }
      for (const right of readonlyBindings) {
        requireNamedDisjoint(label, left, right, "read-only");
      }
      for (
        let rightIndex = leftIndex + 1;
        rightIndex < group.entries.length;
        rightIndex += 1
      ) {
        const right = group.entries[rightIndex]!;
        if (!bindingsOverlap(left.binding, right.binding)) continue;
        const exactReuse = bindingRangesEqual(left.binding, right.binding);
        if (!exactReuse || rightIndex - leftIndex < 2) {
          throw new RangeError(
            `${label} ${left.name} overlaps live ${right.name}; only exact non-adjacent ping-pong reuse is permitted`,
          );
        }
      }
      for (let otherIndex = groupIndex + 1; otherIndex < layers.length; otherIndex += 1) {
        for (const right of layers[otherIndex]!.entries) {
          requireNamedDisjoint(label, left, right, "other encoder output");
        }
      }
    }
  }
}

function requireNamedDisjoint(
  label: string,
  left: NamedBinding,
  right: NamedBinding,
  rightKind: string,
): void {
  if (bindingsOverlap(left.binding, right.binding)) {
    throw new RangeError(
      `${label} ${left.name} overlaps ${rightKind} ${right.name}`,
    );
  }
}

function bindingsOverlap(
  left: GPUBufferBinding,
  right: GPUBufferBinding,
): boolean {
  if (left.buffer !== right.buffer) return false;
  const leftRange = bindingRange(left, "ACE graph left binding");
  const rightRange = bindingRange(right, "ACE graph right binding");
  return leftRange.start < rightRange.end && rightRange.start < leftRange.end;
}

function bindingRangesEqual(
  left: GPUBufferBinding,
  right: GPUBufferBinding,
): boolean {
  if (left.buffer !== right.buffer) return false;
  const leftRange = bindingRange(left, "ACE graph left binding");
  const rightRange = bindingRange(right, "ACE graph right binding");
  return leftRange.start === rightRange.start && leftRange.end === rightRange.end;
}

function bindingRange(
  binding: GPUBufferBinding,
  label: string,
): { readonly start: number; readonly end: number } {
  const start = binding.offset ?? 0;
  const size = binding.size ?? binding.buffer.size - start;
  if (
    !Number.isSafeInteger(start) ||
    start < 0 ||
    !Number.isSafeInteger(size) ||
    size <= 0 ||
    start + size > binding.buffer.size
  ) {
    throw new RangeError(`${label} is not a valid buffer range`);
  }
  return { start, end: start + size };
}
