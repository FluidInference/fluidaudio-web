import { describe, expect, it } from "vitest";

import {
  ACE_AUDIO_LATENT_CHANNELS,
  ACE_CONDITION_HIDDEN_SIZE,
  ACE_DETOKENIZER_LAYER_COUNT,
  ACE_DIRECT_CONDITIONER_TENSOR_NAMES,
  ACE_LYRIC_ENCODER_LAYER_COUNT,
  ACE_MAX_LYRIC_TOKENS,
  ACE_MAX_TEXT_TOKENS,
  ACE_NO_REFERENCE_TIMBRE_FRAMES,
  ACE_SEMANTIC_POOL_WIDTH,
  ACE_SEMANTIC_TENSOR_NAMES,
  ACE_TIMBRE_ENCODER_LAYER_COUNT,
  aceEncoderLayerTensorNames,
  createAceDirectV1ChunkMask,
  createAceNoReferenceTimbreControls,
  expandAceDetokenizerPatchesCpu,
  packAceSequencesCpu,
  planAceConditioner,
  planAceDirectConditioner,
  planAceSemanticDecode,
  validateAceDirectConditionerBindingAliases,
  validateAceConditionerBindingAliases,
  validateAceSemanticDecodeBindingAliases,
  type AceDirectConditionerBindings,
  type AceSemanticDecodeBindings,
} from "../src/webgpu/semantic-conditioner.js";
import type {
  AceEncoderBlockScratch,
  AceEncoderBlockWeights,
} from "../src/webgpu/ace-encoder.js";

describe("ACE semantic decode and direct conditioner contract", () => {
  it("pins subsystem dimensions and layer counts", () => {
    expect(ACE_CONDITION_HIDDEN_SIZE).toBe(2_048);
    expect(ACE_AUDIO_LATENT_CHANNELS).toBe(64);
    expect(ACE_SEMANTIC_POOL_WIDTH).toBe(5);
    expect(ACE_DETOKENIZER_LAYER_COUNT).toBe(2);
    expect(ACE_LYRIC_ENCODER_LAYER_COUNT).toBe(8);
    expect(ACE_TIMBRE_ENCODER_LAYER_COUNT).toBe(4);
    expect(ACE_NO_REFERENCE_TIMBRE_FRAMES).toBe(750);
    expect(ACE_MAX_TEXT_TOKENS).toBe(256);
    expect(ACE_MAX_LYRIC_TOKENS).toBe(2_048);
  });

  it("plans exact 5 Hz to 25 Hz semantic reconstruction", () => {
    const plan = planAceSemanticDecode({ batch: 2, codeTokens: 12 });
    expect(plan).toMatchObject({
      batch: 2,
      codeTokens: 12,
      codeCount: 24,
      outputFrames: 60,
      outputElements: 7_680,
      detokenizerBatch: 24,
      detokenizerTokens: 5,
      layerAttentionModes: ["sliding", "full"],
      quantumCount: 8,
    });
    expect(plan.stages.map((entry) => entry.id)).toEqual([
      "fsq.inverse",
      "fsq.project_out",
      "detokenizer.embed_tokens",
      "detokenizer.repeat_add_special_tokens",
      "detokenizer.layer.0.sliding",
      "detokenizer.layer.1.full",
      "detokenizer.norm",
      "detokenizer.proj_out",
    ]);
  });

  it("plans planner-disabled direct mode as a first-class graph", () => {
    const plan = planAceDirectConditioner({
      batch: 1,
      textTokens: 32,
      lyricTokens: 64,
      latentFrames: 300,
    });
    expect(plan).toMatchObject({
      plannerEnabled: false,
      semanticCodeCount: 0,
      referenceMode: "silence-750",
      sourceSelection: "silence-direct-no-cover",
      textRows: 32,
      lyricRows: 64,
      timbreRows: 750,
      firstPackedTokens: 65,
      conditionTokens: 97,
      conditionElements: 198_656,
      contextChannels: 128,
      contextElements: 38_400,
      quantumCount: 23,
    });
    expect(plan.lyricLayerAttentionModes).toEqual([
      "sliding", "full", "sliding", "full",
      "sliding", "full", "sliding", "full",
    ]);
    expect(plan.timbreLayerAttentionModes).toEqual([
      "sliding", "full", "sliding", "full",
    ]);
    expect(plan.stages.some((entry) => entry.id.includes("semantic"))).toBe(false);
  });

  it("routes planner semantic hints through the shared conditioner graph", () => {
    const plan = planAceConditioner({
      batch: 1,
      textTokens: 32,
      lyricTokens: 64,
      latentFrames: 300,
    }, {
      kind: "planner-semantic-cover",
      semanticCodeCount: 60,
      semanticFrames: 300,
    });
    expect(plan).toMatchObject({
      plannerEnabled: true,
      semanticCodeCount: 60,
      referenceMode: "silence-750",
      sourceSelection: "semantic-precomputed-lm-hints-cover",
      quantumCount: 23,
    });
    expect(plan.stages.at(-1)?.id).toBe("condition.planner_semantic_context");
    expect(() => planAceConditioner({
      batch: 1,
      textTokens: 1,
      lyricTokens: 1,
      latentFrames: 1,
    }, {
      kind: "planner-semantic-cover",
      semanticCodeCount: 0,
      semanticFrames: 5,
    })).toThrow(/semanticCodeCount/);
  });

  it("repeats each semantic token into five patches before adding specials", () => {
    const actual = expandAceDetokenizerPatchesCpu(
      new Float32Array([1, 10, 2, 20]),
      new Float32Array([
        0, 0,
        0.1, 1,
        0.2, 2,
        0.3, 3,
        0.4, 4,
      ]),
      2,
      2,
    );
    expect([...actual]).toEqual([
      1, 10,
      Math.fround(1.1), 11,
      Math.fround(1.2), 12,
      Math.fround(1.3), 13,
      Math.fround(1.4), 14,
      2, 20,
      Math.fround(2.1), 21,
      Math.fround(2.2), 22,
      Math.fround(2.3), 23,
      Math.fround(2.4), 24,
    ]);
  });

  it("matches upstream stable valid-first sequence packing", () => {
    const packed = packAceSequencesCpu(
      new Float32Array([10, 11, 12]),
      new Float32Array([20, 21]),
      [1, 0, 1],
      [0, 1],
      1,
      3,
      2,
      1,
    );
    expect([...packed.hiddenStates]).toEqual([10, 12, 21, 11, 20]);
    expect([...packed.mask]).toEqual([1, 1, 1, 0, 0]);
    expect([...packed.sourceIndices]).toEqual([0, 2, 4, 1, 3]);
  });

  it("pins no-reference timbre unpack controls", () => {
    const controls = createAceNoReferenceTimbreControls(3);
    expect([...controls.firstRowIndices]).toEqual([0, 0, 0]);
    expect([...controls.mask]).toEqual([1, 1, 1]);
    expect([...controls.orderMask]).toEqual([0, 1, 2]);
    expect([...createAceDirectV1ChunkMask(2, 3)]).toEqual([
      1, 1, 1, 1, 1, 1,
    ]);
  });

  it("maps every generic layer binding to canonical package tensors", () => {
    expect(ACE_SEMANTIC_TENSOR_NAMES.fsqProjectOut).toBe(
      "ace.tokenizer.quantizer.project_out.weight",
    );
    expect(ACE_DIRECT_CONDITIONER_TENSOR_NAMES.silenceSource).toBe(
      "constants.silence_latent",
    );
    expect(aceEncoderLayerTensorNames("ace.detokenizer", 1)).toEqual({
      inputLayerNorm: "ace.detokenizer.layers.1.input_layernorm.weight",
      queryProjection: "ace.detokenizer.layers.1.self_attn.q_proj.weight",
      keyProjection: "ace.detokenizer.layers.1.self_attn.k_proj.weight",
      valueProjection: "ace.detokenizer.layers.1.self_attn.v_proj.weight",
      queryNorm: "ace.detokenizer.layers.1.self_attn.q_norm.weight",
      keyNorm: "ace.detokenizer.layers.1.self_attn.k_norm.weight",
      outputProjection: "ace.detokenizer.layers.1.self_attn.o_proj.weight",
      postAttentionLayerNorm:
        "ace.detokenizer.layers.1.post_attention_layernorm.weight",
      gateProjection: "ace.detokenizer.layers.1.mlp.gate_proj.weight",
      upProjection: "ace.detokenizer.layers.1.mlp.up_proj.weight",
      downProjection: "ace.detokenizer.layers.1.mlp.down_proj.weight",
    });
  });

  it("rejects scoped token and shape violations", () => {
    expect(() => planAceSemanticDecode({ batch: 1, codeTokens: 0 })).toThrow(
      /positive/,
    );
    expect(() => planAceDirectConditioner({
      batch: 1,
      textTokens: 257,
      lyricTokens: 1,
      latentFrames: 1,
    })).toThrow(/textTokens/);
    expect(() => planAceDirectConditioner({
      batch: 1,
      textTokens: 1,
      lyricTokens: 2_049,
      latentFrames: 1,
    })).toThrow(/lyricTokens/);
    expect(planAceDirectConditioner({
      batch: 1,
      textTokens: 1,
      lyricTokens: 1,
      latentFrames: 15_001,
    }).latentFrames).toBe(15_001);
    expect(() => packAceSequencesCpu(
      new Float32Array([1]),
      new Float32Array([2]),
      [1],
      [2],
      1,
      1,
      1,
      1,
    )).toThrow(/zero or one/);
  });

  it("rejects semantic cross-dispatch aliases before compilation", () => {
    const bindings = semanticBindings();
    expect(() =>
      validateAceSemanticDecodeBindingAliases("semantic", bindings)
    ).not.toThrow();

    expect(() => validateAceSemanticDecodeBindingAliases("semantic", {
      ...bindings,
      output: bindings.weights.acousticProjection,
    })).toThrow(/output overlaps read-only acoustic projection weight/);

    expect(() => validateAceSemanticDecodeBindingAliases("semantic", {
      ...bindings,
      scratch: {
        ...bindings.scratch,
        layerOutputs: [
          bindings.scratch.layerOutputs[0]!,
          bindings.scratch.layerOutputs[0]!,
        ],
      },
    })).toThrow(/overlaps live/);

    expect(() => validateAceSemanticDecodeBindingAliases("semantic", {
      ...bindings,
      scratch: {
        ...bindings.scratch,
        block: {
          ...bindings.scratch.block,
          gate: bindings.controls.cosine,
        },
      },
    })).toThrow(/block scratch gate overlaps read-only cosine/);

    expect(() => validateAceSemanticDecodeBindingAliases("semantic", {
      ...bindings,
      scratch: {
        ...bindings.scratch,
        block: {
          ...bindings.scratch.block,
          gate: bindings.scratch.layerOutputs[0]!,
        },
      },
    })).toThrow(/layer output 0 overlaps writable block scratch gate/);
  });

  it("permits only exact non-adjacent direct layer ping-pong aliases", () => {
    const bindings = directBindings();
    const lyricA = gpuBinding();
    const lyricB = gpuBinding();
    const pingPong = {
      ...bindings,
      scratch: {
        ...bindings.scratch,
        lyricLayerOutputs: [
          lyricA, lyricB, lyricA, lyricB, lyricA, lyricB, lyricA, lyricB,
        ],
      },
    } satisfies AceDirectConditionerBindings;
    expect(() =>
      validateAceDirectConditionerBindingAliases("direct", pingPong)
    ).not.toThrow();

    expect(() => validateAceDirectConditionerBindingAliases("direct", {
      ...pingPong,
      scratch: {
        ...pingPong.scratch,
        lyricLayerOutputs: [
          lyricA, lyricA, lyricA, lyricB, lyricA, lyricB, lyricA, lyricB,
        ],
      },
    })).toThrow(/overlaps live/);

    expect(() => validateAceDirectConditionerBindingAliases("direct", {
      ...bindings,
      output: {
        ...bindings.output,
        contextLatents: bindings.weights.timbreFinalNorm,
      },
    })).toThrow(/context output overlaps read-only timbre final norm weight/);
  });

  it("treats a prepared planner semantic source as read-only", () => {
    const bindings = directBindings();
    expect(() => validateAceConditionerBindingAliases(
      "planner-conditioner",
      {
        kind: "planner-semantic-cover",
        semanticCodeCount: 4,
        semanticFrames: 20,
        semanticHints: gpuBinding(),
      },
      bindings,
    )).not.toThrow();
    expect(() => validateAceConditionerBindingAliases(
      "planner-conditioner",
      {
        kind: "planner-semantic-cover",
        semanticCodeCount: 4,
        semanticFrames: 20,
        semanticHints: bindings.output.contextLatents,
      },
      bindings,
    )).toThrow(/context output overlaps read-only semantic detokenizer hints/);
  });
});

const ENCODER_WEIGHT_NAMES = [
  "inputLayerNorm",
  "queryProjection",
  "keyProjection",
  "valueProjection",
  "queryNorm",
  "keyNorm",
  "outputProjection",
  "postAttentionLayerNorm",
  "gateProjection",
  "upProjection",
  "downProjection",
] as const;

const ENCODER_SCRATCH_NAMES = [
  "normalizedInput",
  "queryFlat",
  "keyFlat",
  "valueFlat",
  "queryHeads",
  "keyHeads",
  "valueHeads",
  "normalizedQueryHeads",
  "normalizedKeyHeads",
  "rotatedQueryHeads",
  "rotatedKeyHeads",
  "attentionHeads",
  "mergedAttention",
  "projectedAttention",
  "afterAttention",
  "normalizedAfterAttention",
  "gate",
  "up",
  "gatedActivation",
  "projectedMlp",
] as const;

function gpuBinding(): GPUBufferBinding {
  return {
    buffer: { size: 1 << 20 } as GPUBuffer,
    offset: 0,
    size: 1 << 20,
  };
}

function encoderWeights(): AceEncoderBlockWeights {
  return Object.fromEntries(
    ENCODER_WEIGHT_NAMES.map((name) => [name, gpuBinding()]),
  ) as unknown as AceEncoderBlockWeights;
}

function encoderScratch(): AceEncoderBlockScratch {
  return Object.fromEntries(
    ENCODER_SCRATCH_NAMES.map((name) => [name, gpuBinding()]),
  ) as unknown as AceEncoderBlockScratch;
}

function semanticBindings(): AceSemanticDecodeBindings {
  return {
    codeIds: gpuBinding(),
    output: gpuBinding(),
    weights: {
      fsqProjectOut: gpuBinding(),
      fsqProjectOutBias: gpuBinding(),
      detokenizerInputProjection: gpuBinding(),
      detokenizerInputProjectionBias: gpuBinding(),
      specialTokens: gpuBinding(),
      layers: [encoderWeights(), encoderWeights()],
      finalNorm: gpuBinding(),
      acousticProjection: gpuBinding(),
      acousticProjectionBias: gpuBinding(),
    },
    controls: {
      validationStatus: gpuBinding(),
      validLengths: gpuBinding(),
      cosine: gpuBinding(),
      sine: gpuBinding(),
    },
    scratch: {
      fsqScalars: gpuBinding(),
      quantized: gpuBinding(),
      embeddedCodes: gpuBinding(),
      patchInput: gpuBinding(),
      block: encoderScratch(),
      layerOutputs: [gpuBinding(), gpuBinding()],
      normalized: gpuBinding(),
    },
  };
}

function directBindings(): AceDirectConditionerBindings {
  return {
    textHiddenStates: gpuBinding(),
    lyricHiddenStates: gpuBinding(),
    textMask: gpuBinding(),
    lyricMask: gpuBinding(),
    silenceSource: gpuBinding(),
    chunkMask: gpuBinding(),
    output: {
      conditionHiddenStates: gpuBinding(),
      conditionMask: gpuBinding(),
      contextLatents: gpuBinding(),
    },
    weights: {
      textProjection: gpuBinding(),
      lyricInputProjection: gpuBinding(),
      lyricInputProjectionBias: gpuBinding(),
      lyricLayers: Array.from({ length: 8 }, encoderWeights),
      lyricFinalNorm: gpuBinding(),
      timbreInputProjection: gpuBinding(),
      timbreInputProjectionBias: gpuBinding(),
      timbreLayers: Array.from({ length: 4 }, encoderWeights),
      timbreFinalNorm: gpuBinding(),
    },
    controls: {
      lyricValidLengths: gpuBinding(),
      lyricCosine: gpuBinding(),
      lyricSine: gpuBinding(),
      timbreValidLengths: gpuBinding(),
      timbreCosine: gpuBinding(),
      timbreSine: gpuBinding(),
      timbreFirstRowIndices: gpuBinding(),
      timbreMask: gpuBinding(),
    },
    scratch: {
      textProjected: gpuBinding(),
      lyricProjected: gpuBinding(),
      lyricBlock: encoderScratch(),
      lyricLayerOutputs: Array.from({ length: 8 }, gpuBinding),
      lyricEncoded: gpuBinding(),
      timbreSource: gpuBinding(),
      timbreProjected: gpuBinding(),
      timbreBlock: encoderScratch(),
      timbreLayerOutputs: Array.from({ length: 4 }, gpuBinding),
      timbreNormalized: gpuBinding(),
      timbreToken: gpuBinding(),
      firstPackIndices: gpuBinding(),
      firstPacked: gpuBinding(),
      firstPackedMask: gpuBinding(),
      secondPackIndices: gpuBinding(),
      sourceLatents: gpuBinding(),
    },
  };
}
