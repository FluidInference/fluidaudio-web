import { describe, expect, it } from "vitest";

import {
  aceEncoderLayerAttentionMode,
  createAceEncoderControlData,
  createAceEncoderFullControlData,
  createAceEncoderRopeTables,
  planAceEncoderBlock,
  validateAceEncoderConfig,
  type AceEncoderConfig,
} from "../src/webgpu/ace-encoder.js";
import { ACE_CONDITION_ENCODER_CONFIG } from "../src/webgpu/semantic-conditioner.js";

describe("ACE bidirectional encoder contract", () => {
  it("pins the shared Turbo encoder geometry", () => {
    expect(ACE_CONDITION_ENCODER_CONFIG).toEqual({
      hiddenSize: 2_048,
      intermediateSize: 6_144,
      queryHeads: 16,
      keyValueHeads: 8,
      headDimension: 128,
      maximumPositionEmbeddings: 32_768,
      ropeTheta: 1_000_000,
      rmsNormEpsilon: 1e-6,
      slidingRadius: 128,
      attentionBias: false,
      hiddenActivation: "silu",
    });
  });

  it("plans the non-square GQA block without a dense score tensor", () => {
    expect(planAceEncoderBlock(ACE_CONDITION_ENCODER_CONFIG, {
      batch: 2,
      tokens: 5,
      attentionMode: "sliding",
    })).toEqual({
      batch: 2,
      tokens: 5,
      attentionMode: "sliding",
      rows: 10,
      queryWidth: 2_048,
      keyValueWidth: 1_024,
      hiddenElements: 20_480,
      queryElements: 20_480,
      keyValueElements: 10_240,
      intermediateElements: 61_440,
    });
  });

  it("pins alternating sliding/full layer types from layer zero", () => {
    expect(Array.from({ length: 8 }, (_, layer) =>
      aceEncoderLayerAttentionMode(layer)
    )).toEqual([
      "sliding",
      "full",
      "sliding",
      "full",
      "sliding",
      "full",
      "sliding",
      "full",
    ]);
  });

  it("converts right padding to key-only prefix controls", () => {
    const controls = createAceEncoderControlData(
      [1, 1, 0, 0, 1, 1, 1, 0],
      2,
      4,
    );
    // All four queries execute exactly like upstream. Only the key prefix is
    // shortened by the supplied 2D attention mask.
    expect([...controls.validLengths]).toEqual([4, 2, 4, 3]);
    expect([...controls.validKeyLengths]).toEqual([2, 3]);

    expect([...createAceEncoderFullControlData(2, 3).validLengths]).toEqual([
      3, 3, 3, 3,
    ]);
  });

  it("rejects holes, empty rows, non-binary masks, and shape drift", () => {
    expect(() => createAceEncoderControlData([1, 0, 1], 1, 3)).toThrow(
      /right-padded/,
    );
    expect(() => createAceEncoderControlData([0, 0], 1, 2)).toThrow(
      /no valid token/,
    );
    expect(() => createAceEncoderControlData([1, 2], 1, 2)).toThrow(
      /zero or one/,
    );
    expect(() => createAceEncoderControlData([1], 1, 2)).toThrow(/entries/);
  });

  it("reuses the exact Qwen split-half RoPE table convention", () => {
    const tables = createAceEncoderRopeTables(2, {
      ...ACE_CONDITION_ENCODER_CONFIG,
      hiddenSize: 4,
      intermediateSize: 6,
      queryHeads: 1,
      keyValueHeads: 1,
      headDimension: 4,
      maximumPositionEmbeddings: 8,
      ropeTheta: 100,
    });
    expect([...tables.cosine.slice(0, 4)]).toEqual([1, 1, 1, 1]);
    expect([...tables.sine.slice(0, 4)]).toEqual([0, 0, 0, 0]);
    expect([...tables.cosine.slice(4)]).toEqual([
      Math.fround(Math.cos(1)),
      Math.fround(Math.cos(0.1)),
      Math.fround(Math.cos(1)),
      Math.fround(Math.cos(0.1)),
    ]);
  });

  it("fails closed for architecture changes", () => {
    const future = (patch: Partial<AceEncoderConfig>): AceEncoderConfig => ({
      ...ACE_CONDITION_ENCODER_CONFIG,
      ...patch,
    }) as AceEncoderConfig;
    expect(() => validateAceEncoderConfig(future({ queryHeads: 15 }))).toThrow(
      /divisible/,
    );
    expect(() => validateAceEncoderConfig(future({ hiddenSize: 2_047 }))).toThrow(
      /query width/,
    );
    expect(() => validateAceEncoderConfig(future({ headDimension: 129 }))).toThrow(
      /even and at most 128/,
    );
    expect(() => planAceEncoderBlock(ACE_CONDITION_ENCODER_CONFIG, {
      batch: 1,
      tokens: 32_769,
      attentionMode: "full",
    })).toThrow(/maximumPositionEmbeddings/);
    expect(() => aceEncoderLayerAttentionMode(-1)).toThrow(/non-negative/);
  });
});
