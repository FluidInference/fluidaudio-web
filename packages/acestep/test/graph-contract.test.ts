import { describe, expect, it } from "vitest";

import {
  ACE_DIT_LAYER_TYPES,
  ACE_GRAPH_CONTRACT,
  assertAceGraphContract,
  deriveAceDurationGraphShape,
} from "../src/model/graph-contract.js";

describe("frozen ACE graph contract", () => {
  it("is internally consistent and alternates local/global DiT layers", () => {
    expect(assertAceGraphContract).not.toThrow();
    expect(ACE_DIT_LAYER_TYPES).toHaveLength(24);
    expect(ACE_DIT_LAYER_TYPES.slice(0, 4)).toEqual([
      "sliding_attention",
      "full_attention",
      "sliding_attention",
      "full_attention",
    ]);
    expect(ACE_GRAPH_CONTRACT.dit.maximumSlidingKeys).toBe(257);
  });

  it.each([
    [12, 300, 300, 0, 150, 60, 576_000],
    [13, 325, 326, 1, 163, 65, 624_000],
    [180, 4_500, 4_500, 0, 2_250, 900, 8_640_000],
    [240, 6_000, 6_000, 0, 3_000, 1_200, 11_520_000],
  ])(
    "derives exact duration geometry for %i seconds",
    (
      duration,
      latentFrames,
      paddedLatentFrames,
      latentPaddingFrames,
      ditTokens,
      semanticCodes,
      audioFrames,
    ) => {
      expect(deriveAceDurationGraphShape(duration)).toEqual({
        durationSeconds: duration,
        latentFrames,
        paddedLatentFrames,
        latentPaddingFrames,
        ditTokens,
        semanticCodes,
        audioFramesPerChannel: audioFrames,
        audioFloat32Bytes: audioFrames * 2 * Float32Array.BYTES_PER_ELEMENT,
      });
    },
  );

  it.each([9, 241, 12.5, Number.NaN, Number.POSITIVE_INFINITY])(
    "rejects unsupported duration %s before allocation",
    (duration) => {
      expect(() => deriveAceDurationGraphShape(duration)).toThrow();
    },
  );

  it("keeps the exact audited tensor-payload accounting", () => {
    expect(ACE_GRAPH_CONTRACT.tensorPayloadBytes).toEqual({
      planner: 1_325_768_704,
      textEncoder: 1_191_553_024,
      conditionEncoder: 1_216_735_232,
      semanticDetokenizer: 210_023_552,
      fsqOutputProjection: 28_672,
      dit: 3_150_917_760,
      vae: 337_583_104,
      silenceLatent: 3_840_000,
      total: 7_436_450_048,
    });
  });
});
