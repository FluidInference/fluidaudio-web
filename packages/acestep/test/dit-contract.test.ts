import { describe, expect, it } from "vitest";

import {
  ACE_DIRECT_DCW_CONFIGURATION,
  ACE_THINKING_DCW_CONFIGURATION,
  ACE_TURBO_V1_CORRECTNESS_PROFILE,
} from "../src/api.js";
import {
  ACE_TURBO_DIT_CONFIG,
  aceDitLayerAttentionMode,
  createAceDitAttentionControlData,
  createAceDitRopeTables,
  createAceDitTurboSamplerSchedule,
  planAceDitEvaluation,
  planAceDitLayer,
  roundAceBfloat16ToNumber,
  validateAceDitInputProjectionBindingAliases,
  validateAceDitConfig,
} from "../src/webgpu/ace-dit.js";

describe("ACE-Step 1.5 Turbo DiT graph contract", () => {
  it("freezes the exact pinned decoder architecture and mask quirk", () => {
    expect(ACE_TURBO_DIT_CONFIG).toMatchObject({
      hiddenSize: 2_048,
      intermediateSize: 6_144,
      layerCount: 24,
      queryHeads: 16,
      keyValueHeads: 8,
      headDimension: 128,
      patchSize: 2,
      inputChannels: 192,
      contextChannels: 128,
      audioChannels: 64,
      conditionInputSize: 2_048,
      slidingRadius: 128,
      ropeTheta: 1_000_000,
      timestepInputSize: 256,
      timestepScale: 1_000,
      timestepMaximumPeriod: 10_000,
      maskPolicy: "discard-all-supplied-masks",
    });
    expect(() => validateAceDitConfig(ACE_TURBO_DIT_CONFIG)).not.toThrow();
  });

  it("derives short, odd, and maximum-duration activation geometry", () => {
    expect(planAceDitEvaluation(ACE_TURBO_DIT_CONFIG, {
      batch: 1,
      latentFrames: 400,
      conditionTokens: 37,
    })).toMatchObject({
      paddedLatentFrames: 400,
      latentPaddingFrames: 0,
      tokens: 200,
      rows: 200,
      hiddenElements: 409_600,
      intermediateElements: 1_228_800,
      queryElements: 409_600,
      selfKeyValueElements: 204_800,
      crossKeyValueElementsPerLayer: 37_888,
      crossKeyValueElementsAllLayers: 1_818_624,
      contextElements: 51_200,
      latentElements: 25_600,
    });
    expect(planAceDitEvaluation(ACE_TURBO_DIT_CONFIG, {
      batch: 1,
      latentFrames: 325,
      conditionTokens: 1,
    })).toMatchObject({
      paddedLatentFrames: 326,
      latentPaddingFrames: 1,
      tokens: 163,
    });
    expect(planAceDitEvaluation(ACE_TURBO_DIT_CONFIG, {
      batch: 1,
      latentFrames: 6_000,
      conditionTokens: 1,
    })).toMatchObject({ tokens: 3_000, rows: 3_000 });
  });

  it("plans one exact AdaLN/self/cross/SwiGLU layer", () => {
    expect(planAceDitLayer(ACE_TURBO_DIT_CONFIG, {
      batch: 2,
      tokens: 200,
      conditionTokens: 37,
      attentionMode: "sliding",
    })).toEqual({
      batch: 2,
      tokens: 200,
      conditionTokens: 37,
      attentionMode: "sliding",
      rows: 400,
      queryWidth: 2_048,
      keyValueWidth: 1_024,
      hiddenElements: 819_200,
      queryElements: 819_200,
      selfKeyValueElements: 409_600,
      crossKeyValueElements: 75_776,
      intermediateElements: 2_457_600,
      modulationElements: 24_576,
    });
  });

  it("alternates inclusive-radius local and bidirectional full attention", () => {
    expect(Array.from({ length: 24 }, (_, index) =>
      aceDitLayerAttentionMode(index)
    )).toEqual(Array.from({ length: 24 }, (_, index) =>
      index % 2 === 0 ? "sliding" : "full"
    ));
    expect(() => aceDitLayerAttentionMode(24)).toThrow();
  });

  it("admits every self and cross row because upstream discards both masks", () => {
    const controls = createAceDitAttentionControlData({
      batch: 2,
      latentFrames: 325,
      conditionTokens: 37,
    });
    expect(Array.from(controls.selfValidLengths)).toEqual([163, 163, 163, 163]);
    expect(Array.from(controls.crossValidLengths)).toEqual([163, 37, 163, 37]);
  });

  it("separates declared shift-3 values from native BF16 effective values", () => {
    const schedule = createAceDitTurboSamplerSchedule(
      ACE_DIRECT_DCW_CONFIGURATION,
    );
    expect(schedule.map((step) => step.declaredTimestep)).toEqual(
      ACE_TURBO_V1_CORRECTNESS_PROFILE.schedulerTimesteps,
    );
    expect(schedule.map((step) => step.timestep)).toEqual(
      ACE_TURBO_V1_CORRECTNESS_PROFILE.effectiveSamplerTimestepsBfloat16,
    );
    expect(schedule.map((step) => step.updateCoefficient)).toEqual([
      0.046875,
      0.0546875,
      0.06640625,
      0.08203125,
      0.10546875,
      0.14453125,
      0.19921875,
      0.30078125,
    ]);
    expect(schedule.map((step) => step.update)).toEqual([
      "euler", "euler", "euler", "euler",
      "euler", "euler", "euler", "predicted-clean",
    ]);
    expect(schedule[0]).toMatchObject({ lowBandScale: Math.fround(0.05), highBandScale: 0 });
    expect(schedule[7]).toMatchObject({
      lowBandScale: Math.fround(0.30078125 * 0.05),
      highBandScale: Math.fround((1 - 0.30078125) * 0.02),
    });
    const thinking = createAceDitTurboSamplerSchedule(
      ACE_THINKING_DCW_CONFIGURATION,
    );
    expect(thinking[7]).toMatchObject({
      lowBandScale: Math.fround(0.30078125 * 0.02),
      highBandScale: Math.fround((1 - 0.30078125) * 0.06),
    });
  });

  it("rounds the fixture schedule with round-to-nearest-even BF16", () => {
    expect(
      ACE_TURBO_V1_CORRECTNESS_PROFILE.schedulerTimesteps.map(
        roundAceBfloat16ToNumber,
      ),
    ).toEqual(
      ACE_TURBO_V1_CORRECTNESS_PROFILE.effectiveSamplerTimestepsBfloat16,
    );
    expect(roundAceBfloat16ToNumber(0.75)).toBe(0.75);
    expect(() => roundAceBfloat16ToNumber(Number.NaN)).toThrow();
  });

  it("creates one shared split-half RoPE table for every batch row", () => {
    const tables = createAceDitRopeTables(3);
    expect(tables.cosine).toHaveLength(3 * 128);
    expect(tables.sine).toHaveLength(3 * 128);
    expect(tables.cosine[0]).toBe(1);
    expect(tables.sine[0]).toBe(0);
  });

  it("rejects incompatible graph edits", () => {
    expect(() => validateAceDitConfig({
      ...ACE_TURBO_DIT_CONFIG,
      maskPolicy: "future" as never,
    })).toThrow(/mask-discard/);
    expect(() => planAceDitEvaluation(ACE_TURBO_DIT_CONFIG, {
      batch: 0,
      latentFrames: 400,
      conditionTokens: 1,
    })).toThrow();
    expect(() => planAceDitLayer(ACE_TURBO_DIT_CONFIG, {
      batch: 1,
      tokens: 1,
      conditionTokens: 1,
      attentionMode: "causal" as never,
    })).toThrow(/attention mode/);
  });

  it("rejects input-projection scratch that aliases a later read or output", () => {
    const buffer = { size: 4_096 } as GPUBuffer;
    const binding = (offset: number, size = 256): GPUBufferBinding => ({
      buffer,
      offset,
      size,
    });
    const safe = {
      context: binding(0),
      latent: binding(256),
      concatenated: binding(512),
      weight: binding(768),
      bias: binding(1_024),
      output: binding(1_280),
    };
    expect(() =>
      validateAceDitInputProjectionBindingAliases("input", safe)
    ).not.toThrow();
    expect(() => validateAceDitInputProjectionBindingAliases("input", {
      ...safe,
      concatenated: binding(768),
    })).toThrow(/concatenated overlaps weight/);
    expect(() => validateAceDitInputProjectionBindingAliases("input", {
      ...safe,
      output: binding(512),
    })).toThrow(/concatenated overlaps output/);
  });
});
