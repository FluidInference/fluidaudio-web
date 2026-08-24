import { describe, expect, it } from "vitest";
import { DICOSE_STFT_HOP_LENGTH, type StereoPcm } from "../src/runtime/audio.js";
import {
  addDiCoSeCoordinateNoise,
  DICOSE_CHUNK_FADE_SAMPLES,
  DICOSE_CHUNK_SAMPLES,
  DICOSE_FAST_CHUNK_GEOMETRY,
  DICOSE_FAST_CHUNK_STEP_SAMPLES,
  DICOSE_FULL_CHUNK_GEOMETRY,
  DICOSE_FULL_CHUNK_STEP_SAMPLES,
  DICOSE_SINGLE_PASS_SAMPLES,
  makeDiCoSeChunkPlan,
  makeDiCoSeChunkWindow,
  materializeDiCoSeChunk,
  normalizeDiCoSeOverlapAdd,
  overlapAddDiCoSeChunk,
} from "../src/runtime/chunking.js";

describe("DiCoSe arbitrary-length chunk geometry", () => {
  it("keeps every model and overlap boundary aligned to an STFT hop", () => {
    expect(DICOSE_CHUNK_SAMPLES / DICOSE_STFT_HOP_LENGTH).toBe(1_100);
    expect(DICOSE_FULL_CHUNK_STEP_SAMPLES / DICOSE_STFT_HOP_LENGTH).toBe(550);
    expect(DICOSE_FAST_CHUNK_STEP_SAMPLES / DICOSE_STFT_HOP_LENGTH).toBe(990);
    expect(
      (DICOSE_CHUNK_SAMPLES - DICOSE_FAST_CHUNK_STEP_SAMPLES) / DICOSE_STFT_HOP_LENGTH,
    ).toBe(110);
    expect(DICOSE_CHUNK_FADE_SAMPLES / DICOSE_STFT_HOP_LENGTH).toBe(110);
  });

  it("preserves the exact Full trust_nobody whole-track plan", () => {
    const plan = makeDiCoSeChunkPlan(5_608_109, DICOSE_FULL_CHUNK_GEOMETRY);
    expect(plan.borderSamples).toBe(242_550);
    expect(plan.paddedSamples).toBe(6_093_209);
    expect(plan.spans).toHaveLength(25);
    expect(plan.spans[0]).toMatchObject({
      index: 0,
      paddedStart: 0,
      chunkReadStart: 242_550,
      outputStart: 0,
      outputSamples: 242_550,
    });
    expect(plan.spans.at(-1)).toMatchObject({
      index: 24,
      paddedStart: 5_821_200,
      validSamples: 272_009,
      tailPadding: "reflect",
      outputStart: 5_578_650,
      outputSamples: 29_459,
    });
  });

  it("uses 13 low-overlap chunks for Fast trust_nobody inference", () => {
    const plan = makeDiCoSeChunkPlan(5_608_109, DICOSE_FAST_CHUNK_GEOMETRY);
    expect(plan.borderSamples).toBe(48_510);
    expect(plan.paddedSamples).toBe(5_705_129);
    expect(plan.spans).toHaveLength(13);
    expect(plan.spans[0]).toMatchObject({
      index: 0,
      paddedStart: 0,
      validSamples: DICOSE_CHUNK_SAMPLES,
      chunkReadStart: 48_510,
      outputStart: 0,
      outputSamples: 436_590,
    });
    expect(plan.spans.at(-1)).toMatchObject({
      index: 12,
      paddedStart: 5_239_080,
      validSamples: 466_049,
      tailPadding: "reflect",
      chunkReadStart: 0,
      outputStart: 5_190_570,
      outputSamples: 417_539,
    });
  });

  it("uses endpoint-inclusive upstream linear fades", () => {
    expect(Array.from(makeDiCoSeChunkWindow(10, 3))).toEqual([
      0, 0.5, 1, 1, 1, 1, 1, 1, 0.5, 0,
    ]);
  });

  it("makes the Fast overlap ramps complementary", () => {
    const window = makeDiCoSeChunkWindow();
    for (let index = 0; index < DICOSE_CHUNK_FADE_SAMPLES; index += 1) {
      expect(
        window[DICOSE_FAST_CHUNK_STEP_SAMPLES + index]! + window[index]!,
      ).toBeCloseTo(1, 6);
    }
  });

  it("matches torch reflection without duplicating endpoints", () => {
    const source = pcm([0, 1, 2, 3, 4, 5]);
    const plan = {
      sourceSamples: 6,
      paddedSamples: 10,
      borderSamples: 2,
      geometry: DICOSE_FULL_CHUNK_GEOMETRY,
      spans: [],
    } as const;
    const reflected = materializeDiCoSeChunk(source, plan, {
      index: 0,
      paddedStart: 0,
      validSamples: 10,
      tailPadding: "zero",
      chunkReadStart: 0,
      outputStart: 0,
      outputSamples: 6,
    });
    expect(Array.from(reflected.left.slice(0, 10))).toEqual([2, 1, 0, 1, 2, 3, 4, 5, 4, 3]);
  });

  it.each([
    ["Full", DICOSE_FULL_CHUNK_GEOMETRY, DICOSE_CHUNK_SAMPLES + 1],
    ["Full", DICOSE_FULL_CHUNK_GEOMETRY, DICOSE_SINGLE_PASS_SAMPLES + 1],
    [
      "Full",
      DICOSE_FULL_CHUNK_GEOMETRY,
      DICOSE_CHUNK_SAMPLES + DICOSE_FULL_CHUNK_STEP_SAMPLES + 1,
    ],
    ["Fast", DICOSE_FAST_CHUNK_GEOMETRY, DICOSE_CHUNK_SAMPLES + 1],
    ["Fast", DICOSE_FAST_CHUNK_GEOMETRY, DICOSE_SINGLE_PASS_SAMPLES + 1],
    [
      "Fast",
      DICOSE_FAST_CHUNK_GEOMETRY,
      DICOSE_CHUNK_SAMPLES + DICOSE_FAST_CHUNK_STEP_SAMPLES + 1,
    ],
  ] as const)("reassembles a %s identity model without gaps at length %i", (_name, geometry, length) => {
    const source = patternedPcm(length);
    const plan = makeDiCoSeChunkPlan(length, geometry);
    const window = makeDiCoSeChunkWindow(DICOSE_CHUNK_SAMPLES, geometry.fadeSamples);
    const accumulator = {
      left: new Float32Array(length),
      right: new Float32Array(length),
    };
    const denominator = new Float32Array(length);
    for (const span of plan.spans) {
      const chunk = materializeDiCoSeChunk(source, plan, span);
      overlapAddDiCoSeChunk([accumulator], denominator, [chunk], span, window);
    }
    expect(denominator.every((value) => Number.isFinite(value) && value > 0)).toBe(true);
    normalizeDiCoSeOverlapAdd([accumulator], denominator);

    let maxError = 0;
    for (let index = 0; index < length; index += 1) {
      maxError = Math.max(
        maxError,
        Math.abs(accumulator.left[index]! - source.left[index]!),
        Math.abs(accumulator.right[index]! - source.right[index]!),
      );
    }
    expect(maxError).toBeLessThan(2e-6);
  });

  it("uses the upstream adaptive reflection for a partial final model item", () => {
    const source = patternedPcm(DICOSE_SINGLE_PASS_SAMPLES + 1);
    const plan = makeDiCoSeChunkPlan(source.length);
    const span = plan.spans.at(-1)!;
    expect(span.tailPadding).toBe("reflect");
    const chunk = materializeDiCoSeChunk(source, plan, span);
    expect(chunk.left[span.validSamples]).toBe(chunk.left[span.validSamples - 2]);
    expect(chunk.right[span.validSamples + 1]).toBe(chunk.right[span.validSamples - 3]);
  });

  it("keeps CD noise identical at shared padded-track coordinates", () => {
    const source = patternedPcm(DICOSE_SINGLE_PASS_SAMPLES + 1);
    const plan = makeDiCoSeChunkPlan(source.length, DICOSE_FULL_CHUNK_GEOMETRY);
    const first = plan.spans[0]!;
    const second = plan.spans[1]!;
    const zero = pcm(new Float32Array(DICOSE_CHUNK_SAMPLES));
    const firstNoise = addDiCoSeCoordinateNoise(zero, first, 2, 0xd1c05e, 1);
    const secondNoise = addDiCoSeCoordinateNoise(zero, second, 2, 0xd1c05e, 1);
    const overlap = first.paddedStart + DICOSE_FULL_CHUNK_STEP_SAMPLES - second.paddedStart;
    expect(overlap).toBe(0);
    for (let index = 0; index < 256; index += 1) {
      expect(firstNoise.left[DICOSE_FULL_CHUNK_STEP_SAMPLES + index]).toBe(secondNoise.left[index]);
      expect(firstNoise.right[DICOSE_FULL_CHUNK_STEP_SAMPLES + index]).toBe(secondNoise.right[index]);
    }
    expect(firstNoise.left[DICOSE_FULL_CHUNK_STEP_SAMPLES]).not.toBe(
      firstNoise.right[DICOSE_FULL_CHUNK_STEP_SAMPLES],
    );
  });
});

function pcm(values: ArrayLike<number>): StereoPcm {
  const left = Float32Array.from(values);
  const right = Float32Array.from(values);
  return { sampleRate: 44_100, length: values.length, left, right, channels: [left, right] };
}

function patternedPcm(length: number): StereoPcm {
  const left = new Float32Array(length);
  const right = new Float32Array(length);
  for (let index = 0; index < length; index += 1) {
    left[index] = Math.fround((((index * 17) % 251) - 125) / 128);
    right[index] = Math.fround((((index * 29 + 7) % 257) - 128) / 128);
  }
  return { sampleRate: 44_100, length, left, right, channels: [left, right] };
}
