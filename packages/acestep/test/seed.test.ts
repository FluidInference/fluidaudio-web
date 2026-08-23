import { describe, expect, it } from "vitest";

import {
  aceCategoricalTokenFromWord,
  aceGaussianF32FromWord,
  acePlannerCategoricalToken,
  aceRandomWord,
  aceRandomWords,
  canonicalizeSeed,
  fillAceDiffusionNoise,
  philox4x32_10,
} from "../src/runtime/seed.js";
import type { AceRandomStream } from "../src/runtime/seed.js";
import vectors from "../golden/prng/ace-seed-v1-vectors.json";

const FLOAT_BITS_VIEW = new DataView(new ArrayBuffer(4));

function float32Bits(value: number): number {
  FLOAT_BITS_VIEW.setFloat32(0, value, true);
  return FLOAT_BITS_VIEW.getUint32(0, true);
}

describe("ACE seed contract", () => {
  it("preserves all 64 bits in a canonical JSON-safe representation", () => {
    expect(canonicalizeSeed(42)).toBe("000000000000002a");
    expect(canonicalizeSeed("18446744073709551615")).toBe(
      "ffffffffffffffff",
    );
    expect(canonicalizeSeed("0x1234abcd")).toBe("000000001234abcd");
    expect(() => canonicalizeSeed(Number.MAX_SAFE_INTEGER + 1)).toThrow(
      /safe integer/,
    );
    expect(() => canonicalizeSeed(0x1_0000_0000_0000_0000n)).toThrow(
      /unsigned 64-bit/,
    );
  });

  it("matches the published Random123 zero-key Philox4x32-10 vector", () => {
    expect(
      philox4x32_10(
        { word0: 0, word1: 0, word2: 0, word3: 0 },
        [0, 0],
      ),
    ).toEqual({
      word0: 0x6627e8d5,
      word1: 0xe169c58d,
      word2: 0xbc57ac4c,
      word3: 0x9b00dbd8,
    });
  });

  it("domain-separates planner sampling from diffusion noise", () => {
    const seed = canonicalizeSeed(7);
    expect(aceRandomWords(seed, "diffusion-noise", 0)).not.toEqual(
      aceRandomWords(seed, "planner-sampling", 0),
    );
    expect(aceRandomWords(seed, "diffusion-noise", 1)).toEqual(
      aceRandomWords(seed, "diffusion-noise", 1),
    );
  });

  it("matches the independently generated stream vectors", () => {
    for (const vector of vectors.philox) {
      const actual = aceRandomWords(
        canonicalizeSeed(vector.seed),
        vector.stream as AceRandomStream,
        BigInt(`0x${vector.blockIndexHex}`),
      );
      expect(
        [actual.word0, actual.word1, actual.word2, actual.word3],
        vector.id,
      ).toEqual(vector.outputWords);
    }
  });

  it("maps uint32 words to the pinned binary32 Gaussian vectors", () => {
    for (const vector of vectors.gaussian) {
      expect(
        float32Bits(aceGaussianF32FromWord(vector.word)),
        vector.id,
      ).toBe(vector.outputF32Bits);
    }
    for (const vector of vectors.streamGaussian) {
      const seed = canonicalizeSeed(vector.seed);
      expect(
        aceRandomWord(seed, "diffusion-noise", vector.wordIndex),
        vector.id,
      ).toBe(vector.word);
      expect(
        float32Bits(
          aceGaussianF32FromWord(
            aceRandomWord(seed, "diffusion-noise", vector.wordIndex),
          ),
        ),
        vector.id,
      ).toBe(vector.outputF32Bits);
    }
  });

  it("preserves exact odd symmetry across complementary words", () => {
    const words = [0, 1, 17, 0x0635_3f7c, 0x1234_5678, 0x7fff_ffff];
    for (const word of words) {
      const negativeBits = float32Bits(aceGaussianF32FromWord(word));
      const positiveBits = float32Bits(
        aceGaussianF32FromWord(0xffff_ffff - word),
      );
      expect(negativeBits ^ 0x8000_0000).toBe(positiveBits);
    }
  });

  it("fills row-major diffusion slices without shifting the stream", () => {
    const seed = canonicalizeSeed("0000000000c0ffee");
    const output = new Float32Array(11);
    fillAceDiffusionNoise(output, seed, 2);
    for (let index = 0; index < output.length; index += 1) {
      expect(float32Bits(output[index] ?? Number.NaN)).toBe(
        float32Bits(
          aceGaussianF32FromWord(
            aceRandomWord(seed, "diffusion-noise", index + 2),
          ),
        ),
      );
    }
  });

  it("matches categorical boundary and planner-stream vectors", () => {
    for (const vector of vectors.categorical) {
      expect(
        aceCategoricalTokenFromWord(vector.weights, vector.word),
        vector.id,
      ).toBe(vector.token);
    }
    for (const vector of vectors.plannerStream) {
      const seed = canonicalizeSeed(vector.seed);
      expect(
        aceRandomWord(seed, "planner-sampling", vector.drawIndex),
        vector.id,
      ).toBe(vector.word);
      expect(
        acePlannerCategoricalToken(seed, vector.drawIndex, vector.weights),
        vector.id,
      ).toBe(vector.token);
    }
  });

  it("rejects malformed categorical distributions and stream ranges", () => {
    expect(() => aceCategoricalTokenFromWord([], 0)).toThrow(/non-empty/);
    expect(() => aceCategoricalTokenFromWord([0, 0], 0)).toThrow(
      /positive sum/,
    );
    expect(() => aceCategoricalTokenFromWord([1, Number.NaN], 0)).toThrow(
      /finite and nonnegative/,
    );
    expect(() => aceCategoricalTokenFromWord([1, -1], 0)).toThrow(
      /finite and nonnegative/,
    );
    expect(() =>
      aceRandomWord(
        canonicalizeSeed(0),
        "diffusion-noise",
        (0xffff_ffff_ffff_ffffn << 2n) + 4n,
      ),
    ).toThrow(/66 unsigned bits/);
  });
});
