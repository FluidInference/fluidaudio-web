import { createHash } from "node:crypto";
import { describe, expect, it } from "vitest";

import {
  ACE_QWEN3_PINNED_INV_FREQUENCY_WORDS,
  ACE_QWEN3_ROPE_REFERENCE_PROVENANCE,
  ACE_PLANNER_QWEN3_CONFIG,
  ACE_TEXT_QWEN3_CONFIG,
  createAceQwen3CausalControlData,
  createAceQwen3RopeTables,
  planAceQwen3Block,
  validateAceQwen3Config,
  validateAceQwen3TiedOutputBindings,
  type AceQwen3Config,
  type AceQwen3TiedWeightShard,
} from "../src/webgpu/qwen3.js";

describe("ACE Qwen3 runtime contract", () => {
  it("pins the exact text and planner Qwen3 geometries", () => {
    expect(ACE_TEXT_QWEN3_CONFIG).toEqual({
      id: "ace-text-qwen3-embedding-0.6b",
      hiddenSize: 1_024,
      intermediateSize: 3_072,
      layerCount: 28,
      queryHeads: 16,
      keyValueHeads: 8,
      headDimension: 128,
      vocabularySize: 151_669,
      maximumPositionEmbeddings: 32_768,
      ropeTheta: 1_000_000,
      rmsNormEpsilon: 1e-6,
      attentionBias: false,
      hiddenActivation: "silu",
      tieWordEmbeddings: true,
    });
    expect(ACE_PLANNER_QWEN3_CONFIG).toEqual({
      ...ACE_TEXT_QWEN3_CONFIG,
      id: "ace-planner-qwen3-0.6b",
      vocabularySize: 217_204,
      maximumPositionEmbeddings: 40_960,
    });
  });

  it("preserves the non-square Qwen3 Q/K/V geometry", () => {
    const text = planAceQwen3Block(ACE_TEXT_QWEN3_CONFIG, {
      batch: 1,
      tokens: 256,
      attention: { kind: "uncached" },
    });
    expect(text).toMatchObject({
      rows: 256,
      queryWidth: 2_048,
      keyValueWidth: 1_024,
      hiddenElements: 262_144,
      queryElements: 524_288,
      keyValueElements: 262_144,
      intermediateElements: 786_432,
      attentionKeyValueTokens: 256,
    });

    const decode = planAceQwen3Block(ACE_PLANNER_QWEN3_CONFIG, {
      batch: 1,
      tokens: 1,
      attention: { kind: "cached", cacheCapacity: 40_960 },
    });
    expect(decode).toMatchObject({
      rows: 1,
      queryWidth: 2_048,
      keyValueWidth: 1_024,
      attentionKeyValueTokens: 40_960,
    });
  });

  it("derives fail-closed left-padded physical cache controls", () => {
    const controls = createAceQwen3CausalControlData({
      batch: 2,
      tokens: 3,
      cacheCapacity: 8,
      rowStartPositions: [1, 3],
      validKeyLengths: [4, 6],
      sourceValidity: [0, 1, 1, 1, 0, 1],
    });
    expect([...controls.rowStartPositions]).toEqual([1, 3]);
    expect([...controls.validLengths]).toEqual([3, 4, 3, 6]);
    expect([...controls.queryPositions]).toEqual([1, 2, 3, 3, 4, 5]);
    expect([...controls.sourceValidity]).toEqual([0, 1, 1, 1, 0, 1]);
  });

  it("keeps rotary IDs independent of physical cache slots", () => {
    const tables = createAceQwen3RopeTables([0, 3], {
      batch: 1,
      tokens: 2,
      headDimension: 4,
      ropeTheta: 100,
      maximumPositionEmbeddings: 16,
    });
    expect([...tables.cosine.slice(0, 4)]).toEqual([1, 1, 1, 1]);
    expect([...tables.sine.slice(0, 4)]).toEqual([0, 0, 0, 0]);
    expectWordsWithinUlp(
      floatWords(tables.cosine.slice(4)),
      [0xbf7d7026, 0x3f7490ef, 0xbf7d7026, 0x3f7490ef],
      1,
    );
    expectWordsWithinUlp(
      floatWords(tables.sine.slice(4)),
      [0x3e1081c3, 0x3e974e6d, 0x3e1081c3, 0x3e974e6d],
      1,
    );
  });

  it("pins the Transformers FP32 inverse-frequency words and provenance", () => {
    expect(ACE_QWEN3_PINNED_INV_FREQUENCY_WORDS).toEqual([
      0x3f800000, 0x3f4e4bad, 0x3f263de0, 0x3f05f6ef,
      0x3ed7e89b, 0x3eadfcff, 0x3e8c3504, 0x3e61f835,
      0x3e361887, 0x3e12bd91, 0x3dec7fd6, 0x3dbe94c6,
      0x3d99940d, 0x3d778513, 0x3d47763f, 0x3d20bc1d,
      0x3d0186e3, 0x3cd0c1a8, 0x3ca8398b, 0x3c879008,
      0x3c5a7bf2, 0x3c301052, 0x3c0de12d, 0x3be4aa46,
      0x3bb8449c, 0x3b947dae, 0x3b6f520e, 0x3b40dac5,
      0x3b1b690d, 0x3afa78f0, 0x3ac9d75c, 0x3aa2a6f7,
      0x3a83126f, 0x3a533f28, 0x3a2a3b44, 0x3a092e02,
      0x39dd1725, 0x39b229fb, 0x398f9272, 0x39676492,
      0x393a7753, 0x39164324, 0x38f22ce2, 0x38c327b5,
      0x389d43a4, 0x387d75d4, 0x384c3fbe, 0x382497ab,
      0x3804a2b2, 0x37d5c441, 0x37ac431d, 0x378ad0ed,
      0x375fba4f, 0x37344a0e, 0x371148e3, 0x36ea2732,
      0x36bcb0c1, 0x36980e02, 0x36751070, 0x36457bab,
      0x361f23e4, 0x36003dec, 0x35ceaf79, 0x35a68e4c,
    ]);
    expect(sha256LittleEndianWords(ACE_QWEN3_PINNED_INV_FREQUENCY_WORDS)).toBe(
      ACE_QWEN3_ROPE_REFERENCE_PROVENANCE.inverseFrequencyLittleEndianSha256,
    );
    expect(ACE_QWEN3_ROPE_REFERENCE_PROVENANCE).toMatchObject({
      aceSourceRevision: "6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0",
      transformersVersion: "4.57.6",
      torchVersion: "2.10.0",
      transformersQwen3SourceSha256:
        "4b95c371fd26d40c69083dab36ac1eafd8cf82b415a0bb827275097c5ad2305b",
    });
  });

  it.each(TRANSFORMERS_ROPE_VECTORS)(
    "matches Transformers-derived rotary words at position $position within one trig ULP",
    ({ position, cosine, sine }) => {
      const tables = createAceQwen3RopeTables([position], {
        batch: 1,
        tokens: 1,
        headDimension: 128,
        ropeTheta: 1_000_000,
        maximumPositionEmbeddings: 40_960,
      });
      const actualCosine = floatWords(tables.cosine);
      const actualSine = floatWords(tables.sine);
      expectWordsWithinUlp(
        ROPE_VECTOR_DIMENSIONS.map((dimension) => actualCosine[dimension]!),
        cosine,
        1,
      );
      expectWordsWithinUlp(
        ROPE_VECTOR_DIMENSIONS.map((dimension) => actualSine[dimension]!),
        sine,
        1,
      );
    },
  );

  it("fails closed for an unauthenticated rotary geometry", () => {
    expect(() => createAceQwen3RopeTables([0], {
      batch: 1,
      tokens: 1,
      headDimension: 64,
      ropeTheta: 10_000,
      maximumPositionEmbeddings: 1,
    })).toThrow(/no authenticated FP32 inverse-frequency vector/);
  });

  it("supports independent rotary positions for every batch row", () => {
    const tables = createAceQwen3RopeTables([0, 1, 3, 4], {
      batch: 2,
      tokens: 2,
      headDimension: 2,
      ropeTheta: 100,
      maximumPositionEmbeddings: 8,
    });
    expect(tables.cosine[0]).toBe(1);
    expect(tables.cosine[2]).toBe(Math.fround(Math.cos(1)));
    expect(tables.cosine[4]).toBe(Math.fround(Math.cos(3)));
    expect(tables.cosine[6]).toBe(Math.fround(Math.cos(4)));
  });

  it.each([
    {
      input: {
        batch: 1,
        tokens: 2,
        cacheCapacity: 2,
        rowStartPositions: [1],
        validKeyLengths: [2],
        sourceValidity: [1, 1],
      },
      message: /append exceeds/,
    },
    {
      input: {
        batch: 1,
        tokens: 1,
        cacheCapacity: 2,
        rowStartPositions: [0],
        validKeyLengths: [1],
        sourceValidity: [2],
      },
      message: /zero or one/,
    },
    {
      input: {
        batch: 1,
        tokens: 1,
        cacheCapacity: 2,
        rowStartPositions: [1],
        validKeyLengths: [1],
        sourceValidity: [1],
      },
      message: /does not contain/,
    },
  ])("rejects unsafe cache control tensors", ({ input, message }) => {
    expect(() => createAceQwen3CausalControlData(input)).toThrow(message);
  });

  it("fails closed for architecture drift", () => {
    const future = (patch: Partial<AceQwen3Config>): AceQwen3Config => ({
      ...ACE_PLANNER_QWEN3_CONFIG,
      ...patch,
    }) as AceQwen3Config;
    expect(() => validateAceQwen3Config(future({ queryHeads: 15 }))).toThrow(
      /divisible/,
    );
    expect(() => validateAceQwen3Config(future({ headDimension: 129 }))).toThrow(
      /even and at most 128/,
    );
    expect(() => validateAceQwen3Config(future({ attentionBias: true as false }))).toThrow(
      /does not permit attention bias/,
    );
    expect(() =>
      validateAceQwen3Config(future({ hiddenActivation: "gelu" as "silu" })),
    ).toThrow(/requires SiLU/);
    expect(() =>
      validateAceQwen3Config(future({ tieWordEmbeddings: false as true })),
    ).toThrow(/tied word embeddings/);
    expect(() =>
      planAceQwen3Block(ACE_PLANNER_QWEN3_CONFIG, {
        batch: 1,
        tokens: 1,
        attention: { kind: "cached", cacheCapacity: 40_961 },
      }),
    ).toThrow(/maximumPositionEmbeddings/);
    expect(() =>
      planAceQwen3Block(ACE_PLANNER_QWEN3_CONFIG, {
        batch: 1,
        tokens: 1,
        attention: { kind: "future" } as never,
      }),
    ).toThrow(/Unknown ACE Qwen3 attention storage/);
  });

  it("accepts disjoint exact ranges for a sharded tied output", () => {
    const buffer = { size: 256 } as GPUBuffer;
    expect(() => validateAceQwen3TiedOutputBindings(
      "tied-safe",
      "reference-bf16",
      tinyQwenConfig(),
      binding(buffer, 0, 16),
      [
        {
          firstRow: 0,
          rowCount: 2,
          weight: binding(buffer, 32, 16),
          logits: binding(buffer, 64, 8),
        },
        {
          firstRow: 2,
          rowCount: 2,
          weight: binding(buffer, 48, 16),
          logits: binding(buffer, 72, 8),
        },
      ],
    )).not.toThrow();
  });

  it.each([
    {
      name: "hidden row",
      shards: (buffer: GPUBuffer): AceQwen3TiedWeightShard[] => [
        {
          firstRow: 0,
          rowCount: 2,
          weight: binding(buffer, 32, 16),
          logits: binding(buffer, 8, 8),
        },
        {
          firstRow: 2,
          rowCount: 2,
          weight: binding(buffer, 48, 16),
          logits: binding(buffer, 72, 8),
        },
      ],
      message: /logits shard 0 overlaps normalized hidden row/,
    },
    {
      name: "weight shard",
      shards: (buffer: GPUBuffer): AceQwen3TiedWeightShard[] => [
        {
          firstRow: 0,
          rowCount: 2,
          weight: binding(buffer, 32, 16),
          logits: binding(buffer, 48, 8),
        },
        {
          firstRow: 2,
          rowCount: 2,
          weight: binding(buffer, 48, 16),
          logits: binding(buffer, 72, 8),
        },
      ],
      message: /logits shard 0 overlaps weight shard 1/,
    },
    {
      name: "other logits shard",
      shards: (buffer: GPUBuffer): AceQwen3TiedWeightShard[] => [
        {
          firstRow: 0,
          rowCount: 2,
          weight: binding(buffer, 32, 16),
          logits: binding(buffer, 64, 8),
        },
        {
          firstRow: 2,
          rowCount: 2,
          weight: binding(buffer, 48, 16),
          logits: binding(buffer, 68, 8),
        },
      ],
      message: /logits shard 0 overlaps logits shard 1/,
    },
  ])("rejects a tied-output race with the $name", ({ shards, message }) => {
    const buffer = { size: 256 } as GPUBuffer;
    expect(() => validateAceQwen3TiedOutputBindings(
      "tied-race",
      "reference-bf16",
      tinyQwenConfig(),
      binding(buffer, 0, 16),
      shards(buffer),
    )).toThrow(message);
  });
});

function tinyQwenConfig(): AceQwen3Config {
  return {
    id: "tiny-qwen3-tied-output-contract",
    hiddenSize: 4,
    intermediateSize: 8,
    layerCount: 1,
    queryHeads: 2,
    keyValueHeads: 1,
    headDimension: 2,
    vocabularySize: 4,
    maximumPositionEmbeddings: 16,
    ropeTheta: 10_000,
    rmsNormEpsilon: 1e-6,
    attentionBias: false,
    hiddenActivation: "silu",
    tieWordEmbeddings: true,
  };
}

function binding(
  buffer: GPUBuffer,
  offset: number,
  size: number,
): GPUBufferBinding {
  return { buffer, offset, size };
}

const ROPE_VECTOR_DIMENSIONS = Object.freeze([
  0, 1, 2, 3, 7, 15, 31, 37, 47, 63, 64, 65, 101, 127,
]);

/**
 * Transformers 4.57.6 / PyTorch 2.10.0 CPU FP32 output words. Generated from
 * `Qwen3RotaryEmbedding` at the source/hash recorded in
 * `ACE_QWEN3_ROPE_REFERENCE_PROVENANCE`; the concatenated full six-position
 * cosine/sine payload has SHA-256
 * `285f5527ddfea2b525f1d58f9847701aa3c416831e6dc7bf0b6246db803c8fe4`.
 */
const TRANSFORMERS_ROPE_VECTORS = Object.freeze([
  {
    position: 0,
    cosine: [
      0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000,
      0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000,
      0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000,
      0x3f800000, 0x3f800000,
    ],
    sine: [
      0x00000000, 0x00000000, 0x00000000, 0x00000000,
      0x00000000, 0x00000000, 0x00000000, 0x00000000,
      0x00000000, 0x00000000, 0x00000000, 0x00000000,
      0x00000000, 0x00000000,
    ],
  },
  {
    position: 1,
    cosine: [
      0x3f0a5141, 0x3f3147ef, 0x3f4be4aa, 0x3f5dbda8,
      0x3f79cac5, 0x3f7fcd8c, 0x3f7ffff3, 0x3f7fffff,
      0x3f800000, 0x3f800000, 0x3f0a5141, 0x3f3147ef,
      0x3f7fffff, 0x3f800000,
    ],
    sine: [
      0x3f576aa4, 0x3f38ae99, 0x3f1acd3a, 0x3effddfd,
      0x3e6023d9, 0x3d20b18e, 0x3aa2a6f4, 0x39b229fb,
      0x382497ab, 0x35a68e4c, 0x3f576aa4, 0x3f38ae99,
      0x39b229fb, 0x35a68e4c,
    ],
  },
  {
    position: 17,
    cosine: [
      0xbe8ce236, 0x3ed91587, 0x3d33cfe6, 0xbf5d0c34,
      0xbf51d9e5, 0x3f491dd8, 0x3f7ff16b, 0x3f7ffee8,
      0x3f7ffffc, 0x3f800000, 0xbe8ce236, 0x3ed91587,
      0x3f7ffee8, 0x3f800000,
    ],
    sine: [
      0xbf761e26, 0x3f67d9b6, 0xbf7fc0d2, 0x3f0120a8,
      0xbf129fa7, 0x3f1e646e, 0x3cacce1e, 0x3bbd4c56,
      0x3a2ee125, 0x37b0f731, 0xbf761e26, 0x3f67d9b6,
      0x3bbd4c56, 0x37b0f731,
    ],
  },
  {
    position: 1_023,
    cosine: [
      0x3eccd5bd, 0x3e9322c2, 0xbe042378, 0x3e9a7eb3,
      0x3f66fb55, 0xbf446b02, 0x3e97f34b, 0x3f70afab,
      0x3f7fcb33, 0x3f7ffff3, 0x3eccd5bd, 0x3e9322c2,
      0x3f70afab, 0x3f7ffff3,
    ],
    sine: [
      0xbf6a9ec9, 0x3f753394, 0xbf7ddc11, 0x3f741138,
      0xbedcc1d3, 0x3f242ec8, 0x3f747760, 0x3eae6d33,
      0x3d246337, 0x3aa664a5, 0xbf6a9ec9, 0x3f753394,
      0x3eae6d33, 0x3aa664a5,
    ],
  },
  {
    position: 4_095,
    cosine: [
      0xbd871e6c, 0x3ea0b21a, 0x3e0f4e40, 0x3f710132,
      0x3ede89c3, 0xbf63bc43, 0x3eb8ca35, 0x3e368cf0,
      0x3f7cb3a6, 0x3f7fff28, 0xbd871e6c, 0x3ea0b21a,
      0x3e368cf0, 0x3f7fff28,
    ],
    sine: [
      0xbf7f7136, 0x3f731072, 0x3f7d7b0f, 0x3eaca89c,
      0xbf668dda, 0xbee9dc16, 0xbf6ebeb9, 0x3f7be632,
      0x3e23d851, 0x3ba683b4, 0xbf7f7136, 0x3f731072,
      0x3f7be632, 0x3ba683b4,
    ],
  },
  {
    position: 40_959,
    cosine: [
      0x3eef0d27, 0x3f1b2314, 0x3e88f18a, 0xbe94b22f,
      0xbf7aecad, 0x3ebff663, 0x3f589cda, 0x3e5de9e6,
      0xbd1585ca, 0x3f7fab5d, 0x3eef0d27, 0x3f1b2314,
      0x3e5de9e6, 0x3f7fab5d,
    ],
    sine: [
      0xbf62623f, 0x3f4ba35f, 0x3f76ac50, 0x3f74f750,
      0xbe4ae4fd, 0xbf6d5371, 0x3f086fc2, 0x3f79ea91,
      0x3f7fd452, 0x3d5019a0, 0xbf62623f, 0x3f4ba35f,
      0x3f79ea91, 0x3d5019a0,
    ],
  },
] as const);

function floatWords(values: Float32Array): number[] {
  return [...new Uint32Array(
    values.buffer,
    values.byteOffset,
    values.byteLength / Float32Array.BYTES_PER_ELEMENT,
  )];
}

function expectWordsWithinUlp(
  actual: readonly number[],
  expected: readonly number[],
  maximumUlp: number,
): void {
  expect(actual).toHaveLength(expected.length);
  for (let index = 0; index < expected.length; index += 1) {
    expect(
      Math.abs(actual[index]! - expected[index]!),
      `FP32 word ${index}: 0x${actual[index]!.toString(16)} vs ` +
        `0x${expected[index]!.toString(16)}`,
    ).toBeLessThanOrEqual(maximumUlp);
  }
}

function sha256LittleEndianWords(words: readonly number[]): string {
  const bytes = new Uint8Array(words.length * Uint32Array.BYTES_PER_ELEMENT);
  const view = new DataView(bytes.buffer);
  for (let index = 0; index < words.length; index += 1) {
    view.setUint32(index * Uint32Array.BYTES_PER_ELEMENT, words[index]!, true);
  }
  return createHash("sha256").update(bytes).digest("hex");
}
