import { describe, expect, it } from "vitest";

import {
  ACE_DIAGNOSTIC_HOST_SOFTMAX,
  AceOpt0084PlannerSamplingWorkspace,
  AcePlannerSamplingCursor,
  createAcePlannerBrowserSamplingWeights,
  createAcePlannerCompactFilteredLogits,
  createAcePlannerFilteredLogits,
  sampleAcePlannerCompactToken,
  sampleAcePlannerCompactTokenOpt0084,
  sampleAcePlannerToken,
  sampleAcePlannerTokenOpt0084,
  type AcePlannerCompactTokenSampleInput,
  type AcePlannerTokenSampleInput,
} from "../src/runtime/planner-sampling.js";
import { canonicalizeSeed } from "../src/runtime/seed.js";

describe("OPT-0084 candidate-domain radix sampler", () => {
  it("orders ties, signed zero, and subnormals by descending value then global ID", () => {
    const smallestSubnormal = floatFromWord(1);
    const workspace = new AceOpt0084PlannerSamplingWorkspace();
    const input: AcePlannerTokenSampleInput = {
      conditionalLogits: [
        Number.NEGATIVE_INFINITY,
        -0,
        0,
        smallestSubnormal,
        -smallestSubnormal,
        2,
        2,
        -3,
      ],
      seenTokenIds: [],
      allowedTokens: { kind: "all" },
      parameters: {
        temperature: 1,
        guidanceScale: 1,
        topK: 0,
        topP: 0.999_999,
        repetitionPenalty: 1,
      },
      word: 0x8000_0000,
    };

    expect(sampleAcePlannerTokenOpt0084(input, workspace)).toEqual(
      sampleAcePlannerToken(input),
    );
    expect([...workspace.copyLastCandidateTokenIds()]).toEqual([
      1, 2, 3, 4, 5, 6, 7,
    ]);
    expect([...workspace.copyLastRadixOrderTokenIds()]).toEqual([
      5, 6, 3, 1, 2, 4, 7,
    ]);
    expectTraceMatchesFullArmA(workspace, input);
  });

  it("normalizes unordered sparse IDs and matches every retained FP32 word", () => {
    const workspace = new AceOpt0084PlannerSamplingWorkspace();
    const conditional = Float32Array.of(
      50,
      -4,
      2,
      -0,
      9,
      -3,
      0,
      2,
      Number.NEGATIVE_INFINITY,
      2,
      -8,
      2,
    );
    const input: AcePlannerTokenSampleInput = {
      conditionalLogits: conditional,
      seenTokenIds: [9, 9, 3, 0],
      allowedTokens: { kind: "ids", tokenIds: [11, 3, 9, 2, 8, 6, 7] },
      parameters: {
        temperature: 0.85,
        guidanceScale: 1,
        topK: 3,
        topP: 0.6,
        repetitionPenalty: 1.25,
      },
      word: 0xffff_ffff,
    };

    expect(sampleAcePlannerTokenOpt0084(input, workspace)).toEqual(
      sampleAcePlannerToken(input),
    );
    expect([...workspace.copyLastCandidateTokenIds()]).toEqual([
      2, 3, 6, 7, 9, 11,
    ]);
    expectTraceMatchesFullArmA(workspace, input);
  });

  it("matches full-vocabulary CFG with distinct pre-CFG and final constraints", () => {
    const workspace = new AceOpt0084PlannerSamplingWorkspace();
    const vocabularySize = 37;
    const conditional = new Float32Array(vocabularySize);
    const unconditional = new Float32Array(vocabularySize);
    for (let tokenId = 0; tokenId < vocabularySize; tokenId += 1) {
      conditional[tokenId] = Math.fround((tokenId % 11) * 0.5 - 3);
      unconditional[tokenId] = Math.fround((tokenId % 7) * 0.25 - 2);
    }
    conditional[13] = Number.NEGATIVE_INFINITY;
    unconditional[13] = Number.NEGATIVE_INFINITY;
    const input: AcePlannerTokenSampleInput = {
      conditionalLogits: conditional,
      unconditionalLogits: unconditional,
      seenTokenIds: [0, 3, 18, 18, 36],
      preCfgAllowedTokens: {
        kind: "range",
        firstTokenId: 10,
        tokenCount: 12,
        additionalTokenIds: [3],
      },
      allowedTokens: { kind: "ids", tokenIds: [21, 3, 13, 10, 18, 15] },
      parameters: {
        temperature: 0.7,
        guidanceScale: 2.75,
        topK: 4,
        topP: 0.72,
        repetitionPenalty: 1.3,
      },
      word: 0x1234_5678,
    };

    expect(sampleAcePlannerTokenOpt0084(input, workspace)).toEqual(
      sampleAcePlannerToken(input),
    );
    expect([...workspace.copyLastCandidateTokenIds()]).toEqual([
      3, 10, 15, 18, 21,
    ]);
    expectTraceMatchesFullArmA(workspace, input);
  });

  it("omits negative-infinity holes created by valid CFG and repetition arithmetic", () => {
    const workspace = new AceOpt0084PlannerSamplingWorkspace();
    const cfgInput: AcePlannerTokenSampleInput = {
      conditionalLogits: Float32Array.of(-3.3e38, 3, 1),
      unconditionalLogits: Float32Array.of(-3e38, 0, 0.5),
      seenTokenIds: [],
      allowedTokens: { kind: "all" },
      parameters: {
        temperature: 0.85,
        guidanceScale: 2,
        topK: 0,
        topP: 0.8,
        repetitionPenalty: 1,
      },
      word: 0x4000_0000,
    };
    expect(sampleAcePlannerTokenOpt0084(cfgInput, workspace)).toEqual(
      sampleAcePlannerToken(cfgInput),
    );
    expect([...workspace.copyLastCandidateTokenIds()]).toEqual([1, 2]);
    expectTraceMatchesFullArmA(workspace, cfgInput);

    const firstTokenId = 29;
    const repetitionInput: AcePlannerCompactTokenSampleInput = {
      firstTokenId,
      vocabularySize: 100,
      conditionalLogits: Float32Array.of(-2e38, 2, 1, -1),
      seenTokenIds: [firstTokenId, firstTokenId, 99],
      parameters: {
        temperature: 1,
        guidanceScale: 1,
        topK: 2,
        topP: 0.9,
        repetitionPenalty: 2,
      },
      word: 0xbfff_ffff,
    };
    expect(
      sampleAcePlannerCompactTokenOpt0084(repetitionInput, workspace),
    ).toEqual(sampleAcePlannerCompactToken(repetitionInput));
    expect([...workspace.copyLastCandidateTokenIds()]).toEqual([30, 31, 32]);
    expectTraceMatchesCompactArmA(workspace, repetitionInput);
  });

  it("differential-fuzzes full and sparse states across top-p boundaries", () => {
    const workspace = new AceOpt0084PlannerSamplingWorkspace();
    const random = createXorShift32(0x0084_cafe);
    const topPs = [1, 0.999_999_94, 0.9, 0.500_000_06, 0.125] as const;
    const temperatures = [0.5, 0.85, 1, 1.3] as const;
    const penalties = [1, 1.1, 1.5] as const;
    const words = [0, 1, 0x7fff_ffff, 0x8000_0000, 0xffff_ffff] as const;

    for (let round = 0; round < 64; round += 1) {
      const vocabularySize = 33 + (random() % 64);
      const conditional = new Float32Array(vocabularySize);
      for (let tokenId = 0; tokenId < vocabularySize; tokenId += 1) {
        conditional[tokenId] = adversarialLogit(random());
      }
      conditional[random() % vocabularySize] = 7;

      const selected = new Set<number>();
      const targetCount = 2 + (random() % (vocabularySize - 1));
      while (selected.size < targetCount) selected.add(random() % vocabularySize);
      const tokenIds = shuffle([...selected], random);
      if (!tokenIds.some((tokenId) => Number.isFinite(conditional[tokenId]))) {
        conditional[tokenIds[0]!] = 1;
      }
      const topKChoices = [0, 1, 3, vocabularySize] as const;
      const inputBase: Omit<AcePlannerTokenSampleInput, "word"> = {
        conditionalLogits: conditional,
        seenTokenIds: [
          random() % vocabularySize,
          random() % vocabularySize,
          random() % vocabularySize,
        ],
        allowedTokens: round % 4 === 0
          ? { kind: "all" }
          : { kind: "ids", tokenIds },
        parameters: {
          temperature: temperatures[random() % temperatures.length]!,
          guidanceScale: 1,
          topK: topKChoices[random() % topKChoices.length]!,
          topP: topPs[random() % topPs.length]!,
          repetitionPenalty: penalties[random() % penalties.length]!,
        },
      };
      for (const word of words) {
        const input = { ...inputBase, word };
        expect(
          sampleAcePlannerTokenOpt0084(input, workspace),
          `round ${round}, word ${word}`,
        ).toEqual(sampleAcePlannerToken(input));
      }
      expectTraceMatchesFullArmA(workspace, { ...inputBase, word: words[0] });
    }
  });

  it("differential-fuzzes compact CFG states with global IDs and holes", () => {
    const workspace = new AceOpt0084PlannerSamplingWorkspace();
    const random = createXorShift32(0x84f0_0d42);
    const words = [0, 0x4000_0000, 0xbfff_ffff, 0xffff_ffff] as const;

    for (let round = 0; round < 64; round += 1) {
      const candidateSourceCount = 17 + (random() % 53);
      const firstTokenId = 5 + (random() % 19);
      const vocabularySize = firstTokenId + candidateSourceCount + 13;
      const conditional = new Float32Array(candidateSourceCount);
      const unconditional = new Float32Array(candidateSourceCount);
      for (let local = 0; local < candidateSourceCount; local += 1) {
        if ((random() & 15) === 0) {
          conditional[local] = Number.NEGATIVE_INFINITY;
          unconditional[local] = Number.NEGATIVE_INFINITY;
        } else {
          conditional[local] = adversarialFiniteLogit(random());
          unconditional[local] = adversarialFiniteLogit(random());
        }
      }
      conditional[0] = 6;
      unconditional[0] = 1;
      const topKChoices = [0, 1, 7, vocabularySize] as const;
      const topPs = [1, 0.95, 0.5, 0.1] as const;
      const inputBase: Omit<AcePlannerCompactTokenSampleInput, "word"> = {
        firstTokenId,
        vocabularySize,
        conditionalLogits: conditional,
        unconditionalLogits: unconditional,
        seenTokenIds: [
          0,
          firstTokenId + (random() % candidateSourceCount),
          vocabularySize - 1,
        ],
        parameters: {
          temperature: [0.6, 0.85, 1, 1.4][random() % 4]!,
          guidanceScale: [1, 2, 3.25][random() % 3]!,
          topK: topKChoices[random() % topKChoices.length]!,
          topP: topPs[random() % topPs.length]!,
          repetitionPenalty: [1, 1.2, 1.5][random() % 3]!,
        },
      };
      for (const word of words) {
        const input = { ...inputBase, word };
        expect(
          sampleAcePlannerCompactTokenOpt0084(input, workspace),
          `round ${round}, word ${word}`,
        ).toEqual(sampleAcePlannerCompactToken(input));
      }
      expectTraceMatchesCompactArmA(workspace, {
        ...inputBase,
        word: words[0],
      });
    }
  });

  it("matches a realistic 64k compact semantic candidate row bit-for-bit", () => {
    const candidateCount = 64_000;
    const firstTokenId = 151_669;
    const vocabularySize = 217_204;
    const conditional = new Float32Array(candidateCount);
    const unconditional = new Float32Array(candidateCount);
    for (let local = 0; local < candidateCount; local += 1) {
      conditional[local] = Math.fround(
        ((local * 73 + 19) % 257) / 16 - 12,
      );
      unconditional[local] = Math.fround(
        ((local * 41 + 7) % 193) / 16 - 10,
      );
    }
    conditional[0] = 8;
    conditional[31_999] = 7.25;
    conditional[63_999] = 7.5;
    unconditional[0] = 1;
    unconditional[31_999] = 1.25;
    unconditional[63_999] = 1.5;
    const input: AcePlannerCompactTokenSampleInput = {
      firstTokenId,
      vocabularySize,
      conditionalLogits: conditional,
      unconditionalLogits: unconditional,
      seenTokenIds: [3, firstTokenId, firstTokenId + 31_999, vocabularySize - 1],
      parameters: {
        temperature: 0.85,
        guidanceScale: 2,
        topK: 0,
        topP: 0.9,
        repetitionPenalty: 1.1,
      },
      word: 0xcafe_babe,
    };
    const workspace = new AceOpt0084PlannerSamplingWorkspace();

    expect(sampleAcePlannerCompactTokenOpt0084(input, workspace)).toEqual(
      sampleAcePlannerCompactToken(input),
    );
    expect(workspace.copyLastCandidateTokenIds()).toHaveLength(candidateCount);
    expectTraceMatchesCompactArmA(workspace, input);
  }, 15_000);

  it("keeps compact storage candidate-sized for safe global IDs above uint32", () => {
    const firstTokenId = 0x1_0000_0000 + 123;
    const vocabularySize = Number.MAX_SAFE_INTEGER;
    const input: AcePlannerCompactTokenSampleInput = {
      firstTokenId,
      vocabularySize,
      conditionalLogits: Float32Array.of(4, -2, 3, 1),
      seenTokenIds: [0, firstTokenId + 1, firstTokenId + 1, vocabularySize - 1],
      parameters: {
        temperature: 0.75,
        guidanceScale: 1,
        topK: vocabularySize,
        topP: 0.7,
        repetitionPenalty: 1.25,
      },
      word: 0x7fff_ffff,
    };
    const workspace = new AceOpt0084PlannerSamplingWorkspace();

    const armB = sampleAcePlannerCompactTokenOpt0084(input, workspace);
    expect(armB).toEqual(sampleAcePlannerCompactToken(input));
    expect(armB.tokenId).toBeGreaterThan(0xffff_ffff);
    expect([...workspace.copyLastCandidateTokenIds()]).toEqual([
      firstTokenId,
      firstTokenId + 1,
      firstTokenId + 2,
      firstTokenId + 3,
    ]);
    expect(workspace.stats.candidateCapacity).toBe(4);
    expect(workspace.stats.maskCapacity).toBe(4);
    expectTraceMatchesCompactArmA(workspace, input);
  });

  it("validates and consumes forced singleton and terminal draws exactly", () => {
    const seed = canonicalizeSeed("0084000000000001");
    const armA = new AcePlannerSamplingCursor(seed, 7n);
    const armB = new AcePlannerSamplingCursor(seed, 7n);
    const forced = {
      conditionalLogits: [100, -5, 8, 1, -3],
      seenTokenIds: [3, 3],
      allowedTokens: { kind: "ids" as const, tokenIds: [3] },
      parameters: {
        temperature: Number.MAX_VALUE,
        guidanceScale: 1,
        topK: 5,
        topP: Number.MIN_VALUE,
        repetitionPenalty: 1.5,
      },
    };
    const forcedB = armB.sampleOpt0084(forced);
    expect(forcedB).toEqual(armA.sample(forced));
    expect(forcedB).toMatchObject({ tokenId: 3, positiveCandidateCount: 1 });
    expect(forcedB.drawIndex).toBe(7n);

    const terminal = {
      firstTokenId: 0x1_0000_0000 + 9,
      vocabularySize: Number.MAX_SAFE_INTEGER,
      conditionalLogits: [2],
      seenTokenIds: [0],
      parameters: {
        temperature: 1,
        guidanceScale: 1,
        topK: Number.MAX_SAFE_INTEGER,
        topP: 1,
        repetitionPenalty: 1,
      },
    };
    const terminalB = armB.sampleCompactOpt0084(terminal);
    expect(terminalB).toEqual(armA.sampleCompact(terminal));
    expect(terminalB).toMatchObject({
      tokenId: terminal.firstTokenId,
      positiveCandidateCount: 1,
      drawIndex: 8n,
    });
    expect(armB.consumed).toBe(9n);

    expect(() => armB.sampleOpt0084({
      ...forced,
      conditionalLogits: [Number.NaN, -5, 8, 1, -3],
    })).toThrow(/finite or negative infinity/);
    expect(armB.consumed).toBe(9n);
  });

  it("reuses retained storage and keeps the Philox cursor uncommitted on errors", () => {
    const workspace = new AceOpt0084PlannerSamplingWorkspace();
    const warmInput: AcePlannerTokenSampleInput = {
      conditionalLogits: Float32Array.from(
        { length: 257 },
        (_, tokenId) => Math.fround((tokenId % 23) - 11),
      ),
      seenTokenIds: [3, 19, 100],
      allowedTokens: { kind: "all" },
      parameters: {
        temperature: 0.85,
        guidanceScale: 1,
        topK: 32,
        topP: 0.9,
        repetitionPenalty: 1.1,
      },
      word: 17,
    };
    workspace.sample(warmInput);
    const warmStats = workspace.stats;
    for (let round = 0; round < 20; round += 1) {
      workspace.sample({
        ...warmInput,
        conditionalLogits: Float32Array.from(
          warmInput.conditionalLogits,
        ).slice(0, 31),
        seenTokenIds: [3, 19],
        parameters: { ...warmInput.parameters, topK: 3 },
        word: round,
      });
    }
    expect(workspace.stats).toEqual(warmStats);

    const seed = canonicalizeSeed("0084face0084cafe");
    const armA = new AcePlannerSamplingCursor(seed, 41n);
    const armB = new AcePlannerSamplingCursor(seed, 41n);
    const parameters = {
      temperature: 1,
      guidanceScale: 1,
      topK: 0,
      topP: 1,
      repetitionPenalty: 1,
    } as const;
    expect(() => armB.sampleOpt0084({
      conditionalLogits: [1, Number.NaN, 2],
      seenTokenIds: [],
      allowedTokens: { kind: "ids", tokenIds: [0, 2] },
      parameters,
    })).toThrow(/finite or negative infinity/);
    expect(() => armB.sampleOpt0084({
      conditionalLogits: [1, 2, 3],
      seenTokenIds: [],
      allowedTokens: {
        kind: "range",
        firstTokenId: 0,
        tokenCount: 2,
        additionalTokenIds: [1],
      },
      parameters,
    })).toThrow(/repeats token ID/);
    expect(() => armB.sampleOpt0084({
      conditionalLogits: [1, 2, 3],
      seenTokenIds: [],
      allowedTokens: { kind: "all" },
      parameters,
      softmax: ACE_DIAGNOSTIC_HOST_SOFTMAX,
    })).toThrow(/diagnostic-only/);
    expect(armB.consumed).toBe(41n);

    const valid = {
      conditionalLogits: [20, -5, 3, 3],
      seenTokenIds: [2],
      allowedTokens: { kind: "ids" as const, tokenIds: [3, 2, 1] },
      parameters: { ...parameters, topP: 0.75 },
    };
    expect(armB.sampleOpt0084(valid)).toEqual(armA.sample(valid));
    expect(armB.consumed).toBe(42n);
    const compactValid = {
      firstTokenId: 2,
      vocabularySize: 7,
      conditionalLogits: [3, -1, 2],
      seenTokenIds: [0, 3, 6],
      parameters: { ...parameters, topP: 0.8 },
    };
    expect(armB.sampleCompactOpt0084(compactValid)).toEqual(
      armA.sampleCompact(compactValid),
    );
    expect(armB.consumed).toBe(43n);
  });

  it("fails sparse duplicates, range overlaps, invalid words, and invalid compact IDs", () => {
    const workspace = new AceOpt0084PlannerSamplingWorkspace();
    const base: Omit<AcePlannerTokenSampleInput, "allowedTokens" | "word"> = {
      conditionalLogits: [1, 2, 3, 4],
      seenTokenIds: [],
      parameters: {
        temperature: 1,
        guidanceScale: 1,
        topK: 0,
        topP: 1,
        repetitionPenalty: 1,
      },
    };
    expect(() => workspace.sample({
      ...base,
      allowedTokens: { kind: "ids", tokenIds: [3, 1, 3] },
      word: 0,
    })).toThrow(/repeats token ID 3/);
    expect(() => workspace.sample({
      ...base,
      allowedTokens: {
        kind: "range",
        firstTokenId: 1,
        tokenCount: 2,
        additionalTokenIds: [2],
      },
      word: 0,
    })).toThrow(/repeats token ID 2/);
    expect(() => workspace.sample({
      ...base,
      allowedTokens: { kind: "all" },
      word: 0x1_0000_0000,
    })).toThrow(/unsigned 32-bit/);
    expect(() => workspace.sampleCompact({
      firstTokenId: 2,
      vocabularySize: 5,
      conditionalLogits: [1, 2],
      seenTokenIds: [5],
      parameters: base.parameters,
      word: 0,
    })).toThrow(/outside the vocabulary/);
  });

  it("recovers after a late failure that grows the retained workspace", () => {
    const workspace = new AceOpt0084PlannerSamplingWorkspace();
    const parameters = {
      temperature: 1,
      guidanceScale: 1,
      topK: 0,
      topP: 1,
      repetitionPenalty: 1,
    } as const;
    const small: AcePlannerTokenSampleInput = {
      conditionalLogits: [1, 2, 3],
      seenTokenIds: [],
      allowedTokens: { kind: "all" },
      parameters,
      word: 0,
    };
    workspace.sample(small);
    const beforeGrowth = workspace.stats;
    const large = new Float32Array(1_025);
    for (let index = 0; index < large.length; index += 1) {
      large[index] = Math.fround((index % 17) - 8);
    }
    expect(() => workspace.sample({
      ...small,
      conditionalLogits: large,
      word: 0x1_0000_0000,
    })).toThrow(/unsigned 32-bit/);
    const afterGrowth = workspace.stats;
    expect(afterGrowth.candidateCapacity).toBeGreaterThan(
      beforeGrowth.candidateCapacity,
    );
    expect(afterGrowth.maskCapacity).toBeGreaterThan(beforeGrowth.maskCapacity);
    expect(() => workspace.copyLastWeights()).toThrow(/no successful sample/);

    expect(workspace.sample({ ...small, word: 0xffff_ffff })).toEqual(
      sampleAcePlannerToken({ ...small, word: 0xffff_ffff }),
    );
    expect(workspace.stats).toEqual(afterGrowth);
  });
});

function expectTraceMatchesFullArmA(
  workspace: AceOpt0084PlannerSamplingWorkspace,
  input: AcePlannerTokenSampleInput,
): void {
  const filtered = createAcePlannerFilteredLogits(input);
  const weights = createAcePlannerBrowserSamplingWeights(
    filtered,
    input.parameters.temperature,
  );
  const tokenIds = workspace.copyLastCandidateTokenIds();
  expect(floatWords(workspace.copyLastFilteredLogits())).toEqual(
    floatWords(gather(filtered, tokenIds)),
  );
  expect(floatWords(workspace.copyLastWeights())).toEqual(
    floatWords(gather(weights, tokenIds)),
  );
}

function expectTraceMatchesCompactArmA(
  workspace: AceOpt0084PlannerSamplingWorkspace,
  input: AcePlannerCompactTokenSampleInput,
): void {
  const filtered = createAcePlannerCompactFilteredLogits(input);
  const weights = createAcePlannerBrowserSamplingWeights(
    filtered,
    input.parameters.temperature,
  );
  const tokenIds = workspace.copyLastCandidateTokenIds();
  const localIds = Uint32Array.from(
    tokenIds,
    (tokenId) => tokenId - input.firstTokenId,
  );
  expect(floatWords(workspace.copyLastFilteredLogits())).toEqual(
    floatWords(gather(filtered, localIds)),
  );
  expect(floatWords(workspace.copyLastWeights())).toEqual(
    floatWords(gather(weights, localIds)),
  );
}

function gather(
  values: Float32Array,
  indices: ArrayLike<number>,
): Float32Array {
  return Float32Array.from(indices, (index) => values[index]!);
}

function floatWords(values: ArrayLike<number>): number[] {
  const storage = new ArrayBuffer(values.length * 4);
  const floats = new Float32Array(storage);
  for (let index = 0; index < values.length; index += 1) {
    floats[index] = values[index]!;
  }
  return [...new Uint32Array(storage)];
}

function floatFromWord(word: number): number {
  const storage = new ArrayBuffer(4);
  new Uint32Array(storage)[0] = word >>> 0;
  return new Float32Array(storage)[0]!;
}

function adversarialLogit(word: number): number {
  switch (word & 15) {
    case 0:
      return Number.NEGATIVE_INFINITY;
    case 1:
      return -0;
    case 2:
      return 0;
    case 3:
      return floatFromWord(1);
    case 4:
      return -floatFromWord(1);
    default:
      return Math.fround(((word >>> 4) % 33) * 0.25 - 4);
  }
}

function adversarialFiniteLogit(word: number): number {
  const value = adversarialLogit(word);
  return value === Number.NEGATIVE_INFINITY ? -12 : value;
}

function createXorShift32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    return state >>> 0;
  };
}

function shuffle(values: number[], random: () => number): number[] {
  for (let index = values.length - 1; index > 0; index -= 1) {
    const other = random() % (index + 1);
    const value = values[index]!;
    values[index] = values[other]!;
    values[other] = value;
  }
  return values;
}
