import { createHash } from "node:crypto";

import { describe, expect, it } from "vitest";

import {
  ACE_BROWSER_SOFTMAX_V1,
  ACE_DIAGNOSTIC_HOST_SOFTMAX,
  ACE_PLANNER_SAMPLING_CONTRACT,
  ACE_PLANNER_SOFTMAX_ACCEPTANCE,
  AcePlannerSamplingCursor,
  acePlannerBrowserExpF32,
  applyAcePlannerRepetitionPenalty,
  applyAcePlannerTopK,
  combineAcePlannerCfgLogits,
  createAcePlannerBrowserSamplingWeights,
  createAcePlannerBrowserTopPKeep,
  createAcePlannerCompactFilteredLogits,
  createAcePlannerFilteredLogits,
  maskAcePlannerLogits,
  sampleAcePlannerCompactToken,
  sampleAcePlannerToken,
} from "../src/runtime/planner-sampling.js";
import { canonicalizeSeed } from "../src/runtime/seed.js";
import vectorsJson from "./planner-sampling-vectors.json";

interface SamplingVector {
  readonly id: string;
  readonly conditional: readonly number[];
  readonly unconditional?: readonly number[];
  readonly allowedTokenIds: readonly number[];
  readonly seenTokenIds: readonly number[];
  readonly guidanceScale: number;
  readonly repetitionPenalty: number;
  readonly topK: number;
  readonly topP: number;
  readonly temperature: number;
  readonly word: number;
  readonly expected: {
    readonly combinedWords: readonly number[];
    readonly penalizedWords: readonly number[];
    readonly postTopKFinite: readonly number[];
    readonly postTopPFinite: readonly number[];
    readonly weightWords: readonly number[];
    readonly sampledTokenId: number;
  };
}

describe("ACE planner deterministic filtering", () => {
  it("pins the source order and accepts only the versioned browser distribution", () => {
    expect(ACE_PLANNER_SAMPLING_CONTRACT.order).toEqual([
      "code-allowed-subspace",
      "cfg",
      "constraint-mask",
      "repetition-penalty",
      "top-k",
      "top-p",
      "temperature",
      "fp32-softmax",
      "philox-categorical",
    ]);
    expect(ACE_PLANNER_SOFTMAX_ACCEPTANCE).toMatchObject({
      status: "accepted-browser-v1",
      productionOracleId: "ace-browser-softmax-v1",
      distributionAuthority: "browser-defined",
      torchDifferential: "pending-teacher-forced-checkpoint",
      torchDifferentialRequiredBeforeListeningCandidate: true,
      torchBitExactClaim: false,
    });
    expect(ACE_BROWSER_SOFTMAX_V1.status).toBe("accepted");
    expect(vectorsJson.browserSoftmaxValidation.errorBounds).toEqual(
      ACE_PLANNER_SOFTMAX_ACCEPTANCE.errorBounds,
    );
  });

  it("matches the independent Python vectors, including stable top-p ties", () => {
    for (const raw of vectorsJson.vectors) {
      const vector = raw as SamplingVector;
      let combined: Float32Array;
      const conditional = maskAcePlannerLogits(vector.conditional, {
        kind: "ids",
        tokenIds: vector.allowedTokenIds,
      });
      if (vector.unconditional === undefined) {
        combined = conditional;
      } else {
        // This public full-vocabulary helper is checked on the admitted slice;
        // the production pipeline computes only that slice before masking.
        const admittedConditional = vector.allowedTokenIds.map(
          (token) => vector.conditional[token]!,
        );
        const admittedUnconditional = vector.allowedTokenIds.map(
          (token) => vector.unconditional![token]!,
        );
        const admitted = combineAcePlannerCfgLogits(
          admittedConditional,
          admittedUnconditional,
          vector.guidanceScale,
        );
        combined = new Float32Array(vector.conditional.length);
        combined.fill(Number.NEGATIVE_INFINITY);
        vector.allowedTokenIds.forEach((token, index) => {
          combined[token] = admitted[index]!;
        });
      }
      expect(floatWords(combined), `${vector.id} combined`).toEqual(
        vector.expected.combinedWords,
      );
      const penalized = applyAcePlannerRepetitionPenalty(
        combined,
        vector.seenTokenIds,
        vector.repetitionPenalty,
      );
      expect(floatWords(penalized), `${vector.id} penalty`).toEqual(
        vector.expected.penalizedWords,
      );
      const keptK = applyAcePlannerTopK(penalized, vector.topK);
      expect(finiteTokenIds(keptK), `${vector.id} top-k`).toEqual(
        vector.expected.postTopKFinite,
      );
      const keepP = createAcePlannerBrowserTopPKeep(keptK, vector.topP);
      const keptP = keptK.slice();
      for (let tokenId = 0; tokenId < keptP.length; tokenId += 1) {
        if (keepP[tokenId] === 0) keptP[tokenId] = Number.NEGATIVE_INFINITY;
      }
      expect(finiteTokenIds(keptP), `${vector.id} top-p`).toEqual(
        vector.expected.postTopPFinite,
      );
      const weights = createAcePlannerBrowserSamplingWeights(
        keptP,
        vector.temperature,
      );
      expect(floatWords(weights), `${vector.id} browser-v1 weights`).toEqual(
        vector.expected.weightWords,
      );
      const sampled = sampleAcePlannerToken({
        conditionalLogits: vector.conditional,
        ...(vector.unconditional === undefined
          ? {}
          : { unconditionalLogits: vector.unconditional }),
        seenTokenIds: vector.seenTokenIds,
        allowedTokens: { kind: "ids", tokenIds: vector.allowedTokenIds },
        parameters: {
          temperature: vector.temperature,
          guidanceScale: vector.guidanceScale,
          topK: vector.topK,
          topP: vector.topP,
          repetitionPenalty: vector.repetitionPenalty,
        },
        word: vector.word,
      });
      expect(sampled.tokenId, `${vector.id} sample`).toBe(
        vector.expected.sampledTokenId,
      );
    }
  });

  it("matches independent software-exp hashes for a realistic sparse 64k code row", () => {
    const fixture = vectorsJson.browserSoftmaxValidation.realisticSparse;
    const logits = realisticSparseLogits();
    const keep = createAcePlannerBrowserTopPKeep(logits, fixture.topP);
    expect(sumBytes(keep)).toBe(fixture.browserKeepCount);
    expect(sha256Bytes(keep)).toBe(fixture.topPKeepSha256);

    for (let tokenId = 0; tokenId < logits.length; tokenId += 1) {
      if (keep[tokenId] === 0) logits[tokenId] = Number.NEGATIVE_INFINITY;
    }
    const weights = createAcePlannerBrowserSamplingWeights(
      logits,
      fixture.temperature,
    );
    expect(positiveCount(weights)).toBe(fixture.positiveWeightCount);
    expect(sha256FloatWords(weights)).toBe(fixture.weightWordsSha256);
  });

  it("has fixed normal and subnormal software-exp behavior without Math.exp", () => {
    expect(floatWords([
      acePlannerBrowserExpF32(0),
      acePlannerBrowserExpF32(-0.6931471824645996),
      acePlannerBrowserExpF32(-10),
      acePlannerBrowserExpF32(-80),
      acePlannerBrowserExpF32(-103.9),
      acePlannerBrowserExpF32(-104),
    ])).toEqual([
      1_065_353_216,
      1_056_964_608,
      943_614_926,
      96_464_058,
      1,
      0,
    ]);
  });

  it("masks the 64,000-code subspace before CFG over the 217,204-row vocabulary", () => {
    const vocabulary = 217_204;
    const firstCode = 151_669;
    const codeCount = 64_000;
    const conditional = new Float32Array(vocabulary);
    const unconditional = new Float32Array(vocabulary);
    conditional.fill(-40);
    unconditional.fill(-41);
    conditional[firstCode] = 4;
    conditional[firstCode + 32_000] = 3;
    conditional[firstCode + 63_999] = 2;
    unconditional[firstCode] = 1;
    unconditional[firstCode + 32_000] = 1.25;
    unconditional[firstCode + 63_999] = 1.5;
    const filtered = createAcePlannerFilteredLogits({
      conditionalLogits: conditional,
      unconditionalLogits: unconditional,
      seenTokenIds: [],
      allowedTokens: {
        kind: "range",
        firstTokenId: firstCode,
        tokenCount: codeCount,
      },
      parameters: {
        temperature: 0.85,
        guidanceScale: 2,
        topK: 32,
        topP: 0.9,
        repetitionPenalty: 1,
      },
      softmax: ACE_BROWSER_SOFTMAX_V1,
    });
    const finite = finiteTokenIds(filtered);
    expect(finite.length).toBeGreaterThan(0);
    expect(finite.every((token) => token >= firstCode && token < firstCode + codeCount)).toBe(true);
    expect(filtered[0]).toBe(Number.NEGATIVE_INFINITY);
  });

  it("selects code plus EOS before CFG, then applies the duration mask", () => {
    const conditional = [100, -10, 30, -20, 8, 7, -30];
    const unconditional = [-100, 20, 10, 40, 2, 3, 50];
    const base = {
      conditionalLogits: conditional,
      unconditionalLogits: unconditional,
      seenTokenIds: [],
      preCfgAllowedTokens: {
        kind: "range" as const,
        firstTokenId: 4,
        tokenCount: 2,
        additionalTokenIds: [2],
      },
      parameters: {
        temperature: 1,
        guidanceScale: 2,
        topK: 0,
        topP: 1,
        repetitionPenalty: 1,
      },
    };
    const codeStep = createAcePlannerFilteredLogits({
      ...base,
      allowedTokens: { kind: "range", firstTokenId: 4, tokenCount: 2 },
    });
    expect(finiteTokenIds(codeStep)).toEqual([4, 5]);
    expect([...codeStep.slice(4, 6)]).toEqual([14, 11]);
    expect(codeStep[2]).toBe(Number.NEGATIVE_INFINITY);

    const forcedEos = createAcePlannerFilteredLogits({
      ...base,
      allowedTokens: { kind: "ids", tokenIds: [2] },
    });
    expect(finiteTokenIds(forcedEos)).toEqual([2]);
    expect(forcedEos[2]).toBe(50);
  });

  it("uses threshold top-k semantics and penalizes each seen ID once", () => {
    expect(finiteTokenIds(applyAcePlannerTopK([3, 2, 2, 1], 2))).toEqual([
      0, 1, 2,
    ]);
    expect([...applyAcePlannerRepetitionPenalty([4, -4], [0, 0, 1], 2)]).toEqual([
      2, -8,
    ]);
  });

  it("fails closed on host exp and does not advance the Philox cursor", () => {
    const cursor = new AcePlannerSamplingCursor(canonicalizeSeed(7));
    expect(() => cursor.sample({
      conditionalLogits: [1, 0],
      seenTokenIds: [],
      allowedTokens: { kind: "all" },
      parameters: {
        temperature: 1,
        guidanceScale: 1,
        topK: 0,
        topP: 1,
        repetitionPenalty: 1,
      },
      softmax: ACE_DIAGNOSTIC_HOST_SOFTMAX,
    })).toThrow(/diagnostic-only/);
    expect(cursor.consumed).toBe(0n);
  });

  it("consumes one word even when a constraint forces exactly one token", () => {
    const cursor = new AcePlannerSamplingCursor(canonicalizeSeed(11));
    const first = cursor.sample({
      conditionalLogits: [10, -10],
      seenTokenIds: [],
      allowedTokens: { kind: "ids", tokenIds: [1] },
      parameters: {
        temperature: 1,
        guidanceScale: 1,
        topK: 0,
        topP: 1,
        repetitionPenalty: 1,
      },
    });
    expect(first.tokenId).toBe(1);
    expect(first.drawIndex).toBe(0n);
    const nextPhase = cursor.sample({
      conditionalLogits: [0, 0],
      seenTokenIds: [1],
      allowedTokens: { kind: "ids", tokenIds: [0] },
      parameters: {
        temperature: 1,
        guidanceScale: 1,
        topK: 0,
        topP: 1,
        repetitionPenalty: 1,
      },
    });
    expect(nextPhase.tokenId).toBe(0);
    expect(nextPhase.drawIndex).toBe(1n);
    expect(cursor.consumed).toBe(2n);
  });

  it("samples the real 64k semantic domain exactly without full-vector reconstruction", () => {
    const vocabularySize = 217_204;
    const firstTokenId = 151_669;
    const candidateCount = 64_000;
    const conditional = new Float32Array(candidateCount);
    const unconditional = new Float32Array(candidateCount);
    for (let index = 0; index < candidateCount; index += 1) {
      conditional[index] = Math.fround(((index * 73 + 19) % 257) / 16 - 12);
      unconditional[index] = Math.fround(((index * 41 + 7) % 193) / 16 - 10);
    }
    conditional[0] = 8;
    conditional[31_999] = 7.25;
    conditional[candidateCount - 1] = 7.5;
    unconditional[0] = 1;
    unconditional[31_999] = 1.25;
    unconditional[candidateCount - 1] = 1.5;

    // Values outside the admitted domain deliberately dominate. The ordinary
    // path must mask them, while the compact path never represents them.
    const fullConditional = new Float32Array(vocabularySize);
    const fullUnconditional = new Float32Array(vocabularySize);
    fullConditional.fill(100);
    fullUnconditional.fill(-100);
    fullConditional.set(conditional, firstTokenId);
    fullUnconditional.set(unconditional, firstTokenId);
    const parameters = {
      temperature: 0.85,
      guidanceScale: 2,
      topK: 0,
      topP: 0.9,
      repetitionPenalty: 1,
    } as const;
    const seenTokenIds = [3, firstTokenId, firstTokenId + 31_999,
      vocabularySize - 1];
    const allowedTokens = {
      kind: "range" as const,
      firstTokenId,
      tokenCount: candidateCount,
    };
    const fullFiltered = createAcePlannerFilteredLogits({
      conditionalLogits: fullConditional,
      unconditionalLogits: fullUnconditional,
      seenTokenIds,
      preCfgAllowedTokens: allowedTokens,
      allowedTokens,
      parameters,
    });
    const compactFiltered = createAcePlannerCompactFilteredLogits({
      firstTokenId,
      vocabularySize,
      conditionalLogits: conditional,
      unconditionalLogits: unconditional,
      seenTokenIds,
      parameters,
    });
    expect(floatWords(compactFiltered)).toEqual(
      floatWords(fullFiltered.slice(firstTokenId, firstTokenId + candidateCount)),
    );

    const word = 0xcafe_babe;
    const full = sampleAcePlannerToken({
      conditionalLogits: fullConditional,
      unconditionalLogits: fullUnconditional,
      seenTokenIds,
      preCfgAllowedTokens: allowedTokens,
      allowedTokens,
      parameters,
      word,
    });
    const compact = sampleAcePlannerCompactToken({
      firstTokenId,
      vocabularySize,
      conditionalLogits: conditional,
      unconditionalLogits: unconditional,
      seenTokenIds,
      parameters,
      word,
    });
    expect(compact).toEqual(full);
  }, 15_000);

  it("preserves global tie order, repetition, and full-vocabulary top-k semantics", () => {
    const vocabularySize = 23;
    const firstTokenId = 7;
    const conditional = new Float32Array([
      -5, 2, 4, 4, 1, -2, 0, 3, 3,
    ]);
    const unconditional = new Float32Array([
      -4, 1, 2, 2, 0, -1, 0, 1, 1,
    ]);
    const fullConditional = new Float32Array(vocabularySize);
    const fullUnconditional = new Float32Array(vocabularySize);
    fullConditional.fill(50);
    fullUnconditional.fill(-50);
    fullConditional.set(conditional, firstTokenId);
    fullUnconditional.set(unconditional, firstTokenId);
    const allowedTokens = {
      kind: "range" as const,
      firstTokenId,
      tokenCount: conditional.length,
    };
    const seenTokenIds = [0, firstTokenId + 2, firstTokenId + 2,
      vocabularySize - 1];
    const cases = [
      {
        temperature: 1,
        guidanceScale: 2,
        topK: 1,
        topP: 0.49,
        repetitionPenalty: 1.25,
      },
      {
        temperature: 0.5,
        guidanceScale: 3,
        // A full-vocabulary topK above the nine finite candidates is a no-op.
        topK: vocabularySize,
        topP: 1,
        repetitionPenalty: 1.5,
      },
    ] as const;
    for (const parameters of cases) {
      for (const word of [0, 0x8000_0000, 0xffff_ffff]) {
        const full = sampleAcePlannerToken({
          conditionalLogits: fullConditional,
          unconditionalLogits: fullUnconditional,
          seenTokenIds,
          preCfgAllowedTokens: allowedTokens,
          allowedTokens,
          parameters,
          word,
        });
        const compact = sampleAcePlannerCompactToken({
          firstTokenId,
          vocabularySize,
          conditionalLogits: conditional,
          unconditionalLogits: unconditional,
          seenTokenIds,
          parameters,
          word,
        });
        expect(compact).toEqual(full);
      }
    }

    const conditionalOnly = sampleAcePlannerCompactToken({
      firstTokenId,
      vocabularySize,
      conditionalLogits: conditional,
      seenTokenIds,
      parameters: {
        temperature: 1,
        guidanceScale: 1,
        topK: 1,
        topP: 1,
        repetitionPenalty: 1,
      },
      word: 0,
    });
    expect(conditionalOnly.tokenId).toBe(firstTokenId + 2);
  });

  it("keeps full and compact draws on one commit-after-success Philox cursor", () => {
    const seed = canonicalizeSeed("0123456789abcdef");
    const fullCursor = new AcePlannerSamplingCursor(seed, 17n);
    const compactCursor = new AcePlannerSamplingCursor(seed, 17n);
    const parameters = {
      temperature: 1,
      guidanceScale: 1,
      topK: 0,
      topP: 1,
      repetitionPenalty: 1,
    } as const;
    const full = fullCursor.sample({
      conditionalLogits: [20, -1, 8, 2, 7],
      seenTokenIds: [],
      allowedTokens: { kind: "ids", tokenIds: [3] },
      parameters,
    });
    const compact = compactCursor.sampleCompact({
      firstTokenId: 3,
      vocabularySize: 5,
      conditionalLogits: [2],
      seenTokenIds: [],
      parameters,
    });
    expect(compact).toEqual(full);
    expect(compact.drawIndex).toBe(17n);
    expect(compactCursor.consumed).toBe(18n);

    expect(() => compactCursor.sampleCompact({
      firstTokenId: 3,
      vocabularySize: 5,
      conditionalLogits: [Number.NaN],
      seenTokenIds: [],
      parameters,
    })).toThrow(/finite or negative infinity/);
    expect(compactCursor.consumed).toBe(18n);

    const nextFull = fullCursor.sample({
      conditionalLogits: [0, 0, 0, 4, 1],
      seenTokenIds: [],
      allowedTokens: { kind: "ids", tokenIds: [3, 4] },
      parameters,
    });
    const nextCompact = compactCursor.sampleCompact({
      firstTokenId: 3,
      vocabularySize: 5,
      conditionalLogits: [4, 1],
      seenTokenIds: [],
      parameters,
    });
    expect(nextCompact).toEqual(nextFull);
    expect(nextCompact.drawIndex).toBe(18n);
    expect(compactCursor.consumed).toBe(19n);
  });
});

function finiteTokenIds(values: ArrayLike<number>): number[] {
  const result: number[] = [];
  for (let index = 0; index < values.length; index += 1) {
    if (Number.isFinite(values[index])) result.push(index);
  }
  return result;
}

function floatWords(values: ArrayLike<number>): number[] {
  const storage = new ArrayBuffer(values.length * 4);
  const floats = new Float32Array(storage);
  for (let index = 0; index < values.length; index += 1) {
    floats[index] = values[index]!;
  }
  return [...new Uint32Array(storage)];
}

function realisticSparseLogits(): Float32Array {
  const logits = new Float32Array(217_204);
  logits.fill(Number.NEGATIVE_INFINITY);
  const firstCode = 151_669;
  for (let offset = 0; offset < 64_000; offset += 1) {
    const phase = (offset * 73 + 19) % 257;
    logits[firstCode + offset] = Math.fround(-12 + phase / 16);
  }
  logits[firstCode] = 8;
  logits[firstCode + 1] = 7.5;
  logits[firstCode + 31_999] = 7;
  logits[firstCode + 63_999] = 6.5;
  return logits;
}

function sumBytes(values: Uint8Array): number {
  let total = 0;
  for (const value of values) total += value;
  return total;
}

function positiveCount(values: Float32Array): number {
  let total = 0;
  for (const value of values) {
    if (value > 0) total += 1;
  }
  return total;
}

function sha256Bytes(values: Uint8Array): string {
  return createHash("sha256").update(values).digest("hex");
}

function sha256FloatWords(values: Float32Array): string {
  const bytes = new Uint8Array(values.length * 4);
  const view = new DataView(bytes.buffer);
  const words = floatWords(values);
  for (let index = 0; index < words.length; index += 1) {
    view.setUint32(index * 4, words[index]!, true);
  }
  return sha256Bytes(bytes);
}
