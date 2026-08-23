import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { describe, expect, it } from "vitest";

import {
  ACE_OPT_0012_CORE_SOURCE_IDENTITIES,
  ACE_OPT_0012_FORCED_EOS_ALLOWED_TOKENS,
  ACE_OPT_0012_NEGATIVE_INFINITY_U32,
  ACE_OPT_0012_PRE_CFG_ALLOWED_TOKENS,
  ACE_OPT_0012_RAW_FP16_TIED_WEIGHT_SHARDS,
  ACE_OPT_0012_REGULAR_ALLOWED_TOKENS,
  ACE_OPT_0012_SAMPLING_PARAMETERS,
  AceOpt0012CompactSamplingCursor,
  aceOpt0012Float16BitsToFloat32U32,
  createAceOpt0012CompactSemanticHeadPlan,
  createAceOpt0012PlanRequest,
  decodeAceOpt0012CompactFp16Readback,
  reconstructAceOpt0012FullPlannerLogits,
  sampleAceOpt0012PlannedCompactToken,
  traceAceOpt0012CompactBrowserV1Sampling,
  type AceOpt0012CompactSemanticHeadPlan,
  type AceOpt0012CompactSamplingTrace,
  type AceOpt0012PlanRequest,
} from "../benchmark/opt-0012-compact-semantic-head.js";
import {
  ACE_PLANNER_SEMANTIC_CODE_COUNT,
  ACE_PLANNER_SEMANTIC_FIRST_TOKEN_ID,
} from "../src/runtime/planner.js";
import {
  AcePlannerSamplingCursor,
  applyAcePlannerRepetitionPenalty,
  applyAcePlannerTopK,
  combineAcePlannerCfgLogits,
  createAcePlannerBrowserSamplingWeights,
  createAcePlannerBrowserTopPKeep,
  createAcePlannerFilteredLogits,
  sampleAcePlannerToken,
  type AcePlannerSamplingParameters,
} from "../src/runtime/planner-sampling.js";
import { aceCategoricalTokenFromWord, canonicalizeSeed } from "../src/runtime/seed.js";
import { ACE_QWEN_IM_END_TOKEN_ID } from "../src/tokenizer/qwen-bpe.js";

const SHARD_BOUNDARY_LOCAL = 44_939;
const FIXTURE_FP16_BITS = Object.freeze([
  0x3c00,
  0xc000,
  0x0001,
  0x7bff,
  0x8000,
  0x3555,
]);
// Independent literal binary32 expansions for FIXTURE_FP16_BITS. The decode
// implementation under test is never used to construct these expected words.
const FIXTURE_FLOAT32_WORDS = Object.freeze([
  0x3f80_0000,
  0xc000_0000,
  0x3380_0000,
  0x477f_e000,
  0x8000_0000,
  0x3eaa_a000,
]);

describe("OPT-0012 compact semantic head static plan", () => {
  it("authenticates every production and transitive core dependency", () => {
    expect(Object.isFrozen(ACE_OPT_0012_CORE_SOURCE_IDENTITIES)).toBe(true);
    expect(ACE_OPT_0012_CORE_SOURCE_IDENTITIES).toEqual({
      "src/model/manifest.ts":
        "77213ed8c096d0e2b49cd4aeb7c0eb96f63c12671173d7d83b7b6eb6331916b1",
      "src/runtime/planner.ts":
        "35c1effda1a52e6a73fabfb467a2ac7fd015197a4a6e494d360f9db78b7950b1",
      "src/runtime/planner-sampling.ts":
        "67055acfbb96e10682092e5d0ccfa9a5d822fd708a091fe46f7f47458226d0f3",
      "src/runtime/seed.ts":
        "157f349808aff5b4e64eb82e36f7b3ace7483845c78336e94ee6707557ee557e",
      "src/tokenizer/qwen-bpe.ts":
        "5a83fa16a178621e8b9de457800564e1c4b832bc70e39d627b39cd0678064737",
      "src/webgpu/planner-model.ts":
        "673a5a3d45a749c12337cc6186212084a4b05c6ffde7f02b782a7e033113cce6",
      "src/webgpu/kernels/gemm.ts":
        "8922c2cebe36186b7cbf173e69126dd45a01fbc79a2cb2f84c34090880268928",
      "src/webgpu/kernels/correctness-utils.ts":
        "f727d347e666e4f4ebdedc7acedcc6b1cdde371850b9fc83a2c214cb83290018",
      "src/webgpu/qwen3.ts":
        "bb982aeb380c52ea842183b1649a6674203c18a421c17ea07f68d7668309158a",
    });
    for (const [file, expectedSha256] of Object.entries(
      ACE_OPT_0012_CORE_SOURCE_IDENTITIES,
    )) {
      const actualSha256 = createHash("sha256")
        .update(readFileSync(resolve(process.cwd(), file)))
        .digest("hex");
      expect(actualSha256, file).toBe(expectedSha256);
    }
  });

  it("derives the exact regular intersections, source bindings, and GEMM work", () => {
    const plan = regularPlan();
    expect(plan).toMatchObject({
      contractId: "opt-0012-compact-semantic-head-v1",
      phase: "semantic-m2",
      state: "regular-code",
      modelProfile: "raw-fp16",
      rows: 2,
      conditionalRow: 0,
      unconditionalRow: 1,
      vocabularySize: 217_204,
      hiddenSize: 1_024,
      firstCandidateTokenId: 151_669,
      candidateCount: 64_000,
      lastCandidateTokenId: 215_668,
      logicalWeightTrafficBytes: 131_072_000,
      logicalMultiplyAdds: 131_072_000,
      scheduledMultiplyAdds: 1_050_673_152,
      workgroupCount: 501,
    });
    expect(plan.intersections).toEqual([
      {
        shardIndex: 3,
        globalFirstRow: 151_669,
        globalLastRow: 196_607,
        localFirstRow: 4_213,
        rowCount: 44_939,
      },
      {
        shardIndex: 4,
        globalFirstRow: 196_608,
        globalLastRow: 215_668,
        localFirstRow: 0,
        rowCount: 19_061,
      },
    ]);
    expect(plan.headSlices).toEqual([
      expect.objectContaining({
        shardIndex: 3,
        sourceTensorName:
          "planner.model.embed_tokens.weight.rows-147456-196608",
        sourceBindingByteOffset: 8_628_224,
        sourceBindingByteLength: 92_035_072,
        sourceBindingByteEnd: 100_663_296,
        sourceOwnerByteOffset: 0,
        sourceOwnerByteLength: 100_663_296,
        sourceOwnerByteEnd: 100_663_296,
        logicalWeightTrafficBytes: 92_035_072,
        logicalMultiplyAdds: 92_035_072,
        scheduledMultiplyAdds: 738_197_504,
        workgroupCount: 352,
        rawLogitBytes: 179_756,
        outputRanges: [{
          firstOutput: 0,
          outputCount: 89_878,
          firstWorkgroup: 0,
          workgroupCount: 352,
          multiplyAdds: 738_197_504,
        }],
      }),
      expect.objectContaining({
        shardIndex: 4,
        sourceBindingByteOffset: 0,
        sourceBindingByteLength: 39_036_928,
        sourceBindingByteEnd: 39_036_928,
        sourceOwnerByteLength: 42_180_608,
        sourceOwnerByteEnd: 42_180_608,
        logicalWeightTrafficBytes: 39_036_928,
        logicalMultiplyAdds: 39_036_928,
        scheduledMultiplyAdds: 312_475_648,
        workgroupCount: 149,
        rawLogitBytes: 76_244,
        outputRanges: [{
          firstOutput: 0,
          outputCount: 38_122,
          firstWorkgroup: 0,
          workgroupCount: 149,
          multiplyAdds: 312_475_648,
        }],
      }),
    ]);
  });

  it("freezes the unpadded shard-major regular readback and four logical spans", () => {
    const readback = regularPlan().readback;
    expect(readback).toMatchObject({
      physicalLayout: "shard-major-row-major",
      rawLogitBytes: 256_000,
      writeStatusByteOffset: 256_000,
      writeStatusByteLength: 8,
      usedBytes: 256_008,
      alignmentPaddingBytes: 248,
      allocationBytes: 256_256,
    });
    expect(readback.copies).toEqual([
      {
        index: 0,
        kind: "logits",
        shardIndex: 3,
        sourceByteOffset: 0,
        destinationByteOffset: 0,
        byteLength: 179_756,
      },
      {
        index: 1,
        kind: "logits",
        shardIndex: 4,
        sourceByteOffset: 0,
        destinationByteOffset: 179_756,
        byteLength: 76_244,
      },
      {
        index: 2,
        kind: "write-status",
        shardIndex: null,
        sourceByteOffset: 0,
        destinationByteOffset: 256_000,
        byteLength: 8,
      },
    ]);
    expect(readback.logicalSpans).toEqual([
      {
        shardIndex: 3,
        physicalRow: 0,
        sourceByteOffset: 0,
        byteLength: 89_878,
        destinationCandidateOffset: 0,
        candidateCount: 44_939,
        globalFirstTokenId: 151_669,
      },
      {
        shardIndex: 3,
        physicalRow: 1,
        sourceByteOffset: 89_878,
        byteLength: 89_878,
        destinationCandidateOffset: 0,
        candidateCount: 44_939,
        globalFirstTokenId: 151_669,
      },
      {
        shardIndex: 4,
        physicalRow: 0,
        sourceByteOffset: 179_756,
        byteLength: 38_122,
        destinationCandidateOffset: 44_939,
        candidateCount: 19_061,
        globalFirstTokenId: 196_608,
      },
      {
        shardIndex: 4,
        physicalRow: 1,
        sourceByteOffset: 217_878,
        byteLength: 38_122,
        destinationCandidateOffset: 44_939,
        candidateCount: 19_061,
        globalFirstTokenId: 196_608,
      },
    ]);
    expect(readback.copies.every((copy) =>
      copy.sourceByteOffset % 4 === 0 &&
      copy.destinationByteOffset % 4 === 0 &&
      copy.byteLength % 4 === 0
    )).toBe(true);
  });

  it("derives the separate one-row EOS slice and exact 256-byte allocation", () => {
    const plan = forcedEosPlan();
    expect(plan).toMatchObject({
      state: "forced-eos",
      firstCandidateTokenId: 151_645,
      candidateCount: 1,
      lastCandidateTokenId: 151_645,
      logicalWeightTrafficBytes: 2_048,
      logicalMultiplyAdds: 2_048,
      scheduledMultiplyAdds: 2_097_152,
      workgroupCount: 1,
    });
    expect(plan.intersections).toEqual([{
      shardIndex: 3,
      globalFirstRow: 151_645,
      globalLastRow: 151_645,
      localFirstRow: 4_189,
      rowCount: 1,
    }]);
    expect(plan.headSlices[0]).toMatchObject({
      sourceBindingByteOffset: 8_579_072,
      sourceBindingByteLength: 2_048,
      sourceBindingByteEnd: 8_581_120,
      rawLogitBytes: 4,
      outputRanges: [{
        firstOutput: 0,
        outputCount: 2,
        firstWorkgroup: 0,
        workgroupCount: 1,
        multiplyAdds: 2_097_152,
      }],
    });
    expect(plan.readback).toEqual({
      physicalLayout: "shard-major-row-major",
      copies: [
        {
          index: 0,
          kind: "logits",
          shardIndex: 3,
          sourceByteOffset: 0,
          destinationByteOffset: 0,
          byteLength: 4,
        },
        {
          index: 1,
          kind: "write-status",
          shardIndex: null,
          sourceByteOffset: 0,
          destinationByteOffset: 4,
          byteLength: 8,
        },
      ],
      logicalSpans: [
        {
          shardIndex: 3,
          physicalRow: 0,
          sourceByteOffset: 0,
          byteLength: 2,
          destinationCandidateOffset: 0,
          candidateCount: 1,
          globalFirstTokenId: 151_645,
        },
        {
          shardIndex: 3,
          physicalRow: 1,
          sourceByteOffset: 2,
          byteLength: 2,
          destinationCandidateOffset: 0,
          candidateCount: 1,
          globalFirstTokenId: 151_645,
        },
      ],
      rawLogitBytes: 4,
      writeStatusByteOffset: 4,
      writeStatusByteLength: 8,
      usedBytes: 12,
      alignmentPaddingBytes: 244,
      allocationBytes: 256,
    });
  });

  it("derives all five raw-FP16 bound parts from production shard constants", () => {
    expect(ACE_OPT_0012_RAW_FP16_TIED_WEIGHT_SHARDS).toEqual([
      expectedSourceShard(0, 0, 49_152, 100_663_296),
      expectedSourceShard(1, 49_152, 49_152, 100_663_296),
      expectedSourceShard(2, 98_304, 49_152, 100_663_296),
      expectedSourceShard(3, 147_456, 49_152, 100_663_296),
      expectedSourceShard(4, 196_608, 20_596, 42_180_608),
    ]);
  });

  it("rejects M1, altered geometry/layout/settings/constraints/FSM state", () => {
    const base = createAceOpt0012PlanRequest("regular-code");
    const rejected: readonly [string, AceOpt0012PlanRequest][] = [
      ["M1", { ...base, phase: "cot-m1" }],
      ["state", { ...base, state: "other" }],
      ["profile", { ...base, modelProfile: "reference-bf16" }],
      ["rows", { ...base, rows: 1 }],
      ["conditional row", { ...base, conditionalRow: 1 }],
      ["unconditional row", { ...base, unconditionalRow: 0 }],
      ["vocabulary", { ...base, vocabularySize: 217_203 }],
      ["hidden", { ...base, hiddenSize: 2_048 }],
      ["manifest layout", { ...base, manifestWeightLayout: "source-row-major" }],
      ["kernel layout", { ...base, kernelWeightLayout: "tile-major" }],
      ["weight dtype", { ...base, weightDtype: "float32" }],
      ["output dtype", { ...base, outputDtype: "float32" }],
      ["readback layout", { ...base, physicalReadbackLayout: "row-major" }],
      ["oracle", { ...base, softmaxOracleId: "diagnostic" }],
      ["temperature", {
        ...base,
        sampling: { ...base.sampling, temperature: 1 },
      }],
      ["guidance", {
        ...base,
        sampling: { ...base.sampling, guidanceScale: 1 },
      }],
      ["top-k", { ...base, sampling: { ...base.sampling, topK: 1 } }],
      ["top-p", { ...base, sampling: { ...base.sampling, topP: 1 } }],
      ["penalty", {
        ...base,
        sampling: { ...base.sampling, repetitionPenalty: 1.1 },
      }],
      ["pre-CFG end", {
        ...base,
        preCfgAllowedTokens: {
          kind: "range",
          firstTokenId: ACE_PLANNER_SEMANTIC_FIRST_TOKEN_ID,
          tokenCount: ACE_PLANNER_SEMANTIC_CODE_COUNT - 1,
          additionalTokenIds: [ACE_QWEN_IM_END_TOKEN_ID],
        },
      }],
      ["pre-CFG EOS", {
        ...base,
        preCfgAllowedTokens: ACE_OPT_0012_REGULAR_ALLOWED_TOKENS,
      }],
      ["regular EOS", {
        ...base,
        allowedTokens: ACE_OPT_0012_PRE_CFG_ALLOWED_TOKENS,
      }],
      ["regular complete", {
        ...base,
        acceptedRegularCodeCount: base.requestedRegularCodeCount,
      }],
      ["zero request", { ...base, requestedRegularCodeCount: 0 }],
    ];
    for (const [label, request] of rejected) {
      expect(
        () => createAceOpt0012CompactSemanticHeadPlan(request),
        label,
      ).toThrow();
    }

    const eos = createAceOpt0012PlanRequest("forced-eos");
    expect(() => createAceOpt0012CompactSemanticHeadPlan({
      ...eos,
      acceptedRegularCodeCount: eos.requestedRegularCodeCount - 1,
    })).toThrow(/count\/state/);
    expect(() => createAceOpt0012CompactSemanticHeadPlan({
      ...eos,
      allowedTokens: ACE_OPT_0012_FORCED_EOS_ALLOWED_TOKENS,
      preCfgAllowedTokens: {
        kind: "range",
        firstTokenId: ACE_PLANNER_SEMANTIC_FIRST_TOKEN_ID,
        tokenCount: ACE_PLANNER_SEMANTIC_CODE_COUNT,
        additionalTokenIds: [ACE_QWEN_IM_END_TOKEN_ID, ACE_QWEN_IM_END_TOKEN_ID],
      },
    })).toThrow(/pre-CFG/);
  });

  it("rejects wrong shard identity, offsets, extents, layout, and buffer bounds", () => {
    const cases: Array<readonly [string, number, Record<string, unknown>]> = [
      ["index", 3, { shardIndex: 2 }],
      ["name", 3, { tensorName: "wrong" }],
      ["first row", 3, { firstRow: 147_457 }],
      ["row count", 4, { rowCount: 20_595 }],
      ["offset", 3, { bindingByteOffset: 2 }],
      ["length", 3, { bindingByteLength: 100_663_294 }],
      ["manifest layout", 3, { manifestLayout: "source-row-major" }],
      ["kernel layout", 3, { kernelWeightLayout: "tile-major" }],
      ["dtype", 3, { dtype: "float32" }],
      ["buffer", 3, { bufferByteLength: 100_663_295 }],
    ];
    for (const [label, index, patch] of cases) {
      const request = createAceOpt0012PlanRequest("regular-code");
      const shards = request.tiedWeightShards.map((shard) => ({ ...shard }));
      shards[index] = { ...shards[index]!, ...patch } as typeof shards[number];
      expect(() => createAceOpt0012CompactSemanticHeadPlan({
        ...request,
        tiedWeightShards: shards,
      }), label).toThrow();
    }
    const short = createAceOpt0012PlanRequest("regular-code");
    expect(() => createAceOpt0012CompactSemanticHeadPlan({
      ...short,
      tiedWeightShards: short.tiedWeightShards.slice(0, 4),
    })).toThrow(/count/);
  });
});

describe("OPT-0012 compact FP16 readback and arm-B reconstruction", () => {
  it("exhaustively decodes all 128,000 logical FP16 values with a fixed receipt", () => {
    const plan = regularPlan();
    const fixture = mappedFixture(plan);
    const paddingBefore = new Uint8Array(
      fixture.buffer,
      plan.readback.usedBytes,
      plan.readback.alignmentPaddingBytes,
    ).slice();
    const decoded = decodeAceOpt0012CompactFp16Readback(fixture.buffer, plan);
    const actualWords = new Uint32Array(2 * plan.candidateCount);
    let comparedWordCount = 0;
    let mismatchCount = 0;
    const rows = [decoded.conditionalLogits, decoded.unconditionalLogits];
    for (let row = 0; row < rows.length; row += 1) {
      for (let candidate = 0; candidate < plan.candidateCount; candidate += 1) {
        const actualWord = floatWord(rows[row]![candidate]!);
        const expectedWord = independentFixtureWord(row, candidate);
        actualWords[comparedWordCount] = actualWord;
        comparedWordCount += 1;
        if (actualWord !== expectedWord) mismatchCount += 1;
      }
    }
    expect(comparedWordCount).toBe(128_000);
    expect(mismatchCount).toBe(0);
    expect(sha256LittleEndianWords(actualWords)).toBe(
      "c5264b4c3ec24ae97094de17f8577d8249305104559b00a7739c321c33025b71",
    );
    expect([...decoded.writeStatus]).toEqual([1, 1]);
    expect(new Uint8Array(
      fixture.buffer,
      plan.readback.usedBytes,
      plan.readback.alignmentPaddingBytes,
    )).toEqual(paddingBefore);
  });

  it("decodes a 128,000-value multi-span fixture covering every FP16 word", () => {
    const plan = regularPlan();
    expect(plan.readback.logicalSpans).toHaveLength(4);
    expect(new Set(plan.readback.logicalSpans.map((span) => span.shardIndex)))
      .toEqual(new Set([3, 4]));
    const fixture = exhaustiveMappedFixture(plan);
    const paddingBefore = new Uint8Array(
      fixture.buffer,
      plan.readback.usedBytes,
      plan.readback.alignmentPaddingBytes,
    ).slice();
    const decoded = decodeAceOpt0012CompactFp16Readback(fixture.buffer, plan);
    const actualWords = new Uint32Array(2 * plan.candidateCount);
    actualWords.set(
      new Uint32Array(
        decoded.conditionalLogits.buffer,
        decoded.conditionalLogits.byteOffset,
        decoded.conditionalLogits.length,
      ),
      0,
    );
    actualWords.set(
      new Uint32Array(
        decoded.unconditionalLogits.buffer,
        decoded.unconditionalLogits.byteOffset,
        decoded.unconditionalLogits.length,
      ),
      plan.candidateCount,
    );
    const coveredFp16Words = new Uint8Array(0x1_0000);
    let mismatchCount = 0;
    for (let index = 0; index < actualWords.length; index += 1) {
      const bits = index & 0xffff;
      coveredFp16Words[bits] = 1;
      if (actualWords[index] !== independentCanonicalFloat16Word(bits)) {
        mismatchCount += 1;
      }
    }
    expect(actualWords).toHaveLength(128_000);
    expect(coveredFp16Words.every((covered) => covered === 1)).toBe(true);
    expect(mismatchCount).toBe(0);
    expect(sha256LittleEndianWords(actualWords)).toBe(
      "5ce3d3bfd1774760f7886f7373ca8e9a09795e2ebb59a1d9bd573e4dee3017b9",
    );
    expect([...decoded.writeStatus]).toEqual([1, 1]);
    expect([...new Uint32Array(
      fixture.buffer,
      plan.readback.writeStatusByteOffset,
      2,
    )]).toEqual([1, 1]);
    expect(new Uint8Array(
      fixture.buffer,
      plan.readback.usedBytes,
      plan.readback.alignmentPaddingBytes,
    )).toEqual(paddingBefore);
  });

  it("exhaustively reconstructs all 434,408 full-vector words with a fixed receipt", () => {
    const plan = regularPlan();
    const decoded = decodeAceOpt0012CompactFp16Readback(
      mappedFixture(plan).buffer,
      plan,
    );
    const [conditional, unconditional] = reconstructAceOpt0012FullPlannerLogits(
      decoded,
      plan,
    );
    const rows = [conditional, unconditional];
    const actualWords = new Uint32Array(2 * plan.vocabularySize);
    let comparedWordCount = 0;
    let mismatchCount = 0;
    for (let row = 0; row < rows.length; row += 1) {
      for (let tokenId = 0; tokenId < plan.vocabularySize; tokenId += 1) {
        const candidate = tokenId - plan.firstCandidateTokenId;
        const expectedWord = candidate >= 0 && candidate < plan.candidateCount
          ? independentFixtureWord(row, candidate)
          : ACE_OPT_0012_NEGATIVE_INFINITY_U32;
        const actualWord = floatWord(rows[row]![tokenId]!);
        actualWords[comparedWordCount] = actualWord;
        comparedWordCount += 1;
        if (actualWord !== expectedWord) mismatchCount += 1;
      }
    }
    expect(comparedWordCount).toBe(434_408);
    expect(mismatchCount).toBe(0);
    expect(sha256LittleEndianWords(actualWords)).toBe(
      "35c23ac37fe5d3d666476873fdc26efd91514996fbd971e14055639f0dfb9acb",
    );
  });

  it("decodes forced EOS alone and leaves every regular code at negative infinity", () => {
    const plan = forcedEosPlan();
    const fixture = mappedFixture(plan);
    const decoded = decodeAceOpt0012CompactFp16Readback(fixture.buffer, plan);
    const [conditional, unconditional] = reconstructAceOpt0012FullPlannerLogits(
      decoded,
      plan,
    );
    expect(floatWord(conditional[ACE_QWEN_IM_END_TOKEN_ID]!)).toBe(
      floatWord(decoded.conditionalLogits[0]!),
    );
    expect(floatWord(unconditional[ACE_QWEN_IM_END_TOKEN_ID]!)).toBe(
      floatWord(decoded.unconditionalLogits[0]!),
    );
    expect(floatWord(conditional[ACE_PLANNER_SEMANTIC_FIRST_TOKEN_ID]!)).toBe(
      ACE_OPT_0012_NEGATIVE_INFINITY_U32,
    );
    expect(floatWord(
      conditional[ACE_PLANNER_SEMANTIC_FIRST_TOKEN_ID + 63_999]!,
    )).toBe(ACE_OPT_0012_NEGATIVE_INFINITY_U32);
  });

  it("pins signed zero, finite boundaries, infinities, and NaN payloads", () => {
    expect([
      aceOpt0012Float16BitsToFloat32U32(0x0000),
      aceOpt0012Float16BitsToFloat32U32(0x8000),
      aceOpt0012Float16BitsToFloat32U32(0x0001),
      aceOpt0012Float16BitsToFloat32U32(0x03ff),
      aceOpt0012Float16BitsToFloat32U32(0x0400),
      aceOpt0012Float16BitsToFloat32U32(0x3c00),
      aceOpt0012Float16BitsToFloat32U32(0x7bff),
      aceOpt0012Float16BitsToFloat32U32(0x7c00),
      aceOpt0012Float16BitsToFloat32U32(0xfc00),
      aceOpt0012Float16BitsToFloat32U32(0x7c01),
      aceOpt0012Float16BitsToFloat32U32(0x7dff),
      aceOpt0012Float16BitsToFloat32U32(0x7e00),
      aceOpt0012Float16BitsToFloat32U32(0xfc01),
      aceOpt0012Float16BitsToFloat32U32(0xffff),
    ]).toEqual([
      0x0000_0000,
      0x8000_0000,
      0x3380_0000,
      0x387f_c000,
      0x3880_0000,
      0x3f80_0000,
      0x477f_e000,
      0x7f80_0000,
      0xff80_0000,
      0x7fc0_2000,
      0x7fff_e000,
      0x7fc0_0000,
      0xffc0_2000,
      0xffff_e000,
    ]);
    const signedZeroWords = new Uint32Array([
      aceOpt0012Float16BitsToFloat32U32(0x0000),
      aceOpt0012Float16BitsToFloat32U32(0x8000),
    ]);
    const signedZeros = new Float32Array(signedZeroWords.buffer);
    expect(Object.is(signedZeros[0], 0)).toBe(true);
    expect(Object.is(signedZeros[1], -0)).toBe(true);
    for (const value of [-1, 0x1_0000, 1.5, Number.NaN]) {
      expect(() => aceOpt0012Float16BitsToFloat32U32(value)).toThrow(/16-bit/);
    }
  });

  it("deterministically expands the complete FP16 domain under the stable NaN envelope", () => {
    const firstWords = new Uint32Array(0x1_0000);
    const secondWords = new Uint32Array(0x1_0000);
    const nonNaNWords = new Uint32Array(63_490);
    const canonicalNaNWords = new Uint32Array(2_046);
    const inputClasses = {
      zero: 0,
      subnormal: 0,
      normal: 0,
      infinity: 0,
      quietNaN: 0,
      signalingNaN: 0,
    };
    const candidateNaNs = {
      positiveQuiet: 0,
      positiveSignaling: 0,
      negativeQuiet: 0,
      negativeSignaling: 0,
    };
    let nonNaNCursor = 0;
    let nanCursor = 0;
    let nonNaNMismatchCount = 0;
    let nanMismatchCount = 0;
    let candidateCallMismatchCount = 0;
    for (let bits = 0; bits <= 0xffff; bits += 1) {
      const first = aceOpt0012Float16BitsToFloat32U32(bits);
      const second = aceOpt0012Float16BitsToFloat32U32(bits);
      const canonical = independentCanonicalFloat16Word(bits);
      firstWords[bits] = first;
      secondWords[bits] = second;
      if (first !== second) candidateCallMismatchCount += 1;
      const exponent = (bits >>> 10) & 0x1f;
      const mantissa = bits & 0x03ff;
      if (exponent === 0) {
        inputClasses[mantissa === 0 ? "zero" : "subnormal"] += 1;
        nonNaNWords[nonNaNCursor] = first;
        nonNaNCursor += 1;
        if (first !== canonical) nonNaNMismatchCount += 1;
      } else if (exponent !== 0x1f) {
        inputClasses.normal += 1;
        nonNaNWords[nonNaNCursor] = first;
        nonNaNCursor += 1;
        if (first !== canonical) nonNaNMismatchCount += 1;
      } else if (mantissa === 0) {
        inputClasses.infinity += 1;
        nonNaNWords[nonNaNCursor] = first;
        nonNaNCursor += 1;
        if (first !== canonical) nonNaNMismatchCount += 1;
      } else if ((mantissa & 0x0200) !== 0) {
        inputClasses.quietNaN += 1;
        canonicalNaNWords[nanCursor] = first;
        nanCursor += 1;
        if (first !== canonical) nanMismatchCount += 1;
      } else {
        inputClasses.signalingNaN += 1;
        canonicalNaNWords[nanCursor] = first;
        nanCursor += 1;
        if (first !== canonical) nanMismatchCount += 1;
      }
      if (exponent === 0x1f && mantissa !== 0) {
        const positive = (first & 0x8000_0000) === 0;
        const quiet = (first & 0x0040_0000) !== 0;
        if (positive && quiet) candidateNaNs.positiveQuiet += 1;
        else if (positive) candidateNaNs.positiveSignaling += 1;
        else if (quiet) candidateNaNs.negativeQuiet += 1;
        else candidateNaNs.negativeSignaling += 1;
      }
    }
    expect(candidateCallMismatchCount).toBe(0);
    expect(firstWords).toEqual(secondWords);
    expect(nonNaNCursor).toBe(63_490);
    expect(nanCursor).toBe(2_046);
    expect(nonNaNMismatchCount).toBe(0);
    expect(nanMismatchCount).toBe(0);
    expect(inputClasses).toEqual({
      zero: 2,
      subnormal: 2_046,
      normal: 61_440,
      infinity: 2,
      quietNaN: 1_024,
      signalingNaN: 1_022,
    });
    expect(candidateNaNs).toEqual({
      positiveQuiet: 1_023,
      positiveSignaling: 0,
      negativeQuiet: 1_023,
      negativeSignaling: 0,
    });
    const outputClasses = {
      zero: 0,
      finiteNonzero: 0,
      infinity: 0,
      nan: 0,
    };
    for (const value of new Float32Array(firstWords.buffer)) {
      if (Number.isNaN(value)) outputClasses.nan += 1;
      else if (!Number.isFinite(value)) outputClasses.infinity += 1;
      else if (value === 0) outputClasses.zero += 1;
      else outputClasses.finiteNonzero += 1;
    }
    expect(outputClasses).toEqual({
      zero: 2,
      finiteNonzero: 63_486,
      infinity: 2,
      nan: 2_046,
    });
    expect(sha256LittleEndianWords(nonNaNWords)).toBe(
      "680bbc22915f61aa1bbfc7265bc3882a6aa42d299bfd2c571807196e5544de2e",
    );
    expect(sha256LittleEndianWords(canonicalNaNWords)).toBe(
      "32f37d24c421f50695da47516d20cffa27c96fb2be53d1ac6f88cfbc1cec9039",
    );
    expect(sha256LittleEndianWords(firstWords)).toBe(
      "b636c5716ff84d972782faf02d0194cb8951526bea4cc487082feb47b1860ddf",
    );
    expect(sha256LittleEndianWords(secondWords)).toBe(
      "b636c5716ff84d972782faf02d0194cb8951526bea4cc487082feb47b1860ddf",
    );
  });

  it("rejects short/long buffers, failed status, malformed spans, and wrong row lengths", () => {
    const plan = regularPlan();
    const fixture = mappedFixture(plan);
    expect(() => decodeAceOpt0012CompactFp16Readback(
      fixture.buffer.slice(0, -1),
      plan,
    )).toThrow(/exactly 256256/);
    const long = new ArrayBuffer(plan.readback.allocationBytes + 4);
    expect(() => decodeAceOpt0012CompactFp16Readback(long, plan)).toThrow(
      /exactly 256256/,
    );
    new Uint32Array(
      fixture.buffer,
      plan.readback.writeStatusByteOffset,
      2,
    )[1] = 0;
    expect(() => decodeAceOpt0012CompactFp16Readback(fixture.buffer, plan)).toThrow(
      /physical row 1/,
    );

    const malformed = {
      ...plan,
      readback: {
        ...plan.readback,
        logicalSpans: plan.readback.logicalSpans.map((span, index) =>
          index === 2 ? { ...span, sourceByteOffset: span.sourceByteOffset + 2 } : span),
      },
    } as AceOpt0012CompactSemanticHeadPlan;
    expect(() => decodeAceOpt0012CompactFp16Readback(
      mappedFixture(plan).buffer,
      malformed,
    )).toThrow(/issued immutable/);
    expect(() => reconstructAceOpt0012FullPlannerLogits({
      conditionalLogits: new Float32Array(plan.candidateCount - 1),
      unconditionalLogits: new Float32Array(plan.candidateCount),
      writeStatus: new Uint32Array([1, 1]),
    }, plan)).toThrow(/64000 candidates/);
    expect(() => reconstructAceOpt0012FullPlannerLogits({
      conditionalLogits: new Float32Array(plan.candidateCount),
      unconditionalLogits: new Float32Array(plan.candidateCount),
      writeStatus: new Uint32Array([1, 0]),
    }, plan)).toThrow(/two successful status words/);
  });
});

describe("OPT-0012 exact compact browser-v1 sampling", () => {
  it("matches the full-vector sampler at every FP32 boundary across varied settings", () => {
    const plan = regularPlan();
    const cases: readonly Readonly<{
      label: string;
      salt: number;
      parameters: AcePlannerSamplingParameters;
      seen: readonly number[];
      word: number;
      expectedTokenId?: number;
      mutate?: (conditional: Float32Array, unconditional: Float32Array) => void;
    }>[] = [
      {
        label: "product settings",
        salt: 1,
        parameters: ACE_OPT_0012_SAMPLING_PARAMETERS,
        seen: [17, plan.firstCandidateTokenId + 2,
          plan.firstCandidateTokenId + 2, plan.lastCandidateTokenId],
        word: 0x1234_5678,
      },
      {
        label: "shard-boundary top-k tie",
        salt: 7,
        parameters: {
          temperature: 1,
          guidanceScale: 3,
          topK: 1,
          topP: 1,
          repetitionPenalty: 1.25,
        },
        seen: [plan.firstCandidateTokenId, plan.firstCandidateTokenId + 10],
        word: 0xffff_ffff,
        mutate(conditional, unconditional) {
          conditional.fill(-8);
          unconditional.fill(-8);
          conditional[SHARD_BOUNDARY_LOCAL - 1] = 6;
          conditional[SHARD_BOUNDARY_LOCAL] = 6;
          unconditional[SHARD_BOUNDARY_LOCAL - 1] = 0;
          unconditional[SHARD_BOUNDARY_LOCAL] = 0;
        },
      },
      {
        label: "top-p cutoff ties and repetition",
        salt: 19,
        parameters: {
          temperature: 0.25,
          guidanceScale: 1.5,
          topK: 31,
          topP: 0.37,
          repetitionPenalty: 1.3,
        },
        seen: [0, ACE_QWEN_IM_END_TOKEN_ID,
          plan.firstCandidateTokenId + 17,
          plan.firstCandidateTokenId + 31_999],
        word: 0,
      },
      {
        label: "first candidate dominant",
        salt: 31,
        parameters: {
          temperature: 2,
          guidanceScale: 2,
          topK: plan.vocabularySize,
          topP: 0.000_001,
          repetitionPenalty: 1,
        },
        seen: [],
        word: 0x8000_0000,
        expectedTokenId: plan.firstCandidateTokenId,
        mutate(conditional, unconditional) {
          conditional.fill(-120);
          unconditional.fill(-120);
          conditional[0] = 12;
          unconditional[0] = 0;
          conditional[conditional.length - 1] = 11.5;
          unconditional[unconditional.length - 1] = 0;
        },
      },
      {
        label: "last candidate dominant",
        salt: 37,
        parameters: {
          temperature: 2,
          guidanceScale: 2,
          topK: plan.vocabularySize,
          topP: 0.000_001,
          repetitionPenalty: 1,
        },
        seen: [],
        word: 0xffff_ffff,
        expectedTokenId: plan.lastCandidateTokenId,
        mutate(conditional, unconditional) {
          conditional.fill(-120);
          unconditional.fill(-120);
          conditional[0] = 11.5;
          unconditional[0] = 0;
          conditional[conditional.length - 1] = 12;
          unconditional[unconditional.length - 1] = 0;
        },
      },
    ];

    for (const fixture of cases) {
      const { conditional, unconditional } = compactRows(plan, fixture.salt);
      fixture.mutate?.(conditional, unconditional);
      const compact = traceAceOpt0012CompactBrowserV1Sampling({
        plan,
        conditionalLogits: conditional,
        unconditionalLogits: unconditional,
        seenTokenIds: fixture.seen,
        parameters: fixture.parameters,
        word: fixture.word,
      });
      const full = fullVectorTrace(
        plan,
        conditional,
        unconditional,
        fixture.seen,
        fixture.parameters,
        fixture.word,
      );
      compareCompactToFull(plan, compact, full);
      if (fixture.expectedTokenId !== undefined) {
        expect(compact.tokenId, fixture.label).toBe(fixture.expectedTokenId);
        expect(full.tokenId, fixture.label).toBe(fixture.expectedTokenId);
      }
    }
  }, 30_000);

  it("preserves ascending global tie order across shard 3/4 for top-k and top-p", () => {
    const plan = regularPlan();
    const conditional = new Float32Array(plan.candidateCount);
    const unconditional = new Float32Array(plan.candidateCount);
    conditional.fill(-20);
    unconditional.fill(-20);
    const left = SHARD_BOUNDARY_LOCAL - 1;
    const right = SHARD_BOUNDARY_LOCAL;
    conditional[left] = 5;
    conditional[right] = 5;
    unconditional[left] = 1;
    unconditional[right] = 1;
    const trace = traceAceOpt0012CompactBrowserV1Sampling({
      plan,
      conditionalLogits: conditional,
      unconditionalLogits: unconditional,
      seenTokenIds: [],
      parameters: {
        temperature: 1,
        guidanceScale: 2,
        topK: 1,
        topP: 0.49,
        repetitionPenalty: 1,
      },
      word: 0xffff_ffff,
    });
    expect(trace.topKGlobalTokenIds).toEqual([
      plan.firstCandidateTokenId + left,
      plan.firstCandidateTokenId + right,
    ]);
    // Stable global-ID order retains the first threshold-crossing token and
    // shifts the nucleus removal mask, so the lower global ID wins this tie.
    expect(trace.topPGlobalTokenIds).toEqual([
      plan.firstCandidateTokenId + left,
    ]);
    expect(trace.tokenId).toBe(plan.firstCandidateTokenId + left);
  });

  it("preserves subnormal/zero tails, dominant endpoints, and positive count", () => {
    const plan = regularPlan();
    const conditional = new Float32Array(plan.candidateCount);
    const unconditional = new Float32Array(plan.candidateCount);
    conditional.fill(-110);
    unconditional.fill(-110);
    conditional[0] = 0;
    unconditional[0] = 0;
    conditional[plan.candidateCount - 1] = -103.9;
    unconditional[plan.candidateCount - 1] = -103.9;
    const trace = traceAceOpt0012CompactBrowserV1Sampling({
      plan,
      conditionalLogits: conditional,
      unconditionalLogits: unconditional,
      seenTokenIds: [],
      parameters: {
        temperature: 1,
        guidanceScale: 2,
        topK: 0,
        topP: 1,
        repetitionPenalty: 1,
      },
      word: 0xffff_ffff,
    });
    expect(trace.weights[0]).toBeGreaterThan(0);
    expect(trace.weights[plan.candidateCount - 1]).toBeGreaterThan(0);
    expect(trace.positiveCandidateCount).toBe(2);
    expect(trace.tokenId).toBe(plan.firstCandidateTokenId);
  });

  it("forces EOS yet consumes the supplied categorical word exactly once", () => {
    const plan = forcedEosPlan();
    for (const word of [0, 1, 0x8000_0000, 0xffff_ffff]) {
      const compact = traceAceOpt0012CompactBrowserV1Sampling({
        plan,
        conditionalLogits: new Float32Array([3]),
        unconditionalLogits: new Float32Array([-2]),
        seenTokenIds: [ACE_QWEN_IM_END_TOKEN_ID],
        parameters: {
          temperature: 0.85,
          guidanceScale: 2,
          topK: 32,
          topP: 0.9,
          repetitionPenalty: 1,
        },
        word,
      });
      const full = fullVectorTrace(
        plan,
        new Float32Array([3]),
        new Float32Array([-2]),
        [ACE_QWEN_IM_END_TOKEN_ID],
        {
          temperature: 0.85,
          guidanceScale: 2,
          topK: 32,
          topP: 0.9,
          repetitionPenalty: 1,
        },
        word,
      );
      compareCompactToFull(plan, compact, full);
      expect(compact).toMatchObject({
        tokenId: ACE_QWEN_IM_END_TOKEN_ID,
        word: word >>> 0,
        positiveCandidateCount: 1,
        selectedCandidateIndex: 0,
      });
    }
  });

  it("keeps the compact and production Philox cursors bit-exact across code and EOS", () => {
    const regular = regularPlan();
    const eos = forcedEosPlan();
    const regularRows = compactRows(regular, 43);
    const eosRows = {
      conditional: new Float32Array([4]),
      unconditional: new Float32Array([1]),
    };
    const seed = canonicalizeSeed("0123456789abcdef");
    const compactCursor = new AceOpt0012CompactSamplingCursor(seed, 17n);
    const fullCursor = new AcePlannerSamplingCursor(seed, 17n);

    const compactCode = compactCursor.sample({
      plan: regular,
      conditionalLogits: regularRows.conditional,
      unconditionalLogits: regularRows.unconditional,
      seenTokenIds: [7, regular.firstCandidateTokenId + 4],
    });
    const fullCodeRows = fullRows(regular, regularRows.conditional,
      regularRows.unconditional);
    const fullCode = fullCursor.sample({
      conditionalLogits: fullCodeRows.conditional,
      unconditionalLogits: fullCodeRows.unconditional,
      seenTokenIds: [7, regular.firstCandidateTokenId + 4],
      preCfgAllowedTokens: ACE_OPT_0012_PRE_CFG_ALLOWED_TOKENS,
      allowedTokens: ACE_OPT_0012_REGULAR_ALLOWED_TOKENS,
      parameters: ACE_OPT_0012_SAMPLING_PARAMETERS,
    });
    expect(compactCode).toMatchObject({
      tokenId: fullCode.tokenId,
      word: fullCode.word,
      positiveCandidateCount: fullCode.positiveCandidateCount,
      drawIndex: fullCode.drawIndex,
      drawEnd: 18n,
    });

    const compactEos = compactCursor.sample({
      plan: eos,
      conditionalLogits: eosRows.conditional,
      unconditionalLogits: eosRows.unconditional,
      seenTokenIds: [compactCode.tokenId],
    });
    const fullEosRows = fullRows(eos, eosRows.conditional, eosRows.unconditional);
    const fullEos = fullCursor.sample({
      conditionalLogits: fullEosRows.conditional,
      unconditionalLogits: fullEosRows.unconditional,
      seenTokenIds: [compactCode.tokenId],
      preCfgAllowedTokens: ACE_OPT_0012_PRE_CFG_ALLOWED_TOKENS,
      allowedTokens: ACE_OPT_0012_FORCED_EOS_ALLOWED_TOKENS,
      parameters: ACE_OPT_0012_SAMPLING_PARAMETERS,
    });
    expect(compactEos).toMatchObject({
      tokenId: fullEos.tokenId,
      word: fullEos.word,
      positiveCandidateCount: fullEos.positiveCandidateCount,
      drawIndex: fullEos.drawIndex,
      drawEnd: 19n,
    });
    expect(compactCursor.consumed).toBe(19n);
    expect(fullCursor.consumed).toBe(19n);
  }, 20_000);

  it("rejects altered planned settings, malformed plans/logits/seen IDs without cursor advance", () => {
    const plan = regularPlan();
    const rows = compactRows(plan, 5);
    expect(() => sampleAceOpt0012PlannedCompactToken({
      plan,
      conditionalLogits: rows.conditional,
      unconditionalLogits: rows.unconditional,
      seenTokenIds: [],
      parameters: { ...ACE_OPT_0012_SAMPLING_PARAMETERS, topK: 1 },
      word: 0,
    })).toThrow(/settings/);
    expect(() => traceAceOpt0012CompactBrowserV1Sampling({
      plan,
      conditionalLogits: rows.conditional.subarray(1),
      unconditionalLogits: rows.unconditional,
      seenTokenIds: [],
      parameters: ACE_OPT_0012_SAMPLING_PARAMETERS,
      word: 0,
    })).toThrow(/64000 candidates/);
    expect(() => traceAceOpt0012CompactBrowserV1Sampling({
      plan,
      conditionalLogits: rows.conditional,
      unconditionalLogits: rows.unconditional,
      seenTokenIds: [plan.vocabularySize],
      parameters: ACE_OPT_0012_SAMPLING_PARAMETERS,
      word: 0,
    })).toThrow(/outside the vocabulary/);
    const nanRows = rows.conditional.slice();
    nanRows[0] = Number.NaN;
    expect(() => traceAceOpt0012CompactBrowserV1Sampling({
      plan,
      conditionalLogits: nanRows,
      unconditionalLogits: rows.unconditional,
      seenTokenIds: [],
      parameters: ACE_OPT_0012_SAMPLING_PARAMETERS,
      word: 0,
    })).toThrow(/finite or negative infinity/);
    const forged = { ...plan } as AceOpt0012CompactSemanticHeadPlan;
    expect(() => traceAceOpt0012CompactBrowserV1Sampling({
      plan: forged,
      conditionalLogits: rows.conditional,
      unconditionalLogits: rows.unconditional,
      seenTokenIds: [],
      parameters: ACE_OPT_0012_SAMPLING_PARAMETERS,
      word: 0,
    })).toThrow(/issued immutable/);

    const cursor = new AceOpt0012CompactSamplingCursor(canonicalizeSeed(9), 3n);
    expect(() => cursor.sample({
      plan,
      conditionalLogits: rows.conditional.subarray(1),
      unconditionalLogits: rows.unconditional,
      seenTokenIds: [],
    })).toThrow(/64000 candidates/);
    expect(cursor.consumed).toBe(3n);
  });
});

interface FullTrace {
  readonly cfg: Float32Array;
  readonly penalized: Float32Array;
  readonly topK: Float32Array;
  readonly topPKeep: Uint8Array;
  readonly topP: Float32Array;
  readonly scaled: Float32Array;
  readonly weights: Float32Array;
  readonly tokenId: number;
  readonly word: number;
  readonly positiveCandidateCount: number;
}

function regularPlan(): AceOpt0012CompactSemanticHeadPlan {
  return createAceOpt0012CompactSemanticHeadPlan(
    createAceOpt0012PlanRequest("regular-code"),
  );
}

function forcedEosPlan(): AceOpt0012CompactSemanticHeadPlan {
  return createAceOpt0012CompactSemanticHeadPlan(
    createAceOpt0012PlanRequest("forced-eos"),
  );
}

function expectedSourceShard(
  shardIndex: number,
  firstRow: number,
  rowCount: number,
  byteLength: number,
): Record<string, unknown> {
  const end = firstRow + rowCount;
  return {
    shardIndex,
    tensorName:
      `planner.model.embed_tokens.weight.rows-${String(firstRow).padStart(6, "0")}-` +
      String(end).padStart(6, "0"),
    firstRow,
    rowCount,
    bindingByteOffset: 0,
    bindingByteLength: byteLength,
    bufferByteLength: byteLength,
    manifestLayout: "row-shard-axis0",
    kernelWeightLayout: "source-row-major",
    dtype: "float16",
  };
}

function mappedFixture(
  plan: AceOpt0012CompactSemanticHeadPlan,
): Readonly<{ readonly buffer: ArrayBuffer }> {
  const buffer = new ArrayBuffer(plan.readback.allocationBytes);
  new Uint8Array(buffer).fill(0xa5);
  for (const span of plan.readback.logicalSpans) {
    const words = new Uint16Array(
      buffer,
      span.sourceByteOffset,
      span.candidateCount,
    );
    for (let index = 0; index < words.length; index += 1) {
      words[index] = fixtureBits(
        span.physicalRow,
        span.destinationCandidateOffset + index,
      );
    }
  }
  new Uint32Array(
    buffer,
    plan.readback.writeStatusByteOffset,
    2,
  ).set([1, 1]);
  return { buffer };
}

function exhaustiveMappedFixture(
  plan: AceOpt0012CompactSemanticHeadPlan,
): Readonly<{ readonly buffer: ArrayBuffer }> {
  const buffer = new ArrayBuffer(plan.readback.allocationBytes);
  new Uint8Array(buffer).fill(0xa5);
  for (const span of plan.readback.logicalSpans) {
    const words = new Uint16Array(
      buffer,
      span.sourceByteOffset,
      span.candidateCount,
    );
    for (let index = 0; index < words.length; index += 1) {
      words[index] = (
        span.physicalRow * plan.candidateCount +
        span.destinationCandidateOffset + index
      ) & 0xffff;
    }
  }
  new Uint32Array(
    buffer,
    plan.readback.writeStatusByteOffset,
    2,
  ).set([1, 1]);
  return { buffer };
}

function fixtureBits(row: number, candidate: number): number {
  return FIXTURE_FP16_BITS[
    (candidate * 5 + row * 3) % FIXTURE_FP16_BITS.length
  ]!;
}

function independentFixtureWord(row: number, candidate: number): number {
  return FIXTURE_FLOAT32_WORDS[
    (candidate * 5 + row * 3) % FIXTURE_FLOAT32_WORDS.length
  ]!;
}

function compactRows(
  plan: AceOpt0012CompactSemanticHeadPlan,
  salt: number,
): Readonly<{
  readonly conditional: Float32Array;
  readonly unconditional: Float32Array;
}> {
  const conditional = new Float32Array(plan.candidateCount);
  const unconditional = new Float32Array(plan.candidateCount);
  for (let index = 0; index < plan.candidateCount; index += 1) {
    conditional[index] = Math.fround(
      (((index * 73 + salt * 19) % 257) - 128) / 16,
    );
    unconditional[index] = Math.fround(
      (((index * 31 + salt * 7) % 193) - 96) / 24,
    );
  }
  return { conditional, unconditional };
}

function fullRows(
  plan: AceOpt0012CompactSemanticHeadPlan,
  conditionalCompact: Float32Array,
  unconditionalCompact: Float32Array,
): Readonly<{
  readonly conditional: Float32Array;
  readonly unconditional: Float32Array;
}> {
  const [conditional, unconditional] = reconstructAceOpt0012FullPlannerLogits({
    conditionalLogits: conditionalCompact,
    unconditionalLogits: unconditionalCompact,
    writeStatus: new Uint32Array([1, 1]),
  }, plan);
  return { conditional, unconditional };
}

function fullVectorTrace(
  plan: AceOpt0012CompactSemanticHeadPlan,
  conditionalCompact: Float32Array,
  unconditionalCompact: Float32Array,
  seenTokenIds: readonly number[],
  parameters: AcePlannerSamplingParameters,
  word: number,
): FullTrace {
  const rows = fullRows(plan, conditionalCompact, unconditionalCompact);
  const compactCfg = combineAcePlannerCfgLogits(
    conditionalCompact,
    unconditionalCompact,
    parameters.guidanceScale,
  );
  const cfg = new Float32Array(plan.vocabularySize);
  cfg.fill(Number.NEGATIVE_INFINITY);
  cfg.set(compactCfg, plan.firstCandidateTokenId);
  const penalized = applyAcePlannerRepetitionPenalty(
    cfg,
    seenTokenIds,
    parameters.repetitionPenalty,
  );
  const topK = applyAcePlannerTopK(penalized, parameters.topK);
  const topPKeep = createAcePlannerBrowserTopPKeep(topK, parameters.topP);
  const topP = topK.slice();
  for (let tokenId = 0; tokenId < topP.length; tokenId += 1) {
    if (topPKeep[tokenId] === 0) topP[tokenId] = Number.NEGATIVE_INFINITY;
  }
  const scaled = scaleTemperature(topP, parameters.temperature);
  const weights = createAcePlannerBrowserSamplingWeights(
    topP,
    parameters.temperature,
  );
  const filtered = createAcePlannerFilteredLogits({
    conditionalLogits: rows.conditional,
    unconditionalLogits: rows.unconditional,
    seenTokenIds,
    preCfgAllowedTokens: ACE_OPT_0012_PRE_CFG_ALLOWED_TOKENS,
    allowedTokens: plan.state === "regular-code"
      ? ACE_OPT_0012_REGULAR_ALLOWED_TOKENS
      : ACE_OPT_0012_FORCED_EOS_ALLOWED_TOKENS,
    parameters,
  });
  expect(floatWords(filtered)).toEqual(floatWords(topP));
  const sampled = sampleAcePlannerToken({
    conditionalLogits: rows.conditional,
    unconditionalLogits: rows.unconditional,
    seenTokenIds,
    preCfgAllowedTokens: ACE_OPT_0012_PRE_CFG_ALLOWED_TOKENS,
    allowedTokens: plan.state === "regular-code"
      ? ACE_OPT_0012_REGULAR_ALLOWED_TOKENS
      : ACE_OPT_0012_FORCED_EOS_ALLOWED_TOKENS,
    parameters,
    word,
  });
  expect(sampled.tokenId).toBe(aceCategoricalTokenFromWord(weights, word));
  return {
    cfg,
    penalized,
    topK,
    topPKeep,
    topP,
    scaled,
    weights,
    tokenId: sampled.tokenId,
    word: sampled.word,
    positiveCandidateCount: sampled.positiveCandidateCount,
  };
}

function compareCompactToFull(
  plan: AceOpt0012CompactSemanticHeadPlan,
  compact: AceOpt0012CompactSamplingTrace,
  full: FullTrace,
): void {
  const first = plan.firstCandidateTokenId;
  const end = first + plan.candidateCount;
  expect(floatWords(compact.cfgLogits)).toEqual(
    floatWords(full.cfg.subarray(first, end)),
  );
  expect(floatWords(compact.finalMaskedLogits)).toEqual(
    floatWords(full.cfg.subarray(first, end)),
  );
  expect(floatWords(compact.repetitionPenalizedLogits)).toEqual(
    floatWords(full.penalized.subarray(first, end)),
  );
  expect(floatWords(compact.topKLogits)).toEqual(
    floatWords(full.topK.subarray(first, end)),
  );
  expect(compact.topKGlobalTokenIds).toEqual(finiteTokenIds(full.topK));
  expect(compact.topPKeep).toEqual(full.topPKeep.slice(first, end));
  expect(compact.topPGlobalTokenIds).toEqual(finiteTokenIds(full.topP));
  expect(floatWords(compact.topPLogits)).toEqual(
    floatWords(full.topP.subarray(first, end)),
  );
  expect(floatWords(compact.temperatureScaledLogits)).toEqual(
    floatWords(full.scaled.subarray(first, end)),
  );
  expect(floatWords(compact.weights)).toEqual(
    floatWords(full.weights.subarray(first, end)),
  );
  expect(compact).toMatchObject({
    tokenId: full.tokenId,
    word: full.word,
    positiveCandidateCount: full.positiveCandidateCount,
  });
  expect(compact.tokenId).toBe(
    plan.firstCandidateTokenId + compact.selectedCandidateIndex,
  );
}

function scaleTemperature(
  logits: Float32Array,
  temperature: number,
): Float32Array {
  const result = logits.slice();
  const rounded = Math.fround(temperature);
  for (let index = 0; index < result.length; index += 1) {
    if (result[index] !== Number.NEGATIVE_INFINITY) {
      result[index] = Math.fround(result[index]! / rounded);
    }
  }
  return result;
}

function finiteTokenIds(values: ArrayLike<number>): number[] {
  const result: number[] = [];
  for (let index = 0; index < values.length; index += 1) {
    if (Number.isFinite(values[index])) result.push(index);
  }
  return result;
}

function floatWord(value: number): number {
  return new Uint32Array(new Float32Array([value]).buffer)[0]!;
}

/** Independent canonical binary16 expansion used only by the exhaustive test. */
function independentCanonicalFloat16Word(bits: number): number {
  const sign = (bits & 0x8000) << 16;
  let exponent = (bits >>> 10) & 0x1f;
  let mantissa = bits & 0x03ff;
  let output: number;
  if (exponent === 0) {
    if (mantissa === 0) {
      output = sign;
    } else {
      exponent = 1;
      while ((mantissa & 0x0400) === 0) {
        mantissa <<= 1;
        exponent -= 1;
      }
      mantissa &= 0x03ff;
      output = sign | ((exponent + 112) << 23) | (mantissa << 13);
    }
  } else if (exponent === 0x1f) {
    output = sign | 0x7f80_0000 | (mantissa << 13);
    if (mantissa !== 0) output |= 0x0040_0000;
  } else {
    output = sign | ((exponent + 112) << 23) | (mantissa << 13);
  }
  return output >>> 0;
}

function floatWords(values: ArrayLike<number>): number[] {
  const floats = new Float32Array(values.length);
  for (let index = 0; index < values.length; index += 1) {
    floats[index] = values[index]!;
  }
  return [...new Uint32Array(floats.buffer)];
}

function sha256LittleEndianWords(words: Uint32Array): string {
  const bytes = new Uint8Array(words.length * Uint32Array.BYTES_PER_ELEMENT);
  const view = new DataView(bytes.buffer);
  for (let index = 0; index < words.length; index += 1) {
    view.setUint32(index * Uint32Array.BYTES_PER_ELEMENT, words[index]!, true);
  }
  return createHash("sha256").update(bytes).digest("hex");
}
