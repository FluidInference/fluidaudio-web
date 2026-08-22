import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";
import {
  aceCorrectnessGemmWgsl,
  planAceCompositeCooperativeQuantumCount,
  planAceTiledGemm,
} from "../src/webgpu/kernels/gemm.js";

import {
  OPT_0012_ALLOCATION_COMMIT,
  OPT_0012_BALANCED_ORDERS,
  OPT_0012_CASE_SPECS,
  OPT_0012_CONVERSION_BALANCED_ORDERS,
  OPT_0012_CORE_COMMIT,
  OPT_0012_CORE_SOURCE_SHA256,
  OPT_0012_CANDIDATE_SHADER_SHA256,
  OPT_0012_CANDIDATE_PREFILL_COMMAND_BUFFER_COUNT,
  OPT_0012_FP16_CANONICAL_NAN_OUTPUT_SHA256,
  OPT_0012_FP16_DOMAIN_OUTPUT_SHA256,
  OPT_0012_FP16_DOMAIN_WORD_COUNT,
  OPT_0012_FP16_NAN_WORD_COUNT,
  OPT_0012_FP16_NON_NAN_OUTPUT_SHA256,
  OPT_0012_FP16_NON_NAN_WORD_COUNT,
  OPT_0012_FP16_MANIFEST_SHA256,
  OPT_0012_EOS_CASE_SPEC,
  OPT_0012_FULL_HEAD_LOGIT_BYTES,
  OPT_0012_FULL_PREFILL_COMMAND_BUFFER_COUNT,
  OPT_0012_FULL_READBACK_ALLOCATION_BYTES,
  OPT_0012_FULL_READBACK_STATUS_BYTE_OFFSET,
  OPT_0012_FULL_READBACK_STATUS_BYTES,
  OPT_0012_FULL_READBACK_USED_BYTES,
  OPT_0012_MINIMUM_NOMINAL_MILLISECONDS,
  OPT_0012_OPT10_SAMPLE_AUTHORITIES,
  OPT_0012_PREPACKAGE_CONVERSION_ORDERS,
  OPT_0012_PRIMARY_TIMING_FP16_NAN_CENSUS_COUNT,
  OPT_0012_SOURCE_IDENTITIES,
  OPT_0012_THERMAL_POLL_MILLISECONDS,
  OPT_0012_THERMAL_SOURCE,
  OPT_0012_RAW_RESULT_CHUNK_CODE_UNITS,
  OPT_0012_REPLAY_BALANCED_ORDERS,
  authenticateOpt0012ObservedCopies,
  authenticateOpt0012ObservedHeadDispatches,
  authenticateOpt0012ProductionHeadTail,
  opt0012ExpectedCommandBufferTopology,
  parseOpt0012RunIdentity,
  parseOpt0012ThermalGateMetadata,
  runOpt0012Fp16ConversionCorrectnessGate,
  parseOpt0012RawResultChunkOffset,
  sliceOpt0012RawResultChunk,
  validateOpt0012TrackedBufferCleanup,
  validateOpt0012PrimaryTimingFp16NaNCensusCoverage,
} from "./browser/opt-0012-compact-semantic-head-ab.js";

const PAGE_SOURCE = readFileSync(new URL(
  "./browser/opt-0012-compact-semantic-head-ab.ts",
  import.meta.url,
), "utf8");
const HTML_SOURCE = readFileSync(new URL(
  "./browser/opt-0012-compact-semantic-head-ab.html",
  import.meta.url,
), "utf8");

describe("OPT-0012 compact semantic head browser A/B/C contract", () => {
  it("keeps the complete raw receipt out of the rendered DOM", () => {
    expect(PAGE_SOURCE).toContain(
      'const OPT_0012_RAW_RESULT_GLOBAL = "__ACE_OPT_0012_RAW_RESULT_JSON__";',
    );
    expect(PAGE_SOURCE).toContain(
      "Reflect.defineProperty(globalThis, OPT_0012_RAW_RESULT_GLOBAL",
    );
    expect(PAGE_SOURCE).toContain("fullReceiptIntentionallyKeptOutOfDom: true");
    expect(PAGE_SOURCE).toContain("rawResultJsonCodeUnitLength: rawResultJson.length");
    expect(PAGE_SOURCE).toContain('rawResultRetrieval: "bounded-dom-chunks"');
    expect(PAGE_SOURCE).toContain(
      "rawResultChunkCodeUnitLimit: OPT_0012_RAW_RESULT_CHUNK_CODE_UNITS",
    );
    expect(HTML_SOURCE).toContain('id="raw-result-retrieval" hidden disabled');
    expect(HTML_SOURCE).toContain(
      'id="raw-result-chunk" data-state="empty" hidden aria-hidden="true"',
    );
    expect(PAGE_SOURCE).toContain("output.textContent = slice.chunk");
    expect(PAGE_SOURCE).toContain(
      "output.dataset.publicationSequence = String(++publicationSequence)",
    );
    expect(PAGE_SOURCE).toContain("delete output.dataset.startOffset");
    expect(PAGE_SOURCE).toContain("output.dataset.endOffsetExclusive = String(slice.end)");
    expect(PAGE_SOURCE).toContain(
      "output.dataset.chunkCodeUnitLength = String(slice.chunk.length)",
    );
    expect(PAGE_SOURCE).not.toContain("innerHTML");
    expect(PAGE_SOURCE).not.toContain(
      'requireElement<HTMLElement>("#result").textContent = JSON.stringify({\n        ...message.result',
    );
  });

  it("publishes bounded contiguous raw-result chunks without splitting surrogate pairs", () => {
    expect(OPT_0012_RAW_RESULT_CHUNK_CODE_UNITS).toBe(100_000);
    const prefix = "a".repeat(OPT_0012_RAW_RESULT_CHUNK_CODE_UNITS - 1);
    const raw = `${prefix}\ud83d\ude80tail`;
    const first = sliceOpt0012RawResultChunk(raw, 0);
    expect(first.start).toBe(0);
    expect(first.end).toBe(OPT_0012_RAW_RESULT_CHUNK_CODE_UNITS - 1);
    expect(first.chunk).toBe(prefix);
    expect(first.complete).toBe(false);
    const second = sliceOpt0012RawResultChunk(raw, first.nextOffset);
    expect(second.chunk).toBe("🚀tail");
    expect(second.end).toBe(raw.length);
    expect(second.complete).toBe(true);
    expect(first.chunk + second.chunk).toBe(raw);
    expect(sliceOpt0012RawResultChunk(raw, raw.length)).toEqual({
      chunk: "",
      start: raw.length,
      end: raw.length,
      nextOffset: raw.length,
      totalCodeUnits: raw.length,
      complete: true,
    });
    expect(() => sliceOpt0012RawResultChunk(raw, -1)).toThrow(/offset/u);
    expect(() => sliceOpt0012RawResultChunk(raw, raw.length + 1)).toThrow(/offset/u);
    expect(() => sliceOpt0012RawResultChunk(raw, 0.5)).toThrow(/offset/u);
    expect(() => sliceOpt0012RawResultChunk(raw, prefix.length + 1)).toThrow(
      /surrogate/u,
    );
    expect(parseOpt0012RawResultChunkOffset("0")).toBe(0);
    expect(parseOpt0012RawResultChunkOffset("9007199254740991")).toBe(
      Number.MAX_SAFE_INTEGER,
    );
    for (const value of ["", " ", "-1", "+1", "01", "1.0", "1e3", "9007199254740992"]) {
      expect(() => parseOpt0012RawResultChunkOffset(value)).toThrow(/offset/u);
    }
  });

  it("pins the three generated production-GEMM candidate shaders", () => {
    const hashes = [44_939, 19_061, 1].map((columns) => createHash("sha256")
      .update(aceCorrectnessGemmWgsl("raw-fp16", {
        rows: 2,
        inner: 1_024,
        columns,
      }, false, "source-row-major"))
      .digest("hex"));
    expect(hashes).toEqual([
      OPT_0012_CANDIDATE_SHADER_SHA256["regular-shard-3"],
      OPT_0012_CANDIDATE_SHADER_SHA256["regular-shard-4"],
      OPT_0012_CANDIDATE_SHADER_SHA256["forced-eos-shard-3"],
    ]);
  });

  it("pins the allocated and committed core identities", () => {
    expect(OPT_0012_ALLOCATION_COMMIT).toBe(
      "f5e8e5db0b88a9a44dc96b73319183114daf136a",
    );
    expect(OPT_0012_CORE_COMMIT).toBe(
      "f73380bceebdd5568d93908a67ff33cea2b7d8f0",
    );
    expect(OPT_0012_FP16_MANIFEST_SHA256).toBe(
      "c5b547cd08aa5e6d2971b2c9c84940b8af193f2e230ce689258ca81fcd292a3b",
    );
    expect(createHash("sha256").update(readFileSync(new URL(
      "../benchmark/opt-0012-compact-semantic-head.ts",
      import.meta.url,
    ))).digest("hex")).toBe(OPT_0012_CORE_SOURCE_SHA256);
  });

  it("requires complete run identity and rejects a different core", () => {
    const valid = new URLSearchParams({
      harnessCommit: "0123456789abcdef0123456789abcdef01234567",
      coreCommit: OPT_0012_CORE_COMMIT,
      allocationCommit: OPT_0012_ALLOCATION_COMMIT,
      machineModel: "Mac15,12",
      osVersion: "26.5.2",
      osBuild: "25F84",
      browserVersion: "Chrome 151.0.7922.138",
      gpuCoreCount: "10",
      memoryBytes: "17179869184",
    });
    expect(parseOpt0012RunIdentity(valid)).toEqual({
      harnessCommit: "0123456789abcdef0123456789abcdef01234567",
      coreCommit: OPT_0012_CORE_COMMIT,
      allocationCommit: OPT_0012_ALLOCATION_COMMIT,
      machineModel: "Mac15,12",
      osVersion: "26.5.2",
      osBuild: "25F84",
      browserVersion: "Chrome 151.0.7922.138",
      gpuCoreCount: 10,
      memoryBytes: 17_179_869_184,
    });
    const wrong = new URLSearchParams(valid);
    wrong.set("coreCommit", "0123456789abcdef0123456789abcdef01234567");
    expect(() => parseOpt0012RunIdentity(wrong)).toThrow(/coreCommit/);
  });

  it("requires a post-correctness continuous nominal thermal interval", () => {
    const warmup = 1_000_000;
    const start = warmup + 1;
    const end = start + OPT_0012_MINIMUM_NOMINAL_MILLISECONDS;
    const valid = new URLSearchParams({
      thermalSource: OPT_0012_THERMAL_SOURCE,
      thermalStartedAtEpochMilliseconds: String(start),
      thermalCompletedAtEpochMilliseconds: String(end),
      thermalObservations: "31",
      thermalPollMilliseconds: String(OPT_0012_THERMAL_POLL_MILLISECONDS),
      thermalMaximumPollGapMilliseconds: "1_000".replace("_", ""),
      thermalNonNominalObservations: "0",
    });
    expect(parseOpt0012ThermalGateMetadata(valid, warmup, end + 1)).toMatchObject({
      source: OPT_0012_THERMAL_SOURCE,
      durationMilliseconds: 30_000,
      observationCount: 31,
      nonNominalObservationCount: 0,
    });
    const before = new URLSearchParams(valid);
    before.set("thermalStartedAtEpochMilliseconds", String(warmup - 1));
    expect(() => parseOpt0012ThermalGateMetadata(before, warmup, end + 1))
      .toThrow(/after correctness\/warmup/);
  });

  it("freezes exact M2 short/mid/long cases and all balanced permutations", () => {
    expect(OPT_0012_CASE_SPECS).toEqual([
      {
        id: "semantic-m2-short", position: "short",
        cachedTokensBeforeAppend: 268, cacheCapacity: 768, drawIndex: 125,
      },
      {
        id: "semantic-m2-mid", position: "mid",
        cachedTokensBeforeAppend: 328, cacheCapacity: 1_280, drawIndex: 185,
      },
      {
        id: "semantic-m2-long", position: "long",
        cachedTokensBeforeAppend: 401, cacheCapacity: 2_048, drawIndex: 258,
      },
    ]);
    expect(OPT_0012_BALANCED_ORDERS).toEqual([
      ["A", "B", "C"], ["A", "C", "B"], ["B", "A", "C"],
      ["B", "C", "A"], ["C", "A", "B"], ["C", "B", "A"],
    ]);
    for (const arm of ["A", "B", "C"] as const) {
      expect(OPT_0012_BALANCED_ORDERS.flat().filter((value) => value === arm))
        .toHaveLength(6);
      for (let position = 0; position < 3; position += 1) {
        expect(OPT_0012_BALANCED_ORDERS.filter(
          (order) => order[position] === arm,
        )).toHaveLength(2);
      }
    }
  });

  it("freezes both six-pair corrective orders at three visits per position", () => {
    expect(OPT_0012_PREPACKAGE_CONVERSION_ORDERS).toEqual([
      ["legacy-allocating", "allocation-free"],
      ["allocation-free", "legacy-allocating"],
    ]);
    expect(OPT_0012_REPLAY_BALANCED_ORDERS).toEqual([
      ["B", "C"], ["C", "B"], ["C", "B"],
      ["B", "C"], ["B", "C"], ["C", "B"],
    ]);
    expect(OPT_0012_CONVERSION_BALANCED_ORDERS).toEqual([
      ["legacy-allocating", "allocation-free"],
      ["allocation-free", "legacy-allocating"],
      ["allocation-free", "legacy-allocating"],
      ["legacy-allocating", "allocation-free"],
      ["legacy-allocating", "allocation-free"],
      ["allocation-free", "legacy-allocating"],
    ]);
    for (const [orders, arms] of [
      [OPT_0012_REPLAY_BALANCED_ORDERS, ["B", "C"]],
      [OPT_0012_CONVERSION_BALANCED_ORDERS,
        ["legacy-allocating", "allocation-free"]],
    ] as const) {
      expect(orders).toHaveLength(6);
      for (const arm of arms) {
        expect(orders.flat().filter((value) => value === arm)).toHaveLength(6);
        for (let position = 0; position < 2; position += 1) {
          expect(orders.filter((order) => order[position] === arm)).toHaveLength(3);
        }
      }
    }
  });

  it("embeds the fixed two-pass stable target-browser NaN-envelope gate", () => {
    expect(OPT_0012_FP16_DOMAIN_WORD_COUNT).toBe(65_536);
    expect(OPT_0012_FP16_NON_NAN_WORD_COUNT).toBe(63_490);
    expect(OPT_0012_FP16_NAN_WORD_COUNT).toBe(2_046);
    expect(OPT_0012_FP16_DOMAIN_OUTPUT_SHA256).toBe(
      "b636c5716ff84d972782faf02d0194cb8951526bea4cc487082feb47b1860ddf",
    );
    expect(OPT_0012_FP16_NON_NAN_OUTPUT_SHA256).toBe(
      "680bbc22915f61aa1bbfc7265bc3882a6aa42d299bfd2c571807196e5544de2e",
    );
    expect(OPT_0012_FP16_CANONICAL_NAN_OUTPUT_SHA256).toBe(
      "32f37d24c421f50695da47516d20cffa27c96fb2be53d1ac6f88cfbc1cec9039",
    );
    expect(typeof runOpt0012Fp16ConversionCorrectnessGate).toBe("function");
    const receipt = runOpt0012Fp16ConversionCorrectnessGate();
    expect(receipt).toMatchObject({
      authority: "target-browser-stable-FP16-NaN-envelope",
      protocol: {
        orders: OPT_0012_PREPACKAGE_CONVERSION_ORDERS,
        passCount: 2,
        exactlyOnePassPerOrder: true,
        noAdaptiveRepetition: true,
      },
      candidateCrossPass: {
        comparedFp16U16Count: 65_536,
        mismatchCount: 0,
        firstMismatchFp16U16: null,
        firstOutputU32LeSha256: OPT_0012_FP16_DOMAIN_OUTPUT_SHA256,
        secondOutputU32LeSha256: OPT_0012_FP16_DOMAIN_OUTPUT_SHA256,
        wordForWordIdentical: true,
      },
      acceptedStableNanEnvelope: true,
    });
    const passes = receipt["passes"] as readonly Readonly<
      Record<string, unknown>
    >[];
    expect(passes).toHaveLength(2);
    expect(passes.map((pass) => pass["order"]))
      .toEqual(OPT_0012_PREPACKAGE_CONVERSION_ORDERS);
    for (const pass of passes) {
      expect(pass["acceptedStableNanEnvelope"]).toBe(true);
      const legacy = pass["legacy"] as Readonly<Record<string, unknown>>;
      const candidate = pass["candidate"] as Readonly<Record<string, unknown>>;
      for (const arm of [legacy, candidate]) {
        expect(arm).toMatchObject({
          canonicalOutputU32LeSha256: OPT_0012_FP16_DOMAIN_OUTPUT_SHA256,
          nonNaNOutputU32LeSha256: OPT_0012_FP16_NON_NAN_OUTPUT_SHA256,
          canonicalNaNOutputU32LeSha256:
            OPT_0012_FP16_CANONICAL_NAN_OUTPUT_SHA256,
          nonNaNMismatchCount: 0,
          nanClassificationMismatchCount: 0,
          signMismatchCount: 0,
          payloadExcludingQuietMismatchCount: 0,
          disallowedWordMismatchCount: 0,
          everyNonNaNExact: true,
          everyNaNAllowedRawOrQuietBitOnly: true,
          acceptedStableNanEnvelope: true,
        });
      }
      expect(legacy["legacyRawHashIsDiagnosticOnly"]).toBe(true);
      expect(candidate).toMatchObject({
        rawOutputU32LeSha256: OPT_0012_FP16_DOMAIN_OUTPUT_SHA256,
        signalingOutputCount: 0,
        quietOutputCount: 2_046,
        candidateDeterministicQuietAll: true,
      });
    }
    const gate = PAGE_SOURCE.slice(
      PAGE_SOURCE.indexOf(
        "export function runOpt0012Fp16ConversionCorrectnessGate()",
      ),
      PAGE_SOURCE.indexOf("function writeOpt0012Fp16ConversionDomain("),
    );
    expect(gate).toContain("runOpt0012Fp16ConversionPass(order, passIndex)");
    expect(gate).toContain(
      "for (let bits = 0; bits < OPT_0012_FP16_DOMAIN_WORD_COUNT; bits += 1)",
    );
    expect(gate).toContain("legacyWords[bits] !== candidateWords[bits]");
    expect(gate).toContain("payloadExcludingQuietMismatchCount");
    expect(gate).toContain("disallowedWordMismatchCount");
    expect(gate).toContain("nanOutputClassesByInputClassAndSign");
    expect(gate).toContain("legacyRawHashIsDiagnosticOnly");
    expect(gate).toContain("candidateCrossPassMismatchCount");
    const initialize = PAGE_SOURCE.slice(
      PAGE_SOURCE.indexOf("async function initializeSession("),
      PAGE_SOURCE.indexOf("async function preparePackage("),
    );
    const gateAt = initialize.indexOf(
      "runOpt0012Fp16ConversionCorrectnessGate()",
    );
    const prepareAt = initialize.indexOf("await preparePackage()");
    expect(gateAt).toBeGreaterThan(0);
    expect(prepareAt).toBeGreaterThan(gateAt);
  });

  it("pins raw-FP16 OPT-0010 samples and a true post-150 EOS fixture", () => {
    expect(OPT_0012_OPT10_SAMPLE_AUTHORITIES).toEqual({
      "semantic-m2-short": {
        tokenId: 192_370, word: 2_004_582_350, positiveCandidateCount: 16,
        drawIndex: "125", drawEnd: "126",
      },
      "semantic-m2-mid": {
        tokenId: 156_326, word: 503_673_048, positiveCandidateCount: 1,
        drawIndex: "185", drawEnd: "186",
      },
      "semantic-m2-long": {
        tokenId: 155_832, word: 3_288_166_745, positiveCandidateCount: 1,
        drawIndex: "258", drawEnd: "259",
      },
    });
    expect(OPT_0012_EOS_CASE_SPEC).toEqual({
      id: "semantic-m2-terminal-eos",
      position: "long",
      cachedTokensBeforeAppend: 402,
      cacheCapacity: 2_048,
      drawIndex: 258,
    });
  });

  it("pins the non-contiguous full readback payload and status layout", () => {
    expect({
      rawLogitEnd: OPT_0012_FULL_HEAD_LOGIT_BYTES,
      statusOffset: OPT_0012_FULL_READBACK_STATUS_BYTE_OFFSET,
      statusBytes: OPT_0012_FULL_READBACK_STATUS_BYTES,
      usedEnd: OPT_0012_FULL_READBACK_USED_BYTES,
      allocation: OPT_0012_FULL_READBACK_ALLOCATION_BYTES,
    }).toEqual({
      rawLogitEnd: 868_816,
      statusOffset: 868_864,
      statusBytes: 8,
      usedEnd: 868_872,
      allocation: 869_120,
    });
  });

  it("fails closed unless the production tail is exactly two tied-head quanta", () => {
    const label = "ace-planner-decode-2x1-capacity-768";
    const logicalId = `${label}-tied-lm-head`;
    const prefix = Array.from({ length: 31 }, (_, index) => ({
      id: `${label}-prefix-${index}`,
      logicalId: `${label}-prefix-${index}`,
      kind: index === 30 ? "last-row-gather" : "layer",
      layer: index < 28 ? index : null,
      primitiveCount: 1,
    }));
    const quanta = [
      ...prefix,
      { id: `${logicalId}-part-0`, logicalId, kind: "tied-lm-head",
        layer: null, primitiveCount: 2 },
      { id: `${logicalId}-part-1`, logicalId, kind: "tied-lm-head",
        layer: null, primitiveCount: 3 },
    ];
    expect(authenticateOpt0012ProductionHeadTail(label, quanta, "decode")).toBe(31);
    expect(() => authenticateOpt0012ProductionHeadTail(
      label,
      quanta.slice(1),
      "decode",
    ))
      .toThrow(/two-quantum/);
    expect(() => authenticateOpt0012ProductionHeadTail(label, [
      ...prefix,
      quanta[32]!,
      quanta[31]!,
    ], "decode")).toThrow(/two-quantum/);
    expect(() => authenticateOpt0012ProductionHeadTail(label, [
      { ...prefix[0]!, kind: "tied-lm-head" },
      ...prefix.slice(1),
      ...quanta.slice(31),
    ], "decode")).toThrow(/two-quantum/);
  });

  it("pins the expanded 253-token prefill separately from every decode", () => {
    const rows = 2 * 253;
    const layerGemmPlans = [
      planAceTiledGemm({ rows, inner: 1_024, columns: 2_048 }),
      planAceTiledGemm({ rows, inner: 1_024, columns: 1_024 }),
      planAceTiledGemm({ rows, inner: 1_024, columns: 1_024 }),
      planAceTiledGemm({ rows, inner: 2_048, columns: 1_024 }),
      planAceTiledGemm({ rows, inner: 1_024, columns: 3_072 }),
      planAceTiledGemm({ rows, inner: 1_024, columns: 3_072 }),
      planAceTiledGemm({ rows, inner: 3_072, columns: 1_024 }),
    ];
    expect(planAceCompositeCooperativeQuantumCount(layerGemmPlans)).toBe(5);
    expect(1 + 28 * 5 + 1 + 1 + 2).toBe(145);
    expect(opt0012ExpectedCommandBufferTopology("A", "prefill")).toEqual({
      preHeadCommandBufferCount: 143,
      modelCommandBufferCount: 145,
      totalCommandBufferCount: 146,
    });
    expect(opt0012ExpectedCommandBufferTopology("B", "prefill")).toEqual({
      preHeadCommandBufferCount: 143,
      modelCommandBufferCount: 144,
      totalCommandBufferCount: 145,
    });
    expect(opt0012ExpectedCommandBufferTopology("C", "decode")).toEqual({
      preHeadCommandBufferCount: 31,
      modelCommandBufferCount: 32,
      totalCommandBufferCount: 33,
    });
    expect(OPT_0012_FULL_PREFILL_COMMAND_BUFFER_COUNT).toBe(146);
    expect(OPT_0012_CANDIDATE_PREFILL_COMMAND_BUFFER_COUNT).toBe(145);

    const label = "ace-planner-prefill-2x253-capacity-768";
    const logicalId = `${label}-tied-lm-head`;
    const preHead = Array.from({ length: 143 }, (_, index) => ({
      id: `${label}-prefix-${index}`,
      logicalId: `${label}-prefix-${index}`,
      kind: "layer",
      layer: Math.min(27, Math.floor(index / 5)),
      primitiveCount: 1,
    }));
    const quanta = [
      ...preHead,
      { id: `${logicalId}-part-0`, logicalId, kind: "tied-lm-head",
        layer: null, primitiveCount: 2 },
      { id: `${logicalId}-part-1`, logicalId, kind: "tied-lm-head",
        layer: null, primitiveCount: 3 },
    ];
    expect(authenticateOpt0012ProductionHeadTail(label, quanta, "prefill"))
      .toBe(143);
    expect(() => authenticateOpt0012ProductionHeadTail(label, quanta, "decode"))
      .toThrow(/two-quantum/);
  });

  it("authenticates exact A/regular/EOS head shard order and workgroups", () => {
    const dispatch = (columns: number, label: string) => ({
      pipelineLabel:
        `ace-correctness-gemm-raw-fp16-source-row-major-2x1024x${columns}-no-bias`,
      bindGroupLabel: label,
      workgroups: [Math.ceil(columns / 128), 1, 1] as const,
    });
    const fullRows = [49_152, 49_152, 49_152, 49_152, 20_596];
    const firstRows = [0, 49_152, 98_304, 147_456, 196_608];
    const full = fullRows.map((columns, index) => dispatch(
      columns,
      `x-lm-head-rows-${firstRows[index]}-source-row-major-range-0-bindings`,
    ));
    expect(() => authenticateOpt0012ObservedHeadDispatches(
      "A", "regular-code", full,
    )).not.toThrow();
    const regular = [
      dispatch(44_939,
        "opt-0012-regular-code-shard-3-source-row-major-range-0-bindings"),
      dispatch(19_061,
        "opt-0012-regular-code-shard-4-source-row-major-range-0-bindings"),
    ];
    expect(() => authenticateOpt0012ObservedHeadDispatches(
      "B", "regular-code", regular,
    )).not.toThrow();
    expect(() => authenticateOpt0012ObservedHeadDispatches(
      "C", "regular-code", [...regular].reverse(),
    )).toThrow(/shard/);
    expect(() => authenticateOpt0012ObservedHeadDispatches(
      "C", "forced-eos", [dispatch(1,
        "opt-0012-forced-eos-shard-3-source-row-major-range-0-bindings")],
    )).not.toThrow();
    expect(() => authenticateOpt0012ObservedHeadDispatches(
      "A", "regular-code", [{ ...full[0]!, workgroups: [385, 1, 1] }, ...full.slice(1)],
    )).toThrow(/identity\/work/);
  });

  it("authenticates every full and compact copy offset, label, and byte count", () => {
    const fullBytes = [196_608, 196_608, 196_608, 196_608, 82_384];
    const fullOffsets = [0, 196_608, 393_216, 589_824, 786_432];
    const full = [
      ...fullBytes.map((copiedBytes, index) => ({
        sourceBufferLabel: `logits-${index}`,
        sourceOffset: 0,
        destinationBufferLabel: "ace-planner-logit-readback",
        destinationOffset: fullOffsets[index]!,
        copiedBytes,
      })),
      { sourceBufferLabel: "write-status", sourceOffset: 0,
        destinationBufferLabel: "ace-planner-logit-readback",
        destinationOffset: 868_864, copiedBytes: 8 },
    ];
    expect(authenticateOpt0012ObservedCopies("A", "regular-code", full))
      .toBe(868_824);
    const regular = [
      { sourceBufferLabel: "opt-0012-regular-code-shard-3-output",
        sourceOffset: 256, destinationBufferLabel: "opt-0012-compact-head-readback",
        destinationOffset: 0, copiedBytes: 179_756 },
      { sourceBufferLabel: "opt-0012-regular-code-shard-4-output",
        sourceOffset: 256, destinationBufferLabel: "opt-0012-compact-head-readback",
        destinationOffset: 179_756, copiedBytes: 76_244 },
      { sourceBufferLabel: "write-status", sourceOffset: 0,
        destinationBufferLabel: "opt-0012-compact-head-readback",
        destinationOffset: 256_000, copiedBytes: 8 },
    ];
    expect(authenticateOpt0012ObservedCopies("B", "regular-code", regular))
      .toBe(256_008);
    expect(() => authenticateOpt0012ObservedCopies("C", "regular-code", [
      { ...regular[0]!, sourceOffset: 0 }, ...regular.slice(1),
    ])).toThrow(/copy 0/);
    expect(authenticateOpt0012ObservedCopies("C", "forced-eos", [
      { sourceBufferLabel: "opt-0012-forced-eos-shard-3-output",
        sourceOffset: 256, destinationBufferLabel: "opt-0012-compact-head-readback",
        destinationOffset: 0, copiedBytes: 4 },
      { sourceBufferLabel: "write-status", sourceOffset: 0,
        destinationBufferLabel: "opt-0012-compact-head-readback",
        destinationOffset: 4, copiedBytes: 8 },
    ])).toBe(12);
  });

  it("requires exact once-only destruction and balanced mapping", () => {
    const good = [{ label: "buffer", destroyCallCount: 1, mapCallCount: 2,
      unmapCallCount: 2, destroyed: true, mapped: false }];
    const supported = {
      destructionTrackingSupported: true,
      mapTrackingSupported: true,
    };
    expect(validateOpt0012TrackedBufferCleanup(good, supported)).toMatchObject({
      zeroLiveOwnedResources: true,
      everyTrackedBufferDestroyedExactlyOnce: true,
      everyMapBalancedByUnmap: true,
    });
    expect(() => validateOpt0012TrackedBufferCleanup(
      [{ ...good[0]!, destroyCallCount: 2 }], supported,
    )).toThrow(/buffer/);
    expect(() => validateOpt0012TrackedBufferCleanup(
      [{ ...good[0]!, mapped: true }], supported,
    )).toThrow(/mapped=true/);
    expect(() => validateOpt0012TrackedBufferCleanup(
      [{ ...good[0]!, unmapCallCount: 1 }], supported,
    )).toThrow(/unmaps=1/);
    expect(() => validateOpt0012TrackedBufferCleanup(good, {
      ...supported, mapTrackingSupported: false,
    })).toThrow(/unsupported/);
  });

  it("authenticates the complete executed source inventory against disk", () => {
    for (const [path, expected] of Object.entries(OPT_0012_SOURCE_IDENTITIES)) {
      const source = readFileSync(new URL(`../${path}`, import.meta.url));
      expect(createHash("sha256").update(source).digest("hex"), path).toBe(expected);
    }
    expect(OPT_0012_SOURCE_IDENTITIES).toMatchObject({
      "src/model/graph-contract.ts": expect.any(String),
      "src/model/strict-json.ts": expect.any(String),
      "src/tokenizer/loader.ts": expect.any(String),
      "src/tokenizer/chat.ts": expect.any(String),
      "src/tokenizer/conditioning-text.ts": expect.any(String),
      "src/webgpu/capabilities.ts": expect.any(String),
      "src/webgpu/kernels/attention.ts": expect.any(String),
      "src/webgpu/kernels/batched-rope.ts": expect.any(String),
      "src/webgpu/kernels/embedding.ts": expect.any(String),
      "src/webgpu/kernels/kv-cache.ts": expect.any(String),
      "src/webgpu/kernels/rmsnorm.ts": expect.any(String),
      "src/webgpu/kernels/tensor-copy.ts": expect.any(String),
      "src/webgpu/kernels/transformer-plumbing.ts": expect.any(String),
    });
  });

  it("prepares every timed candidate allocation and dispatch before the wall", () => {
    const execute = PAGE_SOURCE.slice(
      PAGE_SOURCE.indexOf("async function executePackageArm("),
      PAGE_SOURCE.indexOf("function sampleFullRows("),
    );
    const prepareAt = execute.indexOf(
      "prepareCandidateResourcesOutsidePrimaryWall(plan, sentinel)",
    );
    const traceAt = execute.indexOf("observer.beginTrace(");
    const wallAt = execute.indexOf("const wallStarted = performance.now()");
    expect(prepareAt).toBeGreaterThan(0);
    expect(traceAt).toBeGreaterThan(prepareAt);
    expect(wallAt).toBeGreaterThan(traceAt);

    const runCandidate = PAGE_SOURCE.slice(
      PAGE_SOURCE.indexOf("private async runCandidate("),
      PAGE_SOURCE.indexOf("private createOutput("),
    );
    expect(runCandidate).not.toContain("createSentinelBuffer(");
    expect(runCandidate).not.toContain("this.kernel.createDispatch(");
    const prepareResources = PAGE_SOURCE.slice(
      PAGE_SOURCE.indexOf("private async createPreparedCandidateResources("),
      PAGE_SOURCE.indexOf("private destroyPreparedCandidateResources("),
    );
    expect(prepareResources).toContain("createSentinelBuffer(");
    expect(prepareResources).toContain("this.kernel.createDispatch(");
    expect(prepareResources).toContain(
      "noAllocationMapFillOrDispatchConstructionInPrimaryWall",
    );
  });

  it("keeps scalar FP16 allocation out of the candidate decoder and timing", () => {
    const coreSource = readFileSync(new URL(
      "../benchmark/opt-0012-compact-semantic-head.ts",
      import.meta.url,
    ), "utf8");
    const decoder = coreSource.slice(
      coreSource.indexOf("export function decodeAceOpt0012CompactFp16Readback("),
      coreSource.indexOf(
        "export function reconstructAceOpt0012FullPlannerLogits(",
      ),
    );
    const decoderScalarLoop = decoder.slice(
      decoder.indexOf("for (const span of plan.readback.logicalSpans)"),
      decoder.indexOf("return Object.freeze({"),
    );
    expect(decoder).toContain("const rowWords = [");
    expect(decoderScalarLoop).toContain(
      "aceOpt0012Float16BitsToFloat32U32(source[index]!)",
    );
    expect(decoderScalarLoop).not.toContain("new Uint32Array([");
    expect(decoderScalarLoop).not.toContain("new Float32Array(words.buffer)");

    const frozenOracle = PAGE_SOURCE.slice(
      PAGE_SOURCE.indexOf("function writeOpt0012Fp16ConversionDomain("),
      PAGE_SOURCE.indexOf("export const OPT_0012_CASE_SPECS"),
    );
    expect(frozenOracle).toContain(
      "floats[bits] = legacyAllocatingAceOpt0012Float16BitsToNumber(bits)",
    );
    expect(frozenOracle).toContain(
      "!Number.isInteger(bits) || bits < 0 || bits > 0xffff",
    );
    expect(frozenOracle).toContain(
      "const words = new Uint32Array([output >>> 0]);",
    );
    expect(frozenOracle).toContain(
      "return new Float32Array(words.buffer)[0]!;",
    );
    expect(frozenOracle).toContain(
      "words[bits] = aceOpt0012Float16BitsToFloat32U32(bits)",
    );

    const conversionTiming = PAGE_SOURCE.slice(
      PAGE_SOURCE.indexOf("async function runOpt0012Fp16ConversionMicrobenchmark()"),
      PAGE_SOURCE.indexOf("async function runTimedAndCleanup("),
    );
    expect(conversionTiming.indexOf("outputs.set(arm, Object.freeze({"))
      .toBeLessThan(conversionTiming.indexOf("const pairs:"));
    expect(conversionTiming).not.toContain("new Uint32Array(output.buffer)");
    expect(conversionTiming).toContain("output.floats,\n        output.words,");
    const conversionPair = conversionTiming.slice(
      conversionTiming.indexOf("for (let pairIndex = 0;"),
      conversionTiming.indexOf("pairs.push(Object.freeze({"),
    );
    expect(conversionPair).not.toContain("await ");
    const pairArmLoopAt = conversionPair.indexOf("for (const arm of order)");
    const legacyEnvelopeAt = conversionPair.indexOf(
      "const legacyEnvelope = analyzeOpt0012Fp16ConversionEnvelope(",
    );
    const candidateEnvelopeAt = conversionPair.indexOf(
      "const candidateEnvelope = analyzeOpt0012Fp16ConversionEnvelope(",
    );
    expect(legacyEnvelopeAt).toBeGreaterThan(pairArmLoopAt);
    expect(candidateEnvelopeAt).toBeGreaterThan(legacyEnvelopeAt);
    expect(conversionPair).toContain(
      'legacyEnvelope["acceptedStableNanEnvelope"] !== true',
    );
    expect(conversionPair).toContain(
      'candidateEnvelope["acceptedStableNanEnvelope"] !== true',
    );
    expect(conversionTiming).toContain(
      "envelopeValidationAndReceiptExpansionOnlyAfterBothTimedArms: true",
    );
    expect(conversionTiming).toContain(
      "legacyRawNanHashesAreDiagnosticsNotPortablePayloadEvidence: true",
    );
    expect(conversionTiming).toContain("doesNotClaimStableJitTier: true");
  });

  it("fails closed on FP16 NaNs in every actual package readback authority", () => {
    const armExecution = PAGE_SOURCE.slice(
      PAGE_SOURCE.indexOf("async function executePackageArm("),
      PAGE_SOURCE.indexOf("interface SamplingStageIntervals"),
    );
    const primaryWallEndedAt = armExecution.indexOf(
      "const wallEnded = performance.now();",
    );
    const censusAt = armExecution.indexOf("const fp16NaNCensus = purpose ===");
    expect(primaryWallEndedAt).toBeGreaterThan(0);
    expect(censusAt).toBeGreaterThan(primaryWallEndedAt);
    expect(armExecution).toContain(
      'const fp16NaNCensus = purpose === "timing"\n      ? null',
    );
    expect(armExecution).toContain("requireZeroOpt0012RawFullFp16NaNs(");
    expect(armExecution).toContain("requireZeroOpt0012DecodedFullFp16NaNs(");
    expect(armExecution).toContain("requireZeroOpt0012RawCompactFp16NaNs(");
    expect(armExecution).toContain("fp16NaNCensus,");
    expect(PAGE_SOURCE).toContain(
      "fp16NaNCensus: execution.fp16NaNCensus ?? Object.freeze({",
    );
    expect(PAGE_SOURCE).toContain(
      'authority: "primaryTimingFp16NaNCensuses"',
    );

    const timed = PAGE_SOURCE.slice(
      PAGE_SOURCE.indexOf("async function runTimedAndCleanup("),
      PAGE_SOURCE.indexOf("function summarizeTimedArm("),
    );
    const primaryCompleteAt = timed.indexOf(
      "const primaryTokenTimedCompletedAtEpochMilliseconds = Date.now();",
    );
    const primaryCensusAt = timed.indexOf(
      "const primaryTimingFp16NaNCensuses = Object.freeze(",
    );
    expect(primaryCompleteAt).toBeGreaterThan(0);
    expect(primaryCensusAt).toBeGreaterThan(primaryCompleteAt);
    expect(timed).toContain("timedReadbackExecutions.push(sample)");
    expect(timed).toContain("primaryTimingFp16NaNCensuses,");
    expect(timed).toContain("primaryTimingFp16NaNCensusCoverage,");
    expect(timed).toContain(
      "validateOpt0012PrimaryTimingFp16NaNCensusCoverage(",
    );
    expect(PAGE_SOURCE).toContain(
      "everyActualPackageReadbackHasZeroBinary16NaNs: true",
    );
    expect(timed).toContain(
      "primaryReadbackNanCensusExecutedOnlyAfterEntirePrimaryWindow: true",
    );

    expect(OPT_0012_PRIMARY_TIMING_FP16_NAN_CENSUS_COUNT).toBe(54);
    const validCoverageReceipts = OPT_0012_CASE_SPECS.flatMap((spec) =>
      OPT_0012_BALANCED_ORDERS.flatMap((order, roundIndex) =>
        order.map((arm, orderPosition) => Object.freeze({
          arm,
          state: "regular-code",
          traceLabel: `${spec.id}-regular-code-${arm}-timing`,
          roundIndex,
          order: order.join(""),
          orderPosition,
          census: Object.freeze({ zeroBinary16NaNs: true }),
        }))));
    expect(validCoverageReceipts).toHaveLength(54);
    expect(validateOpt0012PrimaryTimingFp16NaNCensusCoverage(
      validCoverageReceipts,
    )).toMatchObject({
      receiptCount: 54,
      expectedReceiptCount: 54,
      A: 18,
      B: 18,
      C: 18,
      uniqueTraceLabelCount: 9,
      uniqueCompositeJoinKeyCount: 54,
      compositeJoinFields: [
        "traceLabel",
        "roundIndex",
        "order",
        "orderPosition",
      ],
      oneToOneCompositeTraceJoin: true,
      everyActualPackageReadbackHasZeroBinary16NaNs: true,
    });
    const duplicateCompositeKey = [...validCoverageReceipts];
    duplicateCompositeKey[duplicateCompositeKey.length - 1] =
      validCoverageReceipts[0]!;
    expect(() => validateOpt0012PrimaryTimingFp16NaNCensusCoverage(
      duplicateCompositeKey,
    )).toThrow(/duplicate composite join key/u);

    const rawCensus = PAGE_SOURCE.slice(
      PAGE_SOURCE.indexOf("function requireZeroOpt0012RawFullFp16NaNs("),
      PAGE_SOURCE.indexOf("function publicArmExecution("),
    );
    expect(rawCensus).toContain(
      "((bits >>> 10) & 0x1f) !== 0x1f || mantissa === 0",
    );
    expect(rawCensus).toContain("binary16NaNCount: nanCount");
    expect(rawCensus).toContain("zeroBinary16NaNs: nanCount === 0");
    expect(rawCensus).toContain(
      "actual package readback contains FP16 NaN",
    );
    expect(rawCensus).toContain(
      'authority: "lossless-full-FP16-readback-decode-NaN-classification"',
    );
    expect(rawCensus).toContain("everyBinary16NaNWouldDecodeToNumberNaN: true");

    const sameByteCase = PAGE_SOURCE.slice(
      PAGE_SOURCE.indexOf("async function runSameImmutableByteReplayCase("),
      PAGE_SOURCE.indexOf("function runSameByteReplayExecution("),
    );
    expect(sameByteCase).toContain(
      "const fp16NaNCensus = requireZeroOpt0012RawCompactFp16NaNs(",
    );
    expect(sameByteCase).toContain("fp16NaNCensus,");

    const trajectoryInvocation = PAGE_SOURCE.slice(
      PAGE_SOURCE.indexOf("async function executeTrajectoryInvocation("),
      PAGE_SOURCE.indexOf("function sampleTrajectoryRows("),
    );
    expect(trajectoryInvocation).toContain(
      "const fp16NaNCensus = arm === \"A\"",
    );
    expect(trajectoryInvocation.indexOf("const ended = performance.now();"))
      .toBeLessThan(trajectoryInvocation.indexOf("const fp16NaNCensus"));
    expect(trajectoryInvocation).toContain("fp16NaNCensus,");
    const trajectorySummary = PAGE_SOURCE.slice(
      PAGE_SOURCE.indexOf("function summarizeTrajectoryTopology("),
      PAGE_SOURCE.indexOf("async function runTrajectoryCancellation("),
    );
    expect(trajectorySummary).toContain(
      "fp16NaNCensuses.some((census) => census.zeroBinary16NaNs !== true)",
    );
    expect(trajectorySummary).toContain(
      "everyActualPackageReadbackHasZeroBinary16NaNs: true",
    );
    expect(trajectorySummary).toContain(
      "fp16NaNCensuses: Object.freeze(fp16NaNCensuses)",
    );
  });

  it("runs corrective CPU evidence after unchanged primary A/B/C timing", () => {
    const sameByteCase = PAGE_SOURCE.slice(
      PAGE_SOURCE.indexOf("async function runSameImmutableByteReplayCase("),
      PAGE_SOURCE.indexOf("function runSameByteReplayExecution("),
    );
    expect(sameByteCase).toContain("const mappedBytes = b.compactMappedBytes;");
    expect(sameByteCase).toContain(
      "mappedBytes,\n        plan,\n        fixture,\n      );",
    );
    expect(sameByteCase).toContain("mappedBytesSha256Before: before");
    expect(sameByteCase).toContain("mappedBytesSha256After: after");
    expect(sameByteCase).toContain("sameArrayBufferObjectForEveryArmAndPair: true");
    const replayPair = sameByteCase.slice(
      sameByteCase.indexOf("for (let pairIndex = 0;"),
      sameByteCase.indexOf("pairs.push(Object.freeze({"),
    );
    expect(replayPair).not.toContain("await ");
    expect(replayPair.indexOf("finalizeSameByteReplayExecution"))
      .toBeGreaterThan(replayPair.indexOf("for (const arm of order)"));
    expect(sameByteCase).toContain(
      "noHashingOrEquivalenceWorkBetweenTimedPairArms: true",
    );

    const replayExecution = PAGE_SOURCE.slice(
      PAGE_SOURCE.indexOf("function runSameByteReplayExecution("),
      PAGE_SOURCE.indexOf("function summarizeSameByteReplayArm("),
    );
    expect(replayExecution).toContain(
      "const decoded = decodeAceOpt0012CompactFp16Readback(mappedBytes, plan)",
    );
    expect(replayExecution).toContain(
      'if (arm === "B") {\n    const reconstructionStarted',
    );
    expect(replayExecution).toContain("totalWallMilliseconds: totalEnded - totalStarted");
    expect(replayExecution).toContain("namedNonOverlappingIntervals");
    expect(replayExecution).toContain("primarySamplingStages: Object.freeze({");
    expect(replayExecution).not.toContain("primarySampling: sampling.intervals");
    expect(replayExecution).not.toContain("primarySamplingOuterWallMilliseconds");
    expect(replayExecution).toContain("decodedRowSha256");
    expect(replayExecution).toContain("reconstructedFullRowSha256");
    expect(replayExecution).toContain("writeStatus: Object.freeze([");

    for (const boundary of [
      "cfg", "finalMask", "repetition", "topKLogits",
      "topKGlobalTokenIds", "topPKeep", "topPGlobalTokenIds",
      "topPLogits", "temperature", "weights",
    ]) {
      expect(PAGE_SOURCE).toContain(`${boundary},`);
    }
    expect(PAGE_SOURCE).toContain("categorical: Object.freeze({");
    expect(PAGE_SOURCE).toContain("everyBoundaryExact: true");
    expect(PAGE_SOURCE).toContain("appliesToEveryPairBecauseInputPlanSeenTokensAndWordAreIdentical");

    const timed = PAGE_SOURCE.slice(
      PAGE_SOURCE.indexOf("async function runTimedAndCleanup("),
      PAGE_SOURCE.indexOf("function summarizeTimedArm("),
    );
    const primaryCompleteAt = timed.indexOf(
      "const primaryTokenTimedCompletedAtEpochMilliseconds",
    );
    const replayAt = timed.indexOf("await runSameImmutableByteReplayCase(");
    const conversionAt = timed.indexOf(
      "await runOpt0012Fp16ConversionMicrobenchmark()",
    );
    const cancellationAt = timed.indexOf("await runCancellationProofs(prepared)");
    expect(primaryCompleteAt).toBeGreaterThan(0);
    expect(replayAt).toBeGreaterThan(primaryCompleteAt);
    expect(conversionAt).toBeGreaterThan(replayAt);
    expect(cancellationAt).toBeGreaterThan(conversionAt);
    expect(timed).toContain(
      "originalSixOrderAbcCompletedFirstAndUnchanged: true",
    );
  });

  it("authenticates exact capability/requested-limit and paired timing receipts", () => {
    const capabilities = PAGE_SOURCE.slice(
      PAGE_SOURCE.indexOf("function authenticateOpt0012Capabilities("),
      PAGE_SOURCE.indexOf("function authenticateOpt0012Sources("),
    );
    expect(capabilities).toContain(
      'name !== "minStorageBufferOffsetAlignment"',
    );
    expect(capabilities).toContain(
      'name !== "minUniformBufferOffsetAlignment"',
    );
    expect(capabilities).toContain("const deviceActual = capabilities.deviceLimits");
    expect(capabilities).toContain("const adapterActual = capabilities.adapterLimits");
    expect(capabilities).toContain("JSON.stringify(observedRequestedLimits)");
    expect(PAGE_SOURCE).toContain("sameRoundComparisons: Object.freeze({");
    expect(PAGE_SOURCE).toContain("rawDeltaConvention: \"right-minus-left-milliseconds\"");
    expect(PAGE_SOURCE).toContain("leftWins + rightWins + ties");
    expect(PAGE_SOURCE).toContain("thresholdApplied: false");
  });

  it("couples cancellation to an observed raw trajectory and stable activity", () => {
    const resident = PAGE_SOURCE.slice(
      PAGE_SOURCE.indexOf("async function runTrajectoryCancellation("),
      PAGE_SOURCE.indexOf("async function destroyPreparedSession("),
    );
    expect(resident).toContain('authority: "observed-raw-fp16-arm-A-prefix"');
    expect(resident).toContain("activeCachePublicationSnapshot()");
    expect(resident).toContain('cancellationBoundary: "after-head"');
    expect(resident).toContain("boundaryAbortController");
    expect(resident).toContain("activityBeforeTurns");
    expect(resident).toContain("activityAfterTurns");
    expect(resident).toContain("cursor.consumed !== cursorBefore");
    expect(resident).toContain("publishedCallbackCount !== callbackCountBefore");
    expect(resident).toContain("postCancellationMacrotaskTurnCount: 2");
  });

  it("keeps one dedicated worker, correctness-first UI, and separate trajectory mode", () => {
    expect(PAGE_SOURCE).toContain("new Worker(");
    expect(PAGE_SOURCE).toContain("{ type: \"module\", name: \"ace-opt-0012-worker\" }");
    expect(PAGE_SOURCE).toContain("run-trajectory");
    expect(HTML_SOURCE).toContain("correctness-first");
    expect(HTML_SOURCE).toContain("all six A/B/C permutations");
    expect(HTML_SOURCE).toContain("raw-FP16");
    expect(HTML_SOURCE).toContain("150-code plus forced-EOS");
    expect(HTML_SOURCE).not.toContain("accepted 150-code");
    expect(HTML_SOURCE).toContain("Keep the external thermal logger running");
    expect(PAGE_SOURCE).toContain("production-executor-onProgress");
    expect(PAGE_SOURCE).toContain("benchmark-candidate-equivalent-progress");
    expect(PAGE_SOURCE).toContain("postLossMacrotaskTurnCount: 2");
    expect(PAGE_SOURCE).toContain("preDestroyRuntimeObservationMacrotaskTurnCount: 2");
    expect(PAGE_SOURCE).toContain("queued runtime event before context destroy");
    expect(PAGE_SOURCE).toContain("terminalNow - lastAnimationFrame");
    expect(PAGE_SOURCE).toContain("terminalNow - lastTimer");
    expect(PAGE_SOURCE).toContain("performance.now() - last");
    expect(PAGE_SOURCE).toContain("explicitPreInvocationStatusResetCount !== 150");
    expect(PAGE_SOURCE).toContain("excludedFromEveryPrimaryTimingObservation: true");
    expect(PAGE_SOURCE).toContain("maximumSimultaneouslyLiveTrackedBufferBytes");
    expect(PAGE_SOURCE).toContain("maximumSimultaneouslyLiveTrackedBufferCount");
    expect(PAGE_SOURCE).toContain("simultaneous GPUBuffer residency accounting drifted");
    expect(PAGE_SOURCE).toContain("primary sampling stage intervals overlap or changed");
    expect(PAGE_SOURCE).toContain("preCfgConstraintMilliseconds");
    expect(PAGE_SOURCE).toContain("postCfgConstraintAndRepetitionMilliseconds");
    expect(PAGE_SOURCE).toContain("temperatureAndSoftmaxMilliseconds");
    expect(PAGE_SOURCE).toContain("categoricalWordAndGlobalMappingMilliseconds");
    expect(PAGE_SOURCE).toContain("callbackInvocationCount: 1");
    expect(PAGE_SOURCE).toContain("everyRealIdleTimerCompleted: true");
    expect(PAGE_SOURCE).toContain("await this.progress.settleIdleAfterFailure()");
    expect(PAGE_SOURCE).toContain("await headIdle;\n        throw error;");
    expect(PAGE_SOURCE).toContain("await readbackIdle;\n          throw error;");
    expect(PAGE_SOURCE).toContain("primary=${primary}");
    expect(PAGE_SOURCE).toContain("exactOverlappingReadbackIdle");
    expect(PAGE_SOURCE).toContain("reportedAsOverlappingDiagnosticWithoutSubtraction: true");
    expect(PAGE_SOURCE).toContain("encoded fresh prefill status clear changed");
    expect(PAGE_SOURCE).toContain("expectedEncodedFreshStatusClears");
    expect(PAGE_SOURCE).not.toContain("realIdleMilliseconds:");
  });
});
