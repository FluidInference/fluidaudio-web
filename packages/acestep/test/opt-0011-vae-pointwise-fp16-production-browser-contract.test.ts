import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

import {
  aceFp16VaeAddWgsl,
  aceFp16VaeIngressWgsl,
  planAceFp16VaeAdd,
  planAceFp16VaeIngress,
  planAceFp16VaePointwiseRange,
} from "../src/webgpu/kernels/vae-pointwise-fp16.js";
import {
  planAceVaeDecoder,
  planAceVaeDecoderQuanta,
  type AceVaeDecoderAddOperation,
} from "../src/webgpu/vae-decoder.js";
import {
  OPT_0011_PRODUCTION_POINTWISE_ADD_ARITHMETIC_CASE,
  OPT_0011_PRODUCTION_POINTWISE_CORE_COMMIT,
  OPT_0011_PRODUCTION_POINTWISE_CORE_SOURCE_SHA256,
  OPT_0011_PRODUCTION_POINTWISE_GENERATED_SHADER_SHA256,
  OPT_0011_PRODUCTION_POINTWISE_GRAPH_ADD_CASES,
  OPT_0011_PRODUCTION_POINTWISE_INGRESS_CASES,
  OPT_0011_PRODUCTION_POINTWISE_RAW_RESULT_CHUNK_CODE_UNITS,
  compareOpt0011ProductionPointwiseRawBits,
  float16BitsToNumber,
  opt0011ProductionPointwiseAddCpuBits,
  opt0011ProductionPointwiseIngressCpuBits,
  parseOpt0011ProductionPointwiseRawResultChunkOffset,
  parseOpt0011ProductionPointwiseRunIdentity,
  sliceOpt0011ProductionPointwiseRawResultChunk,
  stopOpt0011ProductionPointwiseHeartbeatAfterFailure,
} from "./browser/opt-0011-vae-pointwise-fp16-production.js";

describe("OPT-0011 production FP16 pointwise actual-browser contract", () => {
  it("pins all 15 Add nodes and the five exact B-256 shapes", () => {
    expect(OPT_0011_PRODUCTION_POINTWISE_GRAPH_ADD_CASES.map(({ id }) => id))
      .toEqual([
        "block-0-res-1-add",
        "block-0-res-2-add",
        "block-0-res-3-add",
        "block-1-res-1-add",
        "block-1-res-2-add",
        "block-1-res-3-add",
        "block-2-res-1-add",
        "block-2-res-2-add",
        "block-2-res-3-add",
        "block-3-res-1-add",
        "block-3-res-2-add",
        "block-3-res-3-add",
        "block-4-res-1-add",
        "block-4-res-2-add",
        "block-4-res-3-add",
      ]);
    expect(OPT_0011_PRODUCTION_POINTWISE_GRAPH_ADD_CASES.map(({ addOrdinal }) =>
      addOrdinal
    )).toEqual(Array.from({ length: 15 }, (_, index) => index));
    expect(OPT_0011_PRODUCTION_POINTWISE_GRAPH_ADD_CASES.map(({ shape }) =>
      `${shape.frames}:${shape.channels}`
    )).toEqual([
      ...Array(3).fill("2560:1024"),
      ...Array(3).fill("15360:512"),
      ...Array(3).fill("61440:256"),
      ...Array(3).fill("245760:128"),
      ...Array(3).fill("491520:128"),
    ]);
    expect(OPT_0011_PRODUCTION_POINTWISE_GRAPH_ADD_CASES.map(
      ({ shapeFamilyIndex }) => shapeFamilyIndex,
    )).toEqual([
      0, 0, 0,
      1, 1, 1,
      2, 2, 2,
      3, 3, 3,
      4, 4, 4,
    ]);
    expect(OPT_0011_PRODUCTION_POINTWISE_GRAPH_ADD_CASES.map(
      ({ ranges }) => ranges.length,
    )).toEqual([
      3, 3, 3,
      8, 8, 8,
      15, 15, 15,
      30, 30, 30,
      60, 60, 60,
    ]);
  });

  it("binds all 348 ranges to the exact unchanged B-256 graph quanta", () => {
    const graph = planAceVaeDecoder(256);
    const cooperative = planAceVaeDecoderQuanta(graph);
    let rangeCount = 0;
    for (const fixture of OPT_0011_PRODUCTION_POINTWISE_GRAPH_ADD_CASES) {
      const operation = graph.operations[fixture.operationIndex];
      expect(operation).toMatchObject({
        kind: "add",
        label: fixture.id,
        shape: fixture.shape,
      });
      expect((operation as AceVaeDecoderAddOperation).shape)
        .toEqual(fixture.shape);
      const plan = planAceFp16VaeAdd(fixture.shape);
      let cursor = 0;
      for (const [index, range] of fixture.ranges.entries()) {
        const quantum = cooperative.quanta[range.quantumIndex]!;
        expect(quantum).toMatchObject({
          operationIndex: fixture.operationIndex,
          operationLabel: fixture.id,
          operationKind: "add",
          logicalOutputBase: range.base,
          logicalOutputCount: range.count,
          estimatedMaximumMultiplyAccumulates: 0,
        });
        expect(range.operationQuantumIndex).toBe(index);
        expect(quantum.primitives).toHaveLength(1);
        expect(quantum.primitives[0]).toMatchObject({
          outputBase: range.base,
          outputCount: range.count,
        });
        expect(range.base).toBe(cursor);
        expect(planAceFp16VaePointwiseRange(plan, range)).toMatchObject({
          base: range.base,
          count: range.count,
        });
        cursor += range.count;
        rangeCount += 1;
      }
      expect(cursor).toBe(plan.elements);
    }
    expect(rangeCount).toBe(348);
  });

  it("pins the complete B-256 ingress and explicit odd-tail fixtures", () => {
    expect(OPT_0011_PRODUCTION_POINTWISE_INGRESS_CASES).toHaveLength(2);
    expect(OPT_0011_PRODUCTION_POINTWISE_INGRESS_CASES[0]).toMatchObject({
      id: "ingress-b256-complete",
      operation: "ingress",
      shape: { batch: 1, frames: 256, channels: 64 },
      ranges: [{ base: 0, count: 16_384 }],
    });
    expect(planAceFp16VaeIngress(
      OPT_0011_PRODUCTION_POINTWISE_INGRESS_CASES[0]!.shape,
    )).toMatchObject({
      elements: 16_384,
      sourceStorageBytes: 65_536,
      outputStorageBytes: 32_768,
      outputBindingBytes: 32_768,
    });
    expect(OPT_0011_PRODUCTION_POINTWISE_INGRESS_CASES[1]).toMatchObject({
      id: "ingress-arithmetic-odd-tail-257",
      shape: { batch: 1, frames: 1, channels: 257 },
      ranges: [
        { base: 0, count: 256 },
        { base: 256, count: 1 },
      ],
    });
    expect(OPT_0011_PRODUCTION_POINTWISE_ADD_ARITHMETIC_CASE).toMatchObject({
      id: "add-arithmetic-odd-tail-257",
      shape: { batch: 1, frames: 1, channels: 257 },
      ranges: [
        { base: 0, count: 256 },
        { base: 256, count: 1 },
      ],
    });
    expect(planAceFp16VaeAdd(
      OPT_0011_PRODUCTION_POINTWISE_ADD_ARITHMETIC_CASE.shape,
    )).toMatchObject({
      elements: 257,
      sourceStorageBytes: 514,
      sourceBindingBytes: 516,
      outputStorageBytes: 514,
      outputBindingBytes: 516,
    });
  });

  it("pins raw signed-zero, subnormal, and RNE CPU-oracle bits", () => {
    expect(Array.from({ length: 16 }, (_, index) =>
      opt0011ProductionPointwiseIngressCpuBits(index)
    )).toEqual([
      0x0000, 0x8000, 0x0000, 0x0002,
      0x0001, 0x8001, 0x3c00, 0x3c02,
      0xbc00, 0xbc02, 0x0400, 0x7bff,
      0x3555, 0xb555, 0x1400, 0x9400,
    ]);
    expect(Array.from({ length: 10 }, (_, index) =>
      opt0011ProductionPointwiseAddCpuBits(index)
    )).toEqual([
      0x8000,
      0x0000,
      0x0002,
      0x0000,
      0x0400,
      0x3c00,
      0x3c02,
      0xbc00,
      0xbc02,
      0x03ff,
    ]);
    expect(Object.is(float16BitsToNumber(0x8000), -0)).toBe(true);
  });

  it("retains the first raw mismatch", () => {
    expect(compareOpt0011ProductionPointwiseRawBits(
      new Uint16Array([0, 1, 2, 3]),
      new Uint16Array([0, 9, 2, 8]),
    )).toEqual({ mismatchCount: 2, firstMismatchIndex: 1 });
    expect(() => compareOpt0011ProductionPointwiseRawBits(
      new Uint16Array([0]),
      new Uint16Array([0, 1]),
    )).toThrow(/output lengths differ/);
  });

  it("publishes bounded contiguous chunks without splitting surrogate pairs", () => {
    expect(OPT_0011_PRODUCTION_POINTWISE_RAW_RESULT_CHUNK_CODE_UNITS)
      .toBe(100_000);
    const prefix = "a".repeat(
      OPT_0011_PRODUCTION_POINTWISE_RAW_RESULT_CHUNK_CODE_UNITS - 1,
    );
    const raw = `${prefix}\ud83d\ude80tail`;
    const first = sliceOpt0011ProductionPointwiseRawResultChunk(raw, 0);
    expect(first).toEqual({
      chunk: prefix,
      start: 0,
      end: prefix.length,
      nextOffset: prefix.length,
      totalCodeUnits: raw.length,
      complete: false,
    });
    const second = sliceOpt0011ProductionPointwiseRawResultChunk(
      raw,
      first.nextOffset,
    );
    expect(second.chunk).toBe("🚀tail");
    expect(second.end).toBe(raw.length);
    expect(second.complete).toBe(true);
    expect(first.chunk + second.chunk).toBe(raw);
    expect(sliceOpt0011ProductionPointwiseRawResultChunk(raw, raw.length))
      .toEqual({
        chunk: "",
        start: raw.length,
        end: raw.length,
        nextOffset: raw.length,
        totalCodeUnits: raw.length,
        complete: true,
      });
    expect(sliceOpt0011ProductionPointwiseRawResultChunk(raw, 0).chunk)
      .toBe(prefix);
    expect(() => sliceOpt0011ProductionPointwiseRawResultChunk(raw, -1))
      .toThrow(/offset/u);
    expect(() => sliceOpt0011ProductionPointwiseRawResultChunk(
      raw,
      raw.length + 1,
    )).toThrow(/offset/u);
    expect(() => sliceOpt0011ProductionPointwiseRawResultChunk(raw, 0.5))
      .toThrow(/offset/u);
    expect(() => sliceOpt0011ProductionPointwiseRawResultChunk(
      raw,
      prefix.length + 1,
    )).toThrow(/surrogate/u);
  });

  it("accepts only canonical safe raw-result offsets", () => {
    expect(parseOpt0011ProductionPointwiseRawResultChunkOffset("0")).toBe(0);
    expect(parseOpt0011ProductionPointwiseRawResultChunkOffset(
      "9007199254740991",
    )).toBe(Number.MAX_SAFE_INTEGER);
    for (
      const value of [
        "",
        " ",
        "-1",
        "+1",
        "01",
        "1.0",
        "1e3",
        "9007199254740992",
      ]
    ) {
      expect(() => parseOpt0011ProductionPointwiseRawResultChunkOffset(value))
        .toThrow(/offset/u);
    }
  });

  it("keeps the full receipt immutable and out of the rendered DOM", () => {
    const harness = readFileSync(new URL(
      "./browser/opt-0011-vae-pointwise-fp16-production.ts",
      import.meta.url,
    ), "utf8");
    const html = readFileSync(new URL(
      "./browser/opt-0011-vae-pointwise-fp16-production.html",
      import.meta.url,
    ), "utf8");
    expect(harness).toContain(
      '"__ACE_OPT_0011_PRODUCTION_POINTWISE_RAW_RESULT_JSON__"',
    );
    expect(harness).toContain("Reflect.defineProperty(");
    expect(harness).toContain("configurable: false");
    expect(harness).toContain("enumerable: false");
    expect(harness).toContain("writable: false");
    expect(harness).toContain("fullReceiptIntentionallyKeptOutOfDom: true");
    expect(harness).toContain(
      "rawResultJsonCodeUnitLength: rawResultJson.length",
    );
    expect(harness).toContain('rawResultRetrieval: "bounded-dom-chunks"');
    expect(harness).toContain(
      "output.dataset.publicationSequence = String(++publicationSequence)",
    );
    expect(harness).toContain("delete output.dataset.startOffset");
    expect(harness).toContain("delete output.dataset.endOffsetExclusive");
    expect(harness).toContain("delete output.dataset.chunkCodeUnitLength");
    expect(harness).toContain("delete output.dataset.totalCodeUnitLength");
    expect(harness).toContain("delete output.dataset.done");
    expect(harness).toContain("output.textContent = slice.chunk");
    expect(harness).toContain('offsetInput.value = "0"');
    expect(harness).toContain("publish.disabled = false");
    expect(harness).not.toContain("publish.disabled = slice.complete");
    expect(harness).not.toContain("innerHTML");
    expect(harness).not.toContain(
      "output.textContent = JSON.stringify(result, null, 2)",
    );
    expect(html).toContain('id="raw-result-retrieval" hidden disabled');
    expect(html).toContain('name="rawResultOffset"');
    expect(html).toContain('id="publish-raw-result-chunk"');
    expect(html).toMatch(
      /id="raw-result-chunk"[\s\S]*data-state="empty"[\s\S]*hidden[\s\S]*aria-hidden="true"/u,
    );
  });

  it("preserves the primary failure if heartbeat stopping itself fails", () => {
    const stopped = stopOpt0011ProductionPointwiseHeartbeatAfterFailure({
      stop(): never {
        throw new Error("synthetic heartbeat stop failure");
      },
    });
    expect(stopped.responsiveness).toBeNull();
    expect(stopped.heartbeatStopError).toMatchObject({
      name: "Error",
      message: "synthetic heartbeat stop failure",
    });
    expect(stopOpt0011ProductionPointwiseHeartbeatAfterFailure({
      stop: () => Object.freeze({ observed: false }),
    })).toEqual({
      responsiveness: { observed: false },
      heartbeatStopError: null,
    });
  });

  it("freezes the exact generated ingress and Add WGSL hashes", () => {
    const ingress = aceFp16VaeIngressWgsl();
    const add = aceFp16VaeAddWgsl();
    expect(OPT_0011_PRODUCTION_POINTWISE_GENERATED_SHADER_SHA256).toEqual({
      ingress: sha256(ingress),
      add: sha256(add),
    });
    expect(OPT_0011_PRODUCTION_POINTWISE_GENERATED_SHADER_SHA256.ingress)
      .toMatch(/^[0-9a-f]{64}$/);
    expect(OPT_0011_PRODUCTION_POINTWISE_GENERATED_SHADER_SHA256.add)
      .toMatch(/^[0-9a-f]{64}$/);
    expect(ingress).toContain("output[index] = f16(input[index]);");
    expect(add).toContain("let sum: f32 = left_operand + right_operand;");
    expect(add).toContain("output[index] = f16(sum);");
    expect(ingress).not.toContain("subgroup");
    expect(add).not.toContain("subgroup");
  });

  it("requires exact immutable commit identities in the browser URL", () => {
    const valid = new URLSearchParams({
      harnessCommit: "1234567890abcdef1234567890abcdef12345678",
      coreCommit: OPT_0011_PRODUCTION_POINTWISE_CORE_COMMIT,
    });
    expect(parseOpt0011ProductionPointwiseRunIdentity(valid)).toEqual({
      harnessCommit: "1234567890abcdef1234567890abcdef12345678",
      coreCommit: OPT_0011_PRODUCTION_POINTWISE_CORE_COMMIT,
    });
    valid.set("coreCommit", "0000000000000000000000000000000000000000");
    expect(() => parseOpt0011ProductionPointwiseRunIdentity(valid))
      .toThrow(/coreCommit changed/);
    valid.set("coreCommit", OPT_0011_PRODUCTION_POINTWISE_CORE_COMMIT);
    valid.delete("harnessCommit");
    expect(() => parseOpt0011ProductionPointwiseRunIdentity(valid))
      .toThrow(/requires one harnessCommit/);
  });

  it("pins the committed production pointwise source bytes", () => {
    expect(OPT_0011_PRODUCTION_POINTWISE_CORE_COMMIT)
      .toBe("dd36a04960f846e53c2fd948d67b9aa9ddced4f2");
    const production = readFileSync(new URL(
      "../src/webgpu/kernels/vae-pointwise-fp16.ts",
      import.meta.url,
    ));
    expect(createHash("sha256").update(production).digest("hex"))
      .toBe(OPT_0011_PRODUCTION_POINTWISE_CORE_SOURCE_SHA256);
  });

  it("statically binds CPU bits, guards, reruns, cancellation, events, and cleanup", () => {
    const harness = readFileSync(new URL(
      "./browser/opt-0011-vae-pointwise-fp16-production.ts",
      import.meta.url,
    ), "utf8");
    expect(harness).toContain('requiredFeatures: ["shader-f16"]');
    expect(harness).toContain("productionCoreSource");
    expect(harness).toContain("generated shader SHA-256 changed");
    expect(harness).toContain("OUTPUT_GUARD_F16 = 0x7e33");
    expect(harness).toContain("OUTPUT_CANARY_F16 = 0x7e11");
    expect(harness).toContain("SOURCE_PADDING_F16 = 0x7e55");
    expect(harness).toContain("completeSelectedRangeRawBitComparison: true");
    expect(harness).toContain("deterministicRerunHashes: true");
    expect(harness).toContain("await device.queue.onSubmittedWorkDone()");
    expect(harness).toContain("await queueEmptyIdleTurn()");
    expect(harness).toContain("readbackCount !== 0");
    expect(harness).toContain("laterEncodingPrevented: true");
    expect(harness).toContain("laterSubmissionPrevented: true");
    expect(harness).toContain("readbackPrevented: true");
    expect(harness).toContain('addEventListener("uncapturederror"');
    expect(harness).toContain("rawDevice.lost.then");
    expect(harness.match(/tracker\.destroyAll\(\);/g)).toHaveLength(2);
    expect(harness).toContain("prepared.destroy();\n    prepared.destroy();");
    const heartbeatStart = harness.indexOf(
      "const heartbeat = startHeartbeat();",
    );
    const browserRun = harness.indexOf("void runBrowser(heartbeat).then(");
    const sourceAuthentication = harness.indexOf(
      "const sourceAuthority = await authenticateSources(identity);",
    );
    const deviceDestroy = harness.indexOf("rawDevice.destroy();");
    const intentionalLoss = harness.indexOf(
      "const intentionalDeviceLoss = await rawDevice.lost;",
    );
    const cleanupTurns = harness.indexOf(
      "const postCleanupEventTurnsCompletedAtEpochMilliseconds",
    );
    const eventSnapshot = harness.indexOf(
      "const finalEventSnapshotAtEpochMilliseconds",
    );
    const eventValidation = harness.indexOf(
      "const cleanupAndEventValidationAtEpochMilliseconds",
    );
    const listenersRemoved = harness.indexOf(
      "const eventListenersRemovedAtEpochMilliseconds",
    );
    const heartbeatStop = harness.indexOf(
      "stopOpt0011ProductionPointwiseHeartbeatAfterFailure(heartbeat);",
      listenersRemoved,
    );
    expect(heartbeatStart).toBeGreaterThan(0);
    expect(heartbeatStart).toBeLessThan(browserRun);
    expect(browserRun).toBeLessThan(sourceAuthentication);
    expect(harness).toContain(
      "stopOpt0011ProductionPointwiseHeartbeatAfterFailure(heartbeat)",
    );
    expect(harness).toContain(
      "primaryErrorPreservedAcrossHeartbeatStop: true",
    );
    expect(deviceDestroy).toBeGreaterThan(0);
    expect(deviceDestroy).toBeLessThan(intentionalLoss);
    expect(intentionalLoss).toBeLessThan(cleanupTurns);
    expect(cleanupTurns).toBeLessThan(eventSnapshot);
    expect(eventSnapshot).toBeLessThan(eventValidation);
    expect(eventValidation).toBeLessThan(listenersRemoved);
    expect(listenersRemoved).toBeLessThan(heartbeatStop);
    expect(harness).toContain(
      "heartbeatStop.heartbeatStopError === null",
    );
    expect(harness).toContain('intentionalDeviceLoss.reason === "destroyed"');
    expect(harness).toContain("responsiveness.observed === true");
    expect(harness).toContain("performanceClaim: null");
    expect(harness).toContain("thermalClaim: null");
    expect(harness).toContain("qualityClaim: null");
    expect(harness).toContain("productionSelectorClaim: null");
  });

  it("labels the page as bounded correctness only", () => {
    const html = readFileSync(new URL(
      "./browser/opt-0011-vae-pointwise-fp16-production.html",
      import.meta.url,
    ), "utf8");
    expect(html).toContain("Correctness only");
    expect(html).toContain("complete B-256 FP32-to-FP16 decoder ingress");
    expect(html).toContain("all 15 Add");
    expect(html).toContain("all 348 exact B-256 graph ranges");
    expect(html).toContain("no timing");
    expect(html).toContain("opt-0011-vae-pointwise-fp16-production.ts");
  });
});

function sha256(source: string): string {
  return createHash("sha256").update(source).digest("hex");
}
