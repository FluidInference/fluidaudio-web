import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

import {
  aceFp16VaeConvTranspose1dWgsl,
  planAceFp16VaeConvTranspose1d,
  planAceFp16VaeConvTranspose1dRange,
} from "../src/webgpu/kernels/vae-conv-transpose1d-fp16.js";
import {
  planAceVaeDecoder,
  planAceVaeDecoderQuanta,
} from "../src/webgpu/vae-decoder.js";
import {
  OPT_0011_PRODUCTION_CONV_TRANSPOSE1D_CASES,
  OPT_0011_PRODUCTION_CONV_TRANSPOSE1D_CORE_COMMIT,
  OPT_0011_PRODUCTION_CONV_TRANSPOSE1D_CORE_SOURCE_SHA256,
  OPT_0011_PRODUCTION_CONV_TRANSPOSE1D_GENERATED_SHADER_SHA256,
  OPT_0011_PRODUCTION_CONV_TRANSPOSE1D_GRAPH_CASES,
  OPT_0011_PRODUCTION_CONV_TRANSPOSE1D_RAW_RESULT_CHUNK_CODE_UNITS,
  compareOpt0011ProductionConvTranspose1dRawBits,
  float16BitsToNumber,
  numberToFloat16Bits,
  opt0011ProductionConvTranspose1dCpuBits,
  parseOpt0011ProductionConvTranspose1dRawResultChunkOffset,
  parseOpt0011ProductionConvTranspose1dRunIdentity,
  sliceOpt0011ProductionConvTranspose1dRawResultChunk,
  stopOpt0011ProductionConvTranspose1dHeartbeatAfterFailure,
} from "./browser/opt-0011-vae-conv-transpose1d-fp16-production.js";

const HARNESS_SOURCE = readFileSync(new URL(
  "./browser/opt-0011-vae-conv-transpose1d-fp16-production.ts",
  import.meta.url,
), "utf8");
const HTML_SOURCE = readFileSync(new URL(
  "./browser/opt-0011-vae-conv-transpose1d-fp16-production.html",
  import.meta.url,
), "utf8");

describe("OPT-0011 production FP16 ConvTranspose1D browser contract", () => {
  it("pins all five canonical B-256 operations and all 322 exact quanta", () => {
    expect(OPT_0011_PRODUCTION_CONV_TRANSPOSE1D_GRAPH_CASES.map(
      ({ label }) => label,
    )).toEqual([
      "block-0-conv-t1",
      "block-1-conv-t1",
      "block-2-conv-t1",
      "block-3-conv-t1",
      "block-4-conv-t1",
    ]);
    expect(OPT_0011_PRODUCTION_CONV_TRANSPOSE1D_GRAPH_CASES.map(
      ({ ranges }) => ranges.length,
    )).toEqual([46, 69, 69, 69, 69]);
    const graph = planAceVaeDecoder(256);
    const cooperative = planAceVaeDecoderQuanta(graph);
    let quantumCount = 0;
    for (const graphCase of OPT_0011_PRODUCTION_CONV_TRANSPOSE1D_GRAPH_CASES) {
      const operation = graph.operations[graphCase.operationIndex]!;
      expect(operation).toMatchObject({
        kind: "conv-transpose1d",
        label: graphCase.label,
        shape: graphCase.shape,
      });
      const plan = planAceFp16VaeConvTranspose1d(graphCase.shape);
      let cursor = 0;
      for (const [index, range] of graphCase.ranges.entries()) {
        const quantum = cooperative.quanta[range.quantumIndex]!;
        expect(range.operationQuantumIndex).toBe(index);
        expect(range.base).toBe(cursor);
        expect(quantum).toMatchObject({
          index: range.quantumIndex,
          operationIndex: graphCase.operationIndex,
          operationLabel: graphCase.label,
          operationKind: "conv-transpose1d",
          logicalOutputBase: range.base,
          logicalOutputCount: range.count,
        });
        expect(quantum.primitives).toHaveLength(1);
        expect(quantum.primitives[0]).toMatchObject({
          physicalPartIndex: 0,
          firstOutputChannel: 0,
          outputChannels: graphCase.shape.outputChannels,
          outputBase: range.base,
          outputCount: range.count,
        });
        expect(planAceFp16VaeConvTranspose1dRange(plan, range)).toMatchObject({
          base: range.base,
          count: range.count,
          firstOutputRow: range.firstOutputRow,
          outputRowCount: range.outputRowCount,
        });
        cursor += range.count;
        quantumCount += 1;
      }
      expect(cursor).toBe(plan.outputElements);
    }
    expect(quantumCount).toBe(322);
  });

  it("pins bounded representative canonical ranges and their containing quanta", () => {
    expect(OPT_0011_PRODUCTION_CONV_TRANSPOSE1D_CASES.map(({ id }) => id))
      .toEqual([
        "block-0-stride10-left-padding",
        "block-1-stride6-interior",
        "block-2-stride4-right-padding",
        "block-3-stride4-time-tail",
        "block-4-stride2-longest-tail",
        "arithmetic-stride6-cin65-cout9",
      ]);
    expect(OPT_0011_PRODUCTION_CONV_TRANSPOSE1D_CASES.map(({ ranges }) =>
      ranges[0]
    )).toEqual([
      { base: 0, count: 1_024 },
      { base: 3_932_160, count: 512 },
      { base: 15_728_384, count: 256 },
      { base: 1_580_160, count: 2_176 },
      { base: 62_913_664, count: 896 },
      { base: 0, count: 918 },
    ]);
    for (const fixture of OPT_0011_PRODUCTION_CONV_TRANSPOSE1D_CASES.slice(0, 5)) {
      const graphCase =
        OPT_0011_PRODUCTION_CONV_TRANSPOSE1D_GRAPH_CASES[
          fixture.graphOperationOrdinal!
        ]!;
      const containing = graphCase.ranges.find(
        ({ quantumIndex }) =>
          quantumIndex === fixture.containingGraphQuantumIndex,
      )!;
      const range = fixture.ranges[0]!;
      expect(range.base).toBeGreaterThanOrEqual(containing.base);
      expect(range.base + range.count)
        .toBeLessThanOrEqual(containing.base + containing.count);
      expect(fixture.shape).toEqual(graphCase.shape);
      expect(planAceFp16VaeConvTranspose1dRange(
        planAceFp16VaeConvTranspose1d(fixture.shape),
        range,
      )).toMatchObject({
        base: range.base,
        count: range.count,
        batch: 0,
      });
    }
  });

  it("pins the complete Cin65/Cout9 arithmetic stride, padding, and tail domain", () => {
    const fixture = OPT_0011_PRODUCTION_CONV_TRANSPOSE1D_CASES.at(-1)!;
    expect(fixture).toMatchObject({
      id: "arithmetic-stride6-cin65-cout9",
      graphOperationIndex: null,
      shape: {
        batch: 1,
        inputFrames: 17,
        inputChannels: 65,
        outputChannels: 9,
        kernelSize: 12,
        stride: 6,
        dilation: 1,
        padding: 3,
        outputPadding: 0,
      },
      ranges: [{ base: 0, count: 918 }],
    });
    expect(planAceFp16VaeConvTranspose1d(fixture.shape)).toMatchObject({
      outputFrames: 102,
      inputChannelChunkCount: 2,
      inputBindingBytes: 2_212,
      weightBindingBytes: 14_040,
      biasBindingBytes: 20,
      outputBindingBytes: 1_836,
    });
    expect(planAceFp16VaeConvTranspose1dRange(
      planAceFp16VaeConvTranspose1d(fixture.shape),
      fixture.ranges[0]!,
    )).toMatchObject({
      outputRowCount: 102,
      workgroupsX: 7,
      workgroupsY: 2,
    });
  });

  it("freezes every generated production WGSL hash", () => {
    const actual = Object.fromEntries(
      OPT_0011_PRODUCTION_CONV_TRANSPOSE1D_CASES.map((fixture) => [
        fixture.id,
        sha256(aceFp16VaeConvTranspose1dWgsl(fixture.shape)),
      ]),
    );
    expect(OPT_0011_PRODUCTION_CONV_TRANSPOSE1D_GENERATED_SHADER_SHA256)
      .toEqual(actual);
    for (const fixture of OPT_0011_PRODUCTION_CONV_TRANSPOSE1D_CASES) {
      const source = aceFp16VaeConvTranspose1dWgsl(fixture.shape);
      expect(source).toContain("enable f16;");
      expect(source).toContain(
        "sum = sum + input_operand * weight_operand;",
      );
      expect(source).toContain(
        "if ((input_numerator % STRIDE) == 0u)",
      );
      expect(source).toContain("output: array<f16>;");
      expect(source).not.toContain("fma(");
      expect(actual[fixture.id]).toMatch(/^[0-9a-f]{64}$/);
    }
  });

  it("pins independent CPU raw-U16 arithmetic and conversion boundaries", () => {
    const arithmetic = OPT_0011_PRODUCTION_CONV_TRANSPOSE1D_CASES.at(-1)!;
    expect(Array.from({ length: 18 }, (_, index) =>
      opt0011ProductionConvTranspose1dCpuBits(arithmetic, index)
    )).toEqual([
      0xbecf, 0xbc54, 0xb80d, 0xbe1c, 0xbdfa, 0xb887,
      0xbba3, 0xbef7, 0xbc14, 0x3ab2, 0x393e, 0x3f42,
      0x3d4f, 0x3533, 0x3cf1, 0x3e9d, 0x3a62, 0x39be,
    ]);
    expect(Object.is(float16BitsToNumber(0x8000), -0)).toBe(true);
    expect(numberToFloat16Bits(float16BitsToNumber(0x0001))).toBe(0x0001);
    expect(numberToFloat16Bits(1 + 2 ** -11)).toBe(0x3c00);
    expect(numberToFloat16Bits(1 + 3 * 2 ** -11)).toBe(0x3c02);
  });

  it("retains the first raw mismatch and rejects length substitution", () => {
    expect(compareOpt0011ProductionConvTranspose1dRawBits(
      new Uint16Array([0, 1, 2, 3]),
      new Uint16Array([0, 9, 2, 8]),
    )).toEqual({ mismatchCount: 2, firstMismatchIndex: 1 });
    expect(() => compareOpt0011ProductionConvTranspose1dRawBits(
      new Uint16Array([0]),
      new Uint16Array([0, 1]),
    )).toThrow(/output lengths differ/);
  });

  it("publishes bounded restartable chunks without splitting surrogate pairs", () => {
    expect(OPT_0011_PRODUCTION_CONV_TRANSPOSE1D_RAW_RESULT_CHUNK_CODE_UNITS)
      .toBe(100_000);
    const prefix = "a".repeat(
      OPT_0011_PRODUCTION_CONV_TRANSPOSE1D_RAW_RESULT_CHUNK_CODE_UNITS - 1,
    );
    const raw = `${prefix}\ud83d\ude80tail`;
    const first = sliceOpt0011ProductionConvTranspose1dRawResultChunk(raw, 0);
    expect(first).toEqual({
      chunk: prefix,
      start: 0,
      end: prefix.length,
      nextOffset: prefix.length,
      totalCodeUnits: raw.length,
      complete: false,
    });
    const second = sliceOpt0011ProductionConvTranspose1dRawResultChunk(
      raw,
      first.nextOffset,
    );
    expect(first.chunk + second.chunk).toBe(raw);
    expect(second.complete).toBe(true);
    expect(sliceOpt0011ProductionConvTranspose1dRawResultChunk(raw, 0).chunk)
      .toBe(prefix);
    expect(() => sliceOpt0011ProductionConvTranspose1dRawResultChunk(
      raw,
      prefix.length + 1,
    )).toThrow(/surrogate/);
    expect(parseOpt0011ProductionConvTranspose1dRawResultChunkOffset("0"))
      .toBe(0);
    for (const value of ["", "-1", "+1", "01", "1.0", "1e3"]) {
      expect(() =>
        parseOpt0011ProductionConvTranspose1dRawResultChunkOffset(value)
      ).toThrow(/offset/);
    }
  });

  it("requires the exact core identity and pins its committed source bytes", () => {
    const valid = new URLSearchParams({
      harnessCommit: "1234567890abcdef1234567890abcdef12345678",
      coreCommit: OPT_0011_PRODUCTION_CONV_TRANSPOSE1D_CORE_COMMIT,
    });
    expect(parseOpt0011ProductionConvTranspose1dRunIdentity(valid)).toEqual({
      harnessCommit: "1234567890abcdef1234567890abcdef12345678",
      coreCommit: OPT_0011_PRODUCTION_CONV_TRANSPOSE1D_CORE_COMMIT,
    });
    valid.set("coreCommit", "0000000000000000000000000000000000000000");
    expect(() => parseOpt0011ProductionConvTranspose1dRunIdentity(valid))
      .toThrow(/coreCommit changed/);
    valid.set(
      "coreCommit",
      OPT_0011_PRODUCTION_CONV_TRANSPOSE1D_CORE_COMMIT,
    );
    valid.delete("harnessCommit");
    expect(() => parseOpt0011ProductionConvTranspose1dRunIdentity(valid))
      .toThrow(/requires one harnessCommit/);

    expect(OPT_0011_PRODUCTION_CONV_TRANSPOSE1D_CORE_COMMIT)
      .toBe("d2bf0819d0460f6bd60ebe0457eb091b45e7bf6a");
    const source = readFileSync(new URL(
      "../src/webgpu/kernels/vae-conv-transpose1d-fp16.ts",
      import.meta.url,
    ));
    expect(createHash("sha256").update(source).digest("hex"))
      .toBe(OPT_0011_PRODUCTION_CONV_TRANSPOSE1D_CORE_SOURCE_SHA256);
  });

  it("preserves the primary failure if heartbeat stopping also fails", () => {
    const stopped =
      stopOpt0011ProductionConvTranspose1dHeartbeatAfterFailure({
        stop(): never {
          throw new Error("synthetic heartbeat stop failure");
        },
      });
    expect(stopped.responsiveness).toBeNull();
    expect(stopped.heartbeatStopError).toMatchObject({
      name: "Error",
      message: "synthetic heartbeat stop failure",
    });
    expect(stopOpt0011ProductionConvTranspose1dHeartbeatAfterFailure({
      stop: () => Object.freeze({ observed: false }),
    })).toEqual({
      responsiveness: { observed: false },
      heartbeatStopError: null,
    });
  });

  it("statically binds source auth, CPU bits, guards, reruns, cancellation, and cleanup", () => {
    expect(HARNESS_SOURCE).toContain('requiredFeatures: ["shader-f16"]');
    expect(HARNESS_SOURCE).toContain("productionCoreSource");
    expect(HARNESS_SOURCE).toContain("generated shader SHA-256 changed");
    expect(HARNESS_SOURCE).toContain("OUTPUT_GUARD_F16 = 0x7e33");
    expect(HARNESS_SOURCE).toContain("OUTPUT_CANARY_F16 = 0x7e11");
    expect(HARNESS_SOURCE).toContain("OUTPUT_PREFILL_QNAN_F16 = 0x7e55");
    expect(HARNESS_SOURCE).toContain("SOURCE_PADDING_F16 = 0x7e77");
    expect(HARNESS_SOURCE).toContain("before / FLOAT16_BYTES");
    expect(HARNESS_SOURCE).toContain("after / FLOAT16_BYTES");
    expect(HARNESS_SOURCE).toContain(
      "completeSelectedRangeRawU16Comparison: true",
    );
    expect(HARNESS_SOURCE).toContain("exactOperationRangeTopology");
    expect(HARNESS_SOURCE).toContain("deterministicRerunHashes: true");
    expect(HARNESS_SOURCE).toContain(
      "opt0011ProductionConvTranspose1dCpuBits(",
    );
    expect(HARNESS_SOURCE).toContain("await device.queue.onSubmittedWorkDone()");
    expect(HARNESS_SOURCE).toContain("await queueEmptyIdleTurn()");
    expect(HARNESS_SOURCE).toContain("readbackCount !== 0");
    expect(HARNESS_SOURCE).toContain("laterEncodingPrevented: true");
    expect(HARNESS_SOURCE).toContain("laterSubmissionPrevented: true");
    expect(HARNESS_SOURCE).toContain("readbackPrevented: true");
    expect(HARNESS_SOURCE).toContain('addEventListener("uncapturederror"');
    expect(HARNESS_SOURCE).toContain("rawDevice.lost.then");
    expect(HARNESS_SOURCE.match(/tracker\.destroyAll\(\);/g)).toHaveLength(2);
    expect(HARNESS_SOURCE).toContain(
      "prepared.destroy();\n    prepared.destroy();",
    );
    const heartbeatStart = HARNESS_SOURCE.indexOf(
      "const heartbeat = startHeartbeat();",
    );
    const sourceAuthentication = HARNESS_SOURCE.indexOf(
      "const sourceAuthority = await authenticateSources(identity);",
    );
    const deviceDestroy = HARNESS_SOURCE.indexOf("rawDevice.destroy();");
    const intentionalLoss = HARNESS_SOURCE.indexOf(
      "const intentionalDeviceLoss = await rawDevice.lost;",
    );
    const eventSnapshot = HARNESS_SOURCE.indexOf(
      "const finalEventSnapshotAtEpochMilliseconds",
    );
    const listenersRemoved = HARNESS_SOURCE.indexOf(
      "const eventListenersRemovedAtEpochMilliseconds",
    );
    const heartbeatStop = HARNESS_SOURCE.indexOf(
      "stopOpt0011ProductionConvTranspose1dHeartbeatAfterFailure(heartbeat);",
      listenersRemoved,
    );
    expect(heartbeatStart).toBeGreaterThan(0);
    expect(heartbeatStart).toBeLessThan(sourceAuthentication);
    expect(deviceDestroy).toBeLessThan(intentionalLoss);
    expect(intentionalLoss).toBeLessThan(eventSnapshot);
    expect(eventSnapshot).toBeLessThan(listenersRemoved);
    expect(listenersRemoved).toBeLessThan(heartbeatStop);
    expect(HARNESS_SOURCE).toContain(
      'intentionalDeviceLoss.reason === "destroyed"',
    );
    expect(HARNESS_SOURCE).toContain("responsiveness.observed === true");
    expect(HARNESS_SOURCE).toContain("performanceClaim: null");
    expect(HARNESS_SOURCE).toContain("thermalClaim: null");
    expect(HARNESS_SOURCE).toContain("qualityClaim: null");
    expect(HARNESS_SOURCE).toContain("productionSelectorClaim: null");
  });

  it("keeps the immutable full receipt out of DOM with retrieval ready from page start", () => {
    expect(HARNESS_SOURCE).toContain(
      '"__ACE_OPT_0011_PRODUCTION_CONV_TRANSPOSE1D_RAW_RESULT_JSON__"',
    );
    expect(HARNESS_SOURCE).toContain("installRawResultChunkRetrieval();");
    expect(HARNESS_SOURCE.indexOf("installRawResultChunkRetrieval();"))
      .toBeLessThan(HARNESS_SOURCE.indexOf("start.addEventListener"));
    expect(HARNESS_SOURCE).toContain("Reflect.defineProperty(globalThis");
    expect(HARNESS_SOURCE).toContain("configurable: false");
    expect(HARNESS_SOURCE).toContain("enumerable: false");
    expect(HARNESS_SOURCE).toContain("writable: false");
    expect(HARNESS_SOURCE).toContain("fullReceiptIntentionallyKeptOutOfDom: true");
    expect(HARNESS_SOURCE).toContain(
      'rawResultRetrieval: "bounded-restartable-dom-chunks-from-page-start"',
    );
    expect(HARNESS_SOURCE).toContain(
      "output.dataset.publicationSequence = String(++publicationSequence)",
    );
    expect(HARNESS_SOURCE).toContain("output.textContent = slice.chunk");
    expect(HARNESS_SOURCE).toContain('offsetInput.value = "0"');
    expect(HARNESS_SOURCE).toContain("publish.disabled = false");
    expect(HARNESS_SOURCE).not.toContain("publish.disabled = slice.complete");
    expect(HARNESS_SOURCE).not.toContain("innerHTML");
    expect(HTML_SOURCE).toContain('id="raw-result-retrieval"');
    expect(HTML_SOURCE).not.toContain(
      'id="raw-result-retrieval" hidden disabled',
    );
    expect(HTML_SOURCE).toContain("Available from page start");
    expect(HTML_SOURCE).toContain('name="rawResultOffset"');
    expect(HTML_SOURCE).toContain('id="publish-raw-result-chunk"');
  });

  it("labels the page as bounded correctness only", () => {
    expect(HTML_SOURCE).toContain("Correctness only");
    expect(HTML_SOURCE).toContain("all five");
    expect(HTML_SOURCE).toContain("all 322 unchanged graph quanta");
    expect(HTML_SOURCE).toContain("Every selected raw FP16 output bit");
    expect(HTML_SOURCE).toContain("no timing");
    expect(HTML_SOURCE).toContain(
      "opt-0011-vae-conv-transpose1d-fp16-production.ts",
    );
  });
});

function sha256(source: string): string {
  return createHash("sha256").update(source).digest("hex");
}
