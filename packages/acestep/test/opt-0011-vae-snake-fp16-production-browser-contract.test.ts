import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

import {
  aceFp16VaeSnakeWgsl,
  planAceFp16VaeSnake,
  planAceFp16VaeSnakeRange,
} from "../src/webgpu/kernels/vae-snake-fp16.js";
import {
  planAceVaeDecoder,
  planAceVaeDecoderQuanta,
} from "../src/webgpu/vae-decoder.js";
import {
  OPT_0011_PRODUCTION_SNAKE_ARITHMETIC_FIXTURES,
  OPT_0011_PRODUCTION_SNAKE_CORE_COMMIT,
  OPT_0011_PRODUCTION_SNAKE_CORE_SOURCE_SHA256,
  OPT_0011_PRODUCTION_SNAKE_GENERATED_SHADER_SHA256,
  OPT_0011_PRODUCTION_SNAKE_GRAPH_CASES,
  OPT_0011_PRODUCTION_SNAKE_GRAPH_TOPOLOGY_SHA256,
  OPT_0011_PRODUCTION_SNAKE_RAW_RESULT_CHUNK_CODE_UNITS,
  OPT_0011_PRODUCTION_SNAKE_SELECTED_GRAPH_FIXTURES,
  compareOpt0011ProductionSnakeRawBits,
  float16BitsToNumber,
  numberToFloat16Bits,
  opt0011ProductionSnakeArithmeticAlphaBits,
  opt0011ProductionSnakeArithmeticBetaBits,
  opt0011ProductionSnakeArithmeticInputBits,
  opt0011ProductionSnakeCpuBits,
  opt0011ProductionSnakeFinalFusedCpuBits,
  parseOpt0011ProductionSnakeRawResultChunkOffset,
  parseOpt0011ProductionSnakeRunIdentity,
  sliceOpt0011ProductionSnakeRawResultChunk,
  stopOpt0011ProductionSnakeHeartbeatAfterFailure,
} from "./browser/opt-0011-vae-snake-fp16-production.js";

const HARNESS_SOURCE = readFileSync(new URL(
  "./browser/opt-0011-vae-snake-fp16-production.ts",
  import.meta.url,
), "utf8");
const HTML_SOURCE = readFileSync(new URL(
  "./browser/opt-0011-vae-snake-fp16-production.html",
  import.meta.url,
), "utf8");

describe("OPT-0011 production FP16 Snake actual-browser contract", () => {
  it("pins all 36 exact B-256 operations and all 813 graph quanta", () => {
    expect(OPT_0011_PRODUCTION_SNAKE_GRAPH_CASES.map((graphCase) => [
      graphCase.operationIndex,
      graphCase.id,
      graphCase.shape.frames,
      graphCase.shape.channels,
      graphCase.ranges.length,
      graphCase.ranges.at(-1)!.count,
    ])).toEqual(EXPECTED_B256_SNAKE_OPERATIONS);

    const graph = planAceVaeDecoder(256);
    const cooperative = planAceVaeDecoderQuanta(graph);
    let quantumCount = 0;
    for (const graphCase of OPT_0011_PRODUCTION_SNAKE_GRAPH_CASES) {
      const operation = graph.operations[graphCase.operationIndex]!;
      expect(operation).toMatchObject({
        kind: "snake",
        label: graphCase.id,
        shape: graphCase.shape,
      });
      const plan = planAceFp16VaeSnake(graphCase.shape);
      let cursor = 0;
      for (const [index, range] of graphCase.ranges.entries()) {
        const quantum = cooperative.quanta[range.quantumIndex]!;
        expect(range.operationQuantumIndex).toBe(index);
        expect(range.base).toBe(cursor);
        expect(quantum).toMatchObject({
          index: range.quantumIndex,
          operationIndex: graphCase.operationIndex,
          operationLabel: graphCase.id,
          operationKind: "snake",
          logicalOutputBase: range.base,
          logicalOutputCount: range.count,
          estimatedMaximumMultiplyAccumulates: 0,
        });
        expect(quantum.primitives).toHaveLength(1);
        expect(quantum.primitives[0]).toMatchObject({
          controlRecordIndex: range.controlRecordIndex,
          firstOutputChannel: 0,
          outputChannels: graphCase.shape.channels,
          outputBase: range.base,
          outputCount: range.count,
        });
        expect(planAceFp16VaeSnakeRange(plan, range)).toMatchObject({
          base: range.base,
          count: range.count,
        });
        cursor += range.count;
        quantumCount += 1;
      }
      expect(cursor).toBe(plan.elements);
    }
    expect(quantumCount).toBe(813);

    const canonical = cooperative.quanta
      .filter((quantum) => quantum.operationKind === "snake")
      .map((quantum) => ({
        index: quantum.index,
        id: quantum.id,
        operationIndex: quantum.operationIndex,
        operationLabel: quantum.operationLabel,
        operationKind: quantum.operationKind,
        logicalOutputBase: quantum.logicalOutputBase,
        logicalOutputCount: quantum.logicalOutputCount,
        estimatedMaximumMultiplyAccumulates:
          quantum.estimatedMaximumMultiplyAccumulates,
        primitives: quantum.primitives,
      }));
    expect(sha256(JSON.stringify(canonical)))
      .toBe(OPT_0011_PRODUCTION_SNAKE_GRAPH_TOPOLOGY_SHA256);
  });

  it("pins eight bounded representative exact ranges across six shape families", () => {
    expect(OPT_0011_PRODUCTION_SNAKE_SELECTED_GRAPH_FIXTURES.map(
      (fixture) => ({
        operationIndex: fixture.graphOperationIndex,
        shape: `${fixture.shape.frames}:${fixture.shape.channels}`,
        ranges: fixture.ranges.map((range) => ({
          operationQuantumIndex: range.operationQuantumIndex,
          base: range.base,
          count: range.count,
        })),
      }),
    )).toEqual([
      {
        operationIndex: 1,
        shape: "256:2048",
        ranges: [{ operationQuantumIndex: 0, base: 0, count: 524_288 }],
      },
      {
        operationIndex: 3,
        shape: "2560:1024",
        ranges: [
          { operationQuantumIndex: 0, base: 0, count: 1_048_576 },
          {
            operationQuantumIndex: 2,
            base: 2_097_152,
            count: 524_288,
          },
        ],
      },
      {
        operationIndex: 22,
        shape: "15360:512",
        ranges: [{
          operationQuantumIndex: 7,
          base: 7_340_032,
          count: 524_288,
        }],
      },
      {
        operationIndex: 42,
        shape: "61440:256",
        ranges: [{
          operationQuantumIndex: 7,
          base: 7_340_032,
          count: 1_048_576,
        }],
      },
      {
        operationIndex: 69,
        shape: "245760:128",
        ranges: [{
          operationQuantumIndex: 29,
          base: 30_408_704,
          count: 1_048_576,
        }],
      },
      {
        operationIndex: 86,
        shape: "491520:128",
        ranges: [
          { operationQuantumIndex: 0, base: 0, count: 1_048_576 },
          {
            operationQuantumIndex: 59,
            base: 61_865_984,
            count: 1_048_576,
          },
        ],
      },
    ]);
    expect(new Set(OPT_0011_PRODUCTION_SNAKE_SELECTED_GRAPH_FIXTURES.map(
      ({ shape }) => `${shape.frames}:${shape.channels}`,
    )).size).toBe(6);
    expect(OPT_0011_PRODUCTION_SNAKE_SELECTED_GRAPH_FIXTURES.reduce(
      (sum, fixture) => sum + fixture.ranges.length,
      0,
    )).toBe(8);
  });

  it("pins complete manageable channel-reuse and odd-tail arithmetic domains", () => {
    expect(OPT_0011_PRODUCTION_SNAKE_ARITHMETIC_FIXTURES).toMatchObject([
      {
        id: "snake-arithmetic-channel-reuse-2x17",
        shape: { batch: 1, frames: 2, channels: 17 },
        ranges: [{ base: 0, count: 34 }],
      },
      {
        id: "snake-arithmetic-odd-tail-257",
        shape: { batch: 1, frames: 1, channels: 257 },
        ranges: [
          { base: 0, count: 256 },
          { base: 256, count: 1 },
        ],
      },
    ]);
    expect(planAceFp16VaeSnake(
      OPT_0011_PRODUCTION_SNAKE_ARITHMETIC_FIXTURES[1]!.shape,
    )).toMatchObject({
      elements: 257,
      inputStorageBytes: 514,
      inputBindingBytes: 516,
      alphaStorageBytes: 514,
      alphaBindingBytes: 516,
      betaStorageBytes: 514,
      betaBindingBytes: 516,
      outputStorageBytes: 514,
      outputBindingBytes: 516,
    });
  });

  it("pins independent FP32-island raw U16 results and contraction stability", () => {
    const bits = Array.from({ length: 16 }, (_, index) => {
      const input = opt0011ProductionSnakeArithmeticInputBits(index, 257);
      const alpha = opt0011ProductionSnakeArithmeticAlphaBits(index);
      const beta = opt0011ProductionSnakeArithmeticBetaBits(index);
      const expected = opt0011ProductionSnakeCpuBits(input, alpha, beta);
      expect(opt0011ProductionSnakeFinalFusedCpuBits(input, alpha, beta))
        .toBe(expected);
      return expected;
    });
    expect(bits).toEqual([
      0x0000, 0x0000, 0x0001, 0x8001,
      0x3c00, 0xbc00, 0x5805, 0x7c00,
      0x7bff, 0xfbff, 0x0400, 0x03ff,
      0x3c01, 0xbc01, 0x3555, 0xb555,
    ]);
    expect(Object.is(float16BitsToNumber(0x8000), -0)).toBe(true);
    expect(numberToFloat16Bits(float16BitsToNumber(0x0001))).toBe(0x0001);
    expect(numberToFloat16Bits(1 + 2 ** -11)).toBe(0x3c00);
    expect(numberToFloat16Bits(1 + 3 * 2 ** -11)).toBe(0x3c02);
    expect(bits).toContain(0x7c00);
    expect(bits).toContain(0x7bff);
    expect(bits).toContain(0xfbff);

    const tieEvenDown = arithmeticTuple(16, 17);
    expect(tieEvenDown).toEqual([0x6936, 0xc770, 0x0000]);
    expect(snakeSourceOrderF32(...tieEvenDown)).toBe(2_669);
    expect(opt0011ProductionSnakeCpuBits(...tieEvenDown)).toBe(0x6936);
    const tieEvenUp = arithmeticTuple(256, 257);
    expect(tieEvenUp).toEqual([0x6887, 0xc74c, 0x0000]);
    expect(snakeSourceOrderF32(...tieEvenUp)).toBe(2_319);
    expect(opt0011ProductionSnakeCpuBits(...tieEvenUp)).toBe(0x6888);
    expect(HARNESS_SOURCE).toContain("roundToNearestEvenBoundaryCount");
    expect(HARNESS_SOURCE).toContain(
      "actual-snake-f16-rne-tie-to-even-up-at-tail",
    );

    for (let index = 0; index < 34; index += 1) {
      expect(opt0011ProductionSnakeArithmeticInputBits(index, 17))
        .toBe(opt0011ProductionSnakeArithmeticInputBits(index % 17, 17));
    }
  });

  it("freezes every generated production WGSL hash", () => {
    const shapes = new Map<string, {
      readonly batch: number;
      readonly frames: number;
      readonly channels: number;
    }>();
    for (const fixture of [
      ...OPT_0011_PRODUCTION_SNAKE_SELECTED_GRAPH_FIXTURES,
      ...OPT_0011_PRODUCTION_SNAKE_ARITHMETIC_FIXTURES,
    ]) {
      shapes.set(
        `${fixture.shape.batch}:${fixture.shape.frames}:${fixture.shape.channels}`,
        fixture.shape,
      );
    }
    const actual = Object.fromEntries([...shapes].map(([key, shape]) => [
      key,
      sha256(aceFp16VaeSnakeWgsl(shape)),
    ]));
    expect(OPT_0011_PRODUCTION_SNAKE_GENERATED_SHADER_SHA256).toEqual(actual);
    for (const shape of shapes.values()) {
      const source = aceFp16VaeSnakeWgsl(shape);
      expect(source).toContain("enable f16;");
      expect(source).toContain("let alpha_value: f32 = exp(alpha_log_scale);");
      expect(source).toContain("let beta_value: f32 = exp(beta_log_scale);");
      expect(source).toContain("let periodic: f32 = sin(alpha_value * value);");
      expect(source).toContain(
        "value + reciprocal_beta * periodic * periodic;",
      );
      expect(source).toContain("output[index] = f16(result);");
      expect(source).not.toContain("fma(");
      expect(source).not.toMatch(/\b(?:clamp|min|max)\s*\(/);
    }
  });

  it("requires the exact core identity and committed source bytes", () => {
    const valid = new URLSearchParams({
      harnessCommit: "1234567890abcdef1234567890abcdef12345678",
      coreCommit: OPT_0011_PRODUCTION_SNAKE_CORE_COMMIT,
    });
    expect(parseOpt0011ProductionSnakeRunIdentity(valid)).toEqual({
      harnessCommit: "1234567890abcdef1234567890abcdef12345678",
      coreCommit: OPT_0011_PRODUCTION_SNAKE_CORE_COMMIT,
    });
    valid.set("coreCommit", "0000000000000000000000000000000000000000");
    expect(() => parseOpt0011ProductionSnakeRunIdentity(valid))
      .toThrow(/coreCommit changed/);
    valid.set("coreCommit", OPT_0011_PRODUCTION_SNAKE_CORE_COMMIT);
    valid.delete("harnessCommit");
    expect(() => parseOpt0011ProductionSnakeRunIdentity(valid))
      .toThrow(/requires one harnessCommit/);

    expect(OPT_0011_PRODUCTION_SNAKE_CORE_COMMIT)
      .toBe("ae2106c9d5834a3cd5cb836cad484665752230e3");
    const source = readFileSync(new URL(
      "../src/webgpu/kernels/vae-snake-fp16.ts",
      import.meta.url,
    ));
    expect(createHash("sha256").update(source).digest("hex"))
      .toBe(OPT_0011_PRODUCTION_SNAKE_CORE_SOURCE_SHA256);
  });

  it("retains first raw mismatch and rejects length substitution", () => {
    expect(compareOpt0011ProductionSnakeRawBits(
      new Uint16Array([0, 1, 2, 3]),
      new Uint16Array([0, 9, 2, 8]),
    )).toEqual({ mismatchCount: 2, firstMismatchIndex: 1 });
    expect(() => compareOpt0011ProductionSnakeRawBits(
      new Uint16Array([0]),
      new Uint16Array([0, 1]),
    )).toThrow(/output lengths differ/);
  });

  it("publishes bounded restartable chunks without splitting surrogate pairs", () => {
    expect(OPT_0011_PRODUCTION_SNAKE_RAW_RESULT_CHUNK_CODE_UNITS).toBe(100_000);
    const prefix = "a".repeat(
      OPT_0011_PRODUCTION_SNAKE_RAW_RESULT_CHUNK_CODE_UNITS - 1,
    );
    const raw = `${prefix}\ud83d\ude80tail`;
    const first = sliceOpt0011ProductionSnakeRawResultChunk(raw, 0);
    expect(first).toEqual({
      chunk: prefix,
      start: 0,
      end: prefix.length,
      nextOffset: prefix.length,
      totalCodeUnits: raw.length,
      complete: false,
    });
    const second = sliceOpt0011ProductionSnakeRawResultChunk(
      raw,
      first.nextOffset,
    );
    expect(first.chunk + second.chunk).toBe(raw);
    expect(second.complete).toBe(true);
    expect(sliceOpt0011ProductionSnakeRawResultChunk(raw, 0).chunk)
      .toBe(prefix);
    expect(() => sliceOpt0011ProductionSnakeRawResultChunk(
      raw,
      prefix.length + 1,
    )).toThrow(/surrogate/);
    expect(parseOpt0011ProductionSnakeRawResultChunkOffset("0")).toBe(0);
    for (const value of ["", "-1", "+1", "01", "1.0", "1e3"]) {
      expect(() => parseOpt0011ProductionSnakeRawResultChunkOffset(value))
        .toThrow(/offset/);
    }
  });

  it("preserves the primary failure if heartbeat stopping also fails", () => {
    const stopped = stopOpt0011ProductionSnakeHeartbeatAfterFailure({
      stop(): never {
        throw new Error("synthetic heartbeat stop failure");
      },
    });
    expect(stopped.liveness).toBeNull();
    expect(stopped.heartbeatStopError).toMatchObject({
      name: "Error",
      message: "synthetic heartbeat stop failure",
    });
    expect(stopOpt0011ProductionSnakeHeartbeatAfterFailure({
      stop: () => Object.freeze({ observed: false }),
    })).toEqual({
      liveness: { observed: false },
      heartbeatStopError: null,
    });
  });

  it("statically binds auth, raw U16, guards, reruns, cancellation, and cleanup", () => {
    expect(HARNESS_SOURCE).toContain('requiredFeatures: ["shader-f16"]');
    expect(HARNESS_SOURCE).toContain("productionCoreSource");
    expect(HARNESS_SOURCE).toContain("generated shader SHA-256 changed");
    expect(HARNESS_SOURCE).toContain(
      "canonical graph topology SHA-256 changed",
    );
    expect(HARNESS_SOURCE).toContain("OUTPUT_GUARD_F16 = 0x7e33");
    expect(HARNESS_SOURCE).toContain("OUTPUT_CANARY_F16 = 0x7e11");
    expect(HARNESS_SOURCE).toContain("OUTPUT_PREFILL_QNAN_F16 = 0x7e55");
    expect(HARNESS_SOURCE).toContain("SOURCE_PADDING_F16 = 0x7e77");
    expect(HARNESS_SOURCE).toContain("prepared.prefill.canaryBuffer");
    expect(HARNESS_SOURCE).toContain("adjacentCanaryRestoreCopies");
    expect(HARNESS_SOURCE).toContain(
      "completeSelectedRangeRawU16Comparison: true",
    );
    expect(HARNESS_SOURCE).toContain("completeOperationAndRangeRecordsIncluded");
    expect(HARNESS_SOURCE).toContain("deterministicRerunHashes: true");
    expect(HARNESS_SOURCE).toContain("fixtureExpectedBits(fixture, globalIndex)");
    expect(HARNESS_SOURCE).toContain("await device.queue.onSubmittedWorkDone()");
    expect(HARNESS_SOURCE).toContain("await queueEmptyIdleTurn()");
    const executeSource = HARNESS_SOURCE.slice(
      HARNESS_SOURCE.indexOf("async function executeAndRead("),
      HARNESS_SOURCE.indexOf("async function readSelectedOutput("),
    );
    expect(executeSource.indexOf("prepared.prefill.canaryBuffer"))
      .toBeLessThan(executeSource.indexOf(".encode(pass)"));
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
    const gpuAvailabilityCheck = HARNESS_SOURCE.indexOf(
      'if (navigator.gpu === undefined)',
    );
    const deviceDestroy = HARNESS_SOURCE.indexOf("rawDevice.destroy();");
    const intentionalLoss = HARNESS_SOURCE.indexOf(
      "const intentionalDeviceLoss = await rawDevice.lost;",
    );
    const eventSnapshot = HARNESS_SOURCE.indexOf(
      'lifecycleOrder.push("final-event-snapshot-captured")',
    );
    const listenersRemoved = HARNESS_SOURCE.indexOf(
      'lifecycleOrder.push("event-listeners-removed")',
    );
    const heartbeatStop = HARNESS_SOURCE.indexOf(
      "stopOpt0011ProductionSnakeHeartbeatAfterFailure(heartbeat);",
      listenersRemoved,
    );
    expect(heartbeatStart).toBeGreaterThan(0);
    expect(heartbeatStart).toBeLessThan(sourceAuthentication);
    expect(sourceAuthentication).toBeLessThan(gpuAvailabilityCheck);
    expect(deviceDestroy).toBeLessThan(intentionalLoss);
    expect(intentionalLoss).toBeLessThan(eventSnapshot);
    expect(eventSnapshot).toBeLessThan(listenersRemoved);
    expect(listenersRemoved).toBeLessThan(heartbeatStop);
    expect(HARNESS_SOURCE).toContain(
      'intentionalDeviceLoss.reason === "destroyed"',
    );
    expect(HARNESS_SOURCE).toContain("heartbeatLiveness.observed === true");
    expect(HARNESS_SOURCE).toContain("performanceClaim: null");
    expect(HARNESS_SOURCE).toContain("thermalClaim: null");
    expect(HARNESS_SOURCE).toContain("qualityClaim: null");
    expect(HARNESS_SOURCE).toContain("productionSelectorClaim: null");
    expect(HARNESS_SOURCE).toContain("productionIntegrationClaim: null");
    expect(HARNESS_SOURCE).toContain(
      "countsAreCorrectnessSignalsNotTimingMeasurements: true",
    );
    expect(HARNESS_SOURCE).not.toContain("performance.now()");
    expect(HARNESS_SOURCE).not.toContain("Date.now()");
    expect(HARNESS_SOURCE).not.toContain("AtEpochMilliseconds");
  });

  it("keeps immutable receipt out of DOM with retrieval ready from page start", () => {
    expect(HARNESS_SOURCE).toContain(
      '"__ACE_OPT_0011_PRODUCTION_SNAKE_RAW_RESULT_JSON__"',
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
    expect(HTML_SOURCE).toContain("all 36 operations");
    expect(HTML_SOURCE).toContain("813 unchanged B-256 graph quanta");
    expect(HTML_SOURCE).toContain("Every selected raw FP16 output bit");
    expect(HTML_SOURCE).toContain(
      "no kernel, performance, or wall-time timing",
    );
    expect(HTML_SOURCE).toContain(
      "opt-0011-vae-snake-fp16-production.ts",
    );
  });
});

const EXPECTED_B256_SNAKE_OPERATIONS = [
  [1, "block-0-snake1", 256, 2_048, 1, 524_288],
  [3, "block-0-res-1-snake1", 2_560, 1_024, 3, 524_288],
  [5, "block-0-res-1-snake2", 2_560, 1_024, 3, 524_288],
  [8, "block-0-res-2-snake1", 2_560, 1_024, 3, 524_288],
  [10, "block-0-res-2-snake2", 2_560, 1_024, 3, 524_288],
  [13, "block-0-res-3-snake1", 2_560, 1_024, 3, 524_288],
  [15, "block-0-res-3-snake2", 2_560, 1_024, 3, 524_288],
  [18, "block-1-snake1", 2_560, 1_024, 3, 524_288],
  [20, "block-1-res-1-snake1", 15_360, 512, 8, 524_288],
  [22, "block-1-res-1-snake2", 15_360, 512, 8, 524_288],
  [25, "block-1-res-2-snake1", 15_360, 512, 8, 524_288],
  [27, "block-1-res-2-snake2", 15_360, 512, 8, 524_288],
  [30, "block-1-res-3-snake1", 15_360, 512, 8, 524_288],
  [32, "block-1-res-3-snake2", 15_360, 512, 8, 524_288],
  [35, "block-2-snake1", 15_360, 512, 8, 524_288],
  [37, "block-2-res-1-snake1", 61_440, 256, 15, 1_048_576],
  [39, "block-2-res-1-snake2", 61_440, 256, 15, 1_048_576],
  [42, "block-2-res-2-snake1", 61_440, 256, 15, 1_048_576],
  [44, "block-2-res-2-snake2", 61_440, 256, 15, 1_048_576],
  [47, "block-2-res-3-snake1", 61_440, 256, 15, 1_048_576],
  [49, "block-2-res-3-snake2", 61_440, 256, 15, 1_048_576],
  [52, "block-3-snake1", 61_440, 256, 15, 1_048_576],
  [54, "block-3-res-1-snake1", 245_760, 128, 30, 1_048_576],
  [56, "block-3-res-1-snake2", 245_760, 128, 30, 1_048_576],
  [59, "block-3-res-2-snake1", 245_760, 128, 30, 1_048_576],
  [61, "block-3-res-2-snake2", 245_760, 128, 30, 1_048_576],
  [64, "block-3-res-3-snake1", 245_760, 128, 30, 1_048_576],
  [66, "block-3-res-3-snake2", 245_760, 128, 30, 1_048_576],
  [69, "block-4-snake1", 245_760, 128, 30, 1_048_576],
  [71, "block-4-res-1-snake1", 491_520, 128, 60, 1_048_576],
  [73, "block-4-res-1-snake2", 491_520, 128, 60, 1_048_576],
  [76, "block-4-res-2-snake1", 491_520, 128, 60, 1_048_576],
  [78, "block-4-res-2-snake2", 491_520, 128, 60, 1_048_576],
  [81, "block-4-res-3-snake1", 491_520, 128, 60, 1_048_576],
  [83, "block-4-res-3-snake2", 491_520, 128, 60, 1_048_576],
  [86, "snake1", 491_520, 128, 60, 1_048_576],
];

function sha256(source: string): string {
  return createHash("sha256").update(source).digest("hex");
}

function arithmeticTuple(
  channel: number,
  channels: number,
): [number, number, number] {
  return [
    opt0011ProductionSnakeArithmeticInputBits(channel, channels),
    opt0011ProductionSnakeArithmeticAlphaBits(channel),
    opt0011ProductionSnakeArithmeticBetaBits(channel),
  ];
}

function snakeSourceOrderF32(
  inputBits: number,
  alphaBits: number,
  betaBits: number,
): number {
  const f32 = Math.fround;
  const input = f32(float16BitsToNumber(inputBits));
  const alpha = f32(float16BitsToNumber(alphaBits));
  const beta = f32(float16BitsToNumber(betaBits));
  const alphaValue = f32(Math.exp(alpha));
  const betaValue = f32(Math.exp(beta));
  const periodic = f32(Math.sin(f32(alphaValue * input)));
  const reciprocalBeta = f32(1 / f32(betaValue + f32(1e-9)));
  return f32(input + f32(f32(reciprocalBeta * periodic) * periodic));
}
