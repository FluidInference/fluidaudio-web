import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

import {
  aceOpt0011VaeConv1dFp16PortableWorkgroupWgsl,
  aceOpt0011VaeConv1dFp16ScalarOracleWgsl,
  planAceOpt0011VaeConv1dFp16,
} from "../benchmark/opt-0011-vae-conv1d-fp16.js";
import {
  aceFp16VaeConv1dWgsl,
  planAceFp16VaeConv1d,
  planAceFp16VaeConv1dRange,
} from "../src/webgpu/kernels/vae-conv1d-fp16.js";
import {
  planAceVaeDecoder,
  planAceVaeDecoderQuanta,
  type AceVaeDecoderConvOperation,
} from "../src/webgpu/vae-decoder.js";
import {
  OPT_0011_AUDITED_CONV1D_CORE_COMMIT,
  OPT_0011_AUDITED_CONV1D_CORE_SOURCE_SHA256,
  OPT_0011_PRODUCTION_CONV1D_CASES,
  OPT_0011_PRODUCTION_CONV1D_CORE_COMMIT,
  OPT_0011_PRODUCTION_CONV1D_CORE_SOURCE_SHA256,
  OPT_0011_PRODUCTION_CONV1D_GENERATED_SHADER_SHA256,
  OPT_0011_PRODUCTION_K1_SCALAR_ORACLE_ID,
  compareOpt0011ProductionRawBits,
  float16BitsToNumber,
  numberToFloat16Bits,
  opt0011ProductionInputBits,
  opt0011ProductionK1CpuBits,
  opt0011ProductionK1ScalarOracleWgsl,
  parseOpt0011ProductionConv1dRunIdentity,
} from "./browser/opt-0011-vae-conv1d-fp16-production.js";

describe("OPT-0011 production FP16 Conv1D actual-browser contract", () => {
  it("pins four promoted K7 and all five distinct biased-K1 B-256 shapes", () => {
    expect(OPT_0011_PRODUCTION_CONV1D_CASES.map(({ id }) => id)).toEqual([
      "k7-block0-d1-first",
      "k7-block0-d3-middle",
      "k7-block0-d9-last",
      "k7-final-last",
      "k1-block0-tail",
      "k1-block1-tail",
      "k1-block2-tail",
      "k1-block3-middle",
      "k1-block4-last",
      "k1-arithmetic-b1-f17-c65-c9",
    ]);
    const graph = planAceVaeDecoder(256);
    const k1Operations = graph.operations.filter(
      (operation): operation is AceVaeDecoderConvOperation =>
        operation.kind === "conv1d" && operation.shape.kernelSize === 1,
    );
    expect(k1Operations).toHaveLength(15);
    expect(new Set(k1Operations.map(({ shape }) =>
      `${shape.inputFrames}:${shape.inputChannels}:${shape.outputChannels}`
    ))).toEqual(new Set([
      "2560:1024:1024",
      "15360:512:512",
      "61440:256:256",
      "245760:128:128",
      "491520:128:128",
    ]));
    const k1GraphFixtures = OPT_0011_PRODUCTION_CONV1D_CASES.filter(
      ({ family, graphOperationLabel }) =>
        family === "k1" && graphOperationLabel !== null,
    );
    expect(k1GraphFixtures).toHaveLength(5);
    expect(k1GraphFixtures.map(({ graphFamilyMultiplicity }) =>
      graphFamilyMultiplicity
    )).toEqual([3, 3, 3, 3, 3]);
    expect(k1GraphFixtures.every(
      ({ cpuOracleScope }) => cpuOracleScope === "complete-selected-range",
    )).toBe(true);
    expect(k1GraphFixtures.map(({ shape }) =>
      `${shape.inputFrames}:${shape.inputChannels}:${shape.outputChannels}`
    )).toEqual([
      "2560:1024:1024",
      "15360:512:512",
      "61440:256:256",
      "245760:128:128",
      "491520:128:128",
    ]);
    for (const fixtureValue of k1GraphFixtures) {
      const channels = fixtureValue.shape.inputChannels;
      for (const channel of [0, Math.floor(channels / 2), channels - 1]) {
        expect(opt0011ProductionInputBits(fixtureValue, channel)).toBe(
          opt0011ProductionInputBits(fixtureValue, 16 * channels + channel),
        );
      }
    }
  });

  it("pins the selected ranges as exact unchanged B-256 graph quanta", () => {
    const graph = planAceVaeDecoder(256);
    const cooperative = planAceVaeDecoderQuanta(graph);
    const expected = [
      ["k7-block0-d1-first", 51, 0, 0, 32_768],
      ["k7-block0-d3-middle", 192, 40, 1_310_720, 32_768],
      ["k7-block0-d9-last", 332, 79, 2_588_672, 32_768],
      ["k7-final-last", 3_941, 3, 786_432, 196_608],
      ["k1-block0-tail", 145, 11, 2_523_136, 98_304],
      ["k1-block1-tail", 738, 17, 7_798_784, 65_536],
      ["k1-block2-tail", 1_519, 17, 15_597_568, 131_072],
      ["k1-block3-middle", 1_814, 15, 15_728_640, 1_048_576],
      ["k1-block4-last", 3_817, 59, 61_865_984, 1_048_576],
    ] as const;
    for (const [id, quantumIndex, operationQuantumIndex, base, count] of expected) {
      const fixtureValue = OPT_0011_PRODUCTION_CONV1D_CASES.find(
        (candidate) => candidate.id === id,
      )!;
      expect(fixtureValue).toMatchObject({
        graphQuantumIndex: quantumIndex,
        graphOperationQuantumIndex: operationQuantumIndex,
        range: { base, count },
      });
      const quantum = cooperative.quanta[quantumIndex]!;
      expect(quantum.operationLabel).toBe(fixtureValue.graphOperationLabel);
      expect(quantum.primitives).toHaveLength(1);
      expect(quantum.primitives[0]).toMatchObject({
        outputBase: base,
        outputCount: count,
      });
      const plan = planAceFp16VaeConv1d(
        fixtureValue.shape,
        fixtureValue.outputStorage,
      );
      expect(planAceFp16VaeConv1dRange(plan, fixtureValue.range))
        .toMatchObject({ base, count });
    }
  });

  it("pins K7 audit topology and the final no-bias FP32 boundary", () => {
    const k7 = OPT_0011_PRODUCTION_CONV1D_CASES.filter(
      ({ family }) => family === "k7",
    );
    expect(k7.map(({ shape }) => shape.dilation)).toEqual([1, 3, 9, 1]);
    expect(k7.map(({ auditedRangeIndex }) => auditedRangeIndex))
      .toEqual([0, 40, 79, 3]);
    for (const fixtureValue of k7) {
      const audited = planAceOpt0011VaeConv1dFp16(
        fixtureValue.shape,
        fixtureValue.outputStorage,
      );
      expect(audited.outputRanges[fixtureValue.auditedRangeIndex!])
        .toMatchObject({
          firstOutput: fixtureValue.range.base,
          outputCount: fixtureValue.range.count,
        });
    }
    expect(k7.at(-1)).toMatchObject({
      id: "k7-final-last",
      hasBias: false,
      outputStorage: "float32",
      shape: {
        batch: 1,
        inputFrames: 491_520,
        inputChannels: 128,
        outputChannels: 2,
        kernelSize: 7,
        dilation: 1,
        padding: 3,
      },
    });
  });

  it("freezes production, authority, and audited-portable generated shader hashes", () => {
    for (const fixtureValue of OPT_0011_PRODUCTION_CONV1D_CASES) {
      const production = aceFp16VaeConv1dWgsl(
        fixtureValue.shape,
        fixtureValue.hasBias,
        fixtureValue.outputStorage,
      );
      const authority = fixtureValue.family === "k7"
        ? aceOpt0011VaeConv1dFp16ScalarOracleWgsl(
            fixtureValue.shape,
            fixtureValue.hasBias,
            fixtureValue.outputStorage,
          )
        : opt0011ProductionK1ScalarOracleWgsl(fixtureValue.shape);
      const hashes =
        OPT_0011_PRODUCTION_CONV1D_GENERATED_SHADER_SHA256[fixtureValue.id];
      expect(hashes.production).toMatch(/^[0-9a-f]{64}$/);
      expect(hashes.authority).toMatch(/^[0-9a-f]{64}$/);
      expect(hashes.production).not.toContain("PENDING");
      expect(sha256(production)).toBe(hashes.production);
      expect(sha256(authority)).toBe(hashes.authority);
      expect(production).toContain("enable f16;");
      expect(authority).toContain("enable f16;");
      expect(production).toContain(
        "sum = sum + input_operand * weight_operand;",
      );
      expect(authority).toContain(
        "sum = sum + input_operand * weight_operand;",
      );
      expect(production).not.toContain("fma(");
      expect(authority).not.toContain("fma(");
      if (fixtureValue.family === "k7") {
        const portable = aceOpt0011VaeConv1dFp16PortableWorkgroupWgsl(
          fixtureValue.shape,
          fixtureValue.hasBias,
          fixtureValue.outputStorage,
        );
        expect("auditedPortable" in hashes).toBe(true);
        expect(sha256(portable)).toBe(
          "auditedPortable" in hashes ? hashes.auditedPortable : undefined,
        );
      } else {
        expect(authority).toContain(
          `// kernel-id: ${OPT_0011_PRODUCTION_K1_SCALAR_ORACLE_ID}`,
        );
        expect(authority).not.toContain("var<workgroup>");
        expect(production).toContain("var<workgroup> input_tile: array<f16");
      }
    }
  });

  it("pins biased K1 signed zero, subnormal, cancellation, and RNE boundaries", () => {
    const fixtureValue = OPT_0011_PRODUCTION_CONV1D_CASES.at(-1)!;
    expect(fixtureValue).toMatchObject({
      id: "k1-arithmetic-b1-f17-c65-c9",
      family: "k1",
      hasBias: true,
      outputStorage: "float16",
      shape: {
        batch: 1,
        inputFrames: 17,
        inputChannels: 65,
        outputChannels: 9,
        kernelSize: 1,
        stride: 1,
        dilation: 1,
        padding: 0,
      },
      range: { base: 0, count: 153 },
      cpuOracleScope: "complete-selected-range",
    });
    const plan = planAceFp16VaeConv1d(fixtureValue.shape, "float16");
    expect(plan).toMatchObject({
      outputStorageBytes: 306,
      outputBindingBytes: 308,
      inputChannelChunkCount: 2,
    });
    expect(Array.from({ length: 9 }, (_, index) =>
      opt0011ProductionK1CpuBits(fixtureValue, index)
    )).toEqual([
      0x8000,
      0x0000,
      0x0001,
      0x1400,
      0x3c00,
      0x3c02,
      0x0401,
      0x07fe,
      0x0000,
    ]);
    expect(Object.is(float16BitsToNumber(0x8000), -0)).toBe(true);
    expect(numberToFloat16Bits(float16BitsToNumber(0x0001))).toBe(0x0001);
    expect(numberToFloat16Bits(1 + 2 ** -11)).toBe(0x3c00);
    expect(numberToFloat16Bits(1 + 3 * 2 ** -11)).toBe(0x3c02);
  });

  it("retains the first raw mismatch and rejects bit-domain substitution", () => {
    expect(compareOpt0011ProductionRawBits(
      new Uint16Array([0, 1, 2, 3]),
      new Uint16Array([0, 9, 2, 8]),
    )).toEqual({ mismatchCount: 2, firstMismatchIndex: 1 });
    expect(() => compareOpt0011ProductionRawBits(
      new Uint16Array([0]),
      new Uint32Array([0]),
    )).toThrow("OPT-0011 production output bit domains differ");
  });

  it("requires exact immutable source identities in the browser URL", () => {
    const valid = new URLSearchParams({
      harnessCommit: "1234567890abcdef1234567890abcdef12345678",
      coreCommit: OPT_0011_PRODUCTION_CONV1D_CORE_COMMIT,
      auditedCoreCommit: OPT_0011_AUDITED_CONV1D_CORE_COMMIT,
    });
    expect(parseOpt0011ProductionConv1dRunIdentity(valid)).toEqual({
      harnessCommit: "1234567890abcdef1234567890abcdef12345678",
      coreCommit: OPT_0011_PRODUCTION_CONV1D_CORE_COMMIT,
      auditedCoreCommit: OPT_0011_AUDITED_CONV1D_CORE_COMMIT,
    });
    valid.set("coreCommit", "0000000000000000000000000000000000000000");
    expect(() => parseOpt0011ProductionConv1dRunIdentity(valid))
      .toThrow(/coreCommit changed/);
    valid.set("coreCommit", OPT_0011_PRODUCTION_CONV1D_CORE_COMMIT);
    valid.delete("harnessCommit");
    expect(() => parseOpt0011ProductionConv1dRunIdentity(valid))
      .toThrow(/requires one harnessCommit/);
  });

  it("pins the committed production and audited source bytes", () => {
    expect(OPT_0011_PRODUCTION_CONV1D_CORE_COMMIT)
      .toBe("75f70f12bdb43ae33b9bd37391b7d49be5aa1704");
    expect(OPT_0011_AUDITED_CONV1D_CORE_COMMIT)
      .toBe("82f0fa4b3d5e676ec9dc967c3563dc9650cc59bd");
    const production = readFileSync(new URL(
      "../src/webgpu/kernels/vae-conv1d-fp16.ts",
      import.meta.url,
    ));
    const audited = readFileSync(new URL(
      "../benchmark/opt-0011-vae-conv1d-fp16.ts",
      import.meta.url,
    ));
    expect(createHash("sha256").update(production).digest("hex"))
      .toBe(OPT_0011_PRODUCTION_CONV1D_CORE_SOURCE_SHA256);
    expect(createHash("sha256").update(audited).digest("hex"))
      .toBe(OPT_0011_AUDITED_CONV1D_CORE_SOURCE_SHA256);
  });

  it("statically binds full selected-bit checks, guards, reruns, cancellation, and cleanup", () => {
    const harness = readFileSync(new URL(
      "./browser/opt-0011-vae-conv1d-fp16-production.ts",
      import.meta.url,
    ), "utf8");
    expect(harness).toContain('requiredFeatures: ["shader-f16"]');
    expect(harness).toContain("productionCoreSource");
    expect(harness).toContain("auditedCoreSource");
    expect(harness).toContain("generated shader SHA-256 changed");
    expect(harness).toContain("AUTHORITY_F16_SENTINEL = 0x7e11");
    expect(harness).toContain("PRODUCTION_F16_SENTINEL = 0x7e22");
    expect(harness).toContain("OUTPUT_GUARD_WORD = 0xa55a_a55a");
    expect(harness).toContain("completeSelectedRangeReadback: true");
    expect(harness).toContain("fullSelectedRangeBitIdentical: true");
    expect(harness).toContain("fullSelectedRangeComparison: true");
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
    const heartbeatStop = harness.indexOf("responsiveness = heartbeat.stop();");
    expect(deviceDestroy).toBeGreaterThan(0);
    expect(deviceDestroy).toBeLessThan(intentionalLoss);
    expect(intentionalLoss).toBeLessThan(cleanupTurns);
    expect(cleanupTurns).toBeLessThan(eventSnapshot);
    expect(eventSnapshot).toBeLessThan(heartbeatStop);
    expect(harness).toContain('intentionalDeviceLoss.reason === "destroyed"');
    expect(harness).toContain("performanceClaim: null");
    expect(harness).toContain("thermalClaim: null");
    expect(harness).toContain("qualityClaim: null");
    expect(harness).toContain("productionSelectorClaim: null");
  });

  it("labels the page as bounded correctness only", () => {
    const html = readFileSync(new URL(
      "./browser/opt-0011-vae-conv1d-fp16-production.html",
      import.meta.url,
    ), "utf8");
    expect(html).toContain("Correctness only");
    expect(html).toContain("bounded exact B-256 decoder graph quanta");
    expect(html).toContain("every selected K7 output bit");
    expect(html).toContain("every selected biased-K1 FP16 output bit");
    expect(html).toContain("no timing");
    expect(html).toContain("opt-0011-vae-conv1d-fp16-production.ts");
  });
});

function sha256(source: string): string {
  return createHash("sha256").update(source).digest("hex");
}
