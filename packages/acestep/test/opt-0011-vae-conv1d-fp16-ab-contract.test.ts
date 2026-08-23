import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

import {
  ACE_OPT_0011_VAE_CONV1D_FP16_PORTABLE_WORKGROUP_ID,
  ACE_OPT_0011_VAE_CONV1D_FP16_SCALAR_ORACLE_ID,
  aceOpt0011VaeConv1dFp16PortableWorkgroupWgsl,
  aceOpt0011VaeConv1dFp16ScalarOracleWgsl,
  planAceOpt0011VaeConv1dFp16,
} from "../benchmark/opt-0011-vae-conv1d-fp16.js";
import {
  OPT_0011_VAE_CONV1D_FP16_CORRECTNESS_CASES,
  OPT_0011_VAE_CONV1D_CORE_COMMIT,
  OPT_0011_VAE_CONV1D_CORE_SOURCE_SHA256,
  OPT_0011_VAE_CONV1D_GENERATED_SHADER_SHA256,
  OPT_0011_VAE_CONV1D_UPLOAD_SHA256,
  classifyOpt0011VaeConv1dOutputBits,
  compareOpt0011VaeConv1dRawBits,
  float16BitsToNumber,
  numberToFloat16Bits,
  opt0011VaeConv1dCpuForms,
  opt0011VaeConv1dFixture,
  opt0011VaeConv1dInputBits,
  opt0011VaeConv1dUploadBytes,
  opt0011VaeConv1dUploadRoles,
  opt0011VaeConv1dWeightBits,
} from "./browser/opt-0011-vae-conv1d-fp16-ab.js";

describe("OPT-0011 FP16 VAE Conv1D actual-browser correctness contract", () => {
  it("pins every required endpoint without substituting smaller shapes", () => {
    expect(OPT_0011_VAE_CONV1D_FP16_CORRECTNESS_CASES.map(({ id }) => id))
      .toEqual([
        "d1-b2-f35-c65-c13-bias",
        "d3-b3-f51-c64-c11-no-bias",
        "d9-b2-f67-c63-c9-bias",
        "signed-zero",
        "arithmetic-subnormal-cancellation-range-edge",
        "long-cin1024",
        "final-b1-f4097-c128-c2-f32",
        "fifo-cancellation-two-range",
        "production-block0-d1-c1024",
      ]);
    expect(opt0011VaeConv1dFixture("d1-b2-f35-c65-c13-bias"))
      .toMatchObject({
        hasBias: true,
        outputStorage: "float16",
        shape: shapeReceipt(2, 35, 65, 13, 1, 3),
      });
    expect(opt0011VaeConv1dFixture("d3-b3-f51-c64-c11-no-bias"))
      .toMatchObject({
        hasBias: false,
        outputStorage: "float16",
        shape: shapeReceipt(3, 51, 64, 11, 3, 9),
      });
    expect(opt0011VaeConv1dFixture("d9-b2-f67-c63-c9-bias"))
      .toMatchObject({
        hasBias: true,
        outputStorage: "float16",
        shape: shapeReceipt(2, 67, 63, 9, 9, 27),
      });
    expect(opt0011VaeConv1dFixture("long-cin1024")).toMatchObject({
      shape: { inputChannels: 1_024 },
      outputStorage: "float16",
    });
    expect(opt0011VaeConv1dFixture("final-b1-f4097-c128-c2-f32"))
      .toMatchObject({
        hasBias: false,
        outputStorage: "float32",
        shape: shapeReceipt(1, 4_097, 128, 2, 1, 3),
      });
  });

  it("pins complete range topology, binding padding, and the 80-range block gate", () => {
    const expectedRanges = [2, 3, 2, 1, 1, 1, 1, 2, 80];
    expect(OPT_0011_VAE_CONV1D_FP16_CORRECTNESS_CASES.map((fixture) =>
      planAceOpt0011VaeConv1dFp16(fixture.shape, fixture.outputStorage)
        .outputRangeCount
    )).toEqual(expectedRanges);

    const d3 = planFor("d3-b3-f51-c64-c11-no-bias");
    expect(d3.outputElements).toBe(1_683);
    expect(d3.outputStorageBytes).toBe(3_366);
    expect(d3.outputBindingBytes).toBe(3_368);
    expect(d3.outputRanges.map(({ batch, outputRowCount }) => ({
      batch,
      outputRowCount,
    }))).toEqual([
      { batch: 0, outputRowCount: 51 },
      { batch: 1, outputRowCount: 51 },
      { batch: 2, outputRowCount: 51 },
    ]);

    const arithmetic = planFor(
      "arithmetic-subnormal-cancellation-range-edge",
    );
    expect(arithmetic.outputElements).toBe(65);
    expect(arithmetic.outputStorageBytes).toBe(130);
    expect(arithmetic.outputBindingBytes).toBe(132);

    const cancellation = planFor("fifo-cancellation-two-range");
    expect(cancellation.outputRanges.map(({ outputRowCount }) => outputRowCount))
      .toEqual([1_048_560, 17]);

    const production = planFor("production-block0-d1-c1024");
    expect(production).toMatchObject({
      batch: 1,
      inputFrames: 2_560,
      inputChannels: 1_024,
      outputChannels: 1_024,
      outputElements: 2_621_440,
      outputRangeCount: 80,
      workgroupStorageBytes: 3_216,
    });
    expect(new Set(production.outputRanges.map(({ outputRowCount }) =>
      outputRowCount
    ))).toEqual(new Set([32]));
    expect(production.outputRanges.map(({ firstOutputTime }) => firstOutputTime))
      .toEqual(Array.from({ length: 80 }, (_, index) => index * 32));
  });

  it("freezes exact little-endian upload bytes and SHA-256 identities", () => {
    for (const fixture of OPT_0011_VAE_CONV1D_FP16_CORRECTNESS_CASES) {
      const plan = planAceOpt0011VaeConv1dFp16(
        fixture.shape,
        fixture.outputStorage,
      );
      const roles = opt0011VaeConv1dUploadRoles(fixture);
      expect(roles).toEqual(fixture.hasBias
        ? ["input", "weight", "bias"]
        : ["input", "weight"]);
      for (const role of roles) {
        const bytes = opt0011VaeConv1dUploadBytes(fixture.id, role);
        const expectedBytes = role === "input"
          ? plan.inputBindingBytes
          : role === "weight"
          ? plan.weightBindingBytes
          : plan.biasBindingBytes;
        expect(bytes.byteLength).toBe(expectedBytes);
        const expectedHash = (
          OPT_0011_VAE_CONV1D_UPLOAD_SHA256 as Readonly<Record<
            string,
            Readonly<Partial<Record<string, string>>>
          >>
        )[fixture.id]![role];
        expect(expectedHash).toMatch(/^[0-9a-f]{64}$/);
        expect(expectedHash).not.toContain("PENDING");
        expect(createHash("sha256").update(bytes).digest("hex"))
          .toBe(expectedHash);
      }
    }

    const arithmetic = opt0011VaeConv1dUploadBytes(
      "arithmetic-subnormal-cancellation-range-edge",
      "input",
    );
    const maxFiniteOffset = 6 * 65 * 2;
    expect(Array.from(arithmetic.slice(maxFiniteOffset, maxFiniteOffset + 2)))
      .toEqual([0xff, 0x7b]);
    expect(new DataView(arithmetic.buffer).getUint16(maxFiniteOffset, true))
      .toBe(0x7bff);

    const oddWeight = opt0011VaeConv1dUploadBytes(
      "d1-b2-f35-c65-c13-bias",
      "weight",
    );
    expect(oddWeight.byteLength).toBe(11_832);
    expect(oddWeight.at(-2)).toBe(0);
    expect(oddWeight.at(-1)).toBe(0);
  });

  it("pins signed-zero, subnormal, cancellation, and range-edge discriminants", () => {
    for (let row = 0; row < 7; row += 1) {
      const negative = opt0011VaeConv1dCpuForms("signed-zero", row * 2);
      const positive = opt0011VaeConv1dCpuForms("signed-zero", row * 2 + 1);
      expect(negative.separateRoundedFp32Bits).toBe(0x8000_0000);
      expect(negative.separateStoreBits).toBe(0x8000);
      expect(positive.separateRoundedFp32Bits).toBe(0x0000_0000);
      expect(positive.separateStoreBits).toBe(0x0000);
    }

    const arithmeticId = "arithmetic-subnormal-cancellation-range-edge";
    expect(opt0011VaeConv1dInputBits(arithmeticId, 6 * 65)).toBe(0x7bff);
    expect(opt0011VaeConv1dInputBits(arithmeticId, 6 * 65 + 3)).toBe(0x0001);
    expect(opt0011VaeConv1dInputBits(arithmeticId, 6 * 65 + 4)).toBe(0x0400);
    expect(opt0011VaeConv1dInputBits(arithmeticId, 6 * 65 + 5)).toBe(0x03ff);
    expect(opt0011VaeConv1dWeightBits(arithmeticId, (0 * 7 + 3) * 65))
      .toBe(0x3c00);

    const cancellation = opt0011VaeConv1dCpuForms(arithmeticId, 6 * 5);
    expect(cancellation.separateRoundedFp32Bits).toBe(0x0000_0000);
    expect(cancellation.separateStoreBits).toBe(0x0000);
    const reassociated = Math.fround(
      Math.fround(65_504 + -65_504) + 2 ** -10,
    );
    expect(reassociated).toBe(2 ** -10);

    const subnormal = opt0011VaeConv1dCpuForms(arithmeticId, 6 * 5 + 1);
    expect(subnormal.separateRoundedFp32).toBe(2 ** -24);
    expect(subnormal.separateStoreBits).toBe(0x0001);
    const rangeEdge = opt0011VaeConv1dCpuForms(arithmeticId, 6 * 5 + 2);
    expect(rangeEdge.separateStoreBits).toBe(0x43ff);
    expect(Number.isFinite(rangeEdge.separateRoundedFp32)).toBe(true);

    expect(classifyOpt0011VaeConv1dOutputBits(
      subnormal.separateStoreBits,
      subnormal,
    )).toBe("both-allowed-forms");
    expect(classifyOpt0011VaeConv1dOutputBits(0x0000, subnormal))
      .toBe("unexpected");
    expect(numberToFloat16Bits(float16BitsToNumber(0x8000))).toBe(0x8000);
    expect(numberToFloat16Bits(float16BitsToNumber(0x0001))).toBe(0x0001);
    expect(numberToFloat16Bits(float16BitsToNumber(0x7bff))).toBe(0x7bff);

    const fifoId = "fifo-cancellation-two-range";
    const exactZeroIndex = 176_232;
    expect(Array.from({ length: 7 }, (_, kernel) =>
      opt0011VaeConv1dInputBits(fifoId, exactZeroIndex + kernel - 3)
    )).toEqual([
      0x0000, 0x8000, 0x8000, 0x0000, 0x0000, 0x3400, 0xb555,
    ]);
    expect(Array.from({ length: 7 }, (_, kernel) =>
      opt0011VaeConv1dWeightBits(fifoId, kernel)
    )).toEqual([
      0x9400, 0x2400, 0x1c00, 0xa400, 0x9800, 0x8000, 0x0000,
    ]);
    const exactZero = opt0011VaeConv1dCpuForms(fifoId, exactZeroIndex);
    expect(exactZero).toMatchObject({
      separateRoundedFp32Bits: 0x0000_0000,
      oneRoundContractedFp32Bits: 0x0000_0000,
      separateStoreBits: 0x0000,
      contractedStoreBits: 0x0000,
    });
  });

  it("retains the first raw-bit mismatch while counting every mismatch", () => {
    expect(compareOpt0011VaeConv1dRawBits(
      new Uint16Array([0x0000, 0x3c00, 0x4000, 0x4200]),
      new Uint16Array([0x0000, 0x3c01, 0x4000, 0x4201]),
    )).toEqual({ mismatchCount: 2, firstMismatchIndex: 1 });
    expect(compareOpt0011VaeConv1dRawBits(
      new Uint32Array([0x0000_0000, 0x3f80_0000]),
      new Uint32Array([0x0000_0000, 0x3f80_0000]),
    )).toEqual({ mismatchCount: 0, firstMismatchIndex: null });
    expect(() => compareOpt0011VaeConv1dRawBits(
      new Uint16Array([0]),
      new Uint32Array([0]),
    )).toThrow("OPT-0011 output bit domains differ");
  });

  it("compiles independent scalar and workgroup shader-f16 sources for every case", () => {
    expect(ACE_OPT_0011_VAE_CONV1D_FP16_SCALAR_ORACLE_ID)
      .not.toBe(ACE_OPT_0011_VAE_CONV1D_FP16_PORTABLE_WORKGROUP_ID);
    for (const fixture of OPT_0011_VAE_CONV1D_FP16_CORRECTNESS_CASES) {
      const scalar = aceOpt0011VaeConv1dFp16ScalarOracleWgsl(
        fixture.shape,
        fixture.hasBias,
        fixture.outputStorage,
      );
      const workgroup = aceOpt0011VaeConv1dFp16PortableWorkgroupWgsl(
        fixture.shape,
        fixture.hasBias,
        fixture.outputStorage,
      );
      expect(scalar).not.toBe(workgroup);
      expect(scalar).toContain("enable f16;");
      expect(workgroup).toContain("enable f16;");
      expect(scalar).toContain(`// kernel-id: ${
        ACE_OPT_0011_VAE_CONV1D_FP16_SCALAR_ORACLE_ID
      }`);
      expect(workgroup).toContain(`// kernel-id: ${
        ACE_OPT_0011_VAE_CONV1D_FP16_PORTABLE_WORKGROUP_ID
      }`);
      expect(scalar).not.toContain("var<workgroup>");
      expect(workgroup).toContain("var<workgroup> input_tile: array<f16");
      expect(workgroup).toContain("var<workgroup> weight_tile: array<f16");
      expect(scalar).toContain("sum = sum + input_operand * weight_operand;");
      expect(workgroup).toContain("sum = sum + input_operand * weight_operand;");
      expect(scalar).not.toContain("fma(");
      expect(workgroup).not.toContain("fma(");
      const exactZeroCanonicalization =
        "select(sum, bitcast<f32>(0u), (bitcast<u32>(sum) & 0x7fffffffu) == 0u)";
      if (fixture.hasBias) {
        expect(scalar).not.toContain(exactZeroCanonicalization);
        expect(workgroup).not.toContain(exactZeroCanonicalization);
      } else {
        expect(scalar).toContain(exactZeroCanonicalization);
        expect(workgroup).toContain(exactZeroCanonicalization);
      }
      expect(createHash("sha256").update(scalar).digest("hex"))
        .toBe(OPT_0011_VAE_CONV1D_GENERATED_SHADER_SHA256[fixture.id].scalar);
      expect(createHash("sha256").update(workgroup).digest("hex"))
        .toBe(OPT_0011_VAE_CONV1D_GENERATED_SHADER_SHA256[fixture.id].workgroup);
    }
  });

  it("pins the committed benchmark core source authority", () => {
    expect(OPT_0011_VAE_CONV1D_CORE_COMMIT)
      .toBe("82f0fa4b3d5e676ec9dc967c3563dc9650cc59bd");
    const core = readFileSync(new URL(
      "../benchmark/opt-0011-vae-conv1d-fp16.ts",
      import.meta.url,
    ));
    expect(createHash("sha256").update(core).digest("hex"))
      .toBe(OPT_0011_VAE_CONV1D_CORE_SOURCE_SHA256);
  });

  it("statically binds complete bit checks, guards, reruns, cancellation, and cleanup", () => {
    const harness = readFileSync(new URL(
      "./browser/opt-0011-vae-conv1d-fp16-ab.ts",
      import.meta.url,
    ), "utf8");
    expect(harness).toContain('requiredFeatures: ["shader-f16"]');
    expect(harness).toContain("opt0011VaeConv1dFp16CoreSource");
    expect(harness).toContain("generated shader SHA-256 changed");
    expect(harness).toContain("SCALAR_F16_SENTINEL = 0x7e11");
    expect(harness).toContain("WORKGROUP_F16_SENTINEL = 0x7e22");
    expect(harness).toContain("OUTPUT_GUARD_WORD = 0xa55a_a55a");
    expect(harness).toContain("dispatch.encodeRange(pass, rangeIndex)");
    expect(harness).toContain("await device.queue.onSubmittedWorkDone()");
    expect(harness).toContain("await queueEmptyIdleTurn()");
    expect(harness).toContain("compareOpt0011VaeConv1dRawBits(");
    expect(harness).toContain("firstMismatchIndex=");
    expect(harness).toContain("rangeFirstOutput=");
    expect(harness).toContain("cpuSeparateStoreBits=");
    expect(harness).toContain("cpuContractedStoreBits=");
    expect(harness).toContain(
      "indices.add(FIFO_EXACT_ZERO_CANONICALIZATION_INDEX)",
    );
    expect(harness).toContain("deterministicRerunHashes: true");
    expect(harness).toContain("readbackCount !== 0");
    expect(harness).toContain("laterEncodingPrevented: true");
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
    const eventValidation = harness.indexOf(
      "const postCleanupValidationAtEpochMilliseconds",
    );
    const heartbeatStop = harness.indexOf("responsiveness = heartbeat.stop();");
    const listenerRemoval = harness.indexOf(
      'rawDevice.removeEventListener("uncapturederror", onUncaptured);',
    );
    expect(deviceDestroy).toBeGreaterThan(0);
    expect(deviceDestroy).toBeLessThan(intentionalLoss);
    expect(intentionalLoss).toBeLessThan(cleanupTurns);
    expect(cleanupTurns).toBeLessThan(eventSnapshot);
    expect(eventSnapshot).toBeLessThan(eventValidation);
    expect(eventValidation).toBeLessThan(heartbeatStop);
    expect(heartbeatStop).toBeLessThan(listenerRemoval);
    expect(harness).toContain('intentionalDeviceLoss.reason === "destroyed"');
    expect(harness).toContain(
      'exactDestroyedReason: intentionalDeviceLoss.reason === "destroyed"',
    );
    expect(harness).toContain("heartbeatCoveredCleanup: true");
    expect(harness).toContain("heartbeatStoppedAtEpochMilliseconds");
    expect(harness).toContain("actualAdapterLimits: gpuLimitReceipt");
    expect(harness).toContain("actualRequestedDeviceLimits: gpuLimitReceipt");
    expect(harness).toContain("performanceClaim: null");
    expect(harness).toContain("thermalClaim: null");
  });

  it("labels the page as a correctness-only full 80-range run", () => {
    const html = readFileSync(new URL(
      "./browser/opt-0011-vae-conv1d-fp16-ab.html",
      import.meta.url,
    ), "utf8");
    expect(html).toContain("Correctness only");
    expect(html).toContain("every FP16 output bit as U16");
    expect(html).toContain("every final\n      FP32 output bit as U32");
    expect(html).toContain("all 80 ranges");
    expect(html).toContain("no timing");
    expect(html).toContain("opt-0011-vae-conv1d-fp16-ab.ts");
  });
});

function planFor(
  id: Parameters<typeof opt0011VaeConv1dFixture>[0],
): ReturnType<typeof planAceOpt0011VaeConv1dFp16> {
  const fixture = opt0011VaeConv1dFixture(id);
  return planAceOpt0011VaeConv1dFp16(fixture.shape, fixture.outputStorage);
}

function shapeReceipt(
  batch: number,
  inputFrames: number,
  inputChannels: number,
  outputChannels: number,
  dilation: number,
  padding: number,
): object {
  return {
    batch,
    inputFrames,
    inputChannels,
    outputChannels,
    kernelSize: 7,
    stride: 1,
    dilation,
    padding,
  };
}
