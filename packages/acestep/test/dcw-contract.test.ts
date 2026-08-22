import { describe, expect, it } from "vitest";

import {
  ACE_DIRECT_DCW_CONFIGURATION,
  ACE_THINKING_DCW_CONFIGURATION,
} from "../src/api.js";
import {
  AceCorrectnessDcwKernel,
  aceCorrectnessDcwWgsl,
  deriveAceDcwStepScales,
  planAceDcw,
} from "../src/webgpu/kernels/dcw.js";

describe("ACE Haar DCW contract", () => {
  it("plans odd latent lengths as zero-padded pairs without storing padding", () => {
    expect(planAceDcw({ batch: 1, time: 325, channels: 64 })).toEqual({
      batch: 1,
      time: 325,
      channels: 64,
      timePairs: 163,
      pairChannels: 10_432,
      elements: 20_800,
      workgroupsX: 41,
      workgroupsY: 1,
    });
  });

  it("resolves complementary direct and Think band schedules", () => {
    expect(deriveAceDcwStepScales(ACE_DIRECT_DCW_CONFIGURATION, 1)).toEqual({
      currentTimestep: 1,
      lowBandScale: Math.fround(0.05),
      highBandScale: 0,
    });
    expect(deriveAceDcwStepScales(ACE_DIRECT_DCW_CONFIGURATION, 0)).toEqual({
      currentTimestep: 0,
      lowBandScale: 0,
      highBandScale: Math.fround(0.02),
    });
    expect(deriveAceDcwStepScales(ACE_THINKING_DCW_CONFIGURATION, 0.3)).toEqual({
      currentTimestep: 0.3,
      lowBandScale: Math.fround(0.3 * 0.02),
      highBandScale: Math.fround(0.7 * 0.06),
    });
  });

  it("emits FP32 Haar analysis, correction, and odd-length crop", () => {
    const source = aceCorrectnessDcwWgsl(
      "reference-bf16",
      { batch: 1, time: 325, channels: 64 },
      { lowBandScale: 0.015, highBandScale: 0.014 },
    );
    expect(source).toContain("INV_SQRT_TWO");
    expect(source).toContain("var x_odd = 0.0");
    expect(source).toContain("if (odd_time < TIME)");
    expect(source).toContain("x_low = x_low + LOW_SCALE * (x_low - y_low)");
    expect(source).not.toContain("enable f16");
  });

  it("casts only reconstructed samples back to the raw-FP16 latent", () => {
    const source = aceCorrectnessDcwWgsl(
      "raw-fp16",
      { batch: 1, time: 300, channels: 64 },
      { lowBandScale: 0.05, highBandScale: 0 },
    );
    expect(source).toContain("enable f16");
    expect(source).toContain("let x_even = f32(stepped[even_index])");
    expect(source).toContain("output[even_index] = f16(corrected_even)");
  });

  it.each([
    { batch: 0, time: 1, channels: 1 },
    { batch: 1, time: 0, channels: 1 },
    { batch: 1, time: 1, channels: -1 },
  ])("rejects malformed DCW geometry", (shape) => {
    expect(() => planAceDcw(shape)).toThrow();
  });

  it("rejects unknown profiles and out-of-contract scheduling", () => {
    expect(() =>
      aceCorrectnessDcwWgsl(
        "future" as never,
        { batch: 1, time: 1, channels: 1 },
        { lowBandScale: 0, highBandScale: 0 },
      ),
    ).toThrow(/Unknown ACE DCW model profile/);
    expect(() => deriveAceDcwStepScales(ACE_DIRECT_DCW_CONFIGURATION, 1.1)).toThrow();
  });

  it.each(["steppedLatent", "predictedCleanLatent"] as const)(
    "rejects output aliasing the %s input",
    async (aliasedInput) => {
      const kernel = AceCorrectnessDcwKernel.create(
        fakeDcwDevice(),
        "reference-bf16",
      );
      const shared = { size: 128 } as GPUBuffer;
      const bindings = {
        steppedLatent: fakeBinding(64),
        predictedCleanLatent: fakeBinding(64),
        output: { buffer: shared, offset: 32, size: 16 },
      } satisfies Record<string, GPUBufferBinding>;
      bindings[aliasedInput] = { buffer: shared, offset: 40, size: 32 };
      try {
        await expect(kernel.createDispatch(
          `dcw-alias-${aliasedInput}`,
          { batch: 1, time: 2, channels: 2 },
          ACE_DIRECT_DCW_CONFIGURATION,
          1,
          bindings,
        )).rejects.toThrow(/output must not overlap/);
      } finally {
        kernel.destroy();
      }
    },
  );
});

function fakeBinding(size: number): GPUBufferBinding {
  return { buffer: { size } as GPUBuffer, offset: 0, size };
}

function fakeDcwDevice(): GPUDevice {
  return {
    features: new Set<GPUFeatureName>(),
    limits: {
      maxComputeInvocationsPerWorkgroup: 256,
      maxComputeWorkgroupSizeX: 256,
    },
  } as unknown as GPUDevice;
}
