import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

import {
  OPT_0007_ARITHMETIC_SENTINELS,
  OPT_0007_CORRECTNESS_CASES,
  OPT_0007_PAIRED_ORDERS,
  OPT_0007_SCREEN_C1024_SHAPE,
  OPT_0007_SCREEN_C128_SHAPE,
  OPT_0007_SEQUENCE_RANGE_COUNT,
  OPT_0007_SEQUENCE_RANGES_PER_COMMAND_BUFFER,
  expectedOpt0007SequenceCounts,
  middleRangeIndex,
  opt0007ExpectedOutputValue,
  parseOpt0007RunMode,
  parseOpt0007ThermalGateMetadata,
  sequenceC128RangeIndices,
  summarizeOpt0007Samples,
} from "./browser/opt-0007-vae-k1-conv1d-ab.js";
import { planAceOpt0007VaeK1Conv1d } from
  "../benchmark/opt-0007-vae-k1-conv1d.js";

describe("OPT-0007 pointwise VAE Conv1D browser contract", () => {
  it("pins independent correctness and endpoint modes", () => {
    for (const mode of [
      "correctness",
      "screen-c1024",
      "screen-c128",
      "sequence-c128",
    ] as const) {
      expect(parseOpt0007RunMode(new URLSearchParams({ runMode: mode }))).toBe(mode);
    }
    expect(() => parseOpt0007RunMode(new URLSearchParams())).toThrow(/explicit runMode/);
  });

  it("pins the bounded C128 batch-eight sequence topology", () => {
    const plan = planAceOpt0007VaeK1Conv1d(OPT_0007_SCREEN_C128_SHAPE);
    const indices = sequenceC128RangeIndices(plan.outputRangeCount);
    const ranges = indices.map((index) => plan.outputRanges[index]!);
    expect(plan.outputRangeCount).toBe(30);
    expect(OPT_0007_SEQUENCE_RANGE_COUNT).toBe(16);
    expect(OPT_0007_SEQUENCE_RANGES_PER_COMMAND_BUFFER).toBe(8);
    expect(indices).toEqual(Array.from({ length: 16 }, (_, index) => index + 7));
    expect(ranges.reduce((sum, range) => sum + range.outputCount, 0))
      .toBe(16_777_216);
    expect(ranges.reduce((sum, range) => sum + range.multiplyAdds, 0))
      .toBe(2_147_483_648);
    expect(ranges.every((range) => range.outputRowCount === 8_192)).toBe(true);
    expect(ranges.every((range) => range.outputCount === 1_048_576)).toBe(true);
    expect(expectedOpt0007SequenceCounts()).toEqual({
      rangeCount: 16,
      passCount: 16,
      dispatchCount: 16,
      progressEventCount: 16,
      commandBufferCount: 2,
      queueDrains: 2,
      explicitIdleCount: 2,
      rangesPerCommandBuffer: 8,
    });
  });

  it("pins the two authenticated production endpoint shapes and middle ranges", () => {
    expect(OPT_0007_SCREEN_C1024_SHAPE).toEqual({
      batch: 1, inputFrames: 2_560, inputChannels: 1_024, outputChannels: 1_024,
      kernelSize: 1, stride: 1, dilation: 1, padding: 0,
    });
    expect(OPT_0007_SCREEN_C128_SHAPE).toEqual({
      batch: 1, inputFrames: 245_760, inputChannels: 128, outputChannels: 128,
      kernelSize: 1, stride: 1, dilation: 1, padding: 0,
    });
    const high = planAceOpt0007VaeK1Conv1d(OPT_0007_SCREEN_C1024_SHAPE);
    const low = planAceOpt0007VaeK1Conv1d(OPT_0007_SCREEN_C128_SHAPE);
    expect(high.outputRanges[middleRangeIndex(high.outputRangeCount)]).toMatchObject({
      outputRowCount: 224, outputCount: 229_376, multiplyAdds: 234_881_024,
    });
    expect(low.outputRanges[middleRangeIndex(low.outputRangeCount)]).toMatchObject({
      outputRowCount: 8_192, outputCount: 1_048_576, multiplyAdds: 134_217_728,
    });
  });

  it("covers bias, batches, chunk boundary/tails and arithmetic order", () => {
    expect(OPT_0007_CORRECTNESS_CASES.map(({ id }) => id)).toEqual([
      "bias-batch-c65-tails", "no-bias-batch-c64", "bias-c63-range-tail",
      "arithmetic-discriminants",
    ]);
    expect(OPT_0007_CORRECTNESS_CASES.map(({ shape }) => shape.inputChannels))
      .toEqual([65, 64, 63, 65]);
    expect(OPT_0007_CORRECTNESS_CASES.map(({ shape }) => shape.batch))
      .toEqual([2, 3, 2, 1]);
    expect(OPT_0007_CORRECTNESS_CASES.map(({ hasBias }) => hasBias))
      .toEqual([true, false, true, false]);
    expect(OPT_0007_ARITHMETIC_SENTINELS).toEqual({
      contractedOutputIndex: 45,
      cancellationOutputIndex: 91,
    });
    expect(float32Bits(opt0007ExpectedOutputValue(
      "arithmetic-discriminants",
      OPT_0007_ARITHMETIC_SENTINELS.contractedOutputIndex,
    ))).toBe(0xbe7d_3830);
    expect(opt0007ExpectedOutputValue(
      "arithmetic-discriminants",
      OPT_0007_ARITHMETIC_SENTINELS.cancellationOutputIndex,
    )).toBe(0);
  });

  it("uses symmetric warmup, balanced pairs, independent outputs and full U32 checks", () => {
    expect(OPT_0007_PAIRED_ORDERS).toEqual([
      ["scalar", "candidate"], ["candidate", "scalar"],
      ["candidate", "scalar"], ["scalar", "candidate"],
    ]);
    const harness = readFileSync(new URL(
      "./browser/opt-0007-vae-k1-conv1d-ab.ts", import.meta.url), "utf8");
    expect(harness).toContain("independentScalarAndCandidateOutputs: true");
    expect(harness).toContain("warmupExecutionsPerKernel: 1");
    expect(harness).toContain("fullRangeBitIdentical: true");
    expect(harness).toContain("OUTPUT_SENTINEL_BITS = 0x7fc0_0000");
    expect(harness).toContain("compileAllocationUploadExcludedFromTiming: true");
    expect(harness).toContain("noPerformanceAcceptanceThreshold: true");
    expect(harness).toContain(
      "cancellation proof requires at least two real ranges",
    );
    expect(harness).toContain("laterRangeEncodingPrevented: true");
    expect(harness).toContain("postTimingGuards");
    expect(harness).toContain("runAceOpt0006QuantumBatches");
    expect(harness).toContain("maximumQuantaPerCommandBuffer:");
    expect(harness).toContain("finalCommandBufferRemains: true");
    expect(harness).toContain("symmetricFullSequenceWarmups: true");
    expect(harness).toContain("idleAfterFinalPhysicalBatch: true");
    expect(harness).toContain("real idle collapsed");
    expect(harness).toContain("laterBatchEncodingPrevented: true");
    expect(harness).toContain("completedAt: new Date().toISOString()");
  });

  it("requires continuous external thermal evidence only for timing screens", () => {
    expect(parseOpt0007ThermalGateMetadata(
      new URLSearchParams({ runMode: "correctness" }),
    )).toBeUndefined();
    const valid = new URLSearchParams({
      runMode: "screen-c1024",
      thermalSource: "notifyutil-com.apple.system.thermalpressurelevel",
      thermalDurationSeconds: "30.1",
      thermalObservations: "31",
      thermalPollMilliseconds: "1000",
      thermalMaximumPollGapMilliseconds: "1100",
      thermalNonNominalObservations: "0",
    });
    expect(parseOpt0007ThermalGateMetadata(valid)).toMatchObject({
      observationCount: 31, nonNominalObservationCount: 0,
    });
    valid.set("thermalNonNominalObservations", "1");
    expect(() => parseOpt0007ThermalGateMetadata(valid)).toThrow(/non-nominal/);
    valid.set("thermalNonNominalObservations", "0");
    valid.set("runMode", "sequence-c128");
    expect(parseOpt0007ThermalGateMetadata(valid)).toMatchObject({
      durationSeconds: 30.1,
      observationCount: 31,
    });
  });

  it("retains raw samples and reports a true median", () => {
    expect(summarizeOpt0007Samples([9, 1, 7, 3])).toEqual({
      count: 4, samples: [9, 1, 7, 3], minimum: 1, median: 5, maximum: 9, range: 8,
    });
  });
});

function float32Bits(value: number): number {
  return new Uint32Array(new Float32Array([value]).buffer)[0]!;
}
