import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

import { ACE_OPT_0006_QUANTA_PER_COMMAND_BUFFER_CANDIDATES } from
  "../benchmark/opt-0006-vae-command-buffer-coalescing.js";
import { planAceOpt0005VaeConv1dChannelChunks } from
  "../benchmark/opt-0005-vae-conv1d.js";
import {
  OPT_0006_BALANCED_BATCH_ORDERS,
  OPT_0006_PRODUCTION_RANGE_INDICES,
  OPT_0006_PRODUCTION_RANGE_SHAPE,
  expectedOpt0006SchedulingCounts,
  parseOpt0006ThermalGateMetadata,
  summarizeOpt0006Samples,
} from "./browser/opt-0006-vae-command-buffer-coalescing-production-ab.js";

describe("OPT-0006 production-range browser A/B contract", () => {
  it("pins 16 distinct consecutive full-size production ranges", () => {
    expect(OPT_0006_PRODUCTION_RANGE_SHAPE).toEqual({
      batch: 1,
      inputFrames: 2_560,
      inputChannels: 1_024,
      outputChannels: 1_024,
      kernelSize: 7,
      stride: 1,
      dilation: 1,
      padding: 3,
    });
    expect(OPT_0006_PRODUCTION_RANGE_INDICES).toEqual(
      Array.from({ length: 16 }, (_, index) => index + 32),
    );
    const plan = planAceOpt0005VaeConv1dChannelChunks(
      OPT_0006_PRODUCTION_RANGE_SHAPE,
    );
    expect(plan.outputRangeCount).toBe(80);
    const ranges = OPT_0006_PRODUCTION_RANGE_INDICES.map(
      (index) => plan.outputRanges[index]!,
    );
    expect(ranges.map(({ firstOutputTime }) => firstOutputTime)).toEqual(
      Array.from({ length: 16 }, (_, index) => (index + 32) * 32),
    );
    expect(new Set(ranges.map(({ outputRowCount }) => outputRowCount)))
      .toEqual(new Set([32]));
    expect(new Set(ranges.map(({ outputCount }) => outputCount)))
      .toEqual(new Set([32_768]));
    expect(new Set(ranges.map(({ multiplyAdds }) => multiplyAdds)))
      .toEqual(new Set([234_881_024]));
    expect(ranges.at(-1)!.firstOutput + ranges.at(-1)!.outputCount -
      ranges[0]!.firstOutput).toBe(524_288);
  });

  it("pins F/R/R/F balanced rounds and four samples per candidate", () => {
    expect(ACE_OPT_0006_QUANTA_PER_COMMAND_BUFFER_CANDIDATES)
      .toEqual([1, 2, 4, 8, 16]);
    expect(OPT_0006_BALANCED_BATCH_ORDERS).toEqual([
      [1, 2, 4, 8, 16],
      [16, 8, 4, 2, 1],
      [16, 8, 4, 2, 1],
      [1, 2, 4, 8, 16],
    ]);
    for (const candidate of ACE_OPT_0006_QUANTA_PER_COMMAND_BUFFER_CANDIDATES) {
      expect(OPT_0006_BALANCED_BATCH_ORDERS.flat().filter(
        (value) => value === candidate,
      )).toHaveLength(4);
    }
  });

  it("accounts every pass, dispatch, quantum progress event, drain, and idle", () => {
    expect(ACE_OPT_0006_QUANTA_PER_COMMAND_BUFFER_CANDIDATES.map(
      (batchSize) => expectedOpt0006SchedulingCounts(batchSize),
    )).toEqual([
      schedulingCounts(16),
      schedulingCounts(8),
      schedulingCounts(4),
      schedulingCounts(2),
      schedulingCounts(1),
    ]);
  });

  it("uses the shipped OPT-0005 kernel and keeps all setup outside timing", () => {
    const source = readFileSync(new URL(
      "./browser/opt-0006-vae-command-buffer-coalescing-production-ab.ts",
      import.meta.url,
    ), "utf8");
    expect(source).toContain("AceChannelChunkedVaeConv1dKernel");
    expect(source).not.toContain("AceOpt0005VaeConv1dChannelChunksKernel");
    expect(source).toContain("runAceOpt0006QuantumBatches");
    expect(source).toContain("finalCommandBufferRemains: true");
    expect(source).toContain("mappedSentinelOutput");
    expect(source).toContain("OUTPUT_SENTINEL_BITS = 0x7fc0_0000");
    expect(source).toContain("validateFullUnionIdentity(outputsByBatch)");
    expect(source).toContain("compilationAllocationUploadExcludedFromTiming: true");
    expect(source).toContain("maximumOutstanding !== 1");
    expect(source).toContain("heartbeat.stop()");
  });

  it("directly cancels from post-drain progress and still completes real idle", () => {
    const source = readFileSync(new URL(
      "./browser/opt-0006-vae-command-buffer-coalescing-production-ab.ts",
      import.meta.url,
    ), "utf8");
    const cancellation = source.slice(
      source.indexOf("async function runCancellationProof"),
      source.indexOf("function selectedQuanta"),
    );
    expect(cancellation).toContain("onProgress: (event)");
    expect(cancellation).toContain("event.completedQuanta === 4");
    expect(cancellation).toContain("controller.abort(");
    expect(cancellation).toContain("yieldQueueIdle: async ()");
    expect(cancellation).toContain("await realQueueEmptyIdle()");
    expect(cancellation).toContain("progressEvents !== 4");
    expect(cancellation).toContain("idleCount !== 1");
    expect(cancellation).toContain("idleMilliseconds < 1");
    const drainBody = cancellation.slice(
      cancellation.indexOf("async onSubmittedWorkDone"),
      cancellation.indexOf("let rejection"),
    );
    expect(drainBody).not.toContain("controller.abort");
  });

  it("fails closed on thermal provenance and explicit run mode", () => {
    const valid = new URLSearchParams({
      runMode: "production-ranges",
      thermalSource: "notifyutil-com.apple.system.thermalpressurelevel",
      thermalDurationSeconds: "30.01",
      thermalObservations: "31",
      thermalPollMilliseconds: "1000",
      thermalMaximumPollGapMilliseconds: "1010",
      thermalNonNominalObservations: "0",
    });
    expect(parseOpt0006ThermalGateMetadata(valid)).toMatchObject({
      durationSeconds: 30.01,
      observationCount: 31,
      nonNominalObservationCount: 0,
    });
    valid.set("runMode", "other");
    expect(() => parseOpt0006ThermalGateMetadata(valid)).toThrow(/explicit/);
    valid.set("runMode", "production-ranges");
    valid.set("thermalNonNominalObservations", "1");
    expect(() => parseOpt0006ThermalGateMetadata(valid)).toThrow(/non-nominal/);
  });

  it("retains raw samples and reports a median", () => {
    expect(summarizeOpt0006Samples([9, 1, 5, 3])).toEqual({
      count: 4,
      samples: [9, 1, 5, 3],
      minimum: 1,
      median: 4,
      maximum: 9,
    });
    expect(() => summarizeOpt0006Samples([])).toThrow(/finite and non-negative/);
    expect(() => summarizeOpt0006Samples([1, Number.NaN])).toThrow(
      /finite and non-negative/,
    );
  });
});

function schedulingCounts(commandBufferCount: number) {
  return {
    quantumCount: 16,
    passCount: 16,
    dispatchCount: 16,
    commandBufferCount,
    queueDrains: commandBufferCount,
    progressEventCount: 16,
    explicitIdleCount: commandBufferCount,
  };
}
