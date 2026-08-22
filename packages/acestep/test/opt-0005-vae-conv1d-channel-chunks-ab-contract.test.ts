import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";

import {
  ACE_OPT_0005_VAE_CONV1D_INPUT_CHANNEL_CHUNK,
  ACE_OPT_0005_VAE_CONV1D_INPUT_TILE_STRIDE,
  ACE_OPT_0005_VAE_CONV1D_WEIGHT_TILE_STRIDE,
  aceOpt0005VaeConv1dChannelChunksWgsl,
  planAceOpt0005VaeConv1dChannelChunks,
} from "../benchmark/opt-0005-vae-conv1d.js";
import {
  OPT_0005_ARITHMETIC_SENTINELS,
  OPT_0005_BLOCK0_D1_SHAPE,
  OPT_0005_CORRECTNESS_CASES,
  OPT_0005_OVERLAP_D1_SHAPE,
  OPT_0005_OVERLAP_ORDERS,
  OPT_0005_PAIRED_ORDERS,
  OPT_0005_SCREEN_D1_SHAPE,
  OPT_0005_SCREEN_D9_SHAPE,
  effectiveOpt0005RangeCount,
  opt0005ExpectedOutputValue,
  opt0005InputValue,
  opt0005OutputSentinelIndices,
  opt0005RangeValidMultiplyAdds,
  opt0005ValidMultiplyAdds,
  opt0005WeightValue,
  parseOpt0005ProductionMode,
  parseOpt0005ThermalGateMetadata,
  screenRangeIndex,
  summarizeOpt0005Samples,
} from "./browser/opt-0005-vae-conv1d-channel-chunks-ab.js";

describe("OPT-0005 channel-chunked VAE Conv1D browser contract", () => {
  it("requests the 256-lane scalar authority while measuring WG(16,8)", () => {
    const harness = readFileSync(
      new URL(
        "./browser/opt-0005-vae-conv1d-channel-chunks-ab.ts",
        import.meta.url,
      ),
      "utf8",
    );
    expect(harness).toContain("maxComputeInvocationsPerWorkgroup: 256");
    expect(harness).toContain("maxComputeWorkgroupSizeX: 256");
    expect(harness).toContain(
      "adapter.limits.maxComputeInvocationsPerWorkgroup < 256",
    );
    expect(harness).toContain("adapter.limits.maxComputeWorkgroupSizeX < 256");
  });

  it("pins sparse production-relevant run modes without an automatic sweep", () => {
    expect(OPT_0005_BLOCK0_D1_SHAPE).toEqual({
      batch: 1,
      inputFrames: 2_560,
      inputChannels: 1_024,
      outputChannels: 1_024,
      kernelSize: 7,
      stride: 1,
      dilation: 1,
      padding: 3,
    });
    const block = planAceOpt0005VaeConv1dChannelChunks(
      OPT_0005_BLOCK0_D1_SHAPE,
    );
    expect(block.outputRangeCount).toBe(80);
    expect(new Set(block.outputRanges.map(({ outputRowCount }) => outputRowCount)))
      .toEqual(new Set([32]));
    expect(block.outputElements).toBe(2_621_440);
    expect(opt0005ValidMultiplyAdds(OPT_0005_BLOCK0_D1_SHAPE))
      .toBe(18_777_899_008);

    for (const shape of [OPT_0005_SCREEN_D1_SHAPE, OPT_0005_SCREEN_D9_SHAPE]) {
      const plan = planAceOpt0005VaeConv1dChannelChunks(shape);
      expect(plan.outputRangeCount).toBe(80);
      expect(plan.inputChannels).toBe(1_024);
      const selected = screenRangeIndex(
        shape.dilation === 1 ? "screen-d1" : "screen-d9",
        plan.outputRangeCount,
      );
      expect(selected).toBe(40);
      expect(effectiveOpt0005RangeCount(
        shape.dilation === 1 ? "screen-d1" : "screen-d9",
        plan.outputRangeCount,
      )).toBe(1);
      expect(plan.outputRanges[selected!]).toMatchObject({
        firstOutputTime: 1_280,
        outputRowCount: 32,
      });
      expect(opt0005RangeValidMultiplyAdds(shape, plan.outputRanges[selected!]!))
        .toBe(234_881_024);
    }
    expect(OPT_0005_SCREEN_D9_SHAPE).toMatchObject({ dilation: 9, padding: 27 });
    expect(OPT_0005_OVERLAP_D1_SHAPE).toMatchObject({
      inputFrames: 2_048,
      inputChannels: 128,
      outputChannels: 128,
      dilation: 1,
    });
    expect(screenRangeIndex("block0-d1", block.outputRangeCount)).toBeUndefined();
    expect(screenRangeIndex("overlap-d1", 1)).toBeUndefined();
  });

  it("pins the fixed chunk-64 FP32 shared-memory geometry", () => {
    const plan = planAceOpt0005VaeConv1dChannelChunks(
      OPT_0005_BLOCK0_D1_SHAPE,
    );
    expect(ACE_OPT_0005_VAE_CONV1D_INPUT_CHANNEL_CHUNK).toBe(64);
    expect(ACE_OPT_0005_VAE_CONV1D_INPUT_TILE_STRIDE).toBe(17);
    expect(ACE_OPT_0005_VAE_CONV1D_WEIGHT_TILE_STRIDE).toBe(65);
    expect(plan.inputChannelChunkCount).toBe(16);
    expect(plan.inputTileBytes).toBe(4_352);
    expect(plan.weightTileBytes).toBe(2_080);
    expect(plan.workgroupStorageBytes).toBe(6_432);
    const source = aceOpt0005VaeConv1dChannelChunksWgsl(
      OPT_0005_BLOCK0_D1_SHAPE,
      true,
    );
    expect(source).toContain("var input_channel_chunk = 0u");
    expect(source).toContain("input_channel_chunk * 64u");
    expect(source).toContain("chunk_channel * 17u");
    expect(source).toContain("tile_output_channel * 65u");
    expect(source).toContain("let chunk_channel_count = min(");
    expect(source.match(/workgroupBarrier\(\)/g)).toHaveLength(2);
    expect(source).not.toContain("fma(");
    expect(source).not.toContain("enable f16");
  });

  it("covers d1/d3/d9, both sides of chunk 64, tails, batches, and bias", () => {
    expect(OPT_0005_CORRECTNESS_CASES.map(({ id }) => id)).toEqual([
      "dilation1-channel-tail-bias",
      "dilation3-batch-padding-no-bias",
      "dilation9-range-tail-bias",
      "arithmetic-discriminants",
    ]);
    expect(OPT_0005_CORRECTNESS_CASES.map(({ shape }) => shape.dilation))
      .toEqual([1, 3, 9, 1]);
    expect(OPT_0005_CORRECTNESS_CASES.map(({ shape }) => shape.inputChannels))
      .toEqual([65, 64, 63, 65]);
    expect(OPT_0005_CORRECTNESS_CASES.map(({ hasBias }) => hasBias))
      .toEqual([true, false, true, false]);
    for (const fixture of OPT_0005_CORRECTNESS_CASES) {
      const plan = planAceOpt0005VaeConv1dChannelChunks(fixture.shape);
      expect(plan.outputRangeCount).toBe(fixture.shape.batch);
      expect(plan.outputFrames % 16).not.toBe(0);
      expect(plan.outputChannels % 8).not.toBe(0);
      expect(plan.outputRanges.map(({ batch }) => batch)).toEqual(
        Array.from({ length: fixture.shape.batch }, (_, index) => index),
      );
      const sentinels = opt0005OutputSentinelIndices(fixture.shape);
      expect(sentinels).toContain(0);
      expect(sentinels).toContain(plan.outputElements - 1);
      for (const range of plan.outputRanges) {
        expect(sentinels).toContain(range.firstOutput);
        expect(sentinels).toContain(range.firstOutput + range.outputCount - 1);
      }
    }
    expect(Object.is(
      opt0005InputValue("dilation1-channel-tail-bias", 0),
      -0,
    )).toBe(true);
    expect(Object.is(
      opt0005WeightValue("dilation1-channel-tail-bias", 0),
      -0,
    )).toBe(true);
  });

  it("pins contraction and cross-chunk source-order discriminants", () => {
    expect(OPT_0005_ARITHMETIC_SENTINELS).toEqual({
      contractedOutputIndex: 72,
      cancellationOutputIndex: 118,
    });
    expect(float32Bits(opt0005ExpectedOutputValue(
      "arithmetic-discriminants",
      OPT_0005_ARITHMETIC_SENTINELS.contractedOutputIndex,
    ))).toBe(0xbe7d_3830);
    expect(opt0005ExpectedOutputValue(
      "arithmetic-discriminants",
      OPT_0005_ARITHMETIC_SENTINELS.cancellationOutputIndex,
    )).toBe(0.5);
    const reassociated = Math.fround(
      Math.fround(16_777_216 + -16_777_216) + Math.fround(1 + 0.5),
    );
    expect(reassociated).toBe(1.5);
  });

  it("uses balanced paired orders and a sparse explicit overlap comparison", () => {
    expect(OPT_0005_PAIRED_ORDERS).toEqual([
      ["scalar", "chunked"],
      ["chunked", "scalar"],
      ["chunked", "scalar"],
      ["scalar", "chunked"],
    ]);
    expect(OPT_0005_OVERLAP_ORDERS).toEqual([
      ["scalar", "opt0004", "chunked"],
      ["chunked", "opt0004", "scalar"],
      ["chunked", "opt0004", "scalar"],
      ["scalar", "opt0004", "chunked"],
    ]);
  });

  it("fails closed on thermal provenance and requires an explicit run mode", () => {
    const valid = new URLSearchParams({
      thermalSource: "notifyutil-com.apple.system.thermalpressurelevel",
      thermalDurationSeconds: "30.01",
      thermalObservations: "31",
      thermalPollMilliseconds: "1000",
      thermalMaximumPollGapMilliseconds: "1020",
      thermalNonNominalObservations: "0",
      runMode: "screen-d1",
    });
    expect(parseOpt0005ThermalGateMetadata(valid)).toMatchObject({
      durationSeconds: 30.01,
      observationCount: 31,
      nonNominalObservationCount: 0,
    });
    for (const mode of [
      "screen-d1",
      "block0-d1",
      "screen-d9",
      "overlap-d1",
    ] as const) {
      valid.set("runMode", mode);
      expect(parseOpt0005ProductionMode(valid)).toBe(mode);
    }
    valid.delete("runMode");
    expect(() => parseOpt0005ProductionMode(valid)).toThrow(/explicit/);
    valid.set("thermalNonNominalObservations", "1");
    expect(() => parseOpt0005ThermalGateMetadata(valid)).toThrow(/non-nominal/);
  });

  it("retains every raw timing sample with median and range", () => {
    expect(summarizeOpt0005Samples([9, 1, 5, 3])).toEqual({
      count: 4,
      samples: [9, 1, 5, 3],
      minimum: 1,
      median: 4,
      maximum: 9,
      range: 8,
    });
    expect(() => summarizeOpt0005Samples([])).toThrow(/must not be empty/);
    expect(() => summarizeOpt0005Samples([1, Number.NaN])).toThrow(
      /finite non-negative/,
    );
  });
});

function float32Bits(value: number): number {
  const scratch = new Float32Array([value]);
  return new Uint32Array(scratch.buffer)[0]!;
}
