import { describe, expect, it } from "vitest";

import {
  ACE_OPT_0004_VAE_CONV1D_INPUT_TILE_STRIDE,
  aceOpt0004VaeConv1dWgsl,
  planAceOpt0004VaeConv1d,
} from "../benchmark/opt-0004-vae-conv1d.js";
import {
  OPT_0004_ARITHMETIC_SENTINELS,
  OPT_0004_CORRECTNESS_CASES,
  OPT_0004_PAIRED_ORDERS,
  OPT_0004_PRODUCTION_SHAPE,
  opt0004ExpectedOutputValue,
  opt0004InputValue,
  opt0004OutputSentinelIndices,
  opt0004ValidMultiplyAdds,
  opt0004WeightValue,
  parseOpt0004ProductionMode,
  parseOpt0004ThermalGateMetadata,
  summarizeOpt0004Samples,
} from "./browser/opt-0004-vae-conv1d-ab.js";

describe("OPT-0004 VAE Conv1D paired A/B browser contract", () => {
  it("pins the complete first production operation and its bounded ranges", () => {
    expect(OPT_0004_PRODUCTION_SHAPE).toEqual({
      batch: 1,
      inputFrames: 491_520,
      inputChannels: 128,
      outputChannels: 128,
      kernelSize: 7,
      stride: 1,
      dilation: 1,
      padding: 3,
    });
    const plan = planAceOpt0004VaeConv1d(OPT_0004_PRODUCTION_SHAPE);
    expect(plan.outputFrames).toBe(491_520);
    expect(plan.outputElements).toBe(62_914_560);
    expect(plan.outputRangeCount).toBe(240);
    expect(new Set(plan.outputRanges.map((range) => range.outputRowCount)))
      .toEqual(new Set([2_048]));
    expect(plan.outputRanges[0]).toMatchObject({
      firstOutput: 0,
      outputCount: 262_144,
      workgroupsX: 128,
      workgroupsY: 16,
    });
    expect(plan.outputRanges.at(-1)).toMatchObject({
      firstOutput: 62_652_416,
      outputCount: 262_144,
    });
    expect(opt0004ValidMultiplyAdds(OPT_0004_PRODUCTION_SHAPE))
      .toBe(56_371_249_152);
  });

  it("reports the padded shared-input and shared-weight memory geometry", () => {
    const plan = planAceOpt0004VaeConv1d(OPT_0004_PRODUCTION_SHAPE);
    expect(plan.inputTileStride).toBe(23);
    expect(plan.weightTileStride).toBe(129);
    expect(plan.inputTileBytes).toBe(11_776);
    expect(plan.weightTileBytes).toBe(4_128);
    expect(plan.workgroupStorageBytes).toBe(15_904);
    expect(ACE_OPT_0004_VAE_CONV1D_INPUT_TILE_STRIDE).toBe(23);
    const source = aceOpt0004VaeConv1dWgsl(
      OPT_0004_PRODUCTION_SHAPE,
      true,
    );
    expect(source).toContain("@workgroup_size(\n  16,\n  8,\n  1,");
    expect(source).toContain("var<workgroup> input_tile");
    expect(source).toContain("var<workgroup> weight_tile");
    expect(source).toContain("input_channel * 23u + tile_time");
    expect(source).toContain("local.y * 129u");
    expect(source).not.toContain("fma(");
    expect(source).not.toContain("enable f16");
  });

  it("uses four balanced scalar/tiled AB/BA samples", () => {
    expect(OPT_0004_PAIRED_ORDERS).toEqual([
      ["scalar", "tiled"],
      ["tiled", "scalar"],
      ["tiled", "scalar"],
      ["scalar", "tiled"],
    ]);
    expect(OPT_0004_PAIRED_ORDERS.flat().filter((id) => id === "scalar"))
      .toHaveLength(4);
    expect(OPT_0004_PAIRED_ORDERS.flat().filter((id) => id === "tiled"))
      .toHaveLength(4);
  });

  it("pins complete multi-range tail, padding, and bias fixtures", () => {
    expect(OPT_0004_CORRECTNESS_CASES.map(({ id }) => id)).toEqual([
      "tail-padding-bias",
      "wide-padding-no-bias",
      "arithmetic-discriminants",
    ]);
    expect(OPT_0004_CORRECTNESS_CASES.map(({ hasBias }) => hasBias))
      .toEqual([true, false, false]);
    for (const fixture of OPT_0004_CORRECTNESS_CASES) {
      const plan = planAceOpt0004VaeConv1d(fixture.shape);
      expect(plan.outputRangeCount).toBe(fixture.shape.batch);
      expect(plan.outputFrames % 16).not.toBe(0);
      expect(plan.outputChannels % 8).not.toBe(0);
      expect(plan.outputRanges.map(({ batch }) => batch)).toEqual(
        Array.from({ length: fixture.shape.batch }, (_, index) => index),
      );
      const sentinels = opt0004OutputSentinelIndices(fixture.shape);
      expect(sentinels).toContain(0);
      expect(sentinels).toContain(plan.outputElements - 1);
      for (const range of plan.outputRanges) {
        expect(sentinels).toContain(range.firstOutput);
        expect(sentinels).toContain(
          range.firstOutput + range.outputCount - 1,
        );
      }
    }
    expect(Object.is(opt0004InputValue("tail-padding-bias", 0), -0)).toBe(true);
    expect(Object.is(opt0004WeightValue("tail-padding-bias", 0), -0)).toBe(true);
  });

  it("pins contraction and source-order cancellation CPU discriminants", () => {
    expect(OPT_0004_ARITHMETIC_SENTINELS).toEqual({
      contractedOutputIndex: 72,
      cancellationOutputIndex: 118,
    });
    expect(float32Bits(opt0004ExpectedOutputValue(
      "arithmetic-discriminants",
      OPT_0004_ARITHMETIC_SENTINELS.contractedOutputIndex,
    ))).toBe(0xbe7d_3830);
    expect(opt0004ExpectedOutputValue(
      "arithmetic-discriminants",
      OPT_0004_ARITHMETIC_SENTINELS.cancellationOutputIndex,
    )).toBe(0.5);
    const reassociated = Math.fround(
      Math.fround(16_777_216 + -16_777_216) + Math.fround(1 + 0.5),
    );
    expect(reassociated).toBe(1.5);
  });

  it("fails closed on thermal provenance and requires explicit run scale", () => {
    const valid = new URLSearchParams({
      thermalSource: "notifyutil-com.apple.system.thermalpressurelevel",
      thermalDurationSeconds: "30.01",
      thermalObservations: "31",
      thermalPollMilliseconds: "1000",
      thermalMaximumPollGapMilliseconds: "1020",
      thermalNonNominalObservations: "0",
      fullProduction: "1",
    });
    expect(parseOpt0004ThermalGateMetadata(valid)).toMatchObject({
      durationSeconds: 30.01,
      observationCount: 31,
      nonNominalObservationCount: 0,
    });
    expect(parseOpt0004ProductionMode(valid)).toBe("full");
    valid.set("fullProduction", "0");
    expect(parseOpt0004ProductionMode(valid)).toBe("representative");
    valid.delete("fullProduction");
    expect(() => parseOpt0004ProductionMode(valid)).toThrow(/explicit/);
    valid.set("thermalNonNominalObservations", "1");
    expect(() => parseOpt0004ThermalGateMetadata(valid)).toThrow(/non-nominal/);
  });

  it("retains raw samples with median and range", () => {
    expect(summarizeOpt0004Samples([9, 1, 5, 3])).toEqual({
      count: 4,
      samples: [9, 1, 5, 3],
      minimum: 1,
      median: 4,
      maximum: 9,
      range: 8,
    });
    expect(() => summarizeOpt0004Samples([])).toThrow(/must not be empty/);
    expect(() => summarizeOpt0004Samples([1, Number.NaN])).toThrow(
      /finite non-negative/,
    );
  });
});

function float32Bits(value: number): number {
  const scratch = new Float32Array([value]);
  return new Uint32Array(scratch.buffer)[0]!;
}
