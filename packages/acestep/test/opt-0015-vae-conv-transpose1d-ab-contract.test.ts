import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import {
  OPT_0015_C300_EXPECTED_TOPOLOGY,
  buildOpt0015C300Topology,
  parseOpt0015ThermalGate,
  summarizeOpt0015WeightedTiming,
} from "./browser/opt-0015-vae-conv-transpose1d-ab.js";

const HARNESS_PATH = fileURLToPath(new URL(
  "./browser/opt-0015-vae-conv-transpose1d-ab.ts",
  import.meta.url,
));
const HTML_PATH = fileURLToPath(new URL(
  "./browser/opt-0015-vae-conv-transpose1d-ab.html",
  import.meta.url,
));
const HARNESS_SOURCE = readFileSync(HARNESS_PATH, "utf8");
const HTML_SOURCE = readFileSync(HTML_PATH, "utf8");

describe("OPT-0015 target-browser ConvTranspose1D gate", () => {
  it("freezes all five exact C300 operations and 378 graph ranges", () => {
    const topology = buildOpt0015C300Topology();
    expect(topology).toHaveLength(5);
    expect(topology.reduce(
      (sum, operation) => sum + operation.ranges.length,
      0,
    )).toBe(378);
    expect(topology.map((operation) => ({
      label: operation.label,
      shape: operation.shape,
      outputFrames: operation.outputFrames,
      rangeCount: operation.ranges.length,
      fullRangeRows: operation.ranges[0]!.outputRowCount,
      tailRangeRows: operation.ranges.at(-1)!.outputRowCount,
    }))).toEqual(OPT_0015_C300_EXPECTED_TOPOLOGY);
    expect(topology.map((operation) =>
      operation.selectedRanges.map((range) => ({
        stratum: range.stratum,
        rangeIndex: range.rangeIndex,
        base: range.base,
        count: range.count,
        weight: range.weight,
      }))
    )).toEqual([
      [
        { stratum: "first", rangeIndex: 0, base: 0, count: 57_344, weight: 1 },
        { stratum: "interior", rangeIndex: 27, base: 1_548_288, count: 57_344, weight: 52 },
        { stratum: "tail", rangeIndex: 53, base: 3_039_232, count: 32_768, weight: 1 },
      ],
      [
        { stratum: "first", rangeIndex: 0, base: 0, count: 114_688, weight: 1 },
        { stratum: "interior", rangeIndex: 40, base: 4_587_520, count: 114_688, weight: 79 },
        { stratum: "tail", rangeIndex: 80, base: 9_175_040, count: 40_960, weight: 1 },
      ],
      [
        { stratum: "first", rangeIndex: 0, base: 0, count: 229_376, weight: 1 },
        { stratum: "interior", rangeIndex: 40, base: 9_175_040, count: 229_376, weight: 79 },
        { stratum: "tail", rangeIndex: 80, base: 18_350_080, count: 81_920, weight: 1 },
      ],
      [
        { stratum: "first", rangeIndex: 0, base: 0, count: 458_752, weight: 1 },
        { stratum: "interior", rangeIndex: 40, base: 18_350_080, count: 458_752, weight: 79 },
        { stratum: "tail", rangeIndex: 80, base: 36_700_160, count: 163_840, weight: 1 },
      ],
      [
        { stratum: "first", rangeIndex: 0, base: 0, count: 917_504, weight: 1 },
        { stratum: "interior", rangeIndex: 40, base: 36_700_160, count: 917_504, weight: 79 },
        { stratum: "tail", rangeIndex: 80, base: 73_400_320, count: 327_680, weight: 1 },
      ],
    ]);
  });

  it("accepts exactly one fresh 30-second nominal pre-gate", () => {
    const parameters = new URLSearchParams({
      thermalSource: "notifyutil-com.apple.system.thermalpressurelevel",
      thermalStartedAtEpochMilliseconds: "2000",
      thermalCompletedAtEpochMilliseconds: "32010",
      thermalObservations: "31",
      thermalPollMilliseconds: "1000",
      thermalMaximumPollGapMilliseconds: "1010",
      thermalNonNominalObservations: "0",
    });
    expect(parseOpt0015ThermalGate(parameters, 1_999, 32_020)).toMatchObject({
      durationMilliseconds: 30_010,
      observationCount: 31,
      launchDelayMilliseconds: 10,
      nonNominalObservationCount: 0,
    });
    parameters.set("thermalNonNominalObservations", "1");
    expect(() => parseOpt0015ThermalGate(parameters, 1_999, 32_020))
      .toThrow(/incomplete or non-nominal/);
  });

  it("weights stratum medians by exact range multiplicity", () => {
    const result = summarizeOpt0015WeightedTiming([
      {
        operationLabel: "block-0-conv-t1",
        stratum: "first",
        weight: 1,
        portableSamples: [10, 12],
        congruentSamples: [2, 4],
      },
      {
        operationLabel: "block-0-conv-t1",
        stratum: "interior",
        weight: 52,
        portableSamples: [20, 22],
        congruentSamples: [4, 6],
      },
      {
        operationLabel: "block-0-conv-t1",
        stratum: "tail",
        weight: 1,
        portableSamples: [6, 8],
        congruentSamples: [1, 3],
      },
    ]);
    expect(result).toMatchObject({
      exactRangeCount: 54,
      portableProjectedMilliseconds: 1_110,
      congruentProjectedMilliseconds: 265,
    });
  });

  it("pins source identity, bounded correctness, balanced timing, and no retry", () => {
    expect(HARNESS_SOURCE).toContain(
      "075ecc0b34b7541cffc0a83412c17ee31bbadab6",
    );
    expect(HARNESS_SOURCE).toContain(
      "cbcb9bcd5f856ce1c9e10aabca0ec0f95651c03d2c45b8076de3ba5022c6c3e2",
    );
    expect(HARNESS_SOURCE).toContain("createCongruent(device)");
    expect(HARNESS_SOURCE).toContain("completeSelectedRangeRawU16Comparison");
    expect(HARNESS_SOURCE).toContain('["portable", "congruent"]');
    expect(HARNESS_SOURCE).toContain('["congruent", "portable"]');
    expect(HARNESS_SOURCE).toContain("unchangedThermalRetryPerformed: false");
    expect(HARNESS_SOURCE).not.toContain("Math.random");
    expect(HTML_SOURCE).toContain("continuous 30-second nominal interval");
    expect(HTML_SOURCE).toContain(
      'src="./opt-0015-vae-conv-transpose1d-ab.ts"',
    );
  });
});
