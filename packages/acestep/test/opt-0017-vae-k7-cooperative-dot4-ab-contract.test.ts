import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import { planAceFp16VaeConv1dSubgroupRange } from
  "../src/webgpu/kernels/vae-conv1d-fp16-subgroup.js";
import { planAceOpt0014VaeConv1dPackedKioWeight } from
  "../src/webgpu/kernels/vae-conv1d-fp16-packed-kio-subgroup.js";
import { planAceOpt0017VaeConv1dCooperativeDot4Range } from
  "../src/webgpu/kernels/vae-conv1d-fp16-cooperative-dot4.js";
import { planAceFp16VaeConv1d } from
  "../src/webgpu/kernels/vae-conv1d-fp16.js";
import {
  assertOpt0017NumericalThresholds,
  buildOpt0017C300Topology,
  buildOpt0017RepresentativeOrders,
  buildOpt0017Representatives,
  parseOpt0017ThermalGate,
  selectOpt0017RepresentativeWinner,
  summarizeOpt0017FullSequence,
  summarizeOpt0017RepresentativeTiming,
  type Opt0017RepresentativeArm,
  type Opt0017RepresentativeTimingInput,
} from "./browser/opt-0017-vae-k7-cooperative-dot4-ab.js";

const HARNESS_PATH = fileURLToPath(new URL(
  "./browser/opt-0017-vae-k7-cooperative-dot4-ab.ts",
  import.meta.url,
));
const HTML_PATH = fileURLToPath(new URL(
  "./browser/opt-0017-vae-k7-cooperative-dot4-ab.html",
  import.meta.url,
));
const CORE_PATH = fileURLToPath(new URL(
  "../src/webgpu/kernels/vae-conv1d-fp16-cooperative-dot4.ts",
  import.meta.url,
));
const HARNESS_SOURCE = readFileSync(HARNESS_PATH, "utf8");
const HTML_SOURCE = readFileSync(HTML_PATH, "utf8");
const ARMS = ["fixed32", "cooperativeDot4"] as const;

describe("OPT-0017 target-browser cooperative-dot4 K7 gate", () => {
  it("pins the 16 biased operations and all 2,399 C300 ranges", () => {
    const topology = buildOpt0017C300Topology();
    expect(topology).toHaveLength(16);
    expect(topology.every((operation) =>
      operation.outputStorage === "float16"
    )).toBe(true);
    expect(topology.reduce(
      (sum, operation) => sum + operation.ranges.length,
      0,
    )).toBe(2_399);
    expect(topology.reduce(
      (sum, operation) => sum + operation.outputElements,
      0,
    )).toBe(424_550_400);
    expect(topology.reduce((sum, operation) =>
      sum + Math.ceil(operation.ranges.length / 8), 0)).toBe(307);
    expect(topology.some(({ label }) => label === "conv2")).toBe(false);
  });

  it("freezes four tiers and the exact first/interior/tail probes", () => {
    const representatives = buildOpt0017Representatives();
    expect(representatives.map((item) => ({
      id: item.id,
      label: item.operation.label,
      weight: item.weight,
      probes: item.probes.map(({ base, count }) => [base, count]),
    }))).toEqual([
      {
        id: "c1024-d1",
        label: "block-0-res-1-conv1",
        weight: 282,
        probes: [[0, 32_768], [1_540_096, 32_768], [3_047_424, 24_576]],
      },
      {
        id: "c512-d3",
        label: "block-1-res-2-conv1",
        weight: 423,
        probes: [[0, 65_536], [4_587_520, 65_536], [9_175_040, 40_960]],
      },
      {
        id: "c256-d1",
        label: "block-2-res-1-conv1",
        weight: 423,
        probes: [[0, 131_072], [9_175_040, 131_072], [18_350_080, 81_920]],
      },
      {
        id: "c128-d9",
        label: "block-4-res-3-conv1",
        weight: 1_269,
        probes: [[0, 262_144], [36_962_304, 262_144], [73_662_464, 65_536]],
      },
    ]);
    expect(representatives.reduce((sum, item) => sum + item.weight, 0))
      .toBe(2_397);
    const valuesPerExecution = representatives.flatMap((item) => item.probes)
      .reduce((sum, probe) => sum + probe.count, 0);
    expect(valuesPerExecution).toBe(1_196_032);
    expect(valuesPerExecution * 2).toBe(2_392_064);
  });

  it("pins selected-four and all-16 repack accounting", () => {
    const topology = buildOpt0017C300Topology();
    const selectedLabels = new Set(buildOpt0017Representatives().map(
      (item) => item.operation.label,
    ));
    const all = topology.map((operation) =>
      planAceOpt0014VaeConv1dPackedKioWeight(operation.shape)
    );
    const selected = topology.filter((operation) =>
      selectedLabels.has(operation.label)
    ).map((operation) =>
      planAceOpt0014VaeConv1dPackedKioWeight(operation.shape)
    );
    const selectedWords = selected.reduce(
      (sum, plan) => sum + plan.packedWordCount * 2,
      0,
    );
    const allWords = all.reduce(
      (sum, plan) => sum + plan.packedWordCount * 2,
      0,
    );
    expect(selectedWords).toBe(9_748_480);
    expect(selected.reduce(
      (sum, plan) => sum + plan.repackWorkgroups,
      0,
    )).toBe(19_040);
    expect(allWords).toBe(30_507_008);
    expect(all.reduce(
      (sum, plan) => sum + plan.packedBindingBytes,
      0,
    )).toBe(61_014_016);
    expect(all.reduce(
      (sum, plan) => sum + plan.repackWorkgroups,
      0,
    )).toBe(59_584);
    expect(allWords + selectedWords).toBe(40_255_488);
  });

  it("uses four balanced alternating samples per arm and tier", () => {
    const orders = buildOpt0017RepresentativeOrders();
    expect(orders).toHaveLength(16);
    for (const entry of orders) {
      expect(entry.rotation).toBe((entry.tierIndex + entry.sampleIndex) % 2);
      expect(entry.order).toEqual(entry.rotation === 0
        ? ["fixed32", "cooperativeDot4"]
        : ["cooperativeDot4", "fixed32"]);
    }
    const counts = Object.fromEntries(ARMS.map((arm) =>
      [arm, [0, 0]]
    )) as Record<Opt0017RepresentativeArm, number[]>;
    for (const entry of orders) {
      entry.order.forEach((arm, position) => counts[arm]![position]! += 1);
    }
    expect(counts).toEqual({
      fixed32: [8, 8],
      cooperativeDot4: [8, 8],
    });
  });

  it("scores median-four tier totals and applies the 1.75x stop", () => {
    const inputs: Opt0017RepresentativeTimingInput[] =
      buildOpt0017Representatives().map((representative) => ({
        representativeId: representative.id,
        tier: representative.tier,
        weight: representative.weight,
        samples: {
          fixed32: [10, 12, 8, 14],
          cooperativeDot4: [5, 6, 4, 7],
        },
      }));
    const summary = summarizeOpt0017RepresentativeTiming(inputs);
    expect(summary).toMatchObject({
      representedRangeWeight: 2_397,
      omittedConv1RangeWeight: 2,
      omittedFinalConv2RangeWeight: 5,
      fixed32WeightedMilliseconds: 26_367,
      candidateWeightedMilliseconds: { cooperativeDot4: 13_183.5 },
      speedups: { cooperativeDot4: 2 },
    });
    expect(selectOpt0017RepresentativeWinner(summary)).toMatchObject({
      selectedWinner: "cooperativeDot4",
      threshold: 1.75,
    });
    expect(selectOpt0017RepresentativeWinner({
      speedups: { cooperativeDot4: 1.749 },
    })).toMatchObject({
      status: "negative-stop-no-representative-qualifier",
      selectedWinner: null,
    });
  });

  it("requires both paired wins and 2.0x aggregate including repack", () => {
    const passed = summarizeOpt0017FullSequence("cooperativeDot4", [
      {
        order: ["fixed32", "winner"],
        fixed32Milliseconds: 20,
        winnerConvolutionMilliseconds: 8,
        repackMilliseconds: 2,
      },
      {
        order: ["winner", "fixed32"],
        fixed32Milliseconds: 20,
        winnerConvolutionMilliseconds: 8,
        repackMilliseconds: 2,
      },
    ]);
    expect(passed).toMatchObject({
      aggregateSpeedup: 2,
      bothOrderWins: true,
      passed: true,
      threshold: 2,
    });
    expect(summarizeOpt0017FullSequence("cooperativeDot4", [
      {
        order: ["fixed32", "winner"],
        fixed32Milliseconds: 9,
        winnerConvolutionMilliseconds: 8,
        repackMilliseconds: 2,
      },
      {
        order: ["winner", "fixed32"],
        fixed32Milliseconds: 40,
        winnerConvolutionMilliseconds: 8,
        repackMilliseconds: 2,
      },
    ])).toMatchObject({ bothOrderWins: false, passed: false });
  });

  it("enforces every frozen numerical threshold without slack", () => {
    const accepted = {
      nrmse: 0.001,
      snrDb: 60,
      pearson: 0.99999,
      relativeMaximumAbsoluteError: 0.01,
      numericOutputRanges: {
        control: { minimum: -1, maximum: 1 },
        candidate: { minimum: -0.99, maximum: 0.99 },
      },
    };
    expect(() => assertOpt0017NumericalThresholds(accepted, "boundary"))
      .not.toThrow();
    expect(() => assertOpt0017NumericalThresholds({
      ...accepted,
      nrmse: 0.001000_001,
    }, "nrmse")).toThrow(/frozen numerical thresholds/);
    expect(() => assertOpt0017NumericalThresholds({
      ...accepted,
      snrDb: 59.999,
    }, "snr")).toThrow(/frozen numerical thresholds/);
    expect(() => assertOpt0017NumericalThresholds({
      ...accepted,
      pearson: 0.999989,
    }, "pearson")).toThrow(/frozen numerical thresholds/);
    expect(() => assertOpt0017NumericalThresholds({
      ...accepted,
      relativeMaximumAbsoluteError: 0.010001,
    }, "max-abs")).toThrow(/frozen numerical thresholds/);
    expect(() => assertOpt0017NumericalThresholds({
      ...accepted,
      numericOutputRanges: {
        ...accepted.numericOutputRanges,
        candidate: { minimum: Number.NaN, maximum: 0.99 },
      },
    }, "non-finite-range")).toThrow(/frozen numerical thresholds/);
  });

  it("pins complete full-C300 workgroup totals", () => {
    let fixed32 = 0;
    let candidate = 0;
    for (const operation of buildOpt0017C300Topology()) {
      const plan = planAceFp16VaeConv1d(operation.shape, "float16");
      for (const range of operation.ranges) {
        const fixed = planAceFp16VaeConv1dSubgroupRange(plan, range);
        const cooperative = planAceOpt0017VaeConv1dCooperativeDot4Range(
          plan,
          range,
        );
        fixed32 += fixed.workgroupsX * fixed.workgroupsY;
        candidate += cooperative.workgroupsX * cooperative.workgroupsY;
      }
    }
    expect(fixed32).toBe(103_672);
    expect(candidate).toBe(103_684);
  });

  it("accepts exactly one fresh 30-second nominal thermal gate", () => {
    const parameters = new URLSearchParams({
      thermalSource: "notifyutil-com.apple.system.thermalpressurelevel",
      thermalStartedAtEpochMilliseconds: "2000",
      thermalCompletedAtEpochMilliseconds: "32010",
      thermalObservations: "31",
      thermalPollMilliseconds: "1000",
      thermalMaximumPollGapMilliseconds: "1010",
      thermalNonNominalObservations: "0",
    });
    expect(parseOpt0017ThermalGate(parameters, 1_999, 32_020)).toMatchObject({
      durationMilliseconds: 30_010,
      observationCount: 31,
      launchDelayMilliseconds: 10,
      nonNominalObservationCount: 0,
    });
    parameters.set("thermalNonNominalObservations", "1");
    expect(() => parseOpt0017ThermalGate(parameters, 1_999, 32_020))
      .toThrow(/incomplete, stale, or non-nominal/);
  });

  it("pins auth, numerical bounds, conditional allocation, and no retry", () => {
    expect(createHash("sha256").update(readFileSync(CORE_PATH))
      .digest("hex")).toBe(
        "83987aa9b16e05a5b6f45c25ebfe33ed08bfb82ae94bd9dfb4fb624c625407b8",
      );
    expect(HARNESS_SOURCE).toContain(
      "b83f4fe94d56787ddb980629ea6f41804543ca69",
    );
    for (const pin of [
      "NRMSE_LIMIT = 0.001",
      "SNR_MINIMUM_DB = 60",
      "PEARSON_MINIMUM = 0.99999",
      "RELATIVE_MAXIMUM_ABSOLUTE_ERROR_LIMIT = 0.01",
      "FULL_REPACK_COMPARED_U16_WORDS = 40_255_488",
      "FULL_SEQUENCE_COMMAND_BUFFERS_PER_ARM = 307",
      "fullSequenceAllocationPerformed",
      "numericOutputRanges",
      "fp16UlpDistribution",
      "signedZeroDifferenceCount",
      "firstDifference",
      "worstLocation",
      "measuredAll16BeforeEachCandidateRun: true",
      "unchangedThermalRetryPerformed: false",
    ]) expect(HARNESS_SOURCE).toContain(pin);
    for (const hash of [
      "a5b263af7413fdcddb14d0cf4468360074e614f8cab1be22b4ce39e12633059b",
      "ae8c34cb62c1b3c6a512328640377b7980971d5b8a508bc7bbb7403c6ed11564",
      "baa781e62b263e89486a05b566635652db893da0064e6d9e02f9d01b97705c19",
      "fd9eb42c7e1a36829ea952af49de1c20e2755354e9263bab53a04233aa2c8f3f",
    ]) expect(HARNESS_SOURCE).toContain(hash);
    expect(HARNESS_SOURCE).not.toContain("Math.random");
    expect(HTML_SOURCE).toMatch(/continuous\s+30-second nominal interval/);
    expect(HTML_SOURCE).toContain("1.75x");
    expect(HTML_SOURCE).toContain("2.0x");
    expect(HTML_SOURCE).toMatch(/measured all-16 GPU\s+repack/);
    expect(HTML_SOURCE).toContain(
      'src="./opt-0017-vae-k7-cooperative-dot4-ab.ts"',
    );
  });
});
