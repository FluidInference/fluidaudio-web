import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import { planAceFp16VaeConv1dSubgroupRange } from
  "../src/webgpu/kernels/vae-conv1d-fp16-subgroup.js";
import { planAceOpt0014VaeConv1dPackedKioWeight } from
  "../src/webgpu/kernels/vae-conv1d-fp16-packed-kio-subgroup.js";
import { planAceOpt0016VaeConv1dPackedKioMicrotileRange } from
  "../src/webgpu/kernels/vae-conv1d-fp16-packed-kio-microtile-subgroup.js";
import { planAceFp16VaeConv1d } from
  "../src/webgpu/kernels/vae-conv1d-fp16.js";
import {
  buildOpt0016C300Topology,
  buildOpt0016RepresentativeOrders,
  buildOpt0016Representatives,
  parseOpt0016ThermalGate,
  selectOpt0016RepresentativeWinner,
  summarizeOpt0016FullSequence,
  summarizeOpt0016RepresentativeTiming,
  type Opt0016CandidateArm,
  type Opt0016RepresentativeArm,
  type Opt0016RepresentativeTimingInput,
} from "./browser/opt-0016-vae-k7-microtile-ab.js";

const HARNESS_PATH = fileURLToPath(new URL(
  "./browser/opt-0016-vae-k7-microtile-ab.ts",
  import.meta.url,
));
const HTML_PATH = fileURLToPath(new URL(
  "./browser/opt-0016-vae-k7-microtile-ab.html",
  import.meta.url,
));
const MICRO_CORE_PATH = fileURLToPath(new URL(
  "../src/webgpu/kernels/vae-conv1d-fp16-packed-kio-microtile-subgroup.ts",
  import.meta.url,
));
const HARNESS_SOURCE = readFileSync(HARNESS_PATH, "utf8");
const HTML_SOURCE = readFileSync(HTML_PATH, "utf8");
const ARMS = ["packed16x64", "8x64", "16x32", "8x32"] as const;
const CANDIDATES = ["8x64", "16x32", "8x32"] as const;

describe("OPT-0016 target-browser packed-KIO K7 microtile gate", () => {
  it("pins all 17 operations, 2,404 ranges, and complete output footprint", () => {
    const topology = buildOpt0016C300Topology();
    expect(topology).toHaveLength(17);
    expect(topology.reduce(
      (sum, operation) => sum + operation.ranges.length,
      0,
    )).toBe(2_404);
    expect(topology.reduce(
      (sum, operation) => sum + operation.outputElements,
      0,
    )).toBe(425_702_400);
    expect(topology.filter((operation) =>
      operation.outputStorage === "float16"
    ).reduce((sum, operation) => sum + operation.outputElements, 0))
      .toBe(424_550_400);
    expect(topology.filter((operation) =>
      operation.outputStorage === "float32"
    ).reduce((sum, operation) => sum + operation.outputElements, 0))
      .toBe(1_152_000);
    expect(topology.reduce((sum, operation) =>
      sum + Math.ceil(operation.ranges.length / 8), 0)).toBe(308);
  });

  it("freezes five timing tiers and first/interior/tail raw-bit probes", () => {
    const representatives = buildOpt0016Representatives();
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
      {
        id: "final-c2-d1",
        label: "conv2",
        weight: 5,
        probes: [[0, 262_144], [524_288, 262_144], [1_048_576, 103_424]],
      },
    ]);
    expect(representatives.reduce((sum, item) => sum + item.weight, 0))
      .toBe(2_402);
    const f16ProbeWords = representatives.filter((item) =>
      item.operation.outputStorage === "float16"
    ).flatMap((item) => item.probes).reduce(
      (sum, probe) => sum + probe.count,
      0,
    );
    const u32ProbeWords = representatives.filter((item) =>
      item.operation.outputStorage === "float32"
    ).flatMap((item) => item.probes).reduce(
      (sum, probe) => sum + probe.count,
      0,
    );
    expect(f16ProbeWords).toBe(1_196_032);
    expect(u32ProbeWords).toBe(627_712);
    expect(f16ProbeWords * CANDIDATES.length * 2).toBe(7_176_192);
    expect(u32ProbeWords * CANDIDATES.length * 2).toBe(3_766_272);
  });

  it("pins selected-five and all-17 repack accounting", () => {
    const topology = buildOpt0016C300Topology();
    const selectedLabels = new Set(buildOpt0016Representatives().map(
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
    expect(selected.reduce(
      (sum, plan) => sum + plan.packedWordCount * 2,
      0,
    )).toBe(9_750_272);
    expect(selected.reduce(
      (sum, plan) => sum + plan.repackWorkgroups,
      0,
    )).toBe(19_044);
    expect(9_750_272 * 2).toBe(19_500_544);
    expect(all.reduce(
      (sum, plan) => sum + plan.packedWordCount * 2,
      0,
    )).toBe(30_508_800);
    expect(all.reduce(
      (sum, plan) => sum + plan.packedBindingBytes,
      0,
    )).toBe(61_017_600);
    expect(all.reduce(
      (sum, plan) => sum + plan.repackWorkgroups,
      0,
    )).toBe(59_588);
    expect(9_750_272 * 2 + (30_508_800 - 9_750_272)).toBe(40_259_072);
  });

  it("uses the pinned cyclic rotation with balanced position counts", () => {
    const orders = buildOpt0016RepresentativeOrders();
    expect(orders).toHaveLength(15);
    for (const entry of orders) {
      expect(entry.rotation).toBe((entry.tierIndex + entry.sampleIndex) % 4);
    }
    const counts = Object.fromEntries(ARMS.map((arm) =>
      [arm, [0, 0, 0, 0]]
    )) as Record<Opt0016RepresentativeArm, number[]>;
    for (const entry of orders) {
      entry.order.forEach((arm, position) => counts[arm]![position]! += 1);
    }
    expect(counts).toEqual({
      packed16x64: [4, 3, 4, 4],
      "8x64": [4, 4, 3, 4],
      "16x32": [4, 4, 4, 3],
      "8x32": [3, 4, 4, 4],
    });
  });

  it("scores median-three tier totals and applies threshold/primary tie-break", () => {
    const inputs: Opt0016RepresentativeTimingInput[] =
      buildOpt0016Representatives().map((representative) => ({
        representativeId: representative.id,
        tier: representative.tier,
        weight: representative.weight,
        samples: {
          packed16x64: [12, 10, 11],
          "8x64": [10, 9, 8],
          "16x32": [9, 8, 7],
          "8x32": [11, 10, 9],
        },
      }));
    const summary = summarizeOpt0016RepresentativeTiming(inputs);
    expect(summary).toMatchObject({
      representedRangeWeight: 2_402,
      omittedConv1RangeWeight: 2,
      packed16x64WeightedMilliseconds: 26_422,
      candidateWeightedMilliseconds: {
        "8x64": 21_618,
        "16x32": 19_216,
        "8x32": 24_020,
      },
    });
    expect(selectOpt0016RepresentativeWinner(summary)).toMatchObject({
      selectedWinner: "16x32",
      threshold: 1.15,
    });
    expect(selectOpt0016RepresentativeWinner({
      speedups: { "8x64": 1.2, "16x32": 1.18, "8x32": 1.1 },
    })).toMatchObject({
      bestArm: "8x64",
      selectedWinner: "16x32",
      withinTwoPercent: ["8x64", "16x32"],
    });
    expect(selectOpt0016RepresentativeWinner({
      speedups: { "8x64": 1.14, "16x32": 1.1, "8x32": 1.0 },
    })).toMatchObject({
      status: "negative-stop-no-representative-qualifier",
      selectedWinner: null,
    });
  });

  it("requires AB/BA pair wins and includes each all-17 repack at 1.25x", () => {
    const passed = summarizeOpt0016FullSequence("16x32", [
      {
        order: ["fixed32", "winner"],
        fixed32Milliseconds: 10,
        winnerConvolutionMilliseconds: 6,
        repackMilliseconds: 2,
      },
      {
        order: ["winner", "fixed32"],
        fixed32Milliseconds: 10,
        winnerConvolutionMilliseconds: 6,
        repackMilliseconds: 2,
      },
    ]);
    expect(passed).toMatchObject({
      winner: "16x32",
      winnerTotalMilliseconds: 16,
      fixed32TotalMilliseconds: 20,
      aggregateSpeedup: 1.25,
      bothOrderWins: true,
      passed: true,
      decision: "positive-full-sequence-qualifier",
    });
    const pairLoss = summarizeOpt0016FullSequence("8x64", [
      {
        order: ["fixed32", "winner"],
        fixed32Milliseconds: 7,
        winnerConvolutionMilliseconds: 6,
        repackMilliseconds: 2,
      },
      {
        order: ["winner", "fixed32"],
        fixed32Milliseconds: 30,
        winnerConvolutionMilliseconds: 6,
        repackMilliseconds: 2,
      },
    ]);
    expect(pairLoss).toMatchObject({ bothOrderWins: false, passed: false });
  });

  it("pins complete full-C300 workgroup totals for every candidate", () => {
    const topology = buildOpt0016C300Topology();
    let fixed32 = 0;
    const candidates: Record<Opt0016CandidateArm, number> = {
      "8x64": 0,
      "16x32": 0,
      "8x32": 0,
    };
    for (const operation of topology) {
      const plan = planAceFp16VaeConv1d(
        operation.shape,
        operation.outputStorage,
      );
      for (const range of operation.ranges) {
        const fixed = planAceFp16VaeConv1dSubgroupRange(plan, range);
        fixed32 += fixed.workgroupsX * fixed.workgroupsY;
        for (const candidate of CANDIDATES) {
          const value = planAceOpt0016VaeConv1dPackedKioMicrotileRange(
            plan,
            candidate,
            range,
          );
          candidates[candidate] += value.workgroupsX * value.workgroupsY;
        }
      }
    }
    expect(fixed32).toBe(121_672);
    expect(candidates).toEqual({
      "8x64": 225_304,
      "16x32": 216_316,
      "8x32": 432_608,
    });
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
    expect(parseOpt0016ThermalGate(parameters, 1_999, 32_020)).toMatchObject({
      durationMilliseconds: 30_010,
      observationCount: 31,
      launchDelayMilliseconds: 10,
      nonNominalObservationCount: 0,
    });
    parameters.set("thermalNonNominalObservations", "1");
    expect(() => parseOpt0016ThermalGate(parameters, 1_999, 32_020))
      .toThrow(/incomplete, stale, or non-nominal/);
  });

  it("pins frozen identity, conditional allocation, guards, and no retry", () => {
    expect(createHash("sha256").update(readFileSync(MICRO_CORE_PATH))
      .digest("hex")).toBe(
        "65658d149b31ca01346011ba83a7186f4550819e16bb4726e3d177d734b5cce3",
      );
    expect(HARNESS_SOURCE).toContain(
      "997891de0fe449c9b6551e80abc55604256969ad",
    );
    expect(HARNESS_SOURCE).toContain("remainingOperationAllocationPerformed");
    expect(HARNESS_SOURCE).toContain("MAX_QUANTA_PER_COMMAND_BUFFER = 8");
    expect(HARNESS_SOURCE).toContain("FULL_SEQUENCE_COMMAND_BUFFERS_PER_ARM = 308");
    expect(HARNESS_SOURCE).toContain("C300_F16_OUTPUT_WORD_COUNT = 424_550_400");
    expect(HARNESS_SOURCE).toContain("C300_F32_OUTPUT_WORD_COUNT = 1_152_000");
    expect(HARNESS_SOURCE).toContain("FULL_REPACK_COMPARED_U16_WORDS = 40_259_072");
    for (const hash of [
      "a5b263af7413fdcddb14d0cf4468360074e614f8cab1be22b4ce39e12633059b",
      "ae8c34cb62c1b3c6a512328640377b7980971d5b8a508bc7bbb7403c6ed11564",
      "baa781e62b263e89486a05b566635652db893da0064e6d9e02f9d01b97705c19",
      "fd9eb42c7e1a36829ea952af49de1c20e2755354e9263bab53a04233aa2c8f3f",
      "956d3041fcc5194c609fc300b520e088cc568fe1107fc7ba742d2efc48ec4de3",
    ]) expect(HARNESS_SOURCE).toContain(hash);
    expect(HARNESS_SOURCE).toContain("measuredAll17BeforeEachWinnerRun: true");
    expect(HARNESS_SOURCE).toContain("qNaNPrefillCompleteWrites: true");
    expect(HARNESS_SOURCE).toContain("redzonesUntouched: true");
    expect(HARNESS_SOURCE).toContain("unchangedThermalRetryPerformed: false");
    expect(HARNESS_SOURCE).not.toContain("Math.random");
    expect(HTML_SOURCE).toContain("continuous 30-second nominal interval");
    expect(HTML_SOURCE).toMatch(/stops before allocating the\s+remaining twelve/);
    expect(HTML_SOURCE).toContain("measured all-17 repack before each candidate run");
    expect(HTML_SOURCE).toContain(
      'src="./opt-0016-vae-k7-microtile-ab.ts"',
    );
  });
});
