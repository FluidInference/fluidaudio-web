import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import { planAceOpt0009DenseGemm } from
  "../src/webgpu/kernels/dit-dense-fp16.js";
import { planAceOpt0019DenseCooperativePanels } from
  "../src/webgpu/kernels/dit-dense-fp16-cooperative-panels.js";
import {
  buildOpt0019ShapeSpecs,
  buildOpt0019TimingOrders,
  parseOpt0019ThermalGate,
  summarizeOpt0019Timing,
  type Opt0019Arm,
  type Opt0019TimingInput,
} from "./browser/opt-0019-dit-dense-cooperative-panels-ab.js";

const HARNESS_PATH = fileURLToPath(new URL(
  "./browser/opt-0019-dit-dense-cooperative-panels-ab.ts",
  import.meta.url,
));
const HTML_PATH = fileURLToPath(new URL(
  "./browser/opt-0019-dit-dense-cooperative-panels-ab.html",
  import.meta.url,
));
const CORE_PATH = fileURLToPath(new URL(
  "../src/webgpu/kernels/dit-dense-fp16-cooperative-panels.ts",
  import.meta.url,
));
const HARNESS_SOURCE = readFileSync(HARNESS_PATH, "utf8");
const HTML_SOURCE = readFileSync(HTML_PATH, "utf8");
const ARMS = ["current", "candidate"] as const;

describe("OPT-0019 target-browser cooperative dense-panel gate", () => {
  it("pins all four M2250 shapes and both registered scores", () => {
    expect(buildOpt0019ShapeSpecs()).toEqual([
      {
        id: "h-h",
        shape: { rows: 2_250, inner: 2_048, columns: 2_048 },
        productionMultiplicity: 4,
        feedForwardMultiplicity: 0,
      },
      {
        id: "h-1024",
        shape: { rows: 2_250, inner: 2_048, columns: 1_024 },
        productionMultiplicity: 2,
        feedForwardMultiplicity: 0,
      },
      {
        id: "h-6144",
        shape: { rows: 2_250, inner: 2_048, columns: 6_144 },
        productionMultiplicity: 2,
        feedForwardMultiplicity: 2,
      },
      {
        id: "6144-h",
        shape: { rows: 2_250, inner: 6_144, columns: 2_048 },
        productionMultiplicity: 1,
        feedForwardMultiplicity: 1,
      },
    ]);
  });

  it("pins current and candidate tile/work accounting", () => {
    let currentWorkgroups = 0;
    let candidateWorkgroups = 0;
    let currentMacs = 0;
    let candidateMacs = 0;
    for (const spec of buildOpt0019ShapeSpecs()) {
      const current = planAceOpt0009DenseGemm(spec.shape);
      const candidate = planAceOpt0019DenseCooperativePanels(spec.shape);
      expect(current).toMatchObject({
        tileRows: 32,
        tileColumns: 256,
        tileInner: 32,
        workgroupSize: 128,
      });
      expect(candidate).toMatchObject({
        tileRows: 64,
        tileColumns: 128,
        tileInner: 16,
        workgroupSize: 256,
      });
      currentWorkgroups += current.workgroupCount * spec.productionMultiplicity;
      candidateWorkgroups += candidate.workgroupCount *
        spec.productionMultiplicity;
      currentMacs += current.outputRanges[0]!.multiplyAdds *
        spec.productionMultiplicity;
      candidateMacs += candidate.outputRanges[0]!.multiplyAdds *
        spec.productionMultiplicity;
    }
    expect(currentWorkgroups).toBe(6_816);
    expect(candidateWorkgroups).toBe(6_912);
    expect(currentMacs).toBe(133_412_421_632);
    expect(candidateMacs).toBe(135_291_469_824);
  });

  it("rotates shape order left and alternates whole-round AB/BA", () => {
    const orders = buildOpt0019TimingOrders();
    expect(orders).toHaveLength(16);
    const positionCounts = Object.fromEntries(ARMS.map((arm) =>
      [arm, [0, 0]]
    )) as Record<Opt0019Arm, number[]>;
    expect(Array.from({ length: 4 }, (_, roundIndex) =>
      orders.filter((entry) => entry.roundIndex === roundIndex)
        .map((entry) => entry.shapeIndex)
    )).toEqual([
      [0, 1, 2, 3],
      [1, 2, 3, 0],
      [2, 3, 0, 1],
      [3, 0, 1, 2],
    ]);
    for (const entry of orders) {
      expect(entry.order).toEqual(entry.roundIndex % 2 === 0
        ? ["current", "candidate"]
        : ["candidate", "current"]);
      entry.order.forEach((arm, position) => {
        positionCounts[arm]![position]! += 1;
      });
    }
    expect(positionCounts).toEqual({
      current: [8, 8],
      candidate: [8, 8],
    });
  });

  it("uses median four and enforces no regression plus both dense thresholds", () => {
    const passing = timingInputs([
      [10, 6],
      [10, 6],
      [30, 18],
      [35, 20],
    ]);
    expect(summarizeOpt0019Timing(passing)).toMatchObject({
      completeDense: {
        currentMilliseconds: 155,
        candidateMilliseconds: 92,
        savingMilliseconds: 63,
        speedup: 155 / 92,
        speedupThreshold: 1.55,
        savingThresholdMilliseconds: 52.0834,
      },
      feedForward: {
        currentMilliseconds: 95,
        candidateMilliseconds: 56,
      },
      everyShapeFaster: true,
      passed: true,
      decision: "positive-primitive-qualifier",
    });
    const regression = timingInputs([
      [10, 10.001],
      [10, 2],
      [30, 10],
      [35, 10],
    ]);
    expect(summarizeOpt0019Timing(regression)).toMatchObject({
      everyShapeFaster: false,
      passed: false,
    });
    const insufficientAbsolute = timingInputs([
      [5, 3],
      [5, 3],
      [10, 6],
      [10, 6],
    ]);
    expect(summarizeOpt0019Timing(insufficientAbsolute)).toMatchObject({
      completeDense: { speedup: 5 / 3, savingMilliseconds: 24 },
      everyShapeFaster: true,
      passed: false,
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
    expect(parseOpt0019ThermalGate(parameters, 1_999, 32_020)).toMatchObject({
      durationMilliseconds: 30_010,
      observationCount: 31,
      launchDelayMilliseconds: 10,
      nonNominalObservationCount: 0,
    });
    parameters.set("thermalNonNominalObservations", "1");
    expect(() => parseOpt0019ThermalGate(parameters, 1_999, 32_020))
      .toThrow(/incomplete, stale, or non-nominal/);
  });

  it("freezes full-output exactness, compact receipt, cleanup, and no retry", () => {
    expect(createHash("sha256").update(readFileSync(CORE_PATH)).digest("hex"))
      .toBe("b5dad12724882d3fc942c7df7b10c7b7b89a4bed595125ff11a5905c03152a37");
    for (const pin of [
      "83de738b5374699778dcaa373d69118a7fbd6715",
      "comparisonCount: 4",
      "comparisonsPerOutputWord: 4",
      "OUTPUT_PREFILL_QNAN_U32",
      "prefixCanaryIntact",
      "suffixCanaryIntact",
      "tailRowWritten",
      "candidateFirstRerunExact",
      "completeDenseSpeedup >= COMPLETE_DENSE_SPEEDUP_THRESHOLD",
      "completeDenseSavingMilliseconds >= COMPLETE_DENSE_SAVING_THRESHOLD_MS",
      "unchangedThermalRetryPerformed: false",
      "productionIntegrationAuthorized: false",
      "m2250IntegrationRunAuthorized: false",
      "__ACE_OPT0019_RESULT__",
    ]) expect(HARNESS_SOURCE).toContain(pin);
    expect(HARNESS_SOURCE).not.toContain("Math.random");
    expect(HTML_SOURCE).toMatch(/continuous 30-second nominal/);
    expect(HTML_SOURCE).toContain("1.55x");
    expect(HTML_SOURCE).toContain("52.0834 ms");
    expect(HTML_SOURCE).toContain(
      'src="./opt-0019-dit-dense-cooperative-panels-ab.ts"',
    );
  });
});

function timingInputs(
  medians: readonly (readonly [number, number])[],
): Opt0019TimingInput[] {
  return buildOpt0019ShapeSpecs().map((spec, index) => {
    const values = medians[index]!;
    return {
      id: spec.id,
      samples: {
        current: [values[0] - 1, values[0], values[0], values[0] + 1],
        candidate: [values[1] - 1, values[1], values[1], values[1] + 1],
      },
    };
  });
}
