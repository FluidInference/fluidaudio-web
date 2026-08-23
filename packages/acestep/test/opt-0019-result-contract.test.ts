import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";

import { describe, expect, it } from "vitest";

import {
  parseOptimizationResultJson,
  stringifyOptimizationResult,
} from "../benchmark/result.js";

interface ShapeTiming {
  readonly id: string;
  readonly productionMultiplicity: number;
  readonly feedForwardMultiplicity: number;
  readonly samples: Readonly<Record<"current" | "candidate", readonly number[]>>;
  readonly medians: Readonly<Record<"current" | "candidate", number>>;
  readonly candidateFaster: boolean;
  readonly speedup: number;
}

interface Opt0019Receipt {
  readonly experimentId: string;
  readonly status: string;
  readonly identity: {
    readonly registrationCommit: string;
    readonly currentCoreSourceSha256: string;
    readonly candidateCoreSourceSha256: string;
    readonly candidateKernelSetId: string;
  };
  readonly protocol: {
    readonly thermal: {
      readonly startedAtEpochMilliseconds: number;
      readonly completedAtEpochMilliseconds: number;
      readonly durationMilliseconds: number;
      readonly observationCount: number;
      readonly maximumPollGapMilliseconds: number;
      readonly nonNominalObservationCount: number;
      readonly launchDelayMilliseconds: number;
    };
    readonly fullOutputCorrectnessCompletedBeforeThermalGate: boolean;
    readonly bothKernelsCompiledAndWarmedBeforeThermalGate: boolean;
    readonly unchangedThermalRetryPerformed: boolean;
  };
  readonly correctness: {
    readonly shapeCount: number;
    readonly executionCount: number;
    readonly comparisonsPerOutputWord: number;
    readonly comparedU32Count: number;
    readonly mismatchCount: number;
    readonly qNaNPrefillCompleteWrites: boolean;
    readonly canariesUntouched: boolean;
    readonly finiteOutputs: boolean;
    readonly deterministicReruns: boolean;
    readonly cases: readonly Readonly<{
      comparedU32Count: number;
      mismatchCount: number;
      currentFirstRerunExact: boolean;
      candidateFirstRerunExact: boolean;
      candidateVersusCurrentExact: boolean;
      prefixCanaryIntact: boolean;
      suffixCanaryIntact: boolean;
      tailRowWritten: boolean;
    }>[];
  };
  readonly timing: {
    readonly samplesPerArmPerShape: number;
    readonly completeDense: {
      readonly currentMilliseconds: number;
      readonly candidateMilliseconds: number;
      readonly savingMilliseconds: number;
      readonly speedup: number;
      readonly speedupThreshold: number;
      readonly savingThresholdMilliseconds: number;
    };
    readonly feedForward: {
      readonly currentMilliseconds: number;
      readonly candidateMilliseconds: number;
      readonly savingMilliseconds: number;
      readonly speedup: number;
    };
    readonly everyShapeFaster: boolean;
    readonly strata: readonly ShapeTiming[];
    readonly passed: boolean;
    readonly decision: string;
    readonly timingStartedAtEpochMilliseconds: number;
    readonly timingCompletedAtEpochMilliseconds: number;
  };
  readonly cleanup: {
    readonly createdBufferCount: number;
    readonly destroyedBufferCount: number;
    readonly liveBufferCount: number;
    readonly liveBytes: number;
    readonly maximumLiveBytes: number;
    readonly idempotent: boolean;
    readonly deviceDestroyed: boolean;
  };
  readonly decision: {
    readonly packageNativeEscalationAuthorized: boolean;
    readonly productionIntegrationAuthorized: boolean;
    readonly m2250IntegrationRunAuthorized: boolean;
  };
  readonly recordedAt: string;
}

interface ThermalObservation {
  readonly index: number;
  readonly epochMilliseconds: number;
  readonly level: number;
}

interface ResultView {
  readonly protocol: {
    readonly authoritativeTimedRunCount: number;
    readonly unchangedThermalRetryPerformed: boolean;
    readonly rejectedPreflight: {
      readonly count: number;
      readonly reason: string;
      readonly timedDispatchCount: number;
      readonly timingSampleCount: number;
    };
  };
  readonly metrics: {
    readonly candidate: {
      readonly qualified: boolean;
      readonly packageNativeEscalationPerformed: boolean;
      readonly productionIntegrationPerformed: boolean;
      readonly m2250IntegrationRunPerformed: boolean;
    };
    readonly delta: {
      readonly completeDenseSavingMilliseconds: number;
      readonly completeDenseSpeedup: number;
      readonly savingConditionPassed: boolean;
      readonly speedupConditionPassed: boolean;
      readonly stopRuleFired: boolean;
    };
  };
  readonly correctness: {
    readonly primitive: {
      readonly comparedU32Count: number;
      readonly mismatchCount: number;
      readonly candidateVersusCurrentRawU32Exact: boolean;
    };
  };
}

const RESULT_URL = new URL(
  "../optimization/results/OPT-0019/result.json",
  import.meta.url,
);
const RECEIPT_URL = new URL(
  "../optimization/artifacts/OPT-0019/raw/dense-cooperative-panels-ab.json",
  import.meta.url,
);
const THERMAL_URL = new URL(
  "../optimization/artifacts/OPT-0019/raw/dense-cooperative-panels-thermal.jsonl",
  import.meta.url,
);
const RECORD_URL = new URL(
  "../optimization/experiments/OPT-0019-dit-dense-cooperative-panels.md",
  import.meta.url,
);
const LEDGER_URL = new URL("../optimization/LEDGER.md", import.meta.url);

const sha256 = (bytes: Uint8Array | string): string =>
  createHash("sha256").update(bytes).digest("hex");
const median4 = (samples: readonly number[]): number => {
  expect(samples).toHaveLength(4);
  const sorted = [...samples].sort((left, right) => left - right);
  return (sorted[1]! + sorted[2]!) / 2;
};

describe("OPT-0019 optimization result", () => {
  it("commits canonical exact-but-negative evidence and the stop decision", async () => {
    const text = await readFile(RESULT_URL, "utf8");
    const committed = parseOptimizationResultJson(text);
    const view = committed as unknown as ResultView;

    expect(committed).toMatchObject({
      experimentId: "OPT-0019",
      riskClass: "exact",
      baselineCommit: "f92de5a209ebb5f05ba9b37e5f3b7bfc88633d82",
      candidateCommit: "8900ce670271cbd227142c584fb4a917e5d9cfb9",
      evidence: { conclusion: "negative" },
      disposition: { state: "abandoned" },
      identity: {
        browserVersion:
          "Google Chrome 151.0.7922.138; reduced user agent 151.0.0.0",
        benchmarkHarnessCommit:
          "8900ce670271cbd227142c584fb4a917e5d9cfb9",
        currentCoreSourceSha256:
          "a238f67da07c6ba1097da9d9e9e97960ae97d2e1d5c129fcbabf69e962cbb6b3",
        candidateCoreSourceSha256:
          "b5dad12724882d3fc942c7df7b10c7b7b89a4bed595125ff11a5905c03152a37",
        productionPackageLoaded: false,
        productionRuntimeSelected: false,
        productionIntegrationPerformed: false,
      },
      correctness: { passed: true, listeningRequired: false },
    });
    expect(committed.artifacts).toEqual([
      {
        location:
          "optimization/artifacts/OPT-0019/raw/dense-cooperative-panels-ab.json",
        sha256:
          "a2434a16dae3db3936202461a6da6548009148c739da58dd80f458931f55471d",
      },
      {
        location:
          "optimization/artifacts/OPT-0019/raw/dense-cooperative-panels-thermal.jsonl",
        sha256:
          "ebb0e4000e68338cd29cd5a888e0ee635cb95f54a745eec25612d2bf26e4bf9f",
      },
    ]);
    expect(view.correctness.primitive).toMatchObject({
      comparedU32Count: 101_376_000,
      mismatchCount: 0,
      candidateVersusCurrentRawU32Exact: true,
    });
    expect(view.metrics.delta).toMatchObject({
      completeDenseSavingMilliseconds: 53.50000023841858,
      completeDenseSpeedup: 1.3152622288873117,
      savingConditionPassed: true,
      speedupConditionPassed: false,
      stopRuleFired: true,
    });
    expect(view.metrics.candidate).toMatchObject({
      qualified: false,
      packageNativeEscalationPerformed: false,
      productionIntegrationPerformed: false,
      m2250IntegrationRunPerformed: false,
    });
    expect(view.protocol).toMatchObject({
      rejectedPreflight: {
        count: 1,
        reason: "stale thermal timestamps",
        timedDispatchCount: 0,
        timingSampleCount: 0,
      },
      authoritativeTimedRunCount: 1,
      unchangedThermalRetryPerformed: false,
    });
    expect(`${stringifyOptimizationResult(committed)}\n`).toBe(text);
  });

  it("recomputes exactness, median4 scores, thermal coverage, and governance", async () => {
    const [receiptBytes, thermalBytes, record, ledger] = await Promise.all([
      readFile(RECEIPT_URL),
      readFile(THERMAL_URL),
      readFile(RECORD_URL, "utf8"),
      readFile(LEDGER_URL, "utf8"),
    ]);
    expect(receiptBytes.byteLength).toBe(14_126);
    expect(sha256(receiptBytes)).toBe(
      "a2434a16dae3db3936202461a6da6548009148c739da58dd80f458931f55471d",
    );
    expect(thermalBytes.byteLength).toBe(8_590);
    expect(sha256(thermalBytes)).toBe(
      "ebb0e4000e68338cd29cd5a888e0ee635cb95f54a745eec25612d2bf26e4bf9f",
    );

    const receipt = JSON.parse(receiptBytes.toString("utf8")) as Opt0019Receipt;
    expect(receipt).toMatchObject({
      experimentId: "OPT-0019",
      status: "passed",
      identity: {
        registrationCommit: "83de738b5374699778dcaa373d69118a7fbd6715",
      },
      correctness: {
        shapeCount: 4,
        executionCount: 16,
        comparisonsPerOutputWord: 4,
        comparedU32Count: 101_376_000,
        mismatchCount: 0,
        qNaNPrefillCompleteWrites: true,
        canariesUntouched: true,
        finiteOutputs: true,
        deterministicReruns: true,
      },
      timing: {
        samplesPerArmPerShape: 4,
        passed: false,
        decision: "negative-stop-primitive-gate",
      },
      cleanup: {
        createdBufferCount: 20,
        destroyedBufferCount: 20,
        liveBufferCount: 0,
        liveBytes: 0,
        idempotent: true,
        deviceDestroyed: true,
      },
      decision: {
        packageNativeEscalationAuthorized: false,
        productionIntegrationAuthorized: false,
        m2250IntegrationRunAuthorized: false,
      },
    });
    expect(receipt.correctness.cases).toHaveLength(4);
    expect(receipt.correctness.cases.reduce(
      (sum, entry) => sum + entry.comparedU32Count,
      0,
    )).toBe(receipt.correctness.comparedU32Count);
    expect(receipt.correctness.cases.every((entry) =>
      entry.mismatchCount === 0 &&
      entry.currentFirstRerunExact &&
      entry.candidateFirstRerunExact &&
      entry.candidateVersusCurrentExact &&
      entry.prefixCanaryIntact &&
      entry.suffixCanaryIntact &&
      entry.tailRowWritten
    )).toBe(true);

    for (const stratum of receipt.timing.strata) {
      expect(median4(stratum.samples.current)).toBe(stratum.medians.current);
      expect(median4(stratum.samples.candidate)).toBe(
        stratum.medians.candidate,
      );
      expect(stratum.medians.candidate).toBeLessThan(stratum.medians.current);
      expect(stratum.medians.current / stratum.medians.candidate).toBe(
        stratum.speedup,
      );
    }
    const weighted = (arm: "current" | "candidate", feedForward: boolean): number =>
      receipt.timing.strata.reduce((sum, stratum) =>
        sum + stratum.medians[arm] * (feedForward
          ? stratum.feedForwardMultiplicity
          : stratum.productionMultiplicity), 0);
    expect(weighted("current", false)).toBe(223.20000022649765);
    expect(weighted("candidate", false)).toBe(169.69999998807907);
    expect(weighted("current", true)).toBe(138.79999989271164);
    expect(weighted("candidate", true)).toBe(109.99999994039536);
    expect(receipt.timing.completeDense).toMatchObject({
      savingMilliseconds: 53.50000023841858,
      speedup: 1.3152622288873117,
      speedupThreshold: 1.55,
      savingThresholdMilliseconds: 52.0834,
    });
    expect(receipt.timing.everyShapeFaster).toBe(true);
    expect(receipt.timing.completeDense.savingMilliseconds).toBeGreaterThanOrEqual(
      receipt.timing.completeDense.savingThresholdMilliseconds,
    );
    expect(receipt.timing.completeDense.speedup).toBeLessThan(
      receipt.timing.completeDense.speedupThreshold,
    );

    const thermal = thermalBytes.toString("utf8").trim().split("\n")
      .map((line) => JSON.parse(line) as ThermalObservation);
    expect(thermal).toHaveLength(150);
    expect(thermal.every((entry, index) =>
      entry.index === index && entry.level === 0
    )).toBe(true);
    const gate = thermal.filter((entry) =>
      entry.epochMilliseconds >=
        receipt.protocol.thermal.startedAtEpochMilliseconds &&
      entry.epochMilliseconds <=
        receipt.protocol.thermal.completedAtEpochMilliseconds
    );
    expect(gate).toHaveLength(45);
    const maximumPollGap = Math.max(...gate.slice(1).map((entry, index) =>
      entry.epochMilliseconds - gate[index]!.epochMilliseconds
    ));
    expect(maximumPollGap).toBe(1_012);
    expect(receipt.protocol.thermal).toMatchObject({
      durationMilliseconds: 44_406,
      observationCount: 45,
      maximumPollGapMilliseconds: 1_012,
      nonNominalObservationCount: 0,
      launchDelayMilliseconds: 907,
    });
    expect(receipt.protocol.unchangedThermalRetryPerformed).toBe(false);
    expect(receipt.timing.timingStartedAtEpochMilliseconds).toBeGreaterThan(
      receipt.protocol.thermal.completedAtEpochMilliseconds,
    );
    expect(thermal.at(-1)!.epochMilliseconds).toBeGreaterThan(
      receipt.timing.timingCompletedAtEpochMilliseconds,
    );
    expect(thermal.at(-1)!.epochMilliseconds).toBeGreaterThan(
      Date.parse(receipt.recordedAt),
    );

    expect(record).toContain("- Evidence: `negative`");
    expect(record).toContain("- Disposition: `abandoned`");
    expect(record).toContain("rejected by the gate's stale-timestamp preflight");
    expect(record).toContain("authoritative attempt was the sole timed run");
    expect(ledger).toContain(
      "| OPT-0019 | DiT dense GEMM | A WG256 M64xN128xK16 cooperative FP16 panel",
    );
    expect(ledger).toContain("| negative | abandoned |");
  });
});
