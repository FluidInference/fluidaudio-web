import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";

import { describe, expect, it } from "vitest";

import {
  parseOptimizationResultJson,
  stringifyOptimizationResult,
} from "../benchmark/result.js";

type Arm = "packed16x64" | "8x64" | "16x32" | "8x32";

interface ArtifactProbe {
  readonly count: number;
  readonly armSha256: Readonly<Record<Arm, string>>;
}

interface ArtifactCase {
  readonly outputStorage: "float16" | "float32";
  readonly probes: readonly ArtifactProbe[];
}

interface TimingStratum {
  readonly weight: number;
  readonly samples: Readonly<Record<Arm, readonly number[]>>;
  readonly medians: Readonly<Record<Arm, number>>;
}

interface Opt0016Artifact {
  readonly status: string;
  readonly correctness: {
    readonly cases: readonly ArtifactCase[];
    readonly mismatchCount: number;
  };
  readonly representativeTiming: {
    readonly representedRangeWeight: number;
    readonly packed16x64WeightedMilliseconds: number;
    readonly candidateWeightedMilliseconds: Readonly<Record<
      Exclude<Arm, "packed16x64">,
      number
    >>;
    readonly speedups: Readonly<Record<Exclude<Arm, "packed16x64">, number>>;
    readonly strata: readonly TimingStratum[];
  };
  readonly repack: {
    readonly selectedFiveCorrectness: {
      readonly uniqueU16WordCount: number;
      readonly comparedU16WordCount: number;
      readonly repackWorkgroupCount: number;
      readonly mismatchCount: number;
      readonly cases: readonly Readonly<{
        label: string;
        uniqueU16WordCount: number;
        comparedU16WordCount: number;
        repackWorkgroups: number;
        sha256: string;
      }>[];
    };
  };
  readonly selection: {
    readonly status: string;
    readonly selectedWinner: string | null;
    readonly qualifyingArms: readonly string[];
    readonly threshold: number;
  };
  readonly remainingOperationAllocationPerformed: boolean;
  readonly fullSequence: unknown;
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
    readonly unchangedThermalRetryPerformed: boolean;
  };
  readonly cleanup: {
    readonly createdBufferCount: number;
    readonly destroyedBufferCount: number;
    readonly liveBufferCount: number;
    readonly liveBytes: number;
    readonly idempotent: boolean;
    readonly deviceDestroyed: boolean;
  };
}

const ARMS = ["packed16x64", "8x64", "16x32", "8x32"] as const;
const CANDIDATES = ["8x64", "16x32", "8x32"] as const;

describe("OPT-0016 optimization result", () => {
  it("commits canonical exact but negative microtile evidence", async () => {
    const text = await readFile(new URL(
      "../optimization/results/OPT-0016/result.json",
      import.meta.url,
    ), "utf8");
    const committed = parseOptimizationResultJson(text);

    expect(committed).toMatchObject({
      experimentId: "OPT-0016",
      riskClass: "exact",
      baselineCommit: "12e128ab323c0024ed683313b4d06c07041213e7",
      candidateCommit: "997891de0fe449c9b6551e80abc55604256969ad",
      identity: {
        benchmarkHarnessCommit:
          "085669d5aec0fc02f3268c8b462385b59fb72ab7",
        packedControlCommit:
          "12e128ab323c0024ed683313b4d06c07041213e7",
        evaluatedProductionBaselineCommit:
          "36608b857827b2b1d31ac91bf5cca9639fb0b9ed",
        microtileCoreSourceSha256:
          "65658d149b31ca01346011ba83a7186f4550819e16bb4726e3d177d734b5cce3",
        productionPackageLoaded: false,
        productionRuntimeSelected: false,
        productionIntegrationPerformed: false,
      },
      protocol: {
        samples: 3,
        thermalGateSeconds: 30,
        thermal: {
          durationMilliseconds: 30_130,
          observationCount: 31,
          maximumPollGapMilliseconds: 1_005,
          nonNominalObservationCount: 0,
          launchDelayMilliseconds: 76,
        },
        unchangedThermalRetryPerformed: false,
        qualificationThreshold: 1.15,
      },
      metrics: {
        baseline: {
          representativeRangeWeight: 2_402,
          weightedRepresentativeMilliseconds: 3_331.9999829530716,
          selectedFivePackedPayloadBytes: 19_500_544,
          fullC300PackedPayloadBytesIfFullPhaseRuns: 61_017_600,
        },
        candidate: {
          bestArm: "8x32",
          bestSpeedup: 1.007712087279326,
          qualifyingArmCount: 0,
          fullPhaseExecuted: false,
          remainingOperationAllocationPerformed: false,
          productionIntegrationPerformed: false,
          selectedFivePackedPayloadBytes: 19_500_544,
          fullC300PackedPayloadBytesIfFullPhaseRuns: 61_017_600,
          arms: {
            "8x64": { speedup: 0.9779577975624927, qualified: false },
            "16x32": { speedup: 0.9815300299687449, qualified: false },
            "8x32": { speedup: 1.007712087279326, qualified: false },
          },
        },
        delta: {
          qualificationThreshold: 1.15,
          stopRuleFired: true,
          selectedWinner: null,
          fullSequenceSpeedup: null,
          persistentPackedWeightBytesDeltaVersusControl: 0,
          productionIntegrationPerformed: false,
        },
      },
      correctness: {
        passed: true,
        listeningRequired: false,
        primitive: {
          probeCount: 15,
          executionsPerProbe: 2,
          comparedU16WordCount: 7_176_192,
          comparedU32WordCount: 3_766_272,
          mismatchCount: 0,
          deterministicRerunHashes: true,
        },
        repack: {
          operationCount: 5,
          uniqueU16WordCount: 9_750_272,
          comparedU16WordCount: 19_500_544,
          mismatchCount: 0,
        },
        fullPhase: { executed: false },
        cleanup: {
          createdBufferCount: 60,
          destroyedBufferCount: 60,
          liveBufferCount: 0,
          liveBytes: 0,
          deviceDestroyed: true,
        },
      },
      evidence: { conclusion: "negative" },
      disposition: { state: "abandoned" },
    });

    expect(committed.artifacts).toEqual([{
      location:
        "optimization/artifacts/OPT-0016/raw/packed-kio-microtile-ab.json",
      sha256:
        "3bfbe588d5aa6595b3f49caff670cb62293157157e4a34d0fb12349265266222",
    }]);
    expect(`${stringifyOptimizationResult(committed)}\n`).toBe(text);
  });

  it("recomputes the persisted screen and its fail-closed early stop", async () => {
    const bytes = await readFile(new URL(
      "../optimization/artifacts/OPT-0016/raw/packed-kio-microtile-ab.json",
      import.meta.url,
    ));
    expect(bytes.byteLength).toBe(31_145);
    expect(createHash("sha256").update(bytes).digest("hex")).toBe(
      "3bfbe588d5aa6595b3f49caff670cb62293157157e4a34d0fb12349265266222",
    );
    const artifact = JSON.parse(bytes.toString("utf8")) as Opt0016Artifact;
    const probes = artifact.correctness.cases.flatMap(({ probes }) => probes);
    expect(probes).toHaveLength(15);
    expect(probes.every((probe) =>
      new Set(ARMS.map((arm) => probe.armSha256[arm])).size === 1
    )).toBe(true);
    expect(artifact.correctness.mismatchCount).toBe(0);

    const compared = (storage: ArtifactCase["outputStorage"]): number =>
      artifact.correctness.cases.filter((entry) =>
        entry.outputStorage === storage
      ).flatMap(({ probes }) => probes).reduce(
        (sum, probe) => sum + probe.count * CANDIDATES.length * 2,
        0,
      );
    expect(compared("float16")).toBe(7_176_192);
    expect(compared("float32")).toBe(3_766_272);

    const repacks = artifact.repack.selectedFiveCorrectness;
    expect(repacks.cases.reduce(
      (sum, entry) => sum + entry.uniqueU16WordCount,
      0,
    )).toBe(repacks.uniqueU16WordCount);
    expect(repacks.cases.reduce(
      (sum, entry) => sum + entry.comparedU16WordCount,
      0,
    )).toBe(repacks.comparedU16WordCount);
    expect(repacks.cases.reduce(
      (sum, entry) => sum + entry.repackWorkgroups,
      0,
    )).toBe(repacks.repackWorkgroupCount);
    expect(repacks).toMatchObject({
      uniqueU16WordCount: 9_750_272,
      comparedU16WordCount: 19_500_544,
      repackWorkgroupCount: 19_044,
      mismatchCount: 0,
    });
    expect(Object.fromEntries(repacks.cases.map(({ label, sha256 }) =>
      [label, sha256]
    ))).toEqual({
      "block-0-res-1-conv1":
        "a5b263af7413fdcddb14d0cf4468360074e614f8cab1be22b4ce39e12633059b",
      "block-1-res-2-conv1":
        "ae8c34cb62c1b3c6a512328640377b7980971d5b8a508bc7bbb7403c6ed11564",
      "block-2-res-1-conv1":
        "baa781e62b263e89486a05b566635652db893da0064e6d9e02f9d01b97705c19",
      "block-4-res-3-conv1":
        "fd9eb42c7e1a36829ea952af49de1c20e2755354e9263bab53a04233aa2c8f3f",
      conv2:
        "956d3041fcc5194c609fc300b520e088cc568fe1107fc7ba742d2efc48ec4de3",
    });

    const weighted: Record<Arm, number> = {
      packed16x64: 0,
      "8x64": 0,
      "16x32": 0,
      "8x32": 0,
    };
    let rangeWeight = 0;
    for (const stratum of artifact.representativeTiming.strata) {
      rangeWeight += stratum.weight;
      for (const arm of ARMS) {
        const median = [...stratum.samples[arm]].sort((a, b) => a - b)[1]!;
        expect(median).toBe(stratum.medians[arm]);
        weighted[arm] += median * stratum.weight;
      }
    }
    expect(rangeWeight).toBe(2_402);
    expect(weighted.packed16x64).toBe(
      artifact.representativeTiming.packed16x64WeightedMilliseconds,
    );
    for (const candidate of CANDIDATES) {
      expect(weighted[candidate]).toBe(
        artifact.representativeTiming.candidateWeightedMilliseconds[candidate],
      );
      expect(weighted.packed16x64 / weighted[candidate]).toBe(
        artifact.representativeTiming.speedups[candidate],
      );
      expect(artifact.representativeTiming.speedups[candidate]).toBeLessThan(
        1.15,
      );
    }

    expect(artifact.selection).toEqual({
      status: "negative-stop-no-representative-qualifier",
      selectedWinner: null,
      qualifyingArms: [],
      threshold: 1.15,
    });
    expect(artifact.remainingOperationAllocationPerformed).toBe(false);
    expect(artifact.fullSequence).toBeNull();
    expect(artifact.protocol.thermal.durationMilliseconds).toBe(
      artifact.protocol.thermal.completedAtEpochMilliseconds -
        artifact.protocol.thermal.startedAtEpochMilliseconds,
    );
    expect(artifact.protocol.thermal).toMatchObject({
      durationMilliseconds: 30_130,
      observationCount: 31,
      maximumPollGapMilliseconds: 1_005,
      nonNominalObservationCount: 0,
      launchDelayMilliseconds: 76,
    });
    expect(artifact.protocol.unchangedThermalRetryPerformed).toBe(false);
    expect(artifact.cleanup).toMatchObject({
      createdBufferCount: 60,
      destroyedBufferCount: 60,
      liveBufferCount: 0,
      liveBytes: 0,
      idempotent: true,
      deviceDestroyed: true,
    });
  });

  it("closes exact-order microtiles without a product claim", async () => {
    const [record, ledger] = await Promise.all([
      readFile(new URL(
        "../optimization/experiments/OPT-0016-vae-k7-accumulator-microtiles.md",
        import.meta.url,
      ), "utf8"),
      readFile(new URL("../optimization/LEDGER.md", import.meta.url), "utf8"),
    ]);

    const row = ledger.split("\n").find((line) =>
      line.startsWith("| OPT-0016 |"),
    );
    expect(row).toContain("| negative | abandoned |");
    expect(row).toContain("[result](results/OPT-0016/result.json)");
    expect(row).toContain("none reached 1.15x");
    expect(row).toContain("`reordered-rounding`");
    expect(row).toContain("not integrated");

    expect(record).toContain("- Evidence: `negative`");
    expect(record).toContain("- Disposition: `abandoned`");
    expect(record).toContain("7,176,192` raw-U16");
    expect(record).toContain("3,766,272` raw-U32");
    expect(record).toContain("No candidate reached the required `1.15x`");
    expect(record).toContain("remaining 12 operation");
    expect(record).toContain("new experiment ID");
    expect(record).toContain("`reordered-rounding` risk");
    expect(record).toContain("No full-C300 winner comparison");
  });
});
