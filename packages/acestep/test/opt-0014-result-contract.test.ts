import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { describe, expect, it } from "vitest";

import {
  parseOptimizationResultJson,
  stringifyOptimizationResult,
} from "../benchmark/result.js";

describe("OPT-0014 optimization result", () => {
  it("commits canonical exact but negative packed-KIO evidence", async () => {
    const text = await readFile(new URL(
      "../optimization/results/OPT-0014/result.json",
      import.meta.url,
    ), "utf8");
    const committed = parseOptimizationResultJson(text);

    expect(committed).toMatchObject({
      experimentId: "OPT-0014",
      riskClass: "exact",
      baselineCommit: "36608b857827b2b1d31ac91bf5cca9639fb0b9ed",
      candidateCommit: "12e128ab323c0024ed683313b4d06c07041213e7",
      identity: {
        benchmarkHarnessCommit:
          "3904d212148cf2ecf93f317f8dcce3d59ef232a8",
        fixed32CoreSourceSha256:
          "7d218516d6b2c8d6e3332a53101be5fdeae1142096c442433915bfa58941ce32",
        packedKioCoreSourceSha256:
          "802cb0ad1d2c57c0cc51cbd4a7c88632e00d543b526f2ed0b94e9fc393a3d8d8",
        productionPackageLoaded: false,
        productionRuntimeSelected: false,
        productionIntegrationPerformed: false,
      },
      protocol: {
        samples: 2,
        pairedOrder: ["A", "B", "B", "A"],
        thermalGateSeconds: 30,
        thermal: {
          durationMilliseconds: 30_118,
          observationCount: 31,
          maximumPollGapMilliseconds: 1_005,
          nonNominalObservationCount: 0,
          launchDelayMilliseconds: 76,
        },
        unchangedThermalRetryPerformed: false,
      },
      metrics: {
        baseline: {
          exactC300OperationCount: 17,
          exactC300RangeCount: 2_404,
          weightedProjectedMilliseconds: 9_506.499997437,
        },
        candidate: {
          timingStratumCount: 50,
          weightedProjectedMilliseconds: 8_562.849999070168,
          persistentPackedWeightBytes: 61_017_600,
          repack: {
            includedInConvProjection: false,
            totalMilliseconds: 4.5,
          },
          productionIntegrationPerformed: false,
        },
        delta: {
          projectedSavingMilliseconds: 943.6499983668327,
          projectedSpeedup: 1.1102027944515322,
          candidateStratumWins: 33,
          candidateStratumLosses: 15,
          candidateStratumTies: 2,
          persistentPackedWeightBytesDelta: 61_017_600,
          observedIntegratedDecoderSpeedup: null,
          observed180SecondProductSpeedup: null,
          under60SecondProjection: null,
          productionIntegrationPerformed: false,
        },
      },
      correctness: {
        passed: true,
        listeningRequired: false,
        listeningDecision: null,
        primitive: {
          operationCount: 17,
          probeCount: 51,
          executionsPerProbe: 2,
          comparedWordCount: 13_854_720,
          comparedU16WordCount: 12_599_296,
          comparedU32WordCount: 1_255_424,
          mismatchCount: 0,
          qNaNPrefillCompleteWrites: true,
          guardsAndAdjacentCanariesUntouched: true,
          deterministicRerunHashes: true,
        },
        repack: {
          operationCount: 17,
          executionsPerOperation: 2,
          uniqueU16WordCount: 30_508_800,
          comparedU16WordCount: 61_017_600,
          mismatchCount: 0,
          qNaNPrefillCompleteWrites: true,
          redzonesUntouched: true,
          deterministicRerunHashes: true,
        },
        cleanup: {
          createdBufferCount: 178,
          destroyedBufferCount: 178,
          liveBufferCount: 0,
          liveBytes: 0,
          idempotent: true,
          deviceDestroyed: true,
        },
      },
      evidence: { conclusion: "negative" },
      disposition: { state: "abandoned" },
    });

    expect(committed.artifacts).toEqual([{
      location:
        "optimization/artifacts/OPT-0014/raw/packed-kio-k7-ab.json",
      sha256:
        "2445e5e3b07a3d950db8e7badcd74bff6fef687013bbc7fd56389acaedd845c3",
    }]);
    expect(`${stringifyOptimizationResult(committed)}\n`).toBe(text);
  });

  it("binds the persisted browser artifact and closes without integration", async () => {
    const [artifact, record, ledger] = await Promise.all([
      readFile(new URL(
        "../optimization/artifacts/OPT-0014/raw/packed-kio-k7-ab.json",
        import.meta.url,
      )),
      readFile(new URL(
        "../optimization/experiments/OPT-0014-vae-k7-packed-kio.md",
        import.meta.url,
      ), "utf8"),
      readFile(new URL("../optimization/LEDGER.md", import.meta.url), "utf8"),
    ]);

    expect(artifact.byteLength).toBe(110_962);
    expect(createHash("sha256").update(artifact).digest("hex")).toBe(
      "2445e5e3b07a3d950db8e7badcd74bff6fef687013bbc7fd56389acaedd845c3",
    );

    const row = ledger.split("\n").find((line) =>
      line.startsWith("| OPT-0014 |"),
    );
    expect(row).toContain("| negative | abandoned |");
    expect(row).toContain("[result](results/OPT-0014/result.json)");
    expect(row).toContain("1.1102x");
    expect(row).toContain("+61,017,600 B");
    expect(row).toContain("not integrated");

    expect(record).toContain("- Evidence: `negative`");
    expect(record).toContain("- Disposition: `abandoned`");
    expect(record).toMatch(/13,854,720\s+raw output words/);
    expect(record).toMatch(/61,017,600 U16\s+comparisons/);
    expect(record).toContain("1.1102027944515322x");
    expect(record).toContain("no integrated decoder, 12-second generation");
  });
});
