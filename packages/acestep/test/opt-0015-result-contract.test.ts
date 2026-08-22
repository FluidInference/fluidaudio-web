import { readFile } from "node:fs/promises";
import { describe, expect, it } from "vitest";

import {
  parseOptimizationResultJson,
  stringifyOptimizationResult,
} from "../benchmark/result.js";

describe("OPT-0015 optimization result", () => {
  it("commits canonical exact integrated ConvTranspose evidence", async () => {
    const text = await readFile(new URL(
      "../optimization/results/OPT-0015/result.json",
      import.meta.url,
    ), "utf8");
    const committed = parseOptimizationResultJson(text);

    expect(committed).toMatchObject({
      experimentId: "OPT-0015",
      riskClass: "exact",
      baselineCommit: "7d4916da0cd480fe03cd5712048cb3f3f4c06310",
      candidateCommit: "36608b857827b2b1d31ac91bf5cca9639fb0b9ed",
      identity: {
        benchmarkHarnessCommit:
          "65603ade17b9f3b9ca92cc0c29be83fe51a6e885",
        coreCommit: "075ecc0b34b7541cffc0a83412c17ee31bbadab6",
        runtimeProfile:
          "opt-0015-mixed-fp16-fixed32-k7-congruent-transpose-v1",
        productionIntegrationPerformed: true,
      },
      protocol: {
        samples: 1,
        thermalGateSeconds: 30,
        integratedRunThermal: {
          durationMilliseconds: 30_125,
          observationCount: 31,
          maximumPollGapMilliseconds: 1_015,
          observedLevels: [0],
          launchDelayMilliseconds: 263,
          unchangedThermalRetryPerformed: false,
        },
      },
      metrics: {
        baseline: {
          familyProfile: {
            decoderSubmitThroughDrainMilliseconds: 14_118.300000786781,
            homogeneousConvTransposeSubmitThroughDrainMilliseconds:
              8_016.400000452995,
          },
          production12Second: {
            totalMilliseconds: 33_168.1,
            vaeMilliseconds: 15_962.4,
          },
        },
        candidate: {
          familyProfile: {
            decoderSubmitThroughDrainMilliseconds: 7_265.799999117851,
            fixed32K7Milliseconds: 3_019.800000190735,
            homogeneousCongruentTransposeMilliseconds: 2_001.9999997615814,
          },
          primitiveGate: {
            comparedU16Count: 8_404_992,
            mismatchCount: 0,
            projectedSpeedup: 3.643482691652564,
          },
          production12Second: {
            totalMilliseconds: 23_018.200000047684,
            vaeMilliseconds: 8_054.5,
            wavExactPreCandidateFp16: true,
            wavSha256:
              "409b7157ac428910fae17776b1abbd9b42db7509984bcc0aac41871f95152ec2",
          },
          productionIntegrationPerformed: true,
        },
        delta: {
          homogeneousTransposeSpeedup: 4.004195804898936,
          decoderProfileSpeedup: 1.9431170693524318,
          observed180SecondProductSpeedup: null,
          under60SecondProjection: null,
        },
      },
      correctness: {
        passed: true,
        listeningRequired: false,
        primitive: { mismatchCount: 0 },
        production: { exactPreCandidateFp16Wav: true },
      },
      evidence: { conclusion: "positive" },
      disposition: { state: "integrated" },
    });

    expect(committed.artifacts).toEqual([{
      location:
        "optimization/artifacts/OPT-0015/raw/conv-transpose-congruent-ab.json",
      sha256:
        "7dcecd275c93d44a503924eef0ddb4b5a44d542e9ba4bb52d179ee5a6ff5cd61",
    }]);
    expect(`${stringifyOptimizationResult(committed)}\n`).toBe(text);
  });

  it("closes the ledger without a three-minute or target claim", async () => {
    const [record, ledger] = await Promise.all([
      readFile(new URL(
        "../optimization/experiments/OPT-0015-vae-conv-transpose-congruent.md",
        import.meta.url,
      ), "utf8"),
      readFile(new URL("../optimization/LEDGER.md", import.meta.url), "utf8"),
    ]);

    const row = ledger.split("\n").find((line) =>
      line.startsWith("| OPT-0015 |"),
    );
    expect(row).toContain("| positive | integrated |");
    expect(row).toContain("[result](results/OPT-0015/result.json)");
    expect(row).toContain("No 180-second or under-60-second claim");

    expect(record).toContain("- Evidence: `positive`");
    expect(record).toContain("- Disposition: `integrated`");
    expect(record).toContain("all `8,404,992` selected");
    expect(record).toContain("No 180-second generation, under-60-second projection");
  });
});
