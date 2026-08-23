import { readFile } from "node:fs/promises";
import { describe, expect, it } from "vitest";

import {
  parseOptimizationResultJson,
  stringifyOptimizationResult,
} from "../benchmark/result.js";

describe("OPT-0013 optimization result", () => {
  it("commits canonical positive benchmark-only M2250 evidence", async () => {
    const text = await readFile(new URL(
      "../optimization/results/OPT-0013/result.json",
      import.meta.url,
    ), "utf8");
    const committed = parseOptimizationResultJson(text);

    expect(committed).toMatchObject({
      experimentId: "OPT-0013",
      riskClass: "reordered-rounding",
      baselineCommit: "a88e1b41c7b127d20c3fc4dbdee63acb77612a8c",
      candidateCommit: "a88e1b41c7b127d20c3fc4dbdee63acb77612a8c",
      identity: {
        benchmarkHarnessCommit:
          "5b9cc1852e5178c8bbb907dbe4c90eae6004d6b8",
        attentionSourceSha256:
          "5f64e5148ee60f26023faeb99ac72b46354086f3db48f7778800a9061d2b9ed3",
        portableWgslSha256:
          "6052f3e86f03ad3f7f3dc80384c63fa3750c0a038b68af0689914c5087929121",
        query8WgslSha256:
          "176cd0988c11944fb47cce813e4b6338d867507a00cf416845920c642aec71a4",
        productionPackageLoaded: false,
        productionRuntimeSelected: false,
        webgpuAdapter: {
          info: { subgroupMinSize: 32, subgroupMaxSize: 32 },
          limits: { maxComputeInvocationsPerWorkgroup: 256 },
        },
      },
      protocol: {
        samples: 1,
        pairedOrder: ["A", "B"],
        thermalGateSeconds: 30,
        thermal: {
          durationMilliseconds: 30_010.05517578125,
          observationCount: 31,
          maximumPollGapMilliseconds: 1_010.141357421875,
          launchDelayMilliseconds: 71.43701171875,
          nonNominalObservationCount: 0,
        },
        unchangedThermalRetryPerformed: false,
      },
      metrics: {
        baseline: {
          computeWallMilliseconds: 1_108.7999999523163,
          readbackWallMilliseconds: 8.5,
          totalWallMilliseconds: 1_117.2999999523163,
          outputSha256:
            "df90123ababd0182d7c7b4f6ec1604e4673ab7530cbc61eba41e14c5d0985d39",
        },
        candidate: {
          computeWallMilliseconds: 136.19999992847443,
          readbackWallMilliseconds: 4,
          totalWallMilliseconds: 140.19999992847443,
          outputSha256:
            "b252684edb92fcc50beec55d300741e48fe103f43bbb65b03c2f6b958e03afa2",
        },
        delta: {
          computeWallSpeedup: 8.140969166920733,
          totalWallSpeedup: 7.969329532969523,
          keyValueLoadReduction: 7.992895204262878,
          productionIntegrationPerformed: false,
        },
      },
      correctness: {
        passed: true,
        listeningRequired: true,
        listeningDecision: null,
        comparedElementCount: 4_608_000,
        comparedEveryOutputF32: true,
        maximumAbsoluteError: 2.6542693376541138e-8,
        meanAbsoluteError: 1.560317666035174e-9,
        nrmse: 3.3672934508159816e-7,
        portableNonFiniteCount: 0,
        query8NonFiniteCount: 0,
      },
      evidence: { conclusion: "positive" },
      disposition: { state: "benchmark-only" },
    });

    expect(committed.artifacts).toEqual([{
      location:
        "optimization/artifacts/OPT-0013/raw/dit-full-attention-query8-ab.json",
      sha256:
        "28a5a368f8b7fc8bf5771e415d8bd90db2ae7adba5961283a2b6010c459fbe28",
    }]);
    expect(`${stringifyOptimizationResult(committed)}\n`).toBe(text);
  });

  it("closes the ledger without claiming production or listening", async () => {
    const [record, ledger] = await Promise.all([
      readFile(new URL(
        "../optimization/experiments/OPT-0013-dit-full-attention-query8.md",
        import.meta.url,
      ), "utf8"),
      readFile(new URL("../optimization/LEDGER.md", import.meta.url), "utf8"),
    ]);

    const row = ledger.split("\n").find((line) =>
      line.startsWith("| OPT-0013 |"),
    );
    expect(row).toContain("| positive | benchmark-only |");
    expect(row).toContain("[result](results/OPT-0013/result.json)");
    expect(row).toContain("no layer, trajectory, listening, or product-speed claim");

    expect(record).toContain("- Evidence: `positive`");
    expect(record).toContain("- Disposition: `benchmark-only`");
    expect(record).toMatch(/immediate\s+production-integration candidate/);
    expect(record).toContain("No layer, denoise trajectory, final latent, listening");
  });
});
