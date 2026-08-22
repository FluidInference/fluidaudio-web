import { readFile } from "node:fs/promises";
import { describe, expect, it } from "vitest";

import {
  parseOptimizationResultJson,
  stringifyOptimizationResult,
} from "../benchmark/result.js";

describe("OPT-0006 optimization result", () => {
  it("commits canonical positive integrated schema-v2 evidence", async () => {
    const text = await readFile(new URL(
      "../optimization/results/OPT-0006/result.json",
      import.meta.url,
    ), "utf8");
    const committed = parseOptimizationResultJson(text);
    expect(committed).toMatchObject({
      experimentId: "OPT-0006",
      riskClass: "exact",
      baselineCommit: "86933d573b4967ecfcfb9c7521e65dd40a2bd03d",
      candidateCommit: "66f69709f7e30a9309e2315de0ea2ad7bae71604",
      evidence: { conclusion: "positive" },
      disposition: { state: "integrated" },
      identity: {
        benchmarkHarnessCommit:
          "4813e3d6b5a548154894ca334ce0f95c0cc16360",
        productionIntegrationCommit:
          "66f69709f7e30a9309e2315de0ea2ad7bae71604",
      },
      protocol: { samples: 4 },
      metrics: {
        baseline: { medianWallMilliseconds: 110.94999998807907 },
        candidate: {
          selectedBatchSize: 8,
          productionRangeCorrectness: {
            comparedElementsPerCandidate: 524_288,
            bitMismatchCountByBatchSize: {
              "1": 0,
              "2": 0,
              "4": 0,
              "8": 0,
              "16": 0,
            },
          },
          batchSizeMeasurements: {
            "8": {
              medianWallMilliseconds: 70,
              maximumObservedBatchDrainMilliseconds: 39.30000001192093,
            },
            "16": {
              medianWallMilliseconds: 68.19999998807907,
              maximumObservedBatchDrainMilliseconds: 69.20000004768372,
            },
          },
          fullProductionWindowExecuted: false,
          fullSongExecuted: false,
          integratedDecoderCorrectness: {
            actualChromePerformed: true,
            bitMismatchCount: 0,
            productionBatchBitMismatchCount: 0,
            finalOutputElements: 12,
            logicalQuantumCount: 109,
            physicalDispatchCount: 115,
            optimizedBatch1: {
              commandBufferCount: 110,
              queueDrainCount: 110,
              idleCount: 109,
              progressEventCount: 110,
            },
            optimizedBatch8: {
              commandBufferCount: 15,
              queueDrainCount: 15,
              idleCount: 14,
              progressEventCount: 110,
            },
            selectorQuantumCounts: {
              tiled: 7,
              channelChunked: 15,
              portable: 18,
            },
          },
          productionSourceIntegrationPerformed: true,
        },
        delta: {
          selectedBatchSize: 8,
          batch8MedianWallSpeedup: 1.584999999829701,
          batch16MedianWallReductionVersusBatch8: 0.025714285884584687,
          productionSourceIntegrationPerformed: true,
        },
      },
      correctness: { passed: true, listeningRequired: false },
    });
    expect(committed.artifacts).toEqual([
      {
        location: "optimization/artifacts/OPT-0006/raw/correctness.json",
        sha256:
          "37f93c720bf94bb9e885d384782c76af7d484b0425fea9d3bc2b40ea5de27cae",
      },
      {
        location:
          "optimization/artifacts/OPT-0006/raw/production-ranges.json",
        sha256:
          "ffd795166e5fe038d60deb0c482e791a8e7552809237f5f1bb392f1ede492cf4",
      },
      {
        location:
          "optimization/artifacts/OPT-0006/raw/production-ranges-thermal.jsonl",
        sha256:
          "344b3374bfe18869354a73af6731d3011301dd1b68e09a25b70efdd19a6b1a24",
      },
      {
        location:
          "optimization/artifacts/OPT-0006/raw/integrated-decoder-correctness.json",
        sha256:
          "8d04e3b84f4740d50807d012fb7b7944e1907f300ae596f1ce955560cbba045b",
      },
    ]);
    expect(`${stringifyOptimizationResult(committed)}\n`).toBe(text);
  });
});
