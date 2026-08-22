import { readFile } from "node:fs/promises";
import { describe, expect, it } from "vitest";

import {
  parseOptimizationResultJson,
  stringifyOptimizationResult,
} from "../benchmark/result.js";

describe("OPT-0007 optimization result", () => {
  it("commits canonical exact negative schema-v2 evidence for the fixed geometry", async () => {
    const text = await readFile(new URL(
      "../optimization/results/OPT-0007/result.json",
      import.meta.url,
    ), "utf8");
    const committed = parseOptimizationResultJson(text);

    expect(committed).toMatchObject({
      experimentId: "OPT-0007",
      riskClass: "exact",
      baselineCommit: "f339266fb91b2e5519fa559b3ca4126361b436ad",
      candidateCommit: "1fa164b91a03cf6838fd94812e21949a8e621664",
      identity: {
        benchmarkHarnessCommit:
          "1fa164b91a03cf6838fd94812e21949a8e621664",
        endpointHarnessCommit:
          "3c7b351b48d72e0397ba2d8260f0b52bb98f9442",
        sequenceHarnessCommit:
          "1fa164b91a03cf6838fd94812e21949a8e621664",
        browserVersion:
          "Google Chrome 151.0.7922.138; reduced user agent 151.0.0.0",
        osBuild: "macOS 26.5.2 build 25F84",
        machineModel: "MacBook Air Mac15,12; Apple M3",
        gpuCores: 10,
        memoryBytes: 17_179_869_184,
      },
      protocol: { samples: 4 },
      metrics: {
        baseline: {
          c128SequenceWallMilliseconds: {
            median: 36.60000002384186,
          },
        },
        candidate: {
          geometry: {
            workgroupSize: [16, 8, 1],
            frameTile: 16,
            outputChannelTile: 8,
            inputChannelChunk: 64,
            workgroupStorageBytes: 6_432,
          },
          manageableCorrectness: {
            caseCount: 4,
            comparedElementCount: 2_374,
            bitMismatchCount: 0,
          },
          c1024Range: {
            outputElements: 229_376,
            bitMismatchCount: 0,
            wallMilliseconds: { median: 22.149999976158142 },
          },
          c128Range: {
            outputElements: 1_048_576,
            bitMismatchCount: 0,
            wallMilliseconds: { median: 13.5 },
          },
          c128ProductionSequence: {
            outputElements: 16_777_216,
            validMultiplyAdds: 2_147_483_648,
            bitMismatchCount: 0,
            postTimingBitIdentical: true,
            wallMilliseconds: { median: 44.099999994039536 },
            scheduling: {
              rangeCount: 16,
              passes: 16,
              dispatches: 16,
              commandBuffers: 2,
              queueDrains: 2,
              realCooperativeIdles: 2,
              maximumOutstandingCommandBuffers: 1,
            },
            cancellation: {
              firstBatchPassesSubmitted: 8,
              firstBatchFullyDrained: true,
              realIdleCompleted: true,
              laterBatchEncodingPrevented: true,
              laterBatchSubmissionPrevented: true,
            },
            fullOperationExecuted: false,
          },
          externalThermalCoverage: expect.arrayContaining([
            {
              id: "sequence-c128",
              observationCount: 75,
              durationMilliseconds: 74_003.344916,
              maximumPollGapMilliseconds: 1_004.5713749714196,
              nonNominalObservationCount: 0,
              spansCompletedResult: true,
            },
          ]),
          fullVaeWindowExecuted: false,
          fullSongExecuted: false,
          productionIntegrationPerformed: false,
        },
        delta: {
          c1024Range: {
            classification: "non-positive-noisy",
            candidateRoundWins: 2,
            medianWallSpeedup: 0.8826185116462391,
          },
          c128Range: {
            classification: "positive-noisy",
            candidateRoundWins: 3,
            medianWallSpeedup: 2.129629631837209,
          },
          c128ProductionSequence: {
            classification: "decisive-negative",
            candidateRoundWins: 0,
            medianWallSpeedup: 0.8299319734419193,
          },
          productionIntegrationPerformed: false,
        },
      },
      correctness: { passed: true, listeningRequired: false },
      evidence: { conclusion: "negative" },
      disposition: { state: "abandoned" },
    });

    expect(committed.artifacts).toEqual([
      {
        location: "optimization/artifacts/OPT-0007/raw/correctness.json",
        sha256:
          "e2e184a43a2d65c633729db5a7258b58468a21ceb582d8298543c8cb37e18ee7",
      },
      {
        location: "optimization/artifacts/OPT-0007/raw/screen-c1024.json",
        sha256:
          "01de030cf4804c7ca5d3a3dde32646678e9f7ccfccafd4951c49573305091607",
      },
      {
        location:
          "optimization/artifacts/OPT-0007/raw/screen-c1024-thermal.jsonl",
        sha256:
          "0b160c320efdc36888293b8b2991be4c22586c5b7bef2a01aab3705bc85f1a94",
      },
      {
        location: "optimization/artifacts/OPT-0007/raw/screen-c128.json",
        sha256:
          "2a95879458999cfc05feebb33fb5ddc0faaf57e8ab40fe5f2d91dd4edfc1162d",
      },
      {
        location:
          "optimization/artifacts/OPT-0007/raw/screen-c128-thermal.jsonl",
        sha256:
          "cd89e344ce2b41ba27d7216f66ae8f6052e8fb2a0c089c9f04b9f236638f4eee",
      },
      {
        location: "optimization/artifacts/OPT-0007/raw/sequence-c128.json",
        sha256:
          "12a080aa990912db4164d4d6b26bcd08be910988c5ab2ec763a397f2828d0d51",
      },
      {
        location:
          "optimization/artifacts/OPT-0007/raw/sequence-c128-thermal.jsonl",
        sha256:
          "0cde766e115b3fc1786491ee2b607c928c58bd25048850e1ae3b0825ad1e72a7",
      },
    ]);
    expect(`${stringifyOptimizationResult(committed)}\n`).toBe(text);
  });
});
