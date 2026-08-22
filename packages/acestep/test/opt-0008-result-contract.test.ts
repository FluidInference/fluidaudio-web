import { readFile } from "node:fs/promises";
import { describe, expect, it } from "vitest";

import {
  parseOptimizationResultJson,
  stringifyOptimizationResult,
} from "../benchmark/result.js";

describe("OPT-0008 optimization result", () => {
  it("commits canonical exact positive measurement-only schema-v2 evidence", async () => {
    const text = await readFile(new URL(
      "../optimization/results/OPT-0008/result.json",
      import.meta.url,
    ), "utf8");
    const committed = parseOptimizationResultJson(text);

    expect(committed).toMatchObject({
      experimentId: "OPT-0008",
      riskClass: "exact",
      baselineCommit: "9dbd6e9cb85da211aa9e8224edfc08a2eef3f706",
      candidateCommit: "511a6696e5229894cf4db70554e6bfb4a6d11486",
      identity: {
        benchmarkHarnessCommit:
          "511a6696e5229894cf4db70554e6bfb4a6d11486",
        modelManifestSha256:
          "18f36c6420976475af65ecd833ca56c6119706322ce54120389d4915d8e80db6",
        browserVersion: "151.0.7922.138",
        osBuild: "macOS 26.5.2 build 25F84",
        machineModel: "Mac15,12",
        gpuCores: 10,
        memoryBytes: 17_179_869_184,
        webgpuAdapter: {
          info: { subgroupMinSize: 32, subgroupMaxSize: 32 },
        },
      },
      protocol: { samples: 1, thermalGateSeconds: 30 },
      metrics: {
        baseline: { decodeWallMilliseconds: 11_560.100000023842 },
        candidate: {
          scope: {
            latentFrames: 256,
            quantaPerCommandBuffer: 8,
            topologyChanged: false,
            productionIntegrationPerformed: false,
            operationBoundaryFlushDiagnosticExecuted: false,
          },
          timedWindow: {
            wallMilliseconds: 11_427.100000023842,
            decoderSubmitThroughDrainMilliseconds: 10_766.599999904633,
            decoderFencedIdleMilliseconds: 639.1000002622604,
            readbackWallMilliseconds: 2.399999976158142,
            wallReconciliationDeltaMilliseconds: 0,
          },
          topology: {
            operationCount: 88,
            logicalQuantumCount: 3942,
            primitiveDispatchCount: 3988,
            pureBatchCount: 420,
            mixedBatchCount: 73,
            decoderCommandBufferCount: 493,
            decoderSubmissionCount: 493,
            decoderQueueDrainCount: 493,
            readbackCommandBufferCount: 1,
            readbackSubmissionCount: 1,
            readbackQueueDrainCount: 1,
            totalCommandBufferCount: 494,
            totalSubmissionCount: 494,
            totalQueueDrainCount: 494,
            requestedIdleMilliseconds: 493,
            completedIdleCount: 493,
            allCountsReconciled: true,
          },
          pureFamilyRanking: [
            { rank: 1, family: "k7-conv1d", submitThroughDrainMilliseconds: 6558.89999973774 },
            { rank: 2, family: "conv-transpose1d", submitThroughDrainMilliseconds: 1924.1999998092651 },
            { rank: 3, family: "k1-conv1d", submitThroughDrainMilliseconds: 1021.3000000119209 },
            { rank: 4, family: "snake", submitThroughDrainMilliseconds: 121.60000014305115 },
            { rank: 5, family: "add", submitThroughDrainMilliseconds: 55.80000001192093 },
          ],
          pureKernelRanking: [
            { rank: 1, selectedKernel: "channel-chunked-conv1d", submitThroughDrainMilliseconds: 6028 },
            { rank: 2, selectedKernel: "portable-conv-transpose1d", submitThroughDrainMilliseconds: 1924.1999998092651 },
            { rank: 3, selectedKernel: "portable-conv1d", submitThroughDrainMilliseconds: 1021.3000000119209 },
            { rank: 4, selectedKernel: "tiled-conv1d", submitThroughDrainMilliseconds: 530.8999997377396 },
            { rank: 5, selectedKernel: "portable-snake", submitThroughDrainMilliseconds: 121.60000014305115 },
            { rank: 6, selectedKernel: "portable-add", submitThroughDrainMilliseconds: 55.80000001192093 },
          ],
          mixed: {
            batchCount: 73,
            quantumCount: 582,
            dispatchCount: 588,
            submitThroughDrainMilliseconds: 1084.8000001907349,
            wallMilliseconds: 1182.7999997138977,
          },
          output: {
            elementCount: 983040,
            byteLength: 3932160,
            finiteCount: 983040,
            nonzeroCount: 983040,
            sentinelBitCount: 0,
            minimum: -0.8777482509613037,
            maximum: 0.7681955099105835,
            warmupSha256:
              "30a08c1ec1209ecaa73284e6af98775786b8ad4bd5440bbde32c6c8d6ab482e4",
            timedSha256:
              "30a08c1ec1209ecaa73284e6af98775786b8ad4bd5440bbde32c6c8d6ab482e4",
            comparedU32WordCount: 983040,
            bitMismatchCount: 0,
            bitExact: true,
            completeWrite: true,
          },
          externalThermalCoverage: {
            startedAtEpochMilliseconds: 1786674607902.7751,
            completedAtEpochMilliseconds: 1786674695907.943,
            observationCount: 89,
            nonNominalObservationCount: 0,
            durationMilliseconds: 88004.23787499312,
            maximumPollGapMilliseconds: 1006.4439999987371,
            spansTimedInterval: true,
            spansCleanupCompletion: true,
          },
          cancellation: {
            rejectionName: "AbortError",
            completedDecoderQuanta: 8,
            encodedCommandBufferCount: 1,
            submissionCount: 1,
            queueDrainCount: 1,
            requestedIdleMilliseconds: 1,
            completedIdleCount: 1,
            firstBatchFullyDrained: true,
            laterBatchEncodingPrevented: true,
            laterBatchSubmissionPrevented: true,
          },
          cleanup: {
            startedAtEpochMilliseconds: 1786674688471,
            completedAtEpochMilliseconds: 1786674688552,
            createdBufferCount: 15,
            uniqueDestroyedBufferCount: 15,
            totalDestroyCallCount: 15,
            liveTrackedBufferCount: 0,
            secondDestroyResolved: true,
            postDestroyDecodeRejected: true,
            postDestroyRejectionName: "InvalidStateError",
            runtimeEventCount: 0,
            deviceDestroyedAfterEventCheck: true,
          },
          responsiveness: {
            workerTimerTickCount: 12721,
            workerMaximumTimerGapMilliseconds: 405.39999997615814,
            mainAnimationFrameCount: 22038,
            mainTimerTickCount: 12991,
            mainMaximumAnimationFrameGapMilliseconds: 12.400000000008731,
            mainMaximumTimerGapMilliseconds: 11.699999988079071,
          },
        },
        delta: {
          rankingStable: true,
          boundaryFlushDecision: "not-run-decision-stable",
          productionIntegrationPerformed: false,
        },
      },
      correctness: { passed: true, listeningRequired: false },
      evidence: { conclusion: "positive" },
      disposition: { state: "benchmark-only" },
    });

    expect(committed.artifacts).toEqual([
      {
        location: "optimization/artifacts/OPT-0008/raw/production-window.json",
        sha256:
          "2696530395f43b5440c1131bf2c231881d34b068a41966419873960212bf9b8e",
      },
      {
        location:
          "optimization/artifacts/OPT-0008/raw/production-window-thermal.jsonl",
        sha256:
          "308c6a94b954fb51252d4e4afeb163a9573873adfd1bda5a3f20c973a8a9be79",
      },
    ]);
    expect(`${stringifyOptimizationResult(committed)}\n`).toBe(text);
  });

  it("keeps the historical record and ledger row closed", async () => {
    const [record, ledger] = await Promise.all([
      readFile(new URL(
        "../optimization/experiments/OPT-0008-package-native-vae-window-profiler.md",
        import.meta.url,
      ), "utf8"),
      readFile(new URL("../optimization/LEDGER.md", import.meta.url), "utf8"),
    ]);

    expect(record).toContain("- Evidence: `positive`");
    expect(record).toContain("- Disposition: `benchmark-only`");
    expect(record).toContain("fixed-32-subgroup implicit-im2col FP32");
    expect(record).toContain("No operation-boundary-flush diagnostic was run");
    expect(record).not.toContain("Result JSON: pending");

    const ledgerRow = ledger.split("\n").find((line) =>
      line.startsWith("| OPT-0008 |"),
    );
    expect(ledgerRow).toContain("| positive | benchmark-only |");
    expect(ledgerRow).toContain("[result](results/OPT-0008/result.json)");
  });
});
