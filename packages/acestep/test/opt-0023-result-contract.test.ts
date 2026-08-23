import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";

import { describe, expect, it } from "vitest";

import {
  parseOptimizationResultJson,
  stringifyOptimizationResult,
} from "../benchmark/result.js";
import { planAceVaeChunkedDecode } from "../src/webgpu/vae-chunks.js";

const RESULT_URL = new URL(
  "../optimization/results/OPT-0023/result.json",
  import.meta.url,
);
const RECEIPT_URL = new URL(
  "../optimization/artifacts/OPT-0023/raw/vae-c4500-production-family-profile.json",
  import.meta.url,
);
const THERMAL_URL = new URL(
  "../optimization/artifacts/OPT-0023/raw/vae-c4500-production-family-profile-thermal.jsonl",
  import.meta.url,
);
const RECORD_URL = new URL(
  "../optimization/experiments/OPT-0023-vae-c4500-production-family-profile.md",
  import.meta.url,
);
const LEDGER_URL = new URL("../optimization/LEDGER.md", import.meta.url);

const RECEIPT_SHA256 =
  "6454a37243849ec9838d998abd9ca478d4b3720aa9b3edd4f497d152bda92d5c";
const THERMAL_SHA256 =
  "ecd7eded7f17dd9a5a585b8859e30a942a573840e365500a718d3d67e0a64161";

type FamilyName =
  | "add"
  | "conv-transpose1d"
  | "k1-conv1d"
  | "k7-conv1d"
  | "snake";

interface FamilyAggregate {
  readonly batchCount: number;
  readonly quantumCount: number;
  readonly submitThroughDrainMilliseconds: number;
}

interface WindowFamilyAggregate {
  readonly batchCount: number;
  readonly quantumCount: number;
  readonly submitThroughDrainMs: number;
}

interface SourceIdentity {
  readonly path: string;
  readonly byteLength: number;
  readonly sha256: string;
}

interface PerWindowProfile {
  readonly windowIndex: number;
  readonly inputFrames: number;
  readonly quantaPerCommandBuffer: number;
  readonly decoderBatchCount: number;
  readonly decoderQuantumCount: number;
  readonly decoderSubmitThroughDrainMs: number;
  readonly homogeneousBatchCount: number;
  readonly homogeneousQuantumCount: number;
  readonly homogeneousSubmitThroughDrainMs: number;
  readonly mixedBatchCount: number;
  readonly mixedQuantumCount: number;
  readonly mixedSubmitThroughDrainMs: number;
  readonly families: Readonly<Record<FamilyName, WindowFamilyAggregate>>;
}

interface ResourceSnapshot {
  readonly createdBufferBytes: number;
  readonly createdBufferCount: number;
  readonly destroyedBufferCount: number;
  readonly everyBufferDestroyedExactlyOnce: boolean;
  readonly liveBufferBytes: number;
  readonly liveBufferCount: number;
  readonly mapCount: number;
  readonly unmapCount: number;
  readonly mappedBufferCount: number;
  readonly mapOverlapDetected: boolean;
  readonly maximumLiveBufferBytes: number;
  readonly maximumLiveBufferCount: number;
  readonly maximumMappedBufferCount: number;
  readonly totalDestroyCallCount: number;
}

interface Opt0023Receipt {
  readonly schema: string;
  readonly experimentId: string;
  readonly status: string;
  readonly identity: {
    readonly browserVersion: string;
    readonly coreCommit: string;
    readonly harnessCommit: string;
    readonly osVersion: string;
    readonly osBuild: string;
    readonly machineModel: string;
    readonly gpuCoreCount: number;
    readonly memoryBytes: number;
  };
  readonly frozenAuthority: {
    readonly opt0015IntegratedCommit: string;
    readonly opt0023RegistrationCommit: string;
    readonly opt0023RegistrationRecordSha256: string;
    readonly sourceAuthority: {
      readonly aggregateSha256: string;
      readonly fileCount: number;
      readonly totalBytes: number;
      readonly files: readonly SourceIdentity[];
    };
  };
  readonly deterministicLatent: {
    readonly generator: string;
    readonly seed: string;
    readonly latentFrames: number;
    readonly channels: number;
    readonly elementCount: number;
    readonly byteLength: number;
    readonly sha256: string;
  };
  readonly package: {
    readonly manifestSha256: string;
    readonly manifestByteLength: number;
    readonly converterRevision: number;
    readonly tensorRecordCount: number;
    readonly logicalTensorCount: number;
    readonly parameterElements: number;
    readonly residentBytes: number;
    readonly files: readonly {
      readonly name: string;
      readonly byteLength: number;
      readonly sha256: string;
    }[];
  };
  readonly runtime: {
    readonly runtimeProfileId: string;
    readonly kernelSetId: string;
    readonly precisionMapSha256: string;
    readonly quantaPerCommandBuffer: number;
    readonly capabilities: {
      readonly adapterInfo: Readonly<Record<string, unknown>>;
      readonly deviceFeatures: readonly string[];
      readonly deviceLimits: Readonly<Record<string, number>>;
    };
  };
  readonly preparation: {
    readonly initializationStartedAtEpochMilliseconds: number;
    readonly initializationCompletedAtEpochMilliseconds: number;
    readonly initializationWallMilliseconds: number;
    readonly packageAcquisitionWallMilliseconds: number;
    readonly phaseUploadWallMilliseconds: number;
    readonly combinedBackendCreateWallMilliseconds: number;
    readonly exactlyOneUntimedC512Warmup: boolean;
    readonly prefillDispatchOrQueueWriteAdded: boolean;
    readonly warmupStartedAtEpochMilliseconds: number;
    readonly warmupCompletedAtEpochMilliseconds: number;
    readonly warmupWallMilliseconds: number;
    readonly warmupOutput: {
      readonly inputFrames: number;
      readonly elementCount: number;
      readonly byteLength: number;
      readonly finiteCount: number;
      readonly nonzeroCount: number;
    };
  };
  readonly protocol: {
    readonly unchangedThermalRetryPerformed: boolean;
    readonly extraCancellationRunPerformed: boolean;
    readonly fullProductRunPerformed: boolean;
    readonly externalThermalArtifactJoined: boolean;
    readonly thermalClassification: string;
    readonly thermalGate: {
      readonly traceStartObservationIndex: number;
      readonly traceStartedAtEpochMilliseconds: number;
      readonly gateStartObservationIndex: number;
      readonly gateStartedAtEpochMilliseconds: number;
      readonly gateCompletedObservationIndex: number;
      readonly gateCompletedAtEpochMilliseconds: number;
      readonly durationMilliseconds: number;
      readonly observationCount: number;
      readonly maximumPollGapMilliseconds: number;
      readonly missingObservationCount: number;
      readonly nonNominalObservationCount: number;
    };
    readonly thermal: {
      readonly traceStartObservationIndex: number;
      readonly traceStartedAtEpochMilliseconds: number;
      readonly completedObservationIndex: number;
      readonly completedAtEpochMilliseconds: number;
      readonly durationMilliseconds: number;
      readonly observationCount: number;
      readonly maximumPollGapMilliseconds: number;
      readonly missingObservationCount: number;
      readonly nonNominalObservationCount: number;
      readonly rawTraceByteLength: number;
      readonly rawTraceSha256: string;
      readonly transitions: readonly unknown[];
      readonly coversWarmupGateRunValidationAndCleanup: boolean;
    };
  };
  readonly plan: {
    readonly latentFrames: number;
    readonly chunkFrames: number;
    readonly overlapFrames: number;
    readonly windowShapes: readonly number[];
    readonly aggregateGraphQuantumCount: number;
    readonly aggregateSequenceQuantumCount: number;
    readonly decoderCommandBufferCount: number;
    readonly readbackCommandBufferCount: number;
    readonly totalCommandBufferCount: number;
    readonly requestedInternalIdleMilliseconds: number;
    readonly requestedBetweenWindowIdleMilliseconds: number;
  };
  readonly progress: {
    readonly allProgressObservedInStrictOrder: boolean;
    readonly perCommandRecordsRetained: boolean;
    readonly eventCount: number;
    readonly decoderCommandBufferCount: number;
    readonly readbackCommandBufferCount: number;
    readonly perWindowEventCounts: readonly number[];
  };
  readonly attribution: {
    readonly decoderBatchCount: number;
    readonly decoderQuantumCount: number;
    readonly decoderSubmitThroughDrainMilliseconds: number;
    readonly homogeneousBatchCount: number;
    readonly homogeneousQuantumCount: number;
    readonly homogeneousSubmitThroughDrainMilliseconds: number;
    readonly mixedBatchCount: number;
    readonly mixedQuantumCount: number;
    readonly mixedSubmitThroughDrainMilliseconds: number;
    readonly readbackIncludedInFamilyWall: boolean;
    readonly mixedWallSplitOrEstimated: boolean;
    readonly families: Readonly<Record<FamilyName, FamilyAggregate>>;
  };
  readonly perWindowProfiles: readonly PerWindowProfile[];
  readonly timing: {
    readonly fullStreamWallMilliseconds: number;
    readonly summedDecodeWallMilliseconds: number;
    readonly decoderSubmitThroughDrainMilliseconds: number;
    readonly withinDecodeNonfamilyResidualMilliseconds: number;
    readonly outsideDecodeStreamResidualMilliseconds: number;
    readonly noReadbackOrResidualProration: boolean;
    readonly windowWalls: readonly {
      readonly windowIndex: number;
      readonly inputFrames: number;
      readonly outputElements: number;
      readonly wallMilliseconds: number;
    }[];
  };
  readonly output: {
    readonly audioFrames: number;
    readonly interleavedElements: number;
    readonly byteLength: number;
    readonly finiteSamples: number;
    readonly windowsDecoded: number;
    readonly rawPeak: number;
    readonly sha256: string;
    readonly hashExcludedFromTiming: boolean;
    readonly normalizationPerformed: boolean;
    readonly wavEncodingPerformed: boolean;
  };
  readonly memory: {
    readonly boundedCpuBytes: number;
    readonly fullSongWaveformMaterialized: boolean;
    readonly rawOutputFileBacked: boolean;
    readonly atTimedStart: ResourceSnapshot;
    readonly timed: {
      readonly liveBufferBytes: number;
      readonly liveBufferCount: number;
      readonly maximumLiveBufferBytes: number;
      readonly maximumLiveBufferCount: number;
      readonly newBufferBytes: number;
      readonly newBufferCount: number;
      readonly mapCount: number;
      readonly unmapCount: number;
      readonly mappedBufferCount: number;
      readonly mapOverlapDetected: boolean;
    };
    readonly afterBackendDestroy: ResourceSnapshot;
  };
  readonly lifecycle: {
    readonly timedStartedAtEpochMilliseconds: number;
    readonly timedCompletedAtEpochMilliseconds: number;
    readonly cleanupCompletedAtEpochMilliseconds: number;
    readonly gateToTimedStartMilliseconds: number;
    readonly backendDestroyIdempotenceChecked: boolean;
    readonly tailCleanup: {
      readonly deviceDestroyed: boolean;
      readonly rawTemporaryEntryRemoved: boolean;
      readonly runtimeEventCount: number;
      readonly resources: ResourceSnapshot;
    };
  };
}

interface ThermalObservation {
  readonly index: number;
  readonly targetEpochMilliseconds: number;
  readonly observedEpochMilliseconds: number;
  readonly raw: string;
  readonly level: number;
}

interface ResultView {
  readonly identity: {
    readonly rawArtifactBytes: { readonly receipt: number; readonly thermal: number };
    readonly harnessSources: {
      readonly preExecutionContractSha256: string;
    };
    readonly sourceAuthority: {
      readonly aggregateSha256: string;
      readonly fileCount: number;
      readonly totalBytes: number;
    };
  };
  readonly protocol: {
    readonly authoritativeTimedRunCount: number;
    readonly unchangedThermalRetryPerformed: boolean;
    readonly rejectedPagePreflights: {
      readonly count: number;
      readonly reason: string;
      readonly gateCompletedAtEpochMilliseconds: number;
      readonly gateCompletedObservationIndex: number;
      readonly failureObservedAtEpochMilliseconds: number;
      readonly exactInternalLaunchDelayPersisted: boolean;
      readonly exactLaunchDelayMilliseconds: null;
      readonly setupOnly: boolean;
      readonly workerRunPostMessageSent: boolean;
      readonly timedWindowDispatchCount: number;
      readonly timingSampleCount: number;
      readonly preparedBackendDisposed: boolean;
    };
    readonly thermalGate: Readonly<Record<string, unknown>>;
    readonly thermalTrace: Readonly<Record<string, unknown>>;
  };
  readonly metrics: {
    readonly candidate: {
      readonly timing: Readonly<Record<string, number | boolean | string>>;
      readonly measuredBucketRanking: readonly {
        readonly rank: number;
        readonly bucket: string;
        readonly batchCount: number;
        readonly quantumCount: number;
        readonly submitThroughDrainMilliseconds: number;
      }[];
      readonly topology: Readonly<Record<string, number | boolean>>;
      readonly output: Readonly<Record<string, number | boolean | string>>;
      readonly perWindow: readonly {
        readonly windowIndex: number;
        readonly inputFrames: number;
        readonly decodeWallMilliseconds: number;
        readonly decoderSubmitThroughDrainMilliseconds: number;
      }[];
    };
    readonly delta: Readonly<Record<string, unknown>>;
    readonly logicalGpuPeakBytes: number;
  };
  readonly correctness: {
    readonly passed: boolean;
    readonly listeningRequired: boolean;
    readonly cleanup: Readonly<Record<string, unknown>>;
  };
}

describe("OPT-0023 optimization result", () => {
  it("commits canonical positive measurement-only schema-v2 governance", async () => {
    const [resultText, record, ledger] = await Promise.all([
      readFile(RESULT_URL, "utf8"),
      readFile(RECORD_URL, "utf8"),
      readFile(LEDGER_URL, "utf8"),
    ]);
    const parsed = parseOptimizationResultJson(resultText);
    const result = parsed as unknown as ResultView;

    expect(`${stringifyOptimizationResult(parsed)}\n`).toBe(resultText);
    expect(parsed).toMatchObject({
      schemaVersion: 2,
      experimentId: "OPT-0023",
      riskClass: "exact",
      baselineCommit: "dc08f76ce44a6a46edbd4b60c9b74e6a7b019363",
      candidateCommit: "02230725e460323de7e82ebed00177ec2103ea55",
      identity: {
        modelManifestSha256:
          "5644bcca87678b4f654b9541459355a73ef136c6bb601aa783b6f50fe2f6dba3",
        fixtureManifestSha256:
          "d4e09d07be457583ff8ed4bf420f2ae4a1e822b4f7d6e8a71c300e53123c5971",
        benchmarkHarnessCommit:
          "02230725e460323de7e82ebed00177ec2103ea55",
        gpuCores: 10,
        memoryBytes: 17_179_869_184,
      },
      protocol: { samples: 1, thermalGateSeconds: 30 },
      correctness: {
        passed: true,
        listeningRequired: false,
        listeningDecision: null,
      },
      evidence: { conclusion: "positive" },
      disposition: { state: "benchmark-only" },
    });
    expect(parsed.artifacts).toEqual([
      { location: "optimization/artifacts/OPT-0023/raw/vae-c4500-production-family-profile.json", sha256: RECEIPT_SHA256 },
      { location: "optimization/artifacts/OPT-0023/raw/vae-c4500-production-family-profile-thermal.jsonl", sha256: THERMAL_SHA256 },
    ]);

    expect(result.protocol.authoritativeTimedRunCount).toBe(1);
    expect(result.protocol.unchangedThermalRetryPerformed).toBe(false);
    expect(result.protocol.rejectedPagePreflights).toEqual({
      count: 1,
      reason: "OPT-0023 launch did not immediately follow the gate",
      gateCompletedAtEpochMilliseconds: 1_786_788_021_049,
      gateCompletedObservationIndex: 92,
      failureObservedAtEpochMilliseconds: 1_786_788_031_924,
      exactInternalLaunchDelayPersisted: false,
      exactLaunchDelayMilliseconds: null,
      setupOnly: true,
      workerRunPostMessageSent: false,
      timedWindowDispatchCount: 0,
      timingSampleCount: 0,
      preparedBackendDisposed: true,
    });

    expect(record).toContain("- Evidence: `positive`");
    expect(record).toContain("- Disposition: `benchmark-only`");
    expect(record).toContain("One earlier page preflight was rejected");
    expect(record).toContain("exact internal click-time delay was not");
    expect(record).toContain("zero timed windows, dispatches, or timing samples");
    expect(record).toContain("sole authoritative timed execution");
    expect(record).toContain("does not measure GPU occupancy, utilization");
    const row = ledger.split("\n").find((line) => line.startsWith("| OPT-0023 |"));
    expect(row).toContain("| positive | benchmark-only |");
    expect(row).toContain("[result](results/OPT-0023/result.json)");
    expect(row).toContain("before worker run/timed dispatch");
  });

  it("authenticates the raw receipt and independently reconciles every timed bucket", async () => {
    const [resultText, receiptBytes] = await Promise.all([
      readFile(RESULT_URL, "utf8"),
      readFile(RECEIPT_URL),
    ]);
    expect(receiptBytes.byteLength).toBe(34_172);
    expect(sha256(receiptBytes)).toBe(RECEIPT_SHA256);
    const result = parseOptimizationResultJson(resultText) as unknown as ResultView;
    const receipt = JSON.parse(receiptBytes.toString("utf8")) as Opt0023Receipt;

    expect(receipt).toMatchObject({
      schema: "ace-opt-0023-vae-c4500-production-family-profile-v2",
      experimentId: "OPT-0023",
      status: "passed",
      identity: {
        coreCommit: "02230725e460323de7e82ebed00177ec2103ea55",
        harnessCommit: "02230725e460323de7e82ebed00177ec2103ea55",
        gpuCoreCount: 10,
        memoryBytes: 17_179_869_184,
      },
      deterministicLatent: {
        generator: "xorshift32-13-17-5-high24-symmetric-f32-v1",
        seed: "0x00110512",
        latentFrames: 4_500,
        channels: 64,
        elementCount: 288_000,
        byteLength: 1_152_000,
        sha256:
          "d4e09d07be457583ff8ed4bf420f2ae4a1e822b4f7d6e8a71c300e53123c5971",
      },
      runtime: {
        runtimeProfileId:
          "opt-0015-mixed-fp16-fixed32-k7-congruent-transpose-v1",
        kernelSetId:
          "opt-0015-vae-fp16-fixed32-k7-congruent-transpose-kernel-set-v1",
        precisionMapSha256:
          "4bd14663b0504e3b890f781e4d01dff62c8dcdc7f87a285a578e35779cd6bc85",
        quantaPerCommandBuffer: 8,
      },
    });
    expect(receipt.package).toMatchObject({
      manifestSha256:
        "5644bcca87678b4f654b9541459355a73ef136c6bb601aa783b6f50fe2f6dba3",
      manifestByteLength: 714_687,
      converterRevision: 5,
      tensorRecordCount: 145,
      logicalTensorCount: 145,
      parameterElements: 84_395_776,
      residentBytes: 168_791_552,
    });
    expect(receipt.package.files).toHaveLength(7);
    expect(receipt.package.files.reduce((sum, file) => sum + file.byteLength, 0))
      .toBe(receipt.package.residentBytes);
    expect(receipt.package.files.every((file) => /^[0-9a-f]{64}$/.test(file.sha256)))
      .toBe(true);

    const chunks = planAceVaeChunkedDecode(4_500, {
      chunkFrames: 512,
      overlapFrames: 64,
    });
    expect(chunks.windows.map((window) => window.latentWindowFrames)).toEqual(
      receipt.plan.windowShapes,
    );
    expect(chunks.windows.reduce(
      (sum, window) => sum + window.latentWindowFrames,
      0,
    )).toBe(5_908);
    expect(chunks).toMatchObject({
      outputAudioFrames: 8_640_000,
      outputInterleavedElements: 17_280_000,
      outputFloat32Bytes: 69_120_000,
      maximumDecodedFloat32Bytes: 7_864_320,
    });

    expect(receipt.perWindowProfiles).toHaveLength(12);
    expect(receipt.timing.windowWalls).toHaveLength(12);
    for (const [index, profile] of receipt.perWindowProfiles.entries()) {
      expect(profile.windowIndex).toBe(index);
      expect(profile.inputFrames).toBe(receipt.plan.windowShapes[index]);
      expect(profile.quantaPerCommandBuffer).toBe(8);
      expect(receipt.timing.windowWalls[index]).toMatchObject({
        windowIndex: index,
        inputFrames: profile.inputFrames,
      });
      expect(profile.homogeneousBatchCount + profile.mixedBatchCount)
        .toBe(profile.decoderBatchCount);
      expect(profile.homogeneousQuantumCount + profile.mixedQuantumCount)
        .toBe(profile.decoderQuantumCount);
      expect(profile.homogeneousSubmitThroughDrainMs +
        profile.mixedSubmitThroughDrainMs)
        .toBeCloseTo(profile.decoderSubmitThroughDrainMs, 6);
      expect(sumFamily(profile.families, "batchCount"))
        .toBe(profile.homogeneousBatchCount);
      expect(sumFamily(profile.families, "quantumCount"))
        .toBe(profile.homogeneousQuantumCount);
      expect(sumWindowFamilyWall(profile.families))
        .toBeCloseTo(profile.homogeneousSubmitThroughDrainMs, 6);
    }

    const expectedFamilies: Readonly<Record<FamilyName, FamilyAggregate>> = {
      "k7-conv1d": { batchCount: 5_763, quantumCount: 46_087,
        submitThroughDrainMilliseconds: 59_993.59999811649 },
      "conv-transpose1d": { batchCount: 896, quantumCount: 7_168,
        submitThroughDrainMilliseconds: 42_401.00000369549 },
      "k1-conv1d": { batchCount: 1_019, quantumCount: 8_152,
        submitThroughDrainMilliseconds: 25_772.300002217293 },
      snake: { batchCount: 1_959, quantumCount: 15_672,
        submitThroughDrainMilliseconds: 4_301.499997854233 },
      add: { batchCount: 848, quantumCount: 6_784,
        submitThroughDrainMilliseconds: 2_041.1999996900558 },
    };
    expect(receipt.attribution.families).toEqual(expectedFamilies);
    for (const family of Object.keys(expectedFamilies) as FamilyName[]) {
      expect(sumWindow(receipt.perWindowProfiles, family, "batchCount"))
        .toBe(expectedFamilies[family].batchCount);
      expect(sumWindow(receipt.perWindowProfiles, family, "quantumCount"))
        .toBe(expectedFamilies[family].quantumCount);
      expect(sumWindowFamily(receipt.perWindowProfiles, family))
        .toBeCloseTo(expectedFamilies[family].submitThroughDrainMilliseconds, 6);
    }

    expect(receipt.attribution).toMatchObject({
      decoderBatchCount: 11_338,
      decoderQuantumCount: 90_687,
      decoderSubmitThroughDrainMilliseconds: 143_453.1000008583,
      homogeneousBatchCount: 10_485,
      homogeneousQuantumCount: 83_863,
      homogeneousSubmitThroughDrainMilliseconds: 134_509.60000157356,
      mixedBatchCount: 853,
      mixedQuantumCount: 6_824,
      mixedSubmitThroughDrainMilliseconds: 8_943.499999284744,
      readbackIncludedInFamilyWall: false,
      mixedWallSplitOrEstimated: false,
    });
    expect(sumFamily(receipt.attribution.families, "batchCount"))
      .toBe(receipt.attribution.homogeneousBatchCount);
    expect(sumFamily(receipt.attribution.families, "quantumCount"))
      .toBe(receipt.attribution.homogeneousQuantumCount);
    expect(sumAggregateFamilyWall(receipt.attribution.families))
      .toBeCloseTo(receipt.attribution.homogeneousSubmitThroughDrainMilliseconds, 6);

    expect(receipt.plan).toMatchObject({
      aggregateGraphQuantumCount: 90_675,
      aggregateSequenceQuantumCount: 90_687,
      decoderCommandBufferCount: 11_338,
      readbackCommandBufferCount: 12,
      totalCommandBufferCount: 11_350,
      requestedInternalIdleMilliseconds: 11_338,
      requestedBetweenWindowIdleMilliseconds: 11,
    });
    expect(receipt.progress).toEqual({
      allProgressObservedInStrictOrder: true,
      decoderCommandBufferCount: 11_338,
      eventCount: 11_350,
      perCommandRecordsRetained: false,
      perWindowEventCounts: [863, 983, 983, 983, 983, 983, 983, 983, 983, 983, 983, 657],
      readbackCommandBufferCount: 12,
    });
    expect(receipt.progress.perWindowEventCounts.reduce((sum, value) => sum + value, 0))
      .toBe(receipt.progress.eventCount);

    const summedDecode = receipt.timing.windowWalls.reduce(
      (sum, window) => sum + window.wallMilliseconds,
      0,
    );
    const summedDecoder = receipt.perWindowProfiles.reduce(
      (sum, profile) => sum + profile.decoderSubmitThroughDrainMs,
      0,
    );
    expect(summedDecode).toBeCloseTo(161_111.30000007153, 6);
    expect(summedDecoder).toBeCloseTo(143_453.1000008583, 6);
    expect(receipt.timing).toMatchObject({
      fullStreamWallMilliseconds: 161_392.39999997616,
      summedDecodeWallMilliseconds: 161_111.30000007153,
      decoderSubmitThroughDrainMilliseconds: 143_453.1000008583,
      withinDecodeNonfamilyResidualMilliseconds: 17_658.19999921322,
      outsideDecodeStreamResidualMilliseconds: 281.09999990463257,
      noReadbackOrResidualProration: true,
    });
    expect(summedDecode - summedDecoder)
      .toBeCloseTo(receipt.timing.withinDecodeNonfamilyResidualMilliseconds, 6);
    expect(receipt.timing.fullStreamWallMilliseconds - summedDecode)
      .toBeCloseTo(receipt.timing.outsideDecodeStreamResidualMilliseconds, 6);

    expect(result.metrics.candidate.timing).toMatchObject({
      fullStreamWallMilliseconds: receipt.timing.fullStreamWallMilliseconds,
      summedDecodeWallMilliseconds: receipt.timing.summedDecodeWallMilliseconds,
      decoderSubmitThroughDrainMilliseconds:
        receipt.timing.decoderSubmitThroughDrainMilliseconds,
      withinDecodeNonfamilyResidualMilliseconds:
        receipt.timing.withinDecodeNonfamilyResidualMilliseconds,
      outsideDecodeStreamResidualMilliseconds:
        receipt.timing.outsideDecodeStreamResidualMilliseconds,
      noReadbackIdleMapOrOpfsProration: true,
    });
    expect(result.metrics.candidate.measuredBucketRanking.map((entry) => entry.bucket))
      .toEqual(["k7-conv1d", "conv-transpose1d", "k1-conv1d", "mixed", "snake", "add"]);
    expect(result.metrics.candidate.perWindow).toEqual(
      receipt.timing.windowWalls.map((window, index) => ({
        windowIndex: index,
        inputFrames: window.inputFrames,
        decodeWallMilliseconds: window.wallMilliseconds,
        decoderSubmitThroughDrainMilliseconds:
          receipt.perWindowProfiles[index]!.decoderSubmitThroughDrainMs,
      })),
    );

    expect(receipt.output).toMatchObject({
      audioFrames: 8_640_000,
      interleavedElements: 17_280_000,
      byteLength: 69_120_000,
      finiteSamples: 17_280_000,
      windowsDecoded: 12,
      rawPeak: 0.9710559248924255,
      sha256: "fb8aae85e21a8a93b39baf738d0f2577e18134c627a05562b710341d0d590f7c",
      hashExcludedFromTiming: true,
      normalizationPerformed: false,
      wavEncodingPerformed: false,
    });
    expect(result.metrics.candidate.output).toMatchObject({
      audioFrames: receipt.output.audioFrames,
      interleavedElements: receipt.output.interleavedElements,
      byteLength: receipt.output.byteLength,
      finiteSamples: receipt.output.finiteSamples,
      rawPeak: receipt.output.rawPeak,
      sha256: receipt.output.sha256,
      fileBacked: true,
      hashExcludedFromTiming: true,
    });
    expect(result.metrics.logicalGpuPeakBytes).toBe(944_808_752);
    expect(result.correctness.passed).toBe(true);
    expect(result.correctness.listeningRequired).toBe(false);
  });

  it("independently verifies thermal cadence, the accepted gate, preparation, memory, and cleanup", async () => {
    const [resultText, receiptBytes, thermalBytes] = await Promise.all([
      readFile(RESULT_URL, "utf8"),
      readFile(RECEIPT_URL),
      readFile(THERMAL_URL),
    ]);
    expect(thermalBytes.byteLength).toBe(59_950);
    expect(sha256(thermalBytes)).toBe(THERMAL_SHA256);
    const result = parseOptimizationResultJson(resultText) as unknown as ResultView;
    const receipt = JSON.parse(receiptBytes.toString("utf8")) as Opt0023Receipt;
    const thermal = thermalBytes.toString("utf8").trim().split("\n")
      .map((line) => JSON.parse(line) as ThermalObservation);

    expect(thermal).toHaveLength(390);
    expect(thermal[0]).toEqual({
      index: 0,
      targetEpochMilliseconds: 1_786_787_929_043,
      observedEpochMilliseconds: 1_786_787_929_046,
      raw: "com.apple.system.thermalpressurelevel 0",
      level: 0,
    });
    expect(thermal.at(-1)).toEqual({
      index: 389,
      targetEpochMilliseconds: 1_786_788_318_043,
      observedEpochMilliseconds: 1_786_788_318_050,
      raw: "com.apple.system.thermalpressurelevel 0",
      level: 0,
    });
    expect(thermal.every((entry, index) =>
      entry.index === index &&
      entry.targetEpochMilliseconds === thermal[0]!.targetEpochMilliseconds + index * 1_000 &&
      entry.observedEpochMilliseconds >= entry.targetEpochMilliseconds &&
      entry.level === 0 &&
      entry.raw === "com.apple.system.thermalpressurelevel 0" &&
      (index === 0 || entry.observedEpochMilliseconds >
        thermal[index - 1]!.observedEpochMilliseconds)
    )).toBe(true);
    expect(maximumGap(thermal)).toBe(1_036);

    const gate = thermal.slice(168, 199);
    expect(gate).toHaveLength(31);
    expect(gate[0]!.observedEpochMilliseconds).toBe(1_786_788_097_053);
    expect(gate.at(-1)!.observedEpochMilliseconds).toBe(1_786_788_127_054);
    expect(gate.at(-1)!.observedEpochMilliseconds -
      gate[0]!.observedEpochMilliseconds).toBe(30_001);
    expect(maximumGap(gate)).toBe(1_005);
    expect(receipt.protocol.thermalGate).toEqual({
      command: "notifyutil -g com.apple.system.thermalpressurelevel",
      durationMilliseconds: 30_001,
      gateCompletedAtEpochMilliseconds: 1_786_788_127_054,
      gateCompletedObservationIndex: 198,
      gateStartObservationIndex: 168,
      gateStartedAtEpochMilliseconds: 1_786_788_097_053,
      maximumPollGapMilliseconds: 1_005,
      missingObservationCount: 0,
      nonNominalObservationCount: 0,
      observationCount: 31,
      pollMilliseconds: 1_000,
      source: "notifyutil-com.apple.system.thermalpressurelevel",
      traceObservationCountThroughGate: 199,
      traceStartObservationIndex: 0,
      traceStartedAtEpochMilliseconds: 1_786_787_929_046,
    });
    expect(receipt.protocol.thermal).toMatchObject({
      traceStartObservationIndex: 0,
      traceStartedAtEpochMilliseconds: thermal[0]!.observedEpochMilliseconds,
      completedObservationIndex: 389,
      completedAtEpochMilliseconds: thermal.at(-1)!.observedEpochMilliseconds,
      durationMilliseconds: 389_004,
      observationCount: 390,
      maximumPollGapMilliseconds: 1_036,
      missingObservationCount: 0,
      nonNominalObservationCount: 0,
      rawTraceByteLength: 59_950,
      rawTraceSha256: THERMAL_SHA256,
      transitions: [],
      coversWarmupGateRunValidationAndCleanup: true,
    });
    expect(receipt.protocol.thermalClassification).toBe("complete-trace-nominal");
    expect(receipt.protocol.unchangedThermalRetryPerformed).toBe(false);
    expect(receipt.protocol.extraCancellationRunPerformed).toBe(false);
    expect(receipt.protocol.fullProductRunPerformed).toBe(false);
    expect(receipt.protocol.externalThermalArtifactJoined).toBe(true);

    expect(thermal[0]!.observedEpochMilliseconds)
      .toBeLessThan(receipt.preparation.warmupStartedAtEpochMilliseconds);
    expect(receipt.preparation.warmupCompletedAtEpochMilliseconds)
      .toBeLessThan(gate[0]!.observedEpochMilliseconds);
    expect(receipt.lifecycle.gateToTimedStartMilliseconds)
      .toBeCloseTo(163.599853515625, 9);
    expect(receipt.lifecycle.timedStartedAtEpochMilliseconds -
      gate.at(-1)!.observedEpochMilliseconds)
      .toBeCloseTo(receipt.lifecycle.gateToTimedStartMilliseconds, 9);
    expect(receipt.lifecycle.timedCompletedAtEpochMilliseconds)
      .toBeLessThanOrEqual(receipt.lifecycle.cleanupCompletedAtEpochMilliseconds);
    expect(thermal.at(-1)!.observedEpochMilliseconds -
      receipt.lifecycle.cleanupCompletedAtEpochMilliseconds).toBe(29_101);

    expect(receipt.preparation).toMatchObject({
      initializationWallMilliseconds: 15_233.100000023842,
      packageAcquisitionWallMilliseconds: 711.1999999284744,
      phaseUploadWallMilliseconds: 79.69999992847443,
      combinedBackendCreateWallMilliseconds: 165.5,
      exactlyOneUntimedC512Warmup: true,
      prefillDispatchOrQueueWriteAdded: false,
      warmupWallMilliseconds: 14_212.799999952316,
      warmupOutput: {
        inputFrames: 512,
        elementCount: 1_966_080,
        byteLength: 7_864_320,
        finiteCount: 1_966_080,
        nonzeroCount: 1_966_080,
      },
    });

    expect(receipt.memory.atTimedStart).toMatchObject({
      createdBufferCount: 17,
      destroyedBufferCount: 0,
      liveBufferCount: 17,
      liveBufferBytes: 944_808_752,
      maximumLiveBufferCount: 17,
      maximumLiveBufferBytes: 944_808_752,
      mapCount: 1,
      unmapCount: 1,
      mappedBufferCount: 0,
      mapOverlapDetected: false,
    });
    expect(receipt.memory.timed).toEqual({
      liveBufferBytes: 944_808_752,
      liveBufferCount: 17,
      mapCount: 12,
      mapOverlapDetected: false,
      mappedBufferCount: 0,
      maximumLiveBufferBytes: 944_808_752,
      maximumLiveBufferCount: 17,
      maximumMappedBufferCount: 1,
      newBufferBytes: 0,
      newBufferCount: 0,
      unmapCount: 12,
    });
    expect(receipt.memory).toMatchObject({
      boundedCpuBytes: 9_016_320,
      fullSongWaveformMaterialized: false,
      rawOutputFileBacked: true,
    });
    expect(receipt.memory.afterBackendDestroy).toMatchObject({
      createdBufferCount: 17,
      destroyedBufferCount: 17,
      totalDestroyCallCount: 17,
      everyBufferDestroyedExactlyOnce: true,
      liveBufferCount: 0,
      liveBufferBytes: 0,
      mapCount: 13,
      unmapCount: 13,
      mappedBufferCount: 0,
      mapOverlapDetected: false,
    });
    expect(receipt.lifecycle).toMatchObject({
      backendDestroyIdempotenceChecked: true,
      tailCleanup: {
        deviceDestroyed: true,
        rawTemporaryEntryRemoved: true,
        runtimeEventCount: 0,
      },
    });
    expect(receipt.lifecycle.tailCleanup.resources)
      .toEqual(receipt.memory.afterBackendDestroy);
    expect(result.correctness.cleanup).toMatchObject({
      createdBufferCount: 17,
      destroyedBufferCount: 17,
      liveBufferCount: 0,
      liveBufferBytes: 0,
      mapCount: 13,
      unmapCount: 13,
      everyBufferDestroyedExactlyOnce: true,
      backendDestroyIdempotenceChecked: true,
      deviceDestroyed: true,
      rawTemporaryEntryRemoved: true,
      runtimeEventCount: 0,
    });
    expect(result.protocol.thermalGate).toMatchObject({
      gateStartObservationIndex: 168,
      gateCompletedObservationIndex: 198,
      durationMilliseconds: 30_001,
      observationCount: 31,
      maximumPollGapMilliseconds: 1_005,
      gateToTimedStartMilliseconds: 163.599853515625,
    });
    expect(result.protocol.thermalTrace).toMatchObject({
      durationMilliseconds: 389_004,
      observationCount: 390,
      maximumPollGapMilliseconds: 1_036,
      postCleanupCoverageMilliseconds: 29_101,
      classification: "complete-trace-nominal",
    });
  });

  it("recomputes every source identity and freezes target-browser limits and nonclaims", {
    timeout: 120_000,
  }, async () => {
    const [resultText, receiptBytes] = await Promise.all([
      readFile(RESULT_URL, "utf8"),
      readFile(RECEIPT_URL),
    ]);
    const result = parseOptimizationResultJson(resultText) as unknown as ResultView;
    const receipt = JSON.parse(receiptBytes.toString("utf8")) as Opt0023Receipt;
    const source = receipt.frozenAuthority.sourceAuthority;

    expect(source).toMatchObject({
      aggregateSha256:
        "6946795009d3e578a00e2c5bc355b0b2d8fb4b9607fa542eafff4cbcbbba7daf",
      fileCount: 33,
      totalBytes: 846_140,
    });
    expect(source.files).toHaveLength(source.fileCount);
    expect(source.files.map((entry) => entry.path))
      .toEqual([...source.files.map((entry) => entry.path)].sort());
    const historicalPreExecutionContract = source.files.find((entry) =>
      entry.path ===
        "test/opt-0023-vae-c4500-production-family-profile-contract.test.ts"
    );
    expect(historicalPreExecutionContract).toEqual({
      path: "test/opt-0023-vae-c4500-production-family-profile-contract.test.ts",
      byteLength: 36_783,
      sha256:
        "638e6eab48355799ee0d183c23dfff8b34f44b50a6b9e84e104c632d87f5a588",
    });
    for (const entry of source.files) {
      // This contract was part of the immutable run authority at 0223072 and
      // is intentionally authenticated by its historical receipt literal.
      // Its current HEAD changed only to hand mutable record governance to
      // this post-execution result contract.
      if (entry === historicalPreExecutionContract) continue;
      const bytes = await readFile(new URL(`../${entry.path}`, import.meta.url));
      expect(bytes.byteLength, entry.path).toBe(entry.byteLength);
      expect(sha256(bytes), entry.path).toBe(entry.sha256);
    }
    expect(source.files.reduce((sum, entry) => sum + entry.byteLength, 0))
      .toBe(source.totalBytes);
    expect(sha256(Buffer.from(JSON.stringify(source.files))))
      .toBe(source.aggregateSha256);
    expect(result.identity.sourceAuthority).toEqual({
      aggregateSha256: source.aggregateSha256,
      fileCount: source.fileCount,
      totalBytes: source.totalBytes,
    });
    expect(result.identity.harnessSources.preExecutionContractSha256).toBe(
      historicalPreExecutionContract!.sha256,
    );
    expect(result.identity.rawArtifactBytes).toEqual({
      receipt: 34_172,
      thermal: 59_950,
    });

    expect(receipt.frozenAuthority).toMatchObject({
      opt0015IntegratedCommit: "36608b857827b2b1d31ac91bf5cca9639fb0b9ed",
      opt0023RegistrationCommit: "9a3e37d48c75139f98bfb9958f35061247b56da6",
      opt0023RegistrationRecordSha256:
        "c87b472ed544ba3a0177c41ba7e66bb33cb4c9ececb88be64da0e4d2845a5ee1",
    });
    expect(receipt.identity).toMatchObject({
      browserVersion:
        "Google Chrome 151.0.7922.138; reduced user agent 151.0.0.0",
      osVersion: "macOS 26.5.2",
      osBuild: "25F84",
      machineModel: "MacBook Air Mac15,12; Apple M3",
    });
    expect(receipt.runtime.capabilities).toMatchObject({
      adapterInfo: {
        vendor: "apple",
        architecture: "metal-3",
        subgroupMinSize: 32,
        subgroupMaxSize: 32,
        isFallbackAdapter: false,
      },
      deviceFeatures: ["core-features-and-limits", "shader-f16", "subgroups"],
      deviceLimits: {
        maxBufferSize: 268_435_456,
        maxStorageBufferBindingSize: 268_435_456,
        maxComputeWorkgroupStorageSize: 16_384,
        maxComputeInvocationsPerWorkgroup: 256,
      },
    });

    expect(result.metrics.delta).toMatchObject({
      performanceSpeedupMeasured: false,
      productionIntegrationPerformed: false,
      c300RelationshipIsPlanningContextOnly: true,
      measuredMinusC300LinearPlanningEstimateMilliseconds:
        2_772.4466666428198,
      measuredToC300LinearPlanningEstimateRatio: 1.017478549251724,
      separatelyMeasuredOpt0018DitOnlyGenerationWallMilliseconds: 73_072.6,
      crossRunArithmeticVaePlusDitMilliseconds: 234_464.99999997616,
      crossRunArithmeticIsNotEndToEndObservation: true,
      under60SecondProjection: null,
      credibleAbsoluteSavingThresholdMilliseconds: 10_000,
      followupMechanismAuthorizedByThisResult: false,
      utilizationOrPureGpuTimeMeasured: false,
    });
  });
});

function sha256(bytes: Uint8Array): string {
  return createHash("sha256").update(bytes).digest("hex");
}

function maximumGap(observations: readonly ThermalObservation[]): number {
  return Math.max(...observations.slice(1).map((entry, index) =>
    entry.observedEpochMilliseconds -
    observations[index]!.observedEpochMilliseconds
  ));
}

function sumFamily<T extends FamilyAggregate | WindowFamilyAggregate>(
  families: Readonly<Record<FamilyName, T>>,
  field: "batchCount" | "quantumCount",
): number {
  return Object.values(families).reduce((sum, family) => sum + family[field], 0);
}

function sumAggregateFamilyWall(
  families: Readonly<Record<FamilyName, FamilyAggregate>>,
): number {
  return Object.values(families).reduce(
    (sum, family) => sum + family.submitThroughDrainMilliseconds,
    0,
  );
}

function sumWindowFamilyWall(
  families: Readonly<Record<FamilyName, WindowFamilyAggregate>>,
): number {
  return Object.values(families).reduce(
    (sum, family) => sum + family.submitThroughDrainMs,
    0,
  );
}

function sumWindow(
  profiles: readonly PerWindowProfile[],
  family: FamilyName,
  field: "batchCount" | "quantumCount",
): number {
  return profiles.reduce((sum, profile) => sum + profile.families[family][field], 0);
}

function sumWindowFamily(
  profiles: readonly PerWindowProfile[],
  family: FamilyName,
): number {
  return profiles.reduce(
    (sum, profile) => sum + profile.families[family].submitThroughDrainMs,
    0,
  );
}
