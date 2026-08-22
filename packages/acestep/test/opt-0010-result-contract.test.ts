import { readFile } from "node:fs/promises";
import { describe, expect, it } from "vitest";

import {
  parseOptimizationResultJson,
  stringifyOptimizationResult,
} from "../benchmark/result.js";

const RAW_ARTIFACTS = [
  [
    "planner-token-timing.json",
    "244838e6cdad0c5faeeb263356f7c0709820875ca59265e3bfae9ff7e7646bcf",
  ],
  [
    "planner-token-timing-thermal.jsonl",
    "2cca9cccd1def85c5df4b02496db5b25cd213d72cf20f8e4e9efdc3aeb693438",
  ],
] as const;

interface PlannerCase {
  readonly id: string;
  readonly cachedTokensBeforeAppend: number;
  readonly cacheCapacity: number;
  readonly correctness: {
    readonly bitExact: boolean;
    readonly bitMismatchCount: number;
    readonly comparedU32WordCount: number;
  };
  readonly referenceSample: unknown;
  readonly timedSample: unknown;
  readonly timing: {
    readonly timedWindowMilliseconds: number;
    readonly samplingMilliseconds: number;
    readonly modelQuantaSubmitThroughDrainMilliseconds: number;
    readonly modelQuantaWallMilliseconds: number;
    readonly modelQuantaIdleMilliseconds: number;
    readonly layersSubmitThroughDrainMilliseconds: number;
    readonly layersWallMilliseconds: number;
    readonly tiedHeadSubmitThroughDrainMilliseconds: number;
    readonly readbackWallMilliseconds: number;
  };
  readonly readback: {
    readonly rows: number;
    readonly rawLogitBytes: number;
    readonly bufferBytes: number;
    readonly shardCount: number;
  };
  readonly work: {
    readonly logicalMultiplyAdds: number;
    readonly scheduledMultiplyAdds: number;
  };
}

interface PlannerCandidate {
  readonly cases: readonly PlannerCase[];
  readonly sharedTopology: Record<string, unknown>;
  readonly semanticHeadOpportunity: {
    readonly firstTokenId: number;
    readonly tokenCount: number;
    readonly fullHead: Record<string, number>;
    readonly restrictedCodeHead: {
      readonly logicalWeightTrafficBytes: number;
      readonly rawLogitBytes: number;
      readonly scheduledMultiplyAdds: number;
      readonly intersections: readonly Record<string, number>[];
    };
    readonly avoidablePerToken: Record<string, number>;
    readonly avoidableFractions: Record<string, number>;
    readonly terminalEos: Record<string, unknown>;
  };
  readonly externalThermalCoverage: Record<string, unknown>;
  readonly browserPreGate: Record<string, unknown>;
  readonly cancellation: Record<string, unknown>;
  readonly cleanup: Record<string, unknown>;
  readonly responsiveness: Record<string, unknown>;
  readonly scope: Record<string, unknown>;
}

describe("OPT-0010 optimization result", () => {
  it("commits canonical positive benchmark-only schema-v2 evidence", async () => {
    const text = await readFile(new URL(
      "../optimization/results/OPT-0010/result.json",
      import.meta.url,
    ), "utf8");
    const committed = parseOptimizationResultJson(text);
    const candidate = committed.metrics.candidate as unknown as PlannerCandidate;

    expect(committed).toMatchObject({
      experimentId: "OPT-0010",
      riskClass: "exact",
      baselineCommit: "00dfd4732aa019bbbb238ae40265fe86cb38f27b",
      candidateCommit: "81e84df955e7cb812a60d9c6decadff3791234e3",
      identity: {
        modelManifestSha256:
          "c5b547cd08aa5e6d2971b2c9c84940b8af193f2e230ce689258ca81fcd292a3b",
        fixtureManifestSha256:
          "554106761fde0a5fab8075324d34fc08cb31b885f044c173cd4ba1ab1facb678",
        benchmarkHarnessCommit:
          "81e84df955e7cb812a60d9c6decadff3791234e3",
        attributionCoreCommit:
          "00dfd4732aa019bbbb238ae40265fe86cb38f27b",
        plannerSourceRevision:
          "148d8ea0225bdab342ee1ae3a354275ccd60ca80",
        browserVersion: "151.0.7922.138",
        osBuild: "macOS 26.5.2 build 25F84",
        machineModel: "Mac15,12",
        gpuCores: 10,
        memoryBytes: 17_179_869_184,
        webgpuAdapter: {
          info: { subgroupMinSize: 32, subgroupMaxSize: 32 },
        },
      },
      protocol: {
        samples: 1,
        thermalGateSeconds: 36,
        thermalPollMilliseconds: 1000,
        oneOutstandingCommandBuffer: true,
        fullLogitBitComparison: true,
      },
      correctness: {
        passed: true,
        listeningRequired: false,
        allSixFullLogitComparisonsBitExact: true,
        totalComparedU32WordCount: 1_954_836,
        totalBitMismatchCount: 0,
        allSampleTokensWordsAndAbsoluteDrawCursorsEqual: true,
        allTopologyCountsReconciled: true,
      },
      evidence: { conclusion: "positive" },
      disposition: { state: "benchmark-only" },
    });

    expect(Object.keys(
      (committed.identity as unknown as {
        productionSourceIdentities: Record<string, string>;
      }).productionSourceIdentities,
    )).toHaveLength(19);
    expect((committed.identity as unknown as {
      productionSourceIdentities: Record<string, string>;
    }).productionSourceIdentities).toMatchObject({
      "src/runtime/scheduler.ts":
        "a6825fb677883df136f480baa0613437316d867f2ea8bf70bb6f60ca25bd5e16",
      "src/webgpu/kernels/gemm.ts":
        "5b46dd4af23a0b89f44103143efae8ed34e9296d58cb616bf48c61cb3190b909",
      "src/webgpu/planner-executor.ts":
        "91e50770b393206acb126c3fa45d7352c733d65906f181c8e8b2423ecf0bf06b",
    });

    expect(candidate.sharedTopology).toEqual({
      commandBufferCount: 34,
      completedIdleCount: 34,
      copyCommandCount: 6,
      maximumOutstandingCommandBuffers: 1,
      modelCommandBufferCount: 33,
      modelDispatchPrimitiveCount: 624,
      modelPhysicalPrimitiveDispatchCount: 628,
      modelQuantumCount: 33,
      queueDrainCount: 34,
      readbackCommandBufferCount: 1,
      requestedIdleMilliseconds: 34,
      residentWeightBytes: 1_325_768_704,
    });

    expect(candidate.cases.map((entry) => ({
      id: entry.id,
      cache: [entry.cachedTokensBeforeAppend, entry.cacheCapacity],
      timed: entry.timing.timedWindowMilliseconds,
      sampling: entry.timing.samplingMilliseconds,
      model: [
        entry.timing.modelQuantaSubmitThroughDrainMilliseconds,
        entry.timing.modelQuantaWallMilliseconds,
        entry.timing.modelQuantaIdleMilliseconds,
      ],
      layers: [
        entry.timing.layersSubmitThroughDrainMilliseconds,
        entry.timing.layersWallMilliseconds,
      ],
      head: entry.timing.tiedHeadSubmitThroughDrainMilliseconds,
      readback: entry.timing.readbackWallMilliseconds,
    }))).toEqual([
      {
        id: "cot-m1-short",
        cache: [120, 512],
        timed: 498.39999997615814,
        sampling: 279.19999998807907,
        model: [148.69999992847443, 196.5, 41.40000003576279],
        layers: [133.30000001192093, 173],
        head: 11.899999976158142,
        readback: 22.600000023841858,
      },
      {
        id: "cot-m1-mid",
        cache: [160, 1024],
        timed: 410,
        sampling: 203.30000001192093,
        model: [154.30000030994415, 203.30000001192093, 43.69999980926514],
        layers: [140.00000017881393, 182.30000001192093],
        head: 10.900000035762787,
        readback: 3.399999976158142,
      },
      {
        id: "cot-m1-long",
        cache: [212, 2048],
        timed: 335.5,
        sampling: 85.89999997615814,
        model: [190.29999995231628, 244.89999997615814, 43.700000047683716],
        layers: [165.5, 207.5],
        head: 15.699999988079071,
        readback: 4.5,
      },
      {
        id: "semantic-m2-short",
        cache: [268, 768],
        timed: 406.69999998807907,
        sampling: 179.39999997615814,
        model: [171.9999998807907, 221.30000001192093, 42.59999996423721],
        layers: [156.30000001192093, 199.20000004768372],
        head: 10.999999940395355,
        readback: 5.899999976158142,
      },
      {
        id: "semantic-m2-mid",
        cache: [328, 1280],
        timed: 385.5,
        sampling: 182.10000002384186,
        model: [152.80000007152557, 198, 40.00000011920929],
        layers: [140.90000015497208, 179.9000000357628],
        head: 9.899999976158142,
        readback: 5.300000011920929,
      },
      {
        id: "semantic-m2-long",
        cache: [401, 2048],
        timed: 387.39999997615814,
        sampling: 169,
        model: [167.50000005960464, 211.69999998807907, 39.5],
        layers: [155.20000004768372, 192.80000001192093],
        head: 9.699999988079071,
        readback: 6.699999988079071,
      },
    ]);

    expect(candidate.cases.every((entry) =>
      entry.correctness.bitExact &&
      entry.correctness.bitMismatchCount === 0 &&
      JSON.stringify(entry.referenceSample) === JSON.stringify(entry.timedSample)
    )).toBe(true);
    expect(candidate.cases.map((entry) => ({
      id: entry.id,
      words: entry.correctness.comparedU32WordCount,
      readback: [
        entry.readback.rows,
        entry.readback.rawLogitBytes,
        entry.readback.bufferBytes,
        entry.readback.shardCount,
      ],
      work: [
        entry.work.logicalMultiplyAdds,
        entry.work.scheduledMultiplyAdds,
      ],
    }))).toEqual([
      { id: "cot-m1-short", words: 217204, readback: [1, 434408, 434688, 5], work: [662818816, 10605297664] },
      { id: "cot-m1-mid", words: 217204, readback: [1, 434408, 434688, 5], work: [662818816, 10605297664] },
      { id: "cot-m1-long", words: 217204, readback: [1, 434408, 434688, 5], work: [662818816, 10605297664] },
      { id: "semantic-m2-short", words: 434408, readback: [2, 868816, 869120, 5], work: [1325637632, 10605297664] },
      { id: "semantic-m2-mid", words: 434408, readback: [2, 868816, 869120, 5], work: [1325637632, 10605297664] },
      { id: "semantic-m2-long", words: 434408, readback: [2, 868816, 869120, 5], work: [1325637632, 10605297664] },
    ]);

    expect(candidate.semanticHeadOpportunity).toMatchObject({
      firstTokenId: 151669,
      tokenCount: 64000,
      fullHead: {
        logicalWeightTrafficBytes: 444833792,
        rawLogitBytes: 868816,
        scheduledMultiplyAdds: 3558866944,
      },
      restrictedCodeHead: {
        logicalWeightTrafficBytes: 131072000,
        rawLogitBytes: 256000,
        scheduledMultiplyAdds: 1050673152,
        intersections: [
          { shardIndex: 3, globalFirstRow: 151669, localFirstRow: 4213, rowCount: 44939 },
          { shardIndex: 4, globalFirstRow: 196608, localFirstRow: 0, rowCount: 19061 },
        ],
      },
      avoidablePerToken: {
        logicalWeightTrafficBytes: 313761792,
        rawLogitBytes: 612816,
        scheduledMultiplyAdds: 2508193792,
      },
      avoidableFractions: {
        logicalWeightAndRawLogit: 0.7053461262223532,
        scheduledMultiplyAdds: 0.704773129051267,
      },
      terminalEos: {
        tokenId: 151645,
        terminalOnly: true,
        includedInRestrictedCodeHead: false,
        tiedEmbeddingRemainsFullyResident: true,
      },
    });

    expect(candidate.externalThermalCoverage).toMatchObject({
      observationCount: 342,
      nonNominalObservationCount: 0,
      firstSequence: 0,
      lastSequence: 341,
      elapsedMilliseconds: 341005.0982500543,
      maximumPollGapMilliseconds: 1005.0118330400437,
      spansPreGate: true,
      spansTimedInterval: true,
      spansCleanupCompletion: true,
    });
    expect(candidate.browserPreGate).toMatchObject({
      durationMilliseconds: 36000,
      observationCount: 37,
      nonNominalObservationCount: 0,
    });
    expect(candidate.cancellation).toMatchObject({
      rejectionName: "AbortError",
      encodedCommandBufferCount: 1,
      submissionCount: 1,
      queueDrainCount: 1,
      completedIdleCount: 1,
      firstQuantumFullyDrained: true,
      laterEncodingPrevented: true,
      laterSubmissionPrevented: true,
    });
    expect(candidate.cleanup).toMatchObject({
      createdBufferCount: 1338,
      uniqueDestroyedBufferCount: 1338,
      totalDestroyCallCount: 1338,
      liveTrackedBufferCount: 0,
      idempotentExecutorDestroy: true,
      runtimeEventCount: 0,
      deviceDestroyedAfterEventCheck: true,
    });
    expect(candidate.responsiveness).toMatchObject({
      workerMaximumTimerGapMilliseconds: 2118.5,
      pageMaximumAnimationFrameGapMilliseconds: 94.09999999999854,
      pageMaximumTimerGapMilliseconds: 107.19999998807907,
    });
    expect(candidate.scope).toMatchObject({
      productionIntegrationPerformed: false,
      restrictedSemanticHeadExecuted: false,
      samplingMechanismChanged: false,
      schedulingTopologyChanged: false,
    });

    const delta = committed.metrics.delta as unknown as {
      rankedNextMechanisms: readonly { rank: number; mechanism: string }[];
      semanticRestrictionProjection: Record<string, unknown>;
      productionIntegrationPerformed: boolean;
    };
    expect(delta.rankedNextMechanisms).toEqual([
      expect.objectContaining({ rank: 1, mechanism: "state-specific compact semantic head plus compact sampler" }),
      expect.objectContaining({ rank: 2, mechanism: "dedicated low-row FP16 GEMV with FP32 accumulation" }),
      expect.objectContaining({ rank: 3, mechanism: "bounded multi-quantum command-buffer batching" }),
    ]);
    expect(delta.semanticRestrictionProjection).toMatchObject({
      estimatedSemanticTokenSavingMilliseconds: [132, 140],
      estimatedSemanticTokenSpeedup: [1.5, 1.6],
      observed: false,
      productionCorrectnessEstablished: false,
    });
    expect(delta.productionIntegrationPerformed).toBe(false);

    expect(committed.artifacts).toEqual(RAW_ARTIFACTS.map(([name, sha256]) => ({
      location: `optimization/artifacts/OPT-0010/raw/${name}`,
      sha256,
    })));
    expect(`${stringifyOptimizationResult(committed)}\n`).toBe(text);
  });

  it("closes the record and ledger without claiming measured projection or integration", async () => {
    const [record, ledger] = await Promise.all([
      readFile(new URL(
        "../optimization/experiments/OPT-0010-package-native-planner-token-profiler.md",
        import.meta.url,
      ), "utf8"),
      readFile(new URL("../optimization/LEDGER.md", import.meta.url), "utf8"),
    ]);

    expect(record).toContain("- Evidence: `positive`");
    expect(record).toContain("- Disposition: `benchmark-only`");
    expect(record).toContain("1,954,836 compared U32 words were bit-identical");
    expect(record).toContain("The semantic restriction was **not executed**");
    expect(record).toContain("unmeasured projection");
    expect(record).toContain("no production integration occurred");
    expect(record).not.toContain("Result JSON: pending");

    const ledgerRow = ledger.split("\n").find((line) =>
      line.startsWith("| OPT-0010 |"),
    );
    expect(ledgerRow).toContain("| positive | benchmark-only |");
    expect(ledgerRow).toContain("[result](results/OPT-0010/result.json)");
    expect(ledgerRow).toContain("unmeasured projection");
    expect(ledgerRow).toContain("no production integration occurred");
  });
});
