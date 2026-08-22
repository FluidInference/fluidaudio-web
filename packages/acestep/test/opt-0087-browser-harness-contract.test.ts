import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { describe, expect, it } from "vitest";

import {
  OPT_0087_COLD_GENERIC_ARM_MAP_COUNT,
  OPT_0087_COLD_GENERIC_CONTROL_BUFFER_COUNT,
  OPT_0087_COLD_GENERIC_CONTROL_TOTAL_BYTES,
  OPT_0087_NO_NEW_CONTROL_ARM_MAP_COUNT,
  OPT_0087_PAIR_ORDERS,
  OPT_0087_THERMAL_COMMAND,
  OPT_0087_THERMAL_TRACE_SCHEMA,
  evaluateOpt0087TimingGate,
  Opt0087ResourcePairTopologyError,
  Opt0087ResourceTopologyError,
  validateOpt0087ExplicitArmResources,
  validateOpt0087ResourcePair,
  validateOpt0087RunIdentity,
  validateOpt0087ThermalCompletion,
  validateOpt0087ThermalLaunch,
  validateOpt0087Topology,
  type Opt0087ArmTimingSamples,
  type Opt0087ExplicitArmResourceDelta,
  type Opt0087ThermalCompletion,
  type Opt0087ThermalLaunch,
} from "./browser/opt-0087-planner-package-native-low-row-gemv-contract.js";
import {
  parseOpt0087ThermalCompletion,
  parseOpt0087ThermalLaunch,
} from "./browser/opt-0087-planner-package-native-low-row-gemv.js";
import type {
  AcePlannerOpt0087InvocationDiagnostics,
} from "../src/webgpu/planner-executor.js";

describe("OPT-0087 browser harness contract", () => {
  it("balances eight pair orders and freezes the full package topology", () => {
    expect(OPT_0087_PAIR_ORDERS).toHaveLength(8);
    expect(OPT_0087_PAIR_ORDERS.filter(([first]) => first === "control"))
      .toHaveLength(4);
    expect(OPT_0087_PAIR_ORDERS.filter(([first]) => first === "candidate"))
      .toHaveLength(4);
    expect(() => validateOpt0087Topology(diagnostics("generic-a", 1),
      "generic-a", 1)).not.toThrow();
    expect(() => validateOpt0087Topology(diagnostics("direct-b", 2),
      "direct-b", 2)).not.toThrow();
    expect(() => validateOpt0087Topology({
      ...diagnostics("direct-b", 2),
      totalCommandBuffers: 33,
    } as unknown as AcePlannerOpt0087InvocationDiagnostics, "direct-b", 2)).toThrow(
      /topology changed/,
    );
  });

  it("passes only the complete frozen timing gate", () => {
    const passing = evaluateOpt0087TimingGate([
      path("cot-m1-middle-full", 155, 80, 30, 20, 250, 170),
      path("semantic-m2-middle-full", 140, 75, 28, 19, 235, 155),
    ]);
    expect(passing).toMatchObject({
      everyPathLayerMedianBelowControl: true,
      everyPathHeadMedianBelowControl: true,
      everyPathModelMedianBelowControl: true,
      everyPathCompleteMedianBelowControl: true,
      candidateModelWins: 16,
      passed: true,
    });
    expect(passing.aggregateLayerSpeedup).toBeGreaterThanOrEqual(1.5);
    expect(passing.aggregateModelMedianSavingMilliseconds)
      .toBeGreaterThanOrEqual(60);
    expect(passing.projected1010DrawSavingSeconds).toBeGreaterThanOrEqual(60);

    const tooSmall = evaluateOpt0087TimingGate([
      path("cot-m1-middle-full", 155, 100, 30, 20, 250, 195),
      path("semantic-m2-middle-full", 140, 90, 28, 19, 235, 180),
    ]);
    expect(tooSmall.aggregateModelMedianSavingMilliseconds).toBe(55);
    expect(tooSmall.passed).toBe(false);
    expect(() => evaluateOpt0087TimingGate([
      path("cot-m1-middle-full", 155, 80, 30, 20, 250, 0),
      path("semantic-m2-middle-full", 140, 75, 28, 19, 235, 155),
    ])).toThrow(/timing samples/);
  });

  it("separates cold generic compile-cache creation from warmed arm resources", () => {
    const coldGeneric: Opt0087ExplicitArmResourceDelta = Object.freeze({
      createdBufferCount: 5,
      destroyedBufferCount: 0,
      createdByteLength: 1_280,
      destroyedByteLength: 0,
      successfulMapCount: 7,
      failedMapCount: 0,
      unmapCount: 7,
      destroyedWhileMappedCount: 0,
      repeatedDestroyCallCount: 0,
      liveBufferCountBefore: 100,
      liveBufferCountAfter: 105,
      liveByteLengthBefore: 2_000_000_000,
      liveByteLengthAfter: 2_000_001_280,
      activeMapCountBefore: 0,
      activeMapCountAfter: 0,
    });
    const noNewBuffers: Opt0087ExplicitArmResourceDelta = Object.freeze({
      ...coldGeneric,
      createdBufferCount: 0,
      createdByteLength: 0,
      successfulMapCount: 2,
      unmapCount: 2,
      liveBufferCountAfter: coldGeneric.liveBufferCountBefore,
      liveByteLengthAfter: coldGeneric.liveByteLengthBefore,
    });
    const directAfterCold: Opt0087ExplicitArmResourceDelta = Object.freeze({
      ...noNewBuffers,
      liveBufferCountBefore: coldGeneric.liveBufferCountAfter,
      liveBufferCountAfter: coldGeneric.liveBufferCountAfter,
      liveByteLengthBefore: coldGeneric.liveByteLengthAfter,
      liveByteLengthAfter: coldGeneric.liveByteLengthAfter,
    });
    expect(OPT_0087_COLD_GENERIC_CONTROL_BUFFER_COUNT).toBe(5);
    expect(OPT_0087_COLD_GENERIC_CONTROL_TOTAL_BYTES).toBe(1_280);
    expect(OPT_0087_COLD_GENERIC_ARM_MAP_COUNT).toBe(7);
    expect(OPT_0087_NO_NEW_CONTROL_ARM_MAP_COUNT).toBe(2);
    expect(() => validateOpt0087ExplicitArmResources(
      coldGeneric,
      "cold-generic-a-compile-cache",
    )).not.toThrow();
    expect(() => validateOpt0087ExplicitArmResources(
      noNewBuffers,
      "no-new-compile-cache-buffer",
    )).not.toThrow();
    expect(() => validateOpt0087ResourcePair(
      coldGeneric,
      directAfterCold,
      "cold-warmup-a-to-direct-b",
    )).not.toThrow();
    expect(() => validateOpt0087ResourcePair(
      noNewBuffers,
      noNewBuffers,
      "warmed-timed-pair",
    )).not.toThrow();

    // The original zero-allocation assumption must not false-pass a first-cold
    // generic A, and the setup correction must not leak into warmed timing.
    try {
      validateOpt0087ExplicitArmResources(
        noNewBuffers,
        "cold-generic-a-compile-cache",
      );
      throw new Error("expected cold-generic OPT-0087 resource mismatch");
    } catch (error) {
      expect(error).toBeInstanceOf(Opt0087ResourceTopologyError);
      const diagnostic = (error as Opt0087ResourceTopologyError).diagnostic;
      expect(diagnostic).toMatchObject({
        schema: "ace-opt-0087-explicit-arm-resource-mismatch-v1",
        expectation: "cold-generic-a-compile-cache",
        expected: {
          createdBufferCount: 5,
          createdByteLength: 1_280,
          successfulMapCount: 7,
          resourceDeltaCompletesBeforeAuthoritativeModelWall: true,
        },
        actual: {
          createdBufferCount: 0,
          createdByteLength: 0,
          successfulMapCount: 2,
        },
      });
    }
    expect(() => validateOpt0087ExplicitArmResources(
      coldGeneric,
      "no-new-compile-cache-buffer",
    )).toThrow(Opt0087ResourceTopologyError);
    expect(() => validateOpt0087ExplicitArmResources({
      ...noNewBuffers,
      liveBufferCountAfter: noNewBuffers.liveBufferCountBefore + 5,
      liveByteLengthAfter: noNewBuffers.liveByteLengthBefore + 1_280,
    }, "no-new-compile-cache-buffer")).toThrow(
      Opt0087ResourceTopologyError,
    );
    expect(() => validateOpt0087ExplicitArmResources(
      noNewBuffers,
      "unknown" as "no-new-compile-cache-buffer",
    )).toThrow(/expectation is invalid/);
    expect(() => validateOpt0087ResourcePair(
      coldGeneric,
      noNewBuffers,
      "cold-warmup-a-to-direct-b",
    )).toThrow(Opt0087ResourcePairTopologyError);
    expect(() => validateOpt0087ResourcePair(
      coldGeneric,
      noNewBuffers,
      "warmed-timed-pair",
    )).toThrow(Opt0087ResourceTopologyError);
    expect(() => validateOpt0087ResourcePair(
      noNewBuffers,
      {
        ...noNewBuffers,
        liveBufferCountBefore: noNewBuffers.liveBufferCountBefore + 1,
        liveBufferCountAfter: noNewBuffers.liveBufferCountAfter + 1,
        liveByteLengthBefore: noNewBuffers.liveByteLengthBefore + 256,
        liveByteLengthAfter: noNewBuffers.liveByteLengthAfter + 256,
      },
      "warmed-timed-pair",
    )).toThrow(Opt0087ResourcePairTopologyError);
  });

  it("requires authenticated identities and the full continuous thermal binding", () => {
    expect(validateOpt0087RunIdentity({
      implementationCommit: "a".repeat(40),
      harnessCommit: "b".repeat(40),
      machineModel: "Mac16,5",
      osVersion: "15.6",
      osBuild: "24G84",
      browserVersion: "Chrome 140",
      gpuCoreCount: 10,
      memoryBytes: 16 * 1024 ** 3,
    }).gpuCoreCount).toBe(10);
    expect(() => validateOpt0087RunIdentity({})).toThrow(/identity/);

    const ready = 1_000_000;
    const launch = thermalLaunch(ready);
    expect(() => validateOpt0087ThermalLaunch(
      launch,
      ready,
      launch.gateCompletedAtEpochMilliseconds,
    ))
      .not.toThrow();
    const cleanup = ready + 90_000;
    const completion = thermalCompletion(launch, cleanup + 1_000);
    expect(() => validateOpt0087ThermalCompletion(
      completion,
      launch,
      cleanup,
      cleanup + 1_000,
    )).not.toThrow();
    expect(completion.laterNonNominalDisclosed).toBe(true);
    expect(() => validateOpt0087ThermalCompletion(
      { ...completion, completedAtEpochMilliseconds: cleanup - 1 },
      launch,
      cleanup,
      cleanup,
    )).toThrow(/through-cleanup/);
    expect(() => validateOpt0087ThermalCompletion({
      ...completion,
      transitions: [{
        atEpochMilliseconds: launch.gateStartedAtEpochMilliseconds + 1_000,
        level: 1,
      }],
      finalLevel: 1,
    }, launch, cleanup, cleanup + 1_000)).toThrow(/through-cleanup/);
    expect(() => validateOpt0087ThermalLaunch({
      ...launch,
      gateStartedAtEpochMilliseconds: Number.NaN,
    }, ready)).toThrow(/launch gate/);
  });

  it("parses the launch and later-nonnominal raw trace without weakening the gate", () => {
    const ready = 2_000_000;
    const launch = thermalLaunch(ready);
    const launchFields = new URLSearchParams({
      thermalSource: launch.source,
      thermalCommand: launch.command,
      thermalTraceStartedAtEpochMilliseconds:
        String(launch.traceStartedAtEpochMilliseconds),
      thermalGateStartedAtEpochMilliseconds:
        String(launch.gateStartedAtEpochMilliseconds),
      thermalGateCompletedAtEpochMilliseconds:
        String(launch.gateCompletedAtEpochMilliseconds),
      thermalGateObservations: String(launch.observationCount),
      thermalPollMilliseconds: String(launch.pollMilliseconds),
      thermalGateMaximumPollGapMilliseconds:
        String(launch.maximumPollGapMilliseconds),
      thermalGateNonNominalObservations: "0",
      thermalGateMissingObservations: "0",
    });
    expect(parseOpt0087ThermalLaunch(
      launchFields,
      ready,
      launch.gateCompletedAtEpochMilliseconds,
    )).toEqual(launch);

    const cleanup = ready + 90_000;
    const completion = thermalCompletion(launch, cleanup + 1_000);
    const completionFields = new URLSearchParams({
      thermalTraceSchema: completion.schema,
      thermalTraceSha256: completion.sha256,
      thermalTraceByteLength: String(completion.byteLength),
      thermalTraceCompletedAtEpochMilliseconds:
        String(completion.completedAtEpochMilliseconds),
      thermalTraceObservations: String(completion.observationCount),
      thermalTraceMaximumPollGapMilliseconds:
        String(completion.maximumPollGapMilliseconds),
      thermalTraceNonNominalObservations:
        String(completion.nonNominalObservationCount),
      thermalTraceMissingObservations: "0",
      thermalTraceInitialLevel: "0",
      thermalTraceFinalLevel: String(completion.finalLevel),
      thermalTraceTransitionsJson: JSON.stringify(completion.transitions),
    });
    expect(parseOpt0087ThermalCompletion(
      completionFields,
      launch,
      cleanup,
      cleanup + 1_000,
    )).toEqual(completion);
  });

  it("keeps the browser runner on one paired owner, full logits, and depth one", () => {
    const worker = readFileSync(resolve(
      process.cwd(),
      "test/browser/opt-0087-planner-package-native-low-row-gemv-worker.ts",
    ), "utf8");
    expect(worker).toContain("AcePlannerGpuExecutor.createForOpt0087({");
    expect(worker).toContain("await executor.prefill(fixture.prefill);");
    expect(worker).toContain("await executor.decodeForOpt0087(");
    expect(worker).toContain("new Uint32Array(");
    expect(worker).toContain("requireExactCacheAppend(");
    expect(worker).toContain("cacheAppendU32LeSha256");
    expect(worker).toContain("cleanupFailures.map(errorText)");
    expect(worker).toMatch(/finally \{\s+try \{\s+context\?\.destroy\(\)/);
    expect(worker).toContain("OPT_0087_PAIR_ORDERS.length !== 8");
    expect(worker).toContain("opt0085DepthTwoDisabled: true");
    expect(worker).toContain("class Opt0087GpuResourceTracker");
    expect(worker).toContain("requireBalancedAfterCleanup");
    expect(worker).toContain(
      "validateOpt0087ExplicitArmResources(receipt, expectation)",
    );
    expect(worker).toContain("compileCacheDispatchConstructionResources");
    expect(worker).toContain("M1-generic-A-cold");
    expect(worker).toContain("M2-direct-B-no-new");
    expect(worker).toMatch(
      /const control = await executeArm\([\s\S]*?"control",\s*"cold-generic-a-compile-cache"/,
    );
    expect(worker).toMatch(
      /const candidate = await executeArm\([\s\S]*?"candidate",\s*"no-new-compile-cache-buffer"/,
    );
    expect(worker).toMatch(
      /const execution = await executeArm\([\s\S]*?arm,\s*"no-new-compile-cache-buffer"/,
    );
    expect(worker).toContain(
      "everyTimedPairRequiresExactNoNewAllocationResources: true",
    );
    expect(worker).toContain("validateOpt0087ResourcePair(");
    expect(worker).toContain("diagnostic: error.diagnostic");
    expect(worker).toContain("error instanceof Opt0087ResourcePairTopologyError");
    expect(worker).toContain("prepared.abortController.signal.throwIfAborted()");
    expect(worker).toContain("await yieldToWorker()");
    expect(worker).toContain("commandBufferBoundariesChanged: false");
    expect(worker).toContain("terminalReadbackEvidenceCopiesAdded: true");
    expect(worker).not.toContain("decodeForOpt0085");
    expect(worker).not.toContain("sampleCompact");

    const html = readFileSync(resolve(
      process.cwd(),
      "test/browser/opt-0087-planner-package-native-low-row-gemv.html",
    ), "utf8");
    expect(html).toContain("exactly 33 model quanta");
    expect(html).toContain("eight balanced same-state pairs per row count");
    expect(html).toContain("first cold generic-A warmup for each M1/M2 shape");
    expect(html).toContain("persist across phase replay until owner cleanup");
    expect(html).toContain("every subsequently warmed A/B arm create no buffers");
    expect(html).toContain('id="cancel"');
    expect(html).toContain(OPT_0087_THERMAL_COMMAND);
    expect(html).toContain(OPT_0087_THERMAL_TRACE_SCHEMA);

    const page = readFileSync(resolve(
      process.cwd(),
      "test/browser/opt-0087-planner-package-native-low-row-gemv.ts",
    ), "utf8");
    expect(page).toContain('"passed-all-opt-0087-browser-gates"');
    expect(page).toContain('"failed-opt-0087-performance-gate"');
    expect(page).toContain('worker.postMessage({ type: "cancel" })');
    expect(page).toContain("owner remains READY");
    expect(page).not.toContain("passed-all-opt-0085-gates");
    const launchHandler = page.slice(
      page.indexOf('runTimed.addEventListener("click"'),
      page.indexOf('finalize.addEventListener("click"'),
    );
    expect(launchHandler).not.toContain("worker.terminate");
    expect(launchHandler).not.toContain("finishFailure");
    expect(launchHandler).not.toContain("once: true");
    const finalizeHandler = page.slice(
      page.indexOf('finalize.addEventListener("click"'),
      page.indexOf('window.addEventListener("beforeunload"'),
    );
    expect(finalizeHandler).not.toContain("once: true");
    expect(finalizeHandler).toContain("completed timing result remains stable");
  });
});

function diagnostics(
  arm: "generic-a" | "direct-b",
  rows: 1 | 2,
): AcePlannerOpt0087InvocationDiagnostics {
  const quantumTimings = [
    timing(0, "embedding", null),
    ...Array.from({ length: 28 }, (_, layer) =>
      timing(layer + 1, "layer", layer)),
    timing(29, "final-norm", null),
    timing(30, "last-row-gather", null),
    timing(31, "tied-lm-head", null),
    timing(32, "tied-lm-head", null),
  ];
  const roles = [
    "query-projection",
    "key-projection",
    "value-projection",
    "attention-output-projection",
    "gate-projection",
    "up-projection",
    "down-projection",
  ] as const;
  const denseSelections = [
    ...Array.from({ length: 28 }, () => roles).flat(),
    ...Array<"tied-lm-head">(5).fill("tied-lm-head"),
  ].map((role) => ({
    role,
    requestedArm: arm,
    selectedArm: arm,
    reason: arm === "generic-a" ? "control-requested" as const :
      "direct-selected" as const,
    rows,
    inner: 1_024,
    columns: 2_048,
  }));
  const cacheAppendKeyValueWordCount = rows === 1 ? 57_344 : 114_688;
  const cacheAppendWords = new Uint32Array(
    cacheAppendKeyValueWordCount + rows,
  );
  cacheAppendWords.fill(1, cacheAppendKeyValueWordCount);
  return Object.freeze({
    schema: "ace-opt-0087-planner-package-invocation-v1",
    arm,
    phaseKind: "decode",
    modelQuantumCount: 33,
    totalCommandBuffers: 34,
    commandBuffersSubmitted: 34,
    trueQueueDrainCount: 34,
    cooperativeIdleTurns: 34,
    requestedCooperativeIdleMs: 34,
    maximumOutstandingCommandBuffers: 1,
    readbackMapCount: 2,
    readbackShardCount: 5,
    readbackByteLength: 1_737_856,
    cacheAppendReadbackByteLength: rows === 1 ? 229_632 : 459_008,
    cacheAppendLogicalByteLength: rows === 1 ? 229_380 : 458_760,
    cacheAppendCopyCount: rows === 1 ? 449 : 898,
    cacheAppendKeyValueWordCount,
    cacheAppendValidityWordCount: rows,
    cacheAppendWords,
    logitRows: rows,
    logitTokenCount: 217_204,
    accountedGpuBytes: 2_000_000_000,
    arenaBufferCount: 100,
    layerQuantumCount: 28,
    tiedHeadQuantumCount: 2,
    quantumTimings,
    transformerLayerWallMilliseconds: 28,
    tiedHeadWallMilliseconds: 2,
    readbackWallMilliseconds: 1,
    modelThroughReadbackWallMilliseconds: 34,
    writeStatusWords: Array<number>(rows).fill(1),
    denseSelections,
    headQuantumSliceFirstRows: [[0, 49_152], [98_304, 147_456, 196_608]],
  });
}

function timing(
  index: number,
  kind: "embedding" | "layer" | "final-norm" | "last-row-gather" |
    "tied-lm-head",
  layer: number | null,
) {
  return Object.freeze({
    index,
    kind,
    layer,
    primitiveCount: 1,
    submitThroughDrainWallMilliseconds: 1,
  });
}

function samples(value: number): Opt0087ArmTimingSamples {
  return Object.freeze({
    transformerLayerWallMilliseconds: Array<number>(8).fill(value),
    tiedHeadWallMilliseconds: Array<number>(8).fill(value),
    modelThroughReadbackWallMilliseconds: Array<number>(8).fill(value),
    completeTokenWallMilliseconds: Array<number>(8).fill(value),
  });
}

function path(
  id: "cot-m1-middle-full" | "semantic-m2-middle-full",
  controlLayer: number,
  candidateLayer: number,
  controlHead: number,
  candidateHead: number,
  controlModel: number,
  candidateModel: number,
) {
  return Object.freeze({
    id,
    control: Object.freeze({
      ...samples(controlModel),
      transformerLayerWallMilliseconds: Array<number>(8).fill(controlLayer),
      tiedHeadWallMilliseconds: Array<number>(8).fill(controlHead),
      completeTokenWallMilliseconds: Array<number>(8).fill(controlModel + 1),
    }),
    candidate: Object.freeze({
      ...samples(candidateModel),
      transformerLayerWallMilliseconds: Array<number>(8).fill(candidateLayer),
      tiedHeadWallMilliseconds: Array<number>(8).fill(candidateHead),
      completeTokenWallMilliseconds: Array<number>(8).fill(candidateModel + 1),
    }),
  });
}

function thermalLaunch(ready: number): Opt0087ThermalLaunch {
  return Object.freeze({
    source: "notifyutil-com.apple.system.thermalpressurelevel",
    command: OPT_0087_THERMAL_COMMAND,
    traceStartedAtEpochMilliseconds: ready - 10_000,
    gateStartedAtEpochMilliseconds: ready,
    gateCompletedAtEpochMilliseconds: ready + 30_000,
    observationCount: 31,
    pollMilliseconds: 1_000,
    maximumPollGapMilliseconds: 1_000,
    nonNominalObservationCount: 0,
    missingObservationCount: 0,
    readyToGateDelayMilliseconds: 0,
    launchDelayMilliseconds: 0,
  });
}

function thermalCompletion(
  launch: Opt0087ThermalLaunch,
  completedAtEpochMilliseconds: number,
): Opt0087ThermalCompletion {
  const transitions = Object.freeze([Object.freeze({
    atEpochMilliseconds: launch.gateCompletedAtEpochMilliseconds + 10_000,
    level: 1 as const,
  }), Object.freeze({
    atEpochMilliseconds: launch.gateCompletedAtEpochMilliseconds + 20_000,
    level: 0 as const,
  })]);
  return Object.freeze({
    schema: OPT_0087_THERMAL_TRACE_SCHEMA,
    sha256: "c".repeat(64),
    byteLength: 12_345,
    completedAtEpochMilliseconds,
    observationCount: Math.floor((completedAtEpochMilliseconds -
      launch.traceStartedAtEpochMilliseconds) / 1_000) + 1,
    maximumPollGapMilliseconds: 1_000,
    nonNominalObservationCount: 10,
    missingObservationCount: 0,
    initialLevel: 0,
    finalLevel: 0,
    transitions,
    coversCleanup: true,
    laterNonNominalDisclosed: true,
  });
}
