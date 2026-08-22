import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";

import { describe, expect, it } from "vitest";

import {
  parseOptimizationResultJson,
  stringifyOptimizationResult,
} from "../benchmark/result.js";
import {
  aceFp16VaeCongruentConvTranspose1dWgsl,
} from "../src/webgpu/kernels/vae-conv-transpose1d-fp16.js";
import {
  aceOpt0022VaeConvTranspose1dSubgroupWgsl,
} from "../src/webgpu/kernels/vae-conv-transpose1d-fp16-subgroup.js";
import {
  buildOpt0022C300Topology,
  buildOpt0022TimingOrders,
  summarizeOpt0022Timing,
} from "./browser/opt-0022-vae-conv-transpose1d-subgroup-polyphase-ab.js";

type Arm = "A" | "B";
type Stratum = "first" | "interior" | "tail";

interface ReceiptTimingRound {
  readonly roundIndex: number;
  readonly order: "AB" | "BA";
  readonly A: number;
  readonly B: number;
}

interface ReceiptTimingStratum {
  readonly operationLabel: string;
  readonly stratum: Stratum;
  readonly weight: number;
  readonly rounds: readonly ReceiptTimingRound[];
  readonly medians: Readonly<Record<Arm, number>>;
  readonly positionMedians: Readonly<Record<"first" | "second",
    Readonly<Record<Arm, number>>>>;
  readonly bFaster: boolean;
  readonly speedup: number;
}

interface ReceiptCorrectnessRange {
  readonly stratum: Stratum;
  readonly weight: number;
  readonly rawU16MismatchCount: number;
  readonly deterministicRerun: boolean;
  readonly allPhasesCovered: boolean;
  readonly outputSha256: string;
}

interface ReceiptCorrectnessCase {
  readonly label: string;
  readonly selectedProbeCount: number;
  readonly exactGraphRangeCount: number;
  readonly comparedU16Count: number;
  readonly rawU16MismatchCount: number;
  readonly correctnessExecutionsPerArmPerProbe: number;
  readonly deterministicRerun: boolean;
  readonly allPhasesCoveredInEveryProbe: boolean;
  readonly firstAndLastOneTapBoundariesCovered: boolean;
  readonly qNaNPrefillCompleteWrites: boolean;
  readonly completeSelectedWrites: boolean;
  readonly guardsAndAdjacentCanariesUntouched: boolean;
  readonly ranges: readonly ReceiptCorrectnessRange[];
}

interface ThermalGate {
  readonly source: string;
  readonly startedAtEpochMilliseconds: number;
  readonly completedAtEpochMilliseconds: number;
  readonly durationMilliseconds: number;
  readonly observationCount: number;
  readonly pollMilliseconds: number;
  readonly maximumPollGapMilliseconds: number;
  readonly nonNominalObservationCount: number;
  readonly launchDelayMilliseconds: number;
}

interface Opt0022Receipt {
  readonly schema: string;
  readonly experimentId: string;
  readonly status: string;
  readonly classification: string;
  readonly recordedAt: string;
  readonly identity: {
    readonly registrationCommit: string;
    readonly integratedCurrentCommit: string;
    readonly currentCore: {
      readonly bytes: number;
      readonly sha256: string;
      readonly kernelId: string;
      readonly weightLayout: string;
    };
    readonly candidateCore: {
      readonly bytes: number;
      readonly sha256: string;
      readonly kernelId: string;
      readonly weightLayout: string;
    };
    readonly generatedShaders: readonly {
      readonly label: string;
      readonly A: string;
      readonly B: string;
    }[];
  };
  readonly environment: {
    readonly userAgent: string;
    readonly adapterInfo: Readonly<Record<string, unknown>>;
    readonly deviceFeatures: readonly string[];
    readonly deviceLimits: Readonly<Record<string, number>>;
  };
  readonly preparation: {
    readonly completedAtEpochMilliseconds: number;
    readonly memoryHighWaterBytes: number;
    readonly allPipelinesCompiledAndWarmedByCorrectness: boolean;
  };
  readonly protocol: {
    readonly thermal: ThermalGate;
    readonly fullLayoutAndOutputCorrectnessCompletedBeforeThermalGate: boolean;
    readonly bothKernelsCompiledAndWarmedBeforeThermalGate: boolean;
    readonly timingRounds: number;
    readonly thermalTraceRequiredThroughEpochMilliseconds: number;
    readonly continuousExternalThermalTraceRequiredThroughCleanup: boolean;
    readonly oneThirtySecondNominalGate: boolean;
    readonly unchangedThermalRetryPerformed: boolean;
  };
  readonly layout: {
    readonly layoutId: string;
    readonly operationCount: number;
    readonly comparedU16Count: number;
    readonly mismatchCount: number;
    readonly residentByteIncrease: number;
    readonly cases: readonly {
      readonly comparedU16Count: number;
      readonly mismatchCount: number;
      readonly rawU16Exact: boolean;
      readonly nativeBytes: number;
      readonly polyphaseBytes: number;
      readonly residentByteIncrease: number;
    }[];
  };
  readonly correctness: {
    readonly operationCount: number;
    readonly selectedProbeCount: number;
    readonly exactGraphRangeCount: number;
    readonly inverseLayoutComparedU16Count: number;
    readonly inverseLayoutMismatchCount: number;
    readonly outputExecutionCount: number;
    readonly outputExactComparisonCount: number;
    readonly outputComparedU16Count: number;
    readonly outputMismatchCount: number;
    readonly deterministicRerunHashes: boolean;
    readonly allPhasesCoveredInEveryProbe: boolean;
    readonly bothOneTapBoundariesCoveredPerOperation: boolean;
    readonly qNaNPrefillCompleteWrites: boolean;
    readonly guardsAndAdjacentCanariesUntouched: boolean;
    readonly uncapturedGpuErrorCount: number;
    readonly cases: readonly ReceiptCorrectnessCase[];
    readonly immutableAfterCorrectness: {
      readonly sourceCount: number;
      readonly comparedBytes: number;
      readonly allInputNativePolyphaseWeightAndBiasSourcesUnchanged: boolean;
      readonly passed: boolean;
    };
    readonly passed: boolean;
  };
  readonly immutableAfterTiming: {
    readonly sourceCount: number;
    readonly comparedBytes: number;
    readonly allInputNativePolyphaseWeightAndBiasSourcesUnchanged: boolean;
    readonly passed: boolean;
  };
  readonly timing: {
    readonly timingRounds: number;
    readonly samplesPerArmPerProbe: number;
    readonly exactRangeCount: number;
    readonly weighted: {
      readonly A: number;
      readonly B: number;
      readonly speedup: number;
      readonly threshold: number;
    };
    readonly operations: readonly {
      readonly label: string;
      readonly exactRangeCount: number;
      readonly A: number;
      readonly B: number;
      readonly bFaster: boolean;
      readonly speedup: number;
    }[];
    readonly positions: readonly {
      readonly position: number;
      readonly A: number;
      readonly B: number;
      readonly bFaster: boolean;
      readonly speedup: number;
    }[];
    readonly everyOperationBFaster: boolean;
    readonly bothTimingPositionsBFaster: boolean;
    readonly strata: readonly ReceiptTimingStratum[];
    readonly passed: boolean;
    readonly decision: string;
    readonly timingStartedAtEpochMilliseconds: number;
    readonly timingCompletedAtEpochMilliseconds: number;
    readonly dispatchSampleCount: number;
    readonly caveat: string;
  };
  readonly cleanup: {
    readonly completedAtEpochMilliseconds: number;
    readonly firstCall: {
      readonly createdBufferCount: number;
      readonly destroyedBufferCount: number;
      readonly liveBufferCount: number;
      readonly liveBytes: number;
      readonly maximumLiveBytes: number;
      readonly mapCount: number;
      readonly unmapCount: number;
      readonly activeMapCount: number;
      readonly mapsBalanced: boolean;
      readonly idempotent: boolean;
      readonly repeatedCall: boolean;
      readonly drainBeforeRelease: boolean;
      readonly deviceDestroyed: boolean;
    };
    readonly secondCall: {
      readonly liveBufferCount: number;
      readonly liveBytes: number;
      readonly idempotent: boolean;
      readonly repeatedCall: boolean;
    };
    readonly idempotent: boolean;
    readonly drainBeforeRelease: boolean;
    readonly balancedMaps: boolean;
    readonly zeroLiveResources: boolean;
    readonly deviceDestroyed: boolean;
  };
  readonly decision: {
    readonly disposition: string;
    readonly packageNativeLayerGateAuthorized: boolean;
    readonly converterAndPackageWorkAuthorized: boolean;
    readonly productionIntegrationAuthorized: boolean;
    readonly c300ProductionRunAuthorized: boolean;
    readonly longWindowRunAuthorized: boolean;
    readonly waveformOrListeningRunAuthorized: boolean;
    readonly m2250ProductRunAuthorized: boolean;
  };
}

interface ThermalObservation {
  readonly index: number;
  readonly epochMilliseconds: number;
  readonly level: number;
}

interface ResultView {
  readonly identity: {
    readonly currentCoreSourceSha256: string;
    readonly candidateCoreSourceSha256: string;
    readonly harnessSourceSha256: string;
    readonly harnessHtmlSha256: string;
    readonly preExecutionContractSha256: string;
    readonly candidateCoreContractSha256: string;
    readonly rawArtifactBytes: { readonly receipt: number; readonly thermal: number };
    readonly productionPackageLoaded: boolean;
    readonly productionRuntimeSelected: boolean;
    readonly productionIntegrationPerformed: boolean;
  };
  readonly protocol: {
    readonly authoritativeTimedRunCount: number;
    readonly dispatchSampleCount: number;
    readonly unchangedThermalRetryPerformed: boolean;
    readonly rejectedPagePreflights: {
      readonly count: number;
      readonly timedDispatchCount: number;
      readonly timingSampleCount: number;
      readonly benchmarkAttempts: number;
      readonly reasons: readonly {
        readonly count: number;
        readonly reason: string;
      }[];
    };
    readonly diagnosticLoggerSmokeAttempts: {
      readonly count: number;
      readonly pageLaunchCount: number;
      readonly timedDispatchCount: number;
    };
    readonly thermal: Readonly<Record<string, unknown>>;
  };
  readonly metrics: {
    readonly baseline: { readonly weightedC300Milliseconds: number };
    readonly candidate: {
      readonly weightedC300Milliseconds: number;
      readonly everyOperationFaster: boolean;
      readonly qualified: boolean;
      readonly productionIntegrationPerformed: boolean;
      readonly c300ProductionRunPerformed: boolean;
      readonly longWindowRunPerformed: boolean;
      readonly m2250ProductRunPerformed: boolean;
    };
    readonly delta: {
      readonly weightedSpeedup: number;
      readonly weightedSpeedupThreshold: number;
      readonly weightedSpeedupConditionPassed: boolean;
      readonly everyOperationFasterConditionPassed: boolean;
      readonly bothTimingPositionsFasterConditionPassed: boolean;
      readonly regressingOperations: readonly string[];
      readonly stopRuleFired: boolean;
    };
  };
  readonly correctness: {
    readonly primitive: {
      readonly inverseLayoutComparedU16Count: number;
      readonly inverseLayoutMismatchCount: number;
      readonly outputComparedU16Count: number;
      readonly outputMismatchCount: number;
      readonly candidateVersusCurrentRawU16Exact: boolean;
    };
    readonly cleanup: {
      readonly createdBufferCount: number;
      readonly destroyedBufferCount: number;
      readonly liveBufferCount: number;
      readonly liveBytes: number;
      readonly maximumLiveBytes: number;
      readonly idempotent: boolean;
      readonly drainBeforeRelease: boolean;
      readonly deviceDestroyed: boolean;
    };
  };
}

const RESULT_URL = new URL(
  "../optimization/results/OPT-0022/result.json",
  import.meta.url,
);
const RECEIPT_URL = new URL(
  "../optimization/artifacts/OPT-0022/raw/conv-transpose-subgroup-polyphase-ab.json",
  import.meta.url,
);
const THERMAL_URL = new URL(
  "../optimization/artifacts/OPT-0022/raw/conv-transpose-subgroup-polyphase-thermal-authoritative-valid.jsonl",
  import.meta.url,
);
const RECORD_URL = new URL(
  "../optimization/experiments/OPT-0022-vae-conv-transpose-subgroup-polyphase.md",
  import.meta.url,
);
const LEDGER_URL = new URL("../optimization/LEDGER.md", import.meta.url);

const SOURCE_IDENTITIES = [
  [
    "../src/webgpu/kernels/vae-conv-transpose1d-fp16.ts",
    "cbcb9bcd5f856ce1c9e10aabca0ec0f95651c03d2c45b8076de3ba5022c6c3e2",
  ],
  [
    "../src/webgpu/kernels/vae-conv-transpose1d-fp16-subgroup.ts",
    "b3a02e29419021d78f669b7ed0333b80c8e5739ed46d9d9645f814d971a9edfa",
  ],
  [
    "browser/opt-0022-vae-conv-transpose1d-subgroup-polyphase-ab.ts",
    "4db0643edc6449087d6b28b5ee6dba3a3b0b9caabc2b4f8eb8d7a44bb27cadb8",
  ],
  [
    "browser/opt-0022-vae-conv-transpose1d-subgroup-polyphase-ab.html",
    "b97ca849b7a2613f3e345cef49bfd50456a25931aab32cf8454adcd5a383d1dc",
  ],
  [
    "opt-0022-vae-conv-transpose1d-subgroup-polyphase-contract.test.ts",
    "64e3af52b7d50f7b6adff079a7f807382c1501417564f079fcb7287e0ecd3f21",
  ],
  [
    "vae-conv-transpose1d-fp16-subgroup.test.ts",
    "19867f4400308a3ab8c321bdd609ed71e1af17421685532e6a21e514390234f6",
  ],
] as const;

const sha256 = (bytes: Uint8Array | string): string =>
  createHash("sha256").update(bytes).digest("hex");

describe("OPT-0022 optimization result", () => {
  it("commits canonical exact-but-negative evidence and every stop boundary", async () => {
    const text = await readFile(RESULT_URL, "utf8");
    const committed = parseOptimizationResultJson(text);
    const view = committed as unknown as ResultView;

    expect(committed).toMatchObject({
      experimentId: "OPT-0022",
      riskClass: "exact",
      baselineCommit: "73a8a85334226a2f2eb888960796ade8875ea6ad",
      candidateCommit: "8345fce46c92afa109c1b38fa45d15df31b94de5",
      evidence: { conclusion: "negative" },
      disposition: { state: "abandoned" },
      identity: {
        browserVersion:
          "Google Chrome 151.0.7922.138; reduced user agent 151.0.0.0",
        benchmarkHarnessCommit:
          "8345fce46c92afa109c1b38fa45d15df31b94de5",
        productionPackageLoaded: false,
        productionRuntimeSelected: false,
        productionIntegrationPerformed: false,
      },
      correctness: {
        passed: true,
        listeningRequired: false,
        listeningDecision: null,
      },
    });
    expect(committed.artifacts).toEqual([
      {
        location:
          "optimization/artifacts/OPT-0022/raw/conv-transpose-subgroup-polyphase-ab.json",
        sha256:
          "6806deefda170661542577a4fe55d978ecb55bc957ced88cb97555c0e2e7bbfa",
      },
      {
        location:
          "optimization/artifacts/OPT-0022/raw/conv-transpose-subgroup-polyphase-thermal-authoritative-valid.jsonl",
        sha256:
          "dda409ff5e63bc65b62b7d65f36a7f69f0256a0afc957263d000e34271bee949",
      },
    ]);
    expect(view.correctness.primitive).toMatchObject({
      inverseLayoutComparedU16Count: 49_610_752,
      inverseLayoutMismatchCount: 0,
      outputComparedU16Count: 8_404_992,
      outputMismatchCount: 0,
      candidateVersusCurrentRawU16Exact: true,
    });
    expect(view.metrics.delta).toEqual({
      bothTimingPositionsFasterConditionPassed: true,
      everyOperationFasterConditionPassed: false,
      observed180SecondProductSpeedup: null,
      observedIntegratedC300Speedup: null,
      observedLongWindowSpeedup: null,
      regressingOperations: ["block-2-conv-t1"],
      stopRuleFired: true,
      under60SecondProjection: null,
      weightedPrimitiveSavingMilliseconds: 956.800001680851,
      weightedSpeedup: 1.2605735458036402,
      weightedSpeedupConditionPassed: false,
      weightedSpeedupThreshold: 1.3398349037268882,
    });
    expect(view.metrics.candidate).toMatchObject({
      qualified: false,
      everyOperationFaster: false,
      productionIntegrationPerformed: false,
      c300ProductionRunPerformed: false,
      longWindowRunPerformed: false,
      m2250ProductRunPerformed: false,
    });
    expect(view.protocol).toMatchObject({
      authoritativeTimedRunCount: 1,
      dispatchSampleCount: 180,
      unchangedThermalRetryPerformed: false,
      rejectedPagePreflights: {
        count: 3,
        timedDispatchCount: 0,
        timingSampleCount: 0,
        benchmarkAttempts: 0,
      },
      diagnosticLoggerSmokeAttempts: {
        count: 2,
        pageLaunchCount: 0,
        timedDispatchCount: 0,
      },
    });
    expect(view.protocol.rejectedPagePreflights.reasons).toEqual([
      {
        count: 1,
        reason:
          "stale launch delay of approximately 11 seconds exceeded the freshness cap",
      },
      {
        count: 2,
        reason:
          "the original shell thermal loop drifted approximately 48 ms per poll, so 31 observations spanned approximately 31.47 seconds and failed observation-count/duration consistency because 32 observations were required",
      },
    ]);
    expect(`${stringifyOptimizationResult(committed)}\n`).toBe(text);
  });

  it("recomputes all 180 timings and authenticates raw-U16 exactness", async () => {
    const [resultText, receiptBytes] = await Promise.all([
      readFile(RESULT_URL, "utf8"),
      readFile(RECEIPT_URL),
    ]);
    expect(receiptBytes.byteLength).toBe(68_022);
    expect(sha256(receiptBytes)).toBe(
      "6806deefda170661542577a4fe55d978ecb55bc957ced88cb97555c0e2e7bbfa",
    );
    const result = parseOptimizationResultJson(resultText) as unknown as ResultView;
    const receipt = JSON.parse(receiptBytes.toString("utf8")) as Opt0022Receipt;

    expect(receipt).toMatchObject({
      schema: "ace-opt-0022-vae-conv-transpose-subgroup-polyphase-ab-v1",
      experimentId: "OPT-0022",
      status: "passed",
      classification: "bounded-exact-primitive-decision-gate-not-integrated",
      identity: {
        registrationCommit: "88a1bc1f3ffb06a6ca714437ba8616c11ed212f2",
        integratedCurrentCommit: "36608b857827b2b1d31ac91bf5cca9639fb0b9ed",
      },
      correctness: {
        operationCount: 5,
        selectedProbeCount: 15,
        exactGraphRangeCount: 378,
        inverseLayoutComparedU16Count: 49_610_752,
        inverseLayoutMismatchCount: 0,
        outputExecutionCount: 60,
        outputExactComparisonCount: 30,
        outputComparedU16Count: 8_404_992,
        outputMismatchCount: 0,
        deterministicRerunHashes: true,
        allPhasesCoveredInEveryProbe: true,
        bothOneTapBoundariesCoveredPerOperation: true,
        qNaNPrefillCompleteWrites: true,
        guardsAndAdjacentCanariesUntouched: true,
        uncapturedGpuErrorCount: 0,
        passed: true,
      },
      timing: {
        timingRounds: 6,
        samplesPerArmPerProbe: 6,
        exactRangeCount: 378,
        everyOperationBFaster: false,
        bothTimingPositionsBFaster: true,
        passed: false,
        decision: "negative-stop-primitive-gate",
        dispatchSampleCount: 180,
      },
      decision: {
        disposition: "negative-stop-primitive-gate",
        packageNativeLayerGateAuthorized: false,
        converterAndPackageWorkAuthorized: false,
        productionIntegrationAuthorized: false,
        c300ProductionRunAuthorized: false,
        longWindowRunAuthorized: false,
        waveformOrListeningRunAuthorized: false,
        m2250ProductRunAuthorized: false,
      },
    });

    expect(receipt.layout).toMatchObject({
      layoutId: "conv-transpose1d-phase-tap-input-output-f16-v1",
      operationCount: 5,
      comparedU16Count: 49_610_752,
      mismatchCount: 0,
      residentByteIncrease: 0,
    });
    expect(receipt.layout.cases).toHaveLength(5);
    expect(receipt.layout.cases.reduce(
      (sum, entry) => sum + entry.comparedU16Count,
      0,
    )).toBe(receipt.layout.comparedU16Count);
    expect(receipt.layout.cases.every((entry) =>
      entry.mismatchCount === 0 && entry.rawU16Exact &&
      entry.nativeBytes === entry.polyphaseBytes &&
      entry.residentByteIncrease === 0
    )).toBe(true);

    expect(receipt.correctness.cases).toHaveLength(5);
    expect(receipt.correctness.cases.reduce(
      (sum, entry) => sum + entry.comparedU16Count,
      0,
    )).toBe(receipt.correctness.outputComparedU16Count);
    expect(receipt.correctness.cases.every((entry) =>
      entry.selectedProbeCount === 3 && entry.ranges.length === 3 &&
      entry.correctnessExecutionsPerArmPerProbe === 2 &&
      entry.rawU16MismatchCount === 0 && entry.deterministicRerun &&
      entry.allPhasesCoveredInEveryProbe &&
      entry.firstAndLastOneTapBoundariesCovered &&
      entry.qNaNPrefillCompleteWrites && entry.completeSelectedWrites &&
      entry.guardsAndAdjacentCanariesUntouched &&
      entry.ranges.every((range) =>
        range.rawU16MismatchCount === 0 && range.deterministicRerun &&
        range.allPhasesCovered && /^[0-9a-f]{64}$/.test(range.outputSha256)
      )
    )).toBe(true);
    expect(receipt.correctness.immutableAfterCorrectness).toEqual({
      phase: "post-correctness",
      sourceCount: 20,
      comparedBytes: 334_843_904,
      allInputNativePolyphaseWeightAndBiasSourcesUnchanged: true,
      cases: expect.any(Array),
      passed: true,
    });
    expect(receipt.immutableAfterTiming).toMatchObject({
      sourceCount: 20,
      comparedBytes: 334_843_904,
      allInputNativePolyphaseWeightAndBiasSourcesUnchanged: true,
      passed: true,
    });

    const timingInput = receipt.timing.strata.map((entry) => ({
      operationLabel: entry.operationLabel,
      stratum: entry.stratum,
      weight: entry.weight,
      rounds: entry.rounds.map((round) => ({
        roundIndex: round.roundIndex,
        order: round.order,
        aMilliseconds: round.A,
        bMilliseconds: round.B,
      })),
    }));
    const recomputed = summarizeOpt0022Timing(timingInput);
    const {
      timingStartedAtEpochMilliseconds: _timingStarted,
      timingCompletedAtEpochMilliseconds: _timingCompleted,
      dispatchSampleCount: _dispatchSampleCount,
      caveat: _caveat,
      ...receiptSummary
    } = receipt.timing;
    expect(recomputed).toEqual(receiptSummary);
    expect(buildOpt0022TimingOrders()).toHaveLength(90);
    expect(receipt.timing.dispatchSampleCount).toBe(
      buildOpt0022TimingOrders().length * 2,
    );
    expect(receipt.timing.weighted).toEqual({
      A: 4628.699997246265,
      B: 3671.8999955654144,
      speedup: 1.2605735458036402,
      threshold: 1.3398349037268882,
    });
    expect(receipt.timing.operations.find((entry) =>
      entry.label === "block-2-conv-t1"
    )).toEqual({
      label: "block-2-conv-t1",
      exactRangeCount: 81,
      A: 656.5999971032143,
      B: 662.6499999761581,
      bFaster: false,
      speedup: 0.9908699873641266,
    });
    expect(receipt.timing.weighted.speedup).toBeLessThan(
      receipt.timing.weighted.threshold,
    );
    expect(result.metrics.baseline.weightedC300Milliseconds).toBe(
      receipt.timing.weighted.A,
    );
    expect(result.metrics.candidate.weightedC300Milliseconds).toBe(
      receipt.timing.weighted.B,
    );
  });

  it("recomputes source, shader, thermal, cleanup, and governance identity", {
    timeout: 120_000,
  }, async () => {
    const [resultText, receiptBytes, thermalBytes, record, ledger] =
      await Promise.all([
        readFile(RESULT_URL, "utf8"),
        readFile(RECEIPT_URL),
        readFile(THERMAL_URL),
        readFile(RECORD_URL, "utf8"),
        readFile(LEDGER_URL, "utf8"),
      ]);
    const result = parseOptimizationResultJson(resultText) as unknown as ResultView;
    const receipt = JSON.parse(receiptBytes.toString("utf8")) as Opt0022Receipt;

    expect(thermalBytes.byteLength).toBe(6_328);
    expect(sha256(thermalBytes)).toBe(
      "dda409ff5e63bc65b62b7d65f36a7f69f0256a0afc957263d000e34271bee949",
    );
    expect(result.identity.rawArtifactBytes).toEqual({
      receipt: 68_022,
      thermal: 6_328,
    });
    for (const [path, expected] of SOURCE_IDENTITIES) {
      expect(sha256(await readFile(new URL(path, import.meta.url)))).toBe(expected);
    }
    expect(result.identity).toMatchObject({
      currentCoreSourceSha256: SOURCE_IDENTITIES[0][1],
      candidateCoreSourceSha256: SOURCE_IDENTITIES[1][1],
      harnessSourceSha256: SOURCE_IDENTITIES[2][1],
      harnessHtmlSha256: SOURCE_IDENTITIES[3][1],
      preExecutionContractSha256: SOURCE_IDENTITIES[4][1],
      candidateCoreContractSha256: SOURCE_IDENTITIES[5][1],
    });

    const generated = buildOpt0022C300Topology().map((operation) => ({
      label: operation.label,
      A: sha256(aceFp16VaeCongruentConvTranspose1dWgsl(operation.shape)),
      B: sha256(aceOpt0022VaeConvTranspose1dSubgroupWgsl(operation.shape)),
    }));
    expect(generated).toEqual(receipt.identity.generatedShaders);
    expect(receipt.identity.currentCore).toMatchObject({
      bytes: 45_075,
      sha256: SOURCE_IDENTITIES[0][1],
      kernelId: "ace-vae-fp16-congruent-two-tap-conv-transpose1d-v1",
      weightLayout: "native-output-kernel-input-f16",
    });
    expect(receipt.identity.candidateCore).toMatchObject({
      bytes: 28_459,
      sha256: SOURCE_IDENTITIES[1][1],
      kernelId:
        "ace-vae-fp16-fixed32-subgroup-polyphase-conv-transpose1d-v1",
      weightLayout: "conv-transpose1d-phase-tap-input-output-f16-v1",
    });
    expect(receipt.environment).toMatchObject({
      userAgent:
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/151.0.0.0 Safari/537.36",
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
        maxStorageBufferBindingSize: 147_456_000,
        maxComputeWorkgroupStorageSize: 16_384,
        maxComputeInvocationsPerWorkgroup: 256,
      },
    });

    const thermal = thermalBytes.toString("utf8").trim().split("\n")
      .map((line) => JSON.parse(line) as ThermalObservation);
    expect(thermal).toHaveLength(111);
    expect(thermal.every((entry, index) =>
      entry.index === index && entry.level === 0 &&
      (index === 0 || entry.epochMilliseconds >
        thermal[index - 1]!.epochMilliseconds)
    )).toBe(true);
    const totalMaximumGap = maximumGap(thermal);
    expect(totalMaximumGap).toBe(1_004);
    const gate = thermal.filter((entry) =>
      entry.epochMilliseconds >=
        receipt.protocol.thermal.startedAtEpochMilliseconds &&
      entry.epochMilliseconds <=
        receipt.protocol.thermal.completedAtEpochMilliseconds
    );
    expect(gate).toHaveLength(31);
    expect(gate[0]).toEqual({
      index: 24,
      epochMilliseconds: 1786780846276,
      level: 0,
    });
    expect(gate.at(-1)).toEqual({
      index: 54,
      epochMilliseconds: 1786780876277,
      level: 0,
    });
    expect(maximumGap(gate)).toBe(1_003);
    expect(receipt.protocol.thermal).toEqual({
      source: "notifyutil-com.apple.system.thermalpressurelevel",
      startedAtEpochMilliseconds: 1786780846276,
      completedAtEpochMilliseconds: 1786780876277,
      durationMilliseconds: 30_001,
      observationCount: 31,
      pollMilliseconds: 1_000,
      maximumPollGapMilliseconds: 1_003,
      nonNominalObservationCount: 0,
      launchDelayMilliseconds: 998,
    });
    expect(receipt.preparation.completedAtEpochMilliseconds).toBeLessThanOrEqual(
      receipt.protocol.thermal.startedAtEpochMilliseconds,
    );
    expect(receipt.timing.timingStartedAtEpochMilliseconds -
      receipt.protocol.thermal.completedAtEpochMilliseconds).toBe(998);
    expect(receipt.timing.timingCompletedAtEpochMilliseconds).toBeLessThanOrEqual(
      receipt.cleanup.completedAtEpochMilliseconds,
    );
    expect(thermal.at(-1)!.epochMilliseconds).toBeGreaterThanOrEqual(
      receipt.protocol.thermalTraceRequiredThroughEpochMilliseconds,
    );
    expect(thermal.at(-1)!.epochMilliseconds -
      receipt.cleanup.completedAtEpochMilliseconds).toBe(52_710);
    expect(result.protocol.thermal).toEqual({
      cleanupCompletedAtEpochMilliseconds:
        receipt.cleanup.completedAtEpochMilliseconds,
      completedAtEpochMilliseconds:
        receipt.protocol.thermal.completedAtEpochMilliseconds,
      durationMilliseconds: receipt.protocol.thermal.durationMilliseconds,
      launchDelayMilliseconds: receipt.protocol.thermal.launchDelayMilliseconds,
      maximumPollGapMilliseconds:
        receipt.protocol.thermal.maximumPollGapMilliseconds,
      nonNominalObservationCount:
        receipt.protocol.thermal.nonNominalObservationCount,
      observationCount: receipt.protocol.thermal.observationCount,
      postCleanupCoverageMilliseconds: 52_710,
      source: receipt.protocol.thermal.source,
      startedAtEpochMilliseconds:
        receipt.protocol.thermal.startedAtEpochMilliseconds,
      totalTraceLastEpochMilliseconds: thermal.at(-1)!.epochMilliseconds,
      totalTraceMaximumPollGapMilliseconds: totalMaximumGap,
      totalTraceObservationCount: thermal.length,
    });

    expect(receipt.cleanup).toMatchObject({
      firstCall: {
        createdBufferCount: 110,
        destroyedBufferCount: 110,
        liveBufferCount: 0,
        liveBytes: 0,
        maximumLiveBytes: 987_542_272,
        mapCount: 100,
        unmapCount: 100,
        activeMapCount: 0,
        mapsBalanced: true,
        idempotent: true,
        repeatedCall: false,
        drainBeforeRelease: true,
        deviceDestroyed: true,
      },
      secondCall: {
        liveBufferCount: 0,
        liveBytes: 0,
        idempotent: true,
        repeatedCall: true,
      },
      idempotent: true,
      drainBeforeRelease: true,
      balancedMaps: true,
      zeroLiveResources: true,
      deviceDestroyed: true,
    });
    expect(result.correctness.cleanup).toMatchObject({
      createdBufferCount: 110,
      destroyedBufferCount: 110,
      liveBufferCount: 0,
      liveBytes: 0,
      maximumLiveBytes: 987_542_272,
      idempotent: true,
      drainBeforeRelease: true,
      deviceDestroyed: true,
    });

    expect(record).toContain("- Evidence: `negative`");
    expect(record).toContain("- Disposition: `abandoned`");
    expect(record).toContain("Exactly three page preflights were rejected");
    expect(record).toContain("There was no performance rerun");
    const row = ledger.split("\n").find((line) =>
      line.startsWith("| OPT-0022 |")
    );
    expect(row).toContain("| negative | abandoned |");
    expect(row).toContain("[result](results/OPT-0022/result.json)");
    expect(row).toContain("not integrated");
  });
});

function maximumGap(observations: readonly ThermalObservation[]): number {
  return Math.max(...observations.slice(1).map((entry, index) =>
    entry.epochMilliseconds - observations[index]!.epochMilliseconds
  ));
}
