import { describe, expect, it } from "vitest";

import {
  buildOpt0081MlpPanelSpecs,
  buildOpt0081MlpTimingRounds,
  classifyOpt0081MlpDisposition,
  numberToFloat16Bits,
  parseOpt0081MlpThermalCompletion,
  parseOpt0081MlpThermalLaunchGate,
  summarizeOpt0081MlpTiming,
  type Opt0081MlpArm,
  type Opt0081MlpPanelId,
  type Opt0081MlpTimestampSample,
} from "./browser/opt-0081-dit-f16-actual-mlp-chain.js";
import browserSource from
  "./browser/opt-0081-dit-f16-actual-mlp-chain.ts?raw";
import browserHtml from
  "./browser/opt-0081-dit-f16-actual-mlp-chain.html?raw";
import {
  planAceOpt0009DenseGemm,
} from "../src/webgpu/kernels/dit-dense-fp16.js";
import {
  planAceOpt0081DenseF16Input,
} from "../src/webgpu/kernels/dit-dense-f16-input.js";
import {
  planAceOpt0081DenseF16InputWeightMulticast,
} from
  "../src/webgpu/kernels/dit-dense-f16-input-weight-multicast.js";

const PANEL_IDS = ["gate", "up", "down", "chain"] as const;
const ARM_ORDERS = [
  "ABC", "CBA", "BCA", "ACB", "CAB", "BAC", "ABC", "CBA",
] as const;

describe("OPT-0081 typed-F16 actual MLP-chain browser gate", () => {
  it("freezes the four actual panels and balanced eight-round order", () => {
    expect(buildOpt0081MlpPanelSpecs().map(({ id, operation,
      denseDispatchCount, producerDispatchCount }) =>
      ({ id, operation, denseDispatchCount, producerDispatchCount })))
      .toEqual([
        { id: "gate", operation: "mlpModulated→gate",
          denseDispatchCount: 1, producerDispatchCount: 1 },
        { id: "up", operation: "mlpModulated→up",
          denseDispatchCount: 1, producerDispatchCount: 1 },
        { id: "down", operation: "gatedActivation→down",
          denseDispatchCount: 1, producerDispatchCount: 1 },
        { id: "chain",
          operation:
            "mlpModulated→gate/up→SwiGLU→gatedActivation→down",
          denseDispatchCount: 3, producerDispatchCount: 2 },
      ]);
    const rounds = buildOpt0081MlpTimingRounds();
    expect(rounds).toHaveLength(8);
    expect(rounds.map(({ armOrder }) => armOrder.join(""))).toEqual(ARM_ORDERS);
    for (const round of rounds) {
      expect(round.panelOrder).toEqual(PANEL_IDS);
    }
    const pairPositions: Record<"AB" | "AC" | "BC", [number, number]> = {
      AB: [0, 0], AC: [0, 0], BC: [0, 0],
    };
    for (const { armOrder } of rounds) {
      for (const pair of Object.keys(pairPositions) as
        (keyof typeof pairPositions)[]) {
        const left = pair[0] as Opt0081MlpArm;
        const right = pair[1] as Opt0081MlpArm;
        const position = armOrder.indexOf(left) <
          armOrder.indexOf(right) ? 0 : 1;
        pairPositions[pair][position] =
          pairPositions[pair][position]! + 1;
      }
    }
    expect(pairPositions).toEqual({ AB: [4, 4], AC: [4, 4], BC: [4, 4] });
  });

  it("uses the exact two M2250 dense plans and reconciles payload ownership", () => {
    const shapes = [
      { rows: 2_250, inner: 2_048, columns: 6_144 },
      { rows: 2_250, inner: 6_144, columns: 2_048 },
    ] as const;
    for (const shape of shapes) {
      const A = planAceOpt0009DenseGemm(shape);
      const B = planAceOpt0081DenseF16Input(shape);
      const C = planAceOpt0081DenseF16InputWeightMulticast(shape);
      expect([A.rows, A.inner, A.columns]).toEqual([
        shape.rows, shape.inner, shape.columns,
      ]);
      expect(B.outputRanges).toEqual(A.outputRanges);
      expect(C.outputRanges).toEqual(A.outputRanges);
      expect([A.workgroupSize, B.workgroupSize, C.workgroupSize])
        .toEqual([128, 128, 256]);
      expect(C.workgroupStorageBytes).toBe(16_384);
      expect(C.subgroupsPerWorkgroup).toBe(8);
    }
    const immutableInputs = 2_250 * 2_048 * 4 + 2 * 2_048 * 4;
    const weights = 3 * 2_048 * 6_144 * 2;
    const scratch =
      2_250 * 2_048 * 4 +
      2_250 * 2_048 * 2 +
      2 * 2_250 * 6_144 * 4 +
      2_250 * 6_144 * 4 +
      2_250 * 6_144 * 2 +
      2_250 * 2_048 * 4;
    expect({ immutableInputs, weights, scratch,
      total: immutableInputs + weights + scratch }).toEqual({
        immutableInputs: 18_448_384,
        weights: 75_497_472,
        scratch: 239_616_000,
        total: 333_561_856,
      });
  });

  it("passes only B while retaining a non-selectable C diagnostic result", () => {
    const result = summarizeOpt0081MlpTiming(timingInput({
      A: { gate: 40, up: 40, down: 40, chain: 100 },
      B: { gate: 28, up: 28, down: 28, chain: 70 },
      C: { gate: 10, up: 10, down: 10, chain: 35 },
    }, {
      A: { gate: 42, up: 42, down: 42, chain: 105 },
      B: { gate: 30, up: 30, down: 30, chain: 75 },
      C: { gate: 11, up: 11, down: 11, chain: 38 },
    }));
    expect(result).toMatchObject({
      passed: true,
      selectedArm: "B",
      bStandalonePassed: true,
      cDiagnosticPassed: true,
      cPrimitiveQualified: false,
      cSelectableUnderOpt0081: false,
      gates: {
        bStandalone: {
          requiredCompleteChainSavingMilliseconds: 10.4167,
          passed: true,
        },
        cDiagnostic: {
          requiredCompleteChainSavingMilliseconds: 15.625,
          primitiveQualified: false,
          selectableUnderOpt0081: false,
          passed: true,
        },
      },
    });
    expect(classifyOpt0081MlpDisposition(validReceipt(result, true))).toBe(
      "positive-B-actual-mlp-representative-layer-follow-up-authorized",
    );
  });

  it("enforces every B panel statistic/pair and both complete-chain clocks", () => {
    const consistentMiss = summarizeOpt0081MlpTiming(timingInput({
      A: { gate: 40, up: 40, down: 40, chain: 100 },
      B: { gate: 45, up: 45, down: 45, chain: 105 },
      C: { gate: 10, up: 10, down: 10, chain: 35 },
    }, {
      A: { gate: 42, up: 42, down: 42, chain: 105 },
      B: { gate: 47, up: 47, down: 47, chain: 110 },
      C: { gate: 11, up: 11, down: 11, chain: 38 },
    }));
    expect(consistentMiss).toMatchObject({
      passed: false, selectedArm: null, bStandalonePassed: false,
      cSelectableUnderOpt0081: false,
    });
    expect(classifyOpt0081MlpDisposition(
      validReceipt(consistentMiss, false),
    )).toBe("negative-stop-B-actual-mlp-materiality-gate-not-met");

    const partialPairs = timingInput({
      A: { gate: 40, up: 40, down: 40, chain: 100 },
      B: { gate: 28, up: 28, down: 28, chain: 70 },
      C: { gate: 10, up: 10, down: 10, chain: 35 },
    }, {
      A: { gate: 42, up: 42, down: 42, chain: 105 },
      B: { gate: 30, up: 30, down: 30, chain: 75 },
      C: { gate: 11, up: 11, down: 11, chain: 38 },
    });
    partialPairs.panels.gate.B[0] = sample(41, 43);
    partialPairs.panels.gate.B[1] = sample(41, 43);
    const partialResult = summarizeOpt0081MlpTiming(partialPairs);
    expect(partialResult["bStandalonePassed"]).toBe(false);
    expect(classifyOpt0081MlpDisposition(
      validReceipt(partialResult, false),
    )).toBe("inconclusive-directional-or-wall-gpu-evidence");
  });

  it("distinguishes an observed raw mismatch from invalid evidence", () => {
    const mismatchCase = (differingWordCount: number) => Object.freeze({
      comparisons: Object.freeze({
        aa: Object.freeze({ differingWordCount }),
      }),
      everyRegisteredComparisonRawExact: differingWordCount === 0,
      finiteCompleteQNaNOverwriteGuardsAndTail: true,
    });
    const mismatchCorrectness = Object.freeze({
      passed: false, completedBeforeReady: true,
      uncapturedGpuErrorCount: 0, deviceLossCount: 0,
      cases: Object.freeze([
        mismatchCase(1), mismatchCase(0), mismatchCase(0),
        mismatchCase(0), mismatchCase(0),
      ]),
    });
    expect(classifyOpt0081MlpDisposition(Object.freeze({
      inPagePassed: false, correctness: mismatchCorrectness,
      cleanup: Object.freeze({ passed: true }),
    }))).toBe("negative-stop-observed-raw-bit-correctness-mismatch");
    expect(classifyOpt0081MlpDisposition(Object.freeze({
      inPagePassed: false,
      correctness: Object.freeze({ passed: false }),
      cleanup: Object.freeze({ passed: true }),
    }))).toBe(
      "inconclusive-invalid-correctness-or-lifecycle-evidence",
    );
    expect(classifyOpt0081MlpDisposition(Object.freeze({
      inPagePassed: false,
      correctness: Object.freeze({ passed: true }),
      cleanup: Object.freeze({ passed: false }),
    }))).toBe(
      "inconclusive-invalid-correctness-or-lifecycle-evidence",
    );
  });

  it("requires the absolute thermal launch and through-cleanup handshake", () => {
    const ready = 1_000_000;
    const launched = 1_036_000;
    const launch = parseOpt0081MlpThermalLaunchGate(new URLSearchParams({
      thermalSource: "notifyutil-com.apple.system.thermalpressurelevel",
      thermalCommand:
        "notifyutil -g com.apple.system.thermalpressurelevel",
      thermalTraceStartedAtEpochMilliseconds: "999000",
      thermalGateStartedAtEpochMilliseconds: "1005000",
      thermalGateCompletedAtEpochMilliseconds: "1035000",
      thermalGateObservations: "31",
      thermalPollMilliseconds: "1000",
      thermalGateMaximumPollGapMilliseconds: "1004",
      thermalGateNonNominalObservations: "0",
      thermalGateMissingObservations: "0",
    }), ready, launched);
    expect(launch).toMatchObject({
      observationCount: 31, maximumPollGapMilliseconds: 1004,
      launchDelayMilliseconds: 1000,
    });
    const completion = parseOpt0081MlpThermalCompletion(
      new URLSearchParams({
        thermalTraceSchema:
          "jsonl-index-target-epoch-observed-epoch-keyed-notifyutil-v1",
        thermalTraceSha256: "a".repeat(64),
        thermalTraceByteLength: "4096",
        thermalTraceCompletedAtEpochMilliseconds: "1045000",
        thermalTraceObservations: "47",
        thermalTraceMaximumPollGapMilliseconds: "1004",
        thermalTraceNonNominalObservations: "0",
        thermalTraceMissingObservations: "0",
        thermalTraceInitialLevel: "0",
        thermalTraceFinalLevel: "0",
      }),
      launch,
      1_044_000,
    );
    expect(completion).toMatchObject({ coversCleanup: true,
      initialLevel: 0, finalLevel: 0 });
    const stale = new URLSearchParams({
      thermalSource: "notifyutil-com.apple.system.thermalpressurelevel",
      thermalCommand:
        "notifyutil -g com.apple.system.thermalpressurelevel",
      thermalTraceStartedAtEpochMilliseconds: "999000",
      thermalGateStartedAtEpochMilliseconds: "1005000",
      thermalGateCompletedAtEpochMilliseconds: "1035000",
      thermalGateObservations: "31",
      thermalPollMilliseconds: "1000",
      thermalGateMaximumPollGapMilliseconds: "1501",
      thermalGateNonNominalObservations: "0",
      thermalGateMissingObservations: "0",
    });
    expect(() => parseOpt0081MlpThermalLaunchGate(stale, ready, launched))
      .toThrow(/thermal launch gate failed/u);
  });

  it("binds real owners, staged raw checkpoints, timed producers, and lifecycle", () => {
    for (const token of [
      "AceOpt0009DenseGemmKernel",
      "AceOpt0081DenseF16InputKernel",
      "AceOpt0081DenseF16InputWeightMulticastKernel",
      "AceOpt0081F16DenseInputProducerKernel",
      "createMlpModulatedDispatch",
      "createGatedActivationDispatch",
      "createAdaLnDispatch",
      "createSwiGluDispatch",
      "GATE_UP_SHAPE",
      "DOWN_SHAPE",
      "mlpProducer, gate",
      "mlpProducer, up",
      "gatedProducer, down",
      "[mlpProducer, gate, up, gatedProducer, down]",
      "raw checkpoint ${checkpointIndex + 1}/5",
      '["A", "A", "B", "B", "C", "C"]',
      "candidateBoundaryComparedToIndependentCast: true",
      "denseU32ComparisonCount: 18",
      "maximumRetainedFullSnapshots: 3",
      "gpuConversionDispatchCount: 0",
      "actualM2250ProducersInsideEveryTimedPanel: true",
      "downConsumesImmediatelyPrecedingValidatedGateUpResults: true",
      "cPrimitiveQualified: false",
      "cSelectableUnderOpt0081: false",
      "armBPostDestroyRejected",
      "armCPostDestroyRejected",
      "producerPostDestroyRejected",
      "onePassOneCommandBufferOneSubmitOneDrainPerPanelSample: true",
      "oneTimestampQueryPairPerPanelSample: true",
      "606d1e29f56867bfda637c117b58778c634c4ee9",
      "0f13bcc486569819df7587349b8b1e049b924ccd",
      "436355ff16fb971d11a959e99e1550abc6186480",
    ]) expect(browserSource).toContain(token);
    expect(browserSource).not.toContain("AceCorrectnessRmsNormKernel");
    expect(browserSource).not.toContain("createGatedResidualDispatch");
    expect(browserSource).not.toContain("selectedArm: \"C\"");
    expect(browserHtml).toContain(
      'src="./opt-0081-dit-f16-actual-mlp-chain.ts"',
    );
    expect(browserHtml).toContain('id="thermal-gate"');
    expect(browserHtml).toContain('id="thermal-completion"');
  });

  it("keeps the independent cast exact at signed-zero and rounding edges", () => {
    expect(numberToFloat16Bits(0)).toBe(0);
    expect(numberToFloat16Bits(-0)).toBe(0x8000);
    expect(numberToFloat16Bits(1)).toBe(0x3c00);
    expect(numberToFloat16Bits(2 ** -24)).toBe(0x0001);
    expect(numberToFloat16Bits(1 + 2 ** -11)).toBe(0x3c00);
    expect(numberToFloat16Bits(1 + 2 ** -11 + 2 ** -23)).toBe(0x3c01);
    expect(numberToFloat16Bits(65_504)).toBe(0x7bff);
  });
});

type PanelValues = Readonly<Record<Opt0081MlpPanelId, number>>;
type ArmPanelValues = Readonly<Record<Opt0081MlpArm, PanelValues>>;

function timingInput(
  gpu: ArmPanelValues,
  wall: ArmPanelValues,
): { panels: Record<Opt0081MlpPanelId,
  Record<Opt0081MlpArm, Opt0081MlpTimestampSample[]>> } {
  const panels = {} as Record<Opt0081MlpPanelId,
    Record<Opt0081MlpArm, Opt0081MlpTimestampSample[]>>;
  for (const panel of PANEL_IDS) {
    panels[panel] = { A: [], B: [], C: [] };
    for (const arm of ["A", "B", "C"] as const) {
      panels[panel][arm] = Array.from({ length: 8 }, () =>
        sample(gpu[arm][panel], wall[arm][panel]));
    }
  }
  return { panels };
}

function sample(
  gpuMilliseconds: number,
  wallMilliseconds: number,
): Opt0081MlpTimestampSample {
  const gpuElapsedNanoseconds = Math.round(gpuMilliseconds * 1_000_000);
  return Object.freeze({
    submitAtPerformanceMilliseconds: 10,
    fenceAtPerformanceMilliseconds: 10 + wallMilliseconds,
    submitAtEpochMilliseconds: 1_000,
    fenceAtEpochMilliseconds: 1_000 + wallMilliseconds,
    wallMilliseconds,
    timestampBeginNanoseconds: "1000",
    timestampEndNanoseconds: String(1000 + gpuElapsedNanoseconds),
    gpuElapsedNanoseconds, gpuMilliseconds,
    gpuToWallRatio: gpuMilliseconds / wallMilliseconds,
    validMultiplyAdds: 1, scheduledMultiplyAdds: 1,
    validGpuTflops: 1, scheduledGpuTflops: 1,
    validWallTflops: 1, scheduledWallTflops: 1,
    commandBufferCount: 1, queueDrainCount: 1,
    timestampResolveCount: 1, timestampCopyCount: 1,
  });
}

function validReceipt(
  timing: Readonly<Record<string, unknown>>,
  inPagePassed: boolean,
): Readonly<Record<string, unknown>> {
  return Object.freeze({
    inPagePassed,
    timing,
    correctness: Object.freeze({ passed: true }),
    cleanup: Object.freeze({ passed: true }),
  });
}
