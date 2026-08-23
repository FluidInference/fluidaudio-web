/// <reference types="@webgpu/types" />
/// <reference types="vite/client" />

import armAKernelSource from
  "../../src/webgpu/kernels/dit-dense-fp16.ts?raw";
import armBKernelSource from
  "../../src/webgpu/kernels/dit-dense-f16-input.ts?raw";
import armCKernelSource from
  "../../src/webgpu/kernels/dit-dense-f16-input-weight-multicast.ts?raw";
import producerKernelSource from
  "../../src/webgpu/kernels/dit-f16-dense-input-producers.ts?raw";
import transformerProducerSource from
  "../../src/webgpu/kernels/transformer-plumbing.ts?raw";
import { ACE_DIT_DENSE_FP16_TILE_LAYOUT } from "../../src/model/manifest.js";
import {
  AceOpt0009DenseGemmKernel,
  aceOpt0009DenseGemmWgsl,
  planAceOpt0009DenseGemm,
} from "../../src/webgpu/kernels/dit-dense-fp16.js";
import {
  AceOpt0081DenseF16InputKernel,
  aceOpt0081DenseF16InputWgsl,
  planAceOpt0081DenseF16Input,
} from "../../src/webgpu/kernels/dit-dense-f16-input.js";
import {
  AceOpt0081DenseF16InputWeightMulticastKernel,
  aceOpt0081DenseF16InputWeightMulticastWgsl,
  planAceOpt0081DenseF16InputWeightMulticast,
} from "../../src/webgpu/kernels/dit-dense-f16-input-weight-multicast.js";
import {
  ACE_OPT_0081_DIT_F16_DENSE_INPUT_PRODUCER_KERNEL_SET_ID,
  ACE_OPT_0081_DIT_F16_DENSE_INPUT_STORAGE_PROFILE,
  AceOpt0081F16DenseInputProducerKernel,
} from "../../src/webgpu/kernels/dit-f16-dense-input-producers.js";
import { AceCorrectnessTransformerPlumbingKernel } from
  "../../src/webgpu/kernels/transformer-plumbing.js";
import type {
  AceGemmDispatch,
  AceGemmShape,
} from "../../src/webgpu/kernels/gemm.js";

declare global {
  interface Window {
    __ACE_OPT0081_MLP_RESULT__?: Readonly<Record<string, unknown>>;
  }
}

export type Opt0081MlpArm = "A" | "B" | "C";

export type Opt0081MlpPanelId = "gate" | "up" | "down" | "chain";

export interface Opt0081MlpPanelSpec {
  readonly id: Opt0081MlpPanelId;
  readonly ordinal: number;
  readonly operation: string;
  readonly denseDispatchCount: 1 | 3;
  readonly producerDispatchCount: 1 | 2;
}

export interface Opt0081MlpTimestampSample {
  readonly submitAtPerformanceMilliseconds: number;
  readonly fenceAtPerformanceMilliseconds: number;
  readonly submitAtEpochMilliseconds: number;
  readonly fenceAtEpochMilliseconds: number;
  readonly wallMilliseconds: number;
  readonly timestampBeginNanoseconds: string;
  readonly timestampEndNanoseconds: string;
  readonly gpuElapsedNanoseconds: number;
  readonly gpuMilliseconds: number;
  readonly gpuToWallRatio: number;
  readonly validMultiplyAdds: number;
  readonly scheduledMultiplyAdds: number;
  readonly validGpuTflops: number;
  readonly scheduledGpuTflops: number;
  readonly validWallTflops: number;
  readonly scheduledWallTflops: number;
  readonly commandBufferCount: 1;
  readonly queueDrainCount: 1;
  readonly timestampResolveCount: 1;
  readonly timestampCopyCount: 1;
}

export interface Opt0081MlpTimingInput {
  readonly panels: Readonly<Record<Opt0081MlpPanelId,
    Readonly<Record<Opt0081MlpArm,
      readonly Opt0081MlpTimestampSample[]>>>>;
}

export interface Opt0081MlpThermalLaunchGate {
  readonly source: "notifyutil-com.apple.system.thermalpressurelevel";
  readonly command: "notifyutil -g com.apple.system.thermalpressurelevel";
  readonly traceStartedAtEpochMilliseconds: number;
  readonly gateStartedAtEpochMilliseconds: number;
  readonly gateCompletedAtEpochMilliseconds: number;
  readonly observationCount: number;
  readonly pollMilliseconds: 1_000;
  readonly maximumPollGapMilliseconds: number;
  readonly nonNominalObservationCount: 0;
  readonly missingObservationCount: 0;
  readonly readyToGateDelayMilliseconds: number;
  readonly launchDelayMilliseconds: number;
}

export interface Opt0081MlpThermalCompletion {
  readonly schema:
    "jsonl-index-target-epoch-observed-epoch-keyed-notifyutil-v1";
  readonly sha256: string;
  readonly byteLength: number;
  readonly completedAtEpochMilliseconds: number;
  readonly observationCount: number;
  readonly maximumPollGapMilliseconds: number;
  readonly nonNominalObservationCount: 0;
  readonly missingObservationCount: 0;
  readonly initialLevel: 0;
  readonly finalLevel: 0;
  readonly coversCleanup: true;
}

interface GuardedTensor {
  readonly buffer: GPUBuffer;
  readonly binding: GPUBufferBinding;
  readonly elements: number;
  readonly bodyBytes: number;
  readonly totalBytes: number;
  readonly columns: number;
  readonly elementBytes: 2 | 4;
  readonly label: string;
}

interface U32Snapshot {
  readonly words: Uint32Array<ArrayBuffer>;
  readonly sha256: string;
  readonly nonFiniteCount: number;
  readonly qNaNPrefillCount: number;
  readonly positiveZeroCount: number;
  readonly negativeZeroCount: number;
  readonly prefixGuardIntact: boolean;
  readonly suffixGuardIntact: boolean;
  readonly adjacentBeforeIntact: boolean;
  readonly adjacentAfterIntact: boolean;
  readonly firstValidWritten: boolean;
  readonly lastValidWritten: boolean;
  readonly partialMTailWritten: boolean;
}

interface U16Snapshot {
  readonly words: Uint16Array<ArrayBuffer>;
  readonly sha256: string;
  readonly nonFiniteCount: number;
  readonly qNaNPrefillCount: number;
  readonly positiveZeroCount: number;
  readonly negativeZeroCount: number;
  readonly prefixGuardIntact: boolean;
  readonly suffixGuardIntact: boolean;
  readonly adjacentBeforeIntact: boolean;
  readonly adjacentAfterIntact: boolean;
  readonly firstValidWritten: boolean;
  readonly lastValidWritten: boolean;
  readonly partialMTailWritten: boolean;
}

interface EncodableDispatch {
  readonly label: string;
  encode(pass: GPUComputePassEncoder): void;
}

interface TimedPanelDispatch {
  readonly id: Opt0081MlpPanelId;
  readonly label: string;
  readonly dispatches: readonly EncodableDispatch[];
  readonly validMultiplyAdds: number;
  readonly scheduledMultiplyAdds: number;
}

interface PreparedArm {
  readonly arm: Opt0081MlpArm;
  readonly mlpProducer: EncodableDispatch;
  readonly gate: AceGemmDispatch;
  readonly up: AceGemmDispatch;
  readonly gatedProducer: EncodableDispatch;
  readonly down: AceGemmDispatch;
  readonly panels: Readonly<Record<Opt0081MlpPanelId, TimedPanelDispatch>>;
}

interface PreparedHarness {
  readonly adapter: GPUAdapter;
  readonly device: GPUDevice;
  readonly tracker: BufferTracker;
  readonly armAKernel: AceOpt0009DenseGemmKernel;
  readonly armBKernel: AceOpt0081DenseF16InputKernel;
  readonly armCKernel: AceOpt0081DenseF16InputWeightMulticastKernel;
  readonly currentTransformerProducer: AceCorrectnessTransformerPlumbingKernel;
  readonly f16Producer: AceOpt0081F16DenseInputProducerKernel;
  readonly querySet: GPUQuerySet;
  readonly queryResolve: GPUBuffer;
  readonly queryReadback: GPUBuffer;
  readonly arms: Readonly<Record<Opt0081MlpArm, PreparedArm>>;
  readonly firstCandidateActivation: GPUBufferBinding;
  readonly gateWeight: GPUBufferBinding;
  readonly downOutput: GPUBufferBinding;
  readonly correctness: Readonly<Record<string, unknown>>;
  readonly identity: Readonly<Record<string, unknown>>;
  readonly warmup: Readonly<Record<string, unknown>>;
  readonly readyAtEpochMilliseconds: number;
  readonly uncapturedErrors: readonly string[];
  readonly deviceLosses: readonly string[];
  cleanup(): Promise<Readonly<Record<string, unknown>>>;
}

class Opt0081MlpPreparationFailure extends Error {
  readonly originalName: string;
  readonly correctness: Readonly<Record<string, unknown>> | undefined;

  constructor(error: unknown,
    readonly setupCleanup: Readonly<Record<string, unknown>>) {
    super(error instanceof Error ? error.message : String(error));
    this.name = "Opt0081MlpPreparationFailure";
    this.originalName = error instanceof Error ? error.name : typeof error;
    this.correctness = error instanceof Opt0081MlpCorrectnessGateFailure
      ? error.correctness : undefined;
  }
}

class Opt0081MlpCorrectnessGateFailure extends Error {
  constructor(readonly correctness: Readonly<Record<string, unknown>>,
    message: string) {
    super(message);
    this.name = "Opt0081MlpCorrectnessGateFailure";
  }
}

const EXPERIMENT_ID = "OPT-0081" as const;
const RECEIPT_SCHEMA = "ace-opt-0081-f16-actual-mlp-chain-v1";
const REGISTRATION_COMMIT =
  "606d1e29f56867bfda637c117b58778c634c4ee9";
const B_ONLY_CORRECTION_COMMIT =
  "0f13bcc486569819df7587349b8b1e049b924ccd";
const HARNESS_IMPLEMENTATION_CHECKPOINT_COMMIT =
  "436355ff16fb971d11a959e99e1550abc6186480";
const PRIMITIVE_REGISTRATION_COMMIT =
  "70a5e4a29c5455ec00a4b757dcdf5cdcc70a5e91";
const PRIMITIVE_IMPLEMENTATION_COMMIT =
  "312d67024978a64b77d2563dd9386b4328f17d33";
const ROWS = 2_250;
const HIDDEN = 2_048;
const INTERMEDIATE = 6_144;
const STORAGE_GUARD_BYTES = 256;
const STORAGE_GUARD_U32 = 0xa55a_c33c;
const OUTPUT_PREFILL_QNAN_U32 = 0x7fc0_7855;
const STORAGE_GUARD_U16 = 0x55aa;
const OUTPUT_PREFILL_QNAN_U16 = 0x7e55;
const F16_CAST_SCRATCH = new ArrayBuffer(4);
const F16_CAST_FLOAT = new Float32Array(F16_CAST_SCRATCH);
const F16_CAST_WORD = new Uint32Array(F16_CAST_SCRATCH);
const TIMESTAMP_QUERY_BYTES = 16;
const REQUIRED_B_OVER_A_SAVING_MILLISECONDS = 10.4167;
const REQUIRED_C_DIAGNOSTIC_SAVING_MILLISECONDS = 15.625;
const REQUIRED_PAIR_WINS = 7;
const PROJECTION_MULTIPLIER = 24 * 8;
const THERMAL_SOURCE =
  "notifyutil-com.apple.system.thermalpressurelevel" as const;
const THERMAL_COMMAND =
  "notifyutil -g com.apple.system.thermalpressurelevel" as const;
const THERMAL_TRACE_SCHEMA =
  "jsonl-index-target-epoch-observed-epoch-keyed-notifyutil-v1" as const;
const THERMAL_POLL_MILLISECONDS = 1_000;
const MINIMUM_NOMINAL_MILLISECONDS = 30_000;
const MAXIMUM_THERMAL_GAP_MILLISECONDS = 1_500;
const MAXIMUM_GATE_TO_LAUNCH_MILLISECONDS = 5_000;
const GATE_UP_SHAPE = Object.freeze({
  rows: ROWS, inner: HIDDEN, columns: INTERMEDIATE,
});
const DOWN_SHAPE = Object.freeze({
  rows: ROWS, inner: INTERMEDIATE, columns: HIDDEN,
});

const PANEL_SPECS: readonly Opt0081MlpPanelSpec[] = Object.freeze([
  Object.freeze({ id: "gate", ordinal: 0,
    operation: "mlpModulated→gate", denseDispatchCount: 1,
    producerDispatchCount: 1 }),
  Object.freeze({ id: "up", ordinal: 1,
    operation: "mlpModulated→up", denseDispatchCount: 1,
    producerDispatchCount: 1 }),
  Object.freeze({ id: "down", ordinal: 2,
    operation: "gatedActivation→down", denseDispatchCount: 1,
    producerDispatchCount: 1 }),
  Object.freeze({ id: "chain", ordinal: 3,
    operation: "mlpModulated→gate/up→SwiGLU→gatedActivation→down",
    denseDispatchCount: 3, producerDispatchCount: 2 }),
] as const);

const ARM_ORDERS = Object.freeze([
  ["A", "B", "C"], ["C", "B", "A"], ["B", "C", "A"],
  ["A", "C", "B"], ["C", "A", "B"], ["B", "A", "C"],
  ["A", "B", "C"], ["C", "B", "A"],
] as const);

const TIMING_ROUNDS = Object.freeze(Array.from({ length: 8 }, (_, index) => {
  const armOrder: readonly Opt0081MlpArm[] = ARM_ORDERS[index]!;
  return Object.freeze({ roundIndex: index, armOrder,
    panelOrder: Object.freeze(PANEL_SPECS.map(({ id }) => id)) });
}));

export function buildOpt0081MlpPanelSpecs(): readonly Opt0081MlpPanelSpec[] {
  return PANEL_SPECS;
}

export function buildOpt0081MlpTimingRounds(): typeof TIMING_ROUNDS {
  return TIMING_ROUNDS;
}

export function summarizeOpt0081MlpTiming(
  input: Opt0081MlpTimingInput,
): Readonly<Record<string, unknown>> {
  const panels = PANEL_SPECS.map((spec) => {
    const samples = input.panels[spec.id];
    requireTimestampSamples(samples.A, `${spec.id} A`);
    requireTimestampSamples(samples.B, `${spec.id} B`);
    requireTimestampSamples(samples.C, `${spec.id} C`);
    const summarizeArm = (values: readonly Opt0081MlpTimestampSample[]) => {
      const gpu = values.map(({ gpuMilliseconds }) => gpuMilliseconds);
      const wall = values.map(({ wallMilliseconds }) => wallMilliseconds);
      return Object.freeze({ samples: values,
        meanGpuMilliseconds: mean(gpu),
        medianGpuMilliseconds: median(gpu),
        meanWallMilliseconds: mean(wall),
        medianWallMilliseconds: median(wall),
        minimumGpuMilliseconds: Math.min(...gpu),
        maximumGpuMilliseconds: Math.max(...gpu),
        minimumWallMilliseconds: Math.min(...wall),
        maximumWallMilliseconds: Math.max(...wall) });
    };
    const A = summarizeArm(samples.A);
    const B = summarizeArm(samples.B);
    const C = summarizeArm(samples.C);
    const faster = (candidate: typeof A, baseline: typeof A) => Object.freeze({
      meanGpuFaster:
        candidate.meanGpuMilliseconds < baseline.meanGpuMilliseconds,
      medianGpuFaster:
        candidate.medianGpuMilliseconds < baseline.medianGpuMilliseconds,
      meanWallFaster:
        candidate.meanWallMilliseconds < baseline.meanWallMilliseconds,
      medianWallFaster:
        candidate.medianWallMilliseconds < baseline.medianWallMilliseconds,
    });
    const gates = Object.freeze({ bOverA: faster(B, A),
      cOverA: faster(C, A), cOverB: faster(C, B) });
    return Object.freeze({ id: spec.id, ordinal: spec.ordinal,
      operation: spec.operation, denseDispatchCount: spec.denseDispatchCount,
      producerDispatchCount: spec.producerDispatchCount,
      arms: Object.freeze({ A, B, C }), gates,
      bOverAPassed: Object.values(gates.bOverA).every(Boolean),
      cOverAPassed: Object.values(gates.cOverA).every(Boolean),
      cOverBPassed: Object.values(gates.cOverB).every(Boolean) });
  });
  const comparison = (panel: typeof panels[number],
    candidate: "B" | "C", baseline: "A" | "B") => {
    const baselineArm = panel.arms[baseline];
    const candidateArm = panel.arms[candidate];
    const rounds = Array.from({ length: TIMING_ROUNDS.length },
      (_, roundIndex) => Object.freeze({
        baselineGpu: baselineArm.samples[roundIndex]!.gpuMilliseconds,
        candidateGpu: candidateArm.samples[roundIndex]!.gpuMilliseconds,
        baselineWall: baselineArm.samples[roundIndex]!.wallMilliseconds,
        candidateWall: candidateArm.samples[roundIndex]!.wallMilliseconds,
      }));
    const summarizeMetric = (metric: "Gpu" | "Wall") => {
      const baselineValues = rounds.map((round) =>
        round[`baseline${metric}`]);
      const candidateValues = rounds.map((round) =>
        round[`candidate${metric}`]);
      const meanBaseline = mean(baselineValues);
      const meanCandidate = mean(candidateValues);
      const medianBaseline = median(baselineValues);
      const medianCandidate = median(candidateValues);
      return Object.freeze({ baselineValues, candidateValues,
        meanBaselineMilliseconds: meanBaseline,
        meanCandidateMilliseconds: meanCandidate,
        meanSpeedup: meanBaseline / meanCandidate,
        meanSavingMilliseconds: meanBaseline - meanCandidate,
        medianBaselineMilliseconds: medianBaseline,
        medianCandidateMilliseconds: medianCandidate,
        medianSpeedup: medianBaseline / medianCandidate,
        medianSavingMilliseconds: medianBaseline - medianCandidate,
        pairWins: candidateValues.filter((value, index) =>
          value < baselineValues[index]!).length });
    };
    const gpu = summarizeMetric("Gpu");
    const wall = summarizeMetric("Wall");
    const meanWallGpuSavingRatio = wall.meanSavingMilliseconds /
      gpu.meanSavingMilliseconds;
    return Object.freeze({ panel: panel.id, baseline, candidate,
      rounds: Object.freeze(rounds), gpu, wall, meanWallGpuSavingRatio,
      wallGpuSavingsAgree: meanWallGpuSavingRatio >= 0.75 &&
        meanWallGpuSavingRatio <= 1.25,
      projectedMeanGpuSavingMilliseconds:
        gpu.meanSavingMilliseconds * PROJECTION_MULTIPLIER,
      projectedMeanWallSavingMilliseconds:
        wall.meanSavingMilliseconds * PROJECTION_MULTIPLIER });
  };
  const comparisonsFor = (id: Opt0081MlpPanelId) => {
    const panel = panels.find((entry) => entry.id === id)!;
    return Object.freeze({ bOverA: comparison(panel, "B", "A"),
      cOverA: comparison(panel, "C", "A"),
      cOverB: comparison(panel, "C", "B") });
  };
  const comparisons = Object.freeze({
    gate: comparisonsFor("gate"), up: comparisonsFor("up"),
    down: comparisonsFor("down"), chain: comparisonsFor("chain"),
  });
  const paired = (entry: ReturnType<typeof comparison>) =>
    entry.gpu.pairWins >= REQUIRED_PAIR_WINS &&
    entry.wall.pairWins >= REQUIRED_PAIR_WINS;
  const bStandalonePassed = panels.every((entry) => entry.bOverAPassed) &&
    PANEL_SPECS.every(({ id }) => paired(comparisons[id].bOverA)) &&
    comparisons.chain.bOverA.gpu.meanSavingMilliseconds >=
      REQUIRED_B_OVER_A_SAVING_MILLISECONDS &&
    comparisons.chain.bOverA.wall.meanSavingMilliseconds >=
      REQUIRED_B_OVER_A_SAVING_MILLISECONDS &&
    comparisons.chain.bOverA.wallGpuSavingsAgree;
  const cDiagnosticPassed = panels.every((entry) =>
      entry.cOverAPassed && entry.cOverBPassed) &&
    PANEL_SPECS.every(({ id }) => paired(comparisons[id].cOverA) &&
      paired(comparisons[id].cOverB)) &&
    comparisons.chain.cOverA.gpu.meanSavingMilliseconds >=
      REQUIRED_C_DIAGNOSTIC_SAVING_MILLISECONDS &&
    comparisons.chain.cOverA.wall.meanSavingMilliseconds >=
      REQUIRED_C_DIAGNOSTIC_SAVING_MILLISECONDS &&
    comparisons.chain.cOverB.gpu.meanSavingMilliseconds >=
      REQUIRED_C_DIAGNOSTIC_SAVING_MILLISECONDS &&
    comparisons.chain.cOverB.wall.meanSavingMilliseconds >=
      REQUIRED_C_DIAGNOSTIC_SAVING_MILLISECONDS &&
    comparisons.chain.cOverA.wallGpuSavingsAgree &&
    comparisons.chain.cOverB.wallGpuSavingsAgree;
  const gates = Object.freeze({ requiredPairWins: REQUIRED_PAIR_WINS,
    bStandalone: Object.freeze({
      requiredCompleteChainSavingMilliseconds:
        REQUIRED_B_OVER_A_SAVING_MILLISECONDS,
      passed: bStandalonePassed }),
    cDiagnostic: Object.freeze({
      requiredCompleteChainSavingMilliseconds:
        REQUIRED_C_DIAGNOSTIC_SAVING_MILLISECONDS,
      primitiveQualified: false, selectableUnderOpt0081: false,
      passed: cDiagnosticPassed }) });
  const selectedArm: "B" | null = bStandalonePassed ? "B" : null;
  return Object.freeze({ samplesPerArmPerPanel: TIMING_ROUNDS.length,
    exactPanelOrder: Object.freeze(PANEL_SPECS.map(({ id }) => id)),
    panels: Object.freeze(panels), comparisons, gates,
    bStandalonePassed, cDiagnosticPassed, cPrimitiveQualified: false,
    cSelectableUnderOpt0081: false, selectedArm,
    passed: selectedArm === "B" });
}

export function parseOpt0081MlpThermalLaunchGate(
  parameters: URLSearchParams,
  readyAtEpochMilliseconds: number,
  launchedAtEpochMilliseconds: number,
): Opt0081MlpThermalLaunchGate {
  const source = requiredParameter(parameters, "thermalSource");
  const command = requiredParameter(parameters, "thermalCommand");
  const traceStartedAtEpochMilliseconds = requiredFiniteParameter(
    parameters, "thermalTraceStartedAtEpochMilliseconds");
  const gateStartedAtEpochMilliseconds = requiredFiniteParameter(
    parameters, "thermalGateStartedAtEpochMilliseconds");
  const gateCompletedAtEpochMilliseconds = requiredFiniteParameter(
    parameters, "thermalGateCompletedAtEpochMilliseconds");
  const observationCount = requiredIntegerParameter(
    parameters, "thermalGateObservations");
  const pollMilliseconds = requiredIntegerParameter(
    parameters, "thermalPollMilliseconds");
  const maximumPollGapMilliseconds = requiredFiniteParameter(
    parameters, "thermalGateMaximumPollGapMilliseconds");
  const nonNominalObservationCount = requiredIntegerParameter(
    parameters, "thermalGateNonNominalObservations");
  const missingObservationCount = requiredIntegerParameter(
    parameters, "thermalGateMissingObservations");
  const duration = gateCompletedAtEpochMilliseconds -
    gateStartedAtEpochMilliseconds;
  const minimumObservations = Math.floor(duration /
    THERMAL_POLL_MILLISECONDS) + 1;
  const readyToGateDelayMilliseconds = gateStartedAtEpochMilliseconds -
    readyAtEpochMilliseconds;
  const launchDelayMilliseconds = launchedAtEpochMilliseconds -
    gateCompletedAtEpochMilliseconds;
  if (source !== THERMAL_SOURCE || command !== THERMAL_COMMAND ||
    pollMilliseconds !== THERMAL_POLL_MILLISECONDS ||
    traceStartedAtEpochMilliseconds > readyAtEpochMilliseconds ||
    readyToGateDelayMilliseconds < 0 ||
    duration < MINIMUM_NOMINAL_MILLISECONDS ||
    observationCount < minimumObservations ||
    maximumPollGapMilliseconds < 0 ||
    maximumPollGapMilliseconds > MAXIMUM_THERMAL_GAP_MILLISECONDS ||
    nonNominalObservationCount !== 0 || missingObservationCount !== 0 ||
    launchDelayMilliseconds < 0 ||
    launchDelayMilliseconds > MAXIMUM_GATE_TO_LAUNCH_MILLISECONDS) {
    throw new Error("OPT-0081 fresh continuous nominal thermal launch gate failed");
  }
  return Object.freeze({ source: THERMAL_SOURCE, command: THERMAL_COMMAND,
    traceStartedAtEpochMilliseconds, gateStartedAtEpochMilliseconds,
    gateCompletedAtEpochMilliseconds, observationCount,
    pollMilliseconds: THERMAL_POLL_MILLISECONDS,
    maximumPollGapMilliseconds, nonNominalObservationCount: 0,
    missingObservationCount: 0, readyToGateDelayMilliseconds,
    launchDelayMilliseconds });
}

export function parseOpt0081MlpThermalCompletion(
  parameters: URLSearchParams,
  launch: Opt0081MlpThermalLaunchGate,
  cleanupCompletedAtEpochMilliseconds: number,
): Opt0081MlpThermalCompletion {
  const schema = requiredParameter(parameters, "thermalTraceSchema");
  const sha256 = requiredParameter(parameters, "thermalTraceSha256");
  const byteLength = requiredIntegerParameter(parameters,
    "thermalTraceByteLength");
  const completedAtEpochMilliseconds = requiredFiniteParameter(
    parameters, "thermalTraceCompletedAtEpochMilliseconds");
  const observationCount = requiredIntegerParameter(
    parameters, "thermalTraceObservations");
  const maximumPollGapMilliseconds = requiredFiniteParameter(
    parameters, "thermalTraceMaximumPollGapMilliseconds");
  const nonNominalObservationCount = requiredIntegerParameter(
    parameters, "thermalTraceNonNominalObservations");
  const missingObservationCount = requiredIntegerParameter(
    parameters, "thermalTraceMissingObservations");
  const initialLevel = requiredIntegerParameter(parameters,
    "thermalTraceInitialLevel");
  const finalLevel = requiredIntegerParameter(parameters,
    "thermalTraceFinalLevel");
  const duration = completedAtEpochMilliseconds -
    launch.traceStartedAtEpochMilliseconds;
  const minimumObservations = Math.floor(duration /
    THERMAL_POLL_MILLISECONDS) + 1;
  if (schema !== THERMAL_TRACE_SCHEMA || !/^[0-9a-f]{64}$/u.test(sha256) ||
    byteLength <= 0 || completedAtEpochMilliseconds <
      cleanupCompletedAtEpochMilliseconds || observationCount <
      minimumObservations || maximumPollGapMilliseconds < 0 ||
    maximumPollGapMilliseconds > MAXIMUM_THERMAL_GAP_MILLISECONDS ||
    nonNominalObservationCount !== 0 || missingObservationCount !== 0 ||
    initialLevel !== 0 || finalLevel !== 0) {
    throw new Error("OPT-0081 complete through-cleanup thermal trace failed");
  }
  return Object.freeze({ schema: THERMAL_TRACE_SCHEMA, sha256, byteLength,
    completedAtEpochMilliseconds, observationCount,
    maximumPollGapMilliseconds, nonNominalObservationCount: 0,
    missingObservationCount: 0, initialLevel: 0, finalLevel: 0,
    coversCleanup: true });
}

if (typeof document !== "undefined") install();

function install(): void {
  const progress = requireElement<HTMLElement>("#progress");
  const runButton = requireElement<HTMLButtonElement>("#run");
  const finalizeButton = requireElement<HTMLButtonElement>("#finalize");
  const downloadButton = requireElement<HTMLButtonElement>("#download");
  const launchFields = requireElement<HTMLFieldSetElement>("#thermal-gate");
  const completionFields = requireElement<HTMLFieldSetElement>(
    "#thermal-completion",
  );
  let active: PreparedHarness | undefined;
  let running: PreparedHarness | undefined;
  let pending: Readonly<Record<string, unknown>> | undefined;
  let latestReceipt: Readonly<Record<string, unknown>> | undefined;
  let started = false;

  void prepareHarness((message) => progress.textContent = message).then(
    (prepared) => {
      active = prepared;
      document.body.dataset.status = "ready";
      progress.textContent =
        "READY — five full-M2250 A,A,B,B,C,C raw checkpoints and symmetric four-panel warmup passed; timing has not run";
      launchFields.disabled = false;
      runButton.disabled = false;
    },
    (error: unknown) => publishFailure(error, "preparation"),
  );

  runButton.addEventListener("click", () => {
    if (started || active === undefined) return;
    started = true;
    runButton.disabled = true;
    launchFields.disabled = true;
    document.body.dataset.status = "running";
    const prepared = active;
    active = undefined;
    running = prepared;
    const launchedAtEpochMilliseconds = Date.now();
    let thermalLaunch: Opt0081MlpThermalLaunchGate;
    try {
      thermalLaunch = parseOpt0081MlpThermalLaunchGate(
        fieldParameters("#thermal-gate"),
        prepared.readyAtEpochMilliseconds,
        launchedAtEpochMilliseconds,
      );
    } catch (error) {
      void prepared.cleanup().then((cleanup) => {
        running = undefined;
        publishFailure(error, "thermal-launch", { cleanup });
      });
      return;
    }
    void runTiming(prepared, thermalLaunch,
      (message) => progress.textContent = message).then((receipt) => {
        running = undefined;
        pending = receipt;
        completionFields.disabled = false;
        finalizeButton.disabled = false;
        document.body.dataset.status = "awaiting-thermal-completion";
        progress.textContent =
          "measurement and cleanup complete — keep polling, enter the through-cleanup trace, then finalize";
      }, (error: unknown) => {
        void prepared.cleanup().then((cleanup) => {
          running = undefined;
          publishFailure(error, "timing", { thermalLaunch, cleanup });
        });
      });
  }, { once: true });

  finalizeButton.addEventListener("click", () => {
    if (pending === undefined) return;
    const cleanupCompletedAtEpochMilliseconds = Number(
      pending["cleanupCompletedAtEpochMilliseconds"],
    );
    const thermalLaunch = pending["thermalLaunch"] as
      Opt0081MlpThermalLaunchGate;
    try {
      const thermalCompletion = parseOpt0081MlpThermalCompletion(
        fieldParameters("#thermal-completion"),
        thermalLaunch,
        cleanupCompletedAtEpochMilliseconds,
      );
      const passed = pending["inPagePassed"] === true;
      const disposition = classifyOpt0081MlpDisposition(pending);
      const timing = pending["timing"] as Readonly<Record<string, unknown>>;
      const selectedArm = timing["selectedArm"];
      const receipt = Object.freeze({ ...pending, schema: RECEIPT_SCHEMA,
        experiment: EXPERIMENT_ID, status: "completed", passed,
        thermal: Object.freeze({ launch: thermalLaunch,
          completion: thermalCompletion, passed: true }),
        decision: Object.freeze({
          disposition,
          selectedArm,
          cDiagnosticPassed: timing["cDiagnosticPassed"] === true,
          cPrimitiveQualified: false,
          cSelectableUnderOpt0081: false,
          bStandalonePassed: timing["bStandalonePassed"] === true,
          representativeLayerFollowUpAuthorized: passed,
          productionIntegrationAuthorized: false,
          packageChangeAuthorized: false,
          trajectoryOrListeningClaim: false,
          unchangedTimingRetryPerformed: false,
        }) });
      latestReceipt = receipt;
      publish(receipt, passed ? "passed" : "failed");
      progress.textContent = passed
        ? "completed — arm B actual MLP gate passed; representative-layer follow-up is authorized and production remains unchanged"
        : "completed — arm B stopped at the frozen actual-MLP gate; arm C was diagnostic-only";
      pending = undefined;
      finalizeButton.disabled = true;
      completionFields.disabled = true;
      downloadButton.disabled = false;
    } catch (error) {
      publishFailure(error, "thermal-completion", { pending });
    }
  });

  downloadButton.addEventListener("click", () => {
    const receipt = latestReceipt ?? window.__ACE_OPT0081_MLP_RESULT__;
    if (receipt === undefined) return;
    const blob = new Blob([JSON.stringify(receipt)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "opt-0081-dit-f16-actual-mlp-chain-receipt.json";
    anchor.click();
    URL.revokeObjectURL(url);
  });

  window.addEventListener("beforeunload", () => {
    void active?.cleanup();
    void running?.cleanup();
    active = undefined;
    running = undefined;
  });
}

async function prepareHarness(
  update: (message: string) => void,
): Promise<PreparedHarness> {
  requireLittleEndianHost();
  if (navigator.gpu === undefined) throw new Error("WebGPU unavailable");
  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: "high-performance",
    forceFallbackAdapter: false,
  });
  if (adapter === null) throw new Error("No WebGPU adapter");
  requireAdapter(adapter);
  const maximumStorageBytes = maximumStorageBindingBytes();
  const maximumBufferBytes = maximumStorageBytes + 2 * STORAGE_GUARD_BYTES;
  const device = await adapter.requestDevice({
    label: "ace-opt-0081-f16-actual-mlp-chain-device",
    requiredFeatures: ["shader-f16", "subgroups", "timestamp-query"],
    requiredLimits: {
      maxBufferSize: maximumBufferBytes,
      maxStorageBufferBindingSize: maximumStorageBytes,
      maxComputeInvocationsPerWorkgroup: 256,
      maxComputeWorkgroupSizeX: 256,
      maxComputeWorkgroupStorageSize: 16_384,
    },
  });
  const tracker = new BufferTracker();
  const uncapturedErrors: string[] = [];
  const deviceLosses: string[] = [];
  device.addEventListener("uncapturederror", (event) => {
    uncapturedErrors.push(event.error.message);
  });
  void device.lost.then((info) => {
    if (info.reason !== "destroyed") {
      deviceLosses.push(`${info.reason}:${info.message}`);
    }
  });
  const capability = Object.freeze({
    subgroupMinSize: 32 as const,
    subgroupMaxSize: 32 as const,
  });
  let armAKernel: AceOpt0009DenseGemmKernel | undefined;
  let armBKernel: AceOpt0081DenseF16InputKernel | undefined;
  let armCKernel: AceOpt0081DenseF16InputWeightMulticastKernel | undefined;
  let currentTransformerProducer:
    AceCorrectnessTransformerPlumbingKernel | undefined;
  let f16Producer: AceOpt0081F16DenseInputProducerKernel | undefined;
  let querySet: GPUQuerySet | undefined;
  let queryResolve: GPUBuffer | undefined;
  let queryReadback: GPUBuffer | undefined;
  let arms: Readonly<Record<Opt0081MlpArm, PreparedArm>> | undefined;
  let firstCandidateActivation: GPUBufferBinding | undefined;
  let gateWeight: GPUBufferBinding | undefined;
  let downOutput: GPUBufferBinding | undefined;
  let normalized: GPUBufferBinding | undefined;
  let scale: GPUBufferBinding | undefined;
  let shift: GPUBufferBinding | undefined;
  let cleanupPromise: Promise<Readonly<Record<string, unknown>>> | undefined;

  const cleanup = (): Promise<Readonly<Record<string, unknown>>> => {
    if (cleanupPromise !== undefined) {
      return cleanupPromise.then((receipt) => Object.freeze({ ...receipt,
        idempotent: true, repeatedCall: true }));
    }
    cleanupPromise = (async () => {
      const cleanupStartedAtEpochMilliseconds = Date.now();
      await device.queue.onSubmittedWorkDone().catch(() => undefined);
      const fallback = queryResolve === undefined
        ? undefined : binding(queryResolve, TIMESTAMP_QUERY_BYTES);
      const rejectionActivation = firstCandidateActivation ?? fallback;
      const rejectionWeight = gateWeight ?? fallback;
      const rejectionOutput = downOutput ?? fallback;
      let armBPostDestroyRejected = false;
      let armCPostDestroyRejected = false;
      let producerPostDestroyRejected = false;
      armCKernel?.destroy();
      if (armCKernel !== undefined && rejectionActivation !== undefined &&
        rejectionWeight !== undefined && rejectionOutput !== undefined) {
        try {
          await armCKernel.createDispatch(
            "opt0081-mlp-C-post-destroy-rejection", GATE_UP_SHAPE,
            { activation: rejectionActivation, weight: rejectionWeight,
              output: rejectionOutput },
          );
        } catch {
          armCPostDestroyRejected = true;
        }
      }
      armBKernel?.destroy();
      if (armBKernel !== undefined && rejectionActivation !== undefined &&
        rejectionWeight !== undefined && rejectionOutput !== undefined) {
        try {
          await armBKernel.createDispatch(
            "opt0081-mlp-B-post-destroy-rejection", GATE_UP_SHAPE,
            { activation: rejectionActivation, weight: rejectionWeight,
              output: rejectionOutput },
          );
        } catch {
          armBPostDestroyRejected = true;
        }
      }
      armAKernel?.destroy();
      f16Producer?.destroy();
      if (f16Producer !== undefined && normalized !== undefined &&
        scale !== undefined && shift !== undefined &&
        rejectionActivation !== undefined) {
        try {
          await f16Producer.createMlpModulatedDispatch(
            "opt0081-mlp-producer-post-destroy-rejection",
            "mlpModulated", "adaln",
            { batch: 1, tokens: ROWS, width: HIDDEN },
            { normalized, scale, shift, output: rejectionActivation },
          );
        } catch {
          producerPostDestroyRejected = true;
        }
      }
      currentTransformerProducer?.destroy();
      querySet?.destroy();
      tracker.destroyAll();
      device.destroy();
      await settlePostDrainEvents();
      const resource = tracker.receipt();
      const cleanupCompletedAtEpochMilliseconds = Date.now();
      return Object.freeze({ cleanupStartedAtEpochMilliseconds,
        cleanupCompletedAtEpochMilliseconds, ...resource,
        createdEqualsDestroyed:
          resource.createdBufferCount === resource.destroyedBufferCount,
        zeroLiveBuffers: resource.liveBufferCount === 0,
        zeroLiveBytes: resource.liveBytes === 0,
        mapsBalanced: resource.mapCount === resource.unmapCount &&
          resource.activeMapCount === 0,
        armBPostDestroyRejected, armCPostDestroyRejected,
        producerPostDestroyRejected,
        postDestroyRejected: armBPostDestroyRejected &&
          armCPostDestroyRejected && producerPostDestroyRejected,
        deviceDestroyed: true, idempotent: true, repeatedCall: false });
    })();
    return cleanupPromise;
  };

  try {
    armAKernel = AceOpt0009DenseGemmKernel.create(device, capability);
    armBKernel = AceOpt0081DenseF16InputKernel.create(device, capability);
    armCKernel = AceOpt0081DenseF16InputWeightMulticastKernel.create(
      device, capability,
    );
    currentTransformerProducer = AceCorrectnessTransformerPlumbingKernel
      .create(device, "reference-bf16");
    f16Producer = AceOpt0081F16DenseInputProducerKernel.create(device);
    querySet = device.createQuerySet({
      label: "opt0081-mlp-panel-timestamps", type: "timestamp", count: 2,
    });
    queryResolve = tracker.create(device, {
      label: "opt0081-mlp-timestamp-resolve", size: TIMESTAMP_QUERY_BYTES,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });
    queryReadback = tracker.create(device, {
      label: "opt0081-mlp-timestamp-readback", size: TIMESTAMP_QUERY_BYTES,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    update("allocating one shared actual-M2250 MLP fixture and three packed weights");
    normalized = createF32Input(device, tracker, "mlp-normalized",
      ROWS * HIDDEN, 0);
    scale = createF32Input(device, tracker, "mlp-scale", HIDDEN, 1);
    shift = createF32Input(device, tracker, "mlp-shift", HIDDEN, 2);
    const gateWeightFixture = await createPackedWeight(
      device, tracker, "gate", GATE_UP_SHAPE, 0x243f_6a88,
    );
    const upWeightFixture = await createPackedWeight(
      device, tracker, "up", GATE_UP_SHAPE, 0x85a3_08d3,
    );
    const downWeightFixture = await createPackedWeight(
      device, tracker, "down", DOWN_SHAPE, 0x1319_8a2e,
    );
    gateWeight = gateWeightFixture.binding;

    const tensors = Object.freeze({
      mlpF32: createGuardedTensor(device, tracker, "mlp-modulated-f32",
        ROWS * HIDDEN, HIDDEN, 4),
      mlpF16: createGuardedTensor(device, tracker, "mlp-modulated-f16",
        ROWS * HIDDEN, HIDDEN, 2),
      gate: createGuardedTensor(device, tracker, "gate-f32",
        ROWS * INTERMEDIATE, INTERMEDIATE, 4),
      up: createGuardedTensor(device, tracker, "up-f32",
        ROWS * INTERMEDIATE, INTERMEDIATE, 4),
      gatedF32: createGuardedTensor(device, tracker, "gated-activation-f32",
        ROWS * INTERMEDIATE, INTERMEDIATE, 4),
      gatedF16: createGuardedTensor(device, tracker, "gated-activation-f16",
        ROWS * INTERMEDIATE, INTERMEDIATE, 2),
      down: createGuardedTensor(device, tracker, "projected-mlp-f32",
        ROWS * HIDDEN, HIDDEN, 4),
    });
    firstCandidateActivation = tensors.mlpF16.binding;
    downOutput = tensors.down.binding;
    const maximumReadbackBytes = Math.max(...Object.values(tensors).map(
      ({ totalBytes }) => totalBytes,
    ));
    const checkpointReadback = tracker.create(device, {
      label: "opt0081-mlp-shared-checkpoint-readback",
      size: maximumReadbackBytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    update("compiling A/B/C actual producer and dense chains");
    const common = Object.freeze({ normalized, scale, shift,
      gateWeight: gateWeightFixture.binding,
      upWeight: upWeightFixture.binding,
      downWeight: downWeightFixture.binding, tensors });
    const A = await createPreparedArm("A", armAKernel,
      currentTransformerProducer, f16Producer, common);
    const B = await createPreparedArm("B", armBKernel,
      currentTransformerProducer, f16Producer, common);
    const C = await createPreparedArm("C", armCKernel,
      currentTransformerProducer, f16Producer, common);
    arms = Object.freeze({ A, B, C });

    const identity = await buildIdentity(adapter, device, Object.freeze({
      gateWeightSha256: gateWeightFixture.sha256,
      upWeightSha256: upWeightFixture.sha256,
      downWeightSha256: downWeightFixture.sha256,
    }));
    update("running staged A,A,B,B,C,C raw checkpoint correctness");
    const correctness = await verifyMlpCorrectness(device, tracker,
      checkpointReadback, arms, tensors, update);
    await settlePostDrainEvents();
    if (correctness["passed"] !== true || uncapturedErrors.length !== 0 ||
      deviceLosses.length !== 0) {
      const failed = Object.freeze({ ...correctness,
        uncapturedGpuErrorCount: uncapturedErrors.length,
        deviceLossCount: deviceLosses.length,
        completedBeforeReady: true, stoppedBeforeTiming: true,
        passed: false });
      throw new Opt0081MlpCorrectnessGateFailure(failed,
        "OPT-0081 actual MLP raw checkpoint correctness failed");
    }

    update("symmetrically warming all four panels for A/B/C");
    const warmupStartedAtEpochMilliseconds = Date.now();
    let panelExecutions = 0;
    for (const armOrder of [ARM_ORDERS[0]!, ARM_ORDERS[1]!] as const) {
      for (const arm of armOrder) {
        for (const { id } of PANEL_SPECS) {
          await executeAndDrain(device, arms[arm].panels[id].dispatches);
          panelExecutions += 1;
        }
      }
    }
    await settlePostDrainEvents();
    const warmupCompletedAtEpochMilliseconds = Date.now();
    if (uncapturedErrors.length !== 0 || deviceLosses.length !== 0) {
      throw new Error("OPT-0081 GPU failure during MLP warmup");
    }
    const readyAtEpochMilliseconds = Date.now();
    return Object.freeze({ adapter, device, tracker, armAKernel, armBKernel,
      armCKernel, currentTransformerProducer, f16Producer,
      querySet, queryResolve, queryReadback, arms,
      firstCandidateActivation, gateWeight, downOutput,
      correctness: Object.freeze({ ...correctness,
        uncapturedGpuErrorCount: 0, deviceLossCount: 0,
        completedBeforeReady: true, passed: true }),
      identity,
      warmup: Object.freeze({ warmupStartedAtEpochMilliseconds,
        warmupCompletedAtEpochMilliseconds, armOccurrences: 6,
        panelExecutions, panelsPerArmOccurrence: 4,
        balancedOrders: Object.freeze([ARM_ORDERS[0], ARM_ORDERS[1]]),
        completedBeforeReady: true }),
      readyAtEpochMilliseconds, uncapturedErrors, deviceLosses, cleanup });
  } catch (error) {
    const setupCleanup = await cleanup();
    throw new Opt0081MlpPreparationFailure(error, setupCleanup);
  }
}

interface PreparedArmCommon {
  readonly normalized: GPUBufferBinding;
  readonly scale: GPUBufferBinding;
  readonly shift: GPUBufferBinding;
  readonly gateWeight: GPUBufferBinding;
  readonly upWeight: GPUBufferBinding;
  readonly downWeight: GPUBufferBinding;
  readonly tensors: Readonly<{
    mlpF32: GuardedTensor;
    mlpF16: GuardedTensor;
    gate: GuardedTensor;
    up: GuardedTensor;
    gatedF32: GuardedTensor;
    gatedF16: GuardedTensor;
    down: GuardedTensor;
  }>;
}

type DenseOwner = AceOpt0009DenseGemmKernel |
  AceOpt0081DenseF16InputKernel |
  AceOpt0081DenseF16InputWeightMulticastKernel;

async function createPreparedArm(
  arm: Opt0081MlpArm,
  denseOwner: DenseOwner,
  currentTransformer: AceCorrectnessTransformerPlumbingKernel,
  f16Producer: AceOpt0081F16DenseInputProducerKernel,
  common: PreparedArmCommon,
): Promise<PreparedArm> {
  const candidate = arm !== "A";
  const mlpOutput = candidate
    ? common.tensors.mlpF16.binding : common.tensors.mlpF32.binding;
  const gatedOutput = candidate
    ? common.tensors.gatedF16.binding : common.tensors.gatedF32.binding;
  const mlpProducer: EncodableDispatch = candidate
    ? await f16Producer.createMlpModulatedDispatch(
      `opt0081-mlp-${arm}-mlpModulated-f16`,
      "mlpModulated", "adaln",
      { batch: 1, tokens: ROWS, width: HIDDEN },
      { normalized: common.normalized, scale: common.scale,
        shift: common.shift, output: mlpOutput },
    )
    : await currentTransformer.createAdaLnDispatch(
      "opt0081-mlp-A-mlpModulated-f32",
      { batch: 1, tokens: ROWS, width: HIDDEN },
      { normalized: common.normalized, scale: common.scale,
        shift: common.shift, output: mlpOutput },
    );
  const gate = await denseOwner.createDispatch(
    `opt0081-mlp-${arm}-gate`, GATE_UP_SHAPE,
    { activation: mlpOutput, weight: common.gateWeight,
      output: common.tensors.gate.binding },
  );
  const up = await denseOwner.createDispatch(
    `opt0081-mlp-${arm}-up`, GATE_UP_SHAPE,
    { activation: mlpOutput, weight: common.upWeight,
      output: common.tensors.up.binding },
  );
  const gatedProducer: EncodableDispatch = candidate
    ? await f16Producer.createGatedActivationDispatch(
      `opt0081-mlp-${arm}-gatedActivation-f16`,
      "gatedActivation", "swiglu",
      { batch: 1, tokens: ROWS, width: INTERMEDIATE },
      { gate: common.tensors.gate.binding, up: common.tensors.up.binding,
        output: gatedOutput },
    )
    : await currentTransformer.createSwiGluDispatch(
      "opt0081-mlp-A-gatedActivation-f32",
      { batch: 1, tokens: ROWS, width: INTERMEDIATE },
      { gate: common.tensors.gate.binding, up: common.tensors.up.binding,
        output: gatedOutput },
    );
  const down = await denseOwner.createDispatch(
    `opt0081-mlp-${arm}-down`, DOWN_SHAPE,
    { activation: gatedOutput, weight: common.downWeight,
      output: common.tensors.down.binding },
  );
  const scheduled = (dispatch: AceGemmDispatch) =>
    dispatch.plan.outputRanges.reduce((sum, range) =>
      sum + range.multiplyAdds, 0);
  const validProjectionMultiplyAdds = ROWS * HIDDEN * INTERMEDIATE;
  const panel = (id: Opt0081MlpPanelId,
    dispatches: readonly EncodableDispatch[],
    denseDispatches: readonly AceGemmDispatch[]): TimedPanelDispatch =>
      Object.freeze({ id, label: `opt0081-mlp-${arm}-${id}`,
        dispatches: Object.freeze(dispatches),
        validMultiplyAdds:
          validProjectionMultiplyAdds * denseDispatches.length,
        scheduledMultiplyAdds: denseDispatches.reduce((sum, dispatch) =>
          sum + scheduled(dispatch), 0) });
  return Object.freeze({ arm, mlpProducer, gate, up, gatedProducer, down,
    panels: Object.freeze({
      gate: panel("gate", [mlpProducer, gate], [gate]),
      up: panel("up", [mlpProducer, up], [up]),
      down: panel("down", [gatedProducer, down], [down]),
      chain: panel("chain",
        [mlpProducer, gate, up, gatedProducer, down], [gate, up, down]),
    }) });
}

type CheckpointId =
  "mlpModulated" | "gate" | "up" | "gatedActivation" | "down";

interface CheckpointSpec {
  readonly id: CheckpointId;
  readonly storage: "producer-boundary-u16" | "dense-output-u32";
  readonly columns: number;
  tensor(arm: Opt0081MlpArm,
    tensors: PreparedArmCommon["tensors"]): GuardedTensor;
}

const CHECKPOINT_SPECS: readonly CheckpointSpec[] = Object.freeze([
  Object.freeze({ id: "mlpModulated", storage: "producer-boundary-u16",
    columns: HIDDEN, tensor: (arm: Opt0081MlpArm,
      tensors: PreparedArmCommon["tensors"]) =>
      arm === "A" ? tensors.mlpF32 : tensors.mlpF16 }),
  Object.freeze({ id: "gate", storage: "dense-output-u32",
    columns: INTERMEDIATE, tensor: (_arm: Opt0081MlpArm,
      tensors: PreparedArmCommon["tensors"]) => tensors.gate }),
  Object.freeze({ id: "up", storage: "dense-output-u32",
    columns: INTERMEDIATE, tensor: (_arm: Opt0081MlpArm,
      tensors: PreparedArmCommon["tensors"]) => tensors.up }),
  Object.freeze({ id: "gatedActivation", storage: "producer-boundary-u16",
    columns: INTERMEDIATE, tensor: (arm: Opt0081MlpArm,
      tensors: PreparedArmCommon["tensors"]) =>
      arm === "A" ? tensors.gatedF32 : tensors.gatedF16 }),
  Object.freeze({ id: "down", storage: "dense-output-u32",
    columns: HIDDEN, tensor: (_arm: Opt0081MlpArm,
      tensors: PreparedArmCommon["tensors"]) => tensors.down }),
]);

async function verifyMlpCorrectness(
  device: GPUDevice,
  tracker: BufferTracker,
  readback: GPUBuffer,
  arms: Readonly<Record<Opt0081MlpArm, PreparedArm>>,
  tensors: PreparedArmCommon["tensors"],
  update: (message: string) => void,
): Promise<Readonly<Record<string, unknown>>> {
  const cases: Readonly<Record<string, unknown>>[] = [];
  for (const [checkpointIndex, checkpoint] of CHECKPOINT_SPECS.entries()) {
    update(`raw checkpoint ${checkpointIndex + 1}/5: ${checkpoint.id} A,A,B,B,C,C`);
    const summaries: Record<string, Readonly<Record<string, unknown>>> = {};
    let complete = true;
    const execute = async (arm: Opt0081MlpArm,
      receiptKey: string): Promise<U32Snapshot | U16Snapshot> => {
      const target = checkpoint.tensor(arm, tensors);
      prefillGuardedTensor(device, target);
      const snapshot = await executeCheckpoint(device, tracker, readback,
        arms[arm], target);
      complete &&= completeTensorSnapshot(snapshot);
      summaries[receiptKey] = Object.freeze({ sha256: snapshot.sha256,
        positiveZeroCount: snapshot.positiveZeroCount,
        negativeZeroCount: snapshot.negativeZeroCount,
        nonFiniteCount: snapshot.nonFiniteCount,
        qNaNPrefillCount: snapshot.qNaNPrefillCount,
        prefixGuardIntact: snapshot.prefixGuardIntact,
        suffixGuardIntact: snapshot.suffixGuardIntact,
        firstValidWritten: snapshot.firstValidWritten,
        lastValidWritten: snapshot.lastValidWritten,
        partialMTailWritten: snapshot.partialMTailWritten });
      await yieldToBrowser();
      return snapshot;
    };
    let A: U32Snapshot | undefined = await execute("A", "A") as U32Snapshot;
    let A2: U32Snapshot | undefined =
      await execute("A", "A2") as U32Snapshot;
    const aa = compareU32Snapshots(A, A2, checkpoint.id,
      checkpoint.columns);
    A2 = undefined;
    let comparisons: Readonly<Record<string, Readonly<Record<string, unknown>>>>;
    if (checkpoint.storage === "dense-output-u32") {
      let B: U32Snapshot | undefined =
        await execute("B", "B") as U32Snapshot;
      const ab = compareU32Snapshots(A, B, checkpoint.id,
        checkpoint.columns);
      let B2: U32Snapshot | undefined =
        await execute("B", "B2") as U32Snapshot;
      const bb = compareU32Snapshots(B, B2, checkpoint.id,
        checkpoint.columns);
      B2 = undefined;
      let C: U32Snapshot | undefined =
        await execute("C", "C") as U32Snapshot;
      const ac = compareU32Snapshots(A, C, checkpoint.id,
        checkpoint.columns);
      const bc = compareU32Snapshots(B, C, checkpoint.id,
        checkpoint.columns);
      A = undefined;
      B = undefined;
      let C2: U32Snapshot | undefined =
        await execute("C", "C2") as U32Snapshot;
      const cc = compareU32Snapshots(C, C2, checkpoint.id,
        checkpoint.columns);
      C = undefined;
      C2 = undefined;
      comparisons = Object.freeze({
        aa, ab, bb, ac, cc, bc,
      });
    } else {
      let B: U16Snapshot | undefined =
        await execute("B", "B") as U16Snapshot;
      const ab = compareF32ToF16Cast(A, B, checkpoint.id,
        checkpoint.columns);
      let B2: U16Snapshot | undefined =
        await execute("B", "B2") as U16Snapshot;
      const bb = compareU16Snapshots(B, B2, checkpoint.id,
        checkpoint.columns);
      B2 = undefined;
      let C: U16Snapshot | undefined =
        await execute("C", "C") as U16Snapshot;
      const ac = compareF32ToF16Cast(A, C, checkpoint.id,
        checkpoint.columns);
      const bc = compareU16Snapshots(B, C, checkpoint.id,
        checkpoint.columns);
      A = undefined;
      B = undefined;
      let C2: U16Snapshot | undefined =
        await execute("C", "C2") as U16Snapshot;
      const cc = compareU16Snapshots(C, C2, checkpoint.id,
        checkpoint.columns);
      C = undefined;
      C2 = undefined;
      comparisons = Object.freeze({
        aa, ab, bb, ac, cc, bc,
      });
    }
    const exact = Object.values(comparisons).every((entry) =>
      entry["differingWordCount"] === 0);
    cases.push(Object.freeze({ id: checkpoint.id,
      storage: checkpoint.storage, rows: ROWS, columns: checkpoint.columns,
      elements: ROWS * checkpoint.columns,
      executionOrder: Object.freeze(["A", "A", "B", "B", "C", "C"]),
      snapshots: Object.freeze(summaries),
      comparisons, everyRegisteredComparisonRawExact: exact,
      finiteCompleteQNaNOverwriteGuardsAndTail: complete,
      passed: exact && complete }));
  }
  const denseCases = cases.filter((entry) =>
    entry["storage"] === "dense-output-u32");
  const boundaryCases = cases.filter((entry) =>
    entry["storage"] === "producer-boundary-u16");
  const denseWordsPerComparison = denseCases.reduce((sum, entry) =>
    sum + Number(entry["elements"]), 0);
  const boundaryWordsPerArmExecution = boundaryCases.reduce((sum, entry) =>
    sum + Number(entry["elements"]), 0);
  const passed = cases.length === 5 &&
    cases.every((entry) => entry["passed"] === true);
  return Object.freeze({
    fixtureVersion: "opt0081-actual-mlp-full-m2250-v1",
    rows: ROWS, hidden: HIDDEN, intermediate: INTERMEDIATE,
    rawCheckpointCount: cases.length,
    boundaryCheckpointIds: Object.freeze([
      "mlpModulated", "gatedActivation",
    ]),
    denseCheckpointIds: Object.freeze(["gate", "up", "down"]),
    executionOrder: "A,A,B,B,C,C",
    checkpointOuterArmInnerExecution: true,
    sharedCheckpointReadbackBufferCount: 1,
    maximumRetainedFullSnapshots: 3,
    denseU32WordsPerComparison: denseWordsPerComparison,
    denseU32ComparisonCount: 18,
    candidateBoundaryU16WordsRead:
      boundaryWordsPerArmExecution * 4,
    candidateBoundaryComparedToIndependentCast: true,
    currentProducerF32Arithmetic: true,
    candidateSingleFinalF16Store: true,
    gpuConversionDispatchCount: 0,
    everyBoundaryRawU16Exact: boundaryCases.every((entry) =>
      entry["passed"] === true),
    everyDenseRawU32Exact: denseCases.every((entry) =>
      entry["passed"] === true),
    everyArmRerunExact: passed,
    directBCRawExact: passed,
    finiteCompleteOutputs: passed,
    exactSignedZeroBehavior: passed,
    qNaNPrefillOverwrite: passed,
    firstLastAndRows2240To2249Covered: passed,
    adjacentGuardsIntact: passed,
    cases: Object.freeze(cases), passed,
  });
}

async function executeCheckpoint(
  device: GPUDevice,
  tracker: BufferTracker,
  readback: GPUBuffer,
  arm: PreparedArm,
  target: GuardedTensor,
): Promise<U32Snapshot | U16Snapshot> {
  if (readback.mapState !== "unmapped") {
    throw new Error("OPT-0081 MLP checkpoint readback must be unmapped");
  }
  const encoder = device.createCommandEncoder({
    label: `opt0081-mlp-${arm.arm}-${target.label}-correctness`,
  });
  const pass = encoder.beginComputePass();
  for (const dispatch of arm.panels.chain.dispatches) dispatch.encode(pass);
  pass.end();
  encoder.copyBufferToBuffer(target.buffer, 0, readback, 0,
    target.totalBytes);
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  await tracker.mapRead(readback);
  try {
    return target.elementBytes === 4
      ? await snapshotU32(readback, target)
      : await snapshotU16(readback, target);
  } finally {
    tracker.unmap(readback);
  }
}

function prefillGuardedTensor(device: GPUDevice, tensor: GuardedTensor): void {
  if (tensor.elementBytes === 4) {
    const guards = new Uint32Array(STORAGE_GUARD_BYTES / 4);
    guards.fill(STORAGE_GUARD_U32);
    device.queue.writeBuffer(tensor.buffer, 0, guards);
    device.queue.writeBuffer(tensor.buffer,
      STORAGE_GUARD_BYTES + tensor.bodyBytes, guards);
    const chunk = new Uint32Array(Math.min(tensor.elements, 262_144));
    chunk.fill(OUTPUT_PREFILL_QNAN_U32);
    for (let index = 0; index < tensor.elements; index += chunk.length) {
      const count = Math.min(chunk.length, tensor.elements - index);
      device.queue.writeBuffer(tensor.buffer,
        STORAGE_GUARD_BYTES + index * 4, chunk, 0, count);
    }
  } else {
    const guards = new Uint16Array(STORAGE_GUARD_BYTES / 2);
    guards.fill(STORAGE_GUARD_U16);
    device.queue.writeBuffer(tensor.buffer, 0, guards);
    device.queue.writeBuffer(tensor.buffer,
      STORAGE_GUARD_BYTES + tensor.bodyBytes, guards);
    const chunk = new Uint16Array(Math.min(tensor.elements, 524_288));
    chunk.fill(OUTPUT_PREFILL_QNAN_U16);
    for (let index = 0; index < tensor.elements; index += chunk.length) {
      const count = Math.min(chunk.length, tensor.elements - index);
      device.queue.writeBuffer(tensor.buffer,
        STORAGE_GUARD_BYTES + index * 2, chunk, 0, count);
    }
  }
}

async function snapshotU32(
  readback: GPUBuffer,
  tensor: GuardedTensor,
): Promise<U32Snapshot> {
  const all = new Uint32Array(readback.getMappedRange(0, tensor.totalBytes));
  const guardWords = STORAGE_GUARD_BYTES / 4;
  const bodyEnd = guardWords + tensor.elements;
  const words = all.slice(guardWords, bodyEnd);
  let nonFiniteCount = 0;
  let qNaNPrefillCount = 0;
  let positiveZeroCount = 0;
  let negativeZeroCount = 0;
  for (const word of words) {
    if ((word & 0x7f80_0000) === 0x7f80_0000) nonFiniteCount += 1;
    if (word === OUTPUT_PREFILL_QNAN_U32) qNaNPrefillCount += 1;
    if (word === 0) positiveZeroCount += 1;
    if (word === 0x8000_0000) negativeZeroCount += 1;
  }
  const prefixGuardIntact = all.slice(0, guardWords).every((word) =>
    word === STORAGE_GUARD_U32);
  const suffixGuardIntact = all.slice(bodyEnd,
    bodyEnd + guardWords).every((word) => word === STORAGE_GUARD_U32);
  return Object.freeze({ words, sha256: await sha256U32(words),
    nonFiniteCount, qNaNPrefillCount, positiveZeroCount, negativeZeroCount,
    prefixGuardIntact, suffixGuardIntact,
    adjacentBeforeIntact: prefixGuardIntact,
    adjacentAfterIntact: suffixGuardIntact,
    firstValidWritten: words[0] !== OUTPUT_PREFILL_QNAN_U32,
    lastValidWritten:
      words[words.length - 1] !== OUTPUT_PREFILL_QNAN_U32,
    partialMTailWritten: tailWrittenU32(words, tensor.columns) });
}

async function snapshotU16(
  readback: GPUBuffer,
  tensor: GuardedTensor,
): Promise<U16Snapshot> {
  const all = new Uint16Array(readback.getMappedRange(0, tensor.totalBytes));
  const guardWords = STORAGE_GUARD_BYTES / 2;
  const bodyEnd = guardWords + tensor.elements;
  const words = all.slice(guardWords, bodyEnd);
  let nonFiniteCount = 0;
  let qNaNPrefillCount = 0;
  let positiveZeroCount = 0;
  let negativeZeroCount = 0;
  for (const word of words) {
    if ((word & 0x7c00) === 0x7c00) nonFiniteCount += 1;
    if (word === OUTPUT_PREFILL_QNAN_U16) qNaNPrefillCount += 1;
    if (word === 0) positiveZeroCount += 1;
    if (word === 0x8000) negativeZeroCount += 1;
  }
  const prefixGuardIntact = all.slice(0, guardWords).every((word) =>
    word === STORAGE_GUARD_U16);
  const suffixGuardIntact = all.slice(bodyEnd,
    bodyEnd + guardWords).every((word) => word === STORAGE_GUARD_U16);
  return Object.freeze({ words, sha256: await sha256U16(words),
    nonFiniteCount, qNaNPrefillCount, positiveZeroCount, negativeZeroCount,
    prefixGuardIntact, suffixGuardIntact,
    adjacentBeforeIntact: prefixGuardIntact,
    adjacentAfterIntact: suffixGuardIntact,
    firstValidWritten: words[0] !== OUTPUT_PREFILL_QNAN_U16,
    lastValidWritten:
      words[words.length - 1] !== OUTPUT_PREFILL_QNAN_U16,
    partialMTailWritten: tailWrittenU16(words, tensor.columns) });
}

function completeTensorSnapshot(
  snapshot: U32Snapshot | U16Snapshot,
): boolean {
  return snapshot.nonFiniteCount === 0 && snapshot.qNaNPrefillCount === 0 &&
    snapshot.prefixGuardIntact && snapshot.suffixGuardIntact &&
    snapshot.adjacentBeforeIntact && snapshot.adjacentAfterIntact &&
    snapshot.firstValidWritten && snapshot.lastValidWritten &&
    snapshot.partialMTailWritten;
}

function compareU32Snapshots(
  expected: U32Snapshot,
  actual: U32Snapshot,
  label: string,
  columns: number,
): Readonly<Record<string, unknown>> {
  if (expected.words.length !== actual.words.length) {
    throw new Error(`OPT-0081 MLP ${label} U32 length changed`);
  }
  let differingWordCount = 0;
  let finiteClassChangeCount = 0;
  let signedZeroDifferenceCount = 0;
  let firstDifference: Readonly<Record<string, unknown>> | null = null;
  for (let index = 0; index < expected.words.length; index += 1) {
    const expectedWord = expected.words[index]!;
    const actualWord = actual.words[index]!;
    if (expectedWord === actualWord) continue;
    differingWordCount += 1;
    if (f32Class(expectedWord) !== f32Class(actualWord)) {
      finiteClassChangeCount += 1;
    }
    if ((expectedWord & 0x7fff_ffff) === 0 &&
      (actualWord & 0x7fff_ffff) === 0) signedZeroDifferenceCount += 1;
    firstDifference ??= Object.freeze({ index,
      row: Math.floor(index / columns), column: index % columns,
      expectedU32: hex(expectedWord, 8), actualU32: hex(actualWord, 8) });
  }
  return Object.freeze({ comparisonStorage: "raw-u32",
    comparedWordCount: expected.words.length, differingWordCount,
    finiteClassChangeCount, signedZeroDifferenceCount, firstDifference });
}

function compareU16Snapshots(
  expected: U16Snapshot,
  actual: U16Snapshot,
  label: string,
  columns: number,
): Readonly<Record<string, unknown>> {
  if (expected.words.length !== actual.words.length) {
    throw new Error(`OPT-0081 MLP ${label} U16 length changed`);
  }
  let differingWordCount = 0;
  let signedZeroDifferenceCount = 0;
  let firstDifference: Readonly<Record<string, unknown>> | null = null;
  for (let index = 0; index < expected.words.length; index += 1) {
    const expectedWord = expected.words[index]!;
    const actualWord = actual.words[index]!;
    if (expectedWord === actualWord) continue;
    differingWordCount += 1;
    if ((expectedWord & 0x7fff) === 0 &&
      (actualWord & 0x7fff) === 0) signedZeroDifferenceCount += 1;
    firstDifference ??= Object.freeze({ index,
      row: Math.floor(index / columns), column: index % columns,
      expectedU16: hex(expectedWord, 4), actualU16: hex(actualWord, 4) });
  }
  return Object.freeze({ comparisonStorage: "raw-u16",
    comparedWordCount: expected.words.length, differingWordCount,
    signedZeroDifferenceCount, firstDifference });
}

function compareF32ToF16Cast(
  expected: U32Snapshot,
  actual: U16Snapshot,
  label: string,
  columns: number,
): Readonly<Record<string, unknown>> {
  if (expected.words.length !== actual.words.length) {
    throw new Error(`OPT-0081 MLP ${label} cast length changed`);
  }
  let differingWordCount = 0;
  let signedZeroDifferenceCount = 0;
  let firstDifference: Readonly<Record<string, unknown>> | null = null;
  const expectedFloats = new Float32Array(expected.words.buffer,
    expected.words.byteOffset, expected.words.length);
  for (let index = 0; index < expected.words.length; index += 1) {
    const expectedWord = numberToFloat16Bits(expectedFloats[index]!);
    const actualWord = actual.words[index]!;
    if (expectedWord === actualWord) continue;
    differingWordCount += 1;
    if ((expectedWord & 0x7fff) === 0 &&
      (actualWord & 0x7fff) === 0) signedZeroDifferenceCount += 1;
    firstDifference ??= Object.freeze({ index,
      row: Math.floor(index / columns), column: index % columns,
      sourceF32U32: hex(expected.words[index]!, 8),
      expectedCastU16: hex(expectedWord, 4),
      actualU16: hex(actualWord, 4) });
  }
  return Object.freeze({
    comparisonStorage: "independent-f32-to-f16-rne-v1",
    comparedWordCount: expected.words.length, differingWordCount,
    signedZeroDifferenceCount, firstDifference });
}

function tailWrittenU32(words: Uint32Array, columns: number): boolean {
  for (let index = (ROWS - 10) * columns; index < words.length; index += 1) {
    if (words[index] === OUTPUT_PREFILL_QNAN_U32) return false;
  }
  return true;
}

function tailWrittenU16(words: Uint16Array, columns: number): boolean {
  for (let index = (ROWS - 10) * columns; index < words.length; index += 1) {
    if (words[index] === OUTPUT_PREFILL_QNAN_U16) return false;
  }
  return true;
}

function createGuardedTensor(
  device: GPUDevice,
  tracker: BufferTracker,
  label: string,
  elements: number,
  columns: number,
  elementBytes: 2 | 4,
): GuardedTensor {
  const bodyBytes = elements * elementBytes;
  const totalBytes = bodyBytes + 2 * STORAGE_GUARD_BYTES;
  const buffer = tracker.create(device, {
    label: `opt0081-mlp-${label}`, size: totalBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST |
      GPUBufferUsage.COPY_SRC,
  });
  return Object.freeze({ buffer,
    binding: Object.freeze({ buffer, offset: STORAGE_GUARD_BYTES,
      size: bodyBytes }), elements, bodyBytes, totalBytes, columns,
    elementBytes, label });
}

function createF32Input(
  device: GPUDevice,
  tracker: BufferTracker,
  label: string,
  elements: number,
  operand: 0 | 1 | 2,
): GPUBufferBinding {
  const buffer = tracker.create(device, {
    label: `opt0081-mlp-${label}`, size: elements * 4,
    usage: GPUBufferUsage.STORAGE, mappedAtCreation: true,
  });
  const values = new Float32Array(buffer.getMappedRange());
  for (let index = 0; index < values.length; index += 1) {
    values[index] = mlpProducerValue(index, operand);
  }
  tracker.unmap(buffer);
  return binding(buffer, elements * 4);
}

async function createPackedWeight(
  device: GPUDevice,
  tracker: BufferTracker,
  label: "gate" | "up" | "down",
  shape: AceGemmShape,
  seed: number,
): Promise<Readonly<{ buffer: GPUBuffer; binding: GPUBufferBinding;
  sha256: string }>> {
  const bytes = shape.inner * shape.columns * 2;
  const buffer = tracker.create(device, {
    label: `opt0081-mlp-${label}-packed-f16-weight`, size: bytes,
    usage: GPUBufferUsage.STORAGE, mappedAtCreation: true,
  });
  const values = new Uint16Array(buffer.getMappedRange());
  fillMlpPackedWeight(values, shape, seed);
  const sha256 = await sha256Bytes(new Uint8Array(values.buffer,
    values.byteOffset, values.byteLength));
  tracker.unmap(buffer);
  return Object.freeze({ buffer, binding: binding(buffer, bytes), sha256 });
}

function fillMlpPackedWeight(
  values: Uint16Array,
  shape: AceGemmShape,
  seed: number,
): void {
  let physical = 0;
  for (let columnTile = 0; columnTile < shape.columns / 256;
    columnTile += 1) {
    for (let innerTile = 0; innerTile < shape.inner / 32;
      innerTile += 1) {
      for (let innerInTile = 0; innerInTile < 32; innerInTile += 1) {
        const inner = innerTile * 32 + innerInTile;
        for (let columnInTile = 0; columnInTile < 256;
          columnInTile += 1) {
          const column = columnTile * 256 + columnInTile;
          const mixed = mix32(seed ^ Math.imul(inner + 1, 0x85eb_ca6b) ^
            Math.imul(column + 1, 0xc2b2_ae35));
          const sign = (mixed >>> 31) << 15;
          const magnitudes = [0x2401, 0x2803, 0x2c05, 0x3007,
            0x3401, 0x3800] as const;
          values[physical] = magnitudes[mixed % magnitudes.length]! | sign;
          physical += 1;
        }
      }
    }
  }
  if (physical !== values.length) {
    throw new Error(`OPT-0081 MLP ${shape.inner}x${shape.columns} weight fill changed`);
  }
}

async function runTiming(
  prepared: PreparedHarness,
  thermalLaunch: Opt0081MlpThermalLaunchGate,
  update: (message: string) => void,
): Promise<Readonly<Record<string, unknown>>> {
  const samples = Object.fromEntries(PANEL_SPECS.map(({ id }) =>
    [id, { A: [] as Opt0081MlpTimestampSample[],
      B: [] as Opt0081MlpTimestampSample[],
      C: [] as Opt0081MlpTimestampSample[] }])) as Record<
        Opt0081MlpPanelId,
        Record<Opt0081MlpArm, Opt0081MlpTimestampSample[]>>;
  const rawSamples: Readonly<Record<string, unknown>>[] = [];
  const measurementStartedAtEpochMilliseconds = Date.now();
  for (const round of TIMING_ROUNDS) {
    for (const [armPosition, arm] of round.armOrder.entries()) {
      for (const [panelPosition, panelId] of round.panelOrder.entries()) {
        update(`round ${round.roundIndex + 1}/8 arm ${armPosition + 1}/3 ${arm}: panel ${panelPosition + 1}/4 ${panelId}`);
        const panel = prepared.arms[arm].panels[panelId];
        const sample = await timePanel(prepared.device, panel,
          prepared.querySet, prepared.queryResolve, prepared.queryReadback,
          prepared.tracker, arm);
        samples[panelId][arm].push(sample);
        rawSamples.push(Object.freeze({ roundIndex: round.roundIndex,
          armPosition, arm, panelPosition, panelId, ...sample }));
      }
      await yieldToBrowser();
    }
  }
  await prepared.device.queue.onSubmittedWorkDone();
  await settlePostDrainEvents();
  const measurementCompletedAtEpochMilliseconds = Date.now();
  if (prepared.uncapturedErrors.length !== 0 ||
    prepared.deviceLosses.length !== 0) {
    throw new Error("OPT-0081 observed an MLP timing GPU error or device loss");
  }
  const timingInput = Object.freeze({ panels: Object.freeze({
    gate: freezeArmSamples(samples.gate),
    up: freezeArmSamples(samples.up),
    down: freezeArmSamples(samples.down),
    chain: freezeArmSamples(samples.chain),
  }) });
  const timing = summarizeOpt0081MlpTiming(timingInput);
  const memoryBeforeCleanup = prepared.tracker.receipt();
  const cleanupFirst = await prepared.cleanup();
  const cleanupSecond = await prepared.cleanup();
  const cleanupCompletedAtEpochMilliseconds = Number(
    cleanupFirst["cleanupCompletedAtEpochMilliseconds"],
  );
  const cleanupPassed = cleanupFirst["zeroLiveBuffers"] === true &&
    cleanupFirst["zeroLiveBytes"] === true &&
    cleanupFirst["createdEqualsDestroyed"] === true &&
    cleanupFirst["mapsBalanced"] === true &&
    cleanupFirst["postDestroyRejected"] === true &&
    cleanupFirst["deviceDestroyed"] === true &&
    cleanupFirst["idempotent"] === true &&
    cleanupSecond["idempotent"] === true &&
    cleanupSecond["repeatedCall"] === true &&
    prepared.uncapturedErrors.length === 0 &&
    prepared.deviceLosses.length === 0;
  const inPagePassed = timing["bStandalonePassed"] === true &&
    timing["selectedArm"] === "B" &&
    prepared.correctness["passed"] === true && cleanupPassed;
  return Object.freeze({ schema: RECEIPT_SCHEMA, experiment: EXPERIMENT_ID,
    status: "awaiting-external-thermal-completion", passed: false,
    inPagePassed, thermalLaunch, cleanupCompletedAtEpochMilliseconds,
    identity: prepared.identity, correctness: prepared.correctness,
    warmup: prepared.warmup,
    protocol: Object.freeze({
      correctnessAndWarmupBeforeReady: true,
      actualM2250ProducersInsideEveryTimedPanel: true,
      gpuConversionDispatchCount: 0,
      rawCheckpointExecutionOrder: "A,A,B,B,C,C",
      rounds: TIMING_ROUNDS.length,
      exactRegisteredArmOrders: Object.freeze(ARM_ORDERS),
      fixedPanelOrder: Object.freeze(PANEL_SPECS.map(({ id }) => id)),
      samplesPerArmPerPanel: TIMING_ROUNDS.length,
      onePassOneCommandBufferOneSubmitOneDrainPerPanelSample: true,
      oneTimestampQueryPairPerPanelSample: true,
      downConsumesImmediatelyPrecedingValidatedGateUpResults: true,
      readbackHashingSerializationInsideTiming: false,
      allocationUploadCompilationInsideTiming: false,
      cPrimitiveQualified: false,
      cSelectableUnderOpt0081: false,
      unchangedTimingRetryPerformed: false,
    }),
    timing: Object.freeze({ ...timing, measurementStartedAtEpochMilliseconds,
      measurementCompletedAtEpochMilliseconds,
      rawSamples: Object.freeze(rawSamples) }),
    uncapturedGpuErrors: prepared.uncapturedErrors,
    deviceLosses: prepared.deviceLosses,
    memoryBeforeCleanup,
    cleanup: Object.freeze({ firstCall: cleanupFirst,
      secondCall: cleanupSecond,
      zeroPostCleanupGpuErrors: prepared.uncapturedErrors.length === 0,
      zeroPostCleanupDeviceLosses: prepared.deviceLosses.length === 0,
      passed: cleanupPassed }) });
}

function freezeArmSamples(
  entry: Record<Opt0081MlpArm, Opt0081MlpTimestampSample[]>,
): Readonly<Record<Opt0081MlpArm,
  readonly Opt0081MlpTimestampSample[]>> {
  return Object.freeze({ A: Object.freeze(entry.A),
    B: Object.freeze(entry.B), C: Object.freeze(entry.C) });
}

async function timePanel(
  device: GPUDevice,
  panel: TimedPanelDispatch,
  querySet: GPUQuerySet,
  queryResolve: GPUBuffer,
  queryReadback: GPUBuffer,
  tracker: BufferTracker,
  arm: Opt0081MlpArm,
): Promise<Opt0081MlpTimestampSample> {
  if (queryReadback.mapState !== "unmapped") {
    throw new Error("OPT-0081 MLP timestamp readback must be unmapped");
  }
  const encoder = device.createCommandEncoder({
    label: `${panel.label}-sample`,
  });
  const pass = encoder.beginComputePass({
    label: `${panel.label}-timestamped-compute`,
    timestampWrites: { querySet, beginningOfPassWriteIndex: 0,
      endOfPassWriteIndex: 1 },
  });
  for (const dispatch of panel.dispatches) dispatch.encode(pass);
  pass.end();
  encoder.resolveQuerySet(querySet, 0, 2, queryResolve, 0);
  encoder.copyBufferToBuffer(queryResolve, 0, queryReadback, 0,
    TIMESTAMP_QUERY_BYTES);
  const command = encoder.finish();
  const submitAtPerformanceMilliseconds = performance.now();
  device.queue.submit([command]);
  await device.queue.onSubmittedWorkDone();
  const fenceAtPerformanceMilliseconds = performance.now();
  const wallMilliseconds = fenceAtPerformanceMilliseconds -
    submitAtPerformanceMilliseconds;
  await tracker.mapRead(queryReadback);
  let timestampBegin: bigint;
  let timestampEnd: bigint;
  try {
    const timestamps = new BigUint64Array(queryReadback.getMappedRange());
    timestampBegin = timestamps[0]!;
    timestampEnd = timestamps[1]!;
  } finally {
    tracker.unmap(queryReadback);
  }
  if (timestampEnd <= timestampBegin) {
    throw new Error(`OPT-0081 MLP ${panel.id} ${arm} non-positive GPU timestamp`);
  }
  const gpuElapsedNanoseconds = Number(timestampEnd - timestampBegin);
  const gpuMilliseconds = gpuElapsedNanoseconds / 1_000_000;
  if (!Number.isSafeInteger(gpuElapsedNanoseconds) ||
    gpuMilliseconds <= 0 || !Number.isFinite(gpuMilliseconds) ||
    wallMilliseconds <= 0 || !Number.isFinite(wallMilliseconds)) {
    throw new Error(`OPT-0081 MLP ${panel.id} ${arm} invalid timing sample`);
  }
  return Object.freeze({ submitAtPerformanceMilliseconds,
    fenceAtPerformanceMilliseconds,
    submitAtEpochMilliseconds:
      performance.timeOrigin + submitAtPerformanceMilliseconds,
    fenceAtEpochMilliseconds:
      performance.timeOrigin + fenceAtPerformanceMilliseconds,
    wallMilliseconds, timestampBeginNanoseconds: timestampBegin.toString(),
    timestampEndNanoseconds: timestampEnd.toString(),
    gpuElapsedNanoseconds, gpuMilliseconds,
    gpuToWallRatio: gpuMilliseconds / wallMilliseconds,
    validMultiplyAdds: panel.validMultiplyAdds,
    scheduledMultiplyAdds: panel.scheduledMultiplyAdds,
    validGpuTflops: tflops(panel.validMultiplyAdds, gpuMilliseconds),
    scheduledGpuTflops: tflops(panel.scheduledMultiplyAdds, gpuMilliseconds),
    validWallTflops: tflops(panel.validMultiplyAdds, wallMilliseconds),
    scheduledWallTflops: tflops(panel.scheduledMultiplyAdds, wallMilliseconds),
    commandBufferCount: 1, queueDrainCount: 1,
    timestampResolveCount: 1, timestampCopyCount: 1 });
}

async function executeAndDrain(
  device: GPUDevice,
  dispatches: readonly EncodableDispatch[],
): Promise<void> {
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  for (const dispatch of dispatches) dispatch.encode(pass);
  pass.end();
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
}

async function buildIdentity(
  adapter: GPUAdapter,
  device: GPUDevice,
  fixtureHashes: Readonly<Record<"gateWeightSha256" |
    "upWeightSha256" | "downWeightSha256", string>>,
): Promise<Readonly<Record<string, unknown>>> {
  const routes = Object.freeze([
    Object.freeze({ id: "gate-up", shape: GATE_UP_SHAPE,
      labels: Object.freeze(["gate", "up"]) }),
    Object.freeze({ id: "down", shape: DOWN_SHAPE,
      labels: Object.freeze(["down"]) }),
  ]);
  const generatedShaders = await Promise.all(routes.map(async (route) => {
    const A = planAceOpt0009DenseGemm(route.shape);
    const B = planAceOpt0081DenseF16Input(route.shape);
    const C = planAceOpt0081DenseF16InputWeightMulticast(route.shape);
    return Object.freeze({ id: route.id, labels: route.labels,
      shape: route.shape,
      armAWgslSha256: await sha256Text(
        aceOpt0009DenseGemmWgsl(route.shape)),
      armBWgslSha256: await sha256Text(
        aceOpt0081DenseF16InputWgsl(route.shape)),
      armCWgslSha256: await sha256Text(
        aceOpt0081DenseF16InputWeightMulticastWgsl(route.shape)),
      plans: Object.freeze({ A: planReceipt(A), B: planReceipt(B),
        C: planReceipt(C) }),
      cMechanism: Object.freeze({
        subgroupsPerWorkgroup: C.subgroupsPerWorkgroup,
        rowsPerSubgroup: C.rowsPerSubgroup,
        columnsPerLane: C.columnsPerLane,
        accumulatorsPerLane: C.accumulatorsPerLane,
        packedRecordsPerInnerTile: C.packedRecordsPerInnerTile,
        packedRecordsPerLane: C.packedRecordsPerLane,
        workgroupStorageBytes: C.workgroupStorageBytes,
        barriersPerWorkgroup: C.barriersPerWorkgroup,
        barrierEvents: C.barrierEvents,
        packedRecordLoadsPerWorkgroup: C.packedRecordLoadsPerWorkgroup,
      }),
      identicalOutputRanges: JSON.stringify(A.outputRanges) ===
        JSON.stringify(B.outputRanges) &&
        JSON.stringify(B.outputRanges) === JSON.stringify(C.outputRanges),
    });
  }));
  return Object.freeze({
    registrationCommit: REGISTRATION_COMMIT,
    bOnlyCorrectionCommit: B_ONLY_CORRECTION_COMMIT,
    implementationBaseCommit: B_ONLY_CORRECTION_COMMIT,
    harnessImplementationCheckpointCommit:
      HARNESS_IMPLEMENTATION_CHECKPOINT_COMMIT,
    primitiveRegistrationCommit: PRIMITIVE_REGISTRATION_COMMIT,
    primitiveImplementationCheckpointCommit:
      PRIMITIVE_IMPLEMENTATION_COMMIT,
    allocationBaselineCommit:
      "bbe180bf7feb59272a5d5f7afbafb3877afee416",
    authoritativePrimitiveReceiptSha256:
      "8cb2c7c30dec7d179729d5644608fb1f0b9ad5b0478ba92125476936644c775c",
    primitiveSelectedArm: "B",
    cPrimitiveQualified: false,
    cSelectableUnderOpt0081: false,
    arms: Object.freeze({ A: "OPT-0009-f32-input-actual-producers",
      B: "OPT-0081-opt0009-typed-f16-input-actual-producers",
      C: "OPT-0081-opt0078-typed-f16-input-multicast-diagnostic-only" }),
    producerProfile: ACE_OPT_0081_DIT_F16_DENSE_INPUT_STORAGE_PROFILE,
    producerKernelSetId:
      ACE_OPT_0081_DIT_F16_DENSE_INPUT_PRODUCER_KERNEL_SET_ID,
    producerRoles: Object.freeze(["mlpModulated", "gatedActivation"]),
    actualMlpOperationOrder: Object.freeze([
      "mlpModulated", "gate", "up", "gatedActivation", "down",
    ]),
    panelOrder: Object.freeze(PANEL_SPECS.map(({ id }) => id)),
    sharedWeightLayout: ACE_DIT_DENSE_FP16_TILE_LAYOUT,
    sharedWeightBufferCount: 3,
    fixtureHashes,
    bufferOwnership: Object.freeze({
      immutableInputsBytes: ROWS * HIDDEN * 4 + HIDDEN * 4 * 2,
      sharedPackedWeightBytes:
        (HIDDEN * INTERMEDIATE + HIDDEN * INTERMEDIATE +
          INTERMEDIATE * HIDDEN) * 2,
      armAF32MlpModulatedBytes: ROWS * HIDDEN * 4,
      candidateSharedF16MlpModulatedBytes: ROWS * HIDDEN * 2,
      sharedGateBytes: ROWS * INTERMEDIATE * 4,
      sharedUpBytes: ROWS * INTERMEDIATE * 4,
      armAF32GatedActivationBytes: ROWS * INTERMEDIATE * 4,
      candidateSharedF16GatedActivationBytes:
        ROWS * INTERMEDIATE * 2,
      sharedProjectedMlpBytes: ROWS * HIDDEN * 4,
      serialDrainedArmReuse: true,
      duplicatedCandidateActivationCount: 0,
    }),
    sourceSha256: Object.freeze({
      A: await sha256Text(armAKernelSource),
      B: await sha256Text(armBKernelSource),
      C: await sha256Text(armCKernelSource),
      f16Producer: await sha256Text(producerKernelSource),
      currentTransformerProducer:
        await sha256Text(transformerProducerSource),
    }),
    generatedShaders: Object.freeze(generatedShaders),
    generatedShaderAggregateSha256: await sha256Text(
      generatedShaders.map((entry) => JSON.stringify(entry)).join("\n")),
    fixtureVersion: "opt0081-actual-mlp-full-m2250-v1",
    adapterInfo: adapter.info,
    browserUserAgent: navigator.userAgent,
    browserLanguage: navigator.language,
    browserHardwareConcurrency: navigator.hardwareConcurrency,
    crossOriginIsolated,
    stockWebGpuOnly: true,
    experimentalBrowserFlags: false,
    requestedFeatures: Object.freeze([
      "shader-f16", "subgroups", "timestamp-query",
    ]),
    deviceFeatures: Object.freeze([...device.features].sort()),
    authenticatedLimits: Object.freeze({
      maxBufferSize: device.limits.maxBufferSize,
      maxStorageBufferBindingSize:
        device.limits.maxStorageBufferBindingSize,
      maxComputeInvocationsPerWorkgroup:
        device.limits.maxComputeInvocationsPerWorkgroup,
      maxComputeWorkgroupSizeX: device.limits.maxComputeWorkgroupSizeX,
      maxComputeWorkgroupStorageSize:
        device.limits.maxComputeWorkgroupStorageSize,
    }),
  });
}

function planReceipt(plan: ReturnType<typeof planAceOpt0009DenseGemm> |
  ReturnType<typeof planAceOpt0081DenseF16Input> |
  ReturnType<typeof planAceOpt0081DenseF16InputWeightMulticast>):
Readonly<Record<string, unknown>> {
  return Object.freeze({ rows: plan.rows, inner: plan.inner,
    columns: plan.columns,
    tile: Object.freeze([plan.tileRows, plan.tileColumns, plan.tileInner]),
    workgroupSize: plan.workgroupSize,
    rowTiles: plan.rowTiles, columnTiles: plan.columnTiles,
    innerTiles: plan.innerTiles, workgroupCount: plan.workgroupCount,
    outputRangeCount: plan.outputRangeCount,
    scheduledMultiplyAdds: plan.outputRanges.reduce((sum, range) =>
      sum + range.multiplyAdds, 0),
    packedWeightStorageShape: plan.packedWeightStorageShape });
}

export function classifyOpt0081MlpDisposition(
  receipt: Readonly<Record<string, unknown>>,
): string {
  const timing = receipt["timing"] as
    Readonly<Record<string, unknown>> | undefined;
  if (receipt["inPagePassed"] === true &&
    timing?.["selectedArm"] === "B" &&
    timing["bStandalonePassed"] === true &&
    timing["cSelectableUnderOpt0081"] === false) {
    return "positive-B-actual-mlp-representative-layer-follow-up-authorized";
  }
  const correctness = receipt["correctness"] as
    Readonly<Record<string, unknown>> | undefined;
  const cleanup = receipt["cleanup"] as
    Readonly<Record<string, unknown>> | undefined;
  if (cleanup?.["passed"] !== true) {
    return "inconclusive-invalid-correctness-or-lifecycle-evidence";
  }
  if (correctness?.["passed"] !== true) {
    if (correctness !== undefined &&
      hasValidObservedCorrectnessMismatch(correctness)) {
      return "negative-stop-observed-raw-bit-correctness-mismatch";
    }
    return "inconclusive-invalid-correctness-or-lifecycle-evidence";
  }
  const panels = timing?.["panels"] as readonly Readonly<
    Record<string, unknown>
  >[] | undefined;
  const comparisons = timing?.["comparisons"] as
    Readonly<Record<string, Readonly<Record<string, unknown>>>> | undefined;
  if (timing?.["bStandalonePassed"] === undefined ||
    timing["cSelectableUnderOpt0081"] !== false ||
    panels?.length !== 4 || comparisons === undefined) {
    return "inconclusive-invalid-thermal-or-protocol-evidence";
  }
  if (hasMixedMlpTimingEvidence(panels, comparisons)) {
    return "inconclusive-directional-or-wall-gpu-evidence";
  }
  return "negative-stop-B-actual-mlp-materiality-gate-not-met";
}

function hasMixedMlpTimingEvidence(
  panels: readonly Readonly<Record<string, unknown>>[],
  comparisons: Readonly<Record<string, Readonly<Record<string, unknown>>>>,
): boolean {
  const panelStates = panels.map((panel) => {
    const gates = panel["gates"] as Readonly<Record<string,
      Readonly<Record<string, unknown>>>> | undefined;
    const values = Object.values(gates?.["bOverA"] ?? {}).map(Boolean);
    return values.length === 4 && values.every(Boolean) ? "pass" :
      values.length === 4 && values.every((value) => !value) ? "fail" :
        "mixed";
  });
  if (panelStates.includes("mixed") ||
    (panelStates.includes("pass") && panelStates.includes("fail"))) {
    return true;
  }
  for (const { id } of PANEL_SPECS) {
    const panelComparisons = comparisons[id] as
      Readonly<Record<string, unknown>> | undefined;
    const comparison = panelComparisons?.["bOverA"] as
      Readonly<Record<string, unknown>> | undefined;
    const gpu = comparison?.["gpu"] as
      Readonly<Record<string, unknown>> | undefined;
    const wall = comparison?.["wall"] as
      Readonly<Record<string, unknown>> | undefined;
    if (comparison === undefined || gpu === undefined || wall === undefined) {
      return true;
    }
    const gpuWins = Number(gpu["pairWins"]);
    const wallWins = Number(wall["pairWins"]);
    if ((gpuWins > 0 && gpuWins < REQUIRED_PAIR_WINS) ||
      (wallWins > 0 && wallWins < REQUIRED_PAIR_WINS) ||
      (gpuWins >= REQUIRED_PAIR_WINS) !==
        (wallWins >= REQUIRED_PAIR_WINS)) return true;
    const directionDiffers = (field: string) =>
      (Number(gpu[field]) > 0) !== (Number(wall[field]) > 0);
    if (directionDiffers("meanSavingMilliseconds") ||
      directionDiffers("medianSavingMilliseconds")) return true;
    if (id === "chain" &&
      ((Number(gpu["meanSavingMilliseconds"]) >=
        REQUIRED_B_OVER_A_SAVING_MILLISECONDS) !==
      (Number(wall["meanSavingMilliseconds"]) >=
        REQUIRED_B_OVER_A_SAVING_MILLISECONDS) ||
      comparison["wallGpuSavingsAgree"] === false)) return true;
  }
  return false;
}

function hasValidObservedCorrectnessMismatch(
  correctness: Readonly<Record<string, unknown>>,
): boolean {
  if (correctness["completedBeforeReady"] !== true ||
    correctness["uncapturedGpuErrorCount"] !== 0 ||
    correctness["deviceLossCount"] !== 0) return false;
  const cases = correctness["cases"] as readonly Readonly<
    Record<string, unknown>
  >[] | undefined;
  if (cases?.length !== 5) return false;
  return cases.some((entry) => {
    const comparisons = entry["comparisons"] as
      Readonly<Record<string, Readonly<Record<string, unknown>>>> |
      undefined;
    const observedDifference = Object.values(comparisons ?? {}).some(
      (comparison) => Number(comparison["differingWordCount"]) > 0,
    );
    return observedDifference ||
      entry["everyRegisteredComparisonRawExact"] === false ||
      entry["finiteCompleteQNaNOverwriteGuardsAndTail"] === false;
  });
}

class BufferTracker {
  private readonly live = new Set<GPUBuffer>();
  private readonly sizes = new Map<GPUBuffer, number>();
  private created = 0;
  private destroyed = 0;
  private liveBytesValue = 0;
  private maximumLiveBytesValue = 0;
  private maps = 0;
  private unmaps = 0;
  private activeMaps = 0;

  create(device: GPUDevice, descriptor: GPUBufferDescriptor): GPUBuffer {
    const buffer = device.createBuffer(descriptor);
    const size = Number(descriptor.size);
    this.live.add(buffer);
    this.sizes.set(buffer, size);
    this.created += 1;
    this.liveBytesValue += size;
    this.maximumLiveBytesValue = Math.max(this.maximumLiveBytesValue,
      this.liveBytesValue);
    if (descriptor.mappedAtCreation === true) {
      this.maps += 1;
      this.activeMaps += 1;
    }
    return buffer;
  }

  async mapRead(buffer: GPUBuffer): Promise<void> {
    await buffer.mapAsync(GPUMapMode.READ);
    this.maps += 1;
    this.activeMaps += 1;
  }

  unmap(buffer: GPUBuffer): void {
    if (buffer.mapState !== "mapped") {
      throw new Error("OPT-0081 attempted to unmap an unmapped buffer");
    }
    buffer.unmap();
    this.unmaps += 1;
    this.activeMaps -= 1;
    if (this.activeMaps < 0) {
      throw new Error("OPT-0081 map accounting became negative");
    }
  }

  destroy(buffer: GPUBuffer): void {
    if (!this.live.delete(buffer)) return;
    buffer.destroy();
    this.destroyed += 1;
    this.liveBytesValue -= this.sizes.get(buffer) ?? 0;
    this.sizes.delete(buffer);
  }

  destroyAll(): void {
    for (const buffer of [...this.live]) this.destroy(buffer);
  }

  receipt(): Readonly<Record<string, number>> {
    return Object.freeze({ createdBufferCount: this.created,
      destroyedBufferCount: this.destroyed,
      liveBufferCount: this.live.size,
      liveBytes: this.liveBytesValue,
      maximumLiveBytes: this.maximumLiveBytesValue,
      mapCount: this.maps, unmapCount: this.unmaps,
      activeMapCount: this.activeMaps });
  }
}

function requireTimestampSamples(
  samples: readonly Opt0081MlpTimestampSample[],
  label: string,
): void {
  if (samples.length !== TIMING_ROUNDS.length || samples.some((sample) =>
    !Number.isFinite(sample.gpuMilliseconds) || sample.gpuMilliseconds <= 0 ||
    !Number.isFinite(sample.wallMilliseconds) || sample.wallMilliseconds <= 0 ||
    !Number.isSafeInteger(sample.gpuElapsedNanoseconds) ||
    sample.gpuElapsedNanoseconds <= 0 || sample.commandBufferCount !== 1 ||
    sample.queueDrainCount !== 1 || sample.timestampResolveCount !== 1 ||
    sample.timestampCopyCount !== 1)) {
    throw new Error(`OPT-0081 ${label} requires eight valid timestamp samples`);
  }
}

function mean(values: readonly number[]): number {
  if (values.length !== TIMING_ROUNDS.length || values.some((value) =>
    !Number.isFinite(value) || value <= 0)) {
    throw new Error("OPT-0081 mean requires eight finite positive samples");
  }
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function median(values: readonly number[]): number {
  if (values.length !== TIMING_ROUNDS.length || values.some((value) =>
    !Number.isFinite(value) || value <= 0)) {
    throw new Error("OPT-0081 median requires eight finite positive samples");
  }
  const sorted = [...values].sort((left, right) => left - right);
  return (sorted[3]! + sorted[4]!) / 2;
}

function requiredParameter(parameters: URLSearchParams, name: string): string {
  const value = parameters.get(name);
  if (value === null || value.trim() === "") {
    throw new Error(`OPT-0081 field ${name} is missing`);
  }
  return value.trim();
}

function requiredFiniteParameter(
  parameters: URLSearchParams,
  name: string,
): number {
  const value = Number(requiredParameter(parameters, name));
  if (!Number.isFinite(value)) {
    throw new Error(`OPT-0081 field ${name} is invalid`);
  }
  return value;
}

function requiredIntegerParameter(
  parameters: URLSearchParams,
  name: string,
): number {
  const value = requiredFiniteParameter(parameters, name);
  if (!Number.isSafeInteger(value)) {
    throw new Error(`OPT-0081 field ${name} must be an integer`);
  }
  return value;
}

function maximumStorageBindingBytes(): number {
  return Math.max(
    ROWS * INTERMEDIATE * 4,
    HIDDEN * INTERMEDIATE * 2,
    ROWS * HIDDEN * 4,
  );
}

function requireAdapter(adapter: GPUAdapter): void {
  const maximumStorageBytes = maximumStorageBindingBytes();
  if (!adapter.features.has("shader-f16") ||
    !adapter.features.has("subgroups") ||
    !adapter.features.has("timestamp-query") ||
    Number(adapter.info.subgroupMinSize) !== 32 ||
    Number(adapter.info.subgroupMaxSize) !== 32 ||
    adapter.limits.maxComputeInvocationsPerWorkgroup < 256 ||
    adapter.limits.maxComputeWorkgroupSizeX < 256 ||
    adapter.limits.maxComputeWorkgroupStorageSize < 16_384 ||
    adapter.limits.maxStorageBufferBindingSize < maximumStorageBytes ||
    adapter.limits.maxBufferSize < maximumStorageBytes +
      2 * STORAGE_GUARD_BYTES) {
    throw new Error(
      "OPT-0081 requires timestamp-query, shader-f16, fixed32 subgroups, WG256, 16KiB workgroup storage, and full-shape storage limits",
    );
  }
}

function binding(buffer: GPUBuffer, size: number): GPUBufferBinding {
  return Object.freeze({ buffer, offset: 0, size });
}

export function numberToFloat16Bits(value: number): number {
  F16_CAST_FLOAT[0] = value;
  const word = F16_CAST_WORD[0]!;
  const sign = (word >>> 16) & 0x8000;
  const exponent = (word >>> 23) & 0xff;
  const mantissa = word & 0x007f_ffff;
  if (exponent === 0xff) {
    return mantissa === 0 ? sign | 0x7c00 : sign | 0x7e00;
  }
  const halfExponent = exponent - 127 + 15;
  if (halfExponent >= 31) return sign | 0x7c00;
  if (halfExponent <= 0) {
    if (halfExponent < -10) return sign;
    const significand = mantissa | 0x0080_0000;
    const shift = 14 - halfExponent;
    let rounded = significand >>> shift;
    const mask = 2 ** shift - 1;
    const remainder = significand & mask;
    const halfway = 2 ** (shift - 1);
    if (remainder > halfway ||
      (remainder === halfway && (rounded & 1) !== 0)) rounded += 1;
    return sign | rounded;
  }
  let roundedMantissa = mantissa >>> 13;
  const remainder = mantissa & 0x1fff;
  if (remainder > 0x1000 ||
    (remainder === 0x1000 && (roundedMantissa & 1) !== 0)) {
    roundedMantissa += 1;
  }
  return sign | ((halfExponent << 10) + roundedMantissa);
}

function mlpProducerValue(
  index: number,
  operand: 0 | 1 | 2,
): number {
  const feature = operand === 0 ? index % HIDDEN : index;
  if (feature < 2) {
    if (operand === 1) return 0;
    return feature === 0 ? 0 : -0;
  }
  const fixtureIndex = index - ((index % HIDDEN) & 1);
  const mixed = mix32(0x243f_6a88 ^
    Math.imul(fixtureIndex + 1, 0x9e37_79b1) ^
    Math.imul(operand + 1, 0x85eb_ca6b));
  const sign = (mixed & 1) === 0 ? 1 : -1;
  if (operand === 0) {
    const boundary = [0, -0, 2 ** -24, -(2 ** -24), 0.125, -0.125,
      1 + 2 ** -11 + 2 ** -23] as const;
    return boundary[mixed % boundary.length]!;
  }
  if (operand === 1) return sign * ((mixed >>> 9) % 17) / 128;
  return sign * ((mixed >>> 13) % 17) / 256;
}

function tflops(multiplyAdds: number, milliseconds: number): number {
  if (!Number.isSafeInteger(multiplyAdds) || multiplyAdds <= 0 ||
    !Number.isFinite(milliseconds) || milliseconds <= 0) {
    throw new Error("OPT-0081 TFLOP/s inputs must be finite and positive");
  }
  return 2 * multiplyAdds / (milliseconds * 1_000_000_000);
}

function f32Class(word: number): string {
  const absolute = word & 0x7fff_ffff;
  const negative = (word & 0x8000_0000) !== 0;
  if (absolute === 0) return negative ? "negative-zero" : "positive-zero";
  const exponent = word & 0x7f80_0000;
  const mantissa = word & 0x007f_ffff;
  if (exponent === 0x7f80_0000) {
    if (mantissa !== 0) return "nan";
    return negative ? "negative-infinity" : "positive-infinity";
  }
  return negative ? "negative-finite" : "positive-finite";
}

function mix32(value: number): number {
  value ^= value >>> 16;
  value = Math.imul(value, 0x7feb_352d);
  value ^= value >>> 15;
  value = Math.imul(value, 0x846c_a68b);
  value ^= value >>> 16;
  return value >>> 0;
}

function requireLittleEndianHost(): void {
  const words = new Uint16Array([0x0102]);
  if (new Uint8Array(words.buffer)[0] !== 0x02) {
    throw new Error("OPT-0081 fixtures require a little-endian host");
  }
}

function fieldParameters(selector: string): URLSearchParams {
  const parameters = new URLSearchParams();
  for (const input of document.querySelectorAll<HTMLInputElement>(
    `${selector} input[name]`,
  )) parameters.set(input.name, input.value);
  return parameters;
}

function requireElement<T extends Element>(selector: string): T {
  const element = document.querySelector<T>(selector);
  if (element === null) throw new Error(`Missing ${selector}`);
  return element;
}

function publish(
  receipt: Readonly<Record<string, unknown>>,
  status: "passed" | "failed",
): void {
  window.__ACE_OPT0081_MLP_RESULT__ = receipt;
  document.body.dataset.status = status;
  requireElement<HTMLElement>("#result").textContent =
    JSON.stringify(receipt, null, 2);
}

function publishFailure(
  error: unknown,
  stage: string,
  evidence: Readonly<Record<string, unknown>> = Object.freeze({}),
): void {
  const failure = error instanceof Error
    ? Object.freeze({ name: error.name, message: error.message,
        stack: error.stack })
    : Object.freeze({ message: String(error) });
  const preparationCleanup = error instanceof Opt0081MlpPreparationFailure
    ? Object.freeze({ originalName: error.originalName,
      receipt: error.setupCleanup }) : undefined;
  const structuredCorrectness = error instanceof Opt0081MlpPreparationFailure
    ? error.correctness : undefined;
  const setupCleanupPassed = error instanceof Opt0081MlpPreparationFailure &&
    cleanupReceiptPassed(error.setupCleanup);
  const structuredCleanup = structuredCorrectness === undefined ? undefined :
    Object.freeze({ receipt: error instanceof Opt0081MlpPreparationFailure
      ? error.setupCleanup : undefined, passed: setupCleanupPassed });
  const disposition = structuredCorrectness === undefined
    ? stage.startsWith("thermal")
      ? "inconclusive-invalid-thermal-or-protocol-evidence"
      : "inconclusive-invalid-correctness-or-lifecycle-evidence"
    : classifyOpt0081MlpDisposition(Object.freeze({
      inPagePassed: false, correctness: structuredCorrectness,
      cleanup: structuredCleanup,
    }));
  const receipt = Object.freeze({ schema: RECEIPT_SCHEMA,
    experiment: EXPERIMENT_ID,
    status: "failed", passed: false, stage, failure, ...evidence,
    preparationCleanup, correctness: structuredCorrectness,
    cleanup: structuredCleanup ?? evidence["cleanup"],
    decision: Object.freeze({ disposition, selectedArm: null,
      cPrimitiveQualified: false, cSelectableUnderOpt0081: false,
      representativeLayerFollowUpAuthorized: false,
      productionIntegrationAuthorized: false }),
    productionIntegrationAuthorized: false });
  publish(receipt, "failed");
  const download = document.querySelector<HTMLButtonElement>("#download");
  if (download !== null) download.disabled = false;
}

function cleanupReceiptPassed(
  cleanup: Readonly<Record<string, unknown>>,
): boolean {
  return cleanup["zeroLiveBuffers"] === true &&
    cleanup["zeroLiveBytes"] === true &&
    cleanup["createdEqualsDestroyed"] === true &&
    cleanup["mapsBalanced"] === true &&
    cleanup["postDestroyRejected"] === true &&
    cleanup["deviceDestroyed"] === true;
}

async function sha256Text(value: string): Promise<string> {
  return sha256Bytes(new TextEncoder().encode(value));
}

async function sha256U32(value: Uint32Array): Promise<string> {
  return sha256Bytes(new Uint8Array(value.buffer, value.byteOffset,
    value.byteLength));
}

async function sha256U16(value: Uint16Array): Promise<string> {
  return sha256Bytes(new Uint8Array(value.buffer, value.byteOffset,
    value.byteLength));
}

function hex(value: number, digits: 4 | 8): string {
  return `0x${value.toString(16).padStart(digits, "0")}`;
}

async function sha256Bytes(value: Uint8Array): Promise<string> {
  const copy = value.slice();
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256",
    copy.buffer));
  return [...digest].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

async function yieldToBrowser(): Promise<void> {
  await new Promise<void>((resolve) => setTimeout(resolve, 0));
}

async function settlePostDrainEvents(): Promise<void> {
  await Promise.resolve();
  await new Promise<void>((resolve) => setTimeout(resolve, 0));
}
