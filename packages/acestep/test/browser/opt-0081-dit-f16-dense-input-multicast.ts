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
import rmsNormProducerSource from
  "../../src/webgpu/kernels/rmsnorm.ts?raw";
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
  ACE_OPT_0081_DIT_F16_DENSE_INPUT_ROLES,
  ACE_OPT_0081_DIT_F16_DENSE_INPUT_STORAGE_PROFILE,
  AceOpt0081F16DenseInputProducerKernel,
  type AceOpt0081F16DenseInputRole,
} from "../../src/webgpu/kernels/dit-f16-dense-input-producers.js";
import { AceCorrectnessTransformerPlumbingKernel } from
  "../../src/webgpu/kernels/transformer-plumbing.js";
import { AceCorrectnessRmsNormKernel } from
  "../../src/webgpu/kernels/rmsnorm.js";
import type {
  AceGemmBufferBindings,
  AceGemmDispatch,
  AceGemmShape,
} from "../../src/webgpu/kernels/gemm.js";

declare global {
  interface Window {
    __ACE_OPT0081_RESULT__?: Readonly<Record<string, unknown>>;
  }
}

export type Opt0081Arm = "A" | "B" | "C";

export interface Opt0081ShapeSpec {
  readonly id: "h-h" | "h-1024" | "h-6144" | "6144-h";
  readonly shape: AceGemmShape;
  readonly productionMultiplicity: 4 | 2 | 1;
  readonly feedForwardMultiplicity: 0 | 2 | 1;
  readonly retainedProducerRole: AceOpt0081F16DenseInputRole;
  readonly fixtureIds: readonly string[];
  readonly ordinal: number;
}

export interface Opt0081TimestampSample {
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

export interface Opt0081TimingInput {
  readonly id: Opt0081ShapeSpec["id"];
  readonly samples: Readonly<Record<
    Opt0081Arm,
    readonly Opt0081TimestampSample[]
  >>;
}

export interface Opt0081ThermalLaunchGate {
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

export interface Opt0081ThermalCompletion {
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

interface GuardedOutput {
  readonly buffer: GPUBuffer;
  readonly binding: GPUBufferBinding;
  readonly prefill: GPUBuffer;
  readonly readback: GPUBuffer;
  readonly outputElements: number;
  readonly outputBytes: number;
  readonly totalBytes: number;
  readonly columns: number;
}

interface OutputSnapshot {
  readonly words: Uint32Array<ArrayBuffer>;
  readonly sha256: string;
  readonly nonFiniteCount: number;
  readonly qNaNPrefillCount: number;
  readonly prefixGuardIntact: boolean;
  readonly suffixGuardIntact: boolean;
  readonly adjacentBeforeIntact: boolean;
  readonly adjacentAfterIntact: boolean;
  readonly firstValidWritten: boolean;
  readonly lastValidWritten: boolean;
  readonly partialMTailWritten: boolean;
}

interface PreparedShape {
  readonly spec: Opt0081ShapeSpec;
  readonly activationF32: GPUBufferBinding;
  readonly activationF16: GPUBufferBinding;
  readonly weight: GPUBuffer;
  readonly output: GuardedOutput;
  readonly dispatches: Readonly<Record<Opt0081Arm, AceGemmDispatch>>;
  readonly correctness: Readonly<Record<string, unknown>>;
}

interface PreparedHarness {
  readonly adapter: GPUAdapter;
  readonly device: GPUDevice;
  readonly tracker: BufferTracker;
  readonly armAKernel: AceOpt0009DenseGemmKernel;
  readonly armBKernel: AceOpt0081DenseF16InputKernel;
  readonly armCKernel: AceOpt0081DenseF16InputWeightMulticastKernel;
  readonly currentTransformerProducer: AceCorrectnessTransformerPlumbingKernel;
  readonly currentRmsNormProducer: AceCorrectnessRmsNormKernel;
  readonly f16Producer: AceOpt0081F16DenseInputProducerKernel;
  readonly querySet: GPUQuerySet;
  readonly queryResolve: GPUBuffer;
  readonly queryReadback: GPUBuffer;
  readonly shapes: readonly PreparedShape[];
  readonly correctness: Readonly<Record<string, unknown>>;
  readonly identity: Readonly<Record<string, unknown>>;
  readonly warmup: Readonly<Record<string, unknown>>;
  readonly readyAtEpochMilliseconds: number;
  readonly uncapturedErrors: readonly string[];
  readonly deviceLosses: readonly string[];
  cleanup(): Promise<Readonly<Record<string, unknown>>>;
}

class Opt0081PreparationFailure extends Error {
  readonly originalName: string;
  readonly correctness: Readonly<Record<string, unknown>> | undefined;

  constructor(error: unknown,
    readonly setupCleanup: Readonly<Record<string, unknown>>) {
    super(error instanceof Error ? error.message : String(error));
    this.name = "Opt0081PreparationFailure";
    this.originalName = error instanceof Error ? error.name : typeof error;
    this.correctness = error instanceof Opt0081CorrectnessGateFailure
      ? error.correctness : undefined;
  }
}

class Opt0081CorrectnessGateFailure extends Error {
  constructor(readonly correctness: Readonly<Record<string, unknown>>,
    message: string) {
    super(message);
    this.name = "Opt0081CorrectnessGateFailure";
  }
}

const EXPERIMENT_ID = "OPT-0081" as const;
const RECEIPT_SCHEMA = "ace-opt-0081-f16-dense-input-multicast-primitive-v1";
const REGISTRATION_COMMIT =
  "70a5e4a29c5455ec00a4b757dcdf5cdcc70a5e91";
const ROWS = 2_250;
const STORAGE_GUARD_BYTES = 256;
const STORAGE_GUARD_U32 = 0xa55a_c33c;
const OUTPUT_PREFILL_QNAN_U32 = 0x7fc0_7855;
const STORAGE_GUARD_U16 = 0x55aa;
const OUTPUT_PREFILL_QNAN_U16 = 0x7e55;
const F16_CAST_SCRATCH = new ArrayBuffer(4);
const F16_CAST_FLOAT = new Float32Array(F16_CAST_SCRATCH);
const F16_CAST_WORD = new Uint32Array(F16_CAST_SCRATCH);
const TIMESTAMP_QUERY_BYTES = 16;
const REQUIRED_B_OVER_A_SPEEDUP = 1.05;
const REQUIRED_B_OVER_A_SAVING_MILLISECONDS = 10.4167;
const REQUIRED_C_OVER_A_SPEEDUP = 1.12;
const REQUIRED_C_OVER_A_SAVING_MILLISECONDS = 20.8334;
const REQUIRED_C_OVER_B_SAVING_MILLISECONDS = 10.4167;
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

const SHAPE_SPECS = Object.freeze([
  shapeSpec("h-h", 2_048, 2_048, 4, 0, "selfMergedAttention", [
    "signed-zero", "subnormal-normal-boundary",
  ]),
  shapeSpec("h-1024", 2_048, 1_024, 2, 0, "selfModulated", [
    "alternating-cancellation",
  ]),
  shapeSpec("h-6144", 2_048, 6_144, 2, 2, "mlpModulated", [
    "maximum-finite-fp16-bounded",
  ]),
  shapeSpec("6144-h", 6_144, 2_048, 1, 1, "gatedActivation", [
    "activation-rounding-boundary",
  ]),
] as const);

const ARM_ORDERS = Object.freeze([
  ["A", "B", "C"], ["C", "B", "A"], ["B", "C", "A"],
  ["A", "C", "B"], ["C", "A", "B"], ["B", "A", "C"],
  ["A", "B", "C"], ["C", "B", "A"],
] as const);

const TIMING_ROUNDS = Object.freeze(Array.from({ length: 8 }, (_, index) => {
  const base = [0, 1, 2, 3] as const;
  const rotation = index % base.length;
  const shapeOrder = Object.freeze([
    ...base.slice(rotation), ...base.slice(0, rotation),
  ]);
  const armOrder: readonly Opt0081Arm[] = ARM_ORDERS[index]!;
  return Object.freeze({ roundIndex: index, shapeOrder, armOrder });
}));

export function buildOpt0081ShapeSpecs(): readonly Opt0081ShapeSpec[] {
  return SHAPE_SPECS;
}

export function buildOpt0081TimingRounds(): typeof TIMING_ROUNDS {
  return TIMING_ROUNDS;
}

export function summarizeOpt0081Timing(
  inputs: readonly Opt0081TimingInput[],
): Readonly<Record<string, unknown>> {
  if (inputs.length !== SHAPE_SPECS.length) {
    throw new Error("OPT-0081 timing requires all four exact M2250 shapes");
  }
  const strata = inputs.map((input, index) => {
    const spec = SHAPE_SPECS[index];
    if (spec === undefined || input.id !== spec.id) {
      throw new Error("OPT-0081 timing shape order changed");
    }
    requireTimestampSamples(input.samples.A, `${spec.id} A`);
    requireTimestampSamples(input.samples.B, `${spec.id} B`);
    requireTimestampSamples(input.samples.C, `${spec.id} C`);
    const summarizeArm = (samples: readonly Opt0081TimestampSample[]) => {
      const gpu = samples.map(({ gpuMilliseconds }) => gpuMilliseconds);
      const wall = samples.map(({ wallMilliseconds }) => wallMilliseconds);
      return Object.freeze({ samples, meanGpuMilliseconds: mean(gpu),
        medianGpuMilliseconds: median(gpu), meanWallMilliseconds: mean(wall),
        medianWallMilliseconds: median(wall), minimumGpuMilliseconds: Math.min(...gpu),
        maximumGpuMilliseconds: Math.max(...gpu), minimumWallMilliseconds: Math.min(...wall),
        maximumWallMilliseconds: Math.max(...wall) });
    };
    const A = summarizeArm(input.samples.A);
    const B = summarizeArm(input.samples.B);
    const C = summarizeArm(input.samples.C);
    const faster = (candidate: typeof A, baseline: typeof A) => Object.freeze({
      meanGpuFaster: candidate.meanGpuMilliseconds < baseline.meanGpuMilliseconds,
      medianGpuFaster: candidate.medianGpuMilliseconds < baseline.medianGpuMilliseconds,
      meanWallFaster: candidate.meanWallMilliseconds < baseline.meanWallMilliseconds,
      medianWallFaster: candidate.medianWallMilliseconds < baseline.medianWallMilliseconds,
    });
    const gates = Object.freeze({ bOverA: faster(B, A), cOverA: faster(C, A),
      cOverB: faster(C, B) });
    return Object.freeze({ id: spec.id, shape: spec.shape,
      productionMultiplicity: spec.productionMultiplicity,
      feedForwardMultiplicity: spec.feedForwardMultiplicity,
      retainedProducerRole: spec.retainedProducerRole,
      arms: Object.freeze({ A, B, C }), gates,
      bOverAPassed: Object.values(gates.bOverA).every(Boolean),
      cOverAPassed: Object.values(gates.cOverA).every(Boolean),
      cOverBPassed: Object.values(gates.cOverB).every(Boolean) });
  });
  const weightedRounds = Array.from({ length: TIMING_ROUNDS.length },
    (_, roundIndex) => {
      const score = (arm: Opt0081Arm, metric: "gpuMilliseconds" |
        "wallMilliseconds", multiplicity: "productionMultiplicity" |
        "feedForwardMultiplicity") => inputs.reduce((sum, input, index) => {
          const sample = input.samples[arm][roundIndex];
          const spec = SHAPE_SPECS[index];
          if (sample === undefined || spec === undefined) {
            throw new Error("OPT-0081 weighted round sample missing");
          }
          return sum + sample[metric] * spec[multiplicity];
        }, 0);
      const scope = (multiplicity: "productionMultiplicity" |
        "feedForwardMultiplicity") => Object.freeze({
          A: Object.freeze({ gpu: score("A", "gpuMilliseconds", multiplicity),
            wall: score("A", "wallMilliseconds", multiplicity) }),
          B: Object.freeze({ gpu: score("B", "gpuMilliseconds", multiplicity),
            wall: score("B", "wallMilliseconds", multiplicity) }),
          C: Object.freeze({ gpu: score("C", "gpuMilliseconds", multiplicity),
            wall: score("C", "wallMilliseconds", multiplicity) }),
        });
      return Object.freeze({ roundIndex, complete: scope("productionMultiplicity"),
        feedForward: scope("feedForwardMultiplicity") });
    });
  const aggregateArms = (scope: "complete" | "feedForward") => {
    const arm = (name: Opt0081Arm) => {
      const gpu = weightedRounds.map((round) => round[scope][name].gpu);
      const wall = weightedRounds.map((round) => round[scope][name].wall);
      return Object.freeze({ samples: Object.freeze({ gpu, wall }),
        meanGpuMilliseconds: mean(gpu), medianGpuMilliseconds: median(gpu),
        meanWallMilliseconds: mean(wall), medianWallMilliseconds: median(wall) });
    };
    return Object.freeze({ A: arm("A"), B: arm("B"), C: arm("C") });
  };
  const completeArms = aggregateArms("complete");
  const feedForwardArms = aggregateArms("feedForward");
  const comparison = (candidate: "B" | "C", baseline: "A" | "B") => {
    const rounds = weightedRounds.map((round) => Object.freeze({
      baselineGpu: round.complete[baseline].gpu,
      candidateGpu: round.complete[candidate].gpu,
      baselineWall: round.complete[baseline].wall,
      candidateWall: round.complete[candidate].wall,
    }));
    const summarizeMetric = (metric: "Gpu" | "Wall") => {
      const baselineValues = rounds.map((round) => round[`baseline${metric}`]);
      const candidateValues = rounds.map((round) => round[`candidate${metric}`]);
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
    const medianWallGpuSavingRatio = wall.medianSavingMilliseconds /
      gpu.medianSavingMilliseconds;
    return Object.freeze({ baseline, candidate, rounds: Object.freeze(rounds),
      gpu, wall, meanWallGpuSavingRatio, medianWallGpuSavingRatio,
      wallGpuSavingsAgree: meanWallGpuSavingRatio >= 0.75 &&
        meanWallGpuSavingRatio <= 1.25,
      projectedMeanGpuSavingMilliseconds:
        gpu.meanSavingMilliseconds * PROJECTION_MULTIPLIER,
      projectedMeanWallSavingMilliseconds:
        wall.meanSavingMilliseconds * PROJECTION_MULTIPLIER });
  };
  const bOverA = comparison("B", "A");
  const cOverA = comparison("C", "A");
  const cOverB = comparison("C", "B");
  const paired = (entry: typeof bOverA) => entry.gpu.pairWins >=
    REQUIRED_PAIR_WINS && entry.wall.pairWins >= REQUIRED_PAIR_WINS;
  const bStandalonePassed = strata.every((entry) => entry.bOverAPassed) &&
    paired(bOverA) && bOverA.gpu.meanSpeedup >= REQUIRED_B_OVER_A_SPEEDUP &&
    bOverA.wall.meanSpeedup >= REQUIRED_B_OVER_A_SPEEDUP &&
    bOverA.gpu.meanSavingMilliseconds >=
      REQUIRED_B_OVER_A_SAVING_MILLISECONDS &&
    bOverA.wall.meanSavingMilliseconds >=
      REQUIRED_B_OVER_A_SAVING_MILLISECONDS && bOverA.wallGpuSavingsAgree;
  const cPassed = strata.every((entry) => entry.cOverAPassed &&
      entry.cOverBPassed) && paired(cOverA) && paired(cOverB) &&
    cOverA.gpu.meanSpeedup >= REQUIRED_C_OVER_A_SPEEDUP &&
    cOverA.wall.meanSpeedup >= REQUIRED_C_OVER_A_SPEEDUP &&
    cOverA.gpu.meanSavingMilliseconds >=
      REQUIRED_C_OVER_A_SAVING_MILLISECONDS &&
    cOverA.wall.meanSavingMilliseconds >=
      REQUIRED_C_OVER_A_SAVING_MILLISECONDS &&
    cOverB.gpu.meanSavingMilliseconds >=
      REQUIRED_C_OVER_B_SAVING_MILLISECONDS &&
    cOverB.wall.meanSavingMilliseconds >=
      REQUIRED_C_OVER_B_SAVING_MILLISECONDS && cOverA.wallGpuSavingsAgree &&
    cOverB.wallGpuSavingsAgree;
  const gates = Object.freeze({ requiredPairWins: REQUIRED_PAIR_WINS,
    bStandalone: Object.freeze({ requiredSpeedup: REQUIRED_B_OVER_A_SPEEDUP,
      requiredSavingMilliseconds: REQUIRED_B_OVER_A_SAVING_MILLISECONDS,
      passed: bStandalonePassed }),
    c: Object.freeze({ requiredOverASpeedup: REQUIRED_C_OVER_A_SPEEDUP,
      requiredOverASavingMilliseconds: REQUIRED_C_OVER_A_SAVING_MILLISECONDS,
      requiredOverBSavingMilliseconds: REQUIRED_C_OVER_B_SAVING_MILLISECONDS,
      passed: cPassed }) });
  const selectedArm: "B" | "C" | null = cPassed ? "C" :
    bStandalonePassed ? "B" : null;
  return Object.freeze({ samplesPerArmPerShape: TIMING_ROUNDS.length,
    multiplicities: Object.freeze({ complete: "4/2/2/1",
      feedForward: "0/0/2/1" }), strata: Object.freeze(strata),
    weightedRounds: Object.freeze(weightedRounds),
    aggregateArms: Object.freeze({ complete: completeArms,
      feedForward: feedForwardArms }),
    comparisons: Object.freeze({ bOverA, cOverA, cOverB }), gates,
    bStandalonePassed, cPassed, selectedArm,
    cPreferredWhenPassing: true, passed: selectedArm !== null });
}

export function parseOpt0081ThermalLaunchGate(
  parameters: URLSearchParams,
  readyAtEpochMilliseconds: number,
  launchedAtEpochMilliseconds: number,
): Opt0081ThermalLaunchGate {
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

export function parseOpt0081ThermalCompletion(
  parameters: URLSearchParams,
  launch: Opt0081ThermalLaunchGate,
  cleanupCompletedAtEpochMilliseconds: number,
): Opt0081ThermalCompletion {
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
        "READY — six producer audits, four A/A/B/B/C/C gates, and symmetric warmup passed; timing has not run";
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
    let thermalLaunch: Opt0081ThermalLaunchGate;
    try {
      thermalLaunch = parseOpt0081ThermalLaunchGate(
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
      Opt0081ThermalLaunchGate;
    try {
      const thermalCompletion = parseOpt0081ThermalCompletion(
        fieldParameters("#thermal-completion"),
        thermalLaunch,
        cleanupCompletedAtEpochMilliseconds,
      );
      const passed = pending["inPagePassed"] === true;
      const disposition = classifyOpt0081PrimitiveDisposition(pending);
      const timing = pending["timing"] as Readonly<Record<string, unknown>>;
      const selectedArm = timing["selectedArm"];
      const receipt = Object.freeze({ ...pending, schema: RECEIPT_SCHEMA,
        experiment: EXPERIMENT_ID, status: "completed", passed,
        thermal: Object.freeze({ launch: thermalLaunch,
          completion: thermalCompletion, passed: true }),
        decision: Object.freeze({
          disposition,
          selectedArm,
          cPreferredWhenPassing: true,
          cPassed: timing["cPassed"] === true,
          bStandalonePassed: timing["bStandalonePassed"] === true,
          diagnosticProfileFollowUpAuthorized: passed,
          productionIntegrationAuthorized: false,
          packageChangeAuthorized: false,
          trajectoryOrListeningClaim: false,
          unchangedTimingRetryPerformed: false,
        }) });
      latestReceipt = receipt;
      publish(receipt, passed ? "passed" : "failed");
      progress.textContent = passed
        ? `completed — typed-F16 ${selectedArm === "C" ? "multicast" : "current-geometry"} primitive gate passed; production remains unchanged`
        : "completed — typed-F16 multicast stopped at its frozen gate";
      pending = undefined;
      finalizeButton.disabled = true;
      completionFields.disabled = true;
      downloadButton.disabled = false;
    } catch (error) {
      publishFailure(error, "thermal-completion", { pending });
    }
  });

  downloadButton.addEventListener("click", () => {
    const receipt = latestReceipt ?? window.__ACE_OPT0081_RESULT__;
    if (receipt === undefined) return;
    const blob = new Blob([JSON.stringify(receipt)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "opt-0081-dit-f16-dense-input-multicast-receipt.json";
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
    label: "ace-opt-0081-f16-dense-input-multicast-profile-device",
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
  const owners = (() => {
    let partialA: AceOpt0009DenseGemmKernel | undefined;
    let partialB: AceOpt0081DenseF16InputKernel | undefined;
    let partialC: AceOpt0081DenseF16InputWeightMulticastKernel | undefined;
    let partialCurrentTransformer: AceCorrectnessTransformerPlumbingKernel |
      undefined;
    let partialCurrentRmsNorm: AceCorrectnessRmsNormKernel | undefined;
    let partialF16Producer: AceOpt0081F16DenseInputProducerKernel | undefined;
    let partialQuerySet: GPUQuerySet | undefined;
    try {
      partialA = AceOpt0009DenseGemmKernel.create(device, capability);
      partialB = AceOpt0081DenseF16InputKernel.create(device, capability);
      partialC = AceOpt0081DenseF16InputWeightMulticastKernel.create(
        device, capability,
      );
      partialCurrentTransformer = AceCorrectnessTransformerPlumbingKernel
        .create(device, "reference-bf16");
      partialCurrentRmsNorm = AceCorrectnessRmsNormKernel.create(
        device, "reference-bf16",
      );
      partialF16Producer = AceOpt0081F16DenseInputProducerKernel.create(device);
      partialQuerySet = device.createQuerySet({
        label: "opt0081-dense-pass-timestamps", type: "timestamp", count: 2,
      });
      const queryResolve = tracker.create(device, {
        label: "opt0081-timestamp-resolve", size: TIMESTAMP_QUERY_BYTES,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
      });
      const queryReadback = tracker.create(device, {
        label: "opt0081-timestamp-readback", size: TIMESTAMP_QUERY_BYTES,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      return Object.freeze({ armAKernel: partialA, armBKernel: partialB,
        armCKernel: partialC,
        currentTransformerProducer: partialCurrentTransformer,
        currentRmsNormProducer: partialCurrentRmsNorm,
        f16Producer: partialF16Producer, querySet: partialQuerySet,
        queryResolve, queryReadback });
    } catch (error) {
      partialC?.destroy();
      partialB?.destroy();
      partialA?.destroy();
      partialF16Producer?.destroy();
      partialCurrentTransformer?.destroy();
      partialCurrentRmsNorm?.destroy();
      partialQuerySet?.destroy();
      tracker.destroyAll();
      device.destroy();
      throw error;
    }
  })();
  const { armAKernel, armBKernel, armCKernel, currentTransformerProducer,
    currentRmsNormProducer, f16Producer, querySet, queryResolve,
    queryReadback } = owners;
  let cleanupPromise: Promise<Readonly<Record<string, unknown>>> | undefined;
  let firstShape: PreparedShape | undefined;
  const cleanup = (): Promise<Readonly<Record<string, unknown>>> => {
    if (cleanupPromise !== undefined) {
      return cleanupPromise.then((receipt) => Object.freeze({ ...receipt,
        idempotent: true, repeatedCall: true }));
    }
    cleanupPromise = (async () => {
      const cleanupStartedAtEpochMilliseconds = Date.now();
      await device.queue.onSubmittedWorkDone().catch(() => undefined);
      let armCPostDestroyRejected = false;
      let armBPostDestroyRejected = false;
      let producerPostDestroyRejected = false;
      const rejectionSpec = firstShape?.spec ?? SHAPE_SPECS[0]!;
      const rejectionActivation = firstShape?.activationF16 ??
        binding(queryResolve, TIMESTAMP_QUERY_BYTES);
      const rejectionWeight = firstShape === undefined
        ? binding(queryResolve, TIMESTAMP_QUERY_BYTES)
        : binding(firstShape.weight,
          rejectionSpec.shape.inner * rejectionSpec.shape.columns * 2);
      const rejectionOutput = firstShape?.output.binding ??
        binding(queryResolve, TIMESTAMP_QUERY_BYTES);
      armCKernel.destroy();
      try {
        await armCKernel.createDispatch(
          "opt0081-post-destroy-rejection", rejectionSpec.shape,
          Object.freeze({ activation: rejectionActivation,
            weight: rejectionWeight, output: rejectionOutput }),
        );
      } catch {
        armCPostDestroyRejected = true;
      }
      armBKernel.destroy();
      try {
        await armBKernel.createDispatch(
          "opt0081-B-post-destroy-rejection", rejectionSpec.shape,
          Object.freeze({ activation: rejectionActivation,
            weight: rejectionWeight, output: rejectionOutput }),
        );
      } catch {
        armBPostDestroyRejected = true;
      }
      armAKernel.destroy();
      f16Producer.destroy();
      try {
        await f16Producer.createSelfMergedAttentionDispatch(
          "opt0081-producer-post-destroy-rejection",
          "selfMergedAttention", "merge-heads",
          Object.freeze({ batch: 1, tokens: ROWS, heads: 16,
            headDimension: 128 }),
          Object.freeze({ input: firstShape?.activationF32 ??
            binding(queryResolve, TIMESTAMP_QUERY_BYTES),
            output: rejectionActivation }),
        );
      } catch {
        producerPostDestroyRejected = true;
      }
      currentTransformerProducer.destroy();
      currentRmsNormProducer.destroy();
      querySet.destroy();
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
        deviceDestroyed: true,
        idempotent: true, repeatedCall: false });
    })();
    return cleanupPromise;
  };
  try {
    const identity = await buildIdentity(adapter, device);
    update("auditing all six full-M2250 F32 producers against direct F16 stores");
    const producerAudit = await auditAllProducers(device, tracker,
      currentTransformerProducer, currentRmsNormProducer, f16Producer,
      update);
    await settlePostDrainEvents();
    if (producerAudit.receipt["passed"] !== true) {
      throw new Opt0081CorrectnessGateFailure(Object.freeze({
        producerAudit: producerAudit.receipt,
        cases: Object.freeze([]),
        uncapturedGpuErrorCount: uncapturedErrors.length,
        deviceLossCount: deviceLosses.length,
        completedBeforeReady: true, stoppedBeforeDenseAndTiming: true,
        passed: false,
      }), "OPT-0081 full producer raw-U16 audit failed");
    }
    const preparationOrder = [2, 3, 0, 1] as const;
    const preparedByIndex = new Map<number, PreparedShape>();
    for (const [position, specIndex] of preparationOrder.entries()) {
      const spec = SHAPE_SPECS[specIndex]!;
      update(`full-output A,A,B,B,C,C ${position + 1}/4: ${spec.id}`);
      const retained = producerAudit.retained.get(spec.retainedProducerRole);
      if (retained === undefined) {
        throw new Error(`OPT-0081 missing retained ${spec.retainedProducerRole}`);
      }
      const prepared = await prepareShape(device, tracker, armAKernel,
        armBKernel, armCKernel, spec, retained);
      firstShape ??= prepared;
      preparedByIndex.set(specIndex, prepared);
      await yieldToBrowser();
    }
    const shapes = SHAPE_SPECS.map((_, index) => preparedByIndex.get(index)!);
    const cases = shapes.map(({ correctness }) => correctness);
    const comparedU32Count = shapes.reduce((sum, { spec }) =>
      sum + spec.shape.rows * spec.shape.columns, 0);
    const fixtureIds = shapes.flatMap(({ spec }) => spec.fixtureIds);
    const correctness = Object.freeze({
      fullShapeCount: shapes.length,
      comparedU32CountPerComparison: comparedU32Count,
      comparisonCount: 24,
      comparisonsPerOutputWord: 6,
      producerAudit: producerAudit.receipt,
      boundedAdversarialFixtureIds: Object.freeze(fixtureIds),
      boundedAdversarialFixtureCount: fixtureIds.length,
      everyOutputRawU32Exact: cases.every((entry) =>
        entry["allSixRawU32Exact"] === true),
      everyArmRerunExact: cases.every((entry) =>
        entry["everyArmRerunExact"] === true),
      directBCRawU32Exact: cases.every((entry) =>
        entry["directBCRawU32Exact"] === true),
      allOutputsFiniteAndComplete: cases.every((entry) =>
        entry["outputsFiniteAndComplete"] === true),
      finiteClassIdentity: cases.every((entry) =>
        entry["finiteClassIdentity"] === true),
      guardsAndPartialMTailsIntact: cases.every((entry) =>
        entry["guardsAndPartialMTailsIntact"] === true),
      cases: Object.freeze(cases),
      uncapturedGpuErrorCount: uncapturedErrors.length,
      deviceLossCount: deviceLosses.length,
      completedBeforeReady: true,
      passed: cases.every((entry) => entry["passed"] === true) &&
        producerAudit.receipt["passed"] === true &&
        uncapturedErrors.length === 0 && deviceLosses.length === 0,
    });
    if (!correctness.passed) {
      throw new Opt0081CorrectnessGateFailure(correctness,
        "OPT-0081 full-output raw-U32 correctness gate failed");
    }
    update("symmetrically warming A/B/C over all four production shapes");
    const warmupStartedAtEpochMilliseconds = Date.now();
    for (const [index, shape] of shapes.entries()) {
      const order: readonly Opt0081Arm[] = ARM_ORDERS[index]!;
      for (const arm of order) {
        await executeAndDrain(device, shape.dispatches[arm]);
      }
    }
    await settlePostDrainEvents();
    const warmupCompletedAtEpochMilliseconds = Date.now();
    if (uncapturedErrors.length !== 0 || deviceLosses.length !== 0) {
      throw new Error("OPT-0081 GPU failure during warmup");
    }
    const readyAtEpochMilliseconds = Date.now();
    return Object.freeze({ adapter, device, tracker, armAKernel, armBKernel,
      armCKernel, currentTransformerProducer, currentRmsNormProducer,
      f16Producer, querySet, queryResolve, queryReadback,
      shapes: Object.freeze(shapes), correctness, identity,
      warmup: Object.freeze({ warmupStartedAtEpochMilliseconds,
        warmupCompletedAtEpochMilliseconds, armExecutions: 12,
        oneUntimedWarmupPerArmPerShape: true, completedBeforeReady: true }),
      readyAtEpochMilliseconds, uncapturedErrors, deviceLosses, cleanup });
  } catch (error) {
    const setupCleanup = await cleanup();
    throw new Opt0081PreparationFailure(error, setupCleanup);
  }
}

interface RetainedProducerPair {
  readonly role: AceOpt0081F16DenseInputRole;
  readonly f32: GPUBuffer;
  readonly f16: GPUBuffer;
  readonly f32Binding: GPUBufferBinding;
  readonly f16Binding: GPUBufferBinding;
  readonly elements: number;
  readonly receipt: Readonly<Record<string, unknown>>;
}

interface EncodableDispatch {
  readonly label: string;
  encode(pass: GPUComputePassEncoder): void;
}

async function auditAllProducers(
  device: GPUDevice,
  tracker: BufferTracker,
  currentTransformer: AceCorrectnessTransformerPlumbingKernel,
  currentRmsNorm: AceCorrectnessRmsNormKernel,
  f16Producer: AceOpt0081F16DenseInputProducerKernel,
  update: (message: string) => void,
): Promise<Readonly<{ retained: ReadonlyMap<AceOpt0081F16DenseInputRole,
  RetainedProducerPair>; receipt: Readonly<Record<string, unknown>> }>> {
  const retained = new Map<AceOpt0081F16DenseInputRole,
    RetainedProducerPair>();
  const receipts: Readonly<Record<string, unknown>>[] = [];
  const retainRoles = new Set<AceOpt0081F16DenseInputRole>(SHAPE_SPECS.map(
    ({ retainedProducerRole }) => retainedProducerRole,
  ));
  for (const [index, role] of ACE_OPT_0081_DIT_F16_DENSE_INPUT_ROLES.entries()) {
    update(`producer raw-U16 audit ${index + 1}/6: ${role}`);
    const pair = await auditProducerRole(device, tracker, currentTransformer,
      currentRmsNorm, f16Producer, role);
    receipts.push(pair.receipt);
    if (retainRoles.has(role)) {
      retained.set(role, pair);
    } else {
      tracker.destroy(pair.f32);
      tracker.destroy(pair.f16);
    }
    await yieldToBrowser();
  }
  const comparedU16Count = receipts.reduce((sum, entry) =>
    sum + Number(entry["comparedU16Count"]), 0);
  return Object.freeze({ retained,
    receipt: Object.freeze({
      profile: ACE_OPT_0081_DIT_F16_DENSE_INPUT_STORAGE_PROFILE,
      producerKernelSetId:
        ACE_OPT_0081_DIT_F16_DENSE_INPUT_PRODUCER_KERNEL_SET_ID,
      exactFullM2250Roles: ACE_OPT_0081_DIT_F16_DENSE_INPUT_ROLES,
      roleCount: receipts.length, retainedRoleCount: retained.size,
      comparedU16Count, currentProfile: "reference-bf16",
      f32InputsAndArithmetic: true, singleFinalF16Store: true,
      gpuConversionDispatchCount: 0, candidateU16ComparedToIndependentCast: true,
      cases: Object.freeze(receipts),
      passed: receipts.length === 6 && retained.size === 4 &&
        receipts.every((entry) => entry["passed"] === true),
    }) });
}

async function auditProducerRole(
  device: GPUDevice,
  tracker: BufferTracker,
  currentTransformer: AceCorrectnessTransformerPlumbingKernel,
  currentRmsNorm: AceCorrectnessRmsNormKernel,
  f16Producer: AceOpt0081F16DenseInputProducerKernel,
  role: AceOpt0081F16DenseInputRole,
): Promise<RetainedProducerPair> {
  const elements = role === "gatedActivation" ? ROWS * 6_144 : ROWS * 2_048;
  const f32Output = createGuardedProducerOutput(device, tracker, role,
    elements, 4);
  const f16Output = createGuardedProducerOutput(device, tracker, role,
    elements, 2);
  const inputs: GPUBuffer[] = [];
  const makeInput = (name: string, count: number, operand: number) => {
    const buffer = tracker.create(device, { label: `opt0081-${role}-${name}`,
      size: count * 4, usage: GPUBufferUsage.STORAGE,
      mappedAtCreation: true });
    const values = new Float32Array(buffer.getMappedRange());
    for (let index = 0; index < values.length; index += 1) {
      values[index] = producerValue(role, index, operand);
    }
    tracker.unmap(buffer);
    inputs.push(buffer);
    return binding(buffer, count * 4);
  };
  let current: EncodableDispatch;
  let candidate: EncodableDispatch;
  try {
    if (role === "selfModulated" || role === "mlpModulated") {
      const normalized = makeInput("normalized", elements, 0);
      const scale = makeInput("scale", 2_048, 1);
      const shift = makeInput("shift", 2_048, 2);
      const shape = Object.freeze({ batch: 1, tokens: ROWS, width: 2_048 });
      current = await currentTransformer.createAdaLnDispatch(
        `opt0081-${role}-current`, shape,
        { normalized, scale, shift, output: f32Output.binding },
      );
      candidate = role === "selfModulated"
        ? await f16Producer.createSelfModulatedDispatch(
          `opt0081-${role}-candidate`, role, "adaln", shape,
          { normalized, scale, shift, output: f16Output.binding })
        : await f16Producer.createMlpModulatedDispatch(
          `opt0081-${role}-candidate`, role, "adaln", shape,
          { normalized, scale, shift, output: f16Output.binding });
    } else if (role === "selfMergedAttention" ||
      role === "crossMergedAttention") {
      const input = makeInput("input", elements, 0);
      const shape = Object.freeze({ batch: 1, tokens: ROWS, heads: 16,
        headDimension: 128 });
      current = await currentTransformer.createHeadTransformDispatch(
        `opt0081-${role}-current`, "merge-heads", shape,
        { input, output: f32Output.binding },
      );
      candidate = role === "selfMergedAttention"
        ? await f16Producer.createSelfMergedAttentionDispatch(
          `opt0081-${role}-candidate`, role, "merge-heads", shape,
          { input, output: f16Output.binding })
        : await f16Producer.createCrossMergedAttentionDispatch(
          `opt0081-${role}-candidate`, role, "merge-heads", shape,
          { input, output: f16Output.binding });
    } else if (role === "crossNormalized") {
      const input = makeInput("input", elements, 0);
      const weight = tracker.create(device, {
        label: `opt0081-${role}-packed-bf16-weight`, size: 1_024 * 4,
        usage: GPUBufferUsage.STORAGE, mappedAtCreation: true,
      });
      new Uint32Array(weight.getMappedRange()).fill(0x3f80_3f80);
      tracker.unmap(weight);
      inputs.push(weight);
      const weightBinding = binding(weight, 1_024 * 4);
      const shape = Object.freeze({ rows: ROWS, width: 2_048,
        epsilon: 1e-6 });
      current = await currentRmsNorm.createDispatch(
        `opt0081-${role}-current`, shape,
        { input, weight: weightBinding, output: f32Output.binding },
      );
      candidate = await f16Producer.createCrossNormalizedDispatch(
        `opt0081-${role}-candidate`, role, "cross-rmsnorm", shape,
        { input, weight: weightBinding, output: f16Output.binding },
      );
    } else {
      const gate = makeInput("gate", elements, 0);
      const up = makeInput("up", elements, 1);
      const shape = Object.freeze({ batch: 1, tokens: ROWS, width: 6_144 });
      current = await currentTransformer.createSwiGluDispatch(
        `opt0081-${role}-current`, shape,
        { gate, up, output: f32Output.binding },
      );
      candidate = await f16Producer.createGatedActivationDispatch(
        `opt0081-${role}-candidate`, role, "swiglu", shape,
        { gate, up, output: f16Output.binding },
      );
    }
    const receipt = await executeProducerAudit(device, tracker, role,
      elements, current, candidate, f32Output, f16Output);
    return Object.freeze({ role, f32: f32Output.buffer,
      f16: f16Output.buffer, f32Binding: f32Output.binding,
      f16Binding: f16Output.binding, elements, receipt });
  } finally {
    for (const input of inputs) tracker.destroy(input);
    tracker.destroy(f32Output.prefill);
    tracker.destroy(f32Output.readback);
    tracker.destroy(f16Output.prefill);
    tracker.destroy(f16Output.readback);
  }
}

interface GuardedProducerOutput {
  readonly buffer: GPUBuffer;
  readonly binding: GPUBufferBinding;
  readonly prefill: GPUBuffer;
  readonly readback: GPUBuffer;
  readonly totalBytes: number;
  readonly elementBytes: 2 | 4;
}

function createGuardedProducerOutput(device: GPUDevice, tracker: BufferTracker,
  role: AceOpt0081F16DenseInputRole, elements: number,
  elementBytes: 2 | 4): GuardedProducerOutput {
  const bodyBytes = elements * elementBytes;
  const totalBytes = bodyBytes + 2 * STORAGE_GUARD_BYTES;
  const prefill = tracker.create(device, { label: `opt0081-${role}-f${
    elementBytes * 8}-prefill`, size: totalBytes,
    usage: GPUBufferUsage.COPY_SRC, mappedAtCreation: true });
  if (elementBytes === 4) {
    const words = new Uint32Array(prefill.getMappedRange());
    words.fill(STORAGE_GUARD_U32);
    words.fill(OUTPUT_PREFILL_QNAN_U32, STORAGE_GUARD_BYTES / 4,
      STORAGE_GUARD_BYTES / 4 + elements);
  } else {
    const words = new Uint16Array(prefill.getMappedRange());
    words.fill(STORAGE_GUARD_U16);
    words.fill(OUTPUT_PREFILL_QNAN_U16, STORAGE_GUARD_BYTES / 2,
      STORAGE_GUARD_BYTES / 2 + elements);
  }
  tracker.unmap(prefill);
  const buffer = tracker.create(device, { label: `opt0081-${role}-f${
    elementBytes * 8}-guarded-output`, size: totalBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST });
  const readback = tracker.create(device, { label: `opt0081-${role}-f${
    elementBytes * 8}-readback`, size: totalBytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  return Object.freeze({ buffer, binding: Object.freeze({ buffer,
    offset: STORAGE_GUARD_BYTES, size: bodyBytes }), prefill, readback,
    totalBytes, elementBytes });
}

async function executeProducerAudit(device: GPUDevice, tracker: BufferTracker,
  role: AceOpt0081F16DenseInputRole, elements: number,
  current: EncodableDispatch, candidate: EncodableDispatch,
  f32: GuardedProducerOutput,
  f16: GuardedProducerOutput): Promise<Readonly<Record<string, unknown>>> {
  const encoder = device.createCommandEncoder({
    label: `opt0081-${role}-producer-audit`,
  });
  encoder.copyBufferToBuffer(f32.prefill, 0, f32.buffer, 0, f32.totalBytes);
  encoder.copyBufferToBuffer(f16.prefill, 0, f16.buffer, 0, f16.totalBytes);
  const pass = encoder.beginComputePass();
  current.encode(pass);
  candidate.encode(pass);
  pass.end();
  encoder.copyBufferToBuffer(f32.buffer, 0, f32.readback, 0, f32.totalBytes);
  encoder.copyBufferToBuffer(f16.buffer, 0, f16.readback, 0, f16.totalBytes);
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  await tracker.mapRead(f32.readback);
  try {
    await tracker.mapRead(f16.readback);
    try {
      const f32All = new Uint32Array(f32.readback.getMappedRange());
      const f16All = new Uint16Array(f16.readback.getMappedRange());
    const f32Start = STORAGE_GUARD_BYTES / 4;
    const f16Start = STORAGE_GUARD_BYTES / 2;
    const f32Words = f32All.subarray(f32Start, f32Start + elements);
    const f16Words = f16All.subarray(f16Start, f16Start + elements);
    const f32Values = new Float32Array(f32Words.buffer, f32Words.byteOffset,
      f32Words.length);
    let differingU16Count = 0;
    let nonFiniteF32Count = 0;
    let nonFiniteF16Count = 0;
    let qNaNPrefillF32Count = 0;
    let qNaNPrefillF16Count = 0;
    let positiveZeroF32Count = 0;
    let negativeZeroF32Count = 0;
    let positiveZeroF16Count = 0;
    let negativeZeroF16Count = 0;
    let firstDifference: Readonly<Record<string, unknown>> | null = null;
    for (let index = 0; index < elements; index += 1) {
      const f32Word = f32Words[index]!;
      const actual = f16Words[index]!;
      const expected = numberToFloat16Bits(f32Values[index]!);
      if ((f32Word & 0x7f80_0000) === 0x7f80_0000) nonFiniteF32Count += 1;
      if ((actual & 0x7c00) === 0x7c00) nonFiniteF16Count += 1;
      if (f32Word === OUTPUT_PREFILL_QNAN_U32) qNaNPrefillF32Count += 1;
      if (actual === OUTPUT_PREFILL_QNAN_U16) qNaNPrefillF16Count += 1;
      if (f32Word === 0) positiveZeroF32Count += 1;
      if (f32Word === 0x8000_0000) negativeZeroF32Count += 1;
      if (actual === 0) positiveZeroF16Count += 1;
      if (actual === 0x8000) negativeZeroF16Count += 1;
      if (actual !== expected) {
        differingU16Count += 1;
        firstDifference ??= Object.freeze({ index,
          expectedU16: `0x${expected.toString(16).padStart(4, "0")}`,
          actualU16: `0x${actual.toString(16).padStart(4, "0")}`,
          sourceU32: `0x${f32Word.toString(16).padStart(8, "0")}` });
      }
    }
    const f32Guards = guardU32(f32All, f32Start, elements);
    const f16Guards = guardU16(f16All, f16Start, elements);
    const f32Sha256 = await sha256Bytes(new Uint8Array(f32Words.buffer,
      f32Words.byteOffset, f32Words.byteLength));
    const f16Sha256 = await sha256Bytes(new Uint8Array(f16Words.buffer,
      f16Words.byteOffset, f16Words.byteLength));
    const passed = differingU16Count === 0 && nonFiniteF32Count === 0 &&
      nonFiniteF16Count === 0 && positiveZeroF32Count > 0 &&
      negativeZeroF32Count > 0 && positiveZeroF16Count > 0 &&
      negativeZeroF16Count > 0 &&
      qNaNPrefillF32Count === 0 && qNaNPrefillF16Count === 0 &&
      f32Guards.passed && f16Guards.passed;
      return Object.freeze({ role, currentDispatchLabel: current.label,
        candidateDispatchLabel: candidate.label, elements,
        comparedU16Count: elements, differingU16Count, nonFiniteF32Count,
        nonFiniteF16Count, qNaNPrefillF32Count, qNaNPrefillF16Count,
        positiveZeroF32Count, negativeZeroF32Count,
        positiveZeroF16Count, negativeZeroF16Count,
        bothSignedZerosObservedAndExact: positiveZeroF16Count > 0 &&
          negativeZeroF16Count > 0 && differingU16Count === 0,
        firstDifference, f32Sha256, f16Sha256,
        expectedCastSemantics: "independent-f32-to-f16-rne-v1",
        guards: Object.freeze({ f32: f32Guards, f16: f16Guards }), passed });
    } finally {
      if (f16.readback.mapState === "mapped") tracker.unmap(f16.readback);
    }
  } finally {
    if (f32.readback.mapState === "mapped") tracker.unmap(f32.readback);
  }
}

async function prepareShape(
  device: GPUDevice,
  tracker: BufferTracker,
  armAKernel: AceOpt0009DenseGemmKernel,
  armBKernel: AceOpt0081DenseF16InputKernel,
  armCKernel: AceOpt0081DenseF16InputWeightMulticastKernel,
  spec: Opt0081ShapeSpec,
  retained: RetainedProducerPair,
): Promise<PreparedShape> {
  if (retained.elements !== spec.shape.rows * spec.shape.inner ||
    retained.role !== spec.retainedProducerRole) {
    throw new Error(`OPT-0081 ${spec.id} retained producer shape mismatch`);
  }
  const weightBytes = spec.shape.inner * spec.shape.columns * 2;
  const weight = tracker.create(device, {
    label: `opt0081-${spec.id}-packed-weight`,
    size: weightBytes,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });
  const output = createGuardedOutput(device, tracker, spec);
  try {
    const weightRange = weight.getMappedRange();
    fillPackedWeight(new Uint16Array(weightRange), spec);
    const weightSha256 = await sha256Bytes(new Uint8Array(weightRange));
    tracker.unmap(weight);
    const bindings: AceGemmBufferBindings = Object.freeze({
      activation: retained.f32Binding,
      weight: binding(weight, weightBytes),
      output: output.binding,
    });
    const A = await armAKernel.createDispatch(
      `opt0081-${spec.id}-A-opt0009`, spec.shape, bindings);
    const f16Bindings = Object.freeze({ ...bindings,
      activation: retained.f16Binding });
    const B = await armBKernel.createDispatch(
      `opt0081-${spec.id}-B-typed-f16-opt0009`, spec.shape, f16Bindings);
    const C = await armCKernel.createDispatch(
      `opt0081-${spec.id}-C-typed-f16-multicast`, spec.shape, f16Bindings);
    if (A.weightLayout !== ACE_DIT_DENSE_FP16_TILE_LAYOUT ||
      B.weightLayout !== ACE_DIT_DENSE_FP16_TILE_LAYOUT ||
      C.weightLayout !== ACE_DIT_DENSE_FP16_TILE_LAYOUT) {
      throw new Error(`OPT-0081 ${spec.id} changed native packed weight layout`);
    }
    const dispatches = Object.freeze({ A, B, C });
    const correctness = await verifyShape(device, tracker, spec, output,
      dispatches, Object.freeze({
        producerRole: retained.role,
        producerF32Sha256: retained.receipt["f32Sha256"],
        producerF16Sha256: retained.receipt["f16Sha256"], weightSha256,
      }));
    tracker.destroy(output.prefill);
    tracker.destroy(output.readback);
    return Object.freeze({ spec, activationF32: retained.f32Binding,
      activationF16: retained.f16Binding, weight, output, dispatches,
      correctness });
  } catch (error) {
    if (weight.mapState === "mapped") tracker.unmap(weight);
    tracker.destroy(weight);
    tracker.destroy(output.buffer);
    tracker.destroy(output.prefill);
    tracker.destroy(output.readback);
    throw error;
  }
}

async function verifyShape(
  device: GPUDevice,
  tracker: BufferTracker,
  spec: Opt0081ShapeSpec,
  output: GuardedOutput,
  dispatches: Readonly<Record<Opt0081Arm, AceGemmDispatch>>,
  inputHashes: Readonly<Record<string, unknown>>,
): Promise<Readonly<Record<string, unknown>>> {
  let A: OutputSnapshot | undefined = await executeCorrectness(
    device, tracker, output, dispatches.A,
  );
  let A2: OutputSnapshot | undefined = await executeCorrectness(
    device, tracker, output, dispatches.A,
  );
  const aa = compareSnapshots(A, A2, spec);
  const hashes: Record<string, string> = { A: A.sha256, A2: A2.sha256 };
  const completeness = [completeSnapshot(A), completeSnapshot(A2)];
  A2 = undefined;
  let B: OutputSnapshot | undefined = await executeCorrectness(
    device, tracker, output, dispatches.B,
  );
  const ab = compareSnapshots(A, B, spec);
  let B2: OutputSnapshot | undefined = await executeCorrectness(
    device, tracker, output, dispatches.B,
  );
  const bb = compareSnapshots(B, B2, spec);
  hashes["B"] = B.sha256;
  hashes["B2"] = B2.sha256;
  completeness.push(completeSnapshot(B), completeSnapshot(B2));
  B2 = undefined;
  let C: OutputSnapshot | undefined = await executeCorrectness(
    device, tracker, output, dispatches.C,
  );
  const ac = compareSnapshots(A, C, spec);
  const bc = compareSnapshots(B, C, spec);
  let C2: OutputSnapshot | undefined = await executeCorrectness(
    device, tracker, output, dispatches.C,
  );
  const cc = compareSnapshots(C, C2, spec);
  hashes["C"] = C.sha256;
  hashes["C2"] = C2.sha256;
  completeness.push(completeSnapshot(C), completeSnapshot(C2));
  A = undefined;
  B = undefined;
  C = undefined;
  C2 = undefined;
  const comparisons = Object.freeze({ aa, ab, bb, ac, cc, bc });
  const complete = completeness.every(Boolean);
  const exact = Object.values(comparisons).every((entry) =>
    entry.differingU32Count === 0);
  const finiteClassIdentity = Object.values(comparisons).every((entry) =>
    entry.finiteClassChangeCount === 0);
  const everyArmRerunExact = aa.differingU32Count === 0 &&
    bb.differingU32Count === 0 && cc.differingU32Count === 0;
  const directBCRawU32Exact = bc.differingU32Count === 0;
  const passed = exact && finiteClassIdentity && complete;
  return Object.freeze({ id: spec.id, shape: spec.shape,
    fixtureIds: spec.fixtureIds, inputHashes,
    comparedU32Count: spec.shape.rows * spec.shape.columns,
    executionOrder: Object.freeze(["A", "A", "B", "B", "C", "C"]),
    resultHashes: Object.freeze(hashes),
    comparisons, allSixRawU32Exact: exact, everyArmRerunExact,
    directBCRawU32Exact,
    outputsFiniteAndComplete: complete, finiteClassIdentity,
    guardsAndPartialMTailsIntact: complete,
    passed });
}

function completeSnapshot(snapshot: OutputSnapshot): boolean {
  return snapshot.nonFiniteCount === 0 && snapshot.qNaNPrefillCount === 0 &&
    snapshot.prefixGuardIntact && snapshot.suffixGuardIntact &&
    snapshot.adjacentBeforeIntact && snapshot.adjacentAfterIntact &&
    snapshot.firstValidWritten && snapshot.lastValidWritten &&
    snapshot.partialMTailWritten;
}

async function executeCorrectness(
  device: GPUDevice,
  tracker: BufferTracker,
  output: GuardedOutput,
  dispatch: AceGemmDispatch,
): Promise<OutputSnapshot> {
  if (output.readback.mapState !== "unmapped") {
    throw new Error("OPT-0081 output readback must be unmapped");
  }
  const encoder = device.createCommandEncoder({
    label: `${dispatch.label}-correctness`,
  });
  encoder.copyBufferToBuffer(output.prefill, 0, output.buffer, 0,
    output.totalBytes);
  const pass = encoder.beginComputePass();
  dispatch.encode(pass);
  pass.end();
  encoder.copyBufferToBuffer(output.buffer, 0, output.readback, 0,
    output.totalBytes);
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  await tracker.mapRead(output.readback);
  try {
    const all = new Uint32Array(output.readback.getMappedRange());
    const guardWords = STORAGE_GUARD_BYTES / 4;
    const outputEnd = guardWords + output.outputElements;
    let prefixGuardIntact = true;
    let suffixGuardIntact = true;
    for (let index = 0; index < guardWords; index += 1) {
      prefixGuardIntact &&= all[index] === STORAGE_GUARD_U32;
      suffixGuardIntact &&= all[outputEnd + index] === STORAGE_GUARD_U32;
    }
    const words = all.slice(guardWords, outputEnd);
    let nonFiniteCount = 0;
    let qNaNPrefillCount = 0;
    for (const word of words) {
      if ((word & 0x7f80_0000) === 0x7f80_0000) nonFiniteCount += 1;
      if (word === OUTPUT_PREFILL_QNAN_U32) qNaNPrefillCount += 1;
    }
    const firstValidWritten = words[0] !== OUTPUT_PREFILL_QNAN_U32;
    const lastValidWritten = words[words.length - 1] !==
      OUTPUT_PREFILL_QNAN_U32;
    let partialMTailWritten = true;
    const tailStart = (ROWS - 10) * output.columns;
    for (let index = tailStart; index < words.length; index += 1) {
      if (words[index] === OUTPUT_PREFILL_QNAN_U32) {
        partialMTailWritten = false;
        break;
      }
    }
    return Object.freeze({ words, sha256: await sha256U32(words),
      nonFiniteCount, qNaNPrefillCount, prefixGuardIntact,
      suffixGuardIntact, adjacentBeforeIntact: prefixGuardIntact,
      adjacentAfterIntact: suffixGuardIntact, firstValidWritten,
      lastValidWritten, partialMTailWritten });
  } finally {
    tracker.unmap(output.readback);
  }
}

function compareSnapshots(
  expected: OutputSnapshot,
  actual: OutputSnapshot,
  spec: Opt0081ShapeSpec,
): Readonly<Record<string, unknown>> {
  if (expected.words.length !== actual.words.length) {
    throw new Error(`OPT-0081 ${spec.id} output length changed`);
  }
  let differingU32Count = 0;
  let finiteClassChangeCount = 0;
  let signedZeroDifferenceCount = 0;
  let firstDifference: Readonly<Record<string, unknown>> | null = null;
  let worstDifference: Readonly<Record<string, unknown>> | null = null;
  let maximumU32Distance = 0;
  for (let index = 0; index < expected.words.length; index += 1) {
    const expectedWord = expected.words[index]!;
    const actualWord = actual.words[index]!;
    if (expectedWord === actualWord) continue;
    differingU32Count += 1;
    const expectedClass = f32Class(expectedWord);
    const actualClass = f32Class(actualWord);
    if (expectedClass !== actualClass) finiteClassChangeCount += 1;
    if ((expectedWord & 0x7fff_ffff) === 0 &&
      (actualWord & 0x7fff_ffff) === 0) signedZeroDifferenceCount += 1;
    const distance = Math.abs(orderedF32(expectedWord) - orderedF32(actualWord));
    const coordinate = Object.freeze({ index,
      row: Math.floor(index / spec.shape.columns),
      column: index % spec.shape.columns,
      expectedU32: `0x${expectedWord.toString(16).padStart(8, "0")}`,
      actualU32: `0x${actualWord.toString(16).padStart(8, "0")}`,
      expectedClass, actualClass, orderedU32Distance: distance });
    firstDifference ??= coordinate;
    if (distance > maximumU32Distance) {
      maximumU32Distance = distance;
      worstDifference = coordinate;
    }
  }
  return Object.freeze({ differingU32Count, finiteClassChangeCount,
    signedZeroDifferenceCount, maximumU32Distance, firstDifference,
    worstDifference });
}

function createGuardedOutput(
  device: GPUDevice,
  tracker: BufferTracker,
  spec: Opt0081ShapeSpec,
): GuardedOutput {
  const outputElements = spec.shape.rows * spec.shape.columns;
  const outputBytes = outputElements * 4;
  const totalBytes = outputBytes + 2 * STORAGE_GUARD_BYTES;
  const prefill = tracker.create(device, {
    label: `opt0081-${spec.id}-prefill`, size: totalBytes,
    usage: GPUBufferUsage.COPY_SRC, mappedAtCreation: true,
  });
  const prefillWords = new Uint32Array(prefill.getMappedRange());
  prefillWords.fill(STORAGE_GUARD_U32);
  prefillWords.fill(OUTPUT_PREFILL_QNAN_U32, STORAGE_GUARD_BYTES / 4,
    STORAGE_GUARD_BYTES / 4 + outputElements);
  tracker.unmap(prefill);
  const buffer = tracker.create(device, {
    label: `opt0081-${spec.id}-guarded-output`, size: totalBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST |
      GPUBufferUsage.COPY_SRC,
  });
  const readback = tracker.create(device, {
    label: `opt0081-${spec.id}-readback`, size: totalBytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  return Object.freeze({ buffer, binding: Object.freeze({ buffer,
    offset: STORAGE_GUARD_BYTES, size: outputBytes }), prefill, readback,
    outputElements, outputBytes, totalBytes, columns: spec.shape.columns });
}

function fillPackedWeight(values: Uint16Array, spec: Opt0081ShapeSpec): void {
  let physical = 0;
  for (let columnTile = 0; columnTile < spec.shape.columns / 256;
    columnTile += 1) {
    for (let innerTile = 0; innerTile < spec.shape.inner / 32;
      innerTile += 1) {
      for (let innerInTile = 0; innerInTile < 32; innerInTile += 1) {
        const inner = innerTile * 32 + innerInTile;
        for (let columnInTile = 0; columnInTile < 256;
          columnInTile += 1) {
          const column = columnTile * 256 + columnInTile;
          values[physical] = weightBitsAt(spec, inner, column);
          physical += 1;
        }
      }
    }
  }
  if (physical !== values.length) {
    throw new Error(`OPT-0081 ${spec.id} packed weight fill changed`);
  }
}

function weightBitsAt(
  spec: Opt0081ShapeSpec,
  inner: number,
  column: number,
): number {
  const mixed = mix32(0x6a09_e667 ^ Math.imul(spec.ordinal + 1, 0x9e37_79b1) ^
    Math.imul(inner + 1, 0x85eb_ca6b) ^ Math.imul(column + 1, 0xc2b2_ae35));
  const sign = (mixed >>> 31) << 15;
  if (spec.id === "h-h") return 0x3c00 | sign;
  if (spec.id === "h-1024") {
    return 0x3400 | ((inner & 1) << 15);
  }
  if (spec.id === "h-6144") return 0x7bff | sign;
  return [0x2401, 0x2c03, 0x3405, 0x3c00][mixed % 4]! | sign;
}

async function runTiming(
  prepared: PreparedHarness,
  thermalLaunch: Opt0081ThermalLaunchGate,
  update: (message: string) => void,
): Promise<Readonly<Record<string, unknown>>> {
  const samples = new Map<Opt0081ShapeSpec["id"],
    Record<Opt0081Arm, Opt0081TimestampSample[]>>(
      prepared.shapes.map(({ spec }) => [spec.id,
        { A: [], B: [], C: [] }]),
    );
  const rawSamples: Readonly<Record<string, unknown>>[] = [];
  const measurementStartedAtEpochMilliseconds = Date.now();
  for (const round of TIMING_ROUNDS) {
    for (const [shapePosition, shapeIndex] of round.shapeOrder.entries()) {
      const shape = prepared.shapes[shapeIndex];
      if (shape === undefined) {
        throw new Error("OPT-0081 timing shape rotation changed");
      }
      for (const [armPosition, arm] of round.armOrder.entries()) {
        update(`round ${round.roundIndex + 1}/8 shape ${shapePosition + 1}/4: ${shape.spec.id} ${arm}`);
        const sample = await timeDispatch(prepared.device,
          shape.dispatches[arm], prepared.querySet, prepared.queryResolve,
          prepared.queryReadback, prepared.tracker, shape.spec, arm);
        samples.get(shape.spec.id)![arm].push(sample);
        rawSamples.push(Object.freeze({ roundIndex: round.roundIndex,
          shapePosition, shapeIndex, shapeId: shape.spec.id,
          armPosition, arm, ...sample }));
      }
      await yieldToBrowser();
    }
  }
  await prepared.device.queue.onSubmittedWorkDone();
  await settlePostDrainEvents();
  const measurementCompletedAtEpochMilliseconds = Date.now();
  if (prepared.uncapturedErrors.length !== 0 ||
    prepared.deviceLosses.length !== 0) {
    throw new Error("OPT-0081 observed a timing GPU error or device loss");
  }
  const timingInputs = prepared.shapes.map(({ spec }) => {
    const entry = samples.get(spec.id)!;
    return Object.freeze({ id: spec.id, samples: Object.freeze({
      A: Object.freeze(entry.A), B: Object.freeze(entry.B),
      C: Object.freeze(entry.C),
    }) });
  });
  const timing = summarizeOpt0081Timing(timingInputs);
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
    cleanupSecond["repeatedCall"] === true &&
    prepared.uncapturedErrors.length === 0 &&
    prepared.deviceLosses.length === 0;
  const inPagePassed = timing["passed"] === true &&
    prepared.correctness["passed"] === true && cleanupPassed;
  return Object.freeze({ schema: RECEIPT_SCHEMA, experiment: EXPERIMENT_ID,
    status: "awaiting-external-thermal-completion", passed: false,
    inPagePassed, thermalLaunch, cleanupCompletedAtEpochMilliseconds,
    identity: prepared.identity, correctness: prepared.correctness,
    warmup: prepared.warmup,
    protocol: Object.freeze({
      correctnessAndWarmupBeforeReady: true,
      allSixFullM2250ProducerRawU16AuditBeforeReady: true,
      producerExecutionInsideTiming: false,
      gpuConversionDispatchCount: 0,
      fullOutputExecutionOrder: "A,A,B,B,C,C",
      rounds: TIMING_ROUNDS.length,
      exactRegisteredArmOrders: Object.freeze(ARM_ORDERS),
      rotatedFourShapeOrders: true,
      samplesPerArmPerShape: TIMING_ROUNDS.length,
      weightedProductionMultiplicities: "4/2/2/1",
      feedForwardMultiplicities: "0/0/2/1",
      onePassOneCommandBufferOneSubmitOneDrainPerSample: true,
      oneTimestampQueryPairPerSample: true,
      outputReadbackInsideTiming: false,
      allocationUploadCompilationInsideTiming: false,
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

async function timeDispatch(
  device: GPUDevice,
  dispatch: AceGemmDispatch,
  querySet: GPUQuerySet,
  queryResolve: GPUBuffer,
  queryReadback: GPUBuffer,
  tracker: BufferTracker,
  spec: Opt0081ShapeSpec,
  arm: Opt0081Arm,
): Promise<Opt0081TimestampSample> {
  if (queryReadback.mapState !== "unmapped") {
    throw new Error("OPT-0081 timestamp readback must be unmapped");
  }
  const encoder = device.createCommandEncoder({
    label: `opt0081-${spec.id}-${arm}-sample`,
  });
  const pass = encoder.beginComputePass({
    label: `opt0081-${spec.id}-${arm}-timestamped-compute`,
    timestampWrites: { querySet, beginningOfPassWriteIndex: 0,
      endOfPassWriteIndex: 1 },
  });
  dispatch.encode(pass);
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
    throw new Error(`OPT-0081 ${spec.id} ${arm} non-positive GPU timestamp`);
  }
  const gpuElapsedNanoseconds = Number(timestampEnd - timestampBegin);
  const gpuMilliseconds = gpuElapsedNanoseconds / 1_000_000;
  if (!Number.isSafeInteger(gpuElapsedNanoseconds) || gpuMilliseconds <= 0 ||
    !Number.isFinite(gpuMilliseconds) || wallMilliseconds <= 0 ||
    !Number.isFinite(wallMilliseconds)) {
    throw new Error(`OPT-0081 ${spec.id} ${arm} invalid timing sample`);
  }
  const scheduledMultiplyAdds = dispatch.plan.outputRanges.reduce(
    (sum, range) => sum + range.multiplyAdds, 0);
  const validMultiplyAdds = spec.shape.rows * spec.shape.inner *
    spec.shape.columns;
  return Object.freeze({ submitAtPerformanceMilliseconds,
    fenceAtPerformanceMilliseconds,
    submitAtEpochMilliseconds:
      performance.timeOrigin + submitAtPerformanceMilliseconds,
    fenceAtEpochMilliseconds:
      performance.timeOrigin + fenceAtPerformanceMilliseconds,
    wallMilliseconds, timestampBeginNanoseconds: timestampBegin.toString(),
    timestampEndNanoseconds: timestampEnd.toString(), gpuElapsedNanoseconds,
    gpuMilliseconds, gpuToWallRatio: gpuMilliseconds / wallMilliseconds,
    validMultiplyAdds, scheduledMultiplyAdds,
    validGpuTflops: tflops(validMultiplyAdds, gpuMilliseconds),
    scheduledGpuTflops: tflops(scheduledMultiplyAdds, gpuMilliseconds),
    validWallTflops: tflops(validMultiplyAdds, wallMilliseconds),
    scheduledWallTflops: tflops(scheduledMultiplyAdds, wallMilliseconds),
    commandBufferCount: 1, queueDrainCount: 1,
    timestampResolveCount: 1, timestampCopyCount: 1 });
}

async function executeAndDrain(
  device: GPUDevice,
  dispatch: AceGemmDispatch,
): Promise<void> {
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  dispatch.encode(pass);
  pass.end();
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
}

async function buildIdentity(
  adapter: GPUAdapter,
  device: GPUDevice,
): Promise<Readonly<Record<string, unknown>>> {
  const generatedShaders = await Promise.all(SHAPE_SPECS.map(async (spec) => {
    const A = planAceOpt0009DenseGemm(spec.shape);
    const B = planAceOpt0081DenseF16Input(spec.shape);
    const C = planAceOpt0081DenseF16InputWeightMulticast(spec.shape);
    return Object.freeze({ id: spec.id,
      retainedProducerRole: spec.retainedProducerRole,
      armAWgslSha256: await sha256Text(
        aceOpt0009DenseGemmWgsl(spec.shape)),
      armBWgslSha256: await sha256Text(
        aceOpt0081DenseF16InputWgsl(spec.shape)),
      armCWgslSha256: await sha256Text(
        aceOpt0081DenseF16InputWeightMulticastWgsl(spec.shape)),
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
        packedRecordLoadsPerWorkgroup:
          C.packedRecordLoadsPerWorkgroup,
      }),
      identicalOutputRanges: JSON.stringify(A.outputRanges) ===
        JSON.stringify(B.outputRanges) && JSON.stringify(B.outputRanges) ===
        JSON.stringify(C.outputRanges),
    });
  }));
  return Object.freeze({ registrationCommit: REGISTRATION_COMMIT,
    allocationBaselineCommit:
      "bbe180bf7feb59272a5d5f7afbafb3877afee416",
    implementationCheckpointCommit:
      "312d67024978a64b77d2563dd9386b4328f17d33",
    arms: Object.freeze({ A: "OPT-0009-f32-input",
      B: "OPT-0081-opt0009-typed-f16-input",
      C: "OPT-0081-opt0078-typed-f16-input-weight-multicast" }),
    producerProfile: ACE_OPT_0081_DIT_F16_DENSE_INPUT_STORAGE_PROFILE,
    producerKernelSetId:
      ACE_OPT_0081_DIT_F16_DENSE_INPUT_PRODUCER_KERNEL_SET_ID,
    producerRoles: ACE_OPT_0081_DIT_F16_DENSE_INPUT_ROLES,
    sharedWeightLayout: ACE_DIT_DENSE_FP16_TILE_LAYOUT,
    sourceSha256: Object.freeze({
      A: await sha256Text(armAKernelSource),
      B: await sha256Text(armBKernelSource),
      C: await sha256Text(armCKernelSource),
      f16Producer: await sha256Text(producerKernelSource),
      currentTransformerProducer: await sha256Text(transformerProducerSource),
      currentRmsNormProducer: await sha256Text(rmsNormProducerSource),
    }),
    generatedShaders: Object.freeze(generatedShaders),
    generatedShaderAggregateSha256: await sha256Text(
      generatedShaders.map((entry) => JSON.stringify(entry)).join("\n")),
    fixtureVersion: "opt0081-six-producer-four-dense-full-m2250-v1",
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
    }) });
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

export function classifyOpt0081PrimitiveDisposition(
  receipt: Readonly<Record<string, unknown>>,
): string {
  if (receipt["inPagePassed"] === true) {
    const timing = receipt["timing"] as
      Readonly<Record<string, unknown>> | undefined;
    if (timing?.["selectedArm"] === "C") {
      return "positive-C-primitive-diagnostic-profile-authorized";
    }
    if (timing?.["selectedArm"] === "B") {
      return "positive-B-standalone-primitive-diagnostic-profile-authorized";
    }
    return "inconclusive-invalid-correctness-or-lifecycle-evidence";
  }
  const correctness = receipt["correctness"] as
    Readonly<Record<string, unknown>> | undefined;
  const cleanupContainer = receipt["cleanup"] as
    Readonly<Record<string, unknown>> | undefined;
  if (cleanupContainer?.["passed"] !== true) {
    return "inconclusive-invalid-correctness-or-lifecycle-evidence";
  }
  if (correctness?.["passed"] !== true) {
    if (correctness !== undefined &&
      hasValidObservedCorrectnessMismatch(correctness)) {
      return "negative-stop-observed-raw-bit-correctness-mismatch";
    }
    return "inconclusive-invalid-correctness-or-lifecycle-evidence";
  }
  const timing = receipt["timing"] as
    Readonly<Record<string, unknown>> | undefined;
  const strata = timing?.["strata"] as readonly Readonly<
    Record<string, unknown>
  >[] | undefined;
  const comparisons = timing?.["comparisons"] as Readonly<Record<string,
    Readonly<Record<string, unknown>>>> | undefined;
  if (timing?.["passed"] === undefined || strata?.length !== 4 ||
    comparisons === undefined || hasMixedTimingEvidence(strata, comparisons)) {
    return "inconclusive-directional-or-wall-gpu-evidence";
  }
  return "negative-stop-typed-f16-weight-multicast-mechanism";
}

function hasMixedTimingEvidence(
  strata: readonly Readonly<Record<string, unknown>>[],
  comparisons: Readonly<Record<string, Readonly<Record<string, unknown>>>>,
): boolean {
  const names = ["bOverA", "cOverA", "cOverB"] as const;
  for (const name of names) {
    const shapeStates = strata.map((stratum) => {
      const gates = stratum["gates"] as Readonly<Record<string,
        Readonly<Record<string, unknown>>>>;
      const values = Object.values(gates[name] ?? {}).map(Boolean);
      return values.length === 4 && values.every(Boolean) ? "pass" :
        values.length === 4 && values.every((value) => !value) ? "fail" :
          "mixed";
    });
    if (shapeStates.includes("mixed") ||
      (shapeStates.includes("pass") && shapeStates.includes("fail"))) {
      return true;
    }
    const comparison = comparisons[name];
    const gpu = comparison?.["gpu"] as Readonly<Record<string, unknown>> |
      undefined;
    const wall = comparison?.["wall"] as Readonly<Record<string, unknown>> |
      undefined;
    if (comparison === undefined || gpu === undefined || wall === undefined ||
      comparison["wallGpuSavingsAgree"] === false) return true;
    const differs = (field: string, threshold: number) =>
      (Number(gpu[field]) >= threshold) !== (Number(wall[field]) >= threshold);
    const directionDiffers = (field: string) =>
      (Number(gpu[field]) > 0) !== (Number(wall[field]) > 0);
    const partialPairEvidence = [gpu, wall].some((clock) => {
      const wins = Number(clock["pairWins"]);
      return wins > 0 && wins < REQUIRED_PAIR_WINS;
    });
    if (partialPairEvidence || differs("pairWins", REQUIRED_PAIR_WINS) ||
      differs("meanSavingMilliseconds", name === "cOverA"
        ? REQUIRED_C_OVER_A_SAVING_MILLISECONDS
        : name === "bOverA" ? REQUIRED_B_OVER_A_SAVING_MILLISECONDS
          : REQUIRED_C_OVER_B_SAVING_MILLISECONDS) ||
      directionDiffers("meanSavingMilliseconds") ||
      directionDiffers("medianSavingMilliseconds")) return true;
    if (name === "bOverA" &&
      differs("meanSpeedup", REQUIRED_B_OVER_A_SPEEDUP)) return true;
    if (name === "cOverA" &&
      differs("meanSpeedup", REQUIRED_C_OVER_A_SPEEDUP)) return true;
  }
  return false;
}

function hasValidObservedCorrectnessMismatch(
  correctness: Readonly<Record<string, unknown>>,
): boolean {
  if (correctness["completedBeforeReady"] !== true ||
    correctness["uncapturedGpuErrorCount"] !== 0 ||
    correctness["deviceLossCount"] !== 0) return false;
  const producer = correctness["producerAudit"] as
    Readonly<Record<string, unknown>> | undefined;
  const producerCases = producer?.["cases"] as readonly Readonly<
    Record<string, unknown>
  >[] | undefined;
  const denseCases = correctness["cases"] as readonly Readonly<
    Record<string, unknown>
  >[] | undefined;
  if (producerCases?.length !== 6) return false;
  const producerMismatch = producerCases.some((entry) =>
    Number(entry["differingU16Count"]) > 0 ||
    Number(entry["nonFiniteF32Count"]) > 0 ||
    Number(entry["nonFiniteF16Count"]) > 0 ||
    Number(entry["qNaNPrefillF32Count"]) > 0 ||
    Number(entry["qNaNPrefillF16Count"]) > 0 ||
    entry["bothSignedZerosObservedAndExact"] === false ||
    (entry["guards"] as Readonly<Record<string,
      Readonly<Record<string, unknown>>>> | undefined)?.["f32"]?.["passed"] ===
        false ||
    (entry["guards"] as Readonly<Record<string,
      Readonly<Record<string, unknown>>>> | undefined)?.["f16"]?.["passed"] ===
        false);
  const denseMismatch = denseCases?.length === 4 && denseCases.some((entry) =>
    entry["allSixRawU32Exact"] === false ||
    entry["everyArmRerunExact"] === false ||
    entry["directBCRawU32Exact"] === false ||
    entry["outputsFiniteAndComplete"] === false ||
    entry["finiteClassIdentity"] === false ||
    entry["guardsAndPartialMTailsIntact"] === false);
  return producerMismatch || denseMismatch === true;
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

function shapeSpec(
  id: Opt0081ShapeSpec["id"],
  inner: number,
  columns: number,
  productionMultiplicity: 4 | 2 | 1,
  feedForwardMultiplicity: 0 | 2 | 1,
  retainedProducerRole: AceOpt0081F16DenseInputRole,
  fixtureIds: readonly string[],
): Opt0081ShapeSpec {
  return Object.freeze({ id, shape: Object.freeze({ rows: ROWS, inner,
    columns }), productionMultiplicity, feedForwardMultiplicity,
    retainedProducerRole,
    fixtureIds: Object.freeze([...fixtureIds]),
    ordinal: id === "h-h" ? 0 : id === "h-1024" ? 1 :
      id === "h-6144" ? 2 : 3 });
}

function requireTimestampSamples(
  samples: readonly Opt0081TimestampSample[],
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
  return Math.max(...SHAPE_SPECS.flatMap(({ shape }) => [
    shape.rows * shape.inner * 4,
    shape.inner * shape.columns * 2,
    shape.rows * shape.columns * 4,
  ]));
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

function producerValue(role: AceOpt0081F16DenseInputRole, index: number,
  operand: number): number {
  if (role === "selfMergedAttention" || role === "crossMergedAttention") {
    if (index === 0) return 0;
    if (index === 1) return -0;
  }
  if (role === "crossNormalized") {
    const feature = index % 2_048;
    if (feature === 0) return 0;
    if (feature === 1) return -0;
  }
  if (role === "gatedActivation" && index < 2) {
    if (operand === 0) return index === 0 ? 0 : -0;
    return 0.5;
  }
  if (role === "selfModulated" || role === "mlpModulated") {
    const feature = operand === 0 ? index % 2_048 : index;
    if (feature < 2) {
      if (operand === 1) return 0;
      return feature === 0 ? 0 : -0;
    }
  }
  const fixtureIndex = role === "selfModulated" || role === "mlpModulated"
    ? index - (index % 2_048 & 1) : index;
  const mixed = mix32(0x243f_6a88 ^
    Math.imul(fixtureIndex + 1, 0x9e37_79b1) ^
    Math.imul(operand + 1, 0x85eb_ca6b));
  if (role === "selfMergedAttention" || role === "crossMergedAttention") {
    const boundary = [0, -0, 2 ** -24, -(2 ** -24), 2 ** -14,
      1 + 2 ** -11, 1 + 2 ** -11 + 2 ** -23,
      -(1 + 2 ** -11 + 2 ** -23)] as const;
    return boundary[mixed % boundary.length]!;
  }
  const sign = (mixed & 1) === 0 ? 1 : -1;
  if (role === "gatedActivation") {
    if (operand === 0) return sign * ((mixed >>> 8) % 257) / 128;
    return sign * (1 + ((mixed >>> 12) % 31)) / 64;
  }
  if (role === "crossNormalized") {
    return sign * (1 + ((mixed >>> 8) % 127)) / 128;
  }
  if (operand === 0) {
    const boundary = [0, -0, 2 ** -24, -(2 ** -24), 0.125, -0.125,
      1 + 2 ** -11 + 2 ** -23] as const;
    return boundary[mixed % boundary.length]!;
  }
  if (operand === 1) return sign * ((mixed >>> 9) % 17) / 128;
  return sign * ((mixed >>> 13) % 17) / 256;
}

function guardU32(all: Uint32Array, bodyStart: number,
  elements: number): Readonly<Record<string, boolean>> {
  let prefixGuardIntact = true;
  let suffixGuardIntact = true;
  for (let index = 0; index < bodyStart; index += 1) {
    prefixGuardIntact &&= all[index] === STORAGE_GUARD_U32;
    suffixGuardIntact &&=
      all[bodyStart + elements + index] === STORAGE_GUARD_U32;
  }
  return Object.freeze({ prefixGuardIntact, suffixGuardIntact,
    adjacentBeforeIntact: all[bodyStart - 1] === STORAGE_GUARD_U32,
    adjacentAfterIntact: all[bodyStart + elements] === STORAGE_GUARD_U32,
    passed: prefixGuardIntact && suffixGuardIntact });
}

function guardU16(all: Uint16Array, bodyStart: number,
  elements: number): Readonly<Record<string, boolean>> {
  let prefixGuardIntact = true;
  let suffixGuardIntact = true;
  for (let index = 0; index < bodyStart; index += 1) {
    prefixGuardIntact &&= all[index] === STORAGE_GUARD_U16;
    suffixGuardIntact &&=
      all[bodyStart + elements + index] === STORAGE_GUARD_U16;
  }
  return Object.freeze({ prefixGuardIntact, suffixGuardIntact,
    adjacentBeforeIntact: all[bodyStart - 1] === STORAGE_GUARD_U16,
    adjacentAfterIntact: all[bodyStart + elements] === STORAGE_GUARD_U16,
    passed: prefixGuardIntact && suffixGuardIntact });
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

function orderedF32(word: number): number {
  return (word & 0x8000_0000) !== 0
    ? 0x8000_0000 - (word & 0x7fff_ffff)
    : 0x8000_0000 + word;
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
  window.__ACE_OPT0081_RESULT__ = receipt;
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
  const preparationCleanup = error instanceof Opt0081PreparationFailure
    ? Object.freeze({ originalName: error.originalName,
      receipt: error.setupCleanup }) : undefined;
  const structuredCorrectness = error instanceof Opt0081PreparationFailure
    ? error.correctness : undefined;
  const setupCleanupPassed = error instanceof Opt0081PreparationFailure &&
    cleanupReceiptPassed(error.setupCleanup);
  const structuredCleanup = structuredCorrectness === undefined ? undefined :
    Object.freeze({ receipt: error instanceof Opt0081PreparationFailure
      ? error.setupCleanup : undefined, passed: setupCleanupPassed });
  const disposition = structuredCorrectness === undefined
    ? stage.startsWith("thermal")
      ? "inconclusive-invalid-thermal-or-protocol-evidence"
      : "inconclusive-invalid-correctness-or-lifecycle-evidence"
    : classifyOpt0081PrimitiveDisposition(Object.freeze({
      inPagePassed: false, correctness: structuredCorrectness,
      cleanup: structuredCleanup,
    }));
  const receipt = Object.freeze({ schema: RECEIPT_SCHEMA,
    experiment: EXPERIMENT_ID,
    status: "failed", passed: false, stage, failure, ...evidence,
    preparationCleanup, correctness: structuredCorrectness,
    cleanup: structuredCleanup ?? evidence["cleanup"],
    decision: Object.freeze({ disposition, selectedArm: null,
      diagnosticProfileFollowUpAuthorized: false,
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
