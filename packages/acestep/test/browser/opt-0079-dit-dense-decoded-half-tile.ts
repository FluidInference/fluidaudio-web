/// <reference types="@webgpu/types" />
/// <reference types="vite/client" />

import currentKernelSource from
  "../../src/webgpu/kernels/dit-dense-fp16.ts?raw";
import candidateKernelSource from
  "../../src/webgpu/kernels/dit-dense-fp16-decoded-half-tile.ts?raw";
import { ACE_DIT_DENSE_FP16_TILE_LAYOUT } from "../../src/model/manifest.js";
import {
  AceOpt0009DenseGemmKernel,
  aceOpt0009DenseGemmWgsl,
  planAceOpt0009DenseGemm,
} from "../../src/webgpu/kernels/dit-dense-fp16.js";
import {
  AceOpt0079DenseDecodedHalfTileKernel,
  aceOpt0079DenseDecodedHalfTileWgsl,
  planAceOpt0079DenseDecodedHalfTile,
} from "../../src/webgpu/kernels/dit-dense-fp16-decoded-half-tile.js";
import type {
  AceGemmBufferBindings,
  AceGemmDispatch,
  AceGemmShape,
} from "../../src/webgpu/kernels/gemm.js";

declare global {
  interface Window {
    __ACE_OPT0079_RESULT__?: Readonly<Record<string, unknown>>;
  }
}

export type Opt0079Arm = "current" | "candidate";

export interface Opt0079ShapeSpec {
  readonly id: "h-h" | "h-1024" | "h-6144" | "6144-h";
  readonly shape: AceGemmShape;
  readonly productionMultiplicity: 4 | 2 | 1;
  readonly feedForwardMultiplicity: 0 | 2 | 1;
  readonly fixtureIds: readonly string[];
  readonly ordinal: number;
}

export interface Opt0079TimestampSample {
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

export interface Opt0079TimingInput {
  readonly id: Opt0079ShapeSpec["id"];
  readonly samples: Readonly<Record<
    Opt0079Arm,
    readonly Opt0079TimestampSample[]
  >>;
}

export interface Opt0079ThermalLaunchGate {
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

export interface Opt0079ThermalCompletion {
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
  readonly partialMTailWritten: boolean;
}

interface PreparedShape {
  readonly spec: Opt0079ShapeSpec;
  readonly activation: GPUBuffer;
  readonly weight: GPUBuffer;
  readonly output: GuardedOutput;
  readonly dispatches: Readonly<Record<Opt0079Arm, AceGemmDispatch>>;
  readonly correctness: Readonly<Record<string, unknown>>;
}

interface PreparedHarness {
  readonly adapter: GPUAdapter;
  readonly device: GPUDevice;
  readonly tracker: BufferTracker;
  readonly currentKernel: AceOpt0009DenseGemmKernel;
  readonly candidateKernel: AceOpt0079DenseDecodedHalfTileKernel;
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

const EXPERIMENT_ID = "OPT-0079" as const;
const RECEIPT_SCHEMA = "ace-opt-0079-dense-decoded-half-tile-primitive-v1";
const REGISTRATION_COMMIT =
  "6313bfbaa61dee91f065d1652ee8eb1446f987b8";
const ROWS = 2_250;
const STORAGE_GUARD_BYTES = 256;
const STORAGE_GUARD_U32 = 0xa55a_c33c;
const OUTPUT_PREFILL_QNAN_U32 = 0x7fc0_7855;
const TIMESTAMP_QUERY_BYTES = 16;
const REQUIRED_SPEEDUP = 1.15;
const REQUIRED_SAVING_MILLISECONDS = 25;
const REQUIRED_PAIR_WINS = 7;
const PROJECTION_MULTIPLIER = 24 * 8;
const REQUIRED_PROJECTED_SAVING_MILLISECONDS = 4_800;
const THERMAL_SOURCE =
  "notifyutil-com.apple.system.thermalpressurelevel" as const;
const THERMAL_COMMAND =
  "notifyutil -g com.apple.system.thermalpressurelevel" as const;
const THERMAL_TRACE_SCHEMA =
  "jsonl-index-target-epoch-observed-epoch-keyed-notifyutil-v1" as const;
const THERMAL_POLL_MILLISECONDS = 1_000;
const MINIMUM_NOMINAL_MILLISECONDS = 30_000;
const MAXIMUM_THERMAL_GAP_MILLISECONDS = 1_250;
const MAXIMUM_GATE_TO_LAUNCH_MILLISECONDS = 5_000;

const SHAPE_SPECS = Object.freeze([
  shapeSpec("h-h", 2_048, 2_048, 4, 0, [
    "signed-zero", "subnormal-normal-boundary",
  ]),
  shapeSpec("h-1024", 2_048, 1_024, 2, 0, [
    "alternating-cancellation",
  ]),
  shapeSpec("h-6144", 2_048, 6_144, 2, 2, [
    "maximum-finite-fp16-bounded",
  ]),
  shapeSpec("6144-h", 6_144, 2_048, 1, 1, [
    "activation-rounding-boundary",
  ]),
] as const);

const TIMING_ROUNDS = Object.freeze(Array.from({ length: 8 }, (_, index) => {
  const base = [0, 1, 2, 3] as const;
  const rotation = index % base.length;
  const shapeOrder = Object.freeze([
    ...base.slice(rotation), ...base.slice(0, rotation),
  ]);
  const armOrder: readonly Opt0079Arm[] = index % 2 === 0
    ? Object.freeze(["current", "candidate"])
    : Object.freeze(["candidate", "current"]);
  return Object.freeze({ roundIndex: index, shapeOrder, armOrder });
}));

export function buildOpt0079ShapeSpecs(): readonly Opt0079ShapeSpec[] {
  return SHAPE_SPECS;
}

export function buildOpt0079TimingRounds(): typeof TIMING_ROUNDS {
  return TIMING_ROUNDS;
}

export function summarizeOpt0079Timing(
  inputs: readonly Opt0079TimingInput[],
): Readonly<Record<string, unknown>> {
  if (inputs.length !== SHAPE_SPECS.length) {
    throw new Error("OPT-0079 timing requires all four exact M2250 shapes");
  }
  const strata = inputs.map((input, index) => {
    const spec = SHAPE_SPECS[index];
    if (spec === undefined || input.id !== spec.id) {
      throw new Error("OPT-0079 timing shape order changed");
    }
    requireTimestampSamples(input.samples.current, `${spec.id} current`);
    requireTimestampSamples(input.samples.candidate, `${spec.id} candidate`);
    const summarizeArm = (samples: readonly Opt0079TimestampSample[]) => {
      const gpu = samples.map(({ gpuMilliseconds }) => gpuMilliseconds);
      const wall = samples.map(({ wallMilliseconds }) => wallMilliseconds);
      return Object.freeze({ samples, meanGpuMilliseconds: mean(gpu),
        medianGpuMilliseconds: median(gpu), meanWallMilliseconds: mean(wall),
        medianWallMilliseconds: median(wall), minimumGpuMilliseconds: Math.min(...gpu),
        maximumGpuMilliseconds: Math.max(...gpu), minimumWallMilliseconds: Math.min(...wall),
        maximumWallMilliseconds: Math.max(...wall) });
    };
    const current = summarizeArm(input.samples.current);
    const candidate = summarizeArm(input.samples.candidate);
    const gates = Object.freeze({
      meanGpuFaster: candidate.meanGpuMilliseconds < current.meanGpuMilliseconds,
      medianGpuFaster:
        candidate.medianGpuMilliseconds < current.medianGpuMilliseconds,
      meanWallFaster:
        candidate.meanWallMilliseconds < current.meanWallMilliseconds,
      medianWallFaster:
        candidate.medianWallMilliseconds < current.medianWallMilliseconds,
    });
    return Object.freeze({ id: spec.id, shape: spec.shape,
      productionMultiplicity: spec.productionMultiplicity,
      feedForwardMultiplicity: spec.feedForwardMultiplicity,
      arms: Object.freeze({ current, candidate }), gates,
      passed: Object.values(gates).every(Boolean) });
  });
  const weightedRounds = Array.from({ length: TIMING_ROUNDS.length },
    (_, roundIndex) => {
      const score = (arm: Opt0079Arm, metric: "gpuMilliseconds" |
        "wallMilliseconds", multiplicity: "productionMultiplicity" |
        "feedForwardMultiplicity") => inputs.reduce((sum, input, index) => {
          const sample = input.samples[arm][roundIndex];
          const spec = SHAPE_SPECS[index];
          if (sample === undefined || spec === undefined) {
            throw new Error("OPT-0079 weighted round sample missing");
          }
          return sum + sample[metric] * spec[multiplicity];
        }, 0);
      const currentGpu = score("current", "gpuMilliseconds",
        "productionMultiplicity");
      const candidateGpu = score("candidate", "gpuMilliseconds",
        "productionMultiplicity");
      const currentWall = score("current", "wallMilliseconds",
        "productionMultiplicity");
      const candidateWall = score("candidate", "wallMilliseconds",
        "productionMultiplicity");
      return Object.freeze({ roundIndex,
        complete: Object.freeze({ currentGpu, candidateGpu,
          currentWall, candidateWall,
          gpuSpeedup: currentGpu / candidateGpu,
          wallSpeedup: currentWall / candidateWall,
          gpuWin: candidateGpu < currentGpu,
          wallWin: candidateWall < currentWall }),
        feedForward: Object.freeze({
          currentGpu: score("current", "gpuMilliseconds",
            "feedForwardMultiplicity"),
          candidateGpu: score("candidate", "gpuMilliseconds",
            "feedForwardMultiplicity"),
          currentWall: score("current", "wallMilliseconds",
            "feedForwardMultiplicity"),
          candidateWall: score("candidate", "wallMilliseconds",
            "feedForwardMultiplicity"),
        }) });
    });
  const aggregateScore = (scope: "complete" | "feedForward") => {
    const currentGpu = weightedRounds.map((round) => round[scope].currentGpu);
    const candidateGpu = weightedRounds.map((round) => round[scope].candidateGpu);
    const currentWall = weightedRounds.map((round) => round[scope].currentWall);
    const candidateWall = weightedRounds.map((round) => round[scope].candidateWall);
    const meanCurrentGpu = mean(currentGpu);
    const meanCandidateGpu = mean(candidateGpu);
    const medianCurrentGpu = median(currentGpu);
    const medianCandidateGpu = median(candidateGpu);
    const meanCurrentWall = mean(currentWall);
    const meanCandidateWall = mean(candidateWall);
    const medianCurrentWall = median(currentWall);
    const medianCandidateWall = median(candidateWall);
    return Object.freeze({
      samples: Object.freeze({ currentGpu, candidateGpu, currentWall,
        candidateWall }),
      mean: Object.freeze({ currentGpu: meanCurrentGpu,
        candidateGpu: meanCandidateGpu, currentWall: meanCurrentWall,
        candidateWall: meanCandidateWall,
        gpuSpeedup: meanCurrentGpu / meanCandidateGpu,
        wallSpeedup: meanCurrentWall / meanCandidateWall,
        gpuSavingMilliseconds: meanCurrentGpu - meanCandidateGpu,
        wallSavingMilliseconds: meanCurrentWall - meanCandidateWall }),
      median: Object.freeze({ currentGpu: medianCurrentGpu,
        candidateGpu: medianCandidateGpu, currentWall: medianCurrentWall,
        candidateWall: medianCandidateWall,
        gpuSpeedup: medianCurrentGpu / medianCandidateGpu,
        wallSpeedup: medianCurrentWall / medianCandidateWall,
        gpuSavingMilliseconds: medianCurrentGpu - medianCandidateGpu,
        wallSavingMilliseconds: medianCurrentWall - medianCandidateWall }),
    });
  };
  const complete = aggregateScore("complete");
  const feedForward = aggregateScore("feedForward");
  const gpuPairWins = weightedRounds.filter((round) =>
    round.complete.gpuWin).length;
  const wallPairWins = weightedRounds.filter((round) =>
    round.complete.wallWin).length;
  const meanWallGpuSavingRatio = complete.mean.wallSavingMilliseconds /
    complete.mean.gpuSavingMilliseconds;
  const medianWallGpuSavingRatio = complete.median.wallSavingMilliseconds /
    complete.median.gpuSavingMilliseconds;
  const gates = Object.freeze({
    everyShapeMeanAndMedianFaster: strata.every(({ passed }) => passed),
    requiredPairWins: REQUIRED_PAIR_WINS,
    gpuPairWins, wallPairWins,
    gpuPairWinsPassed: gpuPairWins >= REQUIRED_PAIR_WINS,
    wallPairWinsPassed: wallPairWins >= REQUIRED_PAIR_WINS,
    requiredMeanAndMedianSpeedup: REQUIRED_SPEEDUP,
    meanGpuSpeedupPassed: complete.mean.gpuSpeedup >= REQUIRED_SPEEDUP,
    medianGpuSpeedupPassed: complete.median.gpuSpeedup >= REQUIRED_SPEEDUP,
    meanWallSpeedupPassed: complete.mean.wallSpeedup >= REQUIRED_SPEEDUP,
    medianWallSpeedupPassed: complete.median.wallSpeedup >= REQUIRED_SPEEDUP,
    requiredMeanAndMedianSavingMilliseconds: REQUIRED_SAVING_MILLISECONDS,
    meanGpuSavingPassed:
      complete.mean.gpuSavingMilliseconds >= REQUIRED_SAVING_MILLISECONDS,
    medianGpuSavingPassed:
      complete.median.gpuSavingMilliseconds >= REQUIRED_SAVING_MILLISECONDS,
    meanWallSavingPassed:
      complete.mean.wallSavingMilliseconds >= REQUIRED_SAVING_MILLISECONDS,
    medianWallSavingPassed:
      complete.median.wallSavingMilliseconds >= REQUIRED_SAVING_MILLISECONDS,
    meanWallGpuSavingRatio, medianWallGpuSavingRatio,
    wallGpuSavingsAgree: meanWallGpuSavingRatio >= 0.75 &&
      meanWallGpuSavingRatio <= 1.25 && medianWallGpuSavingRatio >= 0.75 &&
      medianWallGpuSavingRatio <= 1.25,
    projectedMeanGpuSavingMilliseconds:
      complete.mean.gpuSavingMilliseconds * PROJECTION_MULTIPLIER,
    projectedMedianGpuSavingMilliseconds:
      complete.median.gpuSavingMilliseconds * PROJECTION_MULTIPLIER,
    projectedMeanWallSavingMilliseconds:
      complete.mean.wallSavingMilliseconds * PROJECTION_MULTIPLIER,
    projectedMedianWallSavingMilliseconds:
      complete.median.wallSavingMilliseconds * PROJECTION_MULTIPLIER,
    requiredProjectedSavingMilliseconds:
      REQUIRED_PROJECTED_SAVING_MILLISECONDS,
    projectedSavingPassed:
      complete.mean.gpuSavingMilliseconds * PROJECTION_MULTIPLIER >=
        REQUIRED_PROJECTED_SAVING_MILLISECONDS &&
      complete.median.gpuSavingMilliseconds * PROJECTION_MULTIPLIER >=
        REQUIRED_PROJECTED_SAVING_MILLISECONDS &&
      complete.mean.wallSavingMilliseconds * PROJECTION_MULTIPLIER >=
        REQUIRED_PROJECTED_SAVING_MILLISECONDS &&
      complete.median.wallSavingMilliseconds * PROJECTION_MULTIPLIER >=
        REQUIRED_PROJECTED_SAVING_MILLISECONDS,
  });
  const passed = gates.everyShapeMeanAndMedianFaster &&
    gates.gpuPairWinsPassed && gates.wallPairWinsPassed &&
    gates.meanGpuSpeedupPassed && gates.medianGpuSpeedupPassed &&
    gates.meanWallSpeedupPassed && gates.medianWallSpeedupPassed &&
    gates.meanGpuSavingPassed && gates.medianGpuSavingPassed &&
    gates.meanWallSavingPassed && gates.medianWallSavingPassed &&
    gates.wallGpuSavingsAgree && gates.projectedSavingPassed;
  return Object.freeze({ samplesPerArmPerShape: TIMING_ROUNDS.length,
    multiplicities: Object.freeze({ complete: "4/2/2/1",
      feedForward: "0/0/2/1" }), strata: Object.freeze(strata),
    weightedRounds: Object.freeze(weightedRounds), complete, feedForward,
    gates, passed });
}

export function parseOpt0079ThermalLaunchGate(
  parameters: URLSearchParams,
  readyAtEpochMilliseconds: number,
  launchedAtEpochMilliseconds: number,
): Opt0079ThermalLaunchGate {
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
    throw new Error("OPT-0079 fresh continuous nominal thermal launch gate failed");
  }
  return Object.freeze({ source: THERMAL_SOURCE, command: THERMAL_COMMAND,
    traceStartedAtEpochMilliseconds, gateStartedAtEpochMilliseconds,
    gateCompletedAtEpochMilliseconds, observationCount,
    pollMilliseconds: THERMAL_POLL_MILLISECONDS,
    maximumPollGapMilliseconds, nonNominalObservationCount: 0,
    missingObservationCount: 0, readyToGateDelayMilliseconds,
    launchDelayMilliseconds });
}

export function parseOpt0079ThermalCompletion(
  parameters: URLSearchParams,
  launch: Opt0079ThermalLaunchGate,
  cleanupCompletedAtEpochMilliseconds: number,
): Opt0079ThermalCompletion {
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
    throw new Error("OPT-0079 complete through-cleanup thermal trace failed");
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
        "READY — four full-output exact/rerun gates and symmetric warmup passed; timing has not run";
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
    let thermalLaunch: Opt0079ThermalLaunchGate;
    try {
      thermalLaunch = parseOpt0079ThermalLaunchGate(
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
      Opt0079ThermalLaunchGate;
    try {
      const thermalCompletion = parseOpt0079ThermalCompletion(
        fieldParameters("#thermal-completion"),
        thermalLaunch,
        cleanupCompletedAtEpochMilliseconds,
      );
      const passed = pending["inPagePassed"] === true;
      const disposition = classifyOpt0079PrimitiveDisposition(pending);
      const receipt = Object.freeze({ ...pending, schema: RECEIPT_SCHEMA,
        experiment: EXPERIMENT_ID, status: "completed", passed,
        thermal: Object.freeze({ launch: thermalLaunch,
          completion: thermalCompletion, passed: true }),
        decision: Object.freeze({
          disposition,
          diagnosticProfileFollowUpAuthorized: passed,
          productionIntegrationAuthorized: false,
          packageChangeAuthorized: false,
          trajectoryOrListeningClaim: false,
          unchangedTimingRetryPerformed: false,
        }) });
      latestReceipt = receipt;
      publish(receipt, passed ? "passed" : "failed");
      progress.textContent = passed
        ? "completed — primitive gate passed; production remains unchanged"
        : "completed — exact decoded half-tile stopped at its frozen gate";
      pending = undefined;
      finalizeButton.disabled = true;
      completionFields.disabled = true;
      downloadButton.disabled = false;
    } catch (error) {
      publishFailure(error, "thermal-completion", { pending });
    }
  });

  downloadButton.addEventListener("click", () => {
    if (latestReceipt === undefined) return;
    const blob = new Blob([JSON.stringify(latestReceipt)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "opt-0079-dense-decoded-half-tile-receipt.json";
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
    label: "ace-opt-0079-dense-decoded-half-tile-profile-device",
    requiredFeatures: ["shader-f16", "subgroups", "timestamp-query"],
    requiredLimits: {
      maxBufferSize: maximumBufferBytes,
      maxStorageBufferBindingSize: maximumStorageBytes,
      maxComputeInvocationsPerWorkgroup: 256,
      maxComputeWorkgroupSizeX: 256,
      maxComputeWorkgroupStorageSize: 8_192,
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
    let partialCurrent: AceOpt0009DenseGemmKernel | undefined;
    let partialCandidate: AceOpt0079DenseDecodedHalfTileKernel | undefined;
    let partialQuerySet: GPUQuerySet | undefined;
    try {
      partialCurrent = AceOpt0009DenseGemmKernel.create(device, capability);
      partialCandidate = AceOpt0079DenseDecodedHalfTileKernel.create(
        device, capability);
      partialQuerySet = device.createQuerySet({
        label: "opt0079-dense-pass-timestamps", type: "timestamp", count: 2,
      });
      const queryResolve = tracker.create(device, {
        label: "opt0079-timestamp-resolve", size: TIMESTAMP_QUERY_BYTES,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
      });
      const queryReadback = tracker.create(device, {
        label: "opt0079-timestamp-readback", size: TIMESTAMP_QUERY_BYTES,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      return Object.freeze({ currentKernel: partialCurrent,
        candidateKernel: partialCandidate, querySet: partialQuerySet,
        queryResolve, queryReadback });
    } catch (error) {
      partialCurrent?.destroy();
      partialCandidate?.destroy();
      partialQuerySet?.destroy();
      tracker.destroyAll();
      device.destroy();
      throw error;
    }
  })();
  const { currentKernel, candidateKernel, querySet, queryResolve,
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
      let postDestroyRejected = false;
      candidateKernel.destroy();
      if (firstShape !== undefined) {
        try {
          await candidateKernel.createDispatch(
            "opt0079-post-destroy-rejection",
            firstShape.spec.shape,
            Object.freeze({ activation: binding(firstShape.activation,
              firstShape.spec.shape.rows * firstShape.spec.shape.inner * 4),
            weight: binding(firstShape.weight,
              firstShape.spec.shape.inner * firstShape.spec.shape.columns * 2),
            output: firstShape.output.binding }),
          );
        } catch {
          postDestroyRejected = true;
        }
      }
      currentKernel.destroy();
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
        postDestroyRejected, deviceDestroyed: true,
        idempotent: true, repeatedCall: false });
    })();
    return cleanupPromise;
  };
  try {
    const identity = await buildIdentity(adapter, device);
    const preparationOrder = [2, 3, 0, 1] as const;
    const preparedByIndex = new Map<number, PreparedShape>();
    for (const [position, specIndex] of preparationOrder.entries()) {
      const spec = SHAPE_SPECS[specIndex]!;
      update(`full-output current/candidate/rerun ${position + 1}/4: ${spec.id}`);
      const prepared = await prepareShape(device, tracker, currentKernel,
        candidateKernel, spec);
      firstShape ??= prepared;
      preparedByIndex.set(specIndex, prepared);
      await yieldToBrowser();
    }
    const shapes = SHAPE_SPECS.map((_, index) => preparedByIndex.get(index)!);
    const cases = shapes.map(({ correctness }) => correctness);
    const setupFailure = await verifyMalformedSetupRejected(
      candidateKernel,
      shapes[0]!,
    );
    const comparedU32Count = shapes.reduce((sum, { spec }) =>
      sum + spec.shape.rows * spec.shape.columns, 0);
    const fixtureIds = shapes.flatMap(({ spec }) => spec.fixtureIds);
    const correctness = Object.freeze({
      fullShapeCount: shapes.length,
      comparedU32CountPerComparison: comparedU32Count,
      comparisonCount: 12,
      comparisonsPerOutputWord: 3,
      snapshotsPerShape: 4,
      currentCandidateAndBothRerunGuardsChecked: true,
      currentCandidateComparedU32Count: comparedU32Count,
      candidateRerunComparedU32Count: comparedU32Count,
      currentRerunComparedU32Count: comparedU32Count,
      boundedAdversarialFixtureIds: Object.freeze(fixtureIds),
      boundedAdversarialFixtureCount: fixtureIds.length,
      setupFailure,
      setupFailureRejected: setupFailure.rejected,
      everyOutputRawU32Exact: cases.every((entry) =>
        entry["currentCandidateRawU32Exact"] === true),
      candidateRerunExact: cases.every((entry) =>
        entry["candidateRerunRawU32Exact"] === true),
      currentRerunExact: cases.every((entry) =>
        entry["currentRerunRawU32Exact"] === true),
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
        setupFailure.rejected &&
        uncapturedErrors.length === 0 && deviceLosses.length === 0,
    });
    if (!correctness.passed) {
      throw new Error("OPT-0079 full-output raw-U32 correctness gate failed");
    }
    update("symmetrically warming both arms over all four production shapes");
    const warmupStartedAtEpochMilliseconds = Date.now();
    for (const [index, shape] of shapes.entries()) {
      const order: readonly Opt0079Arm[] = index % 2 === 0
        ? ["current", "candidate"] : ["candidate", "current"];
      for (const arm of order) {
        await executeAndDrain(device, shape.dispatches[arm]);
      }
    }
    await settlePostDrainEvents();
    const warmupCompletedAtEpochMilliseconds = Date.now();
    if (uncapturedErrors.length !== 0 || deviceLosses.length !== 0) {
      throw new Error("OPT-0079 GPU failure during warmup");
    }
    const readyAtEpochMilliseconds = Date.now();
    return Object.freeze({ adapter, device, tracker, currentKernel,
      candidateKernel, querySet, queryResolve, queryReadback,
      shapes: Object.freeze(shapes), correctness, identity,
      warmup: Object.freeze({ warmupStartedAtEpochMilliseconds,
        warmupCompletedAtEpochMilliseconds, armExecutions: 8,
        oneUntimedWarmupPerArmPerShape: true, completedBeforeReady: true }),
      readyAtEpochMilliseconds, uncapturedErrors, deviceLosses, cleanup });
  } catch (error) {
    await cleanup();
    throw error;
  }
}

async function prepareShape(
  device: GPUDevice,
  tracker: BufferTracker,
  currentKernel: AceOpt0009DenseGemmKernel,
  candidateKernel: AceOpt0079DenseDecodedHalfTileKernel,
  spec: Opt0079ShapeSpec,
): Promise<PreparedShape> {
  const activationBytes = spec.shape.rows * spec.shape.inner * 4;
  const weightBytes = spec.shape.inner * spec.shape.columns * 2;
  const activation = tracker.create(device, {
    label: `opt0079-${spec.id}-activation`,
    size: activationBytes,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });
  const weight = tracker.create(device, {
    label: `opt0079-${spec.id}-packed-weight`,
    size: weightBytes,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });
  const output = createGuardedOutput(device, tracker, spec);
  try {
    const activationRange = activation.getMappedRange();
    fillActivation(new Float32Array(activationRange), spec);
    const activationSha256 = await sha256Bytes(new Uint8Array(activationRange));
    tracker.unmap(activation);
    const weightRange = weight.getMappedRange();
    fillPackedWeight(new Uint16Array(weightRange), spec);
    const weightSha256 = await sha256Bytes(new Uint8Array(weightRange));
    tracker.unmap(weight);
    const bindings: AceGemmBufferBindings = Object.freeze({
      activation: binding(activation, activationBytes),
      weight: binding(weight, weightBytes),
      output: output.binding,
    });
    const current = await currentKernel.createDispatch(
      `opt0079-${spec.id}-current-opt0009`, spec.shape, bindings);
    const candidate = await candidateKernel.createDispatch(
      `opt0079-${spec.id}-candidate-decoded-half-tile`, spec.shape, bindings);
    if (current.weightLayout !== ACE_DIT_DENSE_FP16_TILE_LAYOUT ||
      candidate.weightLayout !== ACE_DIT_DENSE_FP16_TILE_LAYOUT) {
      throw new Error(`OPT-0079 ${spec.id} changed native packed weight layout`);
    }
    const dispatches = Object.freeze({ current, candidate });
    const correctness = await verifyShape(device, tracker, spec, output,
      dispatches, Object.freeze({ activationSha256, weightSha256 }));
    tracker.destroy(output.prefill);
    tracker.destroy(output.readback);
    return Object.freeze({ spec, activation, weight, output, dispatches,
      correctness });
  } catch (error) {
    if (activation.mapState === "mapped") tracker.unmap(activation);
    if (weight.mapState === "mapped") tracker.unmap(weight);
    tracker.destroy(activation);
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
  spec: Opt0079ShapeSpec,
  output: GuardedOutput,
  dispatches: Readonly<Record<Opt0079Arm, AceGemmDispatch>>,
  inputHashes: Readonly<Record<string, string>>,
): Promise<Readonly<Record<string, unknown>>> {
  let current: OutputSnapshot | undefined = await executeCorrectness(
    device, tracker, output, dispatches.current);
  let currentRerun: OutputSnapshot | undefined = await executeCorrectness(
    device, tracker, output, dispatches.current);
  const currentRepeat = compareSnapshots(current, currentRerun, spec);
  const currentRerunHash = currentRerun.sha256;
  const currentComplete = completeSnapshot(current);
  const currentRerunComplete = completeSnapshot(currentRerun);
  const currentSnapshot = snapshotReceipt(current);
  const currentRerunSnapshot = snapshotReceipt(currentRerun);
  currentRerun = undefined;
  let candidate: OutputSnapshot | undefined = await executeCorrectness(
    device, tracker, output, dispatches.candidate);
  const currentCandidate = compareSnapshots(current, candidate, spec);
  const currentHash = current.sha256;
  const candidateHash = candidate.sha256;
  const candidateComplete = completeSnapshot(candidate);
  const candidateSnapshot = snapshotReceipt(candidate);
  current = undefined;
  const candidateRerun = await executeCorrectness(
    device, tracker, output, dispatches.candidate);
  const repeat = compareSnapshots(candidate, candidateRerun, spec);
  const candidateRerunHash = candidateRerun.sha256;
  const candidateRerunComplete = completeSnapshot(candidateRerun);
  const candidateRerunSnapshot = snapshotReceipt(candidateRerun);
  candidate = undefined;
  const complete = currentComplete && currentRerunComplete &&
    candidateComplete && candidateRerunComplete;
  const currentCandidateRawU32Exact =
    currentCandidate.differingU32Count === 0;
  const currentRerunRawU32Exact = currentRepeat.differingU32Count === 0;
  const candidateRerunRawU32Exact = repeat.differingU32Count === 0;
  const finiteClassIdentity = currentCandidate.finiteClassChangeCount === 0 &&
    currentRepeat.finiteClassChangeCount === 0 &&
    repeat.finiteClassChangeCount === 0;
  const passed = currentCandidateRawU32Exact && currentRerunRawU32Exact &&
    candidateRerunRawU32Exact && finiteClassIdentity && complete;
  return Object.freeze({ id: spec.id, shape: spec.shape,
    fixtureIds: spec.fixtureIds, inputHashes,
    comparedU32Count: spec.shape.rows * spec.shape.columns,
    resultHashes: Object.freeze({ current: currentHash,
      currentRerun: currentRerunHash, candidate: candidateHash,
      candidateRerun: candidateRerunHash }),
    snapshots: Object.freeze({ current: currentSnapshot,
      currentRerun: currentRerunSnapshot, candidate: candidateSnapshot,
      candidateRerun: candidateRerunSnapshot }),
    currentCandidate, currentRerun: currentRepeat, candidateRerun: repeat,
    currentCandidateRawU32Exact, currentRerunRawU32Exact,
    candidateRerunRawU32Exact,
    outputsFiniteAndComplete: complete, finiteClassIdentity,
    guardsAndPartialMTailsIntact: complete,
    candidateFirstRerunExact: candidateRerunRawU32Exact,
    passed });
}

function snapshotReceipt(
  snapshot: OutputSnapshot,
): Readonly<Record<string, unknown>> {
  return Object.freeze({ sha256: snapshot.sha256,
    outputU32Count: snapshot.words.length,
    nonFiniteCount: snapshot.nonFiniteCount,
    qNaNPrefillCount: snapshot.qNaNPrefillCount,
    prefixGuardIntact: snapshot.prefixGuardIntact,
    suffixGuardIntact: snapshot.suffixGuardIntact,
    adjacentBeforeIntact: snapshot.adjacentBeforeIntact,
    adjacentAfterIntact: snapshot.adjacentAfterIntact,
    partialMTailWritten: snapshot.partialMTailWritten,
    complete: completeSnapshot(snapshot) });
}

function completeSnapshot(snapshot: OutputSnapshot): boolean {
  return snapshot.nonFiniteCount === 0 && snapshot.qNaNPrefillCount === 0 &&
    snapshot.prefixGuardIntact && snapshot.suffixGuardIntact &&
    snapshot.adjacentBeforeIntact && snapshot.adjacentAfterIntact &&
    snapshot.partialMTailWritten;
}

async function executeCorrectness(
  device: GPUDevice,
  tracker: BufferTracker,
  output: GuardedOutput,
  dispatch: AceGemmDispatch,
): Promise<OutputSnapshot> {
  if (output.readback.mapState !== "unmapped") {
    throw new Error("OPT-0079 output readback must be unmapped");
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
    let partialMTailWritten = true;
    const tailStart = words.length - output.columns;
    for (let index = tailStart; index < words.length; index += 1) {
      if (words[index] === OUTPUT_PREFILL_QNAN_U32) {
        partialMTailWritten = false;
        break;
      }
    }
    return Object.freeze({ words, sha256: await sha256U32(words),
      nonFiniteCount, qNaNPrefillCount, prefixGuardIntact,
      suffixGuardIntact, adjacentBeforeIntact: prefixGuardIntact,
      adjacentAfterIntact: suffixGuardIntact, partialMTailWritten });
  } finally {
    tracker.unmap(output.readback);
  }
}

function compareSnapshots(
  expected: OutputSnapshot,
  actual: OutputSnapshot,
  spec: Opt0079ShapeSpec,
): Readonly<Record<string, unknown>> {
  if (expected.words.length !== actual.words.length) {
    throw new Error(`OPT-0079 ${spec.id} output length changed`);
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
  spec: Opt0079ShapeSpec,
): GuardedOutput {
  const outputElements = spec.shape.rows * spec.shape.columns;
  const outputBytes = outputElements * 4;
  const totalBytes = outputBytes + 2 * STORAGE_GUARD_BYTES;
  const prefill = tracker.create(device, {
    label: `opt0079-${spec.id}-prefill`, size: totalBytes,
    usage: GPUBufferUsage.COPY_SRC, mappedAtCreation: true,
  });
  const prefillWords = new Uint32Array(prefill.getMappedRange());
  prefillWords.fill(STORAGE_GUARD_U32);
  prefillWords.fill(OUTPUT_PREFILL_QNAN_U32, STORAGE_GUARD_BYTES / 4,
    STORAGE_GUARD_BYTES / 4 + outputElements);
  tracker.unmap(prefill);
  const buffer = tracker.create(device, {
    label: `opt0079-${spec.id}-guarded-output`, size: totalBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST |
      GPUBufferUsage.COPY_SRC,
  });
  const readback = tracker.create(device, {
    label: `opt0079-${spec.id}-readback`, size: totalBytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  return Object.freeze({ buffer, binding: Object.freeze({ buffer,
    offset: STORAGE_GUARD_BYTES, size: outputBytes }), prefill, readback,
    outputElements, outputBytes, totalBytes, columns: spec.shape.columns });
}

function fillActivation(values: Float32Array, spec: Opt0079ShapeSpec): void {
  let physical = 0;
  for (let row = 0; row < spec.shape.rows; row += 1) {
    for (let inner = 0; inner < spec.shape.inner; inner += 1) {
      values[physical] = activationValueAt(spec, row, inner);
      physical += 1;
    }
  }
}

function fillPackedWeight(values: Uint16Array, spec: Opt0079ShapeSpec): void {
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
    throw new Error(`OPT-0079 ${spec.id} packed weight fill changed`);
  }
}

function activationValueAt(
  spec: Opt0079ShapeSpec,
  row: number,
  inner: number,
): number {
  const mixed = mix32(0x3141_5926 ^ Math.imul(spec.ordinal + 1, 0x9e37_79b1) ^
    Math.imul(row + 1, 0x85eb_ca6b) ^ Math.imul(inner + 1, 0xc2b2_ae35));
  const sign = (mixed >>> 31) === 0 ? 1 : -1;
  if (spec.id === "h-h") {
    const boundary = [0x0000, 0x8000, 0x0001, 0x03ff, 0x0400, 0x0401][
      mixed % 6
    ]!;
    return halfToNumber(boundary);
  }
  if (spec.id === "h-1024") {
    return ((inner & 1) === 0 ? 1 : -1) * halfToNumber(0x3000);
  }
  if (spec.id === "h-6144") {
    return sign * halfToNumber([0x0001, 0x0002, 0x0003, 0x0004][mixed % 4]!);
  }
  const tie = 1 + 2 ** -11;
  const side = (mixed & 1) === 0 ? -(2 ** -24) : 2 ** -24;
  return sign * (tie + side);
}

function weightBitsAt(
  spec: Opt0079ShapeSpec,
  inner: number,
  column: number,
): number {
  const mixed = mix32(0x6a09_e667 ^ Math.imul(spec.ordinal + 1, 0x9e37_79b1) ^
    Math.imul(inner + 1, 0x85eb_ca6b) ^ Math.imul(column + 1, 0xc2b2_ae35));
  const sign = (mixed >>> 31) << 15;
  if (spec.id === "h-h") return 0x3c00 | sign;
  if (spec.id === "h-1024") {
    return 0x3400 | ((column & 1) << 15);
  }
  if (spec.id === "h-6144") return 0x7bff | sign;
  return [0x2401, 0x2c03, 0x3405, 0x3c00][mixed % 4]! | sign;
}

async function runTiming(
  prepared: PreparedHarness,
  thermalLaunch: Opt0079ThermalLaunchGate,
  update: (message: string) => void,
): Promise<Readonly<Record<string, unknown>>> {
  const samples = new Map<Opt0079ShapeSpec["id"],
    Record<Opt0079Arm, Opt0079TimestampSample[]>>(
      prepared.shapes.map(({ spec }) => [spec.id,
        { current: [], candidate: [] }]),
    );
  const rawSamples: Readonly<Record<string, unknown>>[] = [];
  const measurementStartedAtEpochMilliseconds = Date.now();
  for (const round of TIMING_ROUNDS) {
    for (const [shapePosition, shapeIndex] of round.shapeOrder.entries()) {
      const shape = prepared.shapes[shapeIndex];
      if (shape === undefined) {
        throw new Error("OPT-0079 timing shape rotation changed");
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
    throw new Error("OPT-0079 observed a timing GPU error or device loss");
  }
  const timingInputs = prepared.shapes.map(({ spec }) => {
    const entry = samples.get(spec.id)!;
    return Object.freeze({ id: spec.id, samples: Object.freeze({
      current: Object.freeze(entry.current),
      candidate: Object.freeze(entry.candidate),
    }) });
  });
  const timing = summarizeOpt0079Timing(timingInputs);
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
      targetBrowserMalformedBindingSetupFailureRejected:
        prepared.correctness["setupFailureRejected"] === true,
      fullOutputRawU32AndCurrentCandidateReruns: true,
      allGuardsCheckedOnBothArmsAndReruns: true,
      rounds: TIMING_ROUNDS.length,
      balancedAlternatingArmOrders: true,
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

async function verifyMalformedSetupRejected(
  candidateKernel: AceOpt0079DenseDecodedHalfTileKernel,
  shape: PreparedShape,
): Promise<Readonly<Record<string, unknown>> & { readonly rejected: boolean }> {
  const activationBytes = shape.spec.shape.rows * shape.spec.shape.inner * 4;
  const weightBytes = shape.spec.shape.inner * shape.spec.shape.columns * 2;
  const outputBytes = shape.spec.shape.rows * shape.spec.shape.columns * 4;
  if (activationBytes !== outputBytes) {
    return Object.freeze({ rejected: false,
      reason: "selected setup-failure fixture cannot expose an exact alias" });
  }
  try {
    await candidateKernel.createDispatch(
      "opt0079-target-browser-malformed-output-alias",
      shape.spec.shape,
      Object.freeze({
        activation: binding(shape.activation, activationBytes),
        weight: binding(shape.weight, weightBytes),
        output: binding(shape.activation, outputBytes),
      }),
    );
    return Object.freeze({ rejected: false,
      reason: "candidate accepted an aliased output binding" });
  } catch (error) {
    const expected = error instanceof RangeError &&
      /output must not overlap an input binding/u.test(error.message);
    return Object.freeze({ rejected: expected,
      fixture: "existing-h-h-activation-aliased-as-output",
      allocatedBufferCount: 0,
      submittedCommandBufferCount: 0,
      failureName: error instanceof Error ? error.name : typeof error,
      failureMessage: error instanceof Error ? error.message : String(error),
    });
  }
}

async function timeDispatch(
  device: GPUDevice,
  dispatch: AceGemmDispatch,
  querySet: GPUQuerySet,
  queryResolve: GPUBuffer,
  queryReadback: GPUBuffer,
  tracker: BufferTracker,
  spec: Opt0079ShapeSpec,
  arm: Opt0079Arm,
): Promise<Opt0079TimestampSample> {
  if (queryReadback.mapState !== "unmapped") {
    throw new Error("OPT-0079 timestamp readback must be unmapped");
  }
  const encoder = device.createCommandEncoder({
    label: `opt0079-${spec.id}-${arm}-sample`,
  });
  const pass = encoder.beginComputePass({
    label: `opt0079-${spec.id}-${arm}-timestamped-compute`,
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
    throw new Error(`OPT-0079 ${spec.id} ${arm} non-positive GPU timestamp`);
  }
  const gpuElapsedNanoseconds = Number(timestampEnd - timestampBegin);
  const gpuMilliseconds = gpuElapsedNanoseconds / 1_000_000;
  if (!Number.isSafeInteger(gpuElapsedNanoseconds) || gpuMilliseconds <= 0 ||
    !Number.isFinite(gpuMilliseconds) || wallMilliseconds <= 0 ||
    !Number.isFinite(wallMilliseconds)) {
    throw new Error(`OPT-0079 ${spec.id} ${arm} invalid timing sample`);
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
    const currentPlan = planAceOpt0009DenseGemm(spec.shape);
    const candidatePlan = planAceOpt0079DenseDecodedHalfTile(spec.shape);
    return Object.freeze({ id: spec.id,
      currentWgslSha256: await sha256Text(
        aceOpt0009DenseGemmWgsl(spec.shape)),
      candidateWgslSha256: await sha256Text(
        aceOpt0079DenseDecodedHalfTileWgsl(spec.shape)),
      currentPlan: planReceipt(currentPlan),
      candidatePlan: planReceipt(candidatePlan),
      candidateMechanism: Object.freeze({
        subgroupsPerWorkgroup: candidatePlan.subgroupsPerWorkgroup,
        rowsPerSubgroup: candidatePlan.rowsPerSubgroup,
        columnsPerLane: candidatePlan.columnsPerLane,
        accumulatorsPerLane: candidatePlan.accumulatorsPerLane,
        packedRecordsPerInnerTile: candidatePlan.packedRecordsPerInnerTile,
        packedRecordsPerLane: candidatePlan.packedRecordsPerLane,
        decodedVectorsPerInnerTile:
          candidatePlan.decodedVectorsPerInnerTile,
        decodedVectorsPerPackedRecord:
          candidatePlan.decodedVectorsPerPackedRecord,
        workgroupStorageBytes: candidatePlan.workgroupStorageBytes,
        barriersPerWorkgroup: candidatePlan.barriersPerWorkgroup,
        barrierEvents: candidatePlan.barrierEvents,
        packedRecordLoadsPerWorkgroup:
          candidatePlan.packedRecordLoadsPerWorkgroup,
        decodedVectorWritesPerWorkgroup:
          candidatePlan.decodedVectorWritesPerWorkgroup,
        physicalColumnTiles: candidatePlan.physicalColumnTiles,
        scheduledRows: candidatePlan.scheduledRows,
        scheduledMultiplyAdds: candidatePlan.scheduledMultiplyAdds,
        validMultiplyAdds: candidatePlan.validMultiplyAdds,
        estimatedGlobalActivationBytes:
          candidatePlan.estimatedGlobalActivationBytes,
        estimatedGlobalWeightBytes:
          candidatePlan.estimatedGlobalWeightBytes,
        estimatedGlobalOperandBytes:
          candidatePlan.estimatedGlobalOperandBytes,
      }),
      sameLogicalOutputInterval:
        currentPlan.outputRanges[0]?.firstOutput ===
          candidatePlan.outputRanges[0]?.firstOutput &&
        currentPlan.outputRanges[0]?.outputCount ===
          candidatePlan.outputRanges[0]?.outputCount,
      sameNativePackedWeightStorageShape:
        JSON.stringify(currentPlan.packedWeightStorageShape) ===
          JSON.stringify(candidatePlan.packedWeightStorageShape),
      candidateWorkgroupRatio:
        candidatePlan.workgroupCount / currentPlan.workgroupCount,
      candidateScheduledMultiplyAddRatio:
        candidatePlan.scheduledMultiplyAdds /
          currentPlan.outputRanges[0]!.multiplyAdds,
    });
  }));
  const staticAccounting = SHAPE_SPECS.reduce((totals, spec) => {
    const current = planAceOpt0009DenseGemm(spec.shape);
    const candidate = planAceOpt0079DenseDecodedHalfTile(spec.shape);
    const multiplicity = spec.productionMultiplicity;
    totals.currentWorkgroups += current.workgroupCount * multiplicity;
    totals.candidateWorkgroups += candidate.workgroupCount * multiplicity;
    totals.currentScheduledMultiplyAdds +=
      current.outputRanges[0]!.multiplyAdds * multiplicity;
    totals.candidateScheduledMultiplyAdds +=
      candidate.scheduledMultiplyAdds * multiplicity;
    totals.currentEstimatedGlobalActivationBytes += current.rowTiles *
      current.tileRows * current.inner * current.columnTiles * 4 * multiplicity;
    totals.currentEstimatedGlobalWeightBytes += current.weightElements * 2 *
      current.rowTiles * (current.workgroupSize / current.subgroupSize) *
      multiplicity;
    totals.candidateEstimatedGlobalActivationBytes +=
      candidate.estimatedGlobalActivationBytes * multiplicity;
    totals.candidateEstimatedGlobalWeightBytes +=
      candidate.estimatedGlobalWeightBytes * multiplicity;
    totals.candidateEstimatedGlobalOperandBytes +=
      candidate.estimatedGlobalOperandBytes * multiplicity;
    return totals;
  }, {
    currentWorkgroups: 0,
    candidateWorkgroups: 0,
    currentScheduledMultiplyAdds: 0,
    candidateScheduledMultiplyAdds: 0,
    currentEstimatedGlobalActivationBytes: 0,
    currentEstimatedGlobalWeightBytes: 0,
    candidateEstimatedGlobalActivationBytes: 0,
    candidateEstimatedGlobalWeightBytes: 0,
    candidateEstimatedGlobalOperandBytes: 0,
  });
  return Object.freeze({ registrationCommit: REGISTRATION_COMMIT,
    allocationBaselineCommit:
      "4084610c5e43dff2d388361965750b0f603400dd",
    currentKernel: "OPT-0009",
    candidateKernel: "OPT-0079",
    sharedWeightLayout: ACE_DIT_DENSE_FP16_TILE_LAYOUT,
    sourceSha256: Object.freeze({
      current: await sha256Text(currentKernelSource),
      candidate: await sha256Text(candidateKernelSource),
    }),
    generatedShaders: Object.freeze(generatedShaders),
    staticAccounting: Object.freeze({ ...staticAccounting,
      currentEstimatedGlobalOperandBytes:
        staticAccounting.currentEstimatedGlobalActivationBytes +
        staticAccounting.currentEstimatedGlobalWeightBytes,
      productionMultiplicities: "4/2/2/1",
    }),
    generatedShaderAggregateSha256: await sha256Text(
      generatedShaders.map((entry) => JSON.stringify(entry)).join("\n")),
    fixtureVersion: "opt0079-full-m2250-five-edge-families-v1",
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
  ReturnType<typeof planAceOpt0079DenseDecodedHalfTile>):
Readonly<Record<string, unknown>> {
  return Object.freeze({ rows: plan.rows, inner: plan.inner,
    columns: plan.columns,
    tile: Object.freeze([plan.tileRows, plan.tileColumns, plan.tileInner]),
    workgroupSize: plan.workgroupSize,
    rowTiles: plan.rowTiles, columnTiles: plan.columnTiles,
    innerTiles: plan.innerTiles, workgroupCount: plan.workgroupCount,
    outputRangeCount: plan.outputRangeCount,
    outputRanges: plan.outputRanges,
    scheduledMultiplyAdds: plan.outputRanges.reduce((sum, range) =>
      sum + range.multiplyAdds, 0),
    packedWeightStorageShape: plan.packedWeightStorageShape,
    ...("scheduledRows" in plan ? {
      scheduledRows: plan.scheduledRows,
      validMultiplyAdds: plan.validMultiplyAdds,
    } : {}),
    ...("physicalColumnTiles" in plan ? {
      physicalColumnTiles: plan.physicalColumnTiles,
    } : {}),
  });
}

export function classifyOpt0079PrimitiveDisposition(
  receipt: Readonly<Record<string, unknown>>,
): string {
  const correctness = receipt["correctness"] as
    Readonly<Record<string, unknown>> | undefined;
  const cleanupContainer = receipt["cleanup"] as
    Readonly<Record<string, unknown>> | undefined;
  if (correctness?.["passed"] !== true ||
    cleanupContainer?.["passed"] !== true) {
    return "inconclusive-invalid-correctness-or-lifecycle-evidence";
  }
  const timing = receipt["timing"] as
    Readonly<Record<string, unknown>> | undefined;
  const strata = timing?.["strata"] as readonly Readonly<
    Record<string, unknown>
  >[] | undefined;
  const gates = timing?.["gates"] as
    Readonly<Record<string, unknown>> | undefined;
  if (timing === undefined || !Array.isArray(strata) ||
    strata.length !== SHAPE_SPECS.length || gates === undefined) {
    return "inconclusive-infrastructure-or-missing-evidence";
  }
  if (receipt["inPagePassed"] === true) {
    return timing["passed"] === true
      ? "positive-primitive-diagnostic-profile-authorized"
      : "inconclusive-infrastructure-or-missing-evidence";
  }
  const shapeGateSets = strata.map((stratum) => {
    const shapeGates = stratum["gates"];
    if (shapeGates === null || typeof shapeGates !== "object") return [];
    return Object.values(shapeGates as Readonly<Record<string, unknown>>)
      .map(Boolean);
  });
  if (shapeGateSets.some((values) => values.length !== 4)) {
    return "inconclusive-infrastructure-or-missing-evidence";
  }
  const anyDirectionallyMixedShape = shapeGateSets.some((values) =>
    values.some(Boolean) && !values.every(Boolean));
  const anyConsistentlyRegressingShape = shapeGateSets.some((values) =>
    values.length === 4 && values.every((value) => !value));
  const wallGpuDisagreement = gates?.["wallGpuSavingsAgree"] === false;
  if (anyDirectionallyMixedShape || wallGpuDisagreement) {
    return "inconclusive-directional-or-wall-gpu-evidence";
  }
  const projectedMeanWall = Number(
    gates?.["projectedMeanWallSavingMilliseconds"],
  );
  const projectedMedianWall = Number(
    gates?.["projectedMedianWallSavingMilliseconds"],
  );
  const projectedMeanGpu = Number(
    gates?.["projectedMeanGpuSavingMilliseconds"],
  );
  const projectedMedianGpu = Number(
    gates?.["projectedMedianGpuSavingMilliseconds"],
  );
  const projections = [projectedMeanWall, projectedMedianWall,
    projectedMeanGpu, projectedMedianGpu];
  const clearSubTwoSecondProjection = projections.every(Number.isFinite) &&
    Math.min(...projections) < 2_000;
  if (anyConsistentlyRegressingShape || clearSubTwoSecondProjection) {
    return "negative-stop-exact-decoded-half-tile-mechanism";
  }
  return "inconclusive-directional-or-wall-gpu-evidence";
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
      throw new Error("OPT-0079 attempted to unmap an unmapped buffer");
    }
    buffer.unmap();
    this.unmaps += 1;
    this.activeMaps -= 1;
    if (this.activeMaps < 0) {
      throw new Error("OPT-0079 map accounting became negative");
    }
  }

  destroy(buffer: GPUBuffer): void {
    if (!this.live.delete(buffer)) return;
    if (buffer.mapState === "mapped") this.unmap(buffer);
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
  id: Opt0079ShapeSpec["id"],
  inner: number,
  columns: number,
  productionMultiplicity: 4 | 2 | 1,
  feedForwardMultiplicity: 0 | 2 | 1,
  fixtureIds: readonly string[],
): Opt0079ShapeSpec {
  return Object.freeze({ id, shape: Object.freeze({ rows: ROWS, inner,
    columns }), productionMultiplicity, feedForwardMultiplicity,
    fixtureIds: Object.freeze([...fixtureIds]),
    ordinal: id === "h-h" ? 0 : id === "h-1024" ? 1 :
      id === "h-6144" ? 2 : 3 });
}

function requireTimestampSamples(
  samples: readonly Opt0079TimestampSample[],
  label: string,
): void {
  if (samples.length !== TIMING_ROUNDS.length || samples.some((sample) =>
    !Number.isFinite(sample.gpuMilliseconds) || sample.gpuMilliseconds <= 0 ||
    !Number.isFinite(sample.wallMilliseconds) || sample.wallMilliseconds <= 0 ||
    !Number.isSafeInteger(sample.gpuElapsedNanoseconds) ||
    sample.gpuElapsedNanoseconds <= 0 || sample.commandBufferCount !== 1 ||
    sample.queueDrainCount !== 1 || sample.timestampResolveCount !== 1 ||
    sample.timestampCopyCount !== 1)) {
    throw new Error(`OPT-0079 ${label} requires eight valid timestamp samples`);
  }
}

function mean(values: readonly number[]): number {
  if (values.length !== TIMING_ROUNDS.length || values.some((value) =>
    !Number.isFinite(value) || value <= 0)) {
    throw new Error("OPT-0079 mean requires eight finite positive samples");
  }
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function median(values: readonly number[]): number {
  if (values.length !== TIMING_ROUNDS.length || values.some((value) =>
    !Number.isFinite(value) || value <= 0)) {
    throw new Error("OPT-0079 median requires eight finite positive samples");
  }
  const sorted = [...values].sort((left, right) => left - right);
  return (sorted[3]! + sorted[4]!) / 2;
}

function requiredParameter(parameters: URLSearchParams, name: string): string {
  const value = parameters.get(name);
  if (value === null || value.trim() === "") {
    throw new Error(`OPT-0079 field ${name} is missing`);
  }
  return value.trim();
}

function requiredFiniteParameter(
  parameters: URLSearchParams,
  name: string,
): number {
  const value = Number(requiredParameter(parameters, name));
  if (!Number.isFinite(value)) {
    throw new Error(`OPT-0079 field ${name} is invalid`);
  }
  return value;
}

function requiredIntegerParameter(
  parameters: URLSearchParams,
  name: string,
): number {
  const value = requiredFiniteParameter(parameters, name);
  if (!Number.isSafeInteger(value)) {
    throw new Error(`OPT-0079 field ${name} must be an integer`);
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
    adapter.limits.maxComputeWorkgroupStorageSize < 8_192 ||
    adapter.limits.maxStorageBufferBindingSize < maximumStorageBytes ||
    adapter.limits.maxBufferSize < maximumStorageBytes +
      2 * STORAGE_GUARD_BYTES) {
    throw new Error(
      "OPT-0079 requires timestamp-query, shader-f16, fixed32 subgroups, WG256, 8KiB workgroup storage, and full-shape storage limits",
    );
  }
}

function binding(buffer: GPUBuffer, size: number): GPUBufferBinding {
  return Object.freeze({ buffer, offset: 0, size });
}

function tflops(multiplyAdds: number, milliseconds: number): number {
  if (!Number.isSafeInteger(multiplyAdds) || multiplyAdds <= 0 ||
    !Number.isFinite(milliseconds) || milliseconds <= 0) {
    throw new Error("OPT-0079 TFLOP/s inputs must be finite and positive");
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

function halfToNumber(bits: number): number {
  const sign = (bits & 0x8000) === 0 ? 1 : -1;
  const exponent = (bits >>> 10) & 0x1f;
  const mantissa = bits & 0x03ff;
  if (exponent === 0) {
    if (mantissa === 0) return sign < 0 ? -0 : 0;
    return sign * 2 ** -14 * (mantissa / 1_024);
  }
  if (exponent === 0x1f) {
    return mantissa === 0 ? sign * Number.POSITIVE_INFINITY : Number.NaN;
  }
  return sign * 2 ** (exponent - 15) * (1 + mantissa / 1_024);
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
    throw new Error("OPT-0079 fixtures require a little-endian host");
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
  window.__ACE_OPT0079_RESULT__ = receipt;
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
  publish(Object.freeze({ schema: RECEIPT_SCHEMA, experiment: EXPERIMENT_ID,
    status: "failed", passed: false, stage, failure, ...evidence,
    productionIntegrationAuthorized: false }), "failed");
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
