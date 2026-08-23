import {
  planAceOpt0011Fp16VaeChunkGpuBackendMemory,
  planAceOpt0035Fp16VaeC2378ChunkGpuBackendMemory,
  type AceOpt0011Fp16VaeChunkGpuBackendMemoryPlan,
} from "../../src/webgpu/vae-fp16-backend.js";
import {
  ACE_OPT_0035_VAE_FP16_C2378_MAXIMUM_WINDOW_FRAMES,
  ACE_OPT_0035_VAE_FP16_C2378_WORKSPACE_BYTES,
  planAceOpt0011Fp16VaeChunkDispatches,
} from "../../src/webgpu/vae-fp16-decoder.js";
import {
  planAceVaeChunkedDecode,
  type AceVaeChunkedDecodePlan,
} from "../../src/webgpu/vae-chunks.js";
import { OPT_0066_NUMERICAL_ENVELOPE } from
  "./opt-0066-vae-dual-k4-quality-c512-contract.js";

export const OPT_0054_0066_LONG_SCHEMA =
  "ace-opt-0054-0066-vae-revision7-c2378-c4500-long-waveform-v1" as const;
export const OPT_0054_0066_LONG_PROTOCOL_ID =
  "opt-0054-0066-rev6-c512-rev7-c2378-candidate-repeat-v1" as const;
export const OPT_0054_0066_EXPERIMENT_ASSOCIATIONS = Object.freeze([
  "OPT-0054",
  "OPT-0066",
] as const);
export const OPT_0054_0066_LATENT_FRAMES = 4_500 as const;
export const OPT_0054_0066_LATENT_CHANNELS = 64 as const;
export const OPT_0054_0066_LATENT_ELEMENTS = 288_000 as const;
export const OPT_0054_0066_LATENT_BYTES = 1_152_000 as const;
export const OPT_0054_0066_LATENT_SHA256 =
  "d4e09d07be457583ff8ed4bf420f2ae4a1e822b4f7d6e8a71c300e53123c5971" as const;
export const OPT_0054_0066_CONTROL_RAW_SHA256 =
  "fb8aae85e21a8a93b39baf738d0f2577e18134c627a05562b710341d0d590f7c" as const;
export const OPT_0054_0066_OUTPUT_AUDIO_FRAMES = 8_640_000 as const;
export const OPT_0054_0066_OUTPUT_ELEMENTS = 17_280_000 as const;
export const OPT_0054_0066_OUTPUT_BYTES = 69_120_000 as const;
export const OPT_0054_0066_AUDIO_CHANNELS = 2 as const;
export const OPT_0054_0066_HOP_LENGTH = 1_920 as const;
export const OPT_0054_0066_CONTROL_CHUNK_FRAMES = 512 as const;
export const OPT_0054_0066_CANDIDATE_CHUNK_FRAMES =
  ACE_OPT_0035_VAE_FP16_C2378_MAXIMUM_WINDOW_FRAMES;
export const OPT_0054_0066_OVERLAP_FRAMES = 64 as const;
export const OPT_0054_0066_QUANTA_PER_COMMAND_BUFFER = 64 as const;
export const OPT_0054_0066_RAW_BLOCK_BYTES = 1_048_576 as const;
export const OPT_0054_0066_SEAM_RADIUS_LATENT_FRAMES = 64 as const;
export const OPT_0054_0066_MAXIMUM_LIVE_GPU_BYTES = 4_000_000_000 as const;
export const OPT_0054_0066_ARM_ORDER = Object.freeze([
  "rev7-cancellation-probe",
  "rev6-opt0028-c512-control",
  "rev7-opt0066-c2378-candidate",
  "rev7-opt0066-c2378-repeat",
] as const);

export type Opt00540066Arm = (typeof OPT_0054_0066_ARM_ORDER)[number];

export interface Opt00540066ArmTopologyPlan {
  readonly maximumWindowFrames: 512 | 2_378;
  readonly uniqueWindowFrames: readonly number[];
  readonly windowCount: number;
  readonly decodedLatentFrames: number;
  readonly aggregateGraphQuantumCount: number;
  readonly aggregateSequenceQuantumCount: number;
  readonly decoderCommandBufferCount: number;
  readonly readbackCommandBufferCount: number;
  readonly totalCommandBufferCount: number;
  readonly decoderRequestedCooperativeIdleMs: number;
  readonly betweenWindowRequestedCooperativeIdleMs: number;
  readonly totalRequestedCooperativeIdleMs: number;
}

export interface Opt00540066LongGatePlan {
  readonly control: AceVaeChunkedDecodePlan;
  readonly candidate: AceVaeChunkedDecodePlan;
  readonly controlTopology: Opt00540066ArmTopologyPlan;
  readonly candidateTopology: Opt00540066ArmTopologyPlan;
  readonly controlMemory: AceOpt0011Fp16VaeChunkGpuBackendMemoryPlan;
  readonly candidateMemory: AceOpt0011Fp16VaeChunkGpuBackendMemoryPlan;
  readonly controlSeams: readonly number[];
  readonly candidateSeams: readonly [2_250];
  readonly comparisonSeams: readonly Readonly<{
    readonly label: string;
    readonly latentFrame: number;
  }>[];
}

export interface Opt00540066WaveformMetric {
  readonly count: number;
  readonly finiteCount: number;
  readonly nonFiniteCount: number;
  readonly nrmse: number;
  readonly snrDb: number;
  readonly pearson: number;
  readonly maximumAbsoluteError: number;
  readonly relativeMaximumAbsoluteError: number;
  readonly controlRms: number;
  readonly candidateRms: number;
  readonly controlEnergy: number;
  readonly candidateEnergy: number;
  readonly relativeRmsDrift: number;
  readonly relativeEnergyDrift: number;
  readonly controlPeak: number;
  readonly candidatePeak: number;
  readonly relativePeakDrift: number;
  readonly controlMean: number;
  readonly candidateMean: number;
  readonly relativeDcOffsetDrift: number;
  readonly finite: boolean;
  readonly passed: boolean;
}

export interface Opt00540066StereoWaveformMetrics {
  readonly joint: Opt00540066WaveformMetric;
  readonly left: Opt00540066WaveformMetric;
  readonly right: Opt00540066WaveformMetric;
}

export interface Opt00540066CapturedWindowTopology {
  readonly inputFrames: number;
  readonly operationCount: number;
  readonly graphQuantumCount: number;
  readonly sequenceQuantumCount: number;
  readonly kernelQuantumCounts: Readonly<Record<string, number>>;
  readonly operationQuantumCounts: readonly Readonly<{
    readonly quantumCount: number;
  }>[];
}

export interface Opt00540066WindowTopologyReconciliation {
  readonly inputFrames: number;
  readonly operationCount: number;
  readonly graphQuantumCount: number;
  readonly sequenceQuantumCount: number;
  readonly ingressQuantumCount: number;
  readonly kernelQuantumTotal: number;
  readonly operationQuantumTotal: number;
  readonly passed: boolean;
}

/**
 * `kernelQuantumCounts` covers the complete sequence, including ingress;
 * `operationQuantumCounts` covers only the decoder graph's 88 operations.
 */
export function reconcileOpt00540066CapturedWindowTopology(
  window: Opt00540066CapturedWindowTopology,
): Opt00540066WindowTopologyReconciliation {
  const kernelQuantumTotal = Object.values(window.kernelQuantumCounts).reduce(
    (total, count) => total + count,
    0,
  );
  const operationQuantumTotal = window.operationQuantumCounts.reduce(
    (total, entry) => total + entry.quantumCount,
    0,
  );
  const ingressQuantumCount = window.sequenceQuantumCount -
    window.graphQuantumCount;
  return Object.freeze({
    inputFrames: window.inputFrames,
    operationCount: window.operationCount,
    graphQuantumCount: window.graphQuantumCount,
    sequenceQuantumCount: window.sequenceQuantumCount,
    ingressQuantumCount,
    kernelQuantumTotal,
    operationQuantumTotal,
    passed: window.operationCount === 88 && ingressQuantumCount === 1 &&
      kernelQuantumTotal === window.sequenceQuantumCount &&
      operationQuantumTotal === window.graphQuantumCount,
  });
}

/** Pure, deterministic C4500 geometry and bounded-memory contract. */
export function planOpt00540066LongGate(): Opt00540066LongGatePlan {
  const control = planAceVaeChunkedDecode(OPT_0054_0066_LATENT_FRAMES, {
    chunkFrames: OPT_0054_0066_CONTROL_CHUNK_FRAMES,
    overlapFrames: OPT_0054_0066_OVERLAP_FRAMES,
  });
  const candidate = planAceVaeChunkedDecode(OPT_0054_0066_LATENT_FRAMES, {
    chunkFrames: OPT_0054_0066_CANDIDATE_CHUNK_FRAMES,
    overlapFrames: OPT_0054_0066_OVERLAP_FRAMES,
  });
  const controlDispatches = planAceOpt0011Fp16VaeChunkDispatches(
    OPT_0054_0066_LATENT_FRAMES,
    OPT_0054_0066_CONTROL_CHUNK_FRAMES,
    256,
  );
  const candidateDispatches = planAceOpt0011Fp16VaeChunkDispatches(
    OPT_0054_0066_LATENT_FRAMES,
    OPT_0054_0066_CANDIDATE_CHUNK_FRAMES,
    256,
  );
  const controlTopology = summarizeTopology(control, controlDispatches);
  const candidateTopology = summarizeTopology(candidate, candidateDispatches);
  const controlMemory = planAceOpt0011Fp16VaeChunkGpuBackendMemory(
    control,
    256,
    OPT_0054_0066_QUANTA_PER_COMMAND_BUFFER,
  );
  const candidateMemory = planAceOpt0035Fp16VaeC2378ChunkGpuBackendMemory(
    candidate,
    256,
    OPT_0054_0066_QUANTA_PER_COMMAND_BUFFER,
  );
  const controlSeams = Object.freeze(control.windows.slice(1).map((window) =>
    window.coreStartLatentFrame
  ));
  const candidateSeams = Object.freeze(candidate.windows.slice(1).map(
    (window) => window.coreStartLatentFrame,
  )) as readonly [2_250];
  const comparisonSeams = Object.freeze([
    ...controlSeams.map((latentFrame) => Object.freeze({
      label: `control-${latentFrame}`,
      latentFrame,
    })),
    ...candidateSeams.map((latentFrame) => Object.freeze({
      label: `candidate-${latentFrame}`,
      latentFrame,
    })),
  ]);

  const candidateWindows = candidate.windows.map((window) => ({
    latent: [window.windowStartLatentFrame, window.windowEndLatentFrame],
    core: [window.coreStartLatentFrame, window.coreEndLatentFrame],
    prefix: window.discardPrefixLatentFrames,
    suffix: window.discardSuffixLatentFrames,
  }));
  if (
    control.outputAudioFrames !== OPT_0054_0066_OUTPUT_AUDIO_FRAMES ||
    control.outputInterleavedElements !== OPT_0054_0066_OUTPUT_ELEMENTS ||
    control.outputFloat32Bytes !== OPT_0054_0066_OUTPUT_BYTES ||
    control.windows.length !== 12 || control.maximumWindowFrames !== 512 ||
    candidate.windows.length !== 2 || candidate.maximumWindowFrames !== 2_314 ||
    JSON.stringify(candidateWindows) !== JSON.stringify([
      { latent: [0, 2_314], core: [0, 2_250], prefix: 0, suffix: 64 },
      { latent: [2_186, 4_500], core: [2_250, 4_500], prefix: 64, suffix: 0 },
    ]) ||
    JSON.stringify(controlDispatches.uniqueWindowFrames) !==
      JSON.stringify([340, 448, 512]) ||
    JSON.stringify(candidateDispatches.uniqueWindowFrames) !==
      JSON.stringify([2_314]) ||
    controlTopology.decodedLatentFrames !== 5_908 ||
    candidateTopology.decodedLatentFrames !== 4_628 ||
    controlTopology.aggregateGraphQuantumCount !== 90_675 ||
    controlTopology.aggregateSequenceQuantumCount !== 90_687 ||
    controlTopology.totalCommandBufferCount !== 1_432 ||
    controlTopology.totalRequestedCooperativeIdleMs !== 1_431 ||
    candidateTopology.aggregateGraphQuantumCount !== 70_994 ||
    candidateTopology.aggregateSequenceQuantumCount !== 70_996 ||
    candidateTopology.totalCommandBufferCount !== 1_112 ||
    candidateTopology.totalRequestedCooperativeIdleMs !== 1_111 ||
    JSON.stringify(controlSeams) !== JSON.stringify([
      384, 768, 1_152, 1_536, 1_920, 2_304, 2_688, 3_072, 3_456,
      3_840, 4_224,
    ]) || JSON.stringify(candidateSeams) !== JSON.stringify([2_250]) ||
    controlMemory.accountedGpuBytes !== 944_808_752 ||
    controlMemory.workspaceBufferBytes !== 251_658_240 ||
    controlMemory.controlBufferBytes !== 5_117_232 ||
    candidateMemory.accountedGpuBytes !== 3_758_347_792 ||
    candidateMemory.workspaceBufferBytes !==
      ACE_OPT_0035_VAE_FP16_C2378_WORKSPACE_BYTES ||
    candidateMemory.controlBufferBytes !== 9_087_248 ||
    candidateMemory.accountedGpuBytes >= OPT_0054_0066_MAXIMUM_LIVE_GPU_BYTES
  ) {
    throw new Error("OPT-0054/0066 C4500 long-gate contract changed");
  }
  return Object.freeze({
    control,
    candidate,
    controlTopology,
    candidateTopology,
    controlMemory,
    candidateMemory,
    controlSeams,
    candidateSeams,
    comparisonSeams,
  });
}

export class Opt00540066MetricAccumulator {
  private count = 0;
  private finiteCount = 0;
  private nonFiniteCount = 0;
  private sumControl = 0;
  private sumCandidate = 0;
  private sumControlSquared = 0;
  private sumCandidateSquared = 0;
  private sumProduct = 0;
  private sumSquaredError = 0;
  private maximumAbsoluteError = 0;
  private controlPeak = 0;
  private candidatePeak = 0;

  add(control: number, candidate: number): void {
    this.count += 1;
    if (!Number.isFinite(control) || !Number.isFinite(candidate)) {
      this.nonFiniteCount += 1;
      return;
    }
    const error = candidate - control;
    this.finiteCount += 1;
    this.sumControl += control;
    this.sumCandidate += candidate;
    this.sumControlSquared += control * control;
    this.sumCandidateSquared += candidate * candidate;
    this.sumProduct += control * candidate;
    this.sumSquaredError += error * error;
    this.maximumAbsoluteError = Math.max(
      this.maximumAbsoluteError,
      Math.abs(error),
    );
    this.controlPeak = Math.max(this.controlPeak, Math.abs(control));
    this.candidatePeak = Math.max(this.candidatePeak, Math.abs(candidate));
  }

  finish(): Opt00540066WaveformMetric {
    if (this.count === 0 || this.finiteCount === 0) {
      throw new RangeError("OPT-0054/0066 waveform metric is empty");
    }
    const controlMean = this.sumControl / this.finiteCount;
    const candidateMean = this.sumCandidate / this.finiteCount;
    const controlEnergy = this.sumControlSquared / this.finiteCount;
    const candidateEnergy = this.sumCandidateSquared / this.finiteCount;
    const controlRms = Math.sqrt(controlEnergy);
    const candidateRms = Math.sqrt(candidateEnergy);
    const rmse = Math.sqrt(this.sumSquaredError / this.finiteCount);
    const nrmse = rmse / Math.max(controlRms, 1e-30);
    const snrDb = rmse === 0
      ? Number.POSITIVE_INFINITY
      : 20 * Math.log10(Math.max(controlRms, 1e-30) / rmse);
    const controlVariance = Math.max(
      0,
      controlEnergy - controlMean * controlMean,
    );
    const candidateVariance = Math.max(
      0,
      candidateEnergy - candidateMean * candidateMean,
    );
    const covariance = this.sumProduct / this.finiteCount -
      controlMean * candidateMean;
    const denominator = Math.sqrt(controlVariance * candidateVariance);
    const pearson = denominator === 0 ? (rmse === 0 ? 1 : 0) :
      covariance / denominator;
    const relativeRmsDrift = Math.abs(candidateRms - controlRms) /
      Math.max(controlRms, 1e-30);
    const relativeEnergyDrift = Math.abs(candidateEnergy - controlEnergy) /
      Math.max(controlEnergy, 1e-30);
    const relativePeakDrift = Math.abs(
      this.candidatePeak - this.controlPeak,
    ) / Math.max(this.controlPeak, 1e-30);
    const relativeDcOffsetDrift = Math.abs(candidateMean - controlMean) /
      Math.max(controlRms, 1e-6);
    const relativeMaximumAbsoluteError = this.maximumAbsoluteError /
      Math.max(this.controlPeak, 1e-6);
    const finite = this.nonFiniteCount === 0 &&
      this.finiteCount === this.count;
    const limits = OPT_0066_NUMERICAL_ENVELOPE;
    const passed = finite && nrmse <= limits.nrmseMaximum &&
      snrDb >= limits.snrMinimumDb && pearson >= limits.pearsonMinimum &&
      relativeRmsDrift <= limits.relativeRmsDriftMaximum &&
      relativeEnergyDrift <= limits.relativeEnergyDriftMaximum &&
      relativePeakDrift <= limits.relativePeakDriftMaximum &&
      relativeDcOffsetDrift <= limits.relativeDcOffsetDriftMaximum &&
      relativeMaximumAbsoluteError <=
        limits.relativeMaximumAbsoluteErrorMaximum;
    return Object.freeze({
      count: this.count,
      finiteCount: this.finiteCount,
      nonFiniteCount: this.nonFiniteCount,
      nrmse,
      snrDb,
      pearson,
      maximumAbsoluteError: this.maximumAbsoluteError,
      relativeMaximumAbsoluteError,
      controlRms,
      candidateRms,
      controlEnergy,
      candidateEnergy,
      relativeRmsDrift,
      relativeEnergyDrift,
      controlPeak: this.controlPeak,
      candidatePeak: this.candidatePeak,
      relativePeakDrift,
      controlMean,
      candidateMean,
      relativeDcOffsetDrift,
      finite,
      passed,
    });
  }
}

export class Opt00540066StereoMetricAccumulator {
  private readonly joint = new Opt00540066MetricAccumulator();
  private readonly left = new Opt00540066MetricAccumulator();
  private readonly right = new Opt00540066MetricAccumulator();

  add(control: Float32Array, candidate: Float32Array): void {
    if (control.length !== candidate.length || control.length % 2 !== 0) {
      throw new RangeError(
        "OPT-0054/0066 comparison requires equal stereo blocks",
      );
    }
    for (let index = 0; index < control.length; index += 2) {
      const controlLeft = control[index]!;
      const controlRight = control[index + 1]!;
      const candidateLeft = candidate[index]!;
      const candidateRight = candidate[index + 1]!;
      this.left.add(controlLeft, candidateLeft);
      this.right.add(controlRight, candidateRight);
      this.joint.add(controlLeft, candidateLeft);
      this.joint.add(controlRight, candidateRight);
    }
  }

  finish(): Opt00540066StereoWaveformMetrics {
    return Object.freeze({
      joint: this.joint.finish(),
      left: this.left.finish(),
      right: this.right.finish(),
    });
  }
}

export function compareOpt00540066Waveforms(
  control: Float32Array,
  candidate: Float32Array,
): Opt00540066StereoWaveformMetrics {
  const accumulator = new Opt00540066StereoMetricAccumulator();
  accumulator.add(control, candidate);
  return accumulator.finish();
}

export function allOpt00540066MetricsPassed(
  metrics: Opt00540066StereoWaveformMetrics,
): boolean {
  return metrics.joint.passed && metrics.left.passed && metrics.right.passed;
}

function summarizeTopology(
  plan: AceVaeChunkedDecodePlan,
  dispatches: ReturnType<typeof planAceOpt0011Fp16VaeChunkDispatches>,
): Opt00540066ArmTopologyPlan {
  const decoderCommandBufferCount = dispatches.windowTopologyIndices.reduce(
    (sum, topologyIndex) => sum + Math.ceil(
      dispatches.topologies[topologyIndex]!.sequenceQuantumCount /
        OPT_0054_0066_QUANTA_PER_COMMAND_BUFFER,
    ),
    0,
  );
  const readbackCommandBufferCount = plan.windows.length;
  const totalCommandBufferCount = decoderCommandBufferCount +
    readbackCommandBufferCount;
  const decoderRequestedCooperativeIdleMs = totalCommandBufferCount -
    plan.windows.length;
  const betweenWindowRequestedCooperativeIdleMs = plan.windows.length - 1;
  return Object.freeze({
    maximumWindowFrames: dispatches.maximumWindowFramesProfile as 512 | 2_378,
    uniqueWindowFrames: dispatches.uniqueWindowFrames,
    windowCount: plan.windows.length,
    decodedLatentFrames: plan.windows.reduce(
      (sum, window) => sum + window.latentWindowFrames,
      0,
    ),
    aggregateGraphQuantumCount: dispatches.aggregateGraphQuantumCount,
    aggregateSequenceQuantumCount: dispatches.aggregateSequenceQuantumCount,
    decoderCommandBufferCount,
    readbackCommandBufferCount,
    totalCommandBufferCount,
    decoderRequestedCooperativeIdleMs,
    betweenWindowRequestedCooperativeIdleMs,
    totalRequestedCooperativeIdleMs:
      decoderRequestedCooperativeIdleMs +
      betweenWindowRequestedCooperativeIdleMs,
  });
}
