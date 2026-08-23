import type { AceDynamicConditionalWeightingConfiguration } from "../api.js";
import type { AceGpuTensorPhase } from "../model/gpu-tensors.js";
import {
  ACE_COOPERATIVE_GPU_IDLE_MILLISECONDS,
  AceCooperativeGpuScheduler,
  planAceDepth2Epoch4CompletionEpochs,
  type AceDepth2Epoch4CommandBufferCompletionTiming,
  type AceDepth2Epoch4CompletionEpochTiming,
  type AceDepth2Epoch4SchedulingProgress,
} from "../runtime/scheduler.js";
import {
  ACE_TURBO_DIT_CONFIG,
  AceCorrectnessDitRuntime,
  aceDitLayerScratchBytes,
  aceDitLayerAttentionMode,
  planAceDitEvaluation,
  requireAceDitDenseInputStorageProfile,
  type AceDitDenseGemmRuntimeConfiguration,
  type AceDitDenseInputStorageProfile,
  type AceDitCrossCacheScratch,
  type AceDitEvaluationPlan,
  type AceDitEvaluationShape,
  type AceDitCompositeDispatch,
  type AceDitLayerBindings,
  type AceDitLayerScratch,
  type AceDitOutputScratch,
  type AceDitTimestepScratch,
} from "./ace-dit.js";
import {
  ACE_FP16_PORTABLE_PROFILE,
  ACE_REFERENCE_PORTABLE_PROFILE,
  ACE_REFERENCE_SUBGROUP_PROFILE,
  type AceExecutionProfile,
  type AceModelProfileId,
} from "./capabilities.js";
import {
  ACE_DIT_QUERY8_ATTENTION_RUNTIME_PROFILE,
  ACE_OPT_0062_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
  ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
  requireAceDitAttentionRuntimeProfile,
  type AceDitAttentionRuntimeProfile,
} from "./dit-attention-profile.js";
import {
  ACE_DIT_PHYSICAL_QUANTA_PER_COMMAND_BUFFER,
  ACE_DIT_PROFILE_FAMILIES,
  ACE_OPT_0018_M2250_GRAPH_COMMAND_BUFFER_COUNT,
  ACE_OPT_0067_M2250_EVALUATION0_GRAPH_COMMAND_BUFFER_COUNT,
  ACE_OPT_0080_M2250_FULL_PHASE_COMMAND_BUFFER_COUNTS,
  AceDitGraphOwner,
  AceDitResidentModel,
  aceDitGraphQuantumCount,
  createAceDitGraphControlData,
  createAceDitGraphQuantumPlan,
  planAceDitGraphMemory,
  planAceDitPhysicalQuantumBatches,
  type AceDitGraphBindings,
  type AceDitGraphCompilationProgress,
  type AceDitGraphMemoryPlan,
  type AceDitPhysicalCommandDescriptor,
  type AceDitPhysicalCommandBatchDescriptor,
  type AceDitPhysicalCommandDescriptorTable,
  type AceDitPhysicalQuantaPerCommandBuffer,
  type AceDitProfileFamily,
  type AceDitGraphProgress,
  type AceDitGraphRunOptions,
  type AceDitGraphRunResult,
  type AceDitSubmissionPolicy,
  type AceDitOpt0080CommandBufferCompletion,
  type AceDitOpt0080CompletionEpochTiming,
  type AceDitOpt0080PhaseStart,
  type AceDitOpt0080SchedulingProfile,
} from "./dit-graph.js";
import { AceGpuArena } from "./arena.js";
import { createAceScopedBuffers } from "./scoped-buffer-allocation.js";
import {
  aceCompositeCooperativeSequence,
  planAceCompositeCooperativeQuantumCount,
  ACE_DIT_GEMM_WEIGHT_LAYOUT,
  planAceTiledGemm,
  type AceCooperativeGemmPlan,
  type AceGemmShape,
  type AceGpuEncodeSequence,
  type AceGpuEncodeSequenceQuantumDescriptor,
} from "./kernels/gemm.js";
import { planAceSubgroupGemm } from "./kernels/subgroup-gemm.js";
import { planAceOpt0009DenseGemm } from "./kernels/dit-dense-fp16.js";
import { planAceOpt0032DenseK4Partials } from
  "./kernels/dit-dense-fp16-k4-partials.js";
import {
  ACE_OPT_0009_DIT_MIXED_RESIDENT_WEIGHT_BYTES,
  ACE_OPT_0009_DIT_DENSE_RUNTIME_PROFILE,
  ACE_OPT_0037_DIT_K4_RUNTIME_PROFILE,
  ACE_OPT_0056_DIT_SELECTIVE_K4_RUNTIME_PROFILE,
  type AceDitDenseRuntimeProfile,
} from "./dit-fp16-package.js";
import {
  ACE_TURBO_EIGHT_SAMPLER_SCHEDULE_PROFILE,
  requireResolvedAceDitSamplerScheduleProfile,
  type AceDitSamplerEvaluationCount,
  type AceDitSamplerScheduleProfile,
  type AceDitSamplerScheduleProfileId,
} from "./dit-sampler-profile.js";
import type { AceOpt0056DenseRouteProfile } from
  "./kernels/dit-dense-fp16-k4-selective-exact.js";
import {
  ACE_OPT_0062_EXPECTED_QUAD_QUERY_ROUTES,
  ACE_OPT_0062_IDENTITY_COUNTER_BYTES,
  ACE_OPT_0062_IDENTITY_COUNTER_STRIDE_BYTES,
  ACE_OPT_0062_IDENTITY_COUNTER_WORDS_PER_ROUTE,
  ACE_OPT_0062_IDENTITY_OUTPUT_ELEMENTS,
  type AceOpt0062AttentionRouteProfile,
} from "./kernels/attention-quad-query-production.js";
import type { AceOpt0070ProductionAttentionRouteProfile } from
  "./kernels/attention-opt0070-production.js";
import {
  ACE_OPT_0081_DIT_F16_DENSE_INPUT_ROLES,
  ACE_OPT_0081_DIT_F16_DENSE_INPUT_STORAGE_PROFILE,
  type AceOpt0081F16DenseInputRole,
} from "./kernels/dit-f16-dense-input-producers.js";
import {
  isAceFixed32TiledFullAttentionShape,
  planAceFixed32TiledFullAttention,
  type AceAttentionRuntimeConfiguration,
} from "./kernels/attention.js";

const STORAGE_ALIGNMENT = 256;
const FLOAT32_BYTES = Float32Array.BYTES_PER_ELEMENT;
const UINT32_BYTES = Uint32Array.BYTES_PER_ELEMENT;
const LAYER_COUNT = 24;
const DENOISING_EVALUATIONS = 8;
const STATIC_QUANTA = 1 + LAYER_COUNT;
const EVALUATION_QUANTA = 1 + 1 + LAYER_COUNT + 1 + 1;
const LAYER_MODULATION_GROUPS = 6;
const OPT_0018_RECONCILIATION_TOLERANCE_MS = 1e-6;
const OPT_0018_SLOWEST_COMMAND_COUNT = 16;

export interface AceDitBackendInputs {
  /** Logical activation values `[batch,conditionTokens,2048]`. */
  readonly condition: Float32Array;
  /** Logical context values `[batch,latentFrames,128]`. */
  readonly context: Float32Array;
  /** Seeded logical latent values `[batch,latentFrames,64]`. */
  readonly initialLatent: Float32Array;
}

/** @internal Frozen control/candidate identities for the OPT-0081 layer gate. */
export type AceOpt0081RepresentativeArm = "A" | "B";
export type AceOpt0081RepresentativeLayer = 0 | 1;

export const ACE_OPT_0081_REPRESENTATIVE_DENSE_TAPS = Object.freeze([
  "selfQueryFlat",
  "selfKeyFlat",
  "selfValueFlat",
  "selfProjectedAttention",
  "crossQueryFlat",
  "crossProjectedAttention",
  "gate",
  "up",
  "projectedMlp",
] as const satisfies readonly (keyof AceDitLayerScratch)[]);

export type AceOpt0081RepresentativeDenseTap =
  typeof ACE_OPT_0081_REPRESENTATIVE_DENSE_TAPS[number];

export type AceOpt0081RepresentativeTap =
  | AceOpt0081F16DenseInputRole
  | AceOpt0081RepresentativeDenseTap
  | "layerOutput";

export interface AceOpt0081RepresentativeDitOwnerOptions {
  readonly device: GPUDevice;
  readonly executionProfile: AceExecutionProfile;
  readonly subgroupMinSize?: number;
  readonly subgroupMaxSize?: number;
  readonly shape: AceDitEvaluationShape;
  readonly inputs: AceDitBackendInputs;
  readonly dcwConfiguration: AceDynamicConditionalWeightingConfiguration;
  /** Ownership transfers at call entry, including on validation failure. */
  readonly ownedDitWeights: AceGpuTensorPhase;
  /** Exact authenticated revision-7 dense phase; ownership also transfers. */
  readonly ownedDitDenseWeights: AceGpuTensorPhase;
  readonly ditDenseRuntimeProfile: AceDitDenseRuntimeProfile;
  readonly ditAttentionRuntimeProfile: AceDitAttentionRuntimeProfile;
  readonly signal?: AbortSignal;
  /** @internal Deterministic timing seam; the browser gate omits it. */
  readonly now?: () => number;
  /** @internal Receives fail-closed setup cleanup evidence when no owner exists. */
  readonly onSetupCleanup?: (
    evidence: AceOpt0081RepresentativeSetupCleanupEvidence,
  ) => void;
  /** @internal Browser preflight only; omission is mandatory for timed owner. */
  readonly setupFailurePoint?:
    | "after-control-arm"
    | "after-candidate-arm"
    | "after-readback";
  /** @internal Evidence from a separate failpoint owner attempt. */
  readonly verifiedSetupFailureCleanup?:
    AceOpt0081RepresentativeSetupCleanupEvidence;
}

export interface AceOpt0081RepresentativeSetupCleanupEvidence {
  readonly schema: "ace-opt-0081-representative-setup-cleanup-v1";
  readonly createdGraphBufferCount: number;
  readonly destroyedGraphBufferCount: number;
  readonly liveGraphBufferCount: 0;
  readonly liveGraphByteCount: 0;
  readonly runtimeOwnerCount: number;
  readonly destroyedRuntimeOwnerCount: number;
  readonly residentModelDestroyed: true;
  readonly mappedRangeCount: 0;
  readonly unmappedRangeCount: 0;
  readonly liveMapCount: 0;
  readonly pendingDescriptorCount: 0;
  readonly activeCallbackCount: 0;
  readonly activeLeaseCount: 0;
  readonly armReleaseCount: number;
  readonly drainOrderViolationCount: 0;
}

export class AceOpt0081RepresentativeInjectedSetupFailure extends Error {
  constructor(
    readonly point: NonNullable<
      AceOpt0081RepresentativeDitOwnerOptions["setupFailurePoint"]
    >,
  ) {
    super(`Injected OPT-0081 setup failure ${point}`);
    this.name = "AceOpt0081RepresentativeInjectedSetupFailure";
  }
}

export interface AceOpt0081RepresentativeMemoryAccounting {
  readonly sharedResidentWeightBytes: number;
  readonly controlArenaBytes: 674_815_488;
  readonly candidateArenaBytes: 601_087_488;
  readonly arenaSavingBytes: 73_728_000;
  readonly simultaneousArenaBytes: 1_275_902_976;
  readonly controlRoleCount: 123;
  readonly candidateRoleCount: 123;
  readonly controlSlotCount: 98;
  readonly candidateSlotCount: 98;
  readonly candidateSelectedDenseInputSlotIndices:
    readonly [61, 69, 72, 77, 81, 84];
  readonly controlLargestArenaBindingBytes: number;
  readonly candidateLargestArenaBindingBytes: number;
  readonly largestArenaBindingIncreaseBytes: 0;
  readonly timedCastOrCopyAuxiliaryBufferCount: 0;
  /** Untimed only; per-role guard bands never enter the timed arenas. */
  readonly correctnessGuardedBytes: number;
  readonly correctnessReadbackBytes: number;
  readonly accountedGpuBytes: number;
}

export interface AceOpt0081RepresentativePhysicalQuantum {
  /** Index within the timed slice; add 25 for the production physical index. */
  readonly slicePhysicalIndex: number;
  readonly productionPhysicalIndex: number;
  readonly logicalLabel: string;
  readonly logicalSubquantumIndex: number;
  readonly logicalSubquantumCount: number;
  readonly descriptor: AceGpuEncodeSequenceQuantumDescriptor;
}

export interface AceOpt0081RepresentativeTopology {
  readonly schema: "ace-opt-0081-representative-topology-v1";
  readonly precomputeCommandBufferCountPerArm: 25;
  readonly timedCommandBufferCount: 28;
  readonly phaseCommandBufferCounts: readonly [28];
  readonly timestepCommandBufferCount: 1;
  readonly inputProjectionCommandBufferCount: 1;
  readonly layerCommandBufferCounts: readonly [11, 15];
  readonly completionEpochCount: 7;
  readonly trueQueueDrainCount: 7;
  readonly cooperativeIdleTurns: 6;
  readonly maximumOutstandingCommandBuffers: 2;
  readonly maximumPendingDescriptorCount: 2;
  readonly attentionRuntimeProfile:
    typeof ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE;
  readonly layerAttentionRoutes: readonly [
    Readonly<{ readonly layer: 0; readonly self: "query8-sliding";
      readonly cross: "query8-cross" }>,
    Readonly<{ readonly layer: 1; readonly self: "quad-query32-full";
      readonly cross: "query8-cross" }>,
  ];
  readonly producerStoreCount: 12;
  readonly denseConsumerCount: 18;
  readonly descriptorOrTapCommandCount: 0;
  readonly timestampQueryCount: 0;
  readonly measurementReadbackCount: 0;
  readonly measurementMapCount: 0;
  readonly fourthEpochProductionPhysicalIndices: readonly [37, 38, 39, 40];
  readonly controlPhysicalQuanta:
    readonly AceOpt0081RepresentativePhysicalQuantum[];
  readonly candidatePhysicalQuanta:
    readonly AceOpt0081RepresentativePhysicalQuantum[];
}

export interface AceOpt0081RepresentativeRunOptions {
  readonly signal?: AbortSignal;
}

export interface AceOpt0081RepresentativeTimedHooks {
  readonly onCommandBufferCompleted?: (
    timing: AceDepth2Epoch4CommandBufferCompletionTiming,
    progress: AceDepth2Epoch4SchedulingProgress,
  ) => void;
  readonly onCompletionEpochDrained?: (
    timing: AceDepth2Epoch4CompletionEpochTiming,
  ) => void;
}

export interface AceOpt0081RepresentativeTimedSliceResult {
  readonly schema: "ace-opt-0081-representative-timed-slice-v1";
  readonly arm: AceOpt0081RepresentativeArm;
  /** Captured inside the owner immediately before scheduler entry. */
  readonly startedAtPerformanceMs: number;
  /** Captured inside the owner immediately after the terminal true drain. */
  readonly drainedAtPerformanceMs: number;
  readonly wallMs: number;
  readonly completionEpochWallsMs:
    readonly [number, number, number, number, number, number, number];
  /** Cumulative under depth two and therefore deliberately non-additive. */
  readonly submitThroughCompletionFenceMs: readonly number[];
  readonly commandBuffersSubmitted: 28;
  readonly completionFenceRequestedCount: 28;
  readonly completionFenceSettledCount: 28;
  readonly completionFenceRejectedCount: 0;
  readonly trueQueueDrainCount: 7;
  readonly completionEpochCount: 7;
  readonly requestedCooperativeIdleMs: 6;
  readonly cooperativeIdleTurns: 6;
  readonly maximumOutstandingCommandBuffers: 2;
  readonly maximumPendingDescriptorCount: 2;
  readonly pendingDescriptorCountAfterRun: 0;
  readonly topology: AceOpt0081RepresentativeTopology;
}

interface AceOpt0081RepresentativeCheckpointSnapshotBase {
  readonly schema: "ace-opt-0081-representative-checkpoint-v1";
  readonly arm: AceOpt0081RepresentativeArm;
  readonly layer: AceOpt0081RepresentativeLayer;
  readonly tap: AceOpt0081RepresentativeTap;
  readonly elementCount: number;
  readonly byteLength: number;
  readonly bytes: Uint8Array<ArrayBuffer>;
  readonly nonFiniteCount: number;
  readonly qNaNPrefillCount: number;
  readonly positiveZeroCount: number;
  readonly negativeZeroCount: number;
  readonly completeQNaNOverwrite: boolean;
  readonly firstValidWritten: boolean;
  readonly lastValidWritten: boolean;
  readonly rows2240Through2249Written: boolean;
  readonly readbackPrefixGuardIntact: boolean;
  readonly readbackSuffixGuardIntact: boolean;
  readonly adjacentBeforeGuardIntact: boolean;
  readonly adjacentAfterGuardIntact: boolean;
}

export type AceOpt0081RepresentativeCheckpointSnapshot =
  | (AceOpt0081RepresentativeCheckpointSnapshotBase & Readonly<{
      readonly storage: "u16";
      readonly words: Uint16Array<ArrayBuffer>;
    }>)
  | (AceOpt0081RepresentativeCheckpointSnapshotBase & Readonly<{
      readonly storage: "u32";
      readonly words: Uint32Array<ArrayBuffer>;
    }>);

export interface AceOpt0081RepresentativeCheckpointOptions
  extends AceOpt0081RepresentativeRunOptions {
  readonly arm: AceOpt0081RepresentativeArm;
  readonly layer: AceOpt0081RepresentativeLayer;
  readonly tap: AceOpt0081RepresentativeTap;
}

export type AceOpt0081RepresentativeCheckpointSubstage =
  | "compile-guarded-target"
  | "restore-canonical-arena"
  | "reset-initial-guards"
  | "prefill-layer-0-target"
  | "execute-through-layer-0"
  | "reset-layer-1-guards"
  | "prefill-layer-1-target"
  | "execute-layer-1"
  | "readback-and-map";

export class AceOpt0081RepresentativeCheckpointExecutionError extends Error {
  override readonly name =
    "AceOpt0081RepresentativeCheckpointExecutionError";
  readonly causeName: unknown;
  readonly causeMessage: unknown;
  readonly causeCode: unknown;

  constructor(
    readonly arm: AceOpt0081RepresentativeArm,
    readonly layer: AceOpt0081RepresentativeLayer,
    readonly tap: AceOpt0081RepresentativeTap,
    readonly substage: AceOpt0081RepresentativeCheckpointSubstage,
    cause: unknown,
  ) {
    super(
      `OPT-0081 ${arm} layer ${layer} ${tap} checkpoint failed during ${substage}`,
      { cause },
    );
    this.causeName = readOpt0081FailureProperty(cause, "name");
    this.causeMessage = readOpt0081FailureProperty(cause, "message");
    this.causeCode = readOpt0081FailureProperty(cause, "code");
  }
}

function readOpt0081FailureProperty(value: unknown, key: string): unknown {
  if (
    (typeof value !== "object" || value === null) &&
    typeof value !== "function"
  ) return undefined;
  try {
    return (value as Record<string, unknown>)[key];
  } catch {
    return undefined;
  }
}

export interface AceOpt0081RepresentativeCancellationEvidence {
  readonly schema: "ace-opt-0081-representative-cancellation-v1";
  readonly arm: AceOpt0081RepresentativeArm;
  readonly residentArmReused: true;
  readonly observedCommandBufferIndex: 0;
  readonly successorAlreadySubmitted: true;
  readonly commandBuffersCreated: 2;
  readonly completionCallbackCount: 1;
  readonly completionCallbackCountAfterAbort: 0;
  readonly noBackfillAfterObservation: true;
  readonly originalReasonPreserved: true;
  readonly submittedFencesSettledBeforeReturn: true;
  readonly settlementWallMs: number;
  readonly cleanupWithinOneSecond: boolean;
  readonly maximumPendingDescriptorCount: 2;
  readonly pendingDescriptorCountAfterRun: 0;
  readonly temporaryCreatedGraphBufferCount: 0;
  readonly temporaryDestroyedGraphBufferCount: 0;
  readonly temporaryLiveGraphBufferCountAfterCleanup: 0;
  readonly temporaryLiveGraphByteCountAfterCleanup: 0;
  readonly temporaryRuntimeOwnerCount: 0;
  readonly temporaryDestroyedRuntimeOwnerCount: 0;
}

export interface AceOpt0081RepresentativeDeviceLossCleanupEvidence {
  readonly schema: "ace-opt-0081-representative-device-loss-cleanup-v1";
  readonly deviceLossInduced: true;
  readonly deviceLossObserved: true;
  readonly ownerDestroyedAfterLoss: true;
  readonly liveGraphBufferCount: 0;
  readonly liveGraphByteCount: 0;
  readonly liveMapCount: 0;
  readonly pendingDescriptorCount: 0;
  readonly activeCallbackCount: 0;
  readonly activeLeaseCount: 0;
  readonly idempotentDestroyVerified: true;
  readonly postDestroyRejected: true;
}

export interface AceOpt0081RepresentativeLifecycleSnapshot {
  readonly schema: "ace-opt-0081-representative-lifecycle-v1";
  readonly state: "ready" | "destroying" | "destroyed";
  /** Authenticated phase, arena, guarded-target, and readback buffers only. */
  readonly createdGraphBufferCount: number;
  readonly destroyedGraphBufferCount: number;
  /** Aliases name the same graph-visible buffer inventory for receipts. */
  readonly createdBufferCount: number;
  readonly destroyedBufferCount: number;
  readonly liveGraphBufferCount: number;
  readonly liveGraphByteCount: number;
  readonly liveBufferCount: number;
  readonly liveByteCount: number;
  readonly maximumLiveGraphByteCount: number;
  readonly mappedRangeCount: number;
  readonly unmappedRangeCount: number;
  readonly liveMapCount: number;
  readonly pendingDescriptorCount: number;
  readonly maximumPendingDescriptorCount: number;
  readonly activeCallbackCount: number;
  readonly activeLeaseCount: number;
  readonly callbackCount: number;
  readonly leaseCount: number;
  readonly maximumActiveLeaseCount: number;
  readonly correctnessTargetCount: 0 | 1;
  readonly maximumCorrectnessTargetCount: 1;
  readonly correctnessRuntimeCount: 0 | 1;
  readonly maximumCorrectnessRuntimeCount: 0 | 1;
  readonly maximumCorrectnessTargetBytes: number;
  readonly correctnessTargetCompilationCount: number;
  readonly correctnessTargetReuseCount: number;
  readonly checkpointSnapshotCount: number;
  readonly maximumDetachedCheckpointBytes: number;
  readonly resetOrPrefillCount: number;
  readonly profileSwitchCount: number;
  readonly snapshotMapCount: number;
  readonly guardedTargetReleaseCount: number;
  readonly armReleaseCount: number;
  readonly drainOrderViolationCount: number;
  readonly profileSwitchWhilePendingCount: number;
  readonly runtimeOwnerCount: number;
  readonly destroyedRuntimeOwnerCount: number;
  readonly residentModelDestroyed: boolean;
  readonly precomputeCompleted: boolean;
  readonly destroyCallCount: number;
  readonly postDestroyRejectedOperationCount: number;
  readonly setupFailureCleanupExecuted: boolean;
  readonly deviceLossCleanupExecuted: boolean;
}

/** @internal Harness-facing structural surface; the production worker omits it. */
export interface AceOpt0081RepresentativeDitSession {
  readonly memory: AceOpt0081RepresentativeMemoryAccounting;
  readonly topology: AceOpt0081RepresentativeTopology;
  readonly lifecycle: AceOpt0081RepresentativeLifecycleSnapshot;
  runPrecompute(options?: AceOpt0081RepresentativeRunOptions): Promise<void>;
  /**
   * Untimed correctness-only snapshot scope. It captures both post-precompute
   * arenas on the GPU so every A1/A2/B1/B2 checkpoint starts from identical
   * graph state without mirroring an arena through JavaScript or WASM memory.
   */
  beginCorrectnessCheckpointing(
    options?: AceOpt0081RepresentativeRunOptions,
  ): Promise<void>;
  /** Restores both canonical arenas and releases the GPU-only snapshots. */
  endCorrectnessCheckpointing(
    options?: AceOpt0081RepresentativeRunOptions,
  ): Promise<void>;
  snapshotCheckpoint(
    options: AceOpt0081RepresentativeCheckpointOptions,
  ): Promise<AceOpt0081RepresentativeCheckpointSnapshot>;
  warmup(
    arm: AceOpt0081RepresentativeArm,
    options?: AceOpt0081RepresentativeRunOptions,
  ): Promise<void>;
  runTimedSlice(
    arm: AceOpt0081RepresentativeArm,
    options?: AceOpt0081RepresentativeRunOptions,
    hooks?: AceOpt0081RepresentativeTimedHooks,
  ): Promise<AceOpt0081RepresentativeTimedSliceResult>;
  runCancellationPreflight(
    arm?: AceOpt0081RepresentativeArm,
    options?: AceOpt0081RepresentativeRunOptions,
  ): Promise<AceOpt0081RepresentativeCancellationEvidence>;
  /** Terminal preflight: intentionally destroys this diagnostic device. */
  runDeviceLossCleanupPreflight():
    Promise<AceOpt0081RepresentativeDeviceLossCleanupEvidence>;
  lifecycleSnapshot(): AceOpt0081RepresentativeLifecycleSnapshot;
  destroy(): Promise<void>;
}

export interface AceDitGpuBackendOptions {
  readonly device: GPUDevice;
  readonly executionProfile: AceExecutionProfile;
  readonly subgroupMinSize?: number;
  readonly subgroupMaxSize?: number;
  readonly shape: AceDitEvaluationShape;
  readonly inputs: AceDitBackendInputs;
  readonly dcwConfiguration: AceDynamicConditionalWeightingConfiguration;
  /** Authenticated internal schedule; omission preserves exact product eight. */
  readonly samplerScheduleProfile?: Readonly<AceDitSamplerScheduleProfile>;
  /** Ownership transfers at call entry, including on validation failure. */
  readonly ownedDitWeights: AceGpuTensorPhase;
  /** Exact authenticated mixed 24-layer replacement; ownership transfers too. */
  readonly ownedDitDenseWeights?: AceGpuTensorPhase;
  /** Served production passes OPT-0037; omission is only for old harnesses. */
  readonly ditDenseRuntimeProfile?: AceDitDenseRuntimeProfile;
  /** OPT-0062 diagnostic seam; omission preserves production query8. */
  readonly ditAttentionRuntimeProfile?: AceDitAttentionRuntimeProfile;
  /** OPT-0081 representative-layer diagnostic; never selected by production. */
  readonly ditDenseInputStorageProfile?: AceDitDenseInputStorageProfile;
  /** Factory cancellation remains a lifetime cancellation after creation. */
  readonly signal?: AbortSignal;
  /** Capture-free product scheduling selected by the pipeline. */
  readonly ditSubmissionPolicy?: AceDitSubmissionPolicy;
  readonly onProgress?: (progress: AceDitGpuBackendProgress) => void;
  /** OPT-0018 capture-only command attribution. */
  readonly onCommandProfile?: (profile: AceOpt0018DitCommandProfile) => void;
  /** OPT-0067 one-evaluation capture; mutually exclusive with other captures. */
  readonly onOpt0067EvaluationProfile?: (
    profile: AceOpt0067DitCommandProfile,
  ) => void;
  /** OPT-0067 only: execute the production precompute and evaluation 0. */
  readonly opt0067EvaluationLimit?: 1;
  /** OPT-0080 exact evaluation-0 scheduling profile and capture. */
  readonly opt0080SchedulingProfile?: AceDitOpt0080SchedulingProfile;
  readonly onOpt0080EvaluationProfile?: (
    profile: AceOpt0080DitCommandProfile,
  ) => void;
  /** OPT-0080 benchmark-only full eight-evaluation confirmation capture. */
  readonly opt0080FullSchedulingProfile?: AceDitOpt0080SchedulingProfile;
  readonly onOpt0080FullProfile?: (
    profile: AceOpt0080FullDitCommandProfile,
  ) => void;
  /** OPT-0080 cancellation preflight; invoked after internal attribution. */
  readonly onOpt0080CommandBufferCompleted?: (
    completion: AceDitOpt0080CommandBufferCompletion,
  ) => void;
  /** OPT-0034 benchmark seam; omission preserves the shipped batch of one. */
  readonly physicalQuantaPerCommandBuffer?:
    AceDitPhysicalQuantaPerCommandBuffer;
  /** OPT-0034 exact batch telemetry; mutually exclusive with OPT-0018 capture. */
  readonly onSchedulingProfile?: (
    profile: AceOpt0034DitSchedulingProfile,
  ) => void;
  /** OPT-0056 benchmark seam; copies all eight sampler outputs in-place. */
  readonly captureEvaluationLatents?: true;
  /** OPT-0062 correctness-only query8/quad actual-layer raw-U32 oracle. */
  readonly captureOpt0062AttentionIdentity?: true;
  /** OPT-0055/0065 diagnostic: trajectory plus raw evaluation-0 velocity. */
  readonly captureSamplerScheduleEvidence?: true;
}

export interface AceOpt0018DitProfileAggregate {
  readonly commandBufferCount: number;
  readonly primitiveCount: number;
  readonly scheduledMultiplyAdds: number;
  readonly submitThroughDrainMs: number;
  readonly maximumSubmitThroughDrainMs: number;
}

export interface AceOpt0018DitSlowCommand {
  readonly physicalIndex: number;
  readonly family: AceDitProfileFamily;
  readonly evaluation: number | null;
  readonly layer: number | null;
  readonly submitThroughDrainMs: number;
}

export interface AceOpt0018DitCommandProfile {
  readonly schema: "ace-dit-m2250-command-profile-v1";
  readonly descriptorTable: AceDitPhysicalCommandDescriptorTable;
  readonly graphCommandBufferCount: 2_553;
  readonly readbackCommandBufferCount: 1;
  readonly totalCommandBufferCount: 2_554;
  readonly timings: readonly number[];
  readonly graphSubmitThroughDrainMs: number;
  readonly graphWallMs: number;
  readonly graphRequestedIdleMs: number;
  readonly graphResidualMs: number;
  readonly graphToReadbackRequestedIdleMs: 1;
  readonly graphToReadbackObservedIdleMs: number;
  readonly readbackSubmitThroughDrainMs: number;
  readonly readbackMapDetachMs: number;
  readonly backendWallMs: number;
  readonly backendResidualMs: number;
  readonly families: Readonly<Record<
    AceDitProfileFamily,
    AceOpt0018DitProfileAggregate
  >>;
  readonly precompute: AceOpt0018DitProfileAggregate;
  readonly evaluations: readonly AceOpt0018DitProfileAggregate[];
  readonly familyByBucket: Readonly<Record<
    AceDitProfileFamily,
    readonly AceOpt0018DitProfileAggregate[]
  >>;
  readonly slowest: readonly AceOpt0018DitSlowCommand[];
  readonly reconciliationToleranceMs: number;
  readonly timingStorageBytes: number;
}

export interface AceOpt0067DitCommandProfile {
  readonly schema: "ace-dit-opt0067-evaluation0-command-profile-v1";
  /** Full compiled-table identity; only the exact prefix below executes. */
  readonly descriptorTable: AceDitPhysicalCommandDescriptorTable;
  readonly graphCommandBufferCount: 341;
  readonly readbackCommandBufferCount: 1;
  readonly totalCommandBufferCount: 342;
  readonly graphQueueDrainCount: 341;
  readonly totalQueueDrainCount: 342;
  readonly evaluationCommandBufferCount: 316;
  readonly timings: readonly number[];
  readonly graphSubmitThroughDrainMs: number;
  readonly graphWallMs: number;
  readonly graphRequestedIdleMs: 340;
  readonly graphResidualMs: number;
  /** From the drained precompute boundary through evaluation 0's final drain. */
  readonly evaluationWallMs: number;
  readonly evaluationRequestedIdleMs: 316;
  readonly evaluationResidualMs: number;
  readonly graphToReadbackRequestedIdleMs: 1;
  readonly graphToReadbackObservedIdleMs: number;
  readonly readbackSubmitThroughDrainMs: number;
  readonly readbackMapDetachMs: number;
  readonly backendWallMs: number;
  readonly backendResidualMs: number;
  readonly families: Readonly<Record<
    AceDitProfileFamily,
    AceOpt0018DitProfileAggregate
  >>;
  readonly precompute: AceOpt0018DitProfileAggregate;
  readonly evaluation: AceOpt0018DitProfileAggregate;
  readonly familyByBucket: Readonly<Record<
    AceDitProfileFamily,
    readonly [AceOpt0018DitProfileAggregate, AceOpt0018DitProfileAggregate]
  >>;
  readonly slowest: readonly AceOpt0018DitSlowCommand[];
  readonly reconciliationToleranceMs: number;
  readonly timingStorageBytes: number;
}

export interface AceOpt0080CompletionEpochProfile
  extends AceDitOpt0080CompletionEpochTiming {}

export interface AceOpt0080DitTopologyEvidence {
  readonly schedulingProfile: AceDitOpt0080SchedulingProfile;
  /** Member count for the full compiled 2,553-command descriptor table. */
  readonly descriptorTableMemberCount: 6_833;
  readonly graphCommandBufferCount: 341;
  readonly readbackCommandBufferCount: 1;
  readonly totalCommandBufferCount: 342;
  readonly commandBuffersSubmitted: 342;
  readonly completionFenceRequestedCount: 342;
  readonly completionFenceSettledCount: 342;
  readonly completionFenceRejectedCount: 0;
  readonly graphTrueQueueDrainCount: 86 | 341;
  readonly totalTrueQueueDrainCount: 87 | 342;
  readonly graphCompletionEpochCount: 86 | 341;
  readonly graphCooperativeIdleTurns: 85 | 340;
  readonly totalCooperativeIdleTurns: 86 | 341;
  readonly graphRequestedCooperativeIdleMs: 85 | 340;
  readonly totalRequestedCooperativeIdleMs: 86 | 341;
  readonly maximumOutstandingCommandBuffers: 1 | 2;
  readonly maximumPendingDescriptorCount: 1 | 2;
  readonly pendingDescriptorCountAfterCleanup: 0;
  readonly graphCompletionEpochs:
    readonly AceOpt0080CompletionEpochProfile[];
  /** Cumulative and overlapping under depth two; this array is non-additive. */
  readonly submitThroughCompletionFenceMs: readonly number[];
  readonly graphToReadbackRequestedIdleMs: 1;
  readonly readbackSubmitThroughCompletionFenceMs: number;
}

/**
 * OPT-0080 keeps cumulative per-submit fence latencies only as non-additive
 * diagnostics. Authoritative performance fields are the phase, graph, and
 * backend walls plus disjoint completion-epoch timings.
 */
export interface AceOpt0080DitCommandProfile {
  readonly schema: "ace-dit-opt0080-evaluation0-command-profile-v1";
  readonly schedulingProfile: AceDitOpt0080SchedulingProfile;
  readonly descriptorTable: AceDitPhysicalCommandDescriptorTable;
  readonly precomputeWallMs: number;
  readonly evaluationWallMs: number;
  readonly graphWallMs: number;
  readonly graphToReadbackObservedIdleMs: number;
  readonly readbackSubmitThroughCompletionFenceMs: number;
  readonly readbackMapDetachMs: number;
  readonly backendWallMs: number;
  readonly topology: AceOpt0080DitTopologyEvidence;
}

export interface AceOpt0080FullDitTopologyEvidence {
  readonly schedulingProfile: AceDitOpt0080SchedulingProfile;
  readonly descriptorTableMemberCount: 6_833;
  readonly graphCommandBufferCount: 2_553;
  readonly readbackCommandBufferCount: 1;
  readonly totalCommandBufferCount: 2_554;
  readonly commandBuffersSubmitted: 2_554;
  readonly completionFenceRequestedCount: 2_554;
  readonly completionFenceSettledCount: 2_554;
  readonly completionFenceRejectedCount: 0;
  readonly graphTrueQueueDrainCount: 639 | 2_553;
  readonly totalTrueQueueDrainCount: 640 | 2_554;
  readonly graphCompletionEpochCount: 639 | 2_553;
  readonly graphCooperativeIdleTurns: 638 | 2_552;
  readonly totalCooperativeIdleTurns: 639 | 2_553;
  readonly graphRequestedCooperativeIdleMs: 638 | 2_552;
  readonly totalRequestedCooperativeIdleMs: 639 | 2_553;
  readonly maximumOutstandingCommandBuffers: 1 | 2;
  readonly maximumPendingDescriptorCount: 1 | 2;
  readonly pendingDescriptorCountAfterCleanup: 0;
  readonly graphCompletionEpochs:
    readonly AceOpt0080CompletionEpochProfile[];
  /** Cumulative and overlapping under depth two; this array is non-additive. */
  readonly submitThroughCompletionFenceMs: readonly number[];
  readonly graphToReadbackRequestedIdleMs: 1;
  readonly readbackSubmitThroughCompletionFenceMs: number;
}

export type AceOpt0080FullEvaluationWallTuple = readonly [
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
];

/** Authoritative full-graph walls; diagnostic fence latencies remain non-additive. */
export interface AceOpt0080FullDitCommandProfile {
  readonly schema: "ace-dit-opt0080-full-command-profile-v1";
  readonly schedulingProfile: AceDitOpt0080SchedulingProfile;
  readonly descriptorTable: AceDitPhysicalCommandDescriptorTable;
  readonly precomputeWallMs: number;
  readonly evaluationWallMs: AceOpt0080FullEvaluationWallTuple;
  readonly graphWallMs: number;
  readonly graphToReadbackObservedIdleMs: number;
  readonly readbackSubmitThroughCompletionFenceMs: number;
  readonly readbackMapDetachMs: number;
  readonly backendWallMs: number;
  readonly topology: AceOpt0080FullDitTopologyEvidence;
}

export interface AceOpt0034DitSchedulingBatch {
  readonly batchIndex: number;
  readonly firstPhysicalIndex: number;
  readonly lastPhysicalIndex: number;
  readonly physicalQuantumCount: number;
  readonly primitiveCount: number;
  readonly scheduledMultiplyAdds: number;
  readonly submitThroughDrainMs: number;
}

export interface AceOpt0034DitSchedulingProfile {
  readonly schema: "ace-dit-opt0034-command-buffer-coalescing-v1";
  readonly physicalGraphQuantumCount: 2_553;
  readonly physicalQuantaPerCommandBuffer:
    AceDitPhysicalQuantaPerCommandBuffer;
  readonly graphCommandBufferCount: number;
  readonly readbackCommandBufferCount: 1;
  readonly totalCommandBufferCount: number;
  readonly graphQueueDrainCount: number;
  readonly totalQueueDrainCount: number;
  readonly graphRequestedIdleMs: number;
  readonly graphToReadbackRequestedIdleMs: 1;
  readonly graphSubmitThroughDrainMs: number;
  readonly graphWallMs: number;
  readonly graphToReadbackObservedIdleMs: number;
  readonly readbackSubmitThroughDrainMs: number;
  readonly readbackMapDetachMs: number;
  readonly backendWallMs: number;
  readonly maximumPhysicalQuantaPerBatch: number;
  readonly maximumPrimitiveCountPerBatch: number;
  readonly maximumScheduledMultiplyAddsPerBatch: number;
  readonly maximumBatchSubmitThroughDrainMs: number;
  readonly descriptorTableSha256: string;
  readonly physicalPrimitiveCount: number;
  readonly scheduledMultiplyAdds: number;
  readonly batches: readonly AceOpt0034DitSchedulingBatch[];
}

export type AceDitGpuBackendProgress =
  | Readonly<{
      stage: "compile";
      compiledQuanta: number;
      totalQuanta: number;
    }>
  | Readonly<{
      stage: "denoise";
      completedCommandBuffers: number;
      totalCommandBuffers: number;
      queueDrains: number;
      cooperativeIdleMs: number;
      completedEvaluations: number;
      graph: AceDitGraphProgress;
    }>
  | Readonly<{
      stage: "readback";
      completedCommandBuffers: number;
      totalCommandBuffers: number;
      queueDrains: number;
      cooperativeIdleMs: number;
      completedEvaluations: 1 | AceDitSamplerEvaluationCount;
    }>;

export interface AceDitGpuBackendResult {
  /** Detached FP32 NLC tensor `[batch,latentFrames,64]` for VAE transfer. */
  readonly finalLatent: Float32Array<ArrayBuffer>;
  readonly shape: Readonly<{
    batch: number;
    latentFrames: number;
    channels: 64;
  }>;
  readonly commandBuffersSubmitted: number;
  readonly queueDrains: number;
  readonly cooperativeIdleMs: number;
  readonly completedEvaluations: 1 | AceDitSamplerEvaluationCount;
  readonly evaluationLatents?: readonly Float32Array<ArrayBuffer>[];
  readonly evaluation0Velocity?: Float32Array<ArrayBuffer>;
  readonly denseRouteProfile?: AceOpt0056DenseRouteProfile;
  readonly attentionRouteProfile?:
    | AceOpt0062AttentionRouteProfile
    | AceOpt0070ProductionAttentionRouteProfile;
  readonly opt0062AttentionIdentity?: AceOpt0062AttentionIdentityResult;
}

export interface AceOpt0062AttentionIdentityRoute {
  readonly routeIndex: number;
  readonly label: string;
  readonly comparedElements: number;
  readonly mismatchCount: number;
  readonly query8NonFiniteCount: number;
  readonly quadNonFiniteCount: number;
  readonly query8CanaryCount: number;
  readonly quadCanaryCount: number;
}

export interface AceOpt0062AttentionIdentityResult {
  readonly schema: "ace-opt-0062-actual-layer-raw-u32-identity-v1";
  readonly routeCount: typeof ACE_OPT_0062_EXPECTED_QUAD_QUERY_ROUTES;
  readonly outputElementsPerRoute:
    typeof ACE_OPT_0062_IDENTITY_OUTPUT_ELEMENTS;
  readonly totalComparedElements: number;
  readonly totalMismatchCount: number;
  readonly totalNonFiniteCount: number;
  readonly totalCanaryCount: number;
  readonly copyCount: 1;
  readonly extraCommandBufferCount: 0;
  readonly extraQueueDrainCount: 0;
  readonly routes: readonly AceOpt0062AttentionIdentityRoute[];
}

export interface AceDitBackendArenaRole {
  readonly key: string;
  readonly label: string;
  readonly byteLength: number;
  readonly lifetimes: readonly Readonly<{
    firstQuantum: number;
    lastQuantum: number;
  }>[];
}

export interface AceDitBackendArenaSlot {
  readonly index: number;
  readonly byteLength: number;
  readonly roles: readonly string[];
}

export interface AceDitBackendArenaPlan {
  readonly denseInputStorageProfile?: AceDitDenseInputStorageProfile;
  readonly roles: readonly AceDitBackendArenaRole[];
  readonly slots: readonly AceDitBackendArenaSlot[];
  readonly roleToSlot: Readonly<Record<string, number>>;
  readonly logicalRoleBytes: number;
  readonly allocatedArenaBytes: number;
  readonly alignmentOverheadBytes: number;
  readonly lifetimeReuseSavingsBytes: number;
}

export interface AceDitGpuBackendMemoryPlan {
  readonly modelProfile: AceModelProfileId;
  readonly denseInputStorageProfile?: AceDitDenseInputStorageProfile;
  readonly gemmBackend: AceDitGemmBackend;
  readonly graph: AceDitGraphMemoryPlan;
  readonly residentWeightBytes: number;
  readonly arena: AceDitBackendArenaPlan;
  readonly readbackBufferBytes: number;
  readonly evaluationReadbackBytes: number;
  readonly evaluation0VelocityReadbackBytes: number;
  readonly opt0062AttentionIdentityCounterBytes: number;
  readonly opt0062AttentionIdentityReadbackBytes: number;
  readonly accountedGpuBytes: number;
  readonly callerInputSnapshotBytes: number;
  readonly controlSnapshotBytes: number;
  readonly maximumEncodingStagingBytes: number;
  readonly detachedFinalLatentBytes: number;
  readonly detachedEvaluationLatentBytes: number;
  readonly detachedEvaluation0VelocityBytes: number;
  readonly detachedOpt0062AttentionIdentityBytes: number;
  readonly boundedCpuBytes: number;
  /** Logical liveness stages; subquanta never shorten a role lifetime. */
  readonly samplerScheduleProfileId: AceDitSamplerScheduleProfileId;
  readonly evaluationCount: AceDitSamplerEvaluationCount;
  readonly logicalGraphQuantumCount: number;
  readonly physicalGraphQuantumCount: number;
  readonly physicalQuantaPerCommandBuffer:
    AceDitPhysicalQuantaPerCommandBuffer;
  readonly graphCommandBufferCount: number;
  readonly readbackCommandBufferCount: 1;
  /** Exact coalesced graph submissions plus the separate final readback. */
  readonly commandBufferCount: number;
}

export class AceDitBackendDeviceLostError extends Error {
  override readonly name = "AceDitBackendDeviceLostError";

  constructor(info: Pick<GPUDeviceLostInfo, "reason" | "message">) {
    super(
      `ACE DiT WebGPU device lost (${info.reason}): ` +
        (info.message || "no device message"),
    );
  }
}

interface AceDitGraphRunner {
  readonly commandBufferCount: number;
  readonly physicalCommandDescriptors?:
    | AceDitPhysicalCommandDescriptorTable
    | undefined;
  run(options: AceDitGraphRunOptions): Promise<AceDitGraphRunResult>;
  destroy(): Promise<void>;
}

export interface AceDitPreparedGpuResources {
  readonly device: GPUDevice;
  readonly modelProfile: AceModelProfileId;
  readonly denseInputStorageProfile?: AceDitDenseInputStorageProfile;
  readonly gemmBackend: AceDitGemmBackend;
  readonly shape: AceDitEvaluationPlan;
  readonly graph: AceDitGraphRunner;
  readonly readback: GPUBuffer;
  readonly evaluationReadbacks?: readonly GPUBuffer[];
  readonly evaluation0VelocityReadback?: GPUBuffer;
  readonly samplerScheduleProfile?: Readonly<AceDitSamplerScheduleProfile>;
  readonly opt0062AttentionIdentityCounters?: GPUBuffer;
  readonly opt0062AttentionIdentityReadback?: GPUBuffer;
  readonly memory: AceDitGpuBackendMemoryPlan;
  readonly signal?: AbortSignal;
  readonly ditSubmissionPolicy?: AceDitSubmissionPolicy;
  readonly onProgress?: (progress: AceDitGpuBackendProgress) => void;
  readonly onCommandProfile?: (profile: AceOpt0018DitCommandProfile) => void;
  readonly onOpt0067EvaluationProfile?: (
    profile: AceOpt0067DitCommandProfile,
  ) => void;
  readonly opt0067EvaluationLimit?: 1;
  readonly opt0080SchedulingProfile?: AceDitOpt0080SchedulingProfile;
  readonly onOpt0080EvaluationProfile?: (
    profile: AceOpt0080DitCommandProfile,
  ) => void;
  readonly opt0080FullSchedulingProfile?: AceDitOpt0080SchedulingProfile;
  readonly onOpt0080FullProfile?: (
    profile: AceOpt0080FullDitCommandProfile,
  ) => void;
  readonly onOpt0080CommandBufferCompleted?: (
    completion: AceDitOpt0080CommandBufferCompletion,
  ) => void;
  readonly physicalQuantaPerCommandBuffer?:
    AceDitPhysicalQuantaPerCommandBuffer;
  readonly onSchedulingProfile?: (
    profile: AceOpt0034DitSchedulingProfile,
  ) => void;
  /** @internal Test seam; production uses a real one-millisecond timer. */
  readonly yieldQueueIdle?: () => Promise<void>;
  /** @internal Deterministic capture timing seam. */
  readonly now?: () => number;
  /** Graph destruction is separate and happens before this callback. */
  destroy(): void;
}

export type AceDitGemmBackend =
  | "portable"
  | "fixed32-subgroups"
  | "mixed-opt-0009"
  | "mixed-opt-0037-k4"
  | "mixed-opt-0056-selective";

export interface AceDitGemmSelection {
  readonly modelProfile: AceModelProfileId;
  readonly backend: AceDitGemmBackend;
  readonly gemmConfiguration:
    | Readonly<{
        backend: "portable";
        weightLayout: typeof ACE_DIT_GEMM_WEIGHT_LAYOUT;
      }>
    | Readonly<{
        backend: "subgroups";
        capability: Readonly<{
          subgroupMinSize: number;
          subgroupMaxSize: number;
        }>;
      }>;
}

export interface AceDitMixedGemmSelection {
  readonly modelProfile: "reference-bf16";
  readonly backend:
    | "mixed-opt-0009"
    | "mixed-opt-0037-k4"
    | "mixed-opt-0056-selective";
  readonly denseRuntimeProfile: AceDitDenseRuntimeProfile;
  readonly attentionRuntimeProfile: AceDitAttentionRuntimeProfile;
  readonly gemmConfiguration: Extract<
    AceDitGemmSelection["gemmConfiguration"],
    { backend: "subgroups" }
  >;
  readonly denseGemmConfiguration: Extract<
    AceDitDenseGemmRuntimeConfiguration,
    {
      backend:
        | "opt-0009-fp16-fp32"
        | "opt-0037-k4-fp16-partials"
        | "opt-0056-selective-k4-exact-down";
    }
  >;
  readonly attentionConfiguration: Extract<
    AceAttentionRuntimeConfiguration,
    {
      backend:
        | "fixed32-subgroup-query8"
        | "opt-0062-fixed32-quad-query32-full-self"
        | "opt-0070-fixed32-quad-query32-full-self-production";
    }
  >;
}

export function resolveAceDitGemmSelection(
  executionProfile: AceExecutionProfile,
  subgroupMinSize?: number,
  subgroupMaxSize?: number,
): AceDitGemmSelection {
  const expected = executionProfile.id === ACE_REFERENCE_PORTABLE_PROFILE.id
    ? ACE_REFERENCE_PORTABLE_PROFILE
    : executionProfile.id === ACE_REFERENCE_SUBGROUP_PROFILE.id
      ? ACE_REFERENCE_SUBGROUP_PROFILE
      : executionProfile.id === ACE_FP16_PORTABLE_PROFILE.id
        ? ACE_FP16_PORTABLE_PROFILE
        : undefined;
  if (
    expected === undefined ||
    executionProfile.modelProfile !== expected.modelProfile ||
    executionProfile.weightStorage !== expected.weightStorage ||
    executionProfile.matrixArithmetic !== expected.matrixArithmetic ||
    executionProfile.sensitiveReductions !== expected.sensitiveReductions ||
    executionProfile.vaeArithmetic !== expected.vaeArithmetic ||
    executionProfile.kernelBackend !== expected.kernelBackend ||
    executionProfile.requiredFeatures.length !== expected.requiredFeatures.length ||
    executionProfile.requiredFeatures.some(
      (feature, index) => feature !== expected.requiredFeatures[index],
    )
  ) {
    throw new Error("ACE DiT execution profile is not an authenticated profile");
  }
  if (expected.id !== ACE_REFERENCE_SUBGROUP_PROFILE.id) {
    return Object.freeze({
      modelProfile: expected.modelProfile,
      backend: "portable" as const,
      gemmConfiguration: Object.freeze({
        backend: "portable" as const,
        weightLayout: ACE_DIT_GEMM_WEIGHT_LAYOUT,
      }),
    });
  }
  if (subgroupMinSize !== 32 || subgroupMaxSize !== 32) {
    throw new Error(
      "ACE DiT subgroup profile requires reported fixed 32-lane subgroups",
    );
  }
  return Object.freeze({
    modelProfile: "reference-bf16" as const,
    backend: "fixed32-subgroups" as const,
    gemmConfiguration: Object.freeze({
      backend: "subgroups" as const,
      capability: Object.freeze({ subgroupMinSize, subgroupMaxSize }),
    }),
  });
}

export function resolveAceDitMixedGemmSelection(
  executionProfile: AceExecutionProfile,
  subgroupMinSize?: number,
  subgroupMaxSize?: number,
  denseRuntimeProfile: AceDitDenseRuntimeProfile =
    ACE_OPT_0009_DIT_DENSE_RUNTIME_PROFILE,
  attentionRuntimeProfile: AceDitAttentionRuntimeProfile =
    ACE_DIT_QUERY8_ATTENTION_RUNTIME_PROFILE,
  expectedQueryTokens?: number,
  expectedConditionTokens?: number,
): AceDitMixedGemmSelection {
  const reference = resolveAceDitGemmSelection(
    executionProfile,
    subgroupMinSize,
    subgroupMaxSize,
  );
  if (
    executionProfile.id !== ACE_REFERENCE_SUBGROUP_PROFILE.id ||
    reference.modelProfile !== "reference-bf16" ||
    reference.backend !== "fixed32-subgroups" ||
    reference.gemmConfiguration.backend !== "subgroups"
  ) {
    throw new Error(
      "Optimized mixed DiT requires the authenticated fixed32 reference profile",
    );
  }
  if (subgroupMinSize !== 32 || subgroupMaxSize !== 32) {
    throw new Error("Optimized mixed DiT requires fixed 32-lane subgroups");
  }
  if (
    denseRuntimeProfile !== ACE_OPT_0009_DIT_DENSE_RUNTIME_PROFILE &&
    denseRuntimeProfile !== ACE_OPT_0037_DIT_K4_RUNTIME_PROFILE &&
    denseRuntimeProfile !== ACE_OPT_0056_DIT_SELECTIVE_K4_RUNTIME_PROFILE
  ) {
    throw new Error("ACE DiT dense runtime profile is not authenticated");
  }
  const attentionProfile = requireAceDitAttentionRuntimeProfile(
    attentionRuntimeProfile,
  );
  if (
    (attentionProfile.id ===
        ACE_OPT_0062_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE ||
      attentionProfile.id ===
        ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE) &&
    denseRuntimeProfile !== ACE_OPT_0009_DIT_DENSE_RUNTIME_PROFILE
  ) {
    throw new Error(
      "OPT-0062 cannot combine with the revision-8 or selective dense profile",
    );
  }
  const selective = denseRuntimeProfile ===
    ACE_OPT_0056_DIT_SELECTIVE_K4_RUNTIME_PROFILE;
  const k4 = denseRuntimeProfile === ACE_OPT_0037_DIT_K4_RUNTIME_PROFILE;
  if (
    attentionProfile.id ===
      ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE &&
    (!Number.isSafeInteger(expectedQueryTokens) || expectedQueryTokens! < 1 ||
      !Number.isSafeInteger(expectedConditionTokens) ||
      expectedConditionTokens! < 1)
  ) {
    throw new Error(
      "OPT-0070 production attention requires authenticated graph token counts",
    );
  }
  const capability = Object.freeze({ subgroupMinSize, subgroupMaxSize });
  return Object.freeze({
    modelProfile: "reference-bf16",
    backend: selective
      ? "mixed-opt-0056-selective"
      : k4
        ? "mixed-opt-0037-k4"
        : "mixed-opt-0009",
    denseRuntimeProfile,
    attentionRuntimeProfile: attentionProfile.id,
    gemmConfiguration: Object.freeze({
      backend: "subgroups",
      capability,
    }),
    denseGemmConfiguration: Object.freeze({
      backend: selective
        ? "opt-0056-selective-k4-exact-down"
        : k4
          ? "opt-0037-k4-fp16-partials"
          : "opt-0009-fp16-fp32",
      capability,
    }),
    attentionConfiguration: attentionProfile.id ===
        ACE_OPT_0062_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE
      ? Object.freeze({
          backend: "opt-0062-fixed32-quad-query32-full-self" as const,
          runtimeProfileId:
            ACE_OPT_0062_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
          capability,
        })
      : attentionProfile.id ===
          ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE
        ? Object.freeze({
            backend:
              "opt-0070-fixed32-quad-query32-full-self-production" as const,
            runtimeProfileId:
              ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
            capability,
            expectedQueryTokens: expectedQueryTokens!,
            expectedConditionTokens: expectedConditionTokens!,
          })
      : Object.freeze({
          backend: "fixed32-subgroup-query8" as const,
          capability,
        }),
  });
}

/**
 * One-shot concrete DiT backend. Creation consumes authenticated weights,
 * snapshots and uploads bounded graph inputs, allocates exact liveness slots,
 * and compiles the fixed graph. Every ranged GEMM fragment is an independent
 * physical command even though arena liveness remains expressed in the 249
 * logical graph stages. `run` adds one final readback command.
 */
export class AceDitGpuBackend {
  readonly memory: AceDitGpuBackendMemoryPlan;

  private readonly lifetime = new AbortController();
  private readonly resources: AceDitPreparedGpuResources;
  private readonly externalSignal: AbortSignal | undefined;
  private readonly externalAbortListener: (() => void) | undefined;
  private destroyPromise: Promise<void> | undefined;
  private state: "ready" | "running" | "destroying" | "destroyed" = "ready";

  private constructor(resources: AceDitPreparedGpuResources) {
    validatePreparedResources(resources);
    resources.signal?.throwIfAborted();
    this.resources = resources;
    this.memory = resources.memory;
    this.externalSignal = resources.signal;
    this.externalAbortListener = resources.signal === undefined
      ? undefined
      : () => {
          if (!this.lifetime.signal.aborted) {
            this.lifetime.abort(resources.signal!.reason);
          }
          void this.beginDestroy(resources.signal!.reason).catch(() => undefined);
        };
    this.externalSignal?.addEventListener(
      "abort",
      this.externalAbortListener!,
      { once: true },
    );
    const weakBackend = new WeakRef(this);
    void resources.device.lost.then((info) => {
      const backend = weakBackend.deref();
      if (
        backend === undefined ||
        backend.state === "destroyed" ||
        backend.state === "destroying"
      ) {
        return;
      }
      const error = new AceDitBackendDeviceLostError(info);
      backend.lifetime.abort(error);
      void backend.beginDestroy(error).catch(() => undefined);
    });
  }

  static async create(options: AceDitGpuBackendOptions): Promise<AceDitGpuBackend> {
    const resources = await createConcreteResources(options);
    try {
      return new AceDitGpuBackend(resources);
    } catch (error) {
      await resources.graph.destroy();
      resources.destroy();
      throw error;
    }
  }

  /** @internal Deterministic coordinator seam for lifecycle/readback tests. */
  static fromPreparedResources(
    resources: AceDitPreparedGpuResources,
  ): AceDitGpuBackend {
    try {
      return new AceDitGpuBackend(resources);
    } catch (error) {
      void resources.graph.destroy();
      resources.destroy();
      throw error;
    }
  }

  async run(signal?: AbortSignal): Promise<AceDitGpuBackendResult> {
    if (this.state !== "ready") {
      throw new DOMException(
        `ACE DiT GPU backend cannot run from state ${this.state}`,
        "InvalidStateError",
      );
    }
    this.state = "running";
    const activeSignal = combineSignals([
      this.lifetime.signal,
      this.externalSignal,
      signal,
    ]);
    const physicalQuantaPerCommandBuffer =
      this.resources.physicalQuantaPerCommandBuffer ??
      ACE_DIT_PHYSICAL_QUANTA_PER_COMMAND_BUFFER;
    const profileCapture = this.resources.onCommandProfile === undefined
      ? undefined
      : createOpt0018ProfileCapture(
          this.resources.graph,
          this.resources.memory,
        );
    const opt0067Capture =
      this.resources.onOpt0067EvaluationProfile === undefined
        ? undefined
        : createOpt0067EvaluationProfileCapture(
            this.resources.graph,
            this.resources.memory,
          );
    const opt0080Capture =
      this.resources.onOpt0080EvaluationProfile === undefined
        ? undefined
        : createOpt0080EvaluationProfileCapture(
            this.resources.graph,
            this.resources.memory,
            this.resources.opt0080SchedulingProfile!,
          );
    const opt0080FullCapture =
      this.resources.onOpt0080FullProfile === undefined
        ? undefined
        : createOpt0080FullProfileCapture(
            this.resources.graph,
            this.resources.memory,
            this.resources.opt0080FullSchedulingProfile!,
          );
    const activeOpt0080Capture = opt0080Capture ?? opt0080FullCapture;
    const schedulingCapture =
      this.resources.onSchedulingProfile === undefined
        ? undefined
        : createOpt0034SchedulingCapture(
            this.resources.graph,
            this.resources.memory,
            physicalQuantaPerCommandBuffer,
          );
    const now = profileCapture === undefined && schedulingCapture === undefined &&
        opt0067Capture === undefined && activeOpt0080Capture === undefined
      ? undefined
      : this.resources.now ?? (() => performance.now());
    const backendStartedAt = now?.() ?? 0;
    let opt0067EvaluationStartedAt: number | undefined;
    let opt0080PrecomputeStartedAt: number | undefined;
    let opt0080EvaluationStartedAt: number | undefined;
    const opt0080FullPhaseStartedAt: number[] = [];
    try {
      activeSignal.throwIfAborted();
      const graphStartedAt = now?.() ?? 0;
      const graphResult = await this.resources.graph.run({
        signal: activeSignal,
        ...(this.resources.ditSubmissionPolicy === undefined
          ? {}
          : { submissionPolicy: this.resources.ditSubmissionPolicy }),
        physicalQuantaPerCommandBuffer,
        ...(this.resources.yieldQueueIdle === undefined
          ? {}
          : { yieldQueueIdle: this.resources.yieldQueueIdle }),
        onProgress: (progress) => {
          this.resources.onProgress?.({
            stage: "denoise",
            completedCommandBuffers: progress.completedCommandBuffers,
            totalCommandBuffers: progress.totalCommandBuffers + 1,
            queueDrains: progress.queueDrains,
            cooperativeIdleMs: progress.cooperativeIdleMs,
            completedEvaluations: progress.completedEvaluations,
            graph: progress,
          });
        },
        ...(profileCapture === undefined
          ? {}
          : {
              onCommandBufferDrained: (descriptor, elapsedMs) => {
                profileCapture.record(descriptor, elapsedMs);
              },
            }),
        ...(opt0067Capture === undefined
          ? {}
          : {
              onCommandBufferDrained: (descriptor, elapsedMs) => {
                opt0067Capture.record(descriptor, elapsedMs);
                if (descriptor.physicalIndex === 24) {
                  if (opt0067EvaluationStartedAt !== undefined) {
                    throw new Error("OPT-0067 evaluation boundary repeated");
                  }
                  opt0067EvaluationStartedAt = now!();
                }
              },
            }),
        ...(activeOpt0080Capture === undefined
          ? {}
          : {
              opt0080SchedulingProfile:
                (this.resources.opt0080SchedulingProfile ??
                  this.resources.opt0080FullSchedulingProfile)!,
              ...(opt0080FullCapture === undefined
                ? {}
                : { opt0080FullGraph: true as const }),
              onOpt0080PhaseStarted: (phase: AceDitOpt0080PhaseStart) => {
                const startedAt = now!();
                activeOpt0080Capture.recordPhaseStart(phase);
                if (opt0080FullCapture !== undefined) {
                  if (
                    phase.phaseIndex !== opt0080FullPhaseStartedAt.length ||
                    opt0080FullPhaseStartedAt.length >=
                      ACE_OPT_0080_M2250_FULL_PHASE_COMMAND_BUFFER_COUNTS.length
                  ) {
                    throw new Error("OPT-0080 full phase boundary repeated");
                  }
                  opt0080FullPhaseStartedAt.push(startedAt);
                } else if (phase.phaseIndex === 0) {
                  if (opt0080PrecomputeStartedAt !== undefined) {
                    throw new Error("OPT-0080 precompute boundary repeated");
                  }
                  opt0080PrecomputeStartedAt = startedAt;
                } else if (phase.phaseIndex === 1) {
                  if (opt0080EvaluationStartedAt !== undefined) {
                    throw new Error("OPT-0080 evaluation boundary repeated");
                  }
                  opt0080EvaluationStartedAt = startedAt;
                } else {
                  throw new Error("OPT-0080 emitted an unexpected phase");
                }
              },
              onOpt0080CommandBufferCompleted: (
                completion: AceDitOpt0080CommandBufferCompletion,
              ) => {
                activeOpt0080Capture.recordCompletion(completion);
                this.resources.onOpt0080CommandBufferCompleted?.(completion);
              },
              onOpt0080CompletionEpochDrained: (
                timing: AceDitOpt0080CompletionEpochTiming,
              ) => {
                activeOpt0080Capture.recordEpoch(timing);
              },
            }),
        ...(schedulingCapture === undefined
          ? {}
          : {
              onPhysicalBatchDrained: (descriptor, elapsedMs) => {
                schedulingCapture.record(descriptor, elapsedMs);
              },
            }),
        ...(this.resources.evaluationReadbacks === undefined
          ? {}
          : { evaluationReadbacks: this.resources.evaluationReadbacks }),
        ...(this.resources.evaluation0VelocityReadback === undefined
          ? {}
          : {
              evaluation0VelocityReadback:
                this.resources.evaluation0VelocityReadback,
            }),
        ...(this.resources.opt0062AttentionIdentityReadback === undefined
          ? {}
          : {
              opt0062AttentionIdentityReadback:
                this.resources.opt0062AttentionIdentityReadback,
            }),
        ...(this.resources.opt0067EvaluationLimit === undefined
          ? {}
          : { opt0067EvaluationLimit: this.resources.opt0067EvaluationLimit }),
      });
      const graphFinishedAt = now?.() ?? 0;
      activeSignal.throwIfAborted();
      const evaluationLatents = this.resources.evaluationReadbacks === undefined
        ? undefined
        : await Promise.all(this.resources.evaluationReadbacks.map(
            async (readback) => await mapDetachedFinalLatent(
              readback,
              this.resources.modelProfile,
              this.resources.shape.latentElements,
              activeSignal,
            ),
          ));
      const evaluation0Velocity =
        this.resources.evaluation0VelocityReadback === undefined
          ? undefined
          : await mapDetachedFinalLatent(
              this.resources.evaluation0VelocityReadback,
              this.resources.modelProfile,
              this.resources.shape.latentElements,
              activeSignal,
            );
      const opt0062AttentionIdentity =
        this.resources.opt0062AttentionIdentityReadback === undefined
          ? undefined
          : await mapOpt0062AttentionIdentity(
              this.resources.opt0062AttentionIdentityReadback,
              activeSignal,
            );
      activeSignal.throwIfAborted();

      // The graph owner regards its sampler as graph-final, but the concrete
      // backend still owes a copy command. Preserve the production scheduler's
      // real queue-empty interval between those two submissions.
      const graphToReadbackIdleStartedAt = now?.() ?? 0;
      const idle = (this.resources.yieldQueueIdle ?? yieldQueueIdle)();
      await idle;
      const graphToReadbackIdleFinishedAt = now?.() ?? 0;
      activeSignal.throwIfAborted();
      const readbackBytes = activationBytes(
        this.resources.modelProfile,
        this.resources.shape.latentElements,
      );
      const command = encodeReadbackCopy(
        this.resources.device,
        graphResult.finalLatent,
        this.resources.readback,
        readbackBytes,
      );
      const readbackSubmittedAt = now?.() ?? 0;
      this.resources.device.queue.submit([command]);
      await this.resources.device.queue.onSubmittedWorkDone();
      const readbackDrainedAt = now?.() ?? 0;
      activeSignal.throwIfAborted();
      this.resources.onProgress?.({
        stage: "readback",
        completedCommandBuffers: graphResult.commandBuffersSubmitted + 1,
        totalCommandBuffers: graphResult.commandBuffersSubmitted + 1,
        queueDrains: graphResult.queueDrains + 1,
        cooperativeIdleMs:
          graphResult.cooperativeIdleMs + ACE_COOPERATIVE_GPU_IDLE_MILLISECONDS,
        completedEvaluations: graphResult.completedEvaluations,
      });
      const mapStartedAt = now?.() ?? 0;
      const finalLatent = await mapDetachedFinalLatent(
        this.resources.readback,
        this.resources.modelProfile,
        this.resources.shape.latentElements,
        activeSignal,
      );
      const mapFinishedAt = now?.() ?? 0;
      if (evaluationLatents !== undefined) {
        requireExactFinalEvaluationSnapshot(
          evaluationLatents,
          finalLatent,
          (this.resources.samplerScheduleProfile ??
            ACE_TURBO_EIGHT_SAMPLER_SCHEDULE_PROFILE).evaluationCount,
        );
      }
      if (profileCapture !== undefined) {
        const profile = profileCapture.finish({
          graphWallMs: nonnegativeTimingElapsed(
            graphFinishedAt,
            graphStartedAt,
          ),
          graphRequestedIdleMs: graphResult.cooperativeIdleMs,
          graphToReadbackObservedIdleMs: nonnegativeTimingElapsed(
            graphToReadbackIdleFinishedAt,
            graphToReadbackIdleStartedAt,
          ),
          readbackSubmitThroughDrainMs: nonnegativeTimingElapsed(
            readbackDrainedAt,
            readbackSubmittedAt,
          ),
          readbackMapDetachMs: nonnegativeTimingElapsed(
            mapFinishedAt,
            mapStartedAt,
          ),
          backendWallMs: nonnegativeTimingElapsed(
            mapFinishedAt,
            backendStartedAt,
          ),
        });
        try {
          this.resources.onCommandProfile?.(profile);
        } catch {
          // Capture is observational and cannot invalidate the final latent.
        }
      }
      if (opt0067Capture !== undefined) {
        if (opt0067EvaluationStartedAt === undefined) {
          throw new Error("OPT-0067 evaluation timing boundary is absent");
        }
        const profile = opt0067Capture.finish({
          graphWallMs: nonnegativeTimingElapsed(
            graphFinishedAt,
            graphStartedAt,
          ),
          evaluationWallMs: nonnegativeTimingElapsed(
            graphFinishedAt,
            opt0067EvaluationStartedAt,
          ),
          graphRequestedIdleMs: graphResult.cooperativeIdleMs,
          graphToReadbackObservedIdleMs: nonnegativeTimingElapsed(
            graphToReadbackIdleFinishedAt,
            graphToReadbackIdleStartedAt,
          ),
          readbackSubmitThroughDrainMs: nonnegativeTimingElapsed(
            readbackDrainedAt,
            readbackSubmittedAt,
          ),
          readbackMapDetachMs: nonnegativeTimingElapsed(
            mapFinishedAt,
            mapStartedAt,
          ),
          backendWallMs: nonnegativeTimingElapsed(
            mapFinishedAt,
            backendStartedAt,
          ),
        });
        this.resources.onOpt0067EvaluationProfile?.(profile);
      }
      if (opt0080Capture !== undefined) {
        if (
          opt0080PrecomputeStartedAt === undefined ||
          opt0080EvaluationStartedAt === undefined
        ) {
          throw new Error("OPT-0080 phase timing boundaries are absent");
        }
        const profile = opt0080Capture.finish({
          graphResult,
          precomputeWallMs: requireOpt0080TimingElapsed(
            opt0080EvaluationStartedAt,
            opt0080PrecomputeStartedAt,
          ),
          evaluationWallMs: requireOpt0080TimingElapsed(
            graphFinishedAt,
            opt0080EvaluationStartedAt,
          ),
          graphWallMs: requireOpt0080TimingElapsed(
            graphFinishedAt,
            opt0080PrecomputeStartedAt,
          ),
          graphToReadbackObservedIdleMs: requireOpt0080TimingElapsed(
            graphToReadbackIdleFinishedAt,
            graphToReadbackIdleStartedAt,
          ),
          readbackSubmitThroughCompletionFenceMs: requireOpt0080TimingElapsed(
            readbackDrainedAt,
            readbackSubmittedAt,
          ),
          readbackMapDetachMs: requireOpt0080TimingElapsed(
            mapFinishedAt,
            mapStartedAt,
          ),
          backendWallMs: requireOpt0080TimingElapsed(
            mapFinishedAt,
            backendStartedAt,
          ),
        });
        this.resources.onOpt0080EvaluationProfile?.(profile);
      }
      if (opt0080FullCapture !== undefined) {
        if (
          opt0080FullPhaseStartedAt.length !==
          ACE_OPT_0080_M2250_FULL_PHASE_COMMAND_BUFFER_COUNTS.length
        ) {
          throw new Error("OPT-0080 full phase timing boundaries are absent");
        }
        const evaluationWallMs = opt0080FullPhaseStartedAt.slice(1).map(
          (startedAt, evaluation) => requireOpt0080TimingElapsed(
            evaluation === 7
              ? graphFinishedAt
              : opt0080FullPhaseStartedAt[evaluation + 2]!,
            startedAt,
          ),
        ) as unknown as AceOpt0080FullEvaluationWallTuple;
        const profile = opt0080FullCapture.finish({
          graphResult,
          precomputeWallMs: requireOpt0080TimingElapsed(
            opt0080FullPhaseStartedAt[1]!,
            opt0080FullPhaseStartedAt[0]!,
          ),
          evaluationWallMs,
          graphWallMs: requireOpt0080TimingElapsed(
            graphFinishedAt,
            opt0080FullPhaseStartedAt[0]!,
          ),
          graphToReadbackObservedIdleMs: requireOpt0080TimingElapsed(
            graphToReadbackIdleFinishedAt,
            graphToReadbackIdleStartedAt,
          ),
          readbackSubmitThroughCompletionFenceMs: requireOpt0080TimingElapsed(
            readbackDrainedAt,
            readbackSubmittedAt,
          ),
          readbackMapDetachMs: requireOpt0080TimingElapsed(
            mapFinishedAt,
            mapStartedAt,
          ),
          backendWallMs: requireOpt0080TimingElapsed(
            mapFinishedAt,
            backendStartedAt,
          ),
        });
        this.resources.onOpt0080FullProfile?.(profile);
      }
      if (schedulingCapture !== undefined) {
        const schedulingProfile = schedulingCapture.finish({
          graphResult,
          graphWallMs: nonnegativeTimingElapsed(
            graphFinishedAt,
            graphStartedAt,
          ),
          graphToReadbackObservedIdleMs: nonnegativeTimingElapsed(
            graphToReadbackIdleFinishedAt,
            graphToReadbackIdleStartedAt,
          ),
          readbackSubmitThroughDrainMs: nonnegativeTimingElapsed(
            readbackDrainedAt,
            readbackSubmittedAt,
          ),
          readbackMapDetachMs: nonnegativeTimingElapsed(
            mapFinishedAt,
            mapStartedAt,
          ),
          backendWallMs: nonnegativeTimingElapsed(
            mapFinishedAt,
            backendStartedAt,
          ),
        });
        this.resources.onSchedulingProfile?.(schedulingProfile);
      }
      return Object.freeze({
        finalLatent,
        shape: Object.freeze({
          batch: this.resources.shape.batch,
          latentFrames: this.resources.shape.latentFrames,
          channels: 64 as const,
        }),
        commandBuffersSubmitted: graphResult.commandBuffersSubmitted + 1,
        queueDrains: graphResult.queueDrains + 1,
        cooperativeIdleMs:
          graphResult.cooperativeIdleMs + ACE_COOPERATIVE_GPU_IDLE_MILLISECONDS,
        completedEvaluations: graphResult.completedEvaluations,
        ...(evaluationLatents === undefined
          ? {}
          : { evaluationLatents: Object.freeze(evaluationLatents) }),
        ...(evaluation0Velocity === undefined
          ? {}
          : { evaluation0Velocity }),
        ...(graphResult.denseRouteProfile === undefined
          ? {}
          : { denseRouteProfile: graphResult.denseRouteProfile }),
        ...(graphResult.attentionRouteProfile === undefined
          ? {}
          : { attentionRouteProfile: graphResult.attentionRouteProfile }),
        ...(opt0062AttentionIdentity === undefined
          ? {}
          : { opt0062AttentionIdentity }),
      });
    } catch (error) {
      if (
        this.lifetime.signal.aborted &&
        this.lifetime.signal.reason instanceof AceDitBackendDeviceLostError
      ) {
        throw this.lifetime.signal.reason;
      }
      throw error;
    } finally {
      await this.beginDestroy(destroyedError());
    }
  }

  destroy(reason: unknown = destroyedError()): Promise<void> {
    return this.beginDestroy(reason);
  }

  private beginDestroy(reason: unknown): Promise<void> {
    if (this.destroyPromise !== undefined) return this.destroyPromise;
    this.state = "destroying";
    if (
      this.externalSignal !== undefined &&
      this.externalAbortListener !== undefined
    ) {
      this.externalSignal.removeEventListener(
        "abort",
        this.externalAbortListener,
      );
    }
    if (!this.lifetime.signal.aborted) this.lifetime.abort(reason);
    this.destroyPromise = (async () => {
      try {
        await this.resources.graph.destroy();
      } finally {
        try {
          this.resources.destroy();
        } finally {
          this.state = "destroyed";
        }
      }
    })();
    return this.destroyPromise;
  }
}

interface Opt0018CaptureFinishTiming {
  readonly graphWallMs: number;
  readonly graphRequestedIdleMs: number;
  readonly graphToReadbackObservedIdleMs: number;
  readonly readbackSubmitThroughDrainMs: number;
  readonly readbackMapDetachMs: number;
  readonly backendWallMs: number;
}

interface Opt0067CaptureFinishTiming extends Opt0018CaptureFinishTiming {
  readonly evaluationWallMs: number;
}

interface Opt0080CaptureFinishTiming {
  readonly graphResult: AceDitGraphRunResult;
  readonly precomputeWallMs: number;
  readonly evaluationWallMs: number;
  readonly graphWallMs: number;
  readonly graphToReadbackObservedIdleMs: number;
  readonly readbackSubmitThroughCompletionFenceMs: number;
  readonly readbackMapDetachMs: number;
  readonly backendWallMs: number;
}

interface Opt0080FullCaptureFinishTiming
  extends Omit<Opt0080CaptureFinishTiming, "evaluationWallMs"> {
  readonly evaluationWallMs: AceOpt0080FullEvaluationWallTuple;
}

interface Opt0034CaptureFinishTiming {
  readonly graphResult: AceDitGraphRunResult;
  readonly graphWallMs: number;
  readonly graphToReadbackObservedIdleMs: number;
  readonly readbackSubmitThroughDrainMs: number;
  readonly readbackMapDetachMs: number;
  readonly backendWallMs: number;
}

interface Opt0034SchedulingCapture {
  record(
    descriptor: AceDitPhysicalCommandBatchDescriptor,
    submitThroughDrainMs: number,
  ): void;
  finish(timing: Opt0034CaptureFinishTiming): AceOpt0034DitSchedulingProfile;
}

function createOpt0034SchedulingCapture(
  graph: AceDitGraphRunner,
  memory: AceDitGpuBackendMemoryPlan,
  physicalQuantaPerCommandBuffer: AceDitPhysicalQuantaPerCommandBuffer,
): Opt0034SchedulingCapture {
  const table = graph.physicalCommandDescriptors;
  const batchPlans = planAceDitPhysicalQuantumBatches(
    graph.commandBufferCount,
    physicalQuantaPerCommandBuffer,
  );
  const expectedBatches = table === undefined
    ? undefined
    : Object.freeze(batchPlans.map((plan) => {
        const physical = table.descriptors.slice(
          plan.firstPhysicalIndex,
          plan.lastPhysicalIndexExclusive,
        );
        return Object.freeze({
          primitiveCount: physical.reduce(
            (total, command) => total + command.primitiveCount,
            0,
          ),
          scheduledMultiplyAdds: physical.reduce(
            (total, command) => total + command.scheduledMultiplyAdds,
            0,
          ),
          commandIds: Object.freeze(physical.map(({ commandId }) => commandId)),
        });
      }));
  if (
    table === undefined ||
    expectedBatches === undefined ||
    graph.commandBufferCount !== ACE_OPT_0018_M2250_GRAPH_COMMAND_BUFFER_COUNT ||
    table.descriptors.length !== graph.commandBufferCount ||
    memory.physicalGraphQuantumCount !== graph.commandBufferCount ||
    memory.physicalQuantaPerCommandBuffer !==
      physicalQuantaPerCommandBuffer ||
    memory.graphCommandBufferCount !== batchPlans.length ||
    memory.commandBufferCount !== batchPlans.length + 1
  ) {
    throw new Error("OPT-0034 capture lost the exact M2250 batch topology");
  }
  const timings = new Float64Array(batchPlans.length);
  const seen = new Uint8Array(batchPlans.length);
  let recordCount = 0;
  return Object.freeze({
    record(
      descriptor: AceDitPhysicalCommandBatchDescriptor,
      submitThroughDrainMs: number,
    ): void {
      const plan = batchPlans[descriptor.batchIndex];
      const expected = expectedBatches[descriptor.batchIndex];
      if (
        plan === undefined ||
        expected === undefined ||
        seen[descriptor.batchIndex] !== 0 ||
        descriptor.batchCount !== batchPlans.length ||
        descriptor.firstPhysicalIndex !== plan.firstPhysicalIndex ||
        descriptor.lastPhysicalIndex !==
          plan.lastPhysicalIndexExclusive - 1 ||
        descriptor.physicalQuantumCount !== plan.physicalQuantumCount ||
        descriptor.commandIds.length !== plan.physicalQuantumCount ||
        !Number.isFinite(submitThroughDrainMs) ||
        submitThroughDrainMs < 0
      ) {
        throw new Error(`OPT-0034 rejected graph batch ${descriptor.batchIndex}`);
      }
      let commandIdsMatch =
        expected.commandIds.length === descriptor.commandIds.length;
      for (
        let index = 0;
        commandIdsMatch && index < expected.commandIds.length;
        index += 1
      ) {
        commandIdsMatch =
          descriptor.commandIds[index] === expected.commandIds[index];
      }
      if (
        expected.commandIds.length !== plan.physicalQuantumCount ||
        descriptor.primitiveCount !== expected.primitiveCount ||
        descriptor.scheduledMultiplyAdds !== expected.scheduledMultiplyAdds ||
        !commandIdsMatch
      ) {
        throw new Error(
          `OPT-0034 batch ${descriptor.batchIndex} escaped physical FIFO`,
        );
      }
      timings[descriptor.batchIndex] = submitThroughDrainMs;
      seen[descriptor.batchIndex] = 1;
      recordCount += 1;
    },
    finish(
      timing: Opt0034CaptureFinishTiming,
    ): AceOpt0034DitSchedulingProfile {
      for (const [name, value] of Object.entries(timing).filter(
        ([name]) => name !== "graphResult",
      ) as [string, number][]) {
        if (!Number.isFinite(value) || value < 0) {
          throw new Error(`OPT-0034 ${name} timing is invalid`);
        }
      }
      if (
        recordCount !== batchPlans.length ||
        seen.some((value) => value !== 1) ||
        timing.graphResult.commandBuffersSubmitted !== batchPlans.length ||
        timing.graphResult.queueDrains !== batchPlans.length ||
        timing.graphResult.cooperativeIdleMs !== batchPlans.length - 1 ||
        timing.graphResult.completedEvaluations !== DENOISING_EVALUATIONS ||
        batchPlans.reduce(
          (total, batch) => total + batch.physicalQuantumCount,
          0,
        ) !== ACE_OPT_0018_M2250_GRAPH_COMMAND_BUFFER_COUNT
      ) {
        throw new Error("OPT-0034 graph batch capture is incomplete");
      }
      const completed = Object.freeze(batchPlans.map((plan, index) => {
        const expected = expectedBatches[index]!;
        return Object.freeze({
          batchIndex: plan.batchIndex,
          firstPhysicalIndex: plan.firstPhysicalIndex,
          lastPhysicalIndex: plan.lastPhysicalIndexExclusive - 1,
          physicalQuantumCount: plan.physicalQuantumCount,
          primitiveCount: expected.primitiveCount,
          scheduledMultiplyAdds: expected.scheduledMultiplyAdds,
          submitThroughDrainMs: timings[index]!,
        });
      }));
      const graphSubmitThroughDrainMs = completed.reduce(
        (total, batch) => total + batch.submitThroughDrainMs,
        0,
      );
      const physicalPrimitiveCount = completed.reduce(
        (total, batch) => total + batch.primitiveCount,
        0,
      );
      const scheduledMultiplyAdds = completed.reduce(
        (total, batch) => total + batch.scheduledMultiplyAdds,
        0,
      );
      return Object.freeze({
        schema: "ace-dit-opt0034-command-buffer-coalescing-v1",
        physicalGraphQuantumCount:
          ACE_OPT_0018_M2250_GRAPH_COMMAND_BUFFER_COUNT,
        physicalQuantaPerCommandBuffer,
        graphCommandBufferCount: batchPlans.length,
        readbackCommandBufferCount: 1 as const,
        totalCommandBufferCount: batchPlans.length + 1,
        graphQueueDrainCount: batchPlans.length,
        totalQueueDrainCount: batchPlans.length + 1,
        graphRequestedIdleMs: batchPlans.length - 1,
        graphToReadbackRequestedIdleMs: 1 as const,
        graphSubmitThroughDrainMs,
        graphWallMs: timing.graphWallMs,
        graphToReadbackObservedIdleMs:
          timing.graphToReadbackObservedIdleMs,
        readbackSubmitThroughDrainMs: timing.readbackSubmitThroughDrainMs,
        readbackMapDetachMs: timing.readbackMapDetachMs,
        backendWallMs: timing.backendWallMs,
        maximumPhysicalQuantaPerBatch: Math.max(
          ...completed.map((batch) => batch.physicalQuantumCount),
        ),
        maximumPrimitiveCountPerBatch: Math.max(
          ...completed.map((batch) => batch.primitiveCount),
        ),
        maximumScheduledMultiplyAddsPerBatch: Math.max(
          ...completed.map((batch) => batch.scheduledMultiplyAdds),
        ),
        maximumBatchSubmitThroughDrainMs: Math.max(
          ...completed.map((batch) => batch.submitThroughDrainMs),
        ),
        descriptorTableSha256: table.sha256,
        physicalPrimitiveCount,
        scheduledMultiplyAdds,
        batches: completed,
      });
    },
  });
}

interface Opt0018ProfileCapture {
  record(
    descriptor: AceDitPhysicalCommandDescriptor,
    submitThroughDrainMs: number,
  ): void;
  finish(timing: Opt0018CaptureFinishTiming): AceOpt0018DitCommandProfile;
}

interface Opt0067EvaluationProfileCapture {
  record(
    descriptor: AceDitPhysicalCommandDescriptor,
    submitThroughDrainMs: number,
  ): void;
  finish(timing: Opt0067CaptureFinishTiming): AceOpt0067DitCommandProfile;
}

interface Opt0080EvaluationProfileCapture {
  recordPhaseStart(phase: AceDitOpt0080PhaseStart): void;
  recordCompletion(completion: AceDitOpt0080CommandBufferCompletion): void;
  recordEpoch(timing: AceDitOpt0080CompletionEpochTiming): void;
  finish(timing: Opt0080CaptureFinishTiming): AceOpt0080DitCommandProfile;
}

interface Opt0080FullProfileCapture {
  recordPhaseStart(phase: AceDitOpt0080PhaseStart): void;
  recordCompletion(completion: AceDitOpt0080CommandBufferCompletion): void;
  recordEpoch(timing: AceDitOpt0080CompletionEpochTiming): void;
  finish(timing: Opt0080FullCaptureFinishTiming): AceOpt0080FullDitCommandProfile;
}

interface MutableOpt0018Aggregate {
  commandBufferCount: number;
  primitiveCount: number;
  scheduledMultiplyAdds: number;
  submitThroughDrainMs: number;
  maximumSubmitThroughDrainMs: number;
}

function createOpt0018ProfileCapture(
  graph: AceDitGraphRunner,
  memory: AceDitGpuBackendMemoryPlan,
): Opt0018ProfileCapture {
  const table = graph.physicalCommandDescriptors;
  if (
    table === undefined ||
    graph.commandBufferCount !== ACE_OPT_0018_M2250_GRAPH_COMMAND_BUFFER_COUNT ||
    table.descriptors.length !== ACE_OPT_0018_M2250_GRAPH_COMMAND_BUFFER_COUNT ||
    memory.commandBufferCount !==
      ACE_OPT_0018_M2250_GRAPH_COMMAND_BUFFER_COUNT + 1
  ) {
    throw new Error("OPT-0018 capture lost the exact 2553+readback topology");
  }
  const timings = new Float64Array(table.descriptors.length);
  const seen = new Uint8Array(table.descriptors.length);
  let recordCount = 0;
  return Object.freeze({
    record(
      descriptor: AceDitPhysicalCommandDescriptor,
      submitThroughDrainMs: number,
    ): void {
      const index = descriptor.physicalIndex;
      if (
        table.descriptors[index] !== descriptor ||
        seen[index] !== 0 ||
        !Number.isFinite(submitThroughDrainMs) ||
        submitThroughDrainMs < 0
      ) {
        throw new Error(`OPT-0018 rejected graph drain ${index}`);
      }
      timings[index] = submitThroughDrainMs;
      seen[index] = 1;
      recordCount += 1;
    },
    finish(
      timing: Opt0018CaptureFinishTiming,
    ): AceOpt0018DitCommandProfile {
      if (
        recordCount !== table.descriptors.length ||
        seen.some((value) => value !== 1)
      ) {
        throw new Error("OPT-0018 graph drain capture is incomplete");
      }
      for (const [name, value] of Object.entries(timing) as [string, number][]) {
        if (!Number.isFinite(value) || value < 0) {
          throw new Error(`OPT-0018 ${name} timing is invalid`);
        }
      }
      if (
        timing.graphRequestedIdleMs !==
          ACE_OPT_0018_M2250_GRAPH_COMMAND_BUFFER_COUNT - 1
      ) {
        throw new Error("OPT-0018 graph idle topology changed");
      }
      const familyMutable = Object.fromEntries(
        ACE_DIT_PROFILE_FAMILIES.map((family) => [
          family,
          mutableOpt0018Aggregate(),
        ]),
      ) as Record<AceDitProfileFamily, MutableOpt0018Aggregate>;
      const buckets = Array.from(
        { length: DENOISING_EVALUATIONS + 1 },
        mutableOpt0018Aggregate,
      );
      const familyBuckets = Object.fromEntries(
        ACE_DIT_PROFILE_FAMILIES.map((family) => [
          family,
          Array.from(
            { length: DENOISING_EVALUATIONS + 1 },
            mutableOpt0018Aggregate,
          ),
        ]),
      ) as Record<
        AceDitProfileFamily,
        MutableOpt0018Aggregate[]
      >;
      let graphSubmitThroughDrainMs = 0;
      for (const descriptor of table.descriptors) {
        const elapsedMs = timings[descriptor.physicalIndex]!;
        graphSubmitThroughDrainMs += elapsedMs;
        const bucketIndex = descriptor.evaluation === null
          ? 0
          : descriptor.evaluation + 1;
        if (
          bucketIndex < 0 ||
          bucketIndex > DENOISING_EVALUATIONS ||
          (descriptor.evaluation !== null &&
            (descriptor.evaluation < 0 ||
              descriptor.evaluation >= DENOISING_EVALUATIONS))
        ) {
          throw new Error(
            `OPT-0018 command ${descriptor.physicalIndex} has an invalid bucket`,
          );
        }
        addOpt0018Aggregate(
          familyMutable[descriptor.family],
          descriptor,
          elapsedMs,
        );
        addOpt0018Aggregate(buckets[bucketIndex]!, descriptor, elapsedMs);
        addOpt0018Aggregate(
          familyBuckets[descriptor.family][bucketIndex]!,
          descriptor,
          elapsedMs,
        );
      }
      requireOpt0018AggregateReconciliation(
        table.descriptors.length,
        graphSubmitThroughDrainMs,
        Object.values(familyMutable),
        "family",
      );
      requireOpt0018AggregateReconciliation(
        table.descriptors.length,
        graphSubmitThroughDrainMs,
        buckets,
        "precompute/evaluation",
      );
      for (const family of ACE_DIT_PROFILE_FAMILIES) {
        requireOpt0018AggregateReconciliation(
          familyMutable[family].commandBufferCount,
          familyMutable[family].submitThroughDrainMs,
          familyBuckets[family],
          `${family} bucket`,
        );
      }
      const families = Object.fromEntries(
        ACE_DIT_PROFILE_FAMILIES.map((family) => [
          family,
          freezeOpt0018Aggregate(familyMutable[family]),
        ]),
      ) as Record<AceDitProfileFamily, AceOpt0018DitProfileAggregate>;
      const familyByBucket = Object.fromEntries(
        ACE_DIT_PROFILE_FAMILIES.map((family) => [
          family,
          Object.freeze(familyBuckets[family].map(freezeOpt0018Aggregate)),
        ]),
      ) as Record<
        AceDitProfileFamily,
        readonly AceOpt0018DitProfileAggregate[]
      >;
      const graphResidualMs = timing.graphWallMs -
        graphSubmitThroughDrainMs - timing.graphRequestedIdleMs;
      const backendResidualMs = timing.backendWallMs - timing.graphWallMs -
        timing.graphToReadbackObservedIdleMs -
        timing.readbackSubmitThroughDrainMs - timing.readbackMapDetachMs;
      return Object.freeze({
        schema: "ace-dit-m2250-command-profile-v1",
        descriptorTable: table,
        graphCommandBufferCount:
          ACE_OPT_0018_M2250_GRAPH_COMMAND_BUFFER_COUNT,
        readbackCommandBufferCount: 1,
        totalCommandBufferCount: 2_554,
        timings: Object.freeze([...timings]),
        graphSubmitThroughDrainMs,
        graphWallMs: timing.graphWallMs,
        graphRequestedIdleMs: timing.graphRequestedIdleMs,
        graphResidualMs,
        graphToReadbackRequestedIdleMs:
          ACE_COOPERATIVE_GPU_IDLE_MILLISECONDS,
        graphToReadbackObservedIdleMs:
          timing.graphToReadbackObservedIdleMs,
        readbackSubmitThroughDrainMs:
          timing.readbackSubmitThroughDrainMs,
        readbackMapDetachMs: timing.readbackMapDetachMs,
        backendWallMs: timing.backendWallMs,
        backendResidualMs,
        families: Object.freeze(families),
        precompute: freezeOpt0018Aggregate(buckets[0]!),
        evaluations: Object.freeze(
          buckets.slice(1).map(freezeOpt0018Aggregate),
        ),
        familyByBucket: Object.freeze(familyByBucket),
        slowest: Object.freeze(table.descriptors.map((descriptor) =>
          Object.freeze({
            physicalIndex: descriptor.physicalIndex,
            family: descriptor.family,
            evaluation: descriptor.evaluation,
            layer: descriptor.layer,
            submitThroughDrainMs: timings[descriptor.physicalIndex]!,
          })
        ).sort((left, right) =>
          right.submitThroughDrainMs - left.submitThroughDrainMs ||
          left.physicalIndex - right.physicalIndex
        ).slice(0, OPT_0018_SLOWEST_COMMAND_COUNT)),
        reconciliationToleranceMs: OPT_0018_RECONCILIATION_TOLERANCE_MS,
        timingStorageBytes: timings.byteLength + seen.byteLength,
      });
    },
  });
}

function createOpt0067EvaluationProfileCapture(
  graph: AceDitGraphRunner,
  memory: AceDitGpuBackendMemoryPlan,
): Opt0067EvaluationProfileCapture {
  const table = graph.physicalCommandDescriptors;
  const graphCount = ACE_OPT_0067_M2250_EVALUATION0_GRAPH_COMMAND_BUFFER_COUNT;
  const precomputeCount = 25;
  const evaluationCount = 316;
  if (
    table === undefined ||
    graph.commandBufferCount !== ACE_OPT_0018_M2250_GRAPH_COMMAND_BUFFER_COUNT ||
    table.descriptors.length !== ACE_OPT_0018_M2250_GRAPH_COMMAND_BUFFER_COUNT ||
    memory.commandBufferCount !==
      ACE_OPT_0018_M2250_GRAPH_COMMAND_BUFFER_COUNT + 1 ||
    graphCount !== precomputeCount + evaluationCount ||
    table.descriptors.slice(0, precomputeCount).some((descriptor) =>
      descriptor.evaluation !== null
    ) ||
    table.descriptors.slice(precomputeCount, graphCount).some((descriptor) =>
      descriptor.evaluation !== 0
    ) ||
    table.descriptors[graphCount]?.evaluation !== 1
  ) {
    throw new Error("OPT-0067 capture lost the exact precompute/evaluation-0 prefix");
  }
  const descriptors = table.descriptors.slice(0, graphCount);
  const timings = new Float64Array(graphCount);
  const seen = new Uint8Array(graphCount);
  let recordCount = 0;
  return Object.freeze({
    record(
      descriptor: AceDitPhysicalCommandDescriptor,
      submitThroughDrainMs: number,
    ): void {
      const index = descriptor.physicalIndex;
      if (
        index < 0 || index >= graphCount ||
        descriptors[index] !== descriptor || seen[index] !== 0 ||
        !Number.isFinite(submitThroughDrainMs) || submitThroughDrainMs < 0
      ) {
        throw new Error(`OPT-0067 rejected evaluation-prefix drain ${index}`);
      }
      timings[index] = submitThroughDrainMs;
      seen[index] = 1;
      recordCount += 1;
    },
    finish(timing: Opt0067CaptureFinishTiming): AceOpt0067DitCommandProfile {
      if (
        recordCount !== graphCount || seen.some((value) => value !== 1) ||
        timing.graphRequestedIdleMs !== graphCount - 1
      ) {
        throw new Error("OPT-0067 evaluation-prefix capture is incomplete");
      }
      for (const [name, value] of Object.entries(timing) as [string, number][]) {
        if (!Number.isFinite(value) || value < 0) {
          throw new Error(`OPT-0067 ${name} timing is invalid`);
        }
      }
      const familyMutable = Object.fromEntries(
        ACE_DIT_PROFILE_FAMILIES.map((family) => [
          family,
          mutableOpt0018Aggregate(),
        ]),
      ) as Record<AceDitProfileFamily, MutableOpt0018Aggregate>;
      const buckets = [mutableOpt0018Aggregate(), mutableOpt0018Aggregate()];
      const familyBuckets = Object.fromEntries(
        ACE_DIT_PROFILE_FAMILIES.map((family) => [
          family,
          [mutableOpt0018Aggregate(), mutableOpt0018Aggregate()],
        ]),
      ) as Record<
        AceDitProfileFamily,
        [MutableOpt0018Aggregate, MutableOpt0018Aggregate]
      >;
      let graphSubmitThroughDrainMs = 0;
      for (const descriptor of descriptors) {
        const elapsedMs = timings[descriptor.physicalIndex]!;
        const bucketIndex = descriptor.evaluation === null ? 0 : 1;
        graphSubmitThroughDrainMs += elapsedMs;
        addOpt0018Aggregate(familyMutable[descriptor.family], descriptor, elapsedMs);
        addOpt0018Aggregate(buckets[bucketIndex]!, descriptor, elapsedMs);
        addOpt0018Aggregate(
          familyBuckets[descriptor.family][bucketIndex],
          descriptor,
          elapsedMs,
        );
      }
      requireOpt0018AggregateReconciliation(
        graphCount,
        graphSubmitThroughDrainMs,
        Object.values(familyMutable),
        "OPT-0067 family",
      );
      requireOpt0018AggregateReconciliation(
        graphCount,
        graphSubmitThroughDrainMs,
        buckets,
        "OPT-0067 precompute/evaluation",
      );
      for (const family of ACE_DIT_PROFILE_FAMILIES) {
        requireOpt0018AggregateReconciliation(
          familyMutable[family].commandBufferCount,
          familyMutable[family].submitThroughDrainMs,
          familyBuckets[family],
          `OPT-0067 ${family} bucket`,
        );
      }
      const families = Object.fromEntries(
        ACE_DIT_PROFILE_FAMILIES.map((family) => [
          family,
          freezeOpt0018Aggregate(familyMutable[family]),
        ]),
      ) as Record<AceDitProfileFamily, AceOpt0018DitProfileAggregate>;
      const frozenFamilyBuckets = Object.fromEntries(
        ACE_DIT_PROFILE_FAMILIES.map((family) => [
          family,
          Object.freeze([
            freezeOpt0018Aggregate(familyBuckets[family][0]),
            freezeOpt0018Aggregate(familyBuckets[family][1]),
          ]) as readonly [
            AceOpt0018DitProfileAggregate,
            AceOpt0018DitProfileAggregate,
          ],
        ]),
      ) as Record<
        AceDitProfileFamily,
        readonly [AceOpt0018DitProfileAggregate, AceOpt0018DitProfileAggregate]
      >;
      const evaluationSubmitThroughDrainMs = buckets[1]!.submitThroughDrainMs;
      const graphResidualMs = timing.graphWallMs -
        graphSubmitThroughDrainMs - (graphCount - 1);
      const evaluationResidualMs = timing.evaluationWallMs -
        evaluationSubmitThroughDrainMs - evaluationCount;
      return Object.freeze({
        schema: "ace-dit-opt0067-evaluation0-command-profile-v1",
        descriptorTable: table,
        graphCommandBufferCount: graphCount as 341,
        readbackCommandBufferCount: 1 as const,
        totalCommandBufferCount: (graphCount + 1) as 342,
        graphQueueDrainCount: graphCount as 341,
        totalQueueDrainCount: (graphCount + 1) as 342,
        evaluationCommandBufferCount: evaluationCount as 316,
        timings: Object.freeze([...timings]),
        graphSubmitThroughDrainMs,
        graphWallMs: timing.graphWallMs,
        graphRequestedIdleMs: (graphCount - 1) as 340,
        graphResidualMs,
        evaluationWallMs: timing.evaluationWallMs,
        evaluationRequestedIdleMs: evaluationCount as 316,
        evaluationResidualMs,
        graphToReadbackRequestedIdleMs:
          ACE_COOPERATIVE_GPU_IDLE_MILLISECONDS,
        graphToReadbackObservedIdleMs:
          timing.graphToReadbackObservedIdleMs,
        readbackSubmitThroughDrainMs: timing.readbackSubmitThroughDrainMs,
        readbackMapDetachMs: timing.readbackMapDetachMs,
        backendWallMs: timing.backendWallMs,
        backendResidualMs: timing.backendWallMs - timing.graphWallMs -
          timing.graphToReadbackObservedIdleMs -
          timing.readbackSubmitThroughDrainMs - timing.readbackMapDetachMs,
        families: Object.freeze(families),
        precompute: freezeOpt0018Aggregate(buckets[0]!),
        evaluation: freezeOpt0018Aggregate(buckets[1]!),
        familyByBucket: Object.freeze(frozenFamilyBuckets),
        slowest: Object.freeze(descriptors.map((descriptor) => Object.freeze({
          physicalIndex: descriptor.physicalIndex,
          family: descriptor.family,
          evaluation: descriptor.evaluation,
          layer: descriptor.layer,
          submitThroughDrainMs: timings[descriptor.physicalIndex]!,
        })).sort((left, right) =>
          right.submitThroughDrainMs - left.submitThroughDrainMs ||
          left.physicalIndex - right.physicalIndex
        ).slice(0, OPT_0018_SLOWEST_COMMAND_COUNT)),
        reconciliationToleranceMs: OPT_0018_RECONCILIATION_TOLERANCE_MS,
        timingStorageBytes: timings.byteLength + seen.byteLength,
      });
    },
  });
}

function createOpt0080EvaluationProfileCapture(
  graph: AceDitGraphRunner,
  memory: AceDitGpuBackendMemoryPlan,
  schedulingProfile: AceDitOpt0080SchedulingProfile,
): Opt0080EvaluationProfileCapture {
  const table = graph.physicalCommandDescriptors;
  const graphCount = ACE_OPT_0067_M2250_EVALUATION0_GRAPH_COMMAND_BUFFER_COUNT;
  const precomputeCount = 25;
  const evaluationCount = 316;
  if (
    table === undefined ||
    graph.commandBufferCount !== ACE_OPT_0018_M2250_GRAPH_COMMAND_BUFFER_COUNT ||
    table.descriptors.length !== ACE_OPT_0018_M2250_GRAPH_COMMAND_BUFFER_COUNT ||
    table.memberCount !== 6_833 ||
    memory.commandBufferCount !==
      ACE_OPT_0018_M2250_GRAPH_COMMAND_BUFFER_COUNT + 1 ||
    graphCount !== precomputeCount + evaluationCount ||
    table.descriptors.slice(0, precomputeCount).some((descriptor) =>
      descriptor.evaluation !== null
    ) ||
    table.descriptors.slice(precomputeCount, graphCount).some((descriptor) =>
      descriptor.evaluation !== 0
    ) ||
    table.descriptors[graphCount]?.evaluation !== 1
  ) {
    throw new Error("OPT-0080 capture lost the exact evaluation-0 prefix");
  }
  const descriptors = table.descriptors.slice(0, graphCount);
  const expectedEpochs = planOpt0080ExpectedCompletionEpochs(
    schedulingProfile,
  );
  const completionTimings = new Float64Array(graphCount);
  const completionSeen = new Uint8Array(graphCount);
  const epochs: AceOpt0080CompletionEpochProfile[] = [];
  const phases: AceDitOpt0080PhaseStart[] = [];

  return Object.freeze({
    recordPhaseStart(phase: AceDitOpt0080PhaseStart): void {
      const expectedPhaseIndex = phases.length;
      const expectedFirst = expectedPhaseIndex === 0 ? 0 : precomputeCount;
      const expectedCount = expectedPhaseIndex === 0
        ? precomputeCount
        : evaluationCount;
      if (
        expectedPhaseIndex > 1 ||
        phase.phaseIndex !== expectedPhaseIndex ||
        phase.firstCommandBufferIndex !== expectedFirst ||
        phase.commandBufferCount !== expectedCount
      ) {
        throw new Error("OPT-0080 phase boundary changed");
      }
      phases.push(Object.freeze({ ...phase }));
    },
    recordCompletion(completion: AceDitOpt0080CommandBufferCompletion): void {
      const index = completion.commandBufferIndex;
      const expectedEpoch = expectedEpochs[completion.completionEpochIndex];
      if (
        index < 0 || index >= graphCount || expectedEpoch === undefined ||
        index < expectedEpoch.firstCommandBufferIndex ||
        index > expectedEpoch.lastCommandBufferIndex ||
        descriptors[index] !== completion.descriptor ||
        completionSeen[index] !== 0 ||
        completion.completionEpochIndex !==
          expectedEpoch.completionEpochIndex ||
        completion.trueQueueDrain !==
          (index === expectedEpoch.lastCommandBufferIndex) ||
        completion.schedulingProgress.completedCommandBuffers !== index + 1 ||
        completion.graphProgress.completedCommandBuffers !== index + 1 ||
        !Number.isFinite(completion.submitThroughCompletionFenceMs) ||
        completion.submitThroughCompletionFenceMs < 0
      ) {
        throw new Error(`OPT-0080 rejected completion fence ${index}`);
      }
      completionTimings[index] =
        completion.submitThroughCompletionFenceMs;
      completionSeen[index] = 1;
    },
    recordEpoch(timing: AceDitOpt0080CompletionEpochTiming): void {
      const expected = expectedEpochs[epochs.length];
      if (
        expected === undefined ||
        timing.completionEpochIndex !== expected.completionEpochIndex ||
        timing.phaseIndex !== expected.phaseIndex ||
        timing.firstCommandBufferIndex !== expected.firstCommandBufferIndex ||
        timing.lastCommandBufferIndex !== expected.lastCommandBufferIndex ||
        timing.commandBufferCount !== expected.commandBufferCount ||
        !Number.isFinite(timing.submitThroughTrueDrainMs) ||
        timing.submitThroughTrueDrainMs < 0
      ) {
        throw new Error(`OPT-0080 rejected completion epoch ${epochs.length}`);
      }
      epochs.push(Object.freeze({ ...timing }));
    },
    finish(timing: Opt0080CaptureFinishTiming): AceOpt0080DitCommandProfile {
      const candidate = schedulingProfile === "opt-0080-depth2-epoch4";
      const expectedGraphDrains: 86 | 341 = candidate ? 86 : 341;
      const expectedGraphIdleTurns: 85 | 340 = candidate ? 85 : 340;
      const expectedTotalIdleTurns: 86 | 341 = candidate ? 86 : 341;
      const expectedTotalDrains: 87 | 342 = candidate ? 87 : 342;
      const expectedMaximum: 1 | 2 = candidate ? 2 : 1;
      const diagnostics = timing.graphResult.opt0080Scheduling;
      if (
        phases.length !== 2 ||
        completionSeen.some((value) => value !== 1) ||
        epochs.length !== expectedEpochs.length ||
        diagnostics === undefined || diagnostics.profile !== schedulingProfile ||
        timing.graphResult.commandBuffersSubmitted !== graphCount ||
        timing.graphResult.queueDrains !== expectedGraphDrains ||
        timing.graphResult.cooperativeIdleMs !== expectedGraphIdleTurns ||
        diagnostics.completionFenceRequestedCount !== graphCount ||
        diagnostics.completionFenceSettledCount !== graphCount ||
        diagnostics.completionFenceRejectedCount !== 0 ||
        diagnostics.trueQueueDrainCount !== expectedGraphDrains ||
        diagnostics.completionEpochCount !== expectedGraphDrains ||
        diagnostics.cooperativeIdleTurns !== expectedGraphIdleTurns ||
        diagnostics.requestedCooperativeIdleMs !== expectedGraphIdleTurns ||
        diagnostics.maximumOutstandingCommandBuffers !== expectedMaximum ||
        diagnostics.maximumPendingDescriptorCount !== expectedMaximum
      ) {
        throw new Error("OPT-0080 scheduling topology did not reconcile");
      }
      const numericTimings = [
        timing.precomputeWallMs,
        timing.evaluationWallMs,
        timing.graphWallMs,
        timing.graphToReadbackObservedIdleMs,
        timing.readbackSubmitThroughCompletionFenceMs,
        timing.readbackMapDetachMs,
        timing.backendWallMs,
      ];
      if (numericTimings.some((value) => !Number.isFinite(value) || value < 0)) {
        throw new Error("OPT-0080 authoritative wall timing is invalid");
      }
      const epochWallSumMs = epochs.reduce(
        (sum, epoch) => sum + epoch.submitThroughTrueDrainMs,
        0,
      );
      const wallToleranceMs = OPT_0018_RECONCILIATION_TOLERANCE_MS;
      if (
        Math.abs(
          timing.precomputeWallMs + timing.evaluationWallMs -
            timing.graphWallMs,
        ) > wallToleranceMs ||
        epochWallSumMs - timing.graphWallMs > wallToleranceMs
      ) {
        throw new Error("OPT-0080 disjoint wall timing did not reconcile");
      }
      return Object.freeze({
        schema: "ace-dit-opt0080-evaluation0-command-profile-v1",
        schedulingProfile,
        descriptorTable: table,
        precomputeWallMs: timing.precomputeWallMs,
        evaluationWallMs: timing.evaluationWallMs,
        graphWallMs: timing.graphWallMs,
        graphToReadbackObservedIdleMs:
          timing.graphToReadbackObservedIdleMs,
        readbackSubmitThroughCompletionFenceMs:
          timing.readbackSubmitThroughCompletionFenceMs,
        readbackMapDetachMs: timing.readbackMapDetachMs,
        backendWallMs: timing.backendWallMs,
        topology: Object.freeze({
          schedulingProfile,
          descriptorTableMemberCount: table.memberCount as 6_833,
          graphCommandBufferCount: graphCount as 341,
          readbackCommandBufferCount: 1 as const,
          totalCommandBufferCount: (graphCount + 1) as 342,
          commandBuffersSubmitted:
            (timing.graphResult.commandBuffersSubmitted + 1) as 342,
          completionFenceRequestedCount:
            (diagnostics.completionFenceRequestedCount + 1) as 342,
          completionFenceSettledCount:
            (diagnostics.completionFenceSettledCount + 1) as 342,
          completionFenceRejectedCount: 0 as const,
          graphTrueQueueDrainCount: expectedGraphDrains,
          totalTrueQueueDrainCount: expectedTotalDrains,
          graphCompletionEpochCount: expectedGraphDrains,
          graphCooperativeIdleTurns: expectedGraphIdleTurns,
          totalCooperativeIdleTurns: expectedTotalIdleTurns,
          graphRequestedCooperativeIdleMs: expectedGraphIdleTurns,
          totalRequestedCooperativeIdleMs: expectedTotalIdleTurns,
          maximumOutstandingCommandBuffers: expectedMaximum,
          maximumPendingDescriptorCount: expectedMaximum,
          pendingDescriptorCountAfterCleanup: 0 as const,
          graphCompletionEpochs: Object.freeze([...epochs]),
          submitThroughCompletionFenceMs:
            Object.freeze([...completionTimings]),
          graphToReadbackRequestedIdleMs:
            ACE_COOPERATIVE_GPU_IDLE_MILLISECONDS as 1,
          readbackSubmitThroughCompletionFenceMs:
            timing.readbackSubmitThroughCompletionFenceMs,
        }),
      });
    },
  });
}

function createOpt0080FullProfileCapture(
  graph: AceDitGraphRunner,
  memory: AceDitGpuBackendMemoryPlan,
  schedulingProfile: AceDitOpt0080SchedulingProfile,
): Opt0080FullProfileCapture {
  const table = graph.physicalCommandDescriptors;
  const graphCount = ACE_OPT_0018_M2250_GRAPH_COMMAND_BUFFER_COUNT;
  const phaseCounts = ACE_OPT_0080_M2250_FULL_PHASE_COMMAND_BUFFER_COUNTS;
  if (
    table === undefined || graph.commandBufferCount !== graphCount ||
    table.descriptors.length !== graphCount || table.memberCount !== 6_833 ||
    memory.commandBufferCount !== graphCount + 1
  ) {
    throw new Error("OPT-0080 full capture lost the exact M2250 graph");
  }
  let descriptorFirst = 0;
  for (let phaseIndex = 0; phaseIndex < phaseCounts.length; phaseIndex += 1) {
    const descriptorLast = descriptorFirst + phaseCounts[phaseIndex]!;
    const expectedEvaluation = phaseIndex === 0 ? null : phaseIndex - 1;
    if (table.descriptors.slice(descriptorFirst, descriptorLast).some(
      (descriptor) => descriptor.evaluation !== expectedEvaluation,
    )) {
      throw new Error(`OPT-0080 full descriptor phase ${phaseIndex} changed`);
    }
    descriptorFirst = descriptorLast;
  }
  if (descriptorFirst !== graphCount) {
    throw new Error("OPT-0080 full descriptor phases are incomplete");
  }

  const expectedEpochs = planOpt0080ExpectedCompletionEpochs(
    schedulingProfile,
    phaseCounts,
  );
  const completionTimings = new Float64Array(graphCount);
  const completionSeen = new Uint8Array(graphCount);
  const epochs: AceOpt0080CompletionEpochProfile[] = [];
  const phases: AceDitOpt0080PhaseStart[] = [];

  return Object.freeze({
    recordPhaseStart(phase: AceDitOpt0080PhaseStart): void {
      const expectedPhaseIndex = phases.length;
      const expectedFirst = phaseCounts.slice(0, expectedPhaseIndex).reduce(
        (sum, count) => sum + count,
        0,
      );
      const expectedCount = phaseCounts[expectedPhaseIndex];
      if (
        expectedCount === undefined || phase.phaseIndex !== expectedPhaseIndex ||
        phase.firstCommandBufferIndex !== expectedFirst ||
        phase.commandBufferCount !== expectedCount
      ) {
        throw new Error("OPT-0080 full phase boundary changed");
      }
      phases.push(Object.freeze({ ...phase }));
    },
    recordCompletion(completion: AceDitOpt0080CommandBufferCompletion): void {
      const index = completion.commandBufferIndex;
      const expectedEpoch = expectedEpochs[completion.completionEpochIndex];
      if (
        index < 0 || index >= graphCount || expectedEpoch === undefined ||
        index < expectedEpoch.firstCommandBufferIndex ||
        index > expectedEpoch.lastCommandBufferIndex ||
        table.descriptors[index] !== completion.descriptor ||
        completionSeen[index] !== 0 ||
        completion.completionEpochIndex !== expectedEpoch.completionEpochIndex ||
        completion.trueQueueDrain !==
          (index === expectedEpoch.lastCommandBufferIndex) ||
        completion.schedulingProgress.completedCommandBuffers !== index + 1 ||
        completion.graphProgress.completedCommandBuffers !== index + 1 ||
        !Number.isFinite(completion.submitThroughCompletionFenceMs) ||
        completion.submitThroughCompletionFenceMs < 0
      ) {
        throw new Error(`OPT-0080 full rejected completion fence ${index}`);
      }
      completionTimings[index] = completion.submitThroughCompletionFenceMs;
      completionSeen[index] = 1;
    },
    recordEpoch(timing: AceDitOpt0080CompletionEpochTiming): void {
      const expected = expectedEpochs[epochs.length];
      if (
        expected === undefined ||
        timing.completionEpochIndex !== expected.completionEpochIndex ||
        timing.phaseIndex !== expected.phaseIndex ||
        timing.firstCommandBufferIndex !== expected.firstCommandBufferIndex ||
        timing.lastCommandBufferIndex !== expected.lastCommandBufferIndex ||
        timing.commandBufferCount !== expected.commandBufferCount ||
        !Number.isFinite(timing.submitThroughTrueDrainMs) ||
        timing.submitThroughTrueDrainMs < 0
      ) {
        throw new Error(`OPT-0080 full rejected completion epoch ${epochs.length}`);
      }
      epochs.push(Object.freeze({ ...timing }));
    },
    finish(timing: Opt0080FullCaptureFinishTiming): AceOpt0080FullDitCommandProfile {
      const candidate = schedulingProfile === "opt-0080-depth2-epoch4";
      const expectedGraphDrains: 639 | 2_553 = candidate ? 639 : 2_553;
      const expectedGraphIdleTurns: 638 | 2_552 = candidate ? 638 : 2_552;
      const expectedTotalIdleTurns: 639 | 2_553 = candidate ? 639 : 2_553;
      const expectedTotalDrains: 640 | 2_554 = candidate ? 640 : 2_554;
      const expectedMaximum: 1 | 2 = candidate ? 2 : 1;
      const diagnostics = timing.graphResult.opt0080Scheduling;
      if (
        phases.length !== phaseCounts.length ||
        completionSeen.some((value) => value !== 1) ||
        epochs.length !== expectedEpochs.length || diagnostics === undefined ||
        diagnostics.profile !== schedulingProfile ||
        timing.graphResult.commandBuffersSubmitted !== graphCount ||
        timing.graphResult.queueDrains !== expectedGraphDrains ||
        timing.graphResult.cooperativeIdleMs !== expectedGraphIdleTurns ||
        timing.graphResult.completedEvaluations !== 8 ||
        diagnostics.completionFenceRequestedCount !== graphCount ||
        diagnostics.completionFenceSettledCount !== graphCount ||
        diagnostics.completionFenceRejectedCount !== 0 ||
        diagnostics.trueQueueDrainCount !== expectedGraphDrains ||
        diagnostics.completionEpochCount !== expectedGraphDrains ||
        diagnostics.cooperativeIdleTurns !== expectedGraphIdleTurns ||
        diagnostics.requestedCooperativeIdleMs !== expectedGraphIdleTurns ||
        diagnostics.maximumOutstandingCommandBuffers !== expectedMaximum ||
        diagnostics.maximumPendingDescriptorCount !== expectedMaximum
      ) {
        throw new Error("OPT-0080 full scheduling topology did not reconcile");
      }
      const numericTimings = [
        timing.precomputeWallMs,
        ...timing.evaluationWallMs,
        timing.graphWallMs,
        timing.graphToReadbackObservedIdleMs,
        timing.readbackSubmitThroughCompletionFenceMs,
        timing.readbackMapDetachMs,
        timing.backendWallMs,
      ];
      if (numericTimings.some((value) => !Number.isFinite(value) || value < 0)) {
        throw new Error("OPT-0080 full authoritative wall timing is invalid");
      }
      const phaseWallSumMs = timing.precomputeWallMs +
        timing.evaluationWallMs.reduce((sum, wall) => sum + wall, 0);
      const epochWallSumMs = epochs.reduce(
        (sum, epoch) => sum + epoch.submitThroughTrueDrainMs,
        0,
      );
      const wallToleranceMs = OPT_0018_RECONCILIATION_TOLERANCE_MS;
      if (
        Math.abs(phaseWallSumMs - timing.graphWallMs) > wallToleranceMs ||
        epochWallSumMs - timing.graphWallMs > wallToleranceMs
      ) {
        throw new Error("OPT-0080 full disjoint wall timing did not reconcile");
      }
      return Object.freeze({
        schema: "ace-dit-opt0080-full-command-profile-v1",
        schedulingProfile,
        descriptorTable: table,
        precomputeWallMs: timing.precomputeWallMs,
        evaluationWallMs: Object.freeze([...timing.evaluationWallMs]) as
          unknown as AceOpt0080FullEvaluationWallTuple,
        graphWallMs: timing.graphWallMs,
        graphToReadbackObservedIdleMs: timing.graphToReadbackObservedIdleMs,
        readbackSubmitThroughCompletionFenceMs:
          timing.readbackSubmitThroughCompletionFenceMs,
        readbackMapDetachMs: timing.readbackMapDetachMs,
        backendWallMs: timing.backendWallMs,
        topology: Object.freeze({
          schedulingProfile,
          descriptorTableMemberCount: table.memberCount as 6_833,
          graphCommandBufferCount: graphCount as 2_553,
          readbackCommandBufferCount: 1 as const,
          totalCommandBufferCount: (graphCount + 1) as 2_554,
          commandBuffersSubmitted:
            (timing.graphResult.commandBuffersSubmitted + 1) as 2_554,
          completionFenceRequestedCount:
            (diagnostics.completionFenceRequestedCount + 1) as 2_554,
          completionFenceSettledCount:
            (diagnostics.completionFenceSettledCount + 1) as 2_554,
          completionFenceRejectedCount: 0 as const,
          graphTrueQueueDrainCount: expectedGraphDrains,
          totalTrueQueueDrainCount: expectedTotalDrains,
          graphCompletionEpochCount: expectedGraphDrains,
          graphCooperativeIdleTurns: expectedGraphIdleTurns,
          totalCooperativeIdleTurns: expectedTotalIdleTurns,
          graphRequestedCooperativeIdleMs: expectedGraphIdleTurns,
          totalRequestedCooperativeIdleMs: expectedTotalIdleTurns,
          maximumOutstandingCommandBuffers: expectedMaximum,
          maximumPendingDescriptorCount: expectedMaximum,
          pendingDescriptorCountAfterCleanup: 0 as const,
          graphCompletionEpochs: Object.freeze([...epochs]),
          submitThroughCompletionFenceMs:
            Object.freeze([...completionTimings]),
          graphToReadbackRequestedIdleMs:
            ACE_COOPERATIVE_GPU_IDLE_MILLISECONDS as 1,
          readbackSubmitThroughCompletionFenceMs:
            timing.readbackSubmitThroughCompletionFenceMs,
        }),
      });
    },
  });
}

function planOpt0080ExpectedCompletionEpochs(
  schedulingProfile: AceDitOpt0080SchedulingProfile,
  phases: readonly number[] = [25, 316],
): readonly Omit<AceOpt0080CompletionEpochProfile, "submitThroughTrueDrainMs">[] {
  const maximumCommandsPerEpoch = schedulingProfile === "depth1-epoch1" ? 1 : 4;
  const epochs: Array<Omit<
    AceOpt0080CompletionEpochProfile,
    "submitThroughTrueDrainMs"
  >> = [];
  let firstCommandBufferIndex = 0;
  for (let phaseIndex = 0; phaseIndex < phases.length; phaseIndex += 1) {
    const phaseEnd = firstCommandBufferIndex + phases[phaseIndex]!;
    while (firstCommandBufferIndex < phaseEnd) {
      const commandBufferCount = Math.min(
        maximumCommandsPerEpoch,
        phaseEnd - firstCommandBufferIndex,
      );
      epochs.push(Object.freeze({
        completionEpochIndex: epochs.length,
        phaseIndex,
        firstCommandBufferIndex,
        lastCommandBufferIndex:
          firstCommandBufferIndex + commandBufferCount - 1,
        commandBufferCount,
      }));
      firstCommandBufferIndex += commandBufferCount;
    }
  }
  return Object.freeze(epochs);
}

function mutableOpt0018Aggregate(): MutableOpt0018Aggregate {
  return {
    commandBufferCount: 0,
    primitiveCount: 0,
    scheduledMultiplyAdds: 0,
    submitThroughDrainMs: 0,
    maximumSubmitThroughDrainMs: 0,
  };
}

function addOpt0018Aggregate(
  aggregate: MutableOpt0018Aggregate,
  descriptor: AceDitPhysicalCommandDescriptor,
  elapsedMs: number,
): void {
  aggregate.commandBufferCount += 1;
  aggregate.primitiveCount += descriptor.primitiveCount;
  aggregate.scheduledMultiplyAdds += descriptor.scheduledMultiplyAdds;
  aggregate.submitThroughDrainMs += elapsedMs;
  aggregate.maximumSubmitThroughDrainMs = Math.max(
    aggregate.maximumSubmitThroughDrainMs,
    elapsedMs,
  );
}

function freezeOpt0018Aggregate(
  aggregate: MutableOpt0018Aggregate,
): AceOpt0018DitProfileAggregate {
  return Object.freeze({ ...aggregate });
}

function requireOpt0018AggregateReconciliation(
  expectedCount: number,
  expectedMs: number,
  aggregates: readonly MutableOpt0018Aggregate[],
  label: string,
): void {
  const count = aggregates.reduce(
    (total, aggregate) => total + aggregate.commandBufferCount,
    0,
  );
  const elapsedMs = aggregates.reduce(
    (total, aggregate) => total + aggregate.submitThroughDrainMs,
    0,
  );
  if (
    count !== expectedCount ||
    Math.abs(elapsedMs - expectedMs) >
      OPT_0018_RECONCILIATION_TOLERANCE_MS
  ) {
    throw new Error(`OPT-0018 ${label} aggregates do not reconcile`);
  }
}

function nonnegativeTimingElapsed(finishedAt: number, startedAt: number): number {
  const elapsed = finishedAt - startedAt;
  return Number.isFinite(elapsed) && elapsed > 0 ? elapsed : 0;
}

function requireOpt0080TimingElapsed(
  finishedAt: number,
  startedAt: number,
): number {
  const elapsed = finishedAt - startedAt;
  if (!Number.isFinite(elapsed) || elapsed < 0) {
    throw new RangeError("OPT-0080 timing clock must be finite and monotonic");
  }
  return elapsed;
}

const OPT_0081_REPRESENTATIVE_CONTROL_ARENA_BYTES = 674_815_488 as const;
const OPT_0081_REPRESENTATIVE_CANDIDATE_ARENA_BYTES = 601_087_488 as const;
const OPT_0081_REPRESENTATIVE_ARENA_SAVING_BYTES = 73_728_000 as const;
const OPT_0081_REPRESENTATIVE_SIMULTANEOUS_ARENA_BYTES = 1_275_902_976 as const;
const OPT_0081_REPRESENTATIVE_PRECOMPUTE_COMMANDS = 25 as const;
const OPT_0081_REPRESENTATIVE_TIMED_COMMANDS = 28 as const;
const OPT_0081_REPRESENTATIVE_LAYER_COMMANDS = Object.freeze([11, 15] as const);
const OPT_0081_REPRESENTATIVE_PHASE_COUNTS = Object.freeze([28] as const);
const OPT_0081_REPRESENTATIVE_READBACK_GUARD_BYTES = 256;
const OPT_0081_REPRESENTATIVE_READBACK_GUARD_U32 = 0x5a17_c3e9;
const OPT_0081_REPRESENTATIVE_QNAN_U32 = 0x7fc0_0001;
const OPT_0081_REPRESENTATIVE_QNAN_U16 = 0x7e01;

interface AceOpt0081RepresentativeDispatch {
  readonly label: string;
  readonly cooperativeSequence: AceGpuEncodeSequence;
}

type AceOpt0081RepresentativeDispatchSource = Readonly<{
  readonly label: string;
  readonly cooperativeSequence?: AceGpuEncodeSequence;
  readonly plan?: unknown;
  readonly rangeCount?: number;
  readonly encodeRange?: (
    pass: GPUComputePassEncoder,
    rangeIndex: number,
  ) => void;
  readonly encode: (pass: GPUComputePassEncoder) => void;
}>;

interface AceOpt0081RepresentativeEncodedQuantum {
  readonly sequence: AceGpuEncodeSequence;
  readonly sequenceQuantumIndex: number;
  readonly publicDescriptor: AceOpt0081RepresentativePhysicalQuantum;
}

interface AceOpt0081RepresentativeArmResources {
  readonly id: AceOpt0081RepresentativeArm;
  readonly denseInputStorageProfile: AceDitDenseInputStorageProfile | undefined;
  readonly arenaPlan: AceDitBackendArenaPlan;
  readonly arena: AceGpuArena;
  readonly bindings: AceDitGraphBindings;
  readonly runtime: AceCorrectnessDitRuntime;
  readonly precompute: readonly AceOpt0081RepresentativeEncodedQuantum[];
  readonly timedSlice: readonly AceOpt0081RepresentativeEncodedQuantum[];
}

interface AceOpt0081RepresentativeGuardedBinding {
  readonly buffer: GPUBuffer;
  readonly binding: GPUBufferBinding;
  readonly bodyBytes: number;
  readonly totalBytes: number;
}

interface AceOpt0081RepresentativeCheckpointTarget {
  readonly arm: AceOpt0081RepresentativeArmResources;
  readonly layer: AceOpt0081RepresentativeLayer;
  readonly tap: AceOpt0081RepresentativeTap;
  readonly runtime: AceCorrectnessDitRuntime;
  readonly guarded: AceOpt0081RepresentativeGuardedBinding;
  readonly storage: "u16" | "u32";
  readonly columns: number;
  readonly timedSlice: readonly AceOpt0081RepresentativeEncodedQuantum[];
}

interface AceOpt0081RepresentativeRunCapture {
  readonly startedAtPerformanceMs: number;
  readonly drainedAtPerformanceMs: number;
  readonly wallMs: number;
  readonly maximumPendingDescriptorCount: number;
  readonly pendingDescriptorCountAfterRun: 0;
  readonly fenceTimings: readonly AceDepth2Epoch4CommandBufferCompletionTiming[];
  readonly epochTimings: readonly AceDepth2Epoch4CompletionEpochTiming[];
  readonly scheduling: Awaited<ReturnType<
    AceCooperativeGpuScheduler["runLazyDepth2Epoch4"]
  >>;
}

/**
 * Benchmark-only owner for the frozen OPT-0081 two-layer graph prefix.
 *
 * One authenticated model backs two independently reset activation arenas.
 * The ordinary product backend cannot construct or select this owner.
 */
export class AceOpt0081RepresentativeDitOwner
  implements AceOpt0081RepresentativeDitSession {
  readonly memory: AceOpt0081RepresentativeMemoryAccounting;
  readonly topology: AceOpt0081RepresentativeTopology;

  private readonly scheduler = new AceCooperativeGpuScheduler();
  private readonly lifetime = new AbortController();
  private readonly externalSignal: AbortSignal | undefined;
  private readonly externalAbortListener: (() => void) | undefined;
  private readonly now: () => number;
  private state: "ready" | "destroying" | "destroyed" = "ready";
  private activeOperation = false;
  private activeOperationDone: Promise<void> | undefined;
  private finishActiveOperation: (() => void) | undefined;
  private precomputed = false;
  private destroyPromise: Promise<void> | undefined;
  private checkpointTarget: AceOpt0081RepresentativeCheckpointTarget |
    undefined;
  private controlCorrectnessArenaSnapshot: AceGpuArena | undefined;
  private candidateCorrectnessArenaSnapshot: AceGpuArena | undefined;
  private createdGraphBufferCount: number;
  private destroyedGraphBufferCount = 0;
  private liveGraphBufferCount: number;
  private liveGraphByteCount: number;
  private maximumLiveGraphByteCount: number;
  private mappedRangeCount = 0;
  private unmappedRangeCount = 0;
  private liveMapCount = 0;
  private pendingDescriptorCount = 0;
  private maximumPendingDescriptorCount = 0;
  private activeCallbackCount = 0;
  private activeLeaseCount = 0;
  private maximumActiveLeaseCount = 0;
  private maximumCorrectnessTargetBytes = 0;
  private liveCorrectnessRuntimeCount: 0 | 1 = 0;
  private maximumLiveCorrectnessRuntimeCount: 0 | 1 = 0;
  private correctnessTargetCompilationCount = 0;
  private correctnessTargetReuseCount = 0;
  private checkpointSnapshotCount = 0;
  private maximumDetachedCheckpointBytes = 0;
  private resetOrPrefillCount = 0;
  private profileSwitchCount = 0;
  private snapshotMapCount = 0;
  private guardedTargetReleaseCount = 0;
  private armReleaseCount = 0;
  private drainOrderViolationCount = 0;
  private profileSwitchWhilePendingCount = 0;
  private lastScheduledArm: AceOpt0081RepresentativeArm | undefined;
  private runtimeOwnerCount = 2;
  private destroyedRuntimeOwnerCount = 0;
  private residentModelDestroyed = false;
  private destroyCallCount = 0;
  private postDestroyRejectedOperationCount = 0;
  private deviceLossCleanupExecuted = false;

  private constructor(
    private readonly device: GPUDevice,
    private readonly model: AceDitResidentModel,
    private readonly plan: AceDitEvaluationPlan,
    private readonly selection: ReturnType<typeof resolveAceDitMixedGemmSelection>,
    private readonly control: AceOpt0081RepresentativeArmResources,
    private readonly candidate: AceOpt0081RepresentativeArmResources,
    private readonly correctnessReadback: GPUBuffer,
    memory: AceOpt0081RepresentativeMemoryAccounting,
    topology: AceOpt0081RepresentativeTopology,
    externalSignal: AbortSignal | undefined,
    now: () => number,
    private readonly residentModelBufferCount: number,
    private readonly setupFailureCleanupExecuted: boolean,
  ) {
    this.memory = memory;
    this.topology = topology;
    this.externalSignal = externalSignal;
    this.now = now;
    this.createdGraphBufferCount = residentModelBufferCount +
      control.arena.buffers.length + candidate.arena.buffers.length + 1;
    this.liveGraphBufferCount = this.createdGraphBufferCount;
    this.liveGraphByteCount = model.residentBytes + control.arena.byteLength +
      candidate.arena.byteLength + correctnessReadback.size;
    this.maximumLiveGraphByteCount = this.liveGraphByteCount;
    this.externalAbortListener = externalSignal === undefined
      ? undefined
      : () => {
          if (!this.lifetime.signal.aborted) {
            this.lifetime.abort(externalSignal.reason);
          }
          void this.beginDestroy(externalSignal.reason).catch(() => undefined);
        };
    if (externalSignal?.aborted === true) this.externalAbortListener!();
    else {
      externalSignal?.addEventListener(
        "abort",
        this.externalAbortListener!,
        { once: true },
      );
    }
    const weakOwner = new WeakRef(this);
    void device.lost.then((info) => {
      const owner = weakOwner.deref();
      if (owner === undefined || owner.state !== "ready") return;
      const error = new AceDitBackendDeviceLostError(info);
      owner.lifetime.abort(error);
      void owner.beginDestroy(error).catch(() => undefined);
    }).catch(() => undefined);
  }

  static async create(
    options: AceOpt0081RepresentativeDitOwnerOptions,
  ): Promise<AceOpt0081RepresentativeDitOwner> {
    let residentModelBufferCount = 0;
    let model: AceDitResidentModel | undefined;
    let control: AceOpt0081RepresentativeArmResources | undefined;
    let candidate: AceOpt0081RepresentativeArmResources | undefined;
    let readback: GPUBuffer | undefined;
    try {
      model = AceDitResidentModel.takeMixed(
        options.ownedDitWeights,
        options.ownedDitDenseWeights,
        "reference-bf16",
        options.ditDenseRuntimeProfile,
      );
      residentModelBufferCount = opt0081RepresentativeResidentBufferCount(
        options.ownedDitWeights,
        options.ownedDitDenseWeights,
      );
      const verifiedSetupFailureCleanup = requireOpt0081SetupEvidence(
        options.verifiedSetupFailureCleanup,
      );
      const signal = options.signal ?? new AbortController().signal;
      signal.throwIfAborted();
      const plan = planAceDitEvaluation(ACE_TURBO_DIT_CONFIG, options.shape);
      if (
        options.executionProfile.id !== ACE_REFERENCE_SUBGROUP_PROFILE.id ||
        options.subgroupMinSize !== 32 || options.subgroupMaxSize !== 32 ||
        plan.batch !== 1 || plan.latentFrames !== 4_500 ||
        plan.tokens !== 2_250 || plan.conditionTokens !== 98 ||
        options.ditDenseRuntimeProfile !==
          ACE_OPT_0009_DIT_DENSE_RUNTIME_PROFILE ||
        options.ditAttentionRuntimeProfile !==
          ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE ||
        model.modelProfile !== "reference-bf16" ||
        model.residentBytes !== ACE_OPT_0009_DIT_MIXED_RESIDENT_WEIGHT_BYTES ||
        ACE_COOPERATIVE_GPU_IDLE_MILLISECONDS !== 1
      ) {
        throw new Error(
          "OPT-0081 representative owner requires the exact authenticated " +
            "reference M2250/C98 OPT-0009/OPT-0070 fixed32 tuple",
        );
      }
      const selection = resolveAceDitMixedGemmSelection(
        options.executionProfile,
        options.subgroupMinSize,
        options.subgroupMaxSize,
        options.ditDenseRuntimeProfile,
        options.ditAttentionRuntimeProfile,
        plan.tokens,
        plan.conditionTokens,
      );
      const snapshots = snapshotInputs(plan, options.inputs);
      const controls = createAceDitGraphControlData(
        options.shape,
        options.dcwConfiguration,
      );
      control = await createOpt0081RepresentativeArmResources(
        options.device,
        "A",
        undefined,
        plan,
        model,
        selection,
        snapshots,
        controls,
        signal,
      );
      if (options.setupFailurePoint === "after-control-arm") {
        throw new AceOpt0081RepresentativeInjectedSetupFailure(
          options.setupFailurePoint,
        );
      }
      candidate = await createOpt0081RepresentativeArmResources(
        options.device,
        "B",
        ACE_OPT_0081_DIT_F16_DENSE_INPUT_STORAGE_PROFILE,
        plan,
        model,
        selection,
        snapshots,
        controls,
        signal,
      );
      if (options.setupFailurePoint === "after-candidate-arm") {
        throw new AceOpt0081RepresentativeInjectedSetupFailure(
          options.setupFailurePoint,
        );
      }
      requireOpt0081RepresentativeArmAgreement(control, candidate);
      const largestTapBytes = Math.max(
        largestOpt0081RepresentativeTapBytes(control),
        largestOpt0081RepresentativeTapBytes(candidate),
      );
      const correctnessReadbackBytes = checkedSum([
        largestTapBytes,
        4 * OPT_0081_REPRESENTATIVE_READBACK_GUARD_BYTES,
      ], "OPT-0081 correctness readback bytes");
      if (
        correctnessReadbackBytes > options.device.limits.maxBufferSize ||
        largestTapBytes > options.device.limits.maxStorageBufferBindingSize
      ) {
        throw new RangeError(
          "OPT-0081 representative correctness readback exceeds device limits",
        );
      }
      [readback] = await createAceScopedBuffers(
        options.device,
        [{
          label: "ace-opt-0081-representative-correctness-readback",
          size: correctnessReadbackBytes,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        }],
        "OPT-0081 representative correctness readback allocation",
      );
      if (readback === undefined) {
        throw new Error("OPT-0081 correctness readback allocation is incomplete");
      }
      if (options.setupFailurePoint === "after-readback") {
        throw new AceOpt0081RepresentativeInjectedSetupFailure(
          options.setupFailurePoint,
        );
      }
      const topology = createOpt0081RepresentativeTopology(control, candidate);
      const memory = createOpt0081RepresentativeMemoryAccounting(
        model,
        control,
        candidate,
        correctnessReadbackBytes,
      );
      signal.throwIfAborted();
      const ownedModel = model;
      const ownedControl = control;
      const ownedCandidate = candidate;
      const ownedReadback = readback;
      model = undefined;
      control = undefined;
      candidate = undefined;
      readback = undefined;
      return new AceOpt0081RepresentativeDitOwner(
        options.device,
        ownedModel,
        plan,
        selection,
        ownedControl,
        ownedCandidate,
        ownedReadback,
        memory,
        topology,
        options.signal,
        options.now ?? (() => performance.now()),
        residentModelBufferCount,
        verifiedSetupFailureCleanup,
      );
    } catch (error) {
      const createdGraphBufferCount = residentModelBufferCount +
        (control?.arena.buffers.length ?? 0) +
        (candidate?.arena.buffers.length ?? 0) + (readback === undefined ? 0 : 1);
      const runtimeOwnerCount = (control === undefined ? 0 : 1) +
        (candidate === undefined ? 0 : 1);
      readback?.destroy();
      destroyOpt0081RepresentativeArm(candidate);
      destroyOpt0081RepresentativeArm(control);
      model?.destroy();
      if (error instanceof AceOpt0081RepresentativeInjectedSetupFailure) {
        options.onSetupCleanup?.(Object.freeze({
          schema: "ace-opt-0081-representative-setup-cleanup-v1",
          createdGraphBufferCount,
          destroyedGraphBufferCount: createdGraphBufferCount,
          liveGraphBufferCount: 0,
          liveGraphByteCount: 0,
          runtimeOwnerCount,
          destroyedRuntimeOwnerCount: runtimeOwnerCount,
          residentModelDestroyed: true,
          mappedRangeCount: 0,
          unmappedRangeCount: 0,
          liveMapCount: 0,
          pendingDescriptorCount: 0,
          activeCallbackCount: 0,
          activeLeaseCount: 0,
          armReleaseCount: (control === undefined ? 0 : 1) +
            (candidate === undefined ? 0 : 1),
          drainOrderViolationCount: 0,
        }));
      }
      throw error;
    }
  }

  async runPrecompute(
    options: AceOpt0081RepresentativeRunOptions = {},
  ): Promise<void> {
    this.beginOperation("precompute");
    try {
      if (this.precomputed) return;
      const signal = this.activeSignal(options.signal);
      await this.runPhysicalQuanta(
        this.control,
        this.control.precompute,
        [OPT_0081_REPRESENTATIVE_PRECOMPUTE_COMMANDS],
        signal,
      );
      await this.runPhysicalQuanta(
        this.candidate,
        this.candidate.precompute,
        [OPT_0081_REPRESENTATIVE_PRECOMPUTE_COMMANDS],
        signal,
      );
      signal.throwIfAborted();
      this.precomputed = true;
    } finally {
      this.endOperation();
    }
  }

  async beginCorrectnessCheckpointing(
    options: AceOpt0081RepresentativeRunOptions = {},
  ): Promise<void> {
    this.beginOperation("begin correctness checkpointing");
    let controlSnapshot: AceGpuArena | undefined;
    let candidateSnapshot: AceGpuArena | undefined;
    try {
      this.requirePrecomputed();
      if (
        this.controlCorrectnessArenaSnapshot !== undefined ||
        this.candidateCorrectnessArenaSnapshot !== undefined
      ) {
        throw new DOMException(
          "OPT-0081 correctness checkpointing is already active",
          "InvalidStateError",
        );
      }
      const signal = this.activeSignal(options.signal);
      signal.throwIfAborted();
      controlSnapshot = await createOpt0081RepresentativeArenaSnapshot(
        this.device,
        this.control,
      );
      this.noteGraphBuffersCreated(
        controlSnapshot.buffers.length,
        controlSnapshot.byteLength,
      );
      candidateSnapshot = await createOpt0081RepresentativeArenaSnapshot(
        this.device,
        this.candidate,
      );
      this.noteGraphBuffersCreated(
        candidateSnapshot.buffers.length,
        candidateSnapshot.byteLength,
      );
      await this.copyCorrectnessArena(
        this.control.arena,
        controlSnapshot,
        "capture-control",
        signal,
      );
      await this.copyCorrectnessArena(
        this.candidate.arena,
        candidateSnapshot,
        "capture-candidate",
        signal,
      );
      this.controlCorrectnessArenaSnapshot = controlSnapshot;
      this.candidateCorrectnessArenaSnapshot = candidateSnapshot;
      controlSnapshot = undefined;
      candidateSnapshot = undefined;
    } finally {
      if (candidateSnapshot !== undefined) {
        candidateSnapshot.destroy();
        this.noteGraphBuffersDestroyed(
          candidateSnapshot.buffers.length,
          candidateSnapshot.byteLength,
        );
      }
      if (controlSnapshot !== undefined) {
        controlSnapshot.destroy();
        this.noteGraphBuffersDestroyed(
          controlSnapshot.buffers.length,
          controlSnapshot.byteLength,
        );
      }
      this.endOperation();
    }
  }

  async endCorrectnessCheckpointing(
    options: AceOpt0081RepresentativeRunOptions = {},
  ): Promise<void> {
    this.beginOperation("end correctness checkpointing");
    try {
      const signal = this.activeSignal(options.signal);
      const controlSnapshot = this.requireCorrectnessArenaSnapshot("A");
      const candidateSnapshot = this.requireCorrectnessArenaSnapshot("B");
      try {
        this.noteResetOrPrefill("final control correctness arena restore");
        await this.copyCorrectnessArena(
          controlSnapshot,
          this.control.arena,
          "final-restore-control",
          signal,
        );
        this.noteResetOrPrefill("final candidate correctness arena restore");
        await this.copyCorrectnessArena(
          candidateSnapshot,
          this.candidate.arena,
          "final-restore-candidate",
          signal,
        );
      } finally {
        this.destroyCorrectnessArenaSnapshots();
      }
    } finally {
      this.endOperation();
    }
  }

  async snapshotCheckpoint(
    options: AceOpt0081RepresentativeCheckpointOptions,
  ): Promise<AceOpt0081RepresentativeCheckpointSnapshot> {
    this.beginOperation("correctness checkpoint");
    let substage: AceOpt0081RepresentativeCheckpointSubstage =
      "compile-guarded-target";
    try {
      this.requirePrecomputed();
      const signal = this.activeSignal(options.signal);
      const arm = this.arm(options.arm);
      const target = await this.prepareCheckpointTarget(
        arm,
        options.layer,
        options.tap,
        signal,
      );
      substage = "restore-canonical-arena";
      this.noteResetOrPrefill("canonical correctness arena restore");
      await this.copyCorrectnessArena(
        this.requireCorrectnessArenaSnapshot(options.arm),
        arm.arena,
        `restore-${options.arm}-layer-${options.layer}-${options.tap}`,
        signal,
      );
      substage = "reset-initial-guards";
      this.noteResetOrPrefill("guard reset");
      resetOpt0081RepresentativeGuardedTarget(this.device.queue, target);
      if (options.layer === 0) {
        substage = "prefill-layer-0-target";
        this.noteResetOrPrefill("qNaN prefill");
        prefillOpt0081RepresentativeTap(
          this.device.queue,
          target.guarded.binding,
          target.storage,
        );
        substage = "execute-through-layer-0";
        await this.runPhysicalQuanta(
          arm,
          target.timedSlice.slice(0, 13),
          [13],
          signal,
        );
      } else {
        substage = "execute-through-layer-0";
        await this.runPhysicalQuanta(
          arm,
          target.timedSlice.slice(0, 13),
          [13],
          signal,
        );
        signal.throwIfAborted();
        substage = "reset-layer-1-guards";
        this.noteResetOrPrefill("guard reset");
        resetOpt0081RepresentativeGuardedTarget(this.device.queue, target);
        substage = "prefill-layer-1-target";
        this.noteResetOrPrefill("qNaN prefill");
        prefillOpt0081RepresentativeTap(
          this.device.queue,
          target.guarded.binding,
          target.storage,
        );
        substage = "execute-layer-1";
        await this.runPhysicalQuanta(
          arm,
          target.timedSlice.slice(13),
          [15],
          signal,
        );
      }
      substage = "readback-and-map";
      return await this.readOpt0081RepresentativeTap(
        arm,
        options.layer,
        options.tap,
        target,
        signal,
      );
    } catch (error) {
      if (error instanceof AceOpt0081RepresentativeCheckpointExecutionError) {
        throw error;
      }
      throw new AceOpt0081RepresentativeCheckpointExecutionError(
        options.arm,
        options.layer,
        options.tap,
        substage,
        error,
      );
    } finally {
      this.endOperation();
    }
  }

  async warmup(
    armId: AceOpt0081RepresentativeArm,
    options: AceOpt0081RepresentativeRunOptions = {},
  ): Promise<void> {
    this.beginOperation("warmup");
    try {
      this.requirePrecomputed();
      this.releaseCheckpointTarget();
      const signal = this.activeSignal(options.signal);
      await this.runPhysicalQuanta(
        this.arm(armId),
        this.arm(armId).timedSlice,
        OPT_0081_REPRESENTATIVE_PHASE_COUNTS,
        signal,
      );
    } finally {
      this.endOperation();
    }
  }

  async runTimedSlice(
    armId: AceOpt0081RepresentativeArm,
    options: AceOpt0081RepresentativeRunOptions = {},
    hooks: AceOpt0081RepresentativeTimedHooks = {},
  ): Promise<AceOpt0081RepresentativeTimedSliceResult> {
    this.beginOperation("timed slice");
    try {
      this.requirePrecomputed();
      this.releaseCheckpointTarget();
      const signal = this.activeSignal(options.signal);
      const arm = this.arm(armId);
      const capture = await this.runPhysicalQuanta(
        arm,
        arm.timedSlice,
        OPT_0081_REPRESENTATIVE_PHASE_COUNTS,
        signal,
        hooks,
      );
      return createOpt0081RepresentativeTimedResult(
        armId,
        capture,
        this.topology,
      );
    } finally {
      this.endOperation();
    }
  }

  async runCancellationPreflight(
    armId: AceOpt0081RepresentativeArm = "B",
    options: AceOpt0081RepresentativeRunOptions = {},
  ): Promise<AceOpt0081RepresentativeCancellationEvidence> {
    this.beginOperation("cancellation preflight");
    try {
      if (armId !== "B") {
        throw new Error("OPT-0081 cancellation preflight is candidate-only");
      }
      this.requirePrecomputed();
      this.releaseCheckpointTarget();
      const ownerSignal = this.activeSignal(options.signal);
      const local = new AbortController();
      const stop = Object.freeze({ kind: "ace-opt-0081-cancellation" });
      const signal = combineSignals([
        ownerSignal,
        local.signal,
      ]);
      let commandBuffersCreated = 0;
      let maximumPendingDescriptorCount = 0;
      let completionCallbackCount = 0;
      let completionCallbackCountAfterAbort = 0;
      let observedCommandBufferIndex = -1;
      let successorAlreadySubmitted = false;
      let abortedAt = Number.NaN;
      let caught: unknown;
      try {
        await this.runPhysicalQuanta(
          this.candidate,
          this.candidate.timedSlice,
          OPT_0081_REPRESENTATIVE_PHASE_COUNTS,
          signal,
          {
            onCommandBufferCompleted: (timing) => {
              if (local.signal.aborted) completionCallbackCountAfterAbort += 1;
              completionCallbackCount += 1;
              if (completionCallbackCount !== 1) return;
              observedCommandBufferIndex = timing.commandBufferIndex;
              successorAlreadySubmitted = commandBuffersCreated === 2;
              abortedAt = this.now();
              local.abort(stop);
            },
          },
          () => {
            commandBuffersCreated += 1;
          },
          (count) => {
            maximumPendingDescriptorCount = Math.max(
              maximumPendingDescriptorCount,
              count,
            );
          },
        );
      } catch (error) {
        caught = error;
      }
      const settlementWallMs = requireOpt0080TimingElapsed(
        this.now(),
        abortedAt,
      );
      if (
        caught !== stop || commandBuffersCreated !== 2 ||
        completionCallbackCount !== 1 ||
        completionCallbackCountAfterAbort !== 0 ||
        observedCommandBufferIndex !== 0 || !successorAlreadySubmitted ||
        maximumPendingDescriptorCount !== 2 ||
        this.pendingDescriptorCount !== 0
      ) {
        throw new Error(
          "OPT-0081 cancellation did not stop with one submitted successor",
          { cause: caught },
        );
      }
      return Object.freeze({
        schema: "ace-opt-0081-representative-cancellation-v1",
        arm: armId,
        residentArmReused: true,
        observedCommandBufferIndex: 0,
        successorAlreadySubmitted: true,
        commandBuffersCreated: 2,
        completionCallbackCount: 1,
        completionCallbackCountAfterAbort: 0,
        noBackfillAfterObservation: true,
        originalReasonPreserved: true,
        submittedFencesSettledBeforeReturn: true,
        settlementWallMs,
        cleanupWithinOneSecond: settlementWallMs <= 1_000,
        maximumPendingDescriptorCount,
        pendingDescriptorCountAfterRun: 0,
        temporaryCreatedGraphBufferCount: 0,
        temporaryDestroyedGraphBufferCount: 0,
        temporaryLiveGraphBufferCountAfterCleanup: 0,
        temporaryLiveGraphByteCountAfterCleanup: 0,
        temporaryRuntimeOwnerCount: 0,
        temporaryDestroyedRuntimeOwnerCount: 0,
      });
    } finally {
      this.endOperation();
    }
  }

  lifecycleSnapshot(): AceOpt0081RepresentativeLifecycleSnapshot {
    return Object.freeze({
      schema: "ace-opt-0081-representative-lifecycle-v1",
      state: this.state,
      createdGraphBufferCount: this.createdGraphBufferCount,
      destroyedGraphBufferCount: this.destroyedGraphBufferCount,
      createdBufferCount: this.createdGraphBufferCount,
      destroyedBufferCount: this.destroyedGraphBufferCount,
      liveGraphBufferCount: this.liveGraphBufferCount,
      liveGraphByteCount: this.liveGraphByteCount,
      liveBufferCount: this.liveGraphBufferCount,
      liveByteCount: this.liveGraphByteCount,
      maximumLiveGraphByteCount: this.maximumLiveGraphByteCount,
      mappedRangeCount: this.mappedRangeCount,
      unmappedRangeCount: this.unmappedRangeCount,
      liveMapCount: this.liveMapCount,
      pendingDescriptorCount: this.pendingDescriptorCount,
      maximumPendingDescriptorCount: this.maximumPendingDescriptorCount,
      activeCallbackCount: this.activeCallbackCount,
      activeLeaseCount: this.activeLeaseCount,
      callbackCount: this.activeCallbackCount,
      leaseCount: this.activeLeaseCount,
      maximumActiveLeaseCount: this.maximumActiveLeaseCount,
      correctnessTargetCount: this.checkpointTarget === undefined ? 0 : 1,
      maximumCorrectnessTargetCount: 1,
      correctnessRuntimeCount: this.liveCorrectnessRuntimeCount,
      maximumCorrectnessRuntimeCount:
        this.maximumLiveCorrectnessRuntimeCount,
      maximumCorrectnessTargetBytes: this.maximumCorrectnessTargetBytes,
      correctnessTargetCompilationCount:
        this.correctnessTargetCompilationCount,
      correctnessTargetReuseCount: this.correctnessTargetReuseCount,
      checkpointSnapshotCount: this.checkpointSnapshotCount,
      maximumDetachedCheckpointBytes: this.maximumDetachedCheckpointBytes,
      resetOrPrefillCount: this.resetOrPrefillCount,
      profileSwitchCount: this.profileSwitchCount,
      snapshotMapCount: this.snapshotMapCount,
      guardedTargetReleaseCount: this.guardedTargetReleaseCount,
      armReleaseCount: this.armReleaseCount,
      drainOrderViolationCount: this.drainOrderViolationCount,
      profileSwitchWhilePendingCount: this.profileSwitchWhilePendingCount,
      runtimeOwnerCount: this.runtimeOwnerCount,
      destroyedRuntimeOwnerCount: this.destroyedRuntimeOwnerCount,
      residentModelDestroyed: this.residentModelDestroyed,
      precomputeCompleted: this.precomputed,
      destroyCallCount: this.destroyCallCount,
      postDestroyRejectedOperationCount:
        this.postDestroyRejectedOperationCount,
      setupFailureCleanupExecuted: this.setupFailureCleanupExecuted,
      deviceLossCleanupExecuted: this.deviceLossCleanupExecuted,
    });
  }

  get lifecycle(): AceOpt0081RepresentativeLifecycleSnapshot {
    return this.lifecycleSnapshot();
  }

  async runDeviceLossCleanupPreflight(): Promise<
    AceOpt0081RepresentativeDeviceLossCleanupEvidence
  > {
    this.beginOperation("terminal device-loss cleanup preflight");
    this.releaseCheckpointTarget();
    if (
      this.pendingDescriptorCount !== 0 || this.activeCallbackCount !== 0 ||
      this.liveMapCount !== 0
    ) {
      this.endOperation();
      throw new Error("OPT-0081 device-loss preflight requires a drained owner");
    }
    this.endOperation();
    this.device.destroy();
    const lost = await this.device.lost;
    await this.beginDestroy(new AceDitBackendDeviceLostError(lost));
    this.deviceLossCleanupExecuted = true;
    await this.destroy();
    await this.destroy();
    let postDestroyRejected = false;
    const rejectionsBefore = this.postDestroyRejectedOperationCount;
    try {
      await this.runPrecompute();
    } catch {
      postDestroyRejected = true;
    }
    if (
      this.postDestroyRejectedOperationCount !== rejectionsBefore + 1
    ) throw new Error("OPT-0081 post-destroy rejection accounting changed");
    const lifecycle = this.lifecycleSnapshot();
    if (
      lifecycle.state !== "destroyed" ||
      lifecycle.liveGraphBufferCount !== 0 ||
      lifecycle.liveGraphByteCount !== 0 || lifecycle.liveMapCount !== 0 ||
      lifecycle.pendingDescriptorCount !== 0 ||
      lifecycle.activeCallbackCount !== 0 || lifecycle.activeLeaseCount !== 0 ||
      !postDestroyRejected
    ) throw new Error("OPT-0081 device-loss cleanup did not reach zero live state");
    return Object.freeze({
      schema: "ace-opt-0081-representative-device-loss-cleanup-v1",
      deviceLossInduced: true,
      deviceLossObserved: true,
      ownerDestroyedAfterLoss: true,
      liveGraphBufferCount: 0,
      liveGraphByteCount: 0,
      liveMapCount: 0,
      pendingDescriptorCount: 0,
      activeCallbackCount: 0,
      activeLeaseCount: 0,
      idempotentDestroyVerified: true,
      postDestroyRejected: true,
    });
  }

  destroy(): Promise<void> {
    this.destroyCallCount += 1;
    return this.beginDestroy(destroyedError());
  }

  private async beginDestroy(reason: unknown): Promise<void> {
    if (this.destroyPromise !== undefined) return await this.destroyPromise;
    if (this.state === "destroyed") return;
    this.state = "destroying";
    if (!this.lifetime.signal.aborted) this.lifetime.abort(reason);
    this.externalSignal?.removeEventListener(
      "abort",
      this.externalAbortListener!,
    );
    this.destroyPromise = (async () => {
      await this.activeOperationDone;
      await this.scheduler.dispose();
      this.releaseCheckpointTarget();
      this.destroyCorrectnessArenaSnapshots();
      this.requireDrainedEvent("correctness readback release");
      this.correctnessReadback.destroy();
      this.noteGraphBufferDestroyed(this.correctnessReadback.size);
      this.noteArmRelease("candidate arm release");
      destroyOpt0081RepresentativeArm(this.candidate);
      this.noteGraphBuffersDestroyed(
        this.candidate.arena.buffers.length,
        this.candidate.arena.byteLength,
      );
      this.noteArmRelease("control arm release");
      destroyOpt0081RepresentativeArm(this.control);
      this.noteGraphBuffersDestroyed(
        this.control.arena.buffers.length,
        this.control.arena.byteLength,
      );
      this.destroyedRuntimeOwnerCount += 2;
      this.requireDrainedEvent("resident model release");
      this.model.destroy();
      this.noteGraphBuffersDestroyed(
        this.residentModelBufferCount,
        this.model.residentBytes,
      );
      this.residentModelDestroyed = true;
      this.state = "destroyed";
    })();
    return await this.destroyPromise;
  }

  private beginOperation(operation: string): void {
    if (this.state !== "ready") {
      this.postDestroyRejectedOperationCount += 1;
      throw destroyedError();
    }
    if (this.activeOperation) {
      throw new DOMException(
        `OPT-0081 cannot start ${operation} while another owner operation is active`,
        "InvalidStateError",
      );
    }
    this.activeOperation = true;
    const operationDone = Promise.withResolvers<void>();
    this.activeOperationDone = operationDone.promise;
    this.finishActiveOperation = () => operationDone.resolve();
  }

  private endOperation(): void {
    this.activeOperation = false;
    this.finishActiveOperation?.();
    this.finishActiveOperation = undefined;
    this.activeOperationDone = undefined;
  }

  private requirePrecomputed(): void {
    if (!this.precomputed) {
      throw new DOMException(
        "OPT-0081 condition/cross-cache precompute has not completed",
        "InvalidStateError",
      );
    }
  }

  private activeSignal(signal?: AbortSignal): AbortSignal {
    return combineSignals([this.lifetime.signal, signal]);
  }

  private noteGraphBufferCreated(bytes: number): void {
    this.noteGraphBuffersCreated(1, bytes);
  }

  private noteGraphBuffersCreated(count: number, bytes: number): void {
    this.createdGraphBufferCount += count;
    this.liveGraphBufferCount += count;
    this.liveGraphByteCount += bytes;
    this.maximumLiveGraphByteCount = Math.max(
      this.maximumLiveGraphByteCount,
      this.liveGraphByteCount,
    );
  }

  private noteGraphBufferDestroyed(bytes: number): void {
    this.noteGraphBuffersDestroyed(1, bytes);
  }

  private noteGraphBuffersDestroyed(count: number, bytes: number): void {
    this.destroyedGraphBufferCount += count;
    this.liveGraphBufferCount -= count;
    this.liveGraphByteCount -= bytes;
    if (this.liveGraphBufferCount < 0 || this.liveGraphByteCount < 0) {
      throw new Error("OPT-0081 graph resource accounting underflow");
    }
  }

  private requireDrainedEvent(label: string): void {
    if (this.pendingDescriptorCount === 0 && this.activeLeaseCount === 0) return;
    this.drainOrderViolationCount += 1;
    throw new Error(`OPT-0081 ${label} occurred before terminal drain`);
  }

  private noteResetOrPrefill(label: string): void {
    this.requireDrainedEvent(label);
    this.resetOrPrefillCount += 1;
  }

  private noteArmRelease(label: string): void {
    this.requireDrainedEvent(label);
    this.armReleaseCount += 1;
  }

  private requireCorrectnessArenaSnapshot(
    arm: AceOpt0081RepresentativeArm,
  ): AceGpuArena {
    const snapshot = arm === "A"
      ? this.controlCorrectnessArenaSnapshot
      : this.candidateCorrectnessArenaSnapshot;
    if (snapshot === undefined) {
      throw new DOMException(
        `OPT-0081 correctness checkpointing is not active for arm ${arm}`,
        "InvalidStateError",
      );
    }
    return snapshot;
  }

  private destroyCorrectnessArenaSnapshots(): void {
    const candidate = this.candidateCorrectnessArenaSnapshot;
    this.candidateCorrectnessArenaSnapshot = undefined;
    if (candidate !== undefined) {
      this.requireDrainedEvent("candidate correctness arena snapshot release");
      candidate.destroy();
      this.noteGraphBuffersDestroyed(
        candidate.buffers.length,
        candidate.byteLength,
      );
    }
    const control = this.controlCorrectnessArenaSnapshot;
    this.controlCorrectnessArenaSnapshot = undefined;
    if (control !== undefined) {
      this.requireDrainedEvent("control correctness arena snapshot release");
      control.destroy();
      this.noteGraphBuffersDestroyed(
        control.buffers.length,
        control.byteLength,
      );
    }
  }

  private async copyCorrectnessArena(
    source: AceGpuArena,
    destination: AceGpuArena,
    label: string,
    signal: AbortSignal,
  ): Promise<void> {
    this.requireDrainedEvent(`arena copy ${label}`);
    signal.throwIfAborted();
    const command = encodeOpt0081RepresentativeArenaCopy(
      this.device,
      source,
      destination,
      label,
    );
    this.activeLeaseCount += 1;
    this.maximumActiveLeaseCount = Math.max(
      this.maximumActiveLeaseCount,
      this.activeLeaseCount,
    );
    this.pendingDescriptorCount = 1;
    this.maximumPendingDescriptorCount = Math.max(
      this.maximumPendingDescriptorCount,
      1,
    );
    try {
      this.device.queue.submit([command]);
      await this.device.queue.onSubmittedWorkDone();
      signal.throwIfAborted();
    } finally {
      this.pendingDescriptorCount = 0;
      this.activeLeaseCount -= 1;
    }
  }

  private noteProfileSelection(arm: AceOpt0081RepresentativeArm): void {
    if (this.lastScheduledArm === undefined || this.lastScheduledArm === arm) {
      this.lastScheduledArm = arm;
      return;
    }
    if (this.pendingDescriptorCount !== 0 || this.activeLeaseCount !== 0) {
      this.profileSwitchWhilePendingCount += 1;
    }
    this.requireDrainedEvent("profile switch");
    this.profileSwitchCount += 1;
    this.lastScheduledArm = arm;
  }

  private arm(
    id: AceOpt0081RepresentativeArm,
  ): AceOpt0081RepresentativeArmResources {
    if (id === "A") return this.control;
    if (id === "B") return this.candidate;
    throw new TypeError(`Unknown OPT-0081 representative arm ${String(id)}`);
  }

  private async runPhysicalQuanta(
    arm: AceOpt0081RepresentativeArmResources,
    quanta: readonly AceOpt0081RepresentativeEncodedQuantum[],
    phaseCommandBufferCounts: readonly number[],
    signal: AbortSignal,
    hooks: AceOpt0081RepresentativeTimedHooks = {},
    onCommandBufferCreated?: () => void,
    onPendingDescriptorCountChanged?: (count: number) => void,
  ): Promise<AceOpt0081RepresentativeRunCapture> {
    this.noteProfileSelection(arm.id);
    signal.throwIfAborted();
    const fenceTimings: AceDepth2Epoch4CommandBufferCompletionTiming[] = [];
    const epochTimings: AceDepth2Epoch4CompletionEpochTiming[] = [];
    const pendingDescriptors = new Map<
      number,
      AceOpt0081RepresentativePhysicalQuantum
    >();
    let maximumPendingDescriptorCount = 0;
    let nextDescriptorIndex = 0;
    const startedAt = this.now();
    let scheduling: Awaited<ReturnType<
      AceCooperativeGpuScheduler["runLazyDepth2Epoch4"]
    >>;
    let drainedAt = Number.NaN;
    this.activeLeaseCount += 1;
    this.maximumActiveLeaseCount = Math.max(
      this.maximumActiveLeaseCount,
      this.activeLeaseCount,
    );
    try {
      scheduling = await this.scheduler.runLazyDepth2Epoch4({
        device: this.device,
        queue: this.device.queue,
        commandBufferCount: quanta.length,
        phaseCommandBufferCounts,
        signal,
        now: this.now,
        createCommandBuffer: (index) => {
          if (index !== nextDescriptorIndex) {
            throw new Error(
              `OPT-0081 created descriptor ${index} out of FIFO order; expected ${nextDescriptorIndex}`,
            );
          }
          if (pendingDescriptors.size >= 2) {
            throw new Error(
              "OPT-0081 exceeded the frozen depth-two pending descriptor bound",
            );
          }
          const quantum = quanta[index];
          if (quantum === undefined) {
            throw new RangeError(`OPT-0081 omitted physical quantum ${index}`);
          }
          if (pendingDescriptors.has(index)) {
            throw new Error(`OPT-0081 duplicated pending descriptor ${index}`);
          }
          pendingDescriptors.set(index, quantum.publicDescriptor);
          nextDescriptorIndex += 1;
          this.pendingDescriptorCount = pendingDescriptors.size;
          onPendingDescriptorCountChanged?.(pendingDescriptors.size);
          maximumPendingDescriptorCount = Math.max(
            maximumPendingDescriptorCount,
            pendingDescriptors.size,
          );
          this.maximumPendingDescriptorCount = Math.max(
            this.maximumPendingDescriptorCount,
            pendingDescriptors.size,
          );
          const encoder = this.device.createCommandEncoder({
            label: `ace-opt-0081-${arm.id}-command-${index}`,
          });
          const pass = encoder.beginComputePass({
            label: `ace-opt-0081-${arm.id}-pass-${index}`,
          });
          quantum.sequence.encodeQuantum(pass, quantum.sequenceQuantumIndex);
          pass.end();
          const command = encoder.finish();
          onCommandBufferCreated?.();
          return command;
        },
        onCommandBufferCompleted: (timing, progress) => {
          const oldestPendingIndex = pendingDescriptors.keys().next().value;
          if (oldestPendingIndex !== timing.commandBufferIndex) {
            throw new Error(
              `OPT-0081 completed descriptor ${timing.commandBufferIndex} out of FIFO order; expected ${String(oldestPendingIndex)}`,
            );
          }
          if (!pendingDescriptors.delete(timing.commandBufferIndex)) {
            throw new Error(
              `OPT-0081 completed unknown descriptor ${timing.commandBufferIndex}`,
            );
          }
          this.pendingDescriptorCount = pendingDescriptors.size;
          onPendingDescriptorCountChanged?.(pendingDescriptors.size);
          fenceTimings.push(timing);
          this.activeCallbackCount += 1;
          try {
            hooks.onCommandBufferCompleted?.(timing, progress);
          } finally {
            this.activeCallbackCount -= 1;
          }
        },
        onCompletionEpochDrained: (timing) => {
          epochTimings.push(timing);
          this.activeCallbackCount += 1;
          try {
            hooks.onCompletionEpochDrained?.(timing);
          } finally {
            this.activeCallbackCount -= 1;
          }
        },
      });
      if (pendingDescriptors.size !== 0 || nextDescriptorIndex !== quanta.length) {
        throw new Error(
          "OPT-0081 scheduler returned before every physical descriptor was created and drained",
        );
      }
      drainedAt = this.now();
    } finally {
      pendingDescriptors.clear();
      this.pendingDescriptorCount = 0;
      onPendingDescriptorCountChanged?.(0);
      this.activeLeaseCount -= 1;
    }
    const wallMs = requireOpt0080TimingElapsed(drainedAt, startedAt);
    return Object.freeze({
      startedAtPerformanceMs: startedAt,
      drainedAtPerformanceMs: drainedAt,
      wallMs,
      maximumPendingDescriptorCount,
      pendingDescriptorCountAfterRun: 0,
      fenceTimings: Object.freeze(fenceTimings),
      epochTimings: Object.freeze(epochTimings),
      scheduling,
    });
  }

  private async prepareCheckpointTarget(
    arm: AceOpt0081RepresentativeArmResources,
    layer: AceOpt0081RepresentativeLayer,
    tap: AceOpt0081RepresentativeTap,
    signal: AbortSignal,
  ): Promise<AceOpt0081RepresentativeCheckpointTarget> {
    const retained = this.checkpointTarget;
    if (
      retained !== undefined && retained.arm === arm &&
      retained.layer === layer && retained.tap === tap
    ) {
      this.correctnessTargetReuseCount += 1;
      return retained;
    }
    this.releaseCheckpointTarget();
    const base = opt0081RepresentativeTapBinding(arm, layer, tap);
    const bodyBytes = requireBindingSize(base.binding, "OPT-0081 guarded tap");
    const totalBytes = checkedSum([
      bodyBytes,
      2 * OPT_0081_REPRESENTATIVE_READBACK_GUARD_BYTES,
    ], "OPT-0081 guarded target bytes");
    if (
      totalBytes > this.device.limits.maxBufferSize ||
      bodyBytes > this.device.limits.maxStorageBufferBindingSize
    ) throw new RangeError("OPT-0081 guarded target exceeds device limits");
    const buffers = await createAceScopedBuffers(
      this.device,
      [{
        label: `ace-opt-0081-${arm.id}-layer-${layer}-${tap}-guarded`,
        size: totalBytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC |
          GPUBufferUsage.COPY_DST,
      }],
      "OPT-0081 guarded correctness target allocation",
    );
    const buffer = buffers[0];
    if (buffer === undefined) {
      throw new Error("OPT-0081 guarded target allocation is incomplete");
    }
    this.noteGraphBufferCreated(totalBytes);
    const guarded: AceOpt0081RepresentativeGuardedBinding = Object.freeze({
      buffer,
      binding: Object.freeze({
        buffer,
        offset: OPT_0081_REPRESENTATIVE_READBACK_GUARD_BYTES,
        size: bodyBytes,
      }),
      bodyBytes,
      totalBytes,
    });
    let runtime: AceCorrectnessDitRuntime | undefined;
    try {
      signal.throwIfAborted();
      const bindings = opt0081RepresentativeCheckpointBindings(
        arm.bindings,
        layer,
        tap,
        guarded.binding,
      );
      if (this.liveCorrectnessRuntimeCount !== 0) {
        throw new Error("OPT-0081 retained more than one correctness runtime");
      }
      runtime = AceCorrectnessDitRuntime.create(
        this.device,
        "reference-bf16",
        this.selection.gemmConfiguration,
        this.selection.denseGemmConfiguration,
        this.selection.attentionConfiguration,
        arm.denseInputStorageProfile,
      );
      this.runtimeOwnerCount += 1;
      this.liveCorrectnessRuntimeCount = 1;
      this.maximumLiveCorrectnessRuntimeCount = 1;
      const compiled = await compileOpt0081RepresentativeArm(
        runtime,
        this.model,
        this.plan,
        bindings,
        signal,
        false,
      );
      const target = Object.freeze({
        arm,
        layer,
        tap,
        runtime,
        guarded,
        storage: base.storage,
        columns: base.columns,
        timedSlice: compiled.timedSlice,
      });
      this.checkpointTarget = target;
      this.correctnessTargetCompilationCount += 1;
      this.maximumCorrectnessTargetBytes = Math.max(
        this.maximumCorrectnessTargetBytes,
        totalBytes,
      );
      runtime = undefined;
      return target;
    } catch (error) {
      if (runtime !== undefined) {
        runtime.destroy();
        this.destroyedRuntimeOwnerCount += 1;
        this.noteCorrectnessRuntimeDestroyed();
      }
      buffer.destroy();
      this.noteGraphBufferDestroyed(totalBytes);
      throw error;
    }
  }

  private releaseCheckpointTarget(): void {
    const target = this.checkpointTarget;
    if (target === undefined) return;
    this.requireDrainedEvent("guarded target release");
    this.guardedTargetReleaseCount += 1;
    this.checkpointTarget = undefined;
    target.runtime.destroy();
    this.destroyedRuntimeOwnerCount += 1;
    this.noteCorrectnessRuntimeDestroyed();
    target.guarded.buffer.destroy();
    this.noteGraphBufferDestroyed(target.guarded.totalBytes);
  }

  private noteCorrectnessRuntimeDestroyed(): void {
    if (this.liveCorrectnessRuntimeCount !== 1) {
      throw new Error("OPT-0081 correctness runtime accounting underflow");
    }
    this.liveCorrectnessRuntimeCount = 0;
  }

  private async readOpt0081RepresentativeTap(
    arm: AceOpt0081RepresentativeArmResources,
    layer: AceOpt0081RepresentativeLayer,
    tap: AceOpt0081RepresentativeTap,
    target: AceOpt0081RepresentativeCheckpointTarget,
    signal: AbortSignal,
  ): Promise<AceOpt0081RepresentativeCheckpointSnapshot> {
    const byteLength = target.guarded.bodyBytes;
    const sourceBytes = target.guarded.totalBytes;
    const guard = new Uint32Array(
      OPT_0081_REPRESENTATIVE_READBACK_GUARD_BYTES / UINT32_BYTES,
    );
    guard.fill(OPT_0081_REPRESENTATIVE_READBACK_GUARD_U32);
    this.device.queue.writeBuffer(this.correctnessReadback, 0, guard);
    this.device.queue.writeBuffer(
      this.correctnessReadback,
      OPT_0081_REPRESENTATIVE_READBACK_GUARD_BYTES + sourceBytes,
      guard,
    );
    const totalBytes = sourceBytes +
      2 * OPT_0081_REPRESENTATIVE_READBACK_GUARD_BYTES;
    this.activeLeaseCount += 1;
    this.maximumActiveLeaseCount = Math.max(
      this.maximumActiveLeaseCount,
      this.activeLeaseCount,
    );
    try {
      await this.scheduler.runLazyDepth2Epoch4({
        device: this.device,
        queue: this.device.queue,
        commandBufferCount: 1,
        phaseCommandBufferCounts: [1],
        signal,
        now: this.now,
        createCommandBuffer: () => {
          if (this.pendingDescriptorCount !== 0) {
            throw new Error("OPT-0081 readback overlapped a pending descriptor");
          }
          this.pendingDescriptorCount = 1;
          this.maximumPendingDescriptorCount = Math.max(
            this.maximumPendingDescriptorCount,
            1,
          );
          const encoder = this.device.createCommandEncoder({
            label: `ace-opt-0081-${arm.id}-layer-${layer}-${tap}-readback`,
          });
          encoder.copyBufferToBuffer(
            target.guarded.buffer,
            0,
            this.correctnessReadback,
            OPT_0081_REPRESENTATIVE_READBACK_GUARD_BYTES,
            sourceBytes,
          );
          return encoder.finish();
        },
        onCommandBufferCompleted: () => {
          if (this.pendingDescriptorCount !== 1) {
            throw new Error("OPT-0081 readback descriptor was not pending");
          }
          this.pendingDescriptorCount = 0;
        },
      });
    } finally {
      this.pendingDescriptorCount = 0;
      this.activeLeaseCount -= 1;
    }
    signal.throwIfAborted();
    if (this.correctnessReadback.mapState !== "unmapped") {
      throw new Error("OPT-0081 correctness readback remained mapped");
    }
    this.requireDrainedEvent("snapshot map");
    this.snapshotMapCount += 1;
    await this.correctnessReadback.mapAsync(GPUMapMode.READ, 0, totalBytes);
    this.mappedRangeCount += 1;
    this.liveMapCount += 1;
    try {
      signal.throwIfAborted();
      const mapped = new Uint8Array(
        this.correctnessReadback.getMappedRange(0, totalBytes),
      );
      const prefix = new Uint32Array(
        mapped.buffer,
        mapped.byteOffset,
        OPT_0081_REPRESENTATIVE_READBACK_GUARD_BYTES / UINT32_BYTES,
      );
      const suffix = new Uint32Array(
        mapped.buffer,
        mapped.byteOffset +
          OPT_0081_REPRESENTATIVE_READBACK_GUARD_BYTES + sourceBytes,
        OPT_0081_REPRESENTATIVE_READBACK_GUARD_BYTES / UINT32_BYTES,
      );
      const adjacentBefore = new Uint32Array(
        mapped.buffer,
        mapped.byteOffset + OPT_0081_REPRESENTATIVE_READBACK_GUARD_BYTES,
        OPT_0081_REPRESENTATIVE_READBACK_GUARD_BYTES / UINT32_BYTES,
      );
      const adjacentAfter = new Uint32Array(
        mapped.buffer,
        mapped.byteOffset +
          2 * OPT_0081_REPRESENTATIVE_READBACK_GUARD_BYTES + byteLength,
        OPT_0081_REPRESENTATIVE_READBACK_GUARD_BYTES / UINT32_BYTES,
      );
      const bytes = mapped.slice(
        2 * OPT_0081_REPRESENTATIVE_READBACK_GUARD_BYTES,
        2 * OPT_0081_REPRESENTATIVE_READBACK_GUARD_BYTES + byteLength,
      );
      const snapshot = snapshotOpt0081RepresentativeWords(
        arm.id,
        layer,
        tap,
        target.storage,
        target.columns,
        bytes,
        prefix.every((word) =>
          word === OPT_0081_REPRESENTATIVE_READBACK_GUARD_U32
        ),
        suffix.every((word) =>
          word === OPT_0081_REPRESENTATIVE_READBACK_GUARD_U32
        ),
        adjacentBefore.every((word) =>
          word === OPT_0081_REPRESENTATIVE_READBACK_GUARD_U32
        ),
        adjacentAfter.every((word) =>
          word === OPT_0081_REPRESENTATIVE_READBACK_GUARD_U32
        ),
      );
      this.checkpointSnapshotCount += 1;
      this.maximumDetachedCheckpointBytes = Math.max(
        this.maximumDetachedCheckpointBytes,
        snapshot.byteLength,
      );
      return snapshot;
    } finally {
      this.correctnessReadback.unmap();
      this.unmappedRangeCount += 1;
      this.liveMapCount -= 1;
    }
  }
}

async function createOpt0081RepresentativeArmResources(
  device: GPUDevice,
  id: AceOpt0081RepresentativeArm,
  denseInputStorageProfile: AceDitDenseInputStorageProfile | undefined,
  plan: AceDitEvaluationPlan,
  model: AceDitResidentModel,
  selection: ReturnType<typeof resolveAceDitMixedGemmSelection>,
  snapshots: InputSnapshots,
  controls: ReturnType<typeof createAceDitGraphControlData>,
  signal: AbortSignal,
): Promise<AceOpt0081RepresentativeArmResources> {
  let arena: AceGpuArena | undefined;
  let runtime: AceCorrectnessDitRuntime | undefined;
  try {
    signal.throwIfAborted();
    const arenaPlan = planAceDitBackendArena(
      "reference-bf16",
      plan,
      ACE_TURBO_EIGHT_SAMPLER_SCHEDULE_PROFILE,
      denseInputStorageProfile,
    );
    const expectedArenaBytes = id === "A"
      ? OPT_0081_REPRESENTATIVE_CONTROL_ARENA_BYTES
      : OPT_0081_REPRESENTATIVE_CANDIDATE_ARENA_BYTES;
    if (
      arenaPlan.allocatedArenaBytes !== expectedArenaBytes ||
      (id === "A") !== (denseInputStorageProfile === undefined) ||
      arenaPlan.slots.some((slot) =>
        slot.byteLength > device.limits.maxBufferSize ||
        slot.byteLength > device.limits.maxStorageBufferBindingSize
      )
    ) {
      throw new Error(`OPT-0081 representative arm ${id} arena changed`);
    }
    arena = await AceGpuArena.create(
      device,
      arenaPlan.slots.map((slot) => ({
        label: `ace-opt-0081-${id}-arena-slot-${slot.index}`,
        byteLength: slot.byteLength,
      })),
    );
    signal.throwIfAborted();
    const bindings = bindArenaRoles(arena, arenaPlan);
    const graphBindings = createGraphBindings(
      bindings,
      ACE_TURBO_EIGHT_SAMPLER_SCHEDULE_PROFILE.evaluationCount,
    );
    await uploadInputsAndControls(
      device,
      "reference-bf16",
      graphBindings,
      snapshots,
      controls,
      signal,
    );
    runtime = AceCorrectnessDitRuntime.create(
      device,
      "reference-bf16",
      selection.gemmConfiguration,
      selection.denseGemmConfiguration,
      selection.attentionConfiguration,
      denseInputStorageProfile,
    );
    const compiled = await compileOpt0081RepresentativeArm(
      runtime,
      model,
      plan,
      graphBindings,
      signal,
    );
    const ownedArena = arena;
    const ownedRuntime = runtime;
    arena = undefined;
    runtime = undefined;
    return Object.freeze({
      id,
      denseInputStorageProfile,
      arenaPlan,
      arena: ownedArena,
      bindings: graphBindings,
      runtime: ownedRuntime,
      precompute: compiled.precompute,
      timedSlice: compiled.timedSlice,
    });
  } catch (error) {
    runtime?.destroy();
    arena?.destroy();
    throw error;
  }
}

async function createOpt0081RepresentativeArenaSnapshot(
  device: GPUDevice,
  arm: AceOpt0081RepresentativeArmResources,
): Promise<AceGpuArena> {
  return await AceGpuArena.create(
    device,
    arm.arena.bufferByteLengths.map((byteLength, index) => ({
      label: `ace-opt-0081-${arm.id}-correctness-snapshot-${index}`,
      byteLength,
    })),
  );
}

/** @internal Exact GPU-only arena snapshot/restore encoding seam. */
export function encodeOpt0081RepresentativeArenaCopy(
  device: GPUDevice,
  source: Pick<AceGpuArena, "buffers" | "bufferByteLengths" | "byteLength">,
  destination: Pick<
    AceGpuArena,
    "buffers" | "bufferByteLengths" | "byteLength"
  >,
  label: string,
): GPUCommandBuffer {
  if (
    !label || source.byteLength !== destination.byteLength ||
    source.buffers.length === 0 ||
    source.buffers.length !== destination.buffers.length ||
    source.bufferByteLengths.length !== source.buffers.length ||
    destination.bufferByteLengths.length !== destination.buffers.length
  ) {
    throw new Error("OPT-0081 correctness arena copy topology changed");
  }
  const encoder = device.createCommandEncoder({
    label: `ace-opt-0081-correctness-arena-${label}`,
  });
  for (let index = 0; index < source.buffers.length; index += 1) {
    const sourceBytes = source.bufferByteLengths[index];
    const destinationBytes = destination.bufferByteLengths[index];
    if (
      sourceBytes === undefined || destinationBytes !== sourceBytes ||
      !Number.isSafeInteger(sourceBytes) || sourceBytes <= 0 ||
      sourceBytes % 4 !== 0
    ) {
      throw new Error("OPT-0081 correctness arena slot topology changed");
    }
    encoder.copyBufferToBuffer(
      source.buffers[index]!,
      0,
      destination.buffers[index]!,
      0,
      sourceBytes,
    );
  }
  return encoder.finish({
    label: `ace-opt-0081-correctness-arena-${label}-command`,
  });
}

async function compileOpt0081RepresentativeArm(
  runtime: AceCorrectnessDitRuntime,
  model: AceDitResidentModel,
  plan: AceDitEvaluationPlan,
  bindings: AceDitGraphBindings,
  signal: AbortSignal,
  includePrecompute = true,
): Promise<Readonly<{
  precompute: readonly AceOpt0081RepresentativeEncodedQuantum[];
  timedSlice: readonly AceOpt0081RepresentativeEncodedQuantum[];
}>> {
  const quanta = createAceDitGraphQuantumPlan();
  const weights = model.weights;
  const requireQuantum = (index: number, expectedLabel: string) => {
    const quantum = quanta[index];
    if (quantum === undefined || quantum.label !== expectedLabel) {
      throw new Error(`OPT-0081 graph quantum ${expectedLabel} changed`);
    }
    return quantum;
  };
  signal.throwIfAborted();
  const precomputeDispatches: AceOpt0081RepresentativeDispatch[] = [];
  if (includePrecompute) {
    const condition = requireQuantum(0, "ace-dit-condition-projection");
    precomputeDispatches.push(asOpt0081RepresentativeDispatch(
      await runtime.createConditionProjectionDispatch(
      condition.label,
      ACE_TURBO_DIT_CONFIG,
      plan.batch,
      plan.conditionTokens,
      {
        input: bindings.conditionInput,
        weight: weights.conditionProjection.weight,
        bias: weights.conditionProjection.bias,
        output: bindings.projectedCondition,
      },
    )));
    for (let layer = 0; layer < LAYER_COUNT; layer += 1) {
      signal.throwIfAborted();
      const quantum = requireQuantum(
        1 + layer,
        `ace-dit-cross-cache-${layer}`,
      );
      const cache = bindings.crossCaches[layer]!;
      precomputeDispatches.push(asOpt0081RepresentativeDispatch(
        await runtime.createCrossCacheDispatch(
        quantum.label,
        ACE_TURBO_DIT_CONFIG,
        {
          batch: plan.batch,
          tokens: plan.tokens,
          conditionTokens: plan.conditionTokens,
          attentionMode: aceDitLayerAttentionMode(layer),
        },
        {
          projectedCondition: bindings.projectedCondition,
          weights: weights.crossCaches[layer]!,
          scratch: bindings.crossCacheScratch,
          key: cache.key,
          value: cache.value,
        },
      )));
    }
  }
  signal.throwIfAborted();
  const timestepQuantum = requireQuantum(25, "ace-dit-eval-0-timestep");
  const timestep = await runtime.createTimestepDispatch(
    timestepQuantum.label,
    ACE_TURBO_DIT_CONFIG,
    plan.batch,
    {
      timestep: bindings.controls.timesteps[0]!,
      relativeTimestep: bindings.controls.relativeTimestepZero,
      weights: weights.timestep,
      scratch: bindings.timestepScratch,
      embedding: bindings.timestepEmbedding,
      projection: bindings.timestepProjection,
    },
  );
  const inputQuantum = requireQuantum(26, "ace-dit-eval-0-input-projection");
  const input = await runtime.createInputProjectionDispatch(
    inputQuantum.label,
    ACE_TURBO_DIT_CONFIG,
    plan,
    {
      context: bindings.context,
      latent: bindings.latents[0],
      concatenated: bindings.concatenatedInput,
      weight: weights.inputProjection.weight,
      bias: weights.inputProjection.bias,
      output: bindings.hidden[0],
    },
  );
  const layers: AceDitCompositeDispatch<unknown>[] = [];
  for (let layer = 0; layer < 2; layer += 1) {
    signal.throwIfAborted();
    const quantum = requireQuantum(
      27 + layer,
      `ace-dit-eval-0-layer-${layer}`,
    );
    const attentionMode = aceDitLayerAttentionMode(layer);
    const layerBindings: AceDitLayerBindings = {
      input: bindings.hidden[layer % 2]!,
      output: bindings.hidden[(layer + 1) % 2]!,
      weights: weights.layers[layer]!,
      scratch: bindings.layerScratch,
      timestepProjection: bindings.timestepProjection,
      crossKey: bindings.crossCaches[layer]!.key,
      crossValue: bindings.crossCaches[layer]!.value,
      selfValidLengths: bindings.controls.selfValidLengths,
      crossValidLengths: bindings.controls.crossValidLengths,
      cosine: bindings.controls.cosine,
      sine: bindings.controls.sine,
    };
    layers.push(await runtime.createLayerDispatch(
      quantum.label,
      ACE_TURBO_DIT_CONFIG,
      {
        batch: plan.batch,
        tokens: plan.tokens,
        conditionTokens: plan.conditionTokens,
        attentionMode,
      },
      layerBindings,
    ));
  }
  signal.throwIfAborted();
  const precompute = flattenOpt0081RepresentativeDispatches(
    precomputeDispatches,
    0,
  );
  const timedSlice = flattenOpt0081RepresentativeDispatches(
    [timestep, input, ...layers],
    OPT_0081_REPRESENTATIVE_PRECOMPUTE_COMMANDS,
  );
  if (
    precompute.length !== (includePrecompute
      ? OPT_0081_REPRESENTATIVE_PRECOMPUTE_COMMANDS
      : 0) ||
    timestep.cooperativeSequence.quantumCount !== 1 ||
    input.cooperativeSequence.quantumCount !== 1 ||
    layers[0]?.cooperativeSequence.quantumCount !==
      OPT_0081_REPRESENTATIVE_LAYER_COMMANDS[0] ||
    layers[1]?.cooperativeSequence.quantumCount !==
      OPT_0081_REPRESENTATIVE_LAYER_COMMANDS[1] ||
    timedSlice.length !== OPT_0081_REPRESENTATIVE_TIMED_COMMANDS
  ) {
    throw new Error("OPT-0081 representative physical command split changed");
  }
  return Object.freeze({ precompute, timedSlice });
}

function asOpt0081RepresentativeDispatch(
  dispatch: AceOpt0081RepresentativeDispatchSource,
): AceOpt0081RepresentativeDispatch {
  if (dispatch.cooperativeSequence !== undefined) {
    return Object.freeze({
      label: dispatch.label,
      cooperativeSequence: dispatch.cooperativeSequence,
    });
  }
  if (
    dispatch.plan === undefined || dispatch.rangeCount === undefined ||
    dispatch.encodeRange === undefined
  ) {
    throw new Error(
      `OPT-0081 dispatch ${dispatch.label} has no cooperative sequence`,
    );
  }
  return Object.freeze({
    label: dispatch.label,
    cooperativeSequence: aceCompositeCooperativeSequence([
      dispatch as Parameters<typeof aceCompositeCooperativeSequence>[0][number],
    ]),
  });
}

function flattenOpt0081RepresentativeDispatches(
  dispatches: readonly AceOpt0081RepresentativeDispatch[],
  productionPhysicalStart: number,
): readonly AceOpt0081RepresentativeEncodedQuantum[] {
  const result: AceOpt0081RepresentativeEncodedQuantum[] = [];
  for (const dispatch of dispatches) {
    for (
      let sequenceQuantumIndex = 0;
      sequenceQuantumIndex < dispatch.cooperativeSequence.quantumCount;
      sequenceQuantumIndex += 1
    ) {
      const slicePhysicalIndex = result.length;
      result.push(Object.freeze({
        sequence: dispatch.cooperativeSequence,
        sequenceQuantumIndex,
        publicDescriptor: Object.freeze({
          slicePhysicalIndex,
          productionPhysicalIndex:
            productionPhysicalStart + slicePhysicalIndex,
          logicalLabel: dispatch.label,
          logicalSubquantumIndex: sequenceQuantumIndex,
          logicalSubquantumCount: dispatch.cooperativeSequence.quantumCount,
          descriptor:
            dispatch.cooperativeSequence.describeQuantum(sequenceQuantumIndex),
        }),
      }));
    }
  }
  return Object.freeze(result);
}

function requireOpt0081RepresentativeArmAgreement(
  control: AceOpt0081RepresentativeArmResources,
  candidate: AceOpt0081RepresentativeArmResources,
): void {
  if (
    control.id !== "A" || candidate.id !== "B" ||
    control.denseInputStorageProfile !== undefined ||
    candidate.denseInputStorageProfile !==
      ACE_OPT_0081_DIT_F16_DENSE_INPUT_STORAGE_PROFILE ||
    control.precompute.length !== candidate.precompute.length ||
    control.timedSlice.length !== candidate.timedSlice.length
  ) {
    throw new Error("OPT-0081 representative arm identities changed");
  }
  const pairs = [
    ...control.precompute.map((entry, index) =>
      [entry, candidate.precompute[index]] as const
    ),
    ...control.timedSlice.map((entry, index) =>
      [entry, candidate.timedSlice[index]] as const
    ),
  ];
  for (const [left, right] of pairs) {
    if (
      right === undefined ||
      !sameOpt0081RepresentativePhysicalQuantum(
        left.publicDescriptor,
        right.publicDescriptor,
      )
    ) {
      throw new Error(
        "OPT-0081 control/candidate physical descriptor order changed",
      );
    }
  }
}

function opt0081RepresentativeResidentBufferCount(
  reference: AceGpuTensorPhase,
  dense: AceGpuTensorPhase,
): number {
  const phaseShardCount = (phase: AceGpuTensorPhase): number =>
    new Set(Object.values(phase.packageManifest.tensors)
      .filter((tensor) => tensor.phase === "dit")
      .map((tensor) => tensor.shard)).size;
  const referenceCount = phaseShardCount(reference);
  const count = referenceCount + phaseShardCount(dense);
  if (referenceCount !== 2 || count !== 50) {
    throw new Error("OPT-0081 authenticated resident buffer inventory changed");
  }
  return count;
}

function requireOpt0081SetupEvidence(
  evidence: AceOpt0081RepresentativeSetupCleanupEvidence | undefined,
): boolean {
  if (evidence === undefined) return false;
  if (
    evidence.schema !== "ace-opt-0081-representative-setup-cleanup-v1" ||
    !Number.isSafeInteger(evidence.createdGraphBufferCount) ||
    evidence.createdGraphBufferCount <= 0 ||
    evidence.destroyedGraphBufferCount !== evidence.createdGraphBufferCount ||
    evidence.liveGraphBufferCount !== 0 || evidence.liveGraphByteCount !== 0 ||
    !Number.isSafeInteger(evidence.runtimeOwnerCount) ||
    evidence.runtimeOwnerCount <= 0 ||
    evidence.destroyedRuntimeOwnerCount !== evidence.runtimeOwnerCount ||
    evidence.residentModelDestroyed !== true ||
    evidence.mappedRangeCount !== 0 || evidence.unmappedRangeCount !== 0 ||
    evidence.liveMapCount !== 0 || evidence.pendingDescriptorCount !== 0 ||
    evidence.activeCallbackCount !== 0 || evidence.activeLeaseCount !== 0 ||
    !Number.isSafeInteger(evidence.armReleaseCount) ||
    evidence.armReleaseCount <= 0 || evidence.drainOrderViolationCount !== 0
  ) throw new Error("OPT-0081 setup cleanup evidence is invalid");
  return true;
}

function sameOpt0081RepresentativePhysicalQuantum(
  left: AceOpt0081RepresentativePhysicalQuantum,
  right: AceOpt0081RepresentativePhysicalQuantum,
): boolean {
  return left.slicePhysicalIndex === right.slicePhysicalIndex &&
    left.productionPhysicalIndex === right.productionPhysicalIndex &&
    left.logicalLabel === right.logicalLabel &&
    left.logicalSubquantumIndex === right.logicalSubquantumIndex &&
    left.logicalSubquantumCount === right.logicalSubquantumCount &&
    left.descriptor.id === right.descriptor.id &&
    left.descriptor.primitiveCount === right.descriptor.primitiveCount &&
    left.descriptor.scheduledMultiplyAdds ===
      right.descriptor.scheduledMultiplyAdds &&
    left.descriptor.members.length === right.descriptor.members.length &&
    left.descriptor.members.every((member, index) => {
      const other = right.descriptor.members[index];
      return other !== undefined && member.id === other.id &&
        member.label === other.label && member.rangeIndex === other.rangeIndex &&
        member.primitiveCount === other.primitiveCount &&
        member.scheduledMultiplyAdds === other.scheduledMultiplyAdds;
    });
}

function createOpt0081RepresentativeTopology(
  control: AceOpt0081RepresentativeArmResources,
  candidate: AceOpt0081RepresentativeArmResources,
): AceOpt0081RepresentativeTopology {
  const epochs = planAceDepth2Epoch4CompletionEpochs(
    OPT_0081_REPRESENTATIVE_TIMED_COMMANDS,
    OPT_0081_REPRESENTATIVE_PHASE_COUNTS,
  );
  if (
    epochs.length !== 7 ||
    epochs.some((epoch, index) =>
      epoch.completionEpochIndex !== index || epoch.phaseIndex !== 0 ||
      epoch.commandBufferCount !== 4 ||
      epoch.firstCommandBufferIndex !== index * 4 ||
      epoch.lastCommandBufferIndex !== index * 4 + 3
    ) ||
    control.runtime.attentionBackend !==
      "opt-0070-fixed32-quad-query32-full-self-production" ||
    candidate.runtime.attentionBackend !== control.runtime.attentionBackend ||
    ACE_OPT_0081_DIT_F16_DENSE_INPUT_ROLES.length !== 6 ||
    ACE_OPT_0081_REPRESENTATIVE_DENSE_TAPS.length !== 9 ||
    control.timedSlice[12]?.publicDescriptor.productionPhysicalIndex !== 37 ||
    control.timedSlice[13]?.publicDescriptor.productionPhysicalIndex !== 38 ||
    control.timedSlice[14]?.publicDescriptor.productionPhysicalIndex !== 39 ||
    control.timedSlice[15]?.publicDescriptor.productionPhysicalIndex !== 40
  ) {
    throw new Error("OPT-0081 completion epoch topology changed");
  }
  return Object.freeze({
    schema: "ace-opt-0081-representative-topology-v1",
    precomputeCommandBufferCountPerArm: 25,
    timedCommandBufferCount: 28,
    phaseCommandBufferCounts: OPT_0081_REPRESENTATIVE_PHASE_COUNTS,
    timestepCommandBufferCount: 1,
    inputProjectionCommandBufferCount: 1,
    layerCommandBufferCounts: OPT_0081_REPRESENTATIVE_LAYER_COMMANDS,
    completionEpochCount: 7,
    trueQueueDrainCount: 7,
    cooperativeIdleTurns: 6,
    maximumOutstandingCommandBuffers: 2,
    maximumPendingDescriptorCount: 2,
    attentionRuntimeProfile:
      ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
    layerAttentionRoutes: Object.freeze([
      Object.freeze({
        layer: 0 as const,
        self: "query8-sliding" as const,
        cross: "query8-cross" as const,
      }),
      Object.freeze({
        layer: 1 as const,
        self: "quad-query32-full" as const,
        cross: "query8-cross" as const,
      }),
    ] as const),
    producerStoreCount: 12,
    denseConsumerCount: 18,
    descriptorOrTapCommandCount: 0,
    timestampQueryCount: 0,
    measurementReadbackCount: 0,
    measurementMapCount: 0,
    fourthEpochProductionPhysicalIndices:
      Object.freeze([37, 38, 39, 40] as const),
    controlPhysicalQuanta: Object.freeze(
      control.timedSlice.map((entry) => entry.publicDescriptor),
    ),
    candidatePhysicalQuanta: Object.freeze(
      candidate.timedSlice.map((entry) => entry.publicDescriptor),
    ),
  });
}

function createOpt0081RepresentativeMemoryAccounting(
  model: AceDitResidentModel,
  control: AceOpt0081RepresentativeArmResources,
  candidate: AceOpt0081RepresentativeArmResources,
  correctnessReadbackBytes: number,
): AceOpt0081RepresentativeMemoryAccounting {
  const controlArenaBytes = control.arenaPlan.allocatedArenaBytes;
  const candidateArenaBytes = candidate.arenaPlan.allocatedArenaBytes;
  const arenaSavingBytes = controlArenaBytes - candidateArenaBytes;
  const simultaneousArenaBytes = controlArenaBytes + candidateArenaBytes;
  const selectedSlotIndices = ACE_OPT_0081_DIT_F16_DENSE_INPUT_ROLES.map(
    (role) => candidate.arenaPlan.roleToSlot[`layerScratch.${role}`],
  );
  const controlLargestArenaBindingBytes = Math.max(
    ...control.arenaPlan.slots.map((slot) => slot.byteLength),
  );
  const candidateLargestArenaBindingBytes = Math.max(
    ...candidate.arenaPlan.slots.map((slot) => slot.byteLength),
  );
  const correctnessGuardedBytes = checkedSum([
    Math.max(
      largestOpt0081RepresentativeTapBytes(control),
      largestOpt0081RepresentativeTapBytes(candidate),
    ),
    2 * OPT_0081_REPRESENTATIVE_READBACK_GUARD_BYTES,
  ], "OPT-0081 maximum guarded target bytes");
  if (
    controlArenaBytes !== OPT_0081_REPRESENTATIVE_CONTROL_ARENA_BYTES ||
    candidateArenaBytes !== OPT_0081_REPRESENTATIVE_CANDIDATE_ARENA_BYTES ||
    arenaSavingBytes !== OPT_0081_REPRESENTATIVE_ARENA_SAVING_BYTES ||
    simultaneousArenaBytes !==
      OPT_0081_REPRESENTATIVE_SIMULTANEOUS_ARENA_BYTES ||
    control.arenaPlan.roles.length !== 123 ||
    candidate.arenaPlan.roles.length !== 123 ||
    control.arenaPlan.slots.length !== 98 ||
    candidate.arenaPlan.slots.length !== 98 ||
    selectedSlotIndices.some((slot, index) =>
      slot !== ([61, 69, 72, 77, 81, 84] as const)[index]
    ) ||
    candidateLargestArenaBindingBytes !== controlLargestArenaBindingBytes
  ) {
    throw new Error("OPT-0081 representative arena reconciliation changed");
  }
  return Object.freeze({
    sharedResidentWeightBytes: model.residentBytes,
    controlArenaBytes: OPT_0081_REPRESENTATIVE_CONTROL_ARENA_BYTES,
    candidateArenaBytes: OPT_0081_REPRESENTATIVE_CANDIDATE_ARENA_BYTES,
    arenaSavingBytes: OPT_0081_REPRESENTATIVE_ARENA_SAVING_BYTES,
    simultaneousArenaBytes:
      OPT_0081_REPRESENTATIVE_SIMULTANEOUS_ARENA_BYTES,
    controlRoleCount: 123,
    candidateRoleCount: 123,
    controlSlotCount: 98,
    candidateSlotCount: 98,
    candidateSelectedDenseInputSlotIndices:
      Object.freeze([61, 69, 72, 77, 81, 84] as const),
    controlLargestArenaBindingBytes,
    candidateLargestArenaBindingBytes,
    largestArenaBindingIncreaseBytes: 0,
    timedCastOrCopyAuxiliaryBufferCount: 0,
    correctnessGuardedBytes,
    correctnessReadbackBytes,
    accountedGpuBytes: checkedSum([
      model.residentBytes,
      simultaneousArenaBytes,
      correctnessGuardedBytes,
      correctnessReadbackBytes,
    ], "OPT-0081 representative accounted GPU bytes"),
  });
}

function allOpt0081RepresentativeTaps(): readonly AceOpt0081RepresentativeTap[] {
  return Object.freeze([
    ...ACE_OPT_0081_DIT_F16_DENSE_INPUT_ROLES,
    ...ACE_OPT_0081_REPRESENTATIVE_DENSE_TAPS,
    "layerOutput" as const,
  ]);
}

function largestOpt0081RepresentativeTapBytes(
  arm: AceOpt0081RepresentativeArmResources,
): number {
  return Math.max(...allOpt0081RepresentativeTaps().map((tap) =>
    requireBindingSize(
      opt0081RepresentativeTapBinding(arm, 0, tap).binding,
      `OPT-0081 ${arm.id} ${tap}`,
    )
  ));
}

function opt0081RepresentativeTapBinding(
  arm: AceOpt0081RepresentativeArmResources,
  layer: AceOpt0081RepresentativeLayer,
  tap: AceOpt0081RepresentativeTap,
): Readonly<{
  binding: GPUBufferBinding;
  storage: "u16" | "u32";
  columns: number;
}> {
  const producer = (ACE_OPT_0081_DIT_F16_DENSE_INPUT_ROLES as readonly string[])
    .includes(tap);
  const dense = (ACE_OPT_0081_REPRESENTATIVE_DENSE_TAPS as readonly string[])
    .includes(tap);
  if (!producer && !dense && tap !== "layerOutput") {
    throw new TypeError(`Unknown OPT-0081 representative tap ${String(tap)}`);
  }
  const binding = tap === "layerOutput"
    ? arm.bindings.hidden[(layer + 1) % 2]!
    : arm.bindings.layerScratch[tap as keyof AceDitLayerScratch];
  const storage = producer && arm.id === "B" ? "u16" : "u32";
  return Object.freeze({
    binding,
    storage,
    columns: opt0081RepresentativeTapColumns(tap),
  });
}

function opt0081RepresentativeCheckpointBindings(
  base: AceDitGraphBindings,
  layer: AceOpt0081RepresentativeLayer,
  tap: AceOpt0081RepresentativeTap,
  guarded: GPUBufferBinding,
): AceDitGraphBindings {
  if (tap === "layerOutput") {
    const hidden: [GPUBufferBinding, GPUBufferBinding] = [
      base.hidden[0],
      base.hidden[1],
    ];
    hidden[(layer + 1) % 2] = guarded;
    return Object.freeze({
      ...base,
      hidden: Object.freeze(hidden) as readonly [
        GPUBufferBinding,
        GPUBufferBinding,
      ],
    });
  }
  return Object.freeze({
    ...base,
    layerScratch: Object.freeze({
      ...base.layerScratch,
      [tap]: guarded,
    }) as AceDitLayerScratch,
  });
}

function resetOpt0081RepresentativeGuardedTarget(
  queue: GPUQueue,
  target: AceOpt0081RepresentativeCheckpointTarget,
): void {
  const guard = new Uint32Array(
    OPT_0081_REPRESENTATIVE_READBACK_GUARD_BYTES / UINT32_BYTES,
  );
  guard.fill(OPT_0081_REPRESENTATIVE_READBACK_GUARD_U32);
  queue.writeBuffer(target.guarded.buffer, 0, guard);
  queue.writeBuffer(
    target.guarded.buffer,
    OPT_0081_REPRESENTATIVE_READBACK_GUARD_BYTES +
      target.guarded.bodyBytes,
    guard,
  );
}

function opt0081RepresentativeTapColumns(
  tap: AceOpt0081RepresentativeTap,
): number {
  switch (tap) {
    case "selfKeyFlat":
    case "selfValueFlat":
      return ACE_TURBO_DIT_CONFIG.keyValueHeads *
        ACE_TURBO_DIT_CONFIG.headDimension;
    case "gate":
    case "up":
    case "gatedActivation":
      return ACE_TURBO_DIT_CONFIG.intermediateSize;
    default:
      return ACE_TURBO_DIT_CONFIG.hiddenSize;
  }
}

function requireBindingSize(binding: GPUBufferBinding, label: string): number {
  const size = Number(binding.size);
  const offset = Number(binding.offset ?? 0);
  if (
    !Number.isSafeInteger(size) || size <= 0 || size % 4 !== 0 ||
    !Number.isSafeInteger(offset) || offset < 0 || offset % 4 !== 0 ||
    offset + size > binding.buffer.size
  ) {
    throw new RangeError(`${label} binding size is invalid`);
  }
  return size;
}

/** @internal Direct fake-queue coverage for the correctness-only checkpoint path. */
export function prefillOpt0081RepresentativeTap(
  queue: GPUQueue,
  binding: GPUBufferBinding,
  storage: "u16" | "u32",
): void {
  const byteLength = requireBindingSize(binding, "OPT-0081 prefill");
  const destinationOffset = Number(binding.offset ?? 0);
  if (storage === "u32") {
    const chunk = new Uint32Array(Math.min(byteLength / UINT32_BYTES, 262_144));
    chunk.fill(OPT_0081_REPRESENTATIVE_QNAN_U32);
    for (let offset = 0; offset < byteLength; offset += chunk.byteLength) {
      const bytes = Math.min(chunk.byteLength, byteLength - offset);
      queue.writeBuffer(
        binding.buffer,
        destinationOffset + offset,
        chunk,
        0,
        bytes / Uint32Array.BYTES_PER_ELEMENT,
      );
    }
    return;
  }
  const chunk = new Uint16Array(Math.min(byteLength / 2, 524_288));
  chunk.fill(OPT_0081_REPRESENTATIVE_QNAN_U16);
  for (let offset = 0; offset < byteLength; offset += chunk.byteLength) {
    const bytes = Math.min(chunk.byteLength, byteLength - offset);
    queue.writeBuffer(
      binding.buffer,
      destinationOffset + offset,
      chunk,
      0,
      bytes / Uint16Array.BYTES_PER_ELEMENT,
    );
  }
}

function snapshotOpt0081RepresentativeWords(
  arm: AceOpt0081RepresentativeArm,
  layer: AceOpt0081RepresentativeLayer,
  tap: AceOpt0081RepresentativeTap,
  storage: "u16" | "u32",
  columns: number,
  bytes: Uint8Array<ArrayBuffer>,
  readbackPrefixGuardIntact: boolean,
  readbackSuffixGuardIntact: boolean,
  adjacentBeforeGuardIntact: boolean,
  adjacentAfterGuardIntact: boolean,
): AceOpt0081RepresentativeCheckpointSnapshot {
  const elementBytes = storage === "u16" ? 2 : 4;
  const elementCount = bytes.byteLength / elementBytes;
  if (
    !Number.isSafeInteger(elementCount) ||
    elementCount !== 2_250 * columns
  ) {
    throw new Error(`OPT-0081 ${tap} checkpoint shape changed`);
  }
  let nonFiniteCount = 0;
  let qNaNPrefillCount = 0;
  let positiveZeroCount = 0;
  let negativeZeroCount = 0;
  if (storage === "u16") {
    const words = new Uint16Array(bytes.buffer, bytes.byteOffset, elementCount);
    for (const word of words) {
      if ((word & 0x7c00) === 0x7c00) nonFiniteCount += 1;
      if (word === OPT_0081_REPRESENTATIVE_QNAN_U16) qNaNPrefillCount += 1;
      if (word === 0) positiveZeroCount += 1;
      if (word === 0x8000) negativeZeroCount += 1;
    }
    return Object.freeze({
      schema: "ace-opt-0081-representative-checkpoint-v1",
      arm,
      layer,
      tap,
      storage,
      elementCount,
      byteLength: bytes.byteLength,
      bytes,
      words,
      nonFiniteCount,
      qNaNPrefillCount,
      positiveZeroCount,
      negativeZeroCount,
      completeQNaNOverwrite: qNaNPrefillCount === 0,
      firstValidWritten: words[0] !== OPT_0081_REPRESENTATIVE_QNAN_U16,
      lastValidWritten:
        words.at(-1) !== OPT_0081_REPRESENTATIVE_QNAN_U16,
      rows2240Through2249Written: words.slice(2_240 * columns).every((word) =>
        word !== OPT_0081_REPRESENTATIVE_QNAN_U16
      ),
      readbackPrefixGuardIntact,
      readbackSuffixGuardIntact,
      adjacentBeforeGuardIntact,
      adjacentAfterGuardIntact,
    });
  }
  const words = new Uint32Array(bytes.buffer, bytes.byteOffset, elementCount);
  for (const word of words) {
    if ((word & 0x7f80_0000) === 0x7f80_0000) nonFiniteCount += 1;
    if (word === OPT_0081_REPRESENTATIVE_QNAN_U32) qNaNPrefillCount += 1;
    if (word === 0) positiveZeroCount += 1;
    if (word === 0x8000_0000) negativeZeroCount += 1;
  }
  return Object.freeze({
    schema: "ace-opt-0081-representative-checkpoint-v1",
    arm,
    layer,
    tap,
    storage,
    elementCount,
    byteLength: bytes.byteLength,
    bytes,
    words,
    nonFiniteCount,
    qNaNPrefillCount,
    positiveZeroCount,
    negativeZeroCount,
    completeQNaNOverwrite: qNaNPrefillCount === 0,
    firstValidWritten: words[0] !== OPT_0081_REPRESENTATIVE_QNAN_U32,
    lastValidWritten: words.at(-1) !== OPT_0081_REPRESENTATIVE_QNAN_U32,
    rows2240Through2249Written: words.slice(2_240 * columns).every((word) =>
      word !== OPT_0081_REPRESENTATIVE_QNAN_U32
    ),
    readbackPrefixGuardIntact,
    readbackSuffixGuardIntact,
    adjacentBeforeGuardIntact,
    adjacentAfterGuardIntact,
  });
}

function createOpt0081RepresentativeTimedResult(
  arm: AceOpt0081RepresentativeArm,
  capture: AceOpt0081RepresentativeRunCapture,
  topology: AceOpt0081RepresentativeTopology,
): AceOpt0081RepresentativeTimedSliceResult {
  const scheduling = capture.scheduling;
  const expectedEpochs = planAceDepth2Epoch4CompletionEpochs(
    OPT_0081_REPRESENTATIVE_TIMED_COMMANDS,
    OPT_0081_REPRESENTATIVE_PHASE_COUNTS,
  );
  if (
    capture.fenceTimings.length !== 28 ||
    capture.epochTimings.length !== 7 ||
    capture.fenceTimings.some((timing, index) =>
      timing.commandBufferIndex !== index ||
      timing.completionEpochIndex !== Math.floor(index / 4) ||
      timing.trueQueueDrain !== (index % 4 === 3)
    ) ||
    capture.epochTimings.some((timing, index) => {
      const expected = expectedEpochs[index];
      return expected === undefined ||
        timing.completionEpochIndex !== expected.completionEpochIndex ||
        timing.phaseIndex !== expected.phaseIndex ||
        timing.firstCommandBufferIndex !== expected.firstCommandBufferIndex ||
        timing.lastCommandBufferIndex !== expected.lastCommandBufferIndex ||
        timing.commandBufferCount !== expected.commandBufferCount;
    }) ||
    scheduling.commandBuffersSubmitted !== 28 ||
    scheduling.completionFenceRequestedCount !== 28 ||
    scheduling.completionFenceSettledCount !== 28 ||
    scheduling.completionFenceRejectedCount !== 0 ||
    scheduling.trueQueueDrainCount !== 7 ||
    scheduling.queueDrains !== 7 ||
    scheduling.completionEpochCount !== 7 ||
    scheduling.requestedCooperativeIdleMs !== 6 ||
    scheduling.cooperativeIdleMs !== 6 ||
    scheduling.cooperativeIdleTurns !== 6 ||
    scheduling.maximumOutstandingCommandBuffers !== 2 ||
    capture.maximumPendingDescriptorCount !== 2 ||
    capture.pendingDescriptorCountAfterRun !== 0 ||
    capture.drainedAtPerformanceMs - capture.startedAtPerformanceMs !==
      capture.wallMs
  ) {
    throw new Error("OPT-0081 representative timed topology changed");
  }
  const completionEpochWallsMs = Object.freeze(
    capture.epochTimings.map((timing) => timing.submitThroughTrueDrainMs),
  );
  if (completionEpochWallsMs.length !== 7) {
    throw new Error("OPT-0081 completion epoch timing inventory changed");
  }
  return Object.freeze({
    schema: "ace-opt-0081-representative-timed-slice-v1",
    arm,
    startedAtPerformanceMs: capture.startedAtPerformanceMs,
    drainedAtPerformanceMs: capture.drainedAtPerformanceMs,
    wallMs: capture.wallMs,
    completionEpochWallsMs: completionEpochWallsMs as unknown as readonly [
      number,
      number,
      number,
      number,
      number,
      number,
      number,
    ],
    submitThroughCompletionFenceMs: Object.freeze(
      capture.fenceTimings.map((timing) =>
        timing.submitThroughCompletionFenceMs
      ),
    ),
    commandBuffersSubmitted: 28,
    completionFenceRequestedCount: 28,
    completionFenceSettledCount: 28,
    completionFenceRejectedCount: 0,
    trueQueueDrainCount: 7,
    completionEpochCount: 7,
    requestedCooperativeIdleMs: 6,
    cooperativeIdleTurns: 6,
    maximumOutstandingCommandBuffers: 2,
    maximumPendingDescriptorCount: 2,
    pendingDescriptorCountAfterRun: 0,
    topology,
  });
}

function destroyOpt0081RepresentativeArm(
  arm: AceOpt0081RepresentativeArmResources | undefined,
): void {
  if (arm === undefined) return;
  arm.runtime.destroy();
  arm.arena.destroy();
}

export function planAceDitGpuBackendMemory(
  modelProfile: AceModelProfileId,
  shape: AceDitEvaluationShape,
  residentWeightBytes: number,
  gemmBackend: AceDitGemmBackend = "portable",
  physicalQuantaPerCommandBuffer: AceDitPhysicalQuantaPerCommandBuffer =
    ACE_DIT_PHYSICAL_QUANTA_PER_COMMAND_BUFFER,
  captureEvaluationLatents = false,
  captureOpt0062AttentionIdentity = false,
  samplerScheduleProfile: Readonly<AceDitSamplerScheduleProfile> =
    ACE_TURBO_EIGHT_SAMPLER_SCHEDULE_PROFILE,
  captureEvaluation0Velocity = false,
  ditDenseInputStorageProfile?: AceDitDenseInputStorageProfile,
): AceDitGpuBackendMemoryPlan {
  requireNonNegativeSafeInteger(
    residentWeightBytes,
    "ACE DiT resident weight bytes",
  );
  const profile = requireResolvedAceDitSamplerScheduleProfile(
    samplerScheduleProfile,
  );
  const denseInputStorageProfile = requireAceDitDenseInputStorageProfile(
    ditDenseInputStorageProfile,
  );
  if (
    denseInputStorageProfile !== undefined &&
    (gemmBackend !== "mixed-opt-0009" ||
      physicalQuantaPerCommandBuffer !==
        ACE_DIT_PHYSICAL_QUANTA_PER_COMMAND_BUFFER ||
      captureEvaluationLatents || captureOpt0062AttentionIdentity ||
      captureEvaluation0Velocity ||
      profile.id !== ACE_TURBO_EIGHT_SAMPLER_SCHEDULE_PROFILE.id ||
      profile.evaluationCount !== DENOISING_EVALUATIONS ||
      profile.role !== "production-default")
  ) {
    throw new Error(
      "OPT-0081 dense-input memory planning requires exact mixed-opt-0009 " +
        "singleton production-eight accounting without legacy captures",
    );
  }
  const graph = planAceDitGraphMemory(
    modelProfile,
    shape,
    profile,
    denseInputStorageProfile,
  );
  const plan = planAceDitEvaluation(ACE_TURBO_DIT_CONFIG, shape);
  const arena = planAceDitBackendArena(
    modelProfile,
    shape,
    profile,
    denseInputStorageProfile,
  );
  if (arena.logicalRoleBytes !== graph.unaliasedGraphBytesExcludingWeights) {
    throw new Error(
      "ACE DiT backend binding inventory diverged from graph byte accounting: " +
        `${arena.logicalRoleBytes} != ${graph.unaliasedGraphBytesExcludingWeights}`,
    );
  }
  if (arena.allocatedArenaBytes < graph.minimumGraphBytesExcludingWeights) {
    throw new Error(
      "ACE DiT backend liveness slots undercut the graph high-water mark",
    );
  }
  const rawReadbackBytes = activationBytes(modelProfile, plan.latentElements);
  const readbackBufferBytes = align(rawReadbackBytes, STORAGE_ALIGNMENT);
  const evaluationReadbackBytes = captureEvaluationLatents
    ? checkedProduct(
        [readbackBufferBytes, profile.evaluationCount],
        "ACE DiT evaluation readback bytes",
      )
    : 0;
  const evaluation0VelocityReadbackBytes = captureEvaluation0Velocity
    ? readbackBufferBytes
    : 0;
  const opt0062AttentionIdentityCounterBytes =
    captureOpt0062AttentionIdentity
      ? ACE_OPT_0062_IDENTITY_COUNTER_BYTES
      : 0;
  const opt0062AttentionIdentityReadbackBytes =
    opt0062AttentionIdentityCounterBytes;
  const detachedFinalLatentBytes = checkedProduct(
    [plan.latentElements, FLOAT32_BYTES],
    "ACE DiT detached latent bytes",
  );
  const detachedEvaluationLatentBytes = captureEvaluationLatents
    ? checkedProduct(
        [detachedFinalLatentBytes, profile.evaluationCount],
        "ACE DiT detached evaluation latent bytes",
      )
    : 0;
  const detachedEvaluation0VelocityBytes = captureEvaluation0Velocity
    ? detachedFinalLatentBytes
    : 0;
  const detachedOpt0062AttentionIdentityBytes =
    opt0062AttentionIdentityReadbackBytes;
  const callerInputSnapshotBytes = checkedProduct([
    plan.batch * plan.conditionTokens * ACE_TURBO_DIT_CONFIG.conditionInputSize +
      plan.contextElements +
      plan.latentElements,
    FLOAT32_BYTES,
  ], "ACE DiT caller input snapshot bytes");
  const controlSnapshotBytes = checkedSum([
    graph.ropeBytes,
    graph.attentionControlBytes,
    graph.timestepInputBytes,
  ], "ACE DiT control snapshot bytes");
  const maximumEncodingStagingBytes = modelProfile === "raw-fp16"
    ? Math.max(
        graph.conditionInputBytes,
        graph.contextBytes,
        graph.latentBytes,
      )
    : 0;
  const boundedFactoryCpuBytes = checkedSum([
    callerInputSnapshotBytes,
    controlSnapshotBytes,
    maximumEncodingStagingBytes,
  ], "ACE DiT bounded factory CPU bytes");
  const physicalGraphQuantumCount = planAceDitPhysicalCommandBufferCount(
    plan,
    gemmBackend,
    profile.evaluationCount,
  );
  const graphCommandBufferCount = planAceDitPhysicalQuantumBatches(
    physicalGraphQuantumCount,
    physicalQuantaPerCommandBuffer,
  ).length;
  return Object.freeze({
    modelProfile,
    ...(denseInputStorageProfile === undefined
      ? {}
      : { denseInputStorageProfile }),
    gemmBackend,
    graph,
    residentWeightBytes,
    arena,
    readbackBufferBytes,
    evaluationReadbackBytes,
    evaluation0VelocityReadbackBytes,
    opt0062AttentionIdentityCounterBytes,
    opt0062AttentionIdentityReadbackBytes,
    accountedGpuBytes: checkedSum([
      residentWeightBytes,
      arena.allocatedArenaBytes,
      readbackBufferBytes,
      evaluationReadbackBytes,
      evaluation0VelocityReadbackBytes,
      opt0062AttentionIdentityCounterBytes,
      opt0062AttentionIdentityReadbackBytes,
    ], "ACE DiT accounted GPU bytes"),
    callerInputSnapshotBytes,
    controlSnapshotBytes,
    maximumEncodingStagingBytes,
    detachedFinalLatentBytes,
    detachedEvaluationLatentBytes,
    detachedEvaluation0VelocityBytes,
    detachedOpt0062AttentionIdentityBytes,
    boundedCpuBytes: Math.max(
      boundedFactoryCpuBytes,
      detachedFinalLatentBytes + detachedEvaluationLatentBytes +
        detachedEvaluation0VelocityBytes +
        detachedOpt0062AttentionIdentityBytes,
    ),
    samplerScheduleProfileId: profile.id,
    evaluationCount: profile.evaluationCount,
    logicalGraphQuantumCount: aceDitGraphQuantumCount(profile),
    physicalGraphQuantumCount,
    physicalQuantaPerCommandBuffer,
    graphCommandBufferCount,
    readbackCommandBufferCount: 1 as const,
    commandBufferCount: graphCommandBufferCount + 1,
  });
}

/** Exact command count emitted by the current conservative DiT composition. */
export function planAceDitPhysicalCommandBufferCount(
  shape: AceDitEvaluationShape,
  gemmBackend: AceDitGemmBackend = "portable",
  evaluationCount: AceDitSamplerEvaluationCount = DENOISING_EVALUATIONS,
): number {
  const plan = planAceDitEvaluation(ACE_TURBO_DIT_CONFIG, shape);
  const config = ACE_TURBO_DIT_CONFIG;
  if (
    gemmBackend !== "portable" &&
    gemmBackend !== "fixed32-subgroups" &&
    gemmBackend !== "mixed-opt-0009" &&
    gemmBackend !== "mixed-opt-0037-k4" &&
    gemmBackend !== "mixed-opt-0056-selective"
  ) {
    throw new TypeError(
      `Unknown ACE DiT GEMM backend ${String(gemmBackend)}`,
    );
  }
  const gemm = (rows: number, inner: number, columns: number): AceGemmShape =>
    Object.freeze({ rows, inner, columns });
  const planGemm = (shape: AceGemmShape): AceCooperativeGemmPlan =>
    gemmBackend === "portable"
      ? planAceTiledGemm(shape)
      : planAceSubgroupGemm(shape);
  const planDenseGemm = (shape: AceGemmShape): AceCooperativeGemmPlan =>
    gemmBackend === "mixed-opt-0037-k4" ||
      gemmBackend === "mixed-opt-0056-selective"
      ? planAceOpt0032DenseK4Partials(shape)
      : gemmBackend === "mixed-opt-0009"
        ? planAceOpt0009DenseGemm(shape)
        : planGemm(shape);
  const ranges = (shape: AceGemmShape): number =>
    planGemm(shape).outputRangeCount;
  const composite = (shapes: readonly AceGemmShape[]): number =>
    planAceCompositeCooperativeQuantumCount(shapes.map(planGemm));
  const conditionRows = plan.batch * plan.conditionTokens;
  const conditionProjection = ranges(gemm(
    conditionRows,
    config.conditionInputSize,
    config.hiddenSize,
  ));
  const keyValueWidth = config.keyValueHeads * config.headDimension;
  const crossCache = composite([
    gemm(conditionRows, config.hiddenSize, keyValueWidth),
    gemm(conditionRows, config.hiddenSize, keyValueWidth),
  ]);
  const timestep = composite([
    gemm(plan.batch, config.timestepInputSize, config.hiddenSize),
    gemm(plan.batch, config.timestepInputSize, config.hiddenSize),
    gemm(plan.batch, config.hiddenSize, config.hiddenSize),
    gemm(plan.batch, config.hiddenSize, config.hiddenSize),
    gemm(
      plan.batch,
      config.hiddenSize,
      LAYER_MODULATION_GROUPS * config.hiddenSize,
    ),
    gemm(
      plan.batch,
      config.hiddenSize,
      LAYER_MODULATION_GROUPS * config.hiddenSize,
    ),
  ]);
  const layerShapes = [
    gemm(plan.rows, config.hiddenSize, config.hiddenSize),
    gemm(plan.rows, config.hiddenSize, keyValueWidth),
    gemm(plan.rows, config.hiddenSize, keyValueWidth),
    gemm(plan.rows, config.hiddenSize, config.hiddenSize),
    gemm(plan.rows, config.hiddenSize, config.hiddenSize),
    gemm(plan.rows, config.hiddenSize, config.hiddenSize),
    gemm(plan.rows, config.hiddenSize, config.intermediateSize),
    gemm(plan.rows, config.hiddenSize, config.intermediateSize),
    gemm(plan.rows, config.intermediateSize, config.hiddenSize),
  ];
  const layerDensePlans = layerShapes.map(planDenseGemm);
  const maybePlanFixed32Attention = (
    shape: Parameters<typeof planAceFixed32TiledFullAttention>[0],
  ) => {
    if (
      (gemmBackend !== "mixed-opt-0009" &&
        gemmBackend !== "mixed-opt-0037-k4" &&
        gemmBackend !== "mixed-opt-0056-selective") ||
      !isAceFixed32TiledFullAttentionShape(shape)
    ) {
      return undefined;
    }
    return planAceFixed32TiledFullAttention(shape);
  };
  const fixed32CrossAttention = maybePlanFixed32Attention({
    batch: plan.batch,
    queryHeads: config.queryHeads,
    keyValueHeads: config.keyValueHeads,
    queryTokens: plan.tokens,
    keyValueTokens: plan.conditionTokens,
    headDimension: config.headDimension,
    mode: "full",
  });
  const allLayerCommands = checkedSum(
    Array.from({ length: LAYER_COUNT }, (_, layerIndex) => {
      if (
        gemmBackend !== "mixed-opt-0009" &&
        gemmBackend !== "mixed-opt-0037-k4" &&
        gemmBackend !== "mixed-opt-0056-selective"
      ) {
        return planAceCompositeCooperativeQuantumCount(layerDensePlans);
      }
      const attentionMode = aceDitLayerAttentionMode(layerIndex);
      const fixed32SelfAttention = maybePlanFixed32Attention({
        batch: plan.batch,
        queryHeads: config.queryHeads,
        keyValueHeads: config.keyValueHeads,
        queryTokens: plan.tokens,
        keyValueTokens: plan.tokens,
        headDimension: config.headDimension,
        mode: attentionMode,
        ...(attentionMode === "sliding"
          ? { slidingRadius: config.slidingRadius }
          : {}),
      });
      const cooperativePlans = [
        ...layerDensePlans.slice(0, 3),
        ...(fixed32SelfAttention === undefined
          ? []
          : [fixed32SelfAttention]),
        ...layerDensePlans.slice(3, 5),
        ...(fixed32CrossAttention === undefined
          ? []
          : [fixed32CrossAttention]),
        ...layerDensePlans.slice(5),
      ];
      return planAceCompositeCooperativeQuantumCount(cooperativePlans);
    }),
    "ACE DiT layer physical commands",
  );
  const perEvaluation = checkedSum([
    timestep,
    1,
    allLayerCommands,
    1,
    1,
  ], "ACE DiT evaluation physical commands");
  return checkedSum([
    conditionProjection,
    LAYER_COUNT * crossCache,
    evaluationCount * perEvaluation,
  ], "ACE DiT graph physical commands");
}

export function planAceDitBackendArena(
  modelProfile: AceModelProfileId,
  shape: AceDitEvaluationShape,
  samplerScheduleProfile: Readonly<AceDitSamplerScheduleProfile> =
    ACE_TURBO_EIGHT_SAMPLER_SCHEDULE_PROFILE,
  ditDenseInputStorageProfile?: AceDitDenseInputStorageProfile,
): AceDitBackendArenaPlan {
  const graph = planAceDitEvaluation(ACE_TURBO_DIT_CONFIG, shape);
  const denseInputStorageProfile = requireAceDitDenseInputStorageProfile(
    ditDenseInputStorageProfile,
  );
  if (
    denseInputStorageProfile !== undefined &&
    (modelProfile !== "reference-bf16" || graph.batch !== 1 ||
      graph.latentFrames !== 4_500 || graph.tokens !== 2_250 ||
      graph.conditionTokens !== 98)
  ) {
    throw new Error(
      "OPT-0081 dense-input arena requires exact reference-bf16 M2250/C98",
    );
  }
  const profile = requireResolvedAceDitSamplerScheduleProfile(
    samplerScheduleProfile,
  );
  const roles = createArenaRoles(
    modelProfile,
    graph,
    profile.evaluationCount,
    denseInputStorageProfile,
  );
  const slots: Array<{
    byteLength: number;
    roles: AceDitBackendArenaRole[];
  }> = [];
  const roleToSlot: Record<string, number> = Object.create(null) as Record<
    string,
    number
  >;
  for (const role of roles) {
    let selected = -1;
    let selectedGrowth = Number.POSITIVE_INFINITY;
    for (let index = 0; index < slots.length; index += 1) {
      const slot = slots[index]!;
      if (slot.roles.some((other) => lifetimesOverlap(role, other))) continue;
      const growth = Math.max(slot.byteLength, align(role.byteLength, STORAGE_ALIGNMENT)) -
        slot.byteLength;
      if (growth < selectedGrowth) {
        selected = index;
        selectedGrowth = growth;
      }
    }
    if (selected < 0) {
      selected = slots.length;
      slots.push({
        byteLength: align(role.byteLength, STORAGE_ALIGNMENT),
        roles: [],
      });
    }
    const slot = slots[selected]!;
    slot.byteLength = Math.max(
      slot.byteLength,
      align(role.byteLength, STORAGE_ALIGNMENT),
    );
    slot.roles.push(role);
    roleToSlot[role.key] = selected;
  }
  const logicalRoleBytes = checkedSum(
    roles.map((role) => role.byteLength),
    "ACE DiT logical arena role bytes",
  );
  const allocatedArenaBytes = checkedSum(
    slots.map((slot) => slot.byteLength),
    "ACE DiT allocated arena bytes",
  );
  const maximumUnalignedSlotBytes = checkedSum(
    slots.map((slot) => Math.max(...slot.roles.map((role) => role.byteLength))),
    "ACE DiT unaligned arena slot bytes",
  );
  return Object.freeze({
    ...(denseInputStorageProfile === undefined
      ? {}
      : { denseInputStorageProfile }),
    roles,
    slots: Object.freeze(slots.map((slot, index) => Object.freeze({
      index,
      byteLength: slot.byteLength,
      roles: Object.freeze(slot.roles.map((role) => role.key)),
    }))),
    roleToSlot: Object.freeze(roleToSlot),
    logicalRoleBytes,
    allocatedArenaBytes,
    alignmentOverheadBytes: allocatedArenaBytes - maximumUnalignedSlotBytes,
    lifetimeReuseSavingsBytes: logicalRoleBytes - maximumUnalignedSlotBytes,
  });
}

async function createConcreteResources(
  options: AceDitGpuBackendOptions,
): Promise<AceDitPreparedGpuResources> {
  // The phase transfers before inspecting any other option.
  let model: AceDitResidentModel | undefined;
  let arena: AceGpuArena | undefined;
  let readback: GPUBuffer | undefined;
  let evaluationReadbacks: GPUBuffer[] | undefined;
  let evaluation0VelocityReadback: GPUBuffer | undefined;
  let opt0062AttentionIdentityCounters: GPUBuffer | undefined;
  let opt0062AttentionIdentityReadback: GPUBuffer | undefined;
  let graph: AceDitGraphOwner | undefined;
  let lifetimeListener: (() => void) | undefined;
  const factoryLifetime = new AbortController();
  try {
    if (options.ownedDitDenseWeights === undefined) {
      options.ownedDitWeights.destroy();
      throw new Error("ACE production DiT requires an authenticated dense phase");
    }
    model = AceDitResidentModel.takeMixed(
      options.ownedDitWeights,
      options.ownedDitDenseWeights,
      options.executionProfile.modelProfile,
      options.ditDenseRuntimeProfile ??
        ACE_OPT_0009_DIT_DENSE_RUNTIME_PROFILE,
    );
    const samplerScheduleProfile = requireResolvedAceDitSamplerScheduleProfile(
      options.samplerScheduleProfile ??
        ACE_TURBO_EIGHT_SAMPLER_SCHEDULE_PROFILE,
    );
    const denseInputStorageProfile = requireAceDitDenseInputStorageProfile(
      options.ditDenseInputStorageProfile,
    );
    const captureSamplerScheduleEvidence =
      options.captureSamplerScheduleEvidence === true;
    const ditSubmissionPolicy = options.ditSubmissionPolicy;
    if (
      ditSubmissionPolicy !== undefined &&
      ditSubmissionPolicy !== "depth1-epoch1" &&
      ditSubmissionPolicy !== "depth2-phase-epoch4"
    ) {
      throw new RangeError("Unknown ACE DiT submission policy");
    }
    if (
      ditSubmissionPolicy === "depth2-phase-epoch4" &&
      (options.executionProfile.id !== ACE_REFERENCE_SUBGROUP_PROFILE.id ||
        options.ditDenseInputStorageProfile !== undefined ||
        samplerScheduleProfile.role !== "production-default" ||
        (options.ditDenseRuntimeProfile ??
          ACE_OPT_0009_DIT_DENSE_RUNTIME_PROFILE) !==
            ACE_OPT_0009_DIT_DENSE_RUNTIME_PROFILE ||
        options.ditAttentionRuntimeProfile !==
          ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE ||
        (options.physicalQuantaPerCommandBuffer ??
          ACE_DIT_PHYSICAL_QUANTA_PER_COMMAND_BUFFER) !== 1 ||
        options.onCommandProfile !== undefined ||
        options.onSchedulingProfile !== undefined ||
        options.onOpt0067EvaluationProfile !== undefined ||
        options.onOpt0080EvaluationProfile !== undefined ||
        options.onOpt0080FullProfile !== undefined ||
        options.opt0067EvaluationLimit !== undefined ||
        options.opt0080SchedulingProfile !== undefined ||
        options.opt0080FullSchedulingProfile !== undefined ||
        options.onOpt0080CommandBufferCompleted !== undefined ||
        options.captureEvaluationLatents !== undefined ||
        options.captureOpt0062AttentionIdentity !== undefined ||
        captureSamplerScheduleEvidence)
    ) {
      throw new Error(
        "Depth-two DiT production scheduling requires the exact current product tuple without diagnostic seams",
      );
    }
    if (
      samplerScheduleProfile.role === "diagnostic-only" &&
      !captureSamplerScheduleEvidence
    ) {
      throw new Error(
        "Diagnostic sampler schedules require their bounded evidence capture",
      );
    }
    if (
      captureSamplerScheduleEvidence &&
      (options.onCommandProfile !== undefined ||
        options.ditDenseInputStorageProfile !== undefined ||
        options.onSchedulingProfile !== undefined ||
        options.onOpt0067EvaluationProfile !== undefined ||
        options.onOpt0080EvaluationProfile !== undefined ||
        options.onOpt0080FullProfile !== undefined ||
        options.opt0080SchedulingProfile !== undefined ||
        options.opt0080FullSchedulingProfile !== undefined ||
        options.opt0067EvaluationLimit !== undefined ||
        options.captureEvaluationLatents !== undefined ||
        options.captureOpt0062AttentionIdentity !== undefined ||
        (options.physicalQuantaPerCommandBuffer ??
          ACE_DIT_PHYSICAL_QUANTA_PER_COMMAND_BUFFER) !== 1 ||
        (options.ditDenseRuntimeProfile ??
          ACE_OPT_0009_DIT_DENSE_RUNTIME_PROFILE) !==
          ACE_OPT_0009_DIT_DENSE_RUNTIME_PROFILE ||
        (options.ditAttentionRuntimeProfile ??
          ACE_DIT_QUERY8_ATTENTION_RUNTIME_PROFILE) !==
          ACE_DIT_QUERY8_ATTENTION_RUNTIME_PROFILE)
    ) {
      throw new Error(
        "Sampler-schedule evidence requires exact OPT-0009 dense/query8 with no legacy diagnostic seam",
      );
    }
    if ([
      options.onCommandProfile,
      options.onSchedulingProfile,
      options.onOpt0067EvaluationProfile,
      options.onOpt0080EvaluationProfile,
      options.onOpt0080FullProfile,
    ].filter((capture) => capture !== undefined).length > 1) {
      throw new Error(
        "ACE DiT command-attribution captures are mutually exclusive",
      );
    }
    if (
      (options.opt0067EvaluationLimit === 1) !==
        (options.onOpt0067EvaluationProfile !== undefined ||
          options.onOpt0080EvaluationProfile !== undefined) ||
      (options.opt0080SchedulingProfile !== undefined) !==
        (options.onOpt0080EvaluationProfile !== undefined) ||
      (options.opt0080FullSchedulingProfile !== undefined) !==
        (options.onOpt0080FullProfile !== undefined) ||
      (options.onOpt0080CommandBufferCompleted !== undefined &&
        (options.opt0080SchedulingProfile === undefined &&
          options.opt0080FullSchedulingProfile === undefined ||
          typeof options.onOpt0080CommandBufferCompleted !== "function")) ||
      (options.opt0080SchedulingProfile !== undefined &&
        options.opt0080SchedulingProfile !== "depth1-epoch1" &&
        options.opt0080SchedulingProfile !== "opt-0080-depth2-epoch4") ||
      (options.opt0080FullSchedulingProfile !== undefined &&
        options.opt0080FullSchedulingProfile !== "depth1-epoch1" &&
        options.opt0080FullSchedulingProfile !==
          "opt-0080-depth2-epoch4") ||
      (options.opt0067EvaluationLimit === 1 &&
        ((options.physicalQuantaPerCommandBuffer ??
            ACE_DIT_PHYSICAL_QUANTA_PER_COMMAND_BUFFER) !== 1 ||
          options.captureEvaluationLatents === true)) ||
      (options.opt0080SchedulingProfile !== undefined &&
        (options.ditDenseInputStorageProfile !== undefined ||
          options.captureOpt0062AttentionIdentity === true ||
          options.shape.batch !== 1 || options.shape.latentFrames !== 4_500 ||
          options.shape.conditionTokens !== 98 ||
          (options.ditDenseRuntimeProfile ??
            ACE_OPT_0009_DIT_DENSE_RUNTIME_PROFILE) !==
              ACE_OPT_0009_DIT_DENSE_RUNTIME_PROFILE ||
          options.ditAttentionRuntimeProfile !==
            ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE)) ||
      (options.opt0080FullSchedulingProfile !== undefined &&
        (options.ditDenseInputStorageProfile !== undefined ||
          options.opt0067EvaluationLimit !== undefined ||
          options.captureOpt0062AttentionIdentity === true ||
          (options.physicalQuantaPerCommandBuffer ??
            ACE_DIT_PHYSICAL_QUANTA_PER_COMMAND_BUFFER) !== 1 ||
          options.shape.batch !== 1 || options.shape.latentFrames !== 4_500 ||
          options.shape.conditionTokens !== 98 ||
          (options.ditDenseRuntimeProfile ??
            ACE_OPT_0009_DIT_DENSE_RUNTIME_PROFILE) !==
              ACE_OPT_0009_DIT_DENSE_RUNTIME_PROFILE ||
          options.ditAttentionRuntimeProfile !==
            ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE))
    ) {
      throw new Error(
        "OPT-0067/0080 require one paired evaluation-0 profile, singleton physical commands, and ordinary result readback",
      );
    }
    if (
      options.captureEvaluationLatents === true &&
      (options.ditDenseInputStorageProfile !== undefined ||
        (options.physicalQuantaPerCommandBuffer ??
          ACE_DIT_PHYSICAL_QUANTA_PER_COMMAND_BUFFER) !== 1)
    ) {
      throw new Error(
        "OPT-0056 sampler snapshots require one physical quantum per command buffer",
      );
    }
    if (
      options.captureOpt0062AttentionIdentity === true &&
      (options.ditDenseInputStorageProfile !== undefined ||
        (options.physicalQuantaPerCommandBuffer ??
          ACE_DIT_PHYSICAL_QUANTA_PER_COMMAND_BUFFER) !== 1 ||
        options.ditAttentionRuntimeProfile !==
          ACE_OPT_0062_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE ||
        (options.captureEvaluationLatents !== true &&
          options.opt0067EvaluationLimit !== 1))
    ) {
      throw new Error(
        "OPT-0062 actual-layer identity requires quad attention, an authorized result capture, and one physical quantum per command buffer",
      );
    }
    const shape = planAceDitEvaluation(ACE_TURBO_DIT_CONFIG, options.shape);
    const gemmSelection = resolveAceDitMixedGemmSelection(
      options.executionProfile,
      options.subgroupMinSize,
      options.subgroupMaxSize,
      options.ditDenseRuntimeProfile ??
        ACE_OPT_0009_DIT_DENSE_RUNTIME_PROFILE,
      options.ditAttentionRuntimeProfile ??
        ACE_DIT_QUERY8_ATTENTION_RUNTIME_PROFILE,
      shape.tokens,
      shape.conditionTokens,
    );
    if (
      denseInputStorageProfile !== undefined &&
      (options.executionProfile.id !== ACE_REFERENCE_SUBGROUP_PROFILE.id ||
        model.modelProfile !== "reference-bf16" ||
        !options.device.features.has("shader-f16") ||
        options.subgroupMinSize !== 32 || options.subgroupMaxSize !== 32 ||
        shape.batch !== 1 || shape.latentFrames !== 4_500 ||
        shape.tokens !== 2_250 || shape.conditionTokens !== 98 ||
        gemmSelection.backend !== "mixed-opt-0009" ||
        gemmSelection.denseRuntimeProfile !==
          ACE_OPT_0009_DIT_DENSE_RUNTIME_PROFILE ||
        gemmSelection.attentionRuntimeProfile !==
          ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE ||
        samplerScheduleProfile.role !== "production-default" ||
        ditSubmissionPolicy !== undefined ||
        options.physicalQuantaPerCommandBuffer !== undefined ||
        options.onCommandProfile !== undefined ||
        options.onSchedulingProfile !== undefined ||
        options.onOpt0067EvaluationProfile !== undefined ||
        options.opt0067EvaluationLimit !== undefined ||
        options.onOpt0080EvaluationProfile !== undefined ||
        options.opt0080SchedulingProfile !== undefined ||
        options.onOpt0080FullProfile !== undefined ||
        options.opt0080FullSchedulingProfile !== undefined ||
        options.onOpt0080CommandBufferCompleted !== undefined ||
        options.captureEvaluationLatents !== undefined ||
        options.captureOpt0062AttentionIdentity !== undefined ||
        options.captureSamplerScheduleEvidence !== undefined)
    ) {
      throw new Error(
        "OPT-0081 dense-input storage requires its exact isolated M2250/C98 " +
          "OPT-0009/OPT-0070 graph tuple without production scheduling or legacy captures",
      );
    }
    if (
      ditSubmissionPolicy === "depth2-phase-epoch4" &&
      (gemmSelection.backend !== "mixed-opt-0009" ||
        gemmSelection.denseRuntimeProfile !==
          ACE_OPT_0009_DIT_DENSE_RUNTIME_PROFILE ||
        gemmSelection.attentionRuntimeProfile !==
          ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE)
    ) {
      throw new Error(
        "Depth-two DiT production scheduling did not resolve the authorized kernel tuple",
      );
    }
    if (model.modelProfile !== gemmSelection.modelProfile) {
      throw new Error("ACE DiT execution and resident model profiles diverged");
    }
    void options.device.lost.then((info) => {
      if (!factoryLifetime.signal.aborted) {
        factoryLifetime.abort(new AceDitBackendDeviceLostError(info));
      }
    }).catch(() => undefined);
    if (options.signal !== undefined) {
      lifetimeListener = () => {
        if (!factoryLifetime.signal.aborted) {
          factoryLifetime.abort(options.signal!.reason);
        }
      };
      if (options.signal.aborted) lifetimeListener();
      else options.signal.addEventListener("abort", lifetimeListener, { once: true });
    }
    const signal = factoryLifetime.signal;
    signal.throwIfAborted();
    const snapshots = snapshotInputs(shape, options.inputs);
    const controls = createAceDitGraphControlData(
      shape,
      options.dcwConfiguration,
      samplerScheduleProfile,
    );
    const captureTrajectory = options.captureEvaluationLatents === true ||
      captureSamplerScheduleEvidence;
    const memory = planAceDitGpuBackendMemory(
      gemmSelection.modelProfile,
      shape,
      model.residentBytes,
      gemmSelection.backend,
      options.physicalQuantaPerCommandBuffer ??
        ACE_DIT_PHYSICAL_QUANTA_PER_COMMAND_BUFFER,
      captureTrajectory,
      options.captureOpt0062AttentionIdentity === true,
      samplerScheduleProfile,
      captureSamplerScheduleEvidence,
      denseInputStorageProfile,
    );
    validateDeviceLimits(options.device, memory);
    arena = await AceGpuArena.create(
      options.device,
      memory.arena.slots.map((slot) => ({
        label: `ace-dit-arena-slot-${slot.index}`,
        byteLength: slot.byteLength,
      })),
    );
    signal.throwIfAborted();
    const createdReadbacks = await createAceScopedBuffers(
      options.device,
      [
        {
          label: "ace-dit-final-latent-readback",
          size: memory.readbackBufferBytes,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        },
        ...(captureTrajectory
          ? Array.from(
              { length: samplerScheduleProfile.evaluationCount },
              (_, evaluation) => ({
              label: captureSamplerScheduleEvidence
                ? `ace-sampler-schedule-evaluation-${evaluation}-latent-readback`
                : `ace-opt-0056-evaluation-${evaluation}-latent-readback`,
              size: memory.readbackBufferBytes,
              usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
            }),
            )
          : []),
        ...(captureSamplerScheduleEvidence
          ? [{
              label: "ace-sampler-schedule-evaluation-0-velocity-readback",
              size: memory.readbackBufferBytes,
              usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
            }]
          : []),
      ],
      "ACE DiT final/evaluation latent readback allocation",
    );
    readback = createdReadbacks[0]!;
    evaluationReadbacks = captureTrajectory
      ? createdReadbacks.slice(
          1,
          1 + samplerScheduleProfile.evaluationCount,
        )
      : undefined;
    evaluation0VelocityReadback = captureSamplerScheduleEvidence
      ? createdReadbacks[1 + samplerScheduleProfile.evaluationCount]
      : undefined;
    if (
      evaluationReadbacks !== undefined &&
      evaluationReadbacks.length !== samplerScheduleProfile.evaluationCount
    ) {
      throw new Error("ACE evaluation readback allocation is incomplete");
    }
    if (
      captureSamplerScheduleEvidence &&
      evaluation0VelocityReadback === undefined
    ) {
      throw new Error("ACE evaluation-0 velocity readback allocation is incomplete");
    }
    if (options.captureOpt0062AttentionIdentity === true) {
      const identityBuffers = await createAceScopedBuffers(
        options.device,
        [
          {
            label: "ace-opt-0062-attention-identity-counters",
            size: ACE_OPT_0062_IDENTITY_COUNTER_BYTES,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
            mappedAtCreation: true,
          },
          {
            label: "ace-opt-0062-attention-identity-readback",
            size: ACE_OPT_0062_IDENTITY_COUNTER_BYTES,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
          },
        ],
        "OPT-0062 attention identity allocation",
      );
      opt0062AttentionIdentityCounters = identityBuffers[0];
      opt0062AttentionIdentityReadback = identityBuffers[1];
      if (
        opt0062AttentionIdentityCounters === undefined ||
        opt0062AttentionIdentityReadback === undefined
      ) {
        throw new Error("OPT-0062 attention identity allocation is incomplete");
      }
      new Uint8Array(
        opt0062AttentionIdentityCounters.getMappedRange(),
      ).fill(0);
      opt0062AttentionIdentityCounters.unmap();
    }
    const bindings = bindArenaRoles(arena, memory.arena);
    const graphBindings = createGraphBindings(
      bindings,
      samplerScheduleProfile.evaluationCount,
      opt0062AttentionIdentityCounters,
    );
    await uploadInputsAndControls(
      options.device,
      gemmSelection.modelProfile,
      graphBindings,
      snapshots,
      controls,
      signal,
    );
    signal.throwIfAborted();
    const consumedModel = model;
    model = undefined;
    graph = await AceDitGraphOwner.create(
      options.device,
      consumedModel,
      shape,
      options.dcwConfiguration,
      graphBindings,
      {
        signal,
        ...(options.onCommandProfile === undefined &&
            options.onSchedulingProfile === undefined &&
            options.onOpt0067EvaluationProfile === undefined &&
            options.onOpt0080EvaluationProfile === undefined &&
            options.onOpt0080FullProfile === undefined
          ? {}
          : { capturePhysicalCommandDescriptors: true }),
        gemmConfiguration: gemmSelection.gemmConfiguration,
        denseGemmConfiguration: gemmSelection.denseGemmConfiguration,
        attentionConfiguration: gemmSelection.attentionConfiguration,
        ...(denseInputStorageProfile === undefined
          ? {}
          : { ditDenseInputStorageProfile: denseInputStorageProfile }),
        samplerScheduleProfile,
        onProgress: (progress: AceDitGraphCompilationProgress) => {
          options.onProgress?.({
            stage: "compile",
            compiledQuanta: progress.compiledQuanta,
            totalQuanta: progress.totalQuanta,
          });
        },
      },
    );
    signal.throwIfAborted();
    const ownedArena = arena;
    const ownedReadback = readback;
    const ownedEvaluationReadbacks = evaluationReadbacks;
    const ownedEvaluation0VelocityReadback = evaluation0VelocityReadback;
    const ownedOpt0062AttentionIdentityCounters =
      opt0062AttentionIdentityCounters;
    const ownedOpt0062AttentionIdentityReadback =
      opt0062AttentionIdentityReadback;
    const ownedGraph = graph;
    arena = undefined;
    readback = undefined;
    evaluationReadbacks = undefined;
    evaluation0VelocityReadback = undefined;
    opt0062AttentionIdentityCounters = undefined;
    opt0062AttentionIdentityReadback = undefined;
    graph = undefined;
    let destroyed = false;
    return Object.freeze({
      device: options.device,
      modelProfile: gemmSelection.modelProfile,
      ...(denseInputStorageProfile === undefined
        ? {}
        : { denseInputStorageProfile }),
      gemmBackend: gemmSelection.backend,
      shape,
      graph: ownedGraph,
      readback: ownedReadback,
      samplerScheduleProfile,
      ...(ownedEvaluationReadbacks === undefined
        ? {}
        : { evaluationReadbacks: Object.freeze(ownedEvaluationReadbacks) }),
      ...(ownedEvaluation0VelocityReadback === undefined
        ? {}
        : {
            evaluation0VelocityReadback:
              ownedEvaluation0VelocityReadback,
          }),
      ...(ownedOpt0062AttentionIdentityCounters === undefined
        ? {}
        : {
            opt0062AttentionIdentityCounters:
              ownedOpt0062AttentionIdentityCounters,
            opt0062AttentionIdentityReadback:
              ownedOpt0062AttentionIdentityReadback!,
          }),
      memory,
      ...(options.signal === undefined ? {} : { signal: options.signal }),
      ...(ditSubmissionPolicy === undefined
        ? {}
        : { ditSubmissionPolicy }),
      ...(options.onProgress === undefined
        ? {}
        : { onProgress: options.onProgress }),
      ...(options.onCommandProfile === undefined
        ? {}
        : { onCommandProfile: options.onCommandProfile }),
      ...(options.onOpt0067EvaluationProfile === undefined
        ? {}
        : {
            onOpt0067EvaluationProfile:
              options.onOpt0067EvaluationProfile,
            opt0067EvaluationLimit: 1 as const,
          }),
      ...(options.onOpt0080EvaluationProfile === undefined
        ? {}
        : {
            onOpt0080EvaluationProfile:
              options.onOpt0080EvaluationProfile,
            opt0080SchedulingProfile: options.opt0080SchedulingProfile!,
            opt0067EvaluationLimit: 1 as const,
          }),
      ...(options.onOpt0080FullProfile === undefined
        ? {}
        : {
            onOpt0080FullProfile: options.onOpt0080FullProfile,
            opt0080FullSchedulingProfile:
              options.opt0080FullSchedulingProfile!,
          }),
      ...(options.onOpt0080CommandBufferCompleted === undefined
        ? {}
        : {
            onOpt0080CommandBufferCompleted:
              options.onOpt0080CommandBufferCompleted,
          }),
      ...(options.physicalQuantaPerCommandBuffer === undefined
        ? {}
        : {
            physicalQuantaPerCommandBuffer:
              options.physicalQuantaPerCommandBuffer,
          }),
      ...(options.onSchedulingProfile === undefined
        ? {}
        : { onSchedulingProfile: options.onSchedulingProfile }),
      destroy(): void {
        if (destroyed) return;
        destroyed = true;
        ownedReadback.destroy();
        ownedEvaluationReadbacks?.forEach((buffer) => buffer.destroy());
        ownedEvaluation0VelocityReadback?.destroy();
        ownedOpt0062AttentionIdentityCounters?.destroy();
        ownedOpt0062AttentionIdentityReadback?.destroy();
        ownedArena.destroy();
      },
    });
  } catch (error) {
    await graph?.destroy();
    model?.destroy();
    readback?.destroy();
    evaluationReadbacks?.forEach((buffer) => buffer.destroy());
    evaluation0VelocityReadback?.destroy();
    opt0062AttentionIdentityCounters?.destroy();
    opt0062AttentionIdentityReadback?.destroy();
    arena?.destroy();
    if (
      factoryLifetime.signal.aborted &&
      factoryLifetime.signal.reason instanceof AceDitBackendDeviceLostError
    ) {
      throw factoryLifetime.signal.reason;
    }
    throw error;
  } finally {
    if (options.signal !== undefined && lifetimeListener !== undefined) {
      options.signal.removeEventListener("abort", lifetimeListener);
    }
  }
}

interface InputSnapshots {
  readonly condition: Float32Array<ArrayBuffer>;
  readonly context: Float32Array<ArrayBuffer>;
  readonly initialLatent: Float32Array<ArrayBuffer>;
}

function snapshotInputs(
  shape: AceDitEvaluationPlan,
  inputs: AceDitBackendInputs,
): InputSnapshots {
  const expected = Object.freeze({
    condition:
      shape.batch * shape.conditionTokens * ACE_TURBO_DIT_CONFIG.conditionInputSize,
    context: shape.contextElements,
    initialLatent: shape.latentElements,
  });
  const snapshot = (
    values: Float32Array,
    length: number,
    label: string,
  ): Float32Array<ArrayBuffer> => {
    if (
      !(values instanceof Float32Array) ||
      !(values.buffer instanceof ArrayBuffer) ||
      values.length !== length
    ) {
      throw new RangeError(
        `${label} must be an owned-memory Float32Array with exactly ` +
          `${length} logical values`,
      );
    }
    const owned = Float32Array.from(values);
    for (let index = 0; index < owned.length; index += 1) {
      if (!Number.isFinite(owned[index])) {
        throw new RangeError(`${label}[${index}] is not finite`);
      }
    }
    return owned;
  };
  return Object.freeze({
    condition: snapshot(inputs.condition, expected.condition, "ACE DiT condition"),
    context: snapshot(inputs.context, expected.context, "ACE DiT context"),
    initialLatent: snapshot(
      inputs.initialLatent,
      expected.initialLatent,
      "ACE DiT initial latent",
    ),
  });
}

function validateDeviceLimits(
  device: GPUDevice,
  memory: AceDitGpuBackendMemoryPlan,
): void {
  for (const slot of memory.arena.slots) {
    if (
      slot.byteLength > device.limits.maxBufferSize ||
      slot.byteLength > device.limits.maxStorageBufferBindingSize
    ) {
      throw new RangeError(
        `ACE DiT arena slot ${slot.index} exceeds this device's storage limits`,
      );
    }
  }
  if (memory.readbackBufferBytes > device.limits.maxBufferSize) {
    throw new RangeError("ACE DiT readback exceeds this device's buffer limit");
  }
}

function bindArenaRoles(
  arena: AceGpuArena,
  plan: AceDitBackendArenaPlan,
): ReadonlyMap<string, GPUBufferBinding> {
  const bindings = new Map<string, GPUBufferBinding>();
  for (const role of plan.roles) {
    const slot = plan.roleToSlot[role.key];
    if (slot === undefined) {
      throw new Error(`ACE DiT arena omitted role ${role.key}`);
    }
    bindings.set(role.key, arena.binding(arena.slice(
      role.label,
      slot,
      0,
      role.byteLength,
    )));
  }
  return bindings;
}

function createGraphBindings(
  bindings: ReadonlyMap<string, GPUBufferBinding>,
  evaluationCount: AceDitSamplerEvaluationCount,
  opt0062AttentionIdentityCounters?: GPUBuffer,
): AceDitGraphBindings {
  const take = (key: string): GPUBufferBinding => {
    const binding = bindings.get(key);
    if (binding === undefined) throw new Error(`ACE DiT arena is missing ${key}`);
    return binding;
  };
  const object = <Keys extends readonly string[]>(
    prefix: string,
    keys: Keys,
  ): { readonly [Key in Keys[number]]: GPUBufferBinding } =>
    Object.freeze(Object.fromEntries(
      keys.map((key) => [key, take(`${prefix}.${key}`)]),
    )) as { readonly [Key in Keys[number]]: GPUBufferBinding };

  return Object.freeze({
    conditionInput: take("conditionInput"),
    projectedCondition: take("projectedCondition"),
    crossCaches: Object.freeze(Array.from({ length: LAYER_COUNT }, (_, layer) =>
      Object.freeze({
        key: take(`crossCache.${layer}.key`),
        value: take(`crossCache.${layer}.value`),
      })
    )),
    crossCacheScratch: object(
      "crossCacheScratch",
      CROSS_CACHE_SCRATCH_KEYS,
    ),
    context: take("context"),
    latents: Object.freeze([
      take("latent.0"),
      take("latent.1"),
      take("latent.2"),
    ]) as unknown as readonly [
      GPUBufferBinding,
      GPUBufferBinding,
      GPUBufferBinding,
    ],
    concatenatedInput: take("concatenatedInput"),
    hidden: Object.freeze([
      take("hidden.0"),
      take("hidden.1"),
    ]) as unknown as readonly [GPUBufferBinding, GPUBufferBinding],
    timestepScratch: object("timestepScratch", TIMESTEP_SCRATCH_KEYS),
    timestepEmbedding: take("timestepEmbedding"),
    timestepProjection: take("timestepProjection"),
    layerScratch: object("layerScratch", LAYER_SCRATCH_KEYS),
    outputScratch: object("outputScratch", OUTPUT_SCRATCH_KEYS),
    velocity: take("velocity"),
    predictedCleanLatent: take("predictedCleanLatent"),
    controls: Object.freeze({
      selfValidLengths: take("control.selfValidLengths"),
      crossValidLengths: take("control.crossValidLengths"),
      cosine: take("control.cosine"),
      sine: take("control.sine"),
      timesteps: Object.freeze(Array.from(
        { length: evaluationCount },
        (_, evaluation) => take(`control.timestep.${evaluation}`),
      )),
      relativeTimestepZero: take("control.relativeTimestepZero"),
    }),
    ...(opt0062AttentionIdentityCounters === undefined
      ? {}
      : {
          opt0062AttentionIdentityCounters: {
            buffer: opt0062AttentionIdentityCounters,
            offset: 0,
            size: ACE_OPT_0062_IDENTITY_COUNTER_BYTES,
          },
        }),
  });
}

async function uploadInputsAndControls(
  device: GPUDevice,
  modelProfile: AceModelProfileId,
  bindings: AceDitGraphBindings,
  inputs: InputSnapshots,
  controls: ReturnType<typeof createAceDitGraphControlData>,
  signal: AbortSignal,
): Promise<void> {
  signal.throwIfAborted();
  device.pushErrorScope("internal");
  device.pushErrorScope("out-of-memory");
  device.pushErrorScope("validation");
  let scopesOpen = true;
  try {
    writeActivation(device.queue, bindings.conditionInput, inputs.condition, modelProfile);
    writeActivation(device.queue, bindings.context, inputs.context, modelProfile);
    writeActivation(
      device.queue,
      bindings.latents[0],
      inputs.initialLatent,
      modelProfile,
    );
    writeExact(device.queue, bindings.controls.selfValidLengths, controls.selfValidLengths);
    writeExact(device.queue, bindings.controls.crossValidLengths, controls.crossValidLengths);
    writeExact(device.queue, bindings.controls.cosine, controls.cosine);
    writeExact(device.queue, bindings.controls.sine, controls.sine);
    if (bindings.controls.timesteps.length !== controls.timesteps.length) {
      throw new Error("ACE DiT control upload schedule inventory changed");
    }
    for (
      let evaluation = 0;
      evaluation < controls.timesteps.length;
      evaluation += 1
    ) {
      writeExact(
        device.queue,
        bindings.controls.timesteps[evaluation]!,
        controls.timesteps[evaluation]!,
      );
    }
    writeExact(
      device.queue,
      bindings.controls.relativeTimestepZero,
      controls.relativeTimestepZero,
    );
    const error = await collectGpuErrorScopes(device, "ACE DiT input upload");
    scopesOpen = false;
    if (error !== undefined) throw error;
    signal.throwIfAborted();
  } catch (error) {
    if (scopesOpen) await discardGpuErrorScopes(device);
    throw error;
  }
}

function writeActivation(
  queue: GPUQueue,
  binding: GPUBufferBinding,
  values: Float32Array<ArrayBuffer>,
  profile: AceModelProfileId,
): void {
  if (profile === "reference-bf16") {
    writeExact(queue, binding, values);
    return;
  }
  if (profile !== "raw-fp16") {
    throw new TypeError(`Unknown ACE DiT model profile ${String(profile)}`);
  }
  const encoded = new Uint16Array(values.length);
  for (let index = 0; index < values.length; index += 1) {
    const bits = numberToFloat16Bits(values[index]!);
    if ((bits & 0x7c00) === 0x7c00) {
      throw new RangeError(`ACE DiT activation ${index} overflows FP16`);
    }
    encoded[index] = bits;
  }
  writeExact(queue, binding, encoded);
}

function writeExact(
  queue: GPUQueue,
  binding: GPUBufferBinding,
  values: ArrayBufferView<ArrayBufferLike>,
): void {
  const offset = binding.offset ?? 0;
  const size = binding.size ?? binding.buffer.size - offset;
  if (values.byteLength !== size || values.byteLength % 4 !== 0) {
    throw new RangeError(
      `ACE DiT immutable upload has ${values.byteLength} bytes; binding exposes ${size}`,
    );
  }
  const source: GPUAllowSharedBufferSource = values.buffer instanceof ArrayBuffer
    ? values as ArrayBufferView<ArrayBuffer>
    : Uint8Array.from(new Uint8Array(
        values.buffer,
        values.byteOffset,
        values.byteLength,
      ));
  queue.writeBuffer(binding.buffer, offset, source);
}

function encodeReadbackCopy(
  device: GPUDevice,
  source: GPUBufferBinding,
  readback: GPUBuffer,
  bytes: number,
): GPUCommandBuffer {
  const offset = source.offset ?? 0;
  const available = source.size ?? source.buffer.size - offset;
  if (
    bytes <= 0 ||
    bytes % 4 !== 0 ||
    available < bytes ||
    readback.size < bytes
  ) {
    throw new RangeError("ACE DiT final latent readback binding is invalid");
  }
  const encoder = device.createCommandEncoder({
    label: "ace-dit-final-latent-readback-encoder",
  });
  encoder.copyBufferToBuffer(source.buffer, offset, readback, 0, bytes);
  return encoder.finish({ label: "ace-dit-final-latent-readback-command" });
}

async function mapDetachedFinalLatent(
  readback: GPUBuffer,
  modelProfile: AceModelProfileId,
  elements: number,
  signal: AbortSignal,
): Promise<Float32Array<ArrayBuffer>> {
  const bytes = activationBytes(modelProfile, elements);
  signal.throwIfAborted();
  await readback.mapAsync(GPUMapMode.READ, 0, bytes);
  try {
    signal.throwIfAborted();
    const mapped = readback.getMappedRange(0, bytes);
    if (modelProfile === "reference-bf16") {
      return Float32Array.from(new Float32Array(mapped, 0, elements));
    }
    if (modelProfile !== "raw-fp16") {
      throw new TypeError(`Unknown ACE DiT model profile ${String(modelProfile)}`);
    }
    return Float32Array.from(
      new Uint16Array(mapped, 0, elements),
      float16BitsToNumber,
    );
  } finally {
    readback.unmap();
  }
}

async function mapOpt0062AttentionIdentity(
  readback: GPUBuffer,
  signal: AbortSignal,
): Promise<AceOpt0062AttentionIdentityResult> {
  signal.throwIfAborted();
  await readback.mapAsync(
    GPUMapMode.READ,
    0,
    ACE_OPT_0062_IDENTITY_COUNTER_BYTES,
  );
  let words: Uint32Array<ArrayBuffer>;
  try {
    signal.throwIfAborted();
    words = Uint32Array.from(new Uint32Array(
      readback.getMappedRange(0, ACE_OPT_0062_IDENTITY_COUNTER_BYTES),
    ));
  } finally {
    readback.unmap();
  }
  const strideWords = ACE_OPT_0062_IDENTITY_COUNTER_STRIDE_BYTES /
    Uint32Array.BYTES_PER_ELEMENT;
  if (strideWords < ACE_OPT_0062_IDENTITY_COUNTER_WORDS_PER_ROUTE) {
    throw new Error("OPT-0062 identity counter stride changed");
  }
  const routes = Object.freeze(Array.from(
    { length: ACE_OPT_0062_EXPECTED_QUAD_QUERY_ROUTES },
    (_, routeIndex): AceOpt0062AttentionIdentityRoute => {
      const evaluation = Math.floor(routeIndex / 12);
      const layer = 1 + 2 * (routeIndex % 12);
      const offset = routeIndex * strideWords;
      return Object.freeze({
        routeIndex,
        label: `ace-dit-eval-${evaluation}-layer-${layer}-self-full-attention`,
        comparedElements: words[offset] ?? 0,
        mismatchCount: words[offset + 1] ?? 0,
        query8NonFiniteCount: words[offset + 2] ?? 0,
        quadNonFiniteCount: words[offset + 3] ?? 0,
        query8CanaryCount: words[offset + 4] ?? 0,
        quadCanaryCount: words[offset + 5] ?? 0,
      });
    },
  ));
  const sum = (
    select: (route: AceOpt0062AttentionIdentityRoute) => number,
  ): number => routes.reduce((total, route) => total + select(route), 0);
  return Object.freeze({
    schema: "ace-opt-0062-actual-layer-raw-u32-identity-v1",
    routeCount: ACE_OPT_0062_EXPECTED_QUAD_QUERY_ROUTES,
    outputElementsPerRoute: ACE_OPT_0062_IDENTITY_OUTPUT_ELEMENTS,
    totalComparedElements: sum(({ comparedElements }) => comparedElements),
    totalMismatchCount: sum(({ mismatchCount }) => mismatchCount),
    totalNonFiniteCount: sum(({ query8NonFiniteCount, quadNonFiniteCount }) =>
      query8NonFiniteCount + quadNonFiniteCount
    ),
    totalCanaryCount: sum(({ query8CanaryCount, quadCanaryCount }) =>
      query8CanaryCount + quadCanaryCount
    ),
    copyCount: 1,
    extraCommandBufferCount: 0,
    extraQueueDrainCount: 0,
    routes,
  });
}

function requireExactFinalEvaluationSnapshot(
  evaluationLatents: readonly Float32Array<ArrayBuffer>[],
  finalLatent: Float32Array<ArrayBuffer>,
  evaluationCount: AceDitSamplerEvaluationCount,
): void {
  if (
    evaluationLatents.length !== evaluationCount ||
    evaluationLatents.some((latent) =>
      !(latent instanceof Float32Array) ||
      !(latent.buffer instanceof ArrayBuffer) ||
      latent.length !== finalLatent.length
    )
  ) {
    throw new Error("ACE detached sampler snapshot inventory changed");
  }
  const finalEvaluation = evaluationLatents.at(-1)!;
  const expectedWords = new Uint32Array(
    finalLatent.buffer,
    finalLatent.byteOffset,
    finalLatent.length,
  );
  const actualWords = new Uint32Array(
    finalEvaluation.buffer,
    finalEvaluation.byteOffset,
    finalEvaluation.length,
  );
  for (let index = 0; index < expectedWords.length; index += 1) {
    if (actualWords[index] !== expectedWords[index]) {
      throw new Error(
        `ACE final sampler snapshot diverged at U32 word ${index}`,
      );
    }
  }
}

function createArenaRoles(
  modelProfile: AceModelProfileId,
  plan: AceDitEvaluationPlan,
  evaluationCount: AceDitSamplerEvaluationCount,
  ditDenseInputStorageProfile?: AceDitDenseInputStorageProfile,
): readonly AceDitBackendArenaRole[] {
  const roles: AceDitBackendArenaRole[] = [];
  const activation = (elements: number): number =>
    activationBytes(modelProfile, elements);
  const interval = (firstQuantum: number, lastQuantum = firstQuantum) =>
    Object.freeze([{ firstQuantum, lastQuantum }]);
  const repeated = (firstOffset: number, lastOffset = firstOffset) =>
    Object.freeze(Array.from({ length: evaluationCount }, (_, evaluation) => ({
      firstQuantum: STATIC_QUANTA + evaluation * EVALUATION_QUANTA + firstOffset,
      lastQuantum: STATIC_QUANTA + evaluation * EVALUATION_QUANTA + lastOffset,
    })));
  const graphQuantumCount = STATIC_QUANTA +
    evaluationCount * EVALUATION_QUANTA;
  const all = interval(0, graphQuantumCount - 1);
  const add = (
    key: string,
    bytes: number,
    lifetimes: readonly Readonly<{
      firstQuantum: number;
      lastQuantum: number;
    }>[],
  ): void => {
    if (bytes % 4 !== 0) {
      throw new Error(`ACE DiT arena role ${key} is not four-byte aligned`);
    }
    roles.push(Object.freeze({
      key,
      label: `ace-dit-${key.replaceAll(".", "-")}`,
      byteLength: bytes,
      lifetimes,
    }));
  };

  add(
    "conditionInput",
    activation(plan.batch * plan.conditionTokens * ACE_TURBO_DIT_CONFIG.conditionInputSize),
    interval(0),
  );
  add(
    "projectedCondition",
    activation(plan.batch * plan.conditionTokens * ACE_TURBO_DIT_CONFIG.hiddenSize),
    interval(0, STATIC_QUANTA - 1),
  );
  for (const key of CROSS_CACHE_SCRATCH_KEYS) {
    add(
      `crossCacheScratch.${key}`,
      activation(plan.crossKeyValueElementsPerLayer),
      interval(1, STATIC_QUANTA - 1),
    );
  }
  for (let layer = 0; layer < LAYER_COUNT; layer += 1) {
    const lastRead =
      STATIC_QUANTA +
      (evaluationCount - 1) * EVALUATION_QUANTA +
      2 + layer;
    add(
      `crossCache.${layer}.key`,
      activation(plan.crossKeyValueElementsPerLayer),
      interval(1 + layer, lastRead),
    );
    add(
      `crossCache.${layer}.value`,
      activation(plan.crossKeyValueElementsPerLayer),
      interval(1 + layer, lastRead),
    );
  }
  add("context", activation(plan.contextElements), all);
  add("latent.0", activation(plan.latentElements), all);
  add("latent.1", activation(plan.latentElements), all);
  add("latent.2", activation(plan.latentElements), repeated(27));
  add(
    "concatenatedInput",
    activation(plan.batch * plan.latentFrames * ACE_TURBO_DIT_CONFIG.inputChannels),
    repeated(1),
  );
  add("hidden.0", activation(plan.hiddenElements), repeated(1, 26));
  // Match the graph owner's deliberately conservative ping/pong lifetime.
  add("hidden.1", activation(plan.hiddenElements), repeated(1, 26));

  const timestepElements: Readonly<Record<keyof AceDitTimestepScratch, number>> = {
    timestepFrequency: plan.batch * ACE_TURBO_DIT_CONFIG.timestepInputSize,
    relativeFrequency: plan.batch * ACE_TURBO_DIT_CONFIG.timestepInputSize,
    timestepLinear1: plan.batch * ACE_TURBO_DIT_CONFIG.hiddenSize,
    relativeLinear1: plan.batch * ACE_TURBO_DIT_CONFIG.hiddenSize,
    timestepActivation1: plan.batch * ACE_TURBO_DIT_CONFIG.hiddenSize,
    relativeActivation1: plan.batch * ACE_TURBO_DIT_CONFIG.hiddenSize,
    timestepEmbedding: plan.batch * ACE_TURBO_DIT_CONFIG.hiddenSize,
    relativeEmbedding: plan.batch * ACE_TURBO_DIT_CONFIG.hiddenSize,
    timestepActivation2: plan.batch * ACE_TURBO_DIT_CONFIG.hiddenSize,
    relativeActivation2: plan.batch * ACE_TURBO_DIT_CONFIG.hiddenSize,
    timestepProjection:
      plan.batch * 6 * ACE_TURBO_DIT_CONFIG.hiddenSize,
    relativeProjection:
      plan.batch * 6 * ACE_TURBO_DIT_CONFIG.hiddenSize,
  };
  for (const key of TIMESTEP_SCRATCH_KEYS) {
    add(
      `timestepScratch.${key}`,
      activation(timestepElements[key]),
      repeated(0),
    );
  }
  add(
    "timestepEmbedding",
    activation(plan.batch * ACE_TURBO_DIT_CONFIG.hiddenSize),
    repeated(0, 26),
  );
  add(
    "timestepProjection",
    activation(plan.batch * 6 * ACE_TURBO_DIT_CONFIG.hiddenSize),
    repeated(0, 25),
  );
  const layerElements = layerScratchElements(plan);
  for (const key of LAYER_SCRATCH_KEYS) {
    add(
      `layerScratch.${key}`,
      aceDitLayerScratchBytes(
        modelProfile,
        key,
        layerElements[key],
        ditDenseInputStorageProfile,
      ),
      repeated(2, 25),
    );
  }
  add("outputScratch.normalized", activation(plan.hiddenElements), repeated(26));
  add(
    "outputScratch.modulation",
    activation(plan.batch * 2 * ACE_TURBO_DIT_CONFIG.hiddenSize),
    repeated(26),
  );
  add("outputScratch.modulated", activation(plan.hiddenElements), repeated(26));
  add("velocity", activation(plan.latentElements), repeated(26, 27));
  add(
    "predictedCleanLatent",
    activation(plan.latentElements),
    repeated(27),
  );

  const finalLayerQuantum =
    STATIC_QUANTA +
    (evaluationCount - 1) * EVALUATION_QUANTA +
    25;
  add(
    "control.selfValidLengths",
    plan.batch * 2 * UINT32_BYTES,
    interval(0, finalLayerQuantum),
  );
  add(
    "control.crossValidLengths",
    plan.batch * 2 * UINT32_BYTES,
    interval(0, finalLayerQuantum),
  );
  add(
    "control.cosine",
    plan.tokens * ACE_TURBO_DIT_CONFIG.headDimension * FLOAT32_BYTES,
    interval(0, finalLayerQuantum),
  );
  add(
    "control.sine",
    plan.tokens * ACE_TURBO_DIT_CONFIG.headDimension * FLOAT32_BYTES,
    interval(0, finalLayerQuantum),
  );
  for (let evaluation = 0; evaluation < evaluationCount; evaluation += 1) {
    add(
      `control.timestep.${evaluation}`,
      plan.batch * FLOAT32_BYTES,
      interval(0, STATIC_QUANTA + evaluation * EVALUATION_QUANTA),
    );
  }
  add("control.relativeTimestepZero", plan.batch * FLOAT32_BYTES, all);
  return Object.freeze(roles);
}

function layerScratchElements(
  plan: AceDitEvaluationPlan,
): Readonly<Record<keyof AceDitLayerScratch, number>> {
  const hidden = plan.hiddenElements;
  const query = plan.queryElements;
  const keyValue = plan.selfKeyValueElements;
  const intermediate = plan.intermediateElements;
  return Object.freeze({
    modulation: 6 * plan.batch * ACE_TURBO_DIT_CONFIG.hiddenSize,
    selfNormalized: hidden,
    selfModulated: hidden,
    selfQueryFlat: query,
    selfKeyFlat: keyValue,
    selfValueFlat: keyValue,
    selfQueryHeads: query,
    selfKeyHeads: keyValue,
    selfValueHeads: keyValue,
    selfNormalizedQueryHeads: query,
    selfNormalizedKeyHeads: keyValue,
    selfRotatedQueryHeads: query,
    selfRotatedKeyHeads: keyValue,
    selfAttentionHeads: query,
    selfMergedAttention: query,
    selfProjectedAttention: hidden,
    afterSelfAttention: hidden,
    crossNormalized: hidden,
    crossQueryFlat: query,
    crossQueryHeads: query,
    crossNormalizedQueryHeads: query,
    crossAttentionHeads: query,
    crossMergedAttention: query,
    crossProjectedAttention: hidden,
    afterCrossAttention: hidden,
    mlpNormalized: hidden,
    mlpModulated: hidden,
    gate: intermediate,
    up: intermediate,
    gatedActivation: intermediate,
    projectedMlp: hidden,
  });
}

function activationBytes(profile: AceModelProfileId, elements: number): number {
  return checkedProduct(
    [elements, profile === "reference-bf16" ? 4 : profile === "raw-fp16" ? 2 : 0],
    "ACE DiT activation bytes",
  );
}

function lifetimesOverlap(
  left: AceDitBackendArenaRole,
  right: AceDitBackendArenaRole,
): boolean {
  return left.lifetimes.some((leftLifetime) =>
    right.lifetimes.some((rightLifetime) =>
      leftLifetime.firstQuantum <= rightLifetime.lastQuantum &&
      rightLifetime.firstQuantum <= leftLifetime.lastQuantum
    )
  );
}

function validatePreparedResources(resources: AceDitPreparedGpuResources): void {
  const samplerScheduleProfile = requireResolvedAceDitSamplerScheduleProfile(
    resources.samplerScheduleProfile ??
      ACE_TURBO_EIGHT_SAMPLER_SCHEDULE_PROFILE,
  );
  const physicalQuantaPerCommandBuffer =
    resources.physicalQuantaPerCommandBuffer ??
    ACE_DIT_PHYSICAL_QUANTA_PER_COMMAND_BUFFER;
  const expected = planAceDitGpuBackendMemory(
    resources.modelProfile,
    resources.shape,
    resources.memory.residentWeightBytes,
    resources.gemmBackend,
    physicalQuantaPerCommandBuffer,
    resources.evaluationReadbacks !== undefined,
    resources.opt0062AttentionIdentityCounters !== undefined,
    samplerScheduleProfile,
    resources.evaluation0VelocityReadback !== undefined,
    resources.denseInputStorageProfile,
  );
  if (
    resources.denseInputStorageProfile !==
      resources.memory.denseInputStorageProfile ||
    resources.memory.denseInputStorageProfile !==
      resources.memory.graph.denseInputStorageProfile ||
    resources.memory.denseInputStorageProfile !==
      resources.memory.arena.denseInputStorageProfile ||
    resources.memory.denseInputStorageProfile !==
      expected.denseInputStorageProfile ||
    resources.memory.modelProfile !== expected.modelProfile ||
    resources.memory.gemmBackend !== expected.gemmBackend ||
    resources.memory.commandBufferCount !== expected.commandBufferCount ||
    resources.memory.physicalGraphQuantumCount !==
      expected.physicalGraphQuantumCount ||
    resources.memory.physicalQuantaPerCommandBuffer !==
      physicalQuantaPerCommandBuffer ||
    resources.memory.graphCommandBufferCount !==
      expected.graphCommandBufferCount ||
    resources.memory.readbackCommandBufferCount !== 1 ||
    resources.memory.samplerScheduleProfileId !== samplerScheduleProfile.id ||
    resources.memory.evaluationCount !== samplerScheduleProfile.evaluationCount ||
    resources.memory.logicalGraphQuantumCount !==
      aceDitGraphQuantumCount(samplerScheduleProfile) ||
    resources.graph.commandBufferCount !==
      resources.memory.physicalGraphQuantumCount ||
    (resources.ditSubmissionPolicy !== undefined &&
      resources.ditSubmissionPolicy !== "depth1-epoch1" &&
      resources.ditSubmissionPolicy !== "depth2-phase-epoch4") ||
    (resources.ditSubmissionPolicy === "depth2-phase-epoch4" &&
      (resources.modelProfile !== "reference-bf16" ||
        resources.denseInputStorageProfile !== undefined ||
        resources.gemmBackend !== "mixed-opt-0009" ||
        samplerScheduleProfile.role !== "production-default" ||
        physicalQuantaPerCommandBuffer !== 1 ||
        resources.graph.physicalCommandDescriptors !== undefined ||
        resources.onCommandProfile !== undefined ||
        resources.onSchedulingProfile !== undefined ||
        resources.onOpt0067EvaluationProfile !== undefined ||
        resources.onOpt0080EvaluationProfile !== undefined ||
        resources.onOpt0080FullProfile !== undefined ||
        resources.opt0067EvaluationLimit !== undefined ||
        resources.opt0080SchedulingProfile !== undefined ||
        resources.opt0080FullSchedulingProfile !== undefined ||
        resources.onOpt0080CommandBufferCompleted !== undefined ||
        resources.evaluationReadbacks !== undefined ||
        resources.evaluation0VelocityReadback !== undefined ||
        resources.opt0062AttentionIdentityCounters !== undefined ||
        resources.opt0062AttentionIdentityReadback !== undefined)) ||
    (resources.onCommandProfile !== undefined ||
      resources.onSchedulingProfile !== undefined ||
      resources.onOpt0067EvaluationProfile !== undefined ||
      resources.onOpt0080EvaluationProfile !== undefined ||
      resources.onOpt0080FullProfile !== undefined) !==
      (resources.graph.physicalCommandDescriptors !== undefined) ||
    [
      resources.onCommandProfile,
      resources.onSchedulingProfile,
      resources.onOpt0067EvaluationProfile,
      resources.onOpt0080EvaluationProfile,
      resources.onOpt0080FullProfile,
    ].filter((capture) => capture !== undefined).length > 1 ||
    (resources.opt0067EvaluationLimit === 1) !==
      (resources.onOpt0067EvaluationProfile !== undefined ||
        resources.onOpt0080EvaluationProfile !== undefined) ||
    (resources.opt0080SchedulingProfile !== undefined) !==
      (resources.onOpt0080EvaluationProfile !== undefined) ||
    (resources.opt0080FullSchedulingProfile !== undefined) !==
      (resources.onOpt0080FullProfile !== undefined) ||
    (resources.onOpt0080CommandBufferCompleted !== undefined &&
      (resources.opt0080SchedulingProfile === undefined &&
        resources.opt0080FullSchedulingProfile === undefined ||
        typeof resources.onOpt0080CommandBufferCompleted !== "function")) ||
    (resources.opt0080SchedulingProfile !== undefined &&
      resources.opt0080SchedulingProfile !== "depth1-epoch1" &&
      resources.opt0080SchedulingProfile !== "opt-0080-depth2-epoch4") ||
    (resources.opt0080FullSchedulingProfile !== undefined &&
      resources.opt0080FullSchedulingProfile !== "depth1-epoch1" &&
      resources.opt0080FullSchedulingProfile !==
        "opt-0080-depth2-epoch4") ||
    (resources.opt0067EvaluationLimit === 1 &&
      (physicalQuantaPerCommandBuffer !== 1 ||
        resources.evaluationReadbacks !== undefined)) ||
    (resources.opt0080SchedulingProfile !== undefined &&
      (resources.shape.batch !== 1 ||
        resources.shape.latentFrames !== 4_500 ||
        resources.shape.conditionTokens !== 98 ||
        resources.opt0062AttentionIdentityCounters !== undefined ||
        resources.opt0062AttentionIdentityReadback !== undefined)) ||
    (resources.opt0080FullSchedulingProfile !== undefined &&
      (resources.opt0067EvaluationLimit !== undefined ||
        physicalQuantaPerCommandBuffer !== 1 ||
        resources.shape.batch !== 1 ||
        resources.shape.latentFrames !== 4_500 ||
        resources.shape.conditionTokens !== 98 ||
        resources.opt0062AttentionIdentityCounters !== undefined ||
        resources.opt0062AttentionIdentityReadback !== undefined)) ||
    resources.memory.detachedFinalLatentBytes !==
      resources.shape.latentElements * FLOAT32_BYTES ||
    resources.memory.evaluationReadbackBytes !==
      expected.evaluationReadbackBytes ||
    resources.memory.detachedEvaluationLatentBytes !==
      expected.detachedEvaluationLatentBytes ||
    resources.memory.evaluation0VelocityReadbackBytes !==
      expected.evaluation0VelocityReadbackBytes ||
    resources.memory.detachedEvaluation0VelocityBytes !==
      expected.detachedEvaluation0VelocityBytes ||
    resources.memory.opt0062AttentionIdentityCounterBytes !==
      expected.opt0062AttentionIdentityCounterBytes ||
    resources.memory.opt0062AttentionIdentityReadbackBytes !==
      expected.opt0062AttentionIdentityReadbackBytes ||
    resources.memory.detachedOpt0062AttentionIdentityBytes !==
      expected.detachedOpt0062AttentionIdentityBytes ||
    (resources.evaluationReadbacks === undefined
      ? resources.memory.evaluationReadbackBytes !== 0
      : physicalQuantaPerCommandBuffer !== 1 ||
        resources.evaluationReadbacks.length !==
          samplerScheduleProfile.evaluationCount ||
        resources.evaluationReadbacks.some((buffer) =>
          buffer.size < resources.memory.readbackBufferBytes
        )) ||
    (resources.evaluation0VelocityReadback === undefined
      ? resources.memory.evaluation0VelocityReadbackBytes !== 0
      : resources.evaluationReadbacks === undefined ||
        resources.evaluation0VelocityReadback.size <
          resources.memory.readbackBufferBytes) ||
    (resources.opt0062AttentionIdentityCounters === undefined
      ? resources.opt0062AttentionIdentityReadback !== undefined ||
        resources.memory.opt0062AttentionIdentityCounterBytes !== 0 ||
        resources.memory.opt0062AttentionIdentityReadbackBytes !== 0
      : resources.opt0062AttentionIdentityReadback === undefined ||
        (resources.evaluationReadbacks === undefined &&
          resources.opt0067EvaluationLimit !== 1) ||
        physicalQuantaPerCommandBuffer !== 1 ||
        resources.opt0062AttentionIdentityCounters.size <
          ACE_OPT_0062_IDENTITY_COUNTER_BYTES ||
        resources.opt0062AttentionIdentityReadback.size <
          ACE_OPT_0062_IDENTITY_COUNTER_BYTES) ||
    resources.readback.size < resources.memory.readbackBufferBytes
  ) {
    throw new Error("ACE DiT prepared resources violate the backend contract");
  }
}

async function collectGpuErrorScopes(
  device: GPUDevice,
  operation: string,
): Promise<unknown | undefined> {
  const validation = device.popErrorScope();
  const outOfMemory = device.popErrorScope();
  const internal = device.popErrorScope();
  const results = await Promise.allSettled([validation, outOfMemory, internal]);
  const names = ["validation", "out-of-memory", "internal"] as const;
  for (let index = 0; index < results.length; index += 1) {
    const result = results[index]!;
    if (result.status === "rejected") return result.reason;
    if (result.value !== null) {
      return new Error(
        `${operation} failed with a WebGPU ${names[index]} error: ${result.value.message}`,
        { cause: result.value },
      );
    }
  }
  return undefined;
}

async function discardGpuErrorScopes(device: GPUDevice): Promise<void> {
  await Promise.allSettled([
    device.popErrorScope(),
    device.popErrorScope(),
    device.popErrorScope(),
  ]);
}

function combineSignals(
  signals: readonly (AbortSignal | undefined)[],
): AbortSignal {
  const present = signals.filter(
    (signal): signal is AbortSignal => signal !== undefined,
  );
  return present.length === 1 ? present[0]! : AbortSignal.any(present);
}

function yieldQueueIdle(): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(resolve, ACE_COOPERATIVE_GPU_IDLE_MILLISECONDS);
  });
}

function destroyedError(): DOMException {
  return new DOMException("ACE DiT GPU backend was destroyed", "AbortError");
}

function numberToFloat16Bits(value: number): number {
  const f32 = new Float32Array([value]);
  const bits = new Uint32Array(f32.buffer)[0]!;
  const sign = (bits >>> 16) & 0x8000;
  const exponent = (bits >>> 23) & 0xff;
  const mantissa = bits & 0x7f_ffff;
  if (exponent === 0xff) return sign | (mantissa === 0 ? 0x7c00 : 0x7e00);
  const halfExponent = exponent - 127 + 15;
  if (halfExponent >= 0x1f) return sign | 0x7c00;
  if (halfExponent <= 0) {
    if (halfExponent < -10) return sign;
    const significant = mantissa | 0x80_0000;
    const shift = 14 - halfExponent;
    const truncated = significant >>> shift;
    const remainder = significant & ((1 << shift) - 1);
    const halfway = 1 << (shift - 1);
    return sign | (truncated + (
      remainder > halfway ||
      (remainder === halfway && (truncated & 1) !== 0)
        ? 1
        : 0
    ));
  }
  let halfMantissa = mantissa >>> 13;
  const remainder = mantissa & 0x1fff;
  if (
    remainder > 0x1000 ||
    (remainder === 0x1000 && (halfMantissa & 1) !== 0)
  ) {
    halfMantissa += 1;
    if (halfMantissa === 0x400) {
      const nextExponent = halfExponent + 1;
      return sign | (nextExponent >= 0x1f ? 0x7c00 : nextExponent << 10);
    }
  }
  return sign | (halfExponent << 10) | halfMantissa;
}

function float16BitsToNumber(bits: number): number {
  const sign = (bits & 0x8000) << 16;
  let exponent = (bits >>> 10) & 0x1f;
  let mantissa = bits & 0x03ff;
  let output: number;
  if (exponent === 0) {
    if (mantissa === 0) output = sign;
    else {
      exponent = 1;
      while ((mantissa & 0x0400) === 0) {
        mantissa <<= 1;
        exponent -= 1;
      }
      mantissa &= 0x03ff;
      output = sign | ((exponent + 112) << 23) | (mantissa << 13);
    }
  } else if (exponent === 0x1f) {
    output = sign | 0x7f80_0000 | (mantissa << 13);
  } else {
    output = sign | ((exponent + 112) << 23) | (mantissa << 13);
  }
  const words = new Uint32Array([output >>> 0]);
  return new Float32Array(words.buffer)[0]!;
}

function align(value: number, alignment: number): number {
  const result = Math.ceil(value / alignment) * alignment;
  if (!Number.isSafeInteger(result)) {
    throw new RangeError("ACE DiT alignment exceeds safe integers");
  }
  return result;
}

function checkedProduct(values: readonly number[], label: string): number {
  let result = 1;
  for (const value of values) {
    if (!Number.isSafeInteger(value) || value <= 0) {
      throw new RangeError(`${label} requires positive safe integers`);
    }
    result *= value;
    if (!Number.isSafeInteger(result)) {
      throw new RangeError(`${label} exceeds safe integer arithmetic`);
    }
  }
  return result;
}

function checkedSum(values: readonly number[], label: string): number {
  let result = 0;
  for (const value of values) {
    requireNonNegativeSafeInteger(value, label);
    result += value;
    if (!Number.isSafeInteger(result)) {
      throw new RangeError(`${label} exceeds safe integer arithmetic`);
    }
  }
  return result;
}

function requireNonNegativeSafeInteger(value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new RangeError(`${label} must be a non-negative safe integer`);
  }
}

const CROSS_CACHE_SCRATCH_KEYS = Object.freeze([
  "keyFlat",
  "valueFlat",
  "keyHeads",
] as const satisfies readonly (keyof AceDitCrossCacheScratch)[]);

const TIMESTEP_SCRATCH_KEYS = Object.freeze([
  "timestepFrequency",
  "relativeFrequency",
  "timestepLinear1",
  "relativeLinear1",
  "timestepActivation1",
  "relativeActivation1",
  "timestepEmbedding",
  "relativeEmbedding",
  "timestepActivation2",
  "relativeActivation2",
  "timestepProjection",
  "relativeProjection",
] as const satisfies readonly (keyof AceDitTimestepScratch)[]);

const LAYER_SCRATCH_KEYS = Object.freeze([
  "modulation",
  "selfNormalized",
  "selfModulated",
  "selfQueryFlat",
  "selfKeyFlat",
  "selfValueFlat",
  "selfQueryHeads",
  "selfKeyHeads",
  "selfValueHeads",
  "selfNormalizedQueryHeads",
  "selfNormalizedKeyHeads",
  "selfRotatedQueryHeads",
  "selfRotatedKeyHeads",
  "selfAttentionHeads",
  "selfMergedAttention",
  "selfProjectedAttention",
  "afterSelfAttention",
  "crossNormalized",
  "crossQueryFlat",
  "crossQueryHeads",
  "crossNormalizedQueryHeads",
  "crossAttentionHeads",
  "crossMergedAttention",
  "crossProjectedAttention",
  "afterCrossAttention",
  "mlpNormalized",
  "mlpModulated",
  "gate",
  "up",
  "gatedActivation",
  "projectedMlp",
] as const satisfies readonly (keyof AceDitLayerScratch)[]);

const OUTPUT_SCRATCH_KEYS = Object.freeze([
  "normalized",
  "modulation",
  "modulated",
] as const satisfies readonly (keyof AceDitOutputScratch)[]);
