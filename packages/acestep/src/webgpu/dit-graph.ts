import {
  ACE_TURBO_DENOISING_EVALUATIONS,
  type AceDynamicConditionalWeightingConfiguration,
} from "../api.js";
import type { AcePackageManifest } from "../model/manifest.js";
import { aceSha256Hex } from "../model/sha256.js";
import {
  AceGpuTensorPhase,
  type AceGpuTensorPhaseProgress,
} from "../model/gpu-tensors.js";
import {
  AceCooperativeGpuScheduler,
  type AceDepth2Epoch4CommandBufferCompletionTiming,
  type AceDepth2Epoch4CompletionEpochTiming,
  type AceDepth2Epoch4PhaseStart,
  type AceDepth2Epoch4SchedulingProgress,
  type AceGpuSchedulingResult,
} from "../runtime/scheduler.js";
import {
  ACE_TURBO_DIT_CONFIG,
  AceCorrectnessDitRuntime,
  aceDitLayerScratchBytes,
  aceDitLayerAttentionMode,
  createAceDitRopeTables,
  createAceDitSamplerSchedule,
  planAceDitEvaluation,
  requireAceDitDenseInputStorageProfile,
  validateAceDitConfig,
  type AceDitCrossCacheScratch,
  type AceDitEvaluationPlan,
  type AceDitEvaluationShape,
  type AceDitDenseGemmRuntimeConfiguration,
  type AceDitDenseInputStorageProfile,
  type AceDitGemmRuntimeConfiguration,
  type AceDitLayerBindings,
  type AceDitLayerScratch,
  type AceDitOutputScratch,
  type AceDitTimestepScratch,
} from "./ace-dit.js";
import {
  ACE_DIT_LAYER_COUNT,
  resolveAceDitPackageWeights,
  type AceDitResolvedPackageWeights,
} from "./ace-dit-package.js";
import type { AceModelProfileId } from "./capabilities.js";
import {
  ACE_DIT_QUERY8_ATTENTION_RUNTIME_PROFILE,
  ACE_OPT_0062_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
  ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
  type AceDitAttentionRuntimeProfile,
} from "./dit-attention-profile.js";
import {
  ACE_OPT_0009_DIT_DENSE_KERNEL_SET_ID,
  ACE_OPT_0009_DIT_DENSE_RUNTIME_PROFILE,
  ACE_OPT_0037_DIT_K4_LAYER_BYTES,
  ACE_OPT_0037_DIT_K4_KERNEL_SET_ID,
  ACE_OPT_0037_DIT_K4_RESIDENT_WEIGHT_BYTES,
  ACE_OPT_0037_DIT_K4_RUNTIME_PROFILE,
  ACE_OPT_0056_DIT_SELECTIVE_K4_RUNTIME_PROFILE,
  ACE_REFERENCE_DIT_SHARED_WEIGHT_BYTES,
  type AceDitDenseRuntimeProfile,
} from "./dit-fp16-package.js";
import {
  ACE_TURBO_EIGHT_SAMPLER_SCHEDULE_PROFILE,
  requireResolvedAceDitSamplerScheduleProfile,
  type AceDitSamplerScheduleProfile,
  type AceDitSamplerEvaluationCount,
} from "./dit-sampler-profile.js";
import { aceActivationBytes } from "./kernels/correctness-utils.js";
import { ACE_OPT_0056_DENSE_K4_EXACT_KERNEL_ID } from
  "./kernels/dit-dense-fp16-k4-exact.js";
import type { AceOpt0056DenseRouteProfile } from
  "./kernels/dit-dense-fp16-k4-selective-exact.js";
import {
  ACE_OPT_0062_IDENTITY_COUNTER_BYTES,
  ACE_OPT_0062_QUAD_QUERY_ATTENTION_KERNEL_ID,
  type AceOpt0062AttentionRouteProfile,
} from "./kernels/attention-quad-query-production.js";
import type { AceOpt0070ProductionAttentionRouteProfile } from
  "./kernels/attention-opt0070-production.js";
import type { AceAttentionRuntimeConfiguration } from "./kernels/attention.js";
import {
  aceCompositeCooperativeSequence,
  type AceGpuEncodeQuantum,
  type AceGpuEncodeSequence,
  type AceGpuEncodeSequenceQuantumDescriptor,
  type AceGemmDispatch,
} from "./kernels/gemm.js";

const CONTROL_ELEMENT_BYTES = Uint32Array.BYTES_PER_ELEMENT;
const FP32_BYTES = Float32Array.BYTES_PER_ELEMENT;
const LAYER_MODULATION_GROUPS = 6;
const OUTPUT_MODULATION_GROUPS = 2;
const EVALUATION_QUANTA = 1 + 1 + ACE_DIT_LAYER_COUNT + 1 + 1;
const STATIC_QUANTA = 1 + ACE_DIT_LAYER_COUNT;
export const ACE_OPT_0018_M2250_GRAPH_COMMAND_BUFFER_COUNT = 2_553;
/** OPT-0067 executes the unchanged 25-command precompute plus evaluation 0. */
export const ACE_OPT_0067_M2250_EVALUATION0_GRAPH_COMMAND_BUFFER_COUNT = 341;
export const ACE_OPT_0080_M2250_EVALUATION0_PHASE_COMMAND_BUFFER_COUNTS =
  Object.freeze([25, 316] as const);
/** OPT-0080 full-graph confirmation preserves all nine production phases. */
export const ACE_OPT_0080_M2250_FULL_PHASE_COMMAND_BUFFER_COUNTS =
  Object.freeze([25, 316, 316, 316, 316, 316, 316, 316, 316] as const);
const ACE_OPT_0018_M2250_LATENT_FRAMES = 4_500;
const ACE_OPT_0018_M2250_TOKENS = 2_250;
const ACE_OPT_0018_CONDITION_TOKENS = 98;
const PROFILE_BASE_GEMM_KERNEL = "fixed32-subgroup-gemm-n128-k32-v1";
const PROFILE_ATTENTION_KERNEL = "fixed32-subgroup-query8";
const PROFILE_RMSNORM_KERNEL = "reference-bf16-rmsnorm";
const PROFILE_ROPE_KERNEL = "reference-bf16-rope";
const PROFILE_TRANSFORMER_KERNEL = "reference-bf16-transformer-plumbing";
const PROFILE_DIT_KERNEL = "reference-bf16-dit-plumbing";
const PROFILE_DCW_KERNEL = "reference-bf16-haar-dcw";

export const ACE_DIT_GRAPH_QUANTUM_COUNT =
  STATIC_QUANTA + ACE_TURBO_DENOISING_EVALUATIONS * EVALUATION_QUANTA;
export const ACE_DIT_PHYSICAL_QUANTA_PER_COMMAND_BUFFER = 1 as const;
export type AceDitPhysicalQuantaPerCommandBuffer = 1 | 8 | 16;

export type AceDitGraphQuantum =
  | Readonly<{
      index: number;
      kind: "condition-projection";
      label: string;
    }>
  | Readonly<{
      index: number;
      kind: "cross-cache";
      layer: number;
      label: string;
    }>
  | Readonly<{
      index: number;
      kind: "timestep" | "input-projection" | "output-projection" | "sampler";
      evaluation: number;
      label: string;
    }>
  | Readonly<{
      index: number;
      kind: "layer";
      evaluation: number;
      layer: number;
      label: string;
    }>;

export interface AceDitGraphMemoryPlan {
  readonly modelProfile: AceModelProfileId;
  readonly denseInputStorageProfile?: AceDitDenseInputStorageProfile;
  readonly activationElementBytes: 2 | 4;
  readonly conditionInputBytes: number;
  readonly projectedConditionBytes: number;
  readonly crossKeyValueBytes: number;
  readonly crossCacheScratchBytes: number;
  readonly contextBytes: number;
  readonly latentBytes: number;
  readonly ropeBytes: number;
  readonly attentionControlBytes: number;
  readonly timestepInputBytes: number;
  readonly timestepScratchBytes: number;
  readonly timestepOutputBytes: number;
  readonly concatenatedInputBytes: number;
  readonly hiddenStateBytes: number;
  readonly layerScratchBytes: number;
  readonly outputScratchBytes: number;
  readonly samplerScratchBytes: number;
  /** Peak live graph storage with the documented cross-stage aliasing plan. */
  readonly minimumGraphBytesExcludingWeights: number;
  /** Straight sum when the caller deliberately gives every role its own range. */
  readonly unaliasedGraphBytesExcludingWeights: number;
  readonly largestRequiredBindingBytes: number;
  readonly relativeZeroPolicy: "immutable-shared-input-recomputed-each-evaluation";
}

export interface AceDitGraphControlData {
  readonly selfValidLengths: Uint32Array;
  readonly crossValidLengths: Uint32Array;
  readonly cosine: Float32Array;
  readonly sine: Float32Array;
  /** One array per authenticated evaluation, filled with its effective BF16 t. */
  readonly timesteps: readonly Float32Array[];
  /** Shared immutable input. The relative branch itself is intentionally rerun. */
  readonly relativeTimestepZero: Float32Array;
}

export interface AceDitGraphCrossCacheBindings {
  readonly key: GPUBufferBinding;
  readonly value: GPUBufferBinding;
}

export interface AceDitGraphControlBindings {
  readonly selfValidLengths: GPUBufferBinding;
  readonly crossValidLengths: GPUBufferBinding;
  readonly cosine: GPUBufferBinding;
  readonly sine: GPUBufferBinding;
  readonly timesteps: readonly GPUBufferBinding[];
  readonly relativeTimestepZero: GPUBufferBinding;
}

/**
 * Caller-owned graph storage. Roles may share an arena range only when the
 * validator's quantum lifetimes do not overlap. Model weights are separate.
 */
export interface AceDitGraphBindings {
  readonly conditionInput: GPUBufferBinding;
  readonly projectedCondition: GPUBufferBinding;
  readonly crossCaches: readonly AceDitGraphCrossCacheBindings[];
  readonly crossCacheScratch: AceDitCrossCacheScratch;
  readonly context: GPUBufferBinding;
  /** Ping-pong state; index zero initially contains the seeded latent. */
  readonly latents: readonly [GPUBufferBinding, GPUBufferBinding, GPUBufferBinding];
  readonly concatenatedInput: GPUBufferBinding;
  readonly hidden: readonly [GPUBufferBinding, GPUBufferBinding];
  readonly timestepScratch: AceDitTimestepScratch;
  readonly timestepEmbedding: GPUBufferBinding;
  readonly timestepProjection: GPUBufferBinding;
  readonly layerScratch: AceDitLayerScratch;
  readonly outputScratch: AceDitOutputScratch;
  readonly velocity: GPUBufferBinding;
  readonly predictedCleanLatent: GPUBufferBinding;
  readonly controls: AceDitGraphControlBindings;
  /** OPT-0062 correctness-only raw-U32 counters; never arena-aliased. */
  readonly opt0062AttentionIdentityCounters?: GPUBufferBinding;
}

export interface AceDitGraphProgress {
  /** Completed physical sequence quanta, not coalesced command buffers. */
  readonly completedQuanta: number;
  readonly totalQuanta: number;
  readonly completedCommandBuffers: number;
  readonly totalCommandBuffers: number;
  readonly queueDrains: number;
  readonly cooperativeIdleMs: number;
  readonly completedEvaluations: number;
  readonly totalEvaluations: AceDitSamplerEvaluationCount;
  readonly quantum: AceDitGraphQuantum;
  readonly subquantumIndex: number;
  readonly subquantumCount: number;
  readonly commandId: string;
  readonly batchIndex: number;
  readonly batchFirstPhysicalQuantum: number;
  readonly batchPhysicalQuantumCount: number;
  readonly physicalQuantaPerCommandBuffer: AceDitPhysicalQuantaPerCommandBuffer;
}

export interface AceDitGraphRunResult extends AceGpuSchedulingResult {
  readonly completedEvaluations: 1 | AceDitSamplerEvaluationCount;
  readonly finalLatent: GPUBufferBinding;
  readonly denseRouteProfile?: AceOpt0056DenseRouteProfile;
  readonly attentionRouteProfile?:
    | AceOpt0062AttentionRouteProfile
    | AceOpt0070ProductionAttentionRouteProfile;
  /** OPT-0080 benchmark-only scheduling and descriptor-owner diagnostics. */
  readonly opt0080Scheduling?: AceDitOpt0080SchedulingDiagnostics;
}

export type AceDitOpt0080SchedulingProfile =
  | "depth1-epoch1"
  | "opt-0080-depth2-epoch4";

/** Capture-free production submission policy selected by the pipeline. */
export type AceDitSubmissionPolicy =
  | "depth1-epoch1"
  | "depth2-phase-epoch4";

export interface AceDitOpt0080SchedulingDiagnostics {
  readonly profile: AceDitOpt0080SchedulingProfile;
  readonly completionFenceRequestedCount: number;
  readonly completionFenceSettledCount: number;
  readonly completionFenceRejectedCount: number;
  readonly trueQueueDrainCount: number;
  readonly completionEpochCount: number;
  readonly requestedCooperativeIdleMs: number;
  readonly cooperativeIdleTurns: number;
  readonly maximumOutstandingCommandBuffers: number;
  readonly maximumPendingDescriptorCount: number;
}

export interface AceDitOpt0080CommandBufferCompletion
  extends AceDepth2Epoch4CommandBufferCompletionTiming {
  readonly descriptor: AceDitPhysicalCommandDescriptor;
  readonly schedulingProgress: AceDepth2Epoch4SchedulingProgress;
  readonly graphProgress: AceDitGraphProgress;
}

export type AceDitOpt0080CompletionEpochTiming =
  AceDepth2Epoch4CompletionEpochTiming;

export type AceDitOpt0080PhaseStart = AceDepth2Epoch4PhaseStart;

export interface AceDitGraphRunOptions {
  readonly signal: AbortSignal;
  /** Internal product policy. Omission retains the conservative depth-one path. */
  readonly submissionPolicy?: AceDitSubmissionPolicy;
  /** OPT-0034 benchmark seam. Stable production omits this and remains one. */
  readonly physicalQuantaPerCommandBuffer?: AceDitPhysicalQuantaPerCommandBuffer;
  readonly yieldQueueIdle?: () => Promise<void>;
  readonly onProgress?: (progress: AceDitGraphProgress) => void;
  readonly onCommandBufferDrained?: (
    descriptor: AceDitPhysicalCommandDescriptor,
    submitThroughDrainMs: number,
  ) => void;
  readonly onPhysicalBatchDrained?: (
    descriptor: AceDitPhysicalCommandBatchDescriptor,
    submitThroughDrainMs: number,
  ) => void;
  /** Diagnostic-only bounded COPY_DST/MAP_READ buffers, one per sampler. */
  readonly evaluationReadbacks?: readonly GPUBuffer[];
  /** OPT-0055/0065 only: copied with evaluation zero's sampler submission. */
  readonly evaluation0VelocityReadback?: GPUBuffer;
  /** OPT-0062 only: copied after evaluation seven in that same submission. */
  readonly opt0062AttentionIdentityReadback?: GPUBuffer;
  /** OPT-0067 only: execute precompute plus evaluation 0, then stop drained. */
  readonly opt0067EvaluationLimit?: 1;
  /** OPT-0080 only: exact evaluation-0 scheduling arm and capture. */
  readonly opt0080SchedulingProfile?: AceDitOpt0080SchedulingProfile;
  /** OPT-0080 confirmation only: execute all eight production evaluations. */
  readonly opt0080FullGraph?: true;
  readonly onOpt0080CommandBufferCompleted?: (
    completion: AceDitOpt0080CommandBufferCompletion,
  ) => void;
  readonly onOpt0080CompletionEpochDrained?: (
    timing: AceDitOpt0080CompletionEpochTiming,
  ) => void;
  readonly onOpt0080PhaseStarted?: (phase: AceDitOpt0080PhaseStart) => void;
}

export interface AceDitGraphCompilationProgress {
  readonly compiledQuanta: number;
  readonly totalQuanta: number;
  readonly quantum: AceDitGraphQuantum;
}

export interface AceDitGraphCompilationOptions {
  readonly signal?: AbortSignal;
  readonly onProgress?: (progress: AceDitGraphCompilationProgress) => void;
  /** OPT-0018 capture-only metadata; omitted by normal production runs. */
  readonly capturePhysicalCommandDescriptors?: boolean;
  /** Authenticated production GEMM selection; omitted only by isolated tests. */
  readonly gemmConfiguration?: AceDitGemmRuntimeConfiguration;
  readonly denseGemmConfiguration?: AceDitDenseGemmRuntimeConfiguration;
  readonly attentionConfiguration?: AceAttentionRuntimeConfiguration;
  /** OPT-0081 diagnostic only; omission preserves the current FP32 graph. */
  readonly ditDenseInputStorageProfile?: AceDitDenseInputStorageProfile;
  /** Authenticated internal schedule; omission is the exact product eight. */
  readonly samplerScheduleProfile?: Readonly<AceDitSamplerScheduleProfile>;
}

export function planAceDitPhysicalQuantumBatches(
  physicalQuantumCount: number,
  physicalQuantaPerCommandBuffer: AceDitPhysicalQuantaPerCommandBuffer =
    ACE_DIT_PHYSICAL_QUANTA_PER_COMMAND_BUFFER,
): readonly AceDitPhysicalQuantumBatchPlan[] {
  if (!Number.isSafeInteger(physicalQuantumCount) || physicalQuantumCount < 1) {
    throw new RangeError("ACE DiT requires at least one physical graph quantum");
  }
  requireAceDitPhysicalQuantaPerCommandBuffer(
    physicalQuantaPerCommandBuffer,
  );
  const batchCount = Math.ceil(
    physicalQuantumCount / physicalQuantaPerCommandBuffer,
  );
  return Object.freeze(Array.from({ length: batchCount }, (_, batchIndex) => {
    const firstPhysicalIndex =
      batchIndex * physicalQuantaPerCommandBuffer;
    const lastPhysicalIndexExclusive = Math.min(
      physicalQuantumCount,
      firstPhysicalIndex + physicalQuantaPerCommandBuffer,
    );
    return Object.freeze({
      batchIndex,
      firstPhysicalIndex,
      lastPhysicalIndexExclusive,
      physicalQuantumCount:
        lastPhysicalIndexExclusive - firstPhysicalIndex,
    });
  }));
}

export function requireAceDitPhysicalQuantaPerCommandBuffer(
  value: number,
): asserts value is AceDitPhysicalQuantaPerCommandBuffer {
  if (value !== 1 && value !== 8 && value !== 16) {
    throw new RangeError(
      "ACE DiT physical quanta per command buffer must be 1, 8, or 16",
    );
  }
}

function createAceDitPhysicalCommandBatchDescriptor(
  table: AceDitPhysicalCommandDescriptorTable,
  batch: AceDitPhysicalQuantumBatchPlan,
  batchCount: number,
): AceDitPhysicalCommandBatchDescriptor {
  const descriptors = table.descriptors.slice(
    batch.firstPhysicalIndex,
    batch.lastPhysicalIndexExclusive,
  );
  if (
    descriptors.length !== batch.physicalQuantumCount ||
    descriptors.some((descriptor, index) =>
      descriptor.physicalIndex !== batch.firstPhysicalIndex + index
    )
  ) {
    throw new Error(`ACE DiT physical batch ${batch.batchIndex} is incomplete`);
  }
  return Object.freeze({
    batchIndex: batch.batchIndex,
    batchCount,
    firstPhysicalIndex: batch.firstPhysicalIndex,
    lastPhysicalIndex: batch.lastPhysicalIndexExclusive - 1,
    physicalQuantumCount: batch.physicalQuantumCount,
    primitiveCount: descriptors.reduce(
      (total, descriptor) => total + descriptor.primitiveCount,
      0,
    ),
    scheduledMultiplyAdds: descriptors.reduce(
      (total, descriptor) => total + descriptor.scheduledMultiplyAdds,
      0,
    ),
    commandIds: Object.freeze(descriptors.map(({ commandId }) => commandId)),
  });
}

type AceDitPhysicalCommandBatchDescriptorTables = Readonly<{
  readonly 1: readonly AceDitPhysicalCommandBatchDescriptor[];
  readonly 8: readonly AceDitPhysicalCommandBatchDescriptor[];
  readonly 16: readonly AceDitPhysicalCommandBatchDescriptor[];
}>;

function createAceDitPhysicalCommandBatchDescriptorTables(
  table: AceDitPhysicalCommandDescriptorTable,
  physicalQuantumCount: number,
): AceDitPhysicalCommandBatchDescriptorTables {
  const create = (
    physicalQuantaPerCommandBuffer: AceDitPhysicalQuantaPerCommandBuffer,
  ): readonly AceDitPhysicalCommandBatchDescriptor[] => {
    const plans = planAceDitPhysicalQuantumBatches(
      physicalQuantumCount,
      physicalQuantaPerCommandBuffer,
    );
    return Object.freeze(plans.map((batch) =>
      createAceDitPhysicalCommandBatchDescriptor(
        table,
        batch,
        plans.length,
      )
    ));
  };
  return Object.freeze({
    1: create(1),
    8: create(8),
    16: create(16),
  });
}

export const ACE_DIT_PROFILE_FAMILIES = [
  "precompute",
  "cross-cache",
  "timestep",
  "input",
  "attention-projections",
  "self-full",
  "self-sliding",
  "cross-attention",
  "feed-forward",
  "plumbing",
  "output",
  "sampler-dcw",
  "mixed",
] as const;

export type AceDitProfileFamily =
  (typeof ACE_DIT_PROFILE_FAMILIES)[number];

export interface AceDitPhysicalCommandMemberDescriptor {
  readonly id: string;
  readonly family: Exclude<AceDitProfileFamily, "mixed">;
  readonly backend: string;
  readonly kernel: string;
  readonly scheduledMultiplyAdds: number;
}

export interface AceDitPhysicalCommandDescriptor {
  readonly physicalIndex: number;
  readonly logicalIndex: number;
  readonly logicalKind: AceDitGraphQuantum["kind"];
  readonly commandId: string;
  readonly subquantumIndex: number;
  readonly subquantumCount: number;
  readonly evaluation: number | null;
  readonly layer: number | null;
  readonly family: AceDitProfileFamily;
  readonly primitiveCount: number;
  readonly scheduledMultiplyAdds: number;
  readonly members: readonly AceDitPhysicalCommandMemberDescriptor[];
}

export interface AceDitPhysicalCommandDescriptorTable {
  readonly descriptors: readonly AceDitPhysicalCommandDescriptor[];
  readonly sha256: string;
  readonly serializedBytes: number;
  readonly memberCount: number;
  readonly preparationMs: number;
}

export interface AceDitPhysicalCommandBatchDescriptor {
  readonly batchIndex: number;
  readonly batchCount: number;
  readonly firstPhysicalIndex: number;
  readonly lastPhysicalIndex: number;
  readonly physicalQuantumCount: number;
  readonly primitiveCount: number;
  readonly scheduledMultiplyAdds: number;
  readonly commandIds: readonly string[];
}

export interface AceDitPhysicalQuantumBatchPlan {
  readonly batchIndex: number;
  readonly firstPhysicalIndex: number;
  readonly lastPhysicalIndexExclusive: number;
  readonly physicalQuantumCount: number;
}

export interface AceDitGraphModel {
  readonly modelProfile: AceModelProfileId;
  readonly residentBytes: number;
  readonly weights: AceDitResolvedPackageWeights;
  destroy(): void;
}

export interface AceLoadDitResidentModelOptions {
  readonly signal?: AbortSignal;
  readonly onProgress?: (progress: AceGpuTensorPhaseProgress) => void;
}

/** Authenticated DiT package phase whose ownership can move into the graph. */
export class AceDitResidentModel implements AceDitGraphModel {
  readonly weights: AceDitResolvedPackageWeights;
  readonly residentBytes: number;
  private destroyed = false;

  private constructor(
    readonly modelProfile: AceModelProfileId,
    private readonly phase: AceGpuTensorPhase,
    private readonly densePhase?: AceGpuTensorPhase,
    denseRuntimeProfile?: AceDitDenseRuntimeProfile,
  ) {
    this.residentBytes = phase.residentBytes + (densePhase?.residentBytes ?? 0);
    try {
      this.weights = resolveAceDitPackageWeights(
        phase,
        modelProfile,
        densePhase,
        denseRuntimeProfile,
      );
      if (
        densePhase !== undefined &&
        (phase.residentBytes !== ACE_REFERENCE_DIT_SHARED_WEIGHT_BYTES ||
          densePhase.residentBytes !== ACE_OPT_0037_DIT_K4_LAYER_BYTES ||
          this.residentBytes !== ACE_OPT_0037_DIT_K4_RESIDENT_WEIGHT_BYTES)
      ) {
        throw new Error(
          "ACE mixed DiT physical resident-byte replacement contract changed",
        );
      }
    } catch (error) {
      phase.destroy();
      densePhase?.destroy();
      this.destroyed = true;
      throw error;
    }
  }

  static async load(
    device: GPUDevice,
    manifest: AcePackageManifest,
    acquiredFiles: ReadonlyMap<string, File>,
    modelProfile: AceModelProfileId,
    options: AceLoadDitResidentModelOptions = {},
  ): Promise<AceDitResidentModel> {
    const phase = await AceGpuTensorPhase.load(
      device,
      manifest,
      acquiredFiles,
      ["dit"],
      options,
    );
    return new AceDitResidentModel(modelProfile, phase);
  }

  /**
   * Consume an already authenticated phase without reloading its shards.
   * Ownership transfers at call entry, including when phase/profile
   * validation fails.
   */
  static take(
    ownedPhase: AceGpuTensorPhase,
    modelProfile: AceModelProfileId,
  ): AceDitResidentModel {
    if (
      ownedPhase.phases.length !== 1 ||
      ownedPhase.phases[0] !== "dit"
    ) {
      ownedPhase.destroy();
      throw new Error(
        "ACE DiT resident model requires an exclusively resident dit phase",
      );
    }
    return new AceDitResidentModel(modelProfile, ownedPhase);
  }

  /** Consume reference shared tensors and one exact authenticated mixed profile. */
  static takeMixed(
    ownedReferencePhase: AceGpuTensorPhase,
    ownedDensePhase: AceGpuTensorPhase,
    modelProfile: AceModelProfileId,
    denseRuntimeProfile: AceDitDenseRuntimeProfile,
  ): AceDitResidentModel {
    if (
      modelProfile !== "reference-bf16" ||
      ownedReferencePhase.phases.length !== 1 ||
      ownedReferencePhase.phases[0] !== "dit" ||
      ownedReferencePhase.packageManifest.profile !== "reference" ||
      ownedDensePhase.phases.length !== 1 ||
      ownedDensePhase.phases[0] !== "dit" ||
      ownedDensePhase.packageManifest.profile !==
        "fp16-dit-dense-experimental" ||
      (denseRuntimeProfile !== ACE_OPT_0009_DIT_DENSE_RUNTIME_PROFILE &&
        denseRuntimeProfile !== ACE_OPT_0037_DIT_K4_RUNTIME_PROFILE &&
        denseRuntimeProfile !== ACE_OPT_0056_DIT_SELECTIVE_K4_RUNTIME_PROFILE)
    ) {
      ownedReferencePhase.destroy();
      ownedDensePhase.destroy();
      throw new Error(
        "ACE mixed DiT requires exclusive reference and authenticated dense dit phases",
      );
    }
    return new AceDitResidentModel(
      modelProfile,
      ownedReferencePhase,
      ownedDensePhase,
      denseRuntimeProfile,
    );
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.phase.destroy();
    this.densePhase?.destroy();
  }
}

type AceDitGraphRuntime = Pick<
  AceCorrectnessDitRuntime,
  | "createConditionProjectionDispatch"
  | "createCrossCacheDispatch"
  | "createTimestepDispatch"
  | "createInputProjectionDispatch"
  | "createLayerDispatch"
  | "createOutputProjectionDispatch"
  | "createSamplerStepDispatch"
  | "destroy"
> & Partial<Pick<
  AceCorrectnessDitRuntime,
  "finalizeDenseRoutes" | "finalizeAttentionRoutes"
>> & Readonly<{
  denseInputStorageProfile?: AceDitDenseInputStorageProfile | undefined;
}>;

interface AceCompiledDitGraph {
  readonly logicalCommands: readonly AceCompiledDitCommand[];
  readonly commandBufferCount: number;
  readonly finalLatent: GPUBufferBinding;
  readonly evaluation0Velocity: GPUBufferBinding;
  readonly evaluationLatents: readonly GPUBufferBinding[];
  readonly denseRouteProfile?: AceOpt0056DenseRouteProfile;
  readonly attentionRouteProfile?:
    | AceOpt0062AttentionRouteProfile
    | AceOpt0070ProductionAttentionRouteProfile;
  readonly opt0062AttentionIdentityCounters?: GPUBufferBinding;
  readonly physicalCommandDescriptors?: AceDitPhysicalCommandDescriptorTable;
}

interface AceCompiledDitCommand {
  readonly quantum: AceDitGraphQuantum;
  readonly sequence: AceGpuEncodeSequence;
}

/**
 * Single-use owner for the full pinned denoise graph. It consumes the resident
 * model on construction and destroys it only after the last submitted quantum
 * has drained (also on cancellation or failure).
 */
export class AceDitGraphOwner {
  readonly shape: AceDitEvaluationPlan;
  readonly memoryPlan: AceDitGraphMemoryPlan;
  readonly quanta: readonly AceDitGraphQuantum[];
  readonly commandBufferCount: number;
  readonly physicalCommandDescriptors:
    | AceDitPhysicalCommandDescriptorTable
    | undefined;

  private readonly scheduler: AceCooperativeGpuScheduler;
  private readonly physicalCommandBatchDescriptors:
    | AceDitPhysicalCommandBatchDescriptorTables
    | undefined;
  private state: "ready" | "running" | "finished" | "destroyed" = "ready";
  private resourcesReleased = false;

  private constructor(
    private readonly device: GPUDevice,
    private readonly model: AceDitGraphModel,
    private readonly runtime: AceDitGraphRuntime,
    private readonly compiled: AceCompiledDitGraph,
    private readonly samplerScheduleProfile:
      Readonly<AceDitSamplerScheduleProfile>,
    shape: AceDitEvaluationPlan,
    memoryPlan: AceDitGraphMemoryPlan,
    scheduler = new AceCooperativeGpuScheduler(),
  ) {
    this.shape = shape;
    this.memoryPlan = memoryPlan;
    this.quanta = createAceDitGraphQuantumPlan(samplerScheduleProfile);
    this.commandBufferCount = compiled.commandBufferCount;
    this.physicalCommandDescriptors = compiled.physicalCommandDescriptors;
    this.physicalCommandBatchDescriptors =
      compiled.physicalCommandDescriptors === undefined
        ? undefined
        : createAceDitPhysicalCommandBatchDescriptorTables(
            compiled.physicalCommandDescriptors,
            compiled.commandBufferCount,
          );
    this.scheduler = scheduler;
  }

  static async create(
    device: GPUDevice,
    model: AceDitGraphModel,
    shape: AceDitEvaluationShape,
    dcwConfiguration: AceDynamicConditionalWeightingConfiguration,
    bindings: AceDitGraphBindings,
    compilation: AceDitGraphCompilationOptions & Readonly<{
      gemmConfiguration: AceDitGemmRuntimeConfiguration;
      denseGemmConfiguration: AceDitDenseGemmRuntimeConfiguration;
      attentionConfiguration: AceAttentionRuntimeConfiguration;
    }>,
  ): Promise<AceDitGraphOwner> {
    let runtime: AceCorrectnessDitRuntime;
    try {
      compilation.signal?.throwIfAborted();
      runtime = AceCorrectnessDitRuntime.create(
        device,
        model.modelProfile,
        compilation.gemmConfiguration,
        compilation.denseGemmConfiguration,
        compilation.attentionConfiguration,
        compilation.ditDenseInputStorageProfile,
      );
    } catch (error) {
      model.destroy();
      throw error;
    }
    return await this.createWithRuntime(
      device,
      model,
      runtime,
      shape,
      dcwConfiguration,
      bindings,
      compilation,
    );
  }

  /** @internal Structural runtime injection used by graph-order contract tests. */
  static async createWithRuntime(
    device: GPUDevice,
    model: AceDitGraphModel,
    runtime: AceDitGraphRuntime,
    shape: AceDitEvaluationShape,
    dcwConfiguration: AceDynamicConditionalWeightingConfiguration,
    bindings: AceDitGraphBindings,
    compilation: AceDitGraphCompilationOptions = {},
    scheduler = new AceCooperativeGpuScheduler(),
  ): Promise<AceDitGraphOwner> {
    let transferred = false;
    try {
      const plan = requirePinnedGraphShape(shape);
      const denseInputStorageProfile = requireAceDitDenseInputStorageProfile(
        compilation.ditDenseInputStorageProfile,
      );
      if (runtime.denseInputStorageProfile !== denseInputStorageProfile) {
        throw new Error(
          "ACE DiT runtime and dense-input storage profile diverged",
        );
      }
      if (
        denseInputStorageProfile !== undefined &&
        (compilation.capturePhysicalCommandDescriptors !== undefined ||
          compilation.gemmConfiguration?.backend !== "subgroups" ||
          compilation.gemmConfiguration.capability.subgroupMinSize !== 32 ||
          compilation.gemmConfiguration.capability.subgroupMaxSize !== 32 ||
          compilation.denseGemmConfiguration?.backend !==
            "opt-0009-fp16-fp32" ||
          compilation.denseGemmConfiguration.capability.subgroupMinSize !== 32 ||
          compilation.denseGemmConfiguration.capability.subgroupMaxSize !== 32 ||
          compilation.attentionConfiguration?.backend !==
            "opt-0070-fixed32-quad-query32-full-self-production" ||
          compilation.attentionConfiguration.capability.subgroupMinSize !== 32 ||
          compilation.attentionConfiguration.capability.subgroupMaxSize !== 32 ||
          compilation.attentionConfiguration.expectedQueryTokens !== 2_250 ||
          compilation.attentionConfiguration.expectedConditionTokens !== 98)
      ) {
        throw new Error(
          "OPT-0081 dense-input graph storage requires matching fixed32 " +
            "OPT-0009/OPT-0070 compilation without legacy descriptor capture",
        );
      }
      const samplerScheduleProfile = requireResolvedAceDitSamplerScheduleProfile(
        compilation.samplerScheduleProfile ??
          ACE_TURBO_EIGHT_SAMPLER_SCHEDULE_PROFILE,
      );
      const memoryPlan = planAceDitGraphMemory(
        model.modelProfile,
        shape,
        samplerScheduleProfile,
        denseInputStorageProfile,
      );
      validateAceDitGraphBindings(
        model.modelProfile,
        plan,
        model.weights,
        bindings,
        samplerScheduleProfile,
        denseInputStorageProfile,
      );
      const compiled = await compileAceDitGraph(
        runtime,
        model.weights,
        plan,
        dcwConfiguration,
        bindings,
        compilation,
      );
      transferred = true;
      return new AceDitGraphOwner(
        device,
        model,
        runtime,
        compiled,
        samplerScheduleProfile,
        plan,
        memoryPlan,
        scheduler,
      );
    } finally {
      if (!transferred) {
        runtime.destroy();
        model.destroy();
        await scheduler.dispose();
      }
    }
  }

  async run(options: AceDitGraphRunOptions): Promise<AceDitGraphRunResult> {
    if (this.state !== "ready") {
      throw new Error(`ACE DiT graph cannot run from state ${this.state}`);
    }
    this.state = "running";
    let completedEvaluations = 0;
    try {
      options.signal.throwIfAborted();
      const physicalQuantaPerCommandBuffer =
        options.physicalQuantaPerCommandBuffer ??
        ACE_DIT_PHYSICAL_QUANTA_PER_COMMAND_BUFFER;
      requireAceDitPhysicalQuantaPerCommandBuffer(
        physicalQuantaPerCommandBuffer,
      );
      const opt0080SchedulingProfile = options.opt0080SchedulingProfile;
      const opt0080FullGraph = options.opt0080FullGraph === true;
      const submissionPolicy = options.submissionPolicy ??
        (opt0080SchedulingProfile === "opt-0080-depth2-epoch4"
          ? "depth2-phase-epoch4"
          : "depth1-epoch1");
      if (
        opt0080SchedulingProfile !== undefined &&
        opt0080SchedulingProfile !== "depth1-epoch1" &&
        opt0080SchedulingProfile !== "opt-0080-depth2-epoch4"
      ) {
        throw new RangeError("Unknown OPT-0080 DiT scheduling profile");
      }
      if (
        submissionPolicy !== "depth1-epoch1" &&
        submissionPolicy !== "depth2-phase-epoch4"
      ) {
        throw new RangeError("Unknown ACE DiT submission policy");
      }
      if (
        opt0080SchedulingProfile !== undefined &&
        (submissionPolicy === "depth2-phase-epoch4") !==
          (opt0080SchedulingProfile === "opt-0080-depth2-epoch4")
      ) {
        throw new Error(
          "OPT-0080 capture profile disagrees with the DiT submission policy",
        );
      }
      const hasOpt0080Callbacks =
        options.onOpt0080CommandBufferCompleted !== undefined ||
        options.onOpt0080CompletionEpochDrained !== undefined ||
        options.onOpt0080PhaseStarted !== undefined;
      if (hasOpt0080Callbacks && opt0080SchedulingProfile === undefined) {
        throw new Error(
          "OPT-0080 scheduling callbacks require an OPT-0080 scheduling profile",
        );
      }
      if (opt0080FullGraph && opt0080SchedulingProfile === undefined) {
        throw new Error(
          "OPT-0080 full-graph execution requires an OPT-0080 scheduling profile",
        );
      }
      const evaluationReadbacks = options.evaluationReadbacks;
      const evaluation0VelocityReadback =
        options.evaluation0VelocityReadback;
      const identityReadback = options.opt0062AttentionIdentityReadback;
      const identityCounters = this.compiled.opt0062AttentionIdentityCounters;
      const evaluationTarget = options.opt0067EvaluationLimit ??
        this.samplerScheduleProfile.evaluationCount;
      const graphCommandBufferTarget = evaluationTarget === 1
        ? ACE_OPT_0067_M2250_EVALUATION0_GRAPH_COMMAND_BUFFER_COUNT
        : this.compiled.commandBufferCount;
      const phaseCommandBufferCounts =
        planAceDitPhaseCommandBufferCounts(
          this.compiled.logicalCommands,
          evaluationTarget,
        );
      if (
        phaseCommandBufferCounts.reduce((sum, count) => sum + count, 0) !==
          graphCommandBufferTarget
      ) {
        throw new Error("ACE DiT phase command counts do not cover the graph");
      }
      const phaseFirstCommandBufferIndices = (() => {
        let first = 0;
        return phaseCommandBufferCounts.map((count) => {
          const current = first;
          first += count;
          return current;
        });
      })();
      if (
        evaluationTarget === 1 &&
        (physicalQuantaPerCommandBuffer !== 1 ||
          this.compiled.commandBufferCount !==
            ACE_OPT_0018_M2250_GRAPH_COMMAND_BUFFER_COUNT ||
          evaluationReadbacks !== undefined ||
          evaluation0VelocityReadback !== undefined)
      ) {
        throw new Error(
          "OPT-0067 evaluation 0 requires the exact M2250 graph, one physical quantum per command buffer, and no sampler snapshots",
        );
      }
      if (
        opt0080SchedulingProfile !== undefined &&
        ((opt0080FullGraph
          ? options.opt0067EvaluationLimit !== undefined ||
            evaluationTarget !== this.samplerScheduleProfile.evaluationCount ||
            evaluationTarget !== 8 ||
            graphCommandBufferTarget !==
              ACE_OPT_0018_M2250_GRAPH_COMMAND_BUFFER_COUNT
          : options.opt0067EvaluationLimit !== 1 ||
            evaluationTarget !== 1 ||
            graphCommandBufferTarget !==
              ACE_OPT_0067_M2250_EVALUATION0_GRAPH_COMMAND_BUFFER_COUNT) ||
          physicalQuantaPerCommandBuffer !== 1 ||
          (!opt0080FullGraph && evaluationReadbacks !== undefined) ||
          evaluation0VelocityReadback !== undefined ||
          identityReadback !== undefined)
      ) {
        throw new Error(
          opt0080FullGraph
            ? "OPT-0080 full scheduling requires the exact eight-evaluation M2250 graph"
            : "OPT-0080 scheduling requires the exact M2250 evaluation-0 graph without diagnostic GPU copies",
        );
      }
      if (
        submissionPolicy === "depth2-phase-epoch4" &&
        physicalQuantaPerCommandBuffer !== 1
      ) {
        throw new Error(
          "Depth-two DiT submission requires singleton physical command buffers",
        );
      }
      if (
        evaluationReadbacks !== undefined &&
        (physicalQuantaPerCommandBuffer !== 1 ||
          evaluationReadbacks.length !==
            this.samplerScheduleProfile.evaluationCount ||
          this.compiled.evaluationLatents.length !==
            this.samplerScheduleProfile.evaluationCount ||
          evaluationReadbacks.some((buffer) =>
            buffer.size < this.memoryPlan.latentBytes
          ))
      ) {
        throw new Error(
          "ACE sampler snapshots require one bounded readback per evaluation and one physical quantum per command buffer",
        );
      }
      if (
        evaluation0VelocityReadback !== undefined &&
        (evaluationReadbacks === undefined ||
          physicalQuantaPerCommandBuffer !== 1 ||
          evaluation0VelocityReadback.size < this.memoryPlan.latentBytes)
      ) {
        throw new Error(
          "ACE sampler evidence requires bounded trajectory and evaluation-0 velocity readbacks",
        );
      }
      if (
        (identityReadback === undefined) !== (identityCounters === undefined) ||
        (identityReadback !== undefined &&
          (physicalQuantaPerCommandBuffer !== 1 ||
            identityReadback.size < ACE_OPT_0062_IDENTITY_COUNTER_BYTES ||
            (identityCounters!.size ??
              identityCounters!.buffer.size -
                (identityCounters!.offset ?? 0)) <
              ACE_OPT_0062_IDENTITY_COUNTER_BYTES))
      ) {
        throw new Error(
          "OPT-0062 identity capture requires exact counters, readback, and one physical quantum per command buffer",
        );
      }
      if (
        options.onCommandBufferDrained !== undefined &&
        physicalQuantaPerCommandBuffer !== 1
      ) {
        throw new Error(
          "ACE DiT per-command attribution requires one physical quantum per command buffer",
        );
      }
      if (
        opt0080SchedulingProfile !== undefined &&
        (options.onCommandBufferDrained !== undefined ||
          options.onPhysicalBatchDrained !== undefined)
      ) {
        throw new Error(
          "OPT-0080 completion-fence attribution cannot use queue-drain callbacks",
        );
      }
      if (
        (options.onCommandBufferDrained !== undefined ||
          options.onPhysicalBatchDrained !== undefined ||
          options.onOpt0080CommandBufferCompleted !== undefined) &&
        this.physicalCommandDescriptors === undefined
      ) {
        throw new Error(
          "ACE DiT completion attribution requires captured physical descriptors",
        );
      }
      const batches = planAceDitPhysicalQuantumBatches(
        graphCommandBufferTarget,
        physicalQuantaPerCommandBuffer,
      );
      const batchDescriptors = options.onPhysicalBatchDrained === undefined
        ? undefined
        : this.physicalCommandBatchDescriptors?.[
            physicalQuantaPerCommandBuffer
          ];
      if (
        options.onPhysicalBatchDrained !== undefined &&
        (batchDescriptors === undefined ||
          batchDescriptors.length !== batches.length)
      ) {
        throw new Error("ACE DiT physical batch descriptors were not prepared");
      }
      let logicalIndex = 0;
      let subquantumIndex = 0;
      let createdIndex = -1;
      let encodedPhysicalQuanta = 0;
      let encodedEvaluationSnapshots = 0;
      let encodedEvaluation0VelocityCopies = 0;
      let encodedAttentionIdentityCopies = 0;
      type PreparedBatch = Readonly<{
        batch: AceDitPhysicalQuantumBatchPlan;
        batchCommandId: string;
        entries: readonly Readonly<{
          command: AceCompiledDitCommand;
          subquantumIndex: number;
          commandId: string;
        }>[];
      }>;
      let active: PreparedBatch | undefined;
      const pending = new Map<number, PreparedBatch>();
      const usesCompletionAttribution =
        opt0080SchedulingProfile !== undefined ||
        submissionPolicy === "depth2-phase-epoch4";
      let maximumPendingDescriptorCount = 0;
      let completedDiagnosticIndex = -1;
      const createCommandBuffer = (index: number): GPUCommandBuffer => {
          if (opt0080SchedulingProfile === "depth1-epoch1") {
            const phaseIndex = phaseFirstCommandBufferIndices.indexOf(
              index,
            );
            if (phaseIndex >= 0) {
              options.onOpt0080PhaseStarted?.(Object.freeze({
                phaseIndex,
                firstCommandBufferIndex: index,
                commandBufferCount:
                  phaseCommandBufferCounts[phaseIndex]!,
              }));
            }
          }
          if (index !== createdIndex + 1) {
            throw new Error("ACE DiT scheduler requested work out of FIFO order");
          }
          const batch = batches[index];
          if (
            batch === undefined ||
            batch.firstPhysicalIndex !== encodedPhysicalQuanta
          ) throw new Error("ACE DiT scheduler lost physical FIFO position");
          const entries: Array<Readonly<{
            command: AceCompiledDitCommand;
            subquantumIndex: number;
            commandId: string;
          }>> = [];
          const firstCommand = this.compiled.logicalCommands[logicalIndex];
          if (firstCommand === undefined) {
            throw new Error("ACE DiT scheduler exceeded the compiled graph");
          }
          const firstCommandId = firstCommand.sequence.quantumCount === 1
            ? firstCommand.quantum.label
            : `${firstCommand.quantum.label}-part-${subquantumIndex}`;
          const batchCommandId = batch.physicalQuantumCount === 1
            ? firstCommandId
            : `ace-dit-physical-${batch.firstPhysicalIndex}-through-${batch.lastPhysicalIndexExclusive - 1}`;
          const encoder = this.device.createCommandEncoder({
            label: `${batchCommandId}-encoder`,
          });
          const pass = encoder.beginComputePass({
            label: `${batchCommandId}-pass`,
          });
          for (
            let physicalIndex = batch.firstPhysicalIndex;
            physicalIndex < batch.lastPhysicalIndexExclusive;
            physicalIndex += 1
          ) {
            options.signal.throwIfAborted();
            const command = this.compiled.logicalCommands[logicalIndex];
            if (command === undefined) {
              throw new Error("ACE DiT scheduler exceeded the compiled graph");
            }
            const activeSubquantumIndex = subquantumIndex;
            const commandId = command.sequence.quantumCount === 1
              ? command.quantum.label
              : `${command.quantum.label}-part-${activeSubquantumIndex}`;
            command.sequence.encodeQuantum(pass, activeSubquantumIndex);
            entries.push(Object.freeze({
              command,
              subquantumIndex: activeSubquantumIndex,
              commandId,
            }));
            subquantumIndex += 1;
            encodedPhysicalQuanta += 1;
            if (subquantumIndex === command.sequence.quantumCount) {
              logicalIndex += 1;
              subquantumIndex = 0;
            }
          }
          pass.end();
          const samplerFinals = entries.filter(({ command, subquantumIndex }) =>
            command.quantum.kind === "sampler" &&
            subquantumIndex + 1 === command.sequence.quantumCount
          );
          if (evaluationReadbacks !== undefined && samplerFinals.length !== 0) {
            if (samplerFinals.length !== 1) {
              throw new Error(
                "OPT-0056 snapshot batch crossed more than one sampler boundary",
              );
            }
            const quantum = samplerFinals[0]!.command.quantum;
            if (
              quantum.kind !== "sampler" ||
              quantum.evaluation !== encodedEvaluationSnapshots
            ) {
              throw new Error("OPT-0056 sampler snapshot order changed");
            }
            const source = this.compiled.evaluationLatents[quantum.evaluation];
            const target = evaluationReadbacks[quantum.evaluation];
            if (source === undefined || target === undefined) {
              throw new Error("OPT-0056 sampler snapshot binding is missing");
            }
            const sourceOffset = source.offset ?? 0;
            const sourceBytes = source.size ?? source.buffer.size - sourceOffset;
            if (sourceBytes < this.memoryPlan.latentBytes) {
              throw new Error("OPT-0056 sampler snapshot source is undersized");
            }
            encoder.copyBufferToBuffer(
              source.buffer,
              sourceOffset,
              target,
              0,
              this.memoryPlan.latentBytes,
            );
            encodedEvaluationSnapshots += 1;
          }
          if (
            evaluation0VelocityReadback !== undefined &&
            samplerFinals.length !== 0
          ) {
            const quantum = samplerFinals[0]!.command.quantum;
            if (quantum.kind === "sampler" && quantum.evaluation === 0) {
              const source = this.compiled.evaluation0Velocity;
              const sourceOffset = source.offset ?? 0;
              const sourceBytes = source.size ??
                source.buffer.size - sourceOffset;
              if (sourceBytes < this.memoryPlan.latentBytes) {
                throw new Error(
                  "ACE evaluation-0 velocity snapshot source is undersized",
                );
              }
              encoder.copyBufferToBuffer(
                source.buffer,
                sourceOffset,
                evaluation0VelocityReadback,
                0,
                this.memoryPlan.latentBytes,
              );
              encodedEvaluation0VelocityCopies += 1;
            }
          }
          if (identityReadback !== undefined && samplerFinals.length !== 0) {
            const quantum = samplerFinals[0]!.command.quantum;
            if (
              quantum.kind === "sampler" &&
              quantum.evaluation === evaluationTarget - 1
            ) {
              const sourceOffset = identityCounters!.offset ?? 0;
              encoder.copyBufferToBuffer(
                identityCounters!.buffer,
                sourceOffset,
                identityReadback,
                0,
                ACE_OPT_0062_IDENTITY_COUNTER_BYTES,
              );
              encodedAttentionIdentityCopies += 1;
            }
          }
          const prepared = Object.freeze({
            batch,
            batchCommandId,
            entries: Object.freeze(entries),
          });
          if (!usesCompletionAttribution) {
            active = prepared;
          } else {
            const pendingLimit = submissionPolicy === "depth2-phase-epoch4"
              ? 2
              : 1;
            if (pending.has(index) || pending.size >= pendingLimit) {
              throw new Error(
                `OPT-0080 command ${index} exceeded its bounded pending descriptor map`,
              );
            }
            const lastPendingIndex = [...pending.keys()].at(-1);
            if (
              lastPendingIndex !== undefined &&
              index !== lastPendingIndex + 1
            ) {
              throw new Error(
                `OPT-0080 command ${index} lost pending descriptor FIFO order`,
              );
            }
            pending.set(index, prepared);
            maximumPendingDescriptorCount = Math.max(
              maximumPendingDescriptorCount,
              pending.size,
            );
          }
          createdIndex = index;
          return encoder.finish({ label: `${batchCommandId}-commands` });
        };
      const completeOpt0080CommandBuffer = (
        timing: AceDepth2Epoch4CommandBufferCompletionTiming,
        schedulingProgress: AceDepth2Epoch4SchedulingProgress,
      ): void => {
        const index = timing.commandBufferIndex;
        const firstPendingIndex = pending.keys().next().value as
          | number
          | undefined;
        const prepared = pending.get(index);
        if (
          index !== completedDiagnosticIndex + 1 ||
          firstPendingIndex !== index ||
          prepared === undefined ||
          prepared.batch.batchIndex !== index ||
          schedulingProgress.completedCommandBuffers !== index + 1 ||
          schedulingProgress.totalCommandBuffers !== batches.length
        ) {
          throw new Error(
            `OPT-0080 command ${index} lost immutable FIFO completion attribution`,
          );
        }
        for (const entry of prepared.entries) {
          if (
            entry.command.quantum.kind === "sampler" &&
            entry.subquantumIndex + 1 === entry.command.sequence.quantumCount
          ) completedEvaluations += 1;
        }
        const last = prepared.entries.at(-1);
        if (last === undefined) {
          throw new Error(`OPT-0080 command ${index} completed an empty batch`);
        }
        const { command, commandId } = last;
        const graphProgress: AceDitGraphProgress = Object.freeze({
          completedQuanta: prepared.batch.lastPhysicalIndexExclusive,
          totalQuanta: graphCommandBufferTarget,
          completedCommandBuffers: schedulingProgress.completedCommandBuffers,
          totalCommandBuffers: schedulingProgress.totalCommandBuffers,
          queueDrains: schedulingProgress.trueQueueDrainCount,
          cooperativeIdleMs:
            schedulingProgress.requestedCooperativeIdleMs,
          completedEvaluations,
          totalEvaluations: this.samplerScheduleProfile.evaluationCount,
          quantum: command.quantum,
          subquantumIndex: last.subquantumIndex,
          subquantumCount: command.sequence.quantumCount,
          commandId,
          batchIndex: prepared.batch.batchIndex,
          batchFirstPhysicalQuantum: prepared.batch.firstPhysicalIndex,
          batchPhysicalQuantumCount: prepared.batch.physicalQuantumCount,
          physicalQuantaPerCommandBuffer,
        });
        options.onProgress?.(graphProgress);
        options.signal.throwIfAborted();
        if (options.onOpt0080CommandBufferCompleted !== undefined) {
          const descriptor = this.physicalCommandDescriptors!
            .descriptors[prepared.batch.firstPhysicalIndex];
          if (
            descriptor === undefined ||
            descriptor.physicalIndex !== index
          ) {
            throw new Error(
              `OPT-0080 command ${index} omitted its immutable physical descriptor`,
            );
          }
          options.onOpt0080CommandBufferCompleted(Object.freeze({
            ...timing,
            descriptor,
            schedulingProgress,
            graphProgress,
          }));
        }
        if (!pending.delete(index)) {
          throw new Error(
            `OPT-0080 command ${index} failed to retire its pending descriptor`,
          );
        }
        completedDiagnosticIndex = index;
      };

      let scheduled: AceGpuSchedulingResult;
      let opt0080Scheduling: AceDitOpt0080SchedulingDiagnostics | undefined;
      try {
        if (submissionPolicy === "depth2-phase-epoch4") {
          const candidate = await this.scheduler.runLazyDepth2Epoch4({
            queue: this.device.queue,
            commandBufferCount: batches.length,
            phaseCommandBufferCounts:
              phaseCommandBufferCounts,
            createCommandBuffer,
            signal: options.signal,
            device: this.device,
            ...(options.yieldQueueIdle === undefined
              ? {}
              : { yieldQueueIdle: options.yieldQueueIdle }),
            onCommandBufferCompleted: completeOpt0080CommandBuffer,
            ...(options.onOpt0080CompletionEpochDrained === undefined
              ? {}
              : {
                  onCompletionEpochDrained:
                    options.onOpt0080CompletionEpochDrained,
                }),
            ...(options.onOpt0080PhaseStarted === undefined
              ? {}
              : { onPhaseStarted: options.onOpt0080PhaseStarted }),
          });
          scheduled = candidate;
          if (opt0080SchedulingProfile !== undefined) {
            opt0080Scheduling = Object.freeze({
              profile: opt0080SchedulingProfile,
              completionFenceRequestedCount:
                candidate.completionFenceRequestedCount,
              completionFenceSettledCount:
                candidate.completionFenceSettledCount,
              completionFenceRejectedCount:
                candidate.completionFenceRejectedCount,
              trueQueueDrainCount: candidate.trueQueueDrainCount,
              completionEpochCount: candidate.completionEpochCount,
              requestedCooperativeIdleMs:
                candidate.requestedCooperativeIdleMs,
              cooperativeIdleTurns: candidate.cooperativeIdleTurns,
              maximumOutstandingCommandBuffers:
                candidate.maximumOutstandingCommandBuffers,
              maximumPendingDescriptorCount,
            });
          }
        } else if (opt0080SchedulingProfile === "depth1-epoch1") {
          scheduled = await this.scheduler.runLazy({
            queue: this.device.queue,
            commandBufferCount: batches.length,
            createCommandBuffer,
            signal: options.signal,
            ...(options.yieldQueueIdle === undefined
              ? {}
              : { yieldQueueIdle: options.yieldQueueIdle }),
            onCommandBufferDrained: (timing) => {
              const index = timing.commandBufferIndex;
              const cooperativeIdleTurns = Math.min(
                index + 1,
                batches.length - 1,
              );
              completeOpt0080CommandBuffer(Object.freeze({
                commandBufferIndex: index,
                submitThroughCompletionFenceMs: timing.submitThroughDrainMs,
                trueQueueDrain: true,
                completionEpochIndex: index,
              }), Object.freeze({
                completedCommandBuffers: index + 1,
                totalCommandBuffers: batches.length,
                completionFenceRequestedCount: index + 1,
                completionFenceSettledCount: index + 1,
                completionFenceRejectedCount: 0,
                trueQueueDrainCount: index + 1,
                completionEpochCount: index + 1,
                requestedCooperativeIdleMs: cooperativeIdleTurns,
                cooperativeIdleTurns,
                outstandingCommandBuffers: 0,
              }));
              options.onOpt0080CompletionEpochDrained?.(Object.freeze({
                completionEpochIndex: index,
                phaseIndex: phaseFirstCommandBufferIndices.findLastIndex(
                  (first) => first <= index,
                ),
                firstCommandBufferIndex: index,
                lastCommandBufferIndex: index,
                commandBufferCount: 1,
                submitThroughTrueDrainMs: timing.submitThroughDrainMs,
              }));
            },
          });
          opt0080Scheduling = Object.freeze({
            profile: opt0080SchedulingProfile,
            completionFenceRequestedCount: batches.length,
            completionFenceSettledCount: batches.length,
            completionFenceRejectedCount: 0,
            trueQueueDrainCount: batches.length,
            completionEpochCount: batches.length,
            requestedCooperativeIdleMs: batches.length - 1,
            cooperativeIdleTurns: batches.length - 1,
            maximumOutstandingCommandBuffers: 1,
            maximumPendingDescriptorCount,
          });
        } else {
          scheduled = await this.scheduler.runLazy({
            queue: this.device.queue,
            commandBufferCount: batches.length,
            createCommandBuffer,
            signal: options.signal,
            ...(options.yieldQueueIdle === undefined
              ? {}
              : { yieldQueueIdle: options.yieldQueueIdle }),
            ...(options.onCommandBufferDrained === undefined &&
                options.onPhysicalBatchDrained === undefined
              ? {}
              : {
                  onCommandBufferDrained: (timing) => {
                    if (
                      active === undefined ||
                      active.batch.batchIndex !== timing.commandBufferIndex
                    ) {
                      throw new Error(
                        `ACE DiT batch ${timing.commandBufferIndex} lost its active descriptor`,
                      );
                    }
                    if (options.onCommandBufferDrained !== undefined) {
                      const descriptor = this.physicalCommandDescriptors!
                        .descriptors[active.batch.firstPhysicalIndex];
                      if (descriptor === undefined) {
                        throw new Error(
                          `ACE DiT command ${active.batch.firstPhysicalIndex} omitted its capture descriptor`,
                        );
                      }
                      options.onCommandBufferDrained(
                        descriptor,
                        timing.submitThroughDrainMs,
                      );
                    }
                    if (options.onPhysicalBatchDrained !== undefined) {
                      const descriptor = batchDescriptors?.[
                        active.batch.batchIndex
                      ];
                      if (
                        descriptor === undefined ||
                        descriptor.firstPhysicalIndex !==
                          active.batch.firstPhysicalIndex ||
                        descriptor.lastPhysicalIndex !==
                          active.batch.lastPhysicalIndexExclusive - 1
                      ) {
                        throw new Error(
                          `ACE DiT batch ${active.batch.batchIndex} lost its prepared descriptor`,
                        );
                      }
                      options.onPhysicalBatchDrained(
                        descriptor,
                        timing.submitThroughDrainMs,
                      );
                    }
                  },
                }),
            onProgress: (progress) => {
              if (
                active === undefined ||
                progress.completedCommandBuffers !== createdIndex + 1
              ) {
                throw new Error(
                  "ACE DiT scheduler progress lost its active command",
                );
              }
              for (const entry of active.entries) {
                if (
                  entry.command.quantum.kind === "sampler" &&
                  entry.subquantumIndex + 1 ===
                    entry.command.sequence.quantumCount
                ) completedEvaluations += 1;
              }
              const last = active.entries.at(-1);
              if (last === undefined) {
                throw new Error("ACE DiT scheduler completed an empty batch");
              }
              const { command, commandId } = last;
              options.onProgress?.({
                completedQuanta: active.batch.lastPhysicalIndexExclusive,
                totalQuanta: graphCommandBufferTarget,
                completedCommandBuffers: progress.completedCommandBuffers,
                totalCommandBuffers: progress.totalCommandBuffers,
                queueDrains: progress.queueDrains,
                cooperativeIdleMs: progress.cooperativeIdleMs,
                completedEvaluations,
                totalEvaluations: this.samplerScheduleProfile.evaluationCount,
                quantum: command.quantum,
                subquantumIndex: last.subquantumIndex,
                subquantumCount: command.sequence.quantumCount,
                commandId,
                batchIndex: active.batch.batchIndex,
                batchFirstPhysicalQuantum: active.batch.firstPhysicalIndex,
                batchPhysicalQuantumCount: active.batch.physicalQuantumCount,
                physicalQuantaPerCommandBuffer,
              });
            },
          });
        }
        if (usesCompletionAttribution && pending.size !== 0) {
          throw new Error(
            `OPT-0080 scheduler left ${pending.size} pending descriptors after success`,
          );
        }
      } finally {
        pending.clear();
      }
      if (
        completedEvaluations !== evaluationTarget ||
        encodedPhysicalQuanta !== graphCommandBufferTarget ||
        subquantumIndex !== 0 ||
        createdIndex + 1 !== batches.length ||
        (evaluationReadbacks === undefined
          ? encodedEvaluationSnapshots !== 0
          : encodedEvaluationSnapshots !==
            this.samplerScheduleProfile.evaluationCount) ||
        (evaluation0VelocityReadback === undefined
          ? encodedEvaluation0VelocityCopies !== 0
          : encodedEvaluation0VelocityCopies !== 1) ||
        (identityReadback === undefined
          ? encodedAttentionIdentityCopies !== 0
          : encodedAttentionIdentityCopies !== 1)
      ) {
        throw new Error("ACE DiT graph drained outside its evaluation boundary");
      }
      if (evaluationTarget === this.samplerScheduleProfile.evaluationCount) {
        if (logicalIndex !== this.compiled.logicalCommands.length) {
          throw new Error("ACE DiT full graph omitted compiled commands");
        }
      } else {
        const next = this.compiled.logicalCommands[logicalIndex];
        if (
          next === undefined || next.quantum.kind !== "timestep" ||
          next.quantum.evaluation !== 1
        ) {
          throw new Error("OPT-0067 did not stop exactly before evaluation 1");
        }
      }
      const finalLatent = this.compiled.evaluationLatents[evaluationTarget - 1];
      if (finalLatent === undefined) {
        throw new Error("ACE DiT evaluation result binding is missing");
      }
      return Object.freeze({
        ...scheduled,
        completedEvaluations: evaluationTarget,
        finalLatent,
        ...(this.compiled.denseRouteProfile === undefined
          ? {}
          : { denseRouteProfile: this.compiled.denseRouteProfile }),
        ...(this.compiled.attentionRouteProfile === undefined
          ? {}
          : { attentionRouteProfile: this.compiled.attentionRouteProfile }),
        ...(opt0080Scheduling === undefined
          ? {}
          : { opt0080Scheduling }),
      });
    } finally {
      this.state = "finished";
      await this.scheduler.dispose();
      this.releaseResources();
    }
  }

  /**
   * Before run, this releases the untouched graph immediately. During run it
   * waits for the active scheduler lease, which is released only after drain.
   */
  async destroy(): Promise<void> {
    if (this.state === "destroyed") return;
    const wasRunning = this.state === "running";
    if (!wasRunning) this.state = "destroyed";
    await this.scheduler.dispose();
    this.releaseResources();
    if (wasRunning) this.state = "destroyed";
  }

  private releaseResources(): void {
    if (this.resourcesReleased) return;
    this.resourcesReleased = true;
    this.runtime.destroy();
    this.model.destroy();
  }
}

/** Derive phase boundaries from the compiled graph, without descriptor capture. */
function planAceDitPhaseCommandBufferCounts(
  logicalCommands: readonly AceCompiledDitCommand[],
  evaluationCount: number,
): readonly number[] {
  if (!Number.isSafeInteger(evaluationCount) || evaluationCount < 1) {
    throw new RangeError("ACE DiT requires at least one evaluation phase");
  }
  const logicalPhaseCounts = Object.freeze([
    STATIC_QUANTA,
    ...Array.from({ length: evaluationCount }, () => EVALUATION_QUANTA),
  ]);
  const phaseCommandBufferCounts: number[] = [];
  let logicalIndex = 0;
  let totalCommandBuffers = 0;
  for (const logicalCount of logicalPhaseCounts) {
    let commandBufferCount = 0;
    const phaseLastLogicalExclusive = logicalIndex + logicalCount;
    for (; logicalIndex < phaseLastLogicalExclusive; logicalIndex += 1) {
      const command = logicalCommands[logicalIndex];
      if (command === undefined) {
        throw new Error("ACE DiT compiled graph ended inside a phase");
      }
      commandBufferCount += command.sequence.quantumCount;
    }
    if (!Number.isSafeInteger(commandBufferCount) || commandBufferCount < 1) {
      throw new Error("ACE DiT compiled an invalid phase command count");
    }
    totalCommandBuffers += commandBufferCount;
    phaseCommandBufferCounts.push(commandBufferCount);
  }
  if (
    logicalIndex > logicalCommands.length ||
    totalCommandBuffers !== logicalCommands.slice(0, logicalIndex).reduce(
      (total, command) => total + command.sequence.quantumCount,
      0,
    )
  ) {
    throw new Error("ACE DiT compiled graph phase inventory changed");
  }
  return Object.freeze(phaseCommandBufferCounts);
}

export function createAceDitGraphControlData(
  shape: AceDitEvaluationShape,
  dcwConfiguration: AceDynamicConditionalWeightingConfiguration,
  samplerScheduleProfile: Readonly<AceDitSamplerScheduleProfile> =
    ACE_TURBO_EIGHT_SAMPLER_SCHEDULE_PROFILE,
): AceDitGraphControlData {
  const plan = requirePinnedGraphShape(shape);
  const profile = requireResolvedAceDitSamplerScheduleProfile(
    samplerScheduleProfile,
  );
  const schedule = createAceDitSamplerSchedule(profile, dcwConfiguration);
  if (schedule.length !== profile.evaluationCount) {
    throw new Error("ACE sampler schedule evaluation inventory changed");
  }
  const selfValidLengths = new Uint32Array(plan.batch * 2);
  const crossValidLengths = new Uint32Array(plan.batch * 2);
  for (let batch = 0; batch < plan.batch; batch += 1) {
    selfValidLengths[batch * 2] = plan.tokens;
    selfValidLengths[batch * 2 + 1] = plan.tokens;
    crossValidLengths[batch * 2] = plan.tokens;
    crossValidLengths[batch * 2 + 1] = plan.conditionTokens;
  }
  const { cosine, sine } = createAceDitRopeTables(plan.tokens);
  return Object.freeze({
    selfValidLengths,
    crossValidLengths,
    cosine,
    sine,
    timesteps: Object.freeze(schedule.map((step) => {
      const values = new Float32Array(plan.batch);
      values.fill(step.timestep);
      return values;
    })),
    relativeTimestepZero: new Float32Array(plan.batch),
  });
}

export function createAceDitGraphQuantumPlan(
  samplerScheduleProfile: Readonly<AceDitSamplerScheduleProfile> =
    ACE_TURBO_EIGHT_SAMPLER_SCHEDULE_PROFILE,
): readonly AceDitGraphQuantum[] {
  const profile = requireResolvedAceDitSamplerScheduleProfile(
    samplerScheduleProfile,
  );
  const quanta: AceDitGraphQuantum[] = [];
  appendQuantum(quanta, { kind: "condition-projection" });
  for (let layer = 0; layer < ACE_DIT_LAYER_COUNT; layer += 1) {
    appendQuantum(quanta, { kind: "cross-cache", layer });
  }
  for (
    let evaluation = 0;
    evaluation < profile.evaluationCount;
    evaluation += 1
  ) {
    appendQuantum(quanta, { kind: "timestep", evaluation });
    appendQuantum(quanta, { kind: "input-projection", evaluation });
    for (let layer = 0; layer < ACE_DIT_LAYER_COUNT; layer += 1) {
      appendQuantum(quanta, { kind: "layer", evaluation, layer });
    }
    appendQuantum(quanta, { kind: "output-projection", evaluation });
    appendQuantum(quanta, { kind: "sampler", evaluation });
  }
  if (quanta.length !== aceDitGraphQuantumCount(profile)) {
    throw new Error(`ACE DiT graph planned ${quanta.length} quanta`);
  }
  return Object.freeze(quanta);
}

export function aceDitGraphQuantumCount(
  samplerScheduleProfile: Readonly<AceDitSamplerScheduleProfile> =
    ACE_TURBO_EIGHT_SAMPLER_SCHEDULE_PROFILE,
): number {
  const profile = requireResolvedAceDitSamplerScheduleProfile(
    samplerScheduleProfile,
  );
  return STATIC_QUANTA + profile.evaluationCount * EVALUATION_QUANTA;
}

export function planAceDitGraphMemory(
  modelProfile: AceModelProfileId,
  shape: AceDitEvaluationShape,
  samplerScheduleProfile: Readonly<AceDitSamplerScheduleProfile> =
    ACE_TURBO_EIGHT_SAMPLER_SCHEDULE_PROFILE,
  ditDenseInputStorageProfile?: AceDitDenseInputStorageProfile,
): AceDitGraphMemoryPlan {
  const plan = requirePinnedGraphShape(shape);
  const denseInputStorageProfile = requireAceDitDenseInputStorageProfile(
    ditDenseInputStorageProfile,
  );
  if (
    denseInputStorageProfile !== undefined &&
    (modelProfile !== "reference-bf16" || plan.batch !== 1 ||
      plan.latentFrames !== 4_500 || plan.tokens !== 2_250 ||
      plan.conditionTokens !== 98)
  ) {
    throw new Error(
      "OPT-0081 dense-input graph storage requires exact reference-bf16 M2250/C98",
    );
  }
  const profile = requireResolvedAceDitSamplerScheduleProfile(
    samplerScheduleProfile,
  );
  const activationElementBytes = modelProfile === "reference-bf16"
    ? 4
    : modelProfile === "raw-fp16"
      ? 2
      : null;
  if (activationElementBytes === null) {
    throw new TypeError(`Unknown ACE DiT model profile ${String(modelProfile)}`);
  }
  const activation = (elements: number): number =>
    aceActivationBytes(modelProfile, elements);
  const conditionInputBytes = activation(
    product(plan.batch, plan.conditionTokens, ACE_TURBO_DIT_CONFIG.conditionInputSize),
  );
  const projectedConditionBytes = activation(
    product(plan.batch, plan.conditionTokens, ACE_TURBO_DIT_CONFIG.hiddenSize),
  );
  const crossKeyValueBytes = activation(plan.crossKeyValueElementsAllLayers);
  const crossCacheScratchBytes = activation(
    3 * plan.crossKeyValueElementsPerLayer,
  );
  const contextBytes = activation(plan.contextElements);
  const latentBytes = activation(plan.latentElements);
  const ropeBytes = product(
    2,
    plan.tokens,
    ACE_TURBO_DIT_CONFIG.headDimension,
    FP32_BYTES,
  );
  const attentionControlBytes = product(2, plan.batch, 2, CONTROL_ELEMENT_BYTES);
  const timestepInputBytes = product(
    profile.evaluationCount + 1,
    plan.batch,
    FP32_BYTES,
  );
  const timestepScratchElements = product(
    2,
    plan.batch,
    ACE_TURBO_DIT_CONFIG.timestepInputSize +
      4 * ACE_TURBO_DIT_CONFIG.hiddenSize +
      LAYER_MODULATION_GROUPS * ACE_TURBO_DIT_CONFIG.hiddenSize,
  );
  const timestepScratchBytes = activation(timestepScratchElements);
  const timestepOutputBytes = activation(
    product(
      plan.batch,
      (1 + LAYER_MODULATION_GROUPS) * ACE_TURBO_DIT_CONFIG.hiddenSize,
    ),
  );
  const concatenatedInputBytes = activation(
    product(plan.batch, plan.latentFrames, ACE_TURBO_DIT_CONFIG.inputChannels),
  );
  const hiddenStateBytes = activation(plan.hiddenElements);
  const layerScratchRequirements = layerScratchElementRequirements(plan);
  const layerScratchBytes = sum(Object.entries(
    layerScratchRequirements,
  ).map(([role, elements]) => aceDitLayerScratchBytes(
    modelProfile,
    role as keyof AceDitLayerScratch,
    elements,
    denseInputStorageProfile,
  )));
  const outputScratchBytes = activation(
    2 * plan.hiddenElements +
      product(OUTPUT_MODULATION_GROUPS, plan.batch, ACE_TURBO_DIT_CONFIG.hiddenSize),
  );
  // Stepped update, predicted-clean estimate, and velocity are simultaneously
  // live beside the two ping/pong state buffers during DCW.
  const samplerScratchBytes = product(3, latentBytes);

  const persistentDenoise = sum([
    crossKeyValueBytes,
    contextBytes,
    2 * latentBytes,
    ropeBytes,
    attentionControlBytes,
    timestepInputBytes,
  ]);
  const preloadedInputs = sum([
    contextBytes,
    2 * latentBytes,
    ropeBytes,
    attentionControlBytes,
    timestepInputBytes,
  ]);
  const conditionPeak = sum([
    preloadedInputs,
    conditionInputBytes,
    projectedConditionBytes,
  ]);
  const crossCachePeak = sum([
    preloadedInputs,
    projectedConditionBytes,
    crossKeyValueBytes,
    crossCacheScratchBytes,
  ]);
  const timestepPeak = sum([
    persistentDenoise,
    timestepScratchBytes,
    timestepOutputBytes,
  ]);
  const inputPeak = sum([
    persistentDenoise,
    timestepOutputBytes,
    concatenatedInputBytes,
    hiddenStateBytes,
  ]);
  const layerPeak = sum([
    persistentDenoise,
    timestepOutputBytes,
    2 * hiddenStateBytes,
    layerScratchBytes,
  ]);
  const outputPeak = sum([
    persistentDenoise,
    timestepOutputBytes,
    hiddenStateBytes,
    outputScratchBytes,
    latentBytes,
  ]);
  const samplerPeak = sum([
    persistentDenoise,
    samplerScratchBytes,
  ]);
  const unaliasedGraphBytesExcludingWeights = sum([
    conditionInputBytes,
    projectedConditionBytes,
    crossKeyValueBytes,
    crossCacheScratchBytes,
    contextBytes,
    2 * latentBytes,
    ropeBytes,
    attentionControlBytes,
    timestepInputBytes,
    timestepScratchBytes,
    timestepOutputBytes,
    concatenatedInputBytes,
    2 * hiddenStateBytes,
    layerScratchBytes,
    outputScratchBytes,
    samplerScratchBytes,
  ]);
  const individualRequirements = [
    conditionInputBytes,
    projectedConditionBytes,
    activation(plan.crossKeyValueElementsPerLayer),
    activation(plan.contextElements),
    latentBytes,
    ropeBytes / 2,
    product(plan.batch, 2, CONTROL_ELEMENT_BYTES),
    product(plan.batch, FP32_BYTES),
    concatenatedInputBytes,
    hiddenStateBytes,
    ...Object.entries(layerScratchRequirements).map(([role, elements]) =>
      aceDitLayerScratchBytes(
        modelProfile,
        role as keyof AceDitLayerScratch,
        elements,
        denseInputStorageProfile,
      )
    ),
  ];
  return Object.freeze({
    modelProfile,
    ...(denseInputStorageProfile === undefined
      ? {}
      : { denseInputStorageProfile }),
    activationElementBytes,
    conditionInputBytes,
    projectedConditionBytes,
    crossKeyValueBytes,
    crossCacheScratchBytes,
    contextBytes,
    latentBytes,
    ropeBytes,
    attentionControlBytes,
    timestepInputBytes,
    timestepScratchBytes,
    timestepOutputBytes,
    concatenatedInputBytes,
    hiddenStateBytes,
    layerScratchBytes,
    outputScratchBytes,
    samplerScratchBytes,
    minimumGraphBytesExcludingWeights: Math.max(
      conditionPeak,
      crossCachePeak,
      timestepPeak,
      inputPeak,
      layerPeak,
      outputPeak,
      samplerPeak,
    ),
    unaliasedGraphBytesExcludingWeights,
    largestRequiredBindingBytes: Math.max(...individualRequirements),
    relativeZeroPolicy: "immutable-shared-input-recomputed-each-evaluation",
  });
}

async function compileAceDitGraph(
  runtime: AceDitGraphRuntime,
  weights: AceDitResolvedPackageWeights,
  plan: AceDitEvaluationPlan,
  dcwConfiguration: AceDynamicConditionalWeightingConfiguration,
  bindings: AceDitGraphBindings,
  compilation: AceDitGraphCompilationOptions,
): Promise<AceCompiledDitGraph> {
  const samplerScheduleProfile = requireResolvedAceDitSamplerScheduleProfile(
    compilation.samplerScheduleProfile ??
      ACE_TURBO_EIGHT_SAMPLER_SCHEDULE_PROFILE,
  );
  const captureDescriptors =
    compilation.capturePhysicalCommandDescriptors === true;
  const captureProfiles = captureDescriptors
    ? requireOpt0018CaptureConfiguration(plan, compilation)
    : Object.freeze({
        denseRuntimeProfile: ACE_OPT_0009_DIT_DENSE_RUNTIME_PROFILE,
        attentionRuntimeProfile: ACE_DIT_QUERY8_ATTENTION_RUNTIME_PROFILE,
      });
  const quanta = createAceDitGraphQuantumPlan(samplerScheduleProfile);
  const logicalCommands: AceCompiledDitCommand[] = [];
  const physicalDescriptors: AceDitPhysicalCommandDescriptor[] | undefined =
    captureDescriptors ? [] : undefined;
  let descriptorPreparationMs = 0;
  let commandBufferCount = 0;
  let quantumIndex = 0;
  const append = (
    dispatch: Readonly<{
      readonly label?: string;
      readonly cooperativeQuanta?: readonly AceGpuEncodeQuantum[];
      readonly cooperativeSequence?: AceGpuEncodeSequence;
      readonly plan?: unknown;
      readonly rangeCount?: number;
      readonly encodeRange?: AceGemmDispatch["encodeRange"];
      readonly encode: (pass: GPUComputePassEncoder) => void;
    }>,
  ): void => {
    const quantum = quanta[quantumIndex];
    if (quantum === undefined) throw new Error("ACE DiT graph emitted excess work");
    const sequence = dispatch.cooperativeSequence ??
      (dispatch.cooperativeQuanta !== undefined
        ? sequenceFromQuanta(dispatch.cooperativeQuanta)
        : dispatch.label !== undefined && dispatch.rangeCount !== undefined &&
          dispatch.encodeRange !== undefined
        ? aceCompositeCooperativeSequence([dispatch as AceGemmDispatch])
        : sequenceFromQuanta(Object.freeze([Object.freeze({
            id: dispatch.label ?? quantum.label,
            primitiveCount: 1,
            encode: dispatch.encode,
          })])));
    if (sequence.quantumCount === 0) {
      throw new Error(`${quantum.label} emitted no cooperative work`);
    }
    if (captureDescriptors) {
      const startedAt = profileNow();
      for (
        let subquantumIndex = 0;
        subquantumIndex < sequence.quantumCount;
        subquantumIndex += 1
      ) {
        physicalDescriptors!.push(createOpt0018PhysicalDescriptor(
          commandBufferCount + subquantumIndex,
          quantum,
          subquantumIndex,
          sequence.quantumCount,
          sequence.describeQuantum(subquantumIndex),
          captureProfiles.denseRuntimeProfile,
          captureProfiles.attentionRuntimeProfile,
        ));
      }
      descriptorPreparationMs += nonnegativeProfileElapsed(
        profileNow(),
        startedAt,
      );
    }
    logicalCommands.push(Object.freeze({ quantum, sequence }));
    commandBufferCount += sequence.quantumCount;
    quantumIndex += 1;
    compilation.onProgress?.({
      compiledQuanta: quantumIndex,
      totalQuanta: quanta.length,
      quantum,
    });
  };
  const compile = async (
    create: (quantum: AceDitGraphQuantum) => Promise<{
      readonly label?: string;
      readonly cooperativeQuanta?: readonly AceGpuEncodeQuantum[];
      readonly cooperativeSequence?: AceGpuEncodeSequence;
      readonly plan?: unknown;
      readonly rangeCount?: number;
      readonly encodeRange?: AceGemmDispatch["encodeRange"];
      readonly encode: (pass: GPUComputePassEncoder) => void;
    }>,
  ): Promise<void> => {
    compilation.signal?.throwIfAborted();
    const quantum = quanta[quantumIndex];
    if (quantum === undefined) throw new Error("ACE DiT graph emitted excess work");
    const dispatch = await create(quantum);
    compilation.signal?.throwIfAborted();
    append(dispatch);
  };

  await compile(async (quantum) => await runtime.createConditionProjectionDispatch(
    quantum.label,
    ACE_TURBO_DIT_CONFIG,
    plan.batch,
    plan.conditionTokens,
    {
      input: bindings.conditionInput,
      weight: weights.conditionProjection.weight,
      bias: weights.conditionProjection.bias,
      output: bindings.projectedCondition,
    },
  ));
  for (let layer = 0; layer < ACE_DIT_LAYER_COUNT; layer += 1) {
    const cache = bindings.crossCaches[layer]!;
    await compile(async (quantum) => await runtime.createCrossCacheDispatch(
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
    ));
  }

  const schedule = createAceDitSamplerSchedule(
    samplerScheduleProfile,
    dcwConfiguration,
  );
  const evaluationLatents: GPUBufferBinding[] = [];
  for (let evaluation = 0; evaluation < schedule.length; evaluation += 1) {
    const step = schedule[evaluation]!;
    const currentLatent = bindings.latents[evaluation % 2]!;
    const nextLatent = bindings.latents[(evaluation + 1) % 2]!;
    const steppedLatent = bindings.latents[2];
    await compile(async (quantum) => await runtime.createTimestepDispatch(
      quantum.label,
      ACE_TURBO_DIT_CONFIG,
      plan.batch,
      {
        timestep: bindings.controls.timesteps[evaluation]!,
        relativeTimestep: bindings.controls.relativeTimestepZero,
        weights: weights.timestep,
        scratch: bindings.timestepScratch,
        embedding: bindings.timestepEmbedding,
        projection: bindings.timestepProjection,
      },
    ));
    await compile(async (quantum) => await runtime.createInputProjectionDispatch(
      quantum.label,
      ACE_TURBO_DIT_CONFIG,
      plan,
      {
        context: bindings.context,
        latent: currentLatent,
        concatenated: bindings.concatenatedInput,
        weight: weights.inputProjection.weight,
        bias: weights.inputProjection.bias,
        output: bindings.hidden[0],
      },
    ));
    for (let layer = 0; layer < ACE_DIT_LAYER_COUNT; layer += 1) {
      const input = bindings.hidden[layer % 2]!;
      const output = bindings.hidden[(layer + 1) % 2]!;
      const attentionMode = aceDitLayerAttentionMode(layer);
      const identityCounters = bindings.opt0062AttentionIdentityCounters;
      const identityRouteIndex = attentionMode === "full"
        ? evaluation * 12 + (layer - 1) / 2
        : undefined;
      const layerBindings: AceDitLayerBindings = {
        input,
        output,
        weights: weights.layers[layer]!,
        scratch: bindings.layerScratch,
        timestepProjection: bindings.timestepProjection,
        crossKey: bindings.crossCaches[layer]!.key,
        crossValue: bindings.crossCaches[layer]!.value,
        selfValidLengths: bindings.controls.selfValidLengths,
        crossValidLengths: bindings.controls.crossValidLengths,
        cosine: bindings.controls.cosine,
        sine: bindings.controls.sine,
        ...(identityCounters === undefined || identityRouteIndex === undefined
          ? {}
          : {
              opt0062AttentionIdentity: {
                counters: identityCounters,
                routeIndex: identityRouteIndex,
              },
            }),
      };
      await compile(async (quantum) => await runtime.createLayerDispatch(
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
    await compile(async (quantum) => await runtime.createOutputProjectionDispatch(
      quantum.label,
      ACE_TURBO_DIT_CONFIG,
      plan,
      {
        hidden: bindings.hidden[0],
        timestepEmbedding: bindings.timestepEmbedding,
        weights: weights.output,
        scratch: bindings.outputScratch,
        velocity: bindings.velocity,
      },
    ));
    await compile(async (quantum) => await runtime.createSamplerStepDispatch(
      quantum.label,
      ACE_TURBO_DIT_CONFIG,
      plan,
      step,
      samplerScheduleProfile,
      dcwConfiguration,
      {
        latent: currentLatent,
        velocity: bindings.velocity,
        stepped: steppedLatent,
        predictedClean: bindings.predictedCleanLatent,
        output: nextLatent,
      },
    ));
    evaluationLatents.push(nextLatent);
  }
  const denseRouteProfile = runtime.finalizeDenseRoutes?.();
  const attentionRouteProfile = runtime.finalizeAttentionRoutes?.();
  const selectiveDense = compilation.denseGemmConfiguration?.backend ===
    "opt-0056-selective-k4-exact-down";
  const diagnosticQuadQueryAttention =
    compilation.attentionConfiguration?.backend ===
    "opt-0062-fixed32-quad-query32-full-self";
  const productionShapeSelectedAttention =
    compilation.attentionConfiguration?.backend ===
      "opt-0070-fixed32-quad-query32-full-self-production";
  const routedAttention = diagnosticQuadQueryAttention ||
    productionShapeSelectedAttention;
  if (
    evaluationLatents.length !== samplerScheduleProfile.evaluationCount ||
    (selectiveDense && denseRouteProfile === undefined) ||
    (!selectiveDense && denseRouteProfile !== undefined) ||
    (routedAttention && attentionRouteProfile === undefined) ||
    (!routedAttention && attentionRouteProfile !== undefined) ||
    (bindings.opt0062AttentionIdentityCounters !== undefined &&
      !diagnosticQuadQueryAttention)
  ) {
    throw new Error("ACE DiT selective route/sampler inventory changed");
  }
  if (
    quantumIndex !== quanta.length ||
    logicalCommands.length !== quanta.length ||
    commandBufferCount < quanta.length
  ) {
    throw new Error("ACE DiT graph did not encode the complete pinned schedule");
  }
  let physicalCommandDescriptors:
    | AceDitPhysicalCommandDescriptorTable
    | undefined;
  if (captureDescriptors) {
    if (
      commandBufferCount !== ACE_OPT_0018_M2250_GRAPH_COMMAND_BUFFER_COUNT ||
      physicalDescriptors === undefined ||
      physicalDescriptors.length !== commandBufferCount ||
      physicalDescriptors.some((descriptor, index) =>
        descriptor.physicalIndex !== index
      )
    ) {
      throw new Error(
        "OPT-0018 M2250 physical descriptor topology changed",
      );
    }
    const serializationStartedAt = profileNow();
    const serialized = new TextEncoder().encode(JSON.stringify(
      physicalDescriptors,
    ));
    const sha256 = aceSha256Hex(serialized);
    descriptorPreparationMs += nonnegativeProfileElapsed(
      profileNow(),
      serializationStartedAt,
    );
    physicalCommandDescriptors = Object.freeze({
      descriptors: Object.freeze(physicalDescriptors),
      sha256,
      serializedBytes: serialized.byteLength,
      memberCount: physicalDescriptors.reduce(
        (total, descriptor) => total + descriptor.members.length,
        0,
      ),
      preparationMs: descriptorPreparationMs,
    });
  }
  return Object.freeze({
    logicalCommands: Object.freeze(logicalCommands),
    commandBufferCount,
    finalLatent:
      bindings.latents[samplerScheduleProfile.evaluationCount % 2]!,
    evaluation0Velocity: bindings.velocity,
    evaluationLatents: Object.freeze(evaluationLatents),
    ...(denseRouteProfile === undefined ? {} : { denseRouteProfile }),
    ...(attentionRouteProfile === undefined
      ? {}
      : { attentionRouteProfile }),
    ...(bindings.opt0062AttentionIdentityCounters === undefined
      ? {}
      : {
          opt0062AttentionIdentityCounters:
            bindings.opt0062AttentionIdentityCounters,
        }),
    ...(physicalCommandDescriptors === undefined
      ? {}
      : { physicalCommandDescriptors }),
  });
}

function requireOpt0018CaptureConfiguration(
  plan: AceDitEvaluationPlan,
  compilation: AceDitGraphCompilationOptions,
): Readonly<{
  denseRuntimeProfile: AceDitDenseRuntimeProfile;
  attentionRuntimeProfile: AceDitAttentionRuntimeProfile;
}> {
  const denseBackend = compilation.denseGemmConfiguration?.backend;
  const attentionConfiguration = compilation.attentionConfiguration;
  const attentionBackend = attentionConfiguration?.backend;
  if (
    plan.batch !== 1 ||
    plan.latentFrames !== ACE_OPT_0018_M2250_LATENT_FRAMES ||
    plan.tokens !== ACE_OPT_0018_M2250_TOKENS ||
    plan.conditionTokens !== ACE_OPT_0018_CONDITION_TOKENS ||
    compilation.gemmConfiguration?.backend !== "subgroups" ||
    compilation.gemmConfiguration.capability.subgroupMinSize !== 32 ||
    compilation.gemmConfiguration.capability.subgroupMaxSize !== 32 ||
    (denseBackend !== "opt-0009-fp16-fp32" &&
      denseBackend !== "opt-0037-k4-fp16-partials" &&
      denseBackend !== "opt-0056-selective-k4-exact-down") ||
    compilation.denseGemmConfiguration?.capability.subgroupMinSize !== 32 ||
    compilation.denseGemmConfiguration?.capability.subgroupMaxSize !== 32 ||
    (attentionBackend !== "fixed32-subgroup-query8" &&
      attentionBackend !== "opt-0062-fixed32-quad-query32-full-self" &&
      attentionBackend !==
        "opt-0070-fixed32-quad-query32-full-self-production") ||
    attentionConfiguration === undefined ||
    attentionConfiguration.capability.subgroupMinSize !== 32 ||
    attentionConfiguration.capability.subgroupMaxSize !== 32 ||
    (attentionConfiguration.backend ===
        "opt-0070-fixed32-quad-query32-full-self-production" &&
      (attentionConfiguration.expectedQueryTokens !== plan.tokens ||
        attentionConfiguration.expectedConditionTokens !==
          plan.conditionTokens))
  ) {
    throw new Error(
      "OPT-0018 capture requires exact M2250/C98 mixed fixed32 production selection",
    );
  }
  const denseRuntimeProfile = denseBackend ===
      "opt-0056-selective-k4-exact-down"
    ? ACE_OPT_0056_DIT_SELECTIVE_K4_RUNTIME_PROFILE
    : denseBackend === "opt-0037-k4-fp16-partials"
      ? ACE_OPT_0037_DIT_K4_RUNTIME_PROFILE
      : ACE_OPT_0009_DIT_DENSE_RUNTIME_PROFILE;
  if (
    (attentionBackend === "opt-0062-fixed32-quad-query32-full-self" ||
      attentionBackend ===
        "opt-0070-fixed32-quad-query32-full-self-production") &&
    denseRuntimeProfile !== ACE_OPT_0009_DIT_DENSE_RUNTIME_PROFILE
  ) {
    throw new Error("OPT-0062 cannot combine with a revision-8 dense profile");
  }
  return Object.freeze({
    denseRuntimeProfile,
    attentionRuntimeProfile: attentionBackend ===
        "opt-0062-fixed32-quad-query32-full-self"
      ? ACE_OPT_0062_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE
      : attentionBackend ===
          "opt-0070-fixed32-quad-query32-full-self-production"
        ? ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE
        : ACE_DIT_QUERY8_ATTENTION_RUNTIME_PROFILE,
  });
}

function createOpt0018PhysicalDescriptor(
  physicalIndex: number,
  quantum: AceDitGraphQuantum,
  subquantumIndex: number,
  subquantumCount: number,
  sequence: AceGpuEncodeSequenceQuantumDescriptor,
  denseRuntimeProfile: AceDitDenseRuntimeProfile,
  attentionRuntimeProfile: AceDitAttentionRuntimeProfile,
): AceDitPhysicalCommandDescriptor {
  const members = sequence.members.map((member) => {
    const { family, backend, kernel } =
      classifyAceOpt0018DitCommandMember(
        quantum,
        member.label,
        denseRuntimeProfile,
        attentionRuntimeProfile,
      );
    return Object.freeze({
      id: member.id,
      family,
      backend,
      kernel,
      scheduledMultiplyAdds: member.scheduledMultiplyAdds,
    });
  });
  if (
    members.length === 0 ||
    sequence.primitiveCount !== sequence.members.reduce(
      (total, member) => total + member.primitiveCount,
      0,
    ) ||
    sequence.scheduledMultiplyAdds !== sequence.members.reduce(
      (total, member) => total + member.scheduledMultiplyAdds,
      0,
    )
  ) {
    throw new Error(
      `OPT-0018 command ${physicalIndex} has an invalid sequence descriptor`,
    );
  }
  const memberFamilies = new Set(members.map((member) => member.family));
  const family: AceDitProfileFamily = memberFamilies.size === 1
    ? members[0]!.family
    : "mixed";
  const commandId = subquantumCount === 1
    ? quantum.label
    : `${quantum.label}-part-${subquantumIndex}`;
  return Object.freeze({
    physicalIndex,
    logicalIndex: quantum.index,
    logicalKind: quantum.kind,
    commandId,
    subquantumIndex,
    subquantumCount,
    evaluation: "evaluation" in quantum ? quantum.evaluation : null,
    layer: "layer" in quantum ? quantum.layer : null,
    family,
    primitiveCount: sequence.primitiveCount,
    scheduledMultiplyAdds: sequence.scheduledMultiplyAdds,
    members: Object.freeze(members),
  });
}

/** @internal Frozen OPT-0018 primitive taxonomy used by focused contracts. */
export function classifyAceOpt0018DitCommandMember(
  quantum: AceDitGraphQuantum,
  label: string,
  denseRuntimeProfile: AceDitDenseRuntimeProfile =
    ACE_OPT_0009_DIT_DENSE_RUNTIME_PROFILE,
  attentionRuntimeProfile: AceDitAttentionRuntimeProfile =
    ACE_DIT_QUERY8_ATTENTION_RUNTIME_PROFILE,
): Readonly<{
  family: Exclude<AceDitProfileFamily, "mixed">;
  backend: string;
  kernel: string;
}> {
  const family = classifyOpt0018MemberFamily(quantum, label);
  return Object.freeze({
    family,
    ...classifyOpt0018MemberKernel(
      quantum,
      label,
      family,
      denseRuntimeProfile,
      attentionRuntimeProfile,
    ),
  });
}

function classifyOpt0018MemberFamily(
  quantum: AceDitGraphQuantum,
  label: string,
): Exclude<AceDitProfileFamily, "mixed"> {
  const suffix = opt0018MemberSuffix(quantum, label);
  switch (quantum.kind) {
    case "condition-projection":
      if (suffix !== "") break;
      return "precompute";
    case "cross-cache":
      if (
        /^(key-projection|value-projection|split-key-heads|split-value-heads|key-norm)$/u
          .test(suffix)
      ) return "cross-cache";
      break;
    case "timestep":
      if (
        /^(timestep-frequency|relative-frequency|timestep-linear-1|relative-linear-1|timestep-silu-1|relative-silu-1|timestep-linear-2|relative-linear-2|timestep-silu-2|relative-silu-2|timestep-projection|relative-projection|embedding-add|projection-add)$/u
          .test(suffix)
      ) return "timestep";
      break;
    case "input-projection":
      if (/^(concatenate|patch)$/u.test(suffix)) return "input";
      break;
    case "layer":
      if (
        /^(self-(query|key|value|output)-projection|cross-(query|output)-projection)$/u
          .test(suffix)
      ) return "attention-projections";
      if (suffix === "self-full-attention") return "self-full";
      if (suffix === "self-sliding-attention") return "self-sliding";
      if (suffix === "cross-attention") return "cross-attention";
      if (
        /^(mlp-norm|mlp-adaln|mlp-gate-projection|mlp-up-projection|mlp-swiglu|mlp-down-projection|mlp-gated-residual)$/u
          .test(suffix)
      ) return "feed-forward";
      if (
        /^(modulation|self-norm|self-adaln|self-split-query-heads|self-split-key-heads|self-split-value-heads|self-query-norm|self-key-norm|self-query-rope|self-key-rope|self-merge-heads|self-gated-residual|cross-norm|cross-split-query-heads|cross-query-norm|cross-merge-heads|cross-residual)$/u
          .test(suffix)
      ) return "plumbing";
      break;
    case "output-projection":
      if (/^(norm|modulation|adaln|unpatch)$/u.test(suffix)) return "output";
      break;
    case "sampler":
      if (/^(sampler-update|predicted-clean|dcw)$/u.test(suffix)) {
        return "sampler-dcw";
      }
      break;
  }
  throw new Error(
    `OPT-0018 cannot classify ${quantum.kind} primitive ${label}`,
  );
}

function classifyOpt0018MemberKernel(
  quantum: AceDitGraphQuantum,
  label: string,
  family: Exclude<AceDitProfileFamily, "mixed">,
  denseRuntimeProfile: AceDitDenseRuntimeProfile,
  attentionRuntimeProfile: AceDitAttentionRuntimeProfile,
): Readonly<{ backend: string; kernel: string }> {
  const suffix = opt0018MemberSuffix(quantum, label);
  if (
    family === "attention-projections" ||
    (family === "feed-forward" && /projection$/u.test(suffix))
  ) {
    if (denseRuntimeProfile === ACE_OPT_0056_DIT_SELECTIVE_K4_RUNTIME_PROFILE) {
      const exactDown = suffix === "mlp-down-projection";
      return Object.freeze({
        backend: exactDown
          ? "opt-0056-k4-exact-fp32"
          : "opt-0032-k4-fp16-partials",
        kernel: exactDown
          ? ACE_OPT_0056_DENSE_K4_EXACT_KERNEL_ID
          : ACE_OPT_0037_DIT_K4_KERNEL_SET_ID,
      });
    }
    if (denseRuntimeProfile === ACE_OPT_0037_DIT_K4_RUNTIME_PROFILE) {
      return Object.freeze({
        backend: "opt-0037-k4-fp16-partials",
        kernel: ACE_OPT_0037_DIT_K4_KERNEL_SET_ID,
      });
    }
    if (denseRuntimeProfile === ACE_OPT_0009_DIT_DENSE_RUNTIME_PROFILE) {
      return Object.freeze({
        backend: "opt-0009-fp16-fp32",
        kernel: ACE_OPT_0009_DIT_DENSE_KERNEL_SET_ID,
      });
    }
    throw new Error("OPT-0018 dense runtime profile is not authenticated");
  }
  if (
    family === "self-full" ||
    family === "self-sliding" ||
    family === "cross-attention"
  ) {
    if (
      family === "self-full" &&
      (attentionRuntimeProfile ===
          ACE_OPT_0062_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE ||
        attentionRuntimeProfile ===
          ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE)
    ) {
      return Object.freeze({
        backend: "opt-0062-fixed32-quad-query32-full-self",
        kernel: ACE_OPT_0062_QUAD_QUERY_ATTENTION_KERNEL_ID,
      });
    }
    if (attentionRuntimeProfile !== ACE_DIT_QUERY8_ATTENTION_RUNTIME_PROFILE &&
      attentionRuntimeProfile !==
        ACE_OPT_0062_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE &&
      attentionRuntimeProfile !==
        ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE) {
      throw new Error("OPT-0018 attention runtime profile is not authenticated");
    }
    return Object.freeze({
      backend: "fixed32-subgroup-query8",
      kernel: PROFILE_ATTENTION_KERNEL,
    });
  }
  if (
    quantum.kind === "condition-projection" ||
    (quantum.kind === "cross-cache" && /projection$/u.test(suffix)) ||
    (quantum.kind === "timestep" &&
      /^(timestep|relative)-(linear-[12]|projection)$/u.test(suffix))
  ) {
    return Object.freeze({
      backend: "fixed32-subgroups",
      kernel: PROFILE_BASE_GEMM_KERNEL,
    });
  }
  if (/norm$/u.test(suffix)) {
    return Object.freeze({
      backend: "reference-bf16",
      kernel: PROFILE_RMSNORM_KERNEL,
    });
  }
  if (/rope$/u.test(suffix)) {
    return Object.freeze({
      backend: "reference-bf16",
      kernel: PROFILE_ROPE_KERNEL,
    });
  }
  if (suffix === "dcw") {
    return Object.freeze({
      backend: "reference-bf16",
      kernel: PROFILE_DCW_KERNEL,
    });
  }
  if (
    /(^|-)silu(-|$)|swiglu$|adaln$|split-(query|key|value)-heads$|merge-heads$|gated-residual$|cross-residual$|embedding-add$|projection-add$/u
      .test(suffix)
  ) {
    return Object.freeze({
      backend: "reference-bf16",
      kernel: PROFILE_TRANSFORMER_KERNEL,
    });
  }
  if (
    /frequency$|modulation$|concatenate$|patch$|unpatch$|sampler-update$|predicted-clean$/u
      .test(suffix)
  ) {
    return Object.freeze({
      backend: "reference-bf16",
      kernel: PROFILE_DIT_KERNEL,
    });
  }
  throw new Error(`OPT-0018 cannot identify kernel for primitive ${label}`);
}

function opt0018MemberSuffix(
  quantum: AceDitGraphQuantum,
  label: string,
): string {
  if (label === quantum.label) return "";
  const prefix = `${quantum.label}-`;
  if (!label.startsWith(prefix)) {
    throw new Error(
      `OPT-0018 primitive ${label} escaped logical command ${quantum.label}`,
    );
  }
  return label.slice(prefix.length);
}

function profileNow(): number {
  return globalThis.performance?.now() ?? Date.now();
}

function nonnegativeProfileElapsed(finishedAt: number, startedAt: number): number {
  const elapsed = finishedAt - startedAt;
  return Number.isFinite(elapsed) && elapsed > 0 ? elapsed : 0;
}

function sequenceFromQuanta(
  quanta: readonly AceGpuEncodeQuantum[],
): AceGpuEncodeSequence {
  if (quanta.length === 0) {
    throw new RangeError("ACE DiT cooperative sequence cannot be empty");
  }
  return Object.freeze({
    quantumCount: quanta.length,
    describeQuantum(index: number): AceGpuEncodeSequenceQuantumDescriptor {
      const quantum = quanta[index];
      if (quantum === undefined) {
        throw new RangeError(`ACE DiT cooperative quantum ${index} is absent`);
      }
      return Object.freeze({
        id: quantum.id,
        primitiveCount: quantum.primitiveCount,
        scheduledMultiplyAdds: quantum.scheduledMultiplyAdds ?? 0,
        members: Object.freeze([Object.freeze({
          id: quantum.id,
          label: quantum.id,
          rangeIndex: null,
          primitiveCount: quantum.primitiveCount,
          scheduledMultiplyAdds: quantum.scheduledMultiplyAdds ?? 0,
        })]),
      });
    },
    encodeQuantum(pass: GPUComputePassEncoder, index: number): void {
      const quantum = quanta[index];
      if (quantum === undefined) {
        throw new RangeError(`ACE DiT cooperative quantum ${index} is absent`);
      }
      quantum.encode(pass);
    },
  });
}

function validateAceDitGraphBindings(
  modelProfile: AceModelProfileId,
  plan: AceDitEvaluationPlan,
  weights: AceDitResolvedPackageWeights,
  bindings: AceDitGraphBindings,
  samplerScheduleProfile: Readonly<AceDitSamplerScheduleProfile>,
  ditDenseInputStorageProfile?: AceDitDenseInputStorageProfile,
): void {
  const denseInputStorageProfile = requireAceDitDenseInputStorageProfile(
    ditDenseInputStorageProfile,
  );
  if (
    denseInputStorageProfile !== undefined &&
    bindings.opt0062AttentionIdentityCounters !== undefined
  ) {
    throw new Error(
      "OPT-0081 dense-input graph storage cannot alias the OPT-0062 identity oracle",
    );
  }
  const evaluationCount = requireResolvedAceDitSamplerScheduleProfile(
    samplerScheduleProfile,
  ).evaluationCount;
  const graphQuantumCount = aceDitGraphQuantumCount(samplerScheduleProfile);
  if (bindings.crossCaches.length !== ACE_DIT_LAYER_COUNT) {
    throw new RangeError("ACE DiT graph requires exactly 24 cross K/V caches");
  }
  if (bindings.controls.timesteps.length !== evaluationCount) {
    throw new RangeError(
      "ACE DiT graph timestep bindings do not match its sampler schedule",
    );
  }
  if (
    weights.layers.length !== ACE_DIT_LAYER_COUNT ||
    weights.crossCaches.length !== ACE_DIT_LAYER_COUNT
  ) {
    throw new RangeError("ACE DiT graph requires exactly 24 resolved weight layers");
  }
  const activation = (elements: number): number =>
    aceActivationBytes(modelProfile, elements);
  const hiddenBytes = activation(plan.hiddenElements);
  const crossBytes = activation(plan.crossKeyValueElementsPerLayer);
  const interval = (first: number, last = first): readonly QuantumInterval[] =>
    Object.freeze([{ first, last }]);
  const repeated = (offset: number): readonly QuantumInterval[] =>
    Object.freeze(Array.from(
      { length: evaluationCount },
      (_, evaluation) => ({
        first: STATIC_QUANTA + evaluation * EVALUATION_QUANTA + offset,
        last: STATIC_QUANTA + evaluation * EVALUATION_QUANTA + offset,
      }),
    ));
  const allGraph = interval(0, graphQuantumCount - 1);
  const denoise = interval(0, graphQuantumCount - 1);
  const controlsLifetime = interval(
    0,
    STATIC_QUANTA +
      (evaluationCount - 1) * EVALUATION_QUANTA +
      25,
  );
  const resources: BindingLifetime[] = [];
  const add = (
    label: string,
    binding: GPUBufferBinding,
    bytes: number,
    intervals: readonly QuantumInterval[],
    writable: boolean,
  ): void => {
    requireBindingBytes(binding, bytes, label);
    resources.push({
      label,
      range: usedBindingRange(binding, bytes, label),
      intervals,
      writable,
    });
  };

  if (bindings.opt0062AttentionIdentityCounters !== undefined) {
    add(
      "OPT-0062 attention identity counters",
      bindings.opt0062AttentionIdentityCounters,
      ACE_OPT_0062_IDENTITY_COUNTER_BYTES,
      allGraph,
      true,
    );
  }

  add(
    "condition input",
    bindings.conditionInput,
    activation(product(plan.batch, plan.conditionTokens, ACE_TURBO_DIT_CONFIG.conditionInputSize)),
    interval(0),
    false,
  );
  add(
    "projected condition",
    bindings.projectedCondition,
    activation(product(plan.batch, plan.conditionTokens, ACE_TURBO_DIT_CONFIG.hiddenSize)),
    interval(0, STATIC_QUANTA - 1),
    true,
  );
  for (const [name, elements] of Object.entries({
    keyFlat: plan.crossKeyValueElementsPerLayer,
    valueFlat: plan.crossKeyValueElementsPerLayer,
    keyHeads: plan.crossKeyValueElementsPerLayer,
  })) {
    add(
      `cross cache scratch ${name}`,
      bindings.crossCacheScratch[name as keyof AceDitCrossCacheScratch],
      activation(elements),
      interval(1, STATIC_QUANTA - 1),
      true,
    );
  }
  for (let layer = 0; layer < ACE_DIT_LAYER_COUNT; layer += 1) {
    const lastRead =
      STATIC_QUANTA +
      (evaluationCount - 1) * EVALUATION_QUANTA +
      2 + layer;
    add(
      `cross cache ${layer} key`,
      bindings.crossCaches[layer]!.key,
      crossBytes,
      interval(1 + layer, lastRead),
      true,
    );
    add(
      `cross cache ${layer} value`,
      bindings.crossCaches[layer]!.value,
      crossBytes,
      interval(1 + layer, lastRead),
      true,
    );
  }
  add("context", bindings.context, activation(plan.contextElements), denoise, false);
  add("latent ping", bindings.latents[0], activation(plan.latentElements), denoise, true);
  add("latent pong", bindings.latents[1], activation(plan.latentElements), denoise, true);
  add("stepped latent", bindings.latents[2], activation(plan.latentElements), repeated(27), true);
  add(
    "concatenated input",
    bindings.concatenatedInput,
    activation(product(plan.batch, plan.latentFrames, ACE_TURBO_DIT_CONFIG.inputChannels)),
    repeated(1),
    true,
  );
  for (const [index, binding] of bindings.hidden.entries()) {
    add(
      `hidden ${index}`,
      binding,
      hiddenBytes,
      repeatedRange(evaluationCount, 1, 26),
      true,
    );
  }
  add(
    "self valid lengths",
    bindings.controls.selfValidLengths,
    product(plan.batch, 2, CONTROL_ELEMENT_BYTES),
    controlsLifetime,
    false,
  );
  add(
    "cross valid lengths",
    bindings.controls.crossValidLengths,
    product(plan.batch, 2, CONTROL_ELEMENT_BYTES),
    controlsLifetime,
    false,
  );
  add(
    "RoPE cosine",
    bindings.controls.cosine,
    product(plan.tokens, ACE_TURBO_DIT_CONFIG.headDimension, FP32_BYTES),
    controlsLifetime,
    false,
  );
  add(
    "RoPE sine",
    bindings.controls.sine,
    product(plan.tokens, ACE_TURBO_DIT_CONFIG.headDimension, FP32_BYTES),
    controlsLifetime,
    false,
  );
  for (let evaluation = 0; evaluation < evaluationCount; evaluation += 1) {
    add(
      `timestep ${evaluation}`,
      bindings.controls.timesteps[evaluation]!,
      product(plan.batch, FP32_BYTES),
      interval(0, STATIC_QUANTA + evaluation * EVALUATION_QUANTA),
      false,
    );
  }
  add(
    "relative timestep zero",
    bindings.controls.relativeTimestepZero,
    product(plan.batch, FP32_BYTES),
    allGraph,
    false,
  );
  const timestepElements: Readonly<Record<keyof AceDitTimestepScratch, number>> = {
    timestepFrequency: product(plan.batch, ACE_TURBO_DIT_CONFIG.timestepInputSize),
    relativeFrequency: product(plan.batch, ACE_TURBO_DIT_CONFIG.timestepInputSize),
    timestepLinear1: product(plan.batch, ACE_TURBO_DIT_CONFIG.hiddenSize),
    relativeLinear1: product(plan.batch, ACE_TURBO_DIT_CONFIG.hiddenSize),
    timestepActivation1: product(plan.batch, ACE_TURBO_DIT_CONFIG.hiddenSize),
    relativeActivation1: product(plan.batch, ACE_TURBO_DIT_CONFIG.hiddenSize),
    timestepEmbedding: product(plan.batch, ACE_TURBO_DIT_CONFIG.hiddenSize),
    relativeEmbedding: product(plan.batch, ACE_TURBO_DIT_CONFIG.hiddenSize),
    timestepActivation2: product(plan.batch, ACE_TURBO_DIT_CONFIG.hiddenSize),
    relativeActivation2: product(plan.batch, ACE_TURBO_DIT_CONFIG.hiddenSize),
    timestepProjection: product(plan.batch, LAYER_MODULATION_GROUPS, ACE_TURBO_DIT_CONFIG.hiddenSize),
    relativeProjection: product(plan.batch, LAYER_MODULATION_GROUPS, ACE_TURBO_DIT_CONFIG.hiddenSize),
  };
  for (const [name, elements] of Object.entries(timestepElements)) {
    add(
      `timestep scratch ${name}`,
      bindings.timestepScratch[name as keyof AceDitTimestepScratch],
      activation(elements),
      repeated(0),
      true,
    );
  }
  add(
    "timestep embedding",
    bindings.timestepEmbedding,
    activation(product(plan.batch, ACE_TURBO_DIT_CONFIG.hiddenSize)),
    repeatedRange(evaluationCount, 0, 26),
    true,
  );
  add(
    "timestep projection",
    bindings.timestepProjection,
    activation(product(plan.batch, LAYER_MODULATION_GROUPS, ACE_TURBO_DIT_CONFIG.hiddenSize)),
    repeatedRange(evaluationCount, 0, 25),
    true,
  );
  const layerRequirements = layerScratchElementRequirements(plan);
  for (const [name, elements] of Object.entries(layerRequirements)) {
    add(
      `layer scratch ${name}`,
      bindings.layerScratch[name as keyof AceDitLayerScratch],
      aceDitLayerScratchBytes(
        modelProfile,
        name as keyof AceDitLayerScratch,
        elements,
        denseInputStorageProfile,
      ),
      repeatedRange(evaluationCount, 2, 25),
      true,
    );
  }
  add(
    "output scratch normalized",
    bindings.outputScratch.normalized,
    hiddenBytes,
    repeated(26),
    true,
  );
  add(
    "output scratch modulation",
    bindings.outputScratch.modulation,
    activation(product(OUTPUT_MODULATION_GROUPS, plan.batch, ACE_TURBO_DIT_CONFIG.hiddenSize)),
    repeated(26),
    true,
  );
  add(
    "output scratch modulated",
    bindings.outputScratch.modulated,
    hiddenBytes,
    repeated(26),
    true,
  );
  add(
    "velocity",
    bindings.velocity,
    activation(plan.latentElements),
    repeatedRange(evaluationCount, 26, 27),
    true,
  );
  add(
    "predicted clean latent",
    bindings.predictedCleanLatent,
    activation(plan.latentElements),
    repeated(27),
    true,
  );

  addWeightLifetimes(resources, weights, allGraph);
  validateLifetimeAliasing(resources);
}

interface QuantumInterval {
  readonly first: number;
  readonly last: number;
}

interface UsedBindingRange {
  readonly buffer: GPUBuffer;
  readonly start: number;
  readonly end: number;
}

interface BindingLifetime {
  readonly label: string;
  readonly range: UsedBindingRange;
  readonly intervals: readonly QuantumInterval[];
  readonly writable: boolean;
}

function addWeightLifetimes(
  resources: BindingLifetime[],
  weights: AceDitResolvedPackageWeights,
  intervals: readonly QuantumInterval[],
): void {
  const addObject = (prefix: string, value: object): void => {
    for (const [name, binding] of Object.entries(value)) {
      if (isGpuBinding(binding)) {
        const range = usedBindingRange(
          binding,
          binding.size ?? binding.buffer.size - (binding.offset ?? 0),
          `${prefix} ${name}`,
        );
        resources.push({
          label: `${prefix} ${name}`,
          range,
          intervals,
          writable: false,
        });
      } else if (typeof binding === "object" && binding !== null) {
        addObject(`${prefix} ${name}`, binding);
      }
    }
  };
  addObject("weight", weights);
}

function validateLifetimeAliasing(resources: readonly BindingLifetime[]): void {
  for (let leftIndex = 0; leftIndex < resources.length; leftIndex += 1) {
    const left = resources[leftIndex]!;
    for (let rightIndex = leftIndex + 1; rightIndex < resources.length; rightIndex += 1) {
      const right = resources[rightIndex]!;
      if (!left.writable && !right.writable) continue;
      if (
        left.range.buffer !== right.range.buffer ||
        left.range.start >= right.range.end ||
        right.range.start >= left.range.end ||
        !intervalSetsOverlap(left.intervals, right.intervals)
      ) {
        continue;
      }
      throw new RangeError(
        `ACE DiT graph lifetime overlap: ${left.label} aliases ${right.label}`,
      );
    }
  }
}

function layerScratchElementRequirements(
  plan: AceDitEvaluationPlan,
): Readonly<Record<keyof AceDitLayerScratch, number>> {
  const hidden = plan.hiddenElements;
  const query = plan.queryElements;
  const keyValue = plan.selfKeyValueElements;
  const intermediate = plan.intermediateElements;
  return Object.freeze({
    modulation: product(LAYER_MODULATION_GROUPS, plan.batch, ACE_TURBO_DIT_CONFIG.hiddenSize),
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

function requirePinnedGraphShape(shape: AceDitEvaluationShape): AceDitEvaluationPlan {
  validateAceDitConfig(ACE_TURBO_DIT_CONFIG);
  if (
    ACE_TURBO_DIT_CONFIG.layerCount !== ACE_DIT_LAYER_COUNT ||
    ACE_TURBO_DENOISING_EVALUATIONS !== 8
  ) {
    throw new Error("ACE DiT graph constants diverged from the pinned product contract");
  }
  return planAceDitEvaluation(ACE_TURBO_DIT_CONFIG, shape);
}

function appendQuantum(
  quanta: AceDitGraphQuantum[],
  value:
    | Readonly<{ kind: "condition-projection" }>
    | Readonly<{ kind: "cross-cache"; layer: number }>
    | Readonly<{
        kind: "timestep" | "input-projection" | "output-projection" | "sampler";
        evaluation: number;
      }>
    | Readonly<{ kind: "layer"; evaluation: number; layer: number }>,
): void {
  const index = quanta.length;
  const label = value.kind === "condition-projection"
    ? "ace-dit-condition-projection"
    : value.kind === "cross-cache"
      ? `ace-dit-cross-cache-${value.layer}`
      : value.kind === "layer"
        ? `ace-dit-eval-${value.evaluation}-layer-${value.layer}`
        : `ace-dit-eval-${value.evaluation}-${value.kind}`;
  quanta.push(Object.freeze({ ...value, index, label }) as AceDitGraphQuantum);
}

function repeatedRange(
  evaluationCount: AceDitSamplerEvaluationCount,
  firstOffset: number,
  lastOffset: number,
): readonly QuantumInterval[] {
  return Object.freeze(Array.from(
    { length: evaluationCount },
    (_, evaluation) => ({
      first: STATIC_QUANTA + evaluation * EVALUATION_QUANTA + firstOffset,
      last: STATIC_QUANTA + evaluation * EVALUATION_QUANTA + lastOffset,
    }),
  ));
}

function intervalSetsOverlap(
  left: readonly QuantumInterval[],
  right: readonly QuantumInterval[],
): boolean {
  return left.some((leftInterval) => right.some((rightInterval) =>
    leftInterval.first <= rightInterval.last &&
    rightInterval.first <= leftInterval.last
  ));
}

function requireBindingBytes(
  binding: GPUBufferBinding,
  requiredBytes: number,
  label: string,
): void {
  usedBindingRange(binding, requiredBytes, label);
}

function usedBindingRange(
  binding: GPUBufferBinding,
  requiredBytes: number,
  label: string,
): UsedBindingRange {
  const start = binding.offset ?? 0;
  const exposed = binding.size ?? binding.buffer.size - start;
  if (
    !Number.isSafeInteger(start) ||
    start < 0 ||
    !Number.isSafeInteger(exposed) ||
    !Number.isSafeInteger(requiredBytes) ||
    requiredBytes <= 0 ||
    exposed < requiredBytes ||
    start + exposed > binding.buffer.size
  ) {
    throw new RangeError(`${label} binding does not expose ${requiredBytes} bytes`);
  }
  return Object.freeze({ buffer: binding.buffer, start, end: start + requiredBytes });
}

function isGpuBinding(value: unknown): value is GPUBufferBinding {
  return (
    typeof value === "object" &&
    value !== null &&
    "buffer" in value &&
    typeof (value as { readonly buffer?: unknown }).buffer === "object"
  );
}

function product(...values: number[]): number {
  let result = 1;
  for (const value of values) {
    if (!Number.isSafeInteger(value) || value <= 0) {
      throw new RangeError("ACE DiT graph product requires positive safe integers");
    }
    result *= value;
    if (!Number.isSafeInteger(result)) {
      throw new RangeError("ACE DiT graph byte accounting exceeds safe integers");
    }
  }
  return result;
}

function sum(values: readonly number[]): number {
  let result = 0;
  for (const value of values) {
    if (!Number.isSafeInteger(value) || value < 0) {
      throw new RangeError("ACE DiT graph sum requires non-negative safe integers");
    }
    result += value;
    if (!Number.isSafeInteger(result)) {
      throw new RangeError("ACE DiT graph byte accounting exceeds safe integers");
    }
  }
  return result;
}
