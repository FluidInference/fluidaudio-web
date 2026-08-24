import {
  ACE_CHANNEL_COUNT,
  ACE_SAMPLE_RATE_HZ,
  assertAceGenerationRequest,
  resolveAceDynamicConditionalWeighting,
  type AceGenerationRequest,
  type AceGenerationResult,
  type AceVaeSchedulingReceipt,
  type AceVaeSchedulingSelection,
  type AceVaeWindowSchedulingReceipt,
} from "../api.js";
import {
  acquireAceModelFiles,
  aceRuntimePackageFiles,
  type AceAcquiredModelFiles,
  type AceCacheAuthenticationOwner,
  type AceModelAcquisitionProgress,
  type AceModelAcquisitionTrace,
} from "../model/acquire.js";
import { AceOpfsModelCache } from "../model/cache.js";
import { AceGpuTensorPhase } from "../model/gpu-tensors.js";
import type {
  AceGpuTensorPhaseProgress,
  AceGpuTensorPhaseUploadTrace,
} from "../model/gpu-tensors.js";
import { aceSha256Hex } from "../model/sha256.js";
import {
  deriveAceDurationGraphShape,
} from "../model/graph-contract.js";
import {
  ACE_EXPERIMENTAL_VAE_PACKAGE_CONVERTER_REVISION,
  ACE_EXPERIMENTAL_VAE_PARAMETER_BYTES,
  ACE_OPT_0054_EXPERIMENTAL_VAE_PACKAGE_CONVERTER_REVISION,
  type AcePackageManifest,
  type AcePackageProfile,
  type AceTensorPhase,
} from "../model/manifest.js";
import {
  loadAcePackageManifest,
  ACE_OPT_0054_VAE_REVISION7_MANIFEST_BYTES,
  ACE_OPT_0054_VAE_REVISION7_MANIFEST_SHA256,
  ACE_REFERENCE_MANIFEST_SHA256,
  type AceLoadedPackageManifest,
} from "../model/package.js";
import {
  formatAceTextEncoderCaptionInput,
  formatAceTextEncoderLyricsInput,
} from "../tokenizer/conditioning-text.js";
import {
  loadPinnedAceTokenizer,
  type AceTokenizerAssetBundle,
  type LoadedAceTokenizer,
} from "../tokenizer/loader.js";
import type { AceQwenBpeTokenizer } from "../tokenizer/qwen-bpe.js";
import type { AcePipelineBackend } from "./pipeline.js";
import {
  ACE_PLANNER_SOURCE_REVISION,
  ACE_SOURCE_REVISION,
  PARAKEET_REFERENCE_REVISION,
  type AceDiagnostic,
  type AceRuntimeDiagnostics,
} from "./diagnostics.js";
import {
  canonicalizeAceGenerationRequest,
  resolveAceConditioningText,
} from "./generation-inputs.js";
import {
  ACE_PLANNER_COT_MAX_NEW_TOKENS,
  runAcePlannerCoordinator,
  type AcePlannerCoordinatorOptions,
  type AcePlannerCoordinatorProgress,
  type AcePlannerCoordinatorResult,
} from "./planner-coordinator.js";
import type {
  AceGenerationContext,
  AceInitializationContext,
} from "./pipeline.js";
import {
  type AceFatalGpuErrorCode,
  type AceWorkerConfiguration,
  type AceWorkerDitDensePackageConfiguration,
  type AceWorkerVaePackageConfiguration,
} from "./protocol.js";
import { fillAceDiffusionNoise } from "./seed.js";
import {
  ACE_INITIALIZATION_STAGES,
  generationStagePlan,
  type AceGenerationProgress,
  type AceGenerationStage,
  type AceInitializationProgress,
  type AceInitializationStage,
  type AceProgressUnit,
  type AceStageTiming,
} from "./stages.js";
import {
  AceAudioOutputTransaction,
  recoverStaleAceAudioOutputs,
  releaseAceAudioOutput,
  type AceAudioOutputCommitOptions,
  type AceAudioOutputStorage,
  type AceAudioOutputTransactionTrace,
  type AceCommittedAudioOutput,
} from "./audio-output.js";
import {
  ACE_REFERENCE_PORTABLE_PROFILE,
  ACE_REFERENCE_SUBGROUP_PROFILE,
  type AceKernelBackend,
  type AceModelProfileId,
} from "../webgpu/capabilities.js";
import {
  AceConditioningGpuExecutor,
  type AceConditioningGpuExecutorOptions,
  type AceConditioningGpuProgress,
  type AceConditioningGpuRequest,
  type AceConditioningGpuResult,
} from "../webgpu/conditioning-executor.js";
import {
  AceDitGpuBackend,
  AceOpt0081RepresentativeDitOwner,
  AceOpt0081RepresentativeInjectedSetupFailure,
  type AceOpt0018DitCommandProfile,
  type AceOpt0034DitSchedulingProfile,
  type AceDitGpuBackendOptions,
  type AceDitGpuBackendProgress,
  type AceDitGpuBackendResult,
  type AceOpt0062AttentionIdentityResult,
  type AceOpt0062AttentionIdentityRoute,
  type AceOpt0067DitCommandProfile,
  type AceOpt0080DitCommandProfile,
  type AceOpt0080FullDitCommandProfile,
  type AceOpt0081RepresentativeDitOwnerOptions,
  type AceOpt0081RepresentativeDitSession,
  type AceOpt0081RepresentativeSetupCleanupEvidence,
} from "../webgpu/dit-backend.js";
import {
  ACE_DIT_QUERY8_ATTENTION_KERNEL_SET_ID,
  ACE_OPT_0062_DIT_QUAD_QUERY_ATTENTION_KERNEL_SET_ID,
  ACE_OPT_0062_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
  ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_KERNEL_SET_ID,
  ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
  ACE_DIT_QUERY8_ATTENTION_RUNTIME_PROFILE,
  resolveAceDitAttentionKernelSetId,
} from "../webgpu/dit-attention-profile.js";
import type {
  AceDitSubmissionPolicy,
  AceDitOpt0080CommandBufferCompletion,
  AceDitOpt0080SchedulingProfile,
  AceDitPhysicalQuantaPerCommandBuffer,
} from "../webgpu/dit-graph.js";
import {
  ACE_OPT_0009_DIT_DENSE_CONVERTER_REVISION,
  ACE_OPT_0009_DIT_DENSE_KERNEL_SET_ID,
  ACE_OPT_0009_DIT_DENSE_MANIFEST_BYTES,
  ACE_OPT_0009_DIT_DENSE_MANIFEST_SHA256,
  ACE_OPT_0009_DIT_MIXED_RESIDENT_WEIGHT_BYTES,
  ACE_OPT_0009_DIT_DENSE_RUNTIME_PROFILE,
  ACE_OPT_0009_DIT_DENSE_WEIGHT_FILES,
  ACE_OPT_0009_DIT_MIXED_LAYER_BYTES,
  ACE_OPT_0037_DIT_K4_KERNEL_SET_ID,
  ACE_OPT_0037_DIT_K4_LAYER_BYTES,
  ACE_OPT_0037_DIT_K4_MANIFEST_BYTES,
  ACE_OPT_0037_DIT_K4_MANIFEST_SHA256,
  ACE_OPT_0037_DIT_K4_RESIDENT_WEIGHT_BYTES,
  ACE_OPT_0037_DIT_K4_RUNTIME_PROFILE,
  ACE_OPT_0037_DIT_K4_WEIGHT_FILES,
  ACE_OPT_0056_DIT_SELECTIVE_K4_KERNEL_SET_ID,
  ACE_OPT_0056_DIT_SELECTIVE_K4_RUNTIME_PROFILE,
  ACE_OPT_0088_DIT_DENSE_PORTABLE_KERNEL_SET_ID,
  createAceReferenceDitSharedManifestView,
  isAceReferenceDitLayerWeightFile,
  requireAceOpt0009DitDensePackageIdentity,
  requireAceOpt0037DitK4PackageIdentity,
} from "../webgpu/dit-fp16-package.js";
import {
  ACE_TURBO_EIGHT_SAMPLER_SCHEDULE_PROFILE,
  resolveAceDitSamplerScheduleProfile,
  type AceDitSamplerScheduleProfile,
  type AceDitSamplerScheduleProfileId,
} from "../webgpu/dit-sampler-profile.js";
import type { AceOpt0056DenseRouteProfile } from
  "../webgpu/kernels/dit-dense-fp16-k4-selective-exact.js";
import type { AceOpt0062AttentionRouteProfile } from
  "../webgpu/kernels/attention-quad-query-production.js";
import {
  requestAceWebGpuDevice,
  type AceGpuRuntimeEvent,
  type AceRequestWebGpuDeviceOptions,
} from "../webgpu/device.js";
import {
  type AceVaeChunkGpuBackendProgress,
} from "../webgpu/vae-backend.js";
import {
  ACE_OPT_0027_VAE_FP16_WINDOW_QUANTA_PER_COMMAND_BUFFER,
  ACE_OPT_0080_VAE_PRODUCTION_SCHEDULING_POLICY,
  AceOpt0011Fp16VaeChunkGpuBackend,
  resolveOpt0080VaeProductionWindowSchedulingProfile,
  type AceOpt0011Fp16VaeChunkGpuBackendOptions,
  type AceOpt0011Fp16VaeProfileFamily,
  type AceOpt0011Fp16VaeWindowFamilyProfile,
  type AceOpt0080VaeProductionSchedulingPolicy,
  type AceOpt0080VaeSchedulingEvidence,
  type AceOpt0080VaeSchedulingProfile,
} from "../webgpu/vae-fp16-backend.js";
import {
  ACE_OPT_0011_VAE_FP16_WINDOW_QUANTA_PER_COMMAND_BUFFER,
} from "../webgpu/vae-fp16-decoder.js";
import {
  ACE_OPT_0070_VAE_C2378_WINDOW_RUNTIME_PROFILE,
  ACE_VAE_CAPPED_C2176_WINDOW_RUNTIME_PROFILE,
  requireAceVaeWindowRuntimeProfile,
  selectAceVaeWindowRuntimeProfileForLimits,
  type AceVaeWindowRuntimeProfileContract,
} from "../webgpu/vae-window-profile.js";
import {
  ACE_OPT_0011_VAE_FP16_WEIGHT_FILES,
} from "../webgpu/vae-fp16-package.js";
import {
  ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_PROFILE,
  ACE_OPT_0028_VAE_FP16_MANIFEST_BYTES,
  ACE_OPT_0028_VAE_FP16_MANIFEST_SHA256,
  ACE_OPT_0040_VAE_FP16_FIXED32_SHAPE_SELECTED_PROFILE,
  ACE_OPT_0054_VAE_FP16_FIXED32_REVISION7_PROFILE,
  ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PROFILE,
  ACE_OPT_0072_VAE_FP16_FIXED32_DUAL_K4_PRODUCTION_RUNTIME_PROFILE,
  ACE_OPT_0088_VAE_FP16_PORTABLE_DUAL_K4_PROFILE,
  requireAceOpt0072VaeProductionRuntimeProfileForBackend,
  requireAceOpt0028Fp16VaePackageIdentity,
  requireAceOpt0054Fp16VaePackageIdentity,
  requireAceOpt0066Fp16VaePackageIdentity,
} from "../webgpu/vae-fp16-profile.js";
import {
  deriveAceVaePostprocessPlan,
  planAceVaeChunkedDecode,
  streamAceVaeRawChunks,
  type AceVaeChunkedDecodePlan,
  type AceVaeDecodeWindow,
  type AceVaeRawChunkSink,
  type AceVaeRawStreamTrace,
  type AceVaeRawStreamStats,
} from "../webgpu/vae-chunks.js";
import type { AceVaeWavWriteTrace } from "../webgpu/vae-wav.js";
import {
  ACE_TEXT_ENCODER_MAX_LYRIC_TOKENS,
  ACE_TEXT_ENCODER_MAX_TEXT_TOKENS,
} from "../webgpu/text-encoder.js";

const SAFE_AUDIO_STORAGE_ID = /^(?!\.{1,2}$)[A-Za-z0-9._-]{1,96}$/;
const OPT_0080_WALL_RECONCILIATION_TOLERANCE_MS = 1e-6;
const ACE_PRODUCTION_DIT_SUBMISSION_POLICY: AceDitSubmissionPolicy =
  "depth2-phase-epoch4";

class AceFatalGpuError extends Error {
  override readonly name = "AceFatalGpuError";

  constructor(
    readonly code: AceFatalGpuErrorCode,
    message: string,
  ) {
    super(message);
  }
}

export interface AceWebGpuPipelineOptions {
  readonly storage?: AceAudioOutputStorage & Pick<StorageManager, "estimate">;
  readonly fetch?: typeof fetch;
  readonly gpu?: GPU;
  readonly crypto?: Pick<Crypto, "randomUUID">;
  /** @internal OPT-0071 warm-cache owner screen; omission uses production. */
  readonly cacheAuthenticationOwner?: AceCacheAuthenticationOwner;
}

interface PipelineDeviceContext {
  readonly device: GPUDevice;
  readonly capabilities: AceRuntimeDiagnostics["capabilities"];
  readonly lost: Promise<AceGpuRuntimeEvent & { readonly type: "device-lost" }>;
  destroy(): void;
}

interface PipelineConditioningRunner {
  run(request: AceConditioningGpuRequest): Promise<AceConditioningGpuResult>;
  destroy(reason?: unknown): Promise<void>;
}

interface PipelineDitRunner {
  readonly memory: Readonly<{ readonly accountedGpuBytes: number }>;
  run(signal?: AbortSignal): Promise<AceDitGpuBackendResult>;
  destroy(reason?: unknown): Promise<void>;
}

export interface AceOpt0018DitCheckpoint {
  readonly schema: "ace-dit-m2250-checkpoint-v1";
  /** Detached final latent; the benchmark callback must not mutate it. */
  readonly finalLatent: Float32Array<ArrayBuffer>;
  readonly finalLatentByteLength: 1_152_000;
  readonly finalLatentElementCount: 288_000;
  readonly finalLatentSha256: string;
  readonly finalLatentNonFiniteCount: 0;
  readonly finalLatentNonzeroCount: number;
  readonly finalLatentMaxAbs: number;
  readonly profile: AceOpt0018DitCommandProfile;
  readonly stageTimings: readonly AceStageTiming[];
}

export interface AceOpt0034DitCheckpoint {
  readonly schema: "ace-dit-opt0034-m2250-checkpoint-v1";
  readonly finalLatent: Float32Array<ArrayBuffer>;
  readonly finalLatentByteLength: 1_152_000;
  readonly finalLatentElementCount: 288_000;
  readonly finalLatentSha256: string;
  readonly finalLatentNonFiniteCount: 0;
  readonly finalLatentNonzeroCount: number;
  readonly finalLatentMaxAbs: number;
  readonly profile: AceOpt0034DitSchedulingProfile;
  readonly stageTimings: readonly AceStageTiming[];
}

export interface AceOpt0034DitRunOptions {
  readonly physicalQuantaPerCommandBuffer:
    AceDitPhysicalQuantaPerCommandBuffer;
  readonly onCheckpoint: (checkpoint: AceOpt0034DitCheckpoint) => void;
}

export interface AceOpt0056DitEvaluationSnapshot {
  readonly evaluation: number;
  readonly latent: Float32Array<ArrayBuffer>;
  readonly latentByteLength: 1_152_000;
  readonly latentElementCount: 288_000;
  readonly latentSha256: string;
  readonly nonFiniteCount: 0;
  readonly nonzeroCount: number;
  readonly maximumAbsolute: number;
}

export interface AceOpt0056DitCheckpoint {
  readonly schema: "ace-dit-opt0056-m2250-trajectory-checkpoint-v1";
  readonly finalLatent: Float32Array<ArrayBuffer>;
  readonly finalLatentByteLength: 1_152_000;
  readonly finalLatentElementCount: 288_000;
  readonly finalLatentSha256: string;
  readonly finalLatentNonFiniteCount: 0;
  readonly finalLatentNonzeroCount: number;
  readonly finalLatentMaxAbs: number;
  readonly evaluations: readonly AceOpt0056DitEvaluationSnapshot[];
  readonly denseRouteProfile?: AceOpt0056DenseRouteProfile;
  readonly profile: AceOpt0018DitCommandProfile;
  readonly stageTimings: readonly AceStageTiming[];
  readonly graphCommandBufferCount: 2_553;
  readonly readbackCommandBufferCount: 1;
  readonly totalCommandBufferCount: 2_554;
  readonly snapshotCopyCount: 8;
  readonly snapshotExtraCommandBufferCount: 0;
  readonly snapshotExtraQueueDrainCount: 0;
}

export interface AceOpt0056DitRunOptions {
  readonly onCheckpoint: (checkpoint: AceOpt0056DitCheckpoint) => void;
}

export interface AceOpt0062DitCheckpoint {
  readonly schema: "ace-dit-opt0062-m2250-trajectory-checkpoint-v1";
  readonly finalLatent: Float32Array<ArrayBuffer>;
  readonly finalLatentByteLength: 1_152_000;
  readonly finalLatentElementCount: 288_000;
  readonly finalLatentSha256: string;
  readonly finalLatentNonFiniteCount: 0;
  readonly finalLatentNonzeroCount: number;
  readonly finalLatentMaxAbs: number;
  readonly evaluations: readonly AceOpt0056DitEvaluationSnapshot[];
  readonly attentionRouteProfile?: AceOpt0062AttentionRouteProfile;
  readonly actualLayerIdentity?: AceOpt0062AttentionIdentityResult;
  readonly profile: AceOpt0018DitCommandProfile;
  readonly stageTimings: readonly AceStageTiming[];
  readonly graphCommandBufferCount: 2_553;
  readonly readbackCommandBufferCount: 1;
  readonly totalCommandBufferCount: 2_554;
  readonly snapshotCopyCount: 8;
  readonly snapshotExtraCommandBufferCount: 0;
  readonly snapshotExtraQueueDrainCount: 0;
}

export interface AceOpt0062DitRunOptions {
  readonly onCheckpoint: (checkpoint: AceOpt0062DitCheckpoint) => void;
  /** Correctness arm only; timed arms omit the doubled attention oracle. */
  readonly captureActualLayerIdentity?: true;
}

export interface AceOpt0067AttentionIdentityResult {
  readonly schema: "ace-opt-0067-evaluation0-actual-layer-raw-u32-identity-v1";
  readonly routeCount: 12;
  readonly outputElementsPerRoute: 4_608_000;
  readonly totalComparedElements: 55_296_000;
  readonly totalMismatchCount: 0;
  readonly totalNonFiniteCount: 0;
  readonly totalCanaryCount: 0;
  readonly copyCount: 1;
  readonly extraCommandBufferCount: 0;
  readonly extraQueueDrainCount: 0;
  readonly inactiveFutureRouteCount: 84;
  readonly routes: readonly AceOpt0062AttentionIdentityRoute[];
}

export interface AceOpt0067DitCheckpoint {
  readonly schema: "ace-dit-opt0067-m2250-evaluation0-checkpoint-v1";
  readonly evaluation: 0;
  readonly result: Float32Array<ArrayBuffer>;
  readonly resultByteLength: 1_152_000;
  readonly resultElementCount: 288_000;
  readonly resultSha256: string;
  readonly resultNonFiniteCount: 0;
  readonly resultNonzeroCount: number;
  readonly resultMaxAbs: number;
  readonly attentionRouteProfile?: AceOpt0062AttentionRouteProfile;
  readonly actualLayerIdentity?: AceOpt0067AttentionIdentityResult;
  readonly profile: AceOpt0067DitCommandProfile;
  readonly stageTimings: readonly AceStageTiming[];
  readonly graphCommandBufferCount: 341;
  readonly readbackCommandBufferCount: 1;
  readonly totalCommandBufferCount: 342;
  readonly completedEvaluations: 1;
  readonly evaluationResultExtraCommandBufferCount: 0;
  readonly evaluationResultExtraQueueDrainCount: 0;
}

export interface AceOpt0067DitRunOptions {
  readonly onCheckpoint: (checkpoint: AceOpt0067DitCheckpoint) => void;
  /** Timed arms pause here after package/input preparation and graph compile. */
  readonly waitForTimingAuthorization?: () => Promise<void>;
  /** Correctness preparation only; timed arms retain ordinary arithmetic. */
  readonly captureActualLayerIdentity?: true;
}

export interface AceOpt0080DitCheckpoint {
  readonly schema: "ace-dit-opt0080-m2250-evaluation0-checkpoint-v1";
  readonly schedulingProfile: AceDitOpt0080SchedulingProfile;
  readonly evaluation: 0;
  readonly result: Float32Array<ArrayBuffer>;
  readonly resultByteLength: 1_152_000;
  readonly resultElementCount: 288_000;
  readonly resultSha256: string;
  readonly resultNonFiniteCount: 0;
  readonly resultNonzeroCount: number;
  readonly resultMaxAbs: number;
  readonly profile: AceOpt0080DitCommandProfile;
  readonly graphCommandBufferCount: 341;
  readonly readbackCommandBufferCount: 1;
  readonly totalCommandBufferCount: 342;
  readonly completedEvaluations: 1;
  readonly uncapturedWebGpuErrorCount: 0;
  readonly deviceLost: false;
}

export interface AceOpt0080DitRunOptions {
  readonly schedulingProfile: AceDitOpt0080SchedulingProfile;
  /** Timed arms pause after graph compilation and before phase zero starts. */
  readonly waitForTimingAuthorization?: () => Promise<void>;
  /** Cancellation-preflight observation after graph attribution. */
  readonly onCommandBufferCompleted?: (
    completion: AceDitOpt0080CommandBufferCompletion,
  ) => void;
  readonly onCheckpoint: (checkpoint: AceOpt0080DitCheckpoint) => void;
}

export type AceOpt0080FullEvaluationIndex = 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7;

export interface AceOpt0080FullDitEvaluationTap {
  readonly evaluation: AceOpt0080FullEvaluationIndex;
  readonly result: Float32Array<ArrayBuffer>;
  /** Raw bit identity; shares `result`'s ArrayBuffer without another song tensor. */
  readonly rawU32: Uint32Array<ArrayBuffer>;
  readonly sha256: string;
  readonly nonFiniteCount: 0;
  readonly nonzeroCount: number;
  readonly maxAbs: number;
}

export interface AceOpt0080FullDitCheckpoint {
  readonly schema: "ace-dit-opt0080-m2250-full-checkpoint-v1";
  readonly schedulingProfile: AceDitOpt0080SchedulingProfile;
  readonly captureEvaluationTaps: boolean;
  readonly finalLatent: Float32Array<ArrayBuffer>;
  /** Raw bit identity; shares `finalLatent`'s ArrayBuffer. */
  readonly finalLatentRawU32: Uint32Array<ArrayBuffer>;
  readonly finalLatentSha256: string;
  readonly finalLatentNonFiniteCount: 0;
  readonly finalLatentNonzeroCount: number;
  readonly finalLatentMaxAbs: number;
  /** Present only in untimed correctness arms; copies occur in sampler commands. */
  readonly evaluationTaps?: readonly AceOpt0080FullDitEvaluationTap[];
  readonly evaluationTapInCommandCopyCount: 0 | 8;
  readonly evaluationTapExtraCommandBufferCount: 0;
  readonly evaluationTapExtraQueueDrainCount: 0;
  readonly profile: AceOpt0080FullDitCommandProfile;
  readonly graphCommandBufferCount: 2_553;
  readonly readbackCommandBufferCount: 1;
  readonly totalCommandBufferCount: 2_554;
  readonly completedEvaluations: 8;
}

export interface AceOpt0080FullDitRunOptions {
  readonly schedulingProfile: AceDitOpt0080SchedulingProfile;
  /** Untimed correctness only; timed arms retain the ordinary final readback. */
  readonly captureEvaluationTaps?: true;
  readonly waitForTimingAuthorization?: () => Promise<void>;
  readonly onCommandBufferCompleted?: (
    completion: AceDitOpt0080CommandBufferCompletion,
  ) => void;
  readonly onCheckpoint: (checkpoint: AceOpt0080FullDitCheckpoint) => void;
}

export interface AceOpt0080ProductEvidence {
  readonly schema: "ace-opt-0080-product-integration-evidence-v1";
  readonly selectedProductionPolicy: AceDitSubmissionPolicy;
  readonly effectiveSubmissionPolicy: AceDitSubmissionPolicy;
  readonly finalLatent: Float32Array<ArrayBuffer>;
  /** Raw bit identity; shares `finalLatent`'s owned ArrayBuffer. */
  readonly finalLatentRawU32: Uint32Array<ArrayBuffer>;
  readonly rawStats: AceVaeRawStreamStats;
  readonly rawSnapshot: Blob;
  readonly vaePlan: AceVaeChunkedDecodePlan;
  /** Present only for an explicit forced VAE scheduling replay. */
  readonly vaeSchedulingEvidence?:
    readonly AceOpt0080VaeSchedulingEvidence[];
  readonly result: AceGenerationResult;
}

export type AceOpt0080ProductVaeSchedulingOverride =
  | "depth1-epoch1"
  | AceOpt0080VaeProductionSchedulingPolicy;

export interface AceOpt0080ProductRunOptions {
  /** Omit for ordinary selection; set only for forced control/candidate gate arms. */
  readonly submissionPolicyOverride?: AceDitSubmissionPolicy;
  /**
   * Forced VAE replay only. The candidate retains the production C2314-only
   * selector; it is never a backend-wide depth-two override.
   */
  readonly vaeSchedulingPolicyOverride?:
    AceOpt0080ProductVaeSchedulingOverride;
  readonly onEvidence: (evidence: AceOpt0080ProductEvidence) => void;
}

export interface AceOpt0064CaptureEvent {
  readonly schema: "ace-opt-0064-direct-request-capture-event-v1";
  readonly scope: "initialization" | "generation";
  readonly category:
    | "storage"
    | "manifest"
    | "authentication"
    | "upload"
    | "construction"
    | "execution"
    | "finalization"
    | "cleanup";
  readonly operation: string;
  readonly startedAtMs: number;
  readonly completedAtMs: number;
  readonly wallMs: number;
  readonly details: Readonly<Record<string, unknown>>;
}

export interface AceOpt0064CaptureSink {
  readonly onEvent: (event: AceOpt0064CaptureEvent) => void;
}

export interface AceWebGpuInitializationContext extends AceInitializationContext {
  /** @internal OPT-0064 capture-only request attribution seam. */
  readonly opt0064Capture?: AceOpt0064CaptureSink;
}

export interface AceOpt0081RepresentativeConditioningAuthority {
  readonly schema: "ace-opt-0081-representative-conditioning-authority-v1";
  readonly textTokenCount: number;
  readonly textTokenSha256: string;
  readonly lyricTokenCount: number;
  readonly lyricTokenSha256: string;
  readonly conditionTokens: number;
  readonly conditionElementCount: number;
  readonly conditionSha256: string;
  readonly contextElementCount: number;
  readonly contextSha256: string;
}

export type AceOpt0081RepresentativePipelineRunOptions =
  | Readonly<{
      readonly mode: "setup-failure";
      readonly onEvidence: (
        evidence: AceOpt0081RepresentativeSetupCleanupEvidence,
        conditioningAuthority:
          AceOpt0081RepresentativeConditioningAuthority,
      ) => void;
    }>
  | Readonly<{
      readonly mode: "run";
      readonly verifiedSetupFailureCleanup:
        AceOpt0081RepresentativeSetupCleanupEvidence;
      readonly run: (
        owner: AceOpt0081RepresentativeDitSession,
        setupFailureCleanup:
          AceOpt0081RepresentativeSetupCleanupEvidence,
        conditioningAuthority:
          AceOpt0081RepresentativeConditioningAuthority,
      ) => Promise<void>;
    }>;

export interface AceWebGpuGenerationContext extends AceGenerationContext {
  /** @internal OPT-0018 benchmark-only stop seam; stable worker omits it. */
  readonly onDitCheckpoint?: (checkpoint: AceOpt0018DitCheckpoint) => void;
  /** @internal OPT-0034 benchmark-only scheduling seam. */
  readonly opt0034DitRun?: AceOpt0034DitRunOptions;
  /** @internal OPT-0056 exact-down trajectory and route diagnostic seam. */
  readonly opt0056DitRun?: AceOpt0056DitRunOptions;
  /** @internal OPT-0062 exact quad-attention trajectory diagnostic seam. */
  readonly opt0062DitRun?: AceOpt0062DitRunOptions;
  /** @internal OPT-0067 thermally isolated evaluation-0 diagnostic seam. */
  readonly opt0067DitRun?: AceOpt0067DitRunOptions;
  /** @internal OPT-0080 exact depth-one/depth-two evaluation-0 screen. */
  readonly opt0080DitRun?: AceOpt0080DitRunOptions;
  /** @internal OPT-0080 exact full-eight confirmation; omission is production. */
  readonly opt0080FullDitRun?: AceOpt0080FullDitRunOptions;
  /** @internal OPT-0080 post-integration exact product A/B; never stops early. */
  readonly opt0080ProductRun?: AceOpt0080ProductRunOptions;
  /** @internal Frozen OPT-0081 two-invocation diagnostic; never product-selected. */
  readonly opt0081RepresentativeRun?:
    AceOpt0081RepresentativePipelineRunOptions;
  /** @internal OPT-0064 capture-only request attribution seam. */
  readonly opt0064Capture?: AceOpt0064CaptureSink;
  /** @internal OPT-0055/0065 quality gate; absent from worker/public protocol. */
  readonly opt0055Opt0065SamplerRun?: AceSamplerScheduleDiagnosticRunOptions;
}

export interface AceSamplerScheduleTensorEvidence {
  readonly values: Float32Array<ArrayBuffer>;
  readonly byteLength: number;
  readonly elementCount: number;
  readonly sha256: string;
  readonly nonFiniteCount: number;
  readonly nonzeroCount: number;
  readonly maximumAbsolute: number;
}

export interface AceSamplerScheduleEvaluationEvidence
  extends AceSamplerScheduleTensorEvidence {
  readonly evaluation: number;
}

export interface AceSamplerScheduleDiagnosticEvidence {
  readonly schema: "ace-opt-0055-0065-sampler-schedule-evidence-v1";
  readonly schedule: Readonly<AceDitSamplerScheduleProfile>;
  readonly evaluation0Velocity: AceSamplerScheduleTensorEvidence;
  readonly evaluations: readonly AceSamplerScheduleEvaluationEvidence[];
  readonly finalLatent: AceSamplerScheduleTensorEvidence;
  readonly rawAudio: Readonly<{
    readonly snapshot: Blob;
    readonly byteLength: number;
    readonly peak: number;
    readonly finiteSamples: number;
    readonly interleavedSamples: number;
  }>;
  readonly wav: Blob;
}

export interface AceSamplerScheduleDiagnosticRunOptions {
  readonly scheduleProfileId: AceDitSamplerScheduleProfileId;
  readonly onEvidence: (evidence: AceSamplerScheduleDiagnosticEvidence) => void;
}

interface PipelineVaeRunner {
  readonly memory: Readonly<{ readonly accountedGpuBytes: number }>;
  decodeWindow: AceOpt0011Fp16VaeChunkGpuBackend["decodeWindow"];
  destroy(reason?: unknown): Promise<void>;
}

interface PipelineAudioTransaction {
  readonly transactionId: string;
  readonly rawSink: AceVaeRawChunkSink;
  commit(
    postprocess: ReturnType<typeof deriveAceVaePostprocessPlan>,
    options?: AceAudioOutputCommitOptions,
  ): Promise<AceCommittedAudioOutput>;
  rollback(): Promise<void>;
}

/** @internal Dependency seam used by focused orchestration tests. */
export interface AceWebGpuPipelineDependencies {
  readonly now: () => number;
  readonly randomUUID: () => string;
  readonly ensureStorage: (
    storage: AceWebGpuPipelineOptions["storage"],
    signal: AbortSignal,
  ) => Promise<void>;
  readonly recoverAudio: (
    storage: AceWebGpuPipelineOptions["storage"],
    signal: AbortSignal,
  ) => Promise<void>;
  readonly loadManifest: (
    configuration: AceWorkerConfiguration,
    signal: AbortSignal,
  ) => Promise<AceLoadedPackageManifest>;
  readonly loadVaeManifest: (
    configuration: AceWorkerVaePackageConfiguration,
    signal: AbortSignal,
  ) => Promise<AceLoadedPackageManifest>;
  readonly loadDitDenseManifest: (
    configuration: AceWorkerDitDensePackageConfiguration,
    signal: AbortSignal,
  ) => Promise<AceLoadedPackageManifest>;
  readonly acquireModel: (options: Readonly<{
    kind: "main" | "dit-dense" | "vae";
    loaded: AceLoadedPackageManifest;
    modelSource: AceInitializationContext["modelSource"];
    storage: AceWebGpuPipelineOptions["storage"];
    signal: AbortSignal;
    onProgress: (progress: AceModelAcquisitionProgress) => void;
    onTrace?: (trace: AceModelAcquisitionTrace) => void;
    now?: () => number;
  }>) => Promise<AceAcquiredModelFiles>;
  readonly requestDevice: (
    options: AceRequestWebGpuDeviceOptions,
  ) => Promise<PipelineDeviceContext>;
  readonly loadTokenizer: typeof loadPinnedAceTokenizer;
  readonly loadPhase: (
    device: GPUDevice,
    manifest: AcePackageManifest,
    files: ReadonlyMap<string, File>,
    phases: readonly AceTensorPhase[],
    signal: AbortSignal,
    onProgress?: (progress: AceGpuTensorPhaseProgress) => void,
    onUploadTrace?: (trace: AceGpuTensorPhaseUploadTrace) => void,
    now?: () => number,
  ) => Promise<AceGpuTensorPhase>;
  readonly runPlanner: (
    options: AcePlannerCoordinatorOptions,
  ) => Promise<AcePlannerCoordinatorResult>;
  readonly createConditioning: (
    options: AceConditioningGpuExecutorOptions,
  ) => PipelineConditioningRunner;
  readonly createDit: (
    options: AceDitGpuBackendOptions,
  ) => Promise<PipelineDitRunner>;
  /** @internal Omitted by ordinary dependency fakes and product selection. */
  readonly createOpt0081Representative?: (
    options: AceOpt0081RepresentativeDitOwnerOptions,
  ) => Promise<AceOpt0081RepresentativeDitSession>;
  readonly createVae: (
    options: AceOpt0011Fp16VaeChunkGpuBackendOptions,
  ) => Promise<PipelineVaeRunner>;
  readonly beginAudio: (
    transactionId: string,
    plan: AceVaeChunkedDecodePlan,
    storage: AceWebGpuPipelineOptions["storage"],
  ) => Promise<PipelineAudioTransaction>;
  readonly releaseAudio: (
    transactionId: string,
    storage: AceWebGpuPipelineOptions["storage"],
  ) => Promise<void>;
}

interface ReadyResources {
  readonly configuration: AceWorkerConfiguration;
  readonly loaded: AceLoadedPackageManifest;
  readonly ditDenseLoaded: AceLoadedPackageManifest;
  readonly vaeLoaded: AceLoadedPackageManifest;
  readonly acquired: AceAcquiredModelFiles;
  readonly ditDenseAcquired: AceAcquiredModelFiles;
  readonly modelSource: AceInitializationContext["modelSource"];
  readonly deviceContext: PipelineDeviceContext;
  readonly textTokenizer: AceQwenBpeTokenizer;
  readonly plannerTokenizer: AceQwenBpeTokenizer;
  readonly diagnostics: AceRuntimeDiagnostics;
}

function selectProductionDitSubmissionPolicy(
  ready: ReadyResources,
  context: AceWebGpuGenerationContext,
  samplerScheduleProfile: Readonly<AceDitSamplerScheduleProfile>,
  plannerMode: AceGenerationRequest["planner"]["mode"],
): AceDitSubmissionPolicy | undefined {
  const configuration = ready.configuration;
  const hasDiagnosticSeam = [
    context.onDitCheckpoint,
    context.opt0034DitRun,
    context.opt0056DitRun,
    context.opt0062DitRun,
    context.opt0067DitRun,
    context.opt0080DitRun,
    context.opt0080FullDitRun,
    context.opt0081RepresentativeRun,
    context.opt0055Opt0065SamplerRun,
    context.opt0064Capture,
  ].some((seam) => seam !== undefined);
  if (
    plannerMode !== "disabled" ||
    configuration.manifestSha256 !== ACE_REFERENCE_MANIFEST_SHA256 ||
    configuration.modelProfile !== "reference-bf16" ||
    ready.deviceContext.capabilities.executionProfile.id !==
      ACE_REFERENCE_SUBGROUP_PROFILE.id ||
    configuration.ditDensePackage.runtimeProfile !==
      ACE_OPT_0009_DIT_DENSE_RUNTIME_PROFILE ||
    configuration.ditAttentionRuntimeProfile !==
      ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE ||
    configuration.vaePackage.manifestSha256 !==
      ACE_OPT_0054_VAE_REVISION7_MANIFEST_SHA256 ||
    configuration.vaePackage.runtimeProfile !==
      ACE_OPT_0072_VAE_FP16_FIXED32_DUAL_K4_PRODUCTION_RUNTIME_PROFILE ||
    configuration.vaePackage.windowRuntimeProfile !==
      ACE_OPT_0070_VAE_C2378_WINDOW_RUNTIME_PROFILE ||
    configuration.vaePackage.maxWindowFrames !== 2_378 ||
    samplerScheduleProfile !== ACE_TURBO_EIGHT_SAMPLER_SCHEDULE_PROFILE ||
    samplerScheduleProfile.role !== "production-default" ||
    samplerScheduleProfile.evaluationCount !== 8
  ) return undefined;
  if (hasDiagnosticSeam) return undefined;
  return ACE_PRODUCTION_DIT_SUBMISSION_POLICY;
}

function selectProductionVaeSchedulingPolicy(
  ready: ReadyResources,
  context: AceWebGpuGenerationContext,
  plannerMode: AceGenerationRequest["planner"]["mode"],
  selectedDitPolicy: AceDitSubmissionPolicy | undefined,
  vaeRuntimeIdentity: AceVaePackageRuntimeIdentity,
  effectiveWindowProfile: Readonly<AceVaeWindowRuntimeProfileContract>,
): AceOpt0080VaeProductionSchedulingPolicy | undefined {
  if (
    plannerMode !== "disabled" ||
    // OPT-0080's depth-two evidence covers only the C2378 window family; a
    // capped-adapter downshift keeps the audited depth-one baseline.
    effectiveWindowProfile.id !==
      ACE_OPT_0070_VAE_C2378_WINDOW_RUNTIME_PROFILE ||
    selectedDitPolicy !== ACE_PRODUCTION_DIT_SUBMISSION_POLICY ||
    context.captureTrace === true ||
    context.opt0080ProductRun !== undefined ||
    ready.configuration.schedulingProfile !== "cooperative" ||
    vaeRuntimeIdentity.role !== "opt-0072-rev7-production" ||
    vaeRuntimeIdentity.physicalRuntimeProfile !==
      ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PROFILE.id ||
    ready.vaeLoaded.manifest.provenance.converterRevision !==
      ACE_OPT_0054_EXPERIMENTAL_VAE_PACKAGE_CONVERTER_REVISION ||
    ready.configuration.vaePackage.windowRuntimeProfile !==
      ACE_OPT_0070_VAE_C2378_WINDOW_RUNTIME_PROFILE ||
    ready.configuration.vaePackage.maxWindowFrames !== 2_378 ||
    ACE_OPT_0027_VAE_FP16_WINDOW_QUANTA_PER_COMMAND_BUFFER !== 64
  ) return undefined;
  return ACE_OPT_0080_VAE_PRODUCTION_SCHEDULING_POLICY;
}

/**
 * Concrete Stage-1 browser pipeline owner. Multi-gigabyte files remain as OPFS
 * `File` handles until one authenticated GPU phase is loaded. Every heavyweight
 * phase transfers to its graph owner and is destroyed before the next begins.
 */
export class AceWebGpuPipelineBackend implements AcePipelineBackend {
  private ready: ReadyResources | undefined;
  private operation: "idle" | "initializing" | "generating" | "disposing" = "idle";
  private diagnosticSink: ((diagnostic: AceDiagnostic) => void) | undefined;
  private diagnosticStart = 0;
  private deviceLoss: Error | undefined;
  private gpuError: Error | undefined;
  private gpuEventToken: object | undefined;
  private readonly committedAudioIds = new Set<string>();
  private readonly pendingRollbacks = new Set<PipelineAudioTransaction>();
  private readonly pendingUnpublishedAudioIds = new Set<string>();

  /** @internal Prefer `createAceWebGpuPipelineBackend`. */
  constructor(
    private readonly options: AceWebGpuPipelineOptions,
    private readonly dependencies: AceWebGpuPipelineDependencies,
  ) {}

  async initialize(
    configuration: AceWorkerConfiguration,
    context: AceWebGpuInitializationContext,
  ): Promise<AceRuntimeDiagnostics> {
    if (this.operation !== "idle" || this.ready !== undefined) {
      throw new DOMException("ACE pipeline is already initialized or busy", "InvalidStateError");
    }
    if (configuration.schedulingProfile !== "cooperative") {
      throw new DOMException(
        "ACE Stage 1 supports only the cooperative scheduling profile",
        "NotSupportedError",
      );
    }
    requireProductionConfiguration(configuration);
    const vaeWindowProfile = requireAceVaeWindowRuntimeProfile(
      configuration.vaePackage.windowRuntimeProfile,
      configuration.vaePackage.maxWindowFrames,
    );
    this.operation = "initializing";
    this.deviceLoss = undefined;
    this.gpuError = undefined;
    const gpuEventToken = Object.freeze({});
    this.gpuEventToken = gpuEventToken;
    this.diagnosticSink = context.onDiagnostic;
    this.diagnosticStart = this.dependencies.now();
    const progress = new InitializationProgressReporter(
      context.onProgress,
      this.dependencies.now,
      this.diagnosticStart,
    );
    let deviceContext: PipelineDeviceContext | undefined;
    try {
      await this.retryPendingCleanup();
      context.signal.throwIfAborted();
      deviceContext = await captureOpt0064Async(
        this.dependencies.now,
        context.opt0064Capture,
        "initialization",
        "construction",
        "webgpu-device-request",
        Object.freeze({
          requiredFeatures: Object.freeze(["shader-f16"]),
          // The execution profile selected inside the device request adds its
          // own required features: fixed-32 subgroup adapters re-add
          // "subgroups"; portable adapters request only shader-f16.
          profileConditionalRequiredFeatures: Object.freeze(["subgroups"]),
        }),
        async () => await this.dependencies.requestDevice({
          modelProfile: configuration.modelProfile,
          schedulingProfile: configuration.schedulingProfile,
          requiredFeatures: ["shader-f16"],
          // The workspace capacity is adapter-aware: a one-GiB adapter that
          // cannot bind the configured C2378 workspace requests the capped
          // C2176 geometry instead; true deficits still fail closed.
          deriveRequiredLimits: (adapterLimits) => {
            const effective = selectAceVaeWindowRuntimeProfileForLimits(
              vaeWindowProfile,
              adapterLimits,
            );
            return {
              maxBufferSize: effective.requiredWorkspaceBytes,
              maxStorageBufferBindingSize: effective.requiredWorkspaceBytes,
            };
          },
          ...(this.options.gpu === undefined ? {} : { gpu: this.options.gpu }),
          signal: context.signal,
          onRuntimeEvent: (event) => {
            if (this.gpuEventToken === gpuEventToken) this.reportGpuEvent(event);
          },
        }),
      );
      requireProductionDeviceCapabilities(
        deviceContext,
        selectAceVaeWindowRuntimeProfileForLimits(
          vaeWindowProfile,
          deviceContext.capabilities.adapterLimits,
        ).requiredWorkspaceBytes,
      );
      const initializingDevice = deviceContext;
      void initializingDevice.lost.then((event) => {
        if (this.gpuEventToken !== gpuEventToken) return;
        this.deviceLoss ??= new AceFatalGpuError(
          "WEBGPU_DEVICE_LOST",
          `ACE WebGPU device lost (${event.reason}): ${event.message || "no device message"}`,
        );
      });
      this.throwIfGpuFailed();
      progress.complete("webgpu", 1, "items", "WebGPU device ready");

      await captureOpt0064Async(
        this.dependencies.now,
        context.opt0064Capture,
        "initialization",
        "storage",
        "opfs-open",
        Object.freeze({ bounded: true }),
        async () => await this.dependencies.ensureStorage(
          this.options.storage,
          context.signal,
        ),
      );
      await captureOpt0064Async(
        this.dependencies.now,
        context.opt0064Capture,
        "initialization",
        "cleanup",
        "stale-audio-recovery",
        Object.freeze({ crossTabLeasePreserved: true }),
        async () => await this.dependencies.recoverAudio(
          this.options.storage,
          context.signal,
        ),
      );
      progress.complete("storage", 1, "items", "OPFS ready");

      const loaded = await captureOpt0064Async(
        this.dependencies.now,
        context.opt0064Capture,
        "initialization",
        "manifest",
        "main-manifest-authentication",
        Object.freeze({ expectedSha256: configuration.manifestSha256 }),
        async () => await this.dependencies.loadManifest(configuration, context.signal),
      );
      requireLoadedManifestIdentity(loaded, configuration);
      const ditDenseLoaded = await captureOpt0064Async(
        this.dependencies.now,
        context.opt0064Capture,
        "initialization",
        "manifest",
        "dit-dense-manifest-authentication",
        Object.freeze({
          expectedSha256: configuration.ditDensePackage.manifestSha256,
        }),
        async () => await this.dependencies.loadDitDenseManifest(
          configuration.ditDensePackage,
          context.signal,
        ),
      );
      requireLoadedDitDenseManifestIdentity(
        ditDenseLoaded,
        configuration.ditDensePackage,
      );
      const vaeLoaded = await captureOpt0064Async(
        this.dependencies.now,
        context.opt0064Capture,
        "initialization",
        "manifest",
        "vae-manifest-authentication",
        Object.freeze({ expectedSha256: configuration.vaePackage.manifestSha256 }),
        async () => await this.dependencies.loadVaeManifest(
          configuration.vaePackage,
          context.signal,
        ),
      );
      requireLoadedVaeManifestIdentity(vaeLoaded, configuration.vaePackage);
      progress.complete(
        "manifest",
        loaded.manifestByteLength + ditDenseLoaded.manifestByteLength +
          vaeLoaded.manifestByteLength,
        "bytes",
        "Authenticated reference, exact mixed DiT, and packed VAE manifests",
      );

      const mainAcquisitionBytes = aceRuntimePackageFiles(
        createAceMainAcquisitionManifest(loaded.manifest),
      ).reduce((sum, file) => sum + file.byteLength, 0);
      const ditDenseIdentity = resolveAceDitDensePackageRuntimeIdentity(
        configuration.ditDensePackage,
      );
      const initializationWeightBytes = mainAcquisitionBytes +
        ditDenseIdentity.layerBytes;
      const acquired = await captureOpt0064Async(
        this.dependencies.now,
        context.opt0064Capture,
        "initialization",
        "authentication",
        "main-package-acquisition",
        Object.freeze({ expectedBytes: mainAcquisitionBytes }),
        async () => await this.dependencies.acquireModel({
          kind: "main",
          loaded,
          modelSource: context.modelSource,
          storage: this.options.storage,
          signal: context.signal,
          onProgress: (event) => {
            progress.report(
              "weights",
              event.completedBytes,
              initializationWeightBytes,
              "bytes",
              `${event.source}: ${event.file}`,
            );
          },
          ...(context.opt0064Capture === undefined
            ? {}
            : {
                onTrace: opt0064AcquisitionTraceSink(
                  this.dependencies.now,
                  context.opt0064Capture,
                  "initialization",
                  "main",
                ),
                now: this.dependencies.now,
              }),
        }),
      );
      emitOpt0064AcquisitionPlan(
        this.dependencies.now,
        context.opt0064Capture,
        "initialization",
        "main",
        acquired,
      );
      context.signal.throwIfAborted();
      if (acquired.plan.runtimeBytes !== mainAcquisitionBytes) {
        throw new Error("ACE main mixed-package acquisition byte total changed");
      }
      const ditDenseAcquired = await captureOpt0064Async(
        this.dependencies.now,
        context.opt0064Capture,
        "initialization",
        "authentication",
        "dit-dense-package-acquisition",
        Object.freeze({ expectedBytes: ditDenseIdentity.layerBytes }),
        async () => await this.dependencies.acquireModel({
          kind: "dit-dense",
          loaded: ditDenseLoaded,
          modelSource: context.modelSource,
          storage: this.options.storage,
          signal: context.signal,
          onProgress: (event) => {
            progress.report(
              "weights",
              mainAcquisitionBytes + event.completedBytes,
              initializationWeightBytes,
              "bytes",
              `${event.source}: ${event.file}`,
            );
          },
          ...(context.opt0064Capture === undefined
            ? {}
            : {
                onTrace: opt0064AcquisitionTraceSink(
                  this.dependencies.now,
                  context.opt0064Capture,
                  "initialization",
                  "dit-dense",
                ),
                now: this.dependencies.now,
              }),
        }),
      );
      emitOpt0064AcquisitionPlan(
        this.dependencies.now,
        context.opt0064Capture,
        "initialization",
        "dit-dense",
        ditDenseAcquired,
      );
      requireDitDenseAcquisition(ditDenseAcquired, ditDenseIdentity);
      context.signal.throwIfAborted();
      progress.complete(
        "weights",
        initializationWeightBytes,
        "bytes",
        "Authenticated reference and mixed DiT packages cached",
      );

      const textLoaded = await captureOpt0064Async(
        this.dependencies.now,
        context.opt0064Capture,
        "initialization",
        "construction",
        "text-tokenizer-load",
        Object.freeze({ tokenizer: "text" }),
        async () => await this.dependencies.loadTokenizer(
          "text",
          tokenizerAssets(acquired.files, "qwen"),
        ),
      );
      context.signal.throwIfAborted();
      const plannerLoaded = await captureOpt0064Async(
        this.dependencies.now,
        context.opt0064Capture,
        "initialization",
        "construction",
        "planner-tokenizer-load",
        Object.freeze({ tokenizer: "planner" }),
        async () => await this.dependencies.loadTokenizer(
          "planner",
          tokenizerAssets(acquired.files, "planner"),
        ),
      );
      context.signal.throwIfAborted();
      progress.complete("tokenizers", 2, "items", "Pinned tokenizers loaded");
      progress.skip("wasm", "No Stage-1 WASM component is required");

      const diagnostics = createRuntimeDiagnostics(
        configuration,
        loaded,
        ditDenseLoaded,
        vaeLoaded,
        deviceContext,
      );
      const ready: ReadyResources = Object.freeze({
        configuration: Object.freeze({
          ...configuration,
          ditDensePackage: Object.freeze({ ...configuration.ditDensePackage }),
          vaePackage: Object.freeze({ ...configuration.vaePackage }),
        }),
        loaded,
        ditDenseLoaded,
        vaeLoaded,
        acquired,
        ditDenseAcquired,
        modelSource: context.modelSource,
        deviceContext,
        textTokenizer: requireTokenizer(textLoaded, "text"),
        plannerTokenizer: requireTokenizer(plannerLoaded, "planner"),
        diagnostics,
      });
      this.ready = ready;
      this.throwIfGpuFailed();
      progress.complete("ready", 1, "items", "ACE runtime ready");
      return diagnostics;
    } catch (error) {
      if (this.gpuEventToken === gpuEventToken) this.gpuEventToken = undefined;
      deviceContext?.destroy();
      this.ready = undefined;
      throw error;
    } finally {
      this.operation = "idle";
      this.diagnosticSink = undefined;
    }
  }

  async generate(
    input: AceGenerationRequest,
    context: AceWebGpuGenerationContext,
  ): Promise<AceGenerationResult> {
    if (this.operation !== "idle") {
      throw new DOMException("ACE pipeline is busy", "InvalidStateError");
    }
    const ready = this.requireReady();
    const samplerScheduleRun = context.opt0055Opt0065SamplerRun;
    const samplerScheduleProfile = resolveAceDitSamplerScheduleProfile(
      samplerScheduleRun?.scheduleProfileId,
    );
    const opt0080ProductRun = context.opt0080ProductRun;
    const opt0056BenchmarkRuntime =
      ready.configuration.ditDensePackage.runtimeProfile ===
        ACE_OPT_0056_DIT_SELECTIVE_K4_RUNTIME_PROFILE;
    const opt0062BenchmarkRuntime =
      ready.configuration.ditAttentionRuntimeProfile ===
        ACE_OPT_0062_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE;
    if (opt0056BenchmarkRuntime && context.opt0056DitRun === undefined) {
      throw new Error(
        "OPT-0056 selective dense runtime is benchmark-only and requires its checkpoint seam",
      );
    }
    if (
      opt0062BenchmarkRuntime && context.opt0062DitRun === undefined &&
      context.opt0067DitRun === undefined
    ) {
      throw new Error(
        "OPT-0062 quad-query runtime is diagnostic-only and requires its checkpoint seam",
      );
    }
    if (
      samplerScheduleRun !== undefined &&
      (typeof samplerScheduleRun.onEvidence !== "function" ||
        context.captureTrace === true ||
        context.opt0064Capture !== undefined ||
        context.onDitCheckpoint !== undefined ||
        context.opt0034DitRun !== undefined ||
        context.opt0056DitRun !== undefined ||
        context.opt0062DitRun !== undefined ||
        context.opt0067DitRun !== undefined ||
        context.opt0080DitRun !== undefined ||
        context.opt0080FullDitRun !== undefined ||
        context.opt0081RepresentativeRun !== undefined ||
        ready.configuration.ditDensePackage.runtimeProfile !==
          ACE_OPT_0009_DIT_DENSE_RUNTIME_PROFILE ||
        (ready.configuration.ditAttentionRuntimeProfile ??
          ACE_DIT_QUERY8_ATTENTION_RUNTIME_PROFILE) !==
          ACE_DIT_QUERY8_ATTENTION_RUNTIME_PROFILE)
    ) {
      throw new Error(
        "OPT-0055/0065 schedule diagnostics require exact OPT-0009 dense/query8 and no legacy diagnostic seam",
      );
    }
    if ([
      context.onDitCheckpoint,
      context.opt0034DitRun,
      context.opt0056DitRun,
      context.opt0062DitRun,
      context.opt0067DitRun,
      context.opt0080DitRun,
      context.opt0080FullDitRun,
      context.opt0080ProductRun,
      context.opt0081RepresentativeRun,
      context.opt0055Opt0065SamplerRun,
    ].filter((seam) => seam !== undefined).length > 1) {
      throw new Error("ACE DiT benchmark checkpoint seams are mutually exclusive");
    }
    const opt0081RepresentativeRun = context.opt0081RepresentativeRun;
    const validOpt0081RepresentativeRun =
      opt0081RepresentativeRun === undefined ||
      (opt0081RepresentativeRun.mode === "setup-failure"
        ? typeof opt0081RepresentativeRun.onEvidence === "function" &&
          !("run" in opt0081RepresentativeRun)
        : opt0081RepresentativeRun.mode === "run"
        ? typeof opt0081RepresentativeRun.run === "function" &&
          opt0081RepresentativeRun.verifiedSetupFailureCleanup !== undefined &&
          !("onEvidence" in opt0081RepresentativeRun)
        : false);
    if (
      opt0081RepresentativeRun !== undefined &&
      (this.dependencies.createOpt0081Representative === undefined ||
        context.opt0064Capture !== undefined ||
        !validOpt0081RepresentativeRun)
    ) {
      throw new Error(
        "OPT-0081 representative diagnostics require the internal owner factory, one valid callback, and no API capture",
      );
    }
    if (
      context.opt0080DitRun !== undefined &&
      (context.opt0080DitRun.schedulingProfile !== "depth1-epoch1" &&
        context.opt0080DitRun.schedulingProfile !==
          "opt-0080-depth2-epoch4" ||
        typeof context.opt0080DitRun.onCheckpoint !== "function" ||
        context.opt0080DitRun.onCommandBufferCompleted !== undefined &&
          typeof context.opt0080DitRun.onCommandBufferCompleted !== "function" ||
        context.opt0080DitRun.waitForTimingAuthorization !== undefined &&
          typeof context.opt0080DitRun.waitForTimingAuthorization !== "function")
    ) {
      throw new Error("OPT-0080 requires one exact scheduling arm and checkpoint");
    }
    if (
      context.opt0080FullDitRun !== undefined &&
      (context.opt0080FullDitRun.schedulingProfile !== "depth1-epoch1" &&
          context.opt0080FullDitRun.schedulingProfile !==
            "opt-0080-depth2-epoch4" ||
        typeof context.opt0080FullDitRun.onCheckpoint !== "function" ||
        context.opt0080FullDitRun.captureEvaluationTaps !== undefined &&
          context.opt0080FullDitRun.captureEvaluationTaps !== true ||
        context.opt0080FullDitRun.onCommandBufferCompleted !== undefined &&
          typeof context.opt0080FullDitRun.onCommandBufferCompleted !==
            "function" ||
        context.opt0080FullDitRun.waitForTimingAuthorization !== undefined &&
          typeof context.opt0080FullDitRun.waitForTimingAuthorization !==
            "function")
    ) {
      throw new Error(
        "OPT-0080 full confirmation requires one exact scheduling arm and checkpoint",
      );
    }
    if (
      opt0080ProductRun !== undefined &&
      ((opt0080ProductRun.submissionPolicyOverride !== undefined &&
          opt0080ProductRun.submissionPolicyOverride !== "depth1-epoch1" &&
          opt0080ProductRun.submissionPolicyOverride !==
            "depth2-phase-epoch4") ||
        (opt0080ProductRun.vaeSchedulingPolicyOverride !== undefined &&
          opt0080ProductRun.vaeSchedulingPolicyOverride !==
            "depth1-epoch1" &&
          opt0080ProductRun.vaeSchedulingPolicyOverride !==
            ACE_OPT_0080_VAE_PRODUCTION_SCHEDULING_POLICY) ||
        typeof opt0080ProductRun.onEvidence !== "function" ||
        context.opt0064Capture !== undefined)
    ) {
      throw new Error(
        "OPT-0080 product integration requires one exact policy, evidence sink, and no API capture",
      );
    }
    assertAceGenerationRequest(input);
    const request = canonicalizeAceGenerationRequest(input);
    if (opt0081RepresentativeRun !== undefined) {
      const opt0081Shape = deriveAceDurationGraphShape(request.durationSeconds);
      if (
        request.planner.mode !== "disabled" ||
        opt0081Shape.latentFrames !== 4_500 ||
        ready.configuration.modelProfile !== "reference-bf16" ||
        ready.configuration.manifestSha256 !== ACE_REFERENCE_MANIFEST_SHA256 ||
        ready.configuration.ditDensePackage.manifestSha256 !==
          ACE_OPT_0009_DIT_DENSE_MANIFEST_SHA256 ||
        ready.configuration.ditDensePackage.runtimeProfile !==
          ACE_OPT_0009_DIT_DENSE_RUNTIME_PROFILE ||
        ready.configuration.ditAttentionRuntimeProfile !==
          ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE
      ) {
        throw new Error(
          "OPT-0081 representative diagnostics require the canonical direct M2250 reference/revision-7 OPT-0009/OPT-0070 tuple",
        );
      }
    }
    if (
      opt0080ProductRun !== undefined &&
      request.planner.mode !== "disabled"
    ) {
      throw new Error("OPT-0080 product integration evidence requires a direct request");
    }
    const selectedProductionDitSubmissionPolicy =
      selectProductionDitSubmissionPolicy(
        ready,
        context,
        samplerScheduleProfile,
        request.planner.mode,
      );
    if (
      opt0080ProductRun !== undefined &&
      selectedProductionDitSubmissionPolicy !==
        ACE_PRODUCTION_DIT_SUBMISSION_POLICY
    ) {
      throw new Error(
        "OPT-0080 product integration requires the exact authenticated production tuple",
      );
    }
    const ditSubmissionPolicy =
      opt0080ProductRun?.submissionPolicyOverride ??
      selectedProductionDitSubmissionPolicy;
    this.operation = "generating";
    this.diagnosticSink = context.onDiagnostic;
    this.diagnosticStart = this.dependencies.now();
    const tracker = new GenerationProgressTracker(
      generationStagePlan(request.planner.mode === "enabled"),
      context.onProgress,
      this.dependencies.now,
      this.diagnosticStart,
    );
    let transaction: PipelineAudioTransaction | undefined;
    let committed: AceCommittedAudioOutput | undefined;
    let conditioning: PipelineConditioningRunner | undefined;
    let dit: PipelineDitRunner | undefined;
    let ditReferencePhase: AceGpuTensorPhase | undefined;
    let ditDensePhase: AceGpuTensorPhase | undefined;
    let ditCommandProfile: AceOpt0018DitCommandProfile | undefined;
    let ditOpt0067Profile: AceOpt0067DitCommandProfile | undefined;
    let ditOpt0080Profile: AceOpt0080DitCommandProfile | undefined;
    let ditOpt0080FullProfile: AceOpt0080FullDitCommandProfile | undefined;
    let ditSchedulingProfile: AceOpt0034DitSchedulingProfile | undefined;
    let vae: PipelineVaeRunner | undefined;
    let queueDrains = 0;
    let cooperativeIdleMs = 0;
    let peakTrackedGpuBytes = 0;
    try {
      await captureOpt0064Async(
        this.dependencies.now,
        context.opt0064Capture,
        "generation",
        "cleanup",
        "retry-pending-cleanup",
        Object.freeze({ requestStartBoundary: true }),
        async () => await this.retryPendingCleanup(),
      );
      context.signal.throwIfAborted();
      this.throwIfGpuFailed();
      const preparedInputs = captureOpt0064Sync(
        this.dependencies.now,
        context.opt0064Capture,
        "generation",
        "construction",
        "request-input-and-noise-preparation",
        Object.freeze({
          durationSeconds: request.durationSeconds,
          deterministicSeed: request.seed,
        }),
        () => {
          const shape = deriveAceDurationGraphShape(request.durationSeconds);
          const initialLatent = new Float32Array(shape.latentFrames * 64);
          fillAceDiffusionNoise(initialLatent, request.seed);
          return Object.freeze({ shape, initialLatent });
        },
      );
      const { shape, initialLatent } = preparedInputs;
      tracker.complete("prepare", 1, "items", "Inputs and deterministic noise ready");

      const planner = await captureOpt0064Async(
        this.dependencies.now,
        context.opt0064Capture,
        "generation",
        "execution",
        "planner",
        Object.freeze({ mode: request.planner.mode }),
        async () => await this.runPlanner(request, ready, context, tracker),
      );
      const semanticCodeIds = planner.plannerMode === "enabled"
        ? Uint32Array.from(planner.semantic.semanticCodeValues)
        : undefined;
      queueDrains += planner.runtime.queueDrains;
      cooperativeIdleMs += planner.runtime.cooperativeIdleMs;
      peakTrackedGpuBytes = Math.max(
        peakTrackedGpuBytes,
        planner.runtime.peakAccountedGpuBytes,
      );
      const tokenized = captureOpt0064Sync(
        this.dependencies.now,
        context.opt0064Capture,
        "generation",
        "construction",
        "conditioning-tokenization",
        Object.freeze({ plannerMode: planner.plannerMode }),
        () => {
          const downstreamText = resolveAceConditioningText(
            planner.downstream,
            planner.plannerMode === "enabled",
          );
          return tokenizeConditioning(ready.textTokenizer, downstreamText);
        },
      );
      this.reportTrace(
        context.captureTrace === true,
        "ACE_PLANNER_CONDITIONING_RESOLVED",
        "Resolved planner and lyric conditioning inputs",
        "semantic-planner",
        () => Object.freeze({
          plannerEnabled: planner.plannerMode === "enabled",
          instrumental: planner.downstream.instrumental,
          resolvedCaption: planner.downstream.caption,
          captionCharacters: planner.downstream.caption.length,
          lyricCharacters: planner.downstream.lyrics.length,
          vocalLanguage: planner.downstream.vocalLanguage,
          metadata: JSON.stringify(planner.downstream.metadata),
          cotRan: planner.plannerMode === "enabled" && planner.cot !== null,
          phase2MetadataThinkBlock: planner.plannerMode === "enabled"
            ? planner.conditioning.cotText
            : null,
          generationProfile: request.generationProfile,
          plannerConfiguration: JSON.stringify(request.planner),
          samplingOracleId: planner.sampling?.oracleId ?? null,
          samplingFirstDraw: planner.sampling?.firstDraw.toString() ?? null,
          samplingFinalDraw: planner.sampling?.finalDraw.toString() ?? null,
          semanticCodeCount: semanticCodeIds?.length ?? 0,
          semanticCodeSha256: semanticCodeIds === undefined
            ? null
            : sha256U32Le(semanticCodeIds),
          semanticCodePrefix: semanticCodeIds === undefined
            ? []
            : [...semanticCodeIds.slice(0, 8)],
          semanticCodeIds: semanticCodeIds === undefined
            ? []
            : [...semanticCodeIds],
          textTokenCount: tokenized.textTokenIds.length,
          textTokenSha256: sha256U32Le(tokenized.textTokenIds),
          lyricTokenCount: tokenized.lyricTokenIds.length,
          lyricTokenSha256: sha256U32Le(tokenized.lyricTokenIds),
        }),
      );
      tracker.progress("text-encoder", 0, 2, "items", "Loading text encoder");
      const textPhase = await this.loadPhase(
        ready,
        ["text"],
        context.signal,
        (event) => mapPhaseUpload(event, tracker, "text-encoder", 0, 1, 2),
        context.opt0064Capture,
      );
      const conditioningApiCapture = createOpt0064GpuApiCapture(
        ready.deviceContext.device,
        this.dependencies.now,
        context.opt0064Capture,
        "conditioning",
      );
      conditioning = captureOpt0064Sync(
        this.dependencies.now,
        context.opt0064Capture,
        "generation",
        "construction",
        "conditioning-backend-construction",
        Object.freeze({ includesGraphConstruction: true }),
        () => this.dependencies.createConditioning({
        device: conditioningApiCapture.device,
        manifest: ready.loaded.manifest,
        modelProfile: ready.configuration.modelProfile,
        ownedTextWeights: textPhase,
        loadSemanticWeights: async (signal) => {
          completeIfCurrent(
            tracker,
            "text-encoder",
            2,
            "items",
            "Text embeddings complete",
          );
          tracker.progress(
            "semantic-detokenizer",
            0,
            2,
            "items",
            "Loading semantic detokenizer",
          );
          return await this.loadPhase(
            ready,
            ["semantic"],
            signal,
            (event) => mapPhaseUpload(
              event,
              tracker,
              "semantic-detokenizer",
              0,
              1,
              2,
            ),
            context.opt0064Capture,
          );
        },
        loadConditionerWeights: async (signal) => {
          completeIfCurrent(
            tracker,
            "text-encoder",
            2,
            "items",
            "Text embeddings complete",
          );
          completeIfCurrent(
            tracker,
            "semantic-detokenizer",
            planner.plannerMode === "enabled" ? 2 : 0,
            "items",
            planner.plannerMode === "enabled"
              ? "Semantic hints decoded"
              : "Skipped: planner disabled",
          );
          tracker.progress(
            "condition-encoder",
            0,
            2,
            "items",
            "Loading condition encoder",
          );
          return await this.loadPhase(
            ready,
            ["conditioner", "constants"],
            signal,
            (event) => mapPhaseUpload(
              event,
              tracker,
              "condition-encoder",
              0,
              1,
              2,
            ),
            context.opt0064Capture,
          );
        },
        signal: context.signal,
        onProgress: (event) => {
          mapConditioningProgress(
            event,
            planner.plannerMode,
            tracker,
          );
          if (event.completedCommandBuffers === event.totalCommandBuffers) {
            queueDrains += event.queueDrains;
            cooperativeIdleMs += event.cooperativeIdleMs;
          }
        },
        }),
      );
      const conditioned = await captureOpt0064Async(
        this.dependencies.now,
        context.opt0064Capture,
        "generation",
        "execution",
        "conditioning-execution",
        Object.freeze({
          includesLazyPipelineCompilation: true,
          includesGraphConstruction: true,
        }),
        async () => await conditioning!.run({
          ...tokenized,
          latentFrames: shape.latentFrames,
          mode: planner.plannerMode === "enabled"
            ? {
                kind: "planner",
                semanticCodeIds: semanticCodeIds!,
              }
            : { kind: "direct" },
        }),
      );
      conditioningApiCapture.finish();
      await captureOpt0064Async(
        this.dependencies.now,
        context.opt0064Capture,
        "generation",
        "cleanup",
        "conditioning-destroy",
        Object.freeze({ drainedBeforeRelease: true }),
        async () => await conditioning!.destroy(),
      );
      conditioning = undefined;
      completeConditioningStages(tracker, planner.plannerMode);
      peakTrackedGpuBytes = Math.max(
        peakTrackedGpuBytes,
        conditioned.memory.peakAccountedGpuBytes,
      );
      const lyricConditionElements = tokenized.lyricTokenIds.length *
        conditioned.hiddenSize;
      this.reportTrace(
        context.captureTrace === true,
        "ACE_CONDITIONING_TENSORS_READY",
        "Condition and context tensors are ready for DiT",
        "condition-encoder",
        () => {
          const conditionStats = finiteStats(conditioned.conditionHiddenStates);
          const contextStats = finiteStats(conditioned.contextLatents);
          const expectedConditionTokens =
            tokenized.lyricTokenIds.length + 1 + tokenized.textTokenIds.length;
          const lyricMaskAllValid = conditioned.conditionMask
            .slice(0, tokenized.lyricTokenIds.length)
            .every((value) => value === 1);
          const lyricConditionPrefixComplete =
            conditioned.conditionTokens === expectedConditionTokens &&
            conditioned.conditionMask.length === expectedConditionTokens &&
            conditioned.conditionHiddenStates.length ===
              expectedConditionTokens * conditioned.hiddenSize &&
            lyricMaskAllValid;
          return Object.freeze({
            mode: conditioned.mode,
            conditionTokens: conditioned.conditionTokens,
            lyricConditionRows: tokenized.lyricTokenIds.length,
            lyricConditionPrefixComplete,
            lyricConditionPrefixSha256: lyricConditionPrefixComplete
              ? sha256F32Le(
                  conditioned.conditionHiddenStates,
                  0,
                  lyricConditionElements,
                )
              : null,
            conditionSha256: sha256F32Le(conditioned.conditionHiddenStates),
            contextSha256: sha256F32Le(conditioned.contextLatents),
            conditionNonFinite: conditionStats.nonFinite,
            contextNonFinite: contextStats.nonFinite,
            conditionMaxAbs: conditionStats.maxAbs,
            contextMaxAbs: contextStats.maxAbs,
            conditionMaskValidRows: sumU32(conditioned.conditionMask),
          });
        },
      );
      tracker.complete(
        "release-conditioning",
        1,
        "items",
        "Conditioning GPU phases released",
      );

      tracker.progress("dit-load", 0, 3, "items", "Loading mixed DiT weights");
      const referenceSharedManifest = createAceReferenceDitSharedManifestView(
        ready.loaded.manifest,
      );
      ditReferencePhase = await this.loadPackagePhase(
        ready,
        referenceSharedManifest,
        ready.acquired.files,
        ["dit"],
        context.signal,
        (event) => mapPhaseUpload(event, tracker, "dit-load", 0, 1, 3),
        context.opt0064Capture,
        "main",
      );
      ditDensePhase = await this.loadPackagePhase(
        ready,
        ready.ditDenseLoaded.manifest,
        ready.ditDenseAcquired.files,
        ["dit"],
        context.signal,
        (event) => mapPhaseUpload(event, tracker, "dit-load", 1, 1, 3),
        context.opt0064Capture,
        "dit-dense",
      );
      if (opt0081RepresentativeRun !== undefined) {
        const conditioningAuthority =
          createOpt0081RepresentativeConditioningAuthority(
            tokenized,
            conditioned,
          );
        const factory = this.dependencies.createOpt0081Representative!;
        const ownedDitWeights = ditReferencePhase;
        const ownedDitDenseWeights = ditDensePhase;
        ditReferencePhase = undefined;
        ditDensePhase = undefined;
        const representativeOptions = Object.freeze({
          device: ready.deviceContext.device,
          executionProfile: ready.deviceContext.capabilities.executionProfile,
          ...(ready.deviceContext.capabilities.adapterInfo.subgroupMinSize ===
              undefined
            ? {}
            : {
                subgroupMinSize:
                  ready.deviceContext.capabilities.adapterInfo.subgroupMinSize,
              }),
          ...(ready.deviceContext.capabilities.adapterInfo.subgroupMaxSize ===
              undefined
            ? {}
            : {
                subgroupMaxSize:
                  ready.deviceContext.capabilities.adapterInfo.subgroupMaxSize,
              }),
          shape: {
            batch: 1,
            latentFrames: shape.latentFrames,
            conditionTokens: conditioned.conditionTokens,
          },
          inputs: {
            condition: conditioned.conditionHiddenStates,
            context: conditioned.contextLatents,
            initialLatent,
          },
          dcwConfiguration:
            resolveAceDynamicConditionalWeighting(request.planner),
          ownedDitWeights,
          ownedDitDenseWeights,
          ditDenseRuntimeProfile:
            ready.configuration.ditDensePackage.runtimeProfile,
          ditAttentionRuntimeProfile:
            ready.configuration.ditAttentionRuntimeProfile!,
          signal: context.signal,
          now: this.dependencies.now,
        } satisfies AceOpt0081RepresentativeDitOwnerOptions);
        if (opt0081RepresentativeRun.mode === "setup-failure") {
          let evidence: AceOpt0081RepresentativeSetupCleanupEvidence |
            undefined;
          let injected = false;
          let unexpectedOwner: AceOpt0081RepresentativeDitSession | undefined;
          try {
            unexpectedOwner = await factory({
              ...representativeOptions,
              setupFailurePoint: "after-readback",
              onSetupCleanup: (value) => {
                if (evidence !== undefined) {
                  throw new Error("OPT-0081 setup cleanup evidence repeated");
                }
                evidence = value;
              },
            });
          } catch (error) {
            if (
              error instanceof AceOpt0081RepresentativeInjectedSetupFailure &&
              error.point === "after-readback"
            ) injected = true;
            else throw error;
          } finally {
            await unexpectedOwner?.destroy();
          }
          if (!injected || evidence === undefined) {
            throw new Error("OPT-0081 setup failure preflight did not execute");
          }
          completeIfCurrent(
            tracker,
            "dit-load",
            3,
            "items",
            "OPT-0081 setup failure cleanup verified",
          );
          opt0081RepresentativeRun.onEvidence(
            evidence,
            conditioningAuthority,
          );
          context.signal.throwIfAborted();
          throw new Error(
            "OPT-0081 setup-failure diagnostic cannot continue into ordinary DiT or VAE",
          );
        }
        const owner = await factory({
          ...representativeOptions,
          verifiedSetupFailureCleanup:
            opt0081RepresentativeRun.verifiedSetupFailureCleanup,
        });
        peakTrackedGpuBytes = Math.max(
          peakTrackedGpuBytes,
          owner.memory.accountedGpuBytes,
        );
        completeIfCurrent(
          tracker,
          "dit-load",
          3,
          "items",
          "OPT-0081 two-layer owner loaded and compiled",
        );
        try {
          await opt0081RepresentativeRun.run(
            owner,
            opt0081RepresentativeRun.verifiedSetupFailureCleanup,
            conditioningAuthority,
          );
          context.signal.throwIfAborted();
          throw new Error(
            "OPT-0081 representative diagnostic cannot continue into ordinary DiT or VAE",
          );
        } finally {
          await owner.destroy();
        }
      }
      const ditApiCapture = createOpt0064GpuApiCapture(
        ready.deviceContext.device,
        this.dependencies.now,
        context.opt0064Capture,
        "dit",
      );
      dit = await captureOpt0064Async(
        this.dependencies.now,
        context.opt0064Capture,
        "generation",
        "construction",
        "dit-backend-pipeline-and-graph-construction",
        Object.freeze({
          includesPipelineCompilation: true,
          includesBindGroupConstruction: true,
          includesInputUpload: true,
        }),
        async () => await this.dependencies.createDit({
        device: ditApiCapture.device,
        executionProfile: ready.deviceContext.capabilities.executionProfile,
        ...(ready.deviceContext.capabilities.adapterInfo.subgroupMinSize === undefined
          ? {}
          : {
              subgroupMinSize:
                ready.deviceContext.capabilities.adapterInfo.subgroupMinSize,
            }),
        ...(ready.deviceContext.capabilities.adapterInfo.subgroupMaxSize === undefined
          ? {}
          : {
              subgroupMaxSize:
                ready.deviceContext.capabilities.adapterInfo.subgroupMaxSize,
            }),
        shape: {
          batch: 1,
          latentFrames: shape.latentFrames,
          conditionTokens: conditioned.conditionTokens,
        },
        inputs: {
          condition: conditioned.conditionHiddenStates,
          context: conditioned.contextLatents,
          initialLatent,
        },
        dcwConfiguration: resolveAceDynamicConditionalWeighting(request.planner),
        samplerScheduleProfile,
        ...(ditSubmissionPolicy === undefined
          ? {}
          : { ditSubmissionPolicy }),
        ownedDitWeights: ditReferencePhase!,
        ownedDitDenseWeights: ditDensePhase!,
        ditDenseRuntimeProfile:
          ready.configuration.ditDensePackage.runtimeProfile,
        ...(ready.configuration.ditAttentionRuntimeProfile === undefined
          ? {}
          : {
              ditAttentionRuntimeProfile:
                ready.configuration.ditAttentionRuntimeProfile,
            }),
        signal: context.signal,
        ...(context.onDitCheckpoint === undefined &&
            context.opt0056DitRun === undefined &&
            context.opt0062DitRun === undefined
          ? {}
          : {
              onCommandProfile: (profile: AceOpt0018DitCommandProfile) => {
                if (ditCommandProfile !== undefined) {
                  throw new Error("ACE benchmark emitted more than one DiT profile");
                }
                ditCommandProfile = profile;
              },
            }),
        ...(context.opt0067DitRun === undefined
          ? {}
          : {
              opt0067EvaluationLimit: 1 as const,
              onOpt0067EvaluationProfile: (
                profile: AceOpt0067DitCommandProfile,
              ) => {
                if (ditOpt0067Profile !== undefined) {
                  throw new Error(
                    "OPT-0067 emitted more than one evaluation profile",
                  );
                }
                ditOpt0067Profile = profile;
              },
            }),
        ...(context.opt0080DitRun === undefined
          ? {}
          : {
              opt0067EvaluationLimit: 1 as const,
              opt0080SchedulingProfile:
                context.opt0080DitRun.schedulingProfile,
              onOpt0080EvaluationProfile: (
                profile: AceOpt0080DitCommandProfile,
              ) => {
                if (ditOpt0080Profile !== undefined) {
                  throw new Error(
                    "OPT-0080 emitted more than one evaluation profile",
                  );
                }
                ditOpt0080Profile = profile;
              },
              ...(context.opt0080DitRun.onCommandBufferCompleted === undefined
                ? {}
                : {
                    onOpt0080CommandBufferCompleted:
                      context.opt0080DitRun.onCommandBufferCompleted,
                  }),
            }),
        ...(context.opt0080FullDitRun === undefined
          ? {}
          : {
              opt0080FullSchedulingProfile:
                context.opt0080FullDitRun.schedulingProfile,
              onOpt0080FullProfile: (
                profile: AceOpt0080FullDitCommandProfile,
              ) => {
                if (ditOpt0080FullProfile !== undefined) {
                  throw new Error(
                    "OPT-0080 emitted more than one full-graph profile",
                  );
                }
                ditOpt0080FullProfile = profile;
              },
              ...(context.opt0080FullDitRun.onCommandBufferCompleted ===
                  undefined
                ? {}
                : {
                    onOpt0080CommandBufferCompleted:
                      context.opt0080FullDitRun.onCommandBufferCompleted,
                  }),
            }),
        ...(context.opt0056DitRun === undefined &&
            context.opt0062DitRun === undefined &&
            context.opt0080FullDitRun?.captureEvaluationTaps !== true
          ? {}
          : { captureEvaluationLatents: true as const }),
        ...(context.opt0062DitRun?.captureActualLayerIdentity === true
          ? { captureOpt0062AttentionIdentity: true as const }
          : {}),
        ...(context.opt0067DitRun?.captureActualLayerIdentity === true
          ? { captureOpt0062AttentionIdentity: true as const }
          : {}),
        ...(samplerScheduleRun === undefined
          ? {}
          : { captureSamplerScheduleEvidence: true as const }),
        ...(context.opt0034DitRun === undefined
          ? {}
          : {
              physicalQuantaPerCommandBuffer:
                context.opt0034DitRun.physicalQuantaPerCommandBuffer,
              onSchedulingProfile: (
                profile: AceOpt0034DitSchedulingProfile,
              ) => {
                if (ditSchedulingProfile !== undefined) {
                  throw new Error("OPT-0034 emitted more than one DiT profile");
                }
                ditSchedulingProfile = profile;
              },
            }),
        onProgress: (event) => mapDitProgress(
          event,
          tracker,
          samplerScheduleProfile.evaluationCount,
        ),
        }),
      );
      ditApiCapture.finish();
      ditReferencePhase = undefined;
      ditDensePhase = undefined;
      peakTrackedGpuBytes = Math.max(
        peakTrackedGpuBytes,
        dit.memory.accountedGpuBytes,
      );
      completeIfCurrent(tracker, "dit-load", 3, "items", "DiT loaded and compiled");
      const waitForDitTimingAuthorization =
        context.opt0067DitRun?.waitForTimingAuthorization ??
        context.opt0080DitRun?.waitForTimingAuthorization ??
        context.opt0080FullDitRun?.waitForTimingAuthorization;
      if (waitForDitTimingAuthorization !== undefined) {
        await waitForDitTimingAuthorization();
        context.signal.throwIfAborted();
        this.throwIfGpuFailed();
      }
      const denoised = await captureOpt0064Async(
        this.dependencies.now,
        context.opt0064Capture,
        "generation",
        "execution",
        samplerScheduleRun === undefined
          ? "dit-eight-evaluation-execution-and-readback"
          : "dit-sampler-schedule-diagnostic-execution-and-readback",
        Object.freeze({
          evaluationCount: context.opt0067DitRun === undefined &&
              context.opt0080DitRun === undefined
            ? samplerScheduleProfile.evaluationCount
            : 1,
          samplerScheduleProfileId: samplerScheduleProfile.id,
        }),
        async () => await dit!.run(context.signal),
      );
      queueDrains += denoised.queueDrains;
      cooperativeIdleMs += denoised.cooperativeIdleMs;
      await captureOpt0064Async(
        this.dependencies.now,
        context.opt0064Capture,
        "generation",
        "cleanup",
        "dit-drain-and-destroy",
        Object.freeze({ hardBoundaryBeforeVae: true }),
        async () => await dit!.destroy(),
      );
      dit = undefined;
      completeIfCurrent(
        tracker,
        "dit-denoise",
        context.opt0067DitRun === undefined &&
            context.opt0080DitRun === undefined
          ? samplerScheduleProfile.evaluationCount
          : 1,
        "denoising-evaluations",
        context.opt0067DitRun === undefined &&
            context.opt0080DitRun === undefined
          ? samplerScheduleRun === undefined
            ? "Eight Turbo evaluations complete"
            : `${samplerScheduleProfile.evaluationCount} diagnostic evaluations complete`
          : context.opt0080DitRun === undefined
            ? "OPT-0067 evaluation 0 complete"
            : "OPT-0080 evaluation 0 complete",
      );
      tracker.complete("release-dit", 1, "items", "DiT GPU lifetime released");
      this.throwIfGpuFailed();

      if (context.onDitCheckpoint !== undefined) {
        if (context.captureTrace !== true || ditCommandProfile === undefined) {
          throw new Error(
            "OPT-0018 checkpoint requires one complete capture-enabled DiT profile",
          );
        }
        const checkpoint = createOpt0018DitCheckpoint(
          denoised,
          ditCommandProfile,
          tracker.timings,
        );
        this.reportTrace(
          true,
          "ACE_DIT_M2250_FAMILY_PROFILE",
          "Attributed every current-production M2250 DiT command drain",
          "release-dit",
          () => summarizeOpt0018DitCheckpoint(
            checkpoint,
            ready.configuration.ditDensePackage.runtimeProfile,
          ),
        );
        context.onDitCheckpoint(checkpoint);
        // The dedicated benchmark aborts with a private identity sentinel.
        // Stable generation omits the callback and reaches this check unchanged.
        context.signal.throwIfAborted();
      }
      if (context.opt0034DitRun !== undefined) {
        if (context.captureTrace !== true || ditSchedulingProfile === undefined) {
          throw new Error(
            "OPT-0034 checkpoint requires one complete scheduling profile",
          );
        }
        const checkpoint = createOpt0034DitCheckpoint(
          denoised,
          ditSchedulingProfile,
          tracker.timings,
        );
        this.reportTrace(
          true,
          "ACE_DIT_OPT0034_SCHEDULING_PROFILE",
          "Measured exact M2250 physical-quantum command-buffer coalescing",
          "release-dit",
          () => summarizeOpt0034DitCheckpoint(checkpoint),
        );
        context.opt0034DitRun.onCheckpoint(checkpoint);
        context.signal.throwIfAborted();
      }
      if (context.opt0056DitRun !== undefined) {
        if (context.captureTrace !== true || ditCommandProfile === undefined) {
          throw new Error(
            "OPT-0056 checkpoint requires one complete capture-enabled DiT profile",
          );
        }
        const checkpoint = createOpt0056DitCheckpoint(
          denoised,
          ditCommandProfile,
          tracker.timings,
          ready.configuration.ditDensePackage.runtimeProfile,
        );
        this.reportTrace(
          true,
          "ACE_DIT_OPT0056_TRAJECTORY",
          "Captured all eight sampler latents without extra submissions or drains",
          "release-dit",
          () => summarizeOpt0056DitCheckpoint(checkpoint),
        );
        context.opt0056DitRun.onCheckpoint(checkpoint);
        context.signal.throwIfAborted();
        if (opt0056BenchmarkRuntime) {
          throw new Error(
            "OPT-0056 selective dense runtime cannot continue beyond its checkpoint",
          );
        }
      }
      if (context.opt0062DitRun !== undefined) {
        if (context.captureTrace !== true || ditCommandProfile === undefined) {
          throw new Error(
            "OPT-0062 checkpoint requires one complete capture-enabled DiT profile",
          );
        }
        const checkpoint = createOpt0062DitCheckpoint(
          denoised,
          ditCommandProfile,
          tracker.timings,
          ready.configuration.ditAttentionRuntimeProfile,
          context.opt0062DitRun.captureActualLayerIdentity === true,
        );
        this.reportTrace(
          true,
          "ACE_DIT_OPT0062_QUAD_QUERY_TRAJECTORY",
          "Captured exact query8/quad sampler trajectory and route attribution",
          "release-dit",
          () => summarizeOpt0062DitCheckpoint(checkpoint),
        );
        context.opt0062DitRun.onCheckpoint(checkpoint);
        context.signal.throwIfAborted();
        throw new Error(
          "OPT-0062 diagnostic checkpoint cannot continue into VAE",
        );
      }
      if (context.opt0067DitRun !== undefined) {
        if (context.captureTrace !== true || ditOpt0067Profile === undefined) {
          throw new Error(
            "OPT-0067 checkpoint requires one capture-enabled evaluation profile",
          );
        }
        const checkpoint = createOpt0067DitCheckpoint(
          denoised,
          ditOpt0067Profile,
          tracker.timings,
          ready.configuration.ditAttentionRuntimeProfile,
          context.opt0067DitRun.captureActualLayerIdentity === true,
        );
        this.reportTrace(
          true,
          "ACE_DIT_OPT0067_EVALUATION0",
          "Captured one exact M2250 denoise evaluation with ordinary readback",
          "release-dit",
          () => summarizeOpt0067DitCheckpoint(checkpoint),
        );
        context.opt0067DitRun.onCheckpoint(checkpoint);
        context.signal.throwIfAborted();
        throw new Error(
          "OPT-0067 diagnostic checkpoint cannot continue to evaluation 1 or VAE",
        );
      }
      if (context.opt0080DitRun !== undefined) {
        if (context.captureTrace !== true || ditOpt0080Profile === undefined) {
          throw new Error(
            "OPT-0080 checkpoint requires one capture-enabled evaluation profile",
          );
        }
        const checkpoint = createOpt0080DitCheckpoint(
          denoised,
          ditOpt0080Profile,
          context.opt0080DitRun.schedulingProfile,
          ready.configuration.ditDensePackage.runtimeProfile,
          ready.configuration.ditAttentionRuntimeProfile,
        );
        this.reportTrace(
          true,
          "ACE_DIT_OPT0080_COMPLETION_EPOCHS",
          "Captured exact M2250 evaluation 0 with completion-epoch scheduling",
          "release-dit",
          () => summarizeOpt0080DitCheckpoint(checkpoint),
        );
        context.opt0080DitRun.onCheckpoint(checkpoint);
        context.signal.throwIfAborted();
        throw new Error(
          "OPT-0080 diagnostic checkpoint cannot continue to evaluation 1 or VAE",
        );
      }
      if (context.opt0080FullDitRun !== undefined) {
        if (
          context.captureTrace !== true ||
          ditOpt0080FullProfile === undefined
        ) {
          throw new Error(
            "OPT-0080 full checkpoint requires one capture-enabled graph profile",
          );
        }
        const checkpoint = createOpt0080FullDitCheckpoint(
          denoised,
          ditOpt0080FullProfile,
          context.opt0080FullDitRun,
          ready.configuration.ditDensePackage.runtimeProfile,
          ready.configuration.ditAttentionRuntimeProfile,
        );
        this.reportTrace(
          true,
          "ACE_DIT_OPT0080_FULL_COMPLETION_EPOCHS",
          "Captured exact full M2250 DiT graph with completion-epoch scheduling",
          "release-dit",
          () => summarizeOpt0080FullDitCheckpoint(checkpoint),
        );
        context.opt0080FullDitRun.onCheckpoint(checkpoint);
        context.signal.throwIfAborted();
        throw new Error(
          "OPT-0080 full diagnostic checkpoint cannot continue into VAE",
        );
      }

      const vaeWindowProfile = selectAceVaeWindowRuntimeProfileForLimits(
        requireAceVaeWindowRuntimeProfile(
          ready.configuration.vaePackage.windowRuntimeProfile,
          ready.configuration.vaePackage.maxWindowFrames,
        ),
        ready.deviceContext.capabilities.adapterLimits,
      );
      const vaeKernelBackend =
        ready.deviceContext.capabilities.executionProfile.kernelBackend;
      const vaeRuntimeIdentity = resolveAceVaePackageRuntimeIdentity(
        ready.configuration.vaePackage,
        vaeKernelBackend,
      );
      const vaeBackendRuntimeProfileId =
        vaeRuntimeIdentity.role === "opt-0072-rev7-production"
          ? vaeRuntimeIdentity.physicalRuntimeProfile
          : vaeRuntimeIdentity.runtimeProfile;
      const selectedProductionVaeSchedulingPolicy =
        selectProductionVaeSchedulingPolicy(
          ready,
          context,
          request.planner.mode,
          selectedProductionDitSubmissionPolicy,
          vaeRuntimeIdentity,
          vaeWindowProfile,
        );
      const vaeBenchmarkSchedulingOverride =
        opt0080ProductRun?.vaeSchedulingPolicyOverride;
      const vaePlan = planAceVaeChunkedDecode(shape.latentFrames, {
        chunkFrames: vaeWindowProfile.maximumWindowFrames,
        overlapFrames: vaeWindowProfile.overlapFrames,
      });
      tracker.progress(
        "vae-load",
        0,
        2,
        "items",
        "Acquiring OPT-0011 FP16 VAE package",
      );
      const vaeAcquired = await captureOpt0064Async(
        this.dependencies.now,
        context.opt0064Capture,
        "generation",
        "authentication",
        "vae-package-acquisition",
        Object.freeze({ cacheAuthenticationRequired: true }),
        async () => await this.dependencies.acquireModel({
          kind: "vae",
          loaded: ready.vaeLoaded,
          modelSource: ready.modelSource,
          storage: this.options.storage,
          signal: context.signal,
          onProgress: (event) => {
            tracker.progress(
              "vae-load",
              0,
              2,
              "items",
              `${event.source}: ${event.file} ` +
                `${event.completedBytes}/${event.totalBytes} bytes`,
            );
          },
          ...(context.opt0064Capture === undefined
            ? {}
            : {
                onTrace: opt0064AcquisitionTraceSink(
                  this.dependencies.now,
                  context.opt0064Capture,
                  "generation",
                  "vae",
                ),
                now: this.dependencies.now,
              }),
        }),
      );
      emitOpt0064AcquisitionPlan(
        this.dependencies.now,
        context.opt0064Capture,
        "generation",
        "vae",
        vaeAcquired,
      );
      requireVaeAcquisition(vaeAcquired);
      const transactionId = createAudioTransactionId(this.dependencies.randomUUID());
      transaction = await captureOpt0064Async(
        this.dependencies.now,
        context.opt0064Capture,
        "generation",
        "finalization",
        "audio-transaction-begin",
        Object.freeze({
          latentFrames: shape.latentFrames,
          outputFrames: shape.audioFramesPerChannel,
        }),
        async () => await this.dependencies.beginAudio(
          transactionId,
          vaePlan,
          this.options.storage,
        ),
      );
      const vaePhase = await this.loadPackagePhase(
        ready,
        ready.vaeLoaded.manifest,
        vaeAcquired.files,
        ["vae"],
        context.signal,
        (event) => mapPhaseUpload(event, tracker, "vae-load", 0, 1, 2),
        context.opt0064Capture,
        "vae",
      );
      const vaeWindowMetrics = new Map<number, Readonly<{
        drains: number;
        idleMs: number;
      }>>();
      const vaeFamilyProfiles: AceOpt0011Fp16VaeWindowFamilyProfile[] = [];
      const vaeSchedulingEvidence: AceOpt0080VaeSchedulingEvidence[] = [];
      const vaeSchedulingWindows = new Map<
        number,
        AceVaeWindowSchedulingReceipt
      >();
      const vaeApiCapture = createOpt0064GpuApiCapture(
        ready.deviceContext.device,
        this.dependencies.now,
        context.opt0064Capture,
        "vae",
      );
      vae = await captureOpt0064Async(
        this.dependencies.now,
        context.opt0064Capture,
        "generation",
        "construction",
        "vae-backend-pipeline-and-graph-construction",
        Object.freeze({
          includesPipelineCompilation: true,
          includesBindGroupConstruction: true,
          windowCount: vaePlan.windows.length,
        }),
        async () => await this.dependencies.createVae({
        device: vaeApiCapture.device,
        plan: vaePlan,
        finalLatents: denoised.finalLatent,
        authenticatedPackage: ready.vaeLoaded,
        ownedVaeWeights: vaePhase,
        maximumWindowFrames: vaeWindowProfile.maximumWindowFrames,
        // The portable OPT-0088 physical profile is resolved only for the
        // portable kernel backend and takes no subgroup-size members; every
        // fixed32 profile keeps the exact 32/32 requirement.
        ...(vaeBackendRuntimeProfileId ===
            ACE_OPT_0088_VAE_FP16_PORTABLE_DUAL_K4_PROFILE.id
          ? { runtimeProfileId: vaeBackendRuntimeProfileId }
          : {
              runtimeProfileId: vaeBackendRuntimeProfileId,
              subgroupMinSize: 32 as const,
              subgroupMaxSize: 32 as const,
            }),
        quantaPerCommandBuffer:
          ACE_OPT_0027_VAE_FP16_WINDOW_QUANTA_PER_COMMAND_BUFFER,
        ...(selectedProductionVaeSchedulingPolicy === undefined
          ? {}
          : {
              productionSchedulingPolicy:
                selectedProductionVaeSchedulingPolicy,
            }),
        signal: context.signal,
        ...(context.captureTrace === true
          ? {
              onFamilyProfile: (
                profile: AceOpt0011Fp16VaeWindowFamilyProfile,
              ) => {
                vaeFamilyProfiles.push(profile);
              },
            }
          : {}),
        onProgress: (event) => {
          mapVaeProgress(event, vaePlan, tracker);
          if (event.stage === "readback") {
            vaeWindowMetrics.set(event.windowIndex, {
              drains: event.queueDrains,
              idleMs: event.cooperativeIdleMs,
            });
            const window = vaePlan.windows[event.windowIndex];
            if (window === undefined) {
              throw new Error("VAE scheduling receipt lost its exact window");
            }
            const selection = resolveVaeSchedulingSelection(
              selectedProductionVaeSchedulingPolicy,
              vaeBenchmarkSchedulingOverride,
            );
            const schedulingProfile = resolvePipelineVaeWindowSchedulingProfile(
              selectedProductionVaeSchedulingPolicy,
              vaeBenchmarkSchedulingOverride,
              window,
            );
            vaeSchedulingWindows.set(
              event.windowIndex,
              createVaeWindowSchedulingReceipt(
                window,
                event,
                selection,
                schedulingProfile,
                ACE_OPT_0027_VAE_FP16_WINDOW_QUANTA_PER_COMMAND_BUFFER,
              ),
            );
          }
        },
        }),
      );
      vaeApiCapture.finish();
      peakTrackedGpuBytes = Math.max(
        peakTrackedGpuBytes,
        vae.memory.accountedGpuBytes,
      );
      tracker.complete("vae-load", 2, "items", "VAE loaded and compiled");
      const vaeDecodeBackend = vaeBenchmarkSchedulingOverride === undefined
        ? vae
        : Object.freeze({
            decodeWindow: async (window: AceVaeDecodeWindow) =>
              await vae!.decodeWindow(window, undefined, {
                schedulingProfile:
                  resolvePipelineVaeWindowSchedulingProfile(
                    undefined,
                    vaeBenchmarkSchedulingOverride,
                    window,
                  ),
                onSchedulingEvidence: (evidence) => {
                  vaeSchedulingEvidence.push(evidence);
                },
              }),
          });
      const rawStats = await captureOpt0064Async(
        this.dependencies.now,
        context.opt0064Capture,
        "generation",
        "execution",
        "vae-decode-and-raw-stream",
        Object.freeze({
          windowCount: vaePlan.windows.length,
          rawBytes: vaePlan.outputFloat32Bytes,
        }),
        async () => await streamAceVaeRawChunks(
          vaePlan,
          vaeDecodeBackend,
          transaction!.rawSink,
          {
            signal: context.signal,
            ...(context.opt0064Capture === undefined
              ? {}
              : {
                  now: this.dependencies.now,
                  onTrace: (trace: AceVaeRawStreamTrace) => {
                    emitOpt0064Event(context.opt0064Capture, {
                      scope: "generation",
                      category: "finalization",
                      operation: "vae-decode-raw-scan-and-opfs-write",
                      startedAtMs: trace.startedAtMs,
                      completedAtMs: trace.completedAtMs,
                      details: Object.freeze({ ...trace }),
                    });
                  },
                }),
          },
        ),
      );
      await captureOpt0064Async(
        this.dependencies.now,
        context.opt0064Capture,
        "generation",
        "cleanup",
        "vae-drain-and-destroy",
        Object.freeze({ drainedBeforeRelease: true }),
        async () => await vae!.destroy(),
      );
      vae = undefined;
      addVaeMetrics(vaeWindowMetrics, rawStats, (drains, idleMs) => {
        queueDrains += drains;
        cooperativeIdleMs += idleMs;
      });
      this.reportTrace(
        context.captureTrace === true &&
          vaeFamilyProfiles.length === vaePlan.windows.length,
        "ACE_VAE_FP16_FAMILY_PROFILE",
        "Attributed homogeneous FP16 VAE decoder batches by kernel family",
        "vae-decode",
        () => summarizeVaeFamilyProfiles(
          vaeFamilyProfiles,
          vaeRuntimeIdentity,
        ),
      );
      completeIfCurrent(
        tracker,
        "vae-decode",
        vaePlan.windows.length,
        "vae-chunks",
        "Raw VAE waveform decoded",
      );

      context.signal.throwIfAborted();
      this.throwIfGpuFailed();
      committed = await captureOpt0064Async(
        this.dependencies.now,
        context.opt0064Capture,
        "generation",
        "finalization",
        "normalize-encode-and-durable-wav-commit",
        Object.freeze({
          rawPeak: rawStats.peak,
          boundedSecondPass: true,
        }),
        async () => await transaction!.commit(
          deriveAceVaePostprocessPlan(rawStats.peak),
          {
            signal: context.signal,
            ...(samplerScheduleRun === undefined &&
                opt0080ProductRun === undefined
              ? {}
              : { retainRawSnapshot: true as const }),
            ...(context.opt0064Capture === undefined
              ? {}
              : {
                  now: this.dependencies.now,
                  onTrace: (trace: AceVaeWavWriteTrace) => {
                    emitOpt0064Event(context.opt0064Capture, {
                      scope: "generation",
                      category: "finalization",
                      operation: "bounded-wav-normalize-encode-write",
                      startedAtMs: trace.startedAtMs,
                      completedAtMs: trace.completedAtMs,
                      details: Object.freeze({ ...trace }),
                    });
                  },
                  onTransactionTrace: (
                    trace: AceAudioOutputTransactionTrace,
                  ) => {
                    emitOpt0064Event(context.opt0064Capture, {
                      scope: "generation",
                      category: "finalization",
                      operation: `audio-${trace.operation}`,
                      startedAtMs: trace.startedAtMs,
                      completedAtMs: trace.completedAtMs,
                      details: Object.freeze({ ...trace }),
                    });
                  },
                }),
          },
        ),
      );
      transaction = undefined;
      const committedFrames = committed.wav.dataBytes /
        (ACE_CHANNEL_COUNT * Float32Array.BYTES_PER_ELEMENT);
      if (
        !Number.isSafeInteger(committedFrames) ||
        committedFrames !== shape.audioFramesPerChannel
      ) {
        throw new Error(
          `ACE committed WAV has ${committedFrames} frames; ` +
            `expected ${shape.audioFramesPerChannel}`,
        );
      }
      this.committedAudioIds.add(committed.transactionId);
      if (samplerScheduleRun !== undefined) {
        const evidence = createSamplerScheduleDiagnosticEvidence(
          samplerScheduleProfile,
          denoised,
          rawStats,
          committed,
        );
        samplerScheduleRun.onEvidence(evidence);
        context.signal.throwIfAborted();
      }
      tracker.complete(
        "wav-encode",
        shape.audioFramesPerChannel,
        "audio-frames",
        "Normalized float WAV committed to OPFS",
      );
      context.signal.throwIfAborted();
      this.throwIfGpuFailed();
      tracker.complete("cleanup", 1, "items", "Generation resources released");
      tracker.complete("done", 1, "items", "Generation complete");

      const vaeScheduling = createVaeSchedulingReceipt(
        vaePlan,
        selectedProductionVaeSchedulingPolicy,
        vaeBenchmarkSchedulingOverride,
        vaeSchedulingWindows,
      );
      const result: AceGenerationResult = Object.freeze({
        audio: committed.audio,
        audioStorageId: committed.transactionId,
        mimeType: "audio/wav",
        sampleRateHz: ACE_SAMPLE_RATE_HZ,
        channelCount: ACE_CHANNEL_COUNT,
        frameCount: committedFrames,
        durationSeconds: request.durationSeconds,
        seed: request.seed,
        generationProfile: request.generationProfile,
        modelManifestId: ready.loaded.manifestId,
        modelManifestSha256: ready.loaded.manifestSha256,
        diagnostics: ready.diagnostics,
        metrics: Object.freeze({
          totalMs: nonnegativeElapsed(this.dependencies.now(), this.diagnosticStart),
          stageTimings: tracker.timings,
          peakTrackedGpuBytes,
          cooperativeGpuQueueDrains: queueDrains,
          cooperativeIdleMs,
          vaeScheduling,
        }),
      });
      if (opt0080ProductRun !== undefined) {
        const rawSnapshot = committed.rawSnapshot;
        if (
          rawSnapshot === undefined ||
          rawSnapshot.size !== vaePlan.outputFloat32Bytes ||
          (vaeBenchmarkSchedulingOverride !== undefined &&
            vaeSchedulingEvidence.length !== vaePlan.windows.length) ||
          denoised.finalLatent.byteOffset !== 0 ||
          denoised.finalLatent.byteLength !==
            denoised.finalLatent.buffer.byteLength
        ) {
          throw new Error(
            "OPT-0080 product integration evidence is incomplete",
          );
        }
        opt0080ProductRun.onEvidence(Object.freeze({
          schema: "ace-opt-0080-product-integration-evidence-v1",
          selectedProductionPolicy: ACE_PRODUCTION_DIT_SUBMISSION_POLICY,
          effectiveSubmissionPolicy: ditSubmissionPolicy!,
          finalLatent: denoised.finalLatent,
          finalLatentRawU32: new Uint32Array(denoised.finalLatent.buffer),
          rawStats,
          rawSnapshot,
          vaePlan,
          ...(vaeBenchmarkSchedulingOverride === undefined
            ? {}
            : {
                vaeSchedulingEvidence: Object.freeze([
                  ...vaeSchedulingEvidence,
                ]),
              }),
          result,
        }));
        context.signal.throwIfAborted();
      }
      return result;
    } catch (error) {
      ditReferencePhase?.destroy();
      ditDensePhase?.destroy();
      await settleWithoutMasking([
        conditioning?.destroy(error),
        dit?.destroy(error),
        vae?.destroy(error),
      ]);
      if (transaction !== undefined) {
        if (!(await tryRollback(transaction))) {
          this.pendingRollbacks.add(transaction);
        }
      }
      if (committed !== undefined) {
        if (await this.tryReleaseAudio(committed.transactionId)) {
          this.committedAudioIds.delete(committed.transactionId);
        } else {
          this.pendingUnpublishedAudioIds.add(committed.transactionId);
        }
      }
      throw error;
    } finally {
      this.operation = "idle";
      this.diagnosticSink = undefined;
    }
  }

  async releaseResult(result: AceGenerationResult): Promise<void> {
    const transactionId = result?.audioStorageId;
    if (
      typeof transactionId !== "string" ||
      !SAFE_AUDIO_STORAGE_ID.test(transactionId) ||
      !this.committedAudioIds.has(transactionId)
    ) {
      throw new DOMException(
        "ACE pipeline cannot release an output it does not own",
        "NotFoundError",
      );
    }
    try {
      await this.dependencies.releaseAudio(transactionId, this.options.storage);
      this.committedAudioIds.delete(transactionId);
      this.pendingUnpublishedAudioIds.delete(transactionId);
    } catch (error) {
      this.pendingUnpublishedAudioIds.add(transactionId);
      throw error;
    }
  }

  async dispose(): Promise<void> {
    if (this.operation !== "idle") {
      throw new DOMException("ACE pipeline disposal requires operation cancellation", "InvalidStateError");
    }
    this.operation = "disposing";
    const ready = this.ready;
    this.ready = undefined;
    this.deviceLoss = undefined;
    this.gpuError = undefined;
    this.gpuEventToken = undefined;
    try {
      await this.retryPendingCleanup();
    } finally {
      ready?.deviceContext.destroy();
      this.diagnosticSink = undefined;
      this.operation = "idle";
    }
  }

  private async retryPendingCleanup(): Promise<void> {
    let cleanupFailure: unknown;
    for (const transaction of [...this.pendingRollbacks]) {
      try {
        await transaction.rollback();
        this.pendingRollbacks.delete(transaction);
      } catch (error) {
        cleanupFailure ??= error;
      }
    }
    for (const transactionId of [...this.pendingUnpublishedAudioIds]) {
      try {
        await this.dependencies.releaseAudio(transactionId, this.options.storage);
        this.pendingUnpublishedAudioIds.delete(transactionId);
        this.committedAudioIds.delete(transactionId);
      } catch (error) {
        cleanupFailure ??= error;
      }
    }
    if (cleanupFailure !== undefined) throw cleanupFailure;
  }

  private async tryReleaseAudio(transactionId: string): Promise<boolean> {
    try {
      await this.dependencies.releaseAudio(transactionId, this.options.storage);
      return true;
    } catch {
      return false;
    }
  }

  private async runPlanner(
    request: AceGenerationRequest,
    ready: ReadyResources,
    context: AceWebGpuGenerationContext,
    tracker: GenerationProgressTracker,
  ): Promise<AcePlannerCoordinatorResult> {
    if (request.planner.mode === "disabled") {
      return await this.dependencies.runPlanner({ request, signal: context.signal });
    }
    const semanticSteps = request.durationSeconds * 5 + 1;
    const totalPlannerSteps = 3;
    tracker.progress(
      "semantic-planner",
      0,
      totalPlannerSteps,
      "planner-steps",
      "Loading planner weights",
    );
    const phase = await this.loadPhase(
      ready,
      ["planner"],
      context.signal,
      (event) => mapPhaseUpload(
        event,
        tracker,
        "semantic-planner",
        0,
        1,
        totalPlannerSteps,
        "planner-steps",
      ),
      context.opt0064Capture,
    );
    const result = await this.dependencies.runPlanner({
      request,
      resources: {
        kind: "owned-phase",
        device: ready.deviceContext.device,
        modelProfile: ready.configuration.modelProfile,
        tokenizer: ready.plannerTokenizer,
        ownedPlannerWeights: phase,
      },
      signal: context.signal,
      onProgress: (event) => mapPlannerProgress(
        event,
        semanticSteps,
        totalPlannerSteps,
        tracker,
      ),
    });
    tracker.complete(
      "semantic-planner",
      totalPlannerSteps,
      "planner-steps",
      "Planner metadata and semantic codes complete",
    );
    return result;
  }

  private async loadPhase(
    ready: ReadyResources,
    phases: readonly AceTensorPhase[],
    signal: AbortSignal,
    onProgress?: (progress: AceGpuTensorPhaseProgress) => void,
    capture?: AceOpt0064CaptureSink,
  ): Promise<AceGpuTensorPhase> {
    return await this.loadPackagePhase(
      ready,
      ready.loaded.manifest,
      ready.acquired.files,
      phases,
      signal,
      onProgress,
      capture,
      "main",
    );
  }

  private async loadPackagePhase(
    ready: ReadyResources,
    manifest: AcePackageManifest,
    files: ReadonlyMap<string, File>,
    phases: readonly AceTensorPhase[],
    signal: AbortSignal,
    onProgress?: (progress: AceGpuTensorPhaseProgress) => void,
    capture?: AceOpt0064CaptureSink,
    packageKind = "main",
  ): Promise<AceGpuTensorPhase> {
    signal.throwIfAborted();
    this.throwIfGpuFailed();
    const phase = await captureOpt0064Async(
      this.dependencies.now,
      capture,
      "generation",
      "upload",
      `${packageKind}-${phases.join("+")}-phase-upload`,
      Object.freeze({ packageKind, phases: Object.freeze([...phases]) }),
      async () => await this.dependencies.loadPhase(
        ready.deviceContext.device,
        manifest,
        files,
        phases,
        signal,
        onProgress,
        capture === undefined
          ? undefined
          : opt0064UploadTraceSink(capture, packageKind),
        capture === undefined ? undefined : this.dependencies.now,
      ),
    );
    if (phase.packageManifest !== manifest) {
      phase.destroy();
      throw new Error(
        `ACE ${phases.join("+")} phase was authenticated by a different manifest object`,
      );
    }
    return phase;
  }

  private requireReady(): ReadyResources {
    if (this.ready === undefined) {
      throw new DOMException("ACE pipeline is not initialized", "InvalidStateError");
    }
    this.throwIfGpuFailed();
    return this.ready;
  }

  private throwIfDeviceLost(): void {
    if (this.deviceLoss !== undefined) throw this.deviceLoss;
  }

  private throwIfGpuFailed(): void {
    this.throwIfDeviceLost();
    if (this.gpuError !== undefined) throw this.gpuError;
  }

  private reportGpuEvent(event: AceGpuRuntimeEvent): void {
    if (event.type === "uncaptured-error" && this.gpuError === undefined) {
      this.gpuError = new AceFatalGpuError(
        "WEBGPU_UNCAPTURED_ERROR",
        `ACE WebGPU ${event.errorType}: ${event.message || "uncaptured GPU error"}`,
      );
    } else if (event.type === "device-lost" && this.deviceLoss === undefined) {
      this.deviceLoss = new AceFatalGpuError(
        "WEBGPU_DEVICE_LOST",
        `ACE WebGPU device lost (${event.reason}): ${event.message || "no device message"}`,
      );
    }
    const elapsedMs = nonnegativeElapsed(
      this.dependencies.now(),
      this.diagnosticStart,
    );
    try {
      this.diagnosticSink?.(Object.freeze({
        severity: "error",
        code: event.type === "device-lost"
          ? "WEBGPU_DEVICE_LOST"
          : "WEBGPU_UNCAPTURED_ERROR",
        message: event.message || event.type,
        elapsedMs,
        details: event.type === "device-lost"
          ? Object.freeze({ reason: event.reason })
          : Object.freeze({ errorType: event.errorType }),
      }));
    } catch {
      // The fatal GPU state above is authoritative. Diagnostics are
      // observational and cannot be allowed to erase it.
    }
  }

  private reportTrace(
    enabled: boolean,
    code: string,
    message: string,
    stage: AceGenerationStage,
    details: () => NonNullable<AceDiagnostic["details"]>,
  ): void {
    if (!enabled) return;
    try {
      this.diagnosticSink?.(Object.freeze({
        severity: "info",
        code,
        message,
        elapsedMs: nonnegativeElapsed(
          this.dependencies.now(),
          this.diagnosticStart,
        ),
        stage,
        details: details(),
      }));
    } catch {
      // Trace receipts are observational and never affect model execution.
    }
  }
}

export function createAceWebGpuPipelineBackend(
  options: AceWebGpuPipelineOptions = {},
): AceWebGpuPipelineBackend {
  const fetchAsset = options.fetch ?? globalThis.fetch.bind(globalThis);
  const random = options.crypto ?? globalThis.crypto;
  if (random?.randomUUID === undefined) {
    throw new Error("ACE audio transactions require crypto.randomUUID()");
  }
  const dependencies: AceWebGpuPipelineDependencies = {
    now: () => performance.now(),
    randomUUID: () => random.randomUUID(),
    ensureStorage: async (storage, signal) => {
      signal.throwIfAborted();
      const manager = storage ?? navigator.storage;
      await manager.getDirectory();
      signal.throwIfAborted();
    },
    recoverAudio: async (storage, signal) => {
      await recoverStaleAceAudioOutputs(storage ?? navigator.storage, {
        signal,
      });
    },
    loadManifest: async (configuration, signal) =>
      await loadAcePackageManifest({
        manifestUrl: configuration.manifestUrl,
        expectedManifestSha256: configuration.manifestSha256,
        expectedProfile: packageProfile(configuration.modelProfile),
        signal,
        fetch: fetchAsset,
      }),
    loadVaeManifest: async (configuration, signal) => {
      const identity = resolveAceVaePackageRuntimeIdentity(configuration);
      return await loadAcePackageManifest({
        manifestUrl: configuration.manifestUrl,
        expectedManifestSha256: configuration.manifestSha256,
        expectedProfile: "fp16-vae-experimental",
        ...(identity.role === "opt-0054-rev7-candidate" ||
            identity.role === "opt-0072-rev7-production"
          ? { authenticatedVaeConverterRevision: 7 as const }
          : {}),
        signal,
        fetch: fetchAsset,
      });
    },
    loadDitDenseManifest: async (configuration, signal) => {
      const identity = resolveAceDitDensePackageRuntimeIdentity(configuration);
      return await loadAcePackageManifest({
        manifestUrl: configuration.manifestUrl,
        expectedManifestSha256: configuration.manifestSha256,
        expectedProfile: "fp16-dit-dense-experimental",
        ...(identity.role === "opt-0009-rev7-oracle"
          ? { authenticatedDitDenseConverterRevision: 7 as const }
          : {}),
        signal,
        fetch: fetchAsset,
      });
    },
    acquireModel: async ({
      kind,
      loaded,
      modelSource,
      storage,
      signal,
      onProgress,
      onTrace,
      now,
    }) => {
      const storageManager = storage ?? navigator.storage;
      const cache = await AceOpfsModelCache.open(storageManager);
      const manifest = kind === "main"
        ? createAceMainAcquisitionManifest(loaded.manifest)
        : kind === "dit-dense"
          ? loaded.manifest.provenance.converterRevision ===
              ACE_OPT_0009_DIT_DENSE_CONVERTER_REVISION
            ? createAceOpt0009DitDenseAcquisitionManifest(loaded.manifest)
            : createAceOpt0037DitK4AcquisitionManifest(loaded.manifest)
          : createAceOpt0011VaeAcquisitionManifest(loaded.manifest);
      return await acquireAceModelFiles({
        manifest,
        manifestUrl: loaded.manifestUrl,
        cache,
        signal,
        storage: storageManager,
        fetch: modelSource === "cache-only" ? cacheOnlyFetch : fetchAsset,
        maximumAttempts: modelSource === "cache-only" ? 1 : 3,
        ...(options.cacheAuthenticationOwner === undefined
          ? {}
          : { cacheAuthenticationOwner: options.cacheAuthenticationOwner }),
        onFileProgress: onProgress,
        ...(onTrace === undefined ? {} : { onTrace }),
        ...(now === undefined ? {} : { now }),
      });
    },
    requestDevice: async (request) => await requestAceWebGpuDevice(request),
    loadTokenizer: loadPinnedAceTokenizer,
    loadPhase: async (
      device,
      manifest,
      files,
      phases,
      signal,
      onProgress,
      onUploadTrace,
      now,
    ) =>
      await AceGpuTensorPhase.load(device, manifest, files, phases, {
        signal,
        ...(onProgress === undefined ? {} : { onProgress }),
        ...(onUploadTrace === undefined ? {} : { onUploadTrace }),
        ...(now === undefined ? {} : { now }),
      }),
    runPlanner: async (plannerOptions) =>
      await runAcePlannerCoordinator(plannerOptions),
    createConditioning: (conditioningOptions) =>
      AceConditioningGpuExecutor.create(conditioningOptions),
    createDit: async (ditOptions) => await AceDitGpuBackend.create(ditOptions),
    createOpt0081Representative: async (ditOptions) =>
      await AceOpt0081RepresentativeDitOwner.create(ditOptions),
    createVae: async (vaeOptions) =>
      await AceOpt0011Fp16VaeChunkGpuBackend.create(vaeOptions),
    beginAudio: async (transactionId, plan, storage) =>
      await AceAudioOutputTransaction.begin(
        transactionId,
        plan,
        storage ?? navigator.storage,
      ),
    releaseAudio: async (transactionId, storage) =>
      await releaseAceAudioOutput(transactionId, storage ?? navigator.storage),
  };
  return new AceWebGpuPipelineBackend(options, dependencies);
}

/** @internal Construct the real coordinator with deterministic dependencies. */
export function createAceWebGpuPipelineBackendForTest(
  dependencies: AceWebGpuPipelineDependencies,
  options: AceWebGpuPipelineOptions = {},
): AceWebGpuPipelineBackend {
  return new AceWebGpuPipelineBackend(options, dependencies);
}

function packageProfile(modelProfile: AceModelProfileId): AcePackageProfile {
  return modelProfile === "reference-bf16" ? "reference" : "fp16";
}

/** @internal Build the runtime inventory while retaining the parsed identity. */
export function createAceMainAcquisitionManifest(
  manifest: AcePackageManifest,
): AcePackageManifest {
  if (manifest.profile !== "reference") {
    throw new Error("ACE production main package must be the reference profile");
  }
  const phasesByShard = new Map<string, Set<AceTensorPhase>>();
  for (const tensor of Object.values(manifest.tensors)) {
    const phases = phasesByShard.get(tensor.shard) ?? new Set<AceTensorPhase>();
    phases.add(tensor.phase);
    phasesByShard.set(tensor.shard, phases);
  }
  const vaeOnlyShards = new Set(
    [...phasesByShard]
      .filter(([, phases]) => phases.size === 1 && phases.has("vae"))
      .map(([shard]) => shard),
  );
  const referenceDitLayerShards = new Set(
    manifest.files
      .filter((file) => isAceReferenceDitLayerWeightFile(file.name))
      .map((file) => file.name),
  );
  const files = Object.freeze(
    manifest.files.filter((file) =>
      !vaeOnlyShards.has(file.name) &&
      !referenceDitLayerShards.has(file.name)
    ),
  );
  const referenceDitLayerBytes = manifest.files
    .filter((file) => referenceDitLayerShards.has(file.name))
    .reduce((sum, file) => sum + file.byteLength, 0);
  if (
    vaeOnlyShards.size === 0 ||
    referenceDitLayerShards.size !== 48 ||
    referenceDitLayerBytes !== ACE_OPT_0037_DIT_K4_LAYER_BYTES ||
    Object.values(manifest.tensors).some((tensor) =>
      tensor.phase !== "vae" && vaeOnlyShards.has(tensor.shard)
    ) ||
    Object.entries(manifest.tensors).some(([name, tensor]) =>
      referenceDitLayerShards.has(tensor.shard) !==
        name.startsWith("ace.decoder.layers.")
    )
  ) {
    throw new Error("ACE mixed main-package physical shard filtering changed");
  }
  return Object.freeze({ ...manifest, files });
}

/** @internal Re-resolve the 48 physical OPT-0009 mixed layer shards. */
export function createAceOpt0009DitDenseAcquisitionManifest(
  manifest: AcePackageManifest,
): AcePackageManifest {
  if (
    manifest.profile !== "fp16-dit-dense-experimental" ||
    manifest.provenance.converterRevision !==
      ACE_OPT_0009_DIT_DENSE_CONVERTER_REVISION
  ) {
    throw new Error("OPT-0009 acquisition requires the revision-7 package");
  }
  const byName = new Map(manifest.files.map((file) => [file.name, file]));
  const files = Object.freeze(ACE_OPT_0009_DIT_DENSE_WEIGHT_FILES.map((name) => {
    const file = byName.get(name);
    if (file === undefined || file.kind !== "weights") {
      throw new Error(`OPT-0009 authenticated manifest is missing ${name}`);
    }
    return file;
  }));
  if (
    files.length !== 48 ||
    files.reduce((sum, file) => sum + file.byteLength, 0) !==
      ACE_OPT_0009_DIT_MIXED_LAYER_BYTES
  ) {
    throw new Error("OPT-0009 acquisition physical layer inventory changed");
  }
  return Object.freeze({ ...manifest, files });
}

/** @internal Re-resolve the exact 48 physical rev8 K4 mixed-layer shards. */
export function createAceOpt0037DitK4AcquisitionManifest(
  manifest: AcePackageManifest,
): AcePackageManifest {
  if (
    manifest.profile !== "fp16-dit-dense-experimental" ||
    manifest.provenance.converterRevision !== 8
  ) {
    throw new Error("OPT-0037 acquisition requires the revision-8 K4 package");
  }
  const byName = new Map(manifest.files.map((file) => [file.name, file]));
  const files = Object.freeze(ACE_OPT_0037_DIT_K4_WEIGHT_FILES.map((name) => {
    const file = byName.get(name);
    if (file === undefined || file.kind !== "weights") {
      throw new Error(`OPT-0037 authenticated manifest is missing ${name}`);
    }
    return file;
  }));
  if (
    files.length !== 48 ||
    files.reduce((sum, file) => sum + file.byteLength, 0) !==
      ACE_OPT_0037_DIT_K4_LAYER_BYTES
  ) {
    throw new Error("OPT-0037 acquisition physical layer inventory changed");
  }
  return Object.freeze({ ...manifest, files });
}

/** @internal Re-resolve the exact seven rev6/rev7 VAE shards from authenticated data. */
export function createAceOpt0011VaeAcquisitionManifest(
  manifest: AcePackageManifest,
): AcePackageManifest {
  if (
    manifest.profile !== "fp16-vae-experimental" ||
    (manifest.provenance.converterRevision !==
        ACE_EXPERIMENTAL_VAE_PACKAGE_CONVERTER_REVISION &&
      manifest.provenance.converterRevision !==
        ACE_OPT_0054_EXPERIMENTAL_VAE_PACKAGE_CONVERTER_REVISION)
  ) {
    throw new Error(
      "FP16 VAE acquisition requires an authenticated revision-6 or revision-7 package",
    );
  }
  const byName = new Map(manifest.files.map((file) => [file.name, file]));
  const files = Object.freeze(ACE_OPT_0011_VAE_FP16_WEIGHT_FILES.map((name) => {
    const file = byName.get(name);
    if (file === undefined || file.kind !== "weights") {
      throw new Error(`OPT-0011 authenticated manifest is missing ${name}`);
    }
    return file;
  }));
  const bytes = files.reduce((sum, file) => sum + file.byteLength, 0);
  const vaeShards = new Set(
    Object.values(manifest.tensors)
      .filter((tensor) => tensor.phase === "vae")
      .map((tensor) => tensor.shard),
  );
  if (
    bytes !== ACE_EXPERIMENTAL_VAE_PARAMETER_BYTES ||
    files.length !== 7 ||
    vaeShards.size !== 7 ||
    files.some((file) => !vaeShards.has(file.name))
  ) {
    throw new Error(
      "OPT-0011 acquisition is not exactly seven manifest-owned VAE shards",
    );
  }
  return Object.freeze({ ...manifest, files });
}

async function cacheOnlyFetch(): Promise<Response> {
  throw new DOMException(
    "ACE cache-only initialization found a missing runtime asset",
    "NotFoundError",
  );
}

function tokenizerAssets(
  files: ReadonlyMap<string, File>,
  directory: "qwen" | "planner",
): AceTokenizerAssetBundle {
  return Object.freeze({
    tokenizerJson: requirePackageFile(files, `assets/${directory}/tokenizer.json`),
    tokenizerConfigJson: requirePackageFile(
      files,
      `assets/${directory}/tokenizer_config.json`,
    ),
    chatTemplate: requirePackageFile(
      files,
      `assets/${directory}/chat_template.jinja`,
    ),
  });
}

function requirePackageFile(
  files: ReadonlyMap<string, File>,
  name: string,
): File {
  const file = files.get(name);
  if (file === undefined) throw new Error(`ACE package is missing ${name}`);
  return file;
}

function requireTokenizer(
  loaded: LoadedAceTokenizer,
  kind: "text" | "planner",
): AceQwenBpeTokenizer {
  if (loaded.tokenizer.kind !== kind) {
    throw new Error(`ACE ${kind} tokenizer loader returned ${loaded.tokenizer.kind}`);
  }
  return loaded.tokenizer;
}

function tokenizeConditioning(
  tokenizer: AceQwenBpeTokenizer,
  text: ReturnType<typeof resolveAceConditioningText>,
): Pick<
  AceConditioningGpuRequest,
  "textTokenIds" | "lyricTokenIds" | "textMask" | "lyricMask"
> {
  const caption = formatAceTextEncoderCaptionInput(
    text.instruction,
    text.caption,
    text.formattedMetadata,
  );
  const lyrics = formatAceTextEncoderLyricsInput(
    text.lyrics,
    text.vocalLanguage,
  );
  const textBatch = tokenizer.encodeBatch([caption], {
    addSpecialTokens: true,
    truncation: true,
    maxLength: ACE_TEXT_ENCODER_MAX_TEXT_TOKENS,
    padding: "longest",
  });
  const lyricBatch = tokenizer.encodeBatch([lyrics], {
    addSpecialTokens: true,
    truncation: true,
    maxLength: ACE_TEXT_ENCODER_MAX_LYRIC_TOKENS,
    padding: "longest",
  });
  return Object.freeze({
    textTokenIds: Uint32Array.from(textBatch.inputIds[0]!),
    lyricTokenIds: Uint32Array.from(lyricBatch.inputIds[0]!),
    textMask: Uint32Array.from(textBatch.attentionMask[0]!),
    lyricMask: Uint32Array.from(lyricBatch.attentionMask[0]!),
  });
}

export type AceDitDensePackageRuntimeIdentity =
  | Readonly<{
      readonly role: "opt-0009-rev7-oracle";
      readonly manifestSha256: typeof ACE_OPT_0009_DIT_DENSE_MANIFEST_SHA256;
      readonly manifestByteLength: typeof ACE_OPT_0009_DIT_DENSE_MANIFEST_BYTES;
      readonly runtimeProfile: typeof ACE_OPT_0009_DIT_DENSE_RUNTIME_PROFILE;
      readonly kernelSetId:
        | typeof ACE_OPT_0009_DIT_DENSE_KERNEL_SET_ID
        | typeof ACE_OPT_0088_DIT_DENSE_PORTABLE_KERNEL_SET_ID;
      readonly layerBytes: typeof ACE_OPT_0009_DIT_MIXED_LAYER_BYTES;
      readonly residentWeightBytes:
        typeof ACE_OPT_0009_DIT_MIXED_RESIDENT_WEIGHT_BYTES;
    }>
  | Readonly<{
      readonly role: "opt-0037-rev8-production";
      readonly manifestSha256: typeof ACE_OPT_0037_DIT_K4_MANIFEST_SHA256;
      readonly manifestByteLength: typeof ACE_OPT_0037_DIT_K4_MANIFEST_BYTES;
      readonly runtimeProfile: typeof ACE_OPT_0037_DIT_K4_RUNTIME_PROFILE;
      readonly kernelSetId: typeof ACE_OPT_0037_DIT_K4_KERNEL_SET_ID;
      readonly layerBytes: typeof ACE_OPT_0037_DIT_K4_LAYER_BYTES;
      readonly residentWeightBytes:
        typeof ACE_OPT_0037_DIT_K4_RESIDENT_WEIGHT_BYTES;
    }>
  | Readonly<{
      readonly role: "opt-0056-rev8-benchmark";
      readonly manifestSha256: typeof ACE_OPT_0037_DIT_K4_MANIFEST_SHA256;
      readonly manifestByteLength: typeof ACE_OPT_0037_DIT_K4_MANIFEST_BYTES;
      readonly runtimeProfile:
        typeof ACE_OPT_0056_DIT_SELECTIVE_K4_RUNTIME_PROFILE;
      readonly kernelSetId:
        typeof ACE_OPT_0056_DIT_SELECTIVE_K4_KERNEL_SET_ID;
      readonly layerBytes: typeof ACE_OPT_0037_DIT_K4_LAYER_BYTES;
      readonly residentWeightBytes:
        typeof ACE_OPT_0037_DIT_K4_RESIDENT_WEIGHT_BYTES;
    }>;

/** @internal Resolve only authenticated package/profile pairs. */
export function resolveAceDitDensePackageRuntimeIdentity(
  configuration: AceWorkerDitDensePackageConfiguration,
  kernelBackend: AceKernelBackend = "subgroups",
): AceDitDensePackageRuntimeIdentity {
  if (
    configuration.manifestSha256 ===
      ACE_OPT_0009_DIT_DENSE_MANIFEST_SHA256 &&
    configuration.runtimeProfile === ACE_OPT_0009_DIT_DENSE_RUNTIME_PROFILE
  ) {
    return Object.freeze({
      role: "opt-0009-rev7-oracle",
      manifestSha256: ACE_OPT_0009_DIT_DENSE_MANIFEST_SHA256,
      manifestByteLength: ACE_OPT_0009_DIT_DENSE_MANIFEST_BYTES,
      runtimeProfile: ACE_OPT_0009_DIT_DENSE_RUNTIME_PROFILE,
      // OPT-0088: same package and layout; only the kernel transport
      // differs on portable devices.
      kernelSetId: kernelBackend === "portable"
        ? ACE_OPT_0088_DIT_DENSE_PORTABLE_KERNEL_SET_ID
        : ACE_OPT_0009_DIT_DENSE_KERNEL_SET_ID,
      layerBytes: ACE_OPT_0009_DIT_MIXED_LAYER_BYTES,
      residentWeightBytes: ACE_OPT_0009_DIT_MIXED_RESIDENT_WEIGHT_BYTES,
    });
  }
  if (kernelBackend === "portable") {
    // Only the OPT-0009 oracle package has a portable kernel counterpart.
    throw new Error(
      "ACE portable mixed DiT accepts only the OPT-0009 rev7 oracle package",
    );
  }
  if (
    configuration.manifestSha256 === ACE_OPT_0037_DIT_K4_MANIFEST_SHA256 &&
    configuration.runtimeProfile ===
      ACE_OPT_0056_DIT_SELECTIVE_K4_RUNTIME_PROFILE
  ) {
    return Object.freeze({
      role: "opt-0056-rev8-benchmark",
      manifestSha256: ACE_OPT_0037_DIT_K4_MANIFEST_SHA256,
      manifestByteLength: ACE_OPT_0037_DIT_K4_MANIFEST_BYTES,
      runtimeProfile: ACE_OPT_0056_DIT_SELECTIVE_K4_RUNTIME_PROFILE,
      kernelSetId: ACE_OPT_0056_DIT_SELECTIVE_K4_KERNEL_SET_ID,
      layerBytes: ACE_OPT_0037_DIT_K4_LAYER_BYTES,
      residentWeightBytes: ACE_OPT_0037_DIT_K4_RESIDENT_WEIGHT_BYTES,
    });
  }
  if (
    configuration.manifestSha256 === ACE_OPT_0037_DIT_K4_MANIFEST_SHA256 &&
    configuration.runtimeProfile === ACE_OPT_0037_DIT_K4_RUNTIME_PROFILE
  ) {
    return Object.freeze({
      role: "opt-0037-rev8-production",
      manifestSha256: ACE_OPT_0037_DIT_K4_MANIFEST_SHA256,
      manifestByteLength: ACE_OPT_0037_DIT_K4_MANIFEST_BYTES,
      runtimeProfile: ACE_OPT_0037_DIT_K4_RUNTIME_PROFILE,
      kernelSetId: ACE_OPT_0037_DIT_K4_KERNEL_SET_ID,
      layerBytes: ACE_OPT_0037_DIT_K4_LAYER_BYTES,
      residentWeightBytes: ACE_OPT_0037_DIT_K4_RESIDENT_WEIGHT_BYTES,
    });
  }
  throw new Error(
    "ACE mixed DiT requires one exact authenticated manifest/profile pair",
  );
}

export type AceVaePackageRuntimeIdentity =
  | Readonly<{
      readonly role: "opt-0028-rev6-production";
      readonly manifestSha256: typeof ACE_OPT_0028_VAE_FP16_MANIFEST_SHA256;
      readonly manifestByteLength: typeof ACE_OPT_0028_VAE_FP16_MANIFEST_BYTES;
      readonly runtimeProfile:
        typeof ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_PROFILE.id;
      readonly kernelSetId:
        typeof ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_PROFILE.kernelSetId;
      readonly precisionMapSha256: string;
    }>
  | Readonly<{
      readonly role: "opt-0040-rev6-scalar-fp32-oracle";
      readonly manifestSha256: typeof ACE_OPT_0028_VAE_FP16_MANIFEST_SHA256;
      readonly manifestByteLength: typeof ACE_OPT_0028_VAE_FP16_MANIFEST_BYTES;
      readonly runtimeProfile:
        typeof ACE_OPT_0040_VAE_FP16_FIXED32_SHAPE_SELECTED_PROFILE.id;
      readonly kernelSetId:
        typeof ACE_OPT_0040_VAE_FP16_FIXED32_SHAPE_SELECTED_PROFILE.kernelSetId;
      readonly precisionMapSha256: string;
    }>
  | Readonly<{
      readonly role: "opt-0054-rev7-candidate";
      readonly manifestSha256:
        typeof ACE_OPT_0054_VAE_REVISION7_MANIFEST_SHA256;
      readonly manifestByteLength:
        typeof ACE_OPT_0054_VAE_REVISION7_MANIFEST_BYTES;
      readonly runtimeProfile:
        typeof ACE_OPT_0054_VAE_FP16_FIXED32_REVISION7_PROFILE.id;
      readonly kernelSetId:
        typeof ACE_OPT_0054_VAE_FP16_FIXED32_REVISION7_PROFILE.kernelSetId;
      readonly precisionMapSha256: string;
    }>
  | Readonly<{
      readonly role: "opt-0072-rev7-production";
      readonly manifestSha256:
        typeof ACE_OPT_0054_VAE_REVISION7_MANIFEST_SHA256;
      readonly manifestByteLength:
        typeof ACE_OPT_0054_VAE_REVISION7_MANIFEST_BYTES;
      readonly runtimeProfile:
        typeof ACE_OPT_0072_VAE_FP16_FIXED32_DUAL_K4_PRODUCTION_RUNTIME_PROFILE;
      /** Exact physical owner; the backend never receives the public ID. */
      readonly physicalRuntimeProfile:
        | typeof ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PROFILE.id
        | typeof ACE_OPT_0088_VAE_FP16_PORTABLE_DUAL_K4_PROFILE.id;
      readonly kernelSetId:
        | typeof ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PROFILE.kernelSetId
        | typeof ACE_OPT_0088_VAE_FP16_PORTABLE_DUAL_K4_PROFILE.kernelSetId;
      readonly precisionMapSha256: string;
    }>;

/** @internal Resolve only exact VAE manifest/profile pairs; no fallback. */
export function resolveAceVaePackageRuntimeIdentity(
  configuration: AceWorkerVaePackageConfiguration,
  kernelBackend: AceKernelBackend = "subgroups",
): AceVaePackageRuntimeIdentity {
  if (
    kernelBackend === "portable" &&
    !(configuration.manifestSha256 ===
        ACE_OPT_0054_VAE_REVISION7_MANIFEST_SHA256 &&
      configuration.runtimeProfile ===
        ACE_OPT_0072_VAE_FP16_FIXED32_DUAL_K4_PRODUCTION_RUNTIME_PROFILE)
  ) {
    // Only the public OPT-0072 production identity has a portable OPT-0088
    // physical counterpart; every fixed32-only profile fails closed.
    throw new Error(
      "ACE portable mixed VAE accepts only the OPT-0072 production profile",
    );
  }
  if (
    configuration.manifestSha256 === ACE_OPT_0028_VAE_FP16_MANIFEST_SHA256 &&
    configuration.runtimeProfile ===
      ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_PROFILE.id
  ) {
    return Object.freeze({
      role: "opt-0028-rev6-production",
      manifestSha256: ACE_OPT_0028_VAE_FP16_MANIFEST_SHA256,
      manifestByteLength: ACE_OPT_0028_VAE_FP16_MANIFEST_BYTES,
      runtimeProfile:
        ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_PROFILE.id,
      kernelSetId:
        ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_PROFILE.kernelSetId,
      precisionMapSha256:
        ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_PROFILE.precisionMapSha256!,
    });
  }
  if (
    configuration.manifestSha256 === ACE_OPT_0028_VAE_FP16_MANIFEST_SHA256 &&
    configuration.runtimeProfile ===
      ACE_OPT_0040_VAE_FP16_FIXED32_SHAPE_SELECTED_PROFILE.id
  ) {
    return Object.freeze({
      role: "opt-0040-rev6-scalar-fp32-oracle",
      manifestSha256: ACE_OPT_0028_VAE_FP16_MANIFEST_SHA256,
      manifestByteLength: ACE_OPT_0028_VAE_FP16_MANIFEST_BYTES,
      runtimeProfile:
        ACE_OPT_0040_VAE_FP16_FIXED32_SHAPE_SELECTED_PROFILE.id,
      kernelSetId:
        ACE_OPT_0040_VAE_FP16_FIXED32_SHAPE_SELECTED_PROFILE.kernelSetId,
      precisionMapSha256:
        ACE_OPT_0040_VAE_FP16_FIXED32_SHAPE_SELECTED_PROFILE.precisionMapSha256!,
    });
  }
  if (
    configuration.manifestSha256 ===
      ACE_OPT_0054_VAE_REVISION7_MANIFEST_SHA256 &&
    configuration.runtimeProfile ===
      ACE_OPT_0054_VAE_FP16_FIXED32_REVISION7_PROFILE.id
  ) {
    return Object.freeze({
      role: "opt-0054-rev7-candidate",
      manifestSha256: ACE_OPT_0054_VAE_REVISION7_MANIFEST_SHA256,
      manifestByteLength: ACE_OPT_0054_VAE_REVISION7_MANIFEST_BYTES,
      runtimeProfile: ACE_OPT_0054_VAE_FP16_FIXED32_REVISION7_PROFILE.id,
      kernelSetId:
        ACE_OPT_0054_VAE_FP16_FIXED32_REVISION7_PROFILE.kernelSetId,
      precisionMapSha256:
        ACE_OPT_0054_VAE_FP16_FIXED32_REVISION7_PROFILE.precisionMapSha256!,
    });
  }
  if (
    configuration.manifestSha256 ===
      ACE_OPT_0054_VAE_REVISION7_MANIFEST_SHA256 &&
    configuration.runtimeProfile ===
      ACE_OPT_0072_VAE_FP16_FIXED32_DUAL_K4_PRODUCTION_RUNTIME_PROFILE
  ) {
    const production = requireAceOpt0072VaeProductionRuntimeProfileForBackend(
      configuration.runtimeProfile,
      kernelBackend,
    );
    return Object.freeze({
      role: "opt-0072-rev7-production",
      manifestSha256: production.manifestSha256,
      manifestByteLength: production.manifestByteLength,
      runtimeProfile: production.id,
      physicalRuntimeProfile: production.physicalRuntimeProfileId,
      kernelSetId: production.kernelSetId,
      precisionMapSha256: production.precisionMapSha256!,
    });
  }
  throw new Error(
    "ACE mixed VAE requires one exact authenticated manifest/profile pair",
  );
}

/**
 * @internal Kernel-set identity of the configured quad-query attention
 * profile under the active kernel backend. The query8 kernel set is a
 * per-shape internal owner and never a configured diagnostics identity.
 */
function requireDiagnosticsAttentionKernelSetId(
  profile:
    | typeof ACE_OPT_0062_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE
    | typeof ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
  kernelBackend: AceKernelBackend,
): NonNullable<AceRuntimeDiagnostics["ditAttentionKernelSetId"]> {
  const kernelSetId = resolveAceDitAttentionKernelSetId(
    profile,
    kernelBackend,
  );
  if (kernelSetId === ACE_DIT_QUERY8_ATTENTION_KERNEL_SET_ID) {
    throw new Error(
      "ACE diagnostics require a quad-query attention kernel-set identity",
    );
  }
  return kernelSetId;
}

function createRuntimeDiagnostics(
  configuration: AceWorkerConfiguration,
  loaded: AceLoadedPackageManifest,
  ditDenseLoaded: AceLoadedPackageManifest,
  vaeLoaded: AceLoadedPackageManifest,
  device: PipelineDeviceContext,
): AceRuntimeDiagnostics {
  const kernelBackend =
    device.capabilities.executionProfile.kernelBackend;
  const ditDenseIdentity = resolveAceDitDensePackageRuntimeIdentity(
    configuration.ditDensePackage,
    kernelBackend,
  );
  const vaeIdentity = resolveAceVaePackageRuntimeIdentity(
    configuration.vaePackage,
    kernelBackend,
  );
  const {
    id: vaeWindowRuntimeProfile,
    maximumWindowFrames: vaeMaxWindowFrames,
  } = requireAceVaeWindowRuntimeProfile(
    configuration.vaePackage.windowRuntimeProfile,
    configuration.vaePackage.maxWindowFrames,
  );
  if (
    vaeWindowRuntimeProfile === ACE_VAE_CAPPED_C2176_WINDOW_RUNTIME_PROFILE ||
    vaeMaxWindowFrames === 2_176
  ) {
    // The capped geometry is an adapter-derived downshift, never a
    // configuration identity; diagnostics report the configured contract.
    throw new Error(
      "ACE runtime diagnostics require a configured VAE window identity",
    );
  }
  return Object.freeze({
    backend: "custom-webgpu-wgsl-and-wasm",
    modelManifestId: loaded.manifestId,
    modelManifestUrl: configuration.manifestUrl,
    modelManifestSha256: loaded.manifestSha256,
    ditDenseManifestId: ditDenseLoaded.manifestId,
    ditDenseManifestUrl: configuration.ditDensePackage.manifestUrl,
    ditDenseManifestSha256: ditDenseLoaded.manifestSha256,
    ditDenseManifestByteLength: ditDenseIdentity.manifestByteLength,
    ditDenseRuntimeProfile: ditDenseIdentity.runtimeProfile,
    ditDenseKernelSetId: ditDenseIdentity.kernelSetId,
    ...(configuration.ditAttentionRuntimeProfile === undefined
      ? {}
      : {
          ditAttentionRuntimeProfile:
            configuration.ditAttentionRuntimeProfile,
          ditAttentionKernelSetId: requireDiagnosticsAttentionKernelSetId(
            configuration.ditAttentionRuntimeProfile,
            kernelBackend,
          ),
        }),
    ditDenseLayerBytes: ditDenseIdentity.layerBytes,
    ditResidentWeightBytes: ditDenseIdentity.residentWeightBytes,
    vaeManifestId: vaeLoaded.manifestId,
    vaeManifestUrl: configuration.vaePackage.manifestUrl,
    vaeManifestSha256: vaeLoaded.manifestSha256,
    vaeManifestByteLength: vaeIdentity.manifestByteLength,
    vaeRuntimeProfile: vaeIdentity.runtimeProfile,
    vaeKernelSetId: vaeIdentity.kernelSetId,
    vaePrecisionMapSha256: vaeIdentity.precisionMapSha256,
    vaeWindowRuntimeProfile,
    vaeMaxWindowFrames,
    executionProfile: device.capabilities.executionProfile,
    schedulingProfile: configuration.schedulingProfile,
    capabilities: device.capabilities,
    aceSourceRevision: ACE_SOURCE_REVISION,
    plannerSourceRevision: ACE_PLANNER_SOURCE_REVISION,
    parakeetReferenceRevision: PARAKEET_REFERENCE_REVISION,
  });
}

function requireLoadedManifestIdentity(
  loaded: AceLoadedPackageManifest,
  configuration: AceWorkerConfiguration,
): void {
  if (
    loaded.manifestSha256 !== configuration.manifestSha256 ||
    loaded.manifest.profile !== packageProfile(configuration.modelProfile)
  ) {
    throw new Error("ACE loaded manifest differs from the initialization trust root");
  }
}

function requireLoadedDitDenseManifestIdentity(
  loaded: AceLoadedPackageManifest,
  configuration: AceWorkerDitDensePackageConfiguration,
): void {
  const identity = resolveAceDitDensePackageRuntimeIdentity(configuration);
  if (identity.role === "opt-0009-rev7-oracle") {
    requireAceOpt0009DitDensePackageIdentity(loaded);
  } else {
    requireAceOpt0037DitK4PackageIdentity(loaded);
  }
  if (
    loaded.manifestSha256 !== configuration.manifestSha256 ||
    loaded.manifestByteLength !== identity.manifestByteLength ||
    loaded.manifest.profile !== "fp16-dit-dense-experimental"
  ) {
    throw new Error(
      "Loaded mixed DiT manifest differs from its exact initialization trust root",
    );
  }
}

function requireProductionConfiguration(
  configuration: AceWorkerConfiguration,
): void {
  let ditDenseIdentity: AceDitDensePackageRuntimeIdentity | undefined;
  let vaeIdentity: AceVaePackageRuntimeIdentity | undefined;
  let vaeWindowIdentity:
    | ReturnType<typeof requireAceVaeWindowRuntimeProfile>
    | undefined;
  try {
    ditDenseIdentity = resolveAceDitDensePackageRuntimeIdentity(
      configuration.ditDensePackage,
    );
  } catch {
    // The single error below is the stable external initialization boundary.
  }
  try {
    vaeIdentity = resolveAceVaePackageRuntimeIdentity(configuration.vaePackage);
    vaeWindowIdentity = requireAceVaeWindowRuntimeProfile(
      configuration.vaePackage.windowRuntimeProfile,
      configuration.vaePackage.maxWindowFrames,
    );
  } catch {
    // The single error below is the stable external initialization boundary.
  }
  if (
    configuration.modelProfile !== "reference-bf16" ||
    ditDenseIdentity === undefined ||
    typeof configuration.ditDensePackage.manifestUrl !== "string" ||
    configuration.ditDensePackage.manifestUrl.length === 0 ||
    (configuration.ditAttentionRuntimeProfile !== undefined &&
      ((configuration.ditAttentionRuntimeProfile !==
          ACE_OPT_0062_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE &&
        configuration.ditAttentionRuntimeProfile !==
          ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE) ||
        configuration.ditDensePackage.runtimeProfile !==
          ACE_OPT_0009_DIT_DENSE_RUNTIME_PROFILE)) ||
    vaeIdentity === undefined ||
    vaeWindowIdentity === undefined ||
    ((configuration.ditAttentionRuntimeProfile ===
        ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE) !==
      (vaeWindowIdentity?.id ===
        ACE_OPT_0070_VAE_C2378_WINDOW_RUNTIME_PROFILE)) ||
    ((vaeIdentity.role === "opt-0072-rev7-production") !==
      (vaeWindowIdentity?.id ===
        ACE_OPT_0070_VAE_C2378_WINDOW_RUNTIME_PROFILE)) ||
    typeof configuration.vaePackage.manifestUrl !== "string" ||
    configuration.vaePackage.manifestUrl.length === 0
  ) {
    throw new Error(
      "ACE production requires a reference main package and the exact " +
        "authenticated mixed DiT and FP16 VAE configurations",
    );
  }
}

function requireLoadedVaeManifestIdentity(
  loaded: AceLoadedPackageManifest,
  configuration: AceWorkerVaePackageConfiguration,
): void {
  const identity = resolveAceVaePackageRuntimeIdentity(configuration);
  if (identity.role === "opt-0072-rev7-production") {
    requireAceOpt0066Fp16VaePackageIdentity(loaded);
  } else if (identity.role === "opt-0054-rev7-candidate") {
    requireAceOpt0054Fp16VaePackageIdentity(loaded);
  } else {
    requireAceOpt0028Fp16VaePackageIdentity(loaded);
  }
  if (
    loaded.manifestSha256 !== configuration.manifestSha256 ||
    loaded.manifestByteLength !== identity.manifestByteLength ||
    loaded.manifest.profile !== "fp16-vae-experimental"
  ) {
    throw new Error(
      "Loaded VAE manifest differs from its exact initialization trust root",
    );
  }
}

function requireVaeAcquisition(acquired: AceAcquiredModelFiles): void {
  const names = [...acquired.files.keys()];
  if (
    acquired.files.size !== ACE_OPT_0011_VAE_FP16_WEIGHT_FILES.length ||
    acquired.plan.files.length !== ACE_OPT_0011_VAE_FP16_WEIGHT_FILES.length ||
    acquired.plan.runtimeBytes !== ACE_EXPERIMENTAL_VAE_PARAMETER_BYTES ||
    ACE_OPT_0011_VAE_FP16_WEIGHT_FILES.some((name, index) =>
      names[index] !== name || acquired.plan.files[index]?.name !== name
    )
  ) {
    throw new Error(
      "OPT-0011 acquired package is not exactly seven VAE weight files",
    );
  }
}

function requireDitDenseAcquisition(
  acquired: AceAcquiredModelFiles,
  identity: AceDitDensePackageRuntimeIdentity,
): void {
  const names = [...acquired.files.keys()];
  const expectedFiles = identity.role === "opt-0009-rev7-oracle"
    ? ACE_OPT_0009_DIT_DENSE_WEIGHT_FILES
    : ACE_OPT_0037_DIT_K4_WEIGHT_FILES;
  if (
    acquired.files.size !== expectedFiles.length ||
    acquired.plan.files.length !== expectedFiles.length ||
    acquired.plan.runtimeBytes !== identity.layerBytes ||
    expectedFiles.some((name, index) =>
      names[index] !== name || acquired.plan.files[index]?.name !== name
    )
  ) {
    throw new Error(
      `Acquired ${identity.role} package is not exactly 48 DiT layer files`,
    );
  }
}

function requireProductionDeviceCapabilities(
  context: PipelineDeviceContext,
  requiredWorkspaceBytes: number,
): void {
  const capabilities = context.capabilities;
  const shader = capabilities.stockFeatures["shader-f16"];
  const subgroups = capabilities.stockFeatures.subgroups;
  // shader-f16 and the VAE workspace limits are hard-required by both
  // accepted production tuples.
  const sharedClausesValid =
    capabilities.requiredFeatures.includes("shader-f16") &&
    capabilities.adapterFeatures.includes("shader-f16") &&
    capabilities.deviceFeatures.includes("shader-f16") &&
    shader.adapterSupported &&
    shader.deviceEnabled &&
    shader.required &&
    shader.requested &&
    capabilities.deviceLimits.maxBufferSize >= requiredWorkspaceBytes &&
    capabilities.deviceLimits.maxStorageBufferBindingSize >=
      requiredWorkspaceBytes;
  // Accepted tuple 1: the audited fixed-32 subgroup device, unchanged.
  const fixed32TupleValid =
    capabilities.executionProfile.id === ACE_REFERENCE_SUBGROUP_PROFILE.id &&
    capabilities.adapterInfo.subgroupMinSize === 32 &&
    capabilities.adapterInfo.subgroupMaxSize === 32 &&
    capabilities.requiredFeatures.includes("subgroups") &&
    capabilities.adapterFeatures.includes("subgroups") &&
    capabilities.deviceFeatures.includes("subgroups") &&
    subgroups.adapterSupported &&
    subgroups.deviceEnabled &&
    subgroups.required &&
    subgroups.requested;
  // Accepted tuple 2: the portable no-subgroup profile. The adapter may
  // legitimately advertise non-fixed-32 subgroups; the device must not have
  // requested, required, or enabled them.
  const portableTupleValid =
    capabilities.executionProfile.id === ACE_REFERENCE_PORTABLE_PROFILE.id &&
    !capabilities.requiredFeatures.includes("subgroups") &&
    !capabilities.deviceFeatures.includes("subgroups") &&
    !subgroups.deviceEnabled &&
    !subgroups.required &&
    !subgroups.requested;
  if (!sharedClausesValid || (!fixed32TupleValid && !portableTupleValid)) {
    throw new Error(
      "ACE production device did not enable a coherent tuple: fixed32 " +
        "subgroup OPT-0037 or portable OPT-0088 with shader-f16 and packed " +
        "VAE capabilities",
    );
  }
}

function createAudioTransactionId(uuid: string): string {
  const id = `ace-${uuid}`;
  if (!SAFE_AUDIO_STORAGE_ID.test(id)) {
    throw new Error("crypto.randomUUID() returned an unsafe ACE transaction identity");
  }
  return id;
}

class InitializationProgressReporter {
  private stageIndex = 0;
  private lastCompleted = 0;
  private lastTotal: number | undefined;

  constructor(
    private readonly emit: (progress: AceInitializationProgress) => void,
    private readonly now: () => number,
    private readonly start: number,
  ) {}

  report(
    stage: AceInitializationStage,
    completedUnits: number,
    totalUnits: number,
    unit: AceProgressUnit,
    message?: string,
  ): void {
    const index = ACE_INITIALIZATION_STAGES.indexOf(stage);
    if (index !== this.stageIndex) {
      throw new Error(`ACE initialization progress expected ${ACE_INITIALIZATION_STAGES[this.stageIndex]}`);
    }
    if (completedUnits < this.lastCompleted) return;
    if (this.lastTotal !== undefined && totalUnits !== this.lastTotal) return;
    this.lastCompleted = completedUnits;
    this.lastTotal = totalUnits;
    const fraction = totalUnits === 0 ? 1 : completedUnits / totalUnits;
    this.emit(Object.freeze({
      stage,
      completedUnits,
      totalUnits,
      unit,
      overallFraction: (index + fraction) / ACE_INITIALIZATION_STAGES.length,
      elapsedMs: nonnegativeElapsed(this.now(), this.start),
      ...(message === undefined ? {} : { message }),
    }));
  }

  complete(
    stage: AceInitializationStage,
    totalUnits: number,
    unit: AceProgressUnit,
    message?: string,
  ): void {
    this.report(stage, totalUnits, totalUnits, unit, message);
    this.stageIndex += 1;
    this.lastCompleted = 0;
    this.lastTotal = undefined;
  }

  skip(stage: AceInitializationStage, message: string): void {
    this.complete(stage, 0, "items", message);
  }
}

class GenerationProgressTracker {
  private index = 0;
  private stageStartedAt: number;
  private readonly completedTimings: AceStageTiming[] = [];

  constructor(
    private readonly plan: readonly AceGenerationStage[],
    private readonly emit: (progress: AceGenerationProgress) => void,
    private readonly now: () => number,
    private readonly start: number,
  ) {
    this.stageStartedAt = start;
  }

  get current(): AceGenerationStage | undefined {
    return this.plan[this.index];
  }

  get timings(): readonly AceStageTiming[] {
    return Object.freeze([...this.completedTimings]);
  }

  progress(
    stage: AceGenerationStage,
    completedUnits: number,
    totalUnits: number,
    unit: AceProgressUnit,
    message?: string,
  ): void {
    if (stage !== this.current) return;
    const fraction = totalUnits === 0 ? 0 : Math.min(1, completedUnits / totalUnits);
    this.emit(Object.freeze({
      stage,
      completedUnits,
      totalUnits,
      unit,
      stageIndex: this.index,
      stageCount: this.plan.length,
      overallFraction: (this.index + fraction) / this.plan.length,
      elapsedMs: nonnegativeElapsed(this.now(), this.start),
      ...(message === undefined ? {} : { message }),
    }));
  }

  complete(
    stage: AceGenerationStage,
    totalUnits: number,
    unit: AceProgressUnit,
    message?: string,
    cooperativeStageIdleMs?: number,
  ): void {
    if (stage !== this.current) {
      throw new Error(`ACE generation progress expected ${String(this.current)}, got ${stage}`);
    }
    this.progress(stage, totalUnits, totalUnits, unit, message);
    const finishedAt = this.now();
    this.completedTimings.push(Object.freeze({
      stage,
      wallMs: nonnegativeElapsed(finishedAt, this.stageStartedAt),
      ...(cooperativeStageIdleMs === undefined
        ? {}
        : { cooperativeIdleMs: cooperativeStageIdleMs }),
    }));
    this.index += 1;
    this.stageStartedAt = finishedAt;
  }
}

function mapConditioningProgress(
  event: AceConditioningGpuProgress,
  mode: AcePlannerCoordinatorResult["plannerMode"],
  tracker: GenerationProgressTracker,
): void {
  const stage = event.phase === "text"
    ? "text-encoder"
    : event.phase === "semantic"
      ? "semantic-detokenizer"
      : "condition-encoder";
  while (tracker.current !== stage) {
    if (tracker.current === "text-encoder") {
      tracker.complete("text-encoder", 2, "items", "Text embeddings complete");
    } else if (tracker.current === "semantic-detokenizer") {
      if (mode === "disabled") {
        tracker.complete(
          "semantic-detokenizer",
          0,
          "items",
          "Skipped: planner disabled",
        );
      } else {
          tracker.complete(
            "semantic-detokenizer",
            2,
          "items",
          "Semantic hints decoded",
        );
      }
    } else {
      return;
    }
  }
  // One logical conditioning stage contains several graph dispatches with
  // different command totals. Events drive honest stage transitions and
  // metrics, while 0/0 is an explicit discovery heartbeat and the fixed
  // logical counter is emitted once at completion.
  tracker.progress(stage, 1, 2, "items", event.quantumId);
}

function completeConditioningStages(
  tracker: GenerationProgressTracker,
  mode: AcePlannerCoordinatorResult["plannerMode"],
): void {
  completeIfCurrent(tracker, "text-encoder", 2, "items", "Text embeddings complete");
  completeIfCurrent(
    tracker,
    "semantic-detokenizer",
    mode === "enabled" ? 2 : 0,
    "items",
    mode === "enabled" ? "Semantic hints decoded" : "Skipped: planner disabled",
  );
  completeIfCurrent(
    tracker,
    "condition-encoder",
    2,
    "items",
    "Condition tensors complete",
  );
}

function mapDitProgress(
  event: AceDitGpuBackendProgress,
  tracker: GenerationProgressTracker,
  evaluationCount = 8,
): void {
  if (event.stage === "compile") {
    tracker.progress("dit-load", 2, 3, "items", "Compiling DiT graph");
    if (event.compiledQuanta === event.totalQuanta) {
      completeIfCurrent(
        tracker,
        "dit-load",
        3,
        "items",
        "DiT loaded and compiled",
      );
    }
    return;
  }
  completeIfCurrent(tracker, "dit-load", 3, "items", "DiT loaded and compiled");
  tracker.progress(
    "dit-denoise",
    event.completedEvaluations,
    evaluationCount,
    "denoising-evaluations",
    event.stage === "readback" ? "Reading final latent" : "Denoising",
  );
}

function mapPlannerProgress(
  event: AcePlannerCoordinatorProgress,
  semanticSteps: number,
  totalPlannerSteps: number,
  tracker: GenerationProgressTracker,
): void {
  if (event.kind !== "token") return;
  const completed = event.phase === "cot"
    ? 1 + event.completedTokens / ACE_PLANNER_COT_MAX_NEW_TOKENS
    : 2 + event.completedTokens / semanticSteps;
  tracker.progress(
    "semantic-planner",
    Math.min(completed, totalPlannerSteps),
    totalPlannerSteps,
    "planner-steps",
    event.phase === "cot"
      ? "Reasoning about caption and metadata"
      : "Generating semantic codes",
  );
}

function mapVaeProgress(
  event: AceVaeChunkGpuBackendProgress,
  plan: AceVaeChunkedDecodePlan,
  tracker: GenerationProgressTracker,
): void {
  completeIfCurrent(tracker, "vae-load", 2, "items", "VAE loaded and compiled");
  const completed = event.stage === "readback"
    ? event.windowIndex + 1
    : event.windowIndex + event.completedDecoderQuanta / event.totalDecoderQuanta;
  tracker.progress(
    "vae-decode",
    completed,
    plan.windows.length,
    "vae-chunks",
    event.stage === "readback"
      ? `VAE window ${event.windowIndex + 1}/${plan.windows.length}`
      : `VAE window ${event.windowIndex + 1}/${plan.windows.length} · ` +
        `quantum ${event.completedDecoderQuanta}/${event.totalDecoderQuanta}`,
  );
}

function resolveVaeSchedulingSelection(
  productionPolicy: AceOpt0080VaeProductionSchedulingPolicy | undefined,
  benchmarkOverride: AceOpt0080ProductVaeSchedulingOverride | undefined,
): AceVaeSchedulingSelection {
  if (benchmarkOverride !== undefined) return "benchmark-override";
  return productionPolicy === undefined ? "depth1-default" : "production";
}

function resolvePipelineVaeWindowSchedulingProfile(
  productionPolicy: AceOpt0080VaeProductionSchedulingPolicy | undefined,
  benchmarkOverride: AceOpt0080ProductVaeSchedulingOverride | undefined,
  window: AceVaeDecodeWindow,
): AceOpt0080VaeSchedulingProfile {
  if (benchmarkOverride === "depth1-epoch1") return "depth1-epoch1";
  return resolveOpt0080VaeProductionWindowSchedulingProfile(
    benchmarkOverride ?? productionPolicy,
    window.latentWindowFrames,
  );
}

function createVaeWindowSchedulingReceipt(
  window: AceVaeDecodeWindow,
  event: AceVaeChunkGpuBackendProgress,
  selection: AceVaeSchedulingSelection,
  schedulingProfile: AceOpt0080VaeSchedulingProfile,
  quantaPerCommandBuffer: number,
): AceVaeWindowSchedulingReceipt {
  const decoderCommandBufferCount = event.totalCommandBuffers - 1;
  const expectedQueueDrains = schedulingProfile === "depth2-phase-epoch4"
    ? Math.ceil(event.totalCommandBuffers / 4)
    : event.totalCommandBuffers;
  const expectedIdleTurns = Math.max(0, expectedQueueDrains - 1);
  if (
    event.stage !== "readback" ||
    event.windowIndex !== window.index ||
    event.completedDecoderQuanta !== event.totalDecoderQuanta ||
    event.completedCommandBuffers !== event.totalCommandBuffers ||
    decoderCommandBufferCount !==
      Math.ceil(event.totalDecoderQuanta / quantaPerCommandBuffer) ||
    event.queueDrains !== expectedQueueDrains ||
    event.cooperativeIdleMs !== expectedIdleTurns
  ) {
    throw new Error("VAE scheduling receipt observed incomplete topology");
  }
  return Object.freeze({
    windowIndex: window.index,
    latentWindowFrames: window.latentWindowFrames,
    selection,
    schedulingProfile,
    decoderQuantumCount: event.totalDecoderQuanta,
    quantaPerCommandBuffer,
    decoderCommandBufferCount,
    readbackCommandBufferCount: 1 as const,
    totalCommandBufferCount: event.totalCommandBuffers,
    commandBuffersSubmitted: event.completedCommandBuffers,
    queueDrains: event.queueDrains,
    cooperativeIdleTurns: expectedIdleTurns,
    maximumOutstandingCommandBuffers:
      schedulingProfile === "depth2-phase-epoch4" ? 2 as const : 1 as const,
  });
}

function createVaeSchedulingReceipt(
  plan: AceVaeChunkedDecodePlan,
  productionPolicy: AceOpt0080VaeProductionSchedulingPolicy | undefined,
  benchmarkOverride: AceOpt0080ProductVaeSchedulingOverride | undefined,
  receipts: ReadonlyMap<number, AceVaeWindowSchedulingReceipt>,
): AceVaeSchedulingReceipt {
  if (
    productionPolicy !== undefined && benchmarkOverride !== undefined ||
    receipts.size !== plan.windows.length
  ) {
    throw new Error("VAE scheduling receipt inventory changed");
  }
  const windows = Object.freeze(plan.windows.map((window) => {
    const receipt = receipts.get(window.index);
    if (
      receipt === undefined ||
      receipt.windowIndex !== window.index ||
      receipt.latentWindowFrames !== window.latentWindowFrames
    ) {
      throw new Error("VAE scheduling receipt lost plan order");
    }
    return receipt;
  }));
  return Object.freeze({
    schema: "ace-vae-window-scheduling-receipt-v1" as const,
    selectedProductionPolicy: productionPolicy ?? null,
    benchmarkPolicyOverride: benchmarkOverride ?? null,
    windows,
  });
}

function mapPhaseUpload(
  event: AceGpuTensorPhaseProgress,
  tracker: GenerationProgressTracker,
  stage: AceGenerationStage,
  offset: number,
  span: number,
  total: number,
  unit: AceProgressUnit = "items",
): void {
  const fraction = event.totalPhaseBytes === 0
    ? 0
    : event.loadedPhaseBytes / event.totalPhaseBytes;
  tracker.progress(
    stage,
    Math.min(total, offset + span * fraction),
    total,
    unit,
    `Loading ${event.upload.file}: ${event.loadedPhaseBytes}/${event.totalPhaseBytes} bytes`,
  );
}

function completeIfCurrent(
  tracker: GenerationProgressTracker,
  stage: AceGenerationStage,
  total: number,
  unit: AceProgressUnit,
  message: string,
): void {
  if (tracker.current === stage) tracker.complete(stage, total, unit, message);
}

function addVaeMetrics(
  windows: ReadonlyMap<number, Readonly<{ drains: number; idleMs: number }>>,
  raw: AceVaeRawStreamStats,
  add: (drains: number, idleMs: number) => void,
): void {
  let drains = 0;
  let idleMs = raw.cooperativeIdleMs;
  for (const value of windows.values()) {
    drains += value.drains;
    idleMs += value.idleMs;
  }
  add(drains, idleMs);
}

function createSamplerScheduleDiagnosticEvidence(
  schedule: Readonly<AceDitSamplerScheduleProfile>,
  denoised: AceDitGpuBackendResult,
  raw: AceVaeRawStreamStats,
  committed: AceCommittedAudioOutput,
): AceSamplerScheduleDiagnosticEvidence {
  const evaluationLatents = denoised.evaluationLatents;
  const evaluation0Velocity = denoised.evaluation0Velocity;
  const rawSnapshot = committed.rawSnapshot;
  if (
    denoised.completedEvaluations !== schedule.evaluationCount ||
    evaluationLatents === undefined ||
    evaluationLatents.length !== schedule.evaluationCount ||
    evaluation0Velocity === undefined ||
    rawSnapshot === undefined ||
    rawSnapshot.size !==
      raw.outputInterleavedElements * Float32Array.BYTES_PER_ELEMENT ||
    raw.finiteSamples !== raw.outputInterleavedElements
  ) {
    throw new Error(
      "OPT-0055/0065 sampler diagnostic evidence inventory changed",
    );
  }
  const evaluations = Object.freeze(evaluationLatents.map(
    (latent, evaluation): AceSamplerScheduleEvaluationEvidence =>
      Object.freeze({
        evaluation: evaluation as AceOpt0080FullEvaluationIndex,
        ...createSamplerScheduleTensorEvidence(latent),
      }),
  ));
  const finalWords = new Uint32Array(
    denoised.finalLatent.buffer,
    denoised.finalLatent.byteOffset,
    denoised.finalLatent.length,
  );
  const last = evaluationLatents.at(-1)!;
  const lastWords = new Uint32Array(
    last.buffer,
    last.byteOffset,
    last.length,
  );
  if (
    finalWords.length !== lastWords.length ||
    finalWords.some((word, index) => word !== lastWords[index])
  ) {
    throw new Error(
      "OPT-0055/0065 final latent diverged from the final sampler snapshot",
    );
  }
  return Object.freeze({
    schema: "ace-opt-0055-0065-sampler-schedule-evidence-v1",
    schedule,
    evaluation0Velocity: createSamplerScheduleTensorEvidence(
      evaluation0Velocity,
    ),
    evaluations,
    finalLatent: createSamplerScheduleTensorEvidence(denoised.finalLatent),
    rawAudio: Object.freeze({
      snapshot: rawSnapshot,
      byteLength: rawSnapshot.size,
      peak: raw.peak,
      finiteSamples: raw.finiteSamples,
      interleavedSamples: raw.outputInterleavedElements,
    }),
    wav: committed.audio,
  });
}

function createSamplerScheduleTensorEvidence(
  values: Float32Array<ArrayBuffer>,
): AceSamplerScheduleTensorEvidence {
  let nonFiniteCount = 0;
  let nonzeroCount = 0;
  let maximumAbsolute = 0;
  for (const value of values) {
    if (!Number.isFinite(value)) {
      nonFiniteCount += 1;
      continue;
    }
    if (value !== 0) nonzeroCount += 1;
    maximumAbsolute = Math.max(maximumAbsolute, Math.abs(value));
  }
  if (nonFiniteCount !== 0 || nonzeroCount === 0 || maximumAbsolute === 0) {
    throw new Error(
      "OPT-0055/0065 tensor evidence is non-finite or all zero",
    );
  }
  return Object.freeze({
    values,
    byteLength: values.byteLength,
    elementCount: values.length,
    sha256: sha256F32Le(values),
    nonFiniteCount,
    nonzeroCount,
    maximumAbsolute,
  });
}

function createOpt0018DitCheckpoint(
  denoised: AceDitGpuBackendResult,
  profile: AceOpt0018DitCommandProfile,
  stageTimings: readonly AceStageTiming[],
): AceOpt0018DitCheckpoint {
  const finalLatent = denoised.finalLatent;
  if (
    denoised.shape.batch !== 1 ||
    denoised.shape.latentFrames !== 4_500 ||
    denoised.shape.channels !== 64 ||
    finalLatent.length !== 288_000 ||
    finalLatent.byteLength !== 1_152_000 ||
    denoised.commandBuffersSubmitted !== 2_554 ||
    denoised.queueDrains !== 2_554 ||
    denoised.completedEvaluations !== 8 ||
    profile.graphCommandBufferCount !== 2_553 ||
    profile.readbackCommandBufferCount !== 1 ||
    profile.totalCommandBufferCount !== 2_554
  ) {
    throw new Error("OPT-0018 final DiT checkpoint topology changed");
  }
  let nonFinite = 0;
  let nonzero = 0;
  let maxAbs = 0;
  for (const value of finalLatent) {
    if (!Number.isFinite(value)) {
      nonFinite += 1;
      continue;
    }
    if (value !== 0) nonzero += 1;
    maxAbs = Math.max(maxAbs, Math.abs(value));
  }
  if (nonFinite !== 0 || nonzero === 0 || !(maxAbs > 0)) {
    throw new Error("OPT-0018 final latent is non-finite or all zero");
  }
  return Object.freeze({
    schema: "ace-dit-m2250-checkpoint-v1",
    finalLatent,
    finalLatentByteLength: 1_152_000,
    finalLatentElementCount: 288_000,
    finalLatentSha256: sha256F32Le(finalLatent),
    finalLatentNonFiniteCount: 0,
    finalLatentNonzeroCount: nonzero,
    finalLatentMaxAbs: maxAbs,
    profile,
    stageTimings: Object.freeze([...stageTimings]),
  });
}

function createOpt0056DitCheckpoint(
  denoised: AceDitGpuBackendResult,
  profile: AceOpt0018DitCommandProfile,
  stageTimings: readonly AceStageTiming[],
  denseRuntimeProfile:
    AceWorkerDitDensePackageConfiguration["runtimeProfile"],
): AceOpt0056DitCheckpoint {
  const base = createOpt0018DitCheckpoint(denoised, profile, stageTimings);
  const rawEvaluations = denoised.evaluationLatents;
  const selective = denseRuntimeProfile ===
    ACE_OPT_0056_DIT_SELECTIVE_K4_RUNTIME_PROFILE;
  if (
    rawEvaluations === undefined ||
    rawEvaluations.length !== 8 ||
    (selective && denoised.denseRouteProfile === undefined) ||
    (!selective && denoised.denseRouteProfile !== undefined) ||
    (denoised.denseRouteProfile !== undefined &&
      (denoised.denseRouteProfile.routeCount !== 216 ||
        denoised.denseRouteProfile.dispatchCount !== 1_728 ||
        denoised.denseRouteProfile.approximateRouteCount !== 192 ||
        denoised.denseRouteProfile.exactDownRouteCount !== 24))
  ) {
    throw new Error("OPT-0056 trajectory or dense route inventory changed");
  }
  const evaluations = Object.freeze(rawEvaluations.map((latent, evaluation) => {
    if (
      !(latent instanceof Float32Array) ||
      !(latent.buffer instanceof ArrayBuffer) ||
      latent.length !== 288_000 ||
      latent.byteLength !== 1_152_000
    ) {
      throw new Error(`OPT-0056 evaluation ${evaluation} latent shape changed`);
    }
    let nonFinite = 0;
    let nonzero = 0;
    let maximumAbsolute = 0;
    for (const value of latent) {
      if (!Number.isFinite(value)) {
        nonFinite += 1;
      } else {
        if (value !== 0) nonzero += 1;
        maximumAbsolute = Math.max(maximumAbsolute, Math.abs(value));
      }
    }
    if (nonFinite !== 0 || nonzero === 0 || !(maximumAbsolute > 0)) {
      throw new Error(`OPT-0056 evaluation ${evaluation} latent is invalid`);
    }
    return Object.freeze({
        evaluation: evaluation as AceOpt0080FullEvaluationIndex,
      latent,
      latentByteLength: 1_152_000 as const,
      latentElementCount: 288_000 as const,
      latentSha256: sha256F32Le(latent),
      nonFiniteCount: 0 as const,
      nonzeroCount: nonzero,
      maximumAbsolute,
    });
  }));
  if (evaluations.at(-1)!.latentSha256 !== base.finalLatentSha256) {
    throw new Error("OPT-0056 final sampler tap differs from final readback");
  }
  return Object.freeze({
    schema: "ace-dit-opt0056-m2250-trajectory-checkpoint-v1",
    finalLatent: base.finalLatent,
    finalLatentByteLength: base.finalLatentByteLength,
    finalLatentElementCount: base.finalLatentElementCount,
    finalLatentSha256: base.finalLatentSha256,
    finalLatentNonFiniteCount: base.finalLatentNonFiniteCount,
    finalLatentNonzeroCount: base.finalLatentNonzeroCount,
    finalLatentMaxAbs: base.finalLatentMaxAbs,
    evaluations,
    ...(denoised.denseRouteProfile === undefined
      ? {}
      : { denseRouteProfile: denoised.denseRouteProfile }),
    profile,
    stageTimings: base.stageTimings,
    graphCommandBufferCount: 2_553 as const,
    readbackCommandBufferCount: 1 as const,
    totalCommandBufferCount: 2_554 as const,
    snapshotCopyCount: 8 as const,
    snapshotExtraCommandBufferCount: 0 as const,
    snapshotExtraQueueDrainCount: 0 as const,
  });
}

function summarizeOpt0056DitCheckpoint(
  checkpoint: AceOpt0056DitCheckpoint,
): NonNullable<AceDiagnostic["details"]> {
  return Object.freeze({
    schema: "ace-opt-0056-m2250-trajectory-receipt-v1",
    checkpointSchema: checkpoint.schema,
    evaluationCount: checkpoint.evaluations.length,
    evaluationHashesJson: JSON.stringify(
      checkpoint.evaluations.map(({ evaluation, latentSha256 }) =>
        [evaluation, latentSha256]
      ),
    ),
    finalLatentSha256: checkpoint.finalLatentSha256,
    graphCommandBufferCount: checkpoint.graphCommandBufferCount,
    readbackCommandBufferCount: checkpoint.readbackCommandBufferCount,
    totalCommandBufferCount: checkpoint.totalCommandBufferCount,
    snapshotCopyCount: checkpoint.snapshotCopyCount,
    snapshotExtraCommandBufferCount:
      checkpoint.snapshotExtraCommandBufferCount,
    snapshotExtraQueueDrainCount: checkpoint.snapshotExtraQueueDrainCount,
    denseRouteCount: checkpoint.denseRouteProfile?.routeCount ?? 0,
    denseDispatchCount: checkpoint.denseRouteProfile?.dispatchCount ?? 0,
    approximateRouteCount:
      checkpoint.denseRouteProfile?.approximateRouteCount ?? 0,
    exactDownRouteCount: checkpoint.denseRouteProfile?.exactDownRouteCount ?? 0,
  });
}

function createOpt0062DitCheckpoint(
  denoised: AceDitGpuBackendResult,
  profile: AceOpt0018DitCommandProfile,
  stageTimings: readonly AceStageTiming[],
  attentionRuntimeProfile:
    AceWorkerConfiguration["ditAttentionRuntimeProfile"],
  requireActualLayerIdentity: boolean,
): AceOpt0062DitCheckpoint {
  const base = createOpt0018DitCheckpoint(denoised, profile, stageTimings);
  const rawEvaluations = denoised.evaluationLatents;
  const quad = attentionRuntimeProfile ===
    ACE_OPT_0062_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE;
  const routeProfile = denoised.attentionRouteProfile;
  const actualLayerIdentity = denoised.opt0062AttentionIdentity;
  if (
    rawEvaluations === undefined ||
    rawEvaluations.length !== 8 ||
    denoised.denseRouteProfile !== undefined ||
    (quad && routeProfile === undefined) ||
    (!quad && routeProfile !== undefined) ||
    (actualLayerIdentity !== undefined && !quad) ||
    (requireActualLayerIdentity &&
      (actualLayerIdentity === undefined ||
        actualLayerIdentity.routeCount !== 96 ||
        actualLayerIdentity.outputElementsPerRoute !== 4_608_000 ||
        actualLayerIdentity.totalComparedElements !== 442_368_000 ||
        actualLayerIdentity.totalMismatchCount !== 0 ||
        actualLayerIdentity.totalNonFiniteCount !== 0 ||
        actualLayerIdentity.totalCanaryCount !== 0 ||
        actualLayerIdentity.copyCount !== 1 ||
        actualLayerIdentity.extraCommandBufferCount !== 0 ||
        actualLayerIdentity.extraQueueDrainCount !== 0 ||
        actualLayerIdentity.routes.length !== 96 ||
        actualLayerIdentity.routes.some((route, routeIndex) =>
          route.routeIndex !== routeIndex ||
          route.comparedElements !== 4_608_000 ||
          route.mismatchCount !== 0 ||
          route.query8NonFiniteCount !== 0 ||
          route.quadNonFiniteCount !== 0 ||
          route.query8CanaryCount !== 0 ||
          route.quadCanaryCount !== 0
        ))) ||
    (!requireActualLayerIdentity && actualLayerIdentity !== undefined) ||
    (routeProfile !== undefined &&
      (routeProfile.runtimeProfileId !==
          ACE_OPT_0062_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE ||
        routeProfile.kernelSetId !==
          ACE_OPT_0062_DIT_QUAD_QUERY_ATTENTION_KERNEL_SET_ID ||
        routeProfile.quadQueryRoutes !== 96 ||
        routeProfile.query8SlidingRoutes !== 96 ||
        routeProfile.query8CrossRoutes !== 192 ||
        routeProfile.query8OtherRoutes !== 0 ||
        routeProfile.unintendedQuadQueryRoutes !== 0 ||
        routeProfile.uniqueQuadQueryRouteIds.length !== 96))
  ) {
    throw new Error("OPT-0062 trajectory or attention route inventory changed");
  }
  const ownerMembers = profile.descriptorTable.descriptors.flatMap(
    ({ members }) => members,
  );
  const quadDescriptorCount = ownerMembers.filter(({ backend }) =>
    backend === "opt-0062-fixed32-quad-query32-full-self"
  ).length;
  const query8SelfFullDescriptorCount = ownerMembers.filter(({ family, backend }) =>
    family === "self-full" && backend === "fixed32-subgroup-query8"
  ).length;
  if (
    (quad &&
      (quadDescriptorCount !== 480 || query8SelfFullDescriptorCount !== 0)) ||
    (!quad &&
      (quadDescriptorCount !== 0 || query8SelfFullDescriptorCount !== 480))
  ) {
    throw new Error("OPT-0062 physical command attribution changed");
  }
  const evaluations = Object.freeze(rawEvaluations.map((latent, evaluation) => {
    if (
      !(latent instanceof Float32Array) ||
      !(latent.buffer instanceof ArrayBuffer) ||
      latent.length !== 288_000 ||
      latent.byteLength !== 1_152_000
    ) {
      throw new Error(`OPT-0062 evaluation ${evaluation} latent shape changed`);
    }
    let nonFinite = 0;
    let nonzero = 0;
    let maximumAbsolute = 0;
    for (const value of latent) {
      if (!Number.isFinite(value)) {
        nonFinite += 1;
      } else {
        if (value !== 0) nonzero += 1;
        maximumAbsolute = Math.max(maximumAbsolute, Math.abs(value));
      }
    }
    if (nonFinite !== 0 || nonzero === 0 || !(maximumAbsolute > 0)) {
      throw new Error(`OPT-0062 evaluation ${evaluation} latent is invalid`);
    }
    return Object.freeze({
      evaluation,
      latent,
      latentByteLength: 1_152_000 as const,
      latentElementCount: 288_000 as const,
      latentSha256: sha256F32Le(latent),
      nonFiniteCount: 0 as const,
      nonzeroCount: nonzero,
      maximumAbsolute,
    });
  }));
  if (evaluations.at(-1)!.latentSha256 !== base.finalLatentSha256) {
    throw new Error("OPT-0062 final sampler tap differs from final readback");
  }
  return Object.freeze({
    schema: "ace-dit-opt0062-m2250-trajectory-checkpoint-v1",
    finalLatent: base.finalLatent,
    finalLatentByteLength: base.finalLatentByteLength,
    finalLatentElementCount: base.finalLatentElementCount,
    finalLatentSha256: base.finalLatentSha256,
    finalLatentNonFiniteCount: base.finalLatentNonFiniteCount,
    finalLatentNonzeroCount: base.finalLatentNonzeroCount,
    finalLatentMaxAbs: base.finalLatentMaxAbs,
    evaluations,
    ...(routeProfile === undefined ? {} : { attentionRouteProfile: routeProfile }),
    ...(actualLayerIdentity === undefined
      ? {}
      : { actualLayerIdentity }),
    profile,
    stageTimings: base.stageTimings,
    graphCommandBufferCount: 2_553 as const,
    readbackCommandBufferCount: 1 as const,
    totalCommandBufferCount: 2_554 as const,
    snapshotCopyCount: 8 as const,
    snapshotExtraCommandBufferCount: 0 as const,
    snapshotExtraQueueDrainCount: 0 as const,
  });
}

function summarizeOpt0062DitCheckpoint(
  checkpoint: AceOpt0062DitCheckpoint,
): NonNullable<AceDiagnostic["details"]> {
  return Object.freeze({
    schema: "ace-opt-0062-m2250-trajectory-receipt-v1",
    checkpointSchema: checkpoint.schema,
    evaluationCount: checkpoint.evaluations.length,
    evaluationHashesJson: JSON.stringify(
      checkpoint.evaluations.map(({ evaluation, latentSha256 }) =>
        [evaluation, latentSha256]
      ),
    ),
    finalLatentSha256: checkpoint.finalLatentSha256,
    graphCommandBufferCount: checkpoint.graphCommandBufferCount,
    readbackCommandBufferCount: checkpoint.readbackCommandBufferCount,
    totalCommandBufferCount: checkpoint.totalCommandBufferCount,
    snapshotCopyCount: checkpoint.snapshotCopyCount,
    snapshotExtraCommandBufferCount:
      checkpoint.snapshotExtraCommandBufferCount,
    snapshotExtraQueueDrainCount: checkpoint.snapshotExtraQueueDrainCount,
    quadQueryRoutes: checkpoint.attentionRouteProfile?.quadQueryRoutes ?? 0,
    query8SlidingRoutes:
      checkpoint.attentionRouteProfile?.query8SlidingRoutes ?? 96,
    query8CrossRoutes:
      checkpoint.attentionRouteProfile?.query8CrossRoutes ?? 192,
    unintendedQuadQueryRoutes:
      checkpoint.attentionRouteProfile?.unintendedQuadQueryRoutes ?? 0,
    actualLayerComparedElements:
      checkpoint.actualLayerIdentity?.totalComparedElements ?? 0,
    actualLayerMismatchCount:
      checkpoint.actualLayerIdentity?.totalMismatchCount ?? 0,
    actualLayerNonFiniteCount:
      checkpoint.actualLayerIdentity?.totalNonFiniteCount ?? 0,
    actualLayerCanaryCount:
      checkpoint.actualLayerIdentity?.totalCanaryCount ?? 0,
  });
}

function createOpt0067DitCheckpoint(
  denoised: AceDitGpuBackendResult,
  profile: AceOpt0067DitCommandProfile,
  stageTimings: readonly AceStageTiming[],
  attentionRuntimeProfile:
    AceWorkerConfiguration["ditAttentionRuntimeProfile"],
  requireActualLayerIdentity: boolean,
): AceOpt0067DitCheckpoint {
  const result = denoised.finalLatent;
  const quad = attentionRuntimeProfile ===
    ACE_OPT_0062_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE;
  const routeProfile = denoised.attentionRouteProfile;
  const rawIdentity = denoised.opt0062AttentionIdentity;
  if (
    denoised.shape.batch !== 1 || denoised.shape.latentFrames !== 4_500 ||
    denoised.shape.channels !== 64 || result.length !== 288_000 ||
    result.byteLength !== 1_152_000 ||
    denoised.commandBuffersSubmitted !== 342 || denoised.queueDrains !== 342 ||
    denoised.cooperativeIdleMs !== 341 || denoised.completedEvaluations !== 1 ||
    denoised.evaluationLatents !== undefined ||
    denoised.denseRouteProfile !== undefined ||
    profile.schema !== "ace-dit-opt0067-evaluation0-command-profile-v1" ||
    profile.graphCommandBufferCount !== 341 ||
    profile.readbackCommandBufferCount !== 1 ||
    profile.totalCommandBufferCount !== 342 ||
    profile.graphQueueDrainCount !== 341 || profile.totalQueueDrainCount !== 342 ||
    profile.evaluationCommandBufferCount !== 316 ||
    profile.precompute.commandBufferCount !== 25 ||
    profile.evaluation.commandBufferCount !== 316 ||
    profile.timings.length !== 341 ||
    (quad !== (routeProfile !== undefined)) ||
    (requireActualLayerIdentity !== (rawIdentity !== undefined))
  ) {
    throw new Error("OPT-0067 evaluation-0 checkpoint topology changed");
  }
  if (
    routeProfile !== undefined &&
    (routeProfile.runtimeProfileId !==
        ACE_OPT_0062_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE ||
      routeProfile.kernelSetId !==
        ACE_OPT_0062_DIT_QUAD_QUERY_ATTENTION_KERNEL_SET_ID ||
      routeProfile.quadQueryRoutes !== 96 ||
      routeProfile.query8SlidingRoutes !== 96 ||
      routeProfile.query8CrossRoutes !== 192 ||
      routeProfile.query8OtherRoutes !== 0 ||
      routeProfile.unintendedQuadQueryRoutes !== 0 ||
      routeProfile.uniqueQuadQueryRouteIds.length !== 96)
  ) {
    throw new Error("OPT-0067 inherited quad route table changed");
  }
  const ownerMembers = profile.descriptorTable.descriptors.flatMap(
    ({ members }) => members,
  );
  const quadDescriptorCount = ownerMembers.filter(({ backend }) =>
    backend === "opt-0062-fixed32-quad-query32-full-self"
  ).length;
  const query8SelfFullDescriptorCount = ownerMembers.filter(
    ({ family, backend }) =>
      family === "self-full" && backend === "fixed32-subgroup-query8",
  ).length;
  if (
    quad
      ? quadDescriptorCount !== 480 || query8SelfFullDescriptorCount !== 0
      : quadDescriptorCount !== 0 || query8SelfFullDescriptorCount !== 480
  ) {
    throw new Error("OPT-0067 command owner attribution changed");
  }
  let actualLayerIdentity: AceOpt0067AttentionIdentityResult | undefined;
  if (rawIdentity !== undefined) {
    const activeRoutes = rawIdentity.routes.slice(0, 12);
    const inactiveRoutes = rawIdentity.routes.slice(12);
    if (
      rawIdentity.routeCount !== 96 || rawIdentity.routes.length !== 96 ||
      rawIdentity.outputElementsPerRoute !== 4_608_000 ||
      rawIdentity.totalComparedElements !== 55_296_000 ||
      rawIdentity.totalMismatchCount !== 0 ||
      rawIdentity.totalNonFiniteCount !== 0 ||
      rawIdentity.totalCanaryCount !== 0 || rawIdentity.copyCount !== 1 ||
      rawIdentity.extraCommandBufferCount !== 0 ||
      rawIdentity.extraQueueDrainCount !== 0 ||
      activeRoutes.some((route, routeIndex) =>
        route.routeIndex !== routeIndex || route.comparedElements !== 4_608_000 ||
        route.label !==
          `ace-dit-eval-0-layer-${1 + 2 * routeIndex}-self-full-attention` ||
        route.mismatchCount !== 0 || route.query8NonFiniteCount !== 0 ||
        route.quadNonFiniteCount !== 0 || route.query8CanaryCount !== 0 ||
        route.quadCanaryCount !== 0
      ) ||
      inactiveRoutes.some((route) =>
        route.comparedElements !== 0 || route.mismatchCount !== 0 ||
        route.query8NonFiniteCount !== 0 || route.quadNonFiniteCount !== 0 ||
        route.query8CanaryCount !== 0 || route.quadCanaryCount !== 0
      )
    ) {
      throw new Error("OPT-0067 evaluation-0 actual-layer identity failed");
    }
    actualLayerIdentity = Object.freeze({
      schema: "ace-opt-0067-evaluation0-actual-layer-raw-u32-identity-v1",
      routeCount: 12,
      outputElementsPerRoute: 4_608_000,
      totalComparedElements: 55_296_000,
      totalMismatchCount: 0,
      totalNonFiniteCount: 0,
      totalCanaryCount: 0,
      copyCount: 1,
      extraCommandBufferCount: 0,
      extraQueueDrainCount: 0,
      inactiveFutureRouteCount: 84,
      routes: Object.freeze(activeRoutes),
    });
  }
  let nonFinite = 0;
  let nonzero = 0;
  let maxAbs = 0;
  for (const value of result) {
    if (!Number.isFinite(value)) nonFinite += 1;
    else {
      if (value !== 0) nonzero += 1;
      maxAbs = Math.max(maxAbs, Math.abs(value));
    }
  }
  if (nonFinite !== 0 || nonzero === 0 || !(maxAbs > 0)) {
    throw new Error("OPT-0067 evaluation result is non-finite or all zero");
  }
  return Object.freeze({
    schema: "ace-dit-opt0067-m2250-evaluation0-checkpoint-v1",
    evaluation: 0,
    result,
    resultByteLength: 1_152_000,
    resultElementCount: 288_000,
    resultSha256: sha256F32Le(result),
    resultNonFiniteCount: 0,
    resultNonzeroCount: nonzero,
    resultMaxAbs: maxAbs,
    ...(routeProfile === undefined ? {} : { attentionRouteProfile: routeProfile }),
    ...(actualLayerIdentity === undefined ? {} : { actualLayerIdentity }),
    profile,
    stageTimings: Object.freeze([...stageTimings]),
    graphCommandBufferCount: 341,
    readbackCommandBufferCount: 1,
    totalCommandBufferCount: 342,
    completedEvaluations: 1,
    evaluationResultExtraCommandBufferCount: 0,
    evaluationResultExtraQueueDrainCount: 0,
  });
}

function summarizeOpt0067DitCheckpoint(
  checkpoint: AceOpt0067DitCheckpoint,
): NonNullable<AceDiagnostic["details"]> {
  return Object.freeze({
    schema: "ace-opt-0067-evaluation0-receipt-v1",
    checkpointSchema: checkpoint.schema,
    evaluation: checkpoint.evaluation,
    resultSha256: checkpoint.resultSha256,
    resultElementCount: checkpoint.resultElementCount,
    graphCommandBufferCount: checkpoint.graphCommandBufferCount,
    readbackCommandBufferCount: checkpoint.readbackCommandBufferCount,
    totalCommandBufferCount: checkpoint.totalCommandBufferCount,
    completedEvaluations: checkpoint.completedEvaluations,
    quadQueryRoutesExecuted: checkpoint.attentionRouteProfile === undefined
      ? 0
      : 12,
    query8SlidingRoutesExecuted: 12,
    query8CrossRoutesExecuted: 24,
    actualLayerComparedElements:
      checkpoint.actualLayerIdentity?.totalComparedElements ?? 0,
    actualLayerMismatchCount:
      checkpoint.actualLayerIdentity?.totalMismatchCount ?? 0,
    identityExtraCommandBufferCount:
      checkpoint.actualLayerIdentity?.extraCommandBufferCount ?? 0,
    identityExtraQueueDrainCount:
      checkpoint.actualLayerIdentity?.extraQueueDrainCount ?? 0,
    evaluationResultExtraCommandBufferCount:
      checkpoint.evaluationResultExtraCommandBufferCount,
    evaluationResultExtraQueueDrainCount:
      checkpoint.evaluationResultExtraQueueDrainCount,
  });
}

function createOpt0080DitCheckpoint(
  denoised: AceDitGpuBackendResult,
  profile: AceOpt0080DitCommandProfile,
  schedulingProfile: AceDitOpt0080SchedulingProfile,
  denseRuntimeProfile: AceWorkerConfiguration["ditDensePackage"]["runtimeProfile"],
  attentionRuntimeProfile:
    AceWorkerConfiguration["ditAttentionRuntimeProfile"],
): AceOpt0080DitCheckpoint {
  const result = denoised.finalLatent;
  const topology = profile.topology;
  const candidate = schedulingProfile === "opt-0080-depth2-epoch4";
  const expectedGraphDrains = candidate ? 86 : 341;
  const expectedGraphIdleTurns = candidate ? 85 : 340;
  const expectedTotalDrains = candidate ? 87 : 342;
  const expectedTotalIdleMs = candidate ? 86 : 341;
  const expectedMaximumOutstanding = candidate ? 2 : 1;
  const routeProfile = denoised.attentionRouteProfile;
  const graphEpochWallSumMs = topology.graphCompletionEpochs.reduce(
    (sum, epoch) => sum + epoch.submitThroughTrueDrainMs,
    0,
  );
  const wallToleranceMs = OPT_0080_WALL_RECONCILIATION_TOLERANCE_MS;
  if (
    denseRuntimeProfile !== ACE_OPT_0009_DIT_DENSE_RUNTIME_PROFILE ||
    attentionRuntimeProfile !==
      ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE ||
    denoised.shape.batch !== 1 || denoised.shape.latentFrames !== 4_500 ||
    denoised.shape.channels !== 64 || result.length !== 288_000 ||
    result.byteLength !== 1_152_000 ||
    result.byteOffset !== 0 || result.buffer.byteLength !== result.byteLength ||
    denoised.commandBuffersSubmitted !== 342 ||
    denoised.queueDrains !== expectedTotalDrains ||
    denoised.cooperativeIdleMs !== expectedTotalIdleMs ||
    denoised.completedEvaluations !== 1 ||
    denoised.evaluationLatents !== undefined ||
    denoised.evaluation0Velocity !== undefined ||
    denoised.denseRouteProfile !== undefined ||
    denoised.opt0062AttentionIdentity !== undefined ||
    profile.schema !== "ace-dit-opt0080-evaluation0-command-profile-v1" ||
    profile.schedulingProfile !== schedulingProfile ||
    topology.schedulingProfile !== schedulingProfile ||
    profile.descriptorTable.memberCount !== 6_833 ||
    profile.descriptorTable.descriptors.length !== 2_553 ||
    profile.descriptorTable.descriptors.slice(0, 25).some((descriptor) =>
      descriptor.evaluation !== null
    ) ||
    profile.descriptorTable.descriptors.slice(25, 341).some((descriptor) =>
      descriptor.evaluation !== 0
    ) ||
    profile.descriptorTable.descriptors[341]?.evaluation !== 1 ||
    [
      profile.precomputeWallMs,
      profile.evaluationWallMs,
      profile.graphWallMs,
      profile.graphToReadbackObservedIdleMs,
      profile.readbackSubmitThroughCompletionFenceMs,
      profile.readbackMapDetachMs,
      profile.backendWallMs,
    ].some((value) => !Number.isFinite(value) || value < 0) ||
    Math.abs(
      profile.precomputeWallMs + profile.evaluationWallMs -
        profile.graphWallMs,
    ) > wallToleranceMs ||
    graphEpochWallSumMs - profile.graphWallMs > wallToleranceMs ||
    topology.descriptorTableMemberCount !== 6_833 ||
    topology.graphCommandBufferCount !== 341 ||
    topology.readbackCommandBufferCount !== 1 ||
    topology.totalCommandBufferCount !== 342 ||
    topology.commandBuffersSubmitted !== 342 ||
    topology.completionFenceRequestedCount !== 342 ||
    topology.completionFenceSettledCount !== 342 ||
    topology.completionFenceRejectedCount !== 0 ||
    topology.graphTrueQueueDrainCount !== expectedGraphDrains ||
    topology.totalTrueQueueDrainCount !== expectedTotalDrains ||
    topology.graphCompletionEpochCount !== expectedGraphDrains ||
    topology.graphCooperativeIdleTurns !== expectedGraphIdleTurns ||
    topology.totalCooperativeIdleTurns !== expectedTotalIdleMs ||
    topology.graphRequestedCooperativeIdleMs !== expectedGraphIdleTurns ||
    topology.totalRequestedCooperativeIdleMs !== expectedTotalIdleMs ||
    topology.maximumOutstandingCommandBuffers !== expectedMaximumOutstanding ||
    topology.maximumPendingDescriptorCount !== expectedMaximumOutstanding ||
    topology.pendingDescriptorCountAfterCleanup !== 0 ||
    topology.graphToReadbackRequestedIdleMs !== 1 ||
    topology.readbackSubmitThroughCompletionFenceMs !==
      profile.readbackSubmitThroughCompletionFenceMs ||
    topology.submitThroughCompletionFenceMs.length !== 341 ||
    topology.submitThroughCompletionFenceMs.some((value) =>
      !Number.isFinite(value) || value < 0
    ) ||
    !hasExactOpt0080CompletionEpochs(
      topology.graphCompletionEpochs,
      candidate,
    ) ||
    routeProfile === undefined ||
    routeProfile.runtimeProfileId !==
      ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE ||
    routeProfile.schema !== "ace-opt-0070-production-attention-routes-v1" ||
    routeProfile.kernelSetId !==
      ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_KERNEL_SET_ID ||
    routeProfile.ownerMode !== "exact-m2250-opt0062-quad" ||
    routeProfile.expectedQueryTokens !== 2_250 ||
    routeProfile.expectedConditionTokens !== 98 ||
    routeProfile.routeCount !== 384 || routeProfile.quadQueryRoutes !== 96 ||
    routeProfile.query8FullSelfRoutes !== 0 ||
    routeProfile.query8SlidingRoutes !== 96 ||
    routeProfile.query8CrossRoutes !== 192 ||
    routeProfile.query8OtherRoutes !== 0 ||
    routeProfile.unintendedQuadQueryRoutes !== 0 ||
    routeProfile.fullSelfRouteIds.length !== 96 ||
    routeProfile.slidingSelfRouteIds.length !== 96 ||
    routeProfile.crossRouteIds.length !== 192
  ) {
    throw new Error("OPT-0080 evaluation-0 checkpoint topology changed");
  }
  let nonFinite = 0;
  let nonzero = 0;
  let maxAbs = 0;
  for (const value of result) {
    if (!Number.isFinite(value)) nonFinite += 1;
    else {
      if (value !== 0) nonzero += 1;
      maxAbs = Math.max(maxAbs, Math.abs(value));
    }
  }
  if (nonFinite !== 0 || nonzero === 0 || !(maxAbs > 0)) {
    throw new Error("OPT-0080 evaluation result is non-finite or all zero");
  }
  return Object.freeze({
    schema: "ace-dit-opt0080-m2250-evaluation0-checkpoint-v1",
    schedulingProfile,
    evaluation: 0 as const,
    result,
    resultByteLength: 1_152_000 as const,
    resultElementCount: 288_000 as const,
    resultSha256: sha256F32Le(result),
    resultNonFiniteCount: 0 as const,
    resultNonzeroCount: nonzero,
    resultMaxAbs: maxAbs,
    profile,
    graphCommandBufferCount: 341 as const,
    readbackCommandBufferCount: 1 as const,
    totalCommandBufferCount: 342 as const,
    completedEvaluations: 1 as const,
    uncapturedWebGpuErrorCount: 0 as const,
    deviceLost: false as const,
  });
}

function hasExactOpt0080CompletionEpochs(
  epochs: AceOpt0080DitCommandProfile["topology"]["graphCompletionEpochs"],
  candidate: boolean,
): boolean {
  const epochSize = candidate ? 4 : 1;
  const expectedCount = candidate ? 86 : 341;
  if (epochs.length !== expectedCount) return false;
  let epochIndex = 0;
  let firstCommandBufferIndex = 0;
  for (const [phaseIndex, commandBufferCount] of [25, 316].entries()) {
    const phaseEnd = firstCommandBufferIndex + commandBufferCount;
    while (firstCommandBufferIndex < phaseEnd) {
      const count = Math.min(epochSize, phaseEnd - firstCommandBufferIndex);
      const epoch = epochs[epochIndex];
      if (
        epoch === undefined || epoch.completionEpochIndex !== epochIndex ||
        epoch.phaseIndex !== phaseIndex ||
        epoch.firstCommandBufferIndex !== firstCommandBufferIndex ||
        epoch.lastCommandBufferIndex !== firstCommandBufferIndex + count - 1 ||
        epoch.commandBufferCount !== count ||
        !Number.isFinite(epoch.submitThroughTrueDrainMs) ||
        epoch.submitThroughTrueDrainMs < 0
      ) return false;
      epochIndex += 1;
      firstCommandBufferIndex += count;
    }
  }
  return epochIndex === expectedCount && firstCommandBufferIndex === 341;
}

function summarizeOpt0080DitCheckpoint(
  checkpoint: AceOpt0080DitCheckpoint,
): NonNullable<AceDiagnostic["details"]> {
  return Object.freeze({
    schema: "ace-opt-0080-evaluation0-receipt-v1",
    checkpointSchema: checkpoint.schema,
    schedulingProfile: checkpoint.schedulingProfile,
    evaluation: checkpoint.evaluation,
    resultSha256: checkpoint.resultSha256,
    resultElementCount: checkpoint.resultElementCount,
    descriptorTableSha256: checkpoint.profile.descriptorTable.sha256,
    descriptorTableMemberCount:
      checkpoint.profile.descriptorTable.memberCount,
    graphCommandBufferCount: checkpoint.graphCommandBufferCount,
    readbackCommandBufferCount: checkpoint.readbackCommandBufferCount,
    totalCommandBufferCount: checkpoint.totalCommandBufferCount,
    graphTrueQueueDrainCount:
      checkpoint.profile.topology.graphTrueQueueDrainCount,
    totalTrueQueueDrainCount:
      checkpoint.profile.topology.totalTrueQueueDrainCount,
    graphCompletionEpochCount:
      checkpoint.profile.topology.graphCompletionEpochCount,
    maximumOutstandingCommandBuffers:
      checkpoint.profile.topology.maximumOutstandingCommandBuffers,
    maximumPendingDescriptorCount:
      checkpoint.profile.topology.maximumPendingDescriptorCount,
    identityExtraCommandBufferCount: 0,
    identityExtraQueueDrainCount: 0,
  });
}

function createOpt0080FullDitCheckpoint(
  denoised: AceDitGpuBackendResult,
  profile: AceOpt0080FullDitCommandProfile,
  run: AceOpt0080FullDitRunOptions,
  denseRuntimeProfile: AceWorkerConfiguration["ditDensePackage"]["runtimeProfile"],
  attentionRuntimeProfile:
    AceWorkerConfiguration["ditAttentionRuntimeProfile"],
): AceOpt0080FullDitCheckpoint {
  const finalLatent = denoised.finalLatent;
  const topology = profile.topology;
  const candidate = run.schedulingProfile === "opt-0080-depth2-epoch4";
  const captureEvaluationTaps = run.captureEvaluationTaps === true;
  const expectedGraphDrains = candidate ? 639 : 2_553;
  const expectedGraphIdleTurns = candidate ? 638 : 2_552;
  const expectedTotalDrains = candidate ? 640 : 2_554;
  const expectedTotalIdleTurns = candidate ? 639 : 2_553;
  const expectedMaximumOutstanding = candidate ? 2 : 1;
  const routeProfile = denoised.attentionRouteProfile;
  const phaseWallSumMs = profile.precomputeWallMs +
    profile.evaluationWallMs.reduce((sum, wall) => sum + wall, 0);
  const graphEpochWallSumMs = topology.graphCompletionEpochs.reduce(
    (sum, epoch) => sum + epoch.submitThroughTrueDrainMs,
    0,
  );
  const wallToleranceMs = OPT_0080_WALL_RECONCILIATION_TOLERANCE_MS;
  if (
    denseRuntimeProfile !== ACE_OPT_0009_DIT_DENSE_RUNTIME_PROFILE ||
    attentionRuntimeProfile !==
      ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE ||
    denoised.shape.batch !== 1 || denoised.shape.latentFrames !== 4_500 ||
    denoised.shape.channels !== 64 || finalLatent.length !== 288_000 ||
    finalLatent.byteLength !== 1_152_000 || finalLatent.byteOffset !== 0 ||
    finalLatent.buffer.byteLength !== finalLatent.byteLength ||
    denoised.commandBuffersSubmitted !== 2_554 ||
    denoised.queueDrains !== expectedTotalDrains ||
    denoised.cooperativeIdleMs !== expectedTotalIdleTurns ||
    denoised.completedEvaluations !== 8 ||
    (denoised.evaluationLatents !== undefined) !== captureEvaluationTaps ||
    denoised.evaluation0Velocity !== undefined ||
    denoised.denseRouteProfile !== undefined ||
    denoised.opt0062AttentionIdentity !== undefined ||
    profile.schema !== "ace-dit-opt0080-full-command-profile-v1" ||
    profile.schedulingProfile !== run.schedulingProfile ||
    topology.schedulingProfile !== run.schedulingProfile ||
    profile.descriptorTable.memberCount !== 6_833 ||
    profile.descriptorTable.descriptors.length !== 2_553 ||
    profile.evaluationWallMs.length !== 8 ||
    [
      profile.precomputeWallMs,
      ...profile.evaluationWallMs,
      profile.graphWallMs,
      profile.graphToReadbackObservedIdleMs,
      profile.readbackSubmitThroughCompletionFenceMs,
      profile.readbackMapDetachMs,
      profile.backendWallMs,
    ].some((value) => !Number.isFinite(value) || value < 0) ||
    Math.abs(phaseWallSumMs - profile.graphWallMs) > wallToleranceMs ||
    graphEpochWallSumMs - profile.graphWallMs > wallToleranceMs ||
    topology.descriptorTableMemberCount !== 6_833 ||
    topology.graphCommandBufferCount !== 2_553 ||
    topology.readbackCommandBufferCount !== 1 ||
    topology.totalCommandBufferCount !== 2_554 ||
    topology.commandBuffersSubmitted !== 2_554 ||
    topology.completionFenceRequestedCount !== 2_554 ||
    topology.completionFenceSettledCount !== 2_554 ||
    topology.completionFenceRejectedCount !== 0 ||
    topology.graphTrueQueueDrainCount !== expectedGraphDrains ||
    topology.totalTrueQueueDrainCount !== expectedTotalDrains ||
    topology.graphCompletionEpochCount !== expectedGraphDrains ||
    topology.graphCooperativeIdleTurns !== expectedGraphIdleTurns ||
    topology.totalCooperativeIdleTurns !== expectedTotalIdleTurns ||
    topology.graphRequestedCooperativeIdleMs !== expectedGraphIdleTurns ||
    topology.totalRequestedCooperativeIdleMs !== expectedTotalIdleTurns ||
    topology.maximumOutstandingCommandBuffers !== expectedMaximumOutstanding ||
    topology.maximumPendingDescriptorCount !== expectedMaximumOutstanding ||
    topology.pendingDescriptorCountAfterCleanup !== 0 ||
    topology.graphToReadbackRequestedIdleMs !== 1 ||
    topology.readbackSubmitThroughCompletionFenceMs !==
      profile.readbackSubmitThroughCompletionFenceMs ||
    topology.submitThroughCompletionFenceMs.length !== 2_553 ||
    topology.submitThroughCompletionFenceMs.some((value) =>
      !Number.isFinite(value) || value < 0
    ) ||
    !hasExactOpt0080FullCompletionEpochs(
      topology.graphCompletionEpochs,
      candidate,
    ) ||
    !hasExactOpt0080FullDescriptorPhases(profile) ||
    routeProfile === undefined ||
    routeProfile.runtimeProfileId !==
      ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE ||
    routeProfile.schema !== "ace-opt-0070-production-attention-routes-v1" ||
    routeProfile.kernelSetId !==
      ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_KERNEL_SET_ID ||
    routeProfile.ownerMode !== "exact-m2250-opt0062-quad" ||
    routeProfile.expectedQueryTokens !== 2_250 ||
    routeProfile.expectedConditionTokens !== 98 ||
    routeProfile.routeCount !== 384 || routeProfile.quadQueryRoutes !== 96 ||
    routeProfile.query8FullSelfRoutes !== 0 ||
    routeProfile.query8SlidingRoutes !== 96 ||
    routeProfile.query8CrossRoutes !== 192 ||
    routeProfile.query8OtherRoutes !== 0 ||
    routeProfile.unintendedQuadQueryRoutes !== 0 ||
    routeProfile.fullSelfRouteIds.length !== 96 ||
    routeProfile.slidingSelfRouteIds.length !== 96 ||
    routeProfile.crossRouteIds.length !== 192
  ) {
    throw new Error("OPT-0080 full checkpoint topology changed");
  }

  const finalStats = analyzeOpt0080FullTensor(finalLatent, "final latent");
  const evaluationTaps = denoised.evaluationLatents?.map(
    (result, evaluation): AceOpt0080FullDitEvaluationTap => {
      if (
        evaluation < 0 || evaluation > 7 ||
        result.length !== 288_000 || result.byteLength !== 1_152_000 ||
        result.byteOffset !== 0 || result.buffer.byteLength !== result.byteLength
      ) {
        throw new Error(`OPT-0080 evaluation tap ${evaluation} shape changed`);
      }
      const stats = analyzeOpt0080FullTensor(
        result,
        `evaluation tap ${evaluation}`,
      );
      return Object.freeze({
        evaluation: evaluation as AceOpt0080FullEvaluationIndex,
        result,
        rawU32: new Uint32Array(result.buffer),
        sha256: sha256F32Le(result),
        nonFiniteCount: 0 as const,
        nonzeroCount: stats.nonzeroCount,
        maxAbs: stats.maxAbs,
      });
    },
  );
  if (
    captureEvaluationTaps && evaluationTaps?.length !== 8 ||
    !captureEvaluationTaps && evaluationTaps !== undefined
  ) {
    throw new Error("OPT-0080 full evaluation taps did not match capture mode");
  }
  return Object.freeze({
    schema: "ace-dit-opt0080-m2250-full-checkpoint-v1",
    schedulingProfile: run.schedulingProfile,
    captureEvaluationTaps,
    finalLatent,
    finalLatentRawU32: new Uint32Array(finalLatent.buffer),
    finalLatentSha256: sha256F32Le(finalLatent),
    finalLatentNonFiniteCount: 0 as const,
    finalLatentNonzeroCount: finalStats.nonzeroCount,
    finalLatentMaxAbs: finalStats.maxAbs,
    ...(evaluationTaps === undefined
      ? {}
      : { evaluationTaps: Object.freeze(evaluationTaps) }),
    evaluationTapInCommandCopyCount: captureEvaluationTaps ? 8 as const : 0 as const,
    evaluationTapExtraCommandBufferCount: 0 as const,
    evaluationTapExtraQueueDrainCount: 0 as const,
    profile,
    graphCommandBufferCount: 2_553 as const,
    readbackCommandBufferCount: 1 as const,
    totalCommandBufferCount: 2_554 as const,
    completedEvaluations: 8 as const,
  });
}

function analyzeOpt0080FullTensor(
  values: Float32Array<ArrayBuffer>,
  label: string,
): Readonly<{ nonzeroCount: number; maxAbs: number }> {
  let nonFiniteCount = 0;
  let nonzeroCount = 0;
  let maxAbs = 0;
  for (const value of values) {
    if (!Number.isFinite(value)) nonFiniteCount += 1;
    else {
      if (value !== 0) nonzeroCount += 1;
      maxAbs = Math.max(maxAbs, Math.abs(value));
    }
  }
  if (nonFiniteCount !== 0 || nonzeroCount === 0 || !(maxAbs > 0)) {
    throw new Error(`OPT-0080 ${label} is non-finite or all zero`);
  }
  return Object.freeze({ nonzeroCount, maxAbs });
}

function hasExactOpt0080FullDescriptorPhases(
  profile: AceOpt0080FullDitCommandProfile,
): boolean {
  const phaseCounts = [25, 316, 316, 316, 316, 316, 316, 316, 316];
  let first = 0;
  for (let phaseIndex = 0; phaseIndex < phaseCounts.length; phaseIndex += 1) {
    const last = first + phaseCounts[phaseIndex]!;
    const expectedEvaluation = phaseIndex === 0 ? null : phaseIndex - 1;
    if (profile.descriptorTable.descriptors.slice(first, last).some(
      (descriptor) => descriptor.evaluation !== expectedEvaluation,
    )) return false;
    first = last;
  }
  return first === 2_553;
}

function hasExactOpt0080FullCompletionEpochs(
  epochs: AceOpt0080FullDitCommandProfile["topology"]["graphCompletionEpochs"],
  candidate: boolean,
): boolean {
  const epochSize = candidate ? 4 : 1;
  const expectedCount = candidate ? 639 : 2_553;
  if (epochs.length !== expectedCount) return false;
  let epochIndex = 0;
  let firstCommandBufferIndex = 0;
  const phases = [25, 316, 316, 316, 316, 316, 316, 316, 316];
  for (const [phaseIndex, commandBufferCount] of phases.entries()) {
    const phaseEnd = firstCommandBufferIndex + commandBufferCount;
    while (firstCommandBufferIndex < phaseEnd) {
      const count = Math.min(epochSize, phaseEnd - firstCommandBufferIndex);
      const epoch = epochs[epochIndex];
      if (
        epoch === undefined || epoch.completionEpochIndex !== epochIndex ||
        epoch.phaseIndex !== phaseIndex ||
        epoch.firstCommandBufferIndex !== firstCommandBufferIndex ||
        epoch.lastCommandBufferIndex !== firstCommandBufferIndex + count - 1 ||
        epoch.commandBufferCount !== count ||
        !Number.isFinite(epoch.submitThroughTrueDrainMs) ||
        epoch.submitThroughTrueDrainMs < 0
      ) return false;
      epochIndex += 1;
      firstCommandBufferIndex += count;
    }
  }
  return epochIndex === expectedCount && firstCommandBufferIndex === 2_553;
}

function summarizeOpt0080FullDitCheckpoint(
  checkpoint: AceOpt0080FullDitCheckpoint,
): NonNullable<AceDiagnostic["details"]> {
  return Object.freeze({
    schema: "ace-opt-0080-full-receipt-v1",
    checkpointSchema: checkpoint.schema,
    schedulingProfile: checkpoint.schedulingProfile,
    captureEvaluationTaps: checkpoint.captureEvaluationTaps,
    evaluationTapInCommandCopyCount:
      checkpoint.evaluationTapInCommandCopyCount,
    evaluationTapExtraCommandBufferCount:
      checkpoint.evaluationTapExtraCommandBufferCount,
    evaluationTapExtraQueueDrainCount:
      checkpoint.evaluationTapExtraQueueDrainCount,
    finalLatentSha256: checkpoint.finalLatentSha256,
    ...(checkpoint.evaluationTaps === undefined
      ? {}
      : {
          evaluationTapSha256:
            Object.freeze(checkpoint.evaluationTaps.map((tap) => tap.sha256)),
        }),
    descriptorTableSha256: checkpoint.profile.descriptorTable.sha256,
    descriptorTableMemberCount: checkpoint.profile.descriptorTable.memberCount,
    graphCommandBufferCount: checkpoint.graphCommandBufferCount,
    readbackCommandBufferCount: checkpoint.readbackCommandBufferCount,
    totalCommandBufferCount: checkpoint.totalCommandBufferCount,
    graphTrueQueueDrainCount:
      checkpoint.profile.topology.graphTrueQueueDrainCount,
    totalTrueQueueDrainCount:
      checkpoint.profile.topology.totalTrueQueueDrainCount,
    graphCompletionEpochCount:
      checkpoint.profile.topology.graphCompletionEpochCount,
    maximumOutstandingCommandBuffers:
      checkpoint.profile.topology.maximumOutstandingCommandBuffers,
    maximumPendingDescriptorCount:
      checkpoint.profile.topology.maximumPendingDescriptorCount,
  });
}

function createOpt0034DitCheckpoint(
  denoised: AceDitGpuBackendResult,
  profile: AceOpt0034DitSchedulingProfile,
  stageTimings: readonly AceStageTiming[],
): AceOpt0034DitCheckpoint {
  const finalLatent = denoised.finalLatent;
  const expectedGraphCommandBuffers = Math.ceil(
    2_553 / profile.physicalQuantaPerCommandBuffer,
  );
  if (
    denoised.shape.batch !== 1 ||
    denoised.shape.latentFrames !== 4_500 ||
    denoised.shape.channels !== 64 ||
    finalLatent.length !== 288_000 ||
    finalLatent.byteLength !== 1_152_000 ||
    profile.physicalGraphQuantumCount !== 2_553 ||
    profile.graphCommandBufferCount !== expectedGraphCommandBuffers ||
    profile.readbackCommandBufferCount !== 1 ||
    profile.totalCommandBufferCount !== expectedGraphCommandBuffers + 1 ||
    profile.graphQueueDrainCount !== expectedGraphCommandBuffers ||
    profile.totalQueueDrainCount !== expectedGraphCommandBuffers + 1 ||
    profile.graphRequestedIdleMs !== expectedGraphCommandBuffers - 1 ||
    profile.graphToReadbackRequestedIdleMs !== 1 ||
    profile.batches.length !== expectedGraphCommandBuffers ||
    denoised.commandBuffersSubmitted !== expectedGraphCommandBuffers + 1 ||
    denoised.queueDrains !== expectedGraphCommandBuffers + 1 ||
    denoised.cooperativeIdleMs !== expectedGraphCommandBuffers ||
    denoised.completedEvaluations !== 8
  ) {
    throw new Error("OPT-0034 final DiT checkpoint topology changed");
  }
  let nonFinite = 0;
  let nonzero = 0;
  let maxAbs = 0;
  for (const value of finalLatent) {
    if (!Number.isFinite(value)) {
      nonFinite += 1;
      continue;
    }
    if (value !== 0) nonzero += 1;
    maxAbs = Math.max(maxAbs, Math.abs(value));
  }
  if (nonFinite !== 0 || nonzero === 0 || !(maxAbs > 0)) {
    throw new Error("OPT-0034 final latent is non-finite or all zero");
  }
  return Object.freeze({
    schema: "ace-dit-opt0034-m2250-checkpoint-v1",
    finalLatent,
    finalLatentByteLength: 1_152_000,
    finalLatentElementCount: 288_000,
    finalLatentSha256: sha256F32Le(finalLatent),
    finalLatentNonFiniteCount: 0 as const,
    finalLatentNonzeroCount: nonzero,
    finalLatentMaxAbs: maxAbs,
    profile,
    stageTimings: Object.freeze([...stageTimings]),
  });
}

function summarizeOpt0034DitCheckpoint(
  checkpoint: AceOpt0034DitCheckpoint,
): NonNullable<AceDiagnostic["details"]> {
  const { profile } = checkpoint;
  return Object.freeze({
    schema: "ace-dit-opt0034-scheduling-receipt-v1",
    checkpointSchema: checkpoint.schema,
    schedulingProfileSchema: profile.schema,
    physicalGraphQuantumCount: profile.physicalGraphQuantumCount,
    physicalQuantaPerCommandBuffer:
      profile.physicalQuantaPerCommandBuffer,
    graphCommandBufferCount: profile.graphCommandBufferCount,
    readbackCommandBufferCount: profile.readbackCommandBufferCount,
    totalCommandBufferCount: profile.totalCommandBufferCount,
    totalQueueDrainCount: profile.totalQueueDrainCount,
    graphRequestedIdleMs: profile.graphRequestedIdleMs,
    graphWallMs: profile.graphWallMs,
    backendWallMs: profile.backendWallMs,
    maximumPhysicalQuantaPerBatch:
      profile.maximumPhysicalQuantaPerBatch,
    maximumPrimitiveCountPerBatch: profile.maximumPrimitiveCountPerBatch,
    maximumBatchSubmitThroughDrainMs:
      profile.maximumBatchSubmitThroughDrainMs,
    descriptorTableSha256: profile.descriptorTableSha256,
    finalLatentSha256: checkpoint.finalLatentSha256,
  });
}

function summarizeOpt0018DitCheckpoint(
  checkpoint: AceOpt0018DitCheckpoint,
  denseRuntimeProfile:
    AceWorkerDitDensePackageConfiguration["runtimeProfile"],
): NonNullable<AceDiagnostic["details"]> {
  const { profile } = checkpoint;
  if (
    profile.timings.length !== profile.graphCommandBufferCount ||
    profile.descriptorTable.descriptors.length !==
      profile.graphCommandBufferCount
  ) {
    throw new Error("OPT-0018 checkpoint profile is incomplete");
  }
  const k4 = denseRuntimeProfile === ACE_OPT_0037_DIT_K4_RUNTIME_PROFILE;
  return Object.freeze({
    schema: "ace-dit-m2250-family-profile-receipt-v1",
    commandProfileSchema: profile.schema,
    checkpointSchema: checkpoint.schema,
    executionProfile: "reference-bf16-subgroups",
    gemmBackend: k4 ? "mixed-opt-0037-k4" : "mixed-opt-0009",
    denseRuntimeProfile,
    denseKernelSetId: k4
      ? ACE_OPT_0037_DIT_K4_KERNEL_SET_ID
      : ACE_OPT_0009_DIT_DENSE_KERNEL_SET_ID,
    attentionBackend: "fixed32-subgroup-query8",
    batch: 1,
    latentFrames: 4_500,
    tokens: 2_250,
    conditionTokens: 98,
    graphCommandBufferCount: profile.graphCommandBufferCount,
    readbackCommandBufferCount: profile.readbackCommandBufferCount,
    totalCommandBufferCount: profile.totalCommandBufferCount,
    descriptorTableSha256: profile.descriptorTable.sha256,
    descriptorTableSerializedBytes: profile.descriptorTable.serializedBytes,
    descriptorMemberCount: profile.descriptorTable.memberCount,
    descriptorPreparationMs: profile.descriptorTable.preparationMs,
    timingStorageBytes: profile.timingStorageBytes,
    descriptorRowsJson: JSON.stringify(
      profile.descriptorTable.descriptors.map((descriptor) => [
      descriptor.physicalIndex,
      descriptor.logicalIndex,
      descriptor.logicalKind,
      descriptor.commandId,
      descriptor.subquantumIndex,
      descriptor.subquantumCount,
      descriptor.evaluation ?? -1,
      descriptor.layer ?? -1,
      descriptor.family,
      descriptor.primitiveCount,
      descriptor.scheduledMultiplyAdds,
      descriptor.members.map((member) => [
        member.id,
        member.family,
        member.backend,
        member.kernel,
        member.scheduledMultiplyAdds,
      ]),
      ]),
    ),
    commandTimingTuplesJson: JSON.stringify(
      profile.timings.map((elapsedMs, physicalIndex) => [
      physicalIndex,
      profile.descriptorTable.descriptors[physicalIndex]!.family,
      profile.descriptorTable.descriptors[physicalIndex]!.evaluation ?? -1,
      profile.descriptorTable.descriptors[physicalIndex]!.layer ?? -1,
      elapsedMs,
      ]),
    ),
    graphSubmitThroughDrainMs: profile.graphSubmitThroughDrainMs,
    graphWallMs: profile.graphWallMs,
    graphRequestedIdleMs: profile.graphRequestedIdleMs,
    graphResidualMs: profile.graphResidualMs,
    graphToReadbackRequestedIdleMs:
      profile.graphToReadbackRequestedIdleMs,
    graphToReadbackObservedIdleMs: profile.graphToReadbackObservedIdleMs,
    readbackSubmitThroughDrainMs: profile.readbackSubmitThroughDrainMs,
    readbackMapDetachMs: profile.readbackMapDetachMs,
    backendWallMs: profile.backendWallMs,
    backendResidualMs: profile.backendResidualMs,
    familiesJson: JSON.stringify(profile.families),
    precomputeJson: JSON.stringify(profile.precompute),
    evaluationsJson: JSON.stringify(profile.evaluations),
    familyByBucketJson: JSON.stringify(profile.familyByBucket),
    slowestJson: JSON.stringify(profile.slowest),
    reconciliationToleranceMs: profile.reconciliationToleranceMs,
    finalLatentByteLength: checkpoint.finalLatentByteLength,
    finalLatentElementCount: checkpoint.finalLatentElementCount,
    finalLatentSha256: checkpoint.finalLatentSha256,
    finalLatentNonFiniteCount: checkpoint.finalLatentNonFiniteCount,
    finalLatentNonzeroCount: checkpoint.finalLatentNonzeroCount,
    finalLatentMaxAbs: checkpoint.finalLatentMaxAbs,
    stageTimingsJson: JSON.stringify(checkpoint.stageTimings),
    checkpointStopRequested: true,
  });
}

function summarizeVaeFamilyProfiles(
  profiles: readonly AceOpt0011Fp16VaeWindowFamilyProfile[],
  runtimeIdentity: AceVaePackageRuntimeIdentity,
): NonNullable<AceDiagnostic["details"]> {
  const ordered = [...profiles].sort((left, right) =>
    left.windowIndex - right.windowIndex
  );
  if (
    ordered.length === 0 ||
    ordered.some((profile, index) =>
      profile.windowIndex !== index ||
      profile.quantaPerCommandBuffer !==
        ordered[0]!.quantaPerCommandBuffer
    )
  ) {
    throw new Error("ACE FP16 VAE family profiles are incomplete or unordered");
  }
  const quantaPerCommandBuffer = ordered[0]!.quantaPerCommandBuffer;
  if (
    quantaPerCommandBuffer !==
      ACE_OPT_0011_VAE_FP16_WINDOW_QUANTA_PER_COMMAND_BUFFER &&
    quantaPerCommandBuffer !==
      ACE_OPT_0027_VAE_FP16_WINDOW_QUANTA_PER_COMMAND_BUFFER
  ) {
    throw new Error("ACE FP16 VAE family profile batch size changed");
  }
  const sum = (
    select: (profile: AceOpt0011Fp16VaeWindowFamilyProfile) => number,
  ): number => ordered.reduce((total, profile) => total + select(profile), 0);
  const decoderBatchCount = sum((profile) => profile.decoderBatchCount);
  const decoderSubmitThroughDrainMs = sum(
    (profile) => profile.decoderSubmitThroughDrainMs,
  );
  const homogeneousBatchCount = sum(
    (profile) => profile.homogeneousBatchCount,
  );
  const homogeneousSubmitThroughDrainMs = sum(
    (profile) => profile.homogeneousSubmitThroughDrainMs,
  );
  const details: Record<string, NonNullable<
    AceDiagnostic["details"]
  >[string]> = {
    schema: "ace-vae-fp16-family-profile-v1",
    runtimeProfile: runtimeIdentity.runtimeProfile,
    physicalRuntimeProfile: runtimeIdentity.role ===
        "opt-0072-rev7-production"
      ? runtimeIdentity.physicalRuntimeProfile
      : runtimeIdentity.runtimeProfile,
    kernelSetId: runtimeIdentity.kernelSetId,
    precisionMapSha256: runtimeIdentity.precisionMapSha256,
    windowCount: ordered.length,
    windowInputFrames: ordered.map((profile) => profile.inputFrames),
    quantaPerCommandBuffer,
    readbackExcluded: true,
    decoderBatchCount,
    decoderQuantumCount: sum((profile) => profile.decoderQuantumCount),
    decoderSubmitThroughDrainMs,
    homogeneousBatchCount,
    homogeneousQuantumCount: sum(
      (profile) => profile.homogeneousQuantumCount,
    ),
    homogeneousSubmitThroughDrainMs,
    homogeneousBatchCoverage: decoderBatchCount === 0
      ? 0
      : homogeneousBatchCount / decoderBatchCount,
    homogeneousTimeCoverage: decoderSubmitThroughDrainMs === 0
      ? 0
      : homogeneousSubmitThroughDrainMs / decoderSubmitThroughDrainMs,
    mixedBatchCount: sum((profile) => profile.mixedBatchCount),
    mixedQuantumCount: sum((profile) => profile.mixedQuantumCount),
    mixedSubmitThroughDrainMs: sum(
      (profile) => profile.mixedSubmitThroughDrainMs,
    ),
  };
  for (const [prefix, family] of [
    ["k7Conv1d", "k7-conv1d"],
    ["k1Conv1d", "k1-conv1d"],
    ["convTranspose1d", "conv-transpose1d"],
    ["snake", "snake"],
    ["add", "add"],
  ] as const satisfies readonly [string, AceOpt0011Fp16VaeProfileFamily][]) {
    details[`${prefix}BatchCount`] = sum(
      (profile) => profile.families[family].batchCount,
    );
    details[`${prefix}QuantumCount`] = sum(
      (profile) => profile.families[family].quantumCount,
    );
    details[`${prefix}SubmitThroughDrainMs`] = sum(
      (profile) => profile.families[family].submitThroughDrainMs,
    );
  }
  return Object.freeze(details);
}

async function settleWithoutMasking(
  promises: readonly (Promise<void> | undefined)[],
): Promise<void> {
  await Promise.allSettled(
    promises.filter((value): value is Promise<void> => value !== undefined),
  );
}

type Opt0064EventInput = Omit<
  AceOpt0064CaptureEvent,
  "schema" | "wallMs"
>;

async function captureOpt0064Async<T>(
  now: () => number,
  sink: AceOpt0064CaptureSink | undefined,
  scope: AceOpt0064CaptureEvent["scope"],
  category: AceOpt0064CaptureEvent["category"],
  operation: string,
  details: Readonly<Record<string, unknown>>,
  run: () => Promise<T>,
): Promise<T> {
  if (sink === undefined) return await run();
  const startedAtMs = now();
  try {
    const value = await run();
    emitOpt0064Event(sink, {
      scope,
      category,
      operation,
      startedAtMs,
      completedAtMs: now(),
      details: Object.freeze({ ...details, outcome: "success" }),
    });
    return value;
  } catch (error) {
    emitOpt0064Event(sink, {
      scope,
      category,
      operation,
      startedAtMs,
      completedAtMs: now(),
      details: Object.freeze({ ...details, outcome: "failure" }),
    });
    throw error;
  }
}

function captureOpt0064Sync<T>(
  now: () => number,
  sink: AceOpt0064CaptureSink | undefined,
  scope: AceOpt0064CaptureEvent["scope"],
  category: AceOpt0064CaptureEvent["category"],
  operation: string,
  details: Readonly<Record<string, unknown>>,
  run: () => T,
): T {
  if (sink === undefined) return run();
  const startedAtMs = now();
  try {
    const value = run();
    emitOpt0064Event(sink, {
      scope,
      category,
      operation,
      startedAtMs,
      completedAtMs: now(),
      details: Object.freeze({ ...details, outcome: "success" }),
    });
    return value;
  } catch (error) {
    emitOpt0064Event(sink, {
      scope,
      category,
      operation,
      startedAtMs,
      completedAtMs: now(),
      details: Object.freeze({ ...details, outcome: "failure" }),
    });
    throw error;
  }
}

function emitOpt0064Event(
  sink: AceOpt0064CaptureSink | undefined,
  event: Opt0064EventInput,
): void {
  if (sink === undefined) return;
  const completedAtMs = Math.max(event.startedAtMs, event.completedAtMs);
  try {
    sink.onEvent(Object.freeze({
      schema: "ace-opt-0064-direct-request-capture-event-v1",
      scope: event.scope,
      category: event.category,
      operation: event.operation,
      startedAtMs: event.startedAtMs,
      completedAtMs,
      wallMs: completedAtMs - event.startedAtMs,
      details: event.details,
    }));
  } catch {
    // OPT-0064 capture is observational and cannot change product execution.
  }
}

function opt0064AcquisitionTraceSink(
  now: () => number,
  sink: AceOpt0064CaptureSink,
  scope: AceOpt0064CaptureEvent["scope"],
  packageKind: "main" | "dit-dense" | "vae",
): (trace: AceModelAcquisitionTrace) => void {
  return (trace) => {
    const instant = now();
    emitOpt0064Event(sink, {
      scope,
      category: "authentication",
      operation: `${packageKind}-${trace.operation}`,
      startedAtMs: trace.operation === "cache-authentication"
        ? trace.startedAtMs
        : instant,
      completedAtMs: trace.operation === "cache-authentication"
        ? trace.completedAtMs
        : instant,
      details: Object.freeze({ packageKind, ...trace }),
    });
  };
}

function emitOpt0064AcquisitionPlan(
  now: () => number,
  sink: AceOpt0064CaptureSink | undefined,
  scope: AceOpt0064CaptureEvent["scope"],
  packageKind: "main" | "dit-dense" | "vae",
  acquired: AceAcquiredModelFiles,
): void {
  if (sink === undefined) return;
  const instant = now();
  emitOpt0064Event(sink, {
    scope,
    category: "authentication",
    operation: `${packageKind}-acquisition-plan`,
    startedAtMs: instant,
    completedAtMs: instant,
    details: Object.freeze({
      packageKind,
      fileCount: acquired.plan.files.length,
      runtimeBytes: acquired.plan.runtimeBytes,
      cachedFileCount: acquired.plan.cachedFiles.length,
      cachedBytes: acquired.plan.cachedBytes,
      downloadFileCount: acquired.plan.downloadFiles.length,
      downloadBytes: acquired.plan.downloadBytes,
      exactFileObjectCount: acquired.files.size,
    }),
  });
}

function opt0064UploadTraceSink(
  sink: AceOpt0064CaptureSink,
  packageKind: string,
): (trace: AceGpuTensorPhaseUploadTrace) => void {
  return (trace) => {
    emitOpt0064Event(sink, {
      scope: "generation",
      category: "upload",
      operation: `${packageKind}-${trace.phases.join("+")}-file-upload`,
      startedAtMs: trace.startedAtMs,
      completedAtMs: trace.completedAtMs,
      details: Object.freeze({ packageKind, ...trace }),
    });
  };
}

interface Opt0064GpuApiCapture {
  readonly device: GPUDevice;
  finish(): void;
}

function createOpt0064GpuApiCapture(
  device: GPUDevice,
  now: () => number,
  sink: AceOpt0064CaptureSink | undefined,
  owner: "conditioning" | "dit" | "vae",
): Opt0064GpuApiCapture {
  if (sink === undefined) {
    return Object.freeze({ device, finish: () => undefined });
  }
  const capturedMethods = new Set<PropertyKey>([
    "createShaderModule",
    "createComputePipeline",
    "createComputePipelineAsync",
    "createBindGroup",
    "createBindGroupLayout",
    "createPipelineLayout",
    "createBuffer",
  ]);
  const stats = new Map<string, { count: number; wallMs: number }>();
  const startedAtMs = now();
  let active = true;
  const record = (method: string, callStartedAtMs: number): void => {
    const previous = stats.get(method) ?? { count: 0, wallMs: 0 };
    stats.set(method, {
      count: previous.count + 1,
      wallMs: previous.wallMs + nonnegativeElapsed(now(), callStartedAtMs),
    });
  };
  const proxy = new Proxy(device, {
    get(target, property): unknown {
      const value = Reflect.get(target, property, target) as unknown;
      if (typeof value !== "function") return value;
      const bound = value.bind(target) as (...args: unknown[]) => unknown;
      if (!capturedMethods.has(property)) return bound;
      return (...args: unknown[]): unknown => {
        if (!active) return bound(...args);
        const callStartedAtMs = now();
        let result: unknown;
        try {
          result = bound(...args);
        } catch (error) {
          record(String(property), callStartedAtMs);
          throw error;
        }
        if (
          result !== null &&
          typeof result === "object" &&
          "then" in result &&
          typeof (result as { then?: unknown }).then === "function"
        ) {
          void Promise.resolve(result).then(
            () => {
              record(String(property), callStartedAtMs);
            },
            () => {
              record(String(property), callStartedAtMs);
            },
          );
          return result;
        }
        record(String(property), callStartedAtMs);
        return result;
      };
    },
  }) as GPUDevice;
  return Object.freeze({
    device: proxy,
    finish: () => {
      if (!active) return;
      active = false;
      const completedAtMs = now();
      const methods = Object.freeze(Object.fromEntries(
        [...stats.entries()].sort(([left], [right]) => left.localeCompare(right))
          .map(([method, value]) => [method, Object.freeze({ ...value })]),
      ));
      const methodWall = (method: string): number => stats.get(method)?.wallMs ?? 0;
      emitOpt0064Event(sink, {
        scope: "generation",
        category: "construction",
        operation: `${owner}-gpu-api-construction-summary`,
        startedAtMs,
        completedAtMs,
        details: Object.freeze({
          owner,
          methods,
          shaderModuleCreationMs: methodWall("createShaderModule"),
          pipelineCompilationMs:
            methodWall("createComputePipeline") +
            methodWall("createComputePipelineAsync"),
          bindGroupAndLayoutConstructionMs:
            methodWall("createBindGroup") +
            methodWall("createBindGroupLayout") +
            methodWall("createPipelineLayout"),
          bufferAllocationMs: methodWall("createBuffer"),
          perCallWallMayOverlap: true,
          proxyAddsNoGpuCommandsDrainsHashesOrCopies: true,
        }),
      });
    },
  });
}

async function tryRollback(
  transaction: PipelineAudioTransaction,
): Promise<boolean> {
  try {
    await transaction.rollback();
    return true;
  } catch {
    return false;
  }
}

function nonnegativeElapsed(now: number, start: number): number {
  return Math.max(0, now - start);
}

function sha256U32Le(values: Uint32Array): string {
  const bytes = new Uint8Array(values.length * Uint32Array.BYTES_PER_ELEMENT);
  const view = new DataView(bytes.buffer);
  for (let index = 0; index < values.length; index += 1) {
    view.setUint32(index * Uint32Array.BYTES_PER_ELEMENT, values[index]!, true);
  }
  return aceSha256Hex(bytes);
}

function sha256F32Le(
  values: Float32Array,
  start = 0,
  end = values.length,
): string {
  if (
    !Number.isSafeInteger(start) ||
    !Number.isSafeInteger(end) ||
    start < 0 ||
    end < start ||
    end > values.length
  ) {
    throw new RangeError("ACE diagnostic float digest range is invalid");
  }
  const bytes = new Uint8Array((end - start) * Float32Array.BYTES_PER_ELEMENT);
  const view = new DataView(bytes.buffer);
  for (let index = start; index < end; index += 1) {
    view.setFloat32(
      (index - start) * Float32Array.BYTES_PER_ELEMENT,
      values[index]!,
      true,
    );
  }
  return aceSha256Hex(bytes);
}

function createOpt0081RepresentativeConditioningAuthority(
  tokenized: Readonly<{
    readonly textTokenIds: Uint32Array;
    readonly lyricTokenIds: Uint32Array;
  }>,
  conditioned: Pick<
    AceConditioningGpuResult,
    | "conditionTokens"
    | "conditionHiddenStates"
    | "contextLatents"
  >,
): AceOpt0081RepresentativeConditioningAuthority {
  const textTokenCount = tokenized.textTokenIds.length;
  const textTokenSha256 = sha256U32Le(tokenized.textTokenIds);
  const lyricTokenCount = tokenized.lyricTokenIds.length;
  const lyricTokenSha256 = sha256U32Le(tokenized.lyricTokenIds);
  const conditionTokens = conditioned.conditionTokens;
  const conditionElementCount = conditioned.conditionHiddenStates.length;
  const conditionSha256 = sha256F32Le(conditioned.conditionHiddenStates);
  const contextElementCount = conditioned.contextLatents.length;
  const contextSha256 = sha256F32Le(conditioned.contextLatents);
  if (
    !Number.isSafeInteger(textTokenCount) || textTokenCount <= 0 ||
    !Number.isSafeInteger(lyricTokenCount) || lyricTokenCount <= 0 ||
    !Number.isSafeInteger(conditionTokens) || conditionTokens <= 0 ||
    conditionTokens !== textTokenCount + lyricTokenCount + 1 ||
    conditionElementCount !== conditionTokens * 2_048 ||
    contextElementCount <= 0 || contextElementCount % 128 !== 0
  ) {
    throw new Error(
      "OPT-0081 representative conditioning authority changed",
    );
  }
  return Object.freeze({
    schema: "ace-opt-0081-representative-conditioning-authority-v1",
    textTokenCount,
    textTokenSha256,
    lyricTokenCount,
    lyricTokenSha256,
    conditionTokens,
    conditionElementCount,
    conditionSha256,
    contextElementCount,
    contextSha256,
  });
}

function finiteStats(values: Float32Array): Readonly<{
  readonly nonFinite: number;
  readonly maxAbs: number;
}> {
  let nonFinite = 0;
  let maxAbs = 0;
  for (const value of values) {
    if (!Number.isFinite(value)) {
      nonFinite += 1;
    } else {
      maxAbs = Math.max(maxAbs, Math.abs(value));
    }
  }
  return Object.freeze({ nonFinite, maxAbs });
}

function sumU32(values: Uint32Array): number {
  let sum = 0;
  for (const value of values) sum += value;
  return sum;
}
