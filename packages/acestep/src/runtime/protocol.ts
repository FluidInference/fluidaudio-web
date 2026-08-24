import {
  ACE_CHANNEL_COUNT,
  ACE_SAMPLE_RATE_HZ,
  isAceGenerationRequest,
  type AceGenerationRequest,
  type AceGenerationResult,
} from "../api.js";
import type {
  AceExecutionProfile,
  AceModelProfileId,
  AceSchedulingProfile,
  AceWebGpuCapabilityReport,
  AceWebGpuLimits,
} from "../webgpu/capabilities.js";
import {
  ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_PROFILE,
  ACE_OPT_0028_VAE_FP16_MANIFEST_SHA256,
  ACE_OPT_0040_VAE_FP16_FIXED32_SHAPE_SELECTED_PROFILE,
  ACE_OPT_0054_VAE_FP16_FIXED32_REVISION7_PROFILE,
  ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PROFILE,
  ACE_OPT_0072_VAE_FP16_FIXED32_DUAL_K4_PRODUCTION_RUNTIME_PROFILE,
  ACE_OPT_0088_VAE_FP16_PORTABLE_DUAL_K4_PROFILE,
} from "../webgpu/vae-fp16-profile.js";
import {
  ACE_REFERENCE_MANIFEST_SHA256,
  ACE_OPT_0054_VAE_REVISION7_MANIFEST_SHA256,
} from "../model/package.js";
import { deriveAceDurationGraphShape } from "../model/graph-contract.js";
import {
  ACE_OPT_0009_DIT_DENSE_KERNEL_SET_ID,
  ACE_OPT_0009_DIT_DENSE_MANIFEST_BYTES,
  ACE_OPT_0009_DIT_DENSE_MANIFEST_SHA256,
  ACE_OPT_0009_DIT_MIXED_LAYER_BYTES,
  ACE_OPT_0009_DIT_MIXED_RESIDENT_WEIGHT_BYTES,
  ACE_OPT_0009_DIT_DENSE_RUNTIME_PROFILE,
  ACE_OPT_0037_DIT_K4_KERNEL_SET_ID,
  ACE_OPT_0037_DIT_K4_LAYER_BYTES,
  ACE_OPT_0037_DIT_K4_MANIFEST_BYTES,
  ACE_OPT_0037_DIT_K4_MANIFEST_SHA256,
  ACE_OPT_0037_DIT_K4_RESIDENT_WEIGHT_BYTES,
  ACE_OPT_0037_DIT_K4_RUNTIME_PROFILE,
  ACE_OPT_0056_DIT_SELECTIVE_K4_KERNEL_SET_ID,
  ACE_OPT_0056_DIT_SELECTIVE_K4_RUNTIME_PROFILE,
  ACE_OPT_0088_DIT_DENSE_PORTABLE_KERNEL_SET_ID,
} from "../webgpu/dit-fp16-package.js";
import {
  ACE_OPT_0062_DIT_QUAD_QUERY_ATTENTION_KERNEL_SET_ID,
  ACE_OPT_0062_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
  ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_KERNEL_SET_ID,
  ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
  ACE_OPT_0088_DIT_PORTABLE_ATTENTION_KERNEL_SET_ID,
} from "../webgpu/dit-attention-profile.js";
import {
  ACE_OPT_0070_VAE_C2378_MAXIMUM_WINDOW_FRAMES,
  ACE_OPT_0070_VAE_C2378_WINDOW_RUNTIME_PROFILE,
  ACE_VAE_C512_WINDOW_RUNTIME_PROFILE,
} from "../webgpu/vae-window-profile.js";
import { planAceOpt0011Fp16VaeChunkDispatches } from
  "../webgpu/vae-fp16-decoder.js";
import {
  ACE_FP16_PORTABLE_PROFILE,
  ACE_REFERENCE_PORTABLE_PROFILE,
  ACE_REFERENCE_SUBGROUP_PROFILE,
  ACE_STOCK_WEBGPU_FEATURES,
  ACE_WEBGPU_LIMIT_NAMES,
} from "../webgpu/capabilities.js";
import {
  isAceGenerationProgress,
  isAceGenerationStage,
  isAceInitializationProgress,
  type AceGenerationProgress,
  type AceInitializationProgress,
  type AceStageTiming,
} from "./stages.js";
import {
  ACE_PLANNER_SOURCE_REVISION,
  ACE_SOURCE_REVISION,
  PARAKEET_REFERENCE_REVISION,
} from "./diagnostics.js";
import type {
  AceDiagnostic,
  AceFailureContext,
  AceRuntimeDiagnostics,
} from "./diagnostics.js";
import { isAceSeed } from "./seed.js";

export type AceModelLoadSource = "cache-only" | "cache-or-network";

export type AceWorkerVaePackageConfiguration =
  | Readonly<{
      readonly manifestUrl: string;
      /** Exact revision-6 C512 control trust root. */
      readonly manifestSha256: typeof ACE_OPT_0028_VAE_FP16_MANIFEST_SHA256;
      readonly runtimeProfile:
        typeof ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_PROFILE.id;
      readonly windowRuntimeProfile?:
        typeof ACE_VAE_C512_WINDOW_RUNTIME_PROFILE;
      readonly maxWindowFrames: 512;
    }>
  | Readonly<{
      readonly manifestUrl: string;
      /** Explicit sequential scalar-FP32-K7 revision-6 diagnostic oracle. */
      readonly manifestSha256: typeof ACE_OPT_0028_VAE_FP16_MANIFEST_SHA256;
      readonly runtimeProfile:
        typeof ACE_OPT_0040_VAE_FP16_FIXED32_SHAPE_SELECTED_PROFILE.id;
      readonly windowRuntimeProfile?:
        typeof ACE_VAE_C512_WINDOW_RUNTIME_PROFILE;
      readonly maxWindowFrames: 512;
    }>
  | Readonly<{
      readonly manifestUrl: string;
      /** Exact OPT-0054 revision-7 candidate trust root. */
      readonly manifestSha256:
        typeof ACE_OPT_0054_VAE_REVISION7_MANIFEST_SHA256;
      readonly runtimeProfile:
        typeof ACE_OPT_0054_VAE_FP16_FIXED32_REVISION7_PROFILE.id;
      readonly windowRuntimeProfile?:
        typeof ACE_VAE_C512_WINDOW_RUNTIME_PROFILE;
      readonly maxWindowFrames: 512;
    }>
  | Readonly<{
      readonly manifestUrl: string;
      /** Public OPT-0072 identity over the authenticated revision-7 bytes. */
      readonly manifestSha256:
        typeof ACE_OPT_0054_VAE_REVISION7_MANIFEST_SHA256;
      readonly runtimeProfile:
        typeof ACE_OPT_0072_VAE_FP16_FIXED32_DUAL_K4_PRODUCTION_RUNTIME_PROFILE;
      readonly windowRuntimeProfile:
        typeof ACE_OPT_0070_VAE_C2378_WINDOW_RUNTIME_PROFILE;
      readonly maxWindowFrames:
        typeof ACE_OPT_0070_VAE_C2378_MAXIMUM_WINDOW_FRAMES;
    }>;

export type AceWorkerDitDensePackageConfiguration =
  | Readonly<{
      readonly manifestUrl: string;
      readonly manifestSha256: typeof ACE_OPT_0009_DIT_DENSE_MANIFEST_SHA256;
      readonly runtimeProfile: typeof ACE_OPT_0009_DIT_DENSE_RUNTIME_PROFILE;
    }>
  | Readonly<{
      readonly manifestUrl: string;
      readonly manifestSha256: typeof ACE_OPT_0037_DIT_K4_MANIFEST_SHA256;
      readonly runtimeProfile: typeof ACE_OPT_0037_DIT_K4_RUNTIME_PROFILE;
    }>
  | Readonly<{
      readonly manifestUrl: string;
      readonly manifestSha256: typeof ACE_OPT_0037_DIT_K4_MANIFEST_SHA256;
      readonly runtimeProfile:
        typeof ACE_OPT_0056_DIT_SELECTIVE_K4_RUNTIME_PROFILE;
    }>;

/** Terminal GPU failures that require a fresh device/worker, not a retry. */
export const ACE_FATAL_GPU_ERROR_CODES = [
  "WEBGPU_DEVICE_LOST",
  "WEBGPU_UNCAPTURED_ERROR",
] as const;

export type AceFatalGpuErrorCode =
  (typeof ACE_FATAL_GPU_ERROR_CODES)[number];

export function isAceFatalGpuErrorCode(
  value: unknown,
): value is AceFatalGpuErrorCode {
  return (
    typeof value === "string" &&
    (ACE_FATAL_GPU_ERROR_CODES as readonly string[]).includes(value)
  );
}

export interface AceWorkerConfiguration {
  readonly manifestUrl: string;
  /** Externally pinned identity; the loader must hash bytes before JSON parsing. */
  readonly manifestSha256: string;
  readonly modelProfile: AceModelProfileId;
  readonly schedulingProfile: AceSchedulingProfile;
  readonly ditDensePackage: AceWorkerDitDensePackageConfiguration;
  /** OPT-0062 is diagnostic-only; OPT-0070 is the exact production identity. */
  readonly ditAttentionRuntimeProfile?:
    | typeof ACE_OPT_0062_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE
    | typeof ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE;
  readonly vaePackage: AceWorkerVaePackageConfiguration;
}

export interface AceInitializeMessage {
  readonly type: "initialize";
  readonly requestId: number;
  readonly configuration: AceWorkerConfiguration;
  readonly modelSource: AceModelLoadSource;
  readonly reportProgress: boolean;
  readonly reportDiagnostics: boolean;
}

export interface AceCancelInitializationMessage {
  readonly type: "cancel-initialization";
  readonly requestId: number;
}

export interface AceGenerateMessage {
  readonly type: "generate";
  readonly jobId: number;
  readonly request: AceGenerationRequest;
  readonly reportProgress: boolean;
  readonly reportDiagnostics: boolean;
}

export interface AceCancelMessage {
  readonly type: "cancel";
  readonly jobId: number;
}

export interface AceDisposeMessage {
  readonly type: "dispose";
  readonly requestId: number;
}

export type AceClientMessage =
  | AceInitializeMessage
  | AceCancelInitializationMessage
  | AceGenerateMessage
  | AceCancelMessage
  | AceDisposeMessage;

export interface AceSerializedWorkerError {
  readonly name: string;
  readonly message: string;
  readonly code: string;
  readonly stack?: string;
  readonly context?: AceFailureContext;
}

export interface AceInitializationProgressMessage {
  readonly type: "initialization-progress";
  readonly requestId: number;
  readonly progress: AceInitializationProgress;
}

export interface AceReadyMessage {
  readonly type: "ready";
  readonly requestId: number;
  readonly diagnostics: AceRuntimeDiagnostics;
}

export interface AceInitializationCancelledMessage {
  readonly type: "initialization-cancelled";
  readonly requestId: number;
}

export interface AceGenerationProgressMessage {
  readonly type: "generation-progress";
  readonly jobId: number;
  readonly progress: AceGenerationProgress;
}

export interface AceDiagnosticMessage {
  readonly type: "diagnostic";
  readonly requestId?: number;
  readonly jobId?: number;
  readonly diagnostic: AceDiagnostic;
}

export interface AceGenerationResultMessage {
  readonly type: "result";
  readonly jobId: number;
  readonly result: AceGenerationResult;
}

export interface AceCancelledMessage {
  readonly type: "cancelled";
  readonly jobId: number;
}

export interface AceDisposedMessage {
  readonly type: "disposed";
  readonly requestId: number;
}

export interface AceWorkerErrorMessage {
  readonly type: "error";
  readonly requestId?: number;
  readonly jobId?: number;
  readonly error: AceSerializedWorkerError;
}

export type AceWorkerMessage =
  | AceInitializationProgressMessage
  | AceReadyMessage
  | AceInitializationCancelledMessage
  | AceGenerationProgressMessage
  | AceDiagnosticMessage
  | AceGenerationResultMessage
  | AceCancelledMessage
  | AceDisposedMessage
  | AceWorkerErrorMessage;

export function isAceClientMessage(value: unknown): value is AceClientMessage {
  if (!isRecord(value) || typeof value.type !== "string") return false;
  switch (value.type) {
    case "initialize":
      return (
        hasOnlyKeys(value, [
          "type",
          "requestId",
          "configuration",
          "modelSource",
          "reportProgress",
          "reportDiagnostics",
        ]) &&
        isRequestId(value.requestId) &&
        isWorkerConfiguration(value.configuration) &&
        (value.modelSource === "cache-only" ||
          value.modelSource === "cache-or-network") &&
        typeof value.reportProgress === "boolean" &&
        typeof value.reportDiagnostics === "boolean"
      );
    case "cancel-initialization":
      return (
        hasOnlyKeys(value, ["type", "requestId"]) &&
        isRequestId(value.requestId)
      );
    case "generate":
      return (
        hasOnlyKeys(value, [
          "type",
          "jobId",
          "request",
          "reportProgress",
          "reportDiagnostics",
        ]) &&
        isRequestId(value.jobId) &&
        isAceGenerationRequest(value.request) &&
        typeof value.reportProgress === "boolean" &&
        typeof value.reportDiagnostics === "boolean"
      );
    case "cancel":
      return (
        hasOnlyKeys(value, ["type", "jobId"]) &&
        isRequestId(value.jobId)
      );
    case "dispose":
      return (
        hasOnlyKeys(value, ["type", "requestId"]) &&
        isRequestId(value.requestId)
      );
    default:
      return false;
  }
}

export function isAceWorkerMessage(value: unknown): value is AceWorkerMessage {
  if (!isRecord(value) || typeof value.type !== "string") return false;
  switch (value.type) {
    case "initialization-progress":
      return (
        hasOnlyKeys(value, ["type", "requestId", "progress"]) &&
        isRequestId(value.requestId) &&
        isAceInitializationProgress(value.progress)
      );
    case "ready":
      return (
        hasOnlyKeys(value, ["type", "requestId", "diagnostics"]) &&
        isRequestId(value.requestId) &&
        isAceRuntimeDiagnosticsValue(value.diagnostics)
      );
    case "initialization-cancelled":
      return (
        hasOnlyKeys(value, ["type", "requestId"]) &&
        isRequestId(value.requestId)
      );
    case "generation-progress":
      return (
        hasOnlyKeys(value, ["type", "jobId", "progress"]) &&
        isRequestId(value.jobId) &&
        isAceGenerationProgress(value.progress)
      );
    case "diagnostic":
      return (
        hasOnlyKeys(value, [
          "type",
          "requestId",
          "jobId",
          "diagnostic",
        ]) &&
        (value.requestId === undefined || isRequestId(value.requestId)) &&
        (value.jobId === undefined || isRequestId(value.jobId)) &&
        isAceDiagnostic(value.diagnostic)
      );
    case "result":
      return (
        hasOnlyKeys(value, ["type", "jobId", "result"]) &&
        isRequestId(value.jobId) &&
        isAceGenerationResultValue(value.result)
      );
    case "cancelled":
      return (
        hasOnlyKeys(value, ["type", "jobId"]) && isRequestId(value.jobId)
      );
    case "disposed":
      return (
        hasOnlyKeys(value, ["type", "requestId"]) &&
        isRequestId(value.requestId)
      );
    case "error":
      return (
        hasOnlyKeys(value, ["type", "requestId", "jobId", "error"]) &&
        (value.requestId === undefined || isRequestId(value.requestId)) &&
        (value.jobId === undefined || isRequestId(value.jobId)) &&
        isSerializedWorkerError(value.error)
      );
    default:
      return false;
  }
}

/** Validate diagnostics emitted asynchronously by a backend implementation. */
export function isAceDiagnosticValue(value: unknown): value is AceDiagnostic {
  return isAceDiagnostic(value);
}

export function serializeAceWorkerError(
  error: unknown,
  fallbackCode = "INTERNAL_ERROR",
  context?: AceFailureContext,
): AceSerializedWorkerError {
  const safeFallbackCode = isNonEmptyString(fallbackCode)
    ? fallbackCode
    : "INTERNAL_ERROR";
  if (error instanceof Error) {
    const coded = error as Error & {
      readonly code?: unknown;
      readonly context?: unknown;
    };
    const errorContext = isFailureContext(coded.context)
      ? coded.context
      : context;
    return {
      name: isNonEmptyString(error.name) ? error.name : "Error",
      message: error.message,
      code: isNonEmptyString(coded.code) ? coded.code : safeFallbackCode,
      ...(typeof error.stack === "string" ? { stack: error.stack } : {}),
      ...(errorContext === undefined ? {} : { context: errorContext }),
    };
  }
  return {
    name: "Error",
    message: String(error),
    code: safeFallbackCode,
    ...(context === undefined ? {} : { context }),
  };
}

function isWorkerConfiguration(value: unknown): value is AceWorkerConfiguration {
  return (
    isRecord(value) &&
    hasOnlyKeys(value, [
      "manifestUrl",
      "manifestSha256",
      "modelProfile",
      "schedulingProfile",
      "ditDensePackage",
      "ditAttentionRuntimeProfile",
      "vaePackage",
    ]) &&
    typeof value.manifestUrl === "string" &&
    value.manifestUrl.length > 0 &&
    typeof value.manifestSha256 === "string" &&
    /^[0-9a-f]{64}$/.test(value.manifestSha256) &&
    value.modelProfile === "reference-bf16" &&
    (value.schedulingProfile === "cooperative" ||
      value.schedulingProfile === "benchmark") &&
    isWorkerDitDensePackageConfiguration(value.ditDensePackage) &&
    (value.ditAttentionRuntimeProfile === undefined ||
      ((value.ditAttentionRuntimeProfile ===
          ACE_OPT_0062_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE ||
        value.ditAttentionRuntimeProfile ===
          ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE) &&
        value.ditDensePackage.runtimeProfile ===
          ACE_OPT_0009_DIT_DENSE_RUNTIME_PROFILE)) &&
    isWorkerVaePackageConfiguration(value.vaePackage) &&
    ((value.ditAttentionRuntimeProfile ===
        ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE) ===
      (value.vaePackage.windowRuntimeProfile ===
        ACE_OPT_0070_VAE_C2378_WINDOW_RUNTIME_PROFILE)) &&
    ((value.ditAttentionRuntimeProfile ===
        ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE) ===
      (value.vaePackage.runtimeProfile ===
        ACE_OPT_0072_VAE_FP16_FIXED32_DUAL_K4_PRODUCTION_RUNTIME_PROFILE))
  );
}

function isWorkerDitDensePackageConfiguration(
  value: unknown,
): value is AceWorkerDitDensePackageConfiguration {
  return (
    isRecord(value) &&
    hasOnlyKeys(value, ["manifestUrl", "manifestSha256", "runtimeProfile"]) &&
    typeof value.manifestUrl === "string" &&
    value.manifestUrl.length > 0 &&
    ((value.manifestSha256 === ACE_OPT_0037_DIT_K4_MANIFEST_SHA256 &&
      (value.runtimeProfile === ACE_OPT_0037_DIT_K4_RUNTIME_PROFILE ||
        value.runtimeProfile ===
          ACE_OPT_0056_DIT_SELECTIVE_K4_RUNTIME_PROFILE)) ||
      (value.manifestSha256 === ACE_OPT_0009_DIT_DENSE_MANIFEST_SHA256 &&
        value.runtimeProfile === ACE_OPT_0009_DIT_DENSE_RUNTIME_PROFILE))
  );
}

function isWorkerVaePackageConfiguration(
  value: unknown,
): value is AceWorkerVaePackageConfiguration {
  return (
    isRecord(value) &&
    hasOnlyKeys(value, [
      "manifestUrl",
      "manifestSha256",
      "runtimeProfile",
      "windowRuntimeProfile",
      "maxWindowFrames",
    ]) &&
    typeof value.manifestUrl === "string" &&
    value.manifestUrl.length > 0 &&
    ((value.manifestSha256 === ACE_OPT_0028_VAE_FP16_MANIFEST_SHA256 &&
      (value.runtimeProfile ===
          ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_PROFILE.id ||
        value.runtimeProfile ===
          ACE_OPT_0040_VAE_FP16_FIXED32_SHAPE_SELECTED_PROFILE.id)) ||
      (value.manifestSha256 ===
          ACE_OPT_0054_VAE_REVISION7_MANIFEST_SHA256 &&
        (value.runtimeProfile ===
            ACE_OPT_0054_VAE_FP16_FIXED32_REVISION7_PROFILE.id ||
          value.runtimeProfile ===
            ACE_OPT_0072_VAE_FP16_FIXED32_DUAL_K4_PRODUCTION_RUNTIME_PROFILE))) &&
    ((value.maxWindowFrames === 512 &&
      (value.windowRuntimeProfile === undefined ||
        value.windowRuntimeProfile === ACE_VAE_C512_WINDOW_RUNTIME_PROFILE) &&
      value.runtimeProfile !==
        ACE_OPT_0072_VAE_FP16_FIXED32_DUAL_K4_PRODUCTION_RUNTIME_PROFILE) ||
      (value.manifestSha256 ===
          ACE_OPT_0054_VAE_REVISION7_MANIFEST_SHA256 &&
        value.runtimeProfile ===
          ACE_OPT_0072_VAE_FP16_FIXED32_DUAL_K4_PRODUCTION_RUNTIME_PROFILE &&
        value.windowRuntimeProfile ===
          ACE_OPT_0070_VAE_C2378_WINDOW_RUNTIME_PROFILE &&
        value.maxWindowFrames ===
          ACE_OPT_0070_VAE_C2378_MAXIMUM_WINDOW_FRAMES))
  );
}

export function isAceGenerationResultValue(
  value: unknown,
): value is AceGenerationResult {
  return (
    isRecord(value) &&
    hasOnlyKeys(value, [
      "audio",
      "audioStorageId",
      "mimeType",
      "sampleRateHz",
      "channelCount",
      "frameCount",
      "durationSeconds",
      "seed",
      "generationProfile",
      "modelManifestId",
      "modelManifestSha256",
      "diagnostics",
      "metrics",
    ]) &&
    value.audio instanceof Blob &&
    value.audio.type === "audio/wav" &&
    typeof value.audioStorageId === "string" &&
    /^(?!\.{1,2}$)[A-Za-z0-9._-]{1,96}$/.test(value.audioStorageId) &&
    value.mimeType === "audio/wav" &&
    value.sampleRateHz === ACE_SAMPLE_RATE_HZ &&
    value.channelCount === ACE_CHANNEL_COUNT &&
    isPositiveSafeInteger(value.frameCount) &&
    isNonNegativeFinite(value.durationSeconds) &&
    value.durationSeconds === value.frameCount / ACE_SAMPLE_RATE_HZ &&
    isAceSeed(value.seed) &&
    value.generationProfile === "ace-turbo-v1-correctness" &&
    isNonEmptyString(value.modelManifestId) &&
    isSha256(value.modelManifestSha256) &&
    isAceRuntimeDiagnosticsValue(value.diagnostics) &&
    value.diagnostics.modelManifestId === value.modelManifestId &&
    value.diagnostics.modelManifestSha256 === value.modelManifestSha256 &&
    isGenerationMetrics(
      value.metrics,
      value.frameCount,
      value.durationSeconds,
      value.diagnostics,
    )
  );
}

function isAceDiagnostic(value: unknown): value is AceDiagnostic {
  return (
    isRecord(value) &&
    hasOnlyKeys(value, [
      "severity",
      "code",
      "message",
      "elapsedMs",
      "stage",
      "details",
    ]) &&
    (value.severity === "info" ||
      value.severity === "warning" ||
      value.severity === "error") &&
    isNonEmptyString(value.code) &&
    isNonEmptyString(value.message) &&
    isNonNegativeFinite(value.elapsedMs) &&
    (value.stage === undefined || isAceGenerationStage(value.stage)) &&
    (value.details === undefined || isDiagnosticDetails(value.details))
  );
}

function isSerializedWorkerError(
  value: unknown,
): value is AceSerializedWorkerError {
  return (
    isRecord(value) &&
    hasOnlyKeys(value, ["name", "message", "code", "stack", "context"]) &&
    isNonEmptyString(value.name) &&
    typeof value.message === "string" &&
    isNonEmptyString(value.code) &&
    (value.stack === undefined || typeof value.stack === "string") &&
    (value.context === undefined || isFailureContext(value.context))
  );
}

function isFailureContext(value: unknown): value is AceFailureContext {
  if (!isRecord(value)) return false;
  return (
    hasOnlyKeys(value, [
      "stage",
      "operation",
      "resourceLabel",
      "requestedBytes",
      "modelManifestId",
      "executionProfileId",
    ]) &&
    (value.stage === undefined || isAceGenerationStage(value.stage)) &&
    (value.operation === undefined || typeof value.operation === "string") &&
    (value.resourceLabel === undefined || typeof value.resourceLabel === "string") &&
    (value.requestedBytes === undefined || isRequestId(value.requestedBytes)) &&
    (value.modelManifestId === undefined || typeof value.modelManifestId === "string") &&
    (value.executionProfileId === undefined ||
      typeof value.executionProfileId === "string")
  );
}

export function isAceRuntimeDiagnosticsValue(
  value: unknown,
): value is AceRuntimeDiagnostics {
  const ditAttentionIdentityValid = isRecord(value) && (
    (value.ditAttentionRuntimeProfile === undefined &&
      value.ditAttentionKernelSetId === undefined) ||
    (value.ditAttentionRuntimeProfile ===
        ACE_OPT_0062_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE &&
      value.ditAttentionKernelSetId ===
        ACE_OPT_0062_DIT_QUAD_QUERY_ATTENTION_KERNEL_SET_ID) ||
    (value.ditAttentionRuntimeProfile ===
        ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE &&
      (value.ditAttentionKernelSetId ===
          ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_KERNEL_SET_ID ||
        value.ditAttentionKernelSetId ===
          ACE_OPT_0088_DIT_PORTABLE_ATTENTION_KERNEL_SET_ID))
  );
  const vaeWindowIdentityValid = isRecord(value) && (
    (value.vaeWindowRuntimeProfile === ACE_VAE_C512_WINDOW_RUNTIME_PROFILE &&
      value.vaeMaxWindowFrames === 512) ||
    (value.vaeWindowRuntimeProfile ===
        ACE_OPT_0070_VAE_C2378_WINDOW_RUNTIME_PROFILE &&
      value.vaeMaxWindowFrames ===
        ACE_OPT_0070_VAE_C2378_MAXIMUM_WINDOW_FRAMES)
  );
  const ditDenseIdentityValid = isRecord(value) && (
    (value.ditDenseManifestSha256 ===
        ACE_OPT_0009_DIT_DENSE_MANIFEST_SHA256 &&
      value.ditDenseManifestByteLength ===
        ACE_OPT_0009_DIT_DENSE_MANIFEST_BYTES &&
      value.ditDenseRuntimeProfile ===
        ACE_OPT_0009_DIT_DENSE_RUNTIME_PROFILE &&
      (value.ditDenseKernelSetId === ACE_OPT_0009_DIT_DENSE_KERNEL_SET_ID ||
        value.ditDenseKernelSetId ===
          ACE_OPT_0088_DIT_DENSE_PORTABLE_KERNEL_SET_ID) &&
      value.ditDenseLayerBytes === ACE_OPT_0009_DIT_MIXED_LAYER_BYTES &&
      value.ditResidentWeightBytes ===
        ACE_OPT_0009_DIT_MIXED_RESIDENT_WEIGHT_BYTES) ||
    (value.ditDenseManifestSha256 === ACE_OPT_0037_DIT_K4_MANIFEST_SHA256 &&
      value.ditDenseManifestByteLength === ACE_OPT_0037_DIT_K4_MANIFEST_BYTES &&
      ((value.ditDenseRuntimeProfile === ACE_OPT_0037_DIT_K4_RUNTIME_PROFILE &&
        value.ditDenseKernelSetId === ACE_OPT_0037_DIT_K4_KERNEL_SET_ID) ||
        (value.ditDenseRuntimeProfile ===
            ACE_OPT_0056_DIT_SELECTIVE_K4_RUNTIME_PROFILE &&
          value.ditDenseKernelSetId ===
            ACE_OPT_0056_DIT_SELECTIVE_K4_KERNEL_SET_ID)) &&
      value.ditDenseLayerBytes === ACE_OPT_0037_DIT_K4_LAYER_BYTES &&
      value.ditResidentWeightBytes ===
        ACE_OPT_0037_DIT_K4_RESIDENT_WEIGHT_BYTES)
  );
  const vaeIdentityValid = isRecord(value) && (
    (value.vaeManifestSha256 === ACE_OPT_0028_VAE_FP16_MANIFEST_SHA256 &&
      value.vaeManifestByteLength ===
        ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_PROFILE.manifestByteLength &&
      ((value.vaeRuntimeProfile ===
          ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_PROFILE.id &&
        value.vaeKernelSetId ===
          ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_PROFILE.kernelSetId &&
        value.vaePrecisionMapSha256 ===
          ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_PROFILE.precisionMapSha256) ||
        (value.vaeRuntimeProfile ===
            ACE_OPT_0040_VAE_FP16_FIXED32_SHAPE_SELECTED_PROFILE.id &&
          value.vaeKernelSetId ===
            ACE_OPT_0040_VAE_FP16_FIXED32_SHAPE_SELECTED_PROFILE.kernelSetId &&
          value.vaePrecisionMapSha256 ===
            ACE_OPT_0040_VAE_FP16_FIXED32_SHAPE_SELECTED_PROFILE.precisionMapSha256))) ||
    (value.vaeManifestSha256 === ACE_OPT_0054_VAE_REVISION7_MANIFEST_SHA256 &&
      value.vaeManifestByteLength ===
        ACE_OPT_0054_VAE_FP16_FIXED32_REVISION7_PROFILE.manifestByteLength &&
      ((value.vaeRuntimeProfile ===
          ACE_OPT_0054_VAE_FP16_FIXED32_REVISION7_PROFILE.id &&
        value.vaeKernelSetId ===
          ACE_OPT_0054_VAE_FP16_FIXED32_REVISION7_PROFILE.kernelSetId &&
        value.vaePrecisionMapSha256 ===
          ACE_OPT_0054_VAE_FP16_FIXED32_REVISION7_PROFILE.precisionMapSha256) ||
        (value.vaeRuntimeProfile ===
            ACE_OPT_0072_VAE_FP16_FIXED32_DUAL_K4_PRODUCTION_RUNTIME_PROFILE &&
          ((value.vaeKernelSetId ===
              ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PROFILE.kernelSetId &&
            value.vaePrecisionMapSha256 ===
              ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PROFILE.precisionMapSha256) ||
            (value.vaeKernelSetId ===
                ACE_OPT_0088_VAE_FP16_PORTABLE_DUAL_K4_PROFILE.kernelSetId &&
              value.vaePrecisionMapSha256 ===
                ACE_OPT_0088_VAE_FP16_PORTABLE_DUAL_K4_PROFILE.precisionMapSha256)))))
  );
  // OPT-0088 coherence: the portable kernel-set identities appear exactly
  // when the portable execution profile ran, and then all of them together.
  // A fixed32/portable mixture is rejected in both directions.
  const portableExecutionProfile = isRecord(value) &&
    isRecord(value.executionProfile) &&
    value.executionProfile.id === ACE_REFERENCE_PORTABLE_PROFILE.id;
  const portableKernelIdentityCoherent = isRecord(value) && (
    portableExecutionProfile
      ? value.ditDenseKernelSetId ===
          ACE_OPT_0088_DIT_DENSE_PORTABLE_KERNEL_SET_ID &&
        (value.ditAttentionKernelSetId === undefined ||
          value.ditAttentionKernelSetId ===
            ACE_OPT_0088_DIT_PORTABLE_ATTENTION_KERNEL_SET_ID) &&
        value.vaeKernelSetId ===
          ACE_OPT_0088_VAE_FP16_PORTABLE_DUAL_K4_PROFILE.kernelSetId
      : value.ditDenseKernelSetId !==
          ACE_OPT_0088_DIT_DENSE_PORTABLE_KERNEL_SET_ID &&
        value.ditAttentionKernelSetId !==
          ACE_OPT_0088_DIT_PORTABLE_ATTENTION_KERNEL_SET_ID &&
        value.vaeKernelSetId !==
          ACE_OPT_0088_VAE_FP16_PORTABLE_DUAL_K4_PROFILE.kernelSetId
  );
  if (
    !isRecord(value) ||
    !hasOnlyKeys(value, [
      "backend",
      "modelManifestId",
      "modelManifestUrl",
      "modelManifestSha256",
      "ditDenseManifestId",
      "ditDenseManifestUrl",
      "ditDenseManifestSha256",
      "ditDenseManifestByteLength",
      "ditDenseRuntimeProfile",
      "ditDenseKernelSetId",
      "ditAttentionRuntimeProfile",
      "ditAttentionKernelSetId",
      "ditDenseLayerBytes",
      "ditResidentWeightBytes",
      "vaeManifestId",
      "vaeManifestUrl",
      "vaeManifestSha256",
      "vaeManifestByteLength",
      "vaeRuntimeProfile",
      "vaeKernelSetId",
      "vaePrecisionMapSha256",
      "vaeWindowRuntimeProfile",
      "vaeMaxWindowFrames",
      "executionProfile",
      "schedulingProfile",
      "capabilities",
      "aceSourceRevision",
      "plannerSourceRevision",
      "parakeetReferenceRevision",
    ]) ||
    value.backend !== "custom-webgpu-wgsl-and-wasm" ||
    !isNonEmptyString(value.modelManifestId) ||
    !isNonEmptyString(value.modelManifestUrl) ||
    !isSha256(value.modelManifestSha256) ||
    !isNonEmptyString(value.ditDenseManifestId) ||
    !isNonEmptyString(value.ditDenseManifestUrl) ||
    !ditDenseIdentityValid ||
    !ditAttentionIdentityValid ||
    (value.ditAttentionRuntimeProfile !== undefined &&
      value.ditDenseRuntimeProfile !==
        ACE_OPT_0009_DIT_DENSE_RUNTIME_PROFILE) ||
    !isNonEmptyString(value.vaeManifestId) ||
    !isNonEmptyString(value.vaeManifestUrl) ||
    !vaeIdentityValid ||
    !vaeWindowIdentityValid ||
    !portableKernelIdentityCoherent ||
    ((value.ditAttentionRuntimeProfile ===
        ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE) !==
      (value.vaeWindowRuntimeProfile ===
        ACE_OPT_0070_VAE_C2378_WINDOW_RUNTIME_PROFILE)) ||
    ((value.vaeRuntimeProfile ===
        ACE_OPT_0072_VAE_FP16_FIXED32_DUAL_K4_PRODUCTION_RUNTIME_PROFILE) !==
      (value.vaeWindowRuntimeProfile ===
        ACE_OPT_0070_VAE_C2378_WINDOW_RUNTIME_PROFILE)) ||
    !isExecutionProfile(value.executionProfile) ||
    !isSchedulingProfile(value.schedulingProfile) ||
    !isWebGpuCapabilityReport(value.capabilities) ||
    value.aceSourceRevision !== ACE_SOURCE_REVISION ||
    value.plannerSourceRevision !== ACE_PLANNER_SOURCE_REVISION ||
    value.parakeetReferenceRevision !== PARAKEET_REFERENCE_REVISION
  ) {
    return false;
  }
  return (
    value.capabilities.executionProfile.id === value.executionProfile.id &&
    value.capabilities.schedulingProfile === value.schedulingProfile
  );
}

const EXECUTION_PROFILES = [
  ACE_REFERENCE_PORTABLE_PROFILE,
  ACE_REFERENCE_SUBGROUP_PROFILE,
  ACE_FP16_PORTABLE_PROFILE,
] as const;

function isExecutionProfile(value: unknown): value is AceExecutionProfile {
  if (
    !isRecord(value) ||
    !hasExactKeys(value, [
      "id",
      "modelProfile",
      "weightStorage",
      "matrixArithmetic",
      "sensitiveReductions",
      "vaeArithmetic",
      "kernelBackend",
      "requiredFeatures",
    ]) ||
    !Array.isArray(value.requiredFeatures)
  ) {
    return false;
  }
  return EXECUTION_PROFILES.some(
    (profile) =>
      value.id === profile.id &&
      value.modelProfile === profile.modelProfile &&
      value.weightStorage === profile.weightStorage &&
      value.matrixArithmetic === profile.matrixArithmetic &&
      value.sensitiveReductions === profile.sensitiveReductions &&
      value.vaeArithmetic === profile.vaeArithmetic &&
      value.kernelBackend === profile.kernelBackend &&
      arraysEqual(
        value.requiredFeatures as readonly unknown[],
        profile.requiredFeatures,
      ),
  );
}

function isWebGpuCapabilityReport(
  value: unknown,
): value is AceWebGpuCapabilityReport {
  if (
    !isRecord(value) ||
    !hasExactKeys(value, [
      "executionProfile",
      "schedulingProfile",
      "adapterInfo",
      "adapterFeatures",
      "deviceFeatures",
      "stockFeatures",
      "requiredFeatures",
      "requestedOptionalFeatures",
      "adapterLimits",
      "deviceLimits",
      "requestedLimits",
    ]) ||
    !isExecutionProfile(value.executionProfile) ||
    !isSchedulingProfile(value.schedulingProfile) ||
    !isAdapterInfo(value.adapterInfo) ||
    !isStringArray(value.adapterFeatures) ||
    !isStringArray(value.deviceFeatures) ||
    !isStockFeatureReport(value.stockFeatures) ||
    !isRequiredFeatureArray(value.requiredFeatures) ||
    !isRequestedOptionalFeatureArray(value.requestedOptionalFeatures) ||
    !isWebGpuLimits(value.adapterLimits) ||
    !isWebGpuLimits(value.deviceLimits) ||
    !isRequestedWebGpuLimits(value.requestedLimits)
  ) {
    return false;
  }
  const report = value as unknown as AceWebGpuCapabilityReport;
  return report.executionProfile.requiredFeatures.every((feature) =>
    report.requiredFeatures.includes(feature)
  ) && ACE_STOCK_WEBGPU_FEATURES.every((feature) => {
    const status = report.stockFeatures[feature];
    const adapterSupported = report.adapterFeatures.includes(feature);
    const deviceEnabled = report.deviceFeatures.includes(feature);
    const required =
      feature !== "timestamp-query" &&
      report.requiredFeatures.includes(feature);
    const requested =
      required ||
      (feature === "timestamp-query" &&
        report.requestedOptionalFeatures.includes(feature));
    return status.adapterSupported === adapterSupported &&
      status.deviceEnabled === deviceEnabled &&
      status.required === required &&
      status.requested === requested &&
      (!required || adapterSupported && deviceEnabled);
  });
}

function isAdapterInfo(value: unknown): boolean {
  return (
    isRecord(value) &&
    hasOnlyKeys(value, [
      "vendor",
      "architecture",
      "device",
      "description",
      "subgroupMinSize",
      "subgroupMaxSize",
      "isFallbackAdapter",
    ]) &&
    typeof value.vendor === "string" &&
    typeof value.architecture === "string" &&
    typeof value.device === "string" &&
    typeof value.description === "string" &&
    (value.subgroupMinSize === undefined ||
      isPositiveSafeInteger(value.subgroupMinSize)) &&
    (value.subgroupMaxSize === undefined ||
      isPositiveSafeInteger(value.subgroupMaxSize)) &&
    (value.isFallbackAdapter === undefined ||
      typeof value.isFallbackAdapter === "boolean")
  );
}

function isStockFeatureReport(value: unknown): boolean {
  return (
    isRecord(value) &&
    hasExactKeys(value, ACE_STOCK_WEBGPU_FEATURES) &&
    ACE_STOCK_WEBGPU_FEATURES.every((feature) =>
      isFeatureStatus(value[feature]),
    )
  );
}

function isFeatureStatus(value: unknown): boolean {
  return (
    isRecord(value) &&
    hasExactKeys(value, [
      "adapterSupported",
      "deviceEnabled",
      "required",
      "requested",
    ]) &&
    typeof value.adapterSupported === "boolean" &&
    typeof value.deviceEnabled === "boolean" &&
    typeof value.required === "boolean" &&
    typeof value.requested === "boolean"
  );
}

function isRequiredFeatureArray(value: unknown): value is string[] {
  return (
    Array.isArray(value) &&
    value.every((feature) => feature === "shader-f16" || feature === "subgroups") &&
    new Set(value).size === value.length
  );
}

function isRequestedOptionalFeatureArray(value: unknown): boolean {
  return (
    Array.isArray(value) &&
    value.every((feature) => feature === "timestamp-query") &&
    new Set(value).size === value.length
  );
}

function isWebGpuLimits(value: unknown): value is AceWebGpuLimits {
  return (
    isRecord(value) &&
    hasExactKeys(value, ACE_WEBGPU_LIMIT_NAMES) &&
    ACE_WEBGPU_LIMIT_NAMES.every((name) => isRequestId(value[name]))
  );
}

function isRequestedWebGpuLimits(value: unknown): boolean {
  return (
    isRecord(value) &&
    hasOnlyKeys(value, ACE_WEBGPU_LIMIT_NAMES) &&
    Object.values(value).every(isRequestId)
  );
}

function isGenerationMetrics(
  value: unknown,
  frameCount: number,
  durationSeconds: number,
  diagnostics: AceRuntimeDiagnostics,
): boolean {
  if (
    !isRecord(value) ||
    !hasOnlyKeys(value, [
      "totalMs",
      "stageTimings",
      "peakTrackedGpuBytes",
      "cooperativeGpuQueueDrains",
      "cooperativeIdleMs",
      "vaeScheduling",
    ]) ||
    !isNonNegativeFinite(value.totalMs) ||
    !Array.isArray(value.stageTimings) ||
    !value.stageTimings.every(isStageTiming) ||
    (value.peakTrackedGpuBytes !== null &&
      !isRequestId(value.peakTrackedGpuBytes)) ||
    !isRequestId(value.cooperativeGpuQueueDrains) ||
    !isNonNegativeFinite(value.cooperativeIdleMs)
  ) return false;
  return value.vaeScheduling === undefined ||
    isVaeSchedulingReceipt(
      value.vaeScheduling,
      frameCount,
      durationSeconds,
      diagnostics,
      value.cooperativeGpuQueueDrains,
      value.cooperativeIdleMs,
    );
}

function isVaeSchedulingReceipt(
  value: unknown,
  frameCount: number,
  durationSeconds: number,
  diagnostics: AceRuntimeDiagnostics,
  aggregateQueueDrains: number,
  aggregateCooperativeIdleMs: number,
): boolean {
  if (
    !isRecord(value) ||
    !hasExactKeys(value, [
      "schema",
      "selectedProductionPolicy",
      "benchmarkPolicyOverride",
      "windows",
    ]) ||
    value.schema !== "ace-vae-window-scheduling-receipt-v1" ||
    (value.selectedProductionPolicy !== null &&
      value.selectedProductionPolicy !==
        "opt-0080-c2314-depth2-phase-epoch4") ||
    value.benchmarkPolicyOverride !== null ||
    !Array.isArray(value.windows) ||
    value.windows.length === 0
  ) return false;
  if (
    value.selectedProductionPolicy !== null &&
    !isOpt0080ProductionVaeSchedulingDiagnostics(diagnostics)
  ) return false;

  let expectedDispatchPlan:
    ReturnType<typeof planAceOpt0011Fp16VaeChunkDispatches>;
  try {
    const shape = deriveAceDurationGraphShape(durationSeconds);
    if (shape.audioFramesPerChannel !== frameCount) return false;
    const maximumWindowFrames = diagnostics.vaeMaxWindowFrames === 512
      ? 512 as const
      : diagnostics.vaeMaxWindowFrames ===
          ACE_OPT_0070_VAE_C2378_MAXIMUM_WINDOW_FRAMES
        ? ACE_OPT_0070_VAE_C2378_MAXIMUM_WINDOW_FRAMES
        : undefined;
    if (maximumWindowFrames === undefined) return false;
    expectedDispatchPlan = planAceOpt0011Fp16VaeChunkDispatches(
      shape.latentFrames,
      maximumWindowFrames,
      256,
    );
  } catch {
    return false;
  }
  const expectedPlan = expectedDispatchPlan.chunkPlan;
  if (
    value.windows.length !== expectedPlan.windows.length ||
    !value.windows.every((window, index) => {
      const topologyIndex = expectedDispatchPlan.windowTopologyIndices[index];
      const topology = topologyIndex === undefined
        ? undefined
        : expectedDispatchPlan.topologies[topologyIndex];
      const plannedWindow = expectedPlan.windows[index];
      return topology !== undefined && plannedWindow !== undefined &&
        isVaeWindowSchedulingReceipt(
          window,
          index,
          plannedWindow.latentWindowFrames,
          topology.sequenceQuantumCount,
          value.selectedProductionPolicy,
        );
    })
  ) return false;
  const receiptWindows = value.windows as readonly Readonly<{
    queueDrains: number;
    cooperativeIdleTurns: number;
  }>[];
  const receiptQueueDrains = receiptWindows.reduce(
    (total, window) => total + window.queueDrains,
    0,
  );
  const receiptCooperativeIdleTurns = receiptWindows.reduce(
    (total, window) => total + window.cooperativeIdleTurns,
    0,
  );
  return Number.isSafeInteger(receiptQueueDrains) &&
    Number.isSafeInteger(receiptCooperativeIdleTurns) &&
    aggregateQueueDrains >= receiptQueueDrains &&
    aggregateCooperativeIdleMs >= receiptCooperativeIdleTurns;
}

function isVaeWindowSchedulingReceipt(
  value: unknown,
  expectedIndex: number,
  expectedLatentWindowFrames: number,
  expectedDecoderQuantumCount: number,
  productionPolicy: unknown,
): boolean {
  if (
    !isRecord(value) ||
    !hasExactKeys(value, [
      "windowIndex",
      "latentWindowFrames",
      "selection",
      "schedulingProfile",
      "decoderQuantumCount",
      "quantaPerCommandBuffer",
      "decoderCommandBufferCount",
      "readbackCommandBufferCount",
      "totalCommandBufferCount",
      "commandBuffersSubmitted",
      "queueDrains",
      "cooperativeIdleTurns",
      "maximumOutstandingCommandBuffers",
    ]) ||
    value.windowIndex !== expectedIndex ||
    value.latentWindowFrames !== expectedLatentWindowFrames ||
    value.decoderQuantumCount !== expectedDecoderQuantumCount ||
    value.quantaPerCommandBuffer !== 64 ||
    !isPositiveSafeInteger(value.decoderCommandBufferCount) ||
    value.readbackCommandBufferCount !== 1 ||
    value.totalCommandBufferCount !== value.decoderCommandBufferCount + 1 ||
    value.commandBuffersSubmitted !== value.totalCommandBufferCount ||
    !isPositiveSafeInteger(value.queueDrains) ||
    !isRequestId(value.cooperativeIdleTurns) ||
    (value.maximumOutstandingCommandBuffers !== 1 &&
      value.maximumOutstandingCommandBuffers !== 2)
  ) return false;
  const expectedSelection = productionPolicy !== null
    ? "production"
    : "depth1-default";
  const candidatePolicy = productionPolicy ===
    "opt-0080-c2314-depth2-phase-epoch4";
  const expectedProfile = candidatePolicy && value.latentWindowFrames === 2_314
    ? "depth2-phase-epoch4"
    : "depth1-epoch1";
  if (
    value.selection !== expectedSelection ||
    value.schedulingProfile !== expectedProfile
  ) return false;
  if (expectedProfile === "depth2-phase-epoch4") {
    return value.decoderQuantumCount === 35_498 &&
      value.decoderCommandBufferCount === 555 &&
      value.totalCommandBufferCount === 556 &&
      value.queueDrains === 139 &&
      value.cooperativeIdleTurns === 138 &&
      value.maximumOutstandingCommandBuffers === 2;
  }
  return value.decoderCommandBufferCount ===
      Math.ceil(value.decoderQuantumCount / 64) &&
    value.queueDrains === value.totalCommandBufferCount &&
    value.cooperativeIdleTurns === value.totalCommandBufferCount - 1 &&
    value.maximumOutstandingCommandBuffers === 1;
}

function isOpt0080ProductionVaeSchedulingDiagnostics(
  value: AceRuntimeDiagnostics,
): boolean {
  const shader = value.capabilities.stockFeatures["shader-f16"];
  const subgroups = value.capabilities.stockFeatures.subgroups;
  return value.modelManifestSha256 === ACE_REFERENCE_MANIFEST_SHA256 &&
    value.ditDenseManifestSha256 ===
      ACE_OPT_0009_DIT_DENSE_MANIFEST_SHA256 &&
    value.ditDenseRuntimeProfile === ACE_OPT_0009_DIT_DENSE_RUNTIME_PROFILE &&
    value.ditAttentionRuntimeProfile ===
      ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE &&
    value.ditAttentionKernelSetId ===
      ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_KERNEL_SET_ID &&
    value.vaeManifestSha256 === ACE_OPT_0054_VAE_REVISION7_MANIFEST_SHA256 &&
    value.vaeRuntimeProfile ===
      ACE_OPT_0072_VAE_FP16_FIXED32_DUAL_K4_PRODUCTION_RUNTIME_PROFILE &&
    value.vaeKernelSetId ===
      ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PROFILE.kernelSetId &&
    value.vaePrecisionMapSha256 ===
      ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PROFILE
        .precisionMapSha256 &&
    value.vaeWindowRuntimeProfile ===
      ACE_OPT_0070_VAE_C2378_WINDOW_RUNTIME_PROFILE &&
    value.vaeMaxWindowFrames ===
      ACE_OPT_0070_VAE_C2378_MAXIMUM_WINDOW_FRAMES &&
    value.executionProfile.id === ACE_REFERENCE_SUBGROUP_PROFILE.id &&
    value.schedulingProfile === "cooperative" &&
    value.capabilities.adapterInfo.subgroupMinSize === 32 &&
    value.capabilities.adapterInfo.subgroupMaxSize === 32 &&
    value.capabilities.requiredFeatures.includes("shader-f16") &&
    value.capabilities.requiredFeatures.includes("subgroups") &&
    value.capabilities.adapterFeatures.includes("shader-f16") &&
    value.capabilities.adapterFeatures.includes("subgroups") &&
    value.capabilities.deviceFeatures.includes("shader-f16") &&
    value.capabilities.deviceFeatures.includes("subgroups") &&
    shader.adapterSupported && shader.deviceEnabled &&
    shader.required && shader.requested &&
    subgroups.adapterSupported && subgroups.deviceEnabled &&
    subgroups.required && subgroups.requested;
}

function isStageTiming(value: unknown): value is AceStageTiming {
  return (
    isRecord(value) &&
    hasOnlyKeys(value, [
      "stage",
      "wallMs",
      "submittedGpuMs",
      "cooperativeIdleMs",
    ]) &&
    isAceGenerationStage(value.stage) &&
    isNonNegativeFinite(value.wallMs) &&
    (value.submittedGpuMs === undefined ||
      isNonNegativeFinite(value.submittedGpuMs)) &&
    (value.cooperativeIdleMs === undefined ||
      isNonNegativeFinite(value.cooperativeIdleMs))
  );
}

function isDiagnosticDetails(value: unknown): boolean {
  return (
    isRecord(value) &&
    Object.values(value).every((detail) => {
      if (isDiagnosticScalar(detail)) return true;
      return Array.isArray(detail) && detail.every(isDiagnosticScalar);
    })
  );
}

function isDiagnosticScalar(
  value: unknown,
): value is string | number | boolean | null {
  return (
    value === null ||
    typeof value === "string" ||
    typeof value === "boolean" ||
    (typeof value === "number" && Number.isFinite(value))
  );
}

function isSchedulingProfile(value: unknown): value is AceSchedulingProfile {
  return value === "cooperative" || value === "benchmark";
}

function isStringArray(value: unknown): value is string[] {
  return Array.isArray(value) && value.every((item) => typeof item === "string");
}

function arraysEqual(
  left: readonly unknown[],
  right: readonly unknown[],
): boolean {
  return (
    left.length === right.length &&
    left.every((item, index) => item === right[index])
  );
}

function isNonEmptyString(value: unknown): value is string {
  return typeof value === "string" && value.length > 0;
}

function isSha256(value: unknown): value is string {
  return typeof value === "string" && /^[0-9a-f]{64}$/.test(value);
}

function hasOnlyKeys(
  value: Record<string, unknown>,
  allowed: readonly string[],
): boolean {
  const keySet = new Set(allowed);
  return Object.keys(value).every((key) => keySet.has(key));
}

function hasExactKeys(
  value: Record<string, unknown>,
  expected: readonly string[],
): boolean {
  return Object.keys(value).length === expected.length && hasOnlyKeys(value, expected);
}

function isRequestId(value: unknown): value is number {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 0;
}

function isPositiveSafeInteger(value: unknown): value is number {
  return typeof value === "number" && Number.isSafeInteger(value) && value > 0;
}

function isNonNegativeFinite(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && value >= 0;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
