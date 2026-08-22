import type { AceDiagnostic, AceRuntimeDiagnostics } from "./runtime/diagnostics.js";
import { canonicalizeSeed, isAceSeed, type AceSeed } from "./runtime/seed.js";
import type { AceGenerationProgress, AceStageTiming } from "./runtime/stages.js";
import {
  inspectWebGpuSupport,
  type AceModelProfileId,
  type AceSupportReport,
} from "./webgpu/capabilities.js";

export const ACE_SAMPLE_RATE_HZ = 48_000;
export const ACE_CHANNEL_COUNT = 2;
export const ACE_TURBO_DENOISING_EVALUATIONS = 8;
export const ACE_MIN_DURATION_SECONDS = 10;
export const ACE_MAX_DURATION_SECONDS = 240;

export const ACE_GENERATION_PROFILE_IDS = [
  "ace-turbo-v1-correctness",
] as const;

export type AceGenerationProfileId =
  (typeof ACE_GENERATION_PROFILE_IDS)[number];

export interface AceDynamicConditionalWeightingConfiguration {
  readonly enabled: true;
  readonly mode: "double";
  readonly wavelet: "haar";
  readonly lowBandScale: number;
  readonly highBandScale: number;
}

export const ACE_DIRECT_DCW_CONFIGURATION: Readonly<AceDynamicConditionalWeightingConfiguration> =
  Object.freeze({
    enabled: true,
    mode: "double",
    wavelet: "haar",
    lowBandScale: 0.05,
    highBandScale: 0.02,
  });

export const ACE_THINKING_DCW_CONFIGURATION: Readonly<AceDynamicConditionalWeightingConfiguration> =
  Object.freeze({
    enabled: true,
    mode: "double",
    wavelet: "haar",
    lowBandScale: 0.02,
    highBandScale: 0.06,
  });

/**
 * Output-affecting math fixed by the initial correctness contract. Changing
 * any field requires a new profile identifier and new golden/listening data.
 */
export const ACE_TURBO_V1_CORRECTNESS_PROFILE = Object.freeze({
  id: "ace-turbo-v1-correctness" as const,
  inferenceMethod: "ode" as const,
  sampler: "euler" as const,
  denoisingEvaluations: ACE_TURBO_DENOISING_EVALUATIONS,
  diffusionGuidanceScale: 1 as const,
  // The generic GenerationParams dataclass says 1.0, but the pinned Turbo
  // model, Gradio product path, and release API all resolve to shift 3.0.
  shift: 3 as const,
  schedulerTimesteps: Object.freeze([
    1.0,
    0.9545454545454546,
    0.9,
    0.8333333333333334,
    0.75,
    0.6428571428571429,
    0.5,
    0.3,
  ]),
  // Pinned Turbo materializes the resolved list with
  // `dtype=context_latents.dtype` before calling `.item()` for the model,
  // Euler update, and DCW. The accepted native oracle is BF16, so these are
  // the actual sampler coefficients; both browser profiles preserve them.
  effectiveSamplerTimestepsBfloat16: Object.freeze([
    1.0,
    0.953125,
    0.8984375,
    0.83203125,
    0.75,
    0.64453125,
    0.5,
    0.30078125,
  ]),
  customTimesteps: null,
  adaptiveDualGuidance: false as const,
  velocityNormThreshold: 0 as const,
  velocityEmaFactor: 0 as const,
  latentShift: 0 as const,
  latentRescale: 1 as const,
  dynamicConditionalWeighting: Object.freeze({
    resolution: "planner-mode" as const,
    direct: ACE_DIRECT_DCW_CONFIGURATION,
    thinking: ACE_THINKING_DCW_CONFIGURATION,
  }),
  outputNormalization: Object.freeze({
    mode: "global-peak" as const,
    targetDbfs: -1,
    silenceThreshold: 1e-6,
  }),
  sampleRateHz: ACE_SAMPLE_RATE_HZ,
  channelCount: ACE_CHANNEL_COUNT,
});

export interface AceMusicMetadata {
  readonly bpm?: number;
  readonly keyScale?: string;
  readonly timeSignature?: string;
  readonly vocalLanguage?: string;
}

export interface AcePlannerDisabled {
  readonly mode: "disabled";
}

export interface AcePlannerEnabled {
  readonly mode: "enabled";
  readonly temperature: number;
  readonly guidanceScale: number;
  /** Zero disables top-k filtering, matching the pinned upstream interface. */
  readonly topK: number;
  readonly topP: number;
  readonly constrainedDecoding: true;
  readonly generateSemanticCodes: true;
  readonly negativePrompt: "NO USER INPUT";
  readonly thinking: {
    readonly enabled: true;
    readonly useCotCaption: boolean;
    readonly useCotLanguage: boolean;
    readonly useCotMissingMetadata: boolean;
  };
}

export type AcePlannerConfiguration = AcePlannerDisabled | AcePlannerEnabled;

/** Resolve the pinned Gradio DCW defaults without an implicit UI-side toggle. */
export function resolveAceDynamicConditionalWeighting(
  planner: AcePlannerConfiguration,
): Readonly<AceDynamicConditionalWeightingConfiguration> {
  return planner.mode === "enabled"
    ? ACE_THINKING_DCW_CONFIGURATION
    : ACE_DIRECT_DCW_CONFIGURATION;
}

const PLANNER_CONFIGURATION_KEYS = [
  "mode",
  "temperature",
  "guidanceScale",
  "topK",
  "topP",
  "constrainedDecoding",
  "generateSemanticCodes",
  "negativePrompt",
  "thinking",
] as const;

const GENERATION_REQUEST_KEYS = [
  "generationProfile",
  "prompt",
  "lyrics",
  "instrumental",
  "durationSeconds",
  "seed",
  "planner",
  "metadata",
] as const;

const PLANNER_THINKING_KEYS = [
  "enabled",
  "useCotCaption",
  "useCotLanguage",
  "useCotMissingMetadata",
] as const;

const MUSIC_METADATA_KEYS = [
  "bpm",
  "keyScale",
  "timeSignature",
  "vocalLanguage",
] as const;

export const DEFAULT_ACE_PLANNER_CONFIGURATION: Readonly<AcePlannerEnabled> =
  Object.freeze({
    mode: "enabled",
    temperature: 0.85,
    guidanceScale: 2.0,
    topK: 0,
    topP: 0.9,
    constrainedDecoding: true,
    generateSemanticCodes: true,
    negativePrompt: "NO USER INPUT",
    thinking: Object.freeze({
      enabled: true,
      useCotCaption: true,
      useCotLanguage: true,
      useCotMissingMetadata: true,
    }),
  });

/**
 * Fully resolved, structured-clone-safe v1 generation request.
 *
 * Seed is mandatory. Callers that want a random seed must obtain it explicitly
 * from a cryptographically secure source and canonicalize it before creating
 * this request; inference never falls back to `Math.random()`.
 */
export interface AceGenerationRequest {
  readonly generationProfile: AceGenerationProfileId;
  readonly prompt: string;
  readonly lyrics?: string;
  /** Explicitly controls instrumental conditioning; never inferred from lyrics. */
  readonly instrumental: boolean;
  readonly durationSeconds: number;
  readonly seed: AceSeed;
  readonly planner: AcePlannerConfiguration;
  readonly metadata?: AceMusicMetadata;
}

export interface AceGenerationMetrics {
  readonly totalMs: number;
  readonly stageTimings: readonly AceStageTiming[];
  readonly peakTrackedGpuBytes: number | null;
  readonly cooperativeGpuQueueDrains: number;
  readonly cooperativeIdleMs: number;
  /**
   * Timing-free production receipt. The WebGPU runtime emits it without
   * enabling profiling, retaining audio, or changing queue submission.
   */
  readonly vaeScheduling?: AceVaeSchedulingReceipt;
}

export type AceVaeSchedulingProfile =
  | "depth1-epoch1"
  | "depth2-phase-epoch4";

export type AceVaeSchedulingSelection =
  | "depth1-default"
  | "production"
  | "benchmark-override";

export interface AceVaeWindowSchedulingReceipt {
  readonly windowIndex: number;
  readonly latentWindowFrames: number;
  readonly selection: AceVaeSchedulingSelection;
  readonly schedulingProfile: AceVaeSchedulingProfile;
  readonly decoderQuantumCount: number;
  readonly quantaPerCommandBuffer: number;
  readonly decoderCommandBufferCount: number;
  readonly readbackCommandBufferCount: 1;
  readonly totalCommandBufferCount: number;
  readonly commandBuffersSubmitted: number;
  readonly queueDrains: number;
  readonly cooperativeIdleTurns: number;
  readonly maximumOutstandingCommandBuffers: 1 | 2;
}

export interface AceVaeSchedulingReceipt {
  readonly schema: "ace-vae-window-scheduling-receipt-v1";
  readonly selectedProductionPolicy:
    | "opt-0080-c2314-depth2-phase-epoch4"
    | null;
  readonly benchmarkPolicyOverride:
    | "depth1-epoch1"
    | "opt-0080-c2314-depth2-phase-epoch4"
    | null;
  readonly windows: readonly AceVaeWindowSchedulingReceipt[];
}

export interface AceGenerationResult {
  /** Stereo 48 kHz IEEE-float WAV. Blob transfer does not materialize it in WASM. */
  readonly audio: Blob;
  /**
   * Exact OPFS job identity backing `audio`. The UI must release it only after
   * playback/download no longer needs the Blob snapshot.
   */
  readonly audioStorageId: string;
  readonly mimeType: "audio/wav";
  readonly sampleRateHz: typeof ACE_SAMPLE_RATE_HZ;
  readonly channelCount: typeof ACE_CHANNEL_COUNT;
  readonly frameCount: number;
  readonly durationSeconds: number;
  readonly seed: AceSeed;
  readonly generationProfile: AceGenerationProfileId;
  readonly modelManifestId: string;
  readonly modelManifestSha256: string;
  readonly diagnostics: AceRuntimeDiagnostics;
  readonly metrics: AceGenerationMetrics;
}

export interface AceGenerateOptions {
  readonly signal?: AbortSignal;
  readonly onProgress?: (progress: AceGenerationProgress) => void;
  readonly onDiagnostic?: (diagnostic: AceDiagnostic) => void;
}

export interface AceSupportOptions {
  /** The correctness profile is the Stage 1 default even when FP16 is exposed. */
  readonly modelProfile?: AceModelProfileId;
}

export async function checkSupport(
  options: AceSupportOptions = {},
): Promise<AceSupportReport> {
  return await inspectWebGpuSupport(options.modelProfile ?? "reference-bf16");
}

/** Validate the stable request at API and worker trust boundaries. */
export function assertAceGenerationRequest(
  request: AceGenerationRequest,
): void {
  if (!isRecord(request)) {
    throw new TypeError("ACE generation request must be an object");
  }
  assertOnlyKeys(request, GENERATION_REQUEST_KEYS, "ACE generation request");
  if (request.generationProfile !== ACE_TURBO_V1_CORRECTNESS_PROFILE.id) {
    throw new TypeError(`Unknown ACE generation profile ${String(request.generationProfile)}`);
  }
  if (typeof request.prompt !== "string" || request.prompt.trim().length === 0) {
    throw new TypeError("ACE generation requires a non-empty prompt");
  }
  if (request.prompt.length > 512) {
    throw new RangeError("ACE prompt cannot exceed 512 characters");
  }
  if (request.lyrics !== undefined && typeof request.lyrics !== "string") {
    throw new TypeError("ACE lyrics must be a string when supplied");
  }
  if (request.lyrics !== undefined && request.lyrics.length > 4096) {
    throw new RangeError("ACE lyrics cannot exceed 4096 characters");
  }
  if (typeof request.instrumental !== "boolean") {
    throw new TypeError("ACE instrumental must be explicitly true or false");
  }
  if (
    !Number.isInteger(request.durationSeconds) ||
    request.durationSeconds < ACE_MIN_DURATION_SECONDS ||
    request.durationSeconds > ACE_MAX_DURATION_SECONDS
  ) {
    throw new RangeError(
      `ACE durationSeconds must be an integer from ${ACE_MIN_DURATION_SECONDS} through ${ACE_MAX_DURATION_SECONDS}`,
    );
  }
  if (!isAceSeed(request.seed)) {
    throw new TypeError("ACE generation requires a canonical 64-bit seed");
  }
  assertPlannerConfiguration(request.planner);
  if (request.metadata !== undefined) assertMusicMetadata(request.metadata);
}

export function isAceGenerationRequest(
  value: unknown,
): value is AceGenerationRequest {
  if (!isRecord(value)) return false;
  try {
    assertAceGenerationRequest(value as unknown as AceGenerationRequest);
    return true;
  } catch {
    return false;
  }
}

/** Convenience for UI input while preserving the canonical serialized form. */
export function aceSeed(value: string | number | bigint): AceSeed {
  return canonicalizeSeed(value);
}

function assertPlannerConfiguration(
  planner: AcePlannerConfiguration,
): void {
  if (!isRecord(planner)) {
    throw new TypeError("ACE planner configuration is missing");
  }
  if (planner.mode === "disabled") {
    if (Object.keys(planner).some((key) => key !== "mode")) {
      throw new TypeError("Disabled planner configuration cannot contain hidden controls");
    }
    return;
  }
  if (planner.mode !== "enabled") {
    throw new TypeError("ACE planner mode must be enabled or disabled");
  }
  const allowed = new Set<string>(PLANNER_CONFIGURATION_KEYS);
  if (Object.keys(planner).some((key) => !allowed.has(key))) {
    throw new TypeError("Enabled planner configuration contains unknown controls");
  }
  if (!isPositiveFinite(planner.temperature)) {
    throw new RangeError("Planner temperature must be positive and finite");
  }
  if (!isPositiveFinite(planner.guidanceScale) || planner.guidanceScale < 1) {
    throw new RangeError("Planner guidanceScale must be at least one");
  }
  if (!Number.isSafeInteger(planner.topK) || planner.topK < 0) {
    throw new RangeError("Planner topK must be a non-negative integer");
  }
  if (!isPositiveFinite(planner.topP) || planner.topP > 1) {
    throw new RangeError("Planner topP must be in the interval (0, 1]");
  }
  if (
    planner.constrainedDecoding !== true ||
    planner.generateSemanticCodes !== true ||
    planner.negativePrompt !== "NO USER INPUT"
  ) {
    throw new TypeError("Planner constrained decoding contract is invalid");
  }
  if (
    !isRecord(planner.thinking) ||
    planner.thinking.enabled !== true ||
    typeof planner.thinking.useCotCaption !== "boolean" ||
    typeof planner.thinking.useCotLanguage !== "boolean" ||
    typeof planner.thinking.useCotMissingMetadata !== "boolean"
  ) {
    throw new TypeError("Planner thinking configuration must be fully resolved");
  }
  assertOnlyKeys(planner.thinking, PLANNER_THINKING_KEYS, "Planner thinking configuration");
}

function assertMusicMetadata(metadata: AceMusicMetadata): void {
  if (!isRecord(metadata)) {
    throw new TypeError("ACE metadata must be an object");
  }
  assertOnlyKeys(metadata, MUSIC_METADATA_KEYS, "ACE metadata");
  const bpm = metadata.bpm;
  if (
    bpm !== undefined &&
    (typeof bpm !== "number" ||
      !Number.isSafeInteger(bpm) ||
      bpm < 30 ||
      bpm > 300)
  ) {
    throw new RangeError(
      "ACE metadata bpm must be an integer in the interval [30, 300]",
    );
  }
  for (const [label, value] of [
    ["keyScale", metadata.keyScale],
    ["timeSignature", metadata.timeSignature],
    ["vocalLanguage", metadata.vocalLanguage],
  ] as const) {
    if (value !== undefined && typeof value !== "string") {
      throw new TypeError(`ACE metadata ${label} must be a string`);
    }
  }
}

function isPositiveFinite(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && value > 0;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function assertOnlyKeys(
  value: Record<string, unknown>,
  allowedKeys: readonly string[],
  label: string,
): void {
  const allowed = new Set(allowedKeys);
  if (Object.keys(value).some((key) => !allowed.has(key))) {
    throw new TypeError(`${label} contains unknown controls`);
  }
}
