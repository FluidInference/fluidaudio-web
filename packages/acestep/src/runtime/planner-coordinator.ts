import {
  ACE_MAX_DURATION_SECONDS,
  assertAceGenerationRequest,
  type AceGenerationRequest,
} from "../api.js";
import {
  AceGpuTensorPhase,
  type AceGpuTensorPhaseProgress,
} from "../model/gpu-tensors.js";
import type { AcePackageManifest } from "../model/manifest.js";
import {
  loadPinnedAceTokenizer,
  type AceTokenizerAssetBundle,
} from "../tokenizer/loader.js";
import { AceQwenBpeTokenizer } from "../tokenizer/qwen-bpe.js";
import type { AceModelProfileId } from "../webgpu/capabilities.js";
import {
  AcePlannerGpuExecutor,
  type AcePlannerGpuExecutorProgress,
} from "../webgpu/planner-executor.js";
import {
  createAcePlannerMetadataConstraintForPlan,
} from "./planner-metadata-fsm.js";
import {
  createAcePlannerRequestPlan,
  finalizeAcePlannerConditioning,
  runAcePlannerCotPhase,
  runAcePlannerSemanticPhase,
  type AcePlannerCotPhaseResult,
  type AcePlannerDecodeBatch,
  type AcePlannerFinalizedConditioning,
  type AcePlannerGraphExecutor,
  type AcePlannerLogitRange,
  type AcePlannerMetadata,
  type AcePlannerPrefillBatch,
  type AcePlannerSemanticPhaseResult,
} from "./planner.js";
import {
  ACE_PLANNER_SOFTMAX_ACCEPTANCE,
  AcePlannerSamplingCursor,
  type AcePlannerCursorSample,
} from "./planner-sampling.js";
import { canonicalizeAceGenerationRequest } from "./generation-inputs.js";

/**
 * Pinned CoT fallback used when upstream deliberately passes no target
 * duration to Phase 1: `DURATION_MAX * 5 + 500`.
 */
export const ACE_PLANNER_COT_MAX_NEW_TOKENS =
  ACE_MAX_DURATION_SECONDS * 5 + 500;

/** Production facts that must not be confused with a native-logit claim. */
export const ACE_PLANNER_COORDINATOR_CONTRACT = Object.freeze({
  id: "ace-planner-production-coordinator-v1",
  samplingOracleId: ACE_PLANNER_SOFTMAX_ACCEPTANCE.productionOracleId,
  samplingDistributionAuthority: "browser-defined" as const,
  torchLogitIdentityClaim: false,
  randomStream: "planner-sampling" as const,
  cotToSemanticTransition: "fresh-prefill-replaces-complete-kv-phase" as const,
  plannerDisabledBehavior: "explicit-boundary-bypass" as const,
});

export interface AcePlannerPackageResources {
  readonly kind: "package";
  readonly device: GPUDevice;
  readonly modelProfile: AceModelProfileId;
  readonly manifest: AcePackageManifest;
  readonly acquiredFiles: ReadonlyMap<string, File>;
  readonly tokenizerAssets: AceTokenizerAssetBundle;
}

export interface AcePlannerOwnedPhaseResources {
  readonly kind: "owned-phase";
  readonly device: GPUDevice;
  readonly modelProfile: AceModelProfileId;
  readonly tokenizer: AceQwenBpeTokenizer;
  /** Ownership transfers to the coordinator when execution begins. */
  readonly ownedPlannerWeights: AceGpuTensorPhase;
}

export type AcePlannerCoordinatorResources =
  | AcePlannerPackageResources
  | AcePlannerOwnedPhaseResources;

export interface AcePlannerCoordinatorOptions {
  readonly request: AceGenerationRequest;
  /** Required only when `request.planner.mode` is `enabled`. */
  readonly resources?: AcePlannerCoordinatorResources;
  readonly signal?: AbortSignal;
  readonly onProgress?: (progress: AcePlannerCoordinatorProgress) => void;
}

export type AcePlannerLogicalPhase = "cot" | "semantic";

export type AcePlannerCoordinatorProgress =
  | Readonly<{
      readonly kind: "boundary";
      readonly phase:
        | "planner-bypass"
        | "tokenizer-load"
        | "weight-load"
        | "cot"
        | "semantic"
        | "release";
      readonly state: "started" | "completed" | "skipped";
    }>
  | Readonly<{
      readonly kind: "weight-upload";
      readonly progress: AceGpuTensorPhaseProgress;
    }>
  | Readonly<{
      readonly kind: "gpu";
      readonly phase: AcePlannerLogicalPhase;
      readonly progress: AcePlannerGpuExecutorProgress;
    }>
  | Readonly<{
      readonly kind: "token";
      readonly phase: AcePlannerLogicalPhase;
      readonly completedTokens: number;
      readonly totalTokens: number;
      readonly sample: AcePlannerCursorSample;
    }>;

/** Fields handed to semantic conditioning and the DiT request builder. */
export interface AcePlannerResolvedDownstreamFields {
  readonly caption: string;
  readonly lyrics: string;
  readonly instrumental: boolean;
  readonly durationSeconds: number;
  readonly vocalLanguage: string;
  readonly metadata: AcePlannerMetadata;
}

export interface AcePlannerRuntimeMetrics {
  readonly peakAccountedGpuBytes: number;
  readonly queueDrains: number;
  readonly cooperativeIdleMs: number;
}

export interface AcePlannerBypassedCoordinatorResult {
  readonly plannerMode: "disabled";
  readonly bypassReason: "planner-disabled-by-request";
  readonly downstream: AcePlannerResolvedDownstreamFields;
  readonly conditioning: null;
  readonly cot: null;
  readonly semantic: null;
  readonly sampling: null;
  readonly runtime: AcePlannerRuntimeMetrics;
}

export interface AcePlannerEnabledCoordinatorResult {
  readonly plannerMode: "enabled";
  readonly downstream: AcePlannerResolvedDownstreamFields;
  readonly conditioning: AcePlannerFinalizedConditioning;
  readonly cot: AcePlannerCotPhaseResult | null;
  readonly semantic: AcePlannerSemanticPhaseResult;
  readonly sampling: Readonly<{
    readonly seed: AceGenerationRequest["seed"];
    readonly oracleId: typeof ACE_PLANNER_SOFTMAX_ACCEPTANCE.productionOracleId;
    readonly distributionAuthority: "browser-defined";
    readonly firstDraw: 0n;
    readonly finalDraw: bigint;
  }>;
  readonly runtime: AcePlannerRuntimeMetrics;
}

export type AcePlannerCoordinatorResult =
  | AcePlannerBypassedCoordinatorResult
  | AcePlannerEnabledCoordinatorResult;

/**
 * Run the complete optional production planner lifetime.
 *
 * The enabled path authenticates and uploads only planner tensors (or accepts
 * one exclusively owned authenticated planner phase), constructs the concrete
 * GPU executor, runs the real constrained metadata FSM, replaces the entire KV
 * phase with the semantic prefill, and destroys all planner GPU state before
 * returning. The disabled path returns before tokenizer/model acquisition.
 */
export async function runAcePlannerCoordinator(
  options: AcePlannerCoordinatorOptions,
): Promise<AcePlannerCoordinatorResult> {
  const transferredOwnedPhase = options.resources?.kind === "owned-phase"
    ? options.resources.ownedPlannerWeights
    : undefined;
  let request: AceGenerationRequest;
  try {
    assertAceGenerationRequest(options.request);
    request = canonicalizeAceGenerationRequest(options.request);
    options.signal?.throwIfAborted();

    if (request.planner.mode === "disabled") {
      if (options.resources !== undefined) {
        throw new TypeError(
          "ACE planner-disabled coordination must not receive planner resources",
        );
      }
      emit(options, {
        kind: "boundary",
        phase: "planner-bypass",
        state: "skipped",
      });
      return Object.freeze({
        plannerMode: "disabled",
        bypassReason: "planner-disabled-by-request",
        downstream: directDownstreamFields(request),
        conditioning: null,
        cot: null,
        semantic: null,
        sampling: null,
        runtime: ZERO_PLANNER_RUNTIME_METRICS,
      });
    }
  } catch (error) {
    transferredOwnedPhase?.destroy();
    throw error;
  }

  const resources = options.resources;
  if (resources === undefined) {
    throw new TypeError("ACE planner-enabled coordination requires planner resources");
  }

  // Owned-phase resources transfer at enabled-path entry. Holding the phase
  // here guarantees tokenizer validation/cancellation cannot leak it.
  let phase: AceGpuTensorPhase | undefined = resources.kind === "owned-phase"
    ? transferredOwnedPhase
    : undefined;
  let executor: AcePlannerGpuExecutor | undefined;
  let activeLogicalPhase: AcePlannerLogicalPhase | null = null;
  let peakAccountedGpuBytes = 0;
  let queueDrains = 0;
  let cooperativeIdleMs = 0;
  let result: AcePlannerEnabledCoordinatorResult | undefined;
  let primaryFailure: unknown;

  try {
    const tokenizer = await acquirePlannerTokenizer(resources, options);
    options.signal?.throwIfAborted();
    phase = await acquirePlannerPhase(resources, options);
    options.signal?.throwIfAborted();

    // AcePlannerGpuExecutor.create owns the phase at call entry, including if
    // construction fails. Clear our local owner before making that call.
    const transferredPhase = phase;
    phase = undefined;
    executor = AcePlannerGpuExecutor.create({
      device: resources.device,
      modelProfile: resources.modelProfile,
      ownedPlannerWeights: transferredPhase,
      ...(options.signal === undefined ? {} : { signal: options.signal }),
      onProgress: (progress) => {
        peakAccountedGpuBytes = Math.max(
          peakAccountedGpuBytes,
          progress.peakAccountedGpuBytes,
        );
        queueDrains = Math.max(queueDrains, progress.cumulativeQueueDrains);
        cooperativeIdleMs = Math.max(
          cooperativeIdleMs,
          progress.cumulativeCooperativeIdleMs,
        );
        const logicalPhase = activeLogicalPhase;
        if (logicalPhase === null) {
          throw new Error("ACE planner GPU progress arrived outside a logical phase");
        }
        emit(options, {
          kind: "gpu",
          phase: logicalPhase,
          progress,
        });
      },
    });

    const plan = createAcePlannerRequestPlan(request);
    const cursor = new AcePlannerSamplingCursor(request.seed);
    let cot: AcePlannerCotPhaseResult | null = null;

    if (plan.shouldRunCot) {
      emit(options, { kind: "boundary", phase: "cot", state: "started" });
      activeLogicalPhase = "cot";
      let completedTokens = 0;
      cot = await runAcePlannerCotPhase({
        graph: freshLogicalPhase(executor, "cot"),
        tokenizer,
        prompt: plan.cotPrompt!,
        cursor,
        constraint: createAcePlannerMetadataConstraintForPlan(tokenizer, plan),
        sampling: {
          temperature: plan.configuration.temperature,
          topK: plan.configuration.topK,
          topP: plan.configuration.topP,
          repetitionPenalty: 1,
        },
        maxNewTokens: ACE_PLANNER_COT_MAX_NEW_TOKENS,
        ...(options.signal === undefined ? {} : { signal: options.signal }),
        onToken: (sample) => {
          completedTokens += 1;
          emitToken(
            options,
            "cot",
            completedTokens,
            ACE_PLANNER_COT_MAX_NEW_TOKENS,
            sample,
          );
        },
      });
      activeLogicalPhase = null;
      options.signal?.throwIfAborted();
      emit(options, { kind: "boundary", phase: "cot", state: "completed" });
    } else {
      emit(options, { kind: "boundary", phase: "cot", state: "skipped" });
    }

    const conditioning = finalizeAcePlannerConditioning(
      plan,
      cot?.outputText ?? null,
    );
    options.signal?.throwIfAborted();

    // This new wrapper requires semantic generation to begin with a prefill.
    // The concrete executor therefore releases the entire CoT allocation and
    // clears a newly sized cache before any semantic decode is accepted.
    emit(options, { kind: "boundary", phase: "semantic", state: "started" });
    activeLogicalPhase = "semantic";
    const totalSemanticTokens = request.durationSeconds * 5 + 1;
    let completedSemanticTokens = 0;
    const semantic = await runAcePlannerSemanticPhase({
      graph: freshLogicalPhase(executor, "semantic"),
      tokenizer,
      conditionalPrompt: conditioning.conditionalCodePrompt,
      unconditionalPrompt: conditioning.unconditionalCodePrompt,
      cursor,
      sampling: {
        temperature: plan.configuration.temperature,
        guidanceScale: plan.configuration.guidanceScale,
        topK: plan.configuration.topK,
        topP: plan.configuration.topP,
      },
      durationSeconds: request.durationSeconds,
      ...(options.signal === undefined ? {} : { signal: options.signal }),
      onToken: (sample) => {
        completedSemanticTokens += 1;
        emitToken(
          options,
          "semantic",
          completedSemanticTokens,
          totalSemanticTokens,
          sample,
        );
      },
    });
    activeLogicalPhase = null;
    options.signal?.throwIfAborted();
    emit(options, {
      kind: "boundary",
      phase: "semantic",
      state: "completed",
    });

    result = Object.freeze({
      plannerMode: "enabled",
      downstream: Object.freeze({
        caption: conditioning.caption,
        lyrics: request.lyrics ?? "",
        instrumental: request.instrumental,
        durationSeconds: request.durationSeconds,
        vocalLanguage: conditioning.vocalLanguage,
        metadata: conditioning.metadata,
      }),
      conditioning,
      cot,
      semantic,
      sampling: Object.freeze({
        seed: request.seed,
        oracleId: ACE_PLANNER_SOFTMAX_ACCEPTANCE.productionOracleId,
        distributionAuthority: "browser-defined",
        firstDraw: 0n,
        finalDraw: cursor.consumed,
      }),
      runtime: Object.freeze({
        peakAccountedGpuBytes,
        queueDrains,
        cooperativeIdleMs,
      }),
    });
  } catch (error) {
    primaryFailure = error;
  }

  activeLogicalPhase = null;
  if (primaryFailure === undefined) {
    try {
      emit(options, { kind: "boundary", phase: "release", state: "started" });
    } catch (progressFailure) {
      primaryFailure = progressFailure;
    }
  }
  try {
    if (executor !== undefined) {
      await executor.destroy(primaryFailure);
    } else {
      phase?.destroy();
    }
  } catch (cleanupFailure) {
    primaryFailure ??= cleanupFailure;
  }
  if (primaryFailure === undefined) {
    try {
      emit(options, { kind: "boundary", phase: "release", state: "completed" });
    } catch (progressFailure) {
      primaryFailure = progressFailure;
    }
  }

  if (primaryFailure !== undefined) throw primaryFailure;
  if (result === undefined) {
    throw new Error("ACE planner coordinator completed without a result");
  }
  return result;
}

const ZERO_PLANNER_RUNTIME_METRICS: AcePlannerRuntimeMetrics = Object.freeze({
  peakAccountedGpuBytes: 0,
  queueDrains: 0,
  cooperativeIdleMs: 0,
});

async function acquirePlannerTokenizer(
  resources: AcePlannerCoordinatorResources,
  options: AcePlannerCoordinatorOptions,
): Promise<AceQwenBpeTokenizer> {
  emit(options, {
    kind: "boundary",
    phase: "tokenizer-load",
    state: "started",
  });
  const tokenizer = resources.kind === "package"
    ? (await loadPinnedAceTokenizer("planner", resources.tokenizerAssets)).tokenizer
    : resources.tokenizer;
  if (!(tokenizer instanceof AceQwenBpeTokenizer) || tokenizer.kind !== "planner") {
    throw new TypeError("ACE planner coordinator requires the pinned planner tokenizer");
  }
  options.signal?.throwIfAborted();
  emit(options, {
    kind: "boundary",
    phase: "tokenizer-load",
    state: "completed",
  });
  return tokenizer;
}

async function acquirePlannerPhase(
  resources: AcePlannerCoordinatorResources,
  options: AcePlannerCoordinatorOptions,
): Promise<AceGpuTensorPhase> {
  emit(options, { kind: "boundary", phase: "weight-load", state: "started" });
  let phase: AceGpuTensorPhase | undefined;
  try {
    phase = resources.kind === "package"
      ? await AceGpuTensorPhase.load(
          resources.device,
          resources.manifest,
          resources.acquiredFiles,
          ["planner"],
          {
            ...(options.signal === undefined ? {} : { signal: options.signal }),
            onProgress: (progress) => {
              emit(options, { kind: "weight-upload", progress });
            },
          },
        )
      : resources.ownedPlannerWeights;
    if (phase.phases.length !== 1 || phase.phases[0] !== "planner") {
      throw new Error(
        "ACE planner coordinator requires an exclusively resident planner phase",
      );
    }
    options.signal?.throwIfAborted();
    emit(options, { kind: "boundary", phase: "weight-load", state: "completed" });
    return phase;
  } catch (error) {
    // For package acquisition, the caller cannot see the phase until this
    // function returns. Destroy it if validation/progress publication fails.
    // Owned phases are idempotent and the outer owner will repeat this safely.
    phase?.destroy();
    throw error;
  }
}

/** Guard one logical phase so its first and only graph root is a fresh prefill. */
function freshLogicalPhase(
  graph: AcePlannerGraphExecutor,
  label: AcePlannerLogicalPhase,
): AcePlannerGraphExecutor {
  let state: "fresh" | "prefilled" = "fresh";
  return Object.freeze({
    async prefill(
      batch: AcePlannerPrefillBatch,
      logitRange?: AcePlannerLogitRange,
    ) {
      if (state !== "fresh") {
        throw new Error(`ACE planner ${label} phase attempted a second prefill`);
      }
      const logits = await graph.prefill(batch, logitRange);
      state = "prefilled";
      return logits;
    },
    async decode(
      batch: AcePlannerDecodeBatch,
      logitRange?: AcePlannerLogitRange,
    ) {
      if (state !== "prefilled") {
        throw new Error(`ACE planner ${label} phase decode preceded its fresh prefill`);
      }
      return await graph.decode(batch, logitRange);
    },
  });
}

function directDownstreamFields(
  request: AceGenerationRequest,
): AcePlannerResolvedDownstreamFields {
  const metadata: {
    bpm?: number;
    duration: number;
    keyscale?: string;
    timesignature?: string;
  } = { duration: request.durationSeconds };
  if (request.metadata?.bpm !== undefined) metadata.bpm = request.metadata.bpm;
  const keyscale = request.metadata?.keyScale?.trim();
  if (keyscale) metadata.keyscale = keyscale;
  const timesignature = request.metadata?.timeSignature?.trim();
  if (timesignature) metadata.timesignature = timesignature;
  return Object.freeze({
    caption: request.prompt,
    lyrics: request.lyrics ?? "",
    instrumental: request.instrumental,
    durationSeconds: request.durationSeconds,
    vocalLanguage: request.metadata?.vocalLanguage?.trim() || "unknown",
    metadata: Object.freeze(metadata),
  });
}

function emitToken(
  options: AcePlannerCoordinatorOptions,
  phase: AcePlannerLogicalPhase,
  completedTokens: number,
  totalTokens: number,
  sample: AcePlannerCursorSample,
): void {
  emit(options, {
    kind: "token",
    phase,
    completedTokens,
    totalTokens,
    sample,
  });
}

function emit(
  options: AcePlannerCoordinatorOptions,
  progress: AcePlannerCoordinatorProgress,
): void {
  options.onProgress?.(Object.freeze(progress));
}
