/// <reference lib="webworker" />
/// <reference types="vite/client" />

import {
  assertAceGenerationRequest,
  resolveAceDynamicConditionalWeighting,
  type AceGenerationRequest,
} from "../../src/api.js";
import { aceSha256Hex } from "../../src/model/sha256.js";
import type { AceDiagnostic, AceRuntimeDiagnostics } from
  "../../src/runtime/diagnostics.js";
import type {
  AceGenerationProgress,
  AceInitializationProgress,
} from "../../src/runtime/stages.js";
import {
  createAceWebGpuPipelineBackend,
  type AceOpt0018DitCheckpoint,
  type AceWebGpuGenerationContext,
} from "../../src/runtime/webgpu-pipeline.js";
import type {
  AceDitProfileFamily,
} from "../../src/webgpu/dit-graph.js";
import type {
  AceOpt0018DitCommandProfile,
  AceOpt0018DitProfileAggregate,
} from "../../src/webgpu/dit-backend.js";
import fixtureManifestSource from "../../golden/MANIFEST.json?raw";
import directInstrumentalFixtureSource from
  "../../golden/fixtures/direct-instrumental-short.json?raw";
import pipelineSource from "../../src/runtime/webgpu-pipeline.ts?raw";
import generationInputsSource from "../../src/runtime/generation-inputs.ts?raw";
import plannerCoordinatorSource from
  "../../src/runtime/planner-coordinator.ts?raw";
import schedulerSource from "../../src/runtime/scheduler.ts?raw";
import apiSource from "../../src/api.ts?raw";
import graphContractSource from "../../src/model/graph-contract.ts?raw";
import manifestSource from "../../src/model/manifest.ts?raw";
import sha256Source from "../../src/model/sha256.ts?raw";
import aceDitPackageSource from "../../src/webgpu/ace-dit-package.ts?raw";
import capabilitiesSource from "../../src/webgpu/capabilities.ts?raw";
import conditioningExecutorSource from
  "../../src/webgpu/conditioning-executor.ts?raw";
import ditBackendSource from "../../src/webgpu/dit-backend.ts?raw";
import ditFp16PackageSource from "../../src/webgpu/dit-fp16-package.ts?raw";
import ditGraphSource from "../../src/webgpu/dit-graph.ts?raw";
import aceDitSource from "../../src/webgpu/ace-dit.ts?raw";
import qwen3Source from "../../src/webgpu/qwen3.ts?raw";
import semanticConditionerSource from
  "../../src/webgpu/semantic-conditioner.ts?raw";
import attentionSource from "../../src/webgpu/kernels/attention.ts?raw";
import correctnessUtilsSource from
  "../../src/webgpu/kernels/correctness-utils.ts?raw";
import dcwSource from "../../src/webgpu/kernels/dcw.ts?raw";
import denseSource from "../../src/webgpu/kernels/dit-dense-fp16.ts?raw";
import ditPlumbingSource from "../../src/webgpu/kernels/dit-plumbing.ts?raw";
import gemmSource from "../../src/webgpu/kernels/gemm.ts?raw";
import rmsnormSource from "../../src/webgpu/kernels/rmsnorm.ts?raw";
import ropeSource from "../../src/webgpu/kernels/rope.ts?raw";
import subgroupGemmSource from "../../src/webgpu/kernels/subgroup-gemm.ts?raw";
import transformerPlumbingSource from
  "../../src/webgpu/kernels/transformer-plumbing.ts?raw";
import conditioningTextSource from
  "../../src/tokenizer/conditioning-text.ts?raw";
import tokenizerLoaderSource from "../../src/tokenizer/loader.ts?raw";
import qwenBpeSource from "../../src/tokenizer/qwen-bpe.ts?raw";
import workerSource from
  "./opt-0018-dit-m2250-production-family-profile-worker.ts?raw";
import pageSource from
  "./opt-0018-dit-m2250-production-family-profile.ts?raw";
import htmlSource from
  "./opt-0018-dit-m2250-production-family-profile.html?raw";
import contractSource from
  "../opt-0018-dit-m2250-production-family-profile-contract.test.ts?raw";
import {
  OPT_0018_CANONICAL_REQUEST_BYTES,
  OPT_0018_CANONICAL_REQUEST_JSON,
  OPT_0018_CANONICAL_REQUEST_SHA256,
  OPT_0018_FIXTURE_CONTRACT_SHA256,
  OPT_0018_MAXIMUM_LAUNCH_DELAY_MILLISECONDS,
  createOpt0018Request,
  serializeOpt0018Failure,
  validateOpt0018RunIdentity,
  type Opt0018RunIdentity,
  type Opt0018ThermalGate,
} from "./opt-0018-dit-m2250-production-family-profile.js";

const MAIN_MANIFEST_SHA256 =
  "18f36c6420976475af65ecd833ca56c6119706322ce54120389d4915d8e80db6";
const FIXTURE_MANIFEST_SHA256 =
  "cb9e0546c58be371581f302b8cd3943c3209ca1dcec296b75838ebf01c0cf7eb";
const DIRECT_INSTRUMENTAL_FIXTURE_FILE_SHA256 =
  "5ef8d23784da0341153289dcb8022a8b3c2ea6c431075d3a77a00d583dd13aa8";
const DIT_MANIFEST_SHA256 =
  "d3fc0020efcf60702db411da2fd4b93e9bb84f1437ed310aef01c892727e452f";
const VAE_MANIFEST_SHA256 =
  "94a1ae61354f7481facbb9787d003488ab1bc351a137fd2bd7ff69dd99aef949";
const ACE_SOURCE_REVISION = "6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0";
const PLANNER_SOURCE_REVISION = "148d8ea0225bdab342ee1ae3a354275ccd60ca80";
const PARAKEET_REVISION = "7ee112738262a6f5a0efd2f150748a4087432fbb";
const GRAPH_COMMAND_COUNT = 2_553;
const TOTAL_COMMAND_COUNT = 2_554;
const DESCRIPTOR_TABLE_SHA256 =
  "aedf8c74d2bb15601d1385e9c8e9da49e58905ab156a04c99271f8de3633dd76";
const DESCRIPTOR_TABLE_SERIALIZED_BYTES = 1_869_566;
const DESCRIPTOR_MEMBER_COUNT = 6_833;
const MIXED_COMMAND_COUNT = 1_344;
const TOTAL_SCHEDULED_MULTIPLY_ADDS = 26_840_955_355_136;
const TEXT_TOKEN_COUNT = 82;
const TEXT_TOKEN_SHA256 =
  "8067ee5c606e45e54d991364aa82a0ef7303e2a4e98831a01bb974236cafb3b2";
const LYRIC_TOKEN_COUNT = 15;
const LYRIC_TOKEN_SHA256 =
  "b4b58cd318163b4dfaa02b7ddbf46b18d84a415909c7662f9538c0b9053f3764";
const PACKED_TIMBRE_ROW_COUNT = 1;
const CONDITION_TOKEN_COUNT = 98;
const FINAL_LATENT_ELEMENTS = 288_000;
const FINAL_LATENT_BYTES = 1_152_000;
const CONDITIONING_DIAGNOSTIC_CODE = "ACE_PLANNER_CONDITIONING_RESOLVED";
const FAMILY_DIAGNOSTIC_CODE = "ACE_DIT_M2250_FAMILY_PROFILE";
const PRIVATE_DIT_STOP = Object.freeze({ opt0018PrivateDitStop: true });
const SOURCE_TEXT = Object.freeze({
  "golden/MANIFEST.json": fixtureManifestSource,
  "golden/fixtures/direct-instrumental-short.json":
    directInstrumentalFixtureSource,
  "src/api.ts": apiSource,
  "src/model/graph-contract.ts": graphContractSource,
  "src/model/manifest.ts": manifestSource,
  "src/model/sha256.ts": sha256Source,
  "src/runtime/generation-inputs.ts": generationInputsSource,
  "src/runtime/planner-coordinator.ts": plannerCoordinatorSource,
  "src/runtime/webgpu-pipeline.ts": pipelineSource,
  "src/runtime/scheduler.ts": schedulerSource,
  "src/tokenizer/conditioning-text.ts": conditioningTextSource,
  "src/tokenizer/loader.ts": tokenizerLoaderSource,
  "src/tokenizer/qwen-bpe.ts": qwenBpeSource,
  "src/webgpu/ace-dit-package.ts": aceDitPackageSource,
  "src/webgpu/capabilities.ts": capabilitiesSource,
  "src/webgpu/conditioning-executor.ts": conditioningExecutorSource,
  "src/webgpu/dit-backend.ts": ditBackendSource,
  "src/webgpu/dit-fp16-package.ts": ditFp16PackageSource,
  "src/webgpu/dit-graph.ts": ditGraphSource,
  "src/webgpu/ace-dit.ts": aceDitSource,
  "src/webgpu/qwen3.ts": qwen3Source,
  "src/webgpu/semantic-conditioner.ts": semanticConditionerSource,
  "src/webgpu/kernels/attention.ts": attentionSource,
  "src/webgpu/kernels/correctness-utils.ts": correctnessUtilsSource,
  "src/webgpu/kernels/dcw.ts": dcwSource,
  "src/webgpu/kernels/dit-dense-fp16.ts": denseSource,
  "src/webgpu/kernels/dit-plumbing.ts": ditPlumbingSource,
  "src/webgpu/kernels/gemm.ts": gemmSource,
  "src/webgpu/kernels/rmsnorm.ts": rmsnormSource,
  "src/webgpu/kernels/rope.ts": ropeSource,
  "src/webgpu/kernels/subgroup-gemm.ts": subgroupGemmSource,
  "src/webgpu/kernels/transformer-plumbing.ts": transformerPlumbingSource,
  "test/browser/opt-0018-dit-m2250-production-family-profile-worker.ts":
    workerSource,
  "test/browser/opt-0018-dit-m2250-production-family-profile.ts": pageSource,
  "test/browser/opt-0018-dit-m2250-production-family-profile.html": htmlSource,
  "test/opt-0018-dit-m2250-production-family-profile-contract.test.ts":
    contractSource,
});

interface InitializeMessage {
  readonly type: "initialize";
  readonly identity: Opt0018RunIdentity;
}

interface RunMessage {
  readonly type: "run";
  readonly thermalGate: Opt0018ThermalGate;
}

type IncomingMessage = InitializeMessage | RunMessage;

interface PreparedSession {
  readonly backend: ReturnType<typeof createAceWebGpuPipelineBackend>;
  readonly identity: Opt0018RunIdentity;
  readonly diagnostics: AceRuntimeDiagnostics;
  readonly sourceAuthority: Readonly<Record<string, string>>;
  readonly initializationStartedAtEpochMilliseconds: number;
  readonly initializationCompletedAtEpochMilliseconds: number;
  readonly initializationWallMilliseconds: number;
  readonly initializationProgressEventCount: number;
  readonly initializationDiagnosticCount: number;
}

interface CapturedProfileDiagnostic {
  readonly code: typeof FAMILY_DIAGNOSTIC_CODE;
  readonly elapsedMs: number;
  readonly stage: string | null;
  readonly details: Readonly<Record<string, unknown>>;
}

interface CapturedConditioningDiagnostic {
  readonly code: typeof CONDITIONING_DIAGNOSTIC_CODE;
  readonly elapsedMs: number;
  readonly stage: string | null;
  readonly details: Readonly<Record<string, unknown>>;
}

let lifecycle: "idle" | "preparing" | "ready" | "running" | "settled" =
  "idle";
let preparedSession: PreparedSession | undefined;

self.addEventListener("message", (event: MessageEvent<IncomingMessage>) => {
  if (event.data.type === "initialize") {
    if (lifecycle !== "idle") return;
    lifecycle = "preparing";
    void initializeSession(event.data.identity).then(
      (prepared) => {
        if (lifecycle !== "preparing") return;
        preparedSession = prepared;
        lifecycle = "ready";
        self.postMessage({
          type: "ready-for-thermal-gate",
          readyAtEpochMilliseconds:
            prepared.initializationCompletedAtEpochMilliseconds,
          preparation: publicPreparationSummary(prepared),
        });
      },
      (error: unknown) => void failAndCleanup(error),
    );
    return;
  }
  if (event.data.type === "run" && lifecycle === "ready") {
    lifecycle = "running";
    const active = preparedSession!;
    preparedSession = undefined;
    void runProfile(active, event.data.thermalGate).then(
      (result) => {
        lifecycle = "settled";
        self.postMessage({ type: "profile-complete", result });
      },
      (error: unknown) => void failAndCleanup(error, active),
    );
  }
});

async function initializeSession(
  identityValue: Opt0018RunIdentity,
): Promise<PreparedSession> {
  const identity = validateOpt0018RunIdentity(identityValue);
  const request = validateCanonicalRequest();
  void request;
  const sourceAuthority = buildSourceAuthority();
  const backend = createAceWebGpuPipelineBackend();
  const controller = new AbortController();
  let initializationProgressEventCount = 0;
  let initializationDiagnosticCount = 0;
  const initializationStartedAtEpochMilliseconds = Date.now();
  const startedAt = performance.now();
  postProgress("authenticating normal production package manifests and cache");
  try {
    const diagnostics = await backend.initialize(
      productionConfiguration(),
      {
        modelSource: "cache-or-network",
        signal: controller.signal,
        onProgress: (progress: AceInitializationProgress) => {
          initializationProgressEventCount += 1;
          if (
            progress.stage === "weights" ||
            progress.stage === "tokenizers" ||
            progress.stage === "ready"
          ) postProgress(progress.message ?? progress.stage);
        },
        onDiagnostic: () => {
          initializationDiagnosticCount += 1;
        },
      },
    );
    validateRuntimeDiagnostics(diagnostics);
    const initializationCompletedAtEpochMilliseconds = Date.now();
    return Object.freeze({
      backend,
      identity,
      diagnostics,
      sourceAuthority,
      initializationStartedAtEpochMilliseconds,
      initializationCompletedAtEpochMilliseconds,
      initializationWallMilliseconds: performance.now() - startedAt,
      initializationProgressEventCount,
      initializationDiagnosticCount,
    });
  } catch (error) {
    await backend.dispose().catch(() => undefined);
    throw error;
  }
}

async function runProfile(
  session: PreparedSession,
  thermalGate: Opt0018ThermalGate,
): Promise<Readonly<Record<string, unknown>>> {
  validateThermalGate(thermalGate, session.initializationCompletedAtEpochMilliseconds);
  const request = validateCanonicalRequest();
  const controller = new AbortController();
  const launchedAtEpochMilliseconds = Date.now();
  const launchDelayMilliseconds =
    launchedAtEpochMilliseconds - thermalGate.completedAtEpochMilliseconds;
  if (
    launchDelayMilliseconds < 0 ||
    launchDelayMilliseconds > OPT_0018_MAXIMUM_LAUNCH_DELAY_MILLISECONDS
  ) {
    throw new Error("OPT-0018 worker launch missed the frozen thermal handoff");
  }

  let checkpoint: AceOpt0018DitCheckpoint | undefined;
  let profileDiagnostic: CapturedProfileDiagnostic | undefined;
  let conditioningDiagnostic: CapturedConditioningDiagnostic | undefined;
  let checkpointCallbackCount = 0;
  let generationProgressEventCount = 0;
  let lastGenerationStage: string | null = null;
  let forbiddenPostDitProgressObserved = false;
  const diagnosticCodes: string[] = [];
  const fatalDiagnostics: Readonly<Record<string, unknown>>[] = [];
  const generationStartedAtEpochMilliseconds = Date.now();
  const generationStartedAt = performance.now();
  let generationRejectedAtEpochMilliseconds = 0;
  let cleanupCompletedAtEpochMilliseconds = 0;
  let backendDisposeStartedAtEpochMilliseconds = 0;
  let backendDisposeCompletedAtEpochMilliseconds = 0;
  let privateSentinelIdentityMatched = false;

  const context: AceWebGpuGenerationContext = {
    signal: controller.signal,
    captureTrace: true,
    onProgress: (progress: AceGenerationProgress) => {
      // This callback performs bounded numeric/state writes only. It never
      // logs, serializes, posts a message, performs asynchrony, or touches GPU.
      generationProgressEventCount += 1;
      lastGenerationStage = progress.stage;
      if (
        progress.stage === "vae-load" ||
        progress.stage === "vae-decode" ||
        progress.stage === "wav-encode" ||
        progress.stage === "cleanup" ||
        progress.stage === "done"
      ) forbiddenPostDitProgressObserved = true;
    },
    onDiagnostic: (diagnostic: AceDiagnostic) => {
      diagnosticCodes.push(diagnostic.code);
      if (diagnostic.severity === "error") {
        fatalDiagnostics.push(Object.freeze({
          code: diagnostic.code,
          message: diagnostic.message,
        }));
      }
      if (diagnostic.code === CONDITIONING_DIAGNOSTIC_CODE) {
        if (
          conditioningDiagnostic !== undefined ||
          diagnostic.details === undefined
        ) {
          fatalDiagnostics.push(Object.freeze({
            code: "OPT0018_DUPLICATE_OR_EMPTY_CONDITIONING_DIAGNOSTIC",
          }));
          return;
        }
        conditioningDiagnostic = Object.freeze({
          code: CONDITIONING_DIAGNOSTIC_CODE,
          elapsedMs: diagnostic.elapsedMs,
          stage: diagnostic.stage ?? null,
          details: diagnostic.details,
        });
        return;
      }
      if (diagnostic.code !== FAMILY_DIAGNOSTIC_CODE) return;
      if (profileDiagnostic !== undefined || diagnostic.details === undefined) {
        fatalDiagnostics.push(Object.freeze({
          code: "OPT0018_DUPLICATE_OR_EMPTY_PROFILE_DIAGNOSTIC",
        }));
        return;
      }
      profileDiagnostic = Object.freeze({
        code: FAMILY_DIAGNOSTIC_CODE,
        elapsedMs: diagnostic.elapsedMs,
        stage: diagnostic.stage ?? null,
        details: diagnostic.details,
      });
    },
    onDitCheckpoint: (value: AceOpt0018DitCheckpoint) => {
      checkpointCallbackCount += 1;
      if (checkpoint !== undefined || checkpointCallbackCount !== 1) {
        throw new Error("OPT-0018 checkpoint callback repeated");
      }
      validateCheckpoint(value);
      checkpoint = value;
      controller.abort(PRIVATE_DIT_STOP);
    },
  };

  try {
    await session.backend.generate(request, context);
    throw new Error("OPT-0018 unexpectedly continued into VAE/product output");
  } catch (error) {
    generationRejectedAtEpochMilliseconds = Date.now();
    privateSentinelIdentityMatched = error === PRIVATE_DIT_STOP;
    if (!privateSentinelIdentityMatched) throw error;
    // The pipeline rejection happens only after its catch path settles all
    // phase/backend cleanup. Record that boundary before disposing the device.
    cleanupCompletedAtEpochMilliseconds = generationRejectedAtEpochMilliseconds;
  }

  backendDisposeStartedAtEpochMilliseconds = Date.now();
  await session.backend.dispose();
  backendDisposeCompletedAtEpochMilliseconds = Date.now();

  if (
    checkpoint === undefined ||
    profileDiagnostic === undefined ||
    conditioningDiagnostic === undefined ||
    checkpointCallbackCount !== 1 ||
    forbiddenPostDitProgressObserved ||
    fatalDiagnostics.length !== 0
  ) {
    throw new Error("OPT-0018 checkpoint, diagnostic, or stop boundary failed");
  }
  const compactProfile = validateAndCompactProfile(
    checkpoint,
    profileDiagnostic,
  );
  const conditioningTokenAuthority = validateConditioningDiagnostic(
    conditioningDiagnostic,
  );
  const generationWallMilliseconds =
    performance.now() - generationStartedAt;

  return Object.freeze({
    schema: "ace-opt-0018-dit-m2250-production-family-profile-v1",
    experimentId: "OPT-0018",
    status: "passed-dit-only-checkpoint",
    identity: Object.freeze({
      run: session.identity,
      sourceAuthority: session.sourceAuthority,
      packageAuthority: Object.freeze({
        aceSourceRevision: ACE_SOURCE_REVISION,
        plannerSourceRevision: PLANNER_SOURCE_REVISION,
        parakeetReferenceRevision: PARAKEET_REVISION,
        fixtureManifestSha256: FIXTURE_MANIFEST_SHA256,
        fixtureContractSha256: OPT_0018_FIXTURE_CONTRACT_SHA256,
        directInstrumentalFixtureFileSha256:
          DIRECT_INSTRUMENTAL_FIXTURE_FILE_SHA256,
        mainManifestSha256: MAIN_MANIFEST_SHA256,
        ditDenseManifestSha256: DIT_MANIFEST_SHA256,
        vaeManifestSha256: VAE_MANIFEST_SHA256,
        modelManifestId: session.diagnostics.modelManifestId,
        ditDenseManifestId: session.diagnostics.ditDenseManifestId,
        vaeManifestId: session.diagnostics.vaeManifestId,
      }),
      requestAuthority: Object.freeze({
        canonicalJson: OPT_0018_CANONICAL_REQUEST_JSON,
        byteLength: OPT_0018_CANONICAL_REQUEST_BYTES,
        sha256: OPT_0018_CANONICAL_REQUEST_SHA256,
        effectiveInstrumentalLyrics: "[Instrumental]",
        directDcw: resolveAceDynamicConditionalWeighting(request.planner),
        conditioningTokenAuthority,
      }),
      runtimeDiagnostics: session.diagnostics,
    }),
    profiler: compactProfile,
    correctness: Object.freeze({
      checkpointSchema: checkpoint.schema,
      finalLatentElementCount: checkpoint.finalLatentElementCount,
      finalLatentByteLength: checkpoint.finalLatentByteLength,
      finalLatentSha256: checkpoint.finalLatentSha256,
      finalLatentNonFiniteCount: checkpoint.finalLatentNonFiniteCount,
      finalLatentNonzeroCount: checkpoint.finalLatentNonzeroCount,
      finalLatentMaxAbs: checkpoint.finalLatentMaxAbs,
      identicalRequestCurrentHeadOracleSha256: null,
      bitIdentityComparisonPerformed: false,
      finalLatentDetachedBeforeCheckpoint: true,
    }),
    lifecycle: Object.freeze({
      initializationStartedAtEpochMilliseconds:
        session.initializationStartedAtEpochMilliseconds,
      initializationCompletedAtEpochMilliseconds:
        session.initializationCompletedAtEpochMilliseconds,
      initializationWallMilliseconds: session.initializationWallMilliseconds,
      initializationProgressEventCount:
        session.initializationProgressEventCount,
      initializationDiagnosticCount: session.initializationDiagnosticCount,
      launchedAtEpochMilliseconds,
      generationStartedAtEpochMilliseconds,
      generationRejectedAtEpochMilliseconds,
      cleanupCompletedAtEpochMilliseconds,
      backendDisposeStartedAtEpochMilliseconds,
      backendDisposeCompletedAtEpochMilliseconds,
      generationWallMilliseconds,
      generationProgressEventCount,
      lastGenerationStage,
      diagnosticCodes: Object.freeze(diagnosticCodes),
      checkpointCallbackCount,
      privateSentinelIdentityMatched,
      ditDestroyedBeforeCheckpoint: true,
      pipelineCleanupAwaitedBeforeReceipt: true,
      backendDisposeAwaitedBeforeReceipt: true,
      vaeManifestAuthenticatedDuringInitialization: true,
      vaeSizedDeviceLimitsRequestedDuringInitialization: true,
      vaeWeightAcquireStarted: false,
      vaeBackendCreated: false,
      audioTransactionStarted: false,
      stableCancelMessageUsed: false,
      publicWorkerProtocolChanged: false,
      productResultPublished: false,
    }),
    protocol: Object.freeze({
      thermalGate,
      launchDelayMilliseconds,
      authoritativeTiming:
        "performance.now-immediately-before-submit-through-matching-queue-drain",
      timingRetryCount: 0,
      unchangedThermalRetryPerformed: false,
      memorySamplingEnabled: false,
      stoppedAfterDetachedFinalLatentAndDitDestroy: true,
      completedProduct: false,
    }),
    scope: Object.freeze({
      productionMathChanged: false,
      productionSelectorChanged: false,
      publicDefaultChanged: false,
      vaeExecuted: false,
      audioExecuted: false,
      optimizationIntegrated: false,
      under60SecondClaim: false,
    }),
  });
}

function validateCanonicalRequest(): AceGenerationRequest {
  const bytes = new TextEncoder().encode(OPT_0018_CANONICAL_REQUEST_JSON);
  if (
    bytes.byteLength !== OPT_0018_CANONICAL_REQUEST_BYTES ||
    aceSha256Hex(bytes) !== OPT_0018_CANONICAL_REQUEST_SHA256 ||
    hashText(fixtureManifestSource) !== FIXTURE_MANIFEST_SHA256 ||
    hashText(directInstrumentalFixtureSource) !==
      DIRECT_INSTRUMENTAL_FIXTURE_FILE_SHA256
  ) throw new Error("OPT-0018 canonical request bytes changed");
  const request = createOpt0018Request() as unknown as AceGenerationRequest;
  assertAceGenerationRequest(request);
  if (JSON.stringify(request) !== OPT_0018_CANONICAL_REQUEST_JSON) {
    throw new Error("OPT-0018 request property order or values changed");
  }
  const dcw = resolveAceDynamicConditionalWeighting(request.planner);
  if (
    dcw.mode !== "double" ||
    dcw.wavelet !== "haar" ||
    dcw.lowBandScale !== 0.05 ||
    dcw.highBandScale !== 0.02
  ) throw new Error("OPT-0018 direct DCW contract changed");
  return Object.freeze(request);
}

function validateConditioningDiagnostic(
  diagnostic: CapturedConditioningDiagnostic,
): Readonly<Record<string, unknown>> {
  const details = diagnostic.details;
  const caption =
    "Warm analog synth arpeggios over a restrained breakbeat, rounded " +
    "electric bass, airy pads, instrumental, detailed stereo production.";
  if (
    diagnostic.stage !== "semantic-planner" ||
    details.plannerEnabled !== false ||
    details.instrumental !== true ||
    details.resolvedCaption !== caption ||
    details.captionCharacters !== caption.length ||
    details.lyricCharacters !== "[Instrumental]".length ||
    details.vocalLanguage !== "unknown" ||
    details.metadata !==
      '{"duration":180,"bpm":104,"keyscale":"D minor","timesignature":"4"}' ||
    details.cotRan !== false ||
    details.phase2MetadataThinkBlock !== null ||
    details.generationProfile !== "ace-turbo-v1-correctness" ||
    details.plannerConfiguration !== '{"mode":"disabled"}' ||
    details.semanticCodeCount !== 0 ||
    details.textTokenCount !== TEXT_TOKEN_COUNT ||
    details.textTokenSha256 !== TEXT_TOKEN_SHA256 ||
    details.lyricTokenCount !== LYRIC_TOKEN_COUNT ||
    details.lyricTokenSha256 !== LYRIC_TOKEN_SHA256
  ) throw new Error("OPT-0018 conditioning token authority changed");
  return Object.freeze({
    diagnosticCode: diagnostic.code,
    diagnosticStage: diagnostic.stage,
    diagnosticElapsedMs: diagnostic.elapsedMs,
    textTokenCount: TEXT_TOKEN_COUNT,
    textTokenSha256: TEXT_TOKEN_SHA256,
    lyricTokenCount: LYRIC_TOKEN_COUNT,
    lyricTokenSha256: LYRIC_TOKEN_SHA256,
    packedTimbreRowCount: PACKED_TIMBRE_ROW_COUNT,
    conditionTokenCount: CONDITION_TOKEN_COUNT,
    derivation:
      "82 text tokens + 15 lyric tokens + 1 packed no-reference timbre row = 98",
    authenticatedFromActualRuntimeDiagnostic: true,
  });
}

function validateCheckpoint(checkpoint: AceOpt0018DitCheckpoint): void {
  if (
    checkpoint.schema !== "ace-dit-m2250-checkpoint-v1" ||
    !(checkpoint.finalLatent instanceof Float32Array) ||
    checkpoint.finalLatent.length !== FINAL_LATENT_ELEMENTS ||
    checkpoint.finalLatent.byteLength !== FINAL_LATENT_BYTES ||
    checkpoint.finalLatentElementCount !== FINAL_LATENT_ELEMENTS ||
    checkpoint.finalLatentByteLength !== FINAL_LATENT_BYTES
  ) throw new Error("OPT-0018 checkpoint latent geometry changed");
  let nonFinite = 0;
  let nonzero = 0;
  let maxAbs = 0;
  for (const value of checkpoint.finalLatent) {
    if (!Number.isFinite(value)) nonFinite += 1;
    else {
      if (value !== 0) nonzero += 1;
      maxAbs = Math.max(maxAbs, Math.abs(value));
    }
  }
  const rawBytes = new Uint8Array(
    checkpoint.finalLatent.buffer,
    checkpoint.finalLatent.byteOffset,
    checkpoint.finalLatent.byteLength,
  );
  if (
    nonFinite !== 0 ||
    nonzero === 0 ||
    maxAbs <= 0 ||
    checkpoint.finalLatentNonFiniteCount !== 0 ||
    checkpoint.finalLatentNonzeroCount !== nonzero ||
    checkpoint.finalLatentMaxAbs !== maxAbs ||
    checkpoint.finalLatentSha256 !== aceSha256Hex(rawBytes)
  ) throw new Error("OPT-0018 final latent hash or finite scan changed");
}

function validateAndCompactProfile(
  checkpoint: AceOpt0018DitCheckpoint,
  diagnostic: CapturedProfileDiagnostic,
): Readonly<Record<string, unknown>> {
  const profile = checkpoint.profile;
  const descriptors = profile.descriptorTable.descriptors;
  const totalScheduledMultiplyAdds = descriptors.reduce(
    (total, descriptor) => total + descriptor.scheduledMultiplyAdds,
    0,
  );
  if (
    profile.schema !== "ace-dit-m2250-command-profile-v1" ||
    profile.graphCommandBufferCount !== GRAPH_COMMAND_COUNT ||
    profile.readbackCommandBufferCount !== 1 ||
    profile.totalCommandBufferCount !== TOTAL_COMMAND_COUNT ||
    descriptors.length !== GRAPH_COMMAND_COUNT ||
    profile.timings.length !== GRAPH_COMMAND_COUNT ||
    profile.descriptorTable.sha256 !== DESCRIPTOR_TABLE_SHA256 ||
    profile.descriptorTable.serializedBytes !==
      DESCRIPTOR_TABLE_SERIALIZED_BYTES ||
    profile.descriptorTable.memberCount !== DESCRIPTOR_MEMBER_COUNT ||
    profile.families.mixed.commandBufferCount !== MIXED_COMMAND_COUNT ||
    totalScheduledMultiplyAdds !== TOTAL_SCHEDULED_MULTIPLY_ADDS ||
    profile.graphRequestedIdleMs !== GRAPH_COMMAND_COUNT - 1 ||
    profile.graphToReadbackRequestedIdleMs !== 1 ||
    profile.timingStorageBytes !== GRAPH_COMMAND_COUNT * 9
  ) throw new Error("OPT-0018 command profile topology changed");
  const firstDescriptor = descriptors[0]!;
  const lastDescriptor = descriptors[GRAPH_COMMAND_COUNT - 1]!;
  if (
    firstDescriptor.physicalIndex !== 0 ||
    firstDescriptor.logicalKind !== "condition-projection" ||
    firstDescriptor.family !== "precompute" ||
    firstDescriptor.members[0]?.id !==
      "ace-dit-condition-projection-range-0" ||
    lastDescriptor.physicalIndex !== GRAPH_COMMAND_COUNT - 1 ||
    lastDescriptor.logicalKind !== "sampler" ||
    lastDescriptor.evaluation !== 7 ||
    lastDescriptor.family !== "sampler-dcw" ||
    lastDescriptor.members.length !== 3 ||
    lastDescriptor.members.map((member) => member.id).join("|") !==
      "ace-dit-eval-7-sampler-sampler-update|" +
        "ace-dit-eval-7-sampler-predicted-clean|" +
        "ace-dit-eval-7-sampler-dcw"
  ) throw new Error("OPT-0018 first/last descriptor authority changed");

  let graphSum = 0;
  const timingTuples = descriptors.map((descriptor, index) => {
    const elapsedMs = profile.timings[index]!;
    if (
      descriptor.physicalIndex !== index ||
      !Number.isFinite(elapsedMs) ||
      elapsedMs < 0
    ) throw new Error(`OPT-0018 command ${index} is incomplete`);
    graphSum += elapsedMs;
    return Object.freeze([
      index,
      descriptor.family,
      descriptor.evaluation ?? -1,
      descriptor.layer ?? -1,
      elapsedMs,
    ] as const);
  });
  requireNear(graphSum, profile.graphSubmitThroughDrainMs, profile);
  requireAggregateReconciliation(
    Object.values(profile.families),
    GRAPH_COMMAND_COUNT,
    profile.graphSubmitThroughDrainMs,
    profile,
  );
  requireAggregateReconciliation(
    [profile.precompute, ...profile.evaluations],
    GRAPH_COMMAND_COUNT,
    profile.graphSubmitThroughDrainMs,
    profile,
  );
  for (const [family, aggregate] of Object.entries(profile.families)) {
    requireAggregateReconciliation(
      profile.familyByBucket[family as AceDitProfileFamily],
      aggregate.commandBufferCount,
      aggregate.submitThroughDrainMs,
      profile,
    );
  }

  const details = diagnostic.details;
  if (
    details.schema !== "ace-dit-m2250-family-profile-receipt-v1" ||
    details.commandProfileSchema !== profile.schema ||
    details.checkpointSchema !== checkpoint.schema ||
    details.conditionTokens !== CONDITION_TOKEN_COUNT ||
    details.graphCommandBufferCount !== GRAPH_COMMAND_COUNT ||
    details.readbackCommandBufferCount !== 1 ||
    details.totalCommandBufferCount !== TOTAL_COMMAND_COUNT ||
    details.descriptorTableSha256 !== profile.descriptorTable.sha256 ||
    details.finalLatentSha256 !== checkpoint.finalLatentSha256 ||
    diagnostic.stage !== "release-dit"
  ) throw new Error("OPT-0018 profiler diagnostic diverged from checkpoint");
  const expectedTimingJson = JSON.stringify(timingTuples);
  if (details.commandTimingTuplesJson !== expectedTimingJson) {
    throw new Error("OPT-0018 diagnostic timing tuples changed");
  }
  const descriptorRowsJson = requiredDiagnosticString(
    details,
    "descriptorRowsJson",
  );
  const descriptorRows = descriptors.map((descriptor) =>
    compactDescriptorRow(descriptor)
  );
  if (descriptorRowsJson !== JSON.stringify(descriptorRows)) {
    throw new Error("OPT-0018 diagnostic descriptor rows changed");
  }

  return Object.freeze({
    diagnostic: Object.freeze({
      code: diagnostic.code,
      schema: details.schema,
      stage: diagnostic.stage,
      elapsedMs: diagnostic.elapsedMs,
      detailsSha256: hashText(JSON.stringify(details)),
      descriptorRowsSha256: hashText(descriptorRowsJson),
      commandTimingTuplesSha256: hashText(expectedTimingJson),
      matchedCheckpoint: true,
    }),
    descriptorTable: Object.freeze({
      sha256: profile.descriptorTable.sha256,
      serializedBytes: profile.descriptorTable.serializedBytes,
      memberCount: profile.descriptorTable.memberCount,
      preparationMs: profile.descriptorTable.preparationMs,
      descriptorCount: descriptors.length,
      mixedDescriptorCount: MIXED_COMMAND_COUNT,
      scheduledMultiplyAdds: totalScheduledMultiplyAdds,
    }),
    descriptorRows: Object.freeze(descriptorRows),
    commandTimingTuples: Object.freeze(timingTuples),
    graphCommandBufferCount: profile.graphCommandBufferCount,
    readbackCommandBufferCount: profile.readbackCommandBufferCount,
    totalCommandBufferCount: profile.totalCommandBufferCount,
    graphSubmitThroughDrainMs: profile.graphSubmitThroughDrainMs,
    graphWallMs: profile.graphWallMs,
    graphRequestedIdleMs: profile.graphRequestedIdleMs,
    graphResidualMs: profile.graphResidualMs,
    graphToReadbackRequestedIdleMs: profile.graphToReadbackRequestedIdleMs,
    graphToReadbackObservedIdleMs: profile.graphToReadbackObservedIdleMs,
    readbackSubmitThroughDrainMs: profile.readbackSubmitThroughDrainMs,
    readbackMapDetachMs: profile.readbackMapDetachMs,
    backendWallMs: profile.backendWallMs,
    backendResidualMs: profile.backendResidualMs,
    families: profile.families,
    precompute: profile.precompute,
    evaluations: profile.evaluations,
    familyByBucket: profile.familyByBucket,
    slowest: profile.slowest,
    stageTimings: checkpoint.stageTimings,
    reconciliationToleranceMs: profile.reconciliationToleranceMs,
    timingStorageBytes: profile.timingStorageBytes,
    captureCpuOverhead: Object.freeze({
      descriptorPreparationMs: profile.descriptorTable.preparationMs,
      timingStorageBytes: profile.timingStorageBytes,
      perDrainOperation: "one bounded Float64/Uint8 write plus aggregate callback",
      perDrainLoggingSerializationPostMessageOrGpuAllocation: false,
    }),
    attributionBoundary: Object.freeze({
      everyPhysicalCommandAssignedExactlyOneFamily: true,
      crossFamilyCommandsAssignedToMixed: true,
      mixedCommandWallSplitOrEstimated: false,
      pureSlidingOrCrossTimingClaimFromMixedCommands: false,
    }),
  });
}

function compactDescriptorRow(
  descriptor: AceOpt0018DitCommandProfile["descriptorTable"]["descriptors"][number],
) {
  return Object.freeze([
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
    Object.freeze(descriptor.members.map((member) => Object.freeze([
      member.id,
      member.family,
      member.backend,
      member.kernel,
      member.scheduledMultiplyAdds,
    ] as const))),
  ] as const);
}

function requireAggregateReconciliation(
  aggregates: readonly AceOpt0018DitProfileAggregate[],
  expectedCount: number,
  expectedMilliseconds: number,
  profile: AceOpt0018DitCommandProfile,
): void {
  const count = aggregates.reduce(
    (total, aggregate) => total + aggregate.commandBufferCount,
    0,
  );
  const milliseconds = aggregates.reduce(
    (total, aggregate) => total + aggregate.submitThroughDrainMs,
    0,
  );
  if (count !== expectedCount) throw new Error("OPT-0018 aggregate count diverged");
  requireNear(milliseconds, expectedMilliseconds, profile);
}

function requireNear(
  actual: number,
  expected: number,
  profile: AceOpt0018DitCommandProfile,
): void {
  if (
    !Number.isFinite(actual) ||
    !Number.isFinite(expected) ||
    Math.abs(actual - expected) > profile.reconciliationToleranceMs
  ) throw new Error("OPT-0018 timing aggregates do not reconcile");
}

function requiredDiagnosticString(
  details: Readonly<Record<string, unknown>>,
  name: string,
): string {
  const value = details[name];
  if (typeof value !== "string") {
    throw new Error(`OPT-0018 diagnostic ${name} is missing`);
  }
  return value;
}

function buildSourceAuthority(): Readonly<Record<string, string>> {
  return Object.freeze(Object.fromEntries(
    Object.entries(SOURCE_TEXT).map(([path, source]) => [path, hashText(source)]),
  ));
}

function hashText(value: string): string {
  return aceSha256Hex(new TextEncoder().encode(value));
}

function validateThermalGate(
  gate: Opt0018ThermalGate,
  readyAtEpochMilliseconds: number,
): void {
  if (
    gate.source !== "notifyutil-com.apple.system.thermalpressurelevel" ||
    gate.startedAtEpochMilliseconds < readyAtEpochMilliseconds ||
    gate.durationMilliseconds < 30_000 ||
    gate.completedAtEpochMilliseconds - gate.startedAtEpochMilliseconds !==
      gate.durationMilliseconds ||
    gate.pollMilliseconds !== 1_000 ||
    gate.maximumPollGapMilliseconds > 1_250 ||
    gate.nonNominalObservationCount !== 0
  ) throw new Error("OPT-0018 worker rejected the thermal gate");
}

function validateRuntimeDiagnostics(diagnostics: AceRuntimeDiagnostics): void {
  const capabilities = diagnostics.capabilities;
  if (
    diagnostics.modelManifestSha256 !== MAIN_MANIFEST_SHA256 ||
    diagnostics.ditDenseManifestSha256 !== DIT_MANIFEST_SHA256 ||
    diagnostics.vaeManifestSha256 !== VAE_MANIFEST_SHA256 ||
    diagnostics.aceSourceRevision !== ACE_SOURCE_REVISION ||
    diagnostics.plannerSourceRevision !== PLANNER_SOURCE_REVISION ||
    diagnostics.parakeetReferenceRevision !== PARAKEET_REVISION ||
    diagnostics.executionProfile.id !== "reference-bf16-subgroups" ||
    diagnostics.schedulingProfile !== "cooperative" ||
    diagnostics.ditDenseRuntimeProfile !== "opt-0009-fp16-fp32-dense-v1" ||
    diagnostics.vaeRuntimeProfile !==
      "opt-0028-mixed-fp16-fixed32-exact-packed-v1" ||
    diagnostics.vaeMaxWindowFrames !== 512 ||
    capabilities.adapterInfo.subgroupMinSize !== 32 ||
    capabilities.adapterInfo.subgroupMaxSize !== 32 ||
    !capabilities.deviceFeatures.includes("shader-f16") ||
    !capabilities.deviceFeatures.includes("subgroups")
  ) throw new Error("OPT-0018 runtime diagnostics escaped frozen production");
}

function productionConfiguration() {
  return Object.freeze({
    manifestUrl: new URL("/model/files-reference/manifest.json", self.location.href).href,
    manifestSha256: MAIN_MANIFEST_SHA256,
    modelProfile: "reference-bf16" as const,
    schedulingProfile: "cooperative" as const,
    ditDensePackage: Object.freeze({
      manifestUrl: new URL(
        "/model/files-fp16-dit-layer-mixed-experimental/manifest.json",
        self.location.href,
      ).href,
      manifestSha256: DIT_MANIFEST_SHA256,
      runtimeProfile: "opt-0009-fp16-fp32-dense-v1" as const,
    }),
    vaePackage: Object.freeze({
      manifestUrl: new URL(
        "/model/files-fp16-vae-experimental/manifest.json",
        self.location.href,
      ).href,
      manifestSha256: VAE_MANIFEST_SHA256,
      runtimeProfile:
        "opt-0028-mixed-fp16-fixed32-exact-packed-v1" as const,
      maxWindowFrames: 512 as const,
    }),
  });
}

function publicPreparationSummary(
  prepared: PreparedSession,
): Readonly<Record<string, unknown>> {
  return Object.freeze({
    initializationWallMilliseconds: prepared.initializationWallMilliseconds,
    initializationProgressEventCount: prepared.initializationProgressEventCount,
    sourceFileCount: Object.keys(prepared.sourceAuthority).length,
    requestSha256: OPT_0018_CANONICAL_REQUEST_SHA256,
    modelManifestSha256: prepared.diagnostics.modelManifestSha256,
    ditDenseManifestSha256: prepared.diagnostics.ditDenseManifestSha256,
    vaeManifestAuthenticatedButWeightsNotAcquired:
      prepared.diagnostics.vaeManifestSha256 === VAE_MANIFEST_SHA256,
    vaeSizedDeviceLimitsRequestedByNormalInitialization: true,
  });
}

async function failAndCleanup(
  error: unknown,
  active: PreparedSession | undefined = preparedSession,
): Promise<void> {
  preparedSession = undefined;
  let cleanupError: unknown;
  try {
    await active?.backend.dispose();
  } catch (failure) {
    cleanupError = failure;
  }
  lifecycle = "settled";
  self.postMessage({
    type: "failed",
    error: serializeOpt0018Failure(error, cleanupError),
  });
}

function postProgress(message: string): void {
  self.postMessage({ type: "progress", message });
}
