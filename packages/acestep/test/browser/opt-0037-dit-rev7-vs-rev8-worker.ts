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
  AceWorkerConfiguration,
} from "../../src/runtime/protocol.js";
import type {
  AceGenerationProgress,
  AceInitializationProgress,
  AceStageTiming,
} from "../../src/runtime/stages.js";
import {
  createAceWebGpuPipelineBackend,
  type AceOpt0018DitCheckpoint,
  type AceWebGpuGenerationContext,
} from "../../src/runtime/webgpu-pipeline.js";
import {
  OPT_0018_CANONICAL_REQUEST_BYTES,
  OPT_0018_CANONICAL_REQUEST_JSON,
  OPT_0018_CANONICAL_REQUEST_SHA256,
  createOpt0018Request,
  serializeOpt0018Failure,
} from "./opt-0018-dit-m2250-production-family-profile.js";
import {
  OPT_0037_CANDIDATE_KERNEL_SET_ID,
  OPT_0037_CANDIDATE_MANIFEST_BYTES,
  OPT_0037_CANDIDATE_MANIFEST_PATH,
  OPT_0037_CANDIDATE_MANIFEST_SHA256,
  OPT_0037_CANDIDATE_RUNTIME_PROFILE,
  OPT_0037_CONTROL_KERNEL_SET_ID,
  OPT_0037_CONTROL_MANIFEST_BYTES,
  OPT_0037_CONTROL_MANIFEST_PATH,
  OPT_0037_CONTROL_MANIFEST_SHA256,
  OPT_0037_CONTROL_RUNTIME_PROFILE,
  OPT_0037_FINAL_LATENT_BYTES,
  OPT_0037_FINAL_LATENT_ELEMENTS,
  compareOpt0037FinalLatents,
} from "./opt-0037-dit-rev7-vs-rev8-contract.js";

const MAIN_MANIFEST_PATH = "/model/files-reference/manifest.json" as const;
const MAIN_MANIFEST_SHA256 =
  "18f36c6420976475af65ecd833ca56c6119706322ce54120389d4915d8e80db6" as const;
const VAE_MANIFEST_PATH =
  "/model/files-fp16-vae-experimental/manifest.json" as const;
const VAE_MANIFEST_SHA256 =
  "94a1ae61354f7481facbb9787d003488ab1bc351a137fd2bd7ff69dd99aef949" as const;
const VAE_RUNTIME_PROFILE =
  "opt-0028-mixed-fp16-fixed32-exact-packed-v1" as const;
const VAE_KERNEL_SET_ID =
  "opt-0028-vae-fp16-fixed32-exact-packed-kernel-set-v1" as const;
const TEXT_TOKEN_COUNT = 82;
const LYRIC_TOKEN_COUNT = 15;
const CONDITION_TOKEN_COUNT = 98;
const CONDITIONING_DIAGNOSTIC_CODE =
  "ACE_PLANNER_CONDITIONING_RESOLVED" as const;
const FAMILY_DIAGNOSTIC_CODE = "ACE_DIT_M2250_FAMILY_PROFILE" as const;
const FORBIDDEN_POST_DIT_STAGES = new Set([
  "vae-load",
  "vae-decode",
  "wav-encode",
  "cleanup",
  "done",
]);

type ArmRole = "rev7-control" | "rev8-candidate";

interface ArmDefinition {
  readonly role: ArmRole;
  readonly manifestPath:
    | typeof OPT_0037_CONTROL_MANIFEST_PATH
    | typeof OPT_0037_CANDIDATE_MANIFEST_PATH;
  readonly manifestSha256:
    | typeof OPT_0037_CONTROL_MANIFEST_SHA256
    | typeof OPT_0037_CANDIDATE_MANIFEST_SHA256;
  readonly manifestByteLength:
    | typeof OPT_0037_CONTROL_MANIFEST_BYTES
    | typeof OPT_0037_CANDIDATE_MANIFEST_BYTES;
  readonly runtimeProfile:
    | typeof OPT_0037_CONTROL_RUNTIME_PROFILE
    | typeof OPT_0037_CANDIDATE_RUNTIME_PROFILE;
  readonly kernelSetId:
    | typeof OPT_0037_CONTROL_KERNEL_SET_ID
    | typeof OPT_0037_CANDIDATE_KERNEL_SET_ID;
  readonly expectedGemmBackend: "mixed-opt-0009" | "mixed-opt-0037-k4";
}

interface CapturedDiagnostic {
  readonly stage: string | null;
  readonly details: Readonly<Record<string, unknown>>;
}

interface InternalArmResult {
  readonly finalLatent: Float32Array<ArrayBuffer>;
  readonly conditioningAuthority: Readonly<Record<string, unknown>>;
  readonly receipt: Readonly<Record<string, unknown>>;
  readonly lifecycle: Readonly<{
    readonly backendCreatedOrdinal: number;
    readonly initializedOrdinal: number;
    readonly checkpointOrdinal: number;
    readonly generationCleanupCompletedOrdinal: number;
    readonly disposeStartedOrdinal: number;
    readonly disposeCompletedOrdinal: number;
  }>;
}

const CONTROL: ArmDefinition = Object.freeze({
  role: "rev7-control",
  manifestPath: OPT_0037_CONTROL_MANIFEST_PATH,
  manifestSha256: OPT_0037_CONTROL_MANIFEST_SHA256,
  manifestByteLength: OPT_0037_CONTROL_MANIFEST_BYTES,
  runtimeProfile: OPT_0037_CONTROL_RUNTIME_PROFILE,
  kernelSetId: OPT_0037_CONTROL_KERNEL_SET_ID,
  expectedGemmBackend: "mixed-opt-0009",
});
const CANDIDATE: ArmDefinition = Object.freeze({
  role: "rev8-candidate",
  manifestPath: OPT_0037_CANDIDATE_MANIFEST_PATH,
  manifestSha256: OPT_0037_CANDIDATE_MANIFEST_SHA256,
  manifestByteLength: OPT_0037_CANDIDATE_MANIFEST_BYTES,
  runtimeProfile: OPT_0037_CANDIDATE_RUNTIME_PROFILE,
  kernelSetId: OPT_0037_CANDIDATE_KERNEL_SET_ID,
  expectedGemmBackend: "mixed-opt-0037-k4",
});

let lifecycle: "idle" | "running" | "settled" = "idle";
let lifecycleOrdinal = 0;

self.addEventListener("message", (event: MessageEvent<unknown>) => {
  if (
    lifecycle !== "idle" ||
    !isRecord(event.data) ||
    event.data.type !== "run"
  ) return;
  lifecycle = "running";
  void runComparison().then(
    (result) => {
      lifecycle = "settled";
      self.postMessage({ type: "comparison-complete", result });
    },
    (error: unknown) => {
      lifecycle = "settled";
      self.postMessage({
        type: "failed",
        error: serializeOpt0018Failure(error),
      });
    },
  );
});

async function runComparison(): Promise<Readonly<Record<string, unknown>>> {
  const request = validateCanonicalRequest();
  postProgress("rev7 control: authenticating and running to final latent");
  const control = await runArm(CONTROL, request);
  if (
    control.lifecycle.disposeCompletedOrdinal <=
      control.lifecycle.disposeStartedOrdinal
  ) throw new Error("OPT-0037 control disposal boundary was not completed");

  postProgress("rev7 disposed; rev8 candidate: authenticating and running");
  const candidate = await runArm(CANDIDATE, request);
  if (
    control.lifecycle.disposeCompletedOrdinal >=
      candidate.lifecycle.backendCreatedOrdinal
  ) {
    throw new Error("OPT-0037 candidate existed before control disposal");
  }
  if (!sameJson(control.conditioningAuthority, candidate.conditioningAuthority)) {
    throw new Error("OPT-0037 arms did not resolve identical C98 conditioning");
  }

  postProgress("comparing all 288,000 detached F32 final-latent values");
  const comparison = compareOpt0037FinalLatents(
    control.finalLatent,
    candidate.finalLatent,
  );
  return Object.freeze({
    schema: "ace-opt-0037-rev7-vs-rev8-final-latent-gate-v1",
    experimentId: "OPT-0037",
    status: comparison.passed
      ? "passed-final-latent-gate"
      : "failed-final-latent-gate",
    fixedExecutionOrder: Object.freeze([
      "rev7-control",
      "rev8-candidate",
    ]),
    identity: Object.freeze({
      request: Object.freeze({
        canonicalJson: OPT_0018_CANONICAL_REQUEST_JSON,
        byteLength: OPT_0018_CANONICAL_REQUEST_BYTES,
        sha256: OPT_0018_CANONICAL_REQUEST_SHA256,
        durationSeconds: 180,
        seed: "0000000000c0ffee",
        conditionTokenCount: CONDITION_TOKEN_COUNT,
      }),
      controlPackage: armIdentity(CONTROL),
      candidatePackage: armIdentity(CANDIDATE),
      mainManifestSha256: MAIN_MANIFEST_SHA256,
      vaeManifestSha256: VAE_MANIFEST_SHA256,
    }),
    arms: Object.freeze({
      control: control.receipt,
      candidate: candidate.receipt,
    }),
    correctness: Object.freeze({
      deterministicCanonicalRequest: true,
      fullFinalLatentsCompared: true,
      elementCount: OPT_0037_FINAL_LATENT_ELEMENTS,
      byteLengthPerArm: OPT_0037_FINAL_LATENT_BYTES,
      controlFinalLatentSha256:
        readRequiredString(control.receipt, "finalLatentSha256"),
      candidateFinalLatentSha256:
        readRequiredString(candidate.receipt, "finalLatentSha256"),
      comparison,
    }),
    lifecycle: Object.freeze({
      control: control.lifecycle,
      candidate: candidate.lifecycle,
      controlDisposedBeforeCandidateCreated:
        control.lifecycle.disposeCompletedOrdinal <
          candidate.lifecycle.backendCreatedOrdinal,
      gpuPackageRuntimeCoResidency: false,
      onlyDetachedCpuFinalLatentRetainedBetweenArms: true,
    }),
    timingContext: Object.freeze({
      fixedOrder: "rev7-control-then-rev8-candidate",
      orderConfounded: true,
      graphAndStageWallsRecorded: true,
      speedupCalculated: false,
      performanceDecisionAuthorized: false,
      thermalProtocolApplied: false,
    }),
    scope: Object.freeze({
      singleSequentialCorrectnessRun: true,
      vaeManifestAuthenticatedPerArm: true,
      vaeWeightsAcquired: false,
      vaeExecuted: false,
      audioExecuted: false,
      productionDemoDefaultChanged: false,
      performanceClaim: false,
      listeningApproval: false,
    }),
  });
}

async function runArm(
  arm: ArmDefinition,
  request: AceGenerationRequest,
): Promise<InternalArmResult> {
  const backendCreatedOrdinal = nextLifecycleOrdinal();
  const backend = createAceWebGpuPipelineBackend();
  const controller = new AbortController();
  const privateStop = Object.freeze({ role: arm.role });
  const diagnosticCodes: string[] = [];
  const fatalDiagnostics: string[] = [];
  let diagnostics: AceRuntimeDiagnostics | undefined;
  let checkpoint: AceOpt0018DitCheckpoint | undefined;
  let conditioningDiagnostic: CapturedDiagnostic | undefined;
  let familyDiagnostic: CapturedDiagnostic | undefined;
  let initializedOrdinal = 0;
  let checkpointOrdinal = 0;
  let generationCleanupCompletedOrdinal = 0;
  let disposeStartedOrdinal = 0;
  let disposeCompletedOrdinal = 0;
  let checkpointCallbackCount = 0;
  let initializationProgressEventCount = 0;
  let generationProgressEventCount = 0;
  let forbiddenPostDitProgressObserved = false;
  let privateStopMatched = false;
  let primaryError: unknown;
  let cleanupError: unknown;

  try {
    diagnostics = await backend.initialize(configurationForArm(arm), {
      modelSource: "cache-or-network",
      signal: controller.signal,
      onProgress: (_progress: AceInitializationProgress) => {
        initializationProgressEventCount += 1;
      },
      onDiagnostic: (diagnostic: AceDiagnostic) => {
        if (diagnostic.severity === "error") {
          fatalDiagnostics.push(diagnostic.code);
        }
      },
    });
    initializedOrdinal = nextLifecycleOrdinal();
    validateRuntimeDiagnostics(diagnostics, arm);

    const context: AceWebGpuGenerationContext = {
      signal: controller.signal,
      captureTrace: true,
      onProgress: (progress: AceGenerationProgress) => {
        generationProgressEventCount += 1;
        if (FORBIDDEN_POST_DIT_STAGES.has(progress.stage)) {
          forbiddenPostDitProgressObserved = true;
        }
      },
      onDiagnostic: (diagnostic: AceDiagnostic) => {
        diagnosticCodes.push(diagnostic.code);
        if (diagnostic.severity === "error") {
          fatalDiagnostics.push(diagnostic.code);
        }
        if (diagnostic.details === undefined) return;
        if (diagnostic.code === CONDITIONING_DIAGNOSTIC_CODE) {
          if (conditioningDiagnostic !== undefined) {
            fatalDiagnostics.push("duplicate-conditioning-diagnostic");
            return;
          }
          conditioningDiagnostic = Object.freeze({
            stage: diagnostic.stage ?? null,
            details: diagnostic.details,
          });
        } else if (diagnostic.code === FAMILY_DIAGNOSTIC_CODE) {
          if (familyDiagnostic !== undefined) {
            fatalDiagnostics.push("duplicate-family-diagnostic");
            return;
          }
          familyDiagnostic = Object.freeze({
            stage: diagnostic.stage ?? null,
            details: diagnostic.details,
          });
        }
      },
      onDitCheckpoint: (value: AceOpt0018DitCheckpoint) => {
        checkpointCallbackCount += 1;
        if (checkpointCallbackCount !== 1 || checkpoint !== undefined) {
          throw new Error(`OPT-0037 ${arm.role} checkpoint repeated`);
        }
        validateCheckpoint(value);
        checkpoint = value;
        checkpointOrdinal = nextLifecycleOrdinal();
        controller.abort(privateStop);
      },
    };

    try {
      await backend.generate(request, context);
      throw new Error(`OPT-0037 ${arm.role} continued into VAE`);
    } catch (error) {
      privateStopMatched = error === privateStop;
      if (!privateStopMatched) throw error;
      generationCleanupCompletedOrdinal = nextLifecycleOrdinal();
    }
  } catch (error) {
    primaryError = error;
  } finally {
    disposeStartedOrdinal = nextLifecycleOrdinal();
    try {
      await backend.dispose();
    } catch (error) {
      cleanupError = error;
    }
    disposeCompletedOrdinal = nextLifecycleOrdinal();
  }

  if (primaryError !== undefined || cleanupError !== undefined) {
    if (primaryError !== undefined && cleanupError !== undefined) {
      throw new AggregateError(
        [primaryError, cleanupError],
        `OPT-0037 ${arm.role} run and disposal both failed`,
      );
    }
    throw primaryError ?? cleanupError;
  }
  if (
    diagnostics === undefined ||
    checkpoint === undefined ||
    conditioningDiagnostic === undefined ||
    familyDiagnostic === undefined ||
    initializedOrdinal === 0 ||
    checkpointOrdinal === 0 ||
    generationCleanupCompletedOrdinal === 0 ||
    checkpointCallbackCount !== 1 ||
    !privateStopMatched ||
    forbiddenPostDitProgressObserved ||
    fatalDiagnostics.length !== 0
  ) {
    throw new Error(`OPT-0037 ${arm.role} checkpoint boundary was incomplete`);
  }

  const conditioningAuthority = validateConditioningDiagnostic(
    conditioningDiagnostic,
  );
  validateFamilyDiagnostic(familyDiagnostic, arm);
  const stageWalls = compactStageWalls(checkpoint.stageTimings);
  const profile = checkpoint.profile;
  const lifecycleReceipt = Object.freeze({
    backendCreatedOrdinal,
    initializedOrdinal,
    checkpointOrdinal,
    generationCleanupCompletedOrdinal,
    disposeStartedOrdinal,
    disposeCompletedOrdinal,
  });
  return Object.freeze({
    finalLatent: checkpoint.finalLatent,
    conditioningAuthority,
    lifecycle: lifecycleReceipt,
    receipt: Object.freeze({
      role: arm.role,
      authenticatedPackage: armIdentity(arm),
      runtimeDiagnostics: Object.freeze({
        modelManifestId: diagnostics.modelManifestId,
        modelManifestSha256: diagnostics.modelManifestSha256,
        ditDenseManifestId: diagnostics.ditDenseManifestId,
        ditDenseManifestUrl: diagnostics.ditDenseManifestUrl,
        ditDenseManifestSha256: diagnostics.ditDenseManifestSha256,
        ditDenseManifestByteLength: diagnostics.ditDenseManifestByteLength,
        ditDenseRuntimeProfile: diagnostics.ditDenseRuntimeProfile,
        ditDenseKernelSetId: diagnostics.ditDenseKernelSetId,
        ditDenseLayerBytes: diagnostics.ditDenseLayerBytes,
        ditResidentWeightBytes: diagnostics.ditResidentWeightBytes,
        vaeManifestId: diagnostics.vaeManifestId,
        vaeManifestSha256: diagnostics.vaeManifestSha256,
        vaeManifestByteLength: diagnostics.vaeManifestByteLength,
        vaeRuntimeProfile: diagnostics.vaeRuntimeProfile,
        vaeKernelSetId: diagnostics.vaeKernelSetId,
        executionProfileId: diagnostics.executionProfile.id,
        schedulingProfile: diagnostics.schedulingProfile,
      }),
      conditioningAuthority,
      finalLatentElementCount: checkpoint.finalLatentElementCount,
      finalLatentByteLength: checkpoint.finalLatentByteLength,
      finalLatentSha256: checkpoint.finalLatentSha256,
      finalLatentNonFiniteCount: checkpoint.finalLatentNonFiniteCount,
      finalLatentNonzeroCount: checkpoint.finalLatentNonzeroCount,
      finalLatentMaxAbs: checkpoint.finalLatentMaxAbs,
      graphWalls: Object.freeze({
        graphSubmitThroughDrainMs: profile.graphSubmitThroughDrainMs,
        graphWallMs: profile.graphWallMs,
      }),
      stageWalls,
      lifecycle: lifecycleReceipt,
      evidence: Object.freeze({
        initializationProgressEventCount,
        generationProgressEventCount,
        diagnosticCodes: Object.freeze(diagnosticCodes),
        checkpointCallbackCount,
        privateStopMatched,
        ditDestroyedBeforeCheckpoint: true,
        pipelineCleanupAwaitedBeforeDispose: true,
        backendAndDeviceDisposeAwaited: true,
        vaeWeightAcquireStarted: false,
        vaeBackendCreated: false,
      }),
    }),
  });
}

function configurationForArm(arm: ArmDefinition): AceWorkerConfiguration {
  const ditDensePackage = arm.role === "rev7-control"
    ? Object.freeze({
        manifestUrl: absoluteUrl(OPT_0037_CONTROL_MANIFEST_PATH),
        manifestSha256: OPT_0037_CONTROL_MANIFEST_SHA256,
        runtimeProfile: OPT_0037_CONTROL_RUNTIME_PROFILE,
      })
    : Object.freeze({
        manifestUrl: absoluteUrl(OPT_0037_CANDIDATE_MANIFEST_PATH),
        manifestSha256: OPT_0037_CANDIDATE_MANIFEST_SHA256,
        runtimeProfile: OPT_0037_CANDIDATE_RUNTIME_PROFILE,
      });
  return Object.freeze({
    manifestUrl: absoluteUrl(MAIN_MANIFEST_PATH),
    manifestSha256: MAIN_MANIFEST_SHA256,
    modelProfile: "reference-bf16",
    schedulingProfile: "cooperative",
    ditDensePackage,
    vaePackage: Object.freeze({
      manifestUrl: absoluteUrl(VAE_MANIFEST_PATH),
      manifestSha256: VAE_MANIFEST_SHA256,
      runtimeProfile: VAE_RUNTIME_PROFILE,
      maxWindowFrames: 512,
    }),
  });
}

function validateCanonicalRequest(): AceGenerationRequest {
  const bytes = new TextEncoder().encode(OPT_0018_CANONICAL_REQUEST_JSON);
  if (
    bytes.byteLength !== OPT_0018_CANONICAL_REQUEST_BYTES ||
    aceSha256Hex(bytes) !== OPT_0018_CANONICAL_REQUEST_SHA256
  ) throw new Error("OPT-0037 inherited canonical request bytes changed");
  const request = createOpt0018Request() as unknown as AceGenerationRequest;
  assertAceGenerationRequest(request);
  if (JSON.stringify(request) !== OPT_0018_CANONICAL_REQUEST_JSON) {
    throw new Error("OPT-0037 inherited canonical request values changed");
  }
  const dcw = resolveAceDynamicConditionalWeighting(request.planner);
  if (
    dcw.mode !== "double" ||
    dcw.wavelet !== "haar" ||
    dcw.lowBandScale !== 0.05 ||
    dcw.highBandScale !== 0.02
  ) throw new Error("OPT-0037 inherited direct DCW contract changed");
  return Object.freeze(request);
}

function validateRuntimeDiagnostics(
  diagnostics: AceRuntimeDiagnostics,
  arm: ArmDefinition,
): void {
  const capabilities = diagnostics.capabilities;
  if (
    diagnostics.modelManifestSha256 !== MAIN_MANIFEST_SHA256 ||
    diagnostics.ditDenseManifestUrl !== absoluteUrl(arm.manifestPath) ||
    diagnostics.ditDenseManifestSha256 !== arm.manifestSha256 ||
    diagnostics.ditDenseManifestByteLength !== arm.manifestByteLength ||
    diagnostics.ditDenseRuntimeProfile !== arm.runtimeProfile ||
    diagnostics.ditDenseKernelSetId !== arm.kernelSetId ||
    diagnostics.ditDenseLayerBytes !== 3_020_808_192 ||
    diagnostics.ditResidentWeightBytes !== 3_150_917_888 ||
    diagnostics.vaeManifestSha256 !== VAE_MANIFEST_SHA256 ||
    diagnostics.vaeManifestByteLength !== 715_301 ||
    diagnostics.vaeRuntimeProfile !== VAE_RUNTIME_PROFILE ||
    diagnostics.vaeKernelSetId !== VAE_KERNEL_SET_ID ||
    diagnostics.vaeMaxWindowFrames !== 512 ||
    diagnostics.executionProfile.id !== "reference-bf16-subgroups" ||
    diagnostics.schedulingProfile !== "cooperative" ||
    capabilities.adapterInfo.subgroupMinSize !== 32 ||
    capabilities.adapterInfo.subgroupMaxSize !== 32 ||
    !capabilities.deviceFeatures.includes("shader-f16") ||
    !capabilities.deviceFeatures.includes("subgroups")
  ) throw new Error(`OPT-0037 ${arm.role} runtime identity changed`);
}

function validateCheckpoint(checkpoint: AceOpt0018DitCheckpoint): void {
  if (
    checkpoint.schema !== "ace-dit-m2250-checkpoint-v1" ||
    !(checkpoint.finalLatent instanceof Float32Array) ||
    checkpoint.finalLatent.length !== OPT_0037_FINAL_LATENT_ELEMENTS ||
    checkpoint.finalLatent.byteLength !== OPT_0037_FINAL_LATENT_BYTES ||
    checkpoint.finalLatentElementCount !== OPT_0037_FINAL_LATENT_ELEMENTS ||
    checkpoint.finalLatentByteLength !== OPT_0037_FINAL_LATENT_BYTES ||
    checkpoint.finalLatentNonFiniteCount !== 0 ||
    checkpoint.finalLatentNonzeroCount <= 0 ||
    !(checkpoint.finalLatentMaxAbs > 0) ||
    checkpoint.profile.schema !== "ace-dit-m2250-command-profile-v1" ||
    checkpoint.profile.graphCommandBufferCount !== 2_553 ||
    checkpoint.profile.readbackCommandBufferCount !== 1 ||
    checkpoint.profile.totalCommandBufferCount !== 2_554 ||
    checkpoint.profile.timings.length !== 2_553
  ) throw new Error("OPT-0037 inherited M2250 checkpoint topology changed");
  const bytes = new Uint8Array(
    checkpoint.finalLatent.buffer,
    checkpoint.finalLatent.byteOffset,
    checkpoint.finalLatent.byteLength,
  );
  let nonFiniteCount = 0;
  let nonzeroCount = 0;
  let maximumAbsolute = 0;
  for (const value of checkpoint.finalLatent) {
    if (!Number.isFinite(value)) nonFiniteCount += 1;
    else {
      if (value !== 0) nonzeroCount += 1;
      maximumAbsolute = Math.max(maximumAbsolute, Math.abs(value));
    }
  }
  if (
    nonFiniteCount !== 0 ||
    nonzeroCount !== checkpoint.finalLatentNonzeroCount ||
    maximumAbsolute !== checkpoint.finalLatentMaxAbs ||
    aceSha256Hex(bytes) !== checkpoint.finalLatentSha256
  ) throw new Error("OPT-0037 final-latent finite scan or hash changed");
}

function validateConditioningDiagnostic(
  diagnostic: CapturedDiagnostic,
): Readonly<Record<string, unknown>> {
  const details = diagnostic.details;
  if (
    diagnostic.stage !== "semantic-planner" ||
    details.plannerEnabled !== false ||
    details.instrumental !== true ||
    details.generationProfile !== "ace-turbo-v1-correctness" ||
    details.plannerConfiguration !== '{"mode":"disabled"}' ||
    details.textTokenCount !== TEXT_TOKEN_COUNT ||
    details.lyricTokenCount !== LYRIC_TOKEN_COUNT ||
    typeof details.textTokenSha256 !== "string" ||
    typeof details.lyricTokenSha256 !== "string"
  ) throw new Error("OPT-0037 inherited C98 conditioning changed");
  return Object.freeze({
    textTokenCount: TEXT_TOKEN_COUNT,
    textTokenSha256: details.textTokenSha256,
    lyricTokenCount: LYRIC_TOKEN_COUNT,
    lyricTokenSha256: details.lyricTokenSha256,
    packedTimbreRowCount: 1,
    conditionTokenCount: CONDITION_TOKEN_COUNT,
  });
}

function validateFamilyDiagnostic(
  diagnostic: CapturedDiagnostic,
  arm: ArmDefinition,
): void {
  const details = diagnostic.details;
  if (
    diagnostic.stage !== "release-dit" ||
    details.checkpointSchema !== "ace-dit-m2250-checkpoint-v1" ||
    details.denseRuntimeProfile !== arm.runtimeProfile ||
    details.denseKernelSetId !== arm.kernelSetId ||
    details.gemmBackend !== arm.expectedGemmBackend ||
    details.tokens !== 2_250 ||
    details.conditionTokens !== CONDITION_TOKEN_COUNT ||
    details.graphCommandBufferCount !== 2_553 ||
    details.readbackCommandBufferCount !== 1 ||
    details.totalCommandBufferCount !== 2_554
  ) throw new Error(`OPT-0037 ${arm.role} profile diagnostic changed`);
}

function compactStageWalls(
  timings: readonly AceStageTiming[],
): readonly Readonly<Record<string, unknown>>[] {
  if (
    timings.length === 0 ||
    timings.some((timing) =>
      FORBIDDEN_POST_DIT_STAGES.has(timing.stage) ||
      !Number.isFinite(timing.wallMs) ||
      timing.wallMs < 0
    ) ||
    !timings.some((timing) => timing.stage === "dit-denoise") ||
    !timings.some((timing) => timing.stage === "release-dit")
  ) throw new Error("OPT-0037 stage-wall checkpoint boundary changed");
  return Object.freeze(timings.map((timing) => Object.freeze({
    stage: timing.stage,
    wallMs: timing.wallMs,
    ...(timing.submittedGpuMs === undefined
      ? {}
      : { submittedGpuMs: timing.submittedGpuMs }),
    ...(timing.cooperativeIdleMs === undefined
      ? {}
      : { cooperativeIdleMs: timing.cooperativeIdleMs }),
  })));
}

function armIdentity(arm: ArmDefinition): Readonly<Record<string, unknown>> {
  return Object.freeze({
    role: arm.role,
    manifestPath: arm.manifestPath,
    manifestSha256: arm.manifestSha256,
    manifestByteLength: arm.manifestByteLength,
    runtimeProfile: arm.runtimeProfile,
    kernelSetId: arm.kernelSetId,
  });
}

function absoluteUrl(path: string): string {
  return new URL(path, self.location.href).href;
}

function nextLifecycleOrdinal(): number {
  lifecycleOrdinal += 1;
  return lifecycleOrdinal;
}

function postProgress(message: string): void {
  self.postMessage({ type: "progress", message });
}

function readRequiredString(
  value: Readonly<Record<string, unknown>>,
  key: string,
): string {
  const field = value[key];
  if (typeof field !== "string" || field.length === 0) {
    throw new Error(`OPT-0037 receipt is missing ${key}`);
  }
  return field;
}

function sameJson(left: unknown, right: unknown): boolean {
  return JSON.stringify(left) === JSON.stringify(right);
}

function isRecord(value: unknown): value is Readonly<Record<string, unknown>> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
