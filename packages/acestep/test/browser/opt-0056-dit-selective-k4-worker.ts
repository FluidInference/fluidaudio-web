/// <reference lib="webworker" />
/// <reference types="@webgpu/types" />
/// <reference types="vite/client" />

import {
  assertAceGenerationRequest,
  type AceGenerationRequest,
} from "../../src/api.js";
import { aceSha256Hex } from "../../src/model/sha256.js";
import type { AceDiagnostic, AceRuntimeDiagnostics } from
  "../../src/runtime/diagnostics.js";
import type { AceWorkerConfiguration } from
  "../../src/runtime/protocol.js";
import type {
  AceGenerationProgress,
  AceInitializationProgress,
  AceStageTiming,
} from "../../src/runtime/stages.js";
import {
  createAceWebGpuPipelineBackend,
  type AceOpt0056DitCheckpoint,
  type AceWebGpuGenerationContext,
} from "../../src/runtime/webgpu-pipeline.js";
import { AceOpt0009DenseGemmKernel } from
  "../../src/webgpu/kernels/dit-dense-fp16.js";
import {
  ACE_OPT_0056_DENSE_K4_EXACT_KERNEL_ID,
  AceOpt0056DenseK4ExactKernel,
} from "../../src/webgpu/kernels/dit-dense-fp16-k4-exact.js";
import {
  ACE_OPT_0056_SELECTIVE_DENSE_KERNEL_SET_ID,
} from "../../src/webgpu/kernels/dit-dense-fp16-k4-selective-exact.js";
import { ACE_OPT_0037_DENSE_K4_KERNEL_ID } from
  "../../src/webgpu/kernels/dit-dense-fp16-k4-production.js";
import type {
  AceGemmDispatch,
  AceGemmShape,
} from "../../src/webgpu/kernels/gemm.js";
import {
  OPT_0018_CANONICAL_REQUEST_BYTES,
  OPT_0018_CANONICAL_REQUEST_JSON,
  OPT_0018_CANONICAL_REQUEST_SHA256,
  createOpt0018Request,
  serializeOpt0018Failure,
} from "./opt-0018-dit-m2250-production-family-profile.js";
import {
  OPT_0056_ALL_K4_KERNEL_SET_ID,
  OPT_0056_ALL_K4_RUNTIME_PROFILE,
  OPT_0056_EVALUATIONS,
  OPT_0056_LATENT_BYTES,
  OPT_0056_LATENT_ELEMENTS,
  OPT_0056_REV7_KERNEL_SET_ID,
  OPT_0056_REV7_MANIFEST_BYTES,
  OPT_0056_REV7_MANIFEST_PATH,
  OPT_0056_REV7_MANIFEST_SHA256,
  OPT_0056_REV7_RUNTIME_PROFILE,
  OPT_0056_REV8_MANIFEST_BYTES,
  OPT_0056_REV8_MANIFEST_PATH,
  OPT_0056_REV8_MANIFEST_SHA256,
  OPT_0056_SELECTIVE_KERNEL_SET_ID,
  OPT_0056_SELECTIVE_RUNTIME_PROFILE,
  compareOpt0056Trajectory,
  exactOpt0056TrajectoryIdentity,
} from "./opt-0056-dit-selective-k4-contract.js";

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
const CONDITIONING_DIAGNOSTIC_CODE =
  "ACE_PLANNER_CONDITIONING_RESOLVED" as const;
const TRAJECTORY_DIAGNOSTIC_CODE = "ACE_DIT_OPT0056_TRAJECTORY" as const;
const FORBIDDEN_POST_DIT_STAGES = new Set([
  "vae-load",
  "vae-decode",
  "wav-encode",
  "cleanup",
  "done",
]);
const GUARD_BYTES = 256;
const GUARD_U32 = 0xa55a_c33c;
const PREFILL_U32 = 0x7fc0_0056;

type ArmRole =
  | "rev7-control"
  | "rev8-all-k4"
  | "rev8-selective"
  | "rev8-selective-repeat";

interface ArmDefinition {
  readonly role: ArmRole;
  readonly manifestPath:
    | typeof OPT_0056_REV7_MANIFEST_PATH
    | typeof OPT_0056_REV8_MANIFEST_PATH;
  readonly manifestSha256:
    | typeof OPT_0056_REV7_MANIFEST_SHA256
    | typeof OPT_0056_REV8_MANIFEST_SHA256;
  readonly manifestByteLength:
    | typeof OPT_0056_REV7_MANIFEST_BYTES
    | typeof OPT_0056_REV8_MANIFEST_BYTES;
  readonly runtimeProfile:
    | typeof OPT_0056_REV7_RUNTIME_PROFILE
    | typeof OPT_0056_ALL_K4_RUNTIME_PROFILE
    | typeof OPT_0056_SELECTIVE_RUNTIME_PROFILE;
  readonly kernelSetId:
    | typeof OPT_0056_REV7_KERNEL_SET_ID
    | typeof OPT_0056_ALL_K4_KERNEL_SET_ID
    | typeof OPT_0056_SELECTIVE_KERNEL_SET_ID;
  readonly selective: boolean;
}

interface ArmResult {
  readonly trajectory: readonly Float32Array<ArrayBuffer>[];
  readonly conditioningAuthority: Readonly<Record<string, unknown>>;
  readonly receipt: Readonly<Record<string, unknown>>;
  readonly lifecycle: Readonly<Record<string, number>>;
}

const CONTROL: ArmDefinition = Object.freeze({
  role: "rev7-control",
  manifestPath: OPT_0056_REV7_MANIFEST_PATH,
  manifestSha256: OPT_0056_REV7_MANIFEST_SHA256,
  manifestByteLength: OPT_0056_REV7_MANIFEST_BYTES,
  runtimeProfile: OPT_0056_REV7_RUNTIME_PROFILE,
  kernelSetId: OPT_0056_REV7_KERNEL_SET_ID,
  selective: false,
});
const ALL_K4: ArmDefinition = Object.freeze({
  role: "rev8-all-k4",
  manifestPath: OPT_0056_REV8_MANIFEST_PATH,
  manifestSha256: OPT_0056_REV8_MANIFEST_SHA256,
  manifestByteLength: OPT_0056_REV8_MANIFEST_BYTES,
  runtimeProfile: OPT_0056_ALL_K4_RUNTIME_PROFILE,
  kernelSetId: OPT_0056_ALL_K4_KERNEL_SET_ID,
  selective: false,
});
const SELECTIVE: ArmDefinition = Object.freeze({
  role: "rev8-selective",
  manifestPath: OPT_0056_REV8_MANIFEST_PATH,
  manifestSha256: OPT_0056_REV8_MANIFEST_SHA256,
  manifestByteLength: OPT_0056_REV8_MANIFEST_BYTES,
  runtimeProfile: OPT_0056_SELECTIVE_RUNTIME_PROFILE,
  kernelSetId: OPT_0056_SELECTIVE_KERNEL_SET_ID,
  selective: true,
});
const SELECTIVE_REPEAT: ArmDefinition = Object.freeze({
  ...SELECTIVE,
  role: "rev8-selective-repeat",
});

let workerState: "idle" | "running" | "settled" = "idle";
let lifecycleOrdinal = 0;

self.addEventListener("message", (event: MessageEvent<unknown>) => {
  if (
    workerState !== "idle" ||
    !isRecord(event.data) ||
    event.data.type !== "run"
  ) return;
  workerState = "running";
  void runGate().then(
    (result) => {
      workerState = "settled";
      self.postMessage({ type: "gate-complete", result });
    },
    (error: unknown) => {
      workerState = "settled";
      self.postMessage({ type: "failed", error: serializeOpt0018Failure(error) });
    },
  );
});

async function runGate(): Promise<Readonly<Record<string, unknown>>> {
  const request = validateCanonicalRequest();
  postProgress("primitive preflight: proving exact U32 identity on four full shapes");
  const primitive = await runPrimitiveIdentityGate();
  const arms: ArmResult[] = [];
  for (const definition of [CONTROL, ALL_K4, SELECTIVE, SELECTIVE_REPEAT]) {
    postProgress(`${definition.role}: authenticating and running eight evaluations`);
    const result = await runArm(definition, request);
    const previous = arms.at(-1);
    if (
      previous !== undefined &&
      previous.lifecycle["disposeCompletedOrdinal"]! >=
        result.lifecycle["backendCreatedOrdinal"]!
    ) throw new Error(`OPT-0056 ${definition.role} overlapped its predecessor`);
    arms.push(result);
  }
  const [control, allK4, selective, repeat] = arms as [
    ArmResult,
    ArmResult,
    ArmResult,
    ArmResult,
  ];
  if (
    !sameJson(control.conditioningAuthority, allK4.conditioningAuthority) ||
    !sameJson(control.conditioningAuthority, selective.conditioningAuthority) ||
    !sameJson(control.conditioningAuthority, repeat.conditioningAuthority)
  ) throw new Error("OPT-0056 arms did not resolve identical conditioning");

  postProgress("comparing eight exact/all-K4/selective sampler trajectories");
  const allK4Comparison = compareOpt0056Trajectory(
    control.trajectory,
    allK4.trajectory,
  );
  const selectiveComparison = compareOpt0056Trajectory(
    control.trajectory,
    selective.trajectory,
  );
  const selectiveDeterministic = exactOpt0056TrajectoryIdentity(
    selective.trajectory,
    repeat.trajectory,
  );
  const passed = primitive.passed === true &&
    selectiveComparison.passedFinalEnvelope && selectiveDeterministic;
  return Object.freeze({
    schema: "ace-opt-0056-selective-k4-exact-down-trajectory-gate-v1",
    experimentId: "OPT-0056",
    status: passed ? "passed-correctness-gate" : "failed-correctness-gate",
    passed,
    fixedExecutionOrder: Object.freeze([
      CONTROL.role,
      ALL_K4.role,
      SELECTIVE.role,
      SELECTIVE_REPEAT.role,
    ]),
    identity: Object.freeze({
      requestCanonicalJson: OPT_0018_CANONICAL_REQUEST_JSON,
      requestSha256: OPT_0018_CANONICAL_REQUEST_SHA256,
      requestByteLength: OPT_0018_CANONICAL_REQUEST_BYTES,
      mainManifestSha256: MAIN_MANIFEST_SHA256,
      vaeManifestSha256: VAE_MANIFEST_SHA256,
      control: armIdentity(CONTROL),
      allK4: armIdentity(ALL_K4),
      selective: armIdentity(SELECTIVE),
    }),
    primitive,
    arms: Object.freeze({
      control: control.receipt,
      allK4: allK4.receipt,
      selective: selective.receipt,
      selectiveRepeat: repeat.receipt,
    }),
    correctness: Object.freeze({
      evaluationCount: OPT_0056_EVALUATIONS,
      allK4Comparison,
      selectiveComparison,
      selectiveDeterministic,
      unchangedOpt0037FinalThresholds: true,
      finalEnvelopePassed: selectiveComparison.passedFinalEnvelope,
      passed,
    }),
    lifecycle: Object.freeze({
      sequentialNonOverlappingArms: true,
      gpuPackageRuntimeCoResidency: false,
      allBackendsAndDevicesDisposed: true,
      onlyDetachedCpuTrajectoriesRetainedBetweenArms: true,
    }),
    topology: Object.freeze({
      graphCommandBufferCountPerArm: 2_553,
      finalReadbackCommandBufferCountPerArm: 1,
      totalCommandBufferCountPerArm: 2_554,
      samplerSnapshotCopyCountPerArm: 8,
      samplerSnapshotExtraSubmitsPerArm: 0,
      samplerSnapshotExtraDrainsPerArm: 0,
    }),
    decision: passed
      ? "correctness-pass-authorizes-balanced-speed-screen-only"
      : "negative-stop-selective-selector",
    scope: Object.freeze({
      vaeWeightsAcquired: false,
      vaeExecuted: false,
      audioExecuted: false,
      productDefaultChanged: false,
      listeningApproval: false,
      speedupCalculated: false,
    }),
  });
}

async function runArm(
  arm: ArmDefinition,
  request: AceGenerationRequest,
): Promise<ArmResult> {
  const backendCreatedOrdinal = nextOrdinal();
  const backend = createAceWebGpuPipelineBackend();
  const controller = new AbortController();
  const privateStop = Object.freeze({ arm: arm.role });
  let initializedOrdinal = 0;
  let checkpointOrdinal = 0;
  let generationCleanupCompletedOrdinal = 0;
  let disposeStartedOrdinal = 0;
  let disposeCompletedOrdinal = 0;
  let checkpoint: AceOpt0056DitCheckpoint | undefined;
  let diagnostics: AceRuntimeDiagnostics | undefined;
  let conditioning: Readonly<Record<string, unknown>> | undefined;
  let trajectoryDiagnostic: Readonly<Record<string, unknown>> | undefined;
  let initializationProgressEventCount = 0;
  let generationProgressEventCount = 0;
  let forbiddenProgress = false;
  const diagnosticCodes: string[] = [];
  const fatalDiagnostics: string[] = [];
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
        if (diagnostic.severity === "error") fatalDiagnostics.push(diagnostic.code);
      },
    });
    initializedOrdinal = nextOrdinal();
    validateDiagnostics(diagnostics, arm);
    const context: AceWebGpuGenerationContext = {
      signal: controller.signal,
      captureTrace: true,
      onProgress: (progress: AceGenerationProgress) => {
        generationProgressEventCount += 1;
        if (FORBIDDEN_POST_DIT_STAGES.has(progress.stage)) forbiddenProgress = true;
      },
      onDiagnostic: (diagnostic: AceDiagnostic) => {
        diagnosticCodes.push(diagnostic.code);
        if (diagnostic.severity === "error") fatalDiagnostics.push(diagnostic.code);
        if (diagnostic.details === undefined) return;
        if (diagnostic.code === CONDITIONING_DIAGNOSTIC_CODE) {
          if (conditioning !== undefined) throw new Error("duplicate conditioning diagnostic");
          conditioning = validateConditioning(diagnostic);
        } else if (diagnostic.code === TRAJECTORY_DIAGNOSTIC_CODE) {
          if (trajectoryDiagnostic !== undefined) throw new Error("duplicate trajectory diagnostic");
          trajectoryDiagnostic = diagnostic.details;
        }
      },
      opt0056DitRun: {
        onCheckpoint(value) {
          if (checkpoint !== undefined) throw new Error(`OPT-0056 ${arm.role} checkpoint repeated`);
          validateCheckpoint(value, arm);
          checkpoint = value;
          checkpointOrdinal = nextOrdinal();
          controller.abort(privateStop);
        },
      },
    };
    try {
      await backend.generate(request, context);
      throw new Error(`OPT-0056 ${arm.role} continued into VAE`);
    } catch (error) {
      if (error !== privateStop) throw error;
      generationCleanupCompletedOrdinal = nextOrdinal();
    }
  } catch (error) {
    primaryError = error;
  } finally {
    disposeStartedOrdinal = nextOrdinal();
    try {
      await backend.dispose();
    } catch (error) {
      cleanupError = error;
    }
    disposeCompletedOrdinal = nextOrdinal();
  }
  if (primaryError !== undefined || cleanupError !== undefined) {
    if (primaryError !== undefined && cleanupError !== undefined) {
      throw new AggregateError([primaryError, cleanupError], `${arm.role} run/dispose failed`);
    }
    throw primaryError ?? cleanupError;
  }
  if (
    diagnostics === undefined || checkpoint === undefined ||
    conditioning === undefined || trajectoryDiagnostic === undefined ||
    initializedOrdinal === 0 || checkpointOrdinal === 0 ||
    generationCleanupCompletedOrdinal === 0 || forbiddenProgress ||
    fatalDiagnostics.length !== 0
  ) throw new Error(`OPT-0056 ${arm.role} checkpoint boundary incomplete`);
  validateTrajectoryDiagnostic(trajectoryDiagnostic, arm);
  const lifecycle = Object.freeze({
    backendCreatedOrdinal,
    initializedOrdinal,
    checkpointOrdinal,
    generationCleanupCompletedOrdinal,
    disposeStartedOrdinal,
    disposeCompletedOrdinal,
  });
  return Object.freeze({
    trajectory: Object.freeze(checkpoint.evaluations.map(({ latent }) => latent)),
    conditioningAuthority: conditioning,
    lifecycle,
    receipt: Object.freeze({
      role: arm.role,
      authenticatedPackage: armIdentity(arm),
      runtimeDiagnostics: Object.freeze({
        ditDenseManifestSha256: diagnostics.ditDenseManifestSha256,
        ditDenseManifestByteLength: diagnostics.ditDenseManifestByteLength,
        ditDenseRuntimeProfile: diagnostics.ditDenseRuntimeProfile,
        ditDenseKernelSetId: diagnostics.ditDenseKernelSetId,
        executionProfileId: diagnostics.executionProfile.id,
        schedulingProfile: diagnostics.schedulingProfile,
      }),
      conditioningAuthority: conditioning,
      evaluationHashes: Object.freeze(checkpoint.evaluations.map(
        ({ evaluation, latentSha256 }) => Object.freeze({ evaluation, latentSha256 }),
      )),
      finalLatentSha256: checkpoint.finalLatentSha256,
      finalLatentNonFiniteCount: checkpoint.finalLatentNonFiniteCount,
      finalLatentMaxAbs: checkpoint.finalLatentMaxAbs,
      routeProfile: checkpoint.denseRouteProfile === undefined
        ? null
        : compactRouteProfile(checkpoint.denseRouteProfile),
      graphWalls: Object.freeze({
        graphSubmitThroughDrainMs: checkpoint.profile.graphSubmitThroughDrainMs,
        graphWallMs: checkpoint.profile.graphWallMs,
      }),
      stageWalls: compactStageWalls(checkpoint.stageTimings),
      lifecycle,
      evidence: Object.freeze({
        initializationProgressEventCount,
        generationProgressEventCount,
        diagnosticCodes: Object.freeze(diagnosticCodes),
        ditDestroyedBeforeCheckpoint: true,
        pipelineCleanupAwaitedBeforeDispose: true,
        backendAndDeviceDisposeAwaited: true,
        vaeWeightAcquireStarted: false,
      }),
    }),
  });
}

function validateCheckpoint(
  checkpoint: AceOpt0056DitCheckpoint,
  arm: ArmDefinition,
): void {
  if (
    checkpoint.schema !== "ace-dit-opt0056-m2250-trajectory-checkpoint-v1" ||
    checkpoint.finalLatent.length !== OPT_0056_LATENT_ELEMENTS ||
    checkpoint.finalLatent.byteLength !== OPT_0056_LATENT_BYTES ||
    checkpoint.finalLatentNonFiniteCount !== 0 ||
    checkpoint.evaluations.length !== OPT_0056_EVALUATIONS ||
    checkpoint.graphCommandBufferCount !== 2_553 ||
    checkpoint.readbackCommandBufferCount !== 1 ||
    checkpoint.totalCommandBufferCount !== 2_554 ||
    checkpoint.snapshotCopyCount !== 8 ||
    checkpoint.snapshotExtraCommandBufferCount !== 0 ||
    checkpoint.snapshotExtraQueueDrainCount !== 0 ||
    checkpoint.profile.timings.length !== 2_553 ||
    (arm.selective !== (checkpoint.denseRouteProfile !== undefined))
  ) throw new Error(`OPT-0056 ${arm.role} checkpoint topology changed`);
  for (const snapshot of checkpoint.evaluations) {
    const bytes = new Uint8Array(
      snapshot.latent.buffer,
      snapshot.latent.byteOffset,
      snapshot.latent.byteLength,
    );
    if (
      snapshot.latent.length !== OPT_0056_LATENT_ELEMENTS ||
      snapshot.latent.byteLength !== OPT_0056_LATENT_BYTES ||
      snapshot.nonFiniteCount !== 0 || snapshot.nonzeroCount <= 0 ||
      aceSha256Hex(bytes) !== snapshot.latentSha256
    ) throw new Error(`OPT-0056 ${arm.role} evaluation ${snapshot.evaluation} invalid`);
  }
  if (checkpoint.evaluations.at(-1)!.latentSha256 !== checkpoint.finalLatentSha256) {
    throw new Error(`OPT-0056 ${arm.role} final tap differs from final readback`);
  }
  if (checkpoint.denseRouteProfile !== undefined) {
    const profile = checkpoint.denseRouteProfile;
    if (
      profile.kernelSetId !== ACE_OPT_0056_SELECTIVE_DENSE_KERNEL_SET_ID ||
      profile.routeCount !== 216 || profile.dispatchCount !== 1_728 ||
      profile.approximateRouteCount !== 192 || profile.exactDownRouteCount !== 24 ||
      profile.routes.length !== 216 ||
      profile.routes.some((route) =>
        route.evaluationDispatchCount !== 8 || route.evaluationLabels.length !== 8 ||
        (route.operation === "mlp-down-projection"
          ? route.owner !== "opt-0056-k4-exact-fp32" ||
            route.kernelId !== ACE_OPT_0056_DENSE_K4_EXACT_KERNEL_ID ||
            route.inner !== 6_144 || route.columns !== 2_048
          : route.owner !== "opt-0032-k4-fp16-partials" ||
            route.kernelId !== ACE_OPT_0037_DENSE_K4_KERNEL_ID)
      )
    ) throw new Error("OPT-0056 selective route ownership changed");
    const owners = checkpoint.profile.descriptorTable.descriptors.flatMap(
      ({ members }) => members.map(({ backend }) => backend),
    );
    if (
      owners.filter((owner) => owner === "opt-0056-k4-exact-fp32").length !== 192 ||
      owners.filter((owner) => owner === "opt-0032-k4-fp16-partials").length !== 1_536
    ) throw new Error("OPT-0056 descriptor owner counts changed");
  }
}

function configurationForArm(arm: ArmDefinition): AceWorkerConfiguration {
  const ditDensePackage: AceWorkerConfiguration["ditDensePackage"] =
    arm.runtimeProfile === OPT_0056_REV7_RUNTIME_PROFILE
      ? Object.freeze({
          manifestUrl: absoluteUrl(OPT_0056_REV7_MANIFEST_PATH),
          manifestSha256: OPT_0056_REV7_MANIFEST_SHA256,
          runtimeProfile: OPT_0056_REV7_RUNTIME_PROFILE,
        })
      : arm.runtimeProfile === OPT_0056_ALL_K4_RUNTIME_PROFILE
        ? Object.freeze({
            manifestUrl: absoluteUrl(OPT_0056_REV8_MANIFEST_PATH),
            manifestSha256: OPT_0056_REV8_MANIFEST_SHA256,
            runtimeProfile: OPT_0056_ALL_K4_RUNTIME_PROFILE,
          })
        : Object.freeze({
            manifestUrl: absoluteUrl(OPT_0056_REV8_MANIFEST_PATH),
            manifestSha256: OPT_0056_REV8_MANIFEST_SHA256,
            runtimeProfile: OPT_0056_SELECTIVE_RUNTIME_PROFILE,
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

function validateDiagnostics(
  diagnostics: AceRuntimeDiagnostics,
  arm: ArmDefinition,
): void {
  if (
    diagnostics.modelManifestSha256 !== MAIN_MANIFEST_SHA256 ||
    diagnostics.ditDenseManifestSha256 !== arm.manifestSha256 ||
    diagnostics.ditDenseManifestByteLength !== arm.manifestByteLength ||
    diagnostics.ditDenseRuntimeProfile !== arm.runtimeProfile ||
    diagnostics.ditDenseKernelSetId !== arm.kernelSetId ||
    diagnostics.vaeManifestSha256 !== VAE_MANIFEST_SHA256 ||
    diagnostics.vaeRuntimeProfile !== VAE_RUNTIME_PROFILE ||
    diagnostics.vaeKernelSetId !== VAE_KERNEL_SET_ID ||
    diagnostics.executionProfile.id !== "reference-bf16-subgroups" ||
    diagnostics.schedulingProfile !== "cooperative" ||
    diagnostics.capabilities.schedulingProfile !== "cooperative" ||
    diagnostics.capabilities.adapterInfo.subgroupMinSize !== 32 ||
    diagnostics.capabilities.adapterInfo.subgroupMaxSize !== 32 ||
    !diagnostics.capabilities.deviceFeatures.includes("shader-f16") ||
    !diagnostics.capabilities.deviceFeatures.includes("subgroups")
  ) throw new Error(`OPT-0056 ${arm.role} runtime identity changed`);
}

function validateConditioning(
  diagnostic: AceDiagnostic,
): Readonly<Record<string, unknown>> {
  const details = diagnostic.details!;
  if (
    diagnostic.stage !== "semantic-planner" || details.plannerEnabled !== false ||
    details.instrumental !== true || details.generationProfile !== "ace-turbo-v1-correctness" ||
    details.textTokenCount !== 82 || details.lyricTokenCount !== 15 ||
    typeof details.textTokenSha256 !== "string" ||
    typeof details.lyricTokenSha256 !== "string"
  ) throw new Error("OPT-0056 conditioning authority changed");
  return Object.freeze({
    textTokenCount: 82,
    textTokenSha256: details.textTokenSha256,
    lyricTokenCount: 15,
    lyricTokenSha256: details.lyricTokenSha256,
    conditionTokenCount: 98,
  });
}

function validateTrajectoryDiagnostic(
  details: Readonly<Record<string, unknown>>,
  arm: ArmDefinition,
): void {
  if (
    details.checkpointSchema !== "ace-dit-opt0056-m2250-trajectory-checkpoint-v1" ||
    details.evaluationCount !== 8 || details.graphCommandBufferCount !== 2_553 ||
    details.readbackCommandBufferCount !== 1 || details.totalCommandBufferCount !== 2_554 ||
    details.snapshotCopyCount !== 8 || details.snapshotExtraCommandBufferCount !== 0 ||
    details.snapshotExtraQueueDrainCount !== 0 ||
    (arm.selective
      ? details.denseRouteCount !== 216 || details.exactDownRouteCount !== 24
      : details.denseRouteCount !== 0 || details.exactDownRouteCount !== 0)
  ) throw new Error(`OPT-0056 ${arm.role} trajectory diagnostic changed`);
}

async function runPrimitiveIdentityGate(): Promise<Readonly<Record<string, unknown>>> {
  const endianMarker = new Uint16Array([0x0102]);
  if (new Uint8Array(endianMarker.buffer)[0] !== 0x02) {
    throw new Error("OPT-0056 primitive preflight requires little-endian typed arrays");
  }
  if (navigator.gpu === undefined) throw new Error("WebGPU unavailable");
  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: "high-performance",
    forceFallbackAdapter: false,
  });
  if (
    adapter === null || !adapter.features.has("shader-f16") ||
    !adapter.features.has("subgroups") || adapter.info.subgroupMinSize !== 32 ||
    adapter.info.subgroupMaxSize !== 32
  ) throw new Error("OPT-0056 primitive preflight requires fixed32 shader-f16");
  const shapes = Object.freeze([
    { id: "h-h", rows: 2_250, inner: 2_048, columns: 2_048 },
    { id: "h-1024", rows: 2_250, inner: 2_048, columns: 1_024 },
    { id: "h-6144", rows: 2_250, inner: 2_048, columns: 6_144 },
    { id: "6144-h", rows: 2_250, inner: 6_144, columns: 2_048 },
  ] as const);
  const shapeIds = shapes.map(({ id }) => id);
  if (
    shapes.length !== 4 || new Set(shapeIds).size !== 4 ||
    shapeIds.join(",") !== "h-h,h-1024,h-6144,6144-h"
  ) throw new Error("OPT-0056 primitive shape inventory changed");
  const maximumStorage = Math.max(...shapes.flatMap((shape) => [
    shape.rows * shape.inner * 4,
    shape.inner * shape.columns * 2,
    shape.rows * shape.columns * 4,
  ]));
  const device = await adapter.requestDevice({
    label: "ace-opt-0056-primitive-identity-device",
    requiredFeatures: ["shader-f16", "subgroups"],
    requiredLimits: {
      maxBufferSize: maximumStorage + 2 * GUARD_BYTES,
      maxStorageBufferBindingSize: maximumStorage,
      maxComputeInvocationsPerWorkgroup: 128,
      maxComputeWorkgroupSizeX: 128,
    },
  });
  const uncapturedErrors: string[] = [];
  const deviceLosses: string[] = [];
  const onUncapturedError = (event: GPUUncapturedErrorEvent): void => {
    uncapturedErrors.push(`${event.error.constructor.name}: ${event.error.message}`);
  };
  device.addEventListener("uncapturederror", onUncapturedError);
  void device.lost.then((info) => {
    if (info.reason !== "destroyed") {
      deviceLosses.push(`${String(info.reason)}: ${info.message}`);
    }
  });
  let control: AceOpt0009DenseGemmKernel | undefined;
  let exact: AceOpt0056DenseK4ExactKernel | undefined;
  const cases: Readonly<Record<string, unknown>>[] = [];
  try {
    control = AceOpt0009DenseGemmKernel.create(device, {
      subgroupMinSize: 32,
      subgroupMaxSize: 32,
    });
    exact = AceOpt0056DenseK4ExactKernel.create(device, {
      subgroupMinSize: 32,
      subgroupMaxSize: 32,
    });
    for (let ordinal = 0; ordinal < shapes.length; ordinal += 1) {
      postProgress(`primitive preflight ${ordinal + 1}/4: ${shapes[ordinal]!.id}`);
      cases.push(await runPrimitiveShape(
        device,
        control,
        exact,
        shapes[ordinal]!,
        ordinal,
      ));
    }
    await device.queue.onSubmittedWorkDone();
    await new Promise<void>((resolve) => setTimeout(resolve, 0));
    if (uncapturedErrors.length !== 0 || deviceLosses.length !== 0) {
      throw new Error(`OPT-0056 primitive GPU failure: ${JSON.stringify({
        uncapturedErrors,
        deviceLosses,
      })}`);
    }
  } finally {
    control?.destroy();
    exact?.destroy();
    device.removeEventListener("uncapturederror", onUncapturedError);
    device.destroy();
  }
  return Object.freeze({
    passed: cases.every((value) => value["passed"] === true),
    shapeCount: cases.length,
    shapeIds: Object.freeze(shapeIds),
    comparedOutputU32Count: cases.reduce(
      (sum, value) => sum + Number(value["outputElements"]),
      0,
    ),
    exactIncreasingKFp32: true,
    exhaustiveLayoutForwardInverseCoveredByFocusedTests: true,
    weightFixture: "deterministic-synthetic-logical-fp16",
    logicalBitsPackedIntoRev7AndAuthenticatedRev8Layouts: true,
    actualPackageWeightIsolatedU32IdentityMeasured: false,
    authenticatedRev8PackageExercisedByFullTrajectory: true,
    uncapturedGpuErrorCount: 0,
    unexpectedDeviceLossCount: 0,
    cases: Object.freeze(cases),
  });
}

async function runPrimitiveShape(
  device: GPUDevice,
  control: AceOpt0009DenseGemmKernel,
  exact: AceOpt0056DenseK4ExactKernel,
  spec: Readonly<{ id: string; rows: number; inner: number; columns: number }>,
  ordinal: number,
): Promise<Readonly<Record<string, unknown>>> {
  const shape: AceGemmShape = spec;
  const activationBytes = spec.rows * spec.inner * 4;
  const weightBytes = spec.inner * spec.columns * 2;
  const outputBytes = spec.rows * spec.columns * 4;
  const totalOutputBytes = outputBytes + 2 * GUARD_BYTES;
  const buffers: GPUBuffer[] = [];
  const create = (descriptor: GPUBufferDescriptor): GPUBuffer => {
    const buffer = device.createBuffer(descriptor);
    buffers.push(buffer);
    return buffer;
  };
  try {
    const activation = create({
      label: `${spec.id}-activation`, size: activationBytes,
      usage: GPUBufferUsage.STORAGE, mappedAtCreation: true,
    });
    fillActivation(new Float32Array(activation.getMappedRange()), spec, ordinal);
    activation.unmap();
    const rev7Weight = create({
      label: `${spec.id}-rev7-weight`, size: weightBytes,
      usage: GPUBufferUsage.STORAGE, mappedAtCreation: true,
    });
    fillRev7Weight(new Uint16Array(rev7Weight.getMappedRange()), spec, ordinal);
    rev7Weight.unmap();
    const rev8Weight = create({
      label: `${spec.id}-rev8-weight`, size: weightBytes,
      usage: GPUBufferUsage.STORAGE, mappedAtCreation: true,
    });
    fillRev8Weight(new Uint16Array(rev8Weight.getMappedRange()), spec, ordinal);
    rev8Weight.unmap();
    const output = create({
      label: `${spec.id}-output`, size: totalOutputBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST,
    });
    const prefill = create({
      label: `${spec.id}-prefill`, size: totalOutputBytes,
      usage: GPUBufferUsage.COPY_SRC, mappedAtCreation: true,
    });
    const prefillWords = new Uint32Array(prefill.getMappedRange());
    prefillWords.fill(GUARD_U32);
    prefillWords.fill(PREFILL_U32, GUARD_BYTES / 4, GUARD_BYTES / 4 + spec.rows * spec.columns);
    prefill.unmap();
    const readback = create({
      label: `${spec.id}-readback`, size: totalOutputBytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const bindings = (weight: GPUBuffer) => Object.freeze({
      activation: Object.freeze({ buffer: activation, offset: 0, size: activationBytes }),
      weight: Object.freeze({ buffer: weight, offset: 0, size: weightBytes }),
      output: Object.freeze({ buffer: output, offset: GUARD_BYTES, size: outputBytes }),
    });
    const controlDispatch = await control.createDispatch(
      `${spec.id}-rev7-control`, shape, bindings(rev7Weight),
    );
    const exactDispatch = await exact.createDispatch(
      `${spec.id}-rev8-exact`, shape, bindings(rev8Weight),
    );
    const a = await executePrimitive(device, controlDispatch, prefill, output, readback, totalOutputBytes);
    const b = await executePrimitive(device, exactDispatch, prefill, output, readback, totalOutputBytes);
    const outputStart = GUARD_BYTES / 4;
    const outputElements = spec.rows * spec.columns;
    const bFloat = new Float32Array(b.buffer, b.byteOffset, b.length);
    let mismatches = 0;
    let nonFinite = 0;
    let prefillRemaining = 0;
    for (let index = 0; index < outputElements; index += 1) {
      const ai = a[outputStart + index]!;
      const bi = b[outputStart + index]!;
      if (ai !== bi) mismatches += 1;
      if (bi === PREFILL_U32) prefillRemaining += 1;
      if (!Number.isFinite(bFloat[outputStart + index])) nonFinite += 1;
    }
    const guardsClean = a.slice(0, outputStart).every((word) => word === GUARD_U32) &&
      b.slice(0, outputStart).every((word) => word === GUARD_U32) &&
      a.slice(outputStart + outputElements).every((word) => word === GUARD_U32) &&
      b.slice(outputStart + outputElements).every((word) => word === GUARD_U32);
    const passed = mismatches === 0 && nonFinite === 0 &&
      prefillRemaining === 0 && guardsClean;
    if (!passed) throw new Error(`OPT-0056 primitive ${spec.id} U32 identity failed`);
    return Object.freeze({
      id: spec.id,
      shape: Object.freeze({ ...spec }),
      outputElements,
      mismatches,
      nonFinite,
      prefillRemaining,
      guardsClean,
      outputSha256: aceSha256Hex(new Uint8Array(
        b.buffer,
        b.byteOffset + GUARD_BYTES,
        outputBytes,
      )),
      passed,
    });
  } finally {
    buffers.forEach((buffer) => buffer.destroy());
  }
}

async function executePrimitive(
  device: GPUDevice,
  dispatch: Pick<AceGemmDispatch, "encode">,
  prefill: GPUBuffer,
  output: GPUBuffer,
  readback: GPUBuffer,
  bytes: number,
): Promise<Uint32Array<ArrayBuffer>> {
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(prefill, 0, output, 0, bytes);
  const pass = encoder.beginComputePass();
  dispatch.encode(pass);
  pass.end();
  encoder.copyBufferToBuffer(output, 0, readback, 0, bytes);
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  await readback.mapAsync(GPUMapMode.READ, 0, bytes);
  try {
    return Uint32Array.from(new Uint32Array(readback.getMappedRange(0, bytes)));
  } finally {
    readback.unmap();
  }
}

function fillActivation(
  values: Float32Array,
  spec: Readonly<{ rows: number; inner: number }>,
  ordinal: number,
): void {
  for (let row = 0; row < spec.rows; row += 1) {
    for (let k = 0; k < spec.inner; k += 1) {
      values[row * spec.inner + k] = halfToNumber(activationBits(row, k, ordinal));
    }
  }
}

function fillRev7Weight(
  values: Uint16Array,
  spec: Readonly<{ inner: number; columns: number }>,
  ordinal: number,
): void {
  let physical = 0;
  for (let columnTile = 0; columnTile < spec.columns / 256; columnTile += 1) {
    for (let innerTile = 0; innerTile < spec.inner / 32; innerTile += 1) {
      for (let innerInTile = 0; innerInTile < 32; innerInTile += 1) {
        const k = innerTile * 32 + innerInTile;
        for (let columnInTile = 0; columnInTile < 256; columnInTile += 1) {
          values[physical++] = weightBits(
            k,
            columnTile * 256 + columnInTile,
            ordinal,
          );
        }
      }
    }
  }
}

function fillRev8Weight(
  values: Uint16Array,
  spec: Readonly<{ inner: number; columns: number }>,
  ordinal: number,
): void {
  let physical = 0;
  for (let columnTile = 0; columnTile < spec.columns / 128; columnTile += 1) {
    for (let innerK4 = 0; innerK4 < spec.inner / 4; innerK4 += 1) {
      for (let outputInLane = 0; outputInLane < 4; outputInLane += 1) {
        for (let lane = 0; lane < 32; lane += 1) {
          const column = columnTile * 128 + lane * 4 + outputInLane;
          for (let innerInK4 = 0; innerInK4 < 4; innerInK4 += 1) {
            values[physical++] = weightBits(
              innerK4 * 4 + innerInK4,
              column,
              ordinal,
            );
          }
        }
      }
    }
  }
}

function activationBits(row: number, k: number, ordinal: number): number {
  const mixed = mix32(
    0x243f_6a88 ^ Math.imul(row + 1, 0x9e37_79b1) ^
      Math.imul(k + 1, 0x85eb_ca6b) ^ Math.imul(ordinal + 1, 0xc2b2_ae35),
  );
  const magnitudes = [0x2801, 0x2c11, 0x30a5, 0x34d3, 0x3821];
  return magnitudes[mixed % magnitudes.length]! | ((mixed >>> 31) << 15);
}

function weightBits(k: number, column: number, ordinal: number): number {
  const mixed = mix32(
    0x1319_8a2e ^ Math.imul(k + 1, 0x9e37_79b1) ^
      Math.imul(column + 1, 0x85eb_ca6b) ^
      Math.imul(ordinal + 1, 0xc2b2_ae35),
  );
  const magnitudes = [0x2401, 0x28b5, 0x2d53, 0x31e7, 0x356b];
  return magnitudes[mixed % magnitudes.length]! | ((mixed >>> 31) << 15);
}

function mix32(value: number): number {
  value ^= value >>> 16;
  value = Math.imul(value, 0x7feb_352d);
  value ^= value >>> 15;
  value = Math.imul(value, 0x846c_a68b);
  return (value ^ (value >>> 16)) >>> 0;
}

function halfToNumber(bits: number): number {
  const sign = (bits & 0x8000) === 0 ? 1 : -1;
  const exponent = (bits >>> 10) & 0x1f;
  const fraction = bits & 0x03ff;
  if (exponent === 0) return sign * 2 ** -14 * fraction / 1024;
  if (exponent === 0x1f) return fraction === 0 ? sign * Infinity : NaN;
  return sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}

function validateCanonicalRequest(): AceGenerationRequest {
  const bytes = new TextEncoder().encode(OPT_0018_CANONICAL_REQUEST_JSON);
  if (
    bytes.byteLength !== OPT_0018_CANONICAL_REQUEST_BYTES ||
    aceSha256Hex(bytes) !== OPT_0018_CANONICAL_REQUEST_SHA256
  ) throw new Error("OPT-0056 inherited request identity changed");
  const request = createOpt0018Request() as unknown as AceGenerationRequest;
  assertAceGenerationRequest(request);
  if (JSON.stringify(request) !== OPT_0018_CANONICAL_REQUEST_JSON) {
    throw new Error("OPT-0056 inherited request values changed");
  }
  return Object.freeze(request);
}

function compactRouteProfile(
  profile: NonNullable<AceOpt0056DitCheckpoint["denseRouteProfile"]>,
): Readonly<Record<string, unknown>> {
  return Object.freeze({
    schema: profile.schema,
    kernelSetId: profile.kernelSetId,
    routeCount: profile.routeCount,
    dispatchCount: profile.dispatchCount,
    approximateRouteCount: profile.approximateRouteCount,
    exactDownRouteCount: profile.exactDownRouteCount,
    routeOwnerRows: Object.freeze(profile.routes.map((route) => Object.freeze([
      route.routeKey,
      route.inner,
      route.columns,
      route.owner,
      route.kernelId,
      route.evaluationDispatchCount,
    ]))),
  });
}

function compactStageWalls(
  timings: readonly AceStageTiming[],
): readonly Readonly<Record<string, unknown>>[] {
  if (
    timings.length === 0 ||
    timings.some(({ stage, wallMs }) =>
      FORBIDDEN_POST_DIT_STAGES.has(stage) || !Number.isFinite(wallMs) || wallMs < 0
    ) ||
    !timings.some(({ stage }) => stage === "dit-denoise") ||
    !timings.some(({ stage }) => stage === "release-dit")
  ) throw new Error("OPT-0056 stage-wall boundary changed");
  return Object.freeze(timings.map((timing) => Object.freeze({ ...timing })));
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

function nextOrdinal(): number {
  lifecycleOrdinal += 1;
  return lifecycleOrdinal;
}

function postProgress(message: string): void {
  self.postMessage({ type: "progress", message });
}

function sameJson(left: unknown, right: unknown): boolean {
  return JSON.stringify(left) === JSON.stringify(right);
}

function isRecord(value: unknown): value is Readonly<Record<string, unknown>> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
