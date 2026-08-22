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
  type AceOpt0062DitCheckpoint,
  type AceWebGpuGenerationContext,
} from "../../src/runtime/webgpu-pipeline.js";
import {
  OPT_0018_CANONICAL_REQUEST_BYTES,
  OPT_0018_CANONICAL_REQUEST_JSON,
  OPT_0018_CANONICAL_REQUEST_SHA256,
  createOpt0018Request,
  serializeOpt0018Failure,
  type Opt0018RunIdentity,
  validateOpt0018RunIdentity,
} from "./opt-0018-dit-m2250-production-family-profile.js";
import {
  OPT_0062_GRAPH_COMMAND_BUFFERS,
  OPT_0062_KERNEL_ID,
  OPT_0062_KERNEL_SET_ID,
  OPT_0062_LATENT_ELEMENTS,
  OPT_0062_RUNTIME_PROFILE,
  OPT_0062_TOTAL_COMMAND_BUFFERS,
  OPT_0062_WGSL_SHA256,
  exactOpt0062TrajectoryIdentity,
  requireOpt0062ThermalGate,
  requireOpt0062ThermalTrace,
  summarizeOpt0062BalancedGate,
  type Opt0062Arm,
  type Opt0062Direction,
  type Opt0062ThermalGate,
  type Opt0062ThermalTrace,
  type Opt0062TimingSample,
} from "./opt-0062-dit-quad-query-contract.js";

const MAIN_MANIFEST_PATH = "/model/files-reference/manifest.json" as const;
const MAIN_MANIFEST_SHA256 =
  "18f36c6420976475af65ecd833ca56c6119706322ce54120389d4915d8e80db6" as const;
const DENSE_MANIFEST_PATH =
  "/model/files-fp16-dit-rev7-oracle/manifest.json" as const;
const DENSE_MANIFEST_SHA256 =
  "d3fc0020efcf60702db411da2fd4b93e9bb84f1437ed310aef01c892727e452f" as const;
const DENSE_MANIFEST_BYTES = 254_357 as const;
const DENSE_RUNTIME_PROFILE = "opt-0009-fp16-fp32-dense-v1" as const;
const DENSE_KERNEL_SET_ID = "opt-0009-n256-k32-fp16-fp32-v1" as const;
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
const TRAJECTORY_DIAGNOSTIC_CODE =
  "ACE_DIT_OPT0062_QUAD_QUERY_TRAJECTORY" as const;
const FORBIDDEN_POST_DIT_STAGES = new Set([
  "vae-load",
  "vae-decode",
  "wav-encode",
  "cleanup",
  "done",
]);

interface PrepareMessage {
  readonly type: "prepare";
  readonly identity: Opt0018RunIdentity;
}

interface RunDirectionMessage {
  readonly type: "run-direction";
  readonly direction: Opt0062Direction;
  readonly thermalGate: Opt0062ThermalGate;
}

interface CompleteThermalMessage {
  readonly type: "complete-thermal";
  readonly direction: Opt0062Direction;
  readonly thermalTrace: Opt0062ThermalTrace;
}

type IncomingMessage = PrepareMessage | RunDirectionMessage |
  CompleteThermalMessage;

interface ArmDefinition {
  readonly arm: Opt0062Arm;
  readonly role: string;
  readonly direction: "correctness" | Opt0062Direction;
  readonly order: 0 | 1 | 2;
  readonly captureActualLayerIdentity: boolean;
  readonly retainTrajectory: boolean;
}

interface ArmResult {
  readonly role: string;
  readonly arm: Opt0062Arm;
  readonly trajectory?: readonly Float32Array<ArrayBuffer>[];
  readonly conditioningAuthority: Readonly<Record<string, unknown>>;
  readonly timing?: Opt0062TimingSample;
  readonly receipt: Readonly<Record<string, unknown>>;
  readonly lifecycle: Readonly<Record<string, number>>;
}

interface CorrectnessResult {
  readonly passed: true;
  readonly receipt: Readonly<Record<string, unknown>>;
  readonly conditioningAuthority: Readonly<Record<string, unknown>>;
}

interface DirectionResult {
  readonly direction: Opt0062Direction;
  readonly gate: Opt0062ThermalGate;
  readonly samples: readonly [Opt0062TimingSample, Opt0062TimingSample];
  readonly armReceipts: readonly Readonly<Record<string, unknown>>[];
  readonly cleanupCompletedAtEpochMilliseconds: number;
  readonly conditioningAuthority: Readonly<Record<string, unknown>>;
}

let state:
  | "idle"
  | "preparing"
  | "ready-forward"
  | "running-forward"
  | "await-forward-trace"
  | "ready-reverse"
  | "running-reverse"
  | "await-reverse-trace"
  | "settled" = "idle";
let identity: Opt0018RunIdentity | undefined;
let correctness: CorrectnessResult | undefined;
let readyAtEpochMilliseconds = 0;
let pendingDirection: DirectionResult | undefined;
let forward: DirectionResult | undefined;
let forwardTrace: Opt0062ThermalTrace | undefined;
let lifecycleOrdinal = 0;

self.addEventListener("message", (event: MessageEvent<IncomingMessage>) => {
  const message = event.data;
  if (message.type === "prepare" && state === "idle") {
    state = "preparing";
    void prepareGate(message.identity).then(
      (value) => {
        identity = validateOpt0018RunIdentity(message.identity);
        correctness = value;
        readyAtEpochMilliseconds = Date.now();
        state = "ready-forward";
        self.postMessage({
          type: "ready-for-direction",
          direction: "forward",
          readyAtEpochMilliseconds,
          correctness: value.receipt,
        });
      },
      fail,
    );
    return;
  }
  if (
    message.type === "run-direction" &&
    ((state === "ready-forward" && message.direction === "forward") ||
      (state === "ready-reverse" && message.direction === "reverse"))
  ) {
    state = message.direction === "forward"
      ? "running-forward"
      : "running-reverse";
    let gate: Opt0062ThermalGate;
    try {
      gate = requireOpt0062ThermalGate(
        message.thermalGate,
        readyAtEpochMilliseconds,
        Date.now(),
      );
    } catch (error) {
      fail(error);
      return;
    }
    void runDirection(message.direction, gate).then(
      (result) => {
        pendingDirection = result;
        state = message.direction === "forward"
          ? "await-forward-trace"
          : "await-reverse-trace";
        self.postMessage({
          type: "direction-complete",
          direction: message.direction,
          cleanupCompletedAtEpochMilliseconds:
            result.cleanupCompletedAtEpochMilliseconds,
          samples: result.samples,
        });
      },
      fail,
    );
    return;
  }
  if (
    message.type === "complete-thermal" && pendingDirection !== undefined &&
    ((state === "await-forward-trace" && message.direction === "forward") ||
      (state === "await-reverse-trace" && message.direction === "reverse"))
  ) {
    let trace: Opt0062ThermalTrace;
    try {
      trace = requireOpt0062ThermalTrace(
        message.thermalTrace,
        pendingDirection.gate,
        pendingDirection.cleanupCompletedAtEpochMilliseconds,
        Date.now(),
      );
    } catch (error) {
      fail(error);
      return;
    }
    if (message.direction === "forward") {
      forward = pendingDirection;
      forwardTrace = trace;
      pendingDirection = undefined;
      readyAtEpochMilliseconds = Date.now();
      state = "ready-reverse";
      self.postMessage({
        type: "ready-for-direction",
        direction: "reverse",
        readyAtEpochMilliseconds,
      });
      return;
    }
    const reverse = pendingDirection;
    pendingDirection = undefined;
    state = "settled";
    try {
      self.postMessage({
        type: "gate-complete",
        result: finalizeGate(reverse, trace),
      });
    } catch (error) {
      fail(error);
    }
  }
});

async function prepareGate(
  identityValue: Opt0018RunIdentity,
): Promise<CorrectnessResult> {
  validateOpt0018RunIdentity(identityValue);
  const request = validateCanonicalRequest();
  postProgress("correctness 1/3: query8 control trajectory");
  const control = await runArm({
    arm: "query8",
    role: "correctness-query8",
    direction: "correctness",
    order: 0,
    captureActualLayerIdentity: false,
    retainTrajectory: true,
  }, request);
  postProgress("correctness 2/3: quad trajectory + all 96 actual-layer oracles");
  const quad = await runArm({
    arm: "quad",
    role: "correctness-quad",
    direction: "correctness",
    order: 1,
    captureActualLayerIdentity: true,
    retainTrajectory: true,
  }, request);
  postProgress("correctness 3/3: deterministic quad repeat");
  const repeat = await runArm({
    arm: "quad",
    role: "correctness-quad-repeat",
    direction: "correctness",
    order: 2,
    captureActualLayerIdentity: true,
    retainTrajectory: true,
  }, request);
  for (const [previous, next] of [
    [control, quad],
    [quad, repeat],
  ] as const) {
    if (
      previous.lifecycle["disposeCompletedOrdinal"]! >=
        next.lifecycle["backendCreatedOrdinal"]!
    ) throw new Error("OPT-0062 correctness arms overlapped");
  }
  if (
    !sameJson(control.conditioningAuthority, quad.conditioningAuthority) ||
    !sameJson(control.conditioningAuthority, repeat.conditioningAuthority) ||
    !exactOpt0062TrajectoryIdentity(control.trajectory!, quad.trajectory!) ||
    !exactOpt0062TrajectoryIdentity(quad.trajectory!, repeat.trajectory!)
  ) throw new Error("OPT-0062 correctness trajectory or conditioning diverged");
  return Object.freeze({
    passed: true,
    conditioningAuthority: control.conditioningAuthority,
    receipt: Object.freeze({
      schema: "ace-opt-0062-correctness-gate-v1",
      actualLayerComparedU32: 442_368_000,
      actualLayerMismatchCount: 0,
      allEightEvaluationTapsRawU32Exact: true,
      finalLatentRawU32Exact: true,
      quadDeterministicRepeatExact: true,
      sequentialNonOverlappingArms: true,
      arms: Object.freeze({
        control: control.receipt,
        quad: quad.receipt,
        quadRepeat: repeat.receipt,
      }),
    }),
  });
}

async function runDirection(
  direction: Opt0062Direction,
  gate: Opt0062ThermalGate,
): Promise<DirectionResult> {
  const request = validateCanonicalRequest();
  const arms: readonly Opt0062Arm[] = direction === "forward"
    ? ["query8", "quad"]
    : ["quad", "query8"];
  const results: ArmResult[] = [];
  for (let order = 0; order < arms.length; order += 1) {
    const arm = arms[order]!;
    postProgress(`${direction} ${order + 1}/2: ${arm} full M2250 graph`);
    const result = await runArm({
      arm,
      role: `${direction}-${arm}`,
      direction,
      order: order as 0 | 1,
      captureActualLayerIdentity: false,
      retainTrajectory: false,
    }, request);
    const previous = results.at(-1);
    if (
      previous !== undefined &&
      previous.lifecycle["disposeCompletedOrdinal"]! >=
        result.lifecycle["backendCreatedOrdinal"]!
    ) throw new Error(`OPT-0062 ${direction} arms overlapped`);
    results.push(result);
  }
  const first = results[0]!;
  const second = results[1]!;
  if (
    first.timing === undefined || second.timing === undefined ||
    !sameJson(first.conditioningAuthority, second.conditioningAuthority) ||
    !sameJson(first.conditioningAuthority, correctness!.conditioningAuthority)
  ) throw new Error(`OPT-0062 ${direction} timing inventory changed`);
  return Object.freeze({
    direction,
    gate,
    samples: Object.freeze([
      first.timing,
      second.timing,
    ]) as DirectionResult["samples"],
    armReceipts: Object.freeze([first.receipt, second.receipt]),
    cleanupCompletedAtEpochMilliseconds: Date.now(),
    conditioningAuthority: first.conditioningAuthority,
  });
}

async function runArm(
  definition: ArmDefinition,
  request: AceGenerationRequest,
): Promise<ArmResult> {
  const backendCreatedOrdinal = nextOrdinal();
  const backend = createAceWebGpuPipelineBackend();
  const controller = new AbortController();
  const privateStop = Object.freeze({ role: definition.role });
  let checkpoint: AceOpt0062DitCheckpoint | undefined;
  let diagnostics: AceRuntimeDiagnostics | undefined;
  let conditioning: Readonly<Record<string, unknown>> | undefined;
  let trajectoryDiagnostic: Readonly<Record<string, unknown>> | undefined;
  let initializedOrdinal = 0;
  let checkpointOrdinal = 0;
  let generationCleanupCompletedOrdinal = 0;
  let disposeStartedOrdinal = 0;
  let disposeCompletedOrdinal = 0;
  let forbiddenProgress = false;
  let initializationProgressEventCount = 0;
  let generationProgressEventCount = 0;
  const diagnosticCodes: string[] = [];
  const fatalDiagnostics: string[] = [];
  let primaryError: unknown;
  let cleanupError: unknown;
  try {
    diagnostics = await backend.initialize(configurationForArm(definition.arm), {
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
    initializedOrdinal = nextOrdinal();
    validateDiagnostics(diagnostics, definition.arm);
    const context: AceWebGpuGenerationContext = {
      signal: controller.signal,
      captureTrace: true,
      onProgress: (progress: AceGenerationProgress) => {
        generationProgressEventCount += 1;
        if (FORBIDDEN_POST_DIT_STAGES.has(progress.stage)) {
          forbiddenProgress = true;
        }
      },
      onDiagnostic: (diagnostic: AceDiagnostic) => {
        diagnosticCodes.push(diagnostic.code);
        if (diagnostic.severity === "error") {
          fatalDiagnostics.push(diagnostic.code);
        }
        if (diagnostic.details === undefined) return;
        if (diagnostic.code === CONDITIONING_DIAGNOSTIC_CODE) {
          if (conditioning !== undefined) throw new Error("duplicate conditioning");
          conditioning = validateConditioning(diagnostic);
        } else if (diagnostic.code === TRAJECTORY_DIAGNOSTIC_CODE) {
          if (trajectoryDiagnostic !== undefined) {
            throw new Error("duplicate OPT-0062 trajectory diagnostic");
          }
          trajectoryDiagnostic = diagnostic.details;
        }
      },
      opt0062DitRun: {
        ...(definition.captureActualLayerIdentity
          ? { captureActualLayerIdentity: true as const }
          : {}),
        onCheckpoint(value) {
          if (checkpoint !== undefined) {
            throw new Error(`OPT-0062 ${definition.role} checkpoint repeated`);
          }
          validateCheckpoint(value, definition);
          checkpoint = value;
          checkpointOrdinal = nextOrdinal();
          controller.abort(privateStop);
        },
      },
    };
    try {
      await backend.generate(request, context);
      throw new Error(`OPT-0062 ${definition.role} continued into VAE`);
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
      throw new AggregateError(
        [primaryError, cleanupError],
        `${definition.role} run/dispose failed`,
      );
    }
    throw primaryError ?? cleanupError;
  }
  if (
    diagnostics === undefined || checkpoint === undefined ||
    conditioning === undefined || trajectoryDiagnostic === undefined ||
    initializedOrdinal === 0 || checkpointOrdinal === 0 ||
    generationCleanupCompletedOrdinal === 0 || forbiddenProgress ||
    fatalDiagnostics.length !== 0
  ) throw new Error(`OPT-0062 ${definition.role} lifecycle is incomplete`);
  validateTrajectoryDiagnostic(trajectoryDiagnostic, definition);
  const lifecycle = Object.freeze({
    backendCreatedOrdinal,
    initializedOrdinal,
    checkpointOrdinal,
    generationCleanupCompletedOrdinal,
    disposeStartedOrdinal,
    disposeCompletedOrdinal,
  });
  const timing = definition.direction === "correctness"
    ? undefined
    : timingSample(checkpoint, definition);
  return Object.freeze({
    role: definition.role,
    arm: definition.arm,
    ...(definition.retainTrajectory
      ? {
          trajectory: Object.freeze(checkpoint.evaluations.map(
            ({ latent }) => latent,
          )),
        }
      : {}),
    conditioningAuthority: conditioning,
    ...(timing === undefined ? {} : { timing }),
    lifecycle,
    receipt: Object.freeze({
      role: definition.role,
      arm: definition.arm,
      runtimeProfile: definition.arm === "quad"
        ? OPT_0062_RUNTIME_PROFILE
        : "fixed32-subgroup-query8-default",
      descriptorTableSha256: checkpoint.profile.descriptorTable.sha256,
      evaluationHashes: Object.freeze(checkpoint.evaluations.map(
        ({ evaluation, latentSha256 }) => Object.freeze({
          evaluation,
          latentSha256,
        }),
      )),
      finalLatentSha256: checkpoint.finalLatentSha256,
      routeProfile: checkpoint.attentionRouteProfile ?? null,
      actualLayerIdentity: checkpoint.actualLayerIdentity ?? null,
      timing: timing ?? null,
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
  checkpoint: AceOpt0062DitCheckpoint,
  definition: ArmDefinition,
): void {
  const candidate = definition.arm === "quad";
  const identity = checkpoint.actualLayerIdentity;
  if (
    checkpoint.schema !== "ace-dit-opt0062-m2250-trajectory-checkpoint-v1" ||
    checkpoint.finalLatent.length !== OPT_0062_LATENT_ELEMENTS ||
    checkpoint.finalLatentNonFiniteCount !== 0 ||
    checkpoint.evaluations.length !== 8 ||
    checkpoint.graphCommandBufferCount !== OPT_0062_GRAPH_COMMAND_BUFFERS ||
    checkpoint.totalCommandBufferCount !== OPT_0062_TOTAL_COMMAND_BUFFERS ||
    checkpoint.snapshotCopyCount !== 8 ||
    checkpoint.snapshotExtraCommandBufferCount !== 0 ||
    checkpoint.snapshotExtraQueueDrainCount !== 0 ||
    checkpoint.profile.timings.length !== OPT_0062_GRAPH_COMMAND_BUFFERS ||
    (candidate !== (checkpoint.attentionRouteProfile !== undefined)) ||
    (definition.captureActualLayerIdentity !== (identity !== undefined))
  ) throw new Error(`OPT-0062 ${definition.role} checkpoint topology changed`);
  for (const snapshot of checkpoint.evaluations) {
    if (
      snapshot.latent.length !== OPT_0062_LATENT_ELEMENTS ||
      snapshot.nonFiniteCount !== 0 || snapshot.nonzeroCount <= 0 ||
      aceSha256Hex(new Uint8Array(
        snapshot.latent.buffer,
        snapshot.latent.byteOffset,
        snapshot.latent.byteLength,
      )) !== snapshot.latentSha256
    ) throw new Error(`OPT-0062 ${definition.role} sampler tap invalid`);
  }
  if (checkpoint.evaluations.at(-1)!.latentSha256 !== checkpoint.finalLatentSha256) {
    throw new Error(`OPT-0062 ${definition.role} final tap/readback changed`);
  }
  if (candidate) {
    const routes = checkpoint.attentionRouteProfile!;
    if (
      routes.runtimeProfileId !== OPT_0062_RUNTIME_PROFILE ||
      routes.kernelSetId !== OPT_0062_KERNEL_SET_ID ||
      routes.quadQueryKernelId !== OPT_0062_KERNEL_ID ||
      routes.authenticatedWgslSha256 !== OPT_0062_WGSL_SHA256 ||
      routes.quadQueryRoutes !== 96 || routes.query8SlidingRoutes !== 96 ||
      routes.query8CrossRoutes !== 192 || routes.query8OtherRoutes !== 0 ||
      routes.unintendedQuadQueryRoutes !== 0 ||
      routes.uniqueQuadQueryRouteIds.length !== 96 ||
      new Set(routes.uniqueQuadQueryRouteIds).size !== 96
    ) throw new Error(`OPT-0062 ${definition.role} route inventory changed`);
  }
  if (
    identity !== undefined &&
    (identity.routeCount !== 96 ||
      identity.totalComparedElements !== 442_368_000 ||
      identity.totalMismatchCount !== 0 || identity.totalNonFiniteCount !== 0 ||
      identity.totalCanaryCount !== 0 || identity.copyCount !== 1 ||
      identity.extraCommandBufferCount !== 0 ||
      identity.extraQueueDrainCount !== 0 || identity.routes.length !== 96)
  ) throw new Error(`OPT-0062 ${definition.role} actual-layer identity failed`);
  const members = checkpoint.profile.descriptorTable.descriptors.flatMap(
    ({ members }) => members,
  );
  const quadDescriptors = members.filter(({ backend }) =>
    backend === "opt-0062-fixed32-quad-query32-full-self"
  ).length;
  const controlDescriptors = members.filter(({ family, backend }) =>
    family === "self-full" && backend === "fixed32-subgroup-query8"
  ).length;
  if (
    candidate
      ? quadDescriptors !== 480 || controlDescriptors !== 0
      : quadDescriptors !== 0 || controlDescriptors !== 480
  ) throw new Error(`OPT-0062 ${definition.role} command attribution changed`);
}

function timingSample(
  checkpoint: AceOpt0062DitCheckpoint,
  definition: ArmDefinition,
): Opt0062TimingSample {
  if (definition.direction === "correctness" || definition.order > 1) {
    throw new Error("OPT-0062 timing sample definition changed");
  }
  const order = definition.order as 0 | 1;
  const familyMs = Object.freeze(Object.fromEntries(
    Object.entries(checkpoint.profile.families).map(([family, aggregate]) =>
      [family, aggregate.submitThroughDrainMs]
    ),
  ));
  const slice = checkpoint.profile.familyByBucket["self-full"]?.[1];
  if (slice === undefined || slice.commandBufferCount === 0) {
    throw new Error("OPT-0062 evaluation-zero full-self slice is absent");
  }
  return Object.freeze({
    direction: definition.direction,
    order,
    arm: definition.arm,
    fullSelfMs: checkpoint.profile.families["self-full"].submitThroughDrainMs,
    sliceEvaluation0FullSelfMs: slice.submitThroughDrainMs,
    graphWallMs: checkpoint.profile.graphWallMs,
    ditStageWallMs: sumDitStageWalls(checkpoint.stageTimings),
    commandDrainMs: checkpoint.profile.graphSubmitThroughDrainMs,
    requestedIdleMs: checkpoint.profile.graphRequestedIdleMs,
    readbackMs: checkpoint.profile.readbackSubmitThroughDrainMs +
      checkpoint.profile.readbackMapDetachMs,
    residualMs: checkpoint.profile.graphResidualMs,
    familyMs,
  });
}

function configurationForArm(arm: Opt0062Arm): AceWorkerConfiguration {
  return Object.freeze({
    manifestUrl: absoluteUrl(MAIN_MANIFEST_PATH),
    manifestSha256: MAIN_MANIFEST_SHA256,
    modelProfile: "reference-bf16",
    schedulingProfile: "cooperative",
    ditDensePackage: Object.freeze({
      manifestUrl: absoluteUrl(DENSE_MANIFEST_PATH),
      manifestSha256: DENSE_MANIFEST_SHA256,
      runtimeProfile: DENSE_RUNTIME_PROFILE,
    }),
    ...(arm === "quad"
      ? { ditAttentionRuntimeProfile: OPT_0062_RUNTIME_PROFILE }
      : {}),
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
  arm: Opt0062Arm,
): void {
  const candidate = arm === "quad";
  if (
    diagnostics.modelManifestSha256 !== MAIN_MANIFEST_SHA256 ||
    diagnostics.ditDenseManifestSha256 !== DENSE_MANIFEST_SHA256 ||
    diagnostics.ditDenseManifestByteLength !== DENSE_MANIFEST_BYTES ||
    diagnostics.ditDenseRuntimeProfile !== DENSE_RUNTIME_PROFILE ||
    diagnostics.ditDenseKernelSetId !== DENSE_KERNEL_SET_ID ||
    diagnostics.vaeManifestSha256 !== VAE_MANIFEST_SHA256 ||
    diagnostics.vaeRuntimeProfile !== VAE_RUNTIME_PROFILE ||
    diagnostics.vaeKernelSetId !== VAE_KERNEL_SET_ID ||
    diagnostics.executionProfile.id !== "reference-bf16-subgroups" ||
    diagnostics.schedulingProfile !== "cooperative" ||
    diagnostics.capabilities.adapterInfo.subgroupMinSize !== 32 ||
    diagnostics.capabilities.adapterInfo.subgroupMaxSize !== 32 ||
    !diagnostics.capabilities.deviceFeatures.includes("subgroups") ||
    (candidate
      ? diagnostics.ditAttentionRuntimeProfile !== OPT_0062_RUNTIME_PROFILE ||
        diagnostics.ditAttentionKernelSetId !== OPT_0062_KERNEL_SET_ID
      : diagnostics.ditAttentionRuntimeProfile !== undefined ||
        diagnostics.ditAttentionKernelSetId !== undefined)
  ) throw new Error(`OPT-0062 ${arm} authenticated runtime changed`);
}

function validateConditioning(
  diagnostic: AceDiagnostic,
): Readonly<Record<string, unknown>> {
  const details = diagnostic.details!;
  if (
    diagnostic.stage !== "semantic-planner" ||
    details.plannerEnabled !== false || details.instrumental !== true ||
    details.generationProfile !== "ace-turbo-v1-correctness" ||
    details.textTokenCount !== 82 || details.lyricTokenCount !== 15 ||
    typeof details.textTokenSha256 !== "string" ||
    typeof details.lyricTokenSha256 !== "string"
  ) throw new Error("OPT-0062 conditioning authority changed");
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
  definition: ArmDefinition,
): void {
  if (
    details.checkpointSchema !==
      "ace-dit-opt0062-m2250-trajectory-checkpoint-v1" ||
    details.evaluationCount !== 8 ||
    details.graphCommandBufferCount !== OPT_0062_GRAPH_COMMAND_BUFFERS ||
    details.totalCommandBufferCount !== OPT_0062_TOTAL_COMMAND_BUFFERS ||
    details.snapshotCopyCount !== 8 ||
    details.snapshotExtraCommandBufferCount !== 0 ||
    details.snapshotExtraQueueDrainCount !== 0 ||
    details.quadQueryRoutes !== (definition.arm === "quad" ? 96 : 0) ||
    details.actualLayerComparedElements !==
      (definition.captureActualLayerIdentity ? 442_368_000 : 0) ||
    details.actualLayerMismatchCount !== 0
  ) throw new Error(`OPT-0062 ${definition.role} diagnostic changed`);
}

function finalizeGate(
  reverse: DirectionResult,
  reverseTrace: Opt0062ThermalTrace,
): Readonly<Record<string, unknown>> {
  if (
    identity === undefined || correctness === undefined ||
    forward === undefined || forwardTrace === undefined ||
    reverse.direction !== "reverse"
  ) throw new Error("OPT-0062 final gate inventory is incomplete");
  const samples = Object.freeze([...forward.samples, ...reverse.samples]);
  const performance = summarizeOpt0062BalancedGate(samples);
  const passed = correctness.passed && performance.passed;
  return Object.freeze({
    schema: "ace-opt-0062-quad-query-full-graph-gate-v1",
    experimentId: "OPT-0062",
    status: passed ? "passed" : "failed",
    passed,
    identity: Object.freeze({
      run: identity,
      requestCanonicalJson: OPT_0018_CANONICAL_REQUEST_JSON,
      requestSha256: OPT_0018_CANONICAL_REQUEST_SHA256,
      requestByteLength: OPT_0018_CANONICAL_REQUEST_BYTES,
      mainManifestSha256: MAIN_MANIFEST_SHA256,
      denseManifestSha256: DENSE_MANIFEST_SHA256,
      denseRuntimeProfile: DENSE_RUNTIME_PROFILE,
      vaeManifestSha256: VAE_MANIFEST_SHA256,
      vaeRuntimeProfile: VAE_RUNTIME_PROFILE,
      candidateRuntimeProfile: OPT_0062_RUNTIME_PROFILE,
      candidateKernelSetId: OPT_0062_KERNEL_SET_ID,
      candidateKernelId: OPT_0062_KERNEL_ID,
      authenticatedWgslSha256: OPT_0062_WGSL_SHA256,
    }),
    correctness: correctness.receipt,
    performance,
    directions: Object.freeze({
      forward: Object.freeze({
        gate: forward.gate,
        thermalTrace: forwardTrace,
        arms: forward.armReceipts,
      }),
      reverse: Object.freeze({
        gate: reverse.gate,
        thermalTrace: reverseTrace,
        arms: reverse.armReceipts,
      }),
    }),
    lifecycle: Object.freeze({
      oneFifoGraphOwnerPerArm: true,
      sequentialNonOverlappingArms: true,
      drainBeforeRelease: true,
      allBackendsAndDevicesDisposed: true,
      postCheckpointVaeAcquireCount: 0,
      identityExtraSubmitCount: 0,
      identityExtraDrainCount: 0,
    }),
    scope: Object.freeze({
      productionDefaultChanged: false,
      denseProfileChanged: false,
      packageBytesChanged: false,
      schedulerChanged: false,
      vaeWeightsAcquired: false,
      vaeExecuted: false,
      audioExecuted: false,
      sixEvaluationCombined: false,
      selectiveDenseCombined: false,
      listeningClaim: false,
      underOneMinuteClaim: false,
    }),
    decision: passed
      ? "integration-gate-pass-await-owner-review"
      : "inconclusive-or-negative-keep-query8-default",
  });
}

function validateCanonicalRequest(): AceGenerationRequest {
  const bytes = new TextEncoder().encode(OPT_0018_CANONICAL_REQUEST_JSON);
  if (
    bytes.byteLength !== OPT_0018_CANONICAL_REQUEST_BYTES ||
    aceSha256Hex(bytes) !== OPT_0018_CANONICAL_REQUEST_SHA256
  ) throw new Error("OPT-0062 inherited request identity changed");
  const request = createOpt0018Request() as unknown as AceGenerationRequest;
  assertAceGenerationRequest(request);
  if (JSON.stringify(request) !== OPT_0018_CANONICAL_REQUEST_JSON) {
    throw new Error("OPT-0062 inherited request values changed");
  }
  return Object.freeze(request);
}

function sumDitStageWalls(timings: readonly AceStageTiming[]): number {
  const stages = new Set(["dit-load", "dit-denoise", "release-dit"]);
  const selected = timings.filter(({ stage }) => stages.has(stage));
  if (
    selected.length !== 3 ||
    selected.some(({ wallMs }) => !Number.isFinite(wallMs) || wallMs < 0) ||
    timings.some(({ stage }) => FORBIDDEN_POST_DIT_STAGES.has(stage))
  ) throw new Error("OPT-0062 complete DiT stage boundary changed");
  return selected.reduce((total, { wallMs }) => total + wallMs, 0);
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

function fail(error: unknown): void {
  state = "settled";
  self.postMessage({ type: "failed", error: serializeOpt0018Failure(error) });
}
