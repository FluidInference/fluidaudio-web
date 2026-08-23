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
} from "../../src/runtime/stages.js";
import {
  createAceWebGpuPipelineBackend,
  type AceOpt0067DitCheckpoint,
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
  OPT_0067_ARM_ORDER,
  OPT_0067_EVALUATION0_SHA256,
  OPT_0067_GRAPH_COMMAND_BUFFERS,
  OPT_0067_KERNEL_ID,
  OPT_0067_KERNEL_SET_ID,
  OPT_0067_RUNTIME_PROFILE,
  OPT_0067_TOTAL_COMMAND_BUFFERS,
  OPT_0067_WGSL_SHA256,
  exactOpt0067ResultIdentity,
  requireOpt0067ThermalGate,
  requireOpt0067ThermalTrace,
  summarizeOpt0067Performance,
  type Opt0067ArmId,
  type Opt0067Owner,
  type Opt0067ThermalGate,
  type Opt0067ThermalTrace,
  type Opt0067TimingSample,
} from "./opt-0067-dit-quad-query-evaluation-slice-contract.js";

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
const EVALUATION_DIAGNOSTIC_CODE =
  "ACE_DIT_OPT0067_EVALUATION0" as const;
const CONTROL_DESCRIPTOR_SHA256 =
  "aedf8c74d2bb15601d1385e9c8e9da49e58905ab156a04c99271f8de3633dd76" as const;
const QUAD_DESCRIPTOR_SHA256 =
  "d480bde986cba12068e462093169ef1a6cf3ceb45987eabb82ef8c8fe07eca47" as const;
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

interface RunArmMessage {
  readonly type: "run-arm";
  readonly armId: Opt0067ArmId;
  readonly thermalGate: Opt0067ThermalGate;
}

interface CompleteThermalMessage {
  readonly type: "complete-thermal";
  readonly armId: Opt0067ArmId;
  readonly thermalTrace: Opt0067ThermalTrace;
}

type IncomingMessage = PrepareMessage | RunArmMessage | CompleteThermalMessage;

interface ArmDefinition {
  readonly armId: "correctness" | Opt0067ArmId;
  readonly order: -1 | 0 | 1 | 2 | 3;
  readonly owner: Opt0067Owner;
  readonly timed: boolean;
  readonly captureActualLayerIdentity: boolean;
}

interface ArmResult {
  readonly definition: ArmDefinition;
  readonly result: Float32Array<ArrayBuffer>;
  readonly timing?: Opt0067TimingSample;
  readonly conditioningAuthority: Readonly<Record<string, unknown>>;
  readonly receipt: Readonly<Record<string, unknown>>;
  readonly lifecycle: Readonly<Record<string, number>>;
}

interface AcceptedTimedArm {
  readonly result: ArmResult;
  readonly gate: Opt0067ThermalGate;
  readonly trace: Opt0067ThermalTrace;
}

interface ActiveTimedRun {
  readonly definition: ArmDefinition;
  readonly authorization: PromiseWithResolvers<void>;
  readyAtEpochMilliseconds: number;
  gate?: Opt0067ThermalGate;
  completed?: ArmResult;
  cleanupCompletedAtEpochMilliseconds: number;
}

let state: "idle" | "preparing-correctness" | "preparing-arm" |
  "ready-arm" | "running-arm" | "await-arm-trace" | "settled" = "idle";
let identity: Opt0018RunIdentity | undefined;
let request: AceGenerationRequest | undefined;
let correctness: ArmResult | undefined;
let active: ActiveTimedRun | undefined;
let nextArmIndex = 0;
const accepted: AcceptedTimedArm[] = [];
const rejectedSetupAttempts: Readonly<Record<string, unknown>>[] = [];
let lifecycleOrdinal = 0;

self.addEventListener("message", (event: MessageEvent<IncomingMessage>) => {
  const message = event.data;
  if (message.type === "prepare" && state === "idle") {
    state = "preparing-correctness";
    void prepareGate(message.identity).catch(fail);
    return;
  }
  if (
    message.type === "run-arm" && state === "ready-arm" &&
    active !== undefined && message.armId === active.definition.armId
  ) {
    try {
      const gate = requireOpt0067ThermalGate(
        message.thermalGate,
        active.readyAtEpochMilliseconds,
        Date.now(),
      );
      active.gate = gate;
      state = "running-arm";
      self.postMessage({
        type: "progress",
        message: `${message.armId}: gate accepted; running evaluation 0`,
      });
      active.authorization.resolve();
    } catch (error) {
      retainRejected("thermal-gate", message.armId, error);
      self.postMessage({
        type: "gate-rejected",
        armId: message.armId,
        readyAtEpochMilliseconds: active.readyAtEpochMilliseconds,
        error: serializeOpt0018Failure(error),
      });
    }
    return;
  }
  if (
    message.type === "complete-thermal" && state === "await-arm-trace" &&
    active !== undefined && active.completed !== undefined &&
    active.gate !== undefined && message.armId === active.definition.armId
  ) {
    let trace: Opt0067ThermalTrace;
    try {
      trace = requireOpt0067ThermalTrace(
        message.thermalTrace,
        active.gate,
        active.cleanupCompletedAtEpochMilliseconds,
        Date.now(),
      );
      if (accepted.some(({ trace: prior }) =>
        prior.rawTraceSha256 === trace.rawTraceSha256
      )) throw new Error("OPT-0067 thermal trace was reused across arms");
    } catch (error) {
      retainRejected("thermal-trace", message.armId, error);
      self.postMessage({
        type: "trace-rejected",
        armId: message.armId,
        error: serializeOpt0018Failure(error),
      });
      return;
    }
    accepted.push(Object.freeze({
      result: active.completed,
      gate: active.gate,
      trace,
    }));
    active = undefined;
    nextArmIndex += 1;
    if (nextArmIndex === OPT_0067_ARM_ORDER.length) {
      state = "settled";
      try {
        self.postMessage({ type: "gate-complete", result: finalizeGate() });
      } catch (error) {
        fail(error);
      }
      return;
    }
    void prepareTimedArm().catch(fail);
  }
});

async function prepareGate(identityValue: Opt0018RunIdentity): Promise<void> {
  identity = validateOpt0018RunIdentity(identityValue);
  request = validateCanonicalRequest();
  self.postMessage({
    type: "progress",
    message: "untimed correctness: quad/query8 oracle across 12 eval0 routes",
  });
  correctness = await executeArm({
    armId: "correctness",
    order: -1,
    owner: "quad",
    timed: false,
    captureActualLayerIdentity: true,
  });
  if (
    correctness.receipt["resultSha256"] !== OPT_0067_EVALUATION0_SHA256 ||
    correctness.receipt["actualLayerComparedU32"] !== 55_296_000
  ) throw new Error("OPT-0067 correctness preparation did not pass");
  await prepareTimedArm();
}

async function prepareTimedArm(): Promise<void> {
  const expected = OPT_0067_ARM_ORDER[nextArmIndex];
  if (expected === undefined || request === undefined || active !== undefined) {
    throw new Error("OPT-0067 timed arm preparation order changed");
  }
  state = "preparing-arm";
  const definition: ArmDefinition = Object.freeze({
    armId: expected.armId,
    order: expected.order,
    owner: expected.owner,
    timed: true,
    captureActualLayerIdentity: false,
  });
  const authorization = Promise.withResolvers<void>();
  const run: ActiveTimedRun = {
    definition,
    authorization,
    readyAtEpochMilliseconds: 0,
    cleanupCompletedAtEpochMilliseconds: 0,
  };
  active = run;
  self.postMessage({
    type: "progress",
    message: `${definition.armId}: authenticating package, inputs, and compiled graph`,
  });
  void executeArm(definition, async () => {
    if (active !== run || state !== "preparing-arm") {
      throw new Error("OPT-0067 timing-ready ownership changed");
    }
    run.readyAtEpochMilliseconds = Date.now();
    state = "ready-arm";
    self.postMessage({
      type: "ready-for-arm",
      armId: definition.armId,
      owner: definition.owner,
      order: definition.order,
      readyAtEpochMilliseconds: run.readyAtEpochMilliseconds,
      ...(definition.order === 0
        ? { correctness: correctness!.receipt }
        : {}),
    });
    await authorization.promise;
  }).then(
    (result) => {
      if (active !== run || state !== "running-arm" || run.gate === undefined) {
        throw new Error("OPT-0067 arm completed outside its accepted gate");
      }
      run.completed = result;
      run.cleanupCompletedAtEpochMilliseconds = Date.now();
      state = "await-arm-trace";
      self.postMessage({
        type: "arm-complete",
        armId: definition.armId,
        cleanupCompletedAtEpochMilliseconds:
          run.cleanupCompletedAtEpochMilliseconds,
        sample: result.timing,
        receipt: result.receipt,
      });
    },
    fail,
  ).catch(fail);
}

async function executeArm(
  definition: ArmDefinition,
  waitForTimingAuthorization?: () => Promise<void>,
): Promise<ArmResult> {
  if (request === undefined) throw new Error("OPT-0067 request is absent");
  const backendCreatedOrdinal = nextOrdinal();
  const backend = createAceWebGpuPipelineBackend();
  const controller = new AbortController();
  const privateStop = Object.freeze({ armId: definition.armId });
  const initializeStartedAtEpochMilliseconds = Date.now();
  const initializeStartedAt = performance.now();
  let initializeWallMs = 0;
  let generatePreTimingReadyWallMs = 0;
  let timingAuthorizationWaitWallMs = 0;
  let checkpoint: AceOpt0067DitCheckpoint | undefined;
  let diagnostics: AceRuntimeDiagnostics | undefined;
  let conditioning: Readonly<Record<string, unknown>> | undefined;
  let checkpointDiagnostic: Readonly<Record<string, unknown>> | undefined;
  let initializedOrdinal = 0;
  let timingReadyOrdinal = 0;
  let timingAuthorizedOrdinal = 0;
  let checkpointOrdinal = 0;
  let generationCleanupCompletedOrdinal = 0;
  let disposeStartedOrdinal = 0;
  let disposeCompletedOrdinal = 0;
  let initializationProgressEventCount = 0;
  let generationProgressEventCount = 0;
  let forbiddenProgress = false;
  const diagnosticCodes: string[] = [];
  const fatalDiagnostics: string[] = [];
  let primaryError: unknown;
  let cleanupError: unknown;
  try {
    diagnostics = await backend.initialize(configurationForOwner(definition.owner), {
      modelSource: "cache-or-network",
      signal: controller.signal,
      onProgress: (_progress: AceInitializationProgress) => {
        initializationProgressEventCount += 1;
      },
      onDiagnostic: (diagnostic: AceDiagnostic) => {
        if (diagnostic.severity === "error") fatalDiagnostics.push(diagnostic.code);
      },
    });
    initializeWallMs = performance.now() - initializeStartedAt;
    initializedOrdinal = nextOrdinal();
    validateDiagnostics(diagnostics, definition.owner);
    const generateStartedAt = performance.now();
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
          if (conditioning !== undefined) throw new Error("duplicate conditioning");
          conditioning = validateConditioning(diagnostic);
        } else if (diagnostic.code === EVALUATION_DIAGNOSTIC_CODE) {
          if (checkpointDiagnostic !== undefined) {
            throw new Error("duplicate OPT-0067 checkpoint diagnostic");
          }
          checkpointDiagnostic = diagnostic.details;
        }
      },
      opt0067DitRun: {
        ...(definition.captureActualLayerIdentity
          ? { captureActualLayerIdentity: true as const }
          : {}),
        ...(waitForTimingAuthorization === undefined
          ? {}
          : {
              async waitForTimingAuthorization(): Promise<void> {
                generatePreTimingReadyWallMs = performance.now() - generateStartedAt;
                timingReadyOrdinal = nextOrdinal();
                const waitStartedAt = performance.now();
                await waitForTimingAuthorization();
                timingAuthorizationWaitWallMs = performance.now() - waitStartedAt;
                timingAuthorizedOrdinal = nextOrdinal();
              },
            }),
        onCheckpoint(value) {
          if (checkpoint !== undefined) {
            throw new Error(`OPT-0067 ${definition.armId} checkpoint repeated`);
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
      throw new Error(`OPT-0067 ${definition.armId} continued beyond checkpoint`);
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
        `OPT-0067 ${definition.armId} run/dispose failed`,
      );
    }
    throw primaryError ?? cleanupError;
  }
  if (
    diagnostics === undefined || checkpoint === undefined ||
    conditioning === undefined || checkpointDiagnostic === undefined ||
    initializedOrdinal === 0 || checkpointOrdinal === 0 ||
    generationCleanupCompletedOrdinal === 0 || disposeStartedOrdinal === 0 ||
    disposeCompletedOrdinal === 0 || forbiddenProgress ||
    fatalDiagnostics.length !== 0 ||
    (definition.timed
      ? timingReadyOrdinal === 0 || timingAuthorizedOrdinal === 0 ||
        !(timingAuthorizedOrdinal < checkpointOrdinal)
      : timingReadyOrdinal !== 0 || timingAuthorizedOrdinal !== 0)
  ) throw new Error(`OPT-0067 ${definition.armId} lifecycle is incomplete`);
  validateCheckpointDiagnostic(checkpointDiagnostic, definition);
  const lifecycle = Object.freeze({
    backendCreatedOrdinal,
    initializedOrdinal,
    timingReadyOrdinal,
    timingAuthorizedOrdinal,
    checkpointOrdinal,
    generationCleanupCompletedOrdinal,
    disposeStartedOrdinal,
    disposeCompletedOrdinal,
  });
  const timing = definition.timed ? timingSample(checkpoint, definition) : undefined;
  return Object.freeze({
    definition,
    result: checkpoint.result,
    ...(timing === undefined ? {} : { timing }),
    conditioningAuthority: conditioning,
    lifecycle,
    receipt: Object.freeze({
      armId: definition.armId,
      order: definition.order,
      owner: definition.owner,
      timed: definition.timed,
      runtimeProfile: definition.owner === "quad"
        ? OPT_0067_RUNTIME_PROFILE
        : "fixed32-subgroup-query8-default",
      descriptorTableSha256: checkpoint.profile.descriptorTable.sha256,
      resultSha256: checkpoint.resultSha256,
      resultElementCount: checkpoint.resultElementCount,
      routeProfile: checkpoint.attentionRouteProfile ?? null,
      actualLayerIdentity: checkpoint.actualLayerIdentity ?? null,
      actualLayerComparedU32:
        checkpoint.actualLayerIdentity?.totalComparedElements ?? 0,
      timing: timing ?? null,
      profileEvidence: profileEvidence(checkpoint),
      untimedPreparation: Object.freeze({
        initializeStartedAtEpochMilliseconds,
        initializeWallMs,
        generatePreTimingReadyWallMs,
        timingAuthorizationWaitWallMs,
        correctnessOnlyDoubledOwner:
          definition.captureActualLayerIdentity,
        excludedFromAcceptedTiming: true,
      }),
      lifecycle,
      evidence: Object.freeze({
        initializationProgressEventCount,
        generationProgressEventCount,
        diagnosticCodes: Object.freeze(diagnosticCodes),
        graphCompiledBeforeThermalGate: definition.timed,
        oneFifoGraphOwner: true,
        laterEvaluationEncodeCount: 0,
        ditDestroyedBeforeCheckpoint: true,
        pipelineCleanupAwaitedBeforeDispose: true,
        backendAndDeviceDisposeAwaited: true,
        vaeWeightAcquireStarted: false,
      }),
    }),
  });
}

function validateCheckpoint(
  checkpoint: AceOpt0067DitCheckpoint,
  definition: ArmDefinition,
): void {
  const candidate = definition.owner === "quad";
  const identityResult = checkpoint.actualLayerIdentity;
  const expectedDescriptor = candidate
    ? QUAD_DESCRIPTOR_SHA256
    : CONTROL_DESCRIPTOR_SHA256;
  if (
    checkpoint.schema !== "ace-dit-opt0067-m2250-evaluation0-checkpoint-v1" ||
    checkpoint.evaluation !== 0 ||
    checkpoint.result.length !== 288_000 || checkpoint.resultNonFiniteCount !== 0 ||
    checkpoint.resultSha256 !== OPT_0067_EVALUATION0_SHA256 ||
    checkpoint.graphCommandBufferCount !== OPT_0067_GRAPH_COMMAND_BUFFERS ||
    checkpoint.totalCommandBufferCount !== OPT_0067_TOTAL_COMMAND_BUFFERS ||
    checkpoint.completedEvaluations !== 1 ||
    checkpoint.evaluationResultExtraCommandBufferCount !== 0 ||
    checkpoint.evaluationResultExtraQueueDrainCount !== 0 ||
    checkpoint.profile.descriptorTable.sha256 !== expectedDescriptor ||
    checkpoint.profile.timings.length !== OPT_0067_GRAPH_COMMAND_BUFFERS ||
    checkpoint.profile.graphQueueDrainCount !== OPT_0067_GRAPH_COMMAND_BUFFERS ||
    checkpoint.profile.totalQueueDrainCount !== OPT_0067_TOTAL_COMMAND_BUFFERS ||
    checkpoint.profile.precompute.commandBufferCount !== 25 ||
    checkpoint.profile.evaluation.commandBufferCount !== 316 ||
    checkpoint.profile.familyByBucket["self-full"][1].commandBufferCount !== 48 ||
    (candidate !== (checkpoint.attentionRouteProfile !== undefined)) ||
    (definition.captureActualLayerIdentity !== (identityResult !== undefined))
  ) throw new Error(`OPT-0067 ${definition.armId} checkpoint changed`);
  if (candidate) {
    const routes = checkpoint.attentionRouteProfile!;
    if (
      routes.runtimeProfileId !== OPT_0067_RUNTIME_PROFILE ||
      routes.kernelSetId !== OPT_0067_KERNEL_SET_ID ||
      routes.quadQueryKernelId !== OPT_0067_KERNEL_ID ||
      routes.authenticatedWgslSha256 !== OPT_0067_WGSL_SHA256 ||
      routes.quadQueryRoutes !== 96 || routes.query8SlidingRoutes !== 96 ||
      routes.query8CrossRoutes !== 192 || routes.query8OtherRoutes !== 0 ||
      routes.unintendedQuadQueryRoutes !== 0 ||
      routes.uniqueQuadQueryRouteIds.length !== 96
    ) throw new Error(`OPT-0067 ${definition.armId} route inventory changed`);
  }
  if (
    identityResult !== undefined &&
    (identityResult.routeCount !== 12 ||
      identityResult.totalComparedElements !== 55_296_000 ||
      identityResult.totalMismatchCount !== 0 ||
      identityResult.totalNonFiniteCount !== 0 ||
      identityResult.totalCanaryCount !== 0 || identityResult.copyCount !== 1 ||
      identityResult.extraCommandBufferCount !== 0 ||
      identityResult.extraQueueDrainCount !== 0 ||
      identityResult.inactiveFutureRouteCount !== 84 ||
      identityResult.routes.length !== 12)
  ) throw new Error("OPT-0067 correctness identity changed");
}

function timingSample(
  checkpoint: AceOpt0067DitCheckpoint,
  definition: ArmDefinition,
): Opt0067TimingSample {
  if (!definition.timed || definition.order < 0) {
    throw new Error("OPT-0067 timing sample definition changed");
  }
  const familyMs = Object.freeze(Object.fromEntries(
    Object.entries(checkpoint.profile.familyByBucket).map(
      ([family, buckets]) => [family, buckets[1].submitThroughDrainMs],
    ),
  ));
  const fullSelfMs = familyMs["self-full"]!;
  const nonFullSelfEvaluationWallMs =
    checkpoint.profile.evaluationWallMs - fullSelfMs;
  const readbackMs = checkpoint.profile.readbackSubmitThroughDrainMs +
    checkpoint.profile.readbackMapDetachMs;
  return Object.freeze({
    armId: definition.armId as Opt0067ArmId,
    order: definition.order as 0 | 1 | 2 | 3,
    owner: definition.owner,
    fullSelfMs,
    evaluationWallMs: checkpoint.profile.evaluationWallMs,
    nonFullSelfEvaluationWallMs,
    graphWallMs: checkpoint.profile.graphWallMs,
    commandDrainMs: checkpoint.profile.graphSubmitThroughDrainMs,
    requestedIdleMs: checkpoint.profile.graphRequestedIdleMs,
    readbackMs,
    residualMs: checkpoint.profile.graphResidualMs,
    familyMs,
  });
}

function profileEvidence(
  checkpoint: AceOpt0067DitCheckpoint,
): Readonly<Record<string, unknown>> {
  const evaluationFamilyMs = Object.freeze(Object.fromEntries(
    Object.entries(checkpoint.profile.familyByBucket).map(
      ([family, buckets]) => [family, buckets[1].submitThroughDrainMs],
    ),
  ));
  const evaluationFamilyCommandBufferCounts = Object.freeze(Object.fromEntries(
    Object.entries(checkpoint.profile.familyByBucket).map(
      ([family, buckets]) => [family, buckets[1].commandBufferCount],
    ),
  ));
  return Object.freeze({
    schema: checkpoint.profile.schema,
    descriptorTableSha256: checkpoint.profile.descriptorTable.sha256,
    descriptorTableSerializedBytes:
      checkpoint.profile.descriptorTable.serializedBytes,
    descriptorTableMemberCount: checkpoint.profile.descriptorTable.memberCount,
    graphCommandBufferCount: checkpoint.profile.graphCommandBufferCount,
    readbackCommandBufferCount: checkpoint.profile.readbackCommandBufferCount,
    totalCommandBufferCount: checkpoint.profile.totalCommandBufferCount,
    graphQueueDrainCount: checkpoint.profile.graphQueueDrainCount,
    totalQueueDrainCount: checkpoint.profile.totalQueueDrainCount,
    precomputeCommandBufferCount:
      checkpoint.profile.precompute.commandBufferCount,
    evaluationCommandBufferCount:
      checkpoint.profile.evaluation.commandBufferCount,
    graphWallMs: checkpoint.profile.graphWallMs,
    evaluationWallMs: checkpoint.profile.evaluationWallMs,
    graphSubmitThroughDrainMs:
      checkpoint.profile.graphSubmitThroughDrainMs,
    graphRequestedIdleMs: checkpoint.profile.graphRequestedIdleMs,
    graphResidualMs: checkpoint.profile.graphResidualMs,
    evaluationRequestedIdleMs:
      checkpoint.profile.evaluationRequestedIdleMs,
    evaluationResidualMs: checkpoint.profile.evaluationResidualMs,
    graphToReadbackRequestedIdleMs:
      checkpoint.profile.graphToReadbackRequestedIdleMs,
    graphToReadbackObservedIdleMs:
      checkpoint.profile.graphToReadbackObservedIdleMs,
    readbackSubmitThroughDrainMs:
      checkpoint.profile.readbackSubmitThroughDrainMs,
    readbackMapDetachMs: checkpoint.profile.readbackMapDetachMs,
    backendWallMs: checkpoint.profile.backendWallMs,
    backendResidualMs: checkpoint.profile.backendResidualMs,
    evaluationFamilyMs,
    evaluationFamilyCommandBufferCounts,
    timingCount: checkpoint.profile.timings.length,
    timingStorageBytes: checkpoint.profile.timingStorageBytes,
  });
}

function configurationForOwner(owner: Opt0067Owner): AceWorkerConfiguration {
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
    ...(owner === "quad"
      ? { ditAttentionRuntimeProfile: OPT_0067_RUNTIME_PROFILE }
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
  owner: Opt0067Owner,
): void {
  const candidate = owner === "quad";
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
      ? diagnostics.ditAttentionRuntimeProfile !== OPT_0067_RUNTIME_PROFILE ||
        diagnostics.ditAttentionKernelSetId !== OPT_0067_KERNEL_SET_ID
      : diagnostics.ditAttentionRuntimeProfile !== undefined ||
        diagnostics.ditAttentionKernelSetId !== undefined)
  ) throw new Error(`OPT-0067 ${owner} authenticated runtime changed`);
}

function validateConditioning(
  diagnostic: AceDiagnostic,
): Readonly<Record<string, unknown>> {
  const details = diagnostic.details!;
  if (
    diagnostic.stage !== "semantic-planner" || details.plannerEnabled !== false ||
    details.instrumental !== true ||
    details.generationProfile !== "ace-turbo-v1-correctness" ||
    details.textTokenCount !== 82 || details.lyricTokenCount !== 15 ||
    typeof details.textTokenSha256 !== "string" ||
    typeof details.lyricTokenSha256 !== "string"
  ) throw new Error("OPT-0067 conditioning authority changed");
  return Object.freeze({
    textTokenCount: 82,
    textTokenSha256: details.textTokenSha256,
    lyricTokenCount: 15,
    lyricTokenSha256: details.lyricTokenSha256,
    conditionTokenCount: 98,
  });
}

function validateCheckpointDiagnostic(
  details: Readonly<Record<string, unknown>>,
  definition: ArmDefinition,
): void {
  if (
    details.checkpointSchema !==
      "ace-dit-opt0067-m2250-evaluation0-checkpoint-v1" ||
    details.evaluation !== 0 || details.resultElementCount !== 288_000 ||
    details.resultSha256 !== OPT_0067_EVALUATION0_SHA256 ||
    details.graphCommandBufferCount !== 341 ||
    details.totalCommandBufferCount !== 342 ||
    details.completedEvaluations !== 1 ||
    details.quadQueryRoutesExecuted !== (definition.owner === "quad" ? 12 : 0) ||
    details.query8SlidingRoutesExecuted !== 12 ||
    details.query8CrossRoutesExecuted !== 24 ||
    details.actualLayerComparedElements !==
      (definition.captureActualLayerIdentity ? 55_296_000 : 0) ||
    details.actualLayerMismatchCount !== 0 ||
    details.identityExtraCommandBufferCount !== 0 ||
    details.identityExtraQueueDrainCount !== 0 ||
    details.evaluationResultExtraCommandBufferCount !== 0 ||
    details.evaluationResultExtraQueueDrainCount !== 0
  ) throw new Error(`OPT-0067 ${definition.armId} diagnostic changed`);
}

function finalizeGate(): Readonly<Record<string, unknown>> {
  if (
    identity === undefined || request === undefined || correctness === undefined ||
    accepted.length !== OPT_0067_ARM_ORDER.length
  ) throw new Error("OPT-0067 final inventory is incomplete");
  const arms = accepted.map(({ result }) => result);
  if (
    arms.some((arm, index) => {
      const expected = OPT_0067_ARM_ORDER[index]!;
      return arm.definition.armId !== expected.armId ||
        arm.definition.owner !== expected.owner ||
        arm.timing === undefined ||
        arm.receipt["resultSha256"] !== OPT_0067_EVALUATION0_SHA256 ||
        !sameJson(arm.conditioningAuthority, correctness!.conditioningAuthority) ||
        !exactOpt0067ResultIdentity(arm.result, correctness!.result);
    })
  ) throw new Error("OPT-0067 timed result or conditioning identity diverged");
  for (let index = 1; index < arms.length; index += 1) {
    if (
      arms[index - 1]!.lifecycle["disposeCompletedOrdinal"]! >=
        arms[index]!.lifecycle["backendCreatedOrdinal"]!
    ) throw new Error("OPT-0067 timed package owners overlapped");
  }
  if (
    correctness.lifecycle["disposeCompletedOrdinal"]! >=
      arms[0]!.lifecycle["backendCreatedOrdinal"]!
  ) throw new Error("OPT-0067 correctness/timing package owners overlapped");
  const samples = Object.freeze(arms.map((arm) => arm.timing!));
  const performance = summarizeOpt0067Performance(samples);
  const passed = performance.passed;
  return Object.freeze({
    schema: "ace-opt-0067-quad-query-evaluation0-thermal-gate-v1",
    experimentId: "OPT-0067",
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
      candidateRuntimeProfile: OPT_0067_RUNTIME_PROFILE,
      candidateKernelSetId: OPT_0067_KERNEL_SET_ID,
      candidateKernelId: OPT_0067_KERNEL_ID,
      authenticatedWgslSha256: OPT_0067_WGSL_SHA256,
      expectedEvaluation0Sha256: OPT_0067_EVALUATION0_SHA256,
    }),
    correctness: Object.freeze({
      passed: true,
      excludedFromTiming: true,
      expectedActualLayerComparedU32: 55_296_000,
      actualLayerMismatchCount: 0,
      evaluationResultRawU32Exact: true,
      arm: correctness.receipt,
    }),
    performance,
    arms: Object.freeze(Object.fromEntries(accepted.map((entry) => [
      entry.result.definition.armId,
      Object.freeze({
        gate: entry.gate,
        thermalTrace: entry.trace,
        receipt: entry.result.receipt,
      }),
    ]))),
    rejectedSetupAttempts: Object.freeze([...rejectedSetupAttempts]),
    lifecycle: Object.freeze({
      oneFifoGraphOwnerPerArm: true,
      sequentialNonOverlappingPackageOwnership: true,
      noCrossArmBackendOrDeviceReuse: true,
      preparationExcludedFromTiming: true,
      graphCompiledBeforeEveryThermalGate: true,
      distinctThermalTracePerArm: true,
      drainBeforeRelease: true,
      allBackendsAndDevicesDisposed: true,
      executedEvaluationCountPerArm: 1,
      laterEvaluationEncodeCount: 0,
      ordinarySubmitCountPerArm: OPT_0067_TOTAL_COMMAND_BUFFERS,
      ordinaryDrainCountPerArm: OPT_0067_TOTAL_COMMAND_BUFFERS,
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
      plannerExecuted: false,
      laterEvaluationsExecuted: false,
      fullGraphClaim: false,
      listeningClaim: false,
      underOneMinuteClaim: false,
    }),
    decision: passed
      ? "evaluation-slice-pass-await-separate-stack-authorization"
      : "evaluation-slice-non-pass-keep-query8-default",
  });
}

function validateCanonicalRequest(): AceGenerationRequest {
  const bytes = new TextEncoder().encode(OPT_0018_CANONICAL_REQUEST_JSON);
  if (
    bytes.byteLength !== OPT_0018_CANONICAL_REQUEST_BYTES ||
    aceSha256Hex(bytes) !== OPT_0018_CANONICAL_REQUEST_SHA256
  ) throw new Error("OPT-0067 inherited request identity changed");
  const value = createOpt0018Request() as unknown as AceGenerationRequest;
  assertAceGenerationRequest(value);
  if (JSON.stringify(value) !== OPT_0018_CANONICAL_REQUEST_JSON) {
    throw new Error("OPT-0067 inherited request values changed");
  }
  return Object.freeze(value);
}

function retainRejected(
  kind: "thermal-gate" | "thermal-trace",
  armId: Opt0067ArmId,
  error: unknown,
): void {
  rejectedSetupAttempts.push(Object.freeze({
    kind,
    armId,
    rejectedAtEpochMilliseconds: Date.now(),
    error: serializeOpt0018Failure(error),
    timedGpuWorkStarted: kind === "thermal-trace",
  }));
}

function absoluteUrl(path: string): string {
  return new URL(path, self.location.href).href;
}

function nextOrdinal(): number {
  lifecycleOrdinal += 1;
  return lifecycleOrdinal;
}

function sameJson(left: unknown, right: unknown): boolean {
  return JSON.stringify(left) === JSON.stringify(right);
}

function fail(error: unknown): void {
  if (state === "settled") return;
  state = "settled";
  active?.authorization.reject(error);
  self.postMessage({ type: "failed", error: serializeOpt0018Failure(error) });
}
