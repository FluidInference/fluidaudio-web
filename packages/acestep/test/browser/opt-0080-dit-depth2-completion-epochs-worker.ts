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
  type AceOpt0080DitCheckpoint,
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
  ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_KERNEL_SET_ID,
  ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
} from "../../src/webgpu/dit-attention-profile.js";
import {
  ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_KERNEL_SET_ID,
  ACE_OPT_0072_VAE_FP16_FIXED32_DUAL_K4_PRODUCTION_RUNTIME_PROFILE,
} from "../../src/webgpu/vae-fp16-profile.js";
import {
  ACE_OPT_0070_VAE_C2378_MAXIMUM_WINDOW_FRAMES,
  ACE_OPT_0070_VAE_C2378_WINDOW_RUNTIME_PROFILE,
} from "../../src/webgpu/vae-window-profile.js";
import {
  OPT_0080_ARM_ORDER,
  OPT_0080_CANDIDATE_PROFILE,
  OPT_0080_CONTROL_PROFILE,
  OPT_0080_DESCRIPTOR_TABLE_MEMBER_COUNT,
  OPT_0080_EVALUATION0_SHA256,
  OPT_0080_GRAPH_COMMAND_BUFFERS,
  OPT_0080_TOTAL_COMMAND_BUFFERS,
  requireOpt0080Heartbeat,
  requireOpt0080Cancellation,
  requireOpt0080ThermalGate,
  requireOpt0080ThermalTrace,
  requireOpt0080Topology,
  type Opt0080ArmId,
  type Opt0080CancellationEvidence,
  type Opt0080HeartbeatCapture,
  type Opt0080SchedulingProfile,
  type Opt0080ThermalGate,
  type Opt0080ThermalTrace,
  type Opt0080TimingSample,
} from "./opt-0080-dit-depth2-completion-epochs-contract.js";
import {
  buildOpt0080Result,
  type Opt0080CorrectnessArm,
  type Opt0080CorrectnessPreflight,
  type Opt0080TimedArmResult,
} from "./opt-0080-dit-depth2-completion-epochs-result.js";

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
  "/model/files-fp16-vae-revision7-experimental/manifest.json" as const;
const VAE_MANIFEST_SHA256 =
  "36a54d79777d6826088095ba6ebc028fb4bea546368c0f0a29cd0eee8d656da7" as const;
const CONDITIONING_DIAGNOSTIC_CODE =
  "ACE_PLANNER_CONDITIONING_RESOLVED" as const;
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
  readonly armId: Opt0080ArmId;
  readonly thermalGate: Opt0080ThermalGate;
}

interface CompleteThermalMessage {
  readonly type: "complete-thermal";
  readonly armId: Opt0080ArmId;
  readonly thermalTrace: Opt0080ThermalTrace;
  readonly heartbeat: Opt0080HeartbeatCapture;
}

type IncomingMessage = PrepareMessage | RunArmMessage | CompleteThermalMessage;

interface CheckpointDefinition {
  readonly id: string;
  readonly schedulingProfile: Opt0080SchedulingProfile;
  readonly timed: boolean;
  readonly order: -1 | 0 | 1 | 2 | 3;
  readonly armId?: Opt0080ArmId;
}

type TimingBase = Omit<Opt0080TimingSample, "heartbeat">;

interface CheckpointExecution {
  readonly definition: CheckpointDefinition;
  readonly checkpoint: AceOpt0080DitCheckpoint;
  readonly conditioningAuthority: Readonly<Record<string, unknown>>;
  readonly receipt: Readonly<Record<string, unknown>>;
  readonly timingBase?: TimingBase;
  readonly backendCreatedOrdinal: number;
  readonly cleanupCompletedOrdinal: number;
  readonly cleanupCompletedAtEpochMilliseconds: number;
}

interface ActiveTimedRun {
  readonly definition: CheckpointDefinition & Readonly<{ armId: Opt0080ArmId }>;
  readonly authorization: PromiseWithResolvers<void>;
  readyAtEpochMilliseconds: number;
  gate?: Opt0080ThermalGate;
  completed?: CheckpointExecution;
}

let state: "idle" | "preflights" | "preparing-arm" | "ready-arm" |
  "running-arm" | "await-arm-trace" | "settled" = "idle";
let identity: Opt0018RunIdentity | undefined;
let request: AceGenerationRequest | undefined;
let correctness: Opt0080CorrectnessPreflight | undefined;
let cancellation: Opt0080CancellationEvidence | undefined;
let active: ActiveTimedRun | undefined;
let nextArmIndex = 0;
let lifecycleOrdinal = 0;
const accepted: Opt0080TimedArmResult[] = [];
const rejectedSetupAttempts: Readonly<Record<string, unknown>>[] = [];

self.addEventListener("message", (event: MessageEvent<IncomingMessage>) => {
  const message = event.data;
  if (message.type === "prepare" && state === "idle") {
    state = "preflights";
    void prepareGate(message.identity).catch(fail);
    return;
  }
  if (
    message.type === "run-arm" && state === "ready-arm" &&
    active !== undefined && message.armId === active.definition.armId
  ) {
    try {
      active.gate = requireOpt0080ThermalGate(
        message.thermalGate,
        active.readyAtEpochMilliseconds,
        Date.now(),
      );
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
    active?.completed !== undefined && active.gate !== undefined &&
    message.armId === active.definition.armId
  ) {
    try {
      const trace = requireOpt0080ThermalTrace(
        message.thermalTrace,
        active.gate,
        active.completed.cleanupCompletedAtEpochMilliseconds,
        Date.now(),
      );
      const heartbeat = requireOpt0080Heartbeat(message.heartbeat);
      if (heartbeat.completedAtEpochMilliseconds <
        active.completed.cleanupCompletedAtEpochMilliseconds) {
        throw new Error("OPT-0080 heartbeat stopped before cleanup");
      }
      if (accepted.some((prior) =>
        prior.trace.rawTraceSha256 === trace.rawTraceSha256
      )) throw new Error("OPT-0080 thermal trace was reused across arms");
      accepted.push(toTimedArm(
        active.completed,
        active.gate,
        trace,
        heartbeat,
        active.readyAtEpochMilliseconds,
      ));
    } catch (error) {
      retainRejected("thermal-trace", message.armId, error);
      self.postMessage({
        type: "trace-rejected",
        armId: message.armId,
        error: serializeOpt0018Failure(error),
      });
      return;
    }
    active = undefined;
    nextArmIndex += 1;
    if (nextArmIndex === OPT_0080_ARM_ORDER.length) {
      try {
        const result = finalizeGate();
        state = "settled";
        self.postMessage({ type: "gate-complete", result });
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
    message: "untimed correctness: depth1/depth2 exact result repeats",
  });
  const controlFirst = await executeCheckpointArm(preflightDefinition(
    "control-first",
    OPT_0080_CONTROL_PROFILE,
  ));
  const controlRepeat = await executeCheckpointArm(preflightDefinition(
    "control-repeat",
    OPT_0080_CONTROL_PROFILE,
  ));
  const candidateFirst = await executeCheckpointArm(preflightDefinition(
    "candidate-first",
    OPT_0080_CANDIDATE_PROFILE,
  ));
  const candidateRepeat = await executeCheckpointArm(preflightDefinition(
    "candidate-repeat",
    OPT_0080_CANDIDATE_PROFILE,
  ));
  correctness = Object.freeze({
    controlFirst: correctnessArm(controlFirst),
    controlRepeat: correctnessArm(controlRepeat),
    candidateFirst: correctnessArm(candidateFirst),
    candidateRepeat: correctnessArm(candidateRepeat),
  });
  // Validate immediately; the same proof is retained in the final builder.
  assertCorrectnessPreflight(correctness);
  self.postMessage({
    type: "progress",
    message: "actual DiT graph cancellation: abort with one successor submitted",
  });
  cancellation = await runCancellationPreflight();
  await prepareTimedArm();
}

function preflightDefinition(
  id: string,
  schedulingProfile: Opt0080SchedulingProfile,
): CheckpointDefinition {
  return Object.freeze({ id, schedulingProfile, timed: false, order: -1 });
}

async function prepareTimedArm(): Promise<void> {
  const expected = OPT_0080_ARM_ORDER[nextArmIndex];
  if (expected === undefined || request === undefined || active !== undefined) {
    throw new Error("OPT-0080 timed arm preparation order changed");
  }
  state = "preparing-arm";
  const definition = Object.freeze({
    id: expected.armId,
    armId: expected.armId,
    schedulingProfile: expected.schedulingProfile,
    timed: true,
    order: expected.order,
  });
  const authorization = Promise.withResolvers<void>();
  const run: ActiveTimedRun = {
    definition,
    authorization,
    readyAtEpochMilliseconds: 0,
  };
  active = run;
  self.postMessage({
    type: "progress",
    message: `${definition.armId}: authenticating package, inputs, and graph`,
  });
  void executeCheckpointArm(definition, async () => {
    if (active !== run || state !== "preparing-arm") {
      throw new Error("OPT-0080 timing-ready ownership changed");
    }
    run.readyAtEpochMilliseconds = Date.now();
    state = "ready-arm";
    self.postMessage({
      type: "ready-for-arm",
      armId: definition.armId,
      schedulingProfile: definition.schedulingProfile,
      order: definition.order,
      readyAtEpochMilliseconds: run.readyAtEpochMilliseconds,
      ...(definition.order === 0
        ? {
            preflights: Object.freeze({
              correctness: Object.freeze({
                resultSha256: correctness!.controlFirst.resultSha256,
                rawU32Exact: true,
                deterministicRepeatsExact: true,
                descriptorSequenceExact: true,
              }),
              cancellation,
            }),
          }
        : {}),
    });
    await authorization.promise;
  }).then(
    (completed) => {
      if (active !== run || state !== "running-arm" || run.gate === undefined) {
        throw new Error("OPT-0080 arm completed outside its accepted gate");
      }
      run.completed = completed;
      state = "await-arm-trace";
      self.postMessage({
        type: "arm-complete",
        armId: definition.armId,
        cleanupCompletedAtEpochMilliseconds:
          completed.cleanupCompletedAtEpochMilliseconds,
        sample: completed.timingBase,
        receipt: completed.receipt,
      });
    },
    fail,
  ).catch(fail);
}

async function executeCheckpointArm(
  definition: CheckpointDefinition,
  waitForTimingAuthorization?: () => Promise<void>,
): Promise<CheckpointExecution> {
  if (request === undefined) throw new Error("OPT-0080 request is absent");
  const backendCreatedOrdinal = nextOrdinal();
  const backend = createAceWebGpuPipelineBackend();
  const controller = new AbortController();
  const privateStop = Object.freeze({ id: definition.id });
  let checkpoint: AceOpt0080DitCheckpoint | undefined;
  let diagnostics: AceRuntimeDiagnostics | undefined;
  let conditioning: Readonly<Record<string, unknown>> | undefined;
  let fatalDiagnosticCount = 0;
  let forbiddenProgress = false;
  let checkpointOrdinal = 0;
  let cleanupCompletedOrdinal = 0;
  let cleanupCompletedAtEpochMilliseconds = 0;
  let primaryError: unknown;
  let cleanupError: unknown;
  try {
    diagnostics = await backend.initialize(configuration(), {
      modelSource: "cache-or-network",
      signal: controller.signal,
      onProgress: (_progress: AceInitializationProgress) => undefined,
      onDiagnostic: (diagnostic: AceDiagnostic) => {
        if (diagnostic.severity === "error") fatalDiagnosticCount += 1;
      },
    });
    validateDiagnostics(diagnostics);
    const context: AceWebGpuGenerationContext = {
      signal: controller.signal,
      captureTrace: true,
      onProgress: (progress: AceGenerationProgress) => {
        if (FORBIDDEN_POST_DIT_STAGES.has(progress.stage)) forbiddenProgress = true;
      },
      onDiagnostic: (diagnostic: AceDiagnostic) => {
        if (diagnostic.severity === "error") fatalDiagnosticCount += 1;
        if (
          diagnostic.code === CONDITIONING_DIAGNOSTIC_CODE &&
          diagnostic.details !== undefined
        ) {
          if (conditioning !== undefined) {
            throw new Error("OPT-0080 duplicate conditioning authority");
          }
          conditioning = validateConditioning(diagnostic);
        }
      },
      opt0080DitRun: {
        schedulingProfile: definition.schedulingProfile,
        ...(waitForTimingAuthorization === undefined
          ? {}
          : { waitForTimingAuthorization }),
        onCheckpoint(value) {
          if (checkpoint !== undefined) {
            throw new Error(`OPT-0080 ${definition.id} checkpoint repeated`);
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
      throw new Error(`OPT-0080 ${definition.id} continued beyond checkpoint`);
    } catch (error) {
      if (error !== privateStop) throw error;
    }
  } catch (error) {
    primaryError = error;
  } finally {
    try {
      await backend.dispose();
    } catch (error) {
      cleanupError = error;
    }
    cleanupCompletedOrdinal = nextOrdinal();
    cleanupCompletedAtEpochMilliseconds = Date.now();
  }
  if (primaryError !== undefined || cleanupError !== undefined) {
    if (primaryError !== undefined && cleanupError !== undefined) {
      throw new AggregateError(
        [primaryError, cleanupError],
        `OPT-0080 ${definition.id} run/dispose failed`,
      );
    }
    throw primaryError ?? cleanupError;
  }
  if (
    checkpoint === undefined || diagnostics === undefined ||
    conditioning === undefined || checkpointOrdinal === 0 ||
    cleanupCompletedOrdinal === 0 || fatalDiagnosticCount !== 0 ||
    forbiddenProgress
  ) throw new Error(`OPT-0080 ${definition.id} lifecycle is incomplete`);
  const timingBase = definition.timed
    ? timingBaseFromCheckpoint(checkpoint, definition)
    : undefined;
  return Object.freeze({
    definition,
    checkpoint,
    conditioningAuthority: conditioning,
    backendCreatedOrdinal,
    cleanupCompletedOrdinal,
    cleanupCompletedAtEpochMilliseconds,
    ...(timingBase === undefined ? {} : { timingBase }),
    receipt: Object.freeze({
      id: definition.id,
      armId: definition.armId ?? null,
      order: definition.order,
      schedulingProfile: definition.schedulingProfile,
      timed: definition.timed,
      resultSha256: checkpoint.resultSha256,
      resultElementCount: checkpoint.resultElementCount,
      descriptorTableSha256: checkpoint.profile.descriptorTable.sha256,
      descriptorTableMemberCount: checkpoint.profile.descriptorTable.memberCount,
      topology: checkpoint.profile.topology,
      timing: timingBase ?? null,
      lifecycle: Object.freeze({
        backendCreatedOrdinal,
        checkpointOrdinal,
        cleanupCompletedOrdinal,
        cleanupCompletedAtEpochMilliseconds,
        terminalSettlementBeforeCleanup: true,
        allBackendAndDeviceDisposalAwaited: true,
      }),
      evidence: Object.freeze({
        graphPreparedBeforeTimingAuthorization: definition.timed,
        physicalCommandBufferContentsUnchanged: true,
        commandBufferCoalescing: false,
        overlappingFenceLatenciesAreNonAdditive: true,
        epochWallsAreDisjoint: true,
        laterEvaluationEncodeCount: 0,
        vaeWeightAcquireStarted: false,
      }),
    }),
  });
}

function validateCheckpoint(
  checkpoint: AceOpt0080DitCheckpoint,
  definition: CheckpointDefinition,
): void {
  const profile = checkpoint.profile;
  if (
    checkpoint.schema !== "ace-dit-opt0080-m2250-evaluation0-checkpoint-v1" ||
    checkpoint.schedulingProfile !== definition.schedulingProfile ||
    checkpoint.evaluation !== 0 || checkpoint.resultByteLength !== 1_152_000 ||
    checkpoint.resultElementCount !== 288_000 ||
    checkpoint.result.length !== 288_000 ||
    checkpoint.resultSha256 !== OPT_0080_EVALUATION0_SHA256 ||
    checkpoint.resultNonFiniteCount !== 0 || checkpoint.resultNonzeroCount <= 0 ||
    !Number.isFinite(checkpoint.resultMaxAbs) || checkpoint.resultMaxAbs <= 0 ||
    checkpoint.graphCommandBufferCount !== OPT_0080_GRAPH_COMMAND_BUFFERS ||
    checkpoint.readbackCommandBufferCount !== 1 ||
    checkpoint.totalCommandBufferCount !== OPT_0080_TOTAL_COMMAND_BUFFERS ||
    checkpoint.completedEvaluations !== 1 ||
    checkpoint.uncapturedWebGpuErrorCount !== 0 || checkpoint.deviceLost !== false ||
    profile.schema !== "ace-dit-opt0080-evaluation0-command-profile-v1" ||
    profile.schedulingProfile !== definition.schedulingProfile ||
    !/^[0-9a-f]{64}$/u.test(profile.descriptorTable.sha256) ||
    profile.descriptorTable.memberCount !==
      OPT_0080_DESCRIPTOR_TABLE_MEMBER_COUNT ||
    profile.topology.schedulingProfile !== definition.schedulingProfile
  ) throw new Error(`OPT-0080 ${definition.id} checkpoint changed`);
  requireOpt0080Topology(profile.topology);
}

function timingBaseFromCheckpoint(
  checkpoint: AceOpt0080DitCheckpoint,
  definition: CheckpointDefinition,
): TimingBase {
  const order = definition.order;
  if (!definition.timed || definition.armId === undefined || order < 0) {
    throw new Error("OPT-0080 timing definition changed");
  }
  const profile = checkpoint.profile;
  // Epoch timings are disjoint. Never sum submitThroughCompletionFenceMs.
  const graphEpochWallSumMs = profile.topology.graphCompletionEpochs.reduce(
    (sum, epoch) => sum + epoch.submitThroughTrueDrainMs,
    0,
  );
  return Object.freeze({
    armId: definition.armId,
    order: order as 0 | 1 | 2 | 3,
    schedulingProfile: definition.schedulingProfile,
    precomputeWallMs: profile.precomputeWallMs,
    evaluationWallMs: profile.evaluationWallMs,
    graphWallMs: profile.graphWallMs,
    graphToReadbackObservedIdleMs: profile.graphToReadbackObservedIdleMs,
    readbackFenceMs: profile.readbackSubmitThroughCompletionFenceMs,
    readbackMapMs: profile.readbackMapDetachMs,
    backendWallMs: profile.backendWallMs,
    graphEpochWallSumMs,
    topology: profile.topology,
  });
}

function correctnessArm(execution: CheckpointExecution): Opt0080CorrectnessArm {
  const checkpoint = execution.checkpoint;
  return Object.freeze({
    schedulingProfile: execution.definition.schedulingProfile,
    result: checkpoint.result,
    resultSha256: checkpoint.resultSha256,
    descriptorTableSha256: checkpoint.profile.descriptorTable.sha256,
    descriptorTableMemberCount: checkpoint.profile.descriptorTable.memberCount,
    graphCommandBufferCount: checkpoint.graphCommandBufferCount,
    resultNonFiniteCount: checkpoint.resultNonFiniteCount,
    resultNonzeroCount: checkpoint.resultNonzeroCount,
    uncapturedWebGpuErrorCount: checkpoint.uncapturedWebGpuErrorCount,
    deviceLost: checkpoint.deviceLost,
    conditioningAuthority: execution.conditioningAuthority,
    receipt: execution.receipt,
  });
}

function assertCorrectnessPreflight(value: Opt0080CorrectnessPreflight): void {
  const arms = [
    value.controlFirst,
    value.controlRepeat,
    value.candidateFirst,
    value.candidateRepeat,
  ] as const;
  const reference = arms[0];
  if (
    reference === undefined ||
    arms.some((arm) =>
      arm.resultSha256 !== OPT_0080_EVALUATION0_SHA256 ||
      !/^[0-9a-f]{64}$/u.test(arm.descriptorTableSha256) ||
      arm.descriptorTableMemberCount !==
        OPT_0080_DESCRIPTOR_TABLE_MEMBER_COUNT ||
      arm.graphCommandBufferCount !== OPT_0080_GRAPH_COMMAND_BUFFERS ||
      arm.resultNonFiniteCount !== 0 || arm.resultNonzeroCount <= 0 ||
      arm.uncapturedWebGpuErrorCount !== 0 ||
      arm.deviceLost || !rawU32Equal(reference.result, arm.result)
    )
  ) throw new Error("OPT-0080 correctness/repeat preflight failed");
}

async function runCancellationPreflight(): Promise<Opt0080CancellationEvidence> {
  if (request === undefined) throw new Error("OPT-0080 request is absent");
  const backend = createAceWebGpuPipelineBackend();
  const controller = new AbortController();
  const stop = Object.freeze({ kind: "opt-0080-cancellation-preflight" });
  let diagnostics: AceRuntimeDiagnostics | undefined;
  let conditioning: Readonly<Record<string, unknown>> | undefined;
  let fatalDiagnosticCount = 0;
  let specializedCompletionCallbackCount = 0;
  let specializedCompletionCallbackCountAfterAbort = 0;
  let publicProgressAfterAbortCount = 0;
  let checkpointCount = 0;
  let abortedCommandBufferIndex = -1;
  let outstandingSuccessorsAtAbort = 0;
  let completionFenceRequestedCountAtAbort = 0;
  let completedCommandBufferCountAtAbort = 0;
  let abortObservedAt = 0;
  let generationRejectedAt = 0;
  let backendDisposeCompletedAt = 0;
  let unhandledRejectionCount = 0;
  let unhandledRejectionListenerRemoved = false;
  const onUnhandledRejection = (event: PromiseRejectionEvent): void => {
    unhandledRejectionCount += 1;
    event.preventDefault();
  };
  self.addEventListener("unhandledrejection", onUnhandledRejection);
  let primaryError: unknown;
  let cleanupError: unknown;
  try {
    diagnostics = await backend.initialize(configuration(), {
      modelSource: "cache-or-network",
      signal: controller.signal,
      onProgress: (_progress: AceInitializationProgress) => undefined,
      onDiagnostic: (diagnostic: AceDiagnostic) => {
        if (diagnostic.severity === "error") fatalDiagnosticCount += 1;
      },
    });
    validateDiagnostics(diagnostics);
    const context: AceWebGpuGenerationContext = {
      signal: controller.signal,
      captureTrace: true,
      onProgress: (_progress: AceGenerationProgress) => {
        if (controller.signal.aborted) publicProgressAfterAbortCount += 1;
      },
      onDiagnostic: (diagnostic: AceDiagnostic) => {
        if (diagnostic.severity === "error") fatalDiagnosticCount += 1;
        if (
          diagnostic.code === CONDITIONING_DIAGNOSTIC_CODE &&
          diagnostic.details !== undefined
        ) {
          if (conditioning !== undefined) {
            throw new Error("OPT-0080 duplicate cancellation conditioning");
          }
          conditioning = validateConditioning(diagnostic);
        }
      },
      opt0080DitRun: {
        schedulingProfile: OPT_0080_CANDIDATE_PROFILE,
        onCommandBufferCompleted(completion) {
          if (controller.signal.aborted) {
            specializedCompletionCallbackCountAfterAbort += 1;
            return;
          }
          specializedCompletionCallbackCount += 1;
          const progress = completion.schedulingProgress;
          if (
            completion.commandBufferIndex !== 0 ||
            completion.descriptor.physicalIndex !== 0 ||
            progress.completedCommandBuffers !== 1 ||
            progress.completionFenceRequestedCount !== 2 ||
            progress.outstandingCommandBuffers !== 1
          ) {
            throw new Error(
              "OPT-0080 actual-graph cancellation callback topology changed",
            );
          }
          abortedCommandBufferIndex = completion.commandBufferIndex;
          completedCommandBufferCountAtAbort = progress.completedCommandBuffers;
          completionFenceRequestedCountAtAbort =
            progress.completionFenceRequestedCount;
          outstandingSuccessorsAtAbort = progress.outstandingCommandBuffers;
          abortObservedAt = performance.now();
          controller.abort(stop);
        },
        onCheckpoint() {
          checkpointCount += 1;
          throw new Error(
            "OPT-0080 cancellation continued to the evaluation checkpoint",
          );
        },
      },
    };
    try {
      await backend.generate(request, context);
      throw new Error("OPT-0080 cancellation graph unexpectedly completed");
    } catch (error) {
      generationRejectedAt = performance.now();
      if (error !== stop) throw error;
    }
  } catch (error) {
    primaryError = error;
  } finally {
    try {
      await backend.dispose();
    } catch (error) {
      cleanupError = error;
    }
    backendDisposeCompletedAt = performance.now();
    self.removeEventListener("unhandledrejection", onUnhandledRejection);
    unhandledRejectionListenerRemoved = true;
  }
  if (primaryError !== undefined || cleanupError !== undefined) {
    if (primaryError !== undefined && cleanupError !== undefined) {
      throw new AggregateError(
        [primaryError, cleanupError],
        "OPT-0080 actual-graph cancellation run/dispose failed",
      );
    }
    throw primaryError ?? cleanupError;
  }
  if (
    diagnostics === undefined || conditioning === undefined ||
    fatalDiagnosticCount !== 0 || abortedCommandBufferIndex !== 0 ||
    outstandingSuccessorsAtAbort !== 1 ||
    completionFenceRequestedCountAtAbort -
      completedCommandBufferCountAtAbort !== 1 ||
    specializedCompletionCallbackCount !== 1 ||
    specializedCompletionCallbackCountAfterAbort !== 0 ||
    publicProgressAfterAbortCount !== 0 || checkpointCount !== 0 ||
    unhandledRejectionCount !== 0 || !unhandledRejectionListenerRemoved ||
    abortObservedAt === 0 || generationRejectedAt < abortObservedAt ||
    backendDisposeCompletedAt < generationRejectedAt
  ) throw new Error("OPT-0080 actual-graph cancellation invariant changed");
  const evidence: Opt0080CancellationEvidence = Object.freeze({
    scope: "actual-dit-evaluation0-graph",
    schedulingProfile: OPT_0080_CANDIDATE_PROFILE,
    abortedCommandBufferIndex: 0,
    abortObservedFromCompletionCallback: true,
    outstandingSuccessorsAtAbort: 1,
    prefetchedTailCommandCount: 1,
    backfillAfterAbortCount: 0,
    publicProgressAfterAbortCount: 0,
    specializedCompletionCallbackCount: 1,
    specializedCompletionCallbackCountAfterAbort: 0,
    completedCommandBufferCountAtAbort: 1,
    completionFenceRequestedCountAtAbort: 2,
    checkpointCount: 0,
    unhandledRejectionCount: 0,
    unhandledRejectionListenerRemoved: true,
    generationRejectedWithAbortReason: true,
    // The graph promise rejects only after the scheduler settles submitted work;
    // the pipeline then awaits graph/model destruction before propagating it.
    terminalSettlementBeforeGenerationRejection: true,
    graphAndModelDestroyAwaitedBeforeGenerationRejection: true,
    backendDisposeAwaitedAfterGenerationRejection: true,
    abortObservationThroughRejectionAndCleanupMs:
      backendDisposeCompletedAt - abortObservedAt,
  });
  return requireOpt0080Cancellation(evidence);
}

function toTimedArm(
  execution: CheckpointExecution,
  gate: Opt0080ThermalGate,
  trace: Opt0080ThermalTrace,
  heartbeat: Opt0080HeartbeatCapture,
  timingReadyAtEpochMilliseconds: number,
): Opt0080TimedArmResult {
  if (
    execution.definition.armId === undefined ||
    execution.timingBase === undefined
  ) throw new Error("OPT-0080 timed execution is incomplete");
  const checkpoint = execution.checkpoint;
  return Object.freeze({
    armId: execution.definition.armId,
    schedulingProfile: execution.definition.schedulingProfile,
    result: checkpoint.result,
    resultSha256: checkpoint.resultSha256,
    descriptorTableSha256: checkpoint.profile.descriptorTable.sha256,
    conditioningAuthority: execution.conditioningAuthority,
    sample: Object.freeze({ ...execution.timingBase, heartbeat }),
    gate,
    trace,
    receipt: execution.receipt,
    backendCreatedOrdinal: execution.backendCreatedOrdinal,
    cleanupCompletedOrdinal: execution.cleanupCompletedOrdinal,
    cleanupCompletedAtEpochMilliseconds:
      execution.cleanupCompletedAtEpochMilliseconds,
    timingReadyAtEpochMilliseconds,
  });
}

function finalizeGate(): Readonly<Record<string, unknown>> {
  if (
    identity === undefined || request === undefined || correctness === undefined ||
    cancellation === undefined || accepted.length !== OPT_0080_ARM_ORDER.length
  ) throw new Error("OPT-0080 final inventory is incomplete");
  return buildOpt0080Result({
    runIdentity: identity,
    requestCanonicalJson: OPT_0018_CANONICAL_REQUEST_JSON,
    requestSha256: OPT_0018_CANONICAL_REQUEST_SHA256,
    requestByteLength: OPT_0018_CANONICAL_REQUEST_BYTES,
    mainManifestSha256: MAIN_MANIFEST_SHA256,
    denseManifestSha256: DENSE_MANIFEST_SHA256,
    denseRuntimeProfile: DENSE_RUNTIME_PROFILE,
    attentionRuntimeProfile:
      ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
    correctness,
    cancellation,
    arms: Object.freeze([...accepted]),
    rejectedSetupAttempts: Object.freeze([...rejectedSetupAttempts]),
  });
}

function configuration(): AceWorkerConfiguration {
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
    ditAttentionRuntimeProfile:
      ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
    vaePackage: Object.freeze({
      manifestUrl: absoluteUrl(VAE_MANIFEST_PATH),
      manifestSha256: VAE_MANIFEST_SHA256,
      runtimeProfile:
        ACE_OPT_0072_VAE_FP16_FIXED32_DUAL_K4_PRODUCTION_RUNTIME_PROFILE,
      windowRuntimeProfile: ACE_OPT_0070_VAE_C2378_WINDOW_RUNTIME_PROFILE,
      maxWindowFrames: ACE_OPT_0070_VAE_C2378_MAXIMUM_WINDOW_FRAMES,
    }),
  });
}

function validateDiagnostics(diagnostics: AceRuntimeDiagnostics): void {
  if (
    diagnostics.modelManifestSha256 !== MAIN_MANIFEST_SHA256 ||
    diagnostics.ditDenseManifestSha256 !== DENSE_MANIFEST_SHA256 ||
    diagnostics.ditDenseManifestByteLength !== DENSE_MANIFEST_BYTES ||
    diagnostics.ditDenseRuntimeProfile !== DENSE_RUNTIME_PROFILE ||
    diagnostics.ditDenseKernelSetId !== DENSE_KERNEL_SET_ID ||
    diagnostics.ditAttentionRuntimeProfile !==
      ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE ||
    diagnostics.ditAttentionKernelSetId !==
      ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_KERNEL_SET_ID ||
    diagnostics.vaeManifestSha256 !== VAE_MANIFEST_SHA256 ||
    diagnostics.vaeRuntimeProfile !==
      ACE_OPT_0072_VAE_FP16_FIXED32_DUAL_K4_PRODUCTION_RUNTIME_PROFILE ||
    diagnostics.vaeKernelSetId !==
      ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_KERNEL_SET_ID ||
    diagnostics.vaeWindowRuntimeProfile !==
      ACE_OPT_0070_VAE_C2378_WINDOW_RUNTIME_PROFILE ||
    diagnostics.vaeMaxWindowFrames !==
      ACE_OPT_0070_VAE_C2378_MAXIMUM_WINDOW_FRAMES ||
    diagnostics.executionProfile.id !== "reference-bf16-subgroups" ||
    diagnostics.schedulingProfile !== "cooperative" ||
    diagnostics.capabilities.adapterInfo.subgroupMinSize !== 32 ||
    diagnostics.capabilities.adapterInfo.subgroupMaxSize !== 32 ||
    !diagnostics.capabilities.deviceFeatures.includes("subgroups") ||
    !diagnostics.capabilities.deviceFeatures.includes("shader-f16")
  ) throw new Error("OPT-0080 authenticated runtime changed");
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
  ) throw new Error("OPT-0080 conditioning authority changed");
  return Object.freeze({
    textTokenCount: 82,
    textTokenSha256: details.textTokenSha256,
    lyricTokenCount: 15,
    lyricTokenSha256: details.lyricTokenSha256,
    conditionTokenCount: 98,
  });
}

function validateCanonicalRequest(): AceGenerationRequest {
  const bytes = new TextEncoder().encode(OPT_0018_CANONICAL_REQUEST_JSON);
  if (
    bytes.byteLength !== OPT_0018_CANONICAL_REQUEST_BYTES ||
    aceSha256Hex(bytes) !== OPT_0018_CANONICAL_REQUEST_SHA256
  ) throw new Error("OPT-0080 inherited request identity changed");
  const value = createOpt0018Request() as unknown as AceGenerationRequest;
  assertAceGenerationRequest(value);
  if (JSON.stringify(value) !== OPT_0018_CANONICAL_REQUEST_JSON) {
    throw new Error("OPT-0080 inherited request values changed");
  }
  return Object.freeze(value);
}

function rawU32Equal(left: Float32Array, right: Float32Array): boolean {
  if (left.length !== 288_000 || right.length !== 288_000) return false;
  const a = new Uint32Array(left.buffer, left.byteOffset, left.length);
  const b = new Uint32Array(right.buffer, right.byteOffset, right.length);
  return a.every((word, index) => word === b[index]);
}

function retainRejected(
  kind: "thermal-gate" | "thermal-trace",
  armId: Opt0080ArmId,
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

function fail(error: unknown): void {
  if (state === "settled") return;
  state = "settled";
  active?.authorization.reject(error);
  self.postMessage({ type: "failed", error: serializeOpt0018Failure(error) });
}
