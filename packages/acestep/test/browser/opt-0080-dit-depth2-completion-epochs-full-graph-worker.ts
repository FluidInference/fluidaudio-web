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
  type AceOpt0080FullDitCheckpoint,
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
  OPT_0080_CANDIDATE_PROFILE,
  OPT_0080_CONTROL_PROFILE,
  OPT_0080_EVALUATION_SLICE_EVIDENCE_SHA256,
  OPT_0080_FULL_ARM_ORDER,
  OPT_0080_FULL_DESCRIPTOR_TABLE_MEMBER_COUNT,
  OPT_0080_FULL_EVALUATION_COUNT,
  OPT_0080_FULL_EVALUATION_SHA256,
  OPT_0080_FULL_FINAL_LATENT_SHA256,
  OPT_0080_FULL_GRAPH_COMMAND_BUFFERS,
  OPT_0080_FULL_TENSOR_BYTES,
  OPT_0080_FULL_TENSOR_ELEMENTS,
  OPT_0080_FULL_TOTAL_COMMAND_BUFFERS,
  exactOpt0080FullTensorIdentity,
  requireOpt0080FullCancellation,
  requireOpt0080FullHeartbeat,
  requireOpt0080FullThermalGate,
  requireOpt0080FullThermalTrace,
  requireOpt0080FullTopology,
  type Opt0080FullArmId,
  type Opt0080FullCancellationEvidence,
  type Opt0080FullHeartbeatCapture,
  type Opt0080FullSchedulingProfile,
  type Opt0080FullThermalGate,
  type Opt0080FullThermalTrace,
  type Opt0080FullTimingSample,
} from
  "./opt-0080-dit-depth2-completion-epochs-full-graph-contract.js";
import {
  buildOpt0080FullResult,
  type Opt0080FullCorrectnessArm,
  type Opt0080FullCorrectnessPreflight,
  type Opt0080FullTimedArmResult,
} from
  "./opt-0080-dit-depth2-completion-epochs-full-graph-result.js";

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
  readonly armId: Opt0080FullArmId;
  readonly thermalGate: Opt0080FullThermalGate;
}

interface CompleteThermalMessage {
  readonly type: "complete-thermal";
  readonly armId: Opt0080FullArmId;
  readonly thermalTrace: Opt0080FullThermalTrace;
  readonly heartbeat: Opt0080FullHeartbeatCapture;
}

type IncomingMessage = PrepareMessage | RunArmMessage | CompleteThermalMessage;

interface CheckpointDefinition {
  readonly id: string;
  readonly schedulingProfile: Opt0080FullSchedulingProfile;
  readonly captureEvaluationTaps: boolean;
  readonly timed: boolean;
  readonly order: -1 | 0 | 1;
  readonly armId?: Opt0080FullArmId;
}

type TimingBase = Omit<Opt0080FullTimingSample, "heartbeat">;

interface CheckpointExecution {
  readonly definition: CheckpointDefinition;
  readonly checkpoint: AceOpt0080FullDitCheckpoint;
  readonly conditioningAuthority: Readonly<Record<string, unknown>>;
  readonly receipt: Readonly<Record<string, unknown>>;
  readonly timingBase?: TimingBase;
  readonly backendCreatedOrdinal: number;
  readonly cleanupCompletedOrdinal: number;
  readonly cleanupCompletedAtEpochMilliseconds: number;
}

interface ActiveTimedRun {
  readonly definition: CheckpointDefinition &
    Readonly<{ readonly armId: Opt0080FullArmId }>;
  readonly authorization: PromiseWithResolvers<void>;
  readyAtEpochMilliseconds: number;
  gate?: Opt0080FullThermalGate;
  completed?: CheckpointExecution;
}

let state: "idle" | "preflights" | "preparing-arm" | "ready-arm" |
  "running-arm" | "await-arm-trace" | "settled" = "idle";
let identity: Opt0018RunIdentity | undefined;
let request: AceGenerationRequest | undefined;
let correctness: Opt0080FullCorrectnessPreflight | undefined;
let cancellation: Opt0080FullCancellationEvidence | undefined;
let active: ActiveTimedRun | undefined;
let nextArmIndex = 0;
let lifecycleOrdinal = 0;
const accepted: Opt0080FullTimedArmResult[] = [];
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
      active.gate = requireOpt0080FullThermalGate(
        message.thermalGate,
        active.readyAtEpochMilliseconds,
        Date.now(),
      );
      state = "running-arm";
      self.postMessage({
        type: "progress",
        message: `${message.armId}: gate accepted; running all 8 evaluations`,
      });
      active.authorization.resolve();
    } catch (error) {
      retainRejected(
        "thermal-gate",
        message.armId,
        error,
        Object.freeze({ thermalGate: message.thermalGate }),
      );
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
      const trace = requireOpt0080FullThermalTrace(
        message.thermalTrace,
        active.gate,
        active.completed.cleanupCompletedAtEpochMilliseconds,
        Date.now(),
      );
      const heartbeat = requireOpt0080FullHeartbeat(message.heartbeat);
      if (
        heartbeat.completedAtEpochMilliseconds <
          active.completed.cleanupCompletedAtEpochMilliseconds
      ) throw new Error("OPT-0080 full heartbeat stopped before cleanup");
      if (accepted.some((prior) =>
        prior.trace.rawTraceSha256 === trace.rawTraceSha256
      )) throw new Error("OPT-0080 full thermal trace was reused");
      accepted.push(toTimedArm(
        active.completed,
        active.gate,
        trace,
        heartbeat,
        active.readyAtEpochMilliseconds,
      ));
    } catch (error) {
      retainRejected(
        "thermal-trace",
        message.armId,
        error,
        Object.freeze({
          thermalTrace: message.thermalTrace,
          heartbeat: message.heartbeat,
        }),
      );
      self.postMessage({
        type: "trace-rejected",
        armId: message.armId,
        error: serializeOpt0018Failure(error),
      });
      return;
    }
    active = undefined;
    nextArmIndex += 1;
    if (nextArmIndex === OPT_0080_FULL_ARM_ORDER.length) {
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
    message: "untimed full control: capturing all 8 evaluation taps",
  });
  const control = await executeCheckpointArm(correctnessDefinition(
    "untimed-control",
    OPT_0080_CONTROL_PROFILE,
  ));
  self.postMessage({
    type: "progress",
    message: "untimed full candidate: capturing all 8 evaluation taps",
  });
  const candidate = await executeCheckpointArm(correctnessDefinition(
    "untimed-candidate",
    OPT_0080_CANDIDATE_PROFILE,
  ));
  correctness = Object.freeze({
    control: correctnessArm(control),
    candidate: correctnessArm(candidate),
  });
  assertCorrectnessPreflight(correctness);
  self.postMessage({
    type: "progress",
    message: "actual full candidate graph cancellation with one successor",
  });
  cancellation = await runCancellationPreflight();
  await prepareTimedArm();
}

function correctnessDefinition(
  id: string,
  schedulingProfile: Opt0080FullSchedulingProfile,
): CheckpointDefinition {
  return Object.freeze({
    id,
    schedulingProfile,
    captureEvaluationTaps: true,
    timed: false,
    order: -1,
  });
}

async function prepareTimedArm(): Promise<void> {
  const expected = OPT_0080_FULL_ARM_ORDER[nextArmIndex];
  if (expected === undefined || request === undefined || active !== undefined) {
    throw new Error("OPT-0080 full timed arm preparation order changed");
  }
  state = "preparing-arm";
  const definition = Object.freeze({
    id: expected.armId,
    armId: expected.armId,
    schedulingProfile: expected.schedulingProfile,
    captureEvaluationTaps: false,
    timed: true,
    order: expected.order,
  });
  const authorization = Promise.withResolvers<void>();
  // A setup failure can occur before the pipeline reaches the authorization
  // callback. Attach a rejection observer immediately so fail() can terminate
  // the worker without creating an unrelated unhandled promise rejection.
  void authorization.promise.catch(() => undefined);
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
      throw new Error("OPT-0080 full timing-ready ownership changed");
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
                evaluationSha256: OPT_0080_FULL_EVALUATION_SHA256,
                finalLatentSha256: OPT_0080_FULL_FINAL_LATENT_SHA256,
                allEvaluationTapRawU32Exact: true,
                finalLatentRawU32Exact: true,
              }),
              cancellation,
              authorizedByEvaluationSliceEvidenceSha256:
                OPT_0080_EVALUATION_SLICE_EVIDENCE_SHA256,
            }),
          }
        : {}),
    });
    await authorization.promise;
  }).then(
    (completed) => {
      if (active !== run || state !== "running-arm" || run.gate === undefined) {
        throw new Error("OPT-0080 full arm completed outside its gate");
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
  if (request === undefined) throw new Error("OPT-0080 full request is absent");
  const backendCreatedOrdinal = nextOrdinal();
  const backend = createAceWebGpuPipelineBackend();
  const controller = new AbortController();
  const privateStop = Object.freeze({ id: definition.id });
  let checkpoint: AceOpt0080FullDitCheckpoint | undefined;
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
            throw new Error("OPT-0080 full duplicate conditioning authority");
          }
          conditioning = validateConditioning(diagnostic);
        }
      },
      opt0080FullDitRun: {
        schedulingProfile: definition.schedulingProfile,
        ...(definition.captureEvaluationTaps
          ? { captureEvaluationTaps: true as const }
          : {}),
        ...(waitForTimingAuthorization === undefined
          ? {}
          : { waitForTimingAuthorization }),
        onCheckpoint(value) {
          if (checkpoint !== undefined) {
            throw new Error(`OPT-0080 full ${definition.id} checkpoint repeated`);
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
      throw new Error(`OPT-0080 full ${definition.id} continued past checkpoint`);
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
        `OPT-0080 full ${definition.id} run/dispose failed`,
      );
    }
    throw primaryError ?? cleanupError;
  }
  if (
    checkpoint === undefined || diagnostics === undefined ||
    conditioning === undefined || checkpointOrdinal === 0 ||
    cleanupCompletedOrdinal === 0 || fatalDiagnosticCount !== 0 ||
    forbiddenProgress
  ) throw new Error(`OPT-0080 full ${definition.id} lifecycle is incomplete`);
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
      captureEvaluationTaps: checkpoint.captureEvaluationTaps,
      evaluationTapCount: checkpoint.evaluationTaps?.length ?? 0,
      evaluationSha256:
        checkpoint.evaluationTaps?.map((tap) => tap.sha256) ?? [],
      finalLatentSha256: checkpoint.finalLatentSha256,
      descriptorTableSha256: checkpoint.profile.descriptorTable.sha256,
      descriptorTableMemberCount:
        checkpoint.profile.descriptorTable.memberCount,
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
        timedOrdinaryFinalReadbackOnly:
          definition.timed && checkpoint.evaluationTaps === undefined,
        commandBufferContentMode: definition.timed
          ? "ordinary-final-readback-only"
          : "eight-in-command-evaluation-tap-copies",
        schedulingPeerCommandBufferContentsExact: true,
        commandBufferCoalescing: false,
        overlappingFenceLatenciesAreNonAdditive: true,
        epochWallsAreDisjoint: true,
        vaeWeightAcquireStarted: false,
      }),
    }),
  });
}

function validateCheckpoint(
  checkpoint: AceOpt0080FullDitCheckpoint,
  definition: CheckpointDefinition,
): void {
  const profile = checkpoint.profile;
  const expectedTapCapture = definition.captureEvaluationTaps;
  if (
    checkpoint.schema !== "ace-dit-opt0080-m2250-full-checkpoint-v1" ||
    checkpoint.schedulingProfile !== definition.schedulingProfile ||
    checkpoint.captureEvaluationTaps !== expectedTapCapture ||
    checkpoint.finalLatent.length !== OPT_0080_FULL_TENSOR_ELEMENTS ||
    checkpoint.finalLatent.byteLength !== OPT_0080_FULL_TENSOR_BYTES ||
    !sameBufferViews(checkpoint.finalLatent, checkpoint.finalLatentRawU32) ||
    checkpoint.finalLatentSha256 !== OPT_0080_FULL_FINAL_LATENT_SHA256 ||
    aceSha256Hex(bytesOf(checkpoint.finalLatent)) !==
      checkpoint.finalLatentSha256 ||
    checkpoint.finalLatentNonFiniteCount !== 0 ||
    checkpoint.finalLatentNonzeroCount <= 0 ||
    !(checkpoint.finalLatentMaxAbs > 0) ||
    checkpoint.graphCommandBufferCount !== OPT_0080_FULL_GRAPH_COMMAND_BUFFERS ||
    checkpoint.readbackCommandBufferCount !== 1 ||
    checkpoint.totalCommandBufferCount !== OPT_0080_FULL_TOTAL_COMMAND_BUFFERS ||
    checkpoint.completedEvaluations !== OPT_0080_FULL_EVALUATION_COUNT ||
    checkpoint.evaluationTapInCommandCopyCount !==
      (expectedTapCapture ? 8 : 0) ||
    checkpoint.evaluationTapExtraCommandBufferCount !== 0 ||
    checkpoint.evaluationTapExtraQueueDrainCount !== 0 ||
    profile.schema !== "ace-dit-opt0080-full-command-profile-v1" ||
    profile.schedulingProfile !== definition.schedulingProfile ||
    !/^[0-9a-f]{64}$/u.test(profile.descriptorTable.sha256) ||
    profile.descriptorTable.memberCount !==
      OPT_0080_FULL_DESCRIPTOR_TABLE_MEMBER_COUNT ||
    profile.descriptorTable.descriptors.length !==
      OPT_0080_FULL_GRAPH_COMMAND_BUFFERS ||
    profile.topology.schedulingProfile !== definition.schedulingProfile ||
    (expectedTapCapture
      ? checkpoint.evaluationTaps === undefined ||
        checkpoint.evaluationTaps.length !== OPT_0080_FULL_EVALUATION_COUNT ||
        checkpoint.evaluationTaps.some((tap, evaluation) =>
          tap.evaluation !== evaluation ||
          tap.result.length !== OPT_0080_FULL_TENSOR_ELEMENTS ||
          tap.result.byteLength !== OPT_0080_FULL_TENSOR_BYTES ||
          !sameBufferViews(tap.result, tap.rawU32) ||
          tap.sha256 !== OPT_0080_FULL_EVALUATION_SHA256[evaluation] ||
          aceSha256Hex(bytesOf(tap.result)) !== tap.sha256 ||
          tap.nonFiniteCount !== 0 || tap.nonzeroCount <= 0 || !(tap.maxAbs > 0)
        ) || !rawU32Equal(
          checkpoint.evaluationTaps[7]!.rawU32,
          checkpoint.finalLatentRawU32,
        )
      : checkpoint.evaluationTaps !== undefined)
  ) throw new Error(`OPT-0080 full ${definition.id} checkpoint changed`);
  requireOpt0080FullTopology(profile.topology);
}

function timingBaseFromCheckpoint(
  checkpoint: AceOpt0080FullDitCheckpoint,
  definition: CheckpointDefinition,
): TimingBase {
  if (
    !definition.timed || definition.armId === undefined ||
    definition.order < 0 || checkpoint.evaluationTaps !== undefined ||
    checkpoint.captureEvaluationTaps
  ) throw new Error("OPT-0080 full timing definition changed");
  const profile = checkpoint.profile;
  // Epoch timings are disjoint. Never sum submitThroughCompletionFenceMs.
  const graphEpochWallSumMs = profile.topology.graphCompletionEpochs.reduce(
    (sum, epoch) => sum + epoch.submitThroughTrueDrainMs,
    0,
  );
  return Object.freeze({
    armId: definition.armId,
    order: definition.order as 0 | 1,
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

function correctnessArm(
  execution: CheckpointExecution,
): Opt0080FullCorrectnessArm {
  const checkpoint = execution.checkpoint;
  if (!checkpoint.captureEvaluationTaps || checkpoint.evaluationTaps === undefined) {
    throw new Error("OPT-0080 full correctness taps are absent");
  }
  return Object.freeze({
    schedulingProfile: execution.definition.schedulingProfile,
    captureEvaluationTaps: true,
    evaluationTaps: checkpoint.evaluationTaps,
    finalLatent: checkpoint.finalLatent,
    finalLatentRawU32: checkpoint.finalLatentRawU32,
    finalLatentSha256: checkpoint.finalLatentSha256,
    finalLatentNonFiniteCount: checkpoint.finalLatentNonFiniteCount,
    finalLatentNonzeroCount: checkpoint.finalLatentNonzeroCount,
    finalLatentMaxAbs: checkpoint.finalLatentMaxAbs,
    descriptorTableSha256: checkpoint.profile.descriptorTable.sha256,
    descriptorTableMemberCount: checkpoint.profile.descriptorTable.memberCount,
    topology: checkpoint.profile.topology,
    evaluationTapInCommandCopyCount:
      checkpoint.evaluationTapInCommandCopyCount as 8,
    evaluationTapExtraCommandBufferCount:
      checkpoint.evaluationTapExtraCommandBufferCount,
    evaluationTapExtraQueueDrainCount:
      checkpoint.evaluationTapExtraQueueDrainCount,
    uncapturedWebGpuErrorCount: 0,
    deviceLost: false,
    conditioningAuthority: execution.conditioningAuthority,
    receipt: execution.receipt,
  });
}

function assertCorrectnessPreflight(
  value: Opt0080FullCorrectnessPreflight,
): void {
  const control = value.control;
  const candidate = value.candidate;
  if (
    control.evaluationTaps.length !== OPT_0080_FULL_EVALUATION_COUNT ||
    candidate.evaluationTaps.length !== OPT_0080_FULL_EVALUATION_COUNT ||
    control.evaluationTaps.some((tap, evaluation) =>
      tap.sha256 !== OPT_0080_FULL_EVALUATION_SHA256[evaluation] ||
      candidate.evaluationTaps[evaluation]!.sha256 !== tap.sha256 ||
      !exactOpt0080FullTensorIdentity(
        tap.result,
        candidate.evaluationTaps[evaluation]!.result,
      )
    ) || control.finalLatentSha256 !== OPT_0080_FULL_FINAL_LATENT_SHA256 ||
    candidate.finalLatentSha256 !== control.finalLatentSha256 ||
    !exactOpt0080FullTensorIdentity(
      control.finalLatent,
      candidate.finalLatent,
    )
  ) throw new Error("OPT-0080 full A/B trajectory preflight failed");
}

async function runCancellationPreflight(): Promise<Opt0080FullCancellationEvidence> {
  if (request === undefined) throw new Error("OPT-0080 full request is absent");
  const backend = createAceWebGpuPipelineBackend();
  const controller = new AbortController();
  const stop = Object.freeze({ kind: "opt-0080-full-cancellation-preflight" });
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
        ) conditioning = validateConditioning(diagnostic);
      },
      opt0080FullDitRun: {
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
          ) throw new Error("OPT-0080 full cancellation topology changed");
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
          throw new Error("OPT-0080 full cancellation reached checkpoint");
        },
      },
    };
    try {
      await backend.generate(request, context);
      throw new Error("OPT-0080 full cancellation unexpectedly completed");
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
    // Let the browser dispatch any rejection event queued by already-settled
    // scheduler/device promises before accepting a zero-unhandled result.
    await new Promise<void>((resolve) => setTimeout(resolve, 0));
    self.removeEventListener("unhandledrejection", onUnhandledRejection);
    unhandledRejectionListenerRemoved = true;
  }
  if (primaryError !== undefined || cleanupError !== undefined) {
    if (primaryError !== undefined && cleanupError !== undefined) {
      throw new AggregateError(
        [primaryError, cleanupError],
        "OPT-0080 full cancellation run/dispose failed",
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
  ) throw new Error("OPT-0080 full cancellation invariant changed");
  return requireOpt0080FullCancellation(Object.freeze({
    scope: "actual-dit-full-graph",
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
    terminalSettlementBeforeGenerationRejection: true,
    graphAndModelDestroyAwaitedBeforeGenerationRejection: true,
    backendDisposeAwaitedAfterGenerationRejection: true,
    abortObservationThroughRejectionAndCleanupMs:
      backendDisposeCompletedAt - abortObservedAt,
  }));
}

function toTimedArm(
  execution: CheckpointExecution,
  gate: Opt0080FullThermalGate,
  trace: Opt0080FullThermalTrace,
  heartbeat: Opt0080FullHeartbeatCapture,
  timingReadyAtEpochMilliseconds: number,
): Opt0080FullTimedArmResult {
  if (
    execution.definition.armId === undefined ||
    execution.timingBase === undefined
  ) throw new Error("OPT-0080 full timed execution is incomplete");
  const checkpoint = execution.checkpoint;
  if (checkpoint.captureEvaluationTaps || checkpoint.evaluationTaps !== undefined) {
    throw new Error("OPT-0080 full timed arm captured evaluation taps");
  }
  return Object.freeze({
    armId: execution.definition.armId,
    schedulingProfile: execution.definition.schedulingProfile,
    captureEvaluationTaps: false,
    evaluationTapCount: 0,
    finalLatent: checkpoint.finalLatent,
    finalLatentRawU32: checkpoint.finalLatentRawU32,
    finalLatentSha256: checkpoint.finalLatentSha256,
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
    cancellation === undefined ||
    accepted.length !== OPT_0080_FULL_ARM_ORDER.length
  ) throw new Error("OPT-0080 full final inventory is incomplete");
  return buildOpt0080FullResult({
    runIdentity: identity,
    requestCanonicalJson: OPT_0018_CANONICAL_REQUEST_JSON,
    requestSha256: OPT_0018_CANONICAL_REQUEST_SHA256,
    requestByteLength: OPT_0018_CANONICAL_REQUEST_BYTES,
    mainManifestSha256: MAIN_MANIFEST_SHA256,
    denseManifestSha256: DENSE_MANIFEST_SHA256,
    denseRuntimeProfile: DENSE_RUNTIME_PROFILE,
    attentionRuntimeProfile:
      ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
    evaluationSliceResultSha256:
      OPT_0080_EVALUATION_SLICE_EVIDENCE_SHA256,
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
  ) throw new Error("OPT-0080 full authenticated runtime changed");
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
  ) throw new Error("OPT-0080 full conditioning authority changed");
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
  ) throw new Error("OPT-0080 full inherited request identity changed");
  const value = createOpt0018Request() as unknown as AceGenerationRequest;
  assertAceGenerationRequest(value);
  if (JSON.stringify(value) !== OPT_0018_CANONICAL_REQUEST_JSON) {
    throw new Error("OPT-0080 full inherited request values changed");
  }
  return Object.freeze(value);
}

function sameBufferViews(floats: Float32Array, words: Uint32Array): boolean {
  return floats.buffer === words.buffer && floats.byteOffset === words.byteOffset &&
    floats.byteLength === words.byteLength && floats.length === words.length;
}

function rawU32Equal(left: Uint32Array, right: Uint32Array): boolean {
  return left.length === OPT_0080_FULL_TENSOR_ELEMENTS &&
    right.length === OPT_0080_FULL_TENSOR_ELEMENTS &&
    left.every((word, index) => word === right[index]);
}

function bytesOf(values: Float32Array): Uint8Array {
  return new Uint8Array(values.buffer, values.byteOffset, values.byteLength);
}

function retainRejected(
  kind: "thermal-gate" | "thermal-trace",
  armId: Opt0080FullArmId,
  error: unknown,
  submittedEvidence: Readonly<Record<string, unknown>>,
): void {
  rejectedSetupAttempts.push(Object.freeze({
    kind,
    armId,
    rejectedAtEpochMilliseconds: Date.now(),
    error: serializeOpt0018Failure(error),
    submittedEvidence,
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
