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
  AceStageTiming,
} from "../../src/runtime/stages.js";
import {
  createAceWebGpuPipelineBackend,
  type AceOpt0034DitCheckpoint,
  type AceWebGpuGenerationContext,
} from "../../src/runtime/webgpu-pipeline.js";
import type { AceDitPhysicalQuantaPerCommandBuffer } from
  "../../src/webgpu/dit-graph.js";
import pipelineSource from "../../src/runtime/webgpu-pipeline.ts?raw";
import schedulerSource from "../../src/runtime/scheduler.ts?raw";
import ditBackendSource from "../../src/webgpu/dit-backend.ts?raw";
import ditGraphSource from "../../src/webgpu/dit-graph.ts?raw";
import workerSource from "./opt-0034-dit-command-buffer-coalescing-worker.ts?raw";
import pageSource from "./opt-0034-dit-command-buffer-coalescing.ts?raw";
import htmlSource from "./opt-0034-dit-command-buffer-coalescing.html?raw";
import thermalContractSource from
  "./opt-0034-dit-command-buffer-coalescing-contract.ts?raw";
import recordSource from
  "../../optimization/experiments/OPT-0034-dit-command-buffer-coalescing.md?raw";
import type { Opt0034ThermalGate } from
  "./opt-0034-dit-command-buffer-coalescing-contract.js";
import {
  OPT_0018_CANONICAL_REQUEST_BYTES,
  OPT_0018_CANONICAL_REQUEST_JSON,
  OPT_0018_CANONICAL_REQUEST_SHA256,
  createOpt0018Request,
  serializeOpt0018Failure,
  validateOpt0018RunIdentity,
  type Opt0018RunIdentity,
} from "./opt-0018-dit-m2250-production-family-profile.js";

const EXPERIMENT_ID = "OPT-0034" as const;
const MAIN_MANIFEST_SHA256 =
  "18f36c6420976475af65ecd833ca56c6119706322ce54120389d4915d8e80db6";
const DIT_MANIFEST_SHA256 =
  "d3fc0020efcf60702db411da2fd4b93e9bb84f1437ed310aef01c892727e452f";
const VAE_MANIFEST_SHA256 =
  "94a1ae61354f7481facbb9787d003488ab1bc351a137fd2bd7ff69dd99aef949";
const ACE_SOURCE_REVISION = "6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0";
const PLANNER_SOURCE_REVISION = "148d8ea0225bdab342ee1ae3a354275ccd60ca80";
const PARAKEET_REVISION = "7ee112738262a6f5a0efd2f150748a4087432fbb";
const DESCRIPTOR_TABLE_SHA256 =
  "aedf8c74d2bb15601d1385e9c8e9da49e58905ab156a04c99271f8de3633dd76";
const PHYSICAL_GRAPH_QUANTA = 2_553;
const PHYSICAL_PRIMITIVES = 6_833;
const SCHEDULED_MULTIPLY_ADDS = 26_840_955_355_136;
const FINAL_LATENT_ELEMENTS = 288_000;
const FINAL_LATENT_BYTES = 1_152_000;
const REQUIRED_COMPLETE_STAGE_SPEEDUP = 1.10;
const MAXIMUM_LAUNCH_DELAY_MILLISECONDS = 30_000;
const ARMS = Object.freeze([1, 8, 16] as const);
const SCHEDULING_DIAGNOSTIC_CODE =
  "ACE_DIT_OPT0034_SCHEDULING_PROFILE" as const;
const PRIVATE_DIT_STOP = Object.freeze({ opt0034PrivateDitStop: true });
const SOURCE_TEXT = Object.freeze({
  "optimization/experiments/OPT-0034-dit-command-buffer-coalescing.md":
    recordSource,
  "src/runtime/scheduler.ts": schedulerSource,
  "src/runtime/webgpu-pipeline.ts": pipelineSource,
  "src/webgpu/dit-backend.ts": ditBackendSource,
  "src/webgpu/dit-graph.ts": ditGraphSource,
  "test/browser/opt-0034-dit-command-buffer-coalescing-worker.ts":
    workerSource,
  "test/browser/opt-0034-dit-command-buffer-coalescing.ts": pageSource,
  "test/browser/opt-0034-dit-command-buffer-coalescing.html": htmlSource,
  "test/browser/opt-0034-dit-command-buffer-coalescing-contract.ts":
    thermalContractSource,
});

interface InitializeMessage {
  readonly type: "initialize";
  readonly identity: Opt0018RunIdentity;
}

interface RunMessage {
  readonly type: "run";
  readonly thermalGate: Opt0034ThermalGate;
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

interface SchedulingDiagnostic {
  readonly elapsedMs: number;
  readonly stage: string | null;
  readonly details: Readonly<Record<string, unknown>>;
}

interface InternalArmResult {
  readonly batch: AceDitPhysicalQuantaPerCommandBuffer;
  readonly finalLatent: Float32Array<ArrayBuffer>;
  readonly receipt: Readonly<Record<string, unknown>> & Readonly<{
    timing: Readonly<{
      generationToCheckpointWallMs: number;
      ditLoadThroughReleaseWallMs: number;
      ditDenoiseStageWallMs: number;
      graphWallMs: number;
      backendWallMs: number;
    }>;
    correctness: Readonly<{ finalLatentSha256: string }>;
  }>;
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
    void runComparison(active, event.data.thermalGate).then(
      (result) => {
        lifecycle = "settled";
        self.postMessage({ type: "comparison-complete", result });
      },
      (error: unknown) => void failAndCleanup(error, active),
    );
  }
});

async function initializeSession(
  identityValue: Opt0018RunIdentity,
): Promise<PreparedSession> {
  const identity = validateOpt0018RunIdentity(identityValue);
  validateCanonicalRequest();
  const backend = createAceWebGpuPipelineBackend();
  const controller = new AbortController();
  let initializationProgressEventCount = 0;
  let initializationDiagnosticCount = 0;
  const initializationStartedAtEpochMilliseconds = Date.now();
  const startedAt = performance.now();
  postProgress("authenticating production manifests, cache, and WebGPU device");
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
    return Object.freeze({
      backend,
      identity,
      diagnostics,
      sourceAuthority: buildSourceAuthority(),
      initializationStartedAtEpochMilliseconds,
      initializationCompletedAtEpochMilliseconds: Date.now(),
      initializationWallMilliseconds: performance.now() - startedAt,
      initializationProgressEventCount,
      initializationDiagnosticCount,
    });
  } catch (error) {
    await backend.dispose().catch(() => undefined);
    throw error;
  }
}

async function runComparison(
  session: PreparedSession,
  thermalGate: Opt0034ThermalGate,
): Promise<Readonly<Record<string, unknown>>> {
  validateThermalGate(
    thermalGate,
    session.initializationCompletedAtEpochMilliseconds,
  );
  const launchedAtEpochMilliseconds = Date.now();
  const launchDelayMilliseconds =
    launchedAtEpochMilliseconds - thermalGate.checkedAtEpochMilliseconds;
  if (
    launchDelayMilliseconds < 0 ||
    launchDelayMilliseconds > MAXIMUM_LAUNCH_DELAY_MILLISECONDS
  ) {
    throw new Error("OPT-0034 launch missed the thermal handoff");
  }
  const request = validateCanonicalRequest();
  const arms: InternalArmResult[] = [];
  const comparisonStartedAtEpochMilliseconds = Date.now();
  const comparisonStartedAt = performance.now();
  let backendDisposeStartedAtEpochMilliseconds = 0;
  let backendDisposeCompletedAtEpochMilliseconds = 0;
  try {
    for (const batch of ARMS) {
      postProgress(
        `running batch${batch}: ${Math.ceil(PHYSICAL_GRAPH_QUANTA / batch)} graph command buffers`,
      );
      arms.push(await runArm(session, request, batch));
      await yieldToWorker();
    }
    const exact = compareFinalLatents(arms);
    const baseline = arms[0]!;
    const candidates = arms.slice(1);
    const fastest = [...candidates].sort((left, right) =>
      left.receipt.timing.generationToCheckpointWallMs -
        right.receipt.timing.generationToCheckpointWallMs
    )[0]!;
    const speedups = Object.freeze(Object.fromEntries(candidates.map((arm) => [
      `batch${arm.batch}`,
      Object.freeze({
        generationToCheckpoint:
          baseline.receipt.timing.generationToCheckpointWallMs /
          arm.receipt.timing.generationToCheckpointWallMs,
        ditLoadThroughRelease:
          baseline.receipt.timing.ditLoadThroughReleaseWallMs /
          arm.receipt.timing.ditLoadThroughReleaseWallMs,
        ditDenoiseStage:
          baseline.receipt.timing.ditDenoiseStageWallMs /
          arm.receipt.timing.ditDenoiseStageWallMs,
        graphWall:
          baseline.receipt.timing.graphWallMs /
          arm.receipt.timing.graphWallMs,
        backendWall:
          baseline.receipt.timing.backendWallMs /
          arm.receipt.timing.backendWallMs,
      }),
    ])));
    const fastestCompleteStageSpeedup =
      baseline.receipt.timing.generationToCheckpointWallMs /
      fastest.receipt.timing.generationToCheckpointWallMs;
    const performanceGatePassed =
      exact.passed &&
      fastestCompleteStageSpeedup >= REQUIRED_COMPLETE_STAGE_SPEEDUP;
    backendDisposeStartedAtEpochMilliseconds = Date.now();
    await session.backend.dispose();
    backendDisposeCompletedAtEpochMilliseconds = Date.now();
    return Object.freeze({
      schema: "ace-opt-0034-dit-command-buffer-coalescing-v1",
      experimentId: EXPERIMENT_ID,
      status: performanceGatePassed ? "positive" : "negative",
      identity: Object.freeze({
        run: session.identity,
        sourceAuthority: session.sourceAuthority,
        packageAuthority: Object.freeze({
          aceSourceRevision: ACE_SOURCE_REVISION,
          plannerSourceRevision: PLANNER_SOURCE_REVISION,
          parakeetReferenceRevision: PARAKEET_REVISION,
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
          directDcw: resolveAceDynamicConditionalWeighting(request.planner),
        }),
        runtimeDiagnostics: session.diagnostics,
      }),
      protocol: Object.freeze({
        thermalGate,
        launchDelayMilliseconds,
        armOrder: ARMS,
        authoritativeRunCount: 1,
        timingSampleCountPerArm: 1,
        unchangedThermalRetryPerformed: false,
        queueDepth: 1,
        drainAfterEveryBatch: true,
        cooperativeIntervalMilliseconds: 1,
        finalReadbackSeparate: true,
        stockWebGpuOnly: true,
      }),
      arms: Object.freeze(arms.map((arm) => arm.receipt)),
      correctness: exact,
      delta: Object.freeze({
        baselineBatch: 1,
        candidateBatches: Object.freeze([8, 16]),
        speedups,
        fastestCandidateBatch: fastest.batch,
        fastestCompleteStageSpeedup,
        requiredCompleteStageSpeedup: REQUIRED_COMPLETE_STAGE_SPEEDUP,
        performanceGatePassed,
      }),
      lifecycle: Object.freeze({
        initializationStartedAtEpochMilliseconds:
          session.initializationStartedAtEpochMilliseconds,
        initializationCompletedAtEpochMilliseconds:
          session.initializationCompletedAtEpochMilliseconds,
        initializationWallMilliseconds: session.initializationWallMilliseconds,
        initializationProgressEventCount:
          session.initializationProgressEventCount,
        initializationDiagnosticCount:
          session.initializationDiagnosticCount,
        comparisonStartedAtEpochMilliseconds,
        comparisonCompletedAtEpochMilliseconds: Date.now(),
        comparisonWallMilliseconds: performance.now() - comparisonStartedAt,
        backendDisposeStartedAtEpochMilliseconds,
        backendDisposeCompletedAtEpochMilliseconds,
        backendDisposed: true,
        armCleanupAwaitedBeforeNextArm: true,
      }),
      scope: Object.freeze({
        productionMathChanged: false,
        productionSelectorChanged: false,
        productionDefaultChanged: false,
        vaeWeightAcquireStarted: false,
        vaeExecuted: false,
        audioExecuted: false,
        optimizationIntegrated: false,
        under60SecondClaim: false,
      }),
      decision: performanceGatePassed
        ? "positive-exact-scheduling-candidate"
        : exact.passed
        ? "negative-below-complete-stage-speed-gate"
        : "negative-final-latent-mismatch",
    });
  } catch (error) {
    if (backendDisposeCompletedAtEpochMilliseconds === 0) {
      await session.backend.dispose().catch(() => undefined);
    }
    throw error;
  }
}

async function runArm(
  session: PreparedSession,
  request: AceGenerationRequest,
  batch: AceDitPhysicalQuantaPerCommandBuffer,
): Promise<InternalArmResult> {
  const controller = new AbortController();
  let checkpoint: AceOpt0034DitCheckpoint | undefined;
  let schedulingDiagnostic: SchedulingDiagnostic | undefined;
  let checkpointCallbackCount = 0;
  let generationProgressEventCount = 0;
  let forbiddenPostDitProgressObserved = false;
  const diagnosticCodes: string[] = [];
  const fatalDiagnostics: Readonly<Record<string, unknown>>[] = [];
  const generationStartedAtEpochMilliseconds = Date.now();
  const generationStartedAt = performance.now();
  let generationRejectedAtEpochMilliseconds = 0;
  const context: AceWebGpuGenerationContext = {
    signal: controller.signal,
    captureTrace: true,
    onProgress: (progress: AceGenerationProgress) => {
      generationProgressEventCount += 1;
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
      if (diagnostic.code !== SCHEDULING_DIAGNOSTIC_CODE) return;
      if (
        schedulingDiagnostic !== undefined ||
        diagnostic.details === undefined
      ) {
        throw new Error(`OPT-0034 batch${batch} scheduling diagnostic repeated`);
      }
      schedulingDiagnostic = Object.freeze({
        elapsedMs: diagnostic.elapsedMs,
        stage: diagnostic.stage ?? null,
        details: diagnostic.details,
      });
    },
    opt0034DitRun: Object.freeze({
      physicalQuantaPerCommandBuffer: batch,
      onCheckpoint: (value: AceOpt0034DitCheckpoint) => {
        checkpointCallbackCount += 1;
        if (checkpoint !== undefined || checkpointCallbackCount !== 1) {
          throw new Error(`OPT-0034 batch${batch} checkpoint repeated`);
        }
        checkpoint = value;
        controller.abort(PRIVATE_DIT_STOP);
      },
    }),
  };
  let privateSentinelIdentityMatched = false;
  try {
    await session.backend.generate(request, context);
    throw new Error(`OPT-0034 batch${batch} continued into VAE`);
  } catch (error) {
    generationRejectedAtEpochMilliseconds = Date.now();
    privateSentinelIdentityMatched = error === PRIVATE_DIT_STOP;
    if (!privateSentinelIdentityMatched) throw error;
  }
  if (
    checkpoint === undefined ||
    schedulingDiagnostic === undefined ||
    checkpointCallbackCount !== 1 ||
    forbiddenPostDitProgressObserved ||
    fatalDiagnostics.length !== 0
  ) {
    throw new Error(`OPT-0034 batch${batch} checkpoint boundary failed`);
  }
  const finalLatent = checkpoint.finalLatent;
  const profile = checkpoint.profile;
  validateCheckpoint(checkpoint, schedulingDiagnostic, batch);
  const ditLoad = requireStageTiming(checkpoint.stageTimings, "dit-load");
  const ditDenoise = requireStageTiming(
    checkpoint.stageTimings,
    "dit-denoise",
  );
  const releaseDit = requireStageTiming(checkpoint.stageTimings, "release-dit");
  const generationToCheckpointWallMs = performance.now() - generationStartedAt;
  const receipt = Object.freeze({
    schema: "ace-opt-0034-arm-v1",
    batch,
    topology: Object.freeze({
      physicalGraphQuantumCount: profile.physicalGraphQuantumCount,
      graphCommandBufferCount: profile.graphCommandBufferCount,
      readbackCommandBufferCount: profile.readbackCommandBufferCount,
      totalCommandBufferCount: profile.totalCommandBufferCount,
      graphQueueDrainCount: profile.graphQueueDrainCount,
      totalQueueDrainCount: profile.totalQueueDrainCount,
      graphRequestedIdleMs: profile.graphRequestedIdleMs,
      graphToReadbackRequestedIdleMs:
        profile.graphToReadbackRequestedIdleMs,
      descriptorTableSha256: profile.descriptorTableSha256,
      physicalPrimitiveCount: profile.physicalPrimitiveCount,
      scheduledMultiplyAdds: profile.scheduledMultiplyAdds,
      batchCount: profile.batches.length,
      firstBatch: profile.batches[0],
      finalBatch: profile.batches.at(-1),
      maximumPhysicalQuantaPerBatch:
        profile.maximumPhysicalQuantaPerBatch,
      maximumPrimitiveCountPerBatch: profile.maximumPrimitiveCountPerBatch,
      maximumScheduledMultiplyAddsPerBatch:
        profile.maximumScheduledMultiplyAddsPerBatch,
    }),
    timing: Object.freeze({
      generationToCheckpointWallMs,
      ditLoadStageWallMs: ditLoad.wallMs,
      ditDenoiseStageWallMs: ditDenoise.wallMs,
      releaseDitStageWallMs: releaseDit.wallMs,
      ditLoadThroughReleaseWallMs:
        ditLoad.wallMs + ditDenoise.wallMs + releaseDit.wallMs,
      graphSubmitThroughDrainMs: profile.graphSubmitThroughDrainMs,
      graphWallMs: profile.graphWallMs,
      graphToReadbackObservedIdleMs:
        profile.graphToReadbackObservedIdleMs,
      readbackSubmitThroughDrainMs: profile.readbackSubmitThroughDrainMs,
      readbackMapDetachMs: profile.readbackMapDetachMs,
      backendWallMs: profile.backendWallMs,
      maximumBatchSubmitThroughDrainMs:
        profile.maximumBatchSubmitThroughDrainMs,
    }),
    correctness: Object.freeze({
      finalLatentElementCount: checkpoint.finalLatentElementCount,
      finalLatentByteLength: checkpoint.finalLatentByteLength,
      finalLatentSha256: checkpoint.finalLatentSha256,
      finalLatentNonFiniteCount: checkpoint.finalLatentNonFiniteCount,
      finalLatentNonzeroCount: checkpoint.finalLatentNonzeroCount,
      finalLatentMaxAbs: checkpoint.finalLatentMaxAbs,
    }),
    diagnostic: Object.freeze({
      code: SCHEDULING_DIAGNOSTIC_CODE,
      stage: schedulingDiagnostic.stage,
      elapsedMs: schedulingDiagnostic.elapsedMs,
      detailsSha256: hashText(JSON.stringify(schedulingDiagnostic.details)),
      matchedCheckpoint: true,
    }),
    lifecycle: Object.freeze({
      generationStartedAtEpochMilliseconds,
      generationRejectedAtEpochMilliseconds,
      generationProgressEventCount,
      checkpointCallbackCount,
      privateSentinelIdentityMatched,
      ditDestroyedBeforeCheckpoint: true,
      pipelineCleanupAwaitedBeforeArmReturn: true,
      forbiddenPostDitProgressObserved,
    }),
    stageTimings: checkpoint.stageTimings,
  });
  return Object.freeze({ batch, finalLatent, receipt });
}

function validateCheckpoint(
  checkpoint: AceOpt0034DitCheckpoint,
  diagnostic: SchedulingDiagnostic,
  batch: AceDitPhysicalQuantaPerCommandBuffer,
): void {
  const profile = checkpoint.profile;
  const expectedGraphCommandBuffers = Math.ceil(
    PHYSICAL_GRAPH_QUANTA / batch,
  );
  const expectedTail = PHYSICAL_GRAPH_QUANTA % batch || batch;
  if (
    checkpoint.schema !== "ace-dit-opt0034-m2250-checkpoint-v1" ||
    !(checkpoint.finalLatent instanceof Float32Array) ||
    checkpoint.finalLatent.length !== FINAL_LATENT_ELEMENTS ||
    checkpoint.finalLatent.byteLength !== FINAL_LATENT_BYTES ||
    checkpoint.finalLatentElementCount !== FINAL_LATENT_ELEMENTS ||
    checkpoint.finalLatentByteLength !== FINAL_LATENT_BYTES ||
    checkpoint.finalLatentNonFiniteCount !== 0 ||
    checkpoint.finalLatentNonzeroCount <= 0 ||
    !(checkpoint.finalLatentMaxAbs > 0) ||
    checkpoint.finalLatentSha256 !== aceSha256Hex(new Uint8Array(
      checkpoint.finalLatent.buffer,
      checkpoint.finalLatent.byteOffset,
      checkpoint.finalLatent.byteLength,
    )) ||
    profile.schema !== "ace-dit-opt0034-command-buffer-coalescing-v1" ||
    profile.physicalGraphQuantumCount !== PHYSICAL_GRAPH_QUANTA ||
    profile.physicalQuantaPerCommandBuffer !== batch ||
    profile.graphCommandBufferCount !== expectedGraphCommandBuffers ||
    profile.readbackCommandBufferCount !== 1 ||
    profile.totalCommandBufferCount !== expectedGraphCommandBuffers + 1 ||
    profile.graphQueueDrainCount !== expectedGraphCommandBuffers ||
    profile.totalQueueDrainCount !== expectedGraphCommandBuffers + 1 ||
    profile.graphRequestedIdleMs !== expectedGraphCommandBuffers - 1 ||
    profile.graphToReadbackRequestedIdleMs !== 1 ||
    profile.maximumPhysicalQuantaPerBatch !== batch ||
    profile.descriptorTableSha256 !== DESCRIPTOR_TABLE_SHA256 ||
    profile.physicalPrimitiveCount !== PHYSICAL_PRIMITIVES ||
    profile.scheduledMultiplyAdds !== SCHEDULED_MULTIPLY_ADDS ||
    profile.batches.length !== expectedGraphCommandBuffers ||
    profile.batches[0]?.firstPhysicalIndex !== 0 ||
    profile.batches.at(-1)?.lastPhysicalIndex !==
      PHYSICAL_GRAPH_QUANTA - 1 ||
    profile.batches.at(-1)?.physicalQuantumCount !== expectedTail ||
    profile.batches.reduce(
      (total, value) => total + value.physicalQuantumCount,
      0,
    ) !== PHYSICAL_GRAPH_QUANTA ||
    profile.batches.reduce(
      (total, value) => total + value.primitiveCount,
      0,
    ) !== PHYSICAL_PRIMITIVES ||
    profile.batches.reduce(
      (total, value) => total + value.scheduledMultiplyAdds,
      0,
    ) !== SCHEDULED_MULTIPLY_ADDS ||
    profile.batches.some((value, index) =>
      value.batchIndex !== index ||
      value.firstPhysicalIndex !== (index === 0
        ? 0
        : profile.batches[index - 1]!.lastPhysicalIndex + 1) ||
      value.lastPhysicalIndex - value.firstPhysicalIndex + 1 !==
        value.physicalQuantumCount ||
      !finiteNonnegative(value.submitThroughDrainMs)
    ) ||
    !finiteNonnegative(profile.graphSubmitThroughDrainMs) ||
    !finiteNonnegative(profile.graphWallMs) ||
    !finiteNonnegative(profile.backendWallMs) ||
    !finiteNonnegative(profile.maximumBatchSubmitThroughDrainMs)
  ) throw new Error(`OPT-0034 batch${batch} checkpoint topology changed`);
  const details = diagnostic.details;
  if (
    diagnostic.stage !== "release-dit" ||
    details.schema !== "ace-dit-opt0034-scheduling-receipt-v1" ||
    details.checkpointSchema !== checkpoint.schema ||
    details.schedulingProfileSchema !== profile.schema ||
    details.physicalGraphQuantumCount !== PHYSICAL_GRAPH_QUANTA ||
    details.physicalQuantaPerCommandBuffer !== batch ||
    details.graphCommandBufferCount !== expectedGraphCommandBuffers ||
    details.totalQueueDrainCount !== expectedGraphCommandBuffers + 1 ||
    details.descriptorTableSha256 !== DESCRIPTOR_TABLE_SHA256 ||
    details.finalLatentSha256 !== checkpoint.finalLatentSha256
  ) throw new Error(`OPT-0034 batch${batch} diagnostic diverged`);
}

function compareFinalLatents(
  arms: readonly InternalArmResult[],
): Readonly<Record<string, unknown>> {
  if (arms.length !== ARMS.length) {
    throw new Error("OPT-0034 did not produce all three arms");
  }
  const baseline = asU32(arms[0]!.finalLatent);
  const comparisons = arms.slice(1).map((arm) => {
    const candidate = asU32(arm.finalLatent);
    let mismatchCount = 0;
    let firstMismatch: Readonly<Record<string, unknown>> | null = null;
    for (let index = 0; index < baseline.length; index += 1) {
      if (baseline[index] === candidate[index]) continue;
      mismatchCount += 1;
      if (firstMismatch === null) {
        firstMismatch = Object.freeze({
          index,
          baselineU32: baseline[index],
          candidateU32: candidate[index],
        });
      }
    }
    return Object.freeze({
      candidateBatch: arm.batch,
      comparedRawU32: baseline.length,
      mismatchCount,
      firstMismatch,
      baselineSha256: arms[0]!.receipt.correctness.finalLatentSha256,
      candidateSha256: arm.receipt.correctness.finalLatentSha256,
      hashIdentical:
        arms[0]!.receipt.correctness.finalLatentSha256 ===
        arm.receipt.correctness.finalLatentSha256,
    });
  });
  const passed = comparisons.every((value) =>
    value.mismatchCount === 0 && value.hashIdentical
  );
  return Object.freeze({
    passed,
    comparisonKind: "complete-detached-final-latent-raw-u32",
    finalLatentElementsPerArm: FINAL_LATENT_ELEMENTS,
    totalComparedRawU32: comparisons.length * FINAL_LATENT_ELEMENTS,
    comparisons: Object.freeze(comparisons),
    executionOrderAndArithmeticChanged: false,
    listeningRequired: false,
  });
}

function asU32(values: Float32Array<ArrayBuffer>): Uint32Array<ArrayBuffer> {
  return new Uint32Array(
    values.buffer,
    values.byteOffset,
    values.length,
  );
}

function requireStageTiming(
  timings: readonly AceStageTiming[],
  stage: AceStageTiming["stage"],
): AceStageTiming {
  const matches = timings.filter((timing) => timing.stage === stage);
  if (
    matches.length !== 1 ||
    !finiteNonnegative(matches[0]!.wallMs)
  ) throw new Error(`OPT-0034 missing ${stage} stage timing`);
  return matches[0]!;
}

function validateCanonicalRequest(): AceGenerationRequest {
  const encoded = new TextEncoder().encode(OPT_0018_CANONICAL_REQUEST_JSON);
  if (
    encoded.byteLength !== OPT_0018_CANONICAL_REQUEST_BYTES ||
    aceSha256Hex(encoded) !== OPT_0018_CANONICAL_REQUEST_SHA256
  ) throw new Error("OPT-0034 canonical request authority changed");
  const request = createOpt0018Request() as unknown as AceGenerationRequest;
  assertAceGenerationRequest(request);
  if (JSON.stringify(request) !== OPT_0018_CANONICAL_REQUEST_JSON) {
    throw new Error("OPT-0034 request property order or values changed");
  }
  const dcw = resolveAceDynamicConditionalWeighting(request.planner);
  if (
    dcw.mode !== "double" ||
    dcw.wavelet !== "haar" ||
    dcw.lowBandScale !== 0.05 ||
    dcw.highBandScale !== 0.02
  ) throw new Error("OPT-0034 direct DCW contract changed");
  return Object.freeze(request);
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
  ) throw new Error("OPT-0034 runtime diagnostics escaped frozen production");
}

function productionConfiguration() {
  return Object.freeze({
    manifestUrl: new URL(
      "/model/files-reference/manifest.json",
      self.location.href,
    ).href,
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

function validateThermalGate(
  gate: Opt0034ThermalGate,
  readyAtEpochMilliseconds: number,
): void {
  if (
    gate.source !== "notifyutil-com.apple.system.thermalpressurelevel" ||
    gate.command !== "notifyutil -g com.apple.system.thermalpressurelevel" ||
    gate.protocol !== "wait-30s-then-one-level0-check" ||
    gate.startedAtEpochMilliseconds < readyAtEpochMilliseconds ||
    gate.durationMilliseconds < 30_000 ||
    gate.checkedAtEpochMilliseconds - gate.startedAtEpochMilliseconds !==
      gate.durationMilliseconds ||
    gate.observationCount !== 1 ||
    gate.observedLevel !== 0 ||
    gate.maximumObservationGapMilliseconds !== gate.durationMilliseconds
  ) throw new Error("OPT-0034 rejected the thermal gate");
}

function publicPreparationSummary(
  prepared: PreparedSession,
): Readonly<Record<string, unknown>> {
  return Object.freeze({
    experimentId: EXPERIMENT_ID,
    initializationWallMilliseconds: prepared.initializationWallMilliseconds,
    initializationProgressEventCount:
      prepared.initializationProgressEventCount,
    sourceFileCount: Object.keys(prepared.sourceAuthority).length,
    requestSha256: OPT_0018_CANONICAL_REQUEST_SHA256,
    modelManifestSha256: prepared.diagnostics.modelManifestSha256,
    ditDenseManifestSha256: prepared.diagnostics.ditDenseManifestSha256,
    vaeManifestAuthenticatedButWeightsNotAcquired:
      prepared.diagnostics.vaeManifestSha256 === VAE_MANIFEST_SHA256,
    physicalGraphQuanta: PHYSICAL_GRAPH_QUANTA,
    armOrder: ARMS,
    plannedGraphCommandBuffers: Object.freeze({
      batch1: 2_553,
      batch8: 320,
      batch16: 160,
    }),
  });
}

function buildSourceAuthority(): Readonly<Record<string, string>> {
  return Object.freeze(Object.fromEntries(
    Object.entries(SOURCE_TEXT).map(([path, source]) => [path, hashText(source)]),
  ));
}

function hashText(value: string): string {
  return aceSha256Hex(new TextEncoder().encode(value));
}

function finiteNonnegative(value: number): boolean {
  return Number.isFinite(value) && value >= 0;
}

async function yieldToWorker(): Promise<void> {
  await new Promise<void>((resolve) => setTimeout(resolve, 0));
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
