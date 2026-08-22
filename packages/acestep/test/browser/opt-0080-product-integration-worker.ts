/// <reference lib="webworker" />
/// <reference types="@webgpu/types" />
/// <reference types="vite/client" />

import type { AceGenerationResult } from "../../src/api.js";
import type { AceDiagnostic, AceRuntimeDiagnostics } from
  "../../src/runtime/diagnostics.js";
import type { AceWorkerConfiguration } from "../../src/runtime/protocol.js";
import type { AceGenerationProgress, AceInitializationProgress } from
  "../../src/runtime/stages.js";
import {
  createAceWebGpuPipelineBackend,
  type AceOpt0080ProductEvidence,
} from "../../src/runtime/webgpu-pipeline.js";
import { AceIncrementalSha256 } from "../../src/model/sha256.js";
import { planAceVaeChunkedDecode } from "../../src/webgpu/vae-chunks.js";
import {
  serializeOpt0018Failure,
  validateOpt0018RunIdentity,
  type Opt0018RunIdentity,
} from "./opt-0018-dit-m2250-production-family-profile.js";
import {
  OPT_0080_PRODUCT_ARM_ORDER,
  OPT_0080_PRODUCT_AUDIO_FRAMES,
  OPT_0080_PRODUCT_DENSE_MANIFEST_SHA256,
  OPT_0080_PRODUCT_LATENT_FRAMES,
  OPT_0080_PRODUCT_MAIN_MANIFEST_SHA256,
  OPT_0080_PRODUCT_RAW_BYTES,
  OPT_0080_PRODUCT_REQUEST,
  OPT_0080_PRODUCT_REQUEST_BYTES,
  OPT_0080_PRODUCT_REQUEST_SHA256,
  OPT_0080_PRODUCT_SEAM_RADIUS_LATENT_FRAMES,
  OPT_0080_PRODUCT_STITCH_SEAM_AUDIO_FRAME,
  OPT_0080_PRODUCT_STITCH_SEAM_LATENT_FRAME,
  OPT_0080_PRODUCT_VAE_MANIFEST_SHA256,
  OPT_0080_PRODUCT_WAV_BYTES,
  OPT_0080_PRODUCT_WINDOW_COUNT,
  type Opt0080ProductArmId,
} from "./opt-0080-product-integration-contract.js";
import {
  OPT_0080_VAE_PRODUCT_ARM_ORDER,
  OPT_0080_VAE_PRODUCT_FINAL_LATENT_SHA256,
  OPT_0080_VAE_PRODUCT_GATE_KIND,
  OPT_0080_VAE_PRODUCT_RAW_SHA256,
  OPT_0080_VAE_PRODUCT_SEAM_SHA256,
  OPT_0080_VAE_PRODUCT_WAV_SHA256,
  requireOpt0080VaeProductSchedulingReceipt,
  summarizeOpt0080VaeProductSchedulingEvidence,
} from "./opt-0080-vae-product-selector-contract.js";

type RunKind = Opt0080ProductArmId | "cancellation";
type ProductGateKind = "dit" | typeof OPT_0080_VAE_PRODUCT_GATE_KIND;

interface RunMessage {
  readonly type: "run";
  readonly runKind: RunKind;
  readonly identity: Opt0018RunIdentity;
  readonly productGate?: ProductGateKind;
}

interface ReleaseMessage {
  readonly type: "release";
  readonly armId: Opt0080ProductArmId;
}

type WorkerPhase = "awaiting-run" | "running" | "awaiting-release" |
  "releasing" | "terminal" | "failed";

interface ReleaseWaiter {
  readonly armId: Opt0080ProductArmId;
  settled: boolean;
  readonly resolve: () => void;
  readonly reject: (error: Error) => void;
}

let workerPhase: WorkerPhase = "awaiting-run";
let activeAbortController: AbortController | undefined;
let releaseWaiter: ReleaseWaiter | undefined;
let protocolFailure: Error | undefined;
let failurePosted = false;
let runInFlight = false;

self.addEventListener("message", (event: MessageEvent<unknown>) => {
  const message = event.data;
  if (workerPhase === "awaiting-run") {
    if (!isRunMessage(message)) {
      rejectProtocol("OPT-0080 product worker expected one initial run command");
      return;
    }
    workerPhase = "running";
    runInFlight = true;
    void run(message).then(() => {
      if (workerPhase !== "failed") workerPhase = "terminal";
    }).catch(fail).finally(() => {
      runInFlight = false;
    });
    return;
  }
  if (workerPhase === "awaiting-release") {
    if (!isReleaseMessage(message)) {
      rejectProtocol("OPT-0080 product worker expected a release command");
      return;
    }
    const waiter = releaseWaiter;
    if (
      waiter === undefined || waiter.settled || message.armId !== waiter.armId
    ) {
      rejectProtocol("OPT-0080 product worker received the wrong release command");
      return;
    }
    waiter.settled = true;
    workerPhase = "releasing";
    waiter.resolve();
    return;
  }
  rejectProtocol(
    `OPT-0080 product worker rejected a command while ${workerPhase}`,
  );
});

async function run(message: RunMessage): Promise<void> {
  const identity = validateOpt0018RunIdentity(message.identity);
  if (message.runKind === "cancellation") {
    await runCancellation(identity);
    return;
  }
  if (message.productGate === OPT_0080_VAE_PRODUCT_GATE_KIND) {
    await runVaeProductArm(message.runKind, identity);
    return;
  }
  const arm = OPT_0080_PRODUCT_ARM_ORDER.find(
    ({ id }) => id === message.runKind,
  );
  if (arm === undefined) throw new Error("OPT-0080 product arm changed");
  const backend = createAceWebGpuPipelineBackend();
  const controller = new AbortController();
  activeAbortController = controller;
  const diagnostics: AceDiagnostic[] = [];
  const initializationProgress: AceInitializationProgress[] = [];
  const generationProgress: AceGenerationProgress[] = [];
  const unhandled: unknown[] = [];
  const unhandledListener = (event: PromiseRejectionEvent): void => {
    event.preventDefault();
    unhandled.push(event.reason);
  };
  self.addEventListener("unhandledrejection", unhandledListener);
  let result: AceGenerationResult | undefined;
  let evidence: AceOpt0080ProductEvidence | undefined;
  let releaseCompletedAt = 0;
  let disposeCompletedAt = 0;
  let releaseCompletedAtPerformance = 0;
  let disposeCompletedAtPerformance = 0;
  try {
    postProgress(`${arm.id}: cache-only initialization on a fresh device`);
    const runtime = await backend.initialize(productionConfiguration(), {
      modelSource: "cache-only",
      signal: controller.signal,
      onProgress: (value) => initializationProgress.push(value),
      onDiagnostic: (value) => diagnostics.push(value),
    });
    validateRuntime(runtime);
    postProgress(`${arm.id}: 96-second exact direct product execution`);
    result = await backend.generate(OPT_0080_PRODUCT_REQUEST, {
      signal: controller.signal,
      onProgress: (value) => generationProgress.push(value),
      onDiagnostic: (value) => diagnostics.push(value),
      opt0080ProductRun: {
        ...(arm.submissionPolicyOverride === undefined
          ? {}
          : { submissionPolicyOverride: arm.submissionPolicyOverride }),
        onEvidence(value) {
          if (evidence !== undefined) {
            throw new Error("OPT-0080 product evidence emitted twice");
          }
          evidence = value;
        },
      },
    });
    const captured = requireEvidence(arm.id, arm.effectiveSubmissionPolicy,
      result, evidence);
    validateProgress(initializationProgress, generationProgress);
    validateDiagnostics(diagnostics);
    const [requestIdentity, finalLatent, raw, wav, wavScan, seam] =
      await Promise.all([
        hashText(JSON.stringify(OPT_0080_PRODUCT_REQUEST)),
        hashBytes(new Uint8Array(
          captured.finalLatentRawU32.buffer,
          captured.finalLatentRawU32.byteOffset,
          captured.finalLatentRawU32.byteLength,
        )),
        hashBlob(captured.rawSnapshot),
        hashBlob(result.audio),
        scanWav(result.audio),
        inspectSeam(captured.rawSnapshot),
      ]);
    if (
      requestIdentity.byteLength !== OPT_0080_PRODUCT_REQUEST_BYTES ||
      requestIdentity.sha256 !== OPT_0080_PRODUCT_REQUEST_SHA256
    ) throw new Error("OPT-0080 canonical request identity changed");
    const summary = Object.freeze({
      schema: "ace-opt-0080-product-arm-v1",
      armId: arm.id,
      identity,
      request: Object.freeze({
        canonicalJson: JSON.stringify(OPT_0080_PRODUCT_REQUEST),
        ...requestIdentity,
      }),
      selectedProductionPolicy: captured.selectedProductionPolicy,
      effectiveSubmissionPolicy: captured.effectiveSubmissionPolicy,
      finalLatent: Object.freeze({
        elementCount: captured.finalLatentRawU32.length,
        byteLength: captured.finalLatentRawU32.byteLength,
        sha256: finalLatent.sha256,
        scan: scanFloat32(captured.finalLatent),
      }),
      raw: Object.freeze({
        ...raw,
        stats: captured.rawStats,
      }),
      seam,
      wav: Object.freeze({ ...wav, scan: wavScan }),
      vaePlan: summarizePlan(captured),
      metadata: summarizeResult(result),
      metrics: result.metrics,
      runtime,
      progress: Object.freeze({
        initializationStages: uniqueStages(initializationProgress),
        generationStages: uniqueStages(generationProgress),
        initializationEventCount: initializationProgress.length,
        generationEventCount: generationProgress.length,
      }),
      diagnosticCodes: Object.freeze(diagnostics.map(({ code }) => code)),
    });
    throwIfProtocolFailed();
    const released = waitForRelease(arm.id);
    const finalWords = captured.finalLatentRawU32.slice();
    self.postMessage({
      type: "arm-ready",
      armId: arm.id,
      summary,
      finalLatentRawU32: finalWords,
      rawSnapshot: captured.rawSnapshot,
      wav: captured.result.audio,
    }, [finalWords.buffer]);
    await released;
    throwIfProtocolFailed();
    postProgress(`${arm.id}: comparison complete; releasing retained output`);
    await backend.releaseResult(result);
    releaseCompletedAt = Date.now();
    releaseCompletedAtPerformance = performance.now();
    result = undefined;
    await backend.dispose();
    disposeCompletedAt = Date.now();
    disposeCompletedAtPerformance = performance.now();
    await nextTask();
    throwIfProtocolFailed();
    if (unhandled.length !== 0) {
      throw new Error("OPT-0080 product arm observed an unhandled rejection");
    }
    self.removeEventListener("unhandledrejection", unhandledListener);
    self.postMessage({
      type: "arm-released",
      armId: arm.id,
      lifecycle: Object.freeze({
        releaseCompletedAtEpochMilliseconds: releaseCompletedAt,
        disposeCompletedAtEpochMilliseconds: disposeCompletedAt,
        releaseBeforeDispose:
          releaseCompletedAtPerformance <= disposeCompletedAtPerformance,
        unhandledRejectionCount: unhandled.length,
        workerReadyToTerminate: true,
      }),
    });
  } finally {
    self.removeEventListener("unhandledrejection", unhandledListener);
    if (result !== undefined) {
      try {
        await backend.releaseResult(result);
      } catch {
        // The primary failure remains authoritative.
      }
    }
    if (disposeCompletedAt === 0) {
      try {
        await backend.dispose();
      } catch {
        // The primary failure remains authoritative.
      }
    }
  }
}

async function runVaeProductArm(
  armId: Opt0080ProductArmId,
  identity: Opt0018RunIdentity,
): Promise<void> {
  const arm = OPT_0080_VAE_PRODUCT_ARM_ORDER.find(({ id }) => id === armId);
  if (arm === undefined) throw new Error("OPT-0080 VAE product arm changed");
  const backend = createAceWebGpuPipelineBackend();
  const controller = new AbortController();
  activeAbortController = controller;
  const diagnostics: AceDiagnostic[] = [];
  const initializationProgress: AceInitializationProgress[] = [];
  const generationProgress: AceGenerationProgress[] = [];
  const unhandled: unknown[] = [];
  const unhandledListener = (event: PromiseRejectionEvent): void => {
    event.preventDefault();
    unhandled.push(event.reason);
  };
  self.addEventListener("unhandledrejection", unhandledListener);
  let result: AceGenerationResult | undefined;
  let evidence: AceOpt0080ProductEvidence | undefined;
  let releaseCompletedAt = 0;
  let disposeCompletedAt = 0;
  let releaseCompletedAtPerformance = 0;
  let disposeCompletedAtPerformance = 0;
  try {
    postProgress(`${arm.id}: cache-only initialization on a fresh device`);
    const runtime = await backend.initialize(productionConfiguration(), {
      modelSource: "cache-only",
      signal: controller.signal,
      onProgress: (value) => initializationProgress.push(value),
      onDiagnostic: (value) => diagnostics.push(value),
    });
    validateRuntime(runtime);
    postProgress(`${arm.id}: 96-second exact VAE product-selector execution`);
    const forcedPolicy = arm.vaeSchedulingPolicyOverride;
    if (arm.id !== "production" && forcedPolicy === undefined) {
      throw new Error("OPT-0080 VAE forced product policy changed");
    }
    result = await backend.generate(OPT_0080_PRODUCT_REQUEST, {
      signal: controller.signal,
      onProgress: (value) => generationProgress.push(value),
      onDiagnostic: (value) => diagnostics.push(value),
      ...(arm.id === "production"
        ? {}
        : {
            opt0080ProductRun: {
              // Keep the already-integrated DiT policy identical in both VAE
              // diagnostic replays; only VAE scheduling differs.
              submissionPolicyOverride: "depth2-phase-epoch4" as const,
              vaeSchedulingPolicyOverride: forcedPolicy!,
              onEvidence(value: AceOpt0080ProductEvidence) {
                if (evidence !== undefined) {
                  throw new Error("OPT-0080 VAE product evidence emitted twice");
                }
                evidence = value;
              },
            },
          }),
    });
    validateProgress(initializationProgress, generationProgress);
    validateDiagnostics(diagnostics);
    const schedulingReceipt = result.metrics.vaeScheduling;
    if (schedulingReceipt === undefined) {
      throw new Error("OPT-0080 VAE product scheduling receipt was omitted");
    }
    requireOpt0080VaeProductSchedulingReceipt(schedulingReceipt, arm.id);
    const captured = arm.id === "production"
      ? undefined
      : requireVaeProductEvidence(arm.id, result, evidence);
    if (arm.id === "production" && evidence !== undefined) {
      throw new Error("OPT-0080 ordinary VAE product arm was not seam-free");
    }

    const [requestIdentity, wav, wavScan] = await Promise.all([
      hashText(JSON.stringify(OPT_0080_PRODUCT_REQUEST)),
      hashBlob(result.audio),
      scanWav(result.audio),
    ]);
    if (
      requestIdentity.byteLength !== OPT_0080_PRODUCT_REQUEST_BYTES ||
      requestIdentity.sha256 !== OPT_0080_PRODUCT_REQUEST_SHA256 ||
      wav.sha256 !== OPT_0080_VAE_PRODUCT_WAV_SHA256
    ) throw new Error("OPT-0080 canonical request identity changed");
    const common = Object.freeze({
      schema: "ace-opt-0080-vae-product-selector-arm-v1" as const,
      armId: arm.id,
      identity,
      request: Object.freeze({
        canonicalJson: JSON.stringify(OPT_0080_PRODUCT_REQUEST),
        ...requestIdentity,
      }),
      evidenceMode: captured === undefined
        ? "seam-free-ordinary" as const
        : "forced-retained-output" as const,
      vaeScheduling: schedulingReceipt,
      wav: Object.freeze({ ...wav, scan: wavScan }),
      vaePlan: captured === undefined
        ? summarizeCanonicalPlan()
        : summarizePlan(captured),
      metadata: summarizeResult(result),
      metrics: result.metrics,
      runtime,
      progress: Object.freeze({
        initializationStages: uniqueStages(initializationProgress),
        generationStages: uniqueStages(generationProgress),
        initializationEventCount: initializationProgress.length,
        generationEventCount: generationProgress.length,
      }),
      diagnosticCodes: Object.freeze(diagnostics.map(({ code }) => code)),
    });
    let summary: Readonly<Record<string, unknown>>;
    let finalWords: Uint32Array<ArrayBuffer> | undefined;
    if (captured === undefined) {
      summary = common;
    } else {
      const schedulingEvidence = captured.vaeSchedulingEvidence;
      if (schedulingEvidence === undefined) {
        throw new Error("OPT-0080 VAE forced topology evidence was omitted");
      }
      const [finalLatent, raw, seam] = await Promise.all([
        hashBytes(new Uint8Array(
          captured.finalLatentRawU32.buffer,
          captured.finalLatentRawU32.byteOffset,
          captured.finalLatentRawU32.byteLength,
        )),
        hashBlob(captured.rawSnapshot),
        inspectSeam(captured.rawSnapshot),
      ]);
      if (
        finalLatent.sha256 !== OPT_0080_VAE_PRODUCT_FINAL_LATENT_SHA256 ||
        raw.sha256 !== OPT_0080_VAE_PRODUCT_RAW_SHA256 ||
        seam.sha256 !== OPT_0080_VAE_PRODUCT_SEAM_SHA256
      ) throw new Error("OPT-0080 VAE retained product identity changed");
      finalWords = captured.finalLatentRawU32.slice();
      summary = Object.freeze({
        ...common,
        selectedProductionDitPolicy: captured.selectedProductionPolicy,
        effectiveDitSubmissionPolicy: captured.effectiveSubmissionPolicy,
        vaeSchedulingEvidence:
          summarizeOpt0080VaeProductSchedulingEvidence(
            schedulingEvidence,
            schedulingReceipt,
          ),
        finalLatent: Object.freeze({
          elementCount: captured.finalLatentRawU32.length,
          byteLength: captured.finalLatentRawU32.byteLength,
          sha256: finalLatent.sha256,
          scan: scanFloat32(captured.finalLatent),
        }),
        raw: Object.freeze({
          ...raw,
          stats: captured.rawStats,
        }),
        seam,
      });
    }

    throwIfProtocolFailed();
    const released = waitForRelease(arm.id);
    if (captured === undefined) {
      self.postMessage({
        type: "arm-ready",
        armId: arm.id,
        summary,
        wav: result.audio,
      });
    } else {
      self.postMessage({
        type: "arm-ready",
        armId: arm.id,
        summary,
        finalLatentRawU32: finalWords,
        rawSnapshot: captured.rawSnapshot,
        wav: result.audio,
      }, [finalWords!.buffer]);
    }
    await released;
    throwIfProtocolFailed();
    postProgress(`${arm.id}: comparison complete; releasing retained output`);
    await backend.releaseResult(result);
    releaseCompletedAt = Date.now();
    releaseCompletedAtPerformance = performance.now();
    result = undefined;
    await backend.dispose();
    disposeCompletedAt = Date.now();
    disposeCompletedAtPerformance = performance.now();
    await nextTask();
    throwIfProtocolFailed();
    if (unhandled.length !== 0) {
      throw new Error("OPT-0080 VAE product arm observed an unhandled rejection");
    }
    self.removeEventListener("unhandledrejection", unhandledListener);
    self.postMessage({
      type: "arm-released",
      armId: arm.id,
      lifecycle: Object.freeze({
        releaseCompletedAtEpochMilliseconds: releaseCompletedAt,
        disposeCompletedAtEpochMilliseconds: disposeCompletedAt,
        releaseBeforeDispose:
          releaseCompletedAtPerformance <= disposeCompletedAtPerformance,
        unhandledRejectionCount: unhandled.length,
        workerReadyToTerminate: true,
      }),
    });
  } finally {
    self.removeEventListener("unhandledrejection", unhandledListener);
    if (result !== undefined) {
      try {
        await backend.releaseResult(result);
      } catch {
        // The primary failure remains authoritative.
      }
    }
    if (disposeCompletedAt === 0) {
      try {
        await backend.dispose();
      } catch {
        // The primary failure remains authoritative.
      }
    }
  }
}

async function runCancellation(identity: Opt0018RunIdentity): Promise<void> {
  const backend = createAceWebGpuPipelineBackend();
  const initializationController = new AbortController();
  const generationController = new AbortController();
  activeAbortController = initializationController;
  const sentinel = Object.freeze({ kind: "opt-0080-product-cancellation" });
  const diagnostics: AceDiagnostic[] = [];
  const progress: AceGenerationProgress[] = [];
  const unhandled: unknown[] = [];
  const unhandledListener = (event: PromiseRejectionEvent): void => {
    event.preventDefault();
    unhandled.push(event.reason);
  };
  self.addEventListener("unhandledrejection", unhandledListener);
  let abortAt = 0;
  let rejectedAt = 0;
  let triggerCount = 0;
  let progressCountAtAbort = -1;
  let rejectionIdentity = false;
  let disposeCompletedAt = 0;
  let disposeCompletedAtPerformance = 0;
  try {
    postProgress("cancellation: cache-only initialization on a fresh device");
    const runtime = await backend.initialize(productionConfiguration(), {
      modelSource: "cache-only",
      signal: initializationController.signal,
      onProgress: () => undefined,
      onDiagnostic: (value) => diagnostics.push(value),
    });
    validateRuntime(runtime);
    activeAbortController = generationController;
    postProgress("cancellation: ordinary production selector, abort at command 0 completion");
    try {
      await backend.generate(OPT_0080_PRODUCT_REQUEST, {
        signal: generationController.signal,
        onProgress(value) {
          progress.push(value);
          if (
            triggerCount === 0 &&
            value.stage === "dit-denoise" &&
            value.completedUnits === 0 &&
            value.totalUnits === 8 &&
            value.message === "Denoising"
          ) {
            triggerCount += 1;
            progressCountAtAbort = progress.length;
            abortAt = performance.now();
            generationController.abort(sentinel);
          }
        },
        onDiagnostic: (value) => diagnostics.push(value),
      });
      throw new Error("OPT-0080 cancellation unexpectedly produced output");
    } catch (error) {
      rejectedAt = performance.now();
      rejectionIdentity = error === sentinel;
      if (!rejectionIdentity) throw error;
    }
    await backend.dispose();
    disposeCompletedAt = Date.now();
    disposeCompletedAtPerformance = performance.now();
    await nextTask();
    throwIfProtocolFailed();
    self.removeEventListener("unhandledrejection", unhandledListener);
    const postAbortStages = progress.slice(progressCountAtAbort).map(
      ({ stage }) => stage,
    );
    const fatalDiagnosticCount = diagnostics.filter(
      ({ severity }) => severity === "error",
    ).length;
    const abortThroughRejectionMs = rejectedAt - abortAt;
    const abortThroughCleanupMs = disposeCompletedAtPerformance - abortAt;
    if (
      triggerCount !== 1 ||
      progress.length !== progressCountAtAbort ||
      postAbortStages.length !== 0 ||
      abortThroughRejectionMs < 0 ||
      abortThroughCleanupMs < 0 ||
      abortThroughCleanupMs > 1_000 ||
      unhandled.length !== 0 ||
      fatalDiagnosticCount !== 0 ||
      progress.some(({ stage }) =>
        stage === "release-dit" || stage === "vae-load" ||
        stage === "vae-decode" || stage === "wav-encode" || stage === "done"
      )
    ) throw new Error("OPT-0080 ordinary product cancellation gate failed");
    self.postMessage({
      type: "cancellation-complete",
      summary: Object.freeze({
        schema: "ace-opt-0080-product-cancellation-v1",
        identity,
        ordinaryProductionSelector: true,
        opt0080ProductRunOmitted: true,
        triggerCount,
        progressCountAtAbort,
        publicProgressAfterAbortCount: postAbortStages.length,
        abortReasonIdentityPreserved: rejectionIdentity,
        abortObservationThroughRejectionMs: abortThroughRejectionMs,
        abortObservationThroughRejectionAndCleanupMs: abortThroughCleanupMs,
        postDitStageCount: 0,
        outputCount: 0,
        evidenceCount: 0,
        fatalDiagnosticCount,
        unhandledRejectionCount: unhandled.length,
        disposeCompletedAtEpochMilliseconds: disposeCompletedAt,
        workerReadyToTerminate: true,
      }),
    });
  } finally {
    self.removeEventListener("unhandledrejection", unhandledListener);
    if (disposeCompletedAt === 0) {
      try {
        await backend.dispose();
      } catch {
        // The primary failure remains authoritative.
      }
    }
  }
}

function requireEvidence(
  armId: Opt0080ProductArmId,
  expectedPolicy: "depth1-epoch1" | "depth2-phase-epoch4",
  result: AceGenerationResult,
  evidence: AceOpt0080ProductEvidence | undefined,
): AceOpt0080ProductEvidence {
  if (
    evidence === undefined || evidence.result !== result ||
    evidence.selectedProductionPolicy !== "depth2-phase-epoch4" ||
    evidence.effectiveSubmissionPolicy !== expectedPolicy ||
    evidence.finalLatent.length !== OPT_0080_PRODUCT_LATENT_FRAMES * 64 ||
    evidence.finalLatentRawU32.buffer !== evidence.finalLatent.buffer ||
    evidence.rawSnapshot.size !== OPT_0080_PRODUCT_RAW_BYTES ||
    evidence.rawStats.outputInterleavedElements !==
      OPT_0080_PRODUCT_AUDIO_FRAMES * 2 ||
    evidence.rawStats.windowsDecoded !== OPT_0080_PRODUCT_WINDOW_COUNT ||
    result.audio.size !== OPT_0080_PRODUCT_WAV_BYTES ||
    result.frameCount !== OPT_0080_PRODUCT_AUDIO_FRAMES
  ) throw new Error(`OPT-0080 ${armId} product evidence changed`);
  return evidence;
}

function requireVaeProductEvidence(
  armId: Exclude<Opt0080ProductArmId, "production">,
  result: AceGenerationResult,
  evidence: AceOpt0080ProductEvidence | undefined,
): AceOpt0080ProductEvidence {
  if (
    evidence === undefined || evidence.result !== result ||
    evidence.selectedProductionPolicy !== "depth2-phase-epoch4" ||
    evidence.effectiveSubmissionPolicy !== "depth2-phase-epoch4" ||
    evidence.vaeSchedulingEvidence?.length !== OPT_0080_PRODUCT_WINDOW_COUNT ||
    evidence.finalLatent.length !== OPT_0080_PRODUCT_LATENT_FRAMES * 64 ||
    evidence.finalLatentRawU32.buffer !== evidence.finalLatent.buffer ||
    evidence.rawSnapshot.size !== OPT_0080_PRODUCT_RAW_BYTES ||
    evidence.rawStats.outputInterleavedElements !==
      OPT_0080_PRODUCT_AUDIO_FRAMES * 2 ||
    evidence.rawStats.windowsDecoded !== OPT_0080_PRODUCT_WINDOW_COUNT ||
    result.audio.size !== OPT_0080_PRODUCT_WAV_BYTES ||
    result.frameCount !== OPT_0080_PRODUCT_AUDIO_FRAMES
  ) throw new Error(`OPT-0080 VAE ${armId} retained evidence changed`);
  return evidence;
}

function summarizeCanonicalPlan() {
  return summarizeVaePlan(planAceVaeChunkedDecode(
    OPT_0080_PRODUCT_LATENT_FRAMES,
    { chunkFrames: 2_378, overlapFrames: 64 },
  ));
}

function summarizePlan(evidence: AceOpt0080ProductEvidence) {
  return summarizeVaePlan(evidence.vaePlan);
}

function summarizeVaePlan(plan: AceOpt0080ProductEvidence["vaePlan"]) {
  const first = plan.windows[0];
  const second = plan.windows[1];
  if (
    plan.latentFrames !== OPT_0080_PRODUCT_LATENT_FRAMES || plan.direct ||
    plan.chunkFrames !== 2_378 || plan.overlapFrames !== 64 ||
    plan.strideFrames !== 2_250 || plan.windows.length !== 2 ||
    first?.windowStartLatentFrame !== 0 ||
    first.windowEndLatentFrame !== 2_314 ||
    first.coreStartLatentFrame !== 0 || first.coreEndLatentFrame !== 2_250 ||
    first.discardSuffixLatentFrames !== 64 ||
    second?.windowStartLatentFrame !== 2_186 ||
    second.windowEndLatentFrame !== 2_400 ||
    second.coreStartLatentFrame !== 2_250 ||
    second.coreEndLatentFrame !== 2_400 ||
    second.discardPrefixLatentFrames !== 64 ||
    second.outputStartAudioFrame !== OPT_0080_PRODUCT_STITCH_SEAM_AUDIO_FRAME
  ) throw new Error("OPT-0080 exact two-window seam plan changed");
  return Object.freeze({
    latentFrames: plan.latentFrames,
    chunkFrames: plan.chunkFrames,
    overlapFrames: plan.overlapFrames,
    strideFrames: plan.strideFrames,
    direct: plan.direct,
    windowCount: plan.windows.length,
    windows: plan.windows,
  });
}

function summarizeResult(result: AceGenerationResult) {
  if (
    result.durationSeconds !== 96 || result.seed !== "0000000000c0ffee" ||
    result.generationProfile !== "ace-turbo-v1-correctness" ||
    result.modelManifestSha256 !== OPT_0080_PRODUCT_MAIN_MANIFEST_SHA256 ||
    result.mimeType !== "audio/wav" || result.sampleRateHz !== 48_000 ||
    result.channelCount !== 2 || result.frameCount !== OPT_0080_PRODUCT_AUDIO_FRAMES
  ) throw new Error("OPT-0080 public result metadata changed");
  return Object.freeze({
    durationSeconds: result.durationSeconds,
    seed: result.seed,
    generationProfile: result.generationProfile,
    modelManifestId: result.modelManifestId,
    modelManifestSha256: result.modelManifestSha256,
    mimeType: result.mimeType,
    sampleRateHz: result.sampleRateHz,
    channelCount: result.channelCount,
    frameCount: result.frameCount,
    audioStorageIdWasSafe:
      /^(?!\.{1,2}$)[A-Za-z0-9._-]{1,96}$/.test(result.audioStorageId),
  });
}

function validateRuntime(runtime: AceRuntimeDiagnostics): void {
  if (
    runtime.modelManifestSha256 !== OPT_0080_PRODUCT_MAIN_MANIFEST_SHA256 ||
    runtime.ditDenseManifestSha256 !== OPT_0080_PRODUCT_DENSE_MANIFEST_SHA256 ||
    runtime.ditDenseRuntimeProfile !== "opt-0009-fp16-fp32-dense-v1" ||
    runtime.ditAttentionRuntimeProfile !==
      "opt-0070-fixed32-quad-query32-full-self-production-v1" ||
    runtime.vaeManifestSha256 !== OPT_0080_PRODUCT_VAE_MANIFEST_SHA256 ||
    runtime.vaeRuntimeProfile !==
      "opt-0072-mixed-fp16-fixed32-dual-k4-production-v1" ||
    runtime.vaeWindowRuntimeProfile !==
      "opt-0070-c2378-overlap64-production-v1" ||
    runtime.vaeMaxWindowFrames !== 2_378 ||
    runtime.executionProfile.id !== "reference-bf16-subgroups" ||
    runtime.schedulingProfile !== "cooperative"
  ) throw new Error("OPT-0080 production runtime identity changed");
}

function validateProgress(
  initialization: readonly AceInitializationProgress[],
  generation: readonly AceGenerationProgress[],
): void {
  const initializationStages = uniqueStages(initialization);
  const generationStages = uniqueStages(generation);
  if (
    initializationStages.at(-1) !== "ready" ||
    generationStages.at(-1) !== "done" ||
    !generationStages.includes("release-dit") ||
    !generationStages.includes("vae-decode") ||
    !generationStages.includes("wav-encode")
  ) throw new Error("OPT-0080 product progress lifecycle changed");
}

function validateDiagnostics(diagnostics: readonly AceDiagnostic[]): void {
  if (diagnostics.some(({ severity }) => severity === "error")) {
    throw new Error("OPT-0080 product arm emitted a fatal diagnostic");
  }
}

async function inspectSeam(raw: Blob) {
  const audioFrameBytes = 2 * Float32Array.BYTES_PER_ELEMENT;
  const radiusAudioFrames = OPT_0080_PRODUCT_SEAM_RADIUS_LATENT_FRAMES * 1_920;
  const startAudioFrame = OPT_0080_PRODUCT_STITCH_SEAM_AUDIO_FRAME -
    radiusAudioFrames;
  const endAudioFrame = OPT_0080_PRODUCT_STITCH_SEAM_AUDIO_FRAME +
    radiusAudioFrames;
  const snapshot = raw.slice(
    startAudioFrame * audioFrameBytes,
    endAudioFrame * audioFrameBytes,
  );
  const scan = await scanRaw(snapshot);
  if (
    snapshot.size !== 1_966_080 || scan.sampleCount !== 491_520 ||
    scan.nonFiniteCount !== 0
  ) throw new Error("OPT-0080 seam neighborhood changed");
  return Object.freeze({
    seamLatentFrame: OPT_0080_PRODUCT_STITCH_SEAM_LATENT_FRAME,
    seamAudioFrame: OPT_0080_PRODUCT_STITCH_SEAM_AUDIO_FRAME,
    radiusLatentFrames: OPT_0080_PRODUCT_SEAM_RADIUS_LATENT_FRAMES,
    startAudioFrame,
    endAudioFrame,
    ...await hashBlob(snapshot),
    scan,
  });
}

async function scanRaw(blob: Blob) {
  let sampleCount = 0;
  let nonFiniteCount = 0;
  let nonzeroCount = 0;
  let peak = 0;
  const blockBytes = 1_048_576;
  for (let offset = 0; offset < blob.size; offset += blockBytes) {
    const end = Math.min(blob.size, offset + blockBytes);
    const values = new Float32Array(await blob.slice(offset, end).arrayBuffer());
    for (const value of values) {
      sampleCount += 1;
      if (!Number.isFinite(value)) nonFiniteCount += 1;
      else {
        if (value !== 0) nonzeroCount += 1;
        peak = Math.max(peak, Math.abs(value));
      }
    }
  }
  return Object.freeze({ sampleCount, nonFiniteCount, nonzeroCount, peak });
}

function scanFloat32(values: Float32Array) {
  let nonFiniteCount = 0;
  let nonzeroCount = 0;
  let maximumAbsolute = 0;
  for (const value of values) {
    if (!Number.isFinite(value)) {
      nonFiniteCount += 1;
      continue;
    }
    if (value !== 0) nonzeroCount += 1;
    maximumAbsolute = Math.max(maximumAbsolute, Math.abs(value));
  }
  if (nonFiniteCount !== 0 || nonzeroCount === 0) {
    throw new Error("OPT-0080 final latent scan changed");
  }
  return Object.freeze({ nonFiniteCount, nonzeroCount, maximumAbsolute });
}

async function scanWav(blob: Blob) {
  if (blob.size !== OPT_0080_PRODUCT_WAV_BYTES) {
    throw new Error("OPT-0080 WAV byte length changed");
  }
  const scan = await scanRaw(blob.slice(44));
  if (
    scan.sampleCount !== OPT_0080_PRODUCT_AUDIO_FRAMES * 2 ||
    scan.nonFiniteCount !== 0 || scan.nonzeroCount === 0 ||
    scan.peak > 0.891_251
  ) throw new Error("OPT-0080 normalized WAV scan changed");
  return scan;
}

async function hashText(value: string) {
  return await hashBytes(new TextEncoder().encode(value));
}

async function hashBlob(blob: Blob) {
  const hash = new AceIncrementalSha256();
  const reader = blob.stream().getReader();
  let byteLength = 0;
  while (true) {
    const chunk = await reader.read();
    if (chunk.done) break;
    hash.update(chunk.value);
    byteLength += chunk.value.byteLength;
  }
  return Object.freeze({ byteLength, sha256: hash.digestHex() });
}

async function hashBytes(bytes: Uint8Array) {
  const hash = new AceIncrementalSha256();
  hash.update(bytes);
  return Object.freeze({ byteLength: bytes.byteLength, sha256: hash.digestHex() });
}

function uniqueStages(values: readonly { readonly stage: string }[]) {
  return Object.freeze(values.reduce<string[]>((stages, { stage }) => {
    if (stages.at(-1) !== stage) stages.push(stage);
    return stages;
  }, []));
}

function productionConfiguration(): AceWorkerConfiguration {
  return Object.freeze({
    manifestUrl: new URL("/model/files-reference/manifest.json", location.href).href,
    manifestSha256: OPT_0080_PRODUCT_MAIN_MANIFEST_SHA256,
    modelProfile: "reference-bf16",
    schedulingProfile: "cooperative",
    ditDensePackage: Object.freeze({
      manifestUrl: new URL(
        "/model/files-fp16-dit-rev7-oracle/manifest.json",
        location.href,
      ).href,
      manifestSha256: OPT_0080_PRODUCT_DENSE_MANIFEST_SHA256,
      runtimeProfile: "opt-0009-fp16-fp32-dense-v1",
    }),
    ditAttentionRuntimeProfile:
      "opt-0070-fixed32-quad-query32-full-self-production-v1",
    vaePackage: Object.freeze({
      manifestUrl: new URL(
        "/model/files-fp16-vae-revision7-experimental/manifest.json",
        location.href,
      ).href,
      manifestSha256: OPT_0080_PRODUCT_VAE_MANIFEST_SHA256,
      runtimeProfile: "opt-0072-mixed-fp16-fixed32-dual-k4-production-v1",
      windowRuntimeProfile: "opt-0070-c2378-overlap64-production-v1",
      maxWindowFrames: 2_378,
    }),
  });
}

function nextTask(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

function waitForRelease(armId: Opt0080ProductArmId): Promise<void> {
  if (workerPhase !== "running" || releaseWaiter !== undefined) {
    throw new Error("OPT-0080 release waiter lifecycle changed");
  }
  workerPhase = "awaiting-release";
  return new Promise<void>((resolve, reject) => {
    releaseWaiter = { armId, settled: false, resolve, reject };
  });
}

function throwIfProtocolFailed(): void {
  if (protocolFailure !== undefined) throw protocolFailure;
}

function rejectProtocol(message: string): void {
  const error = protocolFailure ?? new Error(message);
  protocolFailure = error;
  workerPhase = "failed";
  if (
    activeAbortController !== undefined &&
    !activeAbortController.signal.aborted
  ) activeAbortController.abort(error);
  if (releaseWaiter !== undefined && !releaseWaiter.settled) {
    releaseWaiter.settled = true;
    releaseWaiter.reject(error);
  }
  // An active run reports only after its finally block has released its result
  // and disposed the backend. Reporting here could let the page terminate the
  // worker before that cleanup completes.
  if (!runInFlight) fail(error);
}

function isRunMessage(value: unknown): value is RunMessage {
  if (typeof value !== "object" || value === null) return false;
  const candidate = value as {
    readonly type?: unknown;
    readonly runKind?: unknown;
    readonly productGate?: unknown;
  };
  return candidate.type === "run" &&
    (candidate.runKind === "control" || candidate.runKind === "candidate" ||
      candidate.runKind === "production" ||
      candidate.runKind === "cancellation") &&
    (candidate.productGate === undefined || candidate.productGate === "dit" ||
      candidate.productGate === OPT_0080_VAE_PRODUCT_GATE_KIND);
}

function isReleaseMessage(value: unknown): value is ReleaseMessage {
  return typeof value === "object" && value !== null &&
    (value as { readonly type?: unknown }).type === "release" &&
    typeof (value as { readonly armId?: unknown }).armId === "string";
}

function postProgress(message: string): void {
  self.postMessage({ type: "progress", message });
}

function fail(error: unknown): void {
  workerPhase = "failed";
  if (failurePosted) return;
  failurePosted = true;
  self.postMessage({ type: "failed", error: serializeOpt0018Failure(error) });
}
