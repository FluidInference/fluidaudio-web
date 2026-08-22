/// <reference lib="webworker" />
/// <reference types="@webgpu/types" />
/// <reference types="vite/client" />

import {
  assertAceGenerationRequest,
  type AceGenerationRequest,
  type AceGenerationResult,
} from "../../src/api.js";
import { AceIncrementalSha256, aceSha256Hex } from
  "../../src/model/sha256.js";
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
  type AceOpt0064CaptureEvent,
  type AceOpt0064CaptureSink,
  type AceWebGpuGenerationContext,
} from "../../src/runtime/webgpu-pipeline.js";
import {
  OPT_0064_ACCEPTED_WAV_SHA256,
  OPT_0064_ACCEPTED_WAV_AUTHORITY,
  OPT_0064_OUTPUT_DATA_BYTES,
  OPT_0064_OUTPUT_FRAMES,
  OPT_0064_OUTPUT_WAV_BYTES,
  OPT_0064_REQUEST_BYTE_LENGTH,
  OPT_0064_REQUEST_CANONICAL_JSON,
  OPT_0064_REQUEST_SHA256,
  OPT_0064_TOTAL_UPLOAD_BYTES,
  OPT_0064_TOTAL_UPLOAD_DRAINS,
  OPT_0064_TOTAL_UPLOAD_FILES,
  OPT_0064_TOTAL_UPLOAD_GAPS,
  OPT_0064_UPLOAD_CHUNK_BYTES,
  OPT_0064_UPLOAD_PHASES,
  serializeOpt0064Failure,
  requireOpt0064ThermalGate,
  requireOpt0064ThermalTrace,
  validateOpt0064RunIdentity,
  type Opt0064RunIdentity,
  type Opt0064ThermalGate,
  type Opt0064ThermalTrace,
  type Opt0064UploadPhaseId,
} from "./opt-0064-direct-request-capture-contract.js";

const MAIN_MANIFEST_PATH = "/model/files-reference/manifest.json" as const;
const MAIN_MANIFEST_SHA256 =
  "18f36c6420976475af65ecd833ca56c6119706322ce54120389d4915d8e80db6" as const;
const DENSE_MANIFEST_PATH =
  "/model/files-fp16-dit-rev7-oracle/manifest.json" as const;
const DENSE_MANIFEST_SHA256 =
  "d3fc0020efcf60702db411da2fd4b93e9bb84f1437ed310aef01c892727e452f" as const;
const DENSE_RUNTIME_PROFILE = "opt-0009-fp16-fp32-dense-v1" as const;
const VAE_MANIFEST_PATH =
  "/model/files-fp16-vae-experimental/manifest.json" as const;
const VAE_MANIFEST_SHA256 =
  "94a1ae61354f7481facbb9787d003488ab1bc351a137fd2bd7ff69dd99aef949" as const;
const VAE_RUNTIME_PROFILE =
  "opt-0028-mixed-fp16-fixed32-exact-packed-v1" as const;
const MAXIMUM_CAPTURE_EVENTS = 768;

interface PrepareMessage {
  readonly type: "prepare";
  readonly identity: Opt0064RunIdentity;
}

interface RunMessage {
  readonly type: "run";
  readonly thermalGate: Opt0064ThermalGate;
}

interface CompleteThermalMessage {
  readonly type: "complete-thermal";
  readonly thermalTrace: Opt0064ThermalTrace;
}

type IncomingMessage = PrepareMessage | RunMessage | CompleteThermalMessage;

let state: "idle" | "ready" | "running" | "await-thermal" | "settled" =
  "idle";
let identity: Opt0064RunIdentity | undefined;
let request: AceGenerationRequest | undefined;
let readyAtEpochMilliseconds = 0;
let acceptedGate: Opt0064ThermalGate | undefined;
let cleanupCompletedAtEpochMilliseconds = 0;
let pendingReceipt: Readonly<Record<string, unknown>> | undefined;
let requestStartedAtEpochMilliseconds = 0;
let requestStartedAt = 0;
let failureEvidence: Readonly<Record<string, unknown>> | undefined;

self.addEventListener("message", (event: MessageEvent<IncomingMessage>) => {
  const message = event.data;
  if (message.type === "prepare" && state === "idle") {
    try {
      identity = validateOpt0064RunIdentity(message.identity);
      request = validateRequest();
      readyAtEpochMilliseconds = Date.now();
      state = "ready";
      self.postMessage({
        type: "ready-for-thermal-gate",
        readyAtEpochMilliseconds,
        preparation: Object.freeze({
          schema: "ace-opt-0064-static-preparation-v1",
          requestSha256: OPT_0064_REQUEST_SHA256,
          packageOrGpuWorkPerformed: false,
          sourceAuthority: Object.freeze({
            coreCommit: identity.coreCommit,
            harnessCommit: identity.harnessCommit,
          }),
        }),
      });
    } catch (error) {
      fail(error);
    }
    return;
  }
  if (message.type === "run" && state === "ready") {
    try {
      acceptedGate = requireOpt0064ThermalGate(
        message.thermalGate,
        readyAtEpochMilliseconds,
        Date.now(),
      );
      requestStartedAtEpochMilliseconds = Date.now();
      requestStartedAt = performance.now();
      state = "running";
      void executeCapture().catch(fail);
    } catch (error) {
      self.postMessage({
        type: "gate-rejected",
        error: serializeOpt0064Failure(error),
      });
    }
    return;
  }
  if (
    message.type === "complete-thermal" && state === "await-thermal" &&
    acceptedGate !== undefined && pendingReceipt !== undefined
  ) {
    try {
      const thermalTrace = requireOpt0064ThermalTrace(
        message.thermalTrace,
        acceptedGate,
        cleanupCompletedAtEpochMilliseconds,
        Date.now(),
      );
      state = "settled";
      self.postMessage({
        type: "capture-complete",
        result: Object.freeze({
          ...pendingReceipt,
          protocol: Object.freeze({
            thermalGate: acceptedGate,
            thermalTrace,
            nominalAtRequestStart: true,
            nominalThroughoutRequest:
              thermalTrace.nonNominalObservationCount === 0,
            oneArmNoThermalRetry: true,
          }),
        }),
      });
    } catch (error) {
      self.postMessage({
        type: "trace-rejected",
        error: serializeOpt0064Failure(error),
      });
    }
  }
});

async function executeCapture(): Promise<void> {
  if (
    identity === undefined || request === undefined || acceptedGate === undefined ||
    requestStartedAtEpochMilliseconds === 0 || requestStartedAt === 0
  ) {
    throw new Error("OPT-0064 request authorization is incomplete");
  }
  const backend = createAceWebGpuPipelineBackend();
  const controller = new AbortController();
  const events: AceOpt0064CaptureEvent[] = [];
  const capture: AceOpt0064CaptureSink = Object.freeze({
    onEvent(event: AceOpt0064CaptureEvent): void {
      if (events.length < MAXIMUM_CAPTURE_EVENTS) events.push(event);
    },
  });
  const diagnosticCodes: string[] = [];
  const fatalDiagnosticCodes: string[] = [];
  const initializationStages = new Set<string>();
  const generationStages = new Set<string>();
  let initializationProgressEventCount = 0;
  let generationProgressEventCount = 0;
  let lastInitializationProgress: AceInitializationProgress | undefined;
  let lastGenerationProgress: AceGenerationProgress | undefined;
  let lastDiagnostic: AceDiagnostic | undefined;
  let activePhase = "initialization";
  let primaryFailurePhase: string | undefined;
  let result: AceGenerationResult | undefined;
  let diagnostics: AceRuntimeDiagnostics | undefined;
  let primaryError: unknown;
  let releaseError: unknown;
  let disposeError: unknown;
  let initializedAtEpochMilliseconds = 0;
  let wavReadyAtEpochMilliseconds = 0;
  let releasedAtEpochMilliseconds = 0;
  let requestWallMs = 0;
  let initializationWallMs = 0;
  let generationWallMs = 0;
  let outputHashWallMs = 0;
  let releaseResultWallMs = 0;
  let disposeWallMs = 0;
  let outputHash: Awaited<ReturnType<typeof hashBlob>> | undefined;
  let captureSummary: Readonly<Record<string, unknown>> | undefined;
  failureEvidence = undefined;
  postProgress("authenticating warm packages and initializing the production device");
  try {
    const initializationStartedAt = performance.now();
    diagnostics = await backend.initialize(productionConfiguration(), {
      modelSource: "cache-only",
      signal: controller.signal,
      opt0064Capture: capture,
      onProgress(progress: AceInitializationProgress): void {
        initializationProgressEventCount += 1;
        initializationStages.add(progress.stage);
        lastInitializationProgress = Object.freeze({ ...progress });
      },
      onDiagnostic(diagnostic: AceDiagnostic): void {
        diagnosticCodes.push(diagnostic.code);
        lastDiagnostic = Object.freeze({ ...diagnostic });
        if (diagnostic.severity === "error") {
          fatalDiagnosticCodes.push(diagnostic.code);
        }
      },
    });
    initializationWallMs = performance.now() - initializationStartedAt;
    initializedAtEpochMilliseconds = Date.now();
    validateDiagnostics(diagnostics);
    activePhase = "generation";
    postProgress("running the unchanged direct 12-second production request");
    const generationStartedAt = performance.now();
    const generationContext: AceWebGpuGenerationContext = {
      signal: controller.signal,
      opt0064Capture: capture,
      onProgress(progress: AceGenerationProgress): void {
        generationProgressEventCount += 1;
        generationStages.add(progress.stage);
        lastGenerationProgress = Object.freeze({ ...progress });
      },
      onDiagnostic(diagnostic: AceDiagnostic): void {
        diagnosticCodes.push(diagnostic.code);
        lastDiagnostic = Object.freeze({ ...diagnostic });
        if (diagnostic.severity === "error") {
          fatalDiagnosticCodes.push(diagnostic.code);
        }
      },
    };
    result = await backend.generate(request, generationContext);
    generationWallMs = performance.now() - generationStartedAt;
    wavReadyAtEpochMilliseconds = Date.now();
    requestWallMs = performance.now() - requestStartedAt;
    validateResult(result);
    if (!generationStages.has("done") || fatalDiagnosticCodes.length !== 0) {
      throw new Error("OPT-0064 generation lifecycle or diagnostics changed");
    }
    activePhase = "output-hash";
    postProgress("hashing the bounded output stream after the WAV-ready boundary");
    const outputHashStartedAt = performance.now();
    outputHash = await hashBlob(result.audio);
    outputHashWallMs = performance.now() - outputHashStartedAt;
    if (events.length >= MAXIMUM_CAPTURE_EVENTS) {
      throw new Error("OPT-0064 capture event storage bound was exhausted");
    }
    const validatedCaptureSummary = validateCapture(events);
    captureSummary = validatedCaptureSummary;
    if (
      outputHash.sha256 !== OPT_0064_ACCEPTED_WAV_SHA256 ||
      outputHash.byteLength !== OPT_0064_OUTPUT_WAV_BYTES
    ) throw new Error("OPT-0064 direct WAV identity changed");
    activePhase = "result-release";
    postProgress("releasing the committed output and disposing the backend/device");
    const releaseStartedAt = performance.now();
    await backend.releaseResult(result);
    releaseResultWallMs = performance.now() - releaseStartedAt;
    releasedAtEpochMilliseconds = Date.now();
    result = undefined;
    pendingReceipt = Object.freeze({
      schema: "ace-opt-0064-direct-request-capture-gate-v1",
      experimentId: "OPT-0064",
      status: "passed",
      passed: true,
      identity: Object.freeze({
        run: identity,
        requestCanonicalJson: OPT_0064_REQUEST_CANONICAL_JSON,
        requestSha256: OPT_0064_REQUEST_SHA256,
        requestByteLength: OPT_0064_REQUEST_BYTE_LENGTH,
        mainManifestSha256: MAIN_MANIFEST_SHA256,
        denseManifestSha256: DENSE_MANIFEST_SHA256,
        denseRuntimeProfile: DENSE_RUNTIME_PROFILE,
        vaeManifestSha256: VAE_MANIFEST_SHA256,
        vaeRuntimeProfile: VAE_RUNTIME_PROFILE,
        acceptedWavSha256: OPT_0064_ACCEPTED_WAV_SHA256,
      }),
      timings: Object.freeze({
        requestStartedAtEpochMilliseconds,
        initializedAtEpochMilliseconds,
        wavReadyAtEpochMilliseconds,
        releasedAtEpochMilliseconds,
        requestGenerateToWavWallMs: requestWallMs,
        initializationWallMs,
        generationWallMs,
        outputHashWallMs,
        releaseResultWallMs,
        // Filled after the mandatory finally-dispose boundary below.
        disposeWallMs: 0,
      }),
      output: Object.freeze({
        ...outputHash,
        expectedSha256: OPT_0064_ACCEPTED_WAV_SHA256,
        frameCount: OPT_0064_OUTPUT_FRAMES,
        dataBytes: OPT_0064_OUTPUT_DATA_BYTES,
        hashExcludedFromGenerateToWavWall: true,
      }),
      capture: Object.freeze({
        ...validatedCaptureSummary,
        events: Object.freeze([...events]),
      }),
      lifecycle: Object.freeze({
        initializationProgressEventCount,
        generationProgressEventCount,
        initializationStages: Object.freeze([...initializationStages]),
        generationStages: Object.freeze([...generationStages]),
        diagnosticCodes: Object.freeze(diagnosticCodes),
        fatalDiagnosticCodes: Object.freeze(fatalDiagnosticCodes),
        exactImmutableFileProofOnly: true,
        ordinaryCaptureTraceDisabled: true,
        mappedAtCreationAttempted: false,
        pipelineOrAuthenticationOverlapAttempted: false,
        simultaneousHeavyweightPhaseOwnershipAttempted: false,
        resultReleased: true,
      }),
      scope: Object.freeze({
        captureOnly: true,
        productionMathChanged: false,
        productionSchedulingChanged: false,
        uploadMechanismChanged: false,
        backgroundPreRequestWork: false,
        performanceComparisonClaim: false,
        underOneMinuteClaim: false,
      }),
      decision: "capture-ready-for-opt-0064-candidate-design",
    });
  } catch (error) {
    primaryError = error;
    primaryFailurePhase = activePhase;
  } finally {
    activePhase = "cleanup";
    if (result !== undefined) {
      const releaseStartedAt = performance.now();
      try {
        await backend.releaseResult(result);
      } catch (error) {
        releaseError = error;
      }
      releaseResultWallMs += performance.now() - releaseStartedAt;
      releasedAtEpochMilliseconds = Date.now();
    }
    const disposeStartedAt = performance.now();
    try {
      await backend.dispose();
    } catch (error) {
      disposeError = error;
    }
    disposeWallMs = performance.now() - disposeStartedAt;
    cleanupCompletedAtEpochMilliseconds = Date.now();
  }
  if (
    primaryError !== undefined || releaseError !== undefined ||
    disposeError !== undefined
  ) {
    failureEvidence = createFailureEvidence({
      primaryFailurePhase,
      activePhase,
      initializationProgressEventCount,
      initializationStages,
      lastInitializationProgress,
      generationProgressEventCount,
      generationStages,
      lastGenerationProgress,
      diagnosticCodes,
      fatalDiagnosticCodes,
      lastDiagnostic,
      events,
      captureSummary,
      outputHash,
      releaseError,
      disposeError,
      cleanupCompletedAtEpochMilliseconds,
    });
    throw new AggregateError(
      [primaryError, releaseError, disposeError].filter(
        (value): value is NonNullable<typeof value> => value !== undefined,
      ),
      "OPT-0064 capture/run/cleanup failed",
    );
  }
  if (pendingReceipt === undefined) {
    throw new Error("OPT-0064 successful capture receipt is absent");
  }
  const timings = pendingReceipt["timings"] as Readonly<Record<string, unknown>>;
  pendingReceipt = Object.freeze({
    ...pendingReceipt,
    timings: Object.freeze({ ...timings, disposeWallMs }),
    lifecycle: Object.freeze({
      ...(pendingReceipt["lifecycle"] as Readonly<Record<string, unknown>>),
      cleanupCompletedAtEpochMilliseconds,
      backendAndDeviceDisposed: true,
    }),
  });
  state = "await-thermal";
  self.postMessage({
    type: "capture-awaiting-thermal-trace",
    cleanupCompletedAtEpochMilliseconds,
    summary: Object.freeze({
      requestGenerateToWavWallMs: requestWallMs,
      wavSha256: outputHash!.sha256,
      uploadBytes: OPT_0064_TOTAL_UPLOAD_BYTES,
      uploadFileCount: OPT_0064_TOTAL_UPLOAD_FILES,
      eventCount: events.length,
      releaseResultWallMs,
      disposeWallMs,
    }),
  });
}

function validateCapture(
  events: readonly AceOpt0064CaptureEvent[],
): Readonly<Record<string, unknown>> {
  if (
    events.length === 0 || events.some((event) =>
      event.schema !== "ace-opt-0064-direct-request-capture-event-v1" ||
      !Number.isFinite(event.startedAtMs) ||
      !Number.isFinite(event.completedAtMs) ||
      !Number.isFinite(event.wallMs) || event.wallMs < 0 ||
      event.completedAtMs < event.startedAtMs
    )
  ) throw new Error("OPT-0064 capture event envelope changed");
  const authenticationEvents = events.filter((event) =>
    event.category === "authentication"
  );
  const cacheAuthentication = authenticationEvents.filter((event) =>
    event.operation.endsWith("-cache-authentication")
  );
  const proofReuse = authenticationEvents.filter((event) =>
    event.operation.endsWith("-proof-reuse")
  );
  const acquisitionPlans = authenticationEvents.filter((event) =>
    event.operation.endsWith("-acquisition-plan")
  );
  if (
    cacheAuthentication.length === 0 || proofReuse.length === 0 ||
    acquisitionPlans.length !== 3 ||
    cacheAuthentication.some(({ details }) =>
      details["matched"] !== true ||
      details["actualSha256"] !== details["expectedSha256"] ||
      details["receivedBytes"] !== details["byteLength"] ||
      details["exactImmutableFileProofPublished"] !== true ||
      number(details["maximumHashChunkBytes"]) > OPT_0064_UPLOAD_CHUNK_BYTES
    ) ||
    proofReuse.some(({ details }) =>
      details["exactImmutableFileIdentity"] !== true ||
      details["redundantHashPerformed"] !== false ||
      details["source"] !== "cache"
    ) ||
    acquisitionPlans.some(({ details }) =>
      details["downloadFileCount"] !== 0 || details["downloadBytes"] !== 0 ||
      details["cachedFileCount"] !== details["fileCount"] ||
      details["cachedBytes"] !== details["runtimeBytes"] ||
      details["exactFileObjectCount"] !== details["fileCount"]
    )
  ) throw new Error("OPT-0064 warm-cache authentication proof changed");

  const uploads = events.filter((event) =>
    event.category === "upload" && event.operation.endsWith("-file-upload")
  );
  const proofKeys = new Set(proofReuse.map(({ details }) =>
    `${String(details["packageKind"])}:${String(details["file"])}:` +
      String(details["sha256"])
  ));
  const phaseRows = OPT_0064_UPLOAD_PHASES.map((expected) => {
    const phaseUploads = uploads.filter(({ details }) =>
      uploadPhase(details) === expected.id
    );
    const aggregate = aggregateUploads(phaseUploads);
    if (
      aggregate.fileCount !== expected.fileCount ||
      aggregate.bytes !== expected.bytes ||
      aggregate.queueDrains !== expected.queueDrains ||
      aggregate.queueEmptyGaps !== expected.queueEmptyGaps
    ) throw new Error(`OPT-0064 ${expected.id} upload inventory changed`);
    return Object.freeze({ ...expected, timing: aggregate.timing });
  });
  const totals = aggregateUploads(uploads);
  if (
    totals.fileCount !== OPT_0064_TOTAL_UPLOAD_FILES ||
    totals.bytes !== OPT_0064_TOTAL_UPLOAD_BYTES ||
    totals.queueDrains !== OPT_0064_TOTAL_UPLOAD_DRAINS ||
    totals.queueEmptyGaps !== OPT_0064_TOTAL_UPLOAD_GAPS ||
    uploads.some(({ details }) =>
      details["schema"] !== "ace-gpu-upload-capture-v1" ||
      details["sourceKind"] !== "file" ||
      details["authentication"] !== "exact-immutable-file-proof-reused" ||
      details["redundantHashPerformed"] !== false ||
      details["boundedChunkBytes"] !== OPT_0064_UPLOAD_CHUNK_BYTES ||
      number(details["maximumObservedStreamChunkBytes"]) >
        OPT_0064_UPLOAD_CHUNK_BYTES ||
      number(details["maximumOwnedCopyBytes"]) > 4 ||
      details["uploadedBytes"] !== details["byteLength"] ||
      !proofKeys.has(
        `${String(details["packageKind"])}:${String(details["file"])}:` +
          String(details["sha256"]),
      )
    )
  ) throw new Error("OPT-0064 aggregate direct upload proof changed");

  const construction = Object.fromEntries(
    (["conditioning", "dit", "vae"] as const).map((owner) => {
      const summary = requireEvent(
        events,
        `${owner}-gpu-api-construction-summary`,
      );
      const details = summary.details;
      const pipelineCount = methodCount(details, "createComputePipeline") +
        methodCount(details, "createComputePipelineAsync");
      if (
        details["owner"] !== owner ||
        details["proxyAddsNoGpuCommandsDrainsHashesOrCopies"] !== true ||
        methodCount(details, "createShaderModule") < 1 || pipelineCount < 1 ||
        methodCount(details, "createBindGroup") < 1 ||
        methodCount(details, "createBuffer") < 1
      ) throw new Error(`OPT-0064 ${owner} construction attribution changed`);
      return [owner, Object.freeze({
        wallMs: summary.wallMs,
        shaderModuleCount: methodCount(details, "createShaderModule"),
        pipelineCount,
        bindGroupCount: methodCount(details, "createBindGroup"),
        bufferCount: methodCount(details, "createBuffer"),
        shaderModuleCreationMs: details["shaderModuleCreationMs"],
        pipelineCompilationMs: details["pipelineCompilationMs"],
        bindGroupAndLayoutConstructionMs:
          details["bindGroupAndLayoutConstructionMs"],
        bufferAllocationMs: details["bufferAllocationMs"],
      })];
    }),
  );
  for (const operation of [
    "conditioning-execution",
    "dit-eight-evaluation-execution-and-readback",
    "vae-decode-and-raw-stream",
    "conditioning-destroy",
    "dit-drain-and-destroy",
    "vae-drain-and-destroy",
    "normalize-encode-and-durable-wav-commit",
  ]) requireEvent(events, operation);
  const raw = requireEvent(events, "vae-decode-raw-scan-and-opfs-write");
  const wav = requireEvent(events, "bounded-wav-normalize-encode-write");
  if (
    raw.details["schema"] !== "ace-vae-raw-stream-capture-v1" ||
    raw.details["outputBytes"] !== OPT_0064_OUTPUT_DATA_BYTES ||
    wav.details["schema"] !== "ace-vae-wav-write-capture-v1" ||
    wav.details["rawReadBytes"] !== OPT_0064_OUTPUT_DATA_BYTES ||
    wav.details["wavDataBytes"] !== OPT_0064_OUTPUT_DATA_BYTES ||
    number(wav.details["maximumReadBufferBytes"]) > 131_072 ||
    number(wav.details["maximumScaledBufferBytes"]) > 131_072
  ) throw new Error("OPT-0064 bounded output finalization changed");
  for (const operation of [
    "audio-raw-finish-and-flush",
    "audio-raw-remove",
    "audio-wav-snapshot-validation",
    "audio-durable-publish-markers",
  ]) requireEvent(events, operation);
  return Object.freeze({
    schema: "ace-opt-0064-capture-summary-v1",
    eventCount: events.length,
    cacheAuthenticationEventCount: cacheAuthentication.length,
    proofReuseEventCount: proofReuse.length,
    acquisitionPlanCount: acquisitionPlans.length,
    phaseUploads: Object.freeze(phaseRows),
    aggregateUpload: Object.freeze(totals),
    construction: Object.freeze(construction),
    execution: Object.freeze({
      conditioningMs: requireEvent(events, "conditioning-execution").wallMs,
      ditMs: requireEvent(
        events,
        "dit-eight-evaluation-execution-and-readback",
      ).wallMs,
      vaeDecodeAndRawStreamMs:
        requireEvent(events, "vae-decode-and-raw-stream").wallMs,
    }),
    finalization: Object.freeze({
      rawStreamMs: raw.wallMs,
      wavWriteMs: wav.wallMs,
      durableCommitMs:
        requireEvent(events, "normalize-encode-and-durable-wav-commit").wallMs,
    }),
  });
}

function aggregateUploads(
  uploads: readonly AceOpt0064CaptureEvent[],
): Readonly<{
  fileCount: number;
  bytes: number;
  queueDrains: number;
  queueEmptyGaps: number;
  timing: Readonly<Record<string, number>>;
}> {
  const timingKeys = [
    "createBufferMs",
    "streamReadMs",
    "ownedCopyMs",
    "incrementalHashMs",
    "writeBufferMs",
    "queueDrainMs",
    "queueEmptyGapMs",
    "errorScopeMs",
  ] as const;
  const timing: Record<string, number> = Object.create(null) as
    Record<string, number>;
  for (const key of timingKeys) timing[key] = 0;
  let bytes = 0;
  let queueDrains = 0;
  let queueEmptyGaps = 0;
  let fileWallMs = 0;
  for (const event of uploads) {
    bytes += number(event.details["uploadedBytes"]);
    queueDrains += number(event.details["queueDrainCount"]);
    queueEmptyGaps += number(event.details["queueEmptyGapCount"]);
    fileWallMs += event.wallMs;
    const detailTiming = record(event.details["timing"]);
    for (const key of timingKeys) timing[key]! += number(detailTiming[key]);
  }
  timing["fileWallMs"] = fileWallMs;
  return Object.freeze({
    fileCount: uploads.length,
    bytes,
    queueDrains,
    queueEmptyGaps,
    timing: Object.freeze(timing),
  });
}

function uploadPhase(
  details: Readonly<Record<string, unknown>>,
): Opt0064UploadPhaseId {
  const phases = details["phases"];
  if (!Array.isArray(phases)) throw new Error("OPT-0064 upload phase is absent");
  if (phases.length === 1 && phases[0] === "text") return "text";
  if (
    phases.length === 2 && phases[0] === "conditioner" &&
    phases[1] === "constants"
  ) return "conditioner-constants";
  if (phases.length === 1 && phases[0] === "dit") return "dit";
  if (phases.length === 1 && phases[0] === "vae") return "vae";
  throw new Error(`OPT-0064 unexpected upload phases ${JSON.stringify(phases)}`);
}

function methodCount(
  details: Readonly<Record<string, unknown>>,
  method: string,
): number {
  const methods = record(details["methods"]);
  const value = methods[method];
  return value === undefined ? 0 : number(record(value)["count"]);
}

function requireEvent(
  events: readonly AceOpt0064CaptureEvent[],
  operation: string,
): AceOpt0064CaptureEvent {
  const matches = events.filter((event) => event.operation === operation);
  if (matches.length !== 1 || matches[0]!.details["outcome"] === "failure") {
    throw new Error(`OPT-0064 expected one successful ${operation} event`);
  }
  return matches[0]!;
}

async function hashBlob(blob: Blob): Promise<Readonly<{
  sha256: string;
  byteLength: number;
  streamReadCount: number;
  maximumStreamChunkBytes: number;
}>> {
  const reader = blob.stream().getReader();
  const hash = new AceIncrementalSha256();
  let byteLength = 0;
  let streamReadCount = 0;
  let maximumStreamChunkBytes = 0;
  try {
    while (true) {
      const item = await reader.read();
      if (item.done) break;
      streamReadCount += 1;
      byteLength += item.value.byteLength;
      maximumStreamChunkBytes = Math.max(
        maximumStreamChunkBytes,
        item.value.byteLength,
      );
      hash.update(item.value);
    }
  } finally {
    reader.releaseLock();
  }
  return Object.freeze({
    sha256: hash.digestHex(),
    byteLength,
    streamReadCount,
    maximumStreamChunkBytes,
  });
}

function validateRequest(): AceGenerationRequest {
  const bytes = new TextEncoder().encode(OPT_0064_REQUEST_CANONICAL_JSON);
  if (
    bytes.byteLength !== OPT_0064_REQUEST_BYTE_LENGTH ||
    aceSha256Hex(bytes) !== OPT_0064_REQUEST_SHA256
  ) throw new Error("OPT-0064 canonical direct request identity changed");
  const value = JSON.parse(OPT_0064_REQUEST_CANONICAL_JSON) as
    AceGenerationRequest;
  assertAceGenerationRequest(value);
  if (JSON.stringify(value) !== OPT_0064_REQUEST_CANONICAL_JSON) {
    throw new Error("OPT-0064 canonical direct request values changed");
  }
  return Object.freeze(value);
}

function validateResult(result: AceGenerationResult): void {
  if (
    result.mimeType !== "audio/wav" || result.sampleRateHz !== 48_000 ||
    result.channelCount !== 2 || result.frameCount !== OPT_0064_OUTPUT_FRAMES ||
    result.durationSeconds !== 12 || result.seed !== "0000000000c0ffee" ||
    result.generationProfile !== "ace-turbo-v1-correctness" ||
    result.modelManifestSha256 !== MAIN_MANIFEST_SHA256 ||
    result.audio.size !== OPT_0064_OUTPUT_WAV_BYTES
  ) throw new Error("OPT-0064 generated result envelope changed");
}

function validateDiagnostics(diagnostics: AceRuntimeDiagnostics): void {
  const capabilities = diagnostics.capabilities;
  if (
    diagnostics.modelManifestSha256 !== MAIN_MANIFEST_SHA256 ||
    diagnostics.ditDenseManifestSha256 !== DENSE_MANIFEST_SHA256 ||
    diagnostics.ditDenseRuntimeProfile !== DENSE_RUNTIME_PROFILE ||
    diagnostics.vaeManifestSha256 !== VAE_MANIFEST_SHA256 ||
    diagnostics.vaeRuntimeProfile !== VAE_RUNTIME_PROFILE ||
    diagnostics.vaeMaxWindowFrames !== 512 ||
    diagnostics.executionProfile.id !== "reference-bf16-subgroups" ||
    diagnostics.schedulingProfile !== "cooperative" ||
    capabilities.adapterInfo.subgroupMinSize !== 32 ||
    capabilities.adapterInfo.subgroupMaxSize !== 32 ||
    !capabilities.deviceFeatures.includes("shader-f16") ||
    !capabilities.deviceFeatures.includes("subgroups")
  ) throw new Error("OPT-0064 authenticated production runtime changed");
}

function productionConfiguration(): AceWorkerConfiguration {
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
    vaePackage: Object.freeze({
      manifestUrl: absoluteUrl(VAE_MANIFEST_PATH),
      manifestSha256: VAE_MANIFEST_SHA256,
      runtimeProfile: VAE_RUNTIME_PROFILE,
      maxWindowFrames: 512,
    }),
  });
}

function record(value: unknown): Readonly<Record<string, unknown>> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error("OPT-0064 capture detail is malformed");
  }
  return value as Readonly<Record<string, unknown>>;
}

function number(value: unknown): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
    throw new Error("OPT-0064 capture number is malformed");
  }
  return value;
}

function absoluteUrl(path: string): string {
  return new URL(path, self.location.href).href;
}

function postProgress(message: string): void {
  self.postMessage({ type: "progress", message });
}

function createFailureEvidence(input: Readonly<{
  primaryFailurePhase: string | undefined;
  activePhase: string;
  initializationProgressEventCount: number;
  initializationStages: ReadonlySet<string>;
  lastInitializationProgress: AceInitializationProgress | undefined;
  generationProgressEventCount: number;
  generationStages: ReadonlySet<string>;
  lastGenerationProgress: AceGenerationProgress | undefined;
  diagnosticCodes: readonly string[];
  fatalDiagnosticCodes: readonly string[];
  lastDiagnostic: AceDiagnostic | undefined;
  events: readonly AceOpt0064CaptureEvent[];
  captureSummary: Readonly<Record<string, unknown>> | undefined;
  outputHash: Awaited<ReturnType<typeof hashBlob>> | undefined;
  releaseError: unknown;
  disposeError: unknown;
  cleanupCompletedAtEpochMilliseconds: number;
}>): Readonly<Record<string, unknown>> {
  const tail = input.events.slice(-16).map((event) => Object.freeze({
    scope: event.scope,
    category: event.category,
    operation: event.operation,
    outcome: event.details["outcome"],
    wallMs: event.wallMs,
  }));
  const failedOperations = input.events.filter((event) =>
    event.details["outcome"] === "failure"
  ).map((event) => event.operation);
  return Object.freeze({
    schema: "ace-opt-0064-worker-failure-evidence-v1",
    primaryFailurePhase: input.primaryFailurePhase ?? "pre-run",
    activePhaseAtFailureSerialization: input.activePhase,
    modelSource: "cache-only",
    initializationProgressEventCount: input.initializationProgressEventCount,
    initializationStages: Object.freeze([...input.initializationStages]),
    ...(input.lastInitializationProgress === undefined
      ? {}
      : { lastInitializationProgress: input.lastInitializationProgress }),
    generationProgressEventCount: input.generationProgressEventCount,
    generationStages: Object.freeze([...input.generationStages]),
    ...(input.lastGenerationProgress === undefined
      ? {}
      : { lastGenerationProgress: input.lastGenerationProgress }),
    diagnosticCodes: Object.freeze([...input.diagnosticCodes]),
    fatalDiagnosticCodes: Object.freeze([...input.fatalDiagnosticCodes]),
    ...(input.lastDiagnostic === undefined
      ? {}
      : { lastDiagnostic: input.lastDiagnostic }),
    captureEventCount: input.events.length,
    captureTail: Object.freeze(tail),
    failedCaptureOperations: Object.freeze(failedOperations),
    captureSummary: input.captureSummary ?? null,
    outputIdentity: Object.freeze({
      configuredVaeManifestSha256: VAE_MANIFEST_SHA256,
      configuredVaeRuntimeProfile: VAE_RUNTIME_PROFILE,
      expected: Object.freeze({
        authority: OPT_0064_ACCEPTED_WAV_AUTHORITY,
        sha256: OPT_0064_ACCEPTED_WAV_SHA256,
        byteLength: OPT_0064_OUTPUT_WAV_BYTES,
      }),
      actual: input.outputHash === undefined
        ? null
        : Object.freeze({ ...input.outputHash }),
      sha256Matches: input.outputHash === undefined
        ? null
        : input.outputHash.sha256 === OPT_0064_ACCEPTED_WAV_SHA256,
      byteLengthMatches: input.outputHash === undefined
        ? null
        : input.outputHash.byteLength === OPT_0064_OUTPUT_WAV_BYTES,
    }),
    resultReleaseFailed: input.releaseError !== undefined,
    backendDisposeFailed: input.disposeError !== undefined,
    cleanupCompletedAtEpochMilliseconds:
      input.cleanupCompletedAtEpochMilliseconds,
  });
}

function fail(error: unknown): void {
  if (state === "settled") return;
  state = "settled";
  self.postMessage({
    type: "failed",
    error: Object.freeze({
      ...serializeOpt0064Failure(error),
      ...(failureEvidence === undefined ? {} : { evidence: failureEvidence }),
    }),
  });
}
