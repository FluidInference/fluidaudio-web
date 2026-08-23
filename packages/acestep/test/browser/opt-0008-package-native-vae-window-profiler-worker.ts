/// <reference lib="webworker" />

import {
  createAceOpt0008VaeWindowAttribution,
  summarizeAceOpt0008VaeWindowTrace,
  validateAceOpt0008VaeWindowTrace,
  type AceOpt0008VaeDecoderBatchTrace,
  type AceOpt0008VaeReadbackTrace,
  type AceOpt0008VaeWindowTrace,
} from "../../benchmark/opt-0008-vae-window-profiler.js";
import { acquireAceModelFiles } from "../../src/model/acquire.js";
import { AceOpfsModelCache } from "../../src/model/cache.js";
import { AceGpuTensorPhase } from "../../src/model/gpu-tensors.js";
import {
  ACE_PACKAGE_CONVERTER_REVISION,
  resolveAceLogicalTensor,
  type AcePackageFileRecord,
  type AcePackageManifest,
} from "../../src/model/manifest.js";
import { loadAcePackageManifest } from "../../src/model/package.js";
import {
  requestAceWebGpuDevice,
  type AceGpuRuntimeEvent,
  type AceWebGpuDeviceContext,
} from "../../src/webgpu/device.js";
import {
  ACE_VAE_DECODER_QUANTA_PER_COMMAND_BUFFER,
  AceVaeChunkGpuBackend,
  type AceVaeChunkGpuBackendMemoryPlan,
  type AceVaeChunkGpuBackendProgress,
} from "../../src/webgpu/vae-backend.js";
import {
  planAceVaeChunkedDecode,
  type AceVaeChunkedDecodePlan,
} from "../../src/webgpu/vae-chunks.js";
import {
  planAceVaeDecoderQuanta,
  type AceVaeDecoderCooperativePlan,
  type AceVaeTransposePartGeometry,
} from "../../src/webgpu/vae-decoder.js";
import type {
  Opt0008RunIdentity,
  Opt0008ThermalGateMetadata,
} from "./opt-0008-package-native-vae-window-profiler.js";

export const OPT_0008_REFERENCE_MANIFEST_SHA256 =
  "18f36c6420976475af65ecd833ca56c6119706322ce54120389d4915d8e80db6";
export const OPT_0008_REFERENCE_MANIFEST_PATH =
  "/model/files-reference/manifest.json";
export const OPT_0008_LATENT_FRAMES = 256;
export const OPT_0008_LATENT_CHANNELS = 64;
export const OPT_0008_LATENT_ELEMENTS = 16_384;
export const OPT_0008_OUTPUT_ELEMENTS = 983_040;
export const OPT_0008_OUTPUT_BYTES = 3_932_160;
export const OPT_0008_VAE_TENSOR_COUNT = 146;
export const OPT_0008_VAE_FILE_COUNT = 8;
export const OPT_0008_VAE_RESIDENT_BYTES = 337_583_104;
export const OPT_0008_LOGICAL_QUANTUM_COUNT = 3_942;
export const OPT_0008_PRIMITIVE_DISPATCH_COUNT = 3_988;
export const OPT_0008_DECODER_BATCH_COUNT = 493;
export const OPT_0008_TOTAL_COMMAND_BUFFER_COUNT = 494;
export const OPT_0008_REQUESTED_IDLE_MILLISECONDS = 493;
export const OPT_0008_TRACKED_BUFFER_COUNT = 15;
export const OPT_0008_OUTPUT_SENTINEL_BITS = 0x7fc0_0000;
export const OPT_0008_DETERMINISTIC_LATENT_SEED = 0x6d2b79f5;
export const OPT_0008_EXPECTED_PRODUCTION_COMMIT =
  "9dbd6e9cb85da211aa9e8224edfc08a2eef3f706";

const STATUS_UPDATE_INTERVAL_MILLISECONDS = 200;
const WORKER_HEARTBEAT_INTERVAL_MILLISECONDS = 10;

interface InitializeMessage {
  readonly type: "initialize";
  readonly identity: Opt0008RunIdentity;
}

interface RunTimedMessage {
  readonly type: "run-timed";
  readonly thermal: Opt0008ThermalGateMetadata;
}

type IncomingMessage = InitializeMessage | RunTimedMessage;

interface OutputSummary {
  readonly elementCount: number;
  readonly byteLength: number;
  readonly finiteCount: number;
  readonly nonzeroCount: number;
  readonly sentinelBitCount: number;
  readonly minimum: number;
  readonly maximum: number;
  readonly sha256: string;
}

interface WorkerHeartbeatSnapshot {
  readonly startedAtEpochMilliseconds: number;
  readonly completedAtEpochMilliseconds: number;
  readonly timerTickCount: number;
  readonly maximumTimerGapMilliseconds: number;
}

interface PhaseProgressSnapshot {
  readonly eventCount: number;
  readonly phaseFileIndex: number;
  readonly phaseFileCount: number;
  readonly loadedPhaseBytes: number;
  readonly totalPhaseBytes: number;
}

interface PreparedPackage {
  readonly manifest: AcePackageManifest;
  readonly manifestUrl: string;
  readonly acquiredFiles: ReadonlyMap<string, File>;
  readonly summary: Readonly<{
    manifestSha256: string;
    manifestByteLength: number;
    converterRevision: number;
    vaeTensorCount: number;
    vaeFileCount: number;
    vaeResidentBytes: number;
    acquiredFileCount: number;
    cachedFileCount: number;
    downloadedFileCount: number;
  }>;
}

interface WarmupSummary {
  readonly output: Float32Array;
  readonly outputSummary: OutputSummary;
  readonly wallMilliseconds: number;
  readonly progressEventCount: number;
  readonly prefillWallMilliseconds: number;
}

interface PreparedSession {
  readonly runIdentity: Opt0008RunIdentity;
  readonly context: AceWebGpuDeviceContext;
  readonly runtimeEvents: AceGpuRuntimeEvent[];
  readonly preparedPackage: PreparedPackage;
  readonly plan: AceVaeChunkedDecodePlan;
  readonly cooperativePlan: AceVaeDecoderCooperativePlan;
  readonly attribution: ReturnType<typeof createAceOpt0008VaeWindowAttribution>;
  readonly backend: AceVaeChunkGpuBackend;
  readonly observer: ProductionDeviceObserver;
  readonly progressRouter: ProgressRouter;
  readonly latent: Float32Array<ArrayBuffer>;
  readonly latentSha256: string;
  readonly memory: AceVaeChunkGpuBackendMemoryPlan;
  readonly phaseProgress: PhaseProgressSnapshot;
  readonly packageAcquisitionWallMilliseconds: number;
  readonly phaseUploadWallMilliseconds: number;
  readonly backendCompileWallMilliseconds: number;
  readonly warmup: WarmupSummary;
  readonly warmupCompletedAtEpochMilliseconds: number;
}

interface MutableCommandRecord {
  readonly kind: "decoder" | "readback";
  readonly batchIndex: number | null;
  readonly commandLabel: string;
  readonly passLabels: string[];
  readonly passDispatchCounts: number[];
  readonly encodeStartedAt: number;
  encodeEndedAt?: number;
  submitStartedAt?: number;
  submitReturnedAt?: number;
  drainStartedAt?: number;
  drainEndedAt?: number;
  progressReportedAt?: number;
  nextCommandEncodeStartedAt?: number;
  decodeResolvedAt?: number;
}

interface ResourceRecord {
  readonly label: string;
  readonly size: number;
  destroyCallCount: number;
  destroyed: boolean;
}

interface CancellationSummary {
  readonly rejectionName: string;
  readonly rejectionMessage: string;
  readonly progressEventCount: number;
  readonly completedDecoderQuanta: number;
  readonly encodedCommandBufferCount: number;
  readonly submissionCount: number;
  readonly queueDrainCount: number;
  readonly requestedIdleMilliseconds: number;
  readonly completedIdleCount: number;
  readonly firstBatchEncodeMilliseconds: number;
  readonly firstBatchSubmitThroughDrainMilliseconds: number;
  readonly firstBatchPostDrainToRejectionMilliseconds: number;
  readonly laterBatchEncodingPrevented: boolean;
  readonly laterBatchSubmissionPrevented: boolean;
  readonly firstBatchFullyDrained: boolean;
  readonly realIdleCompletedBeforeRejection: boolean;
}

let lifecycle: "idle" | "preparing" | "ready" | "running" | "settled" =
  "idle";
let session: PreparedSession | undefined;
let workerHeartbeat: ReturnType<typeof startWorkerHeartbeat> | undefined;

self.addEventListener("message", (event: MessageEvent<IncomingMessage>) => {
  if (event.data.type === "initialize") {
    if (lifecycle !== "idle") return;
    lifecycle = "preparing";
    workerHeartbeat = startWorkerHeartbeat();
    void initializeSession(event.data.identity).then(
      (prepared) => {
        if (lifecycle !== "preparing") return;
        session = prepared;
        lifecycle = "ready";
        self.postMessage({
          type: "ready-for-thermal-gate",
          warmupCompletedAtEpochMilliseconds:
            prepared.warmupCompletedAtEpochMilliseconds,
          preparation: publicPreparationSummary(prepared),
        });
      },
      (error: unknown) => void failAndCleanup(error),
    );
    return;
  }
  if (event.data.type === "run-timed" && lifecycle === "ready") {
    lifecycle = "running";
    const active = session!;
    session = undefined;
    void runTimedAndCleanup(active, event.data.thermal).then(
      (result) => {
        lifecycle = "settled";
        self.postMessage({ type: "passed", result });
      },
      (error: unknown) => void failAndCleanup(error, active),
    );
  }
});

async function initializeSession(
  identity: unknown,
): Promise<PreparedSession> {
  const runIdentity = validateRunIdentity(identity);
  postProgress("authenticating the converter-revision-4 reference manifest");
  const acquisitionStarted = performance.now();
  const preparedPackage = await preparePackage();
  const packageAcquisitionWallMilliseconds =
    performance.now() - acquisitionStarted;
  const runtimeEvents: AceGpuRuntimeEvent[] = [];
  postProgress("requesting the shipped cooperative WebGPU device");
  const context = await requestAceWebGpuDevice({
    modelProfile: "reference-bf16",
    schedulingProfile: "cooperative",
    onRuntimeEvent: (event) => runtimeEvents.push(event),
  });
  let phase: AceGpuTensorPhase | undefined;
  let backend: AceVaeChunkGpuBackend | undefined;
  try {
    const plan = validateProductionPlan(planAceVaeChunkedDecode(
      OPT_0008_LATENT_FRAMES,
    ));
    const transposeParts = transposeGeometry(
      plan,
      preparedPackage.manifest,
    );
    const cooperativePlan = planAceVaeDecoderQuanta(
      plan.decoderWorkspacePlan,
      transposeParts,
    );
    validateCooperativePlan(cooperativePlan);
    const attribution = createAceOpt0008VaeWindowAttribution({
      graph: plan.decoderWorkspacePlan,
      cooperativePlan,
      limits: context.device.limits,
      quantaPerCommandBuffer: ACE_VAE_DECODER_QUANTA_PER_COMMAND_BUFFER,
    });
    const observer = new ProductionDeviceObserver(context.device);
    let phaseProgressEventCount = 0;
    let phaseProgress: PhaseProgressSnapshot | undefined;
    let lastPhaseStatusAt = 0;
    postProgress("uploading eight authenticated VAE shards");
    const phaseStarted = performance.now();
    phase = await AceGpuTensorPhase.load(
      observer.device,
      preparedPackage.manifest,
      preparedPackage.acquiredFiles,
      ["vae"],
      {
        onProgress: (event) => {
          phaseProgressEventCount += 1;
          phaseProgress = Object.freeze({
            eventCount: phaseProgressEventCount,
            phaseFileIndex: event.phaseFileIndex,
            phaseFileCount: event.phaseFileCount,
            loadedPhaseBytes: event.loadedPhaseBytes,
            totalPhaseBytes: event.totalPhaseBytes,
          });
          const now = performance.now();
          if (
            now - lastPhaseStatusAt >= STATUS_UPDATE_INTERVAL_MILLISECONDS ||
            event.loadedPhaseBytes === event.totalPhaseBytes
          ) {
            lastPhaseStatusAt = now;
            postProgress(
              `VAE GPU upload ${event.phaseFileIndex + 1}/${event.phaseFileCount} ` +
                `(${formatBytes(event.loadedPhaseBytes)}/` +
                `${formatBytes(event.totalPhaseBytes)})`,
            );
          }
        },
      },
    );
    const phaseUploadWallMilliseconds = performance.now() - phaseStarted;
    if (
      phaseProgress === undefined ||
      phase.phases.length !== 1 || phase.phases[0] !== "vae" ||
      phase.packageManifest !== preparedPackage.manifest ||
      phase.residentBytes !== OPT_0008_VAE_RESIDENT_BYTES ||
      phaseProgress.phaseFileCount !== OPT_0008_VAE_FILE_COUNT ||
      phaseProgress.loadedPhaseBytes !== OPT_0008_VAE_RESIDENT_BYTES ||
      phaseProgress.totalPhaseBytes !== OPT_0008_VAE_RESIDENT_BYTES
    ) throw new Error("OPT-0008 VAE phase upload accounting diverged");

    const latent = createDeterministicLatent();
    const latentSha256 = await sha256Hex(bytesOf(latent));
    const progressRouter = new ProgressRouter(observer);
    postProgress("compiling and allocating the shipped production VAE backend");
    const compileStarted = performance.now();
    const transferredPhase = phase;
    phase = undefined;
    backend = await AceVaeChunkGpuBackend.create({
      device: observer.device,
      plan,
      finalLatents: latent,
      ownedVaeWeights: transferredPhase,
      onProgress: (event) => progressRouter.accept(event),
    });
    const backendCompileWallMilliseconds = performance.now() - compileStarted;
    validateMemory(backend.memory);
    observer.requireOutputBuffer();

    postProgress("qNaN-prefilling and running one untimed symmetric warmup");
    const warmupPrefill = await prefillCompleteOutput(
      context.device.queue,
      observer.requireOutputBuffer(),
    );
    progressRouter.begin("warmup");
    const warmupStarted = performance.now();
    const warmupOutput = await backend.decodeWindow(plan.windows[0]!);
    const warmupWallMilliseconds = performance.now() - warmupStarted;
    const warmupProgress = progressRouter.end("warmup");
    validateProductionProgress(warmupProgress);
    const warmupOutputSummary = await summarizeOutput(warmupOutput);
    validateCompleteOutput(warmupOutputSummary, "warmup");
    if (runtimeEvents.length !== 0) {
      throw new Error("WebGPU emitted an event during OPT-0008 preparation");
    }
    const warmupCompletedAtEpochMilliseconds = Date.now();
    const stableBackend = backend;
    backend = undefined;
    return Object.freeze({
      runIdentity,
      context,
      runtimeEvents,
      preparedPackage,
      plan,
      cooperativePlan,
      attribution,
      backend: stableBackend,
      observer,
      progressRouter,
      latent,
      latentSha256,
      memory: stableBackend.memory,
      phaseProgress,
      packageAcquisitionWallMilliseconds,
      phaseUploadWallMilliseconds,
      backendCompileWallMilliseconds,
      warmup: Object.freeze({
        output: warmupOutput,
        outputSummary: warmupOutputSummary,
        wallMilliseconds: warmupWallMilliseconds,
        progressEventCount: warmupProgress.length,
        prefillWallMilliseconds: warmupPrefill,
      }),
      warmupCompletedAtEpochMilliseconds,
    });
  } catch (error) {
    await backend?.destroy();
    phase?.destroy();
    context.destroy();
    throw error;
  }
}

async function runTimedAndCleanup(
  prepared: PreparedSession,
  thermal: Opt0008ThermalGateMetadata,
): Promise<Readonly<Record<string, unknown>>> {
  let destroyed = false;
  const progressRouter = prepared.progressRouter;
  try {
    validateThermalGate(
      thermal,
      prepared.warmupCompletedAtEpochMilliseconds,
    );
    if (prepared.runtimeEvents.length !== 0) {
      throw new Error("WebGPU emitted an event before the timed OPT-0008 run");
    }
    postProgress("qNaN-prefilling the complete timed output outside timing");
    const timedPrefillWallMilliseconds = await prefillCompleteOutput(
      prepared.context.device.queue,
      prepared.observer.requireOutputBuffer(),
    );
    progressRouter.begin("timed");
    prepared.observer.beginTimedTrace();
    postProgress("timing 493 unchanged batch-eight decoder command buffers");
    const timedOutput = await prepared.backend.decodeWindow(
      prepared.plan.windows[0]!,
    );
    const decodeResolvedAt = performance.now();
    const timedProgress = progressRouter.end("timed");
    const trace = prepared.observer.endTimedTrace(decodeResolvedAt);
    const timedStartedAtEpochMilliseconds = performance.timeOrigin +
      trace.decoderBatches[0]!.encodeStartedAt;
    const timedCompletedAtEpochMilliseconds = performance.timeOrigin +
      trace.readback.decodeResolvedAt;
    const preGateToTimedStartMilliseconds =
      timedStartedAtEpochMilliseconds - thermal.completedAtEpochMilliseconds;
    if (preGateToTimedStartMilliseconds < 0) {
      throw new Error("OPT-0008 timed work began before its thermal pre-gate");
    }
    validateProductionProgress(timedProgress);
    const actualCommandTags = prepared.observer.actualPassLabels();
    validateActualPassLabels(prepared.attribution, trace, actualCommandTags);
    validateAceOpt0008VaeWindowTrace(prepared.attribution, trace);
    const attributionSummary = summarizeAceOpt0008VaeWindowTrace(
      prepared.attribution,
      trace,
    );

    const timedOutputSummary = await summarizeOutput(timedOutput);
    validateCompleteOutput(timedOutputSummary, "timed");
    const outputComparison = compareOutputs(
      prepared.warmup.output,
      timedOutput,
    );
    if (outputComparison.bitMismatchCount !== 0) {
      throw new Error(
        `OPT-0008 warmup/timed output diverged at ` +
          `${outputComparison.bitMismatchCount} U32 words`,
      );
    }

    const cleanupStartedAtEpochMilliseconds = Date.now();
    postProgress("running the untimed post-drain one-batch cancellation proof");
    const cancellation = await runCancellationProof(
      prepared.backend,
      prepared.plan,
      progressRouter,
      prepared.observer,
    );
    const destroyStarted = performance.now();
    await prepared.backend.destroy();
    const destroyWallMilliseconds = performance.now() - destroyStarted;
    destroyed = true;
    const postDestroy = await verifyPostDestroy(prepared.backend, prepared.plan);
    const resources = prepared.observer.resourceSummary();
    if (
      !resources.destructionTrackingSupported ||
      resources.createdBufferCount !== OPT_0008_TRACKED_BUFFER_COUNT ||
      resources.uniqueDestroyedBufferCount !== OPT_0008_TRACKED_BUFFER_COUNT ||
      resources.totalDestroyCallCount !== OPT_0008_TRACKED_BUFFER_COUNT ||
      resources.liveTrackedBufferCount !== 0
    ) throw new Error("OPT-0008 backend cleanup left tracked buffers live");
    if (prepared.runtimeEvents.length !== 0) {
      throw new Error("WebGPU emitted an event during OPT-0008 execution");
    }
    prepared.context.destroy();
    const heartbeat = workerHeartbeat?.stop();
    workerHeartbeat = undefined;
    if (heartbeat === undefined) {
      throw new Error("OPT-0008 worker heartbeat never started");
    }
    validateWorkerHeartbeat(heartbeat);
    const cleanupCompletedAtEpochMilliseconds = Date.now();
    return Object.freeze({
      schema: "ace-opt-0008-package-native-vae-window-profiler-v1",
      experimentId: "OPT-0008",
      identity: prepared.runIdentity,
      recordedAtEpochMilliseconds: timedStartedAtEpochMilliseconds,
      recordedAt: new Date(timedStartedAtEpochMilliseconds).toISOString(),
      status: "passed",
      allocationCommit:
        "9df51f38198f17c714e67d4a15cc382eb36af2cb",
      scope: Object.freeze({
        packageNative: true,
        shippedProductionBackend: true,
        latentFrames: OPT_0008_LATENT_FRAMES,
        decodedSeconds: 10.24,
        decoderQuantaPerCommandBuffer:
          ACE_VAE_DECODER_QUANTA_PER_COMMAND_BUFFER,
        topologyChanged: false,
        oneHeavyweightPhaseResident: true,
        warmupTimedAndCancellationReuseOneBackend: true,
        outputPrefillExcludedFromTiming: true,
        hashingExcludedFromTiming: true,
        fullSongExecuted: false,
      }),
      package: prepared.preparedPackage.summary,
      environment: Object.freeze({
        userAgent: navigator.userAgent,
        executionProfile: prepared.context.capabilities.executionProfile,
        schedulingProfile: prepared.context.capabilities.schedulingProfile,
        adapterInfo: prepared.context.capabilities.adapterInfo,
        adapterFeatures: prepared.context.capabilities.adapterFeatures,
        deviceFeatures: prepared.context.capabilities.deviceFeatures,
        adapterLimits: prepared.context.capabilities.adapterLimits,
        deviceLimits: prepared.context.capabilities.deviceLimits,
        runtimeEvents: prepared.runtimeEvents,
      }),
      thermal,
      thermalClassification: Object.freeze({
        status: "pending-external-artifact-join",
        preGateOnly: true,
        browserReceiptClaimsPlanValidThermalCoverage: false,
        continuousLoggerRequiredFromEpochMilliseconds:
          thermal.startedAtEpochMilliseconds,
        continuousLoggerRequiredThroughEpochMilliseconds:
          cleanupCompletedAtEpochMilliseconds,
      }),
      timedInterval: Object.freeze({
        startedAtEpochMilliseconds: timedStartedAtEpochMilliseconds,
        completedAtEpochMilliseconds: timedCompletedAtEpochMilliseconds,
        startedAt: new Date(timedStartedAtEpochMilliseconds).toISOString(),
        completedAt: new Date(timedCompletedAtEpochMilliseconds).toISOString(),
        preGateToTimedStartMilliseconds,
        prefillWallMilliseconds: timedPrefillWallMilliseconds,
        prefillExcludedFromTiming: true,
      }),
      timingSemantics: Object.freeze({
        actualIdleMilliseconds:
          "fenced drain-end-to-next-encode interval; includes progress callback " +
          "and scheduler overhead around the real queue-empty timer",
        rawTimestampClock: "worker-performance-now-monotonic",
        epochCorrelation: "performance.timeOrigin + performance.now",
      }),
      deterministicLatent: Object.freeze({
        seed: OPT_0008_DETERMINISTIC_LATENT_SEED,
        recipe: "lcg-u32-1664525-1013904223-high24-to-symmetric-f32-v1",
        elementCount: prepared.latent.length,
        byteLength: prepared.latent.byteLength,
        sha256: prepared.latentSha256,
      }),
      preparation: publicPreparationSummary(prepared),
      productionPlan: Object.freeze({
        logicalQuantumCount: prepared.cooperativePlan.quantumCount,
        primitiveDispatchCount: prepared.cooperativePlan.primitiveDispatchCount,
        decoderBatchCount: OPT_0008_DECODER_BATCH_COUNT,
        commandBufferCount: OPT_0008_TOTAL_COMMAND_BUFFER_COUNT,
        submissionCount: OPT_0008_TOTAL_COMMAND_BUFFER_COUNT,
        queueDrainCount: OPT_0008_TOTAL_COMMAND_BUFFER_COUNT,
        requestedIdleMilliseconds: OPT_0008_REQUESTED_IDLE_MILLISECONDS,
        progressEventCount: timedProgress.length,
      }),
      rawTrace: trace,
      attributionPlan: prepared.attribution,
      actualCommandTags,
      attribution: attributionSummary,
      output: Object.freeze({
        warmup: prepared.warmup.outputSummary,
        timed: timedOutputSummary,
        comparison: outputComparison,
      }),
      cancellation,
      cleanup: Object.freeze({
        startedAtEpochMilliseconds: cleanupStartedAtEpochMilliseconds,
        completedAtEpochMilliseconds: cleanupCompletedAtEpochMilliseconds,
        startedAt: new Date(cleanupStartedAtEpochMilliseconds).toISOString(),
        completedAt: new Date(cleanupCompletedAtEpochMilliseconds).toISOString(),
        destroyWallMilliseconds,
        resources,
        postDestroy,
        deviceDestroyedAfterEventCheck: true,
      }),
      workerHeartbeat: heartbeat,
      completedAtEpochMilliseconds: cleanupCompletedAtEpochMilliseconds,
      completedAt: new Date(cleanupCompletedAtEpochMilliseconds).toISOString(),
    });
  } finally {
    try {
      if (!destroyed) await prepared.backend.destroy();
    } finally {
      prepared.context.destroy();
    }
  }
}

class ProgressRouter {
  private mode: "none" | "warmup" | "timed" | "cancellation" = "none";
  private events: AceVaeChunkGpuBackendProgress[] = [];
  private hook:
    | ((event: AceVaeChunkGpuBackendProgress) => void)
    | undefined;

  constructor(private readonly observer: ProductionDeviceObserver) {}

  begin(
    mode: Exclude<ProgressRouter["mode"], "none">,
    hook?: (event: AceVaeChunkGpuBackendProgress) => void,
  ): void {
    if (this.mode !== "none") {
      throw new DOMException("OPT-0008 progress collection overlaps", "InvalidStateError");
    }
    this.mode = mode;
    this.events = [];
    this.hook = hook;
  }

  accept(event: AceVaeChunkGpuBackendProgress): void {
    if (this.mode === "none") {
      throw new Error("OPT-0008 received VAE progress outside an active run");
    }
    const snapshot = Object.freeze({ ...event });
    this.events.push(snapshot);
    if (this.mode === "timed" || this.mode === "cancellation") {
      this.observer.noteProgress(snapshot);
    }
    this.hook?.(snapshot);
  }

  end(expectedMode: Exclude<ProgressRouter["mode"], "none">): readonly AceVaeChunkGpuBackendProgress[] {
    if (this.mode !== expectedMode) {
      throw new DOMException(
        `OPT-0008 expected ${expectedMode} progress, got ${this.mode}`,
        "InvalidStateError",
      );
    }
    const snapshot = Object.freeze([...this.events]);
    this.mode = "none";
    this.events = [];
    this.hook = undefined;
    return snapshot;
  }

  abandon(): void {
    this.mode = "none";
    this.events = [];
    this.hook = undefined;
  }
}

class ProductionDeviceObserver {
  readonly device: GPUDevice;

  private readonly queue: GPUQueue;
  private readonly records: MutableCommandRecord[] = [];
  private readonly commandRecords = new WeakMap<GPUCommandBuffer, MutableCommandRecord>();
  private readonly resources = new Map<GPUBuffer, ResourceRecord>();
  private outputBuffer: GPUBuffer | undefined;
  private activeMode: "off" | "timed" | "cancellation" = "off";
  private pendingSubmission: MutableCommandRecord | undefined;
  private destructionTrackingSupported = true;
  private submissionCount = 0;
  private drainCount = 0;

  constructor(private readonly target: GPUDevice) {
    this.queue = this.createQueueProxy(target.queue);
    this.device = new Proxy(target, {
      get: (device, property) => {
        if (property === "queue") return this.queue;
        if (property === "createBuffer") {
          return (descriptor: GPUBufferDescriptor): GPUBuffer =>
            this.createTrackedBuffer(descriptor);
        }
        if (property === "createCommandEncoder") {
          return (descriptor?: GPUCommandEncoderDescriptor): GPUCommandEncoder =>
            this.createObservedCommandEncoder(descriptor);
        }
        const value = Reflect.get(device, property, device) as unknown;
        return typeof value === "function"
          ? (value as (...args: unknown[]) => unknown).bind(device)
          : value;
      },
    }) as GPUDevice;
  }

  beginTimedTrace(): void {
    this.beginTrace("timed");
  }

  beginCancellationTrace(): void {
    this.beginTrace("cancellation");
  }

  noteProgress(event: AceVaeChunkGpuBackendProgress): void {
    if (this.activeMode === "off") {
      throw new Error("OPT-0008 observed progress without an active trace");
    }
    const commandIndex = event.completedCommandBuffers - 1;
    const record = this.records[commandIndex];
    if (record === undefined) {
      throw new Error(
        `OPT-0008 progress references absent command ${commandIndex}`,
      );
    }
    const expectedKind = event.stage === "decoder" ? "decoder" : "readback";
    if (record.kind !== expectedKind) {
      throw new Error("OPT-0008 progress stage disagrees with command label");
    }
    record.progressReportedAt ??= performance.now();
  }

  endTimedTrace(decodeResolvedAt: number): AceOpt0008VaeWindowTrace {
    if (this.activeMode !== "timed") {
      throw new DOMException("OPT-0008 timed trace is not active", "InvalidStateError");
    }
    if (this.pendingSubmission !== undefined) {
      throw new Error("OPT-0008 timed trace ended with an undrained submission");
    }
    if (
      this.submissionCount !== this.records.length ||
      this.drainCount !== this.records.length
    ) throw new Error("OPT-0008 timed trace lost a submission or queue drain");
    const readback = this.records.at(-1);
    if (readback === undefined || readback.kind !== "readback") {
      throw new Error("OPT-0008 timed trace omitted its final readback");
    }
    readback.decodeResolvedAt = decodeResolvedAt;
    const decoderBatches = this.records.slice(0, -1).map(
      (record) => decoderBatchTrace(record),
    );
    const trace = Object.freeze({
      decoderBatches: Object.freeze(decoderBatches),
      readback: readbackTrace(readback),
    }) satisfies AceOpt0008VaeWindowTrace;
    this.activeMode = "off";
    return trace;
  }

  endCancellationTrace(): Readonly<{
    records: readonly MutableCommandRecord[];
    submissionCount: number;
    drainCount: number;
  }> {
    if (this.activeMode !== "cancellation") {
      throw new DOMException(
        "OPT-0008 cancellation trace is not active",
        "InvalidStateError",
      );
    }
    if (this.pendingSubmission !== undefined) {
      throw new Error("OPT-0008 cancellation returned before its queue drain");
    }
    this.activeMode = "off";
    return Object.freeze({
      records: Object.freeze([...this.records]),
      submissionCount: this.submissionCount,
      drainCount: this.drainCount,
    });
  }

  requireOutputBuffer(): GPUBuffer {
    if (
      this.outputBuffer === undefined ||
      this.outputBuffer.size !== OPT_0008_OUTPUT_BYTES
    ) {
      throw new Error("OPT-0008 did not capture the complete VAE output buffer");
    }
    return this.outputBuffer;
  }

  resourceSummary(): Readonly<{
    createdBufferCount: number;
    uniqueDestroyedBufferCount: number;
    totalDestroyCallCount: number;
    liveTrackedBufferCount: number;
    destructionTrackingSupported: boolean;
    outputBufferCaptured: boolean;
    records: readonly Readonly<ResourceRecord>[];
  }> {
    const records = [...this.resources.values()];
    return Object.freeze({
      createdBufferCount: records.length,
      uniqueDestroyedBufferCount: records.filter((record) => record.destroyed).length,
      totalDestroyCallCount: records.reduce(
        (total, record) => total + record.destroyCallCount,
        0,
      ),
      liveTrackedBufferCount: records.filter((record) => !record.destroyed).length,
      destructionTrackingSupported: this.destructionTrackingSupported,
      outputBufferCaptured: this.outputBuffer !== undefined,
      records: Object.freeze(records.map((record) => Object.freeze({ ...record }))),
    });
  }

  actualPassLabels(): readonly Readonly<{
    commandLabel: string;
    passLabels: readonly string[];
    passDispatchCounts: readonly number[];
  }>[] {
    return Object.freeze(this.records.map((record) => Object.freeze({
      commandLabel: record.commandLabel,
      passLabels: Object.freeze([...record.passLabels]),
      passDispatchCounts: Object.freeze([...record.passDispatchCounts]),
    })));
  }

  private beginTrace(mode: "timed" | "cancellation"): void {
    if (this.activeMode !== "off" || this.pendingSubmission !== undefined) {
      throw new DOMException("OPT-0008 command tracing overlaps", "InvalidStateError");
    }
    this.records.length = 0;
    this.submissionCount = 0;
    this.drainCount = 0;
    this.activeMode = mode;
  }

  private createTrackedBuffer(descriptor: GPUBufferDescriptor): GPUBuffer {
    const buffer = this.target.createBuffer(descriptor);
    const label = descriptor.label ?? "";
    const record: ResourceRecord = {
      label,
      size: Number(descriptor.size),
      destroyCallCount: 0,
      destroyed: false,
    };
    this.resources.set(buffer, record);
    if (label === "ace-vae-chunk-output") {
      if (this.outputBuffer !== undefined) {
        throw new Error("OPT-0008 observed duplicate VAE output allocations");
      }
      this.outputBuffer = buffer;
    }
    const destroy = buffer.destroy.bind(buffer);
    try {
      Object.defineProperty(buffer, "destroy", {
        configurable: true,
        value: (): void => {
          record.destroyCallCount += 1;
          record.destroyed = true;
          destroy();
        },
      });
    } catch {
      this.destructionTrackingSupported = false;
    }
    return buffer;
  }

  private createObservedCommandEncoder(
    descriptor?: GPUCommandEncoderDescriptor,
  ): GPUCommandEncoder {
    const encoder = this.target.createCommandEncoder(descriptor);
    if (this.activeMode === "off") return encoder;
    const commandLabel = descriptor?.label ?? "";
    const decoderMatch =
      /^ace-vae-window-0-batch-(\d+)$/.exec(commandLabel);
    const readback = commandLabel === "ace-vae-window-0-readback";
    if (decoderMatch === null && !readback) {
      throw new Error(`OPT-0008 traced unexpected command ${commandLabel}`);
    }
    const now = performance.now();
    const prior = this.records.at(-1);
    if (prior !== undefined) prior.nextCommandEncodeStartedAt = now;
    const record: MutableCommandRecord = {
      kind: readback ? "readback" : "decoder",
      batchIndex: readback ? null : Number(decoderMatch![1]),
      commandLabel,
      passLabels: [],
      passDispatchCounts: [],
      encodeStartedAt: now,
    };
    if (
      record.kind === "decoder" &&
      record.batchIndex !== this.records.length
    ) throw new Error("OPT-0008 production decoder batch order changed");
    this.records.push(record);
    return new Proxy(encoder, {
      get: (target, property) => {
        if (property === "beginComputePass") {
          return (passDescriptor?: GPUComputePassDescriptor): GPUComputePassEncoder => {
            record.passLabels.push(passDescriptor?.label ?? "");
            record.passDispatchCounts.push(0);
            const passIndex = record.passDispatchCounts.length - 1;
            const pass = target.beginComputePass(passDescriptor);
            return new Proxy(pass, {
              get: (passTarget, passProperty) => {
                if (passProperty === "dispatchWorkgroups") {
                  return (
                    workgroupCountX: GPUSize32,
                    workgroupCountY?: GPUSize32,
                    workgroupCountZ?: GPUSize32,
                  ): void => {
                    record.passDispatchCounts[passIndex] =
                      record.passDispatchCounts[passIndex]! + 1;
                    passTarget.dispatchWorkgroups(
                      workgroupCountX,
                      workgroupCountY,
                      workgroupCountZ,
                    );
                  };
                }
                const passValue = Reflect.get(
                  passTarget,
                  passProperty,
                  passTarget,
                ) as unknown;
                return typeof passValue === "function"
                  ? (passValue as (...args: unknown[]) => unknown).bind(passTarget)
                  : passValue;
              },
            }) as GPUComputePassEncoder;
          };
        }
        if (property === "finish") {
          return (finishDescriptor?: GPUCommandBufferDescriptor): GPUCommandBuffer => {
            if (record.encodeEndedAt !== undefined) {
              throw new Error("OPT-0008 command encoder finished more than once");
            }
            const commandBuffer = target.finish(finishDescriptor);
            record.encodeEndedAt = performance.now();
            this.commandRecords.set(commandBuffer, record);
            return commandBuffer;
          };
        }
        const value = Reflect.get(target, property, target) as unknown;
        return typeof value === "function"
          ? (value as (...args: unknown[]) => unknown).bind(target)
          : value;
      },
    }) as GPUCommandEncoder;
  }

  private createQueueProxy(target: GPUQueue): GPUQueue {
    return new Proxy(target, {
      get: (queue, property) => {
        if (property === "submit") {
          return (commandBuffers: Iterable<GPUCommandBuffer>): void => {
            const retained = [...commandBuffers];
            if (this.activeMode === "off") {
              queue.submit(retained);
              return;
            }
            if (retained.length !== 1 || this.pendingSubmission !== undefined) {
              throw new Error(
                "OPT-0008 production must retain one outstanding command buffer",
              );
            }
            const record = this.commandRecords.get(retained[0]!);
            if (record === undefined) {
              throw new Error("OPT-0008 submitted an unobserved command buffer");
            }
            record.submitStartedAt = performance.now();
            queue.submit(retained);
            record.submitReturnedAt = performance.now();
            this.submissionCount += 1;
            this.pendingSubmission = record;
          };
        }
        if (property === "onSubmittedWorkDone") {
          return async (): Promise<void> => {
            if (this.activeMode === "off") {
              await queue.onSubmittedWorkDone();
              return;
            }
            const record = this.pendingSubmission;
            if (record === undefined) {
              throw new Error("OPT-0008 drained without a pending submission");
            }
            record.drainStartedAt = performance.now();
            await queue.onSubmittedWorkDone();
            record.drainEndedAt = performance.now();
            this.drainCount += 1;
            this.pendingSubmission = undefined;
          };
        }
        const value = Reflect.get(queue, property, queue) as unknown;
        return typeof value === "function"
          ? (value as (...args: unknown[]) => unknown).bind(queue)
          : value;
      },
    }) as GPUQueue;
  }
}

function decoderBatchTrace(
  record: MutableCommandRecord,
): AceOpt0008VaeDecoderBatchTrace {
  if (record.kind !== "decoder" || record.batchIndex === null) {
    throw new Error("OPT-0008 expected a decoder batch trace");
  }
  return Object.freeze({
    batchIndex: record.batchIndex,
    encodeStartedAt: record.encodeStartedAt,
    encodeEndedAt: requiredTimestamp(record.encodeEndedAt, "encode end"),
    submitStartedAt: requiredTimestamp(record.submitStartedAt, "submit start"),
    submitReturnedAt: requiredTimestamp(record.submitReturnedAt, "submit return"),
    drainStartedAt: requiredTimestamp(record.drainStartedAt, "drain start"),
    drainEndedAt: requiredTimestamp(record.drainEndedAt, "drain end"),
    progressReportedAt: requiredTimestamp(
      record.progressReportedAt,
      "progress report",
    ),
    nextCommandEncodeStartedAt: requiredTimestamp(
      record.nextCommandEncodeStartedAt,
      "next command encode start",
    ),
    commandBufferCount: 1,
    submissionCount: 1,
    queueDrainCount: 1,
    requestedIdleMilliseconds: 1,
    completedIdleCount: 1,
  });
}

function readbackTrace(
  record: MutableCommandRecord,
): AceOpt0008VaeReadbackTrace {
  if (record.kind !== "readback") {
    throw new Error("OPT-0008 expected a readback trace");
  }
  return Object.freeze({
    encodeStartedAt: record.encodeStartedAt,
    encodeEndedAt: requiredTimestamp(record.encodeEndedAt, "readback encode end"),
    submitStartedAt: requiredTimestamp(
      record.submitStartedAt,
      "readback submit start",
    ),
    submitReturnedAt: requiredTimestamp(
      record.submitReturnedAt,
      "readback submit return",
    ),
    drainStartedAt: requiredTimestamp(
      record.drainStartedAt,
      "readback drain start",
    ),
    drainEndedAt: requiredTimestamp(record.drainEndedAt, "readback drain end"),
    progressReportedAt: requiredTimestamp(
      record.progressReportedAt,
      "readback progress report",
    ),
    decodeResolvedAt: requiredTimestamp(
      record.decodeResolvedAt,
      "decode resolution",
    ),
    commandBufferCount: 1,
    submissionCount: 1,
    queueDrainCount: 1,
    requestedIdleMilliseconds: 0,
    completedIdleCount: 0,
  });
}

function requiredTimestamp(value: number | undefined, label: string): number {
  if (value === undefined || !Number.isFinite(value) || value < 0) {
    throw new Error(`OPT-0008 trace omitted ${label}`);
  }
  return value;
}

async function preparePackage(): Promise<PreparedPackage> {
  const manifestUrl = new URL(
    OPT_0008_REFERENCE_MANIFEST_PATH,
    self.location.href,
  ).href;
  const loaded = await loadAcePackageManifest({
    manifestUrl,
    expectedManifestSha256: OPT_0008_REFERENCE_MANIFEST_SHA256,
    expectedProfile: "reference",
  });
  if (loaded.manifestSha256 !== OPT_0008_REFERENCE_MANIFEST_SHA256) {
    throw new Error("OPT-0008 manifest digest diverged from its trust root");
  }
  const inventory = validateVaeInventory(loaded.manifest);
  const acquisitionManifest = Object.freeze({
    ...loaded.manifest,
    files: inventory.files,
  });
  const cache = await AceOpfsModelCache.open();
  let lastStatusAt = 0;
  const acquired = await acquireAceModelFiles({
    manifest: acquisitionManifest,
    manifestUrl: loaded.manifestUrl,
    cache,
    onFileProgress: (progress) => {
      const now = performance.now();
      if (
        now - lastStatusAt >= STATUS_UPDATE_INTERVAL_MILLISECONDS ||
        progress.fileIndex + 1 === progress.fileCount &&
          progress.fileReceivedBytes === progress.fileBytes
      ) {
        lastStatusAt = now;
        postProgress(
          `acquiring VAE shard ${progress.fileIndex + 1}/${progress.fileCount} ` +
            `(${formatBytes(progress.completedBytes)}/` +
            `${formatBytes(progress.totalBytes)}, ${progress.source})`,
        );
      }
    },
  });
  if (
    acquired.files.size !== OPT_0008_VAE_FILE_COUNT ||
    acquired.plan.files.length !== OPT_0008_VAE_FILE_COUNT ||
    acquired.plan.runtimeBytes !== OPT_0008_VAE_RESIDENT_BYTES
  ) throw new Error("OPT-0008 bounded VAE acquisition accounting diverged");
  return Object.freeze({
    manifest: loaded.manifest,
    manifestUrl: loaded.manifestUrl,
    acquiredFiles: acquired.files,
    summary: Object.freeze({
      manifestSha256: loaded.manifestSha256,
      manifestByteLength: loaded.manifestByteLength,
      converterRevision: loaded.manifest.provenance.converterRevision,
      vaeTensorCount: inventory.tensorCount,
      vaeFileCount: inventory.files.length,
      vaeResidentBytes: inventory.residentBytes,
      acquiredFileCount: acquired.files.size,
      cachedFileCount: acquired.plan.cachedFiles.length,
      downloadedFileCount: acquired.plan.downloadFiles.length,
    }),
  });
}

function validateVaeInventory(manifest: AcePackageManifest): Readonly<{
  files: readonly AcePackageFileRecord[];
  tensorCount: number;
  residentBytes: number;
}> {
  if (
    manifest.provenance.converterRevision !== ACE_PACKAGE_CONVERTER_REVISION ||
    ACE_PACKAGE_CONVERTER_REVISION !== 4
  ) throw new Error("OPT-0008 requires converter revision 4");
  const tensors = Object.values(manifest.tensors).filter(
    (tensor) => tensor.phase === "vae",
  );
  const shardNames = new Set(tensors.map((tensor) => tensor.shard));
  const files = manifest.files.filter((file) => shardNames.has(file.name));
  const residentBytes = sumSafe(
    files.map((file) => file.byteLength),
    "OPT-0008 VAE resident bytes",
  );
  const tensorBytes = sumSafe(
    tensors.map((tensor) => tensor.byteLength),
    "OPT-0008 VAE tensor bytes",
  );
  if (
    tensors.length !== OPT_0008_VAE_TENSOR_COUNT ||
    shardNames.size !== OPT_0008_VAE_FILE_COUNT ||
    files.length !== OPT_0008_VAE_FILE_COUNT ||
    files.some((file) => file.kind !== "weights") ||
    residentBytes !== OPT_0008_VAE_RESIDENT_BYTES ||
    tensorBytes !== OPT_0008_VAE_RESIDENT_BYTES
  ) throw new Error("OPT-0008 VAE inventory changed from revision 4");
  return Object.freeze({
    files: Object.freeze(files),
    tensorCount: tensors.length,
    residentBytes,
  });
}

function transposeGeometry(
  plan: AceVaeChunkedDecodePlan,
  manifest: AcePackageManifest,
): Readonly<Record<string, readonly AceVaeTransposePartGeometry[]>> {
  return Object.freeze(Object.fromEntries(
    plan.decoderWorkspacePlan.operations
      .filter((operation) => operation.kind === "conv-transpose1d")
      .map((operation) => [
        operation.weight,
        Object.freeze(resolveAceLogicalTensor(manifest, operation.weight).parts
          .map(({ tensor }) => Object.freeze({
            partStart: tensor.partStart,
            partEnd: tensor.partEnd,
          }))),
      ]),
  ));
}

function validateProductionPlan(
  plan: AceVaeChunkedDecodePlan,
): AceVaeChunkedDecodePlan {
  const window = plan.windows[0];
  if (
    plan.latentFrames !== OPT_0008_LATENT_FRAMES ||
    plan.chunkFrames !== 256 ||
    plan.overlapFrames !== 64 ||
    plan.direct !== true ||
    plan.windows.length !== 1 ||
    window === undefined ||
    window.index !== 0 ||
    window.latentWindowFrames !== OPT_0008_LATENT_FRAMES ||
    window.decodedAudioFrames !== 491_520 ||
    plan.outputInterleavedElements !== OPT_0008_OUTPUT_ELEMENTS ||
    plan.outputFloat32Bytes !== OPT_0008_OUTPUT_BYTES ||
    plan.maximumDecodedInterleavedElements !== OPT_0008_OUTPUT_ELEMENTS ||
    plan.maximumDecodedFloat32Bytes !== OPT_0008_OUTPUT_BYTES
  ) throw new Error("OPT-0008 production 256-frame window geometry changed");
  return plan;
}

function validateCooperativePlan(plan: AceVaeDecoderCooperativePlan): void {
  const kindCounts = Object.freeze({
    conv1d: plan.quanta.filter((quantum) => quantum.operationKind === "conv1d").length,
    convTranspose: plan.quanta.filter(
      (quantum) => quantum.operationKind === "conv-transpose1d",
    ).length,
    snake: plan.quanta.filter((quantum) => quantum.operationKind === "snake").length,
    add: plan.quanta.filter((quantum) => quantum.operationKind === "add").length,
  });
  if (
    plan.quantumCount !== OPT_0008_LOGICAL_QUANTUM_COUNT ||
    plan.quanta.length !== OPT_0008_LOGICAL_QUANTUM_COUNT ||
    plan.primitiveDispatchCount !== OPT_0008_PRIMITIVE_DISPATCH_COUNT ||
    kindCounts.conv1d !== 2_459 ||
    kindCounts.convTranspose !== 322 ||
    kindCounts.snake !== 813 ||
    kindCounts.add !== 348
  ) throw new Error("OPT-0008 shipped cooperative graph counts changed");
}

function validateMemory(memory: AceVaeChunkGpuBackendMemoryPlan): void {
  if (
    memory.residentWeightBytes !== OPT_0008_VAE_RESIDENT_BYTES ||
    memory.inputBufferBytes !== 65_536 ||
    memory.outputBufferBytes !== OPT_0008_OUTPUT_BYTES ||
    memory.workspaceBufferBytes !== 251_658_240 ||
    memory.workspaceBufferCount !== 3 ||
    memory.arenaBytes !== 758_972_416 ||
    memory.rangeControlBytes !== 1_020_688 ||
    memory.readbackBufferBytes !== OPT_0008_OUTPUT_BYTES ||
    memory.accountedGpuBytes !== 1_101_508_368 ||
    memory.latentSnapshotBytes !== 65_536 ||
    memory.maximumReturnedWindowBytes !== OPT_0008_OUTPUT_BYTES ||
    memory.boundedCpuBytes !== 3_997_696 ||
    JSON.stringify(memory.uniqueDecoderInputFrames) !== "[256]" ||
    memory.decoderQuantaPerCommandBuffer !== 8 ||
    memory.maximumDecoderQuantaPerWindow !== OPT_0008_LOGICAL_QUANTUM_COUNT ||
    memory.maximumDecoderCommandBuffersPerWindow !==
      OPT_0008_DECODER_BATCH_COUNT ||
    memory.maximumCommandBuffersPerWindow !==
      OPT_0008_TOTAL_COMMAND_BUFFER_COUNT
  ) throw new Error("OPT-0008 production VAE memory/count plan changed");
}

function createDeterministicLatent(): Float32Array<ArrayBuffer> {
  const values = new Float32Array(OPT_0008_LATENT_ELEMENTS);
  let state = OPT_0008_DETERMINISTIC_LATENT_SEED;
  for (let index = 0; index < values.length; index += 1) {
    state = (Math.imul(state, 1_664_525) + 1_013_904_223) >>> 0;
    const unit = (state >>> 8) / 0x0100_0000;
    values[index] = Math.fround(unit * 2 - 1);
  }
  return values;
}

async function prefillCompleteOutput(
  queue: GPUQueue,
  output: GPUBuffer,
): Promise<number> {
  if (output.size !== OPT_0008_OUTPUT_BYTES) {
    throw new Error("OPT-0008 cannot prefill a partial VAE output");
  }
  const sentinel = new Uint32Array(OPT_0008_OUTPUT_ELEMENTS);
  sentinel.fill(OPT_0008_OUTPUT_SENTINEL_BITS);
  const started = performance.now();
  queue.writeBuffer(output, 0, sentinel);
  await queue.onSubmittedWorkDone();
  return performance.now() - started;
}

async function summarizeOutput(
  output: Float32Array,
): Promise<OutputSummary> {
  if (
    output.length !== OPT_0008_OUTPUT_ELEMENTS ||
    output.byteLength !== OPT_0008_OUTPUT_BYTES
  ) throw new Error("OPT-0008 production output has the wrong extent");
  const bits = new Uint32Array(
    output.buffer,
    output.byteOffset,
    output.length,
  );
  let finiteCount = 0;
  let nonzeroCount = 0;
  let sentinelBitCount = 0;
  let minimum = Number.POSITIVE_INFINITY;
  let maximum = Number.NEGATIVE_INFINITY;
  for (let index = 0; index < output.length; index += 1) {
    const value = output[index]!;
    if (Number.isFinite(value)) finiteCount += 1;
    if (value !== 0) nonzeroCount += 1;
    if (bits[index] === OPT_0008_OUTPUT_SENTINEL_BITS) sentinelBitCount += 1;
    minimum = Math.min(minimum, value);
    maximum = Math.max(maximum, value);
  }
  return Object.freeze({
    elementCount: output.length,
    byteLength: output.byteLength,
    finiteCount,
    nonzeroCount,
    sentinelBitCount,
    minimum,
    maximum,
    sha256: await sha256Hex(bytesOf(output)),
  });
}

function validateCompleteOutput(summary: OutputSummary, label: string): void {
  if (
    summary.elementCount !== OPT_0008_OUTPUT_ELEMENTS ||
    summary.byteLength !== OPT_0008_OUTPUT_BYTES ||
    summary.finiteCount !== OPT_0008_OUTPUT_ELEMENTS ||
    summary.nonzeroCount === 0 ||
    summary.sentinelBitCount !== 0 ||
    !Number.isFinite(summary.minimum) ||
    !Number.isFinite(summary.maximum) ||
    !/^[0-9a-f]{64}$/.test(summary.sha256)
  ) throw new Error(`OPT-0008 ${label} output failed its complete-write gate`);
}

function compareOutputs(
  warmup: Float32Array,
  timed: Float32Array,
): Readonly<{
  comparedU32WordCount: number;
  bitMismatchCount: number;
  firstMismatchIndex: number | null;
  bitExact: boolean;
}> {
  if (
    warmup.length !== timed.length ||
    warmup.length !== OPT_0008_OUTPUT_ELEMENTS
  ) throw new Error("OPT-0008 output comparison lengths differ");
  const warmupBits = new Uint32Array(
    warmup.buffer,
    warmup.byteOffset,
    warmup.length,
  );
  const timedBits = new Uint32Array(
    timed.buffer,
    timed.byteOffset,
    timed.length,
  );
  let bitMismatchCount = 0;
  let firstMismatchIndex: number | null = null;
  for (let index = 0; index < warmupBits.length; index += 1) {
    if (warmupBits[index] !== timedBits[index]) {
      bitMismatchCount += 1;
      firstMismatchIndex ??= index;
    }
  }
  return Object.freeze({
    comparedU32WordCount: warmupBits.length,
    bitMismatchCount,
    firstMismatchIndex,
    bitExact: bitMismatchCount === 0,
  });
}

function validateProductionProgress(
  progress: readonly AceVaeChunkGpuBackendProgress[],
): void {
  if (progress.length !== OPT_0008_LOGICAL_QUANTUM_COUNT + 1) {
    throw new Error("OPT-0008 production progress count changed");
  }
  for (let index = 0; index < OPT_0008_LOGICAL_QUANTUM_COUNT; index += 1) {
    const event = progress[index]!;
    const physicalBatch = Math.floor(index / 8) + 1;
    if (
      event.windowIndex !== 0 ||
      event.stage !== "decoder" ||
      event.completedDecoderQuanta !== index + 1 ||
      event.totalDecoderQuanta !== OPT_0008_LOGICAL_QUANTUM_COUNT ||
      event.completedCommandBuffers !== physicalBatch ||
      event.totalCommandBuffers !== OPT_0008_TOTAL_COMMAND_BUFFER_COUNT ||
      event.queueDrains !== physicalBatch ||
      event.cooperativeIdleMs !== physicalBatch
    ) throw new Error(`OPT-0008 decoder progress ${index} changed`);
  }
  const readback = progress.at(-1)!;
  if (
    readback.windowIndex !== 0 ||
    readback.stage !== "readback" ||
    readback.completedDecoderQuanta !== OPT_0008_LOGICAL_QUANTUM_COUNT ||
    readback.totalDecoderQuanta !== OPT_0008_LOGICAL_QUANTUM_COUNT ||
    readback.completedCommandBuffers !== OPT_0008_TOTAL_COMMAND_BUFFER_COUNT ||
    readback.totalCommandBuffers !== OPT_0008_TOTAL_COMMAND_BUFFER_COUNT ||
    readback.queueDrains !== OPT_0008_TOTAL_COMMAND_BUFFER_COUNT ||
    readback.cooperativeIdleMs !== OPT_0008_REQUESTED_IDLE_MILLISECONDS
  ) throw new Error("OPT-0008 readback progress changed");
}

function validateActualPassLabels(
  attribution: ReturnType<typeof createAceOpt0008VaeWindowAttribution>,
  trace: AceOpt0008VaeWindowTrace,
  commands: ReturnType<ProductionDeviceObserver["actualPassLabels"]>,
): void {
  if (
    commands.length !== OPT_0008_TOTAL_COMMAND_BUFFER_COUNT ||
    trace.decoderBatches.length !== OPT_0008_DECODER_BATCH_COUNT
  ) throw new Error("OPT-0008 actual command-label count changed");
  let actualDispatchCount = 0;
  for (const batch of attribution.batches) {
    const command = commands[batch.batchIndex]!;
    const expectedLabels = attribution.quanta
      .slice(batch.firstQuantumIndex, batch.lastQuantumIndex + 1)
      .map((quantum) =>
        `ace-vae-window-0-batch-${batch.batchIndex}-${quantum.quantumId}`
      );
    const expectedDispatchCounts = attribution.quanta
      .slice(batch.firstQuantumIndex, batch.lastQuantumIndex + 1)
      .map((quantum) => quantum.dispatchCount);
    if (
      command.commandLabel !== `ace-vae-window-0-batch-${batch.batchIndex}` ||
      JSON.stringify(command.passLabels) !== JSON.stringify(expectedLabels) ||
      JSON.stringify(command.passDispatchCounts) !==
        JSON.stringify(expectedDispatchCounts)
    ) throw new Error(`OPT-0008 actual batch ${batch.batchIndex} tags changed`);
    actualDispatchCount += command.passDispatchCounts.reduce(
      (total, count) => total + count,
      0,
    );
  }
  const readback = commands.at(-1)!;
  if (
    readback.commandLabel !== "ace-vae-window-0-readback" ||
    readback.passLabels.length !== 0 ||
    readback.passDispatchCounts.length !== 0 ||
    actualDispatchCount !== OPT_0008_PRIMITIVE_DISPATCH_COUNT
  ) throw new Error("OPT-0008 actual pass/dispatch reconciliation changed");
}

async function runCancellationProof(
  backend: AceVaeChunkGpuBackend,
  plan: AceVaeChunkedDecodePlan,
  progressRouter: ProgressRouter,
  observer: ProductionDeviceObserver,
): Promise<CancellationSummary> {
  const controller = new AbortController();
  let abortRequested = false;
  observer.beginCancellationTrace();
  progressRouter.begin("cancellation", (event) => {
    if (
      !abortRequested &&
      event.stage === "decoder" &&
      event.completedCommandBuffers === 1
    ) {
      abortRequested = true;
      controller.abort(new DOMException(
        "OPT-0008 post-drain cancellation probe",
        "AbortError",
      ));
    }
  });
  let rejection: unknown;
  try {
    await backend.decodeWindow(plan.windows[0]!, controller.signal);
  } catch (error) {
    rejection = error;
  }
  const rejectedAt = performance.now();
  const progress = progressRouter.end("cancellation");
  const observed = observer.endCancellationTrace();
  if (!(rejection instanceof DOMException) || rejection.name !== "AbortError") {
    throw new Error("OPT-0008 cancellation did not reject with AbortError");
  }
  const first = observed.records[0];
  if (
    !abortRequested ||
    progress.length !== 8 ||
    observed.records.length !== 1 ||
    observed.submissionCount !== 1 ||
    observed.drainCount !== 1 ||
    first === undefined ||
    first.kind !== "decoder" ||
    first.batchIndex !== 0 ||
    first.passLabels.length !== 8 ||
    first.passDispatchCounts.reduce((total, count) => total + count, 0) < 8 ||
    first.nextCommandEncodeStartedAt !== undefined ||
    first.drainEndedAt === undefined ||
    first.progressReportedAt === undefined ||
    first.progressReportedAt < first.drainEndedAt ||
    rejectedAt < first.progressReportedAt
  ) throw new Error("OPT-0008 post-drain cancellation accounting changed");
  for (let index = 0; index < progress.length; index += 1) {
    const event = progress[index]!;
    if (
      event.stage !== "decoder" ||
      event.completedDecoderQuanta !== index + 1 ||
      event.totalDecoderQuanta !== OPT_0008_LOGICAL_QUANTUM_COUNT ||
      event.completedCommandBuffers !== 1 ||
      event.totalCommandBuffers !== OPT_0008_TOTAL_COMMAND_BUFFER_COUNT ||
      event.queueDrains !== 1 ||
      event.cooperativeIdleMs !== 1
    ) throw new Error("OPT-0008 cancellation progress changed");
  }
  const encodeEndedAt = requiredTimestamp(first.encodeEndedAt, "cancel encode end");
  const submitStartedAt = requiredTimestamp(
    first.submitStartedAt,
    "cancel submit start",
  );
  const drainEndedAt = requiredTimestamp(first.drainEndedAt, "cancel drain end");
  return Object.freeze({
    rejectionName: rejection.name,
    rejectionMessage: rejection.message,
    progressEventCount: progress.length,
    completedDecoderQuanta: progress.at(-1)!.completedDecoderQuanta,
    encodedCommandBufferCount: observed.records.length,
    submissionCount: observed.submissionCount,
    queueDrainCount: observed.drainCount,
    requestedIdleMilliseconds: progress[0]!.cooperativeIdleMs,
    completedIdleCount: 1,
    firstBatchEncodeMilliseconds: encodeEndedAt - first.encodeStartedAt,
    firstBatchSubmitThroughDrainMilliseconds: drainEndedAt - submitStartedAt,
    firstBatchPostDrainToRejectionMilliseconds: rejectedAt - drainEndedAt,
    laterBatchEncodingPrevented: observed.records.length === 1,
    laterBatchSubmissionPrevented: observed.submissionCount === 1,
    firstBatchFullyDrained: observed.drainCount === 1,
    realIdleCompletedBeforeRejection: true,
  });
}

async function verifyPostDestroy(
  backend: AceVaeChunkGpuBackend,
  plan: AceVaeChunkedDecodePlan,
): Promise<Readonly<{
  secondDestroyResolved: boolean;
  decodeRejected: boolean;
  rejectionName: string;
  rejectionMessage: string;
}>> {
  await backend.destroy();
  let rejection: unknown;
  try {
    await backend.decodeWindow(plan.windows[0]!);
  } catch (error) {
    rejection = error;
  }
  if (!(rejection instanceof DOMException) || rejection.name !== "InvalidStateError") {
    throw new Error("OPT-0008 destroyed backend accepted another decode");
  }
  return Object.freeze({
    secondDestroyResolved: true,
    decodeRejected: true,
    rejectionName: rejection.name,
    rejectionMessage: rejection.message,
  });
}

function validateThermalGate(
  thermal: Opt0008ThermalGateMetadata,
  warmupCompletedAtEpochMilliseconds: number,
): void {
  if (
    thermal.source !== "notifyutil-com.apple.system.thermalpressurelevel" ||
    thermal.startedAtEpochMilliseconds < warmupCompletedAtEpochMilliseconds ||
    thermal.completedAtEpochMilliseconds < thermal.startedAtEpochMilliseconds ||
    thermal.durationMilliseconds !==
      thermal.completedAtEpochMilliseconds - thermal.startedAtEpochMilliseconds ||
    thermal.durationMilliseconds < 30_000 ||
    !Number.isSafeInteger(thermal.observationCount) ||
    thermal.observationCount < Math.floor(thermal.durationMilliseconds / 1_000) + 1 ||
    thermal.pollMilliseconds !== 1_000 ||
    !Number.isFinite(thermal.maximumPollGapMilliseconds) ||
    thermal.maximumPollGapMilliseconds < 0 ||
    thermal.maximumPollGapMilliseconds > 1_250 ||
    thermal.nonNominalObservationCount !== 0
  ) throw new Error("OPT-0008 worker rejected the external thermal gate");
}

function validateRunIdentity(identity: unknown): Opt0008RunIdentity {
  if (typeof identity !== "object" || identity === null) {
    throw new Error("OPT-0008 worker requires a frozen run identity");
  }
  const candidate = identity as Readonly<Record<string, unknown>>;
  const requiredString = (name: string): string => {
    const value = candidate[name];
    if (typeof value !== "string" || value.trim() === "" || value !== value.trim()) {
      throw new Error(`OPT-0008 worker rejected run identity ${name}`);
    }
    return value;
  };
  const requiredPositiveInteger = (name: string): number => {
    const value = candidate[name];
    if (!Number.isSafeInteger(value) || (value as number) <= 0) {
      throw new Error(`OPT-0008 worker rejected run identity ${name}`);
    }
    return value as number;
  };
  const harnessCommit = requiredString("harnessCommit");
  const productionCommit = requiredString("productionCommit");
  if (!/^[0-9a-f]{40}$/.test(harnessCommit)) {
    throw new Error("OPT-0008 worker rejected harnessCommit");
  }
  if (productionCommit !== OPT_0008_EXPECTED_PRODUCTION_COMMIT) {
    throw new Error("OPT-0008 worker rejected productionCommit");
  }
  return Object.freeze({
    harnessCommit,
    productionCommit: OPT_0008_EXPECTED_PRODUCTION_COMMIT,
    machineModel: requiredString("machineModel"),
    osVersion: requiredString("osVersion"),
    osBuild: requiredString("osBuild"),
    browserVersion: requiredString("browserVersion"),
    gpuCoreCount: requiredPositiveInteger("gpuCoreCount"),
    memoryBytes: requiredPositiveInteger("memoryBytes"),
  });
}

function publicPreparationSummary(
  prepared: PreparedSession,
): Readonly<Record<string, unknown>> {
  return Object.freeze({
    packageAcquisitionWallMilliseconds:
      prepared.packageAcquisitionWallMilliseconds,
    phaseUploadWallMilliseconds: prepared.phaseUploadWallMilliseconds,
    backendCompileWallMilliseconds: prepared.backendCompileWallMilliseconds,
    phaseProgress: prepared.phaseProgress,
    memory: prepared.memory,
    warmup: Object.freeze({
      prefillWallMilliseconds: prepared.warmup.prefillWallMilliseconds,
      decodeWallMilliseconds: prepared.warmup.wallMilliseconds,
      progressEventCount: prepared.warmup.progressEventCount,
      output: prepared.warmup.outputSummary,
    }),
    warmupCompletedAtEpochMilliseconds:
      prepared.warmupCompletedAtEpochMilliseconds,
  });
}

function startWorkerHeartbeat(): { stop(): WorkerHeartbeatSnapshot } {
  const startedAtEpochMilliseconds = Date.now();
  const gaps: number[] = [];
  let tickCount = 0;
  let stopped = false;
  let last = performance.now();
  const handle = setInterval(() => {
    const now = performance.now();
    gaps.push(now - last);
    last = now;
    tickCount += 1;
  }, WORKER_HEARTBEAT_INTERVAL_MILLISECONDS);
  return {
    stop(): WorkerHeartbeatSnapshot {
      if (!stopped) {
        stopped = true;
        clearInterval(handle);
      }
      return Object.freeze({
        startedAtEpochMilliseconds,
        completedAtEpochMilliseconds: Date.now(),
        timerTickCount: tickCount,
        maximumTimerGapMilliseconds: Math.max(0, ...gaps),
      });
    },
  };
}

function validateWorkerHeartbeat(heartbeat: WorkerHeartbeatSnapshot): void {
  if (
    heartbeat.timerTickCount < 1 ||
    !Number.isFinite(heartbeat.maximumTimerGapMilliseconds) ||
    heartbeat.maximumTimerGapMilliseconds < 0 ||
    heartbeat.completedAtEpochMilliseconds < heartbeat.startedAtEpochMilliseconds
  ) throw new Error("OPT-0008 worker heartbeat telemetry is invalid");
}

async function failAndCleanup(
  error: unknown,
  active: PreparedSession | undefined = session,
): Promise<void> {
  if (lifecycle === "settled") return;
  lifecycle = "settled";
  session = undefined;
  let cleanupError: unknown;
  if (active !== undefined) {
    try {
      await active.backend.destroy();
    } catch (caught) {
      cleanupError = caught;
    } finally {
      active.context.destroy();
    }
  }
  const heartbeat = workerHeartbeat?.stop();
  workerHeartbeat = undefined;
  self.postMessage({
    type: "failed",
    error: Object.freeze({
      primary: errorValue(error),
      ...(cleanupError === undefined
        ? {}
        : { cleanup: errorValue(cleanupError) }),
      ...(heartbeat === undefined ? {} : { workerHeartbeat: heartbeat }),
    }),
  });
}

function bytesOf(values: Float32Array): Uint8Array<ArrayBuffer> {
  if (!(values.buffer instanceof ArrayBuffer)) {
    throw new TypeError("OPT-0008 hashes require owned ArrayBuffer storage");
  }
  return new Uint8Array(values.buffer, values.byteOffset, values.byteLength);
}

async function sha256Hex(bytes: Uint8Array<ArrayBuffer>): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(digest), (value) =>
    value.toString(16).padStart(2, "0")).join("");
}

function sumSafe(values: readonly number[], label: string): number {
  let total = 0;
  for (const value of values) {
    total += value;
    if (!Number.isSafeInteger(total)) throw new RangeError(`${label} overflowed`);
  }
  return total;
}

function formatBytes(bytes: number): string {
  return `${(bytes / (1024 * 1024)).toFixed(1)} MiB`;
}

function postProgress(message: string): void {
  self.postMessage({ type: "progress", message });
}

function errorValue(error: unknown): unknown {
  return error instanceof Error
    ? Object.freeze({ name: error.name, message: error.message, stack: error.stack })
    : Object.freeze({ error: String(error) });
}
