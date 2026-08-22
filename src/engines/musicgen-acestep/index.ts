// Page-side client for ACE-Step 1.5 Turbo music generation. Wraps the
// packages/acestep worker protocol (see AceClientMessage/AceWorkerMessage)
// behind one promise-per-generation API so pages don't hand-roll the
// initialize/generate/cancel/dispose state machine the way the upstream
// demo's main.ts does.

import {
  isAceFatalGpuErrorCode,
  isAceWorkerMessage,
  type AceGenerationProgress,
  type AceGenerationRequest,
  type AceGenerationResult,
  type AceInitializationProgress,
  type AceRuntimeDiagnostics,
} from "ace-step-1.5.wgsl";

import { aceProductionWorkerConfiguration } from "./config.js";
import { INITIAL_MODEL_DOWNLOAD_PROGRESS, updateModelDownloadProgress, type ModelDownloadProgress } from "./model-download-progress.js";

export {
  ACE_MAX_DURATION_SECONDS,
  ACE_MIN_DURATION_SECONDS,
  aceSeed,
  checkSupport,
  deleteAceModelCache,
  inspectAceModelCache,
  releaseAceAudioOutput,
  requestAceModelStoragePersistence,
} from "ace-step-1.5.wgsl";
export type { AceGenerationRequest, AceGenerationResult, AceModelCacheInfo, AceSupportReport } from "ace-step-1.5.wgsl";
export {
  isModelDownloadComplete,
  MODEL_DOWNLOAD_TOTAL_BYTES,
  formatDecimalBytes,
  formatModelDownloadAmount,
  type ModelDownloadProgress,
} from "./model-download-progress.js";

export interface AceMusicGenerateHandlers {
  /** Model download / preparation progress (bytes across all packages). */
  onDownloadProgress?: (progress: ModelDownloadProgress) => void;
  /** Initialization stage progress that is not a byte-download update. */
  onInitializationProgress?: (progress: AceInitializationProgress) => void;
  /** Generation stage progress that is not a byte-download update. */
  onGenerationProgress?: (progress: AceGenerationProgress) => void;
  onDiagnostic?: (diagnostic: unknown) => void;
}

interface ActiveOperation {
  readonly resolve: (result: AceGenerationResult) => void;
  readonly reject: (reason: unknown) => void;
  readonly handlers: AceMusicGenerateHandlers;
  initializationRequestId: number | undefined;
  jobId: number | undefined;
  request: AceGenerationRequest | undefined;
}

export class AceStepMusicClient {
  private worker: Worker | undefined;
  private workerReady = false;
  private nextRequestId = 1;
  private nextJobId = 1;
  private active: ActiveOperation | undefined;
  private disposal: { requestId: number; resolve: () => void; reject: (reason: unknown) => void } | undefined;
  private fatalGpuDiagnostic = false;
  private downloadProgress: ModelDownloadProgress = INITIAL_MODEL_DOWNLOAD_PROGRESS;
  /** Diagnostics reported by the worker's ready message, when initialized. */
  runtimeDiagnostics: AceRuntimeDiagnostics | undefined;

  get busy(): boolean {
    return this.active !== undefined;
  }

  get initialized(): boolean {
    return this.workerReady;
  }

  /**
   * Generate one song. Initializes the worker (downloading/loading model
   * packages) on first use. Only one generation may be in flight.
   */
  async generate(request: AceGenerationRequest, handlers: AceMusicGenerateHandlers = {}): Promise<AceGenerationResult> {
    if (this.active !== undefined) {
      throw new Error("A generation is already in progress");
    }
    return await new Promise<AceGenerationResult>((resolve, reject) => {
      this.active = {
        resolve,
        reject,
        handlers,
        initializationRequestId: undefined,
        jobId: undefined,
        request,
      };
      if (this.workerReady && this.worker !== undefined) {
        this.startGeneration();
      } else {
        this.startInitialization();
      }
    });
  }

  /** Cancel the in-flight initialization or generation, if any. */
  cancel(): void {
    const active = this.active;
    if (this.worker === undefined || active === undefined) return;
    if (active.initializationRequestId !== undefined) {
      this.worker.postMessage({
        type: "cancel-initialization",
        requestId: active.initializationRequestId,
      });
    } else if (active.jobId !== undefined) {
      this.worker.postMessage({ type: "cancel", jobId: active.jobId });
    }
  }

  /** Release the worker's GPU/runtime resources and terminate it. */
  async dispose(): Promise<void> {
    const current = this.worker;
    if (current === undefined) return;
    if (!this.workerReady) {
      this.resetWorker(new DOMException("Client disposed", "AbortError"));
      return;
    }
    const requestId = this.nextRequestId++;
    await new Promise<void>((resolve, reject) => {
      this.disposal = { requestId, resolve, reject };
      current.postMessage({ type: "dispose", requestId });
    });
    current.terminate();
    if (this.worker === current) this.worker = undefined;
    this.workerReady = false;
    this.runtimeDiagnostics = undefined;
  }

  /** Terminate immediately without an orderly runtime dispose. */
  terminate(): void {
    this.resetWorker(new DOMException("Client terminated", "AbortError"));
  }

  private startInitialization(): void {
    const active = this.active;
    if (active === undefined) return;
    this.worker?.terminate();
    this.workerReady = false;
    this.fatalGpuDiagnostic = false;
    this.worker = new Worker(new URL("./worker.ts", import.meta.url), {
      type: "module",
      name: "ace-step-inference",
    });
    this.worker.addEventListener("message", (event) => this.onMessage(event));
    this.worker.addEventListener("error", (event) => {
      this.fail(new Error(`Inference worker error: ${event.message}`), true);
    });
    active.initializationRequestId = this.nextRequestId++;
    this.worker.postMessage({
      type: "initialize",
      requestId: active.initializationRequestId,
      configuration: aceProductionWorkerConfiguration(),
      modelSource: "cache-or-network",
      reportProgress: true,
      reportDiagnostics: true,
    });
  }

  private startGeneration(): void {
    const active = this.active;
    if (active === undefined || this.worker === undefined) return;
    const request = active.request;
    if (request === undefined) return;
    active.request = undefined;
    active.jobId = this.nextJobId++;
    this.worker.postMessage({
      type: "generate",
      jobId: active.jobId,
      request,
      reportProgress: true,
      reportDiagnostics: true,
    });
  }

  private onMessage(event: MessageEvent<unknown>): void {
    if (!isAceWorkerMessage(event.data)) {
      this.fail(new Error("The inference worker emitted an invalid message"), true);
      return;
    }
    const message = event.data;
    const active = this.active;
    switch (message.type) {
      case "initialization-progress": {
        if (message.requestId !== active?.initializationRequestId) return;
        const updated = updateModelDownloadProgress(this.downloadProgress, message);
        if (updated !== this.downloadProgress) {
          this.downloadProgress = updated;
          active.handlers.onDownloadProgress?.(updated);
        } else {
          active.handlers.onInitializationProgress?.(message.progress);
        }
        return;
      }
      case "ready":
        if (message.requestId !== active?.initializationRequestId) return;
        active.initializationRequestId = undefined;
        this.workerReady = true;
        this.runtimeDiagnostics = message.diagnostics;
        this.startGeneration();
        return;
      case "initialization-cancelled":
        if (message.requestId !== active?.initializationRequestId) return;
        this.resetWorker(new DOMException("Initialization cancelled", "AbortError"));
        return;
      case "generation-progress": {
        if (message.jobId !== active?.jobId) return;
        const updated = updateModelDownloadProgress(this.downloadProgress, message);
        if (updated !== this.downloadProgress) {
          this.downloadProgress = updated;
          active.handlers.onDownloadProgress?.(updated);
        } else {
          active.handlers.onGenerationProgress?.(message.progress);
        }
        return;
      }
      case "diagnostic":
        if (message.diagnostic.code === "WEBGPU_DEVICE_LOST" || message.diagnostic.code === "WEBGPU_UNCAPTURED_ERROR") {
          this.fatalGpuDiagnostic = true;
        }
        active?.handlers.onDiagnostic?.(message.diagnostic);
        return;
      case "result":
        if (message.jobId !== active?.jobId) return;
        this.active = undefined;
        active.resolve(message.result);
        return;
      case "cancelled":
        if (message.jobId !== active?.jobId) return;
        this.active = undefined;
        active.reject(new DOMException("Generation cancelled", "AbortError"));
        return;
      case "disposed":
        if (this.disposal?.requestId !== message.requestId) return;
        this.disposal.resolve();
        this.disposal = undefined;
        this.workerReady = false;
        return;
      case "error": {
        if (this.disposal !== undefined && message.requestId === this.disposal.requestId) {
          this.disposal.reject(new Error(message.error.message));
          this.disposal = undefined;
          return;
        }
        const fatal = this.fatalGpuDiagnostic || isAceFatalGpuErrorCode(message.error.code);
        this.fail(new Error(`${message.error.code}: ${message.error.message}`), fatal);
        return;
      }
    }
  }

  private fail(reason: Error, resetWorker: boolean): void {
    const active = this.active;
    this.active = undefined;
    if (resetWorker) {
      this.worker?.terminate();
      this.worker = undefined;
      this.workerReady = false;
      this.runtimeDiagnostics = undefined;
    }
    active?.reject(reason);
  }

  private resetWorker(reason: unknown): void {
    const active = this.active;
    this.active = undefined;
    this.worker?.terminate();
    this.worker = undefined;
    this.workerReady = false;
    this.runtimeDiagnostics = undefined;
    if (active !== undefined) active.reject(reason);
  }
}
