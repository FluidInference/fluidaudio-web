import { ACE_SAMPLE_RATE_HZ } from "../api.js";
import { ACE_VAE_C512_WINDOW_RUNTIME_PROFILE } from
  "../webgpu/vae-window-profile.js";
import type {
  AceDiagnostic,
  AceRuntimeDiagnostics,
} from "./diagnostics.js";
import type { AcePipelineBackend } from "./pipeline.js";
import {
  isAceClientMessage,
  isAceDiagnosticValue,
  isAceGenerationResultValue,
  isAceRuntimeDiagnosticsValue,
  serializeAceWorkerError,
  type AceGenerateMessage,
  type AceInitializeMessage,
  type AceWorkerMessage,
} from "./protocol.js";
import {
  AceInitializationProgressSequence,
  AceProgressSequence,
  generationStagePlan,
  isAceGenerationProgress,
  isAceInitializationProgress,
} from "./stages.js";

export type AceWorkerState =
  | "new"
  | "initializing"
  | "ready"
  | "generating"
  | "disposing"
  | "disposed";

export interface AceWorkerScope {
  postMessage(message: AceWorkerMessage): void;
  addEventListener(
    type: "message",
    listener: (event: MessageEvent<unknown>) => void,
  ): void;
}

export interface AceWorkerRuntimeOptions {
  readonly backend: AcePipelineBackend;
  readonly postMessage: (message: AceWorkerMessage) => void;
}

interface ActiveOperation {
  readonly id: number;
  readonly controller: AbortController;
  callbackError?: unknown;
  promise: Promise<void>;
}

export class AceWorkerRuntimeError extends Error {
  readonly code: string;

  constructor(code: string, message: string) {
    super(message);
    this.name = "AceWorkerRuntimeError";
    this.code = code;
  }
}

/**
 * Testable message shell for a dedicated module worker.
 *
 * Installed listeners intentionally do not await `handleMessage`: cancel and
 * dispose messages must be processed while the backend awaits a GPU fence.
 */
export class AceWorkerRuntime {
  private readonly backend: AcePipelineBackend;
  private readonly postMessage: (message: AceWorkerMessage) => void;
  private initialization: ActiveOperation | undefined;
  private generation: ActiveOperation | undefined;
  private readyDiagnostics: AceRuntimeDiagnostics | undefined;
  private _state: AceWorkerState = "new";

  constructor(options: AceWorkerRuntimeOptions) {
    this.backend = options.backend;
    this.postMessage = options.postMessage;
  }

  get state(): AceWorkerState {
    return this._state;
  }

  async handleMessage(value: unknown): Promise<void> {
    if (!isAceClientMessage(value)) {
      this.postError(
        new AceWorkerRuntimeError(
          "INVALID_MESSAGE",
          "Worker received an invalid ACE protocol message",
        ),
      );
      return;
    }

    switch (value.type) {
      case "initialize":
        await this.initialize(value);
        return;
      case "cancel-initialization":
        if (this.initialization?.id === value.requestId) {
          this.initialization.controller.abort();
        }
        return;
      case "generate":
        await this.generate(value);
        return;
      case "cancel":
        if (this.generation?.id === value.jobId) {
          this.generation.controller.abort();
        }
        return;
      case "dispose":
        await this.dispose(value.requestId);
        return;
    }
  }

  private async initialize(message: AceInitializeMessage): Promise<void> {
    if (this._state === "disposed" || this._state === "disposing") {
      this.postError(
        new AceWorkerRuntimeError("WORKER_DISPOSED", "ACE worker is disposed"),
        { requestId: message.requestId },
      );
      return;
    }
    if (this._state !== "new") {
      this.postError(
        new AceWorkerRuntimeError(
          "INITIALIZATION_CONFLICT",
          `ACE worker cannot initialize while ${this._state}`,
        ),
        { requestId: message.requestId },
      );
      return;
    }

    this._state = "initializing";
    const active: ActiveOperation = {
      id: message.requestId,
      controller: new AbortController(),
      promise: Promise.resolve(),
    };
    this.initialization = active;
    active.promise = this.runInitialization(message, active);
    try {
      await active.promise;
    } finally {
      if (this.initialization === active) this.initialization = undefined;
    }
  }

  private async runInitialization(
    message: AceInitializeMessage,
    active: ActiveOperation,
  ): Promise<void> {
    const progressSequence = new AceInitializationProgressSequence();
    let terminal:
      | { readonly kind: "cancelled" }
      | { readonly kind: "error"; readonly error: unknown }
      | undefined;
    try {
      const diagnostics = await this.backend.initialize(message.configuration, {
        modelSource: message.modelSource,
        signal: active.controller.signal,
        onProgress: (progress) => {
          if (!this.canPublishInitialization(active)) return;
          if (!isAceInitializationProgress(progress)) {
            this.rejectBackendCallback(
              active,
              "INVALID_INITIALIZATION_PROGRESS",
              "ACE backend emitted malformed initialization progress",
            );
            return;
          }
          try {
            progressSequence.accept(progress);
          } catch (error) {
            this.rejectBackendCallback(
              active,
              "INVALID_INITIALIZATION_PROGRESS_SEQUENCE",
              `ACE backend emitted invalid initialization progress: ${formatUnknownError(error)}`,
            );
            return;
          }
          if (message.reportProgress) {
            this.postMessage({
              type: "initialization-progress",
              requestId: message.requestId,
              progress,
            });
          }
        },
        onDiagnostic: (diagnostic) => {
          if (!this.canPublishInitialization(active)) return;
          if (!isAceDiagnosticValue(diagnostic)) {
            this.rejectBackendCallback(
              active,
              "INVALID_DIAGNOSTIC",
              "ACE backend emitted a malformed diagnostic",
            );
            return;
          }
          this.postDiagnostic(
            message.reportDiagnostics,
            diagnostic,
            message.requestId,
          );
        },
      });
      if (
        !isAceRuntimeDiagnosticsValue(diagnostics) ||
        diagnostics.modelManifestUrl !== message.configuration.manifestUrl ||
        diagnostics.modelManifestSha256 !==
          message.configuration.manifestSha256 ||
        diagnostics.vaeManifestUrl !==
          message.configuration.vaePackage.manifestUrl ||
        diagnostics.vaeManifestSha256 !==
          message.configuration.vaePackage.manifestSha256 ||
        diagnostics.vaeRuntimeProfile !==
          message.configuration.vaePackage.runtimeProfile ||
        diagnostics.vaeWindowRuntimeProfile !==
          (message.configuration.vaePackage.windowRuntimeProfile ??
            ACE_VAE_C512_WINDOW_RUNTIME_PROFILE) ||
        diagnostics.vaeMaxWindowFrames !==
          message.configuration.vaePackage.maxWindowFrames ||
        diagnostics.executionProfile.modelProfile !==
          message.configuration.modelProfile ||
        diagnostics.schedulingProfile !==
          message.configuration.schedulingProfile
      ) {
        terminal = {
          kind: "error",
          error: new AceWorkerRuntimeError(
            "INVALID_RUNTIME_DIAGNOSTICS",
            "ACE backend returned diagnostics that do not match the initialization contract",
          ),
        };
      }
      if (active.callbackError !== undefined) {
        terminal = { kind: "error", error: active.callbackError };
      } else if (terminal !== undefined) {
        // The diagnostics contract failure is handled transactionally below.
      } else if (active.controller.signal.aborted) {
        terminal = { kind: "cancelled" };
      } else if (this.canPublishInitialization(active)) {
        this._state = "ready";
        this.readyDiagnostics = diagnostics;
        this.postMessage({
          type: "ready",
          requestId: message.requestId,
          diagnostics,
        });
        return;
      } else {
        return;
      }
    } catch (error) {
      if (active.callbackError !== undefined) {
        terminal = { kind: "error", error: active.callbackError };
      } else if (active.controller.signal.aborted) {
        terminal = { kind: "cancelled" };
      } else {
        terminal = { kind: "error", error };
      }
    }

    if (
      terminal === undefined ||
      this._state === "disposing" ||
      this._state === "disposed"
    ) {
      return;
    }

    // Initialization is a transaction. Abort retained callbacks first, then
    // release every partial resource before advertising cancellation/failure
    // or allowing another initialize request.
    active.controller.abort();
    try {
      await this.backend.dispose();
    } catch (cleanupError) {
      if (
        this._state === "initializing" &&
        this.initialization === active
      ) {
        this._state = "disposed";
        this.postError(
          new AceWorkerRuntimeError(
            "INITIALIZATION_CLEANUP_FAILED",
            `ACE backend cleanup failed after initialization stopped: ${formatUnknownError(cleanupError)}`,
          ),
          { requestId: message.requestId },
        );
      }
      return;
    }

    if (
      this._state !== "initializing" ||
      this.initialization !== active
    ) {
      return;
    }
    this._state = "new";
    if (terminal.kind === "cancelled") {
      this.postMessage({
        type: "initialization-cancelled",
        requestId: message.requestId,
      });
    } else {
      this.postError(terminal.error, { requestId: message.requestId });
    }
  }

  private canPublishInitialization(active: ActiveOperation): boolean {
    return (
      this.initialization === active &&
      this._state === "initializing" &&
      !active.controller.signal.aborted
    );
  }

  private async generate(message: AceGenerateMessage): Promise<void> {
    if (this._state !== "ready") {
      this.postError(
        new AceWorkerRuntimeError(
          this._state === "generating" ? "WORKER_BUSY" : "WORKER_NOT_READY",
          `ACE worker cannot generate while ${this._state}`,
        ),
        { jobId: message.jobId },
      );
      return;
    }

    this._state = "generating";
    const active: ActiveOperation = {
      id: message.jobId,
      controller: new AbortController(),
      promise: Promise.resolve(),
    };
    this.generation = active;
    active.promise = this.runGeneration(message, active);
    await active.promise;
  }

  private async runGeneration(
    message: AceGenerateMessage,
    active: ActiveOperation,
  ): Promise<void> {
    const progressSequence = new AceProgressSequence();
    const stagePlan = generationStagePlan(message.request.planner.mode === "enabled");
    let unpublishedResult: Awaited<ReturnType<AcePipelineBackend["generate"]>> | undefined;
    try {
      const result = await this.backend.generate(message.request, {
        signal: active.controller.signal,
        captureTrace: message.reportDiagnostics,
        onProgress: (progress) => {
          if (!this.canPublishGeneration(active)) return;
          if (!isAceGenerationProgress(progress)) {
            this.rejectBackendCallback(
              active,
              "INVALID_GENERATION_PROGRESS",
              "ACE backend emitted malformed generation progress",
            );
            return;
          }
          if (
            progress.stageCount !== stagePlan.length ||
            progress.stage !== stagePlan[progress.stageIndex]
          ) {
            this.rejectBackendCallback(
              active,
              "INVALID_GENERATION_PROGRESS_PLAN",
              "ACE backend progress does not match the request-specific stage plan",
            );
            return;
          }
          try {
            progressSequence.accept(progress);
          } catch (error) {
            this.rejectBackendCallback(
              active,
              "INVALID_GENERATION_PROGRESS_SEQUENCE",
              `ACE backend emitted invalid generation progress: ${formatUnknownError(error)}`,
            );
            return;
          }
          if (message.reportProgress) {
            this.postMessage({
              type: "generation-progress",
              jobId: message.jobId,
              progress,
            });
          }
        },
        onDiagnostic: (diagnostic) => {
          if (!this.canPublishGeneration(active)) return;
          if (!isAceDiagnosticValue(diagnostic)) {
            this.rejectBackendCallback(
              active,
              "INVALID_DIAGNOSTIC",
              "ACE backend emitted a malformed diagnostic",
            );
            return;
          }
          this.postDiagnostic(
            message.reportDiagnostics,
            diagnostic,
            undefined,
            message.jobId,
          );
        },
      });
      unpublishedResult = result;
      if (
        !isAceGenerationResultValue(result) ||
        (message.request.planner.mode !== "disabled" &&
          result.metrics.vaeScheduling?.selectedProductionPolicy !==
            undefined &&
          result.metrics.vaeScheduling.selectedProductionPolicy !== null) ||
        this.readyDiagnostics === undefined ||
        result.seed !== message.request.seed ||
        result.durationSeconds !== message.request.durationSeconds ||
        result.frameCount !==
          message.request.durationSeconds * ACE_SAMPLE_RATE_HZ ||
        result.modelManifestId !== this.readyDiagnostics.modelManifestId ||
        result.modelManifestSha256 !==
          this.readyDiagnostics.modelManifestSha256 ||
        result.diagnostics.ditDenseManifestId !==
          this.readyDiagnostics.ditDenseManifestId ||
        result.diagnostics.ditDenseManifestSha256 !==
          this.readyDiagnostics.ditDenseManifestSha256 ||
        result.diagnostics.ditDenseManifestByteLength !==
          this.readyDiagnostics.ditDenseManifestByteLength ||
        result.diagnostics.ditDenseRuntimeProfile !==
          this.readyDiagnostics.ditDenseRuntimeProfile ||
        result.diagnostics.ditDenseKernelSetId !==
          this.readyDiagnostics.ditDenseKernelSetId ||
        result.diagnostics.ditDenseLayerBytes !==
          this.readyDiagnostics.ditDenseLayerBytes ||
        result.diagnostics.ditResidentWeightBytes !==
          this.readyDiagnostics.ditResidentWeightBytes ||
        result.diagnostics.vaeManifestId !==
          this.readyDiagnostics.vaeManifestId ||
        result.diagnostics.vaeManifestSha256 !==
          this.readyDiagnostics.vaeManifestSha256 ||
        result.diagnostics.vaeManifestByteLength !==
          this.readyDiagnostics.vaeManifestByteLength ||
        result.diagnostics.vaeRuntimeProfile !==
          this.readyDiagnostics.vaeRuntimeProfile ||
        result.diagnostics.vaeKernelSetId !==
          this.readyDiagnostics.vaeKernelSetId ||
        result.diagnostics.vaePrecisionMapSha256 !==
          this.readyDiagnostics.vaePrecisionMapSha256 ||
        result.diagnostics.vaeMaxWindowFrames !==
          this.readyDiagnostics.vaeMaxWindowFrames ||
        result.diagnostics.executionProfile.id !==
          this.readyDiagnostics.executionProfile.id ||
        result.diagnostics.schedulingProfile !==
          this.readyDiagnostics.schedulingProfile
      ) {
        const ready = this.readyDiagnostics;
        const mismatches = ready === undefined
          ? ["ready-diagnostics-missing"]
          : [
              !isAceGenerationResultValue(result) ? "result-shape-or-receipt" : undefined,
              result.seed !== message.request.seed ? "seed" : undefined,
              result.durationSeconds !== message.request.durationSeconds ? "durationSeconds" : undefined,
              result.frameCount !== message.request.durationSeconds * ACE_SAMPLE_RATE_HZ ? "frameCount" : undefined,
              result.modelManifestId !== ready.modelManifestId ? "modelManifestId" : undefined,
              result.modelManifestSha256 !== ready.modelManifestSha256 ? "modelManifestSha256" : undefined,
              result.diagnostics.vaeWindowRuntimeProfile !== ready.vaeWindowRuntimeProfile ? "vaeWindowRuntimeProfile" : undefined,
              result.diagnostics.vaeMaxWindowFrames !== ready.vaeMaxWindowFrames ? "vaeMaxWindowFrames" : undefined,
              result.diagnostics.executionProfile.id !== ready.executionProfile.id ? "executionProfile" : undefined,
              result.diagnostics.schedulingProfile !== ready.schedulingProfile ? "schedulingProfile" : undefined,
            ].filter((name): name is string => name !== undefined);
        this.rejectBackendCallback(
          active,
          "INVALID_GENERATION_RESULT",
          `ACE backend returned a result that does not match the ready runtime (${
            mismatches.length === 0 ? "manifest/kernel identity" : mismatches.join(", ")
          })`,
        );
      }
      if (active.callbackError !== undefined) {
        if (this._state === "generating") {
          this.postError(active.callbackError, { jobId: message.jobId });
        }
      } else if (active.controller.signal.aborted) {
        if (this._state === "generating") {
          this.postMessage({ type: "cancelled", jobId: message.jobId });
        }
      } else if (this.canPublishGeneration(active)) {
        this.postMessage({ type: "result", jobId: message.jobId, result });
        unpublishedResult = undefined;
      }
    } catch (error) {
      if (active.callbackError !== undefined) {
        if (this._state === "generating") {
          this.postError(active.callbackError, { jobId: message.jobId });
        }
      } else if (active.controller.signal.aborted) {
        if (this._state === "generating") {
          this.postMessage({ type: "cancelled", jobId: message.jobId });
        }
      } else if (this.canPublishGeneration(active)) {
        this.postError(error, { jobId: message.jobId });
      }
    } finally {
      if (unpublishedResult !== undefined) {
        try {
          await this.backend.releaseResult(unpublishedResult);
        } catch (error) {
          if (this._state === "generating" && !active.controller.signal.aborted) {
            this.postError(
              new AceWorkerRuntimeError(
                "UNPUBLISHED_RESULT_CLEANUP_FAILED",
                `ACE could not release an unpublished result: ${formatUnknownError(error)}`,
              ),
              { jobId: message.jobId },
            );
          }
        }
      }
      if (this.generation === active) this.generation = undefined;
      if (this._state === "generating") this._state = "ready";
    }
  }

  private canPublishGeneration(active: ActiveOperation): boolean {
    return (
      this.generation === active &&
      this._state === "generating" &&
      !active.controller.signal.aborted
    );
  }

  private rejectBackendCallback(
    active: ActiveOperation,
    code: string,
    message: string,
  ): void {
    if (active.callbackError !== undefined) return;
    active.callbackError = new AceWorkerRuntimeError(code, message);
    active.controller.abort();
  }

  private async dispose(requestId: number): Promise<void> {
    if (this._state === "disposed") {
      this.postMessage({ type: "disposed", requestId });
      return;
    }
    if (this._state === "disposing") {
      this.postError(
        new AceWorkerRuntimeError(
          "DISPOSAL_IN_PROGRESS",
          "ACE worker disposal is already in progress",
        ),
        { requestId },
      );
      return;
    }

    this._state = "disposing";
    this.readyDiagnostics = undefined;
    this.initialization?.controller.abort();
    this.generation?.controller.abort();
    const activePromises = [
      this.initialization?.promise,
      this.generation?.promise,
    ].filter((promise): promise is Promise<void> => promise !== undefined);
    await Promise.allSettled(activePromises);
    try {
      await this.backend.dispose();
      this._state = "disposed";
      this.postMessage({ type: "disposed", requestId });
    } catch (error) {
      this._state = "disposed";
      this.postError(error, { requestId });
    }
  }

  private postDiagnostic(
    enabled: boolean,
    diagnostic: AceDiagnostic,
    requestId?: number,
    jobId?: number,
  ): void {
    if (!enabled) return;
    this.postMessage({
      type: "diagnostic",
      ...(requestId === undefined ? {} : { requestId }),
      ...(jobId === undefined ? {} : { jobId }),
      diagnostic,
    });
  }

  private postError(
    error: unknown,
    ids: { readonly requestId?: number; readonly jobId?: number } = {},
  ): void {
    this.postMessage({
      type: "error",
      ...ids,
      error: serializeAceWorkerError(error),
    });
  }
}

export function installAceWorkerRuntime(
  scope: AceWorkerScope,
  backend: AcePipelineBackend,
): AceWorkerRuntime {
  const runtime = new AceWorkerRuntime({
    backend,
    postMessage: (message) => scope.postMessage(message),
  });
  scope.addEventListener("message", (event) => {
    void runtime.handleMessage(event.data);
  });
  return runtime;
}

function formatUnknownError(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
