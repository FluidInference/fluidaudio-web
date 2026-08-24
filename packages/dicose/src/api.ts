import { decodeAudioBlob, type StereoPcm } from "./runtime/audio.js";
import {
  DICOSE_STEM_NAMES,
  isDiCoSeWorkerEvent,
  type DiCoSeProgress,
  type DiCoSeStemName,
  type DiCoSeWorkerError,
  type DiCoSeWorkerInitOptions,
  type DiCoSeWorkerResult,
  type DiCoSeWorkerSeparateOptions,
} from "./worker-protocol.js";

export interface DiCoSeClientOptions extends DiCoSeWorkerInitOptions {
  /** Invoked from the page thread for loading and inference progress. */
  readonly onProgress?: (progress: DiCoSeProgress) => void;
  /**
   * Host-bundler seam: supply the dedicated worker instead of the default
   * `new Worker(new URL("./worker.ts", import.meta.url))`, whose relative URL
   * does not survive every consumer's bundling of a prebuilt dist. The worker
   * must run this package's worker entry (`dicose-wgsl/worker`).
   */
  readonly createWorker?: () => Worker;
}

export interface DiCoSeSeparateOptions extends DiCoSeWorkerSeparateOptions {
  /** Overrides the progress callback supplied when the worker was created. */
  readonly onProgress?: (progress: DiCoSeProgress) => void;
}

export interface DiCoSeSeparation {
  readonly outputMode: DiCoSeWorkerResult["outputMode"];
  readonly stems: Readonly<Record<DiCoSeStemName, StereoPcm>>;
  /** Decoded input mixture minus vocals; derived after native-timeline restoration. */
  readonly instrumental: StereoPcm;
  readonly timing: Readonly<Record<string, number>>;
  readonly diagnostics: DiCoSeWorkerResult["diagnostics"];
}

interface PendingRequest<T> {
  readonly resolve: (value: T) => void;
  readonly reject: (reason: Error) => void;
  readonly onProgress?: (progress: DiCoSeProgress) => void;
}

/**
 * A page-side facade for the dedicated WebGPU worker. The weight package,
 * device, and model execution stay off the UI thread; only decoder work and
 * result ownership cross the boundary.
 */
export class DiCoSeWorkerClient {
  private readonly worker: Worker;
  private readonly pending = new Map<number, PendingRequest<unknown>>();
  private nextId = 1;
  private initialized: Promise<void> | undefined;
  private disposed = false;

  constructor(private readonly options: DiCoSeClientOptions = {}) {
    this.worker =
      options.createWorker?.() ??
      new Worker(new URL("./worker.ts", import.meta.url), {
        type: "module",
        name: "dicose-webgpu",
      });
    this.worker.addEventListener("message", this.handleMessage);
    this.worker.addEventListener("messageerror", this.handleMessageError);
    this.worker.addEventListener("error", this.handleWorkerError);
  }

  /** Create the WebGPU device and stream the model package once. */
  async initialize(): Promise<void> {
    this.requireAlive();
    if (this.initialized === undefined) {
      const manifestUrl = resolveOptionalUrl(this.options.manifestUrl);
      const attentionKernel = this.options.attentionKernel;
      this.initialized = this.request<void>(
        "initialize",
        {
          options: {
            ...(manifestUrl === undefined ? {} : { manifestUrl }),
            ...(attentionKernel === undefined ? {} : { attentionKernel }),
          },
        },
        [],
        this.options.onProgress,
      );
    }
    return await this.initialized;
  }

  /** Decode a browser-supported audio blob and restore all outputs to its native timeline. */
  async separateAudio(
    source: Blob | ArrayBuffer,
    options: DiCoSeSeparateOptions = {},
  ): Promise<DiCoSeSeparation> {
    this.requireAlive();
    const blob = source instanceof Blob ? source : new Blob([source]);
    const pcm = await decodeAudioBlob(blob, { targetSampleRate: "source" });
    return await this.separatePcm(pcm, options);
  }

  /**
   * Separate already-decoded stereo PCM. Input channel data are copied before
   * transfer, so callers retain ownership of their original arrays.
   */
  async separatePcm(
    pcm: StereoPcm,
    options: DiCoSeSeparateOptions = {},
  ): Promise<DiCoSeSeparation> {
    this.requireAlive();
    validatePcm(pcm);
    await this.initialize();
    const left = pcm.left.slice();
    const right = pcm.right.slice();
    const raw = await this.request<DiCoSeWorkerResult>(
      "separate",
      {
        pcm: {
          sampleRate: pcm.sampleRate,
          length: pcm.length,
          left: left.buffer,
          right: right.buffer,
        },
        options: {
          ...(options.seed === undefined ? {} : { seed: options.seed }),
          ...(options.outputMode === undefined ? {} : { outputMode: options.outputMode }),
        },
      },
      [left.buffer, right.buffer],
      options.onProgress ?? this.options.onProgress,
    );
    return materializeResult(raw);
  }

  /** Release the worker's GPU buffers and terminate its isolated thread. */
  async dispose(): Promise<void> {
    if (this.disposed) return;
    this.disposed = true;
    try {
      await this.request<void>("dispose", {}, [], undefined, true);
    } finally {
      this.worker.removeEventListener("message", this.handleMessage);
      this.worker.removeEventListener("messageerror", this.handleMessageError);
      this.worker.removeEventListener("error", this.handleWorkerError);
      this.worker.terminate();
      this.rejectAll(new Error("DiCoSe worker was disposed"));
    }
  }

  private request<T>(
    type: "initialize" | "separate" | "dispose",
    payload: object,
    transfer: Transferable[],
    onProgress: ((progress: DiCoSeProgress) => void) | undefined,
    allowDisposed = false,
  ): Promise<T> {
    if (!allowDisposed) this.requireAlive();
    const id = this.nextId;
    this.nextId += 1;
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, {
        resolve: resolve as (value: unknown) => void,
        reject,
        ...(onProgress === undefined ? {} : { onProgress }),
      });
      try {
        this.worker.postMessage({ type, id, ...payload }, transfer);
      } catch (error) {
        this.pending.delete(id);
        reject(error instanceof Error ? error : new Error(String(error)));
      }
    });
  }

  private readonly handleMessage = (event: MessageEvent<unknown>): void => {
    if (!isDiCoSeWorkerEvent(event.data)) return;
    const message = event.data;
    const pending = this.pending.get(message.id);
    if (pending === undefined) return;
    if (message.type === "progress") {
      pending.onProgress?.(message.progress);
      return;
    }
    this.pending.delete(message.id);
    switch (message.type) {
      case "initialized":
      case "disposed":
        pending.resolve(undefined);
        break;
      case "result":
        pending.resolve(message.result);
        break;
      case "error":
        pending.reject(workerError(message.error));
        break;
      default:
        assertNever(message);
    }
  };

  private readonly handleMessageError = (): void => {
    this.rejectAll(new Error("DiCoSe worker sent an unreadable message"));
  };

  private readonly handleWorkerError = (event: ErrorEvent): void => {
    this.rejectAll(new Error(event.message || "DiCoSe worker failed"));
  };

  private rejectAll(error: Error): void {
    for (const pending of this.pending.values()) pending.reject(error);
    this.pending.clear();
  }

  private requireAlive(): void {
    if (this.disposed) throw new Error("DiCoSe worker client has been disposed");
  }
}

function resolveOptionalUrl(value: string | undefined): string | undefined {
  return value === undefined ? undefined : new URL(value, globalThis.location.href).href;
}

function materializeResult(result: DiCoSeWorkerResult): DiCoSeSeparation {
  const stems = {} as Record<DiCoSeStemName, StereoPcm>;
  for (const name of DICOSE_STEM_NAMES) {
    const stem = result.stems[name];
    if (stem === undefined) throw new Error(`DiCoSe worker omitted the ${name} stem`);
    const left = new Float32Array(stem.left);
    const right = new Float32Array(stem.right);
    if (left.length !== stem.length || right.length !== stem.length) {
      throw new Error(`DiCoSe worker returned malformed ${name} PCM`);
    }
    stems[name] = makeStereoPcm(stem.sampleRate, left, right);
  }
  const instrumental = materializePcm(result.instrumental, "instrumental");
  if (
    instrumental.sampleRate !== stems.vocals.sampleRate ||
    instrumental.length !== stems.vocals.length
  ) {
    throw new Error("DiCoSe worker returned instrumental PCM on a different timeline");
  }
  return {
    outputMode: result.outputMode,
    stems,
    instrumental,
    timing: Object.freeze({ ...result.timing }),
    diagnostics: result.diagnostics,
  };
}

function materializePcm(
  transfer: DiCoSeWorkerResult["instrumental"],
  label: string,
): StereoPcm {
  const left = new Float32Array(transfer.left);
  const right = new Float32Array(transfer.right);
  if (left.length !== transfer.length || right.length !== transfer.length) {
    throw new Error(`DiCoSe worker returned malformed ${label} PCM`);
  }
  return makeStereoPcm(transfer.sampleRate, left, right);
}

function makeStereoPcm(sampleRate: number, left: Float32Array, right: Float32Array): StereoPcm {
  if (!Number.isFinite(sampleRate) || sampleRate <= 0 || !Number.isInteger(sampleRate)) {
    throw new Error("DiCoSe worker returned an invalid PCM sample rate");
  }
  if (left.length !== right.length) throw new Error("DiCoSe worker returned unequal stereo channels");
  return Object.freeze({
    sampleRate,
    length: left.length,
    left,
    right,
    channels: [left, right] as const,
  });
}

function validatePcm(pcm: StereoPcm): void {
  if (pcm.left.length !== pcm.length || pcm.right.length !== pcm.length) {
    throw new RangeError("Stereo PCM channel lengths must match the declared length");
  }
  if (!Number.isInteger(pcm.sampleRate) || pcm.sampleRate <= 0) {
    throw new RangeError("Stereo PCM sample rate must be a positive integer");
  }
}

function workerError(error: DiCoSeWorkerError): Error {
  const result = new Error(error.message);
  result.name = error.name;
  if (error.stack !== undefined) result.stack = error.stack;
  return result;
}

function assertNever(value: never): never {
  throw new Error(`Unhandled DiCoSe worker event: ${JSON.stringify(value)}`);
}
