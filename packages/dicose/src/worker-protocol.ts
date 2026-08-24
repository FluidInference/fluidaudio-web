/**
 * Structured-clone-safe protocol shared by the page-side client and the
 * dedicated WebGPU worker. Keeping PCM planar avoids an extra interleave and
 * lets all large buffers move between threads without copying.
 */

export const DICOSE_STEM_NAMES = ["drums", "bass", "other", "vocals"] as const;

export type DiCoSeStemName = (typeof DICOSE_STEM_NAMES)[number];
export type DiCoSeOutputMode = "refined" | "deterministic";

export interface DiCoSeProgress {
  readonly phase: string;
  readonly completed?: number;
  readonly total?: number;
  readonly detail?: string;
}

export interface DiCoSeWorkerInitOptions {
  /** Absolute or page-relative model manifest URL. */
  readonly manifestUrl?: string;
  /** Test seam for exact-q64 versus blockwise-Flash quality comparisons. */
  readonly attentionKernel?: "q64" | "flash";
}

export interface DiCoSeWorkerSeparateOptions {
  /** Fixed CD-noise seed. Omit to use the runtime's documented default. */
  readonly seed?: number;
  /** Refined is the released one-step CD path and remains the default. */
  readonly outputMode?: DiCoSeOutputMode;
}

export interface DiCoSePcmTransfer {
  readonly sampleRate: number;
  readonly length: number;
  readonly left: ArrayBuffer;
  readonly right: ArrayBuffer;
}

export interface DiCoSeStemTransfer extends DiCoSePcmTransfer {}

export type DiCoSeStemTransfers = Readonly<Record<DiCoSeStemName, DiCoSeStemTransfer>>;

export interface DiCoSeWorkerResult {
  readonly outputMode: DiCoSeOutputMode;
  readonly stems: DiCoSeStemTransfers;
  /** Decoded input mixture minus the restored vocal estimate; not a model head. */
  readonly instrumental: DiCoSePcmTransfer;
  /** Named, monotonic timing values in milliseconds supplied by the runtime. */
  readonly timing: Readonly<Record<string, number>>;
  readonly diagnostics: Readonly<Record<string, Readonly<Record<DiCoSeStemName, Readonly<{
    peak: number;
    rms: number;
  }>>>>>
}

export interface DiCoSeWorkerError {
  readonly name: string;
  readonly message: string;
  readonly stack?: string;
}

export type DiCoSeWorkerRequest =
  | {
    readonly type: "initialize";
    readonly id: number;
    readonly options: DiCoSeWorkerInitOptions;
  }
  | {
    readonly type: "separate";
    readonly id: number;
    readonly pcm: DiCoSePcmTransfer;
    readonly options: DiCoSeWorkerSeparateOptions;
  }
  | {
    readonly type: "dispose";
    readonly id: number;
  };

export type DiCoSeWorkerEvent =
  | {
    readonly type: "initialized";
    readonly id: number;
  }
  | {
    readonly type: "progress";
    readonly id: number;
    readonly progress: DiCoSeProgress;
  }
  | {
    readonly type: "result";
    readonly id: number;
    readonly result: DiCoSeWorkerResult;
  }
  | {
    readonly type: "disposed";
    readonly id: number;
  }
  | {
    readonly type: "error";
    readonly id: number;
    readonly error: DiCoSeWorkerError;
  };

export function isDiCoSeWorkerEvent(value: unknown): value is DiCoSeWorkerEvent {
  if (typeof value !== "object" || value === null || !("type" in value) || !("id" in value)) {
    return false;
  }
  const event = value as { readonly type?: unknown; readonly id?: unknown };
  return typeof event.id === "number" && ["initialized", "progress", "result", "disposed", "error"].includes(
    String(event.type),
  );
}
