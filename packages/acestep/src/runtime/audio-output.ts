import type {
  AceVaeChunkedDecodePlan,
  AceVaePostprocessPlan,
} from "../webgpu/vae-chunks.js";
import {
  AceVaeRawF32FileSink,
  type AceVaeWavCooperativeWriteOptions,
  type AceVaeWavWriteResult,
  writeNormalizedAceVaeFloat32WavCooperatively,
} from "../webgpu/vae-wav.js";

const AUDIO_DIRECTORY = "ace-step-1.5.wgsl-audio-v1";
const RAW_FILE = "raw.f32.partial";
const WAV_FILE = "output.wav";
const ACTIVE_MARKER_FILE = "active.partial";
const COMMITTED_MARKER_FILE = "committed";
const TRANSACTION_ID = /^(?!\.{1,2}$)[A-Za-z0-9._-]{1,96}$/;
const DEFAULT_STALE_INCOMPLETE_AGE_MS = 24 * 60 * 60 * 1_000;

export interface AceAudioOutputStorage {
  getDirectory(): Promise<FileSystemDirectoryHandle>;
}

export interface AceCommittedAudioOutput {
  /** OPFS-backed immutable snapshot. No full-song ArrayBuffer is created. */
  readonly audio: Blob;
  readonly wav: AceVaeWavWriteResult;
  readonly transactionId: string;
  /** @internal Diagnostic-only OPFS snapshot of raw pre-normalization F32. */
  readonly rawSnapshot?: Blob;
}

export interface AceAudioOutputTransactionTrace {
  readonly schema: "ace-audio-output-transaction-capture-v1";
  readonly operation:
    | "raw-finish-and-flush"
    | "raw-remove"
    | "raw-snapshot-validation"
    | "wav-snapshot-validation"
    | "durable-publish-markers";
  readonly startedAtMs: number;
  readonly completedAtMs: number;
  readonly wallMs: number;
  readonly byteLength?: number;
}

export interface AceAudioOutputCommitOptions
  extends AceVaeWavCooperativeWriteOptions {
  /** @internal Capture-only transaction attribution. */
  readonly onTransactionTrace?: (trace: AceAudioOutputTransactionTrace) => void;
  /** @internal Preserve raw OPFS storage until the committed job is released. */
  readonly retainRawSnapshot?: true;
}

export interface AceRawAudioSnapshotComparisonOptions {
  readonly signal?: AbortSignal;
  /** Maximum bytes held for each input at once; rounded down to whole F32s. */
  readonly blockBytes?: number;
}

export interface AceRawAudioSnapshotComparison {
  readonly schema: "ace-raw-audio-snapshot-comparison-v1";
  readonly byteLength: number;
  readonly sampleCount: number;
  readonly exactU32MismatchCount: number;
  readonly leftNonFiniteCount: number;
  readonly rightNonFiniteCount: number;
  readonly finitePairCount: number;
  readonly maximumAbsoluteDifference: number;
  readonly meanAbsoluteDifference: number;
  readonly rootMeanSquareDifference: number;
}

export interface AceAudioOutputRecoveryOptions {
  readonly signal?: AbortSignal;
  /** @internal Deterministic test clock. */
  readonly nowMs?: number;
  /** @internal Production uses a 24-hour cross-tab safety lease. */
  readonly minimumIncompleteAgeMs?: number;
}

/**
 * Worker-only, per-generation OPFS audio transaction.
 *
 * The job directory is not part of the public model cache and is addressed by
 * an explicit, validated ID. A failed or cancelled transaction recursively
 * removes only that directory. A committed WAV remains in OPFS because the
 * returned `File` snapshot can outlive the inference worker and must remain
 * readable by the UI without copying the complete song into JS memory.
 */
export class AceAudioOutputTransaction {
  readonly rawSink: AceVaeRawF32FileSink;

  private state: "active" | "committed" | "rolled-back" = "active";

  private constructor(
    readonly transactionId: string,
    private readonly audioRoot: FileSystemDirectoryHandle,
    private readonly jobDirectory: FileSystemDirectoryHandle,
    private readonly wavHandle: FileSystemFileHandle,
    private readonly rawHandle: FileSystemFileHandle,
    private readonly rawFile: OwnedSyncFile,
    private readonly wavFile: OwnedSyncFile,
    private readonly plan: AceVaeChunkedDecodePlan,
  ) {
    this.rawSink = new AceVaeRawF32FileSink(rawFile, plan);
  }

  static async begin(
    transactionId: string,
    plan: AceVaeChunkedDecodePlan,
    storage: AceAudioOutputStorage = navigator.storage,
  ): Promise<AceAudioOutputTransaction> {
    assertTransactionId(transactionId);
    const opfsRoot = await storage.getDirectory();
    const audioRoot = await opfsRoot.getDirectoryHandle(AUDIO_DIRECTORY, {
      create: true,
    });
    // Reusing an identity could invalidate a Blob still held by the UI. Every
    // generation therefore owns a fresh ID; stale jobs are released explicitly.
    await assertEntryAbsent(audioRoot, transactionId);
    const jobDirectory = await audioRoot.getDirectoryHandle(transactionId, {
      create: true,
    });
    let rawFile: OwnedSyncFile | undefined;
    let wavFile: OwnedSyncFile | undefined;
    try {
      await jobDirectory.getFileHandle(ACTIVE_MARKER_FILE, { create: true });
      const rawHandle = await jobDirectory.getFileHandle(RAW_FILE, {
        create: true,
      });
      const wavHandle = await jobDirectory.getFileHandle(WAV_FILE, {
        create: true,
      });
      rawFile = new OwnedSyncFile(await rawHandle.createSyncAccessHandle());
      wavFile = new OwnedSyncFile(await wavHandle.createSyncAccessHandle());
      return new AceAudioOutputTransaction(
        transactionId,
        audioRoot,
        jobDirectory,
        wavHandle,
        rawHandle,
        rawFile,
        wavFile,
        plan,
      );
    } catch (error) {
      closeWithoutMasking(rawFile);
      closeWithoutMasking(wavFile);
      try {
        await removeEntryIfPresent(audioRoot, transactionId, true);
      } catch {
        // Preserve the setup failure; a later retry removes this exact stale ID.
      }
      throw error;
    }
  }

  /**
   * Finish raw storage, run the bounded second pass, and publish the WAV File.
   * Call `rollback()` if this throws.
   */
  async commit(
    postprocess: AceVaePostprocessPlan,
    options: AceAudioOutputCommitOptions = {},
  ): Promise<AceCommittedAudioOutput> {
    this.requireActive();
    const now = options.now ?? defaultNow;
    let startedAtMs = options.onTransactionTrace === undefined ? 0 : now();
    this.rawSink.finish();
    emitTransactionTrace(options.onTransactionTrace, {
      operation: "raw-finish-and-flush",
      startedAtMs,
      completedAtMs: options.onTransactionTrace === undefined ? 0 : now(),
      byteLength: this.plan.outputFloat32Bytes,
    });
    const wav = await writeNormalizedAceVaeFloat32WavCooperatively(
      this.rawFile,
      this.wavFile,
      this.plan,
      postprocess,
      options,
    );
    this.rawFile.close();
    this.wavFile.close();
    let rawSnapshot: Blob | undefined;
    if (options.retainRawSnapshot === true) {
      startedAtMs = options.onTransactionTrace === undefined ? 0 : now();
      const rawFile = await this.rawHandle.getFile();
      if (rawFile.size !== this.plan.outputFloat32Bytes) {
        throw new Error(
          `ACE raw snapshot has ${rawFile.size} bytes; ` +
            `expected ${this.plan.outputFloat32Bytes}`,
        );
      }
      rawSnapshot = rawFile.slice(
        0,
        rawFile.size,
        "application/x-ace-raw-f32",
      );
      emitTransactionTrace(options.onTransactionTrace, {
        operation: "raw-snapshot-validation",
        startedAtMs,
        completedAtMs: options.onTransactionTrace === undefined ? 0 : now(),
        byteLength: rawFile.size,
      });
    } else {
      startedAtMs = options.onTransactionTrace === undefined ? 0 : now();
      await removeEntryIfPresent(this.jobDirectory, RAW_FILE, false);
      emitTransactionTrace(options.onTransactionTrace, {
        operation: "raw-remove",
        startedAtMs,
        completedAtMs: options.onTransactionTrace === undefined ? 0 : now(),
        byteLength: this.plan.outputFloat32Bytes,
      });
    }
    startedAtMs = options.onTransactionTrace === undefined ? 0 : now();
    const file = await this.wavHandle.getFile();
    if (file.size !== wav.wavBytes) {
      throw new Error(
        `ACE committed WAV has ${file.size} bytes; expected ${wav.wavBytes}`,
      );
    }
    // OPFS File.type is implementation-defined. Blob.slice preserves the
    // immutable backing snapshot while pinning the protocol's exact MIME type.
    const audio = file.slice(0, file.size, "audio/wav");
    emitTransactionTrace(options.onTransactionTrace, {
      operation: "wav-snapshot-validation",
      startedAtMs,
      completedAtMs: options.onTransactionTrace === undefined ? 0 : now(),
      byteLength: file.size,
    });
    // Remove incomplete state before the final committed marker is created. If
    // either fallible operation fails, no result is returned and marker-less
    // WAV files remain recognizable by the age-based startup recovery.
    startedAtMs = options.onTransactionTrace === undefined ? 0 : now();
    await removeEntryIfPresent(this.jobDirectory, ACTIVE_MARKER_FILE, false);
    await this.jobDirectory.getFileHandle(COMMITTED_MARKER_FILE, {
      create: true,
    });
    emitTransactionTrace(options.onTransactionTrace, {
      operation: "durable-publish-markers",
      startedAtMs,
      completedAtMs: options.onTransactionTrace === undefined ? 0 : now(),
    });
    this.state = "committed";
    return Object.freeze({
      audio,
      wav,
      transactionId: this.transactionId,
      ...(rawSnapshot === undefined ? {} : { rawSnapshot }),
    });
  }

  /** Idempotent cancellation/error cleanup for an uncommitted transaction. */
  async rollback(): Promise<void> {
    if (this.state === "rolled-back") return;
    if (this.state === "committed") {
      throw new Error("ACE committed audio output cannot be rolled back");
    }
    let closeError: unknown;
    try {
      this.rawFile.close();
    } catch (error) {
      closeError = error;
    }
    try {
      this.wavFile.close();
    } catch (error) {
      closeError ??= error;
    }
    let removalError: unknown;
    try {
      await removeEntryIfPresent(this.audioRoot, this.transactionId, true);
    } catch (error) {
      removalError = error;
    }
    if (removalError !== undefined) throw removalError;
    this.state = "rolled-back";
    if (closeError !== undefined) throw closeError;
  }

  private requireActive(): void {
    if (this.state !== "active") {
      throw new Error(`ACE audio transaction is ${this.state}`);
    }
  }
}

/** Compare two retained OPFS snapshots without materializing either song. */
export async function compareAceRawAudioSnapshots(
  left: Blob,
  right: Blob,
  options: AceRawAudioSnapshotComparisonOptions = {},
): Promise<AceRawAudioSnapshotComparison> {
  if (!(left instanceof Blob) || !(right instanceof Blob)) {
    throw new TypeError("ACE raw snapshot comparison requires two Blobs");
  }
  if (left.size !== right.size || left.size % Float32Array.BYTES_PER_ELEMENT !== 0) {
    throw new RangeError(
      "ACE raw snapshots must have identical whole-F32 byte lengths",
    );
  }
  const requestedBlockBytes = options.blockBytes ?? 1_048_576;
  if (!Number.isSafeInteger(requestedBlockBytes) || requestedBlockBytes < 4) {
    throw new RangeError("ACE raw snapshot block size must be at least four bytes");
  }
  const blockBytes = requestedBlockBytes -
    requestedBlockBytes % Float32Array.BYTES_PER_ELEMENT;
  let exactU32MismatchCount = 0;
  let leftNonFiniteCount = 0;
  let rightNonFiniteCount = 0;
  let finitePairCount = 0;
  let maximumAbsoluteDifference = 0;
  let absoluteDifferenceSum = 0;
  let squaredDifferenceSum = 0;
  for (let offset = 0; offset < left.size; offset += blockBytes) {
    options.signal?.throwIfAborted();
    const end = Math.min(left.size, offset + blockBytes);
    const [leftBuffer, rightBuffer] = await Promise.all([
      left.slice(offset, end).arrayBuffer(),
      right.slice(offset, end).arrayBuffer(),
    ]);
    options.signal?.throwIfAborted();
    const leftValues = new Float32Array(leftBuffer);
    const rightValues = new Float32Array(rightBuffer);
    const leftWords = new Uint32Array(leftBuffer);
    const rightWords = new Uint32Array(rightBuffer);
    for (let index = 0; index < leftValues.length; index += 1) {
      if (leftWords[index] !== rightWords[index]) exactU32MismatchCount += 1;
      const leftValue = leftValues[index]!;
      const rightValue = rightValues[index]!;
      const leftFinite = Number.isFinite(leftValue);
      const rightFinite = Number.isFinite(rightValue);
      if (!leftFinite) leftNonFiniteCount += 1;
      if (!rightFinite) rightNonFiniteCount += 1;
      if (!leftFinite || !rightFinite) continue;
      const difference = Math.abs(leftValue - rightValue);
      finitePairCount += 1;
      maximumAbsoluteDifference = Math.max(
        maximumAbsoluteDifference,
        difference,
      );
      absoluteDifferenceSum += difference;
      squaredDifferenceSum += difference * difference;
    }
  }
  return Object.freeze({
    schema: "ace-raw-audio-snapshot-comparison-v1",
    byteLength: left.size,
    sampleCount: left.size / Float32Array.BYTES_PER_ELEMENT,
    exactU32MismatchCount,
    leftNonFiniteCount,
    rightNonFiniteCount,
    finitePairCount,
    maximumAbsoluteDifference,
    meanAbsoluteDifference: finitePairCount === 0
      ? 0
      : absoluteDifferenceSum / finitePairCount,
    rootMeanSquareDifference: finitePairCount === 0
      ? 0
      : Math.sqrt(squaredDifferenceSum / finitePairCount),
  });
}

function emitTransactionTrace(
  sink: ((trace: AceAudioOutputTransactionTrace) => void) | undefined,
  value: Readonly<{
    operation: AceAudioOutputTransactionTrace["operation"];
    startedAtMs: number;
    completedAtMs: number;
    byteLength?: number;
  }>,
): void {
  if (sink === undefined) return;
  try {
    sink(Object.freeze({
      schema: "ace-audio-output-transaction-capture-v1",
      operation: value.operation,
      startedAtMs: value.startedAtMs,
      completedAtMs: value.completedAtMs,
      wallMs: Math.max(0, value.completedAtMs - value.startedAtMs),
      ...(value.byteLength === undefined ? {} : { byteLength: value.byteLength }),
    }));
  } catch {
    // Capture is observational and cannot change output publication.
  }
}

function defaultNow(): number {
  return globalThis.performance?.now() ?? Date.now();
}

/**
 * Remove abandoned worker transactions without touching committed outputs.
 *
 * A generous age safety margin protects normal (at most four-minute) jobs in
 * another tab that is actively generating in the same OPFS root. This is not a
 * renewable cross-tab lock: a live job suspended beyond the margin can become
 * eligible. Crashed worker handles are released by the browser and their
 * incomplete files age into eligibility for a later startup.
 */
export async function recoverStaleAceAudioOutputs(
  storage: AceAudioOutputStorage = navigator.storage,
  options: AceAudioOutputRecoveryOptions = {},
): Promise<readonly string[]> {
  const {
    signal,
    nowMs = Date.now(),
    minimumIncompleteAgeMs = DEFAULT_STALE_INCOMPLETE_AGE_MS,
  } = options;
  if (!Number.isFinite(nowMs) || nowMs < 0) {
    throw new RangeError("ACE audio recovery clock must be non-negative");
  }
  if (
    !Number.isFinite(minimumIncompleteAgeMs) ||
    minimumIncompleteAgeMs < 0
  ) {
    throw new RangeError("ACE audio recovery age must be non-negative");
  }
  signal?.throwIfAborted();
  const opfsRoot = await storage.getDirectory();
  let audioRoot: FileSystemDirectoryHandle;
  try {
    audioRoot = await opfsRoot.getDirectoryHandle(AUDIO_DIRECTORY);
  } catch (error) {
    if (isNotFound(error)) return Object.freeze([]);
    throw error;
  }
  const removed: string[] = [];
  const iterableRoot = audioRoot as FileSystemDirectoryHandle & {
    entries(): AsyncIterable<readonly [string, FileSystemHandle]>;
  };
  for await (const [transactionId, entry] of iterableRoot.entries()) {
    signal?.throwIfAborted();
    if (
      entry.kind !== "directory" ||
      !TRANSACTION_ID.test(transactionId)
    ) {
      continue;
    }
    const jobDirectory = entry as FileSystemDirectoryHandle;
    if (await fileExists(jobDirectory, COMMITTED_MARKER_FILE)) continue;
    const newestIncompleteWrite = await newestKnownIncompleteWrite(
      jobDirectory,
    );
    // Unknown/empty entries are preserved: recovery only deletes directories
    // positively identified as our incomplete transaction format.
    if (
      newestIncompleteWrite === undefined ||
      nowMs - newestIncompleteWrite < minimumIncompleteAgeMs
    ) {
      continue;
    }
    try {
      await audioRoot.removeEntry(transactionId, { recursive: true });
      removed.push(transactionId);
    } catch (error) {
      if (!isNotFound(error)) throw error;
    }
  }
  return Object.freeze(removed);
}

/**
 * Delete one committed output after the UI has finished reading/playing it.
 * This is intentionally separate from worker teardown: removing the backing
 * OPFS entry too early can make a returned Blob snapshot unreadable.
 */
export async function releaseAceAudioOutput(
  transactionId: string,
  storage: AceAudioOutputStorage = navigator.storage,
): Promise<void> {
  assertTransactionId(transactionId);
  const opfsRoot = await storage.getDirectory();
  let audioRoot: FileSystemDirectoryHandle;
  try {
    audioRoot = await opfsRoot.getDirectoryHandle(AUDIO_DIRECTORY);
  } catch (error) {
    if (isNotFound(error)) return;
    throw error;
  }
  await removeEntryIfPresent(audioRoot, transactionId, true);
}

class OwnedSyncFile {
  private closed = false;

  constructor(private readonly handle: FileSystemSyncAccessHandle) {}

  write(buffer: ArrayBufferView, options: { readonly at: number }): number {
    this.requireOpen();
    return this.handle.write(buffer, options);
  }

  read(buffer: ArrayBufferView, options: { readonly at: number }): number {
    this.requireOpen();
    return this.handle.read(buffer, options);
  }

  truncate(newSize: number): void {
    this.requireOpen();
    this.handle.truncate(newSize);
  }

  flush(): void {
    this.requireOpen();
    this.handle.flush();
  }

  close(): void {
    if (this.closed) return;
    this.closed = true;
    this.handle.close();
  }

  private requireOpen(): void {
    if (this.closed) throw new Error("ACE audio OPFS handle is closed");
  }
}

function assertTransactionId(transactionId: string): void {
  if (!TRANSACTION_ID.test(transactionId)) {
    throw new TypeError(
      "ACE audio transaction ID must contain 1-96 safe filename characters",
    );
  }
}

async function assertEntryAbsent(
  directory: FileSystemDirectoryHandle,
  name: string,
): Promise<void> {
  try {
    await directory.getDirectoryHandle(name);
  } catch (error) {
    if (isNotFound(error)) return;
    throw error;
  }
  throw new Error(`ACE audio transaction ${name} already exists`);
}

async function removeEntryIfPresent(
  directory: FileSystemDirectoryHandle,
  name: string,
  recursive: boolean,
): Promise<void> {
  try {
    await directory.removeEntry(name, { recursive });
  } catch (error) {
    if (!isNotFound(error)) {
      throw error;
    }
  }
}

async function newestKnownIncompleteWrite(
  directory: FileSystemDirectoryHandle,
): Promise<number | undefined> {
  let newest: number | undefined;
  for (const name of [ACTIVE_MARKER_FILE, RAW_FILE, WAV_FILE]) {
    try {
      const handle = await directory.getFileHandle(name);
      const file = await handle.getFile();
      newest = Math.max(newest ?? 0, file.lastModified);
    } catch (error) {
      if (!isNotFound(error)) throw error;
    }
  }
  return newest;
}

async function fileExists(
  directory: FileSystemDirectoryHandle,
  name: string,
): Promise<boolean> {
  try {
    await directory.getFileHandle(name);
    return true;
  } catch (error) {
    if (isNotFound(error)) return false;
    throw error;
  }
}

function isNotFound(error: unknown): boolean {
  return error instanceof DOMException && error.name === "NotFoundError";
}

function closeWithoutMasking(file: OwnedSyncFile | undefined): void {
  try {
    file?.close();
  } catch {
    // Preserve the authoritative setup failure.
  }
}
