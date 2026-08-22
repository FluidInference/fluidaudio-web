import type { AcePackageFileRecord } from "./manifest.js";
import { AceIncrementalSha256 } from "./sha256.js";

export const ACE_GPU_UPLOAD_CHUNK_BYTES = 4 * 1024 * 1024;
/** Maximum bytes submitted with `writeBuffer` before yielding to the GPU queue. */
export const ACE_GPU_UPLOAD_MAX_QUEUED_BYTES = 64 * 1024 * 1024;

const ACE_GPU_UPLOAD_QUEUE_EMPTY_GAP_MS = 1;

interface AceAuthenticatedGpuSourceProof {
  readonly byteLength: number;
  readonly sha256: string;
}

/*
 * A File is an immutable byte snapshot. Acquisition hashes that exact object
 * before retaining it in ReadyResources; remembering the proof by identity
 * lets later phase uploads avoid hashing and copying the same multi-gigabyte
 * snapshot again. Arbitrary Blob callers never enter this fast path.
 */
const authenticatedGpuSources = new WeakMap<File, AceAuthenticatedGpuSourceProof>();

/** @internal Publish only after hashing this exact immutable File snapshot. */
export function markAceAuthenticatedGpuSource(
  source: File,
  record: Pick<AcePackageFileRecord, "byteLength" | "sha256">,
): void {
  if (
    source.size !== record.byteLength ||
    !Number.isSafeInteger(record.byteLength) ||
    record.byteLength <= 0 ||
    !/^[0-9a-f]{64}$/.test(record.sha256)
  ) {
    throw new TypeError("ACE authenticated GPU source proof does not match its record");
  }
  authenticatedGpuSources.set(source, Object.freeze({
    byteLength: record.byteLength,
    sha256: record.sha256,
  }));
}

/** @internal Test and upload seam; identity, size, and digest must all match. */
export function isAceAuthenticatedGpuSource(
  source: Blob,
  record: Pick<AcePackageFileRecord, "byteLength" | "sha256">,
): source is File {
  if (typeof File === "undefined" || !(source instanceof File)) return false;
  const proof = authenticatedGpuSources.get(source);
  return (
    proof?.byteLength === record.byteLength &&
    proof.sha256 === record.sha256 &&
    source.size === record.byteLength
  );
}

export interface AceGpuUploadProgress {
  readonly file: string;
  readonly uploadedBytes: number;
  readonly totalBytes: number;
}

export interface AceGpuUploadTrace {
  readonly schema: "ace-gpu-upload-capture-v1";
  readonly file: string;
  readonly byteLength: number;
  readonly sha256: string;
  readonly sourceKind: "file" | "blob";
  readonly authentication:
    | "exact-immutable-file-proof-reused"
    | "ordinary-blob-stream-hashed";
  readonly redundantHashPerformed: boolean;
  readonly boundedChunkBytes: number;
  readonly maximumQueuedBytes: number;
  readonly maximumObservedStreamChunkBytes: number;
  readonly maximumOwnedCopyBytes: number;
  readonly streamReadCount: number;
  readonly writeBufferCallCount: number;
  readonly queueDrainCount: number;
  readonly queueEmptyGapCount: number;
  readonly uploadedBytes: number;
  readonly startedAtMs: number;
  readonly completedAtMs: number;
  readonly wallMs: number;
  readonly timing: Readonly<{
    readonly createBufferMs: number;
    readonly streamReadMs: number;
    readonly ownedCopyMs: number;
    readonly incrementalHashMs: number;
    readonly writeBufferMs: number;
    readonly queueDrainMs: number;
    readonly queueEmptyGapMs: number;
    readonly errorScopeMs: number;
  }>;
}

export interface AceGpuUploadOptions {
  readonly signal?: AbortSignal;
  readonly label?: string;
  readonly onProgress?: (progress: AceGpuUploadProgress) => void;
  /** @internal Capture-only attribution; normal uploads omit it. */
  readonly onTrace?: (trace: AceGpuUploadTrace) => void;
  /** @internal Deterministic capture clock. */
  readonly now?: () => number;
  /** @internal Deterministic bounded-queue seam for contract tests. */
  readonly maximumQueuedBytes?: number;
  /** @internal Deterministic queue-empty seam for contract tests. */
  readonly yieldQueueIdle?: () => Promise<void>;
}

/**
 * Allocate and fill one authenticated weight/constant shard without ever
 * materializing the whole file in JavaScript or WASM memory.
 *
 * Ordinary Blob sources are hashed while their bounded chunks are copied to
 * WebGPU. Acquisition-authenticated immutable File snapshots reuse the digest
 * proof attached to that exact object and stream directly without a second
 * hash or JavaScript copy. The returned buffer becomes visible to graph
 * assembly only after exact size/proof and WebGPU error-scope checks pass.
 */
export async function uploadAcePackageFileToGpu(
  device: GPUDevice,
  record: AcePackageFileRecord,
  source: Blob,
  options: AceGpuUploadOptions = {},
): Promise<GPUBuffer> {
  const file = Object.freeze({
    name: record.name,
    byteLength: record.byteLength,
    sha256: record.sha256,
    kind: record.kind,
  });
  if (file.kind !== "weights" && file.kind !== "constant") {
    throw new TypeError(`ACE GPU upload cannot consume ${file.kind} files`);
  }
  if (
    !Number.isSafeInteger(file.byteLength) ||
    file.byteLength <= 0 ||
    file.byteLength % 4 !== 0 ||
    source.size !== file.byteLength
  ) {
    throw new RangeError("ACE GPU upload source length must exactly match an aligned shard");
  }
  if (!/^[0-9a-f]{64}$/.test(file.sha256)) {
    throw new TypeError("ACE GPU upload requires a canonical SHA-256 digest");
  }
  if (
    file.byteLength > device.limits.maxBufferSize ||
    file.byteLength > device.limits.maxStorageBufferBindingSize
  ) {
    throw new RangeError(`${file.name} exceeds this device's storage-buffer limits`);
  }
  options.signal?.throwIfAborted();
  const capture = options.onTrace !== undefined;
  const now = options.now ?? defaultNow;
  const startedAtMs = capture ? now() : 0;
  let createBufferMs = 0;
  let streamReadMs = 0;
  let ownedCopyMs = 0;
  let incrementalHashMs = 0;
  let writeBufferMs = 0;
  let queueDrainMs = 0;
  let queueEmptyGapMs = 0;
  let errorScopeMs = 0;
  let maximumObservedStreamChunkBytes = 0;
  let maximumOwnedCopyBytes = 0;
  let streamReadCount = 0;
  let writeBufferCallCount = 0;
  let queueDrainCount = 0;
  let queueEmptyGapCount = 0;
  const maximumQueuedBytes =
    options.maximumQueuedBytes ?? ACE_GPU_UPLOAD_MAX_QUEUED_BYTES;
  if (
    !Number.isSafeInteger(maximumQueuedBytes) ||
    maximumQueuedBytes < 4 ||
    maximumQueuedBytes % 4 !== 0
  ) {
    throw new RangeError(
      "ACE GPU upload queue budget must be a positive four-byte-aligned integer",
    );
  }

  // Error scopes are device-global. The dedicated worker's FIFO graph owner
  // must exclusively own the device for this entire streamed operation so
  // unrelated commands cannot be captured and misattributed here.
  device.pushErrorScope("internal");
  device.pushErrorScope("out-of-memory");
  device.pushErrorScope("validation");
  let scopesOpen = true;
  let buffer: GPUBuffer | undefined;
  let reader: ReadableStreamDefaultReader<Uint8Array> | undefined;
  try {
    const createBufferStartedAt = capture ? now() : 0;
    buffer = device.createBuffer({
      label: options.label ?? `ace-model-${file.name}`,
      size: file.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    if (capture) {
      createBufferMs += nonnegativeElapsed(now(), createBufferStartedAt);
    }
    const authenticatedSnapshot = isAceAuthenticatedGpuSource(source, file);
    const hash = authenticatedSnapshot ? undefined : new AceIncrementalSha256();
    reader = source.stream().getReader();
    let readBytes = 0;
    let uploadedBytes = 0;
    let queuedBytes = 0;
    let needsIdleBeforeNextWrite = false;
    let carry = new Uint8Array(0);
    const targetBuffer = buffer;
    const uploadAligned = async (
      bytes: Uint8Array<ArrayBuffer>,
    ): Promise<void> => {
      if (bytes.byteLength % 4 !== 0) {
        throw new Error("ACE GPU upload attempted an unaligned queue write");
      }
      let sourceOffset = 0;
      while (sourceOffset < bytes.byteLength) {
        options.signal?.throwIfAborted();
        if (needsIdleBeforeNextWrite) {
          const idleStartedAt = capture ? now() : 0;
          await (options.yieldQueueIdle ?? yieldQueueIdle)();
          if (capture) {
            queueEmptyGapMs += nonnegativeElapsed(now(), idleStartedAt);
            queueEmptyGapCount += 1;
          }
          needsIdleBeforeNextWrite = false;
          options.signal?.throwIfAborted();
        }
        const writeBytes = Math.min(
          maximumQueuedBytes - queuedBytes,
          bytes.byteLength - sourceOffset,
        );
        const writeStartedAt = capture ? now() : 0;
        device.queue.writeBuffer(
          targetBuffer,
          uploadedBytes,
          bytes.subarray(sourceOffset, sourceOffset + writeBytes),
        );
        if (capture) {
          writeBufferMs += nonnegativeElapsed(now(), writeStartedAt);
          writeBufferCallCount += 1;
        }
        uploadedBytes += writeBytes;
        queuedBytes += writeBytes;
        sourceOffset += writeBytes;
        if (queuedBytes === maximumQueuedBytes) {
          const drainStartedAt = capture ? now() : 0;
          await device.queue.onSubmittedWorkDone();
          if (capture) {
            queueDrainMs += nonnegativeElapsed(now(), drainStartedAt);
            queueDrainCount += 1;
          }
          queuedBytes = 0;
          needsIdleBeforeNextWrite = true;
          options.signal?.throwIfAborted();
        }
      }
    };
    while (true) {
      options.signal?.throwIfAborted();
      const readStartedAt = capture ? now() : 0;
      const item = await reader.read();
      if (capture) {
        streamReadMs += nonnegativeElapsed(now(), readStartedAt);
        streamReadCount += 1;
      }
      if (item.done) break;
      for (
        let outerOffset = 0;
        outerOffset < item.value.byteLength;
        outerOffset += ACE_GPU_UPLOAD_CHUNK_BYTES
      ) {
        options.signal?.throwIfAborted();
        // Ordinary Blob inputs hash and upload the same owned copy. Although
        // File/Blob streams are not SharedArrayBuffer-backed, the Blob-shaped
        // API must not permit a mutable producer to create a digest/GPU-byte
        // time-of-check/time-of-use gap. The authenticated File path already
        // proved this exact immutable object and can use the observed view.
        const observedChunk = item.value.subarray(
          outerOffset,
          Math.min(outerOffset + ACE_GPU_UPLOAD_CHUNK_BYTES, item.value.byteLength),
        );
        if (capture) {
          maximumObservedStreamChunkBytes = Math.max(
            maximumObservedStreamChunkBytes,
            observedChunk.byteLength,
          );
        }
        let chunk: Uint8Array<ArrayBuffer>;
        if (authenticatedSnapshot) {
          if (!(observedChunk.buffer instanceof ArrayBuffer)) {
            throw new Error(
              `${file.name} authenticated File stream returned shared storage`,
            );
          }
          chunk = new Uint8Array(
            observedChunk.buffer,
            observedChunk.byteOffset,
            observedChunk.byteLength,
          );
        } else {
          const copyStartedAt = capture ? now() : 0;
          chunk = Uint8Array.from(observedChunk);
          if (capture) {
            ownedCopyMs += nonnegativeElapsed(now(), copyStartedAt);
            maximumOwnedCopyBytes = Math.max(
              maximumOwnedCopyBytes,
              chunk.byteLength,
            );
          }
        }
        readBytes += chunk.byteLength;
        if (readBytes > file.byteLength) {
          throw new Error(`${file.name} exceeded its manifest byte length during upload`);
        }
        if (hash !== undefined) {
          const hashStartedAt = capture ? now() : 0;
          hash.update(chunk);
          if (capture) {
            incrementalHashMs += nonnegativeElapsed(now(), hashStartedAt);
          }
        }
        let offset = 0;
        if (carry.byteLength !== 0) {
          const needed = Math.min(4 - carry.byteLength, chunk.byteLength);
          const combined = new Uint8Array(carry.byteLength + needed);
          combined.set(carry);
          combined.set(chunk.subarray(0, needed), carry.byteLength);
          if (capture) maximumOwnedCopyBytes = Math.max(maximumOwnedCopyBytes, 4);
          offset = needed;
          if (combined.byteLength === 4) {
            await uploadAligned(combined);
            carry = new Uint8Array(0);
          } else {
            carry = combined;
          }
        }
        const alignedBytes = (chunk.byteLength - offset) & ~3;
        if (alignedBytes !== 0) {
          // WebGPU snapshots this owned source during `writeBuffer`'s content-
          // timeline steps, so a subview does not require a second 4 MiB copy.
          await uploadAligned(chunk.subarray(offset, offset + alignedBytes));
          offset += alignedBytes;
        }
        if (offset < chunk.byteLength) {
          carry = Uint8Array.from(chunk.subarray(offset));
          if (capture) {
            maximumOwnedCopyBytes = Math.max(
              maximumOwnedCopyBytes,
              carry.byteLength,
            );
          }
        }
        options.onProgress?.({
          file: file.name,
          uploadedBytes: readBytes,
          totalBytes: file.byteLength,
        });
      }
    }
    options.signal?.throwIfAborted();
    if (
      readBytes !== file.byteLength ||
      uploadedBytes !== file.byteLength ||
      carry.byteLength !== 0
    ) {
      throw new Error(`${file.name} ended before its aligned declared length`);
    }
    // A phase may comprise many physical files. Drain each file's final tail
    // so queued write staging cannot accumulate again across shard boundaries.
    // This is the final GPU work for this file, so no artificial idle follows.
    if (queuedBytes !== 0) {
      const drainStartedAt = capture ? now() : 0;
      await device.queue.onSubmittedWorkDone();
      if (capture) {
        queueDrainMs += nonnegativeElapsed(now(), drainStartedAt);
        queueDrainCount += 1;
      }
      queuedBytes = 0;
      options.signal?.throwIfAborted();
    }
    if (hash !== undefined) {
      const actualDigest = hash.digestHex();
      if (actualDigest !== file.sha256) {
        throw new Error(`${file.name} SHA-256 mismatch during GPU upload: ${actualDigest}`);
      }
    }
    const errorScopeStartedAt = capture ? now() : 0;
    const gpuError = await collectGpuErrorScopes(device);
    if (capture) {
      errorScopeMs += nonnegativeElapsed(now(), errorScopeStartedAt);
    }
    scopesOpen = false;
    if (gpuError !== undefined) throw gpuError;
    options.onProgress?.({
      file: file.name,
      uploadedBytes: file.byteLength,
      totalBytes: file.byteLength,
    });
    if (capture) {
      const completedAtMs = now();
      emitTrace(options.onTrace, Object.freeze({
        schema: "ace-gpu-upload-capture-v1",
        file: file.name,
        byteLength: file.byteLength,
        sha256: file.sha256,
        sourceKind:
          typeof File !== "undefined" && source instanceof File
            ? "file"
            : "blob",
        authentication: authenticatedSnapshot
          ? "exact-immutable-file-proof-reused"
          : "ordinary-blob-stream-hashed",
        redundantHashPerformed: false,
        boundedChunkBytes: ACE_GPU_UPLOAD_CHUNK_BYTES,
        maximumQueuedBytes,
        maximumObservedStreamChunkBytes,
        maximumOwnedCopyBytes,
        streamReadCount,
        writeBufferCallCount,
        queueDrainCount,
        queueEmptyGapCount,
        uploadedBytes,
        startedAtMs,
        completedAtMs,
        wallMs: nonnegativeElapsed(completedAtMs, startedAtMs),
        timing: Object.freeze({
          createBufferMs,
          streamReadMs,
          ownedCopyMs,
          incrementalHashMs,
          writeBufferMs,
          queueDrainMs,
          queueEmptyGapMs,
          errorScopeMs,
        }),
      }));
    }
    return buffer;
  } catch (error) {
    if (reader !== undefined) {
      try {
        await reader.cancel(error);
      } catch {
        // Preserve the upload, integrity, or cancellation failure.
      }
    }
    if (scopesOpen) {
      await discardGpuErrorScopes(device);
      scopesOpen = false;
    }
    buffer?.destroy();
    throw error;
  } finally {
    reader?.releaseLock();
  }
}

function emitTrace(
  sink: ((trace: AceGpuUploadTrace) => void) | undefined,
  trace: AceGpuUploadTrace,
): void {
  try {
    sink?.(trace);
  } catch {
    // Capture is observational and cannot alter package publication.
  }
}

function defaultNow(): number {
  return globalThis.performance?.now() ?? Date.now();
}

function nonnegativeElapsed(completedAtMs: number, startedAtMs: number): number {
  return Math.max(0, completedAtMs - startedAtMs);
}

async function yieldQueueIdle(): Promise<void> {
  await new Promise<void>((resolve) => {
    setTimeout(resolve, ACE_GPU_UPLOAD_QUEUE_EMPTY_GAP_MS);
  });
}

async function collectGpuErrorScopes(device: GPUDevice): Promise<unknown | undefined> {
  const validation = device.popErrorScope();
  const outOfMemory = device.popErrorScope();
  const internal = device.popErrorScope();
  const [validationResult, memoryResult, internalResult] = await Promise.allSettled([
    validation,
    outOfMemory,
    internal,
  ]);
  if (validationResult.status === "rejected") return validationResult.reason;
  if (memoryResult.status === "rejected") return memoryResult.reason;
  if (internalResult.status === "rejected") return internalResult.reason;
  if (validationResult.value !== null) {
    return new Error(`ACE GPU upload failed validation: ${validationResult.value.message}`);
  }
  if (memoryResult.value !== null) {
    return new Error(`ACE GPU upload exhausted GPU memory: ${memoryResult.value.message}`);
  }
  if (internalResult.value !== null) {
    return new Error(`ACE GPU upload failed internally: ${internalResult.value.message}`);
  }
  return undefined;
}

async function discardGpuErrorScopes(device: GPUDevice): Promise<void> {
  const validation = device.popErrorScope();
  const outOfMemory = device.popErrorScope();
  const internal = device.popErrorScope();
  await Promise.allSettled([validation, outOfMemory, internal]);
}
