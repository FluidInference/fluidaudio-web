import type { AcePackageFileRecord } from "./manifest.js";
import { AceIncrementalSha256 } from "./sha256.js";
import { parseStrictJson } from "./strict-json.js";

const CACHE_DIRECTORY = "ace-step-1.5.wgsl-model-cache-v1";
const PAYLOAD_FILE = "payload.bin";
const MARKER_FILE = "verified.json";
const MARKER_FORMAT = "ace-opfs-verified-v1";
const CACHE_READ_CHUNK_BYTES = 4 * 1024 * 1024;

export interface AceStoredModelInfo {
  readonly supported: boolean;
  readonly persisted: boolean;
  readonly usageBytes?: number;
  readonly quotaBytes?: number;
}

export interface AceModelCacheInfo {
  readonly supported: boolean;
  readonly persisted: boolean;
  /** Marker-qualified, size-qualified content-addressed payloads. */
  readonly assetCount: number;
  readonly sizeBytes: number;
  /** Incomplete or invalid cache identities retained for resume or recovery. */
  readonly partialAssetCount: number;
}

export interface AceOpfsPartialAsset {
  readonly resumeOffset: number;
  /** Hash the exact partial prefix before issuing a Range request. */
  hashExistingPrefix(signal?: AbortSignal): Promise<void>;
  /** Truncate tentative bytes and restart from offset zero. */
  restart(): Promise<void>;
  /** Append at exactly the current offset. */
  write(offset: number, bytes: Uint8Array): Promise<void>;
  /** Publish only after the caller has checked full byte length and SHA-256. */
  commit(): Promise<File>;
  /** Close while retaining non-visible partial bytes for a later resume. */
  rollback(): Promise<void>;
  /** Close and delete the invalid partial identity. */
  discard(): Promise<void>;
}

interface CacheMarker {
  readonly format: typeof MARKER_FORMAT;
  readonly byteLength: number;
  readonly sha256: string;
}

/**
 * Content-addressed OPFS cache primitive for the dedicated inference worker.
 * A payload without its exact marker is always partial and never a cache hit.
 * The final streaming consumer must still hash cached bytes while uploading so
 * post-commit corruption cannot enter GPU state unnoticed.
 */
export class AceOpfsModelCache {
  private readonly activeDigests = new Set<string>();

  constructor(private readonly cacheRoot: FileSystemDirectoryHandle) {}

  static async open(
    storage: Pick<StorageManager, "getDirectory"> = navigator.storage,
  ): Promise<AceOpfsModelCache> {
    const opfsRoot = await storage.getDirectory();
    const cacheRoot = await opfsRoot.getDirectoryHandle(CACHE_DIRECTORY, {
      create: true,
    });
    return new AceOpfsModelCache(cacheRoot);
  }

  /** Return a size/marker-qualified candidate; bytes remain untrusted. */
  async openCandidate(file: AcePackageFileRecord): Promise<File | undefined> {
    const record = snapshotCacheRecord(file);
    let directory: FileSystemDirectoryHandle;
    try {
      directory = await this.cacheRoot.getDirectoryHandle(record.sha256);
    } catch (error) {
      if (isNotFound(error)) return undefined;
      throw error;
    }
    const marker = await readMarker(directory);
    if (
      marker === undefined ||
      marker.sha256 !== record.sha256 ||
      marker.byteLength !== record.byteLength
    ) {
      return undefined;
    }
    try {
      const payload = await (await directory.getFileHandle(PAYLOAD_FILE)).getFile();
      return payload.size === record.byteLength ? payload : undefined;
    } catch (error) {
      if (isNotFound(error)) return undefined;
      throw error;
    }
  }

  /**
   * Open resumable tentative storage. Only one writer for a digest may exist in
   * this worker. Existing verified identities are immutable.
   */
  async begin(file: AcePackageFileRecord): Promise<AceOpfsPartialAsset> {
    const record = snapshotCacheRecord(file);
    if (this.activeDigests.has(record.sha256)) {
      throw new DOMException("ACE cache identity already has an active writer", "InvalidStateError");
    }
    // Reserve synchronously before the first await. Two same-worker callers
    // must never both pass an asynchronous candidate check and open writers.
    this.activeDigests.add(record.sha256);
    let writable: FileSystemWritableFileStream | undefined;
    try {
      if (await this.openCandidate(record)) {
        throw new DOMException(
          "ACE cache identity is already verified",
          "InvalidModificationError",
        );
      }
      const directory = await this.cacheRoot.getDirectoryHandle(record.sha256, {
        create: true,
      });
      await removeEntryIfPresent(directory, MARKER_FILE);
      const handle = await directory.getFileHandle(PAYLOAD_FILE, { create: true });
      let snapshot = await handle.getFile();
      writable = await handle.createWritable({ keepExistingData: true });
      let cursor = snapshot.size;
      if (cursor > record.byteLength) {
        await writable.truncate(0);
        cursor = 0;
        snapshot = await handle.getFile();
      }
      await writable.seek(cursor);
      return new OpfsPartialAsset(
        record,
        directory,
        handle,
        writable,
        snapshot,
        cursor,
        () => this.activeDigests.delete(record.sha256),
        () => this.removeReservedIdentity(record.sha256),
        () => this.remove(record),
      );
    } catch (error) {
      if (writable !== undefined) {
        try {
          await writable.abort(error);
        } catch {
          // The setup failure remains authoritative. In particular, never
          // leak the digest reservation because abort itself also failed.
        }
      }
      this.activeDigests.delete(record.sha256);
      throw error;
    }
  }

  async remove(file: Pick<AcePackageFileRecord, "sha256">): Promise<void> {
    const digest = snapshotDigest(file.sha256);
    if (this.activeDigests.has(digest)) {
      throw new DOMException("Cannot remove an active ACE cache writer", "InvalidStateError");
    }
    this.activeDigests.add(digest);
    try {
      await this.removeReservedIdentity(digest);
    } finally {
      this.activeDigests.delete(digest);
    }
  }

  /** Delete while the caller still owns the digest reservation. */
  private async removeReservedIdentity(digest: string): Promise<void> {
    try {
      await this.cacheRoot.removeEntry(digest, { recursive: true });
    } catch (error) {
      if (!isNotFound(error)) throw error;
    }
  }
}

class OpfsPartialAsset implements AceOpfsPartialAsset {
  private cursor: number;
  private closed = false;
  private hash = new AceIncrementalSha256();
  private prefixHashed: boolean;

  constructor(
    private readonly record: AcePackageFileRecord,
    private readonly directory: FileSystemDirectoryHandle,
    private readonly handle: FileSystemFileHandle,
    private readonly writable: FileSystemWritableFileStream,
    private snapshot: File,
    cursor: number,
    private readonly release: () => void,
    private readonly removeIdentity: () => Promise<void>,
    private readonly removeUnreservedIdentity: () => Promise<void>,
  ) {
    this.cursor = cursor;
    this.prefixHashed = cursor === 0;
  }

  get resumeOffset(): number {
    return this.cursor;
  }

  async hashExistingPrefix(signal?: AbortSignal): Promise<void> {
    this.assertOpen();
    if (this.prefixHashed) return;
    this.hash = new AceIncrementalSha256();
    if (this.snapshot.size !== this.cursor) {
      throw new Error("ACE OPFS partial snapshot does not match its resume offset");
    }
    const reader = this.snapshot.stream().getReader();
    let readBytes = 0;
    try {
      while (true) {
        signal?.throwIfAborted();
        const item = await reader.read();
        if (item.done) break;
        for (let offset = 0; offset < item.value.byteLength; offset += CACHE_READ_CHUNK_BYTES) {
          const chunk = item.value.subarray(
            offset,
            Math.min(offset + CACHE_READ_CHUNK_BYTES, item.value.byteLength),
          );
          readBytes += chunk.byteLength;
          if (readBytes > this.cursor) {
            throw new Error("ACE OPFS partial grew while its prefix was hashed");
          }
          this.hash.update(chunk);
        }
      }
      if (readBytes !== this.cursor) {
        throw new Error("ACE OPFS partial produced a short prefix read");
      }
      this.prefixHashed = true;
    } catch (error) {
      try {
        await reader.cancel(error);
      } catch {
        // Preserve the read or cancellation failure.
      }
      throw error;
    } finally {
      reader.releaseLock();
    }
  }

  async restart(): Promise<void> {
    this.assertOpen();
    await this.writable.truncate(0);
    await this.writable.seek(0);
    this.cursor = 0;
    this.snapshot = new File([], PAYLOAD_FILE);
    this.hash = new AceIncrementalSha256();
    this.prefixHashed = true;
  }

  async write(offset: number, bytes: Uint8Array): Promise<void> {
    this.assertOpen();
    if (!this.prefixHashed) {
      throw new Error("ACE OPFS partial prefix must be hashed before appending");
    }
    if (offset !== this.cursor) {
      throw new RangeError(`ACE OPFS write offset ${offset} does not match ${this.cursor}`);
    }
    const next = this.cursor + bytes.byteLength;
    if (!Number.isSafeInteger(next) || next > this.record.byteLength) {
      throw new RangeError("ACE OPFS write exceeds the manifest byte length");
    }
    // OPFS does not accept a SharedArrayBuffer-backed view. Copying one bounded
    // transport chunk also prevents a producer from mutating it after write.
    const stableBytes = Uint8Array.from(bytes);
    await this.writable.write(stableBytes);
    this.hash.update(stableBytes);
    this.cursor = next;
  }

  async commit(): Promise<File> {
    this.assertOpen();
    if (this.cursor !== this.record.byteLength) {
      throw new Error("Cannot commit an incomplete ACE OPFS payload");
    }
    if (!this.prefixHashed) {
      throw new Error("Cannot commit an unhashed ACE OPFS prefix");
    }
    const actualDigest = this.hash.digestHex();
    if (actualDigest !== this.record.sha256) {
      this.closed = true;
      let cleanupFailure: unknown;
      try {
        try {
          await this.writable.close();
        } catch (error) {
          cleanupFailure = error;
          // The digest mismatch remains authoritative, but deletion is still
          // attempted while this transaction owns the identity reservation.
        }
        try {
          await this.removeIdentity();
        } catch (error) {
          cleanupFailure ??= error;
          // A marker was never published, so even failed cleanup cannot make
          // these unauthenticated bytes a cache hit.
        }
      } finally {
        this.release();
      }
      throw new Error(`ACE OPFS payload SHA-256 mismatch: ${actualDigest}`, {
        ...(cleanupFailure === undefined ? {} : { cause: cleanupFailure }),
      });
    }
    this.closed = true;
    try {
      await this.writable.close();
      const payload = await this.handle.getFile();
      if (payload.size !== this.record.byteLength) {
        throw new Error("ACE OPFS payload length changed before commit");
      }
      const marker: CacheMarker = {
        format: MARKER_FORMAT,
        byteLength: this.record.byteLength,
        sha256: this.record.sha256,
      };
      const markerHandle = await this.directory.getFileHandle(MARKER_FILE, {
        create: true,
      });
      const markerWriter = await markerHandle.createWritable();
      try {
        await markerWriter.write(
          new TextEncoder().encode(`${JSON.stringify(marker)}\n`),
        );
        await markerWriter.close();
      } catch (error) {
        try {
          await markerWriter.abort(error);
        } catch {
          // Preserve the marker write/close failure.
        }
        throw error;
      }
      return payload;
    } finally {
      this.release();
    }
  }

  async rollback(): Promise<void> {
    if (this.closed) return;
    this.closed = true;
    try {
      await this.writable.close();
    } catch (error) {
      try {
        await this.writable.abort(error);
      } catch {
        // Preserve the primary close failure.
      }
      throw error;
    } finally {
      this.release();
    }
  }

  async discard(): Promise<void> {
    if (this.closed) {
      await this.removeUnreservedIdentity();
      return;
    }
    this.closed = true;
    try {
      try {
        await this.writable.close();
      } finally {
        // Keep the reservation until recursive deletion is finished. This
        // prevents a new writer from racing into the identity being removed.
        await this.removeIdentity();
      }
    } finally {
      this.release();
    }
  }

  private assertOpen(): void {
    if (this.closed) {
      throw new DOMException("ACE OPFS transaction is closed", "InvalidStateError");
    }
  }
}

export async function inspectAceModelStorage(
  storage: StorageManager | undefined = globalThis.navigator?.storage,
): Promise<AceStoredModelInfo> {
  if (storage === undefined || typeof storage.getDirectory !== "function") {
    return { supported: false, persisted: false };
  }
  const [persisted, estimate]: [boolean, StorageEstimate] = await Promise.all([
    storage.persisted().catch(() => false),
    storage.estimate().catch((): StorageEstimate => ({})),
  ]);
  return {
    supported: true,
    persisted,
    ...(estimate.usage === undefined ? {} : { usageBytes: estimate.usage }),
    ...(estimate.quota === undefined ? {} : { quotaBytes: estimate.quota }),
  };
}

/** Inspect only the model cache owned by this package, excluding audio output. */
export async function inspectAceModelCache(
  storage: Pick<StorageManager, "getDirectory" | "persisted"> | undefined =
    globalThis.navigator?.storage,
): Promise<AceModelCacheInfo> {
  if (storage === undefined || typeof storage.getDirectory !== "function") {
    return {
      supported: false,
      persisted: false,
      assetCount: 0,
      sizeBytes: 0,
      partialAssetCount: 0,
    };
  }
  const [opfsRoot, persisted] = await Promise.all([
    storage.getDirectory(),
    storage.persisted().catch(() => false),
  ]);
  let cacheRoot: FileSystemDirectoryHandle;
  try {
    cacheRoot = await opfsRoot.getDirectoryHandle(CACHE_DIRECTORY);
  } catch (error) {
    if (isNotFound(error)) {
      return {
        supported: true,
        persisted,
        assetCount: 0,
        sizeBytes: 0,
        partialAssetCount: 0,
      };
    }
    throw error;
  }

  let assetCount = 0;
  let sizeBytes = 0;
  let partialAssetCount = 0;
  const entries = (cacheRoot as unknown as {
    entries(): AsyncIterable<
      [string, FileSystemDirectoryHandle | FileSystemFileHandle]
    >;
  }).entries();
  for await (const [name, handle] of entries) {
    if (handle.kind !== "directory" || !/^[0-9a-f]{64}$/.test(name)) {
      partialAssetCount += 1;
      continue;
    }
    const marker = await readMarker(handle);
    if (marker === undefined || marker.sha256 !== name) {
      partialAssetCount += 1;
      continue;
    }
    try {
      const payload = await (await handle.getFileHandle(PAYLOAD_FILE)).getFile();
      if (payload.size !== marker.byteLength) {
        partialAssetCount += 1;
        continue;
      }
      assetCount += 1;
      sizeBytes += payload.size;
    } catch (error) {
      if (!isNotFound(error)) throw error;
      partialAssetCount += 1;
    }
  }
  return {
    supported: true,
    persisted,
    assetCount,
    sizeBytes,
    partialAssetCount,
  };
}

/**
 * Delete the complete package-owned model cache after active runtimes have
 * been disposed. Audio outputs and unrelated origin storage are untouched.
 */
export async function deleteAceModelCache(
  storage: Pick<StorageManager, "getDirectory"> | undefined =
    globalThis.navigator?.storage,
): Promise<boolean> {
  if (storage === undefined || typeof storage.getDirectory !== "function") {
    return false;
  }
  const opfsRoot = await storage.getDirectory();
  try {
    await opfsRoot.removeEntry(CACHE_DIRECTORY, { recursive: true });
    return true;
  } catch (error) {
    if (isNotFound(error)) return false;
    throw error;
  }
}

async function readMarker(
  directory: FileSystemDirectoryHandle,
): Promise<CacheMarker | undefined> {
  try {
    const file = await (await directory.getFileHandle(MARKER_FILE)).getFile();
    if (file.size > 256) return undefined;
    const decoded = parseStrictJson(await file.text());
    if (
      !isRecord(decoded) ||
      !hasExactKeys(decoded, ["format", "byteLength", "sha256"]) ||
      decoded.format !== MARKER_FORMAT ||
      !Number.isSafeInteger(decoded.byteLength) ||
      (decoded.byteLength as number) < 0 ||
      typeof decoded.sha256 !== "string" ||
      !/^[0-9a-f]{64}$/.test(decoded.sha256)
    ) {
      return undefined;
    }
    return decoded as unknown as CacheMarker;
  } catch (error) {
    if (isNotFound(error) || error instanceof SyntaxError) return undefined;
    throw error;
  }
}

async function removeEntryIfPresent(
  directory: FileSystemDirectoryHandle,
  name: string,
): Promise<void> {
  try {
    await directory.removeEntry(name);
  } catch (error) {
    if (!isNotFound(error)) throw error;
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function hasExactKeys(value: Record<string, unknown>, expected: readonly string[]): boolean {
  const actual = Object.keys(value).sort();
  const sortedExpected = [...expected].sort();
  return (
    actual.length === sortedExpected.length &&
    actual.every((key, index) => key === sortedExpected[index])
  );
}

function isNotFound(error: unknown): boolean {
  return error instanceof DOMException && error.name === "NotFoundError";
}

function assertCacheRecord(file: AcePackageFileRecord): void {
  assertDigest(file.sha256);
  if (!Number.isSafeInteger(file.byteLength) || file.byteLength < 0) {
    throw new TypeError("ACE cache byteLength must be a non-negative safe integer");
  }
}

function snapshotCacheRecord(file: AcePackageFileRecord): AcePackageFileRecord {
  assertCacheRecord(file);
  return Object.freeze({
    name: file.name,
    byteLength: file.byteLength,
    sha256: file.sha256,
    kind: file.kind,
  });
}

function snapshotDigest(value: string): string {
  assertDigest(value);
  return value;
}

function assertDigest(value: string): void {
  if (!/^[0-9a-f]{64}$/.test(value)) {
    throw new TypeError("ACE cache SHA-256 must be a lowercase hexadecimal digest");
  }
}
