import {
  AceOpfsModelCache,
  type AceOpfsPartialAsset,
} from "./cache.js";
import type {
  AcePackageFileKind,
  AcePackageFileRecord,
  AcePackageManifest,
} from "./manifest.js";
import { fetchAceModelAssetResumable } from "./package.js";
import { AceIncrementalSha256 } from "./sha256.js";
import { markAceAuthenticatedGpuSource } from "./gpu-upload.js";

export const ACE_MODEL_STORAGE_HEADROOM_BYTES = 512 * 1024 * 1024;
const ACE_MODEL_CACHE_HASH_CHUNK_BYTES = 4 * 1024 * 1024;
export const ACE_MODEL_CACHE_WEBCRYPTO_MAX_FILE_BYTES = 128 * 1024 * 1024;

export type AceCacheAuthenticationOwner =
  | "scalar-stream"
  | "webcrypto-whole-file";

/** Production warm-cache authentication uses the bounded OPT-0071 owner. */
export const ACE_PRODUCTION_CACHE_AUTHENTICATION_OWNER:
  AceCacheAuthenticationOwner = "webcrypto-whole-file";

const RUNTIME_FILE_KINDS = new Set<AcePackageFileKind>([
  "weights",
  "constant",
  "upstream-asset",
]);

export interface AceModelCacheBackend {
  openCandidate(file: AcePackageFileRecord): Promise<File | undefined>;
  begin(file: AcePackageFileRecord): Promise<AceOpfsPartialAsset>;
  remove(file: Pick<AcePackageFileRecord, "sha256">): Promise<void>;
}

export interface AceModelAcquisitionPlan {
  readonly files: readonly AcePackageFileRecord[];
  readonly cachedFiles: readonly string[];
  readonly downloadFiles: readonly string[];
  readonly runtimeBytes: number;
  readonly cachedBytes: number;
  /** Physical content-addressed bytes; alias filenames are counted once. */
  readonly downloadBytes: number;
  readonly requiredFreeBytes: number;
  readonly availableFreeBytes?: number;
  readonly quotaSufficient?: boolean;
}

export interface AceAcquireModelOptions {
  readonly manifest: AcePackageManifest;
  readonly manifestUrl: string;
  readonly cache: AceModelCacheBackend;
  readonly signal?: AbortSignal;
  readonly fetch?: typeof fetch;
  readonly storage?: Pick<StorageManager, "estimate">;
  readonly maximumAttempts?: number;
  readonly headroomBytes?: number;
  readonly onFileProgress?: (progress: AceModelAcquisitionProgress) => void;
  /** @internal Capture-only attribution; normal acquisition omits it. */
  readonly onTrace?: (trace: AceModelAcquisitionTrace) => void;
  /** @internal Deterministic capture clock. */
  readonly now?: () => number;
  /** @internal Authenticated benchmark seam; ordinary acquisition uses production. */
  readonly cacheAuthenticationOwner?: AceCacheAuthenticationOwner;
}

export interface AceModelAcquisitionProgress {
  readonly file: string;
  readonly fileIndex: number;
  readonly fileCount: number;
  readonly fileReceivedBytes: number;
  readonly fileBytes: number;
  readonly completedBytes: number;
  readonly totalBytes: number;
  readonly source: "cache" | "network";
}

export type AceModelAcquisitionTrace =
  | Readonly<{
      readonly operation: "cache-authentication";
      readonly file: string;
      readonly byteLength: number;
      readonly expectedSha256: string;
      readonly actualSha256: string | null;
      readonly authenticationOwner: AceCacheAuthenticationOwner;
      readonly matched: boolean;
      readonly receivedBytes: number;
      readonly hashChunkCount: number;
      readonly maximumHashChunkBytes: number;
      readonly startedAtMs: number;
      readonly completedAtMs: number;
      readonly wallMs: number;
      readonly exactImmutableFileProofPublished: boolean;
    }>
  | Readonly<{
      readonly operation: "proof-reuse";
      readonly file: string;
      readonly byteLength: number;
      readonly sha256: string;
      readonly source: "cache" | "network";
      readonly exactImmutableFileIdentity: true;
      readonly redundantHashPerformed: false;
    }>;

export interface AceAcquiredModelFiles {
  readonly plan: AceModelAcquisitionPlan;
  /** Immutable File handles; multi-gigabyte payload bytes are never mirrored. */
  readonly files: ReadonlyMap<string, File>;
}

/** Runtime package records, in canonical manifest order. */
export function aceRuntimePackageFiles(
  manifest: AcePackageManifest,
): readonly AcePackageFileRecord[] {
  return Object.freeze(
    manifest.files
      .filter((file) => RUNTIME_FILE_KINDS.has(file.kind))
      .map((file) => Object.freeze({
        name: file.name,
        byteLength: file.byteLength,
        sha256: file.sha256,
        kind: file.kind,
      })),
  );
}

/**
 * Authenticate marker-qualified cache candidates and inspect storage headroom
 * without requesting persistence or starting a download. Corrupt identities
 * are invalidated so byte and quota accounting describe the real acquisition.
 */
export async function planAceModelAcquisition(
  manifest: AcePackageManifest,
  cache: AceModelCacheBackend,
  storage: Pick<StorageManager, "estimate"> | undefined = globalThis.navigator?.storage,
  headroomBytes = ACE_MODEL_STORAGE_HEADROOM_BYTES,
): Promise<AceModelAcquisitionPlan> {
  return (
    await inspectAceModelAcquisition(
      manifest,
      cache,
      storage,
      headroomBytes,
    )
  ).plan;
}

interface AceInspectedModelAcquisition {
  readonly plan: AceModelAcquisitionPlan;
  readonly cachedByDigest: ReadonlyMap<string, File>;
}

async function inspectAceModelAcquisition(
  manifest: AcePackageManifest,
  cache: AceModelCacheBackend,
  storage: Pick<StorageManager, "estimate"> | undefined,
  headroomBytes: number,
  signal?: AbortSignal,
  onTrace?: (trace: AceModelAcquisitionTrace) => void,
  now: () => number = defaultNow,
  authenticationOwner: AceCacheAuthenticationOwner =
    ACE_PRODUCTION_CACHE_AUTHENTICATION_OWNER,
): Promise<AceInspectedModelAcquisition> {
  if (!Number.isSafeInteger(headroomBytes) || headroomBytes < 0) {
    throw new RangeError("ACE model storage headroom must be a non-negative integer");
  }
  const files = aceRuntimePackageFiles(manifest);
  const cachedFiles: string[] = [];
  const downloadFiles: string[] = [];
  const cachedByDigest = new Map<string, File>();
  const missingDigests = new Set<string>();
  const digestByteLengths = new Map<string, number>();
  let runtimeBytes = 0;
  let cachedBytes = 0;
  let downloadBytes = 0;
  for (const file of files) {
    signal?.throwIfAborted();
    runtimeBytes = checkedAdd(runtimeBytes, file.byteLength);
    const knownByteLength = digestByteLengths.get(file.sha256);
    if (knownByteLength !== undefined && knownByteLength !== file.byteLength) {
      throw new Error(
        `ACE manifest assigns conflicting byte lengths to digest ${file.sha256}`,
      );
    }
    digestByteLengths.set(file.sha256, file.byteLength);
    if (cachedByDigest.has(file.sha256)) {
      cachedFiles.push(file.name);
      cachedBytes = checkedAdd(cachedBytes, file.byteLength);
      continue;
    }
    if (missingDigests.has(file.sha256)) {
      downloadFiles.push(file.name);
      continue;
    }

    const candidate = await cache.openCandidate(file);
    let authentication: AceCachedFileAuthentication | undefined;
    if (candidate !== undefined) {
      const startedAtMs = onTrace === undefined ? 0 : now();
      try {
        authentication = await cachedFileMatchesRecord(
          candidate,
          file,
          authenticationOwner,
          signal,
        );
      } catch (error) {
        signal?.throwIfAborted();
        // An unreadable cache entry is no safer than a digest mismatch. Remove
        // the identity and let the authenticated network path rebuild it.
        authentication = Object.freeze({
          matched: false,
          receivedBytes: 0,
          hashChunkCount: 0,
          maximumHashChunkBytes: 0,
          actualSha256: null,
        });
      }
      if (onTrace !== undefined) {
        const completedAtMs = now();
        emitAcquisitionTrace(onTrace, Object.freeze({
          operation: "cache-authentication",
          file: file.name,
          byteLength: file.byteLength,
          expectedSha256: file.sha256,
          actualSha256: authentication.actualSha256,
          authenticationOwner,
          matched: authentication.matched,
          receivedBytes: authentication.receivedBytes,
          hashChunkCount: authentication.hashChunkCount,
          maximumHashChunkBytes: authentication.maximumHashChunkBytes,
          startedAtMs,
          completedAtMs,
          wallMs: nonnegativeElapsed(completedAtMs, startedAtMs),
          exactImmutableFileProofPublished: authentication.matched,
        }));
      }
    }
    if (candidate === undefined || authentication?.matched !== true) {
      if (candidate !== undefined) await cache.remove(file);
      missingDigests.add(file.sha256);
      downloadFiles.push(file.name);
      downloadBytes = checkedAdd(downloadBytes, file.byteLength);
    } else {
      markAceAuthenticatedGpuSource(candidate, file);
      cachedByDigest.set(file.sha256, candidate);
      cachedFiles.push(file.name);
      cachedBytes = checkedAdd(cachedBytes, file.byteLength);
    }
  }
  const requiredFreeBytes = checkedAdd(downloadBytes, headroomBytes);
  let availableFreeBytes: number | undefined;
  let quotaSufficient: boolean | undefined;
  if (storage !== undefined) {
    try {
      const estimate = await storage.estimate();
      if (
        estimate.quota !== undefined &&
        estimate.usage !== undefined &&
        Number.isSafeInteger(estimate.quota) &&
        Number.isSafeInteger(estimate.usage) &&
        estimate.quota >= estimate.usage
      ) {
        availableFreeBytes = estimate.quota - estimate.usage;
        quotaSufficient = availableFreeBytes >= requiredFreeBytes;
      }
    } catch {
      // Quota estimation is advisory and can be rejected by browsers. Unknown
      // quota must not be misreported as either sufficient or insufficient.
    }
  }
  const plan = Object.freeze({
    files,
    cachedFiles: Object.freeze(cachedFiles),
    downloadFiles: Object.freeze(downloadFiles),
    runtimeBytes,
    cachedBytes,
    downloadBytes,
    requiredFreeBytes,
    ...(availableFreeBytes === undefined ? {} : { availableFreeBytes }),
    ...(quotaSufficient === undefined ? {} : { quotaSufficient }),
  });
  return { plan, cachedByDigest };
}

/** Explicit user-action helper; inspection never requests persistence. */
export async function requestAceModelStoragePersistence(
  storage: Pick<StorageManager, "persist"> = navigator.storage,
): Promise<boolean> {
  return await storage.persist();
}

/**
 * Sequentially acquire the phase/runtime package into OPFS. Sequential order
 * deliberately bounds network, writable-stream, and hash state to one shard.
 */
export async function acquireAceModelFiles(
  options: AceAcquireModelOptions,
): Promise<AceAcquiredModelFiles> {
  const inspected = await inspectAceModelAcquisition(
    options.manifest,
    options.cache,
    options.storage,
    options.headroomBytes ?? ACE_MODEL_STORAGE_HEADROOM_BYTES,
    options.signal,
    options.onTrace,
    options.now ?? defaultNow,
    options.cacheAuthenticationOwner ?? ACE_PRODUCTION_CACHE_AUTHENTICATION_OWNER,
  );
  const plan = inspected.plan;
  if (plan.quotaSufficient === false) {
    throw new DOMException(
      `ACE model cache needs ${plan.requiredFreeBytes} free quota bytes; ` +
        `${plan.availableFreeBytes ?? 0} are available`,
      "QuotaExceededError",
    );
  }
  const acquired = new Map<string, File>();
  const availableByDigest = new Map(inspected.cachedByDigest);
  const networkDigests = new Set<string>();
  let completedBytes = 0;
  for (let index = 0; index < plan.files.length; index += 1) {
    options.signal?.throwIfAborted();
    const file = plan.files[index]!;
    let candidate = availableByDigest.get(file.sha256);
    if (candidate !== undefined) {
      completedBytes = checkedAdd(completedBytes, file.byteLength);
      acquired.set(file.name, candidate);
      options.onFileProgress?.({
        file: file.name,
        fileIndex: index,
        fileCount: plan.files.length,
        fileReceivedBytes: file.byteLength,
        fileBytes: file.byteLength,
        completedBytes,
        totalBytes: plan.runtimeBytes,
        source: networkDigests.has(file.sha256) ? "network" : "cache",
      });
      emitAcquisitionTrace(options.onTrace, Object.freeze({
        operation: "proof-reuse",
        file: file.name,
        byteLength: file.byteLength,
        sha256: file.sha256,
        source: networkDigests.has(file.sha256) ? "network" : "cache",
        exactImmutableFileIdentity: true,
        redundantHashPerformed: false,
      }));
      continue;
    }
    await fetchAceModelAssetResumable({
      manifestUrl: options.manifestUrl,
      file,
      beginTransaction: async () => await options.cache.begin(file),
      ...(options.signal === undefined ? {} : { signal: options.signal }),
      ...(options.fetch === undefined ? {} : { fetch: options.fetch }),
      ...(options.maximumAttempts === undefined
        ? {}
        : { maximumAttempts: options.maximumAttempts }),
      onProgress: (progress) => {
        options.onFileProgress?.({
          file: file.name,
          fileIndex: index,
          fileCount: plan.files.length,
          fileReceivedBytes: progress.receivedBytes,
          fileBytes: file.byteLength,
          completedBytes: completedBytes + progress.receivedBytes,
          totalBytes: plan.runtimeBytes,
          source: "network",
        });
      },
    });
    candidate = await options.cache.openCandidate(file);
    if (candidate === undefined) {
      throw new Error(`${file.name} was not visible after authenticated cache commit`);
    }
    const startedAtMs = options.onTrace === undefined
      ? 0
      : (options.now ?? defaultNow)();
    const authentication = await cachedFileMatchesRecord(
      candidate,
      file,
      options.cacheAuthenticationOwner ??
        ACE_PRODUCTION_CACHE_AUTHENTICATION_OWNER,
      options.signal,
    );
    if (options.onTrace !== undefined) {
      const completedAtMs = (options.now ?? defaultNow)();
      emitAcquisitionTrace(options.onTrace, Object.freeze({
        operation: "cache-authentication",
        file: file.name,
        byteLength: file.byteLength,
        expectedSha256: file.sha256,
        actualSha256: authentication.actualSha256,
        authenticationOwner: options.cacheAuthenticationOwner ??
          ACE_PRODUCTION_CACHE_AUTHENTICATION_OWNER,
        matched: authentication.matched,
        receivedBytes: authentication.receivedBytes,
        hashChunkCount: authentication.hashChunkCount,
        maximumHashChunkBytes: authentication.maximumHashChunkBytes,
        startedAtMs,
        completedAtMs,
        wallMs: nonnegativeElapsed(completedAtMs, startedAtMs),
        exactImmutableFileProofPublished: authentication.matched,
      }));
    }
    if (!authentication.matched) {
      await options.cache.remove(file);
      throw new Error(`${file.name} failed cache authentication after commit`);
    }
    markAceAuthenticatedGpuSource(candidate, file);
    availableByDigest.set(file.sha256, candidate);
    networkDigests.add(file.sha256);
    acquired.set(file.name, candidate);
    completedBytes = checkedAdd(completedBytes, file.byteLength);
    emitAcquisitionTrace(options.onTrace, Object.freeze({
      operation: "proof-reuse",
      file: file.name,
      byteLength: file.byteLength,
      sha256: file.sha256,
      source: "network",
      exactImmutableFileIdentity: true,
      redundantHashPerformed: false,
    }));
  }
  return Object.freeze({ plan, files: acquired });
}

interface AceCachedFileAuthentication {
  readonly matched: boolean;
  readonly receivedBytes: number;
  readonly hashChunkCount: number;
  readonly maximumHashChunkBytes: number;
  readonly actualSha256: string | null;
}

async function cachedFileMatchesRecord(
  candidate: File,
  record: AcePackageFileRecord,
  authenticationOwner: AceCacheAuthenticationOwner,
  signal?: AbortSignal,
): Promise<AceCachedFileAuthentication> {
  signal?.throwIfAborted();
  if (candidate.size !== record.byteLength) {
    return Object.freeze({
      matched: false,
      receivedBytes: 0,
      hashChunkCount: 0,
      maximumHashChunkBytes: 0,
      actualSha256: null,
    });
  }
  return authenticationOwner === "scalar-stream"
    ? await authenticateCachedFileScalar(candidate, record, signal)
    : await authenticateCachedFileWebCrypto(candidate, record, signal);
}

async function authenticateCachedFileScalar(
  candidate: File,
  record: AcePackageFileRecord,
  signal?: AbortSignal,
): Promise<AceCachedFileAuthentication> {
  const reader = candidate.stream().getReader();
  const hash = new AceIncrementalSha256();
  let receivedBytes = 0;
  let hashChunkCount = 0;
  let maximumHashChunkBytes = 0;
  try {
    while (true) {
      signal?.throwIfAborted();
      const item = await reader.read();
      if (item.done) break;
      for (
        let offset = 0;
        offset < item.value.byteLength;
        offset += ACE_MODEL_CACHE_HASH_CHUNK_BYTES
      ) {
        signal?.throwIfAborted();
        const chunk = item.value.subarray(
          offset,
          Math.min(
            offset + ACE_MODEL_CACHE_HASH_CHUNK_BYTES,
            item.value.byteLength,
          ),
        );
        receivedBytes = checkedAdd(receivedBytes, chunk.byteLength);
        hashChunkCount += 1;
        maximumHashChunkBytes = Math.max(
          maximumHashChunkBytes,
          chunk.byteLength,
        );
        if (receivedBytes > record.byteLength) {
          await reader.cancel();
          return Object.freeze({
            matched: false,
            receivedBytes,
            hashChunkCount,
            maximumHashChunkBytes,
            actualSha256: null,
          });
        }
        hash.update(chunk);
      }
    }
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
  signal?.throwIfAborted();
  const actualSha256 = hash.digestHex();
  return Object.freeze({
    matched: receivedBytes === record.byteLength && actualSha256 === record.sha256,
    receivedBytes,
    hashChunkCount,
    maximumHashChunkBytes,
    actualSha256,
  });
}

async function authenticateCachedFileWebCrypto(
  candidate: File,
  record: AcePackageFileRecord,
  signal?: AbortSignal,
): Promise<AceCachedFileAuthentication> {
  if (candidate.size >= ACE_MODEL_CACHE_WEBCRYPTO_MAX_FILE_BYTES) {
    throw new RangeError(
      "ACE cached-file WebCrypto authentication requires a file below 128 MiB",
    );
  }
  const subtle = globalThis.crypto?.subtle;
  if (subtle === undefined) {
    throw new Error("ACE cached-file authentication requires WebCrypto SHA-256");
  }
  let payload: ArrayBuffer | undefined;
  try {
    payload = await candidate.arrayBuffer();
    signal?.throwIfAborted();
    const receivedBytes = payload.byteLength;
    if (receivedBytes !== record.byteLength) {
      return Object.freeze({
        matched: false,
        receivedBytes,
        hashChunkCount: 1,
        maximumHashChunkBytes: receivedBytes,
        actualSha256: null,
      });
    }
    const digest = await subtle.digest("SHA-256", payload);
    signal?.throwIfAborted();
    const actualSha256 = hexadecimal(new Uint8Array(digest));
    return Object.freeze({
      matched: actualSha256 === record.sha256,
      receivedBytes,
      hashChunkCount: 1,
      maximumHashChunkBytes: receivedBytes,
      actualSha256,
    });
  } finally {
    payload = undefined;
  }
}

/** Convenience for the ordinary OPFS backend. */
export async function acquireAceModelFilesFromOpfs(
  options: Omit<AceAcquireModelOptions, "cache" | "storage"> & {
    readonly cache?: AceOpfsModelCache;
    readonly storage?: Pick<StorageManager, "estimate" | "getDirectory">;
  },
): Promise<AceAcquiredModelFiles> {
  const cache = options.cache ?? (await AceOpfsModelCache.open(options.storage));
  return await acquireAceModelFiles({ ...options, cache });
}

function checkedAdd(left: number, right: number): number {
  const value = left + right;
  if (!Number.isSafeInteger(value)) {
    throw new RangeError("ACE package byte accounting exceeds safe integers");
  }
  return value;
}

function defaultNow(): number {
  return globalThis.performance?.now() ?? Date.now();
}

function hexadecimal(bytes: Uint8Array): string {
  let result = "";
  for (const byte of bytes) result += byte.toString(16).padStart(2, "0");
  return result;
}

function emitAcquisitionTrace(
  sink: ((trace: AceModelAcquisitionTrace) => void) | undefined,
  trace: AceModelAcquisitionTrace,
): void {
  try {
    sink?.(trace);
  } catch {
    // Capture is observational and cannot alter acquisition or proof publication.
  }
}

function nonnegativeElapsed(completedAtMs: number, startedAtMs: number): number {
  return Math.max(0, completedAtMs - startedAtMs);
}
