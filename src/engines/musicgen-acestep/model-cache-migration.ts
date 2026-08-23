import { deleteAceModelCache } from "ace-step-1.5.wgsl";

/**
 * Bump this identity whenever the demo's selected model payload inventory
 * changes. The library cache remains content-addressed, so the demo owns this
 * one-time migration boundary independently of the library's cache format.
 */
export const ACE_DEMO_MODEL_CACHE_GENERATION = "direct-b44a3d157009d035-d3fc0020efcf6070-36a54d79777d6826-v1";

/**
 * Exported so runtimes can hold this lock in "shared" mode for the lifetime of
 * an active worker: the migration takes it exclusively, which would otherwise
 * delete the cache out from under another tab that is downloading/generating.
 */
export const ACE_MODEL_CACHE_LIFECYCLE_LOCK = "ace-step-1.5.wgsl-demo-model-cache-migration";
const MIGRATION_LOCK_NAME = ACE_MODEL_CACHE_LIFECYCLE_LOCK;
const METADATA_DIRECTORY = "ace-step-1.5.wgsl-demo-metadata-v1";
const GENERATION_MARKER_FILE = "model-cache-generation.txt";
const MAX_GENERATION_MARKER_BYTES = 256;

type ModelCacheStorage = Pick<StorageManager, "getDirectory">;

export interface AceDemoExclusiveLockManager {
  request<Result>(name: string, options: Readonly<{ mode: "exclusive" }>, operation: () => Promise<Result>): Promise<Result>;
}

export interface AceDemoModelCacheMigrationOptions {
  /** @internal Test seam. Production defaults to `navigator.storage`. */
  readonly storage?: ModelCacheStorage | undefined;
  /** @internal Test seam. Production defaults to `navigator.locks`. */
  readonly locks?: AceDemoExclusiveLockManager | undefined;
  /** @internal Test seam around the package-owned model-cache deletion API. */
  readonly deleteModelCache?: (storage: ModelCacheStorage) => Promise<boolean>;
}

export interface AceDemoModelCacheMigrationResult {
  readonly supported: boolean;
  readonly migrated: boolean;
  readonly previousGeneration: string | null;
}

/**
 * Ensure that OPFS contains model payloads from only the demo's current model
 * generation.
 *
 * The generation marker deliberately lives outside both the package-owned
 * model-cache directory and the separate audio-output directory. A stale or
 * absent marker deletes the complete model cache (including partial payloads)
 * before publishing the new marker. The marker is never advanced after a
 * failed deletion or failed write, so the next visit retries safely.
 */
export async function ensureCurrentAceDemoModelCache(options: AceDemoModelCacheMigrationOptions = {}): Promise<AceDemoModelCacheMigrationResult> {
  const storage = Object.hasOwn(options, "storage") ? options.storage : browserStorage();
  if (storage === undefined || typeof storage.getDirectory !== "function") {
    return Object.freeze({
      supported: false,
      migrated: false,
      previousGeneration: null,
    });
  }

  const locks = Object.hasOwn(options, "locks") ? options.locks : browserLocks();
  if (locks === undefined || typeof locks.request !== "function") {
    throw new Error("Web Locks are required to migrate the ACE demo model cache safely");
  }
  const deleteModelCache = options.deleteModelCache ?? deleteAceModelCache;

  return locks.request(MIGRATION_LOCK_NAME, { mode: "exclusive" }, async () => {
    // Read inside the exclusive lock. Another current-version tab may have
    // completed the migration while this tab was waiting for ownership.
    const root = await storage.getDirectory();
    const metadata = await root.getDirectoryHandle(METADATA_DIRECTORY, {
      create: true,
    });
    const previousGeneration = await readGenerationMarker(metadata);
    if (previousGeneration === ACE_DEMO_MODEL_CACHE_GENERATION) {
      return Object.freeze({
        supported: true,
        migrated: false,
        previousGeneration,
      });
    }

    // This public package API removes only the package-owned model cache.
    // In particular, it does not touch the separate OPFS audio directory.
    await deleteModelCache(storage);
    await writeGenerationMarker(metadata, ACE_DEMO_MODEL_CACHE_GENERATION);
    return Object.freeze({
      supported: true,
      migrated: true,
      previousGeneration,
    });
  });
}

async function readGenerationMarker(metadata: FileSystemDirectoryHandle): Promise<string | null> {
  try {
    const handle = await metadata.getFileHandle(GENERATION_MARKER_FILE);
    const file = await handle.getFile();
    if (file.size > MAX_GENERATION_MARKER_BYTES) return null;
    return await file.text();
  } catch (error) {
    if (isNotFound(error)) return null;
    throw error;
  }
}

async function writeGenerationMarker(metadata: FileSystemDirectoryHandle, generation: string): Promise<void> {
  const handle = await metadata.getFileHandle(GENERATION_MARKER_FILE, {
    create: true,
  });
  const writable = await handle.createWritable();
  try {
    await writable.write(generation);
    await writable.close();
  } catch (error) {
    try {
      await writable.abort(error);
    } catch {
      // The original marker publication failure remains authoritative.
    }
    throw error;
  }
}

function browserStorage(): ModelCacheStorage | undefined {
  return globalThis.navigator?.storage;
}

function browserLocks(): AceDemoExclusiveLockManager | undefined {
  return globalThis.navigator?.locks as AceDemoExclusiveLockManager | undefined;
}

function isNotFound(error: unknown): boolean {
  return error instanceof DOMException && error.name === "NotFoundError";
}
