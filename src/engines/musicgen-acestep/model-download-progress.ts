/** Physical bytes retained by the cold browser cache before the deferred VAE. */
export const INITIALIZATION_CACHE_PHYSICAL_BYTES = 5_578_938_623;

/** Logical byte total reported by the runtime's initialization weights stage. */
export const INITIALIZATION_WEIGHTS_LOGICAL_BYTES = 5_578_938_623;

/** Physical bytes retained when the VAE package is fetched on first generation. */
export const DEFERRED_VAE_CACHE_PHYSICAL_BYTES = 168_791_552;

/** Complete physical footprint of a cold production-model cache. */
export const MODEL_DOWNLOAD_TOTAL_BYTES = 5_747_730_175;

/** Complete physical footprint when the packed INT8 DiT is selected. */
export const INT8_MODEL_DOWNLOAD_TOTAL_BYTES = 4_426_524_415;

/** Unique content-addressed payloads in the complete production-model cache. */
export const MODEL_DOWNLOAD_ASSET_COUNT = 113;

export interface ModelCacheSummary {
  readonly supported: boolean;
  readonly assetCount: number;
  readonly sizeBytes: number;
  readonly partialAssetCount: number;
}

export interface ModelDownloadProgress {
  readonly fraction: number;
  readonly percentage: number;
  readonly completed: number;
  readonly total: number;
}

export const INITIAL_MODEL_DOWNLOAD_PROGRESS: ModelDownloadProgress = createProgress(0);

export function initialModelDownloadProgress(total = MODEL_DOWNLOAD_TOTAL_BYTES): ModelDownloadProgress {
  return createProgress(0, total);
}

/** Show the first-download note until the complete pinned cache is present. */
export function shouldShowModelDownloadNote(cache: ModelCacheSummary | undefined): boolean {
  return cache?.supported === true && !isModelDownloadComplete(cache);
}

export function isModelDownloadComplete(cache: ModelCacheSummary | undefined): boolean {
  return (
    cache?.supported === true &&
    cache.assetCount === MODEL_DOWNLOAD_ASSET_COUNT &&
    cache.sizeBytes === MODEL_DOWNLOAD_TOTAL_BYTES &&
    cache.partialAssetCount === 0
  );
}

/**
 * Fold one runtime progress event into the displayed physical download total.
 *
 * The input may be either a progress payload or a worker message containing a
 * `progress` payload. Unknown, malformed, and regressing events are ignored.
 */
export function updateModelDownloadProgress(current: ModelDownloadProgress, event: unknown): ModelDownloadProgress {
  const candidate = physicalCompletedBytes(event, current.total);
  if (candidate === undefined || candidate <= current.completed) return current;
  return createProgress(candidate, current.total);
}

/** Format bytes with decimal units (1 GB = 1,000,000,000 bytes). */
export function formatDecimalBytes(value: number): string {
  const bytes = safeNonnegativeInteger(value);
  if (bytes >= 1_000_000_000) {
    return `${compactDecimal(bytes / 1_000_000_000, 2)} GB`;
  }
  return `${compactDecimal(bytes / 1_000_000, 1)} MB`;
}

export function formatModelDownloadAmount(progress: Pick<ModelDownloadProgress, "completed" | "total">): string {
  return `${formatDecimalBytes(progress.completed)} / ${formatDecimalBytes(progress.total)}`;
}

function physicalCompletedBytes(event: unknown, totalBytes: number): number | undefined {
  const outer = record(event);
  if (outer === undefined) return undefined;
  const progress = record(outer.progress) ?? outer;

  if (progress.stage === "weights") {
    return initializationPhysicalBytes(progress);
  }
  if (progress.stage === "vae-load") {
    return deferredVaePhysicalBytes(progress.message, totalBytes);
  }
  return undefined;
}

function initializationPhysicalBytes(progress: Readonly<Record<string, unknown>>): number | undefined {
  if (
    progress.unit !== "bytes" ||
    (progress.totalUnits !== INITIALIZATION_WEIGHTS_LOGICAL_BYTES &&
      progress.totalUnits !== INT8_MODEL_DOWNLOAD_TOTAL_BYTES - DEFERRED_VAE_CACHE_PHYSICAL_BYTES)
  ) {
    return undefined;
  }

  const logicalCompleted = clampedByteCount(progress.completedUnits, progress.totalUnits);
  if (logicalCompleted === undefined) return undefined;

  return logicalCompleted;
}

function deferredVaePhysicalBytes(message: unknown, totalBytes: number): number | undefined {
  if (typeof message !== "string") return undefined;
  const match = /(?:^|\s)([0-9]+)\/([0-9]+) bytes$/u.exec(message.trim());
  if (match === null) return undefined;

  const completed = Number(match[1]);
  const total = Number(match[2]);
  if (total !== DEFERRED_VAE_CACHE_PHYSICAL_BYTES) return undefined;
  const vaeCompleted = clampedByteCount(completed, DEFERRED_VAE_CACHE_PHYSICAL_BYTES);
  if (vaeCompleted === undefined) return undefined;
  return totalBytes - DEFERRED_VAE_CACHE_PHYSICAL_BYTES + vaeCompleted;
}

function createProgress(completed: number, total = MODEL_DOWNLOAD_TOTAL_BYTES): ModelDownloadProgress {
  const safeCompleted = Math.min(total, safeNonnegativeInteger(completed));
  const fraction = safeCompleted / total;
  return Object.freeze({
    fraction,
    percentage: fraction * 100,
    completed: safeCompleted,
    total,
  });
}

function clampedByteCount(value: unknown, total: number): number | undefined {
  if (typeof value !== "number" || !Number.isSafeInteger(value) || !Number.isFinite(value)) {
    return undefined;
  }
  return Math.min(total, Math.max(0, value));
}

function safeNonnegativeInteger(value: number): number {
  if (!Number.isFinite(value) || value <= 0) return 0;
  return Math.min(Number.MAX_SAFE_INTEGER, Math.floor(value));
}

function compactDecimal(value: number, digits: number): string {
  return Number(value.toFixed(digits)).toString();
}

function record(value: unknown): Readonly<Record<string, unknown>> | undefined {
  return typeof value === "object" && value !== null && !Array.isArray(value) ? (value as Readonly<Record<string, unknown>>) : undefined;
}
