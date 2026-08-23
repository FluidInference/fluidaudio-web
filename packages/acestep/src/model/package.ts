import {
  ACE_PACKAGE_FORMAT,
  type AcePackageFileRecord,
  type AcePackageManifest,
  type AcePackageProfile,
  parseAcePackageManifest,
} from "./manifest.js";
import { AceIncrementalSha256 } from "./sha256.js";
import { parseStrictJson } from "./strict-json.js";

export const ACE_MAX_MANIFEST_BYTES = 2 * 1024 * 1024;
export const ACE_MODEL_TRANSPORT_CHUNK_BYTES = 4 * 1024 * 1024;
export const ACE_REFERENCE_MANIFEST_SHA256 =
  "18f36c6420976475af65ecd833ca56c6119706322ce54120389d4915d8e80db6" as const;
export const ACE_OPT_0054_VAE_REVISION7_MANIFEST_SHA256 =
  "36a54d79777d6826088095ba6ebc028fb4bea546368c0f0a29cd0eee8d656da7" as const;
export const ACE_OPT_0054_VAE_REVISION7_MANIFEST_BYTES = 716_185 as const;

export interface AceManifestLoadProgress {
  readonly receivedBytes: number;
  readonly totalBytes?: number;
}

export interface AceLoadedPackageManifest {
  readonly manifest: AcePackageManifest;
  readonly manifestUrl: string;
  readonly manifestSha256: string;
  readonly manifestByteLength: number;
  readonly manifestId: string;
}

export interface AceLoadPackageManifestOptions {
  readonly manifestUrl: string;
  /** Trust root supplied outside the fetched manifest. */
  readonly expectedManifestSha256: string;
  readonly expectedProfile: AcePackageProfile;
  /** Exact revision-7 VAE trust root; rejected unless paired with its SHA. */
  readonly authenticatedVaeConverterRevision?: 7;
  /** Exact OPT-0009 oracle only; rejected unless paired with its pinned SHA. */
  readonly authenticatedDitDenseConverterRevision?: 7;
  readonly signal?: AbortSignal;
  readonly fetch?: typeof fetch;
  readonly onProgress?: (progress: AceManifestLoadProgress) => void;
}

export class AceModelTransportError extends Error {
  readonly code: string;

  constructor(code: string, message: string) {
    super(message);
    this.name = "AceModelTransportError";
    this.code = code;
  }
}

/**
 * Fetch and authenticate the exact manifest bytes before decoding or parsing
 * any JSON. The manifest's own fields can therefore never select their trust
 * root.
 */
export async function loadAcePackageManifest(
  options: AceLoadPackageManifestOptions,
): Promise<AceLoadedPackageManifest> {
  const expectedDigest = requireSha256(
    options.expectedManifestSha256,
    "expected manifest SHA-256",
  );
  if (
    options.authenticatedVaeConverterRevision !== undefined &&
    (options.authenticatedVaeConverterRevision !== 7 ||
      options.expectedProfile !== "fp16-vae-experimental" ||
      expectedDigest !== ACE_OPT_0054_VAE_REVISION7_MANIFEST_SHA256)
  ) {
    throw new AceModelTransportError(
      "MANIFEST_IDENTITY_ERROR",
      "Revision-7 VAE parsing requires the exact authenticated revision-7 manifest",
    );
  }
  if (
    options.authenticatedDitDenseConverterRevision !== undefined &&
    (options.authenticatedDitDenseConverterRevision !== 7 ||
      options.expectedProfile !== "fp16-dit-dense-experimental" ||
      expectedDigest !==
        "d3fc0020efcf60702db411da2fd4b93e9bb84f1437ed310aef01c892727e452f")
  ) {
    throw new AceModelTransportError(
      "MANIFEST_IDENTITY_ERROR",
      "Legacy mixed-DiT parsing requires the exact authenticated OPT-0009 manifest",
    );
  }
  const url = absoluteUrl(options.manifestUrl);
  options.signal?.throwIfAborted();
  const fetchAsset = options.fetch ?? globalThis.fetch.bind(globalThis);
  const response = await fetchAsset(url, {
    method: "GET",
    cache: "no-store",
    ...(options.signal === undefined ? {} : { signal: options.signal }),
  });
  if (response.status !== 200) {
    try {
      await response.body?.cancel();
    } catch {
      // The HTTP status remains the authoritative manifest failure.
    }
    throw new AceModelTransportError(
      "MANIFEST_HTTP_ERROR",
      `ACE manifest request failed (${response.status} ${response.statusText})`,
    );
  }
  if (response.body === null) {
    throw new AceModelTransportError(
      "MANIFEST_HTTP_ERROR",
      `ACE manifest request failed (${response.status} ${response.statusText})`,
    );
  }
  const declaredLength = contentLength(response.headers);
  if (declaredLength !== undefined && declaredLength > ACE_MAX_MANIFEST_BYTES) {
    await response.body.cancel();
    throw new AceModelTransportError(
      "MANIFEST_TOO_LARGE",
      `ACE manifest declares ${declaredLength} bytes; limit is ${ACE_MAX_MANIFEST_BYTES}`,
    );
  }

  const reader = response.body.getReader();
  const hash = new AceIncrementalSha256();
  const chunks: Uint8Array[] = [];
  let receivedBytes = 0;
  try {
    while (true) {
      options.signal?.throwIfAborted();
      const item = await reader.read();
      if (item.done) break;
      for (const chunk of boundedSlices(item.value, ACE_MODEL_TRANSPORT_CHUNK_BYTES)) {
        receivedBytes += chunk.byteLength;
        if (receivedBytes > ACE_MAX_MANIFEST_BYTES) {
          throw new AceModelTransportError(
            "MANIFEST_TOO_LARGE",
            `ACE manifest exceeded the ${ACE_MAX_MANIFEST_BYTES}-byte limit`,
          );
        }
        hash.update(chunk);
        chunks.push(chunk.slice());
        options.onProgress?.({
          receivedBytes,
          ...(declaredLength === undefined ? {} : { totalBytes: declaredLength }),
        });
      }
    }
  } catch (error) {
    try {
      await reader.cancel(error);
    } catch {
      // Preserve the primary transport or cancellation error.
    }
    throw error;
  } finally {
    reader.releaseLock();
  }
  options.signal?.throwIfAborted();
  if (declaredLength !== undefined && receivedBytes !== declaredLength) {
    throw new AceModelTransportError(
      "MANIFEST_LENGTH_MISMATCH",
      `ACE manifest received ${receivedBytes} bytes; expected ${declaredLength}`,
    );
  }

  // Deliberately finalize and compare before allocating decoded text or calling
  // either JSON parser.
  const actualDigest = hash.digestHex();
  if (actualDigest !== expectedDigest) {
    throw new AceModelTransportError(
      "MANIFEST_SHA256_MISMATCH",
      `ACE manifest SHA-256 changed: ${actualDigest}`,
    );
  }
  const bytes = joinChunks(chunks, receivedBytes);
  if (hasUtf8Bom(bytes)) {
    throw new AceModelTransportError(
      "MANIFEST_ENCODING_ERROR",
      "ACE manifest must be canonical UTF-8 without a byte-order mark",
    );
  }
  let text: string;
  try {
    text = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  } catch (error) {
    throw new AceModelTransportError(
      "MANIFEST_ENCODING_ERROR",
      `ACE manifest is not valid UTF-8: ${formatUnknownError(error)}`,
    );
  }
  let decoded: unknown;
  try {
    decoded = parseStrictJson(text);
  } catch (error) {
    throw new AceModelTransportError(
      "MANIFEST_JSON_ERROR",
      `ACE manifest is not strict JSON: ${formatUnknownError(error)}`,
    );
  }
  const manifest = parseAcePackageManifest(
    decoded,
    options.expectedProfile,
    {
      ...(options.authenticatedVaeConverterRevision === 7
        ? { authenticatedVaeConverterRevision: 7 as const }
        : {}),
      ...(options.authenticatedDitDenseConverterRevision === 7
        ? { authenticatedDitDenseConverterRevision: 7 as const }
        : {}),
    },
  );
  return {
    manifest,
    manifestUrl: url,
    manifestSha256: actualDigest,
    manifestByteLength: receivedBytes,
    manifestId: `${ACE_PACKAGE_FORMAT}:${manifest.profile}:${actualDigest}`,
  };
}

export interface AceModelAssetTransaction {
  /** Write tentative bytes at an exact monotonically increasing offset. */
  write(offset: number, bytes: Uint8Array): void | Promise<void>;
  /** Make the complete, authenticated identity visible. */
  commit(): unknown | Promise<unknown>;
  /** Discard or retain only non-visible partial state. Must be idempotent. */
  rollback(reason: unknown): void | Promise<void>;
}

export interface AceResumableModelAssetTransaction extends AceModelAssetTransaction {
  /** Exact tentative prefix already present in the destination. */
  readonly resumeOffset: number;
  /** Authenticate the retained prefix before appending any network bytes. */
  hashExistingPrefix(signal?: AbortSignal): Promise<void>;
  /** Discard tentative prefix bytes while retaining this open transaction. */
  restart(): Promise<void>;
}

export interface AceModelAssetProgress {
  readonly file: string;
  readonly receivedBytes: number;
  readonly totalBytes: number;
}

export interface AceFetchModelAssetOptions {
  readonly manifestUrl: string;
  readonly file: AcePackageFileRecord;
  readonly transaction: AceModelAssetTransaction;
  readonly signal?: AbortSignal;
  readonly fetch?: typeof fetch;
  readonly onProgress?: (progress: AceModelAssetProgress) => void;
}

export interface AceResumableModelAssetProgress extends AceModelAssetProgress {
  readonly attempt: number;
  readonly resumedBytes: number;
}

export interface AceFetchResumableModelAssetOptions {
  readonly manifestUrl: string;
  readonly file: AcePackageFileRecord;
  /** Open a fresh transaction for each attempt; failed attempts are rolled back first. */
  readonly beginTransaction: () => Promise<AceResumableModelAssetTransaction>;
  readonly signal?: AbortSignal;
  readonly fetch?: typeof fetch;
  readonly maximumAttempts?: number;
  readonly onProgress?: (progress: AceResumableModelAssetProgress) => void;
  /** Testable backoff hook. `attempt` is the completed one-based attempt. */
  readonly waitBeforeRetry?: (attempt: number, error: unknown) => Promise<void>;
}

/**
 * Stream one manifest-pinned asset through bounded writes. The destination is
 * transactional: it is committed only after exact length and SHA-256 checks.
 */
export async function fetchAceModelAsset(
  options: AceFetchModelAssetOptions,
): Promise<void> {
  const url = new URL(options.file.name, absoluteUrl(options.manifestUrl)).href;
  const expectedDigest = requireSha256(options.file.sha256, "file SHA-256");
  const fetchAsset = options.fetch ?? globalThis.fetch.bind(globalThis);
  let reader: ReadableStreamDefaultReader<Uint8Array> | undefined;
  try {
    options.signal?.throwIfAborted();
    const response = await fetchAsset(url, {
      method: "GET",
      cache: "no-store",
      ...(options.signal === undefined ? {} : { signal: options.signal }),
    });
    if (response.status !== 200 || response.body === null) {
      throw new AceModelTransportError(
        "MODEL_ASSET_HTTP_ERROR",
        `${options.file.name} request failed (${response.status} ${response.statusText})`,
      );
    }
    const declaredLength = contentLength(response.headers);
    if (declaredLength !== undefined && declaredLength !== options.file.byteLength) {
      await response.body.cancel();
      throw new AceModelTransportError(
        "MODEL_ASSET_LENGTH_MISMATCH",
        `${options.file.name} Content-Length does not match its manifest record`,
      );
    }

    reader = response.body.getReader();
    const hash = new AceIncrementalSha256();
    let receivedBytes = 0;
    while (true) {
      options.signal?.throwIfAborted();
      const item = await reader.read();
      if (item.done) break;
      for (const chunk of boundedSlices(item.value, ACE_MODEL_TRANSPORT_CHUNK_BYTES)) {
        options.signal?.throwIfAborted();
        const nextBytes = receivedBytes + chunk.byteLength;
        if (nextBytes > options.file.byteLength) {
          throw new AceModelTransportError(
            "MODEL_ASSET_TOO_LONG",
            `${options.file.name} exceeded its manifest byte length`,
          );
        }
        hash.update(chunk);
        await options.transaction.write(receivedBytes, chunk);
        receivedBytes = nextBytes;
        options.onProgress?.({
          file: options.file.name,
          receivedBytes,
          totalBytes: options.file.byteLength,
        });
      }
    }
    options.signal?.throwIfAborted();
    if (receivedBytes !== options.file.byteLength) {
      throw new AceModelTransportError(
        "MODEL_ASSET_TRUNCATED",
        `${options.file.name} received ${receivedBytes} bytes; expected ${options.file.byteLength}`,
      );
    }
    const actualDigest = hash.digestHex();
    if (actualDigest !== expectedDigest) {
      throw new AceModelTransportError(
        "MODEL_ASSET_SHA256_MISMATCH",
        `${options.file.name} SHA-256 changed: ${actualDigest}`,
      );
    }
    await options.transaction.commit();
  } catch (error) {
    if (reader !== undefined) {
      try {
        await reader.cancel(error);
      } catch {
        // Preserve the primary failure.
      }
    }
    try {
      await options.transaction.rollback(error);
    } catch {
      // A rollback implementation must report its own health separately; the
      // integrity or cancellation failure remains the operation result.
    }
    throw error;
  } finally {
    reader?.releaseLock();
  }
}

/**
 * Resume one manifest-pinned asset into transactional storage.
 *
 * A retained prefix is hashed before the Range request. A 206 response must
 * describe exactly the requested suffix; a server may ignore Range only by
 * returning a complete, length-qualified 200 response, in which case the
 * tentative destination is truncated before any response byte is written.
 * Only transient HTTP failures, truncated bodies, and browser fetch/network
 * TypeErrors are retried. Integrity, protocol, quota, and cancellation errors
 * fail immediately.
 */
export async function fetchAceModelAssetResumable(
  options: AceFetchResumableModelAssetOptions,
): Promise<void> {
  const maximumAttempts = options.maximumAttempts ?? 3;
  if (
    !Number.isSafeInteger(maximumAttempts) ||
    maximumAttempts < 1 ||
    maximumAttempts > 10
  ) {
    throw new RangeError("ACE model maximumAttempts must be an integer from 1 through 10");
  }
  const waitBeforeRetry = options.waitBeforeRetry ?? defaultRetryWait;
  for (let attempt = 1; attempt <= maximumAttempts; attempt += 1) {
    options.signal?.throwIfAborted();
    const transaction = await options.beginTransaction();
    try {
      await fetchAceModelAssetResumeAttempt(options, transaction, attempt);
      return;
    } catch (error) {
      if (
        attempt === maximumAttempts ||
        options.signal?.aborted === true ||
        !isRetryableAssetFailure(error)
      ) {
        throw error;
      }
      await waitBeforeRetry(attempt, error);
    }
  }
  throw new Error("Unreachable ACE model retry state");
}

async function fetchAceModelAssetResumeAttempt(
  options: AceFetchResumableModelAssetOptions,
  transaction: AceResumableModelAssetTransaction,
  attempt: number,
): Promise<void> {
  const url = new URL(options.file.name, absoluteUrl(options.manifestUrl)).href;
  requireSha256(options.file.sha256, "file SHA-256");
  const initialOffset = transaction.resumeOffset;
  if (
    !Number.isSafeInteger(initialOffset) ||
    initialOffset < 0 ||
    initialOffset > options.file.byteLength
  ) {
    await rollbackQuietly(transaction, new RangeError("Invalid ACE resume offset"));
    throw new RangeError("ACE model resume offset is outside the manifest payload");
  }
  let reader: ReadableStreamDefaultReader<Uint8Array> | undefined;
  let responseBody: ReadableStream<Uint8Array> | undefined;
  try {
    options.signal?.throwIfAborted();
    if (initialOffset > 0) {
      await transaction.hashExistingPrefix(options.signal);
    }
    if (initialOffset === options.file.byteLength) {
      await transaction.commit();
      options.onProgress?.({
        file: options.file.name,
        receivedBytes: options.file.byteLength,
        totalBytes: options.file.byteLength,
        attempt,
        resumedBytes: initialOffset,
      });
      return;
    }

    const fetchAsset = options.fetch ?? globalThis.fetch.bind(globalThis);
    let response: Response;
    try {
      response = await fetchAsset(url, {
        method: "GET",
        cache: "no-store",
        ...(initialOffset === 0
          ? {}
          : { headers: { Range: `bytes=${initialOffset}-` } }),
        ...(options.signal === undefined ? {} : { signal: options.signal }),
      });
    } catch (error) {
      throw networkTransportError(error, options.file.name);
    }
    if (isTransientHttpStatus(response.status)) {
      await response.body?.cancel();
      throw new AceModelTransportError(
        "MODEL_ASSET_RETRYABLE_HTTP",
        `${options.file.name} request failed transiently (${response.status} ${response.statusText})`,
      );
    }
    if (response.body === null) {
      throw new AceModelTransportError(
        "MODEL_ASSET_HTTP_ERROR",
        `${options.file.name} response has no body`,
      );
    }
    responseBody = response.body;

    let cursor = initialOffset;
    if (initialOffset === 0) {
      if (response.status !== 200) {
        await response.body.cancel();
        throw new AceModelTransportError(
          "MODEL_ASSET_HTTP_ERROR",
          `${options.file.name} request failed (${response.status} ${response.statusText})`,
        );
      }
      requireResponseLength(response.headers, options.file.byteLength, options.file.name);
    } else if (response.status === 206) {
      validateContentRange(
        response.headers,
        initialOffset,
        options.file.byteLength,
        options.file.name,
      );
    } else if (response.status === 200) {
      requireResponseLength(
        response.headers,
        options.file.byteLength,
        options.file.name,
        true,
      );
      await transaction.restart();
      cursor = 0;
    } else {
      await response.body.cancel();
      throw new AceModelTransportError(
        "MODEL_ASSET_HTTP_ERROR",
        `${options.file.name} resume request failed (${response.status} ${response.statusText})`,
      );
    }

    reader = response.body.getReader();
    while (true) {
      options.signal?.throwIfAborted();
      let item: ReadableStreamReadResult<Uint8Array>;
      try {
        item = await reader.read();
      } catch (error) {
        throw networkTransportError(error, options.file.name);
      }
      if (item.done) break;
      for (const chunk of boundedSlices(item.value, ACE_MODEL_TRANSPORT_CHUNK_BYTES)) {
        options.signal?.throwIfAborted();
        const next = cursor + chunk.byteLength;
        if (!Number.isSafeInteger(next) || next > options.file.byteLength) {
          throw new AceModelTransportError(
            "MODEL_ASSET_TOO_LONG",
            `${options.file.name} exceeded its manifest byte length`,
          );
        }
        await transaction.write(cursor, chunk);
        cursor = next;
        options.onProgress?.({
          file: options.file.name,
          receivedBytes: cursor,
          totalBytes: options.file.byteLength,
          attempt,
          resumedBytes: initialOffset,
        });
      }
    }
    options.signal?.throwIfAborted();
    if (cursor !== options.file.byteLength) {
      throw new AceModelTransportError(
        "MODEL_ASSET_TRUNCATED",
        `${options.file.name} received through byte ${cursor}; expected ${options.file.byteLength}`,
      );
    }
    await transaction.commit();
  } catch (error) {
    if (reader !== undefined) {
      try {
        await reader.cancel(error);
      } catch {
        // Preserve the primary transport or cancellation error.
      }
    } else if (responseBody !== undefined) {
      try {
        await responseBody.cancel(error);
      } catch {
        // Preserve the primary transport or cancellation error.
      }
    }
    await rollbackQuietly(transaction, error);
    throw error;
  } finally {
    reader?.releaseLock();
  }
}

function validateContentRange(
  headers: Headers,
  expectedStart: number,
  expectedTotal: number,
  label: string,
): void {
  const raw = headers.get("content-range");
  const match = raw?.match(/^bytes ([0-9]+)-([0-9]+)\/([0-9]+)$/);
  if (match === undefined || match === null) {
    throw new AceModelTransportError(
      "INVALID_CONTENT_RANGE",
      `${label} resume response lacks an exact Content-Range`,
    );
  }
  const start = strictHeaderInteger(match[1]!, "Content-Range start");
  const end = strictHeaderInteger(match[2]!, "Content-Range end");
  const total = strictHeaderInteger(match[3]!, "Content-Range total");
  if (start !== expectedStart || end !== expectedTotal - 1 || total !== expectedTotal) {
    throw new AceModelTransportError(
      "INVALID_CONTENT_RANGE",
      `${label} Content-Range ${JSON.stringify(raw)} does not match bytes ${expectedStart}-${expectedTotal - 1}/${expectedTotal}`,
    );
  }
  const declaredLength = contentLength(headers);
  const suffixBytes = expectedTotal - expectedStart;
  if (declaredLength !== undefined && declaredLength !== suffixBytes) {
    throw new AceModelTransportError(
      "MODEL_ASSET_LENGTH_MISMATCH",
      `${label} resumed Content-Length does not match Content-Range`,
    );
  }
}

function requireResponseLength(
  headers: Headers,
  expected: number,
  label: string,
  requirePresent = false,
): void {
  const declaredLength = contentLength(headers);
  if (requirePresent && declaredLength === undefined) {
    throw new AceModelTransportError(
      "MODEL_ASSET_LENGTH_MISMATCH",
      `${label} ignored Range without a complete Content-Length`,
    );
  }
  if (declaredLength !== undefined && declaredLength !== expected) {
    throw new AceModelTransportError(
      "MODEL_ASSET_LENGTH_MISMATCH",
      `${label} Content-Length does not match its manifest record`,
    );
  }
}

function strictHeaderInteger(raw: string, label: string): number {
  const value = Number(raw);
  if (!Number.isSafeInteger(value) || value < 0 || String(value) !== raw) {
    throw new AceModelTransportError("INVALID_CONTENT_RANGE", `${label} is not canonical`);
  }
  return value;
}

function isTransientHttpStatus(status: number): boolean {
  return status === 408 || status === 425 || status === 429 || status === 500 ||
    status === 502 || status === 503 || status === 504;
}

function isRetryableAssetFailure(error: unknown): boolean {
  return (
    error instanceof AceModelTransportError &&
      (error.code === "MODEL_ASSET_RETRYABLE_HTTP" ||
        error.code === "MODEL_ASSET_TRUNCATED" ||
        error.code === "MODEL_ASSET_NETWORK_ERROR")
  );
}

function networkTransportError(error: unknown, label: string): unknown {
  if (error instanceof DOMException && error.name === "AbortError") return error;
  if (!(error instanceof TypeError)) return error;
  return new AceModelTransportError(
    "MODEL_ASSET_NETWORK_ERROR",
    `${label} network stream failed: ${formatUnknownError(error)}`,
  );
}

async function rollbackQuietly(
  transaction: AceModelAssetTransaction,
  reason: unknown,
): Promise<void> {
  try {
    await transaction.rollback(reason);
  } catch {
    // The original failure remains authoritative; cache health is reported by
    // the next open attempt rather than replacing an integrity error.
  }
}

async function defaultRetryWait(attempt: number): Promise<void> {
  const delayMs = Math.min(250 * 2 ** (attempt - 1), 2_000);
  await new Promise<void>((resolve) => setTimeout(resolve, delayMs));
}

function contentLength(headers: Headers): number | undefined {
  const raw = headers.get("content-length");
  if (raw === null) return undefined;
  const value = Number(raw);
  if (!Number.isSafeInteger(value) || value < 0 || String(value) !== raw.trim()) {
    throw new AceModelTransportError(
      "INVALID_CONTENT_LENGTH",
      `Invalid Content-Length header ${JSON.stringify(raw)}`,
    );
  }
  return value;
}

function* boundedSlices(
  bytes: Uint8Array,
  maximumBytes: number,
): Generator<Uint8Array> {
  for (let offset = 0; offset < bytes.byteLength; offset += maximumBytes) {
    yield bytes.subarray(offset, Math.min(offset + maximumBytes, bytes.byteLength));
  }
}

function joinChunks(chunks: readonly Uint8Array[], totalBytes: number): Uint8Array {
  const result = new Uint8Array(totalBytes);
  let offset = 0;
  for (const chunk of chunks) {
    result.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return result;
}

function hasUtf8Bom(bytes: Uint8Array): boolean {
  return bytes[0] === 0xef && bytes[1] === 0xbb && bytes[2] === 0xbf;
}

function requireSha256(value: string, label: string): string {
  if (!/^[0-9a-f]{64}$/.test(value)) {
    throw new TypeError(`${label} must be a lowercase 64-character hexadecimal digest`);
  }
  return value;
}

function absoluteUrl(url: string): string {
  try {
    return new URL(url, globalThis.location?.href ?? "http://localhost/").href;
  } catch (error) {
    throw new TypeError(`Invalid model URL ${JSON.stringify(url)}: ${formatUnknownError(error)}`);
  }
}

function formatUnknownError(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
