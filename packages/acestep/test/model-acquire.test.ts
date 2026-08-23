import { describe, expect, it, vi } from "vitest";

import {
  ACE_MODEL_CACHE_WEBCRYPTO_MAX_FILE_BYTES,
  ACE_PRODUCTION_CACHE_AUTHENTICATION_OWNER,
  acquireAceModelFiles,
  aceRuntimePackageFiles,
  planAceModelAcquisition,
  requestAceModelStoragePersistence,
  type AceModelCacheBackend,
  type AceModelAcquisitionTrace,
} from "../src/model/acquire.js";
import type {
  AcePackageFileRecord,
  AcePackageManifest,
} from "../src/model/manifest.js";
import { isAceAuthenticatedGpuSource } from "../src/model/gpu-upload.js";
import { aceSha256Hex } from "../src/model/sha256.js";

describe("whole-package acquisition", () => {
  it("plans only runtime assets and accounts for cache and storage headroom", async () => {
    const first = Uint8Array.of(1, 2, 3, 4);
    const second = Uint8Array.of(5, 6, 7, 8);
    const manifest = manifestWith([
      file("weights/a.bin", "weights", first),
      file("constants/b.bin", "constant", second),
      file("licenses/notice", "license", Uint8Array.of(9)),
      file("conversion-plan.json", "conversion-plan", Uint8Array.of(10)),
    ]);
    const cache = new MemoryAssetCache(new Map([["weights/a.bin", first]]));
    const plan = await planAceModelAcquisition(
      manifest,
      cache,
      { estimate: async () => ({ usage: 100, quota: 1_000 }) },
      100,
    );
    expect(aceRuntimePackageFiles(manifest).map((item) => item.name)).toEqual([
      "weights/a.bin",
      "constants/b.bin",
    ]);
    expect(plan).toMatchObject({
      runtimeBytes: 8,
      cachedBytes: 4,
      downloadBytes: 4,
      requiredFreeBytes: 104,
      availableFreeBytes: 900,
      quotaSufficient: true,
      cachedFiles: ["weights/a.bin"],
      downloadFiles: ["constants/b.bin"],
    });
  });

  it("fails before downloading when quota cannot hold missing bytes plus headroom", async () => {
    const bytes = Uint8Array.of(1, 2, 3, 4);
    const fetchMock = vi.fn<typeof fetch>();
    await expect(
      acquireAceModelFiles({
        manifest: manifestWith([file("weights/a.bin", "weights", bytes)]),
        manifestUrl: "https://models.example/manifest.json",
        cache: new MemoryAssetCache(),
        storage: { estimate: async () => ({ usage: 90, quota: 100 }) },
        headroomBytes: 20,
        fetch: fetchMock,
      }),
    ).rejects.toMatchObject({ name: "QuotaExceededError" });
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("treats a rejected quota estimate as unknown rather than failing planning", async () => {
    const bytes = Uint8Array.of(1, 2, 3, 4);
    const plan = await planAceModelAcquisition(
      manifestWith([file("weights/a.bin", "weights", bytes)]),
      new MemoryAssetCache(),
      {
        estimate: async () => {
          throw new DOMException("estimate denied", "NotAllowedError");
        },
      },
      20,
    );
    expect(plan.downloadBytes).toBe(4);
    expect(plan).not.toHaveProperty("availableFreeBytes");
    expect(plan).not.toHaveProperty("quotaSufficient");
  });

  it("acquires missing files sequentially while preserving cache hits as File handles", async () => {
    const cached = Uint8Array.of(1, 1, 1, 1);
    const downloaded = Uint8Array.of(2, 2, 2, 2);
    const cachedRecord = file("weights/cached.bin", "weights", cached);
    const networkRecord = file("weights/network.bin", "weights", downloaded);
    const cache = new MemoryAssetCache(new Map([["weights/cached.bin", cached]]));
    const events: string[] = [];
    const traces: AceModelAcquisitionTrace[] = [];
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      events.push(`fetch:${new URL(String(input)).pathname}`);
      return new Response(downloaded, {
        headers: { "content-length": String(downloaded.byteLength) },
      });
    });
    const result = await acquireAceModelFiles({
      manifest: manifestWith([cachedRecord, networkRecord]),
      manifestUrl: "https://models.example/v1/manifest.json",
      cache,
      storage: { estimate: async () => ({ usage: 0, quota: 1_000 }) },
      headroomBytes: 0,
      fetch: fetchMock as typeof fetch,
      onFileProgress: (progress) =>
        events.push(`${progress.source}:${progress.file}:${progress.completedBytes}`),
      onTrace: (trace) => traces.push(trace),
    });
    expect([...result.files.keys()]).toEqual([
      "weights/cached.bin",
      "weights/network.bin",
    ]);
    expect(
      new Uint8Array(await result.files.get("weights/network.bin")!.arrayBuffer()),
    ).toEqual(downloaded);
    expect(
      isAceAuthenticatedGpuSource(
        result.files.get("weights/cached.bin")!,
        cachedRecord,
      ),
    ).toBe(true);
    expect(
      isAceAuthenticatedGpuSource(
        result.files.get("weights/network.bin")!,
        networkRecord,
      ),
    ).toBe(true);
    expect(events).toEqual([
      "cache:weights/cached.bin:4",
      "fetch:/v1/weights/network.bin",
      "network:weights/network.bin:8",
    ]);
    expect(cache.maximumActiveWriters).toBe(1);
    expect(traces.filter((trace) => trace.operation === "cache-authentication"))
      .toEqual([
        expect.objectContaining({
          file: cachedRecord.name,
          authenticationOwner: "webcrypto-whole-file",
          matched: true,
          actualSha256: cachedRecord.sha256,
          maximumHashChunkBytes: cachedRecord.byteLength,
          exactImmutableFileProofPublished: true,
        }),
        expect.objectContaining({
          file: networkRecord.name,
          authenticationOwner: "webcrypto-whole-file",
          matched: true,
          actualSha256: networkRecord.sha256,
          maximumHashChunkBytes: networkRecord.byteLength,
          exactImmutableFileProofPublished: true,
        }),
      ]);
    expect(traces.filter((trace) => trace.operation === "proof-reuse")).toEqual([
      expect.objectContaining({
        file: cachedRecord.name,
        source: "cache",
        exactImmutableFileIdentity: true,
        redundantHashPerformed: false,
      }),
      expect.objectContaining({
        file: networkRecord.name,
        source: "network",
        exactImmutableFileIdentity: true,
        redundantHashPerformed: false,
      }),
    ]);
  });

  it("invalidates and redownloads a corrupt cached upstream asset", async () => {
    const expected = Uint8Array.of(40, 41, 42, 43);
    const record = file("assets/qwen-tokenizer.json", "upstream-asset", expected);
    const cache = new MemoryAssetCache();
    cache.seedIdentity(record.sha256, Uint8Array.of(40, 41, 42, 44));
    const fetchMock = vi.fn(async () =>
      new Response(expected, {
        headers: { "content-length": String(expected.byteLength) },
      }),
    );
    const result = await acquireAceModelFiles({
      manifest: manifestWith([record]),
      manifestUrl: "https://models.example/manifest.json",
      cache,
      storage: { estimate: async () => ({ usage: 0, quota: 1_000 }) },
      headroomBytes: 0,
      fetch: fetchMock as typeof fetch,
    });
    expect(fetchMock).toHaveBeenCalledOnce();
    expect(cache.removedDigests).toEqual([record.sha256]);
    expect(result.plan.downloadFiles).toEqual([record.name]);
    expect(
      new Uint8Array(await result.files.get(record.name)!.arrayBuffer()),
    ).toEqual(expected);
  });

  it("deduplicates physical download accounting and transport by digest", async () => {
    const bytes = Uint8Array.of(50, 51, 52, 53);
    const first = file("weights/alias-a.bin", "weights", bytes);
    const second = file("weights/alias-b.bin", "weights", bytes);
    const cache = new MemoryAssetCache();
    const plan = await planAceModelAcquisition(
      manifestWith([first, second]),
      cache,
      { estimate: async () => ({ usage: 0, quota: 1_000 }) },
      0,
    );
    expect(plan).toMatchObject({
      runtimeBytes: 8,
      downloadBytes: 4,
      requiredFreeBytes: 4,
      downloadFiles: [first.name, second.name],
    });

    const fetchMock = vi.fn(async () =>
      new Response(bytes, {
        headers: { "content-length": String(bytes.byteLength) },
      }),
    );
    const result = await acquireAceModelFiles({
      manifest: manifestWith([first, second]),
      manifestUrl: "https://models.example/manifest.json",
      cache,
      storage: { estimate: async () => ({ usage: 0, quota: 1_000 }) },
      headroomBytes: 0,
      fetch: fetchMock as typeof fetch,
    });
    expect(fetchMock).toHaveBeenCalledOnce();
    expect([...result.files.keys()]).toEqual([first.name, second.name]);
  });

  it("requests persistence only through the explicit helper", async () => {
    const persist = vi.fn(async () => true);
    await expect(requestAceModelStoragePersistence({ persist })).resolves.toBe(true);
    expect(persist).toHaveBeenCalledOnce();
  });

  it("keeps capture callbacks observational after exact proof publication", async () => {
    const bytes = Uint8Array.of(1, 2, 3, 4);
    const record = file("weights/a.bin", "weights", bytes);
    const result = await acquireAceModelFiles({
      manifest: manifestWith([record]),
      manifestUrl: "https://models.example/manifest.json",
      cache: new MemoryAssetCache(new Map([[record.name, bytes]])),
      storage: { estimate: async () => ({ usage: 0, quota: 1_000 }) },
      headroomBytes: 0,
      onTrace: () => {
        throw new Error("observational sink failure");
      },
    });
    expect(result.plan.cachedFiles).toEqual([record.name]);
    expect(isAceAuthenticatedGpuSource(result.files.get(record.name)!, record))
      .toBe(true);
  });

  it("uses one bounded WebCrypto payload and honors cancellation after an in-flight file", async () => {
    expect(ACE_MODEL_CACHE_WEBCRYPTO_MAX_FILE_BYTES).toBe(128 * 1024 * 1024);
    expect(ACE_PRODUCTION_CACHE_AUTHENTICATION_OWNER).toBe(
      "webcrypto-whole-file",
    );
    const bytes = Uint8Array.of(9, 8, 7, 6);
    const record = file("weights/cancel.bin", "weights", bytes);
    const controller = new AbortController();
    const candidate = new File([bytes], record.name);
    Object.defineProperty(candidate, "arrayBuffer", {
      value: async () => {
        controller.abort();
        return Uint8Array.from(bytes).buffer;
      },
    });
    const remove = vi.fn(async () => {});
    const cache: AceModelCacheBackend = {
      openCandidate: async () => candidate,
      begin: async () => {
        throw new Error("cancelled cache candidate must not start a download");
      },
      remove,
    };
    await expect(acquireAceModelFiles({
      manifest: manifestWith([record]),
      manifestUrl: "https://models.example/manifest.json",
      cache,
      signal: controller.signal,
      storage: { estimate: async () => ({ usage: 0, quota: 1_000 }) },
      headroomBytes: 0,
      cacheAuthenticationOwner: "webcrypto-whole-file",
    })).rejects.toMatchObject({ name: "AbortError" });
    expect(remove).not.toHaveBeenCalled();
  });

  it("never retries a failed production WebCrypto proof with scalar hashing", async () => {
    const bytes = Uint8Array.of(2, 7, 1, 8, 2, 8);
    const record = file("weights/no-fallback.bin", "weights", bytes);
    const candidate = new File([bytes], record.name);
    const arrayBuffer = vi.fn(async () => {
      throw new DOMException("injected WebCrypto read failure", "OperationError");
    });
    const scalarStream = vi.fn(() => {
      throw new Error("scalar fallback must not run");
    });
    Object.defineProperty(candidate, "arrayBuffer", { value: arrayBuffer });
    Object.defineProperty(candidate, "stream", { value: scalarStream });
    const downloaded = new MemoryAssetCache();
    const remove = vi.fn(async () => undefined);
    let firstOpen = true;
    const cache: AceModelCacheBackend = {
      openCandidate: async (file) => {
        if (firstOpen) {
          firstOpen = false;
          return candidate;
        }
        return await downloaded.openCandidate(file);
      },
      remove,
      begin: async (file) => await downloaded.begin(file),
    };
    const fetchMock = vi.fn(async () => new Response(bytes, {
      headers: { "content-length": String(bytes.byteLength) },
    }));

    const result = await acquireAceModelFiles({
      manifest: manifestWith([record]),
      manifestUrl: "https://models.example/manifest.json",
      cache,
      storage: { estimate: async () => ({ usage: 0, quota: 1_000 }) },
      headroomBytes: 0,
      fetch: fetchMock as typeof fetch,
    });

    expect(ACE_PRODUCTION_CACHE_AUTHENTICATION_OWNER).toBe(
      "webcrypto-whole-file",
    );
    expect(arrayBuffer).toHaveBeenCalledOnce();
    expect(scalarStream).not.toHaveBeenCalled();
    expect(remove).toHaveBeenCalledOnce();
    expect(fetchMock).toHaveBeenCalledOnce();
    expect(result.plan.downloadFiles).toEqual([record.name]);
  });

  it("honors an explicit WebCrypto owner at the authenticated diagnostic seam", async () => {
    const bytes = Uint8Array.of(3, 1, 4, 1, 5, 9);
    const record = file("weights/webcrypto.bin", "weights", bytes);
    const traces: AceModelAcquisitionTrace[] = [];
    const result = await acquireAceModelFiles({
      manifest: manifestWith([record]),
      manifestUrl: "https://models.example/manifest.json",
      cache: new MemoryAssetCache(new Map([[record.name, bytes]])),
      storage: { estimate: async () => ({ usage: 0, quota: 1_000 }) },
      headroomBytes: 0,
      cacheAuthenticationOwner: "webcrypto-whole-file",
      onTrace: (trace) => traces.push(trace),
    });
    expect(result.plan.downloadFiles).toEqual([]);
    expect(traces).toContainEqual(expect.objectContaining({
      operation: "cache-authentication",
      authenticationOwner: "webcrypto-whole-file",
      receivedBytes: bytes.byteLength,
      hashChunkCount: 1,
      maximumHashChunkBytes: bytes.byteLength,
      actualSha256: record.sha256,
      matched: true,
    }));
  });
});

function file(
  name: string,
  kind: AcePackageFileRecord["kind"],
  bytes: Uint8Array,
): AcePackageFileRecord {
  return { name, kind, byteLength: bytes.byteLength, sha256: aceSha256Hex(bytes) };
}

function manifestWith(files: readonly AcePackageFileRecord[]): AcePackageManifest {
  return { files } as unknown as AcePackageManifest;
}

class MemoryAssetCache implements AceModelCacheBackend {
  private readonly records = new Map<string, Uint8Array>();
  private activeWriters = 0;
  maximumActiveWriters = 0;
  readonly removedDigests: string[] = [];

  constructor(initial: ReadonlyMap<string, Uint8Array> = new Map()) {
    for (const bytes of initial.values()) {
      this.records.set(aceSha256Hex(bytes), Uint8Array.from(bytes));
    }
  }

  seedIdentity(digest: string, bytes: Uint8Array): void {
    this.records.set(digest, Uint8Array.from(bytes));
  }

  async openCandidate(file: AcePackageFileRecord): Promise<File | undefined> {
    const bytes = this.records.get(file.sha256);
    return bytes === undefined ? undefined : new File([Uint8Array.from(bytes)], file.name);
  }

  async remove(file: Pick<AcePackageFileRecord, "sha256">): Promise<void> {
    this.removedDigests.push(file.sha256);
    this.records.delete(file.sha256);
  }

  async begin(file: AcePackageFileRecord) {
    if (this.activeWriters !== 0) throw new Error("test acquisition was not sequential");
    this.activeWriters += 1;
    this.maximumActiveWriters = Math.max(this.maximumActiveWriters, this.activeWriters);
    let bytes = new Uint8Array();
    let closed = false;
    const finish = () => {
      if (!closed) {
        closed = true;
        this.activeWriters -= 1;
      }
    };
    const thisCache = this;
    return {
      get resumeOffset() {
        return bytes.byteLength;
      },
      async hashExistingPrefix() {},
      async restart() {
        bytes = new Uint8Array();
      },
      async write(offset: number, chunk: Uint8Array) {
        if (offset !== bytes.byteLength) throw new RangeError("test offset mismatch");
        const next = new Uint8Array(offset + chunk.byteLength);
        next.set(bytes);
        next.set(chunk, offset);
        bytes = next;
      },
      async commit() {
        if (bytes.byteLength !== file.byteLength || aceSha256Hex(bytes) !== file.sha256) {
          finish();
          throw new Error("test cache integrity mismatch");
        }
        thisCache.records.set(file.sha256, Uint8Array.from(bytes));
        finish();
        return new File([Uint8Array.from(bytes)], file.name);
      },
      async rollback() {
        finish();
      },
      async discard() {
        finish();
        thisCache.records.delete(file.sha256);
      },
    };
  }
}
