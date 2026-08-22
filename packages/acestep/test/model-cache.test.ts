import { describe, expect, it } from "vitest";

import {
  AceOpfsModelCache,
  deleteAceModelCache,
  inspectAceModelCache,
  inspectAceModelStorage,
} from "../src/model/cache.js";
import type { AcePackageFileRecord } from "../src/model/manifest.js";
import { aceSha256Hex } from "../src/model/sha256.js";

describe("content-addressed OPFS model cache", () => {
  it("publishes a payload only after its internal digest check", async () => {
    const root = new MemoryDirectory();
    const cache = new AceOpfsModelCache(root.asHandle());
    const bytes = Uint8Array.of(1, 2, 3, 4);
    const record = asset(bytes);
    const partial = await cache.begin(record);
    expect(partial.resumeOffset).toBe(0);
    await partial.write(0, bytes.subarray(0, 2));
    await partial.write(2, bytes.subarray(2));
    expect(await cache.openCandidate(record)).toBeUndefined();
    await partial.commit();
    expect(new Uint8Array(await (await cache.openCandidate(record))!.arrayBuffer())).toEqual(
      bytes,
    );
    await expect(cache.begin(record)).rejects.toMatchObject({
      name: "InvalidModificationError",
    });
  });

  it("retains a rolled-back partial and hashes it before append/resume", async () => {
    const cache = new AceOpfsModelCache(new MemoryDirectory().asHandle());
    const bytes = Uint8Array.of(5, 6, 7, 8, 9);
    const record = asset(bytes);
    const first = await cache.begin(record);
    await first.write(0, bytes.subarray(0, 2));
    await first.rollback();

    const resumed = await cache.begin(record);
    expect(resumed.resumeOffset).toBe(2);
    await expect(resumed.write(2, bytes.subarray(2))).rejects.toThrow(
      /prefix must be hashed/,
    );
    await resumed.hashExistingPrefix();
    await resumed.write(2, bytes.subarray(2));
    await resumed.commit();
    expect((await cache.openCandidate(record))?.size).toBe(bytes.byteLength);
  });

  it("discards a complete payload whose bytes do not match its identity", async () => {
    const cache = new AceOpfsModelCache(new MemoryDirectory().asHandle());
    const expected = Uint8Array.of(1, 2, 3);
    const partial = await cache.begin(asset(expected));
    await partial.write(0, Uint8Array.of(1, 2, 4));
    await expect(partial.commit()).rejects.toThrow(/SHA-256 mismatch/);
    expect(await cache.openCandidate(asset(expected))).toBeUndefined();
  });

  it("can restart a partial when a server ignores a Range request", async () => {
    const cache = new AceOpfsModelCache(new MemoryDirectory().asHandle());
    const bytes = Uint8Array.of(10, 11, 12);
    const record = asset(bytes);
    const first = await cache.begin(record);
    await first.write(0, bytes.subarray(0, 1));
    await first.rollback();
    const second = await cache.begin(record);
    expect(second.resumeOffset).toBe(1);
    await second.restart();
    expect(second.resumeOffset).toBe(0);
    await second.write(0, bytes);
    await second.commit();
    expect((await cache.openCandidate(record))?.size).toBe(3);
  });

  it("reserves an identity before asynchronous candidate lookup", async () => {
    const cache = new AceOpfsModelCache(new MemoryDirectory().asHandle());
    const record = asset(Uint8Array.of(1, 2, 3));
    const first = cache.begin(record);
    await expect(cache.begin(record)).rejects.toMatchObject({
      name: "InvalidStateError",
    });
    await (await first).discard();
  });

  it("hashes the stable write copy when a producer mutates its input", async () => {
    const cache = new AceOpfsModelCache(new MemoryDirectory().asHandle());
    const original = Uint8Array.of(7, 8, 9, 10);
    const record = asset(original);
    const partial = await cache.begin(record);
    const write = partial.write(0, original);
    original.fill(0);
    await write;
    await partial.commit();
    const candidate = await cache.openCandidate(record);
    expect(new Uint8Array(await candidate!.arrayBuffer())).toEqual(
      Uint8Array.of(7, 8, 9, 10),
    );
  });

  it("aborts a writable and releases its reservation when begin setup fails", async () => {
    const hooks: MemoryFsHooks = { failPayloadSeek: true, abortCalls: 0 };
    const cache = new AceOpfsModelCache(new MemoryDirectory(hooks).asHandle());
    const record = asset(Uint8Array.of(11, 12, 13));
    await expect(cache.begin(record)).rejects.toThrow(/injected seek failure/);
    expect(hooks.abortCalls).toBe(1);

    hooks.failPayloadSeek = false;
    const recovered = await cache.begin(record);
    await recovered.discard();
  });

  it("keeps the digest reserved until mismatched bytes are deleted", async () => {
    const removalStarted = deferred<void>();
    const allowRemoval = deferred<void>();
    const hooks: MemoryFsHooks = {
      beforeDirectoryRemove: async () => {
        removalStarted.resolve();
        await allowRemoval.promise;
      },
    };
    const cache = new AceOpfsModelCache(new MemoryDirectory(hooks).asHandle());
    const record = asset(Uint8Array.of(21, 22, 23));
    const partial = await cache.begin(record);
    await partial.write(0, Uint8Array.of(21, 22, 24));
    const committing = partial.commit();
    await removalStarted.promise;
    await expect(cache.begin(record)).rejects.toMatchObject({
      name: "InvalidStateError",
    });
    allowRemoval.resolve();
    await expect(committing).rejects.toThrow(/SHA-256 mismatch/);
  });

  it("keeps the digest reserved through explicit discard deletion", async () => {
    const removalStarted = deferred<void>();
    const allowRemoval = deferred<void>();
    const hooks: MemoryFsHooks = {
      beforeDirectoryRemove: async () => {
        removalStarted.resolve();
        await allowRemoval.promise;
      },
    };
    const cache = new AceOpfsModelCache(new MemoryDirectory(hooks).asHandle());
    const record = asset(Uint8Array.of(31, 32, 33));
    const partial = await cache.begin(record);
    await partial.write(0, Uint8Array.of(31));
    const discarding = partial.discard();
    await removalStarted.promise;
    await expect(cache.begin(record)).rejects.toMatchObject({
      name: "InvalidStateError",
    });
    allowRemoval.resolve();
    await expect(discarding).resolves.toBeUndefined();
  });

  it("reports OPFS persistence and quota without requesting persistence", async () => {
    const storage = {
      getDirectory: async () => new MemoryDirectory().asHandle(),
      persisted: async () => true,
      estimate: async () => ({ usage: 10, quota: 100 }),
      persist: async () => {
        throw new Error("must not request persistence during inspection");
      },
    } as unknown as StorageManager;
    await expect(inspectAceModelStorage(storage)).resolves.toEqual({
      supported: true,
      persisted: true,
      usageBytes: 10,
      quotaBytes: 100,
    });
  });

  it("inspects and deletes only the package-owned model cache", async () => {
    const root = new MemoryDirectory();
    const storage = {
      getDirectory: async () => root.asHandle(),
      persisted: async () => true,
    } as Pick<StorageManager, "getDirectory" | "persisted">;
    const cache = await AceOpfsModelCache.open(storage);
    const bytes = Uint8Array.of(41, 42, 43, 44);
    const partial = await cache.begin(asset(bytes));
    await partial.write(0, bytes);
    await partial.commit();
    await root.getDirectoryHandle("ace-step-1.5.wgsl-audio-v1", { create: true });

    await expect(inspectAceModelCache(storage)).resolves.toEqual({
      supported: true,
      persisted: true,
      assetCount: 1,
      sizeBytes: bytes.byteLength,
      partialAssetCount: 0,
    });
    await expect(deleteAceModelCache(storage)).resolves.toBe(true);
    await expect(deleteAceModelCache(storage)).resolves.toBe(false);
    await expect(
      root.getDirectoryHandle("ace-step-1.5.wgsl-audio-v1"),
    ).resolves.toBeDefined();
  });

  it("reports unavailable or absent model storage without creating a cache", async () => {
    await expect(
      inspectAceModelCache({} as Pick<StorageManager, "getDirectory" | "persisted">),
    ).resolves.toEqual({
      supported: false,
      persisted: false,
      assetCount: 0,
      sizeBytes: 0,
      partialAssetCount: 0,
    });

    const root = new MemoryDirectory();
    const storage = {
      getDirectory: async () => root.asHandle(),
      persisted: async () => {
        throw new DOMException("denied", "NotAllowedError");
      },
    } as Pick<StorageManager, "getDirectory" | "persisted">;
    await expect(inspectAceModelCache(storage)).resolves.toEqual({
      supported: true,
      persisted: false,
      assetCount: 0,
      sizeBytes: 0,
      partialAssetCount: 0,
    });
    expect(root.empty).toBe(true);
  });

  it("reports resumable partial identities separately from verified assets", async () => {
    const root = new MemoryDirectory();
    const storage = {
      getDirectory: async () => root.asHandle(),
      persisted: async () => false,
    } as Pick<StorageManager, "getDirectory" | "persisted">;
    const cache = await AceOpfsModelCache.open(storage);
    const partial = await cache.begin(asset(Uint8Array.of(51, 52, 53)));
    await partial.write(0, Uint8Array.of(51));
    await partial.rollback();

    await expect(inspectAceModelCache(storage)).resolves.toEqual({
      supported: true,
      persisted: false,
      assetCount: 0,
      sizeBytes: 0,
      partialAssetCount: 1,
    });
  });
});

function asset(bytes: Uint8Array): AcePackageFileRecord {
  return {
    name: "weights/test.bin",
    byteLength: bytes.byteLength,
    sha256: aceSha256Hex(bytes),
    kind: "weights",
  };
}

interface MemoryFsHooks {
  failPayloadSeek?: boolean;
  abortCalls?: number;
  beforeDirectoryRemove?: () => Promise<void>;
}

class MemoryDirectory {
  private readonly directories = new Map<string, MemoryDirectory>();
  private readonly files = new Map<string, MemoryFile>();

  constructor(private readonly hooks: MemoryFsHooks = {}) {}

  asHandle(): FileSystemDirectoryHandle {
    return this as unknown as FileSystemDirectoryHandle;
  }

  async getDirectoryHandle(
    name: string,
    options: FileSystemGetDirectoryOptions = {},
  ): Promise<FileSystemDirectoryHandle> {
    let directory = this.directories.get(name);
    if (directory === undefined && options.create === true) {
      directory = new MemoryDirectory(this.hooks);
      this.directories.set(name, directory);
    }
    if (directory === undefined) throw notFound();
    return directory.asHandle();
  }

  async getFileHandle(
    name: string,
    options: FileSystemGetFileOptions = {},
  ): Promise<FileSystemFileHandle> {
    let file = this.files.get(name);
    if (file === undefined && options.create === true) {
      file = new MemoryFile(name, this.hooks);
      this.files.set(name, file);
    }
    if (file === undefined) throw notFound();
    return file.asHandle();
  }

  async removeEntry(name: string, options: FileSystemRemoveOptions = {}): Promise<void> {
    if (this.files.delete(name)) return;
    const directory = this.directories.get(name);
    if (directory === undefined) throw notFound();
    if (!options.recursive && !directory.empty) {
      throw new DOMException("not empty", "InvalidModificationError");
    }
    await this.hooks.beforeDirectoryRemove?.();
    this.directories.delete(name);
  }

  async *entries(): AsyncIterableIterator<
    [string, FileSystemDirectoryHandle | FileSystemFileHandle]
  > {
    for (const [name, directory] of this.directories) {
      yield [name, directory.asHandle()];
    }
    for (const [name, file] of this.files) {
      yield [name, file.asHandle()];
    }
  }

  readonly kind = "directory" as const;

  get empty(): boolean {
    return this.files.size === 0 && this.directories.size === 0;
  }
}

class MemoryFile {
  bytes = new Uint8Array();

  constructor(
    private readonly name: string,
    private readonly hooks: MemoryFsHooks,
  ) {}

  asHandle(): FileSystemFileHandle {
    return this as unknown as FileSystemFileHandle;
  }

  async getFile(): Promise<File> {
    return new File([Uint8Array.from(this.bytes)], this.name);
  }

  async createWritable(
    options: FileSystemCreateWritableOptions = {},
  ): Promise<FileSystemWritableFileStream> {
    return new MemoryWritable(
      this,
      options.keepExistingData === true,
      this.name,
      this.hooks,
    ).asStream();
  }
}

class MemoryWritable {
  private bytes: Uint8Array;
  private cursor = 0;
  private closed = false;

  constructor(
    private readonly target: MemoryFile,
    keepExisting: boolean,
    private readonly name: string,
    private readonly hooks: MemoryFsHooks,
  ) {
    this.bytes = keepExisting ? target.bytes.slice() : new Uint8Array();
  }

  asStream(): FileSystemWritableFileStream {
    return this as unknown as FileSystemWritableFileStream;
  }

  async seek(position: number): Promise<void> {
    this.assertOpen();
    if (this.name === "payload.bin" && this.hooks.failPayloadSeek === true) {
      throw new Error("injected seek failure");
    }
    this.cursor = position;
  }

  async truncate(size: number): Promise<void> {
    this.assertOpen();
    const next = new Uint8Array(size);
    next.set(this.bytes.subarray(0, size));
    this.bytes = next;
    this.cursor = Math.min(this.cursor, size);
  }

  async write(chunk: FileSystemWriteChunkType): Promise<void> {
    this.assertOpen();
    if (!(chunk instanceof Uint8Array)) throw new TypeError("test expects Uint8Array");
    const nextLength = Math.max(this.bytes.byteLength, this.cursor + chunk.byteLength);
    const next = new Uint8Array(nextLength);
    next.set(this.bytes);
    next.set(chunk, this.cursor);
    this.bytes = next;
    this.cursor += chunk.byteLength;
  }

  async close(): Promise<void> {
    this.assertOpen();
    this.closed = true;
    this.target.bytes = Uint8Array.from(this.bytes);
  }

  async abort(): Promise<void> {
    this.closed = true;
    this.hooks.abortCalls = (this.hooks.abortCalls ?? 0) + 1;
  }

  private assertOpen(): void {
    if (this.closed) throw new DOMException("closed", "InvalidStateError");
  }
}

function notFound(): DOMException {
  return new DOMException("missing", "NotFoundError");
}

function deferred<T>() {
  let resolve!: (value: T | PromiseLike<T>) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
}
