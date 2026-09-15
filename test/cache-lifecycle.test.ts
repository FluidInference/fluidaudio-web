import { afterEach, describe, expect, it, vi } from "vitest";
import { acquireCacheLease } from "../src/engines/musicgen-acestep/cache-lease.js";
import { AceStepMusicClient, aceSeed, type AceGenerationRequest } from "../src/engines/musicgen-acestep/index.js";
import {
  acquireAceDemoModelCache,
  ACE_DEMO_MODEL_CACHE_GENERATION,
  ACE_MODEL_CACHE_LIFECYCLE_LOCK,
  deleteAceDemoModelCache,
} from "../src/engines/musicgen-acestep/model-cache-migration.js";

// FIFO Web Locks test adapter; no model or audio data are needed.
class TestLocks {
  active: { mode: string }[] = [];
  queue: { mode: string; grant: () => void; reject: (error: unknown) => void; signal?: AbortSignal }[] = [];
  request(name: string, options: LockOptions, callback: (lock: Lock | null) => unknown): Promise<unknown> {
    return new Promise((resolve, reject) => {
      const mode = options.mode ?? "exclusive";
      if (options.signal?.aborted) {
        reject(options.signal.reason);
        return;
      }
      const request = {
        mode,
        reject,
        signal: options.signal,
        grant: () => {
          this.active.push(request);
          Promise.resolve()
            .then(() => callback({ name, mode } as Lock))
            .then(
              (value) => {
                this.active.splice(this.active.indexOf(request), 1);
                this.drain();
                resolve(value);
              },
              (error) => {
                this.active.splice(this.active.indexOf(request), 1);
                this.drain();
                reject(error);
              },
            );
        },
      };
      const blocked = () => this.active.some((lock) => lock.mode === "exclusive" || mode === "exclusive");
      if (options.ifAvailable && (blocked() || this.queue.length > 0)) {
        Promise.resolve()
          .then(() => callback(null))
          .then(resolve, reject);
        return;
      }
      options.signal?.addEventListener(
        "abort",
        () => {
          const index = this.queue.indexOf(request);
          if (index < 0) return;
          this.queue.splice(index, 1);
          reject(options.signal?.reason);
          this.drain();
        },
        { once: true },
      );
      this.queue.push(request);
      this.drain();
    });
  }
  drain() {
    for (;;) {
      const next = this.queue[0];
      if (!next || this.active.some((held) => held.mode === "exclusive" || next.mode === "exclusive")) return;
      this.queue.shift();
      next.grant();
    }
  }
  asManager() {
    return this as unknown as LockManager;
  }
}

function browserStorage(generation: string | null = ACE_DEMO_MODEL_CACHE_GENERATION) {
  let marker = generation;
  const removeEntry = vi.fn(async () => {});
  const metadata = {
    async getFileHandle(_name: string, options?: { create?: boolean }) {
      if (marker === null && !options?.create) throw new DOMException("Missing", "NotFoundError");
      return {
        async getFile() {
          return new Blob([marker ?? ""]);
        },
        async createWritable() {
          return {
            async write(value: string) {
              marker = value;
            },
            async close() {},
            async abort() {},
          };
        },
      };
    },
  };
  return {
    async getDirectory() {
      return {
        async getDirectoryHandle() {
          return metadata;
        },
        removeEntry,
      };
    },
    removeEntry,
  };
}

afterEach(() => vi.unstubAllGlobals());

describe("music cache ownership", () => {
  const request: AceGenerationRequest = {
    generationProfile: "ace-turbo-v1-correctness",
    prompt: "Piano instrumental",
    instrumental: true,
    durationSeconds: 30,
    seed: aceSeed("42"),
    planner: { mode: "disabled" },
  };

  it.each(["cancel", "dispose"] as const)("%s while waiting never starts a late worker", async (action) => {
    const locks = new TestLocks();
    vi.stubGlobal("navigator", { locks: locks.asManager(), storage: browserStorage() });
    const worker = vi.fn();
    vi.stubGlobal("Worker", worker);
    let finish!: () => void;
    const exclusive = locks.request(
      ACE_MODEL_CACHE_LIFECYCLE_LOCK,
      { mode: "exclusive" },
      () =>
        new Promise<void>((resolve) => {
          finish = resolve;
        }),
    );
    const client = new AceStepMusicClient();
    const rejected = expect(client.generate(request)).rejects.toMatchObject({ name: "AbortError" });
    await vi.waitFor(() => expect(locks.queue).toHaveLength(1));
    expect(worker).not.toHaveBeenCalled();
    await client[action]();
    await rejected;
    finish();
    await exclusive;
    await vi.waitFor(() => expect(locks.active).toHaveLength(0));
    expect(worker).not.toHaveBeenCalled();
    expect(client.busy).toBe(false);
  });

  it("releases the worker and lease after initialization errors, including retries", async () => {
    const locks = new TestLocks();
    vi.stubGlobal("navigator", { locks: locks.asManager(), storage: browserStorage() });
    vi.stubGlobal("location", { search: "" });
    // Worker transport only: no model implementation or generated audio.
    const workers: WorkerTransport[] = [];
    class WorkerTransport extends EventTarget {
      terminate = vi.fn();
      constructor() {
        super();
        workers.push(this);
      }
      postMessage(message: { type: string; requestId: number }) {
        expect(message.type).toBe("initialize");
        expect(locks.active.map((lock) => lock.mode)).toEqual(["shared"]);
        queueMicrotask(() =>
          this.dispatchEvent(
            new MessageEvent("message", {
              data: {
                type: "error",
                requestId: message.requestId,
                error: { name: "Error", code: "NETWORK_ERROR", message: "Download interrupted" },
              },
            }),
          ),
        );
      }
    }
    vi.stubGlobal("Worker", WorkerTransport);
    const client = new AceStepMusicClient();
    for (let attempt = 0; attempt < 2; attempt++) {
      await expect(client.generate(request)).rejects.toThrow("Download interrupted");
      await vi.waitFor(() => expect(locks.active).toHaveLength(0));
      expect(workers[attempt].terminate).toHaveBeenCalledOnce();
    }
    expect(client.initialized).toBe(false);
    await client.dispose();
  });

  it("does not grant a runtime lease until an exclusive operation finishes", async () => {
    const locks = new TestLocks();
    let finish!: () => void;
    const exclusive = locks.request(
      "cache",
      { mode: "exclusive" },
      () =>
        new Promise<void>((resolve) => {
          finish = resolve;
        }),
    );
    const ready = vi.fn();
    const lease = acquireCacheLease(locks.asManager(), "cache").then((release) => {
      ready();
      return release;
    });
    await vi.waitFor(() => expect(locks.queue).toHaveLength(1));
    expect(ready).not.toHaveBeenCalled();
    finish();
    await exclusive;
    const release = await lease;
    expect(ready).toHaveBeenCalledOnce();
    await release();
  });

  it("cancels a queued lease without acquiring an orphaned lock later", async () => {
    const locks = new TestLocks();
    let finish!: () => void;
    const exclusive = locks.request(
      "cache",
      { mode: "exclusive" },
      () =>
        new Promise<void>((resolve) => {
          finish = resolve;
        }),
    );
    const controller = new AbortController();
    const lease = acquireCacheLease(locks.asManager(), "cache", controller.signal);
    const rejected = expect(lease).rejects.toMatchObject({ name: "AbortError" });
    await vi.waitFor(() => expect(locks.queue).toHaveLength(1));
    controller.abort();
    await rejected;
    finish();
    await exclusive;
    await vi.waitFor(() => expect(locks.active).toHaveLength(0));
    expect(locks.queue).toHaveLength(0);
  });

  it("allows same-version tabs to share the cache and refuses deletion until both release", async () => {
    const locks = new TestLocks();
    const storage = browserStorage();
    vi.stubGlobal("navigator", { locks: locks.asManager(), storage });
    const release1 = await acquireAceDemoModelCache();
    const release2 = await acquireAceDemoModelCache();
    expect(locks.active).toHaveLength(2);
    await expect(deleteAceDemoModelCache()).rejects.toThrow("in use in another tab");
    expect(storage.removeEntry).not.toHaveBeenCalled();
    await release1();
    await expect(deleteAceDemoModelCache()).rejects.toThrow("in use in another tab");
    await release2();
    await expect(deleteAceDemoModelCache()).resolves.toBe(true);
    expect(storage.removeEntry).toHaveBeenCalledOnce();
  });

  it("waits for this page's requested release before trying an immediate deletion", async () => {
    const locks = new TestLocks();
    const storage = browserStorage();
    vi.stubGlobal("navigator", { locks: locks.asManager(), storage });
    const release = await acquireAceDemoModelCache();
    void release();
    await expect(deleteAceDemoModelCache()).resolves.toBe(true);
    expect(storage.removeEntry).toHaveBeenCalledOnce();
  });

  it("migrates a stale cache before publishing a runtime lease", async () => {
    const locks = new TestLocks();
    const storage = browserStorage("old-generation");
    vi.stubGlobal("navigator", { locks: locks.asManager(), storage });
    const release = await acquireAceDemoModelCache();
    expect(storage.removeEntry).toHaveBeenCalledOnce();
    expect(locks.active.map((held) => held.mode)).toEqual(["shared"]);
    await release();
  });

  it("cancels preparation while a migration is blocking it", async () => {
    const locks = new TestLocks();
    const storage = browserStorage();
    vi.stubGlobal("navigator", { locks: locks.asManager(), storage });
    let finish!: () => void;
    const exclusive = locks.request(
      ACE_MODEL_CACHE_LIFECYCLE_LOCK,
      { mode: "exclusive" },
      () =>
        new Promise<void>((resolve) => {
          finish = resolve;
        }),
    );
    const controller = new AbortController();
    const rejected = expect(acquireAceDemoModelCache(controller.signal)).rejects.toMatchObject({ name: "AbortError" });
    await vi.waitFor(() => expect(locks.queue).toHaveLength(1));
    controller.abort();
    await rejected;
    finish();
    await exclusive;
    expect(storage.removeEntry).not.toHaveBeenCalled();
  });
});
