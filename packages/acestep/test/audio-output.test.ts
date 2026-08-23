import { describe, expect, it } from "vitest";

import {
  AceAudioOutputTransaction,
  compareAceRawAudioSnapshots,
  recoverStaleAceAudioOutputs,
  releaseAceAudioOutput,
} from "../src/runtime/audio-output.js";
import {
  deriveAceVaePostprocessPlan,
  planAceVaeChunkedDecode,
} from "../src/webgpu/vae-chunks.js";
import type { AceVaeDecoderConfig } from "../src/webgpu/vae-decoder.js";

const TOY_CONFIG: AceVaeDecoderConfig = {
  id: "audio-output-test",
  decoderInputChannels: 2,
  decoderChannels: 4,
  audioChannels: 2,
  channelMultiples: [1],
  downsamplingRatios: [2],
  sampleRateHz: 48_000,
};

describe("transactional worker audio output", () => {
  it("streams raw samples into an OPFS-backed float WAV and removes raw state", async () => {
    const storage = new MemoryStorage();
    const plan = planAceVaeChunkedDecode(1, {
      config: TOY_CONFIG,
      chunkFrames: 2,
      overlapFrames: 0,
    });
    const transaction = await AceAudioOutputTransaction.begin(
      "job-17",
      plan,
      storage as never,
    );
    await transaction.rawSink.writeCore(plan.windows[0]!, new Float32Array([
      2, -2, 0.5, -0.5,
    ]));
    const committed = await transaction.commit(deriveAceVaePostprocessPlan(2));

    expect(committed.audio).toBeInstanceOf(Blob);
    expect(committed.audio.type).toBe("audio/wav");
    expect(committed.wav.wavBytes).toBe(60);
    const bytes = new Uint8Array(await committed.audio.arrayBuffer());
    expect(new TextDecoder().decode(bytes.subarray(0, 4))).toBe("RIFF");
    expect(new DataView(bytes.buffer).getUint16(20, true)).toBe(3);
    expect(storage.path("ace-step-1.5.wgsl-audio-v1/job-17/raw.f32.partial"))
      .toBeUndefined();
    expect(storage.path("ace-step-1.5.wgsl-audio-v1/job-17/output.wav"))
      .toBeDefined();
    await expect(AceAudioOutputTransaction.begin("job-17", plan, storage as never))
      .rejects.toThrow(/already exists/);
    await releaseAceAudioOutput(committed.transactionId, storage as never);
    expect(storage.path("ace-step-1.5.wgsl-audio-v1/job-17"))
      .toBeUndefined();
  });

  it("rolls back only its validated job directory and permits exact retry", async () => {
    const storage = new MemoryStorage();
    const plan = planAceVaeChunkedDecode(1, {
      config: TOY_CONFIG,
      chunkFrames: 2,
      overlapFrames: 0,
    });
    const first = await AceAudioOutputTransaction.begin(
      "cancelled-9",
      plan,
      storage as never,
    );
    await first.rollback();
    await first.rollback();
    expect(storage.path("ace-step-1.5.wgsl-audio-v1/cancelled-9"))
      .toBeUndefined();
    const retry = await AceAudioOutputTransaction.begin(
      "cancelled-9",
      plan,
      storage as never,
    );
    await retry.rollback();
    await expect(AceAudioOutputTransaction.begin("../escape", plan, storage as never))
      .rejects.toThrow(/transaction ID/);
    await expect(AceAudioOutputTransaction.begin(".", plan, storage as never))
      .rejects.toThrow(/transaction ID/);
    await expect(AceAudioOutputTransaction.begin("..", plan, storage as never))
      .rejects.toThrow(/transaction ID/);
  });

  it("keeps rollback retryable when recursive cleanup fails", async () => {
    const storage = new MemoryStorage();
    const plan = planAceVaeChunkedDecode(1, {
      config: TOY_CONFIG,
      chunkFrames: 2,
      overlapFrames: 0,
    });
    const transaction = await AceAudioOutputTransaction.begin(
      "retry-cleanup",
      plan,
      storage as never,
    );
    const audioRoot = storage.path("ace-step-1.5.wgsl-audio-v1");
    if (!(audioRoot instanceof MemoryDirectory)) throw new Error("missing audio root");
    audioRoot.failNextRemoval = true;
    await expect(transaction.rollback()).rejects.toThrow(/injected removal/);
    expect(storage.path("ace-step-1.5.wgsl-audio-v1/retry-cleanup"))
      .toBeDefined();
    await transaction.rollback();
    expect(storage.path("ace-step-1.5.wgsl-audio-v1/retry-cleanup"))
      .toBeUndefined();
  });

  it("yields during WAV conversion so cancellation can roll back partial output", async () => {
    const storage = new MemoryStorage();
    const plan = planAceVaeChunkedDecode(1, {
      config: TOY_CONFIG,
      chunkFrames: 2,
      overlapFrames: 0,
    });
    const transaction = await AceAudioOutputTransaction.begin(
      "cancel-wav-pass",
      plan,
      storage as never,
    );
    await transaction.rawSink.writeCore(
      plan.windows[0]!,
      new Float32Array([2, -2, 0.5, -0.5]),
    );
    const controller = new AbortController();
    await expect(transaction.commit(deriveAceVaePostprocessPlan(2), {
      blockAudioFrames: 1,
      signal: controller.signal,
      yieldEveryBlocks: 1,
      yieldToEventLoop: async () => controller.abort(),
    })).rejects.toMatchObject({ name: "AbortError" });
    expect(storage.path(
      "ace-step-1.5.wgsl-audio-v1/cancel-wav-pass/output.wav",
    )).toBeDefined();
    await transaction.rollback();
    expect(storage.path("ace-step-1.5.wgsl-audio-v1/cancel-wav-pass"))
      .toBeUndefined();
  });

  it("reclaims only aged incomplete jobs and preserves committed or fresh jobs", async () => {
    const storage = new MemoryStorage();
    const plan = planAceVaeChunkedDecode(1, {
      config: TOY_CONFIG,
      chunkFrames: 2,
      overlapFrames: 0,
    });
    const stale = await AceAudioOutputTransaction.begin(
      "stale-partial",
      plan,
      storage as never,
    );
    const fresh = await AceAudioOutputTransaction.begin(
      "fresh-partial",
      plan,
      storage as never,
    );
    const committed = await AceAudioOutputTransaction.begin(
      "committed-output",
      plan,
      storage as never,
    );
    await committed.rawSink.writeCore(
      plan.windows[0]!,
      new Float32Array([2, -2, 0.5, -0.5]),
    );
    await committed.commit(deriveAceVaePostprocessPlan(2), {
      yieldToEventLoop: async () => undefined,
    });
    storage.setLastModified(
      "ace-step-1.5.wgsl-audio-v1/stale-partial/active.partial",
      1_000,
    );
    storage.setLastModified(
      "ace-step-1.5.wgsl-audio-v1/stale-partial/raw.f32.partial",
      1_000,
    );
    storage.setLastModified(
      "ace-step-1.5.wgsl-audio-v1/stale-partial/output.wav",
      1_000,
    );
    storage.setLastModified(
      "ace-step-1.5.wgsl-audio-v1/fresh-partial/active.partial",
      9_500,
    );

    await expect(recoverStaleAceAudioOutputs(storage as never, {
      nowMs: 10_000,
      minimumIncompleteAgeMs: 1_000,
    })).resolves.toEqual(["stale-partial"]);
    expect(storage.path("ace-step-1.5.wgsl-audio-v1/stale-partial"))
      .toBeUndefined();
    expect(storage.path("ace-step-1.5.wgsl-audio-v1/fresh-partial"))
      .toBeDefined();
    expect(storage.path("ace-step-1.5.wgsl-audio-v1/committed-output"))
      .toBeDefined();
    await fresh.rollback();
    // The stale transaction simulates a terminated worker; do not reuse its
    // invalid in-memory handle after recovery removed the directory.
    void stale;
  });

  it("does not publish a committed marker when incomplete-marker removal fails", async () => {
    const storage = new MemoryStorage();
    const plan = planAceVaeChunkedDecode(1, {
      config: TOY_CONFIG,
      chunkFrames: 2,
      overlapFrames: 0,
    });
    const transaction = await AceAudioOutputTransaction.begin(
      "marker-failure",
      plan,
      storage as never,
    );
    await transaction.rawSink.writeCore(
      plan.windows[0]!,
      new Float32Array([2, -2, 0.5, -0.5]),
    );
    const job = storage.path("ace-step-1.5.wgsl-audio-v1/marker-failure");
    if (!(job instanceof MemoryDirectory)) throw new Error("missing audio job");
    job.failNextRemoval = true;
    await expect(transaction.commit(deriveAceVaePostprocessPlan(2), {
      yieldToEventLoop: async () => undefined,
    })).rejects.toThrow(/injected removal/);
    expect(storage.path(
      "ace-step-1.5.wgsl-audio-v1/marker-failure/committed",
    )).toBeUndefined();
    await transaction.rollback();
    expect(storage.path("ace-step-1.5.wgsl-audio-v1/marker-failure"))
      .toBeUndefined();
  });

  it("retains diagnostic raw snapshots and compares them with bounded blocks", async () => {
    const storage = new MemoryStorage();
    const plan = planAceVaeChunkedDecode(1, {
      config: TOY_CONFIG,
      chunkFrames: 2,
      overlapFrames: 0,
    });
    const commit = async (id: string, samples: readonly number[]) => {
      const transaction = await AceAudioOutputTransaction.begin(
        id,
        plan,
        storage as never,
      );
      await transaction.rawSink.writeCore(
        plan.windows[0]!,
        Float32Array.from(samples),
      );
      return await transaction.commit(deriveAceVaePostprocessPlan(2), {
        retainRawSnapshot: true,
        yieldToEventLoop: async () => undefined,
      });
    };
    const left = await commit("raw-left", [2, -2, 0.5, -0.5]);
    const exact = await commit("raw-exact", [2, -2, 0.5, -0.5]);
    const drift = await commit("raw-drift", [2, -2, 0.25, -0.5]);
    expect(left.rawSnapshot?.size).toBe(plan.outputFloat32Bytes);
    expect(storage.path("ace-step-1.5.wgsl-audio-v1/raw-left/raw.f32.partial"))
      .toBeDefined();
    await expect(compareAceRawAudioSnapshots(
      left.rawSnapshot!,
      exact.rawSnapshot!,
      { blockBytes: 4 },
    )).resolves.toMatchObject({
      sampleCount: 4,
      exactU32MismatchCount: 0,
      maximumAbsoluteDifference: 0,
      rootMeanSquareDifference: 0,
    });
    await expect(compareAceRawAudioSnapshots(
      left.rawSnapshot!,
      drift.rawSnapshot!,
      { blockBytes: 8 },
    )).resolves.toMatchObject({
      sampleCount: 4,
      exactU32MismatchCount: 1,
      maximumAbsoluteDifference: 0.25,
      meanAbsoluteDifference: 0.0625,
      rootMeanSquareDifference: 0.125,
    });
    await releaseAceAudioOutput(left.transactionId, storage as never);
    await releaseAceAudioOutput(exact.transactionId, storage as never);
    await releaseAceAudioOutput(drift.transactionId, storage as never);
  });
});

class MemoryStorage {
  readonly root = new MemoryDirectory("");

  async getDirectory(): Promise<MemoryDirectory> {
    return this.root;
  }

  path(path: string): MemoryDirectory | MemoryFile | undefined {
    let entry: MemoryDirectory | MemoryFile = this.root;
    for (const component of path.split("/")) {
      if (!(entry instanceof MemoryDirectory)) return undefined;
      const next = entry.children.get(component);
      if (next === undefined) return undefined;
      entry = next;
    }
    return entry;
  }

  setLastModified(path: string, lastModified: number): void {
    const entry = this.path(path);
    if (!(entry instanceof MemoryFile)) throw new Error(`missing file ${path}`);
    entry.lastModified = lastModified;
  }
}

class MemoryDirectory {
  readonly children = new Map<string, MemoryDirectory | MemoryFile>();
  failNextRemoval = false;
  readonly kind = "directory" as const;

  constructor(readonly name: string) {}

  async *entries(): AsyncIterableIterator<
    [string, MemoryDirectory | MemoryFileHandle]
  > {
    for (const [name, entry] of this.entriesMap()) {
      yield [
        name,
        entry instanceof MemoryDirectory ? entry : new MemoryFileHandle(entry),
      ];
    }
  }

  private entriesMap(): IterableIterator<
    [string, MemoryDirectory | MemoryFile]
  > {
    return this.children.entries();
  }

  async getDirectoryHandle(
    name: string,
    options?: { readonly create?: boolean },
  ): Promise<MemoryDirectory> {
    const existing = this.children.get(name);
    if (existing instanceof MemoryDirectory) return existing;
    if (existing !== undefined || options?.create !== true) throw notFound();
    const directory = new MemoryDirectory(name);
    this.children.set(name, directory);
    return directory;
  }

  async getFileHandle(
    name: string,
    options?: { readonly create?: boolean },
  ): Promise<MemoryFileHandle> {
    const existing = this.children.get(name);
    if (existing instanceof MemoryFile) return new MemoryFileHandle(existing);
    if (existing !== undefined || options?.create !== true) throw notFound();
    const file = new MemoryFile(name);
    this.children.set(name, file);
    return new MemoryFileHandle(file);
  }

  async removeEntry(
    name: string,
    options?: { readonly recursive?: boolean },
  ): Promise<void> {
    if (this.failNextRemoval) {
      this.failNextRemoval = false;
      throw new DOMException("injected removal failure", "UnknownError");
    }
    const existing = this.children.get(name);
    if (existing === undefined) throw notFound();
    if (
      existing instanceof MemoryDirectory &&
      existing.children.size > 0 &&
      options?.recursive !== true
    ) {
      throw new DOMException("directory is not empty", "InvalidModificationError");
    }
    this.children.delete(name);
  }
}

class MemoryFile {
  bytes = new Uint8Array();
  lastModified = Date.now();

  readonly kind = "file" as const;

  constructor(readonly name: string) {}
}

class MemoryFileHandle {
  readonly kind = "file" as const;

  constructor(private readonly file: MemoryFile) {}

  async createSyncAccessHandle(): Promise<MemorySyncHandle> {
    return new MemorySyncHandle(this.file);
  }

  async getFile(): Promise<File> {
    return new File([this.file.bytes.slice()], this.file.name, {
      type: this.file.name.endsWith(".wav") ? "audio/wav" : "",
      lastModified: this.file.lastModified,
    });
  }
}

class MemorySyncHandle {
  private closed = false;

  constructor(private readonly file: MemoryFile) {}

  write(buffer: ArrayBufferView, options: { readonly at: number }): number {
    this.requireOpen();
    const bytes = new Uint8Array(buffer.buffer, buffer.byteOffset, buffer.byteLength);
    const end = options.at + bytes.byteLength;
    if (end > this.file.bytes.byteLength) {
      const grown = new Uint8Array(end);
      grown.set(this.file.bytes);
      this.file.bytes = grown;
    }
    this.file.bytes.set(bytes, options.at);
    return bytes.byteLength;
  }

  read(buffer: ArrayBufferView, options: { readonly at: number }): number {
    this.requireOpen();
    const target = new Uint8Array(buffer.buffer, buffer.byteOffset, buffer.byteLength);
    const available = Math.max(0, this.file.bytes.byteLength - options.at);
    const count = Math.min(target.byteLength, available);
    target.set(this.file.bytes.subarray(options.at, options.at + count));
    return count;
  }

  truncate(size: number): void {
    this.requireOpen();
    const resized = new Uint8Array(size);
    resized.set(this.file.bytes.subarray(0, size));
    this.file.bytes = resized;
  }

  flush(): void {
    this.requireOpen();
  }

  close(): void {
    this.closed = true;
  }

  private requireOpen(): void {
    if (this.closed) throw new Error("closed");
  }
}

function notFound(): DOMException {
  return new DOMException("missing", "NotFoundError");
}
