import { beforeAll, describe, expect, it, vi } from "vitest";

import {
  ACE_GPU_UPLOAD_CHUNK_BYTES,
  isAceAuthenticatedGpuSource,
  markAceAuthenticatedGpuSource,
  uploadAcePackageFileToGpu,
  type AceGpuUploadTrace,
} from "../src/model/gpu-upload.js";
import { aceSha256Hex } from "../src/model/sha256.js";

beforeAll(() => {
  Object.defineProperty(globalThis, "GPUBufferUsage", {
    configurable: true,
    value: { STORAGE: 1, COPY_DST: 2 },
  });
});

describe("bounded authenticated GPU package upload", () => {
  it("rehashes fragmented cached bytes and uploads only aligned bounded copies", async () => {
    const bytes = Uint8Array.from({ length: 20 }, (_, index) => index + 1);
    const fixture = fakeDevice(bytes.byteLength);
    const progress: number[] = [];
    const buffer = await uploadAcePackageFileToGpu(
      fixture.device,
      record(bytes),
      fragmentedBlob(bytes, [1, 2, 7, 10]),
      { onProgress: (event) => progress.push(event.uploadedBytes) },
    );
    expect(buffer).toBe(fixture.buffer);
    expect(fixture.uploaded).toEqual(bytes);
    expect(fixture.writeSizes.every((size) => size <= ACE_GPU_UPLOAD_CHUNK_BYTES)).toBe(
      true,
    );
    expect(fixture.writeSizes.every((size) => size % 4 === 0)).toBe(true);
    expect(progress.at(-1)).toBe(bytes.byteLength);
    expect(fixture.buffer.destroy).not.toHaveBeenCalled();
  });

  it("destroys a tentative buffer when the cached body fails its manifest digest", async () => {
    const expected = Uint8Array.of(1, 2, 3, 4);
    const actual = Uint8Array.of(1, 2, 3, 5);
    const fixture = fakeDevice(expected.byteLength);
    await expect(
      uploadAcePackageFileToGpu(
        fixture.device,
        record(expected),
        fragmentedBlob(actual, [2, 2]),
      ),
    ).rejects.toThrow(/SHA-256 mismatch/);
    expect(fixture.buffer.destroy).toHaveBeenCalledOnce();
  });

  it("reuses only an exact authenticated File identity and keeps records fail-closed", async () => {
    const bytes = Uint8Array.of(1, 2, 3, 4, 5, 6, 7, 8);
    const expected = record(bytes);
    const source = new File([bytes], expected.name);
    expect(isAceAuthenticatedGpuSource(source, expected)).toBe(false);

    markAceAuthenticatedGpuSource(source, expected);
    expect(isAceAuthenticatedGpuSource(source, expected)).toBe(true);
    expect(isAceAuthenticatedGpuSource(new File([bytes], expected.name), expected)).toBe(
      false,
    );
    expect(
      isAceAuthenticatedGpuSource(source, {
        ...expected,
        sha256: "0".repeat(64),
      }),
    ).toBe(false);

    const fixture = fakeDevice(bytes.byteLength);
    await uploadAcePackageFileToGpu(fixture.device, expected, source);
    expect(fixture.uploaded).toEqual(bytes);

    expect(() =>
      markAceAuthenticatedGpuSource(
        new File([bytes.subarray(0, bytes.byteLength - 1)], expected.name),
        expected,
      ),
    ).toThrow(/does not match/);
    expect(() =>
      markAceAuthenticatedGpuSource(source, {
        ...expected,
        sha256: expected.sha256.toUpperCase(),
      }),
    ).toThrow(/does not match/);
  });

  it("attributes exact-File proof reuse separately from bounded Blob hash/copy work", async () => {
    const bytes = Uint8Array.of(1, 2, 3, 4, 5, 6, 7, 8);
    const expected = record(bytes);
    const source = new File([bytes], expected.name);
    markAceAuthenticatedGpuSource(source, expected);
    const authenticatedFixture = fakeDevice(bytes.byteLength);
    const authenticatedTrace: AceGpuUploadTrace[] = [];
    let authenticatedClock = 0;
    await uploadAcePackageFileToGpu(
      authenticatedFixture.device,
      expected,
      source,
      {
        maximumQueuedBytes: 4,
        yieldQueueIdle: async () => undefined,
        now: () => authenticatedClock++,
        onTrace: (trace) => authenticatedTrace.push(trace),
      },
    );
    expect(authenticatedTrace).toEqual([
      expect.objectContaining({
        schema: "ace-gpu-upload-capture-v1",
        authentication: "exact-immutable-file-proof-reused",
        sourceKind: "file",
        redundantHashPerformed: false,
        uploadedBytes: bytes.byteLength,
        queueDrainCount: 2,
        queueEmptyGapCount: 1,
        maximumOwnedCopyBytes: 0,
      }),
    ]);
    expect(authenticatedTrace[0]!.timing).toMatchObject({
      ownedCopyMs: 0,
      incrementalHashMs: 0,
    });

    const blobFixture = fakeDevice(bytes.byteLength);
    const blobTrace: AceGpuUploadTrace[] = [];
    let blobClock = 0;
    await uploadAcePackageFileToGpu(
      blobFixture.device,
      expected,
      fragmentedBlob(bytes, [3, 5]),
      {
        maximumQueuedBytes: 4,
        yieldQueueIdle: async () => undefined,
        now: () => blobClock++,
        onTrace: (trace) => blobTrace.push(trace),
      },
    );
    expect(blobTrace).toEqual([
      expect.objectContaining({
        authentication: "ordinary-blob-stream-hashed",
        sourceKind: "blob",
        redundantHashPerformed: false,
        maximumOwnedCopyBytes: 5,
        queueDrainCount: 2,
        queueEmptyGapCount: 1,
      }),
    ]);
    expect(blobTrace[0]!.timing.ownedCopyMs).toBeGreaterThan(0);
    expect(blobTrace[0]!.timing.incrementalHashMs).toBeGreaterThan(0);
  });

  it("surfaces scoped WebGPU validation and OOM errors and destroys the buffer", async () => {
    const bytes = Uint8Array.of(9, 8, 7, 6);
    for (const errors of [
      [gpuError("bad binding"), null, null],
      [null, gpuError("out of memory"), null],
      [null, null, gpuError("device internal failure")],
    ] as const) {
      const fixture = fakeDevice(bytes.byteLength, errors);
      await expect(
        uploadAcePackageFileToGpu(
          fixture.device,
          record(bytes),
          fragmentedBlob(bytes, [4]),
        ),
      ).rejects.toThrow(
        errors[0] !== null
          ? /validation/
          : errors[1] !== null
            ? /GPU memory/
            : /internally/,
      );
      expect(fixture.buffer.destroy).toHaveBeenCalledOnce();
    }
  });

  it("rejects non-runtime files and device-limit overflow before allocating", async () => {
    const bytes = Uint8Array.of(1, 2, 3, 4);
    const fixture = fakeDevice(2);
    await expect(
      uploadAcePackageFileToGpu(
        fixture.device,
        { ...record(bytes), kind: "license" },
        fragmentedBlob(bytes, [4]),
      ),
    ).rejects.toThrow(/cannot consume/);
    await expect(
      uploadAcePackageFileToGpu(
        fixture.device,
        record(bytes),
        fragmentedBlob(bytes, [4]),
      ),
    ).rejects.toThrow(/storage-buffer limits/);
    expect(fixture.createBuffer).not.toHaveBeenCalled();
  });

  it("drains at the queued-byte bound and leaves a real gap before another write", async () => {
      const maximumQueuedBytes = 16;
      const byteLength = maximumQueuedBytes + 4;
      const bytes = new Uint8Array(byteLength);
      const events: string[] = [];
      let resolveDrain: (() => void) | undefined;
      let drainCount = 0;
      const fixture = fakeDevice(byteLength, undefined, {
        onWrite: () => events.push("write"),
        onDrain: () => {
          drainCount += 1;
          events.push("drain-start");
          if (drainCount > 1) {
            events.push("drain-done");
            return Promise.resolve();
          }
          return new Promise<void>((resolve) => {
            resolveDrain = () => {
              events.push("drain-done");
              resolve();
            };
          });
        },
      });
      const upload = uploadAcePackageFileToGpu(
        fixture.device,
        record(bytes),
        fragmentedBlob(bytes, [4, 4, 4, 4, 4]),
        {
          maximumQueuedBytes,
          yieldQueueIdle: async () => {
            events.push("idle");
          },
        },
      );

      await vi.waitFor(() => expect(resolveDrain).toBeTypeOf("function"));
      expect(fixture.writeSizes.reduce((sum, size) => sum + size, 0)).toBe(
        maximumQueuedBytes,
      );
      expect(events.at(-1)).toBe("drain-start");
      resolveDrain!();
      await Promise.resolve();
      expect(fixture.writeSizes.reduce((sum, size) => sum + size, 0)).toBe(
        maximumQueuedBytes,
      );
      await upload;
      expect(fixture.uploaded).toEqual(bytes);
      expect(events).toEqual([
        ...Array.from({ length: 4 }, () => "write"),
        "drain-start",
        "drain-done",
        "idle",
        "write",
        "drain-start",
        "drain-done",
      ]);
      expect(fixture.onSubmittedWorkDone).toHaveBeenCalledTimes(2);
  });

  it("honors cancellation after a backpressure drain without submitting more bytes", async () => {
    const maximumQueuedBytes = 16;
    const byteLength = maximumQueuedBytes + 4;
    const bytes = new Uint8Array(byteLength);
    const controller = new AbortController();
    const fixture = fakeDevice(byteLength, undefined, {
      onDrain: async () => {
        controller.abort(new Error("stop upload"));
      },
    });

    await expect(
      uploadAcePackageFileToGpu(
        fixture.device,
        record(bytes),
        fragmentedBlob(bytes, [4, 4, 4, 4, 4]),
        { signal: controller.signal, maximumQueuedBytes },
      ),
    ).rejects.toThrow("stop upload");
    expect(fixture.writeSizes.reduce((sum, size) => sum + size, 0)).toBe(
      maximumQueuedBytes,
    );
    expect(fixture.onSubmittedWorkDone).toHaveBeenCalledOnce();
    expect(fixture.buffer.destroy).toHaveBeenCalledOnce();
  });
});

function record(bytes: Uint8Array) {
  return {
    name: "weights/test.bin",
    byteLength: bytes.byteLength,
    sha256: aceSha256Hex(bytes),
    kind: "weights" as const,
  };
}

function fragmentedBlob(bytes: Uint8Array, sizes: readonly number[]): Blob {
  if (sizes.reduce((total, value) => total + value, 0) !== bytes.byteLength) {
    throw new Error("test fragment sizes do not cover the payload");
  }
  return {
    size: bytes.byteLength,
    stream() {
      return new ReadableStream<Uint8Array>({
        start(controller) {
          let offset = 0;
          for (const size of sizes) {
            controller.enqueue(bytes.slice(offset, offset + size));
            offset += size;
          }
          controller.close();
        },
      });
    },
  } as Blob;
}

function fakeDevice(
  limit: number,
  errors: readonly [GPUError | null, GPUError | null, GPUError | null] = [
    null,
    null,
    null,
  ],
  hooks: {
    readonly onWrite?: () => void;
    readonly onDrain?: () => Promise<void>;
  } = {},
) {
  const uploaded = new Uint8Array(Math.max(limit, 0));
  const writeSizes: number[] = [];
  const buffer = {
    size: limit,
    destroy: vi.fn(),
  } as unknown as GPUBuffer & { readonly destroy: ReturnType<typeof vi.fn> };
  const createBuffer = vi.fn(() => buffer);
  const popValues = [...errors];
  const device = {
    limits: {
      maxBufferSize: limit,
      maxStorageBufferBindingSize: limit,
    },
    pushErrorScope: vi.fn(),
    popErrorScope: vi.fn(async () => popValues.shift() ?? null),
    createBuffer,
    queue: {
      writeBuffer: vi.fn(
        (_target: GPUBuffer, offset: number, data: AllowSharedBufferSource) => {
          const view = ArrayBuffer.isView(data)
            ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
            : new Uint8Array(data);
          uploaded.set(view, offset);
          writeSizes.push(view.byteLength);
          hooks.onWrite?.();
        },
      ),
      onSubmittedWorkDone: vi.fn(hooks.onDrain ?? (async () => {})),
    },
  } as unknown as GPUDevice;
  return {
    device,
    buffer,
    createBuffer,
    uploaded,
    writeSizes,
    onSubmittedWorkDone: device.queue.onSubmittedWorkDone as ReturnType<typeof vi.fn>,
  };
}

function gpuError(message: string): GPUError {
  return { message } as GPUError;
}
