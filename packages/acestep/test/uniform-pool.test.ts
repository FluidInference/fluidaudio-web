import { beforeAll, describe, expect, it, vi } from "vitest";

import { AceUniformPool } from "../src/webgpu/uniform-pool.js";

beforeAll(() => {
  Object.defineProperty(globalThis, "GPUBufferUsage", {
    configurable: true,
    value: { UNIFORM: 1, COPY_DST: 2 },
  });
});

describe("ACE uniform pool", () => {
  it("copies and zero-pads aligned slots without recycling before queue drain", async () => {
    const fixture = fakeDevice();
    const pool = await AceUniformPool.create(fixture.device, 1_024);
    pool.beginQuantum();
    const callerOwned = Uint8Array.of(1, 2, 3, 4);
    const first = pool.write("first", callerOwned);
    callerOwned.fill(99);
    const second = pool.write("second", Uint32Array.of(7));
    expect(first).toMatchObject({ byteOffset: 0, byteLength: 16 });
    expect(second).toMatchObject({ byteOffset: 256, byteLength: 16 });
    expect(fixture.writes[0]).toEqual({
      offset: 0,
      bytes: Uint8Array.of(1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    });
    expect(() => pool.beginQuantum()).toThrow(/expected idle/);
    expect(() => pool.beginQuantum()).toThrow(/expected idle/);

    let drained = false;
    const reclaim = pool.submitQuantum({} as GPUCommandBuffer).then(() => {
      drained = true;
    });
    expect(fixture.events.slice(-2)).toEqual(["submit", "drain"]);
    await expect(pool.submitQuantum({} as GPUCommandBuffer)).rejects.toThrow(
      /expected encoding/,
    );
    await Promise.resolve();
    expect(drained).toBe(false);
    fixture.resolveDrain();
    await reclaim;
    pool.beginQuantum();
    expect(pool.write("reused", Uint32Array.of(1)).byteOffset).toBe(0);
  });

  it("rejects overflow and destroys exactly once", async () => {
    const fixture = fakeDevice();
    const pool = await AceUniformPool.create(fixture.device, 256);
    pool.beginQuantum();
    pool.write("only-slot", Uint32Array.of(1));
    expect(() => pool.write("overflow", Uint32Array.of(2))).toThrow(/exhausted/);
    pool.destroy();
    pool.destroy();
    expect(fixture.buffer.destroy).toHaveBeenCalledOnce();
    expect(() => pool.write("dead", Uint32Array.of(3))).toThrow(/destroyed/);
  });

  it("validates capacity and per-binding limits", async () => {
    const fixture = fakeDevice({ maxUniformBufferBindingSize: 16 });
    await expect(AceUniformPool.create(fixture.device, 255)).rejects.toThrow(
      /capacity/,
    );
    const pool = await AceUniformPool.create(fixture.device, 256);
    pool.beginQuantum();
    expect(() => pool.write("too-wide", new Uint8Array(17))).toThrow(
      /maxUniformBufferBindingSize/,
    );
  });

  it("fails closed on asynchronous buffer allocation errors", async () => {
    const fixture = fakeDevice({}, [null, gpuError("allocation failed"), null]);
    await expect(AceUniformPool.create(fixture.device, 256)).rejects.toThrow(
      /out-of-memory.*allocation failed/,
    );
    expect(fixture.buffer.destroy).toHaveBeenCalledOnce();
  });

  it("stays sealed after submit or drain failure", async () => {
    const fixture = fakeDevice();
    const pool = await AceUniformPool.create(fixture.device, 256);
    pool.beginQuantum();
    pool.write("uniform", Uint32Array.of(1));
    fixture.submit.mockImplementationOnce(() => {
      throw new Error("submit failed");
    });
    await expect(pool.submitQuantum({} as GPUCommandBuffer)).rejects.toThrow(
      /submit failed/,
    );
    expect(() => pool.beginQuantum()).toThrow(/submitted/);
  });
});

function fakeDevice(
  overrides: Partial<Pick<GPUSupportedLimits, "maxUniformBufferBindingSize">> = {},
  scopeErrors: readonly [GPUError | null, GPUError | null, GPUError | null] = [
    null,
    null,
    null,
  ],
) {
  const writes: Array<{ offset: number; bytes: Uint8Array }> = [];
  const events: string[] = [];
  const buffer = { destroy: vi.fn() } as unknown as GPUBuffer & {
    destroy: ReturnType<typeof vi.fn>;
  };
  let resolveDrain = () => {};
  const drain = new Promise<void>((resolve) => {
    resolveDrain = resolve;
  });
  const device = {
    limits: {
      minUniformBufferOffsetAlignment: 256,
      maxUniformBufferBindingSize: 64 * 1024,
      maxBufferSize: 4_096,
      ...overrides,
    },
    pushErrorScope: vi.fn(),
    popErrorScope: vi.fn(async () => scopeResults.shift() ?? null),
    createBuffer: vi.fn(() => buffer),
    queue: {
      writeBuffer: vi.fn(
        (_buffer: GPUBuffer, offset: number, data: AllowSharedBufferSource) => {
          const view = ArrayBuffer.isView(data)
            ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
            : new Uint8Array(data);
          writes.push({ offset, bytes: Uint8Array.from(view) });
        },
      ),
      submit: vi.fn(() => {
        events.push("submit");
      }),
      onSubmittedWorkDone: vi.fn(async () => {
        events.push("drain");
        await drain;
      }),
    },
  } as unknown as GPUDevice;
  const scopeResults = [...scopeErrors];
  return {
    device,
    buffer,
    writes,
    resolveDrain,
    events,
    submit: device.queue.submit as ReturnType<typeof vi.fn>,
  };
}

function gpuError(message: string): GPUError {
  return { message } as GPUError;
}
