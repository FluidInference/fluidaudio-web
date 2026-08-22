import { beforeAll, describe, expect, it, vi } from "vitest";

import { AceGpuArena, planAceLifetimeArena } from "../src/webgpu/arena.js";

beforeAll(() => {
  Object.defineProperty(globalThis, "GPUBufferUsage", {
    configurable: true,
    value: { STORAGE: 1, COPY_SRC: 2, COPY_DST: 4 },
  });
});

describe("ACE lifetime arena planning", () => {
  it("aliases only non-overlapping inclusive lifetimes", () => {
    const plan = planAceLifetimeArena([
      { label: "input", byteLength: 1_000, firstQuantum: 0, lastQuantum: 2 },
      { label: "scratch", byteLength: 500, firstQuantum: 1, lastQuantum: 1 },
      { label: "output", byteLength: 900, firstQuantum: 3, lastQuantum: 4 },
      { label: "tail", byteLength: 300, firstQuantum: 2, lastQuantum: 3 },
    ]);
    expect(plan.slices.input).toMatchObject({ byteOffset: 0, byteLength: 1_000 });
    expect(plan.slices.scratch).toMatchObject({ byteOffset: 1_024, byteLength: 500 });
    expect(plan.slices.output).toMatchObject({ byteOffset: 0, byteLength: 900 });
    // `tail` overlaps input at quantum 2 and output at quantum 3.
    expect(plan.slices.tail).toMatchObject({ byteOffset: 1_024, byteLength: 300 });
    expect(plan.byteLength).toBe(1_536);
  });

  it("rejects duplicate labels and malformed lifetimes", () => {
    expect(() =>
      planAceLifetimeArena([
        { label: "same", byteLength: 4, firstQuantum: 0, lastQuantum: 0 },
        { label: "same", byteLength: 4, firstQuantum: 1, lastQuantum: 1 },
      ]),
    ).toThrow(/Duplicate/);
    expect(() =>
      planAceLifetimeArena([
        { label: "backwards", byteLength: 4, firstQuantum: 2, lastQuantum: 1 },
      ]),
    ).toThrow(/Invalid/);
    expect(() =>
      planAceLifetimeArena([
        {
          label: "unsafe-alignment",
          byteLength: Number.MAX_SAFE_INTEGER,
          firstQuantum: 0,
          lastQuantum: 0,
        },
      ]),
    ).toThrow(/alignment exceeds safe integers/);
  });
});

describe("ACE GPU arena", () => {
  it("routes bounded aligned slices and rejects writable alias hazards", async () => {
    const buffers = [fakeBuffer(2_048), fakeBuffer(1_024)];
    const device = fakeDevice(buffers, 4_096);
    const arena = await AceGpuArena.create(device, [
      { label: "activations", byteLength: 2_048 },
      { label: "scratch", byteLength: 1_024 },
    ]);
    const a = arena.slice("a", 0, 0, 512);
    const b = arena.slice("b", 0, 512, 512);
    const overlapping = arena.slice("overlap", 0, 256, 512);
    const other = arena.slice("other", 1, 0, 512);
    expect(arena.binding(other)).toEqual({
      buffer: buffers[1],
      offset: 0,
      size: 512,
    });
    expect(() => arena.assertNoWritableOverlap([a, b], [other])).not.toThrow();
    expect(() => arena.assertNoWritableOverlap([a], [overlapping])).toThrow(
      /read\/write.*overlap/,
    );
    expect(() => arena.assertNoWritableOverlap([a, overlapping])).toThrow(
      /writable.*overlap/,
    );
    arena.destroy();
    arena.destroy();
    expect(buffers[0]!.destroy).toHaveBeenCalledOnce();
    expect(buffers[1]!.destroy).toHaveBeenCalledOnce();
    expect(() => arena.binding(a)).toThrow(/destroyed/);
  });

  it("destroys earlier allocations if a later arena buffer fails", async () => {
    const first = fakeBuffer(1_024);
    const device = {
      limits: { maxBufferSize: 4_096, maxStorageBufferBindingSize: 4_096 },
      pushErrorScope: vi.fn(),
      popErrorScope: vi.fn(async () => null),
      createBuffer: vi.fn()
        .mockReturnValueOnce(first)
        .mockImplementationOnce(() => {
          throw new Error("allocation failed");
        }),
    } as unknown as GPUDevice;
    await expect(
      AceGpuArena.create(device, [
        { label: "first", byteLength: 1_024 },
        { label: "second", byteLength: 1_024 },
      ]),
    ).rejects.toThrow(/allocation failed/);
    expect(first.destroy).toHaveBeenCalledOnce();
  });

  it("rejects device-timeline allocation errors and destroys invalid objects", async () => {
    const invalid = fakeBuffer(1_024);
    const device = fakeDevice(
      [invalid],
      4_096,
      4_096,
      [gpuError("invalid descriptor"), null, null],
    );
    await expect(
      AceGpuArena.create(device, [{ label: "invalid", byteLength: 1_024 }]),
    ).rejects.toThrow(/validation.*invalid descriptor/);
    expect(invalid.destroy).toHaveBeenCalledOnce();
  });

  it("drains all allocation scopes after a synchronous create failure", async () => {
    const first = fakeBuffer(1_024);
    const pops = vi.fn(async () => null);
    const device = {
      limits: { maxBufferSize: 4_096, maxStorageBufferBindingSize: 4_096 },
      pushErrorScope: vi.fn(),
      popErrorScope: pops,
      createBuffer: vi.fn()
        .mockReturnValueOnce(first)
        .mockImplementationOnce(() => {
          throw new RangeError("synchronous create failure");
        }),
    } as unknown as GPUDevice;
    await expect(
      AceGpuArena.create(device, [
        { label: "first", byteLength: 1_024 },
        { label: "second", byteLength: 1_024 },
      ]),
    ).rejects.toThrow(/synchronous create failure/);
    expect(pops).toHaveBeenCalledTimes(3);
    expect(first.destroy).toHaveBeenCalledOnce();
  });

  it("allows a large backing arena while bounding each storage binding", async () => {
    const device = fakeDevice([fakeBuffer(2_048)], 4_096, 1_024);
    const arena = await AceGpuArena.create(device, [
      { label: "multi-slice", byteLength: 2_048 },
    ]);
    expect(() => arena.slice("allowed", 0, 1_024, 1_024)).not.toThrow();
    expect(() => arena.slice("oversized-binding", 0, 0, 2_048)).toThrow(
      /Invalid ACE arena slice/,
    );
  });
});

function fakeBuffer(size: number) {
  return {
    size,
    destroy: vi.fn(),
  } as unknown as GPUBuffer & { readonly destroy: ReturnType<typeof vi.fn> };
}

function fakeDevice(
  buffers: readonly GPUBuffer[],
  maxBufferSize: number,
  maxStorageBufferBindingSize = maxBufferSize,
  scopeErrors: readonly [GPUError | null, GPUError | null, GPUError | null] = [
    null,
    null,
    null,
  ],
): GPUDevice {
  let index = 0;
  const errors = [...scopeErrors];
  return {
    limits: { maxBufferSize, maxStorageBufferBindingSize },
    pushErrorScope: vi.fn(),
    popErrorScope: vi.fn(async () => errors.shift() ?? null),
    createBuffer: vi.fn(() => buffers[index++]!),
  } as unknown as GPUDevice;
}

function gpuError(message: string): GPUError {
  return { message } as GPUError;
}
