import { describe, expect, it, vi } from "vitest";

import {
  AceOpt0081RepresentativeDitOwner,
  encodeOpt0081RepresentativeArenaCopy,
  prefillOpt0081RepresentativeTap,
  type AceOpt0081RepresentativeDitOwnerOptions,
  type AceOpt0081RepresentativeDitSession,
} from "../src/webgpu/dit-backend.js";
import { AceCorrectnessDitRuntime } from "../src/webgpu/ace-dit.js";
import { AceDitResidentModel } from "../src/webgpu/dit-graph.js";
import type { AceGpuTensorPhase } from "../src/model/gpu-tensors.js";

describe("OPT-0081 representative owner lifecycle", () => {
  it("copies every exact arena slot into a disjoint GPU-only snapshot", () => {
    const copies: unknown[][] = [];
    const finish = vi.fn().mockReturnValue({ command: true });
    const encoder = {
      copyBufferToBuffer: vi.fn((...args: unknown[]) => copies.push(args)),
      finish,
    };
    const device = {
      createCommandEncoder: vi.fn().mockReturnValue(encoder),
    } as unknown as GPUDevice;
    const sourceBuffers = [{}, {}] as unknown as readonly GPUBuffer[];
    const destinationBuffers = [{}, {}] as unknown as readonly GPUBuffer[];
    const source = {
      buffers: sourceBuffers,
      bufferByteLengths: [256, 1_024],
      byteLength: 1_280,
    };
    const destination = {
      buffers: destinationBuffers,
      bufferByteLengths: [256, 1_024],
      byteLength: 1_280,
    };

    expect(encodeOpt0081RepresentativeArenaCopy(
      device,
      source,
      destination,
      "capture-control",
    )).toEqual({ command: true });
    expect(copies).toEqual([
      [sourceBuffers[0], 0, destinationBuffers[0], 0, 256],
      [sourceBuffers[1], 0, destinationBuffers[1], 0, 1_024],
    ]);
    expect(finish).toHaveBeenCalledWith({
      label: "ace-opt-0081-correctness-arena-capture-control-command",
    });
  });

  it("rejects a partial or shape-mismatched arena snapshot", () => {
    const device = {
      createCommandEncoder: vi.fn(),
    } as unknown as GPUDevice;
    const buffer = {} as GPUBuffer;
    expect(() => encodeOpt0081RepresentativeArenaCopy(
      device,
      { buffers: [buffer], bufferByteLengths: [256], byteLength: 256 },
      { buffers: [buffer], bufferByteLengths: [128], byteLength: 256 },
      "restore-control",
    )).toThrow(/slot topology changed/);
  });

  it.each([
    {
      storage: "u32" as const,
      bodyBytes: 1_048_576 + 12,
      expectedElementSizes: [262_144, 3],
      expectedConstructor: Uint32Array,
    },
    {
      storage: "u16" as const,
      bodyBytes: 1_048_576 + 4,
      expectedElementSizes: [524_288, 2],
      expectedConstructor: Uint16Array,
    },
  ])(
    "passes typed-array element counts for $storage prefill including the final partial chunk",
    ({ storage, bodyBytes, expectedElementSizes, expectedConstructor }) => {
      const calls: Array<Readonly<{
        bufferOffset: number;
        data: Uint16Array | Uint32Array;
        dataOffset: number | undefined;
        size: number | undefined;
      }>> = [];
      const queue = {
        writeBuffer(
          _buffer: unknown,
          bufferOffset: number,
          data: unknown,
          dataOffset?: number,
          size?: number,
        ) {
          calls.push(Object.freeze({
            bufferOffset,
            data: data as Uint16Array | Uint32Array,
            dataOffset,
            size,
          }));
        },
      } as unknown as GPUQueue;
      const binding = {
        buffer: { size: 256 + bodyBytes } as GPUBuffer,
        offset: 256,
        size: bodyBytes,
      } satisfies GPUBufferBinding;

      prefillOpt0081RepresentativeTap(queue, binding, storage);

      expect(calls).toHaveLength(2);
      expect(calls.map(({ bufferOffset }) => bufferOffset)).toEqual([
        256,
        256 + 1_048_576,
      ]);
      expect(calls.map(({ dataOffset }) => dataOffset)).toEqual([0, 0]);
      expect(calls.map(({ size }) => size)).toEqual(expectedElementSizes);
      for (const call of calls) {
        expect(call.data).toBeInstanceOf(expectedConstructor);
        expect(call.size).toBeLessThanOrEqual(call.data.length);
      }
    },
  );

  it("records exactly one rejection for one post-destroy operation", async () => {
    const owner = Object.create(
      AceOpt0081RepresentativeDitOwner.prototype,
    ) as AceOpt0081RepresentativeDitSession;
    const internal = owner as unknown as {
      state: "destroyed";
      postDestroyRejectedOperationCount: number;
    };
    internal.state = "destroyed";
    internal.postDestroyRejectedOperationCount = 0;

    await expect(owner.runPrecompute()).rejects.toMatchObject({
      name: "AbortError",
    });
    expect(internal.postDestroyRejectedOperationCount).toBe(1);
  });

  it("destroys and accounts a fresh target runtime when compilation fails", async () => {
    vi.stubGlobal("GPUBufferUsage", {
      STORAGE: 1 << 0,
      COPY_SRC: 1 << 1,
      COPY_DST: 1 << 2,
    });
    const compileFailure = new Error("target compilation failed");
    const runtimeDestroy = vi.fn();
    const createRuntime = vi.spyOn(AceCorrectnessDitRuntime, "create")
      .mockReturnValue({
        createTimestepDispatch: vi.fn().mockRejectedValue(compileFailure),
        destroy: runtimeDestroy,
      } as unknown as AceCorrectnessDitRuntime);
    const guardedBufferDestroy = vi.fn();
    const guardedBuffer = {
      size: 516,
      destroy: guardedBufferDestroy,
    } as unknown as GPUBuffer;
    const device = {
      limits: {
        maxBufferSize: 1_024,
        maxStorageBufferBindingSize: 1_024,
      },
      pushErrorScope: vi.fn(),
      popErrorScope: vi.fn().mockResolvedValue(null),
      createBuffer: vi.fn().mockReturnValue(guardedBuffer),
    } as unknown as GPUDevice;
    const baseBuffer = { size: 4 } as GPUBuffer;
    const baseBinding = { buffer: baseBuffer, size: 4 } satisfies GPUBufferBinding;
    const arm = {
      id: "A",
      denseInputStorageProfile: undefined,
      bindings: {
        hidden: [baseBinding, baseBinding],
        controls: {
          timesteps: [baseBinding],
          relativeTimestepZero: baseBinding,
        },
        timestepScratch: baseBinding,
        timestepEmbedding: baseBinding,
        timestepProjection: baseBinding,
      },
    };
    const owner = Object.create(
      AceOpt0081RepresentativeDitOwner.prototype,
    ) as AceOpt0081RepresentativeDitSession;
    const internal = owner as unknown as {
      device: GPUDevice;
      model: { weights: { timestep: object } };
      plan: { batch: number };
      selection: {
        gemmConfiguration: object;
        denseGemmConfiguration: object;
        attentionConfiguration: object;
      };
      checkpointTarget: undefined;
      createdGraphBufferCount: number;
      destroyedGraphBufferCount: number;
      liveGraphBufferCount: number;
      liveGraphByteCount: number;
      maximumLiveGraphByteCount: number;
      runtimeOwnerCount: number;
      destroyedRuntimeOwnerCount: number;
      liveCorrectnessRuntimeCount: number;
      maximumLiveCorrectnessRuntimeCount: number;
      prepareCheckpointTarget(
        arm: unknown,
        layer: 0,
        tap: "layerOutput",
        signal: AbortSignal,
      ): Promise<unknown>;
    };
    Object.assign(internal, {
      device,
      model: { weights: { timestep: {} } },
      plan: { batch: 1 },
      selection: {
        gemmConfiguration: {},
        denseGemmConfiguration: {},
        attentionConfiguration: {},
      },
      checkpointTarget: undefined,
      createdGraphBufferCount: 0,
      destroyedGraphBufferCount: 0,
      liveGraphBufferCount: 0,
      liveGraphByteCount: 0,
      maximumLiveGraphByteCount: 0,
      runtimeOwnerCount: 2,
      destroyedRuntimeOwnerCount: 0,
      liveCorrectnessRuntimeCount: 0,
      maximumLiveCorrectnessRuntimeCount: 0,
    });

    try {
      await expect(internal.prepareCheckpointTarget(
        arm,
        0,
        "layerOutput",
        new AbortController().signal,
      )).rejects.toBe(compileFailure);
      expect(createRuntime).toHaveBeenCalledOnce();
      expect(runtimeDestroy).toHaveBeenCalledOnce();
      expect(guardedBufferDestroy).toHaveBeenCalledOnce();
      expect(internal.runtimeOwnerCount).toBe(3);
      expect(internal.destroyedRuntimeOwnerCount).toBe(1);
      expect(internal.liveCorrectnessRuntimeCount).toBe(0);
      expect(internal.maximumLiveCorrectnessRuntimeCount).toBe(1);
      expect(internal.createdGraphBufferCount).toBe(1);
      expect(internal.destroyedGraphBufferCount).toBe(1);
      expect(internal.liveGraphBufferCount).toBe(0);
      expect(internal.liveGraphByteCount).toBe(0);
      expect(internal.checkpointTarget).toBeUndefined();
    } finally {
      createRuntime.mockRestore();
      vi.unstubAllGlobals();
    }
  });

  it("destroys transferred model ownership when setup evidence is invalid", async () => {
    const modelDestroy = vi.fn();
    const takeMixed = vi.spyOn(AceDitResidentModel, "takeMixed")
      .mockReturnValue({ destroy: modelDestroy } as unknown as AceDitResidentModel);
    const phase = (shardCount: number): AceGpuTensorPhase => ({
      packageManifest: {
        tensors: Object.fromEntries(Array.from(
          { length: shardCount },
          (_, index) => [`tensor-${index}`, { phase: "dit", shard: `s-${index}` }],
        )),
      },
      destroy: vi.fn(),
    } as unknown as AceGpuTensorPhase);
    const options = {
      ownedDitWeights: phase(2),
      ownedDitDenseWeights: phase(48),
      ditDenseRuntimeProfile: "opt-0009-fp16-fp32-v1",
      verifiedSetupFailureCleanup: {
        schema: "ace-opt-0081-representative-setup-cleanup-v1",
        createdGraphBufferCount: 0,
      },
    } as unknown as AceOpt0081RepresentativeDitOwnerOptions;

    await expect(AceOpt0081RepresentativeDitOwner.create(options))
      .rejects.toThrow(/setup cleanup evidence is invalid/);
    expect(modelDestroy).toHaveBeenCalledOnce();
    takeMixed.mockRestore();
  });
});
