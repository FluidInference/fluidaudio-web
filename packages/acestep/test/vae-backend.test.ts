import { describe, expect, it, vi } from "vitest";

import {
  ACE_VAE_DECODER_QUANTA_PER_COMMAND_BUFFER,
  AceVaeBackendDeviceLostError,
  AceVaeChunkGpuBackend,
  planAceVaeChunkGpuBackendMemory,
  type AceVaePreparedChunkGpuResources,
} from "../src/webgpu/vae-backend.js";
import {
  planAceVaeChunkedDecode,
  type AceVaeChunkedDecodePlan,
} from "../src/webgpu/vae-chunks.js";
import {
  ACE_VAE_DEFAULT_QUANTUM_WORK_POLICY,
  AceCorrectnessVaeDecoderRuntime,
  planAceVaeDecoder,
  planAceVaeDecoderQuanta,
  type AceVaeDecoderDispatch,
  type AceVaeDecoderQuantum,
  type AceVaeDecoderConfig,
} from "../src/webgpu/vae-decoder.js";
import type { AceGpuTensorPhase } from "../src/model/gpu-tensors.js";
import { AceGpuArena } from "../src/webgpu/arena.js";

Object.defineProperty(globalThis, "GPUMapMode", {
  configurable: true,
  value: Object.freeze({ READ: 1 }),
});
Object.defineProperty(globalThis, "GPUBufferUsage", {
  configurable: true,
  value: Object.freeze({ COPY_DST: 1, MAP_READ: 2 }),
});

const TOY_CONFIG: AceVaeDecoderConfig = Object.freeze({
  id: "vae-backend-toy",
  decoderInputChannels: 2,
  decoderChannels: 1,
  audioChannels: 2,
  channelMultiples: Object.freeze([1]),
  downsamplingRatios: Object.freeze([2]),
  sampleRateHz: 48_000,
});

describe("bounded VAE chunk GPU backend", () => {
  it("accounts the complete production window allocation without opaque driver memory", () => {
    const plan = planAceVaeChunkedDecode(750);
    const residentWeightBytes = 337_583_104;
    const memory = planAceVaeChunkGpuBackendMemory(
      plan,
      residentWeightBytes,
      256,
      {
        "vae.decoder.block.0.conv_t1.weight": [
          { partStart: 0, partEnd: 614 },
          { partStart: 614, partEnd: 1_024 },
        ],
      },
    );

    expect(memory).toMatchObject({
      residentWeightBytes,
      inputBufferBytes: 65_536,
      outputBufferBytes: 3_932_160,
      workspaceBufferBytes: 251_658_240,
      workspaceBufferCount: 3,
      arenaBytes: 758_972_416,
      readbackBufferBytes: 3_932_160,
      latentSnapshotBytes: 192_000,
      maximumReturnedWindowBytes: 3_932_160,
      boundedCpuBytes: 4_124_160,
    });
    expect(memory.rangeControlBytes).toBeGreaterThan(0);
    expect(memory.rangeControlBytes).toBe(2_491_184);
    expect(memory.uniqueDecoderInputFrames).toEqual([174, 192, 256]);
    expect(memory.accountedGpuBytes).toBe(
      residentWeightBytes + memory.arenaBytes + memory.rangeControlBytes +
        memory.readbackBufferBytes,
    );
    expect(memory.accountedGpuBytes).toBe(1_102_978_864);
    expect(memory.maximumCommandBuffersPerWindow).toBe(
      memory.maximumDecoderCommandBuffersPerWindow + 1,
    );
    expect(memory.decoderQuantaPerCommandBuffer).toBe(8);
    expect(memory.decoderQuantaPerCommandBuffer).toBe(
      ACE_VAE_DECODER_QUANTA_PER_COMMAND_BUFFER,
    );
    expect(memory.maximumDecoderQuantaPerWindow).toBe(3_942);
    expect(memory.maximumDecoderCommandBuffersPerWindow).toBe(493);
    expect(memory.maximumCommandBuffersPerWindow).toBe(494);

    const legacy = planAceVaeChunkGpuBackendMemory(
      plan,
      residentWeightBytes,
      256,
      {
        "vae.decoder.block.0.conv_t1.weight": [
          { partStart: 0, partEnd: 614 },
          { partStart: 614, partEnd: 1_024 },
        ],
      },
      {
        maximumConvolutionMultiplyAccumulates: 234_881_024,
        maximumOutputElements: 32_768,
      },
    );
    expect(legacy.rangeControlBytes).toBe(39_012_400);
    expect(legacy.accountedGpuBytes).toBe(1_139_500_080);
    expect(legacy.maximumDecoderQuantaPerWindow).toBe(62_622);
    expect(legacy.maximumDecoderCommandBuffersPerWindow).toBe(7_828);
    expect(legacy.maximumCommandBuffersPerWindow).toBe(7_829);
  });

  it("batches a partial decoder tail while preserving FIFO quantum progress", async () => {
    const fixture = preparedFixture(10);
    const originalLatents = Float32Array.from(fixture.resources.finalLatents);
    const backend = AceVaeChunkGpuBackend.fromPreparedResources(
      fixture.resources,
    );
    fixture.resources.finalLatents.fill(99);

    const actual = await backend.decodeWindow(fixture.plan.windows[0]!);
    expect(actual).toEqual(fixture.expectedOutput);
    expect(fixture.uploads).toEqual([originalLatents]);
    expect(fixture.maximumOutstanding()).toBe(1);
    expect(fixture.submissions).toHaveLength(3);
    expect(fixture.submissions.every((submission) => submission.length === 1))
      .toBe(true);
    expect(fixture.idle).toHaveBeenCalledTimes(2);
    expect(fixture.encodeEvents).toEqual(
      Array.from({ length: 10 }, (_, index) => `quantum-${index}`),
    );
    expect(fixture.passEvents).toEqual(
      Array.from({ length: 10 }, (_, index) => [
        `begin:${index}`,
        `end:${index}`,
      ]).flat(),
    );
    expect(fixture.progress.map((event) => ({
      quantum: event.completedDecoderQuanta,
      completed: event.completedCommandBuffers,
      drains: event.queueDrains,
      idleMs: event.cooperativeIdleMs,
      stage: event.stage,
    }))).toEqual([
      ...Array.from({ length: 8 }, (_, index) => ({
        quantum: index + 1,
        completed: 1,
        drains: 1,
        idleMs: 1,
        stage: "decoder",
      })),
      { quantum: 9, completed: 2, drains: 2, idleMs: 2, stage: "decoder" },
      { quantum: 10, completed: 2, drains: 2, idleMs: 2, stage: "decoder" },
      { quantum: 10, completed: 3, drains: 3, idleMs: 2, stage: "readback" },
    ]);
    expect(fixture.readback.mapAsync).toHaveBeenCalledWith(
      GPUMapMode.READ,
      0,
      fixture.expectedOutput.byteLength,
    );
    expect(fixture.readback.unmap).toHaveBeenCalledTimes(1);

    await backend.destroy();
    await backend.destroy();
    expect(fixture.destroy).toHaveBeenCalledTimes(1);
    await expect(backend.decodeWindow(fixture.plan.windows[0]!)).rejects
      .toMatchObject({ name: "InvalidStateError" });
  });

  it("finishes the real queue-empty interval before honoring cancellation", async () => {
    const idle = Promise.withResolvers<void>();
    const fixture = preparedFixture(2, {
      yieldQueueIdle: vi.fn(() => idle.promise),
    });
    const backend = AceVaeChunkGpuBackend.fromPreparedResources(
      fixture.resources,
    );
    const controller = new AbortController();
    const decoding = backend.decodeWindow(fixture.plan.windows[0]!, controller.signal);
    const rejection = expect(decoding).rejects.toMatchObject({ name: "AbortError" });
    await waitUntil(() => fixture.idle.mock.calls.length === 1);
    controller.abort();
    await flushMicrotasks();
    expect(fixture.submissions).toHaveLength(1);
    idle.resolve();
    await rejection;
    expect(fixture.submissions).toHaveLength(1);
    expect(fixture.readback.mapAsync).not.toHaveBeenCalled();
    await backend.destroy();
    expect(fixture.destroy).toHaveBeenCalledTimes(1);
  });

  it("creates the next decoder batch only after the preceding batch drains", async () => {
    const firstDrain = Promise.withResolvers<undefined>();
    const fixture = preparedFixture(10, { drain: firstDrain });
    const backend = AceVaeChunkGpuBackend.fromPreparedResources(
      fixture.resources,
    );
    const decoding = backend.decodeWindow(fixture.plan.windows[0]!);
    await waitUntil(() => fixture.submissions.length === 1);
    expect(fixture.encodeEvents).toEqual(
      Array.from({ length: 8 }, (_, index) => `quantum-${index}`),
    );
    expect(fixture.passEvents).toHaveLength(16);
    firstDrain.resolve(undefined);
    await decoding;
    expect(fixture.encodeEvents).toHaveLength(10);
    expect(fixture.submissions).toHaveLength(3);
    await backend.destroy();
  });

  it("checks cancellation between quantum passes and submits no partial batch", async () => {
    const controller = new AbortController();
    const failure = new DOMException("cancelled during encode", "AbortError");
    const fixture = preparedFixture(3, {
      onQuantumEncode: (index) => {
        if (index === 0) controller.abort(failure);
      },
    });
    const backend = AceVaeChunkGpuBackend.fromPreparedResources(
      fixture.resources,
    );
    await expect(backend.decodeWindow(
      fixture.plan.windows[0]!,
      controller.signal,
    )).rejects.toBe(failure);
    expect(fixture.encodeEvents).toEqual(["quantum-0"]);
    expect(fixture.submissions).toHaveLength(0);
    expect(fixture.idle).not.toHaveBeenCalled();
    expect(fixture.readback.mapAsync).not.toHaveBeenCalled();
    await backend.destroy();
  });

  it("completes the batch idle before surfacing per-quantum progress failure", async () => {
    const idle = Promise.withResolvers<void>();
    const failure = new Error("progress failed");
    const fixture = preparedFixture(3, {
      yieldQueueIdle: vi.fn(() => idle.promise),
      onProgress: (event) => {
        if (event.completedDecoderQuanta === 2) throw failure;
      },
    });
    const backend = AceVaeChunkGpuBackend.fromPreparedResources(
      fixture.resources,
    );
    const decoding = backend.decodeWindow(fixture.plan.windows[0]!);
    const rejection = expect(decoding).rejects.toBe(failure);
    await waitUntil(() => fixture.idle.mock.calls.length === 1);
    expect(fixture.submissions).toHaveLength(1);
    expect(fixture.progress.map((event) => event.completedDecoderQuanta))
      .toEqual([1, 2]);
    idle.resolve();
    await rejection;
    expect(fixture.submissions).toHaveLength(1);
    expect(fixture.readback.mapAsync).not.toHaveBeenCalled();
    await backend.destroy();
    expect(fixture.destroy).toHaveBeenCalledTimes(1);
  });

  it("supports a validated internal batch-size override with matching memory", async () => {
    const fixture = preparedFixture(5, {
      decoderQuantaPerCommandBuffer: 2,
    });
    expect(fixture.resources.memory.decoderQuantaPerCommandBuffer).toBe(2);
    const backend = AceVaeChunkGpuBackend.fromPreparedResources(
      fixture.resources,
    );
    await backend.decodeWindow(fixture.plan.windows[0]!);
    expect(fixture.submissions).toHaveLength(4);
    expect(fixture.idle).toHaveBeenCalledTimes(3);
    expect(fixture.progress.at(-1)).toMatchObject({
      completedDecoderQuanta: 5,
      totalDecoderQuanta: 5,
      completedCommandBuffers: 4,
      totalCommandBuffers: 4,
      queueDrains: 4,
      cooperativeIdleMs: 3,
      stage: "readback",
    });
    await backend.destroy();
  });

  it("rejects inconsistent prepared batch and memory-plan bounds", () => {
    const batchMismatch = preparedFixture(1);
    expect(() => AceVaeChunkGpuBackend.fromPreparedResources({
      ...batchMismatch.resources,
      decoderQuantaPerCommandBuffer: 2,
    })).toThrow(/does not match decoder command batching/);
    expect(batchMismatch.destroy).toHaveBeenCalledTimes(1);

    const formulaMismatch = preparedFixture(1);
    expect(() => AceVaeChunkGpuBackend.fromPreparedResources({
      ...formulaMismatch.resources,
      memory: {
        ...formulaMismatch.resources.memory,
        maximumDecoderCommandBuffersPerWindow:
          formulaMismatch.resources.memory
            .maximumDecoderCommandBuffersPerWindow + 1,
      },
    })).toThrow(/inconsistent command-buffer bounds/);
    expect(formulaMismatch.destroy).toHaveBeenCalledTimes(1);
  });

  it("stops after a decoder drain failure and remains safely destroyable", async () => {
    const drain = Promise.withResolvers<undefined>();
    const failure = new Error("queue drain failed");
    const fixture = preparedFixture(10, { drain });
    const backend = AceVaeChunkGpuBackend.fromPreparedResources(
      fixture.resources,
    );
    const decoding = backend.decodeWindow(fixture.plan.windows[0]!);
    await waitUntil(() => fixture.submissions.length === 1);
    drain.reject(failure);
    await expect(decoding).rejects.toBe(failure);
    expect(fixture.submissions).toHaveLength(1);
    expect(fixture.idle).not.toHaveBeenCalled();
    expect(fixture.progress).toHaveLength(0);
    expect(fixture.readback.mapAsync).not.toHaveBeenCalled();
    await backend.destroy();
    expect(fixture.destroy).toHaveBeenCalledTimes(1);
  });

  it("aborts an active graph on device loss, drains it, then tears down once", async () => {
    const drain = Promise.withResolvers<undefined>();
    const fixture = preparedFixture(2, { drain });
    const backend = AceVaeChunkGpuBackend.fromPreparedResources(
      fixture.resources,
    );
    const decoding = backend.decodeWindow(fixture.plan.windows[0]!);
    await waitUntil(() => fixture.submissions.length === 1);
    fixture.deviceLost.resolve({
      reason: "unknown",
      message: "test device vanished",
    } as GPUDeviceLostInfo);
    await flushMicrotasks();
    expect(fixture.destroy).not.toHaveBeenCalled();
    drain.resolve(undefined);
    await expect(decoding).rejects.toBeInstanceOf(AceVaeBackendDeviceLostError);
    await backend.destroy();
    expect(fixture.destroy).toHaveBeenCalledTimes(1);
    expect(fixture.submissions).toHaveLength(1);
  });

  it("treats the factory signal as backend-lifetime cancellation", async () => {
    const controller = new AbortController();
    const fixture = preparedFixture(1, { signal: controller.signal });
    const backend = AceVaeChunkGpuBackend.fromPreparedResources(
      fixture.resources,
    );
    controller.abort(new DOMException("job cancelled", "AbortError"));
    await backend.destroy();
    expect(fixture.destroy).toHaveBeenCalledTimes(1);
    await expect(backend.decodeWindow(fixture.plan.windows[0]!)).rejects
      .toMatchObject({ name: "InvalidStateError" });
  });

  it("fails closed and releases transferred weights before allocation", async () => {
    const plan = toyPlan();
    const destroy = vi.fn();
    const wrongPhase = {
      phases: ["dit"],
      residentBytes: 1,
      destroy,
    } as unknown as AceGpuTensorPhase;
    await expect(AceVaeChunkGpuBackend.create({
      device: {} as GPUDevice,
      plan,
      finalLatents: new Float32Array(6),
      ownedVaeWeights: wrongPhase,
    })).rejects.toThrow(/exclusively resident vae phase/);
    expect(destroy).toHaveBeenCalledTimes(1);
  });

  it("snapshots a custom work policy before gated allocation", async () => {
    const plan = toyPlan();
    const policy = {
      maximumConvolutionMultiplyAccumulates: 1_000_000,
      maximumOutputElements: 2,
    };
    const expectedMemory = planAceVaeChunkGpuBackendMemory(
      plan,
      1_024,
      256,
      {},
      policy,
    );
    const expectedDispatch = planAceVaeDecoderQuanta(
      planAceVaeDecoder(3, TOY_CONFIG),
      {},
      policy,
    );
    const allocationEntered = Promise.withResolvers<void>();
    const releaseAllocation = Promise.withResolvers<void>();
    const arenaDestroy = vi.fn();
    const arenaCreate = vi.spyOn(AceGpuArena, "create").mockImplementation(
      async (_device, plans) => {
        const buffers = plans.map((entry) => ({
          size: entry.byteLength,
          destroy: vi.fn(),
        } as unknown as GPUBuffer));
        allocationEntered.resolve();
        await releaseAllocation.promise;
        return {
          slice(label: string, bufferIndex: number, byteOffset: number, byteLength: number) {
            return { label, bufferIndex, byteOffset, byteLength };
          },
          binding(slice: { readonly bufferIndex: number; readonly byteOffset: number; readonly byteLength: number }) {
            return {
              buffer: buffers[slice.bufferIndex]!,
              offset: slice.byteOffset,
              size: slice.byteLength,
            };
          },
          destroy: arenaDestroy,
        } as unknown as AceGpuArena;
      },
    );
    const observedPolicies: unknown[] = [];
    const decoderDestroy = vi.fn();
    const decoderCreate = vi.spyOn(
      AceCorrectnessVaeDecoderRuntime,
      "create",
    ).mockReturnValue({
      async createDecoderDispatch(
        label: string,
        inputFrames: number,
        _bindings: unknown,
        config: AceVaeDecoderConfig,
        batch: number,
        options: { readonly quantumWorkPolicy?: typeof policy },
      ) {
        observedPolicies.push(options.quantumWorkPolicy);
        const graph = planAceVaeDecoder(inputFrames, config, batch);
        const cooperativePlan = planAceVaeDecoderQuanta(
          graph,
          {},
          options.quantumWorkPolicy,
        );
        return {
          label,
          plan: graph,
          cooperativePlan,
          quanta: cooperativePlan.quanta,
          primitiveCount: cooperativePlan.primitiveDispatchCount,
        } as unknown as AceVaeDecoderDispatch;
      },
      destroy: decoderDestroy,
    } as unknown as AceCorrectnessVaeDecoderRuntime);
    const weights = fakeToyVaePhase(plan);
    const device = fakeFactoryDevice();
    let backend: AceVaeChunkGpuBackend | undefined;
    try {
      const creating = AceVaeChunkGpuBackend.create({
        device,
        plan,
        finalLatents: new Float32Array(6),
        ownedVaeWeights: weights.value,
        quantumWorkPolicy: policy,
      });
      await allocationEntered.promise;
      policy.maximumConvolutionMultiplyAccumulates = 1;
      policy.maximumOutputElements = 100;
      releaseAllocation.resolve();
      backend = await creating;

      expect(backend.memory).toEqual(expectedMemory);
      expect(observedPolicies).toHaveLength(1);
      expect(observedPolicies[0]).toEqual({
        maximumConvolutionMultiplyAccumulates: 1_000_000,
        maximumOutputElements: 2,
      });
      expect(Object.isFrozen(observedPolicies[0])).toBe(true);
      const compiled = (observedPolicies[0] as typeof policy);
      expect(planAceVaeDecoderQuanta(
        planAceVaeDecoder(3, TOY_CONFIG),
        {},
        compiled,
      ).quantumCount).toBe(expectedDispatch.quantumCount);
    } finally {
      await backend?.destroy();
      arenaCreate.mockRestore();
      decoderCreate.mockRestore();
    }
    expect(weights.destroy).toHaveBeenCalledTimes(1);
    expect(arenaDestroy).toHaveBeenCalledTimes(1);
    expect(decoderDestroy).toHaveBeenCalledTimes(1);
  });

  it("rejects altered windows before uploading or submitting", async () => {
    const fixture = preparedFixture(1);
    const backend = AceVaeChunkGpuBackend.fromPreparedResources(
      fixture.resources,
    );
    const canonical = fixture.plan.windows[0]!;
    await expect(backend.decodeWindow({
      ...canonical,
      windowEndLatentFrame: canonical.windowEndLatentFrame - 1,
    })).rejects.toThrow(/does not match the fixed plan/);
    expect(fixture.uploads).toHaveLength(0);
    expect(fixture.submissions).toHaveLength(0);
    await backend.destroy();
  });
});

interface FixtureOptions {
  readonly yieldQueueIdle?: ReturnType<typeof vi.fn<() => Promise<void>>>;
  readonly drain?: PromiseWithResolvers<undefined>;
  readonly signal?: AbortSignal;
  readonly decoderQuantaPerCommandBuffer?: number;
  readonly onQuantumEncode?: (index: number) => void;
  readonly onProgress?: NonNullable<
    AceVaePreparedChunkGpuResources["onProgress"]
  >;
}

function preparedFixture(
  quantumCount: number,
  options: FixtureOptions = {},
): Readonly<{
  plan: AceVaeChunkedDecodePlan;
  resources: AceVaePreparedChunkGpuResources;
  expectedOutput: Float32Array;
  readback: ReturnType<typeof fakeReadback>;
  uploads: Float32Array[];
  submissions: GPUCommandBuffer[][];
  encodeEvents: string[];
  passEvents: string[];
  progress: Array<{
    readonly completedDecoderQuanta: number;
    readonly completedCommandBuffers: number;
    readonly queueDrains: number;
    readonly cooperativeIdleMs: number;
    readonly stage: string;
  }>;
  idle: ReturnType<typeof vi.fn<() => Promise<void>>>;
  destroy: ReturnType<typeof vi.fn>;
  deviceLost: PromiseWithResolvers<GPUDeviceLostInfo>;
  maximumOutstanding(): number;
}> {
  const plan = toyPlan();
  const expectedOutput = Float32Array.from(
    { length: plan.maximumDecodedInterleavedElements },
    (_, index) => Math.fround(index / 16 - 0.25),
  );
  const readback = fakeReadback(expectedOutput);
  const uploads: Float32Array[] = [];
  const submissions: GPUCommandBuffer[][] = [];
  const encodeEvents: string[] = [];
  const passEvents: string[] = [];
  const progress: Array<{
    readonly completedDecoderQuanta: number;
    readonly completedCommandBuffers: number;
    readonly queueDrains: number;
    readonly cooperativeIdleMs: number;
    readonly stage: string;
  }> = [];
  const deviceLost = Promise.withResolvers<GPUDeviceLostInfo>();
  let outstanding = 0;
  let maximumOutstanding = 0;
  const queue = {
    writeBuffer(
      _buffer: GPUBuffer,
      _bufferOffset: number,
      data: GPUAllowSharedBufferSource,
    ): void {
      const view = ArrayBuffer.isView(data)
        ? new Float32Array(data.buffer, data.byteOffset, data.byteLength / 4)
        : new Float32Array(data);
      uploads.push(Float32Array.from(view));
    },
    submit(commandBuffers: Iterable<GPUCommandBuffer>): void {
      const batch = [...commandBuffers];
      submissions.push(batch);
      outstanding += 1;
      maximumOutstanding = Math.max(maximumOutstanding, outstanding);
    },
    async onSubmittedWorkDone(): Promise<undefined> {
      if (options.drain !== undefined) await options.drain.promise;
      outstanding -= 1;
      return undefined;
    },
  } as Pick<GPUQueue, "writeBuffer" | "submit" | "onSubmittedWorkDone">;
  const device = {
    queue,
    lost: deviceLost.promise,
    createCommandEncoder({ label }: GPUCommandEncoderDescriptor = {}) {
      return {
        beginComputePass() {
          const index = passEvents.length / 2;
          passEvents.push(`begin:${index}`);
          return {
            end: vi.fn(() => passEvents.push(`end:${index}`)),
          } as unknown as GPUComputePassEncoder;
        },
        copyBufferToBuffer: vi.fn(),
        finish() {
          return { label } as GPUCommandBuffer;
        },
      } as unknown as GPUCommandEncoder;
    },
  } as unknown as GPUDevice;
  const graph = planAceVaeDecoder(3, TOY_CONFIG);
  const quanta = Array.from(
    { length: quantumCount },
    (_, index) => fakeQuantum(index, encodeEvents, options.onQuantumEncode),
  );
  const dispatch: AceVaeDecoderDispatch = Object.freeze({
    label: "toy-dispatch",
    plan: graph,
    cooperativePlan: Object.freeze({
      quantumWorkPolicy: ACE_VAE_DEFAULT_QUANTUM_WORK_POLICY,
      quantumCount: quanta.length,
      primitiveDispatchCount: quanta.length,
      quanta,
    }),
    quanta: Object.freeze(quanta),
    primitiveCount: quanta.length,
  });
  const inputBuffer = { size: 256 } as GPUBuffer;
  const outputBuffer = { size: 256 } as GPUBuffer;
  const idle = options.yieldQueueIdle ?? vi.fn(async () => undefined);
  const destroy = vi.fn();
  const memory = planAceVaeChunkGpuBackendMemory(
    plan,
    1_024,
    256,
    {},
    ACE_VAE_DEFAULT_QUANTUM_WORK_POLICY,
    options.decoderQuantaPerCommandBuffer ??
      ACE_VAE_DECODER_QUANTA_PER_COMMAND_BUFFER,
  );
  const resources: AceVaePreparedChunkGpuResources = {
    device,
    plan,
    finalLatents: Float32Array.from([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
    input: { buffer: inputBuffer, offset: 0, size: 256 },
    output: { buffer: outputBuffer, offset: 0, size: 256 },
    readback: readback.value,
    decoderDispatches: new Map([[3, dispatch]]),
    memory,
    ...(options.signal === undefined ? {} : { signal: options.signal }),
    ...(options.decoderQuantaPerCommandBuffer === undefined
      ? {}
      : {
        decoderQuantaPerCommandBuffer:
          options.decoderQuantaPerCommandBuffer,
      }),
    yieldQueueIdle: idle,
    onProgress: (event) => {
      progress.push(event);
      options.onProgress?.(event);
    },
    destroy,
  };
  return Object.freeze({
    plan,
    resources,
    expectedOutput,
    readback,
    uploads,
    submissions,
    encodeEvents,
    passEvents,
    progress,
    idle,
    destroy,
    deviceLost,
    maximumOutstanding: () => maximumOutstanding,
  });
}

function toyPlan(): AceVaeChunkedDecodePlan {
  return planAceVaeChunkedDecode(3, {
    chunkFrames: 3,
    overlapFrames: 0,
    config: TOY_CONFIG,
  });
}

function fakeToyVaePhase(plan: AceVaeChunkedDecodePlan): Readonly<{
  value: AceGpuTensorPhase;
  destroy: ReturnType<typeof vi.fn>;
}> {
  const destroy = vi.fn();
  const weightBuffer = { size: 1_000_000_000 } as GPUBuffer;
  const transposeRows = new Map(
    plan.decoderWorkspacePlan.operations
      .filter((operation) => operation.kind === "conv-transpose1d")
      .map((operation) => [operation.weight, operation.shape.outputChannels]),
  );
  const logicalTensor = (name: string) => {
    const partEnd = transposeRows.get(name) ?? 1;
    return {
      logicalTensor: name,
      logicalShape: [partEnd],
      parts: [{
        tensorName: name,
        tensor: {
          logicalTensor: name,
          logicalShape: [partEnd],
          phase: "vae",
          partAxis: 0,
          partStart: 0,
          partEnd,
        },
        binding: { buffer: weightBuffer, offset: 0, size: weightBuffer.size },
      }],
    };
  };
  return Object.freeze({
    value: {
      phases: Object.freeze(["vae"]),
      residentBytes: 1_024,
      logicalTensor,
      destroy,
    } as unknown as AceGpuTensorPhase,
    destroy,
  });
}

function fakeFactoryDevice(): GPUDevice {
  const lost = Promise.withResolvers<GPUDeviceLostInfo>();
  return {
    limits: {
      maxBufferSize: 1_000_000_000,
      maxStorageBufferBindingSize: 1_000_000_000,
      minUniformBufferOffsetAlignment: 256,
    },
    lost: lost.promise,
    queue: { writeBuffer: vi.fn() },
    pushErrorScope: vi.fn(),
    popErrorScope: vi.fn(async () => null),
    createBuffer(descriptor: GPUBufferDescriptor) {
      return {
        size: descriptor.size,
        destroy: vi.fn(),
      } as unknown as GPUBuffer;
    },
  } as unknown as GPUDevice;
}

function fakeQuantum(
  index: number,
  events: string[],
  onEncode?: (index: number) => void,
): AceVaeDecoderQuantum {
  return Object.freeze({
    index,
    id: `quantum-${index}`,
    operationIndex: index,
    operationLabel: `operation-${index}`,
    operationKind: "add",
    logicalOutputBase: index,
    logicalOutputCount: 1,
    estimatedMaximumMultiplyAccumulates: 0,
    primitives: Object.freeze([Object.freeze({
      controlRecordIndex: index,
      firstOutputChannel: 0,
      outputChannels: 1,
      outputBase: index,
      outputCount: 1,
    })]),
    primitiveCount: 1,
    encode(): void {
      events.push(`quantum-${index}`);
      onEncode?.(index);
    },
  });
}

function fakeReadback(output: Float32Array): Readonly<{
  value: GPUBuffer;
  mapAsync: ReturnType<typeof vi.fn>;
  getMappedRange: ReturnType<typeof vi.fn>;
  unmap: ReturnType<typeof vi.fn>;
}> {
  const data = new ArrayBuffer(output.byteLength);
  new Float32Array(data).set(output);
  const mapAsync = vi.fn(async () => undefined);
  const getMappedRange = vi.fn(() => data);
  const unmap = vi.fn();
  return Object.freeze({
    value: {
      size: output.byteLength,
      mapAsync,
      getMappedRange,
      unmap,
    } as unknown as GPUBuffer,
    mapAsync,
    getMappedRange,
    unmap,
  });
}

async function flushMicrotasks(): Promise<void> {
  for (let turn = 0; turn < 8; turn += 1) await Promise.resolve();
}

async function waitUntil(predicate: () => boolean): Promise<void> {
  for (let turn = 0; turn < 100; turn += 1) {
    if (predicate()) return;
    await Promise.resolve();
  }
  throw new Error("test condition did not become true");
}
