import { describe, expect, it, vi } from "vitest";

import {
  ACE_OPT_0006_QUANTA_PER_COMMAND_BUFFER_CANDIDATES,
  encodeAceOpt0006QuantumBatch,
  planAceOpt0006QuantumBatches,
  runAceOpt0006QuantumBatches,
  type AceOpt0006EncodableQuantum,
} from "../benchmark/opt-0006-vae-command-buffer-coalescing.js";

describe("OPT-0006 bounded VAE command-buffer coalescing", () => {
  it("plans exact bounded batches including a partial final batch", () => {
    expect(ACE_OPT_0006_QUANTA_PER_COMMAND_BUFFER_CANDIDATES).toEqual(
      [1, 2, 4, 8, 16],
    );
    expect(planAceOpt0006QuantumBatches(40, 16)).toEqual([
      { index: 0, firstQuantumIndex: 0, quantumCount: 16, final: false },
      { index: 1, firstQuantumIndex: 16, quantumCount: 16, final: false },
      { index: 2, firstQuantumIndex: 32, quantumCount: 8, final: true },
    ]);
    expect(() => planAceOpt0006QuantumBatches(0, 1)).toThrow(/positive/);
    expect(() => planAceOpt0006QuantumBatches(1, 0)).toThrow(/positive/);
  });

  it("encodes every quantum once in its own FIFO compute pass", () => {
    const { device, events } = fakeDevice();
    const quanta = fakeQuanta(5, events);
    encodeAceOpt0006QuantumBatch(
      device,
      quanta,
      planAceOpt0006QuantumBatches(5, 4)[0]!,
    );
    expect(events).toEqual([
      "begin:0", "encode:0", "end:0",
      "begin:1", "encode:1", "end:1",
      "begin:2", "encode:2", "end:2",
      "begin:3", "encode:3", "end:3",
      "finish",
    ]);
  });

  it("submits, drains, idles, and reports only complete drained batches", async () => {
    const { device, events } = fakeDevice();
    const queue = fakeQueue(events);
    const progress: unknown[] = [];
    const result = await runAceOpt0006QuantumBatches({
      device,
      queue,
      quanta: fakeQuanta(5, events),
      maximumQuantaPerCommandBuffer: 2,
      signal: new AbortController().signal,
      finalCommandBufferRemains: true,
      yieldQueueIdle: async () => { events.push("idle"); },
      onProgress: (value) => progress.push(value),
    });
    expect(result).toEqual({
      completedQuanta: 5,
      commandBuffersSubmitted: 3,
      queueDrains: 3,
      cooperativeIdleMs: 3,
      batchCount: 3,
    });
    expect(progress).toEqual([
      expect.objectContaining({ completedQuanta: 1, queueDrains: 1 }),
      expect.objectContaining({ completedQuanta: 2, queueDrains: 1 }),
      expect.objectContaining({ completedQuanta: 3, queueDrains: 2 }),
      expect.objectContaining({ completedQuanta: 4, queueDrains: 2 }),
      expect.objectContaining({ completedQuanta: 5, queueDrains: 3 }),
    ]);
    expect(events.filter((event) => event === "submit")).toHaveLength(3);
    expect(events.filter((event) => event === "drain")).toHaveLength(3);
    expect(events.filter((event) => event === "idle")).toHaveLength(3);
  });

  it("preserves size-one topology and omits only a truly final idle", async () => {
    const { device, events } = fakeDevice();
    const result = await runAceOpt0006QuantumBatches({
      device,
      queue: fakeQueue(events),
      quanta: fakeQuanta(3, events),
      maximumQuantaPerCommandBuffer: 1,
      signal: new AbortController().signal,
      yieldQueueIdle: async () => { events.push("idle"); },
    });
    expect(result).toMatchObject({
      commandBuffersSubmitted: 3,
      queueDrains: 3,
      cooperativeIdleMs: 2,
    });
  });

  it("drains and completes idle before surfacing progress failure", async () => {
    const { device, events } = fakeDevice();
    const failure = new Error("progress failed");
    await expect(runAceOpt0006QuantumBatches({
      device,
      queue: fakeQueue(events),
      quanta: fakeQuanta(2, events),
      maximumQuantaPerCommandBuffer: 1,
      signal: new AbortController().signal,
      yieldQueueIdle: async () => { events.push("idle"); },
      onProgress: () => { throw failure; },
    })).rejects.toBe(failure);
    expect(events).toContain("drain");
    expect(events).toContain("idle");
    expect(events.filter((event) => event === "submit")).toHaveLength(1);
  });

  it("aborts after a drained batch and never submits the next batch", async () => {
    const { device, events } = fakeDevice();
    const controller = new AbortController();
    const failure = new DOMException("cancelled", "AbortError");
    await expect(runAceOpt0006QuantumBatches({
      device,
      queue: fakeQueue(events),
      quanta: fakeQuanta(5, events),
      maximumQuantaPerCommandBuffer: 2,
      signal: controller.signal,
      yieldQueueIdle: async () => { events.push("idle"); },
      onProgress: () => controller.abort(failure),
    })).rejects.toBe(failure);
    expect(events.filter((event) => event === "submit")).toHaveLength(1);
    expect(events.filter((event) => event === "drain")).toHaveLength(1);
    expect(events.filter((event) => event === "idle")).toHaveLength(1);
  });

  it("does not submit after encode failure", async () => {
    const { device, events } = fakeDevice();
    const failure = new Error("encode failed");
    const quanta = fakeQuanta(3, events);
    quanta[1] = { id: "failure", encode: () => { throw failure; } };
    await expect(runAceOpt0006QuantumBatches({
      device,
      queue: fakeQueue(events),
      quanta,
      maximumQuantaPerCommandBuffer: 3,
      signal: new AbortController().signal,
    })).rejects.toBe(failure);
    expect(events).not.toContain("submit");
  });

  it("stops encoding the batch when cancellation is raised between passes", async () => {
    const { device, events } = fakeDevice();
    const controller = new AbortController();
    const failure = new DOMException("cancelled while encoding", "AbortError");
    const quanta = fakeQuanta(3, events);
    quanta[0] = {
      id: "abort",
      encode: () => {
        events.push("encode:abort");
        controller.abort(failure);
      },
    };
    await expect(runAceOpt0006QuantumBatches({
      device,
      queue: fakeQueue(events),
      quanta,
      maximumQuantaPerCommandBuffer: 3,
      signal: controller.signal,
    })).rejects.toBe(failure);
    expect(events).not.toContain("encode:1");
    expect(events).not.toContain("finish");
    expect(events).not.toContain("submit");
  });

  it("does not drain or progress after submit throws", async () => {
    const { device, events } = fakeDevice();
    const failure = new Error("submit failed");
    const queue = fakeQueue(events);
    queue.submit = vi.fn(() => { throw failure; });
    const progress = vi.fn();
    await expect(runAceOpt0006QuantumBatches({
      device,
      queue,
      quanta: fakeQuanta(2, events),
      maximumQuantaPerCommandBuffer: 2,
      signal: new AbortController().signal,
      onProgress: progress,
    })).rejects.toBe(failure);
    expect(queue.onSubmittedWorkDone).not.toHaveBeenCalled();
    expect(progress).not.toHaveBeenCalled();
  });

  it("does not progress, idle, or submit later work after a drain rejects", async () => {
    const { device, events } = fakeDevice();
    const failure = new Error("drain failed");
    const queue = fakeQueue(events);
    queue.onSubmittedWorkDone = vi.fn(async () => { throw failure; });
    const progress = vi.fn();
    const idle = vi.fn(async () => undefined);
    await expect(runAceOpt0006QuantumBatches({
      device,
      queue,
      quanta: fakeQuanta(5, events),
      maximumQuantaPerCommandBuffer: 2,
      signal: new AbortController().signal,
      yieldQueueIdle: idle,
      onProgress: progress,
    })).rejects.toBe(failure);
    expect(queue.submit).toHaveBeenCalledOnce();
    expect(progress).not.toHaveBeenCalled();
    expect(idle).not.toHaveBeenCalled();
  });
});

function fakeQuanta(
  count: number,
  events: string[],
): AceOpt0006EncodableQuantum[] {
  return Array.from({ length: count }, (_, index) => ({
    id: String(index),
    encode: () => { events.push(`encode:${index}`); },
  }));
}

function fakeDevice(): {
  device: Pick<GPUDevice, "createCommandEncoder">;
  events: string[];
} {
  const events: string[] = [];
  let passIndex = 0;
  return {
    events,
    device: {
      createCommandEncoder: vi.fn(() => ({
        beginComputePass: vi.fn(() => {
          const index = passIndex++;
          events.push(`begin:${index}`);
          return { end: vi.fn(() => events.push(`end:${index}`)) };
        }),
        finish: vi.fn(() => {
          events.push("finish");
          return {} as GPUCommandBuffer;
        }),
      })),
    } as unknown as Pick<GPUDevice, "createCommandEncoder">,
  };
}

function fakeQueue(events: string[]): Pick<GPUQueue, "submit" | "onSubmittedWorkDone"> {
  return {
    submit: vi.fn(() => {
      events.push("submit");
      return undefined;
    }),
    onSubmittedWorkDone: vi.fn(async () => {
      events.push("drain");
      return undefined;
    }),
  };
}
