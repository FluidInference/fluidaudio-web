import { describe, expect, it, vi } from "vitest";

import {
  ACE_COOPERATIVE_GPU_IDLE_MILLISECONDS,
  AceCooperativeGpuScheduler,
  AceFifoGraphOwner,
  submitAceCommandBufferFactoriesCooperatively,
  submitAceCommandBuffersCooperatively,
} from "../src/runtime/scheduler.js";

function commandBuffers(count: number): GPUCommandBuffer[] {
  return Array.from(
    { length: count },
    (_, index) => ({ label: `command-${index}` }) as GPUCommandBuffer,
  );
}

async function flushMicrotasks(): Promise<void> {
  for (let turn = 0; turn < 8; turn += 1) await Promise.resolve();
}

describe("cooperative GPU submission", () => {
  it("keeps exactly one singleton command buffer outstanding", async () => {
    const buffers = commandBuffers(3);
    const drains: PromiseWithResolvers<undefined>[] = [];
    const idles: PromiseWithResolvers<void>[] = [];
    const events: string[] = [];
    let outstanding = 0;
    let maximumOutstanding = 0;
    const queue = {
      submit(submitted: Iterable<GPUCommandBuffer>) {
        const values = [...submitted];
        expect(values).toHaveLength(1);
        expect(outstanding).toBe(0);
        outstanding += 1;
        maximumOutstanding = Math.max(maximumOutstanding, outstanding);
        events.push(`submit:${buffers.indexOf(values[0]!)}`);
      },
      onSubmittedWorkDone() {
        const deferred = Promise.withResolvers<undefined>();
        const index = drains.length;
        drains.push(deferred);
        events.push(`fence:${index}`);
        return deferred.promise.then(() => {
          outstanding -= 1;
          events.push(`drained:${index}`);
          return undefined;
        });
      },
    } satisfies Pick<GPUQueue, "submit" | "onSubmittedWorkDone">;

    const completion = submitAceCommandBuffersCooperatively({
      queue,
      commandBuffers: buffers,
      signal: new AbortController().signal,
      yieldQueueIdle: () => {
        const deferred = Promise.withResolvers<void>();
        const index = idles.length;
        idles.push(deferred);
        events.push(`idle:${index}`);
        return deferred.promise.then(() => {
          events.push(`idled:${index}`);
        });
      },
      onProgress: ({ completedCommandBuffers }) => {
        events.push(`progress:${completedCommandBuffers}`);
      },
    });

    expect(events).toEqual(["submit:0", "fence:0"]);
    drains[0]!.resolve(undefined);
    await flushMicrotasks();
    expect(events).toEqual([
      "submit:0",
      "fence:0",
      "drained:0",
      "idle:0",
      "progress:1",
    ]);
    idles[0]!.resolve();
    await flushMicrotasks();
    expect(events.at(-2)).toBe("submit:1");
    expect(events.at(-1)).toBe("fence:1");
    drains[1]!.resolve(undefined);
    await flushMicrotasks();
    expect(events.at(-1)).toBe("progress:2");
    idles[1]!.resolve();
    await flushMicrotasks();
    expect(events.at(-2)).toBe("submit:2");
    expect(events.at(-1)).toBe("fence:2");
    drains[2]!.resolve(undefined);
    const result = await completion;
    expect(events.at(-1)).toBe("progress:3");
    expect(maximumOutstanding).toBe(1);
    expect(outstanding).toBe(0);
    expect(result).toEqual({
      commandBuffersSubmitted: 3,
      queueDrains: 3,
      cooperativeIdleMs: 2 * ACE_COOPERATIVE_GPU_IDLE_MILLISECONDS,
    });
  });

  it("does not submit again when aborted during the idle interval", async () => {
    const controller = new AbortController();
    const idle = Promise.withResolvers<void>();
    const submissions: GPUCommandBuffer[][] = [];
    const queue = {
      submit(buffers: Iterable<GPUCommandBuffer>) {
        submissions.push([...buffers]);
      },
      onSubmittedWorkDone: () => Promise.resolve(undefined),
    } satisfies Pick<GPUQueue, "submit" | "onSubmittedWorkDone">;
    const completion = submitAceCommandBuffersCooperatively({
      queue,
      commandBuffers: commandBuffers(2),
      signal: controller.signal,
      yieldQueueIdle: () => idle.promise,
    });
    const rejection = expect(completion).rejects.toMatchObject({ name: "AbortError" });
    await flushMicrotasks();
    expect(submissions).toHaveLength(1);
    controller.abort();
    idle.resolve();
    await rejection;
    expect(submissions).toHaveLength(1);
  });

  it("awaits an already-started idle timer when progress throws", async () => {
    const failure = new Error("progress failed");
    const idle = Promise.withResolvers<void>();
    let rejected = false;
    const completion = submitAceCommandBuffersCooperatively({
      queue: {
        submit: vi.fn(),
        onSubmittedWorkDone: () => Promise.resolve(undefined),
      },
      commandBuffers: commandBuffers(2),
      signal: new AbortController().signal,
      yieldQueueIdle: () => idle.promise,
      onProgress: () => {
        throw failure;
      },
    });
    void completion.catch(() => {
      rejected = true;
    });
    await flushMicrotasks();
    expect(rejected).toBe(false);
    idle.resolve();
    await expect(completion).rejects.toBe(failure);
  });

  it("stops on a drain failure without starting an idle timer", async () => {
    const failure = new Error("device lost");
    const idle = vi.fn(async () => undefined);
    const submit = vi.fn();
    await expect(
      submitAceCommandBuffersCooperatively({
        queue: {
          submit,
          onSubmittedWorkDone: () => Promise.reject(failure),
        },
        commandBuffers: commandBuffers(2),
        signal: new AbortController().signal,
        yieldQueueIdle: idle,
      }),
    ).rejects.toBe(failure);
    expect(submit).toHaveBeenCalledTimes(1);
    expect(idle).not.toHaveBeenCalled();
  });

  it("rejects an empty graph without touching the queue", async () => {
    const queue = {
      submit: vi.fn(),
      onSubmittedWorkDone: vi.fn(),
    } satisfies Pick<GPUQueue, "submit" | "onSubmittedWorkDone">;
    await expect(
      submitAceCommandBuffersCooperatively({
        queue,
        commandBuffers: [],
        signal: new AbortController().signal,
      }),
    ).rejects.toThrow(/At least one/);
    expect(queue.submit).not.toHaveBeenCalled();
  });

  it("creates lazy command buffers only after the prior drain and idle", async () => {
    const drains: PromiseWithResolvers<undefined>[] = [];
    const idles: PromiseWithResolvers<void>[] = [];
    const events: string[] = [];
    let outstanding = 0;
    let maximumOutstanding = 0;
    const completion = submitAceCommandBufferFactoriesCooperatively({
      queue: {
        submit(buffers) {
          expect([...buffers]).toHaveLength(1);
          expect(outstanding).toBe(0);
          outstanding += 1;
          maximumOutstanding = Math.max(maximumOutstanding, outstanding);
          events.push("submit");
        },
        onSubmittedWorkDone() {
          const drain = Promise.withResolvers<undefined>();
          drains.push(drain);
          return drain.promise.then(() => {
            outstanding -= 1;
            events.push("drain");
            return undefined;
          });
        },
      },
      commandBufferCount: 3,
      createCommandBuffer(index) {
        events.push(`create:${index}`);
        return { label: `lazy-${index}` } as GPUCommandBuffer;
      },
      signal: new AbortController().signal,
      yieldQueueIdle() {
        const idle = Promise.withResolvers<void>();
        idles.push(idle);
        return idle.promise.then(() => { events.push("idle"); });
      },
    });
    expect(events).toEqual(["create:0", "submit"]);
    drains[0]!.resolve(undefined);
    await flushMicrotasks();
    expect(events).toEqual(["create:0", "submit", "drain"]);
    idles[0]!.resolve();
    await flushMicrotasks();
    expect(events.slice(-3)).toEqual(["idle", "create:1", "submit"]);
    drains[1]!.resolve(undefined);
    await flushMicrotasks();
    idles[1]!.resolve();
    await flushMicrotasks();
    expect(events.slice(-3)).toEqual(["idle", "create:2", "submit"]);
    drains[2]!.resolve(undefined);
    await expect(completion).resolves.toMatchObject({ queueDrains: 3 });
    expect(maximumOutstanding).toBe(1);
  });

  it("reports bounded submit-through-drain timing without changing order", async () => {
    const events: string[] = [];
    const clock = [10, 15, 20, 27];
    const timings: Array<{
      commandBufferIndex: number;
      submitThroughDrainMs: number;
    }> = [];
    const completion = await submitAceCommandBufferFactoriesCooperatively({
      queue: {
        submit() {
          events.push("submit");
        },
        async onSubmittedWorkDone() {
          events.push("drain");
        },
      },
      commandBufferCount: 2,
      createCommandBuffer: (index) =>
        ({ label: `timed-${index}` }) as GPUCommandBuffer,
      signal: new AbortController().signal,
      now: () => clock.shift()!,
      yieldQueueIdle: async () => {
        events.push("idle-started");
      },
      onCommandBufferDrained: (timing) => {
        events.push(`timing:${timing.commandBufferIndex}`);
        timings.push(timing);
      },
      onProgress: ({ completedCommandBuffers }) => {
        events.push(`progress:${completedCommandBuffers}`);
      },
    });

    expect(events).toEqual([
      "submit",
      "drain",
      "idle-started",
      "timing:0",
      "progress:1",
      "submit",
      "drain",
      "timing:1",
      "progress:2",
    ]);
    expect(timings).toEqual([
      { commandBufferIndex: 0, submitThroughDrainMs: 5 },
      { commandBufferIndex: 1, submitThroughDrainMs: 7 },
    ]);
    expect(completion).toMatchObject({
      commandBuffersSubmitted: 2,
      queueDrains: 2,
    });
  });

  it("does not invoke a lazy factory after abort and propagates factory failure", async () => {
    const controller = new AbortController();
    controller.abort();
    const create = vi.fn(() => ({}) as GPUCommandBuffer);
    const queue = {
      submit: vi.fn(),
      onSubmittedWorkDone: vi.fn(async () => undefined),
    } satisfies Pick<GPUQueue, "submit" | "onSubmittedWorkDone">;
    await expect(submitAceCommandBufferFactoriesCooperatively({
      queue,
      commandBufferCount: 1,
      createCommandBuffer: create,
      signal: controller.signal,
    })).rejects.toMatchObject({ name: "AbortError" });
    expect(create).not.toHaveBeenCalled();

    const failure = new Error("encode failed");
    await expect(submitAceCommandBufferFactoriesCooperatively({
      queue,
      commandBufferCount: 1,
      createCommandBuffer() {
        throw failure;
      },
      signal: new AbortController().signal,
    })).rejects.toBe(failure);
    expect(queue.submit).not.toHaveBeenCalled();
    expect(queue.onSubmittedWorkDone).not.toHaveBeenCalled();
  });

  it("awaits a lazy submission's started idle when progress throws", async () => {
    const failure = new Error("lazy progress failed");
    const idle = Promise.withResolvers<void>();
    let rejected = false;
    const completion = submitAceCommandBufferFactoriesCooperatively({
      queue: {
        submit: vi.fn(),
        onSubmittedWorkDone: vi.fn(async () => undefined),
      },
      commandBufferCount: 2,
      createCommandBuffer: () => ({}) as GPUCommandBuffer,
      signal: new AbortController().signal,
      yieldQueueIdle: () => idle.promise,
      onProgress() { throw failure; },
    });
    void completion.catch(() => { rejected = true; });
    await flushMicrotasks();
    expect(rejected).toBe(false);
    idle.resolve();
    await expect(completion).rejects.toBe(failure);
  });

  it("holds a lazy scheduler lease through drain and disposal", async () => {
    const scheduler = new AceCooperativeGpuScheduler();
    const drain = Promise.withResolvers<undefined>();
    const events: string[] = [];
    let call = 0;
    const queue = {
      submit() { events.push(`submit:${call}`); },
      onSubmittedWorkDone() {
        call += 1;
        return call === 1 ? drain.promise : Promise.resolve(undefined);
      },
    } satisfies Pick<GPUQueue, "submit" | "onSubmittedWorkDone">;
    const run = () => scheduler.runLazy({
      queue,
      commandBufferCount: 1,
      createCommandBuffer(index) {
        events.push(`create:${call}:${index}`);
        return {} as GPUCommandBuffer;
      },
      signal: new AbortController().signal,
    });
    const first = run();
    const second = run();
    await flushMicrotasks();
    expect(events).toEqual(["create:0:0", "submit:0"]);
    let disposed = false;
    const disposal = scheduler.dispose().then(() => { disposed = true; });
    await expect(second).rejects.toMatchObject({ name: "InvalidStateError" });
    expect(disposed).toBe(false);
    drain.resolve(undefined);
    await first;
    await disposal;
    expect(disposed).toBe(true);
    expect(events).toHaveLength(2);
  });
});

describe("FIFO graph ownership", () => {
  it("grants leases in order and makes release idempotent", async () => {
    const owner = new AceFifoGraphOwner();
    const first = await owner.acquire();
    const order = [first.sequence];
    const secondPromise = owner.acquire().then((lease) => {
      order.push(lease.sequence);
      return lease;
    });
    const thirdPromise = owner.acquire().then((lease) => {
      order.push(lease.sequence);
      return lease;
    });
    await flushMicrotasks();
    expect(order).toEqual([1]);
    first.release();
    const second = await secondPromise;
    first.release();
    await flushMicrotasks();
    expect(order).toEqual([1, 2]);
    second.release();
    const third = await thirdPromise;
    expect(order).toEqual([1, 2, 3]);
    third.release();
  });

  it("removes an aborted waiter without disturbing FIFO order", async () => {
    const owner = new AceFifoGraphOwner();
    const first = await owner.acquire();
    const controller = new AbortController();
    const aborted = owner.acquire(controller.signal);
    const third = owner.acquire();
    controller.abort();
    await expect(aborted).rejects.toMatchObject({ name: "AbortError" });
    first.release();
    expect((await third).sequence).toBe(3);
  });

  it("rejects queued owners on dispose and waits for the active lease", async () => {
    const owner = new AceFifoGraphOwner();
    const active = await owner.acquire();
    const waiting = owner.acquire();
    let disposed = false;
    const disposal = owner.dispose().then(() => {
      disposed = true;
    });
    await expect(waiting).rejects.toMatchObject({ name: "InvalidStateError" });
    await flushMicrotasks();
    expect(disposed).toBe(false);
    active.release();
    await disposal;
    expect(disposed).toBe(true);
    await expect(owner.acquire()).rejects.toMatchObject({ name: "InvalidStateError" });
  });

  it("holds graph ownership through a final queue drain", async () => {
    const scheduler = new AceCooperativeGpuScheduler();
    const finalDrain = Promise.withResolvers<undefined>();
    const events: string[] = [];
    const queue = {
      submit(buffers: Iterable<GPUCommandBuffer>) {
        events.push(`submit:${[...buffers][0]!.label}`);
      },
      onSubmittedWorkDone() {
        events.push("fence");
        return finalDrain.promise;
      },
    } satisfies Pick<GPUQueue, "submit" | "onSubmittedWorkDone">;
    const first = scheduler.run({
      queue,
      commandBuffers: commandBuffers(1),
      signal: new AbortController().signal,
    });
    const second = scheduler.run({
      queue,
      commandBuffers: commandBuffers(1),
      signal: new AbortController().signal,
    });
    await flushMicrotasks();
    expect(events).toEqual(["submit:command-0", "fence"]);
    finalDrain.resolve(undefined);
    await first;
    await flushMicrotasks();
    expect(events).toEqual([
      "submit:command-0",
      "fence",
      "submit:command-0",
      "fence",
    ]);
    await second;
  });
});
