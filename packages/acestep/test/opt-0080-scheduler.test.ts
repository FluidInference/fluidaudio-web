import { describe, expect, it, vi } from "vitest";

import {
  ACE_COOPERATIVE_GPU_IDLE_MILLISECONDS,
  AceCooperativeGpuScheduler,
  planAceDepth2Epoch4CompletionEpochs,
  submitAceCommandBufferFactoriesDepth2Epoch4,
  type AceDepth2Epoch4LazySubmissionOptions,
} from "../src/runtime/scheduler.js";

type TestQueue = Pick<GPUQueue, "submit" | "onSubmittedWorkDone">;

function buffer(index: number): GPUCommandBuffer {
  return { label: `command-${index}` } as GPUCommandBuffer;
}

function neverLost(): Promise<never> {
  return new Promise(() => undefined);
}

function testDevice(
  lost: Promise<unknown> = neverLost(),
): Pick<GPUDevice, "destroy" | "lost"> {
  return {
    destroy: vi.fn(),
    lost: lost as Promise<GPUDeviceLostInfo>,
  };
}

async function flushMicrotasks(): Promise<void> {
  for (let turn = 0; turn < 12; turn += 1) await Promise.resolve(undefined);
}

function baseOptions(
  queue: TestQueue,
  commandBufferCount: number,
  overrides: Partial<AceDepth2Epoch4LazySubmissionOptions> = {},
): AceDepth2Epoch4LazySubmissionOptions {
  let clock = 0;
  return {
    queue,
    commandBufferCount,
    phaseCommandBufferCounts: [commandBufferCount],
    createCommandBuffer: buffer,
    signal: new AbortController().signal,
    device: testDevice(),
    yieldQueueIdle: async () => undefined,
    now: () => {
      clock += 1;
      return clock;
    },
    ...overrides,
  };
}

describe("OPT-0080 completion-epoch planning", () => {
  it.each([
    [1, [1]],
    [2, [2]],
    [3, [3]],
    [4, [4]],
    [5, [4, 1]],
    [10, [4, 4, 2]],
  ] as const)("plans a %i-command tail exactly", (count, expectedCounts) => {
    const plans = planAceDepth2Epoch4CompletionEpochs(count, [count]);
    expect(plans.map((plan) => plan.commandBufferCount)).toEqual(expectedCounts);
    expect(plans.map((plan) => plan.completionEpochIndex)).toEqual(
      plans.map((_, index) => index),
    );
    expect(plans.at(-1)?.lastCommandBufferIndex).toBe(count - 1);
    expect(Object.isFrozen(plans)).toBe(true);
    expect(plans.every(Object.isFrozen)).toBe(true);
  });

  it("preserves the evaluation-0 and full-graph phase boundaries", () => {
    const evaluation0 = planAceDepth2Epoch4CompletionEpochs(341, [25, 316]);
    expect(evaluation0).toHaveLength(86);
    expect(evaluation0.filter(({ phaseIndex }) => phaseIndex === 0)).toHaveLength(7);
    expect(evaluation0.filter(({ phaseIndex }) => phaseIndex === 1)).toHaveLength(79);
    expect(evaluation0[6]).toMatchObject({
      phaseIndex: 0,
      firstCommandBufferIndex: 24,
      lastCommandBufferIndex: 24,
      commandBufferCount: 1,
    });
    expect(evaluation0[7]).toMatchObject({
      phaseIndex: 1,
      firstCommandBufferIndex: 25,
      lastCommandBufferIndex: 28,
      commandBufferCount: 4,
    });

    const fullPhaseCounts = [25, ...Array<number>(8).fill(316)];
    const full = planAceDepth2Epoch4CompletionEpochs(2_553, fullPhaseCounts);
    expect(full).toHaveLength(7 + 8 * 79);
    expect(full.at(-1)?.lastCommandBufferIndex).toBe(2_552);
    expect(new Set(full.map(({ phaseIndex }) => phaseIndex))).toEqual(
      new Set([0, 1, 2, 3, 4, 5, 6, 7, 8]),
    );
  });

  it("rejects empty, fractional, unsafe, zero-phase, and mismatched plans", () => {
    expect(() => planAceDepth2Epoch4CompletionEpochs(0, [])).toThrow(/At least one/);
    expect(() => planAceDepth2Epoch4CompletionEpochs(1.5, [1.5])).toThrow(
      /At least one/,
    );
    expect(() => planAceDepth2Epoch4CompletionEpochs(1, [])).toThrow(/phase/);
    expect(() => planAceDepth2Epoch4CompletionEpochs(2, [1, 0])).toThrow(
      /phase 1/,
    );
    expect(() => planAceDepth2Epoch4CompletionEpochs(2, [1])).toThrow(
      /does not match/,
    );
    expect(() => planAceDepth2Epoch4CompletionEpochs(
      Number.MAX_SAFE_INTEGER,
      [Number.MAX_SAFE_INTEGER, 1],
    )).toThrow(/unsafe/);
  });
});

describe("OPT-0080 depth-two/four-completion scheduling", () => {
  it("keeps two singleton buffers outstanding and does not submit five before the true drain and idle", async () => {
    const fences: PromiseWithResolvers<undefined>[] = [];
    const idles: PromiseWithResolvers<undefined>[] = [];
    const events: string[] = [];
    const completionIndices: number[] = [];
    const epochIndices: number[] = [];
    let submitted = 0;
    const queue: TestQueue = {
      submit(values) {
        const submittedValues = [...values];
        expect(submittedValues).toHaveLength(1);
        events.push(`submit:${submittedValues[0]!.label}`);
        submitted += 1;
      },
      onSubmittedWorkDone() {
        const fence = Promise.withResolvers<undefined>();
        const index = fences.length;
        fences.push(fence);
        events.push(`fence:${index}`);
        return fence.promise;
      },
    };
    const completion = submitAceCommandBufferFactoriesDepth2Epoch4(
      baseOptions(queue, 5, {
        createCommandBuffer(index) {
          events.push(`create:${index}`);
          return buffer(index);
        },
        yieldQueueIdle() {
          const idle = Promise.withResolvers<undefined>();
          idles.push(idle);
          events.push("idle:start");
          return idle.promise.then(() => { events.push("idle:end"); });
        },
        onCommandBufferCompleted(timing, progress) {
          completionIndices.push(timing.commandBufferIndex);
          events.push(`complete:${timing.commandBufferIndex}`);
          expect(progress.outstandingCommandBuffers).toBe(
            timing.trueQueueDrain ? 0 : 1,
          );
        },
        onCompletionEpochDrained(timing) {
          epochIndices.push(timing.completionEpochIndex);
          events.push(`epoch:${timing.completionEpochIndex}`);
        },
      }),
    );

    expect(events).toEqual([
      "create:0", "submit:command-0", "fence:0",
      "create:1", "submit:command-1", "fence:1",
    ]);
    expect(submitted).toBe(2);

    fences[0]!.resolve(undefined);
    await flushMicrotasks();
    expect(events.slice(-4)).toEqual([
      "complete:0", "create:2", "submit:command-2", "fence:2",
    ]);
    fences[1]!.resolve(undefined);
    await flushMicrotasks();
    expect(events.slice(-4)).toEqual([
      "complete:1", "create:3", "submit:command-3", "fence:3",
    ]);
    fences[2]!.resolve(undefined);
    await flushMicrotasks();
    expect(events.at(-1)).toBe("complete:2");
    expect(submitted).toBe(4);
    fences[3]!.resolve(undefined);
    await flushMicrotasks();
    expect(events.slice(-3)).toEqual(["complete:3", "epoch:0", "idle:start"]);
    expect(submitted).toBe(4);

    idles[0]!.resolve(undefined);
    await flushMicrotasks();
    expect(events.slice(-4)).toEqual([
      "idle:end", "create:4", "submit:command-4", "fence:4",
    ]);
    fences[4]!.resolve(undefined);
    const result = await completion;
    expect(completionIndices).toEqual([0, 1, 2, 3, 4]);
    expect(epochIndices).toEqual([0, 1]);
    expect(result).toEqual({
      commandBuffersSubmitted: 5,
      queueDrains: 2,
      cooperativeIdleMs: ACE_COOPERATIVE_GPU_IDLE_MILLISECONDS,
      completionFenceRequestedCount: 5,
      completionFenceSettledCount: 5,
      completionFenceRejectedCount: 0,
      trueQueueDrainCount: 2,
      completionEpochCount: 2,
      requestedCooperativeIdleMs: ACE_COOPERATIVE_GPU_IDLE_MILLISECONDS,
      cooperativeIdleTurns: 1,
      maximumOutstandingCommandBuffers: 2,
    });
  });

  it.each([1, 2, 3, 4, 5, 10])(
    "submits and completes every command exactly once for count %i",
    async (commandBufferCount) => {
      const creates: number[] = [];
      const submits: number[] = [];
      const completions: number[] = [];
      let fenceRequests = 0;
      const queue: TestQueue = {
        submit(values) {
          const [value] = [...values];
          submits.push(Number(value!.label!.split("-").at(-1)));
        },
        onSubmittedWorkDone() {
          fenceRequests += 1;
          return Promise.resolve(undefined);
        },
      };
      const result = await submitAceCommandBufferFactoriesDepth2Epoch4(
        baseOptions(queue, commandBufferCount, {
          createCommandBuffer(index) {
            creates.push(index);
            return buffer(index);
          },
          onCommandBufferCompleted(timing) {
            completions.push(timing.commandBufferIndex);
          },
        }),
      );
      const expected = Array.from({ length: commandBufferCount }, (_, index) => index);
      expect(creates).toEqual(expected);
      expect(submits).toEqual(expected);
      expect(completions).toEqual(expected);
      expect(fenceRequests).toBe(commandBufferCount);
      expect(result.maximumOutstandingCommandBuffers).toBe(
        commandBufferCount === 1 ? 1 : 2,
      );
      expect(result.trueQueueDrainCount).toBe(Math.ceil(commandBufferCount / 4));
      expect(result.cooperativeIdleTurns).toBe(
        Math.ceil(commandBufferCount / 4) - 1,
      );
    },
  );

  it("starts a new phase only after the prior boundary idle", async () => {
    const fences: PromiseWithResolvers<undefined>[] = [];
    const idle = Promise.withResolvers<undefined>();
    const events: string[] = [];
    const queue: TestQueue = {
      submit(values) { events.push(`submit:${[...values][0]!.label}`); },
      onSubmittedWorkDone() {
        const fence = Promise.withResolvers<undefined>();
        fences.push(fence);
        return fence.promise;
      },
    };
    const completion = submitAceCommandBufferFactoriesDepth2Epoch4(
      baseOptions(queue, 2, {
        phaseCommandBufferCounts: [1, 1],
        onPhaseStarted(phase) { events.push(`phase:${phase.phaseIndex}`); },
        createCommandBuffer(index) {
          events.push(`create:${index}`);
          return buffer(index);
        },
        yieldQueueIdle() {
          events.push("idle:start");
          return idle.promise.then(() => { events.push("idle:end"); });
        },
      }),
    );
    expect(events).toEqual(["phase:0", "create:0", "submit:command-0"]);
    fences[0]!.resolve(undefined);
    await flushMicrotasks();
    expect(events.at(-1)).toBe("idle:start");
    expect(events).not.toContain("phase:1");
    idle.resolve(undefined);
    await flushMicrotasks();
    expect(events.slice(-4)).toEqual([
      "idle:end", "phase:1", "create:1", "submit:command-1",
    ]);
    fences[1]!.resolve(undefined);
    await completion;
  });

  it("records overlapping per-fence timing but disjoint epoch timing", async () => {
    const clock = [10, 12, 20, 24, 30, 40];
    const perCommand: number[] = [];
    const epochs: number[] = [];
    const queue: TestQueue = {
      submit: vi.fn(),
      onSubmittedWorkDone: vi.fn(async () => undefined),
    };
    await submitAceCommandBufferFactoriesDepth2Epoch4(
      baseOptions(queue, 2, {
        now: () => clock.shift()!,
        onCommandBufferCompleted(timing) {
          perCommand.push(timing.submitThroughCompletionFenceMs);
        },
        onCompletionEpochDrained(timing) {
          epochs.push(timing.submitThroughTrueDrainMs);
        },
      }),
    );
    expect(perCommand).toEqual([10, 12]);
    expect(epochs).toEqual([14]);
    expect(clock).toEqual([30, 40]);
  });
});

describe("OPT-0080 stop, drain, and ownership failures", () => {
  it("observes callback abort with exactly one successor and drains it before rejecting", async () => {
    const controller = new AbortController();
    const fences: PromiseWithResolvers<undefined>[] = [];
    const creates: number[] = [];
    const completions: number[] = [];
    const queue: TestQueue = {
      submit: vi.fn(),
      onSubmittedWorkDone() {
        const fence = Promise.withResolvers<undefined>();
        fences.push(fence);
        return fence.promise;
      },
    };
    const completion = submitAceCommandBufferFactoriesDepth2Epoch4(
      baseOptions(queue, 4, {
        signal: controller.signal,
        createCommandBuffer(index) {
          creates.push(index);
          return buffer(index);
        },
        onCommandBufferCompleted(timing) {
          completions.push(timing.commandBufferIndex);
          controller.abort();
        },
      }),
    );
    expect(creates).toEqual([0, 1]);
    fences[0]!.resolve(undefined);
    await flushMicrotasks();
    expect(creates).toEqual([0, 1]);
    expect(completions).toEqual([0]);
    let rejected = false;
    void completion.catch(() => { rejected = true; });
    await flushMicrotasks();
    expect(rejected).toBe(false);
    fences[1]!.resolve(undefined);
    await expect(completion).rejects.toMatchObject({ name: "AbortError" });
    expect(creates).toEqual([0, 1]);
    expect(completions).toEqual([0]);
  });

  it("drains the successor after a lazy factory failure", async () => {
    const failure = new Error("encode failed");
    const fences: PromiseWithResolvers<undefined>[] = [];
    const creates: number[] = [];
    const queue: TestQueue = {
      submit: vi.fn(),
      onSubmittedWorkDone() {
        const fence = Promise.withResolvers<undefined>();
        fences.push(fence);
        return fence.promise;
      },
    };
    const completion = submitAceCommandBufferFactoriesDepth2Epoch4(
      baseOptions(queue, 4, {
        createCommandBuffer(index) {
          creates.push(index);
          if (index === 2) throw failure;
          return buffer(index);
        },
      }),
    );
    fences[0]!.resolve(undefined);
    await flushMicrotasks();
    expect(creates).toEqual([0, 1, 2]);
    let rejected = false;
    void completion.catch(() => { rejected = true; });
    await flushMicrotasks();
    expect(rejected).toBe(false);
    fences[1]!.resolve(undefined);
    await expect(completion).rejects.toBe(failure);
    expect(creates).toEqual([0, 1, 2]);
  });

  it("drains the first submit when the second submission throws", async () => {
    const failure = new Error("submit failed");
    const firstFence = Promise.withResolvers<undefined>();
    let submissions = 0;
    const queue: TestQueue = {
      submit() {
        submissions += 1;
        if (submissions === 2) throw failure;
      },
      onSubmittedWorkDone: vi.fn(() => firstFence.promise),
    };
    const completion = submitAceCommandBufferFactoriesDepth2Epoch4(
      baseOptions(queue, 4),
    );
    let rejected = false;
    void completion.catch(() => { rejected = true; });
    await flushMicrotasks();
    expect(rejected).toBe(false);
    expect(submissions).toBe(2);
    firstFence.resolve(undefined);
    await expect(completion).rejects.toBe(failure);
  });

  it("uses a terminal recovery fence after synchronous fence capture failure", async () => {
    const failure = new Error("capture failed");
    const firstFence = Promise.withResolvers<undefined>();
    const recovery = Promise.withResolvers<undefined>();
    let fenceCalls = 0;
    const queue: TestQueue = {
      submit: vi.fn(),
      onSubmittedWorkDone() {
        fenceCalls += 1;
        if (fenceCalls === 1) return firstFence.promise;
        if (fenceCalls === 2) throw failure;
        return recovery.promise;
      },
    };
    const completion = submitAceCommandBufferFactoriesDepth2Epoch4(
      baseOptions(queue, 4),
    );
    firstFence.resolve(undefined);
    await flushMicrotasks();
    expect(fenceCalls).toBe(3);
    let rejected = false;
    void completion.catch(() => { rejected = true; });
    await flushMicrotasks();
    expect(rejected).toBe(false);
    recovery.resolve(undefined);
    await expect(completion).rejects.toBe(failure);
  });

  it("wraps a younger rejection immediately and preserves it until FIFO observation", async () => {
    const firstFence = Promise.withResolvers<undefined>();
    const secondFence = Promise.withResolvers<undefined>();
    const recovery = Promise.withResolvers<undefined>();
    const youngerFailure = new Error("younger fence failed");
    const creates: number[] = [];
    const completions: number[] = [];
    let fenceCalls = 0;
    const queue: TestQueue = {
      submit: vi.fn(),
      onSubmittedWorkDone() {
        fenceCalls += 1;
        if (fenceCalls === 1) return firstFence.promise;
        if (fenceCalls === 2) return secondFence.promise;
        return recovery.promise;
      },
    };
    const completion = submitAceCommandBufferFactoriesDepth2Epoch4(
      baseOptions(queue, 4, {
        createCommandBuffer(index) {
          creates.push(index);
          return buffer(index);
        },
        onCommandBufferCompleted(timing) {
          completions.push(timing.commandBufferIndex);
        },
      }),
    );
    let rejected = false;
    void completion.catch(() => { rejected = true; });
    secondFence.reject(youngerFailure);
    await flushMicrotasks();
    expect(rejected).toBe(false);
    firstFence.resolve(undefined);
    await flushMicrotasks();
    expect(fenceCalls).toBe(3);
    expect(rejected).toBe(false);
    recovery.resolve(undefined);
    await expect(completion).rejects.toBe(youngerFailure);
    expect(creates).toEqual([0, 1]);
    expect(completions).toEqual([]);
  });

  it("waits for confirmed device loss when the terminal recovery fence rejects", async () => {
    const primary = new Error("oldest fence failed");
    const younger = new Error("younger fence failed");
    const recoveryFailure = new Error("recovery fence failed");
    const loss = Promise.withResolvers<undefined>();
    const outcomes = [
      Promise.reject(primary),
      Promise.reject(younger),
      Promise.reject(recoveryFailure),
    ];
    const queue: TestQueue = {
      submit: vi.fn(),
      onSubmittedWorkDone: vi.fn(() => outcomes.shift()!),
    };
    const completion = submitAceCommandBufferFactoriesDepth2Epoch4(
      baseOptions(queue, 2, {
        device: testDevice(loss.promise),
      }),
    );
    let rejected = false;
    void completion.catch(() => { rejected = true; });
    await flushMicrotasks();
    expect(rejected).toBe(false);
    loss.resolve(undefined);
    await expect(completion).rejects.toBe(primary);
    expect(
      (primary as Error & { aceSecondarySchedulingFailures: unknown[] })
        .aceSecondarySchedulingFailures,
    ).toEqual([younger, recoveryFailure]);
  });

  it("never releases aliased storage when a nonconforming loss signal rejects", async () => {
    const primary = new Error("oldest fence failed");
    const recoveryFailure = new Error("recovery fence failed");
    const lossFailure = new Error("loss signal rejected");
    const destroy = vi.fn();
    const outcomes = [
      Promise.reject(primary),
      Promise.reject(recoveryFailure),
    ];
    const queue: TestQueue = {
      submit: vi.fn(),
      onSubmittedWorkDone: vi.fn(() => outcomes.shift()!),
    };
    const completion = submitAceCommandBufferFactoriesDepth2Epoch4(
      baseOptions(queue, 1, {
        device: {
          destroy,
          lost: Promise.reject(lossFailure),
        },
      }),
    );
    let settled = false;
    void completion.finally(() => { settled = true; });
    await flushMicrotasks();
    expect(destroy).toHaveBeenCalledTimes(1);
    expect(settled).toBe(false);
  });

  it("suppresses successor progress and drains it after a completion callback throws", async () => {
    const failure = new Error("callback failed");
    const fences: PromiseWithResolvers<undefined>[] = [];
    const completions: number[] = [];
    const queue: TestQueue = {
      submit: vi.fn(),
      onSubmittedWorkDone() {
        const fence = Promise.withResolvers<undefined>();
        fences.push(fence);
        return fence.promise;
      },
    };
    const completion = submitAceCommandBufferFactoriesDepth2Epoch4(
      baseOptions(queue, 4, {
        onCommandBufferCompleted(timing) {
          completions.push(timing.commandBufferIndex);
          throw failure;
        },
      }),
    );
    fences[0]!.resolve(undefined);
    await flushMicrotasks();
    let rejected = false;
    void completion.catch(() => { rejected = true; });
    expect(rejected).toBe(false);
    fences[1]!.resolve(undefined);
    await expect(completion).rejects.toBe(failure);
    expect(completions).toEqual([0]);
  });

  it("drains the successor after timing validation fails", async () => {
    const fences: PromiseWithResolvers<undefined>[] = [];
    const queue: TestQueue = {
      submit: vi.fn(),
      onSubmittedWorkDone() {
        const fence = Promise.withResolvers<undefined>();
        fences.push(fence);
        return fence.promise;
      },
    };
    let clockCalls = 0;
    const completion = submitAceCommandBufferFactoriesDepth2Epoch4(
      baseOptions(queue, 4, {
        now() {
          clockCalls += 1;
          return clockCalls <= 2 ? 10 + clockCalls : 0;
        },
      }),
    );
    fences[0]!.resolve(undefined);
    await flushMicrotasks();
    let rejected = false;
    void completion.catch(() => { rejected = true; });
    expect(rejected).toBe(false);
    fences[1]!.resolve(undefined);
    await expect(completion).rejects.toThrow(/monotonic/);
  });

  it("stops at epoch, idle, and next-phase callback failures without later submits", async () => {
    const epochFailure = new Error("epoch failed");
    const epochQueue: TestQueue = {
      submit: vi.fn(),
      onSubmittedWorkDone: vi.fn(async () => undefined),
    };
    await expect(submitAceCommandBufferFactoriesDepth2Epoch4(
      baseOptions(epochQueue, 5, {
        onCompletionEpochDrained() { throw epochFailure; },
      }),
    )).rejects.toBe(epochFailure);
    expect(epochQueue.submit).toHaveBeenCalledTimes(4);

    const idleFailure = new Error("idle failed");
    const idleQueue: TestQueue = {
      submit: vi.fn(),
      onSubmittedWorkDone: vi.fn(async () => undefined),
    };
    await expect(submitAceCommandBufferFactoriesDepth2Epoch4(
      baseOptions(idleQueue, 5, {
        yieldQueueIdle: vi.fn(async () => { throw idleFailure; }),
      }),
    )).rejects.toBe(idleFailure);
    expect(idleQueue.submit).toHaveBeenCalledTimes(4);

    const phaseFailure = new Error("phase failed");
    const phaseQueue: TestQueue = {
      submit: vi.fn(),
      onSubmittedWorkDone: vi.fn(async () => undefined),
    };
    await expect(submitAceCommandBufferFactoriesDepth2Epoch4(
      baseOptions(phaseQueue, 2, {
        phaseCommandBufferCounts: [1, 1],
        onPhaseStarted({ phaseIndex }) {
          if (phaseIndex === 1) throw phaseFailure;
        },
      }),
    )).rejects.toBe(phaseFailure);
    expect(phaseQueue.submit).toHaveBeenCalledTimes(1);
  });

  it("holds the FIFO lease and disposal until a prefetched successor settles", async () => {
    const scheduler = new AceCooperativeGpuScheduler();
    const controller = new AbortController();
    const fences: PromiseWithResolvers<undefined>[] = [];
    const queue: TestQueue = {
      submit: vi.fn(),
      onSubmittedWorkDone() {
        const fence = Promise.withResolvers<undefined>();
        fences.push(fence);
        return fence.promise;
      },
    };
    const run = scheduler.runLazyDepth2Epoch4(baseOptions(queue, 4, {
      signal: controller.signal,
      onCommandBufferCompleted() { controller.abort(); },
    }));
    await flushMicrotasks();
    fences[0]!.resolve(undefined);
    await flushMicrotasks();
    let disposed = false;
    const disposal = scheduler.dispose().then(() => { disposed = true; });
    await flushMicrotasks();
    expect(disposed).toBe(false);
    fences[1]!.resolve(undefined);
    await expect(run).rejects.toMatchObject({ name: "AbortError" });
    await disposal;
    expect(disposed).toBe(true);
    await expect(scheduler.runLazyDepth2Epoch4(baseOptions(queue, 1))).rejects
      .toMatchObject({ name: "InvalidStateError" });
  });
});
