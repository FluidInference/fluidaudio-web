import { describe, expect, it, vi } from "vitest";

import type {
  AcePlannerDecodeBatch,
  AcePlannerLogitRange,
  AcePlannerPrefillBatch,
} from "../src/runtime/planner.js";
import {
  AcePlannerGpuDeviceLostError,
  AcePlannerGpuExecutor,
  createPlannerReadbackLayout,
  planAcePlannerGpuPhaseMemory,
  type AcePlannerGpuExecutorProgress,
  type AcePlannerOpt0085SchedulingDiagnostics,
  type AcePlannerPreparedGpuExecutorResources,
  type AcePlannerPreparedPhaseGpuResources,
} from "../src/webgpu/planner-executor.js";
import {
  ACE_PLANNER_EMBEDDING_ROW_PARTS,
  planAcePlannerHeadSlices,
  planAcePlannerModel,
  type AcePlannerModelBindings,
  type AcePlannerModelDispatch,
  type AcePlannerModelQuantum,
  type AcePlannerModelWeights,
} from "../src/webgpu/planner-model.js";

Object.defineProperty(globalThis, "GPUMapMode", {
  configurable: true,
  value: Object.freeze({ READ: 1 }),
});

const COMPACT_RANGE = Object.freeze({
  firstTokenId: 151_669,
  tokenCount: 64_000,
});
const EOS_RANGE = Object.freeze({ firstTokenId: 151_645, tokenCount: 1 });

describe("OPT-0085 planner depth-two completion epochs", () => {
  it("keeps ordinary production selector-free on depth one", async () => {
    const fixture = plannerFixture();
    const executor = AcePlannerGpuExecutor.fromPreparedResources(
      fixture.resources,
    );

    await executor.prefill(prefillBatch(2));

    expect(fixture.submissions).toHaveLength(34);
    expect(fixture.maximumOutstanding()).toBe(1);
    expect(fixture.idle).toHaveBeenCalledTimes(34);
    expect(fixture.diagnostics).toHaveLength(0);
    await executor.destroy();
  });

  it("captures the explicit depth-one arm with its terminal idle", async () => {
    const terminalIdle = Promise.withResolvers<undefined>();
    const fixture = plannerFixture({
      depthOneEvidence: true,
      deferredIdle: { call: 34, completion: terminalIdle },
    });
    const executor = AcePlannerGpuExecutor.fromPreparedResources(
      fixture.resources,
    );
    let settled = false;
    const running = executor.prefillForOpt0085(
      "depth1-epoch1",
      prefillBatch(2),
    );
    void running.then(() => { settled = true; });

    await waitUntil(() => fixture.readbackUnmap.mock.calls.length === 1);

    expect(fixture.submissions).toHaveLength(34);
    expect(fixture.submissions.every((batch) => batch.length === 1)).toBe(true);
    expect(fixture.maximumOutstanding()).toBe(1);
    expect(fixture.idle).toHaveBeenCalledTimes(34);
    expect(fixture.diagnostics).toEqual([expect.objectContaining({
      schedulingProfile: "depth1-epoch1",
      totalCommandBuffers: 34,
      commandBuffersSubmitted: 34,
      completionFenceRequestedCount: 34,
      completionFenceSettledCount: 34,
      completionFenceRejectedCount: 0,
      trueQueueDrainCount: 34,
      completionEpochCount: 34,
      cooperativeIdleTurns: 34,
      requestedCooperativeIdleMs: 34,
      maximumOutstandingCommandBuffers: 1,
    })]);
    // Depth one requests its final real queue-empty interval before mapping,
    // but deliberately overlaps that non-GPU wait with the CPU map/detach.
    expect(fixture.events.lastIndexOf("idle:start")).toBeLessThan(
      fixture.events.indexOf("map:start"),
    );
    expect(fixture.events.indexOf("diagnostics")).toBeLessThan(
      fixture.events.indexOf("map:start"),
    );
    expect(fixture.events.at(-1)).toBe("map:unmap");
    expect(settled).toBe(false);

    terminalIdle.resolve(undefined);
    await running;
    expect(settled).toBe(true);
    expect(fixture.events.at(-1)).toBe("idle:end");

    await executor.destroy();
  });

  it.each([
    ["M1 full", 1, undefined, 34],
    ["M2 full", 2, undefined, 34],
    ["M2 compact", 2, COMPACT_RANGE, 33],
    ["M2 forced EOS", 2, EOS_RANGE, 33],
  ] as const)(
    "preserves singleton FIFO commands and exact %s topology",
    async (_label, rows, logitRange, totalCommandBuffers) => {
      const fixture = plannerFixture({ depthTwo: true });
      const executor = AcePlannerGpuExecutor.fromPreparedResources(
        fixture.resources,
      );

      await executor.prefillForOpt0085(
        "opt-0085-depth2-epoch4",
        prefillBatch(rows),
        logitRange,
      );

      const expectedDrains = Math.ceil(totalCommandBuffers / 4);
      expect(fixture.submissions).toHaveLength(totalCommandBuffers);
      expect(fixture.submissions.every((batch) => batch.length === 1)).toBe(true);
      expect(fixture.submittedLabels.at(-1)).toBe(
        "ace-planner-logit-readback",
      );
      expect(fixture.events.filter((event) =>
        event.startsWith("encode:") || event.startsWith("submit:")
      )).toEqual(fixture.submittedLabels.flatMap((label) => [
        `encode:${label}`,
        `submit:${label}`,
      ]));
      expect(fixture.maximumOutstanding()).toBe(2);
      expect(fixture.idle).toHaveBeenCalledTimes(expectedDrains - 1);
      expect(fixture.progress).toHaveLength(totalCommandBuffers);
      expect(fixture.progress.map(({ completedCommandBuffers }) =>
        completedCommandBuffers)).toEqual(
          Array.from({ length: totalCommandBuffers }, (_, index) => index + 1),
        );
      expect(fixture.progress.at(-1)).toMatchObject({
        completedCommandBuffers: totalCommandBuffers,
        totalCommandBuffers,
        queueDrains: expectedDrains,
        cooperativeIdleMs: expectedDrains - 1,
        stage: "readback",
        quantum: null,
        cumulativeQueueDrains: expectedDrains,
        cumulativeCooperativeIdleMs: expectedDrains - 1,
      });
      expect(fixture.diagnostics).toEqual([expect.objectContaining({
        schema: "ace-opt-0085-planner-scheduling-v1",
        schedulingProfile: "opt-0085-depth2-epoch4",
        phaseKind: "prefill",
        totalCommandBuffers,
        commandBuffersSubmitted: totalCommandBuffers,
        completionFenceRequestedCount: totalCommandBuffers,
        completionFenceSettledCount: totalCommandBuffers,
        completionFenceRejectedCount: 0,
        trueQueueDrainCount: expectedDrains,
        completionEpochCount: expectedDrains,
        cooperativeIdleTurns: expectedDrains - 1,
        requestedCooperativeIdleMs: expectedDrains - 1,
        maximumOutstandingCommandBuffers: 2,
      })]);
      expect(fixture.clears).toEqual(["cache-validity", "write-status"]);
      expect(fixture.clearRecords).toEqual([
        { commandIndex: 0, bufferLabel: "cache-validity" },
        { commandIndex: 0, bufferLabel: "write-status" },
      ]);

      await executor.destroy();
      expect(fixture.phaseDestroy).toHaveBeenCalledTimes(1);
      expect(fixture.rootDestroy).toHaveBeenCalledTimes(1);
    },
  );

  it.each([
    ["M1 full decode", 1, undefined, 34],
    ["M2 full decode", 2, undefined, 34],
    ["M2 compact decode", 2, COMPACT_RANGE, 33],
    ["M2 EOS decode", 2, EOS_RANGE, 33],
  ] as const)(
    "uses the same dynamic topology for %s",
    async (_label, rows, logitRange, expectedCommands) => {
      const fixture = plannerFixture({ depthTwo: true });
      const executor = AcePlannerGpuExecutor.fromPreparedResources(
        fixture.resources,
      );
      await executor.prefill(prefillBatch(rows));
      const submissionsBefore = fixture.submissions.length;
      const idleBefore = fixture.idle.mock.calls.length;

      await executor.decodeForOpt0085(
        "opt-0085-depth2-epoch4",
        decodeBatch(rows, 4),
        logitRange,
      );

      expect(fixture.submissions.length - submissionsBefore).toBe(
        expectedCommands,
      );
      expect(fixture.idle.mock.calls.length - idleBefore).toBe(8);
      expect(fixture.diagnostics.at(-1)).toMatchObject({
        phaseKind: "decode",
        totalCommandBuffers: expectedCommands,
        commandBuffersSubmitted: expectedCommands,
        completionFenceRequestedCount: expectedCommands,
        completionFenceSettledCount: expectedCommands,
        trueQueueDrainCount: 9,
        cooperativeIdleTurns: 8,
        maximumOutstandingCommandBuffers: 2,
      });
      expect(fixture.clears).toEqual(["cache-validity", "write-status"]);
      expect(fixture.clearRecords).toEqual([
        { commandIndex: 0, bufferLabel: "cache-validity" },
        { commandIndex: 0, bufferLabel: "write-status" },
      ]);
      await executor.destroy();
    },
  );

  it("maps only after the final readback fence and keeps decode cache intact", async () => {
    const fixture = plannerFixture({ depthTwo: true, manualFences: true });
    const executor = AcePlannerGpuExecutor.fromPreparedResources(
      fixture.resources,
    );
    const running = executor.prefillForOpt0085(
      "opt-0085-depth2-epoch4",
      prefillBatch(2),
    );

    await drainOldestUntil(fixture, 33);
    expect(fixture.submissions).toHaveLength(34);
    expect(fixture.submittedLabels.at(-1)).toBe(
      "ace-planner-logit-readback",
    );
    expect(fixture.readbackMap).not.toHaveBeenCalled();
    expect(fixture.diagnostics).toHaveLength(0);

    fixture.fences[33]!.resolve(undefined);
    await running;
    expect(fixture.events.indexOf("fence:resolve:33")).toBeLessThan(
      fixture.events.indexOf("map:start"),
    );
    expect(fixture.readbackMap).toHaveBeenCalledTimes(1);
    expect(fixture.readbackUnmap).toHaveBeenCalledTimes(1);

    fixture.enableImmediateFences();
    await executor.decodeForOpt0085(
      "opt-0085-depth2-epoch4",
      decodeBatch(2, 4),
    );
    expect(fixture.clears).toEqual(["cache-validity", "write-status"]);
    expect(fixture.progress.at(-1)).toMatchObject({
      stage: "readback",
      cumulativeQueueDrains: 18,
      cumulativeCooperativeIdleMs: 16,
    });
    expect(fixture.diagnostics).toHaveLength(2);
    await executor.destroy();
  });

  it("holds the outer FIFO lease through mapping before replacing a phase", async () => {
    const mapCompletion = Promise.withResolvers<undefined>();
    const fixture = plannerFixture({
      depthTwo: true,
      manualFences: true,
      mapDeferred: mapCompletion,
    });
    const executor = AcePlannerGpuExecutor.fromPreparedResources(
      fixture.resources,
    );
    let firstSettled = false;
    const first = executor.prefillForOpt0085(
      "opt-0085-depth2-epoch4",
      prefillBatch(2),
    );
    void first.then(() => { firstSettled = true; });
    await waitUntil(() => fixture.fences.length === 2);
    const second = executor.prefillForOpt0085(
      "opt-0085-depth2-epoch4",
      prefillBatch(2),
    );
    await flushMicrotasks();

    expect(fixture.phaseCreate).toHaveBeenCalledTimes(1);
    expect(fixture.submissions).toHaveLength(2);
    expect(fixture.phaseDestroy).not.toHaveBeenCalled();

    await drainFenceRange(fixture, 0, 34);
    await waitUntil(() => fixture.readbackMap.mock.calls.length === 1);
    expect(firstSettled).toBe(false);
    expect(fixture.readbackUnmap).not.toHaveBeenCalled();
    expect(fixture.phaseCreate).toHaveBeenCalledTimes(1);
    expect(fixture.phaseDestroy).not.toHaveBeenCalled();
    expect(fixture.rootDestroy).not.toHaveBeenCalled();
    expect(fixture.submissions).toHaveLength(34);

    mapCompletion.resolve(undefined);
    await first;
    expect(firstSettled).toBe(true);
    expect(fixture.readbackUnmap).toHaveBeenCalledTimes(1);
    await waitUntil(() => fixture.submissions.length === 36);
    expect(fixture.phaseCreate).toHaveBeenCalledTimes(2);
    expect(fixture.phaseDestroy).toHaveBeenCalledTimes(1);
    expect(fixture.submissions).toHaveLength(36);

    await drainFenceRange(fixture, 34, 34);
    await second;
    await executor.destroy();
    expect(fixture.phaseDestroy).toHaveBeenCalledTimes(2);
    expect(fixture.rootDestroy).toHaveBeenCalledTimes(1);
  });

  it.each([
    ["mapAsync", 0],
    ["getMappedRange", 1],
  ] as const)(
    "releases only the phase after candidate %s failure",
    async (point, expectedUnmaps) => {
      const failure = new Error(`OPT-0085 ${point} failed`);
      const fixture = plannerFixture({
        depthTwo: true,
        mapFailure: { point, error: failure },
      });
      const executor = AcePlannerGpuExecutor.fromPreparedResources(
        fixture.resources,
      );

      await expect(executor.prefillForOpt0085(
        "opt-0085-depth2-epoch4",
        prefillBatch(2),
      )).rejects.toBe(failure);
      expect(fixture.submissions).toHaveLength(34);
      expect(fixture.diagnostics).toHaveLength(1);
      expect(fixture.readbackMap).toHaveBeenCalledTimes(1);
      expect(fixture.readbackUnmap).toHaveBeenCalledTimes(expectedUnmaps);
      expect(fixture.phaseDestroy).toHaveBeenCalledTimes(1);
      expect(fixture.rootDestroy).not.toHaveBeenCalled();

      await executor.destroy();
      expect(fixture.phaseDestroy).toHaveBeenCalledTimes(1);
      expect(fixture.rootDestroy).toHaveBeenCalledTimes(1);
    },
  );

  it("holds mapped storage and both owners when aborted during mapAsync", async () => {
    const mapCompletion = Promise.withResolvers<undefined>();
    const fixture = plannerFixture({
      depthTwo: true,
      mapDeferred: mapCompletion,
    });
    const executor = AcePlannerGpuExecutor.fromPreparedResources(
      fixture.resources,
    );
    const running = executor.prefillForOpt0085(
      "opt-0085-depth2-epoch4",
      prefillBatch(2),
    );
    await waitUntil(() => fixture.readbackMap.mock.calls.length === 1);
    const reason = new DOMException("OPT-0085 map cancelled", "AbortError");
    const destroying = executor.destroy(reason);
    await flushMicrotasks();

    expect(fixture.diagnostics).toHaveLength(1);
    expect(fixture.readbackUnmap).not.toHaveBeenCalled();
    expect(fixture.phaseDestroy).not.toHaveBeenCalled();
    expect(fixture.rootDestroy).not.toHaveBeenCalled();

    mapCompletion.resolve(undefined);
    await expect(running).rejects.toBe(reason);
    await destroying;
    expect(fixture.readbackUnmap).toHaveBeenCalledTimes(1);
    expect(fixture.phaseDestroy).toHaveBeenCalledTimes(1);
    expect(fixture.rootDestroy).toHaveBeenCalledTimes(1);
  });

  it("does not map after the candidate diagnostics callback fails", async () => {
    const failure = new Error("OPT-0085 diagnostics failed");
    const fixture = plannerFixture({
      depthTwo: true,
      diagnosticsFailure: failure,
    });
    const executor = AcePlannerGpuExecutor.fromPreparedResources(
      fixture.resources,
    );

    await expect(executor.prefillForOpt0085(
      "opt-0085-depth2-epoch4",
      prefillBatch(2),
    )).rejects.toBe(failure);
    expect(fixture.submissions).toHaveLength(34);
    expect(fixture.diagnostics).toHaveLength(1);
    expect(fixture.readbackMap).not.toHaveBeenCalled();
    expect(fixture.readbackUnmap).not.toHaveBeenCalled();
    expect(fixture.phaseDestroy).toHaveBeenCalledTimes(1);
    expect(fixture.rootDestroy).not.toHaveBeenCalled();

    await executor.destroy();
    expect(fixture.rootDestroy).toHaveBeenCalledTimes(1);
  });

  it("stops backfill and holds resources when progress throws", async () => {
    const failure = new Error("OPT-0085 progress failed");
    let callbackCount = 0;
    const fixture = plannerFixture({
      depthTwo: true,
      manualFences: true,
      onProgress() {
        callbackCount += 1;
        throw failure;
      },
    });
    const executor = AcePlannerGpuExecutor.fromPreparedResources(
      fixture.resources,
    );
    const running = executor.prefillForOpt0085(
      "opt-0085-depth2-epoch4",
      prefillBatch(2),
    );
    await waitUntil(() => fixture.fences.length === 2);

    fixture.fences[0]!.resolve(undefined);
    await flushMicrotasks();
    expect(callbackCount).toBe(1);
    expect(fixture.submissions).toHaveLength(2);
    expect(fixture.phaseDestroy).not.toHaveBeenCalled();
    expect(fixture.readbackMap).not.toHaveBeenCalled();

    fixture.fences[1]!.resolve(undefined);
    await expect(running).rejects.toBe(failure);
    expect(fixture.phaseDestroy).toHaveBeenCalledTimes(1);
    expect(fixture.rootDestroy).not.toHaveBeenCalled();
    expect(fixture.diagnostics).toHaveLength(0);
    await executor.destroy();
    expect(fixture.rootDestroy).toHaveBeenCalledTimes(1);
  });

  it("aborts with two in flight, then balances repeated destruction", async () => {
    const fixture = plannerFixture({ depthTwo: true, manualFences: true });
    const executor = AcePlannerGpuExecutor.fromPreparedResources(
      fixture.resources,
    );
    const running = executor.prefillForOpt0085(
      "opt-0085-depth2-epoch4",
      prefillBatch(2),
    );
    await waitUntil(() => fixture.fences.length === 2);
    const reason = new DOMException("OPT-0085 cancelled", "AbortError");
    const destroying = executor.destroy(reason);
    const repeated = executor.destroy(reason);
    await flushMicrotasks();

    expect(fixture.submissions).toHaveLength(2);
    expect(fixture.phaseDestroy).not.toHaveBeenCalled();
    expect(fixture.rootDestroy).not.toHaveBeenCalled();
    fixture.fences[0]!.resolve(undefined);
    await flushMicrotasks();
    expect(fixture.phaseDestroy).not.toHaveBeenCalled();
    fixture.fences[1]!.resolve(undefined);

    await expect(running).rejects.toBe(reason);
    await destroying;
    await repeated;
    expect(fixture.submissions).toHaveLength(2);
    expect(fixture.readbackMap).not.toHaveBeenCalled();
    expect(fixture.phaseDestroy).toHaveBeenCalledTimes(1);
    expect(fixture.rootDestroy).toHaveBeenCalledTimes(1);
  });

  it("settles a rejected fence before releasing the active phase", async () => {
    const failure = new Error("OPT-0085 completion fence rejected");
    const fixture = plannerFixture({ depthTwo: true, manualFences: true });
    const executor = AcePlannerGpuExecutor.fromPreparedResources(
      fixture.resources,
    );
    const running = executor.prefillForOpt0085(
      "opt-0085-depth2-epoch4",
      prefillBatch(2),
    );
    await waitUntil(() => fixture.fences.length === 2);

    fixture.fences[0]!.reject(failure);
    await flushMicrotasks();
    expect(fixture.phaseDestroy).not.toHaveBeenCalled();
    expect(fixture.submissions).toHaveLength(2);
    fixture.fences[1]!.resolve(undefined);

    await expect(running).rejects.toBe(failure);
    expect(fixture.phaseDestroy).toHaveBeenCalledTimes(1);
    expect(fixture.readbackMap).not.toHaveBeenCalled();
    await executor.destroy();
  });

  it("waits for both submitted commands after device loss", async () => {
    const fixture = plannerFixture({ depthTwo: true, manualFences: true });
    const executor = AcePlannerGpuExecutor.fromPreparedResources(
      fixture.resources,
    );
    const running = executor.prefillForOpt0085(
      "opt-0085-depth2-epoch4",
      prefillBatch(2),
    );
    await waitUntil(() => fixture.fences.length === 2);
    fixture.deviceLost.resolve({
      reason: "unknown",
      message: "OPT-0085 test device vanished",
    } as GPUDeviceLostInfo);
    await flushMicrotasks();

    expect(fixture.phaseDestroy).not.toHaveBeenCalled();
    expect(fixture.rootDestroy).not.toHaveBeenCalled();
    fixture.fences[0]!.resolve(undefined);
    await flushMicrotasks();
    expect(fixture.phaseDestroy).not.toHaveBeenCalled();
    fixture.fences[1]!.resolve(undefined);

    await expect(running).rejects.toBeInstanceOf(AcePlannerGpuDeviceLostError);
    await executor.destroy();
    expect(fixture.submissions).toHaveLength(2);
    expect(fixture.phaseDestroy).toHaveBeenCalledTimes(1);
    expect(fixture.rootDestroy).toHaveBeenCalledTimes(1);
  });

  it("stops after a failed epoch idle without encoding the next epoch", async () => {
    const failure = new Error("OPT-0085 idle failed");
    const fixture = plannerFixture({
      depthTwo: true,
      idleFailure: failure,
    });
    const executor = AcePlannerGpuExecutor.fromPreparedResources(
      fixture.resources,
    );

    await expect(executor.prefillForOpt0085(
      "opt-0085-depth2-epoch4",
      prefillBatch(2),
    )).rejects.toBe(failure);
    expect(fixture.submissions).toHaveLength(4);
    expect(fixture.readbackMap).not.toHaveBeenCalled();
    expect(fixture.phaseDestroy).toHaveBeenCalledTimes(1);
    await executor.destroy();
  });

  it("fails an unknown internal selector before creating phase storage", async () => {
    const fixture = plannerFixture({ depthTwo: true });
    const executor = AcePlannerGpuExecutor.fromPreparedResources(
      fixture.resources,
    );
    expect(() => executor.prefillForOpt0085(
      "depth-three" as never,
      prefillBatch(2),
    )).toThrow(/OPT-0085 scheduling profile is invalid/);
    expect(fixture.phaseCreate).not.toHaveBeenCalled();
    await executor.destroy();
  });

  it("requires every explicit arm to carry its topology hook", async () => {
    const fixture = plannerFixture();
    const executor = AcePlannerGpuExecutor.fromPreparedResources(
      fixture.resources,
    );
    expect(() => executor.prefillForOpt0085(
      "opt-0085-depth2-epoch4",
      prefillBatch(2),
    )).toThrow(/require reference-BF16 diagnostics/);
    expect(fixture.phaseCreate).not.toHaveBeenCalled();
    await executor.destroy();
  });

  it("does not let experiment configuration change ordinary production", async () => {
    const fixture = plannerFixture({ depthTwo: true });
    const executor = AcePlannerGpuExecutor.fromPreparedResources(
      fixture.resources,
    );

    await executor.prefill(prefillBatch(2));
    expect(fixture.submissions).toHaveLength(34);
    expect(fixture.maximumOutstanding()).toBe(1);
    expect(fixture.idle).toHaveBeenCalledTimes(34);
    expect(fixture.diagnostics).toHaveLength(0);
    await executor.destroy();
  });

  it("scopes each explicit arm to one invocation without a sticky override", async () => {
    const fixture = plannerFixture({ depthTwo: true });
    const executor = AcePlannerGpuExecutor.fromPreparedResources(
      fixture.resources,
    );

    await executor.prefillForOpt0085(
      "opt-0085-depth2-epoch4",
      prefillBatch(2),
    );
    expect(fixture.diagnostics).toHaveLength(1);
    expect(fixture.diagnostics[0]?.schedulingProfile).toBe(
      "opt-0085-depth2-epoch4",
    );

    const submissionsBeforeOrdinary = fixture.submissions.length;
    const idlesBeforeOrdinary = fixture.idle.mock.calls.length;
    await executor.decode(decodeBatch(2, 4));
    expect(fixture.submissions.length - submissionsBeforeOrdinary).toBe(34);
    expect(fixture.idle.mock.calls.length - idlesBeforeOrdinary).toBe(34);
    expect(fixture.diagnostics).toHaveLength(1);

    await executor.decodeForOpt0085(
      "depth1-epoch1",
      decodeBatch(2, 5),
    );
    expect(fixture.diagnostics).toHaveLength(2);
    expect(fixture.diagnostics[1]?.schedulingProfile).toBe("depth1-epoch1");
    await executor.destroy();
  });
});

interface PlannerFixtureOptions {
  readonly depthTwo?: boolean;
  readonly depthOneEvidence?: boolean;
  readonly manualFences?: boolean;
  readonly idleFailure?: Error;
  readonly deferredIdle?: Readonly<{
    readonly call: number;
    readonly completion: PromiseWithResolvers<undefined>;
  }>;
  readonly mapDeferred?: PromiseWithResolvers<undefined>;
  readonly mapFailure?: Readonly<{
    readonly point: "mapAsync" | "getMappedRange";
    readonly error: Error;
  }>;
  readonly diagnosticsFailure?: Error;
  readonly onProgress?: (progress: AcePlannerGpuExecutorProgress) => void;
}

function plannerFixture(options: PlannerFixtureOptions = {}) {
  const submissions: GPUCommandBuffer[][] = [];
  const submittedLabels: string[] = [];
  const fences: PromiseWithResolvers<undefined>[] = [];
  const clears: string[] = [];
  const clearRecords: Array<{
    readonly commandIndex: number;
    readonly bufferLabel: string;
  }> = [];
  const events: string[] = [];
  const progress: AcePlannerGpuExecutorProgress[] = [];
  const diagnostics: AcePlannerOpt0085SchedulingDiagnostics[] = [];
  const deviceLost = Promise.withResolvers<GPUDeviceLostInfo>();
  const phaseDestroy = vi.fn();
  const rootDestroy = vi.fn();
  const readbackMap = vi.fn(async () => {
    if (options.mapFailure?.point === "mapAsync") {
      throw options.mapFailure.error;
    }
    await options.mapDeferred?.promise;
  });
  const readbackUnmap = vi.fn();
  const phaseCreate = vi.fn(async (batch: AcePlannerPrefillBatch) =>
    fakePhase(
      batch,
      phaseDestroy,
      events,
      readbackMap,
      readbackUnmap,
      options.mapFailure?.point === "getMappedRange"
        ? options.mapFailure.error
        : undefined,
    ));
  let useManualFences = options.manualFences === true;
  let outstanding = 0;
  let maximumOutstanding = 0;
  let fenceIndex = 0;
  let encoderIndex = 0;
  let idleCall = 0;

  const queue = {
    writeBuffer: vi.fn(),
    submit(values: Iterable<GPUCommandBuffer>) {
      const batch = [...values];
      submissions.push(batch);
      const label = batch[0]!.label ?? "unlabelled";
      submittedLabels.push(label);
      events.push(`submit:${label}`);
      outstanding += 1;
      maximumOutstanding = Math.max(maximumOutstanding, outstanding);
    },
    onSubmittedWorkDone(): Promise<undefined> {
      const index = fenceIndex;
      fenceIndex += 1;
      if (!useManualFences) {
        return Promise.resolve(undefined).then(() => {
          outstanding -= 1;
          events.push(`fence:resolve:${index}`);
          return undefined;
        });
      }
      const fence = Promise.withResolvers<undefined>();
      fences.push(fence);
      return fence.promise.then(
        () => {
          outstanding -= 1;
          events.push(`fence:resolve:${index}`);
          return undefined;
        },
        (error: unknown) => {
          outstanding -= 1;
          events.push(`fence:reject:${index}`);
          throw error;
        },
      );
    },
  } as unknown as GPUQueue;
  const device = {
    queue,
    lost: deviceLost.promise,
    destroy: vi.fn(),
    createCommandEncoder({ label }: GPUCommandEncoderDescriptor = {}) {
      const commandIndex = encoderIndex;
      encoderIndex += 1;
      return {
        clearBuffer(buffer: GPUBuffer) {
          const bufferLabel = (buffer as unknown as { label: string }).label;
          clears.push(bufferLabel);
          clearRecords.push({ commandIndex, bufferLabel });
        },
        beginComputePass() {
          return { end: vi.fn() } as unknown as GPUComputePassEncoder;
        },
        copyBufferToBuffer: vi.fn(),
        finish() {
          events.push(`encode:${label ?? "unlabelled"}`);
          return { label } as GPUCommandBuffer;
        },
      } as unknown as GPUCommandEncoder;
    },
  } as unknown as GPUDevice;
  const runtime = {
    createPlannerDispatch: vi.fn(async (
      _label: string,
      batch: AcePlannerPrefillBatch | AcePlannerDecodeBatch,
      _bindings: AcePlannerModelBindings,
      logitRange?: AcePlannerLogitRange,
    ) => fakeDispatch(batch, logitRange)),
    destroy: vi.fn(),
  };
  const idle = vi.fn(async () => {
    idleCall += 1;
    events.push("idle:start");
    if (options.idleFailure !== undefined) throw options.idleFailure;
    if (options.deferredIdle?.call === idleCall) {
      await options.deferredIdle.completion.promise;
    }
    events.push("idle:end");
  });
  const resources: AcePlannerPreparedGpuExecutorResources = {
    device,
    modelProfile: "reference-bf16",
    runtime,
    weights: {} as AcePlannerModelWeights,
    residentWeightBytes: 1_024,
    ...(options.depthTwo === true || options.depthOneEvidence === true
      ? {
          onOpt0085Scheduling(event: AcePlannerOpt0085SchedulingDiagnostics) {
            diagnostics.push(event);
            events.push("diagnostics");
            if (options.diagnosticsFailure !== undefined) {
              throw options.diagnosticsFailure;
            }
          },
        }
      : {}),
    yieldQueueIdle: idle,
    onProgress(event) {
      progress.push(event);
      options.onProgress?.(event);
    },
    createPhase: phaseCreate,
    destroy() {
      rootDestroy();
      runtime.destroy();
    },
  };
  return {
    resources,
    submissions,
    submittedLabels,
    fences,
    clears,
    clearRecords,
    events,
    progress,
    diagnostics,
    idle,
    phaseCreate,
    phaseDestroy,
    rootDestroy,
    deviceLost,
    readbackMap,
    readbackUnmap,
    maximumOutstanding: () => maximumOutstanding,
    enableImmediateFences() {
      useManualFences = false;
    },
  };
}

function fakePhase(
  batch: AcePlannerPrefillBatch,
  destroy: () => void,
  events: string[],
  readbackMap: () => Promise<void>,
  readbackUnmap: () => void,
  getMappedRangeFailure: Error | undefined,
): AcePlannerPreparedPhaseGpuResources {
  const memory = planAcePlannerGpuPhaseMemory("reference-bf16", batch, 1_024);
  const bytes = new ArrayBuffer(memory.readbackBytes);
  for (const range of [undefined, COMPACT_RANGE, EOS_RANGE]) {
    const layout = createPlannerReadbackLayout(
      "reference-bf16",
      batch.rows as 1 | 2,
      range,
    );
    new Uint32Array(
      bytes,
      layout.writeStatusByteOffset,
      batch.rows,
    ).fill(1);
  }
  const readback = {
    size: memory.readbackBytes,
    async mapAsync() {
      events.push("map:start");
      await readbackMap();
    },
    getMappedRange: vi.fn(() => {
      if (getMappedRangeFailure !== undefined) throw getMappedRangeFailure;
      return bytes;
    }),
    unmap() {
      events.push("map:unmap");
      readbackUnmap();
    },
    destroy: vi.fn(),
  } as unknown as GPUBuffer;
  return {
    batch: batch.rows as 1 | 2,
    prefillTokens: batch.tokens,
    cacheCapacity: batch.cacheCapacity,
    bindings: fakeBindings(batch),
    readback,
    memory,
    destroy,
  };
}

function fakeDispatch(
  batch: AcePlannerPrefillBatch | AcePlannerDecodeBatch,
  logitRange?: AcePlannerLogitRange,
): AcePlannerModelDispatch {
  const quanta: AcePlannerModelQuantum[] = [fakeQuantum("embedding", null)];
  for (let layer = 0; layer < 28; layer += 1) {
    quanta.push(fakeQuantum("layer", layer));
  }
  quanta.push(fakeQuantum("final-norm", null));
  quanta.push(fakeQuantum("last-row-gather", null));
  // The authenticated full head has one more physical singleton than either
  // compact range. This freezes OPT-0085's dynamic 34/33-command topologies.
  quanta.push(fakeQuantum("tied-lm-head", null));
  if (logitRange === undefined) {
    quanta.push(fakeQuantum("tied-lm-head", null));
  }
  return {
    label: `opt-0085-${batch.kind}`,
    plan: planAcePlannerModel("reference-bf16", batch),
    logitRange: logitRange === undefined ? null : logitRange,
    headSlices: planAcePlannerHeadSlices(logitRange),
    primitiveCount: quanta.length,
    quanta,
    encode: vi.fn(),
  };
}

function fakeQuantum(
  kind: AcePlannerModelQuantum["kind"],
  layer: number | null,
): AcePlannerModelQuantum {
  return {
    id: `opt-0085-${kind}-${layer ?? "shared"}`,
    kind,
    layer,
    primitiveCount: 1,
    encode: vi.fn(),
  };
}

function fakeBindings(batch: AcePlannerPrefillBatch): AcePlannerModelBindings {
  const buffer = (label: string, size = 4 * 1024 * 1024): GPUBuffer =>
    ({ label, size } as unknown as GPUBuffer);
  const binding = (label: string, size?: number): GPUBufferBinding => {
    const value = buffer(label, size);
    return { buffer: value, offset: 0, size: value.size };
  };
  const scratch = binding("scratch");
  return {
    tokenIds: binding("token-ids"),
    weights: {} as AcePlannerModelWeights,
    controls: {
      validLengths: binding("valid-lengths"),
      queryPositions: binding("query-positions"),
      sourceValidity: binding("source-validity"),
      rowStartPositions: binding("row-start-positions"),
      cosine: binding("cosine"),
      sine: binding("sine"),
      lastPhysicalRowIndices: binding("last-physical-row-indices"),
      cacheValidity: binding("cache-validity", batch.rows * batch.cacheCapacity * 4),
      writeStatus: binding("write-status"),
    },
    cache: {
      layers: Array.from({ length: 28 }, (_, layer) => ({
        key: binding(`layer-${layer}-key`),
        value: binding(`layer-${layer}-value`),
      })),
    },
    scratch: {
      embedded: scratch,
      block: {
        normalizedInput: scratch,
        queryFlat: scratch,
        keyFlat: scratch,
        valueFlat: scratch,
        queryHeads: scratch,
        keyHeads: scratch,
        valueHeads: scratch,
        normalizedQueryHeads: scratch,
        normalizedKeyHeads: scratch,
        rotatedQueryHeads: scratch,
        rotatedKeyHeads: scratch,
        attentionHeads: scratch,
        mergedAttention: scratch,
        projectedAttention: scratch,
        afterAttention: scratch,
        normalizedAfterAttention: scratch,
        gate: scratch,
        up: scratch,
        gatedActivation: scratch,
        projectedMlp: scratch,
      },
      layerOutputs: [scratch, scratch],
      normalizedSequence: scratch,
      lastHiddenRows: scratch,
    },
    logits: ACE_PLANNER_EMBEDDING_ROW_PARTS.map((part, index) =>
      binding(`logits-${index}`, batch.rows * part.rowCount * 4)),
  };
}

function prefillBatch(rows: 1 | 2): AcePlannerPrefillBatch {
  const tokens = 4;
  const inputIds = new Uint32Array(rows * tokens);
  const positions = new Uint32Array(rows * tokens);
  const validity = new Uint32Array(rows * tokens).fill(1);
  const validLengths = new Uint32Array(rows * 2);
  for (let row = 0; row < rows; row += 1) {
    for (let token = 0; token < tokens; token += 1) {
      inputIds[row * tokens + token] = 10 + row * tokens + token;
      positions[row * tokens + token] = token;
    }
    validLengths[row * 2] = tokens;
    validLengths[row * 2 + 1] = tokens;
  }
  return {
    kind: "prefill",
    rows,
    tokens,
    cacheCapacity: 512,
    inputIds,
    keyValidity: validity.slice(),
    rotaryPositionIds: positions.slice(),
    causal: {
      rowStartPositions: new Uint32Array(rows),
      validLengths,
      queryPositions: positions,
      sourceValidity: validity,
    },
    conditionalRow: 0,
    unconditionalRow: rows === 2 ? 1 : null,
  };
}

function decodeBatch(rows: 1 | 2, cachedTokensBeforeAppend: number) {
  return {
    kind: "decode" as const,
    rows,
    tokens: 1 as const,
    cacheCapacity: 512,
    cachedTokensBeforeAppend,
    inputIds: new Uint32Array(rows).fill(42),
    rotaryPositionIds: new Uint32Array(rows).fill(cachedTokensBeforeAppend),
    causal: {
      rowStartPositions: new Uint32Array(rows).fill(cachedTokensBeforeAppend),
      validLengths: Uint32Array.from(
        { length: rows * 2 },
        (_, index) => index % 2 === 0 ? 1 : cachedTokensBeforeAppend + 1,
      ),
      queryPositions: new Uint32Array(rows).fill(cachedTokensBeforeAppend),
      sourceValidity: new Uint32Array(rows).fill(1),
    },
    conditionalRow: 0,
    unconditionalRow: rows === 2 ? 1 : null,
  } satisfies AcePlannerDecodeBatch;
}

async function drainOldestUntil(
  fixture: ReturnType<typeof plannerFixture>,
  count: number,
): Promise<void> {
  await drainFenceRange(fixture, 0, count);
}

async function drainFenceRange(
  fixture: ReturnType<typeof plannerFixture>,
  first: number,
  count: number,
): Promise<void> {
  for (let index = first; index < first + count; index += 1) {
    await waitUntil(() => fixture.fences.length > index);
    fixture.fences[index]!.resolve(undefined);
    await flushMicrotasks();
  }
}

async function waitUntil(predicate: () => boolean): Promise<void> {
  for (let turn = 0; turn < 1_000; turn += 1) {
    if (predicate()) return;
    await Promise.resolve(undefined);
  }
  throw new Error("OPT-0085 deterministic test did not reach its checkpoint");
}

async function flushMicrotasks(): Promise<void> {
  for (let turn = 0; turn < 16; turn += 1) await Promise.resolve(undefined);
}
