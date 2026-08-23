import { describe, expect, it, vi } from "vitest";

import type { AceGpuTensorPhase } from "../src/model/gpu-tensors.js";
import type {
  AcePlannerDecodeBatch,
  AcePlannerPrefillBatch,
} from "../src/runtime/planner.js";
import {
  AcePlannerGpuDeviceLostError,
  AcePlannerGpuExecutor,
  createPlannerReadbackLayout,
  planAcePlannerGpuPhaseMemory,
  reconstructAcePlannerLogits,
  type AcePlannerOpt0087InvocationDiagnostics,
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

describe("ACE planner production GPU executor", () => {
  it("accounts exact phase-sized disjoint caches and never substitutes 40960", () => {
    const prefill = prefillBatch(512);
    const memory = planAcePlannerGpuPhaseMemory(
      "reference-bf16",
      prefill,
      1_234_567,
    );
    expect(memory.allocatedCacheCapacity).toBe(512);
    expect(memory.keyValueCacheBufferCount).toBe(56);
    expect(memory.keyValueCacheBytes).toBe(234_881_024);
    expect(memory.model.cacheCapacity).toBe(512);
    expect(memory.model.kvCacheBytesPerLayer).toBe(8_388_608);
    expect(memory.arenaBufferCount).toBeGreaterThan(80);
    expect(memory.accountedGpuBytes).toBe(
      memory.residentWeightBytes + memory.arenaBytes + memory.readbackBytes,
    );
    expect(memory.readbackLayout.shards.every(
      ({ byteOffset }) => byteOffset % 256 === 0,
    )).toBe(true);
    expect(memory.readbackLayout.writeStatusByteOffset % 256).toBe(0);

    const oneMore = planAcePlannerGpuPhaseMemory(
      "reference-bf16",
      prefillBatch(513),
      1_234_567,
    );
    expect(oneMore.keyValueCacheBytes - memory.keyValueCacheBytes).toBe(458_752);
    expect(oneMore.allocatedCacheCapacity).toBe(513);
  });

  it("reconstructs row-major sharded reference and FP16 logits by token ID", () => {
    for (const profile of ["reference-bf16", "raw-fp16"] as const) {
      const memory = planAcePlannerGpuPhaseMemory(profile, prefillBatch(), 0);
      const layout = memory.readbackLayout;
      const mapped = new ArrayBuffer(layout.byteLength);
      new Uint32Array(
        mapped,
        layout.writeStatusByteOffset,
        layout.rows,
      ).fill(1);
      for (let shardIndex = 0; shardIndex < layout.shards.length; shardIndex += 1) {
        const shard = layout.shards[shardIndex]!;
        for (let row = 0; row < layout.rows; row += 1) {
          const first = row * shard.rowCount;
          const values = [
            shardIndex + row / 2 + 0.25,
            -(shardIndex + row + 0.5),
          ];
          if (profile === "reference-bf16") {
            const view = new Float32Array(
              mapped,
              shard.byteOffset,
              layout.rows * shard.rowCount,
            );
            view[first] = values[0]!;
            view[first + shard.rowCount - 1] = values[1]!;
          } else {
            const view = new Uint16Array(
              mapped,
              shard.byteOffset,
              layout.rows * shard.rowCount,
            );
            view[first] = numberToFp16Bits(values[0]!);
            view[first + shard.rowCount - 1] = numberToFp16Bits(values[1]!);
          }
        }
      }
      const logits = reconstructAcePlannerLogits(mapped, layout, profile);
      expect(logits).toHaveLength(2);
      for (let shardIndex = 0; shardIndex < layout.shards.length; shardIndex += 1) {
        const shard = layout.shards[shardIndex]!;
        for (let row = 0; row < layout.rows; row += 1) {
          expect(logits[row]![shard.firstRow]).toBe(
            shardIndex + row / 2 + 0.25,
          );
          expect(logits[row]![shard.firstRow + shard.rowCount - 1]).toBe(
            -(shardIndex + row + 0.5),
          );
        }
      }
      expect(logits[0]).toHaveLength(217_204);
    }
  });

  it("copies compact semantic shards into one ascending local domain", () => {
    const layout = createPlannerReadbackLayout(
      "reference-bf16",
      2,
      { firstTokenId: 151_669, tokenCount: 64_000 },
    );
    expect(layout).toMatchObject({
      rows: 2,
      firstTokenId: 151_669,
      tokenCount: 64_000,
      shards: [
        {
          sourceShardIndex: 3,
          firstRow: 151_669,
          destinationFirstRow: 0,
          rowCount: 44_939,
        },
        {
          sourceShardIndex: 4,
          firstRow: 196_608,
          destinationFirstRow: 44_939,
          rowCount: 19_061,
        },
      ],
    });
    expect(layout.byteLength).toBeLessThan(1_737_856);
    const mapped = new ArrayBuffer(layout.byteLength);
    new Uint32Array(mapped, layout.writeStatusByteOffset, 2).fill(1);
    for (const shard of layout.shards) {
      const values = new Float32Array(
        mapped,
        shard.byteOffset,
        2 * shard.rowCount,
      );
      for (let row = 0; row < 2; row += 1) {
        values[row * shard.rowCount] = shard.firstRow + row / 2;
        values[(row + 1) * shard.rowCount - 1] =
          shard.firstRow + shard.rowCount - 1 + row / 2;
      }
    }
    const logits = reconstructAcePlannerLogits(
      mapped,
      layout,
      "reference-bf16",
    );
    expect(logits[0]).toHaveLength(64_000);
    expect(logits[0]![0]).toBe(151_669);
    expect(logits[0]![44_938]).toBe(196_607);
    expect(logits[0]![44_939]).toBe(196_608);
    expect(logits[1]![63_999]).toBe(215_668.5);
  });

  it("clears a fresh cache, preserves it across decode, and drains singleton submits", async () => {
    const fixture = preparedFixture();
    const executor = AcePlannerGpuExecutor.fromPreparedResources(fixture.resources);
    const prefill = prefillBatch();

    const prefillLogits = await executor.prefill(prefill);
    expect(prefillLogits).toHaveLength(2);
    expect(fixture.clears).toEqual(["cache-validity", "write-status"]);
    expect(fixture.maximumOutstanding()).toBe(1);
    expect(fixture.submissions).toHaveLength(33);
    expect(fixture.submissions.every((batch) => batch.length === 1)).toBe(true);
    expect(fixture.idle).toHaveBeenCalledTimes(33);
    expect(fixture.events.at(-2)).toBe("drain");
    expect(fixture.events.at(-1)).toBe("idle");
    expect(fixture.uploads.map(({ label, byteLength }) => ({ label, byteLength })))
      .toEqual([
        { label: "token-ids", byteLength: 32 },
        { label: "valid-lengths", byteLength: 16 },
        { label: "query-positions", byteLength: 32 },
        { label: "source-validity", byteLength: 32 },
        { label: "row-start-positions", byteLength: 8 },
        { label: "cosine", byteLength: 4_096 },
        { label: "sine", byteLength: 4_096 },
        { label: "last-physical-row-indices", byteLength: 8 },
      ]);
    expect(fixture.progress.at(-1)).toMatchObject({
      completedCommandBuffers: 33,
      totalCommandBuffers: 33,
      queueDrains: 33,
      cooperativeIdleMs: 33,
      stage: "readback",
    });

    const prefillEventCount = fixture.events.length;
    await executor.decode(decodeBatch(4));
    // The prefill readback's queue-empty interval completed before decode was
    // permitted to upload/submit the next invocation.
    expect(fixture.events[prefillEventCount]).toBe("submit");
    await executor.decode(decodeBatch(5));
    expect(fixture.clears).toEqual(["cache-validity", "write-status"]);
    expect(fixture.runtime.createPlannerDispatch).toHaveBeenCalledTimes(2);
    expect(fixture.runtime.createPlannerDispatchForOpt0087).not.toHaveBeenCalled();
    expect(fixture.opt0087Diagnostics).toHaveLength(0);
    expect(fixture.copyRecords.some(({ destinationLabel }) =>
      destinationLabel === "ace-opt-0087-planner-cache-append-readback"
    )).toBe(false);
    expect(fixture.phaseCreates).toHaveLength(1);
    expect(fixture.phaseDestroys[0]).toHaveBeenCalledTimes(0);

    await executor.destroy();
    await executor.destroy();
    expect(fixture.phaseDestroys[0]).toHaveBeenCalledTimes(1);
    expect(fixture.rootDestroy).toHaveBeenCalledTimes(1);
    await expect(executor.decode(decodeBatch(6))).rejects.toMatchObject({
      name: "InvalidStateError",
    });
  });

  it("replays only a range-specialized tied head over the completed hidden rows", async () => {
    const fixture = preparedFixture();
    const executor = AcePlannerGpuExecutor.fromPreparedResources(fixture.resources);
    const range = { firstTokenId: 151_669, tokenCount: 64_000 } as const;

    await expect(executor.replayTiedHeadForOpt0082(range)).rejects.toThrow(
      /requires a completed planner call/,
    );
    await executor.prefill(prefillBatch());
    const submissionsBefore = fixture.submissions.length;
    const logits = await executor.replayTiedHeadForOpt0082(range);

    expect(logits).toHaveLength(2);
    expect(logits[0]).toHaveLength(range.tokenCount);
    expect(fixture.submissions).toHaveLength(submissionsBefore + 2);
    expect(fixture.runtime.createPlannerDispatch).toHaveBeenLastCalledWith(
      expect.stringContaining("prefill"),
      expect.anything(),
      expect.anything(),
      range,
    );
    expect(fixture.runtime.createPlannerDispatch).toHaveBeenCalledTimes(2);
    await executor.replayTiedHeadForOpt0082(range);
    expect(fixture.runtime.createPlannerDispatch).toHaveBeenCalledTimes(2);
    await executor.destroy();
  });

  it("rejects a mismatched replay dispatch before submitting benchmark work", async () => {
    const fixture = preparedFixture();
    const executor = AcePlannerGpuExecutor.fromPreparedResources(fixture.resources);
    await executor.prefill(prefillBatch());
    const submissionsBefore = fixture.submissions.length;
    fixture.runtime.createPlannerDispatch.mockImplementationOnce(async (
      _label: string,
      batch: AcePlannerPrefillBatch | AcePlannerDecodeBatch,
    ) => fakeDispatch(batch));

    await expect(executor.replayTiedHeadForOpt0082({
      firstTokenId: 151_669,
      tokenCount: 64_000,
    })).rejects.toThrow(/different logit range/);
    expect(fixture.submissions).toHaveLength(submissionsBefore);
    await executor.destroy();
  });

  it("fails compact range dispatch closed outside reference BF16", async () => {
    const fixture = preparedFixture();
    const executor = AcePlannerGpuExecutor.fromPreparedResources({
      ...fixture.resources,
      modelProfile: "raw-fp16",
    });
    await expect(executor.prefill(prefillBatch(), {
      firstTokenId: 151_669,
      tokenCount: 64_000,
    })).rejects.toThrow(/require reference-BF16/);
    expect(fixture.phaseCreates).toHaveLength(0);
    expect(fixture.submissions).toHaveLength(0);
    await executor.destroy();

    const rowFixture = preparedFixture();
    const rowExecutor = AcePlannerGpuExecutor.fromPreparedResources(
      rowFixture.resources,
    );
    await expect(rowExecutor.prefill({
      ...prefillBatch(),
      rows: 1,
    }, {
      firstTokenId: 151_669,
      tokenCount: 64_000,
    })).rejects.toThrow(/require two planner rows/);
    expect(rowFixture.phaseCreates).toHaveLength(0);
    expect(rowFixture.submissions).toHaveLength(0);
    await rowExecutor.destroy();
  });

  it("replaces the complete phase allocation on each prefill", async () => {
    const fixture = preparedFixture();
    const executor = AcePlannerGpuExecutor.fromPreparedResources(fixture.resources);
    await executor.prefill(prefillBatch(512));
    await executor.prefill(prefillBatch(513));
    expect(fixture.phaseCreates.map(({ cacheCapacity }) => cacheCapacity)).toEqual([
      512,
      513,
    ]);
    expect(fixture.phaseDestroys[0]).toHaveBeenCalledTimes(1);
    expect(fixture.phaseDestroys[1]).toHaveBeenCalledTimes(0);
    expect(fixture.clears).toEqual([
      "cache-validity",
      "write-status",
      "cache-validity",
      "write-status",
    ]);
    await executor.destroy();
    expect(fixture.phaseDestroys[1]).toHaveBeenCalledTimes(1);
  });

  it("rejects non-contiguous decode before mutating or submitting", async () => {
    const fixture = preparedFixture();
    const executor = AcePlannerGpuExecutor.fromPreparedResources(fixture.resources);
    await executor.prefill(prefillBatch());
    const submissionsBefore = fixture.submissions.length;
    await expect(executor.decode(decodeBatch(5))).rejects.toThrow(
      /resident cache ends at 4/,
    );
    expect(fixture.submissions).toHaveLength(submissionsBefore);
    await executor.destroy();
  });

  it("waits for submitted work to drain before device-loss teardown", async () => {
    const drain = Promise.withResolvers<undefined>();
    const fixture = preparedFixture({ drain });
    const executor = AcePlannerGpuExecutor.fromPreparedResources(fixture.resources);
    const running = executor.prefill(prefillBatch());
    await waitUntil(() => fixture.submissions.length === 1);
    fixture.deviceLost.resolve({
      reason: "unknown",
      message: "test planner device vanished",
    } as GPUDeviceLostInfo);
    await flushMicrotasks();
    expect(fixture.rootDestroy).not.toHaveBeenCalled();
    expect(fixture.phaseDestroys[0]).not.toHaveBeenCalled();

    drain.resolve(undefined);
    await expect(running).rejects.toBeInstanceOf(AcePlannerGpuDeviceLostError);
    await executor.destroy();
    expect(fixture.phaseDestroys[0]).toHaveBeenCalledTimes(1);
    expect(fixture.rootDestroy).toHaveBeenCalledTimes(1);
    expect(fixture.submissions).toHaveLength(1);
  });

  it("honors lifetime cancellation only after the active singleton drains", async () => {
    const controller = new AbortController();
    const drain = Promise.withResolvers<undefined>();
    const fixture = preparedFixture({ drain, signal: controller.signal });
    const executor = AcePlannerGpuExecutor.fromPreparedResources(fixture.resources);
    const running = executor.prefill(prefillBatch());
    await waitUntil(() => fixture.submissions.length === 1);
    controller.abort(new DOMException("planner request cancelled", "AbortError"));
    await flushMicrotasks();
    expect(fixture.phaseDestroys[0]).not.toHaveBeenCalled();
    expect(fixture.rootDestroy).not.toHaveBeenCalled();
    drain.resolve(undefined);
    await expect(running).rejects.toMatchObject({ name: "AbortError" });
    await executor.destroy();
    expect(fixture.phaseDestroys[0]).toHaveBeenCalledTimes(1);
    expect(fixture.rootDestroy).toHaveBeenCalledTimes(1);
  });

  it("releases transferred weights when the production factory fails closed", () => {
    const destroy = vi.fn();
    const wrongPhase = {
      phases: ["dit"],
      residentBytes: 1,
      destroy,
    } as unknown as AceGpuTensorPhase;
    expect(() => AcePlannerGpuExecutor.create({
      device: {} as GPUDevice,
      modelProfile: "reference-bf16",
      ownedPlannerWeights: wrongPhase,
    })).toThrow(/exclusively resident planner phase/);
    expect(destroy).toHaveBeenCalledTimes(1);
  });

  it.each([1, 2] as const)(
    "keeps OPT-0087 M%i A/B executor topology, cache status, and readback identical",
    async (rows) => {
      const fixture = preparedFixture({ opt0087: true });
      const executor = AcePlannerGpuExecutor.fromPreparedResources(
        fixture.resources,
      );
      const executions: Array<Readonly<{
        arm: "generic-a" | "direct-b";
        logits: readonly ArrayLike<number>[];
        submissions: number;
        idleTurns: number;
        uploads: readonly Readonly<{ label: string; byteLength: number }>[];
        copies: readonly Readonly<{
          sourceLabel: string;
          sourceOffset: number;
          destinationLabel: string;
          destinationOffset: number;
          size: number;
        }>[];
      }>> = [];
      for (const arm of ["generic-a", "direct-b"] as const) {
        await executor.prefill(prefillBatchForRows(rows));
        const submissionStart = fixture.submissions.length;
        const idleStart = fixture.idle.mock.calls.length;
        const uploadStart = fixture.uploads.length;
        const copyStart = fixture.copyRecords.length;
        const logits = await executor.decodeForOpt0087(
          arm,
          decodeBatchForRows(rows, 4),
        );
        executions.push(Object.freeze({
          arm,
          logits,
          submissions: fixture.submissions.length - submissionStart,
          idleTurns: fixture.idle.mock.calls.length - idleStart,
          uploads: Object.freeze(fixture.uploads.slice(uploadStart)),
          copies: Object.freeze(fixture.copyRecords.slice(copyStart).map((copy) => ({
            sourceLabel: copy.sourceLabel,
            sourceOffset: copy.sourceOffset,
            destinationLabel: copy.destinationLabel,
            destinationOffset: copy.destinationOffset,
            size: copy.size,
          }))),
        }));
      }

      expect(executions[0]!.logits).toEqual(executions[1]!.logits);
      expect(executions.map(({ submissions }) => submissions)).toEqual([34, 34]);
      expect(executions.map(({ idleTurns }) => idleTurns)).toEqual([34, 34]);
      expect(executions[0]!.uploads).toEqual(executions[1]!.uploads);
      expect(executions[0]!.copies).toEqual(executions[1]!.copies);
      const cacheCopies = executions[0]!.copies.filter(({ destinationLabel }) =>
        destinationLabel === "ace-opt-0087-planner-cache-append-readback"
      );
      expect(cacheCopies).toHaveLength(28 * 2 * rows * 8 + rows);
      expect(cacheCopies.slice(0, 8).map((copy) => ({
        sourceLabel: copy.sourceLabel,
        sourceOffset: copy.sourceOffset,
        destinationOffset: copy.destinationOffset,
        size: copy.size,
      }))).toEqual(Array.from({ length: 8 }, (_, head) => ({
        sourceLabel: "layer-0-key",
        sourceOffset: (head * 512 + 4) * 128 * 4,
        destinationOffset: head * 128 * 4,
        size: 128 * 4,
      })));
      const validityCopies = cacheCopies.slice(-rows);
      expect(validityCopies.map((copy, row) => ({
        sourceLabel: copy.sourceLabel,
        sourceOffset: copy.sourceOffset,
        destinationOffset: copy.destinationOffset,
        size: copy.size,
        row,
      }))).toEqual(Array.from({ length: rows }, (_, row) => ({
        sourceLabel: "cache-validity",
        sourceOffset: (row * 512 + 4) * 4,
        destinationOffset: (28 * 2 * rows * 8 * 128 + row) * 4,
        size: 4,
        row,
      })));
      expect(fixture.clears).toEqual([
        "cache-validity",
        "write-status",
        "cache-validity",
        "write-status",
      ]);
      expect(fixture.opt0087Diagnostics).toHaveLength(2);
      for (let index = 0; index < 2; index += 1) {
        expect(fixture.opt0087Diagnostics[index]).toMatchObject({
          arm: executions[index]!.arm,
          phaseKind: "decode",
          modelQuantumCount: 33,
          totalCommandBuffers: 34,
          commandBuffersSubmitted: 34,
          trueQueueDrainCount: 34,
          cooperativeIdleTurns: 34,
          requestedCooperativeIdleMs: 34,
          maximumOutstandingCommandBuffers: 1,
          readbackMapCount: 2,
          cacheAppendReadbackByteLength: rows === 1 ? 229_632 : 459_008,
          cacheAppendLogicalByteLength: rows === 1 ? 229_380 : 458_760,
          cacheAppendCopyCount: rows === 1 ? 449 : 898,
          cacheAppendKeyValueWordCount: rows === 1 ? 57_344 : 114_688,
          cacheAppendValidityWordCount: rows,
          layerQuantumCount: 28,
          tiedHeadQuantumCount: 2,
          writeStatusWords: Array<number>(rows).fill(1),
          headQuantumSliceFirstRows: [
            [0, 49_152],
            [98_304, 147_456, 196_608],
          ],
        });
        expect(fixture.opt0087Diagnostics[index]!.quantumTimings)
          .toHaveLength(33);
        expect(fixture.opt0087Diagnostics[index]!.cacheAppendWords)
          .toEqual(fixture.opt0087Diagnostics[0]!.cacheAppendWords);
      }
      expect(fixture.opt0087Dispatches.map(({ arm }) => arm)).toEqual([
        "generic-a",
        "direct-b",
      ]);
      expect(fixture.submissions.every((submission) => submission.length === 1))
        .toBe(true);
      expect(fixture.maximumOutstanding()).toBe(1);

      await executor.destroy();
      await executor.destroy();
      expect(fixture.phaseDestroys).toHaveLength(2);
      expect(fixture.phaseDestroys.every((destroy) =>
        destroy.mock.calls.length === 1
      )).toBe(true);
      expect(fixture.rootDestroy).toHaveBeenCalledTimes(1);
    },
  );

  it("fails the OPT-0087 executor arm closed outside its paired BF16 owner", async () => {
    const fixture = preparedFixture();
    const executor = AcePlannerGpuExecutor.fromPreparedResources(
      fixture.resources,
    );
    await executor.prefill(prefillBatch());
    const submissionsBefore = fixture.submissions.length;
    expect(() => executor.decodeForOpt0087(
      "direct-b",
      decodeBatch(4),
    )).toThrow(/require paired reference-BF16 diagnostics/);
    expect(fixture.submissions).toHaveLength(submissionsBefore);
    expect(fixture.runtime.createPlannerDispatchForOpt0087).not.toHaveBeenCalled();
    await executor.decode(decodeBatch(4));
    await executor.destroy();

    const invalid = preparedFixture({ opt0087: true });
    const invalidExecutor = AcePlannerGpuExecutor.fromPreparedResources(
      invalid.resources,
    );
    expect(() => invalidExecutor.decodeForOpt0087(
      "invalid" as "direct-b",
      decodeBatch(4),
    )).toThrow(/dense arm is invalid/);
    expect(invalid.phaseCreates).toHaveLength(0);
    await invalidExecutor.destroy();
  });
});

interface FixtureOptions {
  readonly drain?: PromiseWithResolvers<undefined>;
  readonly signal?: AbortSignal;
  readonly opt0087?: boolean;
}

type VoidMock = ReturnType<typeof vi.fn<() => void>>;

function preparedFixture(options: FixtureOptions = {}): Readonly<{
  resources: AcePlannerPreparedGpuExecutorResources;
  runtime: {
    createPlannerDispatch: ReturnType<typeof vi.fn>;
    createPlannerDispatchForOpt0087: ReturnType<typeof vi.fn>;
    destroy: ReturnType<typeof vi.fn>;
  };
  submissions: GPUCommandBuffer[][];
  uploads: Array<{ readonly label: string; readonly byteLength: number }>;
  clears: string[];
  idle: ReturnType<typeof vi.fn<() => Promise<void>>>;
  progress: Array<Record<string, unknown>>;
  phaseCreates: AcePlannerPrefillBatch[];
  phaseDestroys: VoidMock[];
  rootDestroy: VoidMock;
  deviceLost: PromiseWithResolvers<GPUDeviceLostInfo>;
  maximumOutstanding(): number;
  events: string[];
  copyRecords: Array<Readonly<{
    commandLabel: string;
    sourceLabel: string;
    sourceOffset: number;
    destinationLabel: string;
    destinationOffset: number;
    size: number;
  }>>;
  opt0087Diagnostics: AcePlannerOpt0087InvocationDiagnostics[];
  opt0087Dispatches: Array<Readonly<{
    arm: "generic-a" | "direct-b";
  }>>;
}> {
  const submissions: GPUCommandBuffer[][] = [];
  const uploads: Array<{ readonly label: string; readonly byteLength: number }> = [];
  const clears: string[] = [];
  const progress: Array<Record<string, unknown>> = [];
  const phaseCreates: AcePlannerPrefillBatch[] = [];
  const phaseDestroys: VoidMock[] = [];
  const deviceLost = Promise.withResolvers<GPUDeviceLostInfo>();
  const idle = vi.fn(async () => undefined);
  const events: string[] = [];
  const copyRecords: Array<Readonly<{
    commandLabel: string;
    sourceLabel: string;
    sourceOffset: number;
    destinationLabel: string;
    destinationOffset: number;
    size: number;
  }>> = [];
  const opt0087Diagnostics: AcePlannerOpt0087InvocationDiagnostics[] = [];
  const opt0087Dispatches: Array<Readonly<{
    arm: "generic-a" | "direct-b";
  }>> = [];
  let outstanding = 0;
  let maximumOutstanding = 0;
  const queue = {
    writeBuffer(
      buffer: GPUBuffer,
      _bufferOffset: number,
      _data: GPUAllowSharedBufferSource,
      _dataOffset?: number,
      size?: number,
    ): void {
      uploads.push({
        label: (buffer as unknown as { label: string }).label,
        byteLength: size ?? 0,
      });
    },
    submit(commandBuffers: Iterable<GPUCommandBuffer>): void {
      events.push("submit");
      submissions.push([...commandBuffers]);
      outstanding += 1;
      maximumOutstanding = Math.max(maximumOutstanding, outstanding);
    },
    async onSubmittedWorkDone(): Promise<undefined> {
      if (options.drain !== undefined) await options.drain.promise;
      outstanding -= 1;
      events.push("drain");
      return undefined;
    },
  } as unknown as GPUQueue;
  const device = {
    queue,
    lost: deviceLost.promise,
    createCommandEncoder({ label }: GPUCommandEncoderDescriptor = {}) {
      return {
        clearBuffer(buffer: GPUBuffer) {
          clears.push((buffer as unknown as { label: string }).label);
        },
        beginComputePass() {
          return {
            end: vi.fn(),
          } as unknown as GPUComputePassEncoder;
        },
        copyBufferToBuffer(
          source: GPUBuffer,
          sourceOffset: number,
          destination: GPUBuffer,
          destinationOffset: number,
          size: number,
        ) {
          copyRecords.push(Object.freeze({
            commandLabel: label ?? "unlabelled",
            sourceLabel: (source as unknown as { label?: string }).label ??
              "unlabelled",
            sourceOffset,
            destinationLabel:
              (destination as unknown as { label?: string }).label ??
                "unlabelled",
            destinationOffset,
            size,
          }));
        },
        finish() {
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
      logitRange?: { readonly firstTokenId: number; readonly tokenCount: number },
    ) => fakeDispatch(batch, logitRange)),
    createPlannerDispatchForOpt0087: vi.fn(async (
      _label: string,
      arm: "generic-a" | "direct-b",
      batch: AcePlannerPrefillBatch | AcePlannerDecodeBatch,
      _bindings: AcePlannerModelBindings,
    ) => {
      if (batch.kind !== "decode") {
        throw new Error("OPT-0087 fake accepts decode only");
      }
      opt0087Dispatches.push(Object.freeze({ arm }));
      return fakeOpt0087Dispatch(batch, arm);
    }),
    destroy: vi.fn(),
  };
  const rootDestroy = vi.fn(() => runtime.destroy());
  const resources: AcePlannerPreparedGpuExecutorResources = {
    device,
    modelProfile: "reference-bf16",
    runtime,
    weights: {} as AcePlannerModelWeights,
    residentWeightBytes: 1_024,
    ...(options.signal === undefined ? {} : { signal: options.signal }),
    yieldQueueIdle: async () => {
      events.push("idle");
      await idle();
    },
    onProgress: (event) => progress.push({ ...event }),
    ...(options.opt0087 === true
      ? {
          opt0087PairedOwner: true as const,
          onOpt0087Invocation: (
            diagnostics: AcePlannerOpt0087InvocationDiagnostics,
          ) => opt0087Diagnostics.push(diagnostics),
        }
      : {}),
    async createPhase(batch) {
      phaseCreates.push(batch);
      const destroy = vi.fn();
      phaseDestroys.push(destroy);
      return fakePhase(batch, destroy, options.opt0087 === true);
    },
    destroy: rootDestroy,
  };
  return Object.freeze({
    resources,
    runtime,
    submissions,
    uploads,
    clears,
    idle,
    progress,
    phaseCreates,
    phaseDestroys,
    rootDestroy,
    deviceLost,
    events,
    copyRecords,
    opt0087Diagnostics,
    opt0087Dispatches,
    maximumOutstanding: () => maximumOutstanding,
  });
}

function fakePhase(
  batch: AcePlannerPrefillBatch,
  destroy: VoidMock,
  opt0087 = false,
): AcePlannerPreparedPhaseGpuResources {
  const memory = planAcePlannerGpuPhaseMemory("reference-bf16", batch, 1_024);
  const readbackBytes = new ArrayBuffer(memory.readbackBytes);
  new Uint32Array(
    readbackBytes,
    memory.readbackLayout.writeStatusByteOffset,
    batch.rows,
  ).fill(1);
  for (const range of [
    { firstTokenId: 151_669, tokenCount: 64_000 },
    { firstTokenId: 151_645, tokenCount: 1 },
  ]) {
    const compact = createPlannerReadbackLayout(
      "reference-bf16",
      batch.rows as 1 | 2,
      range,
    );
    new Uint32Array(
      readbackBytes,
      compact.writeStatusByteOffset,
      batch.rows,
    ).fill(1);
  }
  const readback = {
    size: memory.readbackBytes,
    mapAsync: vi.fn(async () => undefined),
    getMappedRange: vi.fn(() => readbackBytes),
    unmap: vi.fn(),
  } as unknown as GPUBuffer;
  const cacheAppendKeyValueWords = 28 * 2 * batch.rows * 8 * 128;
  const cacheAppendLogicalBytes =
    (cacheAppendKeyValueWords + batch.rows) * Uint32Array.BYTES_PER_ELEMENT;
  const cacheAppendBytes = Math.ceil(cacheAppendLogicalBytes / 256) * 256;
  const cacheAppendMapped = new ArrayBuffer(cacheAppendBytes);
  new Uint32Array(
    cacheAppendMapped,
    cacheAppendKeyValueWords * Uint32Array.BYTES_PER_ELEMENT,
    batch.rows,
  ).fill(1);
  const cacheAppendReadback = {
    label: "ace-opt-0087-planner-cache-append-readback",
    size: cacheAppendBytes,
    mapAsync: vi.fn(async () => undefined),
    getMappedRange: vi.fn(() => cacheAppendMapped),
    unmap: vi.fn(),
    destroy: vi.fn(),
  } as unknown as GPUBuffer;
  return {
    batch: batch.rows as 1 | 2,
    prefillTokens: batch.tokens,
    cacheCapacity: batch.cacheCapacity,
    bindings: fakeBindings(batch),
    readback,
    ...(opt0087
      ? {
          opt0087CacheAppendReadback: {
            buffer: cacheAppendReadback,
            layout: {
              rows: batch.rows as 1 | 2,
              keyValueWordCount: cacheAppendKeyValueWords,
              validityWordCount: batch.rows,
              logicalByteLength: cacheAppendLogicalBytes,
              byteLength: cacheAppendBytes,
              copyCount: 28 * 2 * batch.rows * 8 + batch.rows,
            },
          },
        }
      : {}),
    memory,
    destroy,
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
      cacheValidity: binding(
        "cache-validity",
        batch.rows * batch.cacheCapacity * 4,
      ),
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
      binding(`logits-${index}`, batch.rows * part.rowCount * 4)
    ),
  };
}

function fakeDispatch(
  batch: AcePlannerPrefillBatch | AcePlannerDecodeBatch,
  logitRange?: { readonly firstTokenId: number; readonly tokenCount: number },
): AcePlannerModelDispatch {
  const quanta: AcePlannerModelQuantum[] = [fakeQuantum("embedding", null)];
  for (let layer = 0; layer < 28; layer += 1) {
    quanta.push(fakeQuantum("layer", layer));
  }
  quanta.push(fakeQuantum("final-norm", null));
  quanta.push(fakeQuantum("last-row-gather", null));
  quanta.push(fakeQuantum("tied-lm-head", null));
  return {
    label: `fake-${batch.kind}`,
    plan: planAcePlannerModel("reference-bf16", batch),
    logitRange: logitRange === undefined ? null : logitRange,
    headSlices: planAcePlannerHeadSlices(logitRange),
    primitiveCount: quanta.length,
    quanta,
    encode() {},
  };
}

function fakeOpt0087Dispatch(
  batch: AcePlannerDecodeBatch,
  arm: "generic-a" | "direct-b",
): AcePlannerModelDispatch {
  const base = fakeDispatch(batch);
  const last = base.quanta.at(-1)!;
  const quanta = Object.freeze([
    ...base.quanta.slice(0, -1),
    { ...last, id: `${last.id}-0`, primitiveCount: 2 },
    { ...last, id: `${last.id}-1`, primitiveCount: 3 },
  ]);
  const binding = { buffer: {} as GPUBuffer, offset: 0, size: 4 };
  return Object.freeze({
    ...base,
    quanta,
    primitiveCount: base.primitiveCount + 1,
    opt0087DenseSelections: Object.freeze([Object.freeze({
      label: "fake-query-projection",
      role: "query-projection" as const,
      requestedArm: arm,
      selectedArm: arm,
      reason: arm === "generic-a"
        ? "control-requested" as const
        : "direct-selected" as const,
      shape: Object.freeze({ rows: batch.rows, inner: 1_024, columns: 2_048 }),
      weightBinding: binding,
      activationBinding: binding,
      outputBinding: binding,
    })]),
    opt0087HeadQuantumSliceFirstRows: Object.freeze([
      Object.freeze([0, 49_152]),
      Object.freeze([98_304, 147_456, 196_608]),
    ]),
  });
}

function fakeQuantum(
  kind: AcePlannerModelQuantum["kind"],
  layer: number | null,
): AcePlannerModelQuantum {
  return {
    id: `fake-${kind}-${layer ?? "shared"}`,
    kind,
    layer,
    primitiveCount: 1,
    encode: vi.fn(),
  };
}

function prefillBatch(cacheCapacity = 512): AcePlannerPrefillBatch {
  const queryPositions = new Uint32Array([
    0, 1, 2, 3,
    0, 1, 2, 3,
  ]);
  const sourceValidity = new Uint32Array([
    0, 0, 1, 1,
    0, 1, 1, 1,
  ]);
  return {
    kind: "prefill",
    rows: 2,
    tokens: 4,
    cacheCapacity,
    inputIds: new Uint32Array([
      151_643, 151_643, 10, 11,
      151_643, 20, 21, 22,
    ]),
    keyValidity: sourceValidity.slice(),
    rotaryPositionIds: queryPositions.slice(),
    causal: {
      rowStartPositions: new Uint32Array([0, 0]),
      validLengths: new Uint32Array([4, 4, 4, 4]),
      queryPositions,
      sourceValidity,
    },
    conditionalRow: 0,
    unconditionalRow: 1,
  };
}

function prefillBatchForRows(rows: 1 | 2): AcePlannerPrefillBatch {
  const batch = prefillBatch();
  if (rows === 2) return batch;
  return Object.freeze({
    ...batch,
    rows: 1 as const,
    inputIds: batch.inputIds.slice(0, batch.tokens),
    keyValidity: batch.keyValidity.slice(0, batch.tokens),
    rotaryPositionIds: batch.rotaryPositionIds.slice(0, batch.tokens),
    causal: Object.freeze({
      rowStartPositions: batch.causal.rowStartPositions.slice(0, 1),
      validLengths: batch.causal.validLengths.slice(0, 2),
      queryPositions: batch.causal.queryPositions.slice(0, batch.tokens),
      sourceValidity: batch.causal.sourceValidity.slice(0, batch.tokens),
    }),
    unconditionalRow: null,
  });
}

function decodeBatch(cachedTokensBeforeAppend: number): AcePlannerDecodeBatch {
  return {
    kind: "decode",
    rows: 2,
    tokens: 1,
    cacheCapacity: 512,
    cachedTokensBeforeAppend,
    inputIds: new Uint32Array([42, 42]),
    rotaryPositionIds: new Uint32Array([
      cachedTokensBeforeAppend,
      cachedTokensBeforeAppend,
    ]),
    causal: {
      rowStartPositions: new Uint32Array([
        cachedTokensBeforeAppend,
        cachedTokensBeforeAppend,
      ]),
      validLengths: new Uint32Array([
        1,
        cachedTokensBeforeAppend + 1,
        1,
        cachedTokensBeforeAppend + 1,
      ]),
      queryPositions: new Uint32Array([
        cachedTokensBeforeAppend,
        cachedTokensBeforeAppend,
      ]),
      sourceValidity: new Uint32Array([1, 1]),
    },
    conditionalRow: 0,
    unconditionalRow: 1,
  };
}

function decodeBatchForRows(
  rows: 1 | 2,
  cachedTokensBeforeAppend: number,
): AcePlannerDecodeBatch {
  const batch = decodeBatch(cachedTokensBeforeAppend);
  if (rows === 2) return batch;
  return Object.freeze({
    ...batch,
    rows: 1 as const,
    inputIds: batch.inputIds.slice(0, 1),
    rotaryPositionIds: batch.rotaryPositionIds.slice(0, 1),
    causal: Object.freeze({
      rowStartPositions: batch.causal.rowStartPositions.slice(0, 1),
      validLengths: batch.causal.validLengths.slice(0, 2),
      queryPositions: batch.causal.queryPositions.slice(0, 1),
      sourceValidity: batch.causal.sourceValidity.slice(0, 1),
    }),
    unconditionalRow: null,
  });
}

const FLOAT_BUFFER = new ArrayBuffer(4);
const FLOAT_F32 = new Float32Array(FLOAT_BUFFER);
const FLOAT_U32 = new Uint32Array(FLOAT_BUFFER);

function numberToFp16Bits(value: number): number {
  FLOAT_F32[0] = value;
  const bits = FLOAT_U32[0]!;
  const sign = (bits >>> 16) & 0x8000;
  const exponent = (bits >>> 23) & 0xff;
  const mantissa = bits & 0x7f_ffff;
  if (exponent === 0xff) return sign | (mantissa === 0 ? 0x7c00 : 0x7e00);
  const halfExponent = exponent - 127 + 15;
  if (halfExponent >= 0x1f) return sign | 0x7c00;
  if (halfExponent <= 0) {
    if (halfExponent < -10) return sign;
    const normalized = mantissa | 0x80_0000;
    const shift = 14 - halfExponent;
    const truncated = normalized >>> shift;
    const remainder = normalized & ((1 << shift) - 1);
    const halfway = 1 << (shift - 1);
    return sign | (truncated +
      (remainder > halfway || (remainder === halfway && (truncated & 1)) ? 1 : 0));
  }
  let halfMantissa = mantissa >>> 13;
  const remainder = mantissa & 0x1fff;
  if (remainder > 0x1000 || (remainder === 0x1000 && (halfMantissa & 1))) {
    halfMantissa += 1;
    if (halfMantissa === 0x400) {
      const nextExponent = halfExponent + 1;
      return sign | (nextExponent >= 0x1f ? 0x7c00 : nextExponent << 10);
    }
  }
  return sign | (halfExponent << 10) | halfMantissa;
}

async function waitUntil(predicate: () => boolean): Promise<void> {
  for (let attempt = 0; attempt < 100; attempt += 1) {
    if (predicate()) return;
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
  throw new Error("Timed out waiting for planner fixture state");
}

async function flushMicrotasks(): Promise<void> {
  await Promise.resolve();
  await Promise.resolve();
  await Promise.resolve();
}
