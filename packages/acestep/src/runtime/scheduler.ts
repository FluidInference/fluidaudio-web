export const ACE_COOPERATIVE_GPU_IDLE_MILLISECONDS = 1;

export interface AceGpuSchedulingProgress {
  readonly completedCommandBuffers: number;
  readonly totalCommandBuffers: number;
  readonly queueDrains: number;
  readonly cooperativeIdleMs: number;
}

export interface AceGpuSchedulingResult {
  readonly commandBuffersSubmitted: number;
  readonly queueDrains: number;
  readonly cooperativeIdleMs: number;
}

export interface AceGpuCommandBufferDrainTiming {
  readonly commandBufferIndex: number;
  /** Wall time from immediately before submit through the matching drain. */
  readonly submitThroughDrainMs: number;
}

export interface AceCooperativeSubmissionOptions {
  readonly queue: Pick<GPUQueue, "submit" | "onSubmittedWorkDone">;
  readonly commandBuffers: readonly GPUCommandBuffer[];
  readonly signal: AbortSignal;
  readonly yieldQueueIdle?: () => Promise<void>;
  readonly onProgress?: (progress: AceGpuSchedulingProgress) => void;
  readonly onCommandBufferDrained?: (
    timing: AceGpuCommandBufferDrainTiming,
  ) => void;
  /** @internal Deterministic timing seam; production uses performance.now. */
  readonly now?: () => number;
}

export interface AceCooperativeLazySubmissionOptions {
  readonly queue: Pick<GPUQueue, "submit" | "onSubmittedWorkDone">;
  readonly commandBufferCount: number;
  /** Encode exactly one command buffer immediately before its submission. */
  readonly createCommandBuffer: (index: number) => GPUCommandBuffer;
  readonly signal: AbortSignal;
  readonly yieldQueueIdle?: () => Promise<void>;
  readonly onProgress?: (progress: AceGpuSchedulingProgress) => void;
  readonly onCommandBufferDrained?: (
    timing: AceGpuCommandBufferDrainTiming,
  ) => void;
  /** @internal Deterministic timing seam; production uses performance.now. */
  readonly now?: () => number;
}

/**
 * OPT-0080 completion timing. A cumulative completion fence is not a queue
 * drain while a younger singleton command buffer remains submitted.
 */
export interface AceDepth2Epoch4CommandBufferCompletionTiming {
  readonly commandBufferIndex: number;
  readonly submitThroughCompletionFenceMs: number;
  readonly trueQueueDrain: boolean;
  readonly completionEpochIndex: number;
}

export interface AceDepth2Epoch4SchedulingProgress {
  readonly completedCommandBuffers: number;
  readonly totalCommandBuffers: number;
  readonly completionFenceRequestedCount: number;
  readonly completionFenceSettledCount: number;
  readonly completionFenceRejectedCount: number;
  readonly trueQueueDrainCount: number;
  readonly completionEpochCount: number;
  readonly requestedCooperativeIdleMs: number;
  readonly cooperativeIdleTurns: number;
  readonly outstandingCommandBuffers: number;
}

export interface AceDepth2Epoch4CompletionEpochPlan {
  readonly completionEpochIndex: number;
  readonly phaseIndex: number;
  readonly firstCommandBufferIndex: number;
  readonly lastCommandBufferIndex: number;
  readonly commandBufferCount: number;
}

export interface AceDepth2Epoch4CompletionEpochTiming
  extends AceDepth2Epoch4CompletionEpochPlan {
  readonly submitThroughTrueDrainMs: number;
}

export interface AceDepth2Epoch4PhaseStart {
  readonly phaseIndex: number;
  readonly firstCommandBufferIndex: number;
  readonly commandBufferCount: number;
}

export interface AceDepth2Epoch4SchedulingResult extends AceGpuSchedulingResult {
  readonly completionFenceRequestedCount: number;
  readonly completionFenceSettledCount: number;
  readonly completionFenceRejectedCount: number;
  readonly trueQueueDrainCount: number;
  readonly completionEpochCount: number;
  readonly requestedCooperativeIdleMs: number;
  readonly cooperativeIdleTurns: number;
  readonly maximumOutstandingCommandBuffers: number;
}

/**
 * Depth-two/four-completion scheduling inputs. The concrete device is required
 * so a rejected terminal recovery fence can force and then confirm device loss
 * before aliased graph storage is released.
 */
export interface AceDepth2Epoch4LazySubmissionOptions {
  readonly queue: Pick<GPUQueue, "submit" | "onSubmittedWorkDone">;
  readonly commandBufferCount: number;
  readonly phaseCommandBufferCounts: readonly number[];
  readonly createCommandBuffer: (index: number) => GPUCommandBuffer;
  readonly signal: AbortSignal;
  readonly device: Pick<GPUDevice, "destroy" | "lost">;
  readonly yieldQueueIdle?: () => Promise<void>;
  readonly onPhaseStarted?: (phase: AceDepth2Epoch4PhaseStart) => void;
  readonly onCommandBufferCompleted?: (
    timing: AceDepth2Epoch4CommandBufferCompletionTiming,
    progress: AceDepth2Epoch4SchedulingProgress,
  ) => void;
  readonly onCompletionEpochDrained?: (
    timing: AceDepth2Epoch4CompletionEpochTiming,
  ) => void;
  /** @internal Deterministic timing seam; production uses performance.now. */
  readonly now?: () => number;
}

export interface AceGraphLease {
  readonly sequence: number;
  release(): void;
}

interface PendingLease {
  readonly sequence: number;
  readonly signal: AbortSignal | undefined;
  readonly resolve: (lease: AceGraphLease) => void;
  readonly reject: (error: unknown) => void;
  readonly onAbort: () => void;
}

/**
 * One FIFO owner for aliased graph storage. A granted lease is never revoked:
 * cancellation must first drain any submitted GPU work and then release it.
 *
 * The design follows Parakeet's ExclusiveAsyncGate at pinned commit
 * 7ee112738262a6f5a0efd2f150748a4087432fbb, with explicit waiting-abort and
 * disposal behavior added for ACE's longer multi-phase graphs.
 */
export class AceFifoGraphOwner {
  private readonly waiting: PendingLease[] = [];
  private active = false;
  private disposed = false;
  private nextSequence = 1;
  private disposal: PromiseWithResolvers<void> | undefined;

  acquire(signal?: AbortSignal): Promise<AceGraphLease> {
    if (this.disposed) return Promise.reject(disposedError());
    if (signal?.aborted === true) return Promise.reject(signal.reason);
    return new Promise<AceGraphLease>((resolve, reject) => {
      const sequence = this.nextSequence;
      this.nextSequence += 1;
      const pending: PendingLease = {
        sequence,
        signal,
        resolve,
        reject,
        onAbort: () => {
          const index = this.waiting.indexOf(pending);
          if (index < 0) return;
          this.waiting.splice(index, 1);
          signal?.removeEventListener("abort", pending.onAbort);
          reject(signal?.reason);
        },
      };
      signal?.addEventListener("abort", pending.onAbort, { once: true });
      this.waiting.push(pending);
      this.grantNext();
    });
  }

  /** Reject queued owners and resolve after the current owner releases. */
  dispose(): Promise<void> {
    if (this.disposal !== undefined) return this.disposal.promise;
    this.disposed = true;
    this.disposal = Promise.withResolvers<void>();
    for (const pending of this.waiting.splice(0)) {
      pending.signal?.removeEventListener("abort", pending.onAbort);
      pending.reject(disposedError());
    }
    if (!this.active) this.disposal.resolve();
    return this.disposal.promise;
  }

  private grantNext(): void {
    if (this.active || this.disposed) return;
    const pending = this.waiting.shift();
    if (pending === undefined) return;
    pending.signal?.removeEventListener("abort", pending.onAbort);
    if (pending.signal?.aborted === true) {
      pending.reject(pending.signal.reason);
      this.grantNext();
      return;
    }
    this.active = true;
    let released = false;
    pending.resolve({
      sequence: pending.sequence,
      release: () => {
        if (released) return;
        released = true;
        this.active = false;
        if (this.disposed) {
          this.disposal?.resolve();
        } else {
          this.grantNext();
        }
      },
    });
  }
}

/**
 * Queue-drained production submission. Every queue submission contains exactly
 * one command buffer. Every buffer, including the final one, is drained before
 * this function resolves; each non-final drain is followed by a real 1 ms
 * queue-empty interval.
 */
export async function submitAceCommandBuffersCooperatively(
  options: AceCooperativeSubmissionOptions,
): Promise<AceGpuSchedulingResult> {
  return await submitAceCommandBufferFactoriesCooperatively({
    queue: options.queue,
    commandBufferCount: options.commandBuffers.length,
    createCommandBuffer: (index) => options.commandBuffers[index]!,
    signal: options.signal,
    ...(options.yieldQueueIdle === undefined
      ? {}
      : { yieldQueueIdle: options.yieldQueueIdle }),
    ...(options.onProgress === undefined
      ? {}
      : { onProgress: options.onProgress }),
    ...(options.onCommandBufferDrained === undefined
      ? {}
      : { onCommandBufferDrained: options.onCommandBufferDrained }),
    ...(options.now === undefined ? {} : { now: options.now }),
  });
}

/**
 * Lazy cooperative submission for very large graphs. The next command buffer
 * is not encoded until the current one has fully drained and completed its
 * required queue-empty interval.
 */
export async function submitAceCommandBufferFactoriesCooperatively(
  options: AceCooperativeLazySubmissionOptions,
): Promise<AceGpuSchedulingResult> {
  if (
    !Number.isSafeInteger(options.commandBufferCount) ||
    options.commandBufferCount < 1
  ) {
    throw new RangeError("At least one GPU command buffer is required");
  }
  const yieldQueueIdle = options.yieldQueueIdle ?? yieldCooperativeGpuIdle;
  const now = options.now ?? (() => performance.now());
  let queueDrains = 0;
  let cooperativeIdleMs = 0;
  for (let index = 0; index < options.commandBufferCount; index += 1) {
    options.signal.throwIfAborted();
    const commandBuffer = options.createCommandBuffer(index);
    const submittedAt = options.onCommandBufferDrained === undefined
      ? 0
      : now();
    options.queue.submit([commandBuffer]);
    await options.queue.onSubmittedWorkDone();
    const submitThroughDrainMs = options.onCommandBufferDrained === undefined
      ? 0
      : nonnegativeTimingElapsed(now(), submittedAt);
    queueDrains += 1;
    options.signal.throwIfAborted();

    const final = index === options.commandBufferCount - 1;
    if (final) {
      options.onCommandBufferDrained?.(Object.freeze({
        commandBufferIndex: index,
        submitThroughDrainMs,
      }));
      options.onProgress?.({
        completedCommandBuffers: index + 1,
        totalCommandBuffers: options.commandBufferCount,
        queueDrains,
        cooperativeIdleMs,
      });
      continue;
    }

    // Start the timer before progress reporting. Even a throwing callback must
    // not collapse the queue-empty interval into an immediate next operation.
    const idleCompletion = yieldQueueIdle();
    cooperativeIdleMs += ACE_COOPERATIVE_GPU_IDLE_MILLISECONDS;
    try {
      options.onCommandBufferDrained?.(Object.freeze({
        commandBufferIndex: index,
        submitThroughDrainMs,
      }));
      options.onProgress?.({
        completedCommandBuffers: index + 1,
        totalCommandBuffers: options.commandBufferCount,
        queueDrains,
        cooperativeIdleMs,
      });
    } catch (error) {
      await idleCompletion;
      throw error;
    }
    await idleCompletion;
    options.signal.throwIfAborted();
  }
  return {
    commandBuffersSubmitted: options.commandBufferCount,
    queueDrains,
    cooperativeIdleMs,
  };
}

/** Build the fixed depth-two/four-completion OPT-0080 phase-aligned epochs. */
export function planAceDepth2Epoch4CompletionEpochs(
  commandBufferCount: number,
  phaseCommandBufferCounts: readonly number[],
): readonly AceDepth2Epoch4CompletionEpochPlan[] {
  assertPositiveSafeInteger(
    commandBufferCount,
    "At least one GPU command buffer is required",
  );
  if (phaseCommandBufferCounts.length === 0) {
    throw new RangeError("At least one OPT-0080 scheduling phase is required");
  }
  let total = 0;
  for (
    let phaseIndex = 0;
    phaseIndex < phaseCommandBufferCounts.length;
    phaseIndex += 1
  ) {
    const phaseCount = phaseCommandBufferCounts[phaseIndex]!;
    assertPositiveSafeInteger(
      phaseCount,
      `OPT-0080 phase ${phaseIndex} must contain at least one command buffer`,
    );
    const phaseFirst = total;
    const phaseLastExclusive = phaseFirst + phaseCount;
    if (!Number.isSafeInteger(phaseLastExclusive)) {
      throw new RangeError("OPT-0080 phase command-buffer total is unsafe");
    }
    total = phaseLastExclusive;
  }
  if (total !== commandBufferCount) {
    throw new RangeError(
      `OPT-0080 phase total ${total} does not match command-buffer count ${commandBufferCount}`,
    );
  }

  total = 0;
  const plans: AceDepth2Epoch4CompletionEpochPlan[] = [];
  for (
    let phaseIndex = 0;
    phaseIndex < phaseCommandBufferCounts.length;
    phaseIndex += 1
  ) {
    const phaseCount = phaseCommandBufferCounts[phaseIndex]!;
    const phaseFirst = total;
    const phaseLastExclusive = phaseFirst + phaseCount;
    for (
      let firstCommandBufferIndex = phaseFirst;
      firstCommandBufferIndex < phaseLastExclusive;
      firstCommandBufferIndex += 4
    ) {
      const epochCommandBufferCount = Math.min(
        4,
        phaseLastExclusive - firstCommandBufferIndex,
      );
      plans.push(Object.freeze({
        completionEpochIndex: plans.length,
        phaseIndex,
        firstCommandBufferIndex,
        lastCommandBufferIndex:
          firstCommandBufferIndex + epochCommandBufferCount - 1,
        commandBufferCount: epochCommandBufferCount,
      }));
    }
    total = phaseLastExclusive;
  }
  return Object.freeze(plans);
}

type AceSettledCompletionFence = Readonly<
  | {
    status: "fulfilled";
    settledAt: number | undefined;
    timingFailure: unknown | undefined;
  }
  | {
    status: "rejected";
    reason: unknown;
  }
>;

interface AcePendingDepth2CommandBuffer {
  readonly commandBufferIndex: number;
  readonly submittedAt: number;
  readonly outcome: Promise<AceSettledCompletionFence>;
}

/**
 * The OPT-0080 scheduler preserves singleton command buffers and one cumulative
 * fence per submit, but permits one FIFO successor to be in flight. Four-command,
 * phase-aligned epochs force a real true drain and idle.
 */
export async function submitAceCommandBufferFactoriesDepth2Epoch4(
  options: AceDepth2Epoch4LazySubmissionOptions,
): Promise<AceDepth2Epoch4SchedulingResult> {
  const epochs = planAceDepth2Epoch4CompletionEpochs(
    options.commandBufferCount,
    options.phaseCommandBufferCounts,
  );
  const yieldQueueIdle = options.yieldQueueIdle ?? yieldCooperativeGpuIdle;
  const now = options.now ?? (() => performance.now());
  const pending: AcePendingDepth2CommandBuffer[] = [];
  const secondaryFailures: unknown[] = [];
  let commandBuffersSubmitted = 0;
  let completionFenceRequestedCount = 0;
  let completionFenceSettledCount = 0;
  let completionFenceRejectedCount = 0;
  let trueQueueDrainCount = 0;
  let completionEpochCount = 0;
  let requestedCooperativeIdleMs = 0;
  let cooperativeIdleTurns = 0;
  let completedCommandBuffers = 0;
  let maximumOutstandingCommandBuffers = 0;
  let terminalQueueEmptyConfirmed = false;
  let uncapturedSubmission = false;
  let activePhaseIndex = -1;
  let asynchronousFenceFailure: Readonly<{ reason: unknown }> | undefined;

  const wrapFenceImmediately = (
    fence: Promise<void>,
  ): Promise<AceSettledCompletionFence> => Promise.resolve(fence).then(
    () => {
      completionFenceSettledCount += 1;
      try {
        return Object.freeze({
          status: "fulfilled" as const,
          settledAt: now(),
          timingFailure: undefined,
        });
      } catch (timingFailure) {
        return Object.freeze({
          status: "fulfilled" as const,
          settledAt: undefined,
          timingFailure,
        });
      }
    },
    (reason: unknown) => {
      completionFenceSettledCount += 1;
      completionFenceRejectedCount += 1;
      asynchronousFenceFailure ??= Object.freeze({ reason });
      return Object.freeze({ status: "rejected" as const, reason });
    },
  );

  const submitOne = (commandBufferIndex: number): number => {
    options.signal.throwIfAborted();
    const commandBuffer = options.createCommandBuffer(commandBufferIndex);
    options.signal.throwIfAborted();
    const submittedAt = now();
    options.queue.submit([commandBuffer]);
    commandBuffersSubmitted += 1;
    terminalQueueEmptyConfirmed = false;
    let fence: Promise<void>;
    try {
      fence = options.queue.onSubmittedWorkDone();
    } catch (error) {
      uncapturedSubmission = true;
      throw error;
    }
    completionFenceRequestedCount += 1;
    pending.push({
      commandBufferIndex,
      submittedAt,
      outcome: wrapFenceImmediately(fence),
    });
    maximumOutstandingCommandBuffers = Math.max(
      maximumOutstandingCommandBuffers,
      pending.length,
    );
    if (pending.length > 2) {
      throw new Error("OPT-0080 exceeded two outstanding command buffers");
    }
    return submittedAt;
  };

  try {
    for (const epoch of epochs) {
      if (epoch.phaseIndex !== activePhaseIndex) {
        options.signal.throwIfAborted();
        activePhaseIndex = epoch.phaseIndex;
        const phaseCommandBufferCount = options.phaseCommandBufferCounts[
          activePhaseIndex
        ]!;
        options.onPhaseStarted?.(Object.freeze({
          phaseIndex: activePhaseIndex,
          firstCommandBufferIndex: epoch.firstCommandBufferIndex,
          commandBufferCount: phaseCommandBufferCount,
        }));
        options.signal.throwIfAborted();
      }

      let nextCommandBufferIndex = epoch.firstCommandBufferIndex;
      const epochLastExclusive = epoch.lastCommandBufferIndex + 1;
      let epochSubmittedAt: number | undefined;
      while (
        nextCommandBufferIndex < epochLastExclusive &&
        pending.length < 2
      ) {
        const submittedAt = submitOne(nextCommandBufferIndex);
        epochSubmittedAt ??= submittedAt;
        nextCommandBufferIndex += 1;
      }

      while (pending.length !== 0) {
        const oldest = pending[0]!;
        const outcome = await oldest.outcome;
        // A younger wrapped fence can reject before the oldest FIFO fence is
        // observed. Stop before attribution, progress, or backfill as soon as
        // that asynchronous rejection is visible.
        if (asynchronousFenceFailure !== undefined) {
          throw asynchronousFenceFailure.reason;
        }
        if (outcome.status === "rejected") throw outcome.reason;

        pending.shift();
        const trueQueueDrain =
          oldest.commandBufferIndex === epoch.lastCommandBufferIndex;
        if (trueQueueDrain) {
          terminalQueueEmptyConfirmed = true;
          trueQueueDrainCount += 1;
          completionEpochCount += 1;
        }
        if (outcome.timingFailure !== undefined) {
          throw outcome.timingFailure;
        }
        const settledAt = outcome.settledAt!;
        const submitThroughCompletionFenceMs = nonnegativeTimingElapsed(
          settledAt,
          oldest.submittedAt,
        );
        completedCommandBuffers += 1;
        options.signal.throwIfAborted();
        options.onCommandBufferCompleted?.(Object.freeze({
          commandBufferIndex: oldest.commandBufferIndex,
          submitThroughCompletionFenceMs,
          trueQueueDrain,
          completionEpochIndex: epoch.completionEpochIndex,
        }), Object.freeze({
          completedCommandBuffers,
          totalCommandBuffers: options.commandBufferCount,
          completionFenceRequestedCount,
          completionFenceSettledCount,
          completionFenceRejectedCount,
          trueQueueDrainCount,
          completionEpochCount,
          requestedCooperativeIdleMs,
          cooperativeIdleTurns,
          outstandingCommandBuffers: pending.length,
        }));
        options.signal.throwIfAborted();

        if (trueQueueDrain) {
          if (epochSubmittedAt === undefined) {
            throw new Error("OPT-0080 completion epoch was never submitted");
          }
          options.onCompletionEpochDrained?.(Object.freeze({
            ...epoch,
            submitThroughTrueDrainMs: nonnegativeTimingElapsed(
              settledAt,
              epochSubmittedAt,
            ),
          }));
          options.signal.throwIfAborted();
          continue;
        }

        if (nextCommandBufferIndex < epochLastExclusive) {
          submitOne(nextCommandBufferIndex);
          nextCommandBufferIndex += 1;
        }
      }

      const finalEpoch = epoch.completionEpochIndex === epochs.length - 1;
      if (!finalEpoch) {
        options.signal.throwIfAborted();
        cooperativeIdleTurns += 1;
        requestedCooperativeIdleMs += ACE_COOPERATIVE_GPU_IDLE_MILLISECONDS;
        await yieldQueueIdle();
        options.signal.throwIfAborted();
      }
    }
  } catch (error) {
    try {
      await settleDepth2Failure(
        options,
        pending,
        uncapturedSubmission,
        terminalQueueEmptyConfirmed,
        commandBuffersSubmitted !== 0,
        error,
        secondaryFailures,
      );
    } catch (cleanupFailure) {
      appendSecondaryFailure(error, secondaryFailures, cleanupFailure);
    }
    retainSecondarySchedulingFailures(error, secondaryFailures);
    throw error;
  }

  if (
    commandBuffersSubmitted !== options.commandBufferCount ||
    completionFenceRequestedCount !== options.commandBufferCount ||
    completionFenceSettledCount !== options.commandBufferCount ||
    completionFenceRejectedCount !== 0 ||
    completedCommandBuffers !== options.commandBufferCount ||
    completionEpochCount !== epochs.length ||
    trueQueueDrainCount !== epochs.length ||
    pending.length !== 0
  ) {
    throw new Error("OPT-0080 scheduling diagnostics did not reconcile");
  }

  return Object.freeze({
    commandBuffersSubmitted,
    queueDrains: trueQueueDrainCount,
    cooperativeIdleMs: requestedCooperativeIdleMs,
    completionFenceRequestedCount,
    completionFenceSettledCount,
    completionFenceRejectedCount,
    trueQueueDrainCount,
    completionEpochCount,
    requestedCooperativeIdleMs,
    cooperativeIdleTurns,
    maximumOutstandingCommandBuffers,
  });
}

export interface AceRunGpuGraphOptions extends AceCooperativeSubmissionOptions {
  readonly ownerSignal?: AbortSignal;
}

export interface AceRunLazyGpuGraphOptions
  extends AceCooperativeLazySubmissionOptions {
  readonly ownerSignal?: AbortSignal;
}

export interface AceRunLazyDepth2Epoch4GpuGraphOptions
  extends AceDepth2Epoch4LazySubmissionOptions {
  readonly ownerSignal?: AbortSignal;
}

/** Own shared graph state from before the first submit through the final drain. */
export class AceCooperativeGpuScheduler {
  private readonly owner = new AceFifoGraphOwner();

  async run(options: AceRunGpuGraphOptions): Promise<AceGpuSchedulingResult> {
    const lease = await this.owner.acquire(options.ownerSignal ?? options.signal);
    try {
      return await submitAceCommandBuffersCooperatively(options);
    } finally {
      lease.release();
    }
  }

  async runLazy(
    options: AceRunLazyGpuGraphOptions,
  ): Promise<AceGpuSchedulingResult> {
    const lease = await this.owner.acquire(options.ownerSignal ?? options.signal);
    try {
      return await submitAceCommandBufferFactoriesCooperatively(options);
    } finally {
      lease.release();
    }
  }

  /** OPT-0080-approved depth-two/four-completion scheduling path. */
  async runLazyDepth2Epoch4(
    options: AceRunLazyDepth2Epoch4GpuGraphOptions,
  ): Promise<AceDepth2Epoch4SchedulingResult> {
    const lease = await this.owner.acquire(options.ownerSignal ?? options.signal);
    try {
      return await submitAceCommandBufferFactoriesDepth2Epoch4(options);
    } finally {
      lease.release();
    }
  }

  dispose(): Promise<void> {
    return this.owner.dispose();
  }
}

function yieldCooperativeGpuIdle(): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(resolve, ACE_COOPERATIVE_GPU_IDLE_MILLISECONDS);
  });
}

async function settleDepth2Failure(
  options: AceDepth2Epoch4LazySubmissionOptions,
  pending: AcePendingDepth2CommandBuffer[],
  uncapturedSubmission: boolean,
  terminalQueueEmptyConfirmed: boolean,
  hasSubmittedWork: boolean,
  primaryFailure: unknown,
  secondaryFailures: unknown[],
): Promise<void> {
  let terminalConfirmed = terminalQueueEmptyConfirmed;
  const outcomes = await Promise.all(pending.map(({ outcome }) => outcome));
  const youngest = outcomes.at(-1);
  for (const outcome of outcomes) {
    if (outcome.status === "rejected") {
      appendSecondaryFailure(
        primaryFailure,
        secondaryFailures,
        outcome.reason,
      );
    } else if (outcome.timingFailure !== undefined) {
      appendSecondaryFailure(
        primaryFailure,
        secondaryFailures,
        outcome.timingFailure,
      );
    }
  }
  if (!uncapturedSubmission && youngest?.status === "fulfilled") {
    // No backfill occurs after entering this cleanup path. The youngest
    // cumulative fence therefore confirms that all submitted work is done.
    terminalConfirmed = true;
  }
  pending.splice(0);

  if (hasSubmittedWork && (uncapturedSubmission || !terminalConfirmed)) {
    let recoveryFence: Promise<void> | undefined;
    try {
      recoveryFence = options.queue.onSubmittedWorkDone();
    } catch (recoveryCaptureFailure) {
      appendSecondaryFailure(
        primaryFailure,
        secondaryFailures,
        recoveryCaptureFailure,
      );
    }
    if (recoveryFence !== undefined) {
      const recoveryOutcome = await Promise.resolve(recoveryFence).then(
        () => Object.freeze({ status: "fulfilled" as const }),
        (reason: unknown) =>
          Object.freeze({ status: "rejected" as const, reason }),
      );
      if (recoveryOutcome.status === "fulfilled") {
        terminalConfirmed = true;
      } else {
        appendSecondaryFailure(
          primaryFailure,
          secondaryFailures,
          recoveryOutcome.reason,
        );
      }
    }
  }

  if (hasSubmittedWork && !terminalConfirmed) {
    // A failed terminal recovery fence is not enough to release aliased graph
    // storage. Force loss of the concrete device, then wait for WebGPU's
    // independently resolved terminal lifetime signal.
    try {
      options.device.destroy();
    } catch (deviceDestroyFailure) {
      appendSecondaryFailure(
        primaryFailure,
        secondaryFailures,
        deviceDestroyFailure,
      );
    }
    try {
      await options.device.lost;
    } catch (deviceLossSignalFailure) {
      appendSecondaryFailure(
        primaryFailure,
        secondaryFailures,
        deviceLossSignalFailure,
      );
      // GPUDevice.lost resolves by contract. If a nonconforming test double or
      // embedding rejects it, never unwind the FIFO lease or release aliased
      // storage without a terminal proof.
      await new Promise<never>(() => undefined);
    }
  }
}

function appendSecondaryFailure(
  primaryFailure: unknown,
  secondaryFailures: unknown[],
  secondaryFailure: unknown,
): void {
  if (
    secondaryFailure !== primaryFailure &&
    !secondaryFailures.includes(secondaryFailure)
  ) {
    secondaryFailures.push(secondaryFailure);
  }
}

function retainSecondarySchedulingFailures(
  primaryFailure: unknown,
  secondaryFailures: readonly unknown[],
): void {
  if (
    secondaryFailures.length === 0 ||
    (typeof primaryFailure !== "object" && typeof primaryFailure !== "function") ||
    primaryFailure === null
  ) return;
  try {
    Object.defineProperty(primaryFailure, "aceSecondarySchedulingFailures", {
      configurable: true,
      enumerable: false,
      value: Object.freeze([...secondaryFailures]),
      writable: false,
    });
  } catch {
    // Preserve the original rejection even when its object is not extensible.
  }
}

function assertPositiveSafeInteger(value: number, message: string): void {
  if (!Number.isSafeInteger(value) || value < 1) throw new RangeError(message);
}

function nonnegativeTimingElapsed(finishedAt: number, startedAt: number): number {
  const elapsed = finishedAt - startedAt;
  if (!Number.isFinite(elapsed) || elapsed < 0) {
    throw new RangeError("ACE GPU drain timing clock must be finite and monotonic");
  }
  return elapsed;
}

function disposedError(): DOMException {
  return new DOMException("ACE GPU graph scheduler is disposed", "InvalidStateError");
}
