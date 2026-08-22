import { ACE_COOPERATIVE_GPU_IDLE_MILLISECONDS } from
  "../src/runtime/scheduler.js";

export const ACE_OPT_0006_QUANTA_PER_COMMAND_BUFFER_CANDIDATES =
  Object.freeze([1, 2, 4, 8, 16] as const);

export interface AceOpt0006EncodableQuantum {
  readonly id: string;
  encode(pass: GPUComputePassEncoder): void;
}

export interface AceOpt0006QuantumBatch {
  readonly index: number;
  readonly firstQuantumIndex: number;
  readonly quantumCount: number;
  readonly final: boolean;
}

export interface AceOpt0006BatchProgress {
  readonly completedQuanta: number;
  readonly totalQuanta: number;
  readonly commandBuffersSubmitted: number;
  readonly queueDrains: number;
  readonly cooperativeIdleMs: number;
  readonly completedBatchIndex: number;
  readonly totalBatches: number;
}

export interface AceOpt0006BatchResult {
  readonly completedQuanta: number;
  readonly commandBuffersSubmitted: number;
  readonly queueDrains: number;
  readonly cooperativeIdleMs: number;
  readonly batchCount: number;
}

export interface AceOpt0006RunOptions {
  readonly device: Pick<GPUDevice, "createCommandEncoder">;
  readonly queue: Pick<GPUQueue, "submit" | "onSubmittedWorkDone">;
  readonly quanta: readonly AceOpt0006EncodableQuantum[];
  readonly maximumQuantaPerCommandBuffer: number;
  readonly signal: AbortSignal;
  /** True when a readback or other command follows the final decoder batch. */
  readonly finalCommandBufferRemains?: boolean;
  readonly label?: string;
  readonly yieldQueueIdle?: () => Promise<void>;
  readonly onProgress?: (progress: AceOpt0006BatchProgress) => void;
}

export function planAceOpt0006QuantumBatches(
  quantumCount: number,
  maximumQuantaPerCommandBuffer: number,
): readonly AceOpt0006QuantumBatch[] {
  requirePositiveSafeInteger(quantumCount, "quantum count");
  requirePositiveSafeInteger(
    maximumQuantaPerCommandBuffer,
    "maximum quanta per command buffer",
  );
  const batchCount = Math.ceil(quantumCount / maximumQuantaPerCommandBuffer);
  return Object.freeze(Array.from({ length: batchCount }, (_, index) => {
    const firstQuantumIndex = index * maximumQuantaPerCommandBuffer;
    return Object.freeze({
      index,
      firstQuantumIndex,
      quantumCount: Math.min(
        maximumQuantaPerCommandBuffer,
        quantumCount - firstQuantumIndex,
      ),
      final: index === batchCount - 1,
    });
  }));
}

export function encodeAceOpt0006QuantumBatch(
  device: Pick<GPUDevice, "createCommandEncoder">,
  quanta: readonly AceOpt0006EncodableQuantum[],
  batch: AceOpt0006QuantumBatch,
  label = "ace-opt-0006-vae",
  signal?: AbortSignal,
): GPUCommandBuffer {
  if (
    batch.firstQuantumIndex < 0 ||
    batch.quantumCount < 1 ||
    batch.firstQuantumIndex + batch.quantumCount > quanta.length
  ) {
    throw new RangeError("OPT-0006 quantum batch is outside the quantum list");
  }
  const encoder = device.createCommandEncoder({
    label: `${label}-batch-${batch.index}`,
  });
  const end = batch.firstQuantumIndex + batch.quantumCount;
  for (let quantumIndex = batch.firstQuantumIndex; quantumIndex < end;
    quantumIndex += 1) {
    signal?.throwIfAborted();
    const quantum = quanta[quantumIndex]!;
    const pass = encoder.beginComputePass({
      label: `${label}-batch-${batch.index}-quantum-${quantumIndex}-${quantum.id}`,
    });
    quantum.encode(pass);
    pass.end();
  }
  signal?.throwIfAborted();
  return encoder.finish();
}

export async function runAceOpt0006QuantumBatches(
  options: AceOpt0006RunOptions,
): Promise<AceOpt0006BatchResult> {
  if (options.quanta.length < 1) {
    throw new RangeError("OPT-0006 requires at least one quantum");
  }
  const batches = planAceOpt0006QuantumBatches(
    options.quanta.length,
    options.maximumQuantaPerCommandBuffer,
  );
  const yieldQueueIdle = options.yieldQueueIdle ?? yieldRealQueueIdle;
  let completedQuanta = 0;
  let commandBuffersSubmitted = 0;
  let queueDrains = 0;
  let cooperativeIdleMs = 0;
  for (const batch of batches) {
    options.signal.throwIfAborted();
    const commandBuffer = encodeAceOpt0006QuantumBatch(
      options.device,
      options.quanta,
      batch,
      options.label,
      options.signal,
    );
    options.signal.throwIfAborted();
    options.queue.submit([commandBuffer]);
    commandBuffersSubmitted += 1;
    await options.queue.onSubmittedWorkDone();
    queueDrains += 1;
    options.signal.throwIfAborted();
    const completedBeforeBatch = completedQuanta;
    completedQuanta += batch.quantumCount;

    const nonFinal = !batch.final || options.finalCommandBufferRemains === true;
    if (!nonFinal) {
      reportDrainedBatchProgress(
        options,
        batch,
        batches.length,
        completedBeforeBatch,
        commandBuffersSubmitted,
        queueDrains,
        cooperativeIdleMs,
      );
      continue;
    }

    // As in production scheduling, start the real timer before progress so a
    // throwing callback cannot collapse the queue-empty interval.
    const idleCompletion = yieldQueueIdle();
    cooperativeIdleMs += ACE_COOPERATIVE_GPU_IDLE_MILLISECONDS;
    try {
      reportDrainedBatchProgress(
        options,
        batch,
        batches.length,
        completedBeforeBatch,
        commandBuffersSubmitted,
        queueDrains,
        cooperativeIdleMs,
      );
    } catch (error) {
      await idleCompletion;
      throw error;
    }
    await idleCompletion;
    options.signal.throwIfAborted();
  }
  return Object.freeze({
    completedQuanta,
    commandBuffersSubmitted,
    queueDrains,
    cooperativeIdleMs,
    batchCount: batches.length,
  });
}

function reportDrainedBatchProgress(
  options: AceOpt0006RunOptions,
  batch: AceOpt0006QuantumBatch,
  totalBatches: number,
  completedBeforeBatch: number,
  commandBuffersSubmitted: number,
  queueDrains: number,
  cooperativeIdleMs: number,
): void {
  for (let offset = 1; offset <= batch.quantumCount; offset += 1) {
    options.onProgress?.(Object.freeze({
      completedQuanta: completedBeforeBatch + offset,
      totalQuanta: options.quanta.length,
      commandBuffersSubmitted,
      queueDrains,
      cooperativeIdleMs,
      completedBatchIndex: batch.index,
      totalBatches,
    }));
  }
}

function yieldRealQueueIdle(): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(resolve, ACE_COOPERATIVE_GPU_IDLE_MILLISECONDS);
  });
}

function requirePositiveSafeInteger(value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value < 1) {
    throw new RangeError(`${label} must be a positive safe integer`);
  }
}
