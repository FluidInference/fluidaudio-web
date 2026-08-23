import { createAceScopedBuffers } from "./scoped-buffer-allocation.js";

const ACE_STORAGE_OFFSET_ALIGNMENT = 256;

export interface AceArenaBufferPlan {
  readonly label: string;
  readonly byteLength: number;
}

export interface AceArenaSlice {
  readonly label: string;
  readonly bufferIndex: number;
  readonly byteOffset: number;
  readonly byteLength: number;
}

export interface AceLifetimeAllocation {
  readonly label: string;
  readonly byteLength: number;
  /** Inclusive graph-quantum index at which the allocation first exists. */
  readonly firstQuantum: number;
  /** Inclusive graph-quantum index of its final read or write. */
  readonly lastQuantum: number;
}

export interface AceLifetimeArenaPlan {
  readonly byteLength: number;
  readonly slices: Readonly<Record<string, AceArenaSlice>>;
}

/** Explicit, fixed GPU allocations shared by one exclusively owned graph. */
export class AceGpuArena {
  readonly buffers: readonly GPUBuffer[];
  readonly bufferByteLengths: readonly number[];
  readonly byteLength: number;
  private destroyed = false;

  private constructor(
    private readonly maxStorageBufferBindingSize: number,
    plans: readonly AceArenaBufferPlan[],
    buffers: readonly GPUBuffer[],
  ) {
    this.buffers = buffers;
    this.bufferByteLengths = Object.freeze(plans.map((plan) => plan.byteLength));
    this.byteLength = checkedTotal(this.bufferByteLengths);
  }

  static async create(
    device: GPUDevice,
    plans: readonly AceArenaBufferPlan[],
  ): Promise<AceGpuArena> {
    const stablePlans = Object.freeze(
      plans.map((plan) => Object.freeze({
        label: plan.label,
        byteLength: plan.byteLength,
      })),
    );
    if (stablePlans.length === 0) {
      throw new RangeError("ACE arena requires a buffer plan");
    }
    for (const plan of stablePlans) requireBufferPlan(device, plan);
    checkedTotal(stablePlans.map((plan) => plan.byteLength));
    const buffers = await createAceScopedBuffers(
      device,
      stablePlans.map((plan) => ({
        label: plan.label,
        size: plan.byteLength,
        usage:
          GPUBufferUsage.STORAGE |
          GPUBufferUsage.COPY_SRC |
          GPUBufferUsage.COPY_DST,
      })),
      "ACE GPU arena allocation",
    );
    return new AceGpuArena(
      device.limits.maxStorageBufferBindingSize,
      stablePlans,
      buffers,
    );
  }

  slice(
    label: string,
    bufferIndex: number,
    byteOffset: number,
    byteLength: number,
  ): AceArenaSlice {
    this.assertAlive();
    const capacity = this.bufferByteLengths[bufferIndex];
    if (
      capacity === undefined ||
      !Number.isSafeInteger(byteOffset) ||
      !Number.isSafeInteger(byteLength) ||
      byteOffset < 0 ||
      byteLength <= 0 ||
      byteOffset % ACE_STORAGE_OFFSET_ALIGNMENT !== 0 ||
      byteLength % 4 !== 0 ||
      byteLength > this.maxStorageBufferBindingSize ||
      byteOffset + byteLength > capacity
    ) {
      throw new RangeError(`Invalid ACE arena slice ${label}`);
    }
    return Object.freeze({ label, bufferIndex, byteOffset, byteLength });
  }

  binding(slice: AceArenaSlice): GPUBufferBinding {
    this.assertSlice(slice);
    return {
      buffer: this.buffers[slice.bufferIndex]!,
      offset: slice.byteOffset,
      size: slice.byteLength,
    };
  }

  /** Reject simultaneous writable aliases before encoding a graph quantum. */
  assertNoWritableOverlap(
    writable: readonly AceArenaSlice[],
    readable: readonly AceArenaSlice[] = [],
  ): void {
    this.assertAlive();
    for (const slice of [...writable, ...readable]) this.assertSlice(slice);
    for (let left = 0; left < writable.length; left += 1) {
      for (let right = left + 1; right < writable.length; right += 1) {
        if (overlaps(writable[left]!, writable[right]!)) {
          throw new Error(
            `ACE writable arena aliases overlap: ${writable[left]!.label} and ${writable[right]!.label}`,
          );
        }
      }
      for (const read of readable) {
        if (overlaps(writable[left]!, read)) {
          throw new Error(
            `ACE read/write arena aliases overlap: ${writable[left]!.label} and ${read.label}`,
          );
        }
      }
    }
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    for (const buffer of this.buffers) buffer.destroy();
  }

  private assertSlice(slice: AceArenaSlice): void {
    this.assertAlive();
    const capacity = this.bufferByteLengths[slice.bufferIndex];
    if (
      capacity === undefined ||
      !Number.isSafeInteger(slice.byteOffset) ||
      !Number.isSafeInteger(slice.byteLength) ||
      slice.byteOffset < 0 ||
      slice.byteLength <= 0 ||
      slice.byteOffset % ACE_STORAGE_OFFSET_ALIGNMENT !== 0 ||
      slice.byteLength % 4 !== 0 ||
      slice.byteLength > this.maxStorageBufferBindingSize ||
      slice.byteOffset + slice.byteLength > capacity
    ) {
      throw new RangeError(`Invalid ACE arena slice ${slice.label}`);
    }
  }

  private assertAlive(): void {
    if (this.destroyed) throw new Error("ACE GPU arena was destroyed");
  }
}

/**
 * Deterministic first-fit lifetime planner. Allocations may alias only when
 * their inclusive graph-quantum lifetimes do not overlap.
 */
export function planAceLifetimeArena(
  allocations: readonly AceLifetimeAllocation[],
): AceLifetimeArenaPlan {
  const labels = new Set<string>();
  const placed: Array<AceLifetimeAllocation & { byteOffset: number; alignedBytes: number }> = [];
  const slices: Record<string, AceArenaSlice> = Object.create(null) as Record<
    string,
    AceArenaSlice
  >;
  let highWater = 0;
  for (const allocation of allocations) {
    validateLifetime(allocation);
    if (labels.has(allocation.label)) {
      throw new Error(`Duplicate ACE lifetime allocation ${allocation.label}`);
    }
    labels.add(allocation.label);
    const alignedBytes = align(allocation.byteLength, 4);
    const conflicting = placed
      .filter((candidate) => lifetimesOverlap(allocation, candidate))
      .sort((left, right) => left.byteOffset - right.byteOffset);
    let offset = 0;
    for (const candidate of conflicting) {
      offset = align(offset, ACE_STORAGE_OFFSET_ALIGNMENT);
      if (offset + alignedBytes <= candidate.byteOffset) break;
      offset = Math.max(offset, candidate.byteOffset + candidate.alignedBytes);
    }
    offset = align(offset, ACE_STORAGE_OFFSET_ALIGNMENT);
    placed.push({ ...allocation, byteOffset: offset, alignedBytes });
    slices[allocation.label] = Object.freeze({
      label: allocation.label,
      bufferIndex: 0,
      byteOffset: offset,
      byteLength: alignedBytes,
    });
    highWater = Math.max(highWater, offset + alignedBytes);
  }
  return Object.freeze({
    byteLength: align(highWater, ACE_STORAGE_OFFSET_ALIGNMENT),
    slices: Object.freeze(slices),
  });
}

function requireBufferPlan(device: GPUDevice, plan: AceArenaBufferPlan): void {
  if (
    !plan.label ||
    !Number.isSafeInteger(plan.byteLength) ||
    plan.byteLength <= 0 ||
    plan.byteLength % ACE_STORAGE_OFFSET_ALIGNMENT !== 0
  ) {
    throw new RangeError("ACE arena buffers must be positive 256-byte multiples");
  }
  if (plan.byteLength > device.limits.maxBufferSize) {
    throw new RangeError(`${plan.label} exceeds this device's buffer-size limit`);
  }
}

function validateLifetime(allocation: AceLifetimeAllocation): void {
  if (
    !allocation.label ||
    !Number.isSafeInteger(allocation.byteLength) ||
    allocation.byteLength <= 0 ||
    !Number.isSafeInteger(allocation.firstQuantum) ||
    !Number.isSafeInteger(allocation.lastQuantum) ||
    allocation.firstQuantum < 0 ||
    allocation.lastQuantum < allocation.firstQuantum
  ) {
    throw new RangeError(`Invalid ACE lifetime allocation ${allocation.label}`);
  }
}

function lifetimesOverlap(
  left: Pick<AceLifetimeAllocation, "firstQuantum" | "lastQuantum">,
  right: Pick<AceLifetimeAllocation, "firstQuantum" | "lastQuantum">,
): boolean {
  return left.firstQuantum <= right.lastQuantum && right.firstQuantum <= left.lastQuantum;
}

function overlaps(left: AceArenaSlice, right: AceArenaSlice): boolean {
  return (
    left.bufferIndex === right.bufferIndex &&
    left.byteOffset < right.byteOffset + right.byteLength &&
    right.byteOffset < left.byteOffset + left.byteLength
  );
}

function align(value: number, alignment: number): number {
  const aligned = Math.ceil(value / alignment) * alignment;
  if (!Number.isSafeInteger(aligned)) {
    throw new RangeError("ACE arena alignment exceeds safe integers");
  }
  return aligned;
}

function checkedTotal(values: readonly number[]): number {
  let total = 0;
  for (const value of values) {
    total += value;
    if (!Number.isSafeInteger(total)) {
      throw new RangeError("ACE arena byte accounting exceeds safe integers");
    }
  }
  return total;
}
