import { createAceScopedBuffers } from "./scoped-buffer-allocation.js";

const ACE_UNIFORM_MIN_BINDING_BYTES = 16;

export interface AceUniformAllocation {
  readonly label: string;
  readonly byteOffset: number;
  readonly byteLength: number;
  readonly binding: GPUBufferBinding;
}

type AceUniformPoolState = "idle" | "encoding" | "submitted" | "destroyed";

/**
 * One bounded uniform-buffer pool per exclusively owned graph.
 *
 * Submission is owned by the pool, and it will not recycle an offset until
 * WebGPU confirms that the submitted quantum is done. This prevents a CPU-side
 * `queue.writeBuffer` from racing an older command buffer that still reads the
 * same dynamic-uniform slot.
 */
export class AceUniformPool {
  readonly buffer: GPUBuffer;
  readonly capacityBytes: number;
  readonly offsetAlignment: number;
  private cursor = 0;
  private state: AceUniformPoolState = "idle";

  private constructor(
    private readonly device: GPUDevice,
    capacityBytes: number,
    buffer: GPUBuffer,
  ) {
    this.capacityBytes = capacityBytes;
    this.offsetAlignment = device.limits.minUniformBufferOffsetAlignment;
    this.buffer = buffer;
  }

  static async create(
    device: GPUDevice,
    capacityBytes: number,
    label = "ace-uniform-pool",
  ): Promise<AceUniformPool> {
    const alignment = device.limits.minUniformBufferOffsetAlignment;
    if (!Number.isSafeInteger(alignment) || alignment <= 0) {
      throw new RangeError("WebGPU reported an invalid uniform-buffer alignment");
    }
    if (
      !Number.isSafeInteger(capacityBytes) ||
      capacityBytes <= 0 ||
      capacityBytes % alignment !== 0 ||
      capacityBytes > device.limits.maxBufferSize
    ) {
      throw new RangeError(
        "ACE uniform-pool capacity must be an aligned positive device buffer size",
      );
    }
    const [buffer] = await createAceScopedBuffers(
      device,
      [{
        label,
        size: capacityBytes,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      }],
      "ACE uniform-pool allocation",
    );
    return new AceUniformPool(device, capacityBytes, buffer!);
  }

  /** Start allocating uniforms for exactly one command-buffer quantum. */
  beginQuantum(): void {
    this.requireState("idle");
    this.cursor = 0;
    this.state = "encoding";
  }

  write(label: string, data: ArrayBuffer | ArrayBufferView): AceUniformAllocation {
    this.requireState("encoding");
    if (label.length === 0) throw new TypeError("ACE uniform labels must be nonempty");
    const source = ArrayBuffer.isView(data)
      ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
      : new Uint8Array(data);
    if (source.byteLength === 0) {
      throw new RangeError("ACE uniform payloads must be nonempty");
    }
    const bindingBytes = align(
      Math.max(source.byteLength, ACE_UNIFORM_MIN_BINDING_BYTES),
      ACE_UNIFORM_MIN_BINDING_BYTES,
    );
    if (bindingBytes > this.device.limits.maxUniformBufferBindingSize) {
      throw new RangeError(`${label} exceeds maxUniformBufferBindingSize`);
    }
    const byteOffset = align(this.cursor, this.offsetAlignment);
    const next = byteOffset + bindingBytes;
    if (!Number.isSafeInteger(next) || next > this.capacityBytes) {
      throw new RangeError(`ACE uniform pool exhausted while allocating ${label}`);
    }

    // Copy into a zero-filled, 16-byte-sized payload. This both satisfies
    // queue.writeBuffer alignment and prevents caller mutation or stale tail
    // bytes from changing a uniform struct after this method returns.
    const stable = new Uint8Array(bindingBytes);
    stable.set(source);
    this.device.queue.writeBuffer(this.buffer, byteOffset, stable);
    this.cursor = next;
    return Object.freeze({
      label,
      byteOffset,
      byteLength: bindingBytes,
      binding: Object.freeze({
        buffer: this.buffer,
        offset: byteOffset,
        size: bindingBytes,
      }),
    });
  }

  /**
   * Submit exactly this quantum and recycle its offsets only after that submit
   * has drained. `queue.submit()` runs synchronously before the drain snapshot,
   * so callers cannot accidentally reclaim against an earlier queue state.
   */
  async submitQuantum(commandBuffer: GPUCommandBuffer): Promise<void> {
    this.requireState("encoding");
    this.state = "submitted";
    try {
      this.device.queue.submit([commandBuffer]);
      await this.device.queue.onSubmittedWorkDone();
      if (this.isDestroyed()) return;
      this.cursor = 0;
      this.state = "idle";
    } catch (error) {
      // A failed submit cannot leave older uniform slots eligible for reuse.
      // Device-loss teardown may still explicitly destroy the pool.
      if (!this.isDestroyed()) this.state = "submitted";
      throw error;
    }
  }

  destroy(): void {
    if (this.state === "destroyed") return;
    this.state = "destroyed";
    this.buffer.destroy();
  }

  private requireState(expected: AceUniformPoolState): void {
    if (this.state !== expected) {
      throw new DOMException(
        `ACE uniform pool is ${this.state}; expected ${expected}`,
        "InvalidStateError",
      );
    }
  }

  private isDestroyed(): boolean {
    return this.state === "destroyed";
  }
}

function align(value: number, alignment: number): number {
  return Math.ceil(value / alignment) * alignment;
}
