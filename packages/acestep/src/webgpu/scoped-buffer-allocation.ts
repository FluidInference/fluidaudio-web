/**
 * Create GPU buffers under immediately bounded error scopes.
 *
 * `GPUDevice.createBuffer()` normally returns before device-timeline
 * validation/allocation completes. A JavaScript try/catch therefore cannot
 * prove that the returned object is usable. This helper pops every scope
 * immediately after the synchronous creation batch so unrelated graph work
 * cannot be captured while the allocation result is awaited.
 */
export async function createAceScopedBuffers(
  device: GPUDevice,
  descriptors: readonly GPUBufferDescriptor[],
  operation: string,
): Promise<readonly GPUBuffer[]> {
  device.pushErrorScope("internal");
  device.pushErrorScope("out-of-memory");
  device.pushErrorScope("validation");

  const buffers: GPUBuffer[] = [];
  let synchronousFailure: unknown;
  try {
    for (const descriptor of descriptors) {
      buffers.push(device.createBuffer(descriptor));
    }
  } catch (error) {
    synchronousFailure = error;
  }

  // Pop synchronously and in reverse order before awaiting any device work.
  // Error scopes are device-global; leaving one pushed over an await could
  // capture a different graph owner's error.
  const validation = device.popErrorScope();
  const outOfMemory = device.popErrorScope();
  const internal = device.popErrorScope();
  const results = await Promise.allSettled([
    validation,
    outOfMemory,
    internal,
  ]);
  const scopedFailure = firstScopeFailure(operation, results);
  const failure = synchronousFailure ?? scopedFailure;
  if (failure !== undefined) {
    for (const buffer of buffers) buffer.destroy();
    throw failure;
  }
  return Object.freeze(buffers);
}

function firstScopeFailure(
  operation: string,
  results: readonly PromiseSettledResult<GPUError | null>[],
): unknown | undefined {
  const filters = ["validation", "out-of-memory", "internal"] as const;
  for (let index = 0; index < results.length; index += 1) {
    const result = results[index]!;
    const filter = filters[index]!;
    if (result.status === "rejected") return result.reason;
    if (result.value !== null) {
      return new Error(
        `${operation} failed with a WebGPU ${filter} error: ${result.value.message}`,
        { cause: result.value },
      );
    }
  }
  return undefined;
}
