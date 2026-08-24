export interface GpuTensor {
  readonly buffer: GPUBuffer;
  readonly byteLength: number;
  readonly label: string;
}

export function createF16Tensor(
  device: GPUDevice,
  elements: number,
  label: string,
): GpuTensor {
  if (!Number.isSafeInteger(elements) || elements <= 0) {
    throw new RangeError(`Invalid tensor element count for ${label}`);
  }
  const byteLength = elements * 2;
  return {
    label,
    byteLength,
    buffer: device.createBuffer({
      label,
      size: align4(byteLength),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    }),
  };
}

export function writeF16Tensor(
  device: GPUDevice,
  tensor: GpuTensor,
  values: Uint16Array,
): void {
  if (values.byteLength !== tensor.byteLength) {
    throw new RangeError(`${tensor.label} received ${values.byteLength} bytes, expected ${tensor.byteLength}`);
  }
  if (values.byteLength % 4 === 0) {
    const upload = new Uint16Array(values.length);
    upload.set(values);
    device.queue.writeBuffer(tensor.buffer, 0, upload.buffer);
    return;
  }
  const padded = new Uint16Array(Math.ceil(values.length / 2) * 2);
  padded.set(values);
  device.queue.writeBuffer(tensor.buffer, 0, padded);
}

export async function readF16Tensor(
  device: GPUDevice,
  tensor: GpuTensor,
): Promise<Uint16Array> {
  const staging = device.createBuffer({
    label: `${tensor.label}-readback`,
    size: align4(tensor.byteLength),
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  let mapped = false;
  try {
    const encoder = device.createCommandEncoder({ label: `${tensor.label}-readback-copy` });
    encoder.copyBufferToBuffer(tensor.buffer, 0, staging, 0, align4(tensor.byteLength));
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    mapped = true;
    const result = new Uint16Array(tensor.byteLength / 2);
    result.set(new Uint16Array(staging.getMappedRange(0, tensor.byteLength)));
    return result;
  } finally {
    if (mapped) staging.unmap();
    staging.destroy();
  }
}

export function destroyTensors(tensors: readonly (GpuTensor | undefined)[]): void {
  for (const tensor of tensors) tensor?.buffer.destroy();
}

function align4(value: number): number {
  return Math.ceil(value / 4) * 4;
}
