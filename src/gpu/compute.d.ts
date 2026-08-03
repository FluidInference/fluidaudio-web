export interface GpuTensor {
  buf: GPUBuffer;
  rows: number;
  cols: number;
}

export type Activation = "none" | "gelu" | "tanh" | "relu";

/**
 * Raw-WebGPU compute core: hand-written WGSL kernels over GPU-resident tensors.
 * Pass a GPUDevice (navigator.gpu in the browser, dawn in Node). Kernels take and
 * return GpuTensors; only `download` copies back to CPU.
 */
export class GpuContext {
  constructor(device: GPUDevice);
  upload(data: Float32Array, rows: number, cols: number): GpuTensor;
  alloc(rows: number, cols: number): GpuTensor;
  /** C = act(A[M,K] @ B[K,N] + bias[1,N]). */
  matmul(a: GpuTensor, b: GpuTensor, opts?: { bias?: GpuTensor | null; act?: Activation }): GpuTensor;
  layernorm(x: GpuTensor, gamma: GpuTensor, beta: GpuTensor, eps?: number): GpuTensor;
  softmax(x: GpuTensor): GpuTensor;
  ewise(a: GpuTensor, b: GpuTensor, op: "add" | "mul"): GpuTensor;
  add(a: GpuTensor, b: GpuTensor): GpuTensor;
  mul(a: GpuTensor, b: GpuTensor): GpuTensor;
  download(t: GpuTensor): Promise<Float32Array>;
}
