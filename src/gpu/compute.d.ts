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
  /** 1-D conv. x:[Cin,L], w = Cout*(Cin/groups)*K f32, bias?:[1,Cout] -> [Cout,Lout]. */
  conv1d(x: GpuTensor, w: GpuTensor, opts: {
    cout: number; k: number; bias?: GpuTensor | null;
    stride?: number; pad?: number; dilation?: number; groups?: number; act?: Activation;
  }): GpuTensor;
  layernorm(x: GpuTensor, gamma: GpuTensor, beta: GpuTensor, eps?: number): GpuTensor;
  softmax(x: GpuTensor): GpuTensor;
  ewise(a: GpuTensor, b: GpuTensor, op: "add" | "mul"): GpuTensor;
  add(a: GpuTensor, b: GpuTensor): GpuTensor;
  mul(a: GpuTensor, b: GpuTensor): GpuTensor;
  /** [rows,cols] -> [cols,rows]. */
  transpose(x: GpuTensor): GpuTensor;
  /** Extract columns [col0, col0+width). */
  sliceCols(x: GpuTensor, col0: number, width: number): GpuTensor;
  /** Write src[rows,width] into dst at column col0 (in place); returns dst. */
  setCols(dst: GpuTensor, src: GpuTensor, col0: number): GpuTensor;
  download(t: GpuTensor): Promise<Float32Array>;
}
