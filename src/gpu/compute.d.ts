export interface GpuTensor {
  buf: GPUBuffer;
  rows: number;
  cols: number;
}

export type Activation = "none" | "gelu" | "tanh" | "relu" | "silu";

/** Request a WebGPU device in the browser (throws if unavailable). */
export function requestGpuDevice(): Promise<GPUDevice>;

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
  /** 1-D transposed conv. x:[Cin,L], w = Cin*(Cout/groups)*K f32 -> [Cout,Lout]. */
  convTranspose1d(x: GpuTensor, w: GpuTensor, opts: {
    cout: number; k: number; bias?: GpuTensor | null; stride?: number; pad?: number;
    dilation?: number; groups?: number; outputPadding?: number; act?: Activation;
  }): GpuTensor;
  /** Bidirectional LSTM (ONNX iofc, batch 1). w/r/b flat: W[2,4H,inp] R[2,4H,H] B[2,8H]. -> [seq, 2*hid]. H<=256. */
  lstm(x: GpuTensor, w: GpuTensor, r: GpuTensor, b: GpuTensor, hid: number): GpuTensor;
  /** im2col: x[Cin,L] -> [Cin*K, Lout]. */
  im2col(x: GpuTensor, k: number, opts?: { stride?: number; pad?: number; dilation?: number }): GpuTensor;
  /** conv1d via im2col + tiled GEMM (groups=1). wRows = weight as [Cout, Cin*K]. -> [Cout, Lout]. */
  conv1dGemm(x: GpuTensor, wRows: GpuTensor, cout: number, k: number, opts?: { stride?: number; pad?: number; dilation?: number; act?: Activation }): GpuTensor;
  /** Fused conv1d via implicit GEMM (groups=1), no im2col materialization. wRows=[Cout,Cin*K]. */
  conv1dFast(x: GpuTensor, wRows: GpuTensor, cout: number, k: number, opts?: { bias?: GpuTensor | null; stride?: number; pad?: number; dilation?: number; act?: Activation }): GpuTensor;
  /** Length regulator: expand x[C,T] to [C, idxMap.length] by column gather. */
  gatherCols(x: GpuTensor, idxMap: Uint32Array): GpuTensor;
  /** Upload raw packed bytes (int4 weights / zero-points) to a storage buffer. */
  uploadBytes(typed: Uint8Array | Uint32Array): { buf: GPUBuffer };
  /** int4 block-quant matmul (ONNX MatMulNBits, bits=4, block_size=32). a:[M,K] f32,
   * bq/zp packed (uploadBytes), scales:[N*nblk] f32 -> [M,N]. Runs where ORT can't. */
  matmulNBits(a: GpuTensor, bq: { buf: GPUBuffer }, scales: GpuTensor, zp: { buf: GPUBuffer }, N: number, blockSize?: number): GpuTensor;
  layernorm(x: GpuTensor, gamma: GpuTensor, beta: GpuTensor, eps?: number): GpuTensor;
  softmax(x: GpuTensor): GpuTensor;
  ewise(a: GpuTensor, b: GpuTensor, op: "add" | "mul"): GpuTensor;
  add(a: GpuTensor, b: GpuTensor): GpuTensor;
  mul(a: GpuTensor, b: GpuTensor): GpuTensor;
  /** AdaIN: instance-norm x[C,L] over time + per-channel affine. scale/shift:[C]. */
  adain(x: GpuTensor, scale: GpuTensor, shift: GpuTensor, eps?: number): GpuTensor;
  /** LeakyReLU (elementwise), default slope 0.2. */
  leakyRelu(x: GpuTensor, slope?: number): GpuTensor;
  /** [rows,cols] -> [cols,rows]. */
  transpose(x: GpuTensor): GpuTensor;
  /** Extract columns [col0, col0+width). */
  sliceCols(x: GpuTensor, col0: number, width: number): GpuTensor;
  /** Write src[rows,width] into dst at column col0 (in place); returns dst. */
  setCols(dst: GpuTensor, src: GpuTensor, col0: number): GpuTensor;
  download(t: GpuTensor): Promise<Float32Array>;
}
