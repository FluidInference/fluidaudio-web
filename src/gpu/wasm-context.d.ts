import type { ComputeContext, Tensor } from "./compute.js";

/** CPU tensor (WasmContext): storage is a plain Float32Array. */
export interface WasmTensor extends Tensor {
  data: Float32Array;
}

/** WASM+SIMD CPU backend. Same kernel interface as GpuContext (batching and
 * arenas are no-ops; f16 variants fall back to fp32). */
export class WasmContext implements ComputeContext<WasmTensor> {
  constructor(exports: WebAssembly.Exports);
  readonly backend: "wasm";
}
export interface WasmContext extends ComputeContext<WasmTensor> {}

/** Instantiate the WASM kernel module and wrap it in a WasmContext. */
export function createWasmContext(wasmBytes: BufferSource): Promise<WasmContext>;
