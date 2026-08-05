import type { GpuContext } from "./compute.js";

/** A compute backend — WebGPU (GpuContext) or WASM/SIMD (WasmContext). Same interface. */
export type ComputeContext = GpuContext | any;

/** Create a compute context: prefers WebGPU, falls back to WASM+SIMD on CPU. */
export function createContext(opts?: { backend?: "auto" | "webgpu" | "wasm"; onBackend?: (b: "webgpu" | "wasm") => void }): Promise<ComputeContext>;
