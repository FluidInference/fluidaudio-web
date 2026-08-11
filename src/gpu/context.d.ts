import type { ComputeContext } from "./compute.js";

export type { ComputeContext, Tensor, GpuTensor, Activation, StagedRead, ArenaHandle } from "./compute.js";

/** Create a compute context: prefers WebGPU, falls back to WASM+SIMD on CPU. */
export function createContext(opts?: { backend?: "auto" | "webgpu" | "wasm"; onBackend?: (b: "webgpu" | "wasm") => void }): Promise<ComputeContext>;
