/// <reference lib="webworker" />

import { installAceWorkerRuntime } from "../src/runtime/worker.js";
import { createAceWebGpuPipelineBackend } from "../src/runtime/webgpu-pipeline.js";

const scope = self as unknown as DedicatedWorkerGlobalScope;
installAceWorkerRuntime(scope, createAceWebGpuPipelineBackend());
