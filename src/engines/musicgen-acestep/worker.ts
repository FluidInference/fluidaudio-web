/// <reference lib="webworker" />

// Dedicated inference worker for ACE-Step music generation. The runtime is
// installed with the upstream demo's direct-only boundary: planner/semantic
// material is structurally excluded, matching the direct-only R2 manifest.

import { installAceWorkerRuntime } from "ace-step-1.5.wgsl";

import { createAceDirectOnlyWebGpuPipelineBackend } from "./direct-only-backend.js";

const scope = self as unknown as DedicatedWorkerGlobalScope;
installAceWorkerRuntime(scope, createAceDirectOnlyWebGpuPipelineBackend());
