/// <reference lib="webworker" />

// Dedicated inference worker for ACE-Step music generation. The runtime is
// installed with the upstream demo's direct-only boundary: planner/semantic
// material is structurally excluded, matching the direct-only R2 manifest.

import { installAceWorkerRuntime } from "ace-step-1.5.wgsl";

// Workers don't inherit the page's <meta name="referrer" content="no-referrer">,
// and Hugging Face hotlink-blocks requests carrying a *.workers.dev Referer
// (served without CORS headers → surfaces as "Failed to fetch"). Force
// no-referrer on every runtime fetch, same as src/core/modelCache.ts does for
// the page-side engines.
const workerFetch = self.fetch.bind(self);
self.fetch = ((input: RequestInfo | URL, init?: RequestInit) => workerFetch(input, { ...init, referrerPolicy: "no-referrer" })) as typeof fetch;

import { createAceDirectOnlyWebGpuPipelineBackend } from "./direct-only-backend.js";

const scope = self as unknown as DedicatedWorkerGlobalScope;
installAceWorkerRuntime(scope, createAceDirectOnlyWebGpuPipelineBackend());
