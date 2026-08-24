import { describe, expect, it, vi } from "vitest";

import type { AcePipelineBackend } from "../src/runtime/pipeline.js";
import type { AceWorkerMessage } from "../src/runtime/protocol.js";
import { AceWorkerRuntime } from "../src/runtime/worker.js";
import type {
  AceGenerationContext,
  AceInitializationContext,
} from "../src/runtime/pipeline.js";
import { canonicalizeSeed } from "../src/runtime/seed.js";
import { DEFAULT_ACE_PLANNER_CONFIGURATION } from "../src/api.js";
import {
  ACE_REFERENCE_MANIFEST_SHA256,
  ACE_OPT_0054_VAE_REVISION7_MANIFEST_BYTES,
  ACE_OPT_0054_VAE_REVISION7_MANIFEST_SHA256,
} from "../src/model/package.js";
import {
  ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_KERNEL_SET_ID,
  ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
} from "../src/webgpu/dit-attention-profile.js";
import {
  ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PROFILE,
  ACE_OPT_0072_VAE_FP16_FIXED32_DUAL_K4_PRODUCTION_RUNTIME_PROFILE,
} from "../src/webgpu/vae-fp16-profile.js";
import { ACE_OPT_0070_VAE_C2378_WINDOW_RUNTIME_PROFILE } from
  "../src/webgpu/vae-window-profile.js";
import { ACE_REFERENCE_SUBGROUP_PROFILE } from
  "../src/webgpu/capabilities.js";
import { ACE_OPT_0088_DIT_DENSE_PORTABLE_KERNEL_SET_ID } from
  "../src/webgpu/dit-fp16-package.js";
import {
  testDiagnostics,
  testGenerateMessage,
  testGenerationRequest,
  testGenerationResult,
  testInitializeMessage,
  testPortableProductionDiagnostics,
} from "./runtime-fixtures.js";

function messageCode(message: AceWorkerMessage | undefined): string | undefined {
  return message?.type === "error" ? message.error.code : undefined;
}

function opt0080ProductionDiagnostics() {
  const base = testDiagnostics();
  return testDiagnostics({
    modelManifestSha256: ACE_REFERENCE_MANIFEST_SHA256,
    ditAttentionRuntimeProfile:
      ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
    ditAttentionKernelSetId:
      ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_KERNEL_SET_ID,
    vaeManifestSha256: ACE_OPT_0054_VAE_REVISION7_MANIFEST_SHA256,
    vaeManifestByteLength: ACE_OPT_0054_VAE_REVISION7_MANIFEST_BYTES,
    vaeRuntimeProfile:
      ACE_OPT_0072_VAE_FP16_FIXED32_DUAL_K4_PRODUCTION_RUNTIME_PROFILE,
    vaeKernelSetId:
      ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PROFILE.kernelSetId,
    vaePrecisionMapSha256:
      ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PROFILE
        .precisionMapSha256!,
    vaeWindowRuntimeProfile: ACE_OPT_0070_VAE_C2378_WINDOW_RUNTIME_PROFILE,
    vaeMaxWindowFrames: 2_378,
    executionProfile: ACE_REFERENCE_SUBGROUP_PROFILE,
    capabilities: {
      ...base.capabilities,
      executionProfile: ACE_REFERENCE_SUBGROUP_PROFILE,
      adapterInfo: {
        ...base.capabilities.adapterInfo,
        subgroupMinSize: 32,
        subgroupMaxSize: 32,
      },
      adapterFeatures: ["shader-f16", "subgroups"],
      deviceFeatures: ["shader-f16", "subgroups"],
      requiredFeatures: ["shader-f16", "subgroups"],
      stockFeatures: {
        ...base.capabilities.stockFeatures,
        subgroups: {
          adapterSupported: true,
          deviceEnabled: true,
          required: true,
          requested: true,
        },
      },
    },
  });
}

describe("dedicated worker runtime shell", () => {
  it("publishes only matching OPT-0072 public diagnostics for the product tuple", async () => {
    const base = testInitializeMessage();
    const initialization = {
      ...base,
      configuration: {
        ...base.configuration,
        ditAttentionRuntimeProfile:
          ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
        vaePackage: {
          manifestUrl:
            "https://example.test/model/files-fp16-vae-revision7-experimental/manifest.json",
          manifestSha256: ACE_OPT_0054_VAE_REVISION7_MANIFEST_SHA256,
          runtimeProfile:
            ACE_OPT_0072_VAE_FP16_FIXED32_DUAL_K4_PRODUCTION_RUNTIME_PROFILE,
          windowRuntimeProfile:
            ACE_OPT_0070_VAE_C2378_WINDOW_RUNTIME_PROFILE,
          maxWindowFrames: 2_378,
        },
      },
    } as const;
    const diagnostics = testDiagnostics({
      ditAttentionRuntimeProfile:
        ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
      ditAttentionKernelSetId:
        ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_KERNEL_SET_ID,
      vaeManifestUrl: initialization.configuration.vaePackage.manifestUrl,
      vaeManifestSha256: ACE_OPT_0054_VAE_REVISION7_MANIFEST_SHA256,
      vaeManifestByteLength: ACE_OPT_0054_VAE_REVISION7_MANIFEST_BYTES,
      vaeRuntimeProfile:
        ACE_OPT_0072_VAE_FP16_FIXED32_DUAL_K4_PRODUCTION_RUNTIME_PROFILE,
      vaeKernelSetId:
        ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PROFILE.kernelSetId,
      vaePrecisionMapSha256:
        ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PROFILE.precisionMapSha256!,
      vaeWindowRuntimeProfile:
        ACE_OPT_0070_VAE_C2378_WINDOW_RUNTIME_PROFILE,
      vaeMaxWindowFrames: 2_378,
    });
    const messages: AceWorkerMessage[] = [];
    const backend: AcePipelineBackend = {
      initialize: vi.fn(async () => diagnostics),
      generate: vi.fn(async () => testGenerationResult({ diagnostics })),
      releaseResult: vi.fn(),
      dispose: vi.fn(),
    };
    const runtime = new AceWorkerRuntime({
      backend,
      postMessage: (message) => messages.push(message),
    });

    await runtime.handleMessage(initialization);
    expect(messages).toHaveLength(1);
    expect(messages[0]).toMatchObject({ type: "ready", diagnostics });
    expect(runtime.state).toBe("ready");
  });

  it("publishes coherent OPT-0088 portable diagnostics and rejects mixed pairings", async () => {
    const base = testInitializeMessage();
    const initialization = {
      ...base,
      configuration: {
        ...base.configuration,
        ditAttentionRuntimeProfile:
          ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
        vaePackage: {
          manifestUrl:
            "https://example.test/model/files-fp16-vae-revision7-experimental/manifest.json",
          manifestSha256: ACE_OPT_0054_VAE_REVISION7_MANIFEST_SHA256,
          runtimeProfile:
            ACE_OPT_0072_VAE_FP16_FIXED32_DUAL_K4_PRODUCTION_RUNTIME_PROFILE,
          windowRuntimeProfile:
            ACE_OPT_0070_VAE_C2378_WINDOW_RUNTIME_PROFILE,
          maxWindowFrames: 2_378,
        },
      },
    } as const;
    const runWith = async (
      diagnostics: ReturnType<typeof testDiagnostics>,
    ) => {
      const messages: AceWorkerMessage[] = [];
      const backend: AcePipelineBackend = {
        initialize: vi.fn(async () => diagnostics),
        generate: vi.fn(async () => testGenerationResult()),
        releaseResult: vi.fn(),
        dispose: vi.fn(),
      };
      const runtime = new AceWorkerRuntime({
        backend,
        postMessage: (message) => messages.push(message),
      });
      await runtime.handleMessage(initialization);
      return { messages, runtime };
    };

    const portable = await runWith(testPortableProductionDiagnostics());
    expect(portable.messages).toHaveLength(1);
    expect(portable.messages[0]).toMatchObject({ type: "ready" });
    expect(portable.runtime.state).toBe("ready");

    // A portable execution profile paired with a fixed32 kernel set fails
    // closed, as does a fixed32 profile paired with a portable kernel set.
    const mixedDense = await runWith(testPortableProductionDiagnostics({
      ditDenseKernelSetId: "opt-0009-n256-k32-fp16-fp32-v1",
    }));
    expect(messageCode(mixedDense.messages.at(-1)))
      .toBe("INVALID_RUNTIME_DIAGNOSTICS");
    const subgroupsProduction = testPortableProductionDiagnostics({
      ditDenseKernelSetId: "opt-0009-n256-k32-fp16-fp32-v1",
      ditAttentionKernelSetId:
        ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_KERNEL_SET_ID,
      vaeKernelSetId:
        ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PROFILE.kernelSetId,
      vaePrecisionMapSha256:
        ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PROFILE
          .precisionMapSha256!,
      executionProfile: testDiagnostics().executionProfile,
      capabilities: testDiagnostics().capabilities,
    });
    expect((await runWith(subgroupsProduction)).messages[0])
      .toMatchObject({ type: "ready" });
    const mixedPortableDense = await runWith({
      ...subgroupsProduction,
      ditDenseKernelSetId: ACE_OPT_0088_DIT_DENSE_PORTABLE_KERNEL_SET_ID,
    });
    expect(messageCode(mixedPortableDense.messages.at(-1)))
      .toBe("INVALID_RUNTIME_DIAGNOSTICS");
  });

  it("rejects a production VAE scheduling receipt for a planner request", async () => {
    const diagnostics = opt0080ProductionDiagnostics();
    const base = testInitializeMessage();
    const initialization = {
      ...base,
      configuration: {
        ...base.configuration,
        manifestSha256: ACE_REFERENCE_MANIFEST_SHA256,
        ditAttentionRuntimeProfile:
          ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
        vaePackage: {
          manifestUrl: diagnostics.vaeManifestUrl,
          manifestSha256: ACE_OPT_0054_VAE_REVISION7_MANIFEST_SHA256,
          runtimeProfile:
            ACE_OPT_0072_VAE_FP16_FIXED32_DUAL_K4_PRODUCTION_RUNTIME_PROFILE,
          windowRuntimeProfile:
            ACE_OPT_0070_VAE_C2378_WINDOW_RUNTIME_PROFILE,
          maxWindowFrames: 2_378,
        },
      },
    } as const;
    const result = testGenerationResult({
      diagnostics,
      frameCount: 4_608_000,
      durationSeconds: 96,
      metrics: {
        totalMs: 1,
        stageTimings: [],
        peakTrackedGpuBytes: 0,
        cooperativeGpuQueueDrains: 193,
        cooperativeIdleMs: 191,
        vaeScheduling: {
          schema: "ace-vae-window-scheduling-receipt-v1",
          selectedProductionPolicy:
            "opt-0080-c2314-depth2-phase-epoch4",
          benchmarkPolicyOverride: null,
          windows: [
            {
              windowIndex: 0,
              latentWindowFrames: 2_314,
              selection: "production",
              schedulingProfile: "depth2-phase-epoch4",
              decoderQuantumCount: 35_498,
              quantaPerCommandBuffer: 64,
              decoderCommandBufferCount: 555,
              readbackCommandBufferCount: 1,
              totalCommandBufferCount: 556,
              commandBuffersSubmitted: 556,
              queueDrains: 139,
              cooperativeIdleTurns: 138,
              maximumOutstandingCommandBuffers: 2,
            },
            {
              windowIndex: 1,
              latentWindowFrames: 214,
              selection: "production",
              schedulingProfile: "depth1-epoch1",
              decoderQuantumCount: 3_342,
              quantaPerCommandBuffer: 64,
              decoderCommandBufferCount: 53,
              readbackCommandBufferCount: 1,
              totalCommandBufferCount: 54,
              commandBuffersSubmitted: 54,
              queueDrains: 54,
              cooperativeIdleTurns: 53,
              maximumOutstandingCommandBuffers: 1,
            },
          ],
        },
      },
    });
    const releaseResult = vi.fn();
    const messages: AceWorkerMessage[] = [];
    const backend: AcePipelineBackend = {
      initialize: vi.fn(async () => diagnostics),
      generate: vi.fn(async () => result),
      releaseResult,
      dispose: vi.fn(),
    };
    const runtime = new AceWorkerRuntime({
      backend,
      postMessage: (message) => messages.push(message),
    });

    await runtime.handleMessage(initialization);
    expect(messages.at(-1)?.type).toBe("ready");
    await runtime.handleMessage(testGenerateMessage({
      request: testGenerationRequest({
        instrumental: false,
        lyrics: "planner receipt must fail closed",
        durationSeconds: 96,
        planner: DEFAULT_ACE_PLANNER_CONFIGURATION,
      }),
    }));

    expect(messageCode(messages.at(-1))).toBe("INVALID_GENERATION_RESULT");
    expect(messages.some(({ type }) => type === "result")).toBe(false);
    expect(releaseResult).toHaveBeenCalledOnce();
    expect(runtime.state).toBe("ready");
  });

  it("keeps cancellation responsive while generation is awaiting work", async () => {
    const messages: AceWorkerMessage[] = [];
    const backend: AcePipelineBackend = {
      initialize: vi.fn(async () => testDiagnostics()),
      generate: vi.fn(async (_request, context) => {
        await new Promise<void>((_resolve, reject) => {
          context.signal.addEventListener(
            "abort",
            () => reject(new DOMException("aborted", "AbortError")),
            { once: true },
          );
        });
        throw new Error("unreachable");
      }),
      releaseResult: vi.fn(),
      dispose: vi.fn(),
    };
    const runtime = new AceWorkerRuntime({
      backend,
      postMessage: (message) => messages.push(message),
    });

    await runtime.handleMessage(testInitializeMessage());
    const generation = runtime.handleMessage(testGenerateMessage());
    await Promise.resolve();
    await runtime.handleMessage({ type: "cancel", jobId: 2 });
    await generation;

    expect(messages.map((message) => message.type)).toEqual([
      "ready",
      "cancelled",
    ]);
    expect(runtime.state).toBe("ready");
  });

  it("cleans failed partial initialization before error and permits retry", async () => {
    const events: string[] = [];
    const messages: AceWorkerMessage[] = [];
    let attempts = 0;
    let failedContext: AceInitializationContext | undefined;
    const backend: AcePipelineBackend = {
      initialize: vi.fn(async (_configuration, context) => {
        attempts += 1;
        if (attempts === 1) {
          failedContext = context;
          throw new Error("partial setup failed");
        }
        return testDiagnostics();
      }),
      generate: vi.fn(async () => testGenerationResult()),
      releaseResult: vi.fn(),
      dispose: vi.fn(async () => {
        events.push("dispose");
      }),
    };
    const runtime = new AceWorkerRuntime({
      backend,
      postMessage: (message) => {
        messages.push(message);
        events.push(message.type);
      },
    });

    await runtime.handleMessage(testInitializeMessage());

    expect(events).toEqual(["dispose", "error"]);
    expect(runtime.state).toBe("new");
    expect(backend.dispose).toHaveBeenCalledTimes(1);

    failedContext?.onProgress({
      stage: "webgpu",
      completedUnits: 1,
      totalUnits: 1,
      unit: "items",
      overallFraction: 1,
      elapsedMs: 1,
    });
    failedContext?.onDiagnostic({
      severity: "info",
      code: "LATE",
      message: "late",
      elapsedMs: 1,
    });
    expect(messages).toHaveLength(1);

    await runtime.handleMessage(testInitializeMessage({ requestId: 2 }));
    expect(messages.map((message) => message.type)).toEqual(["error", "ready"]);
    expect(runtime.state).toBe("ready");
  });

  it("cleans cancelled partial initialization, suppresses late events, and permits retry", async () => {
    const messages: AceWorkerMessage[] = [];
    let attempts = 0;
    let cancelledContext: AceInitializationContext | undefined;
    const backend: AcePipelineBackend = {
      initialize: vi.fn(async (_configuration, context) => {
        attempts += 1;
        if (attempts !== 1) return testDiagnostics();
        cancelledContext = context;
        await new Promise<void>((_resolve, reject) => {
          context.signal.addEventListener(
            "abort",
            () => {
              context.onDiagnostic({
                severity: "info",
                code: "LATE",
                message: "late after abort",
                elapsedMs: 1,
              });
              reject(new DOMException("aborted", "AbortError"));
            },
            { once: true },
          );
        });
        throw new Error("unreachable");
      }),
      generate: vi.fn(async () => testGenerationResult()),
      releaseResult: vi.fn(),
      dispose: vi.fn(),
    };
    const runtime = new AceWorkerRuntime({
      backend,
      postMessage: (message) => messages.push(message),
    });

    const initialization = runtime.handleMessage(
      testInitializeMessage({
        reportProgress: true,
        reportDiagnostics: true,
      }),
    );
    await Promise.resolve();
    await runtime.handleMessage({ type: "cancel-initialization", requestId: 1 });
    await initialization;

    cancelledContext?.onProgress({
      stage: "webgpu",
      completedUnits: 1,
      totalUnits: 1,
      unit: "items",
      overallFraction: 1,
      elapsedMs: 2,
    });
    expect(messages.map((message) => message.type)).toEqual([
      "initialization-cancelled",
    ]);
    expect(backend.dispose).toHaveBeenCalledTimes(1);
    expect(runtime.state).toBe("new");

    await runtime.handleMessage(testInitializeMessage({ requestId: 2 }));
    expect(messages.at(-1)?.type).toBe("ready");
  });

  it("fails terminally when partial-initialization cleanup fails", async () => {
    const messages: AceWorkerMessage[] = [];
    const backend: AcePipelineBackend = {
      initialize: vi.fn(async () => {
        throw new Error("setup failed");
      }),
      generate: vi.fn(async () => testGenerationResult()),
      releaseResult: vi.fn(),
      dispose: vi.fn(async () => {
        throw new Error("cleanup failed");
      }),
    };
    const runtime = new AceWorkerRuntime({
      backend,
      postMessage: (message) => messages.push(message),
    });

    await runtime.handleMessage(testInitializeMessage());
    expect(messageCode(messages[0])).toBe("INITIALIZATION_CLEANUP_FAILED");
    expect(runtime.state).toBe("disposed");

    await runtime.handleMessage(testInitializeMessage({ requestId: 2 }));
    expect(messageCode(messages[1])).toBe("WORKER_DISPOSED");
  });

  it("rejects malformed initialization callbacks and cleans the backend", async () => {
    const messages: AceWorkerMessage[] = [];
    const backend: AcePipelineBackend = {
      initialize: vi.fn(async (_configuration, context) => {
        context.onProgress({ surprise: true } as never);
        return testDiagnostics();
      }),
      generate: vi.fn(async () => testGenerationResult()),
      releaseResult: vi.fn(),
      dispose: vi.fn(),
    };
    const runtime = new AceWorkerRuntime({
      backend,
      postMessage: (message) => messages.push(message),
    });

    await runtime.handleMessage(
      testInitializeMessage({ reportProgress: true }),
    );

    expect(messageCode(messages[0])).toBe("INVALID_INITIALIZATION_PROGRESS");
    expect(backend.dispose).toHaveBeenCalledTimes(1);
    expect(runtime.state).toBe("new");
  });

  it("rejects regressing initialization progress", async () => {
    const messages: AceWorkerMessage[] = [];
    const backend: AcePipelineBackend = {
      initialize: vi.fn(async (_configuration, context) => {
        context.onProgress({
          stage: "weights",
          completedUnits: 1,
          totalUnits: 2,
          unit: "items",
          overallFraction: 0.5,
          elapsedMs: 1,
        });
        context.onProgress({
          stage: "manifest",
          completedUnits: 2,
          totalUnits: 2,
          unit: "items",
          overallFraction: 0.6,
          elapsedMs: 2,
        });
        return testDiagnostics();
      }),
      generate: vi.fn(async () => testGenerationResult()),
      releaseResult: vi.fn(),
      dispose: vi.fn(),
    };
    const runtime = new AceWorkerRuntime({
      backend,
      postMessage: (message) => messages.push(message),
    });

    await runtime.handleMessage(
      testInitializeMessage({ reportProgress: true }),
    );

    expect(messageCode(messages.at(-1))).toBe(
      "INVALID_INITIALIZATION_PROGRESS_SEQUENCE",
    );
    expect(backend.dispose).toHaveBeenCalledTimes(1);
  });

  it("binds generation progress to its plan and rejects regressions", async () => {
    const messages: AceWorkerMessage[] = [];
    let run = 0;
    const backend: AcePipelineBackend = {
      initialize: vi.fn(async () => testDiagnostics()),
      generate: vi.fn(async (_request, context) => {
        run += 1;
        context.onProgress({
          stage: "prepare",
          stageIndex: 0,
          stageCount: 13,
          completedUnits: 1,
          totalUnits: 1,
          unit: "items",
          overallFraction: 0.2,
          elapsedMs: 1,
        });
        if (run === 1) {
          context.onProgress({
            stage: "semantic-planner",
            stageIndex: 1,
            stageCount: 13,
            completedUnits: 1,
            totalUnits: 1,
            unit: "items",
            overallFraction: 0.3,
            elapsedMs: 2,
          });
        } else {
          context.onProgress({
            stage: "prepare",
            stageIndex: 0,
            stageCount: 13,
            completedUnits: 1,
            totalUnits: 1,
            unit: "items",
            overallFraction: 0.1,
            elapsedMs: 2,
          });
        }
        return testGenerationResult();
      }),
      releaseResult: vi.fn(),
      dispose: vi.fn(),
    };
    const runtime = new AceWorkerRuntime({
      backend,
      postMessage: (message) => messages.push(message),
    });
    await runtime.handleMessage(testInitializeMessage());

    await runtime.handleMessage(
      testGenerateMessage({ reportProgress: true }),
    );
    expect(messageCode(messages.at(-1))).toBe(
      "INVALID_GENERATION_PROGRESS_PLAN",
    );
    expect(runtime.state).toBe("ready");

    await runtime.handleMessage(
      testGenerateMessage({ jobId: 3, reportProgress: true }),
    );
    expect(messageCode(messages.at(-1))).toBe(
      "INVALID_GENERATION_PROGRESS_SEQUENCE",
    );
  });

  it("rejects malformed diagnostics emitted by a backend", async () => {
    const messages: AceWorkerMessage[] = [];
    const backend: AcePipelineBackend = {
      initialize: vi.fn(async () => testDiagnostics()),
      generate: vi.fn(async (_request, context) => {
        context.onDiagnostic({
          severity: "info",
          code: "TEST",
          message: "test",
          elapsedMs: 1,
          hidden: true,
        } as never);
        return testGenerationResult();
      }),
      releaseResult: vi.fn(),
      dispose: vi.fn(),
    };
    const runtime = new AceWorkerRuntime({
      backend,
      postMessage: (message) => messages.push(message),
    });
    await runtime.handleMessage(testInitializeMessage());

    await runtime.handleMessage(
      testGenerateMessage({ reportDiagnostics: true }),
    );
    expect(messageCode(messages.at(-1))).toBe("INVALID_DIAGNOSTIC");
    expect(runtime.state).toBe("ready");
  });

  it("correlates a backend result to the current request", async () => {
    const messages: AceWorkerMessage[] = [];
    const releaseResult = vi.fn();
    const backend: AcePipelineBackend = {
      initialize: vi.fn(async () => testDiagnostics()),
      generate: vi.fn(async () => testGenerationResult()),
      releaseResult,
      dispose: vi.fn(),
    };
    const runtime = new AceWorkerRuntime({
      backend,
      postMessage: (message) => messages.push(message),
    });
    await runtime.handleMessage(testInitializeMessage());

    await runtime.handleMessage(
      testGenerateMessage({
        request: testGenerationRequest({ seed: canonicalizeSeed(10) }),
      }),
    );

    expect(messageCode(messages.at(-1))).toBe("INVALID_GENERATION_RESULT");
    expect(releaseResult).toHaveBeenCalledTimes(1);
    expect(runtime.state).toBe("ready");
  });

  it("releases a committed result when late cancellation suppresses publication", async () => {
    const messages: AceWorkerMessage[] = [];
    let finishGeneration: (() => void) | undefined;
    const releaseResult = vi.fn();
    const backend: AcePipelineBackend = {
      initialize: vi.fn(async () => testDiagnostics()),
      generate: vi.fn(async () => {
        await new Promise<void>((resolve) => {
          finishGeneration = resolve;
        });
        return testGenerationResult();
      }),
      releaseResult,
      dispose: vi.fn(),
    };
    const runtime = new AceWorkerRuntime({
      backend,
      postMessage: (message) => messages.push(message),
    });
    await runtime.handleMessage(testInitializeMessage());
    const generation = runtime.handleMessage(testGenerateMessage());
    await Promise.resolve();
    await runtime.handleMessage({ type: "cancel", jobId: 2 });
    finishGeneration?.();
    await generation;

    expect(messages.map((message) => message.type)).toEqual([
      "ready",
      "cancelled",
    ]);
    expect(releaseResult).toHaveBeenCalledTimes(1);
  });

  it("releases a result if publication throws", async () => {
    const releaseResult = vi.fn();
    const backend: AcePipelineBackend = {
      initialize: vi.fn(async () => testDiagnostics()),
      generate: vi.fn(async () => testGenerationResult()),
      releaseResult,
      dispose: vi.fn(),
    };
    const runtime = new AceWorkerRuntime({
      backend,
      postMessage: (message) => {
        if (message.type === "result" || message.type === "error") {
          throw new Error("structured clone publication failed");
        }
      },
    });
    await runtime.handleMessage(testInitializeMessage());
    await expect(runtime.handleMessage(testGenerateMessage())).rejects.toThrow(
      /publication failed/,
    );
    expect(releaseResult).toHaveBeenCalledTimes(1);
  });

  it("suppresses cancelled generation callbacks retained by the backend", async () => {
    const messages: AceWorkerMessage[] = [];
    let retainedContext: AceGenerationContext | undefined;
    const backend: AcePipelineBackend = {
      initialize: vi.fn(async () => testDiagnostics()),
      generate: vi.fn(async (_request, context) => {
        retainedContext = context;
        await new Promise<void>((_resolve, reject) => {
          context.signal.addEventListener(
            "abort",
            () => reject(new DOMException("aborted", "AbortError")),
            { once: true },
          );
        });
        return testGenerationResult();
      }),
      releaseResult: vi.fn(),
      dispose: vi.fn(),
    };
    const runtime = new AceWorkerRuntime({
      backend,
      postMessage: (message) => messages.push(message),
    });
    await runtime.handleMessage(testInitializeMessage());
    const generation = runtime.handleMessage(
      testGenerateMessage({ reportProgress: true, reportDiagnostics: true }),
    );
    await Promise.resolve();
    await runtime.handleMessage({ type: "cancel", jobId: 2 });
    await generation;

    retainedContext?.onDiagnostic({
      severity: "info",
      code: "LATE",
      message: "late",
      elapsedMs: 2,
    });
    retainedContext?.onProgress({
      stage: "prepare",
      stageIndex: 0,
      stageCount: 13,
      completedUnits: 1,
      totalUnits: 1,
      unit: "items",
      overallFraction: 0.1,
      elapsedMs: 2,
    });
    expect(messages.map((message) => message.type)).toEqual([
      "ready",
      "cancelled",
    ]);
  });

  it("rejects malformed messages before they reach the backend", async () => {
    const messages: AceWorkerMessage[] = [];
    const backend = {
      initialize: vi.fn(),
      generate: vi.fn(),
      releaseResult: vi.fn(),
      dispose: vi.fn(),
    } as unknown as AcePipelineBackend;
    const runtime = new AceWorkerRuntime({
      backend,
      postMessage: (message) => messages.push(message),
    });

    await runtime.handleMessage({ type: "cancel", jobId: -1 });

    expect(messages[0]?.type).toBe("error");
    expect(backend.initialize).not.toHaveBeenCalled();
  });
});
