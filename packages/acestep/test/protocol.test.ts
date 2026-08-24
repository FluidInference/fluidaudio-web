import { describe, expect, it } from "vitest";

import {
  ACE_FATAL_GPU_ERROR_CODES,
  isAceFatalGpuErrorCode,
  isAceClientMessage,
  isAceWorkerMessage,
  serializeAceWorkerError,
  type AceWorkerMessage,
} from "../src/runtime/protocol.js";
import { canonicalizeSeed } from "../src/runtime/seed.js";
import {
  ACE_OPT_0009_DIT_DENSE_KERNEL_SET_ID,
  ACE_OPT_0009_DIT_DENSE_MANIFEST_SHA256,
  ACE_OPT_0009_DIT_DENSE_RUNTIME_PROFILE,
  ACE_OPT_0088_DIT_DENSE_PORTABLE_KERNEL_SET_ID,
} from "../src/webgpu/dit-fp16-package.js";
import {
  ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PROFILE,
  ACE_OPT_0072_VAE_FP16_FIXED32_DUAL_K4_PRODUCTION_RUNTIME_PROFILE,
  ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_PROFILE,
  ACE_OPT_0028_VAE_FP16_MANIFEST_SHA256,
  ACE_OPT_0040_VAE_FP16_FIXED32_SHAPE_SELECTED_PROFILE,
  ACE_OPT_0054_VAE_FP16_FIXED32_REVISION7_PROFILE,
  ACE_OPT_0088_VAE_FP16_PORTABLE_DUAL_K4_PROFILE,
} from "../src/webgpu/vae-fp16-profile.js";
import {
  ACE_REFERENCE_MANIFEST_SHA256,
  ACE_OPT_0054_VAE_REVISION7_MANIFEST_BYTES,
  ACE_OPT_0054_VAE_REVISION7_MANIFEST_SHA256,
} from "../src/model/package.js";
import {
  ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_KERNEL_SET_ID,
  ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
  ACE_OPT_0088_DIT_PORTABLE_ATTENTION_KERNEL_SET_ID,
} from "../src/webgpu/dit-attention-profile.js";
import {
  ACE_OPT_0070_VAE_C2378_WINDOW_RUNTIME_PROFILE,
} from "../src/webgpu/vae-window-profile.js";
import { ACE_REFERENCE_SUBGROUP_PROFILE } from
  "../src/webgpu/capabilities.js";
import {
  testDiagnostics,
  testGenerationResult,
  testPortableProductionDiagnostics,
} from "./runtime-fixtures.js";

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

describe("worker protocol validation", () => {
  it("requires an externally pinned manifest digest during initialization", () => {
    const base = {
      type: "initialize",
      requestId: 1,
      configuration: {
        manifestUrl: "https://example.test/model/manifest.json",
        manifestSha256: "a".repeat(64),
        modelProfile: "reference-bf16",
        schedulingProfile: "cooperative",
        ditDensePackage: {
          manifestUrl:
            "https://example.test/model/files-fp16-dit-rev7-oracle/manifest.json",
          manifestSha256: ACE_OPT_0009_DIT_DENSE_MANIFEST_SHA256,
          runtimeProfile: ACE_OPT_0009_DIT_DENSE_RUNTIME_PROFILE,
        },
        vaePackage: {
          manifestUrl:
            "https://example.test/model/files-fp16-vae-experimental/manifest.json",
          manifestSha256:
            ACE_OPT_0028_VAE_FP16_MANIFEST_SHA256,
          runtimeProfile:
            ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_PROFILE.id,
          maxWindowFrames: 512,
        },
      },
      modelSource: "cache-or-network",
      reportProgress: true,
      reportDiagnostics: true,
    };
    expect(isAceClientMessage(base)).toBe(true);
    const candidate = {
      ...base,
      configuration: {
        ...base.configuration,
        vaePackage: {
          manifestUrl:
            "https://example.test/model/files-fp16-vae-revision7-experimental/manifest.json",
          manifestSha256: ACE_OPT_0054_VAE_REVISION7_MANIFEST_SHA256,
          runtimeProfile:
            ACE_OPT_0054_VAE_FP16_FIXED32_REVISION7_PROFILE.id,
          maxWindowFrames: 512,
        },
      },
    };
    expect(isAceClientMessage(candidate)).toBe(true);
    expect(isAceClientMessage({
      ...base,
      configuration: {
        ...base.configuration,
        vaePackage: {
          ...base.configuration.vaePackage,
          runtimeProfile:
            ACE_OPT_0040_VAE_FP16_FIXED32_SHAPE_SELECTED_PROFILE.id,
        },
      },
    })).toBe(true);
    expect(isAceClientMessage({
      ...candidate,
      configuration: {
        ...candidate.configuration,
        vaePackage: {
          ...candidate.configuration.vaePackage,
          runtimeProfile:
            ACE_OPT_0040_VAE_FP16_FIXED32_SHAPE_SELECTED_PROFILE.id,
        },
      },
    })).toBe(false);
    expect(
      isAceClientMessage({
        ...base,
        configuration: { ...base.configuration, manifestSha256: "mutable" },
      }),
    ).toBe(false);
    expect(
      isAceClientMessage({
        ...base,
        configuration: {
          ...base.configuration,
          ditDensePackage: {
            ...base.configuration.ditDensePackage,
            manifestSha256: "f".repeat(64),
          },
        },
      }),
    ).toBe(false);
    expect(
      isAceClientMessage({
        ...base,
        configuration: {
          ...base.configuration,
          vaePackage: {
            ...base.configuration.vaePackage,
            runtimeProfile:
              "opt-0011-mixed-fp16-fixed32-k7-hybrid-v1",
          },
        },
      }),
    ).toBe(false);
    expect(
      isAceClientMessage({
        ...base,
        configuration: {
          ...base.configuration,
          vaePackage: { ...base.configuration.vaePackage, maxWindowFrames: 256 },
        },
      }),
    ).toBe(false);
    expect(
      isAceClientMessage({
        ...base,
        configuration: {
          ...base.configuration,
          vaePackage: {
            ...base.configuration.vaePackage,
            runtimeProfile: "opt-0011-mixed-fp16-portable-v1",
          },
        },
      }),
    ).toBe(false);
    expect(
      isAceClientMessage({
        ...base,
        configuration: { ...base.configuration, modelProfile: "raw-fp16" },
      }),
    ).toBe(false);
  });

  it("accepts a fully resolved structured-clone-safe request", () => {
    expect(
      isAceClientMessage({
        type: "generate",
        jobId: 1,
        request: {
          generationProfile: "ace-turbo-v1-correctness",
          prompt: "test",
          instrumental: true,
          durationSeconds: 10,
          seed: canonicalizeSeed(1),
          planner: { mode: "disabled" },
        },
        reportProgress: true,
        reportDiagnostics: true,
      }),
    ).toBe(true);
  });

  it("rejects unknown message fields and malformed requests", () => {
    expect(
      isAceClientMessage({ type: "cancel", jobId: 1, surprise: true }),
    ).toBe(false);
    expect(
      isAceClientMessage({
        type: "generate",
        jobId: 1,
        request: {
          generationProfile: "ace-turbo-v1-correctness",
          prompt: "test",
          instrumental: true,
          durationSeconds: 10,
          seed: "1",
          planner: { mode: "disabled" },
        },
        reportProgress: false,
        reportDiagnostics: false,
      }),
    ).toBe(false);
  });

  it("exact-key validates every worker-to-client message variant", () => {
    const messages: AceWorkerMessage[] = [
      {
        type: "initialization-progress",
        requestId: 1,
        progress: {
          stage: "webgpu",
          completedUnits: 1,
          totalUnits: 1,
          unit: "items",
          overallFraction: 0.1,
          elapsedMs: 1,
        },
      },
      { type: "ready", requestId: 1, diagnostics: testDiagnostics() },
      { type: "initialization-cancelled", requestId: 1 },
      {
        type: "generation-progress",
        jobId: 2,
        progress: {
          stage: "prepare",
          completedUnits: 1,
          totalUnits: 1,
          unit: "items",
          overallFraction: 0.1,
          elapsedMs: 1,
          stageIndex: 0,
          stageCount: 13,
        },
      },
      {
        type: "diagnostic",
        jobId: 2,
        diagnostic: {
          severity: "info",
          code: "TEST",
          message: "test",
          elapsedMs: 1,
          details: { count: 1, flags: [true, false] },
        },
      },
      { type: "result", jobId: 2, result: testGenerationResult() },
      { type: "cancelled", jobId: 2 },
      { type: "disposed", requestId: 3 },
      {
        type: "error",
        jobId: 2,
        error: {
          name: "Error",
          message: "test",
          code: "TEST",
          context: { stage: "prepare", requestedBytes: 1 },
        },
      },
    ];

    for (const message of messages) {
      expect(isAceWorkerMessage(message)).toBe(true);
      expect(isAceWorkerMessage({ ...message, hidden: true })).toBe(false);
    }
  });

  it("validates revision-7 diagnostics only as one exact identity tuple", () => {
    const candidate = {
      ...testDiagnostics(),
      vaeManifestSha256: ACE_OPT_0054_VAE_REVISION7_MANIFEST_SHA256,
      vaeManifestByteLength: ACE_OPT_0054_VAE_REVISION7_MANIFEST_BYTES,
      vaeRuntimeProfile: ACE_OPT_0054_VAE_FP16_FIXED32_REVISION7_PROFILE.id,
      vaeKernelSetId:
        ACE_OPT_0054_VAE_FP16_FIXED32_REVISION7_PROFILE.kernelSetId,
      vaePrecisionMapSha256:
        ACE_OPT_0054_VAE_FP16_FIXED32_REVISION7_PROFILE.precisionMapSha256!,
    };
    expect(isAceWorkerMessage({
      type: "ready",
      requestId: 1,
      diagnostics: candidate,
    })).toBe(true);
    expect(isAceWorkerMessage({
      type: "ready",
      requestId: 1,
      diagnostics: {
        ...candidate,
        vaeRuntimeProfile:
          ACE_OPT_0040_VAE_FP16_FIXED32_SHAPE_SELECTED_PROFILE.id,
      },
    })).toBe(false);
  });

  it("validates OPT-0088 portable diagnostics only as one coherent tuple", () => {
    const portable = testPortableProductionDiagnostics();
    const subgroups = opt0080ProductionDiagnostics();
    const ready = (diagnostics: unknown) => ({
      type: "ready" as const,
      requestId: 1,
      diagnostics,
    });
    expect(isAceWorkerMessage(ready(portable))).toBe(true);
    expect(isAceWorkerMessage(ready(subgroups))).toBe(true);
    // A portable execution profile with any fixed32 kernel-set identity is
    // rejected in every position.
    expect(isAceWorkerMessage(ready({
      ...portable,
      ditDenseKernelSetId: ACE_OPT_0009_DIT_DENSE_KERNEL_SET_ID,
    }))).toBe(false);
    expect(isAceWorkerMessage(ready({
      ...portable,
      ditAttentionKernelSetId:
        ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_KERNEL_SET_ID,
    }))).toBe(false);
    expect(isAceWorkerMessage(ready({
      ...portable,
      vaeKernelSetId:
        ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PROFILE.kernelSetId,
      vaePrecisionMapSha256:
        ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PROFILE
          .precisionMapSha256!,
    }))).toBe(false);
    // A fixed32 execution profile with any portable kernel-set identity is
    // rejected in every position.
    expect(isAceWorkerMessage(ready({
      ...subgroups,
      ditDenseKernelSetId: ACE_OPT_0088_DIT_DENSE_PORTABLE_KERNEL_SET_ID,
    }))).toBe(false);
    expect(isAceWorkerMessage(ready({
      ...subgroups,
      ditAttentionKernelSetId:
        ACE_OPT_0088_DIT_PORTABLE_ATTENTION_KERNEL_SET_ID,
    }))).toBe(false);
    expect(isAceWorkerMessage(ready({
      ...subgroups,
      vaeKernelSetId:
        ACE_OPT_0088_VAE_FP16_PORTABLE_DUAL_K4_PROFILE.kernelSetId,
      vaePrecisionMapSha256:
        ACE_OPT_0088_VAE_FP16_PORTABLE_DUAL_K4_PROFILE.precisionMapSha256,
    }))).toBe(false);
    // The top-level execution profile cannot disagree with the portable
    // capability report either.
    expect(isAceWorkerMessage(ready({
      ...portable,
      executionProfile: ACE_REFERENCE_SUBGROUP_PROFILE,
    }))).toBe(false);
  });

  it("rejects hidden or malformed nested diagnostics and result metrics", () => {
    const diagnostics = testDiagnostics();
    expect(
      isAceWorkerMessage({
        type: "ready",
        requestId: 1,
        diagnostics: { ...diagnostics, hidden: true },
      }),
    ).toBe(false);
    expect(
      isAceWorkerMessage({
        type: "ready",
        requestId: 1,
        diagnostics: { ...diagnostics, vaeMaxWindowFrames: 256 },
      }),
    ).toBe(false);
    expect(
      isAceWorkerMessage({
        type: "ready",
        requestId: 1,
        diagnostics: {
          ...diagnostics,
          vaeRuntimeProfile: "opt-0011-mixed-fp16-portable-v1",
        },
      }),
    ).toBe(false);
    expect(
      isAceWorkerMessage({
        type: "ready",
        requestId: 1,
        diagnostics: {
          ...diagnostics,
          capabilities: {
            ...diagnostics.capabilities,
            deviceFeatures: [],
          },
        },
      }),
    ).toBe(false);
    expect(
      isAceWorkerMessage({
        type: "ready",
        requestId: 1,
        diagnostics: {
          ...diagnostics,
          aceSourceRevision: "0".repeat(40),
        },
      }),
    ).toBe(false);
    expect(
      isAceWorkerMessage({
        type: "diagnostic",
        jobId: 2,
        diagnostic: {
          severity: "info",
          code: "TEST",
          message: "test",
          elapsedMs: 1,
          hidden: true,
        },
      }),
    ).toBe(false);

    const result = testGenerationResult();
    expect(
      isAceWorkerMessage({
        type: "result",
        jobId: 2,
        result: {
          ...result,
          metrics: { ...result.metrics, hidden: true },
        },
      }),
    ).toBe(false);
    expect(
      isAceWorkerMessage({
        type: "result",
        jobId: 2,
        result: { ...result, modelManifestSha256: "b".repeat(64) },
      }),
    ).toBe(false);
    expect(
      isAceWorkerMessage({
        type: "result",
        jobId: 2,
        result: { ...result, metrics: [] },
      }),
    ).toBe(false);
  });

  it("binds the timing-free production scheduling receipt to its exact result", () => {
    const diagnostics = opt0080ProductionDiagnostics();
    const firstWindow = {
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
    } as const;
    const remainderWindow = {
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
    } as const;
    const receipt = {
      schema: "ace-vae-window-scheduling-receipt-v1",
      selectedProductionPolicy:
        "opt-0080-c2314-depth2-phase-epoch4",
      benchmarkPolicyOverride: null,
      windows: [firstWindow, remainderWindow],
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
      },
    });
    const message = (
      vaeScheduling: unknown,
      overrides: Readonly<Record<string, unknown>> = {},
    ) => ({
      type: "result" as const,
      jobId: 2,
      result: {
        ...result,
        ...overrides,
        metrics: {
          ...result.metrics,
          ...(typeof overrides.metrics === "object" &&
              overrides.metrics !== null
            ? overrides.metrics
            : {}),
          vaeScheduling,
        },
      },
    });

    expect(isAceWorkerMessage(message(receipt))).toBe(true);
    expect(isAceWorkerMessage(message({
      ...receipt,
      selectedProductionPolicy: null,
      benchmarkPolicyOverride: "depth1-epoch1",
      windows: [
        {
          ...firstWindow,
          selection: "benchmark-override",
          schedulingProfile: "depth1-epoch1",
          queueDrains: 556,
          cooperativeIdleTurns: 555,
          maximumOutstandingCommandBuffers: 1,
        },
        {
          ...remainderWindow,
          selection: "benchmark-override",
        },
      ],
    }, {
      metrics: {
        ...result.metrics,
        cooperativeGpuQueueDrains: 610,
        cooperativeIdleMs: 608,
      },
    }))).toBe(false);
    expect(isAceWorkerMessage(message({
      ...receipt,
      windows: [{ ...firstWindow, queueDrains: 140 }, remainderWindow],
    }))).toBe(false);
    expect(isAceWorkerMessage(message({
      ...receipt,
      windows: [{
        ...firstWindow,
        latentWindowFrames: 2_313,
      }, remainderWindow],
    }))).toBe(false);
    expect(isAceWorkerMessage(message({
      ...receipt,
      windows: [firstWindow, {
        ...remainderWindow,
        decoderQuantumCount: 2,
        decoderCommandBufferCount: 1,
        totalCommandBufferCount: 2,
        commandBuffersSubmitted: 2,
        queueDrains: 2,
        cooperativeIdleTurns: 1,
      }],
    }))).toBe(false);
    const wrongDiagnostics = testDiagnostics();
    expect(isAceWorkerMessage(message(receipt, {
      diagnostics: wrongDiagnostics,
      modelManifestId: wrongDiagnostics.modelManifestId,
      modelManifestSha256: wrongDiagnostics.modelManifestSha256,
    }))).toBe(false);
    expect(isAceWorkerMessage(message(receipt, {
      frameCount: 480_000,
      durationSeconds: 10,
    }))).toBe(false);
    expect(isAceWorkerMessage(message(receipt, {
      metrics: {
        ...result.metrics,
        cooperativeGpuQueueDrains: 192,
      },
    }))).toBe(false);
    expect(isAceWorkerMessage(message(receipt, {
      metrics: {
        ...result.metrics,
        cooperativeIdleMs: 190,
      },
    }))).toBe(false);
  });

  it("always serializes a protocol-valid error identity", () => {
    const error = new Error("test") as Error & { code?: string };
    error.name = "";
    error.code = "";
    Object.defineProperty(error, "stack", { value: 42, configurable: true });
    const serialized = serializeAceWorkerError(error, "");
    expect(serialized).toMatchObject({
      name: "Error",
      code: "INTERNAL_ERROR",
      message: "test",
    });
    expect("stack" in serialized).toBe(false);
    expect(
      isAceWorkerMessage({ type: "error", jobId: 1, error: serialized }),
    ).toBe(true);
  });

  it("preserves and recognizes fatal GPU error codes for UI recovery", () => {
    for (const code of ACE_FATAL_GPU_ERROR_CODES) {
      const error = Object.assign(new Error("fatal GPU failure"), { code });
      const serialized = serializeAceWorkerError(error);
      expect(serialized.code).toBe(code);
      expect(isAceFatalGpuErrorCode(serialized.code)).toBe(true);
    }
    expect(isAceFatalGpuErrorCode("INTERNAL_ERROR")).toBe(false);
  });
});
