import type {
  AceGenerationRequest,
  AceGenerationResult,
} from "../src/api.js";
import {
  ACE_PLANNER_SOURCE_REVISION,
  ACE_SOURCE_REVISION,
  PARAKEET_REFERENCE_REVISION,
  type AceRuntimeDiagnostics,
} from "../src/runtime/diagnostics.js";
import type {
  AceGenerateMessage,
  AceInitializeMessage,
} from "../src/runtime/protocol.js";
import { canonicalizeSeed } from "../src/runtime/seed.js";
import {
  ACE_REFERENCE_PORTABLE_PROFILE,
  ACE_REFERENCE_SUBGROUP_PROFILE,
  ACE_STOCK_WEBGPU_FEATURES,
  ACE_WEBGPU_LIMIT_NAMES,
  type AceWebGpuFeatureReport,
  type AceWebGpuLimits,
} from "../src/webgpu/capabilities.js";
import {
  ACE_OPT_0009_DIT_DENSE_KERNEL_SET_ID,
  ACE_OPT_0009_DIT_DENSE_MANIFEST_BYTES,
  ACE_OPT_0009_DIT_DENSE_MANIFEST_SHA256,
  ACE_OPT_0009_DIT_DENSE_RUNTIME_PROFILE,
  ACE_OPT_0009_DIT_MIXED_LAYER_BYTES,
  ACE_OPT_0009_DIT_MIXED_RESIDENT_WEIGHT_BYTES,
  ACE_OPT_0088_DIT_DENSE_PORTABLE_KERNEL_SET_ID,
} from "../src/webgpu/dit-fp16-package.js";
import {
  ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
  ACE_OPT_0088_DIT_PORTABLE_ATTENTION_KERNEL_SET_ID,
} from "../src/webgpu/dit-attention-profile.js";
import {
  ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_PROFILE,
  ACE_OPT_0028_VAE_FP16_MANIFEST_BYTES,
  ACE_OPT_0028_VAE_FP16_MANIFEST_SHA256,
  ACE_OPT_0072_VAE_FP16_FIXED32_DUAL_K4_PRODUCTION_RUNTIME_PROFILE,
  ACE_OPT_0088_VAE_FP16_PORTABLE_DUAL_K4_PROFILE,
} from "../src/webgpu/vae-fp16-profile.js";
import {
  ACE_OPT_0054_VAE_REVISION7_MANIFEST_BYTES,
  ACE_OPT_0054_VAE_REVISION7_MANIFEST_SHA256,
} from "../src/model/package.js";
import {
  ACE_OPT_0070_VAE_C2378_WINDOW_RUNTIME_PROFILE,
  ACE_VAE_C512_WINDOW_RUNTIME_PROFILE,
} from "../src/webgpu/vae-window-profile.js";

export const TEST_MANIFEST_URL = "https://example.test/manifest.json";
export const TEST_MANIFEST_SHA256 = "a".repeat(64);

const TEST_LIMITS = Object.freeze(
  Object.fromEntries(
    ACE_WEBGPU_LIMIT_NAMES.map((name) => [name, 1024 * 1024 * 1024]),
  ),
) as AceWebGpuLimits;

// The default fixture is the audited fixed-32 subgroup production device;
// OPT-0088 portable-device fixtures override the full coherent tuple.
const TEST_STOCK_FEATURES = Object.freeze(
  Object.fromEntries(
    ACE_STOCK_WEBGPU_FEATURES.map((feature) => [
      feature,
      Object.freeze({
        adapterSupported: feature !== "timestamp-query",
        deviceEnabled: feature !== "timestamp-query",
        required: feature !== "timestamp-query",
        requested: feature !== "timestamp-query",
      }),
    ]),
  ),
) as AceWebGpuFeatureReport;

export function testDiagnostics(
  overrides: Partial<AceRuntimeDiagnostics> = {},
): AceRuntimeDiagnostics {
  return {
    backend: "custom-webgpu-wgsl-and-wasm",
    modelManifestId: "ace-test-manifest",
    modelManifestUrl: TEST_MANIFEST_URL,
    modelManifestSha256: TEST_MANIFEST_SHA256,
    ditDenseManifestId: "ace-test-opt-0009-dit-manifest",
    ditDenseManifestUrl:
      "https://example.test/model/files-fp16-dit-rev7-oracle/manifest.json",
    ditDenseManifestSha256: ACE_OPT_0009_DIT_DENSE_MANIFEST_SHA256,
    ditDenseManifestByteLength: ACE_OPT_0009_DIT_DENSE_MANIFEST_BYTES,
    ditDenseRuntimeProfile: ACE_OPT_0009_DIT_DENSE_RUNTIME_PROFILE,
    ditDenseKernelSetId: ACE_OPT_0009_DIT_DENSE_KERNEL_SET_ID,
    ditDenseLayerBytes: ACE_OPT_0009_DIT_MIXED_LAYER_BYTES,
    ditResidentWeightBytes: ACE_OPT_0009_DIT_MIXED_RESIDENT_WEIGHT_BYTES,
    vaeManifestId: "ace-test-vae-manifest",
    vaeManifestUrl:
      "https://example.test/model/files-fp16-vae-experimental/manifest.json",
    vaeManifestSha256: ACE_OPT_0028_VAE_FP16_MANIFEST_SHA256,
    vaeManifestByteLength: ACE_OPT_0028_VAE_FP16_MANIFEST_BYTES,
    vaeRuntimeProfile:
      ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_PROFILE.id,
    vaeKernelSetId:
      ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_PROFILE.kernelSetId,
    vaePrecisionMapSha256:
      ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_PROFILE.precisionMapSha256!,
    vaeWindowRuntimeProfile: ACE_VAE_C512_WINDOW_RUNTIME_PROFILE,
    vaeMaxWindowFrames: 512,
    executionProfile: ACE_REFERENCE_SUBGROUP_PROFILE,
    schedulingProfile: "cooperative",
    capabilities: {
      executionProfile: ACE_REFERENCE_SUBGROUP_PROFILE,
      schedulingProfile: "cooperative",
      adapterInfo: {
        vendor: "test-vendor",
        architecture: "test-architecture",
        device: "test-device",
        description: "test-adapter",
        subgroupMinSize: 32,
        subgroupMaxSize: 32,
      },
      adapterFeatures: ["shader-f16", "subgroups"],
      deviceFeatures: ["shader-f16", "subgroups"],
      stockFeatures: TEST_STOCK_FEATURES,
      requiredFeatures: ["shader-f16", "subgroups"],
      requestedOptionalFeatures: [],
      adapterLimits: TEST_LIMITS,
      deviceLimits: TEST_LIMITS,
      requestedLimits: {},
    },
    aceSourceRevision: ACE_SOURCE_REVISION,
    plannerSourceRevision: ACE_PLANNER_SOURCE_REVISION,
    parakeetReferenceRevision: PARAKEET_REFERENCE_REVISION,
    ...overrides,
  };
}

/**
 * OPT-0088 coherent portable production diagnostics: the OPT-0009 dense +
 * OPT-0070 attention + OPT-0072 VAE production tuple served entirely by the
 * portable kernel sets on a no-subgroups device. The adapter advertises
 * non-fixed-32 subgroups, which legitimately lands the portable profile.
 */
export function testPortableProductionDiagnostics(
  overrides: Partial<AceRuntimeDiagnostics> = {},
): AceRuntimeDiagnostics {
  const base = testDiagnostics();
  return testDiagnostics({
    ditDenseKernelSetId: ACE_OPT_0088_DIT_DENSE_PORTABLE_KERNEL_SET_ID,
    ditAttentionRuntimeProfile:
      ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
    ditAttentionKernelSetId:
      ACE_OPT_0088_DIT_PORTABLE_ATTENTION_KERNEL_SET_ID,
    vaeManifestUrl:
      "https://example.test/model/files-fp16-vae-revision7-experimental/manifest.json",
    vaeManifestSha256: ACE_OPT_0054_VAE_REVISION7_MANIFEST_SHA256,
    vaeManifestByteLength: ACE_OPT_0054_VAE_REVISION7_MANIFEST_BYTES,
    vaeRuntimeProfile:
      ACE_OPT_0072_VAE_FP16_FIXED32_DUAL_K4_PRODUCTION_RUNTIME_PROFILE,
    vaeKernelSetId: ACE_OPT_0088_VAE_FP16_PORTABLE_DUAL_K4_PROFILE.kernelSetId,
    vaePrecisionMapSha256:
      ACE_OPT_0088_VAE_FP16_PORTABLE_DUAL_K4_PROFILE.precisionMapSha256,
    vaeWindowRuntimeProfile: ACE_OPT_0070_VAE_C2378_WINDOW_RUNTIME_PROFILE,
    vaeMaxWindowFrames: 2_378,
    executionProfile: ACE_REFERENCE_PORTABLE_PROFILE,
    capabilities: {
      ...base.capabilities,
      executionProfile: ACE_REFERENCE_PORTABLE_PROFILE,
      adapterInfo: {
        vendor: "test-vendor",
        architecture: "test-architecture",
        device: "test-device",
        description: "test-adapter",
        subgroupMinSize: 4,
        subgroupMaxSize: 64,
      },
      adapterFeatures: ["shader-f16", "subgroups"],
      deviceFeatures: ["shader-f16"],
      requiredFeatures: ["shader-f16"],
      stockFeatures: {
        ...base.capabilities.stockFeatures,
        subgroups: {
          adapterSupported: true,
          deviceEnabled: false,
          required: false,
          requested: false,
        },
      },
    },
    ...overrides,
  });
}

export function testGenerationRequest(
  overrides: Partial<AceGenerationRequest> = {},
): AceGenerationRequest {
  return {
    generationProfile: "ace-turbo-v1-correctness",
    prompt: "worker test fixture",
    instrumental: true,
    durationSeconds: 10,
    seed: canonicalizeSeed(9),
    planner: { mode: "disabled" },
    ...overrides,
  };
}

export function testGenerationResult(
  overrides: Partial<AceGenerationResult> = {},
): AceGenerationResult {
  const diagnostics = overrides.diagnostics ?? testDiagnostics();
  return {
    audio: new Blob([new Uint8Array([82, 73, 70, 70])], {
      type: "audio/wav",
    }),
    audioStorageId: "runtime-fixture-9",
    mimeType: "audio/wav",
    sampleRateHz: 48_000,
    channelCount: 2,
    frameCount: 480_000,
    durationSeconds: 10,
    seed: canonicalizeSeed(9),
    generationProfile: "ace-turbo-v1-correctness",
    modelManifestId: diagnostics.modelManifestId,
    modelManifestSha256: diagnostics.modelManifestSha256,
    diagnostics,
    metrics: {
      totalMs: 1,
      stageTimings: [],
      peakTrackedGpuBytes: 0,
      cooperativeGpuQueueDrains: 0,
      cooperativeIdleMs: 0,
    },
    ...overrides,
  };
}

export function testInitializeMessage(
  overrides: Partial<AceInitializeMessage> = {},
): AceInitializeMessage {
  return {
    type: "initialize",
    requestId: 1,
    configuration: {
      manifestUrl: TEST_MANIFEST_URL,
      manifestSha256: TEST_MANIFEST_SHA256,
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
        manifestSha256: ACE_OPT_0028_VAE_FP16_MANIFEST_SHA256,
        runtimeProfile: "opt-0028-mixed-fp16-fixed32-exact-packed-v1",
        maxWindowFrames: 512,
      },
    },
    modelSource: "cache-only",
    reportProgress: false,
    reportDiagnostics: false,
    ...overrides,
  };
}

export function testGenerateMessage(
  overrides: Partial<AceGenerateMessage> = {},
): AceGenerateMessage {
  return {
    type: "generate",
    jobId: 2,
    request: testGenerationRequest(),
    reportProgress: false,
    reportDiagnostics: false,
    ...overrides,
  };
}
