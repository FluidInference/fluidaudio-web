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
} from "../src/webgpu/dit-fp16-package.js";
import {
  ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_PROFILE,
  ACE_OPT_0028_VAE_FP16_MANIFEST_BYTES,
  ACE_OPT_0028_VAE_FP16_MANIFEST_SHA256,
} from "../src/webgpu/vae-fp16-profile.js";
import { ACE_VAE_C512_WINDOW_RUNTIME_PROFILE } from
  "../src/webgpu/vae-window-profile.js";

export const TEST_MANIFEST_URL = "https://example.test/manifest.json";
export const TEST_MANIFEST_SHA256 = "a".repeat(64);

const TEST_LIMITS = Object.freeze(
  Object.fromEntries(
    ACE_WEBGPU_LIMIT_NAMES.map((name) => [name, 1024 * 1024 * 1024]),
  ),
) as AceWebGpuLimits;

const TEST_STOCK_FEATURES = Object.freeze(
  Object.fromEntries(
    ACE_STOCK_WEBGPU_FEATURES.map((feature) => [
      feature,
      Object.freeze({
        adapterSupported: feature === "shader-f16",
        deviceEnabled: feature === "shader-f16",
        required: feature === "shader-f16",
        requested: feature === "shader-f16",
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
    executionProfile: ACE_REFERENCE_PORTABLE_PROFILE,
    schedulingProfile: "cooperative",
    capabilities: {
      executionProfile: ACE_REFERENCE_PORTABLE_PROFILE,
      schedulingProfile: "cooperative",
      adapterInfo: {
        vendor: "test-vendor",
        architecture: "test-architecture",
        device: "test-device",
        description: "test-adapter",
      },
      adapterFeatures: ["shader-f16"],
      deviceFeatures: ["shader-f16"],
      stockFeatures: TEST_STOCK_FEATURES,
      requiredFeatures: ["shader-f16"],
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
