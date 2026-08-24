import { describe, expect, it, vi } from "vitest";

import {
  DEFAULT_ACE_PLANNER_CONFIGURATION,
  type AceGenerationRequest,
} from "../src/api.js";
import {
  aceRuntimePackageFiles,
  type AceAcquiredModelFiles,
} from "../src/model/acquire.js";
import type { AceGpuTensorPhase } from "../src/model/gpu-tensors.js";
import type {
  AcePackageManifest,
  AceTensorPhase,
} from "../src/model/manifest.js";
import {
  ACE_OPT_0054_VAE_REVISION7_MANIFEST_BYTES,
  ACE_OPT_0054_VAE_REVISION7_MANIFEST_SHA256,
  ACE_REFERENCE_MANIFEST_SHA256,
  type AceLoadedPackageManifest,
} from "../src/model/package.js";
import type { AceDiagnostic } from "../src/runtime/diagnostics.js";
import {
  createAceMainAcquisitionManifest,
  createAceOpt0037DitK4AcquisitionManifest,
  createAceWebGpuPipelineBackendForTest,
  resolveAceDitDensePackageRuntimeIdentity,
  resolveAceVaePackageRuntimeIdentity,
  type AceOpt0018DitCheckpoint,
  type AceOpt0034DitCheckpoint,
  type AceOpt0056DitCheckpoint,
  type AceOpt0067DitCheckpoint,
  type AceOpt0080DitCheckpoint,
  type AceOpt0080FullDitCheckpoint,
  type AceOpt0080ProductEvidence,
  type AceOpt0081RepresentativeConditioningAuthority,
  type AceSamplerScheduleDiagnosticEvidence,
  type AceWebGpuPipelineDependencies,
} from "../src/runtime/webgpu-pipeline.js";
import { isAceGenerationResultValue } from "../src/runtime/protocol.js";
import {
  AceInitializationProgressSequence,
  AceProgressSequence,
  generationStagePlan,
  type AceGenerationProgress,
  type AceInitializationProgress,
} from "../src/runtime/stages.js";
import type { LoadedAceTokenizer } from "../src/tokenizer/loader.js";
import type { AceQwenBpeTokenizer } from "../src/tokenizer/qwen-bpe.js";
import type { AceConditioningGpuProgress } from "../src/webgpu/conditioning-executor.js";
import {
  AceOpt0081RepresentativeInjectedSetupFailure,
  type AceDitGpuBackendProgress,
  type AceOpt0018DitCommandProfile,
  type AceOpt0034DitSchedulingProfile,
  type AceOpt0067DitCommandProfile,
  type AceOpt0080DitCommandProfile,
  type AceOpt0080FullDitCommandProfile,
  type AceOpt0081RepresentativeDitSession,
  type AceOpt0081RepresentativeSetupCleanupEvidence,
} from "../src/webgpu/dit-backend.js";
import type { AceVaeChunkGpuBackendProgress } from "../src/webgpu/vae-backend.js";
import {
  ACE_OPT_0080_VAE_PRODUCTION_SCHEDULING_POLICY,
  resolveOpt0080VaeProductionWindowSchedulingProfile,
  type AceOpt0011Fp16VaeWindowFamilyProfile,
  type AceOpt0080VaeRunOptions,
} from "../src/webgpu/vae-fp16-backend.js";
import type { AceVaeDecodeWindow } from "../src/webgpu/vae-chunks.js";
import type { AceGpuRuntimeEvent } from "../src/webgpu/device.js";
import {
  ACE_REFERENCE_PORTABLE_PROFILE,
  ACE_REFERENCE_SUBGROUP_PROFILE,
} from "../src/webgpu/capabilities.js";
import {
  ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_KERNEL_SET_ID,
  ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
  ACE_OPT_0088_DIT_PORTABLE_ATTENTION_KERNEL_SET_ID,
} from "../src/webgpu/dit-attention-profile.js";
import {
  ACE_OPT_0009_DIT_DENSE_KERNEL_SET_ID,
  ACE_OPT_0009_DIT_DENSE_MANIFEST_BYTES,
  ACE_OPT_0009_DIT_DENSE_MANIFEST_SHA256,
  ACE_OPT_0009_DIT_DENSE_RUNTIME_PROFILE,
  ACE_OPT_0088_DIT_DENSE_PORTABLE_KERNEL_SET_ID,
  ACE_OPT_0037_DIT_K4_LAYER_BYTES,
  ACE_OPT_0037_DIT_K4_MANIFEST_BYTES,
  ACE_OPT_0037_DIT_K4_MANIFEST_SHA256,
  ACE_OPT_0037_DIT_K4_WEIGHT_FILES,
  ACE_OPT_0056_DIT_SELECTIVE_K4_RUNTIME_PROFILE,
  ACE_REFERENCE_DIT_SHARED_WEIGHT_BYTES,
  isAceReferenceDitLayerWeightFile,
} from "../src/webgpu/dit-fp16-package.js";
import {
  ACE_OPT_0055_SIX_SAMPLER_SCHEDULE_PROFILE_ID,
  ACE_OPT_0065_FIVE_SAMPLER_SCHEDULE_PROFILE_ID,
  ACE_TURBO_EIGHT_SAMPLER_SCHEDULE_PROFILE_ID,
} from "../src/webgpu/dit-sampler-profile.js";
import {
  ACE_OPT_0056_SELECTIVE_DENSE_KERNEL_SET_ID,
  type AceOpt0056DenseOperation,
  type AceOpt0056DenseRouteProfile,
} from "../src/webgpu/kernels/dit-dense-fp16-k4-selective-exact.js";
import { ACE_OPT_0056_DENSE_K4_EXACT_KERNEL_ID } from
  "../src/webgpu/kernels/dit-dense-fp16-k4-exact.js";
import { ACE_OPT_0037_DENSE_K4_KERNEL_ID } from
  "../src/webgpu/kernels/dit-dense-fp16-k4-production.js";
import {
  ACE_OPT_0011_VAE_FP16_WEIGHT_FILES,
} from "../src/webgpu/vae-fp16-package.js";
import {
  ACE_OPT_0035_VAE_FP16_C2378_WORKSPACE_BYTES,
  planAceOpt0011Fp16VaeWindowDynamicControls,
} from "../src/webgpu/vae-fp16-decoder.js";
import {
  ACE_OPT_0070_VAE_C2378_WINDOW_RUNTIME_PROFILE,
  ACE_VAE_CAPPED_C2176_MAXIMUM_WINDOW_FRAMES,
  ACE_VAE_CAPPED_C2176_REQUIRED_WORKSPACE_BYTES,
} from "../src/webgpu/vae-window-profile.js";
import {
  ACE_OPT_0028_VAE_FP16_MANIFEST_BYTES,
  ACE_OPT_0028_VAE_FP16_MANIFEST_SHA256,
  ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_PROFILE,
  ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PROFILE,
  ACE_OPT_0072_VAE_FP16_FIXED32_DUAL_K4_PRODUCTION_RUNTIME_PROFILE,
  ACE_OPT_0088_VAE_FP16_PORTABLE_DUAL_K4_PROFILE,
} from "../src/webgpu/vae-fp16-profile.js";
import {
  testDiagnostics,
  testGenerationRequest,
  testInitializeMessage,
} from "./runtime-fixtures.js";

describe("concrete WebGPU pipeline coordinator", () => {
  it("physically replaces all 48 reference layer shards at zero DiT byte delta", () => {
    const layerFiles = ACE_OPT_0037_DIT_K4_WEIGHT_FILES.map(
      (name, index) => ({
        name,
        byteLength: index % 2 === 0 ? 121_668_608 : 4_198_400,
        sha256: String(index).padStart(64, "0"),
        kind: "weights" as const,
      }),
    );
    const sharedFiles = [
      {
        name: "weights/dit/shared-00.bin",
        byteLength: ACE_REFERENCE_DIT_SHARED_WEIGHT_BYTES,
        sha256: "a".repeat(64),
        kind: "weights" as const,
      },
    ];
    const metadataFile = {
      name: "LICENSE.txt",
      byteLength: 12_345,
      sha256: "c".repeat(64),
      kind: "license" as const,
    };
    const reference = {
      profile: "reference",
      files: [
        ...layerFiles,
        ...sharedFiles,
        metadataFile,
        {
          name: "weights/vae/test.bin",
          byteLength: 4,
          sha256: "b".repeat(64),
          kind: "weights",
        },
      ],
      tensors: Object.fromEntries([
        ...layerFiles.map((file, index) => [
          `ace.decoder.layers.${Math.floor(index / 2)}.test-${index}`,
          { phase: "dit", shard: file.name },
        ]),
        ["ace.decoder.shared", { phase: "dit", shard: sharedFiles[0]!.name }],
        ["vae.test", { phase: "vae", shard: "weights/vae/test.bin" }],
      ]),
    } as unknown as AcePackageManifest;
    const candidate = {
      profile: "fp16-dit-dense-experimental",
      provenance: { converterRevision: 8 },
      files: layerFiles,
    } as unknown as AcePackageManifest;

    const mainSelection = createAceMainAcquisitionManifest(reference);
    const candidateSelection = createAceOpt0037DitK4AcquisitionManifest(
      candidate,
    );
    expect(mainSelection.files).toEqual([...sharedFiles, metadataFile]);
    expect(aceRuntimePackageFiles(mainSelection)).toEqual(sharedFiles);
    expect(mainSelection.files.reduce(
      (sum, file) => sum + file.byteLength,
      0,
    )).toBe(ACE_REFERENCE_DIT_SHARED_WEIGHT_BYTES + metadataFile.byteLength);
    expect(aceRuntimePackageFiles(mainSelection).reduce(
      (sum, file) => sum + file.byteLength,
      0,
    )).toBe(ACE_REFERENCE_DIT_SHARED_WEIGHT_BYTES);
    expect(mainSelection.files.some((file) =>
      isAceReferenceDitLayerWeightFile(file.name)
    )).toBe(false);
    expect(candidateSelection.files).toEqual(layerFiles);
    expect(candidateSelection.files.reduce(
      (sum, file) => sum + file.byteLength,
      0,
    )).toBe(ACE_OPT_0037_DIT_K4_LAYER_BYTES);
    expect(
      ACE_REFERENCE_DIT_SHARED_WEIGHT_BYTES +
        ACE_OPT_0037_DIT_K4_LAYER_BYTES,
    ).toBe(3_150_917_888);
  });

  it("runs the direct graph in exact stage order and emits the semantic zero-work skip", async () => {
    const harness = createHarness();
    const initialization = await initialize(harness);
    const initializationSequence = new AceInitializationProgressSequence();
    initialization.progress.forEach((event) => initializationSequence.accept(event));

    const generationProgress: AceGenerationProgress[] = [];
    const diagnostics: Array<Readonly<{
      readonly code: string;
      readonly details?: Readonly<Record<string, unknown>>;
    }>> = [];
    const result = await harness.backend.generate(testGenerationRequest(), {
      signal: new AbortController().signal,
      captureTrace: true,
      onProgress: (event) => generationProgress.push(event),
      onDiagnostic: (diagnostic) => diagnostics.push(diagnostic),
    });

    const sequence = new AceProgressSequence();
    generationProgress.forEach((event) => sequence.accept(event));
    expect(uniqueStages(generationProgress)).toEqual(
      generationStagePlan(false),
    );
    expect(
      generationProgress.find((event) =>
        event.stage === "semantic-detokenizer"
      ),
    ).toMatchObject({
      completedUnits: 0,
      totalUnits: 0,
      message: "Skipped: planner disabled",
    });
    expect(harness.loadedPhases).toEqual([
      "text",
      "conditioner+constants",
      "dit",
      "dit-dense",
      "vae",
    ]);
    expect(harness.loadedPhases).not.toContain("planner");
    expect(harness.loadedPhases).not.toContain("semantic");
    const deviceRequest = harness.requestDevice.mock.calls[0]![0];
    // The pipeline hard-requires only shader-f16; the execution profile
    // selected inside the device request re-adds "subgroups" on fixed-32
    // subgroup adapters and omits it on portable adapters.
    expect(deviceRequest).toMatchObject({
      modelProfile: "reference-bf16",
      requiredFeatures: ["shader-f16"],
    });
    expect(deviceRequest.requiredLimits).toBeUndefined();
    expect(deviceRequest.deriveRequiredLimits!(testDiagnostics()
      .capabilities.adapterLimits)).toEqual({
      maxBufferSize: 251_658_240,
      maxStorageBufferBindingSize: 251_658_240,
    });
    expect(harness.packageEvents).toEqual([
      "main-manifest",
      "dit-dense-manifest",
      "vae-manifest",
      "main-acquire",
      "dit-dense-acquire",
      "phase-text",
      "phase-conditioner+constants",
      "phase-dit",
      "phase-dit-dense",
      "dit-destroy",
      "vae-acquire",
      "phase-vae",
      "vae-backend",
    ]);
    expect(harness.packageEvents.indexOf("vae-acquire")).toBeGreaterThan(
      harness.packageEvents.indexOf("dit-destroy"),
    );
    expect(isAceGenerationResultValue(result)).toBe(true);
    expect(result.audioStorageId).toBe(
      "ace-00000000-0000-4000-8000-000000000001",
    );
    expect(result.frameCount).toBe(480_000);
    expect(result.metrics.peakTrackedGpuBytes).toBe(300);
    expect(result.metrics.cooperativeGpuQueueDrains).toBeGreaterThan(0);
    expect(result.metrics.stageTimings.map(({ stage }) => stage)).toEqual(
      generationStagePlan(false),
    );
    expect(generationProgress).toContainEqual(expect.objectContaining({
      stage: "dit-load",
      completedUnits: 1,
      totalUnits: 3,
      message: "Loading dit-test: 1024/1024 bytes",
    }));
    const directDecoderQuantumCount = exactVaeDecoderQuantumCount(250);
    expect(generationProgress).toContainEqual(expect.objectContaining({
      stage: "vae-decode",
      completedUnits: 1 / directDecoderQuantumCount,
      unit: "vae-chunks",
      message:
        `VAE window 1/1 · quantum 1/${directDecoderQuantumCount}`,
    }));
    expect(diagnostics).toContainEqual(expect.objectContaining({
      code: "ACE_PLANNER_CONDITIONING_RESOLVED",
      details: expect.objectContaining({
        plannerEnabled: false,
        instrumental: true,
        semanticCodeCount: 0,
        semanticCodeIds: [],
        samplingOracleId: null,
      }),
    }));
    expect(diagnostics).toContainEqual(expect.objectContaining({
      code: "ACE_CONDITIONING_TENSORS_READY",
      details: expect.objectContaining({
        mode: "direct",
        lyricConditionPrefixComplete: true,
      }),
    }));
    const familyDiagnostics = diagnostics.filter(({ code }) =>
      code === "ACE_VAE_FP16_FAMILY_PROFILE"
    );
    expect(familyDiagnostics).toHaveLength(1);
    expect(familyDiagnostics[0]).toEqual(expect.objectContaining({
      details: expect.objectContaining({
        schema: "ace-vae-fp16-family-profile-v1",
        runtimeProfile:
          ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_PROFILE.id,
        physicalRuntimeProfile:
          ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_PROFILE.id,
        kernelSetId:
          ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_PROFILE.kernelSetId,
        precisionMapSha256:
          ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_PROFILE
            .precisionMapSha256,
        windowCount: 1,
        windowInputFrames: [250],
        quantaPerCommandBuffer: 64,
        decoderBatchCount: 1,
        decoderQuantumCount: 48,
        decoderSubmitThroughDrainMs: 50,
        homogeneousBatchCount: 0,
        homogeneousSubmitThroughDrainMs: 0,
        homogeneousBatchCoverage: 0,
        homogeneousTimeCoverage: 0,
        mixedBatchCount: 1,
        mixedSubmitThroughDrainMs: 10,
        k7Conv1dBatchCount: 0,
        k7Conv1dQuantumCount: 0,
        k7Conv1dSubmitThroughDrainMs: 0,
        snakeSubmitThroughDrainMs: 0,
        addSubmitThroughDrainMs: 0,
        readbackExcluded: true,
      }),
    }));
  });

  it("runs the complete OPT-0070 production path without a diagnostic checkpoint", async () => {
    const harness = createHarness({
      largeVaeLimits: true,
      mainManifestSha256: ACE_REFERENCE_MANIFEST_SHA256,
    });
    const base = testInitializeMessage();
    const candidate = {
      ...base,
      configuration: {
        ...base.configuration,
        manifestSha256: ACE_REFERENCE_MANIFEST_SHA256,
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
    // Below even the C512 workspace there is no downshift: fail closed.
    const insufficient = createHarness({ vaeLimitBytes: 200_000_000 });
    await expect(initialize(insufficient, candidate)).rejects.toThrow(
      /production device did not enable/,
    );
    expect(insufficient.deviceDestroy).toHaveBeenCalledTimes(1);

    await initialize(harness, candidate);
    const diagnostics: AceDiagnostic[] = [];
    await expect(harness.backend.generate(testGenerationRequest(), {
      signal: new AbortController().signal,
      captureTrace: true,
      onProgress: vi.fn(),
      onDiagnostic: (diagnostic) => diagnostics.push(diagnostic),
    })).resolves.toMatchObject({ frameCount: 480_000 });

    const productionRequest = harness.requestDevice.mock.calls[0]![0];
    expect(productionRequest.requiredLimits).toBeUndefined();
    expect(productionRequest.deriveRequiredLimits!(Object.freeze({
      ...testDiagnostics().capabilities.adapterLimits,
      maxBufferSize: 2_000_000_000,
      maxStorageBufferBindingSize: 2_000_000_000,
    }))).toEqual({
      maxBufferSize: ACE_OPT_0035_VAE_FP16_C2378_WORKSPACE_BYTES,
      maxStorageBufferBindingSize:
        ACE_OPT_0035_VAE_FP16_C2378_WORKSPACE_BYTES,
    });
    expect(harness.lastDitAttentionRuntimeProfile).toBe(
      ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
    );
    expect(harness.lastDitSubmissionPolicy).toBe("depth2-phase-epoch4");
    expect(harness.lastDitCaptureOpt0062AttentionIdentity).toBe(false);
    expect(harness.lastVaeMaximumWindowFrames).toBe(2_378);
    expect(harness.lastVaePlanChunkFrames).toBe(2_378);
    expect(harness.lastVaeRuntimeProfile).toBe(
      ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PROFILE.id,
    );
    expect(harness.lastVaeProductionSchedulingPolicy).toBeUndefined();
    expect(diagnostics).toContainEqual(expect.objectContaining({
      code: "ACE_VAE_FP16_FAMILY_PROFILE",
      details: expect.objectContaining({
        runtimeProfile:
          ACE_OPT_0072_VAE_FP16_FIXED32_DUAL_K4_PRODUCTION_RUNTIME_PROFILE,
        physicalRuntimeProfile:
          ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PROFILE.id,
        kernelSetId:
          ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PROFILE.kernelSetId,
        precisionMapSha256:
          ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PROFILE
            .precisionMapSha256,
      }),
    }));
    expect(harness.packageEvents).toContain("vae-backend");
  });

  it("runs the OPT-0070 production tuple end to end on a portable no-subgroups device", async () => {
    const harness = createHarness({
      portable: true,
      largeVaeLimits: true,
      mainManifestSha256: ACE_REFERENCE_MANIFEST_SHA256,
    });
    await initialize(harness, opt0080InitializeMessage());

    const deviceRequest = harness.requestDevice.mock.calls[0]![0];
    expect(deviceRequest).toMatchObject({
      modelProfile: "reference-bf16",
      requiredFeatures: ["shader-f16"],
    });

    const result = await harness.backend.generate(testGenerationRequest(), {
      signal: new AbortController().signal,
      onProgress: vi.fn(),
      onDiagnostic: vi.fn(),
    });

    expect(result.diagnostics.executionProfile).toEqual(
      ACE_REFERENCE_PORTABLE_PROFILE,
    );
    expect(result.diagnostics.ditDenseKernelSetId).toBe(
      ACE_OPT_0088_DIT_DENSE_PORTABLE_KERNEL_SET_ID,
    );
    expect(result.diagnostics.ditAttentionRuntimeProfile).toBe(
      ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
    );
    expect(result.diagnostics.ditAttentionKernelSetId).toBe(
      ACE_OPT_0088_DIT_PORTABLE_ATTENTION_KERNEL_SET_ID,
    );
    expect(result.diagnostics.vaeKernelSetId).toBe(
      ACE_OPT_0088_VAE_FP16_PORTABLE_DUAL_K4_PROFILE.kernelSetId,
    );
    expect(result.diagnostics.vaePrecisionMapSha256).toBe(
      ACE_OPT_0088_VAE_FP16_PORTABLE_DUAL_K4_PROFILE.precisionMapSha256,
    );
    // The portable diagnostics tuple round-trips the protocol validator.
    expect(isAceGenerationResultValue(result)).toBe(true);
    expect(harness.lastVaeRuntimeProfile).toBe(
      ACE_OPT_0088_VAE_FP16_PORTABLE_DUAL_K4_PROFILE.id,
    );
    // The depth-two DiT and OPT-0080 VAE policies stay fixed32-pinned.
    expect(harness.lastDitSubmissionPolicy).toBeUndefined();
    expect(harness.lastVaeProductionSchedulingPolicy).toBeUndefined();
    expect(harness.packageEvents).toContain("vae-backend");
  });

  it("fails closed when a portable device still enabled or lacks required features", async () => {
    // A portable execution profile with subgroups still enabled on the
    // device is not a coherent tuple.
    const leaked = createHarness({
      portable: true,
      portableDeviceSubgroupsLeak: true,
      largeVaeLimits: true,
      mainManifestSha256: ACE_REFERENCE_MANIFEST_SHA256,
    });
    await expect(initialize(leaked, opt0080InitializeMessage())).rejects
      .toThrow(/production device did not enable/);
    expect(leaked.deviceDestroy).toHaveBeenCalledTimes(1);

    // shader-f16 stays hard-required by both accepted tuples.
    const withoutShaderF16 = createHarness({
      portable: true,
      missingShaderF16: true,
      largeVaeLimits: true,
      mainManifestSha256: ACE_REFERENCE_MANIFEST_SHA256,
    });
    await expect(initialize(withoutShaderF16, opt0080InitializeMessage()))
      .rejects.toThrow(/production device did not enable/);
    expect(withoutShaderF16.deviceDestroy).toHaveBeenCalledTimes(1);
  });

  it("rejects portable initialization outside the OPT-0072 production VAE tuple", async () => {
    const harness = createHarness({
      portable: true,
      largeVaeLimits: true,
    });
    await expect(initialize(harness)).rejects.toThrow(
      /portable mixed VAE accepts only the OPT-0072 production profile/,
    );
  });

  it("resolves backend-keyed package identities and rejects portable non-oracle pairs", () => {
    const opt0009 = {
      manifestUrl: "https://example.test/rev7/manifest.json",
      manifestSha256: ACE_OPT_0009_DIT_DENSE_MANIFEST_SHA256,
      runtimeProfile: ACE_OPT_0009_DIT_DENSE_RUNTIME_PROFILE,
    } as const;
    expect(resolveAceDitDensePackageRuntimeIdentity(opt0009))
      .toMatchObject({ kernelSetId: ACE_OPT_0009_DIT_DENSE_KERNEL_SET_ID });
    expect(resolveAceDitDensePackageRuntimeIdentity(opt0009, "subgroups"))
      .toMatchObject({ kernelSetId: ACE_OPT_0009_DIT_DENSE_KERNEL_SET_ID });
    expect(resolveAceDitDensePackageRuntimeIdentity(opt0009, "portable"))
      .toMatchObject({
        role: "opt-0009-rev7-oracle",
        kernelSetId: ACE_OPT_0088_DIT_DENSE_PORTABLE_KERNEL_SET_ID,
      });
    expect(() =>
      resolveAceDitDensePackageRuntimeIdentity({
        manifestUrl: "https://example.test/rev8/manifest.json",
        manifestSha256: ACE_OPT_0037_DIT_K4_MANIFEST_SHA256,
        runtimeProfile: "opt-0037-k4-fp16-partials-v1",
      }, "portable")
    ).toThrow(/portable mixed DiT accepts only the OPT-0009 rev7 oracle/);

    const opt0072 = {
      manifestUrl: "https://example.test/rev7-vae/manifest.json",
      manifestSha256: ACE_OPT_0054_VAE_REVISION7_MANIFEST_SHA256,
      runtimeProfile:
        ACE_OPT_0072_VAE_FP16_FIXED32_DUAL_K4_PRODUCTION_RUNTIME_PROFILE,
      windowRuntimeProfile: ACE_OPT_0070_VAE_C2378_WINDOW_RUNTIME_PROFILE,
      maxWindowFrames: 2_378,
    } as const;
    expect(resolveAceVaePackageRuntimeIdentity(opt0072)).toMatchObject({
      role: "opt-0072-rev7-production",
      physicalRuntimeProfile:
        ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PROFILE.id,
      kernelSetId:
        ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PROFILE.kernelSetId,
    });
    expect(resolveAceVaePackageRuntimeIdentity(opt0072, "portable"))
      .toMatchObject({
        role: "opt-0072-rev7-production",
        physicalRuntimeProfile:
          ACE_OPT_0088_VAE_FP16_PORTABLE_DUAL_K4_PROFILE.id,
        kernelSetId:
          ACE_OPT_0088_VAE_FP16_PORTABLE_DUAL_K4_PROFILE.kernelSetId,
        precisionMapSha256:
          ACE_OPT_0088_VAE_FP16_PORTABLE_DUAL_K4_PROFILE.precisionMapSha256,
      });
    expect(() =>
      resolveAceVaePackageRuntimeIdentity({
        manifestUrl: "https://example.test/vae/manifest.json",
        manifestSha256: ACE_OPT_0028_VAE_FP16_MANIFEST_SHA256,
        runtimeProfile: ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_PROFILE.id,
        maxWindowFrames: 512,
      }, "portable")
    ).toThrow(/portable mixed VAE accepts only the OPT-0072 production/);
  });

  it("downshifts the production C2378 windows to C512 on one-GiB adapters", async () => {
    // The default fixture limits are exactly 2^30 bytes: the iOS adapter cap.
    // The capped C2176 workspace would bind, but its three whole-window
    // workspaces total ~3.2 GB; the C512 baseline geometry (~755 MB total)
    // is selected so an iPhone-class tab survives the VAE phase.
    const harness = createHarness({
      mainManifestSha256: ACE_REFERENCE_MANIFEST_SHA256,
    });
    await initialize(harness, opt0080InitializeMessage());

    const cappedRequest = harness.requestDevice.mock.calls[0]![0];
    expect(cappedRequest.requiredLimits).toBeUndefined();
    expect(cappedRequest.deriveRequiredLimits!(
      testDiagnostics().capabilities.adapterLimits,
    )).toEqual({
      maxBufferSize: 251_658_240,
      maxStorageBufferBindingSize: 251_658_240,
    });
    expect(ACE_VAE_CAPPED_C2176_REQUIRED_WORKSPACE_BYTES)
      .toBeLessThanOrEqual(1_073_741_824);

    // 30 s is 750 latent frames: one window under C2176/C2378 but two under
    // C512, so the scheduling receipt must validate against the effective
    // adapter-derived plan, not the configured C2378 identity.
    const result = await harness.backend.generate(testGenerationRequest({
      durationSeconds: 30,
    }), {
      signal: new AbortController().signal,
      onProgress: vi.fn(),
      onDiagnostic: vi.fn(),
    });
    expect(result.frameCount).toBe(1_440_000);
    expect(result.metrics.vaeScheduling?.windows).toHaveLength(2);
    expect(isAceGenerationResultValue(result)).toBe(true);

    expect(harness.lastVaeMaximumWindowFrames).toBe(512);
    expect(harness.lastVaePlanChunkFrames).toBe(512);
    expect(harness.lastVaeRuntimeProfile).toBe(
      ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PROFILE.id,
    );
    // OPT-0080 depth-two evidence covers only C2378 window families.
    expect(harness.lastVaeProductionSchedulingPolicy).toBeUndefined();
  });

  it("keeps the capped C2176 windows for sub-C2378 adapters above one GiB", async () => {
    // Desktop-class memory with only bindability short of C2378: the PR #55
    // capped contract remains the selection, unchanged.
    const harness = createHarness({
      vaeLimitBytes: 1_100_000_000,
      mainManifestSha256: ACE_REFERENCE_MANIFEST_SHA256,
    });
    await initialize(harness, opt0080InitializeMessage());

    const cappedRequest = harness.requestDevice.mock.calls[0]![0];
    expect(cappedRequest.deriveRequiredLimits!(Object.freeze({
      ...testDiagnostics().capabilities.adapterLimits,
      maxBufferSize: 1_100_000_000,
      maxStorageBufferBindingSize: 1_100_000_000,
    }))).toEqual({
      maxBufferSize: ACE_VAE_CAPPED_C2176_REQUIRED_WORKSPACE_BYTES,
      maxStorageBufferBindingSize:
        ACE_VAE_CAPPED_C2176_REQUIRED_WORKSPACE_BYTES,
    });

    const result = await harness.backend.generate(testGenerationRequest({
      durationSeconds: 30,
    }), {
      signal: new AbortController().signal,
      onProgress: vi.fn(),
      onDiagnostic: vi.fn(),
    });
    expect(result.frameCount).toBe(1_440_000);
    expect(result.metrics.vaeScheduling?.windows).toHaveLength(1);
    expect(isAceGenerationResultValue(result)).toBe(true);

    expect(harness.lastVaeMaximumWindowFrames).toBe(
      ACE_VAE_CAPPED_C2176_MAXIMUM_WINDOW_FRAMES,
    );
    expect(harness.lastVaePlanChunkFrames).toBe(
      ACE_VAE_CAPPED_C2176_MAXIMUM_WINDOW_FRAMES,
    );
    expect(harness.lastVaeProductionSchedulingPolicy).toBeUndefined();
  });

  it("selects VAE depth two only for seam-free direct C2314 production windows", async () => {
    const harness = createOpt0080Harness();
    await initialize(harness, opt0080InitializeMessage());

    const result = await harness.backend.generate(testGenerationRequest({
      durationSeconds: 96,
    }), {
      signal: new AbortController().signal,
      onProgress: vi.fn(),
      onDiagnostic: vi.fn(),
    });

    expect(harness.lastVaeProductionSchedulingPolicy).toBe(
      ACE_OPT_0080_VAE_PRODUCTION_SCHEDULING_POLICY,
    );
    expect(result.metrics.vaeScheduling).toEqual({
      schema: "ace-vae-window-scheduling-receipt-v1",
      selectedProductionPolicy:
        ACE_OPT_0080_VAE_PRODUCTION_SCHEDULING_POLICY,
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
        expect.objectContaining({
          windowIndex: 1,
          latentWindowFrames: 214,
          selection: "production",
          schedulingProfile: "depth1-epoch1",
          maximumOutstandingCommandBuffers: 1,
        }),
      ],
    });
    expect(isAceGenerationResultValue(result)).toBe(true);
  });

  it("keeps depth two disabled for an unauthorised main manifest", async () => {
    const harness = createHarness({ largeVaeLimits: true });
    await initialize(harness, opt0080InitializeMessage("a".repeat(64)));

    await expect(harness.backend.generate(testGenerationRequest(), {
      signal: new AbortController().signal,
      onProgress: vi.fn(),
      onDiagnostic: vi.fn(),
    })).resolves.toMatchObject({ frameCount: 480_000 });

    expect(harness.lastDitSubmissionPolicy).toBeUndefined();
    expect(harness.lastVaeProductionSchedulingPolicy).toBeUndefined();
  });

  it("keeps depth two disabled for the unaudited planner-enabled tuple", async () => {
    const harness = createOpt0080Harness();
    await initialize(harness, opt0080InitializeMessage());

    await expect(harness.backend.generate(testGenerationRequest({
      instrumental: false,
      lyrics: "hello",
      planner: DEFAULT_ACE_PLANNER_CONFIGURATION,
    }), {
      signal: new AbortController().signal,
      onProgress: vi.fn(),
      onDiagnostic: vi.fn(),
    })).resolves.toMatchObject({ frameCount: 480_000 });

    expect(harness.lastDitSubmissionPolicy).toBeUndefined();
    expect(harness.lastVaeProductionSchedulingPolicy).toBeUndefined();
  });

  it("keeps depth two disabled for exact-tuple OPT-0064 API capture", async () => {
    const harness = createOpt0080Harness();
    await initialize(harness, opt0080InitializeMessage());
    const onEvent = vi.fn();

    await expect(harness.backend.generate(testGenerationRequest(), {
      signal: new AbortController().signal,
      onProgress: vi.fn(),
      onDiagnostic: vi.fn(),
      opt0064Capture: { onEvent },
    })).resolves.toMatchObject({ frameCount: 480_000 });

    expect(harness.lastDitSubmissionPolicy).toBeUndefined();
    expect(harness.lastVaeProductionSchedulingPolicy).toBeUndefined();
    expect(onEvent).toHaveBeenCalled();
  });

  it.each([
    ["ordinary", undefined, "depth2-phase-epoch4"],
    ["forced control", "depth1-epoch1", "depth1-epoch1"],
    ["forced candidate", "depth2-phase-epoch4", "depth2-phase-epoch4"],
  ] as const)(
    "captures complete product evidence with %s scheduling",
    async (_label, submissionPolicyOverride, effectiveSubmissionPolicy) => {
      const harness = createOpt0080Harness();
      await initialize(harness, opt0080InitializeMessage());
      let evidence: AceOpt0080ProductEvidence | undefined;

      const result = await harness.backend.generate(testGenerationRequest(), {
        signal: new AbortController().signal,
        onProgress: vi.fn(),
        onDiagnostic: vi.fn(),
        opt0080ProductRun: {
          ...(submissionPolicyOverride === undefined
            ? {}
            : { submissionPolicyOverride }),
          onEvidence(value) {
            evidence = value;
          },
        },
      });

      expect(harness.lastDitSubmissionPolicy).toBe(effectiveSubmissionPolicy);
      expect(harness.lastVaeProductionSchedulingPolicy).toBeUndefined();
      expect(evidence).toMatchObject({
        schema: "ace-opt-0080-product-integration-evidence-v1",
        selectedProductionPolicy: "depth2-phase-epoch4",
        effectiveSubmissionPolicy,
        result,
      });
      expect(evidence?.finalLatentRawU32.buffer).toBe(
        evidence?.finalLatent.buffer,
      );
      expect(evidence?.rawSnapshot.size).toBe(
        evidence?.vaePlan.outputFloat32Bytes,
      );
    },
  );

  it.each([
    ["forced depth one", "depth1-epoch1", [
      "depth1-epoch1",
      "depth1-epoch1",
    ]],
    ["forced exact C2314 candidate",
      ACE_OPT_0080_VAE_PRODUCTION_SCHEDULING_POLICY,
      ["depth2-phase-epoch4", "depth1-epoch1"]],
  ] as const)(
    "captures %s VAE replay evidence without enabling automatic selection",
    async (_label, vaeSchedulingPolicyOverride, expectedProfiles) => {
      const harness = createOpt0080Harness();
      await initialize(harness, opt0080InitializeMessage());
      let evidence: AceOpt0080ProductEvidence | undefined;

      const result = await harness.backend.generate(testGenerationRequest({
        durationSeconds: 96,
      }), {
        signal: new AbortController().signal,
        onProgress: vi.fn(),
        onDiagnostic: vi.fn(),
        opt0080ProductRun: {
          vaeSchedulingPolicyOverride,
          onEvidence(value) {
            evidence = value;
          },
        },
      });

      expect(harness.lastVaeProductionSchedulingPolicy).toBeUndefined();
      expect(evidence?.vaeSchedulingEvidence?.map(
        ({ schedulingProfile }) => schedulingProfile,
      )).toEqual(expectedProfiles);
      expect(result.metrics.vaeScheduling).toMatchObject({
        selectedProductionPolicy: null,
        benchmarkPolicyOverride: vaeSchedulingPolicyOverride,
        windows: [
          {
            windowIndex: 0,
            latentWindowFrames: 2_314,
            selection: "benchmark-override",
            schedulingProfile: expectedProfiles[0],
          },
          {
            windowIndex: 1,
            latentWindowFrames: 214,
            selection: "benchmark-override",
            schedulingProfile: expectedProfiles[1],
          },
        ].map((expected) => expect.objectContaining(expected)),
      });
    },
  );

  it("rejects product evidence capture outside the direct authenticated tuple", async () => {
    const harness = createHarness({ largeVaeLimits: true });
    await initialize(harness, opt0080InitializeMessage("a".repeat(64)));
    const onEvidence = vi.fn();
    await expect(harness.backend.generate(testGenerationRequest(), {
      signal: new AbortController().signal,
      onProgress: vi.fn(),
      onDiagnostic: vi.fn(),
      opt0080ProductRun: {
        onEvidence,
      },
    })).rejects.toThrow(/exact authenticated production tuple/u);
    expect(onEvidence).not.toHaveBeenCalled();
    expect(harness.lastDitSubmissionPolicy).toBeUndefined();
  });

  it("rejects product evidence capture for a planner-enabled request", async () => {
    const harness = createOpt0080Harness();
    await initialize(harness, opt0080InitializeMessage());
    const onEvidence = vi.fn();

    await expect(harness.backend.generate(testGenerationRequest({
      instrumental: false,
      lyrics: "hello",
      planner: DEFAULT_ACE_PLANNER_CONFIGURATION,
    }), {
      signal: new AbortController().signal,
      onProgress: vi.fn(),
      onDiagnostic: vi.fn(),
      opt0080ProductRun: { onEvidence },
    })).rejects.toThrow(/requires a direct request/u);
    expect(onEvidence).not.toHaveBeenCalled();
    expect(harness.lastDitSubmissionPolicy).toBeUndefined();
  });

  it("stops at the detached M2250 checkpoint without acquiring VAE or audio", async () => {
    const harness = createOpt0080Harness();
    await initialize(harness, opt0080InitializeMessage());
    const controller = new AbortController();
    const sentinel = Object.freeze({ kind: "opt-0018-controlled-stop" });
    let checkpoint: AceOpt0018DitCheckpoint | undefined;
    const diagnostics: Array<Readonly<{
      readonly code: string;
      readonly details?: Readonly<Record<string, unknown>>;
    }>> = [];
    const progress: AceGenerationProgress[] = [];
    await expect(harness.backend.generate(
      testGenerationRequest({ durationSeconds: 180 }),
      {
        signal: controller.signal,
        captureTrace: true,
        onProgress: (event) => progress.push(event),
        onDiagnostic: (diagnostic) => diagnostics.push(diagnostic),
        onDitCheckpoint: (value) => {
          checkpoint = value;
          expect(harness.packageEvents.at(-1)).toBe("dit-destroy");
          expect(progress.at(-1)).toMatchObject({
            stage: "release-dit",
            completedUnits: 1,
          });
          controller.abort(sentinel);
        },
      },
    )).rejects.toBe(sentinel);
    expect(checkpoint).toMatchObject({
      schema: "ace-dit-m2250-checkpoint-v1",
      finalLatentByteLength: 1_152_000,
      finalLatentElementCount: 288_000,
      finalLatentNonFiniteCount: 0,
      finalLatentNonzeroCount: 288_000,
      finalLatentMaxAbs: 0.25,
      profile: {
        schema: "ace-dit-m2250-command-profile-v1",
        graphCommandBufferCount: 2_553,
        readbackCommandBufferCount: 1,
        totalCommandBufferCount: 2_554,
      },
    });
    expect(checkpoint!.finalLatent).toHaveLength(288_000);
    expect(checkpoint!.finalLatentSha256).toMatch(/^[0-9a-f]{64}$/u);
    expect(checkpoint!.stageTimings.at(-1)?.stage).toBe("release-dit");
    expect(diagnostics).toContainEqual(expect.objectContaining({
      code: "ACE_DIT_M2250_FAMILY_PROFILE",
      details: expect.objectContaining({
        schema: "ace-dit-m2250-family-profile-receipt-v1",
        checkpointSchema: "ace-dit-m2250-checkpoint-v1",
        graphCommandBufferCount: 2_553,
        readbackCommandBufferCount: 1,
        totalCommandBufferCount: 2_554,
        descriptorTableSha256:
          "aedf8c74d2bb15601d1385e9c8e9da49e58905ab156a04c99271f8de3633dd76",
        conditionTokens: 98,
        checkpointStopRequested: true,
      }),
    }));
    expect(harness.packageEvents).not.toContain("vae-acquire");
    expect(harness.packageEvents).not.toContain("vae-backend");
    expect(harness.loadedPhases).not.toContain("vae");
    expect(harness.audioBegins).toBe(0);
    expect(harness.phaseDestroyed.get("dit")).toBe(true);
    expect(harness.phaseDestroyed.get("dit-dense")).toBe(true);
    expect(harness.lastDitSubmissionPolicy).toBeUndefined();

    const next = await harness.backend.generate(testGenerationRequest(), {
      signal: new AbortController().signal,
      captureTrace: false,
      onProgress: () => undefined,
      onDiagnostic: () => undefined,
    });
    expect(isAceGenerationResultValue(next)).toBe(true);
    expect(harness.lastDitSubmissionPolicy).toBe("depth2-phase-epoch4");
    expect(harness.audioBegins).toBe(1);
  });

  it("captures eight OPT-0056 sampler taps without changing graph topology", async () => {
    const harness = createOpt0080Harness();
    await initialize(harness, opt0080InitializeMessage());
    const controller = new AbortController();
    const sentinel = Object.freeze({ kind: "opt-0056-controlled-stop" });
    let checkpoint: AceOpt0056DitCheckpoint | undefined;
    const diagnostics: Array<Readonly<{
      readonly code: string;
      readonly details?: Readonly<Record<string, unknown>>;
    }>> = [];
    await expect(harness.backend.generate(
      testGenerationRequest({ durationSeconds: 180 }),
      {
        signal: controller.signal,
        captureTrace: true,
        onProgress: () => undefined,
        onDiagnostic: (diagnostic) => diagnostics.push(diagnostic),
        opt0056DitRun: {
          onCheckpoint(value) {
            checkpoint = value;
            expect(harness.packageEvents.at(-1)).toBe("dit-destroy");
            controller.abort(sentinel);
          },
        },
      },
    )).rejects.toBe(sentinel);
    expect(checkpoint).toMatchObject({
      schema: "ace-dit-opt0056-m2250-trajectory-checkpoint-v1",
      finalLatentByteLength: 1_152_000,
      finalLatentElementCount: 288_000,
      finalLatentNonFiniteCount: 0,
      graphCommandBufferCount: 2_553,
      readbackCommandBufferCount: 1,
      totalCommandBufferCount: 2_554,
      snapshotCopyCount: 8,
      snapshotExtraCommandBufferCount: 0,
      snapshotExtraQueueDrainCount: 0,
    });
    expect(checkpoint?.evaluations).toHaveLength(8);
    expect(checkpoint?.evaluations.map(({ evaluation }) => evaluation))
      .toEqual([0, 1, 2, 3, 4, 5, 6, 7]);
    expect(checkpoint?.evaluations.at(-1)?.latentSha256).toBe(
      checkpoint?.finalLatentSha256,
    );
    expect(checkpoint?.denseRouteProfile).toBeUndefined();
    expect(diagnostics).toContainEqual(expect.objectContaining({
      code: "ACE_DIT_OPT0056_TRAJECTORY",
      details: expect.objectContaining({
        schema: "ace-opt-0056-m2250-trajectory-receipt-v1",
        evaluationCount: 8,
        graphCommandBufferCount: 2_553,
        totalCommandBufferCount: 2_554,
        snapshotExtraCommandBufferCount: 0,
        snapshotExtraQueueDrainCount: 0,
      }),
    }));
    expect(harness.packageEvents).not.toContain("vae-acquire");
    expect(harness.audioBegins).toBe(0);
    expect(harness.lastDitSubmissionPolicy).toBeUndefined();
  });

  it("pauses OPT-0067 after preparation and stops after ordinary eval0 readback", async () => {
    const harness = createOpt0080Harness();
    await initialize(harness, opt0080InitializeMessage());
    const controller = new AbortController();
    const sentinel = Object.freeze({ kind: "opt-0067-controlled-stop" });
    const timingGate = Promise.withResolvers<void>();
    const timingReady = vi.fn(async () => await timingGate.promise);
    let checkpoint: AceOpt0067DitCheckpoint | undefined;
    const diagnostics: AceDiagnostic[] = [];
    const generation = harness.backend.generate(
      testGenerationRequest({ durationSeconds: 180 }),
      {
        signal: controller.signal,
        captureTrace: true,
        onProgress: vi.fn(),
        onDiagnostic: (diagnostic) => diagnostics.push(diagnostic),
        opt0067DitRun: {
          waitForTimingAuthorization: timingReady,
          onCheckpoint(value) {
            checkpoint = value;
            expect(harness.packageEvents.at(-1)).toBe("dit-destroy");
            controller.abort(sentinel);
          },
        },
      },
    );
    await vi.waitFor(() => expect(timingReady).toHaveBeenCalledOnce());
    expect(harness.loadedPhases).toContain("dit");
    expect(harness.loadedPhases).toContain("dit-dense");
    expect(harness.phaseDestroyed.get("dit")).toBe(false);
    expect(harness.phaseDestroyed.get("dit-dense")).toBe(false);
    expect(harness.packageEvents).not.toContain("dit-destroy");
    expect(harness.packageEvents).not.toContain("vae-acquire");
    timingGate.resolve();
    await expect(generation).rejects.toBe(sentinel);
    expect(checkpoint).toMatchObject({
      schema: "ace-dit-opt0067-m2250-evaluation0-checkpoint-v1",
      evaluation: 0,
      resultByteLength: 1_152_000,
      resultElementCount: 288_000,
      resultNonFiniteCount: 0,
      graphCommandBufferCount: 341,
      readbackCommandBufferCount: 1,
      totalCommandBufferCount: 342,
      completedEvaluations: 1,
      evaluationResultExtraCommandBufferCount: 0,
      evaluationResultExtraQueueDrainCount: 0,
      profile: {
        schema: "ace-dit-opt0067-evaluation0-command-profile-v1",
        graphCommandBufferCount: 341,
        evaluationCommandBufferCount: 316,
      },
    });
    expect(checkpoint?.result).toHaveLength(288_000);
    expect(checkpoint?.resultSha256).toMatch(/^[0-9a-f]{64}$/u);
    expect(checkpoint?.attentionRouteProfile).toBeUndefined();
    expect(checkpoint?.actualLayerIdentity).toBeUndefined();
    expect(diagnostics).toContainEqual(expect.objectContaining({
      code: "ACE_DIT_OPT0067_EVALUATION0",
      details: expect.objectContaining({
        schema: "ace-opt-0067-evaluation0-receipt-v1",
        evaluation: 0,
        resultElementCount: 288_000,
        graphCommandBufferCount: 341,
        totalCommandBufferCount: 342,
        completedEvaluations: 1,
        evaluationResultExtraCommandBufferCount: 0,
        evaluationResultExtraQueueDrainCount: 0,
      }),
    }));
    expect(harness.packageEvents).not.toContain("vae-acquire");
    expect(harness.loadedPhases).not.toContain("vae");
    expect(harness.audioBegins).toBe(0);
    expect(harness.lastDitSubmissionPolicy).toBeUndefined();
  });

  it("cleans a rejected OPT-0067 timing authorization before any evaluation", async () => {
    const harness = createHarness();
    await initialize(harness);
    const sentinel = Object.freeze({ kind: "opt-0067-rejected-authorization" });
    const checkpoint = vi.fn();
    await expect(harness.backend.generate(
      testGenerationRequest({ durationSeconds: 180 }),
      {
        signal: new AbortController().signal,
        captureTrace: true,
        onProgress: vi.fn(),
        onDiagnostic: vi.fn(),
        opt0067DitRun: {
          waitForTimingAuthorization: async () => {
            throw sentinel;
          },
          onCheckpoint: checkpoint,
        },
      },
    )).rejects.toBe(sentinel);
    expect(checkpoint).not.toHaveBeenCalled();
    expect(harness.packageEvents.at(-1)).toBe("dit-destroy");
    expect(harness.phaseDestroyed.get("dit")).toBe(true);
    expect(harness.phaseDestroyed.get("dit-dense")).toBe(true);
    expect(harness.packageEvents).not.toContain("vae-acquire");
    expect(harness.loadedPhases).not.toContain("vae");
    expect(harness.audioBegins).toBe(0);
  });

  it.each([
    ["depth-one control", "depth1-epoch1", 341, 342, 340, 341, 1],
    ["depth-two candidate", "opt-0080-depth2-epoch4", 86, 87, 85, 86, 2],
  ] as const)(
    "captures the exact nested OPT-0080 topology for the %s",
    async (
      _label,
      schedulingProfile,
      graphDrains,
      totalDrains,
      graphIdleTurns,
      totalIdleTurns,
      maximumOutstanding,
    ) => {
      const harness = createOpt0080Harness();
      await initialize(harness, opt0080InitializeMessage());
      const controller = new AbortController();
      const sentinel = Object.freeze({
        kind: `opt-0080-${schedulingProfile}-controlled-stop`,
      });
      const timingGate = Promise.withResolvers<void>();
      const timingReady = vi.fn(async () => await timingGate.promise);
      let checkpoint: AceOpt0080DitCheckpoint | undefined;
      const diagnostics: AceDiagnostic[] = [];
      const generation = harness.backend.generate(
        testGenerationRequest({ durationSeconds: 180 }),
        {
          signal: controller.signal,
          captureTrace: true,
          onProgress: vi.fn(),
          onDiagnostic: (diagnostic) => diagnostics.push(diagnostic),
          opt0080DitRun: {
            schedulingProfile,
            waitForTimingAuthorization: timingReady,
            onCheckpoint(value) {
              checkpoint = value;
              expect(harness.packageEvents.at(-1)).toBe("dit-destroy");
              controller.abort(sentinel);
            },
          },
        },
      );
      await vi.waitFor(() => expect(timingReady).toHaveBeenCalledOnce());
      expect(harness.phaseDestroyed.get("dit")).toBe(false);
      expect(harness.phaseDestroyed.get("dit-dense")).toBe(false);
      expect(harness.packageEvents).not.toContain("dit-destroy");
      expect(harness.packageEvents).not.toContain("vae-acquire");
      timingGate.resolve();
      await expect(generation).rejects.toBe(sentinel);
      expect(harness.lastDitSubmissionPolicy).toBeUndefined();

      expect(checkpoint).toMatchObject({
        schema: "ace-dit-opt0080-m2250-evaluation0-checkpoint-v1",
        schedulingProfile,
        evaluation: 0,
        resultByteLength: 1_152_000,
        resultElementCount: 288_000,
        resultNonFiniteCount: 0,
        resultNonzeroCount: 288_000,
        resultMaxAbs: 0.25,
        graphCommandBufferCount: 341,
        readbackCommandBufferCount: 1,
        totalCommandBufferCount: 342,
        completedEvaluations: 1,
        uncapturedWebGpuErrorCount: 0,
        deviceLost: false,
        profile: {
          schema: "ace-dit-opt0080-evaluation0-command-profile-v1",
          schedulingProfile,
          precomputeWallMs: 100,
          evaluationWallMs: 500,
          graphWallMs: 600,
          graphToReadbackObservedIdleMs: 1,
          readbackSubmitThroughCompletionFenceMs: 2,
          readbackMapDetachMs: 1,
          backendWallMs: 604,
          descriptorTable: {
            memberCount: 6_833,
          },
          topology: {
            schedulingProfile,
            descriptorTableMemberCount: 6_833,
            graphCommandBufferCount: 341,
            readbackCommandBufferCount: 1,
            totalCommandBufferCount: 342,
            commandBuffersSubmitted: 342,
            completionFenceRequestedCount: 342,
            completionFenceSettledCount: 342,
            completionFenceRejectedCount: 0,
            graphTrueQueueDrainCount: graphDrains,
            totalTrueQueueDrainCount: totalDrains,
            graphCompletionEpochCount: graphDrains,
            graphCooperativeIdleTurns: graphIdleTurns,
            totalCooperativeIdleTurns: totalIdleTurns,
            graphRequestedCooperativeIdleMs: graphIdleTurns,
            totalRequestedCooperativeIdleMs: totalIdleTurns,
            maximumOutstandingCommandBuffers: maximumOutstanding,
            maximumPendingDescriptorCount: maximumOutstanding,
            pendingDescriptorCountAfterCleanup: 0,
            graphToReadbackRequestedIdleMs: 1,
            readbackSubmitThroughCompletionFenceMs: 2,
          },
        },
      });
      expect(checkpoint?.result).toHaveLength(288_000);
      expect(checkpoint?.resultSha256).toMatch(/^[0-9a-f]{64}$/u);
      expect(checkpoint?.profile.topology.graphCompletionEpochs)
        .toHaveLength(graphDrains);
      expect(checkpoint?.profile.topology.submitThroughCompletionFenceMs)
        .toHaveLength(341);
      expect(checkpoint!.profile.topology.submitThroughCompletionFenceMs
        .reduce((sum, value) => sum + value, 0)).toBeGreaterThan(
          checkpoint!.profile.graphWallMs,
        );
      expect(checkpoint!.profile.descriptorTable.descriptors.slice(0, 341)
        .reduce((sum, descriptor) => sum + descriptor.members.length, 0))
        .not.toBe(checkpoint!.profile.descriptorTable.memberCount);
      const evaluationFirstEpoch = schedulingProfile === "depth1-epoch1"
        ? 25
        : 7;
      expect(checkpoint?.profile.topology.graphCompletionEpochs[
        evaluationFirstEpoch - 1
      ]).toMatchObject({
        phaseIndex: 0,
        lastCommandBufferIndex: 24,
      });
      expect(checkpoint?.profile.topology.graphCompletionEpochs[
        evaluationFirstEpoch
      ]).toMatchObject({
        phaseIndex: 1,
        firstCommandBufferIndex: 25,
      });
      expect(harness.lastDitCaptureOpt0062AttentionIdentity).toBe(false);
      expect(diagnostics).toContainEqual(expect.objectContaining({
        code: "ACE_DIT_OPT0080_COMPLETION_EPOCHS",
        details: expect.objectContaining({
          schema: "ace-opt-0080-evaluation0-receipt-v1",
          schedulingProfile,
          descriptorTableMemberCount: 6_833,
          graphCommandBufferCount: 341,
          totalCommandBufferCount: 342,
          graphTrueQueueDrainCount: graphDrains,
          totalTrueQueueDrainCount: totalDrains,
          graphCompletionEpochCount: graphDrains,
          maximumOutstandingCommandBuffers: maximumOutstanding,
          maximumPendingDescriptorCount: maximumOutstanding,
          identityExtraCommandBufferCount: 0,
          identityExtraQueueDrainCount: 0,
        }),
      }));
      expect(harness.packageEvents).not.toContain("vae-acquire");
      expect(harness.loadedPhases).not.toContain("vae");
      expect(harness.audioBegins).toBe(0);
      expect(harness.phaseDestroyed.get("dit")).toBe(true);
      expect(harness.phaseDestroyed.get("dit-dense")).toBe(true);
    },
  );

  it.each([
    ["depth1-epoch1", false, 2_553, 2_554, 2_552, 2_553, 1],
    ["opt-0080-depth2-epoch4", false, 639, 640, 638, 639, 2],
    ["opt-0080-depth2-epoch4", true, 639, 640, 638, 639, 2],
  ] as const)(
    "captures full OPT-0080 topology for %s (evaluation taps %s)",
    async (
      schedulingProfile,
      captureEvaluationTaps,
      graphDrains,
      totalDrains,
      graphIdleTurns,
      totalIdleTurns,
      maximumOutstanding,
    ) => {
      const harness = createOpt0080Harness();
      await initialize(harness, opt0080InitializeMessage());
      const controller = new AbortController();
      const sentinel = Object.freeze({
        kind: `opt-0080-full-${schedulingProfile}-${captureEvaluationTaps}`,
      });
      let checkpoint: AceOpt0080FullDitCheckpoint | undefined;
      const diagnostics: AceDiagnostic[] = [];
      await expect(harness.backend.generate(
        testGenerationRequest({ durationSeconds: 180 }),
        {
          signal: controller.signal,
          captureTrace: true,
          onProgress: vi.fn(),
          onDiagnostic: (diagnostic) => diagnostics.push(diagnostic),
          opt0080FullDitRun: {
            schedulingProfile,
            ...(captureEvaluationTaps
              ? { captureEvaluationTaps: true as const }
              : {}),
            onCheckpoint(value) {
              checkpoint = value;
              expect(harness.packageEvents.at(-1)).toBe("dit-destroy");
              controller.abort(sentinel);
            },
          },
        },
      )).rejects.toBe(sentinel);
      expect(harness.lastDitSubmissionPolicy).toBeUndefined();

      expect(checkpoint).toMatchObject({
        schema: "ace-dit-opt0080-m2250-full-checkpoint-v1",
        schedulingProfile,
        captureEvaluationTaps,
        evaluationTapInCommandCopyCount: captureEvaluationTaps ? 8 : 0,
        evaluationTapExtraCommandBufferCount: 0,
        evaluationTapExtraQueueDrainCount: 0,
        finalLatentNonFiniteCount: 0,
        finalLatentNonzeroCount: 288_000,
        finalLatentMaxAbs: 0.25,
        graphCommandBufferCount: 2_553,
        readbackCommandBufferCount: 1,
        totalCommandBufferCount: 2_554,
        completedEvaluations: 8,
        profile: {
          schema: "ace-dit-opt0080-full-command-profile-v1",
          schedulingProfile,
          precomputeWallMs: 25,
          evaluationWallMs: [316, 316, 316, 316, 316, 316, 316, 316],
          graphWallMs: 2_553,
          topology: {
            graphTrueQueueDrainCount: graphDrains,
            totalTrueQueueDrainCount: totalDrains,
            graphCompletionEpochCount: graphDrains,
            graphCooperativeIdleTurns: graphIdleTurns,
            totalCooperativeIdleTurns: totalIdleTurns,
            maximumOutstandingCommandBuffers: maximumOutstanding,
            maximumPendingDescriptorCount: maximumOutstanding,
          },
        },
      });
      expect(checkpoint?.finalLatentSha256).toMatch(/^[0-9a-f]{64}$/u);
      expect(checkpoint?.finalLatentRawU32.buffer).toBe(
        checkpoint?.finalLatent.buffer,
      );
      if (captureEvaluationTaps) {
        expect(checkpoint?.evaluationTaps).toHaveLength(8);
      } else {
        expect(checkpoint?.evaluationTaps).toBeUndefined();
      }
      for (const tap of checkpoint?.evaluationTaps ?? []) {
        expect(tap.rawU32.buffer).toBe(tap.result.buffer);
        expect(tap.sha256).toMatch(/^[0-9a-f]{64}$/u);
      }
      expect(checkpoint?.profile.topology.graphCompletionEpochs)
        .toHaveLength(graphDrains);
      expect(checkpoint?.profile.topology.submitThroughCompletionFenceMs)
        .toHaveLength(2_553);
      expect(diagnostics).toContainEqual(expect.objectContaining({
        code: "ACE_DIT_OPT0080_FULL_COMPLETION_EPOCHS",
        details: expect.objectContaining({
          schema: "ace-opt-0080-full-receipt-v1",
          captureEvaluationTaps,
          graphCommandBufferCount: 2_553,
          totalCommandBufferCount: 2_554,
          graphTrueQueueDrainCount: graphDrains,
          totalTrueQueueDrainCount: totalDrains,
        }),
      }));
      expect(harness.packageEvents).not.toContain("vae-acquire");
      expect(harness.phaseDestroyed.get("dit")).toBe(true);
      expect(harness.phaseDestroyed.get("dit-dense")).toBe(true);
    },
  );

  it("cleans a rejected OPT-0080 timing authorization before evaluation", async () => {
    const harness = createOpt0080Harness();
    await initialize(harness, opt0080InitializeMessage());
    const sentinel = Object.freeze({ kind: "opt-0080-rejected-authorization" });
    const checkpoint = vi.fn();
    await expect(harness.backend.generate(
      testGenerationRequest({ durationSeconds: 180 }),
      {
        signal: new AbortController().signal,
        captureTrace: true,
        onProgress: vi.fn(),
        onDiagnostic: vi.fn(),
        opt0080DitRun: {
          schedulingProfile: "opt-0080-depth2-epoch4",
          waitForTimingAuthorization: async () => {
            throw sentinel;
          },
          onCheckpoint: checkpoint,
        },
      },
    )).rejects.toBe(sentinel);
    expect(checkpoint).not.toHaveBeenCalled();
    expect(harness.packageEvents.at(-1)).toBe("dit-destroy");
    expect(harness.phaseDestroyed.get("dit")).toBe(true);
    expect(harness.phaseDestroyed.get("dit-dense")).toBe(true);
    expect(harness.packageEvents).not.toContain("vae-acquire");
    expect(harness.loadedPhases).not.toContain("vae");
    expect(harness.audioBegins).toBe(0);
  });

  it("propagates OPT-0080 completion cancellation and emits no checkpoint", async () => {
    const harness = createOpt0080Harness();
    await initialize(harness, opt0080InitializeMessage());
    const controller = new AbortController();
    const sentinel = Object.freeze({ kind: "opt-0080-completion-abort" });
    const checkpoint = vi.fn();
    const completions: number[] = [];
    const progress: AceGenerationProgress[] = [];
    let progressCountAtAbort = -1;
    await expect(harness.backend.generate(
      testGenerationRequest({ durationSeconds: 180 }),
      {
        signal: controller.signal,
        captureTrace: true,
        onProgress: (event) => progress.push(event),
        onDiagnostic: vi.fn(),
        opt0080DitRun: {
          schedulingProfile: "opt-0080-depth2-epoch4",
          onCommandBufferCompleted(completion) {
            completions.push(completion.commandBufferIndex);
            expect(completion.schedulingProgress).toMatchObject({
              completedCommandBuffers: 1,
              completionFenceRequestedCount: 2,
              completionFenceSettledCount: 1,
              outstandingCommandBuffers: 1,
            });
            progressCountAtAbort = progress.length;
            controller.abort(sentinel);
          },
          onCheckpoint: checkpoint,
        },
      },
    )).rejects.toBe(sentinel);
    expect(completions).toEqual([0]);
    expect(progress).toHaveLength(progressCountAtAbort);
    expect(checkpoint).not.toHaveBeenCalled();
    expect(harness.packageEvents.at(-1)).toBe("dit-destroy");
    expect(harness.phaseDestroyed.get("dit")).toBe(true);
    expect(harness.phaseDestroyed.get("dit-dense")).toBe(true);
    expect(harness.packageEvents).not.toContain("vae-acquire");
    expect(harness.loadedPhases).not.toContain("vae");
    expect(harness.audioBegins).toBe(0);
  });

  it("settles full-graph OPT-0080 completion cancellation before cleanup", async () => {
    const harness = createOpt0080Harness();
    await initialize(harness, opt0080InitializeMessage());
    const controller = new AbortController();
    const sentinel = Object.freeze({ kind: "opt-0080-full-completion-abort" });
    const checkpoint = vi.fn();
    const completions: number[] = [];
    await expect(harness.backend.generate(
      testGenerationRequest({ durationSeconds: 180 }),
      {
        signal: controller.signal,
        captureTrace: true,
        onProgress: vi.fn(),
        onDiagnostic: vi.fn(),
        opt0080FullDitRun: {
          schedulingProfile: "opt-0080-depth2-epoch4",
          onCommandBufferCompleted(completion) {
            completions.push(completion.commandBufferIndex);
            expect(completion.schedulingProgress).toMatchObject({
              completedCommandBuffers: 1,
              totalCommandBuffers: 2_553,
              completionFenceRequestedCount: 2,
              completionFenceSettledCount: 1,
              outstandingCommandBuffers: 1,
            });
            controller.abort(sentinel);
          },
          onCheckpoint: checkpoint,
        },
      },
    )).rejects.toBe(sentinel);
    expect(completions).toEqual([0]);
    expect(checkpoint).not.toHaveBeenCalled();
    expect(harness.packageEvents.at(-1)).toBe("dit-destroy");
    expect(harness.phaseDestroyed.get("dit")).toBe(true);
    expect(harness.phaseDestroyed.get("dit-dense")).toBe(true);
    expect(harness.packageEvents).not.toContain("vae-acquire");
    expect(harness.audioBegins).toBe(0);
  });

  it("rejects malformed or combined OPT-0080 seams before model work", async () => {
    const harness = createOpt0080Harness();
    await initialize(harness, opt0080InitializeMessage());
    const before = [...harness.packageEvents];
    await expect(harness.backend.generate(
      testGenerationRequest({ durationSeconds: 180 }),
      {
        signal: new AbortController().signal,
        captureTrace: true,
        onProgress: vi.fn(),
        onDiagnostic: vi.fn(),
        opt0067DitRun: { onCheckpoint: vi.fn() },
        opt0080DitRun: {
          schedulingProfile: "depth1-epoch1",
          onCheckpoint: vi.fn(),
        },
      },
    )).rejects.toThrow(/mutually exclusive/u);
    expect(harness.packageEvents).toEqual(before);

    await expect(harness.backend.generate(
      testGenerationRequest({ durationSeconds: 180 }),
      {
        signal: new AbortController().signal,
        captureTrace: true,
        onProgress: vi.fn(),
        onDiagnostic: vi.fn(),
        opt0080DitRun: {
          schedulingProfile: "future-depth" as never,
          onCheckpoint: vi.fn(),
        },
      },
    )).rejects.toThrow(/one exact scheduling arm/u);
    expect(harness.packageEvents).toEqual(before);
    expect(harness.packageEvents).not.toContain("vae-acquire");
    expect(harness.audioBegins).toBe(0);
  });

  it("keeps the authenticated OPT-0056 runtime behind its private stop seam", async () => {
    const harness = createHarness({ ditDenseRevision: 8 });
    const base = testInitializeMessage();
    const configuration = {
      ...base.configuration,
      ditDensePackage: {
        manifestUrl: "https://example.test/rev8/manifest.json",
        manifestSha256: ACE_OPT_0037_DIT_K4_MANIFEST_SHA256,
        runtimeProfile: ACE_OPT_0056_DIT_SELECTIVE_K4_RUNTIME_PROFILE,
      },
    } as const;
    await harness.backend.initialize(configuration, initializationContext());
    await expect(harness.backend.generate(
      testGenerationRequest({ durationSeconds: 180 }),
      generationContext(),
    )).rejects.toThrow(/benchmark-only.*checkpoint seam/u);

    const controller = new AbortController();
    const sentinel = Object.freeze({ kind: "opt-0056-selective-stop" });
    let checkpoint: AceOpt0056DitCheckpoint | undefined;
    await expect(harness.backend.generate(
      testGenerationRequest({ durationSeconds: 180 }),
      {
        signal: controller.signal,
        captureTrace: true,
        onProgress: vi.fn(),
        onDiagnostic: vi.fn(),
        opt0056DitRun: {
          onCheckpoint(value) {
            checkpoint = value;
            expect(harness.packageEvents.at(-1)).toBe("dit-destroy");
            controller.abort(sentinel);
          },
        },
      },
    )).rejects.toBe(sentinel);
    expect(checkpoint?.denseRouteProfile).toMatchObject({
      schema: "ace-opt-0056-selective-dense-routes-v1",
      routeCount: 216,
      dispatchCount: 1_728,
      approximateRouteCount: 192,
      exactDownRouteCount: 24,
    });
    expect(checkpoint?.denseRouteProfile?.routes).toHaveLength(216);
    expect(harness.packageEvents).not.toContain("vae-acquire");
    expect(harness.loadedPhases).not.toContain("vae");
    expect(harness.audioBegins).toBe(0);

    await expect(harness.backend.generate(
      testGenerationRequest({ durationSeconds: 180 }),
      {
        signal: new AbortController().signal,
        captureTrace: true,
        onProgress: vi.fn(),
        onDiagnostic: vi.fn(),
        opt0056DitRun: { onCheckpoint: vi.fn() },
      },
    )).rejects.toThrow(/cannot continue beyond its checkpoint/u);
    expect(harness.packageEvents).not.toContain("vae-acquire");
    expect(harness.loadedPhases).not.toContain("vae");
    expect(harness.audioBegins).toBe(0);
  });

  it("stops at an exact OPT-0034 batch8 checkpoint without acquiring VAE", async () => {
    const harness = createOpt0080Harness();
    await initialize(harness, opt0080InitializeMessage());
    const controller = new AbortController();
    const sentinel = Object.freeze({ kind: "opt-0034-controlled-stop" });
    let checkpoint: AceOpt0034DitCheckpoint | undefined;
    const diagnostics: Array<Readonly<{
      readonly code: string;
      readonly details?: Readonly<Record<string, unknown>>;
    }>> = [];
    await expect(harness.backend.generate(
      testGenerationRequest({ durationSeconds: 180 }),
      {
        signal: controller.signal,
        captureTrace: true,
        onProgress: () => undefined,
        onDiagnostic: (diagnostic) => diagnostics.push(diagnostic),
        opt0034DitRun: {
          physicalQuantaPerCommandBuffer: 8,
          onCheckpoint: (value) => {
            checkpoint = value;
            expect(harness.packageEvents.at(-1)).toBe("dit-destroy");
            controller.abort(sentinel);
          },
        },
      },
    )).rejects.toBe(sentinel);
    expect(checkpoint).toMatchObject({
      schema: "ace-dit-opt0034-m2250-checkpoint-v1",
      finalLatentByteLength: 1_152_000,
      finalLatentElementCount: 288_000,
      finalLatentNonFiniteCount: 0,
      finalLatentNonzeroCount: 288_000,
      finalLatentMaxAbs: 0.25,
      profile: {
        schema: "ace-dit-opt0034-command-buffer-coalescing-v1",
        physicalGraphQuantumCount: 2_553,
        physicalQuantaPerCommandBuffer: 8,
        graphCommandBufferCount: 320,
        readbackCommandBufferCount: 1,
        totalCommandBufferCount: 321,
        graphQueueDrainCount: 320,
        totalQueueDrainCount: 321,
        graphRequestedIdleMs: 319,
      },
    });
    expect(checkpoint!.finalLatent).toHaveLength(288_000);
    expect(checkpoint!.finalLatentSha256).toMatch(/^[0-9a-f]{64}$/u);
    expect(checkpoint!.stageTimings.at(-1)?.stage).toBe("release-dit");
    expect(diagnostics).toContainEqual(expect.objectContaining({
      code: "ACE_DIT_OPT0034_SCHEDULING_PROFILE",
      details: expect.objectContaining({
        schema: "ace-dit-opt0034-scheduling-receipt-v1",
        checkpointSchema: "ace-dit-opt0034-m2250-checkpoint-v1",
        schedulingProfileSchema:
          "ace-dit-opt0034-command-buffer-coalescing-v1",
        physicalGraphQuantumCount: 2_553,
        physicalQuantaPerCommandBuffer: 8,
        graphCommandBufferCount: 320,
        totalCommandBufferCount: 321,
        totalQueueDrainCount: 321,
        descriptorTableSha256:
          "aedf8c74d2bb15601d1385e9c8e9da49e58905ab156a04c99271f8de3633dd76",
        finalLatentSha256: checkpoint!.finalLatentSha256,
      }),
    }));
    expect(harness.packageEvents).not.toContain("vae-acquire");
    expect(harness.packageEvents).not.toContain("vae-backend");
    expect(harness.loadedPhases).not.toContain("vae");
    expect(harness.audioBegins).toBe(0);
    expect(harness.phaseDestroyed.get("dit")).toBe(true);
    expect(harness.phaseDestroyed.get("dit-dense")).toBe(true);
    expect(harness.lastDitSubmissionPolicy).toBeUndefined();
  });

  it("does not build tensor trace receipts unless capture is explicitly enabled", async () => {
    const harness = createHarness();
    await initialize(harness);
    const diagnostics: Array<Readonly<{ readonly code: string }>> = [];
    await harness.backend.generate(testGenerationRequest(), {
      signal: new AbortController().signal,
      captureTrace: false,
      onProgress: vi.fn(),
      onDiagnostic: (diagnostic) => diagnostics.push(diagnostic),
    });
    expect(diagnostics.some(({ code }) => code.startsWith("ACE_"))).toBe(false);
  });

  it("loads planner and semantic phases only for a planner-enabled request", async () => {
    const harness = createHarness();
    await initialize(harness);
    const request: AceGenerationRequest = testGenerationRequest({
      instrumental: false,
      lyrics: "hello",
      planner: DEFAULT_ACE_PLANNER_CONFIGURATION,
    });
    const progress: AceGenerationProgress[] = [];
    const diagnostics: Array<Readonly<{
      readonly code: string;
      readonly details?: Readonly<Record<string, unknown>>;
    }>> = [];
    await harness.backend.generate(request, {
      signal: new AbortController().signal,
      captureTrace: true,
      onProgress: (event) => progress.push(event),
      onDiagnostic: (diagnostic) => diagnostics.push(diagnostic),
    });

    expect(uniqueStages(progress)).toEqual(generationStagePlan(true));
    expect(harness.loadedPhases).toEqual([
      "planner",
      "text",
      "semantic",
      "conditioner+constants",
      "dit",
      "dit-dense",
      "vae",
    ]);
    expect(harness.plannerResources).toEqual(["owned-phase"]);
    expect(diagnostics).toContainEqual(expect.objectContaining({
      code: "ACE_PLANNER_CONDITIONING_RESOLVED",
      details: expect.objectContaining({
        plannerEnabled: true,
        instrumental: false,
        lyricCharacters: 5,
        phase2MetadataThinkBlock: "<think>\nduration: 10\n</think>",
        plannerConfiguration: JSON.stringify(DEFAULT_ACE_PLANNER_CONFIGURATION),
        semanticCodeCount: 50,
        semanticCodeIds: Array.from({ length: 50 }, (_, index) => index),
        semanticCodeSha256:
          "f234d0f65ba480abeac60b2ef9635cb0598776c0223f709cda254f196e6f8486",
        textTokenSha256:
          "34fb5c825de7ca4aea6e712f19d439c1da0c92c37b423936c5f618545ca4fa1f",
        lyricTokenCount: 2,
        lyricTokenSha256:
          "34fb5c825de7ca4aea6e712f19d439c1da0c92c37b423936c5f618545ca4fa1f",
      }),
    }));
    expect(diagnostics).toContainEqual(expect.objectContaining({
      code: "ACE_CONDITIONING_TENSORS_READY",
      details: expect.objectContaining({
        mode: "planner",
        lyricConditionRows: 2,
        lyricConditionPrefixComplete: true,
        lyricConditionPrefixSha256:
          "4fe7b59af6de3b665b67788cc2f99892ab827efae3a467342b3bb4e3bc8e5bfe",
        conditionSha256:
          "02b1c2234680617802901a77eae606ad02e4ddb4282ccbc60061eac5b2d90bba",
        contextSha256:
          "eec19bc6af0b3b6dfb97a08782c65f4bb3c3203e789a015d2008b0d689ad08be",
        conditionNonFinite: 0,
        contextNonFinite: 0,
      }),
    }));
  });

  it("destroys a VAE phase from another manifest and rolls back its audio transaction", async () => {
    const harness = createHarness({ wrongManifestPhase: "vae" });
    await initialize(harness);

    await expect(
      harness.backend.generate(testGenerationRequest(), generationContext()),
    ).rejects.toThrow(/different manifest object/);
    expect(harness.phaseDestroyed.get("vae")).toBe(true);
    expect(harness.audioRollbacks).toBe(1);
    expect(harness.audioCommits).toBe(0);
  });

  it("releases a committed OPFS output when cancellation wins after commit", async () => {
    const controller = new AbortController();
    const harness = createHarness({ abortAfterCommit: controller });
    await initialize(harness);

    await expect(
      harness.backend.generate(testGenerationRequest(), {
        signal: controller.signal,
        onProgress: vi.fn(),
        onDiagnostic: vi.fn(),
      }),
    ).rejects.toMatchObject({ name: "AbortError" });
    expect(harness.releasedIds).toEqual([
      "ace-00000000-0000-4000-8000-000000000001",
    ]);
    await expect(
      harness.backend.releaseResult({
        audioStorageId: "ace-00000000-0000-4000-8000-000000000001",
      } as never),
    ).rejects.toMatchObject({ name: "NotFoundError" });
  });

  it("releases only results committed by this backend", async () => {
    const harness = createHarness();
    await initialize(harness);
    const result = await harness.backend.generate(
      testGenerationRequest(),
      generationContext(),
    );

    await expect(
      harness.backend.releaseResult({
        ...result,
        audioStorageId: "ace-foreign",
      }),
    ).rejects.toMatchObject({ name: "NotFoundError" });
    expect(harness.releasedIds).toEqual([]);
    await harness.backend.releaseResult(result);
    expect(harness.releasedIds).toEqual([result.audioStorageId]);
    await expect(harness.backend.releaseResult(result)).rejects.toMatchObject({
      name: "NotFoundError",
    });
  });

  it("fails closed on the unaudited benchmark scheduling profile", async () => {
    const harness = createHarness();
    const initializeMessage = testInitializeMessage();
    await expect(
      harness.backend.initialize(
        {
          ...initializeMessage.configuration,
          schedulingProfile: "benchmark",
        },
        initializationContext(),
      ),
    ).rejects.toMatchObject({ name: "NotSupportedError" });
    expect(harness.requestDevice).not.toHaveBeenCalled();
  });

  it("reserves initialize and generate before their first asynchronous cleanup", async () => {
    const initializeGate = deferred<void>();
    const initializing = createHarness({ recoverAudioGate: initializeGate });
    const firstInitialize = initialize(initializing);
    await Promise.resolve();
    await expect(initialize(initializing)).rejects.toMatchObject({
      name: "InvalidStateError",
    });
    initializeGate.resolve();
    await firstInitialize;

    const generationGate = deferred<void>();
    const generating = createHarness({ beginAudioGate: generationGate });
    await initialize(generating);
    const firstGenerate = generating.backend.generate(
      testGenerationRequest(),
      generationContext(),
    );
    await waitFor(() => generating.audioBegins === 1);
    await expect(generating.backend.generate(
      testGenerationRequest(),
      generationContext(),
    )).rejects.toMatchObject({ name: "InvalidStateError" });
    await expect(generating.backend.dispose()).rejects.toMatchObject({
      name: "InvalidStateError",
    });
    generationGate.resolve();
    await firstGenerate;
  });

  it("retries transient rollback and unpublished release failures without losing ownership", async () => {
    const rollback = createHarness({
      wrongManifestPhase: "vae",
      failRollbackAttempts: 1,
    });
    await initialize(rollback);
    await expect(rollback.backend.generate(
      testGenerationRequest(),
      generationContext(),
    )).rejects.toThrow(/different manifest object/);
    expect(rollback.audioRollbacks).toBe(1);
    await rollback.backend.dispose();
    expect(rollback.audioRollbacks).toBe(2);
    expect(rollback.deviceDestroy).toHaveBeenCalledOnce();

    const release = createHarness({ failReleaseAttempts: 1 });
    await initialize(release);
    const result = await release.backend.generate(
      testGenerationRequest(),
      generationContext(),
    );
    await expect(release.backend.releaseResult(result)).rejects.toThrow(
      /injected release/,
    );
    await release.backend.dispose();
    expect(release.releaseAttempts).toBe(2);
    expect(release.releasedIds).toEqual([result.audioStorageId]);
    expect(release.deviceDestroy).toHaveBeenCalledOnce();
  });

  it("fails closed on an uncaptured GPU error before VAE/audio publication", async () => {
    const harness = createHarness({ uncapturedErrorAfterDit: true });
    await initialize(harness);
    await expect(harness.backend.generate(
      testGenerationRequest(),
      generationContext(),
    )).rejects.toThrow(/GPUValidationError.*injected invalid dispatch/);
    expect(harness.audioBegins).toBe(0);
    expect(harness.audioCommits).toBe(0);
  });

  it("latches a GPU error even when its diagnostic callback throws", async () => {
    const harness = createHarness({ uncapturedErrorAfterDit: true });
    await initialize(harness);
    await expect(harness.backend.generate(testGenerationRequest(), {
      signal: new AbortController().signal,
      onProgress: vi.fn(),
      onDiagnostic: () => {
        throw new Error("diagnostic transport failed");
      },
    })).rejects.toThrow(/GPUValidationError.*injected invalid dispatch/);
    expect(harness.audioBegins).toBe(0);
  });

  it.each([
    {
      code: "WEBGPU_DEVICE_LOST" as const,
      event: {
        type: "device-lost" as const,
        reason: "unknown",
        message: "injected idle loss",
      },
    },
    {
      code: "WEBGPU_UNCAPTURED_ERROR" as const,
      event: {
        type: "uncaptured-error" as const,
        errorType: "GPUValidationError",
        message: "injected idle validation failure",
      },
    },
  ])("preserves stable $code after an idle GPU failure", async ({ code, event }) => {
    const harness = createHarness();
    await initialize(harness);
    harness.emitRuntimeEvent(event);

    await expect(harness.backend.generate(
      testGenerationRequest(),
      generationContext(),
    )).rejects.toMatchObject({ code });
    expect(harness.loadedPhases).toEqual([]);
    expect(harness.audioBegins).toBe(0);
  });

  it("rolls back when the progress callback throws", async () => {
    const harness = createHarness();
    await initialize(harness);
    await expect(harness.backend.generate(testGenerationRequest(), {
      signal: new AbortController().signal,
      onProgress: (event) => {
        if (event.stage === "vae-decode") throw new Error("progress failed");
      },
      onDiagnostic: vi.fn(),
    })).rejects.toThrow(/progress failed/);
    expect(harness.audioRollbacks).toBe(1);
    expect(harness.phaseDestroyed.get("vae")).toBe(true);
  });

  it("runs sealed 8/6/5 sampler diagnostics and retains complete evidence", async () => {
    const harness = createHarness();
    await initialize(harness);
    const evidence: AceSamplerScheduleDiagnosticEvidence[] = [];
    for (const scheduleProfileId of [
      ACE_TURBO_EIGHT_SAMPLER_SCHEDULE_PROFILE_ID,
      ACE_OPT_0055_SIX_SAMPLER_SCHEDULE_PROFILE_ID,
      ACE_OPT_0065_FIVE_SAMPLER_SCHEDULE_PROFILE_ID,
    ] as const) {
      await harness.backend.generate(testGenerationRequest(), {
        signal: new AbortController().signal,
        onProgress: vi.fn(),
        onDiagnostic: vi.fn(),
        opt0055Opt0065SamplerRun: {
          scheduleProfileId,
          onEvidence: (value) => evidence.push(value),
        },
      });
    }
    expect(evidence.map(({ schedule }) => schedule.evaluationCount)).toEqual([
      8,
      6,
      5,
    ]);
    expect(evidence.map(({ evaluations }) => evaluations.length)).toEqual([
      8,
      6,
      5,
    ]);
    expect(new Set(evidence.map(({ evaluation0Velocity }) =>
      evaluation0Velocity.sha256
    )).size).toBe(1);
    for (const arm of evidence) {
      expect(arm.evaluations.at(-1)?.sha256).toBe(arm.finalLatent.sha256);
      expect(arm.rawAudio.snapshot.size).toBe(arm.rawAudio.byteLength);
      expect(arm.rawAudio.finiteSamples).toBe(arm.rawAudio.interleavedSamples);
      expect(arm.wav.type).toBe("audio/wav");
    }
  });

  it("rejects schedule diagnostics combined with legacy capture work", async () => {
    const harness = createHarness();
    await initialize(harness);
    await expect(harness.backend.generate(testGenerationRequest(), {
      signal: new AbortController().signal,
      captureTrace: true,
      onProgress: vi.fn(),
      onDiagnostic: vi.fn(),
      opt0055Opt0065SamplerRun: {
        scheduleProfileId: ACE_OPT_0065_FIVE_SAMPLER_SCHEDULE_PROFILE_ID,
        onEvidence: vi.fn(),
      },
    })).rejects.toThrow(/no legacy diagnostic seam/);
  });

  it("destroys an unexpected OPT-0081 setup owner when the failpoint is not thrown", async () => {
    const harness = createHarness({
      largeVaeLimits: true,
      mainManifestSha256: ACE_REFERENCE_MANIFEST_SHA256,
      opt0081UnexpectedSetupOwner: true,
    });
    await initialize(harness, opt0080InitializeMessage());
    await expect(harness.backend.generate(
      testGenerationRequest({ durationSeconds: 180 }),
      {
        signal: new AbortController().signal,
        captureTrace: true,
        onProgress: vi.fn(),
        onDiagnostic: vi.fn(),
        opt0081RepresentativeRun: {
          mode: "setup-failure",
          onEvidence: vi.fn(),
        },
      },
    )).rejects.toThrow(/setup failure preflight did not execute/);
    expect(harness.opt0081OwnerDestroy).toHaveBeenCalledTimes(1);
    expect(harness.phaseDestroyed.get("dit")).toBe(true);
    expect(harness.phaseDestroyed.get("dit-dense")).toBe(true);
  });

  it("binds identical immutable conditioning authority to both OPT-0081 invocations", async () => {
    const setupHarness = createHarness({
      largeVaeLimits: true,
      mainManifestSha256: ACE_REFERENCE_MANIFEST_SHA256,
      opt0081WorkingOwner: true,
    });
    await initialize(setupHarness, opt0080InitializeMessage());
    const setupController = new AbortController();
    const setupStop = Object.freeze({ kind: "opt0081-setup-complete" });
    let setupEvidence: AceOpt0081RepresentativeSetupCleanupEvidence |
      undefined;
    let setupAuthority: AceOpt0081RepresentativeConditioningAuthority |
      undefined;
    await expect(setupHarness.backend.generate(
      testGenerationRequest({ durationSeconds: 180 }),
      {
        signal: setupController.signal,
        captureTrace: true,
        onProgress: vi.fn(),
        onDiagnostic: vi.fn(),
        opt0081RepresentativeRun: {
          mode: "setup-failure",
          onEvidence(evidence, authority) {
            setupEvidence = evidence;
            setupAuthority = authority;
            setupController.abort(setupStop);
          },
        },
      },
    )).rejects.toBe(setupStop);
    if (setupEvidence === undefined || setupAuthority === undefined) {
      throw new Error("test OPT-0081 setup authority was not captured");
    }

    const runHarness = createHarness({
      largeVaeLimits: true,
      mainManifestSha256: ACE_REFERENCE_MANIFEST_SHA256,
      opt0081WorkingOwner: true,
    });
    await initialize(runHarness, opt0080InitializeMessage());
    const runController = new AbortController();
    const runStop = Object.freeze({ kind: "opt0081-run-complete" });
    let runAuthority: AceOpt0081RepresentativeConditioningAuthority |
      undefined;
    await expect(runHarness.backend.generate(
      testGenerationRequest({ durationSeconds: 180 }),
      {
        signal: runController.signal,
        captureTrace: true,
        onProgress: vi.fn(),
        onDiagnostic: vi.fn(),
        opt0081RepresentativeRun: {
          mode: "run",
          verifiedSetupFailureCleanup: setupEvidence,
          async run(_owner, verifiedEvidence, authority) {
            expect(verifiedEvidence).toBe(setupEvidence);
            runAuthority = authority;
            runController.abort(runStop);
          },
        },
      },
    )).rejects.toBe(runStop);

    expect(Object.isFrozen(setupAuthority)).toBe(true);
    expect(Object.isFrozen(runAuthority)).toBe(true);
    expect(runAuthority).toEqual(setupAuthority);
    expect(runAuthority).toEqual({
      schema: "ace-opt-0081-representative-conditioning-authority-v1",
      textTokenCount: 2,
      textTokenSha256:
        "34fb5c825de7ca4aea6e712f19d439c1da0c92c37b423936c5f618545ca4fa1f",
      lyricTokenCount: 2,
      lyricTokenSha256:
        "34fb5c825de7ca4aea6e712f19d439c1da0c92c37b423936c5f618545ca4fa1f",
      conditionTokens: 5,
      conditionElementCount: 10_240,
      conditionSha256:
        "02b1c2234680617802901a77eae606ad02e4ddb4282ccbc60061eac5b2d90bba",
      contextElementCount: 576_000,
      contextSha256:
        "0d2039545940e7775d22c8d82079282c91be29af0203faa77e2bff78a67bed0f",
    });
    expect(setupHarness.phaseDestroyed.get("dit")).toBe(true);
    expect(setupHarness.phaseDestroyed.get("dit-dense")).toBe(true);
    expect(runHarness.phaseDestroyed.get("dit")).toBe(true);
    expect(runHarness.phaseDestroyed.get("dit-dense")).toBe(true);
    expect(runHarness.opt0081OwnerDestroy).toHaveBeenCalledOnce();
  });
});

interface HarnessOptions {
  readonly wrongManifestPhase?: AceTensorPhase;
  readonly abortAfterCommit?: AbortController;
  readonly recoverAudioGate?: Deferred<void>;
  readonly beginAudioGate?: Deferred<void>;
  readonly failRollbackAttempts?: number;
  readonly failReleaseAttempts?: number;
  readonly uncapturedErrorAfterDit?: boolean;
  readonly ditDenseRevision?: 7 | 8;
  readonly largeVaeLimits?: boolean;
  /** Exact adapter/device maxBufferSize + maxStorageBufferBindingSize. */
  readonly vaeLimitBytes?: number;
  readonly mainManifestSha256?: string;
  readonly opt0081UnexpectedSetupOwner?: boolean;
  readonly opt0081WorkingOwner?: boolean;
  /** OPT-0088: no-subgroups device landing the portable execution profile. */
  readonly portable?: boolean;
  /** Incoherent portable device that still enabled the subgroups feature. */
  readonly portableDeviceSubgroupsLeak?: boolean;
  /** Device without shader-f16; both production tuples must fail closed. */
  readonly missingShaderF16?: boolean;
}

interface Harness {
  readonly backend: ReturnType<typeof createAceWebGpuPipelineBackendForTest>;
  readonly loadedPhases: string[];
  readonly plannerResources: string[];
  readonly packageEvents: string[];
  readonly phaseDestroyed: Map<string, boolean>;
  readonly releasedIds: string[];
  readonly requestDevice: ReturnType<typeof vi.fn>;
  readonly deviceDestroy: ReturnType<typeof vi.fn>;
  readonly opt0081OwnerDestroy: ReturnType<typeof vi.fn>;
  emitRuntimeEvent(event: AceGpuRuntimeEvent): void;
  readonly audioRollbacks: number;
  readonly audioCommits: number;
  readonly audioBegins: number;
  readonly releaseAttempts: number;
  readonly lastDitSubmissionPolicy: string | undefined;
  readonly lastDitAttentionRuntimeProfile: string | undefined;
  readonly lastDitCaptureOpt0062AttentionIdentity: boolean;
  readonly lastVaeMaximumWindowFrames: number | undefined;
  readonly lastVaePlanChunkFrames: number | undefined;
  readonly lastVaeRuntimeProfile: string | undefined;
  readonly lastVaeProductionSchedulingPolicy: string | undefined;
}

function createHarness(options: HarnessOptions = {}): Harness {
  let clock = 0;
  let audioRollbacks = 0;
  let audioCommits = 0;
  let audioBegins = 0;
  let releaseAttempts = 0;
  let runtimeEvent: ((event: AceGpuRuntimeEvent) => void) | undefined;
  let lastDitSubmissionPolicy: string | undefined;
  let lastDitAttentionRuntimeProfile: string | undefined;
  let lastDitCaptureOpt0062AttentionIdentity = false;
  let lastVaeMaximumWindowFrames: number | undefined;
  let lastVaePlanChunkFrames: number | undefined;
  let lastVaeRuntimeProfile: string | undefined;
  let lastVaeProductionSchedulingPolicy: string | undefined;
  const loadedPhases: string[] = [];
  const plannerResources: string[] = [];
  const packageEvents: string[] = [];
  const phaseDestroyed = new Map<string, boolean>();
  const releasedIds: string[] = [];
  const layerFiles = ACE_OPT_0037_DIT_K4_WEIGHT_FILES.map((name, index) => ({
    name,
    byteLength: index % 2 === 0 ? 121_668_608 : 4_198_400,
    sha256: String(index).padStart(64, "0"),
    kind: "weights" as const,
  }));
  const sharedFiles = [
    {
      name: "weights/dit/shared-00.bin",
      byteLength: 10_506_496,
      sha256: "b".repeat(64),
      kind: "weights" as const,
    },
    {
      name: "weights/dit/time-embedding-00.bin",
      byteLength: 119_603_200,
      sha256: "c".repeat(64),
      kind: "weights" as const,
    },
  ];
  const referenceTensors = Object.fromEntries([
    ...layerFiles.map((file, index) => [
      `ace.decoder.layers.${Math.floor(index / 2)}.test-${index}`,
      { phase: "dit", shard: file.name },
    ]),
    ...Array.from({ length: 20 }, (_, index) => [
      `ace.decoder.shared-test-${index}`,
      { phase: "dit", shard: sharedFiles[index % 2]!.name },
    ]),
    ["vae.test", { phase: "vae", shard: "weights/vae/test.bin" }],
  ]);
  const manifest = {
    profile: "reference",
    files: [
      ...layerFiles,
      ...sharedFiles,
      {
        name: "weights/vae/test.bin",
        byteLength: 4,
        sha256: "d".repeat(64),
        kind: "weights",
      },
    ],
    tensors: referenceTensors,
  } as unknown as AcePackageManifest;
  const loaded: AceLoadedPackageManifest = {
    manifest,
    manifestUrl: "https://example.test/manifest.json",
    manifestSha256: options.mainManifestSha256 ?? "a".repeat(64),
    manifestByteLength: 100,
    manifestId: `ace-step-webgpu-v1:reference:${
      options.mainManifestSha256 ?? "a".repeat(64)
    }`,
  };
  const vaeManifest = {
    profile: "fp16-vae-experimental",
    provenance: { converterRevision: 6 },
  } as AcePackageManifest;
  const vaeLoaded: AceLoadedPackageManifest = {
    manifest: vaeManifest,
    manifestUrl:
      "https://example.test/model/files-fp16-vae-experimental/manifest.json",
    manifestSha256: ACE_OPT_0028_VAE_FP16_MANIFEST_SHA256,
    manifestByteLength: ACE_OPT_0028_VAE_FP16_MANIFEST_BYTES,
    manifestId:
      `ace-step-webgpu-v1:fp16-vae-experimental:${ACE_OPT_0028_VAE_FP16_MANIFEST_SHA256}`,
  };
  const revision7VaeLoaded: AceLoadedPackageManifest = {
    manifest: {
      profile: "fp16-vae-experimental",
      provenance: { converterRevision: 7 },
    } as AcePackageManifest,
    manifestUrl:
      "https://example.test/model/files-fp16-vae-revision7-experimental/manifest.json",
    manifestSha256: ACE_OPT_0054_VAE_REVISION7_MANIFEST_SHA256,
    manifestByteLength: ACE_OPT_0054_VAE_REVISION7_MANIFEST_BYTES,
    manifestId:
      `ace-step-webgpu-v1:fp16-vae-experimental:${ACE_OPT_0054_VAE_REVISION7_MANIFEST_SHA256}`,
  };
  const ditDenseManifest = {
    profile: "fp16-dit-dense-experimental",
    provenance: { converterRevision: options.ditDenseRevision ?? 7 },
    files: layerFiles,
    tensors: {},
  } as unknown as AcePackageManifest;
  const ditDenseLoaded: AceLoadedPackageManifest = {
    manifest: ditDenseManifest,
    manifestUrl: options.ditDenseRevision === 8
      ? "https://example.test/rev8/manifest.json"
      : "https://example.test/model/files-fp16-dit-rev7-oracle/manifest.json",
    manifestSha256: options.ditDenseRevision === 8
      ? ACE_OPT_0037_DIT_K4_MANIFEST_SHA256
      : ACE_OPT_0009_DIT_DENSE_MANIFEST_SHA256,
    manifestByteLength: options.ditDenseRevision === 8
      ? ACE_OPT_0037_DIT_K4_MANIFEST_BYTES
      : ACE_OPT_0009_DIT_DENSE_MANIFEST_BYTES,
    manifestId:
      `ace-step-webgpu-v1:fp16-dit-dense-experimental:${
        options.ditDenseRevision === 8
          ? ACE_OPT_0037_DIT_K4_MANIFEST_SHA256
          : ACE_OPT_0009_DIT_DENSE_MANIFEST_SHA256
      }`,
  };
  const files = new Map<string, File>();
  for (const directory of ["qwen", "planner"] as const) {
    for (const name of [
      "tokenizer.json",
      "tokenizer_config.json",
      "chat_template.jinja",
    ]) {
      files.set(
        `assets/${directory}/${name}`,
        new File([directory], name),
      );
    }
  }
  const acquired = {
    files,
    plan: {
      files: [],
      cachedFiles: [],
      downloadFiles: [],
      runtimeBytes: ACE_REFERENCE_DIT_SHARED_WEIGHT_BYTES,
      cachedBytes: ACE_REFERENCE_DIT_SHARED_WEIGHT_BYTES,
      downloadBytes: 0,
      requiredFreeBytes: 0,
    },
  } as unknown as AceAcquiredModelFiles;
  const ditDenseFiles = new Map<string, File>(
    ACE_OPT_0037_DIT_K4_WEIGHT_FILES.map((name) => [
      name,
      new File([name], name),
    ]),
  );
  const ditDenseAcquired = {
    files: ditDenseFiles,
    plan: {
      files: ACE_OPT_0037_DIT_K4_WEIGHT_FILES.map((name) => ({ name })),
      cachedFiles: [...ACE_OPT_0037_DIT_K4_WEIGHT_FILES],
      downloadFiles: [],
      runtimeBytes: ACE_OPT_0037_DIT_K4_LAYER_BYTES,
      cachedBytes: ACE_OPT_0037_DIT_K4_LAYER_BYTES,
      downloadBytes: 0,
      requiredFreeBytes: 0,
    },
  } as unknown as AceAcquiredModelFiles;
  const vaeFiles = new Map<string, File>(
    ACE_OPT_0011_VAE_FP16_WEIGHT_FILES.map((name) => [
      name,
      new File([name], name),
    ]),
  );
  const vaeAcquired = {
    files: vaeFiles,
    plan: {
      files: ACE_OPT_0011_VAE_FP16_WEIGHT_FILES.map((name) => ({ name })),
      cachedFiles: [],
      downloadFiles: [...ACE_OPT_0011_VAE_FP16_WEIGHT_FILES],
      runtimeBytes: 168_791_552,
      cachedBytes: 0,
      downloadBytes: 168_791_552,
      requiredFreeBytes: 0,
    },
  } as unknown as AceAcquiredModelFiles;
  const baseCapabilities = testDiagnostics().capabilities;
  const portable = options.portable === true;
  const shaderFeatures: readonly ("shader-f16" | "subgroups")[] =
    options.missingShaderF16 === true ? [] : ["shader-f16"];
  const harnessRequiredFeatures: readonly ("shader-f16" | "subgroups")[] =
    portable ? [...shaderFeatures] : [...shaderFeatures, "subgroups"];
  const capabilities = Object.freeze({
    ...baseCapabilities,
    executionProfile: portable
      ? ACE_REFERENCE_PORTABLE_PROFILE
      : ACE_REFERENCE_SUBGROUP_PROFILE,
    adapterInfo: Object.freeze({
      ...baseCapabilities.adapterInfo,
      // Portable devices may advertise non-fixed-32 subgroups (Chrome-class
      // adapters); the fixed32 tuple reports the exact 32/32 pair.
      subgroupMinSize: portable ? 4 : 32,
      subgroupMaxSize: portable ? 64 : 32,
    }),
    adapterFeatures: Object.freeze(["shader-f16", "subgroups"]),
    deviceFeatures: Object.freeze(
      portable && options.portableDeviceSubgroupsLeak !== true
        ? [...shaderFeatures]
        : [...shaderFeatures, "subgroups"],
    ),
    requiredFeatures: Object.freeze(harnessRequiredFeatures),
    ...(options.largeVaeLimits === true || options.vaeLimitBytes !== undefined
      ? {
          adapterLimits: Object.freeze({
            ...baseCapabilities.adapterLimits,
            maxBufferSize: options.vaeLimitBytes ?? 2_000_000_000,
            maxStorageBufferBindingSize:
              options.vaeLimitBytes ?? 2_000_000_000,
          }),
          deviceLimits: Object.freeze({
            ...baseCapabilities.deviceLimits,
            maxBufferSize: options.vaeLimitBytes ?? 2_000_000_000,
            maxStorageBufferBindingSize:
              options.vaeLimitBytes ?? 2_000_000_000,
          }),
        }
      : {}),
    stockFeatures: Object.freeze({
      ...baseCapabilities.stockFeatures,
      ...(options.missingShaderF16 === true
        ? {
            "shader-f16": Object.freeze({
              adapterSupported: false,
              deviceEnabled: false,
              required: false,
              requested: false,
            }),
          }
        : {}),
      subgroups: Object.freeze({
        adapterSupported: true,
        deviceEnabled: !portable ||
          options.portableDeviceSubgroupsLeak === true,
        required: !portable,
        requested: !portable,
      }),
    }),
  });
  const deviceDestroy = vi.fn();
  const opt0081OwnerDestroy = vi.fn();
  const requestDevice = vi.fn(async (
    request: Parameters<AceWebGpuPipelineDependencies["requestDevice"]>[0],
  ) => {
    runtimeEvent = request.onRuntimeEvent;
    return {
      device: {} as GPUDevice,
      capabilities,
      lost: new Promise<never>(() => undefined),
      destroy: deviceDestroy,
    };
  });

  const dependencies: AceWebGpuPipelineDependencies = {
    now: () => ++clock,
    randomUUID: () => "00000000-0000-4000-8000-000000000001",
    ensureStorage: async (_storage, signal) => signal.throwIfAborted(),
    recoverAudio: async (_storage, signal) => {
      signal.throwIfAborted();
      await options.recoverAudioGate?.promise;
      signal.throwIfAborted();
    },
    loadManifest: async (_configuration, signal) => {
      signal.throwIfAborted();
      packageEvents.push("main-manifest");
      return loaded;
    },
    loadDitDenseManifest: async (_configuration, signal) => {
      signal.throwIfAborted();
      packageEvents.push("dit-dense-manifest");
      return ditDenseLoaded;
    },
    loadVaeManifest: async (configuration, signal) => {
      signal.throwIfAborted();
      packageEvents.push("vae-manifest");
      return configuration.manifestSha256 ===
          ACE_OPT_0054_VAE_REVISION7_MANIFEST_SHA256
        ? revision7VaeLoaded
        : vaeLoaded;
    },
    acquireModel: async ({ kind, signal, onProgress }) => {
      signal.throwIfAborted();
      packageEvents.push(`${kind}-acquire`);
      if (kind === "dit-dense") {
        onProgress({
          file: ACE_OPT_0037_DIT_K4_WEIGHT_FILES[0]!,
          fileIndex: 0,
          fileCount: 48,
          fileReceivedBytes: ACE_OPT_0037_DIT_K4_LAYER_BYTES,
          fileBytes: 121_668_608,
          completedBytes: ACE_OPT_0037_DIT_K4_LAYER_BYTES,
          totalBytes: ACE_OPT_0037_DIT_K4_LAYER_BYTES,
          source: "cache",
        });
        return ditDenseAcquired;
      }
      if (kind === "vae") {
        onProgress({
          file: ACE_OPT_0011_VAE_FP16_WEIGHT_FILES[0],
          fileIndex: 0,
          fileCount: 7,
          fileReceivedBytes: 168_791_552,
          fileBytes: 168_791_552,
          completedBytes: 168_791_552,
          totalBytes: 168_791_552,
          source: "cache",
        });
        return vaeAcquired;
      }
      onProgress({
        file: "all",
        fileIndex: 0,
        fileCount: 1,
        fileReceivedBytes: ACE_REFERENCE_DIT_SHARED_WEIGHT_BYTES,
        fileBytes: ACE_REFERENCE_DIT_SHARED_WEIGHT_BYTES,
        completedBytes: ACE_REFERENCE_DIT_SHARED_WEIGHT_BYTES,
        totalBytes: ACE_REFERENCE_DIT_SHARED_WEIGHT_BYTES,
        source: "cache",
      });
      return acquired;
    },
    requestDevice,
    loadTokenizer: async (kind) => loadedTokenizer(kind),
    loadPhase: async (
      _device,
      expectedManifest,
      _files,
      phases,
      signal,
      onProgress,
    ) => {
      signal.throwIfAborted();
      const key = expectedManifest.profile === "fp16-dit-dense-experimental"
        ? "dit-dense"
        : phases.join("+");
      loadedPhases.push(key);
      packageEvents.push(`phase-${key}`);
      onProgress?.({
        phaseFileIndex: 0,
        phaseFileCount: 1,
        loadedPhaseBytes: 1_024,
        totalPhaseBytes: 1_024,
        upload: {
          file: `${key}-test`,
          uploadedBytes: 1_024,
          totalBytes: 1_024,
        },
      });
      phaseDestroyed.set(key, false);
      return fakePhase(
        phases,
        options.wrongManifestPhase !== undefined &&
            phases.includes(options.wrongManifestPhase)
          ? ({ profile: "reference" } as AcePackageManifest)
          : expectedManifest,
        () => phaseDestroyed.set(key, true),
      );
    },
    runPlanner: async (plannerOptions) => {
      plannerResources.push(plannerOptions.resources?.kind ?? "none");
      if (plannerOptions.request.planner.mode === "disabled") {
        return {
          plannerMode: "disabled",
          bypassReason: "planner-disabled-by-request",
          downstream: {
            caption: plannerOptions.request.prompt,
            lyrics: plannerOptions.request.lyrics ?? "[Instrumental]",
            instrumental: plannerOptions.request.instrumental,
            durationSeconds: plannerOptions.request.durationSeconds,
            vocalLanguage: "unknown",
            metadata: { duration: plannerOptions.request.durationSeconds },
          },
          conditioning: null,
          cot: null,
          semantic: null,
          sampling: null,
          runtime: {
            peakAccountedGpuBytes: 0,
            queueDrains: 0,
            cooperativeIdleMs: 0,
          },
        };
      }
      plannerOptions.resources?.kind === "owned-phase" &&
        plannerOptions.resources.ownedPlannerWeights.destroy();
      return {
        plannerMode: "enabled",
        downstream: {
          caption: "planned caption",
          lyrics: plannerOptions.request.lyrics ?? "",
          instrumental: plannerOptions.request.instrumental,
          durationSeconds: plannerOptions.request.durationSeconds,
          vocalLanguage: "en",
          metadata: { duration: plannerOptions.request.durationSeconds },
        },
        semantic: {
          semanticCodeValues: Array.from({ length: 50 }, (_, index) => index),
        },
        conditioning: {
          cotText: "<think>\nduration: 10\n</think>",
        },
        cot: null,
        runtime: {
          peakAccountedGpuBytes: 90,
          queueDrains: 5,
          cooperativeIdleMs: 4,
        },
      } as never;
    },
    createConditioning: (conditioningOptions) => {
      let destroyed = false;
      return {
        async run(request) {
          emitConditioning(conditioningOptions.onProgress, "text");
          conditioningOptions.ownedTextWeights.destroy();
          if (request.mode.kind === "planner") {
            const semantic = await conditioningOptions.loadSemanticWeights!(
              conditioningOptions.signal ?? new AbortController().signal,
            );
            emitConditioning(conditioningOptions.onProgress, "semantic");
            semantic.destroy();
          }
          const conditioner = await conditioningOptions.loadConditionerWeights(
            conditioningOptions.signal ?? new AbortController().signal,
          );
          emitConditioning(conditioningOptions.onProgress, "conditioner");
          conditioner.destroy();
          return {
            mode: request.mode.kind,
            batch: 1,
            conditionTokens:
              request.lyricTokenIds.length + 1 + request.textTokenIds.length,
            latentFrames: request.latentFrames,
            hiddenSize: 2_048,
            contextChannels: 128,
            conditionHiddenStates: new Float32Array(
              (request.lyricTokenIds.length + 1 + request.textTokenIds.length) *
                2_048,
            ),
            conditionMask: new Uint32Array(
              request.lyricTokenIds.length + 1 + request.textTokenIds.length,
            ).fill(1),
            contextLatents: new Float32Array(request.latentFrames * 128),
            memory: {
              textWeightBytes: 1,
              textRetainedBytes: 1,
              textWorkingBytes: 1,
              semanticWeightBytes: request.mode.kind === "planner" ? 1 : 0,
              semanticRetainedBytes: 1,
              semanticWorkingBytes: 1,
              conditionerWeightBytes: 1,
              conditionerWorkingBytes: 1,
              resultBytes: 1,
              resultReadbackBytes: 1,
              returnedCpuBytes: 1,
              peakAccountedGpuBytes: 100,
            },
          };
        },
        async destroy() {
          destroyed = true;
          void destroyed;
        },
      };
    },
    createDit: async (ditOptions) => {
      lastDitSubmissionPolicy = ditOptions.ditSubmissionPolicy;
      lastDitAttentionRuntimeProfile =
        ditOptions.ditAttentionRuntimeProfile;
      lastDitCaptureOpt0062AttentionIdentity =
        ditOptions.captureOpt0062AttentionIdentity === true;
      ditOptions.onProgress?.({
        stage: "compile",
        compiledQuanta: 1,
        totalQuanta: 1,
      });
      let destroyed = false;
      return {
        memory: { accountedGpuBytes: 200 },
        async run() {
          const completedEvaluations = ditOptions.opt0067EvaluationLimit === 1
            ? 1 as const
            : ditOptions.samplerScheduleProfile?.evaluationCount ?? 8;
          ditOptions.onProgress?.({
            stage: "denoise",
            completedCommandBuffers: 1,
            totalCommandBuffers: 2,
            queueDrains: 1,
            cooperativeIdleMs: 1,
            completedEvaluations,
            graph: {} as never,
          } satisfies AceDitGpuBackendProgress);
          ditOptions.ownedDitWeights.destroy();
          ditOptions.ownedDitDenseWeights?.destroy();
          if (options.uncapturedErrorAfterDit === true) {
            runtimeEvent?.({
              type: "uncaptured-error",
              errorType: "GPUValidationError",
              message: "injected invalid dispatch",
            });
          }
          if (ditOptions.onCommandProfile !== undefined) {
            ditOptions.onCommandProfile(fakeOpt0018DitCommandProfile());
            const finalLatent = new Float32Array(288_000).fill(0.25);
            return {
              finalLatent,
              shape: {
                batch: 1,
                latentFrames: 4_500,
                channels: 64,
              },
              commandBuffersSubmitted: 2_554,
              queueDrains: 2_554,
              cooperativeIdleMs: 2_553,
              completedEvaluations: 8,
              ...(ditOptions.captureEvaluationLatents === true
                ? {
                    evaluationLatents: Object.freeze(Array.from(
                      { length: 8 },
                      (_, evaluation) => new Float32Array(288_000).fill(
                        evaluation === 7 ? 0.25 : (evaluation + 1) / 32,
                      ),
                    )),
                  }
                : {}),
              ...(ditOptions.ditDenseRuntimeProfile ===
                  ACE_OPT_0056_DIT_SELECTIVE_K4_RUNTIME_PROFILE
                ? { denseRouteProfile: fakeOpt0056DenseRouteProfile() }
                : {}),
            };
          }
          if (ditOptions.onOpt0080EvaluationProfile !== undefined) {
            const schedulingProfile = ditOptions.opt0080SchedulingProfile;
            if (schedulingProfile === undefined) {
              throw new Error("test OPT-0080 scheduling profile missing");
            }
            const profile = fakeOpt0080DitCommandProfile(schedulingProfile);
            if (ditOptions.onOpt0080CommandBufferCompleted !== undefined) {
              const descriptor = profile.descriptorTable.descriptors[0]!;
              ditOptions.onOpt0080CommandBufferCompleted(Object.freeze({
                commandBufferIndex: 0,
                submitThroughCompletionFenceMs: 1,
                trueQueueDrain: false,
                completionEpochIndex: 0,
                descriptor,
                schedulingProgress: Object.freeze({
                  completedCommandBuffers: 1,
                  totalCommandBuffers: 341,
                  completionFenceRequestedCount: 2,
                  completionFenceSettledCount: 1,
                  completionFenceRejectedCount: 0,
                  trueQueueDrainCount: 0,
                  completionEpochCount: 0,
                  requestedCooperativeIdleMs: 0,
                  cooperativeIdleTurns: 0,
                  outstandingCommandBuffers: 1,
                }),
                graphProgress: Object.freeze({
                  completedQuanta: 1,
                  totalQuanta: 341,
                  completedCommandBuffers: 1,
                  totalCommandBuffers: 341,
                  queueDrains: 0,
                  cooperativeIdleMs: 0,
                  completedEvaluations: 0,
                  totalEvaluations: 8,
                  quantum: {} as never,
                  subquantumIndex: descriptor.subquantumIndex,
                  subquantumCount: descriptor.subquantumCount,
                  commandId: descriptor.commandId,
                  batchIndex: 0,
                  batchFirstPhysicalQuantum: 0,
                  batchPhysicalQuantumCount: 1,
                  physicalQuantaPerCommandBuffer: 1,
                }),
              }));
              ditOptions.signal?.throwIfAborted();
            }
            ditOptions.onOpt0080EvaluationProfile(profile);
            const candidate = schedulingProfile ===
              "opt-0080-depth2-epoch4";
            return {
              finalLatent: new Float32Array(288_000).fill(0.25),
              shape: {
                batch: 1,
                latentFrames: 4_500,
                channels: 64,
              },
              commandBuffersSubmitted: 342,
              queueDrains: candidate ? 87 : 342,
              cooperativeIdleMs: candidate ? 86 : 341,
              completedEvaluations: 1,
              attentionRouteProfile: fakeOpt0070AttentionRouteProfile(),
            };
          }
          if (ditOptions.onOpt0080FullProfile !== undefined) {
            const schedulingProfile = ditOptions.opt0080FullSchedulingProfile;
            if (schedulingProfile === undefined) {
              throw new Error("test OPT-0080 full scheduling profile missing");
            }
            const profile = fakeOpt0080FullDitCommandProfile(
              schedulingProfile,
            );
            if (ditOptions.onOpt0080CommandBufferCompleted !== undefined) {
              const descriptor = profile.descriptorTable.descriptors[0]!;
              ditOptions.onOpt0080CommandBufferCompleted(Object.freeze({
                commandBufferIndex: 0,
                submitThroughCompletionFenceMs: 1,
                trueQueueDrain: false,
                completionEpochIndex: 0,
                descriptor,
                schedulingProgress: Object.freeze({
                  completedCommandBuffers: 1,
                  totalCommandBuffers: 2_553,
                  completionFenceRequestedCount: 2,
                  completionFenceSettledCount: 1,
                  completionFenceRejectedCount: 0,
                  trueQueueDrainCount: 0,
                  completionEpochCount: 0,
                  requestedCooperativeIdleMs: 0,
                  cooperativeIdleTurns: 0,
                  outstandingCommandBuffers: 1,
                }),
                graphProgress: Object.freeze({
                  completedQuanta: 1,
                  totalQuanta: 2_553,
                  completedCommandBuffers: 1,
                  totalCommandBuffers: 2_553,
                  queueDrains: 0,
                  cooperativeIdleMs: 0,
                  completedEvaluations: 0,
                  totalEvaluations: 8,
                  quantum: {} as never,
                  subquantumIndex: descriptor.subquantumIndex,
                  subquantumCount: descriptor.subquantumCount,
                  commandId: descriptor.commandId,
                  batchIndex: 0,
                  batchFirstPhysicalQuantum: 0,
                  batchPhysicalQuantumCount: 1,
                  physicalQuantaPerCommandBuffer: 1,
                }),
              }));
              ditOptions.signal?.throwIfAborted();
            }
            ditOptions.onOpt0080FullProfile(profile);
            const candidate = schedulingProfile ===
              "opt-0080-depth2-epoch4";
            const finalLatent = new Float32Array(288_000).fill(0.25);
            return {
              finalLatent,
              shape: {
                batch: 1,
                latentFrames: 4_500,
                channels: 64,
              },
              commandBuffersSubmitted: 2_554,
              queueDrains: candidate ? 640 : 2_554,
              cooperativeIdleMs: candidate ? 639 : 2_553,
              completedEvaluations: 8,
              ...(ditOptions.captureEvaluationLatents === true
                ? {
                    evaluationLatents: Object.freeze(Array.from(
                      { length: 8 },
                      (_, evaluation) => new Float32Array(288_000).fill(
                        evaluation === 7 ? 0.25 : (evaluation + 1) / 32,
                      ),
                    )),
                  }
                : {}),
              attentionRouteProfile: fakeOpt0070AttentionRouteProfile(),
            };
          }
          if (ditOptions.onOpt0067EvaluationProfile !== undefined) {
            ditOptions.onOpt0067EvaluationProfile(
              fakeOpt0067DitCommandProfile(),
            );
            return {
              finalLatent: new Float32Array(288_000).fill(0.25),
              shape: {
                batch: 1,
                latentFrames: 4_500,
                channels: 64,
              },
              commandBuffersSubmitted: 342,
              queueDrains: 342,
              cooperativeIdleMs: 341,
              completedEvaluations: 1,
            };
          }
          if (ditOptions.onSchedulingProfile !== undefined) {
            const batch = ditOptions.physicalQuantaPerCommandBuffer ?? 1;
            const profile = fakeOpt0034DitSchedulingProfile(batch);
            ditOptions.onSchedulingProfile(profile);
            return {
              finalLatent: new Float32Array(288_000).fill(0.25),
              shape: {
                batch: 1,
                latentFrames: 4_500,
                channels: 64,
              },
              commandBuffersSubmitted: profile.totalCommandBufferCount,
              queueDrains: profile.totalQueueDrainCount,
              cooperativeIdleMs: profile.graphCommandBufferCount,
              completedEvaluations: 8,
            };
          }
          if (ditOptions.captureSamplerScheduleEvidence === true) {
            const evaluationCount = ditOptions.samplerScheduleProfile
              ?.evaluationCount;
            if (evaluationCount === undefined) {
              throw new Error("test sampler profile missing");
            }
            const elements = ditOptions.shape.latentFrames * 64;
            const finalLatent = new Float32Array(elements).fill(
              evaluationCount / 16,
            );
            return {
              finalLatent,
              shape: {
                batch: 1,
                latentFrames: ditOptions.shape.latentFrames,
                channels: 64,
              },
              commandBuffersSubmitted: 2,
              queueDrains: 2,
              cooperativeIdleMs: 1,
              completedEvaluations: evaluationCount,
              evaluation0Velocity: new Float32Array(elements).fill(0.125),
              evaluationLatents: Object.freeze(Array.from(
                { length: evaluationCount },
                (_, evaluation) => evaluation === evaluationCount - 1
                  ? Float32Array.from(finalLatent)
                  : new Float32Array(elements).fill((evaluation + 1) / 32),
              )),
            };
          }
          return {
            finalLatent: new Float32Array(1),
            shape: {
              batch: 1,
              latentFrames: ditOptions.shape.latentFrames,
              channels: 64,
            },
            commandBuffersSubmitted: 2,
            queueDrains: 2,
            cooperativeIdleMs: 1,
            completedEvaluations: 8,
          };
        },
        async destroy() {
          if (destroyed) return;
          destroyed = true;
          packageEvents.push("dit-destroy");
          ditOptions.ownedDitWeights.destroy();
          ditOptions.ownedDitDenseWeights?.destroy();
        },
      };
    },
    ...(options.opt0081WorkingOwner === true
      ? {
          createOpt0081Representative: async (ditOptions) => {
            if (ditOptions.setupFailurePoint === "after-readback") {
              ditOptions.ownedDitWeights.destroy();
              ditOptions.ownedDitDenseWeights.destroy();
              ditOptions.onSetupCleanup?.(Object.freeze({
                schema: "ace-opt-0081-representative-setup-cleanup-v1",
                createdGraphBufferCount: 247,
                destroyedGraphBufferCount: 247,
                liveGraphBufferCount: 0,
                liveGraphByteCount: 0,
                runtimeOwnerCount: 2,
                destroyedRuntimeOwnerCount: 2,
                residentModelDestroyed: true,
                mappedRangeCount: 0,
                unmappedRangeCount: 0,
                liveMapCount: 0,
                pendingDescriptorCount: 0,
                activeCallbackCount: 0,
                activeLeaseCount: 0,
                armReleaseCount: 2,
                drainOrderViolationCount: 0,
              }));
              throw new AceOpt0081RepresentativeInjectedSetupFailure(
                "after-readback",
              );
            }
            let destroyed = false;
            return {
              memory: { accountedGpuBytes: 1 },
              async destroy() {
                if (destroyed) return;
                destroyed = true;
                opt0081OwnerDestroy();
                ditOptions.ownedDitWeights.destroy();
                ditOptions.ownedDitDenseWeights.destroy();
              },
            } as unknown as AceOpt0081RepresentativeDitSession;
          },
        }
      : options.opt0081UnexpectedSetupOwner !== true
      ? {}
      : {
          createOpt0081Representative: async (ditOptions) => {
            let destroyed = false;
            return {
              memory: { accountedGpuBytes: 1 },
              async destroy() {
                if (destroyed) return;
                destroyed = true;
                opt0081OwnerDestroy();
                ditOptions.ownedDitWeights.destroy();
                ditOptions.ownedDitDenseWeights.destroy();
              },
            } as unknown as AceOpt0081RepresentativeDitSession;
          },
        }),
    createVae: async (vaeOptions) => {
      lastVaeMaximumWindowFrames = vaeOptions.maximumWindowFrames;
      lastVaePlanChunkFrames = vaeOptions.plan.chunkFrames;
      lastVaeRuntimeProfile = vaeOptions.runtimeProfileId;
      lastVaeProductionSchedulingPolicy =
        vaeOptions.productionSchedulingPolicy;
      expect([
        "opt-0028-mixed-fp16-fixed32-exact-packed-v1",
        ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PROFILE.id,
        ACE_OPT_0088_VAE_FP16_PORTABLE_DUAL_K4_PROFILE.id,
      ]).toContain(vaeOptions.runtimeProfileId);
      if (
        vaeOptions.runtimeProfileId ===
          ACE_OPT_0088_VAE_FP16_PORTABLE_DUAL_K4_PROFILE.id
      ) {
        // The portable backend options variant has no subgroup members.
        expect("subgroupMinSize" in vaeOptions).toBe(false);
        expect("subgroupMaxSize" in vaeOptions).toBe(false);
      } else {
        expect(vaeOptions).toMatchObject({
          subgroupMinSize: 32,
          subgroupMaxSize: 32,
        });
      }
      expect(vaeOptions.quantaPerCommandBuffer).toBe(64);
      packageEvents.push("vae-backend");
      let destroyed = false;
      return {
        memory: { accountedGpuBytes: 300 },
        async decodeWindow(
          window: AceVaeDecodeWindow,
          _signal?: AbortSignal,
          opt0080Run?: AceOpt0080VaeRunOptions,
        ) {
          const schedulingProfile = opt0080Run?.schedulingProfile ??
            resolveOpt0080VaeProductionWindowSchedulingProfile(
              vaeOptions.productionSchedulingPolicy,
              window.latentWindowFrames,
            );
          emitVae(
            vaeOptions.onProgress,
            window,
            "decoder",
            schedulingProfile,
          );
          emitVae(
            vaeOptions.onProgress,
            window,
            "readback",
            schedulingProfile,
          );
          opt0080Run?.onSchedulingEvidence(
            fakeOpt0080VaeSchedulingEvidence(window, schedulingProfile),
          );
          vaeOptions.onFamilyProfile?.(fakeVaeFamilyProfile(
            window,
            vaeOptions.quantaPerCommandBuffer ?? 8,
          ));
          return new Float32Array(
            window.decodedAudioFrames * vaeOptions.plan.audioChannels,
          );
        },
        async destroy() {
          if (destroyed) return;
          destroyed = true;
          vaeOptions.ownedVaeWeights.destroy();
        },
      };
    },
    beginAudio: async (transactionId, plan) => {
      audioBegins += 1;
      await options.beginAudioGate?.promise;
      let rollbackAttempts = 0;
      return {
      transactionId,
      rawSink: { writeCore: async () => undefined },
      async commit(_postprocess, commitOptions) {
        audioCommits += 1;
        options.abortAfterCommit?.abort();
        return {
          audio: new Blob([new Uint8Array(44)], { type: "audio/wav" }),
          transactionId,
          wav: {
            headerBytes: 44,
            dataBytes: plan.outputFloat32Bytes,
            wavBytes: plan.outputFloat32Bytes + 44,
            outputPeak: 0,
          },
          ...(commitOptions?.retainRawSnapshot === true
            ? {
                rawSnapshot: new Blob([
                  new Uint8Array(plan.outputFloat32Bytes),
                ], { type: "application/x-ace-raw-f32" }),
              }
            : {}),
        };
      },
      async rollback() {
        audioRollbacks += 1;
        rollbackAttempts += 1;
        if (rollbackAttempts <= (options.failRollbackAttempts ?? 0)) {
          throw new Error("injected rollback failure");
        }
      },
    };
    },
    releaseAudio: async (transactionId) => {
      releaseAttempts += 1;
      if (releaseAttempts <= (options.failReleaseAttempts ?? 0)) {
        throw new Error("injected release failure");
      }
      releasedIds.push(transactionId);
    },
  };
  const backend = createAceWebGpuPipelineBackendForTest(dependencies);
  return {
    backend,
    loadedPhases,
    plannerResources,
    packageEvents,
    phaseDestroyed,
    releasedIds,
    requestDevice,
    deviceDestroy,
    opt0081OwnerDestroy,
    emitRuntimeEvent(event) {
      if (runtimeEvent === undefined) {
        throw new Error("ACE test runtime event sink is unavailable");
      }
      runtimeEvent(event);
    },
    get audioRollbacks() {
      return audioRollbacks;
    },
    get audioCommits() {
      return audioCommits;
    },
    get audioBegins() {
      return audioBegins;
    },
    get releaseAttempts() {
      return releaseAttempts;
    },
    get lastDitAttentionRuntimeProfile() {
      return lastDitAttentionRuntimeProfile;
    },
    get lastDitSubmissionPolicy() {
      return lastDitSubmissionPolicy;
    },
    get lastDitCaptureOpt0062AttentionIdentity() {
      return lastDitCaptureOpt0062AttentionIdentity;
    },
    get lastVaeMaximumWindowFrames() {
      return lastVaeMaximumWindowFrames;
    },
    get lastVaePlanChunkFrames() {
      return lastVaePlanChunkFrames;
    },
    get lastVaeRuntimeProfile() {
      return lastVaeRuntimeProfile;
    },
    get lastVaeProductionSchedulingPolicy() {
      return lastVaeProductionSchedulingPolicy;
    },
  };
}

interface Deferred<T> {
  readonly promise: Promise<T>;
  resolve(value: T): void;
}

function deferred<T>(): Deferred<T> {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((accept) => {
    resolve = accept;
  });
  return { promise, resolve };
}

async function waitFor(predicate: () => boolean): Promise<void> {
  for (let index = 0; index < 100; index += 1) {
    if (predicate()) return;
    await Promise.resolve();
  }
  throw new Error("test condition was not reached");
}

function fakePhase(
  phases: readonly AceTensorPhase[],
  manifest: AcePackageManifest,
  onDestroy: () => void,
): AceGpuTensorPhase {
  let destroyed = false;
  return {
    phases,
    packageManifest: manifest,
    residentBytes: 1,
    destroy() {
      if (destroyed) return;
      destroyed = true;
      onDestroy();
    },
  } as AceGpuTensorPhase;
}

function loadedTokenizer(kind: "text" | "planner"): LoadedAceTokenizer {
  const tokenizer = {
    kind,
    encodeBatch(texts: readonly string[]) {
      return {
        inputIds: texts.map(() => [1, 2]),
        attentionMask: texts.map(() => [1, 1]),
      };
    },
  } as unknown as AceQwenBpeTokenizer;
  return {
    tokenizer,
    chatTemplate: "",
    assetIdentity: {
      tokenizerSha256: "a".repeat(64),
      tokenizerConfigSha256: "b".repeat(64),
      chatTemplateSha256: "c".repeat(64),
    },
  };
}

function emitConditioning(
  callback: ((progress: AceConditioningGpuProgress) => void) | undefined,
  phase: AceConditioningGpuProgress["phase"],
): void {
  callback?.({
    phase,
    completedCommandBuffers: 1,
    totalCommandBuffers: 1,
    queueDrains: 1,
    cooperativeIdleMs: 1,
    quantumId: `${phase}-test`,
  });
}

function emitVae(
  callback: ((progress: AceVaeChunkGpuBackendProgress) => void) | undefined,
  window: AceVaeDecodeWindow,
  stage: AceVaeChunkGpuBackendProgress["stage"],
  schedulingProfile: "depth1-epoch1" | "depth2-phase-epoch4" =
    "depth1-epoch1",
): void {
  const decoderQuantumCount = exactVaeDecoderQuantumCount(
    window.latentWindowFrames,
  );
  const decoderCommandBufferCount = Math.ceil(decoderQuantumCount / 64);
  const totalCommandBuffers = decoderCommandBufferCount + 1;
  const candidate = schedulingProfile === "depth2-phase-epoch4";
  const finalQueueDrains = candidate
    ? Math.ceil(totalCommandBuffers / 4)
    : totalCommandBuffers;
  const queueDrains = stage === "decoder" ? 1 : finalQueueDrains;
  callback?.({
    windowIndex: window.index,
    completedDecoderQuanta: stage === "decoder"
      ? 1
      : decoderQuantumCount,
    totalDecoderQuanta: decoderQuantumCount,
    completedCommandBuffers: stage === "decoder"
      ? 1
      : totalCommandBuffers,
    totalCommandBuffers,
    queueDrains,
    cooperativeIdleMs: Math.max(0, queueDrains - 1),
    stage,
  });
}

function fakeOpt0080VaeSchedulingEvidence(
  window: AceVaeDecodeWindow,
  schedulingProfile: "depth1-epoch1" | "depth2-phase-epoch4",
) {
  const decoderQuantumCount = exactVaeDecoderQuantumCount(
    window.latentWindowFrames,
  );
  const decoderCommandBufferCount = Math.ceil(decoderQuantumCount / 64);
  const totalCommandBufferCount = decoderCommandBufferCount + 1;
  const candidate = schedulingProfile === "depth2-phase-epoch4";
  const trueQueueDrainCount = candidate
    ? Math.ceil(totalCommandBufferCount / 4)
    : totalCommandBufferCount;
  return {
    schema: "ace-opt-0080-vae-window-scheduling-v1" as const,
    windowIndex: window.index,
    schedulingProfile,
    decoderQuantumCount,
    quantaPerCommandBuffer: 64,
    decoderCommandBufferCount,
    readbackCommandBufferCount: 1 as const,
    totalCommandBufferCount,
    schedulingWallMs: 1,
    commandBuffersSubmitted: totalCommandBufferCount,
    completionFenceRequestedCount: totalCommandBufferCount,
    completionFenceSettledCount: totalCommandBufferCount,
    completionFenceRejectedCount: 0,
    trueQueueDrainCount,
    completionEpochCount: trueQueueDrainCount,
    cooperativeIdleTurns: Math.max(0, trueQueueDrainCount - 1),
    requestedCooperativeIdleMs: Math.max(0, trueQueueDrainCount - 1),
    maximumOutstandingCommandBuffers: candidate ? 2 as const : 1 as const,
    commandCompletions: Object.freeze([]),
    completionEpochs: Object.freeze([]),
  };
}

const vaeDecoderQuantumCounts = new Map<number, number>();

function exactVaeDecoderQuantumCount(latentWindowFrames: number): number {
  const retained = vaeDecoderQuantumCounts.get(latentWindowFrames);
  if (retained !== undefined) return retained;
  const count = planAceOpt0011Fp16VaeWindowDynamicControls(
    latentWindowFrames,
    256,
  ).recordCount;
  vaeDecoderQuantumCounts.set(latentWindowFrames, count);
  return count;
}

function fakeOpt0056DenseRouteProfile(): AceOpt0056DenseRouteProfile {
  const operations = Object.freeze([
    "self-query-projection",
    "self-key-projection",
    "self-value-projection",
    "self-output-projection",
    "cross-query-projection",
    "cross-output-projection",
    "mlp-gate-projection",
    "mlp-up-projection",
    "mlp-down-projection",
  ] as const satisfies readonly AceOpt0056DenseOperation[]);
  const routes = Object.freeze(Array.from({ length: 24 }, (_, layer) =>
    operations.map((operation) => {
      const exact = operation === "mlp-down-projection";
      const inner = exact ? 6_144 : 2_048;
      const columns = operation === "self-key-projection" ||
          operation === "self-value-projection"
        ? 1_024
        : operation === "mlp-gate-projection" ||
            operation === "mlp-up-projection"
          ? 6_144
          : 2_048;
      return Object.freeze({
        routeKey: `ace-dit-layer-${layer}-${operation}`,
        layer,
        operation,
        rows: 2_250 as const,
        inner,
        columns,
        owner: exact
          ? "opt-0056-k4-exact-fp32" as const
          : "opt-0032-k4-fp16-partials" as const,
        kernelId: exact
          ? ACE_OPT_0056_DENSE_K4_EXACT_KERNEL_ID
          : ACE_OPT_0037_DENSE_K4_KERNEL_ID,
        evaluationDispatchCount: 8 as const,
        evaluationLabels: Object.freeze(Array.from(
          { length: 8 },
          (_, evaluation) =>
            `ace-dit-eval-${evaluation}-layer-${layer}-${operation}`,
        )),
      });
    })
  ).flat());
  return Object.freeze({
    schema: "ace-opt-0056-selective-dense-routes-v1",
    kernelSetId: ACE_OPT_0056_SELECTIVE_DENSE_KERNEL_SET_ID,
    routeCount: 216,
    dispatchCount: 1_728,
    approximateRouteCount: 192,
    exactDownRouteCount: 24,
    routes,
  });
}

function fakeOpt0034DitSchedulingProfile(
  physicalQuantaPerCommandBuffer: 1 | 8 | 16,
): AceOpt0034DitSchedulingProfile {
  const physicalGraphQuantumCount = 2_553;
  const graphCommandBufferCount = Math.ceil(
    physicalGraphQuantumCount / physicalQuantaPerCommandBuffer,
  );
  const primitiveBase = Math.floor(6_833 / graphCommandBufferCount);
  const primitiveRemainder = 6_833 % graphCommandBufferCount;
  const multiplyAddTotal = 26_840_955_355_136;
  const multiplyAddBase = Math.floor(
    multiplyAddTotal / graphCommandBufferCount,
  );
  const multiplyAddRemainder =
    multiplyAddTotal - multiplyAddBase * graphCommandBufferCount;
  const batches = Object.freeze(Array.from(
    { length: graphCommandBufferCount },
    (_, batchIndex) => {
      const firstPhysicalIndex =
        batchIndex * physicalQuantaPerCommandBuffer;
      const lastPhysicalIndex = Math.min(
        physicalGraphQuantumCount,
        firstPhysicalIndex + physicalQuantaPerCommandBuffer,
      ) - 1;
      return Object.freeze({
        batchIndex,
        firstPhysicalIndex,
        lastPhysicalIndex,
        physicalQuantumCount: lastPhysicalIndex - firstPhysicalIndex + 1,
        primitiveCount:
          primitiveBase + (batchIndex < primitiveRemainder ? 1 : 0),
        scheduledMultiplyAdds:
          multiplyAddBase + (batchIndex < multiplyAddRemainder ? 1 : 0),
        submitThroughDrainMs: 0.5,
      });
    },
  ));
  return Object.freeze({
    schema: "ace-dit-opt0034-command-buffer-coalescing-v1",
    physicalGraphQuantumCount: 2_553 as const,
    physicalQuantaPerCommandBuffer,
    graphCommandBufferCount,
    readbackCommandBufferCount: 1 as const,
    totalCommandBufferCount: graphCommandBufferCount + 1,
    graphQueueDrainCount: graphCommandBufferCount,
    totalQueueDrainCount: graphCommandBufferCount + 1,
    graphRequestedIdleMs: graphCommandBufferCount - 1,
    graphToReadbackRequestedIdleMs: 1 as const,
    graphSubmitThroughDrainMs: graphCommandBufferCount * 0.5,
    graphWallMs: graphCommandBufferCount * 0.5 + 10,
    graphToReadbackObservedIdleMs: 1,
    readbackSubmitThroughDrainMs: 2,
    readbackMapDetachMs: 1,
    backendWallMs: graphCommandBufferCount * 0.5 + 14,
    maximumPhysicalQuantaPerBatch: physicalQuantaPerCommandBuffer,
    maximumPrimitiveCountPerBatch: Math.max(
      ...batches.map((batch) => batch.primitiveCount),
    ),
    maximumScheduledMultiplyAddsPerBatch: Math.max(
      ...batches.map((batch) => batch.scheduledMultiplyAdds),
    ),
    maximumBatchSubmitThroughDrainMs: 0.5,
    descriptorTableSha256:
      "aedf8c74d2bb15601d1385e9c8e9da49e58905ab156a04c99271f8de3633dd76",
    physicalPrimitiveCount: 6_833,
    scheduledMultiplyAdds: multiplyAddTotal,
    batches,
  });
}

function fakeOpt0018DitCommandProfile(): AceOpt0018DitCommandProfile {
  const familyNames = [
    "precompute",
    "cross-cache",
    "timestep",
    "input",
    "attention-projections",
    "self-full",
    "self-sliding",
    "cross-attention",
    "feed-forward",
    "plumbing",
    "output",
    "sampler-dcw",
    "mixed",
  ] as const;
  const aggregate = (commandBufferCount: number) => Object.freeze({
    commandBufferCount,
    primitiveCount: commandBufferCount,
    scheduledMultiplyAdds: 0,
    submitThroughDrainMs: commandBufferCount / 10,
    maximumSubmitThroughDrainMs: commandBufferCount === 0 ? 0 : 0.1,
  });
  const descriptors = Object.freeze(Array.from({ length: 2_553 }, (_, index) => {
    const first = index === 0;
    const last = index === 2_552;
    const precompute = index < 25;
    const family = first
      ? "precompute"
      : precompute
      ? "cross-cache"
      : last
      ? "sampler-dcw"
      : "mixed";
    const commandId = first
      ? "ace-dit-condition-projection"
      : last
      ? "ace-dit-eval-7-sampler"
      : `opt-0018-fixture-${index}`;
    const ids = first
      ? ["ace-dit-condition-projection-range-0"]
      : last
      ? [
          "ace-dit-eval-7-sampler-sampler-update",
          "ace-dit-eval-7-sampler-predicted-clean",
          "ace-dit-eval-7-sampler-dcw",
        ]
      : [`${commandId}-member-0`];
    return Object.freeze({
      physicalIndex: index,
      logicalIndex: first ? 0 : last ? 248 : Math.min(index, 247),
      logicalKind: first
        ? "condition-projection"
        : precompute
        ? "cross-cache"
        : last
        ? "sampler"
        : "layer",
      commandId,
      subquantumIndex: 0,
      subquantumCount: 1,
      evaluation: precompute ? null : Math.min(7, Math.floor((index - 25) / 316)),
      layer: first || last ? null : index % 24,
      family,
      primitiveCount: ids.length,
      scheduledMultiplyAdds: 0,
      members: Object.freeze(ids.map((id) => Object.freeze({
        id,
        family: family === "mixed" ? "plumbing" : family,
        backend: "fixture",
        kernel: "fixture",
        scheduledMultiplyAdds: 0,
      }))),
    });
  }));
  const counts = Object.freeze({
    precompute: 1,
    "cross-cache": 24,
    timestep: 8,
    input: 8,
    "attention-projections": 192,
    "self-full": 384,
    "self-sliding": 0,
    "cross-attention": 0,
    "feed-forward": 576,
    plumbing: 0,
    output: 8,
    "sampler-dcw": 8,
    mixed: 1_344,
  });
  const families = Object.freeze(Object.fromEntries(familyNames.map((family) => [
    family,
    aggregate(counts[family]),
  ])));
  const familyByBucket = Object.freeze(Object.fromEntries(familyNames.map(
    (family) => [
      family,
      Object.freeze(Array.from({ length: 9 }, (_unused, bucket) =>
        aggregate(bucket === 0
          ? (family === "precompute" ? 1 : family === "cross-cache" ? 24 : 0)
          : counts[family] / 8)
      )),
    ],
  )));
  return Object.freeze({
    schema: "ace-dit-m2250-command-profile-v1",
    descriptorTable: Object.freeze({
      descriptors,
      sha256:
        "aedf8c74d2bb15601d1385e9c8e9da49e58905ab156a04c99271f8de3633dd76",
      serializedBytes: 1_869_566,
      memberCount: 6_833,
      preparationMs: 1,
    }),
    graphCommandBufferCount: 2_553,
    readbackCommandBufferCount: 1,
    totalCommandBufferCount: 2_554,
    timings: Object.freeze(Array.from({ length: 2_553 }, () => 0.1)),
    graphSubmitThroughDrainMs: 255.3,
    graphWallMs: 260,
    graphRequestedIdleMs: 2_552,
    graphResidualMs: -2_547.3,
    graphToReadbackRequestedIdleMs: 1,
    graphToReadbackObservedIdleMs: 1,
    readbackSubmitThroughDrainMs: 1,
    readbackMapDetachMs: 1,
    backendWallMs: 263,
    backendResidualMs: 0,
    families,
    precompute: aggregate(25),
    evaluations: Object.freeze(Array.from({ length: 8 }, () => aggregate(316))),
    familyByBucket,
    slowest: Object.freeze(Array.from({ length: 16 }, (_, index) =>
      Object.freeze({
        physicalIndex: index,
        family: descriptors[index]!.family,
        evaluation: descriptors[index]!.evaluation,
        layer: descriptors[index]!.layer,
        submitThroughDrainMs: 0.1,
      })
    )),
    reconciliationToleranceMs: 1e-6,
    timingStorageBytes: 2_553 * 9,
  }) as AceOpt0018DitCommandProfile;
}

function fakeOpt0067DitCommandProfile(): AceOpt0067DitCommandProfile {
  const full = fakeOpt0018DitCommandProfile();
  const descriptors = Object.freeze(full.descriptorTable.descriptors.map(
    (descriptor) => {
      if (descriptor.physicalIndex !== 25) return descriptor;
      const members = [
        ...descriptor.members,
        ...Array.from({ length: 480 }, (_, index) => Object.freeze({
          id: `opt-0067-fixture-self-full-${index}`,
          family: "self-full" as const,
          backend: "fixed32-subgroup-query8",
          kernel: "fixed32-subgroup-query8",
          scheduledMultiplyAdds: 0,
        })),
      ];
      return Object.freeze({ ...descriptor, members: Object.freeze(members) });
    },
  ));
  const add = (
    left: AceOpt0018DitCommandProfile["precompute"],
    right: AceOpt0018DitCommandProfile["precompute"],
  ) => Object.freeze({
    commandBufferCount: left.commandBufferCount + right.commandBufferCount,
    primitiveCount: left.primitiveCount + right.primitiveCount,
    scheduledMultiplyAdds:
      left.scheduledMultiplyAdds + right.scheduledMultiplyAdds,
    submitThroughDrainMs:
      left.submitThroughDrainMs + right.submitThroughDrainMs,
    maximumSubmitThroughDrainMs: Math.max(
      left.maximumSubmitThroughDrainMs,
      right.maximumSubmitThroughDrainMs,
    ),
  });
  const familyByBucket = Object.freeze(Object.fromEntries(
    Object.entries(full.familyByBucket).map(([family, buckets]) => [
      family,
      Object.freeze([buckets[0]!, buckets[1]!]),
    ]),
  )) as AceOpt0067DitCommandProfile["familyByBucket"];
  const families = Object.freeze(Object.fromEntries(
    Object.entries(familyByBucket).map(([family, buckets]) => [
      family,
      add(buckets[0], buckets[1]),
    ]),
  )) as AceOpt0067DitCommandProfile["families"];
  return Object.freeze({
    schema: "ace-dit-opt0067-evaluation0-command-profile-v1",
    descriptorTable: Object.freeze({
      ...full.descriptorTable,
      descriptors,
      memberCount: descriptors.reduce(
        (total, descriptor) => total + descriptor.members.length,
        0,
      ),
    }),
    graphCommandBufferCount: 341,
    readbackCommandBufferCount: 1,
    totalCommandBufferCount: 342,
    graphQueueDrainCount: 341,
    totalQueueDrainCount: 342,
    evaluationCommandBufferCount: 316,
    timings: Object.freeze(Array.from({ length: 341 }, () => 0.1)),
    graphSubmitThroughDrainMs: 34.1,
    graphWallMs: 500,
    graphRequestedIdleMs: 340,
    graphResidualMs: 125.9,
    evaluationWallMs: 450,
    evaluationRequestedIdleMs: 316,
    evaluationResidualMs: 102.4,
    graphToReadbackRequestedIdleMs: 1,
    graphToReadbackObservedIdleMs: 1,
    readbackSubmitThroughDrainMs: 1,
    readbackMapDetachMs: 1,
    backendWallMs: 503,
    backendResidualMs: 0,
    families,
    precompute: full.precompute,
    evaluation: full.evaluations[0]!,
    familyByBucket,
    slowest: full.slowest,
    reconciliationToleranceMs: full.reconciliationToleranceMs,
    timingStorageBytes: 341 * 9,
  });
}

function fakeOpt0080DitCommandProfile(
  schedulingProfile: AceOpt0080DitCheckpoint["schedulingProfile"],
): AceOpt0080DitCommandProfile {
  const full = fakeOpt0018DitCommandProfile();
  const candidate = schedulingProfile === "opt-0080-depth2-epoch4";
  const graphTrueQueueDrainCount = candidate ? 86 as const : 341 as const;
  const totalTrueQueueDrainCount = candidate ? 87 as const : 342 as const;
  const graphCooperativeIdleTurns = candidate ? 85 as const : 340 as const;
  const totalCooperativeIdleTurns = candidate ? 86 as const : 341 as const;
  const maximumOutstandingCommandBuffers = candidate ? 2 as const : 1 as const;
  const graphCompletionEpochs = [];
  let completionEpochIndex = 0;
  let firstCommandBufferIndex = 0;
  for (const [phaseIndex, commandBufferCount] of [25, 316].entries()) {
    const phaseEnd = firstCommandBufferIndex + commandBufferCount;
    const epochSize = candidate ? 4 : 1;
    for (
      let first = firstCommandBufferIndex;
      first < phaseEnd;
      first += epochSize
    ) {
      const last = Math.min(first + epochSize, phaseEnd) - 1;
      graphCompletionEpochs.push(Object.freeze({
        completionEpochIndex,
        phaseIndex,
        firstCommandBufferIndex: first,
        lastCommandBufferIndex: last,
        commandBufferCount: last - first + 1,
        submitThroughTrueDrainMs: last - first + 1,
      }));
      completionEpochIndex += 1;
    }
    firstCommandBufferIndex = phaseEnd;
  }
  return Object.freeze({
    schema: "ace-dit-opt0080-evaluation0-command-profile-v1",
    schedulingProfile,
    descriptorTable: full.descriptorTable,
    precomputeWallMs: 100,
    evaluationWallMs: 500,
    graphWallMs: 600,
    graphToReadbackObservedIdleMs: 1,
    readbackSubmitThroughCompletionFenceMs: 2,
    readbackMapDetachMs: 1,
    backendWallMs: 604,
    topology: Object.freeze({
      schedulingProfile,
      descriptorTableMemberCount: 6_833 as const,
      graphCommandBufferCount: 341 as const,
      readbackCommandBufferCount: 1 as const,
      totalCommandBufferCount: 342 as const,
      commandBuffersSubmitted: 342 as const,
      completionFenceRequestedCount: 342 as const,
      completionFenceSettledCount: 342 as const,
      completionFenceRejectedCount: 0 as const,
      graphTrueQueueDrainCount,
      totalTrueQueueDrainCount,
      graphCompletionEpochCount: graphTrueQueueDrainCount,
      graphCooperativeIdleTurns,
      totalCooperativeIdleTurns,
      graphRequestedCooperativeIdleMs: graphCooperativeIdleTurns,
      totalRequestedCooperativeIdleMs: totalCooperativeIdleTurns,
      maximumOutstandingCommandBuffers,
      maximumPendingDescriptorCount: maximumOutstandingCommandBuffers,
      pendingDescriptorCountAfterCleanup: 0 as const,
      graphCompletionEpochs: Object.freeze(graphCompletionEpochs),
      // Deliberately much larger than graphWallMs in aggregate. These
      // cumulative depth-two intervals are evidence, never additive wall time.
      submitThroughCompletionFenceMs: Object.freeze(
        Array.from({ length: 341 }, () => 10),
      ),
      graphToReadbackRequestedIdleMs: 1 as const,
      readbackSubmitThroughCompletionFenceMs: 2,
    }),
  });
}

function fakeOpt0080FullDitCommandProfile(
  schedulingProfile: AceOpt0080DitCheckpoint["schedulingProfile"],
): AceOpt0080FullDitCommandProfile {
  const full = fakeOpt0018DitCommandProfile();
  const candidate = schedulingProfile === "opt-0080-depth2-epoch4";
  const graphTrueQueueDrainCount = candidate ? 639 as const : 2_553 as const;
  const totalTrueQueueDrainCount = candidate ? 640 as const : 2_554 as const;
  const graphCooperativeIdleTurns = candidate ? 638 as const : 2_552 as const;
  const totalCooperativeIdleTurns = candidate ? 639 as const : 2_553 as const;
  const maximumOutstandingCommandBuffers = candidate ? 2 as const : 1 as const;
  const graphCompletionEpochs = [];
  let completionEpochIndex = 0;
  let firstCommandBufferIndex = 0;
  const phaseCounts = [25, 316, 316, 316, 316, 316, 316, 316, 316];
  for (const [phaseIndex, commandBufferCount] of phaseCounts.entries()) {
    const phaseEnd = firstCommandBufferIndex + commandBufferCount;
    const epochSize = candidate ? 4 : 1;
    for (let first = firstCommandBufferIndex; first < phaseEnd; first += epochSize) {
      const last = Math.min(first + epochSize, phaseEnd) - 1;
      graphCompletionEpochs.push(Object.freeze({
        completionEpochIndex,
        phaseIndex,
        firstCommandBufferIndex: first,
        lastCommandBufferIndex: last,
        commandBufferCount: last - first + 1,
        submitThroughTrueDrainMs: last - first + 1,
      }));
      completionEpochIndex += 1;
    }
    firstCommandBufferIndex = phaseEnd;
  }
  return Object.freeze({
    schema: "ace-dit-opt0080-full-command-profile-v1",
    schedulingProfile,
    descriptorTable: full.descriptorTable,
    precomputeWallMs: 25,
    evaluationWallMs: Object.freeze([
      316, 316, 316, 316, 316, 316, 316, 316,
    ] as const),
    graphWallMs: 2_553,
    graphToReadbackObservedIdleMs: 1,
    readbackSubmitThroughCompletionFenceMs: 2,
    readbackMapDetachMs: 1,
    backendWallMs: 2_557,
    topology: Object.freeze({
      schedulingProfile,
      descriptorTableMemberCount: 6_833 as const,
      graphCommandBufferCount: 2_553 as const,
      readbackCommandBufferCount: 1 as const,
      totalCommandBufferCount: 2_554 as const,
      commandBuffersSubmitted: 2_554 as const,
      completionFenceRequestedCount: 2_554 as const,
      completionFenceSettledCount: 2_554 as const,
      completionFenceRejectedCount: 0 as const,
      graphTrueQueueDrainCount,
      totalTrueQueueDrainCount,
      graphCompletionEpochCount: graphTrueQueueDrainCount,
      graphCooperativeIdleTurns,
      totalCooperativeIdleTurns,
      graphRequestedCooperativeIdleMs: graphCooperativeIdleTurns,
      totalRequestedCooperativeIdleMs: totalCooperativeIdleTurns,
      maximumOutstandingCommandBuffers,
      maximumPendingDescriptorCount: maximumOutstandingCommandBuffers,
      pendingDescriptorCountAfterCleanup: 0 as const,
      graphCompletionEpochs: Object.freeze(graphCompletionEpochs),
      submitThroughCompletionFenceMs: Object.freeze(
        Array.from({ length: 2_553 }, () => 10),
      ),
      graphToReadbackRequestedIdleMs: 1 as const,
      readbackSubmitThroughCompletionFenceMs: 2,
    }),
  });
}

function fakeOpt0070AttentionRouteProfile() {
  return Object.freeze({
    schema: "ace-opt-0070-production-attention-routes-v1" as const,
    runtimeProfileId:
      ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
    kernelSetId: ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_KERNEL_SET_ID,
    ownerMode: "exact-m2250-opt0062-quad" as const,
    expectedQueryTokens: 2_250,
    expectedConditionTokens: 98,
    routeCount: 384 as const,
    quadQueryRoutes: 96,
    query8FullSelfRoutes: 0,
    query8SlidingRoutes: 96 as const,
    query8CrossRoutes: 192 as const,
    query8OtherRoutes: 0 as const,
    unintendedQuadQueryRoutes: 0 as const,
    fullSelfRouteIds: Object.freeze(Array.from(
      { length: 96 },
      (_, index) => `opt-0080-full-self-${index}`,
    )),
    slidingSelfRouteIds: Object.freeze(Array.from(
      { length: 96 },
      (_, index) => `opt-0080-sliding-self-${index}`,
    )),
    crossRouteIds: Object.freeze(Array.from(
      { length: 192 },
      (_, index) => `opt-0080-cross-${index}`,
    )),
  });
}

function fakeVaeFamilyProfile(
  window: AceVaeDecodeWindow,
  quantaPerCommandBuffer = 8,
): AceOpt0011Fp16VaeWindowFamilyProfile {
  if (quantaPerCommandBuffer === 64) {
    const emptyFamily = Object.freeze({
      batchCount: 0,
      quantumCount: 0,
      submitThroughDrainMs: 0,
    });
    return Object.freeze({
      windowIndex: window.index,
      inputFrames: window.latentWindowFrames,
      quantaPerCommandBuffer,
      decoderBatchCount: 1,
      decoderQuantumCount: 48,
      decoderSubmitThroughDrainMs: 50,
      homogeneousBatchCount: 0,
      homogeneousQuantumCount: 0,
      homogeneousSubmitThroughDrainMs: 0,
      mixedBatchCount: 1,
      mixedQuantumCount: 48,
      mixedSubmitThroughDrainMs: 10,
      families: Object.freeze({
        "k7-conv1d": emptyFamily,
        "k1-conv1d": emptyFamily,
        "conv-transpose1d": emptyFamily,
        snake: emptyFamily,
        add: emptyFamily,
      }),
    });
  }
  const family = (
    submitThroughDrainMs: number,
  ) => Object.freeze({
    batchCount: 1,
    quantumCount: 8,
    submitThroughDrainMs,
  });
  return Object.freeze({
    windowIndex: window.index,
    inputFrames: window.latentWindowFrames,
    quantaPerCommandBuffer,
    decoderBatchCount: 6,
    decoderQuantumCount: 48,
    decoderSubmitThroughDrainMs: 50,
    homogeneousBatchCount: 5,
    homogeneousQuantumCount: 40,
    homogeneousSubmitThroughDrainMs: 40,
    mixedBatchCount: 1,
    mixedQuantumCount: 8,
    mixedSubmitThroughDrainMs: 10,
    families: Object.freeze({
      "k7-conv1d": family(10),
      "k1-conv1d": family(9),
      "conv-transpose1d": family(8),
      snake: family(7),
      add: family(6),
    }),
  });
}

async function initialize(
  harness: Harness,
  message = testInitializeMessage(),
): Promise<Readonly<{
  progress: readonly AceInitializationProgress[];
}>> {
  const progress: AceInitializationProgress[] = [];
  await harness.backend.initialize(message.configuration, {
    modelSource: message.modelSource,
    signal: new AbortController().signal,
    onProgress: (event) => progress.push(event),
    onDiagnostic: vi.fn(),
  });
  return { progress };
}

function createOpt0080Harness(): Harness {
  return createHarness({
    largeVaeLimits: true,
    mainManifestSha256: ACE_REFERENCE_MANIFEST_SHA256,
  });
}

function opt0080InitializeMessage(
  mainManifestSha256: string = ACE_REFERENCE_MANIFEST_SHA256,
) {
  const base = testInitializeMessage();
  return Object.freeze({
    ...base,
    configuration: Object.freeze({
      ...base.configuration,
      manifestSha256: mainManifestSha256,
      ditAttentionRuntimeProfile:
        ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
      vaePackage: Object.freeze({
        manifestUrl:
          "https://example.test/model/files-fp16-vae-revision7-experimental/manifest.json",
        manifestSha256: ACE_OPT_0054_VAE_REVISION7_MANIFEST_SHA256,
        runtimeProfile:
          ACE_OPT_0072_VAE_FP16_FIXED32_DUAL_K4_PRODUCTION_RUNTIME_PROFILE,
        windowRuntimeProfile:
          ACE_OPT_0070_VAE_C2378_WINDOW_RUNTIME_PROFILE,
        maxWindowFrames: 2_378 as const,
      }),
    }),
  });
}

function initializationContext() {
  return {
    modelSource: "cache-only" as const,
    signal: new AbortController().signal,
    onProgress: vi.fn(),
    onDiagnostic: vi.fn(),
  };
}

function generationContext() {
  return {
    signal: new AbortController().signal,
    onProgress: vi.fn(),
    onDiagnostic: vi.fn(),
  };
}

function uniqueStages(
  events: readonly AceGenerationProgress[],
): readonly string[] {
  const stages: string[] = [];
  for (const event of events) {
    if (stages.at(-1) !== event.stage) stages.push(event.stage);
  }
  return stages;
}
