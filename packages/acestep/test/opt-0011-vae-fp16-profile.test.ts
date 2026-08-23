import { createHash } from "node:crypto";

import { describe, expect, it } from "vitest";

import type { AcePackageManifest } from "../src/model/manifest.js";
import { ACE_MODEL_PROFILE_IDS } from "../src/webgpu/capabilities.js";
import { planAceVaeDecoder } from "../src/webgpu/vae-decoder.js";
import {
  ACE_OPT_0011_VAE_FP16_FIXED32_K7_HYBRID_KERNEL_SET_ID,
  ACE_OPT_0011_VAE_FP16_FIXED32_K7_HYBRID_PRECISION_MAP,
  ACE_OPT_0011_VAE_FP16_FIXED32_K7_HYBRID_PRECISION_MAP_CANONICAL_JSON,
  ACE_OPT_0011_VAE_FP16_FIXED32_K7_HYBRID_PRECISION_MAP_SHA256,
  ACE_OPT_0011_VAE_FP16_FIXED32_K7_HYBRID_PROFILE,
  ACE_OPT_0011_VAE_FP16_MANIFEST_BYTES,
  ACE_OPT_0011_VAE_FP16_MANIFEST_SHA256,
  ACE_OPT_0011_VAE_FP16_PORTABLE_KERNEL_SET_ID,
  ACE_OPT_0011_VAE_FP16_PORTABLE_PROFILE,
  ACE_OPT_0011_VAE_FP16_PRECISION_MAP,
  ACE_OPT_0011_VAE_FP16_PRECISION_MAP_CANONICAL_JSON,
  ACE_OPT_0011_VAE_FP16_PRECISION_MAP_SHA256,
  ACE_OPT_0028_VAE_FP16_PORTABLE_EXACT_PACKED_KERNEL_SET_ID,
  ACE_OPT_0028_VAE_FP16_PORTABLE_EXACT_PACKED_PRECISION_MAP,
  ACE_OPT_0028_VAE_FP16_PORTABLE_EXACT_PACKED_PRECISION_MAP_CANONICAL_JSON,
  ACE_OPT_0028_VAE_FP16_PORTABLE_EXACT_PACKED_PRECISION_MAP_SHA256,
  ACE_OPT_0028_VAE_FP16_PORTABLE_EXACT_PACKED_PROFILE,
  ACE_OPT_0015_VAE_FP16_FIXED32_K7_CONGRUENT_TRANSPOSE_KERNEL_SET_ID,
  ACE_OPT_0015_VAE_FP16_FIXED32_K7_CONGRUENT_TRANSPOSE_PRECISION_MAP,
  ACE_OPT_0015_VAE_FP16_FIXED32_K7_CONGRUENT_TRANSPOSE_PRECISION_MAP_CANONICAL_JSON,
  ACE_OPT_0015_VAE_FP16_FIXED32_K7_CONGRUENT_TRANSPOSE_PRECISION_MAP_SHA256,
  ACE_OPT_0015_VAE_FP16_FIXED32_K7_CONGRUENT_TRANSPOSE_PROFILE,
  ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_KERNEL_SET_ID,
  ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_PRECISION_MAP,
  ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_PRECISION_MAP_CANONICAL_JSON,
  ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_PRECISION_MAP_SHA256,
  ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_PROFILE,
  ACE_OPT_0028_VAE_FP16_MANIFEST_BYTES,
  ACE_OPT_0028_VAE_FP16_MANIFEST_SHA256,
  ACE_OPT_0040_VAE_FP16_FIXED32_SHAPE_SELECTED_KERNEL_SET_ID,
  ACE_OPT_0040_VAE_FP16_FIXED32_SHAPE_SELECTED_PRECISION_MAP,
  ACE_OPT_0040_VAE_FP16_FIXED32_SHAPE_SELECTED_PRECISION_MAP_CANONICAL_JSON,
  ACE_OPT_0040_VAE_FP16_FIXED32_SHAPE_SELECTED_PRECISION_MAP_SHA256,
  ACE_OPT_0040_VAE_FP16_FIXED32_SHAPE_SELECTED_PROFILE,
  ACE_OPT_0088_VAE_FP16_PORTABLE_DUAL_K4_KERNEL_SET_ID,
  ACE_OPT_0088_VAE_FP16_PORTABLE_DUAL_K4_PRECISION_MAP_SHA256,
  ACE_OPT_0088_VAE_FP16_PORTABLE_DUAL_K4_PROFILE,
  ACE_VAE_FP32_ORACLE_PROFILE,
  ACE_VAE_RUNTIME_PROFILE_IDS,
  hashAceVaePrecisionMap,
  selectAceVaeRuntimeProfile,
  type AceVaeAuthenticatedPackageIdentity,
  type AceVaeRuntimeProfileId,
} from "../src/webgpu/vae-fp16-profile.js";
import {
  ACE_OPT_0054_VAE_REVISION7_MANIFEST_BYTES,
  ACE_OPT_0054_VAE_REVISION7_MANIFEST_SHA256,
} from "../src/model/package.js";

const CANDIDATE = packageIdentity(
  "fp16-vae-experimental",
  5,
  ACE_OPT_0011_VAE_FP16_MANIFEST_SHA256,
  ACE_OPT_0011_VAE_FP16_MANIFEST_BYTES,
);
const ORACLE = packageIdentity(
  "reference",
  4,
  "18f36c6420976475af65ecd833ca56c6119706322ce54120389d4915d8e80db6",
  759_692,
);
const PACKED_CANDIDATE = packageIdentity(
  "fp16-vae-experimental",
  6,
  ACE_OPT_0028_VAE_FP16_MANIFEST_SHA256,
  ACE_OPT_0028_VAE_FP16_MANIFEST_BYTES,
);
const REVISION7_CANDIDATE = packageIdentity(
  "fp16-vae-experimental",
  7,
  ACE_OPT_0054_VAE_REVISION7_MANIFEST_SHA256,
  ACE_OPT_0054_VAE_REVISION7_MANIFEST_BYTES,
);
const DEVICE_LIMITS = Object.freeze({
  maxBufferSize: 256 * 1024 * 1024,
  maxStorageBufferBindingSize: 256 * 1024 * 1024,
  maxComputeWorkgroupStorageSize: 32 * 1024,
  maxComputeInvocationsPerWorkgroup: 256,
});

describe("OPT-0011 orthogonal VAE runtime profile", () => {
  it("selects only the exact authenticated FP16-256 portable profile", () => {
    const selected = selectAceVaeRuntimeProfile({
      requestedProfile: "opt-0011-mixed-fp16-portable-v1",
      package: CANDIDATE,
      deviceFeatures: ["shader-f16", "subgroups"],
      deviceLimits: DEVICE_LIMITS,
      decoderPlan: planAceVaeDecoder(256),
    });
    expect(selected).toBe(ACE_OPT_0011_VAE_FP16_PORTABLE_PROFILE);
    expect(selected).toMatchObject({
      id: "opt-0011-mixed-fp16-portable-v1",
      packageProfile: "fp16-vae-experimental",
      packageConverterRevision: 5,
      manifestSha256: ACE_OPT_0011_VAE_FP16_MANIFEST_SHA256,
      manifestByteLength: ACE_OPT_0011_VAE_FP16_MANIFEST_BYTES,
      windowFrames: 256,
      batch: 1,
      kernelBackend: "portable-workgroup",
      kernelSetId: ACE_OPT_0011_VAE_FP16_PORTABLE_KERNEL_SET_ID,
      requiredFeatures: ["shader-f16"],
      requiredSubgroupSize: null,
      requiredLimits: {
        maxBufferSize: 125_829_120,
        maxStorageBufferBindingSize: 125_829_120,
        maxComputeWorkgroupStorageSize: 16_384,
        maxComputeInvocationsPerWorkgroup: 256,
      },
      storage: {
        parameters: "float16",
        decoderInput: "float16",
        internalWorkspaces: "float16",
        accumulation: "float32",
        nonlinear: "float32",
        finalOutput: "float32",
        internalStoreRounding:
          "ieee-binary16-round-to-nearest-ties-to-even",
      },
      precisionMapSha256: ACE_OPT_0011_VAE_FP16_PRECISION_MAP_SHA256,
    });
  });

  it("selects the explicit fixed32-K7 hybrid only with authenticated 32/32 bounds", () => {
    const selected = selectAceVaeRuntimeProfile({
      requestedProfile: "opt-0011-mixed-fp16-fixed32-k7-hybrid-v1",
      package: CANDIDATE,
      deviceFeatures: ["shader-f16", "subgroups"],
      subgroupMinSize: 32,
      subgroupMaxSize: 32,
      deviceLimits: DEVICE_LIMITS,
      decoderPlan: planAceVaeDecoder(256),
    });
    expect(selected).toBe(
      ACE_OPT_0011_VAE_FP16_FIXED32_K7_HYBRID_PROFILE,
    );
    expect(selected).toMatchObject({
      id: "opt-0011-mixed-fp16-fixed32-k7-hybrid-v1",
      kernelBackend: "fixed32-subgroup-k7-hybrid",
      kernelSetId:
        ACE_OPT_0011_VAE_FP16_FIXED32_K7_HYBRID_KERNEL_SET_ID,
      requiredFeatures: ["shader-f16", "subgroups"],
      requiredSubgroupSize: 32,
      precisionMapSha256:
        ACE_OPT_0011_VAE_FP16_FIXED32_K7_HYBRID_PRECISION_MAP_SHA256,
    });
    const request = {
      requestedProfile:
        "opt-0011-mixed-fp16-fixed32-k7-hybrid-v1" as const,
      package: CANDIDATE,
      deviceFeatures: ["shader-f16", "subgroups"],
      subgroupMinSize: 32,
      subgroupMaxSize: 32,
      deviceLimits: DEVICE_LIMITS,
      decoderPlan: planAceVaeDecoder(256),
    };
    expect(() => selectAceVaeRuntimeProfile({
      ...request,
      deviceFeatures: ["shader-f16"],
    })).toThrow(/requires subgroups/);
    expect(() => selectAceVaeRuntimeProfile({
      ...request,
      subgroupMinSize: 16,
    })).toThrow(/fixed 32-lane subgroups/);
    expect(() => selectAceVaeRuntimeProfile({
      requestedProfile:
        "opt-0011-mixed-fp16-fixed32-k7-hybrid-v1",
      package: CANDIDATE,
      deviceFeatures: ["shader-f16", "subgroups"],
      subgroupMinSize: 32,
      deviceLimits: DEVICE_LIMITS,
      decoderPlan: planAceVaeDecoder(256),
    })).toThrow(/fixed 32-lane subgroups/);
  });

  it("selects the explicit fixed32-K7/congruent-transpose production profile", () => {
    const request = {
      requestedProfile:
        "opt-0015-mixed-fp16-fixed32-k7-congruent-transpose-v1" as const,
      package: CANDIDATE,
      deviceFeatures: ["shader-f16", "subgroups"],
      subgroupMinSize: 32,
      subgroupMaxSize: 32,
      deviceLimits: DEVICE_LIMITS,
      decoderPlan: planAceVaeDecoder(256),
    };
    const selected = selectAceVaeRuntimeProfile(request);
    expect(selected).toBe(
      ACE_OPT_0015_VAE_FP16_FIXED32_K7_CONGRUENT_TRANSPOSE_PROFILE,
    );
    expect(selected).toMatchObject({
      id: "opt-0015-mixed-fp16-fixed32-k7-congruent-transpose-v1",
      kernelBackend: "fixed32-subgroup-k7-congruent-transpose-hybrid",
      kernelSetId:
        ACE_OPT_0015_VAE_FP16_FIXED32_K7_CONGRUENT_TRANSPOSE_KERNEL_SET_ID,
      requiredFeatures: ["shader-f16", "subgroups"],
      requiredSubgroupSize: 32,
      precisionMapSha256:
        ACE_OPT_0015_VAE_FP16_FIXED32_K7_CONGRUENT_TRANSPOSE_PRECISION_MAP_SHA256,
    });
    expect(() => selectAceVaeRuntimeProfile({
      ...request,
      subgroupMaxSize: 64,
    })).toThrow(/fixed 32-lane subgroups/);
  });

  it("selects the revision-6 OPT-0028 exact-packed production profile", () => {
    const selected = selectAceVaeRuntimeProfile({
      requestedProfile: "opt-0028-mixed-fp16-fixed32-exact-packed-v1",
      package: PACKED_CANDIDATE,
      deviceFeatures: ["shader-f16", "subgroups"],
      subgroupMinSize: 32,
      subgroupMaxSize: 32,
      deviceLimits: DEVICE_LIMITS,
      decoderPlan: planAceVaeDecoder(256),
    });
    expect(selected).toBe(ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_PROFILE);
    expect(selected).toMatchObject({
      packageConverterRevision: 6,
      kernelBackend: "fixed32-subgroup-exact-packed",
      kernelSetId: ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_KERNEL_SET_ID,
      requiredFeatures: ["shader-f16", "subgroups"],
      requiredSubgroupSize: 32,
      precisionMapSha256:
        ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_PRECISION_MAP_SHA256,
    });
  });

  it("selects OPT-0040 only as an explicit fixed32 revision-6 profile", () => {
    const request = {
      requestedProfile:
        "opt-0040-mixed-fp16-fixed32-exact-packed-shape-selected-v1" as const,
      package: PACKED_CANDIDATE,
      deviceFeatures: ["shader-f16", "subgroups"],
      subgroupMinSize: 32,
      subgroupMaxSize: 32,
      deviceLimits: DEVICE_LIMITS,
      decoderPlan: planAceVaeDecoder(256),
    };
    const selected = selectAceVaeRuntimeProfile(request);
    expect(selected).toBe(
      ACE_OPT_0040_VAE_FP16_FIXED32_SHAPE_SELECTED_PROFILE,
    );
    expect(selected).toMatchObject({
      id: "opt-0040-mixed-fp16-fixed32-exact-packed-shape-selected-v1",
      packageConverterRevision: 6,
      manifestSha256: ACE_OPT_0028_VAE_FP16_MANIFEST_SHA256,
      kernelBackend: "fixed32-subgroup-exact-packed-shape-selected",
      kernelSetId:
        ACE_OPT_0040_VAE_FP16_FIXED32_SHAPE_SELECTED_KERNEL_SET_ID,
      requiredFeatures: ["shader-f16", "subgroups"],
      requiredSubgroupSize: 32,
      precisionMapSha256:
        ACE_OPT_0040_VAE_FP16_FIXED32_SHAPE_SELECTED_PRECISION_MAP_SHA256,
    });
    expect(() => selectAceVaeRuntimeProfile({
      ...request,
      requestedProfile: "opt-0028-mixed-fp16-fixed32-exact-packed-v1",
    })).not.toThrow();
    expect(() => selectAceVaeRuntimeProfile({
      ...request,
      subgroupMaxSize: 64,
    })).toThrow(/fixed 32-lane subgroups/);
  });

  it("selects revision-6 portable exact-packed without subgroup claims", () => {
    const selected = selectAceVaeRuntimeProfile({
      requestedProfile: "opt-0028-mixed-fp16-portable-exact-packed-v1",
      package: PACKED_CANDIDATE,
      deviceFeatures: ["shader-f16"],
      deviceLimits: DEVICE_LIMITS,
      decoderPlan: planAceVaeDecoder(256),
    });
    expect(selected).toBe(ACE_OPT_0028_VAE_FP16_PORTABLE_EXACT_PACKED_PROFILE);
    expect(selected).toMatchObject({
      packageConverterRevision: 6,
      kernelBackend: "portable-workgroup-exact-packed",
      kernelSetId:
        ACE_OPT_0028_VAE_FP16_PORTABLE_EXACT_PACKED_KERNEL_SET_ID,
      requiredFeatures: ["shader-f16"],
      requiredSubgroupSize: null,
      precisionMapSha256:
        ACE_OPT_0028_VAE_FP16_PORTABLE_EXACT_PACKED_PRECISION_MAP_SHA256,
    });
    expect(() => selectAceVaeRuntimeProfile({
      requestedProfile: "opt-0028-mixed-fp16-portable-exact-packed-v1",
      package: PACKED_CANDIDATE,
      deviceFeatures: [],
      deviceLimits: DEVICE_LIMITS,
      decoderPlan: planAceVaeDecoder(256),
    })).toThrow(/requires shader-f16/);
  });

  it("selects revision-7 portable dual-K4 without subgroup claims", () => {
    const selected = selectAceVaeRuntimeProfile({
      requestedProfile: "opt-0088-mixed-fp16-portable-dual-k4-v1",
      package: REVISION7_CANDIDATE,
      deviceFeatures: ["shader-f16"],
      deviceLimits: DEVICE_LIMITS,
      decoderPlan: planAceVaeDecoder(256),
    });
    expect(selected).toBe(ACE_OPT_0088_VAE_FP16_PORTABLE_DUAL_K4_PROFILE);
    expect(selected).toMatchObject({
      packageConverterRevision: 7,
      manifestSha256: ACE_OPT_0054_VAE_REVISION7_MANIFEST_SHA256,
      kernelBackend: "portable-workgroup-dual-k4",
      kernelSetId: ACE_OPT_0088_VAE_FP16_PORTABLE_DUAL_K4_KERNEL_SET_ID,
      requiredFeatures: ["shader-f16"],
      requiredSubgroupSize: null,
      precisionMapSha256:
        ACE_OPT_0088_VAE_FP16_PORTABLE_DUAL_K4_PRECISION_MAP_SHA256,
    });
    expect(selected.requiredFeatures).not.toContain("subgroups");
    expect(() => selectAceVaeRuntimeProfile({
      requestedProfile: "opt-0088-mixed-fp16-portable-dual-k4-v1",
      package: REVISION7_CANDIDATE,
      deviceFeatures: [],
      deviceLimits: DEVICE_LIMITS,
      decoderPlan: planAceVaeDecoder(256),
    })).toThrow(/requires shader-f16/);
  });

  it("keeps the FP32 oracle orthogonal to existing global profiles", () => {
    expect(ACE_VAE_RUNTIME_PROFILE_IDS).toEqual([
      "fp32-oracle",
      "opt-0011-mixed-fp16-portable-v1",
      "opt-0028-mixed-fp16-portable-exact-packed-v1",
      "opt-0011-mixed-fp16-fixed32-k7-hybrid-v1",
      "opt-0015-mixed-fp16-fixed32-k7-congruent-transpose-v1",
      "opt-0028-mixed-fp16-fixed32-exact-packed-v1",
      "opt-0040-mixed-fp16-fixed32-exact-packed-shape-selected-v1",
      "opt-0054-mixed-fp16-fixed32-revision7-v1",
      "opt-0066-mixed-fp16-fixed32-dual-k4-quality-v1",
      "opt-0088-mixed-fp16-portable-dual-k4-v1",
    ]);
    expect(ACE_MODEL_PROFILE_IDS).toEqual(["reference-bf16", "raw-fp16"]);
    const selected = selectAceVaeRuntimeProfile({
      requestedProfile: "fp32-oracle",
      package: ORACLE,
      deviceFeatures: [],
      deviceLimits: DEVICE_LIMITS,
      decoderPlan: planAceVaeDecoder(256),
    });
    expect(selected).toBe(ACE_VAE_FP32_ORACLE_PROFILE);
    expect(selected.storage).toEqual({
      parameters: "float32",
      decoderInput: "float32",
      internalWorkspaces: "float32",
      accumulation: "float32",
      nonlinear: "float32",
      finalOutput: "float32",
      internalStoreRounding: "none",
    });
  });

  it("deep-freezes selector-governing feature and profile ID arrays", () => {
    expect(Object.isFrozen(ACE_VAE_RUNTIME_PROFILE_IDS)).toBe(true);
    expect(Object.isFrozen(
      ACE_OPT_0011_VAE_FP16_PORTABLE_PROFILE.requiredFeatures,
    )).toBe(true);
    expect(Object.isFrozen(
      ACE_OPT_0028_VAE_FP16_PORTABLE_EXACT_PACKED_PROFILE.requiredFeatures,
    )).toBe(true);
    expect(Object.isFrozen(
      ACE_OPT_0011_VAE_FP16_FIXED32_K7_HYBRID_PROFILE.requiredFeatures,
    )).toBe(true);
    expect(Object.isFrozen(
      ACE_OPT_0015_VAE_FP16_FIXED32_K7_CONGRUENT_TRANSPOSE_PROFILE.requiredFeatures,
    )).toBe(true);
    expect(Object.isFrozen(
      ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_PROFILE.requiredFeatures,
    )).toBe(true);
    expect(Object.isFrozen(
      ACE_OPT_0040_VAE_FP16_FIXED32_SHAPE_SELECTED_PROFILE.requiredFeatures,
    )).toBe(true);
    expect(Object.isFrozen(ACE_VAE_FP32_ORACLE_PROFILE.requiredFeatures))
      .toBe(true);
    const mutableFeatures =
      ACE_OPT_0011_VAE_FP16_PORTABLE_PROFILE.requiredFeatures as unknown as
        string[];
    expect(() => mutableFeatures.splice(0)).toThrow(TypeError);
    expect(ACE_OPT_0011_VAE_FP16_PORTABLE_PROFILE.requiredFeatures).toEqual([
      "shader-f16",
    ]);
    expect(() => selectAceVaeRuntimeProfile({
      requestedProfile: "opt-0011-mixed-fp16-portable-v1",
      package: CANDIDATE,
      deviceFeatures: [],
      deviceLimits: DEVICE_LIMITS,
      decoderPlan: planAceVaeDecoder(256),
    })).toThrow(/requires shader-f16/);
  });

  it("fails closed instead of changing precision, package, or geometry", () => {
    const candidateRequest = {
      requestedProfile: "opt-0011-mixed-fp16-portable-v1" as const,
      package: CANDIDATE,
      deviceFeatures: ["shader-f16"],
      deviceLimits: DEVICE_LIMITS,
      decoderPlan: planAceVaeDecoder(256),
    };
    expect(() => selectAceVaeRuntimeProfile({
      ...candidateRequest,
      deviceFeatures: [],
    })).toThrow(/requires shader-f16/);
    expect(() => selectAceVaeRuntimeProfile({
      ...candidateRequest,
      deviceLimits: {
        ...DEVICE_LIMITS,
        maxBufferSize: 125_829_119,
      },
    })).toThrow(/maxBufferSize >= 125829120/);
    expect(() => selectAceVaeRuntimeProfile({
      ...candidateRequest,
      deviceLimits: {
        ...DEVICE_LIMITS,
        maxStorageBufferBindingSize: 125_829_119,
      },
    })).toThrow(/maxStorageBufferBindingSize >= 125829120/);
    expect(() => selectAceVaeRuntimeProfile({
      ...candidateRequest,
      deviceLimits: {
        ...DEVICE_LIMITS,
        maxComputeWorkgroupStorageSize: 16_383,
      },
    })).toThrow(/maxComputeWorkgroupStorageSize >= 16384/);
    expect(() => selectAceVaeRuntimeProfile({
      ...candidateRequest,
      deviceLimits: {
        ...DEVICE_LIMITS,
        maxComputeInvocationsPerWorkgroup: 255,
      },
    })).toThrow(/maxComputeInvocationsPerWorkgroup >= 256/);
    expect(() => selectAceVaeRuntimeProfile({
      ...candidateRequest,
      package: ORACLE,
    })).toThrow(/exact authenticated package identity/);
    expect(() => selectAceVaeRuntimeProfile({
      ...candidateRequest,
      package: {
        ...CANDIDATE,
        manifestSha256: "0".repeat(64),
      },
    })).toThrow(/exact authenticated package identity/);
    expect(() => selectAceVaeRuntimeProfile({
      ...candidateRequest,
      decoderPlan: planAceVaeDecoder(512),
    })).toThrow(/exact batch-1 256-frame decoder graph/);
    expect(() => selectAceVaeRuntimeProfile({
      ...candidateRequest,
      requestedProfile: "unknown" as AceVaeRuntimeProfileId,
    })).toThrow(/Unknown ACE VAE runtime profile/);
  });
});

describe("OPT-0011 frozen FP16 VAE precision map", () => {
  it("pins all 88 graph operations plus the explicit ingress boundary", () => {
    const map = ACE_OPT_0011_VAE_FP16_PRECISION_MAP;
    expect(map).toMatchObject({
      schema: "ace-opt-0011-vae-precision-map-v1",
      profileId: "opt-0011-mixed-fp16-portable-v1",
      kernelSetId: ACE_OPT_0011_VAE_FP16_PORTABLE_KERNEL_SET_ID,
      decoderConfigId: "ace-step-1.5-oobleck-decoder-v1",
      batch: 1,
      inputFrames: 256,
      graphOperationCount: 88,
    });
    expect(map.entries).toHaveLength(89);
    expect(map.entries[0]).toEqual(expect.objectContaining({
      sequenceIndex: 0,
      graphOperationIndex: null,
      kind: "ingress-cast",
      kernelFamily: "f32-to-f16-cast",
      inputs: [{
        role: "input",
        slot: "ingress-staging",
        storage: "float32",
      }],
      output: {
        slot: "input",
        storage: "float16",
        rounding: "ieee-binary16-round-to-nearest-ties-to-even",
      },
    }));
    expect(map.entries.slice(1).map((entry) => entry.graphOperationIndex))
      .toEqual(Array.from({ length: 88 }, (_, index) => index));
    expect(countBy(map.entries, (entry) => entry.kernelFamily)).toEqual({
      "f32-to-f16-cast": 1,
      "conv1d-k7": 17,
      snake: 36,
      "conv-transpose1d": 5,
      "conv1d-k1": 15,
      add: 15,
    });
  });

  it("freezes FP16 boundaries, FP32 islands, ordering, and contraction", () => {
    const graphEntries = ACE_OPT_0011_VAE_FP16_PRECISION_MAP.entries.slice(1);
    expect(graphEntries.every((entry) =>
      entry.inputs.every((input) => input.storage === "float16") &&
      entry.parameters.every((parameter) => parameter.storage === "float16") &&
      entry.registerArithmetic === "float32"
    )).toBe(true);
    expect(graphEntries.flatMap((entry) => entry.parameters)).toHaveLength(145);
    expect(graphEntries.filter((entry) => entry.accumulation === "float32"))
      .toHaveLength(37);
    expect(graphEntries.filter((entry) =>
      entry.nonlinearArithmetic === "float32"
    )).toHaveLength(36);
    expect(graphEntries.filter((entry) =>
      entry.output.storage === "float32"
    )).toEqual([
      expect.objectContaining({
        graphOperationIndex: 87,
        label: "conv2",
        kernelFamily: "conv1d-k7",
      }),
    ]);
    expect(graphEntries.slice(0, -1).every((entry) =>
      entry.output.storage === "float16" &&
      entry.output.rounding ===
        "ieee-binary16-round-to-nearest-ties-to-even"
    )).toBe(true);
    const snake = graphEntries.find((entry) => entry.kind === "snake")!;
    expect(snake).toMatchObject({
      nonlinearArithmetic: "float32",
      contraction: "wgsl-multiply-add-contraction-permitted",
      transcendentals: [
        "exp-alpha",
        "exp-beta",
        "sin-alpha-times-input",
      ],
    });
  });

  it("pins a stable SHA-256 over the canonical machine-readable map", () => {
    const independent = createHash("sha256")
      .update(ACE_OPT_0011_VAE_FP16_PRECISION_MAP_CANONICAL_JSON)
      .digest("hex");
    expect(hashAceVaePrecisionMap(ACE_OPT_0011_VAE_FP16_PRECISION_MAP)).toBe(
      ACE_OPT_0011_VAE_FP16_PRECISION_MAP_SHA256,
    );
    expect(independent).toBe(ACE_OPT_0011_VAE_FP16_PRECISION_MAP_SHA256);
  });

  it("gives the hybrid kernel set its own truthful map identity", () => {
    const map = ACE_OPT_0011_VAE_FP16_FIXED32_K7_HYBRID_PRECISION_MAP;
    expect(map).toMatchObject({
      profileId: "opt-0011-mixed-fp16-fixed32-k7-hybrid-v1",
      kernelSetId:
        ACE_OPT_0011_VAE_FP16_FIXED32_K7_HYBRID_KERNEL_SET_ID,
      entries: ACE_OPT_0011_VAE_FP16_PRECISION_MAP.entries,
    });
    expect(JSON.stringify(map)).toBe(
      ACE_OPT_0011_VAE_FP16_FIXED32_K7_HYBRID_PRECISION_MAP_CANONICAL_JSON,
    );
    expect(hashAceVaePrecisionMap(map)).toBe(
      ACE_OPT_0011_VAE_FP16_FIXED32_K7_HYBRID_PRECISION_MAP_SHA256,
    );
  });

  it("gives portable exact-packed its own truthful map identity", () => {
    const map = ACE_OPT_0028_VAE_FP16_PORTABLE_EXACT_PACKED_PRECISION_MAP;
    expect(map).toMatchObject({
      profileId: "opt-0028-mixed-fp16-portable-exact-packed-v1",
      kernelSetId:
        ACE_OPT_0028_VAE_FP16_PORTABLE_EXACT_PACKED_KERNEL_SET_ID,
      entries: ACE_OPT_0011_VAE_FP16_PRECISION_MAP.entries,
    });
    expect(JSON.stringify(map)).toBe(
      ACE_OPT_0028_VAE_FP16_PORTABLE_EXACT_PACKED_PRECISION_MAP_CANONICAL_JSON,
    );
    expect(hashAceVaePrecisionMap(map)).toBe(
      ACE_OPT_0028_VAE_FP16_PORTABLE_EXACT_PACKED_PRECISION_MAP_SHA256,
    );
  });

  it("gives the congruent-transpose kernel set its own truthful map identity", () => {
    const map =
      ACE_OPT_0015_VAE_FP16_FIXED32_K7_CONGRUENT_TRANSPOSE_PRECISION_MAP;
    expect(map).toMatchObject({
      profileId:
        "opt-0015-mixed-fp16-fixed32-k7-congruent-transpose-v1",
      kernelSetId:
        ACE_OPT_0015_VAE_FP16_FIXED32_K7_CONGRUENT_TRANSPOSE_KERNEL_SET_ID,
      entries: ACE_OPT_0011_VAE_FP16_PRECISION_MAP.entries,
    });
    expect(JSON.stringify(map)).toBe(
      ACE_OPT_0015_VAE_FP16_FIXED32_K7_CONGRUENT_TRANSPOSE_PRECISION_MAP_CANONICAL_JSON,
    );
    expect(hashAceVaePrecisionMap(map)).toBe(
      ACE_OPT_0015_VAE_FP16_FIXED32_K7_CONGRUENT_TRANSPOSE_PRECISION_MAP_SHA256,
    );
  });

  it("gives the OPT-0028 exact-packed kernel set its own truthful map identity", () => {
    const map = ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_PRECISION_MAP;
    expect(map).toMatchObject({
      profileId: "opt-0028-mixed-fp16-fixed32-exact-packed-v1",
      kernelSetId: ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_KERNEL_SET_ID,
      entries: ACE_OPT_0011_VAE_FP16_PRECISION_MAP.entries,
    });
    expect(JSON.stringify(map)).toBe(
      ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_PRECISION_MAP_CANONICAL_JSON,
    );
    expect(hashAceVaePrecisionMap(map)).toBe(
      ACE_OPT_0028_VAE_FP16_FIXED32_EXACT_PACKED_PRECISION_MAP_SHA256,
    );
  });

  it("gives the OPT-0040 shape-selected kernel set its own truthful map identity", () => {
    const map = ACE_OPT_0040_VAE_FP16_FIXED32_SHAPE_SELECTED_PRECISION_MAP;
    expect(map).toMatchObject({
      profileId:
        "opt-0040-mixed-fp16-fixed32-exact-packed-shape-selected-v1",
      kernelSetId:
        ACE_OPT_0040_VAE_FP16_FIXED32_SHAPE_SELECTED_KERNEL_SET_ID,
      entries: ACE_OPT_0011_VAE_FP16_PRECISION_MAP.entries,
    });
    expect(JSON.stringify(map)).toBe(
      ACE_OPT_0040_VAE_FP16_FIXED32_SHAPE_SELECTED_PRECISION_MAP_CANONICAL_JSON,
    );
    expect(hashAceVaePrecisionMap(map)).toBe(
      ACE_OPT_0040_VAE_FP16_FIXED32_SHAPE_SELECTED_PRECISION_MAP_SHA256,
    );
  });
});

function packageIdentity(
  profile: "reference" | "fp16-vae-experimental",
  converterRevision: number,
  manifestSha256: string,
  manifestByteLength: number,
): AceVaeAuthenticatedPackageIdentity {
  return Object.freeze({
    manifest: {
      profile,
      provenance: { converterRevision },
    } as AcePackageManifest,
    manifestSha256,
    manifestByteLength,
  });
}

function countBy<T>(
  values: readonly T[],
  key: (value: T) => string,
): Record<string, number> {
  const counts: Record<string, number> = {};
  for (const value of values) {
    const name = key(value);
    counts[name] = (counts[name] ?? 0) + 1;
  }
  return counts;
}
