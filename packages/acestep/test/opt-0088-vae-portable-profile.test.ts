import { createHash } from "node:crypto";

import { describe, expect, it } from "vitest";

import {
  ACE_OPT_0054_VAE_REVISION7_MANIFEST_BYTES,
  ACE_OPT_0054_VAE_REVISION7_MANIFEST_SHA256,
} from "../src/model/package.js";
import { planAceVaeDecoder } from "../src/webgpu/vae-decoder.js";
import {
  ACE_OPT_0088_VAE_K7_PORTABLE_SHAPE_SELECTOR_KERNEL_ID,
} from "../src/webgpu/kernels/vae-conv1d-fp16-k4-row-reuse-shape-selector.js";
import {
  ACE_OPT_0088_VAE_CONV_TRANSPOSE1D_K4_PORTABLE_SHAPE_SELECTOR_KERNEL_ID,
} from
  "../src/webgpu/kernels/vae-conv-transpose1d-fp16-k4-shape-selector.js";
import {
  ACE_OPT_0028_VAE_K1_PORTABLE_PACKED_KERNEL_ID,
} from "../src/webgpu/kernels/vae-k1-fp16-portable-packed.js";
import {
  ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PRECISION_MAP,
  ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PROFILE,
  ACE_OPT_0072_VAE_FP16_FIXED32_DUAL_K4_PRODUCTION_PROFILE_CONTRACT,
  ACE_OPT_0072_VAE_FP16_FIXED32_DUAL_K4_PRODUCTION_RUNTIME_PROFILE,
  ACE_OPT_0088_VAE_FP16_PORTABLE_DUAL_K4_KERNEL_SET_ID,
  ACE_OPT_0088_VAE_FP16_PORTABLE_DUAL_K4_PRECISION_MAP,
  ACE_OPT_0088_VAE_FP16_PORTABLE_DUAL_K4_PRECISION_MAP_CANONICAL_JSON,
  ACE_OPT_0088_VAE_FP16_PORTABLE_DUAL_K4_PRECISION_MAP_SHA256,
  ACE_OPT_0088_VAE_FP16_PORTABLE_DUAL_K4_PROFILE,
  ACE_OPT_0088_VAE_PORTABLE_PRODUCTION_PROFILE_CONTRACT,
  ACE_VAE_RUNTIME_PROFILE_IDS,
  hashAceVaePrecisionMap,
  requireAceOpt0072VaeProductionRuntimeProfile,
  requireAceOpt0072VaeProductionRuntimeProfileForBackend,
  selectAceVaeRuntimeProfile,
} from "../src/webgpu/vae-fp16-profile.js";
import {
  ACE_OPT_0088_VAE_FP16_PORTABLE_DUAL_K4_KERNEL_TOPOLOGY,
} from "../src/webgpu/vae-fp16-decoder.js";

const REVISION7_PACKAGE = Object.freeze({
  manifest: {
    profile: "fp16-vae-experimental",
    provenance: { converterRevision: 7 },
  },
  manifestSha256: ACE_OPT_0054_VAE_REVISION7_MANIFEST_SHA256,
  manifestByteLength: ACE_OPT_0054_VAE_REVISION7_MANIFEST_BYTES,
}) as Parameters<typeof selectAceVaeRuntimeProfile>[0]["package"];

describe("OPT-0088 portable dual-K4 runtime profile", () => {
  it("freezes the portable precision-map identity", () => {
    const independent = createHash("sha256")
      .update(ACE_OPT_0088_VAE_FP16_PORTABLE_DUAL_K4_PRECISION_MAP_CANONICAL_JSON)
      .digest("hex");
    expect(hashAceVaePrecisionMap(
      ACE_OPT_0088_VAE_FP16_PORTABLE_DUAL_K4_PRECISION_MAP,
    )).toBe(ACE_OPT_0088_VAE_FP16_PORTABLE_DUAL_K4_PRECISION_MAP_SHA256);
    expect(independent).toBe(
      ACE_OPT_0088_VAE_FP16_PORTABLE_DUAL_K4_PRECISION_MAP_SHA256,
    );
  });

  it("matches the OPT-0066 arithmetic entries modulo profile identity", () => {
    const map = ACE_OPT_0088_VAE_FP16_PORTABLE_DUAL_K4_PRECISION_MAP;
    expect(map).toMatchObject({
      profileId: "opt-0088-mixed-fp16-portable-dual-k4-v1",
      kernelSetId: ACE_OPT_0088_VAE_FP16_PORTABLE_DUAL_K4_KERNEL_SET_ID,
    });
    // Bit-identical arithmetic claim: every entry (storage boundaries,
    // reduction order, contraction allowance) is exactly OPT-0066's.
    expect(map.entries).toEqual(
      ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PRECISION_MAP.entries,
    );
    expect({
      ...map,
      profileId:
        ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PRECISION_MAP.profileId,
      kernelSetId:
        ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PRECISION_MAP.kernelSetId,
    }).toEqual(ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PRECISION_MAP);
  });

  it("pins the portable profile fields to the rev7 OPT-0066 envelope", () => {
    const profile = ACE_OPT_0088_VAE_FP16_PORTABLE_DUAL_K4_PROFILE;
    const quality = ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PROFILE;
    expect(ACE_VAE_RUNTIME_PROFILE_IDS).toContain(profile.id);
    expect(profile).toMatchObject({
      id: "opt-0088-mixed-fp16-portable-dual-k4-v1",
      packageProfile: quality.packageProfile,
      packageConverterRevision: quality.packageConverterRevision,
      manifestSha256: quality.manifestSha256,
      manifestByteLength: quality.manifestByteLength,
      windowFrames: quality.windowFrames,
      batch: 1,
      kernelBackend: "portable-workgroup-dual-k4",
      kernelSetId: ACE_OPT_0088_VAE_FP16_PORTABLE_DUAL_K4_KERNEL_SET_ID,
      requiredFeatures: ["shader-f16"],
      requiredSubgroupSize: null,
      requiredLimits: quality.requiredLimits,
      storage: quality.storage,
      precisionMapSha256:
        ACE_OPT_0088_VAE_FP16_PORTABLE_DUAL_K4_PRECISION_MAP_SHA256,
    });
    expect(profile.requiredFeatures).not.toContain("subgroups");
    expect(profile.precisionMapSha256).not.toBe(quality.precisionMapSha256);
    expect(ACE_OPT_0088_VAE_FP16_PORTABLE_DUAL_K4_KERNEL_TOPOLOGY)
      .toMatchObject({
        id: profile.kernelSetId,
        backend: "portable-workgroup-dual-k4",
        conv1dK1: ACE_OPT_0028_VAE_K1_PORTABLE_PACKED_KERNEL_ID,
        conv1dK7: ACE_OPT_0088_VAE_K7_PORTABLE_SHAPE_SELECTOR_KERNEL_ID,
        convTranspose1d:
          ACE_OPT_0088_VAE_CONV_TRANSPOSE1D_K4_PORTABLE_SHAPE_SELECTOR_KERNEL_ID,
      });
  });

  it("selects with shader-f16 only and never requires subgroups", () => {
    const profile = ACE_OPT_0088_VAE_FP16_PORTABLE_DUAL_K4_PROFILE;
    expect(selectAceVaeRuntimeProfile({
      requestedProfile: profile.id,
      package: REVISION7_PACKAGE,
      deviceFeatures: ["shader-f16"],
      deviceLimits: profile.requiredLimits,
      decoderPlan: planAceVaeDecoder(256),
    })).toBe(profile);
    expect(() => selectAceVaeRuntimeProfile({
      requestedProfile: profile.id,
      package: REVISION7_PACKAGE,
      deviceFeatures: [],
      deviceLimits: profile.requiredLimits,
      decoderPlan: planAceVaeDecoder(256),
    })).toThrow(/requires shader-f16/);
    expect(() => selectAceVaeRuntimeProfile({
      requestedProfile: profile.id,
      package: {
        ...REVISION7_PACKAGE,
        manifestSha256: "0".repeat(64),
      },
      deviceFeatures: ["shader-f16"],
      deviceLimits: profile.requiredLimits,
      decoderPlan: planAceVaeDecoder(256),
    })).toThrow(/exact authenticated package identity/);
  });

  it("maps the public OPT-0072 identity per backend without weakening fixed32", () => {
    const portable = requireAceOpt0072VaeProductionRuntimeProfileForBackend(
      ACE_OPT_0072_VAE_FP16_FIXED32_DUAL_K4_PRODUCTION_RUNTIME_PROFILE,
      "portable",
    );
    expect(portable).toBe(ACE_OPT_0088_VAE_PORTABLE_PRODUCTION_PROFILE_CONTRACT);
    expect(Object.isFrozen(portable)).toBe(true);
    expect(portable).toEqual({
      id: ACE_OPT_0072_VAE_FP16_FIXED32_DUAL_K4_PRODUCTION_RUNTIME_PROFILE,
      physicalRuntimeProfileId:
        ACE_OPT_0088_VAE_FP16_PORTABLE_DUAL_K4_PROFILE.id,
      manifestSha256: ACE_OPT_0054_VAE_REVISION7_MANIFEST_SHA256,
      manifestByteLength: ACE_OPT_0054_VAE_REVISION7_MANIFEST_BYTES,
      kernelSetId: ACE_OPT_0088_VAE_FP16_PORTABLE_DUAL_K4_KERNEL_SET_ID,
      precisionMapSha256:
        ACE_OPT_0088_VAE_FP16_PORTABLE_DUAL_K4_PRECISION_MAP_SHA256,
    });
    expect(ACE_VAE_RUNTIME_PROFILE_IDS).not.toContain(portable.id);

    const subgroups = requireAceOpt0072VaeProductionRuntimeProfileForBackend(
      ACE_OPT_0072_VAE_FP16_FIXED32_DUAL_K4_PRODUCTION_RUNTIME_PROFILE,
      "subgroups",
    );
    expect(subgroups).toBe(
      ACE_OPT_0072_VAE_FP16_FIXED32_DUAL_K4_PRODUCTION_PROFILE_CONTRACT,
    );
    expect(subgroups).toBe(requireAceOpt0072VaeProductionRuntimeProfile(
      ACE_OPT_0072_VAE_FP16_FIXED32_DUAL_K4_PRODUCTION_RUNTIME_PROFILE,
    ));

    expect(() => requireAceOpt0072VaeProductionRuntimeProfileForBackend(
      ACE_OPT_0088_VAE_FP16_PORTABLE_DUAL_K4_PROFILE.id,
      "portable",
    )).toThrow(/not authenticated/);
    expect(() => requireAceOpt0072VaeProductionRuntimeProfileForBackend(
      ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PROFILE.id,
      "subgroups",
    )).toThrow(/not authenticated/);
  });
});
