import { readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

import {
  ACE_OPT_0054_VAE_FP16_FIXED32_REVISION7_PROFILE,
  ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PRECISION_MAP,
  ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PRECISION_MAP_SHA256,
  ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PROFILE,
  hashAceVaePrecisionMap,
  selectAceVaeRuntimeProfile,
} from "../src/webgpu/vae-fp16-profile.js";
import { planAceVaeDecoder } from "../src/webgpu/vae-decoder.js";
import {
  ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_KERNEL_TOPOLOGY,
} from "../src/webgpu/vae-fp16-decoder.js";

const PROTOCOL_SOURCE = readFileSync(new URL(
  "../src/runtime/protocol.ts",
  import.meta.url,
), "utf8");

describe("OPT-0066 dual-K4 diagnostic runtime profile", () => {
  it("is a distinct hash-bound profile over only the authenticated rev7 bytes", () => {
    const profile = ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PROFILE;
    expect(profile.id).toBe(
      "opt-0066-mixed-fp16-fixed32-dual-k4-quality-v1",
    );
    expect(profile.id).not.toBe(
      ACE_OPT_0054_VAE_FP16_FIXED32_REVISION7_PROFILE.id,
    );
    expect(profile.kernelSetId).not.toBe(
      ACE_OPT_0054_VAE_FP16_FIXED32_REVISION7_PROFILE.kernelSetId,
    );
    expect(
      ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_KERNEL_TOPOLOGY,
    ).toMatchObject({
      id: profile.kernelSetId,
      backend: "fixed32-subgroup-dual-k4-quality",
    });
    expect(profile.manifestSha256).toBe(
      "36a54d79777d6826088095ba6ebc028fb4bea546368c0f0a29cd0eee8d656da7",
    );
    expect(profile.manifestByteLength).toBe(716_185);
    expect(hashAceVaePrecisionMap(
      ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PRECISION_MAP,
    )).toBe(ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PRECISION_MAP_SHA256);
  });

  it("declares both selected K7 and selected ConvTranspose as K4 arithmetic", () => {
    const entries =
      ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PRECISION_MAP.entries;
    const k4 = entries.filter((entry) =>
      entry.registerArithmetic ===
        "float16-dot4-partials-then-float32-running-state"
    );
    expect(k4.filter((entry) => entry.kernelFamily === "conv1d-k7"))
      .toHaveLength(12);
    expect(k4.filter((entry) => entry.kernelFamily === "conv-transpose1d"))
      .toHaveLength(4);
    expect(k4.every((entry) =>
      entry.contraction === "wgsl-f16-dot4-partials-then-f32-add"
    )).toBe(true);
  });

  it("selects fail-closed and is absent from the product protocol", () => {
    const profile = ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PROFILE;
    const packageIdentity = {
      manifest: {
        profile: "fp16-vae-experimental",
        provenance: { converterRevision: 7 },
      },
      manifestSha256: profile.manifestSha256,
      manifestByteLength: profile.manifestByteLength,
    } as Parameters<typeof selectAceVaeRuntimeProfile>[0]["package"];
    expect(selectAceVaeRuntimeProfile({
      requestedProfile: profile.id,
      package: packageIdentity,
      deviceFeatures: ["shader-f16", "subgroups"],
      subgroupMinSize: 32,
      subgroupMaxSize: 32,
      deviceLimits: profile.requiredLimits,
      decoderPlan: planAceVaeDecoder(256),
    })).toBe(profile);
    expect(() => selectAceVaeRuntimeProfile({
      requestedProfile: profile.id,
      package: {
        ...packageIdentity,
        manifestSha256:
          ACE_OPT_0054_VAE_FP16_FIXED32_REVISION7_PROFILE.manifestSha256
            .replace(/^./u, "0"),
      },
      deviceFeatures: ["shader-f16", "subgroups"],
      subgroupMinSize: 32,
      subgroupMaxSize: 32,
      deviceLimits: profile.requiredLimits,
      decoderPlan: planAceVaeDecoder(256),
    })).toThrow(/exact authenticated package identity/);
    expect(PROTOCOL_SOURCE).not.toContain(profile.id);
  });
});
