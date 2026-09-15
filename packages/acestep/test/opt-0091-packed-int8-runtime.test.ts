import { describe, expect, it } from "vitest";

import { isAceClientMessage } from "../src/runtime/protocol.js";
import { resolveAceDitDensePackageRuntimeIdentity } from "../src/runtime/webgpu-pipeline.js";
import { resolveAceDitMixedGemmSelection } from "../src/webgpu/dit-backend.js";
import {
  ACE_OPT_0091_DIT_INT8_KERNEL_SET_ID,
  ACE_OPT_0091_DIT_INT8_LAYER_BYTES,
  ACE_OPT_0091_DIT_INT8_MANIFEST_BYTES,
  ACE_OPT_0091_DIT_INT8_MANIFEST_SHA256,
  ACE_OPT_0091_DIT_INT8_PORTABLE_KERNEL_SET_ID,
  ACE_OPT_0091_DIT_INT8_RESIDENT_WEIGHT_BYTES,
  ACE_OPT_0091_DIT_INT8_RUNTIME_PROFILE,
} from "../src/webgpu/dit-fp16-package.js";
import {
  ACE_OPT_0091_DENSE_INT8_TILE_BYTES,
  ACE_OPT_0091_DENSE_INT8_TILE_WORDS,
  aceOpt0091DenseInt8Wgsl,
} from "../src/webgpu/kernels/dit-dense-int8-weight-only.js";
import {
  ACE_REFERENCE_PORTABLE_PROFILE,
  ACE_REFERENCE_SUBGROUP_PROFILE,
} from "../src/webgpu/capabilities.js";
import { testInitializeMessage } from "./runtime-fixtures.js";

describe("OPT-0091 packed INT8 runtime", () => {
  const packedPackage = {
    manifestUrl: "https://example.test/dit-int8-packed/manifest.json",
    manifestSha256: ACE_OPT_0091_DIT_INT8_MANIFEST_SHA256,
    runtimeProfile: ACE_OPT_0091_DIT_INT8_RUNTIME_PROFILE,
  } as const;

  it("pins the compressed package identity in the public protocol", () => {
    const base = testInitializeMessage();
    const message = {
      ...base,
      configuration: { ...base.configuration, ditDensePackage: packedPackage },
    };
    expect(isAceClientMessage(message)).toBe(true);
    expect(resolveAceDitDensePackageRuntimeIdentity(packedPackage)).toEqual({
      role: "opt-0091-packed-int8-preview",
      manifestSha256: ACE_OPT_0091_DIT_INT8_MANIFEST_SHA256,
      manifestByteLength: ACE_OPT_0091_DIT_INT8_MANIFEST_BYTES,
      runtimeProfile: ACE_OPT_0091_DIT_INT8_RUNTIME_PROFILE,
      kernelSetId: ACE_OPT_0091_DIT_INT8_KERNEL_SET_ID,
      layerBytes: ACE_OPT_0091_DIT_INT8_LAYER_BYTES,
      residentWeightBytes: ACE_OPT_0091_DIT_INT8_RESIDENT_WEIGHT_BYTES,
    });
    expect(
      resolveAceDitDensePackageRuntimeIdentity(packedPackage, "portable"),
    ).toMatchObject({
      kernelSetId: ACE_OPT_0091_DIT_INT8_PORTABLE_KERNEL_SET_ID,
    });
  });

  it("selects packed kernels for subgroup and portable adapters", () => {
    expect(
      resolveAceDitMixedGemmSelection(
        ACE_REFERENCE_SUBGROUP_PROFILE,
        32,
        32,
        ACE_OPT_0091_DIT_INT8_RUNTIME_PROFILE,
        "opt-0070-fixed32-quad-query32-full-self-production-v1",
        2_250,
        98,
      ),
    ).toMatchObject({
      backend: "mixed-opt-0091",
      denseGemmConfiguration: { backend: "opt-0091-int8-weight-only" },
    });
    expect(
      resolveAceDitMixedGemmSelection(
        ACE_REFERENCE_PORTABLE_PROFILE,
        undefined,
        undefined,
        ACE_OPT_0091_DIT_INT8_RUNTIME_PROFILE,
        "opt-0070-fixed32-quad-query32-full-self-production-v1",
        2_250,
        98,
      ),
    ).toMatchObject({
      backend: "mixed-opt-0091-portable",
      denseGemmConfiguration: { backend: "opt-0091-int8-weight-only-portable" },
    });
  });

  it("keeps INT8 weights resident and FP32 accumulation in both shaders", () => {
    expect(ACE_OPT_0091_DENSE_INT8_TILE_BYTES).toBe(8_704);
    expect(ACE_OPT_0091_DENSE_INT8_TILE_WORDS).toBe(2_176);
    for (const portable of [false, true]) {
      const wgsl = aceOpt0091DenseInt8Wgsl(
        { rows: 321, inner: 2_048, columns: 2_048 },
        portable,
      );
      expect(wgsl).toContain("var<storage, read> weight: array<u32>");
      expect(wgsl).toContain("fn unpack_i8x4(word: u32) -> vec4<f16>");
      expect(wgsl).toContain("var acc0_0 = vec4<f32>(0.0)");
      expect(wgsl).toContain("vec4<f32>(f32(a0)) * vec4<f32>(b0)");
    }
  });
});
