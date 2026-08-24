import { describe, expect, it } from "vitest";
import demoSource from "../demo/main.ts?raw";

import {
  ACE_OPT_0054_VAE_REVISION7_MANIFEST_BYTES,
  ACE_OPT_0054_VAE_REVISION7_MANIFEST_SHA256,
} from "../src/model/package.js";
import {
  isAceClientMessage,
  isAceRuntimeDiagnosticsValue,
} from "../src/runtime/protocol.js";
import { resolveAceVaePackageRuntimeIdentity } from
  "../src/runtime/webgpu-pipeline.js";
import {
  ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_KERNEL_SET_ID,
  ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
} from "../src/webgpu/dit-attention-profile.js";
import {
  ACE_OPT_0052_VAE_CONV_TRANSPOSE1D_K4_SHAPE_SELECTOR_KERNEL_ID,
} from
  "../src/webgpu/kernels/vae-conv-transpose1d-fp16-k4-shape-selector.js";
import {
  ACE_OPT_0057_VAE_K7_SHAPE_SELECTOR_KERNEL_ID,
} from
  "../src/webgpu/kernels/vae-conv1d-fp16-k4-row-reuse-shape-selector.js";
import {
  planAceVaeDecoder,
  planAceVaeDecoderQuanta,
} from "../src/webgpu/vae-decoder.js";
import {
  ACE_OPT_0054_VAE_FP16_FIXED32_REVISION7_PROFILE,
  ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_KERNEL_SET_ID,
  ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PRECISION_MAP,
  ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PRECISION_MAP_SHA256,
  ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PROFILE,
  ACE_OPT_0072_VAE_FP16_FIXED32_DUAL_K4_PRODUCTION_PROFILE_CONTRACT,
  ACE_OPT_0072_VAE_FP16_FIXED32_DUAL_K4_PRODUCTION_RUNTIME_PROFILE,
  ACE_OPT_0088_VAE_FP16_PORTABLE_DUAL_K4_PROFILE,
  ACE_OPT_0088_VAE_PORTABLE_PRODUCTION_PROFILE_CONTRACT,
  ACE_VAE_RUNTIME_PROFILE_IDS,
  requireAceOpt0072VaeProductionRuntimeProfile,
  requireAceOpt0072VaeProductionRuntimeProfileForBackend,
} from "../src/webgpu/vae-fp16-profile.js";
import {
  ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_KERNEL_TOPOLOGY,
} from "../src/webgpu/vae-fp16-decoder.js";
import {
  ACE_OPT_0070_VAE_C2378_WINDOW_RUNTIME_PROFILE,
} from "../src/webgpu/vae-window-profile.js";
import {
  testDiagnostics,
  testInitializeMessage,
} from "./runtime-fixtures.js";

describe("OPT-0072 revision-7 dual-K4 production promotion", () => {
  it("maps one distinct public identity to the frozen OPT-0066 physical owner", () => {
    const contract = requireAceOpt0072VaeProductionRuntimeProfile(
      ACE_OPT_0072_VAE_FP16_FIXED32_DUAL_K4_PRODUCTION_RUNTIME_PROFILE,
    );
    expect(contract).toBe(
      ACE_OPT_0072_VAE_FP16_FIXED32_DUAL_K4_PRODUCTION_PROFILE_CONTRACT,
    );
    expect(Object.isFrozen(contract)).toBe(true);
    expect(contract).toEqual({
      id: ACE_OPT_0072_VAE_FP16_FIXED32_DUAL_K4_PRODUCTION_RUNTIME_PROFILE,
      physicalRuntimeProfileId:
        ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PROFILE.id,
      manifestSha256: ACE_OPT_0054_VAE_REVISION7_MANIFEST_SHA256,
      manifestByteLength: ACE_OPT_0054_VAE_REVISION7_MANIFEST_BYTES,
      kernelSetId:
        ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_KERNEL_SET_ID,
      precisionMapSha256:
        ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PRECISION_MAP_SHA256,
    });
    expect(contract.id).not.toBe(
      ACE_OPT_0054_VAE_FP16_FIXED32_REVISION7_PROFILE.id,
    );
    expect(contract.id).not.toBe(
      ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PROFILE.id,
    );
    expect(ACE_VAE_RUNTIME_PROFILE_IDS).not.toContain(contract.id);
    expect(() => requireAceOpt0072VaeProductionRuntimeProfile(
      ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PROFILE.id,
    )).toThrow(/not authenticated/);
  });

  it("keys the public identity by backend without changing the fixed32 arm", () => {
    const publicId =
      ACE_OPT_0072_VAE_FP16_FIXED32_DUAL_K4_PRODUCTION_RUNTIME_PROFILE;
    // The subgroups arm delegates to the unchanged fixed32 authenticator.
    expect(requireAceOpt0072VaeProductionRuntimeProfileForBackend(
      publicId,
      "subgroups",
    )).toBe(ACE_OPT_0072_VAE_FP16_FIXED32_DUAL_K4_PRODUCTION_PROFILE_CONTRACT);
    expect(requireAceOpt0072VaeProductionRuntimeProfileForBackend(
      publicId,
      "subgroups",
    )).toBe(requireAceOpt0072VaeProductionRuntimeProfile(publicId));
    // The portable arm maps the same public identity onto physical OPT-0088.
    const portable = requireAceOpt0072VaeProductionRuntimeProfileForBackend(
      publicId,
      "portable",
    );
    expect(portable).toBe(ACE_OPT_0088_VAE_PORTABLE_PRODUCTION_PROFILE_CONTRACT);
    expect(portable).toMatchObject({
      id: publicId,
      physicalRuntimeProfileId:
        ACE_OPT_0088_VAE_FP16_PORTABLE_DUAL_K4_PROFILE.id,
      manifestSha256: ACE_OPT_0054_VAE_REVISION7_MANIFEST_SHA256,
      manifestByteLength: ACE_OPT_0054_VAE_REVISION7_MANIFEST_BYTES,
      kernelSetId: ACE_OPT_0088_VAE_FP16_PORTABLE_DUAL_K4_PROFILE.kernelSetId,
      precisionMapSha256:
        ACE_OPT_0088_VAE_FP16_PORTABLE_DUAL_K4_PROFILE.precisionMapSha256,
    });
    expect(portable.precisionMapSha256).not.toBe(
      ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PROFILE.precisionMapSha256,
    );
    expect(() => requireAceOpt0072VaeProductionRuntimeProfileForBackend(
      ACE_OPT_0088_VAE_FP16_PORTABLE_DUAL_K4_PROFILE.id,
      "portable",
    )).toThrow(/not authenticated/);
  });

  it("reconciles every C512 K7 and ConvTranspose quantum under the physical owner", () => {
    expect(
      ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_KERNEL_TOPOLOGY,
    ).toMatchObject({
      id: ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_KERNEL_SET_ID,
      backend: "fixed32-subgroup-dual-k4-quality",
      conv1dK7: ACE_OPT_0057_VAE_K7_SHAPE_SELECTOR_KERNEL_ID,
      convTranspose1d:
        ACE_OPT_0052_VAE_CONV_TRANSPOSE1D_K4_SHAPE_SELECTOR_KERNEL_ID,
    });
    const plan = planAceVaeDecoder(512);
    const cooperative = planAceVaeDecoderQuanta(plan);
    const quantumCounts = plan.operations.map((_operation, operationIndex) =>
      cooperative.quanta.filter((quantum) =>
        quantum.operationIndex === operationIndex
      ).length
    );
    const entries =
      ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PRECISION_MAP.entries
        .filter((entry) => entry.graphOperationIndex !== null);
    const count = (selected: typeof entries) => selected.reduce(
      (sum, entry) => sum + quantumCounts[entry.graphOperationIndex!]!,
      0,
    );
    const k7 = entries.filter((entry) => entry.kernelFamily === "conv1d-k7");
    const transpose = entries.filter((entry) =>
      entry.kernelFamily === "conv-transpose1d"
    );
    const k4 = (entry: (typeof entries)[number]) =>
      entry.registerArithmetic ===
        "float16-dot4-partials-then-float32-running-state";

    expect(k7).toHaveLength(17);
    expect(count(k7)).toBe(4_090);
    expect(k7.filter(k4)).toHaveLength(12);
    expect(count(k7.filter(k4))).toBe(3_360);
    expect(count(k7.filter((entry) => !k4(entry)))).toBe(730);
    expect(transpose).toHaveLength(5);
    expect(count(transpose)).toBe(644);
    expect(transpose.filter(k4)).toHaveLength(4);
    expect(count(transpose.filter(k4))).toBe(552);
    expect(count(transpose.filter((entry) => !k4(entry)))).toBe(92);
    expect(entries.filter((entry) =>
      entry.kernelFamily !== "conv1d-k7" &&
      entry.kernelFamily !== "conv-transpose1d"
    ).every((entry) => entry.registerArithmetic === "float32")).toBe(true);
  });

  it("accepts only OPT-0072 plus OPT-0070 attention plus OPT-0070 C2378", () => {
    const base = testInitializeMessage();
    const product = {
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
    expect(isAceClientMessage(product)).toBe(true);
    expect(isAceClientMessage({
      ...product,
      configuration: {
        ...product.configuration,
        ditAttentionRuntimeProfile: undefined,
      },
    })).toBe(false);
    expect(isAceClientMessage({
      ...product,
      configuration: {
        ...product.configuration,
        vaePackage: {
          ...product.configuration.vaePackage,
          windowRuntimeProfile: undefined,
          maxWindowFrames: 512,
        },
      },
    })).toBe(false);
    expect(isAceClientMessage({
      ...product,
      configuration: {
        ...product.configuration,
        vaePackage: {
          ...product.configuration.vaePackage,
          runtimeProfile:
            ACE_OPT_0054_VAE_FP16_FIXED32_REVISION7_PROFILE.id,
        },
      },
    })).toBe(false);
    expect(isAceClientMessage({
      ...product,
      configuration: {
        ...product.configuration,
        vaePackage: {
          ...base.configuration.vaePackage,
          windowRuntimeProfile:
            ACE_OPT_0070_VAE_C2378_WINDOW_RUNTIME_PROFILE,
          maxWindowFrames: 2_378,
        },
      },
    })).toBe(false);
    expect(isAceClientMessage({
      ...product,
      configuration: {
        ...product.configuration,
        vaePackage: {
          ...product.configuration.vaePackage,
          runtimeProfile:
            ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PROFILE.id,
        },
      },
    })).toBe(false);
  });

  it("authenticates public diagnostics while retaining physical trust roots", () => {
    const diagnostics = testDiagnostics({
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
        ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PROFILE.precisionMapSha256!,
      vaeWindowRuntimeProfile:
        ACE_OPT_0070_VAE_C2378_WINDOW_RUNTIME_PROFILE,
      vaeMaxWindowFrames: 2_378,
    });
    expect(isAceRuntimeDiagnosticsValue(diagnostics)).toBe(true);
    expect(isAceRuntimeDiagnosticsValue({
      ...diagnostics,
      vaeRuntimeProfile:
        ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PROFILE.id,
    })).toBe(false);
    expect(isAceRuntimeDiagnosticsValue({
      ...diagnostics,
      vaeKernelSetId:
        ACE_OPT_0054_VAE_FP16_FIXED32_REVISION7_PROFILE.kernelSetId,
    })).toBe(false);
    expect(isAceRuntimeDiagnosticsValue({
      ...diagnostics,
      vaePrecisionMapSha256:
        ACE_OPT_0054_VAE_FP16_FIXED32_REVISION7_PROFILE.precisionMapSha256,
    })).toBe(false);
  });

  it("resolves the public package identity to OPT-0066 and selects it in the demo", () => {
    expect(resolveAceVaePackageRuntimeIdentity({
      manifestUrl:
        "/model/files-fp16-vae-revision7-experimental/manifest.json",
      manifestSha256: ACE_OPT_0054_VAE_REVISION7_MANIFEST_SHA256,
      runtimeProfile:
        ACE_OPT_0072_VAE_FP16_FIXED32_DUAL_K4_PRODUCTION_RUNTIME_PROFILE,
      windowRuntimeProfile:
        ACE_OPT_0070_VAE_C2378_WINDOW_RUNTIME_PROFILE,
      maxWindowFrames: 2_378,
    })).toEqual({
      role: "opt-0072-rev7-production",
      manifestSha256: ACE_OPT_0054_VAE_REVISION7_MANIFEST_SHA256,
      manifestByteLength: ACE_OPT_0054_VAE_REVISION7_MANIFEST_BYTES,
      runtimeProfile:
        ACE_OPT_0072_VAE_FP16_FIXED32_DUAL_K4_PRODUCTION_RUNTIME_PROFILE,
      physicalRuntimeProfile:
        ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PROFILE.id,
      kernelSetId:
        ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PROFILE.kernelSetId,
      precisionMapSha256:
        ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PROFILE.precisionMapSha256,
    });
    expect(demoSource).toContain(
      "/model/files-fp16-vae-revision7-experimental/manifest.json",
    );
    expect(demoSource).toContain(ACE_OPT_0054_VAE_REVISION7_MANIFEST_SHA256);
    expect(demoSource).toContain(
      ACE_OPT_0072_VAE_FP16_FIXED32_DUAL_K4_PRODUCTION_RUNTIME_PROFILE,
    );
    expect(demoSource).not.toContain(
      '"opt-0028-mixed-fp16-fixed32-exact-packed-v1"',
    );
  });
});
