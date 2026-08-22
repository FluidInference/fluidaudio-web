import { describe, expect, it } from "vitest";
import demoSource from "../demo/main.ts?raw";

import { isAceClientMessage, isAceRuntimeDiagnosticsValue } from
  "../src/runtime/protocol.js";
import {
  ACE_OPT_0054_VAE_REVISION7_MANIFEST_BYTES,
  ACE_OPT_0054_VAE_REVISION7_MANIFEST_SHA256,
} from "../src/model/package.js";
import { ACE_REFERENCE_SUBGROUP_PROFILE } from
  "../src/webgpu/capabilities.js";
import {
  ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_KERNEL_SET_ID,
  ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
  requireAceDitAttentionRuntimeProfile,
} from "../src/webgpu/dit-attention-profile.js";
import { resolveAceDitMixedGemmSelection } from
  "../src/webgpu/dit-backend.js";
import {
  ACE_OPT_0009_DIT_DENSE_RUNTIME_PROFILE,
  ACE_OPT_0056_DIT_SELECTIVE_K4_RUNTIME_PROFILE,
} from "../src/webgpu/dit-fp16-package.js";
import {
  ACE_OPT_0066_VAE_FP16_FIXED32_DUAL_K4_QUALITY_PROFILE,
  ACE_OPT_0072_VAE_FP16_FIXED32_DUAL_K4_PRODUCTION_RUNTIME_PROFILE,
} from "../src/webgpu/vae-fp16-profile.js";
import { classifyAceOpt0018DitCommandMember } from
  "../src/webgpu/dit-graph.js";
import {
  ACE_OPT_0035_VAE_FP16_C2378_MAXIMUM_WINDOW_FRAMES,
  ACE_OPT_0035_VAE_FP16_C2378_WORKSPACE_BYTES,
} from "../src/webgpu/vae-fp16-decoder.js";
import {
  ACE_OPT_0070_VAE_C2378_WINDOW_RUNTIME_PROFILE,
  ACE_VAE_C512_WINDOW_RUNTIME_PROFILE,
  requireAceVaeWindowRuntimeProfile,
} from "../src/webgpu/vae-window-profile.js";
import { planAceVaeChunkedDecode } from "../src/webgpu/vae-chunks.js";
import {
  testDiagnostics,
  testInitializeMessage,
} from "./runtime-fixtures.js";

describe("OPT-0070 exact production promotion", () => {
  it("maps the public production identity to the unchanged physical quad owner", () => {
    expect(requireAceDitAttentionRuntimeProfile(
      ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
    )).toEqual({
      id: ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
      kernelSetId: ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_KERNEL_SET_ID,
      fullSelfAttentionOwner: "opt-0062-fixed32-quad-query32",
      slidingSelfAttentionOwner: "fixed32-subgroup-query8",
      crossAttentionOwner: "fixed32-subgroup-query8",
    });
    const selection = resolveAceDitMixedGemmSelection(
      ACE_REFERENCE_SUBGROUP_PROFILE,
      32,
      32,
      ACE_OPT_0009_DIT_DENSE_RUNTIME_PROFILE,
      ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
      2_250,
      98,
    );
    expect(selection).toMatchObject({
      attentionRuntimeProfile:
        ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
      attentionConfiguration: {
        backend: "opt-0070-fixed32-quad-query32-full-self-production",
        runtimeProfileId:
          ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
        expectedQueryTokens: 2_250,
        expectedConditionTokens: 98,
      },
    });
    expect(() => resolveAceDitMixedGemmSelection(
      ACE_REFERENCE_SUBGROUP_PROFILE,
      32,
      32,
      ACE_OPT_0056_DIT_SELECTIVE_K4_RUNTIME_PROFILE,
      ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
    )).toThrow(/cannot combine/);

    const quantum = Object.freeze({
      index: 27,
      kind: "layer" as const,
      evaluation: 0,
      layer: 1,
      label: "ace-dit-eval-0-layer-1",
    });
    expect(classifyAceOpt0018DitCommandMember(
      quantum,
      `${quantum.label}-self-full-attention`,
      ACE_OPT_0009_DIT_DENSE_RUNTIME_PROFILE,
      ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
    )).toMatchObject({
      family: "self-full",
      backend: "opt-0062-fixed32-quad-query32-full-self",
    });
    expect(classifyAceOpt0018DitCommandMember(
      quantum,
      `${quantum.label}-self-sliding-attention`,
      ACE_OPT_0009_DIT_DENSE_RUNTIME_PROFILE,
      ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
    )).toMatchObject({
      family: "self-sliding",
      backend: "fixed32-subgroup-query8",
    });
  });

  it("authenticates only C512/64 and the explicit C2378/64 production geometry", () => {
    expect(requireAceVaeWindowRuntimeProfile(undefined, 512)).toEqual({
      id: ACE_VAE_C512_WINDOW_RUNTIME_PROFILE,
      maximumWindowFrames: 512,
      overlapFrames: 64,
      requiredWorkspaceBytes: 251_658_240,
    });
    expect(requireAceVaeWindowRuntimeProfile(
      ACE_OPT_0070_VAE_C2378_WINDOW_RUNTIME_PROFILE,
      2_378,
    )).toEqual({
      id: ACE_OPT_0070_VAE_C2378_WINDOW_RUNTIME_PROFILE,
      maximumWindowFrames:
        ACE_OPT_0035_VAE_FP16_C2378_MAXIMUM_WINDOW_FRAMES,
      overlapFrames: 64,
      requiredWorkspaceBytes:
        ACE_OPT_0035_VAE_FP16_C2378_WORKSPACE_BYTES,
    });
    expect(() => requireAceVaeWindowRuntimeProfile(undefined, 2_378))
      .toThrow(/not authenticated/);
    expect(() => requireAceVaeWindowRuntimeProfile(
      ACE_OPT_0070_VAE_C2378_WINDOW_RUNTIME_PROFILE,
      512,
    )).toThrow(/not authenticated/);

    const c4500 = planAceVaeChunkedDecode(4_500, {
      chunkFrames: 2_378,
      overlapFrames: 64,
    });
    expect(c4500.maximumWindowFrames).toBe(2_314);
    expect(c4500.windows.map((window) => [
      window.windowStartLatentFrame,
      window.windowEndLatentFrame,
      window.coreStartLatentFrame,
      window.coreEndLatentFrame,
    ])).toEqual([
      [0, 2_314, 0, 2_250],
      [2_186, 4_500, 2_250, 4_500],
    ]);
    const c6000 = planAceVaeChunkedDecode(6_000, {
      chunkFrames: 2_378,
      overlapFrames: 64,
    });
    expect(c6000.maximumWindowFrames).toBe(2_378);
    expect(c6000.windows.map((window) => window.latentWindowFrames)).toEqual([
      2_314,
      2_378,
      1_564,
    ]);
  });

  it("accepts only the complete public OPT-0070 tuple and authenticates diagnostics", () => {
    const base = testInitializeMessage();
    const candidate = {
      ...base,
      configuration: {
        ...base.configuration,
        ditAttentionRuntimeProfile:
          ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE,
        vaePackage: {
          ...base.configuration.vaePackage,
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
    expect(isAceClientMessage(candidate)).toBe(true);
    expect(isAceClientMessage({
      ...candidate,
      configuration: {
        ...candidate.configuration,
        vaePackage: base.configuration.vaePackage,
      },
    })).toBe(false);
    expect(isAceClientMessage({
      ...candidate,
      configuration: {
        ...candidate.configuration,
        ditAttentionRuntimeProfile: undefined,
      },
    })).toBe(false);

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
      ditAttentionKernelSetId:
        "opt-0062-query8-plus-quad-query32-full-self-v1",
    })).toBe(false);
  });

  it("selects both explicit public identities in the demo default", () => {
    expect(demoSource).toContain(
      `"${ACE_OPT_0070_DIT_QUAD_QUERY_ATTENTION_RUNTIME_PROFILE}"`,
    );
    expect(demoSource).toContain(
      `"${ACE_OPT_0070_VAE_C2378_WINDOW_RUNTIME_PROFILE}"`,
    );
    expect(demoSource).toContain(
      `"${ACE_OPT_0072_VAE_FP16_FIXED32_DUAL_K4_PRODUCTION_RUNTIME_PROFILE}"`,
    );
    expect(demoSource).toContain("maxWindowFrames: 2378");
  });
});
