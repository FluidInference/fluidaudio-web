import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import {
  ACE_OPT_0054_VAE_REVISION7_MANIFEST_BYTES,
  ACE_OPT_0054_VAE_REVISION7_MANIFEST_SHA256,
  loadAcePackageManifest,
  type AceLoadedPackageManifest,
} from "../src/model/package.js";
import {
  ACE_VAE_REVISION7_K7_ROW_REUSE_CONTRACTS,
  ACE_VAE_REVISION7_TRANSPOSE_K4_CONTRACTS,
  type AcePackageManifest,
} from "../src/model/manifest.js";
import {
  ACE_OPT_0054_VAE_FP16_FIXED32_REVISION7_PRECISION_MAP,
  ACE_OPT_0054_VAE_FP16_FIXED32_REVISION7_PRECISION_MAP_CANONICAL_JSON,
  ACE_OPT_0054_VAE_FP16_FIXED32_REVISION7_PRECISION_MAP_SHA256,
  ACE_OPT_0054_VAE_FP16_FIXED32_REVISION7_PROFILE,
  ACE_OPT_0054_REVISION6_SCALAR_FP32_SEQUENTIAL_ORACLE_PROFILE,
  ACE_OPT_0040_VAE_FP16_FIXED32_SHAPE_SELECTED_PROFILE,
  hashAceVaePrecisionMap,
  selectAceVaeRuntimeProfile,
} from "../src/webgpu/vae-fp16-profile.js";
import {
  ACE_OPT_0054_VAE_FP16_FIXED32_REVISION7_KERNEL_TOPOLOGY,
} from "../src/webgpu/vae-fp16-decoder.js";
import { planAceVaeDecoder } from "../src/webgpu/vae-decoder.js";
import {
  ACE_OPT_0057_VAE_K7_ROUTES,
  ACE_OPT_0057_VAE_K7_SHAPE_SELECTOR_KERNEL_ID,
  selectAceOpt0057VaeK7,
} from "../src/webgpu/kernels/vae-conv1d-fp16-k4-row-reuse-shape-selector.js";
import {
  ACE_OPT_0052_VAE_CONV_TRANSPOSE1D_K4_SHAPE_SELECTOR_KERNEL_ID,
  selectAceOpt0052VaeConvTranspose1d,
} from "../src/webgpu/kernels/vae-conv-transpose1d-fp16-k4-shape-selector.js";
import {
  resolveAceVaePackageRuntimeIdentity,
} from "../src/runtime/webgpu-pipeline.js";

const DEVICE_LIMITS = Object.freeze({
  maxBufferSize: 251_658_240,
  maxStorageBufferBindingSize: 251_658_240,
  maxComputeWorkgroupStorageSize: 16 * 1024,
  maxComputeInvocationsPerWorkgroup: 256,
});
const REVISION7_MANIFEST_URL = new URL(
  "../model/files-fp16-vae-revision7-experimental/manifest.json",
  import.meta.url,
);

describe("OPT-0054 revision-7 VAE runtime contract", () => {
  it("freezes the truthful mixed scalar/K4 precision map", () => {
    const independent = createHash("sha256")
      .update(ACE_OPT_0054_VAE_FP16_FIXED32_REVISION7_PRECISION_MAP_CANONICAL_JSON)
      .digest("hex");
    expect(hashAceVaePrecisionMap(
      ACE_OPT_0054_VAE_FP16_FIXED32_REVISION7_PRECISION_MAP,
    )).toBe(ACE_OPT_0054_VAE_FP16_FIXED32_REVISION7_PRECISION_MAP_SHA256);
    expect(independent).toBe(
      ACE_OPT_0054_VAE_FP16_FIXED32_REVISION7_PRECISION_MAP_SHA256,
    );

    const entries = ACE_OPT_0054_VAE_FP16_FIXED32_REVISION7_PRECISION_MAP.entries;
    const k7Labels: ReadonlySet<string> = new Set(
      ACE_VAE_REVISION7_K7_ROW_REUSE_CONTRACTS.map(({ operationLabel }) =>
        operationLabel
      ),
    );
    const transposeLabels: ReadonlySet<string> = new Set(
      ACE_VAE_REVISION7_TRANSPOSE_K4_CONTRACTS.map(({ operationLabel }) =>
        operationLabel
      ),
    );
    const k4 = entries.filter(({ label }) =>
      k7Labels.has(label) || transposeLabels.has(label)
    );
    expect(k4).toHaveLength(16);
    expect(k4.every(({ registerArithmetic, contraction }) =>
      registerArithmetic ===
        "float16-dot4-partials-then-float32-running-state" &&
      contraction === "wgsl-f16-dot4-partials-then-f32-add"
    )).toBe(true);
    expect(entries.filter(({ label }) => !k7Labels.has(label) &&
      !transposeLabels.has(label)).every(({ registerArithmetic }) =>
      registerArithmetic === "float32"
    )).toBe(true);
  });

  it("selects only the exact authenticated revision-7 identity", () => {
    const loaded = authenticatedRevision7();
    expect(selectAceVaeRuntimeProfile({
      requestedProfile: ACE_OPT_0054_VAE_FP16_FIXED32_REVISION7_PROFILE.id,
      package: loaded,
      deviceFeatures: ["shader-f16", "subgroups"],
      subgroupMinSize: 32,
      subgroupMaxSize: 32,
      deviceLimits: DEVICE_LIMITS,
      decoderPlan: planAceVaeDecoder(256),
    })).toBe(ACE_OPT_0054_VAE_FP16_FIXED32_REVISION7_PROFILE);

    expect(() => selectAceVaeRuntimeProfile({
      requestedProfile: ACE_OPT_0054_VAE_FP16_FIXED32_REVISION7_PROFILE.id,
      package: { ...loaded, manifestSha256: "0".repeat(64) },
      deviceFeatures: ["shader-f16", "subgroups"],
      subgroupMinSize: 32,
      subgroupMaxSize: 32,
      deviceLimits: DEVICE_LIMITS,
      decoderPlan: planAceVaeDecoder(256),
    })).toThrow(/exact authenticated package identity/);
  });

  it("rejects rev7 parser authorization unless the external SHA is exact", async () => {
    let fetched = false;
    await expect(loadAcePackageManifest({
      manifestUrl: "https://example.invalid/vae/manifest.json",
      expectedManifestSha256: "0".repeat(64),
      expectedProfile: "fp16-vae-experimental",
      authenticatedVaeConverterRevision: 7,
      fetch: (async () => {
        fetched = true;
        throw new Error("must reject before fetch");
      }) as typeof fetch,
    })).rejects.toThrow(/exact authenticated revision-7 manifest/);
    expect(fetched).toBe(false);
  });

  it.skipIf(!existsSync(REVISION7_MANIFEST_URL))(
    "authenticates and parses the generated revision-7 manifest only explicitly",
    async () => {
      const bytes = readFileSync(REVISION7_MANIFEST_URL);
      const fetchManifest = (async () => new Response(bytes, {
        status: 200,
        headers: { "content-length": String(bytes.byteLength) },
      })) as typeof fetch;
      const loaded = await loadAcePackageManifest({
        manifestUrl: REVISION7_MANIFEST_URL.href,
        expectedManifestSha256: ACE_OPT_0054_VAE_REVISION7_MANIFEST_SHA256,
        expectedProfile: "fp16-vae-experimental",
        authenticatedVaeConverterRevision: 7,
        fetch: fetchManifest,
      });
      expect(loaded.manifestSha256).toBe(
        ACE_OPT_0054_VAE_REVISION7_MANIFEST_SHA256,
      );
      expect(loaded.manifestByteLength).toBe(
        ACE_OPT_0054_VAE_REVISION7_MANIFEST_BYTES,
      );
      expect(loaded.manifest.provenance.converterRevision).toBe(7);

      await expect(loadAcePackageManifest({
        manifestUrl: REVISION7_MANIFEST_URL.href,
        expectedManifestSha256: ACE_OPT_0054_VAE_REVISION7_MANIFEST_SHA256,
        expectedProfile: "fp16-vae-experimental",
        fetch: fetchManifest,
      })).rejects.toThrow(/unsupported converter revision/);
    },
  );

  it("keeps the revision-6 scalar oracle explicit and distinct", () => {
    expect(ACE_OPT_0054_REVISION6_SCALAR_FP32_SEQUENTIAL_ORACLE_PROFILE).toBe(
      ACE_OPT_0040_VAE_FP16_FIXED32_SHAPE_SELECTED_PROFILE,
    );
    expect(ACE_OPT_0054_REVISION6_SCALAR_FP32_SEQUENTIAL_ORACLE_PROFILE.id)
      .not.toBe(ACE_OPT_0054_VAE_FP16_FIXED32_REVISION7_PROFILE.id);
    expect(ACE_OPT_0054_REVISION6_SCALAR_FP32_SEQUENTIAL_ORACLE_PROFILE.manifestSha256)
      .not.toBe(ACE_OPT_0054_VAE_FP16_FIXED32_REVISION7_PROFILE.manifestSha256);
  });

  it("routes every K7 label by exact authenticated shape without fallback", () => {
    const operations = planAceVaeDecoder(256).operations.filter(
      (operation) => operation.kind === "conv1d" &&
        operation.shape.kernelSize === 7,
    );
    expect(operations).toHaveLength(17);
    expect(ACE_OPT_0057_VAE_K7_ROUTES).toHaveLength(17);
    const owners = operations.map((operation) => {
      if (operation.kind !== "conv1d") throw new Error("unreachable");
      return selectAceOpt0057VaeK7(
        operation.label,
        operation.shape,
        operation.bias !== undefined,
        operation.output === "output" ? "float32" : "float16",
      ).route.owner;
    });
    expect(owners.filter((owner) => owner === "row-reuse-k4")).toHaveLength(12);
    expect(owners.filter((owner) => owner === "native-scalar-fp32"))
      .toHaveLength(5);
    const selected = operations.find(({ label }) =>
      label === "block-0-res-1-conv1"
    )!;
    if (selected.kind !== "conv1d") throw new Error("unreachable");
    expect(() => selectAceOpt0057VaeK7(
      selected.label,
      { ...selected.shape, dilation: 3 },
      true,
      "float16",
    )).toThrow(/authenticated K7 contract/);
    expect(() => selectAceOpt0057VaeK7(
      "undeclared-k7",
      selected.shape,
      true,
      "float16",
    )).toThrow(/no K7 route/);
  });

  it("routes block 0 to the exact rev6 owner and blocks 1-4 to K4", () => {
    const operations = planAceVaeDecoder(256).operations.filter(
      (operation) => operation.kind === "conv-transpose1d",
    );
    expect(operations).toHaveLength(5);
    const selections = operations.map((operation) => {
      if (operation.kind !== "conv-transpose1d") throw new Error("unreachable");
      return selectAceOpt0052VaeConvTranspose1d(
        operation.label,
        operation.shape,
      );
    });
    expect(selections.map(({ owner }) => owner)).toEqual([
      "revision6-polyphase",
      "k4-channel-reuse",
      "k4-channel-reuse",
      "k4-row-reuse",
      "k4-row-reuse",
    ]);
    expect(selections[0]!.plan).toBeNull();
    expect(selections.slice(1).every(({ plan }) => plan !== null)).toBe(true);
    const block1 = operations[1]!;
    if (block1.kind !== "conv-transpose1d") throw new Error("unreachable");
    expect(() => selectAceOpt0052VaeConvTranspose1d(
      block1.label,
      { ...block1.shape, stride: 4 },
    )).toThrow();
    expect(() => selectAceOpt0052VaeConvTranspose1d(
      "undeclared-transpose",
      block1.shape,
    )).toThrow(/no ConvTranspose1D route/);
  });

  it("freezes the candidate topology and exact protocol identity pairs", () => {
    expect(ACE_OPT_0054_VAE_FP16_FIXED32_REVISION7_KERNEL_TOPOLOGY).toMatchObject({
      id: ACE_OPT_0054_VAE_FP16_FIXED32_REVISION7_PROFILE.kernelSetId,
      conv1dK7: ACE_OPT_0057_VAE_K7_SHAPE_SELECTOR_KERNEL_ID,
      convTranspose1d:
        ACE_OPT_0052_VAE_CONV_TRANSPOSE1D_K4_SHAPE_SELECTOR_KERNEL_ID,
    });
    expect(resolveAceVaePackageRuntimeIdentity({
      manifestUrl: "/model/files-fp16-vae-revision7-experimental/manifest.json",
      manifestSha256: ACE_OPT_0054_VAE_REVISION7_MANIFEST_SHA256,
      runtimeProfile: ACE_OPT_0054_VAE_FP16_FIXED32_REVISION7_PROFILE.id,
      maxWindowFrames: 512,
    })).toMatchObject({
      role: "opt-0054-rev7-candidate",
      manifestByteLength: ACE_OPT_0054_VAE_REVISION7_MANIFEST_BYTES,
      kernelSetId: ACE_OPT_0054_VAE_FP16_FIXED32_REVISION7_PROFILE.kernelSetId,
    });
    expect(() => resolveAceVaePackageRuntimeIdentity({
      manifestUrl: "/forged/manifest.json",
      manifestSha256: ACE_OPT_0054_VAE_REVISION7_MANIFEST_SHA256,
      runtimeProfile: ACE_OPT_0040_VAE_FP16_FIXED32_SHAPE_SELECTED_PROFILE.id,
      maxWindowFrames: 512,
    } as never)).toThrow(/exact authenticated manifest\/profile pair/);
  });
});

function authenticatedRevision7(): AceLoadedPackageManifest {
  const manifest = {
    profile: "fp16-vae-experimental",
    provenance: { converterRevision: 7 },
  } as AcePackageManifest;
  return {
    manifest,
    manifestUrl: "https://example.invalid/vae-revision7/manifest.json",
    manifestSha256: ACE_OPT_0054_VAE_REVISION7_MANIFEST_SHA256,
    manifestByteLength: ACE_OPT_0054_VAE_REVISION7_MANIFEST_BYTES,
    manifestId: `ace-step-webgpu-v1:fp16-vae-experimental:${ACE_OPT_0054_VAE_REVISION7_MANIFEST_SHA256}`,
  };
}
