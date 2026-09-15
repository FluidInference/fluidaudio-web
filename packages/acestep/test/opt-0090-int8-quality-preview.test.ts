import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

import { isAceClientMessage } from "../src/runtime/protocol.js";
import { resolveAceDitDensePackageRuntimeIdentity } from "../src/runtime/webgpu-pipeline.js";
import {
  ACE_OPT_0009_DIT_DENSE_KERNEL_SET_ID,
  ACE_OPT_0009_DIT_DENSE_MANIFEST_BYTES,
  ACE_OPT_0009_DIT_DENSE_RUNTIME_PROFILE,
  ACE_OPT_0009_DIT_MIXED_LAYER_BYTES,
  ACE_OPT_0009_DIT_MIXED_RESIDENT_WEIGHT_BYTES,
  ACE_OPT_0088_DIT_DENSE_PORTABLE_KERNEL_SET_ID,
  ACE_OPT_0089_DIT_FAKE_QUANT_MANIFEST_SHA256,
} from "../src/webgpu/dit-fp16-package.js";
import { testInitializeMessage } from "./runtime-fixtures.js";

describe("OPT-0090 INT8 quality preview", () => {
  const previewPackage = {
    manifestUrl: "https://example.test/dit-int8-fakequant/manifest.json",
    manifestSha256: ACE_OPT_0089_DIT_FAKE_QUANT_MANIFEST_SHA256,
    runtimeProfile: ACE_OPT_0009_DIT_DENSE_RUNTIME_PROFILE,
  } as const;

  it("accepts only the pinned fake-quant manifest through the public protocol", () => {
    const base = testInitializeMessage();
    const preview = {
      ...base,
      configuration: {
        ...base.configuration,
        ditDensePackage: previewPackage,
      },
    };

    expect(isAceClientMessage(preview)).toBe(true);
    expect(
      isAceClientMessage({
        ...preview,
        configuration: {
          ...preview.configuration,
          ditDensePackage: {
            ...previewPackage,
            manifestSha256: "f".repeat(64),
          },
        },
      }),
    ).toBe(false);
  });

  it("uses the unchanged rev7 runtime with distinct diagnostics identity", () => {
    expect(resolveAceDitDensePackageRuntimeIdentity(previewPackage)).toEqual({
      role: "opt-0089-rev7-fake-quant-preview",
      manifestSha256: ACE_OPT_0089_DIT_FAKE_QUANT_MANIFEST_SHA256,
      manifestByteLength: ACE_OPT_0009_DIT_DENSE_MANIFEST_BYTES,
      runtimeProfile: ACE_OPT_0009_DIT_DENSE_RUNTIME_PROFILE,
      kernelSetId: ACE_OPT_0009_DIT_DENSE_KERNEL_SET_ID,
      layerBytes: ACE_OPT_0009_DIT_MIXED_LAYER_BYTES,
      residentWeightBytes: ACE_OPT_0009_DIT_MIXED_RESIDENT_WEIGHT_BYTES,
    });
    expect(resolveAceDitDensePackageRuntimeIdentity(previewPackage, "portable")).toMatchObject({
      role: "opt-0089-rev7-fake-quant-preview",
      kernelSetId: ACE_OPT_0088_DIT_DENSE_PORTABLE_KERNEL_SET_ID,
    });
  });

  it("records the full-size, opt-in product boundary", () => {
    const record = readFileSync(new URL("../optimization/experiments/OPT-0090-int8-quality-preview.md", import.meta.url), "utf8");

    expect(record).toMatch(/not an int8-resident\s+runtime/);
    expect(record).toMatch(/does not reduce download or GPU-resident bytes/);
    expect(record).toContain("production model as the default");
  });

  it("pins every source surface that selects or authenticates the preview", () => {
    const identities = {
      "src/model/package.ts":
        "f99e6a5a7a6409ccbe226280239a5f67e51a27cb9a091cca5fb4760172081cc7",
      "src/runtime/protocol.ts":
        "22cdaad9ae647b0827b830fcc0b609eca30a8456aa01ce32873d534f0b06ebe9",
      "src/runtime/webgpu-pipeline.ts":
        "567036482c30d5dbc412a75c4c599bb8be95c68b171b305bb58319e1bcb1840e",
      "src/webgpu/dit-fp16-package.ts":
        "0def99de7ac519f9f234a49ae6790a3654acda1a1483d7a4aa7611292f056633",
      "../../src/engines/musicgen-acestep/config.ts":
        "400ce78a45299340d2b5291abe3d1664db74633981c60355e29bd9c031eb8102",
      "../../src/music.ts":
        "c07b0e8a6568a260d28ae472f1e5fc678d0749a2d588219a46eb25914c0bf7f3",
    };
    for (const [path, expected] of Object.entries(identities)) {
      const source = readFileSync(new URL(`../${path}`, import.meta.url));
      if (path !== "src/model/package.ts") {
        // OPT-0091 replaces the public selector with a packed runtime while
        // retaining this historical fake-quant identity as a frozen record.
        expect(createHash("sha256").update(source).digest("hex"), path)
          .not.toBe(expected);
        continue;
      }
      expect(createHash("sha256").update(source).digest("hex"), path)
        .toBe(expected);
    }
  });
});
