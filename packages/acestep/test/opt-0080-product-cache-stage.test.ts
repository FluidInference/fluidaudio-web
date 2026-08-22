import { readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

import { ACE_EXPERIMENTAL_VAE_PARAMETER_BYTES } from
  "../src/model/manifest.js";
import { ACE_OPT_0054_VAE_REVISION7_MANIFEST_SHA256 } from
  "../src/model/package.js";
import { ACE_OPT_0011_VAE_FP16_WEIGHT_FILES } from
  "../src/webgpu/vae-fp16-package.js";

const SOURCE = readFileSync(new URL(
  "./browser/opt-0080-product-cache-stage.ts",
  import.meta.url,
), "utf8");

describe("OPT-0080 product cache staging", () => {
  it("pins only the exact production revision-7 VAE acquisition", () => {
    expect(ACE_OPT_0054_VAE_REVISION7_MANIFEST_SHA256).toBe(
      "36a54d79777d6826088095ba6ebc028fb4bea546368c0f0a29cd0eee8d656da7",
    );
    expect(ACE_OPT_0011_VAE_FP16_WEIGHT_FILES).toHaveLength(7);
    expect(ACE_EXPERIMENTAL_VAE_PARAMETER_BYTES).toBe(168_791_552);
    expect(SOURCE).toContain("createAceOpt0011VaeAcquisitionManifest");
    expect(SOURCE).toContain("acquireAceModelFiles");
    expect(SOURCE).toContain("authenticatedVaeConverterRevision: 7");
  });

  it("cannot request a GPU or start product inference", () => {
    expect(SOURCE).not.toContain("requestAceWebGpuDevice");
    expect(SOURCE).not.toContain("createAceWebGpuPipelineBackend");
    expect(SOURCE).not.toContain(".generate(");
    expect(SOURCE).toContain("gpuDeviceRequested: false");
    expect(SOURCE).toContain("productGenerationStarted: false");
  });
});
