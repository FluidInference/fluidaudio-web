import { readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

import {
  INT8_QUALITY_PREVIEW_DIT_MANIFEST_SHA256,
  aceDemoWorkerConfiguration,
  aceProductionWorkerConfiguration,
} from "../src/engines/musicgen-acestep/config.js";

describe("music model variants", () => {
  it("keeps production as the default and isolates the preview trust root", () => {
    const production = aceProductionWorkerConfiguration();
    const preview = aceDemoWorkerConfiguration("int8-quality-preview");

    expect(aceDemoWorkerConfiguration("production")).toEqual(production);
    expect(preview).toEqual({
      ...production,
      ditDensePackage: {
        ...production.ditDensePackage,
        manifestUrl: `https://huggingface.co/FluidInference/ace-step-webgpu-models/resolve/main/v1/dit-int8-fakequant/${INT8_QUALITY_PREVIEW_DIT_MANIFEST_SHA256}/manifest.json`,
        manifestSha256: INT8_QUALITY_PREVIEW_DIT_MANIFEST_SHA256,
      },
    });
  });

  it("labels the preview as experimental and full-size", () => {
    const html = readFileSync(new URL("../music.html", import.meta.url), "utf8");

    expect(html).toContain("Production (recommended)");
    expect(html).toContain("INT8 quality preview (experimental)");
    expect(html).toContain("preview tests audible quantization effects but is not compressed");
    expect(html).toContain("separate 3.02 GB DiT download");
    expect(html).toMatch(/option value="production" selected/);
  });
});
