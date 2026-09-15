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
        manifestUrl: `https://huggingface.co/FluidInference/ace-step-webgpu-models/resolve/main/v1/dit-int8-packed/${INT8_QUALITY_PREVIEW_DIT_MANIFEST_SHA256}/manifest.json`,
        manifestSha256: INT8_QUALITY_PREVIEW_DIT_MANIFEST_SHA256,
        runtimeProfile: "opt-0091-int8-weight-only-v1",
      },
    });
  });

  it("labels the preview with its measured packed size", () => {
    const html = readFileSync(new URL("../music.html", import.meta.url), "utf8");

    expect(html).toContain("Production (recommended)");
    expect(html).toContain("Compressed INT8 preview (experimental)");
    expect(html).toMatch(/id="int8-model-option"[^>]+hidden/);
    expect(html).toMatch(/id="int8-model-hint" hidden/);
    expect(html).toContain("1.70 GB packed INT8 DiT");
    expect(html).toContain("43.7% smaller");
    expect(html).toMatch(/option value="production" selected/);
  });

  it("preselects the compressed model only for the private verification URL", () => {
    const source = readFileSync(new URL("../src/music.ts", import.meta.url), "utf8");

    expect(source).toContain('new URLSearchParams(location.search).get("int8") !== "1"');
    expect(source).toContain('modelVariantSelect.value = "int8-quality-preview"');
  });
});
