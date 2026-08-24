import { afterEach, describe, expect, it, vi } from "vitest";

import { loadGpuWeightPackage } from "../src/model/package.js";

const WEIGHT_SHA = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

describe("DiCoSe GPU weight package loading", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("binds a stable weight filename to the freshly fetched manifest digest", async () => {
    const manifest = {
      schema: "dicose-wgsl-package-v1",
      source: {
        upstreamRevision: "test",
        deterministicCheckpointSha256: WEIGHT_SHA,
        cdCheckpointSha256: WEIGHT_SHA,
      },
      config: {
        sampleRate: 44_100,
        nFft: 2_048,
        hopLength: 441,
        winLength: 2_048,
        stereo: true,
        stems: ["drums", "bass", "other", "vocals"],
        dim: 384,
        depth: 8,
        heads: 8,
        dimHead: 64,
        freqsPerBands: [1],
      },
      weights: {
        file: "weights.f16.bin",
        byteLength: 2,
        sha256: WEIGHT_SHA,
      },
      tensors: [{
        name: "test.weight",
        shape: [1],
        offset: 0,
        byteLength: 2,
        dtype: "f16",
        layout: "row-major",
      }],
    };
    const requests: Array<{ readonly url: string; readonly cache: RequestCache | undefined }> = [];
    const fetchMock = vi.fn(async (input: URL | RequestInfo, init?: RequestInit) => {
      requests.push({ url: String(input), cache: init?.cache });
      if (requests.length === 1) {
        return new Response(JSON.stringify(manifest), {
          status: 200,
          headers: { "content-type": "application/json" },
        });
      }
      return new Response(new Uint8Array([0, 0]), { status: 200 });
    });
    const destroy = vi.fn();
    const writeBuffer = vi.fn();
    const device = {
      createBuffer: vi.fn(() => ({ destroy })),
      queue: { writeBuffer },
    } as unknown as GPUDevice;
    vi.stubGlobal("fetch", fetchMock);
    vi.stubGlobal("GPUBufferUsage", { STORAGE: 1, COPY_SRC: 2, COPY_DST: 4 });

    const loaded = await loadGpuWeightPackage(
      device,
      "https://example.test/model/manifest.json",
    );

    expect(requests).toEqual([
      { url: "https://example.test/model/manifest.json", cache: "no-store" },
      {
        url: `https://example.test/model/weights.f16.bin?sha256=${WEIGHT_SHA}`,
        cache: "force-cache",
      },
    ]);
    expect(writeBuffer).toHaveBeenCalledOnce();
    loaded.destroy();
    expect(destroy).toHaveBeenCalledOnce();
  });
});
