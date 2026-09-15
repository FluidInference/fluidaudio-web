import { describe, expect, it } from "vitest";

import {
  DEFERRED_VAE_CACHE_PHYSICAL_BYTES,
  INT8_MODEL_DOWNLOAD_TOTAL_BYTES,
  initialModelDownloadProgress,
  updateModelDownloadProgress,
} from "../src/engines/musicgen-acestep/model-download-progress.js";

describe("music model download progress", () => {
  it("tracks the packed INT8 total through initialization and deferred VAE", () => {
    const initializationBytes = INT8_MODEL_DOWNLOAD_TOTAL_BYTES - DEFERRED_VAE_CACHE_PHYSICAL_BYTES;
    const initial = initialModelDownloadProgress(INT8_MODEL_DOWNLOAD_TOTAL_BYTES);
    const initialized = updateModelDownloadProgress(initial, {
      stage: "weights",
      unit: "bytes",
      completedUnits: initializationBytes,
      totalUnits: initializationBytes,
    });
    expect(initialized).toMatchObject({
      completed: initializationBytes,
      total: INT8_MODEL_DOWNLOAD_TOTAL_BYTES,
    });

    const complete = updateModelDownloadProgress(initialized, {
      stage: "vae-load",
      message: `network: complete ${DEFERRED_VAE_CACHE_PHYSICAL_BYTES}/${DEFERRED_VAE_CACHE_PHYSICAL_BYTES} bytes`,
    });
    expect(complete).toMatchObject({
      completed: INT8_MODEL_DOWNLOAD_TOTAL_BYTES,
      total: INT8_MODEL_DOWNLOAD_TOTAL_BYTES,
      fraction: 1,
    });
  });
});
