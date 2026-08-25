import assert from "node:assert/strict";
import test from "node:test";

import { formatLoadProgress } from "../src/core/loadProgress.js";

test("labels streamed model bytes as a download with byte progress", () => {
  assert.equal(
    formatLoadProgress({
      file: "parakeet/encoder-int8.bin",
      phase: "download",
      loaded: 195_743_171,
      total: 611_697_408,
      fraction: 0.32,
    }),
    "Downloading parakeet/encoder-int8.bin — 32% (196 / 612 MB)",
  );
});

test("keeps local initialization labeled as loading", () => {
  assert.equal(formatLoadProgress({ file: "WebGPU pipelines", phase: "load", loaded: 1, total: 4, fraction: 0.25 }), "Loading WebGPU pipelines — 25%");
});

test("supports existing progress producers without a phase", () => {
  assert.equal(formatLoadProgress({ file: "model", loaded: 1, total: 1, fraction: 1 }), "Loading model — 100%");
});

test("omits byte totals when the server does not provide one", () => {
  assert.equal(formatLoadProgress({ file: "weights.bin", phase: "download", loaded: 1024, total: 0, fraction: 0 }), "Downloading weights.bin — 0%");
});
