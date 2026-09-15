import { afterEach, expect, it, vi } from "vitest";
import { loadKokoroBackend } from "../src/engines/tts-kokoro/synth-backend.js";
import { WasmContext } from "../src/gpu/wasm-context.js";

afterEach(() => vi.restoreAllMocks());

it("destroys each real WASM context after a download failure, including retries", async () => {
  // Real compute context; the download fails before any model data is returned.
  const destroy = vi.spyOn(WasmContext.prototype, "destroy");
  const failure = new Error("Download interrupted");
  const download = vi.fn(async () => {
    throw failure;
  });
  for (let attempt = 1; attempt <= 2; attempt++) {
    await expect(loadKokoroBackend(download, (_repo, path) => path, {})).rejects.toBe(failure);
    expect(destroy).toHaveBeenCalledTimes(attempt);
  }
  expect(destroy.mock.instances[0]).toBeInstanceOf(WasmContext);
  expect(destroy.mock.instances[0]).not.toBe(destroy.mock.instances[1]);
});
