import { readFile } from "node:fs/promises";
import { describe, expect, it } from "vitest";

describe("OPT-0006 browser coalescing contract", () => {
  it("pins the complete mixed decoder and every declared batch size", async () => {
    const source = await readFile(
      new URL("./browser/opt-0006-vae-command-buffer-coalescing-ab.ts", import.meta.url),
      "utf8",
    );
    expect(source).toContain("decoderInputChannels: 136");
    expect(source).toContain("decoderChannels: 128");
    expect(source).toContain("maximumOutputElements: 128");
    expect(source).toContain("dispatch.quanta.length !== 109");
    expect(source).toContain("dispatch.primitiveCount !== 115");
    expect(source).toContain("ACE_OPT_0006_QUANTA_PER_COMMAND_BUFFER_CANDIDATES");
    expect(source).toContain("runAceOpt0006QuantumBatches");
    expect(source).toContain("requireBitIdentity");
    expect(source).toContain("finalCommandBufferRemains: true");
    expect(source).toContain("runCancellationProof");
    expect(source).toContain("maximumAnimationFrameGapMilliseconds");
    expect(source).toContain("productionWindowExecuted: false");
    expect(source).toContain("songExecuted: false");
  });

  it("keeps the page explicit and user-triggered", async () => {
    const html = await readFile(
      new URL("./browser/opt-0006-vae-command-buffer-coalescing-ab.html", import.meta.url),
      "utf8",
    );
    expect(html).toContain("Run bounded VAE coalescing A/B");
    expect(html).toContain("opt-0006-vae-command-buffer-coalescing-ab.ts");
    expect(html).toContain('data-status="ready"');
  });
});
