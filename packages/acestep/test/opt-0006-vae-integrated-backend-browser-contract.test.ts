import { readFile } from "node:fs/promises";
import { describe, expect, it } from "vitest";

describe("OPT-0006 integrated browser backend gate contract", () => {
  it("runs independent portable-1, optimized-1, and production optimized-8 paths", async () => {
    const source = await integratedGateSource();
    expect(source).toContain('runProfile("portable", 1)');
    expect(source).toContain('runProfile("optimized-when-eligible", 1)');
    expect(source).toContain('runProfile("optimized-when-eligible", 8)');
    expect(source).toContain("AceVaeChunkGpuBackend.fromPreparedResources");
    expect(source).toContain("planAceVaeChunkGpuBackendMemory(");
    expect(source).toContain("decoderQuantaPerCommandBuffer: batchSize");
    expect(source).toContain("poisonedOutput.fill(0x7fc0_0000)");
    expect(source).toContain("optimizedBits[index] !== optimizedProductionBits[index]");
    expect(source).toContain("portableBits[index] !== optimizedBits[index]");
    expect(source).not.toContain("for (const quantum of dispatch.quanta)");
  });

  it("pins logical work and exact backend scheduling telemetry", async () => {
    const source = await integratedGateSource();
    expect(source).toContain(
      "dispatch.quanta.length !== 109 || dispatch.primitiveCount !== 115",
    );
    expect(source).toContain("optimized.commandBufferCount !== 110");
    expect(source).toContain("optimized.queueDrainCount !== 110");
    expect(source).toContain("optimized.idleCount !== 109");
    expect(source).toContain("optimizedProduction.commandBufferCount !== 15");
    expect(source).toContain("optimizedProduction.queueDrainCount !== 15");
    expect(source).toContain("optimizedProduction.idleCount !== 14");
    expect(source).toContain("optimized.progress.length !== 110");
    expect(source).toContain("optimizedProduction.progress.length !== 110");
    expect(source).toContain("event.completedDecoderQuanta !== index + 1");
    expect(source).toContain("event.completedCommandBuffers !== completedBatch");
    expect(source).toContain('readback.stage !== "readback"');
  });
});

async function integratedGateSource(): Promise<string> {
  const file = await readFile(new URL(
    "./browser/vae-decoder-correctness.ts",
    import.meta.url,
  ), "utf8");
  return file.slice(
    file.indexOf("async function runIntegratedOptimizedDecoderAb"),
    file.indexOf("async function preflightShaders"),
  );
}
