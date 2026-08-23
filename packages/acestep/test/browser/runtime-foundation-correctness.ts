import { AceGpuArena } from "../../src/webgpu/arena.js";
import { AceUniformPool } from "../../src/webgpu/uniform-pool.js";

const output = document.querySelector<HTMLPreElement>("#output")!;

void run().catch((error: unknown) => {
  document.body.dataset.status = "failed";
  output.textContent = error instanceof Error ? error.stack ?? error.message : String(error);
});

async function run(): Promise<void> {
  if (navigator.gpu === undefined) throw new Error("WebGPU is unavailable");
  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: "high-performance",
  });
  if (adapter === null) throw new Error("No WebGPU adapter is available");
  const device = await adapter.requestDevice();
  const arena = await AceGpuArena.create(device, [
    { label: "runtime-foundation-arena", byteLength: 512 },
  ]);
  const first = arena.slice("first", 0, 0, 256);
  const second = arena.slice("second", 0, 256, 256);
  arena.assertNoWritableOverlap([first], [second]);

  const uniforms = await AceUniformPool.create(device, 512);
  uniforms.beginQuantum();
  const initial = uniforms.write("initial", Uint32Array.of(7, 11, 13, 17));
  const commandBuffer = device.createCommandEncoder({
    label: "runtime-foundation-empty-quantum",
  }).finish();
  await uniforms.submitQuantum(commandBuffer);
  uniforms.beginQuantum();
  const recycled = uniforms.write("recycled", Uint32Array.of(19));
  if (initial.byteOffset !== 0 || recycled.byteOffset !== 0) {
    throw new Error("Uniform slots were not recycled only after queue drain");
  }

  uniforms.destroy();
  arena.destroy();
  device.destroy();
  document.body.dataset.status = "passed";
  output.textContent = "PASS: scoped allocations and submit/drain ownership";
}
