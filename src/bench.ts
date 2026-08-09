// In-browser GEMM bench at the encoder's real shapes — pins down kernel truth
// on actual Chrome/Metal (dawn-node numbers can differ). Mirrors
// /tmp scripts/gemm bench methodology: warmup, then N iters between
// onSubmittedWorkDone fences.
import { GpuContext, requestGpuDevice } from "./gpu/compute.js";

const out = document.getElementById("out")!;
const log = (s: string) => (out.textContent += s + "\n");

async function bench(ctx: any, dev: any, label: string, M: number, K: number, N: number, f16 = true) {
  const rand = (n: number) => Float32Array.from({ length: n }, () => (Math.random() * 2 - 1) * 0.05);
  const a = ctx.upload(rand(M * K), M, K);
  const b = f16 ? ctx.uploadF16(rand(K * N), K, N) : ctx.upload(rand(K * N), K, N);
  const fn = () => (ctx as any).matmul(a, b);
  for (let w = 0; w < 8; w++) fn();
  await dev.queue.onSubmittedWorkDone();
  const iters = 60;
  const t0 = performance.now();
  for (let i = 0; i < iters; i++) fn();
  await dev.queue.onSubmittedWorkDone();
  const ms = (performance.now() - t0) / iters;
  const tf = (2 * M * K * N) / (ms / 1000) / 1e12;
  log(`${label.padEnd(30)} ${ms.toFixed(2).padStart(7)}ms  ${tf.toFixed(2)} TFLOP/s`);
  (ctx as any).trimPool?.(0);
}

document.getElementById("go")!.addEventListener("click", async () => {
  out.textContent = "";
  try {
    const dev = await requestGpuDevice();
    if (!dev) {
      log("WebGPU unavailable.");
      return;
    }
    const ctx: any = new GpuContext(dev);
    await (ctx as any).probeSubgroups?.();
    const adapter = await (navigator as any).gpu?.requestAdapter();
    const info = (adapter as any)?.info ?? {};
    log(`ua: ${navigator.userAgent}`);
    log(`adapter: ${info.vendor ?? "?"} ${info.architecture ?? ""} ${info.device ?? ""}`);
    log(`f16: ${(ctx as any).hasF16 ? "yes" : "no"} · subgroups32: ${(ctx as any).hasSubgroups32 ? "yes" : "no"}\n`);
    log("— shipping kernel (LDS-staged v4/f16C) —");
    await bench(ctx, dev, "FFN1  1128x1024x4096 f16B", 1128, 1024, 4096);
    await bench(ctx, dev, "FFN2  1128x4096x1024 f16B", 1128, 4096, 1024);
    await bench(ctx, dev, "QKV   1128x1024x1024 f16B", 1128, 1024, 1024);
    await bench(ctx, dev, "FFN1  1128x1024x4096 f32B", 1128, 1024, 4096, false);
    await bench(ctx, dev, "FFN1  7520x1024x4096 f16B", 7520, 1024, 4096);
    log("\n— subgroup tile-major path (opt-in) —");
    (globalThis as any).__sgGemm = true;
    await bench(ctx, dev, "FFN1sg 1128x1024x4096", 1128, 1024, 4096);
    await bench(ctx, dev, "FFN1sg 7520x1024x4096", 7520, 1024, 4096);
    (globalThis as any).__sgGemm = false;
    log("\ndone — copy this block.");
    dev.destroy();
  } catch (err) {
    log(String(err));
  }
});
