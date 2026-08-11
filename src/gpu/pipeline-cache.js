// Compiled compute-pipeline cache: one pipeline per (key) — shader modules are
// compiled once and reused across every dispatch. Labels feed the profiler.
export class PipelineCache {
  /** @param {GPUDevice} device */
  constructor(device) {
    this.device = device;
    this.pipelines = new Map();
  }

  get(key, code, entry = "main") {
    let p = this.pipelines.get(key);
    if (!p) {
      const module = this.device.createShaderModule({ code });
      p = this.device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: entry } });
      p.__label = key; // for the profiler
      this.pipelines.set(key, p);
    }
    return p;
  }
}
