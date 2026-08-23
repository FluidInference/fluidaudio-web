import { readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

const HARNESS_SOURCE = readFileSync(new URL(
  "./browser/opt-0043-webgpu-timestamp-utilization-profile.ts",
  import.meta.url,
), "utf8");
const HARNESS_HTML = readFileSync(new URL(
  "./browser/opt-0043-webgpu-timestamp-utilization-profile.html",
  import.meta.url,
), "utf8");

describe("OPT-0043 WebGPU timestamp utilization profile", () => {
  it("reuses the four production shapes and the complete OPT-0032 gates", () => {
    for (const shape of [
      'fullSpec("h-h", 2_048, 2_048, 4, 0)',
      'fullSpec("h-1024", 2_048, 1_024, 2, 1)',
      'fullSpec("h-6144", 2_048, 6_144, 2, 2)',
      'fullSpec("6144-h", 6_144, 2_048, 1, 3)',
    ]) expect(HARNESS_SOURCE).toContain(shape);
    for (const gate of [
      "FULL_OUTPUT_COUNT = 25_344_000",
      "OUTPUT_PREFILL_QNAN_U32",
      "candidateDeterministicRawU32: true",
      "prefixCanaryIntact",
      "candidateNonFiniteCount",
      "nrmse",
      "snrDecibels",
      "pearsonCorrelation",
      "completedBeforeReady: true",
    ]) expect(HARNESS_SOURCE).toContain(gate);
    expect(HARNESS_SOURCE).toContain(
      'fixtureVersion: "opt0032-full-and-adversarial-fp16-v1"',
    );
  });

  it("uses standard pass timestamps and keeps timing behind one button", () => {
    expect(HARNESS_SOURCE).toContain(
      'requiredFeatures: ["shader-f16", "subgroups", "timestamp-query"]',
    );
    expect(HARNESS_SOURCE).toContain('type: "timestamp"');
    expect(HARNESS_SOURCE).toContain("timestampWrites: {");
    expect(HARNESS_SOURCE).toContain("beginningOfPassWriteIndex: 0");
    expect(HARNESS_SOURCE).toContain("endOfPassWriteIndex: 1");
    expect(HARNESS_SOURCE).toContain("encoder.resolveQuerySet(");
    expect(HARNESS_SOURCE).toContain("encoder.copyBufferToBuffer(");
    expect(HARNESS_SOURCE).toContain("await device.queue.onSubmittedWorkDone()");
    expect(HARNESS_SOURCE).toContain("await timestampReadback.mapAsync(");
    expect(HARNESS_SOURCE.indexOf("pass.end();")).toBeLessThan(
      HARNESS_SOURCE.indexOf("encoder.resolveQuerySet("),
    );
    expect(HARNESS_SOURCE.indexOf("await device.queue.onSubmittedWorkDone();",
      HARNESS_SOURCE.indexOf("async function timeDispatch"))).toBeLessThan(
      HARNESS_SOURCE.indexOf("await timestampReadback.mapAsync("),
    );
    expect(HARNESS_HTML).toContain('id="run" type="button" disabled');
    expect(HARNESS_SOURCE).toContain('runButton.addEventListener("click"');
    expect(HARNESS_SOURCE).toContain(
      "GPU timestamp timing has not run",
    );
  });

  it("reports balanced wall/GPU ratios and both throughput conventions", () => {
    expect(HARNESS_SOURCE).toContain(
      'shapeOrder: Object.freeze([0, 1, 2, 3])',
    );
    expect(HARNESS_SOURCE).toContain(
      'shapeOrder: Object.freeze([3, 2, 1, 0])',
    );
    expect(HARNESS_SOURCE).toContain(
      'armOrder: Object.freeze(["control", "candidate"]',
    );
    expect(HARNESS_SOURCE).toContain(
      'armOrder: Object.freeze(["candidate", "control"]',
    );
    for (const metric of [
      "gpuElapsedNanoseconds",
      "gpuMilliseconds",
      "wallMilliseconds",
      "gpuToWallRatio",
      "validGpuTflops",
      "scheduledGpuTflops",
      "validWallTflops",
      "scheduledWallTflops",
    ]) expect(HARNESS_SOURCE).toContain(metric);
    expect(HARNESS_SOURCE).toContain(
      'arithmeticConvention: "two FLOPs per multiply-add"',
    );
    expect(HARNESS_SOURCE).toContain(
      'disposition: "diagnostic-only-no-integration-authority"',
    );
    expect(HARNESS_SOURCE).toContain("productionIntegrationAuthorized: false");
  });

  it("defines the external AGX correlation boundary without sampling in-page", () => {
    expect(HARNESS_SOURCE).toContain("externalAgxUtilizationSampling");
    expect(HARNESS_SOURCE).toContain("capturedByPage: false");
    expect(HARNESS_SOURCE).toContain(
      'placement: "outside every browser timed interval"',
    );
    expect(HARNESS_SOURCE).toContain("submitAtEpochMilliseconds");
    expect(HARNESS_SOURCE).toContain("fenceAtEpochMilliseconds");
    expect(HARNESS_HTML).toContain("external coarse AGX utilization logger");
    expect(HARNESS_SOURCE).not.toContain("ioreg -r");
  });
});
