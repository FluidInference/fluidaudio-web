import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import { planAceFixed32TiledFullAttention } from
  "../src/webgpu/kernels/attention.js";

const HARNESS_PATH = fileURLToPath(new URL(
  "./browser/dit-full-attention-query8-ab.ts",
  import.meta.url,
));
const HTML_PATH = fileURLToPath(new URL(
  "./browser/dit-full-attention-query8-ab.html",
  import.meta.url,
));
const HARNESS_SOURCE = readFileSync(HARNESS_PATH, "utf8");
const HTML_SOURCE = readFileSync(HTML_PATH, "utf8");

describe("production M2250 portable/query8 attention browser gate", () => {
  it("pins the exact three-minute full-attention geometry and traffic", () => {
    const plan = planAceFixed32TiledFullAttention({
      batch: 1,
      queryHeads: 16,
      keyValueHeads: 8,
      queryTokens: 2_250,
      keyValueTokens: 2_250,
      headDimension: 128,
      mode: "full",
    });
    expect(plan).toMatchObject({
      outputElements: 4_608_000,
      workgroupCount: 4_504,
      outputRangeCount: 5,
      portableKeyValueScalarLoads: 20_736_000_000,
      tiledKeyValueScalarLoads: 2_594_304_000,
      portableBarriersPerKey: 10,
      tiledBarriersPerKey: 2,
    });
    expect(plan.outputRanges.map((range) => range.workgroupCount)).toEqual([
      932,
      932,
      932,
      932,
      776,
    ]);
    expect(HARNESS_SOURCE).toContain("queryTokens: 2_250");
    expect(HARNESS_SOURCE).toContain("keyValueTokens: 2_250");
    expect(HARNESS_SOURCE).not.toContain("Math.random");
    expect(HARNESS_SOURCE).toContain("lcg1664525-u24-f32-v1");
  });

  it("compiles and warmups before one externally gated portable-query8 order", () => {
    const compile = HARNESS_SOURCE.indexOf("await Promise.all([");
    const warmup = HARNESS_SOURCE.indexOf("await executeOnly(device, portableDispatch");
    const warmupCompleted = HARNESS_SOURCE.indexOf(
      "const warmupCompletedAtEpochMilliseconds = Date.now()",
    );
    const preparedReturn = HARNESS_SOURCE.indexOf(
      "return Object.freeze({",
      warmupCompleted,
    );
    const portable = HARNESS_SOURCE.indexOf(
      "const portable = await executeAndRead(",
    );
    const query8 = HARNESS_SOURCE.indexOf(
      "const query8 = await executeAndRead(",
    );
    expect(compile).toBeGreaterThan(0);
    expect(warmup).toBeGreaterThan(compile);
    expect(warmupCompleted).toBeGreaterThan(warmup);
    expect(preparedReturn).toBeGreaterThan(warmupCompleted);
    expect(HARNESS_SOURCE).toContain("void prepareGate().then(");
    expect(HARNESS_SOURCE).toContain('document.body.dataset.status = "ready"');
    expect(portable).toBeGreaterThan(preparedReturn);
    expect(query8).toBeGreaterThan(portable);
    expect(HARNESS_SOURCE).toContain(
      'armOrder: ["portable", "fixed32-subgroup-query8"]',
    );
    expect(HARNESS_SOURCE).toContain("samplesPerArm: 1");
    expect(HARNESS_SOURCE).toContain("unchangedThermalRetryPerformed: false");
    expect(HTML_SOURCE).toContain("Run portable → query8 once");
  });

  it("requires one post-warmup 30-second nominal interval", () => {
    expect(HARNESS_SOURCE).toContain(
      '"notifyutil-com.apple.system.thermalpressurelevel"',
    );
    expect(HARNESS_SOURCE).toContain(
      "const MINIMUM_NOMINAL_MILLISECONDS = 30_000",
    );
    expect(HARNESS_SOURCE).toContain(
      "startedAtEpochMilliseconds < warmupCompletedAtEpochMilliseconds",
    );
    expect(HARNESS_SOURCE).toContain("nonNominalObservationCount !== 0");
    expect(HARNESS_SOURCE).toContain(
      "maximumPollGapMilliseconds > MAXIMUM_THERMAL_POLL_GAP_MILLISECONDS",
    );
    expect(HTML_SOURCE).toContain("continuous 30-second nominal");
  });

  it("times through drain and full readback, then compares every F32", () => {
    expect(HARNESS_SOURCE).toContain("await device.queue.onSubmittedWorkDone()");
    expect(HARNESS_SOURCE).toContain("readbackEncoder.copyBufferToBuffer(");
    expect(HARNESS_SOURCE).toContain("await readback.mapAsync(GPUMapMode.READ)");
    expect(HARNESS_SOURCE).toContain(
      "new Float32Array(readback.getMappedRange().slice(0))",
    );
    expect(HARNESS_SOURCE).toContain(
      "for (let index = 0; index < portable.length; index += 1)",
    );
    expect(HARNESS_SOURCE).toContain("portableNonFiniteCount");
    expect(HARNESS_SOURCE).toContain("query8NonFiniteCount");
    expect(HARNESS_SOURCE).toContain("maximumAbsoluteError");
    expect(HARNESS_SOURCE).toContain("nrmse");
    expect(HARNESS_SOURCE).toContain("comparedEveryOutputF32: true");
    expect(HARNESS_SOURCE).not.toContain("timestamp-query");
  });

  it("records source, inputs, capabilities, traffic, hashes, and cleanup", () => {
    expect(HARNESS_SOURCE).toContain(
      '"a88e1b41c7b127d20c3fc4dbdee63acb77612a8c"',
    );
    expect(HARNESS_SOURCE).toContain(
      '"5f64e5148ee60f26023faeb99ac72b46354086f3db48f7778800a9061d2b9ed3"',
    );
    expect(HARNESS_SOURCE).toContain("portableWgslSha256");
    expect(HARNESS_SOURCE).toContain("query8WgslSha256");
    expect(HARNESS_SOURCE).toContain("adapterInfo:");
    expect(HARNESS_SOURCE).toContain("portableKeyValueBytes");
    expect(HARNESS_SOURCE).toContain("query8KeyValueBytes");
    expect(HARNESS_SOURCE).toContain("portableOutputSha256");
    expect(HARNESS_SOURCE).toContain("query8OutputSha256");
    expect(HARNESS_SOURCE).toContain("for (const buffer of ownedBuffers) buffer.destroy()");
    expect(HARNESS_SOURCE).toContain("device.destroy()");
    expect(HTML_SOURCE).toContain(
      '<script type="module" src="./dit-full-attention-query8-ab.ts"></script>',
    );
  });
});
