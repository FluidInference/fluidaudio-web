import { describe, expect, it } from "vitest";

import {
  OPT_0024_C512_FIXTURE_SHA256,
  OPT_0024_C512_FRAMES,
  OPT_0024_C512_INPUT_ELEMENTS,
  OPT_0024_C512_K7_CANDIDATE_QUANTA,
  OPT_0024_C512_K7_PURE_BATCHES,
  OPT_0024_C512_K7_PURE_QUANTA,
  OPT_0024_C512_K7_TOTAL_QUANTA,
  OPT_0024_C512_OUTPUT_BYTES,
  OPT_0024_C512_OUTPUT_ELEMENTS,
  OPT_0024_C512_SCHEMA,
  OPT_0024_C512_TIMED_ORDER,
  OPT_0024_C512_WARMUP_ORDER,
  classifyOpt0024C512Batch,
  compareOpt0024C512Waveforms,
} from "./browser/opt-0024-vae-k7-c512-subsystem.js";

describe("OPT-0024 C512 subsystem harness", () => {
  it("pins the exact production C512 geometry and balanced sequence", () => {
    expect(OPT_0024_C512_SCHEMA).toBe(
      "ace-opt-0024-vae-k7-c512-subsystem-v1",
    );
    expect(OPT_0024_C512_FRAMES).toBe(512);
    expect(OPT_0024_C512_INPUT_ELEMENTS).toBe(32_768);
    expect(OPT_0024_C512_OUTPUT_ELEMENTS).toBe(1_966_080);
    expect(OPT_0024_C512_OUTPUT_BYTES).toBe(7_864_320);
    expect(OPT_0024_C512_K7_PURE_BATCHES).toBe(500);
    expect(OPT_0024_C512_K7_PURE_QUANTA).toBe(3_999);
    expect(OPT_0024_C512_K7_CANDIDATE_QUANTA).toBe(4_082);
    expect(OPT_0024_C512_K7_TOTAL_QUANTA).toBe(4_090);
    expect(OPT_0024_C512_FIXTURE_SHA256).toBe(
      "eff0005ae48353fbc0a9ec86a5b2824b49e6fff6e899ea89af7d1c6e5870e899",
    );
    expect(OPT_0024_C512_WARMUP_ORDER).toEqual(["shipped", "candidate"]);
    expect(OPT_0024_C512_TIMED_ORDER).toEqual([
      "shipped",
      "candidate",
      "candidate",
      "shipped",
    ]);
  });

  it("classifies homogeneous and mixed production batches", () => {
    const operations = [
      { kind: "conv1d", shape: { kernelSize: 7 } },
      { kind: "conv1d", shape: { kernelSize: 1 } },
      { kind: "snake" },
    ];
    expect(classifyOpt0024C512Batch([
      { operationIndex: 0, operationKind: "conv1d" },
      { operationIndex: 0, operationKind: "conv1d" },
    ], operations)).toBe("k7-conv1d");
    expect(classifyOpt0024C512Batch([
      { operationIndex: 0, operationKind: "conv1d" },
      { operationIndex: 1, operationKind: "conv1d" },
    ], operations)).toBe("mixed");
    expect(classifyOpt0024C512Batch([
      { operationIndex: null, operationKind: "ingress-cast" },
    ], operations)).toBe("mixed");
    expect(classifyOpt0024C512Batch([
      { operationIndex: 2, operationKind: "snake" },
    ], operations)).toBe("snake");
  });

  it("computes the frozen waveform envelope for joint and channel views", () => {
    const control = new Float32Array([1, -1, 0.5, -0.5, 0.25, -0.25]);
    const exact = compareOpt0024C512Waveforms(control, control);
    expect(exact).toMatchObject({
      count: 6,
      nrmse: 0,
      snrDb: Number.POSITIVE_INFINITY,
      pearson: 1,
      maximumAbsoluteError: 0,
      finite: true,
      passed: true,
    });
    const left = compareOpt0024C512Waveforms(control, control, 2, 0);
    expect(left.count).toBe(3);
    expect(left.passed).toBe(true);
    const bad = new Float32Array(control);
    bad[0] = 0;
    expect(compareOpt0024C512Waveforms(control, bad).passed).toBe(false);
  });

  it("keeps preparation and timed actions physically separate in the page", async () => {
    const [html, source] = await Promise.all([
      import("./browser/opt-0024-vae-k7-c512-subsystem.html?raw")
        .then((module) => module.default as string),
      import("./browser/opt-0024-vae-k7-c512-subsystem.ts?raw")
        .then((module) => module.default as string),
    ]);
    expect(html).toContain('id="prepare"');
    expect(html).toContain('id="run" type="button" disabled');
    expect(source).toContain('type: "prepared"');
    expect(source).toContain('type: "run"');
    expect(source).toContain("submitAceCommandBufferFactoriesCooperatively");
    expect(source).toContain("onCommandBufferDrained");
    expect(source).toContain("externalThermalGateRequiredBeforeTimedAction: true");
    expect(source).toContain("experimentalBrowserFlags: false");
  });
});
