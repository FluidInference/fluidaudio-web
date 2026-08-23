import { describe, expect, it } from "vitest";
import pipelineSource from "../src/runtime/webgpu-pipeline.ts?raw";
import backendSource from "../src/webgpu/dit-backend.ts?raw";
import graphSource from "../src/webgpu/dit-graph.ts?raw";
import experimentSource from
  "../optimization/experiments/OPT-0067-dit-quad-query-evaluation-slice-thermal-screen.md?raw";
import workerSource from
  "./browser/opt-0067-dit-quad-query-evaluation-slice-worker.ts?raw";
import pageSource from
  "./browser/opt-0067-dit-quad-query-evaluation-slice.ts?raw";
import htmlSource from
  "./browser/opt-0067-dit-quad-query-evaluation-slice.html?raw";
import {
  OPT_0067_ARM_ORDER,
  OPT_0067_EVALUATION0_SHA256,
  OPT_0067_GRAPH_COMMAND_BUFFERS,
  OPT_0067_TOTAL_COMMAND_BUFFERS,
  exactOpt0067ResultIdentity,
  requireOpt0067ThermalGate,
  requireOpt0067ThermalTrace,
  summarizeOpt0067Performance,
  type Opt0067ThermalGate,
  type Opt0067ThermalTrace,
  type Opt0067TimingSample,
} from "./browser/opt-0067-dit-quad-query-evaluation-slice-contract.js";

describe("OPT-0067 thermally isolated evaluation-0 screen", () => {
  it("freezes the exact ABBA inventory, result hash, and 341+1 topology", () => {
    expect(OPT_0067_ARM_ORDER).toEqual([
      { armId: "A1", order: 0, owner: "query8" },
      { armId: "B1", order: 1, owner: "quad" },
      { armId: "B2", order: 2, owner: "quad" },
      { armId: "A2", order: 3, owner: "query8" },
    ]);
    expect(OPT_0067_EVALUATION0_SHA256).toBe(
      "d7f4280fdc43a038728df167f02819c35d99dac812347731d2fb8ac421a36286",
    );
    expect(OPT_0067_GRAPH_COMMAND_BUFFERS).toBe(25 + 316);
    expect(OPT_0067_TOTAL_COMMAND_BUFFERS).toBe(342);
    expect(experimentSource).toContain("`55,296,000` raw-U32");
    expect(experimentSource).toContain("`288,000` words");
    expect(experimentSource).toContain("`8 * (((A1_eval - B1_eval)");
  });

  it("accepts only a fresh nominal gate and its continuous through-cleanup trace", () => {
    const gate: Opt0067ThermalGate = Object.freeze({
      source: "notifyutil-com.apple.system.thermalpressurelevel",
      command: "notifyutil -g com.apple.system.thermalpressurelevel",
      startedAtEpochMilliseconds: 2_000,
      completedAtEpochMilliseconds: 33_000,
      observationCount: 32,
      maximumPollGapMilliseconds: 1_000,
      nonNominalObservationCount: 0,
      observations: Object.freeze(Array.from({ length: 32 }, (_, index) =>
        Object.freeze({
          atEpochMilliseconds: 2_000 + index * 1_000,
          level: 0,
          rawValue: "0",
        })
      )),
    });
    const acceptedGate = requireOpt0067ThermalGate(gate, 1_000, 33_500);
    const trace: Opt0067ThermalTrace = Object.freeze({
      source: gate.source,
      command: gate.command,
      rawTraceSha256: "a".repeat(64),
      completedAtEpochMilliseconds: 40_000,
      observationCount: 39,
      maximumPollGapMilliseconds: 1_000,
      nonNominalObservationCount: 3,
      observations: Object.freeze(Array.from({ length: 39 }, (_, index) => {
        const level = index >= 33 && index <= 35 ? 1 : 0;
        return Object.freeze({
          atEpochMilliseconds: 2_000 + index * 1_000,
          level,
          rawValue: String(level),
        });
      })),
      transitions: Object.freeze([
        Object.freeze({ atEpochMilliseconds: 33_000, level: 0 }),
        Object.freeze({ atEpochMilliseconds: 35_000, level: 1 }),
      ]),
    });
    expect(requireOpt0067ThermalTrace(
      trace,
      acceptedGate,
      39_500,
      40_200,
    )).toEqual(trace);
    expect(trace.observations).toHaveLength(trace.observationCount);
    expect(() => requireOpt0067ThermalGate(
      { ...gate, startedAtEpochMilliseconds: 500 },
      1_000,
      33_500,
    )).toThrow(/thermal/);
    expect(() => requireOpt0067ThermalTrace(
      { ...trace, completedAtEpochMilliseconds: 39_000 },
      acceptedGate,
      39_500,
      40_200,
    )).toThrow(/thermal|through-cleanup/);
  });

  it("requires both directional wins, 1.30x full-self, 3 s projection, and 2% isolation", () => {
    const sample = (
      armId: "A1" | "B1" | "B2" | "A2",
      order: 0 | 1 | 2 | 3,
      owner: "query8" | "quad",
      evaluationWallMs: number,
      fullSelfMs: number,
      feedForwardMs: number,
    ): Opt0067TimingSample => Object.freeze({
      armId,
      order,
      owner,
      fullSelfMs,
      evaluationWallMs,
      nonFullSelfEvaluationWallMs: evaluationWallMs - fullSelfMs,
      graphWallMs: evaluationWallMs + 150,
      commandDrainMs: evaluationWallMs - 200,
      requestedIdleMs: 341,
      readbackMs: 3,
      residualMs: 50,
      familyMs: Object.freeze({
        "self-full": fullSelfMs,
        "feed-forward": feedForwardMs,
        "cross-attention": 0,
      }),
    });
    const passingSamples = [
      sample("A1", 0, "query8", 4_000, 1_400, 1_000),
      sample("B1", 1, "quad", 3_500, 900, 1_010),
      sample("B2", 2, "quad", 3_550, 910, 1_010),
      sample("A2", 3, "query8", 4_100, 1_410, 1_000),
    ] as const;
    const passed = summarizeOpt0067Performance(passingSamples);
    expect(passed).toMatchObject({
      forwardFullSelfImproved: true,
      reverseFullSelfImproved: true,
      forwardEvaluationImproved: true,
      reverseEvaluationImproved: true,
      projectedEightEvaluationSavingMs: 4_200,
      passed: true,
    });
    expect(passed.maximumNonFullSelfRegression).toBeCloseTo(0.01, 12);
    expect(passed.aggregateFullSelfSpeedup).toBeGreaterThan(1.30);
    expect(passed.nonFullSelfAbsoluteDeltasMs).toMatchObject({
      "forward:feed-forward": 10,
      "reverse:feed-forward": 10,
    });
    expect(summarizeOpt0067Performance([
      passingSamples[0],
      { ...passingSamples[1], familyMs: { ...passingSamples[1].familyMs,
        "feed-forward": 1_021 } },
      passingSamples[2],
      passingSamples[3],
    ]).passed).toBe(false);
    expect(() => summarizeOpt0067Performance([
      passingSamples[1],
      passingSamples[0],
      passingSamples[2],
      passingSamples[3],
    ])).toThrow(/ABBA sample inventory/);
  });

  it("compares all 288,000 evaluation words as raw U32", () => {
    const left = new Float32Array(288_000);
    left[0] = 1;
    left[287_999] = -0;
    const right = left.slice();
    expect(exactOpt0067ResultIdentity(left, right)).toBe(true);
    new Uint32Array(right.buffer)[287_999] = 0;
    expect(exactOpt0067ResultIdentity(left, right)).toBe(false);
    expect(exactOpt0067ResultIdentity(left, new Float32Array(1))).toBe(false);
  });

  it("stops drained before evaluation 1 and uses the ordinary result readback", () => {
    expect(graphSource).toContain(
      "ACE_OPT_0067_M2250_EVALUATION0_GRAPH_COMMAND_BUFFER_COUNT = 341",
    );
    expect(graphSource).toContain(
      'throw new Error("OPT-0067 did not stop exactly before evaluation 1")',
    );
    expect(graphSource).toContain(
      "finalLatent = this.compiled.evaluationLatents[evaluationTarget - 1]",
    );
    expect(backendSource).toContain(
      'schema: "ace-dit-opt0067-evaluation0-command-profile-v1"',
    );
    expect(backendSource).toContain(
      "evaluationReadbacks !== undefined",
    );
    expect(backendSource).toContain(
      "opt0067EvaluationLimit: 1 as const",
    );
    expect(pipelineSource).toContain(
      'schema: "ace-dit-opt0067-m2250-evaluation0-checkpoint-v1"',
    );
    expect(pipelineSource).toContain(
      "evaluationResultExtraCommandBufferCount: 0",
    );
    expect(pipelineSource).toContain(
      "evaluationResultExtraQueueDrainCount: 0",
    );
    expect(pipelineSource).toContain(
      "OPT-0067 diagnostic checkpoint cannot continue to evaluation 1 or VAE",
    );
  });

  it("pauses after graph preparation and gives every arm a separate full lifecycle", () => {
    const timingReady = pipelineSource.indexOf(
      "await waitForDitTimingAuthorization()",
    );
    const graphRun = pipelineSource.indexOf("await dit!.run(context.signal)");
    expect(timingReady).toBeGreaterThan(0);
    expect(graphRun).toBeGreaterThan(timingReady);
    expect(workerSource).toContain("await authorization.promise");
    expect(workerSource).toContain("graphCompiledBeforeThermalGate: definition.timed");
    expect(workerSource).toContain("profileEvidence: profileEvidence(checkpoint)");
    expect(workerSource).toContain("await backend.dispose()");
    expect(workerSource).toContain("noCrossArmBackendOrDeviceReuse: true");
    expect(workerSource).toContain("accepted.push(Object.freeze");
    expect(workerSource).toContain("await prepareTimedArm()");
    expect(workerSource).not.toContain("captureEvaluationLatents");
    expect(pageSource).toContain("gate-rejected");
    expect(pageSource).toContain("trace-rejected");
    expect(htmlSource).toContain("A1 query8, B1 quad, B2 quad, A2 query8");
    expect(htmlSource).toContain("The next arm is not");
  });
});
