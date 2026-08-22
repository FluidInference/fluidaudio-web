import { describe, expect, it } from "vitest";

import {
  ACE_TURBO_V1_CORRECTNESS_PROFILE,
  DEFAULT_ACE_PLANNER_CONFIGURATION,
  assertAceGenerationRequest,
  resolveAceDynamicConditionalWeighting,
  type AceGenerationRequest,
} from "../src/api.js";
import { canonicalizeSeed } from "../src/runtime/seed.js";
import {
  AceProgressSequence,
  generationStagePlan,
  type AceGenerationProgress,
} from "../src/runtime/stages.js";

function request(
  overrides: Partial<AceGenerationRequest> = {},
): AceGenerationRequest {
  return {
    generationProfile: "ace-turbo-v1-correctness",
    prompt: "A sparse instrumental test fixture",
    instrumental: true,
    durationSeconds: 10,
    seed: canonicalizeSeed(42),
    planner: { mode: "disabled" },
    ...overrides,
  };
}

describe("generation request contract", () => {
  it("pins every output-affecting v1 sampler default", () => {
    expect(ACE_TURBO_V1_CORRECTNESS_PROFILE).toMatchObject({
      inferenceMethod: "ode",
      sampler: "euler",
      denoisingEvaluations: 8,
      diffusionGuidanceScale: 1,
      shift: 3,
      customTimesteps: null,
      adaptiveDualGuidance: false,
      velocityNormThreshold: 0,
      velocityEmaFactor: 0,
      latentShift: 0,
      latentRescale: 1,
      dynamicConditionalWeighting: {
        resolution: "planner-mode",
        direct: {
          enabled: true,
          mode: "double",
          wavelet: "haar",
          lowBandScale: 0.05,
          highBandScale: 0.02,
        },
        thinking: {
          enabled: true,
          mode: "double",
          wavelet: "haar",
          lowBandScale: 0.02,
          highBandScale: 0.06,
        },
      },
      outputNormalization: {
        mode: "global-peak",
        targetDbfs: -1,
        silenceThreshold: 1e-6,
      },
    });
    expect(ACE_TURBO_V1_CORRECTNESS_PROFILE.schedulerTimesteps).toEqual([
      1.0, 0.9545454545454546, 0.9, 0.8333333333333334,
      0.75, 0.6428571428571429, 0.5, 0.3,
    ]);
    expect(
      ACE_TURBO_V1_CORRECTNESS_PROFILE.effectiveSamplerTimestepsBfloat16,
    ).toEqual([
      1.0, 0.953125, 0.8984375, 0.83203125,
      0.75, 0.64453125, 0.5, 0.30078125,
    ]);
  });

  it("resolves the pinned direct and Think-mode DCW strengths", () => {
    expect(resolveAceDynamicConditionalWeighting({ mode: "disabled" })).toMatchObject({
      lowBandScale: 0.05,
      highBandScale: 0.02,
    });
    expect(
      resolveAceDynamicConditionalWeighting(DEFAULT_ACE_PLANNER_CONFIGURATION),
    ).toMatchObject({
      lowBandScale: 0.02,
      highBandScale: 0.06,
    });
  });

  it("accepts explicit planner-disabled and planner-enabled requests", () => {
    expect(() => assertAceGenerationRequest(request())).not.toThrow();
    expect(() =>
      assertAceGenerationRequest(
        request({ planner: DEFAULT_ACE_PLANNER_CONFIGURATION }),
      ),
    ).not.toThrow();
  });

  it("enforces the pinned profile, duration, seed, and sampling controls", () => {
    expect(() =>
      assertAceGenerationRequest(request({ durationSeconds: 9 })),
    ).toThrow(/10 through 240/);
    expect(() =>
      assertAceGenerationRequest(
        request({ seed: "42" as AceGenerationRequest["seed"] }),
      ),
    ).toThrow(/canonical/);
    expect(() =>
      assertAceGenerationRequest(
        request({
          planner: {
            ...DEFAULT_ACE_PLANNER_CONFIGURATION,
            topP: 1.1,
          },
        }),
      ),
    ).toThrow(/topP/);
    expect(() =>
      assertAceGenerationRequest(
        request({
          planner: { mode: "disabled", thinking: true } as never,
        }),
      ),
    ).toThrow(/hidden controls/);
    expect(() =>
      assertAceGenerationRequest({ ...request(), shift: 1 } as never),
    ).toThrow(/unknown controls/);
    expect(() =>
      assertAceGenerationRequest(
        request({ metadata: { bpm: 120, shift: 1 } as never }),
      ),
    ).toThrow(/unknown controls/);
    expect(() =>
      assertAceGenerationRequest(request({ metadata: [] as never })),
    ).toThrow(/object/);
    expect(() =>
      assertAceGenerationRequest(request({ metadata: { bpm: 120.5 } })),
    ).toThrow(/integer/);
    expect(() =>
      assertAceGenerationRequest(
        request({
          planner: {
            ...DEFAULT_ACE_PLANNER_CONFIGURATION,
            thinking: {
              ...DEFAULT_ACE_PLANNER_CONFIGURATION.thinking,
              hiddenSamplingControl: true,
            } as never,
          },
        }),
      ),
    ).toThrow(/unknown controls/);
  });
});

describe("stage and progress contract", () => {
  it("removes only semantic planning from the direct-Turbo plan", () => {
    const direct = generationStagePlan(false);
    const planned = generationStagePlan(true);
    expect(direct).not.toContain("semantic-planner");
    expect(planned).toContain("semantic-planner");
    expect(planned.indexOf("text-encoder")).toBeLessThan(
      planned.indexOf("semantic-detokenizer"),
    );
    expect(planned.indexOf("semantic-detokenizer")).toBeLessThan(
      planned.indexOf("condition-encoder"),
    );
    expect(direct.at(-1)).toBe("done");
  });

  it("rejects fraction and stage regressions", () => {
    const sequence = new AceProgressSequence();
    const progress: AceGenerationProgress = {
      stage: "prepare",
      stageIndex: 0,
      stageCount: 13,
      completedUnits: 1,
      totalUnits: 1,
      unit: "items",
      overallFraction: 0.1,
      elapsedMs: 10,
    };
    sequence.accept(progress);
    expect(() =>
      sequence.accept({ ...progress, overallFraction: 0.09 }),
    ).toThrow(/decrease/);
  });

  it("holds same-stage units and timing monotonic after discovery", () => {
    const sequence = new AceProgressSequence();
    const progress: AceGenerationProgress = {
      stage: "prepare",
      stageIndex: 0,
      stageCount: 13,
      completedUnits: 0,
      totalUnits: 0,
      unit: "items",
      overallFraction: 0,
      elapsedMs: 1,
    };
    sequence.accept(progress);
    sequence.accept({
      ...progress,
      completedUnits: 1,
      totalUnits: 2,
      overallFraction: 0.05,
      elapsedMs: 2,
    });
    expect(() =>
      sequence.accept({
        ...progress,
        completedUnits: 0,
        totalUnits: 2,
        overallFraction: 0.06,
        elapsedMs: 3,
      }),
    ).toThrow(/completedUnits/);
    expect(() =>
      sequence.accept({
        ...progress,
        completedUnits: 2,
        totalUnits: 3,
        overallFraction: 0.06,
        elapsedMs: 3,
      }),
    ).toThrow(/totalUnits/);
    expect(() =>
      sequence.accept({
        ...progress,
        completedUnits: 2,
        totalUnits: 2,
        unit: "tokens",
        overallFraction: 0.06,
        elapsedMs: 3,
      }),
    ).toThrow(/unit/);
    expect(() =>
      sequence.accept({
        ...progress,
        completedUnits: 2,
        totalUnits: 2,
        overallFraction: 0.06,
        elapsedMs: 1,
      }),
    ).toThrow(/elapsedMs/);
  });
});
