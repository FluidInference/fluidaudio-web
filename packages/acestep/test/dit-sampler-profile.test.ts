import { describe, expect, it } from "vitest";

import { ACE_DIRECT_DCW_CONFIGURATION } from "../src/api.js";
import {
  createAceDitSamplerSchedule,
  createAceDitTurboSamplerSchedule,
} from "../src/webgpu/ace-dit.js";
import { planAceDitGpuBackendMemory } from "../src/webgpu/dit-backend.js";
import {
  aceDitGraphQuantumCount,
  createAceDitGraphControlData,
  createAceDitGraphQuantumPlan,
} from "../src/webgpu/dit-graph.js";
import {
  ACE_DIT_SAMPLER_SCHEDULE_PROFILE_IDS,
  ACE_OPT_0055_SIX_SAMPLER_SCHEDULE_PROFILE,
  ACE_OPT_0055_SIX_SAMPLER_SCHEDULE_PROFILE_ID,
  ACE_OPT_0065_FIVE_SAMPLER_SCHEDULE_PROFILE,
  ACE_OPT_0065_FIVE_SAMPLER_SCHEDULE_PROFILE_ID,
  ACE_TURBO_EIGHT_SAMPLER_SCHEDULE_PROFILE,
  ACE_TURBO_EIGHT_SAMPLER_SCHEDULE_PROFILE_ID,
  requireResolvedAceDitSamplerScheduleProfile,
  resolveAceDitSamplerScheduleProfile,
} from "../src/webgpu/dit-sampler-profile.js";

describe("authenticated DiT sampler schedule profiles", () => {
  it("resolves only the frozen registered contracts with their committed hashes", () => {
    expect(ACE_DIT_SAMPLER_SCHEDULE_PROFILE_IDS).toEqual([
      ACE_TURBO_EIGHT_SAMPLER_SCHEDULE_PROFILE_ID,
      ACE_OPT_0055_SIX_SAMPLER_SCHEDULE_PROFILE_ID,
      ACE_OPT_0065_FIVE_SAMPLER_SCHEDULE_PROFILE_ID,
    ]);
    expect(resolveAceDitSamplerScheduleProfile()).toBe(
      ACE_TURBO_EIGHT_SAMPLER_SCHEDULE_PROFILE,
    );
    expect(resolveAceDitSamplerScheduleProfile(
      ACE_OPT_0055_SIX_SAMPLER_SCHEDULE_PROFILE_ID,
    )).toBe(ACE_OPT_0055_SIX_SAMPLER_SCHEDULE_PROFILE);
    expect(resolveAceDitSamplerScheduleProfile(
      ACE_OPT_0065_FIVE_SAMPLER_SCHEDULE_PROFILE_ID,
    )).toBe(ACE_OPT_0065_FIVE_SAMPLER_SCHEDULE_PROFILE);
    expect(ACE_TURBO_EIGHT_SAMPLER_SCHEDULE_PROFILE.contractSha256).toBe(
      "6e81531b45fff521fb1ea5f454cf1b9fd7559b7c8be0d78b7023ac4b6b9ddbd2",
    );
    expect(ACE_OPT_0055_SIX_SAMPLER_SCHEDULE_PROFILE.contractSha256).toBe(
      "c2d8a31df4d0023bb85b5a72f8eca3b781f006ddf763689c750622b231291bf0",
    );
    expect(ACE_OPT_0065_FIVE_SAMPLER_SCHEDULE_PROFILE.contractSha256).toBe(
      "a585b8bf567bd5cca30bd89bf60709afa6c72f2ef09d9c1015d2f83db7198324",
    );
    expect(() => resolveAceDitSamplerScheduleProfile(
      "forged" as never,
    )).toThrow(/not registered/);
    expect(() => requireResolvedAceDitSamplerScheduleProfile({
      ...ACE_OPT_0055_SIX_SAMPLER_SCHEDULE_PROFILE,
    })).toThrow(/resolver object/);
  });

  it("materializes the exact BF16 timestep and update arrays", () => {
    const profiles = [
      ACE_TURBO_EIGHT_SAMPLER_SCHEDULE_PROFILE,
      ACE_OPT_0055_SIX_SAMPLER_SCHEDULE_PROFILE,
      ACE_OPT_0065_FIVE_SAMPLER_SCHEDULE_PROFILE,
    ] as const;
    for (const profile of profiles) {
      const schedule = createAceDitSamplerSchedule(
        profile,
        ACE_DIRECT_DCW_CONFIGURATION,
      );
      expect(schedule.map(({ declaredTimestep }) => declaredTimestep)).toEqual(
        profile.declaredTimesteps,
      );
      expect(schedule.map(({ timestep }) => timestep)).toEqual(
        profile.effectiveBfloat16Timesteps,
      );
      expect(schedule.map(({ updateCoefficient }) => updateCoefficient)).toEqual(
        profile.effectiveBfloat16UpdateCoefficients,
      );
      expect(schedule.at(-1)?.update).toBe("predicted-clean");
      expect(schedule.slice(0, -1).every(({ update }) => update === "euler"))
        .toBe(true);
    }
    expect(createAceDitTurboSamplerSchedule(ACE_DIRECT_DCW_CONFIGURATION))
      .toEqual(createAceDitSamplerSchedule(
        ACE_TURBO_EIGHT_SAMPLER_SCHEDULE_PROFILE,
        ACE_DIRECT_DCW_CONFIGURATION,
      ));
  });

  it("plans dynamic graph/control topology while leaving default eight unchanged", () => {
    const cases = [
      [ACE_TURBO_EIGHT_SAMPLER_SCHEDULE_PROFILE, 249],
      [ACE_OPT_0055_SIX_SAMPLER_SCHEDULE_PROFILE, 193],
      [ACE_OPT_0065_FIVE_SAMPLER_SCHEDULE_PROFILE, 165],
    ] as const;
    for (const [profile, quantumCount] of cases) {
      expect(aceDitGraphQuantumCount(profile)).toBe(quantumCount);
      const quanta = createAceDitGraphQuantumPlan(profile);
      expect(quanta).toHaveLength(quantumCount);
      expect(quanta.at(-1)).toMatchObject({
        kind: "sampler",
        evaluation: profile.evaluationCount - 1,
      });
      const controls = createAceDitGraphControlData(
        { batch: 1, latentFrames: 2, conditionTokens: 1 },
        ACE_DIRECT_DCW_CONFIGURATION,
        profile,
      );
      expect(controls.timesteps).toHaveLength(profile.evaluationCount);
      expect(controls.timesteps.map((values) => values[0])).toEqual(
        profile.effectiveBfloat16Timesteps,
      );
    }
    expect(createAceDitGraphQuantumPlan()).toHaveLength(249);
  });

  it("accounts only the selected trajectory and one bounded velocity readback", () => {
    const shape = { batch: 1, latentFrames: 2, conditionTokens: 1 };
    const five = planAceDitGpuBackendMemory(
      "reference-bf16",
      shape,
      1_024,
      "mixed-opt-0009",
      1,
      true,
      false,
      ACE_OPT_0065_FIVE_SAMPLER_SCHEDULE_PROFILE,
      true,
    );
    expect(five.evaluationCount).toBe(5);
    expect(five.logicalGraphQuantumCount).toBe(165);
    expect(five.evaluationReadbackBytes).toBe(5 * five.readbackBufferBytes);
    expect(five.evaluation0VelocityReadbackBytes).toBe(
      five.readbackBufferBytes,
    );
    expect(five.detachedEvaluationLatentBytes).toBe(
      5 * five.detachedFinalLatentBytes,
    );
    expect(five.detachedEvaluation0VelocityBytes).toBe(
      five.detachedFinalLatentBytes,
    );
  });
});
