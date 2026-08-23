import { describe, expect, it } from "vitest";

import { canonicalizeSeed } from "../src/runtime/seed.js";
import {
  ACE_DIRECT_CONDITIONING_INSTRUCTION,
  ACE_INSTRUMENTAL_LYRICS,
  ACE_PLANNER_CONDITIONING_INSTRUCTION,
  canonicalizeAceGenerationRequest,
  resolveAceConditioningText,
} from "../src/runtime/generation-inputs.js";
import type { AceGenerationRequest } from "../src/api.js";

describe("ACE generation input resolution", () => {
  it("canonicalizes explicit instrumental state before planner or text models", () => {
    const request = generationRequest({
      instrumental: true,
      lyrics: "These words must not reach either model",
    });
    const resolved = canonicalizeAceGenerationRequest(request);
    expect(resolved.lyrics).toBe(ACE_INSTRUMENTAL_LYRICS);
    expect(request.lyrics).toBe("These words must not reach either model");
  });

  it("preserves supplied vocal lyrics and resolves absent vocal lyrics to empty", () => {
    const supplied = generationRequest({ instrumental: false, lyrics: "Line" });
    expect(canonicalizeAceGenerationRequest(supplied)).toBe(supplied);
    expect(canonicalizeAceGenerationRequest(
      generationRequest({ instrumental: false }),
    ).lyrics).toBe("");
  });

  it("builds exact direct/planner instructions and metadata text", () => {
    const downstream = {
      caption: "Warm synths",
      lyrics: ACE_INSTRUMENTAL_LYRICS,
      instrumental: true,
      durationSeconds: 12,
      vocalLanguage: "unknown",
      metadata: {
        bpm: 104,
        duration: 12,
        keyscale: "D minor",
        timesignature: "4",
      },
    } as const;
    const direct = resolveAceConditioningText(downstream, false);
    expect(direct).toMatchObject({
      instruction: ACE_DIRECT_CONDITIONING_INSTRUCTION,
      caption: "Warm synths",
      lyrics: "[Instrumental]",
      vocalLanguage: "unknown",
    });
    expect(direct.formattedMetadata).toBe(
      "- bpm: 104\n" +
      "- timesignature: 4\n" +
      "- keyscale: D minor\n" +
      "- duration: 12 seconds\n",
    );
    expect(resolveAceConditioningText(downstream, true).instruction)
      .toBe(ACE_PLANNER_CONDITIONING_INSTRUCTION);
  });
});

function generationRequest(
  overrides: Partial<AceGenerationRequest>,
): AceGenerationRequest {
  return {
    generationProfile: "ace-turbo-v1-correctness",
    prompt: "Prompt",
    instrumental: false,
    durationSeconds: 12,
    seed: canonicalizeSeed(17),
    planner: { mode: "disabled" },
    ...overrides,
  };
}
