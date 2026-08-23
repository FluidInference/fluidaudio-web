import {
  assertAceGenerationRequest,
  type AceGenerationRequest,
} from "../api.js";
import type { AcePlannerMetadata } from "./planner.js";

export const ACE_INSTRUMENTAL_LYRICS = "[Instrumental]" as const;
export const ACE_DIRECT_CONDITIONING_INSTRUCTION =
  "Fill the audio semantic mask based on the given conditions:" as const;
export const ACE_PLANNER_CONDITIONING_INSTRUCTION =
  "Generate audio semantic tokens based on the given conditions:" as const;

/**
 * Resolve the product's explicit instrumental control before any model sees
 * lyrics. Upstream's request/UI layers express this state as the literal
 * `[Instrumental]`; relying on an optional caller string would let the planner
 * and text conditioner receive contradictory inputs.
 */
export function canonicalizeAceGenerationRequest(
  request: AceGenerationRequest,
): AceGenerationRequest {
  assertAceGenerationRequest(request);
  const lyrics = request.instrumental
    ? ACE_INSTRUMENTAL_LYRICS
    : request.lyrics ?? "";
  if (request.lyrics === lyrics) return request;
  return Object.freeze({ ...request, lyrics });
}

export interface AceResolvedConditioningText {
  readonly instruction:
    | typeof ACE_DIRECT_CONDITIONING_INSTRUCTION
    | typeof ACE_PLANNER_CONDITIONING_INSTRUCTION;
  readonly caption: string;
  readonly lyrics: string;
  readonly vocalLanguage: string;
  readonly formattedMetadata: string;
}

export interface AceConditioningTextSource {
  readonly caption: string;
  readonly lyrics: string;
  readonly instrumental: boolean;
  readonly durationSeconds: number;
  readonly vocalLanguage: string;
  readonly metadata: AcePlannerMetadata;
}

/** Exact strings entering the pinned text and lyric tokenizers. */
export function resolveAceConditioningText(
  downstream: AceConditioningTextSource,
  plannerEnabled: boolean,
): AceResolvedConditioningText {
  const bpm = downstream.metadata.bpm;
  const keyScale = downstream.metadata.keyscale?.trim() ?? "";
  const timeSignature = downstream.metadata.timesignature;
  const formattedMetadata =
    `- bpm: ${bpm ?? "N/A"}\n` +
    `- timesignature: ${timeSignature ?? "N/A"}\n` +
    `- keyscale: ${keyScale || "N/A"}\n` +
    `- duration: ${downstream.durationSeconds} seconds\n`;
  return Object.freeze({
    instruction: plannerEnabled
      ? ACE_PLANNER_CONDITIONING_INSTRUCTION
      : ACE_DIRECT_CONDITIONING_INSTRUCTION,
    caption: downstream.caption,
    lyrics: downstream.instrumental
      ? ACE_INSTRUMENTAL_LYRICS
      : downstream.lyrics,
    vocalLanguage: downstream.vocalLanguage || "unknown",
    formattedMetadata,
  });
}
