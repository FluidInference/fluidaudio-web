import { ACE_MAX_DURATION_SECONDS } from "../api.js";
import {
  ACE_PLANNER_AUDIO_CODE_FIRST_TOKEN_ID,
  ACE_QWEN_IM_END_TOKEN_ID,
  AceQwenBpeTokenizer,
} from "../tokenizer/index.js";
import { ACE_PLANNER_QWEN3_CONFIG } from "../webgpu/qwen3.js";
import type {
  AcePlannerCotAcceptedToken,
  AcePlannerCotConstraintController,
  AcePlannerCotConstraintInput,
  AcePlannerMetadata,
  AcePlannerRequestPlan,
} from "./planner.js";
import type { AcePlannerAllowedTokens } from "./planner-sampling.js";

const SOURCE_REVISION = "6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0";
const NEWLINE_TOKEN_ID = 198;
const BACKTICK_TOKEN_ID = 63;
const VALID_AUDIO_CODE_COUNT = 64_000;
const CAPTION_TOKEN_LIMIT = 512;

/**
 * Exact active CoT constraint profile used by pinned Turbo product generation.
 *
 * `skipGenres` is deliberately fixed true: the pinned two-phase entry point
 * passes `skip_genres=True`, and the 4.8 MB source genre vocabulary is not a
 * browser model input. An unexpected free-form caption transition to
 * `genres:` therefore fails closed instead of silently using a different
 * vocabulary. The upstream direct caption transition can bypass skip flags;
 * this guard makes that malformed path explicit.
 */
export const ACE_PLANNER_METADATA_FSM_CONTRACT = Object.freeze({
  id: "ace-planner-metadata-fsm-v1",
  aceSourceRevision: SOURCE_REVISION,
  upstreamSource: "acestep/constrained_logits_processor.py:53-2338",
  upstreamSourceSha256:
    "84cf84ad894130397ba53a4cbd8666961bf578c77295b79599b288a3825faa32",
  upstreamConstantsSha256:
    "7b8d4ce49649c819d1b3be87a434be2d90768308768d4620639328f906209b22",
  tokenizerSha256:
    "35af56c3f5cb3ea2cc578aa28a8937770981d504f183ac5c8c38baf4bbd4af4d",
  generationPhase: "cot" as const,
  stopAtReasoning: true,
  skipGenres: true,
  bpmRange: Object.freeze([30, 300] as const),
  durationRange: Object.freeze([10, ACE_MAX_DURATION_SECONDS] as const),
  timeSignatures: Object.freeze([2, 3, 4, 6] as const),
  captionTokenLimit: CAPTION_TOKEN_LIMIT,
  languageSelection: "raw-logit-top-1" as const,
  fixedStringSegmentation: "longest-prefix-encoding-to-one-token" as const,
  terminalBehavior: "force-im-end-before-think-end-tag" as const,
  failClosed: true,
});

export interface AcePlannerMetadataConstraintOptions {
  readonly tokenizer: AceQwenBpeTokenizer;
  readonly userMetadata?: AcePlannerMetadata;
  readonly skipCaption?: boolean;
  readonly skipLanguage?: boolean;
  /** Defaults to the browser product maximum of 240 seconds. */
  readonly maxDuration?: number;
}

/** Map a validated Phase-1 request plan to the pinned FSM configuration. */
export function createAcePlannerMetadataConstraintForPlan(
  tokenizer: AceQwenBpeTokenizer,
  plan: AcePlannerRequestPlan,
): AcePlannerMetadataConstraintController {
  if (plan.shouldRunCot !== true || plan.cotPrompt === null) {
    throw new TypeError("ACE planner metadata FSM requires a plan with an active CoT phase");
  }
  if (plan.configuration.constrainedDecoding !== true) {
    throw new TypeError("ACE planner metadata FSM requires constrained decoding");
  }
  return new AcePlannerMetadataConstraintController({
    tokenizer,
    userMetadata: plan.userMetadata,
    skipCaption: !plan.configuration.thinking.useCotCaption,
    skipLanguage: !plan.configuration.thinking.useCotLanguage,
  });
}

type MetadataField =
  | "bpm"
  | "caption"
  | "duration"
  | "keyscale"
  | "language"
  | "timesignature";

type FsmState =
  | "think-tag"
  | "newline-after-think"
  | "bpm-name"
  | "bpm-value"
  | "caption-name"
  | "caption-value"
  | "duration-name"
  | "duration-value"
  | "keyscale-name"
  | "keyscale-value"
  | "language-name"
  | "language-value"
  | "timesignature-name"
  | "timesignature-value"
  | "think-end-tag"
  | "completed";

interface PrefixTree {
  readonly continuations: ReadonlyMap<string, readonly number[]>;
}

interface PendingConstraint {
  readonly step: number;
  readonly allowed: AcePlannerAllowedTokens;
  readonly admits: (tokenId: number) => boolean;
}

const FIXED_STRINGS: Readonly<Partial<Record<FsmState, string>>> = Object.freeze({
  "think-tag": "<think>",
  "newline-after-think": "\n",
  "bpm-name": "bpm:",
  "caption-name": "caption:",
  "duration-name": "duration:",
  "keyscale-name": "keyscale:",
  "language-name": "language:",
  "timesignature-name": "timesignature:",
  "think-end-tag": "</think>",
});

const LANGUAGE_VALUES = Object.freeze([
  "ar", "az", "bg", "bn", "ca", "cs", "da", "de", "el", "en",
  "es", "fa", "fi", "fr", "he", "hi", "hr", "ht", "hu", "id",
  "is", "it", "ja", "ko", "la", "lt", "ms", "ne", "nl", "no",
  "pa", "pl", "pt", "ro", "ru", "sa", "sk", "sr", "sv", "sw",
  "ta", "te", "th", "tl", "tr", "uk", "ur", "vi", "yue", "zh",
  "unknown",
] as const);

// CPython 3.13 iteration order of the pinned processor's integer set at the
// empty language prefix. It affects only an exact raw-logit tie. Capturing the
// order preserves Torch argmax's first-index rule instead of inventing a new
// browser tie break.
const UPSTREAM_LANGUAGE_FIRST_TOKEN_ORDER = Object.freeze([
  15_236, 902, 1_422, 655, 270, 18_962, 34_323, 40_597, 662, 409,
  796, 926, 11_937, 1_187, 10_532, 2_218, 9_136, 432, 2_994, 566,
  822, 9_788, 10_817, 834, 7_106, 32_965, 3_275, 14_670, 25_175,
  12_376, 15_588, 9_829, 2_021, 29_796, 59_748, 489, 18_026, 9_450,
  877, 1_901, 4_335, 625, 2_162, 11_122, 379, 13_559, 374, 8_951,
  20_216, 1_013, 1_531,
] as const);

const KEYSCALE_VALUES = Object.freeze(
  ["A", "B", "C", "D", "E", "F", "G"].flatMap((note) =>
    ["", "#", "b", "♯", "♭"].flatMap((accidental) =>
      ["major", "minor"].map((mode) => `${note}${accidental} ${mode}`),
    ),
  ),
);

/**
 * Upstream caption masking admits every vocabulary row except one backtick
 * token and the first 64,000 audio-code tokens. Its extra 1,535 audio-code
 * vocabulary rows are intentionally still admitted. The frozen list is about
 * 1.2 MiB in a normal JS number array and is shared by every controller.
 */
const CAPTION_ALLOWED_TOKEN_IDS = Object.freeze(buildCaptionAllowedTokenIds());
const CAPTION_ALLOWED_TOKENS: AcePlannerAllowedTokens = Object.freeze({
  kind: "ids" as const,
  tokenIds: CAPTION_ALLOWED_TOKEN_IDS,
});
const ALL_TOKENS: AcePlannerAllowedTokens = Object.freeze({ kind: "all" as const });

/** Pinned browser implementation of the active metadata constraint FSM. */
export class AcePlannerMetadataConstraintController
  implements AcePlannerCotConstraintController {
  private readonly tokenizer: AceQwenBpeTokenizer;
  private readonly userMetadata: Readonly<Partial<Record<MetadataField, string>>>;
  private readonly skipCaption: boolean;
  private readonly skipLanguage: boolean;
  private readonly fixedTokenPaths: ReadonlyMap<FsmState, readonly number[]>;
  private readonly bpmTree: PrefixTree;
  private readonly durationTree: PrefixTree;
  private readonly keyscaleTree: PrefixTree;
  private readonly languageTree: PrefixTree;
  private readonly timesignatureTree: PrefixTree;

  private state: FsmState = "think-tag";
  private positionInState = 0;
  private accumulatedValue = "";
  private accumulatedTokenIds: number[] = [];
  private captionAfterNewline = false;
  private captionTokenCount = 0;
  private captionEnding = false;
  private pendingFieldName = "";
  private userFieldTokenQueue: number[] = [];
  private currentUserField: MetadataField | null = null;
  private acceptedTokenIds: number[] = [];
  private promptTokenIds: number[] | null = null;
  private pending: PendingConstraint | null = null;

  constructor(options: AcePlannerMetadataConstraintOptions) {
    if (!(options.tokenizer instanceof AceQwenBpeTokenizer)) {
      throw new TypeError("ACE planner metadata FSM requires an ACE Qwen tokenizer");
    }
    if (options.tokenizer.kind !== "planner") {
      throw new TypeError("ACE planner metadata FSM requires the planner tokenizer");
    }
    this.tokenizer = options.tokenizer;
    this.skipCaption = requireBooleanDefault(
      options.skipCaption,
      false,
      "ACE planner metadata skipCaption",
    );
    this.skipLanguage = requireBooleanDefault(
      options.skipLanguage,
      false,
      "ACE planner metadata skipLanguage",
    );
    const maxDuration = options.maxDuration ?? ACE_MAX_DURATION_SECONDS;
    if (
      !Number.isSafeInteger(maxDuration) ||
      maxDuration < 10 ||
      maxDuration > 600
    ) {
      throw new RangeError("ACE planner metadata maxDuration must be an integer from 10 through 600");
    }
    this.userMetadata = normalizeUserMetadata(options.userMetadata ?? {});
    assertPinnedTokenizer(this.tokenizer);
    this.fixedTokenPaths = buildFixedTokenPaths(this.tokenizer);
    this.bpmTree = buildValuePrefixTree(
      this.tokenizer,
      integerStrings(30, 300),
      "bpm:",
      "bpm: ",
    );
    this.durationTree = buildValuePrefixTree(
      this.tokenizer,
      integerStrings(10, maxDuration),
      "duration:",
      "duration: ",
    );
    this.keyscaleTree = buildValuePrefixTree(
      this.tokenizer,
      KEYSCALE_VALUES,
      "keyscale:",
      "keyscale: ",
      (tokens) => {
        const first = tokens[0];
        if (first === undefined) return false;
        return "ABCDEFG".includes(this.tokenizer.decode([first]).trimStart()[0]?.toUpperCase() ?? "");
      },
    );
    this.languageTree = buildValuePrefixTree(
      this.tokenizer,
      LANGUAGE_VALUES,
      "language:",
      "language: ",
    );
    this.timesignatureTree = buildValuePrefixTree(
      this.tokenizer,
      ["2", "3", "4", "6"],
      "timesignature:",
      "timesignature: ",
    );
    assertLanguageTieOrder(this.languageTree);
  }

  allowedTokens(input: AcePlannerCotConstraintInput): AcePlannerAllowedTokens {
    this.requireInputSequence(input);
    if (this.pending !== null) {
      throw new Error("ACE planner metadata FSM allowedTokens was called twice for one step");
    }
    if (this.state === "completed") {
      throw new Error("ACE planner metadata FSM is already complete");
    }
    const pending = this.createPendingConstraint(input.logits);
    this.pending = Object.freeze({ ...pending, step: input.step });
    return pending.allowed;
  }

  acceptToken(input: AcePlannerCotAcceptedToken): Readonly<{ readonly finished: boolean }> {
    const pending = this.pending;
    if (pending === null) {
      throw new Error("ACE planner metadata FSM acceptToken requires a preceding allowedTokens call");
    }
    if (input.step !== pending.step || input.step !== this.acceptedTokenIds.length) {
      throw new RangeError("ACE planner metadata FSM accepted an out-of-order step");
    }
    requireVocabularyToken(input.tokenId, "ACE planner metadata accepted token");
    if (!pending.admits(input.tokenId)) {
      throw new RangeError(
        `ACE planner metadata FSM rejected token ID ${input.tokenId} at step ${input.step}`,
      );
    }
    const expectedText = this.tokenizer.decode([input.tokenId]);
    if (input.tokenText !== expectedText) {
      throw new Error("ACE planner metadata FSM token text does not match the tokenizer");
    }
    const expectedEmitted = [...this.acceptedTokenIds, input.tokenId];
    requireExactTokenSequence(
      input.emittedTokenIds,
      expectedEmitted,
      "ACE planner metadata accepted history",
    );
    this.pending = null;
    this.acceptedTokenIds.push(input.tokenId);
    this.updateState(input.tokenId, input.tokenText);
    return Object.freeze({ finished: this.state === "completed" });
  }

  private requireInputSequence(input: AcePlannerCotConstraintInput): void {
    if (!Number.isSafeInteger(input.step) || input.step < 0) {
      throw new RangeError("ACE planner metadata FSM step must be non-negative");
    }
    if (input.step !== this.acceptedTokenIds.length) {
      throw new RangeError("ACE planner metadata FSM received an out-of-order step");
    }
    requireExactTokenSequence(
      input.emittedTokenIds,
      this.acceptedTokenIds,
      "ACE planner metadata emitted history",
    );
    if (this.promptTokenIds === null) {
      if (input.promptTokenIds.length === 0) {
        throw new RangeError("ACE planner metadata FSM requires a non-empty prompt");
      }
      this.promptTokenIds = copyTokenSequence(
        input.promptTokenIds,
        "ACE planner metadata prompt",
      );
    } else {
      requireExactTokenSequence(
        input.promptTokenIds,
        this.promptTokenIds,
        "ACE planner metadata prompt",
      );
    }
    if (input.logits.length !== ACE_PLANNER_QWEN3_CONFIG.vocabularySize) {
      throw new RangeError(
        `ACE planner metadata logits have length ${input.logits.length}; ` +
          `${ACE_PLANNER_QWEN3_CONFIG.vocabularySize} required`,
      );
    }
  }

  private createPendingConstraint(logits: ArrayLike<number>): Omit<PendingConstraint, "step"> {
    if (this.userFieldTokenQueue.length > 0) {
      return singleTokenConstraint(this.userFieldTokenQueue[0]!);
    }
    const fixed = FIXED_STRINGS[this.state];
    if (fixed !== undefined) {
      // This reproduces the pinned stop_at_reasoning bug exactly: because the
      // entire eight-character closing tag is <= 10 characters, EOS is forced
      // before any `</think>` token is emitted.
      if (this.state === "think-end-tag" && fixed.length - this.positionInState <= 10) {
        return singleTokenConstraint(ACE_QWEN_IM_END_TOKEN_ID);
      }
      const path = this.fixedTokenPaths.get(this.state);
      const tokenId = path?.[this.positionInState];
      if (tokenId === undefined) {
        throw new Error(`ACE planner metadata fixed state ${this.state} has no continuation`);
      }
      return singleTokenConstraint(tokenId);
    }

    switch (this.state) {
      case "bpm-value":
        return this.numericConstraint("bpm", this.bpmTree);
      case "caption-value":
        return this.captionConstraint(logits);
      case "duration-value":
        return this.numericConstraint("duration", this.durationTree);
      case "keyscale-value":
        return this.completeThenNewlineConstraint("keyscale", this.keyscaleTree);
      case "language-value":
        return this.languageConstraint(logits);
      case "timesignature-value":
        return this.completeThenNewlineConstraint("timesignature", this.timesignatureTree);
      case "completed":
        throw new Error("ACE planner metadata FSM is already complete");
      default:
        return unexpectedState(this.state);
    }
  }

  private numericConstraint(
    field: "bpm" | "duration",
    tree: PrefixTree,
  ): Omit<PendingConstraint, "step"> {
    const injected = this.beginUserField(field);
    if (injected !== null) return injected;
    const allowed = treeContinuation(tree, this.accumulatedTokenIds, field);
    // Upstream appends a second copy of newline for a complete BPM/duration
    // prefix. The shared browser sampler rejects duplicate IDs; deduplicating
    // that list is mask-identical and leaves every score unchanged.
    return idListConstraint(allowed);
  }

  private completeThenNewlineConstraint(
    field: "keyscale" | "timesignature",
    tree: PrefixTree,
  ): Omit<PendingConstraint, "step"> {
    const injected = this.beginUserField(field);
    if (injected !== null) return injected;
    const allowed = treeContinuation(tree, this.accumulatedTokenIds, field);
    if (allowed.includes(NEWLINE_TOKEN_ID)) {
      return singleTokenConstraint(NEWLINE_TOKEN_ID);
    }
    return idListConstraint(allowed);
  }

  private languageConstraint(logits: ArrayLike<number>): Omit<PendingConstraint, "step"> {
    const injected = this.beginUserField("language");
    if (injected !== null) return injected;
    const allowed = treeContinuation(
      this.languageTree,
      this.accumulatedTokenIds,
      "language",
    );
    if (this.accumulatedTokenIds.length > 0) {
      return allowed.includes(NEWLINE_TOKEN_ID)
        ? singleTokenConstraint(NEWLINE_TOKEN_ID)
        : idListConstraint(allowed);
    }
    const candidateSet = new Set(allowed);
    let bestToken: number | null = null;
    let bestLogit = Number.NEGATIVE_INFINITY;
    for (const tokenId of UPSTREAM_LANGUAGE_FIRST_TOKEN_ORDER) {
      if (!candidateSet.has(tokenId)) continue;
      const logit = requireRawLogit(logits[tokenId], `language token ${tokenId}`);
      if (logit > bestLogit) {
        bestLogit = logit;
        bestToken = tokenId;
      }
    }
    if (bestToken === null || !Number.isFinite(bestLogit)) {
      throw new RangeError("ACE planner metadata language constraint has no finite candidate");
    }
    return singleTokenConstraint(bestToken);
  }

  private captionConstraint(logits: ArrayLike<number>): Omit<PendingConstraint, "step"> {
    const injected = this.beginUserField("caption");
    if (injected !== null) return injected;
    if (this.captionAfterNewline) {
      const topToken = rawArgmax(logits);
      const topText = this.tokenizer.decode([topToken]);
      if (topText.length > 0 && topText[0] !== " " && topText[0] !== "\t") {
        this.captionAfterNewline = false;
        this.captionEnding = true;
        this.pendingFieldName = "";
        return allTokenConstraint();
      }
      this.captionAfterNewline = false;
    }
    if (this.captionEnding) return allTokenConstraint();
    if (this.captionTokenCount >= CAPTION_TOKEN_LIMIT) {
      return singleTokenConstraint(NEWLINE_TOKEN_ID);
    }
    return captionTokenConstraint();
  }

  private beginUserField(field: MetadataField): Omit<PendingConstraint, "step"> | null {
    const value = this.userMetadata[field];
    const hasNotStarted = field === "caption"
      ? this.accumulatedValue.length === 0
      : this.accumulatedTokenIds.length === 0;
    if (value === undefined || !hasNotStarted) return null;
    if (this.userFieldTokenQueue.length !== 0 || this.currentUserField !== null) {
      throw new Error("ACE planner metadata user-field queue is inconsistent");
    }
    const tokens = this.tokenizer.encode(` ${value}\n`, { addSpecialTokens: false });
    if (tokens.length === 0) {
      throw new Error(`ACE planner metadata user field ${field} tokenized to nothing`);
    }
    this.userFieldTokenQueue = [...tokens];
    this.currentUserField = field;
    return singleTokenConstraint(this.userFieldTokenQueue[0]!);
  }

  private updateState(tokenId: number, tokenText: string): void {
    if (this.userFieldTokenQueue.length > 0) {
      if (tokenId !== this.userFieldTokenQueue[0]) {
        throw new Error("ACE planner metadata user-field queue accepted the wrong token");
      }
      this.userFieldTokenQueue.shift();
      if (this.userFieldTokenQueue.length === 0) {
        const field = this.currentUserField;
        if (field === null) {
          throw new Error("ACE planner metadata user-field queue lost its field");
        }
        this.currentUserField = null;
        this.enterState(this.nextFieldState(field));
      }
      return;
    }

    const fixed = FIXED_STRINGS[this.state];
    if (fixed !== undefined) {
      if (this.state === "think-end-tag" && tokenId === ACE_QWEN_IM_END_TOKEN_ID) {
        this.enterState("completed");
        return;
      }
      this.positionInState += tokenText.length;
      if (this.positionInState >= fixed.length) {
        this.enterState(this.nextFixedState(this.state));
      }
      return;
    }

    switch (this.state) {
      case "bpm-value":
      case "duration-value":
      case "timesignature-value":
        if (tokenId === NEWLINE_TOKEN_ID) {
          this.enterState(this.nextValueState(this.state));
        } else {
          this.accumulatedTokenIds.push(tokenId);
          if (/^\d+$/.test(tokenText.trim())) this.accumulatedValue += tokenText.trim();
        }
        return;
      case "keyscale-value":
      case "language-value":
        if (tokenId === NEWLINE_TOKEN_ID) {
          this.enterState(this.nextValueState(this.state));
        } else {
          this.accumulatedTokenIds.push(tokenId);
          this.accumulatedValue += tokenText;
        }
        return;
      case "caption-value":
        this.updateCaptionState(tokenText);
        return;
      case "completed":
        throw new Error("ACE planner metadata FSM accepted a token after completion");
      default:
        unexpectedState(this.state);
    }
  }

  private updateCaptionState(tokenText: string): void {
    this.captionTokenCount += 1;
    this.accumulatedValue += tokenText;
    this.captionAfterNewline = tokenText.includes("\n");
    if (!this.captionEnding) return;
    this.pendingFieldName += tokenText;
    if (!tokenText.includes(":")) return;
    const fieldName = this.pendingFieldName.trim().replace(/:+$/u, "").trim().toLowerCase();
    const destination: Readonly<Record<string, FsmState>> = Object.freeze({
      duration: "duration-value",
      keyscale: "keyscale-value",
      language: "language-value",
      timesignature: "timesignature-value",
    });
    if (fieldName === "genres") {
      throw new Error(
        "ACE planner metadata caption attempted the skipped genres field; refusing an unauthenticated genre vocabulary",
      );
    }
    const next = destination[fieldName];
    if (next === undefined) {
      throw new Error(`ACE planner metadata caption emitted unknown field ${JSON.stringify(fieldName)}`);
    }
    this.enterState(next);
  }

  private nextFixedState(state: FsmState): FsmState {
    switch (state) {
      case "think-tag": return "newline-after-think";
      case "newline-after-think": return "bpm-name";
      case "bpm-name": return "bpm-value";
      case "caption-name": return "caption-value";
      case "duration-name": return "duration-value";
      case "keyscale-name": return "keyscale-value";
      case "language-name": return "language-value";
      case "timesignature-name": return "timesignature-value";
      case "think-end-tag": return "completed";
      default:
        return unexpectedState(state);
    }
  }

  private nextValueState(state: FsmState): FsmState {
    switch (state) {
      case "bpm-value": return this.nextFieldState("bpm");
      case "duration-value": return this.nextFieldState("duration");
      case "keyscale-value": return this.nextFieldState("keyscale");
      case "language-value": return this.nextFieldState("language");
      case "timesignature-value": return "think-end-tag";
      default:
        return unexpectedState(state);
    }
  }

  private nextFieldState(field: MetadataField): FsmState {
    switch (field) {
      case "bpm": return this.skipCaption ? "duration-name" : "caption-name";
      case "caption": return "duration-name";
      case "duration": return "keyscale-name";
      case "keyscale": return this.skipLanguage ? "timesignature-name" : "language-name";
      case "language": return "timesignature-name";
      case "timesignature": return "think-end-tag";
    }
  }

  private enterState(state: FsmState): void {
    this.state = state;
    this.positionInState = 0;
    this.accumulatedValue = "";
    this.accumulatedTokenIds = [];
    this.captionAfterNewline = false;
    this.captionTokenCount = 0;
    this.captionEnding = false;
    this.pendingFieldName = "";
  }
}

function buildCaptionAllowedTokenIds(): number[] {
  const result: number[] = [];
  const audioEnd = ACE_PLANNER_AUDIO_CODE_FIRST_TOKEN_ID + VALID_AUDIO_CODE_COUNT;
  for (let tokenId = 0; tokenId < ACE_PLANNER_QWEN3_CONFIG.vocabularySize; tokenId += 1) {
    if (tokenId === BACKTICK_TOKEN_ID) continue;
    if (tokenId >= ACE_PLANNER_AUDIO_CODE_FIRST_TOKEN_ID && tokenId < audioEnd) continue;
    result.push(tokenId);
  }
  return result;
}

function normalizeUserMetadata(
  metadata: AcePlannerMetadata,
): Readonly<Partial<Record<MetadataField, string>>> {
  if (typeof metadata !== "object" || metadata === null || Array.isArray(metadata)) {
    throw new TypeError("ACE planner metadata FSM userMetadata must be an object");
  }
  const result: Partial<Record<MetadataField, string>> = {};
  for (const field of [
    "bpm", "caption", "duration", "keyscale", "language", "timesignature",
  ] as const) {
    const value = metadata[field];
    if (value === undefined) continue;
    if (typeof value !== "string" && typeof value !== "number") {
      throw new TypeError(`ACE planner metadata user field ${field} must be a string or number`);
    }
    if (typeof value === "number" && !Number.isFinite(value)) {
      throw new RangeError(`ACE planner metadata user field ${field} must be finite`);
    }
    result[field] = String(value);
  }
  return Object.freeze(result);
}

function assertPinnedTokenizer(tokenizer: AceQwenBpeTokenizer): void {
  const checks: readonly (readonly [string, readonly number[]])[] = [
    ["\n", [NEWLINE_TOKEN_ID]],
    ["`", [BACKTICK_TOKEN_ID]],
    ["<think>", [151_667]],
    ["</think>", [151_668]],
  ];
  for (const [text, expected] of checks) {
    const actual = tokenizer.encode(text, { addSpecialTokens: false });
    requireExactTokenSequence(actual, expected, `ACE planner metadata tokenizer ${JSON.stringify(text)}`);
  }
  if (tokenizer.eosTokenId !== ACE_QWEN_IM_END_TOKEN_ID) {
    throw new Error("ACE planner metadata tokenizer has the wrong EOS token");
  }
}

function buildFixedTokenPaths(
  tokenizer: AceQwenBpeTokenizer,
): ReadonlyMap<FsmState, readonly number[]> {
  const paths = new Map<FsmState, readonly number[]>();
  for (const [state, fixed] of Object.entries(FIXED_STRINGS) as [FsmState, string][]) {
    const byPosition: number[] = Array(fixed.length);
    let position = 0;
    while (position < fixed.length) {
      const remaining = fixed.slice(position);
      let tokenId: number | null = null;
      for (let end = remaining.length; end >= 1; end -= 1) {
        const encoded = tokenizer.encode(remaining.slice(0, end), {
          addSpecialTokens: false,
        });
        if (encoded.length === 1) {
          tokenId = encoded[0]!;
          break;
        }
      }
      if (tokenId === null) {
        throw new Error(`ACE planner metadata fixed string ${fixed} has no one-token prefix`);
      }
      const decoded = tokenizer.decode([tokenId]);
      if (decoded.length === 0 || !remaining.startsWith(decoded)) {
        throw new Error(`ACE planner metadata fixed string ${fixed} has an unsafe token boundary`);
      }
      byPosition[position] = tokenId;
      position += decoded.length;
    }
    paths.set(state, Object.freeze(byPosition));
  }
  return paths;
}

function buildValuePrefixTree(
  tokenizer: AceQwenBpeTokenizer,
  values: readonly string[],
  contextForMatching: string,
  contextForTokenization: string,
  acceptValueTokens: (tokens: readonly number[]) => boolean = () => true,
): PrefixTree {
  const context = tokenizer.encode(contextForMatching, { addSpecialTokens: false });
  const mutable = new Map<string, Set<number>>();
  for (const value of values) {
    const full = tokenizer.encode(contextForTokenization + value, {
      addSpecialTokens: false,
    });
    if (!startsWithNumbers(full, context)) {
      throw new Error(
        `ACE planner metadata value ${JSON.stringify(value)} does not retain context ${contextForMatching}`,
      );
    }
    const valueTokens = full.slice(context.length);
    if (valueTokens.length === 0 || !acceptValueTokens(valueTokens)) continue;
    for (let length = 0; length <= valueTokens.length; length += 1) {
      const prefix = tokenPrefixKey(valueTokens.slice(0, length));
      let continuations = mutable.get(prefix);
      if (continuations === undefined) {
        continuations = new Set<number>();
        mutable.set(prefix, continuations);
      }
      continuations.add(
        length < valueTokens.length ? valueTokens[length]! : NEWLINE_TOKEN_ID,
      );
    }
  }
  const continuations = new Map<string, readonly number[]>();
  for (const [prefix, allowed] of mutable) {
    continuations.set(prefix, Object.freeze([...allowed].sort((left, right) => left - right)));
  }
  return Object.freeze({ continuations });
}

function treeContinuation(
  tree: PrefixTree,
  prefix: readonly number[],
  field: string,
): readonly number[] {
  const allowed = tree.continuations.get(tokenPrefixKey(prefix));
  if (allowed === undefined || allowed.length === 0) {
    throw new Error(`ACE planner metadata ${field} prefix has no valid continuation`);
  }
  return allowed;
}

function assertLanguageTieOrder(tree: PrefixTree): void {
  const root = tree.continuations.get("");
  if (root === undefined || root.length !== UPSTREAM_LANGUAGE_FIRST_TOKEN_ORDER.length) {
    throw new Error("ACE planner metadata language root differs from the pinned tokenizer");
  }
  const rootSet = new Set(root);
  for (const tokenId of UPSTREAM_LANGUAGE_FIRST_TOKEN_ORDER) {
    if (!rootSet.has(tokenId)) {
      throw new Error("ACE planner metadata language tie order differs from the pinned tokenizer");
    }
  }
}

function singleTokenConstraint(tokenId: number): Omit<PendingConstraint, "step"> {
  requireVocabularyToken(tokenId, "ACE planner metadata allowed token");
  const tokenIds = Object.freeze([tokenId]);
  return Object.freeze({
    allowed: Object.freeze({ kind: "ids" as const, tokenIds }),
    admits: (candidate: number): boolean => candidate === tokenId,
  });
}

function idListConstraint(tokenIds: readonly number[]): Omit<PendingConstraint, "step"> {
  if (tokenIds.length === 0) {
    throw new Error("ACE planner metadata constraint removed every token");
  }
  const copy = Object.freeze([...new Set(tokenIds)]);
  for (const tokenId of copy) requireVocabularyToken(tokenId, "ACE planner metadata allowed token");
  const admitted = new Set(copy);
  return Object.freeze({
    allowed: Object.freeze({ kind: "ids" as const, tokenIds: copy }),
    admits: (candidate: number): boolean => admitted.has(candidate),
  });
}

function captionTokenConstraint(): Omit<PendingConstraint, "step"> {
  const firstAudio = ACE_PLANNER_AUDIO_CODE_FIRST_TOKEN_ID;
  const audioEnd = firstAudio + VALID_AUDIO_CODE_COUNT;
  return Object.freeze({
    allowed: CAPTION_ALLOWED_TOKENS,
    admits: (tokenId: number): boolean =>
      tokenId >= 0 &&
      tokenId < ACE_PLANNER_QWEN3_CONFIG.vocabularySize &&
      tokenId !== BACKTICK_TOKEN_ID &&
      !(tokenId >= firstAudio && tokenId < audioEnd),
  });
}

function allTokenConstraint(): Omit<PendingConstraint, "step"> {
  return Object.freeze({
    allowed: ALL_TOKENS,
    admits: (tokenId: number): boolean =>
      tokenId >= 0 && tokenId < ACE_PLANNER_QWEN3_CONFIG.vocabularySize,
  });
}

function rawArgmax(logits: ArrayLike<number>): number {
  let bestToken = -1;
  let bestLogit = Number.NEGATIVE_INFINITY;
  for (let tokenId = 0; tokenId < logits.length; tokenId += 1) {
    const value = requireRawLogit(logits[tokenId], `caption token ${tokenId}`);
    if (value > bestLogit) {
      bestLogit = value;
      bestToken = tokenId;
    }
  }
  if (bestToken < 0 || !Number.isFinite(bestLogit)) {
    throw new RangeError("ACE planner metadata caption transition has no finite candidate");
  }
  return bestToken;
}

function requireRawLogit(value: number | undefined, label: string): number {
  if (value === undefined || Number.isNaN(value) || value === Number.POSITIVE_INFINITY) {
    throw new RangeError(`ACE planner metadata ${label} must be finite or negative infinity`);
  }
  return Math.fround(value);
}

function integerStrings(first: number, last: number): readonly string[] {
  return Array.from({ length: last - first + 1 }, (_, index) => String(first + index));
}

function tokenPrefixKey(tokens: readonly number[]): string {
  return tokens.join(",");
}

function startsWithNumbers(values: readonly number[], prefix: readonly number[]): boolean {
  if (values.length < prefix.length) return false;
  return prefix.every((value, index) => values[index] === value);
}

function copyTokenSequence(values: readonly number[], label: string): number[] {
  const result: number[] = [];
  for (const value of values) {
    requireVocabularyToken(value, label);
    result.push(value);
  }
  return result;
}

function requireExactTokenSequence(
  actual: readonly number[],
  expected: readonly number[],
  label: string,
): void {
  if (actual.length !== expected.length) {
    throw new RangeError(`${label} has length ${actual.length}; ${expected.length} required`);
  }
  for (let index = 0; index < expected.length; index += 1) {
    requireVocabularyToken(actual[index]!, label);
    if (actual[index] !== expected[index]) {
      throw new Error(`${label} differs at token ${index}`);
    }
  }
}

function requireVocabularyToken(value: number, label: string): void {
  if (
    !Number.isSafeInteger(value) ||
    value < 0 ||
    value >= ACE_PLANNER_QWEN3_CONFIG.vocabularySize
  ) {
    throw new RangeError(`${label} ${String(value)} is outside the planner vocabulary`);
  }
}

function requireBooleanDefault(
  value: boolean | undefined,
  fallback: boolean,
  label: string,
): boolean {
  if (value === undefined) return fallback;
  if (typeof value !== "boolean") throw new TypeError(`${label} must be boolean`);
  return value;
}

function unexpectedState(state: FsmState): never {
  throw new Error(`Unreachable ACE planner metadata state ${String(state)}`);
}
