import {
  assertAceGenerationRequest,
  type AceGenerationRequest,
  type AceMusicMetadata,
  type AcePlannerEnabled,
} from "../api.js";
import {
  ACE_PLANNER_AUDIO_CODE_FIRST_TOKEN_ID,
  ACE_QWEN_IM_END_TOKEN_ID,
  ACE_QWEN_PAD_TOKEN_ID,
  AceQwenBpeTokenizer,
  renderAceQwenChat,
} from "../tokenizer/index.js";
import {
  ACE_PLANNER_QWEN3_CONFIG,
  createAceQwen3CausalControlData,
  type AceQwen3CausalControlData,
} from "../webgpu/qwen3.js";
import {
  ACE_BROWSER_SOFTMAX_V1,
  AcePlannerSamplingCursor,
  type AcePlannerAllowedTokens,
  type AcePlannerCursorSample,
  type AcePlannerSamplingParameters,
  type AcePlannerSoftmaxOracle,
} from "./planner-sampling.js";

export const ACE_PLANNER_INSTRUCTION =
  "Generate audio semantic tokens based on the given conditions:";
export const ACE_PLANNER_SEMANTIC_RATE_HZ = 5;
export const ACE_PLANNER_SEMANTIC_CODE_COUNT = 64_000;
export const ACE_PLANNER_SEMANTIC_FIRST_TOKEN_ID =
  ACE_PLANNER_AUDIO_CODE_FIRST_TOKEN_ID;
export const ACE_PLANNER_SEMANTIC_LAST_TOKEN_ID =
  ACE_PLANNER_SEMANTIC_FIRST_TOKEN_ID + ACE_PLANNER_SEMANTIC_CODE_COUNT - 1;

/**
 * Pinned orchestration references inspected for this browser control slice.
 */
export const ACE_PLANNER_ORCHESTRATION_CONTRACT = Object.freeze({
  aceSourceRevision: "6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0",
  plannerSnapshotRevision: "148d8ea0225bdab342ee1ae3a354275ccd60ca80",
  referenceBackend: "pytorch-eager",
  upstreamSources: Object.freeze([
    "acestep/llm_inference.py:1282-1520 generate_with_stop_condition",
    "acestep/llm_inference.py:985-1087 _run_pt_single left-padded CFG prefill",
    "acestep/llm_inference.py:2618-2765 _generate_with_cfg_custom",
    "acestep/constrained_logits_processor.py:1568-1628 duration/EOS FSM",
  ] as const),
  codeRows: 2,
  conditionalRow: 0,
  unconditionalRow: 1,
  paddingSide: "left",
  rotaryPositions: "physical-cache-slots",
  semanticTokenRange: Object.freeze([
    ACE_PLANNER_SEMANTIC_FIRST_TOKEN_ID,
    ACE_PLANNER_SEMANTIC_LAST_TOKEN_ID,
  ] as const),
  eosTokenId: ACE_QWEN_IM_END_TOKEN_ID,
} as const);

/**
 * Reviewed intentional product correction to the pinned source.
 *
 * The metadata FSM/parser emits `language`, but pinned
 * `acestep/inference.py:_update_metadata_from_lm` and its CoT override read
 * `vocal_language`. The browser remaps the generated field when requested;
 * native browser-policy captures must install the same one-line remap.
 */
export const ACE_PLANNER_BROWSER_POLICY = Object.freeze({
  id: "ace-planner-browser-policy-v1",
  generatedLanguageRemap: "language-to-vocal-language",
  sourceRevision: "6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0",
  sourceProducer:
    "acestep/llm_inference.py:2792-2877 -> parse_lm_output writes metadata['language']",
  sourceConsumer:
    "acestep/inference.py:803-807 -> use_cot_language reads metadata.get('vocal_language')",
  nativeCapturePatch:
    "acestep/inference.py:807 assigns params.vocal_language or metadata.get('language', dit_input_vocal_language)",
  explicitUserLanguagePrecedence: true,
  nativeCapturePatchRequired: true,
} as const);

export interface AcePlannerMetadata {
  readonly bpm?: number | string;
  readonly caption?: string;
  readonly duration?: number | string;
  readonly genres?: string;
  readonly keyscale?: string;
  readonly language?: string;
  readonly timesignature?: number | string;
}

export interface AcePlannerRequestPlan {
  readonly request: AceGenerationRequest;
  readonly configuration: AcePlannerEnabled;
  readonly userMetadata: AcePlannerMetadata;
  readonly shouldRunCot: boolean;
  readonly cotPrompt: string | null;
}

export interface AcePlannerFinalizedConditioning {
  /** Exact metadata serialized into the Phase-2 semantic-code think block. */
  readonly metadata: AcePlannerMetadata;
  readonly caption: string;
  readonly vocalLanguage: string;
  readonly cotText: string;
  readonly conditionalCodePrompt: string;
  readonly unconditionalCodePrompt: string;
}

export interface AcePlannerPrefillBatch {
  readonly kind: "prefill";
  readonly rows: number;
  readonly tokens: number;
  readonly cacheCapacity: number;
  readonly inputIds: Uint32Array;
  /** Row-major U32 validity. Left-padding is zero and content is one. */
  readonly keyValidity: Uint32Array;
  /** Exact physical IDs used by the pinned eager PyTorch Qwen3 forward. */
  readonly rotaryPositionIds: Uint32Array;
  readonly causal: AceQwen3CausalControlData;
  readonly conditionalRow: 0;
  readonly unconditionalRow: 1 | null;
}

export interface AcePlannerDecodeBatch {
  readonly kind: "decode";
  readonly rows: number;
  readonly tokens: 1;
  readonly cacheCapacity: number;
  readonly cachedTokensBeforeAppend: number;
  readonly inputIds: Uint32Array;
  readonly rotaryPositionIds: Uint32Array;
  readonly causal: AceQwen3CausalControlData;
  readonly conditionalRow: 0;
  readonly unconditionalRow: 1 | null;
}

/** Ascending contiguous global-token interval requested from the tied head. */
export interface AcePlannerLogitRange {
  readonly firstTokenId: number;
  readonly tokenCount: number;
}

/**
 * Whole-model boundary consumed by the control coordinator.
 *
 * `prefill` starts a fresh phase and clears the graph's KV/cache-validity
 * storage. `decode` appends to that same cache. Both return one FP32 logical
 * vocabulary row per input row, ordered by token ID. GPU readback is allowed
 * in Stage 1; keeping sampling resident is a later Stage 2 experiment.
 */
export interface AcePlannerGraphExecutor {
  prefill(
    batch: AcePlannerPrefillBatch,
    logitRange?: AcePlannerLogitRange,
  ): Promise<readonly ArrayLike<number>[]>;
  decode(
    batch: AcePlannerDecodeBatch,
    logitRange?: AcePlannerLogitRange,
  ): Promise<readonly ArrayLike<number>[]>;
}

export interface AcePlannerCotConstraintInput {
  readonly step: number;
  readonly promptTokenIds: readonly number[];
  readonly emittedTokenIds: readonly number[];
  readonly logits: ArrayLike<number>;
}

export interface AcePlannerCotAcceptedToken {
  readonly step: number;
  readonly tokenId: number;
  readonly tokenText: string;
  readonly emittedTokenIds: readonly number[];
}

/**
 * Adapter boundary for the metadata FSM. The controller owns only CoT state;
 * CFG, filtering, Philox word order, KV controls, and termination remain here.
 */
export interface AcePlannerCotConstraintController {
  allowedTokens(input: AcePlannerCotConstraintInput): AcePlannerAllowedTokens;
  acceptToken(input: AcePlannerCotAcceptedToken): Readonly<{
    readonly finished: boolean;
  }>;
}

export interface AcePlannerCotPhaseOptions {
  readonly graph: AcePlannerGraphExecutor;
  readonly tokenizer: AceQwenBpeTokenizer;
  readonly prompt: string;
  readonly cursor: AcePlannerSamplingCursor;
  readonly constraint: AcePlannerCotConstraintController;
  readonly sampling: Omit<AcePlannerSamplingParameters, "guidanceScale">;
  /** Defaults to the accepted deterministic browser-v1 implementation. */
  readonly softmax?: AcePlannerSoftmaxOracle;
  readonly maxNewTokens: number;
  /** Exact allocation capacity; defaults to prompt tokens plus this phase's draws. */
  readonly cacheCapacity?: number;
  readonly signal?: AbortSignal;
  readonly onToken?: (sample: AcePlannerCursorSample) => void;
}

export interface AcePlannerCotPhaseResult {
  readonly emittedTokenIds: readonly number[];
  readonly outputText: string;
  readonly parsedMetadata: AcePlannerMetadata;
  readonly drawStart: bigint;
  readonly drawEnd: bigint;
  readonly stoppedBy: "constraint" | "eos" | "pad";
}

export interface AcePlannerSemanticPhaseOptions {
  readonly graph: AcePlannerGraphExecutor;
  readonly tokenizer: AceQwenBpeTokenizer;
  readonly conditionalPrompt: string;
  readonly unconditionalPrompt: string;
  readonly cursor: AcePlannerSamplingCursor;
  readonly sampling: Omit<AcePlannerSamplingParameters, "repetitionPenalty">;
  /** Defaults to the accepted deterministic browser-v1 implementation. */
  readonly softmax?: AcePlannerSoftmaxOracle;
  readonly durationSeconds: number;
  /** Exact allocation capacity; defaults to prompt tokens plus the target codes. */
  readonly cacheCapacity?: number;
  readonly signal?: AbortSignal;
  readonly onToken?: (sample: AcePlannerCursorSample) => void;
  /** @internal OPT-0082 candidate seam; ordinary production retains full logits. */
  readonly compactLogits?: boolean;
}

export interface AcePlannerSemanticPhaseResult {
  readonly semanticCodeValues: readonly number[];
  /** The constrained code tokens followed by the one EOS token. */
  readonly emittedTokenIds: readonly number[];
  readonly audioCodeText: string;
  readonly drawStart: bigint;
  readonly drawEnd: bigint;
  readonly prefillPhysicalTokens: number;
}

/** Exact conditional CoT prompt before planner tokenization. */
export function createAcePlannerCotPrompt(caption: string, lyrics: string): string {
  requireString(caption, "ACE planner caption");
  requireString(lyrics, "ACE planner lyrics");
  return renderAceQwenChat(
    [
      { role: "system", content: plannerSystemMessage() },
      {
        role: "user",
        content: `# Caption\n${caption}\n\n# Lyric\n${lyrics}\n`,
      },
    ],
    { addGenerationPrompt: true },
  );
}

/**
 * Exact two code-phase prompts.
 *
 * The unconditional row uses raw `NO USER INPUT`, an empty think block, and
 * an open assistant turn. It deliberately does not retain caption or lyrics.
 */
export function createAcePlannerCodePrompts(
  caption: string,
  lyrics: string,
  cotText: string,
  negativePrompt: "NO USER INPUT" = "NO USER INPUT",
): Readonly<{
  readonly conditional: string;
  readonly unconditional: string;
}> {
  requireString(caption, "ACE planner code caption");
  requireString(lyrics, "ACE planner code lyrics");
  requireCotText(cotText);
  if (negativePrompt !== "NO USER INPUT") {
    throw new TypeError("ACE planner v1 requires the pinned negative prompt");
  }
  const conditional = renderAceQwenChat(
    [
      { role: "system", content: plannerSystemMessage() },
      {
        role: "user",
        content: `# Caption\n${caption}\n\n# Lyric\n${lyrics}\n`,
      },
    ],
    { addGenerationPrompt: true },
  ) + `${cotText}\n\n`;
  const unconditional = renderAceQwenChat(
    [
      { role: "system", content: plannerSystemMessage() },
      { role: "user", content: "NO USER INPUT" },
    ],
    { addGenerationPrompt: true },
  ) + "<think>\n\n</think>\n\n";
  return Object.freeze({ conditional, unconditional });
}

/** Resolve whether the pinned two-phase path runs CoT before semantic codes. */
export function createAcePlannerRequestPlan(
  request: AceGenerationRequest,
): AcePlannerRequestPlan {
  assertAceGenerationRequest(request);
  if (request.planner.mode !== "enabled") {
    throw new TypeError("ACE planner request planning requires enabled mode");
  }
  const userMetadata = createAcePlannerUserMetadata(
    request.durationSeconds,
    request.metadata,
  );
  // Pinned upstream gates the entire CoT phase on missing core metadata plus
  // `use_cot_metas`; caption/language toggles only alter fields inside it.
  const shouldRunCot =
    request.planner.thinking.useCotMissingMetadata &&
    !hasAllAcePlannerCoreMetadata(userMetadata);
  const cotPrompt = shouldRunCot
    ? createAcePlannerCotPrompt(request.prompt, request.lyrics ?? "")
    : null;
  return Object.freeze({
    request,
    configuration: request.planner,
    userMetadata,
    shouldRunCot,
    cotPrompt,
  });
}

/**
 * Parse/merge Phase 1 and construct the exact two-row Phase 2 prompt pair.
 * Explicit request metadata wins generated metadata.
 */
export function finalizeAcePlannerConditioning(
  plan: AcePlannerRequestPlan,
  cotOutputText: string | null,
): AcePlannerFinalizedConditioning {
  if (plan.shouldRunCot && cotOutputText === null) {
    throw new TypeError("ACE planner CoT output is required for this request");
  }
  if (!plan.shouldRunCot && cotOutputText !== null) {
    throw new TypeError("ACE planner received unexpected CoT output for a skipped phase");
  }
  const parsed: AcePlannerMetadata = cotOutputText === null
    ? Object.freeze({})
    : parseAcePlannerCotMetadata(cotOutputText);
  // The pinned constraint FSM omits disabled generated fields entirely. Keep
  // the same contract even if an injected/test controller emits them anyway.
  const generated = filterGeneratedPlannerMetadata(
    parsed,
    plan.configuration.thinking.useCotCaption,
    plan.configuration.thinking.useCotLanguage,
  );
  const merged = mergePlannerMetadata(generated, plan.userMetadata);
  const generatedCaption = generated.caption?.trim();
  const caption =
    plan.configuration.thinking.useCotCaption && generatedCaption
      ? generatedCaption
      : plan.request.prompt;
  const generatedLanguage = generated.language?.trim();
  const requestedLanguage = plan.request.metadata?.vocalLanguage?.trim();
  const vocalLanguage =
    requestedLanguage ||
    (plan.configuration.thinking.useCotLanguage && generatedLanguage
      ? generatedLanguage
      : "unknown");
  // Phase 2 serializes exactly the Phase-1/user-core metadata. Caption is a
  // separate prompt argument and vocal language is a downstream DiT setting;
  // injecting either here when CoT skipped them changes semantic-code logits.
  const metadata = merged;
  const cotText = formatAcePlannerCotMetadata(metadata);
  const prompts = createAcePlannerCodePrompts(
    // Pinned Phase 2 retains the original caller caption in the user block;
    // a generated rewrite exists only inside the serialized think metadata
    // and becomes the downstream DiT caption after planner return.
    plan.request.prompt,
    plan.request.lyrics ?? "",
    cotText,
    plan.configuration.negativePrompt,
  );
  return Object.freeze({
    metadata,
    caption,
    vocalLanguage,
    cotText,
    conditionalCodePrompt: prompts.conditional,
    unconditionalCodePrompt: prompts.unconditional,
  });
}

/** Upstream-compatible metadata extraction from the reasoning region. */
export function parseAcePlannerCotMetadata(outputText: string): AcePlannerMetadata {
  requireString(outputText, "ACE planner CoT output");
  const tagged = /<think>([\s\S]*?)<\/think>/.exec(outputText);
  const beforeCodes = outputText.split("<|audio_code_", 1)[0]!;
  const reasoning = (tagged?.[1] ?? beforeCodes).trim();
  if (reasoning.length === 0) {
    throw new RangeError("ACE planner CoT output does not contain metadata");
  }

  const result: MutablePlannerMetadata = {};
  let key: string | null = null;
  let valueLines: string[] = [];
  const save = (): void => {
    if (key === null || valueLines.length === 0) {
      key = null;
      valueLines = [];
      return;
    }
    const value = valueLines.join("\n");
    switch (key) {
      case "bpm":
        result.bpm = parsePlannerIntegerOrString(value);
        break;
      case "caption":
        result.caption = collapsePlannerCaption(value);
        break;
      case "duration":
        result.duration = parsePlannerIntegerOrString(value);
        break;
      case "genres":
        result.genres = value.trim();
        break;
      case "keyscale":
        result.keyscale = value.trim();
        break;
      case "language":
        result.language = value.trim();
        break;
      case "timesignature":
        result.timesignature = value.trim();
        break;
      default:
        break;
    }
    key = null;
    valueLines = [];
  };

  for (const line of reasoning.split("\n")) {
    if (line.trimStart().startsWith("<")) continue;
    if (line.length > 0 && !/^\s/.test(line) && line.includes(":")) {
      save();
      const separator = line.indexOf(":");
      key = line.slice(0, separator).trim().toLowerCase();
      const firstValue = line.slice(separator + 1);
      if (firstValue.trim().length > 0) valueLines.push(firstValue);
    } else if (/^[ \t]/.test(line) && key !== null) {
      valueLines.push(line);
    }
  }
  save();
  if (Object.keys(result).length === 0) {
    throw new RangeError("ACE planner CoT output has no recognized metadata fields");
  }
  return Object.freeze(result);
}

/**
 * Deterministic PyYAML-compatible scalar subset used by pinned constrained
 * fields. Captions are normalized to one line by the upstream parser first.
 */
export function formatAcePlannerCotMetadata(metadata: AcePlannerMetadata): string {
  const lines: string[] = [];
  for (const key of [
    "bpm",
    "caption",
    "duration",
    "keyscale",
    "language",
    "timesignature",
  ] as const) {
    let value = metadata[key];
    if (value === undefined) continue;
    if (key === "timesignature" && typeof value === "string" && value.endsWith("/4")) {
      value = value.split("/", 1)[0]!;
    }
    if (typeof value === "string" && /^\d+$/.test(value)) {
      value = Number(value);
    } else if (
      typeof value === "string" &&
      value.length > 0 &&
      /^\p{Decimal_Number}+$/u.test(value)
    ) {
      // Python's `str.isdigit()` + `int()` accepts Unicode decimal scripts,
      // while JavaScript Number() does not. Until that conversion is given an
      // authenticated browser implementation, reject instead of serializing a
      // string that upstream would have converted to an integer.
      throw new RangeError(
        `ACE planner metadata ${key} uses unsupported non-ASCII decimal digits`,
      );
    }
    lines.push(...yamlScalarLines(key, value));
  }
  return `<think>\n${lines.join("\n")}\n</think>`;
}

/** Construct exact left-padded prefill/cache controls for one or two rows. */
export function createAcePlannerPrefillBatch(
  tokenizer: AceQwenBpeTokenizer,
  prompts: readonly [string] | readonly [string, string],
  cacheCapacity = ACE_PLANNER_QWEN3_CONFIG.maximumPositionEmbeddings,
): AcePlannerPrefillBatch {
  requirePlannerTokenizer(tokenizer);
  requireCacheCapacity(cacheCapacity);
  if (prompts.length !== 1 && prompts.length !== 2) {
    throw new RangeError("ACE planner prefill requires one row or a two-row CFG pair");
  }
  for (const prompt of prompts) requireString(prompt, "ACE planner prompt");
  const tokenized = tokenizer.encodeBatch(prompts, {
    addSpecialTokens: true,
    truncation: true,
    maxLength: cacheCapacity,
    padding: "longest",
    paddingSide: "left",
  });
  const rows = prompts.length;
  const tokens = tokenized.inputIds[0]?.length ?? 0;
  if (tokens <= 0 || tokens >= cacheCapacity) {
    throw new RangeError("ACE planner prefill leaves no physical cache slot for decoding");
  }
  const inputIds = flattenTokenRows(tokenized.inputIds, rows, tokens);
  const keyValidity = flattenValidityRows(tokenized.attentionMask, rows, tokens);
  const rowStarts = Array<number>(rows).fill(0);
  const validLengths = Array<number>(rows).fill(tokens);
  const causal = createAceQwen3CausalControlData({
    batch: rows,
    tokens,
    cacheCapacity,
    rowStartPositions: rowStarts,
    validKeyLengths: validLengths,
    sourceValidity: [...keyValidity],
  });
  // Pinned eager PyTorch does not pass position_ids. Qwen3 derives the shared
  // physical cache_position arange, so left pads consume rotary positions.
  const rotaryPositionIds = causal.queryPositions.slice();
  return Object.freeze({
    kind: "prefill",
    rows,
    tokens,
    cacheCapacity,
    inputIds,
    keyValidity,
    rotaryPositionIds,
    causal,
    conditionalRow: 0,
    unconditionalRow: rows === 2 ? 1 : null,
  });
}

/** Construct one-token cache append controls; CFG rows receive one shared token. */
export function createAcePlannerDecodeBatch(
  tokenId: number,
  rows: 1 | 2,
  cachedTokensBeforeAppend: number,
  cacheCapacity = ACE_PLANNER_QWEN3_CONFIG.maximumPositionEmbeddings,
): AcePlannerDecodeBatch {
  requireCacheCapacity(cacheCapacity);
  requirePlannerVocabularyToken(tokenId);
  if (!Number.isSafeInteger(cachedTokensBeforeAppend) || cachedTokensBeforeAppend <= 0) {
    throw new RangeError("ACE planner cached token count must be positive");
  }
  if (cachedTokensBeforeAppend >= cacheCapacity) {
    throw new RangeError("ACE planner decode append exceeds cache capacity");
  }
  const inputIds = new Uint32Array(rows);
  inputIds.fill(tokenId);
  const causal = createAceQwen3CausalControlData({
    batch: rows,
    tokens: 1,
    cacheCapacity,
    rowStartPositions: Array<number>(rows).fill(cachedTokensBeforeAppend),
    validKeyLengths: Array<number>(rows).fill(cachedTokensBeforeAppend + 1),
    sourceValidity: Array<number>(rows).fill(1),
  });
  return Object.freeze({
    kind: "decode",
    rows,
    tokens: 1,
    cacheCapacity,
    cachedTokensBeforeAppend,
    inputIds,
    rotaryPositionIds: causal.queryPositions.slice(),
    causal,
    conditionalRow: 0,
    unconditionalRow: rows === 2 ? 1 : null,
  });
}

/** Run one constrained, non-CFG CoT phase with a fresh graph cache. */
export async function runAcePlannerCotPhase(
  options: AcePlannerCotPhaseOptions,
): Promise<AcePlannerCotPhaseResult> {
  requirePlannerTokenizer(options.tokenizer);
  requirePositiveInteger(options.maxNewTokens, "ACE planner CoT maxNewTokens");
  const { cacheCapacity, prefill } = createPlannerPhasePrefill(
    options.tokenizer,
    [options.prompt],
    // A terminal token is sampled from the last logits row and is not
    // appended. Keep one spare slot because prefill itself rejects a full
    // cache, including the maxNewTokens=1 edge.
    Math.max(1, options.maxNewTokens - 1),
    options.cacheCapacity,
  );
  if (prefill.tokens + options.maxNewTokens - 1 > cacheCapacity) {
    throw new RangeError("ACE planner CoT phase can exceed cache capacity");
  }
  const drawStart = options.cursor.consumed;
  const promptTokens = [...prefill.inputIds];
  const emitted: number[] = [];
  let logits = requirePlannerGraphLogits(
    await options.graph.prefill(prefill),
    1,
    "ACE planner CoT prefill",
  );

  for (let step = 0; step < options.maxNewTokens; step += 1) {
    throwIfAborted(options.signal);
    const allowedTokens = options.constraint.allowedTokens({
      step,
      promptTokenIds: promptTokens,
      emittedTokenIds: emitted,
      logits: logits[0]!,
    });
    const sample = options.cursor.sample({
      conditionalLogits: logits[0]!,
      seenTokenIds: [...promptTokens, ...emitted],
      allowedTokens,
      parameters: {
        ...options.sampling,
        guidanceScale: 1,
      },
      softmax: options.softmax ?? ACE_BROWSER_SOFTMAX_V1,
    });
    emitted.push(sample.tokenId);
    options.onToken?.(sample);
    const tokenText = options.tokenizer.decode([sample.tokenId]);
    const accepted = options.constraint.acceptToken({
      step,
      tokenId: sample.tokenId,
      tokenText,
      emittedTokenIds: emitted,
    });
    const stop = sample.tokenId === ACE_QWEN_IM_END_TOKEN_ID
      ? "eos"
      : sample.tokenId === ACE_QWEN_PAD_TOKEN_ID
        ? "pad"
        : accepted.finished
          ? "constraint"
          : null;
    if (stop !== null) {
      const outputText = options.tokenizer.decode(emitted);
      return Object.freeze({
        emittedTokenIds: Object.freeze([...emitted]),
        outputText,
        parsedMetadata: parseAcePlannerCotMetadata(outputText),
        drawStart,
        drawEnd: options.cursor.consumed,
        stoppedBy: stop,
      });
    }
    const decode = createAcePlannerDecodeBatch(
      sample.tokenId,
      1,
      prefill.tokens + emitted.length - 1,
      cacheCapacity,
    );
    logits = requirePlannerGraphLogits(
      await options.graph.decode(decode),
      1,
      `ACE planner CoT decode step ${step}`,
    );
  }
  throw new Error("ACE planner CoT exhausted maxNewTokens without termination");
}

/**
 * Run the exact two-row semantic phase through target codes plus forced EOS.
 */
export async function runAcePlannerSemanticPhase(
  options: AcePlannerSemanticPhaseOptions,
): Promise<AcePlannerSemanticPhaseResult> {
  requirePlannerTokenizer(options.tokenizer);
  if (
    !Number.isSafeInteger(options.durationSeconds) ||
    options.durationSeconds <= 0
  ) {
    throw new RangeError("ACE planner semantic duration must be a positive integer");
  }
  const targetCodes = options.durationSeconds * ACE_PLANNER_SEMANTIC_RATE_HZ;
  if (!Number.isSafeInteger(targetCodes) || targetCodes <= 0) {
    throw new RangeError("ACE planner semantic code count is not a safe integer");
  }
  const cfgEnabled = options.sampling.guidanceScale > 1;
  const { cacheCapacity, prefill } = createPlannerPhasePrefill(
    options.tokenizer,
    cfgEnabled
      ? [options.conditionalPrompt, options.unconditionalPrompt]
      : [options.conditionalPrompt],
    // Every one of the N semantic codes is appended before the forced-EOS
    // logits row, so those N physical cache slots are all required.
    targetCodes,
    options.cacheCapacity,
  );
  if (prefill.tokens + targetCodes > cacheCapacity) {
    throw new RangeError("ACE planner semantic phase exceeds cache capacity");
  }
  const drawStart = options.cursor.consumed;
  const conditionalPromptTokens = [
    ...prefill.inputIds.slice(0, prefill.tokens),
  ];
  const emitted: number[] = [];
  const codeValues: number[] = [];
  const regularLogitRange: AcePlannerLogitRange = Object.freeze({
    firstTokenId: ACE_PLANNER_SEMANTIC_FIRST_TOKEN_ID,
    tokenCount: ACE_PLANNER_SEMANTIC_CODE_COUNT,
  });
  const eosLogitRange: AcePlannerLogitRange = Object.freeze({
    firstTokenId: ACE_QWEN_IM_END_TOKEN_ID,
    tokenCount: 1,
  });
  const compactLogits = options.compactLogits === true;
  if (compactLogits && !cfgEnabled) {
    throw new Error("OPT-0082 compact semantic logits require the two-row CFG path");
  }
  if (
    compactLogits &&
    options.softmax !== undefined &&
    options.softmax !== ACE_BROWSER_SOFTMAX_V1
  ) {
    throw new Error("OPT-0082 compact semantic logits require browser softmax v1");
  }
  let logits = requirePlannerGraphLogits(
    await options.graph.prefill(
      prefill,
      compactLogits ? regularLogitRange : undefined,
    ),
    cfgEnabled ? 2 : 1,
    "ACE planner semantic prefill",
    compactLogits
      ? regularLogitRange.tokenCount
      : ACE_PLANNER_QWEN3_CONFIG.vocabularySize,
  );

  for (let step = 0; step <= targetCodes; step += 1) {
    throwIfAborted(options.signal);
    const forcingEos = codeValues.length === targetCodes;
    const logitRange = forcingEos ? eosLogitRange : regularLogitRange;
    const sample = compactLogits
      ? options.cursor.sampleCompact({
          firstTokenId: logitRange.firstTokenId,
          vocabularySize: ACE_PLANNER_QWEN3_CONFIG.vocabularySize,
          conditionalLogits: logits[0]!,
          ...(cfgEnabled ? { unconditionalLogits: logits[1]! } : {}),
          seenTokenIds: [...conditionalPromptTokens, ...emitted],
          parameters: {
            ...options.sampling,
            repetitionPenalty: 1,
          },
          softmax: options.softmax ?? ACE_BROWSER_SOFTMAX_V1,
        })
      : options.cursor.sample({
          conditionalLogits: logits[0]!,
          ...(cfgEnabled ? { unconditionalLogits: logits[1]! } : {}),
          seenTokenIds: [...conditionalPromptTokens, ...emitted],
          preCfgAllowedTokens: {
            kind: "range",
            firstTokenId: ACE_PLANNER_SEMANTIC_FIRST_TOKEN_ID,
            tokenCount: ACE_PLANNER_SEMANTIC_CODE_COUNT,
            additionalTokenIds: [ACE_QWEN_IM_END_TOKEN_ID],
          },
          allowedTokens: forcingEos
            ? { kind: "ids", tokenIds: [ACE_QWEN_IM_END_TOKEN_ID] }
            : {
                kind: "range",
                firstTokenId: ACE_PLANNER_SEMANTIC_FIRST_TOKEN_ID,
                tokenCount: ACE_PLANNER_SEMANTIC_CODE_COUNT,
              },
          parameters: {
            ...options.sampling,
            repetitionPenalty: 1,
          },
          softmax: options.softmax ?? ACE_BROWSER_SOFTMAX_V1,
        });
    emitted.push(sample.tokenId);
    options.onToken?.(sample);

    if (forcingEos) {
      if (sample.tokenId !== ACE_QWEN_IM_END_TOKEN_ID) {
        throw new Error("ACE planner semantic phase did not terminate with EOS");
      }
      break;
    }
    const code = plannerTokenIdToSemanticCode(sample.tokenId);
    codeValues.push(code);
    const decode = createAcePlannerDecodeBatch(
      sample.tokenId,
      cfgEnabled ? 2 : 1,
      prefill.tokens + emitted.length - 1,
      cacheCapacity,
    );
    const nextLogitRange = codeValues.length === targetCodes
      ? eosLogitRange
      : regularLogitRange;
    logits = requirePlannerGraphLogits(
      await options.graph.decode(
        decode,
        compactLogits ? nextLogitRange : undefined,
      ),
      cfgEnabled ? 2 : 1,
      `ACE planner semantic decode step ${step}`,
      compactLogits
        ? nextLogitRange.tokenCount
        : ACE_PLANNER_QWEN3_CONFIG.vocabularySize,
    );
  }

  const parsed = parseAcePlannerSemanticTokenSequence(emitted, targetCodes);
  if (!equalNumbers(parsed, codeValues)) {
    throw new Error("ACE planner parsed semantic codes differ from sampled tokens");
  }
  return Object.freeze({
    semanticCodeValues: Object.freeze([...codeValues]),
    emittedTokenIds: Object.freeze([...emitted]),
    audioCodeText: serializeAcePlannerAudioCodes(codeValues),
    drawStart,
    drawEnd: options.cursor.consumed,
    prefillPhysicalTokens: prefill.tokens,
  });
}

/** Validate `N` constrained audio tokens followed by exactly one EOS token. */
export function parseAcePlannerSemanticTokenSequence(
  tokenIds: readonly number[],
  expectedCodes: number,
): readonly number[] {
  if (!Number.isSafeInteger(expectedCodes) || expectedCodes < 0) {
    throw new RangeError("ACE planner expected code count must be non-negative");
  }
  if (tokenIds.length !== expectedCodes + 1) {
    throw new RangeError(
      `ACE planner emitted ${tokenIds.length} semantic-phase tokens; ` +
        `${expectedCodes + 1} required including EOS`,
    );
  }
  const values: number[] = [];
  for (let index = 0; index < expectedCodes; index += 1) {
    values.push(plannerTokenIdToSemanticCode(tokenIds[index]!));
  }
  if (tokenIds[expectedCodes] !== ACE_QWEN_IM_END_TOKEN_ID) {
    throw new RangeError("ACE planner semantic token sequence does not end in EOS");
  }
  return Object.freeze(values);
}

export function serializeAcePlannerAudioCodes(codeValues: readonly number[]): string {
  return codeValues.map((value) => {
    requireSemanticCode(value);
    return `<|audio_code_${value}|>`;
  }).join("");
}

/**
 * Avoid treating Qwen's 40,960-position architectural ceiling as an
 * allocation request. At two CFG rows that would reserve multi-gigabyte K/V
 * storage despite a typical phase using only a few thousand physical slots.
 */
function createPlannerPhasePrefill(
  tokenizer: AceQwenBpeTokenizer,
  prompts: readonly [string] | readonly [string, string],
  appendSlots: number,
  explicitCapacity: number | undefined,
): Readonly<{
  readonly cacheCapacity: number;
  readonly prefill: AcePlannerPrefillBatch;
}> {
  requirePositiveInteger(appendSlots, "ACE planner append slots");
  if (explicitCapacity !== undefined) {
    return Object.freeze({
      cacheCapacity: explicitCapacity,
      prefill: createAcePlannerPrefillBatch(
        tokenizer,
        prompts,
        explicitCapacity,
      ),
    });
  }

  const ceiling = ACE_PLANNER_QWEN3_CONFIG.maximumPositionEmbeddings;
  const provisional = createAcePlannerPrefillBatch(tokenizer, prompts, ceiling);
  const cacheCapacity = provisional.tokens + appendSlots;
  if (!Number.isSafeInteger(cacheCapacity) || cacheCapacity > ceiling) {
    throw new RangeError("ACE planner phase exceeds maximumPositionEmbeddings");
  }
  return Object.freeze({
    cacheCapacity,
    prefill: cacheCapacity === ceiling
      ? provisional
      : createAcePlannerPrefillBatch(tokenizer, prompts, cacheCapacity),
  });
}

export function plannerTokenIdToSemanticCode(tokenId: number): number {
  requirePlannerVocabularyToken(tokenId);
  const value = tokenId - ACE_PLANNER_SEMANTIC_FIRST_TOKEN_ID;
  requireSemanticCode(value);
  return value;
}

function plannerSystemMessage(): string {
  return `# Instruction\n${ACE_PLANNER_INSTRUCTION}\n\n`;
}

function createAcePlannerUserMetadata(
  durationSeconds: number,
  metadata: AceMusicMetadata | undefined,
): AcePlannerMetadata {
  const result: MutablePlannerMetadata = { duration: durationSeconds };
  if (metadata?.bpm !== undefined) result.bpm = metadata.bpm;
  const keyscale = metadata?.keyScale?.trim();
  if (keyscale) result.keyscale = keyscale;
  const timesignature = metadata?.timeSignature?.trim();
  if (timesignature) result.timesignature = timesignature;
  return Object.freeze(result);
}

function filterGeneratedPlannerMetadata(
  metadata: AcePlannerMetadata,
  includeCaption: boolean,
  includeLanguage: boolean,
): AcePlannerMetadata {
  const result: MutablePlannerMetadata = { ...metadata };
  // The pinned product CoT phase always configures skip_genres=true. Keep the
  // parser tolerant for reference inspection, but never let an injected or
  // buggy controller smuggle the skipped field into Phase 2.
  delete result.genres;
  if (!includeCaption) delete result.caption;
  if (!includeLanguage) delete result.language;
  return Object.freeze(result);
}

function hasAllAcePlannerCoreMetadata(metadata: AcePlannerMetadata): boolean {
  return (
    metadata.bpm !== undefined &&
    metadata.keyscale !== undefined &&
    metadata.timesignature !== undefined &&
    metadata.duration !== undefined
  );
}

function mergePlannerMetadata(
  base: AcePlannerMetadata,
  override: AcePlannerMetadata,
): AcePlannerMetadata {
  const result: MutablePlannerMetadata = { ...base };
  for (const key of [
    "bpm",
    "caption",
    "duration",
    "genres",
    "keyscale",
    "language",
    "timesignature",
  ] as const) {
    const value = override[key];
    if (value !== undefined) result[key] = value as never;
  }
  return Object.freeze(result);
}

function yamlScalarLines(
  key: string,
  value: number | string,
): string[] {
  if (typeof value === "number") {
    if (!Number.isSafeInteger(value)) {
      throw new RangeError(`ACE planner metadata ${key} must be a safe integer`);
    }
    return [`${key}: ${value}`];
  }
  requirePyYamlSingleLineScalar(value, key);
  const plain = pyYamlBlockPlainAllowed(value) && !pyYamlImplicitlyTyped(value);
  return emitPyYamlScalarLines(key, value, plain);
}

/** Exact bounded subset of pinned PyYAML 6.0.3 scalar analysis. */
function requirePyYamlSingleLineScalar(value: string, key: string): void {
  for (const character of value) {
    const code = character.codePointAt(0)!;
    if (
      code < 0x20 ||
      (code >= 0x7f && code <= 0x9f) ||
      code === 0xfeff ||
      code === 0xfffe ||
      code === 0xffff ||
      code === 0x10ffff ||
      code === 0x2028 ||
      code === 0x2029 ||
      (code >= 0xd800 && code <= 0xdfff)
    ) {
      throw new RangeError(
        `ACE planner metadata ${key} requires unsupported PyYAML escaping`,
      );
    }
  }
}

function pyYamlBlockPlainAllowed(value: string): boolean {
  if (value.length === 0 || value.startsWith(" ") || value.endsWith(" ")) {
    return false;
  }
  if (value.startsWith("---") || value.startsWith("...")) return false;
  const characters = Array.from(value);
  const first = characters[0]!;
  if ("#,[]{}&*!|>'\"%@`".includes(first)) return false;
  if (
    (first === "?" || first === ":" || first === "-") &&
    (characters.length === 1 || characters[1] === " ")
  ) {
    return false;
  }
  for (let index = 1; index < characters.length; index += 1) {
    const character = characters[index]!;
    if (
      character === ":" &&
      (index + 1 === characters.length || characters[index + 1] === " ")
    ) {
      return false;
    }
    if (character === "#" && characters[index - 1] === " ") return false;
  }
  return true;
}

/** Pinned YAML-1.1 implicit resolvers used by PyYAML 6.0.3. */
function pyYamlImplicitlyTyped(value: string): boolean {
  return (
    /^(?:yes|Yes|YES|no|No|NO|true|True|TRUE|false|False|FALSE|on|On|ON|off|Off|OFF)$/.test(value) ||
    /^(?:~|null|Null|NULL|)$/.test(value) ||
    /^(?:[-+]?0b[0-1_]+|[-+]?0[0-7_]+|[-+]?(?:0|[1-9][0-9_]*)|[-+]?0x[0-9a-fA-F_]+|[-+]?[1-9][0-9_]*(?::[0-5]?[0-9])+)$/.test(value) ||
    /^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+][0-9]+)?|\.[0-9][0-9_]*(?:[eE][-+][0-9]+)?|[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\.[0-9_]*|[-+]?\.(?:inf|Inf|INF)|\.(?:nan|NaN|NAN))$/.test(value) ||
    /^(?:[0-9]{4}-[0-9]{2}-[0-9]{2}|[0-9]{4} -[0-9]{1,2} -[0-9]{1,2}(?:[Tt]|[ \t]+)[0-9]{1,2} :[0-9]{2} :[0-9]{2}(?:\.[0-9]*)?(?:[ \t]*(?:Z|[-+][0-9]{1,2}(?::[0-9]{2})?))?)$/.test(value) ||
    value === "<<" ||
    value === "=" ||
    value === "!" ||
    value === "&" ||
    value === "*"
  );
}

/** Port of PyYAML Emitter.write_plain/write_single_quoted at width 80. */
function emitPyYamlScalarLines(
  key: string,
  value: string,
  plain: boolean,
): string[] {
  const prefix = `${key}: `;
  const characters = Array.from(value);
  const lines = [prefix + (plain ? "" : "'")];
  let column = Array.from(lines[0]!).length;
  let spaces = false;
  let start = 0;
  let end = 0;
  const append = (text: string): void => {
    lines[lines.length - 1] += text;
    column += Array.from(text).length;
  };
  const indent = (): void => {
    lines.push("  ");
    column = 2;
  };
  while (end <= characters.length) {
    const character = end < characters.length ? characters[end]! : null;
    if (spaces) {
      if (character !== " ") {
        const internal = plain || (start !== 0 && end !== characters.length);
        if (start + 1 === end && column > 80 && internal) {
          indent();
        } else {
          append(characters.slice(start, end).join(""));
        }
        start = end;
      }
    } else if (
      character === null ||
      character === " " ||
      (!plain && character === "'")
    ) {
      if (start < end) append(characters.slice(start, end).join(""));
      start = end;
    }
    if (!plain && character === "'") {
      append("''");
      start = end + 1;
    }
    if (character !== null) spaces = character === " ";
    end += 1;
  }
  if (!plain) append("'");
  return lines;
}

function parsePlannerIntegerOrString(value: string): number | string {
  const trimmed = value.trim();
  if (/^\d+$/.test(trimmed)) {
    const parsed = Number(trimmed);
    if (Number.isSafeInteger(parsed)) return parsed;
  }
  return trimmed;
}

function collapsePlannerCaption(value: string): string {
  return value.split("\n").map((line) => line.trim()).filter(Boolean).join(" ");
}

function flattenTokenRows(
  rows: readonly (readonly number[])[],
  expectedRows: number,
  expectedTokens: number,
): Uint32Array {
  if (rows.length !== expectedRows) {
    throw new RangeError("ACE planner tokenizer returned the wrong row count");
  }
  const output = new Uint32Array(expectedRows * expectedTokens);
  rows.forEach((row, rowIndex) => {
    if (row.length !== expectedTokens) {
      throw new RangeError("ACE planner tokenizer returned ragged padded rows");
    }
    row.forEach((tokenId, tokenIndex) => {
      requirePlannerVocabularyToken(tokenId);
      output[rowIndex * expectedTokens + tokenIndex] = tokenId;
    });
  });
  return output;
}

function flattenValidityRows(
  rows: readonly (readonly number[])[],
  expectedRows: number,
  expectedTokens: number,
): Uint32Array {
  if (rows.length !== expectedRows) {
    throw new RangeError("ACE planner tokenizer returned the wrong mask row count");
  }
  const output = new Uint32Array(expectedRows * expectedTokens);
  rows.forEach((row, rowIndex) => {
    if (row.length !== expectedTokens) {
      throw new RangeError("ACE planner tokenizer returned a ragged attention mask");
    }
    row.forEach((value, tokenIndex) => {
      if (value !== 0 && value !== 1) {
        throw new RangeError("ACE planner tokenizer mask must contain only zero or one");
      }
      output[rowIndex * expectedTokens + tokenIndex] = value;
    });
  });
  return output;
}

function requirePlannerGraphLogits(
  rows: readonly ArrayLike<number>[],
  expectedRows: number,
  label: string,
  expectedColumns = ACE_PLANNER_QWEN3_CONFIG.vocabularySize,
): readonly ArrayLike<number>[] {
  if (rows.length !== expectedRows) {
    throw new RangeError(`${label} returned ${rows.length} rows; ${expectedRows} required`);
  }
  for (const [index, row] of rows.entries()) {
    if (row.length !== expectedColumns) {
      throw new RangeError(
        `${label} row ${index} has ${row.length} logits; ` +
          `${expectedColumns} required`,
      );
    }
  }
  return rows;
}

function requirePlannerTokenizer(tokenizer: AceQwenBpeTokenizer): void {
  if (!(tokenizer instanceof AceQwenBpeTokenizer) || tokenizer.kind !== "planner") {
    throw new TypeError("ACE planner runtime requires the authenticated planner tokenizer");
  }
}

function requirePlannerVocabularyToken(tokenId: number): void {
  if (
    !Number.isSafeInteger(tokenId) ||
    tokenId < 0 ||
    tokenId >= ACE_PLANNER_QWEN3_CONFIG.vocabularySize
  ) {
    throw new RangeError(`ACE planner token ID ${String(tokenId)} is outside the vocabulary`);
  }
}

function requireSemanticCode(value: number): void {
  if (
    !Number.isSafeInteger(value) ||
    value < 0 ||
    value >= ACE_PLANNER_SEMANTIC_CODE_COUNT
  ) {
    throw new RangeError(`ACE planner semantic code ${String(value)} is outside [0, 63999]`);
  }
}

function requireCacheCapacity(value: number): void {
  if (
    !Number.isSafeInteger(value) ||
    value <= 0 ||
    value > ACE_PLANNER_QWEN3_CONFIG.maximumPositionEmbeddings
  ) {
    throw new RangeError("ACE planner cache capacity exceeds the pinned model limit");
  }
}

function requireCotText(value: string): void {
  requireString(value, "ACE planner CoT text");
  if (!value.startsWith("<think>\n") || !value.endsWith("\n</think>")) {
    throw new TypeError("ACE planner CoT text must be one complete think block");
  }
}

function requireString(value: unknown, label: string): asserts value is string {
  if (typeof value !== "string") throw new TypeError(`${label} must be a string`);
}

function requirePositiveInteger(value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new RangeError(`${label} must be a positive safe integer`);
  }
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  signal?.throwIfAborted();
}

function equalNumbers(left: readonly number[], right: readonly number[]): boolean {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

type MutablePlannerMetadata = {
  -readonly [Key in keyof AcePlannerMetadata]?: AcePlannerMetadata[Key];
};
