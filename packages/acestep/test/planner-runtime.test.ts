import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { beforeAll, describe, expect, it } from "vitest";

import {
  ACE_PLANNER_BROWSER_POLICY,
  ACE_PLANNER_ORCHESTRATION_CONTRACT,
  ACE_PLANNER_SEMANTIC_FIRST_TOKEN_ID,
  createAcePlannerCodePrompts,
  createAcePlannerCotPrompt,
  createAcePlannerDecodeBatch,
  createAcePlannerPrefillBatch,
  createAcePlannerRequestPlan,
  finalizeAcePlannerConditioning,
  formatAcePlannerCotMetadata,
  parseAcePlannerCotMetadata,
  parseAcePlannerSemanticTokenSequence,
  runAcePlannerCotPhase,
  runAcePlannerSemanticPhase,
  serializeAcePlannerAudioCodes,
  type AcePlannerDecodeBatch,
  type AcePlannerGraphExecutor,
  type AcePlannerLogitRange,
  type AcePlannerPrefillBatch,
} from "../src/runtime/planner.js";
import {
  ACE_BROWSER_SOFTMAX_V1,
  AcePlannerSamplingCursor,
} from "../src/runtime/planner-sampling.js";
import { canonicalizeSeed } from "../src/runtime/seed.js";
import { loadPinnedAceTokenizer } from "../src/tokenizer/loader.js";
import {
  ACE_QWEN_IM_END_TOKEN_ID,
  type AceQwenBpeTokenizer,
} from "../src/tokenizer/qwen-bpe.js";

const MODEL_ASSET_ROOT = resolve("model/files-reference/assets/planner");

describe("ACE planner prompt and metadata orchestration", () => {
  it("matches the pinned CoT and training-aligned CFG prompt layout", () => {
    expect(createAcePlannerCotPrompt("calm piano", "hello")).toBe(
      "<|im_start|>system\n# Instruction\n" +
        "Generate audio semantic tokens based on the given conditions:\n\n" +
        "<|im_end|>\n<|im_start|>user\n# Caption\ncalm piano\n\n" +
        "# Lyric\nhello\n<|im_end|>\n<|im_start|>assistant\n",
    );
    const prompts = createAcePlannerCodePrompts(
      "calm piano",
      "hello",
      "<think>\nbpm: 120\n</think>",
    );
    expect(prompts.conditional).toContain(
      "<|im_start|>assistant\n<think>\nbpm: 120\n</think>\n\n",
    );
    expect(prompts.unconditional).toContain(
      "<|im_start|>user\nNO USER INPUT<|im_end|>\n" +
        "<|im_start|>assistant\n<think>\n\n</think>\n\n",
    );
    expect(prompts.unconditional).not.toContain("calm piano");
    expect(prompts.unconditional).not.toContain("hello");
  });

  it("parses upstream YAML-like fields and reproduces the pinned sorted CoT subset", () => {
    const parsed = parseAcePlannerCotMetadata(
      "<think>\n" +
        "bpm: 120\n" +
        "caption: Bright neo-soul with elastic bass, clipped guitar,\n" +
        "  crisp drums and warm keys.\n" +
        "duration: 12\n" +
        "genres: neo soul\n" +
        "keyscale: G major\n" +
        "language: en\n" +
        "timesignature: 4\n" +
        "</think>",
    );
    expect(parsed).toEqual({
      bpm: 120,
      caption: "Bright neo-soul with elastic bass, clipped guitar, crisp drums and warm keys.",
      duration: 12,
      genres: "neo soul",
      keyscale: "G major",
      language: "en",
      timesignature: "4",
    });
    expect(formatAcePlannerCotMetadata(parsed)).toBe(
      "<think>\n" +
        "bpm: 120\n" +
        "caption: Bright neo-soul with elastic bass, clipped guitar, crisp drums and warm keys.\n" +
        "duration: 12\n" +
        "keyscale: G major\n" +
        "language: en\n" +
        "timesignature: 4\n" +
        "</think>",
    );
  });

  it("matches PyYAML 6.0.3 scalar style, resolver quoting, and width-80 wrapping", () => {
    expect(formatAcePlannerCotMetadata({
      caption: "word ".repeat(40),
      keyscale: "0x10",
      language: "yes",
      timesignature: "12:34",
    })).toBe(
      "<think>\n" +
        "caption: 'word word word word word word word word word word word word word word word\n" +
        "  word word word word word word word word word word word word word word word word\n" +
        "  word word word word word word word word word '\n" +
        "keyscale: '0x10'\n" +
        "language: 'yes'\n" +
        "timesignature: '12:34'\n" +
        "</think>",
    );
    expect(formatAcePlannerCotMetadata({ caption: "it's fine 🎵" })).toBe(
      "<think>\ncaption: it's fine 🎵\n</think>",
    );
    expect(() => formatAcePlannerCotMetadata({ caption: "tab\ttext" })).toThrow(
      /unsupported PyYAML escaping/,
    );
  });

  it("runs CoT for missing core metadata and preserves explicit request precedence", () => {
    const plan = createAcePlannerRequestPlan(request({
      bpm: 88,
      vocalLanguage: "fr",
    }));
    expect(plan.shouldRunCot).toBe(true);
    const resolved = finalizeAcePlannerConditioning(
      plan,
      "<think>\n" +
        "bpm: 140\n" +
        "caption: Rewritten caption.\n" +
        "duration: 99\n" +
        "keyscale: F minor\n" +
        "language: ja\n" +
        "timesignature: 3\n" +
        "</think>",
    );
    expect(resolved.metadata.bpm).toBe(88);
    expect(resolved.metadata.duration).toBe(12);
    expect(resolved.metadata.keyscale).toBe("F minor");
    expect(resolved.caption).toBe("Rewritten caption.");
    expect(resolved.vocalLanguage).toBe("fr");
    expect(resolved.metadata.language).toBe("ja");
    expect(resolved.conditionalCodePrompt).toContain(
      "# Caption\nPrompt\n\n# Lyric\nLyrics",
    );
    expect(resolved.conditionalCodePrompt).toContain(
      "caption: Rewritten caption.",
    );
    expect(resolved.conditionalCodePrompt).not.toContain(
      "# Caption\nRewritten caption.",
    );
  });

  it("records and applies the reviewed generated-language browser policy", () => {
    expect(ACE_PLANNER_BROWSER_POLICY).toMatchObject({
      id: "ace-planner-browser-policy-v1",
      generatedLanguageRemap: "language-to-vocal-language",
      sourceProducer: expect.stringContaining(":2792-2877"),
      sourceConsumer: expect.stringContaining(":803-807"),
      explicitUserLanguagePrecedence: true,
      nativeCapturePatchRequired: true,
    });
    const resolved = finalizeAcePlannerConditioning(
      createAcePlannerRequestPlan(request()),
      "<think>\n" +
        "bpm: 120\ncaption: Rewritten.\nduration: 12\n" +
        "keyscale: C major\nlanguage: ja\ntimesignature: 4\n</think>",
    );
    expect(resolved.vocalLanguage).toBe("ja");
  });

  it("skips CoT only when all four core fields are explicit", () => {
    const plan = createAcePlannerRequestPlan(request({
      bpm: 120,
      keyScale: "C major",
      timeSignature: "4",
      vocalLanguage: "en",
    }));
    expect(plan.shouldRunCot).toBe(false);
    const resolved = finalizeAcePlannerConditioning(plan, null);
    expect(resolved.caption).toBe("Prompt");
    expect(resolved.metadata).toEqual({
      bpm: 120,
      duration: 12,
      keyscale: "C major",
      timesignature: "4",
    });
    expect(resolved.vocalLanguage).toBe("en");
    expect(resolved.cotText).toBe(
      "<think>\n" +
        "bpm: 120\n" +
        "duration: 12\n" +
        "keyscale: C major\n" +
        "timesignature: 4\n" +
        "</think>",
    );
    expect(resolved.conditionalCodePrompt).not.toContain("language:");
    expect(resolved.conditionalCodePrompt).not.toContain("caption:");
  });

  it("does not leak disabled generated caption or language into Phase 2", () => {
    const base = request({ bpm: 88 });
    const plan = createAcePlannerRequestPlan({
      ...base,
      planner: {
        ...base.planner,
        thinking: {
          ...base.planner.thinking,
          useCotCaption: false,
          useCotLanguage: false,
        },
      },
    });
    const resolved = finalizeAcePlannerConditioning(
      plan,
      "<think>\n" +
        "bpm: 140\ncaption: Must be ignored.\nduration: 12\n" +
        "genres: must also be ignored\n" +
        "keyscale: F minor\nlanguage: ja\ntimesignature: 3\n</think>",
    );
    expect(resolved.caption).toBe("Prompt");
    expect(resolved.vocalLanguage).toBe("unknown");
    expect(resolved.metadata).toEqual({
      bpm: 88,
      duration: 12,
      keyscale: "F minor",
      timesignature: "3",
    });
    expect(resolved.conditionalCodePrompt).not.toContain("Must be ignored");
    expect(resolved.conditionalCodePrompt).not.toContain("language:");
    expect(resolved.conditionalCodePrompt).not.toContain("genres:");
    expect(resolved.metadata).not.toHaveProperty("genres");
  });

  it("pins the accepted physical-position semantics", () => {
    expect(ACE_PLANNER_ORCHESTRATION_CONTRACT).toMatchObject({
      paddingSide: "left",
      rotaryPositions: "physical-cache-slots",
      codeRows: 2,
    });
  });
});

describe("ACE planner cache and semantic coordinator", () => {
  let tokenizer: AceQwenBpeTokenizer;

  beforeAll(async () => {
    tokenizer = (await loadPinnedAceTokenizer("planner", {
      tokenizerJson: readFileSync(resolve(MODEL_ASSET_ROOT, "tokenizer.json"), "utf8"),
      tokenizerConfigJson: readFileSync(
        resolve(MODEL_ASSET_ROOT, "tokenizer_config.json"),
        "utf8",
      ),
      chatTemplate: readFileSync(
        resolve(MODEL_ASSET_ROOT, "chat_template.jinja"),
        "utf8",
      ),
    })).tokenizer;
  });

  it("left-pads the short CFG row but assigns both rows shared physical RoPE IDs", () => {
    const batch = createAcePlannerPrefillBatch(
      tokenizer,
      ["hellohello", "hello"],
      32,
    );
    expect(batch.rows).toBe(2);
    expect(batch.tokens).toBe(2);
    expect([...batch.keyValidity]).toEqual([1, 1, 0, 1]);
    expect([...batch.rotaryPositionIds]).toEqual([0, 1, 0, 1]);
    expect([...batch.causal.queryPositions]).toEqual([0, 1, 0, 1]);
    expect([...batch.inputIds.slice(2)]).toEqual([151_643, 14_990]);
  });

  it("appends one shared CFG token at the next physical position", () => {
    const decode = createAcePlannerDecodeBatch(
      ACE_PLANNER_SEMANTIC_FIRST_TOKEN_ID + 42,
      2,
      11,
      20,
    );
    expect([...decode.inputIds]).toEqual([
      ACE_PLANNER_SEMANTIC_FIRST_TOKEN_ID + 42,
      ACE_PLANNER_SEMANTIC_FIRST_TOKEN_ID + 42,
    ]);
    expect([...decode.rotaryPositionIds]).toEqual([11, 11]);
    expect([...decode.causal.validLengths]).toEqual([1, 12, 1, 12]);
  });

  it("prefills and appends the constrained CoT row through metadata termination", async () => {
    const cotText =
      "<think>\nbpm: 120\nduration: 1\nkeyscale: C major\n" +
      "language: en\ntimesignature: 4\n</think>";
    const emitted = tokenizer.encode(cotText, { addSpecialTokens: false });
    const graph = new FakePlannerGraph(emitted);
    const cursor = new AcePlannerSamplingCursor(canonicalizeSeed(99));
    const result = await runAcePlannerCotPhase({
      graph,
      tokenizer,
      prompt: createAcePlannerCotPrompt("Prompt", "Lyrics"),
      cursor,
      constraint: {
        allowedTokens(input) {
          return { kind: "ids", tokenIds: [emitted[input.step]!] };
        },
        acceptToken(input) {
          return { finished: input.step === emitted.length - 1 };
        },
      },
      sampling: {
        temperature: 1,
        topK: 0,
        topP: 1,
        repetitionPenalty: 1,
      },
      maxNewTokens: emitted.length,
    });
    expect(result.stoppedBy).toBe("constraint");
    expect(result.outputText).toBe(cotText);
    expect(result.parsedMetadata).toMatchObject({
      bpm: 120,
      duration: 1,
      keyscale: "C major",
      language: "en",
      timesignature: "4",
    });
    expect(result.drawEnd - result.drawStart).toBe(BigInt(emitted.length));
    expect(graph.prefills).toHaveLength(1);
    expect(graph.prefills[0]!.cacheCapacity).toBe(
      graph.prefills[0]!.tokens + Math.max(1, emitted.length - 1),
    );
    expect(graph.decodes).toHaveLength(emitted.length - 1);
    expect(graph.decodes.at(-1)?.cachedTokensBeforeAppend).toBe(
      graph.prefills[0]!.tokens + emitted.length - 2,
    );
  });

  it("forces exactly N codes then EOS and consumes one Philox word for EOS", async () => {
    const prompts = createAcePlannerCodePrompts(
      "Prompt",
      "Lyrics",
      "<think>\nbpm: 120\nduration: 1\n</think>",
    );
    const graph = new FakePlannerGraph([
      ACE_PLANNER_SEMANTIC_FIRST_TOKEN_ID + 7,
      ACE_PLANNER_SEMANTIC_FIRST_TOKEN_ID + 8,
      ACE_PLANNER_SEMANTIC_FIRST_TOKEN_ID + 9,
      ACE_PLANNER_SEMANTIC_FIRST_TOKEN_ID + 10,
      ACE_PLANNER_SEMANTIC_FIRST_TOKEN_ID + 11,
      // Even though the model prefers another code here, the post-CFG duration
      // constraint must force EOS and still consume the draw.
      ACE_PLANNER_SEMANTIC_FIRST_TOKEN_ID + 12,
    ]);
    const cursor = new AcePlannerSamplingCursor(canonicalizeSeed(123));
    const result = await runAcePlannerSemanticPhase({
      graph,
      tokenizer,
      conditionalPrompt: prompts.conditional,
      unconditionalPrompt: prompts.unconditional,
      cursor,
      sampling: {
        temperature: 1,
        guidanceScale: 2,
        topK: 0,
        topP: 1,
      },
      durationSeconds: 1,
    });
    expect(result.semanticCodeValues).toEqual([7, 8, 9, 10, 11]);
    expect(result.emittedTokenIds.at(-1)).toBe(ACE_QWEN_IM_END_TOKEN_ID);
    expect(result.drawEnd - result.drawStart).toBe(6n);
    expect(cursor.consumed).toBe(6n);
    expect(graph.prefills).toHaveLength(1);
    expect(graph.prefills[0]!.cacheCapacity).toBe(
      graph.prefills[0]!.tokens + 5,
    );
    // The sampled code after each of the five logits rows is appended so the
    // following logits are computed; forced EOS terminates without append.
    expect(graph.decodes).toHaveLength(5);
  });

  it("keeps compact semantic ranges token- and cursor-exact", async () => {
    const prompts = createAcePlannerCodePrompts(
      "Prompt",
      "Lyrics",
      "<think>\nbpm: 120\nduration: 1\n</think>",
    );
    const preferred = Array.from(
      { length: 6 },
      (_, index) => ACE_PLANNER_SEMANTIC_FIRST_TOKEN_ID + 100 + index,
    );
    const seed = canonicalizeSeed(456);
    const fullCursor = new AcePlannerSamplingCursor(seed);
    const compactCursor = new AcePlannerSamplingCursor(seed);
    const common = {
      tokenizer,
      conditionalPrompt: prompts.conditional,
      unconditionalPrompt: prompts.unconditional,
      sampling: {
        temperature: 1,
        guidanceScale: 2,
        topK: 0,
        topP: 1,
      },
      durationSeconds: 1,
    } as const;
    const full = await runAcePlannerSemanticPhase({
      ...common,
      graph: new FakePlannerGraph(preferred),
      cursor: fullCursor,
    });
    const compactGraph = new FakePlannerGraph(preferred);
    const compact = await runAcePlannerSemanticPhase({
      ...common,
      graph: compactGraph,
      cursor: compactCursor,
      compactLogits: true,
    });
    expect(compact).toEqual(full);
    expect(compactCursor.consumed).toBe(fullCursor.consumed);
    expect(compactGraph.logitRanges.slice(0, -1).every((range) =>
      range?.firstTokenId === ACE_PLANNER_SEMANTIC_FIRST_TOKEN_ID &&
      range.tokenCount === 64_000
    )).toBe(true);
    expect(compactGraph.logitRanges.at(-1)).toEqual({
      firstTokenId: ACE_QWEN_IM_END_TOKEN_ID,
      tokenCount: 1,
    });
  });

  it("fails compact semantic selection closed outside the exact OPT-0082 tuple", async () => {
    const prompts = createAcePlannerCodePrompts(
      "Prompt",
      "Lyrics",
      "<think>\nbpm: 120\nduration: 1\n</think>",
    );
    const base = {
      tokenizer,
      conditionalPrompt: prompts.conditional,
      unconditionalPrompt: prompts.unconditional,
      cursor: new AcePlannerSamplingCursor(canonicalizeSeed(456)),
      durationSeconds: 1,
      compactLogits: true,
    } as const;
    const oneRowGraph = new FakePlannerGraph([
      ACE_PLANNER_SEMANTIC_FIRST_TOKEN_ID,
    ]);
    await expect(runAcePlannerSemanticPhase({
      ...base,
      graph: oneRowGraph,
      sampling: {
        temperature: 1,
        guidanceScale: 1,
        topK: 0,
        topP: 1,
      },
    })).rejects.toThrow(/require the two-row CFG path/);
    expect(oneRowGraph.prefills).toHaveLength(0);

    const customSoftmaxGraph = new FakePlannerGraph([
      ACE_PLANNER_SEMANTIC_FIRST_TOKEN_ID,
    ]);
    await expect(runAcePlannerSemanticPhase({
      ...base,
      graph: customSoftmaxGraph,
      sampling: {
        temperature: 1,
        guidanceScale: 2,
        topK: 0,
        topP: 1,
      },
      softmax: Object.freeze({
        ...ACE_BROWSER_SOFTMAX_V1,
        id: "test-accepted-softmax-copy",
      }),
    })).rejects.toThrow(/require browser softmax v1/);
    expect(customSoftmaxGraph.prefills).toHaveLength(0);
  });

  it("uses the pinned one-row non-CFG path when guidance is exactly one", async () => {
    const prompts = createAcePlannerCodePrompts(
      "Prompt",
      "Lyrics",
      "<think>\nbpm: 120\nduration: 1\n</think>",
    );
    const preferred = Array.from(
      { length: 6 },
      (_, index) => ACE_PLANNER_SEMANTIC_FIRST_TOKEN_ID + index,
    );
    const graph = new FakePlannerGraph(preferred);
    await runAcePlannerSemanticPhase({
      graph,
      tokenizer,
      conditionalPrompt: prompts.conditional,
      unconditionalPrompt: prompts.unconditional,
      cursor: new AcePlannerSamplingCursor(canonicalizeSeed(321)),
      sampling: {
        temperature: 1,
        guidanceScale: 1,
        topK: 0,
        topP: 1,
      },
      durationSeconds: 1,
      cacheCapacity: 512,
    });
    expect(graph.prefills[0]).toMatchObject({
      rows: 1,
      unconditionalRow: null,
    });
    expect(graph.decodes.every((batch) =>
      batch.rows === 1 && batch.unconditionalRow === null
    )).toBe(true);
  });

  it("parses and serializes only canonical 0..63999 semantic codes", () => {
    expect(parseAcePlannerSemanticTokenSequence([
      ACE_PLANNER_SEMANTIC_FIRST_TOKEN_ID,
      ACE_PLANNER_SEMANTIC_FIRST_TOKEN_ID + 63_999,
      ACE_QWEN_IM_END_TOKEN_ID,
    ], 2)).toEqual([0, 63_999]);
    expect(serializeAcePlannerAudioCodes([0, 42, 63_999])).toBe(
      "<|audio_code_0|><|audio_code_42|><|audio_code_63999|>",
    );
    expect(() => parseAcePlannerSemanticTokenSequence([
      ACE_PLANNER_SEMANTIC_FIRST_TOKEN_ID + 64_000,
      ACE_QWEN_IM_END_TOKEN_ID,
    ], 1)).toThrow(/outside \[0, 63999\]/);
    expect(() => parseAcePlannerSemanticTokenSequence([
      ACE_PLANNER_SEMANTIC_FIRST_TOKEN_ID,
      ACE_PLANNER_SEMANTIC_FIRST_TOKEN_ID + 1,
    ], 1)).toThrow(/does not end in EOS/);
  });
});

class FakePlannerGraph implements AcePlannerGraphExecutor {
  readonly prefills: AcePlannerPrefillBatch[] = [];
  readonly decodes: AcePlannerDecodeBatch[] = [];
  readonly logitRanges: Array<AcePlannerLogitRange | undefined> = [];
  private step = 0;

  constructor(private readonly preferredTokens: readonly number[]) {}

  async prefill(
    batch: AcePlannerPrefillBatch,
    logitRange?: AcePlannerLogitRange,
  ): Promise<readonly ArrayLike<number>[]> {
    this.prefills.push(batch);
    this.logitRanges.push(logitRange);
    return this.logits(batch.rows, logitRange);
  }

  async decode(
    batch: AcePlannerDecodeBatch,
    logitRange?: AcePlannerLogitRange,
  ): Promise<readonly ArrayLike<number>[]> {
    this.decodes.push(batch);
    this.logitRanges.push(logitRange);
    return this.logits(batch.rows, logitRange);
  }

  private logits(
    rows: number,
    logitRange?: AcePlannerLogitRange,
  ): readonly Float32Array[] {
    const preferred = this.preferredTokens[this.step];
    if (preferred === undefined) throw new Error("Fake planner graph exhausted logits");
    this.step += 1;
    return Array.from({ length: rows }, (_, row) => {
      const logits = new Float32Array(217_204);
      logits.fill(-20);
      logits[preferred] = row === 0 ? 20 : 0;
      logits[ACE_QWEN_IM_END_TOKEN_ID] = -10;
      return logitRange === undefined
        ? logits
        : logits.slice(
            logitRange.firstTokenId,
            logitRange.firstTokenId + logitRange.tokenCount,
          );
    });
  }
}

function request(metadata?: {
  readonly bpm?: number;
  readonly keyScale?: string;
  readonly timeSignature?: string;
  readonly vocalLanguage?: string;
}) {
  return {
    generationProfile: "ace-turbo-v1-correctness" as const,
    prompt: "Prompt",
    lyrics: "Lyrics",
    instrumental: false,
    durationSeconds: 12,
    seed: canonicalizeSeed(1),
    planner: {
      mode: "enabled" as const,
      temperature: 0.85,
      guidanceScale: 2,
      topK: 0,
      topP: 0.9,
      constrainedDecoding: true as const,
      generateSemanticCodes: true as const,
      negativePrompt: "NO USER INPUT" as const,
      thinking: {
        enabled: true as const,
        useCotCaption: true,
        useCotLanguage: true,
        useCotMissingMetadata: true,
      },
    },
    ...(metadata === undefined ? {} : { metadata }),
  };
}
