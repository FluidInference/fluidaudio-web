import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { beforeAll, describe, expect, it } from "vitest";

import {
  ACE_PLANNER_METADATA_FSM_CONTRACT,
  AcePlannerMetadataConstraintController,
  createAcePlannerMetadataConstraintForPlan,
  type AcePlannerMetadataConstraintOptions,
} from "../src/runtime/planner-metadata-fsm.js";
import type {
  AcePlannerCotAcceptedToken,
  AcePlannerCotConstraintInput,
} from "../src/runtime/planner.js";
import { createAcePlannerRequestPlan } from "../src/runtime/planner.js";
import type { AcePlannerAllowedTokens } from "../src/runtime/planner-sampling.js";
import { canonicalizeSeed } from "../src/runtime/seed.js";
import { loadPinnedAceTokenizer } from "../src/tokenizer/loader.js";
import {
  ACE_PLANNER_AUDIO_CODE_FIRST_TOKEN_ID,
  ACE_QWEN_IM_END_TOKEN_ID,
  type AceQwenBpeTokenizer,
} from "../src/tokenizer/qwen-bpe.js";
import { ACE_PLANNER_QWEN3_CONFIG } from "../src/webgpu/qwen3.js";

const MODEL_ASSET_ROOT = resolve("model/files-reference/assets/planner");
const VECTORS = JSON.parse(
  readFileSync(resolve("test/planner-metadata-fsm-vectors.json"), "utf8"),
) as MetadataFsmVectors;

describe("pinned ACE planner metadata constraint FSM", () => {
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

  it("matches the exact upstream user-injection trace and closing-tag EOS behavior", () => {
    expect(ACE_PLANNER_METADATA_FSM_CONTRACT).toMatchObject({
      aceSourceRevision: VECTORS.aceSourceRevision,
      skipGenres: true,
      durationRange: [10, 240],
      terminalBehavior: "force-im-end-before-think-end-tag",
      failClosed: true,
    });
    const harness = new FsmHarness(tokenizer, {
      userMetadata: VECTORS.injectedTrace.metadata,
    });
    harness.acceptVector(VECTORS.injectedTrace.emittedTokenIds);
    expect(harness.emitted).toEqual(VECTORS.injectedTrace.emittedTokenIds);
    expect(tokenizer.decode(harness.emitted)).toBe(VECTORS.injectedTrace.decoded);
    expect(harness.lastTokenId).toBe(ACE_QWEN_IM_END_TOKEN_ID);
    expect(harness.finished).toBe(true);
  });

  it("matches upstream numeric prefix tries and the 240-second product bound", () => {
    assertVector(tokenizer, VECTORS.prefixes.bpmRoot);
    assertVector(tokenizer, VECTORS.prefixes.bpmAfterSpace);
    assertVector(tokenizer, VECTORS.prefixes.durationRoot, {
      userMetadata: { bpm: 120 },
      skipCaption: true,
      skipLanguage: true,
    });
    assertVector(tokenizer, VECTORS.prefixes.durationAfterSpace, {
      userMetadata: { bpm: 120 },
      skipCaption: true,
      skipLanguage: true,
    });
    assertVector(tokenizer, VECTORS.prefixes.duration240Complete, {
      userMetadata: { bpm: 120 },
      skipCaption: true,
      skipLanguage: true,
    });
  });

  it("matches upstream keyscale and time-signature token-prefix tries", () => {
    const throughDuration = {
      userMetadata: { bpm: 120, duration: 12 },
      skipCaption: true,
      skipLanguage: true,
    } as const;
    assertVector(tokenizer, VECTORS.prefixes.keyscaleRoot, throughDuration);
    assertVector(tokenizer, VECTORS.prefixes.keyscaleCMajorComplete, throughDuration);
    assertVector(tokenizer, VECTORS.prefixes.timesignatureAfterSpace, {
      userMetadata: { bpm: 120, duration: 12, keyscale: "C major" },
      skipCaption: true,
      skipLanguage: true,
    });
  });

  it("maps the validated request plan into the active FSM profile", () => {
    const plan = createAcePlannerRequestPlan({
      generationProfile: "ace-turbo-v1-correctness",
      prompt: "Prompt",
      lyrics: "Lyrics",
      instrumental: false,
      durationSeconds: 12,
      seed: canonicalizeSeed(1),
      metadata: { bpm: 120 },
      planner: {
        mode: "enabled",
        temperature: 0.85,
        guidanceScale: 2,
        topK: 0,
        topP: 0.9,
        constrainedDecoding: true,
        generateSemanticCodes: true,
        negativePrompt: "NO USER INPUT",
        thinking: {
          enabled: true,
          useCotCaption: false,
          useCotLanguage: false,
          useCotMissingMetadata: true,
        },
      },
    });
    const controller = createAcePlannerMetadataConstraintForPlan(tokenizer, plan);
    const harness = new FsmHarness(tokenizer, {}, controller);
    harness.acceptVector(VECTORS.prefixes.keyscaleRoot.emittedTokenIds);
    expect(ids(harness.allowed())).toEqual(VECTORS.prefixes.keyscaleRoot.allowedTokenIds);
  });

  it("locks language greedily from raw logits with the captured upstream tie order", () => {
    const equal = new FsmHarness(tokenizer, {
      userMetadata: { bpm: 120, duration: 12, keyscale: "C major" },
      skipCaption: true,
    });
    equal.acceptVector(VECTORS.languageEqualLogits.emittedTokenIds);
    expect(ids(equal.allowed())).toEqual(VECTORS.languageEqualLogits.allowedTokenIds);
    expect(tokenizer.decode(VECTORS.languageEqualLogits.allowedTokenIds)).toBe(
      VECTORS.languageEqualLogits.decoded,
    );

    const biased = new FsmHarness(tokenizer, {
      userMetadata: { bpm: 120, duration: 12, keyscale: "C major" },
      skipCaption: true,
    });
    biased.acceptVector(VECTORS.languageEqualLogits.emittedTokenIds);
    const en = tokenizer.encode(" en", { addSpecialTokens: false })[0]!;
    const fr = tokenizer.encode(" fr", { addSpecialTokens: false })[0]!;
    const logits = biased.blankLogits();
    logits[en] = 8;
    logits[fr] = 9;
    expect(ids(biased.allowed(logits))).toEqual([fr]);
    biased.accept(fr);
    expect(ids(biased.allowed())).toEqual([198]);
  });

  it("matches the caption mask and preserves its invalid audio-code tail", () => {
    const harness = new FsmHarness(tokenizer, { userMetadata: { bpm: 120 } });
    harness.acceptVector(VECTORS.captionMask.emittedTokenIds);
    const allowed = harness.allowed();
    const tokenIds = ids(allowed);
    expect(tokenIds).toHaveLength(VECTORS.captionMask.allowedTokenCount);
    expect(u32LeSha256(tokenIds)).toBe(VECTORS.captionMask.sortedU32LeSha256);
    expect(Object.isFrozen(allowed)).toBe(true);
    expect(Object.isFrozen(tokenIds)).toBe(true);
    expect(tokenIds).not.toContain(63);
    expect(tokenIds).not.toContain(ACE_PLANNER_AUDIO_CODE_FIRST_TOKEN_ID);
    expect(tokenIds).not.toContain(ACE_PLANNER_AUDIO_CODE_FIRST_TOKEN_ID + 63_999);
    // The tokenizer exposes 65,535 audio rows, but upstream masks canonical
    // values 0..63,999 only. The 1,535-row invalid tail remains admitted.
    expect(tokenIds).toContain(ACE_PLANNER_AUDIO_CODE_FIRST_TOKEN_ID + 64_000);
    expect(tokenIds).toContain(ACE_PLANNER_QWEN3_CONFIG.vocabularySize - 1);
    expect(() => (tokenIds as unknown as number[]).push(1)).toThrow();
  });

  it("tracks an unindented field after a multiline caption", () => {
    const harness = new FsmHarness(tokenizer, { userMetadata: { bpm: 120 } });
    harness.acceptVector(VECTORS.captionMask.emittedTokenIds);
    harness.acceptEncoded(" Free caption.\n");
    const duration = tokenizer.encode("duration", { addSpecialTokens: false })[0]!;
    const logits = harness.blankLogits();
    logits[duration] = 10;
    expect(harness.allowed(logits).kind).toBe("all");
    harness.accept(duration);
    expect(harness.allowed().kind).toBe("all");
    harness.accept(tokenizer.encode(":", { addSpecialTokens: false })[0]!);
    expect(ids(harness.allowed())).toEqual([220]);
  });

  it("fails closed on unknown caption fields and controller protocol violations", () => {
    const badField = new FsmHarness(tokenizer, { userMetadata: { bpm: 120 } });
    badField.acceptVector(VECTORS.captionMask.emittedTokenIds);
    badField.acceptEncoded(" Free caption.\n");
    const unknown = tokenizer.encode("nonsense", { addSpecialTokens: false })[0]!;
    const logits = badField.blankLogits();
    logits[unknown] = 10;
    expect(badField.allowed(logits).kind).toBe("all");
    badField.accept(unknown);
    expect(badField.allowed().kind).toBe("all");
    expect(() => badField.accept(tokenizer.encode(":", { addSpecialTokens: false })[0]!)).toThrow(
      /unknown field/,
    );

    const protocol = new FsmHarness(tokenizer);
    expect(() => protocol.allowed(new Float32Array(3))).toThrow(/217204 required/);
    const first = protocol.allowed();
    expect(() => protocol.controller.allowedTokens(protocol.input(protocol.blankLogits()))).toThrow(
      /called twice/,
    );
    expect(() => protocol.accept(0)).toThrow(/rejected token/);
    protocol.accept(ids(first)[0]!);
    expect(() => protocol.controller.allowedTokens({
      ...protocol.input(protocol.blankLogits()),
      emittedTokenIds: [0],
    })).toThrow(/history differs/);
    const second = protocol.allowed();
    expect(() => protocol.controller.acceptToken({
      step: protocol.emitted.length,
      tokenId: ids(second)[0]!,
      tokenText: "wrong",
      emittedTokenIds: [...protocol.emitted, ids(second)[0]!],
    })).toThrow(/token text/);
  });
});

class FsmHarness {
  readonly controller: AcePlannerMetadataConstraintController;
  readonly prompt = Object.freeze([1, 2, 3]);
  readonly emitted: number[] = [];
  finished = false;

  constructor(
    readonly tokenizer: AceQwenBpeTokenizer,
    options: Omit<AcePlannerMetadataConstraintOptions, "tokenizer"> = {},
    controller?: AcePlannerMetadataConstraintController,
  ) {
    this.controller = controller ??
      new AcePlannerMetadataConstraintController({ tokenizer, ...options });
  }

  get lastTokenId(): number | undefined {
    return this.emitted.at(-1);
  }

  blankLogits(): Float32Array {
    return new Float32Array(ACE_PLANNER_QWEN3_CONFIG.vocabularySize);
  }

  input(logits: ArrayLike<number>): AcePlannerCotConstraintInput {
    return {
      step: this.emitted.length,
      promptTokenIds: this.prompt,
      emittedTokenIds: this.emitted,
      logits,
    };
  }

  allowed(logits = this.blankLogits()): AcePlannerAllowedTokens {
    return this.controller.allowedTokens(this.input(logits));
  }

  accept(tokenId: number): void {
    const emitted = [...this.emitted, tokenId];
    const input: AcePlannerCotAcceptedToken = {
      step: this.emitted.length,
      tokenId,
      tokenText: this.tokenizer.decode([tokenId]),
      emittedTokenIds: emitted,
    };
    this.finished = this.controller.acceptToken(input).finished;
    this.emitted.push(tokenId);
  }

  acceptVector(tokenIds: readonly number[]): void {
    for (const tokenId of tokenIds) {
      expect(admits(this.allowed(), tokenId)).toBe(true);
      this.accept(tokenId);
    }
  }

  acceptEncoded(text: string): void {
    this.acceptVector(this.tokenizer.encode(text, { addSpecialTokens: false }));
  }
}

function assertVector(
  tokenizer: AceQwenBpeTokenizer,
  vector: PrefixVector,
  options: Omit<AcePlannerMetadataConstraintOptions, "tokenizer"> = {},
): void {
  const harness = new FsmHarness(tokenizer, options);
  harness.acceptVector(vector.emittedTokenIds);
  expect(ids(harness.allowed())).toEqual(vector.allowedTokenIds);
}

function ids(allowed: AcePlannerAllowedTokens): readonly number[] {
  if (allowed.kind !== "ids") throw new Error(`Expected ID constraint, got ${allowed.kind}`);
  return allowed.tokenIds;
}

function admits(allowed: AcePlannerAllowedTokens, tokenId: number): boolean {
  if (allowed.kind === "all") return true;
  if (allowed.kind === "ids") return allowed.tokenIds.includes(tokenId);
  if (tokenId >= allowed.firstTokenId && tokenId < allowed.firstTokenId + allowed.tokenCount) {
    return true;
  }
  return allowed.additionalTokenIds?.includes(tokenId) ?? false;
}

function u32LeSha256(tokenIds: readonly number[]): string {
  const buffer = Buffer.allocUnsafe(tokenIds.length * 4);
  [...tokenIds].sort((left, right) => left - right).forEach((tokenId, index) => {
    buffer.writeUInt32LE(tokenId, index * 4);
  });
  return createHash("sha256").update(buffer).digest("hex");
}

interface PrefixVector {
  readonly emittedTokenIds: readonly number[];
  readonly allowedTokenIds: readonly number[];
}

interface MetadataFsmVectors {
  readonly aceSourceRevision: string;
  readonly injectedTrace: Readonly<{
    readonly metadata: Readonly<{
      readonly bpm: string;
      readonly caption: string;
      readonly duration: string;
      readonly keyscale: string;
      readonly language: string;
      readonly timesignature: string;
    }>;
    readonly emittedTokenIds: readonly number[];
    readonly decoded: string;
  }>;
  readonly prefixes: Readonly<{
    readonly bpmRoot: PrefixVector;
    readonly bpmAfterSpace: PrefixVector;
    readonly durationRoot: PrefixVector;
    readonly durationAfterSpace: PrefixVector;
    readonly duration240Complete: PrefixVector;
    readonly keyscaleRoot: PrefixVector;
    readonly keyscaleCMajorComplete: PrefixVector;
    readonly timesignatureAfterSpace: PrefixVector;
  }>;
  readonly languageEqualLogits: PrefixVector & Readonly<{ readonly decoded: string }>;
  readonly captionMask: Readonly<{
    readonly emittedTokenIds: readonly number[];
    readonly allowedTokenCount: number;
    readonly sortedU32LeSha256: string;
  }>;
}
