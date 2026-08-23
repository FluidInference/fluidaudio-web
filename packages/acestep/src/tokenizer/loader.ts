import { aceSha256Hex } from "../model/sha256.js";
import {
  ACE_PLANNER_AUDIO_CODE_COUNT,
  ACE_PLANNER_AUDIO_CODE_FIRST_TOKEN_ID,
  ACE_QWEN_BASE_ADDED_TOKENS,
  ACE_QWEN_BASE_VOCABULARY_SIZE,
  AceQwenBpeTokenizer,
  aceQwenMergeKey,
  type AceQwenBpeDefinition,
  type AceTokenizerKind,
} from "./qwen-bpe.js";

export type AceTokenizerAssetSource = string | Blob;

export interface AceTokenizerAssetBundle {
  readonly tokenizerJson: AceTokenizerAssetSource;
  readonly tokenizerConfigJson: AceTokenizerAssetSource;
  readonly chatTemplate: AceTokenizerAssetSource;
}

export interface LoadedAceTokenizer {
  readonly tokenizer: AceQwenBpeTokenizer;
  readonly chatTemplate: string;
  readonly assetIdentity: Readonly<{
    tokenizerSha256: string;
    tokenizerConfigSha256: string;
    chatTemplateSha256: string;
  }>;
}

interface AssetIdentity {
  readonly byteLength: number;
  readonly sha256: string;
}

interface PinnedTokenizerAssets {
  readonly tokenizer: AssetIdentity;
  readonly tokenizerConfig: AssetIdentity;
  readonly chatTemplate: AssetIdentity;
}

const PINNED_ASSETS: Readonly<Record<AceTokenizerKind, PinnedTokenizerAssets>> =
  Object.freeze({
    text: Object.freeze({
      tokenizer: Object.freeze({
        byteLength: 11_423_705,
        sha256: "def76fb086971c7867b829c23a26261e38d9d74e02139253b38aeb9df8b4b50a",
      }),
      tokenizerConfig: Object.freeze({
        byteLength: 5_404,
        sha256: "443bfa629eb16387a12edbf92a76f6a6f10b2af3b53d87ba1550adfcf45f7fa0",
      }),
      chatTemplate: Object.freeze({
        byteLength: 4_116,
        sha256: "87a2728cb8dc9fe424d624542f6060ec05a1d285ebbec578bb078900e33396b5",
      }),
    }),
    planner: Object.freeze({
      tokenizer: Object.freeze({
        byteLength: 24_321_939,
        sha256: "35af56c3f5cb3ea2cc578aa28a8937770981d504f183ac5c8c38baf4bbd4af4d",
      }),
      tokenizerConfig: Object.freeze({
        byteLength: 14_072_925,
        sha256: "6cd70cdd89425971794f5235562edcc608b0629a6c4686ae51a8b8c8b8ba5e95",
      }),
      chatTemplate: Object.freeze({
        byteLength: 4_168,
        sha256: "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8",
      }),
    }),
  });

const EXPECTED_PRETOKENIZER_REGEX =
  "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

// The two authenticated tokenizer.json files contain the identical base BPE.
// Keep one runtime copy when planner and text tokenizers are both needed.
let sharedBaseDefinition: AceQwenBpeDefinition | undefined;

/**
 * Load only the exact tokenizer assets from the two pinned model snapshots.
 *
 * Package authentication normally verifies these files first. The local hash
 * check deliberately repeats that boundary because even a syntactically valid
 * tokenizer revision changes model inputs and is therefore model math.
 */
export async function loadPinnedAceTokenizer(
  kind: AceTokenizerKind,
  assets: AceTokenizerAssetBundle,
): Promise<LoadedAceTokenizer> {
  const pinned = PINNED_ASSETS[kind];

  // Validate and release the redundant configuration before materializing the
  // much larger tokenizer graph. The exact digest makes JSON.parse safe here:
  // duplicate-key or other mutations cannot share the pinned identity.
  const configText = await readPinnedText(
    assets.tokenizerConfigJson,
    `${kind} tokenizer_config.json`,
    pinned.tokenizerConfig,
  );
  validateTokenizerConfig(kind, parseJson(configText, "tokenizer_config.json"));

  const chatTemplate = await readPinnedText(
    assets.chatTemplate,
    `${kind} chat_template.jinja`,
    pinned.chatTemplate,
  );

  const tokenizerText = await readPinnedText(
    assets.tokenizerJson,
    `${kind} tokenizer.json`,
    pinned.tokenizer,
  );
  const definition = parseTokenizerDefinition(
    kind,
    parseJson(tokenizerText, "tokenizer.json"),
  );
  sharedBaseDefinition ??= definition;

  return Object.freeze({
    tokenizer: new AceQwenBpeTokenizer(kind, definition),
    chatTemplate,
    assetIdentity: Object.freeze({
      tokenizerSha256: pinned.tokenizer.sha256,
      tokenizerConfigSha256: pinned.tokenizerConfig.sha256,
      chatTemplateSha256: pinned.chatTemplate.sha256,
    }),
  });
}

async function readPinnedText(
  source: AceTokenizerAssetSource,
  label: string,
  expected: AssetIdentity,
): Promise<string> {
  let bytes: Uint8Array;
  if (typeof source === "string") {
    bytes = new TextEncoder().encode(source);
  } else if (source instanceof Blob) {
    if (source.size !== expected.byteLength) {
      throw new Error(
        `${label} byte length mismatch: expected ${expected.byteLength}, got ${source.size}`,
      );
    }
    bytes = new Uint8Array(await source.arrayBuffer());
  } else {
    throw new TypeError(`${label} must be supplied as a string or Blob`);
  }
  if (bytes.byteLength !== expected.byteLength) {
    throw new Error(
      `${label} byte length mismatch: expected ${expected.byteLength}, got ${bytes.byteLength}`,
    );
  }
  const digest = aceSha256Hex(bytes);
  if (digest !== expected.sha256) {
    throw new Error(
      `${label} SHA-256 mismatch: expected ${expected.sha256}, got ${digest}`,
    );
  }
  try {
    return new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  } catch (error) {
    throw new Error(`${label} is not valid UTF-8`, { cause: error });
  }
}

function parseJson(text: string, label: string): unknown {
  try {
    return JSON.parse(text) as unknown;
  } catch (error) {
    throw new SyntaxError(`Invalid ${label}`, { cause: error });
  }
}

function parseTokenizerDefinition(
  kind: AceTokenizerKind,
  value: unknown,
): AceQwenBpeDefinition {
  const root = object(value, "tokenizer root");
  equal(root.version, "1.0", "tokenizer version");
  equal(root.truncation, null, "tokenizer truncation");
  equal(root.padding, null, "tokenizer padding");

  const normalizer = object(root.normalizer, "tokenizer normalizer");
  equal(normalizer.type, "NFC", "tokenizer normalizer type");
  validatePretokenizer(root.pre_tokenizer);
  validatePostProcessor(kind, root.post_processor);
  validateByteLevel(root.decoder, "tokenizer decoder");

  const model = object(root.model, "tokenizer model");
  equal(model.type, "BPE", "tokenizer model type");
  equal(model.dropout, null, "tokenizer BPE dropout");
  equal(model.unk_token, null, "tokenizer unknown token");
  equal(model.continuing_subword_prefix, "", "continuing subword prefix");
  equal(model.end_of_word_suffix, "", "end-of-word suffix");
  equal(model.fuse_unk, false, "fuse unknown tokens");
  equal(model.byte_fallback, false, "byte fallback");
  equal(model.ignore_merges, false, "ignore merges");

  validateAddedTokens(kind, root.added_tokens);
  if (sharedBaseDefinition !== undefined) return sharedBaseDefinition;

  const vocabularyObject = object(model.vocab, "tokenizer vocabulary");
  const vocabularyEntries = Object.entries(vocabularyObject);
  equal(
    vocabularyEntries.length,
    ACE_QWEN_BASE_VOCABULARY_SIZE,
    "base vocabulary size",
  );
  const vocabulary = new Map<string, number>();
  const tokensById: string[] = Array(ACE_QWEN_BASE_VOCABULARY_SIZE);
  for (const [token, rawId] of vocabularyEntries) {
    const tokenId = integer(rawId, `vocabulary ID for ${JSON.stringify(token)}`);
    if (tokenId < 0 || tokenId >= ACE_QWEN_BASE_VOCABULARY_SIZE) {
      throw new RangeError(`Base vocabulary ID ${tokenId} is out of range`);
    }
    if (tokensById[tokenId] !== undefined) {
      throw new Error(`Duplicate base vocabulary ID ${tokenId}`);
    }
    vocabulary.set(token, tokenId);
    tokensById[tokenId] = token;
  }
  for (let tokenId = 0; tokenId < tokensById.length; tokenId += 1) {
    if (tokensById[tokenId] === undefined) {
      throw new Error(`Base vocabulary is missing token ID ${tokenId}`);
    }
  }

  const rawMerges = array(model.merges, "tokenizer merges");
  equal(rawMerges.length, 151_387, "tokenizer merge count");
  const mergeRanks = new Map<string, number>();
  rawMerges.forEach((rawMerge, rank) => {
    const pair = array(rawMerge, `merge ${rank}`);
    equal(pair.length, 2, `merge ${rank} arity`);
    const left = string(pair[0], `merge ${rank} left token`);
    const right = string(pair[1], `merge ${rank} right token`);
    const key = aceQwenMergeKey(left, right);
    if (mergeRanks.has(key)) throw new Error(`Duplicate tokenizer merge at rank ${rank}`);
    if (!vocabulary.has(left + right)) {
      throw new Error(`Tokenizer merge ${rank} produces a token absent from vocabulary`);
    }
    mergeRanks.set(key, rank);
  });
  return { vocabulary, tokensById, mergeRanks };
}

function validateAddedTokens(kind: AceTokenizerKind, value: unknown): void {
  const tokens = array(value, "added tokens");
  const expectedCount =
    ACE_QWEN_BASE_ADDED_TOKENS.length +
    (kind === "planner" ? ACE_PLANNER_AUDIO_CODE_COUNT : 0);
  equal(tokens.length, expectedCount, "added-token count");
  tokens.forEach((rawToken, index) => {
    const token = object(rawToken, `added token ${index}`);
    const expectedId = ACE_QWEN_BASE_VOCABULARY_SIZE + index;
    equal(token.id, expectedId, `added token ${index} ID`);
    const isAudio = index >= ACE_QWEN_BASE_ADDED_TOKENS.length;
    const expectedContent = isAudio
      ? `<|audio_code_${expectedId - ACE_PLANNER_AUDIO_CODE_FIRST_TOKEN_ID}|>`
      : ACE_QWEN_BASE_ADDED_TOKENS[index]!;
    equal(token.content, expectedContent, `added token ${index} content`);
    equal(token.single_word, false, `added token ${index} single_word`);
    equal(token.lstrip, false, `added token ${index} lstrip`);
    equal(token.rstrip, false, `added token ${index} rstrip`);
    equal(token.normalized, false, `added token ${index} normalized`);
    equal(token.special, isAudio || expectedId <= 151_656, `added token ${index} special`);
  });
}

function validateTokenizerConfig(kind: AceTokenizerKind, value: unknown): void {
  const config = object(value, "tokenizer configuration");
  equal(config.tokenizer_class, "Qwen2Tokenizer", "tokenizer class");
  equal(config.model_max_length, 131_072, "tokenizer model maximum length");
  equal(config.add_bos_token, false, "add BOS token");
  equal(config.add_prefix_space, false, "add prefix space");
  equal(config.bos_token, null, "BOS token");
  equal(config.eos_token, "<|im_end|>", "EOS token");
  equal(config.pad_token, "<|endoftext|>", "padding token");
  equal(config.unk_token, null, "unknown token");
  equal(config.clean_up_tokenization_spaces, false, "cleanup tokenization spaces");
  equal(config.split_special_tokens, false, "split special tokens");
  const decoder = object(config.added_tokens_decoder, "added-token decoder");
  const expectedDecoderCount =
    ACE_QWEN_BASE_ADDED_TOKENS.length +
    (kind === "planner" ? ACE_PLANNER_AUDIO_CODE_COUNT : 0);
  equal(Object.keys(decoder).length, expectedDecoderCount, "added-token decoder count");
  const additional = array(config.additional_special_tokens, "additional special tokens");
  equal(
    additional.length,
    kind === "planner" ? ACE_PLANNER_AUDIO_CODE_COUNT : 13,
    "additional special-token count",
  );
}

function validatePretokenizer(value: unknown): void {
  const sequence = object(value, "tokenizer pre-tokenizer");
  equal(sequence.type, "Sequence", "pre-tokenizer type");
  const parts = array(sequence.pretokenizers, "pre-tokenizer sequence");
  equal(parts.length, 2, "pre-tokenizer sequence length");
  const split = object(parts[0], "split pre-tokenizer");
  equal(split.type, "Split", "split pre-tokenizer type");
  const pattern = object(split.pattern, "split pre-tokenizer pattern");
  equal(pattern.Regex, EXPECTED_PRETOKENIZER_REGEX, "split pre-tokenizer regex");
  equal(split.behavior, "Isolated", "split pre-tokenizer behavior");
  equal(split.invert, false, "split pre-tokenizer inversion");
  validateByteLevel(parts[1], "byte-level pre-tokenizer");
}

function validatePostProcessor(kind: AceTokenizerKind, value: unknown): void {
  if (kind === "planner") {
    validateByteLevel(value, "planner post-processor");
    return;
  }
  const sequence = object(value, "text post-processor");
  equal(sequence.type, "Sequence", "text post-processor type");
  const processors = array(sequence.processors, "text post-processors");
  equal(processors.length, 2, "text post-processor count");
  validateByteLevel(processors[0], "text byte-level post-processor");
  const template = object(processors[1], "text template post-processor");
  equal(template.type, "TemplateProcessing", "text template post-processor type");
  const specialTokens = object(template.special_tokens, "template special tokens");
  const endOfText = object(specialTokens["<|endoftext|>"], "end-of-text template token");
  const ids = array(endOfText.ids, "end-of-text template IDs");
  equal(ids.length, 1, "end-of-text template ID count");
  equal(ids[0], 151_643, "end-of-text template ID");
}

function validateByteLevel(value: unknown, label: string): void {
  const byteLevel = object(value, label);
  equal(byteLevel.type, "ByteLevel", `${label} type`);
  equal(byteLevel.add_prefix_space, false, `${label} add_prefix_space`);
  equal(byteLevel.trim_offsets, false, `${label} trim_offsets`);
  equal(byteLevel.use_regex, false, `${label} use_regex`);
}

function object(value: unknown, label: string): Record<string, unknown> {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new TypeError(`${label} must be an object`);
  }
  return value as Record<string, unknown>;
}

function array(value: unknown, label: string): unknown[] {
  if (!Array.isArray(value)) throw new TypeError(`${label} must be an array`);
  return value;
}

function string(value: unknown, label: string): string {
  if (typeof value !== "string") throw new TypeError(`${label} must be a string`);
  return value;
}

function integer(value: unknown, label: string): number {
  if (typeof value !== "number" || !Number.isSafeInteger(value)) {
    throw new TypeError(`${label} must be a safe integer`);
  }
  return value;
}

function equal(actual: unknown, expected: unknown, label: string): void {
  if (actual !== expected) {
    throw new Error(
      `${label} mismatch: expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`,
    );
  }
}

export const ACE_PINNED_TOKENIZER_ASSETS = PINNED_ASSETS;
