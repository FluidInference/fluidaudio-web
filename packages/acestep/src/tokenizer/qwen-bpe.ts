const BASE_VOCABULARY_SIZE = 151_643;
const FIRST_ADDED_TOKEN_ID = BASE_VOCABULARY_SIZE;
const FIRST_AUDIO_CODE_ID = 151_669;
const AUDIO_CODE_COUNT = 65_535;

export const ACE_QWEN_PAD_TOKEN_ID = 151_643;
export const ACE_QWEN_IM_START_TOKEN_ID = 151_644;
export const ACE_QWEN_IM_END_TOKEN_ID = 151_645;
export const ACE_QWEN_TEXT_POST_TOKEN_ID = 151_643;
export const ACE_PLANNER_AUDIO_CODE_FIRST_TOKEN_ID = FIRST_AUDIO_CODE_ID;
export const ACE_PLANNER_AUDIO_CODE_COUNT = AUDIO_CODE_COUNT;

export type AceTokenizerKind = "text" | "planner";

export interface AceTokenizerEncodeOptions {
  readonly addSpecialTokens?: boolean;
  readonly truncation?: boolean;
  readonly maxLength?: number;
}

export interface AceTokenizerDecodeOptions {
  readonly skipSpecialTokens?: boolean;
}

export interface AceTokenizerBatchOptions extends AceTokenizerEncodeOptions {
  readonly padding?: false | "longest" | "max-length";
  readonly paddingSide?: "left" | "right";
}

export interface AceTokenizedBatch {
  readonly inputIds: readonly (readonly number[])[];
  readonly attentionMask: readonly (readonly number[])[];
}

export interface AceQwenBpeDefinition {
  readonly vocabulary: ReadonlyMap<string, number>;
  readonly tokensById: readonly string[];
  readonly mergeRanks: ReadonlyMap<string, number>;
}

interface AddedTokenMatch {
  readonly end: number;
  readonly id: number;
}

interface TrieNode {
  readonly children: Map<string, TrieNode>;
  id?: number;
}

const BASE_ADDED_TOKENS = Object.freeze([
  "<|endoftext|>",
  "<|im_start|>",
  "<|im_end|>",
  "<|object_ref_start|>",
  "<|object_ref_end|>",
  "<|box_start|>",
  "<|box_end|>",
  "<|quad_start|>",
  "<|quad_end|>",
  "<|vision_start|>",
  "<|vision_end|>",
  "<|vision_pad|>",
  "<|image_pad|>",
  "<|video_pad|>",
  "<tool_call>",
  "</tool_call>",
  "<|fim_prefix|>",
  "<|fim_middle|>",
  "<|fim_suffix|>",
  "<|fim_pad|>",
  "<|repo_name|>",
  "<|file_sep|>",
  "<tool_response>",
  "</tool_response>",
  "<think>",
  "</think>",
]);

const QWEN_PRETOKEN_PATTERN =
  /'(?:s|t|re|ve|m|ll|d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+/giu;

const BYTE_TO_UNICODE = buildByteToUnicode();
const UNICODE_TO_BYTE = new Map(
  BYTE_TO_UNICODE.map((character, byte) => [character, byte] as const),
);
const UTF8_ENCODER = new TextEncoder();
const UTF8_DECODER = new TextDecoder("utf-8", { fatal: false });

/**
 * Exact NFC + byte-level BPE used by both pinned Qwen branches.
 *
 * The planner's large audio-code vocabulary is represented as a contiguous
 * arithmetic range instead of 65,535 duplicated JavaScript strings.
 */
export class AceQwenBpeTokenizer {
  readonly kind: AceTokenizerKind;
  readonly padTokenId = ACE_QWEN_PAD_TOKEN_ID;
  readonly eosTokenId = ACE_QWEN_IM_END_TOKEN_ID;
  readonly modelMaxLength = 131_072;

  private readonly addedTokenTrie = buildAddedTokenTrie();

  constructor(
    kind: AceTokenizerKind,
    private readonly definition: AceQwenBpeDefinition,
  ) {
    this.kind = kind;
  }

  encode(text: string, options: AceTokenizerEncodeOptions = {}): number[] {
    if (typeof text !== "string") throw new TypeError("Tokenizer input must be a string");
    const addSpecialTokens = options.addSpecialTokens ?? true;
    const appendPostToken = this.kind === "text" && addSpecialTokens;
    const content = this.encodeContent(text);

    if (options.truncation === true) {
      const maxLength = options.maxLength ?? this.modelMaxLength;
      assertPositiveInteger(maxLength, "maxLength");
      const contentLimit = maxLength - (appendPostToken ? 1 : 0);
      if (contentLimit < 0) {
        throw new RangeError("maxLength cannot hold the text tokenizer post token");
      }
      if (content.length > contentLimit) content.length = contentLimit;
    } else if (options.maxLength !== undefined) {
      throw new TypeError("maxLength requires truncation: true");
    }

    if (appendPostToken) content.push(ACE_QWEN_TEXT_POST_TOKEN_ID);
    return content;
  }

  encodeBatch(
    texts: readonly string[],
    options: AceTokenizerBatchOptions = {},
  ): AceTokenizedBatch {
    if (texts.length === 0) throw new RangeError("Tokenizer batch cannot be empty");
    const padding = options.padding ?? false;
    if (
      options.maxLength !== undefined &&
      options.truncation !== true &&
      padding !== "max-length"
    ) {
      throw new TypeError(
        "maxLength requires truncation: true or padding: max-length",
      );
    }
    const encodeOptions: AceTokenizerEncodeOptions = {
      ...(options.addSpecialTokens === undefined
        ? {}
        : { addSpecialTokens: options.addSpecialTokens }),
      ...(options.truncation === undefined ? {} : { truncation: options.truncation }),
      ...(options.truncation === true && options.maxLength !== undefined
        ? { maxLength: options.maxLength }
        : {}),
    };
    const encoded = texts.map((text) => this.encode(text, encodeOptions));
    if (padding === false) {
      return {
        inputIds: encoded,
        attentionMask: encoded.map((ids) => ids.map(() => 1)),
      };
    }

    let paddedLength: number;
    if (padding === "longest") {
      paddedLength = Math.max(...encoded.map((ids) => ids.length));
    } else {
      if (options.maxLength === undefined) {
        throw new TypeError("max-length padding requires maxLength");
      }
      assertPositiveInteger(options.maxLength, "maxLength");
      paddedLength = options.maxLength;
    }
    const paddingSide = options.paddingSide ?? "right";
    const inputIds: number[][] = [];
    const attentionMask: number[][] = [];
    for (const ids of encoded) {
      if (ids.length > paddedLength) {
        throw new RangeError(
          `Encoded sequence length ${ids.length} exceeds padding length ${paddedLength}`,
        );
      }
      const missing = paddedLength - ids.length;
      const pads = Array<number>(missing).fill(this.padTokenId);
      const zeros = Array<number>(missing).fill(0);
      const ones = Array<number>(ids.length).fill(1);
      inputIds.push(paddingSide === "left" ? [...pads, ...ids] : [...ids, ...pads]);
      attentionMask.push(
        paddingSide === "left" ? [...zeros, ...ones] : [...ones, ...zeros],
      );
    }
    return { inputIds, attentionMask };
  }

  decode(
    tokenIds: readonly number[],
    options: AceTokenizerDecodeOptions = {},
  ): string {
    const skipSpecialTokens = options.skipSpecialTokens ?? false;
    const result: string[] = [];
    const pendingBytes: number[] = [];
    const flushBytes = (): void => {
      if (pendingBytes.length === 0) return;
      result.push(UTF8_DECODER.decode(Uint8Array.from(pendingBytes)));
      pendingBytes.length = 0;
    };

    for (const tokenId of tokenIds) {
      assertTokenId(tokenId);
      if (skipSpecialTokens && this.isSpecialTokenId(tokenId)) continue;
      const addedContent = this.addedTokenContent(tokenId);
      if (addedContent !== undefined) {
        flushBytes();
        result.push(addedContent);
        continue;
      }
      const token = this.definition.tokensById[tokenId];
      if (token === undefined) throw new RangeError(`Unknown tokenizer token ID ${tokenId}`);
      for (const character of token) {
        const byte = UNICODE_TO_BYTE.get(character);
        if (byte === undefined) {
          throw new Error(`Base vocabulary token ${tokenId} is not byte-level encoded`);
        }
        pendingBytes.push(byte);
      }
    }
    flushBytes();
    return result.join("");
  }

  tokenToId(token: string): number | undefined {
    const base = this.definition.vocabulary.get(token);
    if (base !== undefined) return base;
    const baseAddedIndex = BASE_ADDED_TOKENS.indexOf(token);
    if (baseAddedIndex >= 0) return FIRST_ADDED_TOKEN_ID + baseAddedIndex;
    if (this.kind !== "planner") return undefined;
    const audioCode = parseCanonicalAudioCode(token);
    return audioCode === undefined ? undefined : FIRST_AUDIO_CODE_ID + audioCode;
  }

  idToToken(tokenId: number): string | undefined {
    assertTokenId(tokenId);
    return this.addedTokenContent(tokenId) ?? this.definition.tokensById[tokenId];
  }

  isSpecialTokenId(tokenId: number): boolean {
    if (tokenId >= 151_643 && tokenId <= 151_656) return true;
    return (
      this.kind === "planner" &&
      tokenId >= FIRST_AUDIO_CODE_ID &&
      tokenId < FIRST_AUDIO_CODE_ID + AUDIO_CODE_COUNT
    );
  }

  private encodeContent(text: string): number[] {
    const result: number[] = [];
    let normalStart = 0;
    let cursor = 0;
    while (cursor < text.length) {
      const match = this.matchAddedToken(text, cursor);
      if (match === undefined) {
        cursor += 1;
        continue;
      }
      if (normalStart < cursor) this.encodeNormal(text.slice(normalStart, cursor), result);
      result.push(match.id);
      cursor = match.end;
      normalStart = cursor;
    }
    if (normalStart < text.length) this.encodeNormal(text.slice(normalStart), result);
    return result;
  }

  private encodeNormal(text: string, destination: number[]): void {
    const normalized = text.normalize("NFC");
    QWEN_PRETOKEN_PATTERN.lastIndex = 0;
    let covered = 0;
    for (const match of normalized.matchAll(QWEN_PRETOKEN_PATTERN)) {
      const piece = match[0];
      const index = match.index;
      if (index !== covered || piece.length === 0) {
        throw new Error("Qwen pre-tokenizer did not cover its input exactly");
      }
      covered += piece.length;
      const bytes = UTF8_ENCODER.encode(piece);
      let encodedPiece = "";
      for (const byte of bytes) encodedPiece += BYTE_TO_UNICODE[byte]!;
      const symbols = this.applyBpe(encodedPiece);
      for (const symbol of symbols) {
        const tokenId = this.definition.vocabulary.get(symbol);
        if (tokenId === undefined) {
          throw new Error(`BPE produced a token absent from the vocabulary: ${symbol}`);
        }
        destination.push(tokenId);
      }
    }
    if (covered !== normalized.length) {
      throw new Error("Qwen pre-tokenizer left unmatched input");
    }
  }

  private applyBpe(piece: string): string[] {
    let symbols = Array.from(piece);
    while (symbols.length > 1) {
      let bestRank = Number.POSITIVE_INFINITY;
      let bestLeft = "";
      let bestRight = "";
      for (let index = 0; index + 1 < symbols.length; index += 1) {
        const left = symbols[index]!;
        const right = symbols[index + 1]!;
        const rank = this.definition.mergeRanks.get(mergeKey(left, right));
        if (rank !== undefined && rank < bestRank) {
          bestRank = rank;
          bestLeft = left;
          bestRight = right;
        }
      }
      if (!Number.isFinite(bestRank)) break;
      const merged: string[] = [];
      for (let index = 0; index < symbols.length; ) {
        if (
          index + 1 < symbols.length &&
          symbols[index] === bestLeft &&
          symbols[index + 1] === bestRight
        ) {
          merged.push(bestLeft + bestRight);
          index += 2;
        } else {
          merged.push(symbols[index]!);
          index += 1;
        }
      }
      symbols = merged;
    }
    return symbols;
  }

  private matchAddedToken(text: string, start: number): AddedTokenMatch | undefined {
    let node = this.addedTokenTrie;
    let best: AddedTokenMatch | undefined;
    for (let cursor = start; cursor < text.length; cursor += 1) {
      const next = node.children.get(text[cursor]!);
      if (next === undefined) break;
      node = next;
      if (node.id !== undefined) best = { end: cursor + 1, id: node.id };
    }
    if (this.kind === "planner" && text.startsWith("<|audio_code_", start)) {
      const closing = text.indexOf("|>", start + 14);
      if (closing >= 0) {
        const content = text.slice(start, closing + 2);
        const audioCode = parseCanonicalAudioCode(content);
        if (audioCode !== undefined) {
          const candidate = { end: closing + 2, id: FIRST_AUDIO_CODE_ID + audioCode };
          if (best === undefined || candidate.end > best.end) best = candidate;
        }
      }
    }
    return best;
  }

  private addedTokenContent(tokenId: number): string | undefined {
    const baseAddedIndex = tokenId - FIRST_ADDED_TOKEN_ID;
    if (baseAddedIndex >= 0 && baseAddedIndex < BASE_ADDED_TOKENS.length) {
      return BASE_ADDED_TOKENS[baseAddedIndex];
    }
    if (
      this.kind === "planner" &&
      tokenId >= FIRST_AUDIO_CODE_ID &&
      tokenId < FIRST_AUDIO_CODE_ID + AUDIO_CODE_COUNT
    ) {
      return `<|audio_code_${tokenId - FIRST_AUDIO_CODE_ID}|>`;
    }
    return undefined;
  }
}

export function aceQwenMergeKey(left: string, right: string): string {
  return mergeKey(left, right);
}

function buildAddedTokenTrie(): TrieNode {
  const root: TrieNode = { children: new Map() };
  BASE_ADDED_TOKENS.forEach((content, index) => {
    let node = root;
    for (const character of content) {
      let child = node.children.get(character);
      if (child === undefined) {
        child = { children: new Map() };
        node.children.set(character, child);
      }
      node = child;
    }
    node.id = FIRST_ADDED_TOKEN_ID + index;
  });
  return root;
}

function parseCanonicalAudioCode(content: string): number | undefined {
  const match = /^<\|audio_code_(0|[1-9]\d*)\|>$/.exec(content);
  if (match === null) return undefined;
  const value = Number(match[1]);
  return Number.isSafeInteger(value) && value >= 0 && value < AUDIO_CODE_COUNT
    ? value
    : undefined;
}

function mergeKey(left: string, right: string): string {
  return `${left}\0${right}`;
}

function buildByteToUnicode(): readonly string[] {
  const bytes: number[] = [];
  for (let byte = 33; byte <= 126; byte += 1) bytes.push(byte);
  for (let byte = 161; byte <= 172; byte += 1) bytes.push(byte);
  for (let byte = 174; byte <= 255; byte += 1) bytes.push(byte);
  const codePoints = [...bytes];
  let extra = 0;
  for (let byte = 0; byte < 256; byte += 1) {
    if (bytes.includes(byte)) continue;
    bytes.push(byte);
    codePoints.push(256 + extra);
    extra += 1;
  }
  const result: string[] = Array(256);
  bytes.forEach((byte, index) => {
    result[byte] = String.fromCodePoint(codePoints[index]!);
  });
  return result;
}

function assertPositiveInteger(value: number, name: string): void {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new RangeError(`${name} must be a positive safe integer`);
  }
}

function assertTokenId(value: number): void {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new RangeError(`Invalid tokenizer token ID ${String(value)}`);
  }
}

export const ACE_QWEN_BASE_VOCABULARY_SIZE = BASE_VOCABULARY_SIZE;
export const ACE_QWEN_BASE_ADDED_TOKENS = BASE_ADDED_TOKENS;
