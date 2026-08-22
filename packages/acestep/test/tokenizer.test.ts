import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";

import { beforeAll, describe, expect, it } from "vitest";

import { renderAceQwenChat } from "../src/tokenizer/chat.js";
import {
  formatAceTextEncoderCaptionInput,
  formatAceTextEncoderLyricsInput,
} from "../src/tokenizer/conditioning-text.js";
import {
  loadPinnedAceTokenizer,
  type LoadedAceTokenizer,
} from "../src/tokenizer/loader.js";
import {
  AceQwenBpeTokenizer,
  aceQwenMergeKey,
  type AceQwenBpeDefinition,
} from "../src/tokenizer/qwen-bpe.js";

const MODEL_ASSET_ROOT = resolve("model/files-reference/assets");
const HAS_LOCAL_MODEL_ASSETS = existsSync(
  resolve(MODEL_ASSET_ROOT, "qwen/tokenizer.json"),
);

const FIXTURE_TOKEN_GOLDENS = {
  "direct-instrumental-short": {
    instruction: [12, "e0793a2513cab09f705364a29f97c224282b9521d1f1e6d27f6a3781f2dd0c8f"],
    caption: [27, "f84b537b1285e804653e9681eaad9d5b5e652350d90da2df7dbe80e8f237b460"],
    lyrics: [5, "8a6d025554c08ec4cd52d8df73dc4cc61d2a0397e57227c89dbbf6fadcbc3922"],
  },
  "direct-lyrics-long-condition": {
    instruction: [12, "e0793a2513cab09f705364a29f97c224282b9521d1f1e6d27f6a3781f2dd0c8f"],
    caption: [87, "67861fae0b0220ea31e95e59ec8d4db7d08b3481cb7e29515ff3c8a85937d226"],
    lyrics: [401, "04c312575145072365ce5aea8592bc20a7ba397600c7b3fd2cb8e8793a5e6913"],
  },
  "direct-lyrics-short": {
    instruction: [12, "e0793a2513cab09f705364a29f97c224282b9521d1f1e6d27f6a3781f2dd0c8f"],
    caption: [28, "2da57e82b9dd8b322269293f52c27e81ee2ab44e187d1f2916cf97b358ae095b"],
    lyrics: [40, "9fa133a1346e06cfc3a951c45bc37f7b645d23e9d57486e3e0a0d94590995884"],
  },
  "planner-lyrics-short": {
    instruction: [11, "061f09e6802a330c6fd17e5da745597cef803ba5141a7fb3233c4c680f8b514d"],
    caption: [30, "df4b4c2978f2d60547d940b46abed00ef71d11d1c89f7f2ff0d022e6503eb636"],
    lyrics: [43, "523312817482cfca92336de07ce72bbabef45239367859b2eea03e3603e6720f"],
  },
} as const;

const FIXTURE_CONDITIONING_GOLDENS = {
  "direct-instrumental-short": [
    [81, "2889ba54a6c6cc7106ecb2a4c41a45931afc0065d545f7f60ad750e15866d454"],
    [15, "b4b58cd318163b4dfaa02b7ddbf46b18d84a415909c7662f9538c0b9053f3764"],
  ],
  "direct-lyrics-long-condition": [
    [141, "aa5500c0cbb4bfd5b617a0cf390380a8e0576f9f1bd8fd3bcaadf965e0aa3dd3"],
    [411, "96bdbd98ce1bfa8cc049e0a2e4156cb4f594feab6f007d2b8b8000f9b671600e"],
  ],
  "direct-lyrics-short": [
    [81, "ac4c8b213ee49a3029e345e05d4d149520902d99feaca8492b8d57432711c3d8"],
    [50, "27c70cdc9f1a0d094bb19e8ee1ce1feed7766011213317b8002281eac661a8f6"],
  ],
  "planner-lyrics-short": [
    [81, "43ab6ee5a9a6ea8616dd448670aebbfb4a65a7e1f90119bddccdb37ccc8e30bd"],
    [53, "cda6ccd1f49d4011a93330a992a5e17c65517ac241ce9ac778834ee57b6e3bbd"],
  ],
} as const;

describe("Qwen byte-level BPE core", () => {
  it("applies ranked merges and the text-only post token", () => {
    const tokenizer = new AceQwenBpeTokenizer("text", miniatureDefinition());
    expect(tokenizer.encode("hello", { addSpecialTokens: false })).toEqual([7]);
    expect(tokenizer.encode("hello")).toEqual([7, 151_643]);
    expect(tokenizer.decode([7, 151_643])).toBe("hello<|endoftext|>");
    expect(tokenizer.decode([7, 151_643], { skipSpecialTokens: true })).toBe("hello");
  });

  it("reserves room for the text post token during right truncation", () => {
    const tokenizer = new AceQwenBpeTokenizer("text", miniatureDefinition());
    expect(
      tokenizer.encode("hellohello", {
        addSpecialTokens: true,
        truncation: true,
        maxLength: 2,
      }),
    ).toEqual([7, 151_643]);
  });

  it("produces exact left/right padding masks", () => {
    const tokenizer = new AceQwenBpeTokenizer("planner", miniatureDefinition());
    expect(
      tokenizer.encodeBatch(["hello", "hellohello"], {
        addSpecialTokens: false,
        padding: "longest",
        paddingSide: "left",
      }),
    ).toEqual({
      inputIds: [
        [151_643, 7],
        [7, 7],
      ],
      attentionMask: [
        [0, 1],
        [1, 1],
      ],
    });
  });

  it("allows fixed padding without implicitly enabling truncation", () => {
    const tokenizer = new AceQwenBpeTokenizer("planner", miniatureDefinition());
    expect(
      tokenizer.encodeBatch(["hello"], {
        addSpecialTokens: false,
        padding: "max-length",
        maxLength: 2,
      }),
    ).toEqual({ inputIds: [[7, 151_643]], attentionMask: [[1, 0]] });
    expect(() =>
      tokenizer.encodeBatch(["hellohellohello"], {
        addSpecialTokens: false,
        padding: "max-length",
        maxLength: 2,
      }),
    ).toThrow(/exceeds padding length/);
  });

  it("recognizes only canonical in-range planner audio codes", () => {
    const tokenizer = new AceQwenBpeTokenizer("planner", miniatureDefinition());
    expect(
      tokenizer.encode("<|audio_code_0|><|audio_code_42|><|audio_code_65534|>", {
        addSpecialTokens: false,
      }),
    ).toEqual([151_669, 151_711, 217_203]);
    expect(tokenizer.idToToken(217_203)).toBe("<|audio_code_65534|>");
    expect(tokenizer.isSpecialTokenId(217_203)).toBe(true);
  });
});

describe("pinned no-tools Qwen chat rendering", () => {
  it("matches the upstream system/user generation prompt byte-for-byte", () => {
    expect(
      renderAceQwenChat(
        [
          { role: "system", content: "SYS\n" },
          { role: "user", content: "Hi 🌧️" },
        ],
        { addGenerationPrompt: true },
      ),
    ).toBe(
      "<|im_start|>system\nSYS\n<|im_end|>\n" +
        "<|im_start|>user\nHi 🌧️<|im_end|>\n" +
        "<|im_start|>assistant\n",
    );
  });

  it("matches Qwen's explicit thinking-disabled open assistant turn", () => {
    expect(
      renderAceQwenChat([{ role: "user", content: "hello" }], {
        addGenerationPrompt: true,
        enableThinking: false,
      }),
    ).toBe(
      "<|im_start|>user\nhello<|im_end|>\n" +
        "<|im_start|>assistant\n<think>\n\n</think>\n\n",
    );
  });

  it("reconstructs assistant reasoning exactly like the pinned template", () => {
    expect(
      renderAceQwenChat([
        { role: "user", content: "prompt" },
        { role: "assistant", content: "<think>\nwhy\n</think>\nanswer" },
      ]),
    ).toBe(
      "<|im_start|>user\nprompt<|im_end|>\n" +
        "<|im_start|>assistant\n<think>\nwhy\n</think>\n\nanswer<|im_end|>\n",
    );
  });
});

describe.skipIf(!HAS_LOCAL_MODEL_ASSETS)("authenticated pinned tokenizer assets", () => {
  let text: LoadedAceTokenizer;
  let planner: LoadedAceTokenizer;

  beforeAll(async () => {
    [text, planner] = await Promise.all([
      load("text", "qwen"),
      load("planner", "planner"),
    ]);
  });

  it("loads and validates the exact text and planner asset identities", () => {
    expect(text.assetIdentity.tokenizerSha256).toBe(
      "def76fb086971c7867b829c23a26261e38d9d74e02139253b38aeb9df8b4b50a",
    );
    expect(planner.assetIdentity.tokenizerSha256).toBe(
      "35af56c3f5cb3ea2cc578aa28a8937770981d504f183ac5c8c38baf4bbd4af4d",
    );
  });

  it("matches Python tokenizers 0.22.1 on Unicode, whitespace, and added tokens", () => {
    const vectors = [
      ["", []],
      ["hello", [14990]],
      [" Hello  ", [21927, 256]],
      [
        "I’m naïve — 東京 🌧️\r\nnext",
        [40, 4249, 94880, 586, 1959, 60596, 109, 46553, 11162, 234, 100, 30543, 319, 3600],
      ],
      ["e\u0301 é", [963, 3958]],
      ["'S 're 'VE", [13272, 364, 265, 364, 4491]],
      ["<think>\n\n</think>", [151667, 271, 151668]],
      ["<|im_start|>user\nx<|im_end|>\n", [151644, 872, 198, 87, 151645, 198]],
    ] as const;
    for (const [source, expected] of vectors) {
      const ids = text.tokenizer.encode(source, { addSpecialTokens: false });
      expect(ids, JSON.stringify(source)).toEqual(expected);
      expect(text.tokenizer.decode(ids), JSON.stringify(source)).toBe(source.normalize("NFC"));
    }
  });

  it("matches Python/HF planner audio-code boundaries and invalid spellings", () => {
    expect(
      planner.tokenizer.encode(
        "<|audio_code_0|><|audio_code_42|><|audio_code_65534|>",
        { addSpecialTokens: false },
      ),
    ).toEqual([151669, 151711, 217203]);
    expect(
      planner.tokenizer.encode("<|audio_code_0001|>", { addSpecialTokens: false }),
    ).toEqual([27, 91, 16736, 4136, 62, 15, 15, 15, 16, 91, 29]);
    expect(
      planner.tokenizer.encode("<|audio_code_65535|>", { addSpecialTokens: false }),
    ).toEqual([27, 91, 16736, 4136, 62, 21, 20, 20, 18, 20, 91, 29]);
  });

  it("matches HF token IDs for the pinned chat prompt", () => {
    const prompt = renderAceQwenChat(
      [
        { role: "system", content: "SYS\n" },
        { role: "user", content: "Hi 🌧️" },
      ],
      { addGenerationPrompt: true },
    );
    expect(planner.tokenizer.encode(prompt)).toEqual([
      151644, 8948, 198, 37931, 198, 151645, 198, 151644, 872, 198, 13048,
      11162, 234, 100, 30543, 151645, 198, 151644, 77091, 198,
    ]);
    expect(text.tokenizer.encode(prompt)).toEqual([
      151644, 8948, 198, 37931, 198, 151645, 198, 151644, 872, 198, 13048,
      11162, 234, 100, 30543, 151645, 198, 151644, 77091, 198, 151643,
    ]);
  });

  it("matches Python/HF for every instruction, caption, and lyric golden field", () => {
    for (const [fixtureId, fields] of Object.entries(FIXTURE_TOKEN_GOLDENS)) {
      const fixture = JSON.parse(
        readFileSync(resolve(`golden/fixtures/${fixtureId}.json`), "utf8"),
      ) as { contract: { request: Record<string, string> } };
      for (const [field, [expectedLength, expectedDigest]] of Object.entries(fields)) {
        const ids = text.tokenizer.encode(fixture.contract.request[field]!);
        expect(ids.length, `${fixtureId}.${field} length`).toBe(expectedLength);
        expect(tokenIdDigest(ids), `${fixtureId}.${field} token digest`).toBe(
          expectedDigest,
        );
      }
    }
  });

  it("matches full pinned caption and lyric conditioning inputs for every fixture", () => {
    for (const [fixtureId, expected] of Object.entries(FIXTURE_CONDITIONING_GOLDENS)) {
      const fixture = JSON.parse(
        readFileSync(resolve(`golden/fixtures/${fixtureId}.json`), "utf8"),
      ) as {
        contract: {
          request: {
            instruction: string;
            caption: string;
            lyrics: string;
            vocalLanguage: string;
            bpm: number | null;
            keyScale: string;
            timeSignature: string;
            durationSeconds: number;
          };
        };
      };
      const request = fixture.contract.request;
      const metadata =
        `- bpm: ${request.bpm ?? "N/A"}\n` +
        `- timesignature: ${request.timeSignature || "N/A"}\n` +
        `- keyscale: ${request.keyScale || "N/A"}\n` +
        `- duration: ${Math.trunc(request.durationSeconds)} seconds\n`;
      const sources = [
        formatAceTextEncoderCaptionInput(
          request.instruction,
          request.caption,
          metadata,
        ),
        formatAceTextEncoderLyricsInput(request.lyrics, request.vocalLanguage),
      ];
      sources.forEach((source, index) => {
        const ids = text.tokenizer.encode(source, {
          truncation: true,
          maxLength: index === 0 ? 256 : 2_048,
        });
        expect(ids.length, `${fixtureId} conditioning input ${index} length`).toBe(
          expected[index]![0],
        );
        expect(tokenIdDigest(ids), `${fixtureId} conditioning input ${index} digest`).toBe(
          expected[index]![1],
        );
        expect(ids.at(-1), `${fixtureId} post token`).toBe(151_643);
      });
    }
  });

  it("pins the canonical OPT-0018 180-second conditioning token count", () => {
    const caption = formatAceTextEncoderCaptionInput(
      "Fill the audio semantic mask based on the given conditions:",
      "Warm analog synth arpeggios over a restrained breakbeat, rounded electric bass, airy pads, instrumental, detailed stereo production.",
      "- bpm: 104\n" +
        "- timesignature: 4\n" +
        "- keyscale: D minor\n" +
        "- duration: 180 seconds\n",
    );
    const lyrics = formatAceTextEncoderLyricsInput(
      "[Instrumental]",
      "unknown",
    );
    const textIds = text.tokenizer.encode(caption, {
      truncation: true,
      maxLength: 256,
    });
    const lyricIds = text.tokenizer.encode(lyrics, {
      truncation: true,
      maxLength: 2_048,
    });

    expect(textIds).toHaveLength(82);
    expect(tokenIdDigest(textIds)).toBe(
      "8067ee5c606e45e54d991364aa82a0ef7303e2a4e98831a01bb974236cafb3b2",
    );
    expect(lyricIds).toHaveLength(15);
    expect(tokenIdDigest(lyricIds)).toBe(
      "b4b58cd318163b4dfaa02b7ddbf46b18d84a415909c7662f9538c0b9053f3764",
    );
    expect(textIds.length + 1 + lyricIds.length).toBe(98);
  });

  it("rejects any mutation before parsing or model construction", async () => {
    const config = assetText("qwen/tokenizer_config.json");
    const mutated = config.replace("Qwen2Tokenizer", "Qwen3Tokenizer");
    expect(mutated).toHaveLength(config.length);
    await expect(
      loadPinnedAceTokenizer("text", {
        tokenizerJson: "not reached",
        tokenizerConfigJson: mutated,
        chatTemplate: "not reached",
      }),
    ).rejects.toThrow(/SHA-256 mismatch/);
  });
});

async function load(
  kind: "text" | "planner",
  assetDirectory: "qwen" | "planner",
): Promise<LoadedAceTokenizer> {
  return loadPinnedAceTokenizer(kind, {
    tokenizerJson: assetText(`${assetDirectory}/tokenizer.json`),
    tokenizerConfigJson: assetText(`${assetDirectory}/tokenizer_config.json`),
    chatTemplate: assetText(`${assetDirectory}/chat_template.jinja`),
  });
}

function assetText(relativePath: string): string {
  return readFileSync(resolve(MODEL_ASSET_ROOT, relativePath), "utf8");
}

function tokenIdDigest(ids: readonly number[]): string {
  const bytes = Buffer.alloc(ids.length * 4);
  ids.forEach((tokenId, index) => bytes.writeUInt32LE(tokenId, index * 4));
  return createHash("sha256").update(bytes).digest("hex");
}

function miniatureDefinition(): AceQwenBpeDefinition {
  const tokens = ["h", "e", "l", "o", "he", "ll", "hell", "hello"];
  const vocabulary = new Map(tokens.map((token, tokenId) => [token, tokenId] as const));
  return {
    vocabulary,
    tokensById: tokens,
    mergeRanks: new Map([
      [aceQwenMergeKey("h", "e"), 0],
      [aceQwenMergeKey("l", "l"), 1],
      [aceQwenMergeKey("he", "ll"), 2],
      [aceQwenMergeKey("hell", "o"), 3],
    ]),
  };
}
