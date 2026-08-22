import type { AceModelProfileId } from "../../src/webgpu/capabilities.js";
import {
  AceCorrectnessTextEncoderRuntime,
  createAceTextEncoderControlData,
} from "../../src/webgpu/text-encoder.js";
import {
  planAceQwen3Block,
  type AceQwen3BlockScratch,
  type AceQwen3BlockWeights,
  type AceQwen3Config,
} from "../../src/webgpu/qwen3.js";

const TOKENS = 3;
const LYRIC_TOKENS = 2;
const HIDDEN = 4;
const INTERMEDIATE = 6;
const QUERY_HEADS = 2;
const KEY_VALUE_HEADS = 1;
const HEAD_DIMENSION = 2;
const VOCABULARY = 7;
const LAYERS = 2;
const TOKEN_IDS = new Uint32Array([1, 5, 3]);
const LYRIC_TOKEN_IDS = new Uint32Array([6, 2]);

const CONFIG: AceQwen3Config = Object.freeze({
  id: "ace-browser-tiny-text-encoder",
  hiddenSize: HIDDEN,
  intermediateSize: INTERMEDIATE,
  layerCount: LAYERS,
  queryHeads: QUERY_HEADS,
  keyValueHeads: KEY_VALUE_HEADS,
  headDimension: HEAD_DIMENSION,
  vocabularySize: VOCABULARY,
  maximumPositionEmbeddings: 8,
  ropeTheta: 100,
  rmsNormEpsilon: 1e-5,
  attentionBias: false,
  hiddenActivation: "silu",
  tieWordEmbeddings: true,
});

interface LayerWeights {
  readonly inputLayerNorm: Float32Array;
  readonly queryProjection: Float32Array;
  readonly keyProjection: Float32Array;
  readonly valueProjection: Float32Array;
  readonly queryNorm: Float32Array;
  readonly keyNorm: Float32Array;
  readonly outputProjection: Float32Array;
  readonly postAttentionLayerNorm: Float32Array;
  readonly gateProjection: Float32Array;
  readonly upProjection: Float32Array;
  readonly downProjection: Float32Array;
}

interface ModelWeights {
  readonly embedding: Float32Array;
  readonly layers: readonly LayerWeights[];
  readonly finalNorm: Float32Array;
}

interface ProfileResult {
  readonly profile: AceModelProfileId;
  readonly primitiveCount: number;
  readonly quantumCount: number;
  readonly textMaximumError: number;
  readonly lyricMaximumError: number;
  readonly maximumAbsoluteTextValue: number;
  readonly ropeTableMaximumUlp: number;
}

const BASE_WEIGHTS: ModelWeights = Object.freeze({
  embedding: matrix(VOCABULARY, HIDDEN, 23),
  layers: Object.freeze([
    layerWeights(3),
    layerWeights(29),
  ]),
  finalNorm: new Float32Array([0.95, 1.05, 0.9, 1.1]),
});

const resultNode = requireResultNode();

void run().then(
  (results) => finish("passed", JSON.stringify(results)),
  (error: unknown) =>
    finish("failed", error instanceof Error ? error.stack ?? error.message : String(error)),
);

async function run(): Promise<readonly ProfileResult[]> {
  if (navigator.gpu === undefined) throw new Error("WebGPU unavailable");
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (adapter === null) throw new Error("No WebGPU adapter");
  const requiredFeatures: GPUFeatureName[] = adapter.features.has("shader-f16")
    ? ["shader-f16"]
    : [];
  const device = await adapter.requestDevice({ requiredFeatures });
  try {
    const results: ProfileResult[] = [await runProfile(device, "reference-bf16")];
    if (device.features.has("shader-f16")) {
      results.push(await runProfile(device, "raw-fp16"));
    }
    return results;
  } finally {
    device.destroy();
  }
}

async function runProfile(
  device: GPUDevice,
  profile: AceModelProfileId,
): Promise<ProfileResult> {
  const runtime = AceCorrectnessTextEncoderRuntime.create(device, profile);
  const buffers: GPUBuffer[] = [];
  const weights = quantizedModelWeights(profile, BASE_WEIGHTS);
  const embeddingShards = [
    {
      firstRow: 0,
      rowCount: 4,
      weight: weightBinding(device, buffers, profile, BASE_WEIGHTS.embedding.slice(0, 4 * HIDDEN)),
    },
    {
      firstRow: 4,
      rowCount: 3,
      weight: weightBinding(device, buffers, profile, BASE_WEIGHTS.embedding.slice(4 * HIDDEN)),
    },
  ] as const;
  const gpuLayers: readonly AceQwen3BlockWeights[] = BASE_WEIGHTS.layers.map(
    (layer) => gpuLayerWeights(device, buffers, profile, layer),
  );
  const plan = planAceQwen3Block(CONFIG, {
    batch: 1,
    tokens: TOKENS,
    attention: { kind: "uncached" },
  });
  try {
    const controls = createAceTextEncoderControlData(TOKENS, CONFIG);
    const ropeTableMaximumUlp = validateRopeTables(controls.cosine, controls.sine);
    const tokenIds = track(buffers, storageBuffer(device, TOKEN_IDS));
    const output = track(
      buffers,
      activationBuffer(device, profile, plan.hiddenElements, undefined, true),
    );
    const dispatch = await runtime.createQwen3ModelDispatch(
      `ace-browser-${profile}-text-encoder`,
      CONFIG,
      { batch: 1, tokens: TOKENS },
      {
        tokenIds: binding(tokenIds),
        output: binding(output),
        weights: {
          embedding: embeddingShards,
          layers: gpuLayers,
          finalNorm: weightBinding(device, buffers, profile, BASE_WEIGHTS.finalNorm),
        },
        controls: {
          validLengths: binding(track(buffers, storageBuffer(device, controls.validLengths))),
          queryPositions: binding(track(buffers, storageBuffer(device, controls.queryPositions))),
          keyValidity: binding(track(buffers, storageBuffer(device, controls.keyValidity))),
          cosine: binding(track(buffers, storageBuffer(device, controls.cosine))),
          sine: binding(track(buffers, storageBuffer(device, controls.sine))),
        },
        scratch: {
          embedded: binding(track(
            buffers,
            activationBuffer(device, profile, plan.hiddenElements),
          )),
          block: allocateScratch(device, buffers, profile, plan),
          layerOutputs: [
            binding(track(buffers, activationBuffer(device, profile, plan.hiddenElements))),
            binding(track(buffers, activationBuffer(device, profile, plan.hiddenElements))),
          ],
        },
      },
    );
    if (dispatch.quanta.length !== LAYERS + 2 || dispatch.primitiveCount !== 44) {
      throw new Error(
        `${profile} composer emitted ${dispatch.quanta.length} quanta and ` +
          `${dispatch.primitiveCount} primitives`,
      );
    }
    for (const quantum of dispatch.quanta) await execute(device, quantum);
    const actualText = await readProfile(
      device,
      profile,
      output,
      plan.hiddenElements,
    );
    const expectedText = cpuModel(profile, weights, TOKEN_IDS);
    const textMaximumError = assertClose(
      actualText,
      expectedText,
      profile === "raw-fp16" ? 3e-2 : 1.5e-4,
      `${profile} full text encoder`,
    );
    const maximumAbsoluteTextValue = actualText.reduce(
      (maximum, value) => Math.max(maximum, Math.abs(value)),
      0,
    );
    if (!(maximumAbsoluteTextValue > 1e-4)) {
      throw new Error(`${profile} text output is unexpectedly zero`);
    }

    const lyricIds = track(buffers, storageBuffer(device, LYRIC_TOKEN_IDS));
    const lyricOutput = track(
      buffers,
      activationBuffer(device, profile, LYRIC_TOKENS * HIDDEN, undefined, true),
    );
    const lyric = await runtime.createQwen3EmbeddingDispatch(
      `ace-browser-${profile}-lyric-embedding`,
      CONFIG,
      LYRIC_TOKENS,
      {
        tokenIds: binding(lyricIds),
        embedding: embeddingShards,
        output: binding(lyricOutput),
      },
    );
    if (lyric.primitiveCount !== embeddingShards.length) {
      throw new Error(`${profile} lyric embedding omitted a vocabulary shard`);
    }
    await execute(device, lyric);
    const actualLyric = await readProfile(
      device,
      profile,
      lyricOutput,
      LYRIC_TOKENS * HIDDEN,
    );
    const expectedLyric = cpuEmbedding(weights.embedding, LYRIC_TOKEN_IDS);
    const lyricMaximumError = assertClose(
      actualLyric,
      expectedLyric,
      0,
      `${profile} lyric embedding lookup`,
    );

    return {
      profile,
      primitiveCount: dispatch.primitiveCount,
      quantumCount: dispatch.quanta.length,
      textMaximumError,
      lyricMaximumError,
      maximumAbsoluteTextValue,
      ropeTableMaximumUlp,
    };
  } finally {
    runtime.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
}

function cpuModel(
  profile: AceModelProfileId,
  weights: ModelWeights,
  tokenIds: Uint32Array,
): Float32Array {
  let hidden = cpuEmbedding(weights.embedding, tokenIds);
  for (const layer of weights.layers) {
    hidden = cpuBlock(profile, layer, hidden);
  }
  return cpuRmsNorm(
    profile,
    hidden,
    weights.finalNorm,
    HIDDEN,
    CONFIG.rmsNormEpsilon,
  );
}

function cpuEmbedding(
  embedding: Float32Array,
  tokenIds: Uint32Array,
): Float32Array {
  const output = new Float32Array(tokenIds.length * HIDDEN);
  for (let token = 0; token < tokenIds.length; token += 1) {
    const id = tokenIds[token]!;
    for (let dimension = 0; dimension < HIDDEN; dimension += 1) {
      output[token * HIDDEN + dimension] = embedding[id * HIDDEN + dimension]!;
    }
  }
  return output;
}

function cpuBlock(
  profile: AceModelProfileId,
  weights: LayerWeights,
  input: Float32Array,
): Float32Array {
  const normalized = cpuRmsNorm(
    profile,
    input,
    weights.inputLayerNorm,
    HIDDEN,
    CONFIG.rmsNormEpsilon,
  );
  const queryFlat = cpuGemm(
    profile,
    normalized,
    weights.queryProjection,
    TOKENS,
    HIDDEN,
    QUERY_HEADS * HEAD_DIMENSION,
  );
  const keyFlat = cpuGemm(
    profile,
    normalized,
    weights.keyProjection,
    TOKENS,
    HIDDEN,
    KEY_VALUE_HEADS * HEAD_DIMENSION,
  );
  const valueFlat = cpuGemm(
    profile,
    normalized,
    weights.valueProjection,
    TOKENS,
    HIDDEN,
    KEY_VALUE_HEADS * HEAD_DIMENSION,
  );
  const queryHeads = splitHeads(queryFlat, TOKENS, QUERY_HEADS, HEAD_DIMENSION);
  const keyHeads = splitHeads(keyFlat, TOKENS, KEY_VALUE_HEADS, HEAD_DIMENSION);
  const valueHeads = splitHeads(valueFlat, TOKENS, KEY_VALUE_HEADS, HEAD_DIMENSION);
  const normalizedQuery = cpuRmsNorm(
    profile,
    queryHeads,
    weights.queryNorm,
    HEAD_DIMENSION,
    CONFIG.rmsNormEpsilon,
  );
  const normalizedKey = cpuRmsNorm(
    profile,
    keyHeads,
    weights.keyNorm,
    HEAD_DIMENSION,
    CONFIG.rmsNormEpsilon,
  );
  const rope = tinyTransformersRopeOracle();
  const query = cpuRope(profile, normalizedQuery, rope, QUERY_HEADS);
  const key = cpuRope(profile, normalizedKey, rope, KEY_VALUE_HEADS);
  const attention = cpuCausalAttention(profile, query, key, valueHeads);
  const merged = mergeHeads(attention, TOKENS, QUERY_HEADS, HEAD_DIMENSION);
  const projectedAttention = cpuGemm(
    profile,
    merged,
    weights.outputProjection,
    TOKENS,
    QUERY_HEADS * HEAD_DIMENSION,
    HIDDEN,
  );
  const afterAttention = cpuAdd(profile, input, projectedAttention);
  const postNorm = cpuRmsNorm(
    profile,
    afterAttention,
    weights.postAttentionLayerNorm,
    HIDDEN,
    CONFIG.rmsNormEpsilon,
  );
  const gate = cpuGemm(
    profile,
    postNorm,
    weights.gateProjection,
    TOKENS,
    HIDDEN,
    INTERMEDIATE,
  );
  const up = cpuGemm(
    profile,
    postNorm,
    weights.upProjection,
    TOKENS,
    HIDDEN,
    INTERMEDIATE,
  );
  const activated = new Float32Array(gate.length);
  for (let index = 0; index < gate.length; index += 1) {
    const silu = store(profile, gate[index]! / (1 + Math.exp(-gate[index]!)));
    activated[index] = store(profile, silu * up[index]!);
  }
  const projectedMlp = cpuGemm(
    profile,
    activated,
    weights.downProjection,
    TOKENS,
    INTERMEDIATE,
    HIDDEN,
  );
  return cpuAdd(profile, afterAttention, projectedMlp);
}

function cpuGemm(
  profile: AceModelProfileId,
  activation: Float32Array,
  weight: Float32Array,
  rows: number,
  inner: number,
  columns: number,
): Float32Array {
  const output = new Float32Array(rows * columns);
  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      let sum = Math.fround(0);
      for (let k = 0; k < inner; k += 1) {
        sum = Math.fround(
          sum + Math.fround(
            activation[row * inner + k]! * weight[column * inner + k]!,
          ),
        );
      }
      output[row * columns + column] = store(profile, sum);
    }
  }
  return output;
}

function cpuRmsNorm(
  profile: AceModelProfileId,
  input: Float32Array,
  weight: Float32Array,
  width: number,
  epsilon: number,
): Float32Array {
  const output = new Float32Array(input.length);
  for (let row = 0; row < input.length / width; row += 1) {
    let sum = Math.fround(0);
    for (let column = 0; column < width; column += 1) {
      const value = input[row * width + column]!;
      sum = Math.fround(sum + Math.fround(value * value));
    }
    const inverse = Math.fround(1 / Math.sqrt(Math.fround(sum / width + epsilon)));
    for (let column = 0; column < width; column += 1) {
      const value = input[row * width + column]!;
      output[row * width + column] = profile === "raw-fp16"
        ? store(profile, store(profile, value * inverse) * weight[column]!)
        : Math.fround(Math.fround(value * inverse) * weight[column]!);
    }
  }
  return output;
}

function cpuRope(
  profile: AceModelProfileId,
  input: Float32Array,
  tables: Readonly<{ cosine: Float32Array; sine: Float32Array }>,
  heads: number,
): Float32Array {
  const output = new Float32Array(input.length);
  const half = HEAD_DIMENSION / 2;
  for (let head = 0; head < heads; head += 1) {
    for (let token = 0; token < TOKENS; token += 1) {
      for (let dimension = 0; dimension < HEAD_DIMENSION; dimension += 1) {
        const index = (head * TOKENS + token) * HEAD_DIMENSION + dimension;
        const rotatedDimension = dimension < half ? dimension + half : dimension - half;
        const rotatedIndex = index - dimension + rotatedDimension;
        const tableIndex = token * HEAD_DIMENSION + dimension;
        const sign = dimension < half ? -1 : 1;
        output[index] = store(profile, Math.fround(
          Math.fround(input[index]! * tables.cosine[tableIndex]!) +
            Math.fround(sign * input[rotatedIndex]! * tables.sine[tableIndex]!),
        ));
      }
    }
  }
  return output;
}

function cpuCausalAttention(
  profile: AceModelProfileId,
  query: Float32Array,
  key: Float32Array,
  value: Float32Array,
): Float32Array {
  const output = new Float32Array(QUERY_HEADS * TOKENS * HEAD_DIMENSION);
  const headsPerKeyValue = QUERY_HEADS / KEY_VALUE_HEADS;
  for (let queryHead = 0; queryHead < QUERY_HEADS; queryHead += 1) {
    const keyValueHead = Math.floor(queryHead / headsPerKeyValue);
    for (let queryToken = 0; queryToken < TOKENS; queryToken += 1) {
      const scores: number[] = [];
      for (let keyToken = 0; keyToken <= queryToken; keyToken += 1) {
        let score = Math.fround(0);
        for (let dimension = 0; dimension < HEAD_DIMENSION; dimension += 1) {
          const queryIndex =
            (queryHead * TOKENS + queryToken) * HEAD_DIMENSION + dimension;
          const keyIndex =
            (keyValueHead * TOKENS + keyToken) * HEAD_DIMENSION + dimension;
          score = Math.fround(
            score + Math.fround(query[queryIndex]! * key[keyIndex]!),
          );
        }
        scores.push(Math.fround(score / Math.sqrt(HEAD_DIMENSION)));
      }
      const maximum = Math.max(...scores);
      const exponentials = scores.map((score) => Math.exp(score - maximum));
      const denominator = exponentials.reduce((sum, value) => sum + value, 0);
      for (let dimension = 0; dimension < HEAD_DIMENSION; dimension += 1) {
        let weighted = 0;
        for (let keyToken = 0; keyToken <= queryToken; keyToken += 1) {
          const valueIndex =
            (keyValueHead * TOKENS + keyToken) * HEAD_DIMENSION + dimension;
          weighted += exponentials[keyToken]! * value[valueIndex]!;
        }
        const outputIndex =
          (queryHead * TOKENS + queryToken) * HEAD_DIMENSION + dimension;
        output[outputIndex] = store(profile, weighted / denominator);
      }
    }
  }
  return output;
}

function splitHeads(
  input: Float32Array,
  tokens: number,
  heads: number,
  headDimension: number,
): Float32Array {
  const output = new Float32Array(input.length);
  for (let token = 0; token < tokens; token += 1) {
    for (let head = 0; head < heads; head += 1) {
      for (let dimension = 0; dimension < headDimension; dimension += 1) {
        output[(head * tokens + token) * headDimension + dimension] =
          input[(token * heads + head) * headDimension + dimension]!;
      }
    }
  }
  return output;
}

function mergeHeads(
  input: Float32Array,
  tokens: number,
  heads: number,
  headDimension: number,
): Float32Array {
  const output = new Float32Array(input.length);
  for (let token = 0; token < tokens; token += 1) {
    for (let head = 0; head < heads; head += 1) {
      for (let dimension = 0; dimension < headDimension; dimension += 1) {
        output[(token * heads + head) * headDimension + dimension] =
          input[(head * tokens + token) * headDimension + dimension]!;
      }
    }
  }
  return output;
}

function cpuAdd(
  profile: AceModelProfileId,
  left: Float32Array,
  right: Float32Array,
): Float32Array {
  return Float32Array.from(left, (value, index) =>
    store(profile, value + right[index]!),
  );
}

function layerWeights(seed: number): LayerWeights {
  return Object.freeze({
    inputLayerNorm: vector(HIDDEN, seed),
    queryProjection: matrix(QUERY_HEADS * HEAD_DIMENSION, HIDDEN, seed + 2),
    keyProjection: matrix(KEY_VALUE_HEADS * HEAD_DIMENSION, HIDDEN, seed + 4),
    valueProjection: matrix(KEY_VALUE_HEADS * HEAD_DIMENSION, HIDDEN, seed + 6),
    queryNorm: vector(HEAD_DIMENSION, seed + 8),
    keyNorm: vector(HEAD_DIMENSION, seed + 10),
    outputProjection: matrix(HIDDEN, QUERY_HEADS * HEAD_DIMENSION, seed + 12),
    postAttentionLayerNorm: vector(HIDDEN, seed + 14),
    gateProjection: matrix(INTERMEDIATE, HIDDEN, seed + 16),
    upProjection: matrix(INTERMEDIATE, HIDDEN, seed + 18),
    downProjection: matrix(HIDDEN, INTERMEDIATE, seed + 20),
  });
}

function vector(length: number, seed: number): Float32Array {
  return Float32Array.from({ length }, (_, index) =>
    Math.fround(0.75 + ((index * seed + seed) % 9) / 20),
  );
}

function matrix(rows: number, columns: number, seed: number): Float32Array {
  return Float32Array.from({ length: rows * columns }, (_, index) =>
    Math.fround((((index * seed + seed * 3) % 19) - 9) / 24),
  );
}

function quantizedModelWeights(
  profile: AceModelProfileId,
  weights: ModelWeights,
): ModelWeights {
  const quantize = (values: Float32Array): Float32Array =>
    Float32Array.from(values, (value) => weightValue(profile, value));
  const quantizeLayer = (layer: LayerWeights): LayerWeights => ({
    inputLayerNorm: quantize(layer.inputLayerNorm),
    queryProjection: quantize(layer.queryProjection),
    keyProjection: quantize(layer.keyProjection),
    valueProjection: quantize(layer.valueProjection),
    queryNorm: quantize(layer.queryNorm),
    keyNorm: quantize(layer.keyNorm),
    outputProjection: quantize(layer.outputProjection),
    postAttentionLayerNorm: quantize(layer.postAttentionLayerNorm),
    gateProjection: quantize(layer.gateProjection),
    upProjection: quantize(layer.upProjection),
    downProjection: quantize(layer.downProjection),
  });
  return {
    embedding: quantize(weights.embedding),
    layers: weights.layers.map(quantizeLayer),
    finalNorm: quantize(weights.finalNorm),
  };
}

function gpuLayerWeights(
  device: GPUDevice,
  buffers: GPUBuffer[],
  profile: AceModelProfileId,
  weights: LayerWeights,
): AceQwen3BlockWeights {
  return {
    inputLayerNorm: weightBinding(device, buffers, profile, weights.inputLayerNorm),
    queryProjection: weightBinding(device, buffers, profile, weights.queryProjection),
    keyProjection: weightBinding(device, buffers, profile, weights.keyProjection),
    valueProjection: weightBinding(device, buffers, profile, weights.valueProjection),
    queryNorm: weightBinding(device, buffers, profile, weights.queryNorm),
    keyNorm: weightBinding(device, buffers, profile, weights.keyNorm),
    outputProjection: weightBinding(device, buffers, profile, weights.outputProjection),
    postAttentionLayerNorm: weightBinding(
      device,
      buffers,
      profile,
      weights.postAttentionLayerNorm,
    ),
    gateProjection: weightBinding(device, buffers, profile, weights.gateProjection),
    upProjection: weightBinding(device, buffers, profile, weights.upProjection),
    downProjection: weightBinding(device, buffers, profile, weights.downProjection),
  };
}

function allocateScratch(
  device: GPUDevice,
  buffers: GPUBuffer[],
  profile: AceModelProfileId,
  plan: ReturnType<typeof planAceQwen3Block>,
): AceQwen3BlockScratch {
  const create = (elements: number): GPUBufferBinding =>
    binding(track(buffers, activationBuffer(device, profile, elements)));
  return {
    normalizedInput: create(plan.hiddenElements),
    queryFlat: create(plan.queryElements),
    keyFlat: create(plan.keyValueElements),
    valueFlat: create(plan.keyValueElements),
    queryHeads: create(plan.queryElements),
    keyHeads: create(plan.keyValueElements),
    valueHeads: create(plan.keyValueElements),
    normalizedQueryHeads: create(plan.queryElements),
    normalizedKeyHeads: create(plan.keyValueElements),
    rotatedQueryHeads: create(plan.queryElements),
    rotatedKeyHeads: create(plan.keyValueElements),
    attentionHeads: create(plan.queryElements),
    mergedAttention: create(plan.queryElements),
    projectedAttention: create(plan.hiddenElements),
    afterAttention: create(plan.hiddenElements),
    normalizedAfterAttention: create(plan.hiddenElements),
    gate: create(plan.intermediateElements),
    up: create(plan.intermediateElements),
    gatedActivation: create(plan.intermediateElements),
    projectedMlp: create(plan.hiddenElements),
  };
}

/**
 * Independent Transformers 4.57.6 / PyTorch FP32 Qwen3RotaryEmbedding words
 * for headDimension=2, theta=100, positions 0..2. CPU model math consumes
 * these literals rather than the browser table builder feeding the GPU.
 */
const TRANSFORMERS_COSINE_WORDS = Object.freeze([
  0x3f800000, 0x3f800000,
  0x3f0a5141, 0x3f0a5141,
  0xbed51133, 0xbed51133,
]);
const TRANSFORMERS_SINE_WORDS = Object.freeze([
  0x00000000, 0x00000000,
  0x3f576aa4, 0x3f576aa4,
  0x3f68c7b7, 0x3f68c7b7,
]);

function tinyTransformersRopeOracle(): Readonly<{
  cosine: Float32Array;
  sine: Float32Array;
}> {
  return {
    cosine: wordsToFloat32(TRANSFORMERS_COSINE_WORDS),
    sine: wordsToFloat32(TRANSFORMERS_SINE_WORDS),
  };
}

function validateRopeTables(
  cosine: Float32Array,
  sine: Float32Array,
): number {
  const expected = tinyTransformersRopeOracle();
  return Math.max(
    maximumWordUlp(cosine, expected.cosine),
    maximumWordUlp(sine, expected.sine),
  );
}

function wordsToFloat32(words: readonly number[]): Float32Array {
  const buffer = new ArrayBuffer(words.length * 4);
  new Uint32Array(buffer).set(words);
  return new Float32Array(buffer);
}

function maximumWordUlp(actual: Float32Array, expected: Float32Array): number {
  if (actual.length !== expected.length) throw new Error("RoPE oracle length mismatch");
  const actualWords = new Uint32Array(actual.buffer, actual.byteOffset, actual.length);
  const expectedWords = new Uint32Array(expected.buffer, expected.byteOffset, expected.length);
  let maximum = 0;
  for (let index = 0; index < actualWords.length; index += 1) {
    const difference = Math.abs(actualWords[index]! - expectedWords[index]!);
    maximum = Math.max(maximum, difference);
    if (difference > 1) {
      throw new Error(`RoPE differs from Transformers by ${difference} ULP at ${index}`);
    }
  }
  return maximum;
}

function weightBinding(
  device: GPUDevice,
  buffers: GPUBuffer[],
  profile: AceModelProfileId,
  values: Float32Array,
): GPUBufferBinding {
  const data = profile === "reference-bf16" ? packBf16(values) : packFp16(values);
  return binding(track(buffers, storageBuffer(device, data)));
}

function activationBuffer(
  device: GPUDevice,
  profile: AceModelProfileId,
  elements: number,
  values?: Float32Array,
  copySource = false,
): GPUBuffer {
  const data = values === undefined
    ? new Uint8Array(elements * (profile === "raw-fp16" ? 2 : 4))
    : profile === "raw-fp16"
      ? packFp16(values)
      : values;
  return storageBuffer(device, data, copySource);
}

function storageBuffer(
  device: GPUDevice,
  data: ArrayBuffer | ArrayBufferView<ArrayBufferLike>,
  copySource = false,
): GPUBuffer {
  const byteLength = "byteLength" in data ? data.byteLength : 0;
  const padded = new Uint8Array(Math.max(4, Math.ceil(byteLength / 4) * 4));
  const bytes = ArrayBuffer.isView(data)
    ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
    : new Uint8Array(data);
  padded.set(bytes);
  const buffer = device.createBuffer({
    size: padded.byteLength,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_DST |
      (copySource ? GPUBufferUsage.COPY_SRC : 0),
  });
  device.queue.writeBuffer(buffer, 0, padded);
  return buffer;
}

function binding(buffer: GPUBuffer): GPUBufferBinding {
  return { buffer, offset: 0, size: buffer.size };
}

function track(buffers: GPUBuffer[], buffer: GPUBuffer): GPUBuffer {
  buffers.push(buffer);
  return buffer;
}

async function execute(
  device: GPUDevice,
  dispatch: { encode(pass: GPUComputePassEncoder): void },
): Promise<void> {
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  dispatch.encode(pass);
  pass.end();
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
}

async function readProfile(
  device: GPUDevice,
  profile: AceModelProfileId,
  source: GPUBuffer,
  count: number,
): Promise<Float32Array> {
  if (profile === "reference-bf16") {
    return new Float32Array(await readBytes(device, source, count * 4));
  }
  const words = new Uint16Array(await readBytes(device, source, count * 2), 0, count);
  return Float32Array.from(words, fp16BitsToNumber);
}

async function readBytes(
  device: GPUDevice,
  source: GPUBuffer,
  bytes: number,
): Promise<ArrayBuffer> {
  const paddedBytes = Math.max(4, Math.ceil(bytes / 4) * 4);
  const readback = device.createBuffer({
    size: paddedBytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  try {
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(source, 0, readback, 0, paddedBytes);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    await readback.mapAsync(GPUMapMode.READ);
    return readback.getMappedRange(0, paddedBytes).slice(0, bytes);
  } finally {
    readback.destroy();
  }
}

function assertClose(
  actual: Float32Array,
  expected: Float32Array,
  tolerance: number,
  label: string,
): number {
  if (actual.length !== expected.length) throw new Error(`${label} length mismatch`);
  let maximumError = 0;
  for (let index = 0; index < actual.length; index += 1) {
    const error = Math.abs(actual[index]! - expected[index]!);
    maximumError = Math.max(maximumError, error);
    if (!Number.isFinite(error) || error > tolerance) {
      throw new Error(
        `${label} mismatch at ${index}: received ${actual[index]}, ` +
          `expected ${expected[index]}, error ${error}`,
      );
    }
  }
  return maximumError;
}

function store(profile: AceModelProfileId, value: number): number {
  return profile === "raw-fp16"
    ? fp16BitsToNumber(numberToFp16Bits(value))
    : Math.fround(value);
}

function weightValue(profile: AceModelProfileId, value: number): number {
  return profile === "raw-fp16"
    ? fp16BitsToNumber(numberToFp16Bits(value))
    : bf16BitsToNumber(numberToBf16Bits(value));
}

function packFp16(values: ArrayLike<number>): Uint16Array<ArrayBuffer> {
  return Uint16Array.from(values, numberToFp16Bits);
}

function packBf16(values: ArrayLike<number>): Uint32Array<ArrayBuffer> {
  const result = new Uint32Array(Math.ceil(values.length / 2));
  for (let index = 0; index < values.length; index += 1) {
    const bits = numberToBf16Bits(values[index]!);
    const pair = index >> 1;
    result[pair] = index % 2 === 0
      ? (result[pair]! & 0xffff_0000) | bits
      : (result[pair]! & 0x0000_ffff) | (bits << 16);
  }
  return result;
}

const FLOAT_BITS_BUFFER = new ArrayBuffer(4);
const FLOAT_BITS_F32 = new Float32Array(FLOAT_BITS_BUFFER);
const FLOAT_BITS_U32 = new Uint32Array(FLOAT_BITS_BUFFER);

function numberToBf16Bits(value: number): number {
  FLOAT_BITS_F32[0] = value;
  const bits = FLOAT_BITS_U32[0]!;
  const rounded = (bits + 0x7fff + ((bits >>> 16) & 1)) >>> 0;
  return rounded >>> 16;
}

function bf16BitsToNumber(bits: number): number {
  FLOAT_BITS_U32[0] = bits << 16;
  return FLOAT_BITS_F32[0]!;
}

function numberToFp16Bits(value: number): number {
  FLOAT_BITS_F32[0] = value;
  const bits = FLOAT_BITS_U32[0]!;
  const sign = (bits >>> 16) & 0x8000;
  const exponent = (bits >>> 23) & 0xff;
  const mantissa = bits & 0x7f_ffff;
  if (exponent === 0xff) return sign | (mantissa === 0 ? 0x7c00 : 0x7e00);
  const halfExponent = exponent - 127 + 15;
  if (halfExponent >= 0x1f) return sign | 0x7c00;
  if (halfExponent <= 0) {
    if (halfExponent < -10) return sign;
    const normalized = mantissa | 0x80_0000;
    const shift = 14 - halfExponent;
    const truncated = normalized >>> shift;
    const remainder = normalized & ((1 << shift) - 1);
    const halfway = 1 << (shift - 1);
    return sign | (
      truncated +
      (remainder > halfway || (remainder === halfway && (truncated & 1)) ? 1 : 0)
    );
  }
  let halfMantissa = mantissa >>> 13;
  const remainder = mantissa & 0x1fff;
  if (remainder > 0x1000 || (remainder === 0x1000 && (halfMantissa & 1))) {
    halfMantissa += 1;
    if (halfMantissa === 0x400) {
      const nextExponent = halfExponent + 1;
      return sign | (nextExponent >= 0x1f ? 0x7c00 : nextExponent << 10);
    }
  }
  return sign | (halfExponent << 10) | halfMantissa;
}

function fp16BitsToNumber(bits: number): number {
  const sign = (bits & 0x8000) === 0 ? 1 : -1;
  const exponent = (bits >>> 10) & 0x1f;
  const fraction = bits & 0x3ff;
  if (exponent === 0) return sign * 2 ** -14 * (fraction / 1024);
  if (exponent === 0x1f) return fraction === 0 ? sign * Infinity : Number.NaN;
  return sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}

function finish(status: "passed" | "failed", message: string): void {
  document.body.dataset.status = status;
  resultNode.textContent = message;
  document.title = `ACE text encoder ${status}`;
}

function requireResultNode(): HTMLPreElement {
  const node = document.querySelector<HTMLPreElement>("#result");
  if (node === null) throw new Error("Missing result element");
  return node;
}
