import type { AceModelProfileId } from "../../src/webgpu/capabilities.js";
import {
  AceCorrectnessQwen3Runtime,
  createAceQwen3CausalControlData,
  createAceQwen3RopeTables,
  planAceQwen3Block,
  type AceQwen3BlockScratch,
  type AceQwen3Config,
} from "../../src/webgpu/qwen3.js";

const BATCH = 2;
const HIDDEN = 4;
const INTERMEDIATE = 6;
const QUERY_HEADS = 2;
const KV_HEADS = 1;
const HEAD_DIMENSION = 2;
const CACHE_CAPACITY = 4;

const CONFIG: AceQwen3Config = Object.freeze({
  id: "ace-browser-tiny-qwen3",
  hiddenSize: HIDDEN,
  intermediateSize: INTERMEDIATE,
  layerCount: 1,
  queryHeads: QUERY_HEADS,
  keyValueHeads: KV_HEADS,
  headDimension: HEAD_DIMENSION,
  vocabularySize: 7,
  maximumPositionEmbeddings: 8,
  ropeTheta: 100,
  rmsNormEpsilon: 1e-5,
  attentionBias: false,
  hiddenActivation: "silu",
  tieWordEmbeddings: true,
});

interface FloatWeights {
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
  readonly finalNorm: Float32Array;
  readonly embedding: Float32Array;
}

interface CpuCache {
  readonly key: Float32Array;
  readonly value: Float32Array;
  readonly validity: Uint32Array;
}

interface ProfileResult {
  readonly profile: AceModelProfileId;
  readonly ropeTableMaximumUlp: number;
  readonly prefillMaximumError: number;
  readonly decodeMaximumError: number;
  readonly logitsMaximumError: number;
  readonly prefillStatus: readonly number[];
  readonly decodeStatus: readonly number[];
}

const BASE_WEIGHTS: FloatWeights = Object.freeze({
  inputLayerNorm: new Float32Array([0.9, 1.1, 0.8, 1.05]),
  queryProjection: matrix(QUERY_HEADS * HEAD_DIMENSION, HIDDEN, 3),
  keyProjection: matrix(KV_HEADS * HEAD_DIMENSION, HIDDEN, 5),
  valueProjection: matrix(KV_HEADS * HEAD_DIMENSION, HIDDEN, 7),
  queryNorm: new Float32Array([1.05, 0.85]),
  keyNorm: new Float32Array([0.95, 1.1]),
  outputProjection: matrix(HIDDEN, QUERY_HEADS * HEAD_DIMENSION, 11),
  postAttentionLayerNorm: new Float32Array([1, 0.8, 1.15, 0.9]),
  gateProjection: matrix(INTERMEDIATE, HIDDEN, 13),
  upProjection: matrix(INTERMEDIATE, HIDDEN, 17),
  downProjection: matrix(HIDDEN, INTERMEDIATE, 19),
  finalNorm: new Float32Array([0.95, 1.05, 0.9, 1.1]),
  embedding: matrix(7, HIDDEN, 23),
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
  const runtime = AceCorrectnessQwen3Runtime.create(device, profile);
  const buffers: GPUBuffer[] = [];
  const ropeTableMaximumUlp = validateTinyRopeTablesAgainstPinnedOracle();
  const weights = quantizedWeights(profile, BASE_WEIGHTS);
  const gpuWeights = {
    inputLayerNorm: weightBinding(device, buffers, profile, BASE_WEIGHTS.inputLayerNorm),
    queryProjection: weightBinding(device, buffers, profile, BASE_WEIGHTS.queryProjection),
    keyProjection: weightBinding(device, buffers, profile, BASE_WEIGHTS.keyProjection),
    valueProjection: weightBinding(device, buffers, profile, BASE_WEIGHTS.valueProjection),
    queryNorm: weightBinding(device, buffers, profile, BASE_WEIGHTS.queryNorm),
    keyNorm: weightBinding(device, buffers, profile, BASE_WEIGHTS.keyNorm),
    outputProjection: weightBinding(device, buffers, profile, BASE_WEIGHTS.outputProjection),
    postAttentionLayerNorm: weightBinding(
      device,
      buffers,
      profile,
      BASE_WEIGHTS.postAttentionLayerNorm,
    ),
    gateProjection: weightBinding(device, buffers, profile, BASE_WEIGHTS.gateProjection),
    upProjection: weightBinding(device, buffers, profile, BASE_WEIGHTS.upProjection),
    downProjection: weightBinding(device, buffers, profile, BASE_WEIGHTS.downProjection),
  };
  const cache: CpuCache = {
    key: new Float32Array(BATCH * KV_HEADS * CACHE_CAPACITY * HEAD_DIMENSION),
    value: new Float32Array(BATCH * KV_HEADS * CACHE_CAPACITY * HEAD_DIMENSION),
    validity: new Uint32Array(BATCH * CACHE_CAPACITY),
  };
  const cacheKey = track(
    buffers,
    activationBuffer(device, profile, cache.key.length, undefined, true),
  );
  const cacheValue = track(
    buffers,
    activationBuffer(device, profile, cache.value.length, undefined, true),
  );
  const cacheValidity = track(
    buffers,
    storageBuffer(device, cache.validity, true),
  );
  const writeStatus = track(
    buffers,
    storageBuffer(device, new Uint32Array(BATCH), true),
  );

  try {
    const prefillInput = new Float32Array([
      0.2, -0.1, 0.3, 0.05,
      0.4, 0.15, -0.2, 0.1,
      -0.3, 0.25, 0.15, -0.05,
      0.1, 0.35, -0.15, 0.2,
      -0.25, 0.05, 0.4, 0.1,
      0.3, -0.2, 0.05, 0.25,
    ]);
    const prefillControls = createAceQwen3CausalControlData({
      batch: BATCH,
      tokens: 3,
      cacheCapacity: CACHE_CAPACITY,
      rowStartPositions: [0, 0],
      validKeyLengths: [3, 3],
      sourceValidity: [0, 1, 1, 1, 1, 1],
    });
    const expectedPrefill = cpuBlock(
      profile,
      weights,
      prefillInput,
      3,
      prefillControls,
      [0, 1, 2, 0, 1, 2],
      cache,
    );
    const prefill = await createGpuBlock(
      runtime,
      device,
      buffers,
      profile,
      "ace-browser-qwen3-prefill",
      gpuWeights,
      prefillInput,
      3,
      prefillControls,
      [0, 1, 2, 0, 1, 2],
      cacheKey,
      cacheValue,
      cacheValidity,
      writeStatus,
    );
    await execute(device, prefill.dispatch);
    const actualPrefill = await readProfile(
      device,
      profile,
      prefill.output,
      expectedPrefill.length,
    );
    const prefillMaximumError = assertClose(
      actualPrefill,
      expectedPrefill,
      profile === "raw-fp16" ? 1.5e-2 : 3e-5,
      `${profile} Qwen3 prefill`,
    );
    const prefillStatus = await readU32(device, writeStatus, BATCH);
    assertExact(prefillStatus, new Uint32Array([1, 1]), `${profile} prefill status`);

    const decodeInput = new Float32Array([
      0.15, -0.35, 0.2, 0.1,
      -0.1, 0.3, 0.25, -0.2,
    ]);
    const decodeControls = createAceQwen3CausalControlData({
      batch: BATCH,
      tokens: 1,
      cacheCapacity: CACHE_CAPACITY,
      rowStartPositions: [3, 3],
      validKeyLengths: [4, 4],
      sourceValidity: [1, 1],
    });
    const expectedDecode = cpuBlock(
      profile,
      weights,
      decodeInput,
      1,
      decodeControls,
      [3, 3],
      cache,
    );
    const decode = await createGpuBlock(
      runtime,
      device,
      buffers,
      profile,
      "ace-browser-qwen3-decode",
      gpuWeights,
      decodeInput,
      1,
      decodeControls,
      [3, 3],
      cacheKey,
      cacheValue,
      cacheValidity,
      writeStatus,
    );
    await execute(device, decode.dispatch);
    const actualDecode = await readProfile(
      device,
      profile,
      decode.output,
      expectedDecode.length,
    );
    const decodeMaximumError = assertClose(
      actualDecode,
      expectedDecode,
      profile === "raw-fp16" ? 1.5e-2 : 3e-5,
      `${profile} Qwen3 decode`,
    );
    const decodeStatus = await readU32(device, writeStatus, BATCH);
    assertExact(decodeStatus, new Uint32Array([1, 1]), `${profile} decode status`);

    // Product sampling projects one normalized row through the exact same
    // row shards used by embedding lookup. Select row 1 to cover a nonzero
    // activation-buffer offset without assembling a duplicate LM-head weight.
    const selectedDecodeRow = track(
      buffers,
      activationBuffer(
        device,
        profile,
        HIDDEN,
        actualDecode.slice(HIDDEN, HIDDEN * 2),
      ),
    );
    const normalizedRow = track(
      buffers,
      activationBuffer(device, profile, HIDDEN, undefined, true),
    );
    const finalNormWeight = weightBinding(
      device,
      buffers,
      profile,
      BASE_WEIGHTS.finalNorm,
    );
    const finalNorm = await runtime.createFinalNormDispatch(
      `ace-browser-${profile}-qwen3-final-norm`,
      CONFIG,
      1,
      {
        input: binding(selectedDecodeRow),
        weight: finalNormWeight,
        output: binding(normalizedRow),
      },
    );
    const embeddingShard0 = BASE_WEIGHTS.embedding.slice(0, 4 * HIDDEN);
    const embeddingShard1 = BASE_WEIGHTS.embedding.slice(4 * HIDDEN);
    const logits0 = track(buffers, activationBuffer(device, profile, 4, undefined, true));
    const logits1 = track(buffers, activationBuffer(device, profile, 3, undefined, true));
    const tiedOutput = await runtime.createTiedOutputDispatch(
      `ace-browser-${profile}-qwen3-tied-output`,
      CONFIG,
      binding(normalizedRow),
      [
        {
          firstRow: 0,
          rowCount: 4,
          weight: weightBinding(device, buffers, profile, embeddingShard0),
          logits: binding(logits0),
        },
        {
          firstRow: 4,
          rowCount: 3,
          weight: weightBinding(device, buffers, profile, embeddingShard1),
          logits: binding(logits1),
        },
      ],
    );
    await execute(device, sequence(finalNorm, tiedOutput));
    const actualLogits = new Float32Array(7);
    actualLogits.set(await readProfile(device, profile, logits0, 4), 0);
    actualLogits.set(await readProfile(device, profile, logits1, 3), 4);
    const expectedNormalized = cpuRmsNorm(
      profile,
      expectedDecode.slice(HIDDEN, HIDDEN * 2),
      weights.finalNorm,
      HIDDEN,
      CONFIG.rmsNormEpsilon,
    );
    const expectedLogits = cpuGemm(
      profile,
      expectedNormalized,
      weights.embedding,
      1,
      HIDDEN,
      7,
    );
    const logitsMaximumError = assertClose(
      actualLogits,
      expectedLogits,
      profile === "raw-fp16" ? 8e-3 : 2e-5,
      `${profile} tied logits`,
    );

    return {
      profile,
      ropeTableMaximumUlp,
      prefillMaximumError,
      decodeMaximumError,
      logitsMaximumError,
      prefillStatus: [...prefillStatus],
      decodeStatus: [...decodeStatus],
    };
  } finally {
    runtime.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
}

async function createGpuBlock(
  runtime: AceCorrectnessQwen3Runtime,
  device: GPUDevice,
  buffers: GPUBuffer[],
  profile: AceModelProfileId,
  label: string,
  weights: Parameters<AceCorrectnessQwen3Runtime["createBlockDispatch"]>[3]["weights"],
  input: Float32Array,
  tokens: number,
  controls: ReturnType<typeof createAceQwen3CausalControlData>,
  ropePositionIds: readonly number[],
  cacheKey: GPUBuffer,
  cacheValue: GPUBuffer,
  cacheValidity: GPUBuffer,
  writeStatus: GPUBuffer,
) {
  const plan = planAceQwen3Block(CONFIG, {
    batch: BATCH,
    tokens,
    attention: { kind: "cached", cacheCapacity: CACHE_CAPACITY },
  });
  const inputBuffer = track(
    buffers,
    activationBuffer(device, profile, input.length, input),
  );
  const output = track(
    buffers,
    activationBuffer(device, profile, plan.hiddenElements, undefined, true),
  );
  const scratch = allocateScratch(device, buffers, profile, plan);
  const rope = createAceQwen3RopeTables(ropePositionIds, {
    ...CONFIG,
    batch: BATCH,
    tokens,
  });
  const dispatch = await runtime.createBlockDispatch(
    label,
    CONFIG,
    {
      batch: BATCH,
      tokens,
      attention: { kind: "cached", cacheCapacity: CACHE_CAPACITY },
    },
    {
      input: binding(inputBuffer),
      output: binding(output),
      weights,
      scratch,
      attention: {
        kind: "cached",
        validLengths: binding(track(buffers, storageBuffer(device, controls.validLengths))),
        queryPositions: binding(track(buffers, storageBuffer(device, controls.queryPositions))),
        cosine: binding(track(buffers, storageBuffer(device, rope.cosine))),
        sine: binding(track(buffers, storageBuffer(device, rope.sine))),
        sourceValidity: binding(
          track(buffers, storageBuffer(device, controls.sourceValidity)),
        ),
        cacheKey: binding(cacheKey),
        cacheValue: binding(cacheValue),
        cacheValidity: binding(cacheValidity),
        rowStartPositions: binding(
          track(buffers, storageBuffer(device, controls.rowStartPositions)),
        ),
        writeStatus: binding(writeStatus),
      },
    },
  );
  return { dispatch, output };
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
 * Independent Transformers 4.57.6 / PyTorch 2.10.0 CPU FP32 oracle for the
 * miniature `headDimension=2, theta=100` graph at positions 0..3. These words
 * were captured from `Qwen3RotaryEmbedding`; cosine followed by sine hashes to
 * `39deec58dc2103b40d4a1de1dd8f1094e3409befe6d913858dd5c18b9b83f690`.
 * The CPU block deliberately consumes these literals, never the production
 * table builder used to feed the GPU.
 */
const TINY_TRANSFORMERS_COSINE_WORDS = Object.freeze([
  0x3f800000, 0x3f800000,
  0x3f0a5141, 0x3f0a5141,
  0xbed51133, 0xbed51133,
  0xbf7d7026, 0xbf7d7026,
]);
const TINY_TRANSFORMERS_SINE_WORDS = Object.freeze([
  0x00000000, 0x00000000,
  0x3f576aa4, 0x3f576aa4,
  0x3f68c7b7, 0x3f68c7b7,
  0x3e1081c3, 0x3e1081c3,
]);

function tinyTransformersRopeOracle(
  positionIds: readonly number[],
): ReturnType<typeof createAceQwen3RopeTables> {
  const cosine = new Float32Array(positionIds.length * HEAD_DIMENSION);
  const sine = new Float32Array(cosine.length);
  const cosineWords = new Uint32Array(cosine.buffer);
  const sineWords = new Uint32Array(sine.buffer);
  for (let token = 0; token < positionIds.length; token += 1) {
    const position = positionIds[token]!;
    if (!Number.isSafeInteger(position) || position < 0 || position > 3) {
      throw new RangeError(`Tiny Qwen3 oracle has no position ${position}`);
    }
    for (let dimension = 0; dimension < HEAD_DIMENSION; dimension += 1) {
      const source = position * HEAD_DIMENSION + dimension;
      const destination = token * HEAD_DIMENSION + dimension;
      cosineWords[destination] = TINY_TRANSFORMERS_COSINE_WORDS[source]!;
      sineWords[destination] = TINY_TRANSFORMERS_SINE_WORDS[source]!;
    }
  }
  return { cosine, sine };
}

function validateTinyRopeTablesAgainstPinnedOracle(): number {
  const positions = [0, 1, 2, 3] as const;
  const actual = createAceQwen3RopeTables(positions, {
    batch: 1,
    tokens: positions.length,
    headDimension: HEAD_DIMENSION,
    ropeTheta: CONFIG.ropeTheta,
    maximumPositionEmbeddings: CONFIG.maximumPositionEmbeddings,
  });
  const expected = tinyTransformersRopeOracle(positions);
  return Math.max(
    maximumWordUlp(actual.cosine, expected.cosine),
    maximumWordUlp(actual.sine, expected.sine),
  );
}

function maximumWordUlp(actual: Float32Array, expected: Float32Array): number {
  if (actual.length !== expected.length) throw new Error("RoPE oracle length mismatch");
  const actualWords = new Uint32Array(
    actual.buffer,
    actual.byteOffset,
    actual.length,
  );
  const expectedWords = new Uint32Array(
    expected.buffer,
    expected.byteOffset,
    expected.length,
  );
  let maximum = 0;
  for (let index = 0; index < actualWords.length; index += 1) {
    const difference = Math.abs(actualWords[index]! - expectedWords[index]!);
    maximum = Math.max(maximum, difference);
    if (difference > 1) {
      throw new Error(
        `Qwen3 RoPE table differs from Transformers by ${difference} ULP at ${index}`,
      );
    }
  }
  return maximum;
}

function cpuBlock(
  profile: AceModelProfileId,
  weights: FloatWeights,
  sourceInput: Float32Array,
  tokens: number,
  controls: ReturnType<typeof createAceQwen3CausalControlData>,
  ropePositionIds: readonly number[],
  cache: CpuCache,
): Float32Array {
  const input = Float32Array.from(sourceInput, (value) => store(profile, value));
  const rows = BATCH * tokens;
  const norm = cpuRmsNorm(
    profile,
    input,
    weights.inputLayerNorm,
    HIDDEN,
    CONFIG.rmsNormEpsilon,
  );
  const queryFlat = cpuGemm(
    profile,
    norm,
    weights.queryProjection,
    rows,
    HIDDEN,
    QUERY_HEADS * HEAD_DIMENSION,
  );
  const keyFlat = cpuGemm(
    profile,
    norm,
    weights.keyProjection,
    rows,
    HIDDEN,
    KV_HEADS * HEAD_DIMENSION,
  );
  const valueFlat = cpuGemm(
    profile,
    norm,
    weights.valueProjection,
    rows,
    HIDDEN,
    KV_HEADS * HEAD_DIMENSION,
  );
  const queryHeads = splitHeads(queryFlat, BATCH, tokens, QUERY_HEADS, HEAD_DIMENSION);
  const keyHeads = splitHeads(keyFlat, BATCH, tokens, KV_HEADS, HEAD_DIMENSION);
  const valueHeads = splitHeads(valueFlat, BATCH, tokens, KV_HEADS, HEAD_DIMENSION);
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
  const rope = tinyTransformersRopeOracle(ropePositionIds);
  const query = cpuRope(profile, normalizedQuery, rope, BATCH, QUERY_HEADS, tokens);
  const key = cpuRope(profile, normalizedKey, rope, BATCH, KV_HEADS, tokens);
  cpuCacheAppend(cache, key, valueHeads, controls, tokens);
  const attention = cpuAttention(profile, query, cache, controls, tokens);
  const merged = mergeHeads(attention, BATCH, tokens, QUERY_HEADS, HEAD_DIMENSION);
  const projectedAttention = cpuGemm(
    profile,
    merged,
    weights.outputProjection,
    rows,
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
    rows,
    HIDDEN,
    INTERMEDIATE,
  );
  const up = cpuGemm(
    profile,
    postNorm,
    weights.upProjection,
    rows,
    HIDDEN,
    INTERMEDIATE,
  );
  const activated = new Float32Array(gate.length);
  for (let index = 0; index < gate.length; index += 1) {
    const silu = store(profile, gate[index]! / (1 + Math.exp(-gate[index]!)));
    activated[index] = store(profile, silu * up[index]!);
  }
  const mlp = cpuGemm(
    profile,
    activated,
    weights.downProjection,
    rows,
    INTERMEDIATE,
    HIDDEN,
  );
  return cpuAdd(profile, afterAttention, mlp);
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
      let sum = store(profile, 0);
      for (let k = 0; k < inner; k += 1) {
        const product = store(
          profile,
          activation[row * inner + k]! * weight[column * inner + k]!,
        );
        sum = store(profile, sum + product);
      }
      output[row * columns + column] = sum;
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
  tables: ReturnType<typeof createAceQwen3RopeTables>,
  batch: number,
  heads: number,
  tokens: number,
): Float32Array {
  const output = new Float32Array(input.length);
  const half = HEAD_DIMENSION / 2;
  for (let b = 0; b < batch; b += 1) {
    for (let head = 0; head < heads; head += 1) {
      for (let token = 0; token < tokens; token += 1) {
        for (let dimension = 0; dimension < HEAD_DIMENSION; dimension += 1) {
          const index = ((b * heads + head) * tokens + token) * HEAD_DIMENSION + dimension;
          const rotatedDimension = dimension < half ? dimension + half : dimension - half;
          const rotatedIndex = index - dimension + rotatedDimension;
          const tableIndex =
            (b * tokens + token) * HEAD_DIMENSION + dimension;
          const sign = dimension < half ? -1 : 1;
          const value = Math.fround(
            Math.fround(input[index]! * tables.cosine[tableIndex]!) +
              Math.fround(sign * input[rotatedIndex]! * tables.sine[tableIndex]!),
          );
          output[index] = store(profile, value);
        }
      }
    }
  }
  return output;
}

function cpuCacheAppend(
  cache: CpuCache,
  key: Float32Array,
  value: Float32Array,
  controls: ReturnType<typeof createAceQwen3CausalControlData>,
  tokens: number,
): void {
  for (let batch = 0; batch < BATCH; batch += 1) {
    const start = controls.rowStartPositions[batch]!;
    for (let head = 0; head < KV_HEADS; head += 1) {
      for (let token = 0; token < tokens; token += 1) {
        for (let dimension = 0; dimension < HEAD_DIMENSION; dimension += 1) {
          const source = ((batch * KV_HEADS + head) * tokens + token) *
            HEAD_DIMENSION + dimension;
          const destination =
            ((batch * KV_HEADS + head) * CACHE_CAPACITY + start + token) *
              HEAD_DIMENSION + dimension;
          cache.key[destination] = key[source]!;
          cache.value[destination] = value[source]!;
        }
      }
    }
    for (let token = 0; token < tokens; token += 1) {
      cache.validity[batch * CACHE_CAPACITY + start + token] =
        controls.sourceValidity[batch * tokens + token]!;
    }
  }
}

function cpuAttention(
  profile: AceModelProfileId,
  query: Float32Array,
  cache: CpuCache,
  controls: ReturnType<typeof createAceQwen3CausalControlData>,
  tokens: number,
): Float32Array {
  const output = new Float32Array(BATCH * QUERY_HEADS * tokens * HEAD_DIMENSION);
  for (let batch = 0; batch < BATCH; batch += 1) {
    for (let queryHead = 0; queryHead < QUERY_HEADS; queryHead += 1) {
      const kvHead = Math.floor(queryHead / (QUERY_HEADS / KV_HEADS));
      for (let queryToken = 0; queryToken < tokens; queryToken += 1) {
        const physical = controls.queryPositions[batch * tokens + queryToken]!;
        const keyEnd = Math.min(
          controls.validLengths[batch * 2 + 1]!,
          physical + 1,
        );
        const keys: number[] = [];
        const scores: number[] = [];
        for (let keyToken = 0; keyToken < keyEnd; keyToken += 1) {
          if (cache.validity[batch * CACHE_CAPACITY + keyToken] !== 1) continue;
          let score = Math.fround(0);
          for (let dimension = 0; dimension < HEAD_DIMENSION; dimension += 1) {
            const queryIndex =
              ((batch * QUERY_HEADS + queryHead) * tokens + queryToken) *
                HEAD_DIMENSION + dimension;
            const keyIndex =
              ((batch * KV_HEADS + kvHead) * CACHE_CAPACITY + keyToken) *
                HEAD_DIMENSION + dimension;
            score = Math.fround(
              score + Math.fround(query[queryIndex]! * cache.key[keyIndex]!),
            );
          }
          keys.push(keyToken);
          scores.push(Math.fround(score / Math.sqrt(HEAD_DIMENSION)));
        }
        if (keys.length === 0) continue;
        const maximum = Math.max(...scores);
        const weights = scores.map((score) => Math.exp(score - maximum));
        const denominator = weights.reduce((sum, weight) => sum + weight, 0);
        for (let dimension = 0; dimension < HEAD_DIMENSION; dimension += 1) {
          let weighted = 0;
          for (let index = 0; index < keys.length; index += 1) {
            const valueIndex =
              ((batch * KV_HEADS + kvHead) * CACHE_CAPACITY + keys[index]!) *
                HEAD_DIMENSION + dimension;
            weighted += weights[index]! * cache.value[valueIndex]!;
          }
          const outputIndex =
            ((batch * QUERY_HEADS + queryHead) * tokens + queryToken) *
              HEAD_DIMENSION + dimension;
          output[outputIndex] = store(profile, weighted / denominator);
        }
      }
    }
  }
  return output;
}

function splitHeads(
  input: Float32Array,
  batch: number,
  tokens: number,
  heads: number,
  headDimension: number,
): Float32Array {
  const output = new Float32Array(input.length);
  for (let b = 0; b < batch; b += 1) {
    for (let token = 0; token < tokens; token += 1) {
      for (let head = 0; head < heads; head += 1) {
        for (let dimension = 0; dimension < headDimension; dimension += 1) {
          output[((b * heads + head) * tokens + token) * headDimension + dimension] =
            input[((b * tokens + token) * heads + head) * headDimension + dimension]!;
        }
      }
    }
  }
  return output;
}

function mergeHeads(
  input: Float32Array,
  batch: number,
  tokens: number,
  heads: number,
  headDimension: number,
): Float32Array {
  const output = new Float32Array(input.length);
  for (let b = 0; b < batch; b += 1) {
    for (let token = 0; token < tokens; token += 1) {
      for (let head = 0; head < heads; head += 1) {
        for (let dimension = 0; dimension < headDimension; dimension += 1) {
          output[((b * tokens + token) * heads + head) * headDimension + dimension] =
            input[((b * heads + head) * tokens + token) * headDimension + dimension]!;
        }
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

function quantizedWeights(
  profile: AceModelProfileId,
  weights: FloatWeights,
): FloatWeights {
  const quantize = (values: Float32Array): Float32Array =>
    Float32Array.from(values, (value) => weightValue(profile, value));
  return {
    inputLayerNorm: quantize(weights.inputLayerNorm),
    queryProjection: quantize(weights.queryProjection),
    keyProjection: quantize(weights.keyProjection),
    valueProjection: quantize(weights.valueProjection),
    queryNorm: quantize(weights.queryNorm),
    keyNorm: quantize(weights.keyNorm),
    outputProjection: quantize(weights.outputProjection),
    postAttentionLayerNorm: quantize(weights.postAttentionLayerNorm),
    gateProjection: quantize(weights.gateProjection),
    upProjection: quantize(weights.upProjection),
    downProjection: quantize(weights.downProjection),
    finalNorm: quantize(weights.finalNorm),
    embedding: quantize(weights.embedding),
  };
}

function matrix(rows: number, columns: number, seed: number): Float32Array {
  return Float32Array.from({ length: rows * columns }, (_, index) =>
    Math.fround((((index * seed + seed * 3) % 19) - 9) / 24),
  );
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

function sequence(
  ...dispatches: readonly { encode(pass: GPUComputePassEncoder): void }[]
) {
  return {
    encode(pass: GPUComputePassEncoder): void {
      for (const dispatch of dispatches) dispatch.encode(pass);
    },
  };
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

async function readU32(
  device: GPUDevice,
  source: GPUBuffer,
  count: number,
): Promise<Uint32Array> {
  return new Uint32Array(await readBytes(device, source, count * 4));
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
        `${label} mismatch at ${index}: received ${actual[index]}, expected ${expected[index]}, error ${error}`,
      );
    }
  }
  return maximumError;
}

function assertExact(
  actual: Uint32Array,
  expected: Uint32Array,
  label: string,
): void {
  if (actual.length !== expected.length) throw new Error(`${label} length mismatch`);
  for (let index = 0; index < actual.length; index += 1) {
    if (actual[index] !== expected[index]) {
      throw new Error(
        `${label} mismatch at ${index}: received ${actual[index]}, expected ${expected[index]}`,
      );
    }
  }
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
    return sign | (truncated + (remainder > halfway || (remainder === halfway && (truncated & 1)) ? 1 : 0));
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
  document.title = `ACE Qwen3 block ${status}`;
}

function requireResultNode(): HTMLPreElement {
  const node = document.querySelector<HTMLPreElement>("#result");
  if (node === null) throw new Error("Missing result element");
  return node;
}
