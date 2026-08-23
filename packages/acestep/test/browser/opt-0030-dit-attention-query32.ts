import {
  aceOpt0030AttentionWgsl,
  planAceOpt0030Attention,
  type AceOpt0030AttentionQueriesPerWorkgroup,
} from "../../src/webgpu/kernels/attention-query32.js";

declare global {
  interface Window {
    __ACE_OPT0030_RESULT__?: Readonly<Record<string, unknown>>;
  }
}

const SHAPE = Object.freeze({
  batch: 1,
  queryHeads: 16,
  keyValueHeads: 8,
  queryTokens: 2_250,
  keyValueTokens: 2_250,
  headDimension: 128,
  mode: "full" as const,
});
const ARMS = Object.freeze([8, 16, 32] as const);
const TIMING_ORDERS = Object.freeze([
  Object.freeze([8, 16, 32] as const),
  Object.freeze([16, 32, 8] as const),
  Object.freeze([32, 8, 16] as const),
] as const);
const QUERY_SEED = 0x1357_9bdf;
const KEY_SEED = 0x2468_ace0;
const VALUE_SEED = 0x1020_3040;

interface CompiledArm {
  readonly queries: AceOpt0030AttentionQueriesPerWorkgroup;
  readonly pipeline: GPUComputePipeline;
  readonly bindGroup: GPUBindGroup;
  readonly workgroupCount: number;
  readonly wgslSha256: string;
}

interface CorrectnessReceipt {
  readonly comparedElementsPerCandidate: number;
  readonly referenceNonFiniteCount: number;
  readonly query16NonFiniteCount: number;
  readonly query32NonFiniteCount: number;
  readonly query16BitMismatchCount: number;
  readonly query32BitMismatchCount: number;
  readonly query16MaximumAbsoluteError: number;
  readonly query32MaximumAbsoluteError: number;
  readonly outputSha256: Readonly<Record<"query8" | "query16" | "query32", string>>;
  readonly bitExact: true;
}

interface PreparedHarness {
  readonly adapter: GPUAdapter;
  readonly device: GPUDevice;
  readonly arms: ReadonlyMap<AceOpt0030AttentionQueriesPerWorkgroup, CompiledArm>;
  readonly ownedBuffers: readonly GPUBuffer[];
  readonly inputSha256: Readonly<Record<"query" | "key" | "value", string>>;
  readonly correctness: CorrectnessReceipt;
  destroy(): void;
}

const progress = requireElement<HTMLElement>("#progress");
const runButton = requireElement<HTMLButtonElement>("#run");
const result = requireElement<HTMLElement>("#result");
let active: PreparedHarness | undefined;
let started = false;

void prepare().then(
  (prepared) => {
    active = prepared;
    document.body.dataset.status = "ready";
    progress.textContent =
      "READY — full M2250 Q8/Q16/Q32 outputs are bit-exact; timing has not run";
    runButton.disabled = false;
  },
  (error: unknown) => fail(error),
);

runButton.addEventListener("click", () => {
  if (started || active === undefined) return;
  started = true;
  runButton.disabled = true;
  document.body.dataset.status = "running";
  const prepared = active;
  active = undefined;
  void runTiming(prepared).catch((error: unknown) => {
    prepared.destroy();
    fail(error);
  });
});

window.addEventListener("beforeunload", () => {
  active?.destroy();
  active = undefined;
});

async function prepare(): Promise<PreparedHarness> {
  if (navigator.gpu === undefined) throw new Error("WebGPU unavailable");
  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: "high-performance",
  });
  if (adapter === null) throw new Error("No WebGPU adapter");
  if (
    !adapter.features.has("subgroups") ||
    adapter.info.subgroupMinSize !== 32 ||
    adapter.info.subgroupMaxSize !== 32
  ) {
    throw new Error("OPT-0030 requires fixed 32-lane stock-WebGPU subgroups");
  }
  if (
    adapter.limits.maxComputeInvocationsPerWorkgroup < 1_024 ||
    adapter.limits.maxComputeWorkgroupSizeX < 1_024
  ) {
    throw new Error("OPT-0030 requires stock adapter workgroup size 1024");
  }

  const device = await adapter.requestDevice({
    requiredFeatures: ["subgroups"],
    requiredLimits: {
      maxComputeInvocationsPerWorkgroup: 1_024,
      maxComputeWorkgroupSizeX: 1_024,
      maxComputeWorkgroupStorageSize: 1_024,
    },
  });
  const ownedBuffers: GPUBuffer[] = [];
  try {
    progress.textContent = "materializing deterministic full-M2250 F32 inputs";
    const query = deterministicF32(
      SHAPE.queryHeads * SHAPE.queryTokens * SHAPE.headDimension,
      QUERY_SEED,
      0.125,
    );
    const keyValueElements =
      SHAPE.keyValueHeads * SHAPE.keyValueTokens * SHAPE.headDimension;
    const key = deterministicF32(keyValueElements, KEY_SEED, 0.125);
    const value = deterministicF32(keyValueElements, VALUE_SEED, 0.25);
    const inputSha256 = Object.freeze({
      query: await sha256(query),
      key: await sha256(key),
      value: await sha256(value),
    });

    const queryBuffer = uploadStorage(device, query, "opt0030-query");
    const keyBuffer = uploadStorage(device, key, "opt0030-key");
    const valueBuffer = uploadStorage(device, value, "opt0030-value");
    const validLengthsBuffer = uploadStorage(
      device,
      new Uint32Array([SHAPE.queryTokens, SHAPE.keyValueTokens]),
      "opt0030-valid-lengths",
    );
    const outputBuffer = device.createBuffer({
      label: "opt0030-output",
      size: query.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const readbackBuffer = device.createBuffer({
      label: "opt0030-readback",
      size: query.byteLength,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    ownedBuffers.push(
      queryBuffer,
      keyBuffer,
      valueBuffer,
      validLengthsBuffer,
      outputBuffer,
      readbackBuffer,
    );

    progress.textContent = "compiling Q8/Q16/Q32 stock-WebGPU pipelines";
    const arms = new Map<
      AceOpt0030AttentionQueriesPerWorkgroup,
      CompiledArm
    >();
    for (const queries of ARMS) {
      const plan = planAceOpt0030Attention(SHAPE, queries);
      const source = aceOpt0030AttentionWgsl(SHAPE, queries);
      const module = device.createShaderModule({
        label: `opt0030-query${queries}-module`,
        code: source,
      });
      await requireCleanCompilation(module, queries);
      const pipeline = await device.createComputePipelineAsync({
        label: `opt0030-query${queries}-pipeline`,
        layout: "auto",
        compute: { module, entryPoint: "main" },
      });
      const bindGroup = device.createBindGroup({
        label: `opt0030-query${queries}-bindings`,
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: binding(queryBuffer) },
          { binding: 1, resource: binding(keyBuffer) },
          { binding: 2, resource: binding(valueBuffer) },
          { binding: 3, resource: binding(validLengthsBuffer) },
          { binding: 4, resource: binding(outputBuffer) },
        ],
      });
      arms.set(queries, Object.freeze({
        queries,
        pipeline,
        bindGroup,
        workgroupCount: plan.workgroupCount,
        wgslSha256: await sha256Text(source),
      }));
    }

    progress.textContent =
      "running untimed full-M2250 Q8/Q16/Q32 correctness gate";
    const query8 = await executeAndRead(
      device,
      requireArm(arms, 8),
      outputBuffer,
      readbackBuffer,
      "opt0030-correctness-query8",
    );
    const query16 = await executeAndRead(
      device,
      requireArm(arms, 16),
      outputBuffer,
      readbackBuffer,
      "opt0030-correctness-query16",
    );
    const query16Comparison = compareBitExact(query8, query16);
    const query32 = await executeAndRead(
      device,
      requireArm(arms, 32),
      outputBuffer,
      readbackBuffer,
      "opt0030-correctness-query32",
    );
    const query32Comparison = compareBitExact(query8, query32);
    const referenceNonFiniteCount = countNonFinite(query8);
    const query16NonFiniteCount = countNonFinite(query16);
    const query32NonFiniteCount = countNonFinite(query32);
    const [query8Sha256, query16Sha256, query32Sha256] = await Promise.all([
      sha256(query8),
      sha256(query16),
      sha256(query32),
    ]);
    if (
      referenceNonFiniteCount !== 0 ||
      query16NonFiniteCount !== 0 ||
      query32NonFiniteCount !== 0 ||
      query16Comparison.bitMismatchCount !== 0 ||
      query32Comparison.bitMismatchCount !== 0
    ) {
      throw new Error(
        "OPT-0030 full-M2250 correctness failed: " +
          JSON.stringify({
            referenceNonFiniteCount,
            query16NonFiniteCount,
            query32NonFiniteCount,
            query16Comparison,
            query32Comparison,
          }),
      );
    }
    const correctness: CorrectnessReceipt = Object.freeze({
      comparedElementsPerCandidate: query8.length,
      referenceNonFiniteCount,
      query16NonFiniteCount,
      query32NonFiniteCount,
      query16BitMismatchCount: query16Comparison.bitMismatchCount,
      query32BitMismatchCount: query32Comparison.bitMismatchCount,
      query16MaximumAbsoluteError: query16Comparison.maximumAbsoluteError,
      query32MaximumAbsoluteError: query32Comparison.maximumAbsoluteError,
      outputSha256: Object.freeze({
        query8: query8Sha256,
        query16: query16Sha256,
        query32: query32Sha256,
      }),
      bitExact: true,
    });

    let destroyed = false;
    return Object.freeze({
      adapter,
      device,
      arms,
      ownedBuffers: Object.freeze(ownedBuffers.slice()),
      inputSha256,
      correctness,
      destroy(): void {
        if (destroyed) return;
        destroyed = true;
        for (const buffer of ownedBuffers) buffer.destroy();
        device.destroy();
      },
    });
  } catch (error) {
    for (const buffer of ownedBuffers) buffer.destroy();
    device.destroy();
    throw error;
  }
}

async function runTiming(prepared: PreparedHarness): Promise<void> {
  const samples = new Map<
    AceOpt0030AttentionQueriesPerWorkgroup,
    number[]
  >(ARMS.map((queries) => [queries, []]));
  for (let round = 0; round < TIMING_ORDERS.length; round += 1) {
    const order = TIMING_ORDERS[round]!;
    for (let position = 0; position < order.length; position += 1) {
      const queries = order[position]!;
      progress.textContent =
        `timing round ${round + 1}/3, position ${position + 1}/3: Q${queries}`;
      const milliseconds = await timeThroughQueueDrain(
        prepared.device,
        requireArm(prepared.arms, queries),
        `opt0030-timed-r${round + 1}-q${queries}`,
      );
      samples.get(queries)!.push(milliseconds);
    }
  }
  const medians = Object.freeze({
    query8: median3(samples.get(8)!),
    query16: median3(samples.get(16)!),
    query32: median3(samples.get(32)!),
  });
  const fastest = medians.query16 <= medians.query32 ? 16 : 32;
  const fastestMilliseconds = fastest === 16
    ? medians.query16
    : medians.query32;
  const speedupVersusQuery8 = medians.query8 / fastestMilliseconds;
  const receipt = Object.freeze({
    schemaVersion: 1,
    experiment: "OPT-0030",
    passed: speedupVersusQuery8 >= 1.35,
    identity: Object.freeze({
      browserUserAgent: navigator.userAgent,
      wgslSha256: Object.freeze(Object.fromEntries(
        ARMS.map((queries) => [
          `query${queries}`,
          requireArm(prepared.arms, queries).wgslSha256,
        ]),
      )),
    }),
    shape: SHAPE,
    capabilities: capabilityReceipt(prepared.adapter, prepared.device),
    deterministicInputs: Object.freeze({
      generator: "lcg1664525-u24-f32-v1",
      querySeed: QUERY_SEED,
      keySeed: KEY_SEED,
      valueSeed: VALUE_SEED,
      sha256: prepared.inputSha256,
    }),
    correctness: prepared.correctness,
    plans: Object.freeze(Object.fromEntries(ARMS.map((queries) => {
      const plan = planAceOpt0030Attention(SHAPE, queries);
      return [`query${queries}`, Object.freeze({
        queriesPerWorkgroup: queries,
        queryTokensPerTile: plan.queryTokensPerTile,
        workgroupSize: plan.workgroupSize,
        workgroupCount: plan.workgroupCount,
        keyValueScalarLoads: plan.keyValueScalarLoads,
        loadReductionVersusQuery8: plan.loadReductionVersusQuery8,
      })];
    }))),
    protocol: Object.freeze({
      compileOutsideTiming: true,
      fullCorrectnessOutsideTiming: true,
      correctnessAlsoServesAsOneWarmupPerArm: true,
      rounds: 3,
      balancedOrders: TIMING_ORDERS,
      samplesPerArm: 3,
      oneCommandBufferPerSample: true,
      oneQueueDrainPerSample: true,
      outputReadbackInsideTiming: false,
    }),
    timing: Object.freeze({
      samplesMilliseconds: Object.freeze({
        query8: Object.freeze(samples.get(8)!.slice()),
        query16: Object.freeze(samples.get(16)!.slice()),
        query32: Object.freeze(samples.get(32)!.slice()),
      }),
      mediansMilliseconds: medians,
      fastestCandidate: `query${fastest}`,
      fastestCandidateMilliseconds: fastestMilliseconds,
      speedupVersusQuery8,
      requiredSpeedupVersusQuery8: 1.35,
    }),
    cleanup: Object.freeze({
      ownedBufferCount: prepared.ownedBuffers.length,
      buffersDestroyed: true,
      deviceDestroyed: true,
    }),
  });
  prepared.destroy();
  window.__ACE_OPT0030_RESULT__ = receipt;
  result.textContent = JSON.stringify(receipt);
  document.body.dataset.status = receipt.passed ? "passed" : "failed";
  progress.textContent = receipt.passed
    ? `passed — Q${fastest} is ${speedupVersusQuery8.toFixed(3)}× Q8`
    : `completed — Q${fastest} is only ${speedupVersusQuery8.toFixed(3)}× Q8`;
}

async function requireCleanCompilation(
  module: GPUShaderModule,
  queries: AceOpt0030AttentionQueriesPerWorkgroup,
): Promise<void> {
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((message) => message.type === "error");
  if (errors.length !== 0) {
    throw new Error(
      `OPT-0030 Q${queries} WGSL compilation failed:\n` +
        errors.map((message) =>
          `${message.lineNum}:${message.linePos} ${message.message}`
        ).join("\n"),
    );
  }
}

async function executeAndRead(
  device: GPUDevice,
  arm: CompiledArm,
  output: GPUBuffer,
  readback: GPUBuffer,
  label: string,
): Promise<Float32Array<ArrayBuffer>> {
  const encoder = device.createCommandEncoder({ label });
  const pass = encoder.beginComputePass({ label });
  encodeArm(pass, arm);
  pass.end();
  encoder.copyBufferToBuffer(output, 0, readback, 0, output.size);
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const detached = new Float32Array(readback.getMappedRange().slice(0));
  readback.unmap();
  return detached;
}

async function timeThroughQueueDrain(
  device: GPUDevice,
  arm: CompiledArm,
  label: string,
): Promise<number> {
  const started = performance.now();
  const encoder = device.createCommandEncoder({ label });
  const pass = encoder.beginComputePass({ label });
  encodeArm(pass, arm);
  pass.end();
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  return performance.now() - started;
}

function encodeArm(pass: GPUComputePassEncoder, arm: CompiledArm): void {
  pass.setPipeline(arm.pipeline);
  pass.setBindGroup(0, arm.bindGroup);
  pass.dispatchWorkgroups(arm.workgroupCount, 1, 1);
}

function compareBitExact(
  reference: Float32Array<ArrayBuffer>,
  candidate: Float32Array<ArrayBuffer>,
): Readonly<{ bitMismatchCount: number; maximumAbsoluteError: number }> {
  if (reference.length !== candidate.length) {
    throw new Error("OPT-0030 output lengths differ");
  }
  const referenceBits = new Uint32Array(
    reference.buffer,
    reference.byteOffset,
    reference.length,
  );
  const candidateBits = new Uint32Array(
    candidate.buffer,
    candidate.byteOffset,
    candidate.length,
  );
  let bitMismatchCount = 0;
  let maximumAbsoluteError = 0;
  for (let index = 0; index < reference.length; index += 1) {
    if (referenceBits[index] !== candidateBits[index]) bitMismatchCount += 1;
    const error = Math.abs(candidate[index]! - reference[index]!);
    if (Number.isFinite(error)) maximumAbsoluteError = Math.max(
      maximumAbsoluteError,
      error,
    );
  }
  return Object.freeze({ bitMismatchCount, maximumAbsoluteError });
}

function countNonFinite(values: Float32Array<ArrayBuffer>): number {
  let count = 0;
  for (const value of values) {
    if (!Number.isFinite(value)) count += 1;
  }
  return count;
}

function deterministicF32(
  length: number,
  seed: number,
  scale: number,
): Float32Array<ArrayBuffer> {
  const values = new Float32Array(length);
  let state = seed >>> 0;
  for (let index = 0; index < values.length; index += 1) {
    state = (Math.imul(state, 1_664_525) + 1_013_904_223) >>> 0;
    const unit = (state >>> 8) / 0x1_000000;
    values[index] = Math.fround((unit * 2 - 1) * scale);
  }
  return values;
}

function uploadStorage(
  device: GPUDevice,
  values: Float32Array<ArrayBuffer> | Uint32Array<ArrayBuffer>,
  label: string,
): GPUBuffer {
  const buffer = device.createBuffer({
    label,
    size: values.byteLength,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });
  const mapped = buffer.getMappedRange();
  if (values instanceof Float32Array) {
    new Float32Array(mapped).set(values);
  } else {
    new Uint32Array(mapped).set(values);
  }
  buffer.unmap();
  return buffer;
}

function binding(buffer: GPUBuffer): GPUBufferBinding {
  return { buffer, offset: 0, size: buffer.size };
}

function requireArm(
  arms: ReadonlyMap<AceOpt0030AttentionQueriesPerWorkgroup, CompiledArm>,
  queries: AceOpt0030AttentionQueriesPerWorkgroup,
): CompiledArm {
  const arm = arms.get(queries);
  if (arm === undefined) throw new Error(`Missing OPT-0030 Q${queries} arm`);
  return arm;
}

function median3(samples: readonly number[]): number {
  if (samples.length !== 3 || samples.some((value) => !Number.isFinite(value))) {
    throw new Error("OPT-0030 requires exactly three finite samples per arm");
  }
  return samples.slice().sort((left, right) => left - right)[1]!;
}

function capabilityReceipt(adapter: GPUAdapter, device: GPUDevice) {
  return Object.freeze({
    adapterInfo: Object.freeze({
      vendor: adapter.info.vendor,
      architecture: adapter.info.architecture,
      device: adapter.info.device,
      description: adapter.info.description,
      subgroupMinSize: adapter.info.subgroupMinSize,
      subgroupMaxSize: adapter.info.subgroupMaxSize,
      isFallbackAdapter: adapter.info.isFallbackAdapter,
    }),
    features: Object.freeze([...device.features].sort()),
    limits: Object.freeze({
      maxComputeInvocationsPerWorkgroup:
        device.limits.maxComputeInvocationsPerWorkgroup,
      maxComputeWorkgroupSizeX: device.limits.maxComputeWorkgroupSizeX,
      maxComputeWorkgroupStorageSize:
        device.limits.maxComputeWorkgroupStorageSize,
      maxComputeWorkgroupsPerDimension:
        device.limits.maxComputeWorkgroupsPerDimension,
    }),
  });
}

async function sha256(
  values: Float32Array<ArrayBuffer>,
): Promise<string> {
  return hex(await crypto.subtle.digest("SHA-256", values));
}

async function sha256Text(value: string): Promise<string> {
  return hex(await crypto.subtle.digest(
    "SHA-256",
    new TextEncoder().encode(value),
  ));
}

function hex(value: ArrayBuffer): string {
  return [...new Uint8Array(value)]
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

function requireElement<T extends Element>(selector: string): T {
  const element = document.querySelector<T>(selector);
  if (element === null) throw new Error(`Missing ${selector}`);
  return element;
}

function fail(error: unknown): void {
  document.body.dataset.status = "failed";
  progress.textContent = "failed";
  result.textContent = error instanceof Error
    ? error.stack ?? error.message
    : String(error);
}

