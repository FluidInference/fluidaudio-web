/// <reference types="@webgpu/types" />
/// <reference types="vite/client" />

import auditedCoreSource from
  "../../benchmark/opt-0011-vae-conv1d-fp16.ts?raw";
import productionCoreSource from
  "../../src/webgpu/kernels/vae-conv1d-fp16.ts?raw";
import {
  AceOpt0011VaeConv1dFp16ScalarOracleKernel,
  aceOpt0011VaeConv1dFp16PortableWorkgroupWgsl,
  aceOpt0011VaeConv1dFp16ScalarOracleWgsl,
  planAceOpt0011VaeConv1dFp16,
  type AceOpt0011VaeConv1dFp16Bindings,
  type AceOpt0011VaeConv1dFp16Dispatch,
} from "../../benchmark/opt-0011-vae-conv1d-fp16.js";
import {
  AceFp16VaeConv1dKernel,
  aceFp16VaeConv1dWgsl,
  planAceFp16VaeConv1d,
  planAceFp16VaeConv1dRange,
  type AceFp16VaeConv1dBindings,
  type AceFp16VaeConv1dDispatch,
  type AceFp16VaeConv1dOutputStorage,
  type AceFp16VaeConv1dPlan,
} from "../../src/webgpu/kernels/vae-conv1d-fp16.js";
import type {
  AceVaeConv1dShape,
  AceVaeOutputRangeBinding,
} from "../../src/webgpu/kernels/vae-primitives.js";
import {
  planAceVaeDecoder,
  planAceVaeDecoderQuanta,
  type AceVaeDecoderConvOperation,
  type AceVaeDecoderQuantumPlan,
} from "../../src/webgpu/vae-decoder.js";

export const OPT_0011_PRODUCTION_CONV1D_CORE_COMMIT =
  "75f70f12bdb43ae33b9bd37391b7d49be5aa1704" as const;
export const OPT_0011_PRODUCTION_CONV1D_CORE_SOURCE_SHA256 =
  "fd14f625e3efeba3277bd9c4e8aa052af92a2b44c078108303173c9bb42a4310" as const;
export const OPT_0011_AUDITED_CONV1D_CORE_COMMIT =
  "82f0fa4b3d5e676ec9dc967c3563dc9650cc59bd" as const;
export const OPT_0011_AUDITED_CONV1D_CORE_SOURCE_SHA256 =
  "bdb1ce2732d8617f61132401ab01155163a4f4197e7c7b01eb550b8408553ceb" as const;
export const OPT_0011_PRODUCTION_K1_SCALAR_ORACLE_ID =
  "opt-0011-production-k1-fp16-scalar-oracle-v1" as const;

export type Opt0011ProductionConv1dCaseId =
  | "k7-block0-d1-first"
  | "k7-block0-d3-middle"
  | "k7-block0-d9-last"
  | "k7-final-last"
  | "k1-block0-tail"
  | "k1-block1-tail"
  | "k1-block2-tail"
  | "k1-block3-middle"
  | "k1-block4-last"
  | "k1-arithmetic-b1-f17-c65-c9";

export interface Opt0011ProductionConv1dFixture {
  readonly id: Opt0011ProductionConv1dCaseId;
  readonly family: "k1" | "k7";
  readonly graphOperationLabel: string | null;
  readonly graphOperationOrdinal: number | null;
  readonly graphFamilyMultiplicity: number | null;
  readonly shape: AceVaeConv1dShape;
  readonly hasBias: boolean;
  readonly outputStorage: AceFp16VaeConv1dOutputStorage;
  readonly range: Readonly<{ readonly base: number; readonly count: number }>;
  readonly graphQuantumIndex: number | null;
  readonly graphOperationQuantumIndex: number | null;
  readonly auditedRangeIndex: number | null;
  readonly inputFirstRow: number;
  readonly inputRowCount: number;
  readonly cpuOracleScope: "complete-selected-range" | "audited-k7-scalar";
  readonly coverage: readonly string[];
}

export interface Opt0011ProductionConv1dRunIdentity {
  readonly harnessCommit: string;
  readonly coreCommit: typeof OPT_0011_PRODUCTION_CONV1D_CORE_COMMIT;
  readonly auditedCoreCommit: typeof OPT_0011_AUDITED_CONV1D_CORE_COMMIT;
}

export interface Opt0011ProductionRawBitComparison {
  readonly mismatchCount: number;
  readonly firstMismatchIndex: number | null;
}

const OUTPUT_GUARD_BYTES = 256;
const OUTPUT_CANARY_BYTES = 256;
const OUTPUT_GUARD_WORD = 0xa55a_a55a;
const AUTHORITY_F16_SENTINEL = 0x7e11;
const PRODUCTION_F16_SENTINEL = 0x7e22;
const AUTHORITY_F32_SENTINEL = 0x7fc1_1111;
const PRODUCTION_F32_SENTINEL = 0x7fc2_2222;
const QUEUE_EMPTY_IDLE_MILLISECONDS = 1;
const K1_SCALAR_WORKGROUP_SIZE = 64;
const CONTROL_BYTES = 16;
const FLOAT32_SCRATCH = new Float32Array(1);
const UINT32_SCRATCH = new Uint32Array(FLOAT32_SCRATCH.buffer);

const INPUT_PATTERN = Object.freeze([
  0x0000, 0x8000, 0x2400, 0xa400, 0x2c00, 0xac00, 0x3000, 0xb000,
  0x3400, 0xb400, 0x3800, 0xb800, 0x3555, 0xb555, 0x1800, 0x9800,
]);
const WEIGHT_PATTERN = Object.freeze([
  0x0000, 0x8000, 0x1400, 0x9400, 0x1800, 0x9800, 0x1c00, 0x9c00,
  0x2000, 0xa000, 0x2200, 0xa200, 0x2400, 0xa400,
]);
const BIAS_PATTERN = Object.freeze([
  0x0000, 0x8000, 0x2000, 0xa000, 0x2400, 0xa400, 0x2800, 0xa800,
]);

const B256_GRAPH = planAceVaeDecoder(256);
const B256_QUANTA = planAceVaeDecoderQuanta(B256_GRAPH);
const K1_GRAPH_OPERATIONS = B256_GRAPH.operations.filter(
  (operation): operation is AceVaeDecoderConvOperation =>
    operation.kind === "conv1d" && operation.shape.kernelSize === 1,
);

export const OPT_0011_PRODUCTION_CONV1D_CASES = Object.freeze([
  graphFixture(
    "k7-block0-d1-first",
    "block-0-res-1-conv1",
    "first",
    ["promoted-k7", "dilation-1", "exact-b256-quantum", "first-range"],
  ),
  graphFixture(
    "k7-block0-d3-middle",
    "block-0-res-2-conv1",
    "middle",
    ["promoted-k7", "dilation-3", "exact-b256-quantum", "middle-range"],
  ),
  graphFixture(
    "k7-block0-d9-last",
    "block-0-res-3-conv1",
    "last",
    ["promoted-k7", "dilation-9", "exact-b256-quantum", "tail-range"],
  ),
  graphFixture(
    "k7-final-last",
    "conv2",
    "last",
    ["promoted-k7", "final-f32-boundary", "no-bias", "tail-range"],
  ),
  graphFixture(
    "k1-block0-tail",
    "block-0-res-1-conv2",
    "last",
    ["biased-k1", "c1024", "exact-b256-quantum", "tail-range"],
  ),
  graphFixture(
    "k1-block1-tail",
    "block-1-res-2-conv2",
    "last",
    ["biased-k1", "c512", "exact-b256-quantum", "tail-range"],
  ),
  graphFixture(
    "k1-block2-tail",
    "block-2-res-3-conv2",
    "last",
    ["biased-k1", "c256", "exact-b256-quantum", "tail-range"],
  ),
  graphFixture(
    "k1-block3-middle",
    "block-3-res-1-conv2",
    "middle",
    ["biased-k1", "c128", "exact-b256-quantum", "middle-range"],
  ),
  graphFixture(
    "k1-block4-last",
    "block-4-res-3-conv2",
    "last",
    ["biased-k1", "c128", "longest-b256-shape", "last-range"],
  ),
  arithmeticFixture(),
] satisfies readonly Opt0011ProductionConv1dFixture[]);

/** SHA-256 over the exact generated WGSL passed to each browser shader module. */
export const OPT_0011_PRODUCTION_CONV1D_GENERATED_SHADER_SHA256 = Object.freeze({
  "k7-block0-d1-first": Object.freeze({
    production: "a5aa535ef9da1a37d1f3c32c8072a1496c790662c7e4bdab6244e50f956d47c8",
    authority: "0e28ef309e8e1ec4beb476bffd57b5ecc6d5591a8d56fff9d74841595983ce8f",
    auditedPortable: "8a6a217d956ba42f4b73952d8d0b9017afa2a286eeb5da08d14d16dc61bc16c2",
  }),
  "k7-block0-d3-middle": Object.freeze({
    production: "6ec51c059c43440b45d175b4a0451a6652f012c62d70af6996203e4936e57e4b",
    authority: "e5185da415207482579d8b432942efb477e2aa87bd82c29f9b54ca3a2a45b995",
    auditedPortable: "c96bd956e65f97cc9600c94312ed55324c90770e85dd235000ba03bafb5037f7",
  }),
  "k7-block0-d9-last": Object.freeze({
    production: "954103f94a3d6358ba52a5579fce3ff3cf71ffcbef73cdc568c9ab8740bcd3dc",
    authority: "5e950d9ed2c7a358620ade359655d19b92de006ad69908c1e16de0e81fd718bd",
    auditedPortable: "4ed0b472ba8d331cf0a98526afedbd1732de8790d5fa9f2d11b7d85c081761f8",
  }),
  "k7-final-last": Object.freeze({
    production: "e678f6d3fc3f2d5c550998da541a403babf30e5bc37445c9809d632e1e7e3693",
    authority: "fae927991838bddbd7564cb2732c64fa69927f8eed304ad7bb2e70aa8efa807e",
    auditedPortable: "1ebda1effae91f71db4066853baf0c3d6e0758cffe0cffb05020eb7212428912",
  }),
  "k1-block0-tail": Object.freeze({
    production: "51b2ba6a701c28a743ffff64379097f87ee18c137c0953237a9913edba45bee6",
    authority: "65ca55b251211f1ce7eaeb63b317cfffeb743461c4aed8bfc3050ad305b4755b",
  }),
  "k1-block1-tail": Object.freeze({
    production: "16ab82114aa5f2201c379cd0d116cf0670198152249d80802a483e5522ea0fc8",
    authority: "777c26cd93ca8e2ccecdf2c5106937130f0067af15c40797014cd1c4f9abee5a",
  }),
  "k1-block2-tail": Object.freeze({
    production: "18cd443742995d2328ed23995c0ae6a19e7feb090ae255f06ea063689235b1b9",
    authority: "1872919eec2cea6741e276e24526e35766dbabcd28e13d481dac3a0f06eb6242",
  }),
  "k1-block3-middle": Object.freeze({
    production: "ce0fe526f5057e22315595884b6ce2d06287462237f5e7f9d2709c10df070783",
    authority: "f7a2cd07906e4066cbff3e0e51dadf6d921209e6769ca9dd2e768b0c9ebcfea1",
  }),
  "k1-block4-last": Object.freeze({
    production: "36ef459b5628705a3ca726615b0b0a53a70e983f2f425b5d9e7383e3470ba742",
    authority: "4d679432cc83d540367b988ded55af9aefeb6e08257ba49cad0a4761573a01fc",
  }),
  "k1-arithmetic-b1-f17-c65-c9": Object.freeze({
    production: "a0528af04fca039cdf15a6b15f2cf3d8ba582fdf3e45c70a1e79bd1b8eeb1f5b",
    authority: "e638d63dd3fa333a15b612c52ebdfbae4ede99225bfd0d576a9c15d264f8c580",
  }),
} satisfies Readonly<Record<
  Opt0011ProductionConv1dCaseId,
  Readonly<{ readonly production: string; readonly authority: string; readonly auditedPortable?: string }>
>>);

export function parseOpt0011ProductionConv1dRunIdentity(
  parameters: URLSearchParams,
): Opt0011ProductionConv1dRunIdentity {
  const harnessCommit = requiredIdentity(parameters, "harnessCommit");
  if (!/^[0-9a-f]{40}$/.test(harnessCommit)) {
    throw new Error("OPT-0011 harnessCommit must be a 40-character lowercase hex commit");
  }
  const coreCommit = requiredIdentity(parameters, "coreCommit");
  if (coreCommit !== OPT_0011_PRODUCTION_CONV1D_CORE_COMMIT) {
    throw new Error("OPT-0011 production Conv1D coreCommit changed");
  }
  const auditedCoreCommit = requiredIdentity(parameters, "auditedCoreCommit");
  if (auditedCoreCommit !== OPT_0011_AUDITED_CONV1D_CORE_COMMIT) {
    throw new Error("OPT-0011 audited Conv1D coreCommit changed");
  }
  return Object.freeze({ harnessCommit, coreCommit, auditedCoreCommit });
}

export function opt0011ProductionK1ScalarOracleWgsl(
  shape: AceVaeConv1dShape,
): string {
  const plan = planAceFp16VaeConv1d(shape, "float16");
  if (plan.family !== "k1") {
    throw new RangeError("OPT-0011 production K1 scalar oracle requires K1");
  }
  return /* wgsl */ `
// kernel-id: ${OPT_0011_PRODUCTION_K1_SCALAR_ORACLE_ID}
enable f16;

const INPUT_FRAMES: u32 = ${plan.inputFrames}u;
const INPUT_CHANNELS: u32 = ${plan.inputChannels}u;
const OUTPUT_CHANNELS: u32 = ${plan.outputChannels}u;

@group(0) @binding(0) var<storage, read> input: array<f16>;
@group(0) @binding(1) var<storage, read> weight: array<f16>;
@group(0) @binding(2) var<storage, read> bias: array<f16>;
@group(0) @binding(3) var<storage, read_write> output: array<f16>;

struct OutputRangeParameters {
  first_output: u32,
  output_count: u32,
  _padding0: u32,
  _padding1: u32,
}
@group(0) @binding(4) var<uniform>
  output_range: OutputRangeParameters;

@compute @workgroup_size(${K1_SCALAR_WORKGROUP_SIZE}, 1, 1)
fn main(@builtin(global_invocation_id) global: vec3<u32>) {
  if (global.x >= output_range.output_count) { return; }
  let output_index = output_range.first_output + global.x;
  let output_channel = output_index % OUTPUT_CHANNELS;
  let output_row = output_index / OUTPUT_CHANNELS;
  var sum = f32(bias[output_channel]);
  let input_base = output_row * INPUT_CHANNELS;
  let weight_base = output_channel * INPUT_CHANNELS;
  for (
    var input_channel = 0u;
    input_channel < INPUT_CHANNELS;
    input_channel += 1u
  ) {
    let input_operand = f32(input[input_base + input_channel]);
    let weight_operand = f32(weight[weight_base + input_channel]);
    sum = sum + input_operand * weight_operand;
  }
  output[output_index] = f16(sum);
}
`;
}

export function compareOpt0011ProductionRawBits(
  left: Uint16Array | Uint32Array,
  right: Uint16Array | Uint32Array,
): Opt0011ProductionRawBitComparison {
  if (left.constructor !== right.constructor || left.length !== right.length) {
    throw new Error("OPT-0011 production output bit domains differ");
  }
  let mismatchCount = 0;
  let firstMismatchIndex: number | null = null;
  for (let index = 0; index < left.length; index += 1) {
    if (left[index] !== right[index]) {
      mismatchCount += 1;
      firstMismatchIndex ??= index;
    }
  }
  return Object.freeze({ mismatchCount, firstMismatchIndex });
}

export function opt0011ProductionInputBits(
  fixtureValue: Opt0011ProductionConv1dFixture,
  globalInputIndex: number,
): number {
  const inputElements = fixtureValue.shape.batch *
    fixtureValue.shape.inputFrames * fixtureValue.shape.inputChannels;
  requireIndex(globalInputIndex, inputElements, `${fixtureValue.id} input`);
  return inputBitsUnchecked(fixtureValue, globalInputIndex);
}

function inputBitsUnchecked(
  fixtureValue: Opt0011ProductionConv1dFixture,
  globalInputIndex: number,
): number {
  if (fixtureValue.id === "k1-arithmetic-b1-f17-c65-c9") {
    const channel = globalInputIndex % fixtureValue.shape.inputChannels;
    return [0x3c00, 0x7bff, 0xfbff, 0x1000, 0x1400][channel] ?? 0x0000;
  }
  const inputChannel = globalInputIndex % fixtureValue.shape.inputChannels;
  const inputRow = Math.floor(
    globalInputIndex / fixtureValue.shape.inputChannels,
  );
  return INPUT_PATTERN[
    mix32(
      Math.imul(inputRow % 16, 0x045d_9f3b) ^ inputChannel ^
        fixtureSalt(fixtureValue.id) ^ 0x1357_9bdf,
    ) %
      INPUT_PATTERN.length
  ]!;
}

export function opt0011ProductionWeightBits(
  fixtureValue: Opt0011ProductionConv1dFixture,
  weightIndex: number,
): number {
  const plan = planAceFp16VaeConv1d(
    fixtureValue.shape,
    fixtureValue.outputStorage,
  );
  requireIndex(weightIndex, plan.weightElements, `${fixtureValue.id} weight`);
  return weightBitsUnchecked(fixtureValue, weightIndex);
}

function weightBitsUnchecked(
  fixtureValue: Opt0011ProductionConv1dFixture,
  weightIndex: number,
): number {
  if (fixtureValue.id === "k1-arithmetic-b1-f17-c65-c9") {
    return arithmeticWeightBits(fixtureValue.shape.inputChannels, weightIndex);
  }
  return WEIGHT_PATTERN[
    mix32(weightIndex ^ fixtureSalt(fixtureValue.id) ^ 0x2468_ace0) %
      WEIGHT_PATTERN.length
  ]!;
}

export function opt0011ProductionBiasBits(
  fixtureValue: Opt0011ProductionConv1dFixture,
  outputChannel: number,
): number {
  if (!fixtureValue.hasBias) throw new Error(`${fixtureValue.id} has no bias`);
  requireIndex(
    outputChannel,
    fixtureValue.shape.outputChannels,
    `${fixtureValue.id} bias`,
  );
  return biasBitsUnchecked(fixtureValue, outputChannel);
}

function biasBitsUnchecked(
  fixtureValue: Opt0011ProductionConv1dFixture,
  outputChannel: number,
): number {
  if (fixtureValue.id === "k1-arithmetic-b1-f17-c65-c9") {
    return [
      0x8000, 0x0000, 0x0000, 0x0000, 0x3c00, 0x3c01,
      0x0400, 0x03ff, 0x0000,
    ][outputChannel]!;
  }
  return BIAS_PATTERN[
    mix32(outputChannel ^ fixtureSalt(fixtureValue.id) ^ 0x0bad_f00d) %
      BIAS_PATTERN.length
  ]!;
}

export function opt0011ProductionK1CpuBits(
  fixtureValue: Opt0011ProductionConv1dFixture,
  outputIndex: number,
): number {
  if (fixtureValue.family !== "k1") {
    throw new Error("OPT-0011 K1 CPU oracle rejects K7");
  }
  const plan = planAceFp16VaeConv1d(fixtureValue.shape, "float16");
  requireIndex(outputIndex, plan.outputElements, `${fixtureValue.id} output`);
  const outputChannel = outputIndex % plan.outputChannels;
  const outputRow = Math.floor(outputIndex / plan.outputChannels);
  const inputBase = outputRow * plan.inputChannels;
  const weightBase = outputChannel * plan.inputChannels;
  let sum = float16BitsToNumber(
    biasBitsUnchecked(fixtureValue, outputChannel),
  );
  for (let inputChannel = 0; inputChannel < plan.inputChannels; inputChannel += 1) {
    const inputValue = float16BitsToNumber(
      inputBitsUnchecked(fixtureValue, inputBase + inputChannel),
    );
    const weightValue = float16BitsToNumber(
      weightBitsUnchecked(fixtureValue, weightBase + inputChannel),
    );
    sum = Math.fround(sum + inputValue * weightValue);
  }
  return numberToFloat16Bits(sum);
}

interface ExecutionCounts {
  readonly encodedCommandBuffers: 1;
  readonly submissions: 1;
  readonly drains: 1;
  readonly dispatches: 1;
  readonly queueEmptyIdleTurns: 1;
}

interface GpuArmDispatch {
  readonly id: "authority" | "production";
  readonly label: string;
  encode(pass: GPUComputePassEncoder): void;
}

interface OutputTarget {
  readonly id: GpuArmDispatch["id"];
  readonly buffer: GPUBuffer;
  readonly binding: GPUBufferBinding;
  readonly sentinelBits: number;
  readonly sentinelWord: number;
}

interface SelectedOutputReadback {
  readonly bits: Uint16Array | Uint32Array;
  readonly sha256: string;
  readonly scan: Readonly<Record<string, unknown>>;
}

interface PreparedFixture {
  readonly fixture: Opt0011ProductionConv1dFixture;
  readonly plan: AceFp16VaeConv1dPlan;
  readonly authority: GpuArmDispatch;
  readonly production: GpuArmDispatch;
  readonly authorityOutput: OutputTarget;
  readonly productionOutput: OutputTarget;
  readonly uploadIdentity: Readonly<Record<string, unknown>>;
  readonly sourceIdentity: Readonly<Record<string, unknown>>;
  destroy(): void;
}

interface K1ScalarDispatch extends GpuArmDispatch {
  readonly id: "authority";
}

class BufferTracker {
  private readonly live = new Set<GPUBuffer>();
  created = 0;
  destroyed = 0;
  maximumLive = 0;

  create(device: GPUDevice, descriptor: GPUBufferDescriptor): GPUBuffer {
    const buffer = device.createBuffer(descriptor);
    this.live.add(buffer);
    this.created += 1;
    this.maximumLive = Math.max(this.maximumLive, this.live.size);
    return buffer;
  }

  destroy(buffer: GPUBuffer): void {
    if (!this.live.delete(buffer)) return;
    buffer.destroy();
    this.destroyed += 1;
  }

  destroyAll(): void {
    for (const buffer of [...this.live]) this.destroy(buffer);
  }

  receipt(): Readonly<Record<string, number>> {
    return Object.freeze({
      created: this.created,
      destroyed: this.destroyed,
      live: this.live.size,
      maximumLive: this.maximumLive,
    });
  }
}

if (typeof document !== "undefined") installBrowserUi();

function installBrowserUi(): void {
  const start = document.querySelector<HTMLButtonElement>("#start");
  if (start === null) throw new Error("Missing OPT-0011 production start button");
  start.addEventListener("click", () => {
    start.disabled = true;
    document.body.dataset.status = "running";
    updateProgress("authenticating frozen production and audited sources");
    void runBrowser().then(
      (result) => finish("passed", result),
      (error: unknown) => finish("failed", {
        schema: "ace-opt-0011-production-fp16-conv1d-correctness-v1",
        status: "failed",
        experimentId: "OPT-0011",
        error: errorReceipt(error),
      }),
    );
  }, { once: true });
}

async function runBrowser(): Promise<unknown> {
  const identity = parseOpt0011ProductionConv1dRunIdentity(
    new URLSearchParams(window.location.search),
  );
  const [actualProductionSourceSha256, actualAuditedSourceSha256] =
    await Promise.all([
      sha256Hex(new TextEncoder().encode(productionCoreSource)),
      sha256Hex(new TextEncoder().encode(auditedCoreSource)),
    ]);
  if (
    actualProductionSourceSha256 !==
      OPT_0011_PRODUCTION_CONV1D_CORE_SOURCE_SHA256
  ) {
    throw new Error("OPT-0011 rejected unauthenticated production Conv1D source");
  }
  if (
    actualAuditedSourceSha256 !== OPT_0011_AUDITED_CONV1D_CORE_SOURCE_SHA256
  ) {
    throw new Error("OPT-0011 rejected unauthenticated audited Conv1D source");
  }
  if (navigator.gpu === undefined) throw new Error("WebGPU unavailable");
  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: "high-performance",
    forceFallbackAdapter: false,
  });
  if (adapter === null) throw new Error("No WebGPU adapter");
  requireAdapter(adapter);
  const rawDevice = await adapter.requestDevice({
    label: "ace-opt-0011-production-fp16-conv1d-correctness-device",
    requiredFeatures: ["shader-f16"],
    requiredLimits: requiredDeviceLimits(),
  });
  const device = rawDevice;
  const tracker = new BufferTracker();
  const uncapturedErrors: Readonly<Record<string, unknown>>[] = [];
  const runtimeErrors: Readonly<Record<string, unknown>>[] = [];
  let unexpectedDeviceLoss: Readonly<Record<string, unknown>> | null = null;
  let destroyingDevice = false;
  const onUncaptured = (event: GPUUncapturedErrorEvent): void => {
    uncapturedErrors.push(errorReceipt(event.error));
  };
  const onWindowError = (event: ErrorEvent): void => {
    runtimeErrors.push(Object.freeze({
      type: "error",
      message: event.message,
      filename: event.filename,
      lineno: event.lineno,
      colno: event.colno,
    }));
  };
  const onUnhandledRejection = (event: PromiseRejectionEvent): void => {
    runtimeErrors.push(Object.freeze({
      type: "unhandledrejection",
      reason: errorReceipt(event.reason),
    }));
  };
  rawDevice.addEventListener("uncapturederror", onUncaptured);
  window.addEventListener("error", onWindowError);
  window.addEventListener("unhandledrejection", onUnhandledRejection);
  void rawDevice.lost.then((info) => {
    if (!destroyingDevice) {
      unexpectedDeviceLoss = Object.freeze({
        reason: info.reason,
        message: info.message,
      });
    }
  });
  const heartbeat = startHeartbeat();
  const cases: unknown[] = [];
  let cancellation: unknown = null;
  let responsiveness: unknown;
  let cleanup: Readonly<Record<string, unknown>> | undefined;
  let postCleanupValidationFailure: Error | undefined;
  try {
    for (
      let index = 0;
      index < OPT_0011_PRODUCTION_CONV1D_CASES.length;
      index += 1
    ) {
      const fixtureValue = OPT_0011_PRODUCTION_CONV1D_CASES[index]!;
      updateProgress(
        `correctness ${index + 1}/${OPT_0011_PRODUCTION_CONV1D_CASES.length}: ${fixtureValue.id}`,
      );
      cases.push(await runFixture(device, tracker, fixtureValue));
      await yieldToBrowser();
    }
    updateProgress("post-drain cancellation and readback-prevention proof");
    cancellation = await runCancellationProof(device, tracker);
    await rawDevice.queue.onSubmittedWorkDone();
    await yieldToBrowser();
  } finally {
    const cleanupStartedAtEpochMilliseconds = Date.now();
    tracker.destroyAll();
    tracker.destroyAll();
    const harnessBuffersDestroyedAtEpochMilliseconds = Date.now();
    const receipt = tracker.receipt();
    const accountingExact =
      receipt.live === 0 && receipt.created === receipt.destroyed;
    destroyingDevice = true;
    rawDevice.destroy();
    const deviceDestroyCalledAtEpochMilliseconds = Date.now();
    const intentionalDeviceLoss = await rawDevice.lost;
    const deviceLossSettledAtEpochMilliseconds = Date.now();
    await yieldToBrowser();
    await yieldToBrowser();
    const postCleanupEventTurnsCompletedAtEpochMilliseconds = Date.now();
    const finalUncapturedErrors = Object.freeze([...uncapturedErrors]);
    const finalRuntimeErrors = Object.freeze([...runtimeErrors]);
    const finalUnexpectedDeviceLoss = unexpectedDeviceLoss;
    const finalEventSnapshotAtEpochMilliseconds = Date.now();
    const postCleanupValidationClean =
      accountingExact && finalUncapturedErrors.length === 0 &&
      finalRuntimeErrors.length === 0 && finalUnexpectedDeviceLoss === null &&
      intentionalDeviceLoss.reason === "destroyed";
    const postCleanupValidationAtEpochMilliseconds = Date.now();
    responsiveness = heartbeat.stop();
    const heartbeatStoppedAtEpochMilliseconds = Date.now();
    cleanup = Object.freeze({
      ...receipt,
      trackedScope: "harness-owned-buffers-only",
      auditedKernelControlBuffers:
        "owned-by-idempotent-audited-kernel-destroy-contract",
      destroyAllCalledTwice: true,
      idempotent: accountingExact,
      deviceDestroyed: true,
      intentionalDeviceLoss: Object.freeze({
        reason: intentionalDeviceLoss.reason,
        message: intentionalDeviceLoss.message,
        exactDestroyedReason: intentionalDeviceLoss.reason === "destroyed",
      }),
      uncapturedErrors: finalUncapturedErrors,
      runtimeErrors: finalRuntimeErrors,
      unexpectedDeviceLoss: finalUnexpectedDeviceLoss,
      heartbeatCoveredCleanup: true,
      heartbeat: responsiveness,
      epochs: Object.freeze({
        cleanupStartedAtEpochMilliseconds,
        harnessBuffersDestroyedAtEpochMilliseconds,
        deviceDestroyCalledAtEpochMilliseconds,
        deviceLossSettledAtEpochMilliseconds,
        postCleanupEventTurnsCompletedAtEpochMilliseconds,
        finalEventSnapshotAtEpochMilliseconds,
        postCleanupValidationAtEpochMilliseconds,
        heartbeatStoppedAtEpochMilliseconds,
      }),
    });
    rawDevice.removeEventListener("uncapturederror", onUncaptured);
    window.removeEventListener("error", onWindowError);
    window.removeEventListener("unhandledrejection", onUnhandledRejection);
    if (!postCleanupValidationClean) {
      postCleanupValidationFailure = new Error(
        "OPT-0011 production post-cleanup resource, runtime, or device-event validation failed",
      );
    }
  }
  if (postCleanupValidationFailure !== undefined) {
    throw postCleanupValidationFailure;
  }
  return Object.freeze({
    schema: "ace-opt-0011-production-fp16-conv1d-correctness-v1",
    status: "passed",
    experimentId: "OPT-0011",
    classification: "correctness-only-no-timing-or-thermal-claim",
    recordedAt: new Date().toISOString(),
    environment: Object.freeze({
      userAgent: navigator.userAgent,
      page: window.location.href,
      adapter: adapterReceipt(adapter, rawDevice),
    }),
    protocol: Object.freeze({
      requiredFeature: "shader-f16",
      productionKernel: "ace-vae-fp16-portable-conv1d-v1",
      k7Authority: "frozen-opt-0011-audited-scalar-oracle",
      k1Authority: OPT_0011_PRODUCTION_K1_SCALAR_ORACLE_ID,
      outputComparison: "full-selected-range-u16-or-u32-plus-k1-cpu",
      exactB256GraphShapesAndQuanta: true,
      oneOutstandingCommandBuffer: true,
      drainAndRealQueueEmptyTurnAfterEveryExecution: true,
      qNaNPrefillAndExternalPlusAdjacentGuards: true,
      compilationUploadAndWallTimeReported: false,
      performanceClaim: null,
      thermalClaim: null,
      qualityClaim: null,
      productionSelectorClaim: null,
    }),
    sourceAuthority: Object.freeze({
      ...identity,
      productionCoreSourceSha256: actualProductionSourceSha256,
      auditedCoreSourceSha256: actualAuditedSourceSha256,
      generatedShaderHashesFrozenAndVerifiedBeforeExecution: true,
    }),
    graphCoverage: Object.freeze({
      decoderInputFrames: B256_GRAPH.inputFrames,
      decoderOutputFrames: B256_GRAPH.outputFrames,
      biasedK1OperationCount: K1_GRAPH_OPERATIONS.length,
      distinctK1ShapeCount: new Set(K1_GRAPH_OPERATIONS.map(
        ({ shape }) => `${shape.inputFrames}:${shape.inputChannels}:${shape.outputChannels}`,
      )).size,
    }),
    cases: Object.freeze(cases),
    cancellation,
    responsiveness,
    cleanup,
  });
}

async function runFixture(
  device: GPUDevice,
  tracker: BufferTracker,
  fixtureValue: Opt0011ProductionConv1dFixture,
): Promise<unknown> {
  const prepared = await prepareFixture(device, tracker, fixtureValue);
  try {
    const authorityFirst = await executeAndRead(
      device,
      tracker,
      prepared.plan,
      fixtureValue.range,
      prepared.authority,
      prepared.authorityOutput,
      "first",
    );
    const productionFirst = await executeAndRead(
      device,
      tracker,
      prepared.plan,
      fixtureValue.range,
      prepared.production,
      prepared.productionOutput,
      "first",
    );
    const authorityRerun = await executeAndRead(
      device,
      tracker,
      prepared.plan,
      fixtureValue.range,
      prepared.authority,
      prepared.authorityOutput,
      "rerun",
    );
    const productionRerun = await executeAndRead(
      device,
      tracker,
      prepared.plan,
      fixtureValue.range,
      prepared.production,
      prepared.productionOutput,
      "rerun",
    );
    if (
      authorityFirst.readback.sha256 !== authorityRerun.readback.sha256 ||
      productionFirst.readback.sha256 !== productionRerun.readback.sha256
    ) {
      throw new Error(`${fixtureValue.id} output hash changed on deterministic rerun`);
    }
    const comparison = compareOpt0011ProductionRawBits(
      authorityFirst.readback.bits,
      productionFirst.readback.bits,
    );
    if (comparison.mismatchCount !== 0) {
      const first = comparison.firstMismatchIndex;
      if (first === null) {
        throw new Error(`${fixtureValue.id} mismatch diagnostic lost its first index`);
      }
      const global = fixtureValue.range.base + first;
      const cpu = fixtureValue.family === "k1"
        ? opt0011ProductionK1CpuBits(fixtureValue, global)
        : null;
      throw new Error(
        `${fixtureValue.id} has ${comparison.mismatchCount} authority/production bit mismatches; ` +
          `selectedIndex=${first}, globalOutputIndex=${global}, ` +
          `authorityBits=0x${hex(authorityFirst.readback.bits[first]!, fixtureValue.outputStorage)}, ` +
          `productionBits=0x${hex(productionFirst.readback.bits[first]!, fixtureValue.outputStorage)}, ` +
          `cpuBits=${cpu === null ? "not-applicable" : `0x${hex(cpu, "float16")}`}`,
      );
    }
    const cpuOracle = fixtureValue.family === "k1"
      ? compareK1ToCpu(
          fixtureValue,
          authorityFirst.readback.bits as Uint16Array,
          productionFirst.readback.bits as Uint16Array,
        )
      : Object.freeze({
          scope: "audited-k7-scalar",
          comparedOutputCount: fixtureValue.range.count,
          fullSelectedRangeBitAuthority: true,
        });
    const rangePlan = planAceFp16VaeConv1dRange(
      prepared.plan,
      fixtureValue.range,
    );
    return Object.freeze({
      id: fixtureValue.id,
      family: fixtureValue.family,
      graphOperationLabel: fixtureValue.graphOperationLabel,
      graphOperationOrdinal: fixtureValue.graphOperationOrdinal,
      graphFamilyMultiplicity: fixtureValue.graphFamilyMultiplicity,
      graphQuantumIndex: fixtureValue.graphQuantumIndex,
      graphOperationQuantumIndex: fixtureValue.graphOperationQuantumIndex,
      auditedRangeIndex: fixtureValue.auditedRangeIndex,
      shape: fixtureValue.shape,
      hasBias: fixtureValue.hasBias,
      outputStorage: fixtureValue.outputStorage,
      coverage: fixtureValue.coverage,
      range: Object.freeze({
        ...fixtureValue.range,
        firstOutputRow: rangePlan.firstOutputRow,
        outputRowCount: rangePlan.outputRowCount,
        workgroupsX: rangePlan.workgroupsX,
        workgroupsY: rangePlan.workgroupsY,
      }),
      plan: Object.freeze({
        family: prepared.plan.family,
        outputFrames: prepared.plan.outputFrames,
        outputElements: prepared.plan.outputElements,
        outputStorageBytes: prepared.plan.outputStorageBytes,
        outputBindingBytes: prepared.plan.outputBindingBytes,
        workgroupStorageBytes: prepared.plan.workgroupStorageBytes,
      }),
      uploadIdentity: prepared.uploadIdentity,
      sourceIdentity: prepared.sourceIdentity,
      outputPrefill: Object.freeze({
        authority: hex(
          prepared.authorityOutput.sentinelBits,
          fixtureValue.outputStorage,
        ),
        production: hex(
          prepared.productionOutput.sentinelBits,
          fixtureValue.outputStorage,
        ),
        independent: true,
        quietNaN: true,
      }),
      authority: executionReceipt(authorityFirst, authorityRerun),
      production: executionReceipt(productionFirst, productionRerun),
      fullSelectedRangeBitUnit:
        fixtureValue.outputStorage === "float16" ? "u16" : "u32",
      fullSelectedRangeBitCount: fixtureValue.range.count,
      fullSelectedRangeBitMismatchCount: comparison.mismatchCount,
      fullSelectedRangeBitIdentical: true,
      deterministicRerunHashes: true,
      cpuOracle,
      performanceClaim: null,
    });
  } finally {
    prepared.destroy();
    prepared.destroy();
  }
}

async function prepareFixture(
  device: GPUDevice,
  tracker: BufferTracker,
  fixtureValue: Opt0011ProductionConv1dFixture,
): Promise<PreparedFixture> {
  const plan = planAceFp16VaeConv1d(
    fixtureValue.shape,
    fixtureValue.outputStorage,
  );
  const owned: GPUBuffer[] = [];
  const productionKernel = AceFp16VaeConv1dKernel.create(device);
  let auditedKernel: AceOpt0011VaeConv1dFp16ScalarOracleKernel | undefined;
  let destroyed = false;
  try {
    const sources = await authenticateGeneratedSources(fixtureValue);
    const uploads = await createFixtureUploads(
      device,
      tracker,
      fixtureValue,
      plan,
    );
    owned.push(...uploads.buffers);
    const authorityOutput = createOutputTarget(
      device,
      tracker,
      `${fixtureValue.id}-authority-output`,
      plan,
      "authority",
    );
    const productionOutput = createOutputTarget(
      device,
      tracker,
      `${fixtureValue.id}-production-output`,
      plan,
      "production",
    );
    owned.push(authorityOutput.buffer, productionOutput.buffer);
    const control = createRangeControlBuffer(
      device,
      tracker,
      `${fixtureValue.id}-range-control`,
      [fixtureValue.range],
    );
    owned.push(control.buffer);
    const common = {
      input: binding(uploads.input, plan.inputBindingBytes),
      weight: binding(uploads.weight, plan.weightBindingBytes),
      ...(uploads.bias === undefined
        ? {}
        : { bias: binding(uploads.bias, plan.biasBindingBytes) }),
    };
    const productionBindings: AceFp16VaeConv1dBindings = {
      ...common,
      output: productionOutput.binding,
    };
    const productionDispatch = await productionKernel.createDispatch(
      `${fixtureValue.id}-production`,
      fixtureValue.shape,
      productionBindings,
      fixtureValue.outputStorage,
      control.bindings[0]!,
    );
    const production: GpuArmDispatch = Object.freeze({
      id: "production",
      label: `${fixtureValue.id}-production`,
      encode(pass: GPUComputePassEncoder): void {
        productionDispatch.encode(pass);
      },
    });
    let authority: GpuArmDispatch;
    if (fixtureValue.family === "k7") {
      auditedKernel =
        AceOpt0011VaeConv1dFp16ScalarOracleKernel.create(device);
      const auditedBindings: AceOpt0011VaeConv1dFp16Bindings = {
        ...common,
        output: authorityOutput.binding,
      };
      const auditedDispatch = await auditedKernel.createDispatch(
        `${fixtureValue.id}-audited-scalar`,
        fixtureValue.shape,
        auditedBindings,
        fixtureValue.outputStorage,
      );
      const rangeIndex = fixtureValue.auditedRangeIndex;
      if (rangeIndex === null) {
        throw new Error(`${fixtureValue.id} lost its audited K7 range index`);
      }
      assertAuditedRangeMatches(fixtureValue, auditedDispatch, rangeIndex);
      authority = Object.freeze({
        id: "authority",
        label: `${fixtureValue.id}-audited-scalar`,
        encode(pass: GPUComputePassEncoder): void {
          auditedDispatch.encodeRange(pass, rangeIndex);
        },
      });
    } else {
      const scalarBindings: AceFp16VaeConv1dBindings = {
        ...common,
        output: authorityOutput.binding,
      };
      authority = await createK1ScalarDispatch(
        device,
        fixtureValue,
        plan,
        scalarBindings,
        control.bindings[0]!,
        sources.authoritySource,
      );
    }
    return Object.freeze({
      fixture: fixtureValue,
      plan,
      authority,
      production,
      authorityOutput,
      productionOutput,
      uploadIdentity: uploads.identity,
      sourceIdentity: sources.identity,
      destroy(): void {
        productionKernel.destroy();
        auditedKernel?.destroy();
        if (destroyed) return;
        destroyed = true;
        for (const buffer of owned) tracker.destroy(buffer);
      },
    });
  } catch (error) {
    productionKernel.destroy();
    auditedKernel?.destroy();
    for (const buffer of owned) tracker.destroy(buffer);
    throw error;
  }
}

async function authenticateGeneratedSources(
  fixtureValue: Opt0011ProductionConv1dFixture,
): Promise<{
  readonly authoritySource: string;
  readonly identity: Readonly<Record<string, unknown>>;
}> {
  const productionSource = aceFp16VaeConv1dWgsl(
    fixtureValue.shape,
    fixtureValue.hasBias,
    fixtureValue.outputStorage,
  );
  const authoritySource = fixtureValue.family === "k7"
    ? aceOpt0011VaeConv1dFp16ScalarOracleWgsl(
        fixtureValue.shape,
        fixtureValue.hasBias,
        fixtureValue.outputStorage,
      )
    : opt0011ProductionK1ScalarOracleWgsl(fixtureValue.shape);
  const auditedPortableSource = fixtureValue.family === "k7"
    ? aceOpt0011VaeConv1dFp16PortableWorkgroupWgsl(
        fixtureValue.shape,
        fixtureValue.hasBias,
        fixtureValue.outputStorage,
      )
    : undefined;
  const [productionSha256, authoritySha256, auditedPortableSha256] =
    await Promise.all([
      sha256Hex(new TextEncoder().encode(productionSource)),
      sha256Hex(new TextEncoder().encode(authoritySource)),
      auditedPortableSource === undefined
        ? Promise.resolve(undefined)
        : sha256Hex(new TextEncoder().encode(auditedPortableSource)),
    ]);
  const expected =
    OPT_0011_PRODUCTION_CONV1D_GENERATED_SHADER_SHA256[fixtureValue.id];
  if (
    productionSha256 !== expected.production ||
    authoritySha256 !== expected.authority ||
    ("auditedPortable" in expected &&
      auditedPortableSha256 !== expected.auditedPortable)
  ) {
    throw new Error(`${fixtureValue.id} generated shader SHA-256 changed`);
  }
  return Object.freeze({
    authoritySource,
    identity: Object.freeze({
      productionCoreCommit: OPT_0011_PRODUCTION_CONV1D_CORE_COMMIT,
      productionCoreSourceSha256:
        OPT_0011_PRODUCTION_CONV1D_CORE_SOURCE_SHA256,
      auditedCoreCommit: OPT_0011_AUDITED_CONV1D_CORE_COMMIT,
      auditedCoreSourceSha256: OPT_0011_AUDITED_CONV1D_CORE_SOURCE_SHA256,
      productionShaderSha256: productionSha256,
      authorityShaderSha256: authoritySha256,
      ...(auditedPortableSha256 === undefined
        ? {}
        : { auditedPortableShaderSha256: auditedPortableSha256 }),
      productionShaderBytes: new TextEncoder().encode(productionSource).byteLength,
      authorityShaderBytes: new TextEncoder().encode(authoritySource).byteLength,
    }),
  });
}

async function createFixtureUploads(
  device: GPUDevice,
  tracker: BufferTracker,
  fixtureValue: Opt0011ProductionConv1dFixture,
  plan: AceFp16VaeConv1dPlan,
): Promise<{
  readonly input: GPUBuffer;
  readonly weight: GPUBuffer;
  readonly bias?: GPUBuffer;
  readonly buffers: readonly GPUBuffer[];
  readonly identity: Readonly<Record<string, unknown>>;
}> {
  const inputPayload = new Uint8Array(
    fixtureValue.inputRowCount * plan.inputChannels * 2,
  );
  for (let row = 0; row < fixtureValue.inputRowCount; row += 1) {
    const globalRow = fixtureValue.inputFirstRow + row;
    for (let channel = 0; channel < plan.inputChannels; channel += 1) {
      writeU16Le(
        inputPayload,
        (row * plan.inputChannels + channel) * 2,
        inputBitsUnchecked(
          fixtureValue,
          globalRow * plan.inputChannels + channel,
        ),
      );
    }
  }
  const weightPayload = new Uint8Array(plan.weightBindingBytes);
  for (let index = 0; index < plan.weightElements; index += 1) {
    writeU16Le(
      weightPayload,
      index * 2,
      weightBitsUnchecked(fixtureValue, index),
    );
  }
  const biasPayload = fixtureValue.hasBias
    ? new Uint8Array(plan.biasBindingBytes)
    : undefined;
  if (biasPayload !== undefined) {
    for (let channel = 0; channel < plan.outputChannels; channel += 1) {
      writeU16Le(
        biasPayload,
        channel * 2,
        biasBitsUnchecked(fixtureValue, channel),
      );
    }
  }
  const inputPayloadOffset = fixtureValue.inputFirstRow *
    plan.inputChannels * 2;
  const input = createSparseUploadBuffer(
    device,
    tracker,
    `${fixtureValue.id}-input`,
    plan.inputBindingBytes,
    inputPayloadOffset,
    inputPayload,
  );
  const weight = createSparseUploadBuffer(
    device,
    tracker,
    `${fixtureValue.id}-weight`,
    plan.weightBindingBytes,
    0,
    weightPayload,
  );
  const bias = biasPayload === undefined
    ? undefined
    : createSparseUploadBuffer(
        device,
        tracker,
        `${fixtureValue.id}-bias`,
        plan.biasBindingBytes,
        0,
        biasPayload,
      );
  const hashes = await Promise.all([
    sha256Hex(inputPayload),
    sha256Hex(weightPayload),
    biasPayload === undefined ? Promise.resolve(undefined) : sha256Hex(biasPayload),
  ]);
  return Object.freeze({
    input,
    weight,
    ...(bias === undefined ? {} : { bias }),
    buffers: Object.freeze([
      input,
      weight,
      ...(bias === undefined ? [] : [bias]),
    ]),
    identity: Object.freeze({
      recipe: "zero-filled-binding-plus-authenticated-active-little-endian-f16-payload-v1",
      input: Object.freeze({
        bindingBytes: plan.inputBindingBytes,
        activePayloadOffsetBytes: inputPayloadOffset,
        activePayloadBytes: inputPayload.byteLength,
        activePayloadSha256: hashes[0],
        zeroOutsideActivePayload: true,
      }),
      weight: Object.freeze({
        bindingBytes: plan.weightBindingBytes,
        activePayloadOffsetBytes: 0,
        activePayloadBytes: weightPayload.byteLength,
        activePayloadSha256: hashes[1],
      }),
      ...(biasPayload === undefined
        ? {}
        : {
            bias: Object.freeze({
              bindingBytes: plan.biasBindingBytes,
              activePayloadOffsetBytes: 0,
              activePayloadBytes: biasPayload.byteLength,
              activePayloadSha256: hashes[2],
            }),
          }),
    }),
  });
}

function createSparseUploadBuffer(
  device: GPUDevice,
  tracker: BufferTracker,
  label: string,
  bindingBytes: number,
  payloadOffset: number,
  payload: Uint8Array,
): GPUBuffer {
  if (
    payloadOffset < 0 || payloadOffset + payload.byteLength > bindingBytes ||
    payloadOffset % 2 !== 0
  ) {
    throw new RangeError(`${label} sparse upload is outside its binding`);
  }
  const buffer = tracker.create(device, {
    label,
    size: bindingBytes,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });
  try {
    const mapped = new Uint8Array(buffer.getMappedRange());
    mapped.fill(0);
    mapped.set(payload, payloadOffset);
    buffer.unmap();
    return buffer;
  } catch (error) {
    if (buffer.mapState === "mapped") buffer.unmap();
    tracker.destroy(buffer);
    throw error;
  }
}

function createRangeControlBuffer(
  device: GPUDevice,
  tracker: BufferTracker,
  label: string,
  ranges: readonly Readonly<{ readonly base: number; readonly count: number }>[],
): {
  readonly buffer: GPUBuffer;
  readonly stride: number;
  readonly bindings: readonly AceVaeOutputRangeBinding[];
} {
  const stride = Math.max(
    CONTROL_BYTES,
    Number(device.limits.minUniformBufferOffsetAlignment),
  );
  const size = (ranges.length - 1) * stride + CONTROL_BYTES;
  const buffer = tracker.create(device, {
    label,
    size,
    usage: GPUBufferUsage.UNIFORM,
    mappedAtCreation: true,
  });
  try {
    const mapped = buffer.getMappedRange();
    new Uint8Array(mapped).fill(0);
    for (const [index, range] of ranges.entries()) {
      const words = new Uint32Array(mapped, index * stride, 4);
      words[0] = range.base;
      words[1] = range.count;
    }
    buffer.unmap();
    return Object.freeze({
      buffer,
      stride,
      bindings: Object.freeze(ranges.map((range, index) => Object.freeze({
        base: range.base,
        count: range.count,
        control: Object.freeze({
          buffer,
          offset: index * stride,
          size: CONTROL_BYTES,
        }),
      }))),
    });
  } catch (error) {
    if (buffer.mapState === "mapped") buffer.unmap();
    tracker.destroy(buffer);
    throw error;
  }
}

async function createK1ScalarDispatch(
  device: GPUDevice,
  fixtureValue: Opt0011ProductionConv1dFixture,
  plan: AceFp16VaeConv1dPlan,
  bindings: AceFp16VaeConv1dBindings,
  control: AceVaeOutputRangeBinding,
  source: string,
): Promise<K1ScalarDispatch> {
  if (bindings.bias === undefined) {
    throw new Error(`${fixtureValue.id} K1 scalar authority requires bias`);
  }
  const label = `${fixtureValue.id}-k1-scalar-authority`;
  const module = device.createShaderModule({ label, code: source });
  const compilation = await module.getCompilationInfo();
  const errors = compilation.messages.filter(({ type }) => type === "error");
  if (errors.length > 0) {
    throw new Error(
      `${label} WGSL failed: ${errors.map((message) =>
        `${message.lineNum}:${message.linePos} ${message.message}`
      ).join("; ")}`,
    );
  }
  const dataEntries: GPUBindGroupLayoutEntry[] = [
    plan.inputBindingBytes,
    plan.weightBindingBytes,
    plan.biasBindingBytes,
    plan.outputBindingBytes,
  ].map((minBindingSize, bindingIndex) => ({
    binding: bindingIndex,
    visibility: GPUShaderStage.COMPUTE,
    buffer: {
      type: bindingIndex === 3 ? "storage" : "read-only-storage",
      minBindingSize,
    },
  }));
  const layout = device.createBindGroupLayout({
    label: `${label}-bindings`,
    entries: [
      ...dataEntries,
      {
        binding: 4,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: "uniform",
          hasDynamicOffset: true,
          minBindingSize: CONTROL_BYTES,
        },
      },
    ],
  });
  const pipelineLayout = device.createPipelineLayout({
    label: `${label}-layout`,
    bindGroupLayouts: [layout],
  });
  const pipeline = await device.createComputePipelineAsync({
    label,
    layout: pipelineLayout,
    compute: { module, entryPoint: "main" },
  });
  const bindGroup = device.createBindGroup({
    label: `${label}-bind-group`,
    layout,
    entries: [
      { binding: 0, resource: bindings.input },
      { binding: 1, resource: bindings.weight },
      { binding: 2, resource: bindings.bias },
      { binding: 3, resource: bindings.output },
      {
        binding: 4,
        resource: {
          buffer: control.control.buffer,
          offset: 0,
          size: CONTROL_BYTES,
        },
      },
    ],
  });
  const dynamicOffset = control.control.offset ?? 0;
  const workgroups = Math.ceil(fixtureValue.range.count / K1_SCALAR_WORKGROUP_SIZE);
  return Object.freeze({
    id: "authority",
    label,
    encode(pass: GPUComputePassEncoder): void {
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup, [dynamicOffset]);
      pass.dispatchWorkgroups(workgroups, 1, 1);
    },
  });
}

function createOutputTarget(
  device: GPUDevice,
  tracker: BufferTracker,
  label: string,
  plan: AceFp16VaeConv1dPlan,
  id: OutputTarget["id"],
): OutputTarget {
  const size = OUTPUT_GUARD_BYTES + plan.outputBindingBytes + OUTPUT_GUARD_BYTES;
  const buffer = tracker.create(device, {
    label,
    size,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  const sentinelBits = plan.outputStorage === "float16"
    ? id === "authority" ? AUTHORITY_F16_SENTINEL : PRODUCTION_F16_SENTINEL
    : id === "authority" ? AUTHORITY_F32_SENTINEL : PRODUCTION_F32_SENTINEL;
  const sentinelWord = plan.outputStorage === "float16"
    ? sentinelBits | (sentinelBits << 16)
    : sentinelBits;
  try {
    const words = new Uint32Array(buffer.getMappedRange());
    words.fill(OUTPUT_GUARD_WORD);
    words.fill(
      sentinelWord,
      OUTPUT_GUARD_BYTES / 4,
      (OUTPUT_GUARD_BYTES + plan.outputBindingBytes) / 4,
    );
    buffer.unmap();
    return Object.freeze({
      id,
      buffer,
      binding: Object.freeze({
        buffer,
        offset: OUTPUT_GUARD_BYTES,
        size: plan.outputBindingBytes,
      }),
      sentinelBits,
      sentinelWord,
    });
  } catch (error) {
    if (buffer.mapState === "mapped") buffer.unmap();
    tracker.destroy(buffer);
    throw error;
  }
}

function prefillSelectedRange(
  device: GPUDevice,
  plan: AceFp16VaeConv1dPlan,
  range: Readonly<{ readonly base: number; readonly count: number }>,
  target: OutputTarget,
): void {
  const elementBytes = plan.outputStorage === "float16" ? 2 : 4;
  const byteOffset = OUTPUT_GUARD_BYTES + range.base * elementBytes;
  const byteLength = align4(range.count * elementBytes);
  if (byteOffset % 4 !== 0) {
    throw new RangeError(`${target.id} selected output offset is not copy aligned`);
  }
  const words = new Uint32Array(byteLength / 4);
  words.fill(target.sentinelWord);
  device.queue.writeBuffer(target.buffer, byteOffset, words);
}

async function executeAndRead(
  device: GPUDevice,
  tracker: BufferTracker,
  plan: AceFp16VaeConv1dPlan,
  range: Readonly<{ readonly base: number; readonly count: number }>,
  dispatch: GpuArmDispatch,
  output: OutputTarget,
  runLabel: string,
): Promise<{
  readonly execution: ExecutionCounts;
  readonly readback: SelectedOutputReadback;
}> {
  prefillSelectedRange(device, plan, range, output);
  await device.queue.onSubmittedWorkDone();
  const encoder = device.createCommandEncoder({
    label: `${dispatch.label}-${runLabel}-encoder`,
  });
  const pass = encoder.beginComputePass({
    label: `${dispatch.label}-${runLabel}-pass`,
  });
  dispatch.encode(pass);
  pass.end();
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  await queueEmptyIdleTurn();
  const execution = Object.freeze({
    encodedCommandBuffers: 1,
    submissions: 1,
    drains: 1,
    dispatches: 1,
    queueEmptyIdleTurns: 1,
  } as const);
  const readback = await readSelectedOutput(
    device,
    tracker,
    plan,
    range,
    output,
  );
  return Object.freeze({ execution, readback });
}

async function readSelectedOutput(
  device: GPUDevice,
  tracker: BufferTracker,
  plan: AceFp16VaeConv1dPlan,
  selected: Readonly<{ readonly base: number; readonly count: number }>,
  target: OutputTarget,
): Promise<SelectedOutputReadback> {
  const elementBytes = plan.outputStorage === "float16" ? 2 : 4;
  const selectedRawBytes = selected.count * elementBytes;
  const selectedCopyBytes = align4(selectedRawBytes);
  const selectedSourceOffset = OUTPUT_GUARD_BYTES + selected.base * elementBytes;
  if (selectedSourceOffset % 4 !== 0) {
    throw new RangeError(`${target.id} selected readback offset is not copy aligned`);
  }
  const selectedEnd = selected.base * elementBytes + selectedCopyBytes;
  const beforeBytes = selected.base * elementBytes >= OUTPUT_CANARY_BYTES
    ? OUTPUT_CANARY_BYTES
    : 0;
  const remainingAfter = plan.outputBindingBytes - selectedEnd;
  const afterBytes = remainingAfter >= OUTPUT_CANARY_BYTES
    ? OUTPUT_CANARY_BYTES
    : 0;
  const prefixOffset = 0;
  const suffixOffset = prefixOffset + OUTPUT_GUARD_BYTES;
  const selectedOffset = suffixOffset + OUTPUT_GUARD_BYTES;
  const beforeOffset = selectedOffset + selectedCopyBytes;
  const afterOffset = beforeOffset + beforeBytes;
  const totalBytes = afterOffset + afterBytes;
  const readback = tracker.create(device, {
    label: `${target.id}-selected-readback`,
    size: totalBytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  let mapped = false;
  try {
    const encoder = device.createCommandEncoder({
      label: `${target.id}-selected-readback-encoder`,
    });
    encoder.copyBufferToBuffer(
      target.buffer,
      0,
      readback,
      prefixOffset,
      OUTPUT_GUARD_BYTES,
    );
    encoder.copyBufferToBuffer(
      target.buffer,
      OUTPUT_GUARD_BYTES + plan.outputBindingBytes,
      readback,
      suffixOffset,
      OUTPUT_GUARD_BYTES,
    );
    encoder.copyBufferToBuffer(
      target.buffer,
      selectedSourceOffset,
      readback,
      selectedOffset,
      selectedCopyBytes,
    );
    if (beforeBytes > 0) {
      encoder.copyBufferToBuffer(
        target.buffer,
        selectedSourceOffset - beforeBytes,
        readback,
        beforeOffset,
        beforeBytes,
      );
    }
    if (afterBytes > 0) {
      encoder.copyBufferToBuffer(
        target.buffer,
        OUTPUT_GUARD_BYTES + selectedEnd,
        readback,
        afterOffset,
        afterBytes,
      );
    }
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    mapped = true;
    const mappedRange = readback.getMappedRange();
    const prefix = new Uint32Array(
      mappedRange,
      prefixOffset,
      OUTPUT_GUARD_BYTES / 4,
    );
    const suffix = new Uint32Array(
      mappedRange,
      suffixOffset,
      OUTPUT_GUARD_BYTES / 4,
    );
    const externalGuardsUntouched = [...prefix, ...suffix].every(
      (word) => word === OUTPUT_GUARD_WORD,
    );
    const before = beforeBytes === 0
      ? new Uint32Array(0)
      : new Uint32Array(mappedRange, beforeOffset, beforeBytes / 4);
    const after = afterBytes === 0
      ? new Uint32Array(0)
      : new Uint32Array(mappedRange, afterOffset, afterBytes / 4);
    const adjacentCanariesUntouched = [...before, ...after].every(
      (word) => word === target.sentinelWord,
    );
    const raw = new Uint8Array(
      mappedRange,
      selectedOffset,
      selectedRawBytes,
    ).slice();
    const bits = plan.outputStorage === "float16"
      ? new Uint16Array(raw.buffer, raw.byteOffset, selected.count)
      : new Uint32Array(raw.buffer, raw.byteOffset, selected.count);
    const padding = new Uint8Array(
      mappedRange,
      selectedOffset + selectedRawBytes,
      selectedCopyBytes - selectedRawBytes,
    );
    const sentinelBytes = new Uint8Array(new Uint32Array([
      target.sentinelWord,
    ]).buffer);
    const copiedAlignmentPaddingUntouched = [...padding].every(
      (value, index) => value === sentinelBytes[index % 4],
    );
    let finiteCount = 0;
    let nonFiniteCount = 0;
    let sentinelCount = 0;
    let saturationCount = 0;
    let positiveZeroCount = 0;
    let negativeZeroCount = 0;
    let subnormalCount = 0;
    for (let index = 0; index < bits.length; index += 1) {
      const rawBits = bits[index]!;
      const value = plan.outputStorage === "float16"
        ? float16BitsToNumber(rawBits)
        : float32FromBits(rawBits);
      if (Number.isFinite(value)) finiteCount += 1;
      else nonFiniteCount += 1;
      if (rawBits === target.sentinelBits) sentinelCount += 1;
      if (
        plan.outputStorage === "float16"
          ? (rawBits & 0x7fff) === 0x7bff
          : (rawBits & 0x7fff_ffff) === 0x7f7f_ffff
      ) saturationCount += 1;
      if (value === 0) {
        const negative = plan.outputStorage === "float16"
          ? (rawBits & 0x8000) !== 0
          : (rawBits & 0x8000_0000) !== 0;
        if (negative) negativeZeroCount += 1;
        else positiveZeroCount += 1;
      }
      if (
        plan.outputStorage === "float16" &&
        (rawBits & 0x7c00) === 0 && (rawBits & 0x03ff) !== 0
      ) subnormalCount += 1;
    }
    if (
      !externalGuardsUntouched || !adjacentCanariesUntouched ||
      !copiedAlignmentPaddingUntouched || finiteCount !== selected.count ||
      nonFiniteCount !== 0 || sentinelCount !== 0 || saturationCount !== 0
    ) {
      throw new Error(
        `${target.id} selected output failed finite qNaN/guard/canary checks`,
      );
    }
    const retained = plan.outputStorage === "float16"
      ? new Uint16Array(bits)
      : new Uint32Array(bits);
    return Object.freeze({
      bits: retained,
      sha256: await sha256Hex(raw),
      scan: Object.freeze({
        selectedElementCount: selected.count,
        finiteCount,
        nonFiniteCount,
        sentinelCount,
        saturationCount,
        positiveZeroCount,
        negativeZeroCount,
        subnormalCount,
        externalPrefixGuardUntouched: externalGuardsUntouched,
        externalSuffixGuardUntouched: externalGuardsUntouched,
        adjacentBeforeCanaryBytes: beforeBytes,
        adjacentAfterCanaryBytes: afterBytes,
        adjacentCanariesUntouched,
        copiedAlignmentPaddingBytes: padding.byteLength,
        copiedAlignmentPaddingUntouched,
        completeSelectedRangeReadback: true,
      }),
    });
  } finally {
    if (mapped) readback.unmap();
    tracker.destroy(readback);
  }
}

function compareK1ToCpu(
  fixtureValue: Opt0011ProductionConv1dFixture,
  authority: Uint16Array,
  production: Uint16Array,
): unknown {
  const rowPeriod = fixtureValue.id === "k1-arithmetic-b1-f17-c65-c9"
    ? 1
    : 16;
  const outputChannels = fixtureValue.shape.outputChannels;
  const periodicCpuBits = new Uint16Array(rowPeriod * outputChannels);
  for (let row = 0; row < rowPeriod; row += 1) {
    for (let outputChannel = 0; outputChannel < outputChannels; outputChannel += 1) {
      periodicCpuBits[row * outputChannels + outputChannel] =
        opt0011ProductionK1CpuBits(
          fixtureValue,
          row * outputChannels + outputChannel,
        );
    }
  }
  let authorityMismatchCount = 0;
  let productionMismatchCount = 0;
  let firstAuthorityMismatch: number | null = null;
  let firstProductionMismatch: number | null = null;
  let positiveZeroCount = 0;
  let negativeZeroCount = 0;
  let subnormalCount = 0;
  for (let localIndex = 0; localIndex < fixtureValue.range.count; localIndex += 1) {
    const globalIndex = fixtureValue.range.base + localIndex;
    const outputChannel = globalIndex % outputChannels;
    const outputRow = Math.floor(globalIndex / outputChannels);
    const expected = periodicCpuBits[
      (outputRow % rowPeriod) * outputChannels + outputChannel
    ]!;
    if (authority[localIndex] !== expected) {
      authorityMismatchCount += 1;
      firstAuthorityMismatch ??= localIndex;
    }
    if (production[localIndex] !== expected) {
      productionMismatchCount += 1;
      firstProductionMismatch ??= localIndex;
    }
    if ((expected & 0x7fff) === 0) {
      if ((expected & 0x8000) === 0) positiveZeroCount += 1;
      else negativeZeroCount += 1;
    }
    if ((expected & 0x7c00) === 0 && (expected & 0x03ff) !== 0) {
      subnormalCount += 1;
    }
  }
  if (authorityMismatchCount !== 0 || productionMismatchCount !== 0) {
    throw new Error(
      `${fixtureValue.id} K1 CPU oracle mismatch: ` +
        `authority=${authorityMismatchCount}@${String(firstAuthorityMismatch)}, ` +
        `production=${productionMismatchCount}@${String(firstProductionMismatch)}`,
    );
  }
  return Object.freeze({
    arithmeticOrder: "increasing-input-channel",
    operands: "exact-f16-bits-expanded-to-f32",
    accumulation: "f32-source-order",
    outputBoundary: "exact-rne-f16-store-including-signed-zero",
    scope: fixtureValue.cpuOracleScope,
    comparedOutputCount: fixtureValue.range.count,
    selectedRangeOutputCount: fixtureValue.range.count,
    periodicInputRowCount: rowPeriod,
    periodicCpuOutputBits: periodicCpuBits.length,
    fullSelectedRangeComparison: true,
    authorityMismatchCount,
    productionMismatchCount,
    positiveZeroCount,
    negativeZeroCount,
    subnormalCount,
  });
}

async function runCancellationProof(
  device: GPUDevice,
  tracker: BufferTracker,
): Promise<unknown> {
  const operation = graphOperation("block-0-res-1-conv2");
  const quanta = graphOperationQuanta(operation.label);
  if (quanta.length < 2) {
    throw new Error("OPT-0011 cancellation proof requires two real graph ranges");
  }
  const ranges = quanta.slice(0, 2).map(quantumRange);
  const template = OPT_0011_PRODUCTION_CONV1D_CASES.find(
    ({ id }) => id === "k1-block0-tail",
  );
  if (template === undefined) throw new Error("Missing K1 block-0 fixture");
  const firstRow = ranges[0]!.base / operation.shape.outputChannels;
  const rowCount = ranges.reduce(
    (sum, range) => sum + range.count / operation.shape.outputChannels,
    0,
  );
  const fixtureValue = Object.freeze({
    ...template,
    range: ranges[0]!,
    inputFirstRow: firstRow,
    inputRowCount: rowCount,
  });
  const plan = planAceFp16VaeConv1d(operation.shape, "float16");
  const owned: GPUBuffer[] = [];
  const kernel = AceFp16VaeConv1dKernel.create(device);
  let destroyed = false;
  try {
    const uploads = await createFixtureUploads(
      device,
      tracker,
      fixtureValue,
      plan,
    );
    owned.push(...uploads.buffers);
    const output = createOutputTarget(
      device,
      tracker,
      "opt-0011-production-cancellation-output",
      plan,
      "production",
    );
    owned.push(output.buffer);
    const control = createRangeControlBuffer(
      device,
      tracker,
      "opt-0011-production-cancellation-controls",
      ranges,
    );
    owned.push(control.buffer);
    const common: AceFp16VaeConv1dBindings = {
      input: binding(uploads.input, plan.inputBindingBytes),
      weight: binding(uploads.weight, plan.weightBindingBytes),
      bias: binding(uploads.bias!, plan.biasBindingBytes),
      output: output.binding,
    };
    const dispatches: AceFp16VaeConv1dDispatch[] = [];
    for (const [index, range] of ranges.entries()) {
      dispatches.push(await kernel.createDispatch(
        `opt-0011-production-cancel-${index}`,
        operation.shape,
        common,
        "float16",
        control.bindings[index]!,
      ));
      const planned = planAceFp16VaeConv1dRange(plan, range);
      if (
        dispatches[index]!.outputRange.base !== planned.base ||
        dispatches[index]!.outputRange.count !== planned.count
      ) {
        throw new Error("OPT-0011 cancellation dispatch range changed");
      }
    }
    prefillSelectedRange(device, plan, ranges[0]!, output);
    await device.queue.onSubmittedWorkDone();
    const controller = new AbortController();
    let encodeCount = 0;
    let submitCount = 0;
    let drainCount = 0;
    let readbackCount = 0;
    let skippedRangeCount = 0;
    let idleTurnDelivered = false;
    for (const [index, dispatch] of dispatches.entries()) {
      if (controller.signal.aborted) {
        skippedRangeCount += 1;
        continue;
      }
      const encoder = device.createCommandEncoder({
        label: `opt-0011-production-cancel-${index}-encoder`,
      });
      const pass = encoder.beginComputePass();
      dispatch.encode(pass);
      pass.end();
      encodeCount += 1;
      device.queue.submit([encoder.finish()]);
      submitCount += 1;
      await device.queue.onSubmittedWorkDone();
      drainCount += 1;
      await queueEmptyIdleTurn();
      idleTurnDelivered = true;
      controller.abort("cancel-after-first-drained-range-and-real-idle");
    }
    if (!controller.signal.aborted) {
      readbackCount += 1;
      await readSelectedOutput(device, tracker, plan, ranges[0]!, output);
    }
    if (
      !controller.signal.aborted || !idleTurnDelivered || encodeCount !== 1 ||
      submitCount !== 1 || drainCount !== 1 || readbackCount !== 0 ||
      skippedRangeCount !== 1
    ) {
      throw new Error("OPT-0011 production cancellation did not stop later work");
    }
    return Object.freeze({
      fixtureId: "k1-block0-first-two-graph-quanta",
      graphOperationLabel: operation.label,
      plannedRangeCount: ranges.length,
      plannedRanges: Object.freeze(ranges),
      cancellationPoint: "after-first-drained-range-and-real-queue-empty-idle",
      encodeCount,
      submitCount,
      drainCount,
      readbackCount,
      skippedRangeCount,
      signalAborted: true,
      realQueueEmptyIdleTurnDelivered: true,
      laterEncodingPrevented: true,
      laterSubmissionPrevented: true,
      readbackPrevented: true,
    });
  } finally {
    kernel.destroy();
    kernel.destroy();
    if (!destroyed) {
      destroyed = true;
      for (const buffer of owned) tracker.destroy(buffer);
    }
  }
}

function graphFixture(
  id: Opt0011ProductionConv1dCaseId,
  operationLabel: string,
  selector: "first" | "middle" | "last",
  coverage: readonly string[],
): Opt0011ProductionConv1dFixture {
  const operation = graphOperation(operationLabel);
  const operationOrdinal = B256_GRAPH.operations.indexOf(operation);
  const quanta = graphOperationQuanta(operationLabel);
  const operationQuantumIndex = selector === "first"
    ? 0
    : selector === "middle"
    ? Math.floor(quanta.length / 2)
    : quanta.length - 1;
  const quantum = quanta[operationQuantumIndex];
  if (quantum === undefined) {
    throw new Error(`${operationLabel} has no ${selector} graph quantum`);
  }
  const range = quantumRange(quantum);
  const hasBias = operation.bias !== undefined;
  const outputStorage = hasBias ? "float16" : "float32";
  const plan = planAceFp16VaeConv1d(operation.shape, outputStorage);
  const rangePlan = planAceFp16VaeConv1dRange(plan, range);
  const inputStart = Math.max(
    0,
    rangePlan.firstOutputTime - operation.shape.padding,
  );
  const inputEnd = Math.min(
    operation.shape.inputFrames,
    rangePlan.firstOutputTime + rangePlan.outputRowCount +
      operation.shape.padding,
  );
  const inputFirstRow = rangePlan.batch * operation.shape.inputFrames + inputStart;
  const inputRowCount = inputEnd - inputStart;
  const family = operation.shape.kernelSize === 1 ? "k1" : "k7";
  const auditedRangeIndex = family === "k7"
    ? findAuditedRangeIndex(operation.shape, outputStorage, range)
    : null;
  const graphFamilyMultiplicity = family === "k1"
    ? K1_GRAPH_OPERATIONS.filter(({ shape }) => sameShape(shape, operation.shape)).length
    : B256_GRAPH.operations.filter((candidate) =>
        candidate.kind === "conv1d" &&
        candidate.shape.kernelSize === 7 &&
        sameShape(candidate.shape, operation.shape)
      ).length;
  return Object.freeze({
    id,
    family,
    graphOperationLabel: operation.label,
    graphOperationOrdinal: operationOrdinal,
    graphFamilyMultiplicity,
    shape: operation.shape,
    hasBias,
    outputStorage,
    range,
    graphQuantumIndex: quantum.index,
    graphOperationQuantumIndex: operationQuantumIndex,
    auditedRangeIndex,
    inputFirstRow,
    inputRowCount,
    cpuOracleScope: family === "k1"
      ? "complete-selected-range"
      : "audited-k7-scalar",
    coverage: Object.freeze([...coverage]),
  });
}

function arithmeticFixture(): Opt0011ProductionConv1dFixture {
  const shape = Object.freeze({
    batch: 1,
    inputFrames: 17,
    inputChannels: 65,
    outputChannels: 9,
    kernelSize: 1,
    stride: 1,
    dilation: 1,
    padding: 0,
  });
  const plan = planAceFp16VaeConv1d(shape, "float16");
  return Object.freeze({
    id: "k1-arithmetic-b1-f17-c65-c9",
    family: "k1",
    graphOperationLabel: null,
    graphOperationOrdinal: null,
    graphFamilyMultiplicity: null,
    shape,
    hasBias: true,
    outputStorage: "float16",
    range: Object.freeze({ base: 0, count: plan.outputElements }),
    graphQuantumIndex: null,
    graphOperationQuantumIndex: null,
    auditedRangeIndex: null,
    inputFirstRow: 0,
    inputRowCount: shape.inputFrames,
    cpuOracleScope: "complete-selected-range",
    coverage: Object.freeze([
      "biased-k1",
      "cin-64-tail",
      "cout-tail",
      "odd-f16-binding-padding",
      "positive-and-negative-zero",
      "f16-subnormal",
      "cancellation",
      "round-to-nearest-ties-to-even",
    ]),
  });
}

function graphOperation(label: string): AceVaeDecoderConvOperation {
  const operation = B256_GRAPH.operations.find(
    (candidate): candidate is AceVaeDecoderConvOperation =>
      candidate.kind === "conv1d" && candidate.label === label,
  );
  if (operation === undefined) {
    throw new Error(`OPT-0011 B-256 graph operation ${label} is missing`);
  }
  return operation;
}

function graphOperationQuanta(label: string): readonly AceVaeDecoderQuantumPlan[] {
  const quanta = B256_QUANTA.quanta.filter(
    ({ operationLabel }) => operationLabel === label,
  );
  if (quanta.length === 0) {
    throw new Error(`OPT-0011 B-256 graph operation ${label} has no quanta`);
  }
  return Object.freeze(quanta);
}

function quantumRange(
  quantum: AceVaeDecoderQuantumPlan,
): Readonly<{ readonly base: number; readonly count: number }> {
  if (quantum.primitives.length !== 1) {
    throw new Error(`${quantum.id} Conv1D quantum is not one primitive`);
  }
  const primitive = quantum.primitives[0]!;
  if (
    primitive.outputBase !== quantum.logicalOutputBase ||
    primitive.outputCount !== quantum.logicalOutputCount
  ) {
    throw new Error(`${quantum.id} Conv1D physical/logical range changed`);
  }
  return Object.freeze({
    base: primitive.outputBase,
    count: primitive.outputCount,
  });
}

function findAuditedRangeIndex(
  shape: AceVaeConv1dShape,
  outputStorage: AceFp16VaeConv1dOutputStorage,
  range: Readonly<{ readonly base: number; readonly count: number }>,
): number {
  const plan = planAceOpt0011VaeConv1dFp16(shape, outputStorage);
  const index = plan.outputRanges.findIndex(
    ({ firstOutput, outputCount }) =>
      firstOutput === range.base && outputCount === range.count,
  );
  if (index < 0) {
    throw new Error("OPT-0011 graph quantum is absent from audited K7 topology");
  }
  return index;
}

function assertAuditedRangeMatches(
  fixtureValue: Opt0011ProductionConv1dFixture,
  dispatch: AceOpt0011VaeConv1dFp16Dispatch,
  rangeIndex: number,
): void {
  const range = dispatch.plan.outputRanges[rangeIndex];
  if (
    range === undefined || range.firstOutput !== fixtureValue.range.base ||
    range.outputCount !== fixtureValue.range.count
  ) {
    throw new Error(`${fixtureValue.id} audited range topology changed`);
  }
}

function sameShape(left: AceVaeConv1dShape, right: AceVaeConv1dShape): boolean {
  return left.batch === right.batch &&
    left.inputFrames === right.inputFrames &&
    left.inputChannels === right.inputChannels &&
    left.outputChannels === right.outputChannels &&
    left.kernelSize === right.kernelSize && left.stride === right.stride &&
    left.dilation === right.dilation && left.padding === right.padding;
}

function arithmeticWeightBits(inputChannels: number, weightIndex: number): number {
  const outputChannel = Math.floor(weightIndex / inputChannels);
  const inputChannel = weightIndex % inputChannels;
  switch (outputChannel) {
    case 0: {
      const input = [0x3c00, 0x7bff, 0xfbff, 0x1000, 0x1400][inputChannel] ??
        0x0000;
      return (input & 0x8000) === 0 ? 0x8000 : 0x0000;
    }
    case 1: {
      const input = [0x3c00, 0x7bff, 0xfbff, 0x1000, 0x1400][inputChannel] ??
        0x0000;
      return (input & 0x8000) === 0 ? 0x0000 : 0x8000;
    }
    case 2: return inputChannel === 0 ? 0x0001 : 0x0000;
    case 3:
      return inputChannel === 1 || inputChannel === 2 || inputChannel === 4
        ? 0x3c00
        : 0x0000;
    case 4:
    case 5: return inputChannel === 3 ? 0x3c00 : 0x0000;
    case 6: return inputChannel === 0 ? 0x0001 : 0x0000;
    case 7: return inputChannel === 0 ? 0x03ff : 0x0000;
    case 8:
      return inputChannel === 1 || inputChannel === 2 ? 0x3c00 : 0x0000;
    default: return 0x0000;
  }
}

function executionReceipt(
  first: Awaited<ReturnType<typeof executeAndRead>>,
  rerun: Awaited<ReturnType<typeof executeAndRead>>,
): unknown {
  return Object.freeze({
    firstExecution: first.execution,
    rerunExecution: rerun.execution,
    firstScan: first.readback.scan,
    rerunScan: rerun.readback.scan,
    firstSha256: first.readback.sha256,
    rerunSha256: rerun.readback.sha256,
    deterministic: first.readback.sha256 === rerun.readback.sha256,
  });
}

function requiredDeviceLimits(): Record<string, number> {
  let maximumBinding = 0;
  let maximumBuffer = 0;
  let maximumStorage = 0;
  let maximumDispatchX = K1_SCALAR_WORKGROUP_SIZE;
  for (const fixtureValue of OPT_0011_PRODUCTION_CONV1D_CASES) {
    const plan = planAceFp16VaeConv1d(
      fixtureValue.shape,
      fixtureValue.outputStorage,
    );
    maximumBinding = Math.max(
      maximumBinding,
      plan.inputBindingBytes,
      plan.weightBindingBytes,
      plan.biasBindingBytes,
      plan.outputBindingBytes,
    );
    maximumBuffer = Math.max(
      maximumBuffer,
      maximumBinding,
      OUTPUT_GUARD_BYTES * 2 + plan.outputBindingBytes,
    );
    maximumStorage = Math.max(maximumStorage, plan.workgroupStorageBytes);
    maximumDispatchX = Math.max(
      maximumDispatchX,
      Math.ceil(fixtureValue.range.count / K1_SCALAR_WORKGROUP_SIZE),
    );
  }
  return {
    maxBufferSize: maximumBuffer,
    maxStorageBufferBindingSize: maximumBinding,
    maxComputeInvocationsPerWorkgroup: 128,
    maxComputeWorkgroupSizeX: 64,
    maxComputeWorkgroupSizeY: 8,
    maxComputeWorkgroupStorageSize: maximumStorage,
    maxComputeWorkgroupsPerDimension: maximumDispatchX,
  };
}

function requireAdapter(adapter: GPUAdapter): void {
  if (!adapter.features.has("shader-f16")) {
    throw new Error("OPT-0011 production gate requires adapter feature shader-f16");
  }
  for (const [name, value] of Object.entries(requiredDeviceLimits())) {
    const actual = Number(adapter.limits[name as keyof GPUSupportedLimits]);
    if (!Number.isFinite(actual) || actual < value) {
      throw new Error(
        `OPT-0011 production adapter limit ${name} is ${actual}, requires ${value}`,
      );
    }
  }
}

function adapterReceipt(adapter: GPUAdapter, device: GPUDevice): unknown {
  return Object.freeze({
    features: Object.freeze([...adapter.features].sort()),
    info: Object.freeze({
      vendor: adapter.info.vendor,
      architecture: adapter.info.architecture,
      device: adapter.info.device,
      description: adapter.info.description,
      subgroupMinSize: adapter.info.subgroupMinSize,
      subgroupMaxSize: adapter.info.subgroupMaxSize,
      isFallbackAdapter: adapter.info.isFallbackAdapter,
    }),
    requiredLimits: Object.freeze(requiredDeviceLimits()),
    actualAdapterLimits: gpuLimitReceipt(adapter.limits),
    actualRequestedDeviceLimits: gpuLimitReceipt(device.limits),
  });
}

function gpuLimitReceipt(limits: GPUSupportedLimits): unknown {
  return Object.freeze({
    maxBufferSize: Number(limits.maxBufferSize),
    maxStorageBufferBindingSize: Number(limits.maxStorageBufferBindingSize),
    maxComputeInvocationsPerWorkgroup: limits.maxComputeInvocationsPerWorkgroup,
    maxComputeWorkgroupSizeX: limits.maxComputeWorkgroupSizeX,
    maxComputeWorkgroupSizeY: limits.maxComputeWorkgroupSizeY,
    maxComputeWorkgroupStorageSize: limits.maxComputeWorkgroupStorageSize,
    maxComputeWorkgroupsPerDimension: limits.maxComputeWorkgroupsPerDimension,
    minStorageBufferOffsetAlignment: limits.minStorageBufferOffsetAlignment,
    minUniformBufferOffsetAlignment: limits.minUniformBufferOffsetAlignment,
  });
}

function startHeartbeat(): { stop(): Readonly<Record<string, unknown>> } {
  const startedAtEpochMilliseconds = Date.now();
  let stopped = false;
  let animationFrameCount = 0;
  let timerTickCount = 0;
  let maximumAnimationFrameGapMilliseconds = 0;
  let maximumTimerGapMilliseconds = 0;
  let lastAnimationFrame = performance.now();
  let lastTimer = performance.now();
  let animationFrame = 0;
  const animate = (now: number): void => {
    if (stopped) return;
    maximumAnimationFrameGapMilliseconds = Math.max(
      maximumAnimationFrameGapMilliseconds,
      now - lastAnimationFrame,
    );
    lastAnimationFrame = now;
    animationFrameCount += 1;
    animationFrame = requestAnimationFrame(animate);
  };
  animationFrame = requestAnimationFrame(animate);
  const timer = window.setInterval(() => {
    const now = performance.now();
    maximumTimerGapMilliseconds = Math.max(
      maximumTimerGapMilliseconds,
      now - lastTimer,
    );
    lastTimer = now;
    timerTickCount += 1;
  }, 16);
  let receipt: Readonly<Record<string, unknown>> | undefined;
  return Object.freeze({
    stop(): Readonly<Record<string, unknown>> {
      if (receipt !== undefined) return receipt;
      stopped = true;
      cancelAnimationFrame(animationFrame);
      window.clearInterval(timer);
      receipt = Object.freeze({
        startedAtEpochMilliseconds,
        stoppedAtEpochMilliseconds: Date.now(),
        animationFrameCount,
        timerTickCount,
        maximumAnimationFrameGapMilliseconds,
        maximumTimerGapMilliseconds,
        observed: animationFrameCount > 0 && timerTickCount > 0,
      });
      return receipt;
    },
  });
}

async function queueEmptyIdleTurn(): Promise<void> {
  await new Promise<void>((resolve) => {
    window.setTimeout(resolve, QUEUE_EMPTY_IDLE_MILLISECONDS);
  });
}

async function yieldToBrowser(): Promise<void> {
  await new Promise<void>((resolve) => window.setTimeout(resolve, 0));
}

async function sha256Hex(bytes: Uint8Array): Promise<string> {
  const copy = new Uint8Array(bytes);
  const digest = await crypto.subtle.digest("SHA-256", copy.buffer);
  return [...new Uint8Array(digest)]
    .map((value) => value.toString(16).padStart(2, "0"))
    .join("");
}

function binding(buffer: GPUBuffer, size: number): GPUBufferBinding {
  return Object.freeze({ buffer, offset: 0, size });
}

function requiredIdentity(parameters: URLSearchParams, name: string): string {
  const values = parameters.getAll(name);
  if (values.length !== 1 || values[0] === "") {
    throw new Error(`OPT-0011 production gate requires one ${name}`);
  }
  return values[0]!;
}

function fixtureSalt(id: Opt0011ProductionConv1dCaseId): number {
  const ordinal = OPT_0011_PRODUCTION_CONV1D_CASES.findIndex(
    (fixtureValue) => fixtureValue.id === id,
  );
  if (ordinal < 0) {
    const staticOrdinals: Record<Opt0011ProductionConv1dCaseId, number> = {
      "k7-block0-d1-first": 1,
      "k7-block0-d3-middle": 2,
      "k7-block0-d9-last": 3,
      "k7-final-last": 4,
      "k1-block0-tail": 5,
      "k1-block1-tail": 6,
      "k1-block2-tail": 7,
      "k1-block3-middle": 8,
      "k1-block4-last": 9,
      "k1-arithmetic-b1-f17-c65-c9": 10,
    };
    return Math.imul(staticOrdinals[id], 0x9e37_79b1) >>> 0;
  }
  return Math.imul(ordinal + 1, 0x9e37_79b1) >>> 0;
}

function mix32(value: number): number {
  let mixed = value >>> 0;
  mixed ^= mixed >>> 16;
  mixed = Math.imul(mixed, 0x7feb_352d) >>> 0;
  mixed ^= mixed >>> 15;
  mixed = Math.imul(mixed, 0x846c_a68b) >>> 0;
  return (mixed ^ (mixed >>> 16)) >>> 0;
}

function writeU16Le(bytes: Uint8Array, offset: number, value: number): void {
  bytes[offset] = value & 0xff;
  bytes[offset + 1] = value >>> 8;
}

function requireIndex(index: number, length: number, label: string): void {
  if (!Number.isSafeInteger(index) || index < 0 || index >= length) {
    throw new RangeError(`${label} index is outside [0, ${length})`);
  }
}

function align4(value: number): number {
  return Math.ceil(value / 4) * 4;
}

export function float16BitsToNumber(bits: number): number {
  const sign = (bits & 0x8000) === 0 ? 1 : -1;
  const exponent = (bits >>> 10) & 0x1f;
  const mantissa = bits & 0x03ff;
  if (exponent === 0) {
    return mantissa === 0 ? (sign < 0 ? -0 : 0) : sign * mantissa * 2 ** -24;
  }
  if (exponent === 0x1f) return mantissa === 0 ? sign * Infinity : NaN;
  return sign * (1 + mantissa / 1_024) * 2 ** (exponent - 15);
}

export function numberToFloat16Bits(value: number): number {
  FLOAT32_SCRATCH[0] = value;
  const bits = UINT32_SCRATCH[0]!;
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
    return sign | (truncated + (
      remainder > halfway || (remainder === halfway && (truncated & 1) !== 0)
        ? 1
        : 0
    ));
  }
  let halfMantissa = mantissa >>> 13;
  const remainder = mantissa & 0x1fff;
  if (remainder > 0x1000 || (remainder === 0x1000 && (halfMantissa & 1) !== 0)) {
    halfMantissa += 1;
    if (halfMantissa === 0x400) {
      const nextExponent = halfExponent + 1;
      return nextExponent >= 0x1f ? sign | 0x7c00 : sign | (nextExponent << 10);
    }
  }
  return sign | (halfExponent << 10) | halfMantissa;
}

function float32FromBits(bits: number): number {
  UINT32_SCRATCH[0] = bits >>> 0;
  return FLOAT32_SCRATCH[0]!;
}

function hex(value: number, storage: AceFp16VaeConv1dOutputStorage): string {
  return value.toString(16).padStart(storage === "float16" ? 4 : 8, "0");
}

function errorReceipt(error: unknown): Readonly<Record<string, unknown>> {
  if (error instanceof Error) {
    return Object.freeze({
      name: error.name,
      message: error.message,
      ...(error.stack === undefined ? {} : { stack: error.stack }),
    });
  }
  return Object.freeze({ name: "Error", message: String(error) });
}

function updateProgress(message: string): void {
  const progress = document.querySelector<HTMLElement>("#progress");
  if (progress !== null) progress.textContent = message;
}

function finish(status: "passed" | "failed", result: unknown): void {
  document.body.dataset.status = status;
  const progress = document.querySelector<HTMLElement>("#progress");
  if (progress !== null) progress.textContent = status;
  const output = document.querySelector<HTMLElement>("#result");
  if (output !== null) output.textContent = JSON.stringify(result, null, 2);
}
