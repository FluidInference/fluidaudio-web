/// <reference types="@webgpu/types" />
/// <reference types="vite/client" />

import k4CoreSource from
  "../../src/webgpu/kernels/vae-conv1d-fp16-direct-dot4-subgroup.ts?raw";
import candidateCoreSource from
  "../../src/webgpu/kernels/vae-conv1d-fp16-k8-k16-partials.ts?raw";
import decoderCoreSource from "../../src/webgpu/vae-decoder.ts?raw";
import {
  ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_KERNEL_ID,
  AceOpt0024VaeConv1dDirectDot4SubgroupKernel,
  aceOpt0024VaeConv1dDirectDot4SubgroupWgsl,
  planAceOpt0024VaeConv1dDirectDot4SubgroupRange,
} from
  "../../src/webgpu/kernels/vae-conv1d-fp16-direct-dot4-subgroup.js";
import {
  ACE_OPT_0041_VAE_K7_K8_PARTIALS_KERNEL_ID,
  ACE_OPT_0041_VAE_K7_K16_PARTIALS_KERNEL_ID,
  AceOpt0041VaeK7BoundedPartialsKernel,
  aceOpt0041VaeK7BoundedPartialsWgsl,
  planAceOpt0041VaeK7BoundedPartialsRange,
} from "../../src/webgpu/kernels/vae-conv1d-fp16-k8-k16-partials.js";
import {
  planAceFp16VaeConv1d,
  type AceFp16VaeConv1dPlan,
} from "../../src/webgpu/kernels/vae-conv1d-fp16.js";
import type {
  AceVaeConv1dShape,
  AceVaeOutputRangeBinding,
} from "../../src/webgpu/kernels/vae-primitives.js";
import {
  planAceVaeDecoder,
  planAceVaeDecoderQuanta,
} from "../../src/webgpu/vae-decoder.js";

type Arm = "k4" | "k8" | "k16";
type ProductionTier = "c1024" | "c512" | "c256" | "c128";
type FixtureKind =
  | "production"
  | "signed-zero"
  | "cancellation"
  | "finite-range"
  | "tail-cin";

interface Probe {
  readonly id: "first" | "interior" | "tail" | "full";
  readonly base: number;
  readonly count: number;
}

interface CaseSpec {
  readonly id: string;
  readonly kind: FixtureKind;
  readonly shape: AceVaeConv1dShape;
  readonly operationIndex: number;
  readonly probes: readonly Probe[];
  readonly tier?: ProductionTier;
  readonly timingWeight?: number;
}

interface EncodableDispatch {
  readonly kernelId: string;
  readonly outputRange: Readonly<{
    readonly base: number;
    readonly count: number;
    readonly workgroupsX: number;
    readonly workgroupsY: number;
  }>;
  encode(pass: GPUComputePassEncoder): void;
}

interface PreparedCase {
  readonly spec: CaseSpec;
  readonly plan: AceFp16VaeConv1dPlan;
  readonly input: GPUBuffer;
  readonly weight: GPUBuffer;
  readonly bias: GPUBuffer;
  readonly dispatches: Readonly<Record<string, Readonly<Record<Arm, EncodableDispatch>>>>;
}

interface PreparedGate {
  readonly adapter: GPUAdapter;
  readonly device: GPUDevice;
  readonly tracker: BufferTracker;
  readonly k4Kernel: AceOpt0024VaeConv1dDirectDot4SubgroupKernel;
  readonly k8Kernel: AceOpt0041VaeK7BoundedPartialsKernel;
  readonly k16Kernel: AceOpt0041VaeK7BoundedPartialsKernel;
  readonly cases: readonly PreparedCase[];
  readonly outputs: Readonly<Record<Arm, GPUBuffer>>;
  readonly qnanPrefill: GPUBuffer;
  readonly canary: GPUBuffer;
  readonly correctness: Readonly<Record<string, unknown>>;
  readonly identity: Readonly<Record<string, unknown>>;
  readonly uncapturedErrors: readonly string[];
  readonly deviceLosses: readonly string[];
  readonly preparedAtEpochMilliseconds: number;
  readonly updateProgress: (message: string) => void;
  cleanup(): Promise<Readonly<Record<string, unknown>>>;
}

interface NumericalAccumulator {
  count: number;
  differingRawU16Count: number;
  signedZeroDifferenceCount: number;
  finiteToZeroCount: number;
  sumError: number;
  sumAbsoluteError: number;
  sumSquaredError: number;
  sumControlSquared: number;
  sumControl: number;
  sumCandidate: number;
  sumCandidateSquared: number;
  sumProduct: number;
  maximumAbsoluteError: number;
  maximumAbsoluteControl: number;
  controlNonFiniteCount: number;
  candidateNonFiniteCount: number;
  firstDifference: Readonly<Record<string, unknown>> | null;
  worstDifference: Readonly<Record<string, unknown>> | null;
}

interface ThermalGate {
  readonly command: "notifyutil -g com.apple.system.thermalpressurelevel";
  readonly waitStartedAtEpochMilliseconds: number;
  readonly checkedAtEpochMilliseconds: number;
  readonly waitDurationMilliseconds: number;
  readonly checkCount: 1;
  readonly thermalLevel: 0;
  readonly launchDelayMilliseconds: number;
}

declare global {
  interface Window {
    __ACE_OPT0041_RESULT__?: Readonly<Record<string, unknown>>;
  }
}

const EXPERIMENT_ID = "OPT-0041" as const;
const C300_INPUT_FRAMES = 300;
const TIMING_WEIGHT_TOTAL = 2_397;
const REQUIRED_SPEEDUP_OVER_K4 = 1.15;
const NRMSE_MAXIMUM = 0.001;
const SNR_DECIBELS_MINIMUM = 60;
const PEARSON_MINIMUM = 0.99999;
const RELATIVE_MAXIMUM_ABSOLUTE_ERROR_MAXIMUM = 0.01;
const ADVERSARIAL_FINITE_TO_ZERO_EVENT_FLOOR = 4;
const RANGE_CONTROL_BYTES = 16;
const STORAGE_GUARD_BYTES = 256;
const OUTPUT_PREFILL_QNAN_F16 = 0x7e55;
const ADJACENT_CANARY_U32 = 0x5aa5_3cc3;
const THERMAL_COMMAND =
  "notifyutil -g com.apple.system.thermalpressurelevel" as const;
const MINIMUM_WAIT_MILLISECONDS = 30_000;
const MAXIMUM_WAIT_MILLISECONDS = 35_000;
const MAXIMUM_CHECK_TO_LAUNCH_MILLISECONDS = 15_000;

const ARMS = Object.freeze(["k4", "k8", "k16"] as const);
const CORRECTNESS_ORDERS = Object.freeze([
  Object.freeze(["k4", "k8", "k16"] as const),
  Object.freeze(["k16", "k8", "k4"] as const),
]);
const TIMING_ORDERS = Object.freeze([
  Object.freeze(["k4", "k8", "k16"] as const),
  Object.freeze(["k8", "k16", "k4"] as const),
  Object.freeze(["k16", "k4", "k8"] as const),
  Object.freeze(["k16", "k8", "k4"] as const),
  Object.freeze(["k4", "k16", "k8"] as const),
  Object.freeze(["k8", "k4", "k16"] as const),
]);
const PRODUCTION_TIERS = Object.freeze([
  Object.freeze({
    tier: "c1024" as const,
    label: "block-0-res-1-conv1",
    weight: 282,
    shape: Object.freeze({ frames: 3_000, channels: 1_024, dilation: 1 }),
  }),
  Object.freeze({
    tier: "c512" as const,
    label: "block-1-res-2-conv1",
    weight: 423,
    shape: Object.freeze({ frames: 18_000, channels: 512, dilation: 3 }),
  }),
  Object.freeze({
    tier: "c256" as const,
    label: "block-2-res-1-conv1",
    weight: 423,
    shape: Object.freeze({ frames: 72_000, channels: 256, dilation: 1 }),
  }),
  Object.freeze({
    tier: "c128" as const,
    label: "block-4-res-3-conv1",
    weight: 1_269,
    shape: Object.freeze({ frames: 576_000, channels: 128, dilation: 9 }),
  }),
]);

const INPUT_PATTERN = new Uint16Array([
  0x0000, 0x8000, 0x2400, 0xa400,
  0x2c00, 0xac00, 0x3000, 0xb000,
  0x3400, 0xb400, 0x3555, 0xb555,
  0x1800, 0x9800, 0x0001, 0x8001,
]);
const BIAS_PATTERN = new Uint16Array([
  0x0000, 0x8000, 0x2000, 0xa000,
  0x2400, 0xa400, 0x2800, 0xa800,
]);

// Filled after the static source is frozen. The page refuses to prepare if
// either benchmark owner changes under the registered experiment.
const EXPECTED_K4_CORE_SHA256 =
  "fe3bf8110cef1a3bb791006e9d376fe549e9f00fe30e4738d7429cb0daf65841";
const EXPECTED_CANDIDATE_CORE_SHA256 =
  "d97892196a9f1426d63b1fcd08cafeab6a1e11e8b3263e701e9102561c2d41cd";
const EXPECTED_DECODER_CORE_SHA256 =
  "07f294e2aadd615c0a8b840884f43205bc00c146362f54048a39a85440da1d3e";
const EXPECTED_GENERATED_SHADER_SHA256 = Object.freeze([
  Object.freeze({
    id: "block-0-res-1-conv1",
    k4: "2038e1c7a740718ec45303422c1860f641a9a2e8ac362490a57d2b29a1779a26",
    k8: "b3bceb14e77d00330876ab61913c1a9d3004e5305ab6db452862f2233b7e2e12",
    k16: "07492c2715d7690c309c1df71aa2e10752614010a88dce2a2d1c90fcaa82d8c7",
  }),
  Object.freeze({
    id: "block-1-res-2-conv1",
    k4: "6b0c5d93ca5aa1e80c07f21a484ae7ba1835811252e73168395ae43a0436ec34",
    k8: "9255db7325b848918b57e0f3404268dabfd134ee94ce54adcd0c560a2761d9d2",
    k16: "672ee785fcc8622c9f3318d08615ca9904c1565f4e613872fd891fcedba6f4fe",
  }),
  Object.freeze({
    id: "block-2-res-1-conv1",
    k4: "266365b9750c5c15a483da2c048b8f2b307e6165d9c565ac9f36c3cc03fe3cd7",
    k8: "ddcee8c4cd6998bea71be45e5c067efec2fc9660820ea81bc3cda3109b9af628",
    k16: "0b7d6c1f35fd88421af710315f3c4b20bc64767405d7917c382731db2259e68a",
  }),
  Object.freeze({
    id: "block-4-res-3-conv1",
    k4: "f9d91c63fb587583f16213b2b2f9d285acefc2b022a7efd42421feedcc0e3082",
    k8: "cf38ff075e0ebaaeda33da264b90bc3e8a94cc32b5a6533848eae7ae86e4da7e",
    k16: "8d4eb731f1b3f4de1c3f052d61e740fcbfbeb89f61a1074f6864d7461230bf13",
  }),
  Object.freeze({
    id: "signed-zero",
    k4: "daa2e0ee71fb34d4cf59799eb710012326288287656fc02b02d84e02daa7d571",
    k8: "90203fdf3441f0f025103c320805d8e57d1ff6bc66728efab4d193311536c300",
    k16: "f777e8177ab955690d632451932de43bed0696c6be361cb4214dce068ac63041",
  }),
  Object.freeze({
    id: "cancellation",
    k4: "85ec2a7a54408d391be6604b5f830dee3394904dec25dbd9caca782072eff84b",
    k8: "8a92b41607e194ab892425e1be7fb90ac5fbe8293ede2637a4d52c825f09a1f4",
    k16: "df8eae0f13273c0fff0d02a00475d1940d51078fb457e96822568ee742cf266a",
  }),
  Object.freeze({
    id: "finite-range",
    k4: "66e0d018d85df18cd15d5c527bb7e5978253a97692683bdc658e89ee509bb50f",
    k8: "dd30455d509a9867c584f5e20d0372958d0e794bc87b82e10c2a792b81e44703",
    k16: "2386c55da0befc30f72137189117224a2f9d94bc888279f1b3f7474aafac0a67",
  }),
  Object.freeze({
    id: "tail-cin",
    k4: "aa0cfe2e1ce7c4ef26696b26d8574c52a08ac9390988af1e2fad2272cb183d3c",
    k8: "3797e1cbb0a0453e7ead66130a89bdab4285acb24c7bea8637a86642418df4d4",
    k16: "2fec9d4f7a507a591a2a57b9dbf64e6d0ab5f536c48b97ab0cd520c029dd499e",
  }),
] as const);

if (typeof document !== "undefined" && document.querySelector("#run") !== null) {
  installBrowserGate();
}

function installBrowserGate(): void {
  const progress = requireElement<HTMLElement>("#progress");
  const fieldset = requireElement<HTMLFieldSetElement>("#thermal-gate");
  const run = requireElement<HTMLButtonElement>("#run");
  let prepared: PreparedGate | undefined;
  void prepareGate((message) => {
    progress.textContent = message;
  }).then(
    (value) => {
      prepared = value;
      document.body.dataset.status = "ready";
      progress.textContent =
        "READY — three-arm production-tier probes and full adversarial outputs completed; begin one 30-second idle wait";
      fieldset.disabled = false;
      run.disabled = false;
    },
    (error: unknown) => finishPage("failed", failureReceipt(error)),
  );
  run.addEventListener("click", () => {
    if (prepared === undefined) return;
    run.disabled = true;
    fieldset.disabled = true;
    const owned = prepared;
    prepared = undefined;
    document.body.dataset.status = "running";
    progress.textContent = "running the sole balanced timing screen";
    const launchedAtEpochMilliseconds = Date.now();
    let thermal: ThermalGate;
    try {
      thermal = parseThermalGate(
        fieldParameters("#thermal-gate"),
        owned.preparedAtEpochMilliseconds,
        launchedAtEpochMilliseconds,
      );
    } catch (error) {
      void owned.cleanup().then((cleanup) => finishPage("failed", Object.freeze({
        ...failureReceipt(error),
        cleanup,
      })));
      return;
    }
    void runTimedGate(owned, thermal, launchedAtEpochMilliseconds).then(
      (receipt) => finishPage("passed", receipt),
      (error: unknown) => finishPage("failed", failureReceipt(error)),
    );
  }, { once: true });
}

export function buildOpt0041Cases(): readonly CaseSpec[] {
  const graph = planAceVaeDecoder(C300_INPUT_FRAMES);
  const quanta = planAceVaeDecoderQuanta(graph);
  const production = PRODUCTION_TIERS.map((registered) => {
    const operationIndex = graph.operations.findIndex(({ label }) =>
      label === registered.label
    );
    const operation = graph.operations[operationIndex];
    if (operation === undefined || operation.kind !== "conv1d" ||
      operation.shape.kernelSize !== 7 || operation.bias === undefined) {
      throw new Error(`OPT-0041 production operation ${registered.label} changed`);
    }
    const shape = operation.shape;
    if (shape.inputFrames !== registered.shape.frames ||
      shape.outputChannels !== registered.shape.channels ||
      shape.inputChannels !== registered.shape.channels ||
      shape.dilation !== registered.shape.dilation) {
      throw new Error(`OPT-0041 production shape ${registered.label} changed`);
    }
    const ranges = quanta.quanta
      .filter((quantum) => quantum.operationIndex === operationIndex)
      .map((quantum) => {
        const primitive = quantum.primitives[0];
        if (quantum.operationKind !== "conv1d" ||
          quantum.primitives.length !== 1 || primitive === undefined ||
          primitive.outputBase !== quantum.logicalOutputBase ||
          primitive.outputCount !== quantum.logicalOutputCount) {
          throw new Error(`OPT-0041 ${registered.label} range topology changed`);
        }
        return Object.freeze({
          base: primitive.outputBase,
          count: primitive.outputCount,
        });
      });
    const first = ranges[0];
    const interior = ranges[Math.floor(ranges.length / 2)];
    const tail = ranges.at(-1);
    if (first === undefined || interior === undefined || tail === undefined) {
      throw new Error(`OPT-0041 ${registered.label} probes missing`);
    }
    return Object.freeze({
      id: registered.label,
      kind: "production" as const,
      shape,
      operationIndex,
      tier: registered.tier,
      timingWeight: registered.weight,
      probes: Object.freeze([
        Object.freeze({ id: "first" as const, ...first }),
        Object.freeze({ id: "interior" as const, ...interior }),
        Object.freeze({ id: "tail" as const, ...tail }),
      ]),
    });
  });
  const adversarial = Object.freeze([
    adversarialCase("signed-zero", 64, 33, 1, 10_001),
    adversarialCase("cancellation", 128, 35, 3, 10_002),
    adversarialCase("finite-range", 256, 37, 9, 10_003),
    adversarialCase("tail-cin", 68, 39, 3, 10_004),
  ]);
  const result = Object.freeze([...production, ...adversarial]);
  if (production.reduce((sum, spec) => sum + spec.timingWeight!, 0) !==
    TIMING_WEIGHT_TOTAL) {
    throw new Error("OPT-0041 timing weight total changed");
  }
  return result;
}

function adversarialCase(
  kind: Exclude<FixtureKind, "production">,
  inputChannels: number,
  frames: number,
  dilation: 1 | 3 | 9,
  operationIndex: number,
): CaseSpec {
  const shape = Object.freeze({
    batch: 1,
    inputFrames: frames,
    inputChannels,
    outputChannels: 128,
    kernelSize: 7,
    stride: 1,
    dilation,
    padding: dilation * 3,
  });
  return Object.freeze({
    id: kind,
    kind,
    shape,
    operationIndex,
    probes: Object.freeze([Object.freeze({
      id: "full" as const,
      base: 0,
      count: frames * 128,
    })]),
  });
}

async function prepareGate(
  updateProgress: (message: string) => void,
): Promise<PreparedGate> {
  const specs = buildOpt0041Cases();
  updateProgress("authenticating K4 and bounded-partial source identities");
  const identity = await buildIdentity(specs);
  if (navigator.gpu === undefined) throw new Error("WebGPU unavailable");
  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: "high-performance",
    forceFallbackAdapter: false,
  });
  if (adapter === null) throw new Error("No WebGPU adapter");
  requireAdapter(adapter, specs);
  const device = await adapter.requestDevice({
    label: "ace-opt-0041-k7-bounded-partials-gate",
    requiredFeatures: ["shader-f16", "subgroups"],
    requiredLimits: requiredDeviceLimits(adapter, specs),
  });
  const uncapturedErrors: string[] = [];
  const deviceLosses: string[] = [];
  const onUncapturedError = (event: GPUUncapturedErrorEvent): void => {
    uncapturedErrors.push(`${event.error.constructor.name}: ${event.error.message}`);
  };
  device.addEventListener("uncapturederror", onUncapturedError);
  void device.lost.then((info) => {
    if (info.reason !== "destroyed") {
      deviceLosses.push(`${info.reason}: ${info.message}`);
    }
  });
  const tracker = new BufferTracker();
  const fixed32 = Object.freeze({ subgroupMinSize: 32, subgroupMaxSize: 32 });
  const k4Kernel = AceOpt0024VaeConv1dDirectDot4SubgroupKernel.create(
    device,
    fixed32,
  );
  const k8Kernel = AceOpt0041VaeK7BoundedPartialsKernel.create(
    device,
    fixed32,
    "k8",
  );
  const k16Kernel = AceOpt0041VaeK7BoundedPartialsKernel.create(
    device,
    fixed32,
    "k16",
  );
  const plans = specs.map(({ shape }) => planAceFp16VaeConv1d(shape, "float16"));
  const maximumOutputBytes = Math.max(...plans.map(({ outputBindingBytes }) =>
    outputBindingBytes
  ));
  const maximumProbeBytes = Math.max(...specs.flatMap(({ probes }) =>
    probes.map(({ count }) => count * 2)
  ));
  const outputs = Object.freeze(Object.fromEntries(ARMS.map((arm) => [
    arm,
    tracker.create(device, {
      label: `opt-0041-${arm}-guarded-output`,
      size: STORAGE_GUARD_BYTES + maximumOutputBytes + STORAGE_GUARD_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST,
    }),
  ]))) as Readonly<Record<Arm, GPUBuffer>>;
  const qnanPrefill = tracker.create(device, {
    label: "opt-0041-qnan-prefill",
    size: maximumProbeBytes,
    usage: GPUBufferUsage.COPY_SRC,
    mappedAtCreation: true,
  });
  new Uint16Array(qnanPrefill.getMappedRange()).fill(OUTPUT_PREFILL_QNAN_F16);
  tracker.unmap(qnanPrefill);
  const canary = tracker.create(device, {
    label: "opt-0041-adjacent-canary",
    size: STORAGE_GUARD_BYTES,
    usage: GPUBufferUsage.COPY_SRC,
    mappedAtCreation: true,
  });
  new Uint32Array(canary.getMappedRange()).fill(ADJACENT_CANARY_U32);
  tracker.unmap(canary);
  const controls = createControls(device, tracker, specs);
  const cases: PreparedCase[] = [];
  let cleanupReceipt: Readonly<Record<string, unknown>> | undefined;
  const cleanup = async (): Promise<Readonly<Record<string, unknown>>> => {
    if (cleanupReceipt !== undefined) {
      return Object.freeze({ ...cleanupReceipt, repeatedCall: true });
    }
    await device.queue.onSubmittedWorkDone().catch(() => undefined);
    k4Kernel.destroy();
    k8Kernel.destroy();
    k16Kernel.destroy();
    k4Kernel.destroy();
    k8Kernel.destroy();
    k16Kernel.destroy();
    tracker.destroyAll();
    tracker.destroyAll();
    device.removeEventListener("uncapturederror", onUncapturedError);
    device.destroy();
    cleanupReceipt = Object.freeze({
      ...tracker.receipt(),
      queueDrainedBeforeRelease: true,
      kernelDestroyIdempotent: true,
      zeroLiveResources: tracker.liveBytes === 0,
      deviceDestroyed: true,
      repeatedCall: false,
    });
    return cleanupReceipt;
  };
  try {
    for (const [index, spec] of specs.entries()) {
      updateProgress(`compiling case ${index + 1}/${specs.length}: ${spec.id}`);
      cases.push(await prepareCase(
        device,
        tracker,
        controls,
        outputs,
        k4Kernel,
        k8Kernel,
        k16Kernel,
        spec,
      ));
      await yieldToBrowser();
    }
    updateProgress("running full-tier and adversarial correctness twice");
    const correctness = await runCorrectness(
      device,
      tracker,
      outputs,
      qnanPrefill,
      canary,
      cases,
      updateProgress,
    );
    await device.queue.onSubmittedWorkDone();
    requireNoGpuFailures(uncapturedErrors, deviceLosses, "correctness");
    updateProgress("symmetrically warming all production-tier composites");
    await warmProductionTiers(device, cases);
    await device.queue.onSubmittedWorkDone();
    requireNoGpuFailures(uncapturedErrors, deviceLosses, "warmup");
    const preparedAtEpochMilliseconds = Date.now();
    return Object.freeze({
      adapter,
      device,
      tracker,
      k4Kernel,
      k8Kernel,
      k16Kernel,
      cases: Object.freeze(cases),
      outputs,
      qnanPrefill,
      canary,
      correctness,
      identity,
      uncapturedErrors,
      deviceLosses,
      preparedAtEpochMilliseconds,
      updateProgress,
      cleanup,
    });
  } catch (error) {
    await cleanup();
    throw error;
  }
}

async function prepareCase(
  device: GPUDevice,
  tracker: BufferTracker,
  controls: Readonly<{ buffer: GPUBuffer; offsets: ReadonlyMap<string, number> }>,
  outputs: Readonly<Record<Arm, GPUBuffer>>,
  k4Kernel: AceOpt0024VaeConv1dDirectDot4SubgroupKernel,
  k8Kernel: AceOpt0041VaeK7BoundedPartialsKernel,
  k16Kernel: AceOpt0041VaeK7BoundedPartialsKernel,
  spec: CaseSpec,
): Promise<PreparedCase> {
  const plan = planAceFp16VaeConv1d(spec.shape, "float16");
  const input = tracker.create(device, {
    label: `opt-0041-${spec.id}-input`,
    size: plan.inputBindingBytes,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });
  fillInputBits(new Uint16Array(input.getMappedRange()), spec.kind);
  tracker.unmap(input);
  const weight = tracker.create(device, {
    label: `opt-0041-${spec.id}-native-o-k-i-weight`,
    size: plan.weightBindingBytes,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });
  fillWeightBits(
    new Uint16Array(weight.getMappedRange()),
    spec.kind,
    spec.operationIndex,
  );
  tracker.unmap(weight);
  const bias = tracker.create(device, {
    label: `opt-0041-${spec.id}-bias`,
    size: plan.biasBindingBytes,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });
  fillBiasBits(new Uint16Array(bias.getMappedRange()), spec.kind);
  tracker.unmap(bias);
  const dispatches: Record<string, Readonly<Record<Arm, EncodableDispatch>>> = {};
  for (const probe of spec.probes) {
    const controlOffset = controls.offsets.get(`${spec.id}:${probe.id}`);
    if (controlOffset === undefined) throw new Error("OPT-0041 control missing");
    const range: AceVaeOutputRangeBinding = Object.freeze({
      base: probe.base,
      count: probe.count,
      control: Object.freeze({
        buffer: controls.buffer,
        offset: controlOffset,
        size: RANGE_CONTROL_BYTES,
      }),
    });
    const common = Object.freeze({
      input: binding(input, plan.inputBindingBytes),
      weight: binding(weight, plan.weightBindingBytes),
      bias: binding(bias, plan.biasBindingBytes),
    });
    const k4 = await k4Kernel.createDispatch(
      `${spec.id}-${probe.id}-k4`,
      spec.shape,
      Object.freeze({
        ...common,
        output: guardedOutputBinding(outputs.k4, plan),
      }),
      "float16",
      range,
    );
    const k8 = await k8Kernel.createDispatch(
      `${spec.id}-${probe.id}-k8`,
      spec.shape,
      Object.freeze({
        ...common,
        output: guardedOutputBinding(outputs.k8, plan),
      }),
      "float16",
      range,
    );
    const k16 = await k16Kernel.createDispatch(
      `${spec.id}-${probe.id}-k16`,
      spec.shape,
      Object.freeze({
        ...common,
        output: guardedOutputBinding(outputs.k16, plan),
      }),
      "float16",
      range,
    );
    assertDispatch(spec, probe, plan, k4, k8, k16);
    dispatches[probe.id] = Object.freeze({ k4, k8, k16 });
  }
  return Object.freeze({
    spec,
    plan,
    input,
    weight,
    bias,
    dispatches: Object.freeze(dispatches),
  });
}

function assertDispatch(
  spec: CaseSpec,
  probe: Probe,
  plan: AceFp16VaeConv1dPlan,
  k4: EncodableDispatch,
  k8: EncodableDispatch,
  k16: EncodableDispatch,
): void {
  const range = Object.freeze({ base: probe.base, count: probe.count });
  const expectedK4 = planAceOpt0024VaeConv1dDirectDot4SubgroupRange(plan, range);
  const expectedCandidate = planAceOpt0041VaeK7BoundedPartialsRange(plan, range);
  if (k4.kernelId !== ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_KERNEL_ID ||
    k8.kernelId !== ACE_OPT_0041_VAE_K7_K8_PARTIALS_KERNEL_ID ||
    k16.kernelId !== ACE_OPT_0041_VAE_K7_K16_PARTIALS_KERNEL_ID ||
    !sameRange(k4.outputRange, expectedK4) ||
    !sameRange(k8.outputRange, expectedCandidate) ||
    !sameRange(k16.outputRange, expectedCandidate) ||
    !sameRange(k4.outputRange, k8.outputRange) ||
    !sameRange(k4.outputRange, k16.outputRange)) {
    throw new Error(`OPT-0041 dispatch mismatch ${spec.id}/${probe.id}`);
  }
}

async function runCorrectness(
  device: GPUDevice,
  tracker: BufferTracker,
  outputs: Readonly<Record<Arm, GPUBuffer>>,
  qnanPrefill: GPUBuffer,
  canary: GPUBuffer,
  cases: readonly PreparedCase[],
  updateProgress: (message: string) => void,
): Promise<Readonly<Record<string, unknown>>> {
  const aggregates = Object.freeze({
    k8: createNumericalAccumulator(),
    k16: createNumericalAccumulator(),
  });
  const receipts: Readonly<Record<string, unknown>>[] = [];
  let probeCount = 0;
  let dispatchCount = 0;
  let comparedU16PerCandidate = 0;
  for (const [caseIndex, prepared] of cases.entries()) {
    const probeReceipts: Readonly<Record<string, unknown>>[] = [];
    for (const probe of prepared.spec.probes) {
      updateProgress(
        `correctness ${caseIndex + 1}/${cases.length}: ${prepared.spec.id}/${probe.id}`,
      );
      const executions = [];
      for (const [executionIndex, order] of CORRECTNESS_ORDERS.entries()) {
        executions.push(await executeCorrectness(
          device,
          tracker,
          outputs,
          qnanPrefill,
          canary,
          prepared,
          probe,
          order,
          executionIndex,
        ));
        dispatchCount += ARMS.length;
        comparedU16PerCandidate += probe.count;
      }
      const firstHashes = executions[0]!.hashes;
      const secondHashes = executions[1]!.hashes;
      const deterministicRawU16 = Object.freeze(Object.fromEntries(
        ARMS.map((arm) => [arm, firstHashes[arm] === secondHashes[arm]]),
      )) as Readonly<Record<Arm, boolean>>;
      if (ARMS.some((arm) => !deterministicRawU16[arm])) {
        throw new Error(`OPT-0041 ${prepared.spec.id}/${probe.id} rerun changed`);
      }
      const probeNumerics = Object.freeze({
        k8: createNumericalAccumulator(),
        k16: createNumericalAccumulator(),
      });
      for (const execution of executions) {
        for (const arm of ["k8", "k16"] as const) {
          mergeNumericalAccumulator(probeNumerics[arm], execution.numerics[arm]);
          mergeNumericalAccumulator(aggregates[arm], execution.numerics[arm]);
        }
      }
      const numerical = Object.freeze({
        k8: summarizeNumerics(probeNumerics.k8),
        k16: summarizeNumerics(probeNumerics.k16),
      });
      assertNumericalEnvelope(numerical.k8, `${prepared.spec.id}/${probe.id}/k8`);
      assertNumericalEnvelope(numerical.k16, `${prepared.spec.id}/${probe.id}/k16`);
      if (prepared.spec.kind !== "production") {
        assertAdversarialEnvelope(numerical.k8, `${prepared.spec.id}/k8`);
        assertAdversarialEnvelope(numerical.k16, `${prepared.spec.id}/k16`);
      }
      probeReceipts.push(Object.freeze({
        probe,
        executionOrders: CORRECTNESS_ORDERS,
        hashes: Object.freeze([firstHashes, secondHashes]),
        deterministicRawU16,
        completeWrites: true,
        allOutputsFinite: true,
        adjacentCanariesIntact: true,
        numerical,
      }));
      probeCount += 1;
    }
    receipts.push(Object.freeze({
      id: prepared.spec.id,
      fixtureKind: prepared.spec.kind,
      tier: prepared.spec.tier ?? null,
      shape: prepared.spec.shape,
      probes: Object.freeze(probeReceipts),
    }));
    await yieldToBrowser();
  }
  const aggregate = Object.freeze({
    k8: summarizeNumerics(aggregates.k8),
    k16: summarizeNumerics(aggregates.k16),
  });
  assertNumericalEnvelope(aggregate.k8, "aggregate/k8");
  assertNumericalEnvelope(aggregate.k16, "aggregate/k16");
  return Object.freeze({
    productionTierCount: PRODUCTION_TIERS.length,
    adversarialFixtureCount: 4,
    probeCount,
    executionsPerProbe: CORRECTNESS_ORDERS.length,
    armsPerExecution: ARMS.length,
    dispatchCount,
    comparedRawU16CountPerCandidate: comparedU16PerCandidate,
    qNaNPrefillValue: OUTPUT_PREFILL_QNAN_F16,
    completeWrites: true,
    allOutputsFinite: true,
    adjacentCanariesIntact: true,
    deterministicRawU16: true,
    thresholds: Object.freeze({
      nrmseMaximum: NRMSE_MAXIMUM,
      snrDecibelsMinimum: SNR_DECIBELS_MINIMUM,
      pearsonMinimum: PEARSON_MINIMUM,
      relativeMaximumAbsoluteErrorMaximum:
        RELATIVE_MAXIMUM_ABSOLUTE_ERROR_MAXIMUM,
      adversarialFiniteToZeroEventFloor:
        ADVERSARIAL_FINITE_TO_ZERO_EVENT_FLOOR,
    }),
    aggregate,
    cases: Object.freeze(receipts),
    passed: true,
  });
}

async function executeCorrectness(
  device: GPUDevice,
  tracker: BufferTracker,
  outputs: Readonly<Record<Arm, GPUBuffer>>,
  qnanPrefill: GPUBuffer,
  canary: GPUBuffer,
  prepared: PreparedCase,
  probe: Probe,
  order: readonly Arm[],
  executionIndex: number,
): Promise<Readonly<{
  hashes: Readonly<Record<Arm, string>>;
  numerics: Readonly<Record<"k8" | "k16", NumericalAccumulator>>;
}>> {
  const selectedBytes = probe.count * 2;
  const readbackBytes = STORAGE_GUARD_BYTES + selectedBytes + STORAGE_GUARD_BYTES;
  const readbacks = Object.freeze(Object.fromEntries(ARMS.map((arm) => [
    arm,
    tracker.create(device, {
      label: `opt-0041-${prepared.spec.id}-${probe.id}-${executionIndex}-${arm}-readback`,
      size: readbackBytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    }),
  ]))) as Readonly<Record<Arm, GPUBuffer>>;
  const selectedOffset = STORAGE_GUARD_BYTES + probe.base * 2;
  const encoder = device.createCommandEncoder({
    label: `opt-0041-${prepared.spec.id}-${probe.id}-${executionIndex}`,
  });
  for (const arm of ARMS) {
    encoder.copyBufferToBuffer(
      canary,
      0,
      outputs[arm],
      selectedOffset - STORAGE_GUARD_BYTES,
      STORAGE_GUARD_BYTES,
    );
    encoder.copyBufferToBuffer(
      qnanPrefill,
      0,
      outputs[arm],
      selectedOffset,
      selectedBytes,
    );
    encoder.copyBufferToBuffer(
      canary,
      0,
      outputs[arm],
      selectedOffset + selectedBytes,
      STORAGE_GUARD_BYTES,
    );
  }
  const pass = encoder.beginComputePass();
  for (const arm of order) {
    prepared.dispatches[probe.id]![arm].encode(pass);
  }
  pass.end();
  for (const arm of ARMS) {
    encoder.copyBufferToBuffer(
      outputs[arm],
      selectedOffset - STORAGE_GUARD_BYTES,
      readbacks[arm],
      0,
      readbackBytes,
    );
  }
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  const words = {} as Record<Arm, Uint16Array>;
  const hashes = {} as Record<Arm, string>;
  try {
    for (const arm of ARMS) {
      await tracker.mapRead(readbacks[arm]);
      const bytes = new Uint8Array(readbacks[arm].getMappedRange());
      const leading = new Uint32Array(
        bytes.buffer,
        bytes.byteOffset,
        STORAGE_GUARD_BYTES / 4,
      );
      const selected = new Uint16Array(
        bytes.buffer,
        bytes.byteOffset + STORAGE_GUARD_BYTES,
        probe.count,
      );
      const trailing = new Uint32Array(
        bytes.buffer,
        bytes.byteOffset + STORAGE_GUARD_BYTES + selectedBytes,
        STORAGE_GUARD_BYTES / 4,
      );
      if (!everyU32(leading, ADJACENT_CANARY_U32) ||
        !everyU32(trailing, ADJACENT_CANARY_U32)) {
        throw new Error(`OPT-0041 ${prepared.spec.id}/${probe.id}/${arm} canary changed`);
      }
      const copied = Uint16Array.from(selected);
      if (copied.some((value) => value === OUTPUT_PREFILL_QNAN_F16)) {
        throw new Error(`OPT-0041 ${prepared.spec.id}/${probe.id}/${arm} incomplete write`);
      }
      if (copied.some((value) => !Number.isFinite(f16ToF32(value)))) {
        throw new Error(`OPT-0041 ${prepared.spec.id}/${probe.id}/${arm} non-finite output`);
      }
      words[arm] = copied;
      hashes[arm] = await sha256Bytes(new Uint8Array(copied.buffer));
      tracker.unmap(readbacks[arm]);
    }
  } finally {
    for (const arm of ARMS) tracker.destroy(readbacks[arm]);
  }
  return Object.freeze({
    hashes: Object.freeze(hashes),
    numerics: Object.freeze({
      k8: compareWords(words.k4, words.k8, prepared.spec, probe),
      k16: compareWords(words.k4, words.k16, prepared.spec, probe),
    }),
  });
}

async function warmProductionTiers(
  device: GPUDevice,
  cases: readonly PreparedCase[],
): Promise<void> {
  for (const prepared of productionCases(cases)) {
    for (const order of [ARMS, [...ARMS].reverse() as Arm[]]) {
      for (const arm of order) {
        await executeComposite(device, prepared, arm);
      }
    }
  }
}

async function runTimedGate(
  prepared: PreparedGate,
  thermal: ThermalGate,
  launchedAtEpochMilliseconds: number,
): Promise<Readonly<Record<string, unknown>>> {
  const samples = new Map<string, Record<Arm, number[]>>();
  let firstSubmitAtEpochMilliseconds: number | null = null;
  let submitDrainCount = 0;
  try {
    for (const production of productionCases(prepared.cases)) {
      const byArm = Object.freeze({ k4: [], k8: [], k16: [] }) as
        unknown as Record<Arm, number[]>;
      samples.set(production.spec.id, byArm);
      for (const order of TIMING_ORDERS) {
        for (const arm of order) {
          prepared.updateProgress(
            `timing ${production.spec.tier} order ${submitDrainCount + 1}/72: ${arm}`,
          );
          const sample = await executeComposite(prepared.device, production, arm);
          firstSubmitAtEpochMilliseconds ??= sample.submittedAtEpochMilliseconds;
          byArm[arm].push(sample.wallMilliseconds);
          submitDrainCount += 1;
        }
      }
    }
    await prepared.device.queue.onSubmittedWorkDone();
    requireNoGpuFailures(prepared.uncapturedErrors, prepared.deviceLosses, "timing");
    if (submitDrainCount !== PRODUCTION_TIERS.length *
      TIMING_ORDERS.length * ARMS.length) {
      throw new Error("OPT-0041 timing submit/drain accounting changed");
    }
    const timing = summarizeTiming(prepared.cases, samples);
    const environment = environmentReceipt(prepared.adapter, prepared.device);
    const memoryBeforeCleanup = prepared.tracker.receipt();
    const cleanup = await prepared.cleanup();
    const receipt = Object.freeze({
      schema: "ace-opt-0041-vae-k7-bounded-fp16-partials-v1",
      experiment: EXPERIMENT_ID,
      status: "completed",
      passed: timing.passed,
      identity: prepared.identity,
      environment,
      thermal,
      protocol: Object.freeze({
        preparedAtEpochMilliseconds: prepared.preparedAtEpochMilliseconds,
        launchedAtEpochMilliseconds,
        firstSubmitAtEpochMilliseconds,
        oneThermalCheckOnly: true,
        pollingLoggerUsed: false,
        timingOrders: TIMING_ORDERS,
        samplesPerArmPerTier: TIMING_ORDERS.length,
        dispatchesPerSample: 3,
        submitDrainCount,
        unchangedRetryPerformed: false,
      }),
      correctness: prepared.correctness,
      timing,
      decision: Object.freeze({
        disposition: timing.passed
          ? "positive-c512-escalation-eligible"
          : "negative-stop-before-c512",
        requiredSpeedupOverK4: REQUIRED_SPEEDUP_OVER_K4,
        c512AuthorizedVariant: timing.winner,
        productionIntegrationAuthorized: false,
        productionSelectorChanged: false,
        fullProductRunAuthorized: false,
      }),
      memoryBeforeCleanup,
      cleanup,
    });
    window.__ACE_OPT0041_RESULT__ = receipt;
    return receipt;
  } catch (error) {
    const cleanup = await prepared.cleanup();
    throw new Error(
      `OPT-0041 timed gate failed: ${error instanceof Error ? error.message : String(error)}; ` +
        `cleanup=${JSON.stringify(cleanup)}`,
      { cause: error },
    );
  }
}

async function executeComposite(
  device: GPUDevice,
  prepared: PreparedCase,
  arm: Arm,
): Promise<Readonly<{
  submittedAtEpochMilliseconds: number;
  wallMilliseconds: number;
}>> {
  if (prepared.spec.probes.length !== 3) {
    throw new Error("OPT-0041 timed composite requires first/interior/tail");
  }
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  for (const probe of prepared.spec.probes) {
    prepared.dispatches[probe.id]![arm].encode(pass);
  }
  pass.end();
  const command = encoder.finish();
  const started = performance.now();
  const submittedAtEpochMilliseconds = Date.now();
  device.queue.submit([command]);
  await device.queue.onSubmittedWorkDone();
  return Object.freeze({
    submittedAtEpochMilliseconds,
    wallMilliseconds: performance.now() - started,
  });
}

function summarizeTiming(
  cases: readonly PreparedCase[],
  samples: ReadonlyMap<string, Readonly<Record<Arm, readonly number[]>>>,
): Readonly<{
  tiers: readonly Readonly<Record<string, unknown>>[];
  weightedMilliseconds: Readonly<Record<Arm, number>>;
  weightedSpeedupOverK4: Readonly<Record<"k8" | "k16", number>>;
  eligible: Readonly<Record<"k8" | "k16", boolean>>;
  winner: "k8" | "k16" | null;
  passed: boolean;
}> {
  const weighted = { k4: 0, k8: 0, k16: 0 };
  const tiers = productionCases(cases).map((prepared) => {
    const input = samples.get(prepared.spec.id);
    if (input === undefined || prepared.spec.timingWeight === undefined) {
      throw new Error("OPT-0041 timing tier samples missing");
    }
    const medians = Object.freeze(Object.fromEntries(ARMS.map((arm) => [
      arm,
      median6(input[arm]),
    ]))) as Readonly<Record<Arm, number>>;
    for (const arm of ARMS) {
      weighted[arm] += medians[arm] * prepared.spec.timingWeight;
    }
    return Object.freeze({
      id: prepared.spec.tier,
      operationLabel: prepared.spec.id,
      weight: prepared.spec.timingWeight,
      samples: input,
      medians,
      speedupOverK4: Object.freeze({
        k8: medians.k4 / medians.k8,
        k16: medians.k4 / medians.k16,
      }),
      candidateWon: Object.freeze({
        k8: medians.k8 < medians.k4,
        k16: medians.k16 < medians.k4,
      }),
    });
  });
  const speedup = Object.freeze({
    k8: weighted.k4 / weighted.k8,
    k16: weighted.k4 / weighted.k16,
  });
  const eligible = Object.freeze({
    k8: speedup.k8 >= REQUIRED_SPEEDUP_OVER_K4 && tiers.every((tier) =>
      (tier.candidateWon as Readonly<Record<"k8", boolean>>).k8
    ),
    k16: speedup.k16 >= REQUIRED_SPEEDUP_OVER_K4 && tiers.every((tier) =>
      (tier.candidateWon as Readonly<Record<"k16", boolean>>).k16
    ),
  });
  const winner = eligible.k8 && eligible.k16
    ? (weighted.k8 <= weighted.k16 ? "k8" : "k16")
    : eligible.k8 ? "k8" : eligible.k16 ? "k16" : null;
  return Object.freeze({
    exactC300WeightTotal: TIMING_WEIGHT_TOTAL,
    tiers: Object.freeze(tiers),
    weightedMilliseconds: Object.freeze(weighted),
    weightedSpeedupOverK4: speedup,
    requiredSpeedupOverK4: REQUIRED_SPEEDUP_OVER_K4,
    allTierWinRequired: true,
    eligible,
    winner,
    passed: winner !== null,
  });
}

export function parseThermalGate(
  parameters: URLSearchParams,
  preparedAtEpochMilliseconds: number,
  launchedAtEpochMilliseconds: number,
): ThermalGate {
  const command = requiredParameter(parameters, "thermalCommand");
  const waitStartedAtEpochMilliseconds = requiredFiniteParameter(
    parameters,
    "waitStartedAtEpochMilliseconds",
  );
  const checkedAtEpochMilliseconds = requiredFiniteParameter(
    parameters,
    "checkedAtEpochMilliseconds",
  );
  const checkCount = requiredFiniteParameter(parameters, "checkCount");
  const thermalLevel = requiredFiniteParameter(parameters, "thermalLevel");
  const waitDurationMilliseconds = checkedAtEpochMilliseconds -
    waitStartedAtEpochMilliseconds;
  const launchDelayMilliseconds = launchedAtEpochMilliseconds -
    checkedAtEpochMilliseconds;
  if (command !== THERMAL_COMMAND || checkCount !== 1 || thermalLevel !== 0 ||
    waitStartedAtEpochMilliseconds < preparedAtEpochMilliseconds ||
    waitDurationMilliseconds < MINIMUM_WAIT_MILLISECONDS ||
    waitDurationMilliseconds > MAXIMUM_WAIT_MILLISECONDS ||
    launchDelayMilliseconds < 0 ||
    launchDelayMilliseconds > MAXIMUM_CHECK_TO_LAUNCH_MILLISECONDS) {
    throw new Error(
      "OPT-0041 requires exactly one level-0 notifyutil check after one fresh 30-second idle wait",
    );
  }
  return Object.freeze({
    command,
    waitStartedAtEpochMilliseconds,
    checkedAtEpochMilliseconds,
    waitDurationMilliseconds,
    checkCount: 1,
    thermalLevel: 0,
    launchDelayMilliseconds,
  });
}

async function buildIdentity(
  specs: readonly CaseSpec[],
): Promise<Readonly<Record<string, unknown>>> {
  const sources = Object.freeze({
    k4CoreSha256: await sha256Text(k4CoreSource),
    candidateCoreSha256: await sha256Text(candidateCoreSource),
    decoderCoreSha256: await sha256Text(decoderCoreSource),
  });
  if (sources.k4CoreSha256 !== EXPECTED_K4_CORE_SHA256 ||
    sources.candidateCoreSha256 !== EXPECTED_CANDIDATE_CORE_SHA256 ||
    sources.decoderCoreSha256 !== EXPECTED_DECODER_CORE_SHA256) {
    throw new Error("OPT-0041 rejected unauthenticated source bytes");
  }
  const generated = await Promise.all(specs.map(async (spec) => Object.freeze({
    id: spec.id,
    k4: await sha256Text(aceOpt0024VaeConv1dDirectDot4SubgroupWgsl(
      spec.shape,
      true,
      "float16",
    )),
    k8: await sha256Text(aceOpt0041VaeK7BoundedPartialsWgsl(
      spec.shape,
      true,
      "float16",
      "k8",
    )),
    k16: await sha256Text(aceOpt0041VaeK7BoundedPartialsWgsl(
      spec.shape,
      true,
      "float16",
      "k16",
    )),
  })));
  if (generated.length !== EXPECTED_GENERATED_SHADER_SHA256.length ||
    generated.some((entry, index) => {
      const expected = EXPECTED_GENERATED_SHADER_SHA256[index];
      return expected === undefined || entry.id !== expected.id ||
        entry.k4 !== expected.k4 || entry.k8 !== expected.k8 ||
        entry.k16 !== expected.k16;
    })) {
    throw new Error("OPT-0041 generated shader identity changed");
  }
  return Object.freeze({
    experiment: EXPERIMENT_ID,
    k4KernelId: ACE_OPT_0024_VAE_CONV1D_DIRECT_DOT4_SUBGROUP_KERNEL_ID,
    k8KernelId: ACE_OPT_0041_VAE_K7_K8_PARTIALS_KERNEL_ID,
    k16KernelId: ACE_OPT_0041_VAE_K7_K16_PARTIALS_KERNEL_ID,
    nativeWeightLayout: "O-K-I FP16, unchanged and shared by all arms",
    ...sources,
    generatedShaders: Object.freeze(generated),
  });
}

function createControls(
  device: GPUDevice,
  tracker: BufferTracker,
  specs: readonly CaseSpec[],
): Readonly<{ buffer: GPUBuffer; offsets: ReadonlyMap<string, number> }> {
  const alignment = Number(device.limits.minUniformBufferOffsetAlignment);
  if (!Number.isSafeInteger(alignment) || alignment < RANGE_CONTROL_BYTES ||
    !Number.isInteger(Math.log2(alignment))) {
    throw new Error("OPT-0041 invalid uniform alignment");
  }
  const entries = specs.flatMap((spec) => spec.probes.map((probe) => ({
    key: `${spec.id}:${probe.id}`,
    ...probe,
  })));
  const buffer = tracker.create(device, {
    label: "opt-0041-range-controls",
    size: entries.length * alignment,
    usage: GPUBufferUsage.UNIFORM,
    mappedAtCreation: true,
  });
  const words = new Uint32Array(buffer.getMappedRange());
  const offsets = new Map<string, number>();
  for (const [index, entry] of entries.entries()) {
    const offset = index * alignment;
    words[offset / 4] = entry.base;
    words[offset / 4 + 1] = entry.count;
    if (offsets.has(entry.key)) throw new Error("OPT-0041 duplicate control");
    offsets.set(entry.key, offset);
  }
  tracker.unmap(buffer);
  return Object.freeze({ buffer, offsets });
}

function fillInputBits(destination: Uint16Array, kind: FixtureKind): void {
  if (kind === "signed-zero") {
    fillPeriodic(destination, new Uint16Array([0x0000, 0x8000]));
  } else if (kind === "cancellation") {
    for (let index = 0; index < destination.length; index += 1) {
      destination[index] = Math.floor(index / 4) % 2 === 0 ? 0x3c00 : 0xbc00;
    }
  } else if (kind === "finite-range") {
    destination.fill(0x4000);
  } else {
    fillPeriodic(destination, INPUT_PATTERN);
  }
}

function fillWeightBits(
  destination: Uint16Array,
  kind: FixtureKind,
  operationIndex: number,
): void {
  if (kind === "signed-zero") {
    fillPeriodic(destination, new Uint16Array([0x3c00, 0xbc00]));
  } else if (kind === "cancellation") {
    destination.fill(0x3c00);
  } else if (kind === "finite-range") {
    destination.fill(0x3400);
  } else {
    for (let index = 0; index < destination.length; index += 1) {
      destination[index] = deterministicWeightBits(operationIndex, index);
    }
  }
}

function fillBiasBits(destination: Uint16Array, kind: FixtureKind): void {
  if (kind === "signed-zero" || kind === "cancellation") {
    destination.fill(0x0000);
  } else if (kind === "finite-range") {
    destination.fill(0x2c00);
  } else {
    fillPeriodic(destination, BIAS_PATTERN);
  }
}

function deterministicWeightBits(operationIndex: number, nativeIndex: number): number {
  const special = nativeIndex & 0x0fff;
  if (special === 0) return 0x0000;
  if (special === 1) return 0x8000;
  if (special === 2) return 0x0001;
  if (special === 3) return 0x8001;
  const mixed = mix32(
    nativeIndex ^ Math.imul(operationIndex + 1, 0x9e37_79b9),
  );
  return ((mixed >>> 16) & 0x8000) | 0x1000 | (mixed & 0x03ff);
}

function mix32(value: number): number {
  let mixed = value >>> 0;
  mixed ^= mixed >>> 16;
  mixed = Math.imul(mixed, 0x7feb_352d);
  mixed ^= mixed >>> 15;
  mixed = Math.imul(mixed, 0x846c_a68b);
  mixed ^= mixed >>> 16;
  return mixed >>> 0;
}

function fillPeriodic(destination: Uint16Array, pattern: Uint16Array): void {
  if (destination.length === 0 || pattern.length === 0) {
    throw new Error("OPT-0041 periodic fixture is empty");
  }
  const initial = Math.min(destination.length, pattern.length);
  destination.set(pattern.subarray(0, initial));
  let filled = initial;
  while (filled < destination.length) {
    const count = Math.min(filled, destination.length - filled);
    destination.copyWithin(filled, 0, count);
    filled += count;
  }
}

function compareWords(
  control: Uint16Array,
  candidate: Uint16Array,
  spec: CaseSpec,
  probe: Probe,
): NumericalAccumulator {
  if (control.length !== candidate.length) {
    throw new Error("OPT-0041 comparison length changed");
  }
  const accumulator = createNumericalAccumulator();
  for (let index = 0; index < control.length; index += 1) {
    const controlBits = control[index]!;
    const candidateBits = candidate[index]!;
    const controlValue = f16ToF32(controlBits);
    const candidateValue = f16ToF32(candidateBits);
    accumulator.count += 1;
    if (!Number.isFinite(controlValue)) accumulator.controlNonFiniteCount += 1;
    if (!Number.isFinite(candidateValue)) accumulator.candidateNonFiniteCount += 1;
    if (controlBits !== candidateBits) {
      accumulator.differingRawU16Count += 1;
      if ((controlBits & 0x7fff) === 0 && (candidateBits & 0x7fff) === 0) {
        accumulator.signedZeroDifferenceCount += 1;
      }
    }
    if (controlValue !== 0 && candidateValue === 0) {
      accumulator.finiteToZeroCount += 1;
    }
    const error = candidateValue - controlValue;
    const absoluteError = Math.abs(error);
    accumulator.sumError += error;
    accumulator.sumAbsoluteError += absoluteError;
    accumulator.sumSquaredError += error * error;
    accumulator.sumControlSquared += controlValue * controlValue;
    accumulator.sumControl += controlValue;
    accumulator.sumCandidate += candidateValue;
    accumulator.sumCandidateSquared += candidateValue * candidateValue;
    accumulator.sumProduct += controlValue * candidateValue;
    accumulator.maximumAbsoluteControl = Math.max(
      accumulator.maximumAbsoluteControl,
      Math.abs(controlValue),
    );
    if (absoluteError > accumulator.maximumAbsoluteError) {
      accumulator.maximumAbsoluteError = absoluteError;
      accumulator.worstDifference = Object.freeze({
        caseId: spec.id,
        probeId: probe.id,
        localIndex: index,
        outputIndex: probe.base + index,
        control: controlValue,
        candidate: candidateValue,
        error,
        absoluteError,
      });
    }
    if (accumulator.firstDifference === null && controlBits !== candidateBits) {
      accumulator.firstDifference = Object.freeze({
        caseId: spec.id,
        probeId: probe.id,
        localIndex: index,
        outputIndex: probe.base + index,
        controlBits,
        candidateBits,
        control: controlValue,
        candidate: candidateValue,
      });
    }
  }
  return accumulator;
}

function createNumericalAccumulator(): NumericalAccumulator {
  return {
    count: 0,
    differingRawU16Count: 0,
    signedZeroDifferenceCount: 0,
    finiteToZeroCount: 0,
    sumError: 0,
    sumAbsoluteError: 0,
    sumSquaredError: 0,
    sumControlSquared: 0,
    sumControl: 0,
    sumCandidate: 0,
    sumCandidateSquared: 0,
    sumProduct: 0,
    maximumAbsoluteError: 0,
    maximumAbsoluteControl: 0,
    controlNonFiniteCount: 0,
    candidateNonFiniteCount: 0,
    firstDifference: null,
    worstDifference: null,
  };
}

function mergeNumericalAccumulator(
  target: NumericalAccumulator,
  source: NumericalAccumulator,
): void {
  for (const key of [
    "count",
    "differingRawU16Count",
    "signedZeroDifferenceCount",
    "finiteToZeroCount",
    "sumError",
    "sumAbsoluteError",
    "sumSquaredError",
    "sumControlSquared",
    "sumControl",
    "sumCandidate",
    "sumCandidateSquared",
    "sumProduct",
    "controlNonFiniteCount",
    "candidateNonFiniteCount",
  ] as const) target[key] += source[key];
  if (source.maximumAbsoluteError > target.maximumAbsoluteError) {
    target.maximumAbsoluteError = source.maximumAbsoluteError;
    target.worstDifference = source.worstDifference;
  }
  target.maximumAbsoluteControl = Math.max(
    target.maximumAbsoluteControl,
    source.maximumAbsoluteControl,
  );
  target.firstDifference ??= source.firstDifference;
}

function summarizeNumerics(
  accumulator: NumericalAccumulator,
): Readonly<Record<string, unknown>> {
  if (accumulator.count === 0) throw new Error("OPT-0041 empty comparison");
  const count = accumulator.count;
  const rmsError = Math.sqrt(accumulator.sumSquaredError / count);
  const nrmse = rmsError / Math.max(accumulator.maximumAbsoluteControl, 1e-6);
  const snrDecibels = accumulator.sumSquaredError === 0
    ? "Infinity"
    : 10 * Math.log10(
        accumulator.sumControlSquared / accumulator.sumSquaredError,
      );
  const covariance = accumulator.sumProduct -
    accumulator.sumControl * accumulator.sumCandidate / count;
  const controlVariance = accumulator.sumControlSquared -
    accumulator.sumControl * accumulator.sumControl / count;
  const candidateVariance = accumulator.sumCandidateSquared -
    accumulator.sumCandidate * accumulator.sumCandidate / count;
  const varianceProduct = controlVariance * candidateVariance;
  const pearsonCorrelation = varianceProduct > 0
    ? covariance / Math.sqrt(varianceProduct)
    : accumulator.sumSquaredError === 0 ? 1 : 0;
  return Object.freeze({
    count,
    differingRawU16Count: accumulator.differingRawU16Count,
    signedZeroDifferenceCount: accumulator.signedZeroDifferenceCount,
    finiteToZeroCount: accumulator.finiteToZeroCount,
    finiteToZeroEventsPerMillion:
      accumulator.finiteToZeroCount / count * 1_000_000,
    signedMeanError: accumulator.sumError / count,
    meanAbsoluteError: accumulator.sumAbsoluteError / count,
    rmsError,
    nrmse,
    snrDecibels,
    pearsonCorrelation,
    maximumAbsoluteControl: accumulator.maximumAbsoluteControl,
    maximumAbsoluteError: accumulator.maximumAbsoluteError,
    relativeMaximumAbsoluteError: accumulator.maximumAbsoluteError /
      Math.max(accumulator.maximumAbsoluteControl, 1e-6),
    controlNonFiniteCount: accumulator.controlNonFiniteCount,
    candidateNonFiniteCount: accumulator.candidateNonFiniteCount,
    firstDifference: accumulator.firstDifference,
    worstDifference: accumulator.worstDifference,
  });
}

function assertNumericalEnvelope(
  summary: Readonly<Record<string, unknown>>,
  label: string,
): void {
  const snr = summary["snrDecibels"] === "Infinity"
    ? Number.POSITIVE_INFINITY
    : Number(summary["snrDecibels"]);
  if (Number(summary["controlNonFiniteCount"]) !== 0 ||
    Number(summary["candidateNonFiniteCount"]) !== 0 ||
    Number(summary["nrmse"]) > NRMSE_MAXIMUM ||
    snr < SNR_DECIBELS_MINIMUM ||
    Number(summary["pearsonCorrelation"]) < PEARSON_MINIMUM ||
    Number(summary["relativeMaximumAbsoluteError"]) >
      RELATIVE_MAXIMUM_ABSOLUTE_ERROR_MAXIMUM) {
    throw new Error(`OPT-0041 ${label} rejected the OPT-0024 numerical envelope`);
  }
}

function assertAdversarialEnvelope(
  summary: Readonly<Record<string, unknown>>,
  label: string,
): void {
  const count = Number(summary["count"]);
  const maximumEvents = Math.max(
    ADVERSARIAL_FINITE_TO_ZERO_EVENT_FLOOR,
    Math.floor(count / 1_000_000),
  );
  if (Number(summary["finiteToZeroCount"]) > maximumEvents) {
    throw new Error(`OPT-0041 ${label} introduced finite-to-zero collapse`);
  }
}

function requiredDeviceLimits(
  adapter: GPUAdapter,
  specs: readonly CaseSpec[],
): Record<string, number> {
  let maximumBuffer = 4;
  let maximumBinding = 4;
  let maximumDispatch = 1;
  for (const spec of specs) {
    const plan = planAceFp16VaeConv1d(spec.shape, "float16");
    maximumBinding = Math.max(
      maximumBinding,
      plan.inputBindingBytes,
      plan.weightBindingBytes,
      plan.biasBindingBytes,
      plan.outputBindingBytes,
    );
    maximumBuffer = Math.max(
      maximumBuffer,
      plan.inputBindingBytes,
      plan.weightBindingBytes,
      STORAGE_GUARD_BYTES + plan.outputBindingBytes + STORAGE_GUARD_BYTES,
    );
    for (const probe of spec.probes) {
      const range = planAceOpt0041VaeK7BoundedPartialsRange(plan, probe);
      maximumDispatch = Math.max(
        maximumDispatch,
        range.workgroupsX,
        range.workgroupsY,
      );
    }
  }
  const requested = {
    maxBufferSize: maximumBuffer,
    maxStorageBufferBindingSize: maximumBinding,
    maxUniformBufferBindingSize: RANGE_CONTROL_BYTES,
    maxComputeInvocationsPerWorkgroup: 128,
    maxComputeWorkgroupSizeX: 128,
    maxComputeWorkgroupStorageSize: 0,
    maxComputeWorkgroupsPerDimension: maximumDispatch,
  };
  for (const [name, minimum] of Object.entries(requested)) {
    const actual = Number(adapter.limits[name as keyof GPUSupportedLimits]);
    if (!Number.isFinite(actual) || actual < minimum) {
      throw new Error(`OPT-0041 adapter ${name}=${actual} is below ${minimum}`);
    }
  }
  return requested;
}

function requireAdapter(adapter: GPUAdapter, specs: readonly CaseSpec[]): void {
  if (!adapter.features.has("shader-f16") || !adapter.features.has("subgroups") ||
    adapter.info.subgroupMinSize !== 32 || adapter.info.subgroupMaxSize !== 32) {
    throw new Error("OPT-0041 requires shader-f16 and fixed 32-lane subgroups");
  }
  if (adapter.limits.minStorageBufferOffsetAlignment > STORAGE_GUARD_BYTES) {
    throw new Error("OPT-0041 storage guard is below adapter alignment");
  }
  requiredDeviceLimits(adapter, specs);
}

function environmentReceipt(adapter: GPUAdapter, device: GPUDevice) {
  return Object.freeze({
    userAgent: navigator.userAgent,
    adapter: Object.freeze({
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
      maxBufferSize: device.limits.maxBufferSize,
      maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize,
      maxComputeInvocationsPerWorkgroup:
        device.limits.maxComputeInvocationsPerWorkgroup,
      maxComputeWorkgroupSizeX: device.limits.maxComputeWorkgroupSizeX,
      maxComputeWorkgroupsPerDimension:
        device.limits.maxComputeWorkgroupsPerDimension,
    }),
  });
}

class BufferTracker {
  private readonly live = new Set<GPUBuffer>();
  private readonly sizes = new WeakMap<GPUBuffer, number>();
  createdBufferCount = 0;
  destroyedBufferCount = 0;
  mapCount = 0;
  unmapCount = 0;
  activeMapCount = 0;
  liveBytes = 0;
  maximumLiveBytes = 0;

  create(device: GPUDevice, descriptor: GPUBufferDescriptor): GPUBuffer {
    const buffer = device.createBuffer(descriptor);
    const size = Number(descriptor.size);
    this.live.add(buffer);
    this.sizes.set(buffer, size);
    this.createdBufferCount += 1;
    this.liveBytes += size;
    this.maximumLiveBytes = Math.max(this.maximumLiveBytes, this.liveBytes);
    if (descriptor.mappedAtCreation === true) {
      this.mapCount += 1;
      this.activeMapCount += 1;
    }
    return buffer;
  }

  async mapRead(buffer: GPUBuffer): Promise<void> {
    await buffer.mapAsync(GPUMapMode.READ);
    this.mapCount += 1;
    this.activeMapCount += 1;
  }

  unmap(buffer: GPUBuffer): void {
    if (buffer.mapState !== "mapped") throw new Error("OPT-0041 unbalanced unmap");
    buffer.unmap();
    this.unmapCount += 1;
    this.activeMapCount -= 1;
  }

  destroy(buffer: GPUBuffer): void {
    if (!this.live.delete(buffer)) return;
    if (buffer.mapState === "mapped") this.unmap(buffer);
    buffer.destroy();
    this.destroyedBufferCount += 1;
    this.liveBytes -= this.sizes.get(buffer) ?? 0;
    this.sizes.delete(buffer);
  }

  destroyAll(): void {
    for (const buffer of [...this.live]) this.destroy(buffer);
  }

  receipt(): Readonly<Record<string, number | boolean>> {
    return Object.freeze({
      createdBufferCount: this.createdBufferCount,
      destroyedBufferCount: this.destroyedBufferCount,
      liveBufferCount: this.live.size,
      liveBytes: this.liveBytes,
      maximumLiveBytes: this.maximumLiveBytes,
      mapCount: this.mapCount,
      unmapCount: this.unmapCount,
      activeMapCount: this.activeMapCount,
      mapsBalanced: this.mapCount === this.unmapCount && this.activeMapCount === 0,
    });
  }
}

function productionCases(cases: readonly PreparedCase[]): readonly PreparedCase[] {
  const result = cases.filter(({ spec }) => spec.kind === "production");
  if (result.length !== PRODUCTION_TIERS.length) {
    throw new Error("OPT-0041 production tier count changed");
  }
  return result;
}

function median6(samples: readonly number[]): number {
  if (samples.length !== 6 ||
    samples.some((value) => !Number.isFinite(value) || value <= 0)) {
    throw new Error("OPT-0041 requires six finite positive timing samples");
  }
  const sorted = [...samples].sort((left, right) => left - right);
  return (sorted[2]! + sorted[3]!) / 2;
}

function binding(buffer: GPUBuffer, size: number): GPUBufferBinding {
  return Object.freeze({ buffer, offset: 0, size });
}

function guardedOutputBinding(
  buffer: GPUBuffer,
  plan: AceFp16VaeConv1dPlan,
): GPUBufferBinding {
  return Object.freeze({
    buffer,
    offset: STORAGE_GUARD_BYTES,
    size: plan.outputBindingBytes,
  });
}

function sameRange(
  left: EncodableDispatch["outputRange"],
  right: EncodableDispatch["outputRange"],
): boolean {
  return left.base === right.base && left.count === right.count &&
    left.workgroupsX === right.workgroupsX &&
    left.workgroupsY === right.workgroupsY;
}

function everyU32(values: Uint32Array, expected: number): boolean {
  for (const value of values) if (value !== expected) return false;
  return true;
}

function f16ToF32(bits: number): number {
  const sign = (bits & 0x8000) === 0 ? 1 : -1;
  const exponent = (bits >>> 10) & 0x1f;
  const fraction = bits & 0x03ff;
  if (exponent === 0) {
    return fraction === 0 ? sign * 0 : sign * 2 ** -14 * fraction / 1_024;
  }
  if (exponent === 0x1f) {
    return fraction === 0 ? sign * Number.POSITIVE_INFINITY : Number.NaN;
  }
  return sign * 2 ** (exponent - 15) * (1 + fraction / 1_024);
}

function fieldParameters(selector: string): URLSearchParams {
  const parameters = new URLSearchParams();
  for (const input of document.querySelectorAll<HTMLInputElement>(
    `${selector} input[name]`,
  )) parameters.set(input.name, input.value);
  return parameters;
}

function requiredParameter(parameters: URLSearchParams, name: string): string {
  const value = parameters.get(name);
  if (value === null || value.length === 0) {
    throw new Error(`OPT-0041 field ${name} is missing`);
  }
  return value;
}

function requiredFiniteParameter(
  parameters: URLSearchParams,
  name: string,
): number {
  const value = Number(requiredParameter(parameters, name));
  if (!Number.isFinite(value)) throw new Error(`OPT-0041 field ${name} invalid`);
  return value;
}

function requireNoGpuFailures(
  uncapturedErrors: readonly string[],
  deviceLosses: readonly string[],
  phase: string,
): void {
  if (uncapturedErrors.length !== 0 || deviceLosses.length !== 0) {
    throw new Error(
      `OPT-0041 ${phase} GPU failure: ${JSON.stringify({ uncapturedErrors, deviceLosses })}`,
    );
  }
}

async function sha256Text(value: string): Promise<string> {
  return sha256Bytes(new TextEncoder().encode(value));
}

async function sha256Bytes(value: Uint8Array): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", Uint8Array.from(value));
  return [...new Uint8Array(digest)]
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

async function yieldToBrowser(): Promise<void> {
  await new Promise<void>((resolve) => setTimeout(resolve, 0));
}

function requireElement<ElementType extends Element>(selector: string): ElementType {
  const element = document.querySelector<ElementType>(selector);
  if (element === null) throw new Error(`Missing OPT-0041 element ${selector}`);
  return element;
}

function finishPage(
  status: "passed" | "failed",
  result: Readonly<Record<string, unknown>>,
): void {
  document.body.dataset.status = status;
  requireElement<HTMLElement>("#progress").textContent = status;
  requireElement<HTMLElement>("#result").textContent = JSON.stringify(
    result,
    null,
    2,
  );
}

function failureReceipt(error: unknown): Readonly<Record<string, unknown>> {
  const receipt = Object.freeze({
    schema: "ace-opt-0041-vae-k7-bounded-fp16-partials-v1",
    experiment: EXPERIMENT_ID,
    status: "failed",
    error: error instanceof Error
      ? Object.freeze({ name: error.name, message: error.message, stack: error.stack })
      : String(error),
    productionIntegrationAuthorized: false,
  });
  window.__ACE_OPT0041_RESULT__ = receipt;
  return receipt;
}
