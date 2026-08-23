import {
  planAceOpt0025VaeK1SubgroupGemm,
  planAceOpt0025VaeK1SubgroupGemmRange,
  planAceOpt0025VaeK1SubgroupGemmRangeDispatch,
} from "../../src/webgpu/kernels/vae-k1-fp16-subgroup-gemm.js";
import {
  planAceOpt0011Fp16VaeWindowDynamicControls,
} from "../../src/webgpu/vae-fp16-decoder.js";
import {
  planAceVaeDecoder,
  planAceVaeDecoderQuanta,
  type AceVaeDecoderConvOperation,
  type AceVaeDecoderOperation,
} from "../../src/webgpu/vae-decoder.js";
import { planAceVaeChunkedDecode } from "../../src/webgpu/vae-chunks.js";

export const OPT_0059_SCHEMA =
  "ace-opt-0059-vae-c2378-short-projection-abba-v1" as const;
export const OPT_0059_EXPERIMENT_ID = "OPT-0059" as const;
export const OPT_0059_SHAPES = Object.freeze([
  340,
  448,
  512,
  2_314,
] as const);
export const OPT_0059_FIXTURE_SHA256 = Object.freeze({
  "340": "286edbeba4ab49407f50564e7e84eef3218316b408cb7cef753d1225c3c50347",
  "448": "a1b3d1bebcfbdce1c4665a76534fd67a888a78072adb66e84d29fa7d05e26653",
  "512": "eff0005ae48353fbc0a9ec86a5b2824b49e6fff6e899ea89af7d1c6e5870e899",
  "2314": "01ec291963276b4784ec0ae3f6b3d7ed80bffd657dfd3b14125729260918783d",
} as const);
export const OPT_0059_MAIN_ORDER = Object.freeze([
  512,
  2_314,
  2_314,
  512,
] as const);
export const OPT_0059_EDGE_ORDER = Object.freeze([
  340,
  448,
  448,
  340,
] as const);
export const OPT_0059_QUANTA_PER_COMMAND_BUFFER = 64 as const;
export const OPT_0059_SPEEDUP_GATE = 1.15 as const;
export const OPT_0059_NORMALIZED_DECODER_REGRESSION_LIMIT = 1.10 as const;
export const OPT_0059_FAMILY_REGRESSION_LIMIT = 1.15 as const;
export const OPT_0059_MAXIMUM_LIVE_GPU_BYTES = 4_000_000_000 as const;
export const OPT_0059_REVISION7_MANIFEST_PATH =
  "/model/files-fp16-vae-revision7-experimental/manifest.json" as const;
export const OPT_0059_REVISION7_RUNTIME_PROFILE =
  "opt-0066-mixed-fp16-fixed32-dual-k4-quality-v1" as const;
export const OPT_0059_JOINT_KERNEL_PROFILE =
  "opt-0059-revision7-dual-k4-c2378-geometry-only-v1" as const;

export const OPT_0059_PROFILE_FAMILIES = Object.freeze([
  "k7-conv1d",
  "k1-conv1d",
  "conv-transpose1d",
  "snake",
  "add",
] as const);

export type Opt0059Shape = (typeof OPT_0059_SHAPES)[number];
export type Opt0059ProfileFamily =
  (typeof OPT_0059_PROFILE_FAMILIES)[number];

export interface Opt0059K1RoutePlan {
  readonly k1OperationIndex: number;
  readonly addOperationIndex: number;
  readonly successorSnakeOperationIndex: number;
  readonly k1Label: string;
  readonly addLabel: string;
  readonly successorSnakeLabel: string;
  readonly channels: number;
  readonly frames: number;
  readonly quantumCount: number;
  readonly mapping: "flat-x" | "column-x-row-y";
  readonly k1InputSlot: AceVaeDecoderOperation["input"];
  readonly k1OutputSlot: AceVaeDecoderOperation["output"];
  readonly addLeftSlot: AceVaeDecoderOperation["input"];
  readonly addOutputSlot: AceVaeDecoderOperation["output"];
  readonly successorOutputSlot: AceVaeDecoderOperation["output"];
  readonly addOutputAliasesK1Input: true;
  readonly successorOutputAliasesAddLeft: true;
}

export interface Opt0059ShapePlan {
  readonly inputFrames: Opt0059Shape;
  readonly operationCount: 88;
  readonly k1RouteCount: 15;
  readonly graphQuantumCount: number;
  readonly sequenceQuantumCount: number;
  readonly decoderCommandBufferCount: number;
  readonly readbackCommandBufferCount: 1;
  readonly totalCommandBufferCount: number;
  readonly requestedCooperativeIdleMilliseconds: number;
  readonly familyQuantumCounts: Readonly<Record<Opt0059ProfileFamily, number>>;
  readonly k1Routes: readonly Opt0059K1RoutePlan[];
  readonly twoDimensionalK1OperationLabels: readonly string[];
  readonly dynamicControlBytes: number;
  readonly stagingInputBytes: number;
  readonly decoderInputBytes: number;
  readonly workspaceBytes: number;
  readonly outputBytes: number;
}

export interface Opt0059GatePlan {
  readonly shapes: Readonly<Record<`${Opt0059Shape}`, Opt0059ShapePlan>>;
  readonly mainOrder: typeof OPT_0059_MAIN_ORDER;
  readonly edgeOrder: typeof OPT_0059_EDGE_ORDER;
  readonly controlC4500: Readonly<{
    readonly windowCounts: Readonly<Record<"340" | "448" | "512", number>>;
    readonly decodedLatentFrames: 5_908;
    readonly graphQuantumCount: number;
    readonly totalCommandBufferCount: number;
    readonly requestedCooperativeIdleMilliseconds: number;
  }>;
  readonly candidateC4500: Readonly<{
    readonly windowCounts: Readonly<Record<"2314", 2>>;
    readonly decodedLatentFrames: 4_628;
    readonly overlapFrames: 64;
    readonly windows: readonly [
      Readonly<{
        readonly latent: readonly [0, 2_314];
        readonly core: readonly [0, 2_250];
        readonly discardPrefixFrames: 0;
        readonly discardSuffixFrames: 64;
      }>,
      Readonly<{
        readonly latent: readonly [2_186, 4_500];
        readonly core: readonly [2_250, 4_500];
        readonly discardPrefixFrames: 64;
        readonly discardSuffixFrames: 0;
      }>,
    ];
    readonly graphQuantumCount: number;
    readonly totalCommandBufferCount: number;
    readonly requestedCooperativeIdleMilliseconds: number;
  }>;
  readonly sixC128K1MappingChanges: readonly string[];
  readonly maximumAllocation: Readonly<{
    readonly maximumActualWindowFrames: 2_314;
    readonly residentWeightBytes: 168_791_552;
    readonly guardedStagingInputBytes: number;
    readonly guardedDecoderInputBytes: number;
    readonly guardedWorkspaceBytesEach: number;
    readonly workspaceBufferCount: 3;
    readonly guardedOutputBytes: number;
    readonly readbackBytes: number;
    readonly retainedDynamicControlBytes: number;
    readonly canaryReadbackBytes: number;
    readonly plannedLiveGpuBytes: number;
    readonly belowFourGigabytes: boolean;
  }>;
}

export interface Opt0059ThermalGate {
  readonly source: "notifyutil-com.apple.system.thermalpressurelevel";
  readonly command: "notifyutil -g com.apple.system.thermalpressurelevel";
  readonly protocol: "wait-30s-then-one-level0-check";
  readonly startedAtEpochMilliseconds: number;
  readonly checkedAtEpochMilliseconds: number;
  readonly durationMilliseconds: number;
  readonly observationCount: 1;
  readonly observedLevel: 0;
  readonly launchDelayMilliseconds: number;
}

export interface Opt0059FamilyTiming {
  readonly batchCount: number;
  readonly quantumCount: number;
  readonly submitThroughDrainMs: number;
}

export interface Opt0059TimingSample {
  readonly inputFrames: Opt0059Shape;
  readonly decoderSubmitThroughDrainMs: number;
  readonly decoderWallMs: number;
  readonly readbackSubmitThroughDrainMs: number;
  readonly readbackMapWallMs: number;
  readonly outerWindowWallMs: number;
  readonly families: Readonly<Record<Opt0059ProfileFamily, Opt0059FamilyTiming>>;
  readonly mixed: Opt0059FamilyTiming;
}

export function planOpt0059Gate(): Opt0059GatePlan {
  const shapePlans = OPT_0059_SHAPES.map(planOpt0059Shape);
  const shapes = Object.freeze(Object.fromEntries(shapePlans.map((plan) => [
    String(plan.inputFrames),
    plan,
  ]))) as Readonly<Record<`${Opt0059Shape}`, Opt0059ShapePlan>>;
  const c340 = shapes["340"];
  const c448 = shapes["448"];
  const c512 = shapes["512"];
  const c2314 = shapes["2314"];
  const controlCommandBuffers = 10 * c512.totalCommandBufferCount +
    c448.totalCommandBufferCount + c340.totalCommandBufferCount;
  const candidateCommandBuffers = 2 * c2314.totalCommandBufferCount;
  const candidateCoverage = planAceVaeChunkedDecode(4_500, {
    chunkFrames: 2_378,
    overlapFrames: 64,
  });
  if (
    candidateCoverage.windows.length !== 2 ||
    JSON.stringify(candidateCoverage.windows.map((window) => ({
      latent: [window.windowStartLatentFrame, window.windowEndLatentFrame],
      core: [window.coreStartLatentFrame, window.coreEndLatentFrame],
      prefix: window.discardPrefixLatentFrames,
      suffix: window.discardSuffixLatentFrames,
    }))) !== JSON.stringify([
      { latent: [0, 2_314], core: [0, 2_250], prefix: 0, suffix: 64 },
      { latent: [2_186, 4_500], core: [2_250, 4_500], prefix: 64, suffix: 0 },
    ])
  ) throw new Error("OPT-0059 exact C4500/C2378 coverage changed");
  const c512Mappings = new Map(c512.k1Routes.map((route) => [
    route.k1Label,
    route.mapping,
  ]));
  const sixC128K1MappingChanges = Object.freeze(c2314.k1Routes
    .filter((route) =>
      c512Mappings.get(route.k1Label) === "flat-x" &&
      route.mapping === "column-x-row-y"
    )
    .map((route) => route.k1Label));
  const retainedDynamicControlBytes = shapePlans.reduce(
    (sum, plan) => sum + plan.dynamicControlBytes,
    0,
  );
  const guardBytes = 512;
  const canaryReadbackBytes = 4_096;
  const plannedLiveGpuBytes =
    168_791_552 +
    c2314.stagingInputBytes + guardBytes +
    c2314.decoderInputBytes + guardBytes +
    3 * (c2314.workspaceBytes + guardBytes) +
    c2314.outputBytes + guardBytes +
    c2314.outputBytes +
    retainedDynamicControlBytes +
    canaryReadbackBytes;
  return Object.freeze({
    shapes,
    mainOrder: OPT_0059_MAIN_ORDER,
    edgeOrder: OPT_0059_EDGE_ORDER,
    controlC4500: Object.freeze({
      windowCounts: Object.freeze({ "340": 1, "448": 1, "512": 10 }),
      decodedLatentFrames: 5_908 as const,
      graphQuantumCount:
        10 * c512.graphQuantumCount +
        c448.graphQuantumCount +
        c340.graphQuantumCount,
      totalCommandBufferCount: controlCommandBuffers,
      requestedCooperativeIdleMilliseconds: controlCommandBuffers - 1,
    }),
    candidateC4500: Object.freeze({
      windowCounts: Object.freeze({ "2314": 2 as const }),
      decodedLatentFrames: 4_628 as const,
      overlapFrames: 64 as const,
      windows: Object.freeze([
        Object.freeze({
          latent: Object.freeze([0, 2_314] as const),
          core: Object.freeze([0, 2_250] as const),
          discardPrefixFrames: 0 as const,
          discardSuffixFrames: 64 as const,
        }),
        Object.freeze({
          latent: Object.freeze([2_186, 4_500] as const),
          core: Object.freeze([2_250, 4_500] as const),
          discardPrefixFrames: 64 as const,
          discardSuffixFrames: 0 as const,
        }),
      ] as const),
      graphQuantumCount: 2 * c2314.graphQuantumCount,
      totalCommandBufferCount: candidateCommandBuffers,
      requestedCooperativeIdleMilliseconds: candidateCommandBuffers - 1,
    }),
    sixC128K1MappingChanges,
    maximumAllocation: Object.freeze({
      maximumActualWindowFrames: 2_314 as const,
      residentWeightBytes: 168_791_552 as const,
      guardedStagingInputBytes: c2314.stagingInputBytes + guardBytes,
      guardedDecoderInputBytes: c2314.decoderInputBytes + guardBytes,
      guardedWorkspaceBytesEach: c2314.workspaceBytes + guardBytes,
      workspaceBufferCount: 3 as const,
      guardedOutputBytes: c2314.outputBytes + guardBytes,
      readbackBytes: c2314.outputBytes,
      retainedDynamicControlBytes,
      canaryReadbackBytes,
      plannedLiveGpuBytes,
      belowFourGigabytes: plannedLiveGpuBytes <
        OPT_0059_MAXIMUM_LIVE_GPU_BYTES,
    }),
  });
}

export function planOpt0059Shape(inputFrames: Opt0059Shape): Opt0059ShapePlan {
  const plan = planAceVaeDecoder(inputFrames);
  const cooperative = planAceVaeDecoderQuanta(plan);
  const k1Routes = plan.operations.flatMap((operation, operationIndex) => {
    if (!isK1(operation)) return [];
    const add = plan.operations[operationIndex + 1];
    const successor = plan.operations[operationIndex + 2];
    const chain = requireK1AddSuccessorChain(
      operation,
      add,
      successor,
      operationIndex,
    );
    const quanta = cooperative.quanta.filter((quantum) =>
      quantum.operationIndex === operationIndex
    );
    const first = quanta[0];
    if (first === undefined) {
      throw new Error(`OPT-0059 ${operation.label} has no cooperative quantum`);
    }
    const k1Plan = planAceOpt0025VaeK1SubgroupGemm(operation.shape);
    const range = planAceOpt0025VaeK1SubgroupGemmRange(k1Plan, {
      base: first.logicalOutputBase,
      count: first.logicalOutputCount,
    });
    const dispatch = planAceOpt0025VaeK1SubgroupGemmRangeDispatch(
      k1Plan,
      range,
    );
    return [Object.freeze({
      k1OperationIndex: operationIndex,
      addOperationIndex: operationIndex + 1,
      successorSnakeOperationIndex: operationIndex + 2,
      k1Label: operation.label,
      addLabel: chain.add.label,
      successorSnakeLabel: chain.successor.label,
      channels: operation.shape.outputChannels,
      frames: operation.shape.inputFrames,
      quantumCount: quanta.length,
      mapping: dispatch.mapping,
      k1InputSlot: operation.input,
      k1OutputSlot: operation.output,
      addLeftSlot: chain.add.input,
      addOutputSlot: chain.add.output,
      successorOutputSlot: chain.successor.output,
      addOutputAliasesK1Input: true as const,
      successorOutputAliasesAddLeft: true as const,
    })];
  });
  if (k1Routes.length !== 15) {
    throw new Error(`OPT-0059 expected 15 K1 routes, got ${k1Routes.length}`);
  }
  const familyQuantumCounts: Record<Opt0059ProfileFamily, number> = {
    "k7-conv1d": 0,
    "k1-conv1d": 0,
    "conv-transpose1d": 0,
    snake: 0,
    add: 0,
  };
  for (const quantum of cooperative.quanta) {
    const operation = plan.operations[quantum.operationIndex]!;
    if (operation.kind === "conv1d") {
      familyQuantumCounts[operation.shape.kernelSize === 1
        ? "k1-conv1d"
        : "k7-conv1d"] += 1;
    } else if (operation.kind === "conv-transpose1d") {
      familyQuantumCounts["conv-transpose1d"] += 1;
    } else if (operation.kind === "snake") {
      familyQuantumCounts.snake += 1;
    } else {
      familyQuantumCounts.add += 1;
    }
  }
  const graphQuantumCount = Object.values(familyQuantumCounts)
    .reduce((sum, count) => sum + count, 0);
  const sequenceQuantumCount = graphQuantumCount + 1;
  const decoderCommandBufferCount = Math.ceil(
    sequenceQuantumCount / OPT_0059_QUANTA_PER_COMMAND_BUFFER,
  );
  const totalCommandBufferCount = decoderCommandBufferCount + 1;
  const controls = planAceOpt0011Fp16VaeWindowDynamicControls(
    inputFrames,
    256,
  );
  return Object.freeze({
    inputFrames,
    operationCount: 88 as const,
    k1RouteCount: 15 as const,
    graphQuantumCount,
    sequenceQuantumCount,
    decoderCommandBufferCount,
    readbackCommandBufferCount: 1 as const,
    totalCommandBufferCount,
    requestedCooperativeIdleMilliseconds: totalCommandBufferCount - 1,
    familyQuantumCounts: Object.freeze(familyQuantumCounts),
    k1Routes: Object.freeze(k1Routes),
    twoDimensionalK1OperationLabels: Object.freeze(k1Routes
      .filter((route) => route.mapping === "column-x-row-y")
      .map((route) => route.k1Label)),
    dynamicControlBytes: controls.byteLength,
    stagingInputBytes: plan.inputElements * Float32Array.BYTES_PER_ELEMENT,
    decoderInputBytes: plan.inputElements * Uint16Array.BYTES_PER_ELEMENT,
    workspaceBytes:
      plan.maximumActivationElements * Uint16Array.BYTES_PER_ELEMENT,
    outputBytes: plan.outputElements * Float32Array.BYTES_PER_ELEMENT,
  });
}

export function parseOpt0059ThermalGate(
  parameters: URLSearchParams,
  readyAtEpochMilliseconds: number,
  nowEpochMilliseconds: number,
): Opt0059ThermalGate {
  const source = parameters.get("thermalSource");
  const startedAtEpochMilliseconds = requiredNumber(
    parameters,
    "thermalStartedAtEpochMilliseconds",
  );
  const checkedAtEpochMilliseconds = requiredNumber(
    parameters,
    "thermalCheckedAtEpochMilliseconds",
  );
  const observationCount = requiredNumber(parameters, "thermalObservations");
  const observedLevel = requiredNumber(parameters, "thermalObservedLevel");
  const durationMilliseconds = checkedAtEpochMilliseconds -
    startedAtEpochMilliseconds;
  const launchDelayMilliseconds = nowEpochMilliseconds -
    checkedAtEpochMilliseconds;
  if (
    source !== "notifyutil-com.apple.system.thermalpressurelevel" ||
    !Number.isFinite(readyAtEpochMilliseconds) ||
    startedAtEpochMilliseconds < readyAtEpochMilliseconds ||
    checkedAtEpochMilliseconds < startedAtEpochMilliseconds ||
    durationMilliseconds < 30_000 ||
    observationCount !== 1 || observedLevel !== 0 ||
    launchDelayMilliseconds < 0 || launchDelayMilliseconds > 5_000
  ) {
    throw new Error(
      "OPT-0059 requires one truthful level-0 notifyutil check after a 30-second wait and launch within 5 seconds",
    );
  }
  return Object.freeze({
    source,
    command: "notifyutil -g com.apple.system.thermalpressurelevel" as const,
    protocol: "wait-30s-then-one-level0-check" as const,
    startedAtEpochMilliseconds,
    checkedAtEpochMilliseconds,
    durationMilliseconds,
    observationCount: 1 as const,
    observedLevel: 0 as const,
    launchDelayMilliseconds,
  });
}

export function evaluateOpt0059Timing(
  main: readonly Opt0059TimingSample[],
  edge: readonly Opt0059TimingSample[],
): Readonly<Record<string, unknown>> & { readonly passed: boolean } {
  requireTimingOrder(main, OPT_0059_MAIN_ORDER, "main");
  requireTimingOrder(edge, OPT_0059_EDGE_ORDER, "edge");
  const gatePlan = planOpt0059Gate();
  const forward = evaluateProjectionDirection(
    "forward",
    main[0]!,
    main[1]!,
    edge[0]!,
    edge[1]!,
    gatePlan,
  );
  const reverse = evaluateProjectionDirection(
    "reverse",
    main[3]!,
    main[2]!,
    edge[3]!,
    edge[2]!,
    gatePlan,
  );
  const aggregate = evaluateProjectionDirection(
    "aggregate-medians",
    medianSample([main[0]!, main[3]!]),
    medianSample([main[1]!, main[2]!]),
    medianSample([edge[0]!, edge[3]!]),
    medianSample([edge[1]!, edge[2]!]),
    gatePlan,
  );
  const passed = forward.passed && reverse.passed && aggregate.passed;
  return Object.freeze({
    mainOrder: OPT_0059_MAIN_ORDER,
    edgeOrder: OPT_0059_EDGE_ORDER,
    forward,
    reverse,
    aggregate,
    passed,
  });
}

function evaluateProjectionDirection(
  direction: "forward" | "reverse" | "aggregate-medians",
  c512: Opt0059TimingSample,
  c2314: Opt0059TimingSample,
  c340: Opt0059TimingSample,
  c448: Opt0059TimingSample,
  gatePlan: Opt0059GatePlan,
): Readonly<Record<string, unknown>> & { readonly passed: boolean } {
  const controlProjectedMilliseconds =
    10 * c512.outerWindowWallMs + c448.outerWindowWallMs +
    c340.outerWindowWallMs + 11;
  const candidateProjectedMilliseconds = 2 * c2314.outerWindowWallMs + 1;
  const speedup = controlProjectedMilliseconds /
    candidateProjectedMilliseconds;
  const normalizedDecoderRatio =
    (c2314.decoderWallMs / 2_314) / (c512.decoderWallMs / 512);
  const routes = Object.freeze([
    ...OPT_0059_PROFILE_FAMILIES,
    "mixed" as const,
  ].map((family) => {
    const control = family === "mixed" ? c512.mixed : c512.families[family];
    const candidate = family === "mixed"
      ? c2314.mixed
      : c2314.families[family];
    const comparable = control.submitThroughDrainMs > 0 &&
      candidate.submitThroughDrainMs > 0;
    const normalizedRatio = comparable
      ? (candidate.submitThroughDrainMs / 2_314) /
        (control.submitThroughDrainMs / 512)
      : null;
    const geometricExplanation =
      family === "k1-conv1d" &&
      gatePlan.sixC128K1MappingChanges.length === 6
        ? "six C128 K1 chains change from flat-X to bounded column-X/row-Y dispatch"
        : null;
    const passed = normalizedRatio === null ||
      normalizedRatio <= OPT_0059_FAMILY_REGRESSION_LIMIT ||
      geometricExplanation !== null;
    return Object.freeze({
      family,
      comparable,
      control,
      candidate,
      normalizedRatio,
      limit: OPT_0059_FAMILY_REGRESSION_LIMIT,
      geometricExplanation,
      passed,
    });
  }));
  const passed = speedup >= OPT_0059_SPEEDUP_GATE &&
    normalizedDecoderRatio <= OPT_0059_NORMALIZED_DECODER_REGRESSION_LIMIT &&
    routes.every((route) => route.passed);
  return Object.freeze({
    direction,
    controlProjection: Object.freeze({
      formula: "10*C512 + C448 + C340 + 11ms",
      milliseconds: controlProjectedMilliseconds,
    }),
    candidateProjection: Object.freeze({
      formula: "2*C2314 + 1ms",
      milliseconds: candidateProjectedMilliseconds,
    }),
    speedup,
    requiredSpeedup: OPT_0059_SPEEDUP_GATE,
    normalizedDecoderRatio,
    maximumNormalizedDecoderRatio:
      OPT_0059_NORMALIZED_DECODER_REGRESSION_LIMIT,
    routes,
    passed,
  });
}

function medianSample(
  samples: readonly [Opt0059TimingSample, Opt0059TimingSample],
): Opt0059TimingSample {
  const inputFrames = samples[0].inputFrames;
  if (samples[1].inputFrames !== inputFrames) {
    throw new Error("OPT-0059 cannot median unlike shapes");
  }
  const timing = (select: (sample: Opt0059TimingSample) => number): number =>
    median(samples.map(select));
  const family = (
    select: (sample: Opt0059TimingSample) => Opt0059FamilyTiming,
  ): Opt0059FamilyTiming => Object.freeze({
    batchCount: median(samples.map((sample) => select(sample).batchCount)),
    quantumCount: median(samples.map((sample) => select(sample).quantumCount)),
    submitThroughDrainMs: median(samples.map((sample) =>
      select(sample).submitThroughDrainMs
    )),
  });
  return Object.freeze({
    inputFrames,
    decoderSubmitThroughDrainMs: timing((sample) =>
      sample.decoderSubmitThroughDrainMs
    ),
    decoderWallMs: timing((sample) => sample.decoderWallMs),
    readbackSubmitThroughDrainMs: timing((sample) =>
      sample.readbackSubmitThroughDrainMs
    ),
    readbackMapWallMs: timing((sample) => sample.readbackMapWallMs),
    outerWindowWallMs: timing((sample) => sample.outerWindowWallMs),
    families: Object.freeze(Object.fromEntries(
      OPT_0059_PROFILE_FAMILIES.map((name) => [
        name,
        family((sample) => sample.families[name]),
      ]),
    )) as Readonly<Record<Opt0059ProfileFamily, Opt0059FamilyTiming>>,
    mixed: family((sample) => sample.mixed),
  });
}

function requireTimingOrder(
  samples: readonly Opt0059TimingSample[],
  order: readonly Opt0059Shape[],
  label: string,
): void {
  if (
    samples.length !== order.length ||
    samples.some((sample, index) =>
      sample.inputFrames !== order[index] ||
      [
        sample.decoderSubmitThroughDrainMs,
        sample.decoderWallMs,
        sample.readbackSubmitThroughDrainMs,
        sample.readbackMapWallMs,
        sample.outerWindowWallMs,
      ].some((value) => !Number.isFinite(value) || value <= 0) ||
      [...OPT_0059_PROFILE_FAMILIES.map((family) => sample.families[family]),
        sample.mixed].some((bucket) =>
          !Number.isFinite(bucket.submitThroughDrainMs) ||
          bucket.submitThroughDrainMs < 0 ||
          !Number.isSafeInteger(bucket.batchCount) || bucket.batchCount < 0 ||
          !Number.isSafeInteger(bucket.quantumCount) || bucket.quantumCount < 0
        )
    )
  ) {
    throw new Error(`OPT-0059 ${label} samples are not finite balanced ABBA`);
  }
}

function requireK1AddSuccessorChain(
  k1: AceVaeDecoderConvOperation,
  add: AceVaeDecoderOperation | undefined,
  successor: AceVaeDecoderOperation | undefined,
  operationIndex: number,
): Readonly<{
  readonly add: Extract<AceVaeDecoderOperation, { kind: "add" }>;
  readonly successor: Extract<AceVaeDecoderOperation, { kind: "snake" }>;
}> {
  if (
    add?.kind !== "add" || successor?.kind !== "snake" ||
    add.right !== k1.output || successor.input !== add.output ||
    add.output !== k1.input || successor.output !== add.input ||
    add.shape.batch !== k1.shape.batch ||
    add.shape.frames !== k1.shape.inputFrames ||
    add.shape.channels !== k1.shape.outputChannels ||
    successor.shape.batch !== add.shape.batch ||
    successor.shape.frames !== add.shape.frames ||
    successor.shape.channels !== add.shape.channels
  ) {
    throw new Error(
      `OPT-0059 operation ${operationIndex} lost its exact K1/Add/Snake slots`,
    );
  }
  return Object.freeze({ add, successor });
}

function isK1(
  operation: AceVaeDecoderOperation,
): operation is AceVaeDecoderConvOperation {
  return operation.kind === "conv1d" && operation.shape.kernelSize === 1;
}

function median(values: readonly number[]): number {
  if (values.length === 0 || values.some((value) => !Number.isFinite(value))) {
    throw new RangeError("OPT-0059 median requires finite values");
  }
  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[middle - 1]! + sorted[middle]!) / 2
    : sorted[middle]!;
}

function requiredNumber(parameters: URLSearchParams, name: string): number {
  const text = parameters.get(name);
  const value = text === null || text.trim() === "" ? Number.NaN : Number(text);
  if (!Number.isFinite(value)) {
    throw new Error(`OPT-0059 requires finite ${name}`);
  }
  return value;
}
