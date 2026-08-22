import {
  ACE_VAE_DEFAULT_QUANTUM_WORK_POLICY,
  AceCorrectnessVaeDecoderRuntime,
  planAceVaeDecoder,
  planAceVaeDecoderQuanta,
  type AceVaeDecoderConfig,
  type AceVaeDecoderOperation,
  type AceVaeDecoderQuantumPlan,
  type AceVaeDecoderQuantumWorkPolicy,
  type AceVaeLogicalTensorBinding,
  type AceVaeTransposePartGeometry,
} from "../../src/webgpu/vae-decoder.js";
import {
  AceCorrectnessVaePrimitiveKernel,
  planAceVaeConv1d,
  planAceVaeConvTranspose1d,
  planAceVaePointwise,
} from "../../src/webgpu/kernels/vae-primitives.js";

type PolicyId = "legacy" | "candidate";
type RepresentativeCaseId =
  | "conv1d-conv1"
  | "conv-transpose-block3"
  | "snake-block1"
  | "add-block1";

export interface Opt0002MillisecondSummary {
  readonly count: number;
  readonly samples: readonly number[];
  readonly minimum: number;
  readonly median: number;
  readonly maximum: number;
  readonly range: number;
}

interface QuantumTiming {
  readonly quantumIndex: number;
  readonly encodeMilliseconds: number;
  readonly submitMilliseconds: number;
  readonly drainMilliseconds: number;
  readonly explicitIdleMilliseconds?: number;
}

interface ExecutionTiming {
  readonly wallMilliseconds: number;
  readonly activeWallMilliseconds: number;
  readonly encodeMilliseconds: number;
  readonly submitMilliseconds: number;
  readonly drainMilliseconds: number;
  readonly explicitIdleMilliseconds: number;
  readonly commandBufferCount: number;
  readonly primitiveDispatchCount: number;
  readonly maximumSingleDrainMilliseconds: number;
  readonly quanta: readonly QuantumTiming[];
}

interface AttributedExecution extends ExecutionTiming {
  readonly roundIndex: number;
  readonly pairedOrder: string;
  readonly orderPosition: number;
}

interface PolicyExecutionSummary {
  readonly sampleCount: number;
  readonly commandBufferCountPerExecution: number;
  readonly primitiveDispatchCountPerExecution: number;
  readonly wallMilliseconds: Opt0002MillisecondSummary;
  readonly activeWallMilliseconds: Opt0002MillisecondSummary;
  readonly encodeMilliseconds: Opt0002MillisecondSummary;
  readonly submitMilliseconds: Opt0002MillisecondSummary;
  readonly drainMilliseconds: Opt0002MillisecondSummary;
  readonly explicitIdleMilliseconds: Opt0002MillisecondSummary;
  readonly maximumSingleDrainMilliseconds: Opt0002MillisecondSummary;
}

interface OutputFingerprint {
  readonly elementCount: number;
  readonly finiteCount: number;
  readonly nonzeroCount: number;
  readonly fnv1a32: string;
}

interface HeartbeatSummary {
  readonly animationFrameCount: number;
  readonly timerTickCount: number;
  readonly maximumAnimationFrameGapMilliseconds: number;
  readonly maximumTimerGapMilliseconds: number;
}

interface RepresentativeFixture {
  readonly id: RepresentativeCaseId;
  readonly operationIndex: number;
  readonly operationLabel: string;
  readonly operation: AceVaeDecoderOperation;
  readonly logicalOutputBase: number;
  readonly logicalOutputCount: number;
  readonly estimatedMaximumMultiplyAccumulates: number;
  readonly legacyQuanta: readonly AceVaeDecoderQuantumPlan[];
  readonly candidateQuanta: readonly AceVaeDecoderQuantumPlan[];
}

interface DirectDispatch {
  readonly primitiveCount: number;
  encode(pass: GPUComputePassEncoder): void;
}

interface PreparedRepresentativeCase {
  readonly fixture: RepresentativeFixture;
  readonly output: GPUBuffer;
  readonly legacy: readonly DirectDispatch[];
  readonly candidate: readonly DirectDispatch[];
  readonly owned: readonly GPUBuffer[];
  readonly kernel: AceCorrectnessVaePrimitiveKernel;
}

const EXPLICIT_IDLE_MILLISECONDS = 1;
const TOY_INPUT_FRAMES = 262_145;
const SENTINEL_BITS = 0x7fc0_0000;
const FLOAT32_BYTES = Float32Array.BYTES_PER_ELEMENT;

/** Test-only oracle for the retired Stage-1 uniform 32,768-output policy. */
export const OPT_0002_LEGACY_UNIFORM_WORK_POLICY = Object.freeze({
  maximumConvolutionMultiplyAccumulates: 0xffff_ffff,
  maximumOutputElements: 32_768,
}) satisfies AceVaeDecoderQuantumWorkPolicy;

export const OPT_0002_PAIRED_ORDERS = Object.freeze([
  Object.freeze(["legacy", "candidate"]),
  Object.freeze(["candidate", "legacy"]),
  Object.freeze(["candidate", "legacy"]),
  Object.freeze(["legacy", "candidate"]),
] satisfies readonly (readonly PolicyId[])[]);

export const OPT_0002_REPRESENTATIVE_OPERATIONS = Object.freeze([
  Object.freeze({ id: "conv1d-conv1", label: "conv1" }),
  Object.freeze({ id: "conv-transpose-block3", label: "block-3-conv-t1" }),
  Object.freeze({ id: "snake-block1", label: "block-1-snake1" }),
  Object.freeze({ id: "add-block1", label: "block-1-res-1-add" }),
] satisfies readonly {
  readonly id: RepresentativeCaseId;
  readonly label: string;
}[]);

export const OPT_0002_TRANSPOSE_PARTS = Object.freeze({
  "vae.decoder.block.0.conv_t1.weight": Object.freeze([
    Object.freeze({ partStart: 0, partEnd: 614 }),
    Object.freeze({ partStart: 614, partEnd: 1_024 }),
  ]),
  "vae.decoder.block.1.conv_t1.weight": Object.freeze([
    Object.freeze({ partStart: 0, partEnd: 512 }),
  ]),
  "vae.decoder.block.2.conv_t1.weight": Object.freeze([
    Object.freeze({ partStart: 0, partEnd: 256 }),
  ]),
  "vae.decoder.block.3.conv_t1.weight": Object.freeze([
    Object.freeze({ partStart: 0, partEnd: 128 }),
  ]),
  "vae.decoder.block.4.conv_t1.weight": Object.freeze([
    Object.freeze({ partStart: 0, partEnd: 128 }),
  ]),
}) satisfies Readonly<
  Record<string, readonly AceVaeTransposePartGeometry[]>
>;

export function summarizeOpt0002Milliseconds(
  samples: readonly number[],
): Opt0002MillisecondSummary {
  if (samples.length === 0) throw new RangeError("samples must not be empty");
  if (samples.some((sample) => !Number.isFinite(sample) || sample < 0)) {
    throw new RangeError("samples must be finite non-negative milliseconds");
  }
  const retained = [...samples];
  const sorted = [...retained].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  const median = sorted.length % 2 === 0
    ? (sorted[middle - 1]! + sorted[middle]!) / 2
    : sorted[middle]!;
  const minimum = sorted[0]!;
  const maximum = sorted.at(-1)!;
  return Object.freeze({
    count: retained.length,
    samples: Object.freeze(retained),
    minimum,
    median,
    maximum,
    range: maximum - minimum,
  });
}

export function createOpt0002StaticProtocol(): {
  readonly productionWindow: {
    readonly inputFrames: 256;
    readonly legacyQuantumCount: number;
    readonly legacyPrimitiveDispatchCount: number;
    readonly candidateQuantumCount: number;
    readonly candidatePrimitiveDispatchCount: number;
  };
  readonly representativeCases: readonly {
    readonly id: RepresentativeCaseId;
    readonly operationLabel: string;
    readonly operationKind: AceVaeDecoderOperation["kind"];
    readonly logicalOutputBase: number;
    readonly logicalOutputCount: number;
    readonly estimatedMaximumMultiplyAccumulates: number;
    readonly legacyQuantumCount: number;
    readonly candidateQuantumCount: number;
  }[];
} {
  const graph = planAceVaeDecoder(256);
  const legacy = planAceVaeDecoderQuanta(
    graph,
    OPT_0002_TRANSPOSE_PARTS,
    OPT_0002_LEGACY_UNIFORM_WORK_POLICY,
  );
  const candidate = planAceVaeDecoderQuanta(
    graph,
    OPT_0002_TRANSPOSE_PARTS,
    ACE_VAE_DEFAULT_QUANTUM_WORK_POLICY,
  );
  const fixtures = createRepresentativeFixtures(
    graph.operations,
    legacy.quanta,
    candidate.quanta,
  );
  return Object.freeze({
    productionWindow: Object.freeze({
      inputFrames: 256,
      legacyQuantumCount: legacy.quantumCount,
      legacyPrimitiveDispatchCount: legacy.primitiveDispatchCount,
      candidateQuantumCount: candidate.quantumCount,
      candidatePrimitiveDispatchCount: candidate.primitiveDispatchCount,
    }),
    representativeCases: Object.freeze(fixtures.map((fixture) =>
      Object.freeze({
        id: fixture.id,
        operationLabel: fixture.operationLabel,
        operationKind: fixture.operation.kind,
        logicalOutputBase: fixture.logicalOutputBase,
        logicalOutputCount: fixture.logicalOutputCount,
        estimatedMaximumMultiplyAccumulates:
          fixture.estimatedMaximumMultiplyAccumulates,
        legacyQuantumCount: fixture.legacyQuanta.length,
        candidateQuantumCount: fixture.candidateQuanta.length,
      })
    )),
  });
}

if (typeof document !== "undefined") installStartHandler();

function installStartHandler(): void {
  const start = document.querySelector<HTMLButtonElement>("#start");
  if (start === null) throw new Error("Missing start button");
  start.addEventListener("click", () => {
    start.disabled = true;
    document.body.dataset.status = "running";
    updateProgress("requesting WebGPU device");
    void run().then(
      (result) => finish("passed", result),
      (error: unknown) => finish("failed", {
        schema: "ace-opt-0002-vae-quantum-paired-ab-v1",
        status: "failed",
        experimentId: "OPT-0002",
        error: {
          name: error instanceof Error ? error.name : "Error",
          message: error instanceof Error ? error.message : String(error),
          ...(error instanceof Error && error.stack !== undefined
            ? { stack: error.stack }
            : {}),
        },
      }),
    );
  });
}

async function run(): Promise<unknown> {
  if (navigator.gpu === undefined) throw new Error("WebGPU unavailable");
  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: "high-performance",
  });
  if (adapter === null) throw new Error("No WebGPU adapter");
  const requiredBindingBytes = 128 * 1024 * 1024;
  if (
    adapter.limits.maxBufferSize < requiredBindingBytes ||
    adapter.limits.maxStorageBufferBindingSize < requiredBindingBytes
  ) {
    throw new Error("WebGPU adapter cannot bind the 128 MiB representative fixture");
  }
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxBufferSize: requiredBindingBytes,
      maxStorageBufferBindingSize: requiredBindingBytes,
    },
  });
  const heartbeat = startHeartbeat();
  try {
    const staticProtocol = createOpt0002StaticProtocol();
    updateProgress("validating complete forced-multirange toy decoder");
    const toyCorrectness = await runToyCorrectness(device);
    const graph = planAceVaeDecoder(256);
    const legacyPlan = planAceVaeDecoderQuanta(
      graph,
      OPT_0002_TRANSPOSE_PARTS,
      OPT_0002_LEGACY_UNIFORM_WORK_POLICY,
    );
    const candidatePlan = planAceVaeDecoderQuanta(
      graph,
      OPT_0002_TRANSPOSE_PARTS,
      ACE_VAE_DEFAULT_QUANTUM_WORK_POLICY,
    );
    const fixtures = createRepresentativeFixtures(
      graph.operations,
      legacyPlan.quanta,
      candidatePlan.quanta,
    );
    const cases = [];
    for (const [caseIndex, fixture] of fixtures.entries()) {
      updateProgress(
        `case ${caseIndex + 1}/${fixtures.length}: ${fixture.id} preparation`,
      );
      const prepared = await prepareRepresentativeCase(device, fixture);
      try {
        cases.push(await runRepresentativeCase(device, prepared, caseIndex));
      } finally {
        prepared.kernel.destroy();
        for (const buffer of prepared.owned) buffer.destroy();
      }
      await yieldToPage();
    }
    const heartbeatResult = heartbeat.stop();
    const adapterInfo = adapter.info;
    return {
      schema: "ace-opt-0002-vae-quantum-paired-ab-v1",
      status: "passed",
      experimentId: "OPT-0002",
      classification: "measurement-only-no-production-math-change",
      recordedAt: new Date().toISOString(),
      browser: {
        userAgent: navigator.userAgent,
        page: location.href,
      },
      adapter: {
        vendor: adapterInfo.vendor,
        architecture: adapterInfo.architecture,
        device: adapterInfo.device,
        description: adapterInfo.description,
        isFallbackAdapter: adapterInfo.isFallbackAdapter,
        ...(adapterInfo.subgroupMinSize === undefined
          ? {}
          : { subgroupMinSize: adapterInfo.subgroupMinSize }),
        ...(adapterInfo.subgroupMaxSize === undefined
          ? {}
          : { subgroupMaxSize: adapterInfo.subgroupMaxSize }),
        features: [...adapter.features].sort(),
        limits: {
          maxBufferSize: adapter.limits.maxBufferSize,
          maxStorageBufferBindingSize:
            adapter.limits.maxStorageBufferBindingSize,
          maxComputeWorkgroupStorageSize:
            adapter.limits.maxComputeWorkgroupStorageSize,
          maxComputeInvocationsPerWorkgroup:
            adapter.limits.maxComputeInvocationsPerWorkgroup,
          maxComputeWorkgroupSizeX: adapter.limits.maxComputeWorkgroupSizeX,
          maxComputeWorkgroupsPerDimension:
            adapter.limits.maxComputeWorkgroupsPerDimension,
        },
      },
      protocol: {
        fixtureId: "opt-0002-production-shape-zero-data-v1",
        thermalGate: "external-30-continuous-nominal-seconds-required",
        visiblePageRequired: true,
        pairedOrders: OPT_0002_PAIRED_ORDERS.map((order) => order.join("-")),
        samplesPerPolicyPerCase: OPT_0002_PAIRED_ORDERS.length,
        warmupExecutionsPerPolicyPerCase: 1,
        oneCommandBufferOutstanding: true,
        queueDrainAfterEveryCommandBuffer: true,
        queueEmptyIdleMillisecondsRequested: EXPLICIT_IDLE_MILLISECONDS,
        idleAfterFinalRepresentativeQuantum: false,
        authoritativeTiming: "performance.now-wall-clock",
        legacyPolicy: OPT_0002_LEGACY_UNIFORM_WORK_POLICY,
        candidatePolicy: ACE_VAE_DEFAULT_QUANTUM_WORK_POLICY,
      },
      staticProductionWindow: staticProtocol.productionWindow,
      toyCorrectness,
      cases,
      responsiveness: heartbeatResult,
    };
  } finally {
    heartbeat.stop();
    device.destroy();
  }
}

function createRepresentativeFixtures(
  operations: readonly AceVaeDecoderOperation[],
  legacyQuanta: readonly AceVaeDecoderQuantumPlan[],
  candidateQuanta: readonly AceVaeDecoderQuantumPlan[],
): readonly RepresentativeFixture[] {
  return Object.freeze(OPT_0002_REPRESENTATIVE_OPERATIONS.map((selection) => {
    const operationIndex = operations.findIndex((operation) =>
      operation.label === selection.label
    );
    if (operationIndex < 0) {
      throw new Error(`Missing representative operation ${selection.label}`);
    }
    const operation = operations[operationIndex]!;
    const allCandidate = candidateQuanta.filter((quantum) =>
      quantum.operationIndex === operationIndex
    );
    if (allCandidate.length === 0) {
      throw new Error(`No candidate quanta for ${selection.label}`);
    }
    const maximumCount = Math.max(...allCandidate.map((quantum) =>
      quantum.logicalOutputCount
    ));
    const candidate = allCandidate.find((quantum) =>
      quantum.logicalOutputCount === maximumCount
    )!;
    const targetEnd = candidate.logicalOutputBase + candidate.logicalOutputCount;
    const legacy = legacyQuanta.filter((quantum) =>
      quantum.operationIndex === operationIndex &&
      quantum.logicalOutputBase >= candidate.logicalOutputBase &&
      quantum.logicalOutputBase < targetEnd
    );
    requireExactQuantumCoverage(
      selection.label,
      candidate.logicalOutputBase,
      candidate.logicalOutputCount,
      legacy,
    );
    requireExactQuantumCoverage(
      selection.label,
      candidate.logicalOutputBase,
      candidate.logicalOutputCount,
      [candidate],
    );
    return Object.freeze({
      id: selection.id,
      operationIndex,
      operationLabel: selection.label,
      operation,
      logicalOutputBase: candidate.logicalOutputBase,
      logicalOutputCount: candidate.logicalOutputCount,
      estimatedMaximumMultiplyAccumulates:
        candidate.estimatedMaximumMultiplyAccumulates,
      legacyQuanta: Object.freeze(legacy),
      candidateQuanta: Object.freeze([candidate]),
    });
  }));
}

function requireExactQuantumCoverage(
  label: string,
  targetBase: number,
  targetCount: number,
  quanta: readonly AceVaeDecoderQuantumPlan[],
): void {
  let cursor = targetBase;
  for (const quantum of quanta) {
    if (quantum.logicalOutputBase !== cursor) {
      throw new Error(`${label} A/B ranges are not contiguous at ${cursor}`);
    }
    cursor += quantum.logicalOutputCount;
  }
  if (cursor !== targetBase + targetCount) {
    throw new Error(`${label} A/B ranges do not cover the same logical output`);
  }
}

async function runToyCorrectness(device: GPUDevice): Promise<unknown> {
  const config: AceVaeDecoderConfig = Object.freeze({
    id: "opt-0002-forced-multirange-toy",
    decoderInputChannels: 2,
    decoderChannels: 2,
    audioChannels: 2,
    channelMultiples: Object.freeze([1]),
    downsamplingRatios: Object.freeze([2]),
    sampleRateHz: 48_000,
  });
  const plan = planAceVaeDecoder(TOY_INPUT_FRAMES, config);
  const inputValues = Float32Array.from(
    { length: plan.inputElements },
    (_, index) => Math.fround((((index * 17 + 3) % 31) - 15) / 32),
  );
  const tensorValues = createToyTensorValues(plan.operations);
  const owned: GPUBuffer[] = [];
  const own = (buffer: GPUBuffer): GPUBuffer => {
    owned.push(buffer);
    return buffer;
  };
  const input = own(storageBuffer(device, inputValues));
  const output = own(outputBuffer(device, plan.outputElements));
  const workspaces = [
    own(outputBuffer(device, plan.maximumActivationElements)),
    own(outputBuffer(device, plan.maximumActivationElements)),
    own(outputBuffer(device, plan.maximumActivationElements)),
  ] as const;
  const tensors = Object.fromEntries(
    [...tensorValues].map(([name, values]) => {
      const operation = plan.operations.find((candidate) =>
        candidate.kind === "conv-transpose1d" && candidate.weight === name
      );
      if (operation?.kind !== "conv-transpose1d") {
        return [name, binding(own(storageBuffer(device, values)))] as const;
      }
      const rowElements =
        operation.shape.kernelSize * operation.shape.inputChannels;
      return [name, [
        Object.freeze({
          binding: binding(own(storageBuffer(
            device,
            values.subarray(0, rowElements),
          ))),
          partStart: 0,
          partEnd: 1,
        }),
        Object.freeze({
          binding: binding(own(storageBuffer(
            device,
            values.subarray(rowElements),
          ))),
          partStart: 1,
          partEnd: 2,
        }),
      ]] as const;
    }),
  ) as Readonly<Record<string, AceVaeLogicalTensorBinding>>;
  const runtime = AceCorrectnessVaeDecoderRuntime.create(device);
  try {
    const bindings = {
      input: binding(input),
      output: binding(output),
      workspaces: [
        binding(workspaces[0]),
        binding(workspaces[1]),
        binding(workspaces[2]),
      ],
      tensors,
    } as const;
    const legacy = await runtime.createDecoderDispatch(
      "opt-0002-toy-legacy",
      TOY_INPUT_FRAMES,
      bindings,
      config,
      1,
      { quantumWorkPolicy: OPT_0002_LEGACY_UNIFORM_WORK_POLICY },
    );
    const candidate = await runtime.createDecoderDispatch(
      "opt-0002-toy-candidate",
      TOY_INPUT_FRAMES,
      bindings,
      config,
      1,
      { quantumWorkPolicy: ACE_VAE_DEFAULT_QUANTUM_WORK_POLICY },
    );
    if (
      legacy.quanta.length <= plan.primitiveCount ||
      candidate.quanta.length <= plan.primitiveCount
    ) {
      throw new Error("Toy decoder did not force multiple ranges under both policies");
    }
    const transposeCandidate = candidate.quanta.filter((quantum) =>
      quantum.operationKind === "conv-transpose1d"
    );
    if (
      transposeCandidate.length < 2 ||
      transposeCandidate.some((quantum) => quantum.primitiveCount !== 2)
    ) {
      throw new Error("Toy candidate did not preserve both transpose parts per row band");
    }
    const sentinel = sentinelArray(plan.outputElements);
    writeTypedArray(device.queue, output, 0, sentinel);
    await device.queue.onSubmittedWorkDone();
    const legacyTiming = await executeDispatches(
      device,
      legacy.quanta,
      legacy.primitiveCount,
      true,
    );
    const legacyOutput = await readOutput(
      device,
      output,
      0,
      plan.outputElements,
    );
    requireFiniteOutput("toy legacy", legacyOutput);
    writeTypedArray(device.queue, output, 0, sentinel);
    await device.queue.onSubmittedWorkDone();
    const candidateTiming = await executeDispatches(
      device,
      candidate.quanta,
      candidate.primitiveCount,
      true,
    );
    const candidateOutput = await readOutput(
      device,
      output,
      0,
      plan.outputElements,
    );
    requireFiniteOutput("toy candidate", candidateOutput);
    const bitMismatchCount = countBitMismatches(legacyOutput, candidateOutput);
    if (bitMismatchCount !== 0) {
      throw new Error(`Toy decoder A/B differs in ${bitMismatchCount} FP32 values`);
    }
    return {
      fixtureId: "opt-0002-forced-multirange-toy-v1",
      inputFrames: TOY_INPUT_FRAMES,
      outputElements: plan.outputElements,
      operationCount: plan.primitiveCount,
      bothPoliciesForcedMultipleRanges: true,
      transposePhysicalPartsPerQuantum: 2,
      bitIdentical: true,
      bitMismatchCount,
      fingerprint: fingerprint(legacyOutput),
      legacy: {
        quantumCount: legacy.quanta.length,
        primitiveDispatchCount: legacy.primitiveCount,
        timing: legacyTiming,
      },
      candidate: {
        quantumCount: candidate.quanta.length,
        primitiveDispatchCount: candidate.primitiveCount,
        timing: candidateTiming,
      },
    };
  } finally {
    runtime.destroy();
    for (const buffer of owned) buffer.destroy();
  }
}

function createToyTensorValues(
  operations: readonly AceVaeDecoderOperation[],
): ReadonlyMap<string, Float32Array> {
  const tensors = new Map<string, Float32Array>();
  const add = (name: string, elements: number, scale: number, offset = 0): void => {
    if (tensors.has(name)) return;
    const seed = stableHash(name);
    tensors.set(name, Float32Array.from(
      { length: elements },
      (_, index) =>
        Math.fround((((seed + index * 17) % 19) - 9) * scale + offset),
    ));
  };
  for (const operation of operations) {
    switch (operation.kind) {
      case "conv1d":
      case "conv-transpose1d":
        add(
          operation.weight,
          operation.shape.outputChannels * operation.shape.kernelSize *
            operation.shape.inputChannels,
          0.0125,
        );
        if (operation.bias !== undefined) {
          add(operation.bias, operation.shape.outputChannels, 0.005);
        }
        break;
      case "snake":
        add(operation.alpha, operation.shape.channels, 0.015, -0.8);
        add(operation.beta, operation.shape.channels, 0.0125, 0.1);
        break;
      case "add":
        break;
    }
  }
  return tensors;
}

async function prepareRepresentativeCase(
  device: GPUDevice,
  fixture: RepresentativeFixture,
): Promise<PreparedRepresentativeCase> {
  const kernel = AceCorrectnessVaePrimitiveKernel.create(device);
  const owned: GPUBuffer[] = [];
  const own = (buffer: GPUBuffer): GPUBuffer => {
    owned.push(buffer);
    return buffer;
  };
  const operation = fixture.operation;
  try {
    let output: GPUBuffer;
    let createForQuanta: (
      quanta: readonly AceVaeDecoderQuantumPlan[],
      policy: PolicyId,
    ) => Promise<readonly DirectDispatch[]>;
    switch (operation.kind) {
      case "conv1d": {
        const plan = planAceVaeConv1d(operation.shape);
        const input = own(zeroStorageBuffer(device, plan.inputElements));
        const weight = own(zeroStorageBuffer(device, plan.weightElements));
        const bias = operation.bias === undefined
          ? undefined
          : own(zeroStorageBuffer(device, operation.shape.outputChannels));
        output = own(outputBuffer(device, plan.outputElements));
        createForQuanta = async (quanta, policy) =>
          await Promise.all(quanta.map(async (quantum, index) => {
            const primitive = requireSinglePrimitive(fixture, quantum);
            const control = own(rangeControlBuffer(
              device,
              primitive.outputBase,
              primitive.outputCount,
            ));
            const dispatch = await kernel.createConv1dDispatch(
              `opt-0002-${fixture.id}-${policy}-${index}`,
              operation.shape,
              {
                input: binding(input),
                weight: binding(weight),
                ...(bias === undefined ? {} : { bias: binding(bias) }),
                output: binding(output),
              },
              {
                base: primitive.outputBase,
                count: primitive.outputCount,
                control: { buffer: control, offset: 0, size: 16 },
              },
            );
            return Object.freeze({
              primitiveCount: 1,
              encode(pass: GPUComputePassEncoder): void {
                dispatch.encode(pass);
              },
            });
          }));
        break;
      }
      case "conv-transpose1d": {
        const plan = planAceVaeConvTranspose1d(operation.shape);
        const input = own(zeroStorageBuffer(device, plan.inputElements));
        const weight = own(zeroStorageBuffer(device, plan.weightElements));
        const bias = own(zeroStorageBuffer(device, operation.shape.outputChannels));
        output = own(outputBuffer(device, plan.outputElements));
        createForQuanta = async (quanta, policy) =>
          await Promise.all(quanta.map(async (quantum, index) => {
            const primitive = requireSinglePrimitive(fixture, quantum);
            if (
              primitive.firstOutputChannel !== 0 ||
              primitive.outputChannels !== operation.shape.outputChannels
            ) {
              throw new Error(`${fixture.id} requires one complete physical part`);
            }
            const control = own(rangeControlBuffer(
              device,
              primitive.outputBase,
              primitive.outputCount,
            ));
            const dispatch = await kernel.createConvTranspose1dDispatch(
              `opt-0002-${fixture.id}-${policy}-${index}`,
              operation.shape,
              {
                input: binding(input),
                weight: binding(weight),
                bias: binding(bias),
                output: binding(output),
              },
              {
                base: primitive.outputBase,
                count: primitive.outputCount,
                control: { buffer: control, offset: 0, size: 16 },
              },
            );
            return Object.freeze({
              primitiveCount: 1,
              encode(pass: GPUComputePassEncoder): void {
                dispatch.encode(pass);
              },
            });
          }));
        break;
      }
      case "snake": {
        const plan = planAceVaePointwise(operation.shape, "OPT-0002 Snake");
        const input = own(zeroStorageBuffer(device, plan.elements));
        const alpha = own(zeroStorageBuffer(device, operation.shape.channels));
        const beta = own(zeroStorageBuffer(device, operation.shape.channels));
        output = own(outputBuffer(device, plan.elements));
        createForQuanta = async (quanta, policy) =>
          await Promise.all(quanta.map(async (quantum, index) => {
            const primitive = requireSinglePrimitive(fixture, quantum);
            const control = own(rangeControlBuffer(
              device,
              primitive.outputBase,
              primitive.outputCount,
            ));
            const dispatch = await kernel.createSnakeDispatch(
              `opt-0002-${fixture.id}-${policy}-${index}`,
              operation.shape,
              {
                input: binding(input),
                alpha: binding(alpha),
                beta: binding(beta),
                output: binding(output),
              },
              {
                base: primitive.outputBase,
                count: primitive.outputCount,
                control: { buffer: control, offset: 0, size: 16 },
              },
            );
            return Object.freeze({
              primitiveCount: 1,
              encode(pass: GPUComputePassEncoder): void {
                dispatch.encode(pass);
              },
            });
          }));
        break;
      }
      case "add": {
        const plan = planAceVaePointwise(operation.shape, "OPT-0002 add");
        const left = own(zeroStorageBuffer(device, plan.elements));
        const right = own(zeroStorageBuffer(device, plan.elements));
        output = own(outputBuffer(device, plan.elements));
        createForQuanta = async (quanta, policy) =>
          await Promise.all(quanta.map(async (quantum, index) => {
            const primitive = requireSinglePrimitive(fixture, quantum);
            const control = own(rangeControlBuffer(
              device,
              primitive.outputBase,
              primitive.outputCount,
            ));
            const dispatch = await kernel.createAddDispatch(
              `opt-0002-${fixture.id}-${policy}-${index}`,
              operation.shape,
              {
                left: binding(left),
                right: binding(right),
                output: binding(output),
              },
              {
                base: primitive.outputBase,
                count: primitive.outputCount,
                control: { buffer: control, offset: 0, size: 16 },
              },
            );
            return Object.freeze({
              primitiveCount: 1,
              encode(pass: GPUComputePassEncoder): void {
                dispatch.encode(pass);
              },
            });
          }));
        break;
      }
    }
    const legacy = await createForQuanta(fixture.legacyQuanta, "legacy");
    const candidate = await createForQuanta(
      fixture.candidateQuanta,
      "candidate",
    );
    await device.queue.onSubmittedWorkDone();
    return Object.freeze({ fixture, output, legacy, candidate, owned, kernel });
  } catch (error) {
    kernel.destroy();
    for (const buffer of owned) buffer.destroy();
    throw error;
  }
}

function requireSinglePrimitive(
  fixture: RepresentativeFixture,
  quantum: AceVaeDecoderQuantumPlan,
): AceVaeDecoderQuantumPlan["primitives"][number] {
  if (quantum.primitives.length !== 1) {
    throw new Error(`${fixture.id} representative quantum must have one primitive`);
  }
  return quantum.primitives[0]!;
}

async function runRepresentativeCase(
  device: GPUDevice,
  prepared: PreparedRepresentativeCase,
  caseIndex: number,
): Promise<unknown> {
  const { fixture } = prepared;
  const sentinel = sentinelArray(fixture.logicalOutputCount);
  const prefill = async (): Promise<void> => {
    writeTypedArray(
      device.queue,
      prepared.output,
      fixture.logicalOutputBase * FLOAT32_BYTES,
      sentinel,
    );
    await device.queue.onSubmittedWorkDone();
  };
  updateProgress(`case ${caseIndex + 1}: ${fixture.id} legacy warmup`);
  await prefill();
  const legacyWarmup = await executeDispatches(
    device,
    prepared.legacy,
    sumPrimitiveCounts(prepared.legacy),
    false,
  );
  const legacyOutput = await readOutput(
    device,
    prepared.output,
    fixture.logicalOutputBase,
    fixture.logicalOutputCount,
  );
  requireFiniteOutput(`${fixture.id} legacy`, legacyOutput);
  updateProgress(`case ${caseIndex + 1}: ${fixture.id} candidate warmup`);
  await prefill();
  const candidateWarmup = await executeDispatches(
    device,
    prepared.candidate,
    sumPrimitiveCounts(prepared.candidate),
    false,
  );
  const candidateOutput = await readOutput(
    device,
    prepared.output,
    fixture.logicalOutputBase,
    fixture.logicalOutputCount,
  );
  requireFiniteOutput(`${fixture.id} candidate`, candidateOutput);
  const bitMismatchCount = countBitMismatches(legacyOutput, candidateOutput);
  if (bitMismatchCount !== 0) {
    throw new Error(`${fixture.id} A/B differs in ${bitMismatchCount} FP32 values`);
  }

  const samples: Record<PolicyId, AttributedExecution[]> = {
    legacy: [],
    candidate: [],
  };
  const measuredHeartbeat = startHeartbeat();
  let measuredResponsiveness: HeartbeatSummary;
  try {
    for (const [roundIndex, order] of OPT_0002_PAIRED_ORDERS.entries()) {
      for (const [orderPosition, policy] of order.entries()) {
        updateProgress(
          `case ${caseIndex + 1}: ${fixture.id} round ${roundIndex + 1} ` +
            `${policy} (${orderPosition + 1}/2)`,
        );
        const dispatches = prepared[policy];
        const timing = await executeDispatches(
          device,
          dispatches,
          sumPrimitiveCounts(dispatches),
          false,
        );
        samples[policy].push(Object.freeze({
          ...timing,
          roundIndex,
          pairedOrder: order.join("-"),
          orderPosition,
        }));
        await yieldToPage();
      }
    }
  } finally {
    measuredResponsiveness = measuredHeartbeat.stop();
  }
  return {
    id: fixture.id,
    operationIndex: fixture.operationIndex,
    operationLabel: fixture.operationLabel,
    operationKind: fixture.operation.kind,
    shape: fixture.operation.shape,
    target: {
      logicalOutputBase: fixture.logicalOutputBase,
      logicalOutputCount: fixture.logicalOutputCount,
      estimatedMaximumMultiplyAccumulates:
        fixture.estimatedMaximumMultiplyAccumulates,
    },
    correctness: {
      outputPrefill: "quiet-NaN-u32-sentinel",
      fullMeasuredRangeFinite: true,
      bitIdentical: true,
      bitMismatchCount,
      legacyFingerprint: fingerprint(legacyOutput),
      candidateFingerprint: fingerprint(candidateOutput),
    },
    legacy: {
      warmup: legacyWarmup,
      samples: samples.legacy,
      summary: summarizeExecutions(samples.legacy),
    },
    candidate: {
      warmup: candidateWarmup,
      samples: samples.candidate,
      summary: summarizeExecutions(samples.candidate),
    },
    measuredResponsiveness,
    delta: {
      commandBufferReduction:
        prepared.legacy.length - prepared.candidate.length,
      commandBufferReductionRatio:
        prepared.legacy.length / prepared.candidate.length,
      medianWallSpeedup:
        summarizeExecutions(samples.legacy).wallMilliseconds.median /
        summarizeExecutions(samples.candidate).wallMilliseconds.median,
    },
  };
}

function summarizeExecutions(
  samples: readonly ExecutionTiming[],
): PolicyExecutionSummary {
  if (samples.length === 0) throw new RangeError("execution samples missing");
  const summarize = (select: (sample: ExecutionTiming) => number) =>
    summarizeOpt0002Milliseconds(samples.map(select));
  return Object.freeze({
    sampleCount: samples.length,
    commandBufferCountPerExecution: samples[0]!.commandBufferCount,
    primitiveDispatchCountPerExecution: samples[0]!.primitiveDispatchCount,
    wallMilliseconds: summarize((sample) => sample.wallMilliseconds),
    activeWallMilliseconds: summarize((sample) => sample.activeWallMilliseconds),
    encodeMilliseconds: summarize((sample) => sample.encodeMilliseconds),
    submitMilliseconds: summarize((sample) => sample.submitMilliseconds),
    drainMilliseconds: summarize((sample) => sample.drainMilliseconds),
    explicitIdleMilliseconds: summarize((sample) =>
      sample.explicitIdleMilliseconds
    ),
    maximumSingleDrainMilliseconds: summarize((sample) =>
      sample.maximumSingleDrainMilliseconds
    ),
  });
}

async function executeDispatches(
  device: GPUDevice,
  dispatches: readonly DirectDispatch[],
  primitiveDispatchCount: number,
  idleAfterFinal: boolean,
): Promise<ExecutionTiming> {
  const quanta: QuantumTiming[] = [];
  let encodeMilliseconds = 0;
  let submitMilliseconds = 0;
  let drainMilliseconds = 0;
  let explicitIdleMilliseconds = 0;
  const wallStart = performance.now();
  for (const [quantumIndex, dispatch] of dispatches.entries()) {
    const encodeStart = performance.now();
    const encoder = device.createCommandEncoder({
      label: `opt-0002-quantum-${quantumIndex}`,
    });
    const pass = encoder.beginComputePass();
    dispatch.encode(pass);
    pass.end();
    const commandBuffer = encoder.finish();
    const encoded = performance.now() - encodeStart;
    const submitStart = performance.now();
    device.queue.submit([commandBuffer]);
    const submitted = performance.now() - submitStart;
    const drainStart = performance.now();
    await device.queue.onSubmittedWorkDone();
    const drained = performance.now() - drainStart;
    const needsIdle = idleAfterFinal || quantumIndex + 1 < dispatches.length;
    let idle: number | undefined;
    if (needsIdle) {
      const idleStart = performance.now();
      await queueEmptyIdle();
      idle = performance.now() - idleStart;
      explicitIdleMilliseconds += idle;
    }
    encodeMilliseconds += encoded;
    submitMilliseconds += submitted;
    drainMilliseconds += drained;
    quanta.push(Object.freeze({
      quantumIndex,
      encodeMilliseconds: encoded,
      submitMilliseconds: submitted,
      drainMilliseconds: drained,
      ...(idle === undefined ? {} : { explicitIdleMilliseconds: idle }),
    }));
  }
  const wallMilliseconds = performance.now() - wallStart;
  return Object.freeze({
    wallMilliseconds,
    activeWallMilliseconds: Math.max(
      0,
      wallMilliseconds - explicitIdleMilliseconds,
    ),
    encodeMilliseconds,
    submitMilliseconds,
    drainMilliseconds,
    explicitIdleMilliseconds,
    commandBufferCount: dispatches.length,
    primitiveDispatchCount,
    maximumSingleDrainMilliseconds: Math.max(
      0,
      ...quanta.map((quantum) => quantum.drainMilliseconds),
    ),
    quanta: Object.freeze(quanta),
  });
}

function sumPrimitiveCounts(dispatches: readonly DirectDispatch[]): number {
  return dispatches.reduce(
    (total, dispatch) => total + dispatch.primitiveCount,
    0,
  );
}

function zeroStorageBuffer(device: GPUDevice, elements: number): GPUBuffer {
  return device.createBuffer({
    size: alignedSize(elements * FLOAT32_BYTES),
    usage: GPUBufferUsage.STORAGE,
  });
}

function storageBuffer(device: GPUDevice, data: Float32Array): GPUBuffer {
  const buffer = device.createBuffer({
    size: alignedSize(data.byteLength),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  writeTypedArray(device.queue, buffer, 0, data);
  return buffer;
}

function outputBuffer(device: GPUDevice, elements: number): GPUBuffer {
  return device.createBuffer({
    size: alignedSize(elements * FLOAT32_BYTES),
    usage:
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
}

function rangeControlBuffer(
  device: GPUDevice,
  outputBase: number,
  outputCount: number,
): GPUBuffer {
  const buffer = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(
    buffer,
    0,
    new Uint32Array([outputBase, outputCount, 0, 0]),
  );
  return buffer;
}

async function readOutput(
  device: GPUDevice,
  output: GPUBuffer,
  firstElement: number,
  elementCount: number,
): Promise<Float32Array> {
  const bytes = elementCount * FLOAT32_BYTES;
  const readback = device.createBuffer({
    size: alignedSize(bytes),
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  let mapped = false;
  try {
    const encoder = device.createCommandEncoder({ label: "opt-0002-readback" });
    encoder.copyBufferToBuffer(
      output,
      firstElement * FLOAT32_BYTES,
      readback,
      0,
      alignedSize(bytes),
    );
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    mapped = true;
    return Float32Array.from(
      new Float32Array(readback.getMappedRange(), 0, elementCount),
    );
  } finally {
    if (mapped) readback.unmap();
    readback.destroy();
  }
}

function sentinelArray(elements: number): Uint32Array {
  const sentinel = new Uint32Array(elements);
  sentinel.fill(SENTINEL_BITS);
  return sentinel;
}

function requireFiniteOutput(label: string, output: Float32Array): void {
  for (const [index, value] of output.entries()) {
    if (!Number.isFinite(value)) {
      throw new Error(`${label} left a non-finite sentinel at ${index}`);
    }
  }
}

function countBitMismatches(
  left: Float32Array,
  right: Float32Array,
): number {
  if (left.length !== right.length) {
    throw new Error("Cannot compare outputs with different lengths");
  }
  const leftBits = new Uint32Array(
    left.buffer,
    left.byteOffset,
    left.length,
  );
  const rightBits = new Uint32Array(
    right.buffer,
    right.byteOffset,
    right.length,
  );
  let mismatchCount = 0;
  for (let index = 0; index < leftBits.length; index += 1) {
    if (leftBits[index] !== rightBits[index]) mismatchCount += 1;
  }
  return mismatchCount;
}

function fingerprint(values: Float32Array): OutputFingerprint {
  const bits = new Uint32Array(
    values.buffer,
    values.byteOffset,
    values.length,
  );
  let finiteCount = 0;
  let nonzeroCount = 0;
  let hash = 2166136261;
  for (let index = 0; index < values.length; index += 1) {
    const value = values[index]!;
    if (Number.isFinite(value)) finiteCount += 1;
    if (value !== 0) nonzeroCount += 1;
    hash = Math.imul(hash ^ bits[index]!, 16777619) >>> 0;
  }
  return Object.freeze({
    elementCount: values.length,
    finiteCount,
    nonzeroCount,
    fnv1a32: hash.toString(16).padStart(8, "0"),
  });
}

function binding(buffer: GPUBuffer): GPUBufferBinding {
  return { buffer, offset: 0, size: buffer.size };
}

function alignedSize(bytes: number): number {
  return Math.max(4, Math.ceil(bytes / 4) * 4);
}

function writeTypedArray(
  queue: GPUQueue,
  buffer: GPUBuffer,
  offset: number,
  data: Float32Array | Uint32Array,
): void {
  const owned = new Uint8Array(data.byteLength);
  owned.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
  queue.writeBuffer(buffer, offset, owned);
}

function stableHash(value: string): number {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    hash = Math.imul(hash ^ value.charCodeAt(index), 16777619) >>> 0;
  }
  return hash;
}

async function queueEmptyIdle(): Promise<void> {
  await new Promise<void>((resolve) => {
    setTimeout(resolve, EXPLICIT_IDLE_MILLISECONDS);
  });
}

async function yieldToPage(): Promise<void> {
  await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
}

function startHeartbeat(): { stop(): HeartbeatSummary } {
  const animationGaps: number[] = [];
  const timerGaps: number[] = [];
  let stopped = false;
  let animationFrame = 0;
  let timerTick = 0;
  let lastAnimation = performance.now();
  let lastTimer = performance.now();
  let frameHandle = 0;
  const frame = (now: number): void => {
    if (stopped) return;
    animationGaps.push(now - lastAnimation);
    lastAnimation = now;
    animationFrame += 1;
    frameHandle = requestAnimationFrame(frame);
  };
  frameHandle = requestAnimationFrame(frame);
  const timerHandle = setInterval(() => {
    const now = performance.now();
    timerGaps.push(now - lastTimer);
    lastTimer = now;
    timerTick += 1;
  }, 10);
  const result = (): HeartbeatSummary => Object.freeze({
    animationFrameCount: animationFrame,
    timerTickCount: timerTick,
    maximumAnimationFrameGapMilliseconds: Math.max(0, ...animationGaps),
    maximumTimerGapMilliseconds: Math.max(0, ...timerGaps),
  });
  return {
    stop(): HeartbeatSummary {
      if (!stopped) {
        stopped = true;
        cancelAnimationFrame(frameHandle);
        clearInterval(timerHandle);
      }
      return result();
    },
  };
}

function updateProgress(message: string): void {
  const node = document.querySelector<HTMLElement>("#progress");
  if (node === null) throw new Error("Missing progress node");
  node.textContent = message;
}

function finish(status: "passed" | "failed", result: unknown): void {
  document.body.dataset.status = status;
  updateProgress(status);
  const node = document.querySelector<HTMLPreElement>("#result");
  if (node === null) throw new Error("Missing result node");
  node.textContent = JSON.stringify(result);
}
