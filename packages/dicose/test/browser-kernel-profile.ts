import type { GpuWeightTensor } from "../src/model/package.js";
import {
  GpuOps,
  type Activation,
  type AttentionGeometry,
  type AttentionKernel,
  type PackedAccumulation,
} from "../src/webgpu/ops.js";
import { createF16Tensor, destroyTensors, type GpuTensor } from "../src/webgpu/tensor.js";

const ROWS = 62 * 1_189;
const WARMUP_PASSES = 2;
const MEASURED_PASSES = 7;
const DENSE_ACCUMULATIONS = ["exact", "k2", "k4"] as const;
const DENSE_MEASURED_ORDERS: readonly (readonly PackedAccumulation[])[] = [
  ["exact", "k2", "k4"],
  ["k2", "k4", "exact"],
  ["k4", "exact", "k2"],
  ["k4", "k2", "exact"],
  ["k2", "exact", "k4"],
  ["exact", "k4", "k2"],
];
const PROFILE_FOCUS = new URL(location.href).searchParams.get("focus") ?? "all";

interface TimedSampleSummary {
  readonly samplesMs: readonly number[];
  readonly medianMs: number;
  readonly minMs: number;
  readonly maxMs: number;
}

interface DenseProfile extends TimedSampleSummary {
  readonly name: string;
  readonly accumulation: PackedAccumulation;
  readonly rows: number;
  readonly inner: number;
  readonly columns: number;
  readonly productionMultiplicity: number;
  readonly projectedGpuMs: number;
  readonly logicalGflop: number;
  readonly medianTflops: number;
}

interface DenseWeightedProfile {
  readonly accumulation: PackedAccumulation;
  readonly projectedGpuMs: number;
  readonly speedupVsExact: number;
  readonly savedGpuMsVsExact: number;
}

interface AttentionProfile extends TimedSampleSummary {
  readonly name: string;
  readonly kernel: AttentionKernel;
  readonly sequences: number;
  readonly tokens: number;
  readonly strided: boolean;
  readonly logicalGflop: number;
  readonly medianTflops: number;
}

interface ConvProfile extends TimedSampleSummary {
  readonly name: string;
  readonly owner: "auto" | "generic";
  readonly height: number;
  readonly width: number;
  readonly inChannels: number;
  readonly outChannels: number;
  readonly kernel: 1 | 3;
  readonly rows: number;
  readonly logicalGflop: number;
  readonly medianTflops: number;
}

interface RmsNormProfile extends TimedSampleSummary {
  readonly name: string;
  readonly owner: "auto" | "row1";
  readonly rows: number;
  readonly columns: number;
  readonly hasMapping: boolean;
  readonly workgroups: number;
  readonly logicalGb: number;
  readonly medianGbps: number;
}

interface KernelProfileReport {
  readonly ok: boolean;
  readonly adapter?: Readonly<{
    readonly features: readonly string[];
    readonly maxBufferSize: number;
    readonly maxStorageBufferBindingSize: number;
  }>;
  readonly methodology?: Readonly<{
    readonly rows: number;
    readonly warmupPasses: number;
    readonly measuredPasses: number;
    readonly densePairedRounds: number;
    readonly focus: string;
    readonly timing: string;
  }>;
  readonly dense?: readonly DenseProfile[];
  readonly denseWeighted?: readonly DenseWeightedProfile[];
  readonly conv?: readonly ConvProfile[];
  readonly rmsNorm?: readonly RmsNormProfile[];
  readonly attention?: readonly AttentionProfile[];
  readonly uncapturedErrors?: readonly string[];
  readonly error?: string;
}

interface DenseCase {
  readonly name: string;
  readonly inner: number;
  readonly columns: number;
  readonly tileColumns: 128 | 256;
  readonly productionMultiplicity: number;
  readonly activation?: Activation;
  readonly residual?: boolean;
  readonly rotaryKeys?: AttentionGeometry;
}

const statusNode = document.querySelector<HTMLElement>("#status");
const resultNode = document.querySelector<HTMLElement>("#result");

void run();

async function run(): Promise<void> {
  let device: GPUDevice | undefined;
  let ops: GpuOps | undefined;
  const uncapturedErrors: string[] = [];
  try {
    if (!["all", "dense", "conv", "norm", "attention"].includes(PROFILE_FOCUS)) {
      throw new Error(`Invalid kernel profile focus ${PROFILE_FOCUS}`);
    }
    const adapter = await navigator.gpu?.requestAdapter({ powerPreference: "high-performance" });
    if (adapter === undefined || adapter === null) throw new Error("WebGPU adapter is unavailable");
    for (const feature of ["shader-f16", "subgroups", "timestamp-query"] as const) {
      if (!adapter.features.has(feature)) throw new Error(`WebGPU adapter lacks ${feature}`);
    }
    device = await adapter.requestDevice({
      requiredFeatures: ["shader-f16", "subgroups", "timestamp-query"],
      requiredLimits: {
        maxBufferSize: adapter.limits.maxBufferSize,
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
      },
    });
    device.addEventListener("uncapturederror", (event) => {
      uncapturedErrors.push(event.error.message);
    });
    ops = new GpuOps(device);

    const denseCases: readonly DenseCase[] = [
      {
        name: "ff-up-384x1536-gelu",
        inner: 384,
        columns: 1_536,
        tileColumns: 256,
        productionMultiplicity: 80,
        activation: "gelu",
      },
      {
        name: "qkv-384x1536-rotary-time",
        inner: 384,
        columns: 1_536,
        tileColumns: 256,
        productionMultiplicity: 80,
        rotaryKeys: { sequences: 62, tokens: 1_189, strided: true },
      },
      {
        name: "ff-down-1536x384-residual",
        inner: 1_536,
        columns: 384,
        tileColumns: 128,
        productionMultiplicity: 80,
        residual: true,
      },
      {
        name: "attention-out-512x384-residual",
        inner: 512,
        columns: 384,
        tileColumns: 128,
        productionMultiplicity: 80,
        residual: true,
      },
      {
        name: "adapter-384x384",
        inner: 384,
        columns: 384,
        tileColumns: 128,
        productionMultiplicity: 34,
      },
    ];

    const dense: DenseProfile[] = [];
    if (PROFILE_FOCUS === "all" || PROFILE_FOCUS === "dense") {
      for (const profileCase of denseCases) {
        setStatus(`Profiling paired exact/K2/K4 ${profileCase.name}…`);
        dense.push(...await profileDenseModes(device, ops, profileCase));
      }
    }
    const exactProjectedGpuMs = projectedDenseGpuMs(dense, "exact");
    const denseWeighted: DenseWeightedProfile[] = dense.length === 0
      ? []
      : DENSE_ACCUMULATIONS.map((accumulation) => {
        const projectedGpuMs = projectedDenseGpuMs(dense, accumulation);
        return {
          accumulation,
          projectedGpuMs,
          speedupVsExact: exactProjectedGpuMs / projectedGpuMs,
          savedGpuMsVsExact: exactProjectedGpuMs - projectedGpuMs,
        };
      });

    const conv: ConvProfile[] = [];
    if (PROFILE_FOCUS === "all" || PROFILE_FOCUS === "conv") {
      setStatus("Profiling 4→128 3×3 convolution…");
      conv.push(...await profileConv3x3Entry(device, ops));
      setStatus("Profiling 128×128 1×1 convolution…");
      conv.push(...await profileConv1x1(device, ops));
      setStatus("Profiling 128→4 3×3 convolution…");
      conv.push(...await profileConv3x3Exit(device, ops));
    }

    const rmsNorm: RmsNormProfile[] = [];
    if (PROFILE_FOCUS === "all" || PROFILE_FOCUS === "norm") {
      setStatus("Profiling production-shape RMSNorm…");
      rmsNorm.push(...await profileRmsNorm(device, ops));
    }

    const attention: AttentionProfile[] = [];
    if (PROFILE_FOCUS === "all" || PROFILE_FOCUS === "attention") {
      for (const kernel of ["q64", "flash"] as const) {
        setStatus(`Profiling ${kernel} time-axis attention…`);
        attention.push(await profileAttention(device, ops, {
          name: `${kernel}-time-attention-62x1189`,
          kernel,
          sequences: 62,
          tokens: 1_189,
          strided: true,
        }));
        setStatus(`Profiling ${kernel} frequency-axis attention…`);
        attention.push(await profileAttention(device, ops, {
          name: `${kernel}-frequency-attention-1189x62`,
          kernel,
          sequences: 1_189,
          tokens: 62,
          strided: false,
        }));
      }
    }

    if (uncapturedErrors.length > 0) {
      throw new Error(`WebGPU emitted uncaptured errors: ${uncapturedErrors.join("; ")}`);
    }
    publish({
      ok: true,
      adapter: {
        features: [...adapter.features].sort(),
        maxBufferSize: adapter.limits.maxBufferSize,
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      },
      methodology: {
        rows: ROWS,
        warmupPasses: WARMUP_PASSES,
        measuredPasses: MEASURED_PASSES,
        densePairedRounds: DENSE_MEASURED_ORDERS.length,
        focus: PROFILE_FOCUS,
        timing: "one timestamped compute pass and one drained submission per sample; dense exact/K2/K4 uses a six-round balanced order; compilation, upload, submission, and readback excluded",
      },
      dense,
      denseWeighted,
      conv,
      rmsNorm,
      attention,
      uncapturedErrors,
    });
  } catch (error) {
    publish({
      ok: false,
      uncapturedErrors,
      error: error instanceof Error ? `${error.name}: ${error.message}` : String(error),
    });
  } finally {
    ops?.destroy();
    device?.destroy();
  }
}

async function profileRmsNorm(device: GPUDevice, ops: GpuOps): Promise<RmsNormProfile[]> {
  const tensors: GpuTensor[] = [];
  const columns = 384;
  try {
    const input = keep(tensors, createF16Tensor(device, ROWS * columns, "profile-rmsnorm-input"));
    const gammaTensor = keep(tensors, createF16Tensor(device, columns, "profile-rmsnorm-gamma"));
    const gamma: GpuWeightTensor = {
      ...gammaTensor,
      name: gammaTensor.label,
      shape: [columns],
      offset: 0,
      dtype: "f16",
      layout: "row-major",
    };
    const mapping = keep(tensors, createF16Tensor(device, columns * 2, "profile-rmsnorm-mapping"));
    const output = keep(tensors, createF16Tensor(device, ROWS * columns, "profile-rmsnorm-output"));
    const profiles: RmsNormProfile[] = [];
    for (const hasMapping of [false, true] as const) {
      for (const owner of ["row1", "auto"] as const) {
        const name = `rmsnorm-384-${hasMapping ? "mapped" : "plain"}-${owner}`;
        const timing = await timeComputePasses(device, ops, name, (pass) => {
          ops.rmsNorm(pass, input, gamma, output, ROWS, columns, hasMapping ? mapping : undefined, owner);
        });
        // Logical traffic counts the input's reduction and output passes separately.
        // Gamma and mapping are deliberately counted per element despite cache reuse.
        const logicalBytes = ROWS * columns * (hasMapping ? 12 : 8);
        profiles.push({
          name,
          owner,
          rows: ROWS,
          columns,
          hasMapping,
          workgroups: Math.ceil(ROWS / (owner === "auto" ? 8 : 1)),
          logicalGb: logicalBytes / 1e9,
          medianGbps: logicalBytes / timing.medianMs / 1e6,
          ...timing,
        });
      }
    }
    return profiles;
  } finally {
    destroyTensors(tensors);
  }
}

async function profileConv1x1(device: GPUDevice, ops: GpuOps): Promise<ConvProfile[]> {
  const tensors: GpuTensor[] = [];
  const rows = 1_025 * 1_189;
  try {
    const input = keep(tensors, createF16Tensor(device, rows * 128, "profile-conv1x1-input"));
    const weightTensor = keep(tensors, createF16Tensor(device, 128 * 128, "profile-conv1x1-weight"));
    const weight: GpuWeightTensor = {
      ...weightTensor,
      name: weightTensor.label,
      shape: [128, 128, 1, 1],
      offset: 0,
      dtype: "f16",
      layout: "conv-oihw",
    };
    const output = keep(tensors, createF16Tensor(device, rows * 128, "profile-conv1x1-output"));
    const logicalFlop = 2 * rows * 128 * 128;
    const profiles: ConvProfile[] = [];
    for (const owner of ["generic", "auto"] as const) {
      const name = `conv1x1-128-${owner}`;
      const timing = await timeComputePasses(device, ops, name, (pass) => {
        ops.conv2d(pass, input, weight, undefined, output, 1_025, 1_189, 128, 128, 1, owner);
      });
      profiles.push({
        name,
        owner,
        height: 1_025,
        width: 1_189,
        inChannels: 128,
        outChannels: 128,
        kernel: 1,
        rows,
        logicalGflop: logicalFlop / 1e9,
        medianTflops: logicalFlop / timing.medianMs / 1e9,
        ...timing,
      });
    }
    return profiles;
  } finally {
    destroyTensors(tensors);
  }
}

async function profileConv3x3Entry(device: GPUDevice, ops: GpuOps): Promise<ConvProfile[]> {
  const tensors: GpuTensor[] = [];
  const height = 1_025;
  const width = 1_189;
  const rows = height * width;
  try {
    const input = keep(tensors, createF16Tensor(device, rows * 4, "profile-conv3x3-entry-input"));
    const weightTensor = keep(tensors, createF16Tensor(device, 128 * 4 * 3 * 3, "profile-conv3x3-entry-weight"));
    const weight: GpuWeightTensor = {
      ...weightTensor,
      name: weightTensor.label,
      shape: [128, 4, 3, 3],
      offset: 0,
      dtype: "f16",
      layout: "conv-oihw",
    };
    const output = keep(tensors, createF16Tensor(device, rows * 128, "profile-conv3x3-entry-output"));
    const validSpatialTaps = (3 * height - 2) * (3 * width - 2);
    const logicalFlop = 2 * validSpatialTaps * 4 * 128;
    const profiles: ConvProfile[] = [];
    for (const owner of ["generic", "auto"] as const) {
      const name = `conv3x3-4x128-${owner}`;
      const timing = await timeComputePasses(device, ops, name, (pass) => {
        ops.conv2d(pass, input, weight, undefined, output, height, width, 4, 128, 3, owner);
      });
      profiles.push({
        name,
        owner,
        height,
        width,
        inChannels: 4,
        outChannels: 128,
        kernel: 3,
        rows,
        logicalGflop: logicalFlop / 1e9,
        medianTflops: logicalFlop / timing.medianMs / 1e9,
        ...timing,
      });
    }
    return profiles;
  } finally {
    destroyTensors(tensors);
  }
}

async function profileConv3x3Exit(device: GPUDevice, ops: GpuOps): Promise<ConvProfile[]> {
  const tensors: GpuTensor[] = [];
  const height = 1_025;
  const width = 1_189;
  const rows = height * width;
  try {
    const input = keep(tensors, createF16Tensor(device, rows * 128, "profile-conv3x3-exit-input"));
    const weightTensor = keep(tensors, createF16Tensor(device, 4 * 128 * 3 * 3, "profile-conv3x3-exit-weight"));
    const weight: GpuWeightTensor = {
      ...weightTensor,
      name: weightTensor.label,
      shape: [4, 128, 3, 3],
      offset: 0,
      dtype: "f16",
      layout: "conv-oihw",
    };
    const output = keep(tensors, createF16Tensor(device, rows * 4, "profile-conv3x3-exit-output"));
    const validSpatialTaps = (3 * height - 2) * (3 * width - 2);
    const logicalFlop = 2 * validSpatialTaps * 128 * 4;
    const name = "conv3x3-128x4-generic";
    const timing = await timeComputePasses(device, ops, name, (pass) => {
      ops.conv2d(pass, input, weight, undefined, output, height, width, 128, 4, 3, "generic");
    });
    return [{
      name,
      owner: "generic",
      height,
      width,
      inChannels: 128,
      outChannels: 4,
      kernel: 3,
      rows,
      logicalGflop: logicalFlop / 1e9,
      medianTflops: logicalFlop / timing.medianMs / 1e9,
      ...timing,
    }];
  } finally {
    destroyTensors(tensors);
  }
}

async function profileDenseModes(
  device: GPUDevice,
  ops: GpuOps,
  profileCase: DenseCase,
): Promise<DenseProfile[]> {
  const tensors: GpuTensor[] = [];
  try {
    const input = keep(tensors, createF16Tensor(
      device,
      ROWS * profileCase.inner,
      `${profileCase.name}-input`,
    ));
    const weightTensor = keep(tensors, createF16Tensor(
      device,
      profileCase.inner * profileCase.columns,
      `${profileCase.name}-weight`,
    ));
    const weight: GpuWeightTensor = {
      ...weightTensor,
      name: weightTensor.label,
      shape: [profileCase.inner, profileCase.columns],
      offset: 0,
      dtype: "f16",
      layout: profileCase.tileColumns === 256
        ? "linear-tile-n256-k32"
        : "linear-tile-n128-k32",
    };
    const output = keep(tensors, createF16Tensor(
      device,
      ROWS * profileCase.columns,
      `${profileCase.name}-output`,
    ));
    const residual = profileCase.residual === true
      ? keep(tensors, createF16Tensor(device, ROWS * profileCase.columns, `${profileCase.name}-residual`))
      : undefined;
    const encode = (
      pass: GPUComputePassEncoder,
      accumulation: PackedAccumulation,
    ): void => {
      ops.linear(pass, input, weight, undefined, output, {
        rows: ROWS,
        inner: profileCase.inner,
        columns: profileCase.columns,
        outputTileColumns: 128,
        vectorizeK: true,
        accumulation,
        ...(profileCase.activation === undefined ? {} : { activation: profileCase.activation }),
        ...(residual === undefined ? {} : { residual }),
        ...(profileCase.rotaryKeys === undefined ? {} : { rotaryKeys: profileCase.rotaryKeys }),
      });
    };
    const timings = await timeDenseModes(device, ops, profileCase.name, encode);
    const logicalFlop = 2 * ROWS * profileCase.inner * profileCase.columns;
    return DENSE_ACCUMULATIONS.map((accumulation) => {
      const timing = timings[accumulation];
      return {
        name: `${profileCase.name}-${accumulation}`,
        accumulation,
        rows: ROWS,
        inner: profileCase.inner,
        columns: profileCase.columns,
        productionMultiplicity: profileCase.productionMultiplicity,
        projectedGpuMs: timing.medianMs * profileCase.productionMultiplicity,
        logicalGflop: logicalFlop / 1e9,
        medianTflops: logicalFlop / timing.medianMs / 1e9,
        ...timing,
      };
    });
  } finally {
    destroyTensors(tensors);
  }
}

function projectedDenseGpuMs(
  profiles: readonly DenseProfile[],
  accumulation: PackedAccumulation,
): number {
  return profiles
    .filter((profile) => profile.accumulation === accumulation)
    .reduce((sum, profile) => sum + profile.projectedGpuMs, 0);
}

async function profileAttention(
  device: GPUDevice,
  ops: GpuOps,
  profileCase: AttentionGeometry & {
    readonly name: string;
    readonly kernel: AttentionKernel;
    readonly strided: boolean;
  },
): Promise<AttentionProfile> {
  const tensors: GpuTensor[] = [];
  try {
    const rows = profileCase.sequences * profileCase.tokens;
    const qkv = keep(tensors, createF16Tensor(device, rows * 1_536, `${profileCase.name}-qkv`));
    const gates = keep(tensors, createF16Tensor(device, rows * 8, `${profileCase.name}-gates`));
    const output = keep(tensors, createF16Tensor(device, rows * 512, `${profileCase.name}-output`));
    const encode = (pass: GPUComputePassEncoder): void => {
      ops.attention(pass, qkv, output, {
        sequences: profileCase.sequences,
        tokens: profileCase.tokens,
        kernel: profileCase.kernel,
        strided: profileCase.strided,
        gates,
        rotatedKeys: true,
      });
    };
    const timing = await timeComputePasses(device, ops, profileCase.name, encode);
    // QK and AV are each a multiply-add over all 8 heads × 64 dimensions.
    const logicalFlop = 4 * profileCase.sequences * 8 * profileCase.tokens ** 2 * 64;
    return {
      name: profileCase.name,
      kernel: profileCase.kernel,
      sequences: profileCase.sequences,
      tokens: profileCase.tokens,
      strided: profileCase.strided,
      logicalGflop: logicalFlop / 1e9,
      medianTflops: logicalFlop / timing.medianMs / 1e9,
      ...timing,
    };
  } finally {
    destroyTensors(tensors);
  }
}

async function timeDenseModes(
  device: GPUDevice,
  ops: GpuOps,
  label: string,
  encode: (pass: GPUComputePassEncoder, accumulation: PackedAccumulation) => void,
): Promise<Readonly<Record<PackedAccumulation, TimedSampleSummary>>> {
  ops.beginGraph();
  const warmup = device.createCommandEncoder({ label: `${label}-dense-modes-warmup` });
  for (let index = 0; index < WARMUP_PASSES; index += 1) {
    const order = index % 2 === 0
      ? DENSE_ACCUMULATIONS
      : [...DENSE_ACCUMULATIONS].reverse();
    for (const accumulation of order) {
      const pass = warmup.beginComputePass({ label: `${label}-${accumulation}-warmup-${index}` });
      encode(pass, accumulation);
      pass.end();
    }
  }
  device.queue.submit([warmup.finish()]);
  await device.queue.onSubmittedWorkDone();

  const querySet = device.createQuerySet({
    label: `${label}-dense-modes-timestamps`,
    type: "timestamp",
    count: 2,
  });
  const queryBytes = 2 * BigUint64Array.BYTES_PER_ELEMENT;
  const resolve = device.createBuffer({
    label: `${label}-dense-modes-timestamp-resolve`,
    size: queryBytes,
    usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
  });
  const readback = device.createBuffer({
    label: `${label}-dense-modes-timestamp-readback`,
    size: queryBytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const samples: Record<PackedAccumulation, number[]> = {
    exact: [],
    k2: [],
    k4: [],
  };
  let mapped = false;
  try {
    for (const [round, order] of DENSE_MEASURED_ORDERS.entries()) {
      for (const accumulation of order) {
        ops.beginGraph();
        const encoder = device.createCommandEncoder({
          label: `${label}-${accumulation}-measured-${round}`,
        });
        const pass = encoder.beginComputePass({
          label: `${label}-${accumulation}-measured-${round}`,
          timestampWrites: {
            querySet,
            beginningOfPassWriteIndex: 0,
            endOfPassWriteIndex: 1,
          },
        });
        encode(pass, accumulation);
        pass.end();
        encoder.resolveQuerySet(querySet, 0, 2, resolve, 0);
        encoder.copyBufferToBuffer(resolve, 0, readback, 0, queryBytes);
        device.queue.submit([encoder.finish()]);
        await readback.mapAsync(GPUMapMode.READ);
        mapped = true;
        const timestamps = new BigUint64Array(readback.getMappedRange());
        const beginning = timestamps[0]!;
        const end = timestamps[1]!;
        if (beginning === 0n || end <= beginning) {
          throw new Error(
            `${label}/${accumulation} returned invalid GPU timestamps ${beginning}..${end}`,
          );
        }
        samples[accumulation].push(Number(end - beginning) / 1e6);
        readback.unmap();
        mapped = false;
      }
    }
    return {
      exact: summarizeSamples(samples.exact),
      k2: summarizeSamples(samples.k2),
      k4: summarizeSamples(samples.k4),
    };
  } finally {
    if (mapped) readback.unmap();
    readback.destroy();
    resolve.destroy();
    querySet.destroy();
  }
}

async function timeComputePasses(
  device: GPUDevice,
  ops: GpuOps,
  label: string,
  encode: (pass: GPUComputePassEncoder) => void,
): Promise<TimedSampleSummary> {
  ops.beginGraph();
  const warmup = device.createCommandEncoder({ label: `${label}-warmup` });
  for (let index = 0; index < WARMUP_PASSES; index += 1) {
    const pass = warmup.beginComputePass({ label: `${label}-warmup-${index}` });
    encode(pass);
    pass.end();
  }
  device.queue.submit([warmup.finish()]);
  await device.queue.onSubmittedWorkDone();

  const querySet = device.createQuerySet({
    label: `${label}-timestamps`,
    type: "timestamp",
    count: 2,
  });
  const queryBytes = 2 * BigUint64Array.BYTES_PER_ELEMENT;
  const resolve = device.createBuffer({
    label: `${label}-timestamp-resolve`,
    size: queryBytes,
    usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
  });
  const readback = device.createBuffer({
    label: `${label}-timestamp-readback`,
    size: queryBytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  let mapped = false;
  try {
    const samplesMs: number[] = [];
    for (let index = 0; index < MEASURED_PASSES; index += 1) {
      ops.beginGraph();
      const encoder = device.createCommandEncoder({ label: `${label}-measured-${index}` });
      const pass = encoder.beginComputePass({
        label: `${label}-measured-${index}`,
        timestampWrites: {
          querySet,
          beginningOfPassWriteIndex: 0,
          endOfPassWriteIndex: 1,
        },
      });
      encode(pass);
      pass.end();
      encoder.resolveQuerySet(querySet, 0, 2, resolve, 0);
      encoder.copyBufferToBuffer(resolve, 0, readback, 0, queryBytes);
      device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      mapped = true;
      const timestamps = new BigUint64Array(readback.getMappedRange());
      const beginning = timestamps[0]!;
      const end = timestamps[1]!;
      if (beginning === 0n || end <= beginning) {
        throw new Error(`${label} returned invalid GPU timestamps ${beginning}..${end}`);
      }
      samplesMs.push(Number(end - beginning) / 1e6);
      readback.unmap();
      mapped = false;
    }
    const sorted = [...samplesMs].sort((left, right) => left - right);
    return {
      samplesMs,
      medianMs: sorted[Math.floor(sorted.length / 2)]!,
      minMs: sorted[0]!,
      maxMs: sorted.at(-1)!,
    };
  } finally {
    if (mapped) readback.unmap();
    readback.destroy();
    resolve.destroy();
    querySet.destroy();
  }
}

function summarizeSamples(samplesMs: readonly number[]): TimedSampleSummary {
  if (samplesMs.length === 0) throw new Error("Cannot summarize an empty timestamp sample");
  const sorted = [...samplesMs].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  const medianMs = sorted.length % 2 === 0
    ? (sorted[middle - 1]! + sorted[middle]!) / 2
    : sorted[middle]!;
  return {
    samplesMs,
    medianMs,
    minMs: sorted[0]!,
    maxMs: sorted.at(-1)!,
  };
}

function keep<T extends GpuTensor>(tensors: GpuTensor[], tensor: T): T {
  tensors.push(tensor);
  return tensor;
}

function setStatus(value: string): void {
  if (statusNode !== null) statusNode.textContent = value;
}

function publish(report: KernelProfileReport): void {
  const source = JSON.stringify(report, null, 2);
  const harnessGlobal = globalThis as unknown as {
    __DICOSE_BROWSER__: { report: KernelProfileReport };
    __DICOSE_BROWSER_REPORT__: KernelProfileReport;
  };
  harnessGlobal.__DICOSE_BROWSER__ = { report };
  harnessGlobal.__DICOSE_BROWSER_REPORT__ = report;
  if (resultNode !== null) resultNode.textContent = source;
  setStatus(report.ok ? "DiCoSe kernel profile complete." : "DiCoSe kernel profile failed.");
}
