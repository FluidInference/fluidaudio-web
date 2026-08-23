import type { AceModelProfileId } from "../../src/webgpu/capabilities.js";
import {
  ACE_TURBO_DIT_CONFIG,
  AceCorrectnessDitRuntime,
  createAceDitRopeTables,
  type AceDitConfig,
  type AceDitLayerScratch,
} from "../../src/webgpu/ace-dit.js";
import {
  AceCorrectnessDitPlumbingKernel,
  aceCorrectnessDitConcatenateWgsl,
  aceCorrectnessDitLinearUpdateWgsl,
  aceCorrectnessDitModulationWgsl,
  aceCorrectnessDitPatchProjectionWgsl,
  aceCorrectnessDitTimestepEmbeddingWgsl,
  aceCorrectnessDitUnpatchProjectionWgsl,
} from "../../src/webgpu/kernels/dit-plumbing.js";
import {
  ACE_TORCH_210_TIMESTEP_VECTORS,
  ACE_TORCH_TIMESTEP_SELECTED_DIMENSIONS,
} from "../dit-timestep-torch-vectors.js";

interface CaseResult {
  readonly profile: AceModelProfileId;
  readonly operation: string;
  readonly valuesChecked: number;
  readonly maximumAbsoluteError: number;
}

const resultNode = requireResultNode();

void run().then(
  (results) => finish("passed", JSON.stringify(results)),
  (error: unknown) => finish(
    "failed",
    error instanceof Error ? error.stack ?? error.message : String(error),
  ),
);

async function run(): Promise<readonly CaseResult[]> {
  if (navigator.gpu === undefined) throw new Error("WebGPU unavailable");
  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: "high-performance",
  });
  if (adapter === null) throw new Error("No WebGPU adapter");
  const requiredFeatures: GPUFeatureName[] = adapter.features.has("shader-f16")
    ? ["shader-f16"]
    : [];
  const device = await adapter.requestDevice({ requiredFeatures });
  try {
    const results = await runProfile(device, "reference-bf16");
    if (device.features.has("shader-f16")) {
      results.push(...await runProfile(device, "raw-fp16"));
    }
    return results;
  } finally {
    device.destroy();
  }
}

async function runProfile(
  device: GPUDevice,
  profile: AceModelProfileId,
): Promise<CaseResult[]> {
  await requireShadersValid(device, profile);
  const kernel = AceCorrectnessDitPlumbingKernel.create(device, profile);
  try {
    return [
      ...await runPatchRoundTrip(device, profile, kernel),
      ...await runModulation(device, profile, kernel),
      await runTimestep(device, profile, kernel),
      await runLinearUpdate(device, profile, kernel),
      ...await runZeroBranchLayers(device, profile),
      ...await runNontrivialLayers(device, profile),
    ];
  } finally {
    kernel.destroy();
  }
}

/**
 * Compile and execute the complete, unfused layer composition with zeroed
 * learned branches. The mathematically exact result is the residual input;
 * this catches binding geometry and graph-order errors across all 32 Stage 1
 * primitives without turning the browser result into a numerical oracle.
 */
async function runZeroBranchLayers(
  device: GPUDevice,
  profile: AceModelProfileId,
): Promise<CaseResult[]> {
  const config: AceDitConfig = Object.freeze({
    ...ACE_TURBO_DIT_CONFIG,
    id: "browser-zero-branch-dit-layer",
    hiddenSize: 128,
    intermediateSize: 256,
    layerCount: 1,
    queryHeads: 1,
    keyValueHeads: 1,
    headDimension: 128,
    maximumPositionEmbeddings: 8,
    slidingRadius: 1,
  });
  const tokens = 2;
  const conditionTokens = 2;
  const hiddenElements = tokens * config.hiddenSize;
  const intermediateElements = tokens * config.intermediateSize;
  const headElements = tokens * config.queryHeads * config.headDimension;
  const keyValueElements = tokens * config.keyValueHeads * config.headDimension;
  const modulationElements = 6 * config.hiddenSize;
  const denseWeightElements = config.hiddenSize * config.intermediateSize;
  const inputValues = Array.from(
    { length: hiddenElements },
    (_, index) => ((index % 11) - 5) / 8,
  );
  const owned: GPUBuffer[] = [];
  const own = (buffer: GPUBuffer): GPUBuffer => {
    owned.push(buffer);
    return buffer;
  };
  const activationScratch = (elements: number): GPUBufferBinding =>
    binding(own(activationOutput(device, profile, elements)));
  const input = own(activationBuffer(device, profile, inputValues));
  const slidingOutput = own(activationOutput(device, profile, hiddenElements));
  const globalOutput = own(activationOutput(device, profile, hiddenElements));
  const denseZero = own(weightBuffer(
    device,
    profile,
    new Array<number>(denseWeightElements).fill(0),
  ));
  const scaleShiftZero = own(weightBuffer(
    device,
    profile,
    new Array<number>(modulationElements).fill(0),
  ));
  const normOne = own(weightBuffer(
    device,
    profile,
    new Array<number>(config.hiddenSize).fill(1),
  ));
  const timestepProjection = own(activationBuffer(
    device,
    profile,
    new Array<number>(modulationElements).fill(0),
  ));
  const crossKey = own(activationBuffer(
    device,
    profile,
    new Array<number>(keyValueElements).fill(0),
  ));
  const crossValue = own(activationBuffer(
    device,
    profile,
    new Array<number>(keyValueElements).fill(0),
  ));
  const selfValidLengths = own(storageBuffer(
    device,
    new Uint32Array([tokens, tokens]),
  ));
  const crossValidLengths = own(storageBuffer(
    device,
    new Uint32Array([tokens, conditionTokens]),
  ));
  const rope = createAceDitRopeTables(tokens, config);
  const cosine = own(storageBuffer(device, rope.cosine));
  const sine = own(storageBuffer(device, rope.sine));
  const scratch: AceDitLayerScratch = {
    modulation: activationScratch(modulationElements),
    selfNormalized: activationScratch(hiddenElements),
    selfModulated: activationScratch(hiddenElements),
    selfQueryFlat: activationScratch(headElements),
    selfKeyFlat: activationScratch(keyValueElements),
    selfValueFlat: activationScratch(keyValueElements),
    selfQueryHeads: activationScratch(headElements),
    selfKeyHeads: activationScratch(keyValueElements),
    selfValueHeads: activationScratch(keyValueElements),
    selfNormalizedQueryHeads: activationScratch(headElements),
    selfNormalizedKeyHeads: activationScratch(keyValueElements),
    selfRotatedQueryHeads: activationScratch(headElements),
    selfRotatedKeyHeads: activationScratch(keyValueElements),
    selfAttentionHeads: activationScratch(headElements),
    selfMergedAttention: activationScratch(headElements),
    selfProjectedAttention: activationScratch(hiddenElements),
    afterSelfAttention: activationScratch(hiddenElements),
    crossNormalized: activationScratch(hiddenElements),
    crossQueryFlat: activationScratch(headElements),
    crossQueryHeads: activationScratch(headElements),
    crossNormalizedQueryHeads: activationScratch(headElements),
    crossAttentionHeads: activationScratch(headElements),
    crossMergedAttention: activationScratch(headElements),
    crossProjectedAttention: activationScratch(hiddenElements),
    afterCrossAttention: activationScratch(hiddenElements),
    mlpNormalized: activationScratch(hiddenElements),
    mlpModulated: activationScratch(hiddenElements),
    gate: activationScratch(intermediateElements),
    up: activationScratch(intermediateElements),
    gatedActivation: activationScratch(intermediateElements),
    projectedMlp: activationScratch(hiddenElements),
  };
  const runtime = AceCorrectnessDitRuntime.create(device, profile, {
    backend: "portable",
    weightLayout: "source-row-major",
  });
  try {
    const zeroWeight = binding(denseZero);
    const oneWeight = binding(normOne);
    const commonBindings = {
      input: binding(input),
      weights: {
        scaleShiftTable: binding(scaleShiftZero),
        selfAttentionNorm: oneWeight,
        selfQueryProjection: zeroWeight,
        selfKeyProjection: zeroWeight,
        selfValueProjection: zeroWeight,
        selfQueryNorm: oneWeight,
        selfKeyNorm: oneWeight,
        selfOutputProjection: zeroWeight,
        crossAttentionNorm: oneWeight,
        crossQueryProjection: zeroWeight,
        crossQueryNorm: oneWeight,
        crossOutputProjection: zeroWeight,
        mlpNorm: oneWeight,
        gateProjection: zeroWeight,
        upProjection: zeroWeight,
        downProjection: zeroWeight,
      },
      scratch,
      timestepProjection: binding(timestepProjection),
      crossKey: binding(crossKey),
      crossValue: binding(crossValue),
      selfValidLengths: binding(selfValidLengths),
      crossValidLengths: binding(crossValidLengths),
      cosine: binding(cosine),
      sine: binding(sine),
    } as const;
    const sliding = await runtime.createLayerDispatch(
      `browser-${profile}-dit-zero-branch-sliding-layer`,
      config,
      { batch: 1, tokens, conditionTokens, attentionMode: "sliding" },
      {
        ...commonBindings,
        output: binding(slidingOutput),
      },
    );
    const global = await runtime.createLayerDispatch(
      `browser-${profile}-dit-zero-branch-global-layer`,
      config,
      { batch: 1, tokens, conditionTokens, attentionMode: "full" },
      {
        ...commonBindings,
        output: binding(globalOutput),
      },
    );
    if (sliding.primitiveCount !== 32 || global.primitiveCount !== 32) {
      throw new Error(
        `complete DiT layers compiled ${sliding.primitiveCount}/${global.primitiveCount} primitives, expected 32/32`,
      );
    }
    const [actualSliding, actualGlobal] = await executeAndReadActivations(
      device,
      [sliding, global],
      [slidingOutput, globalOutput],
      [hiddenElements, hiddenElements],
      profile,
    );
    const expected = Float32Array.from(
      inputValues,
      (value) => activationValue(profile, value),
    );
    return [
      compare(
        profile,
        "complete-zero-branch-sliding-dit-layer",
        actualSliding!,
        expected,
      ),
      compare(
        profile,
        "complete-zero-branch-global-dit-layer",
        actualGlobal!,
        expected,
      ),
    ];
  } finally {
    runtime.destroy();
    for (const buffer of owned) buffer.destroy();
  }
}

/**
 * A nontrivial CPU-oracled layer: three self tokens make radius-one sliding
 * attention differ from global attention, and every learned branch is active.
 */
async function runNontrivialLayers(
  device: GPUDevice,
  profile: AceModelProfileId,
): Promise<CaseResult[]> {
  const config: AceDitConfig = Object.freeze({
    ...ACE_TURBO_DIT_CONFIG,
    id: "browser-nontrivial-dit-layer",
    hiddenSize: 128,
    intermediateSize: 8,
    layerCount: 1,
    queryHeads: 1,
    keyValueHeads: 1,
    headDimension: 128,
    maximumPositionEmbeddings: 8,
    slidingRadius: 1,
  });
  const tokens = 3;
  const conditionTokens = 2;
  const hiddenElements = tokens * config.hiddenSize;
  const intermediateElements = tokens * config.intermediateSize;
  const headElements = hiddenElements;
  const keyValueElements = hiddenElements;
  const modulationElements = 6 * config.hiddenSize;
  const inputValues = Array.from(
    { length: hiddenElements },
    (_, index) => {
      const token = Math.floor(index / config.hiddenSize);
      const dimension = index % config.hiddenSize;
      return ((dimension % 13) - 6) / 16 + token / 32;
    },
  );
  const scaleShiftTable = Array.from(
    { length: modulationElements },
    (_, index) => [-1 / 64, 1 / 32, 1 / 8, 1 / 128, -1 / 64, 1 / 8][
      Math.floor(index / config.hiddenSize)
    ]!,
  );
  const timestepProjectionValues = Array.from(
    { length: modulationElements },
    (_, index) => [1 / 128, 1 / 32, 1 / 8, -1 / 256, 1 / 32, 1 / 8][
      Math.floor(index / config.hiddenSize)
    ]!,
  );
  const selfNorm = Array.from(
    { length: config.hiddenSize },
    (_, index) => index % 2 === 0 ? 3 / 4 : 7 / 8,
  );
  const crossNorm = new Array<number>(config.hiddenSize).fill(13 / 16);
  const mlpNorm = new Array<number>(config.hiddenSize).fill(11 / 16);
  const queryNorm = new Array<number>(config.headDimension).fill(7 / 8);
  const keyNorm = new Array<number>(config.headDimension).fill(3 / 4);
  const crossQueryNorm = new Array<number>(config.headDimension).fill(5 / 8);
  const weights: CpuAceLayerWeights = {
    scaleShiftTable,
    selfAttentionNorm: selfNorm,
    selfQueryProjection: selectedRowWeight(128, 128, 1 / 2, (row) => row),
    selfKeyProjection: selectedRowWeight(128, 128, 3 / 8, (row) => row),
    selfValueProjection: selectedRowWeight(128, 128, -1 / 4, (row) => row),
    selfQueryNorm: queryNorm,
    selfKeyNorm: keyNorm,
    selfOutputProjection: selectedRowWeight(128, 128, 1 / 4, (row) => row),
    crossAttentionNorm: crossNorm,
    crossQueryProjection: selectedRowWeight(128, 128, -1 / 2, (row) => row),
    crossQueryNorm,
    crossOutputProjection: selectedRowWeight(128, 128, 1 / 8, (row) => row),
    mlpNorm,
    gateProjection: selectedRowWeight(8, 128, 1 / 2, (row) => row * 13),
    upProjection: selectedRowWeight(8, 128, -3 / 8, (row) => row * 13 + 5),
    downProjection: selectedRowWeight(128, 8, 1 / 4, (row) => row % 8),
  };
  const crossKeyValues = Array.from(
    { length: conditionTokens * config.headDimension },
    (_, index) => {
      const token = Math.floor(index / config.headDimension);
      const dimension = index % config.headDimension;
      return ((dimension % 9) - 4) / 32 + token / 64;
    },
  );
  const crossValueValues = Array.from(
    { length: conditionTokens * config.headDimension },
    (_, index) => {
      const token = Math.floor(index / config.headDimension);
      const dimension = index % config.headDimension;
      return ((dimension % 7) - 3) / 16 - token / 32;
    },
  );

  const owned: GPUBuffer[] = [];
  const own = (buffer: GPUBuffer): GPUBuffer => {
    owned.push(buffer);
    return buffer;
  };
  const activationScratch = (elements: number): GPUBufferBinding =>
    binding(own(activationOutput(device, profile, elements)));
  const scratch: AceDitLayerScratch = {
    modulation: activationScratch(modulationElements),
    selfNormalized: activationScratch(hiddenElements),
    selfModulated: activationScratch(hiddenElements),
    selfQueryFlat: activationScratch(headElements),
    selfKeyFlat: activationScratch(keyValueElements),
    selfValueFlat: activationScratch(keyValueElements),
    selfQueryHeads: activationScratch(headElements),
    selfKeyHeads: activationScratch(keyValueElements),
    selfValueHeads: activationScratch(keyValueElements),
    selfNormalizedQueryHeads: activationScratch(headElements),
    selfNormalizedKeyHeads: activationScratch(keyValueElements),
    selfRotatedQueryHeads: activationScratch(headElements),
    selfRotatedKeyHeads: activationScratch(keyValueElements),
    selfAttentionHeads: activationScratch(headElements),
    selfMergedAttention: activationScratch(headElements),
    selfProjectedAttention: activationScratch(hiddenElements),
    afterSelfAttention: activationScratch(hiddenElements),
    crossNormalized: activationScratch(hiddenElements),
    crossQueryFlat: activationScratch(headElements),
    crossQueryHeads: activationScratch(headElements),
    crossNormalizedQueryHeads: activationScratch(headElements),
    crossAttentionHeads: activationScratch(headElements),
    crossMergedAttention: activationScratch(headElements),
    crossProjectedAttention: activationScratch(hiddenElements),
    afterCrossAttention: activationScratch(hiddenElements),
    mlpNormalized: activationScratch(hiddenElements),
    mlpModulated: activationScratch(hiddenElements),
    gate: activationScratch(intermediateElements),
    up: activationScratch(intermediateElements),
    gatedActivation: activationScratch(intermediateElements),
    projectedMlp: activationScratch(hiddenElements),
  };
  const input = own(activationBuffer(device, profile, inputValues));
  const slidingOutput = own(activationOutput(device, profile, hiddenElements));
  const globalOutput = own(activationOutput(device, profile, hiddenElements));
  const timestepProjection = own(activationBuffer(
    device,
    profile,
    timestepProjectionValues,
  ));
  const crossKey = own(activationBuffer(device, profile, crossKeyValues));
  const crossValue = own(activationBuffer(device, profile, crossValueValues));
  const selfValidLengths = own(storageBuffer(
    device,
    new Uint32Array([tokens, tokens]),
  ));
  const crossValidLengths = own(storageBuffer(
    device,
    new Uint32Array([tokens, conditionTokens]),
  ));
  const rope = createAceDitRopeTables(tokens, config);
  const cosine = own(storageBuffer(device, rope.cosine));
  const sine = own(storageBuffer(device, rope.sine));
  const gpuWeights = {
    scaleShiftTable: binding(own(weightBuffer(device, profile, weights.scaleShiftTable))),
    selfAttentionNorm: binding(own(weightBuffer(device, profile, weights.selfAttentionNorm))),
    selfQueryProjection: binding(own(weightBuffer(device, profile, weights.selfQueryProjection))),
    selfKeyProjection: binding(own(weightBuffer(device, profile, weights.selfKeyProjection))),
    selfValueProjection: binding(own(weightBuffer(device, profile, weights.selfValueProjection))),
    selfQueryNorm: binding(own(weightBuffer(device, profile, weights.selfQueryNorm))),
    selfKeyNorm: binding(own(weightBuffer(device, profile, weights.selfKeyNorm))),
    selfOutputProjection: binding(own(weightBuffer(device, profile, weights.selfOutputProjection))),
    crossAttentionNorm: binding(own(weightBuffer(device, profile, weights.crossAttentionNorm))),
    crossQueryProjection: binding(own(weightBuffer(device, profile, weights.crossQueryProjection))),
    crossQueryNorm: binding(own(weightBuffer(device, profile, weights.crossQueryNorm))),
    crossOutputProjection: binding(own(weightBuffer(device, profile, weights.crossOutputProjection))),
    mlpNorm: binding(own(weightBuffer(device, profile, weights.mlpNorm))),
    gateProjection: binding(own(weightBuffer(device, profile, weights.gateProjection))),
    upProjection: binding(own(weightBuffer(device, profile, weights.upProjection))),
    downProjection: binding(own(weightBuffer(device, profile, weights.downProjection))),
  } as const;
  const runtime = AceCorrectnessDitRuntime.create(device, profile, {
    backend: "portable",
    weightLayout: "source-row-major",
  });
  try {
    const commonBindings = {
      input: binding(input),
      weights: gpuWeights,
      scratch,
      timestepProjection: binding(timestepProjection),
      crossKey: binding(crossKey),
      crossValue: binding(crossValue),
      selfValidLengths: binding(selfValidLengths),
      crossValidLengths: binding(crossValidLengths),
      cosine: binding(cosine),
      sine: binding(sine),
    } as const;
    const sliding = await runtime.createLayerDispatch(
      `browser-${profile}-dit-nontrivial-sliding-layer`,
      config,
      { batch: 1, tokens, conditionTokens, attentionMode: "sliding" },
      { ...commonBindings, output: binding(slidingOutput) },
    );
    const global = await runtime.createLayerDispatch(
      `browser-${profile}-dit-nontrivial-global-layer`,
      config,
      { batch: 1, tokens, conditionTokens, attentionMode: "full" },
      { ...commonBindings, output: binding(globalOutput) },
    );
    const [actualSliding, actualGlobal] = await executeAndReadActivations(
      device,
      [sliding, global],
      [slidingOutput, globalOutput],
      [hiddenElements, hiddenElements],
      profile,
    );
    const expectedSliding = cpuAceLayer(
      profile,
      config,
      tokens,
      conditionTokens,
      "sliding",
      inputValues,
      timestepProjectionValues,
      crossKeyValues,
      crossValueValues,
      rope.cosine,
      rope.sine,
      weights,
    );
    const expectedGlobal = cpuAceLayer(
      profile,
      config,
      tokens,
      conditionTokens,
      "full",
      inputValues,
      timestepProjectionValues,
      crossKeyValues,
      crossValueValues,
      rope.cosine,
      rope.sine,
      weights,
    );
    const modeDifference = maximumDifference(expectedSliding, expectedGlobal);
    if (modeDifference <= 1e-4) {
      throw new Error(`CPU sliding/global paths did not diverge (${modeDifference})`);
    }
    return [
      compare(
        profile,
        "cpu-oracled-nontrivial-sliding-dit-layer",
        actualSliding!,
        expectedSliding,
      ),
      compare(
        profile,
        "cpu-oracled-nontrivial-global-dit-layer",
        actualGlobal!,
        expectedGlobal,
      ),
    ];
  } finally {
    runtime.destroy();
    for (const buffer of owned) buffer.destroy();
  }
}

async function requireShadersValid(
  device: GPUDevice,
  profile: AceModelProfileId,
): Promise<void> {
  const sources = [
    ["concatenate", aceCorrectnessDitConcatenateWgsl(profile, {
      batch: 1, time: 3, leftWidth: 2, rightWidth: 1,
    })],
    ["patch", aceCorrectnessDitPatchProjectionWgsl(profile, {
      batch: 1, time: 3, inputChannels: 3, hiddenSize: 2, patchSize: 2,
    })],
    ["unpatch", aceCorrectnessDitUnpatchProjectionWgsl(profile, {
      batch: 1, time: 3, outputChannels: 2, hiddenSize: 2, patchSize: 2,
    })],
    ["modulation", aceCorrectnessDitModulationWgsl(profile, {
      batch: 1, groups: 2, width: 2, projectionLayout: "per-group",
    })],
    ["timestep", aceCorrectnessDitTimestepEmbeddingWgsl(profile, {
      batch: 8, dimension: 256, scale: 1_000, maximumPeriod: 10_000,
    })],
    ["linear-update", aceCorrectnessDitLinearUpdateWgsl(profile, {
      batch: 1, time: 2, channels: 2, coefficient: 0.5,
    })],
  ] as const;
  for (const [operation, code] of sources) {
    const module = device.createShaderModule({
      label: `browser-${profile}-dit-${operation}-preflight`,
      code,
    });
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter((message) => message.type === "error");
    if (errors.length > 0) {
      throw new Error(`${profile} ${operation}: ${errors.map((message) =>
        `${message.lineNum}:${message.linePos} ${message.message}`
      ).join("\n")}`);
    }
  }
}

async function runPatchRoundTrip(
  device: GPUDevice,
  profile: AceModelProfileId,
  kernel: AceCorrectnessDitPlumbingKernel,
): Promise<CaseResult[]> {
  const contextValues = [1, 2, 3, 4, 5, 6];
  const latentValues = [10, 20, 30];
  const patchWeights = [
    1, 0.5, -1, 0.25, 0.1, -0.2,
    0, 1, 0, 1, 0, 1,
  ];
  const patchBias = [0.5, -1];
  const unpatchWeights = [
    1, 2, 0.5, -1,
    -0.25, 0.5, 1, 0.25,
  ];
  const unpatchBias = [0.1, -0.2];
  const context = activationBuffer(device, profile, contextValues);
  const latent = activationBuffer(device, profile, latentValues);
  const concatenated = activationOutput(device, profile, 9);
  const patchWeight = weightBuffer(device, profile, patchWeights);
  const patchBiasBuffer = weightBuffer(device, profile, patchBias);
  const patched = activationOutput(device, profile, 4);
  const unpatchWeight = weightBuffer(device, profile, unpatchWeights);
  const unpatchBiasBuffer = weightBuffer(device, profile, unpatchBias);
  const unpatched = activationOutput(device, profile, 6);
  try {
    const concatenate = await kernel.createConcatenateDispatch(
      `browser-${profile}-dit-concatenate`,
      { batch: 1, time: 3, leftWidth: 2, rightWidth: 1 },
      {
        left: binding(context),
        right: binding(latent),
        output: binding(concatenated),
      },
    );
    const patch = await kernel.createPatchProjectionDispatch(
      `browser-${profile}-dit-patch`,
      {
        batch: 1,
        time: 3,
        inputChannels: 3,
        hiddenSize: 2,
        patchSize: 2,
      },
      {
        input: binding(concatenated),
        weight: binding(patchWeight),
        bias: binding(patchBiasBuffer),
        output: binding(patched),
      },
    );
    const unpatch = await kernel.createUnpatchProjectionDispatch(
      `browser-${profile}-dit-unpatch`,
      {
        batch: 1,
        time: 3,
        outputChannels: 2,
        hiddenSize: 2,
        patchSize: 2,
      },
      {
        input: binding(patched),
        weight: binding(unpatchWeight),
        bias: binding(unpatchBiasBuffer),
        output: binding(unpatched),
      },
    );
    const [actualConcatenated, actualPatched, actualUnpatched] =
      await executeAndReadActivations(
        device,
        [concatenate, patch, unpatch],
        [concatenated, patched, unpatched],
        [9, 4, 6],
        profile,
      );
    const expectedConcatenated = Float32Array.from(
      [1, 2, 10, 3, 4, 20, 5, 6, 30],
      (value) => activationValue(profile, value),
    );
    const expectedPatched = cpuPatch(
      profile,
      expectedConcatenated,
      patchWeights,
      patchBias,
    );
    const expectedUnpatched = cpuUnpatch(
      profile,
      expectedPatched,
      unpatchWeights,
      unpatchBias,
    );
    return [
      compare(profile, "context-latent-concatenate", actualConcatenated!, expectedConcatenated),
      compare(profile, "source-layout-patch", actualPatched!, expectedPatched),
      compare(profile, "source-layout-unpatch-and-crop", actualUnpatched!, expectedUnpatched),
    ];
  } finally {
    for (const buffer of [
      context,
      latent,
      concatenated,
      patchWeight,
      patchBiasBuffer,
      patched,
      unpatchWeight,
      unpatchBiasBuffer,
      unpatched,
    ]) buffer.destroy();
  }
}

async function runModulation(
  device: GPUDevice,
  profile: AceModelProfileId,
  kernel: AceCorrectnessDitPlumbingKernel,
): Promise<CaseResult[]> {
  const tableValues = [0.5, -0.5, 1, -1];
  const table = weightBuffer(device, profile, tableValues);
  const perGroupProjection = activationBuffer(device, profile, [1, 2, 3, 4]);
  const perBatchProjection = activationBuffer(device, profile, [1, 2]);
  const perGroupOutput = activationOutput(device, profile, 4);
  const perBatchOutput = activationOutput(device, profile, 4);
  try {
    const perGroup = await kernel.createModulationDispatch(
      `browser-${profile}-dit-modulation-per-group`,
      { batch: 1, groups: 2, width: 2, projectionLayout: "per-group" },
      {
        projection: binding(perGroupProjection),
        table: binding(table),
        output: binding(perGroupOutput),
      },
    );
    const perBatch = await kernel.createModulationDispatch(
      `browser-${profile}-dit-modulation-per-batch`,
      { batch: 1, groups: 2, width: 2, projectionLayout: "per-batch" },
      {
        projection: binding(perBatchProjection),
        table: binding(table),
        output: binding(perBatchOutput),
      },
    );
    const [actualPerGroup, actualPerBatch] = await executeAndReadActivations(
      device,
      [perGroup, perBatch],
      [perGroupOutput, perBatchOutput],
      [4, 4],
      profile,
    );
    const tableRounded = tableValues.map((value) => weightValue(profile, value));
    const perGroupRounded = [1, 2, 3, 4].map((value) => activationValue(profile, value));
    const perBatchRounded = [1, 2].map((value) => activationValue(profile, value));
    const expectedPerGroup = Float32Array.from(perGroupRounded, (value, index) =>
      add(profile, value, tableRounded[index]!)
    );
    const expectedPerBatch = Float32Array.from({ length: 4 }, (_, index) =>
      add(profile, perBatchRounded[index % 2]!, tableRounded[index]!)
    );
    return [
      compare(profile, "layer-modulation-group-major", actualPerGroup!, expectedPerGroup),
      compare(profile, "output-modulation-broadcast", actualPerBatch!, expectedPerBatch),
    ];
  } finally {
    table.destroy();
    perGroupProjection.destroy();
    perBatchProjection.destroy();
    perGroupOutput.destroy();
    perBatchOutput.destroy();
  }
}

async function runTimestep(
  device: GPUDevice,
  profile: AceModelProfileId,
  kernel: AceCorrectnessDitPlumbingKernel,
): Promise<CaseResult> {
  const timestep = storageBuffer(
    device,
    Float32Array.from(ACE_TORCH_210_TIMESTEP_VECTORS, (vector) => vector.timestep),
  );
  const outputElements = ACE_TORCH_210_TIMESTEP_VECTORS.length * 256;
  const output = activationOutput(device, profile, outputElements);
  try {
    const dispatch = await kernel.createTimestepEmbeddingDispatch(
      `browser-${profile}-dit-timestep`,
      {
        batch: ACE_TORCH_210_TIMESTEP_VECTORS.length,
        dimension: 256,
        scale: 1_000,
        maximumPeriod: 10_000,
      },
      { timestep: binding(timestep), output: binding(output) },
    );
    const [actual] = await executeAndReadActivations(
      device,
      [dispatch],
      [output],
      [outputElements],
      profile,
    );
    const selectedActual = new Float32Array(
      ACE_TORCH_210_TIMESTEP_VECTORS.length *
        ACE_TORCH_TIMESTEP_SELECTED_DIMENSIONS.length,
    );
    const selectedExpected = new Float32Array(selectedActual.length);
    for (
      let timestepIndex = 0;
      timestepIndex < ACE_TORCH_210_TIMESTEP_VECTORS.length;
      timestepIndex += 1
    ) {
      const vector = ACE_TORCH_210_TIMESTEP_VECTORS[timestepIndex]!;
      for (
        let selectionIndex = 0;
        selectionIndex < ACE_TORCH_TIMESTEP_SELECTED_DIMENSIONS.length;
        selectionIndex += 1
      ) {
        const flat = timestepIndex * ACE_TORCH_TIMESTEP_SELECTED_DIMENSIONS.length +
          selectionIndex;
        selectedActual[flat] = actual![
          timestepIndex * 256 +
          ACE_TORCH_TIMESTEP_SELECTED_DIMENSIONS[selectionIndex]!
        ]!;
        selectedExpected[flat] = profile === "reference-bf16"
          ? f32BitsToNumber(vector.selectedFloat32Bits[selectionIndex]!)
          : fp16BitsToNumber(vector.selectedFloat16Bits[selectionIndex]!);
      }
    }
    return compare(
      profile,
      "torch-2.10-all-effective-timestep-sinusoids",
      selectedActual,
      selectedExpected,
    );
  } finally {
    timestep.destroy();
    output.destroy();
  }
}

async function runLinearUpdate(
  device: GPUDevice,
  profile: AceModelProfileId,
  kernel: AceCorrectnessDitPlumbingKernel,
): Promise<CaseResult> {
  const latentValues = [1, 2, -1, 4];
  const velocityValues = [0.5, -1, 2, 0.25];
  const latent = activationBuffer(device, profile, latentValues);
  const velocity = activationBuffer(device, profile, velocityValues);
  const output = activationOutput(device, profile, 4);
  try {
    const dispatch = await kernel.createLinearUpdateDispatch(
      `browser-${profile}-dit-linear-update`,
      { batch: 1, time: 2, channels: 2, coefficient: 0.5 },
      { latent: binding(latent), velocity: binding(velocity), output: binding(output) },
    );
    const [actual] = await executeAndReadActivations(
      device,
      [dispatch],
      [output],
      [4],
      profile,
    );
    const expected = Float32Array.from(latentValues, (value, index) =>
      subtract(profile, activationValue(profile, value), multiply(
        profile,
        activationValue(profile, velocityValues[index]!),
        activationValue(profile, 0.5),
      ))
    );
    return compare(profile, "euler-or-clean-linear-update", actual!, expected);
  } finally {
    latent.destroy();
    velocity.destroy();
    output.destroy();
  }
}

function cpuPatch(
  profile: AceModelProfileId,
  input: Float32Array,
  rawWeights: readonly number[],
  rawBias: readonly number[],
): Float32Array {
  const weights = rawWeights.map((value) => weightValue(profile, value));
  const bias = rawBias.map((value) => weightValue(profile, value));
  const output = new Float32Array(4);
  for (let token = 0; token < 2; token += 1) {
    for (let hidden = 0; hidden < 2; hidden += 1) {
      let sum = bias[hidden]!;
      for (let channel = 0; channel < 3; channel += 1) {
        for (let patch = 0; patch < 2; patch += 1) {
          const time = token * 2 + patch;
          if (time < 3) {
            sum = add(
              profile,
              sum,
              multiply(
                profile,
                input[time * 3 + channel]!,
                weights[(hidden * 3 + channel) * 2 + patch]!,
              ),
            );
          }
        }
      }
      output[token * 2 + hidden] = sum;
    }
  }
  return output;
}

function cpuUnpatch(
  profile: AceModelProfileId,
  input: Float32Array,
  rawWeights: readonly number[],
  rawBias: readonly number[],
): Float32Array {
  const weights = rawWeights.map((value) => weightValue(profile, value));
  const bias = rawBias.map((value) => weightValue(profile, value));
  const output = new Float32Array(6);
  for (let time = 0; time < 3; time += 1) {
    const token = Math.floor(time / 2);
    const patch = time % 2;
    for (let channel = 0; channel < 2; channel += 1) {
      let sum = bias[channel]!;
      for (let hidden = 0; hidden < 2; hidden += 1) {
        sum = add(
          profile,
          sum,
          multiply(
            profile,
            input[token * 2 + hidden]!,
            weights[(hidden * 2 + channel) * 2 + patch]!,
          ),
        );
      }
      output[time * 2 + channel] = sum;
    }
  }
  return output;
}

interface CpuAceLayerWeights {
  readonly scaleShiftTable: readonly number[];
  readonly selfAttentionNorm: readonly number[];
  readonly selfQueryProjection: readonly number[];
  readonly selfKeyProjection: readonly number[];
  readonly selfValueProjection: readonly number[];
  readonly selfQueryNorm: readonly number[];
  readonly selfKeyNorm: readonly number[];
  readonly selfOutputProjection: readonly number[];
  readonly crossAttentionNorm: readonly number[];
  readonly crossQueryProjection: readonly number[];
  readonly crossQueryNorm: readonly number[];
  readonly crossOutputProjection: readonly number[];
  readonly mlpNorm: readonly number[];
  readonly gateProjection: readonly number[];
  readonly upProjection: readonly number[];
  readonly downProjection: readonly number[];
}

function selectedRowWeight(
  rows: number,
  columns: number,
  value: number,
  selectedColumn: (row: number) => number,
): number[] {
  const output = new Array<number>(rows * columns).fill(0);
  for (let row = 0; row < rows; row += 1) {
    const column = selectedColumn(row);
    if (!Number.isSafeInteger(column) || column < 0 || column >= columns) {
      throw new RangeError(`selected weight column ${column} is outside ${columns}`);
    }
    output[row * columns + column] = value;
  }
  return output;
}

function cpuAceLayer(
  profile: AceModelProfileId,
  config: AceDitConfig,
  tokens: number,
  conditionTokens: number,
  mode: "sliding" | "full",
  rawInput: readonly number[],
  rawTimestepProjection: readonly number[],
  rawCrossKey: readonly number[],
  rawCrossValue: readonly number[],
  cosine: Float32Array,
  sine: Float32Array,
  weights: CpuAceLayerWeights,
): Float32Array {
  const hidden = config.hiddenSize;
  const intermediate = config.intermediateSize;
  const input = Float32Array.from(rawInput, (value) => activationValue(profile, value));
  const timestepProjection = Float32Array.from(
    rawTimestepProjection,
    (value) => activationValue(profile, value),
  );
  const table = Float32Array.from(
    weights.scaleShiftTable,
    (value) => weightValue(profile, value),
  );
  const modulation = Float32Array.from(table, (value, index) =>
    add(profile, value, timestepProjection[index]!));
  const group = (index: number): Float32Array =>
    modulation.slice(index * hidden, (index + 1) * hidden);

  const selfNormalized = cpuRmsNorm(
    profile,
    input,
    tokens,
    hidden,
    weights.selfAttentionNorm,
    config.rmsNormEpsilon,
  );
  const selfModulated = cpuAdaLn(
    profile,
    selfNormalized,
    tokens,
    hidden,
    group(1),
    group(0),
  );
  const selfQuery = cpuGemm(
    profile,
    selfModulated,
    tokens,
    hidden,
    hidden,
    weights.selfQueryProjection,
  );
  const selfKey = cpuGemm(
    profile,
    selfModulated,
    tokens,
    hidden,
    hidden,
    weights.selfKeyProjection,
  );
  const selfValue = cpuGemm(
    profile,
    selfModulated,
    tokens,
    hidden,
    hidden,
    weights.selfValueProjection,
  );
  const normalizedQuery = cpuRmsNorm(
    profile,
    selfQuery,
    tokens,
    hidden,
    weights.selfQueryNorm,
    config.rmsNormEpsilon,
  );
  const normalizedKey = cpuRmsNorm(
    profile,
    selfKey,
    tokens,
    hidden,
    weights.selfKeyNorm,
    config.rmsNormEpsilon,
  );
  const rotatedQuery = cpuRope(
    profile,
    normalizedQuery,
    tokens,
    hidden,
    cosine,
    sine,
  );
  const rotatedKey = cpuRope(
    profile,
    normalizedKey,
    tokens,
    hidden,
    cosine,
    sine,
  );
  const selfAttention = cpuOnlineAttention(
    profile,
    rotatedQuery,
    rotatedKey,
    selfValue,
    tokens,
    tokens,
    hidden,
    mode,
    config.slidingRadius,
  );
  const selfProjected = cpuGemm(
    profile,
    selfAttention,
    tokens,
    hidden,
    hidden,
    weights.selfOutputProjection,
  );
  const afterSelf = cpuGatedResidual(
    profile,
    input,
    selfProjected,
    tokens,
    hidden,
    group(2),
  );

  const crossNormalized = cpuRmsNorm(
    profile,
    afterSelf,
    tokens,
    hidden,
    weights.crossAttentionNorm,
    config.rmsNormEpsilon,
  );
  const crossQuery = cpuGemm(
    profile,
    crossNormalized,
    tokens,
    hidden,
    hidden,
    weights.crossQueryProjection,
  );
  const normalizedCrossQuery = cpuRmsNorm(
    profile,
    crossQuery,
    tokens,
    hidden,
    weights.crossQueryNorm,
    config.rmsNormEpsilon,
  );
  const crossKey = Float32Array.from(
    rawCrossKey,
    (value) => activationValue(profile, value),
  );
  const crossValue = Float32Array.from(
    rawCrossValue,
    (value) => activationValue(profile, value),
  );
  const crossAttention = cpuOnlineAttention(
    profile,
    normalizedCrossQuery,
    crossKey,
    crossValue,
    tokens,
    conditionTokens,
    hidden,
    "full",
    0,
  );
  const crossProjected = cpuGemm(
    profile,
    crossAttention,
    tokens,
    hidden,
    hidden,
    weights.crossOutputProjection,
  );
  const afterCross = Float32Array.from(afterSelf, (value, index) =>
    add(profile, value, crossProjected[index]!));

  const mlpNormalized = cpuRmsNorm(
    profile,
    afterCross,
    tokens,
    hidden,
    weights.mlpNorm,
    config.rmsNormEpsilon,
  );
  const mlpModulated = cpuAdaLn(
    profile,
    mlpNormalized,
    tokens,
    hidden,
    group(4),
    group(3),
  );
  const gate = cpuGemm(
    profile,
    mlpModulated,
    tokens,
    hidden,
    intermediate,
    weights.gateProjection,
  );
  const up = cpuGemm(
    profile,
    mlpModulated,
    tokens,
    hidden,
    intermediate,
    weights.upProjection,
  );
  const activated = Float32Array.from(gate, (value, index) => {
    const silu = cpuSiluF32(value);
    return profile === "raw-fp16"
      ? multiply(profile, activationValue(profile, silu), up[index]!)
      : multiply(profile, silu, up[index]!);
  });
  const projectedMlp = cpuGemm(
    profile,
    activated,
    tokens,
    intermediate,
    hidden,
    weights.downProjection,
  );
  return cpuGatedResidual(
    profile,
    afterCross,
    projectedMlp,
    tokens,
    hidden,
    group(5),
  );
}

function cpuGemm(
  profile: AceModelProfileId,
  activation: Float32Array,
  rows: number,
  inner: number,
  columns: number,
  rawWeight: readonly number[],
): Float32Array {
  const weight = rawWeight.map((value) => weightValue(profile, value));
  const output = new Float32Array(rows * columns);
  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      let sum = 0;
      for (let contracted = 0; contracted < inner; contracted += 1) {
        sum = add(
          profile,
          sum,
          multiply(
            profile,
            activation[row * inner + contracted]!,
            weight[column * inner + contracted]!,
          ),
        );
      }
      output[row * columns + column] = sum;
    }
  }
  return output;
}

function cpuRmsNorm(
  profile: AceModelProfileId,
  input: Float32Array,
  rows: number,
  width: number,
  rawWeight: readonly number[],
  epsilon: number,
): Float32Array {
  const weights = rawWeight.map((value) => weightValue(profile, value));
  const output = new Float32Array(rows * width);
  for (let row = 0; row < rows; row += 1) {
    const partial = new Float32Array(256);
    for (let lane = 0; lane < width; lane += 1) {
      const value = input[row * width + lane]!;
      partial[lane] = Math.fround(value * value);
    }
    for (let stride = 128; stride > 0; stride >>= 1) {
      for (let lane = 0; lane < stride; lane += 1) {
        partial[lane] = Math.fround(partial[lane]! + partial[lane + stride]!);
      }
    }
    const mean = Math.fround(partial[0]! / width);
    const inverseRms = Math.fround(1 / Math.sqrt(Math.fround(mean + epsilon)));
    for (let column = 0; column < width; column += 1) {
      const value = input[row * width + column]!;
      output[row * width + column] = profile === "raw-fp16"
        ? multiply(
          profile,
          activationValue(profile, Math.fround(value * inverseRms)),
          weights[column]!,
        )
        : Math.fround(
          Math.fround(value * inverseRms) * weights[column]!,
        );
    }
  }
  return output;
}

function cpuAdaLn(
  profile: AceModelProfileId,
  normalized: Float32Array,
  tokens: number,
  width: number,
  scale: Float32Array,
  shift: Float32Array,
): Float32Array {
  if (normalized.length !== tokens * width) throw new Error("CPU AdaLN shape mismatch");
  return Float32Array.from(normalized, (value, index) => {
    const feature = index % width;
    return add(
      profile,
      multiply(profile, value, add(profile, 1, scale[feature]!)),
      shift[feature]!,
    );
  });
}

function cpuGatedResidual(
  profile: AceModelProfileId,
  residual: Float32Array,
  branch: Float32Array,
  tokens: number,
  width: number,
  gate: Float32Array,
): Float32Array {
  if (residual.length !== tokens * width) throw new Error("CPU residual shape mismatch");
  return Float32Array.from(residual, (value, index) =>
    add(
      profile,
      value,
      multiply(profile, branch[index]!, gate[index % width]!),
    ));
}

function cpuRope(
  profile: AceModelProfileId,
  input: Float32Array,
  tokens: number,
  headDimension: number,
  cosine: Float32Array,
  sine: Float32Array,
): Float32Array {
  const half = headDimension / 2;
  const output = new Float32Array(input.length);
  for (let token = 0; token < tokens; token += 1) {
    for (let dimension = 0; dimension < headDimension; dimension += 1) {
      const index = token * headDimension + dimension;
      const rotatedDimension = dimension < half
        ? dimension + half
        : dimension - half;
      const sign = dimension < half ? -1 : 1;
      const left = Math.fround(input[index]! * cosine[index]!);
      const right = Math.fround(
        Math.fround(sign * input[token * headDimension + rotatedDimension]!) *
          sine[index]!,
      );
      output[index] = activationValue(profile, Math.fround(left + right));
    }
  }
  return output;
}

function cpuOnlineAttention(
  profile: AceModelProfileId,
  query: Float32Array,
  key: Float32Array,
  value: Float32Array,
  queryTokens: number,
  keyTokens: number,
  headDimension: number,
  mode: "sliding" | "full",
  slidingRadius: number,
): Float32Array {
  const output = new Float32Array(queryTokens * headDimension);
  const scale = Math.fround(1 / Math.sqrt(headDimension));
  for (let queryToken = 0; queryToken < queryTokens; queryToken += 1) {
    const keyStart = mode === "sliding"
      ? Math.max(0, queryToken - slidingRadius)
      : 0;
    const keyEnd = mode === "sliding"
      ? Math.min(keyTokens, queryToken + slidingRadius + 1)
      : keyTokens;
    let onlineMaximum = -3.4028234663852886e38;
    let onlineDenominator = 0;
    const weighted = new Float32Array(headDimension);
    for (let keyToken = keyStart; keyToken < keyEnd; keyToken += 1) {
      const partial = new Float32Array(128);
      for (let dimension = 0; dimension < headDimension; dimension += 1) {
        partial[dimension] = Math.fround(
          query[queryToken * headDimension + dimension]! *
            key[keyToken * headDimension + dimension]!,
        );
      }
      for (let stride = 64; stride > 0; stride >>= 1) {
        for (let lane = 0; lane < stride; lane += 1) {
          partial[lane] = Math.fround(partial[lane]! + partial[lane + stride]!);
        }
      }
      const score = Math.fround(partial[0]! * scale);
      const nextMaximum = Math.max(onlineMaximum, score);
      const alpha = Math.fround(Math.exp(Math.fround(onlineMaximum - nextMaximum)));
      const beta = Math.fround(Math.exp(Math.fround(score - nextMaximum)));
      onlineDenominator = Math.fround(
        Math.fround(onlineDenominator * alpha) + beta,
      );
      onlineMaximum = nextMaximum;
      for (let dimension = 0; dimension < headDimension; dimension += 1) {
        weighted[dimension] = Math.fround(
          Math.fround(weighted[dimension]! * alpha) +
            Math.fround(beta * value[keyToken * headDimension + dimension]!),
        );
      }
    }
    for (let dimension = 0; dimension < headDimension; dimension += 1) {
      output[queryToken * headDimension + dimension] = activationValue(
        profile,
        Math.fround(weighted[dimension]! / onlineDenominator),
      );
    }
  }
  return output;
}

function cpuSiluF32(value: number): number {
  const denominator = Math.fround(1 + Math.fround(Math.exp(Math.fround(-value))));
  return Math.fround(value / denominator);
}

function maximumDifference(left: Float32Array, right: Float32Array): number {
  if (left.length !== right.length) throw new Error("difference length mismatch");
  let maximum = 0;
  for (let index = 0; index < left.length; index += 1) {
    maximum = Math.max(maximum, Math.abs(left[index]! - right[index]!));
  }
  return maximum;
}

async function executeAndReadActivations(
  device: GPUDevice,
  dispatches: readonly { encode(pass: GPUComputePassEncoder): void }[],
  outputs: readonly GPUBuffer[],
  elementCounts: readonly number[],
  profile: AceModelProfileId,
): Promise<Float32Array[]> {
  const readbacks = outputs.map((_, index) => {
    const bytes = elementCounts[index]! * (profile === "raw-fp16" ? 2 : 4);
    return device.createBuffer({
      size: alignedSize(bytes),
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
  });
  try {
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    for (const dispatch of dispatches) dispatch.encode(pass);
    pass.end();
    for (let index = 0; index < outputs.length; index += 1) {
      const bytes = elementCounts[index]! * (profile === "raw-fp16" ? 2 : 4);
      encoder.copyBufferToBuffer(
        outputs[index]!,
        0,
        readbacks[index]!,
        0,
        alignedSize(bytes),
      );
    }
    device.queue.submit([encoder.finish()]);
    await Promise.all(readbacks.map((buffer) => buffer.mapAsync(GPUMapMode.READ)));
    return readbacks.map((buffer, index) => {
      const elements = elementCounts[index]!;
      const bytes = buffer.getMappedRange().slice(
        0,
        elements * (profile === "raw-fp16" ? 2 : 4),
      );
      return profile === "reference-bf16"
        ? Float32Array.from(new Float32Array(bytes, 0, elements))
        : Float32Array.from(new Uint16Array(bytes, 0, elements), fp16BitsToNumber);
    });
  } finally {
    for (const readback of readbacks) readback.destroy();
  }
}

function compare(
  profile: AceModelProfileId,
  operation: string,
  actual: Float32Array,
  expected: Float32Array,
): CaseResult {
  if (actual.length !== expected.length) throw new Error(`${operation} length mismatch`);
  let maximumAbsoluteError = 0;
  const tolerance = profile === "raw-fp16" ? 0.015625 : 1e-5;
  for (let index = 0; index < actual.length; index += 1) {
    const error = Math.abs(actual[index]! - expected[index]!);
    maximumAbsoluteError = Math.max(maximumAbsoluteError, error);
    if (error > tolerance) {
      throw new Error(
        `${profile} ${operation}[${index}] ${actual[index]} != ${expected[index]} (${error})`,
      );
    }
  }
  return { profile, operation, valuesChecked: actual.length, maximumAbsoluteError };
}

function activationBuffer(
  device: GPUDevice,
  profile: AceModelProfileId,
  values: readonly number[],
): GPUBuffer {
  return storageBuffer(
    device,
    profile === "reference-bf16"
      ? Float32Array.from(values)
      : Uint16Array.from(values, numberToFp16Bits),
  );
}

function weightBuffer(
  device: GPUDevice,
  profile: AceModelProfileId,
  values: readonly number[],
): GPUBuffer {
  return storageBuffer(
    device,
    profile === "reference-bf16"
      ? packBf16(values)
      : Uint16Array.from(values, numberToFp16Bits),
  );
}

function activationOutput(
  device: GPUDevice,
  profile: AceModelProfileId,
  elements: number,
): GPUBuffer {
  return device.createBuffer({
    size: alignedSize(elements * (profile === "raw-fp16" ? 2 : 4)),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
}

function storageBuffer(
  device: GPUDevice,
  data: ArrayBufferView<ArrayBufferLike>,
): GPUBuffer {
  const bytes = new Uint8Array(alignedSize(data.byteLength));
  bytes.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
  const buffer = device.createBuffer({
    size: bytes.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(buffer, 0, bytes);
  return buffer;
}

function binding(buffer: GPUBuffer): GPUBufferBinding {
  return { buffer, offset: 0, size: buffer.size };
}

function alignedSize(bytes: number): number {
  return Math.max(4, Math.ceil(bytes / 4) * 4);
}

function activationValue(profile: AceModelProfileId, value: number): number {
  return profile === "raw-fp16" ? roundFp16(value) : Math.fround(value);
}

function weightValue(profile: AceModelProfileId, value: number): number {
  return profile === "raw-fp16" ? roundFp16(value) : roundBf16(value);
}

function add(profile: AceModelProfileId, left: number, right: number): number {
  return profile === "raw-fp16" ? roundFp16(left + right) : Math.fround(left + right);
}

function subtract(profile: AceModelProfileId, left: number, right: number): number {
  return profile === "raw-fp16" ? roundFp16(left - right) : Math.fround(left - right);
}

function multiply(profile: AceModelProfileId, left: number, right: number): number {
  return profile === "raw-fp16" ? roundFp16(left * right) : Math.fround(left * right);
}

function packBf16(values: readonly number[]): Uint32Array<ArrayBuffer> {
  const words = new Uint32Array(Math.ceil(values.length / 2));
  for (let index = 0; index < values.length; index += 1) {
    const bits = numberToBf16Bits(values[index]!);
    words[index >> 1] = words[index >> 1]! | (bits << ((index & 1) * 16));
  }
  return words;
}

function roundBf16(value: number): number {
  const bits = numberToBf16Bits(value);
  return new Float32Array(new Uint32Array([bits << 16]).buffer)[0]!;
}

function numberToBf16Bits(value: number): number {
  const f32 = new Float32Array([value]);
  const bits = new Uint32Array(f32.buffer)[0]!;
  return ((bits + 0x7fff + ((bits >>> 16) & 1)) >>> 16) & 0xffff;
}

function roundFp16(value: number): number {
  return fp16BitsToNumber(numberToFp16Bits(value));
}

function numberToFp16Bits(value: number): number {
  const float = new Float32Array(1);
  const bits = new Uint32Array(float.buffer);
  float[0] = value;
  const word = bits[0]!;
  const sign = (word >>> 16) & 0x8000;
  const exponent = (word >>> 23) & 0xff;
  let mantissa = word & 0x7fffff;
  if (exponent === 0xff) return sign | (mantissa === 0 ? 0x7c00 : 0x7e00);
  const halfExponent = exponent - 127 + 15;
  if (halfExponent >= 0x1f) return sign | 0x7c00;
  if (halfExponent <= 0) {
    if (halfExponent < -10) return sign;
    mantissa |= 0x800000;
    const shift = 14 - halfExponent;
    const rounded = (mantissa + (1 << (shift - 1)) - 1 + ((mantissa >> shift) & 1)) >> shift;
    return sign | rounded;
  }
  const roundedMantissa = mantissa + 0xfff + ((mantissa >>> 13) & 1);
  if ((roundedMantissa & 0x800000) !== 0) {
    if (halfExponent + 1 >= 0x1f) return sign | 0x7c00;
    return sign | ((halfExponent + 1) << 10);
  }
  return sign | (halfExponent << 10) | (roundedMantissa >>> 13);
}

function fp16BitsToNumber(bits: number): number {
  const sign = (bits & 0x8000) << 16;
  let exponent = (bits >>> 10) & 0x1f;
  let mantissa = bits & 0x3ff;
  let word: number;
  if (exponent === 0) {
    if (mantissa === 0) {
      word = sign;
    } else {
      exponent = 1;
      while ((mantissa & 0x400) === 0) {
        mantissa <<= 1;
        exponent -= 1;
      }
      mantissa &= 0x3ff;
      word = sign | ((exponent + 112) << 23) | (mantissa << 13);
    }
  } else if (exponent === 0x1f) {
    word = sign | 0x7f800000 | (mantissa << 13);
  } else {
    word = sign | ((exponent + 112) << 23) | (mantissa << 13);
  }
  return new Float32Array(new Uint32Array([word >>> 0]).buffer)[0]!;
}

function f32BitsToNumber(bits: number): number {
  return new Float32Array(new Uint32Array([bits >>> 0]).buffer)[0]!;
}

function requireResultNode(): HTMLElement {
  const node = document.querySelector<HTMLElement>("#result");
  if (node === null) throw new Error("Missing #result node");
  return node;
}

function finish(status: "passed" | "failed", text: string): void {
  document.body.dataset.status = status;
  resultNode.textContent = text;
}
