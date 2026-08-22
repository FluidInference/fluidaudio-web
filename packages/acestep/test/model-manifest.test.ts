import { describe, expect, it } from "vitest";

import {
  ACE_DIT_DENSE_K4_FP16_LAYOUT,
  ACE_DIT_DENSE_K4_FP16_TRANSFORMATION,
  ACE_DIT_GEMM_FP16_TRANSFORMATION,
  ACE_DIT_GEMM_PACKED_BF16_TRANSFORMATION,
  ACE_DIT_GEMM_TILE_LAYOUT,
  ACE_EXPERIMENTAL_DIT_DENSE_LOGICAL_TENSOR_COUNT,
  ACE_EXPERIMENTAL_DIT_DENSE_PACKAGE_CONVERTER_REVISION,
  ACE_EXPERIMENTAL_DIT_DENSE_PARAMETER_BYTES,
  ACE_EXPERIMENTAL_DIT_DENSE_PARAMETER_ELEMENTS,
  ACE_EXPERIMENTAL_VAE_LOGICAL_TENSOR_COUNT,
  ACE_EXPERIMENTAL_VAE_PACKAGE_CONVERTER_REVISION,
  ACE_EXPERIMENTAL_VAE_PARAMETER_BYTES,
  ACE_EXPERIMENTAL_VAE_PARAMETER_ELEMENTS,
  ACE_EXPERIMENTAL_VAE_TENSOR_RECORD_COUNT,
  ACE_OPT_0054_EXPERIMENTAL_VAE_PACKAGE_CONVERTER_REVISION,
  ACE_MAX_WEIGHT_SHARD_BYTES,
  ACE_VAE_BIAS_FP16_TRANSFORMATION,
  ACE_VAE_CHANNEL_VECTOR_FP16_LAYOUT,
  ACE_VAE_CHANNEL_VECTOR_FP16_TRANSFORMATION,
  ACE_VAE_CONV1D_FP16_LAYOUT,
  ACE_VAE_CONV1D_FP16_TRANSFORMATION,
  ACE_VAE_CONV1D_LAYOUT,
  ACE_VAE_CONV1D_TRANSFORMATION,
  ACE_VAE_CONV_TRANSPOSE1D_FP16_POLYPHASE_LAYOUT,
  ACE_VAE_CONV_TRANSPOSE1D_FP16_POLYPHASE_TRANSFORMATION,
  ACE_VAE_CONV_TRANSPOSE1D_FP16_LAYOUT,
  ACE_VAE_CONV_TRANSPOSE1D_FP16_TRANSFORMATION,
  ACE_VAE_CONV_TRANSPOSE1D_K4_FP16_LAYOUT,
  ACE_VAE_CONV_TRANSPOSE1D_K4_FP16_TRANSFORMATION,
  ACE_VAE_K1_FP16_TILE_LAYOUT,
  ACE_VAE_K1_FP16_TILE_TRANSFORMATION,
  ACE_VAE_K7_ROW_REUSE_FP16_LAYOUT,
  ACE_VAE_K7_ROW_REUSE_FP16_TRANSFORMATION,
  ACE_VAE_REVISION7_K7_ROW_REUSE_CONTRACTS,
  ACE_VAE_REVISION7_TRANSPOSE_K4_CONTRACTS,
  parseAcePackageManifest,
  resolveAceLogicalTensor,
  resolveAceLogicalTensorRows,
} from "../src/model/manifest.js";
import { syntheticAceManifest } from "./model-fixtures.js";

describe("ACE package manifest", () => {
  it("accepts the canonical reference schema", () => {
    const manifest = parseAcePackageManifest(syntheticAceManifest(), "reference");
    expect(manifest.profile).toBe("reference");
    expect(Object.keys(manifest.tensors)).toEqual(["ace.decoder.weight"]);
    expect(manifest.files).toHaveLength(5);
  });

  it("rejects unknown fields at every trust boundary", () => {
    const root = syntheticAceManifest();
    root.surprise = true;
    expect(() => parseAcePackageManifest(root)).toThrow(/unknown or missing fields/);

    const nested = syntheticAceManifest();
    const files = nested.files as Array<Record<string, unknown>>;
    files[0]!.surprise = true;
    expect(() => parseAcePackageManifest(nested)).toThrow(/unknown or missing fields/);
  });

  it("binds the selected profile and its exact dtype policy", () => {
    expect(() =>
      parseAcePackageManifest(syntheticAceManifest(), "fp16"),
    ).toThrow(/does not match fp16/);

    const manifest = syntheticAceManifest();
    manifest.profile = "fp16";
    expect(() => parseAcePackageManifest(manifest, "fp16")).toThrow(
      /violates the fp16 storage policy/,
    );
  });

  it("accepts only the exact revision-6 experimental FP16 VAE contract", () => {
    const raw = experimentalVaeManifest();
    const manifest = parseAcePackageManifest(raw, "fp16-vae-experimental");
    const records = Object.values(manifest.tensors);
    const vaeTensors = records.filter((tensor) => tensor.phase === "vae");
    const constants = records.filter((tensor) => tensor.phase === "constants");
    expect(vaeTensors).toHaveLength(ACE_EXPERIMENTAL_VAE_TENSOR_RECORD_COUNT);
    expect(new Set(vaeTensors.map((tensor) => tensor.logicalTensor)).size).toBe(
      ACE_EXPERIMENTAL_VAE_LOGICAL_TENSOR_COUNT,
    );
    expect(constants).toHaveLength(1);
    expect(constants[0]!.logicalTensor).toBe("constants.silence_latent");
    expect(manifest.accounting.constantTensors).toBe(1);
    expect(manifest.accounting.outputTensorsAfterRowSharding).toBe(
      ACE_EXPERIMENTAL_VAE_TENSOR_RECORD_COUNT + 1,
    );
    expect(vaeTensors.reduce((total, tensor) => total + tensor.byteLength, 0)).toBe(
      ACE_EXPERIMENTAL_VAE_PARAMETER_BYTES,
    );
    expect(manifest.provenance.converterRevision).toBe(
      ACE_EXPERIMENTAL_VAE_PACKAGE_CONVERTER_REVISION,
    );
    expect(manifest.tensors["vae.decoder.fixture-000.weight"]!.layout).toBe(
      ACE_VAE_CONV1D_FP16_LAYOUT,
    );
    expect(manifest.tensors["vae.decoder.fixture-001.weight"]!.layout).toBe(
      ACE_VAE_CONV_TRANSPOSE1D_FP16_POLYPHASE_LAYOUT,
    );
    expect(manifest.tensors["vae.decoder.fixture-002.weight"]!.layout).toBe(
      ACE_VAE_K1_FP16_TILE_LAYOUT,
    );
    expect(manifest.tensors["vae.decoder.fixture-003.alpha"]!.layout).toBe(
      ACE_VAE_CHANNEL_VECTOR_FP16_LAYOUT,
    );

    const unpackedK1 = experimentalVaeManifest();
    const unpackedK1Tensor = (
      unpackedK1.tensors as Record<string, Record<string, unknown>>
    )["vae.decoder.fixture-002.weight"]!;
    unpackedK1Tensor.storageShape = [128, 1, 128];
    unpackedK1Tensor.layout = ACE_VAE_CONV1D_FP16_LAYOUT;
    unpackedK1Tensor.transformation = ACE_VAE_CONV1D_FP16_TRANSFORMATION;
    expect(() => parseAcePackageManifest(unpackedK1)).toThrow(
      /revision-6 layout contract/,
    );

    const nativeTranspose = experimentalVaeManifest();
    const nativeTransposeTensor = (
      nativeTranspose.tensors as Record<string, Record<string, unknown>>
    )["vae.decoder.fixture-001.weight"]!;
    nativeTransposeTensor.storageShape = [2, 4, 3];
    nativeTransposeTensor.layout = ACE_VAE_CONV_TRANSPOSE1D_FP16_LAYOUT;
    nativeTransposeTensor.transformation =
      ACE_VAE_CONV_TRANSPOSE1D_FP16_TRANSFORMATION;
    expect(() => parseAcePackageManifest(nativeTranspose)).toThrow(
      /revision-6 layout contract/,
    );

    const legacyRevision = experimentalVaeManifest();
    (legacyRevision.provenance as Record<string, unknown>).converterRevision = 4;
    expect(() => parseAcePackageManifest(legacyRevision)).toThrow(
      /converter revision/,
    );

    const repurposedStable = experimentalVaeManifest();
    repurposedStable.profile = "fp16";
    (repurposedStable.provenance as Record<string, unknown>).converterRevision = 4;
    expect(() => parseAcePackageManifest(repurposedStable)).toThrow(
      /violates the fp16 storage policy/,
    );

    const relabelledNativeSource = experimentalVaeManifest();
    const relabelledTensor = (
      relabelledNativeSource.tensors as Record<string, Record<string, unknown>>
    )["vae.decoder.fixture-000.weight"]!;
    relabelledTensor.transformation = ACE_VAE_BIAS_FP16_TRANSFORMATION;
    relabelledTensor.layout = "source-row-major";
    expect(() => parseAcePackageManifest(relabelledNativeSource)).toThrow(
      /Conv1d native-layout contract/,
    );

    const sharded = experimentalVaeManifest();
    const shardedTensor = (
      sharded.tensors as Record<string, Record<string, unknown>>
    )["vae.decoder.fixture-004.bias"]!;
    shardedTensor.partEnd = (shardedTensor.partEnd as number) - 1;
    shardedTensor.storageShape = [(shardedTensor.storageShape as number[])[0]! - 1];
    shardedTensor.byteLength = (shardedTensor.byteLength as number) - 2;
    shardedTensor.layout = "row-shard-axis0";
    expect(() => parseAcePackageManifest(sharded)).toThrow(/is incomplete/);

    const missingSilence = experimentalVaeManifest();
    delete (missingSilence.tensors as Record<string, unknown>)[
      "constants.silence_latent"
    ];
    missingSilence.files = (missingSilence.files as Array<Record<string, unknown>>).filter(
      (file) => file.name !== "constants/silence-latent-f32.bin",
    );
    Object.assign(missingSilence.accounting as Record<string, unknown>, {
      constantTensors: 0,
      outputTensorsAfterRowSharding: ACE_EXPERIMENTAL_VAE_TENSOR_RECORD_COUNT,
    });
    expect(() => parseAcePackageManifest(missingSilence)).toThrow(
      /canonical constants\.silence_latent/,
    );

    const duplicateSilence = experimentalVaeManifest();
    const duplicateTensors = duplicateSilence.tensors as Record<
      string,
      Record<string, unknown>
    >;
    duplicateTensors["constants.extra"] = {
      ...duplicateTensors["constants.silence_latent"]!,
      shard: "constants/extra-f32.bin",
      logicalTensor: "constants.extra",
    };
    duplicateSilence.tensors = Object.fromEntries(
      Object.entries(duplicateTensors).sort(([left], [right]) =>
        left < right ? -1 : 1,
      ),
    );
    (duplicateSilence.files as Array<Record<string, unknown>>).push({
      name: "constants/extra-f32.bin",
      byteLength: 3_840_000,
      sha256: "9".repeat(64),
      kind: "constant",
    });
    (duplicateSilence.files as Array<Record<string, unknown>>).sort((left, right) =>
      String(left.name) < String(right.name) ? -1 : 1,
    );
    Object.assign(duplicateSilence.accounting as Record<string, unknown>, {
      constantTensors: 2,
      outputTensorsAfterRowSharding:
        ACE_EXPERIMENTAL_VAE_TENSOR_RECORD_COUNT + 2,
    });
    expect(() => parseAcePackageManifest(duplicateSilence)).toThrow(
      /canonical constants\.silence_latent/,
    );

    const relabelledSilence = experimentalVaeManifest();
    const relabelledTensors = relabelledSilence.tensors as Record<
      string,
      Record<string, unknown>
    >;
    const silence = relabelledTensors["constants.silence_latent"]!;
    delete relabelledTensors["constants.silence_latent"];
    relabelledTensors["constants.renamed"] = {
      ...silence,
      logicalTensor: "constants.renamed",
    };
    relabelledSilence.tensors = Object.fromEntries(
      Object.entries(relabelledTensors).sort(([left], [right]) =>
        left < right ? -1 : 1,
      ),
    );
    expect(() => parseAcePackageManifest(relabelledSilence)).toThrow(
      /canonical constants\.silence_latent/,
    );

    const badBiasSource = experimentalVaeManifest();
    const badBiasSourceTensor = (
      badBiasSource.tensors as Record<string, Record<string, unknown>>
    )["vae.decoder.fixture-004.bias"]!;
    badBiasSourceTensor.source = "vae-weights:decoder.unrelated.bias";
    expect(() => parseAcePackageManifest(badBiasSource)).toThrow(
      /VAE bias contract/,
    );

    const badBiasRank = experimentalVaeManifest();
    const badBiasRankTensor = (
      badBiasRank.tensors as Record<string, Record<string, unknown>>
    )["vae.decoder.fixture-004.bias"]!;
    const biasElements = (badBiasRankTensor.logicalShape as number[])[0]!;
    badBiasRankTensor.logicalShape = [biasElements, 1];
    badBiasRankTensor.storageShape = [biasElements, 1];
    expect(() => parseAcePackageManifest(badBiasRank)).toThrow(
      /VAE bias contract/,
    );

    const untransformedBias = experimentalVaeManifest();
    const untransformedBiasTensor = (
      untransformedBias.tensors as Record<string, Record<string, unknown>>
    )["vae.decoder.fixture-004.bias"]!;
    untransformedBiasTensor.transformation = "bf16-to-ieee-fp16";
    expect(() => parseAcePackageManifest(untransformedBias)).toThrow(
      /VAE bias contract/,
    );
  });

  it("accepts revision 7 only through the authenticated mixed VAE layout seam", () => {
    const raw = experimentalVaeRevision7Manifest();
    expect(() =>
      parseAcePackageManifest(raw, "fp16-vae-experimental"),
    ).toThrow(/converter revision/);
    const manifest = parseAcePackageManifest(
      raw,
      "fp16-vae-experimental",
      { authenticatedVaeConverterRevision: 7 },
    );
    const vaeTensors = Object.values(manifest.tensors).filter(
      (tensor) => tensor.phase === "vae",
    );
    expect(manifest.provenance.converterRevision).toBe(
      ACE_OPT_0054_EXPERIMENTAL_VAE_PACKAGE_CONVERTER_REVISION,
    );
    expect(
      vaeTensors.filter(
        (tensor) =>
          tensor.transformation === ACE_VAE_K7_ROW_REUSE_FP16_TRANSFORMATION,
      ),
    ).toHaveLength(ACE_VAE_REVISION7_K7_ROW_REUSE_CONTRACTS.length);
    expect(
      vaeTensors.filter(
        (tensor) =>
          tensor.transformation ===
            ACE_VAE_CONV_TRANSPOSE1D_K4_FP16_TRANSFORMATION,
      ),
    ).toHaveLength(ACE_VAE_REVISION7_TRANSPOSE_K4_CONTRACTS.length);
    expect(
      manifest.tensors["vae.decoder.block.0.res_unit1.conv1.weight"]!
        .storageShape,
    ).toEqual([7, 256, 16, 32, 2, 4]);
    expect(
      manifest.tensors["vae.decoder.block.2.res_unit1.conv1.weight"]!.layout,
    ).toBe(ACE_VAE_CONV1D_FP16_LAYOUT);
    expect(
      manifest.tensors["vae.decoder.block.3.conv_t1.weight"]!.storageShape,
    ).toEqual([4, 2, 64, 1, 32, 4, 4]);

    const forgedC256 = experimentalVaeRevision7Manifest();
    const forgedC256Tensor = (
      forgedC256.tensors as Record<string, Record<string, unknown>>
    )["vae.decoder.block.2.res_unit1.conv1.weight"]!;
    forgedC256Tensor.transformation = ACE_VAE_K7_ROW_REUSE_FP16_TRANSFORMATION;
    forgedC256Tensor.layout = ACE_VAE_K7_ROW_REUSE_FP16_LAYOUT;
    forgedC256Tensor.storageShape = [7, 64, 4, 32, 2, 4];
    expect(() => parseAcePackageManifest(
      forgedC256,
      "fp16-vae-experimental",
      { authenticatedVaeConverterRevision: 7 },
    )).toThrow(/revision-7 layout contract/);

    const wrongProfile = experimentalVaeRevision7Manifest();
    expect(() => parseAcePackageManifest(
      wrongProfile,
      "reference",
      { authenticatedVaeConverterRevision: 7 },
    )).toThrow(/does not match reference|wrong profile/);
  });

  it("accepts only the revision-8 K4 replacement for all mixed DiT dense weights", () => {
    const raw = experimentalDitDenseK4Manifest();
    const manifest = parseAcePackageManifest(
      raw,
      "fp16-dit-dense-experimental",
    );
    const ditTensors = Object.values(manifest.tensors).filter(
      (tensor) => tensor.phase === "dit",
    );
    const denseTensors = ditTensors.filter(
      (tensor) => tensor.dtype === "float16",
    );
    expect(ditTensors).toHaveLength(ACE_EXPERIMENTAL_DIT_DENSE_LOGICAL_TENSOR_COUNT);
    expect(denseTensors).toHaveLength(24 * 9);
    expect(new Set(denseTensors.map((tensor) => tensor.layout))).toEqual(
      new Set([ACE_DIT_DENSE_K4_FP16_LAYOUT]),
    );
    expect(new Set(denseTensors.map((tensor) => tensor.transformation))).toEqual(
      new Set([ACE_DIT_DENSE_K4_FP16_TRANSFORMATION]),
    );
    expect(ditTensors.reduce((total, tensor) => total + tensor.byteLength, 0)).toBe(
      ACE_EXPERIMENTAL_DIT_DENSE_PARAMETER_BYTES,
    );
    expect(manifest.provenance.converterRevision).toBe(
      ACE_EXPERIMENTAL_DIT_DENSE_PACKAGE_CONVERTER_REVISION,
    );
    const down = manifest.tensors[
      "ace.decoder.layers.23.mlp.down_proj.weight"
    ]!;
    expect(down.logicalShape).toEqual([2_048, 6_144]);
    expect(down.storageShape).toEqual([16, 1_536, 4, 32, 4]);

    const legacyRevision = experimentalDitDenseK4Manifest();
    (legacyRevision.provenance as Record<string, unknown>).converterRevision = 7;
    expect(() => parseAcePackageManifest(legacyRevision)).toThrow(
      /converter revision/,
    );

    const oldPhysicalContract = experimentalDitDenseK4Manifest();
    const oldDense = experimentalDitDenseTensor(oldPhysicalContract);
    oldDense.layout = "dit-gemm-n256-k32-tile-major-v1";
    oldDense.transformation =
      "bf16-to-ieee-fp16-dit-gemm-n256-k32-tile-major-v1";
    oldDense.storageShape = oldDense.logicalShape;
    expect(() => parseAcePackageManifest(oldPhysicalContract)).toThrow(
      /transformation is invalid/,
    );

    const logicalStorageForgery = experimentalDitDenseK4Manifest();
    const forgedDense = experimentalDitDenseTensor(logicalStorageForgery);
    forgedDense.storageShape = forgedDense.logicalShape;
    expect(() => parseAcePackageManifest(logicalStorageForgery)).toThrow(
      /storage shape/,
    );

    const missingDense = experimentalDitDenseK4Manifest();
    delete (missingDense.tensors as Record<string, unknown>)[
      "ace.decoder.layers.0.self_attn.q_proj.weight"
    ];
    Object.assign(missingDense.accounting as Record<string, unknown>, {
      directlyIncluded: ACE_EXPERIMENTAL_DIT_DENSE_LOGICAL_TENSOR_COUNT - 1,
      excluded: 1,
      outputTensorsBeforeRowSharding:
        ACE_EXPERIMENTAL_DIT_DENSE_LOGICAL_TENSOR_COUNT - 1,
      outputTensorsAfterRowSharding: ACE_EXPERIMENTAL_DIT_DENSE_LOGICAL_TENSOR_COUNT,
    });
    expect(() => parseAcePackageManifest(missingDense)).toThrow(
      /not exactly the 24 repeated layers/,
    );
  });

  it("rejects unpinned source and provenance revisions", () => {
    const sourceChanged = syntheticAceManifest();
    const source = (sourceChanged.source as Array<Record<string, unknown>>)[0]!;
    source.revision = "0".repeat(40);
    expect(() => parseAcePackageManifest(sourceChanged)).toThrow(/pinned source/);

    const provenanceChanged = syntheticAceManifest();
    const provenance = provenanceChanged.provenance as Record<string, unknown>;
    provenance.converterRevision = 5;
    expect(() => parseAcePackageManifest(provenanceChanged)).toThrow(
      /converter revision/,
    );
  });

  it("rejects unsafe paths, excessive shards, and changed license identities", () => {
    const unsafe = syntheticAceManifest();
    const unsafeFiles = unsafe.files as Array<Record<string, unknown>>;
    unsafeFiles.at(-1)!.name = "../weights.bin";
    expect(() => parseAcePackageManifest(unsafe)).toThrow(/safe canonical relative path/);

    const excessive = syntheticAceManifest();
    const excessiveFiles = excessive.files as Array<Record<string, unknown>>;
    excessiveFiles.at(-1)!.byteLength = ACE_MAX_WEIGHT_SHARD_BYTES + 1;
    expect(() => parseAcePackageManifest(excessive)).toThrow(/weight-shard limit/);

    const licenseChanged = syntheticAceManifest();
    const licenseFiles = licenseChanged.files as Array<Record<string, unknown>>;
    licenseFiles[1]!.sha256 = "f".repeat(64);
    expect(() => parseAcePackageManifest(licenseChanged)).toThrow(
      /license payload.*changed/,
    );
  });

  it("rejects overlapping, incomplete, and storage-inconsistent tensor parts", () => {
    const storage = syntheticAceManifest();
    const storageTensor = (storage.tensors as Record<string, Record<string, unknown>>)[
      "ace.decoder.weight"
    ]!;
    storageTensor.storageShape = [2];
    expect(() => parseAcePackageManifest(storage)).toThrow(/storage shape/);

    const incomplete = syntheticAceManifest();
    const incompleteTensor = (
      incomplete.tensors as Record<string, Record<string, unknown>>
    )["ace.decoder.weight"]!;
    incompleteTensor.partEnd = 1;
    incompleteTensor.storageShape = [1];
    incompleteTensor.byteLength = 4;
    incompleteTensor.layout = "row-shard-axis0-bf16-pairs-lsb-u32";
    expect(() => parseAcePackageManifest(incomplete)).toThrow(/is incomplete/);

    const overlap = syntheticAceManifest();
    const tensors = overlap.tensors as Record<string, Record<string, unknown>>;
    const first = tensors["ace.decoder.weight"]!;
    tensors["ace.decoder.weight.part"] = {
      ...first,
      logicalTensor: "another.weight",
    };
    (overlap.accounting as Record<string, unknown>).outputTensorsAfterRowSharding = 2;
    (overlap.accounting as Record<string, unknown>).outputTensorsBeforeRowSharding = 2;
    (overlap.accounting as Record<string, unknown>).directlyIncluded = 2;
    (overlap.accounting as Record<string, unknown>).sourceTensors = 2;
    ((overlap.source as Array<Record<string, unknown>>)[0]!).tensorCount = 2;
    expect(() => parseAcePackageManifest(overlap)).toThrow(/overlaps/);
  });

  it("accepts only the versioned operation-native VAE layout contract", () => {
    const native = syntheticAceManifest();
    const source = (native.source as Array<Record<string, unknown>>)[0]!;
    source.key = "vae-weights";
    const tensor = (native.tensors as Record<string, Record<string, unknown>>)[
      "ace.decoder.weight"
    ]!;
    tensor.dtype = "float32";
    tensor.logicalShape = [2, 3, 4];
    tensor.storageShape = [2, 3, 4];
    tensor.byteLength = 96;
    tensor.layout = ACE_VAE_CONV1D_LAYOUT;
    tensor.source = "vae-weights:decoder.conv1.weight_v";
    tensor.transformation = ACE_VAE_CONV1D_TRANSFORMATION;
    tensor.phase = "vae";
    tensor.lifetime = "vae";
    tensor.partEnd = 2;
    expect(parseAcePackageManifest(native).tensors["ace.decoder.weight"]!.layout).toBe(
      ACE_VAE_CONV1D_LAYOUT,
    );

    tensor.layout = "source-row-major";
    expect(() => parseAcePackageManifest(native)).toThrow(/layout is inconsistent/);
    tensor.layout = ACE_VAE_CONV1D_LAYOUT;
    tensor.source = "vae-weights:decoder.block.0.conv_t1.weight_v";
    expect(() => parseAcePackageManifest(native)).toThrow(/Conv1d native-layout/);
  });

  it("accepts only exact canonical DiT GEMM weights in the tile-major layout", () => {
    const canonicalSources = canonicalDitGemmSourceTensors();
    expect(canonicalSources).toHaveLength(271);
    expect(new Set(canonicalSources).size).toBe(271);
    for (const sourceTensor of canonicalSources) {
      const reference = tiledDitGemmManifest();
      tiledDitGemmTensor(reference).source = `ace-turbo-weights:${sourceTensor}`;
      expect(
        parseAcePackageManifest(reference, "reference").tensors[
          "ace.decoder.condition_embedder.weight"
        ]!.layout,
      ).toBe(ACE_DIT_GEMM_TILE_LAYOUT);
    }

    const fp16 = tiledDitGemmManifest();
    fp16.profile = "fp16";
    const fp16Tensor = tiledDitGemmTensor(fp16);
    fp16Tensor.dtype = "float16";
    fp16Tensor.storageShape = [128, 32];
    fp16Tensor.transformation = ACE_DIT_GEMM_FP16_TRANSFORMATION;
    expect(
      parseAcePackageManifest(fp16, "fp16").tensors[
        "ace.decoder.condition_embedder.weight"
      ]!.layout,
    ).toBe(ACE_DIT_GEMM_TILE_LAYOUT);

    for (const invalidSource of [
      "ace-turbo-weights:decoder.layers.24.self_attn.q_proj.weight",
      "ace-turbo-weights:decoder.layers.00.self_attn.q_proj.weight",
      "ace-turbo-weights:decoder.layers.0.self_attn.q_proj.bias",
      "ace-turbo-weights:decoder.layers.0.mlp.down_proj.weight.extra",
      "ace-turbo-weights:decoder.time_embed.linear_3.weight",
      "ace-turbo-weights:decoder.input_proj.weight",
      "ace-turbo-weights:encoder.layers.0.self_attn.q_proj.weight",
    ]) {
      const invalid = tiledDitGemmManifest();
      tiledDitGemmTensor(invalid).source = invalidSource;
      expect(() => parseAcePackageManifest(invalid)).toThrow(
        /DiT GEMM tile-major contract/,
      );
    }

    const wrongPhase = tiledDitGemmManifest();
    const wrongPhaseTensor = tiledDitGemmTensor(wrongPhase);
    wrongPhaseTensor.phase = "conditioner";
    wrongPhaseTensor.lifetime = "conditioner";
    expect(() => parseAcePackageManifest(wrongPhase)).toThrow(
      /DiT GEMM tile-major contract/,
    );

    for (const logicalShape of [
      [128],
      [127, 32],
      [128, 31],
      [128, 32, 1],
    ]) {
      const invalid = tiledDitGemmManifest();
      tiledDitGemmTensor(invalid).logicalShape = logicalShape;
      expect(() => parseAcePackageManifest(invalid)).toThrow(
        /DiT GEMM tile-major contract/,
      );
    }

    const wrongLayout = tiledDitGemmManifest();
    tiledDitGemmTensor(wrongLayout).layout =
      "source-row-major-bf16-pairs-lsb-u32";
    expect(() => parseAcePackageManifest(wrongLayout)).toThrow(
      /layout is inconsistent/,
    );

    const relabelledRowMajor = tiledDitGemmManifest();
    const relabelledTensor = tiledDitGemmTensor(relabelledRowMajor);
    relabelledTensor.transformation = "preserve-bf16-bits-pack-u32-pairs";
    relabelledTensor.layout = "source-row-major-bf16-pairs-lsb-u32";
    expect(() => parseAcePackageManifest(relabelledRowMajor)).toThrow(
      /DiT GEMM tile-major contract/,
    );

    const rowSharded = tiledDitGemmManifest();
    tiledDitGemmTensor(rowSharded).partEnd = 64;
    expect(() => parseAcePackageManifest(rowSharded)).toThrow(
      /complete tile-major DiT GEMM matrix/,
    );
  });

  it("resolves logical tensor pieces and embedding row ranges across shards", () => {
    const raw = syntheticAceManifest();
    const files = raw.files as Array<Record<string, unknown>>;
    files.at(-1)!.byteLength = 512;
    const tensors = raw.tensors as Record<string, Record<string, unknown>>;
    const first = tensors["ace.decoder.weight"]!;
    first.logicalShape = [4, 2];
    first.storageShape = [2];
    first.byteLength = 8;
    first.layout = "row-shard-axis0-bf16-pairs-lsb-u32";
    first.partStart = 0;
    first.partEnd = 2;
    tensors["ace.decoder.weight.rows-000002-000004"] = {
      ...first,
      byteOffset: 256,
      partStart: 2,
      partEnd: 4,
    };
    (raw.accounting as Record<string, unknown>).outputTensorsAfterRowSharding = 2;

    const manifest = parseAcePackageManifest(raw);
    const resolved = resolveAceLogicalTensor(manifest, "ace.decoder.weight");
    expect(resolved.logicalShape).toEqual([4, 2]);
    expect(resolved.parts.map((part) => part.tensor.partStart)).toEqual([0, 2]);
    expect(resolveAceLogicalTensorRows(manifest, "ace.decoder.weight", 1, 4)).toEqual([
      expect.objectContaining({
        tensorName: "ace.decoder.weight",
        logicalRowStart: 1,
        logicalRowEnd: 2,
        partRowStart: 1,
        partRowEnd: 2,
        storageElementOffset: 2,
        storageElementCount: 2,
      }),
      expect.objectContaining({
        tensorName: "ace.decoder.weight.rows-000002-000004",
        logicalRowStart: 2,
        logicalRowEnd: 4,
        partRowStart: 0,
        partRowEnd: 2,
        storageElementOffset: 0,
        storageElementCount: 4,
      }),
    ]);
    expect(() => resolveAceLogicalTensor(manifest, "missing")).toThrow(/is absent/);
    expect(() =>
      resolveAceLogicalTensorRows(manifest, "ace.decoder.weight", 4, 5),
    ).toThrow(/invalid requested row interval/);
  });
});

function tiledDitGemmManifest(): Record<string, unknown> {
  const manifest = syntheticAceManifest();
  const files = manifest.files as Array<Record<string, unknown>>;
  files.at(-1)!.byteLength = 8_192;
  const tensors = manifest.tensors as Record<string, Record<string, unknown>>;
  const tensor = tensors["ace.decoder.weight"]!;
  delete tensors["ace.decoder.weight"];
  Object.assign(tensor, {
    byteLength: 8_192,
    logicalShape: [128, 32],
    storageShape: [2_048],
    layout: ACE_DIT_GEMM_TILE_LAYOUT,
    source: "ace-turbo-weights:decoder.condition_embedder.weight",
    transformation: ACE_DIT_GEMM_PACKED_BF16_TRANSFORMATION,
    logicalTensor: "ace.decoder.condition_embedder.weight",
    partEnd: 128,
  });
  tensors["ace.decoder.condition_embedder.weight"] = tensor;
  return manifest;
}

function tiledDitGemmTensor(
  manifest: Record<string, unknown>,
): Record<string, unknown> {
  return (manifest.tensors as Record<string, Record<string, unknown>>)[
    "ace.decoder.condition_embedder.weight"
  ]!;
}

function canonicalDitGemmSourceTensors(): readonly string[] {
  const shared = [
    "decoder.condition_embedder.weight",
    "decoder.time_embed.linear_1.weight",
    "decoder.time_embed.linear_2.weight",
    "decoder.time_embed.time_proj.weight",
    "decoder.time_embed_r.linear_1.weight",
    "decoder.time_embed_r.linear_2.weight",
    "decoder.time_embed_r.time_proj.weight",
  ];
  const layerSuffixes = [
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "cross_attn.q_proj.weight",
    "cross_attn.k_proj.weight",
    "cross_attn.v_proj.weight",
    "cross_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
  ];
  return [
    ...shared,
    ...Array.from({ length: 24 }, (_, layer) =>
      layerSuffixes.map((suffix) => `decoder.layers.${layer}.${suffix}`),
    ).flat(),
  ];
}

function experimentalVaeManifest(): Record<string, unknown> {
  const manifest = syntheticAceManifest();
  manifest.profile = "fp16-vae-experimental";
  (manifest.provenance as Record<string, unknown>).converterRevision =
    ACE_EXPERIMENTAL_VAE_PACKAGE_CONVERTER_REVISION;

  const source = (manifest.source as Array<Record<string, unknown>>)[0]!;
  source.key = "vae-weights";
  source.tensorCount = ACE_EXPERIMENTAL_VAE_LOGICAL_TENSOR_COUNT;
  source.parameterCount = ACE_EXPERIMENTAL_VAE_PARAMETER_ELEMENTS;
  (manifest.source as Array<Record<string, unknown>>).push({
    key: "ace-silence-latent",
    component: "initial latent",
    repository: source.repository,
    revision: source.revision,
    path: "acestep-v15-turbo/silence_latent.pt",
    byteLength: 4,
    sha256: "f".repeat(64),
  });

  const tinyCount = ACE_EXPERIMENTAL_VAE_LOGICAL_TENSOR_COUNT - 6;
  const largeElements =
    (ACE_EXPERIMENTAL_VAE_PARAMETER_ELEMENTS - 24 - 24 - 16_384 - 4 -
      (tinyCount + 1)) / 2;
  expect(Number.isInteger(largeElements)).toBe(true);
  const elementCounts = [
    24,
    24,
    16_384,
    4,
    largeElements,
    largeElements,
    2,
    ...Array(tinyCount - 1).fill(1),
  ];
  const tensors: Record<string, Record<string, unknown>> = Object.create(null) as Record<
    string,
    Record<string, unknown>
  >;
  tensors["constants.silence_latent"] = {
    shard: "constants/silence-latent-f32.bin",
    byteOffset: 0,
    byteLength: 3_840_000,
    dtype: "float32",
    logicalShape: [1, 64, 15_000],
    storageShape: [1, 64, 15_000],
    layout: "contiguous-nct-f32",
    phase: "constants",
    lifetime: "initial-latent",
    source: "ace-silence-latent:silence_latent/data/0",
    transformation: "validated-pytorch-zip-storage-extraction",
    logicalTensor: "constants.silence_latent",
    partAxis: 0,
    partStart: 0,
    partEnd: 1,
  };
  const shardCursors = [0, 0];
  for (let index = 0; index < elementCounts.length; index += 1) {
    const elements = elementCounts[index]!;
    const byteLength = elements * 2;
    let shardIndex = 0;
    let byteOffset = align256(shardCursors[shardIndex]!);
    if (byteOffset + byteLength > ACE_MAX_WEIGHT_SHARD_BYTES) {
      shardIndex = 1;
      byteOffset = align256(shardCursors[shardIndex]!);
    }
    if (byteOffset + byteLength > ACE_MAX_WEIGHT_SHARD_BYTES) {
      throw new Error("experimental VAE fixture exceeded its two bounded shards");
    }
    shardCursors[shardIndex] = byteOffset + byteLength;

    const suffix = index.toString().padStart(3, "0");
    const name = index < 4
      ? `vae.decoder.fixture-${suffix}.${index === 3 ? "alpha" : "weight"}`
      : `vae.decoder.fixture-${suffix}.bias`;
    const shape = index === 0
      ? [2, 3, 4]
      : index === 1
        ? [2, 4, 3]
        : index === 2
          ? [128, 1, 128]
          : [elements];
    const sourceTensor = index === 0
      ? "decoder.conv1.weight_v"
      : index === 1
        ? "decoder.block.0.conv_t1.weight_v"
        : index === 2
          ? "decoder.block.0.res_unit1.conv2.weight_v"
          : index === 3
            ? "decoder.snake1.alpha"
            : "decoder.conv1.bias";
    const transformation = index === 0
      ? ACE_VAE_CONV1D_FP16_TRANSFORMATION
      : index === 1
        ? ACE_VAE_CONV_TRANSPOSE1D_FP16_POLYPHASE_TRANSFORMATION
        : index === 2
          ? ACE_VAE_K1_FP16_TILE_TRANSFORMATION
          : index === 3
            ? ACE_VAE_CHANNEL_VECTOR_FP16_TRANSFORMATION
            : ACE_VAE_BIAS_FP16_TRANSFORMATION;
    const layout = index === 0
      ? ACE_VAE_CONV1D_FP16_LAYOUT
      : index === 1
        ? ACE_VAE_CONV_TRANSPOSE1D_FP16_POLYPHASE_LAYOUT
        : index === 2
          ? ACE_VAE_K1_FP16_TILE_LAYOUT
          : index === 3
            ? ACE_VAE_CHANNEL_VECTOR_FP16_LAYOUT
            : "source-row-major";
    const storageShape = index === 1
      ? [2, 2, 3, 2]
      : index === 2
        ? [1, 4, 32, 128]
        : shape;
    tensors[name] = {
      shard: `weights/vae/shared-0${shardIndex}.bin`,
      byteOffset,
      byteLength,
      dtype: "float16",
      logicalShape: shape,
      storageShape,
      layout,
      phase: "vae",
      lifetime: "vae",
      source: `vae-weights:${sourceTensor}`,
      transformation,
      logicalTensor: name,
      partAxis: 0,
      partStart: 0,
      partEnd: shape[0],
    };
  }
  manifest.tensors = tensors;

  Object.assign(manifest.accounting as Record<string, unknown>, {
    sourceTensors: ACE_EXPERIMENTAL_VAE_LOGICAL_TENSOR_COUNT,
    directlyIncluded: ACE_EXPERIMENTAL_VAE_LOGICAL_TENSOR_COUNT,
    consumedByTransform: 0,
    excluded: 0,
    outputTensorsBeforeRowSharding: ACE_EXPERIMENTAL_VAE_LOGICAL_TENSOR_COUNT,
    constantTensors: 1,
    outputTensorsAfterRowSharding: ACE_EXPERIMENTAL_VAE_TENSOR_RECORD_COUNT + 1,
  });

  const files = manifest.files as Array<Record<string, unknown>>;
  files.unshift({
    name: "constants/silence-latent-f32.bin",
    byteLength: 3_840_000,
    sha256: "f".repeat(64),
    kind: "constant",
  });
  const firstWeight = files.at(-1)!;
  firstWeight.name = "weights/vae/shared-00.bin";
  firstWeight.byteLength = align256(shardCursors[0]!);
  firstWeight.sha256 = "d".repeat(64);
  files.push({
    name: "weights/vae/shared-01.bin",
    byteLength: align256(shardCursors[1]!),
    sha256: "e".repeat(64),
    kind: "weights",
  });
  return manifest;
}

function experimentalVaeRevision7Manifest(): Record<string, unknown> {
  const manifest = syntheticAceManifest();
  manifest.profile = "fp16-vae-experimental";
  (manifest.provenance as Record<string, unknown>).converterRevision =
    ACE_OPT_0054_EXPERIMENTAL_VAE_PACKAGE_CONVERTER_REVISION;
  const source = (manifest.source as Array<Record<string, unknown>>)[0]!;
  source.key = "vae-weights";
  source.tensorCount = ACE_EXPERIMENTAL_VAE_LOGICAL_TENSOR_COUNT;
  source.parameterCount = ACE_EXPERIMENTAL_VAE_PARAMETER_ELEMENTS;
  (manifest.source as Array<Record<string, unknown>>).push({
    key: "ace-silence-latent",
    component: "initial latent",
    repository: source.repository,
    revision: source.revision,
    path: "acestep-v15-turbo/silence_latent.pt",
    byteLength: 4,
    sha256: "f".repeat(64),
  });

  const runtimeShapes = new Map<string, readonly number[]>();
  const blockChannels = [1_024, 512, 256, 128, 128] as const;
  const blockInputs = [2_048, 1_024, 512, 256, 128] as const;
  const transposeKernels = [20, 12, 8, 8, 4] as const;
  runtimeShapes.set("decoder.conv1.weight_v", [2_048, 7, 64]);
  runtimeShapes.set("decoder.conv2.weight_v", [2, 7, 128]);
  runtimeShapes.set("decoder.conv1.bias", [2_048]);
  for (let block = 0; block < blockChannels.length; block += 1) {
    const channels = blockChannels[block]!;
    runtimeShapes.set(
      `decoder.block.${block}.conv_t1.weight_v`,
      [channels, transposeKernels[block]!, blockInputs[block]!],
    );
    runtimeShapes.set(`decoder.block.${block}.conv_t1.bias`, [channels]);
    for (let residual = 1; residual <= 3; residual += 1) {
      runtimeShapes.set(
        `decoder.block.${block}.res_unit${residual}.conv1.weight_v`,
        [channels, 7, channels],
      );
      runtimeShapes.set(
        `decoder.block.${block}.res_unit${residual}.conv2.weight_v`,
        [channels, 1, channels],
      );
      for (const convolution of [1, 2] as const) {
        runtimeShapes.set(
          `decoder.block.${block}.res_unit${residual}.conv${convolution}.bias`,
          [channels],
        );
      }
      for (const snake of [1, 2] as const) {
        for (const parameter of ["alpha", "beta"] as const) {
          runtimeShapes.set(
            `decoder.block.${block}.res_unit${residual}.snake${snake}.${parameter}`,
            [channels],
          );
        }
      }
    }
    for (const parameter of ["alpha", "beta"] as const) {
      runtimeShapes.set(
        `decoder.block.${block}.snake1.${parameter}`,
        [blockInputs[block]!],
      );
    }
  }
  for (const parameter of ["alpha", "beta"] as const) {
    runtimeShapes.set(`decoder.snake1.${parameter}`, [128]);
  }
  expect(runtimeShapes.size).toBe(ACE_EXPERIMENTAL_VAE_LOGICAL_TENSOR_COUNT);

  const rowReuse: ReadonlySet<string> = new Set<string>(
    ACE_VAE_REVISION7_K7_ROW_REUSE_CONTRACTS.map((contract) => contract.tensor),
  );
  const transposeK4: ReadonlyMap<string, "channel" | "row"> = new Map<
    string,
    "channel" | "row"
  >(
    ACE_VAE_REVISION7_TRANSPOSE_K4_CONTRACTS.map((contract) => [
      contract.tensor,
      contract.reuseAxis,
    ] as const),
  );
  const specs = [...runtimeShapes].map(([sourceTensor, logicalShape]) => {
    const name = sourceTensor.endsWith(".weight_v")
      ? `vae.${sourceTensor.slice(0, -".weight_v".length)}.weight`
      : `vae.${sourceTensor}`;
    let transformation: string;
    let layout: string;
    let storageShape = [...logicalShape];
    if (rowReuse.has(name)) {
      transformation = ACE_VAE_K7_ROW_REUSE_FP16_TRANSFORMATION;
      layout = ACE_VAE_K7_ROW_REUSE_FP16_LAYOUT;
      const [outputChannels, kernel, inputChannels] = logicalShape;
      storageShape = [
        kernel!, inputChannels! / 4, outputChannels! / 64, 32, 2, 4,
      ];
    } else if (transposeK4.has(name)) {
      transformation = ACE_VAE_CONV_TRANSPOSE1D_K4_FP16_TRANSFORMATION;
      layout = ACE_VAE_CONV_TRANSPOSE1D_K4_FP16_LAYOUT;
      const [outputChannels, kernel, inputChannels] = logicalShape;
      const outputsPerLane = transposeK4.get(name) === "channel" ? 8 : 4;
      storageShape = [
        kernel! / 2,
        2,
        inputChannels! / 4,
        outputChannels! / (32 * outputsPerLane),
        32,
        outputsPerLane,
        4,
      ];
    } else if (name === "vae.decoder.block.0.conv_t1.weight") {
      transformation = ACE_VAE_CONV_TRANSPOSE1D_FP16_POLYPHASE_TRANSFORMATION;
      layout = ACE_VAE_CONV_TRANSPOSE1D_FP16_POLYPHASE_LAYOUT;
      const [outputChannels, kernel, inputChannels] = logicalShape;
      storageShape = [kernel! / 2, 2, inputChannels!, outputChannels!];
    } else if (sourceTensor.endsWith(".weight_v")) {
      const k1 = logicalShape[1] === 1;
      transformation = k1
        ? ACE_VAE_K1_FP16_TILE_TRANSFORMATION
        : ACE_VAE_CONV1D_FP16_TRANSFORMATION;
      layout = k1 ? ACE_VAE_K1_FP16_TILE_LAYOUT : ACE_VAE_CONV1D_FP16_LAYOUT;
      if (k1) {
        const [outputChannels, , inputChannels] = logicalShape;
        storageShape = [
          outputChannels! / 128, inputChannels! / 32, 32, 128,
        ];
      }
    } else if (sourceTensor.endsWith(".bias")) {
      transformation = ACE_VAE_BIAS_FP16_TRANSFORMATION;
      layout = "source-row-major";
    } else {
      transformation = ACE_VAE_CHANNEL_VECTOR_FP16_TRANSFORMATION;
      layout = ACE_VAE_CHANNEL_VECTOR_FP16_LAYOUT;
    }
    return { name, sourceTensor, logicalShape, storageShape, transformation, layout };
  }).sort((left, right) => left.name < right.name ? -1 : 1);
  expect(
    specs.reduce(
      (total, spec) =>
        total + spec.logicalShape.reduce((product, value) => product * value, 1),
      0,
    ),
  ).toBe(ACE_EXPERIMENTAL_VAE_PARAMETER_ELEMENTS);

  const tensors: Record<string, Record<string, unknown>> = Object.create(null) as
    Record<string, Record<string, unknown>>;
  tensors["constants.silence_latent"] = {
    shard: "constants/silence-latent-f32.bin",
    byteOffset: 0,
    byteLength: 3_840_000,
    dtype: "float32",
    logicalShape: [1, 64, 15_000],
    storageShape: [1, 64, 15_000],
    layout: "contiguous-nct-f32",
    phase: "constants",
    lifetime: "initial-latent",
    source: "ace-silence-latent:silence_latent/data/0",
    transformation: "validated-pytorch-zip-storage-extraction",
    logicalTensor: "constants.silence_latent",
    partAxis: 0,
    partStart: 0,
    partEnd: 1,
  };
  const shardCursors = [0, 0];
  for (const spec of specs) {
    const elements = spec.logicalShape.reduce(
      (product, value) => product * value,
      1,
    );
    const byteLength = elements * 2;
    let shardIndex = 0;
    let byteOffset = align256(shardCursors[shardIndex]!);
    if (byteOffset + byteLength > ACE_MAX_WEIGHT_SHARD_BYTES) {
      shardIndex = 1;
      byteOffset = align256(shardCursors[shardIndex]!);
    }
    if (byteOffset + byteLength > ACE_MAX_WEIGHT_SHARD_BYTES) {
      throw new Error("revision-7 VAE fixture exceeded two bounded shards");
    }
    shardCursors[shardIndex] = byteOffset + byteLength;
    tensors[spec.name] = {
      shard: `weights/vae/shared-0${shardIndex}.bin`,
      byteOffset,
      byteLength,
      dtype: "float16",
      logicalShape: spec.logicalShape,
      storageShape: spec.storageShape,
      layout: spec.layout,
      phase: "vae",
      lifetime: "vae",
      source: `vae-weights:${spec.sourceTensor}`,
      transformation: spec.transformation,
      logicalTensor: spec.name,
      partAxis: 0,
      partStart: 0,
      partEnd: spec.logicalShape[0],
    };
  }
  manifest.tensors = tensors;
  Object.assign(manifest.accounting as Record<string, unknown>, {
    sourceTensors: ACE_EXPERIMENTAL_VAE_LOGICAL_TENSOR_COUNT,
    directlyIncluded: ACE_EXPERIMENTAL_VAE_LOGICAL_TENSOR_COUNT,
    consumedByTransform: 0,
    excluded: 0,
    outputTensorsBeforeRowSharding: ACE_EXPERIMENTAL_VAE_LOGICAL_TENSOR_COUNT,
    constantTensors: 1,
    outputTensorsAfterRowSharding: ACE_EXPERIMENTAL_VAE_TENSOR_RECORD_COUNT + 1,
  });
  const files = manifest.files as Array<Record<string, unknown>>;
  files.unshift({
    name: "constants/silence-latent-f32.bin",
    byteLength: 3_840_000,
    sha256: "f".repeat(64),
    kind: "constant",
  });
  const firstWeight = files.at(-1)!;
  firstWeight.name = "weights/vae/shared-00.bin";
  firstWeight.byteLength = align256(shardCursors[0]!);
  firstWeight.sha256 = "d".repeat(64);
  files.push({
    name: "weights/vae/shared-01.bin",
    byteLength: align256(shardCursors[1]!),
    sha256: "e".repeat(64),
    kind: "weights",
  });
  return manifest;
}

function experimentalDitDenseK4Manifest(): Record<string, unknown> {
  const manifest = syntheticAceManifest();
  manifest.profile = "fp16-dit-dense-experimental";
  (manifest.provenance as Record<string, unknown>).converterRevision =
    ACE_EXPERIMENTAL_DIT_DENSE_PACKAGE_CONVERTER_REVISION;

  const source = (manifest.source as Array<Record<string, unknown>>)[0]!;
  source.tensorCount = ACE_EXPERIMENTAL_DIT_DENSE_LOGICAL_TENSOR_COUNT;
  source.parameterCount = ACE_EXPERIMENTAL_DIT_DENSE_PARAMETER_ELEMENTS;
  (manifest.source as Array<Record<string, unknown>>).push({
    key: "ace-silence-latent",
    component: "initial latent",
    repository: source.repository,
    revision: source.revision,
    path: "acestep-v15-turbo/silence_latent.pt",
    byteLength: 4,
    sha256: "f".repeat(64),
  });

  const shapes: Readonly<Record<string, readonly number[]>> = {
    "scale_shift_table": [1, 6, 2_048],
    "self_attn_norm.weight": [2_048],
    "self_attn.q_proj.weight": [2_048, 2_048],
    "self_attn.k_proj.weight": [1_024, 2_048],
    "self_attn.v_proj.weight": [1_024, 2_048],
    "self_attn.q_norm.weight": [128],
    "self_attn.k_norm.weight": [128],
    "self_attn.o_proj.weight": [2_048, 2_048],
    "cross_attn_norm.weight": [2_048],
    "cross_attn.q_proj.weight": [2_048, 2_048],
    "cross_attn.k_proj.weight": [1_024, 2_048],
    "cross_attn.v_proj.weight": [1_024, 2_048],
    "cross_attn.q_norm.weight": [128],
    "cross_attn.k_norm.weight": [128],
    "cross_attn.o_proj.weight": [2_048, 2_048],
    "mlp_norm.weight": [2_048],
    "mlp.gate_proj.weight": [6_144, 2_048],
    "mlp.up_proj.weight": [6_144, 2_048],
    "mlp.down_proj.weight": [2_048, 6_144],
  };
  const denseSuffixes = new Set([
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "cross_attn.q_proj.weight",
    "cross_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
  ]);
  const crossCacheSuffixes = new Set([
    "cross_attn.k_proj.weight",
    "cross_attn.v_proj.weight",
  ]);
  const tensors: Record<string, Record<string, unknown>> = Object.create(null) as Record<
    string,
    Record<string, unknown>
  >;
  const shardCursors = new Map<string, number>();
  for (let layer = 0; layer < 24; layer += 1) {
    for (const [suffix, logicalShape] of Object.entries(shapes)) {
      const elements = logicalShape.reduce((product, extent) => product * extent, 1);
      const byteLength = elements * 2;
      let part = 0;
      let shard = `weights/dit/layer-${String(layer).padStart(2, "0")}-00.bin`;
      let byteOffset = align256(shardCursors.get(shard) ?? 0);
      if (byteOffset + byteLength > ACE_MAX_WEIGHT_SHARD_BYTES) {
        part = 1;
        shard = `weights/dit/layer-${String(layer).padStart(2, "0")}-01.bin`;
        byteOffset = align256(shardCursors.get(shard) ?? 0);
      }
      if (part > 1 || byteOffset + byteLength > ACE_MAX_WEIGHT_SHARD_BYTES) {
        throw new Error(`experimental DiT layer ${layer} exceeded two shards`);
      }
      shardCursors.set(shard, byteOffset + byteLength);

      const name = `ace.decoder.layers.${layer}.${suffix}`;
      const dense = denseSuffixes.has(suffix);
      const crossCache = crossCacheSuffixes.has(suffix);
      const columns = logicalShape[0]!;
      const inner = logicalShape[1];
      tensors[name] = {
        shard,
        byteOffset,
        byteLength,
        dtype: dense ? "float16" : "uint32-bf16-pairs",
        logicalShape: [...logicalShape],
        storageShape: dense
          ? [columns / 128, inner! / 4, 4, 32, 4]
          : [elements / 2],
        layout: dense
          ? ACE_DIT_DENSE_K4_FP16_LAYOUT
          : crossCache
            ? ACE_DIT_GEMM_TILE_LAYOUT
            : "source-row-major-bf16-pairs-lsb-u32",
        phase: "dit",
        lifetime: "dit",
        source: `ace-turbo-weights:${name.slice("ace.".length)}`,
        transformation: dense
          ? ACE_DIT_DENSE_K4_FP16_TRANSFORMATION
          : crossCache
            ? ACE_DIT_GEMM_PACKED_BF16_TRANSFORMATION
            : "preserve-bf16-bits-pack-u32-pairs",
        logicalTensor: name,
        partAxis: 0,
        partStart: 0,
        partEnd: columns,
      };
    }
  }
  tensors["constants.silence_latent"] = {
    shard: "constants/silence-latent-f32.bin",
    byteOffset: 0,
    byteLength: 3_840_000,
    dtype: "float32",
    logicalShape: [1, 64, 15_000],
    storageShape: [1, 64, 15_000],
    layout: "contiguous-nct-f32",
    phase: "constants",
    lifetime: "initial-latent",
    source: "ace-silence-latent:silence_latent/data/0",
    transformation: "validated-pytorch-zip-storage-extraction",
    logicalTensor: "constants.silence_latent",
    partAxis: 0,
    partStart: 0,
    partEnd: 1,
  };
  manifest.tensors = Object.fromEntries(
    Object.entries(tensors).sort(([left], [right]) => left < right ? -1 : 1),
  );

  Object.assign(manifest.accounting as Record<string, unknown>, {
    sourceTensors: ACE_EXPERIMENTAL_DIT_DENSE_LOGICAL_TENSOR_COUNT,
    directlyIncluded: ACE_EXPERIMENTAL_DIT_DENSE_LOGICAL_TENSOR_COUNT,
    consumedByTransform: 0,
    excluded: 0,
    outputTensorsBeforeRowSharding: ACE_EXPERIMENTAL_DIT_DENSE_LOGICAL_TENSOR_COUNT,
    constantTensors: 1,
    outputTensorsAfterRowSharding:
      ACE_EXPERIMENTAL_DIT_DENSE_LOGICAL_TENSOR_COUNT + 1,
  });

  const files = (manifest.files as Array<Record<string, unknown>>).filter(
    (file) => file.kind !== "weights",
  );
  files.push({
    name: "constants/silence-latent-f32.bin",
    byteLength: 3_840_000,
    sha256: "f".repeat(64),
    kind: "constant",
  });
  let shardOrdinal = 0;
  for (const [name, cursor] of shardCursors) {
    shardOrdinal += 1;
    files.push({
      name,
      byteLength: align256(cursor),
      sha256: shardOrdinal.toString(16).padStart(64, "0"),
      kind: "weights",
    });
  }
  manifest.files = files.sort((left, right) =>
    String(left.name) < String(right.name) ? -1 : 1
  );
  return manifest;
}

function experimentalDitDenseTensor(
  manifest: Record<string, unknown>,
): Record<string, unknown> {
  return (manifest.tensors as Record<string, Record<string, unknown>>)[
    "ace.decoder.layers.0.self_attn.q_proj.weight"
  ]!;
}

function align256(value: number): number {
  return Math.ceil(value / 256) * 256;
}
