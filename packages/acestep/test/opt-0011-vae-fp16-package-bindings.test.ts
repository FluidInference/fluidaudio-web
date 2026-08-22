import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";

import { describe, expect, expectTypeOf, it } from "vitest";

import type {
  AceGpuLogicalTensor,
  AceGpuTensorPhase,
} from "../src/model/gpu-tensors.js";
import {
  ACE_EXPERIMENTAL_VAE_PARAMETER_BYTES,
  ACE_PACKAGE_ALIGNMENT_BYTES,
  ACE_PACKAGE_FORMAT,
  ACE_PORTABLE_STORAGE_BINDING_BYTES,
  ACE_VAE_BIAS_FP16_TRANSFORMATION,
  ACE_VAE_CHANNEL_VECTOR_FP16_LAYOUT,
  ACE_VAE_CHANNEL_VECTOR_FP16_TRANSFORMATION,
  ACE_VAE_CONV1D_FP16_LAYOUT,
  ACE_VAE_CONV1D_FP16_TRANSFORMATION,
  ACE_VAE_CONV_TRANSPOSE1D_FP16_POLYPHASE_LAYOUT,
  ACE_VAE_CONV_TRANSPOSE1D_FP16_POLYPHASE_TRANSFORMATION,
  ACE_VAE_CONV_TRANSPOSE1D_K4_FP16_LAYOUT,
  ACE_VAE_CONV_TRANSPOSE1D_K4_FP16_TRANSFORMATION,
  ACE_VAE_K1_FP16_TILE_LAYOUT,
  ACE_VAE_K1_FP16_TILE_TRANSFORMATION,
  ACE_VAE_K7_ROW_REUSE_FP16_LAYOUT,
  ACE_VAE_K7_ROW_REUSE_FP16_TRANSFORMATION,
  ACE_VAE_REVISION7_K7_ROW_REUSE_CONTRACTS,
  ACE_VAE_REVISION7_TRANSPOSE_K4_CONTRACTS,
  parseAcePackageManifest,
  type AcePackageManifest,
  type AcePackageTensorRecord,
  type AceTensorLayout,
  type AceTensorTransformation,
} from "../src/model/manifest.js";
import {
  ACE_OPT_0054_VAE_REVISION7_MANIFEST_BYTES,
  ACE_OPT_0054_VAE_REVISION7_MANIFEST_SHA256,
} from "../src/model/package.js";
import {
  planAceVaeDecoder,
  type AceVaeDecoderOperation,
} from "../src/webgpu/vae-decoder.js";
import {
  ACE_OPT_0011_VAE_FP16_WEIGHT_FILES,
  resolveAceOpt0011Fp16VaePackageBindings,
  resolveAceOpt0054Fp16VaePackageBindings,
} from "../src/webgpu/vae-fp16-package.js";
import {
  ACE_OPT_0028_VAE_FP16_MANIFEST_BYTES,
  ACE_OPT_0028_VAE_FP16_MANIFEST_SHA256,
  type AceVaeAuthenticatedPackageIdentity,
} from "../src/webgpu/vae-fp16-profile.js";

const MANIFEST_URL = new URL(
  "../model/files-fp16-vae-experimental/manifest.json",
  import.meta.url,
);
const HAS_LOCAL_REVISION_6_MANIFEST = localManifestConverterRevision() === 6;
const MANIFEST = createFixtureManifest();
const LOADED: AceVaeAuthenticatedPackageIdentity = Object.freeze({
  manifest: MANIFEST,
  manifestSha256: ACE_OPT_0028_VAE_FP16_MANIFEST_SHA256,
  manifestByteLength: ACE_OPT_0028_VAE_FP16_MANIFEST_BYTES,
});
const REVISION7_MANIFEST = createRevision7FixtureManifest(MANIFEST);
const REVISION7_LOADED: AceVaeAuthenticatedPackageIdentity = Object.freeze({
  manifest: REVISION7_MANIFEST,
  manifestSha256: ACE_OPT_0054_VAE_REVISION7_MANIFEST_SHA256,
  manifestByteLength: ACE_OPT_0054_VAE_REVISION7_MANIFEST_BYTES,
});

describe("OPT-0011 typed experimental FP16 VAE package bindings", () => {
  it("resolves the exact 145 tensors into all 88 operation roles", () => {
    const phase = authenticatedFixturePhase(fakeVaePhase(MANIFEST));
    const resolved = resolveAceOpt0011Fp16VaePackageBindings(
      planAceVaeDecoder(256),
      LOADED,
      phase,
    );
    expect(resolved).toMatchObject({
      manifestSha256: ACE_OPT_0028_VAE_FP16_MANIFEST_SHA256,
      manifestByteLength: ACE_OPT_0028_VAE_FP16_MANIFEST_BYTES,
      residentWeightBytes: ACE_EXPERIMENTAL_VAE_PARAMETER_BYTES,
      weightFiles: ACE_OPT_0011_VAE_FP16_WEIGHT_FILES,
    });
    expect(Object.keys(resolved.tensors)).toHaveLength(145);
    expect(resolved.operations).toHaveLength(88);
    expect(countBy(resolved.operations, (operation) => operation.kind)).toEqual({
      conv1d: 32,
      snake: 36,
      "conv-transpose1d": 5,
      add: 15,
    });
    expect(resolved.operations[0]).toMatchObject({
      operationIndex: 0,
      label: "conv1",
      kind: "conv1d",
      weight: {
        logicalTensor: "vae.decoder.conv1.weight",
        logicalShape: [2048, 7, 64],
        physicalTensor: "vae.decoder.conv1.weight",
        record: {
          dtype: "float16",
          layout: "conv1d-output-kernel-input-f16-v1",
          transformation:
            "weightnorm-fused-fp32-pairwise-oik-to-oki-ieee-fp16-v1",
        },
      },
      bias: {
        logicalTensor: "vae.decoder.conv1.bias",
        logicalShape: [2048],
        record: {
          dtype: "float16",
          layout: "source-row-major",
          transformation: "bf16-to-fp32-to-ieee-fp16-v1",
        },
      },
    });
    expect(resolved.operations.find((operation) =>
      operation.kind === "conv-transpose1d"
    )).toMatchObject({
      kind: "conv-transpose1d",
      weight: {
        logicalShape: [1024, 20, 2048],
        record: {
          layout: "conv-transpose1d-phase-tap-input-output-f16-v1",
          transformation:
            "weightnorm-fused-fp32-pairwise-iok-to-phase-tap-input-output-ieee-fp16-v1",
        },
      },
    });
    expect(resolved.operations.find((operation) =>
      operation.kind === "conv1d" && operation.weight.logicalShape[1] === 1
    )).toMatchObject({
      kind: "conv1d",
      weight: {
        logicalShape: [1024, 1, 1024],
        record: {
          storageShape: [8, 32, 32, 128],
          layout: "conv1d-k1-cout128-cin32-tile-major-f16-v1",
          transformation:
            "weightnorm-fused-fp32-pairwise-oik-to-k1-cout128-cin32-tile-major-ieee-fp16-v1",
        },
      },
    });
    expect(resolved.operations.at(-1)).toMatchObject({
      operationIndex: 87,
      label: "conv2",
      kind: "conv1d",
      weight: { logicalShape: [2, 7, 128] },
    });
    expect(resolved.operations.at(-1)).not.toHaveProperty("bias");
    for (const tensor of Object.values(resolved.tensors)) {
      expect(tensor.record).toBe(MANIFEST.tensors[tensor.logicalTensor]);
      expect(tensor.binding.offset).toBe(tensor.record.byteOffset);
      expect(tensor.binding.size).toBe(tensor.record.byteLength);
    }
  });

  it("pins the complete unsharded seven-file resident inventory", () => {
    const records = Object.entries(MANIFEST.tensors)
      .filter(([, record]) => record.phase === "vae");
    expect(records).toHaveLength(145);
    expect(new Set(records.map(([, record]) => record.logicalTensor)).size)
      .toBe(145);
    expect(new Set(records.map(([, record]) => record.shard))).toEqual(
      new Set(ACE_OPT_0011_VAE_FP16_WEIGHT_FILES),
    );
    expect(records.reduce((total, [, record]) =>
      total + record.byteLength, 0)).toBe(168_791_552);
    expect(records.every(([physicalName, record]) =>
      physicalName === record.logicalTensor &&
      record.dtype === "float16" &&
      record.phase === "vae" &&
      record.lifetime === "vae" &&
      record.partAxis === 0 &&
      record.partStart === 0 &&
      record.partEnd === record.logicalShape[0]
    )).toBe(true);
    expect(records.filter(([, record]) =>
      record.transformation === ACE_VAE_K1_FP16_TILE_TRANSFORMATION
    )).toHaveLength(15);
    expect(records.filter(([, record]) =>
      record.transformation ===
        ACE_VAE_CONV_TRANSPOSE1D_FP16_POLYPHASE_TRANSFORMATION
    )).toHaveLength(5);
  });

  it.skipIf(!HAS_LOCAL_REVISION_6_MANIFEST)(
    "resolves the locally generated authenticated revision-6 manifest",
    () => {
      const bytes = readFileSync(MANIFEST_URL);
      const manifest = parseAcePackageManifest(
        JSON.parse(bytes.toString("utf8")) as unknown,
        "fp16-vae-experimental",
      );
      const loaded = Object.freeze({
        manifest,
        manifestSha256: createHash("sha256").update(bytes).digest("hex"),
        manifestByteLength: bytes.byteLength,
      });
      const resolved = resolveAceOpt0011Fp16VaePackageBindings(
        planAceVaeDecoder(256),
        loaded,
        authenticatedFixturePhase(fakeVaePhase(manifest)),
      );
      expect(Object.keys(resolved.tensors)).toHaveLength(145);
      expect(resolved.operations).toHaveLength(88);
      expect(resolved.weightFiles).toEqual(ACE_OPT_0011_VAE_FP16_WEIGHT_FILES);
    },
  );

  it("rejects forged dtype, layout, shape, transformation, and binding spans", () => {
    const target = "vae.decoder.conv1.weight";
    const mutations: readonly Partial<AcePackageTensorRecord>[] = [
      { dtype: "float32" },
      { layout: "source-row-major" },
      { logicalShape: [2048, 1, 64], storageShape: [2048, 1, 64] },
      { transformation: "bf16-to-ieee-fp16" },
      { partEnd: 2047 },
    ];
    for (const mutation of mutations) {
      const forged = forgeLoadedTensor(LOADED, target, mutation);
      expect(
        () => resolveAceOpt0011Fp16VaePackageBindings(
          planAceVaeDecoder(256),
          forged,
          authenticatedFixturePhase(fakeVaePhase(forged.manifest)),
        ),
        JSON.stringify(mutation),
      ).toThrow(/OPT-0011/);
    }

    const validPhase = fakeVaePhase(MANIFEST);
    expect(() => resolveAceOpt0011Fp16VaePackageBindings(
      planAceVaeDecoder(256),
      LOADED,
      authenticatedFixturePhase(
        replacingLogicalTensor(validPhase, target, (logical) => ({
          ...logical,
          parts: [Object.freeze({
            ...logical.parts[0]!,
            binding: Object.freeze({
              ...logical.parts[0]!.binding,
              size: logical.parts[0]!.tensor.byteLength - 2,
            }),
          })],
        })),
      ),
    )).toThrow(/authenticated byte span/);
  });

  it("rejects non-exclusive phases, another manifest object, and wrong residency", () => {
    const phase = fakeVaePhase(MANIFEST);
    expect(() => resolveAceOpt0011Fp16VaePackageBindings(
      planAceVaeDecoder(256),
      LOADED,
      authenticatedFixturePhase({
        ...phase,
        phases: ["vae", "constants"],
      }),
    )).toThrow(/exclusive VAE phase/);
    expect(() => resolveAceOpt0011Fp16VaePackageBindings(
      planAceVaeDecoder(256),
      LOADED,
      authenticatedFixturePhase({
        ...phase,
        packageManifest: { ...MANIFEST },
      }),
    )).toThrow(/authenticated manifest/);
    expect(() => resolveAceOpt0011Fp16VaePackageBindings(
      planAceVaeDecoder(256),
      LOADED,
      authenticatedFixturePhase({
        ...phase,
        residentBytes: phase.residentBytes - 2,
      }),
    )).toThrow(/145 unsharded tensors/);
    expect(() => resolveAceOpt0011Fp16VaePackageBindings(
      planAceVaeDecoder(512),
      LOADED,
      authenticatedFixturePhase(phase),
    )).toThrow(/exact batch-1 256-frame decoder graph/);
  });

  it("requires the nominal authenticated GPU tensor phase", () => {
    type ResolverPhase = Parameters<
      typeof resolveAceOpt0011Fp16VaePackageBindings
    >[2];
    expectTypeOf<ResolverPhase>().toEqualTypeOf<AceGpuTensorPhase>();
    expectTypeOf(fakeVaePhase(MANIFEST)).not.toMatchTypeOf<AceGpuTensorPhase>();
  });

  it("resolves the exhaustive revision-7 replace-not-duplicate layouts", () => {
    const resolved = resolveAceOpt0054Fp16VaePackageBindings(
      planAceVaeDecoder(256),
      REVISION7_LOADED,
      authenticatedFixturePhase(fakeVaePhase(REVISION7_MANIFEST)),
    );
    const records = Object.values(resolved.tensors).map(({ record }) => record);
    expect(records).toHaveLength(145);
    expect(records.filter(({ layout }) =>
      layout === ACE_VAE_K7_ROW_REUSE_FP16_LAYOUT
    )).toHaveLength(12);
    expect(records.filter(({ layout }) =>
      layout === ACE_VAE_CONV_TRANSPOSE1D_K4_FP16_LAYOUT
    )).toHaveLength(4);
    expect(records.filter(({ layout }) =>
      layout === ACE_VAE_CONV_TRANSPOSE1D_FP16_POLYPHASE_LAYOUT
    )).toHaveLength(1);
    expect(records.filter(({ layout }) =>
      layout === ACE_VAE_K1_FP16_TILE_LAYOUT
    )).toHaveLength(15);
    expect(records.reduce((sum, { byteLength }) => sum + byteLength, 0)).toBe(
      ACE_EXPERIMENTAL_VAE_PARAMETER_BYTES,
    );

    for (const contract of ACE_VAE_REVISION7_K7_ROW_REUSE_CONTRACTS) {
      expect(resolved.tensors[contract.tensor]?.record).toMatchObject({
        layout: ACE_VAE_K7_ROW_REUSE_FP16_LAYOUT,
        transformation: ACE_VAE_K7_ROW_REUSE_FP16_TRANSFORMATION,
        storageShape: [
          7,
          contract.channels / 4,
          contract.channels / 64,
          32,
          2,
          4,
        ],
      });
    }
    for (const contract of ACE_VAE_REVISION7_TRANSPOSE_K4_CONTRACTS) {
      const outputsPerLane = contract.reuseAxis === "channel" ? 8 : 4;
      const record = resolved.tensors[contract.tensor]?.record;
      expect(record).toMatchObject({
        layout: ACE_VAE_CONV_TRANSPOSE1D_K4_FP16_LAYOUT,
        transformation: ACE_VAE_CONV_TRANSPOSE1D_K4_FP16_TRANSFORMATION,
        storageShape: [
          record!.logicalShape[1]! / 2,
          2,
          contract.inputChannels / 4,
          contract.outputChannels / (outputsPerLane * 32),
          32,
          outputsPerLane,
          4,
        ],
      });
    }
  });

  it("keeps revision-6 and revision-7 package resolvers mutually exclusive", () => {
    expect(() => resolveAceOpt0011Fp16VaePackageBindings(
      planAceVaeDecoder(256),
      REVISION7_LOADED,
      authenticatedFixturePhase(fakeVaePhase(REVISION7_MANIFEST)),
    )).toThrow(/exact authenticated package identity/);
    expect(() => resolveAceOpt0054Fp16VaePackageBindings(
      planAceVaeDecoder(256),
      LOADED,
      authenticatedFixturePhase(fakeVaePhase(MANIFEST)),
    )).toThrow(/exact authenticated package identity/);

    const target = ACE_VAE_REVISION7_K7_ROW_REUSE_CONTRACTS[0]!.tensor;
    const forged = forgeLoadedTensor(REVISION7_LOADED, target, {
      layout: ACE_VAE_CONV1D_FP16_LAYOUT,
      transformation: ACE_VAE_CONV1D_FP16_TRANSFORMATION,
      storageShape: REVISION7_MANIFEST.tensors[target]!.logicalShape,
    });
    expect(() => resolveAceOpt0054Fp16VaePackageBindings(
      planAceVaeDecoder(256),
      forged,
      authenticatedFixturePhase(fakeVaePhase(forged.manifest)),
    )).toThrow(/FP16 package contract/);
  });
});

interface FixtureTensorContract {
  readonly logicalTensor: string;
  readonly shape: readonly number[];
  readonly layout: AceTensorLayout;
  readonly transformation: AceTensorTransformation;
  readonly source: string;
}

function createFixtureManifest(): AcePackageManifest {
  const plan = planAceVaeDecoder(256);
  const contracts = plan.operations.flatMap(fixtureContractsForOperation);
  if (contracts.length !== 145) {
    throw new Error(`fixture expected 145 tensors, received ${contracts.length}`);
  }
  const cursors = new Array<number>(
    ACE_OPT_0011_VAE_FP16_WEIGHT_FILES.length,
  ).fill(0);
  const tensors: Record<string, AcePackageTensorRecord> = {};
  for (let index = 0; index < contracts.length; index += 1) {
    const contract = contracts[index]!;
    if (tensors[contract.logicalTensor] !== undefined) {
      throw new Error(`duplicate fixture tensor ${contract.logicalTensor}`);
    }
    const shardIndex = index % ACE_OPT_0011_VAE_FP16_WEIGHT_FILES.length;
    const byteLength = contract.shape.reduce(
      (product, extent) => product * extent,
      1,
    ) * 2;
    const byteOffset = cursors[shardIndex]!;
    cursors[shardIndex] = byteOffset + byteLength;
    tensors[contract.logicalTensor] = Object.freeze({
      shard: ACE_OPT_0011_VAE_FP16_WEIGHT_FILES[shardIndex]!,
      byteOffset,
      byteLength,
      dtype: "float16",
      logicalShape: Object.freeze([...contract.shape]),
      storageShape: fixtureStorageShape(contract),
      layout: contract.layout,
      phase: "vae",
      lifetime: "vae",
      source: contract.source,
      transformation: contract.transformation,
      logicalTensor: contract.logicalTensor,
      partAxis: 0,
      partStart: 0,
      partEnd: contract.shape[0]!,
    });
  }
  const payloadBytes = Object.values(tensors).reduce(
    (total, record) => total + record.byteLength,
    0,
  );
  if (payloadBytes !== ACE_EXPERIMENTAL_VAE_PARAMETER_BYTES) {
    throw new Error(`fixture VAE bytes changed to ${payloadBytes}`);
  }
  return Object.freeze({
    format: ACE_PACKAGE_FORMAT,
    profile: "fp16-vae-experimental",
    alignment: ACE_PACKAGE_ALIGNMENT_BYTES,
    portableStorageBindingBytes: ACE_PORTABLE_STORAGE_BINDING_BYTES,
    source: Object.freeze([]),
    files: Object.freeze(ACE_OPT_0011_VAE_FP16_WEIGHT_FILES.map(
      (name, index) => Object.freeze({
        name,
        byteLength: cursors[index]!,
        sha256: (index + 1).toString(16).padStart(64, "0"),
        kind: "weights" as const,
      }),
    )),
    tensors: Object.freeze(tensors),
    accounting: Object.freeze({
      sourceTensors: 145,
      directlyIncluded: 145,
      consumedByTransform: 0,
      excluded: 0,
      outputTensorsBeforeRowSharding: 145,
      constantTensors: 0,
      outputTensorsAfterRowSharding: 145,
    }),
    licenses: Object.freeze([]),
    provenance: Object.freeze({
      converterRevision: 6,
      aceSnapshot: "19671f406d603126926c1b7e2adc169acbcade22",
      plannerSnapshot: "148d8ea0225bdab342ee1ae3a354275ccd60ca80",
      referenceRepository: "ACE-Step/ACE-Step",
      referenceCommit: "6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0",
      referenceLicenseGitBlob: "fixture",
      referenceLicenseSha256: "0".repeat(64),
      determinism: "OPT-0011 focused typed-binding fixture",
    }),
  });
}

function createRevision7FixtureManifest(
  revision6: AcePackageManifest,
): AcePackageManifest {
  const rowReuse = new Set<string>(
    ACE_VAE_REVISION7_K7_ROW_REUSE_CONTRACTS.map(({ tensor }) => tensor),
  );
  const transpose = new Map<string, "channel" | "row">(
    ACE_VAE_REVISION7_TRANSPOSE_K4_CONTRACTS.map(
      ({ tensor, reuseAxis }) => [tensor, reuseAxis] as const,
    ),
  );
  const tensors: Record<string, AcePackageTensorRecord> = {};
  for (const [name, record] of Object.entries(revision6.tensors)) {
    if (rowReuse.has(name)) {
      const [outputChannels, , inputChannels] = record.logicalShape;
      tensors[name] = Object.freeze({
        ...record,
        layout: ACE_VAE_K7_ROW_REUSE_FP16_LAYOUT,
        transformation: ACE_VAE_K7_ROW_REUSE_FP16_TRANSFORMATION,
        storageShape: Object.freeze([
          7,
          inputChannels! / 4,
          outputChannels! / 64,
          32,
          2,
          4,
        ]),
      });
      continue;
    }
    const reuseAxis = transpose.get(name);
    if (reuseAxis !== undefined) {
      const [outputChannels, kernel, inputChannels] = record.logicalShape;
      const outputsPerLane = reuseAxis === "channel" ? 8 : 4;
      tensors[name] = Object.freeze({
        ...record,
        layout: ACE_VAE_CONV_TRANSPOSE1D_K4_FP16_LAYOUT,
        transformation: ACE_VAE_CONV_TRANSPOSE1D_K4_FP16_TRANSFORMATION,
        storageShape: Object.freeze([
          kernel! / 2,
          2,
          inputChannels! / 4,
          outputChannels! / (outputsPerLane * 32),
          32,
          outputsPerLane,
          4,
        ]),
      });
      continue;
    }
    tensors[name] = record;
  }
  return Object.freeze({
    ...revision6,
    tensors: Object.freeze(tensors),
    provenance: Object.freeze({
      ...revision6.provenance,
      converterRevision: 7,
      determinism: "OPT-0054 focused revision-7 typed-binding fixture",
    }),
  });
}

function fixtureContractsForOperation(
  operation: AceVaeDecoderOperation,
): readonly FixtureTensorContract[] {
  switch (operation.kind) {
    case "conv1d": {
      const k1 = operation.shape.kernelSize === 1;
      return [
        fixtureTensorContract(
          operation.weight,
          [
            operation.shape.outputChannels,
            operation.shape.kernelSize,
            operation.shape.inputChannels,
          ],
          k1 ? ACE_VAE_K1_FP16_TILE_LAYOUT : ACE_VAE_CONV1D_FP16_LAYOUT,
          k1
            ? ACE_VAE_K1_FP16_TILE_TRANSFORMATION
            : ACE_VAE_CONV1D_FP16_TRANSFORMATION,
          true,
        ),
        ...(operation.bias === undefined
          ? []
          : [fixtureTensorContract(
              operation.bias,
              [operation.shape.outputChannels],
              "source-row-major",
              ACE_VAE_BIAS_FP16_TRANSFORMATION,
              false,
            )]),
      ];
    }
    case "conv-transpose1d":
      return [
        fixtureTensorContract(
          operation.weight,
          [
            operation.shape.outputChannels,
            operation.shape.kernelSize,
            operation.shape.inputChannels,
          ],
          ACE_VAE_CONV_TRANSPOSE1D_FP16_POLYPHASE_LAYOUT,
          ACE_VAE_CONV_TRANSPOSE1D_FP16_POLYPHASE_TRANSFORMATION,
          true,
        ),
        fixtureTensorContract(
          operation.bias,
          [operation.shape.outputChannels],
          "source-row-major",
          ACE_VAE_BIAS_FP16_TRANSFORMATION,
          false,
        ),
      ];
    case "snake":
      return [
        fixtureTensorContract(
          operation.alpha,
          [operation.shape.channels],
          ACE_VAE_CHANNEL_VECTOR_FP16_LAYOUT,
          ACE_VAE_CHANNEL_VECTOR_FP16_TRANSFORMATION,
          false,
        ),
        fixtureTensorContract(
          operation.beta,
          [operation.shape.channels],
          ACE_VAE_CHANNEL_VECTOR_FP16_LAYOUT,
          ACE_VAE_CHANNEL_VECTOR_FP16_TRANSFORMATION,
          false,
        ),
      ];
    case "add":
      return [];
  }
}

function fixtureTensorContract(
  logicalTensor: string,
  shape: readonly number[],
  layout: AceTensorLayout,
  transformation: AceTensorTransformation,
  weightNormalized: boolean,
): FixtureTensorContract {
  const sourceName = logicalTensor.slice("vae.".length);
  return Object.freeze({
    logicalTensor,
    shape: Object.freeze([...shape]),
    layout,
    transformation,
    source: `vae-weights:${weightNormalized
      ? sourceName.replace(/\.weight$/, ".weight_v")
      : sourceName}`,
  });
}

function fixtureStorageShape(
  contract: FixtureTensorContract,
): readonly number[] {
  if (contract.transformation === ACE_VAE_K1_FP16_TILE_TRANSFORMATION) {
    const [outputChannels, , inputChannels] = contract.shape;
    return Object.freeze([
      outputChannels! / 128,
      inputChannels! / 32,
      32,
      128,
    ]);
  }
  if (
    contract.transformation ===
      ACE_VAE_CONV_TRANSPOSE1D_FP16_POLYPHASE_TRANSFORMATION
  ) {
    const [outputChannels, kernel, inputChannels] = contract.shape;
    return Object.freeze([kernel! / 2, 2, inputChannels!, outputChannels!]);
  }
  return Object.freeze([...contract.shape]);
}

function localManifestConverterRevision(): number | undefined {
  if (!existsSync(MANIFEST_URL)) return undefined;
  try {
    const raw = JSON.parse(readFileSync(MANIFEST_URL, "utf8")) as {
      provenance?: { converterRevision?: unknown };
    };
    const revision = raw.provenance?.converterRevision;
    return typeof revision === "number" ? revision : undefined;
  } catch {
    return undefined;
  }
}

type FixtureVaePhase = Pick<
  AceGpuTensorPhase,
  "packageManifest" | "phases" | "residentBytes" | "logicalTensor"
>;

function fakeVaePhase(manifest: AcePackageManifest): FixtureVaePhase {
  const files = new Map(manifest.files.map((file) => [file.name, file]));
  const shards = new Set(Object.values(manifest.tensors)
    .filter((record) => record.phase === "vae")
    .map((record) => record.shard));
  const buffers = new Map<string, GPUBuffer>();
  let residentBytes = 0;
  for (const shard of shards) {
    const file = files.get(shard);
    if (file === undefined) throw new Error(`missing fixture shard ${shard}`);
    residentBytes += file.byteLength;
    buffers.set(shard, { size: file.byteLength } as GPUBuffer);
  }
  return {
    packageManifest: manifest,
    phases: Object.freeze(["vae"]),
    residentBytes,
    logicalTensor(logicalTensor: string): AceGpuLogicalTensor {
      const record = manifest.tensors[logicalTensor];
      if (record === undefined) throw new Error(`missing fixture ${logicalTensor}`);
      return Object.freeze({
        logicalTensor,
        logicalShape: record.logicalShape,
        parts: Object.freeze([Object.freeze({
          tensorName: logicalTensor,
          tensor: record,
          binding: Object.freeze({
            buffer: buffers.get(record.shard)!,
            offset: record.byteOffset,
            size: record.byteLength,
          }),
        })]),
      });
    },
  };
}

function forgeLoadedTensor(
  loaded: AceVaeAuthenticatedPackageIdentity,
  name: string,
  mutation: Partial<AcePackageTensorRecord>,
): AceVaeAuthenticatedPackageIdentity {
  const original = loaded.manifest.tensors[name]!;
  const manifest = {
    ...loaded.manifest,
    tensors: Object.freeze({
      ...loaded.manifest.tensors,
      [name]: Object.freeze({ ...original, ...mutation }),
    }),
  } as AcePackageManifest;
  return Object.freeze({ ...loaded, manifest });
}

function replacingLogicalTensor(
  phase: FixtureVaePhase,
  target: string,
  replace: (logical: AceGpuLogicalTensor) => AceGpuLogicalTensor,
): FixtureVaePhase {
  return {
    ...phase,
    logicalTensor(name: string): AceGpuLogicalTensor {
      const logical = phase.logicalTensor(name);
      return name === target ? replace(logical) : logical;
    },
  };
}

/** Tests alone assert the fixture as loader-authenticated after type checks. */
function authenticatedFixturePhase(
  phase: FixtureVaePhase,
): AceGpuTensorPhase {
  return phase as unknown as AceGpuTensorPhase;
}

function countBy<T>(
  values: readonly T[],
  key: (value: T) => string,
): Record<string, number> {
  const counts: Record<string, number> = {};
  for (const value of values) {
    const name = key(value);
    counts[name] = (counts[name] ?? 0) + 1;
  }
  return counts;
}
