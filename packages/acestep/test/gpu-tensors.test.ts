import { describe, expect, it, vi } from "vitest";

import { AceGpuTensorPhase } from "../src/model/gpu-tensors.js";
import {
  parseAcePackageManifest,
  type AcePackageManifest,
} from "../src/model/manifest.js";
import { syntheticAceManifest } from "./model-fixtures.js";

describe("phase-resident GPU tensor store", () => {
  it("uploads each selected shard once and returns exact tensor spans", async () => {
    const manifest = fixtureManifest();
    const source = new File([new Uint8Array(256)], "shared-00.bin");
    const buffer = fakeBuffer();
    const upload = vi.fn(async (_device, record, file, options) => {
      options.onProgress?.({
        file: record.name,
        uploadedBytes: file.size,
        totalBytes: file.size,
      });
      return buffer.value;
    });
    const progress: number[] = [];
    const phase = await AceGpuTensorPhase.load(
      {} as GPUDevice,
      manifest,
      new Map([["weights/dit/shared-00.bin", source]]),
      ["dit"],
      {
        onProgress: (event) => progress.push(event.loadedPhaseBytes),
        upload,
      },
    );

    expect(upload).toHaveBeenCalledTimes(1);
    expect(phase.residentBytes).toBe(256);
    expect(phase.packageManifest).toBe(manifest);
    expect(progress).toEqual([256]);
    const tensor = phase.logicalTensor("ace.decoder.weight");
    expect(tensor.logicalShape).toEqual([2]);
    expect(tensor.parts).toHaveLength(1);
    expect(tensor.parts[0]!.binding).toEqual({
      buffer: buffer.value,
      offset: 0,
      size: 4,
    });
    expect(phase.binding("ace.decoder.weight")).toEqual(
      tensor.parts[0]!.binding,
    );

    phase.destroy();
    phase.destroy();
    expect(buffer.destroy).toHaveBeenCalledTimes(1);
    expect(() => phase.binding("ace.decoder.weight")).toThrow(/destroyed/);
  });

  it("destroys every earlier upload when a later shard fails", async () => {
    const manifest = twoShardManifest();
    const first = fakeBuffer();
    const upload = vi.fn(async (_device, record) => {
      if (record.name.endsWith("01.bin")) throw new Error("allocation failed");
      return first.value;
    });
    const files = new Map([
      ["weights/dit/shared-00.bin", new File([new Uint8Array(256)], "00.bin")],
      ["weights/dit/shared-01.bin", new File([new Uint8Array(256)], "01.bin")],
    ]);
    await expect(
      AceGpuTensorPhase.load(
        {} as GPUDevice,
        manifest,
        files,
        ["dit"],
        { upload },
      ),
    ).rejects.toThrow("allocation failed");
    expect(first.destroy).toHaveBeenCalledTimes(1);
  });

  it("fails closed for missing files, repeated phases, and implicit sharding", async () => {
    const manifest = twoPartLogicalManifest();
    const buffers = [fakeBuffer(), fakeBuffer()];
    let next = 0;
    const files = new Map([
      ["weights/dit/shared-00.bin", new File([new Uint8Array(256)], "00.bin")],
      ["weights/dit/shared-01.bin", new File([new Uint8Array(256)], "01.bin")],
    ]);
    const phase = await AceGpuTensorPhase.load(
      {} as GPUDevice,
      manifest,
      files,
      ["dit"],
      { upload: async () => buffers[next++]!.value },
    );
    expect(phase.logicalTensor("ace.decoder.weight").parts).toHaveLength(2);
    expect(() => phase.binding("ace.decoder.weight")).toThrow(/explicit row shards/);
    phase.destroy();

    await expect(
      AceGpuTensorPhase.load(
        {} as GPUDevice,
        fixtureManifest(),
        new Map(),
        ["dit"],
        { upload: async () => fakeBuffer().value },
      ),
    ).rejects.toThrow(/missing/);
    await expect(
      AceGpuTensorPhase.load(
        {} as GPUDevice,
        fixtureManifest(),
        files,
        ["dit", "dit"],
        { upload: async () => fakeBuffer().value },
      ),
    ).rejects.toThrow(/repeats/);
  });

  it("does not expose another lifetime that happens to share a physical shard", async () => {
    const raw = syntheticAceManifest();
    const tensors = raw.tensors as Record<string, Record<string, unknown>>;
    const original = tensors["ace.decoder.weight"]!;
    tensors["ace.conditioner.weight"] = {
      ...original,
      source: "ace-turbo-weights:conditioner.weight",
      logicalTensor: "ace.conditioner.weight",
      phase: "conditioner",
      lifetime: "conditioner",
      byteOffset: 256,
    };
    raw.tensors = Object.fromEntries(
      Object.entries(tensors).sort(([left], [right]) => left.localeCompare(right)),
    );
    const accounting = raw.accounting as Record<string, number>;
    accounting.sourceTensors = 2;
    accounting.directlyIncluded = 2;
    accounting.outputTensorsBeforeRowSharding = 2;
    accounting.outputTensorsAfterRowSharding = 2;
    const source = (raw.source as Array<Record<string, number>>)[0]!;
    source.tensorCount = 2;
    source.parameterCount = 4;
    const sharedFile = (raw.files as Array<Record<string, unknown>>).find(
      (file) => file.name === "weights/dit/shared-00.bin",
    )!;
    sharedFile.byteLength = 512;
    const manifest = parseAcePackageManifest(raw, "reference");
    const resident = fakeBuffer();
    const phase = await AceGpuTensorPhase.load(
      {} as GPUDevice,
      manifest,
      new Map([
        [
          "weights/dit/shared-00.bin",
          new File([new Uint8Array(512)], "shared-00.bin"),
        ],
      ]),
      ["dit"],
      { upload: async () => resident.value },
    );

    expect(() => phase.binding("ace.conditioner.weight")).toThrow(
      /belongs to phase conditioner/,
    );
    phase.destroy();
  });
});

function fixtureManifest(): AcePackageManifest {
  return parseAcePackageManifest(syntheticAceManifest(), "reference");
}

function twoShardManifest(): AcePackageManifest {
  const raw = syntheticAceManifest();
  const files = raw.files as Array<Record<string, unknown>>;
  files.push({
    name: "weights/dit/shared-01.bin",
    byteLength: 256,
    sha256: "c".repeat(64),
    kind: "weights",
  });
  const tensors = raw.tensors as Record<string, Record<string, unknown>>;
  tensors["ace.zdecoder.bias"] = {
    ...tensors["ace.decoder.weight"],
    shard: "weights/dit/shared-01.bin",
    source: "ace-turbo-weights:zdecoder.bias",
    logicalTensor: "ace.zdecoder.bias",
  };
  const accounting = raw.accounting as Record<string, number>;
  accounting.sourceTensors = 2;
  accounting.directlyIncluded = 2;
  accounting.outputTensorsBeforeRowSharding = 2;
  accounting.outputTensorsAfterRowSharding = 2;
  const source = (raw.source as Array<Record<string, number>>)[0]!;
  source.tensorCount = 2;
  source.parameterCount = 4;
  return parseAcePackageManifest(raw, "reference");
}

function twoPartLogicalManifest(): AcePackageManifest {
  const raw = syntheticAceManifest();
  const files = raw.files as Array<Record<string, unknown>>;
  files.push({
    name: "weights/dit/shared-01.bin",
    byteLength: 256,
    sha256: "c".repeat(64),
    kind: "weights",
  });
  const tensors = raw.tensors as Record<string, Record<string, unknown>>;
  const first = tensors["ace.decoder.weight"]!;
  delete tensors["ace.decoder.weight"];
  first.logicalShape = [4];
  first.partEnd = 2;
  first.layout = "row-shard-axis0-bf16-pairs-lsb-u32";
  tensors["ace.decoder.weight.rows-000000-000002"] = first;
  tensors["ace.decoder.weight.rows-000002-000004"] = {
    ...first,
    shard: "weights/dit/shared-01.bin",
    source: "ace-turbo-weights:decoder.weight",
    partStart: 2,
    partEnd: 4,
  };
  const accounting = raw.accounting as Record<string, number>;
  accounting.outputTensorsAfterRowSharding = 2;
  const source = (raw.source as Array<Record<string, number>>)[0]!;
  source.parameterCount = 4;
  return parseAcePackageManifest(raw, "reference");
}

function fakeBuffer(): {
  readonly value: GPUBuffer;
  readonly destroy: ReturnType<typeof vi.fn>;
} {
  const destroy = vi.fn();
  return { value: { destroy } as unknown as GPUBuffer, destroy };
}
