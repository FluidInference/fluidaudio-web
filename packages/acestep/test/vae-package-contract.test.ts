import { existsSync, readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

import type {
  AcePackageManifest,
} from "../src/model/manifest.js";
import {
  ACE_OOBLECK_DECODER_CONFIG,
  createAceVaeLogicalTensorBindingsFromManifest,
  planAceVaeDecoder,
} from "../src/webgpu/vae-decoder.js";

const LOCAL_MANIFEST_URLS = Object.freeze({
  reference: new URL("../model/files-reference/manifest.json", import.meta.url),
  fp16: new URL("../model/files-fp16/manifest.json", import.meta.url),
});
const HAS_LOCAL_CANONICAL_MANIFESTS = Object.values(LOCAL_MANIFEST_URLS)
  .every((url) => existsSync(url));

describe.skipIf(!HAS_LOCAL_CANONICAL_MANIFESTS)(
  "ACE VAE generated-package resolution",
  () => {
    it.each(["reference", "fp16"] as const)(
      "resolves the exact decoder tensor set and all logical parts in %s",
      (profile) => {
        const manifest = JSON.parse(readFileSync(
          LOCAL_MANIFEST_URLS[profile],
          "utf8",
        )) as AcePackageManifest;
        const plan = planAceVaeDecoder(256);
        const vaeRecords = Object.entries(manifest.tensors)
          .filter(([, record]) => record.phase === "vae");
        const constantRecords = Object.entries(manifest.tensors)
          .filter(([, record]) => record.phase === "constants");
        const logicalNames = [...new Set(vaeRecords.map(([, record]) =>
          record.logicalTensor))].sort();
        expect([...plan.requiredTensorNames].sort()).toEqual(logicalNames);
        expect(logicalNames).toHaveLength(145);
        expect(vaeRecords).toHaveLength(146);
        expect(constantRecords).toEqual([
          [
            "constants.silence_latent",
            expect.objectContaining({
              logicalTensor: "constants.silence_latent",
              dtype: "float32",
              phase: "constants",
            }),
          ],
        ]);
        expect(manifest.accounting.constantTensors).toBe(1);

        for (const logicalName of plan.requiredTensorNames) {
          const records = vaeRecords
            .filter(([, record]) => record.logicalTensor === logicalName)
            .sort((left, right) => left[1].partStart - right[1].partStart);
          expect(records.length, `missing ${profile} ${logicalName}`).toBeGreaterThan(0);
          const operation = plan.operations.find((candidate) => {
            if (
              candidate.kind !== "conv1d" &&
              candidate.kind !== "conv-transpose1d"
            ) return false;
            return candidate.weight === logicalName;
          });
          const logicalRows = records[0]![1].logicalShape[0]!;
          if (
            operation?.kind === "conv1d" ||
            operation?.kind === "conv-transpose1d"
          ) {
            expect(logicalRows).toBe(operation.shape.outputChannels);
          }
          let cursor = 0;
          for (const [physicalName, record] of records) {
            expect(record.logicalTensor).toBe(logicalName);
            expect(record.dtype).toBe("float32");
            expect(record.partAxis).toBe(0);
            expect(record.partStart).toBe(cursor);
            expect(record.partEnd).toBeGreaterThan(record.partStart);
            if (records.length > 1) {
              expect(physicalName).toMatch(/\.rows-\d{6}-\d{6}$/);
            } else {
              expect(physicalName).toBe(logicalName);
            }
            cursor = record.partEnd;
          }
          expect(cursor).toBe(logicalRows);
        }

        const split = vaeRecords.filter(([, record]) =>
          record.logicalTensor === "vae.decoder.block.0.conv_t1.weight");
        expect(split.map(([, record]) => [record.partStart, record.partEnd]))
          .toEqual([[0, 614], [614, 1_024]]);
        expect(split.map(([, record]) => record.storageShape))
          .toEqual([[614, 20, 2_048], [410, 20, 2_048]]);

        const fake = { size: 120 * 1024 * 1024 } as GPUBuffer;
        const physicalBindings = vaeRecords.map(([physicalName]) => ({
          physicalName,
          binding: { buffer: fake },
        }));
        const resolved = createAceVaeLogicalTensorBindingsFromManifest(
          plan,
          manifest,
          physicalBindings,
        );
        expect(Object.keys(resolved).sort()).toEqual(logicalNames);
        expect(resolved["vae.decoder.block.0.conv_t1.weight"]?.map((part) =>
          [part.partStart, part.partEnd])).toEqual([[0, 614], [614, 1_024]]);
        expect(resolved["vae.decoder.conv1.bias"]).toHaveLength(1);
        expect(() => createAceVaeLogicalTensorBindingsFromManifest(
          plan,
          manifest,
          [...physicalBindings, {
            physicalName: "vae.decoder.unexpected",
            binding: { buffer: fake },
          }],
        )).toThrow(/unexpected names: vae\.decoder\.unexpected/);
      },
    );

    it("matches the pinned AutoencoderOobleck operation order and final bias contract", () => {
      const plan = planAceVaeDecoder(1, ACE_OOBLECK_DECODER_CONFIG);
      expect(plan.operations[0]).toMatchObject({
        kind: "conv1d",
        label: "conv1",
        weight: "vae.decoder.conv1.weight",
        bias: "vae.decoder.conv1.bias",
      });
      let cursor = 1;
      for (let block = 0; block < 5; block += 1) {
        expect(plan.operations[cursor++]).toMatchObject({
          kind: "snake",
          label: `block-${block}-snake1`,
        });
        expect(plan.operations[cursor++]).toMatchObject({
          kind: "conv-transpose1d",
          label: `block-${block}-conv-t1`,
        });
        for (let residual = 1; residual <= 3; residual += 1) {
          expect(plan.operations.slice(cursor, cursor + 5).map((operation) =>
            operation.kind)).toEqual([
            "snake", "conv1d", "snake", "conv1d", "add",
          ]);
          expect(plan.operations[cursor]).toMatchObject({
            label: `block-${block}-res-${residual}-snake1`,
          });
          cursor += 5;
        }
      }
      expect(plan.operations[cursor++]).toMatchObject({
        kind: "snake",
        label: "snake1",
      });
      expect(plan.operations[cursor++]).toEqual(expect.objectContaining({
        kind: "conv1d",
        label: "conv2",
        weight: "vae.decoder.conv2.weight",
      }));
      expect(plan.operations.at(-1)).not.toHaveProperty("bias");
      expect(cursor).toBe(plan.operations.length);
    });
  },
);
