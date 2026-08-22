import {
  ACE_MODEL_SNAPSHOT_REVISION,
  ACE_PACKAGE_ALIGNMENT_BYTES,
  ACE_PACKAGE_CONVERTER_REVISION,
  ACE_PACKAGE_FORMAT,
  ACE_PLANNER_SNAPSHOT_REVISION,
  ACE_PORTABLE_STORAGE_BINDING_BYTES,
  ACE_REFERENCE_SOURCE_REVISION,
} from "../src/model/manifest.js";

const SHA_A = "a".repeat(64);
const SHA_B = "b".repeat(64);
const SHA_C = "c".repeat(64);

export function syntheticAceManifest(): Record<string, unknown> {
  return {
    format: ACE_PACKAGE_FORMAT,
    profile: "reference",
    alignment: ACE_PACKAGE_ALIGNMENT_BYTES,
    portableStorageBindingBytes: ACE_PORTABLE_STORAGE_BINDING_BYTES,
    source: [
      {
        key: "ace-turbo-weights",
        component: "dit",
        repository: "ACE-Step/Ace-Step1.5",
        revision: ACE_MODEL_SNAPSHOT_REVISION,
        path: "acestep-v15-turbo/model.safetensors",
        byteLength: 128,
        sha256: SHA_A,
        tensorCount: 1,
        parameterCount: 2,
        headerLength: 32,
        headerSha256: SHA_B,
        inventorySha256: SHA_C,
      },
    ],
    files: [
      {
        name: "conversion-plan.json",
        byteLength: 1,
        sha256: SHA_A,
        kind: "conversion-plan",
      },
      {
        name: "licenses/ACE-Step-LICENSE",
        byteLength: 1_064,
        sha256: "05a6bce42a62636d2cfb24139cc008b6b899754e244175814bb5dd2f4a485357",
        kind: "license",
      },
      {
        name: "licenses/Apache-2.0-LICENSE",
        byteLength: 11_358,
        sha256: "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
        kind: "license",
      },
      {
        name: "licenses/Qwen-NOTICE.txt",
        byteLength: 439,
        sha256: "c57cecae352eb5793befd1f28f44f351e148c9a28044d855b8c361c562195f0b",
        kind: "license",
      },
      {
        name: "weights/dit/shared-00.bin",
        byteLength: 256,
        sha256: SHA_B,
        kind: "weights",
      },
    ],
    tensors: {
      "ace.decoder.weight": {
        shard: "weights/dit/shared-00.bin",
        byteOffset: 0,
        byteLength: 4,
        dtype: "uint32-bf16-pairs",
        logicalShape: [2],
        storageShape: [1],
        layout: "source-row-major-bf16-pairs-lsb-u32",
        phase: "dit",
        lifetime: "dit",
        source: "ace-turbo-weights:decoder.weight",
        transformation: "preserve-bf16-bits-pack-u32-pairs",
        logicalTensor: "ace.decoder.weight",
        partAxis: 0,
        partStart: 0,
        partEnd: 2,
      },
    },
    accounting: {
      sourceTensors: 1,
      directlyIncluded: 1,
      consumedByTransform: 0,
      excluded: 0,
      outputTensorsBeforeRowSharding: 1,
      constantTensors: 0,
      outputTensorsAfterRowSharding: 1,
    },
    licenses: [
      {
        component: "ACE-Step source and model snapshots",
        spdx: "MIT",
        notice: "retain notice",
        source: "https://github.com/ace-step/ACE-Step-1.5.git",
      },
      {
        component: "Qwen derived weights",
        spdx: "Apache-2.0",
        notice: "retain notice",
        source: "https://huggingface.co/Qwen/Qwen3-Embedding-0.6B",
      },
    ],
    provenance: {
      converterRevision: ACE_PACKAGE_CONVERTER_REVISION,
      aceSnapshot: ACE_MODEL_SNAPSHOT_REVISION,
      plannerSnapshot: ACE_PLANNER_SNAPSHOT_REVISION,
      referenceRepository: "https://github.com/ace-step/ACE-Step-1.5.git",
      referenceCommit: ACE_REFERENCE_SOURCE_REVISION,
      referenceLicenseGitBlob: "600451d484a555c1273baa2602f32a37fdd0d0ab",
      referenceLicenseSha256:
        "05a6bce42a62636d2cfb24139cc008b6b899754e244175814bb5dd2f4a485357",
      determinism: "sorted source/output inventories, fixed transforms, canonical JSON",
    },
  };
}
