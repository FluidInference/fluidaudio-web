#!/usr/bin/env node

import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { open, readFile, stat } from "node:fs/promises";
import { dirname, relative, resolve } from "node:path";

const repositoryRoot = resolve(import.meta.dirname, "..");
const manifestPath = resolveManifestPath(process.argv.slice(2));

try {
  const result = await verifyPackage(manifestPath);
  process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
} catch (error) {
  process.stderr.write(
    `${JSON.stringify(
      {
        ok: false,
        manifest: manifestPath,
        error: error instanceof Error ? error.message : String(error),
      },
      null,
      2,
    )}\n`,
  );
  process.exitCode = 1;
}

async function verifyPackage(currentManifestPath) {
  const manifest = JSON.parse(await readFile(currentManifestPath, "utf8"));
  assert(manifest?.schema === "dicose-wgsl-package-v1", "Unsupported package schema");
  assert(manifest.weights !== null && typeof manifest.weights === "object", "Manifest omits weights metadata");
  assert(Array.isArray(manifest.tensors) && manifest.tensors.length > 0, "Manifest has no tensors");
  assert(manifest.weights.dtype === "f16", "Package weight dtype must be f16");
  assert(manifest.weights.endianness === "little", "Package weights must be little-endian");
  assertPositiveInteger(manifest.weights.alignment, "weights.alignment");
  assertPositiveInteger(manifest.weights.byteLength, "weights.byteLength");
  assertSha256(manifest.weights.sha256, "weights.sha256");
  assertSafeRelativeFile(manifest.weights.file, "weights.file");

  const weightsPath = resolve(dirname(currentManifestPath), manifest.weights.file);
  const weightsInfo = await stat(weightsPath);
  assert(
    weightsInfo.size === manifest.weights.byteLength,
    `Weight byte length mismatch: manifest=${manifest.weights.byteLength}, file=${weightsInfo.size}`,
  );
  const fileSha256 = await sha256File(weightsPath);
  assert(
    fileSha256 === manifest.weights.sha256,
    `Weight SHA-256 mismatch: manifest=${manifest.weights.sha256}, file=${fileSha256}`,
  );

  const names = new Set();
  const payloads = new Map();
  let logicalElementCount = 0;
  for (const tensor of manifest.tensors) {
    validateTensor(tensor, manifest.weights, names);
    const elementCount = tensor.shape.reduce((total, dimension) => total * dimension, 1);
    logicalElementCount += elementCount;
    const key = `${tensor.offset}:${tensor.byteLength}`;
    const payload = payloads.get(key) ?? {
      offset: tensor.offset,
      byteLength: tensor.byteLength,
      sha256: tensor.sha256,
      tensors: [],
    };
    assert(
      payload.sha256 === tensor.sha256,
      `Aliased payload at ${key} has conflicting tensor SHA-256 values`,
    );
    payload.tensors.push(tensor.name);
    payloads.set(key, payload);
  }
  assert(
    Number.isSafeInteger(logicalElementCount),
    "Logical tensor element count exceeds JavaScript safe integer precision",
  );

  const uniquePayloads = [...payloads.values()].sort((left, right) => left.offset - right.offset);
  let previousEnd = 0;
  for (const payload of uniquePayloads) {
    assert(
      payload.offset >= previousEnd,
      `Payload ${payload.tensors[0]} overlaps a preceding payload`,
    );
    const end = payload.offset + payload.byteLength;
    assert(end <= manifest.weights.byteLength, `Payload ${payload.tensors[0]} exceeds weights file`);
    previousEnd = end;
  }

  const handle = await open(weightsPath, "r");
  try {
    for (const payload of uniquePayloads) {
      const actualSha256 = await sha256Range(handle, payload.offset, payload.byteLength);
      assert(
        actualSha256 === payload.sha256,
        `Tensor payload SHA-256 mismatch for ${payload.tensors.join(", ")}: expected ${payload.sha256}, got ${actualSha256}`,
      );
    }
  } finally {
    await handle.close();
  }

  validateAggregateMetadata(manifest, {
    logicalElementCount,
    uniquePayloads,
  });
  return {
    ok: true,
    schema: manifest.schema,
    manifest: relative(repositoryRoot, currentManifestPath) || ".",
    weights: {
      file: relative(repositoryRoot, weightsPath),
      byteLength: weightsInfo.size,
      sha256: fileSha256,
    },
    tensors: {
      logicalCount: manifest.tensors.length,
      uniquePayloadCount: uniquePayloads.length,
      logicalElementCount,
      uniquePayloadBytes: uniquePayloads.reduce((total, payload) => total + payload.byteLength, 0),
    },
  };
}

function validateTensor(tensor, weights, names) {
  assert(tensor !== null && typeof tensor === "object", "Tensor metadata must be an object");
  assert(typeof tensor.name === "string" && tensor.name.length > 0, "Tensor is missing a name");
  assert(!names.has(tensor.name), `Duplicate tensor name: ${tensor.name}`);
  names.add(tensor.name);
  assert(tensor.dtype === "f16", `Tensor ${tensor.name} has unsupported dtype`);
  assert(
    tensor.layout === "row-major" || tensor.layout === "linear-in-out" ||
      tensor.layout === "linear-tile-n128-k32" ||
      tensor.layout === "linear-tile-n256-k32" || tensor.layout === "conv-oihw",
    `Tensor ${tensor.name} has unsupported layout`,
  );
  assert(Array.isArray(tensor.shape) && tensor.shape.length > 0, `Tensor ${tensor.name} has no shape`);
  for (const dimension of tensor.shape) {
    assertPositiveInteger(dimension, `Tensor ${tensor.name} shape dimension`);
  }
  const elementCount = tensor.shape.reduce((total, dimension) => total * dimension, 1);
  assert(Number.isSafeInteger(elementCount), `Tensor ${tensor.name} has an unsafe element count`);
  assertNonNegativeInteger(tensor.offset, `Tensor ${tensor.name} offset`);
  assert(
    tensor.offset % weights.alignment === 0,
    `Tensor ${tensor.name} offset is not ${weights.alignment}-byte aligned`,
  );
  assertPositiveInteger(tensor.byteLength, `Tensor ${tensor.name} byteLength`);
  assert(
    tensor.byteLength === elementCount * 2,
    `Tensor ${tensor.name} f16 byteLength does not match its shape`,
  );
  assertSha256(tensor.sha256, `Tensor ${tensor.name} sha256`);
  assert(
    tensor.offset + tensor.byteLength <= weights.byteLength,
    `Tensor ${tensor.name} exceeds the weights file`,
  );
}

function validateAggregateMetadata(manifest, { logicalElementCount, uniquePayloads }) {
  const weights = manifest.weights;
  assert(
    weights.logicalTensorCount === manifest.tensors.length,
    "weights.logicalTensorCount does not match tensors.length",
  );
  assert(
    weights.uniqueTensorCount === uniquePayloads.length,
    "weights.uniqueTensorCount does not match unique payload ranges",
  );
  assert(
    weights.logicalElementCount === logicalElementCount,
    "weights.logicalElementCount does not match tensor shapes",
  );
  const uniquePayloadBytes = uniquePayloads.reduce(
    (total, payload) => total + payload.byteLength,
    0,
  );
  assert(
    weights.uniquePayloadBytes === uniquePayloadBytes,
    "weights.uniquePayloadBytes does not match unique tensor ranges",
  );

  if (Array.isArray(manifest.components)) {
    for (const component of manifest.components) {
      assert(typeof component.namespace === "string", "Component is missing a namespace");
      const tensors = manifest.tensors.filter((tensor) =>
        tensor.name.startsWith(`${component.namespace}.`),
      );
      const elementCount = tensors.reduce(
        (total, tensor) => total + tensor.shape.reduce((size, dimension) => size * dimension, 1),
        0,
      );
      assert(
        component.expectedTensorCount === tensors.length,
        `Component ${component.namespace} tensor count does not match its manifest`,
      );
      assert(
        component.expectedElementCount === elementCount,
        `Component ${component.namespace} element count does not match its manifest`,
      );
    }
  }
}

async function sha256File(path) {
  const hash = createHash("sha256");
  await new Promise((resolvePromise, rejectPromise) => {
    const source = createReadStream(path);
    source.on("data", (chunk) => hash.update(chunk));
    source.once("end", resolvePromise);
    source.once("error", rejectPromise);
  });
  return hash.digest("hex");
}

async function sha256Range(handle, offset, byteLength) {
  const hash = createHash("sha256");
  const buffer = Buffer.allocUnsafe(Math.min(byteLength, 4 * 1024 * 1024));
  let position = offset;
  let remaining = byteLength;
  while (remaining > 0) {
    const requested = Math.min(buffer.byteLength, remaining);
    const { bytesRead } = await handle.read(buffer, 0, requested, position);
    assert(bytesRead === requested, `Unexpected EOF at byte ${position}`);
    hash.update(buffer.subarray(0, bytesRead));
    position += bytesRead;
    remaining -= bytesRead;
  }
  return hash.digest("hex");
}

function resolveManifestPath(argumentsList) {
  if (argumentsList.length === 0) {
    return joinPublicManifest();
  }
  if (argumentsList.length === 2 && argumentsList[0] === "--manifest") {
    return resolve(repositoryRoot, argumentsList[1]);
  }
  throw new Error("Usage: node scripts/verify-package.mjs [--manifest public/model/manifest.json]");
}

function joinPublicManifest() {
  return resolve(repositoryRoot, "public/model/manifest.json");
}

function assertSafeRelativeFile(value, label) {
  assert(typeof value === "string" && value.length > 0, `${label} must be a non-empty path`);
  assert(!value.startsWith("/") && !value.includes("\\"), `${label} must be a relative POSIX path`);
  assert(!value.split("/").includes(".."), `${label} may not traverse out of the package`);
}

function assertSha256(value, label) {
  assert(typeof value === "string" && /^[a-f0-9]{64}$/.test(value), `${label} must be a lowercase SHA-256`);
}

function assertNonNegativeInteger(value, label) {
  assert(Number.isSafeInteger(value) && value >= 0, `${label} must be a non-negative safe integer`);
}

function assertPositiveInteger(value, label) {
  assert(Number.isSafeInteger(value) && value > 0, `${label} must be a positive safe integer`);
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}
