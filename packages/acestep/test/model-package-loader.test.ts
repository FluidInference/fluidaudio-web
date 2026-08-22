import { describe, expect, it, vi } from "vitest";

import { parseAcePackageManifest } from "../src/model/manifest.js";
import {
  ACE_MODEL_TRANSPORT_CHUNK_BYTES,
  AceModelTransportError,
  fetchAceModelAsset,
  fetchAceModelAssetResumable,
  loadAcePackageManifest,
  type AceModelAssetTransaction,
  type AceResumableModelAssetTransaction,
} from "../src/model/package.js";
import { aceSha256Hex } from "../src/model/sha256.js";
import { parseStrictJson } from "../src/model/strict-json.js";
import { syntheticAceManifest } from "./model-fixtures.js";

const encoder = new TextEncoder();

describe("strict JSON", () => {
  it("parses ordinary JSON and rejects duplicate keys", () => {
    expect(parseStrictJson('{"a":[true,null,"x"],"b":-1.25e2}')).toEqual({
      a: [true, null, "x"],
      b: -125,
    });
    expect(() => parseStrictJson('{"a":1,"a":2}')).toThrow(/duplicate object key/);
  });

  it("matches the standard SHA-256 vectors", () => {
    expect(aceSha256Hex(new Uint8Array())).toBe(
      "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    );
    expect(aceSha256Hex(encoder.encode("abc"))).toBe(
      "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
    );
  });
});

describe("authenticated ACE manifest loader", () => {
  it("hashes streamed bytes before parsing and returns a bound identity", async () => {
    const bytes = encoder.encode(`${JSON.stringify(syntheticAceManifest())}\n`);
    const fetchMock = vi.fn(async () => streamedResponse(bytes, 73));
    const progress: number[] = [];
    const loaded = await loadAcePackageManifest({
      manifestUrl: "https://models.example/v1/manifest.json",
      expectedManifestSha256: aceSha256Hex(bytes),
      expectedProfile: "reference",
      fetch: fetchMock as typeof fetch,
      onProgress: (event) => progress.push(event.receivedBytes),
    });
    expect(loaded.manifest.profile).toBe("reference");
    expect(loaded.manifestByteLength).toBe(bytes.byteLength);
    expect(loaded.manifestId).toContain(loaded.manifestSha256);
    expect(progress.at(-1)).toBe(bytes.byteLength);
  });

  it("rejects a digest mismatch before malformed JSON can be observed", async () => {
    const bytes = encoder.encode('{"format":');
    await expect(
      loadAcePackageManifest({
        manifestUrl: "https://models.example/manifest.json",
        expectedManifestSha256: "0".repeat(64),
        expectedProfile: "reference",
        fetch: (async () => new Response(bytes)) as typeof fetch,
      }),
    ).rejects.toMatchObject({ code: "MANIFEST_SHA256_MISMATCH" });
  });

  it("rejects duplicate keys and unknown fields after successful authentication", async () => {
    const duplicate = encoder.encode('{"format":1,"format":2}');
    await expect(loadManifestBytes(duplicate)).rejects.toMatchObject({
      code: "MANIFEST_JSON_ERROR",
    });

    const unknown = syntheticAceManifest();
    unknown.extra = true;
    const bytes = encoder.encode(JSON.stringify(unknown));
    await expect(loadManifestBytes(bytes)).rejects.toThrow(/unknown or missing fields/);
  });

  it("rejects an oversized declared manifest before reading its body", async () => {
    let cancelled = false;
    const body = new ReadableStream<Uint8Array>({
      cancel: () => {
        cancelled = true;
      },
    });
    await expect(
      loadAcePackageManifest({
        manifestUrl: "https://models.example/manifest.json",
        expectedManifestSha256: "0".repeat(64),
        expectedProfile: "reference",
        fetch: (async () =>
          new Response(body, {
            headers: { "content-length": String(2 * 1024 * 1024 + 1) },
          })) as typeof fetch,
      }),
    ).rejects.toMatchObject({ code: "MANIFEST_TOO_LARGE" });
    expect(cancelled).toBe(true);
  });

  it("cancels a non-200 manifest response body before rejecting it", async () => {
    let cancelled = false;
    const body = new ReadableStream<Uint8Array>({
      cancel: () => {
        cancelled = true;
      },
    });
    await expect(
      loadAcePackageManifest({
        manifestUrl: "https://models.example/manifest.json",
        expectedManifestSha256: "0".repeat(64),
        expectedProfile: "reference",
        fetch: (async () =>
          new Response(body, {
            status: 503,
            statusText: "Unavailable",
          })) as typeof fetch,
      }),
    ).rejects.toMatchObject({ code: "MANIFEST_HTTP_ERROR" });
    expect(cancelled).toBe(true);
  });
});

describe("bounded model asset transport", () => {
  it("commits only after exact streamed length and digest verification", async () => {
    const bytes = Uint8Array.from({ length: 31 }, (_, index) => index);
    const events: string[] = [];
    const written: number[] = [];
    const transaction: AceModelAssetTransaction = {
      write: (offset, chunk) => {
        events.push(`write:${offset}`);
        written.push(...chunk);
      },
      commit: () => {
        events.push("commit");
      },
      rollback: () => {
        events.push("rollback");
      },
    };
    await fetchAceModelAsset({
      manifestUrl: "https://models.example/v1/manifest.json",
      file: {
        name: "weights/test.bin",
        byteLength: bytes.byteLength,
        sha256: aceSha256Hex(bytes),
        kind: "weights",
      },
      transaction,
      fetch: (async () => streamedResponse(bytes, 5)) as typeof fetch,
    });
    expect(written).toEqual([...bytes]);
    expect(events.at(-1)).toBe("commit");
    expect(events).not.toContain("rollback");
  });

  it("slices an oversized response chunk before writing", async () => {
    const bytes = new Uint8Array(ACE_MODEL_TRANSPORT_CHUNK_BYTES + 17);
    const writeSizes: number[] = [];
    await fetchAceModelAsset({
      manifestUrl: "https://models.example/manifest.json",
      file: {
        name: "weights/large.bin",
        byteLength: bytes.byteLength,
        sha256: aceSha256Hex(bytes),
        kind: "weights",
      },
      transaction: {
        write: (_offset, chunk) => {
          writeSizes.push(chunk.byteLength);
        },
        commit: vi.fn(),
        rollback: vi.fn(),
      },
      fetch: (async () => new Response(bytes)) as typeof fetch,
    });
    expect(writeSizes).toEqual([ACE_MODEL_TRANSPORT_CHUNK_BYTES, 17]);
  });

  it("rolls back truncated and digest-mismatched tentative writes", async () => {
    for (const [bytes, expectedLength, expectedDigest] of [
      [Uint8Array.of(1, 2), 3, aceSha256Hex(Uint8Array.of(1, 2, 3))],
      [Uint8Array.of(1, 2, 3), 3, "0".repeat(64)],
    ] as const) {
      const transaction = {
        write: vi.fn(),
        commit: vi.fn(),
        rollback: vi.fn(),
      };
      await expect(
        fetchAceModelAsset({
          manifestUrl: "https://models.example/manifest.json",
          file: {
            name: "weights/fail.bin",
            byteLength: expectedLength,
            sha256: expectedDigest,
            kind: "weights",
          },
          transaction,
          fetch: (async () => new Response(bytes)) as typeof fetch,
        }),
      ).rejects.toBeInstanceOf(AceModelTransportError);
      expect(transaction.commit).not.toHaveBeenCalled();
      expect(transaction.rollback).toHaveBeenCalledTimes(1);
    }
  });

  it("drains rollback before surfacing cancellation", async () => {
    const controller = new AbortController();
    const events: string[] = [];
    const response = new ReadableStream<Uint8Array>({
      start(stream) {
        stream.enqueue(Uint8Array.of(1));
        stream.enqueue(Uint8Array.of(2));
        stream.close();
      },
    });
    const operation = fetchAceModelAsset({
      manifestUrl: "https://models.example/manifest.json",
      file: {
        name: "weights/cancel.bin",
        byteLength: 2,
        sha256: aceSha256Hex(Uint8Array.of(1, 2)),
        kind: "weights",
      },
      signal: controller.signal,
      transaction: {
        write: () => {
          events.push("write");
          controller.abort();
        },
        commit: () => {
          events.push("commit");
        },
        rollback: async () => {
          await Promise.resolve();
          events.push("rollback");
        },
      },
      fetch: (async () => new Response(response)) as typeof fetch,
    });
    await expect(operation).rejects.toMatchObject({ name: "AbortError" });
    expect(events).toEqual(["write", "rollback"]);
  });
});

describe("resumable model asset transport", () => {
  it("authenticates a retained prefix and requires an exact suffix Content-Range", async () => {
    const bytes = Uint8Array.of(1, 2, 3, 4, 5);
    const storage = new RetainedPartial(bytes.subarray(0, 2));
    const fetchMock = vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
      expect(new Headers(init?.headers).get("range")).toBe("bytes=2-");
      return new Response(bytes.subarray(2), {
        status: 206,
        headers: {
          "content-range": "bytes 2-4/5",
          "content-length": "3",
        },
      });
    });
    await fetchAceModelAssetResumable({
      manifestUrl: "https://models.example/v1/manifest.json",
      file: assetRecord(bytes),
      beginTransaction: async () => storage.open(),
      fetch: fetchMock as typeof fetch,
    });
    expect(storage.bytes).toEqual(bytes);
    expect(storage.events).toEqual(["hash:2", "write:2:3", "commit:5"]);
  });

  it("restarts safely when a server ignores Range with a complete 200 body", async () => {
    const bytes = Uint8Array.of(10, 11, 12, 13);
    const storage = new RetainedPartial(bytes.subarray(0, 2));
    await fetchAceModelAssetResumable({
      manifestUrl: "https://models.example/manifest.json",
      file: assetRecord(bytes),
      beginTransaction: async () => storage.open(),
      fetch: (async () =>
        new Response(bytes, {
          status: 200,
          headers: { "content-length": "4" },
        })) as typeof fetch,
    });
    expect(storage.bytes).toEqual(bytes);
    expect(storage.events).toEqual([
      "hash:2",
      "restart",
      "write:0:4",
      "commit:4",
    ]);
  });

  it("rejects a Range-ignoring 200 response without a complete length", async () => {
    const bytes = Uint8Array.of(10, 11, 12, 13);
    const storage = new RetainedPartial(bytes.subarray(0, 2));
    await expect(
      fetchAceModelAssetResumable({
        manifestUrl: "https://models.example/manifest.json",
        file: assetRecord(bytes),
        beginTransaction: async () => storage.open(),
        fetch: (async () => new Response(bytes, { status: 200 })) as typeof fetch,
      }),
    ).rejects.toMatchObject({ code: "MODEL_ASSET_LENGTH_MISMATCH" });
    expect(storage.events).toEqual(["hash:2", "rollback:2"]);
  });

  it("reopens and resumes after a truncated retryable body", async () => {
    const bytes = Uint8Array.of(20, 21, 22, 23, 24);
    const storage = new RetainedPartial();
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValueOnce(
        new Response(bytes.subarray(0, 2), {
          status: 200,
          headers: { "content-length": "5" },
        }),
      )
      .mockImplementationOnce(async (_input, init) => {
        expect(new Headers(init?.headers).get("range")).toBe("bytes=2-");
        return new Response(bytes.subarray(2), {
          status: 206,
          headers: {
            "content-range": "bytes 2-4/5",
            "content-length": "3",
          },
        });
      });
    const waits: number[] = [];
    await fetchAceModelAssetResumable({
      manifestUrl: "https://models.example/manifest.json",
      file: assetRecord(bytes),
      beginTransaction: async () => storage.open(),
      fetch: fetchMock,
      waitBeforeRetry: async (attempt) => {
        waits.push(attempt);
      },
    });
    expect(storage.bytes).toEqual(bytes);
    expect(storage.openCount).toBe(2);
    expect(waits).toEqual([1]);
    expect(storage.events).toEqual([
      "write:0:2",
      "rollback:2",
      "hash:2",
      "write:2:3",
      "commit:5",
    ]);
  });

  it("commits an already complete authenticated partial without fetching", async () => {
    const bytes = Uint8Array.of(30, 31, 32);
    const storage = new RetainedPartial(bytes);
    const fetchMock = vi.fn<typeof fetch>();
    await fetchAceModelAssetResumable({
      manifestUrl: "https://models.example/manifest.json",
      file: assetRecord(bytes),
      beginTransaction: async () => storage.open(),
      fetch: fetchMock,
    });
    expect(fetchMock).not.toHaveBeenCalled();
    expect(storage.events).toEqual(["hash:3", "commit:3"]);
  });

  it("fails closed on a mismatched Content-Range without retrying", async () => {
    const bytes = Uint8Array.of(40, 41, 42, 43);
    const storage = new RetainedPartial(bytes.subarray(0, 1));
    const fetchMock = vi.fn(async () =>
      new Response(bytes.subarray(1), {
        status: 206,
        headers: {
          "content-range": "bytes 0-3/4",
          "content-length": "3",
        },
      }),
    );
    await expect(
      fetchAceModelAssetResumable({
        manifestUrl: "https://models.example/manifest.json",
        file: assetRecord(bytes),
        beginTransaction: async () => storage.open(),
        fetch: fetchMock as typeof fetch,
        waitBeforeRetry: vi.fn(),
      }),
    ).rejects.toMatchObject({ code: "INVALID_CONTENT_RANGE" });
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(storage.events).toEqual(["hash:1", "rollback:1"]);
  });
});

async function loadManifestBytes(bytes: Uint8Array) {
  return await loadAcePackageManifest({
    manifestUrl: "https://models.example/manifest.json",
    expectedManifestSha256: aceSha256Hex(bytes),
    expectedProfile: "reference",
    fetch: (async () => new Response(Uint8Array.from(bytes))) as typeof fetch,
  });
}

function streamedResponse(bytes: Uint8Array, chunkBytes: number): Response {
  return new Response(
    new ReadableStream<Uint8Array>({
      start(controller) {
        for (let offset = 0; offset < bytes.byteLength; offset += chunkBytes) {
          controller.enqueue(bytes.slice(offset, offset + chunkBytes));
        }
        controller.close();
      },
    }),
    { headers: { "content-length": String(bytes.byteLength) } },
  );
}

function assetRecord(bytes: Uint8Array) {
  return {
    name: "weights/resumable.bin",
    byteLength: bytes.byteLength,
    sha256: aceSha256Hex(bytes),
    kind: "weights" as const,
  };
}

class RetainedPartial {
  bytes: Uint8Array;
  readonly events: string[] = [];
  openCount = 0;

  constructor(initial: Uint8Array = new Uint8Array()) {
    this.bytes = Uint8Array.from(initial);
  }

  open(): AceResumableModelAssetTransaction {
    this.openCount += 1;
    const state = this;
    let cursor = state.bytes.byteLength;
    let closed = false;
    return {
      get resumeOffset() {
        return cursor;
      },
      async hashExistingPrefix() {
        if (closed) throw new DOMException("closed", "InvalidStateError");
        state.events.push(`hash:${cursor}`);
      },
      async restart() {
        if (closed) throw new DOMException("closed", "InvalidStateError");
        state.events.push("restart");
        state.bytes = new Uint8Array();
        cursor = 0;
      },
      async write(offset, chunk) {
        if (closed) throw new DOMException("closed", "InvalidStateError");
        if (offset !== cursor) throw new RangeError("test cursor mismatch");
        const next = new Uint8Array(offset + chunk.byteLength);
        next.set(state.bytes);
        next.set(chunk, offset);
        state.bytes = next;
        cursor = next.byteLength;
        state.events.push(`write:${offset}:${chunk.byteLength}`);
      },
      async commit() {
        if (closed) throw new DOMException("closed", "InvalidStateError");
        state.events.push(`commit:${cursor}`);
        closed = true;
      },
      async rollback() {
        if (closed) return;
        state.events.push(`rollback:${cursor}`);
        closed = true;
      },
    };
  }
}

// Keep a direct parser reference in this suite so bundlers cannot accidentally
// tree-shake the deep validator away from the authenticated load path.
void parseAcePackageManifest;
