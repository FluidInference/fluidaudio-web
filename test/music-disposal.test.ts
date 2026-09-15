import { readFileSync } from "node:fs";
import { expect, it, vi } from "vitest";
import { AceStepMusicClient, aceSeed } from "../src/engines/musicgen-acestep/index.js";

const request = {
  generationProfile: "ace-turbo-v1-correctness" as const,
  prompt: "Piano instrumental",
  instrumental: true,
  durationSeconds: 30,
  seed: aceSeed("42"),
  planner: { mode: "disabled" as const },
};

function readyClient() {
  const client = new AceStepMusicClient();
  // Set only lifecycle state; this transport never loads or simulates a model.
  const worker = { postMessage: vi.fn(), terminate: vi.fn() };
  const release = vi.fn(async () => {});
  Object.assign(client, { worker, workerReady: true, releaseRuntimeLock: release });
  const receive = (data: unknown) => Reflect.get(client, "onMessage").call(client, { data });
  return { client, worker, release, receive };
}

it("shares disposal until both callers settle and prevents new generation meanwhile", async () => {
  const { client, worker, release, receive } = readyClient();
  const first = client.dispose();
  const second = client.dispose();
  expect(second).toBe(first);
  expect(worker.postMessage).toHaveBeenCalledExactlyOnceWith({ type: "dispose", requestId: 1 });
  await expect(client.generate(request)).rejects.toThrow("already in progress");
  receive({ type: "disposed", requestId: 1 });
  await Promise.all([first, second]);
  expect(worker.terminate).toHaveBeenCalledOnce();
  expect(release).toHaveBeenCalledOnce();
  expect(client.initialized).toBe(false);
  await client.dispose();
  expect(worker.terminate).toHaveBeenCalledOnce();
});

it("does not generate when ready was already queued before initialization cancellation", async () => {
  const { client, worker, receive } = readyClient();
  let reject!: (error: unknown) => void;
  const generation = new Promise((_, rejectPromise) => {
    reject = rejectPromise;
  });
  Object.assign(client, {
    workerReady: false,
    active: {
      resolve: vi.fn(),
      reject,
      handlers: {},
      initializationRequestId: 7,
      jobId: undefined,
      request,
      cancelRequested: false,
    },
  });
  const rejected = expect(generation).rejects.toMatchObject({ name: "AbortError" });
  client.cancel();
  const receipt = JSON.parse(readFileSync(new URL("../packages/acestep/optimization/results/OPT-0073/final-180s-receipt.json", import.meta.url), "utf8")) as {
    result: { diagnostics: unknown };
  };
  receive({ type: "ready", requestId: 7, diagnostics: receipt.result.diagnostics });
  await rejected;
  expect(worker.postMessage).toHaveBeenCalledExactlyOnceWith({ type: "cancel-initialization", requestId: 7 });
  expect(client.busy).toBe(false);
  expect(client.initialized).toBe(true);
  client.terminate();
});

it("releases the worker and lease even when orderly disposal fails", async () => {
  const { client, worker, release, receive } = readyClient();
  const first = client.dispose();
  const second = client.dispose();
  const settled = Promise.allSettled([first, second]);
  receive({ type: "error", requestId: 1, error: { name: "Error", code: "DISPOSE_FAILED", message: "Disposal failed" } });
  expect((await settled).map((result) => result.status)).toEqual(["rejected", "rejected"]);
  expect(worker.terminate).toHaveBeenCalledOnce();
  expect(release).toHaveBeenCalledOnce();
  expect(client.initialized).toBe(false);
  await client.dispose();
});

it("cleans up when posting the disposal request throws", async () => {
  const { client, worker, release } = readyClient();
  worker.postMessage.mockImplementation(() => {
    throw new Error("Worker unavailable");
  });
  await expect(client.dispose()).rejects.toThrow("Worker unavailable");
  expect(worker.terminate).toHaveBeenCalledOnce();
  expect(release).toHaveBeenCalledOnce();
});

it("settles all disposal callers when terminated mid-disposal", async () => {
  const { client, worker, release } = readyClient();
  const settled = Promise.allSettled([client.dispose(), client.dispose()]);
  client.terminate();
  expect((await settled).map((result) => result.status)).toEqual(["rejected", "rejected"]);
  expect(worker.terminate).toHaveBeenCalledOnce();
  expect(release).toHaveBeenCalledOnce();
});
