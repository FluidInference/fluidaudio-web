import { readFileSync } from "node:fs";
import vm from "node:vm";
import ts from "typescript";
import { isAceWorkerMessage } from "ace-step-1.5.wgsl";
import { expect, it, vi } from "vitest";
import { ACE_WORKER_DISPOSAL_TIMEOUT_MS, waitForWorkerDisposal } from "../src/engines/musicgen-acestep/worker-disposal.js";

it("terminates the page worker and awaits its cache lease after disposal errors", async () => {
  const source = readFileSync(new URL("../src/music.ts", import.meta.url), "utf8");
  const ast = ts.createSourceFile("music.ts", source, ts.ScriptTarget.Latest, true);
  const handlers = ast.statements.filter((node) => ts.isFunctionDeclaration(node) && ["disposeWorker", "onWorkerMessage"].includes(node.name?.text ?? ""));
  const release = vi.fn(async () => {});
  const worker = { postMessage: vi.fn(), terminate: vi.fn() };
  const state: Record<string, any> = {
    worker,
    workerReady: true,
    nextRequestId: 1,
    disposal: undefined,
    releaseRuntimeLock: release,
    isAceWorkerMessage,
    waitForWorkerDisposal,
  };
  vm.createContext(state);
  vm.runInContext(ts.transpile(handlers.map((node) => node.getText(ast)).join("\n"), { target: ts.ScriptTarget.ES2022 }), state);
  const disposed = state.disposeWorker();
  state.onWorkerMessage({
    data: { type: "error", requestId: 1, error: { name: "Error", code: "DISPOSE_FAILED", message: "Device cleanup failed" } },
  });
  await expect(disposed).rejects.toThrow("Device cleanup failed");
  expect(worker.terminate).toHaveBeenCalledOnce();
  expect(release).toHaveBeenCalledOnce();
  expect(state.worker).toBeUndefined();
  expect(state.workerReady).toBe(false);
});

it("forces page worker cleanup when disposal gets no response", async () => {
  vi.useFakeTimers();
  try {
    const source = readFileSync(new URL("../src/music.ts", import.meta.url), "utf8");
    const ast = ts.createSourceFile("music.ts", source, ts.ScriptTarget.Latest, true);
    const handler = ast.statements.find((node) => ts.isFunctionDeclaration(node) && node.name?.text === "disposeWorker");
    expect(handler).toBeDefined();
    const release = vi.fn(async () => {});
    const worker = { postMessage: vi.fn(), terminate: vi.fn() };
    const state: Record<string, any> = {
      worker,
      workerReady: true,
      nextRequestId: 1,
      disposal: undefined,
      releaseRuntimeLock: release,
      waitForWorkerDisposal,
    };
    vm.createContext(state);
    vm.runInContext(ts.transpile(handler!.getText(ast), { target: ts.ScriptTarget.ES2022 }), state);
    const rejected = expect(state.disposeWorker()).rejects.toThrow("disposal timed out");
    await vi.advanceTimersByTimeAsync(ACE_WORKER_DISPOSAL_TIMEOUT_MS);
    await rejected;
    expect(worker.terminate).toHaveBeenCalledOnce();
    expect(release).toHaveBeenCalledOnce();
    expect(state.worker).toBeUndefined();
  } finally {
    vi.useRealTimers();
  }
});

it("waits for stem GPU disposal before releasing the previous result", async () => {
  const source = readFileSync(new URL("../src/music.ts", import.meta.url), "utf8");
  const ast = ts.createSourceFile("music.ts", source, ts.ScriptTarget.Latest, true);
  const handlers = ast.statements.filter(
    (node) => ts.isFunctionDeclaration(node) && ["releaseCurrentOutput", "resetStemSplitter"].includes(node.name?.text ?? ""),
  );
  let finish!: () => void;
  const disposal = new Promise<void>((resolve) => {
    finish = resolve;
  });
  const dispose = vi.fn(() => disposal);
  const state: Record<string, any> = {
    stemEngine: { dispose },
    stemDisposal: undefined,
    resetStemsUi: vi.fn(),
    output: undefined,
  };
  vm.createContext(state);
  vm.runInContext(ts.transpile(handlers.map((node) => node.getText(ast)).join("\n"), { target: ts.ScriptTarget.ES2022 }), state);
  let settled = false;
  const release = state.releaseCurrentOutput().then(() => {
    settled = true;
  });
  await Promise.resolve();
  expect(dispose).toHaveBeenCalledOnce();
  expect(settled).toBe(false);
  finish();
  await release;
  expect(settled).toBe(true);
});
