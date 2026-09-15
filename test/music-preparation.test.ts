import { readFileSync } from "node:fs";
import vm from "node:vm";
import ts from "typescript";
import { isAceWorkerMessage } from "ace-step-1.5.wgsl";
import { expect, it, vi } from "vitest";

// Run the actual page handlers with UI/worker transports supplied by the test.
// No models or audio are substituted; execution stops before inference starts.
function page() {
  const source = readFileSync(new URL("../src/music.ts", import.meta.url), "utf8");
  const ast = ts.createSourceFile("music.ts", source, ts.ScriptTarget.Latest, true);
  const names = ["beginGeneration", "cancelActiveOperation"];
  const handlers = ast.statements.filter((node) => ts.isFunctionDeclaration(node) && names.includes(node.name?.text ?? ""));
  expect(handlers).toHaveLength(names.length);
  let finish!: () => void;
  let fail!: (error: Error) => void;
  const cleanup = new Promise<void>((resolve, reject) => {
    finish = resolve;
    fail = reject;
  });
  const state: Record<string, any> = {
    AbortController,
    pageLifecycle: new AbortController(),
    generationPreparation: undefined,
    busy: false,
    deletingModel: false,
    splittingStems: false,
    supportDetails: { supported: true },
    readGenerationRequest: () => ({ prompt: "Piano instrumental" }),
    formError: {},
    resultPanel: {},
    cancelButton: {},
    releaseCurrentOutput: vi.fn(() => cleanup),
    errorMessage: String,
    cacheAcquisition: undefined,
    initializationRequestId: undefined,
    activeJobId: undefined,
    INITIAL_MODEL_DOWNLOAD_PROGRESS: {},
    isModelDownloadComplete: () => true,
    cacheDetails: undefined,
    requestAceModelStoragePersistence: async () => false,
    recordValue: () => ({}),
    updateRuntimeDetails: () => {},
    workerDetails: undefined,
    workerReady: true,
    worker: { postMessage: vi.fn() },
    startPendingGeneration: vi.fn(),
    setDeterminateProgress: vi.fn(),
    setIndeterminateProgress: vi.fn(),
  };
  state.setBusy = (value: boolean) => {
    state.busy = value;
  };
  vm.createContext(state);
  vm.runInContext(ts.transpile(handlers.map((node) => node.getText(ast)).join("\n"), { target: ts.ScriptTarget.ES2022 }), state);
  return { state, finish, fail };
}

it("honors Cancel during output cleanup and blocks another generation until cleanup ends", async () => {
  const { state, finish } = page();
  const running = state.beginGeneration();
  state.cancelActiveOperation();
  expect(state.busy).toBe(true);
  await state.beginGeneration();
  expect(state.releaseCurrentOutput).toHaveBeenCalledOnce();
  finish();
  await running;
  expect(state.startPendingGeneration).not.toHaveBeenCalled();
  expect(state.worker.postMessage).not.toHaveBeenCalled();
  expect(state.busy).toBe(false);
  expect(state.generationPreparation).toBeUndefined();
  expect(state.setDeterminateProgress).toHaveBeenCalledWith(0, "Cancelled", "Song generation cancelled", "");
  await state.beginGeneration();
  expect(state.startPendingGeneration).toHaveBeenCalledOnce();
});

it("does not start generation if cleanup fails after cancellation", async () => {
  const { state, fail } = page();
  const running = state.beginGeneration();
  state.cancelActiveOperation();
  fail(new Error("Storage unavailable"));
  await running;
  expect(state.startPendingGeneration).not.toHaveBeenCalled();
  expect(state.busy).toBe(false);
  expect(state.generationPreparation).toBeUndefined();
  expect(state.formError.textContent).toContain("Storage unavailable");
});

it("does not start generation when the page closes during cleanup", async () => {
  const { state, finish } = page();
  const running = state.beginGeneration();
  state.pageLifecycle.abort();
  finish();
  await running;
  expect(state.startPendingGeneration).not.toHaveBeenCalled();
  expect(state.generationPreparation).toBeUndefined();
});

it("does not generate when ready was already queued before page cancellation", async () => {
  const source = readFileSync(new URL("../src/music.ts", import.meta.url), "utf8");
  const ast = ts.createSourceFile("music.ts", source, ts.ScriptTarget.Latest, true);
  const handlers = ast.statements.filter(
    (node) => ts.isFunctionDeclaration(node) && ["cancelActiveOperation", "onWorkerMessage"].includes(node.name?.text ?? ""),
  );
  const worker = { postMessage: vi.fn() };
  const state: Record<string, any> = {
    busy: true,
    worker,
    cacheAcquisition: undefined,
    generationPreparation: undefined,
    initializationRequestId: 7,
    initializationCancelRequested: false,
    activeJobId: undefined,
    pendingRequest: { prompt: "Piano instrumental" },
    cancelButton: {},
    workerReady: false,
    workerDetails: undefined,
    modelProgress: { fraction: 0.5 },
    isAceWorkerMessage,
    updateRuntimeDetails: vi.fn(),
    refreshCacheInfo: vi.fn(),
    setBusy: vi.fn((value: boolean) => {
      state.busy = value;
    }),
    setDeterminateProgress: vi.fn(),
    startPendingGeneration: vi.fn(),
  };
  vm.createContext(state);
  vm.runInContext(ts.transpile(handlers.map((node) => node.getText(ast)).join("\n"), { target: ts.ScriptTarget.ES2022 }), state);
  state.cancelActiveOperation();
  const receipt = JSON.parse(readFileSync(new URL("../packages/acestep/optimization/results/OPT-0073/final-180s-receipt.json", import.meta.url), "utf8")) as {
    result: { diagnostics: unknown };
  };
  state.onWorkerMessage({ data: { type: "ready", requestId: 7, diagnostics: receipt.result.diagnostics } });
  expect(worker.postMessage).toHaveBeenCalledExactlyOnceWith({ type: "cancel-initialization", requestId: 7 });
  expect(state.startPendingGeneration).not.toHaveBeenCalled();
  expect(state.pendingRequest).toBeUndefined();
  expect(state.busy).toBe(false);
  expect(state.workerReady).toBe(true);
});
