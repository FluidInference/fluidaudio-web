import { readFileSync } from "node:fs";
import vm from "node:vm";
import ts from "typescript";
import { isAceWorkerMessage } from "ace-step-1.5.wgsl";
import { expect, it, vi } from "vitest";

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
