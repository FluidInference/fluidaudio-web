import { afterEach, expect, it, vi } from "vitest";
import { DicoseStemEngine } from "../src/engines/stem-dicose/index.js";

interface AdapterGate {
  readonly promise: Promise<GPUAdapter>;
  readonly resolve: (adapter: GPUAdapter) => void;
}

function adapter(): GPUAdapter {
  return {
    features: new Set(["shader-f16", "subgroups"]),
    info: { vendor: "test", architecture: "test", device: "test", description: "test", subgroupMinSize: 32, subgroupMaxSize: 32 },
    limits: { maxBufferSize: 1_073_741_824, maxStorageBufferBindingSize: 1_073_741_824, maxComputeWorkgroupStorageSize: 25_344 },
  } as unknown as GPUAdapter;
}

function deferredAdapter(): AdapterGate {
  let resolve!: (adapter: GPUAdapter) => void;
  return {
    promise: new Promise<GPUAdapter>((done) => {
      resolve = done;
    }),
    resolve,
  };
}

afterEach(() => vi.unstubAllGlobals());

it("does not create a stem worker when disposed during the capability check", async () => {
  const gate = deferredAdapter();
  vi.stubGlobal("isSecureContext", true);
  vi.stubGlobal("navigator", { gpu: { requestAdapter: () => gate.promise } });
  const worker = vi.fn();
  vi.stubGlobal("Worker", worker);
  const engine = new DicoseStemEngine();
  const load = engine.load();
  await engine.dispose();
  gate.resolve(adapter());
  await expect(load).rejects.toMatchObject({ name: "AbortError" });
  expect(worker).not.toHaveBeenCalled();
});

it("disposes a stem worker whose initialization finishes after teardown", async () => {
  vi.stubGlobal("isSecureContext", true);
  vi.stubGlobal("navigator", { gpu: { requestAdapter: async () => adapter() } });
  vi.stubGlobal("location", { href: "http://localhost/" });
  let transport!: WorkerTransport;
  class WorkerTransport extends EventTarget {
    readonly messages: { type: string; id: number }[] = [];
    readonly terminate = vi.fn();
    constructor() {
      super();
      transport = this;
    }
    postMessage(message: { type: string; id: number }) {
      this.messages.push(message);
    }
  }
  vi.stubGlobal("Worker", WorkerTransport);
  const engine = new DicoseStemEngine();
  const load = engine.load();
  await vi.waitFor(() => expect(transport.messages.map((message) => message.type)).toEqual(["initialize"]));
  const disposal = engine.dispose();
  expect(transport.messages.map((message) => message.type)).toEqual(["initialize", "dispose"]);
  const initializeId = transport.messages[0].id;
  transport.dispatchEvent(new MessageEvent("message", { data: { type: "initialized", id: initializeId } }));
  const disposeId = transport.messages[1].id;
  transport.dispatchEvent(new MessageEvent("message", { data: { type: "disposed", id: disposeId } }));
  await expect(load).rejects.toMatchObject({ name: "AbortError" });
  await disposal;
  expect(transport.terminate).toHaveBeenCalledOnce();
});
