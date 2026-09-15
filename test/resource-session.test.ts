import { describe, expect, it, vi } from "vitest";
import { ResourceSession } from "../src/core/resource-session.js";

function deferred() {
  let resolve!: () => void;
  const promise = new Promise<void>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

// Disposable handles exercise ownership without creating models or audio.
function handle() {
  return { dispose: vi.fn(async () => {}) };
}

describe("resource ownership", () => {
  it("releases the previous resource before constructing its replacement", async () => {
    const session = new ResourceSession<ReturnType<typeof handle>>();
    const first = handle();
    const next = handle();
    await session.load(
      async () => first,
      async () => {},
    );
    await session.load(
      async () => {
        expect(first.dispose).toHaveBeenCalledOnce();
        return next;
      },
      async () => {},
    );
    expect(session.current).toBe(next);
    await session.close();
    expect(next.dispose).toHaveBeenCalledOnce();
  });

  it("rejects overlapping loads and runs without constructing another resource", async () => {
    const session = new ResourceSession<ReturnType<typeof handle>>();
    const gate = deferred();
    const item = handle();
    const load = session.load(
      async () => item,
      () => gate.promise,
    );
    const factory = vi.fn(async () => handle());
    await expect(session.load(factory, async () => {})).rejects.toThrow("already in progress");
    await expect(session.run(async () => {})).rejects.toThrow("already in progress");
    expect(factory).not.toHaveBeenCalled();
    gate.resolve();
    await load;
    await session.close();
  });

  it("disposes partial initialization and allows a retry", async () => {
    const session = new ResourceSession<ReturnType<typeof handle>>();
    const item = handle();
    await expect(
      session.load(
        async () => item,
        async () => {
          throw new Error("load failed");
        },
      ),
    ).rejects.toThrow("load failed");
    expect(item.dispose).toHaveBeenCalledOnce();
    expect(session.current).toBeUndefined();
    await session.load(
      async () => handle(),
      async () => {},
    );
    await session.close();
  });

  it("closes a resource whose load finishes after navigation", async () => {
    const session = new ResourceSession<ReturnType<typeof handle>>();
    const item = handle();
    const gate = deferred();
    const entered = deferred();
    const load = session.load(
      async () => item,
      async () => {
        entered.resolve();
        await gate.promise;
      },
    );
    const rejected = expect(load).rejects.toMatchObject({ name: "AbortError" });
    await entered.promise;
    const close = session.close();
    gate.resolve();
    await Promise.all([rejected, close]);
    expect(item.dispose).toHaveBeenCalledOnce();
    expect(session.current).toBeUndefined();
    await expect(
      session.load(
        async () => handle(),
        async () => {},
      ),
    ).rejects.toThrow("closed");
  });

  it("waits for in-flight use before teardown and disposes once", async () => {
    const session = new ResourceSession<ReturnType<typeof handle>>();
    const item = handle();
    await session.load(
      async () => item,
      async () => {},
    );
    const gate = deferred();
    const run = session.run(() => gate.promise);
    const close = session.close();
    expect(session.close()).toBe(close);
    expect(item.dispose).not.toHaveBeenCalled();
    gate.resolve();
    await Promise.all([run, close]);
    expect(item.dispose).toHaveBeenCalledOnce();
  });
});
