import { expect, it, vi } from "vitest";
import { claimPendingOutput, reclaimOrphanedOutputs, recordPendingOutput } from "../src/engines/musicgen-acestep/pending-output-registry.js";

class OwnershipLocks {
  private readonly shared = new Map<string, number>();

  async request(name: string, options: LockOptions, callback: (lock: Lock | null) => unknown): Promise<unknown> {
    if (options.mode === "exclusive" && options.ifAvailable && (this.shared.get(name) ?? 0) > 0) {
      return await callback(null);
    }
    if (options.mode === "shared") this.shared.set(name, (this.shared.get(name) ?? 0) + 1);
    try {
      return await callback({ name, mode: options.mode ?? "exclusive" } as Lock);
    } finally {
      if (options.mode === "shared") {
        const remaining = (this.shared.get(name) ?? 1) - 1;
        if (remaining === 0) this.shared.delete(name);
        else this.shared.set(name, remaining);
      }
    }
  }

  asManager(): LockManager {
    return this as unknown as LockManager;
  }
}

function memoryStorage(initial: { id: string; at: number }[] = []) {
  let value = JSON.stringify(initial);
  return {
    getItem: () => value,
    setItem: (_key: string, next: string) => {
      value = next;
    },
    records: () => JSON.parse(value) as { id: string; at: number }[],
  };
}

it("preserves a record added while stale output deletion is in flight", async () => {
  const storage = memoryStorage([{ id: "old-output", at: 0 }]);
  let finish!: () => void;
  const deletion = reclaimOrphanedOutputs(
    undefined,
    () =>
      new Promise<void>((resolve) => {
        finish = resolve;
      }),
    { storage, now: () => 10_000_000 },
  );
  await Promise.resolve();
  await recordPendingOutput("new-output", { storage, now: () => 10_000_000 });
  finish();
  await deletion;
  expect(storage.records()).toEqual([{ id: "new-output", at: 10_000_000 }]);
});

it("retains stale records when output deletion fails", async () => {
  const storage = memoryStorage([{ id: "retry-output", at: 0 }]);
  await reclaimOrphanedOutputs(
    undefined,
    async () => {
      throw new Error("Storage unavailable");
    },
    { storage, now: () => 10_000_000 },
  );
  expect(storage.records()).toEqual([{ id: "retry-output", at: 0 }]);
});

it("does not reclaim a stale output while another tab still owns it", async () => {
  const storage = memoryStorage();
  const locks = new OwnershipLocks().asManager();
  const releaseOwnership = await claimPendingOutput("active-output", { storage, locks, now: () => 0 });
  const release = vi.fn(async () => {});
  await reclaimOrphanedOutputs(undefined, release, { storage, locks, now: () => 10_000_000 });
  expect(release).not.toHaveBeenCalled();
  expect(storage.records()).toEqual([{ id: "active-output", at: 0 }]);

  await releaseOwnership();
  await reclaimOrphanedOutputs(undefined, release, { storage, locks, now: () => 10_000_000 });
  expect(release).toHaveBeenCalledExactlyOnceWith("active-output");
  expect(storage.records()).toEqual([]);
});

it("keeps output publication best-effort when storage rejects access", async () => {
  const storage = {
    getItem() {
      throw new DOMException("Storage blocked", "SecurityError");
    },
    setItem() {
      throw new DOMException("Storage blocked", "SecurityError");
    },
  };
  await expect(recordPendingOutput("output", { storage })).resolves.toBeUndefined();
});
