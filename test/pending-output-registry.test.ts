import { expect, it } from "vitest";
import { reclaimOrphanedOutputs, recordPendingOutput } from "../src/engines/musicgen-acestep/pending-output-registry.js";

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
