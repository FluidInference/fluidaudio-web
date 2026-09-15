const PENDING_OUTPUTS_KEY = "ace-step-pending-output-ids";
const PENDING_OUTPUTS_LOCK = "ace-step-pending-output-registry";
const STALE_AFTER_MS = 60 * 60 * 1000;

interface PendingOutput {
  readonly id: string;
  readonly at: number;
}

type RegistryStorage = Pick<Storage, "getItem" | "setItem">;

export interface PendingOutputRegistryOptions {
  readonly storage?: RegistryStorage;
  readonly locks?: LockManager;
  readonly now?: () => number;
}

export async function recordPendingOutput(id: string, options: PendingOutputRegistryOptions = {}): Promise<void> {
  await updateRecords((records) => [...records.filter((record) => record.id !== id), { id, at: now(options) }], options);
}

export async function forgetPendingOutput(id: string, options: PendingOutputRegistryOptions = {}): Promise<void> {
  await updateRecords((records) => records.filter((record) => record.id !== id), options);
}

export async function reclaimOrphanedOutputs(
  currentId: string | undefined,
  release: (id: string) => Promise<unknown>,
  options: PendingOutputRegistryOptions = {},
): Promise<void> {
  const cutoff = now(options) - STALE_AFTER_MS;
  const deleted = new Set<string>();
  for (const record of readRecords(storage(options))) {
    if (record.id === currentId || record.at > cutoff) continue;
    try {
      await release(record.id);
      deleted.add(record.id);
    } catch {
      // Keep the record so a later visit retries the deletion.
    }
  }
  if (deleted.size > 0) {
    await updateRecords((records) => records.filter((record) => !deleted.has(record.id)), options);
  }
}

async function updateRecords(transform: (records: PendingOutput[]) => PendingOutput[], options: PendingOutputRegistryOptions): Promise<void> {
  const target = storage(options);
  const update = () => writeRecords(target, transform(readRecords(target)));
  const locks = options.locks ?? globalThis.navigator?.locks;
  if (locks === undefined) {
    update();
    return;
  }
  try {
    await locks.request(PENDING_OUTPUTS_LOCK, { mode: "exclusive" }, update);
  } catch {
    // Registry bookkeeping is best-effort when Web Locks are unavailable.
    update();
  }
}

function storage(options: PendingOutputRegistryOptions): RegistryStorage | undefined {
  try {
    return options.storage ?? globalThis.localStorage;
  } catch {
    return undefined;
  }
}

function now(options: PendingOutputRegistryOptions): number {
  return (options.now ?? Date.now)();
}

function readRecords(target: RegistryStorage | undefined): PendingOutput[] {
  if (target === undefined) return [];
  try {
    const parsed = JSON.parse(target.getItem(PENDING_OUTPUTS_KEY) ?? "[]") as unknown;
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(
      (record): record is PendingOutput =>
        typeof record === "object" && record !== null && typeof record.id === "string" && typeof record.at === "number" && Number.isFinite(record.at),
    );
  } catch {
    return [];
  }
}

function writeRecords(target: RegistryStorage | undefined, records: PendingOutput[]): void {
  if (target === undefined) return;
  try {
    target.setItem(PENDING_OUTPUTS_KEY, JSON.stringify(records.slice(-20)));
  } catch {
    // Storage unavailable; persistent outputs wait for another cleanup path.
  }
}
