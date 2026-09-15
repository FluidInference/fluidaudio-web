export const ACE_WORKER_DISPOSAL_TIMEOUT_MS = 30_000;

export interface PendingWorkerDisposal {
  readonly requestId: number;
  readonly resolve: () => void;
  readonly reject: (reason: unknown) => void;
}

/** Start a disposal request and reject if the worker never acknowledges it. */
export function waitForWorkerDisposal(
  requestId: number,
  start: (pending: PendingWorkerDisposal) => void,
  timeoutMs = ACE_WORKER_DISPOSAL_TIMEOUT_MS,
): Promise<void> {
  return new Promise<void>((resolve, reject) => {
    let settled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const finish = (outcome: () => void) => {
      if (settled) return;
      settled = true;
      if (timer !== undefined) clearTimeout(timer);
      outcome();
    };
    const pending: PendingWorkerDisposal = {
      requestId,
      resolve: () => finish(resolve),
      reject: (reason) => finish(() => reject(reason)),
    };
    timer = setTimeout(() => pending.reject(new Error(`ACE worker disposal timed out after ${timeoutMs} ms`)), timeoutMs);
    try {
      start(pending);
    } catch (error) {
      pending.reject(error);
    }
  });
}
