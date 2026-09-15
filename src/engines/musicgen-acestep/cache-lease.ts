/** Acquire a Web Lock before returning an idempotent, awaitable release function. */
export function acquireCacheLease(locks: LockManager, name: string, signal?: AbortSignal): Promise<() => Promise<void>> {
  return new Promise((resolve, reject) => {
    const released = locks.request(name, { mode: "shared", ...(signal === undefined ? {} : { signal }) }, async () => {
      await new Promise<void>((releaseLock) => {
        let releasePromise: Promise<void> | undefined;
        resolve(async () => {
          if (releasePromise === undefined) {
            releaseLock();
            releasePromise = released.then(() => undefined);
          }
          await releasePromise;
        });
      });
    });
    void released.catch(reject);
  });
}
