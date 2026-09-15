/** Owns one disposable resource and prevents overlapping operations on it. */
export class ResourceSession<T extends { dispose(): Promise<void> }> {
  private resource: T | undefined;
  private pending: Promise<unknown> | undefined;
  private closed = false;
  private closing: Promise<void> | undefined;

  get current(): T | undefined {
    return this.resource;
  }

  get busy(): boolean {
    return this.pending !== undefined || this.closed;
  }

  async load(factory: () => Promise<T>, initialize: (resource: T) => Promise<void>): Promise<T> {
    return this.exclusive(async () => {
      await this.release();
      const resource = await factory();
      try {
        if (this.closed) throw new DOMException("Page closed", "AbortError");
        await initialize(resource);
        if (this.closed) throw new DOMException("Page closed", "AbortError");
        this.resource = resource;
        return resource;
      } catch (error) {
        await resource.dispose();
        throw error;
      }
    });
  }

  async run<Result>(operation: (resource: T) => Promise<Result>): Promise<Result> {
    return this.exclusive(async () => {
      if (this.resource === undefined) throw new Error("Load a model first");
      return operation(this.resource);
    });
  }

  async clear(): Promise<void> {
    return this.exclusive(() => this.release());
  }

  /** Waits for in-flight work before disposing; no new work may start. */
  close(): Promise<void> {
    if (this.closing !== undefined) return this.closing;
    this.closed = true;
    this.closing = (async () => {
      await this.pending?.catch(() => undefined);
      await this.release();
    })();
    return this.closing;
  }

  private async release(): Promise<void> {
    await this.resource?.dispose();
    this.resource = undefined;
  }

  private async exclusive<Result>(operation: () => Promise<Result>): Promise<Result> {
    if (this.busy) throw new Error("An operation is already in progress or the page is closed");
    const pending = Promise.resolve().then(operation);
    this.pending = pending;
    try {
      return await pending;
    } finally {
      this.pending = undefined;
    }
  }
}
