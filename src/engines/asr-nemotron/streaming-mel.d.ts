export class StreamingMel {
  constructor(nMels?: number);
  readonly samples: number;
  push(samples: Float32Array): { data: Float32Array | null; count: number };
  flush(): { data: Float32Array | null; count: number };
}
