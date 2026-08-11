// Mel scheduling for the group pipeline. With a melPool (browser workers),
// group g's mels compute WHILE the GPU runs earlier groups — main-thread mel
// was ~1.8s of unhidden wall on the 1-hour browser run — and resolveMels()
// times only the UNHIDDEN wait. The sync path (node gates / no workers) times
// the full extraction.

export function createMelScheduler({ mel, melPool, groups, starts, samples, winSamples, single, stats, now }) {
  const sliceFor = (i) => (single ? samples : samples.subarray(starts[i], Math.min(starts[i] + winSamples, samples.length)));

  const melsFor = (g) => {
    if (melPool) {
      return Promise.all(groups[g].map((i) => melPool.mel(sliceFor(i)))).then((rs) => rs.filter((r) => r.length > 0).map((r) => r.features));
    }
    const t0 = now();
    const mels = [];
    for (const i of groups[g]) {
      const { features, length } = mel.process(sliceFor(i));
      if (length > 0) mels.push(features);
    }
    stats.melMs += now() - t0;
    return mels;
  };

  const resolveMels = async (m) => {
    if (!melPool) return m;
    const t0 = now();
    const arr = await m;
    stats.melMs += now() - t0; // unhidden mel wait only
    return arr;
  };

  return { melsFor, resolveMels };
}
