// Window planning for long-form transcription: overlapping fixed-length
// windows, batched into equal-length groups for the encoder.

/** Window start offsets at a fixed hop; short input gets a single window. */
export function planWindows(totalSamples, winSamples, hop) {
  const starts = [];
  const single = totalSamples <= winSamples;
  for (let s = 0; s < totalSamples; s += hop) {
    starts.push(s);
    if (single) break;
  }
  return starts;
}

/** Groups of up to wb equal-length windows (batched through the encoder:
 * bigger GEMMs + one readback per group). A short tail encodes alone. */
export function groupWindows(starts, totalSamples, winSamples, wb) {
  const groups = [];
  let cur = [];
  for (let i = 0; i < starts.length; i++) {
    const len = Math.min(starts[i] + winSamples, totalSamples) - starts[i];
    if (cur.length && (len !== winSamples || cur.length >= wb)) {
      groups.push(cur);
      cur = [];
    }
    cur.push(i);
    if (len !== winSamples) {
      groups.push(cur);
      cur = [];
    }
  }
  if (cur.length) groups.push(cur);
  return groups;
}
