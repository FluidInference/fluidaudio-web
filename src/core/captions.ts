// Word timestamps → SRT/VTT captions.
//
// The RNNT/TDT decoders emit (tokenId, encoderFrame) pairs; engines convert
// frames to seconds (×80ms) and pass them here. SentencePiece marks word
// starts with ▁, so words = token runs between markers; a word spans its
// first token's time to its last token's time (+ one frame).

import type { AsrSegment } from "./types.js";

const FRAME_SEC = 0.08;

/** Token (id, time) stream → word-level segments. `skipId` filters control
 * tokens (blank/EOU/EOB); tokens missing from id2token are ignored. */
export function tokensToWords(ids: number[], times: number[], id2token: string[] | Record<number, string>, skipId?: (id: number) => boolean): AsrSegment[] {
  const words: AsrSegment[] = [];
  let text = "";
  let start = 0;
  let end = 0;
  const flush = () => {
    if (text.trim()) words.push({ text: text.trim(), start, end });
    text = "";
  };
  for (let i = 0; i < ids.length; i++) {
    const tok = (id2token as Record<number, string>)[ids[i]];
    if (tok === undefined || (skipId && skipId(ids[i]))) continue;
    if (tok.startsWith("▁")) {
      flush();
      start = times[i];
    }
    if (!text) start = times[i];
    text += tok.replace(/▁/g, " ");
    end = times[i] + FRAME_SEC;
  }
  flush();
  return words;
}

/** Group words into caption cues: break on silence gaps, cue duration, or
 * line length — the standard readable-captions heuristics. */
export function groupCues(words: AsrSegment[], { maxGapSec = 0.9, maxDurSec = 5, maxChars = 84 } = {}): AsrSegment[] {
  const cues: AsrSegment[] = [];
  let cur: AsrSegment | null = null;
  for (const w of words) {
    const wouldBreak = cur && (w.start - cur.end > maxGapSec || w.end - cur.start > maxDurSec || cur.text.length + 1 + w.text.length > maxChars);
    if (!cur || wouldBreak) {
      if (cur) cues.push(cur);
      cur = { ...w };
    } else {
      cur.text += " " + w.text;
      cur.end = w.end;
    }
  }
  if (cur) cues.push(cur);
  return cues;
}

function ts(sec: number, sep: string): string {
  // Total-ms first: rounding the fraction alone can carry to ms=1000 and emit
  // a malformed "00:00:01,1000" (sec=1.9995).
  const t = Math.round(sec * 1000);
  const ms = t % 1000;
  const s = Math.floor(t / 1000) % 60;
  const m = Math.floor(t / 60000) % 60;
  const h = Math.floor(t / 3600000);
  const p = (n: number, w = 2) => String(n).padStart(w, "0");
  return `${p(h)}:${p(m)}:${p(s)}${sep}${p(ms, 3)}`;
}

/** Word segments → SubRip. */
export function segmentsToSrt(words: AsrSegment[]): string {
  return groupCues(words)
    .map((c, i) => `${i + 1}\n${ts(c.start, ",")} --> ${ts(c.end, ",")}\n${c.text}\n`)
    .join("\n");
}

/** Word segments → WebVTT. */
export function segmentsToVtt(words: AsrSegment[]): string {
  return (
    "WEBVTT\n\n" +
    groupCues(words)
      .map((c) => `${ts(c.start, ".")} --> ${ts(c.end, ".")}\n${c.text}\n`)
      .join("\n")
  );
}
