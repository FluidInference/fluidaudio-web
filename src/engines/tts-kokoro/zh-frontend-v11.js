// Kokoro v1.1-zh G2P frontend: hanzi → misaki[zh]-format phonemes
// (Bopomofo + hanzi-coded finals + tone digits), matching the v1.1 tokenizer
// vocab. Port of misaki/zh_frontend.py's ZH_MAP path: pinyin (pinyin-pro,
// polyphone-aware) → pypinyin-STRICT initial/final split → ZH_MAP → tone digit.
// No espeak / onnx. Punctuation passes through; unknowns are skipped.

import { pinyin } from "pinyin-pro";

// misaki/zh_frontend.py ZH_MAP (v1.1): pinyin initial/strict-final → Bopomofo /
// hanzi-coded final. Inlined (node ESM JSON imports need attributes; Vite doesn't).
const ZH_MAP = {
  b: "ㄅ",
  p: "ㄆ",
  m: "ㄇ",
  f: "ㄈ",
  d: "ㄉ",
  t: "ㄊ",
  n: "ㄋ",
  l: "ㄌ",
  g: "ㄍ",
  k: "ㄎ",
  h: "ㄏ",
  j: "ㄐ",
  q: "ㄑ",
  x: "ㄒ",
  zh: "ㄓ",
  ch: "ㄔ",
  sh: "ㄕ",
  r: "ㄖ",
  z: "ㄗ",
  c: "ㄘ",
  s: "ㄙ",
  a: "ㄚ",
  o: "ㄛ",
  e: "ㄜ",
  ie: "ㄝ",
  ai: "ㄞ",
  ei: "ㄟ",
  ao: "ㄠ",
  ou: "ㄡ",
  an: "ㄢ",
  en: "ㄣ",
  ang: "ㄤ",
  eng: "ㄥ",
  er: "ㄦ",
  i: "ㄧ",
  u: "ㄨ",
  v: "ㄩ",
  ii: "ㄭ",
  iii: "十",
  ve: "月",
  ia: "压",
  ian: "言",
  iang: "阳",
  iao: "要",
  in: "阴",
  ing: "应",
  iong: "用",
  iou: "又",
  ong: "中",
  ua: "穵",
  uai: "外",
  uan: "万",
  uang: "王",
  uei: "为",
  uen: "文",
  ueng: "瓮",
  uo: "我",
  van: "元",
  vn: "云",
};

const INITIALS = ["zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h", "j", "q", "x", "r", "z", "c", "s"];
const PUNC = new Set([";", ":", ",", ".", "!", "?", "—", "…", '"', "(", ")", "“", "”", " "]);
// zh punctuation → vocab punctuation
const PUNC_MAP = { "，": ",", "。": ".", "！": "!", "？": "?", "；": ";", "：": ":", "（": "(", "）": ")", "、": ",", "「": "“", "」": "”" };

/** Split an orthographic pinyin syllable (no tone) into pypinyin-STRICT initial+final. */
function strictSplit(syl) {
  let initial = "";
  for (const i of INITIALS) {
    if (syl.startsWith(i)) {
      initial = i;
      break;
    }
  }
  let fin = syl.slice(initial.length);
  if (initial) {
    // apical vowels: zhi/chi/shi/ri → iii ; zi/ci/si → ii
    if (fin === "i" && ["zh", "ch", "sh", "r"].includes(initial)) fin = "iii";
    else if (fin === "i" && ["z", "c", "s"].includes(initial)) fin = "ii";
    // j/q/x + u… are really ü…
    else if (["j", "q", "x"].includes(initial) && fin.startsWith("u")) fin = "v" + fin.slice(1);
    // orthographic contractions
    if (fin === "iu") fin = "iou";
    else if (fin === "ui") fin = "uei";
    else if (fin === "un") fin = ["j", "q", "x"].includes(initial) ? "vn" : "uen";
    else if (fin === "ue") fin = "ve"; // lüe/nüe written lve/nve handled below
    if (fin === "vn" && !["j", "q", "x"].includes(initial)) fin = "uen"; // safety
  } else if (syl.startsWith("y")) {
    const r = syl.slice(1);
    if (r.startsWith("u"))
      fin = r === "u" ? "v" : "v" + r.slice(1); // yu/yue/yuan/yun → v/ve/van/vn
    else if (r.startsWith("i"))
      fin = r; // yi/yin/ying → i/in/ing
    else fin = "i" + r; // ya/ye/yao/you/yan/yang/yong → ia/ie/iao/iou*/ian/iang/iong
    if (fin === "iou" || fin === "iu") fin = "iou";
    if (syl === "you") fin = "iou";
  } else if (syl.startsWith("w")) {
    const r = syl.slice(1);
    fin = r === "u" || r === "" ? "u" : "u" + r; // wu→u, wa→ua, wei→uei, wen→uen, …
  }
  if (fin.endsWith("ve") && fin !== "ve") fin = "ve";
  return { initial, fin };
}

/**
 * hanzi text → v1.1 phoneme string + coverage (fraction of hanzi converted).
 */
export function chineseToZh11(text) {
  const sylls = pinyin(text, { toneType: "num", type: "array", v: true, nonZh: "consecutive" });
  let out = "",
    han = 0,
    ok = 0;
  for (let i = 0; i < sylls.length; i++) {
    const raw = sylls[i].trim();
    if (!raw) continue;
    // passthrough punctuation
    if (raw.length === 1 && (PUNC.has(raw) || PUNC_MAP[raw])) {
      out += PUNC_MAP[raw] ?? raw;
      continue;
    }
    const m = raw.match(/^([a-zv]+)([0-9])$/);
    if (!m) continue; // non-pinyin run (latin etc.) — skipped for now
    han++;
    const tone = m[2] === "0" ? "5" : m[2];
    const { initial, fin } = strictSplit(m[1]);
    const iSym = initial ? ZH_MAP[initial] : "";
    const fSym = ZH_MAP[fin];
    if ((initial && !iSym) || !fSym) continue;
    out += iSym + fSym + tone;
    ok++;
  }
  return { phonemes: out, coverage: han ? ok / han : 0 };
}
