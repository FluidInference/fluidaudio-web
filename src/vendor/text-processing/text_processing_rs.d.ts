/* tslint:disable */
/* eslint-disable */

export function addRule(spoken: string, written: string): void;

export function clearRules(): void;

export function normalize(input: string): string;

export function normalizeSentence(input: string): string;

/**
 * Unified sentence normalize. `concatCompoundNumbers` mirrors the
 * single-expression flag; `maxSpanTokens == 0` means "use library default"
 * (16). `disableBareSecond=true` keeps phrases like `"give me a second"`
 * literal (issue #22).
 */
export function normalizeSentenceWithOptions(input: string, concat_compound_numbers: boolean, max_span_tokens: number, disable_bare_second: boolean): string;

export function normalizeWithLang(input: string, lang: string): string;

/**
 * Unified single-expression normalize. `concatCompoundNumbers=true` reads
 * consecutive number words as concatenation rather than addition, e.g.
 * `"thirty five sixty two"` → `"3562"`, `"seven eighty eight"` → `"788"`.
 * `disableBareSecond=true` blocks the bare word `"second"` from converting
 * to `"2nd"` (issue #22).
 */
export function normalizeWithOptions(input: string, concat_compound_numbers: boolean, disable_bare_second: boolean): string;

export function removeRule(spoken: string): boolean;

export function ruleCount(): number;

/**
 * Initialize panic hook for better error messages in browser devtools.
 */
export function set_panic_hook(): void;

export function tnNormalize(input: string): string;

export function tnNormalizeLang(input: string, lang: string): string;

export function tnNormalizeSentence(input: string): string;

export function tnNormalizeSentenceLang(input: string, lang: string): string;

export function tnNormalizeSentenceWithMaxSpan(input: string, max_span_tokens: number): string;

export function tnNormalizeSentenceWithMaxSpanLang(input: string, lang: string, max_span_tokens: number): string;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly addRule: (a: number, b: number, c: number, d: number) => void;
  readonly normalize: (a: number, b: number) => [number, number];
  readonly normalizeSentence: (a: number, b: number) => [number, number];
  readonly normalizeSentenceWithOptions: (a: number, b: number, c: number, d: number, e: number) => [number, number];
  readonly normalizeWithLang: (a: number, b: number, c: number, d: number) => [number, number];
  readonly normalizeWithOptions: (a: number, b: number, c: number, d: number) => [number, number];
  readonly removeRule: (a: number, b: number) => number;
  readonly tnNormalize: (a: number, b: number) => [number, number];
  readonly tnNormalizeLang: (a: number, b: number, c: number, d: number) => [number, number];
  readonly tnNormalizeSentence: (a: number, b: number) => [number, number];
  readonly tnNormalizeSentenceLang: (a: number, b: number, c: number, d: number) => [number, number];
  readonly tnNormalizeSentenceWithMaxSpan: (a: number, b: number, c: number) => [number, number];
  readonly tnNormalizeSentenceWithMaxSpanLang: (a: number, b: number, c: number, d: number, e: number) => [number, number];
  readonly set_panic_hook: () => void;
  readonly ruleCount: () => number;
  readonly clearRules: () => void;
  readonly __wbindgen_free: (a: number, b: number, c: number) => void;
  readonly __wbindgen_malloc: (a: number, b: number) => number;
  readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
  readonly __wbindgen_externrefs: WebAssembly.Table;
  readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init(module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
