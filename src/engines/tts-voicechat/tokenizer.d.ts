export class VoicechatTokenizer {
  constructor(tok: { vocab: string[]; merges: string[]; ignoreMerges: boolean; splitRegex: string });
  vocab: Map<string, number>;
  encode(text: string): number[];
}
