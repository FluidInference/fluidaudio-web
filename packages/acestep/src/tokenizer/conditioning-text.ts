/** Exact pinned SFT caption/metadata wrapper consumed by the Qwen text encoder. */
export function formatAceTextEncoderCaptionInput(
  instruction: string,
  caption: string,
  formattedMetadata: string,
): string {
  const normalizedInstruction = instruction.endsWith(":")
    ? instruction
    : `${instruction}:`;
  return (
    `# Instruction\n${normalizedInstruction}\n\n` +
    `# Caption\n${caption}\n\n` +
    `# Metas\n${formattedMetadata}<|endoftext|>\n`
  );
}

/** Exact pinned lyric/language wrapper consumed by Qwen token embeddings. */
export function formatAceTextEncoderLyricsInput(
  lyrics: string,
  language: string,
): string {
  return `# Languages\n${language}\n\n# Lyric\n${lyrics}<|endoftext|>`;
}
