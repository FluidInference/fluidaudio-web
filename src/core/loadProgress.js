/** Format one model-load progress update for the browser UI. */
export function formatLoadProgress(progress) {
  const fraction = Number.isFinite(progress.fraction) ? Math.min(1, Math.max(0, progress.fraction)) : 0;
  const percentage = Math.round(fraction * 100);
  const file = progress.file || "model";

  if (progress.phase !== "download") return `Loading ${file} — ${percentage}%`;

  const amount = formatDownloadAmount(progress.loaded, progress.total);
  return `Downloading ${file} — ${percentage}%${amount ? ` (${amount})` : ""}`;
}

function formatDownloadAmount(loaded, total) {
  if (!Number.isFinite(loaded) || !Number.isFinite(total) || total <= 0) return "";

  const units = ["B", "KB", "MB", "GB"];
  const unitIndex = Math.min(units.length - 1, Math.max(0, Math.floor(Math.log10(total) / 3)));
  const divisor = 1000 ** unitIndex;
  const decimals = total / divisor < 10 && unitIndex > 0 ? 1 : 0;
  return `${(Math.max(0, loaded) / divisor).toFixed(decimals)} / ${(total / divisor).toFixed(decimals)} ${units[unitIndex]}`;
}
