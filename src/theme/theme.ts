// Shared theme wiring for the non-music pages. Mirrors the ACE-Step demo's
// applyTheme (src/music.ts): same localStorage key, same data attribute, same
// moon/sun icons — so light/dark follows the user across every page.

import lightModeIcon from "../engines/musicgen-acestep/assets/light-mode.png";
import moonIcon from "../engines/musicgen-acestep/assets/moon.png";

type SiteTheme = "light" | "dark";

const THEME_STORAGE_KEY = "ace-step-wgsl-demo-theme";

/**
 * Applies the saved theme (localStorage, falling back to the pre-paint
 * attribute already on <html>) and wires the #theme-toggle / #theme-icon
 * header button when the page has one. Safe no-op when it doesn't.
 */
export function initSiteTheme(): void {
  let theme: SiteTheme = document.documentElement.dataset.aceDemoTheme === "dark" ? "dark" : "light";
  try {
    const saved = localStorage.getItem(THEME_STORAGE_KEY);
    if (saved === "dark" || saved === "light") theme = saved;
  } catch {
    // Storage disabled — the pre-paint default still applies.
  }
  applyTheme(theme);

  const toggle = document.getElementById("theme-toggle");
  if (!(toggle instanceof HTMLButtonElement)) return;
  toggle.addEventListener("click", () => {
    const next: SiteTheme = document.documentElement.dataset.aceDemoTheme === "dark" ? "light" : "dark";
    try {
      localStorage.setItem(THEME_STORAGE_KEY, next);
    } catch {
      // Theme selection still applies for the current visit.
    }
    applyTheme(next);
  });
}

function applyTheme(theme: SiteTheme): void {
  document.documentElement.dataset.aceDemoTheme = theme;
  const dark = theme === "dark";

  const toggle = document.getElementById("theme-toggle");
  const icon = document.getElementById("theme-icon");
  if (toggle instanceof HTMLButtonElement && icon instanceof HTMLImageElement) {
    const label = dark ? "Switch to light theme" : "Switch to dark theme";
    toggle.setAttribute("aria-pressed", String(dark));
    toggle.setAttribute("aria-label", label);
    toggle.title = label;
    icon.src = dark ? lightModeIcon : moonIcon;
    icon.classList.toggle("is-sun", dark);
  }

  const meta = document.querySelector<HTMLMetaElement>('meta[name="theme-color"]');
  if (meta !== null) meta.content = dark ? "#141517" : "#f5f4ef";
}
