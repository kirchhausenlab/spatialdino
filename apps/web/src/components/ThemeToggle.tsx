import { useEffect, useMemo, useState } from "react";

type Theme = "light" | "dark";

const STORAGE_KEY = "spatialdino.theme";

function getSystemTheme(): Theme {
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

type ThemePreference = {
  theme: Theme;
  persisted: boolean;
};

function readStoredTheme(): Theme | null {
  try {
    const stored = window.localStorage.getItem(STORAGE_KEY);
    return stored === "light" || stored === "dark" ? stored : null;
  } catch (error) {
    console.warn("[SpatialDINO] Unable to read theme from localStorage:", error);
    return null;
  }
}

function writeStoredTheme(theme: Theme) {
  try {
    window.localStorage.setItem(STORAGE_KEY, theme);
  } catch (error) {
    console.warn("[SpatialDINO] Unable to persist theme to localStorage:", error);
  }
}

function applyThemeToDom(theme: Theme) {
  document.documentElement.dataset.theme = theme;
}

function getInitialThemePreference(): ThemePreference {
  const stored = readStoredTheme();
  if (stored) return { theme: stored, persisted: true };
  return { theme: getSystemTheme(), persisted: false };
}

export default function ThemeToggle() {
  const [pref, setPref] = useState<ThemePreference>(() => getInitialThemePreference());

  useEffect(() => {
    applyThemeToDom(pref.theme);
    if (pref.persisted) writeStoredTheme(pref.theme);
  }, [pref.persisted, pref.theme]);

  useEffect(() => {
    if (pref.persisted) return;
    const mq = window.matchMedia?.("(prefers-color-scheme: dark)");
    if (!mq) return;

    const handleChange = () => {
      setPref({ theme: mq.matches ? "dark" : "light", persisted: false });
    };

    mq.addEventListener("change", handleChange);
    return () => mq.removeEventListener("change", handleChange);
  }, [pref.persisted]);

  const nextTheme = pref.theme === "dark" ? "light" : "dark";
  const label = useMemo(() => `Switch to ${nextTheme} mode`, [nextTheme]);

  return (
    <button
      type="button"
      className="themeToggle"
      onClick={() => setPref({ theme: nextTheme, persisted: true })}
      aria-pressed={pref.theme === "dark"}
      aria-label={label}
      title={label}
    >
      {pref.theme === "dark" ? <SunIcon /> : <MoonIcon />}
    </button>
  );
}

function MoonIcon() {
  return (
    <svg
      viewBox="0 0 24 24"
      width="18"
      height="18"
      role="img"
      aria-hidden="true"
      focusable="false"
    >
      <path
        d="M21 14.6A8.5 8.5 0 0 1 9.4 3 7.5 7.5 0 1 0 21 14.6Z"
        fill="currentColor"
      />
    </svg>
  );
}

function SunIcon() {
  return (
    <svg
      viewBox="0 0 24 24"
      width="18"
      height="18"
      role="img"
      aria-hidden="true"
      focusable="false"
    >
      <path
        d="M12 18a6 6 0 1 0 0-12 6 6 0 0 0 0 12Zm0-16a1 1 0 0 1 1 1v1a1 1 0 1 1-2 0V3a1 1 0 0 1 1-1Zm0 18a1 1 0 0 1 1 1v1a1 1 0 1 1-2 0v-1a1 1 0 0 1 1-1ZM4 11a1 1 0 0 1 0 2H3a1 1 0 1 1 0-2h1Zm18 0a1 1 0 0 1 0 2h-1a1 1 0 1 1 0-2h1ZM6.2 6.2a1 1 0 0 1 1.4 0l.7.7A1 1 0 1 1 6.9 8.3l-.7-.7a1 1 0 0 1 0-1.4Zm10.2 10.2a1 1 0 0 1 1.4 0l.7.7a1 1 0 1 1-1.4 1.4l-.7-.7a1 1 0 0 1 0-1.4ZM17.8 6.2a1 1 0 0 1 0 1.4l-.7.7A1 1 0 1 1 15.7 6.9l.7-.7a1 1 0 0 1 1.4 0ZM8.3 15.7a1 1 0 0 1 0 1.4l-.7.7a1 1 0 1 1-1.4-1.4l.7-.7a1 1 0 0 1 1.4 0Z"
        fill="currentColor"
      />
    </svg>
  );
}
