const STORAGE_KEY = "spatialdino.clientId";

function generateId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return `cid_${Math.random().toString(16).slice(2)}_${Date.now().toString(16)}`;
}

export function getClientId(): string {
  try {
    const existing = window.localStorage.getItem(STORAGE_KEY);
    if (existing && existing.trim().length > 0) return existing.trim();
    const next = generateId();
    window.localStorage.setItem(STORAGE_KEY, next);
    return next;
  } catch {
    return generateId();
  }
}
