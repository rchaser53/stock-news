export function envString(name: string, fallback?: string): string | undefined {
  const v = process.env[name];
  if (v === undefined) return fallback;
  const s = v.trim();
  return s === '' ? fallback : s;
}

export function envBool(name: string, fallback: boolean): boolean {
  const raw = (process.env[name] ?? '').trim().toLowerCase();
  if (!raw) return fallback;
  if (['1', 'true', 'yes', 'y', 'on'].includes(raw)) return true;
  if (['0', 'false', 'no', 'n', 'off'].includes(raw)) return false;
  return fallback;
}
