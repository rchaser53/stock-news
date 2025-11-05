function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function envInt(name: string, fallback: number): number {
  const raw = process.env[name];
  if (!raw) return fallback;
  const n = Number(raw);
  return Number.isFinite(n) ? n : fallback;
}

function isRetryableError(err: unknown): boolean {
  if (!err || typeof err !== 'object') return false;

  const anyErr = err as {
    status?: unknown;
    message?: unknown;
    name?: unknown;
    code?: unknown;
    error?: unknown;
  };

  const status = typeof anyErr.status === 'number' ? anyErr.status : undefined;
  const message = typeof anyErr.message === 'string' ? anyErr.message : '';
  const name = typeof anyErr.name === 'string' ? anyErr.name : '';
  const code = typeof anyErr.code === 'string' ? anyErr.code : '';

  const nestedCode =
    anyErr.error && typeof anyErr.error === 'object' && typeof (anyErr.error as any).code === 'string'
      ? ((anyErr.error as any).code as string)
      : '';

  if (status === 429) return true;
  if (status === 500 || status === 502 || status === 503 || status === 504) return true;

  const haystack = `${name} ${code} ${nestedCode} ${message}`.toLowerCase();
  if (haystack.includes('rate limit')) return true;
  if (haystack.includes('resource_exhausted')) return true;
  if (haystack.includes('too many requests')) return true;
  if (haystack.includes('overloaded')) return true;

  return false;
}

export type RetryOptions = {
  maxRetries?: number; // retries in addition to the first attempt
  initialBackoffMs?: number;
  maxBackoffMs?: number;
  jitterRatio?: number; // 0..1
  spacingMs?: number; // delay before each attempt
};

export async function withRetry<T>(fn: () => Promise<T>, options: RetryOptions = {}): Promise<T> {
  const maxRetries = options.maxRetries ?? envInt('MAX_RETRIES', 6);
  const initialBackoffMs = options.initialBackoffMs ?? envInt('INITIAL_BACKOFF_MS', 1500);
  const maxBackoffMs = options.maxBackoffMs ?? envInt('MAX_BACKOFF_MS', 20000);
  const jitterRatio = options.jitterRatio ?? 0.2;
  const spacingMs = options.spacingMs ?? envInt('REQUEST_SPACING_MS', 0);

  let attempt = 0;
  // attempt=0 is the first try, then up to maxRetries additional retries
  // total tries = 1 + maxRetries
  while (true) {
    if (spacingMs > 0) await sleep(spacingMs);

    try {
      return await fn();
    } catch (err) {
      attempt += 1;
      const canRetry = isRetryableError(err);
      if (!canRetry || attempt > maxRetries) throw err;

      const exp = Math.min(maxBackoffMs, initialBackoffMs * 2 ** (attempt - 1));
      const jitter = exp * jitterRatio * (Math.random() * 2 - 1); // +/-
      const waitMs = Math.max(0, Math.round(exp + jitter));
      await sleep(waitMs);
    }
  }
}

// Serialize calls (concurrency=1) to avoid bursty traffic.
let queue: Promise<unknown> = Promise.resolve();
export function serialize<T>(fn: () => Promise<T>): Promise<T> {
  const run = queue.then(fn, fn);
  queue = run.catch(() => undefined);
  return run;
}
