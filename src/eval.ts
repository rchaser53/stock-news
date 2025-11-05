import fs from 'node:fs/promises';
import path from 'node:path';

export type EvalRecord = {
  id: string;
  ts: string;
  task: string;
  question: string;
  answer: string;
  contexts?: string[];
  ground_truth?: string;
  meta?: Record<string, unknown>;
};

export function evalOutputDir(): string {
  const v = (process.env.EVAL_OUTPUT_DIR ?? '').trim();
  return v || 'eval_runs';
}

export function evalRunFilePath(): string {
  const d = new Date();
  const pad = (n: number) => String(n).padStart(2, '0');
  const ts = `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}_${pad(d.getHours())}${pad(
    d.getMinutes()
  )}${pad(d.getSeconds())}`;
  return path.join(evalOutputDir(), `stock-news_${ts}.jsonl`);
}

export async function appendEvalRecord(filePath: string, rec: EvalRecord): Promise<void> {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  const line = JSON.stringify(rec) + '\n';
  await fs.appendFile(filePath, line, 'utf-8');
}
