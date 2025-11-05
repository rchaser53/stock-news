import fs from 'node:fs/promises';
import yaml from 'js-yaml';

export type Company = {
  name: string;
  ticker: string;
  ir_url?: string;
};

export type Config = {
  companies: Company[];
};

export async function loadConfig(path: string): Promise<Config> {
  const raw = await fs.readFile(path, 'utf-8');
  const doc = yaml.load(raw);
  if (!doc || typeof doc !== 'object') {
    throw new Error(`設定ファイルのパースに失敗しました: ルートがobjectではありません (${path})`);
  }

  const anyDoc = doc as any;
  const companiesRaw = anyDoc.companies;
  if (!Array.isArray(companiesRaw)) {
    throw new Error(`設定ファイルのパースに失敗しました: companies が配列ではありません (${path})`);
  }

  const companies: Company[] = companiesRaw.map((c: any, i: number) => {
    if (!c || typeof c !== 'object') throw new Error(`companies[${i}] がobjectではありません`);
    const name = typeof c.name === 'string' ? c.name : '';
    const ticker = typeof c.ticker === 'string' ? c.ticker : '';
    const ir_url = typeof c.ir_url === 'string' ? c.ir_url : undefined;
    if (!name.trim() || !ticker.trim()) {
      throw new Error(`companies[${i}] の name/ticker が不正です`);
    }
    return { name: name.trim(), ticker: ticker.trim(), ir_url: ir_url?.trim() };
  });

  return { companies };
}
