import { envBool, envString } from './env.js';
import { serialize, withRetry } from './retry.js';

export type GeminiGenerateContentRequest = {
  contents: Array<{ role?: string; parts: Array<{ text?: string }> }>;
  tools?: Array<{ googleSearch?: Record<string, unknown> }>;
  generationConfig?: { maxOutputTokens?: number; temperature?: number };
};

type GeminiGenerateContentResponse = {
  candidates?: Array<{ content?: { parts?: Array<{ text?: string }> }; finishReason?: string }>;
  promptFeedback?: { blockReason?: string; blockReasonMessage?: string };
  error?: { code?: number; message?: string; status?: string };
};

let lastGeminiApiVersionUsed = '';

export function getLastGeminiApiVersionUsed(): string {
  return lastGeminiApiVersionUsed;
}

function recordGeminiApiVersionUsed(v: string) {
  lastGeminiApiVersionUsed = v;
}

function normalizeGeminiApiVersion(raw: string): 'v1' | 'v1beta' {
  const v = (raw ?? '').trim().toLowerCase();
  // 実際のAPIエンドポイントは v1 / v1beta を使用する。
  // 過去設定との互換のため v3/3 は v1beta として扱う。
  if (!v) return 'v1beta';
  if (v === 'v1' || v === 'v1beta') return v;
  if (v === 'v3' || v === '3') return 'v1beta';
  return 'v1beta';
}

export function geminiApiVersionFromEnv(): string {
  return normalizeGeminiApiVersion(process.env.GEMINI_API_VERSION ?? '');
}

function normalizeModelName(model: string): string {
  const m = (model ?? '').trim();
  return m.startsWith('models/') ? m.slice('models/'.length) : m;
}

function truncateForError(s: string, max: number): string {
  if (max <= 0) return '';
  return s.length <= max ? s : `${s.slice(0, max)}...`;
}

export function geminiDefaultChatModel(): string {
  // rag-playground と同じ env 名を優先
  return envString('GEMINI_CHAT_MODEL', envString('GEMINI_MODEL', 'gemini-3-pro-preview'))!;
}

export function geminiDebugEnabled(): boolean {
  return envBool('GEMINI_DEBUG', false);
}

async function callGenerateContentOnceWithVersion(
  apiVersion: string,
  apiKey: string,
  model: string,
  prompt: string,
  maxOutputTokens: number,
  enableGoogleSearch: boolean
): Promise<string> {
  const version = normalizeGeminiApiVersion(apiVersion);
  const normalizedModel = normalizeModelName(model);
  if (!apiKey.trim()) throw new Error('GEMINI_API_KEY が空です');
  if (!normalizedModel) throw new Error('Gemini model が空です');

  const reqBody: GeminiGenerateContentRequest = {
    contents: [{ role: 'user', parts: [{ text: prompt }] }],
    generationConfig: { maxOutputTokens }
  };
  if (enableGoogleSearch && version !== 'v1') {
    reqBody.tools = [{ googleSearch: {} }];
  }

  const endpoint = `https://generativelanguage.googleapis.com/${version}/models/${encodeURIComponent(
    normalizedModel
  )}:generateContent?key=${encodeURIComponent(apiKey)}`;

  const res = await serialize(() =>
    withRetry(() =>
      fetch(endpoint, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify(reqBody)
      })
    )
  );

  const body = await res.text();
  if (!body) throw new Error(`Gemini API応答が空です (status ${res.status}, version=${version}, model=${normalizedModel})`);

  let parsed: GeminiGenerateContentResponse;
  try {
    parsed = JSON.parse(body) as GeminiGenerateContentResponse;
  } catch (err) {
    throw new Error(`Gemini generateContent JSONパース失敗 (status ${res.status}, body=${JSON.stringify(body)}): ${String(err)}`);
  }

  if (!res.ok) {
    const msg = parsed?.error?.message ?? body;
    throw Object.assign(new Error(`Gemini APIエラー (status ${res.status}): ${msg}`), { status: res.status });
  }
  if (parsed.error?.message) throw new Error(`Gemini APIエラー: ${parsed.error.message}`);

  const candidates = parsed.candidates ?? [];
  const parts = candidates[0]?.content?.parts ?? [];
  if (candidates.length === 0 || parts.length === 0) {
    const br = parsed.promptFeedback?.blockReason?.trim() ?? '';
    const msg = parsed.promptFeedback?.blockReasonMessage?.trim() ?? '';
    if (br || msg) {
      throw new Error(msg ? `Gemini応答がブロックされました (blockReason=${br}): ${msg}` : `Gemini応答がブロックされました (blockReason=${br})`);
    }
    if (geminiDebugEnabled()) {
      throw new Error(
        `Gemini応答が空です (version=${version}, model=${normalizedModel}, status=${res.status}, body=${JSON.stringify(
          truncateForError(body, 1200)
        )})`
      );
    }
    throw new Error(`Gemini応答が空です (version=${version}, model=${normalizedModel}, status=${res.status})`);
  }

  const text = parts.map((p) => p.text ?? '').join('').trim();
  if (!text) throw new Error('テキスト応答が見つかりません');

  recordGeminiApiVersionUsed(version);
  return text;
}

async function callGenerateContentOnce(
  apiKey: string,
  model: string,
  prompt: string,
  maxOutputTokens: number,
  enableGoogleSearch: boolean
): Promise<string> {
  return callGenerateContentOnceWithVersion(geminiApiVersionFromEnv(), apiKey, model, prompt, maxOutputTokens, enableGoogleSearch);
}

export async function callGeminiGenerateContent(
  apiKey: string,
  model: string,
  prompt: string,
  maxOutputTokens: number,
  enableGoogleSearch: boolean
): Promise<string> {
  const normalizedModel = normalizeModelName(model);

  // 失敗時のフォールバック（APIバージョン切替/モデル切替）は行わない。
  return await callGenerateContentOnce(apiKey, normalizedModel, prompt, maxOutputTokens, enableGoogleSearch);
}

export function geminiModelFromEnv(envKey: string, fallback: string): string {
  return envString(envKey, fallback) ?? fallback;
}

export async function getStockNews(apiKey: string, company: { name: string; ticker: string; ir_url?: string }): Promise<string> {
  const irUrlInfo = company.ir_url
    ? `\n\n**重要**: 必ず以下のIRサイトも確認してください:\n${company.ir_url}`
    : '';

  const prompt = `${company.name}（証券コード: ${company.ticker}）について、以下の手順で情報を収集してください:\n\n` +
    `1. IRサイト（${company.ir_url ?? ''}）から最新のIR情報（決算、開示資料、プレスリリース）を確認\n` +
    `2. Web検索で直近30日以内の株価関連ニュースを調査\n\n` +
    `## 出力形式\n` +
    `### IRサイトからの最新情報\n` +
    `- 日付と内容を箇条書き（最大3件）\n\n` +
    `### Web検索からのニュース\n` +
    `1. **記事タイトル** (YYYY-MM-DD)\n` +
    `   - 要約: [株価への影響を中心に]\n` +
    `   - 出典: [URL]\n\n` +
    `IRサイトに情報がない、またはアクセスできない場合はその旨を記載。\n` +
    `代わりにhttps://irbank.net/からの情報を参考にしてください。\n` +
    `Web検索でもニュースが見つからない場合は「該当ニュースなし」と記載。${irUrlInfo}`;

  const newsModel = geminiModelFromEnv('GEMINI_MODEL_NEWS', geminiDefaultChatModel());
  const enableSearch = envBool('GEMINI_ENABLE_GOOGLE_SEARCH', true);

  try {
    return await callGeminiGenerateContent(apiKey, newsModel, prompt, 1500, enableSearch);
  } catch (err) {
    // googleSearch tool が利用できない環境もあるため、フォールバックでツール無し再試行
    if (enableSearch) {
      return await callGeminiGenerateContent(apiKey, newsModel, prompt, 1500, false);
    }
    throw err;
  }
}

export async function summarizeDiff(
  apiKey: string,
  companyName: string,
  oldContent: string,
  newContent: string
): Promise<{ prompt: string; summary: string }> {
  const prompt = `以下は${companyName}に関する株価情報の旧版と新版です。\n` +
    `変更点を簡潔に要約してください（200文字以内）。\n\n` +
    `## 旧版の内容\n${oldContent}\n\n` +
    `## 新版の内容\n${newContent}\n\n` +
    `## 要約の形式\n` +
    `変更点を箇条書きで3点以内にまとめてください。株価に影響する重要な情報を優先してください。`;

  const summaryModel = geminiModelFromEnv('GEMINI_MODEL_SUMMARY', geminiDefaultChatModel());
  const summary = await callGeminiGenerateContent(apiKey, summaryModel, prompt, 300, false);
  return { prompt, summary };
}
