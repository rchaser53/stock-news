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

type GeminiListModelsResponse = {
  models?: Array<{ name?: string; supportedGenerationMethods?: string[] }>;
  nextPageToken?: string;
  error?: { code?: number; message?: string; status?: string };
};

let lastGeminiApiVersionUsed = '';

export function getLastGeminiApiVersionUsed(): string {
  return lastGeminiApiVersionUsed;
}

function recordGeminiApiVersionUsed(v: string) {
  lastGeminiApiVersionUsed = v;
}

export function geminiApiVersionFromEnv(): string {
  const v = (process.env.GEMINI_API_VERSION ?? '').trim().toLowerCase();
  if (!v) return 'v3';
  if (v === '3') return 'v3';
  if (v !== 'v1' && v !== 'v1beta' && v !== 'v3') return 'v3';
  return v;
}

function effectiveGeminiApiVersion(requested: string): string {
  const v = (requested ?? '').trim().toLowerCase() || 'v3';
  return v === 'v3' ? 'v1beta' : v;
}

function geminiApiVersionFallbackOrder(requested: string): string[] {
  const effective = effectiveGeminiApiVersion(requested);
  const order: string[] = [effective];
  const add = (v: string) => {
    if (!order.includes(v)) order.push(v);
  };
  add('v1beta');
  add('v1');
  return order;
}

function normalizeModelName(model: string): string {
  const m = (model ?? '').trim();
  return m.startsWith('models/') ? m.slice('models/'.length) : m;
}

function isApiVersionNotSupportedError(err: unknown): boolean {
  const msg = err instanceof Error ? err.message.toLowerCase() : String(err).toLowerCase();
  return msg.includes('404') || msg.includes('not found') || msg.includes('unknown') || msg.includes('unimplemented');
}

function isModelNotFoundError(err: unknown): boolean {
  const msg = err instanceof Error ? err.message.toLowerCase() : String(err).toLowerCase();
  return msg.includes('not found') || msg.includes('not_supported') || msg.includes('not supported') || msg.includes('model_not_found');
}

function isGeminiEmptyResponseError(err: unknown): boolean {
  const msg = err instanceof Error ? err.message.toLowerCase() : String(err).toLowerCase();
  return msg.includes('応答が空') || msg.includes('テキスト応答が見つかりません') || msg.includes('empty response');
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

async function listGeminiModels(apiKey: string, apiVersion: string): Promise<string[]> {
  if (!apiKey.trim()) throw new Error('GEMINI_API_KEY が空です');

  let lastErr: unknown;
  for (const v of geminiApiVersionFallbackOrder(apiVersion)) {
    const endpoint = `https://generativelanguage.googleapis.com/${v}/models?key=${encodeURIComponent(apiKey)}`;
    try {
      const res = await serialize(() => withRetry(() => fetch(endpoint, { method: 'GET' })));
      const body = await res.text();
      if (!body) throw new Error(`Gemini ListModels応答が空です (status ${res.status})`);

      const parsed = JSON.parse(body) as GeminiListModelsResponse;
      if (!res.ok) {
        const msg = parsed?.error?.message ?? body;
        throw Object.assign(new Error(`Gemini ListModelsエラー (status ${res.status}): ${msg}`), { status: res.status });
      }
      if (parsed.error?.message) throw new Error(`Gemini ListModelsエラー: ${parsed.error.message}`);

      const models = (parsed.models ?? [])
        .filter((m) => (m.supportedGenerationMethods ?? []).includes('generateContent'))
        .map((m) => normalizeModelName(m.name ?? ''))
        .filter((m) => m);

      return models;
    } catch (err) {
      lastErr = err;
      if (isApiVersionNotSupportedError(err)) continue;
    }
  }
  throw lastErr instanceof Error ? lastErr : new Error('Gemini ListModelsに失敗しました');
}

function pickPreferredGeminiModel(models: string[]): string {
  for (const m of models) if (m.includes('flash')) return m;
  for (const m of models) if (m.includes('pro')) return m;
  return models[0] ?? '';
}

function pickPreferredGeminiModelExcluding(models: string[], exclude: string): string {
  const normalizedExclude = normalizeModelName(exclude);
  const filtered = models.filter((m) => normalizeModelName(m) !== normalizedExclude);
  return pickPreferredGeminiModel(filtered);
}

async function callGenerateContentOnceWithVersion(
  apiVersion: string,
  apiKey: string,
  model: string,
  prompt: string,
  maxOutputTokens: number,
  enableGoogleSearch: boolean
): Promise<string> {
  const version = effectiveGeminiApiVersion(apiVersion);
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

  try {
    return await callGenerateContentOnce(apiKey, normalizedModel, prompt, maxOutputTokens, enableGoogleSearch);
  } catch (err) {
    // APIバージョン未対応フォールバック
    if (isApiVersionNotSupportedError(err)) {
      const order = geminiApiVersionFallbackOrder(geminiApiVersionFromEnv()).slice(1);
      for (const v of order) {
        try {
          return await callGenerateContentOnceWithVersion(v, apiKey, normalizedModel, prompt, maxOutputTokens, enableGoogleSearch);
        } catch {
          // continue
        }
      }
    }

    // googleSearch tool が使えない環境向け: tools無しで再試行（ただし呼び出し側でやることもある）

    // モデル not found フォールバック
    if (isModelNotFoundError(err)) {
      if (!normalizedModel.endsWith('-latest')) {
        try {
          return await callGenerateContentOnce(apiKey, `${normalizedModel}-latest`, prompt, maxOutputTokens, enableGoogleSearch);
        } catch {
          // continue
        }
      }

      try {
        const models = await listGeminiModels(apiKey, geminiApiVersionFromEnv());
        const chosen = pickPreferredGeminiModel(models);
        if (chosen && chosen !== normalizedModel) {
          return await callGenerateContentOnce(apiKey, chosen, prompt, maxOutputTokens, enableGoogleSearch);
        }
      } catch {
        // ignore
      }
    }

    // 200 でも candidates 空があり得る: 利用可能モデルへ再試行
    if (isGeminiEmptyResponseError(err)) {
      try {
        const models = await listGeminiModels(apiKey, geminiApiVersionFromEnv());
        // まず、要求モデルが利用可能一覧に無い場合は最適モデルへ
        if (!models.includes(normalizedModel)) {
          const chosen = pickPreferredGeminiModel(models);
          if (chosen && chosen !== normalizedModel) {
            return await callGenerateContentOnce(apiKey, chosen, prompt, maxOutputTokens, enableGoogleSearch);
          }
        }

        // 要求モデルが存在していても「空応答」になることがあるため、別モデルへ切り替えて再試行する
        const alternative = pickPreferredGeminiModelExcluding(models, normalizedModel);
        if (alternative && alternative !== normalizedModel) {
          return await callGenerateContentOnce(apiKey, alternative, prompt, maxOutputTokens, enableGoogleSearch);
        }

        // それでもダメなら、一覧の2番目以降を順に軽く試す（最大3つ）
        const rest = models.filter((m) => m !== normalizedModel);
        for (const m of rest.slice(0, 3)) {
          try {
            return await callGenerateContentOnce(apiKey, m, prompt, maxOutputTokens, enableGoogleSearch);
          } catch {
            // continue
          }
        }
      } catch {
        // ignore
      }
    }

    throw err instanceof Error ? err : new Error(String(err));
  }
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
