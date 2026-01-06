import { envBool, envInt, envString } from './env.js';
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

type GenerateOnceResult = { text: string; finishReason: string };

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

function takeTail(s: string, maxChars: number): string {
  if (maxChars <= 0) return '';
  return s.length <= maxChars ? s : s.slice(Math.max(0, s.length - maxChars));
}

function isMaxTokensFinishReason(finishReason: string): boolean {
  const fr = (finishReason ?? '').trim().toUpperCase();
  if (!fr) return false;
  return fr === 'MAX_TOKENS' || fr === 'MAX_OUTPUT_TOKENS' || fr.includes('MAX_TOKENS');
}

export function geminiDefaultChatModel(): string {
  // rag-playground と同じ env 名を優先
  return envString('GEMINI_CHAT_MODEL', envString('GEMINI_MODEL', 'gemini-3-pro-preview'))!;
}

export function geminiDebugEnabled(): boolean {
  return envBool('GEMINI_DEBUG', false);
}

function isEmptyGeminiResponseError(err: unknown): boolean {
  const msg = err instanceof Error ? err.message : String(err);
  return (
    msg.includes('Gemini応答が空です') ||
    msg.includes('Gemini API応答が空です') ||
    msg.includes('テキスト応答が見つかりません')
  );
}

async function callGenerateContentOnceWithVersionDetailed(
  apiVersion: string,
  apiKey: string,
  model: string,
  prompt: string,
  maxOutputTokens: number,
  enableGoogleSearch: boolean
): Promise<GenerateOnceResult> {
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
  const finishReason = (candidates[0]?.finishReason ?? '').trim();
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
  return { text, finishReason };
}

async function callGenerateContentOnceWithVersion(
  apiVersion: string,
  apiKey: string,
  model: string,
  prompt: string,
  maxOutputTokens: number,
  enableGoogleSearch: boolean
): Promise<string> {
  const r = await callGenerateContentOnceWithVersionDetailed(
    apiVersion,
    apiKey,
    model,
    prompt,
    maxOutputTokens,
    enableGoogleSearch
  );
  return r.text;
}

async function callGenerateContentOnceDetailed(
  apiKey: string,
  model: string,
  prompt: string,
  maxOutputTokens: number,
  enableGoogleSearch: boolean
): Promise<GenerateOnceResult> {
  return callGenerateContentOnceWithVersionDetailed(
    geminiApiVersionFromEnv(),
    apiKey,
    model,
    prompt,
    maxOutputTokens,
    enableGoogleSearch
  );
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

  const autoContinue = envBool('GEMINI_AUTO_CONTINUE', true);
  const maxContinuations = envInt('GEMINI_MAX_CONTINUATIONS', 2);
  const continuationContextChars = envInt('GEMINI_CONTINUATION_CONTEXT_CHARS', 6000);

  let combined = '';
  let promptForCall = prompt;
  let searchForCall = enableGoogleSearch;

  for (let i = 0; i <= maxContinuations; i += 1) {
    const { text, finishReason } = await callGenerateContentOnceDetailed(
      apiKey,
      normalizedModel,
      promptForCall,
      maxOutputTokens,
      searchForCall
    );

    if (!combined) {
      combined = text;
    } else {
      const sep = combined.endsWith('\n') || text.startsWith('\n') ? '' : '\n';
      combined = `${combined}${sep}${text}`;
    }

    if (geminiDebugEnabled()) {
      console.log(
        `[gemini] finishReason=${finishReason || '(none)'} chars=${text.length} combinedChars=${combined.length} cont=${i}/${maxContinuations}`
      );
    }

    if (!autoContinue) return combined;

    if (isMaxTokensFinishReason(finishReason) && i < maxContinuations) {
      const tail = takeTail(combined, continuationContextChars);
      promptForCall =
        `あなたの前回の回答はトークン上限で途中終了しました。\n` +
        `以下の「元の依頼」と「これまでの回答（末尾）」を踏まえて、続きを出力してください。\n` +
        `重複は避け、続きの部分だけを同じ形式で書いてください。\n\n` +
        `## 元の依頼\n${prompt}\n\n` +
        `## これまでの回答（末尾）\n${tail}\n`;
      // 続き生成では検索ツールは不要なことが多いので無効化
      searchForCall = false;
      continue;
    }

    return combined;
  }

  // 失敗時のフォールバック（APIバージョン切替/モデル切替）は行わない。
  return combined;
}

export function geminiModelFromEnv(envKey: string, fallback: string): string {
  return envString(envKey, fallback) ?? fallback;
}

export async function getStockNews(apiKey: string, company: { name: string; ticker: string; ir_url?: string }): Promise<string> {
  const irUrlInfo = company.ir_url
    ? `\n\n**重要**: 必ず以下のIRサイトも確認してください:\n${company.ir_url}`
    : '';

  const prompt = `対象企業（会社名: ${company.name}, 証券コード: ${company.ticker}）について、以下の手順で情報を収集してください。\n` +
    `1. IRサイト（${company.ir_url ?? ''}）から最新のIR情報（決算、開示資料、プレスリリース）を確認\n` +
    `2. Web検索で直近30日以内の株価関連ニュースを調査\n\n` +
    `## 出力形式\n` +
    `- **注意**: 出力本文には会社名や証券コードを書かないでください（ファイル名に含まれるため）。\n` +
    `- 余計な導入文は不要です。以下の見出しから開始してください。\n\n` +
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
  const maxTokens = envInt('GEMINI_NEWS_MAX_OUTPUT_TOKENS', 2500);

  try {
    return await callGeminiGenerateContent(apiKey, newsModel, prompt, maxTokens, enableSearch);
  } catch (err) {
    // googleSearch tool が利用できない環境もあるため、まずツール無しで再試行
    if (enableSearch) {
      try {
        return await callGeminiGenerateContent(apiKey, newsModel, prompt, maxTokens, false);
      } catch (err2) {
        // fall through
        err = err2;
      }
    }

    // まれに status=200 でも candidates/parts が空になることがあるため、別経路で再試行して回避
    if (isEmptyGeminiResponseError(err)) {
      const retryPrompt =
        prompt +
        `\n\n# 追加指示\n` +
        `- 出力は必ずテキストで返してください。\n` +
        `- 情報が取得できない場合でも、各セクションに「該当情報なし」と明記して空にしないでください。\n`;

      try {
        return await callGeminiGenerateContent(apiKey, newsModel, retryPrompt, Math.min(maxTokens, 1800), false);
      } catch {
        // v1 での単発生成も試す（v1betaで稀に空返答になる環境向け）
        try {
          const r = await callGenerateContentOnceWithVersion('v1', apiKey, newsModel, retryPrompt, Math.min(maxTokens, 1800), false);
          if (r.trim()) return r;
        } catch {
          // ignore
        }
      }

      // 最終フォールバック（処理を止めない）
      return (
        `### IRサイトからの最新情報\n` +
        `- 該当情報なし（Geminiの応答が空でした）\n\n` +
        `### Web検索からのニュース\n` +
        `1. 該当ニュースなし\n` +
        `   - 要約: 該当情報なし\n` +
        `   - 出典: \n`
      );
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
