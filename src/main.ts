import 'dotenv/config';

import fs from 'node:fs/promises';
import type { Dirent } from 'node:fs';
import path from 'node:path';

import { parseArgs } from './cli.js';
import { collectDiffs, getLatestOutputDir, writeDiffReport } from './diff.js';
import { appendEvalRecord, evalRunFilePath } from './eval.js';
import {
  geminiApiVersionFromEnv,
  geminiDefaultChatModel,
  geminiModelFromEnv,
  getLastGeminiApiVersionUsed,
  getStockNews,
  summarizeDiff
} from './gemini.js';
import { speakWithVoicevox } from './voicevox.js';
import { loadConfig } from './yamlConfig.js';

async function readAloudFiles(dirPath: string): Promise<void> {
  // existence
  await fs.stat(dirPath);

  const files: Dirent[] = await fs.readdir(dirPath, { withFileTypes: true });
  const txtFiles = files.filter((f) => !f.isDirectory() && f.name.endsWith('.txt')).map((f) => f.name);
  if (txtFiles.length === 0) throw new Error('読み上げ可能な.txtファイルが見つかりません');

  console.log('\n=== 音声読み上げ開始 ===');
  console.log(`ディレクトリ: ${dirPath}`);
  console.log(`対象ファイル数: ${txtFiles.length}\n`);

  for (let i = 0; i < txtFiles.length; i += 1) {
    const filename = txtFiles[i];
    const filePath = path.join(dirPath, filename);

    console.log(`[${i + 1}/${txtFiles.length}] ${filename} を読み上げ中...`);
    try {
      const content = await fs.readFile(filePath, 'utf-8');
      await speakWithVoicevox(content, 3);
      console.log('  ✓ 完了');
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      console.log(`  エラー: 音声読み上げに失敗 - ${msg}`);
    }
  }

  console.log('\n=== 音声読み上げ完了 ===');
}

async function compareAndReadDiffs(apiKey: string, newDir: string, oldDir: string): Promise<void> {
  console.log('\n=== 差分チェック ===');
  console.log(`新規データ: ${newDir}`);
  console.log(`比較対象: ${oldDir}\n`);

  const { diffs, hasChanges } = await collectDiffs(newDir, oldDir);

  // ログ出し（Go版と同じ雰囲気）
  const newFiles: Dirent[] = await fs.readdir(newDir, { withFileTypes: true });
  for (const file of newFiles) {
    if (file.isDirectory() || !file.name.endsWith('.txt')) continue;
    const diff = diffs.find((d) => d.fileName === file.name);
    if (!diff) {
      console.log(`  ${file.name}: 変更なし`);
    } else if (!diff.oldContent) {
      console.log(`  ${file.name}: 新規ファイル（差分あり）`);
    } else {
      console.log(`  ${file.name}: 差分検出`);
    }
  }

  if (!hasChanges) {
    console.log('\n変更のあるファイルはありませんでした。');
    return;
  }

  console.log('\n=== 差分情報をファイルに保存 ===');
  const diffPath = await writeDiffReport('diffs', oldDir, newDir, diffs);
  console.log(`✓ 差分情報を保存しました: ${diffPath}`);

  console.log(`\n=== 差分を要約して読み上げます（${diffs.length}件） ===\n`);

  for (let i = 0; i < diffs.length; i += 1) {
    const diff = diffs[i];
    console.log(`[${i + 1}/${diffs.length}] ${diff.fileName} の差分を処理中...`);

    let summaryText = '';
    let usedLlm = false;
    let summaryPrompt = '';
    let summaryOnly = '';

    if (!diff.oldContent) {
      summaryText = `${diff.companyName}の新規情報です。${diff.newContent}`;
      console.log('  新規ファイル: 全文を読み上げます');
    } else {
      console.log('  差分を要約中...');
      try {
        const { prompt, summary } = await summarizeDiff(apiKey, diff.companyName, diff.oldContent, diff.newContent);
        usedLlm = true;
        summaryPrompt = prompt;
        summaryOnly = summary;
        summaryText = `${diff.companyName}に関する変更点です。${summary}`;
        console.log('  要約完了');
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        console.log(`  警告: 要約の生成に失敗 - ${msg}`);
        console.log('  元の内容を読み上げます');
        summaryText = `${diff.companyName}に関する更新情報です。${diff.newContent}`;
      }
    }

    if (usedLlm) {
      const evalFile = (process.env.EVAL_RUN_FILE ?? '').trim() || evalRunFilePath();
      process.env.EVAL_RUN_FILE = evalFile;

      await appendEvalRecord(evalFile, {
        id: String(Date.now()) + String(Math.random()).slice(2),
        ts: new Date().toISOString(),
        task: 'diff_summary',
        question: summaryPrompt,
        answer: summaryOnly,
        contexts: [diff.oldContent, diff.newContent],
        meta: {
          company: diff.companyName,
          file: diff.fileName,
          new_dir: newDir,
          old_dir: oldDir,
          model: geminiModelFromEnv('GEMINI_MODEL_SUMMARY', geminiDefaultChatModel())
        }
      });
    }

    console.log('  読み上げ中...');
    try {
      await speakWithVoicevox(summaryText, 3);
      console.log('  ✓ 完了\n');
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      console.log(`  エラー: 音声読み上げに失敗 - ${msg}`);
    }
  }

  console.log('=== 差分読み上げ完了 ===');
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));

  if (args.read) {
    if (!args.dir) {
      throw new Error('読み上げモードでは -dir オプションでディレクトリを指定してください\n使用例: npm run dev -- -read -dir output/2025-12-06');
    }
    await readAloudFiles(args.dir);
    return;
  }

  const apiKey = (process.env.GEMINI_API_KEY ?? '').trim();
  if (!apiKey) throw new Error('環境変数 GEMINI_API_KEY が設定されていません');

  const requestedVersion = geminiApiVersionFromEnv();
  console.log(`Gemini API version: ${requestedVersion}\n`);

  const config = await loadConfig('config.yaml');
  if (config.companies.length === 0) throw new Error('設定ファイルに会社が登録されていません');

  const now = new Date();
  const yyyy = now.getFullYear();
  const mm = String(now.getMonth() + 1).padStart(2, '0');
  const dd = String(now.getDate()).padStart(2, '0');
  const dateStr = `${yyyy}-${mm}-${dd}`;

  const outputDir = path.join('output', dateStr);
  await fs.mkdir(outputDir, { recursive: true });

  console.log('=== 株価関連情報 ===');
  console.log(`出力ディレクトリ: ${outputDir}`);
  console.log('');

  const evalFile = (process.env.EVAL_RUN_FILE ?? '').trim() || evalRunFilePath();
  process.env.EVAL_RUN_FILE = evalFile;
  await fs.mkdir(path.dirname(evalFile), { recursive: true });

  for (let i = 0; i < config.companies.length; i += 1) {
    const company = config.companies[i];
    console.log(`[${i + 1}] ${company.name} (${company.ticker}) を処理中...`);

    try {
      const news = await getStockNews(apiKey, company);

      // ragas評価用JSONL
      const promptForEval = `${company.name}（証券コード: ${company.ticker}）の株価関連ニュースを要約して出力してください。`;
      await appendEvalRecord(evalFile, {
        id: String(Date.now()) + String(Math.random()).slice(2),
        ts: new Date().toISOString(),
        task: 'news_report',
        question: promptForEval,
        answer: news,
        meta: {
          company: company.name,
          ticker: company.ticker,
          ir_url: company.ir_url,
          model: geminiModelFromEnv('GEMINI_MODEL_NEWS', geminiDefaultChatModel())
        }
      });

      const filename = `${company.name}_${company.ticker}.txt`;
      const filePath = path.join(outputDir, filename);

      const jpNow = now.toISOString().replace('T', ' ').slice(0, 19);
      const content = `会社名: ${company.name}\n証券コード: ${company.ticker}\nIRサイト: ${company.ir_url ?? ''}\n取得日時: ${jpNow}\n\n${news}\n`;
      await fs.writeFile(filePath, content, 'utf-8');

      console.log(`  ✓ 完了: ${filePath}`);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      console.log(`  エラー: ${msg}`);
    }
  }

  console.log('\n=== 処理完了 ===');
  const last = getLastGeminiApiVersionUsed();
  if (last) console.log(`Gemini API version used (last success): ${last}`);

  if (args.autoRead) {
    console.log('');
    try {
      const latestDir = await getLatestOutputDir(dateStr);
      await compareAndReadDiffs(apiKey, outputDir, latestDir);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      console.log(`比較対象のディレクトリが見つかりません: ${msg}`);
      console.log('（初回実行のため、比較をスキップします）');
    }
  }
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
