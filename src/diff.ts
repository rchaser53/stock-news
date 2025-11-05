import fs from 'node:fs/promises';
import type { Dirent } from 'node:fs';
import path from 'node:path';

export type DiffInfo = {
  filePath: string;
  fileName: string;
  companyName: string;
  oldContent: string;
  newContent: string;
};

export function extractNewsContent(content: string): string {
  const lines = content.split('\n');
  const newsLines: string[] = [];
  let skipHeader = true;

  for (const line of lines) {
    if (skipHeader) {
      if (line.trim() === '' && newsLines.length === 0) {
        skipHeader = false;
      }
      continue;
    }
    newsLines.push(line);
  }

  return newsLines.join('\n').trim();
}

export async function getLatestOutputDir(excludeDate: string): Promise<string> {
  const outputBase = 'output';
  const entries: Dirent[] = await fs.readdir(outputBase, { withFileTypes: true });
  const dirs = entries
    .filter((e) => e.isDirectory())
    .map((e) => e.name)
    .filter((name) => name !== excludeDate)
    .sort((a, b) => (a > b ? -1 : a < b ? 1 : 0));

  if (dirs.length === 0) {
    throw new Error('比較対象のディレクトリがありません');
  }

  return path.join(outputBase, dirs[0]);
}

export async function collectDiffs(newDir: string, oldDir: string): Promise<{ diffs: DiffInfo[]; hasChanges: boolean }> {
  const newFiles = await fs.readdir(newDir, { withFileTypes: true });
  let hasChanges = false;
  const diffs: DiffInfo[] = [];

  for (const file of newFiles) {
    if (file.isDirectory() || !file.name.endsWith('.txt')) continue;

    const newFilePath = path.join(newDir, file.name);
    const oldFilePath = path.join(oldDir, file.name);

    const newContentRaw = await fs.readFile(newFilePath, 'utf-8');
    const companyName = file.name.split('_')[0] ?? file.name;

    try {
      const oldContentRaw = await fs.readFile(oldFilePath, 'utf-8');
      const newNews = extractNewsContent(newContentRaw);
      const oldNews = extractNewsContent(oldContentRaw);

      if (newNews !== oldNews) {
        hasChanges = true;
        diffs.push({
          filePath: newFilePath,
          fileName: file.name,
          companyName,
          oldContent: oldNews,
          newContent: newNews
        });
      }
    } catch {
      // 新規ファイル扱い
      hasChanges = true;
      diffs.push({
        filePath: newFilePath,
        fileName: file.name,
        companyName,
        oldContent: '',
        newContent: extractNewsContent(newContentRaw)
      });
    }
  }

  return { diffs, hasChanges };
}

export async function writeDiffReport(diffDir: string, oldDir: string, newDir: string, diffs: DiffInfo[]): Promise<string> {
  await fs.mkdir(diffDir, { recursive: true });

  const now = new Date();
  const ts = now
    .toISOString()
    .replace(/T/, '_')
    .replace(/:/g, '-')
    .slice(0, 19);

  const outPath = path.join(diffDir, `diff_${ts}.txt`);

  const lines: string[] = [];
  const jpNow = now.toISOString().replace('T', ' ').slice(0, 19);
  lines.push(`差分検出日時: ${jpNow}`);
  lines.push(`比較元: ${oldDir}`);
  lines.push(`比較先: ${newDir}`);
  lines.push(`差分ファイル数: ${diffs.length}`);
  lines.push('');
  lines.push('=== 差分詳細 ===');
  lines.push('');

  diffs.forEach((d, idx) => {
    lines.push(`[${idx + 1}] ${d.fileName}`);
    lines.push(`会社名: ${d.companyName}`);
    if (!d.oldContent) {
      lines.push('状態: 新規ファイル');
      lines.push('');
      lines.push('内容:');
      lines.push(d.newContent);
    } else {
      lines.push('状態: 更新');
      lines.push('');
      lines.push('【旧版】');
      lines.push(d.oldContent);
      lines.push('');
      lines.push('【新版】');
      lines.push(d.newContent);
    }
    lines.push('');
    lines.push('='.repeat(60));
    lines.push('');
  });

  await fs.writeFile(outPath, lines.join('\n'), 'utf-8');
  return outPath;
}
