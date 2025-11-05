import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { spawn } from 'node:child_process';

const voicevoxBaseUrl = 'http://localhost:50021';

async function postBinary(url: string, body?: Uint8Array, headers?: Record<string, string>): Promise<Uint8Array> {
  const res = await fetch(url, {
    method: 'POST',
    headers,
    body: body ? Buffer.from(body) : undefined
  });
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw Object.assign(new Error(`APIエラー (status ${res.status}): ${text}`), { status: res.status });
  }
  const ab = await res.arrayBuffer();
  return new Uint8Array(ab);
}

export async function createAudioQuery(text: string, speakerId: number): Promise<Uint8Array> {
  const url = `${voicevoxBaseUrl}/audio_query?text=${encodeURIComponent(text)}&speaker=${speakerId}`;
  return postBinary(url, undefined, { 'content-type': 'application/json' });
}

export async function synthesis(queryJson: Uint8Array, speakerId: number): Promise<Uint8Array> {
  const url = `${voicevoxBaseUrl}/synthesis?speaker=${speakerId}`;
  return postBinary(url, queryJson, { 'content-type': 'application/json' });
}

export async function playAudioWav(audioData: Uint8Array): Promise<void> {
  const tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), 'voicevox_'));
  const wavPath = path.join(tmpDir, `audio_${Date.now()}.wav`);
  await fs.writeFile(wavPath, audioData);

  await new Promise<void>((resolve, reject) => {
    const p = spawn('afplay', [wavPath], { stdio: 'ignore' });
    p.on('error', reject);
    p.on('exit', (code: number | null) => (code === 0 ? resolve() : reject(new Error(`afplay failed (code=${code})`))));
  }).finally(async () => {
    await fs.rm(tmpDir, { recursive: true, force: true });
  });
}

export async function speakWithVoicevox(text: string, speakerId: number): Promise<void> {
  const query = await createAudioQuery(text, speakerId);
  const audio = await synthesis(query, speakerId);
  await playAudioWav(audio);
}
