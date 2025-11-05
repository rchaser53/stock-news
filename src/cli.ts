export type CliArgs = {
  read: boolean;
  dir?: string;
  autoRead: boolean;
};

export function parseArgs(argv: string[]): CliArgs {
  const out: CliArgs = { read: false, autoRead: false, dir: undefined };

  for (let i = 0; i < argv.length; i += 1) {
    const a = argv[i];

    if (a === '-read' || a === '--read') {
      out.read = true;
      continue;
    }
    if (a === '-auto-read' || a === '--auto-read' || a === '--autoread' || a === '-autoread') {
      out.autoRead = true;
      continue;
    }

    if (a === '-dir' || a === '--dir') {
      const v = argv[i + 1];
      if (v && !v.startsWith('-')) {
        out.dir = v;
        i += 1;
      }
      continue;
    }

    if (a.startsWith('-dir=')) {
      out.dir = a.slice('-dir='.length);
      continue;
    }
    if (a.startsWith('--dir=')) {
      out.dir = a.slice('--dir='.length);
      continue;
    }
  }

  return out;
}
