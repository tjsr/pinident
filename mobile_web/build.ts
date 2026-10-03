import { cp, mkdir, rm } from 'node:fs/promises';
import { spawnSync } from 'node:child_process';

await rm('dist', { recursive: true, force: true });
await mkdir('dist', { recursive: true });
for (const project of ['tsconfig.json', 'tsconfig.sw.json']) {
  const result = spawnSync(process.execPath, ['node_modules/typescript/bin/tsc', '-p', project],
    { stdio: 'inherit' });
  if (result.status !== 0) process.exit(result.status ?? 1);
}
for (const asset of ['index.html', 'styles.css', 'manifest.webmanifest',
  'icon.svg', 'icon-192.png', 'icon-512.png']) {
  await cp(asset, `dist/${asset}`);
}
