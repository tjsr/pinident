import { access, cp, mkdir, readFile, rm } from 'node:fs/promises';
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
try {
  await access('model.json');
  await cp('model.json', 'dist/model.json');
} catch { /* A deployment may run before the desktop classifier is trained. */ }
const localReferencePreview = process.argv.includes('--local-index');
const requireReferenceIndex = process.argv.includes('--require-index');
const referenceFile = localReferencePreview
  ? 'identity-index.local.json' : 'identity-index.json';
try {
  const index = JSON.parse(await readFile(referenceFile, 'utf8')) as {
    format?: string; version?: number; catalog_rights_unverified?: number;
    references?: Array<{ rights?: string }> };
  if (index.format !== 'pinident-catalog-reference-index' || index.version !== 1 ||
      !Array.isArray(index.references) || !index.references.length)
    throw new Error('Invalid catalog reference index.');
  if (!localReferencePreview && (index.catalog_rights_unverified !== 0 ||
      index.references.some(reference => !reference.rights ||
        reference.rights.toLowerCase().includes('unverified'))))
    throw new Error('Catalog reference publication requires verified artwork rights.');
  await cp(referenceFile, 'dist/identity-index.json');
} catch (error) {
  if ((error as NodeJS.ErrnoException).code !== 'ENOENT' ||
      localReferencePreview || requireReferenceIndex) throw error;
}
