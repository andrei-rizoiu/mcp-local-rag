// INT-1: `handleListFiles(scope)` MCP-handler integration (Phase 2 Task 2).
//
// AC: AC3 (scan-path basis + alias contract) / AC6 (raw-data sources always
//     kept; real-file entries scope-filtered from both files[] and sources[]).
// Behavior: `handleListFiles({ scope })` over a real LanceDB + real-FS fixture
//     → BFS pushdown prunes scope-outside subtrees → scoped files[], raw-data
//     sources retained, out-of-scope real-file orphan excluded, symlink-alias
//     entry included with its stored filePath spelling.
//
// DB and FS are real.

import { mkdirSync, mkdtempSync, rmSync, symlinkSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import { RAGServer } from '../../server/index.js'
import { testModelCacheDir, withTestDevice } from '../test-device.js'

// Directory symlinks require admin/developer mode on windows-latest (in the CI
// matrix); probe support once so the alias contract describe is skipped there
// rather than failing the job on an environment limitation.
function directorySymlinkSupported(): boolean {
  const probeBase = mkdtempSync(join(tmpdir(), 'list-scope-symlink-probe-'))
  try {
    const target = join(probeBase, 'target')
    mkdirSync(target)
    symlinkSync(target, join(probeBase, 'link'), 'dir')
    return true
  } catch {
    return false
  } finally {
    rmSync(probeBase, { recursive: true, force: true })
  }
}

describe('INT-1: handleListFiles(scope) — scoped files, sources split', () => {
  const base = resolve('./tmp/test-list-files-scope-int')
  const dataDir = join(base, 'data')
  const dbPath = join(base, 'lancedb')
  const cacheDir = join(base, 'cache')

  const inScopeDir = join(dataDir, 'in-scope')
  const inScopeFile = join(inScopeDir, 'keep.txt')
  const deepDir = join(inScopeDir, 'deep')
  const deepFile = join(deepDir, 'deep-keep.txt')

  const outScopeDir = join(dataDir, 'out-scope')
  const outScopeUningested = join(outScopeDir, 'uningested.txt')
  const outScopeIngested = join(outScopeDir, 'ingested-out.txt')

  const rawSource = 'https://example.com/int1-raw-source'

  let server: RAGServer

  beforeAll(async () => {
    mkdirSync(inScopeDir, { recursive: true })
    mkdirSync(deepDir, { recursive: true })
    mkdirSync(outScopeDir, { recursive: true })
    mkdirSync(dbPath, { recursive: true })
    mkdirSync(cacheDir, { recursive: true })

    writeFileSync(inScopeFile, 'In-scope file content. '.repeat(20))
    writeFileSync(deepFile, 'Deep in-scope file content. '.repeat(20))
    writeFileSync(outScopeUningested, 'Out-of-scope UNINGESTED file. '.repeat(20))
    writeFileSync(outScopeIngested, 'Out-of-scope ingested file. '.repeat(20))

    server = new RAGServer(
      withTestDevice({
        dbPath,
        modelName: 'Xenova/all-MiniLM-L6-v2',
        cacheDir: testModelCacheDir(cacheDir),
        baseDir: dataDir,
        maxFileSize: 100 * 1024 * 1024,
      })
    )
    await server.initialize()

    // Ingest the two in-scope files and the out-of-scope real file, plus one
    // raw-data source. `uningested.txt` is deliberately NOT ingested.
    await server.handleIngestFile({ filePath: inScopeFile })
    await server.handleIngestFile({ filePath: deepFile })
    await server.handleIngestFile({ filePath: outScopeIngested })
    await server.handleIngestData({
      content:
        'Raw-data content ingested via ingest_data for the INT-1 scope test. ' +
        'Long enough to produce at least one chunk in the vector store.',
      metadata: { source: rawSource, format: 'text' },
    })
  }, 120000)

  afterAll(async () => {
    await server.close()
    rmSync(base, { recursive: true, force: true })
  })

  it('restricts files[] to the in-scope subtree, keeps raw-data sources, and drops out-of-scope real-file sources', async () => {
    const result = await server.handleListFiles({ scope: inScopeDir })
    const parsed = JSON.parse(result.content[0].text)
    const filePaths: string[] = parsed.files.map((f: { filePath: string }) => f.filePath)

    expect(filePaths).toContain(inScopeFile)
    expect(filePaths).toContain(deepFile)
    expect(filePaths).not.toContain(outScopeUningested)
    expect(filePaths).not.toContain(outScopeIngested)

    // Raw-data sources are kept regardless of scope.
    const sourceEntry = parsed.sources.find((s: { source?: string }) => s.source === rawSource)
    expect(sourceEntry).toBeDefined()

    // An out-of-scope real-file ingested entry is excluded from sources[] too
    // (AC6 negative).
    const sourcePaths: string[] = parsed.sources.map((s: { filePath?: string }) => s.filePath)
    expect(sourcePaths).not.toContain(outScopeIngested)
  })
})

// AC3 path-basis / symlink-alias contract: under an aliased (dir-symlinked)
// baseDir, an ingested file whose SCAN path is under scope but whose stored
// filePath uses a different (real) spelling is included, and the returned
// filePath keeps its stored spelling (scope is guaranteed over the reachable
// scan path, not the stored spelling).
const describeAlias = directorySymlinkSupported() ? describe : describe.skip

describeAlias('INT-1: handleListFiles(scope) — symlink-alias contract (AC3)', () => {
  const base = resolve('./tmp/test-list-files-scope-alias')
  const realRoot = join(base, 'real')
  const aliasRoot = join(base, 'alias')
  const realSubDir = join(realRoot, 'sub')
  const realFile = join(realSubDir, 'aliased.txt')
  const dbPath = join(base, 'lancedb')
  const cacheDir = join(base, 'cache')

  let server: RAGServer

  beforeAll(async () => {
    mkdirSync(realSubDir, { recursive: true })
    mkdirSync(dbPath, { recursive: true })
    mkdirSync(cacheDir, { recursive: true })
    writeFileSync(realFile, 'Aliased file content. '.repeat(20))

    // Symlink support was confirmed by `directorySymlinkSupported` before this
    // describe was selected, so the dir symlink creation here is expected to
    // succeed.
    symlinkSync(realRoot, aliasRoot, 'dir')

    server = new RAGServer(
      withTestDevice({
        dbPath,
        modelName: 'Xenova/all-MiniLM-L6-v2',
        cacheDir: testModelCacheDir(cacheDir),
        // Scan through the aliased (symlinked) root; the walker follows the root
        // symlink so scanned paths use the alias spelling.
        baseDir: aliasRoot,
        maxFileSize: 100 * 1024 * 1024,
      })
    )
    await server.initialize()

    // Ingest via the REAL spelling so the stored filePath differs from the scan
    // path; realpath containment (real path under realpath(aliasRoot)) permits it.
    await server.handleIngestFile({ filePath: realFile })
  }, 120000)

  afterAll(async () => {
    if (server) {
      await server.close()
    }
    rmSync(base, { recursive: true, force: true })
  })

  it('includes an aliased ingested file (scan path under scope) with its stored filePath spelling', async () => {
    const aliasScopeDir = join(aliasRoot, 'sub')
    const result = await server.handleListFiles({ scope: aliasScopeDir })
    const parsed = JSON.parse(result.content[0].text)

    const filePaths: string[] = parsed.files.map((f: { filePath: string }) => f.filePath)
    // Stored (real) spelling is returned, not the alias scan spelling.
    expect(filePaths).toContain(realFile)
    expect(filePaths).not.toContain(join(aliasScopeDir, 'aliased.txt'))

    const entry = parsed.files.find((f: { filePath: string }) => f.filePath === realFile)
    expect(entry.ingested).toBe(true)
  })
})
