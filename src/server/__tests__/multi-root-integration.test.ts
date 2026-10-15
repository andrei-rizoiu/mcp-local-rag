// Multi-root MCP server integration tests.
//
// Adds the end-to-end gaps not covered by the list_files-shape and
// warning-visibility files: ingest → list round-trip with per-root annotation,
// delete scoped to one root, query and read_chunk_neighbors across roots,
// raw-data behavior unchanged under multi-root, per-root list_files error
// tolerance, and the sensitive-path policy applied by the REAL
// `resolveServerConfig`. `startServer()` is deliberately not called: it owns
// `process.exit`, which is unsafe in a vitest worker.

import { mkdirSync, rmSync, writeFileSync } from 'node:fs'
import { join, resolve } from 'node:path'
import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import { testModelCacheDir, withTestDevice } from '../../__tests__/test-device.js'
import { resolveServerConfig } from '../../server-main.js'
import { BaseDirsConfigError, displayPath } from '../../utils/base-dirs.js'
import { MAX_SCAN_DEPTH } from '../../utils/limits.js'
import { RAGServer } from '../index.js'
import { scanBaseDir } from '../list-scanner.js'

// =============================================================================
// Helpers
// =============================================================================

type ContentBlock = { type: string; text: string; annotations?: unknown }

/** One block of an MCP tool result, before the text-block narrowing below. */
type ResultBlock = { type: string; text?: string | undefined; annotations?: unknown }

/** Find the first content block whose text contains `needle`. */
function findBlock(content: ReadonlyArray<ResultBlock>, needle: string): ContentBlock | undefined {
  for (const block of content) {
    if (block.type === 'text' && typeof block.text === 'string' && block.text.includes(needle)) {
      return { ...block, text: block.text }
    }
  }
  return undefined
}

// =============================================================================
// AC-008/AC-011/AC-012/AC-013: end-to-end multi-root workflows
// =============================================================================
describe('AC-008/AC-011: multi-root ingest -> list -> query -> delete workflow', () => {
  const testBase = resolve('./tmp/test-multi-root-e2e')
  const rootA = resolve(testBase, 'rootA')
  const rootB = resolve(testBase, 'rootB')
  const dbPath = resolve(testBase, 'lancedb')
  const cacheDir = testModelCacheDir()

  let server: RAGServer
  let fileA: string
  let fileB: string

  beforeAll(async () => {
    mkdirSync(rootA, { recursive: true })
    mkdirSync(rootB, { recursive: true })
    mkdirSync(dbPath, { recursive: true })
    mkdirSync(cacheDir, { recursive: true })

    fileA = resolve(rootA, 'doc-a.txt')
    fileB = resolve(rootB, 'doc-b.txt')
    writeFileSync(
      fileA,
      'Alpha root document about photosynthesis. ' +
        'Photosynthesis is the process by which plants convert sunlight into chemical energy. ' +
        'This sentence pads the document so it clears the minimum chunk filter for ingestion.'
    )
    writeFileSync(
      fileB,
      'Beta root document about gravitational waves. ' +
        'Gravitational waves are ripples in spacetime predicted by general relativity. ' +
        'This sentence pads the document so it clears the minimum chunk filter for ingestion.'
    )

    server = new RAGServer(
      withTestDevice({
        dbPath,
        modelName: 'Xenova/all-MiniLM-L6-v2',
        cacheDir,
        baseDirs: [rootA, rootB],
        maxFileSize: 100 * 1024 * 1024,
      })
    )
    await server.initialize()

    await server.handleIngestFile({ filePath: fileA })
    await server.handleIngestFile({ filePath: fileB })
  }, 120000)

  afterAll(async () => {
    await server.close()
    rmSync(testBase, { recursive: true, force: true })
  })

  // AC interpretation: [AC-008] After ingesting files under different roots, `list_files` reports each file's producing root
  // Validation: list_files annotates fileA with rootA and fileB with rootB, both ingested=true with chunkCount > 0
  it('ingest_file under two roots produces per-root annotated list_files entries', async () => {
    const result = await server.handleListFiles()
    const parsed = JSON.parse(result.content[0].text)

    const entryA = parsed.files.find((f: { filePath: string }) => f.filePath === fileA)
    const entryB = parsed.files.find((f: { filePath: string }) => f.filePath === fileB)
    expect(entryA).toBeDefined()
    expect(entryB).toBeDefined()
    expect(entryA.baseDir).toBe(rootA)
    expect(entryB.baseDir).toBe(rootB)
    expect(entryA.ingested).toBe(true)
    expect(entryB.ingested).toBe(true)
    expect(entryA.chunkCount).toBeGreaterThan(0)
    expect(entryB.chunkCount).toBeGreaterThan(0)
  })

  // AC interpretation: [AC-008/AC-011] query_documents returns chunks ingested under any effective root through the single shared DB
  // Validation: A query that semantically matches doc-b under rootB returns at least one chunk whose filePath equals fileB
  it('query_documents returns chunks ingested from any effective root', async () => {
    const result = await server.handleQueryDocuments({
      query: 'gravitational waves spacetime ripples',
      limit: 5,
    })
    const parsed = JSON.parse(result.content[0].text)
    const fileBHits = parsed.filter((r: { filePath: string }) => r.filePath === fileB)
    expect(fileBHits.length).toBeGreaterThan(0)
  }, 30000)

  // AC interpretation: [AC-008] read_chunk_neighbors works against a file ingested under a non-first root
  // Validation: Calling read_chunk_neighbors on fileB (rootB) returns at least one item with isTarget=true at chunkIndex 0
  it('read_chunk_neighbors returns chunks for a file ingested under a non-first root', async () => {
    const result = await server.handleReadChunkNeighbors({ filePath: fileB, chunkIndex: 0 })
    const parsed = JSON.parse(result.content[0].text)
    expect(Array.isArray(parsed)).toBe(true)
    expect(parsed.length).toBeGreaterThan(0)
    const target = parsed.find((row: { isTarget: boolean }) => row.isTarget === true)
    expect(target).toBeDefined()
    expect(target.filePath).toBe(fileB)
  }, 30000)

  // AC interpretation: [AC-008] delete_file scoped to one root removes only that file's chunks; the other root's chunks remain
  // Validation: After deleting fileA, list_files shows fileA with ingested=false (file still on disk under rootA) and fileB still ingested=true
  it('delete_file under one root leaves the other root untouched', async () => {
    await server.handleDeleteFile({ filePath: fileA })

    const result = await server.handleListFiles()
    const parsed = JSON.parse(result.content[0].text)

    const entryA = parsed.files.find((f: { filePath: string }) => f.filePath === fileA)
    const entryB = parsed.files.find((f: { filePath: string }) => f.filePath === fileB)
    expect(entryA).toBeDefined()
    expect(entryA.ingested).toBe(false)
    expect(entryB).toBeDefined()
    expect(entryB.ingested).toBe(true)
    expect(entryB.chunkCount).toBeGreaterThan(0)
  }, 30000)
})

// =============================================================================
// AC-009 (raw-data path): ingest_data response shape unchanged in multi-root
// =============================================================================
describe('AC-009: ingest_data behavior unchanged in multi-root mode (warnings additive)', () => {
  const testBase = resolve('./tmp/test-multi-root-raw-data')
  const rootA = resolve(testBase, 'rootA')
  const rootB = resolve(testBase, 'rootB')
  const dbPath = resolve(testBase, 'lancedb')
  const cacheDir = testModelCacheDir()

  let server: RAGServer

  // Mirrors the precedence-warning string emitted by resolveBaseDirs so the
  // additive-warning assertion below is anchored to the real warning text.
  const PRECEDENCE_WARNING =
    'BASE_DIRS is set; BASE_DIR is ignored. Unset BASE_DIR or remove BASE_DIRS to silence this warning.'

  beforeAll(async () => {
    mkdirSync(rootA, { recursive: true })
    mkdirSync(rootB, { recursive: true })
    mkdirSync(dbPath, { recursive: true })
    mkdirSync(cacheDir, { recursive: true })

    server = new RAGServer(
      withTestDevice({
        dbPath,
        modelName: 'Xenova/all-MiniLM-L6-v2',
        cacheDir,
        baseDirs: [rootA, rootB],
        maxFileSize: 100 * 1024 * 1024,
        configWarnings: [PRECEDENCE_WARNING],
      })
    )
    await server.initialize()
  }, 60000)

  afterAll(async () => {
    await server.close()
    rmSync(testBase, { recursive: true, force: true })
  })

  // AC interpretation: [AC-009] ingest_data preserves its single-root response shape (filePath/chunkCount/timestamp/fileTitle)
  // even when the server is configured with multiple roots; warnings are an ADDITIVE content block.
  // Validation: Primary content block parses to IngestResult-shaped JSON with chunkCount>0 and filePath under dbPath/raw-data;
  // the warning block appears alongside but does not alter the primary block.
  it('ingest_data raw-data response shape is unchanged; warning block is additive', async () => {
    const result = await server.handleIngestData({
      content:
        'Raw-data ingestion in multi-root mode. ' +
        'This content is long enough to clear the minimum chunk filter and produce at least one chunk. ' +
        'It exists solely to confirm that ingest_data behavior is unaffected by multi-root configuration.',
      metadata: {
        source: 'clipboard://2026-05-23/multi-root-raw-data',
        format: 'text',
      },
    })

    // Primary block: same shape as single-root contract.
    const primary = result.content[0]
    expect(primary.type).toBe('text')
    const parsed = JSON.parse(primary.text)
    expect(parsed.chunkCount).toBeGreaterThan(0)
    expect(typeof parsed.timestamp).toBe('string')
    expect(typeof parsed.filePath).toBe('string')
    // Raw-data path lives under dbPath/raw-data — confirms ingest_data did not
    // accidentally route through any root's filesystem.
    expect(parsed.filePath.startsWith(resolve(dbPath))).toBe(true)
    expect(parsed.filePath).toContain('raw-data')

    // Warning block is additive.
    const warningBlock = findBlock(result.content, PRECEDENCE_WARNING)
    expect(warningBlock).toBeDefined()
  }, 60000)
})

// `list_files` survives a permission-denied error on one root, warning per
// root instead of failing the call. rootA is made unreadable with chmod, which
// only works on POSIX, so the test is skipped on Windows.
describe('post-launch finding #10: list_files per-root error tolerance', () => {
  const testBase = resolve('./tmp/test-list-files-per-root-err')
  const rootA = resolve(testBase, 'rootA')
  const rootB = resolve(testBase, 'rootB')
  const dbPath = resolve(testBase, 'lancedb')
  const cacheDir = testModelCacheDir()

  beforeAll(() => {
    mkdirSync(rootA, { recursive: true })
    mkdirSync(rootB, { recursive: true })
    mkdirSync(dbPath, { recursive: true })
    mkdirSync(cacheDir, { recursive: true })
    writeFileSync(resolve(rootB, 'survives.md'), 'a small markdown file under the accessible root')
  })

  afterAll(() => {
    // Restore permissions before tearing down so rmSync can recurse.
    try {
      // eslint-disable-next-line @typescript-eslint/no-require-imports
      require('node:fs').chmodSync(rootA, 0o755)
    } catch {
      // Ignore — chmod may fail if rootA was already removed.
    }
    rmSync(testBase, { recursive: true, force: true })
  })

  it.skipIf(process.platform === 'win32')(
    'list_files surfaces files from accessible roots when another root is inaccessible',
    async () => {
      // Make rootA unreadable using chmod so `readdir(rootA)` throws EACCES
      // at the syscall layer (Linux/macOS only; Windows skips this test).
      // We rely on the bounded BFS new in Finding #10 to capture the error
      // as a per-root warning and keep scanning rootB.
      const { chmodSync } = await import('node:fs')
      chmodSync(rootA, 0o000)

      const server = new RAGServer(
        withTestDevice({
          dbPath,
          modelName: 'Xenova/all-MiniLM-L6-v2',
          cacheDir,
          baseDirs: [rootA, rootB],
          maxFileSize: 100 * 1024 * 1024,
        })
      )
      await server.initialize()
      try {
        const result = await server.handleListFiles()
        const parsed = JSON.parse(result.content[0]?.text ?? '{}')
        const filePaths: string[] = parsed.files.map((f: { filePath: string }) => f.filePath)
        // rootB's file survives even though rootA failed.
        expect(filePaths).toContain(resolve(rootB, 'survives.md'))
        // Warning content block names the failing root via `displayPath`
        // (HOME prefix is collapsed to `~` to avoid leaking the OS username
        // through MCP responses; see Finding #10 sanitization).
        const warningBlock = findBlock(
          result.content,
          `cannot read directory: ${displayPath(rootA)}`
        )
        expect(warningBlock).toBeDefined()
        // The raw OS error message must not leak into the warning text.
        expect(warningBlock?.text ?? '').not.toContain('permission denied')
      } finally {
        await server.close()
        // Restore permissions so afterAll's rmSync can recurse.
        chmodSync(rootA, 0o755)
      }
    },
    60000
  )
})

// `scanBaseDir` owns no traversal — only the `list_files` warning wording, the
// once-per-call depth warning, and the sort. These pin that presentation so the
// shared walker cannot silently change `list_files` output.
describe('scanBaseDir presentation contract over the shared walker', () => {
  const testBase = resolve('./tmp/test-list-scanner-adapter')
  const root = resolve(testBase, 'root')
  const missing = resolve(testBase, 'does-not-exist')

  /** Nested chain deep enough that its tail is dequeued past MAX_SCAN_DEPTH. */
  function deepChain(prefix: string): string {
    const segments = Array.from({ length: MAX_SCAN_DEPTH + 2 }, (_, i) => `${prefix}-l${i + 1}`)
    return join(root, ...segments)
  }

  beforeAll(() => {
    mkdirSync(root, { recursive: true })
    mkdirSync(join(root, 'sub'), { recursive: true })
    mkdirSync(deepChain('a'), { recursive: true })
    mkdirSync(deepChain('b'), { recursive: true })
    writeFileSync(join(root, 'zebra.md'), 'z')
    writeFileSync(join(root, 'alpha.md'), 'a')
    writeFileSync(join(root, 'sub', 'middle.md'), 'm')
    writeFileSync(join(root, 'unsupported.bin'), 'skip me')
  })

  afterAll(() => {
    rmSync(testBase, { recursive: true, force: true })
  })

  it('returns sorted supported files from the shared walker', async () => {
    const { files } = await scanBaseDir(root, [])

    expect(files).toEqual(
      [join(root, 'alpha.md'), join(root, 'sub', 'middle.md'), join(root, 'zebra.md')].sort()
    )
    expect(files).not.toContain(join(root, 'unsupported.bin'))
  })

  it('renders the unreadable-directory warning verbatim', async () => {
    const { files, warnings } = await scanBaseDir(missing, [])

    expect(files).toEqual([])
    expect(warnings).toEqual([`cannot read directory: ${displayPath(missing)} (ENOENT)`])
  })

  it('emits the depth-limit warning once per call even with several depth-limited branches', async () => {
    const { warnings } = await scanBaseDir(root, [])

    const depthWarnings = warnings.filter((w) => w.includes('exceed the maximum depth'))
    expect(depthWarnings).toEqual([
      `some directories under ${displayPath(root)} were skipped because they exceed the maximum depth (${MAX_SCAN_DEPTH})`,
    ])
  })
})

// The MCP entry point must apply the sensitive-path policy to env-resolved
// roots and must not fall back to cwd on a config error. These call the REAL
// `resolveServerConfig`, so the assertions hold against production wiring.
describe('post-launch findings #3 + #4: server-main wiring rejects sensitive roots and never falls back to cwd', () => {
  const testBase = resolve('./tmp/test-server-main-policy')
  const dbPath = resolve(testBase, 'lancedb')
  const cacheDir = testModelCacheDir()

  beforeAll(() => {
    mkdirSync(dbPath, { recursive: true })
    mkdirSync(cacheDir, { recursive: true })
  })

  afterAll(() => {
    rmSync(testBase, { recursive: true, force: true })
  })

  /**
   * Build a server through the REAL `resolveServerConfig`, so the test is
   * anchored to production wiring, without `startServer`'s `process.exit`.
   */
  async function buildServerLikeMain(opts: {
    envBaseDirs?: string | undefined
    envBaseDir?: string | undefined
  }): Promise<{ server: RAGServer; config: Awaited<ReturnType<typeof resolveServerConfig>> }> {
    const env: NodeJS.ProcessEnv = {
      DB_PATH: dbPath,
      CACHE_DIR: cacheDir,
      MODEL_NAME: 'Xenova/all-MiniLM-L6-v2',
      MAX_FILE_SIZE: String(100 * 1024 * 1024),
      ...(opts.envBaseDirs !== undefined ? { BASE_DIRS: opts.envBaseDirs } : {}),
      ...(opts.envBaseDir !== undefined ? { BASE_DIR: opts.envBaseDir } : {}),
    }
    const config = await resolveServerConfig(env, testBase)
    const server = new RAGServer(withTestDevice(config))
    return { server, config }
  }

  // AC: BASE_DIRS=["/etc"] must NOT be silently accepted by server-main.
  // The server enters degraded mode (configError set, baseDirs empty); root-
  // dependent tools fail fast; `status` surfaces the diagnostic.
  it('rejects BASE_DIRS=["/etc"] as a sensitive system path (degraded mode)', async () => {
    const { server, config } = await buildServerLikeMain({
      envBaseDirs: JSON.stringify(['/etc']),
    })

    expect(config.configError).toBeDefined()
    expect(config.configError?.message).toMatch(/sensitive system path/)
    expect(config.configError?.message).toContain('BASE_DIRS')

    // baseDirs MUST be empty: no silent cwd fallback.
    expect(config.baseDirs).toEqual([])

    // Root-dependent tool: must fail fast. The config gate now throws the
    // structured BaseDirsConfigError with original identity (the central
    // dispatcher mapper maps it to McpError(InvalidParams) at the boundary —
    // see rag-server.dispatcher-mapping.test.ts).
    await expect(server.handleListFiles()).rejects.toBeInstanceOf(BaseDirsConfigError)

    // status: must remain callable and expose the diagnostic.
    await server.initialize()
    try {
      const status = await server.handleStatus()
      const diagnostic = findBlock(status.content, 'Configuration error:')
      expect(diagnostic).toBeDefined()
      expect(diagnostic?.text).toMatch(/sensitive system path/)
    } finally {
      await server.close()
    }
  }, 60000)

  // AC: BASE_DIR=/usr (single-root sensitive path) is likewise rejected and
  // attributed to BASE_DIR (not BASE_DIRS).
  it('rejects BASE_DIR=/usr as a sensitive system path attributed to BASE_DIR', async () => {
    const { config } = await buildServerLikeMain({ envBaseDir: '/usr' })

    expect(config.configError).toBeDefined()
    expect(config.configError?.message).toMatch(/sensitive system path/)
    expect(config.configError?.message).toContain('BASE_DIR')
    expect(config.baseDirs).toEqual([])
  })
})
