// CLI Ingest Tests
// Tests runIngest functionality with mocked dependencies

import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

// ============================================
// Mock Setup (vi.hoisted for isolate: false)
// ============================================

const mocks = vi.hoisted(() => {
  // Default base-dirs resolution: CLI roots when provided, otherwise a fixed
  // cwd-derived stand-in. Each test can reassign `mocks.resolveCliBaseDirs`
  // before invoking `runIngest` / `resolveConfig` to inject scenario-specific
  // results (precedence, warnings, ...).
  const defaultResolve = (cliRoots: string[]) => {
    const baseDirs = cliRoots.length > 0 ? cliRoots : ['/mock/cwd/']
    return Promise.resolve({ config: { baseDirs }, warnings: [] })
  }
  return {
    // fs/promises
    stat: vi.fn(),
    readdir: vi.fn(),
    // `collectFiles` now realpath-resolves the positional path before the
    // "inside any configured root" check (Finding #1). The default mock is
    // identity (no symlinks) so existing tests behave as if the positional
    // path is its own realpath; per-test mocks can override.
    realpath: vi.fn().mockImplementation((p: string) => Promise.resolve(p)),

    // Shared CLI base-dirs resolver
    resolveCliBaseDirs: vi.fn().mockImplementation(defaultResolve),

    // Component instances
    parseFile: vi.fn(),
    parsePdf: vi.fn(),
    // Real DocumentParser methods, called by the pre-parse `contentHash` read.
    validateFilePath: vi.fn().mockResolvedValue(undefined),
    validateFileSize: vi.fn(),
    chunkText: vi.fn(),
    embedBatch: vi.fn(),
    initialize: vi.fn(),
    deleteChunks: vi.fn(),
    insertChunks: vi.fn().mockImplementation((chunks: unknown[]) => {
      // Log chunk key fields to stderr for verification
      for (const chunk of chunks) {
        const c = expectRecord(chunk)
        console.error(
          `[mock:insertChunks] filePath=${c.filePath} chunkIndex=${c.chunkIndex} text=${c.text} vectorLen=${Array.isArray(c.vector) ? c.vector.length : 'none'}`
        )
      }
      return Promise.resolve(undefined)
    }),
    optimize: vi.fn().mockImplementation(() => {
      // Log optimize call to stderr for verification
      console.error('[mock:optimize] called')
      return Promise.resolve(undefined)
    }),
  }
})

// Mock factories are installed via `vi.doMock` in `beforeAll` and removed
// via `vi.doUnmock` in `afterAll`. See `.claude/skills/project-context/SKILL.md`
// "Test Environment Constraints".

const fsPromisesFactory = async (
  importOriginal: () => Promise<typeof import('node:fs/promises')>
) => {
  const actual = await importOriginal()
  return {
    ...actual,
    stat: mocks.stat,
    readdir: mocks.readdir,
    realpath: mocks.realpath,
    // `ingestSingleFile` reads the raw source bytes to compute `contentHash`.
    // These tests drive fully mocked parsers over paths that never exist on
    // disk, so the byte read returns fixed content.
    readFile: async () => Buffer.from('fixture bytes', 'utf-8'),
  }
}

const parserFactory = () => ({
  DocumentParser: vi.fn().mockImplementation(function (this: Record<string, unknown>) {
    this.parseFile = mocks.parseFile
    this.parsePdf = mocks.parsePdf
    // Real DocumentParser boundary checks, as recording spies: the pre-parse
    // `contentHash` read runs them itself, because it no longer sits behind the
    // parse that used to. Their decisions are pinned against a real
    // `DocumentParser` in `ingest-content-hash-pre-parse.test.ts`.
    this.validateFilePath = mocks.validateFilePath
    this.validateFileSize = mocks.validateFileSize
  }),
  SUPPORTED_EXTENSIONS: new Set(['.pdf', '.docx', '.txt', '.md']),
})

const chunkerFactory = () => ({
  SemanticChunker: vi.fn().mockImplementation(function (this: Record<string, unknown>) {
    this.chunkText = mocks.chunkText
  }),
})

const cliCommonFactory = () => ({
  createEmbedder: vi.fn().mockImplementation(() => ({
    embedBatch: mocks.embedBatch,
    dispose: vi.fn(),
  })),
  createVectorStore: vi.fn().mockImplementation(() => ({
    initialize: mocks.initialize,
    deleteChunks: mocks.deleteChunks,
    insertChunks: mocks.insertChunks,
    optimize: mocks.optimize,
    close: vi.fn(),
  })),
  // Mock the shared CLI base-dirs resolver to skip realpath I/O. Each test
  // sets `mocks.resolveCliBaseDirs` to mirror the precedence under test
  // (e.g. CLI roots replace env roots; env roots fall through to cwd).
  resolveCliBaseDirsOrExit: vi
    .fn()
    .mockImplementation((cliRoots: string[]) => mocks.resolveCliBaseDirs(cliRoots)),
  // Catch-block renderer; faithful shim preserves the per-file
  // `... FAILED: <message>` stderr behavior the tests assert.
  formatCliError: formatCliErrorShim,
})

const MOCKED_PATHS = [
  'node:fs/promises',
  '../../parser/index.js',
  '../../chunker/index.js',
  '../../cli/common.js',
] as const

// `cli/ingest.js` and `cli/options.js` are imported dynamically in beforeAll
// after vi.resetModules(). Load-bearing under `isolate: false`: a sibling file
// mocking the same paths can otherwise win the module-registry race and bind
// runIngest's closures to its factories instead of this file's.
import { resolve } from 'node:path'
import { expectError, expectRecord } from '../test-doubles.js'
import { formatCliErrorShim } from './cli-error-shim.js'

let runIngest: typeof import('../../cli/ingest.js').runIngest
let parseArgs: typeof import('../../cli/ingest.js').parseArgs
let resolveConfig: typeof import('../../cli/ingest.js').resolveConfig
let resolveGlobalConfig: typeof import('../../cli/options.js').resolveGlobalConfig

// ============================================
// Helpers
// ============================================

/**
 * Capture stderr output during a function call.
 * Uses vi.spyOn on console.error since the implementation uses console.error for stderr.
 */
function captureStderr(fn: () => Promise<void>): Promise<{ output: string[]; error: unknown }> {
  const output: string[] = []
  const spy = vi.spyOn(console, 'error').mockImplementation((...args: unknown[]) => {
    output.push(args.map(String).join(' '))
  })

  return fn()
    .then(() => ({ output, error: undefined }))
    .catch((error: unknown) => ({ output, error }))
    .finally(() => {
      spy.mockRestore()
    })
}

/**
 * Create a mock stat result for a file.
 */
function mockFileStat() {
  return { isFile: () => true, isDirectory: () => false }
}

/**
 * Create a mock stat result for a directory.
 */
function mockDirStat() {
  return { isFile: () => false, isDirectory: () => true }
}

/**
 * Create a mock Dirent entry.
 */
function mockDirent(
  name: string,
  type: 'file' | 'directory' = 'file'
): {
  name: string
  isFile: () => boolean
  isDirectory: () => boolean
  isSymbolicLink: () => boolean
} {
  return {
    name,
    isFile: () => type === 'file',
    isDirectory: () => type === 'directory',
    isSymbolicLink: () => false,
  }
}

/**
 * Drive the `readdir({ withFileTypes: true })` mock from a directory map;
 * an absent directory resolves to an empty array.
 */
function setupMockReaddir(dirMap: Record<string, ReturnType<typeof mockDirent>[]>) {
  mocks.readdir.mockImplementation(async (dirPath: string) => dirMap[dirPath] ?? [])
}

/**
 * Set up default successful mocks for single file ingestion.
 */
function setupSuccessfulIngestion() {
  mocks.parseFile.mockResolvedValue({ content: 'parsed text content', title: 'Test Title' })
  mocks.chunkText.mockResolvedValue([
    { text: 'chunk 1', index: 0 },
    { text: 'chunk 2', index: 1 },
  ])
  mocks.embedBatch.mockResolvedValue([
    [0.1, 0.2],
    [0.3, 0.4],
  ])
  mocks.deleteChunks.mockResolvedValue(undefined)
  mocks.initialize.mockResolvedValue(undefined)
  // insertChunks and optimize use default implementations from mock setup
  // that log to stderr for verification
}

// ============================================
// Tests
// ============================================

describe('CLI ingest', () => {
  let exitSpy: ReturnType<typeof vi.spyOn>

  beforeAll(async () => {
    vi.resetModules()
    vi.doMock('node:fs/promises', fsPromisesFactory)
    vi.doMock('../../parser/index.js', parserFactory)
    vi.doMock('../../chunker/index.js', chunkerFactory)
    vi.doMock('../../cli/common.js', cliCommonFactory)
    ;({ runIngest, parseArgs, resolveConfig } = await import('../../cli/ingest.js'))
    ;({ resolveGlobalConfig } = await import('../../cli/options.js'))
  })

  afterAll(() => {
    for (const p of MOCKED_PATHS) {
      vi.doUnmock(p)
    }
    vi.resetModules()
  })

  beforeEach(() => {
    vi.clearAllMocks()
    // Restore the default base-dirs resolution after vi.clearAllMocks so
    // tests that don't customize the resolver still get a valid config.
    mocks.resolveCliBaseDirs.mockImplementation((cliRoots: string[]) => {
      const baseDirs = cliRoots.length > 0 ? cliRoots : ['/mock/cwd/']
      return Promise.resolve({ config: { baseDirs }, warnings: [] })
    })
    // Mock process.exit to throw so we can catch it
    exitSpy = vi
      .spyOn(process, 'exit')
      .mockImplementation((code?: number | string | null | undefined) => {
        throw new Error(`process.exit(${code})`)
      })
  })

  afterEach(() => {
    exitSpy.mockRestore()
    process.exitCode = undefined
  })

  // --------------------------------------------
  // Single file ingest
  // --------------------------------------------
  it('should parse, chunk, embed, delete, insert, and optimize once for a single file', async () => {
    // Arrange
    const filePath = resolve('/tmp/test/document.md')
    mocks.stat.mockResolvedValue(mockFileStat())
    setupSuccessfulIngestion()

    // Act
    const { output, error } = await captureStderr(() => runIngest([filePath]))

    // Assert: returns normally without process.exit
    expect(error).toBeUndefined()
    expect(process.exitCode).toBeUndefined()

    // Assert: output shows success
    const joined = output.join('\n')
    expect(joined).toContain('OK (2 chunks)')
    expect(joined).toContain('Succeeded: 1')
    expect(joined).toContain('Failed:    0')
    expect(joined).toContain('Total chunks: 2')

    // Assert: optimize was called exactly once (verified via stderr marker)
    const optimizeLines = output.filter((line) => line.includes('[mock:optimize] called'))
    expect(optimizeLines).toHaveLength(1)

    // Assert: insertChunks received VectorChunk with expected structure
    const insertLines = output.filter((line) => line.includes('[mock:insertChunks]'))
    expect(insertLines).toHaveLength(2) // 2 chunks
    // Verify chunk 0 has correct filePath, chunkIndex, text, and vector
    expect(insertLines[0]).toContain(`filePath=${filePath}`)
    expect(insertLines[0]).toContain('chunkIndex=0')
    expect(insertLines[0]).toContain('text=chunk 1')
    expect(insertLines[0]).toContain('vectorLen=2')
    // Verify chunk 1
    expect(insertLines[1]).toContain(`filePath=${filePath}`)
    expect(insertLines[1]).toContain('chunkIndex=1')
    expect(insertLines[1]).toContain('text=chunk 2')
    expect(insertLines[1]).toContain('vectorLen=2')
  })

  it('should pass parser atomic ranges through the actual runIngest path', async () => {
    const filePath = resolve('/tmp/test/document.docx')
    const parsedContent = 'Intro\n\nHeader: First Second'
    const atomicRanges = [
      {
        start: 'Intro\n\n'.length,
        end: parsedContent.length,
      },
    ] as const
    mocks.stat.mockResolvedValue(mockFileStat())
    setupSuccessfulIngestion()
    mocks.parseFile.mockResolvedValue({
      content: parsedContent,
      title: 'Table document',
      atomicRanges,
    })

    const { error } = await captureStderr(() => runIngest([filePath]))

    expect(error).toBeUndefined()
    // No title prefix unless EMBED_TITLE_PREFIX is on.
    expect(mocks.chunkText).toHaveBeenCalledWith(parsedContent, expect.anything(), atomicRanges, '')
  })

  // --------------------------------------------
  // Directory ingest
  // --------------------------------------------
  it('should recursively find supported files and ingest all when given a directory', async () => {
    // Arrange: first stat call for path validation, second for collectFiles.
    // After P2-T2 the directory-mode scan walks every effective root in
    // `config.baseDirs.baseDirs` rather than the positional path, so the
    // resolver mock must echo `dirPath` as the effective root.
    const dirPath = resolve('/tmp/test/docs')
    mocks.stat
      .mockResolvedValueOnce(mockDirStat()) // path validation in runIngest
      .mockResolvedValueOnce(mockDirStat()) // stat in collectFiles
    mocks.resolveCliBaseDirs.mockResolvedValue({ config: { baseDirs: [dirPath] }, warnings: [] })

    setupMockReaddir({
      [dirPath]: [mockDirent('file1.md'), mockDirent('sub', 'directory')],
      [resolve('/tmp/test/docs/sub')]: [mockDirent('file2.txt')],
    })

    setupSuccessfulIngestion()

    // Act
    const { output, error } = await captureStderr(() => runIngest([dirPath]))

    // Assert: returns normally without process.exit
    expect(error).toBeUndefined()
    expect(process.exitCode).toBeUndefined()

    // Assert: both files processed, optimize called once
    const joined = output.join('\n')
    expect(joined).toContain('[1/2]')
    expect(joined).toContain('[2/2]')
    expect(joined).toContain('Succeeded: 2')
    expect(joined).toContain('Total chunks: 4')

    // Assert: optimize was called exactly once (not per-file)
    const optimizeLines = output.filter((line) => line.includes('[mock:optimize] called'))
    expect(optimizeLines).toHaveLength(1)
  })

  // --------------------------------------------
  // Multi-root directory ingest (P2-T2)
  // --------------------------------------------
  it('should scan only the positional directory even when multiple roots are configured', async () => {
    // `ingest <dir>` scans the positional path. The configured roots are the
    // VALIDATION boundary passed to the parser, not a replacement scan target:
    // aggregating every root would ingest unrelated content under `rootB` when
    // the user asked only for `rootA`.
    const rootA = resolve('/tmp/test/rootA')
    const rootB = resolve('/tmp/test/rootB')
    mocks.stat.mockResolvedValueOnce(mockDirStat()).mockResolvedValueOnce(mockDirStat())
    mocks.resolveCliBaseDirs.mockResolvedValue({
      config: { baseDirs: [rootA, rootB] },
      warnings: [],
    })

    setupMockReaddir({
      [rootA]: [mockDirent('a.md')],
      // rootB has a file too, but it must NOT be ingested because the
      // positional path is `rootA`.
      [rootB]: [mockDirent('b.md')],
    })
    setupSuccessfulIngestion()

    // Act: positional path is rootA.
    const { output, error } = await captureStderr(() =>
      runIngest(['--base-dir', rootA, '--base-dir', rootB, rootA])
    )

    // Assert: resolver received the CLI roots verbatim and in order.
    expect(error).toBeUndefined()
    expect(mocks.resolveCliBaseDirs).toHaveBeenCalledWith([rootA, rootB])

    // Assert: exactly one file (rootA/a.md) ingested.
    const joined = output.join('\n')
    expect(joined).toContain('Found 1 file(s) to ingest')
    expect(joined).toContain(resolve(rootA, 'a.md'))
    expect(joined).not.toContain(resolve(rootB, 'b.md'))
    expect(joined).toContain('Succeeded: 1')
  })

  it('should exit 1 with a clear message when the positional directory is outside all configured roots', async () => {
    const rootA = resolve('/tmp/test/rootA')
    const rootB = resolve('/tmp/test/rootB')
    const outsidePath = resolve('/tmp/test/outside')
    mocks.stat.mockResolvedValue(mockDirStat())
    mocks.resolveCliBaseDirs.mockResolvedValue({
      config: { baseDirs: [rootA, rootB] },
      warnings: [],
    })

    const { output, error } = await captureStderr(() =>
      runIngest(['--base-dir', rootA, '--base-dir', rootB, outsidePath])
    )

    expect(error).toBeInstanceOf(Error)
    expect(expectError(error).message).toBe('process.exit(1)')

    const joined = output.join('\n')
    expect(joined).toContain('not under any configured base directory')
    expect(joined).toContain(outsidePath)
    expect(joined).toContain(rootA)
    expect(joined).toContain(rootB)
  })

  it('should preserve dbPath and cacheDir exclusion when scanning the positional directory', async () => {
    // Post-Finding-#1: only the positional path is scanned. The dbPath /
    // cacheDir exclusion still applies under that single scanned tree, so
    // files under either excluded path are skipped.
    const rootA = resolve('/tmp/test/dbA')
    const dbPath = resolve('/tmp/test/dbA/lancedb')
    const cacheDir = resolve('/tmp/test/dbA/cache')
    mocks.stat.mockResolvedValueOnce(mockDirStat()).mockResolvedValueOnce(mockDirStat())
    mocks.resolveCliBaseDirs.mockResolvedValue({
      config: { baseDirs: [rootA] },
      warnings: [],
    })

    setupMockReaddir({
      [rootA]: [
        mockDirent('keep.md'),
        mockDirent('lancedb', 'directory'),
        mockDirent('cache', 'directory'),
      ],
      [dbPath]: [mockDirent('chunks.md')], // excluded
      [cacheDir]: [mockDirent('model.md')], // excluded
    })
    setupSuccessfulIngestion()

    const { output, error } = await captureStderr(() =>
      runIngest([rootA], { dbPath, cacheDir, modelName: 'm' })
    )

    expect(error).toBeUndefined()
    const joined = output.join('\n')
    expect(joined).toContain('Found 1 file(s) to ingest')
    expect(joined).toContain('keep.md')
    expect(joined).not.toContain('chunks.md')
    expect(joined).not.toContain('model.md')
  })

  it('should surface nested-root pruning warnings from the resolver to stderr', async () => {
    // Arrange: resolver returns a single effective root plus a
    // nested-root-pruned warning describing the dropped child. The CLI must
    // print the warning on stderr so the user sees it in the same stream
    // as the scan output.
    const root = resolve('/tmp/test/parent')
    mocks.stat.mockResolvedValueOnce(mockDirStat()).mockResolvedValueOnce(mockDirStat())
    mocks.resolveCliBaseDirs.mockResolvedValue({
      config: { baseDirs: [root] },
      warnings: [
        {
          kind: 'nested-root-pruned',
          message: `Nested base directory pruned: ${root}/child/ is inside ${root}/. Keeping ${root}/ only.`,
          parent: `${root}/`,
          pruned: `${root}/child/`,
        },
      ],
    })

    setupMockReaddir({ [root]: [mockDirent('a.md')] })
    setupSuccessfulIngestion()

    // Act
    const { output, error } = await captureStderr(() => runIngest([root]))

    // Assert: the pruning warning appears on stderr
    expect(error).toBeUndefined()
    const joined = output.join('\n')
    expect(joined).toContain('Nested base directory pruned')
  })

  // --------------------------------------------
  // Max depth limit
  // --------------------------------------------
  it('should include files within max depth and skip directories beyond it', async () => {
    // Arrange: nested directories, depth 10 directory is not entered
    const dirPath = resolve('/tmp/test/docs')
    mocks.stat.mockResolvedValueOnce(mockDirStat()).mockResolvedValueOnce(mockDirStat())
    mocks.resolveCliBaseDirs.mockResolvedValue({ config: { baseDirs: [dirPath] }, warnings: [] })

    // Build a chain of 10 nested directories (depth 0..9), plus one at depth 10
    const dirMap: Record<string, ReturnType<typeof mockDirent>[]> = {
      [dirPath]: [mockDirent('root.md'), mockDirent('d1', 'directory')],
    }
    let current = dirPath
    for (let i = 1; i <= 10; i++) {
      const next = resolve(`${current}/d${i}`)
      if (i < 10) {
        // Depths 1-9: directory with a subdirectory
        dirMap[next] = [mockDirent(`d${i + 1}`, 'directory')]
      }
      // Depth 10: should never be opened (BFS skips it)
      if (i === 9) {
        dirMap[next] = [mockDirent('deep-ok.md'), mockDirent('d10', 'directory')]
      }
      current = next
    }

    setupMockReaddir(dirMap)
    setupSuccessfulIngestion()

    // Act
    const { output, error } = await captureStderr(() => runIngest([dirPath]))

    // Assert: returns normally
    expect(error).toBeUndefined()
    expect(process.exitCode).toBeUndefined()

    // Assert: 2 files processed (root.md at depth 0, deep-ok.md at depth 9)
    const joined = output.join('\n')
    expect(joined).toContain('[1/2]')
    expect(joined).toContain('[2/2]')
    expect(joined).toContain('Succeeded: 2')
    expect(joined).toContain(
      'Warning: some directories were skipped because they exceed the maximum depth'
    )
  })

  // --------------------------------------------
  // Permission error handling
  // --------------------------------------------
  it('should skip inaccessible directories and continue processing others', async () => {
    // Arrange
    const dirPath = resolve('/tmp/test/docs')
    const restrictedPath = resolve('/tmp/test/docs/restricted')
    const subPath = resolve('/tmp/test/docs/sub')
    mocks.stat.mockResolvedValueOnce(mockDirStat()).mockResolvedValueOnce(mockDirStat())
    mocks.resolveCliBaseDirs.mockResolvedValue({ config: { baseDirs: [dirPath] }, warnings: [] })

    mocks.readdir.mockImplementation(async (path: string) => {
      if (path === restrictedPath) {
        throw new Error('EACCES: permission denied')
      }
      const dirMap: Record<string, ReturnType<typeof mockDirent>[]> = {
        [dirPath]: [
          mockDirent('ok.md'),
          mockDirent('restricted', 'directory'),
          mockDirent('sub', 'directory'),
        ],
        [subPath]: [mockDirent('also-ok.md')],
      }
      return dirMap[path] ?? []
    })

    setupSuccessfulIngestion()

    // Act
    const { output, error } = await captureStderr(() => runIngest([dirPath]))

    // Assert: 2 files processed, restricted directory skipped with warning
    expect(error).toBeUndefined()
    const joined = output.join('\n')
    expect(joined).toContain('[1/2]')
    expect(joined).toContain('[2/2]')
    expect(joined).toContain('Succeeded: 2')
    expect(joined).toContain(`Warning: cannot read directory: ${restrictedPath}`)
  })

  // --------------------------------------------
  // Skip unsupported files
  // --------------------------------------------
  it('should skip unsupported file extensions like .jpg', async () => {
    // Arrange
    const filePath = resolve('/tmp/test/image.jpg')
    mocks.stat.mockResolvedValue(mockFileStat())

    // Act
    const { output, error } = await captureStderr(() => runIngest([filePath]))

    // Assert: exit(1) because no supported files found
    expect(error).toBeInstanceOf(Error)
    expect(expectError(error).message).toBe('process.exit(1)')

    const joined = output.join('\n')
    expect(joined).toContain('Unsupported file extension: .jpg')
  })

  // --------------------------------------------
  // Error skip in bulk
  // --------------------------------------------
  it('should skip failed files and continue processing remaining files', async () => {
    // Arrange
    const dirPath = resolve('/tmp/test/docs')
    mocks.stat.mockResolvedValueOnce(mockDirStat()).mockResolvedValueOnce(mockDirStat())
    mocks.resolveCliBaseDirs.mockResolvedValue({ config: { baseDirs: [dirPath] }, warnings: [] })

    setupMockReaddir({
      [dirPath]: [mockDirent('bad.md'), mockDirent('good.md'), mockDirent('good2.txt')],
    })

    mocks.initialize.mockResolvedValue(undefined)
    mocks.optimize.mockResolvedValue(undefined)
    mocks.deleteChunks.mockResolvedValue(undefined)
    mocks.insertChunks.mockResolvedValue(undefined)
    mocks.embedBatch.mockResolvedValue([
      [0.1, 0.2],
      [0.3, 0.4],
    ])
    mocks.chunkText.mockResolvedValue([
      { text: 'chunk 1', index: 0 },
      { text: 'chunk 2', index: 1 },
    ])

    // Files sorted: bad.md, good.md, good2.txt — first file (bad.md) fails at parse
    mocks.parseFile
      .mockRejectedValueOnce(new Error('Parse error: corrupted file'))
      .mockResolvedValueOnce({ content: 'good content', title: 'Good' })
      .mockResolvedValueOnce({ content: 'good content 2', title: 'Good2' })

    // Act
    const { output, error } = await captureStderr(() => runIngest([dirPath]))

    // Assert: returns normally with exitCode=1 for partial failure
    expect(error).toBeUndefined()
    expect(process.exitCode).toBe(1)

    const joined = output.join('\n')
    // formatCliError now renders the failing file's diagnostic (message + stack)
    // on the per-file FAILED line; the original message is still present.
    expect(joined).toContain('FAILED:')
    expect(joined).toContain('Parse error: corrupted file')
    expect(joined).toContain('Succeeded: 2')
    expect(joined).toContain('Failed:    1')
  })

  // --------------------------------------------
  // Empty directory
  // --------------------------------------------
  it('should exit gracefully with message when directory has no supported files', async () => {
    // Arrange
    const dirPath = resolve('/tmp/test/empty')
    mocks.stat.mockResolvedValueOnce(mockDirStat()).mockResolvedValueOnce(mockDirStat())
    mocks.resolveCliBaseDirs.mockResolvedValue({ config: { baseDirs: [dirPath] }, warnings: [] })

    setupMockReaddir({
      [dirPath]: [mockDirent('readme.jpg')],
    })

    // Act
    const { output, error } = await captureStderr(() => runIngest([dirPath]))

    // Assert: exit(1) with "No supported files found"
    expect(error).toBeInstanceOf(Error)
    expect(expectError(error).message).toBe('process.exit(1)')

    const joined = output.join('\n')
    expect(joined).toContain('No supported files found')
  })

  // --------------------------------------------
  // Non-existent path
  // --------------------------------------------
  it('should show error message and exit code 1 for non-existent path', async () => {
    // Arrange
    const filePath = '/tmp/test/nonexistent.md'
    mocks.stat.mockRejectedValue(new Error('ENOENT: no such file or directory'))

    // Act
    const { output, error } = await captureStderr(() => runIngest([filePath]))

    // Assert: exit(1) with error message
    expect(error).toBeInstanceOf(Error)
    expect(expectError(error).message).toBe('process.exit(1)')

    const joined = output.join('\n')
    expect(joined).toContain('Error: path does not exist')
    expect(joined).toContain(filePath)
  })

  // --------------------------------------------
  // --help shows usage and exits
  // --------------------------------------------
  it('should show help text and exit with code 0 when --help is passed', async () => {
    // Act
    const { output, error } = await captureStderr(() => runIngest(['--help']))

    // Assert: exit(0)
    expect(error).toBeInstanceOf(Error)
    expect(expectError(error).message).toBe('process.exit(0)')

    // Assert: help text contains ingest-specific information
    const joined = output.join('\n')
    expect(joined).toContain('Usage: mcp-local-rag')
    expect(joined).toContain('ingest')
    expect(joined).toContain('--base-dir')
    expect(joined).toContain('--max-file-size')
    expect(joined).toContain('-h, --help')
    expect(joined).toContain('104857600')
  })

  // --------------------------------------------
  // Global options passed via globalOptions parameter
  // --------------------------------------------
  it('should use global options passed as parameter', async () => {
    // Arrange: ensure no env vars are set
    delete process.env['DB_PATH']
    delete process.env['BASE_DIR']
    delete process.env['CACHE_DIR']
    delete process.env['MODEL_NAME']
    delete process.env['MAX_FILE_SIZE']

    const filePath = resolve('/tmp/test/document.md')
    mocks.stat.mockResolvedValue(mockFileStat())
    setupSuccessfulIngestion()

    // Act: pass global options via second parameter, ingest-specific via args
    const { error } = await captureStderr(() =>
      runIngest(['--base-dir', '/cli/base', '--max-file-size', '555', filePath], {
        dbPath: '/cli/db',
        cacheDir: '/cli/cache',
        modelName: 'cli-model',
      })
    )

    // Assert: returns normally without process.exit
    expect(error).toBeUndefined()
    expect(process.exitCode).toBeUndefined()

    // Assert: createVectorStore was called with global db-path
    const { createVectorStore } = await import('../../cli/common.js')
    expect(createVectorStore).toHaveBeenCalledWith(expect.objectContaining({ dbPath: '/cli/db' }))

    // Assert: createEmbedder was called with global model-name and cache-dir
    const { createEmbedder } = await import('../../cli/common.js')
    expect(createEmbedder).toHaveBeenCalledWith(
      expect.objectContaining({
        modelName: 'cli-model',
        cacheDir: '/cli/cache',
      })
    )

    // Assert: DocumentParser was called with the resolved multi-root config.
    // The CLI resolver mock echoes CLI roots verbatim, so a single
    // --base-dir A surfaces as `baseDirs: ['/cli/base']` here. P2-T1 keeps
    // the single-root scan path identical; only the constructor shape
    // changes (baseDir → baseDirs).
    const { DocumentParser } = await import('../../parser/index.js')
    expect(DocumentParser).toHaveBeenCalledWith(
      expect.objectContaining({
        baseDirs: ['/cli/base'],
        maxFileSize: 555,
      })
    )
  })

  // --------------------------------------------
  // Unknown options error (including global flags after subcommand)
  // --------------------------------------------
  it('should error when global flags are passed after subcommand', async () => {
    // Act
    const { output, error } = await captureStderr(() =>
      runIngest(['/some/path', '--db-path', '/db'])
    )

    // Assert: exit(1) with unknown option error
    expect(error).toBeInstanceOf(Error)
    expect(expectError(error).message).toBe('process.exit(1)')

    const joined = output.join('\n')
    expect(joined).toContain('Unknown option: --db-path')
  })

  // --------------------------------------------
  // parseArgs unit tests
  // --------------------------------------------
  describe('parseArgs', () => {
    it('should parse positional argument only', () => {
      const result = parseArgs(['/some/path'])
      expect(result).toEqual({
        positional: '/some/path',
        options: {},
        help: false,
      })
    })

    it('should parse ingest-specific flags with positional', () => {
      const result = parseArgs(['--base-dir', '/base', '--max-file-size', '1024', '/target'])

      expect(result.positional).toBe('/target')
      expect(result.options).toEqual({
        baseDirs: ['/base'],
        maxFileSize: 1024,
      })
      expect(result.help).toBe(false)
    })

    it('should accumulate repeated --base-dir into baseDirs array in CLI order', () => {
      const result = parseArgs(['--base-dir', '/a', '--base-dir', '/b', '/target'])
      expect(result.positional).toBe('/target')
      expect(result.options.baseDirs).toEqual(['/a', '/b'])
    })

    it('should parse --help flag', () => {
      const result = parseArgs(['--help'])
      expect(result.help).toBe(true)
    })

    it('should parse -h flag', () => {
      const result = parseArgs(['-h'])
      expect(result.help).toBe(true)
    })

    it('should handle flags after positional', () => {
      const result = parseArgs(['/target', '--base-dir', '/base'])
      expect(result.positional).toBe('/target')
      expect(result.options.baseDirs).toEqual(['/base'])
    })

    it('should error when --max-file-size value is missing', () => {
      const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {})
      try {
        expect(() => parseArgs(['--max-file-size'])).toThrow('process.exit(1)')
        expect(errorSpy).toHaveBeenCalledWith('Missing value for --max-file-size')
      } finally {
        errorSpy.mockRestore()
      }
    })

    it('should error when --max-file-size value is non-numeric', () => {
      const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {})
      try {
        expect(() => parseArgs(['--max-file-size', 'abc', '/target'])).toThrow('process.exit(1)')
        expect(errorSpy).toHaveBeenCalledWith('Invalid value for --max-file-size: "abc"')
      } finally {
        errorSpy.mockRestore()
      }
    })

    it('should error when --max-file-size value contains trailing non-digits', () => {
      const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {})
      try {
        expect(() => parseArgs(['--max-file-size', '100abc', '/target'])).toThrow('process.exit(1)')
        expect(errorSpy).toHaveBeenCalledWith('Invalid value for --max-file-size: "100abc"')
      } finally {
        errorSpy.mockRestore()
      }
    })

    it('should parse --chunk-min-length flag', () => {
      const result = parseArgs(['--chunk-min-length', '100', '/target'])
      expect(result.positional).toBe('/target')
      expect(result.options.chunkMinLength).toBe(100)
    })

    it('should error when --chunk-min-length value is non-numeric', () => {
      const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {})
      try {
        expect(() => parseArgs(['--chunk-min-length', 'abc', '/target'])).toThrow('process.exit(1)')
        expect(errorSpy).toHaveBeenCalledWith('Invalid value for --chunk-min-length: "abc"')
      } finally {
        errorSpy.mockRestore()
      }
    })
  })

  // --------------------------------------------
  // Issue #79 regression: multiple positional args
  // --------------------------------------------
  it('should error with message when extra positional arguments are given (issue #79)', async () => {
    // Act
    const { output, error } = await captureStderr(() => runIngest(['/path1', '/path2']))

    // Assert: exit(1) with descriptive error
    expect(error).toBeInstanceOf(Error)
    expect(expectError(error).message).toBe('process.exit(1)')

    const joined = output.join('\n')
    expect(joined).toContain('Unexpected argument: /path2')
    expect(joined).toContain('Only one path is accepted')
  })

  // --------------------------------------------
  // No arguments shows usage
  // --------------------------------------------
  it('should show usage and exit with code 1 when no arguments provided', async () => {
    // Act
    const { output, error } = await captureStderr(() => runIngest([]))

    // Assert: exit(1)
    expect(error).toBeInstanceOf(Error)
    expect(expectError(error).message).toBe('process.exit(1)')

    const joined = output.join('\n')
    expect(joined).toContain('Usage: mcp-local-rag ingest')
  })

  // --------------------------------------------
  // resolveConfig validation
  // --------------------------------------------
  describe('resolveConfig validation', () => {
    afterEach(() => {
      delete process.env['MAX_FILE_SIZE']
      delete process.env['CHUNK_MIN_LENGTH']
    })

    it.each(['0', '-100', '999999999', '1e8', '0x100000', '100MB', '52428800.9'])(
      'should error naming MAX_FILE_SIZE when the env var is %s',
      async (value) => {
        process.env['MAX_FILE_SIZE'] = value
        const globalConfig = resolveGlobalConfig({})
        const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {})
        try {
          await expect(resolveConfig(globalConfig, {})).rejects.toThrow('process.exit(1)')
          expect(errorSpy).toHaveBeenCalledWith(
            expect.stringContaining(`Invalid MAX_FILE_SIZE value: "${value}"`)
          )
          expect(errorSpy).not.toHaveBeenCalledWith(expect.stringContaining('--max-file-size'))
        } finally {
          errorSpy.mockRestore()
        }
      }
    )

    it.each(['0', '10001', '50abc', '50.5', '1e2'])(
      'should error naming CHUNK_MIN_LENGTH when the env var is %s',
      async (value) => {
        process.env['CHUNK_MIN_LENGTH'] = value
        const globalConfig = resolveGlobalConfig({})
        const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {})
        try {
          await expect(resolveConfig(globalConfig, {})).rejects.toThrow('process.exit(1)')
          expect(errorSpy).toHaveBeenCalledWith(
            expect.stringContaining(`Invalid CHUNK_MIN_LENGTH value: "${value}"`)
          )
          expect(errorSpy).not.toHaveBeenCalledWith(expect.stringContaining('--chunk-min-length'))
        } finally {
          errorSpy.mockRestore()
        }
      }
    )

    it('should resolve maxFileSize from a valid MAX_FILE_SIZE env var', async () => {
      process.env['MAX_FILE_SIZE'] = '1024'
      const globalConfig = resolveGlobalConfig({})
      const result = await resolveConfig(globalConfig)
      expect(result.maxFileSize).toBe(1024)
    })

    it('should error when --base-dir CLI option points to sensitive path', async () => {
      const globalConfig = resolveGlobalConfig({})
      const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {})
      try {
        await expect(resolveConfig(globalConfig, { baseDirs: ['/proc/self'] })).rejects.toThrow(
          'process.exit(1)'
        )
        expect(errorSpy).toHaveBeenCalledWith(expect.stringContaining('sensitive system path'))
      } finally {
        errorSpy.mockRestore()
      }
    })

    it('should error when --max-file-size CLI option is zero', async () => {
      const globalConfig = resolveGlobalConfig({})
      const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {})
      try {
        await expect(resolveConfig(globalConfig, { maxFileSize: 0 })).rejects.toThrow(
          'process.exit(1)'
        )
        expect(errorSpy).toHaveBeenCalledWith(
          expect.stringContaining('must be between 1 and 524288000')
        )
      } finally {
        errorSpy.mockRestore()
      }
    })

    it('should resolve chunkMinLength from CLI option', async () => {
      const globalConfig = resolveGlobalConfig({})
      const result = await resolveConfig(globalConfig, { chunkMinLength: 200 })
      expect(result.chunkMinLength).toBe(200)
    })

    it('should resolve chunkMinLength from CHUNK_MIN_LENGTH env var', async () => {
      process.env['CHUNK_MIN_LENGTH'] = '300'
      const globalConfig = resolveGlobalConfig({})
      const result = await resolveConfig(globalConfig)
      expect(result.chunkMinLength).toBe(300)
    })

    it('should prefer CLI chunkMinLength over env var', async () => {
      process.env['CHUNK_MIN_LENGTH'] = '300'
      const globalConfig = resolveGlobalConfig({})
      const result = await resolveConfig(globalConfig, { chunkMinLength: 100 })
      expect(result.chunkMinLength).toBe(100)
    })

    it('should leave chunkMinLength undefined when not specified', async () => {
      const globalConfig = resolveGlobalConfig({})
      const result = await resolveConfig(globalConfig)
      expect(result.chunkMinLength).toBeUndefined()
    })

    it('should error when --chunk-min-length CLI option is zero', async () => {
      const globalConfig = resolveGlobalConfig({})
      const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {})
      try {
        await expect(resolveConfig(globalConfig, { chunkMinLength: 0 })).rejects.toThrow(
          'process.exit(1)'
        )
        expect(errorSpy).toHaveBeenCalledWith(
          expect.stringContaining('must be between 1 and 10000')
        )
      } finally {
        errorSpy.mockRestore()
      }
    })

    it('should error when --chunk-min-length CLI option exceeds 10000', async () => {
      const globalConfig = resolveGlobalConfig({})
      const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {})
      try {
        await expect(resolveConfig(globalConfig, { chunkMinLength: 10001 })).rejects.toThrow(
          'process.exit(1)'
        )
        expect(errorSpy).toHaveBeenCalledWith(
          expect.stringContaining('must be between 1 and 10000')
        )
      } finally {
        errorSpy.mockRestore()
      }
    })
  })

  // Multi-root fallback, driven from the resolver mock, so no real env or
  // filesystem.
  describe('multi-root precedence (P2-T3)', () => {
    it('uses BASE_DIRS fallback (multi-root) when no --base-dir is provided', async () => {
      // Arrange: no CLI roots; resolver returns the BASE_DIRS-driven multi-root
      // set. The CLI must forward an empty `cliRoots` array so the resolver
      // applies env precedence. Post-Finding-#1: only the positional path is
      // scanned, but the resolver invocation contract is still
      // "empty cliRoots when --base-dir is absent".
      const envRootA = resolve('/env/multi/a')
      const envRootB = resolve('/env/multi/b')
      mocks.stat.mockResolvedValueOnce(mockDirStat()).mockResolvedValueOnce(mockDirStat())
      mocks.resolveCliBaseDirs.mockResolvedValue({
        config: { baseDirs: [envRootA, envRootB] },
        warnings: [],
      })
      setupMockReaddir({ [envRootA]: [mockDirent('a.md')] })
      setupSuccessfulIngestion()

      // Act
      const { output, error } = await captureStderr(() => runIngest([envRootA]))

      // Assert: resolver invoked with empty CLI roots.
      expect(error).toBeUndefined()
      expect(mocks.resolveCliBaseDirs).toHaveBeenCalledWith([])

      // Assert: only the positional tree was scanned.
      const joined = output.join('\n')
      expect(joined).toContain('Found 1 file(s) to ingest')
      expect(joined).toContain(resolve(envRootA, 'a.md'))
      expect(joined).not.toContain(resolve(envRootB, 'b.md'))
    })
  })
})
