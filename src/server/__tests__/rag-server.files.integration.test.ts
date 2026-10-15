// RAG MCP Server Integration Test - File Management
// Split from: rag-server.integration.test.ts (AC-007)

import { mkdirSync, rmSync, symlinkSync, writeFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import { testModelCacheDir, withTestDevice } from '../../__tests__/test-device.js'
import { expectError } from '../../__tests__/test-doubles.js'
import { RAGServer } from '../index.js'

describe('AC-007: File Management', () => {
  let localRagServer: RAGServer
  const localTestDbPath = resolve('./tmp/test-lancedb-ac007')
  const localTestDataDir = resolve('./tmp/test-data-ac007')
  const localCacheDir = resolve('./tmp/test-cache-ac007')

  beforeAll(async () => {
    mkdirSync(localTestDbPath, { recursive: true })
    mkdirSync(localTestDataDir, { recursive: true })
    mkdirSync(localCacheDir, { recursive: true })

    localRagServer = new RAGServer(
      withTestDevice({
        dbPath: localTestDbPath,
        modelName: 'Xenova/all-MiniLM-L6-v2',
        cacheDir: testModelCacheDir(localCacheDir),
        baseDir: localTestDataDir,
        maxFileSize: 100 * 1024 * 1024,
      })
    )

    await localRagServer.initialize()

    // Ingest test documents (3 files)
    const testFile1 = resolve(localTestDataDir, 'test-file-1.txt')
    writeFileSync(testFile1, 'This is test file 1. '.repeat(50))
    await localRagServer.handleIngestFile({ filePath: testFile1 })

    const testFile2 = resolve(localTestDataDir, 'test-file-2.txt')
    writeFileSync(testFile2, 'This is test file 2. '.repeat(30))
    await localRagServer.handleIngestFile({ filePath: testFile2 })

    const testFile3 = resolve(localTestDataDir, 'test-file-3.txt')
    writeFileSync(testFile3, 'This is test file 3. '.repeat(20))
    await localRagServer.handleIngestFile({ filePath: testFile3 })
  }, 120000)

  afterAll(async () => {
    await localRagServer.close()
    rmSync(localTestDbPath, { recursive: true, force: true })
    rmSync(localTestDataDir, { recursive: true, force: true })
  })

  // AC interpretation: [Functional requirement] Filename, path, chunk count, ingestion time accurately displayed
  // Validation: list_files result contains detailed information for each file
  it('list_files result accurately contains detailed information (filePath, chunkCount, timestamp) for each file', async () => {
    const result = await localRagServer.handleListFiles()
    expect(result.content.length).toBe(1)
    expect(result.content[0].type).toBe('text')

    const files = JSON.parse(result.content[0].text)
    const { files: filesInBaseDir } = files
    expect(filesInBaseDir.length).toBe(3)

    // Verify test-file-1.txt information
    const testFile1Path = resolve(localTestDataDir, 'test-file-1.txt')
    const file1 = filesInBaseDir.find((f: { filePath: string }) => f.filePath === testFile1Path)
    expect(file1).toBeDefined()
    expect(file1.chunkCount).toBeGreaterThan(0)
    expect(file1.timestamp).toMatch(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}/)

    // Verify test-file-2.txt information
    const testFile2Path = resolve(localTestDataDir, 'test-file-2.txt')
    const file2 = filesInBaseDir.find((f: { filePath: string }) => f.filePath === testFile2Path)
    expect(file2).toBeDefined()
    expect(file2.chunkCount).toBeGreaterThan(0)
    expect(file2.timestamp).toMatch(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}/)

    // Verify test-file-3.txt information
    const testFile3Path = resolve(localTestDataDir, 'test-file-3.txt')
    const file3 = filesInBaseDir.find((f: { filePath: string }) => f.filePath === testFile3Path)
    expect(file3).toBeDefined()
    expect(file3.chunkCount).toBeGreaterThan(0)
    expect(file3.timestamp).toMatch(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}/)
  })

  // AC interpretation: [Functional requirement] Supported file in BASE_DIR not yet ingested appears as ingested: false
  // Validation: Place a file in BASE_DIR without ingesting it, list_files shows { filePath, ingested: false }
  it('File in BASE_DIR not yet ingested appears with ingested: false in list_files', async () => {
    const uningestedFile = resolve(localTestDataDir, 'not-yet-ingested.txt')
    writeFileSync(uningestedFile, 'This file has not been ingested.')

    try {
      const result = await localRagServer.handleListFiles()
      const files = JSON.parse(result.content[0].text)

      const entry = files.files.find((f: { filePath: string }) => f.filePath === uningestedFile)
      expect(entry).toBeDefined()
      expect(entry.ingested).toBe(false)
      expect(entry.chunkCount).toBeUndefined()
      expect(entry.timestamp).toBeUndefined()
    } finally {
      rmSync(uningestedFile, { force: true })
    }
  })

  // AC interpretation: [Functional requirement] System status displayed via status tool
  // Validation: Call status, document count, chunk count, memory usage, uptime are returned
  it('System status (documentCount, chunkCount, memoryUsage, uptime) displayed via status tool', async () => {
    const result = await localRagServer.handleStatus()

    expect(result).toBeDefined()
    expect(result.content).toBeDefined()
    expect(result.content.length).toBe(1)
    expect(result.content[0].type).toBe('text')

    const status = JSON.parse(result.content[0].text)
    expect(status.documentCount).toBe(3)
    expect(status.chunkCount).toBeGreaterThan(0)
    expect(status.memoryUsage).toBeGreaterThan(0)
    expect(status.uptime).toBeGreaterThan(0)
  })

  describe('System-managed path exclusion from list_files', () => {
    let excludeServer: RAGServer
    const excludeTestBase = resolve('./tmp/test-exclude-base')
    const excludeTestDb = resolve(excludeTestBase, 'lancedb')
    const excludeTestCache = resolve(excludeTestBase, 'models')

    beforeAll(async () => {
      mkdirSync(excludeTestBase, { recursive: true })
      mkdirSync(excludeTestDb, { recursive: true })
      mkdirSync(excludeTestCache, { recursive: true })

      writeFileSync(resolve(excludeTestDb, 'db-internal.txt'), 'Database internal file')
      writeFileSync(resolve(excludeTestCache, 'model-cache.txt'), 'Model cache file')
      writeFileSync(resolve(excludeTestBase, 'user-document.txt'), 'User document content')

      mkdirSync(resolve(excludeTestBase, 'docs'), { recursive: true })
      writeFileSync(resolve(excludeTestBase, 'docs', 'notes.txt'), 'Notes in docs subdirectory')

      excludeServer = new RAGServer(
        withTestDevice({
          dbPath: excludeTestDb,
          modelName: 'Xenova/all-MiniLM-L6-v2',
          cacheDir: testModelCacheDir(excludeTestCache),
          baseDir: excludeTestBase,
          maxFileSize: 100 * 1024 * 1024,
        })
      )

      await excludeServer.initialize()
    }, 120000)

    afterAll(async () => {
      rmSync(excludeTestBase, { recursive: true, force: true })
    })

    it('System-managed paths excluded from list_files scan', async () => {
      const result = await excludeServer.handleListFiles()
      const parsed = JSON.parse(result.content[0].text)

      const filePaths: string[] = parsed.files.map((f: { filePath: string }) => f.filePath)

      expect(filePaths).toContain(resolve(excludeTestBase, 'user-document.txt'))
      expect(filePaths).toContain(resolve(excludeTestBase, 'docs', 'notes.txt'))

      expect(filePaths).not.toContain(resolve(excludeTestDb, 'db-internal.txt'))
      expect(filePaths).not.toContain(resolve(excludeTestCache, 'model-cache.txt'))
    })

    it('raw-data .md files inside dbPath excluded from files array', async () => {
      await excludeServer.handleIngestData({
        content:
          'Integration test content for raw-data exclusion verification. ' +
          'This content is long enough to produce at least one chunk in the system.',
        metadata: {
          source: 'https://example.com/exclude-test',
          format: 'text',
        },
      })

      const result = await excludeServer.handleListFiles()
      const parsed = JSON.parse(result.content[0].text)

      const filePaths: string[] = parsed.files.map((f: { filePath: string }) => f.filePath)

      const rawDataFiles = filePaths.filter((fp) => fp.includes('raw-data'))
      expect(rawDataFiles).toHaveLength(0)

      expect(parsed.sources.length).toBeGreaterThan(0)
      const sourceEntry = parsed.sources.find(
        (s: { source?: string }) => s.source === 'https://example.com/exclude-test'
      )
      expect(sourceEntry).toBeDefined()
    }, 30000)
  })
})

// AC-008, AC-011, AC-012, AC-013 — list_files multi-root contract (P3-T2)
//
// Covers the multi-root response shape produced by `handleListFiles` when
// the server is configured with `baseDirs`: top-level `baseDirs`, preserved
// legacy `baseDir`, sources without root annotation, and exact-path
// de-duplication across roots.
describe('AC-008: list_files multi-root contract', () => {
  const multiRootBase = resolve('./tmp/test-list-multi-root')
  const rootA = resolve(multiRootBase, 'rootA')
  const rootB = resolve(multiRootBase, 'rootB')
  const multiDbPath = resolve(multiRootBase, 'lancedb')
  const multiCacheDir = resolve(multiRootBase, 'cache-models')

  let multiServer: RAGServer

  beforeAll(async () => {
    mkdirSync(rootA, { recursive: true })
    mkdirSync(rootB, { recursive: true })
    mkdirSync(multiDbPath, { recursive: true })
    mkdirSync(multiCacheDir, { recursive: true })

    writeFileSync(resolve(rootA, 'a-file.txt'), 'Content in root A')

    multiServer = new RAGServer(
      withTestDevice({
        dbPath: multiDbPath,
        modelName: 'Xenova/all-MiniLM-L6-v2',
        cacheDir: testModelCacheDir(multiCacheDir),
        baseDirs: [rootA, rootB],
        maxFileSize: 100 * 1024 * 1024,
      })
    )

    await multiServer.initialize()
  }, 60000)

  afterAll(async () => {
    await multiServer.close()
    rmSync(multiRootBase, { recursive: true, force: true })
  })

  // AC interpretation: [AC-008/AC-011] Top-level `baseDirs` exposes the resolved effective roots in configured order
  // Validation: Parsed response `baseDirs` deep-equals `[rootA, rootB]` matching `RAGServer` configuration order
  it('top-level baseDirs equals the resolved effective roots in order', async () => {
    const result = await multiServer.handleListFiles()
    const parsed = JSON.parse(result.content[0].text)

    expect(parsed.baseDirs).toEqual([rootA, rootB])
  })

  // AC interpretation: [AC-008/AC-012] Legacy single-root `baseDir` field preserved and equal to `baseDirs[0]` for backward compatibility
  // Validation: `parsed.baseDir` equals `rootA` and equals `parsed.baseDirs[0]`
  it('top-level baseDir equals baseDirs[0] (legacy compatibility)', async () => {
    const result = await multiServer.handleListFiles()
    const parsed = JSON.parse(result.content[0].text)

    expect(parsed.baseDir).toBe(rootA)
    expect(parsed.baseDir).toBe(parsed.baseDirs[0])
  })

  // AC interpretation: [AC-008] Raw-data ingested via `ingest_data` is reported under `sources` and carries no producing-root annotation
  // Validation: Ingested raw source appears in `parsed.sources`, its `baseDir` is undefined, and no raw-data path leaks into `parsed.files`
  it('raw-data ingested via ingest_data appears under sources without baseDir', async () => {
    await multiServer.handleIngestData({
      content:
        'Multi-root raw-data source test content. ' +
        'Long enough to produce at least one chunk for the integration verification.',
      metadata: {
        source: 'https://example.com/multi-root-source',
        format: 'text',
      },
    })

    const result = await multiServer.handleListFiles()
    const parsed = JSON.parse(result.content[0].text)

    const sourceEntry = parsed.sources.find(
      (s: { source?: string }) => s.source === 'https://example.com/multi-root-source'
    )
    expect(sourceEntry).toBeDefined()
    // sources carry no producing-root annotation
    expect(sourceEntry.baseDir).toBeUndefined()

    // raw-data files do not leak into `files`
    const filePaths: string[] = parsed.files.map((f: { filePath: string }) => f.filePath)
    const rawDataLeaks = filePaths.filter((fp) => fp.includes('raw-data'))
    expect(rawDataLeaks).toHaveLength(0)
  }, 30000)

  // AC interpretation: [AC-008] When the same underlying file is reachable from multiple roots (via a symlink under rootB pointing at a file in rootA),
  // exact-path de-duplication produces exactly one entry whose `baseDir` is the first root in `baseDirs` order (first-occurrence wins).
  // Validation: Create a symlink under rootB targeting an existing file in rootA, then call `handleListFiles` and assert (a) exactly one entry
  // exists for the shared file and (b) its `baseDir` equals rootA (the first configured root). This exercises the seenPaths logic at
  // src/server/index.ts:651-671 and documents the first-occurrence-wins acceptance criterion across roots.
  it('de-duplicates files reachable from multiple roots via symlink, keeping the first root in baseDirs order', async () => {
    const sharedTargetPath = resolve(rootA, 'a-file.txt')
    const symlinkPath = resolve(rootB, 'a-file.txt')

    try {
      symlinkSync(sharedTargetPath, symlinkPath)
    } catch (error) {
      // Symlink creation can fail on platforms without filesystem-link
      // privileges; in that environment the cross-root dedup contract is not
      // observable so we surface the failure rather than silently skipping.
      throw new Error(
        `symlink creation failed for cross-root dedup test: ${expectError(error).message}`,
        { cause: error }
      )
    }

    try {
      const result = await multiServer.handleListFiles()
      const parsed = JSON.parse(result.content[0].text)

      const entries = parsed.files.filter(
        (f: { filePath: string }) => f.filePath === sharedTargetPath || f.filePath === symlinkPath
      )
      expect(entries).toHaveLength(1)
      expect(entries[0].filePath).toBe(sharedTargetPath)
      expect(entries[0].baseDir).toBe(rootA)
    } finally {
      rmSync(symlinkPath, { force: true })
    }
  })
})

describe('AC-008: list_files single-root regression (legacy shape preserved)', () => {
  const singleBase = resolve('./tmp/test-list-single-root-regression')
  const singleData = resolve(singleBase, 'data')
  const singleDb = resolve(singleBase, 'db')
  const singleCache = resolve(singleBase, 'cache')

  let singleServer: RAGServer

  beforeAll(async () => {
    mkdirSync(singleData, { recursive: true })
    mkdirSync(singleDb, { recursive: true })
    mkdirSync(singleCache, { recursive: true })
    writeFileSync(resolve(singleData, 'only-file.txt'), 'single-root content')

    singleServer = new RAGServer(
      withTestDevice({
        dbPath: singleDb,
        modelName: 'Xenova/all-MiniLM-L6-v2',
        cacheDir: testModelCacheDir(singleCache),
        baseDir: singleData,
        maxFileSize: 100 * 1024 * 1024,
      })
    )

    await singleServer.initialize()
  }, 60000)

  afterAll(async () => {
    await singleServer.close()
    rmSync(singleBase, { recursive: true, force: true })
  })

  // AC interpretation: [AC-008/AC-012] Single-root legacy configuration still produces the legacy `baseDir` plus the additive `baseDirs` and per-file `baseDir` fields
  // Validation: `parsed.baseDir` equals the single root, `parsed.baseDirs` equals `[singleData]`, and each file entry's `baseDir` equals `singleData`
  it('preserves baseDir, adds baseDirs and per-file baseDir for single-root configs', async () => {
    const result = await singleServer.handleListFiles()
    const parsed = JSON.parse(result.content[0].text)

    expect(parsed.baseDir).toBe(singleData)
    expect(parsed.baseDirs).toEqual([singleData])

    const filePath = resolve(singleData, 'only-file.txt')
    const entry = parsed.files.find((f: { filePath: string }) => f.filePath === filePath)
    expect(entry).toBeDefined()
    expect(entry.baseDir).toBe(singleData)
  })
})
