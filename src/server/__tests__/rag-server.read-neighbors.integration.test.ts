// read_chunk_neighbors integration tests.
//
// Nothing is mocked; RAGServer, VectorStore, LanceDB and DocumentParser are
// all real.
// PRD: docs/prd/read-chunk-neighbors-prd.md

import { mkdirSync, rmSync, writeFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import { testModelCacheDir, withTestDevice } from '../../__tests__/test-device.js'
import { parseJson } from '../../__tests__/test-doubles.js'
import { isManagedRawDataPath } from '../../utils/raw-data-utils.js'
import { RAGServer } from '../index.js'
import type { ReadChunkNeighborsResultItem } from '../types.js'

function createTestRagServer(config: ConstructorParameters<typeof RAGServer>[0]): RAGServer {
  return new RAGServer(withTestDevice(config))
}

/**
 * Local helper: parse the JSON payload of a tool response.
 */
function parseItems(response: {
  content: Array<{ type: 'text'; text: string }>
}): ReadChunkNeighborsResultItem[] {
  const text = response.content[0]?.text
  if (typeof text !== 'string') {
    throw new Error('Response content[0].text is missing')
  }
  return parseJson<ReadChunkNeighborsResultItem[]>(text)
}

describe('read_chunk_neighbors integration', () => {
  // AC-001/002/008/018/019: the default window, its field set, and isTarget.
  describe('Test 1: Default window returns 5 sorted chunks with core fields and isTarget', () => {
    let ragServer: RAGServer
    const testDbPath = resolve('./tmp/test-lancedb-read-neighbors-t1')
    const testDataDir = resolve('./tmp/test-data-read-neighbors-t1')
    let ingestedFilePath: string

    beforeAll(async () => {
      mkdirSync(testDbPath, { recursive: true })
      mkdirSync(testDataDir, { recursive: true })
      ragServer = createTestRagServer({
        dbPath: testDbPath,
        modelName: 'Xenova/all-MiniLM-L6-v2',
        cacheDir: testModelCacheDir(),
        baseDir: testDataDir,
        maxFileSize: 100 * 1024 * 1024,
      })
      await ragServer.initialize()

      ingestedFilePath = resolve(testDataDir, 'default-window.txt')
      // Large enough content that chunker produces >= 7 chunks.
      writeFileSync(ingestedFilePath, 'The quick brown fox jumps over the lazy dog. '.repeat(200))
      const ingestRes = await ragServer.handleIngestFile({ filePath: ingestedFilePath })
      const ingest = JSON.parse(ingestRes.content[0].text)
      expect(ingest.chunkCount).toBeGreaterThanOrEqual(7)
    })

    afterAll(async () => {
      await ragServer.close()
      rmSync(testDbPath, { recursive: true, force: true })
      rmSync(testDataDir, { recursive: true, force: true })
    })

    it('returns 5 sorted items with core fields, exactly one isTarget true', async () => {
      const response = await ragServer.handleReadChunkNeighbors({
        filePath: ingestedFilePath,
        chunkIndex: 3,
      })
      const items = parseItems(response)

      expect(items).toHaveLength(5)
      expect(items.map((i) => i.chunkIndex)).toEqual([1, 2, 3, 4, 5])

      for (const item of items) {
        expect(typeof item.chunkIndex).toBe('number')
        expect(typeof item.text).toBe('string')
        expect(item.text.length).toBeGreaterThan(0)
        expect(item.filePath).toBe(ingestedFilePath)
        expect(typeof item.isTarget).toBe('boolean')
        // fileTitle is string | null per ReadChunkNeighborsResultItem
        expect(item.fileTitle === null || typeof item.fileTitle === 'string').toBe(true)
        // AC-002 minimal core: no score, no metadata on ChunkRow-derived items.
        expect('score' in item).toBe(false)
        expect('metadata' in item).toBe(false)
      }

      const targets = items.filter((i) => i.isTarget)
      expect(targets).toHaveLength(1)
      expect(targets[0]?.chunkIndex).toBe(3)
    })
  })

  // AC-005: a target near the start clamps the window instead of erroring.
  describe('Test 3: Near-start target returns clamped window (AC-005)', () => {
    let ragServer: RAGServer
    const testDbPath = resolve('./tmp/test-lancedb-read-neighbors-t3')
    const testDataDir = resolve('./tmp/test-data-read-neighbors-t3')
    let ingestedFilePath: string

    beforeAll(async () => {
      mkdirSync(testDbPath, { recursive: true })
      mkdirSync(testDataDir, { recursive: true })
      ragServer = createTestRagServer({
        dbPath: testDbPath,
        modelName: 'Xenova/all-MiniLM-L6-v2',
        cacheDir: testModelCacheDir(),
        baseDir: testDataDir,
        maxFileSize: 100 * 1024 * 1024,
      })
      await ragServer.initialize()

      ingestedFilePath = resolve(testDataDir, 'near-start.txt')
      writeFileSync(ingestedFilePath, 'Alpha beta gamma delta epsilon zeta eta theta. '.repeat(120))
      const ingestRes = await ragServer.handleIngestFile({ filePath: ingestedFilePath })
      const ingest = JSON.parse(ingestRes.content[0].text)
      expect(ingest.chunkCount).toBeGreaterThanOrEqual(4)
    })

    afterAll(async () => {
      await ragServer.close()
      rmSync(testDbPath, { recursive: true, force: true })
      rmSync(testDataDir, { recursive: true, force: true })
    })

    it('returns only existing chunks [0,1,2] with isTarget at 0', async () => {
      const response = await ragServer.handleReadChunkNeighbors({
        filePath: ingestedFilePath,
        chunkIndex: 0,
      })
      const items = parseItems(response)

      expect(items).toHaveLength(3)
      expect(items.map((i) => i.chunkIndex)).toEqual([0, 1, 2])
      for (const item of items) {
        expect(item.chunkIndex).toBeGreaterThanOrEqual(0)
      }
      expect(items[0]?.isTarget).toBe(true)
      expect(items[1]?.isTarget).toBe(false)
      expect(items[2]?.isTarget).toBe(false)
    })
  })

  // AC-006/019: a target that does not exist returns whatever of the range
  // does, with no isTarget, and an empty array when none of it exists.
  describe('Test 4: Missing target and fully out-of-range behavior (AC-006)', () => {
    let ragServer: RAGServer
    const testDbPath = resolve('./tmp/test-lancedb-read-neighbors-t4')
    const testDataDir = resolve('./tmp/test-data-read-neighbors-t4')
    let ingestedFilePath: string
    let chunkCount: number

    beforeAll(async () => {
      mkdirSync(testDbPath, { recursive: true })
      mkdirSync(testDataDir, { recursive: true })
      ragServer = createTestRagServer({
        dbPath: testDbPath,
        modelName: 'Xenova/all-MiniLM-L6-v2',
        cacheDir: testModelCacheDir(),
        baseDir: testDataDir,
        maxFileSize: 100 * 1024 * 1024,
      })
      await ragServer.initialize()

      ingestedFilePath = resolve(testDataDir, 'missing-target.txt')
      writeFileSync(
        ingestedFilePath,
        'Lorem ipsum dolor sit amet consectetur adipiscing elit. '.repeat(150)
      )
      const ingestRes = await ragServer.handleIngestFile({ filePath: ingestedFilePath })
      const ingest = JSON.parse(ingestRes.content[0].text)
      chunkCount = ingest.chunkCount
      expect(chunkCount).toBeGreaterThanOrEqual(3)
    })

    afterAll(async () => {
      await ragServer.close()
      rmSync(testDbPath, { recursive: true, force: true })
      rmSync(testDataDir, { recursive: true, force: true })
    })

    it('(a) target just past last index: returns surrounding chunks with all isTarget false', async () => {
      // Request chunkIndex = chunkCount (one past the last valid index = chunkCount-1)
      const response = await ragServer.handleReadChunkNeighbors({
        filePath: ingestedFilePath,
        chunkIndex: chunkCount,
      })
      const items = parseItems(response)

      expect(items.length).toBeGreaterThan(0)
      for (const item of items) {
        expect(item.isTarget).toBe(false)
        expect(item.chunkIndex).toBeLessThanOrEqual(chunkCount - 1)
      }
      // Strictly ascending
      for (let i = 1; i < items.length; i++) {
        const prev = items[i - 1]
        const curr = items[i]
        if (!prev || !curr) {
          throw new Error('Unexpected undefined item in ascending check')
        }
        expect(curr.chunkIndex).toBeGreaterThan(prev.chunkIndex)
      }
    })

    it('(b) target far outside document: returns empty array', async () => {
      const response = await ragServer.handleReadChunkNeighbors({
        filePath: ingestedFilePath,
        chunkIndex: 999,
      })
      const items = parseItems(response)
      expect(items).toEqual([])
    })
  })

  // AC-020: raw-data rows carry `source`, derived from the resolved path
  // rather than the input key — so file-backed rows must not carry it.
  describe('Test 6: Raw-data row includes source field (AC-020)', () => {
    let ragServer: RAGServer
    const testDbPath = resolve('./tmp/test-lancedb-read-neighbors-t6')
    const testDataDir = resolve('./tmp/test-data-read-neighbors-t6')
    const KNOWN_SOURCE = 'https://example.com/read-neighbors-source-field'
    let fileBackedPath: string

    beforeAll(async () => {
      mkdirSync(testDbPath, { recursive: true })
      mkdirSync(testDataDir, { recursive: true })
      ragServer = createTestRagServer({
        dbPath: testDbPath,
        modelName: 'Xenova/all-MiniLM-L6-v2',
        cacheDir: testModelCacheDir(),
        baseDir: testDataDir,
        maxFileSize: 100 * 1024 * 1024,
      })
      await ragServer.initialize()

      // Raw-data document
      const rawContent = `# Source field test\n\n${'Paragraph body content for source field test. '.repeat(120)}`
      await ragServer.handleIngestData({
        content: rawContent,
        metadata: { source: KNOWN_SOURCE, format: 'markdown' },
      })

      // File-backed document (cross-check negative)
      fileBackedPath = resolve(testDataDir, 'file-backed.txt')
      writeFileSync(fileBackedPath, 'File backed document content. '.repeat(100))
      await ragServer.handleIngestFile({ filePath: fileBackedPath })
    })

    afterAll(async () => {
      await ragServer.close()
      rmSync(testDbPath, { recursive: true, force: true })
      rmSync(testDataDir, { recursive: true, force: true })
    })

    it('raw-data items carry source === ingestion identifier', async () => {
      const response = await ragServer.handleReadChunkNeighbors({
        source: KNOWN_SOURCE,
        chunkIndex: 0,
      })
      const items = parseItems(response)

      expect(items.length).toBeGreaterThan(0)
      for (const item of items) {
        expect(typeof item.source).toBe('string')
        expect(item.source).toBe(KNOWN_SOURCE)
      }
    })

    it('file-backed items do NOT carry a source field', async () => {
      const response = await ragServer.handleReadChunkNeighbors({
        filePath: fileBackedPath,
        chunkIndex: 0,
      })
      const items = parseItems(response)

      expect(items.length).toBeGreaterThan(0)
      for (const item of items) {
        // source key is either absent or undefined on file-backed items
        expect(item.source).toBeUndefined()
      }
    })
  })

  // Regression: an empty string must read as "not provided" in BOTH the XOR
  // validation and the path resolution. Otherwise `source: ''` alongside a
  // valid filePath passes validation, then resolves against an empty-source
  // raw-data path and finds nothing.
  describe('Test 11: Empty-string filePath/source resolution (regression)', () => {
    let ragServer: RAGServer
    const testDbPath = resolve('./tmp/test-lancedb-read-neighbors-t11')
    const testDataDir = resolve('./tmp/test-data-read-neighbors-t11')
    const SOURCE = 'https://example.com/read-neighbors-empty-input'
    let ingestedFilePath: string

    beforeAll(async () => {
      mkdirSync(testDbPath, { recursive: true })
      mkdirSync(testDataDir, { recursive: true })
      ragServer = createTestRagServer({
        dbPath: testDbPath,
        modelName: 'Xenova/all-MiniLM-L6-v2',
        cacheDir: testModelCacheDir(),
        baseDir: testDataDir,
        maxFileSize: 100 * 1024 * 1024,
      })
      await ragServer.initialize()

      ingestedFilePath = resolve(testDataDir, 'empty-input-doc.txt')
      writeFileSync(ingestedFilePath, 'Empty input resolution test content. '.repeat(120))
      const fileIngest = JSON.parse(
        (await ragServer.handleIngestFile({ filePath: ingestedFilePath })).content[0].text
      )
      expect(fileIngest.chunkCount).toBeGreaterThanOrEqual(5)

      const content = `# Empty input source test\n\n${'Markdown paragraph content with stable wording. '.repeat(200)}`
      const dataIngest = JSON.parse(
        (
          await ragServer.handleIngestData({
            content,
            metadata: { source: SOURCE, format: 'markdown' },
          })
        ).content[0].text
      )
      expect(dataIngest.chunkCount).toBeGreaterThanOrEqual(3)
    })

    afterAll(async () => {
      await ragServer.close()
      rmSync(testDbPath, { recursive: true, force: true })
      rmSync(testDataDir, { recursive: true, force: true })
    })

    it('(a) empty source alongside a valid filePath resolves via filePath', async () => {
      const response = await ragServer.handleReadChunkNeighbors({
        filePath: ingestedFilePath,
        source: '',
        chunkIndex: 2,
      })
      const items = parseItems(response)

      expect(items.length).toBeGreaterThan(0)
      for (const item of items) {
        expect(item.filePath).toBe(ingestedFilePath)
        expect(item.source).toBeUndefined()
      }
      const targets = items.filter((i) => i.isTarget)
      expect(targets).toHaveLength(1)
      expect(targets[0]?.chunkIndex).toBe(2)
    })

    it('(b) empty filePath alongside a valid source resolves via source', async () => {
      const response = await ragServer.handleReadChunkNeighbors({
        filePath: '',
        source: SOURCE,
        chunkIndex: 1,
      })
      const items = parseItems(response)

      expect(items.length).toBeGreaterThan(0)
      const filePaths = new Set(items.map((i) => i.filePath))
      expect(filePaths.size).toBe(1)
      expect(isManagedRawDataPath(items[0]?.filePath ?? '', testDbPath)).toBe(true)
      for (const item of items) {
        expect(item.source).toBe(SOURCE)
      }
    })
  })
})
