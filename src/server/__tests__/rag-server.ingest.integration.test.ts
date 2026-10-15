// RAG MCP Server Integration Test - Re-ingestion & Error Handling
// Split from: rag-server.integration.test.ts (AC-008, AC-009)

import { mkdirSync, rmSync, writeFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { afterAll, beforeAll, describe, expect, it, vi } from 'vitest'
import { buildDocxFixture, headingXml, tableXml } from '../../__tests__/docx-fixture.js'
import { buildPdfWithImageBytes } from '../../__tests__/pdf-image-fixture.js'
import { testModelCacheDir, withTestDevice } from '../../__tests__/test-device.js'
import { parseJson, privateMembers } from '../../__tests__/test-doubles.js'
import type { Embedder } from '../../embedder/index.js'
import { parseHydratedVisualAttachments, type VisualAttachment } from '../../vectordb/types.js'
import { RAGServer } from '../index.js'

function deterministicEmbeddings(texts: string[]): number[][] {
  return texts.map((_text, index) => {
    const vector = new Array<number>(384).fill(0)
    vector[index % vector.length] = 1
    return vector
  })
}

describe('AC-008: File Re-ingestion', () => {
  let localRagServer: RAGServer
  const localTestDbPath = resolve('./tmp/test-lancedb-ac008')
  const localTestDataDir = resolve('./tmp/test-data-ac008')

  beforeAll(async () => {
    mkdirSync(localTestDbPath, { recursive: true })
    mkdirSync(localTestDataDir, { recursive: true })

    localRagServer = new RAGServer(
      withTestDevice({
        dbPath: localTestDbPath,
        modelName: 'Xenova/all-MiniLM-L6-v2',
        cacheDir: testModelCacheDir(),
        baseDir: localTestDataDir,
        maxFileSize: 100 * 1024 * 1024,
      })
    )

    await localRagServer.initialize()
  })

  afterAll(async () => {
    await localRagServer.close()
    rmSync(localTestDbPath, { recursive: true, force: true })
    rmSync(localTestDataDir, { recursive: true, force: true })
  })

  it('compacts once after a successful direct ingestion', async () => {
    const testFile = resolve(localTestDataDir, 'test-optimize.txt')
    writeFileSync(testFile, 'Content that produces a persisted chunk. '.repeat(20))
    const vectorStore = privateMembers<{
      vectorStore: { optimize(): Promise<void> }
    }>(localRagServer).vectorStore
    const optimizeSpy = vi.spyOn(vectorStore, 'optimize')

    await localRagServer.handleIngestFile({ filePath: testFile })

    expect(optimizeSpy).toHaveBeenCalledTimes(1)
    optimizeSpy.mockRestore()
  })

  it('persists and queries one coherent DOCX table row with the core title', async () => {
    const testFile = resolve(localTestDataDir, 'issue-176.docx')
    const expectedTitle = 'DOCX Field Reference'
    const expectedValues = [
      '42',
      'Retry Policy Identifier',
      'Optional',
      'Integer(11)',
      'First sentence. Second sentence.',
    ]
    const fixture = await buildDocxFixture({
      coreTitle: expectedTitle,
      bodyXml: `${headingXml('Heading Fallback')}${tableXml([
        ['Field No.', 'Field Name', 'Required', 'Type', 'Description'],
        expectedValues,
      ])}`,
    })
    writeFileSync(testFile, fixture)

    const ingestResult = await localRagServer.handleIngestFile({ filePath: testFile })
    const ingestData = parseJson<{
      chunkCount: number
      fileTitle: string | null
    }>(ingestResult.content[0].text)
    expect(ingestData.chunkCount).toBeGreaterThan(0)
    expect(ingestData.fileTitle).toBe(expectedTitle)

    const vectorStore = privateMembers<{
      vectorStore: {
        getChunksByFilePath(
          filePath: string
        ): Promise<Array<{ text: string; fileTitle: string | null }>>
      }
    }>(localRagServer).vectorStore
    const persisted = await vectorStore.getChunksByFilePath(testFile)
    const rowChunk = persisted.find((chunk) =>
      expectedValues.every((value) => chunk.text.includes(value))
    )
    expect(rowChunk).toBeDefined()
    expect(rowChunk?.fileTitle).toBe(expectedTitle)

    const queryResult = await localRagServer.handleQueryDocuments({
      query: 'Retry Policy Identifier 42',
      limit: 10,
    })
    const queryRows = parseJson<
      Array<{
        filePath: string
        text: string
        fileTitle: string | null
      }>
    >(queryResult.content[0].text)
    expect(
      queryRows.find(
        (row) =>
          row.filePath === testFile &&
          row.fileTitle === expectedTitle &&
          expectedValues.every((value) => row.text.includes(value))
      )
    ).toBeDefined()
  }, 60000)

  it('writes configured PDF image state and direct ingest replaces it request-locally when disabled', async () => {
    const testFile = resolve(localTestDataDir, 'image-state.pdf')
    const imageDbPath = resolve('./tmp/test-lancedb-direct-image-state')
    writeFileSync(testFile, buildPdfWithImageBytes())
    mkdirSync(imageDbPath, { recursive: true })

    const makeConfiguredServer = (storeImages: boolean) => {
      const server = new RAGServer(
        withTestDevice({
          dbPath: imageDbPath,
          modelName: 'Xenova/all-MiniLM-L6-v2',
          cacheDir: testModelCacheDir(),
          baseDir: localTestDataDir,
          maxFileSize: 100 * 1024 * 1024,
          storeImages,
        })
      )
      const embedder = privateMembers<{ embedder: Embedder }>(server).embedder
      vi.spyOn(embedder, 'embedBatch').mockImplementation(async (texts) =>
        deterministicEmbeddings(texts)
      )
      return server
    }
    const readRows = async (server: RAGServer) =>
      await privateMembers<{
        vectorStore: {
          getChunksByFilePath(path: string): Promise<Array<{ visualAttachments: string | null }>>
        }
      }>(server).vectorStore.getChunksByFilePath(testFile)

    const enabled = makeConfiguredServer(true)
    await enabled.initialize()
    try {
      await enabled.handleIngestFile({ filePath: testFile })
      const rows = await readRows(enabled)
      expect(rows.length).toBeGreaterThan(0)
      const attachmentRows = rows
        .map((row) => parseJson<VisualAttachment[]>(row.visualAttachments ?? '[]'))
        .filter((attachments) => attachments.length > 0)
      // The fixture holds exactly one raster image, so it is stored exactly once.
      expect(attachmentRows.flat().map((attachment) => attachment.imageIndex)).toEqual([0])
      for (const attachments of attachmentRows) {
        expect(parseHydratedVisualAttachments(JSON.stringify(attachments))).toEqual({
          attachments,
          omittedCount: 0,
        })
      }
    } finally {
      await enabled.close()
    }

    const disabled = makeConfiguredServer(false)
    await disabled.initialize()
    try {
      await disabled.handleIngestFile({ filePath: testFile })
      const rows = await readRows(disabled)
      expect(rows.length).toBeGreaterThan(0)
      expect(rows.every((row) => row.visualAttachments === '[]')).toBe(true)
    } finally {
      await disabled.close()
      rmSync(imageDbPath, { recursive: true, force: true })
    }
  }, 60000)
})

describe('AC-009: Error Handling (Complete)', () => {
  let localRagServer: RAGServer
  const localTestDbPath = resolve('./tmp/test-lancedb-ac009')
  const localTestDataDir = resolve('./tmp/test-data-ac009')

  beforeAll(async () => {
    mkdirSync(localTestDbPath, { recursive: true })
    mkdirSync(localTestDataDir, { recursive: true })

    localRagServer = new RAGServer(
      withTestDevice({
        dbPath: localTestDbPath,
        modelName: 'Xenova/all-MiniLM-L6-v2',
        cacheDir: testModelCacheDir(),
        baseDir: localTestDataDir,
        maxFileSize: 100 * 1024 * 1024,
      })
    )

    await localRagServer.initialize()
  })

  afterAll(async () => {
    await localRagServer.close()
    rmSync(localTestDbPath, { recursive: true, force: true })
    rmSync(localTestDataDir, { recursive: true, force: true })
  })

  // AC interpretation: [Security] Path traversal attacks are rejected (S-002)
  // Validation: Call ingest_file with invalid path like `../../etc/passwd`, ValidationError is returned
  it('Path traversal attack (e.g., ../../etc/passwd) rejected with ValidationError (S-002)', async () => {
    await expect(localRagServer.handleIngestFile({ filePath: '../../etc/passwd' })).rejects.toThrow(
      'absolute path'
    )
  })
})
