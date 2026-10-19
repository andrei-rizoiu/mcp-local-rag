// Ingest Rollback Tests
// Tests rollback behavior when insertChunks fails during re-ingestion

import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { resolve } from 'node:path'
import type { MockInstance } from 'vitest'
import { afterAll, beforeAll, describe, expect, it, vi } from 'vitest'
import { testModelCacheDir, withTestDevice } from '../../__tests__/test-device.js'
import { expectError, expectRecord, privateMembers } from '../../__tests__/test-doubles.js'
import { embeddingSnapshotPath } from '../../embedder/embedding-snapshot.js'
import { buildVectorChunks } from '../../ingest/compute.js'
import * as rawDataUtils from '../../utils/raw-data-utils.js'
import type { VectorChunk, VectorStore } from '../../vectordb/index.js'
import { DatabaseError } from '../../vectordb/types.js'
import { RAGServer } from '../index.js'

/** Rollback is only observable through the server's own store. */
function privateVectorStore(server: RAGServer): VectorStore {
  return privateMembers<{ vectorStore: VectorStore }>(server).vectorStore
}

/**
 * Replace `filePath`'s rows with a set carrying `visualProfile`, standing in for
 * a prior visual ingest without loading a VLM. The profile is what zero-chunk
 * preservation must carry with the rest of the row.
 */
async function seedProfiledRows(
  store: VectorStore,
  filePath: string,
  visualProfile: string
): Promise<VectorChunk[]> {
  await store.deleteChunks(filePath)
  await store.insertChunks(
    buildVectorChunks({
      filePath,
      chunks: [0, 1].map((index) => ({
        index,
        text: `prior ${visualProfile} chunk ${index} with a caption marker for ${filePath}`,
        sourceStart: index * 80,
        sourceEnd: index * 80 + 79,
      })),
      embeddings: [0, 1].map((index) => {
        const vector = new Array<number>(384).fill(0)
        vector[index] = 1
        return vector
      }),
      fileTitle: 'Prior visual title',
      contentHash: `seeded-hash-${visualProfile}`,
      visualProfile,
    })
  )
  return await store.getChunksByFilePath(filePath)
}

describe('Ingest Rollback', () => {
  let ragServer: RAGServer
  const testDbPath = resolve('./tmp/test-lancedb-rollback')
  const testDataDir = resolve('./tmp/test-data-rollback')

  beforeAll(async () => {
    mkdirSync(testDbPath, { recursive: true })
    mkdirSync(testDataDir, { recursive: true })

    ragServer = new RAGServer(
      withTestDevice({
        dbPath: testDbPath,
        modelName: 'Xenova/all-MiniLM-L6-v2',
        cacheDir: testModelCacheDir(),
        baseDir: testDataDir,
        maxFileSize: 100 * 1024 * 1024,
      })
    )

    await ragServer.initialize()
  })

  afterAll(async () => {
    vi.restoreAllMocks()
    await ragServer.close()
    rmSync(testDbPath, { recursive: true, force: true })
    rmSync(testDataDir, { recursive: true, force: true })
  })

  it.each([false, true])(
    'removes partially inserted rows before restoring existing=%s',
    async (existing) => {
      const filePath = resolve(testDataDir, `partial-insert-${existing}.txt`)
      const store = privateMembers<{ vectorStore: VectorStore }>(ragServer).vectorStore
      if (existing) {
        writeFileSync(filePath, 'Prior material for partial insert. '.repeat(20))
        await ragServer.handleIngestFile({ filePath })
      }
      const originalRows = await store.getChunksByFilePath(filePath)
      const insert = store.insertChunks.bind(store)
      const spy = vi.spyOn(store, 'insertChunks').mockImplementationOnce(async (rows) => {
        await insert(rows)
        throw new Error('Post-insert setup failed')
      })
      try {
        writeFileSync(filePath, 'Replacement material for partial insert. '.repeat(20))
        await expect(ragServer.handleIngestFile({ filePath })).rejects.toThrow(
          'Post-insert setup failed'
        )
        expect(await store.getChunksByFilePath(filePath)).toEqual(originalRows)
      } finally {
        spy.mockRestore()
      }
    }
  )

  it('keeps only committed replacement rows when optimization fails', async () => {
    const filePath = resolve(testDataDir, 'post-commit.txt')
    writeFileSync(filePath, 'Original material for optimization test. '.repeat(20))
    await ragServer.handleIngestFile({ filePath })
    const store = privateMembers<{ vectorStore: VectorStore }>(ragServer).vectorStore
    const optimize = vi
      .spyOn(store, 'optimize')
      .mockRejectedValueOnce(new Error('Optimization failed'))
    try {
      writeFileSync(filePath, 'Replacement material for optimization test. '.repeat(20))
      await expect(ragServer.handleIngestFile({ filePath })).rejects.toThrow('Optimization failed')
      const rows = await store.getChunksByFilePath(filePath)
      expect(rows.length).toBeGreaterThan(0)
      expect(rows.every((row) => row.text.includes('Replacement material'))).toBe(true)
      expect(new Set(rows.map((row) => row.chunkIndex)).size).toBe(rows.length)
    } finally {
      optimize.mockRestore()
    }
  })

  it.each(['empty', 'insert', 'sidecar', 'optimize'] as const)(
    'preserves coherent source artifacts and rows after a replacement %s failure',
    async (failure) => {
      const source = `audit-replacement-${failure}`
      const metadata = { source, format: 'markdown' as const }
      const original = `# Original title\n\n${'Original content for source preservation. '.repeat(15)}`
      const replacement = `# Replacement title\n\n${'Replacement content for source preservation. '.repeat(15)}`
      await ragServer.handleIngestData({ content: original, metadata })
      const filePath = rawDataUtils.generateRawDataPath(testDbPath, source)
      const snapshotPath = embeddingSnapshotPath(testDbPath, filePath)
      expect(existsSync(snapshotPath)).toBe(true)
      const originalSnapshot = readFileSync(snapshotPath)
      const metaPath = rawDataUtils.generateMetaJsonPath(filePath)
      const originalMeta = readFileSync(metaPath, 'utf8')
      const store = privateMembers<{ vectorStore: VectorStore }>(ragServer).vectorStore
      const originalRows = await store.getChunksByFilePath(filePath)
      const saveMeta = rawDataUtils.saveMetaJson
      function installFailure(): MockInstance | undefined {
        switch (failure) {
          case 'insert':
            return vi.spyOn(store, 'insertChunks').mockRejectedValueOnce(new Error('Insert failed'))
          case 'optimize':
            return vi
              .spyOn(store, 'optimize')
              .mockRejectedValueOnce(new Error('Optimization failed'))
          case 'sidecar':
            return vi
              .spyOn(rawDataUtils, 'saveMetaJson')
              .mockImplementationOnce(async (...args) => {
                await saveMeta(...args)
                throw new Error('Sidecar write failed')
              })
          default:
            return undefined
        }
      }
      const spy = installFailure()
      try {
        await expect(
          ragServer.handleIngestData({
            content: failure === 'empty' ? '   ' : replacement,
            metadata,
          })
        ).rejects.toThrow()
        const rows = await store.getChunksByFilePath(filePath)
        if (failure === 'optimize') {
          expect(readFileSync(filePath, 'utf8')).toBe(replacement)
          expect(JSON.parse(readFileSync(metaPath, 'utf8')).title).toBe('Replacement title')
          expect(rows.length).toBeGreaterThan(0)
          expect(rows.every((row) => row.text.includes('Replacement'))).toBe(true)
          expect(new Set(rows.map((row) => row.chunkIndex)).size).toBe(rows.length)
          expect(readFileSync(snapshotPath)).not.toEqual(originalSnapshot)
        } else {
          expect(readFileSync(filePath, 'utf8')).toBe(original)
          expect(readFileSync(metaPath, 'utf8')).toBe(originalMeta)
          expect(rows).toEqual(originalRows)
          expect(readFileSync(snapshotPath)).toEqual(originalSnapshot)
        }
      } finally {
        spy?.mockRestore()
      }
    }
  )

  it('removes both newly created artifacts if sidecar saving fails', async () => {
    const source = 'audit-first-sidecar-failure'
    const filePath = rawDataUtils.generateRawDataPath(testDbPath, source)
    const saveMeta = rawDataUtils.saveMetaJson
    const spy = vi.spyOn(rawDataUtils, 'saveMetaJson').mockImplementationOnce(async (...args) => {
      await saveMeta(...args)
      throw new Error('Sidecar write failed')
    })
    try {
      await expect(
        ragServer.handleIngestData({
          content: 'New source content. '.repeat(20),
          metadata: { source, format: 'text' },
        })
      ).rejects.toThrow('Sidecar write failed')
      expect(existsSync(filePath)).toBe(false)
      expect(existsSync(rawDataUtils.generateMetaJsonPath(filePath))).toBe(false)
      const store = privateMembers<{ vectorStore: VectorStore }>(ragServer).vectorStore
      expect(await store.getChunksByFilePath(filePath)).toEqual([])
    } finally {
      spy.mockRestore()
    }
  })

  it('surfaces a distinct DatabaseError (cause = insert error) when rollback also fails', async () => {
    // Arrange: Ingest a file normally first
    const testFile = resolve(testDataDir, 'rollback-double-fail.txt')
    writeFileSync(testFile, 'Content for double failure test. '.repeat(50))

    await ragServer.handleIngestFile({ filePath: testFile })

    const vectorStore = privateVectorStore(ragServer)

    // Both insert calls fail (new data insert + rollback restore): the prior
    // data is now gone. The handler surfaces a distinct DatabaseError that says
    // so, preserving the original insert error as `.cause`.
    const insertError = new Error('Insert failed')
    const insertSpy = vi
      .spyOn(vectorStore, 'insertChunks')
      .mockRejectedValueOnce(insertError)
      .mockRejectedValueOnce(new Error('Rollback also failed'))

    vi.spyOn(vectorStore, 'optimize').mockResolvedValue(undefined)
    const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {})

    writeFileSync(testFile, 'Updated content for double failure. '.repeat(30))

    let thrown: unknown
    try {
      await ragServer.handleIngestFile({ filePath: testFile })
    } catch (e) {
      thrown = e
    }
    expect(thrown).toBeInstanceOf(DatabaseError)
    expect(expectError(thrown).message).toContain('rollback failed')
    expect(expectError(thrown).message).toContain('may not have been restored')
    expect(expectRecord(thrown)['cause']).toBe(insertError)

    // The rollback failure is still recorded on stderr for diagnostics.
    const logged = errorSpy.mock.calls.map((c) => c.join(' ')).join('\n')
    expect(logged).toContain('Rollback failed')

    insertSpy.mockRestore()
    errorSpy.mockRestore()
  })

  it('preserves the recorded visual profile when preparation yields zero chunks', async () => {
    // Arrange: rows recording a `fast` visual ingest, then a replacement source
    // too short to produce a single chunk.
    const testFile = resolve(testDataDir, 'zero-chunk-visual-profile.txt')
    writeFileSync(testFile, 'seed')
    const vectorStore = privateVectorStore(ragServer)
    const original = await seedProfiledRows(vectorStore, testFile, 'fast')

    writeFileSync(testFile, '   ')

    // Act
    await expect(ragServer.handleIngestFile({ filePath: testFile })).rejects.toThrow(
      'No chunks generated from file'
    )

    // Assert: the pre-delete rejection leaves every prior row, and its recorded
    // intent, exactly as it was — no new intent is written either.
    const preserved = await vectorStore.getChunksByFilePath(testFile)
    expect(preserved).toEqual(original)
    expect(preserved.map((row) => row.visualProfile)).toEqual(['fast', 'fast'])
  })
})
