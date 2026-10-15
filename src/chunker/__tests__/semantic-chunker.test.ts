// Semantic Chunker Unit Test
// Created: 2025-12-27
// Purpose: Verify Max-Min semantic chunking algorithm

import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { TextChunk } from '../index.js'
import {
  DEFAULT_MIN_CHUNK_LENGTH,
  isGarbageChunk,
  SemanticChunker,
  type SemanticChunkerConfig,
} from '../semantic-chunker.js'
import {
  denseScriptChunks,
  denseScriptDocument,
  fixtureEmbeddings,
  latinChunks,
  latinDocument,
} from './main-boundary-fixture.js'

// Mock embedder interface
interface MockEmbedder {
  embedBatch(texts: string[]): Promise<number[][]>
}

describe('SemanticChunker', () => {
  let chunker: SemanticChunker
  let mockEmbedder: MockEmbedder

  // Helper to create mock embeddings with controlled similarity
  // Vectors are normalized (magnitude = 1) for cosine similarity
  function createMockEmbedding(values: number[]): number[] {
    const magnitude = Math.sqrt(values.reduce((sum, v) => sum + v * v, 0))
    return values.map((v) => v / magnitude)
  }

  beforeEach(() => {
    // Default config based on paper recommendations
    const config: SemanticChunkerConfig = {
      hardThreshold: 0.6,
      initConst: 1.5,
      c: 0.9,
      minChunkLength: 50,
    }
    chunker = new SemanticChunker(config)

    // Mock embedder that returns predictable embeddings
    mockEmbedder = {
      embedBatch: vi.fn(),
    }
  })

  // --------------------------------------------
  // Basic functionality
  // --------------------------------------------
  describe('Basic chunking', () => {
    it('should return empty array for empty text', async () => {
      const result = await chunker.chunkText('', mockEmbedder)
      expect(result).toEqual([])
    })

    it('should return empty array for whitespace only', async () => {
      const result = await chunker.chunkText('   \n\n   ', mockEmbedder)
      expect(result).toEqual([])
    })

    it('should handle single sentence', async () => {
      const text = 'This is a single sentence that is long enough to be a valid chunk on its own.'

      // Mock embedding for the single sentence
      vi.mocked(mockEmbedder.embedBatch).mockResolvedValue([createMockEmbedding([1, 0, 0])])

      const result = await chunker.chunkText(text, mockEmbedder)

      expect(result).toHaveLength(1)
      expect(result[0]?.text).toContain('This is a single sentence')
      expect(result[0]?.index).toBe(0)
    })
  })

  // --------------------------------------------
  // Max-Min algorithm behavior
  // --------------------------------------------
  describe('Max-Min algorithm', () => {
    it('should split on semantic boundaries', async () => {
      const text = `Topic A sentence one. Topic A sentence two. Topic A sentence three.
Topic B is completely different. Topic B continues here.`

      // Mock embeddings: Topic A sentences similar, Topic B sentences similar, but A and B different
      // A1-A2 ≈ 0.98, A2-A3 ≈ 0.97, A3-B1 ≈ 0 (semantic shift), B1-B2 ≈ 0.98
      vi.mocked(mockEmbedder.embedBatch).mockResolvedValue([
        createMockEmbedding([1, 0, 0]),
        createMockEmbedding([0.98, 0.1, 0]),
        createMockEmbedding([0.95, 0.15, 0]),
        createMockEmbedding([0, 0, 1]), // Big semantic shift
        createMockEmbedding([0.1, 0, 0.98]),
      ])

      const result = await chunker.chunkText(text, mockEmbedder)

      // Should detect the semantic boundary between Topic A and Topic B
      // Result: 2 chunks - Topic A (3 sentences) and Topic B (2 sentences)
      expect(result).toHaveLength(2)

      // Verify chunk contents
      expect(result[0]?.text).toContain('Topic A')
      expect(result[0]?.text).not.toContain('Topic B')
      expect(result[1]?.text).toContain('Topic B')
      expect(result[1]?.text).not.toContain('Topic A')
    })
  })

  // --------------------------------------------
  // Configuration options
  // --------------------------------------------
  describe('Configuration', () => {
    it('should respect hardThreshold setting', async () => {
      // Create chunker with very high threshold (forces more splits)
      const strictChunker = new SemanticChunker({
        hardThreshold: 0.95,
        initConst: 1.5,
        c: 0.9,
        minChunkLength: 10,
      })

      const text = 'First sentence here. Second sentence here. Third sentence here.'

      // Similarities: 1-2 ≈ 0.8, 2-3 ≈ 0.7 (both below 0.95 threshold)
      vi.mocked(mockEmbedder.embedBatch).mockResolvedValue([
        createMockEmbedding([1, 0, 0]),
        createMockEmbedding([0.8, 0.2, 0]), // Below 0.95 threshold
        createMockEmbedding([0.6, 0.4, 0]), // Below 0.95 threshold
      ])

      const result = await strictChunker.chunkText(text, mockEmbedder)

      // hardThreshold 0.95 splits sentence 3 (sim ≈0.94 < 0.95) into its own
      // chunk; a lower threshold would merge all three. → deterministically 2.
      expect(result).toHaveLength(2)
      expect(result[0]?.text).toContain('First sentence')
      expect(result[0]?.text).toContain('Second sentence')
      expect(result[1]?.text).toContain('Third sentence')
      expect(result[1]?.text).not.toContain('Second sentence')
    })

    it('should filter chunks shorter than minChunkLength', async () => {
      const chunkerWithHighMin = new SemanticChunker({
        hardThreshold: 0.6,
        initConst: 1.5,
        c: 0.9,
        minChunkLength: 100,
      })

      const text = 'Short. Also short.'

      vi.mocked(mockEmbedder.embedBatch).mockResolvedValue([
        createMockEmbedding([1, 0, 0]),
        createMockEmbedding([0, 1, 0]),
      ])

      const result = await chunkerWithHighMin.chunkText(text, mockEmbedder)

      // Orthogonal sentences split into two chunks, both < minChunkLength (100),
      // so every chunk is filtered out → empty result.
      expect(result).toHaveLength(0)
    })

    it('should retain a short atomic row without changing ordinary prose filtering', async () => {
      const shortRow = 'Code: 42'
      vi.mocked(mockEmbedder.embedBatch).mockResolvedValue([createMockEmbedding([1, 0, 0])])

      await expect(chunker.chunkText(shortRow, mockEmbedder)).resolves.toEqual([])
      await expect(
        chunker.chunkText(shortRow, mockEmbedder, [{ start: 0, end: shortRow.length }])
      ).resolves.toEqual([{ text: shortRow, index: 0, sourceStart: 0, sourceEnd: shortRow.length }])
    })

    it('should embed and persist a multi-sentence atomic row as one unit', async () => {
      const row = 'Field: 42\nDescription: First sentence. Second sentence.'
      vi.mocked(mockEmbedder.embedBatch).mockResolvedValue([createMockEmbedding([1, 0, 0])])

      const result = await chunker.chunkText(row, mockEmbedder, [{ start: 0, end: row.length }])

      expect(mockEmbedder.embedBatch).toHaveBeenCalledWith([row])
      expect(result).toEqual([{ text: row, index: 0, sourceStart: 0, sourceEnd: row.length }])
    })

    it('should keep an atomic row intact when grouped with neighboring prose', async () => {
      const before = 'Context before the table row is intentionally long enough.'
      const row = 'Code: 42\nDescription: First sentence. Second sentence.'
      const after = 'Context after the table row is also intentionally long enough.'
      const text = `${before}\n\n${row}\n\n${after}`
      const rowStart = before.length + 2
      const embedding = createMockEmbedding([1, 0, 0])
      vi.mocked(mockEmbedder.embedBatch).mockResolvedValue([embedding, embedding, embedding])

      const result = await chunker.chunkText(text, mockEmbedder, [
        { start: rowStart, end: rowStart + row.length },
      ])

      expect(mockEmbedder.embedBatch).toHaveBeenCalledWith([before, row, after])
      expect(result).toEqual([
        {
          text: `${before} ${row} ${after}`,
          index: 0,
          sourceStart: 0,
          sourceEnd: text.length,
        },
      ])
    })

    it('should continue applying the garbage filter to atomic ranges', async () => {
      const decoration = '--------'
      vi.mocked(mockEmbedder.embedBatch).mockResolvedValue([createMockEmbedding([1, 0, 0])])

      const result = await chunker.chunkText(decoration, mockEmbedder, [
        { start: 0, end: decoration.length },
      ])

      expect(result).toEqual([])
    })

    it('exposes exact source envelopes without changing atomic grouping or chunk indices', async () => {
      const source =
        '  Repeated sentence.  \n\n[Visual content on page 1, visual 0: 📊 repeated.]\n\nRepeated sentence.'
      const caption = '[Visual content on page 1, visual 0: 📊 repeated.]'
      const captionStart = source.indexOf(caption)
      const envelopeChunker = new SemanticChunker({
        hardThreshold: 0.99,
        initConst: 0.1,
        c: 0.9,
        minChunkLength: 1,
      })
      vi.mocked(mockEmbedder.embedBatch).mockResolvedValue([
        createMockEmbedding([1, 0, 0]),
        createMockEmbedding([0, 1, 0]),
        createMockEmbedding([0, 0, 1]),
      ])

      const result = await envelopeChunker.chunkText(source, mockEmbedder, [
        { start: captionStart, end: captionStart + caption.length },
      ])

      expect(result).toEqual([
        {
          text: 'Repeated sentence.',
          index: 0,
          sourceStart: 2,
          sourceEnd: 20,
        },
        {
          text: caption,
          index: 1,
          sourceStart: captionStart,
          sourceEnd: captionStart + caption.length,
        },
        {
          text: 'Repeated sentence.',
          index: 2,
          sourceStart: source.lastIndexOf('Repeated sentence.'),
          sourceEnd: source.length,
        },
      ])
    })
  })

  // --------------------------------------------
  // Edge cases
  // --------------------------------------------
  describe('Edge cases', () => {
    it('should handle embedder errors gracefully', async () => {
      const text = 'This is a test sentence.'

      vi.mocked(mockEmbedder.embedBatch).mockRejectedValue(new Error('Embedder failed'))

      await expect(chunker.chunkText(text, mockEmbedder)).rejects.toThrow('Embedder failed')
    })
  })

  // --------------------------------------------
  // Cosine similarity calculation
  // --------------------------------------------
  describe('Cosine similarity', () => {
    it('should correctly calculate similarity between identical vectors', () => {
      const vec = createMockEmbedding([1, 2, 3])
      const similarity = chunker.cosineSimilarity(vec, vec)
      expect(similarity).toBeCloseTo(1.0, 5)
    })

    it('should correctly calculate similarity between orthogonal vectors', () => {
      const vec1 = createMockEmbedding([1, 0, 0])
      const vec2 = createMockEmbedding([0, 1, 0])
      const similarity = chunker.cosineSimilarity(vec1, vec2)
      expect(similarity).toBeCloseTo(0.0, 5)
    })

    it('should correctly calculate similarity between opposite vectors', () => {
      const vec1 = [1, 0, 0]
      const vec2 = [-1, 0, 0]
      const similarity = chunker.cosineSimilarity(vec1, vec2)
      expect(similarity).toBeCloseTo(-1.0, 5)
    })
  })

  // --------------------------------------------
  // Boundary value tests (MAX_SENTENCES=15)
  // --------------------------------------------
  describe('Boundary values', () => {
    it('should handle exactly MAX_SENTENCES (15) sentences without split', async () => {
      // Create 15 sentences with high similarity (should stay in one chunk)
      const sentences = Array.from({ length: 15 }, (_, i) => `Similar sentence number ${i + 1}.`)
      const text = sentences.join(' ')

      // All embeddings are similar (high cosine similarity)
      const embeddings = Array.from({ length: 15 }, () => createMockEmbedding([1, 0, 0]))
      vi.mocked(mockEmbedder.embedBatch).mockResolvedValue(embeddings)

      const result = await chunker.chunkText(text, mockEmbedder)

      // 15 sentences with high similarity → single chunk (at the MAX_SENTENCES limit)
      expect(result).toHaveLength(1)
      expect(result[0]?.text).toContain('sentence number 1')
      expect(result[0]?.text).toContain('sentence number 15')
    })

    it('should force split at MAX_SENTENCES+1 (16) sentences', async () => {
      // Create 17 sentences with high similarity (should force split at 15, then 16 and 17 form second chunk)
      // Using 17 sentences ensures second chunk exceeds minChunkLength (50 chars)
      const sentences = Array.from({ length: 17 }, (_, i) => `Similar sentence number ${i + 1}.`)
      const text = sentences.join(' ')

      // All embeddings are identical (maximum similarity)
      const embeddings = Array.from({ length: 17 }, () => createMockEmbedding([1, 0, 0]))
      vi.mocked(mockEmbedder.embedBatch).mockResolvedValue(embeddings)

      const result = await chunker.chunkText(text, mockEmbedder)

      // 17 sentences → forced split after 15 → 2 chunks (sentences 1-15, sentences 16-17)
      expect(result).toHaveLength(2)
      expect(result[0]?.text).toContain('sentence number 1')
      expect(result[0]?.text).toContain('sentence number 15')
      expect(result[0]?.text).not.toContain('sentence number 16')
      expect(result[1]?.text).toContain('sentence number 16')
      expect(result[1]?.text).toContain('sentence number 17')
    })
  })
})

// --------------------------------------------
// isGarbageChunk tests
// --------------------------------------------
describe('isGarbageChunk', () => {
  describe('should identify garbage', () => {
    it('should return true for empty string', () => {
      expect(isGarbageChunk('')).toBe(true)
    })

    it('should return true for whitespace only', () => {
      expect(isGarbageChunk('   ')).toBe(true)
      expect(isGarbageChunk('\n\t')).toBe(true)
    })

    it('should return true for decoration lines (dashes)', () => {
      expect(isGarbageChunk('--------')).toBe(true)
      expect(isGarbageChunk('-----------')).toBe(true)
    })

    it('should return true for decoration lines (equals)', () => {
      expect(isGarbageChunk('========')).toBe(true)
      expect(isGarbageChunk('===========')).toBe(true)
    })

    it('should return true for decoration lines (asterisks)', () => {
      expect(isGarbageChunk('********')).toBe(true)
      expect(isGarbageChunk('***')).toBe(true)
    })

    it('should return true for mixed decoration characters', () => {
      expect(isGarbageChunk('---===---')).toBe(true)
      expect(isGarbageChunk('***---***')).toBe(true)
    })

    it('should return true for excessive repetition (>80%)', () => {
      expect(isGarbageChunk('ああああああああああ')).toBe(true) // 100% same char
    })
  })

  describe('should identify valid content', () => {
    it('should return false for text with alphanumeric', () => {
      expect(isGarbageChunk('function foo() {}')).toBe(false)
      expect(isGarbageChunk('Hello World')).toBe(false)
    })

    it('should return false for code with decorations', () => {
      // These contain alphanumeric characters along with decorations
      expect(isGarbageChunk('/* Section 1 ============ */')).toBe(false)
      expect(isGarbageChunk('// ---------- Header ----------')).toBe(false)
      expect(isGarbageChunk('/* TODO: fix this */')).toBe(false)
    })

    it('should return false for Japanese text', () => {
      expect(isGarbageChunk('こんにちは')).toBe(false)
      expect(isGarbageChunk('日本語のテキスト')).toBe(false)
    })

    it('should return false for numbers', () => {
      expect(isGarbageChunk('12345')).toBe(false)
      expect(isGarbageChunk('2024年')).toBe(false)
    })

    it('should return false for mixed content', () => {
      expect(isGarbageChunk('Section 1: Introduction')).toBe(false)
      expect(isGarbageChunk('Chapter 5 - Summary')).toBe(false)
    })
  })
})

// --------------------------------------------
// Measured token containment (stages A and C)
// --------------------------------------------
describe('Measured token containment', () => {
  interface MeasuredEmbedder {
    embedBatch: (texts: string[]) => Promise<number[][]>
    getTokenLimit: () => Promise<number | null>
    countTokens: (texts: string[]) => Promise<number[]>
  }

  interface EmbedderLog {
    /** Method names in call order, so stage A can be proven to run before embedding. */
    calls: string[]
    /** Every batch handed to `embedBatch`, in call order. */
    batches: string[][]
  }

  /** One token per UTF-16 code unit: the cap is then readable as a character count. */
  const codeUnitTokens = (text: string): number => text.length

  /**
   * Identical unit vectors put every unit in one semantic group, which is the
   * only state where a group can exceed the cap.
   */
  function measuredEmbedder(
    cap: number | null,
    tokensOf: (text: string) => number = codeUnitTokens
  ): { embedder: MeasuredEmbedder; log: EmbedderLog } {
    const log: EmbedderLog = { calls: [], batches: [] }
    const embedder: MeasuredEmbedder = {
      embedBatch: (texts) => {
        log.calls.push('embedBatch')
        log.batches.push([...texts])
        return Promise.resolve(texts.map(() => [1, 0]))
      },
      getTokenLimit: () => {
        log.calls.push('getTokenLimit')
        return Promise.resolve(cap)
      },
      countTokens: (texts) => {
        log.calls.push('countTokens')
        return Promise.resolve(texts.map(tokensOf))
      },
    }
    return { embedder, log }
  }

  function containmentChunker(minChunkLength = DEFAULT_MIN_CHUNK_LENGTH): SemanticChunker {
    return new SemanticChunker({ hardThreshold: 0.6, initConst: 1.5, c: 0.9, minChunkLength })
  }

  /** Independent check of AC-017: spans are ordered and inside the source. */
  function expectOrderedSpans(chunks: TextChunk[], text: string): void {
    let previousEnd = 0
    for (const chunk of chunks) {
      expect(chunk.sourceStart).toBeGreaterThanOrEqual(previousEnd)
      expect(chunk.sourceEnd).toBeGreaterThan(chunk.sourceStart)
      expect(chunk.sourceEnd).toBeLessThanOrEqual(text.length)
      previousEnd = chunk.sourceEnd
    }
  }

  /**
   * What #203's `chunk.text.length <= 400` assertion approximated: a stored
   * chunk fits the model's window. Measured, so the check holds for any script
   * and any tokenizer instead of for the character ratio of one script family.
   */
  function expectMeasuredWithinCap(
    chunks: TextChunk[],
    cap: number,
    tokensOf: (text: string) => number = codeUnitTokens
  ): void {
    for (const chunk of chunks) {
      expect(tokensOf(chunk.text)).toBeLessThanOrEqual(cap)
    }
  }

  it('measures and splits sentence units before the first embedBatch call', async () => {
    const text = 'abcdefghij'.repeat(12)
    const { embedder, log } = measuredEmbedder(30)

    await containmentChunker().chunkText(text, embedder)

    expect(log.calls.indexOf('countTokens')).toBeLessThan(log.calls.indexOf('embedBatch'))
    expect(log.batches).toHaveLength(1)
    expect(log.batches[0]).toEqual([
      text.slice(0, 30),
      text.slice(30, 60),
      text.slice(60, 90),
      text.slice(90, 120),
    ])
  })

  it('splits an oversized group at unit boundaries only', async () => {
    const sentences = [
      'Alpha sentence number one.',
      'Beta sentence number two.',
      'Gamma sentence number three.',
      'Delta sentence number four.',
      'Epsilon sentence number five.',
      'Zeta sentence number six.',
    ]
    const text = sentences.join(' ')
    const { embedder } = measuredEmbedder(60)

    const chunks = await containmentChunker().chunkText(text, embedder)

    expect(chunks.length).toBeGreaterThan(1)
    for (const chunk of chunks) {
      // Every cut fell between sentences: each part of a chunk is a whole unit.
      for (const part of chunk.text.split(/(?<=\.)\s/)) {
        expect(sentences).toContain(part)
      }
      expect(text.slice(chunk.sourceStart, chunk.sourceEnd)).toBe(chunk.text)
    }
    expect(chunks.map((chunk) => chunk.text).join(' ')).toBe(text)
    expect(chunks.map((chunk) => chunk.index)).toEqual(chunks.map((_, index) => index))
    expectMeasuredWithinCap(chunks, 60)
    expectOrderedSpans(chunks, text)
  })

  it('returns every piece of a split unit, including a tail under minChunkLength', async () => {
    const text = `${'abcdefghij'.repeat(6)}abcde`
    const { embedder } = measuredEmbedder(60)

    const chunks = await containmentChunker().chunkText(text, embedder)

    expect(chunks.map((chunk) => chunk.text).join('')).toBe(text)
    expect(chunks[chunks.length - 1]?.text).toBe('abcde')
    for (const chunk of chunks) {
      expect(text.slice(chunk.sourceStart, chunk.sourceEnd)).toBe(chunk.text)
    }
    expectMeasuredWithinCap(chunks, 60)
    expectOrderedSpans(chunks, text)
  })

  it('leaves text that measures inside the cap unsplit, whatever its script mix', async () => {
    // #203 cut at 400 characters once 30% of them were CJK, so this fixture —
    // 457 characters, 32% ideographs — was split there. Measured against the
    // cap it fits, so it keeps its single-chunk boundary.
    const text = `Release notes for the 位置窗口 guard: ${'密集文字的容纳以真实词元度量为准 containment keeps prose whole '.repeat(9)}`
    const { embedder } = measuredEmbedder(512)

    const chunks = await containmentChunker().chunkText(text, embedder)

    expect(chunks).toHaveLength(1)
    expect(chunks[0]?.text.length).toBeGreaterThan(400)
    expect(text.slice(chunks[0]?.sourceStart, chunks[0]?.sourceEnd)).toBe(chunks[0]?.text)
    expectMeasuredWithinCap(chunks, 512)
  })

  it('stores a grapheme that measures over the cap and keeps chunking the rest', async () => {
    // U+20000 is one grapheme of two UTF-16 code units; the counter makes it
    // indivisibly oversized, which is the exception stage A cannot reduce.
    const astral = String.fromCodePoint(0x20000)
    const head = 'Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu.'
    const tail = 'Tail sentence also long enough to be its own ordinary chunk in this fixture.'
    const text = `${head} ${astral}. ${tail}`
    const { embedder } = measuredEmbedder(30, (piece) =>
      [...piece].reduce((sum, character) => sum + (character === astral ? 100 : 1), 0)
    )

    const chunks = await containmentChunker().chunkText(text, embedder)

    expect(chunks.some((chunk) => chunk.text.includes(astral))).toBe(true)
    const joined = chunks.map((chunk) => chunk.text).join(' ')
    expect(joined).toContain('Alpha beta gamma')
    expect(joined).toContain('in this fixture.')
    expectOrderedSpans(chunks, text)
  })

  it('keeps rejecting an oversized garbage group', async () => {
    const { embedder } = measuredEmbedder(30)

    const chunks = await containmentChunker().chunkText('-'.repeat(600), embedder)

    expect(chunks).toEqual([])
  })

  it('keeps every piece of a non-garbage unit whose fragments look like noise', async () => {
    // A long run of one character is garbage on its own, but this unit is not:
    // the Latin tail takes it under the repetition threshold. The run divides
    // into more pieces than MAX_SENTENCES, so they reach several groups, and
    // judging those groups individually used to discard the whole run.
    const text = `${'\u6f22'.repeat(800)}${'abcdefghijklmnopqrstuvwxyz'.repeat(8)}tail`
    expect(isGarbageChunk(text)).toBe(false)
    const { embedder } = measuredEmbedder(50)

    const chunks = await containmentChunker().chunkText(text, embedder)

    const returned = chunks.reduce((total, chunk) => total + chunk.text.length, 0)
    expect(returned).toBe(text.length)
    expect(chunks[0]?.sourceStart).toBe(0)
    expect(chunks.at(-1)?.sourceEnd).toBe(text.length)
    expectOrderedSpans(chunks, text)
  })

  it('splits an oversized atomic range into pieces with exact offsets', async () => {
    const text = 'abcdefghij'.repeat(9)
    const { embedder } = measuredEmbedder(30)

    const chunks = await containmentChunker().chunkText(text, embedder, [
      { start: 0, end: text.length },
    ])

    expect(chunks.map((chunk) => chunk.text)).toEqual([
      text.slice(0, 30),
      text.slice(30, 60),
      text.slice(60, 90),
    ])
    for (const chunk of chunks) {
      expect(text.slice(chunk.sourceStart, chunk.sourceEnd)).toBe(chunk.text)
    }
    expectOrderedSpans(chunks, text)
  })

  it('budgets for the text prefix the caller will embed each chunk behind', async () => {
    const text = 'abcdefghij'.repeat(6)
    const prefix = 'Title: T\n\n'
    const { embedder } = measuredEmbedder(30)

    const chunks = await containmentChunker().chunkText(text, embedder, [], prefix)

    expect(chunks.map((chunk) => chunk.text).join('')).toBe(text)
    expect(chunks.every((chunk) => !chunk.text.startsWith(prefix))).toBe(true)
    expectMeasuredWithinCap(
      chunks.map((chunk) => ({ ...chunk, text: prefix + chunk.text })),
      30
    )
    expectOrderedSpans(chunks, text)
  })
})

// --------------------------------------------
// Boundary preservation against `main` (AC-011, AC-016)
// --------------------------------------------
describe('Boundary preservation against main', () => {
  /** Every fixture unit and group measures well inside this cap. */
  const CAP = 512

  /** One token per UTF-16 code unit, so a measurement is readable as a length. */
  const embedBatch = (texts: string[]): Promise<number[][]> =>
    Promise.resolve(fixtureEmbeddings(texts))
  const countTokens = (texts: string[]): Promise<number[]> =>
    Promise.resolve(texts.map((text) => text.length))

  const documents = [
    { document: 'Latin', text: latinDocument, mainChunks: latinChunks },
    { document: 'dense-script', text: denseScriptDocument, mainChunks: denseScriptChunks },
  ]

  const configurations = [
    {
      configuration: 'a resolved cap that every unit and group fits',
      embedder: { embedBatch, getTokenLimit: () => Promise.resolve(CAP), countTokens },
    },
    { configuration: 'an embedder exposing neither optional member', embedder: { embedBatch } },
    {
      configuration: 'an embedder whose getTokenLimit resolves to null',
      embedder: { embedBatch, getTokenLimit: () => Promise.resolve(null), countTokens },
    },
  ]

  // A degraded embedder skips containment before either script matters.
  const [resolvedCap, ...degraded] = configurations
  const cases = [
    ...documents.map((document) => ({ ...resolvedCap, ...document })),
    ...degraded.map((configuration) => ({ ...configuration, ...documents[0] })),
  ]

  it.each(cases)(
    'reproduces main’s chunks for the $document document with $configuration',
    async ({ text, mainChunks, embedder }) => {
      const chunker = new SemanticChunker({
        hardThreshold: 0.6,
        initConst: 1.5,
        c: 0.9,
        minChunkLength: DEFAULT_MIN_CHUNK_LENGTH,
      })

      const chunks = await chunker.chunkText(text, embedder)

      expect(chunks).toEqual(mainChunks)
    }
  )
})
