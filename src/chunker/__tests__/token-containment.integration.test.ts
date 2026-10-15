// Measured containment against the real tokenizer (AC-008, AC-009).
//
// An injected counter cannot show that the chunker's measurement is wired to
// the model that will embed the text, so this case loads the cached default
// model and asserts the overflow before containment as well as the containment
// after. The structural properties are covered by counters elsewhere.

import { afterEach, describe, expect, it, vi } from 'vitest'
import { getTestDevice, testModelCacheDir } from '../../__tests__/test-device.js'
import { expectDefined } from '../../__tests__/test-doubles.js'
import { Embedder } from '../../embedder/index.js'
import { SemanticChunker } from '../semantic-chunker.js'
import { splitIntoSentenceUnits } from '../sentence-splitter.js'

/**
 * A CJK paragraph with no sentence terminator, so the splitter keeps it as one
 * oversized unit. Ideographs only: the default tokenizer splits them per
 * character, while hiragana collapses to a single `[UNK]` and would leave
 * nothing to contain. The repeated phrase stays under `isGarbageChunk`'s 80%
 * repetition bound.
 */
const DENSE_PHRASE =
  '检索增强生成系统在处理密集文字时会遇到位置窗口限制的问题因此需要以真实词元度量为依据的容纳策略而不是基于字符数量的粗略估计'
const CJK_SENTENCE =
  '本项目的分块流程先按语义边界切分文本再度量真实词元数量然后仅对超出模型位置窗口的部分进行切分以保证内容完整可检索。'
const CJK_DOCUMENT = `${DENSE_PHRASE.repeat(10)}${CJK_SENTENCE.repeat(3)}`

const TRUNCATION_WARNING = /input exceeds the model token limit/

function defaultModelEmbedder(): Embedder {
  return new Embedder({
    modelPath: 'Xenova/all-MiniLM-L6-v2',
    batchSize: 8,
    cacheDir: testModelCacheDir(),
    device: getTestDevice(),
  })
}

/** Whitespace is dropped because units are joined with a single space while the
 * source holds none between CJK sentences (`semantic-chunker.ts` joinUnits). */
function withoutWhitespace(text: string): string {
  return text.replace(/\s+/g, '')
}

describe('Measured containment on the real tokenizer', () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('contains a CJK document that overflows the cap before chunking', async () => {
    const stderr = vi.spyOn(console, 'error').mockImplementation(() => {})
    const embedder = defaultModelEmbedder()
    const cap = expectDefined(await embedder.getTokenLimit())

    // Before containment: the units `chunkText` will measure, measured the same
    // way. Without this the case would pass with no containment code at all.
    const units = splitIntoSentenceUnits(CJK_DOCUMENT, [])
    const unitLengths = await embedder.countTokens(units.map((unit) => unit.text))
    expect(Math.max(...unitLengths)).toBeGreaterThan(cap)

    const chunks = await new SemanticChunker().chunkText(CJK_DOCUMENT, embedder)

    // After containment: every stored chunk fits the window the model has.
    expect(chunks.length).toBeGreaterThan(1)
    const chunkLengths = await embedder.countTokens(chunks.map((chunk) => chunk.text))
    for (const length of chunkLengths) {
      expect(length).toBeLessThanOrEqual(cap)
    }
    // Nothing the garbage filter would reject is present, so the whole source
    // is covered by the stored chunks, in order.
    expect(withoutWhitespace(chunks.map((chunk) => chunk.text).join(''))).toBe(
      withoutWhitespace(CJK_DOCUMENT)
    )
    expect(chunks.map((chunk) => chunk.index)).toEqual(chunks.map((_, index) => index))
    expect(chunks[0]?.sourceStart).toBe(0)
    expect(chunks[chunks.length - 1]?.sourceEnd).toBe(CJK_DOCUMENT.length)

    // AC-009: containment happened instead of truncation, so the embedder never
    // saw an input above the cap and reported nothing.
    const warnings = stderr.mock.calls
      .map((call) => String(call[0]))
      .filter((message) => TRUNCATION_WARNING.test(message))
    expect(warnings).toEqual([])
  }, 180000)
})
