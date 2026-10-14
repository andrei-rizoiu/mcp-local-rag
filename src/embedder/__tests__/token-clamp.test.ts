// Cap resolution and clamp installation (AC-002, AC-007).
//
// Fake pipelines go to the module-level clamp function directly; no module
// factory for `@huggingface/transformers` (see lazy-initialization.test.ts).

import { describe, expect, it } from 'vitest'
import { installTokenLimitClamp } from '../index.js'

interface TokenizerOptions {
  padding?: boolean
  truncation?: boolean
  return_tensor?: boolean
  max_length?: number
}

interface FakePipelineShape {
  modelMaxLength?: number
  maxPositionEmbeddings?: number
  textConfigMaxPositionEmbeddings?: number
}

function createFakePipeline(shape: FakePipelineShape) {
  const tokenizerCalls: (TokenizerOptions | undefined)[] = []
  const tokenizer = (texts: string[], options?: TokenizerOptions) => {
    tokenizerCalls.push(options)
    return { input_ids: texts.map(() => [1]) }
  }
  if (shape.modelMaxLength !== undefined) {
    Object.defineProperty(tokenizer, 'model_max_length', {
      get: () => shape.modelMaxLength,
    })
  }
  const config = {
    ...(shape.maxPositionEmbeddings === undefined
      ? {}
      : { max_position_embeddings: shape.maxPositionEmbeddings }),
    ...(shape.textConfigMaxPositionEmbeddings === undefined
      ? {}
      : { text_config: { max_position_embeddings: shape.textConfigMaxPositionEmbeddings } }),
  }
  const pipeline = Object.assign(async () => ({ data: new Float32Array([1]), dims: [1, 1] }), {
    tokenizer,
    ...(Object.keys(config).length === 0 ? {} : { model: { config } }),
  })
  return { pipeline, tokenizer, tokenizerCalls }
}

describe('installTokenLimitClamp', () => {
  it('takes the smaller of a usable tokenizer limit and the position window', () => {
    const { pipeline } = createFakePipeline({ modelMaxLength: 8192, maxPositionEmbeddings: 512 })

    expect(installTokenLimitClamp(pipeline).tokenLimit).toBe(512)
  })

  it('keeps the tokenizer limit as-is for the repo default shape', () => {
    const { pipeline } = createFakePipeline({ modelMaxLength: 512, maxPositionEmbeddings: 512 })

    expect(installTokenLimitClamp(pipeline).tokenLimit).toBe(512)
  })

  it('reserves two positions when only the position window is usable', () => {
    const { pipeline } = createFakePipeline({ maxPositionEmbeddings: 512 })

    expect(installTokenLimitClamp(pipeline).tokenLimit).toBe(510)
  })

  it('rejects the 1e30 sentinel and falls back to the reserved position window', () => {
    const { pipeline } = createFakePipeline({ modelMaxLength: 1e30, maxPositionEmbeddings: 512 })

    expect(installTokenLimitClamp(pipeline).tokenLimit).toBe(510)
  })

  it('reads the position window from text_config when the top level has none', () => {
    const { pipeline } = createFakePipeline({
      modelMaxLength: 1e30,
      textConfigMaxPositionEmbeddings: 2048,
    })

    expect(installTokenLimitClamp(pipeline).tokenLimit).toBe(2046)
  })

  it('uses the tokenizer limit when no position window is reported', () => {
    const { pipeline } = createFakePipeline({ modelMaxLength: 512 })

    expect(installTokenLimitClamp(pipeline).tokenLimit).toBe(512)
  })

  it('resolves no cap and leaves the pipeline untouched when neither limit is usable', () => {
    const { pipeline, tokenizer } = createFakePipeline({ modelMaxLength: 1e30 })

    const installation = installTokenLimitClamp(pipeline)

    expect(installation.tokenLimit).toBeNull()
    expect(pipeline.tokenizer).toBe(tokenizer)
  })

  it('resolves no cap for an unrecognized pipeline shape', () => {
    const notAPipeline = { tokenizer: 'not-callable' }

    const installation = installTokenLimitClamp(notAPipeline)

    expect(installation.tokenLimit).toBeNull()
    expect(installation.measurementTokenizer).toBeNull()
    expect(notAPipeline.tokenizer).toBe('not-callable')
  })

  it('injects the resolved cap and truncation into every tokenizer call', () => {
    const { pipeline, tokenizer, tokenizerCalls } = createFakePipeline({
      maxPositionEmbeddings: 512,
    })

    installTokenLimitClamp(pipeline)
    expect(pipeline.tokenizer).not.toBe(tokenizer)
    pipeline.tokenizer(['clamped'], { padding: true, truncation: false })

    expect(tokenizerCalls).toEqual([{ padding: true, truncation: true, max_length: 510 }])
  })

  it('preserves the other tokenizer members through the clamp proxy', () => {
    const { pipeline } = createFakePipeline({ modelMaxLength: 1e30, maxPositionEmbeddings: 512 })

    installTokenLimitClamp(pipeline)

    expect(Reflect.get(pipeline.tokenizer, 'model_max_length')).toBe(1e30)
  })
})
