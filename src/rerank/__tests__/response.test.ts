// Tests validateRerankResponse: the server checks the form of the command's
// answer and leaves the content to the command.

import { describe, expect, it } from 'vitest'
import { validateRerankResponse } from '../response.js'

const result = {
  filePath: '/docs/auth.md',
  chunkIndex: 0,
  text: 'bearer token flow',
  score: 0.21,
  fileTitle: 'Auth Guide',
  images: [],
}

const out = (items: unknown[]): string => JSON.stringify(items)

describe('validateRerankResponse acceptance', () => {
  it('accepts a schema-conforming result set', () => {
    const match = validateRerankResponse(out([result]))

    expect(match.ok).toBe(true)
    if (match.ok) {
      expect(match.results).toEqual([result])
    }
  })

  it('accepts an empty result set as an answer rather than a failure', () => {
    const match = validateRerankResponse('[]')

    expect(match.ok).toBe(true)
    if (match.ok) {
      expect(match.results).toEqual([])
    }
  })

  it('accepts the same chunk twice', () => {
    const match = validateRerankResponse(out([result, result]))

    expect(match.ok).toBe(true)
    if (match.ok) {
      expect(match.results).toHaveLength(2)
    }
  })

  it('carries through a property the schema does not describe', () => {
    const match = validateRerankResponse(out([{ ...result, rerankScore: 0.93 }]))

    expect(match.ok).toBe(true)
    if (match.ok) {
      expect(match.results[0]).toHaveProperty('rerankScore', 0.93)
    }
  })

  it('accepts an attachment that matches the schema', () => {
    const images = [{ imageIndex: 0, mimeType: 'image/png', data: 'aW1hZ2U=' }]
    const match = validateRerankResponse(out([{ ...result, images }]))

    expect(match.ok).toBe(true)
    if (match.ok) {
      expect(match.results[0]?.images).toEqual(images)
    }
  })

  it('accepts a source string on a raw-data result', () => {
    const match = validateRerankResponse(out([{ ...result, source: 'https://example.com/page' }]))

    expect(match.ok).toBe(true)
  })

  it('tolerates surrounding whitespace around the JSON array', () => {
    const match = validateRerankResponse(`\n  ${out([result])}\n`)

    expect(match.ok).toBe(true)
  })
})

describe('validateRerankResponse rejection', () => {
  const rejects = (label: string, stdout: string): void => {
    it(`rejects ${label}`, () => {
      const match = validateRerankResponse(stdout)

      expect(match.ok).toBe(false)
    })
  }

  rejects('stdout that does not parse as JSON', 'not json')
  rejects('JSON that is not an array', '{"results":[]}')
  rejects('a missing required property', out([{ ...result, text: undefined }]))
  rejects('a filePath of the wrong type', out([{ ...result, filePath: 7 }]))
  rejects('a non-integer chunkIndex', out([{ ...result, chunkIndex: 1.5 }]))
  rejects('a negative chunkIndex', out([{ ...result, chunkIndex: -1 }]))
  rejects('a non-finite score', out([{ ...result, score: Number.NaN }]))
  rejects('a fileTitle that is neither string nor null', out([{ ...result, fileTitle: 3 }]))
  rejects('a source of the wrong type', out([{ ...result, source: 42 }]))
  rejects('images that are not an array', out([{ ...result, images: {} }]))
  rejects(
    'an unsupported mimeType',
    out([{ ...result, images: [{ imageIndex: 0, mimeType: 'image/gif', data: 'x' }] }])
  )
  rejects(
    'an attachment missing its data',
    out([{ ...result, images: [{ imageIndex: 0, mimeType: 'image/png' }] }])
  )

  it('carries neither document text nor a path in its rejection reason', () => {
    const match = validateRerankResponse(out([{ ...result, chunkIndex: 'zero' }]))

    expect(match.ok).toBe(false)
    if (!match.ok) {
      expect(match.reason).not.toContain('bearer token flow')
      expect(match.reason).not.toContain('/docs/auth.md')
    }
  })
})
