// Lazy initialization tests for Embedder
// TDD Red phase: These tests should fail initially

import type { MockInstance } from 'vitest'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { getTestDevice, testModelCacheDir } from '../../__tests__/test-device.js'
import { privateMembers } from '../../__tests__/test-doubles.js'
import type { EmbedderConfig } from '../index.js'
import { Embedder } from '../index.js'

/**
 * There is no public init-count surface, and mocking transformers instead
 * would leak across files (`isolate: false`) and lose real-model coverage.
 */
function spyOnInitialize(embedder: Embedder): MockInstance {
  return vi.spyOn(privateMembers<{ initialize: () => Promise<void> }>(embedder), 'initialize')
}

describe('Embedder - Lazy Initialization', () => {
  let testConfig: EmbedderConfig

  beforeEach(() => {
    testConfig = {
      modelPath: 'Xenova/all-MiniLM-L6-v2',
      batchSize: 8,
      cacheDir: testModelCacheDir(),
      device: getTestDevice(),
    }
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  // Test 1: Lazy initialization on first embed() call
  it('should initialize on first embed() call without explicit initialize()', async () => {
    const embedder = new Embedder(testConfig)
    // Note: NOT calling await embedder.initialize()

    const result = await embedder.embed('test text for lazy initialization')

    expect(result).toBeDefined()
    expect(Array.isArray(result)).toBe(true)
    expect(result.length).toBe(384)
    expect(result.every((value) => typeof value === 'number')).toBe(true)
  }, 180000) // 3 minute timeout for model download

  // Test 3: Initialization should happen only once for concurrent calls
  it('should initialize only once for concurrent embed() calls', async () => {
    const embedder = new Embedder(testConfig)

    // The lazy-init-once contract under concurrency.
    const initializeSpy = spyOnInitialize(embedder)

    // Make 5 concurrent embed() calls
    const promises = Array.from({ length: 5 }, (_, i) => embedder.embed(`concurrent test ${i}`))

    const results = await Promise.all(promises)

    // Verify all calls succeeded
    expect(results).toHaveLength(5)
    expect(results.every((result) => result.length === 384)).toBe(true)

    // Verify initialize was called only once
    expect(initializeSpy).toHaveBeenCalledTimes(1)
  }, 180000)

  // Init-failure surfacing on the lazy path is covered in
  // __tests__/embedder/embedder.test.ts ('device validation'), which asserts the
  // same EmbeddingError plus the underlying `Unsupported device` text.
})
