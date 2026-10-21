// RAGServer degraded-mode construction guards (P3-T1)
//
// Empty `baseDirs` without a `configError` throws; with a `configError` the
// server stays constructible but the parser fails closed on every path.
//
// The config *shape* contract (baseDir/baseDirs wiring, configWarnings/
// configError) is verified observably via real handlers in
// rag-server.files.integration.test.ts (list_files) and
// rag-server.warning-visibility.test.ts, so it is not re-checked here.

import { mkdirSync, rmSync } from 'node:fs'
import { resolve } from 'node:path'
import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import { testModelCacheDir } from '../../__tests__/test-device.js'
import { privateMembers } from '../../__tests__/test-doubles.js'
import type { DocumentParser } from '../../parser/index.js'
import {
  parseRerankCmd,
  parseRerankTimeoutMs,
  parseStoreImages,
  resolveServerConfig,
} from '../../server-main.js'
import { BaseDirsConfigError } from '../../utils/base-dirs.js'
import {
  DEFAULT_RERANK_TIMEOUT_MS,
  RERANK_TIMEOUT_MAX_MS,
  RERANK_TIMEOUT_MIN_MS,
} from '../../utils/limits.js'
import { RAGServer } from '../index.js'

describe('STORE_IMAGES configuration', () => {
  it.each([
    [undefined, false],
    ['', false],
    ['   ', false],
    ['1', true],
    [' true ', true],
    ['YES', true],
    ['on', true],
    ['0', false],
    [' false ', false],
    ['NO', false],
    ['off', false],
  ] as const)('parses %j as %s', (raw, expected) => {
    expect(parseStoreImages(raw)).toEqual({ value: expected })
  })

  it('warns and disables image storage for an invalid value', () => {
    expect(parseStoreImages('sometimes')).toEqual({
      value: false,
      warning:
        'Invalid STORE_IMAGES value: "sometimes". Expected one of 1, true, yes, on, 0, false, no, or off. Using false.',
    })
  })

  it('threads the parsed value and warning through resolveServerConfig', async () => {
    const cwd = resolve('./tmp/test-lancedb-config-shape')
    const enabled = await resolveServerConfig({ BASE_DIR: cwd, STORE_IMAGES: 'yes' }, cwd)
    expect(enabled.storeImages).toBe(true)

    const invalid = await resolveServerConfig({ BASE_DIR: cwd, STORE_IMAGES: 'invalid' }, cwd)
    expect(invalid.storeImages).toBe(false)
    expect(invalid.configWarnings).toContain(
      'Invalid STORE_IMAGES value: "invalid". Expected one of 1, true, yes, on, 0, false, no, or off. Using false.'
    )
  })
})

describe('EMBED_TITLE_PREFIX configuration', () => {
  it('threads the parsed value and warning through resolveServerConfig', async () => {
    const cwd = resolve('./tmp/test-lancedb-config-shape')
    const unset = await resolveServerConfig({ BASE_DIR: cwd }, cwd)
    expect(unset.titlePrefix).toBe(false)

    const enabled = await resolveServerConfig({ BASE_DIR: cwd, EMBED_TITLE_PREFIX: '1' }, cwd)
    expect(enabled.titlePrefix).toBe(true)

    const invalid = await resolveServerConfig({ BASE_DIR: cwd, EMBED_TITLE_PREFIX: 'maybe' }, cwd)
    expect(invalid.titlePrefix).toBe(false)
    expect(invalid.configWarnings).toContain(
      'Invalid EMBED_TITLE_PREFIX value: "maybe". Expected one of 1, true, yes, on, 0, false, no, or off. Using false.'
    )
  })
})

describe('EMBED_HEADING_PREFIX configuration', () => {
  it('threads the independent value and invalid-value warning', async () => {
    const cwd = resolve('./tmp/test-lancedb-config-shape')
    expect((await resolveServerConfig({ BASE_DIR: cwd }, cwd)).headingPrefix).toBe(false)
    const enabled = await resolveServerConfig({ BASE_DIR: cwd, EMBED_HEADING_PREFIX: 'on' }, cwd)
    expect(enabled.headingPrefix).toBe(true)
    expect(enabled.titlePrefix).toBe(false)
    const invalid = await resolveServerConfig({ BASE_DIR: cwd, EMBED_HEADING_PREFIX: 'maybe' }, cwd)
    expect(invalid.headingPrefix).toBe(false)
    expect(invalid.configWarnings).toContain(
      'Invalid EMBED_HEADING_PREFIX value: "maybe". Expected one of 1, true, yes, on, 0, false, no, or off. Using false.'
    )
  })
})

describe('IGNORE_PATHS configuration', () => {
  const cwd = resolve('./tmp/test-lancedb-config-shape')

  it('threads the parsed patterns, and no patterns when unset', async () => {
    expect((await resolveServerConfig({ BASE_DIR: cwd }, cwd)).ignorePatterns).toEqual([])
    const configured = await resolveServerConfig(
      { BASE_DIR: cwd, IGNORE_PATHS: '["plugs/**","**/*.draft.md"]' },
      cwd
    )
    expect(configured.ignorePatterns).toEqual(['plugs/**', '**/*.draft.md'])
  })

  it('refuses to start on a value that is not a JSON array', async () => {
    await expect(
      resolveServerConfig({ BASE_DIR: cwd, IGNORE_PATHS: 'plugs/**:drafts/**' }, cwd)
    ).rejects.toThrow(/^IGNORE_PATHS must be a JSON array/)
  })
})

describe('RAGServerConfig degraded-mode construction guards (P3-T1)', () => {
  const testDbPath = resolve('./tmp/test-lancedb-config-shape')

  beforeAll(() => {
    mkdirSync(testDbPath, { recursive: true })
  })

  afterAll(() => {
    rmSync(testDbPath, { recursive: true, force: true })
  })

  it('rejects construction with an empty baseDirs array when configError is absent', () => {
    // Without configError, empty `baseDirs` is misconfiguration: the
    // constructor must throw rather than silently build a parser that
    // rejects every path.
    expect(
      () =>
        new RAGServer({
          dbPath: testDbPath,
          modelName: 'Xenova/all-MiniLM-L6-v2',
          cacheDir: testModelCacheDir(),
          baseDirs: [],
          maxFileSize: 100 * 1024 * 1024,
        })
    ).toThrow(/non-empty `baseDirs` array/)
  })

  it('parser constructed with empty baseDirs fails closed on validateFilePath', async () => {
    // Defense-in-depth: even when a handler bypasses `assertConfigOk`, the
    // parser must reject every path under degraded mode.
    const configError = new BaseDirsConfigError(
      'BASE_DIRS must be a JSON array of non-empty path strings.'
    )
    const server = new RAGServer({
      dbPath: testDbPath,
      modelName: 'Xenova/all-MiniLM-L6-v2',
      cacheDir: testModelCacheDir(),
      baseDirs: [],
      maxFileSize: 100 * 1024 * 1024,
      configError,
    })
    const { parser } = privateMembers<{ parser: DocumentParser }>(server)
    await expect(parser.validateFilePath('/tmp/anything.txt')).rejects.toThrow(
      /No configured base directory/
    )
  })
})

describe('RAG_RERANK_CMD configuration', () => {
  const cwd = resolve('./tmp/test-lancedb-config-shape')

  it('leaves the command unset when the variable is unset or blank', () => {
    expect(parseRerankCmd(undefined)).toEqual({ value: undefined })
    expect(parseRerankCmd('   ')).toEqual({ value: undefined })
  })

  it('keeps the configured command verbatim apart from surrounding whitespace', () => {
    expect(parseRerankCmd(' /usr/local/bin/reranker --model base ')).toEqual({
      value: '/usr/local/bin/reranker --model base',
    })
  })

  it('threads the command through resolveServerConfig', async () => {
    const disabled = await resolveServerConfig({ BASE_DIR: cwd }, cwd)
    expect(disabled.rerankCommand).toBeUndefined()

    const enabled = await resolveServerConfig(
      { BASE_DIR: cwd, RAG_RERANK_CMD: '/usr/local/bin/reranker' },
      cwd
    )
    expect(enabled.rerankCommand).toBe('/usr/local/bin/reranker')
  })
})

describe('RAG_RERANK_TIMEOUT_MS configuration', () => {
  const cwd = resolve('./tmp/test-lancedb-config-shape')

  it('uses the default when the variable is unset or empty', () => {
    expect(parseRerankTimeoutMs(undefined)).toEqual({ value: DEFAULT_RERANK_TIMEOUT_MS })
    expect(parseRerankTimeoutMs('')).toEqual({ value: DEFAULT_RERANK_TIMEOUT_MS })
  })

  it.each([String(RERANK_TIMEOUT_MIN_MS), '30000', String(RERANK_TIMEOUT_MAX_MS)])(
    'accepts %s',
    (raw) => {
      expect(parseRerankTimeoutMs(raw)).toEqual({ value: Number(raw) })
    }
  )

  // Node's timers clamp a delay below 1 or above 2^31-1 to 1ms and truncate a
  // fraction, so each of these would otherwise make every rerank time out at once.
  it.each(['0', '-1', '1.5', String(RERANK_TIMEOUT_MAX_MS + 1), '2147483648', 'soon'])(
    'warns and keeps the default for %j',
    (raw) => {
      const parsed = parseRerankTimeoutMs(raw)
      expect(parsed.value).toBe(DEFAULT_RERANK_TIMEOUT_MS)
      expect(parsed.warning).toBe(
        `Invalid RAG_RERANK_TIMEOUT_MS value: "${raw}". Expected integer between ${RERANK_TIMEOUT_MIN_MS} and ${RERANK_TIMEOUT_MAX_MS}. Using default (${DEFAULT_RERANK_TIMEOUT_MS}).`
      )
    }
  )

  it('threads the parsed value and warning through resolveServerConfig', async () => {
    const valid = await resolveServerConfig({ BASE_DIR: cwd, RAG_RERANK_TIMEOUT_MS: '2000' }, cwd)
    expect(valid.rerankTimeoutMs).toBe(2000)
    expect(valid.configWarnings ?? []).not.toContainEqual(
      expect.stringContaining('RAG_RERANK_TIMEOUT_MS')
    )

    const invalid = await resolveServerConfig({ BASE_DIR: cwd, RAG_RERANK_TIMEOUT_MS: '0' }, cwd)
    expect(invalid.rerankTimeoutMs).toBe(DEFAULT_RERANK_TIMEOUT_MS)
    expect(invalid.configWarnings).toContain(
      `Invalid RAG_RERANK_TIMEOUT_MS value: "0". Expected integer between ${RERANK_TIMEOUT_MIN_MS} and ${RERANK_TIMEOUT_MAX_MS}. Using default (${DEFAULT_RERANK_TIMEOUT_MS}).`
    )
  })
})
