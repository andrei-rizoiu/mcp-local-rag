// RAG MCP Server Security Test - Design Doc: rag-mcp-server-design.md (v1.1)
// Note: Reduced based on YAGNI principle and avoiding redundancy

import { mkdir, rm, writeFile } from 'node:fs/promises'
import { resolve } from 'node:path'
import { format } from 'node:util'
import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'
import { DocumentParser } from '../../parser/index.js'
import { RAGServer } from '../../server/index.js'
import { testModelCacheDir, withTestDevice } from '../test-device.js'
import { asDouble, expectError, privateMembers } from '../test-doubles.js'

// ============================================
// Test Configuration
// ============================================

const testConfig = {
  dbPath: './tmp/test-security-db',
  modelName: 'Xenova/all-MiniLM-L6-v2',
  cacheDir: testModelCacheDir(),
  baseDir: resolve('./'), // Project root (accessible to both tests/fixtures and tmp)
  maxFileSize: 100 * 1024 * 1024, // 100MB
}

// ============================================
// Network Monitoring Helper
// ============================================

interface NetworkMonitor {
  requests: string[]
  restore: () => void
}

// The stack-trace policy is enforced on the CLIENT-facing message, which the
// central dispatcher mapper builds via `toMcpError`/`formatErrorForClient`.
// Handlers themselves rethrow the original error identity, so the policy is
// observed by invoking the registered dispatcher closure (the MCP boundary)
// rather than the handler method directly.
type RegisteredHandler = (
  request: { method: string; params: { name: string; arguments?: unknown } },
  extra: { signal: AbortSignal }
) => Promise<{ content: { type: string; text: string }[] }>

async function dispatchTool(server: RAGServer, name: string, args: unknown): Promise<unknown> {
  const inner = privateMembers<{
    server: { _requestHandlers: Map<string, RegisteredHandler> }
  }>(server)
  const handler = inner.server._requestHandlers.get('tools/call')
  if (handler === undefined) {
    throw new Error('tools/call handler not registered')
  }
  return handler(
    { method: 'tools/call', params: { name, arguments: args } },
    { signal: new AbortController().signal }
  )
}

function createNetworkMonitor(): NetworkMonitor {
  const requests: string[] = []
  const originalFetch = global.fetch

  global.fetch = asDouble<typeof fetch>(
    vi.fn(async (url: RequestInfo | URL) => {
      requests.push(url.toString())
      return originalFetch(url)
    })
  )

  return {
    requests,
    restore: () => {
      global.fetch = originalFetch
    },
  }
}

interface StderrCapture {
  lines: string[]
  restore: () => void
}

function captureStderr(): StderrCapture {
  const lines: string[] = []
  const record = (...args: unknown[]) => {
    lines.push(format(...args))
  }
  const errorSpy = vi.spyOn(console, 'error').mockImplementation(record)
  const warnSpy = vi.spyOn(console, 'warn').mockImplementation(record)
  return {
    lines,
    restore: () => {
      errorSpy.mockRestore()
      warnSpy.mockRestore()
    },
  }
}

// ============================================
// ============================================

describe('RAG MCP Server Security Test', () => {
  let server: RAGServer
  const fixturesDir = resolve('./tmp/test-security-fixtures')

  beforeAll(async () => {
    // Setup: Prepare environment for security testing
    await mkdir(testConfig.dbPath, { recursive: true })
    await mkdir(fixturesDir, { recursive: true })
  })

  afterAll(async () => {
    // Cleanup: Delete security test data (only our directories)
    await rm(testConfig.dbPath, { recursive: true, force: true })
    await rm(fixturesDir, { recursive: true, force: true })
  })

  afterEach(async () => {
    if (server) {
      await server.close()
    }
  })

  beforeEach(async () => {
    // Initialize server before each test
    server = new RAGServer(withTestDevice(testConfig))
    await server.initialize()

    // Create test fixture directory and files
    await mkdir(fixturesDir, { recursive: true })
    await writeFile(
      resolve(fixturesDir, 'sample.txt'),
      `This is a comprehensive sample text file for security testing purposes.

TypeScript is a powerful programming language that adds static typing to JavaScript.
It helps developers catch errors at compile time rather than runtime.
Many large-scale applications use TypeScript for improved maintainability.

Security testing is an essential part of software development.
It ensures that applications are protected against common vulnerabilities.
Proper security measures include input validation and authentication.

The RAG (Retrieval-Augmented Generation) system processes documents efficiently.
It uses vector embeddings to find semantically similar content.
This approach provides accurate search results for natural language queries.`
    )
  })

  // --------------------------------------------
  // S-001: No external network communication (except model download)
  // --------------------------------------------
  describe('S-001: No external network communication (except model download)', () => {
    // AC interpretation: [Security requirement] No external communication detected by packet capture (except model download)
    // Validation: No external communication occurs during file ingestion → search workflow after server startup
    it('No external communication detected during file ingestion → search workflow after server startup (simulated)', async () => {
      const monitor = createNetworkMonitor()

      try {
        // Snapshot before test operations (isolate: false may have concurrent fetch from other files)
        const beforeCount = monitor.requests.length

        // Ingest file
        const sampleFile = resolve(fixturesDir, 'sample.txt')
        await server.handleIngestFile({ filePath: sampleFile })

        // Execute search
        await server.handleQueryDocuments({ query: 'TypeScript', limit: 5 })

        // Verify no external communication occurred (exclude model download per AC)
        const newRequests = monitor.requests
          .slice(beforeCount)
          .filter((url) => !url.includes('huggingface.co'))
        expect(newRequests.length).toBe(0)
      } finally {
        monitor.restore()
      }
    })

    // LanceDB network isolation is covered by the workflow test above; a separate
    // absolute `requests.length === 0` check is omitted as fragile under isolate:false.
  })

  // --------------------------------------------
  // S-003: No document content in logs
  // --------------------------------------------
  describe('S-003: No document content in logs', () => {
    // AC interpretation: [Security requirement] Document content not output to logs
    // Validation: Logs during file ingestion do not contain document body
    it('Logs during file ingestion do not contain document body (max 100 characters allowed)', async () => {
      // Capture stderr logs (server-side logging goes through console.error/warn)
      const stderr = captureStderr()

      try {
        // Create test file (containing confidential information)
        const testFile = resolve('./tmp/secret-document.txt')
        await writeFile(
          testFile,
          `This is a secret document with confidential information: PASSWORD123.

The document contains sensitive data that should never appear in logs.
Proper logging practices require filtering out confidential information.
Security best practices dictate that passwords should be masked in output.

This paragraph provides additional content for proper semantic chunking.
Multiple sentences ensure the document is processed correctly by the RAG system.
The chunker requires sufficient text length to generate meaningful chunks.`
        )

        // Ingest file
        await server.handleIngestFile({ filePath: testFile })

        // Verify document content not included in logs
        const containsSecret = stderr.lines.some(
          (log) => log.includes('PASSWORD123') || log.includes('confidential information')
        )
        expect(containsSecret).toBe(false)
      } finally {
        stderr.restore()
      }
    })

    // AC interpretation: [Security requirement] Search queries not output to logs
    // Validation: Logs during search do not contain search query body (max 100 characters allowed)
    it('Logs during search do not contain search query body (max 100 characters allowed)', async () => {
      // Capture stderr logs (server-side logging goes through console.error/warn)
      const stderr = captureStderr()

      try {
        // Ingest sample file
        const sampleFile = resolve(fixturesDir, 'sample.txt')
        await server.handleIngestFile({ filePath: sampleFile })

        // Search with confidential query
        const secretQuery = 'secret query with confidential information PASSWORD123'
        await server.handleQueryDocuments({ query: secretQuery, limit: 5 })

        // Verify search query not included in logs
        const containsQuery = stderr.lines.some(
          (log) => log.includes('PASSWORD123') || log.includes('confidential information')
        )
        expect(containsQuery).toBe(false)
      } finally {
        stderr.restore()
      }
    })
  })

  // --------------------------------------------
  // S-004: MCP security best practices compliance
  // --------------------------------------------
  describe('S-004: MCP security best practices compliance', () => {
    // The stack-trace policy (formatErrorForClient) governs INTERNAL (non-input)
    // failures only. Input errors (bad path, missing file, unsupported format,
    // size) surface as McpError InvalidParams with a clean message and never
    // reach formatErrorForClient. To exercise the policy substantively we induce
    // a genuine internal failure: path/size validation passes for an existing
    // fixture, then the parse step throws a generic Error. The client message
    // never carries a stack trace, regardless of NODE_ENV.
    const internalErrorMessage = 'Induced internal parse failure'

    // Development mode: Stack traces are STILL withheld from the client. The
    // dev-mode stack is available on the LOG side (stderr) only; the
    // client-facing message never carries it, regardless of NODE_ENV.
    it('Stack traces not included even when NODE_ENV=development', async () => {
      const originalEnv = process.env['NODE_ENV']
      process.env['NODE_ENV'] = 'development'

      const sampleFile = resolve(fixturesDir, 'sample.txt')
      const parseSpy = vi
        .spyOn(DocumentParser.prototype, 'parseFile')
        .mockRejectedValue(new Error(internalErrorMessage))

      try {
        await dispatchTool(server, 'ingest_file', { filePath: sampleFile })
        expect.fail('Expected error to be thrown')
      } catch (error) {
        const errorMessage = expectError(error).message

        // The internal failure message still surfaces, but the client message
        // never carries stack-trace details even in development mode (the
        // client message is built by the central mapper's formatErrorForClient).
        expect(errorMessage).toContain(internalErrorMessage)
        expect(errorMessage).not.toContain(' at ')
        expect(errorMessage).not.toContain('.ts:')
        expect(errorMessage).not.toContain('.js:')
      } finally {
        parseSpy.mockRestore()
        // Restore environment variable
        process.env['NODE_ENV'] = originalEnv
      }
    })
  })
})
