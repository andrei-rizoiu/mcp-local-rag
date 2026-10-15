import type { Tool } from '@modelcontextprotocol/sdk/types.js'
import { describe, expect, it } from 'vitest'
import { expectString } from '../../__tests__/test-doubles.js'
import { MAX_QUERY_LIMIT, MIN_QUERY_LIMIT } from '../../utils/limits.js'
import { isRecord } from '../../utils/type-guards.js'
import { toolDefinitions } from '../tool-definitions.js'

const findTool = (name: string): Tool => {
  const tool = toolDefinitions.find((t) => t.name === name)
  if (!tool) {
    throw new Error(`tool ${name} not found`)
  }
  return tool
}

/** One property of a tool's JSON-schema input, or an empty record when absent. */
function propertyOf(tool: Tool, property: string): Record<string, unknown> {
  const properties = tool.inputSchema.properties
  if (!isRecord(properties)) {
    return {}
  }
  const value = properties[property]
  return isRecord(value) ? value : {}
}

describe('query_documents tool definition limit', () => {
  const limit = propertyOf(findTool('query_documents'), 'limit')

  // Both sides read the same constant, so this does not prove the value; it
  // proves the wiring and fails if a number is hardcoded into the schema.
  it('advertises the shared limit range', () => {
    expect(limit['minimum']).toBe(MIN_QUERY_LIMIT)
    expect(limit['maximum']).toBe(MAX_QUERY_LIMIT)
  })

  it('names the shared range in its description', () => {
    expect(expectString(limit['description'])).toContain(
      `range ${MIN_QUERY_LIMIT}-${MAX_QUERY_LIMIT}`
    )
  })
})

describe('list_files tool definition scope', () => {
  const listFiles = findTool('list_files')
  const scope = propertyOf(listFiles, 'scope')

  it('advertises an optional scope property', () => {
    expect(Object.keys(scope).length).toBeGreaterThan(0)
    // scope is optional: list_files declares no required arguments (backward compatible)
    expect(listFiles.inputSchema.required).toBeUndefined()
  })

  it('uses the string | array<string> oneOf shape', () => {
    expect(scope['oneOf']).toEqual([
      { type: 'string' },
      { type: 'array', items: { type: 'string' } },
    ])
  })

  // Regression guard with provenance: the description once carried the
  // query_documents wording, which claims stored-filePath semantics and
  // contradicts the scan-path basis list_files actually filters on.
  it('does not claim the query_documents stored-filePath semantics', () => {
    expect(expectString(scope['description'])).not.toMatch(/filePath equal to or under/)
  })
})

describe('declared tool surface', () => {
  // SYNC-006 / ADR decision 1: the sync tools are ordinary tools/call entries.
  // Any extra top-level key would be a negotiated capability the server does
  // not implement.
  it.each(['sync_start', 'sync_status'])(
    '%s declares no notification, task, or experimental field',
    (name) => {
      expect(Object.keys(findTool(name)).sort()).toEqual(['description', 'inputSchema', 'name'])
    }
  )
})

describe('sync_start tool definition', () => {
  const syncStart = findTool('sync_start')

  it('declares path as an optional string', () => {
    expect(propertyOf(syncStart, 'path').type).toBe('string')
    // Optional: no required array at all, matching list_files and delete_file.
    expect(syncStart.inputSchema.required).toBeUndefined()
  })

  it('declares no property other than path', () => {
    expect(Object.keys(syncStart.inputSchema.properties ?? {})).toEqual(['path'])
  })

  it('exposes no visual input field while documenting per-file inheritance', () => {
    expect(JSON.stringify(syncStart.inputSchema)).not.toMatch(/visual/i)
    expect(syncStart.description).toMatch(/visual profile/i)
    expect(syncStart.description).toMatch(/already recorded/i)
  })
})

describe('sync_status tool definition', () => {
  const syncStatus = findTool('sync_status')

  it('declares jobId as a required string', () => {
    expect(propertyOf(syncStatus, 'jobId').type).toBe('string')
    expect(Object.keys(syncStatus.inputSchema.properties ?? {})).toEqual(['jobId'])
    expect(syncStatus.inputSchema.required).toEqual(['jobId'])
  })
})
