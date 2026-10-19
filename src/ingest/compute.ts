// Shared chunk + embed computation for the ingest pipeline: the single
// `chunkText -> embedBatch` call site for any ingest path.
//
// Persistence (delete + insert + rollback + optimize) stays in each caller,
// because the rollback semantics differ between the MCP and CLI paths.

import { createHash, randomUUID } from 'node:crypto'
import type { AtomicTextRange, SemanticChunker, TextChunk } from '../chunker/index.js'
import { type EmbedderInterface, resolveContainmentBudget } from '../chunker/semantic-chunker.js'
import type { ParseResult } from '../parser/index.js'
import type { ImageRendition } from '../pdf-visual/types.js'
import { MAX_VISUAL_RENDITION_BYTES } from '../utils/limits.js'
import {
  type SourceContext,
  type SourceMap,
  sourceContextForRange,
} from '../utils/source-context.js'
import type { VectorChunk, VisualAttachment } from '../vectordb/index.js'

/** Distance from `offset` to a chunk's source span; 0 when it falls inside. */
function distanceToSpan(offset: number, chunk: TextChunk): number {
  if (offset < chunk.sourceStart) {
    return chunk.sourceStart - offset
  }
  if (offset > chunk.sourceEnd) {
    return offset - chunk.sourceEnd
  }
  return 0
}

const TITLE_KEY = 'Title:'

/**
 * The `Title:` header a chunk is embedded behind, or `''` when the embedder
 * does not opt in, there is no title, the model's document prompt already
 * starts with the same key, or the header would take more than half the token
 * window: the chunker budgets every chunk for it, so a header near the cap
 * would shred the body into near-empty chunks.
 */
async function resolveTitlePrefix(
  title: string | null | undefined,
  embedder: EmbedderInterface
): Promise<string> {
  if (!embedder.titlePrefix || !title) {
    return ''
  }
  const documentPrompt = (await embedder.getDocumentPrompt?.()) ?? ''
  if (documentPrompt.toLowerCase().startsWith(TITLE_KEY.toLowerCase())) {
    return ''
  }
  const prefix = `${TITLE_KEY} ${title}\n\n`
  const budget = await resolveContainmentBudget(embedder, '')
  if (budget === null) {
    return prefix
  }
  const [tokens = 0] = await budget.countTokens([prefix])
  return tokens <= budget.cap / 2 ? prefix : ''
}

/** `embeddings` has the same length as `chunks`, index for index. */
export interface BuildChunksAndEmbeddingsResult {
  chunks: (TextChunk & { sourceContext?: SourceContext })[]
  embeddings: number[][]
}

export interface BuildChunksFromParseResultResult extends BuildChunksAndEmbeddingsResult {
  visualAttachments: Map<number, VisualAttachment[]>
  omittedImageCount: number
}

export function findNearestChunk(
  chunks: readonly TextChunk[],
  sourceOffset: number
): TextChunk | undefined {
  let owner: TextChunk | undefined
  let bestDistance = Number.POSITIVE_INFINITY
  for (const chunk of chunks) {
    const distance = distanceToSpan(sourceOffset, chunk)
    if (!owner) {
      owner = chunk
      bestDistance = distance
      continue
    }
    const isPreceding = chunk.sourceEnd <= sourceOffset
    const ownerIsPreceding = owner.sourceEnd <= sourceOffset
    if (
      distance < bestDistance ||
      (distance === bestDistance && isPreceding && !ownerIsPreceding) ||
      (distance === bestDistance && isPreceding === ownerIsPreceding && chunk.index < owner.index)
    ) {
      owner = chunk
      bestDistance = distance
    }
  }
  return owner
}

export function createVisualAttachment(
  imageIndex: number,
  rendition: ImageRendition
): VisualAttachment {
  if (
    !Number.isInteger(imageIndex) ||
    imageIndex < 0 ||
    rendition.bytes.byteLength === 0 ||
    rendition.bytes.byteLength > MAX_VISUAL_RENDITION_BYTES
  ) {
    throw new Error('Invalid bounded image rendition')
  }
  return {
    imageIndex,
    mimeType: rendition.mimeType,
    data: Buffer.from(rendition.bytes).toString('base64'),
  }
}

/**
 * Compute semantic chunks and their embeddings for already-extracted text.
 *
 * Does not fail fast on zero chunks — the MCP handler throws, the CLI warns
 * and returns 0. Chunker and embedder errors propagate verbatim.
 */
export async function buildChunksAndEmbeddings(
  text: string,
  chunker: SemanticChunker,
  embedder: EmbedderInterface,
  options: {
    sourceMap?: SourceMap | undefined
    atomicRanges?: readonly AtomicTextRange[] | undefined
    title?: string | null | undefined
  } = {}
): Promise<BuildChunksAndEmbeddingsResult> {
  const prefix = await resolveTitlePrefix(options.title, embedder)
  const chunks = await chunker.chunkText(text, embedder, options.atomicRanges, prefix)
  // F5: Skip `embedBatch` entirely on zero chunks. `embedBatch` runs
  // `ensureInitialized()` (which triggers the ~90MB MiniLM download on a
  // cold cache) BEFORE checking for the empty-array short-circuit, so an
  // empty file would otherwise pay the model-load cost for no work.
  if (chunks.length === 0) {
    return { chunks: [], embeddings: [] }
  }
  const sourceMap = options.sourceMap
  const contextualChunks = chunks.map((chunk) => ({
    ...chunk,
    ...(sourceMap
      ? {
          sourceContext: sourceContextForRange(sourceMap, chunk.sourceStart, chunk.sourceEnd),
        }
      : {}),
  }))
  const inputs = await embeddingInputs(contextualChunks, prefix, embedder)
  return { chunks: contextualChunks, embeddings: await embedder.embedBatch(inputs, 'document') }
}

/** Add optional section context only when the full body still fits. */
async function embeddingInputs(
  chunks: BuildChunksAndEmbeddingsResult['chunks'],
  prefix: string,
  embedder: EmbedderInterface
): Promise<string[]> {
  // Prefixes affect vectors only; stored text and the FTS index stay body-only.
  const inputs = chunks.map((chunk) => prefix + chunk.text)
  if (
    !embedder.headingPrefix ||
    !chunks.some((chunk) => chunk.sourceContext?.headingPaths.length)
  ) {
    return inputs
  }
  const prefixes = chunks.map((chunk) => {
    const paths = chunk.sourceContext?.headingPaths ?? []
    return paths.length
      ? `${prefix}${paths.map((path) => `Section: ${path.join(' > ')}`).join('\n')}\n\n`
      : prefix
  })
  const candidates = chunks.map((chunk, i) => prefixes[i] + chunk.text)
  const budget = await resolveContainmentBudget(embedder, '')
  if (!budget) {
    return candidates
  }
  const prefixSizes = await budget.countTokens(prefixes)
  const inputSizes = await budget.countTokens(candidates)
  return candidates.map((candidate, i) =>
    (prefixSizes[i] ?? Infinity) <= budget.cap / 2 && (inputSizes[i] ?? Infinity) <= budget.cap
      ? candidate
      : (inputs[i] ?? '')
  )
}

/**
 * Preserve the parser content/range mapping at one shared boundary. The title
 * affects embeddings only when the embedder opts into the title prefix.
 */
export async function buildChunksFromParseResult(
  result: ParseResult,
  chunker: SemanticChunker,
  embedder: EmbedderInterface
): Promise<BuildChunksFromParseResultResult> {
  const computed = await buildChunksAndEmbeddings(result.content, chunker, embedder, {
    sourceMap: result.sourceMap,
    atomicRanges: result.atomicRanges,
    title: result.embeddingTitle ?? result.title,
  })
  const visualAttachments = new Map<number, VisualAttachment[]>()
  if (!result.imageAnchors?.length || computed.chunks.length === 0) {
    return { ...computed, visualAttachments, omittedImageCount: 0 }
  }

  const { renderImageRendition } = await import('../pdf-visual/renderer.js')
  let omittedCount = 0
  for (const anchor of [...result.imageAnchors].sort(
    (left, right) => left.imageIndex - right.imageIndex
  )) {
    const owner = findNearestChunk(computed.chunks, anchor.offset)
    if (!owner) {
      throw new Error(`Image ${anchor.imageIndex} has no owning chunk`)
    }

    try {
      const rendition = renderImageRendition(anchor.bytes, anchor.mimeType)
      const attachment = createVisualAttachment(anchor.imageIndex, rendition)
      const owned = visualAttachments.get(owner.index) ?? []
      owned.push(attachment)
      visualAttachments.set(owner.index, owned)
    } catch {
      omittedCount += 1
    }
  }
  return { ...computed, visualAttachments, omittedImageCount: omittedCount }
}

/**
 * Content identity of a source file: SHA-256 of its raw BYTES, not of the
 * parsed text, so any caller that can read the file reproduces it — which is
 * what lets a later sync decide "unchanged" without re-parsing.
 */
export function computeContentHash(bytes: Uint8Array): string {
  return createHash('sha256').update(bytes).digest('hex')
}

/**
 * Build persistable `VectorChunk`s from computed chunks + embeddings — the
 * single chunk→row mapping for both the MCP handler and both CLI paths. One
 * shared `timestamp` per call, a fresh `id`, and `fileName`/`fileType` derived
 * with `node:path` so they hold on Windows.
 *
 * Throws when a chunk has no embedding; the two arrays must align 1:1.
 *
 * `contentHash` is `null` for a chunk set with no source file, and the key is
 * then omitted rather than stored empty, so a hashless row is never mistaken
 * for a real hash. Required, not optional, so a new call site cannot silently
 * write hashless rows.
 *
 * `visualProfile` is the requested visual intent of this ingestion, copied to
 * every row. It accepts `null`/`undefined` and then omits the optional property,
 * so a caller holding a nullable value can pass it directly under
 * `exactOptionalPropertyTypes` and a caller with no visual concept can omit it.
 */
export function buildVectorChunks(params: {
  filePath: string
  chunks: (TextChunk & { sourceContext?: SourceContext })[]
  embeddings: number[][]
  fileTitle: string | null
  contentHash: string | null
  visualAttachments?: ReadonlyMap<number, readonly VisualAttachment[]>
  visualProfile?: string | null
}): VectorChunk[] {
  const {
    filePath,
    chunks,
    embeddings,
    fileTitle,
    contentHash,
    visualAttachments = new Map(),
    visualProfile = null,
  } = params
  const timestamp = new Date().toISOString()
  return chunks.map((chunk, index) => {
    const embedding = embeddings[index]
    if (!embedding) {
      throw new Error(`Missing embedding for chunk ${index}`)
    }
    const attachments = [...(visualAttachments.get(chunk.index) ?? [])]
      .sort((left, right) => left.imageIndex - right.imageIndex)
      .map(({ imageIndex, mimeType, data }) => ({
        imageIndex,
        mimeType,
        data,
      }))
    return {
      id: randomUUID(),
      filePath,
      chunkIndex: chunk.index,
      text: chunk.text,
      vector: embedding,
      fileTitle,
      ...(chunk.sourceContext ? { sourceContext: JSON.stringify(chunk.sourceContext) } : {}),
      ...(contentHash === null ? {} : { contentHash }),
      ...(visualProfile === null || visualProfile === undefined ? {} : { visualProfile }),
      visualAttachments: JSON.stringify(attachments),
      timestamp,
    }
  })
}
