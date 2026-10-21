#!/usr/bin/env node
// Shared read-only server entry point: one process, one vector store and one
// embedder, serving every client that connects to a unix socket. Each
// connection speaks the stdio framing (newline-delimited JSON-RPC), so a client
// reaches it through `socat STDIO UNIX-CONNECT:<socket>`. Ingestion stays with
// the stdio server the import scripts spawn; LanceDB versioning plus
// `readConsistencyInterval: 0` lets this reader see each import on its next query.

import { chmodSync, unlinkSync } from 'node:fs'
import { createServer, type Socket } from 'node:net'
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js'
import { RAGServer } from './server/index.js'
import { resolveServerConfig } from './server-main.js'

/** systemd passes activated sockets from fd 3 (sd_listen_fds). */
const SD_LISTEN_FDS_START = 3

function activatedBySystemd(env: NodeJS.ProcessEnv): boolean {
  return env['LISTEN_FDS'] === '1' && env['LISTEN_PID'] === String(process.pid)
}

interface ConnectionTracker {
  opened(): void
  closed(): void
}

/**
 * Call `onIdle` once no connection has been open for `idleMs`, so an unused
 * server frees its memory. Under socket activation systemd keeps listening and
 * starts the server again on the next connection. `idleMs` 0 disables it.
 */
function createIdleTracker(idleMs: number, onIdle: () => void): ConnectionTracker {
  let open = 0
  let timer: NodeJS.Timeout | undefined
  const arm = (): void => {
    if (idleMs > 0 && open === 0) {
      timer = setTimeout(onIdle, idleMs)
    }
  }
  arm()
  return {
    opened(): void {
      open++
      clearTimeout(timer)
    },
    closed(): void {
      open--
      arm()
    },
  }
}

function parseIdleExitMs(value: string | undefined): number {
  if (!value) {
    return 0
  }
  if (!/^\d+$/.test(value)) {
    console.error(`Invalid RAG_IDLE_EXIT_SEC value: "${value.slice(0, 100)}". Idle exit disabled.`)
    return 0
  }
  return Number(value) * 1000
}

function serveConnection(rag: RAGServer, socket: Socket, tracker: ConnectionTracker): void {
  tracker.opened()
  // A client that disappears mid-reply raises EPIPE on the socket; the close
  // handler below cleans up, so the error itself needs no further handling.
  socket.on('error', () => {})
  const transport = new StdioServerTransport(socket, socket)
  const served = rag.connectReadOnly(transport)
  socket.on('close', () => {
    tracker.closed()
    served.then((server) => server.close()).catch(() => {})
  })
  served.catch((error: unknown) => {
    console.error('Failed to serve connection:', error)
    socket.destroy()
  })
}

async function startSharedServer(): Promise<void> {
  const config = await resolveServerConfig(process.env, process.cwd())
  if (config.configWarnings && config.configWarnings.length > 0) {
    console.error('Configuration warnings:', config.configWarnings.join(' | '))
  }

  const rag = new RAGServer(config)
  await rag.initialize()

  const idleExitMs = parseIdleExitMs(process.env['RAG_IDLE_EXIT_SEC'])
  // Stop accepting before exiting: a connection that arrives meanwhile stays
  // queued on the systemd-held socket and starts the next instance.
  const tracker = createIdleTracker(idleExitMs, () => {
    console.error(`No connection for ${idleExitMs / 1000} s: exiting to free memory`)
    listener.close()
    process.exit(0)
  })
  const listener = createServer((socket) => serveConnection(rag, socket, tracker))
  if (activatedBySystemd(process.env)) {
    listener.listen({ fd: SD_LISTEN_FDS_START })
    console.error(`Shared RAG server (read-only) on socket-activated fd for ${config.dbPath}`)
    return
  }
  const socketPath = process.env['RAG_SOCKET']
  if (!socketPath) {
    throw new Error('RAG_SOCKET is not set and the process was not socket-activated.')
  }
  try {
    unlinkSync(socketPath)
  } catch {
    // No stale socket to remove.
  }
  listener.listen(socketPath, () => {
    chmodSync(socketPath, 0o600)
    console.error(`Shared RAG server (read-only) on ${socketPath} for ${config.dbPath}`)
  })
  // Closing the listener removes the socket file, so a client never reaches a
  // stale path after shutdown.
  for (const signal of ['SIGTERM', 'SIGINT'] as const) {
    process.on(signal, () => {
      listener.close()
      process.exit(0)
    })
  }
}

startSharedServer().catch((error: unknown) => {
  console.error('Failed to start shared RAG server:', error)
  process.exit(1)
})
