---
name: imessage
description: Search and inspect the user's local iMessage history through the read-only Seaglass MCP server.
---

Use only when the user explicitly asks to search or inspect their iMessage
history. Seaglass owns this capability; do not look for an iMessage adapter in
the calling harness.

1. Call `index_status` before relying on retrieval. If the index is stale and
   the request depends on recent messages, call `sync_index` with `wait=true`,
   then check `index_status` again. If synchronization cannot start, times out,
   or leaves the index stale, explain that recent messages cannot be searched
   reliably and do not present stale results as a complete answer.
2. Call `search_messages` with the user's natural-language query. Use
   `person` only for conversation participants, not people merely mentioned in
   message text.
3. Use `get_conversation` only to inspect more context around a conversation
   already returned by `search_messages`.
4. Cite returned message ids when summarizing or drawing conclusions.

Search is ranked recall, not exhaustive enumeration or an exact counting
interface. Say so when the request asks for every occurrence or a total.

Seaglass is read-only. It does not draft, send, delete, react to, or otherwise
modify messages. Never imply that this skill authorizes an outbound action.
