"""`index/render.py` — the two body renderings, `format_semantic()` and
`format_lexical()`. See development-plans/PLAN.md §5 "Two body renderings"
and DESIGN-NOTES.md §9 for the full rationale; this module implements it.

The same raw message list is rendered TWICE, by two independent functions
with independent format versions (`meta.semantic_format_version` /
`meta.lexical_format_version`, written by build.py):

    format_semantic() -> embedder + reranker input (512-token cap, roles kept,
                          URLs collapsed to domain, no place names)
    format_lexical()  -> chunks_fts.body (no cap, roles stripped, URLs
                          verbatim, place names inline in the media placeholder)

Changing `format_lexical` costs an FTS rebuild (minutes). Changing
`format_semantic` costs a full re-embed (hours). Keep them independent.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence

from seaglass.imessage.source import AttachmentRow, Message

MAX_SEMANTIC_TOKENS = 512
# Approximate whitespace/punctuation tokenizer. This is a stand-in for the
# real embedder tokenizer (wired in Phase 3's embed.py); it exists so the
# 512-token cap can be enforced and measured *now*, in the chunker, without
# depending on MLX being loaded. Re-validate the cap against the real
# tokenizer once embed.py exists -- PLAN.md §5 warns: "if it fires on more
# than a few percent of chunks, the chunk size target is wrong."
_TOKEN_RE = re.compile(r"\S+")
_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_DOMAIN_RE = re.compile(r"https?://(?:www\.)?([^/\s]+)", re.IGNORECASE)


def approx_token_count(text: str) -> int:
    return len(_TOKEN_RE.findall(text))


def _speaker_labels(messages: Sequence[Message]) -> Dict[Optional[str], str]:
    """Stable per-chunk speaker labels. 1:1 chats get "Them" (a single other
    participant); group chats get "A"/"B"/"C"... in order of first
    appearance -- never the contact's real name (DESIGN-NOTES.md §9,
    "Names and dates inside embed_text": kept exception is speaker ROLE,
    not identity).
    """
    distinct_handles: List[Optional[str]] = []
    for msg in messages:
        if msg.is_from_me:
            continue
        if msg.handle not in distinct_handles:
            distinct_handles.append(msg.handle)

    if len(distinct_handles) <= 1:
        return {handle: "Them" for handle in distinct_handles}

    labels: Dict[Optional[str], str] = {}
    next_letter = ord("A")
    for handle in distinct_handles:
        labels[handle] = chr(next_letter)
        next_letter += 1
    return labels


def _label_for(msg: Message, speaker_labels: Dict[Optional[str], str]) -> str:
    if msg.is_from_me:
        return "Me"
    return speaker_labels.get(msg.handle, "Them")


def _collapse_urls_to_domain(text: str) -> str:
    def _replace(match: "re.Match[str]") -> str:
        domain_match = _DOMAIN_RE.match(match.group(0))
        domain = domain_match.group(1) if domain_match else "link"
        return f"[link:{domain}]"

    return _URL_RE.sub(_replace, text)


def _media_placeholder_semantic() -> str:
    # Bare -- no place names, no filenames. Geography is not what the
    # embedding should encode (PLAN.md §5, "Why place names must be text").
    return "[attachment]"


def _media_placeholder_lexical(
    attachments: List[AttachmentRow], places_by_attachment: Dict[int, str]
) -> str:
    """The media placeholder is the single carrier for all attachment-derived
    lexical signal: place name (if geotagged) and descriptive filename,
    inline at the position the attachment occurs (PLAN.md §5).
    """
    parts = []
    for attachment in attachments:
        place = places_by_attachment.get(attachment.attachment_id)
        if place:
            parts.append(place)
        if attachment.filename:
            # Bare device filenames (IMG_1234.HEIC) are noise but a
            # descriptive one is signal; include verbatim minus extension
            # noise is a later refinement -- keep simple for now.
            parts.append(attachment.filename)
    inner = " ".join(parts).strip()
    return f"[attachment {inner}]" if inner else "[attachment]"


def _middle_drop(tokens: List[str], max_tokens: int) -> List[str]:
    """When format_semantic exceeds the cap, drop from the MIDDLE -- the
    opening establishes the topic, the closing usually carries the
    resolution (PLAN.md §5).
    """
    if len(tokens) <= max_tokens:
        return tokens
    keep_each_side = max_tokens // 2
    return tokens[:keep_each_side] + tokens[-(max_tokens - keep_each_side):]


def format_semantic(
    messages: Sequence[Message],
    max_tokens: int = MAX_SEMANTIC_TOKENS,
) -> str:
    """Render for the embedder/reranker. Role labels kept, URLs collapsed to
    domain-only, media reduced to a bare placeholder, place names excluded,
    hard-capped at `max_tokens` with middle-drop truncation.
    """
    speaker_labels = _speaker_labels(messages)
    lines: List[str] = []
    for msg in messages:
        label = _label_for(msg, speaker_labels)
        body_parts = []
        if msg.text:
            body_parts.append(_collapse_urls_to_domain(msg.text))
        if msg.has_attachment:
            body_parts.append(_media_placeholder_semantic())
        if not body_parts:
            continue
        lines.append(f"{label}: {' '.join(body_parts)}")
    rendered = "\n".join(lines)
    tokens = _TOKEN_RE.findall(rendered)
    if len(tokens) <= max_tokens:
        return rendered
    # Middle-drop at the character level via token spans, preserving line
    # structure is not required -- reconstruct from the retained tokens.
    kept = _middle_drop(tokens, max_tokens)
    return " ".join(kept)


def format_lexical(
    messages: Sequence[Message],
    attachments_by_msg: Optional[Dict[int, List[AttachmentRow]]] = None,
    places_by_attachment: Optional[Dict[int, str]] = None,
) -> str:
    """Render for FTS5 indexing (never stored -- see PLAN.md §5). Role
    labels stripped, URLs verbatim, no length cap, place names and
    descriptive filenames inline inside the media placeholder at the
    position the attachment occurs.
    """
    attachments_by_msg = attachments_by_msg or {}
    places_by_attachment = places_by_attachment or {}
    lines: List[str] = []
    for msg in messages:
        body_parts = []
        if msg.text:
            body_parts.append(msg.text)  # verbatim -- URLs untouched
        attachments = attachments_by_msg.get(msg.rowid, [])
        if msg.has_attachment or attachments:
            body_parts.append(_media_placeholder_lexical(attachments, places_by_attachment))
        if not body_parts:
            continue
        lines.append(" ".join(body_parts))
    return "\n".join(lines)
