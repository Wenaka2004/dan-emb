"""Danbooru DText markup cleaner."""

import re
import numpy as np


def clean_dtext(text: str) -> str:
    """Convert Danbooru DText markup to plain text."""
    if not text:
        return ""

    # DText links: "display text":url → display text (must come before bare URL strip)
    text = re.sub(r'"([^"]+)":https?://\S+', r'\1', text)

    # Markdown links: [text](url) → text
    text = re.sub(r'\[([^\]]+)\]\(https?://\S+\)', r'\1', text)

    # DText bracket links: "text":[url] → text
    text = re.sub(r'"([^"]+)":\[[^\]]*\]', r'\1', text)

    # Bare URLs
    text = re.sub(r'https?://\S+', '', text)

    # Leftover bracket-colon artifacts from DText links: "text":[
    text = re.sub(r':\[', '', text)

    # Wiki links: [[display|tag]] → display, [[tag]] → tag
    text = re.sub(r'\[\[([^|\]]+)\|[^\]]*\]\]', r'\1', text)
    text = re.sub(r'\[\[([^\]]+)\]\]', r'\1', text)

    # Inline formatting
    for tag in ('b', 'i', 'u', 's', 'sup', 'sub', 'code', 'tn'):
        text = re.sub(rf'\[/?{tag}\]', '', text)

    # Headings: h4. Title → Title
    text = re.sub(r'^h[1-6]\.\s*', '', text, flags=re.MULTILINE)

    # List items: * item → item
    text = re.sub(r'^\*\s+', '', text, flags=re.MULTILINE)

    # Numbered lists: # item → item
    text = re.sub(r'^#\s+', '', text, flags=re.MULTILINE)

    # Quote blocks
    text = re.sub(r'^>\s*', '', text, flags=re.MULTILINE)

    # Table separators
    text = re.sub(r'\|', ' ', text)

    # Collapse whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = text.strip()

    return text


# Sections that are pure lists — not useful for semantic retrieval
_DROP_SECTIONS = frozenset({
    'external links', 'see also', 'references', 'members',
    'characters', 'staff', 'voice actors', 'cast', 'seiyuu',
    'gallery', 'images', 'tracks', 'track list', 'track listing',
    'tracklist', 'discography', 'episode list', 'episodes',
    'chapter list', 'chapters', 'versions', 'navigation',
    'song list', 'songs', 'cover list', 'covers',
})


def select_useful_sections(body: str) -> str:
    """Keep description-rich sections, drop list-heavy sections."""
    if not body:
        return ""

    # Split by heading boundaries
    parts = re.split(r'(?=^h[1-6]\.\s*)', body, flags=re.MULTILINE)

    result = []
    for part in parts:
        heading_match = re.match(r'^h[1-6]\.\s*(.+?)(?:\n|$)', part)
        if heading_match:
            heading = heading_match.group(1).strip().lower().rstrip(':')
            if heading in _DROP_SECTIONS:
                continue

        # Skip sections that are >70% list items (heuristic)
        lines = [l for l in part.strip().split('\n') if l.strip()]
        if len(lines) > 5:
            list_lines = sum(
                1 for l in lines
                if l.strip().startswith('* ') or l.strip().startswith('- ')
            )
            if list_lines / len(lines) > 0.7:
                continue

        result.append(part)

    return '\n\n'.join(result)


def build_embedding_text(row) -> str:
    """Build the text to embed for one wiki entry.

    Format:
        Tag: <tag>
        Also known as: <other_names>
        Category: <category>
        <cleaned body>
    """
    parts = []

    tag = row['tag']
    parts.append(f"Tag: {tag}")

    other_names = row.get('other_names')
    if isinstance(other_names, np.ndarray) and len(other_names) > 0:
        names = ', '.join(str(n) for n in other_names if n)
        if names:
            parts.append(f"Also known as: {names}")

    category = row.get('category', '')
    if category:
        parts.append(f"Category: {category}")

    body = row.get('body', '')
    if body and body.strip():
        body = select_useful_sections(body)
        body = clean_dtext(body)
        if body:
            parts.append(body)

    return '\n'.join(parts)
