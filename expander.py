"""Tag expansion service: two-stage RAG + LLM with streaming and step callbacks."""

import re
import time
from typing import Generator

import numpy as np
from openai import OpenAI

from config import (
    SILICONFLOW_API_KEY, SILICONFLOW_BASE_URL, LLM_MODEL,
    EMBEDDING_MODEL, DEFAULT_TOP_K, DEFAULT_TEMPERATURE,
)
from rag import DanbooruRAG

# ── LLM Client ──────────────────────────────────────────────────────────────

client = OpenAI(api_key=SILICONFLOW_API_KEY, base_url=SILICONFLOW_BASE_URL)

# ── Prompts ─────────────────────────────────────────────────────────────────

TRANSLATE_PROMPT = """\
Translate the following text to English. This is a description for anime artwork tags.
- If already in English, return as-is.
- If it contains Danbooru tags (underscored words), preserve them exactly.
- Output ONLY the English translation, nothing else.
"""

STAGE1_SYSTEM = """\
You are a Danbooru tag expert. You expand a visual description into a COMPREHENSIVE set of Danbooru tags.

You must be THOROUGH. A good Danbooru tag set for a single character image typically has 20-40 tags covering:
- Meta: image quality, source (e.g. original, photorealistic)
- Subject count: 1girl, solo, etc.
- Body: hair color, hair length, hair style, eye color, skin tone, breast size, body features
- Expression: smile, blush, closed eyes, etc.
- Pose: standing, sitting, arms up, looking at viewer, etc.
- Clothing: specific garment types, colors, accessories
- Setting: background, location, lighting, time of day
- Other: artistic style, composition

Examples of good expansions:

Input: "a girl with silver hair and red eyes in school uniform"
Output: 1girl, solo, original, long hair, silver hair, red eyes, school uniform, white shirt, pleated skirt, short sleeves, looking at viewer, smile, standing, full body, simple background, daytime

Input: "dragon girl with horns and tail"
Output: 1girl, solo, original, dragon girl, dragon horns, dragon tail, horns, tail, long hair, green eyes, fantasy, scales, standing, looking at viewer, simple background

Rules:
1. Output ONLY comma-separated Danbooru tags, no explanation.
2. Use EXACT tag names from the reference when available.
3. Follow Danbooru ordering: meta → character count → body → expression → pose → clothing → setting → other.
4. Add commonly co-occurring tags even if not explicitly described (e.g. "school uniform" implies "shirt", "skirt"; "1girl" implies "solo").
5. For characters from the reference, include the character tag and copyright tag.
6. Do NOT add tags that contradict the description.
"""

STAGE2_SYSTEM = """\
You are a Danbooru tag expert. You are given an initial tag set and additional reference tags found via similarity search.
Your job is to SUPPLEMENT the initial tags with relevant tags from the reference.

Rules:
1. Output the COMPLETE expanded tag set (initial + supplemented), comma-separated.
2. Only add tags from the reference that are genuinely relevant.
3. Remove duplicates.
4. Keep Danbooru ordering: meta → character count → body → expression → pose → clothing → setting → copyright/character/artist.
5. Do NOT remove tags from the initial set unless they conflict.
6. Output ONLY tags, no explanation.
"""


# ── Helpers ─────────────────────────────────────────────────────────────────

def _llm_call(system: str, user: str, temperature: float = 0.3, max_tokens: int = 512) -> str:
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


def _llm_stream(system: str, user: str, temperature: float = 0.3, max_tokens: int = 512) -> Generator[str, None, None]:
    stream = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def _parse_tags(raw: str) -> list[str]:
    return [t.strip() for t in raw.split(",") if t.strip()]


def _is_english(text: str) -> bool:
    return bool(re.match(r'^[a-zA-Z0-9_\s,()\-\.]+$', text))


def build_context(rag_results: list[dict], max_entries: int = 20) -> str:
    lines = ["<reference>", "Relevant Danbooru tags and descriptions:", ""]
    for r in rag_results[:max_entries]:
        tag_line = f"[{r['category']}] {r['tag']}"
        if r["copyrights"]:
            tag_line += f" (from: {', '.join(r['copyrights'][:3])})"
        lines.append(tag_line)
        wiki = r["wiki_text"]
        if len(wiki) > 200:
            wiki = wiki[:200] + "..."
        lines.append(f"  {wiki}")
        lines.append("")
    lines.append("</reference>")
    return "\n".join(lines)


# ── Non-streaming API (for FastAPI endpoint) ────────────────────────────────

def expand_tags(
    user_input: str,
    rag: DanbooruRAG,
    top_k: int = DEFAULT_TOP_K,
    category_filter: list[str] | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    two_stage: bool = True,
) -> dict:
    t0 = time.perf_counter()
    steps = {}

    t_en = time.perf_counter()
    english_input = _llm_call(TRANSLATE_PROMPT, user_input, temperature=0.0, max_tokens=256) if not _is_english(user_input) else user_input
    steps["translate_ms"] = (time.perf_counter() - t_en) * 1000

    t1 = time.perf_counter()
    rag_results = rag.search(english_input, top_k=top_k, expand_copyright=True, category_filter=category_filter)
    context = build_context(rag_results)
    stage1_prompt = f"{context}\n\nDescription: {english_input}\n\nTags:"
    stage1_raw = _llm_call(STAGE1_SYSTEM, stage1_prompt, temperature=temperature)
    stage1_tags = _parse_tags(stage1_raw)
    steps["stage1_ms"] = (time.perf_counter() - t1) * 1000

    if not two_stage or not stage1_tags:
        return {
            "input": user_input, "english_input": english_input, "tags": stage1_tags,
            "tag_count": len(stage1_tags), "rag_results": rag_results,
            "raw_output": stage1_raw, "latency_ms": (time.perf_counter() - t0) * 1000, "steps": steps,
        }

    t2 = time.perf_counter()
    supplement_results = []
    key_tags = [t for t in stage1_tags if not t.startswith(("1girl", "1boy", "solo", "original"))][:3]
    for tag in key_tags:
        try:
            supplement_results.extend(rag.search(tag, top_k=5, expand_copyright=True))
        except Exception:
            pass
    seen = {r["tag"] for r in rag_results}
    unique_supplement = [r for r in supplement_results if r["tag"] not in seen][:10]

    if unique_supplement:
        supplement_context = build_context(unique_supplement, max_entries=10)
        stage2_prompt = (
            f"Initial tags: {', '.join(stage1_tags)}\n\n"
            f"{supplement_context}\n\nDescription: {english_input}\n\nExpanded tags:"
        )
        stage2_raw = _llm_call(STAGE2_SYSTEM, stage2_prompt, temperature=max(temperature - 0.1, 0.1))
        final_tags = _parse_tags(stage2_raw)
    else:
        final_tags = stage1_tags

    steps["stage2_ms"] = (time.perf_counter() - t2) * 1000
    return {
        "input": user_input, "english_input": english_input, "tags": final_tags,
        "tag_count": len(final_tags), "rag_results": rag_results,
        "supplement_results": unique_supplement,
        "raw_output": ", ".join(final_tags),
        "latency_ms": (time.perf_counter() - t0) * 1000, "steps": steps,
    }


# ── Streaming API (for WebUI) ───────────────────────────────────────────────

def expand_tags_streaming(
    user_input: str,
    rag: DanbooruRAG,
    top_k: int = DEFAULT_TOP_K,
    category_filter: list[str] | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    two_stage: bool = True,
) -> Generator[dict, None, None]:
    t0 = time.perf_counter()

    if not _is_english(user_input):
        yield {"type": "status", "step": "translate", "message": "Translating to English..."}
        english_input = _llm_call(TRANSLATE_PROMPT, user_input, temperature=0.0, max_tokens=256)
        yield {"type": "status", "step": "translate", "message": f"English: {english_input}"}
    else:
        english_input = user_input
        yield {"type": "status", "step": "translate", "message": f"Input (English): {english_input}"}

    yield {"type": "status", "step": "rag1", "message": "Searching vector index..."}
    rag_results = rag.search(english_input, top_k=top_k, expand_copyright=True, category_filter=category_filter)
    rag_summary = [
        {"tag": r["tag"], "category": r["category"], "score": round(r["score"], 4), "copyrights": r.get("copyrights", [])}
        for r in rag_results[:5]
    ]
    yield {"type": "rag", "results": rag_summary}
    yield {"type": "status", "step": "rag1", "message": f"Found {len(rag_results)} relevant tags"}

    yield {"type": "status", "step": "llm1", "message": "Stage 1: Expanding tags..."}
    context = build_context(rag_results)
    stage1_prompt = f"{context}\n\nDescription: {english_input}\n\nTags:"
    stage1_text = ""
    for chunk in _llm_stream(STAGE1_SYSTEM, stage1_prompt, temperature=temperature):
        stage1_text += chunk
        yield {"type": "stream", "stage": 1, "text": stage1_text}
    stage1_tags = _parse_tags(stage1_text)
    yield {"type": "status", "step": "llm1", "message": f"Stage 1 done: {len(stage1_tags)} tags"}

    if not two_stage or not stage1_tags:
        yield {"type": "done", "tags": stage1_tags, "tag_count": len(stage1_tags), "latency_ms": (time.perf_counter() - t0) * 1000}
        return

    yield {"type": "status", "step": "rag2", "message": "Searching supplementary tags..."}
    supplement_results = []
    key_tags = [t for t in stage1_tags if not t.startswith(("1girl", "1boy", "solo", "original"))][:3]
    for tag in key_tags:
        try:
            supplement_results.extend(rag.search(tag, top_k=5, expand_copyright=True))
        except Exception:
            pass
    seen = {r["tag"] for r in rag_results}
    unique_supplement = [r for r in supplement_results if r["tag"] not in seen][:10]
    supplement_summary = [
        {"tag": r["tag"], "category": r["category"], "score": round(r["score"], 4), "copyrights": r.get("copyrights", [])}
        for r in unique_supplement[:5]
    ]
    yield {"type": "rag", "results": supplement_summary}
    yield {"type": "status", "step": "rag2", "message": f"Found {len(unique_supplement)} supplementary tags"}

    if unique_supplement:
        yield {"type": "status", "step": "llm2", "message": "Stage 2: Refining tags..."}
        supplement_context = build_context(unique_supplement, max_entries=10)
        stage2_prompt = (
            f"Initial tags: {', '.join(stage1_tags)}\n\n"
            f"{supplement_context}\n\nDescription: {english_input}\n\nExpanded tags:"
        )
        stage2_text = ""
        for chunk in _llm_stream(STAGE2_SYSTEM, stage2_prompt, temperature=max(temperature - 0.1, 0.1)):
            stage2_text += chunk
            yield {"type": "stream", "stage": 2, "text": stage2_text}
        final_tags = _parse_tags(stage2_text)
    else:
        yield {"type": "status", "step": "llm2", "message": "No supplementary tags found, using stage 1 result"}
        final_tags = stage1_tags

    yield {"type": "done", "tags": final_tags, "tag_count": len(final_tags), "latency_ms": (time.perf_counter() - t0) * 1000}
