"""Build character-copyright binding table from wiki data."""

import re
import json
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import pandas as pd
from tqdm import tqdm

PARQUET_PATH = "danbooru_wiki.parquet"
OUTPUT_PATH = Path("embedding_index/char_copyright.json")

# ── Extract character tags from copyright wiki ─────────────────────────────

def extract_char_tags(body: str) -> list[str]:
    """Extract [[tag]] links from character/member list sections in copyright wiki."""
    if not body or not body.strip():
        return []

    sections = re.split(r'(?=^h[1-6]\.\s*)', body, flags=re.MULTILINE)
    tags = []
    for section in sections:
        heading_match = re.match(r'^h[1-6]\.\s*(.+?)(?:\n|$)', section)
        if heading_match:
            heading = heading_match.group(1).strip().lower()
            if any(kw in heading for kw in ['character', 'member', 'cast', 'seiyuu', 'voice actor']):
                for f in re.findall(r'\[\[([^\]]+)\]\]', section):
                    parts = f.split('|')
                    tag = parts[1].strip() if len(parts) == 2 and parts[1].strip() else parts[0].strip()
                    if tag:
                        tags.append(tag)
    return tags


def main():
    print("Loading data...")
    df = pq.read_table(PARQUET_PATH).to_pandas()

    copyright_df = df[df["category"] == "copyright"]
    character_df = df[df["category"] == "character"]
    print(f"  Copyright: {len(copyright_df)}, Character: {len(character_df)}")

    # Build lookup tables
    print("Building lookup tables...")
    char_tag_set = set(character_df["tag"])
    char_title_to_tag = dict(zip(character_df["title"], character_df["tag"]))
    cp_tag_set = set(copyright_df["tag"])
    cp_title_to_tag = dict(zip(copyright_df["title"], copyright_df["tag"]))

    def normalize(s):
        return s.lower().replace(" ", "_").replace("-", "_").strip()

    char_norm_to_tag = {}
    for tag in char_tag_set:
        char_norm_to_tag[normalize(tag)] = tag
    for title, tag in char_title_to_tag.items():
        char_norm_to_tag[normalize(title)] = tag

    cp_norm_to_tag = {}
    for tag in cp_tag_set:
        cp_norm_to_tag[normalize(tag)] = tag
    for title, tag in cp_title_to_tag.items():
        cp_norm_to_tag[normalize(title)] = tag

    def resolve_char(name):
        """Try to resolve a name to a character tag."""
        if name in char_tag_set:
            return name
        if name in char_title_to_tag:
            return char_title_to_tag[name]
        n = normalize(name)
        return char_norm_to_tag.get(n)

    def resolve_cp(name):
        """Try to resolve a name to a copyright tag."""
        if name in cp_tag_set:
            return name
        if name in cp_title_to_tag:
            return cp_title_to_tag[name]
        n = normalize(name)
        return cp_norm_to_tag.get(n)

    # ── Step 1: Copyright wiki → character list ────────────────────────────
    print("\nStep 1: Extracting characters from copyright wikis...")
    char_to_cps = defaultdict(set)
    cp_with_chars = 0
    total_chars_extracted = 0

    for _, row in tqdm(copyright_df.iterrows(), total=len(copyright_df), desc="  Copyright wikis"):
        tags = extract_char_tags(row["body"])
        if tags:
            cp_with_chars += 1
            cp_tag = row["tag"]
            for ch_name in tags:
                ch_tag = resolve_char(ch_name)
                if ch_tag:
                    char_to_cps[ch_tag].add(cp_tag)
                    total_chars_extracted += 1

    print(f"  Copyrights with character lists: {cp_with_chars}")
    print(f"  Characters resolved: {len(char_to_cps)}")

    # ── Step 2: Character wiki → copyright references in body ──────────────
    print("\nStep 2: Extracting copyright refs from character wikis...")
    body_linked = 0
    valid_chars = character_df[character_df["body"].notna() & (character_df["body"].str.strip() != "")]

    for _, row in tqdm(valid_chars.iterrows(), total=len(valid_chars), desc="  Character wikis"):
        links = re.findall(r'\[\[([^\]]+)\]\]', row["body"])
        for link in links:
            parts = link.split("|")
            name = parts[1].strip() if len(parts) == 2 and parts[1].strip() else parts[0].strip()
            cp_tag = resolve_cp(name)
            if cp_tag:
                char_to_cps[row["tag"]].add(cp_tag)
                body_linked += 1
                break

    print(f"  Characters with body copyright refs: {body_linked}")

    # ── Summary & save ─────────────────────────────────────────────────────
    print(f"\n=== Result ===")
    print(f"  Total characters with copyright: {len(char_to_cps)}/{len(character_df)} ({100*len(char_to_cps)/len(character_df):.1f}%)")

    # Convert sets to lists for JSON
    result = {k: sorted(v) for k, v in sorted(char_to_cps.items())}

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  Saved to {OUTPUT_PATH} ({len(result)} entries)")

    # Show examples
    print("\n=== Examples ===")
    examples = sorted(result.items(), key=lambda x: -len(x[1]))[:10]
    for char, cps in examples:
        print(f"  {char} → {cps}")


if __name__ == "__main__":
    main()
