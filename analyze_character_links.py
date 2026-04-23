"""Analyze wiki body structure for character-copyright relationships."""

import re
import pyarrow.parquet as pq
import pandas as pd
from collections import defaultdict

df = pq.read_table("D:/Coding/Python/dan_emb/danbooru_wiki.parquet").to_pandas()

# Focus on copyright entries
copyright_df = df[df["category"] == "copyright"].copy()
print(f"Copyright entries: {len(copyright_df)}")

# Find entries with character/member sections
SECTION_PATTERNS = [
    r'h[1-6]\.\s*(?:characters?|members?|main characters?|cast|voice actors?|seiyuu)',
]

def extract_section_tags(body: str) -> list[str]:
    """Extract [[tag]] links from character/member list sections."""
    if not body or not body.strip():
        return []

    # Split into sections by headings
    sections = re.split(r'(?=^h[1-6]\.\s*)', body, flags=re.MULTILINE)

    tags = []
    for section in sections:
        heading_match = re.match(r'^h[1-6]\.\s*(.+?)(?:\n|$)', section)
        if heading_match:
            heading = heading_match.group(1).strip().lower()
            # Check if this is a character/member section
            if any(kw in heading for kw in ['character', 'member', 'cast', 'seiyuu', 'voice actor']):
                # Extract all [[tag]] links
                found = re.findall(r'\[\[([^\]]+)\]\]', section)
                # Handle [[display|tag]] format — take the last part (actual tag)
                cleaned = []
                for f in found:
                    if '|' in f:
                        parts = f.split('|')
                        # Use the part after | if it exists, else the part before
                        tag = parts[-1] if parts[-1] else parts[0]
                    else:
                        tag = f
                    tag = tag.strip().lower()
                    if tag:
                        cleaned.append(tag)
                tags.extend(cleaned)

    return tags

# Extract character tags from all copyright entries
copyright_to_chars = {}
for _, row in copyright_df.iterrows():
    tags = extract_section_tags(row["body"])
    if tags:
        copyright_to_chars[row["tag"]] = tags

print(f"\nCopyright entries with character lists: {len(copyright_to_chars)}")

# Show some examples
print("\n=== Examples ===")
count = 0
for cp, chars in list(copyright_to_chars.items()):
    if len(chars) >= 5:
        print(f"\n{cp} ({len(chars)} characters):")
        print(f"  {chars[:15]}{'...' if len(chars) > 15 else ''}")
        count += 1
        if count >= 10:
            break

# Check how many character tags in our dataset are covered
all_char_tags = set()
for chars in copyright_to_chars.values():
    all_char_tags.update(chars)

character_df = df[df["category"] == "character"]
our_char_tags = set(character_df["tag"].str.lower())
overlap = all_char_tags & our_char_tags
print(f"\n=== Coverage ===")
print(f"Total character tags extracted from copyright wikis: {len(all_char_tags)}")
print(f"Character tags in our dataset: {len(our_char_tags)}")
print(f"Overlap: {len(overlap)} ({100*len(overlap)/len(our_char_tags):.1f}%)")

# Also check: how many character entries mention a copyright in their body?
print("\n=== Character body → Copyright references ===")
char_with_copyright = 0
char_copyright_map = defaultdict(list)
for _, row in character_df.iterrows():
    if not row["body"] or not row["body"].strip():
        continue
    # Find [[copyright_tag]] references in body
    links = re.findall(r'\[\[([^\]]+)\]\]', row["body"])
    for link in links:
        if '|' in link:
            link = link.split('|')[-1] or link.split('|')[0]
        link_lower = link.strip().lower()
        # Check if this link is a known copyright tag
        if link_lower in set(copyright_df["tag"].str.lower()):
            char_copyright_map[row["tag"]].append(link)
            char_with_copyright += 1
            break  # just count once per character

print(f"Characters with copyright references in body: {char_with_copyright}/{len(character_df)} ({100*char_with_copyright/len(character_df):.1f}%)")

# Combine both sources
print("\n=== Combined mapping stats ===")
# Forward: copyright → characters
# Reverse: character → copyrights (from both sources)
char_to_copyrights = defaultdict(set)

# From copyright wiki character lists
for cp, chars in copyright_to_chars.items():
    for char in chars:
        char_to_copyrights[char].add(cp)

# From character body references
for char_tag, cp_tags in char_copyright_map.items():
    for cp in cp_tags:
        char_to_copyrights[char_tag.lower()].add(cp.lower())

print(f"Characters mapped to at least one copyright: {len(char_to_copyrights)}")
if char_to_copyrights:
    lens = [len(v) for v in char_to_copyrights.values()]
    print(f"Avg copyrights per character: {sum(lens)/len(lens):.1f}")
