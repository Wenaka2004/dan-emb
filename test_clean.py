"""Quick test: verify data cleaning and embedding text quality."""

import pyarrow.parquet as pq
from dtext import build_embedding_text, clean_dtext, select_useful_sections

df = pq.read_table("D:/Coding/Python/dan_emb/danbooru_wiki.parquet").to_pandas()
valid = df[df["body"].notna() & (df["body"].str.strip() != "") & (~df["is_deleted"])]

print("=== DText cleaning examples ===\n")

# Test on a few entries with different characteristics
test_indices = []
# Find a general tag with wiki links
for i, row in valid.iterrows():
    if row["category"] == "general" and "[[" in str(row["body"]) and len(str(row["body"])) > 100:
        test_indices.append(i)
        break
# Find a copyright with h4 sections
for i, row in valid.iterrows():
    if row["category"] == "copyright" and "h4." in str(row["body"]) and len(str(row["body"])) > 500:
        test_indices.append(i)
        break
# Find a character with other_names
for i, row in valid.iterrows():
    if row["category"] == "character" and len(row["other_names"]) > 0 and len(str(row["body"])) > 200:
        test_indices.append(i)
        break

for idx in test_indices:
    row = valid.loc[idx]
    print(f"--- [{row['category']}] {row['tag']} ---")
    print(f"Raw body ({len(str(row['body']))} chars):")
    print(str(row['body'])[:300])
    print()
    print(f"Other names: {row['other_names']}")
    print()

    # Step by step
    filtered = select_useful_sections(row['body'])
    cleaned = clean_dtext(filtered)
    full_text = build_embedding_text(row)

    print(f"After section filter ({len(filtered)} chars):")
    print(filtered[:300])
    print()
    print(f"After DText clean ({len(cleaned)} chars):")
    print(cleaned[:300])
    print()
    print(f"Final embedding text ({len(full_text)} chars):")
    print(full_text[:500])
    print("\n" + "="*60 + "\n")

# Stats after cleaning
print("\n=== Embedding text length stats ===")
import numpy as np
mask = valid["body"].notna() & (valid["body"].str.strip() != "") & (~valid["is_deleted"])
subset = valid[mask].copy()
texts = subset.apply(build_embedding_text, axis=1)
lengths = texts.str.len()
print(f"Total entries: {len(texts)}")
print(f"Mean: {lengths.mean():.0f} chars")
print(f"Median: {lengths.median():.0f}")
print(f"P95: {lengths.quantile(0.95):.0f}")
print(f"P99: {lengths.quantile(0.99):.0f}")
print(f"Max: {lengths.max()}")

# How many would be too short
min_len = len("Tag: x\nCategory: x") + 10
too_short = (lengths < min_len).sum()
print(f"\nToo short (<{min_len} chars): {too_short} ({100*too_short/len(texts):.1f}%)")

# By category
for cat in subset["category"].unique():
    cat_texts = texts[subset["category"] == cat]
    cat_lens = cat_texts.str.len()
    print(f"  {cat}: n={len(cat_texts)}, mean={cat_lens.mean():.0f}, median={cat_lens.median():.0f}, p95={cat_lens.quantile(0.95):.0f}")
