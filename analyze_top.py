import pyarrow.parquet as pq
import pandas as pd

df = pq.read_table("D:/Coding/Python/dan_emb/danbooru_wiki.parquet").to_pandas()
valid = df[df["body"].notna() & (df["body"].str.strip() != "")]

# Longest bodies
valid_sorted = valid.copy()
valid_sorted["body_len"] = valid_sorted["body"].str.len()
longest = valid_sorted.nlargest(5, "body_len")
print("=== Longest bodies ===")
for _, row in longest.iterrows():
    print(f"  title={row['title']}, len={row['body_len']}, category={row['category']}")
    print(f"  body preview: {row['body'][:300]}")
    print()

# Category-wise body stats
print("=== Body length by category ===")
for cat in df["category"].unique():
    subset = valid[valid["category"] == cat]
    if len(subset) == 0:
        continue
    lens = subset["body"].str.len()
    print(f"  {cat}: count={len(subset)}, mean={lens.mean():.0f}, median={lens.median():.0f}, p95={lens.quantile(0.95):.0f}")

# Short bodies
print(f"\n=== Very short bodies (<=50 chars) ===")
short = valid[valid["body"].str.len() <= 50]
print(f"  Count: {len(short)}")
for i in range(min(10, len(short))):
    row = short.iloc[i]
    print(f"  [{row['category']}] {row['title']}: '{row['body']}'")
