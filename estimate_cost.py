import pyarrow.parquet as pq

df = pq.read_table("D:/Coding/Python/dan_emb/danbooru_wiki.parquet").to_pandas()

# Filter non-empty body
valid = df[df["body"].notna() & (df["body"].str.strip() != "")]
print(f"Valid entries: {len(valid)}")

# Rough token estimate: ~4 chars per token for English, ~1.5 chars per token for Chinese
# Danbooru wiki is mostly English, use ~3.5 chars/token as rough estimate
chars = valid["body"].str.len()
print(f"\nBody length stats (chars):")
print(f"  Mean: {chars.mean():.0f}")
print(f"  Median: {chars.median():.0f}")
print(f"  P25: {chars.quantile(0.25):.0f}")
print(f"  P75: {chars.quantile(0.75):.0f}")
print(f"  P95: {chars.quantile(0.95):.0f}")
print(f"  Max: {chars.max()}")

total_chars = chars.sum()
# Also add title and other_names
title_chars = valid["title"].str.len().sum()

# Rough: ~3.5 chars/token for English-dominant text
est_tokens_per_entry = (chars.mean() + valid["title"].str.len().mean()) / 3.5
total_tokens = (total_chars + title_chars) / 3.5

print(f"\nTotal chars (body): {total_chars:,.0f}")
print(f"Total chars (title): {title_chars:,.0f}")
print(f"Estimated total tokens: {total_tokens:,.0f}")
print(f"Estimated tokens per entry: {est_tokens_per_entry:.0f}")

cost_per_million = 0.28  # yuan
cost = total_tokens / 1_000_000 * cost_per_million
print(f"\nEstimated cost: {cost:.2f} yuan")
