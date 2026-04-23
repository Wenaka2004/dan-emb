import pyarrow.parquet as pq
import pandas as pd

df = pq.read_table("D:/Coding/Python/dan_emb/danbooru_wiki.parquet").to_pandas()

print("=== Schema & Types ===")
print(df.dtypes)
print()

print("=== Category Distribution ===")
print(df["category"].value_counts())
print()

# is_deleted
print("=== is_deleted ===")
print(df["is_deleted"].value_counts())
print()

# is_locked
print("=== is_locked ===")
print(df["is_locked"].value_counts())
print()

# other_names structure
print("=== other_names samples ===")
for i in range(5):
    print(f"  Row {i}: {df.iloc[i]['other_names']} (type: {type(df.iloc[i]['other_names'])})")
print()

# Check body format patterns
valid = df[df["body"].notna() & (df["body"].str.strip() != "")]
print("=== Body Format Patterns ===")
# DText markers
patterns = {
    "[[": "wiki link [[tag]]",
    "h4.": "h4. heading",
    "h6.": "h6. heading",
    "[b]": "[b] bold",
    "[i]": "[i] italic",
    "* ": "* list item",
    "| ": "| table",
    '"': 'quoted text',
    "http": "URL",
}
for pat, label in patterns.items():
    count = valid["body"].str.contains(pat, regex=False, na=False).sum()
    print(f"  {label}: {count}/{len(valid)} ({100*count/len(valid):.1f}%)")
print()

# Show samples per category
print("=== Sample entries per category ===")
for cat in df["category"].unique():
    subset = valid[valid["category"] == cat]
    if len(subset) == 0:
        continue
    sample = subset.iloc[0]
    print(f"\n--- category: {cat} ({len(subset)} valid entries) ---")
    print(f"  title: {sample['title']}")
    print(f"  other_names: {sample['other_names']}")
    print(f"  body: {str(sample['body'])[:400]}")
    print(f"  tag: {sample['tag']}")
print()

# title vs tag difference
diff = df[df["title"] != df["tag"]]
print(f"=== title != tag: {len(diff)} entries ===")
if len(diff) > 0:
    for i in range(min(5, len(diff))):
        row = diff.iloc[i]
        print(f"  title='{row['title']}' vs tag='{row['tag']}'")
print()

# Very long bodies
print("=== Longest bodies (top 5) ===")
longest = valid.nlargest(5, "body", keep="first")
for _, row in longest.iterrows():
    print(f"  title={row['title']}, len={len(row['body'])}, category={row['category']}")
    print(f"  body preview: {row['body'][:200]}")
    print()
