import pyarrow.parquet as pq

table = pq.read_table("D:/Coding/Python/dan_emb/danbooru_wiki.parquet")
print(f"Rows: {table.num_rows}")
print(f"Columns: {table.column_names}")
print()

# Show 3 samples
df = table.to_pandas()
for i in range(min(3, len(df))):
    print(f"--- Row {i} ---")
    for col in df.columns:
        val = str(df.iloc[i][col])
        if len(val) > 300:
            val = val[:300] + "..."
        print(f"  {col}: {val}")
    print()

# Stats
if "body" in df.columns:
    empty = df["body"].isna().sum() + (df["body"].str.strip() == "").sum()
    print(f"Empty body rows: {empty}/{len(df)}")
if "title" in df.columns:
    print(f"Unique titles: {df['title'].nunique()}/{len(df)}")
