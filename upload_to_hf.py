"""Upload embedding index to HuggingFace."""

from huggingface_hub import HfApi, create_repo

REPO_ID = "Wenaka/Danbooru_Wiki_Embedding_Qwen3_8B"
INDEX_DIR = "embedding_index"

api = HfApi()

# Upload files
files = [
    f"{INDEX_DIR}/embeddings_fp16.npz",
    f"{INDEX_DIR}/embeddings.npz",
    f"{INDEX_DIR}/metadata.parquet",
    f"{INDEX_DIR}/char_copyright.json",
]

for f in files:
    print(f"Uploading {f}...")
    api.upload_file(
        path_or_fileobj=f,
        path_in_repo=f.split("/")[-1],
        repo_id=REPO_ID,
        repo_type="dataset",
    )
    print(f"  Done: {f.split('/')[-1]}")

print(f"\nAll files uploaded to https://huggingface.co/datasets/{REPO_ID}")
