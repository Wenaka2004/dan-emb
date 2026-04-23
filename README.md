# Danbooru Tag Expander

RAG-based Danbooru tag expansion system. Enter a natural language description, get a comprehensive set of Danbooru tags.

## How It Works

1. **Translate** — Non-English input is translated to English via LLM
2. **RAG Retrieval** — Query is embedded and matched against 137k Danbooru wiki entries using vector similarity
3. **Stage 1 LLM** — Retrieved wiki context is fed to the LLM for initial tag expansion
4. **Supplementary RAG** — Key tags from Stage 1 are used for additional retrieval
5. **Stage 2 LLM** — LLM refines the tag set with supplementary context

Character-copyright relationships are automatically resolved — searching for a character also brings in its source IP.

## Quick Start

### 1. Install dependencies

```bash
pip install numpy pandas pyarrow requests openai fastapi uvicorn gradio tqdm
```

### 2. Configure API key

Copy `.env.example` to `.env` and fill in your key:

```bash
cp .env.example .env
```

```env
SILICONFLOW_API_KEY=sk-your-key-here
```

Get your key from [SiliconFlow](https://cloud.siliconflow.cn/). The service uses:
- **Qwen3-Embedding-8B** for embedding (~0.28 CNY/M tokens)
- **DeepSeek-V3.2** for LLM expansion

### 3. Prepare the index

Download the Danbooru wiki dataset from [isek-ai/danbooru-wiki-2024](https://huggingface.co/datasets/isek-ai/danbooru-wiki-2024) and place the parquet file in the project root as `danbooru_wiki.parquet`.

Build the embedding index:

```bash
python build_embeddings.py
```

This will:
- Clean DText markup from wiki bodies
- Filter useless sections (member lists, external links, etc.)
- Build character-copyright mappings
- Embed all entries via API (~4 CNY, ~30 min with 20 concurrent workers)

When done, `embedding_index/` will contain:
- `embeddings.npy` — 137502 × 4096 float32 vectors (~4.2 GB)
- `metadata.parquet` — Tag metadata with cleaned wiki text
- `char_copyright.json` — Character → copyright mappings

### 4. Launch the server

**API only:**

```bash
python server.py
```

**API + WebUI:**

```bash
python server.py --webui
```

Open http://localhost:7860 for the WebUI, or http://localhost:8000/docs for the Swagger API docs.

## API Usage

```bash
curl -X POST http://localhost:8000/expand \
  -H "Content-Type: application/json" \
  -d '{"prompt": "银发红眼穿校服的少女"}'
```

Response:

```json
{
  "input": "银发红眼穿校服的少女",
  "english_input": "A girl with silver hair and red eyes wearing a school uniform",
  "tags": ["1girl", "solo", "original", "long hair", "silver hair", "red eyes", "school uniform", "..."],
  "tag_count": 19,
  "latency_ms": 11797
}
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `prompt` | string | required | Natural language description |
| `top_k` | int | 20 | Number of RAG results |
| `temperature` | float | 0.4 | LLM temperature |
| `two_stage` | bool | true | Enable two-stage expansion |
| `category_filter` | list | null | Filter by category (e.g. `["general", "character"]`) |

## Configuration

All settings can be configured via environment variables or `.env` file:

| Variable | Default | Description |
|---|---|---|
| `SILICONFLOW_API_KEY` | — | **Required**. Your SiliconFlow API key |
| `SILICONFLOW_BASE_URL` | `https://api.siliconflow.cn/v1` | API base URL |
| `EMBEDDING_MODEL` | `Qwen/Qwen3-Embedding-8B` | Embedding model |
| `LLM_MODEL` | `deepseek-ai/DeepSeek-V3.2` | LLM model for expansion |
| `SERVER_HOST` | `0.0.0.0` | Server bind address |
| `SERVER_PORT` | `8000` | API server port |
| `WEBUI_PORT` | `7860` | Gradio WebUI port |
| `INDEX_DIR` | `embedding_index` | Path to embedding index |
| `DEFAULT_TOP_K` | `20` | Default number of RAG results |
| `DEFAULT_TEMPERATURE` | `0.4` | Default LLM temperature |

## Project Structure

```
├── config.py                # Configuration (env vars / .env)
├── dtext.py                 # DText markup cleaner
├── rag.py                   # RAG retrieval service (numpy)
├── expander.py              # Two-stage tag expansion (streaming + non-streaming)
├── server.py                # FastAPI server + Gradio WebUI
├── build_embeddings.py      # Build embedding index from wiki data
├── build_char_copyright.py  # Build character-copyright mapping
├── .env.example             # Environment variable template
├── embedding_index/         # Generated index (not in git)
│   ├── embeddings.npy
│   ├── metadata.parquet
│   └── char_copyright.json
└── danbooru_wiki.parquet    # Source dataset (not in git)
```

## Why RAG?

General-purpose LLMs struggle with Danbooru tag expansion because they lack:
- Knowledge of tag semantics and boundaries (e.g. `skindentation` vs `tight`)
- Character-copyright relationships
- Tag co-occurrence patterns
- Proper Danbooru tag ordering conventions

RAG injects real wiki knowledge into the LLM context, enabling accurate and comprehensive tag expansion.

## License

MIT
