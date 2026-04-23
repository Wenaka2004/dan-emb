# Danbooru Tag Expander / Danbooru 标签扩写器

[English](#english) | [中文](#中文)

---

<a id="中文"></a>

## 中文

基于 RAG 的 Danbooru 标签扩写系统。输入自然语言描述，输出完整的 Danbooru 标签集。

### 工作原理

1. **翻译** — 非英文输入通过 LLM 翻译为英文
2. **RAG 检索** — 将查询向量化，在 13.7 万条 Danbooru wiki 条目中进行相似度匹配
3. **第一阶段 LLM** — 将检索到的 wiki 上下文喂给 LLM 进行初始标签扩写
4. **补充 RAG** — 用第一阶段产出的关键标签进行二次检索
5. **第二阶段 LLM** — LLM 结合补充上下文精炼标签集

角色-版权关系会自动解析 — 检索角色时会同时带入其所属 IP。

### 快速开始

#### 1. 安装依赖

```bash
pip install numpy pandas pyarrow requests openai fastapi uvicorn gradio tqdm
```

#### 2. 配置 API 密钥

复制 `.env.example` 为 `.env` 并填入密钥：

```bash
cp .env.example .env
```

```env
SILICONFLOW_API_KEY=sk-your-key-here
```

在 [SiliconFlow](https://cloud.siliconflow.cn/) 获取密钥。服务使用：
- **Qwen3-Embedding-8B** 用于向量化（约 ¥0.28/百万 token）
- **DeepSeek-V3.2** 用于 LLM 扩写

#### 3. 准备索引

**方案 A：下载预构建索引（推荐）**

直接下载预构建的嵌入索引，跳过耗时的构建步骤：

```bash
pip install huggingface_hub
python -c "
from huggingface_hub import hf_hub_download
import os
os.makedirs('embedding_index', exist_ok=True)
for f in ['embeddings.npy', 'metadata.parquet', 'char_copyright.json']:
    path = hf_hub_download('Wenaka/Danbooru_Wiki_Embedding_Qwen3_8B', f, local_dir='embedding_index')
    print(f'Downloaded: {f}')
"
```

索引内容：
- `embeddings.npy` — 137502 × 4096 float32 向量（约 4.2 GB）
- `metadata.parquet` — 标签元数据与清洗后的 wiki 文本
- `char_copyright.json` — 角色→版权映射

**方案 B：自行构建索引**

从 [isek-ai/danbooru-wiki-2024](https://huggingface.co/datasets/isek-ai/danbooru-wiki-2024) 下载数据集，将 parquet 文件放在项目根目录，命名为 `danbooru_wiki.parquet`。

```bash
python build_embeddings.py
```

该步骤会：
- 清洗 wiki 正文中的 DText 标记
- 过滤无用章节（成员列表、外链等）
- 构建角色-版权映射
- 通过 API 嵌入所有条目（约 ¥4，20 并发约 30 分钟）

#### 4. 启动服务

**仅 API：**

```bash
python server.py
```

**API + WebUI：**

```bash
python server.py --webui
```

打开 http://localhost:7860 使用 WebUI，或 http://localhost:8000/docs 查看 Swagger API 文档。

### API 用法

```bash
curl -X POST http://localhost:8000/expand \
  -H "Content-Type: application/json" \
  -d '{"prompt": "银发红眼穿校服的少女"}'
```

响应：

```json
{
  "input": "银发红眼穿校服的少女",
  "english_input": "A girl with silver hair and red eyes wearing a school uniform",
  "tags": ["1girl", "solo", "original", "long hair", "silver hair", "red eyes", "school uniform", "..."],
  "tag_count": 19,
  "latency_ms": 11797
}
```

#### 参数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `prompt` | string | 必填 | 自然语言描述 |
| `top_k` | int | 20 | RAG 检索结果数 |
| `temperature` | float | 0.4 | LLM 温度 |
| `two_stage` | bool | true | 启用两阶段扩写 |
| `category_filter` | list | null | 按类别过滤（如 `["general", "character"]`） |

### 配置

所有设置可通过环境变量或 `.env` 文件配置：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `SILICONFLOW_API_KEY` | — | **必填**。SiliconFlow API 密钥 |
| `SILICONFLOW_BASE_URL` | `https://api.siliconflow.cn/v1` | API 基础地址 |
| `EMBEDDING_MODEL` | `Qwen/Qwen3-Embedding-8B` | 嵌入模型 |
| `LLM_MODEL` | `deepseek-ai/DeepSeek-V3.2` | LLM 扩写模型 |
| `SERVER_HOST` | `0.0.0.0` | 服务绑定地址 |
| `SERVER_PORT` | `8000` | API 端口 |
| `WEBUI_PORT` | `7860` | Gradio WebUI 端口 |
| `INDEX_DIR` | `embedding_index` | 嵌入索引路径 |
| `DEFAULT_TOP_K` | `20` | 默认 RAG 检索数量 |
| `DEFAULT_TEMPERATURE` | `0.4` | 默认 LLM 温度 |

### 项目结构

```
├── config.py                # 配置（环境变量 / .env）
├── dtext.py                 # DText 标记清洗器
├── rag.py                   # RAG 检索服务（numpy）
├── expander.py              # 两阶段标签扩写（流式 + 非流式）
├── server.py                # FastAPI 服务 + Gradio WebUI
├── build_embeddings.py      # 从 wiki 数据构建嵌入索引
├── build_char_copyright.py  # 构建角色-版权映射
├── .env.example             # 环境变量模板
├── embedding_index/         # 生成的索引（不含在 git 中）
│   ├── embeddings.npy
│   ├── metadata.parquet
│   └── char_copyright.json
└── danbooru_wiki.parquet    # 源数据集（不含在 git 中）
```

### 为什么用 RAG？

通用 LLM 难以胜任 Danbooru 标签扩写，因为它们缺乏：
- 对标签语义和边界的理解（如 `skindentation` 与 `tight` 的区别）
- 角色-版权归属关系
- 标签共现规律
- Danbooru 标签排序惯例

RAG 将真实 wiki 知识注入 LLM 上下文，实现准确且全面的标签扩写。

### 许可证

MIT

---

<a id="english"></a>

## English

RAG-based Danbooru tag expansion system. Enter a natural language description, get a comprehensive set of Danbooru tags.

### How It Works

1. **Translate** — Non-English input is translated to English via LLM
2. **RAG Retrieval** — Query is embedded and matched against 137k Danbooru wiki entries using vector similarity
3. **Stage 1 LLM** — Retrieved wiki context is fed to the LLM for initial tag expansion
4. **Supplementary RAG** — Key tags from Stage 1 are used for additional retrieval
5. **Stage 2 LLM** — LLM refines the tag set with supplementary context

Character-copyright relationships are automatically resolved — searching for a character also brings in its source IP.

### Quick Start

#### 1. Install dependencies

```bash
pip install numpy pandas pyarrow requests openai fastapi uvicorn gradio tqdm
```

#### 2. Configure API key

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

#### 3. Prepare the index

**Option A: Download pre-built index (Recommended)**

Download the pre-built embedding index directly, skip the time-consuming build step:

```bash
pip install huggingface_hub
python -c "
from huggingface_hub import hf_hub_download
import os
os.makedirs('embedding_index', exist_ok=True)
for f in ['embeddings.npy', 'metadata.parquet', 'char_copyright.json']:
    path = hf_hub_download('Wenaka/Danbooru_Wiki_Embedding_Qwen3_8B', f, local_dir='embedding_index')
    print(f'Downloaded: {f}')
"
```

The index contains:
- `embeddings.npy` — 137502 × 4096 float32 vectors (~4.2 GB)
- `metadata.parquet` — Tag metadata with cleaned wiki text
- `char_copyright.json` — Character → copyright mappings

**Option B: Build the index yourself**

Download the Danbooru wiki dataset from [isek-ai/danbooru-wiki-2024](https://huggingface.co/datasets/isek-ai/danbooru-wiki-2024) and place the parquet file in the project root as `danbooru_wiki.parquet`.

```bash
python build_embeddings.py
```

This will:
- Clean DText markup from wiki bodies
- Filter useless sections (member lists, external links, etc.)
- Build character-copyright mappings
- Embed all entries via API (~4 CNY, ~30 min with 20 concurrent workers)

#### 4. Launch the server

**API only:**

```bash
python server.py
```

**API + WebUI:**

```bash
python server.py --webui
```

Open http://localhost:7860 for the WebUI, or http://localhost:8000/docs for the Swagger API docs.

### API Usage

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

#### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `prompt` | string | required | Natural language description |
| `top_k` | int | 20 | Number of RAG results |
| `temperature` | float | 0.4 | LLM temperature |
| `two_stage` | bool | true | Enable two-stage expansion |
| `category_filter` | list | null | Filter by category (e.g. `["general", "character"]`) |

### Configuration

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

### Project Structure

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

### Why RAG?

General-purpose LLMs struggle with Danbooru tag expansion because they lack:
- Knowledge of tag semantics and boundaries (e.g. `skindentation` vs `tight`)
- Character-copyright relationships
- Tag co-occurrence patterns
- Proper Danbooru tag ordering conventions

RAG injects real wiki knowledge into the LLM context, enabling accurate and comprehensive tag expansion.

### License

MIT
