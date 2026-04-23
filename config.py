"""Configuration — set environment variables or create a .env file."""

import os
from pathlib import Path

# Load .env file if present
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    with open(_env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())

# ── SiliconFlow API ─────────────────────────────────────────────────────────

SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "")
SILICONFLOW_BASE_URL = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")

# Embedding model
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-8B")

# LLM model
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-ai/DeepSeek-V3.2")

# ── Server ──────────────────────────────────────────────────────────────────

SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))
WEBUI_PORT = int(os.getenv("WEBUI_PORT", "7860"))

# ── RAG ─────────────────────────────────────────────────────────────────────

INDEX_DIR = os.getenv("INDEX_DIR", "embedding_index")
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "20"))
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.4"))

# ── Validation ──────────────────────────────────────────────────────────────

def validate():
    if not SILICONFLOW_API_KEY:
        raise ValueError(
            "SILICONFLOW_API_KEY is not set. "
            "Set it via environment variable or edit config.py"
        )
