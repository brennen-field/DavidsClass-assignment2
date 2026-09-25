"""Settings loaded from the local .env file.

Keys are read here and passed to service clients. Never print, log, or
display them in the UI.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

CLASS_API_KEY = os.getenv("CLASS_API_KEY", "")

LLM_URL = os.getenv("LLM_URL", "")
TEXT_EMBED_URL = os.getenv("TEXT_EMBED_URL", "")
TEXT_EMBED_MODEL = os.getenv(
    "TEXT_EMBED_MODEL",
    "nvidia/Nemotron-3-Embed-1B-BF16",
)
VISUAL_EMBED_URL = os.getenv("VISUAL_EMBED_URL", "")
VISUAL_EMBED_MODEL = os.getenv(
    "VISUAL_EMBED_MODEL",
    "Qwen/Qwen3-VL-Embedding-2B",
)
RERANK_URL = os.getenv("RERANK_URL", "")
RERANK_MODEL = os.getenv(
    "RERANK_MODEL",
    "Qwen/Qwen3-VL-Reranker-2B",
)
PARSE_URL = os.getenv("PARSE_URL", "")

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
