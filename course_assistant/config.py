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
VISUAL_EMBED_URL = os.getenv("VISUAL_EMBED_URL", "")
RERANK_URL = os.getenv("RERANK_URL", "")
PARSE_URL = os.getenv("PARSE_URL", "")

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
