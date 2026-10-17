from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class Config:
    model: Path = Path(os.environ.get("MC3_MODEL_DIR", "/models/vision"))
    embedding: Path = Path(os.environ.get("MC3_EMBED_DIR", "/models/embedding"))
    index: Path = Path(os.environ.get("MC3_INDEX_DIR", "/app/index"))
    output: Path = Path(os.environ.get("MC3_OUTPUT_DIR", "/app/output"))
    socket: str = os.environ.get("MC3_SOCKET", "/tmp/mc3.sock")
    context_tokens: int = int(os.environ.get("MC3_CONTEXT_TOKENS", "10000"))
    query_seconds: float = float(os.environ.get("MC3_QUERY_SECONDS", "26"))
    startup_seconds: float = float(os.environ.get("MC3_STARTUP_SECONDS", "570"))
    parse_seconds: float = float(os.environ.get("MC3_PARSE_SECONDS", "12"))
    parse_workers: int = max(1, min(4, int(os.environ.get("MC3_PARSE_WORKERS", "4"))))
    verify_answers: bool = os.environ.get("MC3_VERIFY_ANSWERS", "1") == "1"
    chunk_chars: int = 2400
    overlap_chars: int = 350
