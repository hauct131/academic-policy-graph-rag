import json
import re
import unicodedata
from pathlib import Path
from typing import Any

def normalize_text(text: str) -> str:
    """Lowercase and remove Vietnamese/Latin diacritics."""
    if not text:
        return ""
    # Explicitly map đ/Đ first because unicodedata NFD decomposition doesn't strip them
    text = text.replace("đ", "d").replace("Đ", "D")
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return text.lower()


def tokenize(text: str) -> list[str]:
    """Normalize text and return list of lowercase alphanumeric tokens."""
    norm = normalize_text(text)
    return re.findall(r"[a-z0-9]+", norm)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
