"""Common utilities for hashing, serialization, logging, and security."""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Union


def to_repo_relative(file_path: Union[str, Path]) -> str:
    """Returns a clean repository-relative path with forward slashes."""
    try:
        from src.config import BASE_DIR
        return str(Path(file_path).resolve().relative_to(BASE_DIR.resolve())).replace("\\", "/")
    except Exception:
        return Path(file_path).name


def setup_logger(name: str = "crop_analytics") -> logging.Logger:
    """Configures and returns a clean console logger that sanitizes local machine paths."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()

        class PathSanitizingFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                formatted = super().format(record)
                try:
                    from src.config import BASE_DIR
                    base_resolved = str(BASE_DIR.resolve())
                    base_fwd = base_resolved.replace("\\", "/")
                    formatted = formatted.replace(base_resolved, ".").replace(base_fwd, ".")
                except Exception:
                    pass
                return formatted

        formatter = PathSanitizingFormatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def compute_sha256(file_path: Union[str, Path]) -> str:
    """Calculates SHA-256 hash of a file on disk."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found for hash calculation: {path}")
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_dict_sha256(obj: Dict[str, Any]) -> str:
    """Calculates deterministic SHA-256 hash of a Python dictionary."""
    encoded = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def save_json(data: Any, file_path: Union[str, Path], indent: int = 2) -> None:
    """Saves data structure as indented JSON."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dumps(data, indent=indent, default=str)
        json.dump(data, f, indent=indent, default=str)


def load_json(file_path: Union[str, Path]) -> Any:
    """Loads and returns JSON file content."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file does not exist: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def mask_secret(secret: str, visible_chars: int = 4) -> str:
    """Masks secret key for safe logging, revealing only the initial characters."""
    if not secret:
        return "<EMPTY>"
    if len(secret) <= visible_chars:
        return "****"
    return f"{secret[:visible_chars]}...{secret[-visible_chars:]} (len={len(secret)})"
