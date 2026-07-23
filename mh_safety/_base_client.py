"""Shared on-disk cache used by every backend client.

All backends map a request key to the same file (`sha256(key)[:24].json`) under
`cfg.cache_dir`, so re-runs are free and a crashed run resumes. Only the *key
string* is backend-specific (each client builds its own so existing caches keep
matching); the storage mechanics live here once.
"""
import hashlib
import json
from pathlib import Path


class CachingClient:
    def __init__(self, llm_cfg):
        self.cfg = llm_cfg
        self.cache_dir = Path(llm_cfg.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, key):
        return self.cache_dir / (hashlib.sha256(key.encode("utf-8")).hexdigest()[:24] + ".json")

    def _cache_get(self, key):
        path = self._cache_path(key)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return None

    def _cache_put(self, key, value):
        self._cache_path(key).write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        return value
