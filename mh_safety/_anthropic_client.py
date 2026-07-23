"""Anthropic API backend.

`generate()` returns free text; `judge_json()` returns schema-valid JSON via
structured outputs. Every call is cached on disk (see ``CachingClient``).
Requires `ANTHROPIC_API_KEY` in the environment.
"""
import json

import anthropic

from ._base_client import CachingClient


class AnthropicClient(CachingClient):
    def __init__(self, llm_cfg):
        super().__init__(llm_cfg)
        self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY

    def generate(self, system, user):
        key = json.dumps(["gen", self.cfg.model_generation, system, user])
        hit = self._cache_get(key)
        if hit is not None:
            return hit["text"]
        kwargs = dict(model=self.cfg.model_generation, max_tokens=self.cfg.max_tokens,
                      messages=[{"role": "user", "content": user}])
        if system:  # omit the system field entirely for a truly bare run
            kwargs["system"] = system
        resp = self._client.messages.create(**kwargs)
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        return self._cache_put(key, {"text": text})["text"]

    def judge_json(self, system, user, schema, max_tokens=500):
        key = json.dumps(["judge", self.cfg.model_judge, system, user])
        hit = self._cache_get(key)
        if hit is not None:
            return hit
        resp = self._client.messages.create(
            model=self.cfg.model_judge, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"format": {"type": "json_schema", "schema": schema}},
        )
        data = json.loads("".join(b.text for b in resp.content if b.type == "text"))
        return self._cache_put(key, data)
