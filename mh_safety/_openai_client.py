"""OpenAI API backend (e.g. GPT-5), with structured-output judging and disk cache.

`judge_json` uses OpenAI **structured outputs** (a strict JSON schema), so the judge
always returns valid JSON that matches the rubric -- including enum fields like
`behavior` -- which small local models could not do reliably. Requires
`OPENAI_API_KEY`.
"""
import json

from ._base_client import CachingClient


class OpenAIClient(CachingClient):
    def __init__(self, llm_cfg):
        super().__init__(llm_cfg)
        from openai import OpenAI  # lazy so the package imports without the SDK/key
        self._client = OpenAI()  # reads OPENAI_API_KEY

    @staticmethod
    def _messages(system, user):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        return messages

    def generate(self, system, user):
        key = json.dumps(["openai_gen", self.cfg.model_generation, system, user])
        hit = self._cache_get(key)
        if hit is not None:
            return hit["text"]
        resp = self._client.chat.completions.create(
            model=self.cfg.model_generation, messages=self._messages(system, user))
        text = (resp.choices[0].message.content or "").strip()
        return self._cache_put(key, {"text": text})["text"]

    def judge_json(self, system, user, schema, max_tokens=None):
        key = json.dumps(["openai_judge", self.cfg.model_judge, system, user, schema])
        hit = self._cache_get(key)
        if hit is not None:
            return hit
        messages = self._messages(system, user)
        try:  # strict structured output -> guaranteed schema-valid JSON (incl. enums)
            resp = self._client.chat.completions.create(
                model=self.cfg.model_judge, messages=messages,
                response_format={"type": "json_schema",
                                 "json_schema": {"name": "judgment", "schema": schema, "strict": True}},
            )
            data = json.loads(resp.choices[0].message.content)
        except Exception:  # fallback: plain JSON mode + lenient coercion to the schema
            from ._hf_client import _coerce_to_schema
            resp = self._client.chat.completions.create(
                model=self.cfg.model_judge, messages=messages,
                response_format={"type": "json_object"},
            )
            data = _coerce_to_schema(json.loads(resp.choices[0].message.content), schema)
        return self._cache_put(key, data)
