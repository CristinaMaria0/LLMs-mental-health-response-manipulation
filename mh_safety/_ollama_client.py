"""Ollama local-server backend, with structured-output judging and disk cache."""
import json
import re

from ollama import Client

from ._base_client import CachingClient


class OllamaClient(CachingClient):
    def __init__(self, llm_cfg):
        super().__init__(llm_cfg)
        self._client = Client(host=llm_cfg.host)

    @staticmethod
    def _messages(system, user):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        return messages

    @staticmethod
    def _parse_json(text):
        """Parse JSON safely if a model wraps it in Markdown fences."""
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text).strip()
        return json.loads(text)

    def generate(self, system, user):
        key = json.dumps(
            ["ollama_generation", self.cfg.model_generation, self.cfg.max_tokens,
             self.cfg.generation_temperature, self.cfg.seed, system, user],
            ensure_ascii=False,
        )
        hit = self._cache_get(key)
        if hit is not None:
            return hit["text"]

        response = self._client.chat(
            model=self.cfg.model_generation,
            messages=self._messages(system, user),
            stream=False,
            keep_alive=self.cfg.keep_alive,
            options={
                "num_predict": self.cfg.max_tokens,
                "temperature": self.cfg.generation_temperature,
                "seed": self.cfg.seed,
            },
        )
        return self._cache_put(key, {"text": response.message.content.strip()})["text"]

    def judge_json(self, system, user, schema, max_tokens=260):
        key = json.dumps(
            ["ollama_judge", self.cfg.model_judge, max_tokens, self.cfg.judge_temperature,
             self.cfg.seed, system, user, schema],
            ensure_ascii=False,
            sort_keys=True,
        )
        hit = self._cache_get(key)
        if hit is not None:
            return hit

        last_error = None
        last_raw = ""
        # A second attempt uses a larger token budget only if the first JSON is incomplete.
        for token_budget in (max_tokens, max_tokens * 2):
            response = self._client.chat(
                model=self.cfg.model_judge,
                messages=self._messages(system, user),
                format=schema,
                stream=False,
                keep_alive=self.cfg.keep_alive,
                options={
                    "num_predict": token_budget,
                    "temperature": self.cfg.judge_temperature,
                    "seed": self.cfg.seed,
                },
            )
            raw = response.message.content.strip()
            try:
                data = self._parse_json(raw)
                if not isinstance(data, dict):
                    raise ValueError("Judge response is not a JSON object.")
                return self._cache_put(key, data)
            except (json.JSONDecodeError, ValueError) as error:
                last_error = error
                last_raw = raw

        raise RuntimeError(
            "The local judge returned incomplete or invalid JSON after two attempts.\n"
            f"Last raw response:\n{last_raw[:1200]!r}"
        ) from last_error
