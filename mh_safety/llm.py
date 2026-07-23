"""Backend-agnostic LLM client factory.

`LLMClient(llm_cfg)` returns the client for the configured backend
(`llm_cfg.backend`): the Anthropic API client, the local Ollama client, or a
local HuggingFace `transformers` model (`hf`, e.g. Gemma 3 / Qwen 3). All expose
the same `generate()` / `judge_json()` interface and cache every call on disk.
The concrete client (and its SDK) is imported lazily, so an Anthropic-only
environment never needs `ollama` or `transformers`, and vice versa.
"""


_JUDGE_CLIENTS = {}


def judge_client(llm_cfg):
    """Return a cached client for the shared judge model, so the judge (e.g. the
    OpenAI GPT-5 API client) is built once and reused to score every backend's responses."""
    key = (llm_cfg.backend, llm_cfg.model_judge, llm_cfg.cache_dir)
    client = _JUDGE_CLIENTS.get(key)
    if client is None:
        client = _JUDGE_CLIENTS[key] = LLMClient(llm_cfg)
    return client


def LLMClient(llm_cfg):
    backend = getattr(llm_cfg, "backend", "anthropic")
    if backend == "ollama":
        from ._ollama_client import OllamaClient
        return OllamaClient(llm_cfg)
    if backend == "hf":
        from ._hf_client import HFClient
        return HFClient(llm_cfg)
    if backend == "anthropic":
        from ._anthropic_client import AnthropicClient
        return AnthropicClient(llm_cfg)
    if backend == "openai":
        from ._openai_client import OpenAIClient
        return OpenAIClient(llm_cfg)
    raise ValueError(f"Unknown LLM backend: {backend!r} (expected 'anthropic', 'ollama', 'hf' or 'openai')")
