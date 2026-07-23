"""mh_safety -- measuring empathy degradation and role/intent safety in LLM
responses to mental-health scenarios.

Two studies share one backend:
  * empathy/      -- empathy degradation under adversarial prompting (Reddit posts)
  * role_intent/  -- role x intent factorial safety study (curated scenarios)

Shared infrastructure lives at the top level: the backend-agnostic LLM client
factory + cache (llm; Anthropic or local Ollama), text utilities (text),
statistics helpers (stats), and plotting helpers (visual).
"""
from .config import LLMConfig, EmpathyConfig, RoleIntentConfig
from .llm import LLMClient

__all__ = ["LLMConfig", "EmpathyConfig", "RoleIntentConfig", "LLMClient"]
