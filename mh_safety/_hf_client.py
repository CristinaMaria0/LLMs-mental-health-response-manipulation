"""HuggingFace `transformers` local backend (e.g. Gemma 3, Qwen 3).

Loads the model + tokenizer once, then exposes the shared
``generate()`` / ``judge_json()`` interface with on-disk cache (see
``CachingClient``). Deterministic by default (``do_sample=False``), so every call
is cached and a crashed run resumes for free. Requires `transformers` and `torch`
(a CUDA GPU is strongly recommended) plus access to the model weights.
"""
import json
import re

from ._base_client import CachingClient


class HFClient(CachingClient):
    def __init__(self, llm_cfg):
        super().__init__(llm_cfg)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

        self._torch = torch
        self._set_seed = set_seed

        auth = {"token": llm_cfg.hf_token} if getattr(llm_cfg, "hf_token", None) else {}
        model_id = llm_cfg.model_generation

        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, **auth)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        kwargs = {"low_cpu_mem_usage": True, **auth}
        if torch.cuda.is_available():
            kwargs["device_map"] = "auto"
            if llm_cfg.use_4bit:
                try:
                    from transformers import BitsAndBytesConfig
                    kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16,
                    )
                except Exception:
                    kwargs["torch_dtype"] = torch.float16
            else:
                kwargs["torch_dtype"] = torch.float16
        else:
            kwargs["torch_dtype"] = torch.float32

        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        model.eval()
        self._tokenizer = tokenizer
        self._model = model

    # ----- prompting / decoding -----
    def _chat_prompt(self, system, user):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        tokenizer = self._tokenizer
        if getattr(tokenizer, "chat_template", None):
            try:
                return tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=self.cfg.enable_thinking,
                )
            except TypeError:  # models whose template has no `enable_thinking` (e.g. Gemma)
                return tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
        return f"System:\n{system or ''}\n\nUser:\n{user}\n\nAssistant:\n"

    def _run(self, prompt, max_new_tokens, do_sample=None, temperature=None, seed=None):
        torch = self._torch
        tokenizer, model = self._tokenizer, self._model
        do_sample = self.cfg.do_sample if do_sample is None else do_sample
        temperature = self.cfg.temperature if temperature is None else temperature
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=self.cfg.max_input_tokens).to(next(model.parameters()).device)
        kwargs = dict(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample,
                      repetition_penalty=self.cfg.repetition_penalty,
                      pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        if do_sample:
            self._set_seed(self.cfg.seed if seed is None else seed)
            kwargs.update(temperature=temperature, top_p=self.cfg.top_p)
        with torch.inference_mode():
            output_ids = model.generate(**kwargs)
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def _key(self, kind, max_new_tokens, system, user, extra=None):
        return json.dumps(
            [kind, self.cfg.model_generation, max_new_tokens, self.cfg.do_sample,
             self.cfg.temperature, self.cfg.top_p, self.cfg.repetition_penalty,
             self.cfg.seed, system, user, extra],
            ensure_ascii=False, sort_keys=True,
        )

    # ----- interface -----
    def generate(self, system, user):
        key = self._key("hf_generation", self.cfg.max_new_tokens, system, user)
        hit = self._cache_get(key)
        if hit is not None:
            return hit["text"]
        text = self._run(self._chat_prompt(system, user), self.cfg.max_new_tokens)
        return self._cache_put(key, {"text": text})["text"]

    @staticmethod
    def _parse_json(raw):
        """Extract the first JSON object from a model response (tolerating fences/prose)."""
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", str(raw).strip(), flags=re.I | re.S).strip()
        match = re.search(r"\{.*\}", text, flags=re.S)
        for candidate in [text] + ([match.group(0)] if match else []):
            try:
                obj = json.loads(candidate)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                continue
        raise ValueError("no JSON object found in judge output")

    def judge_json(self, system, user, schema, max_tokens=None):
        base_tokens = max_tokens or self.cfg.judge_max_new_tokens
        key = self._key("hf_judge", base_tokens, system, user, extra=schema)
        hit = self._cache_get(key)
        if hit is not None:
            return hit

        prompt = self._chat_prompt(system, user)
        last_error = None
        attempts = max(1, getattr(self.cfg, "judge_max_attempts", 6))
        # Attempt 0: greedy. Attempt 1: greedy with 2x tokens (fixes truncation).
        # Later attempts: sample with rising temperature so the output actually differs
        # (repeating a greedy decode would just reproduce the same non-JSON text).
        for i in range(attempts):
            budget = base_tokens if i == 0 else base_tokens * 2
            do_sample = i >= 2
            temperature = min(0.3 + 0.2 * (i - 2), 0.9) if do_sample else 0.0
            raw = self._run(prompt, budget, do_sample=do_sample, temperature=temperature,
                            seed=self.cfg.seed + i)
            try:
                obj = self._parse_json(raw)
            except ValueError as error:
                last_error = error
                continue
            return self._cache_put(key, _coerce_to_schema(obj, schema))

        raise RuntimeError(f"HF judge returned invalid JSON after {attempts} attempts: {last_error}")


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _to_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y", "t"}
    return value is not None and bool(value)


# some models emit "danger" instead of the schema's "danger_flag"
_KEY_ALIASES = {"danger_flag": "danger"}


def _coerce_to_schema(obj, schema):
    """Coerce a judge's parsed JSON to exactly the schema's keys/types, so one
    judge model works for any rubric (empathy or role/intent). Missing keys
    default to 0.0 / False / ""."""
    out = {}
    for key, spec in schema.get("properties", {}).items():
        value = obj.get(key)
        if value is None and key in _KEY_ALIASES:
            value = obj.get(_KEY_ALIASES[key])
        kind = spec.get("type")
        if kind == "number":
            out[key] = _to_float(value)
        elif kind == "boolean":
            out[key] = _to_bool(value)
        elif kind == "string":
            out[key] = str(value).strip() if value is not None else ""
        else:
            out[key] = value
    return out
