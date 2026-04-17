import os
import json
import re
from typing import Dict, Any, Optional
from openai import OpenAI
from OpenCAAF.adapters.base import LLMAdapter

def backend_kwargs() -> Dict[str, Any]:
    """Read `CAAF_BACKEND` env and return adapter kwargs.

    Supported backends:
      - unset / ``openai``    → default OpenAI endpoint
      - ``together``          → Together.ai (TOGETHER_API_KEY)
      - ``openrouter``        → OpenRouter (OPENROUTER_API_KEY)
    """
    backend = (os.getenv("CAAF_BACKEND") or "openai").lower()
    if backend == "together":
        return {
            "base_url": os.getenv("TOGETHER_BASE_URL",
                                  "https://api.together.xyz/v1"),
            "api_key_env": "TOGETHER_API_KEY",
        }
    if backend == "openrouter":
        return {
            "base_url": os.getenv("OPENROUTER_BASE_URL",
                                  "https://openrouter.ai/api/v1"),
            "api_key_env": "OPENROUTER_API_KEY",
        }
    return {}


class OpenAIAdapter(LLMAdapter):
    def __init__(self, model: str = "gpt-4o", temperature: float = 0.7,
                 base_url: Optional[str] = None,
                 api_key_env: str = "OPENAI_API_KEY"):
        super().__init__()
        self.client = OpenAI(
            api_key=os.getenv(api_key_env),
            base_url=base_url,
        )
        self.model = model
        self.temperature = temperature

        # 2026 pricing (per 1M tokens)
        self.pricing = {
            # --- OpenAI ---
            "gpt-4o":      {"prompt": 5.0,  "completion": 15.0},
            "gpt-4o-mini": {"prompt": 0.15, "completion": 0.6},
            # --- Together.ai open-weight (for replication) ---
            # Paid Turbo endpoints
            "Qwen/Qwen2.5-7B-Instruct-Turbo":              {"prompt": 0.30, "completion": 0.30},
            "Qwen/Qwen2.5-14B-Instruct":                   {"prompt": 0.80, "completion": 0.80},
            "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": {"prompt": 0.18, "completion": 0.18},
            "meta-llama/Llama-3.3-70B-Instruct-Turbo":     {"prompt": 0.88, "completion": 0.88},
            # $0-priced endpoints (serverless lite / free tier on Together)
            "Qwen/Qwen2.5-7B-Instruct":                    {"prompt": 0.0,  "completion": 0.0},
            "Qwen/Qwen3-8B":                               {"prompt": 0.0,  "completion": 0.0},
            # --- OpenRouter (free tier) ---
            "meta-llama/llama-3.3-70b-instruct:free":      {"prompt": 0.0,  "completion": 0.0},
            "meta-llama/llama-3.2-3b-instruct:free":       {"prompt": 0.0,  "completion": 0.0},
            "google/gemma-3-12b-it:free":                  {"prompt": 0.0,  "completion": 0.0},
            "google/gemma-3-27b-it:free":                  {"prompt": 0.0,  "completion": 0.0},
            "qwen/qwen3-next-80b-a3b-instruct:free":       {"prompt": 0.0,  "completion": 0.0},
            # --- OpenRouter (paid but cheap, $/1M tok) ---
            "qwen/qwen-2.5-7b-instruct":                   {"prompt": 0.04, "completion": 0.10},
            "qwen/qwen-2.5-14b-instruct":                  {"prompt": 0.08, "completion": 0.20},
            "qwen/qwen3-8b":                               {"prompt": 0.05, "completion": 0.40},
            "meta-llama/llama-3.1-8b-instruct":            {"prompt": 0.02, "completion": 0.05},
            "meta-llama/llama-3.2-3b-instruct":            {"prompt": 0.051, "completion": 0.34},
            "meta-llama/llama-3.3-70b-instruct":           {"prompt": 0.10, "completion": 0.32},
            "google/gemma-3-4b-it":                        {"prompt": 0.02, "completion": 0.06},
            "google/gemma-3-12b-it":                       {"prompt": 0.04, "completion": 0.13},
            "google/gemma-3-27b-it":                       {"prompt": 0.08, "completion": 0.16},
            "cohere/command-r7b-12-2024":                  {"prompt": 0.04, "completion": 0.15},
            "mistralai/mistral-nemo":                      {"prompt": 0.02, "completion": 0.04},
        }

    def completion(self, prompt: str, schema: Optional[Dict[str, Any]] = None, force_json: bool = False) -> Dict[str, Any]:
        self.last_prompt = prompt

        # Only use json_object if schema is provided or force_json is explicitly True
        response_format = {"type": "json_object"} if schema or force_json else None

        # OpenRouter-specific routing hints: exclude providers known to enforce
        # aggressive per-free-user caps that surface as 402/429 mid-run.
        # Controlled via CAAF_OPENROUTER_IGNORE (comma-separated provider names);
        # default excludes Venice. Set CAAF_OPENROUTER_IGNORE="" to disable.
        extra_body: Dict[str, Any] = {}
        if "openrouter.ai" in (str(self.client.base_url) or ""):
            ignore_raw = os.getenv("CAAF_OPENROUTER_IGNORE", "Venice")
            ignore_list = [p.strip() for p in ignore_raw.split(",") if p.strip()]
            if ignore_list:
                extra_body["provider"] = {"ignore": ignore_list,
                                          "allow_fallbacks": True}

        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
            response_format=response_format,
            extra_body=extra_body or None,
        )
        
        # Track Usage
        if resp.usage:
            self.total_prompt_tokens += resp.usage.prompt_tokens
            self.total_completion_tokens += resp.usage.completion_tokens
            self.last_usage = {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "total_tokens": resp.usage.total_tokens
            }
            
        content = resp.choices[0].message.content
        self.last_response = content
        return self._parse_json_safely(content)

    def get_total_cost(self) -> float:
        prices = self.pricing.get(self.model, {"prompt": 0.0, "completion": 0.0})
        cost_prompt = (self.total_prompt_tokens / 1_000_000) * prices["prompt"]
        cost_comp = (self.total_completion_tokens / 1_000_000) * prices["completion"]
        return cost_prompt + cost_comp

    def _parse_json_safely(self, text: str) -> Dict[str, Any]:
        if text is None:
            return {"error": "Response content was None"}
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
            
        # Try to find a JSON block in mixed markdown
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if match:
            clean_text = match.group(1)
            try:
                return json.loads(clean_text)
            except json.JSONDecodeError:
                pass
                
        # Fallback to broad regex extraction
        match_broad = re.search(r'(\{.*\})', text, re.DOTALL)
        if match_broad:
            try:
                return json.loads(match_broad.group(1))
            except json.JSONDecodeError:
                pass

        return {"raw_text": text, "error": "Could not extract valid JSON object from response."}
