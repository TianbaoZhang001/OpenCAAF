import os
import json
import re
from typing import Dict, Any, List, Optional

from anthropic import Anthropic
from OpenCAAF.adapters.base import LLMAdapter


class AnthropicAdapter(LLMAdapter):
    """Native Anthropic Messages API adapter.

    Uses client.messages.create directly so that system prompt, stop_reason,
    and tool_use keep their native Anthropic semantics. An OpenAI-compat shim
    would translate these into fields with different meanings (e.g. mapping
    stop_reason="tool_use" onto finish_reason="tool_calls"), which is a
    translation artifact a paper reviewer could legitimately attack.
    """

    def __init__(
        self,
        model: str = "claude-opus-4-7",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        api_key_env: str = "ANTHROPIC_API_KEY",
        thinking_effort: Optional[str] = None,
    ):
        super().__init__()
        self.client = Anthropic(api_key=os.getenv(api_key_env))
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        # Extended thinking — newer Claude 4.x models use the adaptive API:
        #   thinking={"type": "adaptive"} + output_config={"effort": "low|medium|high"}
        # When set, the request forces temperature=1 and forbids assistant prefill.
        self.thinking_effort = thinking_effort

        self.last_stop_reason: Optional[str] = None
        self.last_content_blocks: List[Any] = []
        self.last_tool_use: List[Dict[str, Any]] = []
        # Adaptive-mode thinking blocks are encrypted — the `thinking` text
        # field is empty and the reasoning is returned as an opaque `signature`.
        # We track presence and signature length as the only observable proxy.
        self.last_thinking_present: bool = False
        self.last_thinking_signature_len: int = 0
        self.last_thinking_tokens: Optional[int] = None  # kept for back-compat

        # Prompt-cache token counters, kept separate from total_prompt_tokens
        # because they price at 1.25x (write) and 0.10x (read) respectively.
        self.total_cache_write_tokens = 0
        self.total_cache_read_tokens = 0

        # 2026 list price, USD per 1M tokens.
        self.pricing = {
            "claude-opus-4-7":   {"prompt": 15.0, "completion": 75.0},
            "claude-sonnet-4-6": {"prompt": 3.0,  "completion": 15.0},
            "claude-haiku-4-5":  {"prompt": 1.0,  "completion": 5.0},
        }

    def completion(
        self,
        prompt: str,
        schema: Optional[Dict[str, Any]] = None,
        force_json: bool = False,
        system: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        cache: bool = False,
        cache_prompt: bool = False,
    ) -> Dict[str, Any]:
        self.last_prompt = prompt

        # When the user prompt itself is static across trials (e.g. N=30 batch
        # runs where only temperature noise differs), wrap it as a content-block
        # list so we can attach cache_control. User-message caching is a first-
        # class Anthropic feature — same 5-min TTL, same 0.10x read / 1.25x
        # write pricing as system/tools caching.
        if cache_prompt:
            user_content: Any = [{
                "type": "text",
                "text": prompt,
                "cache_control": {"type": "ephemeral"},
            }]
        else:
            user_content = prompt
        messages: List[Dict[str, Any]] = [{"role": "user", "content": user_content}]

        # Anthropic has no response_format={"type":"json_object"}. The idiomatic
        # equivalent is assistant prefill: we seed the assistant turn with "{"
        # so the model continues inside a JSON object. The prefill token is NOT
        # echoed in resp.content, so we re-prepend "{" before parsing.
        # Extended thinking forbids assistant prefill, so skip it when enabled.
        want_json = bool(schema or force_json)
        thinking_on = self.thinking_effort is not None
        prefill = "{"
        if want_json and not thinking_on:
            messages.append({"role": "assistant", "content": prefill})

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": 1.0 if thinking_on else self.temperature,
            "messages": messages,
        }
        if thinking_on:
            kwargs["thinking"] = {"type": "adaptive"}
            kwargs["output_config"] = {"effort": self.thinking_effort}
            # Thinking at effort=high can exceed the SDK's 10-min sync cap;
            # bump timeout explicitly so we stay in non-streaming mode.
            kwargs["timeout"] = 1800.0
        # Prompt caching: mark system and the tail of tools as cache breakpoints.
        # Breakpoints cache everything from request start up through that block,
        # so marking system alone caches just system; marking the last tool also
        # captures all tools above it. Blocks shorter than the per-model minimum
        # (Haiku 2048, Sonnet/Opus 1024) are silently ignored by the API.
        if system is not None:
            if cache:
                kwargs["system"] = [{
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }]
            else:
                kwargs["system"] = system
        if tools is not None:
            if cache and tools:
                tools_marked = [dict(t) for t in tools]
                tools_marked[-1] = {
                    **tools_marked[-1],
                    "cache_control": {"type": "ephemeral"},
                }
                kwargs["tools"] = tools_marked
            else:
                kwargs["tools"] = tools

        resp = self.client.messages.create(**kwargs)

        self.last_stop_reason = resp.stop_reason
        self.last_content_blocks = list(resp.content)

        text_parts: List[str] = []
        tool_uses: List[Dict[str, Any]] = []
        thinking_present = False
        thinking_sig_len = 0
        for block in resp.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(block.text)
            elif btype == "tool_use":
                tool_uses.append({
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })
            elif btype == "thinking":
                thinking_present = True
                # Adaptive mode returns an opaque `signature`; legacy `enabled`
                # mode returned visible `thinking` text. Track both paths.
                thinking_sig_len += len(getattr(block, "signature", "") or "")
                thinking_sig_len += len(getattr(block, "thinking", "") or "")
        self.last_tool_use = tool_uses
        self.last_thinking_present = thinking_present
        self.last_thinking_signature_len = thinking_sig_len
        self.last_thinking_tokens = thinking_sig_len if thinking_present else None
        text = "".join(text_parts)

        if resp.usage:
            in_tok = resp.usage.input_tokens
            out_tok = resp.usage.output_tokens
            # input_tokens already excludes cache-hit and cache-write tokens;
            # the three counters are disjoint and billed at different rates.
            cache_write = getattr(resp.usage, "cache_creation_input_tokens", 0) or 0
            cache_read = getattr(resp.usage, "cache_read_input_tokens", 0) or 0
            self.total_prompt_tokens += in_tok
            self.total_completion_tokens += out_tok
            self.total_cache_write_tokens += cache_write
            self.total_cache_read_tokens += cache_read
            self.last_usage = {
                "prompt_tokens": in_tok,
                "completion_tokens": out_tok,
                "cache_write_tokens": cache_write,
                "cache_read_tokens": cache_read,
                "total_tokens": in_tok + out_tok + cache_write + cache_read,
            }

        # Tool-use path: surface the native blocks instead of forcing them
        # through JSON extraction, so callers can dispatch on tool name/input
        # without reparsing stringified arguments.
        if tool_uses:
            self.last_response = text
            return {
                "tool_use": tool_uses,
                "text": text,
                "stop_reason": resp.stop_reason,
            }

        if want_json:
            text = prefill + text
        self.last_response = text
        return self._parse_json_safely(text)

    def get_total_cost(self) -> float:
        prices = self.pricing.get(self.model, {"prompt": 0.0, "completion": 0.0})
        p_in = prices["prompt"]
        cost_prompt = (self.total_prompt_tokens / 1_000_000) * p_in
        cost_cache_write = (self.total_cache_write_tokens / 1_000_000) * p_in * 1.25
        cost_cache_read = (self.total_cache_read_tokens / 1_000_000) * p_in * 0.10
        cost_comp = (self.total_completion_tokens / 1_000_000) * prices["completion"]
        return cost_prompt + cost_cache_write + cost_cache_read + cost_comp

    def _parse_json_safely(self, text: str) -> Dict[str, Any]:
        if text is None:
            return {"error": "Response content was None"}
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        match_broad = re.search(r'(\{.*\})', text, re.DOTALL)
        if match_broad:
            try:
                return json.loads(match_broad.group(1))
            except json.JSONDecodeError:
                pass

        return {"raw_text": text, "error": "Could not extract valid JSON object from response."}
