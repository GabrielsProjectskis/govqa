"""One small wrapper around LiteLLM so the rest of the code is provider independent.

Set LLM_MODEL in `.env` to any LiteLLM model string and put that provider's API key next to it
(GEMINI_API_KEY, OPENAI_API_KEY, MISTRAL_API_KEY, ...). See `.env.example`.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

from govqa import config


@dataclass
class LLMResult:
    text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_s: float


def llm_available() -> bool:
    return bool(config.LLM_MODEL)


def complete(
    messages: list[dict[str, str]],
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 700,
    **kwargs,
) -> LLMResult:
    import litellm

    model = model or config.LLM_MODEL
    if not model:
        raise RuntimeError("LLM_MODEL is not set. Copy .env.example to .env and fill it in.")
    start = time.perf_counter()
    resp = litellm.completion(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        num_retries=3,
        **kwargs,
    )
    latency = time.perf_counter() - start
    try:
        cost = float(litellm.completion_cost(completion_response=resp) or 0.0)
    except Exception:  # model missing from LiteLLM's price table
        cost = 0.0
    usage = resp.usage
    return LLMResult(
        text=(resp.choices[0].message.content or "").strip(),
        input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
        output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
        cost_usd=cost,
        latency_s=latency,
    )


def parse_json(text: str) -> dict:
    """Pull the first JSON object out of a model reply (tolerates code fences and chatter)."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object in model reply: {text[:200]!r}")
    return json.loads(match.group(0))
