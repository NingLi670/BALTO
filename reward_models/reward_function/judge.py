"""OpenAI-compatible Judge client and validated claim output."""

import json
import os
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from urllib.parse import urlsplit

from reward_models.reward_function.prompt import render_prompt


def final_answer(text: str) -> tuple[str, int]:
    """Return the answer and its offset in the complete rollout text."""
    closing = re.search(r"</think>", text, flags=re.IGNORECASE)
    if closing:
        return text[closing.end() :], closing.end()
    if re.search(r"<think>", text, flags=re.IGNORECASE):
        raise ValueError("Response has an unclosed thinking block")
    return text, 0


def parse_claims(raw: str) -> list[dict]:
    """Read the details JSON embedded in the original detection templates.

    Invalid verdicts or missing claim text fail explicitly. An explicit empty
    details list is valid; malformed output never becomes a faithful response.
    """
    if not isinstance(raw, str):
        raise ValueError("Judge must return text")
    decoder = json.JSONDecoder()
    value = None
    opening = re.compile(r'\{\s*"details"\s*:')
    for match in opening.finditer(raw):
        try:
            candidate, _ = decoder.raw_decode(raw[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and "details" in candidate:
            value = candidate
            break
    if value is None or not isinstance(value["details"], list):
        raise ValueError("Judge response must contain a JSON object with a details list")
    claims = []
    for detail in value["details"]:
        if not isinstance(detail, dict):
            raise ValueError("Judge details must contain objects")
        text = detail.get("claim_text")
        verdict = detail.get("judgment_result")
        if not isinstance(text, str) or not text or verdict not in {"Correct", "Incorrect"}:
            raise ValueError("Each claim requires claim_text and Correct/Incorrect judgment_result")
        # Missing/invalid error fragments are passed through for whole-claim
        # fallback in the shared BALTO projection; no replacement text is used.
        claims.append({"claim_text": text, "judgment_result": verdict, "error_spans": detail.get("error_spans", [])})
    return claims


@dataclass(frozen=True)
class JudgeConfig:
    base_url: str
    model: str
    api_key_env: str = "BALTO_JUDGE_API_KEY"
    timeout: float = 120.0
    retries: int = 3
    max_tokens: int = 8192
    temperature: float = 0.0
    enable_thinking: bool | None = None


class JudgeClient:
    def __init__(self, config: JudgeConfig):
        if not config.base_url or not config.model:
            raise ValueError("Judge base_url and model must be configured")
        parts = urlsplit(config.base_url)
        if parts.scheme not in {"http", "https"} or not parts.hostname:
            raise ValueError("Judge base_url must be an HTTP(S) service URL")
        if parts.username is not None or parts.password is not None or parts.query or parts.fragment:
            raise ValueError("Judge credentials must use environment variables, not URL components")
        if config.retries < 1 or config.timeout <= 0 or config.max_tokens < 1:
            raise ValueError("Judge retries, timeout and max_tokens must be positive")
        from openai import OpenAI

        self.config = config
        self.client = OpenAI(
            base_url=config.base_url,
            api_key=os.environ.get(config.api_key_env) or "EMPTY",
            timeout=config.timeout,
            max_retries=0,
        )

    def complete(self, prompt: str) -> str:
        kwargs = {}
        if self.config.enable_thinking is not None:
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": self.config.enable_thinking}}
        result = self.client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            **kwargs,
        )
        if result.choices[0].finish_reason == "length":
            raise ValueError("Judge output reached max_tokens")
        content = result.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Judge returned no content")
        return content

    def validated_completion(self, prompt: str, parser):
        """Retry API and output-validation failures without logging credentials."""
        for attempt in range(self.config.retries):
            try:
                raw = self.complete(prompt)
                return parser(raw), raw
            except Exception:
                if attempt + 1 == self.config.retries:
                    raise RuntimeError(f"Judge failed after {self.config.retries} attempts") from None
                time.sleep(min(2**attempt, 8))

    def detect(self, task: str, query: str, response: str, ground_truth=None):
        answer, offset = final_answer(response)
        prompt = render_prompt(task, query, answer, ground_truth)
        claims, raw = self.validated_completion(prompt, parse_claims)
        return claims, raw, offset


@lru_cache(maxsize=8)
def get_judge(config: JudgeConfig) -> JudgeClient:
    return JudgeClient(config)


def task_for_source(source: str) -> str:
    source = source.lower()
    if "confiqa" in source:
        return "confiqa"
    if "ragtruth" in source:
        return "ragtruth"
    if source in {"financial", "finllm"}:
        return "financial"
    raise ValueError(f"Unknown data_source {source!r}; specify task explicitly")
