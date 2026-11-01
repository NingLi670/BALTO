"""Generate ordered responses from prepared Parquet or message records."""

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor

from evaluation.common import prompt_text, public_rollout_fields, read_records, record_id, unique_records, write_jsonl


def generate_records(
    records, client, model, workers=8, temperature=0.6, max_tokens=8192, retries=3, enable_thinking=None
):
    unique_records(records)
    if workers < 1 or retries < 1 or max_tokens < 1:
        raise ValueError("workers, retries and max_tokens must be positive")

    def generate(record):
        messages = record.get("prompt")
        if hasattr(messages, "tolist"):
            messages = messages.tolist()
        if isinstance(messages, str) or messages is None:
            messages = [{"role": "user", "content": prompt_text(record)}]
        kwargs = {}
        if enable_thinking is not None:
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": enable_thinking}}
        for attempt in range(retries):
            try:
                result = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                )
                content = result.choices[0].message.content
                if not isinstance(content, str):
                    raise ValueError("Generation returned no content")
                return {
                    **public_rollout_fields(record),
                    "output": content,
                    "finish_reason": result.choices[0].finish_reason,
                }
            except Exception:
                if attempt + 1 == retries:
                    raise RuntimeError(
                        f"Generation failed for sample {record_id(record)} after {retries} attempts"
                    ) from None
                time.sleep(min(2**attempt, 8))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(generate, records))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base_url", default=os.environ.get("BALTO_GENERATION_BASE_URL"))
    parser.add_argument("--api_key_env", default="BALTO_GENERATION_API_KEY")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--thinking", choices=["true", "false"], default=None, help="Optional local-server chat template parameter"
    )
    args = parser.parse_args()
    if not args.base_url:
        parser.error("--base_url or BALTO_GENERATION_BASE_URL is required")
    from openai import OpenAI

    client = OpenAI(
        base_url=args.base_url, api_key=os.environ.get(args.api_key_env) or "EMPTY", timeout=args.timeout, max_retries=0
    )
    records = generate_records(
        read_records(args.data),
        client,
        args.model,
        args.workers,
        args.temperature,
        args.max_tokens,
        args.retries,
        None if args.thinking is None else args.thinking == "true",
    )
    write_jsonl(args.output, records)
    print(f"Saved {len(records)} responses to {args.output}")


if __name__ == "__main__":
    main()
