"""Batch claim detection shared by reference statistics and offline evaluation."""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor

from evaluation.common import (
    public_rollout_fields,
    read_records,
    response_text,
    rollout_extra,
    unique_records,
    write_jsonl,
)
from reward_models.reward_function.judge import JudgeConfig, get_judge, task_for_source
from reward_models.reward_function.reward_function import claim_metrics


def detect_records(records, judge, task="auto", workers=16, save_judge_response=False):
    if workers < 1:
        raise ValueError("workers must be positive")
    unique_records(records)

    def detect(record):
        extra = rollout_extra(record)
        selected_task = task_for_source(record.get("data_source", "")) if task == "auto" else task
        ground_truth = extra.get("ground_truth")
        if hasattr(ground_truth, "tolist"):
            ground_truth = ground_truth.tolist()
        claims, raw, _ = judge.detect(selected_task, extra["query"], response_text(record), ground_truth)
        reference = extra.get("reference_claim_count", extra.get("avg_claim_num_Qwen3-8B"))
        result = {
            **public_rollout_fields(record),
            "output": response_text(record),
            "status": "ok",
            "claims": claims,
            **claim_metrics(claims, reference),
        }
        if save_judge_response:
            result["judge_response"] = raw
        return result

    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(detect, records))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task", choices=["auto", "confiqa", "ragtruth", "financial"], default="auto")
    parser.add_argument("--base_url", default=os.environ.get("BALTO_JUDGE_BASE_URL"))
    parser.add_argument("--model", default=os.environ.get("BALTO_JUDGE_MODEL"))
    parser.add_argument("--api_key_env", default="BALTO_JUDGE_API_KEY")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--thinking", choices=["true", "false"], default=None)
    parser.add_argument("--save_judge_response", action="store_true", help="Save full Judge text for local debugging")
    args = parser.parse_args()
    judge = get_judge(
        JudgeConfig(
            base_url=args.base_url,
            model=args.model,
            api_key_env=args.api_key_env,
            timeout=args.timeout,
            retries=args.retries,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            enable_thinking=None if args.thinking is None else args.thinking == "true",
        )
    )
    rows = detect_records(read_records(args.input), judge, args.task, args.workers, args.save_judge_response)
    write_jsonl(args.output, rows)
    print(f"Saved {len(rows)} detections to {args.output}")


if __name__ == "__main__":
    main()
