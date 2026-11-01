# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright 2026 BALTO contributors
# SPDX-License-Identifier: Apache-2.0

"""Batch reward adapter for paper BALTO; scalar values are logging metrics."""

import math
import os
from concurrent.futures import ThreadPoolExecutor

from reward_models.reward_function.alignment import load_tokenizer, original_token_offsets
from reward_models.reward_function.judge import JudgeConfig, get_judge, task_for_source


def claim_metrics(claims, reference_claim_count=None):
    total = len(claims)
    errors = sum(claim["judgment_result"] == "Incorrect" for claim in claims)
    info = None
    if reference_claim_count is not None:
        reference = float(reference_claim_count)
        if not math.isfinite(reference) or reference < 0:
            raise ValueError("reference_claim_count must be finite and nonnegative")
        info = float(total >= reference)
    return {
        "faithfulness": float(errors == 0),
        "claim_accuracy": (total - errors) / total if total else 0.0,
        "informativeness": info,
        "total_claims": total,
        "error_claims": errors,
    }


def annotate_tokens(response, claims, original_ids, tokenizer, preferred_start_char=0):
    from verl.trainer.ppo.balto import project_claims_to_token_masks

    offsets, path = original_token_offsets(response, original_ids, tokenizer)
    hallucinated, faithful = project_claims_to_token_masks(response, claims, offsets, preferred_start_char)
    return hallucinated.nonzero().flatten().tolist(), faithful.nonzero().flatten().tolist(), path


def compute_score_batch(
    data_sources,
    solution_strs,
    ground_truths,
    extra_infos,
    tokenizer_path=None,
    judge_base_url=None,
    judge_model=None,
    task="auto",
    workers=16,
    timeout=120.0,
    retries=3,
    max_tokens=8192,
    temperature=0.0,
    enable_thinking=None,
    api_key_env="BALTO_JUDGE_API_KEY",
):
    size = len(solution_strs)
    if any(len(values) != size for values in (data_sources, ground_truths, extra_infos)):
        raise ValueError("All batch reward inputs must have the same length")
    if int(workers) < 1:
        raise ValueError("workers must be positive")
    if not tokenizer_path:
        raise ValueError("tokenizer_path must match the rollout model tokenizer")
    if not size:
        return []
    tokenizer = load_tokenizer(str(tokenizer_path))
    judge = get_judge(
        JudgeConfig(
            base_url=judge_base_url or os.environ.get("BALTO_JUDGE_BASE_URL", ""),
            model=judge_model or os.environ.get("BALTO_JUDGE_MODEL", ""),
            api_key_env=api_key_env,
            timeout=float(timeout),
            retries=int(retries),
            max_tokens=int(max_tokens),
            temperature=float(temperature),
            enable_thinking=enable_thinking,
        )
    )

    def score(index):
        extra = extra_infos[index]
        query = extra.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"Response {index} requires extra_info.query with question and context")
        selected_task = task_for_source(str(data_sources[index])) if task == "auto" else task
        ground_truth = extra.get("ground_truth", ground_truths[index])
        # Arrow list fields can arrive as ndarrays rather than Python lists.
        if hasattr(ground_truth, "tolist"):
            ground_truth = ground_truth.tolist()
        response = solution_strs[index]
        claims, _, start = judge.detect(selected_task, query, response, ground_truth)
        errors, correct, path = annotate_tokens(
            response,
            claims,
            extra.get("original_response_token_ids"),
            tokenizer,
            start,
        )
        reference = extra.get("reference_claim_count", extra.get("avg_claim_num_Qwen3-8B"))
        metrics = claim_metrics(claims, reference)
        return {
            "score": metrics["faithfulness"],
            "hallucination_token_indices": errors,
            "correct_token_indices": correct,
            "faithfulness": metrics["faithfulness"],
            "claim_accuracy": metrics["claim_accuracy"],
            "informativeness": metrics["informativeness"],
            "total_claims": metrics["total_claims"],
            "error_claims": metrics["error_claims"],
            "no_claims": int(not claims),
            "alignment_path": path,
        }

    # executor.map preserves rollout order even when Judge requests finish out of order.
    with ThreadPoolExecutor(max_workers=int(workers)) as executor:
        return list(executor.map(score, range(size)))
