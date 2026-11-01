# Copyright 2026 BALTO contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""BALTO token labels, balanced advantages (Eq. 5), and policy objective (Eq. 7).

These functions require only NumPy and PyTorch; Judge calls belong to the custom
reward function. Indices must refer to the original rollout response tokens.
"""

from collections.abc import Sequence
from numbers import Integral
from typing import Any

import numpy as np
import torch


def build_token_mask(raw_indices: Any, response_mask: torch.Tensor) -> torch.Tensor:
    """Convert per-response index lists or a dense Tensor/ndarray to a bool mask.

    Lists always contain indices, even when an index list is [0, 1]. Dense masks
    must be two-dimensional Tensors or non-object ndarrays of binary values.
    Padding is excluded. Missing/malformed annotations raise instead of silently
    skipping the BALTO update; an explicit empty index list means no labeled token.
    """
    if response_mask.ndim != 2:
        raise ValueError("response_mask must have shape [batch_size, response_length]")
    if raw_indices is None:
        raise ValueError("BALTO requires explicit token annotations for every response")

    if isinstance(raw_indices, torch.Tensor) or (
        isinstance(raw_indices, np.ndarray) and raw_indices.dtype != object and raw_indices.ndim == 2
    ):
        dense = torch.as_tensor(raw_indices, device=response_mask.device)
        if dense.shape != response_mask.shape or not torch.all((dense == 0) | (dense == 1)):
            raise ValueError("Dense token masks must match response_mask and contain only 0/1")
        return dense.bool() & response_mask.bool()

    if isinstance(raw_indices, np.ndarray):
        raw_indices = raw_indices.tolist()
    if not isinstance(raw_indices, (list, tuple)) or len(raw_indices) != response_mask.shape[0]:
        raise ValueError("Token index lists must contain one list per response")

    mask = torch.zeros_like(response_mask, dtype=torch.bool)
    for row, indices in enumerate(raw_indices):
        if isinstance(indices, (np.ndarray, torch.Tensor)):
            indices = indices.tolist()
        if not isinstance(indices, (list, tuple)):
            raise ValueError(f"Token indices for response {row} must be a list")
        for index in indices:
            if isinstance(index, (bool, np.bool_)) or not isinstance(index, Integral):
                raise ValueError(f"Token index for response {row} must be an integer: {index!r}")
            if not 0 <= index < response_mask.shape[1]:
                raise ValueError(f"Token index {index} is outside response length {response_mask.shape[1]}")
            mask[row, index] = True
    return mask & response_mask.bool()


def project_claims_to_token_masks(
    response_text: str,
    claims: Sequence[dict[str, Any]],
    token_offsets: Sequence[tuple[int, int]],
    preferred_start_char: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project Judge claims onto offsets aligned to the original response (Eq. 3).

    Each claim has claim_text, judgment_result (Correct/Incorrect), and optionally
    error_spans. Repeated claims are located in response order. Error fragments
    are searched only within their claim, so identical text in another claim is
    not penalized. If any fragment cannot be localized, the whole incorrect
    claim is penalized. Supported portions of an incorrect claim remain neutral.
    """
    hallucinated = torch.zeros(len(token_offsets), dtype=torch.bool)
    faithful = torch.zeros_like(hallucinated)
    if not 0 <= preferred_start_char <= len(response_text):
        raise ValueError("preferred_start_char must be a valid response character offset")
    cursor = preferred_start_char
    previous_occurrences: dict[str, int] = {}

    def mark(mask: torch.Tensor, start: int, end: int) -> None:
        for index, (token_start, token_end) in enumerate(token_offsets):
            if token_end > token_start and token_start < end and token_end > start:
                mask[index] = True

    for claim in claims:
        text = claim.get("claim_text")
        verdict = claim.get("judgment_result")
        if not isinstance(text, str) or not text or verdict not in ("Correct", "Incorrect"):
            raise ValueError("Each claim requires nonempty claim_text and Correct/Incorrect judgment_result")
        search_start = max(cursor, previous_occurrences.get(text, -1) + 1)
        start = response_text.find(text, search_start)
        if start < 0:
            raise ValueError("Judge claim_text must be an exact response substring in response order")
        end = start + len(text)
        # Allow overlapping claims; the next claim may begin inside this one.
        cursor = start
        previous_occurrences[text] = start
        if verdict == "Correct":
            mark(faithful, start, end)
            continue

        fragments = claim.get("error_spans")
        localized = []
        if isinstance(fragments, list) and fragments:
            for fragment in fragments:
                if not isinstance(fragment, str) or not fragment:
                    localized = []
                    break
                fragment_start = text.find(fragment)
                if fragment_start < 0:
                    localized = []
                    break
                if text.find(fragment, fragment_start + 1) >= 0:
                    # An ambiguous repeated fragment cannot be reliably isolated.
                    localized = []
                    break
                localized.append((start + fragment_start, start + fragment_start + len(fragment)))
        for fragment_start, fragment_end in localized or [(start, end)]:
            mark(hallucinated, fragment_start, fragment_end)

    return hallucinated, faithful & ~hallucinated


@torch.no_grad()
def compute_balto_advantages(
    hallucination_mask: torch.Tensor, correct_mask: torch.Tensor, response_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assign -1 to H, |H|/|F| to F, and 0 to N, independently per response.

    H takes precedence over F. No errors means zero advantages. If F is empty,
    only the negative correction remains. The ratio is neither centered nor
    clipped: when both sets exist, their total advantages cancel exactly.
    Returns float32 advantages and the factual mask H union F for normalization.
    """
    if (
        response_mask.ndim != 2
        or hallucination_mask.shape != response_mask.shape
        or correct_mask.shape != response_mask.shape
    ):
        raise ValueError("All BALTO masks must have the same [batch_size, response_length] shape")
    valid = response_mask.bool()
    hallucinated = hallucination_mask.to(device=valid.device).bool() & valid
    faithful = correct_mask.to(device=valid.device).bool() & valid & ~hallucinated
    n_error = hallucinated.sum(dim=-1, keepdim=True).float()
    n_faithful = faithful.sum(dim=-1, keepdim=True).float()
    advantages = -hallucinated.float() + faithful.float() * (n_error / n_faithful.clamp_min(1))
    return advantages, hallucinated | faithful


def apply_balto_advantages(
    data: Any,
    hallucination_mask_key: str = "hallucination_token_indices",
    correct_token_mask_key: str = "correct_token_indices",
) -> Any:
    """Attach advantages and factual mask to a DataProto, without group rewards."""
    response_mask = data.batch["response_mask"]
    masks = []
    for key in (hallucination_mask_key, correct_token_mask_key):
        if key in data.batch:
            raw_mask = data.batch[key]
        elif key in data.non_tensor_batch:
            raw_mask = data.non_tensor_batch[key]
        else:
            raise ValueError(f"BALTO reward function did not provide required field {key!r}")
        masks.append(build_token_mask(raw_mask, response_mask))
    advantages, factual_mask = compute_balto_advantages(*masks, response_mask)
    data.batch["advantages"] = advantages
    data.batch["returns"] = advantages.clone()
    data.batch["balto_factual_mask"] = factual_mask
    return data


def compute_balto_policy_loss(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    factual_mask: torch.Tensor,
    clip_ratio_low: float = 0.2,
    clip_ratio_high: float = 0.28,
    sequence_normalizer: float | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Negative Eq. 7: clipped token objective / factual count, then mean responses.

    Fully faithful and annotation-free responses stay in the response average,
    contributing zero. sequence_normalizer defaults to the microbatch size;
    Megatron supplies local_minibatch_size / num_microbatches to account for its
    schedule's division by num_microbatches, including uneven dynamic batches.
    There is no dual clipping, response reward, KL loss, or distillation term.
    """
    if log_prob.ndim != 2 or any(tensor.shape != log_prob.shape for tensor in (old_log_prob, advantages, factual_mask)):
        raise ValueError("BALTO loss inputs must have the same [batch_size, response_length] shape")
    if not 0 <= clip_ratio_low < 1 or clip_ratio_high < 0:
        raise ValueError("Clipping requires 0 <= clip_ratio_low < 1 and clip_ratio_high >= 0")
    denominator = log_prob.shape[0] if sequence_normalizer is None else sequence_normalizer
    if denominator <= 0:
        raise ValueError("sequence_normalizer must be positive")

    # Zero out inactive positions before arithmetic so padding/neutral NaNs
    # cannot pollute an otherwise valid update.
    factual = factual_mask.bool()
    active = factual & (advantages != 0)
    log_ratio = torch.where(active, log_prob - old_log_prob, torch.zeros_like(log_prob)).float()
    ratio = log_ratio.clamp(-20, 20).exp()
    advantage = torch.where(active, advantages, torch.zeros_like(advantages)).float()
    unclipped = ratio * advantage
    clipped = ratio.clamp(1 - clip_ratio_low, 1 + clip_ratio_high) * advantage
    token_loss = -torch.minimum(unclipped, clipped)
    per_response = token_loss.sum(dim=-1) / factual.sum(dim=-1).clamp_min(1)
    loss = per_response.sum() / denominator
    with torch.no_grad():
        clipped_fraction = ((unclipped > clipped) & active).float().sum() / active.sum().clamp_min(1)
    return loss, {"actor/pg_clipfrac": clipped_fraction.item()}
