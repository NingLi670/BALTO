"""Map decoded characters to the original rollout token IDs.

The fast path retains tokenizer offsets only after an exact ID comparison.
The research prefix-decode bridge is used on mismatch, with content validation
and overlapping coverage for characters produced by multiple byte tokens.
"""

from functools import lru_cache
from numbers import Integral


@lru_cache(maxsize=4)
def load_tokenizer(path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path, use_fast=True)


def original_token_offsets(response: str, original_ids, tokenizer):
    if original_ids is None:
        raise ValueError("BALTO reward requires original_response_token_ids")
    ids = list(original_ids)
    if any(isinstance(i, bool) or not isinstance(i, Integral) or i < 0 for i in ids):
        raise ValueError("Original token IDs must be nonnegative integers")
    decoded = tokenizer.decode(ids, skip_special_tokens=True)
    if decoded != response:
        raise ValueError("Reward tokenizer does not decode original IDs to the rollout text")
    try:
        encoding = tokenizer(response, add_special_tokens=False, return_offsets_mapping=True)
    except (NotImplementedError, ValueError, TypeError):
        encoding = {}
    if list(encoding.get("input_ids", [])) == ids and "offset_mapping" in encoding:
        offsets = [tuple(pair) for pair in encoding["offset_mapping"]]
        if len(offsets) != len(ids) or any(not 0 <= a <= b <= len(response) for a, b in offsets):
            raise ValueError("Tokenizer returned invalid character offsets")
        return offsets, "exact"

    boundaries = [0]
    for end in range(1, len(ids) + 1):
        prefix = tokenizer.decode(ids[:end], skip_special_tokens=True)
        stable = 0
        for a, b in zip(prefix, response):
            if a != b:
                break
            stable += 1
        if stable < boundaries[-1]:
            raise ValueError("Original token prefixes cannot be aligned monotonically")
        boundaries.append(stable)
    if boundaries[-1] != len(response):
        raise ValueError("Original token alignment did not cover the response")

    offsets = []
    special_ids = set(getattr(tokenizer, "all_special_ids", []))
    next_end = len(response)
    # UTF-8 byte tokens may decode to replacement characters until the final
    # byte arrives. All contributing tokens must cover the resulting character.
    for index in range(len(ids) - 1, -1, -1):
        start, end = boundaries[index : index + 2]
        if end > start:
            next_end = end
        if start == end and ids[index] not in special_ids:
            if start == len(response):
                raise ValueError("A trailing non-special token has no character coverage")
            end = next_end
        offsets.append((start, end))
    return list(reversed(offsets)), "prefix_decode"
