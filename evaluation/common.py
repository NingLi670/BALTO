"""Shared record I/O for inference and evaluation."""

import json
from pathlib import Path


def json_default(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def to_python(value):
    if isinstance(value, dict):
        return {key: to_python(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [to_python(item) for item in value]
    if hasattr(value, "tolist"):
        return to_python(value.tolist())
    return value


def read_records(path):
    path = Path(path)
    if path.suffix == ".parquet":
        import pandas as pd

        return to_python(pd.read_parquet(path).to_dict("records"))
    with path.open(encoding="utf-8") as handle:
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in handle if line.strip()]
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError("Input JSON must contain a list of records")
    return records


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8")


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=json_default) + "\n")


def record_id(record):
    for key in ("source_id", "id", "traceId"):
        value = record.get(key)
        if value is not None and str(value):
            return str(value)
    raise ValueError("Each record requires a nonempty source_id, id or traceId")


def unique_records(records):
    lookup = {}
    for record in records:
        key = record_id(record)
        if key in lookup:
            raise ValueError(f"Duplicate sample ID: {key}")
        lookup[key] = record
    return lookup


def prompt_text(record):
    prompt = record.get("prompt", record.get("input"))
    if isinstance(prompt, str):
        return prompt
    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()
    if not isinstance(prompt, list) or not prompt:
        raise ValueError("Each record needs prompt messages or input text")
    return "\n".join(message["content"] for message in prompt)


def rollout_extra(record):
    extra = dict(record.get("extra_info") or {})
    extra.setdefault("query", prompt_text(record))
    return extra


def response_text(record):
    text = record.get("output", record.get("answer", record.get("original_answer")))
    if not isinstance(text, str):
        raise ValueError("Response record requires output text")
    return text


def public_rollout_fields(record):
    """Keep fields used by evaluation, without copying arbitrary input metadata."""
    allowed = ("query", "ground_truth", "reference_claim_count", "avg_claim_num_Qwen3-8B")
    extra = dict(record.get("extra_info") or {})
    return {
        "source_id": record_id(record),
        "data_source": record.get("data_source", ""),
        "task_type": record.get("task_type", ""),
        "input": prompt_text(record),
        "extra_info": {key: extra[key] for key in allowed if key in extra},
    }
