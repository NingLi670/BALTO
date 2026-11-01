"""Dataset-level paper Faith., Info. and Q-Score in percentages."""

import argparse
import math

from evaluation.common import read_records, write_json


def aggregate_metrics(records):
    if not records:
        raise ValueError("Cannot evaluate an empty set")
    for record in records:
        if record.get("status") != "ok":
            raise ValueError("All detections must succeed before computing dataset metrics")
        for key in ("faithfulness", "informativeness"):
            value = record.get(key)
            if value is None or not math.isfinite(float(value)) or value not in (0, 1):
                raise ValueError(f"Missing or invalid {key}; Info. requires base-model reference claim counts")
    size = len(records)
    faith = sum(record["faithfulness"] for record in records) / size
    info = sum(record["informativeness"] for record in records) / size
    return {
        "n": size,
        "Faith.": 100 * faith,
        "Info.": 100 * info,
        "Q-Score": 100 * faith * info,
        "no_claim_responses": sum(record.get("total_claims") == 0 for record in records),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    metrics = aggregate_metrics(read_records(args.input))
    write_json(args.output, metrics)
    print(metrics)


if __name__ == "__main__":
    main()
