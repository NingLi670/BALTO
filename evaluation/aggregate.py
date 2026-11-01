"""Collect metric summaries into CSV; do not average different test sets."""

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = []
    for filename in args.inputs:
        summary = json.loads(Path(filename).read_text())
        metrics = summary.get("metrics", summary)
        row = {"file": filename}
        row.update({key: value for key, value in metrics.items() if isinstance(value, (int, float, str))})
        rows.append(row)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
