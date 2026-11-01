"""ConFiQA paper accuracy: (Pc - Po + 100) / 2, using percentage recall scores."""

import argparse
import re
import string
from collections import Counter

from evaluation.common import prompt_text, read_records, record_id, response_text, unique_records, write_json
from reward_models.reward_function.judge import final_answer


def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


NEGATION_WORDS = [
    "no",
    "not",
    "never",
    "none",
    "cannot",
    "nobody",
    "nothing",
    "nowhere",
    "neither",
    "nor",
    "without",
    "hardly",
]


def _has_negation(prediction: str) -> bool:
    return any(w in normalize_answer(prediction).split() for w in NEGATION_WORDS)


def exact_match_score(prediction: str, gt: str, is_cf: bool) -> int:
    contains_negation = _has_negation(prediction)
    return int((not contains_negation if is_cf else True) and (normalize_answer(prediction) == normalize_answer(gt)))


def recall_score(prediction: str, gt: str, is_cf: bool) -> int:
    np_pred = normalize_answer(prediction)
    np_gt = normalize_answer(gt)
    contains_negation = _has_negation(prediction)
    return int((np_gt in np_pred) and (not contains_negation if is_cf else True))


def get_score(preds: list, golds: list, origs: list) -> tuple[float, float, float]:
    em, gold_recall, orig_recall = 0, 0, 0
    n = len(preds)
    for pred, gold, orig in zip(preds, golds, origs):
        if isinstance(gold, list):
            _em = max((exact_match_score(pred, g, True) for g in gold), default=0)
            _recall = max((recall_score(pred, g, True) for g in gold), default=0)
        else:
            _em = exact_match_score(pred, gold, True)
            _recall = recall_score(pred, gold, True)
        if isinstance(orig, list):
            _recall_orig = max((recall_score(pred, o, False) for o in orig), default=0)
        else:
            _recall_orig = recall_score(pred, orig, False)
        em += _em
        gold_recall += int(_recall and not _recall_orig)
        orig_recall += _recall_orig
    em = em * 100 / max(n, 1)
    gold_recall = gold_recall * 100 / max(n, 1)
    orig_recall = orig_recall * 100 / max(n, 1)
    return em, gold_recall, orig_recall


def evaluate_records(sources, outputs):
    if not sources or not outputs:
        raise ValueError("ConFiQA evaluation requires a nonempty source and response set")
    lookup = unique_records(sources)
    by_question = {}
    for source in sources:
        extra = source.get("extra_info") or {}
        question = source.get("question", extra.get("question"))
        if question:
            if question in by_question:
                raise ValueError("Duplicate source question prevents reliable rollout matching")
            by_question[question] = source
    predictions, golds, originals, detailed = [], [], [], []
    for output in outputs:
        try:
            source_id = record_id(output)
        except ValueError:
            source_id = None
        if source_id is not None:
            if source_id not in lookup:
                raise ValueError(f"Rollout ID {source_id} is absent from the source dataset")
            source = lookup[source_id]
        else:
            # Upstream verl validation dumps input/output but not always source IDs.
            match = re.search(r"Question:\s*(.+?)\s*\n", prompt_text(output))
            source = by_question.get(match.group(1).strip()) if match else None
            if source is None:
                raise ValueError("Rollout has neither a known ID nor an unambiguous source question")
            source_id = record_id(source)
        extra = source.get("extra_info") or source.get("meta") or source
        gold = [extra.get("cf_answer"), *list(extra.get("cf_alias") or [])]
        original = [extra.get("orig_answer"), *list(extra.get("orig_alias") or [])]
        gold = [answer for answer in gold if isinstance(answer, str) and normalize_answer(answer)]
        original = [answer for answer in original if isinstance(answer, str) and normalize_answer(answer)]
        if not gold:
            raise ValueError(f"No context-faithful answer for sample {source_id}")
        prediction, _ = final_answer(response_text(output))
        predictions.append(prediction)
        golds.append(gold)
        originals.append(original)
        em, pc, po = get_score([prediction], [gold], [original])
        detailed.append({"source_id": source_id, "EM": em, "Pc": pc, "Po": po, "Acc.": (pc - po + 100) / 2})
    covered = {row["source_id"] for row in detailed}
    if covered != set(lookup):
        raise ValueError("Response set must cover every source sample")
    # Multiple rollouts are allowed only with the same count for every prompt.
    counts = Counter(row["source_id"] for row in detailed)
    if len(set(counts.values())) != 1:
        raise ValueError("Each source sample must have the same number of responses")
    em, pc, po = get_score(predictions, golds, originals)
    return {
        "n": len(predictions),
        "metrics": {"EM": em, "Pc": pc, "Po": po, "Acc.": (pc - po + 100) / 2},
        "detailed": detailed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = evaluate_records(read_records(args.source), read_records(args.input))
    write_json(args.output, result)
    print(result["metrics"])


if __name__ == "__main__":
    main()
