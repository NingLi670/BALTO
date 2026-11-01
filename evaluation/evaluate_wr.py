"""Paper WR: two orderings per sample, ties contribute zero wins."""

import argparse
import os
import re
from concurrent.futures import ThreadPoolExecutor

from evaluation.common import prompt_text, read_records, response_text, unique_records, write_json
from reward_models.reward_function.judge import JudgeConfig, final_answer, get_judge

# English response comparison template from the paper appendix.
COMPARISON_PROMPT = '''You are a helpful assistant, that ranks models by the quality of their answers.

I want you to create a leaderboard of different of large-language models. To do so, I will give you the instructions (prompts) given to the models, and the responses of two models. Please rank the models based on which responses would be preferred by humans. All inputs and outputs should be python dictionaries.

Here is the prompt:
{
    "instruction": """{{instruction}}"""
}

Here are the outputs of the models:
[
    {
        "model": "model_1",
        "answer": """{{output_1}}"""
    },
    {
        "model": "model_2",
        "answer": """{{output_2}}"""
    }
]

Now please analyze the two models' answers and return the better model.

Output format:

### Analysis
xxx (your analysis)

### Better Model
model_1 / model_2 / tie (The model you think is better. If you think they are equally good, return "tie")
'''


def comparison_prompt(instruction, first, second):
    values = {"instruction": instruction, "output_1": first, "output_2": second}
    return re.sub(r"\{\{(instruction|output_1|output_2)\}\}", lambda m: values[m.group(1)], COMPARISON_PROMPT)


def parse_winner(raw):
    match = re.search(r"###\s*Better Model\s*\n\s*(model_1|model_2|tie)\s*$", raw.strip(), re.IGNORECASE)
    if not match:
        raise ValueError("WR Judge must return a single model_1, model_2 or tie under ### Better Model")
    return match.group(1).lower()


def evaluate_pairs(candidate_records, reference_records, judge, workers=8, save_judge_response=False):
    candidate, reference = unique_records(candidate_records), unique_records(reference_records)
    if not candidate or set(candidate) != set(reference):
        raise ValueError("Candidate and reference must have the same nonempty set of sample IDs")
    if workers < 1:
        raise ValueError("workers must be positive")

    def compare(source_id):
        instruction = prompt_text(candidate[source_id])
        if instruction != prompt_text(reference[source_id]):
            raise ValueError(f"Different prompts for sample {source_id}")
        candidate_answer, _ = final_answer(response_text(candidate[source_id]))
        reference_answer, _ = final_answer(response_text(reference[source_id]))
        results = []
        for direction, first, second, candidate_slot in (
            ("forward", candidate_answer, reference_answer, "model_1"),
            ("reverse", reference_answer, candidate_answer, "model_2"),
        ):
            winner, raw = judge.validated_completion(comparison_prompt(instruction, first, second), parse_winner)
            result = {"direction": direction, "winner": winner, "candidate_win": int(winner == candidate_slot)}
            if save_judge_response:
                result["judge_response"] = raw
            results.append(result)
        return {
            "source_id": source_id,
            "win": sum(result["candidate_win"] for result in results) / 2,
            "comparisons": results,
        }

    with ThreadPoolExecutor(max_workers=workers) as executor:
        detailed = list(executor.map(compare, candidate))
    return {
        "n": len(detailed),
        "WR": 100 * sum(row["win"] for row in detailed) / len(detailed),
        "ties": sum(r["winner"] == "tie" for row in detailed for r in row["comparisons"]),
        "tie_policy": "zero candidate wins; retained in the denominator",
        "detailed": detailed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base_url", default=os.environ.get("BALTO_WR_BASE_URL"))
    parser.add_argument("--model", default=os.environ.get("BALTO_WR_MODEL"))
    parser.add_argument("--api_key_env", default="BALTO_WR_API_KEY")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max_tokens", type=int, default=2048)
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
        )
    )
    result = evaluate_pairs(
        read_records(args.candidate), read_records(args.reference), judge, args.workers, args.save_judge_response
    )
    write_json(args.output, result)
    print({key: value for key, value in result.items() if key != "detailed"})


if __name__ == "__main__":
    main()
