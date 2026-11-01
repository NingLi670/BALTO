# BALTO: Balanced Token-Level Policy Optimization for Hallucination Mitigation

[Paper](https://arxiv.org/abs/2606.15893) · [License](LICENSE)

Hallucinations remain a major obstacle to deploying large language models (LLMs) in knowledge-intensive settings, where generated responses must be faithfully grounded in provided evidence. Reinforcement learning (RL) is a promising direction for hallucination mitigation, but response-level faithfulness rewards suffer from a granularity mismatch: localized hallucinations can cause supported content to receive spurious penalties. Although recent work introduces fine-grained feedback such as claim-level verification and token-level rewards, unbalanced credit assignment can still induce length, verbosity, or optimization-noise biases. We propose **BALTO**, a **Bal**anced **T**oken-level Policy **O**ptimization framework for hallucination mitigation. BALTO extracts checkable factual claims, verifies them against the reference context, and projects claim-level judgments to token-level labels. A balanced token-level credit assignment mechanism is introduced into the framework. This design redistributes probability mass from unsupported content toward faithful content, rather than suppressing the entire response. We systematically analyze the limitations of response-level rewards from a theoretical standpoint, and prove BALTO’s advantages in training stability and optimization efficiency for hallucination mitigation. Experiments on ConFiQA, RAGTruth, and FinLLM-Eval show that BALTO achieves the highest faithfulness across all six model--benchmark settings and consistently outperforms existing post-training baselines in Q-Score, demonstrating a stronger faithfulness--informativeness trade-off.

## Installation

Use Python 3.10+ with a compatible PyTorch/CUDA environment.

```bash
pip install -e '.[sglang,megatron]'
# For FSDP with vLLM instead:
# pip install -e '.[vllm]'
```

Megatron also requires compatible Megatron-Core dependencies; see the [verl setup guide](https://verl.readthedocs.io/).

## Training

Configure an OpenAI-compatible verifier and launch training:

```bash
export BALTO_JUDGE_BASE_URL=http://127.0.0.1:8000/v1
export BALTO_JUDGE_MODEL=your-judge-model

bash scripts/train.sh ragtruth /path/to/model \
  /path/to/train.parquet /path/to/test.parquet
```

Prepare verl-format Parquet data with `source_id`, `data_source`, `prompt`, `reward_model`, and `extra_info.query` containing the question and reference context.

Defaults use Megatron + SGLang on eight GPUs. Settings are in [configs/](configs/) and accept Hydra overrides. For FSDP + vLLM, set `BALTO_BACKEND=fsdp` and append `actor_rollout_ref.rollout.name=vllm`.

## Evaluation

Serve a Hugging Face-format model in a separate terminal:

```bash
bash scripts/deploy_generation.sh /path/to/model --port 8001
```

Generate responses and compute Faith., Info., and Q-Score using the verifier configured above:

```bash
export BALTO_GENERATION_BASE_URL=http://127.0.0.1:8001/v1

python -m inference.inference --data /path/to/test.parquet \
  --model /path/to/model --output outputs/responses.jsonl
python -m evaluation.detect --input outputs/responses.jsonl \
  --output outputs/detections.jsonl
python -m evaluation.metrics --input outputs/detections.jsonl \
  --output outputs/metrics.json
```

Info. and Q-Score require `extra_info.reference_claim_count`: the claim count of the corresponding base-model response, measured with the same verifier and attached to the input data before generation.

For ConFiQA accuracy, provide `cf_answer` and `orig_answer` in the source dataset's `extra_info`:

```bash
python -m evaluation.evaluate_confiqa --source /path/to/confiqa_test.parquet \
  --input outputs/responses.jsonl --output outputs/confiqa_metrics.json
```

For pairwise win rate, configure a comparison model and use response files with matching sample IDs and prompts:

```bash
export BALTO_WR_BASE_URL=http://127.0.0.1:8002/v1
export BALTO_WR_MODEL=your-comparison-model

python -m evaluation.evaluate_wr --candidate outputs/candidate.jsonl \
  --reference outputs/reference.jsonl --output outputs/win_rate.json
```

All Python entry points support `--help`. Service and API-key settings are in [.env.example](.env.example).

## Repository structure

| Directory | Purpose |
| --- | --- |
| [verl/](verl/) | Training framework and core BALTO algorithm |
| [reward_models/](reward_models/) | Claim verification and token alignment |
| [configs/](configs/) | Training configurations |
| [scripts/](scripts/) | Training and serving launchers |
| [inference/](inference/) | Response generation |
| [evaluation/](evaluation/) | Hallucination detection and evaluation metrics |

## Citation

If you use BALTO in your research, please cite:

```bibtex
@article{li2026balto,
  title={BALTO: Balanced Token-Level Policy Optimization for Hallucination Mitigation},
  author={Li, Ning and Guo, Zixuan and Xu, Yan and Fei, Wenbo and Niu, Yifan and Luo, Chang and Wang, Yasheng and Liu, Weiwen and Yu, Yong and Zhang, Weinan},
  journal={Advances in Neural Information Processing Systems},
  year={2026}
}
```

## Acknowledgements

We thank the [verl](https://github.com/volcengine/verl) team for their open-source reinforcement learning framework, which serves as the foundation for BALTO's training implementation.
