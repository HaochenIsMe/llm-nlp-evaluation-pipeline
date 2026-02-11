from __future__ import annotations

import argparse
import json
import os
import random
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from transformers.utils import logging as hf_logging

from data.data_loader import load_20newsgroups


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return text
    return text[:max_chars]


def _normalize_label(label: str) -> str:
    cleaned = label.strip().lower()
    if cleaned.startswith("label:"):
        cleaned = cleaned.replace("label:", "", 1).strip()
    return cleaned


def _map_to_label(label: str, label_names: Sequence[str]) -> Optional[str]:
    normalized = _normalize_label(label)
    for name in label_names:
        if normalized == name.lower():
            return name

    # Common case: model returns label followed by explanation/newlines.
    first_line = normalized.splitlines()[0].strip() if normalized else ""
    for name in label_names:
        label_lower = name.lower()
        if first_line == label_lower or first_line.startswith(label_lower):
            return name

    # Fallback: find first valid label mention with non-alnum boundaries.
    best_match: Optional[Tuple[int, str]] = None
    for name in label_names:
        label_lower = name.lower()
        pattern = rf"(?<![a-z0-9]){re.escape(label_lower)}(?![a-z0-9])"
        match = re.search(pattern, normalized)
        if match is None:
            continue
        if best_match is None or match.start() < best_match[0]:
            best_match = (match.start(), name)
    if best_match is not None:
        return best_match[1]
    return None


def _build_zero_shot_prompt(text: str, label_names: Sequence[str]) -> str:
    label_list = ", ".join(label_names)
    return (
        "You are a news topic classifier for the 20 Newsgroups dataset. "
        "Choose exactly one label from the list below and reply with only the label.\n\n"
        f"Labels: {label_list}\n\n"
        "Text:\n"
        f"{text}\n\n"
        "Label:"
    )


def _build_few_shot_prompt(
    text: str, label_names: Sequence[str], examples: Iterable[Tuple[str, str]]
) -> str:
    label_list = ", ".join(label_names)
    example_blocks = []
    for example_text, example_label in examples:
        example_blocks.append(
            "Text:\n"
            f"{example_text}\n"
            f"Label: {example_label}\n"
        )
    examples_section = "\n---\n".join(example_blocks)
    return (
        "You are a news topic classifier for the 20 Newsgroups dataset. "
        "Choose exactly one label from the list below and reply with only the label.\n\n"
        f"Labels: {label_list}\n\n"
        "Examples:\n"
        f"{examples_section}\n\n"
        "---\n"
        "Text:\n"
        f"{text}\n\n"
        "Label:"
    )


def _select_few_shot_examples(
    texts: Sequence[str],
    targets: Sequence[int],
    label_names: Sequence[str],
    count: int,
    random_state: int,
    max_chars: int,
) -> List[Tuple[str, str]]:
    rng = random.Random(random_state)
    indices = list(range(len(texts)))
    rng.shuffle(indices)
    selected: List[Tuple[str, str]] = []
    for idx in indices:
        selected.append((_truncate(texts[idx], max_chars), label_names[targets[idx]]))
        if len(selected) >= count:
            break
    return selected


def _estimate_tokens(tokenizer, text: str) -> int:
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except Exception:
        return max(1, len(text) // 4)


def _configure_hf_console(quiet_hf: bool) -> None:
    if not quiet_hf:
        return
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
    warnings.filterwarnings("ignore", message=r".*torch_dtype.*deprecated.*")
    warnings.filterwarnings("ignore", message=r".*Both `max_new_tokens`.*")
    warnings.filterwarnings(
        "ignore",
        message=r".*generation_config.*generation-related arguments.*deprecated.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*decoder-only architecture.*right-padding.*",
    )


def _report_torch_runtime() -> bool:
    cuda_available = torch.cuda.is_available()
    cuda_build = torch.version.cuda or "cpu-only"
    print(f"[llm] torch={torch.__version__}, torch_cuda_build={cuda_build}")
    if cuda_available:
        device_name = torch.cuda.get_device_name(0)
        print(f"[llm] CUDA is available. Using GPU: {device_name}")
    else:
        print("[llm] CUDA is not available. Falling back to CPU.")
    return cuda_available


def _build_generator(model_name: str, load_in_4bit: bool, gpu_only: bool):
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not hf_token:
        print(
            "[warning] HF token not found in environment "
            "(set HF_TOKEN or HUGGINGFACE_HUB_TOKEN). "
            "Hub requests may be rate-limited."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        token=hf_token,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "trust_remote_code": True,
    }
    if torch.cuda.is_available():
        model_kwargs["dtype"] = torch.float16
        # Keep weights strictly on GPU to avoid CPU/shared-memory spillover.
        model_kwargs["device_map"] = {"": 0} if gpu_only else "auto"
        if load_in_4bit:
            try:
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
                model_kwargs["quantization_config"] = quant_config
            except Exception as exc:
                print(f"[warning] 4-bit quantization setup failed, falling back: {exc}")
    else:
        if gpu_only:
            raise RuntimeError(
                "gpu_only=True but CUDA is unavailable. "
                "Install CUDA-enabled PyTorch and ensure GPU is visible."
            )
        model_kwargs["dtype"] = torch.float32

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=hf_token,
            **model_kwargs,
        )
    except Exception as exc:
        if "quantization_config" not in model_kwargs:
            raise
        print(f"[warning] 4-bit model load failed, retrying without quantization: {exc}")
        model_kwargs.pop("quantization_config", None)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=hf_token,
            **model_kwargs,
        )
    return tokenizer, model


def _generate_label(
    tokenizer,
    model,
    prompt: str,
    temperature: float,
    max_new_tokens: int,
) -> str:
    return _generate_labels_batch(
        tokenizer=tokenizer,
        model=model,
        prompts=[prompt],
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        batch_size=1,
    )[0]


def _generate_labels_batch(
    tokenizer,
    model,
    prompts: Sequence[str],
    temperature: float,
    max_new_tokens: int,
    batch_size: int,
) -> List[str]:
    do_sample = temperature > 0
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "num_return_sequences": 1,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature

    results: List[str] = []
    for start in range(0, len(prompts), max(1, batch_size)):
        chunk = list(prompts[start : start + max(1, batch_size)])
        inputs = tokenizer(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                **generation_kwargs,
            )
        prompt_len = inputs["input_ids"].shape[1]
        continuation = generated[:, prompt_len:]
        decoded = tokenizer.batch_decode(
            continuation,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        results.extend([text.strip() for text in decoded])
    return results


def _generate_labels_adaptive(
    tokenizer,
    model,
    prompts: Sequence[str],
    temperature: float,
    max_new_tokens: int,
    batch_size: int,
) -> List[str]:
    if not prompts:
        return []

    size = max(1, batch_size)
    while True:
        try:
            return _generate_labels_batch(
                tokenizer=tokenizer,
                model=model,
                prompts=prompts,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                batch_size=size,
            )
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            if size <= 1:
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            size = max(1, size // 2)
            print(f"[warning] CUDA OOM; retrying with smaller batch_size={size}")


def _evaluate(
    tokenizer,
    model,
    eval_name: str,
    label_names: Sequence[str],
    texts: Sequence[str],
    targets: Sequence[int],
    prompt_builder,
    temperature: float,
    max_output_tokens: int,
    max_chars: int,
    batch_size: int,
    token_monitor_every: int,
) -> Tuple[Dict[str, object], List[Optional[str]], List[str], List[str]]:
    predictions: List[Optional[str]] = []
    raw_outputs: List[str] = []
    estimated_prompt_tokens_sum = 0
    total_samples = len(texts)
    effective_batch_size = max(1, batch_size)
    prompts = [prompt_builder(_truncate(text, max_chars)) for text in texts]
    for batch_start in range(0, total_samples, effective_batch_size):
        batch_end = min(batch_start + effective_batch_size, total_samples)
        batch_prompts = prompts[batch_start:batch_end]

        for offset, prompt in enumerate(batch_prompts, start=1):
            sample_idx = batch_start + offset
            # Avoid expensive per-sample tokenization unless monitor output is requested.
            if token_monitor_every > 0:
                estimated_prompt_tokens = _estimate_tokens(tokenizer, prompt)
            else:
                estimated_prompt_tokens = max(1, len(prompt) // 4)
            estimated_prompt_tokens_sum += estimated_prompt_tokens

            if (
                token_monitor_every > 0
                and (sample_idx == 1 or sample_idx % token_monitor_every == 0 or sample_idx == total_samples)
            ):
                print(
                    f"[token-monitor][{eval_name}] "
                    f"sample={sample_idx}/{total_samples} "
                    f"estimated_prompt_tokens={estimated_prompt_tokens} "
                    f"cumulative_estimated_prompt_tokens={estimated_prompt_tokens_sum}"
                )

        raw_labels = _generate_labels_adaptive(
            tokenizer=tokenizer,
            model=model,
            prompts=batch_prompts,
            temperature=temperature,
            max_new_tokens=max_output_tokens,
            batch_size=effective_batch_size,
        )

        for raw in raw_labels:
            raw_outputs.append(raw)
            mapped = _map_to_label(raw, label_names)
            predictions.append(mapped)

    correct = 0
    unknown = 0
    for pred, target in zip(predictions, targets):
        if pred is None:
            unknown += 1
            continue
        if pred == label_names[target]:
            correct += 1

    total = len(targets)
    accuracy = correct / total if total else 0.0
    unknown_rate = unknown / total if total else 0.0

    metrics = {
        "accuracy": accuracy,
        "unknown_rate": unknown_rate,
        "total": total,
        "correct": correct,
        "unknown": unknown,
    }
    return metrics, predictions, raw_outputs, prompts


def _build_prediction_rows(
    targets: Sequence[int],
    predictions: Sequence[Optional[str]],
    raw_outputs: Sequence[str],
    label_names: Sequence[str],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for idx, (target_idx, pred_label, raw_output) in enumerate(
        zip(targets, predictions, raw_outputs), start=1
    ):
        true_label = label_names[target_idx]
        if pred_label is None:
            status = "unknown"
        elif pred_label == true_label:
            status = "correct"
        else:
            status = "incorrect"
        rows.append(
            {
                "sample_index": idx,
                "true_label": true_label,
                "predicted_label": pred_label,
                "raw_output": raw_output,
                "status": status,
            }
        )
    return rows


def _write_predictions_jsonl(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _build_raw_output_rows(
    prompts: Sequence[str],
    raw_outputs: Sequence[str],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for idx, (prompt, raw_output) in enumerate(zip(prompts, raw_outputs), start=1):
        rows.append(
            {
                "sample_index": idx,
                "prompt": prompt,
                "raw_output": raw_output,
            }
        )
    return rows


def run_llm_baseline(
    mode: str = "both",
    max_test_samples: int = 200,
    few_shot_count: int = 8,
    max_chars: int = 1200,
    temperature: float = 0.0,
    max_output_tokens: int = 16,
    random_state: int = 42,
    model: str = "Qwen/Qwen2.5-7B-Instruct",
    output_dir: Optional[Path] = None,
    load_in_4bit: bool = True,
    batch_size: int = 32,
    token_monitor_every: int = 20,
    quiet_hf: bool = True,
    gpu_only: bool = True,
    remove: Sequence[str] = ("headers", "footers", "quotes"),
) -> Path:
    _configure_hf_console(quiet_hf=quiet_hf)
    cuda_available = _report_torch_runtime()

    if output_dir is None:
        output_dir = Path(__file__).resolve().parents[1] / "outputs"

    tokenizer, model_instance = _build_generator(
        model_name=model,
        load_in_4bit=load_in_4bit,
        gpu_only=gpu_only,
    )

    train_dataset = load_20newsgroups(subset="train", remove=remove)
    test_dataset = load_20newsgroups(subset="test", remove=remove)

    max_samples = min(max_test_samples, len(test_dataset.data))
    test_texts = list(test_dataset.data[:max_samples])
    test_targets = list(test_dataset.target[:max_samples])
    label_names = list(test_dataset.target_names)

    output_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, object] = {
        "platform": "huggingface",
        "framework": "transformers",
        "model": model,
        "device": "cuda" if cuda_available else "cpu",
        "load_in_4bit": load_in_4bit,
        "max_test_samples": max_samples,
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
        "max_chars": max_chars,
        "batch_size": batch_size,
        "token_monitor_every": token_monitor_every,
        "quiet_hf": quiet_hf,
        "gpu_only": gpu_only,
        "remove": list(remove),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if mode in {"zero-shot", "both"}:
        print(f"\n[llm][zero-shot] Starting evaluation on {max_samples} samples...")
        zero_shot_builder = lambda text: _build_zero_shot_prompt(text, label_names)
        zero_shot_metrics, zero_shot_predictions, zero_shot_raw_outputs, zero_shot_prompts = _evaluate(
            tokenizer=tokenizer,
            model=model_instance,
            eval_name="zero-shot",
            label_names=label_names,
            texts=test_texts,
            targets=test_targets,
            prompt_builder=zero_shot_builder,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            max_chars=max_chars,
            batch_size=batch_size,
            token_monitor_every=token_monitor_every,
        )
        zero_shot_path = output_dir / "llm_predictions_zero_shot.jsonl"
        _write_predictions_jsonl(
            path=zero_shot_path,
            rows=_build_prediction_rows(
                targets=test_targets,
                predictions=zero_shot_predictions,
                raw_outputs=zero_shot_raw_outputs,
                label_names=label_names,
            ),
        )
        zero_shot_raw_path = output_dir / "llm_raw_outputs_zero_shot.jsonl"
        _write_predictions_jsonl(
            path=zero_shot_raw_path,
            rows=_build_raw_output_rows(
                prompts=zero_shot_prompts,
                raw_outputs=zero_shot_raw_outputs,
            ),
        )
        results["zero_shot"] = {
            **zero_shot_metrics,
            "predictions_path": str(zero_shot_path),
            "raw_outputs_path": str(zero_shot_raw_path),
        }
        print(
            "[llm][zero-shot] complete: "
            f"accuracy={results['zero_shot']['accuracy']:.4f}, "
            f"unknown_rate={results['zero_shot']['unknown_rate']:.4f}"
        )
        print(f"[llm][zero-shot] predictions saved: {zero_shot_path}")
        print(f"[llm][zero-shot] raw outputs saved: {zero_shot_raw_path}")

    # TEMP: few-shot is disabled to focus on zero-shot debugging.
    # if mode in {"few-shot", "both"}:
    #     print(f"\n[llm][few-shot] Preparing {few_shot_count} examples...")
    #     few_shot_examples = _select_few_shot_examples(
    #         texts=train_dataset.data,
    #         targets=train_dataset.target,
    #         label_names=label_names,
    #         count=few_shot_count,
    #         random_state=random_state,
    #         max_chars=max_chars,
    #     )
    #     few_shot_builder = lambda text: _build_few_shot_prompt(
    #         text, label_names, few_shot_examples
    #     )
    #     print(f"[llm][few-shot] Starting evaluation on {max_samples} samples...")
    #     few_shot_metrics, few_shot_predictions, few_shot_raw_outputs, few_shot_prompts = _evaluate(
    #         tokenizer=tokenizer,
    #         model=model_instance,
    #         eval_name="few-shot",
    #         label_names=label_names,
    #         texts=test_texts,
    #         targets=test_targets,
    #         prompt_builder=few_shot_builder,
    #         temperature=temperature,
    #         max_output_tokens=max_output_tokens,
    #         max_chars=max_chars,
    #         batch_size=batch_size,
    #         token_monitor_every=token_monitor_every,
    #     )
    #     few_shot_path = output_dir / "llm_predictions_few_shot.jsonl"
    #     _write_predictions_jsonl(
    #         path=few_shot_path,
    #         rows=_build_prediction_rows(
    #             targets=test_targets,
    #             predictions=few_shot_predictions,
    #             raw_outputs=few_shot_raw_outputs,
    #             label_names=label_names,
    #         ),
    #     )
    #     few_shot_raw_path = output_dir / "llm_raw_outputs_few_shot.jsonl"
    #     _write_predictions_jsonl(
    #         path=few_shot_raw_path,
    #         rows=_build_raw_output_rows(
    #             prompts=few_shot_prompts,
    #             raw_outputs=few_shot_raw_outputs,
    #         ),
    #     )
    #     results["few_shot"] = {
    #         "examples_used": len(few_shot_examples),
    #         "random_state": random_state,
    #         "predictions_path": str(few_shot_path),
    #         "raw_outputs_path": str(few_shot_raw_path),
    #         "metrics": few_shot_metrics,
    #     }
    #     print(
    #         "[llm][few-shot] complete: "
    #         f"accuracy={results['few_shot']['metrics']['accuracy']:.4f}, "
    #         f"unknown_rate={results['few_shot']['metrics']['unknown_rate']:.4f}"
    #     )
    #     print(f"[llm][few-shot] predictions saved: {few_shot_path}")
    #     print(f"[llm][few-shot] raw outputs saved: {few_shot_raw_path}")

    output_path = output_dir / "llm_run_metadata.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Local Hugging Face baseline (Qwen2.5-7B-Instruct) for 20 Newsgroups classification",
    )
    parser.add_argument(
        "--mode",
        choices=["zero-shot", "few-shot", "both"],
        default="both",
        help="Prompting mode to run",
    )
    parser.add_argument(
        "--remove",
        nargs="*",
        default=["headers", "footers", "quotes"],
        choices=["headers", "footers", "quotes"],
        help="Elements to remove from 20 Newsgroups texts",
    )
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=200,
        help="Limit evaluation samples",
    )
    parser.add_argument(
        "--few-shot-count",
        type=int,
        default=8,
        help="Number of few-shot examples to include",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=1200,
        help="Max characters per document to send to the model",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=16,
        help="Max output tokens for the label",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for few-shot sampling",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Hugging Face model name",
    )
    parser.add_argument(
        "--load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable 4-bit quantized loading (recommended on RTX 3060)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Generation batch size target (higher is faster; auto-reduces on CUDA OOM)",
    )
    parser.add_argument(
        "--token-monitor-every",
        type=int,
        default=20,
        help="Print token-monitor every N samples; set 0 to disable",
    )
    parser.add_argument(
        "--quiet-hf",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Suppress non-actionable Hugging Face/Transformers logs for cleaner output",
    )
    parser.add_argument(
        "--gpu-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep model strictly on GPU; fail on OOM instead of CPU/shared-memory offload",
    )
    args = parser.parse_args()

    output_path = run_llm_baseline(
        mode=args.mode,
        max_test_samples=args.max_test_samples,
        few_shot_count=args.few_shot_count,
        max_chars=args.max_chars,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        random_state=args.random_state,
        model=args.model,
        load_in_4bit=args.load_in_4bit,
        batch_size=args.batch_size,
        token_monitor_every=args.token_monitor_every,
        quiet_hf=args.quiet_hf,
        gpu_only=args.gpu_only,
        remove=args.remove,
    )

    print(f"LLM run metadata saved to: {output_path}")


if __name__ == "__main__":
    main()
