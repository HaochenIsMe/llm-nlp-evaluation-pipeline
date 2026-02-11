from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

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


def _build_generator(model_name: str, load_in_4bit: bool):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "trust_remote_code": True,
    }
    if torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.float16
        model_kwargs["device_map"] = "auto"
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
        model_kwargs["torch_dtype"] = torch.float32

    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    except Exception as exc:
        if "quantization_config" not in model_kwargs:
            raise
        print(f"[warning] 4-bit model load failed, retrying without quantization: {exc}")
        model_kwargs.pop("quantization_config", None)
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    text_generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )
    return tokenizer, text_generator


def _generate_label(
    text_generator,
    prompt: str,
    temperature: float,
    max_new_tokens: int,
) -> str:
    outputs = text_generator(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else None,
        return_full_text=False,
        num_return_sequences=1,
    )
    return outputs[0]["generated_text"].strip()


def _evaluate(
    tokenizer,
    text_generator,
    label_names: Sequence[str],
    texts: Sequence[str],
    targets: Sequence[int],
    prompt_builder,
    temperature: float,
    max_output_tokens: int,
    max_chars: int,
) -> Dict[str, object]:
    predictions: List[Optional[str]] = []
    estimated_prompt_tokens_sum = 0
    estimated_total_tokens_sum = 0
    total_samples = len(texts)
    for idx, text in enumerate(texts, start=1):
        prompt = prompt_builder(_truncate(text, max_chars))
        estimated_prompt_tokens = _estimate_tokens(tokenizer, prompt)
        estimated_total_tokens = estimated_prompt_tokens + max_output_tokens
        estimated_prompt_tokens_sum += estimated_prompt_tokens
        estimated_total_tokens_sum += estimated_total_tokens

        print(
            "[token-monitor] "
            f"sample={idx}/{total_samples} "
            f"estimated_prompt_tokens={estimated_prompt_tokens} "
            f"estimated_total_tokens<={estimated_total_tokens} "
            f"cumulative_estimated_prompt_tokens={estimated_prompt_tokens_sum} "
            f"cumulative_estimated_total_tokens<={estimated_total_tokens_sum}"
        )

        raw = _generate_label(
            text_generator=text_generator,
            prompt=prompt,
            temperature=temperature,
            max_new_tokens=max_output_tokens,
        )
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

    return {
        "accuracy": accuracy,
        "unknown_rate": unknown_rate,
        "total": total,
        "correct": correct,
        "unknown": unknown,
    }


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
) -> Path:
    if output_dir is None:
        output_dir = Path(__file__).resolve().parents[1] / "outputs"

    tokenizer, text_generator = _build_generator(model_name=model, load_in_4bit=load_in_4bit)

    train_dataset = load_20newsgroups(subset="train")
    test_dataset = load_20newsgroups(subset="test")

    max_samples = min(max_test_samples, len(test_dataset.data))
    test_texts = list(test_dataset.data[:max_samples])
    test_targets = list(test_dataset.target[:max_samples])
    label_names = list(test_dataset.target_names)

    output_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, object] = {
        "platform": "huggingface",
        "framework": "transformers",
        "model": model,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "load_in_4bit": load_in_4bit,
        "max_test_samples": max_samples,
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
        "max_chars": max_chars,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if mode in {"zero-shot", "both"}:
        zero_shot_builder = lambda text: _build_zero_shot_prompt(text, label_names)
        results["zero_shot"] = _evaluate(
            tokenizer=tokenizer,
            text_generator=text_generator,
            label_names=label_names,
            texts=test_texts,
            targets=test_targets,
            prompt_builder=zero_shot_builder,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            max_chars=max_chars,
        )

    if mode in {"few-shot", "both"}:
        few_shot_examples = _select_few_shot_examples(
            texts=train_dataset.data,
            targets=train_dataset.target,
            label_names=label_names,
            count=few_shot_count,
            random_state=random_state,
            max_chars=max_chars,
        )
        few_shot_builder = lambda text: _build_few_shot_prompt(
            text, label_names, few_shot_examples
        )
        results["few_shot"] = {
            "examples_used": len(few_shot_examples),
            "random_state": random_state,
            "metrics": _evaluate(
                tokenizer=tokenizer,
                text_generator=text_generator,
                label_names=label_names,
                texts=test_texts,
                targets=test_targets,
                prompt_builder=few_shot_builder,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                max_chars=max_chars,
            ),
        }

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
    )

    print(f"LLM run metadata saved to: {output_path}")


if __name__ == "__main__":
    main()
