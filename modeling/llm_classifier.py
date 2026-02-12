from __future__ import annotations

import argparse
import json
import os
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from sklearn.utils import Bunch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from transformers.utils import logging as hf_logging

from data.data_loader import load_20newsgroups


def _log(message: str) -> None:
    print(message, flush=True)


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


def _contains_label(text: str, label: str) -> bool:
    normalized = _normalize_label(text)
    label_lower = label.lower()
    pattern = rf"(?<![a-z0-9]){re.escape(label_lower)}(?![a-z0-9])"
    return re.search(pattern, normalized) is not None


def _categorize_output(
    predicted_label: Optional[str],
    true_label: str,
    raw_output: str,
) -> str:
    if _contains_label(raw_output, true_label):
        return "correct"
    if predicted_label is None:
        return "unrelated"
    return "incorrect"


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


def _estimate_tokens(tokenizer, text: str) -> int:
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except Exception:
        return max(1, len(text) // 4)


def _estimate_tokens_batch(tokenizer, texts: Sequence[str]) -> List[int]:
    if not texts:
        return []
    try:
        encoded = tokenizer(
            list(texts),
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_attention_mask=False,
            return_length=True,
        )
        lengths = encoded.get("length")
        if lengths is not None and len(lengths) == len(texts):
            return [int(length) for length in lengths]
    except Exception:
        pass
    return [_estimate_tokens(tokenizer, text) for text in texts]


def _estimate_tokens_chunked(
    tokenizer,
    texts: Sequence[str],
    chunk_size: int = 256,
) -> List[int]:
    lengths: List[int] = []
    size = max(1, chunk_size)
    for start in range(0, len(texts), size):
        chunk = texts[start : start + size]
        lengths.extend(_estimate_tokens_batch(tokenizer, chunk))
    return lengths


def _normal_interval_95(values: Sequence[int]) -> Tuple[float, float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0, 0.0
    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    std_value = variance**0.5
    return mean_value, std_value, mean_value - 1.96 * std_value, mean_value + 1.96 * std_value


def _resolve_effective_input_limits(
    tokenizer,
    texts: Sequence[str],
    prompt_builder,
) -> Tuple[int, int, List[str], List[int], Dict[str, float]]:
    char_lengths = [len(text) for text in texts]
    chars_mean, chars_std, chars_low, chars_high = _normal_interval_95(char_lengths)
    chars_in_interval = [
        value for value in char_lengths if chars_low <= value <= chars_high
    ]
    effective_max_chars = max(chars_in_interval, default=max(char_lengths, default=0))
    prompts = [prompt_builder(_truncate(text, effective_max_chars)) for text in texts]
    prompt_token_lengths = _estimate_tokens_chunked(tokenizer, prompts, chunk_size=256)
    tokens_mean, tokens_std, tokens_low, tokens_high = _normal_interval_95(prompt_token_lengths)
    tokens_in_interval = [
        value for value in prompt_token_lengths if tokens_low <= value <= tokens_high
    ]
    longest_prompt_tokens = max(tokens_in_interval, default=max(prompt_token_lengths, default=0))
    tokenizer_limit = int(getattr(tokenizer, "model_max_length", 0) or 0)
    if tokenizer_limit > 0 and tokenizer_limit < 1_000_000_000:
        effective_max_input_tokens = min(longest_prompt_tokens, tokenizer_limit)
    else:
        effective_max_input_tokens = longest_prompt_tokens
    stats = {
        "chars_mean": chars_mean,
        "chars_std": chars_std,
        "chars_interval_low": chars_low,
        "chars_interval_high": chars_high,
        "chars_in_interval_count": float(len(chars_in_interval)),
        "tokens_mean": tokens_mean,
        "tokens_std": tokens_std,
        "tokens_interval_low": tokens_low,
        "tokens_interval_high": tokens_high,
        "tokens_in_interval_count": float(len(tokens_in_interval)),
    }
    return (
        effective_max_chars,
        effective_max_input_tokens,
        prompts,
        prompt_token_lengths,
        stats,
    )


def _configure_hf_console(quiet_hf: bool) -> None:
    # On Windows without Developer Mode/admin symlink support, this warning is noisy but non-fatal.
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    # Reduce CUDA allocator fragmentation risk on long-running generation workloads.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
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
    _log(f"[llm] torch={torch.__version__}, torch_cuda_build={cuda_build}")
    if cuda_available:
        device_name = torch.cuda.get_device_name(0)
        _log(f"[llm] CUDA is available. Using GPU: {device_name}")
    else:
        _log("[llm] CUDA is not available. Falling back to CPU.")
    return cuda_available


def _build_generator(model_name: str, load_in_4bit: bool, gpu_only: bool):
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not hf_token:
        _log(
            "[warning] HF token not found in environment "
            "(set HF_TOKEN or HUGGINGFACE_HUB_TOKEN). "
            "Hub requests may be rate-limited."
        )

    _log(f"[llm] Loading tokenizer: {model_name}")
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

    _log("[llm] Loading model weights (first run may take several minutes)...")
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
    max_input_tokens: int,
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
            truncation=True,
            max_length=max_input_tokens,
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
    max_input_tokens: int,
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
                max_input_tokens=max_input_tokens,
            )
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            if size <= 1:
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            size = max(1, size // 2)
            _log(f"[warning] CUDA OOM; retrying with smaller batch_size={size}")


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
    max_input_tokens: int,
    prompts: Optional[Sequence[str]] = None,
    prompt_token_lengths: Optional[Sequence[int]] = None,
) -> Tuple[Dict[str, object], List[Optional[str]], List[str], List[str]]:
    predictions: List[Optional[str]] = []
    raw_outputs: List[str] = []
    estimated_prompt_tokens_sum = 0
    prompts_over_max_input_tokens = 0
    total_samples = len(texts)
    eval_start = perf_counter()
    if prompts is None:
        prompts = [prompt_builder(_truncate(text, max_chars)) for text in texts]
    if prompt_token_lengths is not None and len(prompt_token_lengths) != len(prompts):
        raise ValueError("prompt_token_lengths length must match prompts length")
    if prompt_token_lengths is None:
        prompt_token_lengths = _estimate_tokens_chunked(tokenizer, prompts, chunk_size=256)
    else:
        prompt_token_lengths = list(prompt_token_lengths)

    for sample_idx, (prompt, estimated_prompt_tokens) in enumerate(
        zip(prompts, prompt_token_lengths), start=1
    ):
        estimated_prompt_tokens_sum += estimated_prompt_tokens
        if estimated_prompt_tokens > max_input_tokens:
            prompts_over_max_input_tokens += 1

        if (
            token_monitor_every > 0
            and (sample_idx == 1 or sample_idx % token_monitor_every == 0 or sample_idx == total_samples)
        ):
            text_tokenization_rate = len(prompt) / max(estimated_prompt_tokens, 1)
            _log(
                f"[token-monitor][{eval_name}] "
                f"sample={sample_idx}/{total_samples} "
                f"estimated_prompt_tokens={estimated_prompt_tokens} "
                f"cumulative_estimated_prompt_tokens={estimated_prompt_tokens_sum} "
                f"text-tokenization-rate={text_tokenization_rate:.2f} chars/token"
            )

    if total_samples > 0:
        mean_prompt_tokens = sum(prompt_token_lengths) / total_samples
        variance = (
            sum((value - mean_prompt_tokens) ** 2 for value in prompt_token_lengths)
            / total_samples
        )
        std_prompt_tokens = variance**0.5
    else:
        mean_prompt_tokens = 0.0
        std_prompt_tokens = 0.0
    interval_low = mean_prompt_tokens - 1.96 * std_prompt_tokens
    interval_high = mean_prompt_tokens + 1.96 * std_prompt_tokens
    marked_abnormal = [
        (value < interval_low) or (value > interval_high)
        for value in prompt_token_lengths
    ]
    abnormal_count = sum(1 for marked in marked_abnormal if marked)
    normal_batch_size = 32
    _log(
        f"[llm][{eval_name}] length distribution: "
        f"mean={mean_prompt_tokens:.2f}, std={std_prompt_tokens:.2f}, "
        f"95pct_interval=[{interval_low:.2f}, {interval_high:.2f}], "
        f"abnormal_samples={abnormal_count}/{total_samples}"
    )

    processed_samples = 0
    generation_step = 0
    while processed_samples < total_samples:
        generation_step += 1
        if marked_abnormal[processed_samples]:
            step_batch_size = 1
            step_mode = "abnormal-single"
        else:
            step_batch_size = min(normal_batch_size, total_samples - processed_samples)
            while (
                step_batch_size > 1
                and any(marked_abnormal[processed_samples : processed_samples + step_batch_size])
            ):
                step_batch_size -= 1
            step_mode = "normal-32"

        retry_batch_size = step_batch_size
        retry_max_input_tokens = max_input_tokens
        raw_labels: List[str] = []
        while True:
            batch_prompts = prompts[processed_samples : processed_samples + retry_batch_size]
            try:
                raw_labels = _generate_labels_batch(
                    tokenizer=tokenizer,
                    model=model,
                    prompts=batch_prompts,
                    temperature=temperature,
                    max_new_tokens=max_output_tokens,
                    batch_size=retry_batch_size,
                    max_input_tokens=retry_max_input_tokens,
                )
                break
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if retry_batch_size > 1:
                    next_retry_batch_size = max(1, retry_batch_size // 2)
                    _log(
                        f"[warning] CUDA OOM at batch_size={retry_batch_size}; "
                        f"retrying with batch_size={next_retry_batch_size}"
                    )
                    retry_batch_size = next_retry_batch_size
                    continue

                # batch_size is already 1; reduce input token budget for this sample and retry.
                min_safe_input_tokens = 128
                if retry_max_input_tokens <= min_safe_input_tokens:
                    raise RuntimeError(
                        "CUDA OOM at batch_size=1 even after reducing max_input_tokens "
                        f"to {retry_max_input_tokens}. "
                        "Use a smaller model or stronger GPU."
                    ) from exc
                next_retry_max_input_tokens = max(
                    min_safe_input_tokens,
                    int(retry_max_input_tokens * 0.75),
                )
                _log(
                    f"[warning] CUDA OOM at batch_size=1 with max_input_tokens={retry_max_input_tokens}; "
                    f"retrying with max_input_tokens={next_retry_max_input_tokens}"
                )
                retry_max_input_tokens = next_retry_max_input_tokens
        if retry_batch_size != step_batch_size:
            step_mode = f"{step_mode}-oom-fallback"
            step_batch_size = retry_batch_size
        if retry_max_input_tokens != max_input_tokens:
            step_mode = f"{step_mode}-token-fallback"

        for raw in raw_labels:
            raw_outputs.append(raw)
            mapped = _map_to_label(raw, label_names)
            predictions.append(mapped)
        processed_samples += step_batch_size

        elapsed = perf_counter() - eval_start
        _log(
            f"[llm][{eval_name}] progress: step={generation_step}, "
            f"mode={step_mode}, batch_size={step_batch_size}, "
            f"samples={processed_samples}/{total_samples}, elapsed={elapsed:.1f}s"
        )

    correct = 0
    unknown = 0
    grouped_correct = 0
    grouped_unrelated = 0
    grouped_incorrect = 0
    for pred, target, raw in zip(predictions, targets, raw_outputs):
        true_label = label_names[target]
        if pred is None:
            unknown += 1
        if pred == true_label:
            correct += 1
        group = _categorize_output(pred, true_label, raw)
        if group == "correct":
            grouped_correct += 1
        elif group == "unrelated":
            grouped_unrelated += 1
        else:
            grouped_incorrect += 1

    total = len(targets)
    accuracy = correct / total if total else 0.0
    unknown_rate = unknown / total if total else 0.0
    grouped_correct_rate = grouped_correct / total if total else 0.0
    grouped_unrelated_rate = grouped_unrelated / total if total else 0.0
    grouped_incorrect_rate = grouped_incorrect / total if total else 0.0
    prompts_over_max_input_tokens_rate = (
        prompts_over_max_input_tokens / total if total else 0.0
    )

    metrics = {
        "accuracy": accuracy,
        "unknown_rate": unknown_rate,
        "total": total,
        "correct": correct,
        "unknown": unknown,
        "grouped_correct": grouped_correct,
        "grouped_unrelated": grouped_unrelated,
        "grouped_incorrect": grouped_incorrect,
        "grouped_correct_rate": grouped_correct_rate,
        "grouped_unrelated_rate": grouped_unrelated_rate,
        "grouped_incorrect_rate": grouped_incorrect_rate,
        "max_input_tokens": max_input_tokens,
        "prompts_over_max_input_tokens": prompts_over_max_input_tokens,
        "prompts_over_max_input_tokens_rate": prompts_over_max_input_tokens_rate,
        "fixed_normal_batch_size": normal_batch_size,
        "abnormal_sample_count": abnormal_count,
        "abnormal_sample_rate": (abnormal_count / total if total else 0.0),
        "length_interval_95pct_low": interval_low,
        "length_interval_95pct_high": interval_high,
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
        status = _categorize_output(pred_label, true_label, raw_output)
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


def _append_results_text(
    path: Path,
    section: str,
    rows: Sequence[Dict[str, object]],
    metrics: Dict[str, object],
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"[{section}]\n")
        handle.write(
            "grouped_correct={}/{} ({:.4f}), grouped_unrelated={}/{} ({:.4f}), grouped_incorrect={}/{} ({:.4f}), total_execution_seconds={:.3f}, avg_execution_seconds_per_sample={:.4f}, max_input_tokens={}, prompts_over_max_input_tokens={}, prompts_over_max_input_tokens_rate={:.4f}\n".format(
                int(metrics.get("grouped_correct", 0)),
                int(metrics.get("total", 0)),
                float(metrics.get("grouped_correct_rate", 0.0)),
                int(metrics.get("grouped_unrelated", 0)),
                int(metrics.get("total", 0)),
                float(metrics.get("grouped_unrelated_rate", 0.0)),
                int(metrics.get("grouped_incorrect", 0)),
                int(metrics.get("total", 0)),
                float(metrics.get("grouped_incorrect_rate", 0.0)),
                float(metrics.get("total_execution_seconds", 0.0)),
                float(metrics.get("avg_execution_seconds_per_sample", 0.0)),
                int(metrics.get("max_input_tokens", 0)),
                int(metrics.get("prompts_over_max_input_tokens", 0)),
                float(metrics.get("prompts_over_max_input_tokens_rate", 0.0)),
            )
        )
        for row in rows:
            handle.write(
                "sample={sample_index} true={true_label} pred={predicted_label} status={status} raw={raw_output}\n".format(**row)
            )
        handle.write("\n")


def _build_timing_metrics(total_seconds: float, sample_count: int) -> Dict[str, float]:
    avg_seconds = total_seconds / sample_count if sample_count else 0.0
    return {
        "total_execution_seconds": total_seconds,
        "avg_execution_seconds_per_sample": avg_seconds,
    }


def run_llm_baseline(
    max_test_samples: int = 1000,
    max_chars: int = 1600,
    temperature: float = 0.0,
    max_output_tokens: int = 16,
    model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    output_dir: Optional[Path] = None,
    metadata_dir: Optional[Path] = None,
    load_in_4bit: bool = True,
    batch_size: int = 32,
    token_monitor_every: int = 20,
    quiet_hf: bool = True,
    gpu_only: bool = True,
    max_input_tokens: int = 512,
    remove: Sequence[str] = ("headers", "footers", "quotes"),
    test_dataset: Optional[Bunch] = None,
) -> Path:
    _configure_hf_console(quiet_hf=quiet_hf)
    cuda_available = _report_torch_runtime()

    if output_dir is None:
        output_dir = Path(__file__).resolve().parents[1] / "outputs"
    if metadata_dir is None:
        metadata_dir = output_dir

    tokenizer, model_instance = _build_generator(
        model_name=model,
        load_in_4bit=load_in_4bit,
        gpu_only=gpu_only,
    )

    if test_dataset is None:
        test_dataset = load_20newsgroups(subset="test", remove=remove)

    max_samples = min(max_test_samples, len(test_dataset.data))
    test_texts = list(test_dataset.data[:max_samples])
    test_targets = list(test_dataset.target[:max_samples])
    label_names = list(test_dataset.target_names)

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    zero_shot_results_txt_path = output_dir / "results_zero_shot.txt"

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
        "max_input_tokens": max_input_tokens,
        "remove": list(remove),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    _log(f"\n[llm][zero-shot] Starting evaluation on {max_samples} samples...")
    zero_shot_start = perf_counter()
    zero_shot_builder = lambda text: _build_zero_shot_prompt(text, label_names)
    (
        effective_max_chars,
        effective_max_input_tokens,
        zero_shot_prompts_prebuilt,
        zero_shot_prompt_token_lengths,
        limit_interval_stats,
    ) = _resolve_effective_input_limits(
        tokenizer=tokenizer,
        texts=test_texts,
        prompt_builder=zero_shot_builder,
    )
    _log(
        "[llm][zero-shot] dynamic limits resolved: "
        f"max_chars={effective_max_chars}, "
        f"max_input_tokens={effective_max_input_tokens}"
    )
    _log(
        "[llm][zero-shot] 95% intervals: "
        f"max_chars_interval=[{limit_interval_stats['chars_interval_low']:.2f}, {limit_interval_stats['chars_interval_high']:.2f}], "
        f"max_input_tokens_interval=[{limit_interval_stats['tokens_interval_low']:.2f}, {limit_interval_stats['tokens_interval_high']:.2f}]"
    )
    if zero_shot_prompt_token_lengths:
        max_token_idx = max(
            range(len(zero_shot_prompt_token_lengths)),
            key=lambda i: zero_shot_prompt_token_lengths[i],
        )
        max_token_count = zero_shot_prompt_token_lengths[max_token_idx]
        max_token_prompt_chars = len(zero_shot_prompts_prebuilt[max_token_idx])
        _log(
            "[llm][zero-shot] maximum-token input: "
            f"sample={max_token_idx + 1}/{max_samples}, "
            f"estimated_prompt_tokens={max_token_count}, "
            f"prompt_chars={max_token_prompt_chars}"
        )
    results["requested_max_chars"] = max_chars
    results["requested_max_input_tokens"] = max_input_tokens
    results["max_chars"] = effective_max_chars
    results["max_input_tokens"] = effective_max_input_tokens
    results["max_chars_interval_95pct_low"] = limit_interval_stats["chars_interval_low"]
    results["max_chars_interval_95pct_high"] = limit_interval_stats["chars_interval_high"]
    results["max_chars_interval_95pct_count"] = int(limit_interval_stats["chars_in_interval_count"])
    results["max_input_tokens_interval_95pct_low"] = limit_interval_stats["tokens_interval_low"]
    results["max_input_tokens_interval_95pct_high"] = limit_interval_stats["tokens_interval_high"]
    results["max_input_tokens_interval_95pct_count"] = int(limit_interval_stats["tokens_in_interval_count"])
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
        max_chars=effective_max_chars,
        batch_size=batch_size,
        token_monitor_every=token_monitor_every,
        max_input_tokens=effective_max_input_tokens,
        prompts=zero_shot_prompts_prebuilt,
        prompt_token_lengths=zero_shot_prompt_token_lengths,
    )
    zero_shot_timing = _build_timing_metrics(
        total_seconds=perf_counter() - zero_shot_start,
        sample_count=max_samples,
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
        **zero_shot_timing,
        "predictions_path": str(zero_shot_path),
        "raw_outputs_path": str(zero_shot_raw_path),
        "results_txt_path": str(zero_shot_results_txt_path),
    }
    _append_results_text(
        path=zero_shot_results_txt_path,
        section="zero-shot",
        rows=_build_prediction_rows(
            targets=test_targets,
            predictions=zero_shot_predictions,
            raw_outputs=zero_shot_raw_outputs,
            label_names=label_names,
        ),
        metrics=results["zero_shot"],
    )
    _log(
        "[llm][zero-shot] complete: "
        f"accuracy={results['zero_shot']['accuracy']:.4f}, "
        f"grouped_correct_rate={results['zero_shot']['grouped_correct_rate']:.4f}, "
        f"grouped_unrelated_rate={results['zero_shot']['grouped_unrelated_rate']:.4f}, "
        f"grouped_incorrect_rate={results['zero_shot']['grouped_incorrect_rate']:.4f}, "
        f"unknown_rate={results['zero_shot']['unknown_rate']:.4f}, "
        f"prompts_over_max_input_tokens_rate={results['zero_shot']['prompts_over_max_input_tokens_rate']:.4f}"
    )
    _log(
        "[llm][zero-shot] timing: "
        f"total={results['zero_shot']['total_execution_seconds']:.3f}s, "
        f"avg_per_sample={results['zero_shot']['avg_execution_seconds_per_sample']:.4f}s"
    )
    _log(f"[llm][zero-shot] text results saved: {zero_shot_results_txt_path}")
    _log(f"[llm][zero-shot] predictions saved: {zero_shot_path}")
    _log(f"[llm][zero-shot] raw outputs saved: {zero_shot_raw_path}")

    output_path = metadata_dir / "llm_run_metadata.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Local Hugging Face baseline (Qwen2.5-1.5B-Instruct) for 20 Newsgroups classification",
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
        default=1000,
        help="Limit evaluation samples",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=1600,
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
        "--model",
        default="Qwen/Qwen2.5-1.5B-Instruct",
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
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=512,
        help="Max input tokens for tokenizer truncation",
    )
    args = parser.parse_args()

    output_path = run_llm_baseline(
        max_test_samples=args.max_test_samples,
        max_chars=args.max_chars,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        model=args.model,
        load_in_4bit=args.load_in_4bit,
        batch_size=args.batch_size,
        token_monitor_every=args.token_monitor_every,
        quiet_hf=args.quiet_hf,
        gpu_only=args.gpu_only,
        max_input_tokens=args.max_input_tokens,
        remove=args.remove,
    )

    print(f"LLM run metadata saved to: {output_path}")


if __name__ == "__main__":
    main()
