from __future__ import annotations

import argparse
from pathlib import Path

from data.data_loader import load_20newsgroups
from modeling.baseline_tfidf_logreg import train_and_evaluate
from modeling.llm_classifier import run_llm_baseline


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "modeling" / "configs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entry point: download 20 Newsgroups and run baselines",
    )
    parser.add_argument(
        "--remove",
        nargs="*",
        default=[],
        choices=["headers", "footers", "quotes"],
        help="Elements to remove from the text",
    )
    parser.add_argument(
        "--llm-mode",
        choices=["zero-shot", "few-shot", "both", "skip"],
        default="both",
        help="Run LLM baseline mode or skip it",
    )
    parser.add_argument(
        "--llm-max-test-samples",
        type=int,
        default=200,
        help="Limit LLM evaluation samples to control local inference runtime",
    )
    parser.add_argument(
        "--llm-few-shot-count",
        type=int,
        default=8,
        help="Number of few-shot examples to include",
    )
    parser.add_argument(
        "--llm-max-chars",
        type=int,
        default=1200,
        help="Max characters per document to send to the LLM",
    )
    parser.add_argument(
        "--llm-temperature",
        type=float,
        default=0.0,
        help="LLM sampling temperature",
    )
    parser.add_argument(
        "--llm-max-output-tokens",
        type=int,
        default=16,
        help="Max output tokens for the LLM label",
    )
    parser.add_argument(
        "--llm-random-state",
        type=int,
        default=42,
        help="Random seed for few-shot sampling",
    )
    parser.add_argument(
        "--llm-model",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Hugging Face model name",
    )
    parser.add_argument(
        "--llm-load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable 4-bit quantized loading for local model inference",
    )
    args = parser.parse_args()

    train_dataset = load_20newsgroups(subset="train", remove=args.remove)
    test_dataset = load_20newsgroups(subset="test", remove=args.remove)

    metadata = train_and_evaluate(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        model_dir=MODEL_DIR,
        output_dir=MODEL_DIR,
    )

    print("Baseline training complete.")
    print(f"Accuracy: {metadata['metrics']['accuracy']:.4f}")
    print(f"Model artifacts saved to: {MODEL_DIR}")
    print(f"Run metadata saved to: {MODEL_DIR / 'run_tfidf_logreg_metadata.json'}")

    if args.llm_mode != "skip":
        llm_metadata_path = run_llm_baseline(
            mode=args.llm_mode,
            max_test_samples=args.llm_max_test_samples,
            few_shot_count=args.llm_few_shot_count,
            max_chars=args.llm_max_chars,
            temperature=args.llm_temperature,
            max_output_tokens=args.llm_max_output_tokens,
            random_state=args.llm_random_state,
            model=args.llm_model,
            output_dir=OUTPUTS_DIR,
            load_in_4bit=args.llm_load_in_4bit,
        )
        print(f"LLM run metadata saved to: {llm_metadata_path}")


if __name__ == "__main__":
    main()
