from __future__ import annotations

import argparse
from pathlib import Path

from data.data_loader import load_20newsgroups
from modeling.baseline_tfidf_logreg import train_and_evaluate
from modeling.llm_classifier import run_llm_baseline


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entry point: load dataset, run baseline + LLM prediction, and save results",
    )
    parser.add_argument(
        "--remove",
        nargs="*",
        default=[],
        choices=["headers", "footers", "quotes"],
        help="Elements to remove from the text",
    )
    parser.add_argument(
        "--llm-max-test-samples",
        type=int,
        default=1000,
        help="Limit evaluation samples for LLM prediction",
    )
    parser.add_argument(
        "--llm-model",
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Hugging Face model name",
    )
    parser.add_argument(
        "--llm-max-chars",
        type=int,
        default=1600,
        help="Max characters per document before prompt construction",
    )
    parser.add_argument(
        "--llm-max-input-tokens",
        type=int,
        default=512,
        help="Tokenizer truncation limit for prompt inputs",
    )
    parser.add_argument(
        "--llm-quiet-hf",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Suppress Hugging Face logs/progress (default: show progress to avoid silent startup)",
    )
    args = parser.parse_args()

    train_dataset = load_20newsgroups(subset="train", remove=args.remove)
    test_dataset = load_20newsgroups(subset="test", remove=args.remove)

    print(f"Train documents: {len(train_dataset.data)}", flush=True)
    print(f"Test documents: {len(test_dataset.data)}", flush=True)
    print(f"Categories: {len(train_dataset.target_names)}", flush=True)

    print("[tfidf-logreg] Starting baseline prediction...", flush=True)
    baseline_metadata = train_and_evaluate(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        model_dir=PROJECT_ROOT / "modeling" / "configs",
        output_dir=OUTPUTS_DIR,
        metadata_dir=PROJECT_ROOT / "modeling" / "configs",
    )
    print(
        "[tfidf-logreg] Completed. "
        f"Metadata: {PROJECT_ROOT / 'modeling' / 'configs' / 'run_tfidf_logreg_metadata.json'}",
        flush=True,
    )
    print(
        f"[tfidf-logreg] Accuracy: {baseline_metadata['metrics']['accuracy']:.4f}",
        flush=True,
    )

    print("[llm] Starting LLM prediction...", flush=True)

    llm_metadata_path = run_llm_baseline(
        max_test_samples=args.llm_max_test_samples,
        model=args.llm_model,
        max_chars=args.llm_max_chars,
        max_input_tokens=args.llm_max_input_tokens,
        quiet_hf=args.llm_quiet_hf,
        output_dir=OUTPUTS_DIR,
        metadata_dir=PROJECT_ROOT / "modeling" / "configs",
        remove=args.remove,
        test_dataset=test_dataset,
    )
    print(f"[llm] Completed. Metadata: {llm_metadata_path}", flush=True)


if __name__ == "__main__":
    main()
