from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

from data.data_loader import load_20newsgroups


def _default_tfidf_params() -> Dict[str, Any]:
    return {
        "max_features": 20000,
        "ngram_range": (1, 2),
        "min_df": 2,
        "max_df": 0.9,
        "sublinear_tf": True,
    }


def _default_logreg_params() -> Dict[str, Any]:
    return {
        "max_iter": 1000,
        "solver": "lbfgs",
    }


def train_and_evaluate(
    train_dataset,
    test_dataset,
    model_dir: Path,
    output_dir: Path,
    tfidf_params: Optional[Dict[str, Any]] = None,
    logreg_params: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    tfidf_params = tfidf_params or _default_tfidf_params()
    logreg_params = logreg_params or _default_logreg_params()

    if verbose:
        print("\n[tfidf-logreg] Starting baseline run")
        print(f"[tfidf-logreg] Train samples: {len(train_dataset.data)}")
        print(f"[tfidf-logreg] Test samples: {len(test_dataset.data)}")
        print("[tfidf-logreg] Building vectorizer and classifier...")

    vectorizer = TfidfVectorizer(**tfidf_params)
    classifier = LogisticRegression(**logreg_params)
    pipeline = Pipeline(
        [
            ("tfidf", vectorizer),
            ("logreg", classifier),
        ]
    )

    # TEMP ANNOTATION: This section is the TF-IDF matrix generation/training block.
    # It is CPU-bound by design (not using GPU), and logs are kept explicit for readability.
    if verbose:
        print("[tfidf-logreg] Fitting TF-IDF matrix + Logistic Regression...")
    pipeline.fit(train_dataset.data, train_dataset.target)
    train_matrix_shape = pipeline.named_steps["tfidf"].transform(train_dataset.data).shape
    test_matrix_shape = pipeline.named_steps["tfidf"].transform(test_dataset.data).shape
    # TEMP ANNOTATION END
    if verbose:
        print(
            "[tfidf-logreg] [matrices generation complete] "
            f"train={train_matrix_shape}, test={test_matrix_shape}"
        )
        print("[tfidf-logreg] Running inference on test split...")

    predictions = pipeline.predict(test_dataset.data)
    accuracy = accuracy_score(test_dataset.target, predictions)

    pipeline_path = model_dir / "baseline_tfidf_logreg_pipeline.joblib"
    vectorizer_path = model_dir / "baseline_tfidf_vectorizer.joblib"
    logreg_path = model_dir / "baseline_tfidf_logreg.joblib"

    joblib.dump(pipeline, pipeline_path)
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(classifier, logreg_path)

    metadata = {
        "train_size": len(train_dataset.data),
        "test_size": len(test_dataset.data),
        "metrics": {
            "accuracy": accuracy,
        },
        "tfidf_params": {
            **tfidf_params,
            "ngram_range": list(tfidf_params.get("ngram_range", (1, 1))),
        },
        "logreg_params": logreg_params,
        "artifacts": {
            "pipeline": str(pipeline_path),
            "vectorizer": str(vectorizer_path),
            "logreg": str(logreg_path),
        },
    }

    metadata_path = output_dir / "run_tfidf_logreg_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    if verbose:
        print("[tfidf-logreg] [complete]")
        print(f"[tfidf-logreg] Accuracy: {accuracy:.4f}")
        print(f"[tfidf-logreg] Artifacts: {model_dir}")
        print(f"[tfidf-logreg] Metadata: {metadata_path}\n")

    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train TF-IDF + Logistic Regression baseline on 20 Newsgroups",
    )
    parser.add_argument(
        "--remove",
        nargs="*",
        default=[],
        choices=["headers", "footers", "quotes"],
        help="Elements to remove from the text",
    )
    args = parser.parse_args()

    train_dataset = load_20newsgroups(subset="train", remove=args.remove)
    test_dataset = load_20newsgroups(subset="test", remove=args.remove)

    metadata = train_and_evaluate(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        model_dir=Path(__file__).resolve().parent / "artifacts",
        output_dir=Path(__file__).resolve().parent / "configs",
    )

    print("Logistic regression baseline training complete.")
    print(f"Accuracy: {metadata['metrics']['accuracy']:.4f}")
    print(
        "Run metadata saved to: "
        f"{Path(__file__).resolve().parent / 'configs' / 'run_tfidf_logreg_metadata.json'}"
    )


if __name__ == "__main__":
    main()
