from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional, Sequence

from sklearn.datasets import fetch_20newsgroups


def load_20newsgroups(
    subset: str = "train",
    categories: Optional[Sequence[str]] = None,
    remove: Iterable[str] = (),
    shuffle: bool = True,
    random_state: int = 42,
    data_home: Optional[Path] = None,
):
    """Fetch 20 Newsgroups, persist a readable raw copy, and return modeling data.

    Args:
        subset: "train", "test", or "all".
        categories: Optional list of category names to filter.
        remove: Iterable of elements to remove from the text.
        shuffle: Whether to shuffle the data.
        random_state: Random seed for shuffling.
        data_home: Optional root directory for data storage.
            Defaults to project-local data/raw.

    Returns:
        A sklearn.utils.Bunch with data, target, and metadata fields.
    """
    if data_home is None:
        data_home = Path(__file__).resolve().parents[1] / "data" / "raw"
    data_home.mkdir(parents=True, exist_ok=True)

    # Always remove these artifacts so both modeling data and exported raw files are clean.
    required_remove = ("headers", "footers", "quotes")
    requested_remove = tuple(remove)
    effective_remove = tuple(dict.fromkeys((*requested_remove, *required_remove)))

    dataset = fetch_20newsgroups(
        subset=subset,
        categories=list(categories) if categories else None,
        remove=effective_remove,
        shuffle=shuffle,
        random_state=random_state,
        data_home=str(data_home),
    )

    raw_export_dir = data_home / "20newsgroups"
    raw_export_dir.mkdir(parents=True, exist_ok=True)
    categories_key = "all" if not categories else "_".join(sorted(categories))
    export_file = raw_export_dir / f"{subset}_{categories_key}.jsonl"

    with export_file.open("w", encoding="utf-8") as f:
        for text, target in zip(dataset.data, dataset.target):
            row = {
                "text": text,
                "target": int(target),
                "target_name": dataset.target_names[int(target)],
                "subset": subset,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return dataset
