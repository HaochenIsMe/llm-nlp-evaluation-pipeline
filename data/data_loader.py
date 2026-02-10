from __future__ import annotations

import argparse
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
    """Fetch the 20 Newsgroups dataset using scikit-learn.

    Args:
        subset: "train", "test", or "all".
        categories: Optional list of category names to filter.
        remove: Iterable of elements to remove from the text. Options include
            "headers", "footers", and "quotes".
        shuffle: Whether to shuffle the data.
        random_state: Random seed for shuffling.
        data_home: Optional directory to store the downloaded dataset.
            Defaults to a project-local data directory.

    Returns:
        A sklearn.utils.Bunch with data, target, and metadata fields.
    """
    if data_home is None:
        data_home = Path(__file__).resolve().parent / "raw"
    data_home.mkdir(parents=True, exist_ok=True)

    return fetch_20newsgroups(
        subset=subset,
        categories=list(categories) if categories else None,
        remove=tuple(remove),
        shuffle=shuffle,
        random_state=random_state,
        data_home=str(data_home),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the 20 Newsgroups dataset via scikit-learn",
    )
    parser.add_argument(
        "--subset",
        default="train",
        choices=["train", "test", "all"],
        help="Dataset split to download",
    )
    parser.add_argument(
        "--remove",
        nargs="*",
        default=[],
        choices=["headers", "footers", "quotes"],
        help="Elements to remove from the text",
    )
    args = parser.parse_args()

    dataset = load_20newsgroups(subset=args.subset, remove=args.remove)
    print(f"Subset: {args.subset}")
    print(f"Documents: {len(dataset.data)}")
    print(f"Categories: {len(dataset.target_names)}")
    print(f"Example category: {dataset.target_names[dataset.target[0]]}")
    print("Example text preview:")
    print(dataset.data[0][:500])


if __name__ == "__main__":
    main()
