from __future__ import annotations

import argparse

from data.data_loader import load_20newsgroups


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entry point: run data loader only",
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

    print(f"Train documents: {len(train_dataset.data)}")
    print(f"Test documents: {len(test_dataset.data)}")
    print(f"Categories: {len(train_dataset.target_names)}")


if __name__ == "__main__":
    main()
