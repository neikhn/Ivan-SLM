"""
Lab 2.1 skeleton: implement normalization, streaming and filtering yourself.

Run `python labs/lab_2_1.py --check-normalization` after completing normalize_text.
See docs/07-dataset-pilot.md for source selection and the order of exercises.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from dataclasses import dataclass

import unicodedata


@dataclass(frozen=True)
class SourceConfig:
    """Values come from configs/data_sources.json; language is en or vi."""

    dataset_id: str
    subset: str
    revision: str
    language: str
    split: str = "train"


def normalize_text(text: str) -> str:
    """Return NFC text, normalize CRLF and horizontal whitespace, preserve paragraphs.

    An all-whitespace input returns ''. Keep U+FFFD so filtering can reject it.
    This function handles strings, not tensors, and performs no network or file I/O.
    """
    # TODO 1: Look up unicodedata.normalize("NFC", text).
    text = unicodedata.normalize("NFC", text)
    # TODO 2: Normalize line endings; clean spaces/tabs without deleting paragraph boundaries.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # TODO 3: Strip outer whitespace and collapse multiple empty lines to one blank line.
    text = text.strip()
    raise NotImplementedError("Implement normalize_text first.")


def iter_source(source: SourceConfig) -> Iterator[dict[str, object]]:
    """Yield raw rows from the configured revision; do not list() the whole dataset."""
    # TODO: Import datasets here; call load_dataset with streaming=True and revision=...
    # TODO: Use yield and a bounded shuffle buffer with seed=42; preserve raw row identity.
    # Do not use the Dataset Viewer preview files as a training corpus.
    raise NotImplementedError("Implement the source generator after normalization.")


def select_text(row: dict[str, object], source: SourceConfig, tier: str) -> str | None:
    """Return one selected clean passage or None; keep metadata outside this function.

    tier_1: 1–3 explanatory lead paragraphs, excluding selected tier_2 articles.
    tier_2: CS/Python/AI explanation; generic 'model'/'mô hình' alone is insufficient.
    """
    # TODO: Validate text, retain paragraph boundaries, then normalize and apply quality filters.
    # TODO: Reject empty, replacement-character, list/table-only and code-only passages.
    # TODO: Apply language-specific topic rules. Do not copy a whole row into both tiers.
    raise NotImplementedError("Implement extraction and filtering after inspecting raw examples.")


def check_normalization() -> None:
    """A few public acceptance examples, not a complete quality gate."""
    examples = [
        ("Mo\u0302 hi\u0300nh", "Mô hình"),
        ("  Mô\t  hình  \r\n\r\n  Học máy.  ", "Mô hình\n\nHọc máy."),
        ("A\n\n\n\nB", "A\n\nB"),
        (" \t\r\n ", ""),
        ("Lỗi \ufffd", "Lỗi \ufffd"),
    ]
    for raw, expected in examples:
        actual = normalize_text(raw)
        assert actual == expected, f"Input {raw!r}: expected {expected!r}, got {actual!r}"
        assert normalize_text(actual) == actual, "Normalization must be idempotent."
    print(f"Passed {len(examples)} normalization examples and idempotence checks.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-normalization", action="store_true")
    args = parser.parse_args()
    if args.check_normalization:
        try:
            check_normalization()
        except NotImplementedError as error:
            print(f"TODO: {error}")
            return 2
        return 0
    print("First: inspect source previews, then implement normalize_text and run --check-normalization.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
