"""Inspect a few raw source rows. This utility does not build the pilot dataset."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "data_sources.json"
PREVIEW_DIR = ROOT / "data" / "source_preview"


def read_remote_json(url: str) -> dict:
    """Read public JSON metadata or a small Dataset Viewer response."""
    request = Request(url, headers={"User-Agent": "Ivan-SLM-source-inspection/1.0"})
    with urlopen(request, timeout=30) as response:
        payload = response.read(2_000_001)
    if len(payload) > 2_000_000:
        raise ValueError("Preview response exceeds 2 MB; use fewer rows.")
    return json.loads(payload)


def fetch_preview(source: dict, limit: int) -> dict:
    """Save a Viewer snapshot, explicitly separate from revision-pinned training data."""
    dataset_id = source["dataset_id"]
    metadata = read_remote_json("https://huggingface.co/api/datasets/" + dataset_id)
    query = urlencode(
        {
            "dataset": dataset_id,
            "config": source["subset"],
            "split": source["split"],
            "offset": 0,
            "length": limit,
        }
    )
    url = "https://datasets-server.huggingface.co/rows?" + query
    page = read_remote_json(url)
    if not page.get("rows"):
        raise ValueError(f"No preview rows returned for {source['key']}.")
    snapshot = {
        "dataset": dataset_id,
        "config": source["subset"],
        "language": source["language"],
        "split": source["split"],
        "observed_revision": metadata["sha"],
        "license_metadata": metadata.get("cardData", {}).get("license"),
        "preview_revision_verified": False,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "preview_url": url,
        "notice": "Viewer snapshot only. Not filtered, not reviewed, not a pinned training shard.",
        "rows": page["rows"],
    }
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    path = PREVIEW_DIR / f"{source['key']}.json"
    temporary_path = path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary_path.replace(path)
    return snapshot


def main() -> int:
    """Print only small excerpts; networking requires the explicit --fetch flag."""
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    sources = {source["key"]: source for source in config["sources"]}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["all", *sources], default="all")
    parser.add_argument("--fetch", action="store_true", help="Download fresh unfiltered Viewer previews.")
    parser.add_argument("--limit", type=int, choices=range(1, 11), default=3)
    args = parser.parse_args()
    selected = list(sources.values()) if args.source == "all" else [sources[args.source]]
    failed = False
    for source in selected:
        path = PREVIEW_DIR / f"{source['key']}.json"
        try:
            snapshot = fetch_preview(source, args.limit) if args.fetch else json.loads(
                path.read_text(encoding="utf-8")
            )
            rows = snapshot["rows"][:args.limit]
            print(f"\n{source['key']} | {source['language']} | {len(rows)} raw examples")
            print(f"Configured training revision: {source['revision']}")
            print(f"Observed source revision: {snapshot['observed_revision']}")
            print("Viewer preview is NOT verified against the pinned training revision.")
            for item in rows:
                row = item["row"]
                print(f"Row {item['row_idx']} | fields: {', '.join(row)}")
                if "title" in row:
                    print(f"Title: {row['title']}")
                else:
                    print(f"Format: {row.get('format')} | audience: {row.get('audience')}")
                print(row[source["text_field"]][:240].replace("\n", " "))
        except (OSError, ValueError, KeyError) as error:
            failed = True
            print(f"{source['key']}: {error}", file=sys.stderr)
            if not args.fetch:
                print("Use --fetch to create the missing preview.", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
