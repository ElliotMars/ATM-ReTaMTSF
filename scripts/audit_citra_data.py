"""Inspect local CSV metadata without treating event dates as availability.

Standard library only. This is a metadata diagnostic, not a proof that any
timestamp is authentic; verified provenance still requires source evidence.
"""

import argparse
import csv
import json
from pathlib import Path


DOMAINS = ("Energy", "Health_US", "Environment", "Climate")
FIELDS = ("issued_at", "available_at", "ingested_at", "valid_from", "valid_to")


def audit(root):
    result = []
    for domain in DOMAINS:
        directory = root / "textual" / domain
        csvs = sorted(directory.glob("*.csv")) if directory.exists() else []
        files = []
        for path in csvs:
            with path.open(encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                columns = reader.fieldnames or []
                populated = {key: 0 for key in FIELDS}
                rows = 0
                for row in reader:
                    rows += 1
                    for key in FIELDS:
                        populated[key] += bool((row.get(key) or "").strip())
            files.append({"file": str(path), "columns": columns, "rows": rows,
                          "time_metadata_populated": populated,
                          "missing_time_columns": [key for key in FIELDS if key not in columns],
                          "availability_provenance": "unverified"})
        embeddings = len(list(directory.glob("NewsDatabase-embedding*/News-*.npy")))
        result.append({"domain": domain, "raw_csvs": files, "embedding_files": embeddings,
                       "status": "metadata_requires_source_verification" if files
                       else "raw_data_missing", "strict_backtest_ready": False})
    return {"root": str(root), "domains": result,
            "policy": "Never infer issued_at/available_at/ingested_at from start_date or file mtime."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("dataset/Time-MMD"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.dumps(audit(args.root), indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
