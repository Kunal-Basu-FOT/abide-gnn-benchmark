"""Extract the two reference benchmark tables from the archived notebook ZIP."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import zipfile
from pathlib import Path

NOTEBOOKS = {
    "AAL": "GNN_Work/run-benchmark-gnn-aal-early-stop-top-10-percent.ipynb",
    "CC200": "GNN_Work/run-benchmark-gnn-cc200-early-stop-top-10-percent.ipynb",
}
MODELS = ("MLP", "GCN", "ChebNet", "GIN", "GraphSAGE", "GAT")
METRICS = ("Accuracy", "F1", "AUROC", "AUPRC", "Balanced_Acc", "Sensitivity", "Specificity")


def table_from_notebook(raw: bytes, atlas: str) -> list[dict[str, str]]:
    notebook = json.loads(raw)
    tables = []
    for cell in notebook["cells"]:
        for output in cell.get("outputs", []):
            text = output.get("text", [])
            text = "".join(text) if isinstance(text, list) else text
            if "BENCHMARK SUMMARY TABLE" in text and f"Atlas: {atlas}" in text:
                tables.append(text)
    if not tables:
        raise ValueError(f"No {atlas} benchmark table in notebook output")
    text = tables[-1]
    rows = []
    for model in MODELS:
        match = re.search(rf"^\s*{model}\s+((?:\d+\.\d+\s*){{7}})$", text, re.M)
        if not match:
            raise ValueError(f"Missing {model} row for {atlas}")
        values = match.group(1).split()
        rows.append({"Atlas": atlas, "Model": model, **dict(zip(METRICS, values))})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="GNN_Work ZIP containing the saved notebooks")
    parser.add_argument("--output", type=Path, default=Path("results/reference/benchmark_summary.csv"))
    args = parser.parse_args()

    rows = []
    provenance = []
    with zipfile.ZipFile(args.archive) as archive:
        for atlas, name in NOTEBOOKS.items():
            raw = archive.read(name)
            rows.extend(table_from_notebook(raw, atlas))
            provenance.append(f"- `{name}` — SHA-256 `{hashlib.sha256(raw).hexdigest()}`")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("Atlas", "Model", *METRICS))
        writer.writeheader()
        writer.writerows(rows)
    (args.output.parent / "SOURCE.md").write_text(
        "# Reference table provenance\n\n"
        "The table was extracted from the saved console outputs of two archived "
        "early-stopping notebooks. It was not recomputed from subject data.\n\n"
        + "\n".join(provenance) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
