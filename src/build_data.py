"""Turn the Excel workbook into the parquet tables the app reads.

Run once:  python src/build_data.py path/to/online_retail_II.xlsx
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import analytics as an  # noqa: E402
import cleaning as cl  # noqa: E402

DEFAULT_XLSX = r"D:\NTI\NTI-Student-Train\ML\Databases\online+retail+ii\online_retail_II.xlsx"


def main(xlsx: str) -> None:
    out = ROOT / "data" / "processed"
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()

    raw_cache = out / "raw.parquet"
    audit_cache = out / "raw_audit.json"
    if raw_cache.exists() and audit_cache.exists():
        import pandas as pd
        raw, audit = pd.read_parquet(raw_cache), json.loads(audit_cache.read_text())
    else:
        raw, audit = cl.load_raw(xlsx)
        raw.astype({"Invoice": str, "StockCode": str, "Description": "string"}).to_parquet(raw_cache)
        audit_cache.write_text(json.dumps(audit))
    print(f"loaded {len(raw):,} rows in {time.time() - start:.0f}s")

    sales, returns = cl.clean(raw, audit)
    sales, returns = cl.match_returns(sales, returns, audit)
    sales, returns = cl.add_features(sales), cl.add_features(returns)

    rfm, snapshot = an.build_rfm(sales)
    rfm, evaluation = an.cluster_customers(rfm)
    products = an.build_products(sales, returns)
    affinity = an.build_affinity(sales)
    cohorts = an.build_cohorts(sales)
    invoices = an.build_invoices(sales)
    gaps = an.build_purchase_gaps(invoices)
    audit["snapshot_date"] = str(snapshot.date())

    categorical = ["Invoice", "StockCode", "Description", "CustomerID", "Country"]
    sales.astype({c: "category" for c in categorical}).to_parquet(out / "sales.parquet")
    returns.astype({c: "category" for c in categorical}).to_parquet(out / "returns.parquet")
    rfm.to_parquet(out / "rfm.parquet")
    products.to_parquet(out / "products.parquet")
    affinity.to_parquet(out / "affinity.parquet")
    cohorts.to_parquet(out / "cohorts.parquet")
    invoices.to_parquet(out / "invoices.parquet")
    gaps.to_parquet(out / "gaps.parquet")
    (out / "audit.json").write_text(json.dumps(audit, indent=2))
    (out / "cluster_eval.json").write_text(json.dumps(evaluation, indent=2))
    print(f"built everything in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_XLSX)
