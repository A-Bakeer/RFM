"""Cleaning for the Online Retail II workbook.

Every step records what it removed in an audit dict, because each decision changes
reported revenue and the reader should be able to see by how much.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

# Stock codes that are charges or bookkeeping entries rather than goods on a shelf.
# Keeping them out stops postage and manual adjustments ranking as "top products".
SERVICE_CODES = {
    "POST", "DOT", "M", "C2", "D", "S", "B", "BANK CHARGES", "ADJUST", "ADJUST2",
    "AMAZONFEE", "TEST001", "TEST002", "PADS", "CRUK", "GIFT",
}
SERVICE_PREFIXES = ("GIFT_0001", "TEST", "ADJUST")

# Overlap point between the two sheets: Sheet 1 runs to 9 Dec 2010, Sheet 2 starts 1 Dec 2010.
SHEET_BOUNDARY = pd.Timestamp("2010-12-01")


def load_raw(path: str) -> tuple[pd.DataFrame, dict]:
    """Read both sheets and resolve the overlapping week so it is not counted twice."""
    sheets = pd.read_excel(path, sheet_name=None)
    first, second = sheets["Year 2009-2010"], sheets["Year 2010-2011"]
    audit = {"raw_rows_sheet1": len(first), "raw_rows_sheet2": len(second)}

    keep_first = first[first["InvoiceDate"] < SHEET_BOUNDARY]
    audit["overlap_rows_removed"] = len(first) - len(keep_first)
    df = pd.concat([keep_first, second], ignore_index=True)
    df = df.rename(columns={"Customer ID": "CustomerID"})
    audit["rows_after_overlap"] = len(df)
    return df, audit


def is_service_code(codes: pd.Series) -> pd.Series:
    upper = codes.str.upper()
    return upper.isin(SERVICE_CODES) | upper.str.startswith(SERVICE_PREFIXES)


def clean(df: pd.DataFrame, audit: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (sale lines, cancellation lines) with every exclusion counted in `audit`."""
    df = df.copy()
    df["Invoice"] = df["Invoice"].astype(str).str.strip()
    df["StockCode"] = df["StockCode"].astype(str).str.strip().str.upper()
    df["Description"] = df["Description"].astype("string").str.strip()
    df["Country"] = df["Country"].astype(str).str.strip()
    df["TotalLineRevenue"] = df["Quantity"] * df["Price"]

    # CustomerID is an identifier: a string, never a number that could enter a model.
    df["CustomerID"] = df["CustomerID"].map(lambda v: "" if pd.isna(v) else str(int(v)))

    dup_mask = df.duplicated()
    audit["exact_duplicate_rows"] = int(dup_mask.sum())
    audit["exact_duplicate_value"] = float(df.loc[dup_mask, "TotalLineRevenue"].clip(lower=0).sum())
    df = df[~dup_mask]

    # Guest checkout check: is a missing ID a property of the whole invoice or of stray lines?
    missing = df["CustomerID"].eq("")
    per_invoice = missing.groupby(df["Invoice"]).agg(["sum", "size"])
    audit["guest_rows"] = int(missing.sum())
    audit["guest_invoices_all_lines"] = int((per_invoice["sum"] == per_invoice["size"]).sum())
    audit["guest_invoices_mixed"] = int(
        ((per_invoice["sum"] > 0) & (per_invoice["sum"] < per_invoice["size"])).sum()
    )
    audit["total_invoices"] = int(len(per_invoice))

    is_cancel = df["Invoice"].str.startswith("C")
    is_debt = df["Invoice"].str.startswith("A")
    bad_price = df["Price"] <= 0
    is_service = is_service_code(df["StockCode"])

    audit["cancel_rows"] = int(is_cancel.sum())
    audit["bad_debt_rows"] = int(is_debt.sum())
    audit["zero_or_negative_price_rows"] = int((bad_price & ~is_cancel).sum())
    audit["non_cancel_negative_qty_rows"] = int(((df["Quantity"] < 0) & ~is_cancel).sum())

    sale_candidates = ~is_cancel & ~is_debt & ~bad_price & (df["Quantity"] > 0)
    audit["service_rows"] = int((sale_candidates & is_service).sum())
    audit["service_value"] = float(df.loc[sale_candidates & is_service, "TotalLineRevenue"].sum())

    sales = df[sale_candidates & ~is_service].copy()
    returns = df[is_cancel & ~is_service & (df["Quantity"] < 0)].copy()
    audit["sale_rows"] = len(sales)
    audit["return_rows_goods"] = len(returns)
    audit["guest_sale_rows"] = int(sales["CustomerID"].eq("").sum())
    audit["guest_sale_value"] = float(sales.loc[sales["CustomerID"].eq(""), "TotalLineRevenue"].sum())

    # One name per product. Descriptions drift over time, so the most common spelling wins.
    names = (
        sales.dropna(subset=["Description"])
        .groupby(["StockCode", "Description"]).size().rename("n").reset_index()
        .sort_values("n", ascending=False).drop_duplicates("StockCode")
        .set_index("StockCode")["Description"]
    )
    sales["Description"] = sales["StockCode"].map(names).fillna(sales["Description"]).fillna("Unnamed item")
    returns["Description"] = returns["StockCode"].map(names).fillna(returns["Description"]).fillna("Unnamed item")
    return sales, returns


def match_returns(sales: pd.DataFrame, returns: pd.DataFrame, audit: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Net cancellations against the same customer's most recent earlier purchase of that item.

    Most recent first, because a return usually refers to the latest delivery. Revenue is
    reduced in the month of the original sale. A return with no earlier purchase in the
    data (its sale predates Dec 2009, or the buyer has no ID) stays unmatched and is
    reported separately rather than subtracted from a sale it cannot be tied to.
    """
    sales = sales.sort_values("InvoiceDate").copy()
    returns = returns.sort_values("InvoiceDate").copy()

    keys = set(zip(returns["CustomerID"], returns["StockCode"]))
    stacks: dict = defaultdict(list)
    returned = np.zeros(len(sales), dtype=np.int64)

    relevant = [k in keys for k in zip(sales["CustomerID"], sales["StockCode"])]
    rel = sales.loc[relevant, ["CustomerID", "StockCode", "Quantity", "InvoiceDate"]]
    positions = np.flatnonzero(relevant)
    sale_events = list(zip(rel["CustomerID"], rel["StockCode"], rel["Quantity"], rel["InvoiceDate"], positions))

    matched_qty = np.zeros(len(returns), dtype=np.int64)
    ret_events = list(zip(returns["CustomerID"], returns["StockCode"], -returns["Quantity"], returns["InvoiceDate"]))

    i = 0
    for r_idx, (cust, code, qty, when) in enumerate(ret_events):
        while i < len(sale_events) and sale_events[i][3] <= when:
            c, s, q, _, pos = sale_events[i]
            stacks[(c, s)].append([pos, q])
            i += 1
        if not cust:
            continue  # guest return: there is no identity to tie it to a sale
        stack = stacks.get((cust, code), [])
        left = qty
        while left > 0 and stack:
            pos, avail = stack[-1]
            take = min(left, avail)
            returned[pos] += take
            left -= take
            stack[-1][1] -= take
            if stack[-1][1] == 0:
                stack.pop()
        matched_qty[r_idx] = qty - left

    sales["ReturnedQty"] = returned
    sales["NetQty"] = sales["Quantity"] - sales["ReturnedQty"]
    sales["NetRevenue"] = sales["NetQty"] * sales["Price"]

    returns["ReturnQty"] = -returns["Quantity"]
    returns["MatchedQty"] = matched_qty
    returns["UnmatchedQty"] = returns["ReturnQty"] - returns["MatchedQty"]
    returns["ReturnValue"] = returns["ReturnQty"] * returns["Price"]
    returns["UnmatchedValue"] = returns["UnmatchedQty"] * returns["Price"]

    audit["return_value_total"] = float(returns["ReturnValue"].sum())
    audit["return_value_unmatched"] = float(returns["UnmatchedValue"].sum())
    audit["return_value_matched"] = float((sales["ReturnedQty"] * sales["Price"]).sum())
    audit["return_rows_fully_matched"] = int((returns["UnmatchedQty"] == 0).sum())
    audit["gross_revenue"] = float(sales["TotalLineRevenue"].sum())
    audit["net_revenue"] = float(sales["NetRevenue"].sum())
    return sales, returns


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df["Year"] = df["InvoiceDate"].dt.year.astype("int16")
    df["Month"] = df["InvoiceDate"].dt.month.astype("int8")
    df["DayOfWeek"] = df["InvoiceDate"].dt.dayofweek.astype("int8")
    df["Hour"] = df["InvoiceDate"].dt.hour.astype("int8")
    df["YearMonth"] = df["InvoiceDate"].dt.to_period("M").dt.to_timestamp()
    df["Is_UK"] = df["Country"].eq("United Kingdom")
    return df
