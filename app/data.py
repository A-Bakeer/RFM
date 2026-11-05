"""Data access for the app. Base tables load once; each filter combination is cached."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

OUT = Path(__file__).resolve().parent.parent / "data" / "processed"

ISO3 = {
    "United Kingdom": "GBR", "EIRE": "IRL", "Germany": "DEU", "France": "FRA", "Netherlands": "NLD", "Spain": "ESP",
    "Switzerland": "CHE", "Belgium": "BEL", "Portugal": "PRT", "Australia": "AUS", "Italy": "ITA", "Norway": "NOR",
    "Sweden": "SWE", "Cyprus": "CYP", "Finland": "FIN", "Austria": "AUT", "Denmark": "DNK", "Greece": "GRC",
    "Japan": "JPN", "Poland": "POL", "USA": "USA", "United Arab Emirates": "ARE", "Israel": "ISR", "Hong Kong": "HKG",
    "Singapore": "SGP", "Malta": "MLT", "Iceland": "ISL", "Canada": "CAN", "Lithuania": "LTU", "RSA": "ZAF",
    "Bahrain": "BHR", "Brazil": "BRA", "Thailand": "THA", "Korea": "KOR", "Lebanon": "LBN", "Bermuda": "BMU",
    "Nigeria": "NGA", "Czech Republic": "CZE", "Saudi Arabia": "SAU",
}
# Channel Islands, West Indies, European Community and Unspecified have no single map shape.

MARKETS = ["All markets", "UK", "International"]


@st.cache_resource(show_spinner="Opening the ledger")
def base() -> dict:
    missing = [n for n in ("sales", "returns", "rfm", "products") if not (OUT / f"{n}.parquet").exists()]
    if missing:
        st.error("Processed data not found. Run this once from the project folder: python src/build_data.py <path to online_retail_II.xlsx>")
        st.stop()
    names = ["sales", "returns", "rfm", "products", "affinity", "cohorts", "invoices", "gaps"]
    tables = {n: pd.read_parquet(OUT / f"{n}.parquet") for n in names}
    tables["sales"]["CustomerID"] = tables["sales"]["CustomerID"].astype(str)
    tables["audit"] = json.loads((OUT / "audit.json").read_text())
    tables["clusters"] = json.loads((OUT / "cluster_eval.json").read_text())
    return tables


def months() -> list[pd.Timestamp]:
    return sorted(base()["sales"]["YearMonth"].unique())


def _scope(df: pd.DataFrame, start, end, market: str, date_col: str = "YearMonth") -> pd.DataFrame:
    mask = (df[date_col] >= start) & (df[date_col] <= end)
    if market == "UK":
        mask &= df["Is_UK"]
    elif market == "International":
        mask &= ~df["Is_UK"]
    return df[mask]


@st.cache_data(show_spinner=False)
def summary(start, end, market: str) -> dict:
    t = base()
    s = _scope(t["sales"], start, end, market)
    r = _scope(t["returns"], start, end, market)
    live = s[s["NetQty"] > 0]
    gross = float(s["TotalLineRevenue"].sum())
    orders = live["Invoice"].astype(str).nunique()
    customers = live.loc[live["CustomerID"].ne(""), "CustomerID"].nunique()
    net = float(s["NetRevenue"].sum())
    units = float(s["NetQty"].sum())
    returned = float(r["ReturnValue"].sum())
    return {
        "net": net, "gross": gross, "orders": orders, "customers": customers, "units": units,
        "aov": net / orders if orders else 0.0, "upo": units / orders if orders else 0.0,
        "returned": returned, "return_rate": returned / gross if gross else 0.0,
    }


@st.cache_data(show_spinner=False)
def monthly(start, end, market: str) -> pd.DataFrame:
    t = base()
    s = _scope(t["sales"], start, end, market)
    r = _scope(t["returns"], start, end, market)
    g = s.groupby("YearMonth").agg(Net=("NetRevenue", "sum"), Gross=("TotalLineRevenue", "sum"))
    g["Orders"] = s[s["NetQty"] > 0].groupby("YearMonth")["Invoice"].nunique()
    g["Returned"] = r.groupby("YearMonth")["ReturnValue"].sum()
    g["Returned"] = g["Returned"].fillna(0)
    g["ReturnRate"] = g["Returned"] / g["Gross"]
    return g.reset_index()


@st.cache_data(show_spinner=False)
def by_country(start, end, market: str) -> pd.DataFrame:
    t = base()
    s = _scope(t["sales"], start, end, market)
    r = _scope(t["returns"], start, end, market)
    g = s.groupby("Country", observed=True).agg(Net=("NetRevenue", "sum"), Gross=("TotalLineRevenue", "sum"))
    live = s[s["NetQty"] > 0]
    g["Orders"] = live.groupby("Country", observed=True)["Invoice"].nunique()
    reg = live[live["CustomerID"].ne("")]
    g["Customers"] = reg.groupby("Country", observed=True)["CustomerID"].nunique()
    g["Returned"] = r.groupby("Country", observed=True)["ReturnValue"].sum()
    g = g.fillna(0)
    g = g[g["Gross"] > 0]
    g["AOV"] = g["Net"] / g["Orders"].replace(0, np.nan)
    g["ReturnRate"] = g["Returned"] / g["Gross"]
    g["ISO"] = g.index.map(ISO3)
    return g.reset_index().sort_values("Net", ascending=False)


@st.cache_data(show_spinner=False)
def country_products(country: str, start, end) -> pd.DataFrame:
    s = _scope(base()["sales"], start, end, "All markets")
    s = s[s["Country"] == country]
    g = s.groupby("Description", observed=True).agg(Net=("NetRevenue", "sum"), Units=("NetQty", "sum"))
    return g.sort_values("Net", ascending=False).head(6).reset_index()


def _invoices(start, end, market: str) -> pd.DataFrame:
    inv = base()["invoices"]
    period = inv["InvoiceDate"].dt.to_period("M").dt.to_timestamp()
    mask = (period >= start) & (period <= end)
    if market == "UK":
        mask &= inv["Country"] == "United Kingdom"
    elif market == "International":
        mask &= inv["Country"] != "United Kingdom"
    return inv[mask]


@st.cache_data(show_spinner=False)
def rhythm(start, end, market: str) -> pd.DataFrame:
    inv = _invoices(start, end, market)
    g = pd.DataFrame({"Day": inv["InvoiceDate"].dt.dayofweek, "Hour": inv["InvoiceDate"].dt.hour, "Value": inv["Value"]})
    return g.groupby(["Day", "Hour"]).agg(Orders=("Value", "size"), Net=("Value", "sum")).reset_index()


@st.cache_data(show_spinner=False)
def baskets(start, end, market: str) -> pd.DataFrame:
    return _invoices(start, end, market)[["Value", "Units", "Lines", "IsBulk"]].copy()


@st.cache_data(show_spinner=False)
def new_vs_returning(start, end, market: str) -> pd.DataFrame:
    t = base()
    s = _scope(t["sales"], start, end, market)
    s = s[s["CustomerID"].ne("")]
    g = s.groupby(["YearMonth", "CustomerID"], observed=True)["NetRevenue"].sum().reset_index()
    first = t["rfm"].set_index("CustomerID")["FirstPurchase"].dt.to_period("M").dt.to_timestamp()
    g["First"] = g["CustomerID"].map(first)
    g["Kind"] = np.where(g["First"] == g["YearMonth"], "New", "Returning")
    return g.groupby(["YearMonth", "Kind"])["NetRevenue"].sum().unstack(fill_value=0).reset_index()


@st.cache_data(show_spinner=False)
def product_table(start, end, market: str) -> pd.DataFrame:
    s = _scope(base()["sales"], start, end, market)
    g = s.groupby("StockCode", observed=True).agg(
        Description=("Description", "first"), Net=("NetRevenue", "sum"), Units=("NetQty", "sum"),
        Orders=("Invoice", "nunique"), Price=("Price", "median"),
    )
    return g[g["Net"] > 0].sort_values("Net", ascending=False).reset_index()


@st.cache_data(show_spinner=False)
def search_products(query: str) -> pd.DataFrame:
    """Every word must appear, in any order, in the name or stock code."""
    p = base()["products"]
    p = p[p["Revenue"] > 0].sort_values("Revenue", ascending=False)
    words = query.upper().split()
    if not words:
        return p.head(8)
    text = p["Description"].astype(str).str.upper() + " " + p["StockCode"].astype(str).str.upper()
    mask = pd.Series(True, index=p.index)
    for w in words:
        mask &= text.str.contains(w, regex=False)
    return p[mask]


@st.cache_data(show_spinner=False)
def product_detail(code: str) -> dict:
    t = base()
    s = t["sales"]
    s = s[s["StockCode"] == code]
    r = t["returns"]
    r = r[r["StockCode"] == code]
    trend = s.groupby("YearMonth").agg(Net=("NetRevenue", "sum"), Units=("NetQty", "sum")).reset_index()
    full = pd.DataFrame({"YearMonth": months()})
    trend = full.merge(trend, how="left").fillna(0)
    countries = s.groupby("Country", observed=True).agg(Net=("NetRevenue", "sum"), Units=("NetQty", "sum")).reset_index()
    countries = countries[countries["Net"] > 0]
    countries["ISO"] = countries["Country"].map(ISO3)
    reg = s[s["CustomerID"].ne("")]
    buyers = reg.groupby("CustomerID", observed=True).agg(Net=("NetRevenue", "sum"), Units=("NetQty", "sum"), Orders=("Invoice", "nunique"))
    buyers = buyers.sort_values("Net", ascending=False).head(8).reset_index()
    a = t["affinity"]
    pairs = a[a["A"] == code].sort_values("Lift", ascending=False).head(8)
    names = t["products"].set_index("StockCode")["Description"]
    pairs = pairs.assign(Name=pairs["B"].map(names))
    return {
        "trend": trend, "countries": countries, "buyers": buyers, "pairs": pairs,
        "returned": float(r["ReturnValue"].sum()),
    }


KEYWORDS = ["CHRISTMAS", "HEART", "VINTAGE", "RETROSPOT", "BAG", "CAKE", "LIGHT", "TEA", "PARTY", "PAPER",
            "BOX", "GARDEN", "BUNTING", "CANDLE", "MUG", "WOODLAND", "SIGN", "GLASS", "DOORMAT", "CHILDRENS"]


@st.cache_data(show_spinner=False)
def keyword_themes(start, end, market: str) -> pd.DataFrame:
    p = product_table(start, end, market)
    text = p["Description"].astype(str).str.upper()
    total = p["Net"].sum()
    rows = [{"Theme": k.capitalize(), "Net": p.loc[text.str.contains(k, regex=False), "Net"].sum(),
             "Products": int(text.str.contains(k, regex=False).sum())} for k in KEYWORDS]
    out = pd.DataFrame(rows)
    out["Share"] = out["Net"] / total
    return out[out["Net"] > 0].sort_values("Net", ascending=False)


@st.cache_data(show_spinner=False)
def findings() -> list[tuple[str, str, str]]:
    """Headline findings computed from the full data, so the text cannot drift from the numbers."""
    t = base()
    s, rfm, p, inv, a, c = t["sales"], t["rfm"], t["products"], t["invoices"], t["audit"], t["cohorts"]
    money = lambda v: f"£{v / 1e6:.2f}m" if abs(v) >= 1e6 else f"£{v / 1e3:.0f}k"
    out = []

    cal = s[s["Year"] == 2010]
    q4 = cal.loc[cal["Month"] >= 10, "NetRevenue"].sum() / cal["NetRevenue"].sum()
    peak = cal.groupby("Month")["NetRevenue"].sum().idxmax()
    out.append((f"Q4 brings {q4:.0%} of the year's revenue",
                f"In calendar 2010, October to December made {q4:.0%} of net revenue, and {pd.Timestamp(2010, int(peak), 1):%B} was the largest month.",
                "Buy stock and confirm carrier capacity before September."))

    y1 = s[(s["InvoiceDate"] >= "2009-12-01") & (s["InvoiceDate"] < "2010-12-01")]["NetRevenue"].sum()
    y2 = s[(s["InvoiceDate"] >= "2010-12-01") & (s["InvoiceDate"] < "2011-12-01")]["NetRevenue"].sum()
    out.append((f"Revenue moved {y2 / y1 - 1:+.1%} year on year",
                f"{money(y1)} in Dec 2009 to Nov 2010, against {money(y2)} in Dec 2010 to Nov 2011. December 2011 is excluded because the data stops on the 9th.",
                "Use these like-for-like windows in any growth claim. The raw last bar in the trend is a partial month."))

    ranked = p[p["Revenue"] > 0].sort_values("Revenue", ascending=False)
    cum = ranked["Revenue"].cumsum() / ranked["Revenue"].sum()
    n80 = int((cum <= 0.8).sum()) + 1
    out.append((f"{n80 / len(ranked):.0%} of products make 80% of revenue",
                f"{n80:,} of {len(ranked):,} products. That is the Pareto shape, close to the classic 80/20 but not exactly it.",
                "Protect availability on this group first, and review the long tail for range cuts."))

    uk = s.loc[s["Is_UK"], "NetRevenue"].sum() / s["NetRevenue"].sum()
    intl = s[~s["Is_UK"]].groupby("Country", observed=True)["NetRevenue"].sum().sort_values(ascending=False)
    out.append((f"The UK makes {uk:.0%} of revenue",
                f"The largest export markets are {intl.index[0]}, {intl.index[1]} and {intl.index[2]}, with {money(intl.iloc[:3].sum())} between them, which is {intl.iloc[:3].sum() / intl.sum():.0%} of everything sold abroad.",
                "Read overall trends as UK trends unless the International filter is on."))

    bulk_rev = inv.loc[inv["IsBulk"], "Value"].sum() / inv["Value"].sum()
    cut = inv["Units"].quantile(0.9)
    out.append((f"Bulk orders are {inv['IsBulk'].mean():.0%} of orders and {bulk_rev:.0%} of revenue",
                f"Orders of {cut:,.0f} units or more count as bulk. The median order is {inv['Units'].median():,.0f} units worth £{inv['Value'].median():,.0f}, far above household shopping.",
                "This behaves like a wholesale business, so volume pricing fits better than consumer promotions."))

    m = rfm.sort_values("Monetary", ascending=False)["Monetary"]
    top1 = m.iloc[: max(1, len(m) // 100)].sum() / m.sum()
    top10 = m.iloc[: max(1, len(m) // 10)].sum() / m.sum()
    out.append((f"The top 10% of customers bring {top10:.0%} of registered revenue",
                f"The top 1% alone bring {top1:.0%}. DBSCAN flags {t['clusters']['dbscan']['noise_customers']} outlier accounts holding {t['clusters']['dbscan']['noise_revenue_share']:.0%} of revenue.",
                "Give the largest accounts a named contact. Losing one would show up in the monthly figures."))

    seg = rfm.groupby("Segment")["Monetary"].agg(["size", "sum"])
    champ = seg.loc["Champions"]
    out.append((f"Champions are {champ['size'] / seg['size'].sum():.0%} of customers and {champ['sum'] / seg['sum'].sum():.0%} of revenue",
                f"{int(champ['size']):,} customers scored top on recency, frequency and spend.",
                "Reward them with access and service. Discounting mostly gives away margin they already pay."))

    risk = seg.loc["At-risk high value"]
    out.append((f"{money(risk['sum'])} of past spend sits with at-risk customers",
                f"{int(risk['size']):,} customers with strong order histories whose last purchase is now old. Their median gap since buying is {rfm.loc[rfm['Segment'] == 'At-risk high value', 'Recency'].median():.0f} days.",
                "Test a win-back campaign on this group first and hold out a control group, so the effect can be measured."))

    one = (rfm["Frequency"] == 1).mean()
    m1 = c[c["Offset"] == 1]
    m1 = m1[m1["Cohort"] > "2009-12-01"]["Retention"].median()
    out.append((f"{one:.0%} of customers ordered only once",
                f"Excluding the first cohort, a median of {m1:.0%} of a month's new customers order again the next month.",
                "The second order is the main growth gap. Put onboarding effort there."))

    gaps = t["gaps"]["Gap"]
    out.append((f"Repeat customers reorder about every {gaps.median():.0f} days",
                f"That is the median gap between orders. One gap in four is longer than {gaps.quantile(0.75):.0f} days.",
                "Use this as the trigger for reorder reminders, instead of a fixed monthly send."))

    names = inv["InvoiceDate"].dt.day_name()
    day = names.value_counts()
    hour = inv["InvoiceDate"].dt.hour.value_counts()
    saturday = int(day.get("Saturday", 0))
    core = inv["InvoiceDate"].dt.hour.between(9, 15).mean()
    out.append((f"{day.index[0]} at {hour.index[0]}:00 is the busiest time, and Saturday has {'no' if saturday == 0 else 'almost no'} orders",
                f"Only {saturday:,} orders were placed on a Saturday in two years, all on {inv.loc[names == 'Saturday', 'InvoiceDate'].dt.date.nunique()} date, and {core:.0%} of orders fall between 09:00 and 16:00.",
                "Schedule maintenance overnight or on Saturday, and test ad spend on weekday mornings."))

    rate = a["return_value_total"] / a["gross_revenue"]
    returns_by = p.sort_values("ReturnedUnits", ascending=False)["ReturnedUnits"]
    share = returns_by.iloc[: max(1, len(returns_by) // 100)].sum() / returns_by.sum()
    out.append((f"Returns take {rate:.1%} of gross sales, concentrated in few products",
                f"The top 1% of products account for {share:.0%} of all returned units, and {a['return_rows_fully_matched'] / a['return_rows_goods']:.0%} of return lines match an earlier purchase by the same customer.",
                "Check listings and packaging on the worst offenders on the Products page."))

    dead = p[p["IsDead"]]
    out.append((f"{len(dead)} high-volume products sold nothing in the final 90 days",
                f"Together they sold {dead['UnitsBefore90'].sum():,.0f} units earlier on. The window is the seasonal peak, so a quiet season does not explain it.",
                "Confirm which were deliberately delisted, then clear any remaining stock."))

    guest = a["guest_sale_value"] / a["gross_revenue"]
    out.append((f"Guest orders make {guest:.0%} of sales with no customer record",
                f"{a['guest_invoices_all_lines']:,} invoices have no ID on any line, and {a['guest_invoices_mixed']} mix guest and registered lines, so these are whole guest checkouts rather than stray gaps.",
                "They count in revenue and product figures but not in RFM. Capturing an email at checkout would make them reachable."))

    aff = t["affinity"]
    aff = aff[aff["Together"] >= 200].sort_values("Lift", ascending=False)
    if len(aff):
        best = aff.iloc[0]
        names = p.set_index("StockCode")["Description"]
        out.append(("Some products sell strongly as pairs",
                    f"{names.get(best['A'])} and {names.get(best['B'])} appear together {best['Lift']:.0f} times more often than chance would predict, on {int(best['Together']):,} orders.",
                    "Offer these as bundles and as add-ons at checkout. This shows association only, so test whether a bundle changes sales."))
    return out
