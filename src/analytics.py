"""Customer and product tables built from clean sale lines."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42

SEGMENT_PLAYBOOK = {
    "Champions": "Reward with loyalty perks, early access to new ranges and a referral programme. Avoid discounting: they already buy at full price.",
    "Loyal customers": "Grow basket size by cross-selling from the ranges they have not tried yet.",
    "Cooling regulars": "Solid buyers whose orders are spacing out. Send a reminder timed to their usual reorder gap before they drift further.",
    "New inflow": "Run a welcome sequence aimed at one thing: a second order.",
    "Promising": "Recent but modest buyers. Test bundles and volume price breaks to find out which of them can grow.",
    "At-risk high value": "Contact directly with a win-back offer. These accounts spent heavily and stopped, so they justify the cost of personal outreach.",
    "Hibernating": "Include in low-cost seasonal campaigns only. Stop sending if they do not respond.",
    "Lost one-timers": "Keep out of paid campaigns. Ask a sample why they did not come back.",
}
SEGMENT_ORDER = list(SEGMENT_PLAYBOOK)


def build_rfm(sales: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp]:
    """R, F and M per registered customer, scored 1-5 by quantile."""
    reg = sales[sales["CustomerID"].ne("") & (sales["NetQty"] > 0)]
    snapshot = sales["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)

    rfm = reg.groupby("CustomerID").agg(
        LastPurchase=("InvoiceDate", "max"),
        FirstPurchase=("InvoiceDate", "min"),
        Frequency=("Invoice", "nunique"),
        Monetary=("NetRevenue", "sum"),
        Units=("NetQty", "sum"),
        Products=("StockCode", "nunique"),
        Country=("Country", lambda s: s.mode().iat[0]),
    )
    rfm = rfm[rfm["Monetary"] > 0].copy()
    rfm["Recency"] = (snapshot - rfm["LastPurchase"]).dt.days
    rfm["Tenure"] = (snapshot - rfm["FirstPurchase"]).dt.days
    rfm["AOV"] = rfm["Monetary"] / rfm["Frequency"]

    # Over half the customers share F=1 or 2, so plain qcut cannot make five bins.
    # Ranking first breaks those ties, at the cost of some same-F customers landing in
    # neighbouring scores. The bins still carry equal head counts, as the brief asks.
    def score(series: pd.Series, reverse: bool = False) -> pd.Series:
        ranks = series.rank(method="first", ascending=not reverse)
        return pd.qcut(ranks, 5, labels=[1, 2, 3, 4, 5]).astype(int)

    rfm["R"] = score(rfm["Recency"], reverse=True)   # lower days -> higher score
    rfm["F"] = score(rfm["Frequency"])
    rfm["M"] = score(rfm["Monetary"])
    rfm["RFM"] = rfm["R"].astype(str) + rfm["F"].astype(str) + rfm["M"].astype(str)
    rfm["Segment"] = assign_segments(rfm)
    return rfm.reset_index(), snapshot


def assign_segments(rfm: pd.DataFrame) -> pd.Series:
    """Business rules on the scores. First match wins and every customer gets exactly one."""
    r, f, m = rfm["R"], rfm["F"], rfm["M"]
    rules = [
        ("Champions", (r == 5) & (f >= 4) & (m >= 4)),
        ("Loyal customers", (r >= 4) & (f >= 3) & (m >= 3)),
        ("At-risk high value", (r <= 2) & (f >= 3) & (m >= 3)),
        ("Cooling regulars", (r == 3) & (f >= 3) & (m >= 3)),
        ("New inflow", (r >= 4) & (f <= 2) & (m <= 2)),
        ("Promising", r >= 3),
        ("Lost one-timers", (r == 1) & (f <= 2)),
    ]
    out = pd.Series("Hibernating", index=rfm.index)
    for name, mask in reversed(rules):
        out[mask] = name
    return out


def cluster_customers(rfm: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Compare k-means, Ward and DBSCAN on log-scaled RFM."""
    log_rfm = np.log1p(rfm[["Recency", "Frequency", "Monetary"]])
    X = StandardScaler().fit_transform(log_rfm)
    out = rfm.copy()
    out[["logR", "logF", "logM"]] = log_rfm.values
    out[["zR", "zF", "zM"]] = X

    evaluation = []
    for k in range(2, 11):
        model = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X)
        sil = silhouette_score(X, model.labels_, sample_size=3000, random_state=RANDOM_STATE)
        # Stability: do different seeds find the same grouping? An elbow cannot tell you this.
        alt = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE + 7).fit(X)
        smallest = np.bincount(model.labels_).min() / len(X)
        evaluation.append({
            "k": k, "wcss": float(model.inertia_), "silhouette": float(sil),
            "stability": float(adjusted_rand_score(model.labels_, alt.labels_)),
            "smallest_share": float(smallest),
        })
        if 3 <= k <= 6:
            # Label 0 is always the weakest group, so colours stay put across reruns.
            spend = pd.Series(X[:, 2]).groupby(model.labels_).mean().sort_values()
            remap = {old: new for new, old in enumerate(spend.index)}
            out[f"k{k}"] = pd.Series(model.labels_).map(remap).values

    # Ward on a sample: linkage needs memory quadratic in rows.
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(X), size=min(3000, len(X)), replace=False)
    Z = linkage(X[idx], method="ward")
    ward = []
    for k in range(2, 11):
        labels = fcluster(Z, k, criterion="maxclust")
        ward.append({"k": k, "silhouette": float(silhouette_score(X[idx], labels))})

    # DBSCAN: eps from the distance to the 10th neighbour, at its 98th percentile.
    dists, _ = NearestNeighbors(n_neighbors=10).fit(X).kneighbors(X)
    eps = float(np.percentile(dists[:, -1], 98))
    db = DBSCAN(eps=eps, min_samples=10).fit(X)
    out["IsNoise"] = db.labels_ == -1
    noise = out[out["IsNoise"]]
    dbscan = {
        "eps": eps,
        "clusters": int(len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)),
        "noise_customers": int(len(noise)),
        "noise_revenue_share": float(noise["Monetary"].sum() / out["Monetary"].sum()),
        "noise_median_spend": float(noise["Monetary"].median()) if len(noise) else 0.0,
        "noise_high_spend": int((noise["Monetary"] > out["Monetary"].quantile(0.9)).sum()),
    }
    return out, {"kmeans": evaluation, "ward": ward, "dbscan": dbscan}


def build_products(sales: pd.DataFrame, returns: pd.DataFrame, window_days: int = 90) -> pd.DataFrame:
    """One row per product, including dead-stock and momentum flags from the final 90 days."""
    end = sales["InvoiceDate"].max()
    last_start = end - pd.Timedelta(days=window_days)
    prev_start = last_start - pd.Timedelta(days=window_days)

    base = sales.groupby("StockCode").agg(
        Description=("Description", "first"),
        Revenue=("NetRevenue", "sum"),
        GrossRevenue=("TotalLineRevenue", "sum"),
        Units=("NetQty", "sum"),
        SoldUnits=("Quantity", "sum"),
        Orders=("Invoice", "nunique"),
        Customers=("CustomerID", lambda s: s[s.ne("")].nunique()),
        AvgPrice=("Price", "median"),
        FirstSold=("InvoiceDate", "min"),
        LastSold=("InvoiceDate", "max"),
    )
    last = sales[sales["InvoiceDate"] > last_start].groupby("StockCode")
    prev = sales[(sales["InvoiceDate"] > prev_start) & (sales["InvoiceDate"] <= last_start)].groupby("StockCode")
    before = sales[sales["InvoiceDate"] <= last_start].groupby("StockCode")["NetQty"].sum()
    q4 = sales[sales["Month"].isin([10, 11, 12])].groupby("StockCode")["NetRevenue"].sum()

    base["RevLast90"] = last["NetRevenue"].sum()
    base["UnitsLast90"] = last["NetQty"].sum()
    base["RevPrev90"] = prev["NetRevenue"].sum()
    base["UnitsBefore90"] = before
    base["Q4Revenue"] = q4
    base[["RevLast90", "UnitsLast90", "RevPrev90", "UnitsBefore90", "Q4Revenue"]] = base[
        ["RevLast90", "UnitsLast90", "RevPrev90", "UnitsBefore90", "Q4Revenue"]
    ].fillna(0)
    base["Q4Share"] = (base["Q4Revenue"] / base["Revenue"].where(base["Revenue"] > 0)).fillna(0)

    # Count all returned units, matched or not, so the rate reflects what came back.
    ret_units = returns.groupby("StockCode")["ReturnQty"].sum()
    base["ReturnedUnits"] = ret_units
    base["ReturnedUnits"] = base["ReturnedUnits"].fillna(0)
    base["ReturnRate"] = (base["ReturnedUnits"] / base["SoldUnits"]).clip(upper=1)

    # Dead stock: top-quartile historic volume, nothing net sold in the final 90 days.
    # The window is Sept-Dec, the busiest stretch of the year, so silence here is telling.
    volume_cut = base["UnitsBefore90"].quantile(0.75)
    base["IsDead"] = (base["UnitsBefore90"] >= volume_cut) & (base["UnitsLast90"] <= 0)

    ranked = base.sort_values("Revenue", ascending=False)
    share = ranked["Revenue"].clip(lower=0).cumsum() / ranked["Revenue"].clip(lower=0).sum()
    base["ABC"] = pd.cut(share, [-1, 0.8, 0.95, 2], labels=["A", "B", "C"]).reindex(base.index).astype(str)
    return base.reset_index()


def build_affinity(sales: pd.DataFrame, top_n: int = 1500, min_together: int = 30) -> pd.DataFrame:
    """Pairs of products on the same invoice, with support, confidence and lift."""
    lines = sales[sales["NetQty"] > 0][["Invoice", "StockCode"]].drop_duplicates()
    top = lines["StockCode"].value_counts().head(top_n).index
    lines = lines[lines["StockCode"].isin(top)]
    n_invoices = sales["Invoice"].nunique()

    inv = lines["Invoice"].astype("category")
    code = lines["StockCode"].astype("category")
    X = sparse.csr_matrix(
        (np.ones(len(lines), dtype=np.int32), (inv.cat.codes, code.cat.codes)),
        shape=(n_invoices, code.cat.categories.size),
    )
    together = (X.T @ X).tocoo()
    counts = np.asarray(X.sum(axis=0)).ravel()
    keep = (together.row != together.col) & (together.data >= min_together)
    a, b, both = together.row[keep], together.col[keep], together.data[keep]
    labels = np.asarray(code.cat.categories)
    return pd.DataFrame({
        "A": labels[a], "B": labels[b], "Together": both,
        "Confidence": both / counts[a],
        "Lift": both / counts[a] / (counts[b] / n_invoices),
    })


def build_cohorts(sales: pd.DataFrame) -> pd.DataFrame:
    """Share of each monthly first-purchase cohort that bought again N months later."""
    reg = sales[sales["CustomerID"].ne("") & (sales["NetQty"] > 0)][["CustomerID", "YearMonth"]].drop_duplicates()
    first = reg.groupby("CustomerID")["YearMonth"].min().rename("Cohort")
    reg = reg.join(first, on="CustomerID")
    reg["Offset"] = (reg["YearMonth"].dt.year - reg["Cohort"].dt.year) * 12 + (reg["YearMonth"].dt.month - reg["Cohort"].dt.month)
    table = reg.groupby(["Cohort", "Offset"])["CustomerID"].nunique().rename("Customers").reset_index()
    sizes = table[table["Offset"] == 0].set_index("Cohort")["Customers"].rename("Size")
    table = table.join(sizes, on="Cohort")
    table["Retention"] = table["Customers"] / table["Size"]
    return table


def build_invoices(sales: pd.DataFrame) -> pd.DataFrame:
    """One row per order. A row in the raw data is a line on an order, not an order."""
    live = sales[sales["NetQty"] > 0]
    inv = live.groupby("Invoice").agg(
        InvoiceDate=("InvoiceDate", "first"),
        CustomerID=("CustomerID", "first"),
        Country=("Country", "first"),
        Value=("NetRevenue", "sum"),
        Units=("NetQty", "sum"),
        Lines=("StockCode", "nunique"),
    ).reset_index()
    inv["UnitsPerLine"] = inv["Units"] / inv["Lines"]
    inv["IsBulk"] = inv["Units"] >= inv["Units"].quantile(0.90)
    return inv


def build_purchase_gaps(invoices: pd.DataFrame) -> pd.DataFrame:
    """Days between consecutive orders for each registered customer."""
    reg = invoices[invoices["CustomerID"].ne("")].sort_values(["CustomerID", "InvoiceDate"])
    gaps = reg.groupby("CustomerID")["InvoiceDate"].diff().dt.total_seconds() / 86400
    return pd.DataFrame({"CustomerID": reg["CustomerID"], "Gap": gaps}).dropna()
