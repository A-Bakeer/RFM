"""Write and execute the two analysis notebooks, so they always match the pipeline.

Run from the project folder, after src/build_data.py:  python notebooks/build_notebooks.py
"""
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent

SETUP = """
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import plotly.express as px, plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "notebook_connected"
pio.templates.default = "plotly_white"
pd.set_option("display.max_columns", 40, "display.width", 180, "display.float_format", lambda v: f"{v:,.2f}")
ROOT = Path.cwd().parent
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "data" / "processed"
sales = pd.read_parquet(OUT / "sales.parquet"); returns = pd.read_parquet(OUT / "returns.parquet")
rfm = pd.read_parquet(OUT / "rfm.parquet"); products = pd.read_parquet(OUT / "products.parquet")
invoices = pd.read_parquet(OUT / "invoices.parquet"); cohorts = pd.read_parquet(OUT / "cohorts.parquet")
gaps = pd.read_parquet(OUT / "gaps.parquet"); affinity = pd.read_parquet(OUT / "affinity.parquet")
audit = json.loads((OUT / "audit.json").read_text()); evals = json.loads((OUT / "cluster_eval.json").read_text())
sales["CustomerID"] = sales["CustomerID"].astype(str)
print(f"{len(sales):,} clean sale lines, {sales['Invoice'].nunique():,} invoices, {len(rfm):,} scored customers")
"""


def build(path, cells):
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(c[1].strip()) if c[0] == "md" else nbf.v4.new_code_cell(c[1].strip()) for c in cells]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(HERE)}}).execute()
    nbf.write(nb, HERE / path)
    print("wrote", path)


cleaning = [
    ("md", """
# 01. Cleaning and exploratory analysis: Online Retail II

## 1. Business understanding
The company is a UK online retailer of giftware, and many of its customers are themselves trade buyers. Leadership wants to know where revenue comes from, how much of it can be trusted, and which customers and products need action. Clustering comes later, in notebook 02, once these questions have answers.

## 2. Dataset loading and documentation
The source is `online_retail_II.xlsx`, with two sheets covering 1 Dec 2009 to 9 Dec 2011. Each row is **a line on an invoice, not an order**. The fields are Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID and Country. Prices are in pounds sterling.

Reading the Excel file takes over a minute, so `src/build_data.py` does it once and writes parquet tables. This notebook reads those tables, and the logic lives in `src/cleaning.py` and `src/analytics.py`.
"""),
    ("code", SETUP),
    ("md", """
## 3. Data audit
What the raw data looks like, before any decision.
"""),
    ("code", """
raw = pd.read_parquet(OUT / "raw.parquet")
print(raw.shape); print(raw.isna().sum())
print("Exact duplicate rows:", raw.duplicated().sum())
print("Invoices starting with C:", raw["Invoice"].astype(str).str.startswith("C").sum())
print("Invoices starting with A:", raw["Invoice"].astype(str).str.startswith("A").sum())
print("Price <= 0:", (raw["Price"] <= 0).sum(), "  Quantity < 0:", (raw["Quantity"] < 0).sum())
raw[["Quantity", "Price"]].describe(percentiles=[.01, .5, .99])
"""),
    ("md", """
**Reading this.** The raw sheets overlap: both contain 1 to 9 Dec 2010, so simply joining them double counts a week. Extreme quantities and negative values also need explaining before they are removed.
"""),
    ("md", """
## 4. Cleaning and data quality decisions

| Issue | Decision | Business reason |
|---|---|---|
| Sheets overlap on 1 to 9 Dec 2010 | Keep the second sheet for that week | Otherwise a peak-season week counts twice |
| Exact duplicate rows | Removed, with their value recorded | Most likely double scans. If any are real, revenue is slightly understated |
| Missing Customer ID | Grouped by invoice first, then kept for revenue and left out of RFM | They are whole guest checkouts and still real sales |
| `C` invoices (cancellations) | Kept in a separate table and matched to earlier sales | Removing them would overstate revenue |
| Price of 0 or less | Isolated | Giveaways, damage write-offs and stock adjustments are not sales |
| `A` invoices (bad debt) | Isolated | Accounting entries |
| Postage, fees and manual entries | Isolated | Postage should not rank as a top product |
"""),
    ("code", """
missing = raw["CustomerID"].isna()
g = missing.groupby(raw["Invoice"].astype(str)).agg(["sum", "size"])
print("Invoices with no ID on any line:", ((g["sum"] == g["size"])).sum())
print("Invoices mixing ID and no ID:   ", (((g["sum"] > 0) & (g["sum"] < g["size"]))).sum())
pd.Series(audit, name="value").to_frame()
"""),
    ("md", """
**Guest checkouts.** No invoice mixes lines with and without an ID, so missing IDs are a property of the whole invoice and not random gaps. That supports keeping them for revenue and products. They cannot enter RFM, because there is no customer to score.

**Cancellations.** Each return is matched to the same customer's most recent earlier purchase of the same item, and the quantity is taken off that sale. A return with no earlier purchase in the data, because the sale predates Dec 2009 or the buyer has no ID, stays unmatched: it counts in the return rate but is not subtracted from net revenue.
"""),
    ("code", """
print(f"Return lines fully matched: {audit['return_rows_fully_matched']:,} of {audit['return_rows_goods']:,}")
print(f"Gross revenue {audit['gross_revenue']:,.0f}, net revenue {audit['net_revenue']:,.0f}")
print(f"Unmatched return value {audit['return_value_unmatched']:,.0f}")
returns.sort_values("ReturnValue", ascending=False).head(5)[["Invoice", "Description", "ReturnQty", "Price", "CustomerID", "MatchedQty"]]
"""),
    ("md", """
## 5. Feature engineering
`TotalLineRevenue = Quantity * Price`, `NetRevenue` after matched returns, date parts (`Year`, `Month`, `DayOfWeek`, `Hour`), and the `Is_UK` flag. CustomerID is stored as text, so it can never enter a model as a number.
"""),
    ("code", """
sales[["Invoice", "StockCode", "Quantity", "Price", "TotalLineRevenue", "ReturnedQty", "NetRevenue", "Year", "Month", "DayOfWeek", "Hour", "Is_UK"]].head()
"""),
    ("md", "## 6. EDA\n### Sales over time: how does revenue change, and is there a season?"),
    ("code", """
m = sales.groupby("YearMonth")["NetRevenue"].sum().reset_index()
px.bar(m, x="YearMonth", y="NetRevenue", title="Monthly net revenue (December 2011 stops on the 9th)").show()
y1 = sales[(sales.InvoiceDate >= "2009-12-01") & (sales.InvoiceDate < "2010-12-01")].NetRevenue.sum()
y2 = sales[(sales.InvoiceDate >= "2010-12-01") & (sales.InvoiceDate < "2011-12-01")].NetRevenue.sum()
cal = sales[sales.Year == 2010]
print(f"Like-for-like years: {y1:,.0f} then {y2:,.0f} ({y2 / y1 - 1:+.1%})")
print(f"Oct to Dec share of 2010: {cal[cal.Month >= 10].NetRevenue.sum() / cal.NetRevenue.sum():.1%}")
"""),
    ("md", """
**Reading this.** Revenue climbs from September and peaks in November both years. Orders peak before December, which fits retailers stocking up ahead of Christmas. The last bar is a partial month and must not be read as a fall.
"""),
    ("md", "### Time: which months, weekdays and hours are strongest?"),
    ("code", """
d = pd.DataFrame({"Day": invoices.InvoiceDate.dt.day_name(), "Hour": invoices.InvoiceDate.dt.hour})
order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
grid = d.groupby(["Day", "Hour"]).size().unstack(fill_value=0).reindex(order).fillna(0)
px.imshow(grid, aspect="auto", title="Orders by weekday and hour", color_continuous_scale="Blues").show()
print(d.Day.value_counts())
sat = invoices[invoices.InvoiceDate.dt.day_name() == "Saturday"]
print("Saturday orders fall on", sat.InvoiceDate.dt.date.nunique(), "date:", sat.InvoiceDate.dt.date.unique())
"""),
    ("md", """
**Reading this.** Thursday is the strongest day, and around noon is the peak. Saturday has only 30 orders in two years, all on one date, so this looks like a store policy, not low demand. Nights and Saturdays suit maintenance. Weekday mornings are the natural slot to test ads, though these data show when people order and not whether ads would work then.
"""),
    ("md", "### Products: units, revenue, and is 80/20 true here?"),
    ("code", """
p = products[products.Revenue > 0].sort_values("Revenue", ascending=False)
display(p.head(10)[["StockCode", "Description", "Revenue", "Units", "Orders"]])
display(products.sort_values("Units", ascending=False).head(10)[["StockCode", "Description", "Units", "Revenue", "AvgPrice"]])
cum = p.Revenue.cumsum() / p.Revenue.sum(); x = np.arange(1, len(p) + 1) / len(p)
n80 = (cum <= .8).sum() + 1
px.line(x=x, y=cum, labels={"x": "Share of products", "y": "Cumulative revenue"}, title=f"Pareto curve: {n80 / len(p):.1%} of products make 80% of revenue").show()
"""),
    ("md", """
**Reading this.** The classic 80/20 rule is a close description, not an exact one: about 22% of products make 80% of revenue. The units list and the revenue list differ, because cheap items lead on volume. Plan stock from units and margin from revenue.
"""),
    ("md", "### Dead stock: high earlier volume, nothing in the final 90 days"),
    ("code", """
dead = products[products.IsDead].sort_values("UnitsBefore90", ascending=False)
print(len(dead), "products; earlier units:", f"{dead.UnitsBefore90.sum():,.0f}")
dead.head(10)[["StockCode", "Description", "UnitsBefore90", "UnitsLast90", "LastSold"]]
"""),
    ("md", """
**Reading this.** The final 90 days cover the seasonal peak, so a quiet season does not explain this. Many stopped selling long before, so some are probably deliberate range changes. The data has no stock levels, so this list says what to check, not what is actually sitting in the warehouse.
"""),
    ("md", "### Customers: who contributes the most?"),
    ("code", """
display(rfm.sort_values("Monetary", ascending=False).head(10)[["CustomerID", "Country", "Monetary", "Frequency"]])
display(rfm.sort_values("Frequency", ascending=False).head(10)[["CustomerID", "Country", "Frequency", "Monetary"]])
s = rfm.Monetary.sort_values(ascending=False)
print(f"Top 1% of customers: {s.iloc[:len(s) // 100].sum() / s.sum():.1%} of registered revenue")
print(f"Top 10% of customers: {s.iloc[:len(s) // 10].sum() / s.sum():.1%}")
rfm[["Recency", "Frequency", "Monetary", "AOV", "Products"]].describe(percentiles=[.25, .5, .75, .9])
"""),
    ("md", """
**Typical customer.** The median customer has about 3 orders and spends roughly £845, last ordering about three months before the data ends. The mean spend is far higher than the median because a few accounts are very large, so the median is the better description of a typical customer.
"""),
    ("md", "### Geography: which countries lead?"),
    ("code", """
g = sales.groupby("Country", observed=True).agg(Revenue=("NetRevenue", "sum"), Orders=("Invoice", "nunique"))
g["Customers"] = rfm.groupby("Country").size()
g = g.sort_values("Revenue", ascending=False); g["Share"] = g.Revenue / g.Revenue.sum()
display(g.head(12))
intl = g.drop("United Kingdom").reset_index()
px.choropleth(intl, locations="Country", locationmode="country names", color="Revenue", title="Revenue outside the UK (EIRE and RSA need the mapping used in the app)", color_continuous_scale="Blues").show()
"""),
    ("md", """
**Reading this.** The UK makes about 85% of revenue, so overall trends are mostly UK trends. EIRE's high revenue from very few customers suggests a handful of trade accounts, which is a concentration risk worth checking.
"""),
    ("md", "### Baskets: is there wholesale behaviour?"),
    ("code", """
print(invoices[["Value", "Units", "Lines"]].describe(percentiles=[.5, .9, .99]))
bulk = invoices[invoices.IsBulk]
print(f"Bulk cut-off: {invoices.Units.quantile(.9):,.0f} units. Bulk is {len(bulk) / len(invoices):.0%} of orders and {bulk.Value.sum() / invoices.Value.sum():.0%} of revenue")
px.histogram(invoices[invoices.Value < 5000], x="Value", nbins=80, title="Order value (orders under £5,000)").show()
"""),
    ("md", """
**Reading this.** A median order of nearly 300 units is far beyond household shopping, so much of this business is trade buying. The top tenth of orders by units bring about 42% of revenue.
"""),
    ("md", "### Returns: how common, and who is involved?"),
    ("code", """
print(f"Returned value is {audit['return_value_total'] / audit['gross_revenue']:.1%} of gross sales")
display(products[products.SoldUnits >= 500].sort_values("ReturnRate", ascending=False).head(10)[["Description", "SoldUnits", "ReturnedUnits", "ReturnRate"]])
r = returns[returns.CustomerID.astype(str) != ""].groupby("CustomerID", observed=True).ReturnValue.sum().sort_values(ascending=False)
display(r.head(10).to_frame())
print(f"Top 10 customers make {r.head(10).sum() / r.sum():.0%} of customer-linked return value")
"""),
    ("md", """
## 7. Business insights
These are computed from the data above. They describe what happened and do not establish why.

1. **Seasonality.** Q4 brings well over a third of annual revenue, peaking in November. Commit stock and carrier capacity before September.
2. **Growth.** Comparing like-for-like 12-month windows gives a truer picture than comparing calendar years, because December 2011 is incomplete.
3. **Product concentration.** About 22% of products make 80% of revenue. Protect availability there first.
4. **Long tail.** The last 5% of revenue comes from about half the range, which makes it the first place to review for range cuts.
5. **Geography.** The UK is about 85% of revenue. EIRE, the Netherlands and Germany lead abroad.
6. **Wholesale behaviour.** A tenth of orders bring about 42% of revenue, so volume pricing fits better than consumer promotions.
7. **Customer concentration.** The top 10% of customers bring about 63% of registered revenue, and losing one large account would show in the monthly figures.
8. **Ordering rhythm.** Nearly all orders arrive on weekdays and Sundays in working hours, which gives clear maintenance windows.
9. **Returns.** Returns are a modest 3.6% of gross but concentrated, and single large cancellations cause the spikes.
10. **Dead stock.** 139 high-volume lines sold nothing in the final 90 days, during the peak. Check which were delisted on purpose.
11. **Guest orders.** About 13% of sales have no customer record, so they cannot be targeted later.
12. **Repeat buying.** The median gap between orders is about 25 days, a better trigger for reminders than a fixed monthly send.

Continue in `02_rfm_clustering.ipynb`.
"""),
]

clustering = [
    ("md", """
# 02. RFM analysis and customer segmentation

## 8. Customer-level feature engineering
Each customer becomes one row. Guests are excluded because they have no identity. Customers whose purchases were entirely returned are excluded too, because they have no net spend to score.
"""),
    ("code", SETUP),
    ("code", """
snapshot = pd.Timestamp(audit["snapshot_date"]); print("Scored as of", snapshot.date())
rfm[["CustomerID", "Recency", "Frequency", "Monetary", "AOV", "Products", "Tenure", "Country"]].head()
"""),
    ("md", """
## 9. RFM analysis
- **Recency** is days from the last purchase to the snapshot date (10 Dec 2011, the day after the data ends). **Lower is better**, so the score is reversed: a score of 5 means most recent.
- **Frequency** is the number of distinct invoices with at least one line not fully returned. Higher is better.
- **Monetary** is net revenue after returns. Higher is better.

Scores split customers into five equal groups with `pd.qcut`. Frequency is heavily tied, with many customers at one or two orders, so plain `qcut` cannot form five bins. Ranking first breaks the ties, which means customers with the same frequency can land in neighbouring scores. A score is therefore a rank within this customer base, not an absolute level.
"""),
    ("code", """
display(rfm[["R", "F", "M"]].apply(pd.Series.value_counts).sort_index())
print(rfm.groupby("R").Recency.agg(["min", "max", "median"]))
print(rfm.groupby("F").Frequency.agg(["min", "max", "median"]))
print(rfm.groupby("M").Monetary.agg(["min", "max", "median"]))
rfm["RFM"].value_counts().head(10)
"""),
    ("md", """
## 10. RFM visualisation and interpretation
Segment rules, first match wins, so every customer gets exactly one segment:

| Segment | Rule | Reasoning |
|---|---|---|
| Champions | R=5, F>=4, M>=4 | The brief's 555 alone is too narrow, so this keeps the top on all three with some tolerance |
| Loyal customers | R>=4, F>=3, M>=3 | Recent and solid, without being top on everything |
| At-risk high value | R<=2, F>=3, M>=3 | Strong history, but gone quiet |
| Cooling regulars | R=3, F>=3, M>=3 | Added from the data: a large group between loyal and at-risk, and early action costs less |
| New inflow | R>=4, F<=2, M<=2 | Recent first purchases |
| Promising | R>=3, not above | Recent but modest |
| Lost one-timers | R=1, F<=2 | Long gone with little history |
| Hibernating | Everyone else | Old and low value |
"""),
    ("code", """
seg = rfm.groupby("Segment").agg(Customers=("CustomerID", "size"), Revenue=("Monetary", "sum"), MedianR=("Recency", "median"), MedianF=("Frequency", "median"), MedianM=("Monetary", "median"))
seg["CustomerShare"] = seg.Customers / seg.Customers.sum(); seg["RevenueShare"] = seg.Revenue / seg.Revenue.sum()
seg = seg.sort_values("Revenue", ascending=False); display(seg)
long = seg.reset_index().melt(id_vars="Segment", value_vars=["CustomerShare", "RevenueShare"])
px.bar(long, y="Segment", x="value", color="variable", barmode="group", orientation="h", title="Share of customers against share of revenue").show()
px.imshow(rfm.pivot_table(index="R", columns="F", values="CustomerID", aggfunc="size").sort_index(ascending=False), text_auto=True, color_continuous_scale="Blues", title="Customers by recency and frequency score").show()
"""),
    ("md", """
**Reading this.** Champions are about 12.5% of customers and over half of revenue. At-risk high-value customers have median recency over a year, so this is past spend that could be worth trying to recover, not revenue currently at stake. Recency and frequency scores are correlated, which is why customers cluster along the grid's diagonal.
"""),
    ("md", """
## 11. Preprocessing for clustering
All three measures are strongly right-skewed. `log1p` compresses that, and `StandardScaler` then gives each mean 0 and variance 1. Without scaling, spend in pounds would dominate days and order counts in any distance calculation. CustomerID is never a feature.
"""),
    ("code", """
print("Skew before log:", rfm[["Recency", "Frequency", "Monetary"]].skew().round(2).to_dict())
print("Skew after log: ", rfm[["logR", "logF", "logM"]].skew().round(2).to_dict())
print("Scaled means:   ", rfm[["zR", "zF", "zM"]].mean().round(3).to_dict())
print("Scaled std:     ", rfm[["zR", "zF", "zM"]].std().round(3).to_dict())
"""),
    ("md", "## 12. Clustering and evaluation"),
    ("code", """
km = pd.DataFrame(evals["kmeans"]); wd = pd.DataFrame(evals["ward"]); db = evals["dbscan"]
display(km.round(3)); display(wd.round(3).T)
print("DBSCAN:", db)
px.line(km, x="k", y="wcss", markers=True, title="Elbow").show()
fig = go.Figure(); fig.add_scatter(x=km.k, y=km.silhouette, name="K-means"); fig.add_scatter(x=wd.k, y=wd.silhouette, name="Ward, sample of 3,000")
fig.update_layout(title="Silhouette"); fig.show()
"""),
    ("md", """
**Choosing k.** The elbow bends gently with no clear point, so it cannot decide alone. Silhouette is highest at k=2, but that only separates active from lapsed customers, which marketing already knows. Among more useful counts, k=4 (0.357) and k=5 (0.343) are nearly the same. **We choose k=5** because results are the same across random starts (adjusted Rand index 1.00), the smallest group is still about 8% of customers, and it splits out a small top tier that makes over half of revenue and needs different treatment. This is a business decision backed by the metrics, and the metrics alone do not force it. Silhouette around 0.34 also means the clusters overlap: customers fall along a continuum, not into sharply separate groups.

**Comparing methods.**

| Method | Result here | Use |
|---|---|---|
| K-means | Stable, interpretable, best silhouette at every k | The segmentation |
| Ward hierarchical | Same broad shape, slightly lower silhouette, sample only | Check on k and structure |
| DBSCAN | One dense mass plus 55 outliers | A list of whale accounts to manage personally, not segments |
"""),
    ("md", "## 13. Segment profiling"),
    ("code", """
prof = rfm.groupby("k5").agg(Customers=("CustomerID", "size"), MedianR=("Recency", "median"), MedianF=("Frequency", "median"), MedianM=("Monetary", "median"), Revenue=("Monetary", "sum"))
prof["CustomerShare"] = prof.Customers / prof.Customers.sum(); prof["RevenueShare"] = prof.Revenue / prof.Revenue.sum()
names = {0: "Dormant one-timers", 1: "Recent starters", 2: "Lapsed regulars", 3: "Active regulars", 4: "Elite repeat buyers"}
prof.insert(0, "Name", prof.index.map(names)); display(prof)
display(pd.crosstab(rfm.k5.map(names), rfm.Segment, normalize="index").round(2))
outliers = rfm[rfm.IsNoise].sort_values("Monetary", ascending=False)
display(outliers.head(8)[["CustomerID", "Country", "Recency", "Frequency", "Monetary"]])
"""),
    ("md", """
**Reading this.** Names come from the medians in the table above, so each one can be checked against the numbers. Rule-based segments and clusters mostly agree: Elite repeat buyers are largely Champions, and Dormant one-timers are largely Hibernating and Lost. Two independent methods landing on similar groups gives more confidence in the structure than either would alone.
"""),
    ("md", """
## 14. Business recommendations

| Group | Evidence | Action | How to measure it |
|---|---|---|---|
| Champions / Elite | About 12.5% of customers, over half of revenue, median 8 days since last order | Loyalty perks, early access, referrals. Avoid blanket discounts | Retention and order frequency against a control group |
| Whale accounts (DBSCAN) | 55 accounts, 26% of revenue | Named account manager and a churn alert when an account goes quiet | Revenue kept among these accounts |
| Loyal customers | About 17% of customers, 21% of revenue | Cross-sell from ranges they have not bought, using the pair list | Items per order |
| Cooling regulars | Median 94 days since last order, against a 25-day reorder gap | Reminder timed to the usual reorder gap | Reactivation rate |
| At-risk high value | £1.33m past spend, median over a year since last order | Direct win-back test with a holdout group, starting with the largest | Reactivated revenue against offer cost |
| New inflow | Recent, one order | Welcome sequence aimed at a second order | Second-order rate within 60 days |
| Hibernating / Lost | About 30% of customers, under 5% of revenue | Low-cost seasonal mail only | Cost per reactivated customer |

Beyond customer groups: commit Q4 stock before September, review dead stock and the long tail, and capture contact details on guest orders. All of these should be tested, because this analysis shows association, not cause.
"""),
    ("md", """
## 15. Limitations and next steps
- **No cost or margin data.** Revenue is not profit, so a large customer may not be a profitable one.
- **No stock data.** Dead stock can only be inferred from sales.
- **Truncated history.** Customers who bought before Dec 2009 look new at first, and the December 2009 cohort is inflated. December 2011 is partial.
- **Guest orders** (about 13% of sales) are outside RFM. The same buyer could also appear under two accounts.
- **Country** is the customer's most common invoice country, which may be a billing location.
- **Exact duplicates** were removed. Any that were real orders are lost, worth under 0.3% of revenue.
- **RFM is a snapshot** as of 10 Dec 2011, just after the Christmas buying season, which favours autumn buyers on recency.
- **Returns** are netted at the original sale price, and unmatched returns are not netted at all.

Next steps: add margin data, test the win-back and second-order campaigns with holdout groups, model churn and lifetime value, and refresh scores monthly.

## 16. Final conclusion
Once the double-counted week, cancellations and non-product entries are dealt with, this is a £18.9m trade-heavy UK business with a sharp autumn peak. Revenue is concentrated in a fifth of products and a tenth of customers, which makes protecting those accounts and that stock the first priority. RFM and clustering point to the same groups, and the most useful next action is a measured test: retain the top tier, try to recover at-risk high-value customers, and turn more first orders into second ones.
"""),
]

if __name__ == "__main__":
    build("01_cleaning_eda.ipynb", cleaning)
    build("02_rfm_clustering.ipynb", clustering)
