"""Page 3: customers, RFM segments and clustering."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.cluster.hierarchy import dendrogram, linkage

import data
import theme as th

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from analytics import SEGMENT_PLAYBOOK  # noqa: E402

tables = data.base()
full_rfm = tables["rfm"]
snapshot = tables["audit"]["snapshot_date"]
_, _, market = st.session_state["scope"]
rfm = full_rfm
if market == "UK":
    rfm = full_rfm[full_rfm["Country"] == "United Kingdom"]
elif market == "International":
    rfm = full_rfm[full_rfm["Country"] != "United Kingdom"]

m_sorted = rfm["Monetary"].sort_values(ascending=False)
top10 = m_sorted.iloc[: max(1, len(m_sorted) // 10)].sum() / m_sorted.sum()
at_risk = rfm[rfm["Segment"] == "At-risk high value"]
th.envelope(
    f"The top 10% of customers bring {top10:.0%} of registered revenue",
    f"Scored as of {snapshot}. RFM covers registered customers across the full two years, so the period filter does not apply on this page; the market filter does.",
    [
        th.stamp(th.num(len(rfm)), "Customers scored", lead=True),
        th.stamp(th.pct((rfm["Frequency"] > 1).mean(), 0), "Ordered more than once"),
        th.stamp(th.money(rfm["Monetary"].median()), "Median spend"),
        th.stamp(th.num(rfm["Frequency"].median()), "Median orders"),
        th.stamp(th.num(rfm["Recency"].median()), "Median days since last order"),
        th.stamp(th.money(at_risk["Monetary"].sum()), "Past spend, at-risk group"),
    ],
)

view = st.segmented_control("Section", ["RFM segments", "Clusters", "Behaviour"], default="RFM segments", key="cust_view", label_visibility="collapsed") or "RFM segments"


# ---------- RFM ----------
def segment_table(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("Segment").agg(Customers=("CustomerID", "size"), Revenue=("Monetary", "sum"), R=("Recency", "median"), F=("Frequency", "median"), M=("Monetary", "median"))
    g["CustShare"] = g["Customers"] / g["Customers"].sum()
    g["RevShare"] = g["Revenue"] / g["Revenue"].sum()
    return g.reindex([s for s in th.SEGMENTS if s in g.index]).reset_index()


@st.fragment
def segments_view():
    g = segment_table(rfm)
    best = g.sort_values("RevShare", ascending=False).iloc[0]
    th.section(f"{best['Segment']} are {best['CustShare']:.0%} of customers and {best['RevShare']:.0%} of revenue",
               "Each customer gets a score of 1 to 5 on three measures: R for how recently they ordered (5 is most recent), F for how often, and M for how much they spent. Scores split customers into five equal groups, so a score is a rank within this customer base and not an absolute level.")
    with st.container(key="panel_seg"):
        o = g.sort_values("RevShare")
        fig = go.Figure()
        fig.add_bar(y=o["Segment"], x=o["CustShare"], orientation="h", name="Share of customers", marker=dict(color=th.tok("line-strong"), cornerradius=4),
                    customdata=o["Customers"], hovertemplate="<b>%{y}</b><br>%{x:.1%} of customers (%{customdata:,})<extra></extra>")
        fig.add_bar(y=o["Segment"], x=o["RevShare"], orientation="h", name="Share of revenue", marker=dict(color=th.series(0), cornerradius=4),
                    customdata=o["Revenue"], hovertemplate="<b>%{y}</b><br>%{x:.1%} of revenue (£%{customdata:,.0f})<extra></extra>")
        th.style(fig, 340, legend=True, barmode="group", bargap=.25, bargroupgap=.08)
        fig.update_xaxes(tickformat=".0%", showgrid=True, gridcolor=th.tok("line"))
        fig.update_yaxes(showgrid=False, type="category")
        th.show(fig, "seg_share")

    th.section("What to do with each segment", "Pick a segment to see its evidence, its action and its customer list. The rules behind each name are in the notebook.")
    pick = st.pills("Segment", list(g["Segment"]), default=list(g["Segment"])[0], key="seg_pick", label_visibility="collapsed") or list(g["Segment"])[0]
    df = rfm[rfm["Segment"] == pick]
    row = g[g["Segment"] == pick].iloc[0]
    left, right = st.columns([1.1, 1], gap="large")
    with left:
        st.html(f'<div class="play" style="--seg:{th.segment_color(pick)}"><h3>{pick}</h3><p>{SEGMENT_PLAYBOOK[pick]}</p></div>')
        st.html(f"""<div class="factline"><div><b>{th.num(row['Customers'])}</b>customers</div><div><b>{th.money(row['Revenue'])}</b>total spend</div>
<div><b>{row['R']:.0f} days</b>median since last order</div><div><b>{row['F']:.0f}</b>median orders</div><div><b>{th.money(row['M'])}</b>median spend</div></div>""")
        top = df.sort_values("Monetary", ascending=False).head(10)
        th.sub("Largest accounts in this segment")
        th.table(top, [{"key": "CustomerID", "label": "Customer", "text": True}, {"key": "Country", "label": "Country", "text": True}, {"key": "RFM", "label": "Score"},
                       {"key": "Recency", "label": "Days ago", "fmt": th.num}, {"key": "Frequency", "label": "Orders", "fmt": th.num}, {"key": "Monetary", "label": "Spend", "fmt": th.money, "bar": True}], 300)
        export = df[["CustomerID", "Country", "Recency", "Frequency", "Monetary", "R", "F", "M", "RFM", "Segment"]]
        st.download_button(f"Download {len(export):,} customers as CSV", export.to_csv(index=False).encode(), f"{pick.lower().replace(' ', '_')}.csv", "text/csv", key="dl_seg")
    with right:
        c = df.groupby("Country").agg(Customers=("CustomerID", "size"), Spend=("Monetary", "sum")).reset_index()
        c["ISO"] = c["Country"].map(data.ISO3)
        abroad = c[(c["Country"] != "United Kingdom") & c["ISO"].notna()]
        uk_share = c.loc[c["Country"] == "United Kingdom", "Customers"].sum() / max(c["Customers"].sum(), 1)
        th.sub(f"Where they are: {uk_share:.0%} in the UK, the rest shown here", "Colour shows total spend. The UK is left out so other countries can be compared.")
        if abroad.empty:
            th.empty("All customers in this view are in the UK. Set the market to All markets to compare countries.")
        else:
            fig = go.Figure(go.Choropleth(locations=abroad["ISO"], z=np.log10(abroad["Spend"].clip(lower=1)), text=abroad["Country"], customdata=np.stack([abroad["Spend"], abroad["Customers"]], axis=1),
                                          colorscale=th.seq(), showscale=False, marker=dict(line=dict(color=th.tok("panel"), width=.5)),
                                          hovertemplate="<b>%{text}</b><br>£%{customdata[0]:,.0f}<br>%{customdata[1]} customers<extra></extra>"))
            th.style(fig, 330, margin=dict(l=0, r=0, t=0, b=0),
                     geo=dict(bgcolor="rgba(0,0,0,0)", showframe=False, showcoastlines=False, showland=True, landcolor=th.tok("line"), showcountries=True, countrycolor=th.tok("panel"),
                              projection_type="natural earth", lataxis_range=[-50, 75]))
            th.show(fig, f"seg_map_{pick}")

    th.section("How the scores spread", "Customers mostly sit along the diagonal: those who bought recently also tend to buy often. The corners are where the actions differ.")
    a, b = st.columns([1, 1.25], gap="large")
    with a:
        measure = st.segmented_control("Cell shows", ["Customers", "Average spend"], default="Customers", key="grid_measure") or "Customers"
        if measure == "Customers":
            grid = rfm.pivot_table(index="R", columns="F", values="CustomerID", aggfunc="size").reindex(index=range(1, 6), columns=range(1, 6)).fillna(0)
            texts, tip = grid.map(lambda v: f"{v:,.0f}"), "%{z:,.0f} customers"
        else:
            grid = rfm.pivot_table(index="R", columns="F", values="Monetary", aggfunc="mean").reindex(index=range(1, 6), columns=range(1, 6)).fillna(0)
            texts, tip = grid.map(th.money), "£%{z:,.0f} average spend"
        fig = go.Figure(go.Heatmap(z=np.log1p(grid.values) if measure != "Customers" else grid.values, x=[f"F{i}" for i in range(1, 6)], y=[f"R{i}" for i in range(1, 6)], text=texts.values, texttemplate="%{text}",
                                   textfont=dict(size=12), colorscale=th.seq(), showscale=False, xgap=3, ygap=3, customdata=grid.values,
                                   hovertemplate="%{y}, %{x}<br><b>" + tip.replace("%{z", "%{customdata") + "</b><extra></extra>"))
        th.style(fig, 330)
        fig.update_xaxes(title_text="Frequency score", side="bottom")
        fig.update_yaxes(title_text="Recency score", showgrid=False)
        th.show(fig, f"grid_{measure}")
    with b:
        th.sub("The raw measures behind the scores", "All three lean heavily to the right, which is why they are logged before clustering.")
        sel = st.segmented_control("Measure", ["Recency", "Frequency", "Monetary"], default="Monetary", key="rfm_dist") or "Monetary"
        vals = rfm[sel]
        fig = go.Figure()
        if sel == "Recency":
            fig.add_histogram(x=vals, nbinsx=40, marker=dict(color=th.series(0), line=dict(color=th.tok("panel"), width=1)), hovertemplate="%{x} days<br><b>%{y:,} customers</b><extra></extra>")
            th.style(fig, 262)
            fig.update_xaxes(title_text="Days since last order")
        else:
            edges = np.logspace(0, np.log10(vals.max()), 34)
            counts, _ = np.histogram(vals, bins=edges)
            fig.add_bar(x=np.sqrt(edges[:-1] * edges[1:]), y=counts, width=edges[1:] - edges[:-1], marker=dict(color=th.series(0), line=dict(color=th.tok("panel"), width=1)),
                        hovertemplate="About %{x:,.0f}<br><b>%{y:,} customers</b><extra></extra>")
            th.style(fig, 262, xaxis_type="log")
            fig.update_xaxes(title_text="Orders (log scale)" if sel == "Frequency" else "Spend in £ (log scale)")
        fig.update_yaxes(title_text="Customers")
        th.show(fig, f"dist_{sel}")

    th.section("Look up a customer", "Enter a customer ID to see their scores and history. For example: 18102.")
    q = st.text_input("Customer ID", placeholder="Enter a customer ID, then press Enter", key="cust_query", label_visibility="collapsed").strip()
    if q:
        hit = full_rfm[full_rfm["CustomerID"] == q]
        if hit.empty:
            th.empty(f"No registered customer has the ID {q}. IDs are five digits, such as 18102.")
        else:
            lookup(hit.iloc[0])


def lookup(r: pd.Series):
    inv = tables["invoices"]
    orders = inv[inv["CustomerID"] == r["CustomerID"]].sort_values("InvoiceDate")
    st.html(f"""<div class="play" style="--seg:{th.segment_color(r['Segment'])}"><h3>Customer {r['CustomerID']}: {r['Segment']}, score {r['RFM']}</h3><p>{SEGMENT_PLAYBOOK[r['Segment']]}</p></div>
<div class="factline"><div><b>{r['Country']}</b>country</div><div><b>{r['Recency']:,} days</b>since last order</div><div><b>{r['Frequency']:,}</b>orders</div>
<div><b>{th.money(r['Monetary'])}</b>spend</div><div><b>{th.money(r['AOV'])}</b>per order</div><div><b>{r['Products']:,}</b>different products</div></div>""")
    fig = go.Figure(go.Scatter(x=orders["InvoiceDate"], y=orders["Value"], mode="markers", marker=dict(size=9, color=th.segment_color(r["Segment"]), line=dict(width=1.5, color=th.tok("panel"))),
                               customdata=orders["Units"], hovertemplate="%{x|%d %b %Y}<br><b>£%{y:,.0f}</b><br>%{customdata:,} units<extra></extra>"))
    th.style(fig, 210)
    fig.update_yaxes(tickprefix="£", tickformat="~s", title_text="Order value")
    th.show(fig, f"lookup_{r['CustomerID']}")


# ---------- Clusters ----------
def cluster_names(df: pd.DataFrame, k: int) -> dict:
    """Name each cluster from its medians, so the label is backed by the numbers beside it."""
    names = {}
    for c, g in df.groupby(f"k{k}"):
        r, f = g["Recency"].median(), g["Frequency"].median()
        if f >= 12:
            n = "Elite repeat buyers"
        elif r <= 75 and f >= 5:
            n = "Active regulars"
        elif r <= 75:
            n = "Recent starters"
        elif f >= 3:
            n = "Cooling regulars" if r <= 200 else "Lapsed regulars"
        else:
            n = "Dormant one-timers"
        while n in names.values():
            n += " (second group)"
        names[c] = n
    return names


@st.cache_data(show_spinner=False)
def dendro(n: int = 1500) -> tuple[list, list]:
    x = full_rfm[["zR", "zF", "zM"]].sample(n, random_state=42).values
    d = dendrogram(linkage(x, "ward"), truncate_mode="lastp", p=30, no_plot=True)
    return d["icoord"], d["dcoord"]


@st.fragment
def clusters_view():
    ev = tables["clusters"]
    km = pd.DataFrame(ev["kmeans"])
    wd = pd.DataFrame(ev["ward"])
    db = ev["dbscan"]
    five = km[km["k"] == 5].iloc[0]
    four = km[km["k"] == 4].iloc[0]
    th.section("Five clusters: the top group is worth separating even though the scores do not force it",
               f"Two clusters score highest ({km.iloc[0]['silhouette']:.2f}), but that only divides active from lapsed customers. Among more useful counts, k=4 ({four['silhouette']:.3f}) and k=5 ({five['silhouette']:.3f}) score almost the same. k=5 gives the same result across random starts (adjusted Rand index {five['stability']:.2f}) and splits out a small top group, so we chose it for that business use.")
    a, b, c = st.columns(3, gap="medium")
    with a:
        th.sub("Elbow", "Within-cluster error keeps falling with no sharp bend, so it cannot settle k alone.")
        fig = go.Figure(go.Scatter(x=km["k"], y=km["wcss"], mode="lines+markers", line=dict(color=th.series(0), width=2), marker=dict(size=8, line=dict(width=2, color=th.tok("panel"))), hovertemplate="k=%{x}<br><b>WCSS %{y:,.0f}</b><extra></extra>"))
        th.style(fig, 230)
        fig.update_xaxes(dtick=1, title_text="k")
        th.show(fig, "elbow")
    with b:
        th.sub("Silhouette", "Higher means cleaner separation. K-means beats Ward at every k from 2 to 10, measured on all customers against a 3,000 sample.")
        fig = go.Figure()
        fig.add_scatter(x=km["k"], y=km["silhouette"], mode="lines+markers", name="K-means", line=dict(color=th.series(0), width=2), marker=dict(size=8, line=dict(width=2, color=th.tok("panel"))), hovertemplate="k=%{x}<br><b>%{y:.3f}</b><extra>K-means</extra>")
        fig.add_scatter(x=wd["k"], y=wd["silhouette"], mode="lines+markers", name="Ward", line=dict(color=th.series(1), width=2), marker=dict(size=8, line=dict(width=2, color=th.tok("panel"))), hovertemplate="k=%{x}<br><b>%{y:.3f}</b><extra>Ward</extra>")
        th.style(fig, 230, legend=True)
        fig.update_xaxes(dtick=1, title_text="k")
        th.show(fig, "sil")
    with c:
        th.sub("Ward dendrogram", "A sample of 1,500 customers, cut to the top 30 joins. The long upper branches show a few broad groups.")
        ic, dc = dendro()
        fig = go.Figure()
        for xs, ys in zip(ic, dc):
            fig.add_scatter(x=xs, y=ys, mode="lines", line=dict(color=th.series(0), width=1.3), hoverinfo="skip", showlegend=False)
        th.style(fig, 230)
        fig.update_xaxes(showticklabels=False, showline=False)
        fig.update_yaxes(title_text="Merge distance")
        th.show(fig, "dendro")

    st.html(f"""<div class="factline" style="margin:6px 0 4px"><div><b>{db['noise_customers']} accounts</b>DBSCAN flags as outliers</div><div><b>{db['noise_revenue_share']:.0%}</b>of revenue among them</div>
<div><b>{th.money(db['noise_median_spend'])}</b>their median spend</div><div><b>{db['clusters']}</b>dense cluster otherwise</div></div>""")
    th.note(f"DBSCAN (eps {db['eps']:.2f}) finds one dense mass of customers plus these outliers. It does not produce segments on this data, because density changes smoothly, but it does give a useful list of whale accounts to manage personally. {db['noise_high_spend']} of the {db['noise_customers']} are in the top tenth by spend.")

    th.section("Explore the clusters", "Change k to see how the groups split. Names come from each group's median recency and frequency, and the table shows the numbers behind them.")
    k = st.segmented_control("Number of clusters", [3, 4, 5, 6], default=5, key="k_pick") or 5
    names = cluster_names(full_rfm, k)
    # Clusters are fitted on all customers. The market filter only narrows who is displayed.
    d = rfm.assign(Cluster=rfm[f"k{k}"].map(names), Idx=rfm[f"k{k}"])
    prof = d.groupby(["Idx", "Cluster"]).agg(Customers=("CustomerID", "size"), R=("Recency", "median"), F=("Frequency", "median"), M=("Monetary", "median"), Total=("Monetary", "sum")).reset_index()
    prof["CustShare"] = prof["Customers"] / prof["Customers"].sum()
    prof["RevShare"] = prof["Total"] / prof["Total"].sum()
    prof = prof.sort_values("Total", ascending=False)

    left, right = st.columns([1.15, 1], gap="large")
    with left:
        sample = d.sample(min(2500, len(d)), random_state=42)
        fig = go.Figure()
        for _, p in prof.sort_values("Idx").iterrows():
            g = sample[sample["Cluster"] == p["Cluster"]]
            fig.add_scatter3d(x=g["logR"], y=g["logF"], z=g["logM"], mode="markers", name=p["Cluster"], marker=dict(size=2.8, color=th.series(int(p["Idx"])), opacity=.85),
                              customdata=np.stack([g["CustomerID"], g["Recency"], g["Frequency"], g["Monetary"]], axis=1),
                              hovertemplate="Customer %{customdata[0]}<br>%{customdata[1]} days ago<br>%{customdata[2]} orders<br>£%{customdata[3]:,.0f}<extra>" + p["Cluster"] + "</extra>")
        axis = dict(backgroundcolor="rgba(0,0,0,0)", gridcolor=th.tok("line"), color=th.tok("ink-3"), showspikes=False)
        th.style(fig, 470, legend=True, margin=dict(l=0, r=0, t=46, b=0),
                 scene=dict(xaxis=dict(title="log recency", **axis), yaxis=dict(title="log frequency", **axis), zaxis=dict(title="log spend", **axis), camera=dict(eye=dict(x=1.5, y=1.5, z=.9))))
        th.show(fig, f"scatter3d_{k}")
        th.note("Drag to rotate. Shows up to 2,500 customers.", small=True)
    with right:
        th.sub("Cluster profiles")
        th.table(prof, [{"key": "Cluster", "label": "Cluster", "text": True, "color": lambda r: th.series(int(r["Idx"]))}, {"key": "CustShare", "label": "Customers", "fmt": lambda v: f"{v:.0%}"},
                        {"key": "RevShare", "label": "Revenue", "fmt": lambda v: f"{v:.0%}"}, {"key": "R", "label": "Days ago", "fmt": th.num}, {"key": "F", "label": "Orders", "fmt": th.num}, {"key": "M", "label": "Spend", "fmt": th.money}], 250)
        ct = pd.crosstab(d["Cluster"], d["Segment"], normalize="index").reindex(index=prof["Cluster"], columns=[s for s in th.SEGMENTS if s in set(d["Segment"])]).fillna(0)
        th.sub("Clusters against RFM segments", "Each row adds up to 100%. Where the methods agree the row is dominated by one or two cells, so the segments are not just an artefact of one method.")
        fig = go.Figure(go.Heatmap(z=ct.values, x=ct.columns, y=ct.index, colorscale=th.seq(), showscale=False, xgap=2, ygap=2, hovertemplate="%{y}<br>%{x}<br><b>%{z:.0%}</b><extra></extra>"))
        th.style(fig, 240)
        fig.update_yaxes(autorange="reversed", showgrid=False, type="category")
        fig.update_xaxes(tickangle=-30, type="category")
        th.show(fig, f"crosstab_{k}")


# ---------- Behaviour ----------
def behaviour_view():
    c = tables["cohorts"]
    later = c[(c["Offset"] == 1) & (c["Cohort"] > "2009-12-01")]["Retention"].median()
    th.section(f"Typically {later:.0%} of new customers order again the next month",
               "Each row is the group who first ordered in that month, and each cell is the share ordering again that many months later. Dec 2009 looks unusually strong because it includes everyone who bought before the data began. Repeat rates dip around 9 to 10 months after a first order and recover about a year out, which fits buyers returning for the same season.")
    grid = c[c["Offset"] > 0].pivot(index="Cohort", columns="Offset", values="Retention")
    fig = go.Figure(go.Heatmap(z=grid.values, x=grid.columns, y=[f"{d:%b %Y}" for d in grid.index], colorscale=th.seq(), showscale=False, xgap=1.5, ygap=1.5, zmax=grid.iloc[1:].max().max(), zmin=0,
                               hovertemplate="First ordered %{y}<br>%{x} months later<br><b>%{z:.0%} ordered</b><extra></extra>", hoverongaps=False))
    th.style(fig, 470)
    fig.update_yaxes(autorange="reversed", showgrid=False, type="category")
    fig.update_xaxes(title_text="Months after first order", dtick=2)
    th.show(fig, "cohort")

    a, b = st.columns(2, gap="large")
    with a:
        m = rfm.sort_values("Monetary", ascending=False)["Monetary"].values
        y = np.cumsum(m) / m.sum()
        x = np.arange(1, len(m) + 1) / len(m)
        i1 = max(0, len(m) // 100 - 1)
        th.sub(f"The top 1% of customers bring {y[i1]:.0%} of revenue", "Registered customers only, ordered by spend. A single account leaving would show in the monthly figures.")
        step = max(1, len(m) // 400)
        fig = go.Figure(go.Scatter(x=x[::step], y=y[::step], mode="lines", line=dict(color=th.series(0), width=2), hovertemplate="Top %{x:.1%} of customers<br><b>%{y:.1%} of revenue</b><extra></extra>"))
        th.style(fig, 280)
        fig.update_xaxes(tickformat=".0%", title_text="Share of customers, largest first")
        fig.update_yaxes(tickformat=".0%", range=[0, 1.02])
        th.show(fig, "lorenz")
    with b:
        gaps = tables["gaps"]
        gaps = gaps[gaps["CustomerID"].isin(rfm["CustomerID"])]["Gap"]
        th.sub(f"The median gap between orders is {gaps.median():.0f} days", f"{(gaps < 1).mean():.0%} of reorders come within a day, and {(gaps <= 60).mean():.0%} within two months. This is a sensible trigger for reminders.")
        v = gaps.clip(upper=240)
        fig = go.Figure(go.Histogram(x=v, xbins=dict(start=0, end=240, size=7), marker=dict(color=th.series(0), line=dict(color=th.tok("panel"), width=1)), hovertemplate="%{x} days<br><b>%{y:,} reorders</b><extra></extra>"))
        th.style(fig, 280)
        fig.update_xaxes(title_text="Days between orders (240 includes everything longer)")
        fig.update_yaxes(title_text="Reorders")
        th.show(fig, "gaps")

    th.section("Who matters most, and who returns most")
    r = tables["returns"]
    by_ret = r[r["CustomerID"].astype(str).ne("")].groupby("CustomerID", observed=True).agg(Returned=("ReturnValue", "sum"), Lines=("ReturnQty", "size")).reset_index()
    by_ret["CustomerID"] = by_ret["CustomerID"].astype(str)
    by_ret = by_ret.merge(rfm[["CustomerID", "Monetary", "Country", "Segment"]], on="CustomerID")
    top_ret = by_ret.sort_values("Returned", ascending=False).head(12)
    c1, c2, c3 = st.columns(3, gap="medium")
    base_cols = [{"key": "CustomerID", "label": "Customer", "text": True}, {"key": "Country", "label": "Country", "text": True}]
    with c1:
        th.sub("Most revenue")
        th.table(rfm.sort_values("Monetary", ascending=False).head(12), base_cols + [{"key": "Monetary", "label": "Spend", "fmt": th.money, "bar": True}], 390)
    with c2:
        th.sub("Most orders")
        th.table(rfm.sort_values("Frequency", ascending=False).head(12), base_cols + [{"key": "Frequency", "label": "Orders", "fmt": th.num, "bar": True}], 390)
    with c3:
        th.sub("Most returned value")
        th.table(top_ret, base_cols + [{"key": "Returned", "label": "Returned", "fmt": th.money, "bar": True, "red": True}], 390)


if view == "RFM segments":
    segments_view()
elif view == "Clusters":
    clusters_view()
else:
    behaviour_view()
