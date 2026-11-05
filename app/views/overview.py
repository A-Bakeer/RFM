"""Page 1: KPIs and the global picture."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import data
import theme as th

start, end, market = st.session_state["scope"]
k = data.summary(start, end, market)
tables = data.base()
LAST_MONTH = pd.Timestamp(data.months()[-1])

if k["orders"] == 0:
    th.empty("No orders in this period and market. Widen the period or choose All markets.")
    st.stop()


def delta(now: float, was: float | None) -> str:
    if not was:
        return ""
    change = now / was - 1
    arrow = "up" if change >= 0 else "down"
    return f'<span class="{arrow}">{abs(change):.1%}</span> vs prior period'


span = (end.year - start.year) * 12 + end.month - start.month + 1
prior_end = start - pd.DateOffset(months=1)
prior_start = start - pd.DateOffset(months=span)
prior = data.summary(prior_start, prior_end, market) if prior_start >= pd.Timestamp(data.months()[0]) else {}

scope_name = {"UK": "UK orders", "International": "Orders outside the UK"}.get(market, "All markets")
th.envelope(
    "A trade business that peaks before Christmas",
    f"{scope_name}, {start:%b %Y} to {end:%b %Y}. Revenue is net of returns and excludes postage and bookkeeping entries. An order is a whole invoice, not a line on it.",
    [
        th.stamp(th.money(k["net"]), "Net revenue", delta(k["net"], prior.get("net")), lead=True),
        th.stamp(th.num(k["orders"]), "Orders", delta(k["orders"], prior.get("orders"))),
        th.stamp(th.num(k["customers"]), "Registered customers", delta(k["customers"], prior.get("customers"))),
        th.stamp(th.money(k["aov"]), "Average order value", delta(k["aov"], prior.get("aov"))),
        th.stamp(th.num(k["upo"]), "Units per order", delta(k["upo"], prior.get("upo"))),
        th.stamp(th.pct(k["return_rate"]), "Returned, share of gross", delta(k["return_rate"], prior.get("return_rate"))),
    ],
)


@st.fragment
def trend_section():
    m = data.monthly(start, end, market)
    full = m[m["YearMonth"] < LAST_MONTH]
    q4 = full.loc[full["YearMonth"].dt.month >= 10, "Net"].sum() / max(full["Net"].sum(), 1)
    peak = m.loc[m["Net"].idxmax()]
    th.section(f"{peak['YearMonth']:%B %Y} is the peak, and Oct to Dec makes {q4:.0%} of revenue in this period",
               "Autumn orders are stock bought ahead of Christmas. They peak in November, not December. December 2011 only runs to the 9th, so its low bar is missing data rather than a fall in sales.")
    view = st.segmented_control("Trend view", ["Timeline", "Year against year"], default="Timeline", key="trend_view",
                                label_visibility="collapsed") or "Timeline"
    fig = go.Figure()
    if view == "Timeline":
        q4_mask = m["YearMonth"].dt.month >= 10
        colors = [th.tok("wash") if d == LAST_MONTH else (th.series(0) if q else th.tok("line-strong"))
                  for d, q in zip(m["YearMonth"], q4_mask)]
        labels = ["partial month" if d == LAST_MONTH else ("Q4" if q else "") for d, q in zip(m["YearMonth"], q4_mask)]
        fig.add_bar(x=m["YearMonth"], y=m["Net"], marker=dict(color=colors, cornerradius=4), customdata=labels,
                    hovertemplate="%{x|%B %Y}<br><b>£%{y:,.0f}</b> %{customdata}<extra></extra>")
        th.style(fig, 330)
        fig.update_xaxes(dtick="M2", tickformat="%b<br>%Y")
    else:
        m2 = m.assign(Y=m["YearMonth"].dt.year, Mo=m["YearMonth"].dt.month)
        for i, y in enumerate(sorted(m2["Y"].unique())):
            row = m2[m2["Y"] == y]
            if len(row) < 2:
                continue
            fig.add_scatter(x=row["Mo"], y=row["Net"], mode="lines+markers", name=str(y),
                            line=dict(width=2, color=th.series(i)), marker=dict(size=8, line=dict(width=2, color=th.tok("panel"))),
                            hovertemplate=f"{y}: " + "<b>£%{y:,.0f}</b><extra></extra>")
        th.style(fig, 330, legend=True, hovermode="x unified")
        fig.update_xaxes(tickvals=list(range(1, 13)), ticktext=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    fig.update_yaxes(tickprefix="£", tickformat="~s")
    th.show(fig, f"trend_{view}")


METRICS = {
    "Revenue": ("Net", "£%{z:,.0f}", True), "Orders": ("Orders", "%{z:,.0f} orders", True),
    "Customers": ("Customers", "%{z:,.0f} customers", True), "Order value": ("AOV", "£%{z:,.0f} per order", False),
    "Return rate": ("ReturnRate", "%{customdata:.1%} of gross returned", False),
}


@st.fragment
def map_section():
    c = data.by_country(start, end, market)
    if c.empty:
        return
    uk = c.loc[c["Country"] == "United Kingdom", "Net"].sum() / max(c["Net"].sum(), 1)
    abroad = c[c["Country"] != "United Kingdom"]
    title = (f"The UK makes {uk:.0%} of revenue, so the map leaves it out by default" if 0 < uk < 1
             else f"{c.iloc[0]['Country']} leads with {th.money(c.iloc[0]['Net'])}")
    th.section(title, "Click a country to see its numbers and best sellers, and click it again to clear. Countries with fewer than 30 orders are hidden from the order value and return rate views, because a few invoices can swing those figures.")

    a, b, cc = st.columns([2.4, 1, 1], vertical_alignment="bottom")
    metric = a.segmented_control("Show", list(METRICS), default="Revenue", key="map_metric") or "Revenue"
    region = b.segmented_control("Area", ["World", "Europe"], default="World", key="map_region") or "World"
    with_uk = cc.toggle("Include the UK", value=False, key="map_uk")

    col, log_scale = METRICS[metric][0], METRICS[metric][2]
    d = c.dropna(subset=["ISO"]).copy()
    if not with_uk and market != "UK":
        d = d[d["Country"] != "United Kingdom"]
    if not log_scale:
        d = d[d["Orders"] >= 30]
    d = d[d[col] > 0]
    if d.empty:
        th.empty("No country has enough orders for this view. Pick another measure or widen the period.")
        return

    z = np.log10(d[col]) if log_scale else d[col]
    fig = go.Figure(go.Choropleth(
        locations=d["ISO"], z=z, text=d["Country"], customdata=d[col],
        colorscale=th.seq(), marker=dict(line=dict(color=th.tok("panel"), width=0.6)),
        showscale=False,
        hovertemplate="<b>%{text}</b><br>" + METRICS[metric][1].replace("%{z", "%{customdata") + "<extra></extra>",
        selected=dict(marker=dict(opacity=1)), unselected=dict(marker=dict(opacity=0.85)),
    ))
    geo = dict(bgcolor="rgba(0,0,0,0)", showframe=False, showcoastlines=False, showland=True, landcolor=th.tok("line"),
               showcountries=True, countrycolor=th.tok("panel"), countrywidth=0.5, showocean=False, showlakes=False,
               projection_type="natural earth", resolution=50, lataxis_range=[-58, 85])
    if region == "Europe":
        geo.update(lataxis_range=[34, 71], lonaxis_range=[-14, 40], projection_type="mercator")
    th.style(fig, 430, geo=geo, margin=dict(l=0, r=0, t=0, b=0), clickmode="event+select")

    left, right = st.columns([2.2, 1])
    with left:
        ev = th.show(fig, f"map_{metric}_{region}_{with_uk}", on_select="rerun", selection_mode="points")
    picked = None
    if ev and ev.selection and ev.selection.points:
        picked = ev.selection.points[0].get("text")
        if picked is None:
            iso = ev.selection.points[0].get("location")
            picked = next((n for n, i in data.ISO3.items() if i == iso), None)
    with right:
        if picked and picked in set(c["Country"]):
            r = c[c["Country"] == picked].iloc[0]
            th.sub(picked, "Top sellers for the selected period")
            st.html(f'<div class="factline"><div><b>{th.money(r["Net"])}</b>net revenue</div><div><b>{th.num(r["Orders"])}</b>orders</div>'
                    f'<div><b>{th.num(r["Customers"])}</b>customers</div><div><b>{th.pct(r["ReturnRate"])}</b>returned</div></div>')
            th.table(data.country_products(picked, start, end),
                     [{"key": "Description", "label": "Product", "text": True, "fmt": lambda v: str(v).title()}, {"key": "Net", "label": "Net", "fmt": th.money}], 250)
        else:
            th.sub("Largest markets abroad" if market != "UK" else "United Kingdom")
            top = (abroad if market != "UK" else c).head(9)
            th.table(top, [{"key": "Country", "label": "Country", "text": True}, {"key": "Net", "label": "Net revenue", "fmt": th.money, "bar": True},
                           {"key": "Customers", "label": "Customers", "fmt": th.num}], 360)
    not_mapped = c[c["ISO"].isna()]
    if len(not_mapped):
        th.note(f"Not on the map because they have no single shape: {', '.join(not_mapped['Country'])}, with {th.money(not_mapped['Net'].sum())} between them.", small=True)


def rhythm_section():
    r = data.rhythm(start, end, market)
    if r.empty:
        return
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    hours = list(range(int(r["Hour"].min()), int(r["Hour"].max()) + 1))
    grid = r.pivot(index="Day", columns="Hour", values="Orders").reindex(index=range(7), columns=hours).fillna(0)
    peak = r.loc[r["Orders"].idxmax()]
    by_hour = r.groupby("Hour")["Orders"].sum()
    core = by_hour[(by_hour.index >= 9) & (by_hour.index < 17)].sum() / by_hour.sum()
    th.section(f"{days[int(peak['Day'])]} at {int(peak['Hour'])}:00 is the busiest slot, and Saturdays are almost empty",
               f"{core:.0%} of orders arrive between 9:00 and 17:00, which fits trade buyers ordering during working hours. Nights and Saturdays are the quiet slots for maintenance, and weekday mornings look like the natural time to test ads. Sunday is a normal trading day here.")
    fig = go.Figure(go.Heatmap(z=grid.values, x=[f"{h}:00" for h in hours], y=days, colorscale=th.seq(), xgap=2, ygap=2,
                               showscale=False, hovertemplate="%{y} %{x}<br><b>%{z:,.0f} orders</b><extra></extra>"))
    th.style(fig, 280)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    th.show(fig, "rhythm")


def basket_section():
    b = data.baskets(start, end, market)
    if b.empty:
        return
    bulk_orders = b["IsBulk"].mean()
    bulk_rev = b.loc[b["IsBulk"], "Value"].sum() / b["Value"].sum()
    th.section(f"Bulk orders are {bulk_orders:.0%} of orders and {bulk_rev:.0%} of revenue",
               f"The median order is {b['Units'].median():,.0f} units worth £{b['Value'].median():,.0f}, which is much more than a household buys. Bulk means 520 units or more, the top tenth of all orders.")
    left, right = st.columns([1.6, 1])
    with left:
        edges = np.logspace(0, np.log10(max(b["Value"].max(), 10)), 36)
        fig = go.Figure()
        for i, (flag, name) in enumerate(((False, "Regular orders"), (True, "Bulk orders"))):
            counts, _ = np.histogram(b.loc[b["IsBulk"] == flag, "Value"], bins=edges)
            fig.add_bar(x=np.sqrt(edges[:-1] * edges[1:]), y=counts, name=name, marker=dict(color=th.series(i), line=dict(color=th.tok("panel"), width=1)),
                        width=edges[1:] - edges[:-1], hovertemplate="About £%{x:,.0f}<br><b>%{y:,} orders</b><extra>" + name + "</extra>")
        th.style(fig, 300, legend=True, barmode="stack", xaxis_type="log")
        fig.update_xaxes(tickprefix="£", title_text="Order value (log scale)", tickvals=[1, 10, 100, 1000, 10000])
        fig.update_yaxes(title_text="Orders")
        th.show(fig, "basket")
    with right:
        rows = pd.DataFrame({
            "Measure": ["Median order value", "Median units", "Median distinct items", "Share of orders", "Share of revenue"],
            "Regular": [th.money(b.loc[~b["IsBulk"], "Value"].median()), th.num(b.loc[~b["IsBulk"], "Units"].median()), th.num(b.loc[~b["IsBulk"], "Lines"].median()), th.pct(1 - bulk_orders, 0), th.pct(1 - bulk_rev, 0)],
            "Bulk": [th.money(b.loc[b["IsBulk"], "Value"].median()), th.num(b.loc[b["IsBulk"], "Units"].median()), th.num(b.loc[b["IsBulk"], "Lines"].median()), th.pct(bulk_orders, 0), th.pct(bulk_rev, 0)],
        })
        th.sub("Regular and bulk side by side")
        th.table(rows, [{"key": "Measure", "label": "", "text": True}, {"key": "Regular", "label": "Regular"}, {"key": "Bulk", "label": "Bulk"}], 260)


def growth_section():
    left, right = st.columns(2)
    with left:
        nr = data.new_vs_returning(start, end, market)
        if len(nr) and {"New", "Returning"} <= set(nr.columns):
            rest = nr[nr["YearMonth"] >= "2010-06-01"]
            share = rest["Returning"].sum() / max((rest["New"] + rest["Returning"]).sum(), 1) if len(rest) else 0
            th.sub(f"Returning customers bring {share:.0%} of registered revenue" if len(rest) else "New and returning revenue",
                   "The data starts in Dec 2009, so everyone looks new at first. The share is measured from June 2010, once the two groups have separated.")
            fig = go.Figure()
            for i, kind in enumerate(("Returning", "New")):
                fig.add_bar(x=nr["YearMonth"], y=nr[kind], name=kind, marker=dict(color=th.series(i), line=dict(color=th.tok("panel"), width=1.5)),
                            hovertemplate="%{x|%b %Y}<br><b>£%{y:,.0f}</b><extra>" + kind + "</extra>")
            th.style(fig, 280, legend=True, barmode="stack")
            fig.update_yaxes(tickprefix="£", tickformat="~s")
            th.show(fig, "newret")
    with right:
        m = data.monthly(start, end, market)
        th.sub(f"Returns run at {k['return_rate']:.1%} of gross sales",
               "A return is dated when it comes back. Single large cancellations cause the spikes, so check which account before reading a spike as a trend.")
        fig = go.Figure()
        fig.add_scatter(x=m["YearMonth"], y=m["ReturnRate"], mode="lines+markers", line=dict(color=th.tok("red"), width=2),
                        marker=dict(size=8, color=th.tok("red"), line=dict(width=2, color=th.tok("panel"))), showlegend=False,
                        hovertemplate="%{x|%b %Y}<br><b>%{y:.1%}</b> of gross returned<extra></extra>")
        th.style(fig, 280)
        fig.update_yaxes(tickformat=".0%", rangemode="tozero")
        th.show(fig, "retrate")


def findings_section():
    th.section("What the data says", "Computed across all two years and all markets, so these do not change with the filters. They describe patterns in past orders and do not establish causes.")
    cards = "".join(f'<div class="finding"><b>{t}</b><span>{e}</span><em>{a}</em></div>' for t, e, a in data.findings())
    st.html(f'<div class="findings">{cards}</div>')


def audit_section():
    a = tables["audit"]
    st.html('<div style="height:26px"></div>')
    with st.expander("How the data was cleaned, and what each decision changed"):
        rows = pd.DataFrame([
            ("Rows in the two Excel sheets", a["raw_rows_sheet1"] + a["raw_rows_sheet2"], "The starting point."),
            ("Overlap between sheets removed", -a["overlap_rows_removed"], "Both sheets contain 1 to 9 Dec 2010. Left in, that week would be counted twice."),
            ("Exact duplicate rows removed", -a["exact_duplicate_rows"], f"Worth {th.money(a['exact_duplicate_value'])}. Some could be real repeat scans, so this slightly understates revenue if so."),
            ("Cancellations set aside", -a["cancel_rows"], f"Invoices starting with C. Kept as a separate table, and {a['return_rows_fully_matched']:,} of the {a['return_rows_goods']:,} lines for goods match fully to an earlier purchase and are netted out of that sale."),
            ("Zero or negative prices set aside", -a["zero_or_negative_price_rows"], "Giveaways, damage write-offs and stock adjustments. These are not sales."),
            ("Bad debt adjustments set aside", -a["bad_debt_rows"], "Invoices starting with A. Accounting entries."),
            ("Postage, fees and manual entries set aside", -a["service_rows"], f"Worth {th.money(a['service_value'])}. Excluded so that postage does not rank as a top product."),
            ("Other negative or non-sale rows", -(a["rows_after_overlap"] - a["exact_duplicate_rows"] - a["cancel_rows"] - a["zero_or_negative_price_rows"] - a["bad_debt_rows"] - a["service_rows"] - a["sale_rows"]), "Rows caught by more than one rule above, so this number balances the total."),
            ("Clean sale lines", a["sale_rows"], f"Gross {th.money(a['gross_revenue'])}, net {th.money(a['net_revenue'])}."),
        ], columns=["Step", "Rows", "Why it matters"])
        th.table(rows, [{"key": "Step", "label": "Step", "text": True}, {"key": "Rows", "label": "Rows", "fmt": lambda v: f"{v:+,}" if v < 0 else f"{v:,}"},
                        {"key": "Why it matters", "label": "Why it matters", "text": True}], 420)
        th.note(f"{a['guest_rows']:,} rows have no customer ID. Grouped by invoice, {a['guest_invoices_all_lines']:,} invoices have no ID on any line and {a['guest_invoices_mixed']} are mixed, so they are whole guest checkouts. They stay in revenue and product analysis and are left out of RFM. Returns with no earlier purchase in the data are worth {th.money(a['return_value_unmatched'])}; they count in the return rate but are not subtracted from net revenue, because there is no sale to tie them to.", small=True)


trend_section()
map_section()
rhythm_section()
basket_section()
growth_section()
findings_section()
audit_section()
