"""Page 2: product search and drill-down, then the shape of the range."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import data
import theme as th

start, end, market = st.session_state["scope"]
tables = data.base()
products = tables["products"]
pt = data.product_table(start, end, market)

if pt.empty:
    th.empty("No product sales in this period and market. Widen the period or choose All markets.")
    st.stop()

share = pt["Net"].cumsum() / pt["Net"].sum()
n80 = int((share <= 0.8).sum()) + 1
dead = products[products["IsDead"]]
th.envelope(
    f"{n80 / len(pt):.0%} of products make 80% of revenue",
    f"{n80:,} of the {len(pt):,} products sold from {start:%b %Y} to {end:%b %Y}. Search below for any product to see where it sells and what it sells with.",
    [
        th.stamp(th.num(len(pt)), "Products sold", lead=True),
        th.stamp(pt.iloc[0]["Description"].title(), "Top seller by revenue", th.money(pt.iloc[0]["Net"])),
        th.stamp(th.money(pt["Net"].median()), "Median product revenue"),
        th.stamp(f"£{pt['Price'].median():.2f}", "Median unit price"),
        th.stamp(th.num(n80), "Products in the 80% group"),
        th.stamp(th.num(len(dead)), "Dead stock lines", "no sales in the final 90 days"),
    ],
)


@st.fragment
def search_section():
    th.section("Find a product", "Search by any words in the name, or by stock code. For example: red bag, or 85123A.")
    query = st.text_input("Search products", placeholder="Search products, then press Enter", key="product_query", label_visibility="collapsed")
    hits = data.search_products(query)
    left, right = st.columns([1, 2.3], gap="large")

    with left:
        if hits.empty:
            th.empty(f"Nothing matches \"{query}\". Try one word, or a stock code.")
            return
        th.note(f"{len(hits):,} matches, largest first" if query.strip() else "Largest products by revenue", small=True)
        codes = hits["StockCode"].astype(str).tolist()
        if st.session_state.get("product") not in codes:
            st.session_state["product"] = codes[0]
        for _, h in hits.head(8).iterrows():
            code = str(h["StockCode"])
            prefix = "picked" if code == st.session_state["product"] else "result"
            if st.button(f"{h['Description'].title()}  ({th.money(h['Revenue'])})", key=f"{prefix}_{code}", use_container_width=True):
                st.session_state["product"] = code
                st.rerun(scope="fragment")
    with right:
        product_panel(st.session_state["product"])


def product_panel(code: str):
    row = products[products["StockCode"].astype(str) == code].iloc[0]
    d = data.product_detail(code)
    abc = {"A": "in the group making 80% of revenue", "B": "in the next 15% of revenue", "C": "in the long tail, the last 5% of revenue"}[row["ABC"]]
    th.sub(row["Description"].title(), f"Stock code {code}, {abc}. All two years and markets.")
    st.html(f"""<div class="factline">
<div><b>{th.money(row['Revenue'])}</b>net revenue</div><div><b>{th.num(row['Units'])}</b>units</div>
<div><b>{th.num(row['Orders'])}</b>orders</div><div><b>{th.num(row['Customers'])}</b>customers</div>
<div><b>£{row['AvgPrice']:.2f}</b>median price</div><div><b>{th.pct(row['ReturnRate'])}</b>of units returned</div>
<div><b>{row['Q4Share']:.0%}</b>sold Oct to Dec</div></div>""")
    if row["IsDead"]:
        st.html(f'<p class="note">This is dead stock: it sold {row["UnitsBefore90"]:,.0f} units earlier on and none in the final 90 days. Last sold {row["LastSold"]:%d %b %Y}.</p>')

    t = d["trend"]
    fig = go.Figure()
    fig.add_bar(x=t["YearMonth"], y=t["Net"], marker=dict(color=th.series(0), cornerradius=3),
                customdata=t["Units"], hovertemplate="%{x|%b %Y}<br><b>£%{y:,.0f}</b><br>%{customdata:,.0f} units<extra></extra>")
    th.style(fig, 200)
    fig.update_yaxes(tickprefix="£", tickformat="~s", nticks=4)
    th.show(fig, f"p_trend_{code}")

    a, b = st.columns(2)
    with a:
        th.sub("Often bought with it", "Lift above 1 means they share an order more often than chance. That is association, not proof that one sells the other.")
        pairs = d["pairs"]
        if pairs.empty:
            th.empty("Too few shared orders to compare. Pairs need at least 30.")
        else:
            th.table(pairs, [{"key": "Name", "label": "Product", "text": True, "fmt": lambda s: str(s).title()},
                             {"key": "Lift", "label": "Lift", "fmt": lambda v: f"{v:.1f}"},
                             {"key": "Confidence", "label": "In orders", "fmt": lambda v: f"{v:.0%}"}], 240)
    with b:
        th.sub("Where it sells", "UK left out of the colour scale.")
        c = d["countries"].dropna(subset=["ISO"])
        c = c[c["Country"] != "United Kingdom"]
        if c.empty:
            uk = d["countries"]
            th.empty("Sold only in the UK." if len(uk) else "No country data for this product.")
        else:
            fig = go.Figure(go.Choropleth(locations=c["ISO"], z=np.log10(c["Net"].clip(lower=1)), text=c["Country"], customdata=c["Net"],
                                          colorscale=th.seq(), showscale=False, marker=dict(line=dict(color=th.tok("panel"), width=.5)),
                                          hovertemplate="<b>%{text}</b><br>£%{customdata:,.0f}<extra></extra>"))
            th.style(fig, 240, margin=dict(l=0, r=0, t=0, b=0),
                     geo=dict(bgcolor="rgba(0,0,0,0)", showframe=False, showcoastlines=False, landcolor=th.tok("line"), showland=True,
                              countrycolor=th.tok("panel"), showcountries=True, projection_type="natural earth", lataxis_range=[-50, 75]))
            th.show(fig, f"p_map_{code}")
            buyers_abroad = d["countries"].sort_values("Net", ascending=False).head(1).iloc[0]
            th.note(f"Largest market: {buyers_abroad['Country']} at {th.money(buyers_abroad['Net'])}.", small=True)


@st.fragment
def ranking_section():
    th.section("Best sellers change with how you measure them", "A cheap item can top the list for units and miss the revenue top 20 entirely, so plan stock by units and margin by revenue.")
    a, b = st.columns([1.5, 1], vertical_alignment="bottom")
    by = a.segmented_control("Rank by", ["Revenue", "Units", "Orders"], default="Revenue", key="rank_by") or "Revenue"
    n = b.select_slider("How many", [10, 15, 20, 30], value=15, key="rank_n")
    col = {"Revenue": "Net", "Units": "Units", "Orders": "Orders"}[by]
    top = pt.sort_values(col, ascending=False).head(n).iloc[::-1]
    fig = go.Figure(go.Bar(
        y=top["Description"].str.title().str.slice(0, 36), x=top[col], orientation="h", marker=dict(color=th.series(0), cornerradius=4),
        customdata=np.stack([top["Net"], top["Units"], top["Orders"]], axis=1),
        hovertemplate="<b>%{y}</b><br>£%{customdata[0]:,.0f}<br>%{customdata[1]:,.0f} units<br>%{customdata[2]:,.0f} orders<extra></extra>",
    ))
    th.style(fig, max(320, n * 25))
    fig.update_yaxes(showgrid=False, type="category")
    fig.update_xaxes(showgrid=True, gridcolor=th.tok("line"), tickformat="~s", tickprefix="£" if by == "Revenue" else "")
    th.show(fig, f"rank_{by}_{n}")


def pareto_section():
    x = np.arange(1, len(pt) + 1) / len(pt)
    y = share.values
    step = max(1, len(pt) // 400)
    th.section(f"Revenue is concentrated, close to the classic 80/20 but not exactly it",
               f"80% of revenue takes {n80 / len(pt):.0%} of products here. The top 1% of products bring {share.iloc[max(0, len(pt) // 100 - 1)]:.0%}.")
    left, right = st.columns([1.3, 1], gap="large")
    with left:
        fig = go.Figure()
        fig.add_scatter(x=x[::step], y=y[::step], mode="lines", line=dict(color=th.series(0), width=2), showlegend=False,
                        hovertemplate="Top %{x:.1%} of products<br><b>%{y:.1%} of revenue</b><extra></extra>")
        fig.add_shape(type="line", x0=0, x1=n80 / len(pt), y0=.8, y1=.8, line=dict(color=th.tok("ink-3"), dash="dot", width=1))
        fig.add_shape(type="line", x0=n80 / len(pt), x1=n80 / len(pt), y0=0, y1=.8, line=dict(color=th.tok("ink-3"), dash="dot", width=1))
        fig.add_annotation(x=n80 / len(pt), y=.8, text=f"{n80 / len(pt):.0%} of products, 80% of revenue", showarrow=False, xanchor="left", yanchor="top",
                           xshift=8, yshift=-6, font=dict(color=th.tok("ink"), size=13))
        th.style(fig, 320)
        fig.update_xaxes(tickformat=".0%", title_text="Share of products, largest first", range=[0, 1])
        fig.update_yaxes(tickformat=".0%", range=[0, 1.02])
        th.show(fig, "pareto")
    with right:
        themes = data.keyword_themes(start, end, market).head(10).iloc[::-1]
        th.sub("Revenue by word in the product name", "There is no category field, so this uses words in the name as a rough guide. A product can count under two words.")
        fig = go.Figure(go.Bar(y=themes["Theme"], x=themes["Share"], orientation="h", marker=dict(color=th.series(0), cornerradius=4),
                               customdata=np.stack([themes["Net"], themes["Products"]], axis=1),
                               hovertemplate="<b>%{y}</b><br>%{x:.1%} of revenue<br>£%{customdata[0]:,.0f} from %{customdata[1]} products<extra></extra>"))
        th.style(fig, 265)
        fig.update_xaxes(tickformat=".0%", showgrid=True, gridcolor=th.tok("line"))
        fig.update_yaxes(showgrid=False, type="category")
        th.show(fig, "themes")


def price_section():
    d = products[(products["Revenue"] > 100) & (products["AvgPrice"] > 0) & (products["Units"] > 0)]
    d = d.sort_values("Revenue", ascending=False).head(1500)
    corr = np.corrcoef(np.log10(d["AvgPrice"]), np.log10(d["Units"]))[0, 1]
    th.section("Cheaper products sell in higher volume, but revenue leaders sit across the price range",
               f"Each dot is a product, for the top 1,500 by revenue over the full two years. Log price and log units correlate at {corr:.2f}. Group A makes the first 80% of revenue. Drag to zoom and double-click to reset.")
    fig = go.Figure()
    for i, group in enumerate(("A", "B", "C")):
        g = d[d["ABC"] == group]
        fig.add_scattergl(x=g["AvgPrice"], y=g["Units"], mode="markers", name=f"Group {group}", text=g["Description"].str.title(), customdata=g["Revenue"],
                          marker=dict(size=8, color=th.series(i), opacity=.75, line=dict(width=1, color=th.tok("panel"))),
                          hovertemplate="<b>%{text}</b><br>£%{x:.2f} each<br>%{y:,.0f} units<br>£%{customdata:,.0f} revenue<extra></extra>")
    th.style(fig, 400, legend=True, dragmode="zoom")
    fig.update_xaxes(type="log", tickprefix="£", dtick=1, title_text="Median unit price (log scale)", showgrid=True, gridcolor=th.tok("line"))
    fig.update_yaxes(type="log", dtick=1, tickformat="~s", title_text="Units sold (log scale)")
    th.show(fig, "price")


@st.fragment
def watchlist_section():
    th.section("Lists to act on", "Based on all two years and all markets. The final 90 days run from 10 Sept to 9 Dec 2011.")
    tab = st.segmented_control("List", ["Dead stock", "Most returned", "Gaining", "Losing", "Most seasonal"], default="Dead stock", key="watch") or "Dead stock"
    p = products
    name = {"key": "Description", "label": "Product", "text": True, "fmt": lambda s: str(s).title()}
    code = {"key": "StockCode", "label": "Code", "text": True}
    if tab == "Dead stock":
        th.note(f"{len(p[p['IsDead']])} products in the top quarter by earlier volume sold nothing in the final 90 days. That window is the seasonal peak, so a quiet season does not explain it. Some may be deliberate range changes: check before clearing stock.")
        rows = p[p["IsDead"]].sort_values("UnitsBefore90", ascending=False).head(40)
        cols = [code, name, {"key": "UnitsBefore90", "label": "Earlier units", "fmt": th.num, "bar": True}, {"key": "UnitsLast90", "label": "Last 90 days", "fmt": th.num},
                {"key": "LastSold", "label": "Last sold", "fmt": lambda v: f"{v:%b %Y}"}]
    elif tab == "Most returned":
        th.note("Among products selling at least 500 units. A high rate can reflect one large cancelled order, so check before blaming the product.")
        rows = p[p["SoldUnits"] >= 500].sort_values("ReturnRate", ascending=False).head(40)
        cols = [code, name, {"key": "ReturnRate", "label": "Units returned", "fmt": th.pct, "bar": True, "red": True}, {"key": "ReturnedUnits", "label": "Returned", "fmt": th.num}, {"key": "SoldUnits", "label": "Sold", "fmt": th.num}]
    elif tab in ("Gaining", "Losing"):
        th.note("Final 90 days against the 90 days before, among products with at least £2k in one of the two periods. The final window includes the pre-Christmas rush, so most products gain: compare against that background.")
        m = p[(p["RevPrev90"] > 0) & ((p["RevLast90"] >= 2000) | (p["RevPrev90"] >= 2000))].copy()
        m["Change"] = m["RevLast90"] - m["RevPrev90"]
        rows = m.sort_values("Change", ascending=tab == "Losing").head(40)
        rows = rows.assign(ChangeAbs=rows["Change"].abs())
        cols = [code, name, {"key": "ChangeAbs", "label": "Change", "fmt": th.money, "bar": True, "red": tab == "Losing"}, {"key": "RevPrev90", "label": "Prior 90 days", "fmt": th.money}, {"key": "RevLast90", "label": "Final 90 days", "fmt": th.money}]
    else:
        th.note("Share of each product's revenue that falls in Oct to Dec, among products with at least £5k revenue. These need ordering ahead and clearing by January.")
        rows = p[p["Revenue"] >= 5000].sort_values("Q4Share", ascending=False).head(40)
        cols = [code, name, {"key": "Q4Share", "label": "Sold Oct to Dec", "fmt": lambda v: f"{v:.0%}", "bar": True}, {"key": "Revenue", "label": "Revenue", "fmt": th.money}]
    with st.container(key="panel_watch"):
        th.table(rows, cols, 420)
    st.download_button("Download this list as CSV", rows.drop(columns=["ChangeAbs"], errors="ignore").to_csv(index=False).encode(), f"{tab.lower().replace(' ', '_')}.csv", "text/csv", key="dl_watch")


search_section()
ranking_section()
pareto_section()
price_section()
watchlist_section()
