"""Entry point. Run from the project folder:  streamlit run app/app.py

Filters and the colour mode live here so they survive switching pages.
"""
import pandas as pd
import streamlit as st

import data
import theme as th

st.set_page_config(page_title="Online Retail Analytics", page_icon="✉️", layout="wide", initial_sidebar_state="collapsed")

PAGES = {
    "overview": st.Page("views/overview.py", title="Overview", url_path="overview", default=True),
    "products": st.Page("views/products.py", title="Products", url_path="products"),
    "customers": st.Page("views/customers.py", title="Customers and RFM", url_path="customers"),
}
page = st.navigation(list(PAGES.values()), position="hidden")
active = next(k for k, v in PAGES.items() if v.url_path == page.url_path or v.title == page.title)

if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = False
th.inject(active)

brand, n1, n2, n3, switch = st.columns([3.2, 1, 1, 1.5, 0.8], vertical_alignment="center")
brand.html('<div class="brand">Online Retail Analytics<small>UK giftware wholesaler, Dec 2009 to Dec 2011</small></div>')
for col, key, label in ((n1, "overview", "Overview"), (n2, "products", "Products"), (n3, "customers", "Customers and RFM")):
    with col.container(key=f"nav_{key}"):
        st.page_link(PAGES[key], label=label, use_container_width=True)
switch.toggle("Dark mode", key="dark_mode", label_visibility="collapsed", help="Switch between light and dark")

all_months = [pd.Timestamp(m) for m in data.months()]
labels = {m.strftime("%b %Y"): m for m in all_months}
with st.container(key="filters"):
    left, right = st.columns([2.4, 1.2], gap="large", vertical_alignment="bottom")
    span = left.select_slider("Period", options=list(labels), value=(all_months[0].strftime("%b %Y"), all_months[-1].strftime("%b %Y")), key="period")
    market = right.segmented_control("Market", data.MARKETS, default="All markets", key="market") or "All markets"

st.session_state["scope"] = (labels[span[0]], labels[span[1]], market)
page.run()
