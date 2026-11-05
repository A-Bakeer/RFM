"""Design tokens, CSS injection and shared Plotly styling for both colour modes."""
from __future__ import annotations

import html
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

TOKENS = {
    "light": {
        "page": "#E9EEF6", "panel": "#FBFCFE", "stamp": "#E4EBF7", "field": "#F1F4FA", "wash": "#DFE7F5",
        "ink": "#12203F", "ink-2": "#44526F", "ink-3": "#5F6C88", "line": "#DDE3EE", "line-strong": "#BAC5D8",
        "blue": "#1E4FC2", "air": "#1E4FC2", "red": "#D2312D", "stamp-rule": "#AEBCD6",
        "switch-bg": "#BFD6F6", "knob": "#F5B82E", "knob-left": "4px",
        "knob-shadow": "0 0 0 3px rgba(245,184,46,.3)", "stars": "0",
    },
    "dark": {
        "page": "#0F1933", "panel": "#17244A", "stamp": "#20305C", "field": "#1E2C54", "wash": "#26396B",
        "ink": "#EDF1FA", "ink-2": "#BAC6DF", "ink-3": "#98A6C4", "line": "#28386A", "line-strong": "#40527F",
        "blue": "#7EA2F5", "air": "#2F5FD0", "red": "#F0605A", "stamp-rule": "#46598A",
        "switch-bg": "#1B2B5B", "knob": "#E6EAF4", "knob-left": "36px",
        "knob-shadow": "inset -8px -3px 0 0 #9BA8C6", "stars": "1",
    },
}

# Validated categorical order: colours follow the entity, never its rank.
CATEGORICAL = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
}
SEQUENTIAL = {
    "light": [[0, "#DCE6F7"], [0.35, "#8FB0EA"], [0.7, "#2F66CF"], [1, "#0F2E75"]],
    "dark": [[0, "#24345E"], [0.35, "#3F62B3"], [0.7, "#7199F0"], [1, "#D3E0FC"]],
}
SEGMENTS = ["Champions", "Loyal customers", "Cooling regulars", "New inflow",
            "Promising", "At-risk high value", "Hibernating", "Lost one-timers"]


def mode() -> str:
    return "dark" if st.session_state.get("dark_mode") else "light"


def tok(name: str) -> str:
    return TOKENS[mode()][name]


def series(i: int) -> str:
    palette = CATEGORICAL[mode()]
    return palette[i % len(palette)]


def segment_color(name: str) -> str:
    return series(SEGMENTS.index(name))


def seq() -> list:
    return SEQUENTIAL[mode()]


def inject(active_page: str | None = None) -> None:
    variables = ";".join(f"--{k}:{v}" for k, v in TOKENS[mode()].items())
    css = (Path(__file__).parent / "assets" / "style.css").read_text(encoding="utf-8")
    active = ""
    if active_page:
        active = (f".st-key-nav_{active_page} a{{border-color:var(--ink)!important;}}"
                  f".st-key-nav_{active_page} a span,.st-key-nav_{active_page} a p{{color:var(--ink)!important;}}")
    # @import only works at the very start of a stylesheet.
    fonts, _, rules = css.partition(chr(10))
    st.html(f"<style>{fonts}:root{{{variables};color-scheme:{mode()}}}{rules}{active}</style>")


def style(fig: go.Figure, height: int = 320, legend: bool = False, **layout) -> go.Figure:
    """Recessive grid, one font, tooltips that follow the theme."""
    grid = tok("line")
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Hanken Grotesk, system-ui, sans-serif", color=tok("ink-2"), size=13),
        margin=dict(l=8, r=12, t=30 if legend else 10, b=8),
        hoverlabel=dict(bgcolor=tok("panel"), bordercolor=tok("line-strong"),
                        font=dict(color=tok("ink"), family="Hanken Grotesk, sans-serif", size=13)),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, title_text="", font=dict(color=tok("ink-2"))),
        bargap=0.22, hovermode="closest", dragmode=False,
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=tok("line-strong"), ticks="", tickfont_color=tok("ink-3"),
                     title_font_color=tok("ink-3"), automargin=True)
    fig.update_yaxes(gridcolor=grid, zeroline=False, showline=False, ticks="", tickfont_color=tok("ink-3"),
                     title_font_color=tok("ink-3"), automargin=True)
    if layout:
        fig.update_layout(**layout)
    return fig


def show(fig: go.Figure, key: str, **kwargs):
    return st.plotly_chart(fig, use_container_width=True, theme=None, key=f"{key}_{mode()}",
                           config={"displayModeBar": False, "scrollZoom": False}, **kwargs)


# ---------- Formatting ----------
def money(v: float) -> str:
    sign = "-" if v < 0 else ""
    v = abs(v)
    if v >= 1e6:
        return f"{sign}£{v / 1e6:.2f}m"
    if v >= 1e4:
        return f"{sign}£{v / 1e3:.0f}k"
    if v >= 1e3:
        return f"{sign}£{v / 1e3:.1f}k"
    return f"{sign}£{v:,.0f}" if v >= 100 else f"{sign}£{v:,.2f}"


def num(v: float) -> str:
    return f"{v:,.0f}"


def pct(v: float, digits: int = 1) -> str:
    return f"{v * 100:.{digits}f}%"


# ---------- HTML pieces ----------
POSTMARK = """<svg class="postmark" viewBox="0 0 92 92" fill="none" stroke="#fff" stroke-width="2.5" aria-hidden="true">
<circle cx="46" cy="46" r="40"/><circle cx="46" cy="46" r="27"/>
<path d="M2 62c11-8 18 8 30 0s18 8 30 0 18 8 30 0M2 72c11-8 18 8 30 0s18 8 30 0 18 8 30 0M2 82c11-8 18 8 30 0s18 8 30 0 18 8 30 0"/></svg>"""


def stamp(value: str, label: str, detail: str = "", lead: bool = False) -> str:
    extra = POSTMARK if lead else ""
    cls = "stamp lead" if lead else "stamp"
    d = f'<span class="d">{detail}</span>' if detail else ""
    if len(value) > 12:
        cls += " wordy"
    return f'<div class="{cls}">{extra}<span class="v">{value}</span><span class="l">{html.escape(label)}</span>{d}</div>'


def envelope(title: str, text: str, stamps: list[str]) -> None:
    st.html(f"""<section class="envelope"><div class="envelope-in">
<div><h1>{html.escape(title)}</h1><p>{html.escape(text)}</p></div>
<div class="stamps">{''.join(stamps)}</div></div></section>""")


def section(title: str, text: str = "") -> None:
    para = f"<p>{html.escape(text)}</p>" if text else ""
    st.html(f'<div class="sec"><h2>{html.escape(title)}</h2>{para}</div>')


def sub(title: str, text: str = "") -> None:
    para = f'<p class="note small">{html.escape(text)}</p>' if text else ""
    st.html(f'<h3 class="t">{html.escape(title)}</h3>{para}')


def note(text: str, small: bool = False) -> None:
    st.html(f'<p class="note{" small" if small else ""}">{html.escape(text)}</p>')


def empty(text: str) -> None:
    st.html(f'<div class="empty">{html.escape(text)}</div>')


def table(rows, columns: list[dict], height: int = 400) -> None:
    """columns: {key, label, fmt?, text?, bar?, red?, color?}. `bar` scales to the column's maximum."""
    head = "".join(f'<th class="{"txt" if c.get("text") else ""}">{html.escape(c["label"])}</th>' for c in columns)
    maxima = {c["key"]: max(float(rows[c["key"]].max()), 1e-9) for c in columns if c.get("bar") and len(rows)}
    body = []
    for _, r in rows.iterrows():
        cells = []
        for c in columns:
            v = r[c["key"]]
            shown = html.escape(str(c["fmt"](v) if c.get("fmt") else v))
            prefix = ""
            if c.get("color"):
                prefix = f'<span class="dot" style="background:{c["color"](r)}"></span>'
            if c.get("bar"):
                width = max(float(v), 0) / maxima[c["key"]] * 64
                red = " red" if c.get("red") else ""
                prefix = f'<span class="bar{red}" style="width:{width:.0f}px"></span>'
            cells.append(f'<td class="{"txt" if c.get("text") else ""}">{prefix}{shown}</td>')
        body.append(f"<tr>{''.join(cells)}</tr>")
    st.html(f'<div class="tw" style="--tw-h:{height}px"><table class="tbl"><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')
