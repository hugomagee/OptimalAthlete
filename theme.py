"""Shared visual theme for the OptimalAthlete dashboard.

The dashboard is styled as a light research report rather than a dark
"analytics" console: paper surface, hairline rules instead of cards, system
sans throughout, tabular figures wherever numbers line up, and colour used only
where it encodes something.

The categorical palette below was checked for colour-vision deficiency
separation and contrast against the white surface before use. Series identity
is always carried by a label as well as a colour, so nothing is encoded by
colour alone.
"""

from __future__ import annotations

import plotly.graph_objects as go

# ── palette ───────────────────────────────────────────────────────────────
SURFACE = "#ffffff"
PLANE = "#f4f4f1"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e6e5de"
AXIS = "#c3c2b7"
RULE = "#d9d8d1"

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7", "#e87ba4"]
POSITIVE = "#2a78d6"
NEGATIVE = "#c02f2f"

FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'


def style_figure(
    fig: go.Figure,
    *,
    height: int = 320,
    reverse_y: bool = False,
    y_title: str = "",
    x_title: str = "",
    show_legend: bool = False,
) -> go.Figure:
    """Apply the report styling to a plotly figure.

    Hairline solid gridlines on the value axis only, no plot border, no chart
    title (the surrounding markdown carries it), recessive axis text.
    """
    fig.update_layout(
        template="simple_white",
        height=height,
        margin=dict(l=8, r=16, t=8, b=8),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family=FONT, size=12, color=INK_2),
        showlegend=show_legend,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            font=dict(size=11, color=INK_2), bgcolor="rgba(0,0,0,0)",
        ),
        hoverlabel=dict(
            bgcolor=SURFACE, bordercolor=AXIS,
            font=dict(family=FONT, size=12, color=INK),
        ),
    )
    fig.update_xaxes(
        title=dict(text=x_title, font=dict(size=11, color=MUTED)),
        showgrid=False, zeroline=False,
        linecolor=AXIS, linewidth=1, ticks="outside", tickcolor=AXIS,
        ticklen=4, tickfont=dict(size=11, color=MUTED),
    )
    fig.update_yaxes(
        title=dict(text=y_title, font=dict(size=11, color=MUTED)),
        showgrid=True, gridcolor=GRID, gridwidth=1, griddash="solid",
        zeroline=False, showline=False, ticks="",
        tickfont=dict(size=11, color=MUTED),
        autorange="reversed" if reverse_y else True,
    )
    return fig


# ── page CSS ──────────────────────────────────────────────────────────────
CSS = f"""
<style>
  html, body, [class*="css"] {{ font-family: {FONT}; }}

  /* Streamlit's own chrome — hidden so the page reads as a report */
  [data-testid="stToolbar"], [data-testid="stDecoration"],
  [data-testid="stStatusWidget"], #MainMenu, header[data-testid="stHeader"] {{
      display: none !important;
  }}

  .stApp {{ background: {PLANE}; }}
  .block-container {{
      background: {SURFACE};
      max-width: 1180px;
      padding: 2.2rem 3rem 3rem;
      border-left: 1px solid {RULE};
      border-right: 1px solid {RULE};
  }}

  /* headings */
  h1 {{ font-size: 1.7rem !important; font-weight: 680 !important;
       letter-spacing: -0.022em; color: {INK}; }}
  h2 {{ font-size: 0.78rem !important; font-weight: 660 !important;
       letter-spacing: 0.09em; text-transform: uppercase; color: {INK};
       border-bottom: 1px solid {INK}; padding-bottom: 0.5rem;
       margin-top: 2rem !important; }}
  h3 {{ font-size: 0.95rem !important; font-weight: 620 !important; color: {INK};
       margin-top: 1.4rem !important; }}

  /* key figures: rules, not cards */
  [data-testid="stMetric"] {{
      background: transparent; border: 0; border-left: 1px solid {RULE};
      border-radius: 0; padding: 0.1rem 1.1rem 0;
  }}
  [data-testid="stColumn"]:first-child [data-testid="stMetric"] {{
      border-left: 0; padding-left: 0;
  }}
  [data-testid="stMetricLabel"] p {{
      color: {MUTED} !important; font-size: 0.68rem !important;
      font-weight: 600 !important; letter-spacing: 0.07em; text-transform: uppercase;
  }}
  [data-testid="stMetricValue"] {{
      color: {INK} !important; font-size: 1.85rem !important;
      font-weight: 640 !important; letter-spacing: -0.02em;
  }}

  /* tabs as a rule-underlined row */
  .stTabs [data-baseweb="tab-list"] {{
      gap: 1.6rem; border-bottom: 1px solid {RULE}; background: transparent;
  }}
  .stTabs [data-baseweb="tab"] {{
      background: transparent; border: 0; border-radius: 0;
      color: {MUTED}; padding: 0.5rem 0; font-size: 0.82rem; font-weight: 560;
      letter-spacing: 0.03em;
  }}
  .stTabs [aria-selected="true"] {{
      color: {INK} !important; border-bottom: 2px solid {INK} !important;
  }}
  .stTabs [data-baseweb="tab-highlight"] {{ background: transparent; }}

  /* sidebar */
  [data-testid="stSidebar"] {{
      background: {PLANE}; border-right: 1px solid {RULE};
  }}
  [data-testid="stSidebar"] h2 {{ border-bottom: 1px solid {RULE}; }}

  /* tables */
  [data-testid="stDataFrame"] {{ border-radius: 0; }}
  [data-testid="stDataFrame"] * {{ font-variant-numeric: tabular-nums; }}

  hr {{ border-color: {RULE}; }}

  /* editorial blocks */
  .note {{
      border-left: 2px solid {AXIS}; padding: 0.1rem 0 0.1rem 0.95rem;
      color: {INK_2}; font-size: 0.87rem; line-height: 1.65; margin: 0.6rem 0 1.1rem;
  }}
  .note strong {{ color: {INK}; }}
  .kicker {{
      display: inline-block; font-size: 0.64rem; font-weight: 640;
      letter-spacing: 0.1em; text-transform: uppercase;
      color: {NEGATIVE}; border: 1px solid currentColor; padding: 2px 7px;
  }}
  .lede {{ color: {INK_2}; font-size: 0.92rem; margin-top: 0.35rem; }}

  /* protocol comparison rows */
  .row {{
      display: flex; justify-content: space-between; align-items: baseline;
      gap: 1rem; padding: 0.5rem 0; border-bottom: 1px solid {GRID};
      font-size: 0.87rem;
  }}
  .row-key {{ color: {INK_2}; }}
  .row-val {{ font-weight: 620; font-variant-numeric: tabular-nums; color: {INK}; }}
  .row-val.retracted {{ color: {MUTED}; text-decoration: line-through; }}

  .footer {{
      color: {MUTED}; font-size: 0.74rem; line-height: 1.7;
      border-top: 1px solid {INK}; padding-top: 0.9rem; margin-top: 2.5rem;
  }}
</style>
"""
