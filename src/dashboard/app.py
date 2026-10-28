"""Streamlit "control room" dashboard: predicted RUL of the engine fleet (FD001 test set).

Usage: streamlit run src/dashboard/app.py
"""

import sys
from pathlib import Path

# streamlit run does not add the project root to the import path: add it for `src.*`
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.data.load import load_cmapss, load_rul
from src.models.predict import load_model, predict_rul

# --- Visual identity ---------------------------------------------------------------
BG = "#0E1117"
CARD = "#1A1F2E"
BORDER = "#2A3142"
TEXT = "#E5E7EB"
MUTED = "#9CA3AF"
ACCENT = "#2DD4BF"

# Risk thresholds (in remaining cycles) and their colors, used everywhere
CRITICAL_RUL, ALERT_RUL = 30, 60
STATUS_COLORS = {"Critical": "#EF4444", "Warning": "#F59E0B", "Healthy": "#10B981"}

# Sensors shown in the engine view (C-MAPSS name + physical quantity)
SENSOR_LABELS = {
    "sensor_2": "T24 · LPC outlet temp. (°R)",
    "sensor_4": "T50 · LPT outlet temp. (°R)",
    "sensor_7": "P30 · HPC outlet pressure (psia)",
    "sensor_11": "Ps30 · HPC static pressure (psia)",
    "sensor_12": "phi · fuel flow / Ps30",
    "sensor_15": "BPR · bypass ratio",
}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* Single spacing scale, shared by all cards (between and inside them) */
:root {{ --gap: 1.25rem; --pad: 1.25rem 1.4rem; --radius: 14px; }}

html, body, [class*="css"], .stApp {{ font-family: 'Inter', sans-serif; }}
.stApp {{ background: {BG}; }}
.block-container {{ padding-top: 2rem; padding-bottom: 3rem; max-width: 1500px; }}

/* Main area (st.container key="page"): same vertical and horizontal gap everywhere */
.st-key-page {{ gap: var(--gap); }}
.st-key-page [data-testid="stHorizontalBlock"] {{ gap: var(--gap) !important; }}
/* Streamlit adds a -16px margin under each Markdown block: cancel it to keep --gap everywhere */
.st-key-page [data-testid="stMarkdownContainer"] {{ margin-bottom: 0 !important; }}
/* Inside a card: tighter gap between the title and the chart */
div[class*="st-key-card"] {{ gap: 0.75rem; }}
#MainMenu, footer {{ visibility: hidden; }}
header[data-testid="stHeader"] {{ background: transparent; }}

/* Sidebar */
section[data-testid="stSidebar"] {{ background: #121725; border-right: 1px solid {BORDER}; }}
.brand {{ padding: 0.4rem 0 1.2rem 0; border-bottom: 1px solid {BORDER}; margin-bottom: 1.2rem; }}
.brand-title {{ font-size: 1.25rem; font-weight: 700; color: {TEXT}; letter-spacing: -0.01em; }}
.brand-title span {{ color: {ACCENT}; }}
.brand-sub {{ font-size: 0.8rem; color: {MUTED}; margin-top: 0.25rem; }}
.side-label {{ font-size: 0.7rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase;
              color: {MUTED}; margin: 1.4rem 0 0.5rem 0; }}
.side-row {{ display: flex; justify-content: space-between; font-size: 0.85rem; color: {TEXT}; padding: 0.2rem 0; }}
.side-row span:first-child {{ color: {MUTED}; }}

/* Header bar: title on the left, fleet status pill on the right */
.topbar {{ display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.8rem;
           background: {CARD}; border: 1px solid {BORDER}; border-radius: var(--radius); padding: 0.9rem 1.4rem;
           box-shadow: 0 4px 18px rgba(0,0,0,0.25); }}
.topbar-title {{ display: flex; align-items: center; gap: 0.7rem; font-size: 1.2rem; font-weight: 700;
                 color: {TEXT}; letter-spacing: -0.01em; }}
.topbar-title::before {{ content: ""; width: 10px; height: 10px; border-radius: 3px; background: {ACCENT};
                         box-shadow: 0 0 10px {ACCENT}; }}
.status-pill {{ display: inline-flex; align-items: center; gap: 0.9rem; padding: 0.4rem 1rem; border-radius: 999px;
                background: #121725; border: 1px solid {BORDER}; font-size: 0.85rem; color: {TEXT};
                font-variant-numeric: tabular-nums; }}
.status-pill .item {{ display: inline-flex; align-items: center; gap: 0.4rem; }}
.status-pill .dot {{ width: 8px; height: 8px; border-radius: 50%; background: var(--c); }}
.status-pill .sep {{ color: {BORDER}; }}

/* Page and section titles */
.page-title {{ font-size: 1.45rem; line-height: 1.2; font-weight: 700; color: {TEXT}; letter-spacing: -0.02em; margin: 0; }}
.page-sub {{ color: {MUTED}; font-size: 0.9rem; margin: 0.3rem 0 0 0; }}
.section-title {{ font-size: 1.02rem; font-weight: 600; color: {TEXT}; margin: 0 0 0.2rem 0; }}
.section-sub {{ font-size: 0.82rem; color: {MUTED}; margin: 0; }}

/* KPI cards */
.kpi {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: var(--radius); padding: var(--pad);
        box-shadow: 0 4px 18px rgba(0,0,0,0.25); position: relative; overflow: hidden; }}
.kpi::before {{ content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 4px; background: var(--c); }}
.kpi-label {{ font-size: 0.75rem; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; color: {MUTED}; }}
.kpi-value {{ font-size: 2.1rem; font-weight: 700; color: {TEXT}; margin-top: 0.35rem; line-height: 1.1;
              font-variant-numeric: tabular-nums; }}
.kpi-value small {{ font-size: 0.95rem; font-weight: 500; color: {MUTED}; margin-left: 0.25rem; }}
.kpi-hint {{ font-size: 0.78rem; color: {MUTED}; margin-top: 0.35rem; }}

/* "Card" containers around charts (st.container with a card_* key) */
div[class*="st-key-card"] {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: var(--radius);
                             padding: var(--pad); box-shadow: 0 4px 18px rgba(0,0,0,0.25); }}

/* Status badges */
.badge {{ display: inline-flex; align-items: center; gap: 0.4rem; padding: 0.18rem 0.65rem; border-radius: 999px;
          font-size: 0.78rem; font-weight: 600; color: var(--c); background: color-mix(in srgb, var(--c) 16%, transparent);
          border: 1px solid color-mix(in srgb, var(--c) 45%, transparent); }}
.badge::before {{ content: ""; width: 7px; height: 7px; border-radius: 50%; background: var(--c); }}

/* Fleet table */
.table-wrap {{ max-height: 460px; overflow-y: auto; border-radius: 10px; border: 1px solid {BORDER}; }}
table.fleet {{ width: 100%; border-collapse: collapse; font-size: 0.88rem; color: {TEXT}; }}
table.fleet th {{ position: sticky; top: 0; background: #151A27; color: {MUTED}; font-weight: 600; font-size: 0.72rem;
                  letter-spacing: 0.06em; text-transform: uppercase; text-align: left; padding: 0.7rem 1rem;
                  border-bottom: 1px solid {BORDER}; }}
table.fleet td {{ padding: 0.55rem 1rem; border-bottom: 1px solid #222838; font-variant-numeric: tabular-nums; }}
table.fleet tr:hover td {{ background: #20273A; }}
.rulbar {{ display: flex; align-items: center; gap: 0.6rem; }}
.rulbar-track {{ flex: 1; max-width: 140px; height: 6px; border-radius: 3px; background: #2A3142; overflow: hidden; }}
.rulbar-fill {{ height: 100%; border-radius: 3px; background: var(--c); }}

/* Engine details */
.gauge-value {{ text-align: center; font-size: 1.6rem; font-weight: 600; color: {TEXT}; margin-top: -0.4rem;
                font-variant-numeric: tabular-nums; }}
.gauge-value span {{ font-size: 0.95rem; font-weight: 500; color: {MUTED}; margin-left: 0.2rem; }}
.info-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; }}
.info {{ background: #151A27; border: 1px solid {BORDER}; border-radius: 10px; padding: 0.7rem 0.9rem; }}
.info-label {{ font-size: 0.7rem; color: {MUTED}; text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600; }}
.info-value {{ font-size: 1.15rem; font-weight: 600; color: {TEXT}; margin-top: 0.2rem; font-variant-numeric: tabular-nums; }}
</style>
"""


# --- Data and model ----------------------------------------------------------------
def risk_status(rul: float) -> str:
    """Risk status from the predicted RUL (exclusive thresholds)."""
    if rul < CRITICAL_RUL:
        return "Critical"
    if rul < ALERT_RUL:
        return "Warning"
    return "Healthy"


@st.cache_resource
def get_model():
    """Load the LSTM + scaler once for the whole session (Streamlit cache)."""
    return load_model()


@st.cache_data
def get_fleet() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Raw FD001 test sensors + fleet table (predicted RUL, status, true RUL)."""
    model, scaler, cfg = get_model()
    test = load_cmapss("FD001", "test")
    pred = predict_rul(test, model, scaler, cfg)

    fleet = pd.DataFrame({
        "unit": pred.index,
        "rul_pred": pred.to_numpy(),
        "cycles": test.groupby("unit_number")["time_in_cycles"].max().loc[pred.index].to_numpy(),
        # True RUL (capped as in training): a reference, unknown in real operation
        "rul_true": load_rul("FD001").loc[pred.index].clip(upper=cfg["cap"]).to_numpy(),
    })
    fleet["status"] = fleet["rul_pred"].map(risk_status)
    return test, fleet.sort_values("rul_pred").reset_index(drop=True)


# --- HTML components ---------------------------------------------------------------
# HTML is written on a single line: in Markdown, an indented line would become a code block.
def badge(status: str) -> str:
    return f'<span class="badge" style="--c:{STATUS_COLORS[status]}">{status}</span>'


def kpi_card(label: str, value: str, hint: str, color: str, unit: str = "") -> str:
    unit_html = f"<small>{unit}</small>" if unit else ""
    return (
        f'<div class="kpi" style="--c:{color}"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}{unit_html}</div><div class="kpi-hint">{hint}</div></div>'
    )


def section(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="section-title">{title}</div><div class="section-sub">{subtitle}</div>', unsafe_allow_html=True)


def style_fig(fig: go.Figure, height: int, top: int = 10) -> go.Figure:
    """Shared Plotly styling: transparent background (the card provides it), subtle grid."""
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=top, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=TEXT, size=12),
        hoverlabel=dict(bgcolor="#111827", bordercolor=BORDER, font=dict(color=TEXT, family="Inter")),
    )
    fig.update_xaxes(gridcolor="#232A3B", zerolinecolor=BORDER, linecolor=BORDER, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor="#232A3B", zerolinecolor=BORDER, linecolor=BORDER, tickfont=dict(color=MUTED))
    return fig


# --- Views ---------------------------------------------------------------------------
def render_topbar(fleet: pd.DataFrame) -> None:
    """Header bar: title on the left, fleet status pill on the right."""
    counts = fleet["status"].value_counts()
    items = [
        ("#64748B", f"{len(fleet)} engines"),
        (STATUS_COLORS["Critical"], f"{counts.get('Critical', 0)} critical"),
        (STATUS_COLORS["Warning"], f"{counts.get('Warning', 0)} warning"),
    ]
    # The colored dot carries the status; the label stays in the text color (readability)
    pill = '<span class="sep">·</span>'.join(
        f'<span class="item"><span class="dot" style="--c:{c}"></span>{label}</span>' for c, label in items
    )
    st.markdown(
        '<div class="topbar"><div class="topbar-title">Predictive fleet monitoring</div>'
        f'<div class="status-pill">{pill}</div></div>',
        unsafe_allow_html=True,
    )


def render_kpis(fleet: pd.DataFrame) -> None:
    n_crit = int((fleet["status"] == "Critical").sum())
    n_alert = int((fleet["status"] == "Warning").sum())
    cards = [
        kpi_card("Critical engines", str(n_crit), f"Predicted RUL &lt; {CRITICAL_RUL} cycles", STATUS_COLORS["Critical"]),
        kpi_card("Warning engines", str(n_alert), f"RUL between {CRITICAL_RUL} and {ALERT_RUL} cycles", STATUS_COLORS["Warning"]),
        kpi_card("Mean predicted RUL", f"{fleet['rul_pred'].mean():.0f}", "across the whole fleet", ACCENT, "cycles"),
        kpi_card("Engines monitored", str(len(fleet)), "NASA C-MAPSS FD001 test set", "#64748B"),
    ]
    for col, card in zip(st.columns(4, gap="medium"), cards):
        col.markdown(card, unsafe_allow_html=True)


def render_fleet(fleet: pd.DataFrame) -> None:
    # Chart: one bar per engine, sorted by increasing RUL, color = status
    with st.container(key="card_fleet_chart"):
        section("Predicted RUL per engine", "Sorted from most urgent to healthiest · hover over a bar for details")
        fig = go.Figure(go.Bar(
            x=[f"#{u:03d}" for u in fleet["unit"]],
            y=fleet["rul_pred"],
            marker=dict(color=fleet["status"].map(STATUS_COLORS), line=dict(width=0)),
            customdata=fleet[["status", "cycles"]],
            hovertemplate="<b>Engine %{x}</b><br>Predicted RUL: %{y:.0f} cycles"
                          "<br>Status: %{customdata[0]}<br>Observed cycles: %{customdata[1]}<extra></extra>",
        ))
        # Threshold lines
        for y, status in ((CRITICAL_RUL, "Critical"), (ALERT_RUL, "Warning")):
            fig.add_hline(y=y, line=dict(color=STATUS_COLORS[status], width=1, dash="dot"),
                          annotation_text=f"{status.lower()} threshold · {y}", annotation_position="top left",
                          annotation_font=dict(color=STATUS_COLORS[status], size=11))
        fig.update_layout(bargap=0.25, barcornerradius=3, showlegend=False)
        # Force a categorical axis: otherwise Plotly parses "#081" as the number 81 and loses the RUL ordering
        fig.update_xaxes(type="category", showgrid=False, showticklabels=False, title=None)
        fig.update_yaxes(title=dict(text="Predicted RUL (cycles)", font=dict(color=MUTED)), range=[0, 130])
        st.plotly_chart(style_fig(fig, 360), use_container_width=True, config={"displayModeBar": False})

    # Table: sorted by risk (increasing RUL), status badge and mini RUL bar
    with st.container(key="card_fleet_table"):
        section("Fleet registry", "Ranked by risk · true RUL shown as a reference to assess the model")
        rows = []
        for r in fleet.itertuples():
            color = STATUS_COLORS[r.status]
            pct = r.rul_pred / 125 * 100
            rows.append(
                f"<tr><td><b>Engine {r.unit:03d}</b></td><td>{badge(r.status)}</td>"
                f'<td><div class="rulbar"><span>{r.rul_pred:.0f}</span><div class="rulbar-track">'
                f'<div class="rulbar-fill" style="width:{pct:.0f}%;--c:{color}"></div></div></div></td>'
                f"<td>{r.cycles}</td><td>{r.rul_true:.0f}</td><td>{round(r.rul_pred - r.rul_true):+d}</td></tr>"
            )
        table = (
            '<div class="table-wrap"><table class="fleet"><thead><tr><th>Engine</th><th>Status</th>'
            "<th>Predicted RUL</th><th>Observed cycles</th><th>True RUL</th><th>Error</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>"
        )
        st.markdown(table, unsafe_allow_html=True)


def render_engine(test: pd.DataFrame, fleet: pd.DataFrame) -> None:
    # Selector: engines ranked by urgency, most critical selected by default
    by_unit = fleet.set_index("unit")
    unit = st.selectbox(
        "Engine",
        fleet["unit"].tolist(),
        format_func=lambda u: f"Engine {u:03d} · {by_unit.loc[u, 'status']} · RUL {by_unit.loc[u, 'rul_pred']:.0f} cycles",
    )
    row = by_unit.loc[unit]

    left, right = st.columns([1, 2.2], gap="medium")

    with left, st.container(key="card_gauge"):
        section("Predicted RUL", "Last 30-cycle window")
        gauge = go.Figure(go.Indicator(
            # The value is rendered in HTML below the gauge (Plotly would overlap it with the arc)
            mode="gauge",
            value=row["rul_pred"],
            gauge=dict(
                # Scale 0 -> 125 (RUL cap), with ticks on the risk thresholds
                axis=dict(range=[0, 125], tickvals=[0, CRITICAL_RUL, ALERT_RUL, 90, 125],
                          tickwidth=1, tickcolor=BORDER, tickfont=dict(color=MUTED)),
                bar=dict(color=TEXT, thickness=0.22),
                bgcolor="rgba(0,0,0,0)",
                borderwidth=0,
                steps=[
                    dict(range=[0, CRITICAL_RUL], color="rgba(239,68,68,0.75)"),
                    dict(range=[CRITICAL_RUL, ALERT_RUL], color="rgba(245,158,11,0.7)"),
                    dict(range=[ALERT_RUL, 125], color="rgba(16,185,129,0.65)"),
                ],
            ),
        ))
        style_fig(gauge, 210, top=20)
        # Wider side margins: otherwise the 0 and 125 tick labels are clipped
        gauge.update_layout(margin=dict(l=30, r=30, b=0))
        st.plotly_chart(gauge, use_container_width=True, config={"displayModeBar": False})
        st.markdown(
            f'<div class="gauge-value">{row["rul_pred"]:.0f}<span>/125 cycles</span></div>'
            f'<div style="text-align:center;margin:0.4rem 0 0.75rem 0">{badge(row["status"])}</div>'
            '<div class="info-grid">'
            f'<div class="info"><div class="info-label">Observed cycles</div><div class="info-value">{row["cycles"]}</div></div>'
            f'<div class="info"><div class="info-label">True RUL</div><div class="info-value">{row["rul_true"]:.0f}</div></div>'
            f'<div class="info"><div class="info-label">Model error</div><div class="info-value">{round(row["rul_pred"] - row["rul_true"]):+d}</div></div>'
            f'<div class="info"><div class="info-label">Risk rank</div><div class="info-value">{fleet.index[fleet["unit"] == unit][0] + 1} / {len(fleet)}</div></div>'
            "</div>",
            unsafe_allow_html=True,
        )

    with right, st.container(key="card_sensors"):
        section("Key sensors", "Raw values over cycles · shaded area = window used by the model")
        d = test[test["unit_number"] == unit].sort_values("time_in_cycles")
        last_cycle = d["time_in_cycles"].max()
        fig = make_subplots(rows=3, cols=2, subplot_titles=list(SENSOR_LABELS.values()),
                            vertical_spacing=0.13, horizontal_spacing=0.08)
        for i, sensor in enumerate(SENSOR_LABELS):
            r, c = i // 2 + 1, i % 2 + 1
            fig.add_trace(go.Scatter(
                x=d["time_in_cycles"], y=d[sensor], mode="lines", line=dict(color=ACCENT, width=1.8),
                name=SENSOR_LABELS[sensor], hovertemplate="Cycle %{x}<br>%{y:.2f}<extra></extra>",
            ), row=r, col=c)
            # Window of the last 30 cycles, the one fed to the LSTM
            fig.add_vrect(x0=max(last_cycle - 29, 1), x1=last_cycle, fillcolor="#FFFFFF", opacity=0.06,
                          line_width=0, row=r, col=c)
        fig.update_annotations(font=dict(size=12, color=MUTED))
        fig.update_layout(showlegend=False)
        st.plotly_chart(style_fig(fig, 540, top=30), use_container_width=True, config={"displayModeBar": False})


def render_sidebar(fleet: pd.DataFrame, cfg: dict) -> str:
    with st.sidebar:
        st.markdown(
            '<div class="brand"><div class="brand-title">PdM <span>Copilot</span></div>'
            '<div class="brand-sub">Predictive maintenance · NASA C-MAPSS</div></div>',
            unsafe_allow_html=True,
        )
        view = st.radio("Navigation", ["Fleet View", "Engine View"], label_visibility="collapsed")

        rmse = ((fleet["rul_pred"] - fleet["rul_true"]) ** 2).mean() ** 0.5
        st.markdown(
            '<div class="side-label">Model</div>'
            '<div class="side-row"><span>Architecture</span><span>LSTM</span></div>'
            f'<div class="side-row"><span>Window</span><span>{cfg["window_size"]} cycles</span></div>'
            f'<div class="side-row"><span>RUL cap</span><span>{cfg["cap"]} cycles</span></div>'
            f'<div class="side-row"><span>Test RMSE</span><span>{rmse:.2f}</span></div>'
            '<div class="side-label">Risk thresholds</div>'
            f'<div class="side-row">{badge("Critical")}<span>&lt; {CRITICAL_RUL}</span></div>'
            f'<div class="side-row">{badge("Warning")}<span>{CRITICAL_RUL} – {ALERT_RUL}</span></div>'
            f'<div class="side-row">{badge("Healthy")}<span>≥ {ALERT_RUL}</span></div>',
            unsafe_allow_html=True,
        )
    return view


def main() -> None:
    st.set_page_config(page_title="PdM Copilot", page_icon=":material/precision_manufacturing:", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    test, fleet = get_fleet()
    _, _, cfg = get_model()
    view = render_sidebar(fleet, cfg)

    subtitles = {
        "Fleet View": "Fleet health overview · RUL predicted by the LSTM from each engine's last window",
        "Engine View": "Single-engine diagnosis · predicted RUL and sensor trends",
    }
    # All main content lives in one container: the gap between cards is set by --gap
    with st.container(key="page"):
        render_topbar(fleet)
        st.markdown(f'<div class="page-title">{view}</div><div class="page-sub">{subtitles[view]}</div>',
                    unsafe_allow_html=True)
        render_kpis(fleet)

        if view == "Fleet View":
            render_fleet(fleet)
        else:
            render_engine(test, fleet)


main()
