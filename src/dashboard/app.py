"""Dashboard Streamlit "salle de contrôle" : RUL prédite de la flotte de moteurs (test FD001).

Usage : streamlit run src/dashboard/app.py
"""

import sys
from pathlib import Path

# streamlit run n'ajoute pas la racine du projet au chemin d'import : on le fait pour `src.*`
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.data.load import load_cmapss, load_rul
from src.models.predict import load_model, predict_rul

# --- Charte graphique --------------------------------------------------------------
BG = "#0E1117"
CARD = "#1A1F2E"
BORDER = "#2A3142"
TEXT = "#E5E7EB"
MUTED = "#9CA3AF"
ACCENT = "#2DD4BF"

# Seuils de risque (en cycles restants) et couleurs associées, utilisés partout
CRITICAL_RUL, ALERT_RUL = 30, 60
STATUS_COLORS = {"Critical": "#EF4444", "Warning": "#F59E0B", "Healthy": "#10B981"}

# Capteurs affichés dans la vue moteur (nom C-MAPSS + grandeur physique)
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

/* Espacement unique, partagé par toutes les cartes (entre elles et à l'intérieur) */
:root {{ --gap: 1.25rem; --pad: 1.25rem 1.4rem; --radius: 14px; }}

html, body, [class*="css"], .stApp {{ font-family: 'Inter', sans-serif; }}
.stApp {{ background: {BG}; }}
.block-container {{ padding-top: 2rem; padding-bottom: 3rem; max-width: 1500px; }}

/* Zone principale (st.container key="page") : même écart vertical et horizontal partout */
.st-key-page {{ gap: var(--gap); }}
.st-key-page [data-testid="stHorizontalBlock"] {{ gap: var(--gap) !important; }}
/* Streamlit met une marge de -16px sous chaque bloc Markdown : on l'annule pour garder --gap partout */
.st-key-page [data-testid="stMarkdownContainer"] {{ margin-bottom: 0 !important; }}
/* À l'intérieur d'une carte : écart plus serré entre le titre et le graphique */
div[class*="st-key-card"] {{ gap: 0.75rem; }}
#MainMenu, footer {{ visibility: hidden; }}
header[data-testid="stHeader"] {{ background: transparent; }}

/* Barre latérale */
section[data-testid="stSidebar"] {{ background: #121725; border-right: 1px solid {BORDER}; }}
.brand {{ padding: 0.4rem 0 1.2rem 0; border-bottom: 1px solid {BORDER}; margin-bottom: 1.2rem; }}
.brand-title {{ font-size: 1.25rem; font-weight: 700; color: {TEXT}; letter-spacing: -0.01em; }}
.brand-title span {{ color: {ACCENT}; }}
.brand-sub {{ font-size: 0.8rem; color: {MUTED}; margin-top: 0.25rem; }}
.side-label {{ font-size: 0.7rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase;
              color: {MUTED}; margin: 1.4rem 0 0.5rem 0; }}
.side-row {{ display: flex; justify-content: space-between; font-size: 0.85rem; color: {TEXT}; padding: 0.2rem 0; }}
.side-row span:first-child {{ color: {MUTED}; }}

/* Barre d'en-tête : titre à gauche, pastille d'état de la flotte à droite */
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

/* Titres de page et de section */
.page-title {{ font-size: 1.45rem; line-height: 1.2; font-weight: 700; color: {TEXT}; letter-spacing: -0.02em; margin: 0; }}
.page-sub {{ color: {MUTED}; font-size: 0.9rem; margin: 0.3rem 0 0 0; }}
.section-title {{ font-size: 1.02rem; font-weight: 600; color: {TEXT}; margin: 0 0 0.2rem 0; }}
.section-sub {{ font-size: 0.82rem; color: {MUTED}; margin: 0; }}

/* Cartes KPI */
.kpi {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: var(--radius); padding: var(--pad);
        box-shadow: 0 4px 18px rgba(0,0,0,0.25); position: relative; overflow: hidden; }}
.kpi::before {{ content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 4px; background: var(--c); }}
.kpi-label {{ font-size: 0.75rem; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; color: {MUTED}; }}
.kpi-value {{ font-size: 2.1rem; font-weight: 700; color: {TEXT}; margin-top: 0.35rem; line-height: 1.1;
              font-variant-numeric: tabular-nums; }}
.kpi-value small {{ font-size: 0.95rem; font-weight: 500; color: {MUTED}; margin-left: 0.25rem; }}
.kpi-hint {{ font-size: 0.78rem; color: {MUTED}; margin-top: 0.35rem; }}

/* Conteneurs "carte" autour des graphiques (st.container avec une clé card_*) */
div[class*="st-key-card"] {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: var(--radius);
                             padding: var(--pad); box-shadow: 0 4px 18px rgba(0,0,0,0.25); }}

/* Badges de statut */
.badge {{ display: inline-flex; align-items: center; gap: 0.4rem; padding: 0.18rem 0.65rem; border-radius: 999px;
          font-size: 0.78rem; font-weight: 600; color: var(--c); background: color-mix(in srgb, var(--c) 16%, transparent);
          border: 1px solid color-mix(in srgb, var(--c) 45%, transparent); }}
.badge::before {{ content: ""; width: 7px; height: 7px; border-radius: 50%; background: var(--c); }}

/* Tableau de flotte */
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

/* Infos moteur */
.gauge-value {{ text-align: center; font-size: 1.6rem; font-weight: 600; color: {TEXT}; margin-top: -0.4rem;
                font-variant-numeric: tabular-nums; }}
.gauge-value span {{ font-size: 0.95rem; font-weight: 500; color: {MUTED}; margin-left: 0.2rem; }}
.info-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; }}
.info {{ background: #151A27; border: 1px solid {BORDER}; border-radius: 10px; padding: 0.7rem 0.9rem; }}
.info-label {{ font-size: 0.7rem; color: {MUTED}; text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600; }}
.info-value {{ font-size: 1.15rem; font-weight: 600; color: {TEXT}; margin-top: 0.2rem; font-variant-numeric: tabular-nums; }}
</style>
"""


# --- Données et modèle -------------------------------------------------------------
def risk_status(rul: float) -> str:
    """Statut de risque à partir de la RUL prédite (seuils exclusifs)."""
    if rul < CRITICAL_RUL:
        return "Critical"
    if rul < ALERT_RUL:
        return "Warning"
    return "Healthy"


@st.cache_resource
def get_model():
    """Charge le LSTM + scaler une seule fois pour toute la session (cache Streamlit)."""
    return load_model()


@st.cache_data
def get_fleet() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Capteurs bruts du test FD001 + tableau de flotte (RUL prédite, statut, RUL réelle)."""
    model, scaler, cfg = get_model()
    test = load_cmapss("FD001", "test")
    pred = predict_rul(test, model, scaler, cfg)

    fleet = pd.DataFrame({
        "unit": pred.index,
        "rul_pred": pred.to_numpy(),
        "cycles": test.groupby("unit_number")["time_in_cycles"].max().loc[pred.index].to_numpy(),
        # Vraie RUL (plafonnée comme à l'entraînement) : référence, inconnue en conditions réelles
        "rul_true": load_rul("FD001").loc[pred.index].clip(upper=cfg["cap"]).to_numpy(),
    })
    fleet["status"] = fleet["rul_pred"].map(risk_status)
    return test, fleet.sort_values("rul_pred").reset_index(drop=True)


# --- Composants HTML ---------------------------------------------------------------
# Le HTML est écrit sur une ligne : en Markdown, une ligne indentée deviendrait un bloc de code.
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
    """Habillage Plotly commun : fond transparent (la carte fait le fond), grille discrète."""
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


# --- Vues ----------------------------------------------------------------------------
def render_topbar(fleet: pd.DataFrame) -> None:
    """Barre d'en-tête : titre à gauche, pastille d'état de la flotte à droite."""
    counts = fleet["status"].value_counts()
    items = [
        ("#64748B", f"{len(fleet)} engines"),
        (STATUS_COLORS["Critical"], f"{counts.get('Critical', 0)} critical"),
        (STATUS_COLORS["Warning"], f"{counts.get('Warning', 0)} warning"),
    ]
    # Le point coloré porte le statut, le texte reste dans la couleur du texte (lisibilité)
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
    # Graphique : une barre par moteur, triée par RUL croissante, couleur = statut
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
        # Lignes de seuil
        for y, status in ((CRITICAL_RUL, "Critical"), (ALERT_RUL, "Warning")):
            fig.add_hline(y=y, line=dict(color=STATUS_COLORS[status], width=1, dash="dot"),
                          annotation_text=f"{status.lower()} threshold · {y}", annotation_position="top left",
                          annotation_font=dict(color=STATUS_COLORS[status], size=11))
        fig.update_layout(bargap=0.25, barcornerradius=3, showlegend=False)
        # Axe forcé en catégories : sinon Plotly lit "#081" comme le nombre 81 et perd le tri par RUL
        fig.update_xaxes(type="category", showgrid=False, showticklabels=False, title=None)
        fig.update_yaxes(title=dict(text="Predicted RUL (cycles)", font=dict(color=MUTED)), range=[0, 130])
        st.plotly_chart(style_fig(fig, 360), use_container_width=True, config={"displayModeBar": False})

    # Tableau : trié par risque (RUL croissante), badge de statut et mini-barre de RUL
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
    # Sélecteur : moteurs classés par urgence, le plus critique par défaut
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
            # La valeur est affichée en HTML sous la jauge (Plotly la superpose à l'arc)
            mode="gauge",
            value=row["rul_pred"],
            gauge=dict(
                # Échelle 0 -> 125 (plafond de RUL), graduée sur les seuils de risque
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
        # Marges latérales plus larges : sinon les graduations 0 et 125 sont coupées
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
            # Fenêtre des 30 derniers cycles, celle qui alimente le LSTM
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
    # Tout le contenu principal dans un seul conteneur : l'écart entre cartes y est fixé par --gap
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
