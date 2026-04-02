"""
═══════════════════════════════════════════════════════════════
  DASHBOARD VISUALISASI TOPIC MODELING
  Dataset : WVI – Tanggapan Anak terhadap Bencana Alam
  Jalankan: streamlit run dashboard_topik.py
═══════════════════════════════════════════════════════════════
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from collections import Counter

# ─── PAGE CONFIG ────────────────────────────────────────────
st.set_page_config(
    page_title="Analisis Topik – WVI Bencana",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CUSTOM CSS ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Space+Mono:wght@400;700&display=swap');

/* ---- Global ---- */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}
.stApp {
    background: #0d1117;
    color: #e6edf3;
}

/* ---- Sidebar ---- */
[data-testid="stSidebar"] {
    background: #161b22 !important;
    border-right: 1px solid #21262d;
}
[data-testid="stSidebar"] * {
    color: #e6edf3 !important;
}

/* ---- Header Banner ---- */
.hero-banner {
    background: linear-gradient(135deg, #0f3460 0%, #16213e 40%, #1a1a2e 70%, #0d1117 100%);
    border: 1px solid #21262d;
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(56,139,253,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #58a6ff, #79c0ff, #a5d6ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.4rem 0;
    font-family: 'Plus Jakarta Sans', sans-serif;
}
.hero-sub {
    font-size: 0.95rem;
    color: #8b949e;
    font-weight: 400;
    margin: 0;
}

/* ---- Metric Cards ---- */
.metric-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.5rem;
    flex-wrap: wrap;
}
.metric-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 1.2rem 1.6rem;
    flex: 1;
    min-width: 140px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #388bfd; }
.metric-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #388bfd, #79c0ff);
    border-radius: 0 0 12px 12px;
}
.metric-num {
    font-size: 2rem;
    font-weight: 800;
    color: #58a6ff;
    font-family: 'Space Mono', monospace;
    line-height: 1;
    margin-bottom: 0.3rem;
}
.metric-label {
    font-size: 0.78rem;
    color: #8b949e;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ---- Section Headers ---- */
.section-header {
    font-size: 1.1rem;
    font-weight: 700;
    color: #e6edf3;
    margin: 0 0 1rem 0;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid #21262d;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ---- Topic Pills ---- */
.topic-pill {
    display: inline-block;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    margin: 2px;
}

/* ---- Cards for sample texts ---- */
.quote-card {
    background: #161b22;
    border-left: 3px solid #388bfd;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    margin-bottom: 0.6rem;
    font-size: 0.88rem;
    color: #c9d1d9;
    line-height: 1.6;
}

/* ---- Plotly chart containers ---- */
.chart-container {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 1rem;
}

/* ---- Streamlit overrides ---- */
.stSelectbox label, .stMultiSelect label, .stRadio label {
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    color: #8b949e !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
div[data-testid="stMetric"] {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 1rem;
}
h1, h2, h3 { color: #e6edf3 !important; }
</style>
""", unsafe_allow_html=True)

# ─── PALETTE ────────────────────────────────────────────────
TOPIC_COLORS = [
    "#388bfd", "#3fb950", "#f0883e",
    "#bc8cff", "#ff7b72", "#58a6ff", "#ffa657",
]

TOPIC_LABELS = {
    0: "Reaksi Emosional",
    1: "Keluhan Fisik",
    2: "Dampak Kerusakan & Kebutuhan Dasar",
    3: "Tindakan Evakuasi & Respons Bencana",
    4: "Dukungan Sosial",
    5: "Korban & Kondisi Pasca-Bencana",
    6: "Aktivitas Sehari-hari & Kebutuhan Makan",
}

TOPIC_ICONS = ["😨", "🤕", "🏚️", "🏃", "🤝", "🌊", "🍜"]

TOPIC_KEYWORDS = {
    0: ["takut", "sedih", "cemas", "hancur", "banjir", "kehilangan", "longsor"],
    1: ["sakit", "pegal", "capek", "perut", "pinggang", "bahu", "gatal"],
    2: ["air bersih", "rumah", "sekolah", "makanan", "hancur", "kebutuhan"],
    3: ["lari", "evakuasi", "suara", "hujan", "petir", "lumpur", "membersihkan"],
    4: ["teman", "bertemu", "cerita", "keluarga", "orangtua", "saudara"],
    5: ["banjir bandang", "hanyut", "korban", "mendengar", "terjadi", "adik"],
    6: ["makan", "indomie", "nasi", "kaki", "jalan", "aktivitas", "sehari"],
}

# ─── LOAD DATA ──────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("hasil_topic_modeling.csv")
    df["topic_name"] = df["dominant_topic_id"].map(TOPIC_LABELS)
    df["topic_icon"] = df["dominant_topic_id"].map(
        {k: v for k, v in enumerate(TOPIC_ICONS)}
    )
    df["topic_display"] = df.apply(
        lambda r: f"{TOPIC_ICONS[r['dominant_topic_id']]} {TOPIC_LABELS[r['dominant_topic_id']]}", axis=1
    )
    return df


df = load_data()

# ─── CHART THEME ────────────────────────────────────────────
CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(22,27,34,1)",
    font=dict(family="Plus Jakarta Sans", color="#c9d1d9", size=12),
    margin=dict(t=40, b=10, l=10, r=10),
    legend=dict(
        bgcolor="rgba(22,27,34,0.9)",
        bordercolor="#21262d",
        borderwidth=1,
        font=dict(size=11),
    ),
)

def style_chart(fig, height=380):
    fig.update_layout(**CHART_LAYOUT, height=height)
    fig.update_xaxes(gridcolor="#21262d", zerolinecolor="#21262d")
    fig.update_yaxes(gridcolor="#21262d", zerolinecolor="#21262d")
    return fig


# ═══════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🌊 Filter Data")
    st.markdown("---")

    sel_umur = st.multiselect(
        "Kelompok Umur",
        options=sorted(df["Umur"].unique()),
        default=sorted(df["Umur"].unique()),
    )
    sel_gender = st.multiselect(
        "Jenis Kelamin",
        options=sorted(df["Jenis Kelamin"].unique()),
        default=sorted(df["Jenis Kelamin"].unique()),
    )
    sel_wilayah = st.multiselect(
        "Wilayah",
        options=sorted(df["Wilayah"].unique()),
        default=sorted(df["Wilayah"].unique()),
    )

    st.markdown("---")
    st.markdown("**Topik yang Ditampilkan**")
    sel_topics = st.multiselect(
        "Pilih topik",
        options=list(TOPIC_LABELS.values()),
        default=list(TOPIC_LABELS.values()),
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.75rem;color:#8b949e;'>"
        "Dataset: WVI – Tanggapan Anak<br>terhadap Bencana Alam<br>"
        "Metode: LDA Topic Modeling<br>"
        "n = 529 dokumen · 7 topik"
        "</div>",
        unsafe_allow_html=True,
    )

# ─── FILTER ─────────────────────────────────────────────────
mask = (
    df["Umur"].isin(sel_umur)
    & df["Jenis Kelamin"].isin(sel_gender)
    & df["Wilayah"].isin(sel_wilayah)
    & df["topic_name"].isin(sel_topics)
)
dff = df[mask].copy()

# ═══════════════════════════════════════════════════════════
# HERO BANNER
# ═══════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-banner">
    <p class="hero-title">🌊 Analisis Topik – Tanggapan Anak terhadap Bencana Alam</p>
    <p class="hero-sub">
        Dashboard interaktif hasil <strong>LDA Topic Modeling</strong> dari dataset WVI
        &nbsp;·&nbsp; 7 topik teridentifikasi &nbsp;·&nbsp; 529 tanggapan anak
    </p>
</div>
""", unsafe_allow_html=True)

# ─── METRIC CARDS ───────────────────────────────────────────
top_topic = dff["topic_name"].value_counts().idxmax() if len(dff) > 0 else "-"
top_count = dff["topic_name"].value_counts().max() if len(dff) > 0 else 0

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-num">{len(dff)}</div>
        <div class="metric-label">Total Tanggapan</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-num">{len(sel_topics)}</div>
        <div class="metric-label">Topik Aktif</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-num">{len(sel_wilayah)}</div>
        <div class="metric-label">Wilayah</div>
    </div>""", unsafe_allow_html=True)
with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-num">{top_count}</div>
        <div class="metric-label">Dokumen Topik Terbesar</div>
    </div>""", unsafe_allow_html=True)
with col5:
    pct = round(top_count / len(dff) * 100, 1) if len(dff) > 0 else 0
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-num">{pct}%</div>
        <div class="metric-label">Proporsi Terbesar</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# TAB NAVIGATION
# ═══════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Distribusi Topik",
    "👥 Demografi",
    "🗺️ Wilayah",
    "🔍 Eksplorasi Topik",
    "📝 Tabel Data",
])

# ═══════════════════════════════════════════════════════════
# TAB 1 – DISTRIBUSI TOPIK
# ═══════════════════════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-header">📊 Sebaran Topik Keseluruhan</p>', unsafe_allow_html=True)

    row1_l, row1_r = st.columns([3, 2])

    # --- Horizontal bar chart ---
    with row1_l:
        topic_count = (
            dff["topic_name"].value_counts()
            .reset_index()
            .rename(columns={"topic_name": "Topik", "count": "Jumlah"})
        )
        topic_count["Persen"] = (topic_count["Jumlah"] / topic_count["Jumlah"].sum() * 100).round(1)

        # assign colors per topic name
        color_map = {v: TOPIC_COLORS[k] for k, v in TOPIC_LABELS.items()}

        fig_bar = px.bar(
            topic_count,
            x="Jumlah",
            y="Topik",
            orientation="h",
            color="Topik",
            color_discrete_map=color_map,
            text=topic_count.apply(lambda r: f"{r['Jumlah']} ({r['Persen']}%)", axis=1),
            title="Jumlah Dokumen per Topik",
        )
        fig_bar.update_traces(textposition="outside", textfont_size=11)
        fig_bar.update_layout(showlegend=False, yaxis=dict(categoryorder="total ascending"))
        fig_bar = style_chart(fig_bar, height=400)
        st.plotly_chart(fig_bar, width="stretch")

    # --- Donut chart ---
    with row1_r:
        fig_pie = go.Figure(go.Pie(
            labels=topic_count["Topik"],
            values=topic_count["Jumlah"],
            hole=0.55,
            marker=dict(colors=[color_map.get(t, "#58a6ff") for t in topic_count["Topik"]]),
            textinfo="percent",
            hovertemplate="<b>%{label}</b><br>%{value} dokumen<br>%{percent}<extra></extra>",
        ))
        fig_pie.add_annotation(
            text=f"<b>{len(dff)}</b><br><span style='font-size:10px'>Tanggapan</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="#e6edf3"),
        )
        fig_pie.update_layout(title="Proporsi Topik", showlegend=True, **CHART_LAYOUT, height=400)
        st.plotly_chart(fig_pie, width="stretch")

    # --- Treemap ---
    st.markdown('<p class="section-header">🗂️ Treemap Distribusi</p>', unsafe_allow_html=True)

    tree_df = dff.groupby(["topic_name"]).size().reset_index(name="count")
    tree_df["icon"] = tree_df["topic_name"].map(
        {v: TOPIC_ICONS[k] for k, v in TOPIC_LABELS.items()}
    )
    tree_df["label"] = tree_df.apply(lambda r: f"{r['icon']} {r['topic_name']}", axis=1)

    fig_tree = px.treemap(
        tree_df,
        path=["label"],
        values="count",
        color="count",
        color_continuous_scale=[[0, "#0d1117"], [0.3, "#0f3460"], [1, "#388bfd"]],
        title="Proporsi Relatif Tiap Topik",
    )
    fig_tree.update_traces(
        textfont=dict(size=14, family="Plus Jakarta Sans"),
        texttemplate="<b>%{label}</b><br>%{value} dok.",
        hovertemplate="<b>%{label}</b><br>%{value} dokumen<extra></extra>",
    )
    fig_tree.update_layout(**CHART_LAYOUT, height=380, coloraxis_showscale=False)
    st.plotly_chart(fig_tree, width="stretch")


# ═══════════════════════════════════════════════════════════
# TAB 2 – DEMOGRAFI
# ═══════════════════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-header">👥 Distribusi Topik berdasarkan Demografi</p>', unsafe_allow_html=True)

    d2_l, d2_r = st.columns(2)

    # --- By Umur (stacked bar) ---
    with d2_l:
        umur_cross = (
            dff.groupby(["Umur", "topic_name"])
            .size()
            .reset_index(name="count")
        )
        fig_umur = px.bar(
            umur_cross,
            x="Umur",
            y="count",
            color="topic_name",
            color_discrete_map=color_map,
            barmode="stack",
            title="Topik × Kelompok Umur",
            labels={"count": "Jumlah", "topic_name": "Topik"},
        )
        fig_umur = style_chart(fig_umur, height=400)
        st.plotly_chart(fig_umur, width="stretch")

    # --- By Gender (grouped bar) ---
    with d2_r:
        gender_cross = (
            dff.groupby(["Jenis Kelamin", "topic_name"])
            .size()
            .reset_index(name="count")
        )
        fig_gender = px.bar(
            gender_cross,
            x="topic_name",
            y="count",
            color="Jenis Kelamin",
            color_discrete_sequence=["#388bfd", "#f0883e"],
            barmode="group",
            title="Topik × Jenis Kelamin",
            labels={"count": "Jumlah", "topic_name": "Topik"},
        )
        fig_gender.update_layout(xaxis_tickangle=-30)
        fig_gender = style_chart(fig_gender, height=400)
        st.plotly_chart(fig_gender, width="stretch")

    # --- Heatmap Umur x Topik ---
    st.markdown('<p class="section-header">🔥 Heatmap Intensitas: Umur × Topik</p>', unsafe_allow_html=True)

    heat = dff.groupby(["Umur", "topic_name"]).size().unstack(fill_value=0)
    fig_heat = go.Figure(go.Heatmap(
        z=heat.values,
        x=heat.columns.tolist(),
        y=heat.index.tolist(),
        colorscale=[[0, "#0d1117"], [0.3, "#0f3460"], [0.7, "#388bfd"], [1, "#79c0ff"]],
        text=heat.values,
        texttemplate="%{text}",
        hovertemplate="<b>%{y}</b><br>%{x}<br>%{z} dokumen<extra></extra>",
        showscale=True,
    ))
    fig_heat.update_layout(
        title="Jumlah Dokumen per Sel (Umur × Topik)",
        xaxis=dict(tickangle=-20),
        **CHART_LAYOUT,
        height=300,
    )
    st.plotly_chart(fig_heat, width="stretch")

    # --- Normalized by gender (100% stacked) ---
    st.markdown('<p class="section-header">⚖️ Proporsi Topik per Jenis Kelamin (100%)</p>', unsafe_allow_html=True)
    gender_norm = (
        dff.groupby(["Jenis Kelamin", "topic_name"])
        .size()
        .reset_index(name="count")
    )
    totals = gender_norm.groupby("Jenis Kelamin")["count"].transform("sum")
    gender_norm["pct"] = (gender_norm["count"] / totals * 100).round(1)

    fig_norm = px.bar(
        gender_norm,
        x="pct",
        y="Jenis Kelamin",
        color="topic_name",
        color_discrete_map=color_map,
        orientation="h",
        text=gender_norm["pct"].apply(lambda x: f"{x:.0f}%"),
        title="Distribusi 100% Topik per Gender",
        labels={"pct": "Persentase (%)", "topic_name": "Topik"},
    )
    fig_norm.update_traces(textposition="inside", textfont_size=10)
    fig_norm = style_chart(fig_norm, height=280)
    st.plotly_chart(fig_norm, width="stretch")

    # ──────────────────────────────────────────────────────
    # TAMBAHAN 1 — Jenis Kelamin × Wilayah
    # ──────────────────────────────────────────────────────
    st.markdown('<p class="section-header">🚻 Distribusi Jenis Kelamin per Wilayah</p>', unsafe_allow_html=True)

    gw_l, gw_r = st.columns(2)

    with gw_l:
        gw_data = dff.groupby(["Wilayah", "Jenis Kelamin"]).size().reset_index(name="count")
        fig_gw_bar = px.bar(
            gw_data,
            x="Wilayah",
            y="count",
            color="Jenis Kelamin",
            color_discrete_sequence=["#388bfd", "#f0883e"],
            barmode="group",
            text="count",
            title="Jumlah Responden: Gender × Wilayah",
            labels={"count": "Jumlah", "Jenis Kelamin": "Gender"},
        )
        fig_gw_bar.update_traces(textposition="outside", textfont_size=11)
        fig_gw_bar = style_chart(fig_gw_bar, height=360)
        st.plotly_chart(fig_gw_bar, width="stretch")

    with gw_r:
        # 100% stacked — proporsi gender dalam tiap wilayah
        gw_tot = gw_data.groupby("Wilayah")["count"].transform("sum")
        gw_data["pct"] = (gw_data["count"] / gw_tot * 100).round(1)
        fig_gw_pct = px.bar(
            gw_data,
            x="pct",
            y="Wilayah",
            color="Jenis Kelamin",
            color_discrete_sequence=["#388bfd", "#f0883e"],
            orientation="h",
            text=gw_data["pct"].apply(lambda x: f"{x:.0f}%"),
            title="Proporsi 100% Gender per Wilayah",
            labels={"pct": "Persentase (%)", "Jenis Kelamin": "Gender"},
        )
        fig_gw_pct.update_traces(textposition="inside", textfont_size=11)
        fig_gw_pct = style_chart(fig_gw_pct, height=360)
        st.plotly_chart(fig_gw_pct, width="stretch")

    # Donut per wilayah (3 kolom)
    wil_list = sorted(dff["Wilayah"].unique())
    donut_cols = st.columns(len(wil_list))
    for col, wil in zip(donut_cols, wil_list):
        wil_df = dff[dff["Wilayah"] == wil]["Jenis Kelamin"].value_counts().reset_index()
        wil_df.columns = ["Gender", "count"]
        fig_d = go.Figure(go.Pie(
            labels=wil_df["Gender"],
            values=wil_df["count"],
            hole=0.55,
            marker=dict(colors=["#388bfd", "#f0883e"]),
            textinfo="label+percent",
            hovertemplate="<b>%{label}</b><br>%{value} orang (%{percent})<extra></extra>",
        ))
        total_wil = wil_df["count"].sum()
        fig_d.add_annotation(
            text=f"<b>{total_wil}</b><br><span style='font-size:9px'>orang</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="#e6edf3"),
        )
        layout_d = {**CHART_LAYOUT, "margin": dict(t=40, b=5, l=5, r=5)}
        fig_d.update_layout(
            title=dict(text=wil, font=dict(size=13)),
            showlegend=False,
            height=280,
            **layout_d,
        )
        col.plotly_chart(fig_d, width="stretch")

    # ──────────────────────────────────────────────────────
    # TAMBAHAN 2 — Jenis Kelamin × Umur
    # ──────────────────────────────────────────────────────
    st.markdown('<p class="section-header">🎂 Distribusi Jenis Kelamin per Kelompok Umur</p>', unsafe_allow_html=True)

    gu_l, gu_r = st.columns(2)

    with gu_l:
        gu_data = dff.groupby(["Umur", "Jenis Kelamin"]).size().reset_index(name="count")
        fig_gu_bar = px.bar(
            gu_data,
            x="Umur",
            y="count",
            color="Jenis Kelamin",
            color_discrete_sequence=["#388bfd", "#f0883e"],
            barmode="group",
            text="count",
            title="Jumlah Responden: Gender × Umur",
            labels={"count": "Jumlah", "Jenis Kelamin": "Gender"},
        )
        fig_gu_bar.update_traces(textposition="outside", textfont_size=11)
        fig_gu_bar.update_layout(xaxis_tickangle=-15)
        fig_gu_bar = style_chart(fig_gu_bar, height=360)
        st.plotly_chart(fig_gu_bar, width="stretch")

    with gu_r:
        gu_tot = gu_data.groupby("Umur")["count"].transform("sum")
        gu_data["pct"] = (gu_data["count"] / gu_tot * 100).round(1)
        fig_gu_pct = px.bar(
            gu_data,
            x="pct",
            y="Umur",
            color="Jenis Kelamin",
            color_discrete_sequence=["#388bfd", "#f0883e"],
            orientation="h",
            text=gu_data["pct"].apply(lambda x: f"{x:.0f}%"),
            title="Proporsi 100% Gender per Kelompok Umur",
            labels={"pct": "Persentase (%)", "Jenis Kelamin": "Gender"},
        )
        fig_gu_pct.update_traces(textposition="inside", textfont_size=11)
        fig_gu_pct = style_chart(fig_gu_pct, height=360)
        st.plotly_chart(fig_gu_pct, width="stretch")

    # Heatmap Gender x Umur (nilai absolut + persen)
    heat_gu = dff.groupby(["Jenis Kelamin", "Umur"]).size().unstack(fill_value=0)
    heat_gu_pct = heat_gu.div(heat_gu.sum().sum()) * 100

    fig_heat_gu = go.Figure(go.Heatmap(
        z=heat_gu.values,
        x=heat_gu.columns.tolist(),
        y=heat_gu.index.tolist(),
        colorscale=[[0, "#0d1117"], [0.4, "#6e3fa3"], [1, "#bc8cff"]],
        text=[[f"{v}<br>({heat_gu_pct.iloc[i,j]:.1f}%)"
               for j, v in enumerate(row)]
              for i, row in enumerate(heat_gu.values)],
        texttemplate="%{text}",
        hovertemplate="<b>%{y}</b> · %{x}<br>%{z} orang<extra></extra>",
        showscale=True,
    ))
    fig_heat_gu.update_layout(
        title="Heatmap: Jenis Kelamin × Kelompok Umur (n & %)",
        **CHART_LAYOUT,
        height=260,
    )
    st.plotly_chart(fig_heat_gu, width="stretch")

    # ──────────────────────────────────────────────────────
    # TAMBAHAN 3 — Umur × Wilayah
    # ──────────────────────────────────────────────────────
    st.markdown('<p class="section-header">📍 Distribusi Umur per Wilayah</p>', unsafe_allow_html=True)

    uw_l, uw_r = st.columns(2)

    with uw_l:
        uw_data = dff.groupby(["Wilayah", "Umur"]).size().reset_index(name="count")
        umur_colors = {"8 sampai 11 tahun": "#3fb950",
                       "12 sampai 15 tahun": "#ffa657",
                       "15 sampai 17 tahun": "#ff7b72"}
        fig_uw_bar = px.bar(
            uw_data,
            x="Wilayah",
            y="count",
            color="Umur",
            color_discrete_map=umur_colors,
            barmode="stack",
            text="count",
            title="Komposisi Umur per Wilayah (Stacked)",
            labels={"count": "Jumlah", "Umur": "Kelompok Umur"},
        )
        fig_uw_bar.update_traces(textposition="inside", textfont_size=10)
        fig_uw_bar = style_chart(fig_uw_bar, height=380)
        st.plotly_chart(fig_uw_bar, width="stretch")

    with uw_r:
        uw_tot = uw_data.groupby("Wilayah")["count"].transform("sum")
        uw_data["pct"] = (uw_data["count"] / uw_tot * 100).round(1)
        fig_uw_pct = px.bar(
            uw_data,
            x="pct",
            y="Wilayah",
            color="Umur",
            color_discrete_map=umur_colors,
            orientation="h",
            text=uw_data["pct"].apply(lambda x: f"{x:.0f}%"),
            title="Proporsi 100% Umur per Wilayah",
            labels={"pct": "Persentase (%)", "Umur": "Kelompok Umur"},
        )
        fig_uw_pct.update_traces(textposition="inside", textfont_size=10)
        fig_uw_pct = style_chart(fig_uw_pct, height=380)
        st.plotly_chart(fig_uw_pct, width="stretch")

    # Heatmap Umur x Wilayah
    heat_uw = dff.groupby(["Umur", "Wilayah"]).size().unstack(fill_value=0)
    heat_uw_pct = heat_uw.div(heat_uw.sum().sum()) * 100

    fig_heat_uw = go.Figure(go.Heatmap(
        z=heat_uw.values,
        x=heat_uw.columns.tolist(),
        y=heat_uw.index.tolist(),
        colorscale=[[0, "#0d1117"], [0.4, "#2d6a4f"], [1, "#3fb950"]],
        text=[[f"{v}<br>({heat_uw_pct.iloc[i,j]:.1f}%)"
               for j, v in enumerate(row)]
              for i, row in enumerate(heat_uw.values)],
        texttemplate="%{text}",
        hovertemplate="<b>%{y}</b> · %{x}<br>%{z} orang<extra></extra>",
        showscale=True,
    ))
    fig_heat_uw.update_layout(
        title="Heatmap: Kelompok Umur × Wilayah (n & %)",
        **CHART_LAYOUT,
        height=280,
    )
    st.plotly_chart(fig_heat_uw, width="stretch")

    # Grouped bar: semua kombinasi Umur × Wilayah × Gender (bubble chart)
    uwg_data = (
        dff.groupby(["Wilayah", "Umur", "Jenis Kelamin"])
        .size()
        .reset_index(name="count")
    )
    fig_bubble = px.scatter(
        uwg_data,
        x="Wilayah",
        y="Umur",
        size="count",
        color="Jenis Kelamin",
        color_discrete_sequence=["#388bfd", "#f0883e"],
        size_max=60,
        text="count",
        title="Bubble Chart: Wilayah × Umur × Gender (ukuran = jumlah)",
        labels={"count": "Jumlah"},
    )
    fig_bubble.update_traces(textposition="middle center", textfont=dict(color="white", size=10))
    fig_bubble.update_layout(**CHART_LAYOUT, height=380)
    st.plotly_chart(fig_bubble, width="stretch")


# ═══════════════════════════════════════════════════════════
# TAB 3 – WILAYAH
# ═══════════════════════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-header">🗺️ Analisis Topik per Wilayah</p>', unsafe_allow_html=True)

    w_l, w_r = st.columns(2)

    with w_l:
        wil_cross = (
            dff.groupby(["Wilayah", "topic_name"])
            .size()
            .reset_index(name="count")
        )
        fig_wil = px.bar(
            wil_cross,
            x="Wilayah",
            y="count",
            color="topic_name",
            color_discrete_map=color_map,
            barmode="stack",
            title="Topik × Wilayah (Stacked)",
            labels={"count": "Jumlah", "topic_name": "Topik"},
        )
        fig_wil = style_chart(fig_wil, height=400)
        st.plotly_chart(fig_wil, width="stretch")

    with w_r:
        # Dominant topic per wilayah
        dom = (
            dff.groupby(["Wilayah", "topic_name"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
            .drop_duplicates("Wilayah")
        )
        fig_dom = px.bar(
            dom,
            x="Wilayah",
            y="count",
            color="topic_name",
            color_discrete_map=color_map,
            text="topic_name",
            title="Topik Dominan per Wilayah",
            labels={"count": "Jumlah", "topic_name": "Topik Dominan"},
        )
        fig_dom.update_traces(textposition="outside", textfont_size=10)
        fig_dom = style_chart(fig_dom, height=400)
        st.plotly_chart(fig_dom, width="stretch")

    # Heatmap Wilayah x Topik
    st.markdown('<p class="section-header">🔥 Heatmap Intensitas: Wilayah × Topik</p>', unsafe_allow_html=True)
    heat_w = dff.groupby(["Wilayah", "topic_name"]).size().unstack(fill_value=0)
    fig_hw = go.Figure(go.Heatmap(
        z=heat_w.values,
        x=heat_w.columns.tolist(),
        y=heat_w.index.tolist(),
        colorscale=[[0, "#0d1117"], [0.4, "#2d6a4f"], [1, "#3fb950"]],
        text=heat_w.values,
        texttemplate="%{text}",
        hovertemplate="<b>%{y}</b><br>%{x}<br>%{z} dokumen<extra></extra>",
    ))
    fig_hw.update_layout(
        title="Jumlah Dokumen per Sel (Wilayah × Topik)",
        xaxis=dict(tickangle=-20),
        **CHART_LAYOUT,
        height=280,
    )
    st.plotly_chart(fig_hw, width="stretch")

    # Grouped bar: Wilayah × Umur × Topik
    st.markdown('<p class="section-header">🧩 Komposisi Wilayah × Umur × Topik</p>', unsafe_allow_html=True)
    wu_cross = (
        dff.groupby(["Wilayah", "Umur", "topic_name"])
        .size()
        .reset_index(name="count")
    )
    fig_wu = px.sunburst(
        wu_cross,
        path=["Wilayah", "topic_name", "Umur"],
        values="count",
        color="Wilayah",
        color_discrete_sequence=["#388bfd", "#3fb950", "#f0883e"],
        title="Sunburst: Wilayah → Topik → Umur",
    )
    fig_wu.update_layout(**CHART_LAYOUT, height=500)
    st.plotly_chart(fig_wu, width="stretch")


# ═══════════════════════════════════════════════════════════
# TAB 4 – EKSPLORASI TOPIK
# ═══════════════════════════════════════════════════════════
with tab4:
    st.markdown('<p class="section-header">🔍 Eksplorasi Detail per Topik</p>', unsafe_allow_html=True)

    sel_one_topic = st.selectbox(
        "Pilih topik untuk dieksplorasi:",
        options=list(TOPIC_LABELS.values()),
        index=0,
    )

    topic_id = {v: k for k, v in TOPIC_LABELS.items()}[sel_one_topic]
    topic_df = dff[dff["dominant_topic_id"] == topic_id]

    # Stats row
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Total Dokumen", len(topic_df))
    e2.metric(
        "Terbanyak: Umur",
        topic_df["Umur"].value_counts().idxmax() if len(topic_df) > 0 else "-",
    )
    e3.metric(
        "Terbanyak: Wilayah",
        topic_df["Wilayah"].value_counts().idxmax() if len(topic_df) > 0 else "-",
    )
    e4.metric(
        "Terbanyak: Gender",
        topic_df["Jenis Kelamin"].value_counts().idxmax() if len(topic_df) > 0 else "-",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    ec_l, ec_r = st.columns([3, 2])

    # --- Word frequency bar ---
    with ec_l:
        stop = {"yang", "dan", "di", "ke", "dari", "ini", "itu", "dengan", "untuk",
                "ada", "pada", "adalah", "saya", "aku", "kamu", "kami", "mereka",
                "tidak", "bisa", "karena", "karna", "juga", "sudah", "akan", "telah",
                "atau", "oleh", "dalam", "saat", "setelah", "lalu", "pun", "jadi",
                "maka", "tapi", "jika", "bila", "namun", "kalau", "agar", "seperti",
                "semua", "sangat", "sekali", "banyak", "ketika", "merasa", "lagi",
                "lebih", "banget", "mau", "harus", "dapat"}

        all_words = []
        for text in topic_df["Tanggapan"].dropna():
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            all_words.extend([w for w in words if w not in stop])

        word_freq = Counter(all_words).most_common(20)
        if word_freq:
            wf_df = pd.DataFrame(word_freq, columns=["Kata", "Frekuensi"])
            fig_wf = px.bar(
                wf_df,
                x="Frekuensi",
                y="Kata",
                orientation="h",
                color="Frekuensi",
                color_continuous_scale=[[0, "#0f3460"], [1, "#388bfd"]],
                title=f"20 Kata Terpopuler — {sel_one_topic}",
            )
            fig_wf.update_layout(
                coloraxis_showscale=False,
                yaxis=dict(categoryorder="total ascending"),
                **CHART_LAYOUT,
                height=420,
            )
            st.plotly_chart(fig_wf, width="stretch")

    # --- Breakdown pie Umur, Gender, Wilayah ---
    with ec_r:
        breakdown_choice = st.radio(
            "Pilah berdasarkan:",
            ["Umur", "Jenis Kelamin", "Wilayah"],
            horizontal=True,
        )
        bd_data = topic_df[breakdown_choice].value_counts().reset_index()
        bd_data.columns = [breakdown_choice, "count"]

        palette = {
            "Umur": ["#388bfd", "#3fb950", "#f0883e"],
            "Jenis Kelamin": ["#388bfd", "#f0883e"],
            "Wilayah": ["#388bfd", "#3fb950", "#f0883e"],
        }[breakdown_choice]

        fig_bd = go.Figure(go.Pie(
            labels=bd_data[breakdown_choice],
            values=bd_data["count"],
            hole=0.5,
            marker=dict(colors=palette),
            textinfo="label+percent",
        ))
        fig_bd.add_annotation(
            text=f"<b>{len(topic_df)}</b><br>dok.",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=15, color="#e6edf3"),
        )
        fig_bd.update_layout(
            title=f"Sebaran {breakdown_choice}",
            showlegend=False,
            **CHART_LAYOUT,
            height=420,
        )
        st.plotly_chart(fig_bd, width="stretch")

    # --- Keyword tags ---
    kw_list = TOPIC_KEYWORDS.get(topic_id, [])
    if kw_list:
        st.markdown('<p class="section-header">🏷️ Kata Kunci Topik</p>', unsafe_allow_html=True)
        pills_html = "".join(
            f'<span class="topic-pill" style="background:{TOPIC_COLORS[topic_id]}22;'
            f'border:1px solid {TOPIC_COLORS[topic_id]};color:{TOPIC_COLORS[topic_id]};">'
            f'{kw}</span>'
            for kw in kw_list
        )
        st.markdown(pills_html, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    # --- Sample quotes ---
    st.markdown('<p class="section-header">💬 Contoh Tanggapan (acak)</p>', unsafe_allow_html=True)
    samples = topic_df["Tanggapan"].dropna().sample(
        min(8, len(topic_df)), random_state=42
    ).tolist()
    for s in samples:
        st.markdown(f'<div class="quote-card">{s}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# TAB 5 – TABEL DATA
# ═══════════════════════════════════════════════════════════
with tab5:
    st.markdown('<p class="section-header">📝 Data Lengkap (Filtered)</p>', unsafe_allow_html=True)

    show_cols = ["Umur", "Jenis Kelamin", "Wilayah", "dominant_topic_id", "topic_name", "Tanggapan"]
    display_df = dff[show_cols].copy()
    display_df.columns = ["Umur", "Jenis Kelamin", "Wilayah", "ID Topik", "Nama Topik", "Tanggapan"]

    # Search
    search = st.text_input("🔎 Cari kata dalam tanggapan:", placeholder="ketik kata kunci...")
    if search:
        display_df = display_df[
            display_df["Tanggapan"].str.contains(search, case=False, na=False)
        ]

    st.markdown(f"**{len(display_df)}** baris ditampilkan")
    st.dataframe(
        display_df,
        width="stretch",
        height=480,
        column_config={
            "ID Topik": st.column_config.NumberColumn(width="small"),
            "Nama Topik": st.column_config.TextColumn(width="medium"),
            "Tanggapan": st.column_config.TextColumn(width="large"),
        },
    )

    # Download
    csv_out = display_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="⬇️ Unduh CSV",
        data=csv_out,
        file_name="hasil_filtered_topik.csv",
        mime="text/csv",
    )

    # --- Summary table ---
    st.markdown("---")
    st.markdown('<p class="section-header">📋 Ringkasan per Topik</p>', unsafe_allow_html=True)

    summary = (
        dff.groupby("topic_name")
        .agg(
            Jumlah=("Tanggapan", "count"),
            Umur_Terbanyak=("Umur", lambda x: x.value_counts().idxmax()),
            Wilayah_Terbanyak=("Wilayah", lambda x: x.value_counts().idxmax()),
            Gender_Terbanyak=("Jenis Kelamin", lambda x: x.value_counts().idxmax()),
        )
        .reset_index()
        .rename(columns={"topic_name": "Topik"})
        .sort_values("Jumlah", ascending=False)
    )
    summary["Proporsi"] = (summary["Jumlah"] / summary["Jumlah"].sum() * 100).round(1).astype(str) + "%"
    st.dataframe(summary, width="stretch", hide_index=True)
