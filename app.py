import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG PAGE
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OPCVM Performance & Optimisation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS CUSTOM
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Palette CDG-inspirée */
    :root {
        --cdg-green:  #005537;
        --cdg-green2: #007850;
        --cdg-gold:   #C8952A;
        --cdg-light:  #f0f5f2;
        --bg-dark:    #0D1B2A;
        --card-bg:    #1A2E42;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #005537 0%, #007850 100%);
    }
    [data-testid="stSidebar"] * { color: white !important; }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stMultiSelect label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stRadio label { color: #d4edda !important; font-weight: 500; }

    /* Main background */
    .main { background-color: #f8f9fa; }

    /* KPI Cards */
    .kpi-card {
        background: white;
        border-left: 4px solid #005537;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 6px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .kpi-label { font-size: 0.75rem; color: #6c757d; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
    .kpi-value { font-size: 1.6rem; font-weight: 700; color: #005537; margin-top: 4px; }
    .kpi-card.positive .kpi-value { color: #007850; }
    .kpi-card.negative .kpi-value { color: #dc3545; }
    .kpi-card.neutral  .kpi-value { color: #C8952A; }

    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #005537, #007850);
        color: white !important;
        padding: 10px 18px;
        border-radius: 6px;
        font-size: 1rem;
        font-weight: 700;
        margin: 18px 0 12px 0;
        letter-spacing: 0.03em;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: #e9ecef;
        border-radius: 6px 6px 0 0;
        padding: 8px 20px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: #005537 !important;
        color: white !important;
    }

    /* Boutons */
    .stButton > button {
        background: #005537;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 8px 24px;
        font-weight: 600;
        transition: background 0.2s;
    }
    .stButton > button:hover { background: #007850; }

    /* DataFrame */
    [data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }

    /* Footer badge */
    .badge {
        display: inline-block;
        background: #005537;
        color: white;
        border-radius: 4px;
        padding: 2px 10px;
        font-size: 0.72rem;
        font-weight: 600;
        margin: 2px;
    }
    .badge.gold { background: #C8952A; }
    .badge.red  { background: #dc3545; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────────────────────
RF_ANNUAL = 0.0225          # Taux sans risque Bank Al-Maghrib
TRADING_DAYS_ANNUAL = 247   # Convention BVC
WEEKS_ANNUAL = 52

# ─────────────────────────────────────────────────────────────────────────────
# FONCTIONS UTILITAIRES
# ─────────────────────────────────────────────────────────────────────────────

def parse_uploaded_file(uploaded_file):
    """Lit CSV ou Excel avec gestion robuste."""
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".csv"):
            for sep in [";", ",", "\t"]:
                try:
                    df = pd.read_csv(uploaded_file, sep=sep, thousands=" ", decimal=",")
                    if df.shape[1] >= 2:
                        uploaded_file.seek(0)
                        return pd.read_csv(uploaded_file, sep=sep, thousands=" ", decimal=",")
                except:
                    uploaded_file.seek(0)
            return None
        else:
            df = pd.read_excel(uploaded_file, index_col=None)
            return df
    except Exception as e:
        st.error(f"Erreur lecture fichier : {e}")
        return None


def detect_date_col(df):
    """Détecte automatiquement la colonne date."""
    for col in df.columns:
        try:
            parsed = pd.to_datetime(df[col], dayfirst=True, errors="coerce")
            if parsed.notna().mean() > 0.7:
                return col
        except:
            continue
    return None


def compute_returns(vl_series, freq="daily"):
    """Rendements à partir des VL."""
    rets = vl_series.pct_change().dropna()
    return rets


def annualize_factor(freq):
    return TRADING_DAYS_ANNUAL if freq == "Journalières" else WEEKS_ANNUAL


def compute_ratios(returns, freq="Journalières", rf=RF_ANNUAL):
    """Calcule tous les ratios de performance."""
    n = annualize_factor(freq)
    rf_period = rf / n

    mean_ret  = returns.mean()
    std_ret   = returns.std()
    cum_ret   = (1 + returns).prod() - 1

    perf_ann  = (1 + mean_ret) ** n - 1
    vol_ann   = std_ret * np.sqrt(n)

    sharpe    = (perf_ann - rf) / vol_ann if vol_ann > 0 else np.nan

    # Sortino
    downside  = returns[returns < rf_period].std() * np.sqrt(n)
    sortino   = (perf_ann - rf) / downside if downside > 0 else np.nan

    # Max Drawdown
    cum_vl    = (1 + returns).cumprod()
    rolling_max = cum_vl.cummax()
    drawdown  = (cum_vl - rolling_max) / rolling_max
    max_dd    = drawdown.min()

    # Calmar
    calmar    = perf_ann / abs(max_dd) if max_dd != 0 else np.nan

    # VaR historique 95% et 99%
    var_95    = np.percentile(returns, 5)
    var_99    = np.percentile(returns, 1)

    # CVaR / Expected Shortfall
    cvar_95   = returns[returns <= var_95].mean()
    cvar_99   = returns[returns <= var_99].mean()

    return {
        "Perf. Cumulée":     f"{cum_ret*100:.2f}%",
        "Perf. Annualisée":  f"{perf_ann*100:.2f}%",
        "Volatilité Ann.":   f"{vol_ann*100:.2f}%",
        "Sharpe":            f"{sharpe:.4f}" if not np.isnan(sharpe) else "N/A",
        "Sortino":           f"{sortino:.4f}" if not np.isnan(sortino) else "N/A",
        "Calmar":            f"{calmar:.4f}" if not np.isnan(calmar) else "N/A",
        "Max Drawdown":      f"{max_dd*100:.2f}%",
        "VaR 95% (1j)":      f"{var_95*100:.4f}%",
        "VaR 99% (1j)":      f"{var_99*100:.4f}%",
        "CVaR 95%":          f"{cvar_95*100:.4f}%",
        "CVaR 99%":          f"{cvar_99*100:.4f}%",
        "_perf_ann":         perf_ann,
        "_vol_ann":          vol_ann,
        "_sharpe":           sharpe,
        "_max_dd":           max_dd,
    }


def portfolio_performance(weights, returns_df, freq):
    """Performance d'un portefeuille pondéré."""
    n = annualize_factor(freq)
    port_rets = returns_df @ weights
    mean_ret  = port_rets.mean()
    perf_ann  = (1 + mean_ret) ** n - 1
    cov       = returns_df.cov()
    vol_ann   = np.sqrt(weights @ cov.values @ weights) * np.sqrt(n)
    sharpe    = (perf_ann - RF_ANNUAL) / vol_ann if vol_ann > 0 else 0
    return perf_ann, vol_ann, sharpe


def efficient_frontier(returns_df, freq, n_points=120):
    """Génère la frontière efficiente."""
    n_assets  = returns_df.shape[1]
    n_ann     = annualize_factor(freq)
    mean_rets = (1 + returns_df.mean()) ** n_ann - 1
    cov       = returns_df.cov().values * n_ann

    bounds      = tuple((0.0, 1.0) for _ in range(n_assets))
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    # Plage de rendements cibles
    ret_min  = mean_rets.min()
    ret_max  = mean_rets.max()
    targets  = np.linspace(ret_min, ret_max, n_points)

    frontier_vols   = []
    frontier_rets   = []
    frontier_sharpe = []
    frontier_weights= []

    for target in targets:
        cons = constraints + [{"type": "eq", "fun": lambda w, t=target: portfolio_performance(w, returns_df, freq)[0] - t}]
        w0   = np.ones(n_assets) / n_assets
        res  = minimize(lambda w: portfolio_performance(w, returns_df, freq)[1],
                        w0, method="SLSQP", bounds=bounds, constraints=cons,
                        options={"maxiter": 500, "ftol": 1e-9})
        if res.success:
            p, v, s = portfolio_performance(res.x, returns_df, freq)
            frontier_rets.append(p)
            frontier_vols.append(v)
            frontier_sharpe.append(s)
            frontier_weights.append(res.x)

    return frontier_rets, frontier_vols, frontier_sharpe, frontier_weights


def optimize_max_sharpe(returns_df, freq, bounds_min=0.0, bounds_max=1.0):
    n = returns_df.shape[1]
    bounds      = tuple((bounds_min, bounds_max) for _ in range(n))
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    w0          = np.ones(n) / n
    res = minimize(lambda w: -portfolio_performance(w, returns_df, freq)[2],
                   w0, method="SLSQP", bounds=bounds, constraints=constraints,
                   options={"maxiter": 500})
    return res.x if res.success else w0


def optimize_min_variance(returns_df, freq, bounds_min=0.0, bounds_max=1.0):
    n = returns_df.shape[1]
    n_ann = annualize_factor(freq)
    cov   = returns_df.cov().values * n_ann
    bounds      = tuple((bounds_min, bounds_max) for _ in range(n))
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    w0          = np.ones(n) / n
    res = minimize(lambda w: w @ cov @ w,
                   w0, method="SLSQP", bounds=bounds, constraints=constraints,
                   options={"maxiter": 500})
    return res.x if res.success else w0


def optimize_min_cvar(returns_df, freq, alpha=0.05, bounds_min=0.0, bounds_max=1.0):
    n = returns_df.shape[1]
    bounds      = tuple((bounds_min, bounds_max) for _ in range(n))
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    w0          = np.ones(n) / n

    def neg_cvar(w):
        port_r = returns_df @ w
        var    = np.percentile(port_r, alpha * 100)
        return port_r[port_r <= var].mean()

    res = minimize(neg_cvar, w0, method="SLSQP", bounds=bounds, constraints=constraints,
                   options={"maxiter": 500})
    return res.x if res.success else w0


def kpi_card(label, value, cls=""):
    st.markdown(f"""
    <div class="kpi-card {cls}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
    </div>""", unsafe_allow_html=True)


def section(title, icon="📌"):
    st.markdown(f'<div class="section-header">{icon} &nbsp; {title}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 OPCVM Analytics")
    st.markdown("---")

    st.markdown("### 📁 Import des données")
    uploaded = st.file_uploader(
        "Fichier VL (CSV ou Excel)",
        type=["csv", "xlsx", "xls"],
        help="Format attendu : 1 colonne Date + N colonnes VL (une par fonds)"
    )

    st.markdown("### ⚙️ Paramètres globaux")
    freq = st.radio("Fréquence des données", ["Journalières", "Hebdomadaires"], index=0)
    rf_input = st.number_input("Taux sans risque annuel (%)", value=2.25, step=0.05, format="%.2f")
    RF_ANNUAL = rf_input / 100

    st.markdown("---")
    st.markdown("### 🔧 Optimisation — Contraintes")
    w_min = st.slider("Poids minimum (%)", 0, 20, 0, step=1) / 100
    w_max = st.slider("Poids maximum (%)", 20, 100, 100, step=5) / 100
    n_sim  = st.slider("Simulations Monte Carlo", 500, 5000, 2000, step=500)

    st.markdown("---")
    st.markdown("""
    <small style='color:#a8d5b5;'>
    CDG Capital — Direction Gestion Risques Financiers<br>
    RB · PFE 2024-2025
    </small>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DEMO DATA
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_demo_data():
    np.random.seed(42)
    dates = pd.bdate_range("2022-01-03", "2024-12-31", freq="B")
    funds = {
        "OMLT_A":  (0.00030, 0.0008),
        "OMLT_B":  (0.00028, 0.0009),
        "Oblig_CT":(0.00012, 0.0003),
        "Divers_A":(0.00025, 0.0020),
        "Divers_B":(0.00035, 0.0025),
        "Monetaire":(0.00008, 0.0001),
    }
    vl = {"Date": dates}
    start_vl = {"OMLT_A": 1000, "OMLT_B": 950, "Oblig_CT": 1200,
                "Divers_A": 800, "Divers_B": 1100, "Monetaire": 500}
    for name, (mu, sigma) in funds.items():
        rets   = np.random.normal(mu, sigma, len(dates))
        prices = [start_vl[name]]
        for r in rets[1:]:
            prices.append(prices[-1] * (1 + r))
        vl[name] = prices
    return pd.DataFrame(vl)


# ─────────────────────────────────────────────────────────────────────────────
# CHARGEMENT DONNÉES
# ─────────────────────────────────────────────────────────────────────────────
df_raw = None
date_col = None
fund_cols = []

if uploaded is None:
    st.info("💡 Aucun fichier importé — affichage des données de démo (6 fonds fictifs)")
    df_raw = load_demo_data()
    date_col = "Date"
    fund_cols = [c for c in df_raw.columns if c != "Date"]
else:
    df_raw = parse_uploaded_file(uploaded)
    if df_raw is not None:
        date_col = detect_date_col(df_raw)
        if date_col:
            df_raw[date_col] = pd.to_datetime(df_raw[date_col], dayfirst=True, errors="coerce")
            df_raw = df_raw.dropna(subset=[date_col]).sort_values(date_col)
            fund_cols = [c for c in df_raw.columns if c != date_col]
        else:
            st.error("❌ Impossible de détecter la colonne Date.")


if df_raw is None or not fund_cols:
    st.stop()

# Nettoyer les VL
for c in fund_cols:
    df_raw[c] = pd.to_numeric(df_raw[c], errors="coerce")

df_raw = df_raw.dropna(subset=fund_cols, how="all")
df_raw = df_raw.set_index(date_col)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Évolution des VL",
    "📊 Ratios de Performance",
    "🏗️ Construction du Portefeuille",
    "🎯 Optimisation & Frontière Efficiente"
])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — ÉVOLUTION DES VL
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    section("Visualisation des Valeurs Liquidatives", "📈")

    c1, c2 = st.columns([2, 1])
    with c1:
        sel_funds_vl = st.multiselect(
            "Sélectionner les fonds à afficher",
            fund_cols, default=fund_cols[:min(4, len(fund_cols))]
        )
    with c2:
        norm_vl = st.checkbox("Normaliser à 100 (base 100)", value=True)

    if sel_funds_vl:
        plot_df = df_raw[sel_funds_vl].copy().dropna()
        if norm_vl:
            plot_df = plot_df / plot_df.iloc[0] * 100

        fig = go.Figure()
        colors = px.colors.qualitative.Set2
        for i, col in enumerate(sel_funds_vl):
            fig.add_trace(go.Scatter(
                x=plot_df.index, y=plot_df[col],
                name=col, mode="lines",
                line=dict(color=colors[i % len(colors)], width=2),
                hovertemplate=f"<b>{col}</b><br>Date: %{{x|%d/%m/%Y}}<br>VL: %{{y:.4f}}<extra></extra>"
            ))
        fig.update_layout(
            title="Évolution des Valeurs Liquidatives",
            xaxis_title="Date", yaxis_title="VL (base 100)" if norm_vl else "VL",
            template="plotly_white", height=420,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Rendements journaliers
        section("Distribution des Rendements", "📊")
        rets_df = plot_df.pct_change().dropna() if not norm_vl else df_raw[sel_funds_vl].pct_change().dropna()

        fig2 = go.Figure()
        for i, col in enumerate(sel_funds_vl):
            fig2.add_trace(go.Violin(
                y=rets_df[col] * 100, name=col,
                box_visible=True, meanline_visible=True,
                line_color=colors[i % len(colors)],
                fillcolor=colors[i % len(colors)], opacity=0.6
            ))
        fig2.update_layout(
            title="Distribution des Rendements (%)",
            yaxis_title="Rendement (%)",
            template="plotly_white", height=380,
            showlegend=True
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Corrélation
        section("Matrice de Corrélation", "🔗")
        corr = rets_df[sel_funds_vl].corr()
        fig3 = px.imshow(
            corr, text_auto=".2f", color_continuous_scale="RdYlGn",
            zmin=-1, zmax=1, aspect="auto",
            title="Corrélation des rendements"
        )
        fig3.update_layout(height=420, template="plotly_white")
        st.plotly_chart(fig3, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — RATIOS DE PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    section("Tableau des Ratios de Performance", "📊")

    sel_funds_r = st.multiselect(
        "Sélectionner les fonds à analyser",
        fund_cols, default=fund_cols, key="sel_r"
    )

    if sel_funds_r:
        results = {}
        rets_all = df_raw[sel_funds_r].pct_change().dropna()
        for f in sel_funds_r:
            ret_s = rets_all[f].dropna()
            if len(ret_s) > 5:
                results[f] = compute_ratios(ret_s, freq, RF_ANNUAL)

        # Tableau récap
        display_keys = [k for k in list(results[sel_funds_r[0]].keys()) if not k.startswith("_")]
        table_data = {k: [results[f].get(k, "N/A") for f in sel_funds_r] for k in display_keys}
        df_table = pd.DataFrame(table_data, index=sel_funds_r)
        st.dataframe(df_table.T, use_container_width=True, height=420)

        # Graphiques comparatifs
        section("Comparaison Visuelle", "📉")
        c1, c2 = st.columns(2)

        with c1:
            sharpe_vals = [results[f]["_sharpe"] for f in sel_funds_r if not np.isnan(results[f]["_sharpe"])]
            sharpe_names = [f for f in sel_funds_r if not np.isnan(results[f]["_sharpe"])]
            fig_s = go.Figure(go.Bar(
                x=sharpe_names, y=sharpe_vals,
                marker_color=["#005537" if v >= 0 else "#dc3545" for v in sharpe_vals],
                text=[f"{v:.3f}" for v in sharpe_vals], textposition="outside"
            ))
            fig_s.update_layout(title="Ratio de Sharpe", template="plotly_white", height=320,
                                 yaxis_title="Sharpe", showlegend=False)
            st.plotly_chart(fig_s, use_container_width=True)

        with c2:
            vol_vals  = [results[f]["_vol_ann"] * 100 for f in sel_funds_r]
            perf_vals = [results[f]["_perf_ann"] * 100 for f in sel_funds_r]
            fig_rv = go.Figure()
            fig_rv.add_trace(go.Scatter(
                x=vol_vals, y=perf_vals,
                mode="markers+text",
                text=sel_funds_r,
                textposition="top center",
                marker=dict(size=12, color="#005537", opacity=0.8),
            ))
            fig_rv.update_layout(
                title="Rendement vs Risque",
                xaxis_title="Volatilité Ann. (%)",
                yaxis_title="Performance Ann. (%)",
                template="plotly_white", height=320
            )
            st.plotly_chart(fig_rv, use_container_width=True)

        # Drawdown
        section("Drawdown Historique", "📉")
        sel_dd = st.selectbox("Fonds pour le drawdown", sel_funds_r)
        cum_vl   = (1 + rets_all[sel_dd]).cumprod()
        rolling_max = cum_vl.cummax()
        drawdown    = (cum_vl - rolling_max) / rolling_max * 100

        fig_dd = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                subplot_titles=("VL Cumulée", "Drawdown (%)"),
                                row_heights=[0.6, 0.4])
        fig_dd.add_trace(go.Scatter(x=cum_vl.index, y=cum_vl.values,
                                     name="VL cum.", line=dict(color="#005537")), row=1, col=1)
        fig_dd.add_trace(go.Scatter(x=drawdown.index, y=drawdown.values,
                                     name="Drawdown", fill="tozeroy",
                                     line=dict(color="#dc3545"),
                                     fillcolor="rgba(220,53,69,0.2)"), row=2, col=1)
        fig_dd.update_layout(template="plotly_white", height=420, showlegend=False)
        st.plotly_chart(fig_dd, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — CONSTRUCTION DU PORTEFEUILLE
# ═══════════════════════════════════════════════════════════════════════════
with tab3:
    section("Sélection & Pondération du Portefeuille", "🏗️")

    st.markdown("Choisissez les fonds à intégrer dans votre portefeuille et définissez les poids manuels ou laissez l'outil les optimiser dans l'onglet suivant.")

    sel_port = st.multiselect(
        "Fonds du portefeuille",
        fund_cols, default=fund_cols[:min(4, len(fund_cols))],
        key="sel_port"
    )

    if len(sel_port) < 2:
        st.warning("⚠️ Sélectionnez au moins 2 fonds pour construire un portefeuille.")
        st.stop()

    st.markdown("#### Poids manuels (la somme doit faire 100%)")
    cols_w = st.columns(len(sel_port))
    manual_weights = []
    for i, (col_w, fund) in enumerate(zip(cols_w, sel_port)):
        default_w = round(100 / len(sel_port), 1)
        w = col_w.number_input(fund, min_value=0.0, max_value=100.0, value=default_w, step=0.5, format="%.1f")
        manual_weights.append(w)

    total_w = sum(manual_weights)
    if abs(total_w - 100) > 0.1:
        st.error(f"⚠️ Somme des poids = {total_w:.1f}% ≠ 100%. Ajustez les poids.")
    else:
        weights_arr = np.array(manual_weights) / 100
        rets_port   = df_raw[sel_port].pct_change().dropna()

        p_ret, p_vol, p_shr = portfolio_performance(weights_arr, rets_port, freq)

        # VaR historique
        port_rets = rets_port @ weights_arr
        var_95_p  = np.percentile(port_rets, 5)
        var_99_p  = np.percentile(port_rets, 1)
        cvar_95_p = port_rets[port_rets <= var_95_p].mean()
        cvar_99_p = port_rets[port_rets <= var_99_p].mean()

        section("Métriques du Portefeuille Manuel", "📌")
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        with k1: kpi_card("Perf. Ann.", f"{p_ret*100:.2f}%", "positive" if p_ret > 0 else "negative")
        with k2: kpi_card("Volatilité", f"{p_vol*100:.2f}%", "neutral")
        with k3: kpi_card("Sharpe", f"{p_shr:.4f}", "positive" if p_shr > 0 else "negative")
        with k4: kpi_card("VaR 95%", f"{var_95_p*100:.4f}%", "negative")
        with k5: kpi_card("VaR 99%", f"{var_99_p*100:.4f}%", "negative")
        with k6: kpi_card("CVaR 99%", f"{cvar_99_p*100:.4f}%", "negative")

        # Pie allocation
        c1, c2 = st.columns([1, 2])
        with c1:
            fig_pie = go.Figure(go.Pie(
                labels=sel_port, values=manual_weights,
                hole=0.4, marker_colors=px.colors.qualitative.Set2,
                textinfo="label+percent"
            ))
            fig_pie.update_layout(title="Allocation", height=350,
                                   margin=dict(t=40, b=0, l=0, r=0))
            st.plotly_chart(fig_pie, use_container_width=True)

        with c2:
            # Performance cumulée du portefeuille
            cum_port = (1 + port_rets).cumprod()
            fig_cp = go.Figure()
            fig_cp.add_trace(go.Scatter(
                x=cum_port.index, y=(cum_port - 1) * 100,
                mode="lines", name="Portefeuille",
                line=dict(color="#005537", width=2.5),
                fill="tozeroy", fillcolor="rgba(0,85,55,0.1)"
            ))
            for f in sel_port:
                cum_f = (1 + rets_port[f]).cumprod()
                fig_cp.add_trace(go.Scatter(
                    x=cum_f.index, y=(cum_f - 1) * 100,
                    mode="lines", name=f, opacity=0.5,
                    line=dict(width=1, dash="dot")
                ))
            fig_cp.update_layout(
                title="Performance Cumulée (%)",
                xaxis_title="Date", yaxis_title="Perf. Cumulée (%)",
                template="plotly_white", height=350,
                legend=dict(orientation="h", y=-0.2)
            )
            st.plotly_chart(fig_cp, use_container_width=True)

        # Contribution au risque
        section("Contribution au Risque", "⚖️")
        n_ann  = annualize_factor(freq)
        cov    = rets_port.cov() * n_ann
        port_var = weights_arr @ cov.values @ weights_arr
        marginal = cov.values @ weights_arr
        contrib  = weights_arr * marginal / port_var * 100

        df_contrib = pd.DataFrame({
            "Fonds": sel_port,
            "Poids (%)": [f"{w:.1f}" for w in manual_weights],
            "Contribution au risque (%)": [f"{c:.2f}" for c in contrib]
        })
        st.dataframe(df_contrib, use_container_width=True, hide_index=True)

        st.session_state["sel_port"]    = sel_port
        st.session_state["rets_port"]   = rets_port
        st.session_state["manual_w"]    = weights_arr


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — OPTIMISATION & FRONTIÈRE EFFICIENTE
# ═══════════════════════════════════════════════════════════════════════════
with tab4:
    section("Optimisation de Portefeuille & Frontière Efficiente", "🎯")

    # Récupérer sélection depuis tab3
    sel_opt = st.session_state.get("sel_port", fund_cols[:min(4, len(fund_cols))])
    rets_opt = df_raw[sel_opt].pct_change().dropna()
    n_opt    = len(sel_opt)

    if n_opt < 2:
        st.warning("Retournez dans l'onglet 'Construction du Portefeuille' et sélectionnez au moins 2 fonds.")
        st.stop()

    st.info(f"💼 Portefeuille actif : **{', '.join(sel_opt)}** ({n_opt} fonds) | Contraintes : poids ∈ [{w_min*100:.0f}%, {w_max*100:.0f}%]")

    with st.spinner("⚙️ Calcul de la frontière efficiente..."):
        f_rets, f_vols, f_sharpe, f_weights = efficient_frontier(rets_opt, freq, n_points=80)

        w_ms  = optimize_max_sharpe(rets_opt, freq, w_min, w_max)
        w_mv  = optimize_min_variance(rets_opt, freq, w_min, w_max)
        w_mc  = optimize_min_cvar(rets_opt, freq, bounds_min=w_min, bounds_max=w_max)

        p_ms  = portfolio_performance(w_ms, rets_opt, freq)
        p_mv  = portfolio_performance(w_mv, rets_opt, freq)
        p_mc  = portfolio_performance(w_mc, rets_opt, freq)

    # ── Portefeuilles individuels (scatter)
    n_ann = annualize_factor(freq)
    ind_perf = (1 + rets_opt.mean()) ** n_ann - 1
    ind_vol  = rets_opt.std() * np.sqrt(n_ann)

    # ── Monte Carlo
    mc_rets, mc_vols, mc_sharpes = [], [], []
    for _ in range(n_sim):
        w = np.random.dirichlet(np.ones(n_opt))
        r, v, s = portfolio_performance(w, rets_opt, freq)
        mc_rets.append(r * 100)
        mc_vols.append(v * 100)
        mc_sharpes.append(s)

    # ── Figure principale
    section("Frontière Efficiente & Cloud Monte Carlo", "🌐")
    fig_ef = go.Figure()

    # Monte Carlo
    fig_ef.add_trace(go.Scatter(
        x=mc_vols, y=mc_rets,
        mode="markers",
        marker=dict(size=3, color=mc_sharpes, colorscale="Viridis",
                    showscale=True, colorbar=dict(title="Sharpe"),
                    opacity=0.5),
        name="Simulations MC",
        hovertemplate="Vol: %{x:.2f}%<br>Perf: %{y:.2f}%<extra></extra>"
    ))

    # Frontière efficiente
    if f_rets:
        fig_ef.add_trace(go.Scatter(
            x=[v * 100 for v in f_vols],
            y=[r * 100 for r in f_rets],
            mode="lines", name="Frontière Efficiente",
            line=dict(color="#C8952A", width=3),
            hovertemplate="Vol: %{x:.2f}%<br>Perf: %{y:.2f}%<extra></extra>"
        ))

    # Capital Market Line
    if f_rets and f_sharpe:
        best_idx = np.argmax(f_sharpe)
        best_ret = f_rets[best_idx]
        best_vol = f_vols[best_idx]
        cml_x = [0, best_vol * 100 * 1.3]
        cml_y = [RF_ANNUAL * 100, RF_ANNUAL * 100 + (best_ret - RF_ANNUAL) / best_vol * best_vol * 1.3 * 100]
        fig_ef.add_trace(go.Scatter(
            x=cml_x, y=cml_y,
            mode="lines", name="CML",
            line=dict(color="#6c757d", width=1.5, dash="dash")
        ))

    # Max Sharpe
    fig_ef.add_trace(go.Scatter(
        x=[p_ms[1] * 100], y=[p_ms[0] * 100],
        mode="markers+text",
        text=["Max Sharpe"], textposition="top center",
        marker=dict(size=14, color="#005537", symbol="star"),
        name=f"Max Sharpe ({p_ms[2]:.3f})"
    ))

    # Min Variance
    fig_ef.add_trace(go.Scatter(
        x=[p_mv[1] * 100], y=[p_mv[0] * 100],
        mode="markers+text",
        text=["Min Variance"], textposition="top right",
        marker=dict(size=14, color="#C8952A", symbol="diamond"),
        name=f"Min Variance"
    ))

    # Min CVaR
    fig_ef.add_trace(go.Scatter(
        x=[p_mc[1] * 100], y=[p_mc[0] * 100],
        mode="markers+text",
        text=["Min CVaR"], textposition="bottom center",
        marker=dict(size=14, color="#dc3545", symbol="triangle-up"),
        name=f"Min CVaR"
    ))

    # Fonds individuels
    fig_ef.add_trace(go.Scatter(
        x=ind_vol.values * 100, y=ind_perf.values * 100,
        mode="markers+text",
        text=list(rets_opt.columns),
        textposition="top right",
        marker=dict(size=9, color="#adb5bd", symbol="circle"),
        name="Fonds individuels"
    ))

    # Rf
    fig_ef.add_trace(go.Scatter(
        x=[0], y=[RF_ANNUAL * 100],
        mode="markers+text",
        text=["Rf"], textposition="top right",
        marker=dict(size=9, color="black", symbol="x"),
        name=f"Rf = {RF_ANNUAL*100:.2f}%"
    ))

    fig_ef.update_layout(
        title=f"Frontière Efficiente — {n_sim} simulations Monte Carlo",
        xaxis_title="Volatilité Annualisée (%)",
        yaxis_title="Performance Annualisée (%)",
        template="plotly_white", height=540,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="left", x=0),
        hovermode="closest"
    )
    st.plotly_chart(fig_ef, use_container_width=True)

    # ── KPI des 3 portefeuilles optimaux
    section("Portefeuilles Optimaux — Comparaison", "⭐")
    k1, k2, k3 = st.columns(3)

    def opt_card(col, label, p, w, color):
        with col:
            st.markdown(f"""
            <div style="border:2px solid {color}; border-radius:10px; padding:16px; text-align:center; margin:4px;">
                <div style="font-weight:700; color:{color}; font-size:1rem; margin-bottom:8px;">{label}</div>
                <div style="font-size:1.3rem; font-weight:700; color:#005537;">{p[0]*100:.2f}%</div>
                <div style="font-size:0.75rem; color:#6c757d;">Performance Ann.</div>
                <div style="font-size:1.1rem; font-weight:600; color:#C8952A; margin-top:6px;">{p[1]*100:.2f}%</div>
                <div style="font-size:0.75rem; color:#6c757d;">Volatilité Ann.</div>
                <div style="font-size:1.2rem; font-weight:700; color:#333; margin-top:6px;">{p[2]:.4f}</div>
                <div style="font-size:0.75rem; color:#6c757d;">Ratio de Sharpe</div>
            </div>""", unsafe_allow_html=True)

    opt_card(k1, "⭐ Max Sharpe",    p_ms, w_ms, "#005537")
    opt_card(k2, "🛡️ Min Variance",  p_mv, w_mv, "#C8952A")
    opt_card(k3, "⚠️ Min CVaR",      p_mc, w_mc, "#dc3545")

    # ── Poids des 3 stratégies
    section("Allocations Optimales", "🍕")
    df_weights = pd.DataFrame({
        "Fonds": sel_opt,
        "Max Sharpe (%)":   [f"{w*100:.2f}" for w in w_ms],
        "Min Variance (%)": [f"{w*100:.2f}" for w in w_mv],
        "Min CVaR (%)":     [f"{w*100:.2f}" for w in w_mc],
    })
    st.dataframe(df_weights, use_container_width=True, hide_index=True)

    # ── Pie charts des 3 stratégies
    c1, c2, c3 = st.columns(3)
    for col, label, w_arr, color in [
        (c1, "Max Sharpe",   w_ms, px.colors.sequential.Greens),
        (c2, "Min Variance", w_mv, px.colors.sequential.Oranges),
        (c3, "Min CVaR",     w_mc, px.colors.sequential.Reds),
    ]:
        with col:
            mask = w_arr > 0.005
            fig_p = go.Figure(go.Pie(
                labels=[f for f, m in zip(sel_opt, mask) if m],
                values=[w * 100 for w, m in zip(w_arr, mask) if m],
                hole=0.35,
                textinfo="label+percent",
                marker_colors=color[2:len(sel_opt)+3]
            ))
            fig_p.update_layout(title=label, height=300,
                                  margin=dict(t=40, b=0, l=0, r=0),
                                  showlegend=False)
            st.plotly_chart(fig_p, use_container_width=True)

    # ── Performance historique des 3 portefeuilles
    section("Backtest — Performance Cumulée des 3 Stratégies", "📜")
    port_cum_ms = (1 + rets_opt @ w_ms).cumprod()
    port_cum_mv = (1 + rets_opt @ w_mv).cumprod()
    port_cum_mc = (1 + rets_opt @ w_mc).cumprod()

    fig_bt = go.Figure()
    for name, cum, color in [
        ("Max Sharpe",   port_cum_ms, "#005537"),
        ("Min Variance", port_cum_mv, "#C8952A"),
        ("Min CVaR",     port_cum_mc, "#dc3545"),
    ]:
        fig_bt.add_trace(go.Scatter(
            x=cum.index, y=(cum - 1) * 100,
            mode="lines", name=name,
            line=dict(color=color, width=2),
            hovertemplate=f"<b>{name}</b><br>%{{x|%d/%m/%Y}}<br>%{{y:.2f}}%<extra></extra>"
        ))
    fig_bt.update_layout(
        title="Performance Cumulée Historique (%)",
        xaxis_title="Date", yaxis_title="Perf. Cumulée (%)",
        template="plotly_white", height=400,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
        hovermode="x unified"
    )
    st.plotly_chart(fig_bt, use_container_width=True)

    # ── Résumé export
    section("Export Résultats", "💾")
    df_export = pd.DataFrame({
        "Stratégie":        ["Max Sharpe", "Min Variance", "Min CVaR"],
        "Perf. Ann. (%)":   [f"{p_ms[0]*100:.4f}", f"{p_mv[0]*100:.4f}", f"{p_mc[0]*100:.4f}"],
        "Vol. Ann. (%)":    [f"{p_ms[1]*100:.4f}", f"{p_mv[1]*100:.4f}", f"{p_mc[1]*100:.4f}"],
        "Sharpe":           [f"{p_ms[2]:.4f}",     f"{p_mv[2]:.4f}",     f"{p_mc[2]:.4f}"],
    })
    for i, f in enumerate(sel_opt):
        df_export[f"w_{f} (%)"] = [f"{w_ms[i]*100:.2f}", f"{w_mv[i]*100:.2f}", f"{w_mc[i]*100:.2f}"]

    csv = df_export.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig")
    st.download_button(
        label="⬇️ Télécharger les résultats (CSV)",
        data=csv,
        file_name="opcvm_optimisation_resultats.csv",
        mime="text/csv"
    )
    st.dataframe(df_export, use_container_width=True, hide_index=True)
