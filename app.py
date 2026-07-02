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
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #005537 0%, #007850 100%);
    }
    [data-testid="stSidebar"] * { color: white !important; }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stMultiSelect label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stRadio label { color: #d4edda !important; font-weight: 500; }

    .main { background-color: #f8f9fa; }

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

    [data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }

    .badge {
        display: inline-block;
        background: #005537;
        color: white;
        border-radius: 4px;
        padding: 3px 10px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 2px 4px 8px 0;
    }
    .badge.gold { background: #C8952A; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────────────────────
TRADING_DAYS_ANNUAL = 247
WEEKS_ANNUAL = 52

# ─────────────────────────────────────────────────────────────────────────────
# FONCTIONS UTILITAIRES
# ─────────────────────────────────────────────────────────────────────────────

def parse_raw_file(uploaded_file):
    """Lit un fichier brut (CSV ou Excel) et retourne un DataFrame tel quel."""
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".csv"):
            for sep in [";", ",", "\t"]:
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, sep=sep, thousands=" ", decimal=",")
                    if df.shape[1] >= 2:
                        return df
                except:
                    pass
            return None
        else:
            return pd.read_excel(uploaded_file, index_col=None)
    except Exception as e:
        st.error(f"Erreur lecture fichier {uploaded_file.name} : {e}")
        return None


def detect_date_col(df):
    for col in df.columns:
        try:
            parsed = pd.to_datetime(df[col], dayfirst=True, errors="coerce")
            if parsed.notna().mean() > 0.7:
                return col
        except:
            continue
    return None


def detect_value_col(df, exclude_col):
    """Détecte la colonne de valeur (VL) parmi les colonnes restantes."""
    candidates = [c for c in df.columns if c != exclude_col]
    for c in candidates:
        if str(c).strip().lower() in ("value", "valeur", "vl", "nav", "val"):
            return c
    for c in candidates:
        if pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.7:
            return c
    return candidates[0] if candidates else None


def fund_name_from_filename(filename):
    """Dérive un nom de fonds propre à partir du nom de fichier."""
    base = filename.rsplit(".", 1)[0]
    base = base.replace("_", " ").replace("-", " ").strip()
    return base if base else filename


def load_single_fund_file(uploaded_file):
    """
    Lit un fichier 'value_date' / 'value' (1 fichier = 1 fonds) et retourne
    une Series indexée par date, nommée d'après le fichier.
    Retourne None si le fichier est illisible ou mal formé.
    """
    df = parse_raw_file(uploaded_file)
    if df is None or df.empty:
        return None

    cols_lower = {str(c).strip().lower(): c for c in df.columns}

    date_col = None
    for key in ("value_date", "date", "valuedate", "value date"):
        if key in cols_lower:
            date_col = cols_lower[key]
            break
    if date_col is None:
        date_col = detect_date_col(df)
    if date_col is None:
        st.warning(f"⚠️ {uploaded_file.name} : colonne date introuvable, fichier ignoré.")
        return None

    value_col = None
    for key in ("value", "valeur", "vl", "nav"):
        if key in cols_lower:
            value_col = cols_lower[key]
            break
    if value_col is None:
        value_col = detect_value_col(df, date_col)
    if value_col is None:
        st.warning(f"⚠️ {uploaded_file.name} : colonne valeur introuvable, fichier ignoré.")
        return None

    out = df[[date_col, value_col]].copy()
    out.columns = ["date", "value"]
    out["date"]  = pd.to_datetime(out["date"], dayfirst=True, errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["date", "value"]).sort_values("date")
    out = out.drop_duplicates(subset="date", keep="last")

    if out.empty:
        st.warning(f"⚠️ {uploaded_file.name} : aucune donnée valide après nettoyage, fichier ignoré.")
        return None

    fund_name = fund_name_from_filename(uploaded_file.name)
    series = out.set_index("date")["value"]
    series.name = fund_name
    return series


def build_wide_df_from_files(uploaded_files):
    """
    Combine plusieurs fichiers (1 fichier = 1 fonds, colonnes value_date/value)
    en un seul DataFrame large : index = Date, 1 colonne par fonds.
    """
    series_list = []
    for f in uploaded_files:
        s = load_single_fund_file(f)
        if s is not None:
            series_list.append(s)

    if not series_list:
        return None

    wide = pd.concat(series_list, axis=1)
    wide = wide.sort_index()
    wide.index.name = "Date"
    return wide.reset_index()


def annualize_factor(freq):
    return TRADING_DAYS_ANNUAL if freq == "Journalières" else WEEKS_ANNUAL


def detect_fund_frequency(date_index):
    """
    Détecte si une série de dates est journalière ou hebdomadaire,
    via l'écart médian (en jours) entre observations consécutives.
    """
    dates = pd.Series(pd.to_datetime(date_index)).sort_values().drop_duplicates()
    if len(dates) < 3:
        return "Journalières"
    diffs = dates.diff().dropna().dt.days
    median_gap = diffs.median()
    # Jour ouvré typique : 1-3j ; hebdo typique : 6-8j (week-end compris)
    return "Hebdomadaires" if median_gap >= 5 else "Journalières"


def resample_to_weekly(series):
    """
    Resample une série de VL en hebdomadaire en gardant la dernière valeur
    observée de chaque semaine (convention 'last occurrence', cohérente
    avec le pipeline de backtesting).
    """
    s = series.dropna().sort_index()
    return s.resample("W-FRI").last().dropna()


def build_weekly_aligned_df(df_raw, cols, fund_freq):
    """
    Construit un DataFrame de VL hebdomadaires aligné pour un ensemble de fonds,
    quelle que soit leur fréquence native (journalière ou hebdomadaire).
    Les fonds déjà hebdomadaires sont simplement réindexés sur grille W-FRI ;
    les fonds journaliers sont resamplés (dernière VL de la semaine).
    """
    weekly_series = []
    for c in cols:
        s = df_raw[c]
        s_weekly = resample_to_weekly(s)
        weekly_series.append(s_weekly)
    aligned = pd.concat(weekly_series, axis=1)
    aligned.columns = cols
    return aligned.dropna(how="all")


def compute_ratios(returns, freq, rf):
    n = annualize_factor(freq)
    rf_period = rf / n
    mean_ret  = returns.mean()
    cum_ret   = (1 + returns).prod() - 1
    perf_ann  = (1 + mean_ret) ** n - 1
    vol_ann   = returns.std() * np.sqrt(n)
    sharpe    = (perf_ann - rf) / vol_ann if vol_ann > 0 else np.nan
    downside  = returns[returns < rf_period].std() * np.sqrt(n)
    sortino   = (perf_ann - rf) / downside if downside > 0 else np.nan
    cum_vl    = (1 + returns).cumprod()
    drawdown  = (cum_vl - cum_vl.cummax()) / cum_vl.cummax()
    max_dd    = drawdown.min()
    calmar    = perf_ann / abs(max_dd) if max_dd != 0 else np.nan
    var_95    = np.percentile(returns.dropna().values, 5)
    var_99    = np.percentile(returns.dropna().values, 1)
    cvar_95   = returns[returns <= var_95].mean()
    cvar_99   = returns[returns <= var_99].mean()

    return {
        "Perf. Cumulée":    f"{cum_ret*100:.2f}%",
        "Perf. Annualisée": f"{perf_ann*100:.2f}%",
        "Volatilité Ann.":  f"{vol_ann*100:.2f}%",
        "Sharpe":           f"{sharpe:.4f}" if not np.isnan(sharpe) else "N/A",
        "Sortino":          f"{sortino:.4f}" if not np.isnan(sortino) else "N/A",
        "Calmar":           f"{calmar:.4f}" if not np.isnan(calmar) else "N/A",
        "Max Drawdown":     f"{max_dd*100:.2f}%",
        "VaR 95% (1j)":     f"{var_95*100:.4f}%",
        "VaR 99% (1j)":     f"{var_99*100:.4f}%",
        "CVaR 95%":         f"{cvar_95*100:.4f}%",
        "CVaR 99%":         f"{cvar_99*100:.4f}%",
        "_perf_ann": perf_ann,
        "_vol_ann":  vol_ann,
        "_sharpe":   sharpe,
        "_max_dd":   max_dd,
    }


def portfolio_performance(weights, returns_df, freq, rf):
    n = annualize_factor(freq)
    port_rets = returns_df @ weights
    perf_ann  = (1 + port_rets.mean()) ** n - 1
    vol_ann   = np.sqrt(weights @ (returns_df.cov().values * n) @ weights)
    sharpe    = (perf_ann - rf) / vol_ann if vol_ann > 0 else 0
    return perf_ann, vol_ann, sharpe


def efficient_frontier(returns_df, freq, rf, n_points=80):
    n_assets  = returns_df.shape[1]
    n_ann     = annualize_factor(freq)
    mean_rets = (1 + returns_df.mean()) ** n_ann - 1
    bounds      = tuple((0.0, 1.0) for _ in range(n_assets))
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    targets  = np.linspace(mean_rets.min(), mean_rets.max(), n_points)
    f_rets, f_vols, f_sharpe, f_weights = [], [], [], []

    for target in targets:
        cons = constraints + [{"type": "eq", "fun": lambda w, t=target: portfolio_performance(w, returns_df, freq, rf)[0] - t}]
        res  = minimize(
            lambda w: portfolio_performance(w, returns_df, freq, rf)[1],
            np.ones(n_assets) / n_assets,
            method="SLSQP", bounds=bounds, constraints=cons,
            options={"maxiter": 500, "ftol": 1e-9}
        )
        if res.success:
            p, v, s = portfolio_performance(res.x, returns_df, freq, rf)
            f_rets.append(p); f_vols.append(v)
            f_sharpe.append(s); f_weights.append(res.x)

    return f_rets, f_vols, f_sharpe, f_weights


def optimize_max_sharpe(returns_df, freq, rf, w_min=0.0, w_max=1.0):
    n = returns_df.shape[1]
    bounds      = tuple((w_min, w_max) for _ in range(n))
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    res = minimize(
        lambda w: -portfolio_performance(w, returns_df, freq, rf)[2],
        np.ones(n) / n, method="SLSQP", bounds=bounds, constraints=constraints,
        options={"maxiter": 500}
    )
    return res.x if res.success else np.ones(n) / n


def optimize_min_variance(returns_df, freq, rf, w_min=0.0, w_max=1.0):
    n   = returns_df.shape[1]
    n_ann = annualize_factor(freq)
    cov = returns_df.cov().values * n_ann
    bounds      = tuple((w_min, w_max) for _ in range(n))
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    res = minimize(
        lambda w: w @ cov @ w,
        np.ones(n) / n, method="SLSQP", bounds=bounds, constraints=constraints,
        options={"maxiter": 500}
    )
    return res.x if res.success else np.ones(n) / n


def optimize_min_cvar(returns_df, freq, rf, alpha=0.05, w_min=0.0, w_max=1.0):
    n = returns_df.shape[1]
    bounds      = tuple((w_min, w_max) for _ in range(n))
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    def neg_cvar(w):
        port_r = (returns_df @ w).dropna().values
        if len(port_r) == 0:
            return 0.0
        var = np.percentile(port_r, alpha * 100)
        tail = port_r[port_r <= var]
        return float(tail.mean()) if len(tail) > 0 else float(var)

    res = minimize(neg_cvar, np.ones(n) / n, method="SLSQP",
                   bounds=bounds, constraints=constraints, options={"maxiter": 500})
    return res.x if res.success else np.ones(n) / n


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
    uploaded_files = st.file_uploader(
        "Fichiers VL — 1 fichier par fonds",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True,
        help="Chaque fichier doit contenir une colonne 'value_date' et une colonne 'value'. Le nom du fichier sert de nom de fonds."
    )
    st.markdown("### ⚙️ Paramètres globaux")
    freq = st.radio(
        "Fréquence (données démo / fallback)", ["Journalières", "Hebdomadaires"], index=0,
        help="Utilisée uniquement pour les données de démo. Avec vos fichiers, la fréquence est détectée automatiquement par fonds."
    )
    rf_input = st.number_input("Taux sans risque annuel (%)", value=2.25, step=0.05, format="%.2f")
    RF       = rf_input / 100

    st.markdown("---")
    st.markdown("### 🔧 Optimisation — Contraintes")
    w_min = st.slider("Poids minimum (%)", 0, 20, 0, step=1) / 100
    w_max = st.slider("Poids maximum (%)", 20, 100, 100, step=5) / 100
    n_sim = st.slider("Simulations Monte Carlo", 500, 5000, 2000, step=500)

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
        "OMLT_A":   (0.00030, 0.0008),
        "OMLT_B":   (0.00028, 0.0009),
        "Oblig_CT": (0.00012, 0.0003),
        "Divers_A": (0.00025, 0.0020),
        "Divers_B": (0.00035, 0.0025),
        "Monetaire":(0.00008, 0.0001),
    }
    start = {"OMLT_A": 1000, "OMLT_B": 950, "Oblig_CT": 1200,
             "Divers_A": 800, "Divers_B": 1100, "Monetaire": 500}
    vl = {"Date": dates}
    for name, (mu, sigma) in funds.items():
        rets   = np.random.normal(mu, sigma, len(dates))
        prices = [start[name]]
        for r in rets[1:]:
            prices.append(prices[-1] * (1 + r))
        vl[name] = prices
    return pd.DataFrame(vl)


# ─────────────────────────────────────────────────────────────────────────────
# CHARGEMENT DONNÉES
# ─────────────────────────────────────────────────────────────────────────────
df_raw    = None
date_col  = None
fund_cols = []

if not uploaded_files:
    st.info("💡 Aucun fichier importé — données de démo (6 fonds fictifs)")
    df_raw    = load_demo_data()
    date_col  = "Date"
    fund_cols = [c for c in df_raw.columns if c != "Date"]
else:
    df_raw = build_wide_df_from_files(uploaded_files)
    if df_raw is not None:
        date_col  = "Date"
        fund_cols = [c for c in df_raw.columns if c != "Date"]
        st.success(f"✅ {len(fund_cols)} fonds chargés : {', '.join(fund_cols)}")
    else:
        st.error("❌ Aucun fichier n'a pu être lu correctement. Vérifiez le format (colonnes value_date / value).")

if df_raw is None or not fund_cols:
    st.stop()

for c in fund_cols:
    df_raw[c] = pd.to_numeric(df_raw[c], errors="coerce")

df_raw = df_raw.dropna(subset=fund_cols, how="all").set_index(date_col)

# ── Détection de la fréquence native de chaque fonds ──────────────────────
fund_freq = {}
for c in fund_cols:
    valid_dates = df_raw[c].dropna().index
    fund_freq[c] = detect_fund_frequency(valid_dates)

n_daily  = sum(1 for v in fund_freq.values() if v == "Journalières")
n_weekly = sum(1 for v in fund_freq.values() if v == "Hebdomadaires")
if n_daily and n_weekly:
    st.warning(
        f"⚠️ Fréquences mixtes détectées : {n_daily} fonds journaliers, {n_weekly} fonds hebdomadaires. "
        f"Les ratios individuels (Tabs 1-2) utilisent la fréquence native de chaque fonds. "
        f"Le portefeuille et l'optimisation (Tabs 3-4) alignent tout en hebdomadaire."
    )

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Évolution des VL",
    "📊 Ratios de Performance",
    "🏗️ Construction du Portefeuille",
    "🎯 Optimisation & Frontière Efficiente",
    "🧮 Allocation par Scoring"
])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — ÉVOLUTION DES VL
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    section("Visualisation des Valeurs Liquidatives", "📈")

    c1, c2 = st.columns([2, 1])
    with c1:
        sel_funds_vl = st.multiselect(
            "Fonds à afficher",
            fund_cols, default=fund_cols[:min(4, len(fund_cols))],
            key="sel_vl"
        )
    with c2:
        norm_vl = st.checkbox("Base 100", value=True)

    if sel_funds_vl:
        freq_badges = " &nbsp; ".join(
            f"<span class='badge {'gold' if fund_freq[f]=='Hebdomadaires' else ''}'>{f} · {fund_freq[f]}</span>"
            for f in sel_funds_vl
        )
        st.markdown(freq_badges, unsafe_allow_html=True)

        plot_df = df_raw[sel_funds_vl].copy().dropna()
        if norm_vl:
            plot_df = plot_df / plot_df.iloc[0] * 100

        colors = px.colors.qualitative.Set2
        fig = go.Figure()
        for i, col in enumerate(sel_funds_vl):
            fig.add_trace(go.Scatter(
                x=plot_df.index, y=plot_df[col],
                name=col, mode="lines",
                line=dict(color=colors[i % len(colors)], width=2),
                hovertemplate=f"<b>{col}</b><br>%{{x|%d/%m/%Y}}<br>%{{y:.4f}}<extra></extra>"
            ))
        fig.update_layout(
            title="Évolution des Valeurs Liquidatives",
            xaxis_title="Date",
            yaxis_title="VL (base 100)" if norm_vl else "VL",
            template="plotly_white", height=420,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

        section("Distribution des Rendements", "📊")
        rets_vl = df_raw[sel_funds_vl].pct_change().dropna()

        fig2 = go.Figure()
        for i, col in enumerate(sel_funds_vl):
            fig2.add_trace(go.Violin(
                y=rets_vl[col] * 100, name=col,
                box_visible=True, meanline_visible=True,
                line_color=colors[i % len(colors)],
                fillcolor=colors[i % len(colors)], opacity=0.6
            ))
        fig2.update_layout(
            title="Distribution des Rendements (%)",
            yaxis_title="Rendement (%)",
            template="plotly_white", height=380
        )
        st.plotly_chart(fig2, use_container_width=True)

        section("Matrice de Corrélation", "🔗")
        corr = rets_vl.corr()
        fig3 = px.imshow(
            corr, text_auto=".2f",
            color_continuous_scale="RdYlGn",
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
        "Fonds à analyser",
        fund_cols, default=fund_cols,
        key="sel_r"
    )

    if sel_funds_r:
        rets_all = df_raw[sel_funds_r].pct_change().dropna(how="all")
        results  = {}
        for f in sel_funds_r:
            ret_s = rets_all[f].dropna()
            if len(ret_s) > 5:
                results[f] = compute_ratios(ret_s, fund_freq[f], RF)

        if results:
            valid_funds = [f for f in sel_funds_r if f in results]

            freq_badges = " &nbsp; ".join(
                f"<span class='badge {'gold' if fund_freq[f]=='Hebdomadaires' else ''}'>{f} · {fund_freq[f]}</span>"
                for f in valid_funds
            )
            st.markdown(freq_badges, unsafe_allow_html=True)

            display_keys = [k for k in list(results[valid_funds[0]].keys()) if not k.startswith("_")]
            table_data   = {k: [results[f].get(k, "N/A") for f in valid_funds] for k in display_keys}
            df_table     = pd.DataFrame(table_data, index=valid_funds)
            st.dataframe(df_table.T, use_container_width=True, height=420)

            section("Comparaison Visuelle", "📉")
            c1, c2 = st.columns(2)

            with c1:
                sharpe_vals  = [results[f]["_sharpe"] for f in valid_funds if not np.isnan(results[f]["_sharpe"])]
                sharpe_names = [f for f in valid_funds if not np.isnan(results[f]["_sharpe"])]
                fig_s = go.Figure(go.Bar(
                    x=sharpe_names, y=sharpe_vals,
                    marker_color=["#005537" if v >= 0 else "#dc3545" for v in sharpe_vals],
                    text=[f"{v:.3f}" for v in sharpe_vals], textposition="outside"
                ))
                fig_s.update_layout(title="Ratio de Sharpe", template="plotly_white",
                                    height=320, yaxis_title="Sharpe", showlegend=False)
                st.plotly_chart(fig_s, use_container_width=True)

            with c2:
                vol_vals  = [results[f]["_vol_ann"] * 100 for f in valid_funds]
                perf_vals = [results[f]["_perf_ann"] * 100 for f in valid_funds]
                fig_rv = go.Figure()
                fig_rv.add_trace(go.Scatter(
                    x=vol_vals, y=perf_vals,
                    mode="markers+text",
                    text=valid_funds, textposition="top center",
                    marker=dict(size=12, color="#005537", opacity=0.8),
                ))
                fig_rv.update_layout(
                    title="Rendement vs Risque",
                    xaxis_title="Volatilité Ann. (%)",
                    yaxis_title="Performance Ann. (%)",
                    template="plotly_white", height=320
                )
                st.plotly_chart(fig_rv, use_container_width=True)

            section("Drawdown Historique", "📉")
            sel_dd = st.selectbox("Fonds pour le drawdown", valid_funds)
            cum_vl      = (1 + rets_all[sel_dd].dropna()).cumprod()
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

    st.markdown("Sélectionnez les fonds et définissez les poids manuels. L'optimisation se fait dans l'onglet suivant.")

    # NOTE : key="sel_port" → Streamlit gère automatiquement session_state["sel_port"]
    # Ne jamais écrire manuellement dans session_state["sel_port"]
    sel_port = st.multiselect(
        "Fonds du portefeuille",
        fund_cols,
        default=fund_cols[:min(4, len(fund_cols))],
        key="sel_port"
    )

    if len(sel_port) < 2:
        st.warning("⚠️ Sélectionnez au moins 2 fonds pour construire un portefeuille.")
    else:
        st.markdown("#### Poids manuels (somme = 100%)")
        cols_w         = st.columns(len(sel_port))
        manual_weights = []
        default_w      = round(100 / len(sel_port), 1)
        for col_w, fund in zip(cols_w, sel_port):
            w = col_w.number_input(
                fund, min_value=0.0, max_value=100.0,
                value=default_w, step=0.5, format="%.1f",
                key=f"w_{fund}"
            )
            manual_weights.append(w)

        total_w = sum(manual_weights)
        if abs(total_w - 100) > 0.1:
            st.error(f"⚠️ Somme des poids = {total_w:.1f}% ≠ 100%. Ajustez les poids.")
        else:
            PORT_FREQ = "Hebdomadaires"  # Alignement systématique en hebdo pour le portefeuille
            vl_port_weekly = build_weekly_aligned_df(df_raw, sel_port, fund_freq)

            if vl_port_weekly.dropna().shape[0] < 10:
                st.error("⚠️ Pas assez d'observations communes après alignement hebdomadaire entre les fonds sélectionnés.")
            else:
                weights_arr = np.array(manual_weights) / 100
                rets_port   = vl_port_weekly.pct_change().dropna()
                p_ret, p_vol, p_shr = portfolio_performance(weights_arr, rets_port, PORT_FREQ, RF)

                port_rets = rets_port @ weights_arr
                port_rets_clean = port_rets.dropna().values
                var_95_p  = np.percentile(port_rets_clean, 5)
                var_99_p  = np.percentile(port_rets_clean, 1)
                cvar_95_p = port_rets_clean[port_rets_clean <= var_95_p].mean()
                cvar_99_p = port_rets_clean[port_rets_clean <= var_99_p].mean()

                st.caption("📅 Portefeuille calculé sur VL hebdomadaires (vendredi) — alignement automatique des fonds journaliers et hebdomadaires.")

                section("Métriques du Portefeuille Manuel", "📌")
                k1, k2, k3, k4, k5, k6 = st.columns(6)
                with k1: kpi_card("Perf. Ann.",  f"{p_ret*100:.2f}%",   "positive" if p_ret > 0 else "negative")
                with k2: kpi_card("Volatilité",  f"{p_vol*100:.2f}%",   "neutral")
                with k3: kpi_card("Sharpe",      f"{p_shr:.4f}",        "positive" if p_shr > 0 else "negative")
                with k4: kpi_card("VaR 95%",     f"{var_95_p*100:.4f}%","negative")
                with k5: kpi_card("VaR 99%",     f"{var_99_p*100:.4f}%","negative")
                with k6: kpi_card("CVaR 99%",    f"{cvar_99_p*100:.4f}%","negative")

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
                    cum_port = (1 + port_rets).cumprod()
                    fig_cp   = go.Figure()
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
                        title="Performance Cumulée (%) — base hebdomadaire",
                        xaxis_title="Date", yaxis_title="Perf. Cumulée (%)",
                        template="plotly_white", height=350,
                        legend=dict(orientation="h", y=-0.2)
                    )
                    st.plotly_chart(fig_cp, use_container_width=True)

                section("Contribution au Risque", "⚖️")
                n_ann    = annualize_factor(PORT_FREQ)
                cov_m    = rets_port.cov() * n_ann
                port_var = weights_arr @ cov_m.values @ weights_arr
                marginal = cov_m.values @ weights_arr
                contrib  = weights_arr * marginal / port_var * 100

                df_contrib = pd.DataFrame({
                    "Fonds":                       sel_port,
                    "Poids (%)":                   [f"{w:.1f}" for w in manual_weights],
                    "Contribution au risque (%)":  [f"{c:.2f}" for c in contrib]
                })
                st.dataframe(df_contrib, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — OPTIMISATION & FRONTIÈRE EFFICIENTE
# ═══════════════════════════════════════════════════════════════════════════
with tab4:
    section("Optimisation de Portefeuille & Frontière Efficiente", "🎯")

    # Lecture de la sélection gérée par Streamlit (via key="sel_port")
    sel_opt = st.session_state.get("sel_port", None)
    if not sel_opt or len(sel_opt) < 2:
        sel_opt = fund_cols[:min(4, len(fund_cols))]

    OPT_FREQ = "Hebdomadaires"  # Alignement systématique en hebdo pour l'optimisation
    vl_opt_weekly = build_weekly_aligned_df(df_raw, sel_opt, fund_freq)
    rets_opt = vl_opt_weekly.pct_change().dropna()
    n_opt    = len(sel_opt)

    if rets_opt.shape[0] < 10:
        st.warning("⚠️ Pas assez d'observations communes après alignement hebdomadaire entre les fonds sélectionnés. Choisissez d'autres fonds dans l'onglet précédent.")
        st.stop()

    st.info(f"💼 Portefeuille actif : **{', '.join(sel_opt)}** ({n_opt} fonds) | Contraintes : poids ∈ [{w_min*100:.0f}%, {w_max*100:.0f}%]")
    st.caption("📅 Optimisation calculée sur VL hebdomadaires (vendredi) — alignement automatique des fonds journaliers et hebdomadaires.")

    with st.spinner("⚙️ Calcul de la frontière efficiente..."):
        f_rets, f_vols, f_sharpe, f_weights = efficient_frontier(rets_opt, OPT_FREQ, RF, n_points=80)
        w_ms = optimize_max_sharpe(rets_opt, OPT_FREQ, RF, w_min, w_max)
        w_mv = optimize_min_variance(rets_opt, OPT_FREQ, RF, w_min, w_max)
        w_mc = optimize_min_cvar(rets_opt, OPT_FREQ, RF, w_min=w_min, w_max=w_max)
        p_ms = portfolio_performance(w_ms, rets_opt, OPT_FREQ, RF)
        p_mv = portfolio_performance(w_mv, rets_opt, OPT_FREQ, RF)
        p_mc = portfolio_performance(w_mc, rets_opt, OPT_FREQ, RF)

    n_ann    = annualize_factor(OPT_FREQ)
    ind_perf = (1 + rets_opt.mean()) ** n_ann - 1
    ind_vol  = rets_opt.std() * np.sqrt(n_ann)

    # Monte Carlo
    mc_rets, mc_vols, mc_sharpes = [], [], []
    for _ in range(n_sim):
        w = np.random.dirichlet(np.ones(n_opt))
        r, v, s = portfolio_performance(w, rets_opt, OPT_FREQ, RF)
        mc_rets.append(r * 100)
        mc_vols.append(v * 100)
        mc_sharpes.append(s)

    section("Frontière Efficiente & Cloud Monte Carlo", "🌐")
    fig_ef = go.Figure()

    fig_ef.add_trace(go.Scatter(
        x=mc_vols, y=mc_rets, mode="markers",
        marker=dict(size=3, color=mc_sharpes, colorscale="Viridis",
                    showscale=True, colorbar=dict(title="Sharpe"), opacity=0.5),
        name="Simulations MC",
        hovertemplate="Vol: %{x:.2f}%<br>Perf: %{y:.2f}%<extra></extra>"
    ))

    if f_rets:
        fig_ef.add_trace(go.Scatter(
            x=[v * 100 for v in f_vols],
            y=[r * 100 for r in f_rets],
            mode="lines", name="Frontière Efficiente",
            line=dict(color="#C8952A", width=3),
            hovertemplate="Vol: %{x:.2f}%<br>Perf: %{y:.2f}%<extra></extra>"
        ))

        best_idx = int(np.argmax(f_sharpe))
        best_ret = f_rets[best_idx]
        best_vol = f_vols[best_idx]
        cml_x    = [0, best_vol * 100 * 1.3]
        slope    = (best_ret - RF) / best_vol if best_vol > 0 else 0
        cml_y    = [RF * 100, RF * 100 + slope * best_vol * 1.3 * 100]
        fig_ef.add_trace(go.Scatter(
            x=cml_x, y=cml_y, mode="lines", name="CML",
            line=dict(color="#6c757d", width=1.5, dash="dash")
        ))

    fig_ef.add_trace(go.Scatter(
        x=[p_ms[1] * 100], y=[p_ms[0] * 100],
        mode="markers+text", text=["Max Sharpe"], textposition="top center",
        marker=dict(size=14, color="#005537", symbol="star"),
        name=f"Max Sharpe ({p_ms[2]:.3f})"
    ))
    fig_ef.add_trace(go.Scatter(
        x=[p_mv[1] * 100], y=[p_mv[0] * 100],
        mode="markers+text", text=["Min Variance"], textposition="top right",
        marker=dict(size=14, color="#C8952A", symbol="diamond"),
        name="Min Variance"
    ))
    fig_ef.add_trace(go.Scatter(
        x=[p_mc[1] * 100], y=[p_mc[0] * 100],
        mode="markers+text", text=["Min CVaR"], textposition="bottom center",
        marker=dict(size=14, color="#dc3545", symbol="triangle-up"),
        name="Min CVaR"
    ))
    fig_ef.add_trace(go.Scatter(
        x=ind_vol.values * 100, y=ind_perf.values * 100,
        mode="markers+text",
        text=list(rets_opt.columns), textposition="top right",
        marker=dict(size=9, color="#adb5bd", symbol="circle"),
        name="Fonds individuels"
    ))
    fig_ef.add_trace(go.Scatter(
        x=[0], y=[RF * 100],
        mode="markers+text", text=["Rf"], textposition="top right",
        marker=dict(size=9, color="black", symbol="x"),
        name=f"Rf = {RF*100:.2f}%"
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

    section("Portefeuilles Optimaux — Comparaison", "⭐")
    k1, k2, k3 = st.columns(3)

    def opt_card(col, label, p, color):
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

    opt_card(k1, "⭐ Max Sharpe",   p_ms, "#005537")
    opt_card(k2, "🛡️ Min Variance", p_mv, "#C8952A")
    opt_card(k3, "⚠️ Min CVaR",     p_mc, "#dc3545")

    section("Allocations Optimales", "🍕")
    df_weights = pd.DataFrame({
        "Fonds":            sel_opt,
        "Max Sharpe (%)":   [f"{w*100:.2f}" for w in w_ms],
        "Min Variance (%)": [f"{w*100:.2f}" for w in w_mv],
        "Min CVaR (%)":     [f"{w*100:.2f}" for w in w_mc],
    })
    st.dataframe(df_weights, use_container_width=True, hide_index=True)

    c1, c2, c3 = st.columns(3)
    for col, label, w_arr, color in [
        (c1, "Max Sharpe",   w_ms, px.colors.sequential.Greens),
        (c2, "Min Variance", w_mv, px.colors.sequential.Oranges),
        (c3, "Min CVaR",     w_mc, px.colors.sequential.Reds),
    ]:
        with col:
            mask   = w_arr > 0.005
            labels = [f for f, m in zip(sel_opt, mask) if m]
            values = [w * 100 for w, m in zip(w_arr, mask) if m]
            if labels:
                fig_p = go.Figure(go.Pie(
                    labels=labels, values=values,
                    hole=0.35, textinfo="label+percent",
                    marker_colors=color[2:len(labels)+3]
                ))
                fig_p.update_layout(title=label, height=300,
                                    margin=dict(t=40, b=0, l=0, r=0),
                                    showlegend=False)
                st.plotly_chart(fig_p, use_container_width=True)

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

    section("Export Résultats", "💾")
    df_export = pd.DataFrame({
        "Stratégie":      ["Max Sharpe", "Min Variance", "Min CVaR"],
        "Perf. Ann. (%)": [f"{p_ms[0]*100:.4f}", f"{p_mv[0]*100:.4f}", f"{p_mc[0]*100:.4f}"],
        "Vol. Ann. (%)":  [f"{p_ms[1]*100:.4f}", f"{p_mv[1]*100:.4f}", f"{p_mc[1]*100:.4f}"],
        "Sharpe":         [f"{p_ms[2]:.4f}",     f"{p_mv[2]:.4f}",     f"{p_mc[2]:.4f}"],
    })
    for i, f in enumerate(sel_opt):
        df_export[f"w_{f} (%)"] = [
            f"{w_ms[i]*100:.2f}", f"{w_mv[i]*100:.2f}", f"{w_mc[i]*100:.2f}"
        ]

    csv = df_export.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig")
    st.download_button(
        label="⬇️ Télécharger les résultats (CSV)",
        data=csv,
        file_name="opcvm_optimisation_resultats.csv",
        mime="text/csv"
    )
    st.dataframe(df_export, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 5 — ALLOCATION PAR SCORING DE RATIOS
# ═══════════════════════════════════════════════════════════════════════════
with tab5:
    section("Allocation par scoring de ratios de performance", "🧮")

    st.markdown(
        "Outil de composition de portefeuille basé uniquement sur les ratios de performance "
        "de chaque fonds — sans optimisation mathématique. Chaque fonds reçoit un score "
        "composite pondéré, et le poids alloué est proportionnel à ce score."
    )

    # ── Sélection des fonds & calcul des ratios ─────────────────────────────
    sel_scoring = st.multiselect(
        "Fonds à inclure dans le scoring",
        fund_cols,
        default=fund_cols,
        key="sel_scoring"
    )

    if len(sel_scoring) < 2:
        st.warning("⚠️ Sélectionnez au moins 2 fonds.")
        st.stop()

    rets_sc = df_raw[sel_scoring].pct_change().dropna(how="all")
    ratios_sc = {}
    for f in sel_scoring:
        r = rets_sc[f].dropna()
        if len(r) > 5:
            ratios_sc[f] = compute_ratios(r, fund_freq[f], RF)

    valid_sc = [f for f in sel_scoring if f in ratios_sc]
    if len(valid_sc) < 2:
        st.warning("⚠️ Données insuffisantes pour calculer les ratios.")
        st.stop()

    # ── Profil investisseur ─────────────────────────────────────────────────
    section("Profil investisseur", "👤")
    profiles = {
        "Équilibré":   dict(sharpe=35, sortino=25, calmar=20, var=20),
        "Croissance":  dict(sharpe=45, sortino=30, calmar=10, var=15),
        "Défensif":    dict(sharpe=20, sortino=20, calmar=25, var=35),
        "Personnalisé":None,
    }
    profile_choice = st.radio(
        "Profil", list(profiles.keys()), horizontal=True, key="profile_sc"
    )

    if profiles[profile_choice] is not None:
        default_w = profiles[profile_choice]
    else:
        default_w = dict(sharpe=25, sortino=25, calmar=25, var=25)

    st.markdown("##### Poids des critères (total doit faire 100%)")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        w_sharpe_sc = st.slider("Sharpe (%)", 0, 100, default_w["sharpe"], 5, key="ws_sc")
    with c2:
        w_sortino_sc = st.slider("Sortino (%)", 0, 100, default_w["sortino"], 5, key="wso_sc")
    with c3:
        w_calmar_sc = st.slider("Calmar (%)", 0, 100, default_w["calmar"], 5, key="wc_sc")
    with c4:
        w_var_sc = st.slider("VaR 99% (%)", 0, 100, default_w["var"], 5, key="wv_sc")

    total_lambda = w_sharpe_sc + w_sortino_sc + w_calmar_sc + w_var_sc
    if total_lambda != 100:
        st.error(f"⚠️ La somme des poids = {total_lambda}% ≠ 100%. Ajustez les curseurs.")
        st.stop()

    # ── Calcul du scoring ───────────────────────────────────────────────────
    def minmax_norm(values, higher_is_better=True):
        arr = np.array(values, dtype=float)
        mn, mx = np.nanmin(arr), np.nanmax(arr)
        if mx == mn:
            return np.full(len(arr), 0.5)
        norm = (arr - mn) / (mx - mn)
        return norm if higher_is_better else 1 - norm

    sharpe_raw  = [ratios_sc[f]["_sharpe"]  for f in valid_sc]
    sortino_raw = []
    calmar_raw  = []
    var_raw     = []

    for f in valid_sc:
        r = rets_sc[f].dropna()
        n = annualize_factor(fund_freq[f])
        rf_p = RF / n
        perf_ann = ratios_sc[f]["_perf_ann"]

        # Sortino
        down = r[r < rf_p].std() * np.sqrt(n)
        sortino_raw.append((perf_ann - RF) / down if down > 0 else np.nan)

        # Calmar
        cum = (1 + r).cumprod()
        dd = (cum - cum.cummax()) / cum.cummax()
        mdd = dd.min()
        calmar_raw.append(perf_ann / abs(mdd) if mdd != 0 else np.nan)

        # VaR 99%
        var99 = np.percentile(r.dropna().values, 1)
        var_raw.append(var99)

    sharpe_norm  = minmax_norm(sharpe_raw,  higher_is_better=True)
    sortino_norm = minmax_norm(sortino_raw, higher_is_better=True)
    calmar_norm  = minmax_norm(calmar_raw,  higher_is_better=True)
    var_norm     = minmax_norm(var_raw,     higher_is_better=False)  # VaR : plus basse = mieux

    lS  = w_sharpe_sc  / 100
    lSo = w_sortino_sc / 100
    lC  = w_calmar_sc  / 100
    lV  = w_var_sc     / 100

    scores = [
        lS * sharpe_norm[i] + lSo * sortino_norm[i] +
        lC * calmar_norm[i] + lV  * var_norm[i]
        for i in range(len(valid_sc))
    ]

    pos_scores = [max(s, 0) for s in scores]
    total_score = sum(pos_scores)
    weights_sc = [s / total_score if total_score > 0 else 0 for s in pos_scores]

    # ── Tableau de scoring ──────────────────────────────────────────────────
    section("Tableau de scoring", "📋")

    ranked = sorted(
        zip(valid_sc, sharpe_raw, sortino_raw, calmar_raw, var_raw, scores, weights_sc),
        key=lambda x: x[5], reverse=True
    )

    df_scoring = pd.DataFrame({
        "Rang":         [f"#{i+1}" for i in range(len(ranked))],
        "Fonds":        [r[0] for r in ranked],
        "Sharpe":       [f"{r[1]:.4f}" if not np.isnan(r[1]) else "N/A" for r in ranked],
        "Sortino":      [f"{r[2]:.4f}" if not np.isnan(r[2]) else "N/A" for r in ranked],
        "Calmar":       [f"{r[3]:.4f}" if not np.isnan(r[3]) else "N/A" for r in ranked],
        "VaR 99%":      [f"{r[4]*100:.4f}%" for r in ranked],
        "Score (0→1)":  [f"{r[5]:.4f}" for r in ranked],
        "Poids alloué": [f"{r[6]*100:.2f}%" for r in ranked],
    })
    st.dataframe(df_scoring, use_container_width=True, hide_index=True)

    # ── KPI portefeuille résultant ──────────────────────────────────────────
    section("Performance du portefeuille scoré", "📌")

    w_arr_sc  = np.array([r[6] for r in ranked])
    funds_ord = [r[0] for r in ranked]

    vl_sc_weekly  = build_weekly_aligned_df(df_raw, funds_ord, fund_freq)
    rets_sc_port  = vl_sc_weekly.pct_change().dropna()
    port_sc_rets  = rets_sc_port @ w_arr_sc
    p_sc_ret, p_sc_vol, p_sc_shr = portfolio_performance(w_arr_sc, rets_sc_port, "Hebdomadaires", RF)

    clean_sc = port_sc_rets.dropna().values
    var95_sc  = np.percentile(clean_sc, 5)
    var99_sc  = np.percentile(clean_sc, 1)
    cvar99_sc = clean_sc[clean_sc <= var99_sc].mean()

    k1, k2, k3, k4, k5_col = st.columns(5)
    with k1: kpi_card("Perf. Ann.",  f"{p_sc_ret*100:.2f}%",  "positive" if p_sc_ret > 0 else "negative")
    with k2: kpi_card("Volatilité",  f"{p_sc_vol*100:.2f}%",  "neutral")
    with k3: kpi_card("Sharpe",      f"{p_sc_shr:.4f}",       "positive" if p_sc_shr > 0 else "negative")
    with k4: kpi_card("VaR 99%",     f"{var99_sc*100:.4f}%",  "negative")
    with k5_col: kpi_card("CVaR 99%",f"{cvar99_sc*100:.4f}%", "negative")

    # ── Recommandation par fonds ────────────────────────────────────────────
    section("Recommandation", "💡")

    rec_include = [(r[0], r[6]) for r in ranked if r[6] >= 0.15]
    rec_watch   = [(r[0], r[6]) for r in ranked if 0 < r[6] < 0.15]
    rec_exclude = [r[0] for r in ranked if r[6] == 0]

    if rec_include:
        fonds_str = " · ".join([f"{f} ({w*100:.1f}%)" for f, w in rec_include])
        st.success(f"**Position principale** — {fonds_str}")
    if rec_watch:
        fonds_str = " · ".join([f"{f} ({w*100:.1f}%)" for f, w in rec_watch])
        st.warning(f"**Position satellite** — {fonds_str}")
    if rec_exclude:
        st.error(f"**Score nul — à exclure** — {' · '.join(rec_exclude)}")

    # ── Visualisations ──────────────────────────────────────────────────────
    section("Visualisations", "📊")
    c1, c2 = st.columns(2)

    with c1:
        labels_pie = [r[0] for r in ranked if r[6] > 0]
        vals_pie   = [r[6]*100 for r in ranked if r[6] > 0]
        fig_pie_sc = go.Figure(go.Pie(
            labels=labels_pie, values=vals_pie,
            hole=0.4, marker_colors=px.colors.qualitative.Set2,
            textinfo="label+percent"
        ))
        fig_pie_sc.update_layout(
            title="Allocation par scoring", height=340,
            margin=dict(t=40, b=0, l=0, r=0)
        )
        st.plotly_chart(fig_pie_sc, use_container_width=True)

    with c2:
        fig_bar_sc = go.Figure(go.Bar(
            x=[r[5] for r in ranked],
            y=[r[0] for r in ranked],
            orientation="h",
            marker_color=["#005537" if r[6] >= 0.15 else "#C8952A" if r[6] > 0 else "#dc3545" for r in ranked],
            text=[f"{r[5]:.3f}" for r in ranked],
            textposition="outside"
        ))
        fig_bar_sc.update_layout(
            title="Score composite par fonds",
            xaxis_title="Score (0 → 1)",
            template="plotly_white", height=340,
            yaxis=dict(autorange="reversed"),
            showlegend=False
        )
        st.plotly_chart(fig_bar_sc, use_container_width=True)

    # Performance cumulée portefeuille scoré vs fonds individuels
    cum_sc = (1 + port_sc_rets).cumprod()
    fig_sc_perf = go.Figure()
    fig_sc_perf.add_trace(go.Scatter(
        x=cum_sc.index, y=(cum_sc - 1) * 100,
        mode="lines", name="Portefeuille scoré",
        line=dict(color="#005537", width=2.5),
        fill="tozeroy", fillcolor="rgba(0,85,55,0.08)"
    ))
    colors_sc = px.colors.qualitative.Set2
    for i, f in enumerate(funds_ord):
        cum_f = (1 + rets_sc_port[f]).cumprod()
        fig_sc_perf.add_trace(go.Scatter(
            x=cum_f.index, y=(cum_f - 1) * 100,
            mode="lines", name=f, opacity=0.5,
            line=dict(width=1, dash="dot", color=colors_sc[i % len(colors_sc)])
        ))
    fig_sc_perf.update_layout(
        title="Performance cumulée — Portefeuille scoré vs fonds individuels (%)",
        xaxis_title="Date", yaxis_title="Perf. cumulée (%)",
        template="plotly_white", height=400,
        legend=dict(orientation="h", y=-0.2),
        hovermode="x unified"
    )
    st.plotly_chart(fig_sc_perf, use_container_width=True)

    # ── Comparaison avec les 3 stratégies d'optimisation ───────────────────
    section("Comparaison avec les stratégies d'optimisation", "⚖️")
    st.caption("Les métriques ci-dessous comparent le portefeuille scoré aux 3 portefeuilles optimaux calculés dans l'onglet Optimisation — sur les mêmes fonds si disponibles.")

    try:
        common_funds = [f for f in funds_ord if f in sel_opt]
        if len(common_funds) >= 2:
            vl_cmp   = build_weekly_aligned_df(df_raw, common_funds, fund_freq)
            rets_cmp = vl_cmp.pct_change().dropna()

            w_ms_cmp = optimize_max_sharpe(rets_cmp, "Hebdomadaires", RF, w_min, w_max)
            w_mv_cmp = optimize_min_variance(rets_cmp, "Hebdomadaires", RF, w_min, w_max)
            p_ms_cmp = portfolio_performance(w_ms_cmp, rets_cmp, "Hebdomadaires", RF)
            p_mv_cmp = portfolio_performance(w_mv_cmp, rets_cmp, "Hebdomadaires", RF)

            w_sc_cmp = np.array([
                weights_sc[valid_sc.index(f)] if f in valid_sc else 0
                for f in common_funds
            ])
            w_sc_cmp = w_sc_cmp / w_sc_cmp.sum() if w_sc_cmp.sum() > 0 else w_sc_cmp
            p_sc_cmp = portfolio_performance(w_sc_cmp, rets_cmp, "Hebdomadaires", RF)

            df_cmp = pd.DataFrame({
                "Stratégie":      ["Scoring (ratios)", "Max Sharpe", "Min Variance"],
                "Perf. Ann. (%)": [f"{p_sc_cmp[0]*100:.2f}", f"{p_ms_cmp[0]*100:.2f}", f"{p_mv_cmp[0]*100:.2f}"],
                "Vol. Ann. (%)":  [f"{p_sc_cmp[1]*100:.2f}", f"{p_ms_cmp[1]*100:.2f}", f"{p_mv_cmp[1]*100:.2f}"],
                "Sharpe":         [f"{p_sc_cmp[2]:.4f}",     f"{p_ms_cmp[2]:.4f}",     f"{p_mv_cmp[2]:.4f}"],
            })
            st.dataframe(df_cmp, use_container_width=True, hide_index=True)
        else:
            st.info("Sélectionnez les mêmes fonds dans l'onglet Optimisation pour afficher la comparaison.")
    except Exception:
        st.info("Lancez d'abord l'onglet Optimisation pour afficher la comparaison.")

    # ── Export ──────────────────────────────────────────────────────────────
    section("Export", "💾")
    csv_sc = df_scoring.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig")
    st.download_button(
        label="⬇️ Télécharger le scoring (CSV)",
        data=csv_sc,
        file_name="opcvm_scoring_allocation.csv",
        mime="text/csv"
    )
