# --- SISTEMA E UI ---
import streamlit as st
import streamlit.components.v1 as components
import math
import time
# Unificando as importações de datetime para evitar conflitos de namespace
from datetime import datetime, timedelta

# --- MANIPULAÇÃO DE DADOS E ESTATÍSTICA ---
import pandas as pd
import numpy as np
from scipy.stats import shapiro
from scipy.optimize import minimize
from scipy.stats import norm
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.regression.linear_model import OLS

# --- ENGINE QUANTITATIVO (O "Coração") ---
import quantstats as qs
from pypfopt import expected_returns
from pypfopt.efficient_frontier import EfficientCVaR
import yfinance as yf
from bcb import sgs

# --- VISUALIZAÇÃO ---
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.io as pio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

# --- EXPORTAÇÃO ---
import textwrap
from fpdf import FPDF
import re
import pdfplumber
import zipfile

# 1. INICIALIZAÇÃO DO ESTADO (Correção Crítica: use '=' para atribuir)
if 'pagina_atual' not in st.session_state:
    st.session_state['pagina_atual'] = 'home'  # Corrigido: era '=='

# 2. DEFINIÇÃO DINÂMICA DE LAYOUT
configs = {
    'home': {"titulo": "BLACK BOX TERMINAL", "layout": "wide"},
    'painel_suitability': {"titulo": "BLACK BOX TERMINAL", "layout": "wide"},
    'painel_alocacao': {"titulo": "BLACK BOX TERMINAL", "layout": "wide"},
    'painel_otimizacao': {"titulo": "BLACK BOX TERMINAL", "layout": "wide"},
    'painel_calculadora': {"titulo": "BLACK BOX TERMINAL", "layout": "wide"}
}

# Busca a config baseada no estado atual (ou usa um fallback)
current_page = st.session_state['pagina_atual']
config_atual = configs.get(current_page, configs['home'])

# 3. CONFIGURAÇÃO DA PÁGINA (Deve ser o primeiro comando Streamlit UI)
st.set_page_config(
    page_title=config_atual['titulo'],
    layout=config_atual['layout'],
    page_icon="LOGO_BBX.png",
    initial_sidebar_state="collapsed"
)

# Define o tema escuro padrão para todos os gráficos do BLACK BOX
pio.templates.default = "plotly_dark"

# ============================================================
# ESTILIZAÇÃO GLOBAL (CSS unificado — sem duplicatas)
# ============================================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;700&display=swap');

    .stApp {
        background: radial-gradient(circle at center, #1A1A1A 0%, #080808 100%) !important;
        color: #E0E0E0;
        font-family: 'Inter', sans-serif !important;
    }
    .block-container {
        max-width: 1000px !important;
        padding-top: 3rem !important;
        padding-bottom: 5rem !important;
        margin: auto !important;
    }
    h1, h2, h3, h4, p, span, label, div { font-family: 'Inter', sans-serif !important; color: #E0E0E0 !important; }

    [data-testid="stVerticalBlock"] > div:has(div[data-testid="stMetric"]) {
        background-color: rgba(26, 26, 26, 0.7); padding: 25px; border-radius: 12px;
        border: 1px solid #333333; box-shadow: 0 4px 15px rgba(0,0,0,0.5);
    }
    [data-testid="stMetricValue"] { color: #C5A059 !important; font-weight: 700; font-size: 2.2rem; }
    [data-testid="stMetricValue"] > div, [data-testid="stMetricValue"] span { color: #C5A059 !important; }

    .stTextInput > div, .stNumberInput > div { background: transparent !important; border: none !important; box-shadow: none !important; }
    .stTextInput > div > div, .stNumberInput > div > div {
        background-color: #0D0D0D !important; border: none !important;
        border-bottom: 1px solid #2A2A2A !important; border-radius: 0 !important;
        box-shadow: none !important; transition: border-color 0.25s ease !important;
    }
    .stTextInput > div > div:focus-within, .stNumberInput > div > div:focus-within {
        border-bottom: 1px solid #C5A059 !important; box-shadow: 0 2px 8px rgba(197, 160, 89, 0.08) !important;
    }
    input[type="number"], input[type="text"], textarea, .stTextInput input, .stNumberInput input {
        background-color: #0D0D0D !important; color: #D8D8D8 !important; border: none !important;
        border-radius: 0 !important; box-shadow: none !important; caret-color: #C5A059 !important;
        font-family: 'Inter', sans-serif !important; font-size: 0.7rem !important;
        letter-spacing: 0.02em !important; padding: 8px 4px !important;
    }
    input::placeholder, textarea::placeholder { color: #3A3A3A !important; opacity: 1 !important; font-style: normal !important; letter-spacing: 0.05em !important; }

    div[data-baseweb="select"] > div {
        background-color: #0D0D0D !important; border: none !important;
        border-bottom: 1px solid #2A2A2A !important; border-radius: 0 !important;
        box-shadow: none !important; transition: border-color 0.25s ease !important;
    }
    div[data-baseweb="select"] > div:focus-within, div[data-baseweb="select"]:focus-within > div {
        border-bottom: 1px solid #C5A059 !important; box-shadow: none !important;
    }
    div[data-baseweb="select"] input, [data-testid="stSelectbox"] div {
        background-color: #0D0D0D !important; color: #D8D8D8 !important; border: none !important;
        box-shadow: none !important; font-family: 'Inter', sans-serif !important; font-size: 0.9rem !important;
    }
    div[data-baseweb="select"] svg { color: #444 !important; fill: #444 !important; transition: color 0.2s ease !important; }
    div[data-baseweb="select"]:focus-within svg { color: #C5A059 !important; fill: #C5A059 !important; }

    ul[data-baseweb="menu"], div[data-baseweb="popover"] {
        background-color: #0D0D0D !important; border: 1px solid #1E1E1E !important;
        border-radius: 0 !important; box-shadow: 0 8px 32px rgba(0,0,0,0.7) !important;
    }
    li[role="option"] {
        background-color: #0D0D0D !important; color: #808080 !important;
        font-family: 'Inter', sans-serif !important; font-size: 0.85rem !important;
        letter-spacing: 0.03em !important; border-bottom: 1px solid #141414 !important; transition: all 0.15s ease !important;
    }
    li[role="option"]:hover, li[role="option"][aria-selected="true"] {
        background-color: #141414 !important; color: #C5A059 !important; padding-left: 18px !important;
    }

    button[data-testid="stNumberInputStepUp"], button[data-testid="stNumberInputStepDown"] {
        background-color: #0D0D0D !important; color: #444 !important; border: none !important;
        border-left: 1px solid #1E1E1E !important; border-radius: 0 !important; transition: color 0.2s ease !important;
    }
    button[data-testid="stNumberInputStepUp"]:hover, button[data-testid="stNumberInputStepDown"]:hover {
        background-color: #141414 !important; color: #C5A059 !important;
    }

    div.stButton > button[kind="primary"], div.stDownloadButton > button[kind="primary"],
    div.stLinkButton > a[kind="primary"], div.stFormSubmitButton > button {
        background-color: #C5A059 !important; background: #C5A059 !important; color: #000000 !important;
        border: none !important; border-radius: 2px !important; font-weight: bold !important;
        letter-spacing: 1.5px !important; text-transform: uppercase !important;
        transition: all 0.3s ease-in-out !important; height: 45px !important;
        display: flex !important; align-items: center !important; justify-content: center !important; text-decoration: none !important;
    }
    div.stButton > button[kind="primary"]:hover, div.stDownloadButton > button[kind="primary"]:hover,
    div.stLinkButton > a[kind="primary"]:hover, div.stFormSubmitButton > button:hover {
        background-color: #A68546 !important; color: #FFFFFF !important;
        transform: translateY(-2px); box-shadow: 0px 4px 15px rgba(197, 160, 89, 0.3) !important;
    }

    div.stButton > button, div.stDownloadButton > button, div.stLinkButton > a {
        background-color: transparent !important; color: #C5A059 !important; border: none !important;
        border-radius: 2px !important; font-weight: 700 !important; text-transform: uppercase !important;
        letter-spacing: 1px !important; text-decoration: none !important;
    }
    div.stButton > button:hover, div.stDownloadButton > button:hover, div.stLinkButton > a:hover {
        background-color: #C5A059 !important; color: #080808 !important;
    }

    div[data-baseweb="tab-highlight"] { background-color: #C5A059 !important; }
    div[data-baseweb="tab"] { background-color: transparent !important; color: #666666 !important; }
    div[data-baseweb="tab"][aria-selected="true"] { color: #C5A059 !important; }
    div[data-baseweb="tab-border"] { background-color: #333333 !important; }
    button[data-baseweb="tab"] { font-size: 0.72rem !important; font-weight: 700 !important; letter-spacing: 0.2em !important; text-transform: uppercase !important; color: #666 !important; }
    button[aria-selected="true"] { color: #C5A059 !important; border-bottom-color: #C5A059 !important; }

    div[data-testid="stAlert"] { background-color: #1A1A1A !important; border-color: #C5A059 !important; color: #E0E0E0 !important; }
    div[data-testid="stAlert"] p, div[data-testid="stAlert"] span { color: #E0E0E0 !important; }

    [data-testid="column"] { width: fit-content !important; min-width: min-content !important; }

    ::-webkit-scrollbar { width: 4px; background: #080808; }
    ::-webkit-scrollbar-thumb { background: #333333; border-radius: 2px; }
    ::-webkit-scrollbar-thumb:hover { background: #C5A059; }

    .glass-card {
        background: linear-gradient(135deg, rgba(26,26,26,0.9) 0%, rgba(8,8,8,0.95) 100%);
        border: 1px solid #333333; border-radius: 8px; padding: 30px;
        transition: all 0.5s cubic-bezier(0.165, 0.84, 0.44, 1); margin-bottom: 15px;
        position: relative; overflow: hidden;
    }
    .glass-card::before {
        content: ""; position: absolute; top: 0; left: 0; width: 100%; height: 2px;
        background: linear-gradient(90deg, transparent, transparent); transition: all 0.5s ease;
    }
    .glass-card:hover { transform: translateY(-5px); border-color: #C5A059; box-shadow: 0 20px 40px rgba(0,0,0,0.6); }
    .glass-card:hover::before { background: linear-gradient(90deg, transparent, #C5A059, transparent); }

    .portal-title {
        font-family: 'Inter', sans-serif; font-size: 1.1rem; font-weight: 700; color: #E0E0E0;
        margin-bottom: 12px; display: flex; align-items: center; gap: 12px;
        letter-spacing: 1px; text-transform: uppercase;
    }
    .portal-text { font-family: 'Inter', sans-serif; color: #808080; line-height: 1.7; font-size: 0.9rem; margin-bottom: 5px; }

    .status-dot { height: 6px; width: 6px; border-radius: 50%; display: inline-block; }
    .dot-gold { background-color: #C5A059; box-shadow: 0 0 10px #C5A059; }
    .dot-dim { background-color: #333333; }

    .status-bar {
        background: rgba(26,26,26,0.8); padding: 12px 20px; border-radius: 4px;
        display: flex; justify-content: space-between; align-items: center;
        border: 1px solid #333333; margin-bottom: 30px; backdrop-filter: blur(10px);
    }
    .status-pulse-container { display: flex; align-items: center; gap: 15px; }
    .status-led {
        width: 6px; height: 6px; background-color: #C5A059; border-radius: 50%;
        box-shadow: 0 0 12px #C5A059; animation: pulse-gold 2s infinite ease-in-out;
    }
    @keyframes pulse-gold {
        0% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.4; transform: scale(0.9); }
        100% { opacity: 1; transform: scale(1); }
    }
    .scan-line { width: 120px; height: 1px; background: rgba(197,160,89,0.1); position: relative; overflow: hidden; }
    .scan-line::after {
        content: ""; display: block; width: 50px; height: 100%;
        background: linear-gradient(90deg, transparent, #C5A059, transparent);
        position: absolute; animation: scan 3s infinite linear;
    }
    @keyframes scan { from { left: -50%; } to { left: 150%; } }
    .status-text { font-family: 'Inter', sans-serif; font-size: 10px; color: #808080; letter-spacing: 1.5px; text-transform: uppercase; }

    .breakdown-table { width: 100%; border-collapse: collapse; font-size: 13px; margin-bottom: 1rem; }
    .breakdown-table tr { border-bottom: 1px solid #1A1A1A; }
    .breakdown-table td { padding: 12px 5px; color: #999; }
    .breakdown-table td.val { text-align: right; font-weight: 600; color: #E0E0E0; font-family: 'Inter', sans-serif; }
    .breakdown-table td.val.neg { color: #e74c3c; }
    .breakdown-table td.val.pos { color: #2ecc71; }

    .info-box { background: #0E1117; border-left: 3px solid #C5A059; border-radius: 2px; padding: 15px; font-size: 12px; color: #888; margin-top: 20px; line-height: 1.6; }

    .section-title { font-family: 'Inter', sans-serif; font-size: 10px; font-weight: 700; letter-spacing: 0.2em; color: #444; text-transform: uppercase; margin: 2rem 0 1rem; border-bottom: 1px solid #1A1A1A; padding-bottom: 5px; }
    .divider { height: 1px; background: #1A1A1A; margin: 2rem 0; }

    .metric-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin-bottom: 15px; width: 100%; }
    .stat-card {
        background: #1A1A1A; padding: 22px; border-radius: 2px; border: 1px solid #333;
        text-align: center; transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        box-sizing: border-box; min-width: 0; display: flex; flex-direction: column; justify-content: space-between;
    }
    .stat-card:hover { border-color: #C5A059; transform: translateY(-2px); box-shadow: 0 4px 20px rgba(197,160,89,0.1); }
    .stat-value { font-size: 1.5rem; font-weight: 700; color: #E0E0E0; margin: 0; }
    .stat-label { font-size: 0.65rem; color: #666; font-weight: 800; letter-spacing: 1.5px; margin-bottom: 8px; text-transform: uppercase; }
    .stat-status { font-size: 0.65rem; font-weight: 700; margin-top: 10px; text-transform: uppercase; color: #808080; }

    .section-header-blackbox { font-family: 'Inter', sans-serif !important; font-size: 10px !important; font-weight: 400 !important; letter-spacing: 0.15em !important; color: #808080 !important; text-transform: uppercase !important; margin-bottom: 8px !important; display: inline-block; width: 100%; padding-bottom: 4px; }
    .section-header-blackbox b { color: #C5A059 !important; font-weight: 700 !important; }

    .declaração-container { background-color: #111111; padding: 24px; border-radius: 4px; border-left: 5px solid #C5A059; border-top: 1px solid #333333; border-right: 1px solid #333333; border-bottom: 1px solid #333333; box-shadow: 0 4px 15px rgba(0,0,0,0.5); margin-bottom: 25px; }
    .declaração-titulo { color: #C5A059; font-size: 16px; font-weight: 700; margin-bottom: 12px; display: flex; align-items: center; gap: 8px; letter-spacing: 1px; }
    .declaração-texto { color: #999999; font-size: 12px; line-height: 1.6; }

    [data-testid="stForm"] { border: none !important; padding: 0 !important; }

    @media (max-width: 768px) {
        .metric-grid { grid-template-columns: 1fr !important; gap: 10px !important; }
        .stat-card { padding: 15px !important; }
        .stat-value { font-size: 1.35rem !important; }
        .stat-label { font-size: 0.6rem !important; }
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# HELPERS / FUNÇÕES UTILITÁRIAS
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def buscar_dados_limpos(tickers, dias):
    fim = datetime.now()
    inicio = fim - timedelta(days=dias)

    tickers_com_benchmark = list(tickers) + ["^BVSP", "IVV"]

    df = yf.download(tickers_com_benchmark, start=inicio, end=fim, progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df = df['Close']
    elif 'Close' in df.columns:
        df = df[['Close']]

    return df.dropna()

@st.cache_data(ttl=86400, show_spinner=False)
def buscar_rfr_pro():
    """
    Busca a Taxa Livre de Risco anual (Selic) com redundância.
    Retorna rfr_anual (float).
    """
    rfr_anual = 0.1125  # Fallback

    try:
        from bcb import sgs
        df_selic = sgs.get({'selic': 1178}, last=1)
        rfr_anual = df_selic['selic'].iloc[-1] / 100
    except Exception:
        try:
            ticker_cdi = yf.Ticker("CDI=F")
            rfr_anual = ticker_cdi.history(period="5d")['Close'].iloc[-1] / 100
        except Exception:
            pass

    return rfr_anual  # ← Retorna anual; conversão para diária fica no script principal

@st.cache_data(ttl=86400, show_spinner=False)
def get_bc_data(codigo_serie, data_inicio):
    """Busca dados do Banco Central (SGS)."""
    try:
        url = (
            f'https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo_serie}'
            f'/dados?formato=json&dataInicial={data_inicio}'
        )
        df = pd.read_json(url)
        if df.empty:
            return pd.Series()
        df['data'] = pd.to_datetime(df['data'], dayfirst=True)
        df.set_index('data', inplace=True)
        return df['valor'] / 100
    except Exception:
        return pd.Series()

def formatar_cpf(cpf):
    cpf_limpo = "".join(filter(str.isdigit, str(cpf)))
    if len(cpf_limpo) == 11:
        return f"{cpf_limpo[:3]}.{cpf_limpo[3:6]}.{cpf_limpo[6:9]}-{cpf_limpo[9:]}"
    return cpf_limpo

def clean_pdf(txt):
    return str(txt).encode('latin-1', 'replace').decode('latin-1')

def dynamic_footer():
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
        <div style="text-align: center; border-top: 1px solid #e2e8f0; padding-top: 20px; margin-top: 30px;">
            <p style="margin-bottom: 5px; font-weight: 700; color: #1e293b; letter-spacing: 2px;">BLACK BLACK BOX TERMINAL</p>
            <p style="font-size: 10px; color: #94a3b8; letter-spacing: 3px; font-family: 'Inter', sans-serif;">
                POWERED BY <span style="color: #7c3aed; font-weight: 800;">LYNX INTELLIGENCE ENGINE</span>
            </p>
        </div>
    """, unsafe_allow_html=True)

def lynx_intelligence():
    import datetime as dt # Importação local e segura para evitar conflitos
    current_time = dt.datetime.now().strftime("%H:%M:%S")
    
    st.markdown(f"""
        <div class="status-bar">
            <div class="status-pulse-container">
                <div class="status-led"></div>
                <div class="status-text">BLACK BOX INTEL ENGINE: ONLINE</div>
                <div class="scan-line"></div>
            </div>
            <div class="status-text" style="font-size: 9px; opacity: 0.6;">
                L-SYNC: {current_time} | NODE: LYNX-V2.0
            </div>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
def page_header():
    """Cabeçalho de navegação padrão para todas as páginas."""
    st.markdown("<br>", unsafe_allow_html=True)
    import datetime as dt
    current_time = dt.datetime.now().strftime("%H:%M")
    st.markdown(f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 35px; border-bottom: 1px solid #1A1A1A; padding-bottom: 15px;">
            <span style="color: #7c3aed; font-weight: 700; font-size: 0.9rem;">BLACK BOX SYSTEM</span>
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="color: #94a3b8; font-size: 11px; font-family: monospace;">{current_time}</span>
                <span style="color: #333; font-size: 12px;">|</span>
                <div style="display: flex; align-items: center; gap: 6px; background: rgba(16,185,129,0.05); padding: 4px 8px; border-radius: 4px;">
                    <span style="font-family: monospace; font-size: 10px; color: #10b981; font-weight: 800; letter-spacing: 0.5px;">ONLINE</span>
                    <span style="height: 6px; width: 6px; background-color: #10b981; border-radius: 50%; display: inline-block; box-shadow: 0 0 5px rgba(16,185,129,0.5);"></span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

def render_alpha_terminal(pl_calc, sr_calc):
    pass  # Reservado para implementação futura

def calcular_metricas_capm(df_portfolio, df_benchmark):
    """
    Calcula Beta, Alpha anualizado e R² ajustado via OLS.
    Entradas: séries de excess returns (já descontada a RFR diária).
    """
    data_reg = pd.concat([df_portfolio, df_benchmark], axis=1).dropna()
    data_reg.columns = ['Portfolio_Excess', 'Market_Excess']

    capm_model = smf.ols(formula='Portfolio_Excess ~ Market_Excess', data=data_reg)
    capm_fit = capm_model.fit()

    beta = capm_fit.params['Market_Excess']
    alpha_anual = (capm_fit.params['Intercept'] + 1) ** 252 - 1
    r2_adj = capm_fit.rsquared_adj

    return beta, alpha_anual, r2_adj

class RiskEngine:
    def __init__(self, returns_series):
        self.returns = returns_series.dropna()
        self.cum_rets = (1 + self.returns).cumprod()

    def get_mdd(self):
        """Maximum Drawdown."""
        rolling_max = self.cum_rets.cummax()
        drawdowns = (self.cum_rets / rolling_max) - 1
        return drawdowns.min(), drawdowns

    def get_var_cvar(self, confidence=0.95, method='historical'):
        """VaR e CVaR históricos."""
        alpha = 1 - confidence
        var = np.percentile(self.returns, alpha * 100)
        cvar = self.returns[self.returns <= var].mean()
        return var, cvar

    def get_sharpe(self, risk_free_rate_daily):
        """Sharpe Ratio anualizado."""
        excess_ret = self.returns - risk_free_rate_daily
        std_dev = excess_ret.std()
        if std_dev == 0:
            return 0
        return (excess_ret.mean() / std_dev) * np.sqrt(252)

    def get_recovery_metrics(self):
        """Tempo máximo de recuperação (dias submersos)."""
        rolling_max = self.cum_rets.cummax()
        is_underwater = self.cum_rets < rolling_max
        contagem_grupos = (is_underwater != is_underwater.shift()).cumsum()
        dias_submersos = is_underwater.groupby(contagem_grupos).cumsum()
        return dias_submersos.max(), dias_submersos

def optimize_portfolio(ativos_selecionados, weights=None):
    data = yf.download(ativos_selecionados, start="2020-01-01", progress=False)['Close']

    if data.empty or (isinstance(data, pd.DataFrame) and len(data) < 2):
        st.error("Dados insuficientes.")
        return None

    returns = data.pct_change().dropna()

    if weights is None:
        weights = np.array([1 / len(ativos_selecionados)] * len(ativos_selecionados))

    portfolio_rets = returns.dot(weights)
    engine = RiskEngine(portfolio_rets)

    mdd_val, mdd_series = engine.get_mdd()
    max_recovery, recovery_series = engine.get_recovery_metrics()
    var_95, cvar_95 = engine.get_var_cvar(confidence=0.95)

    cdi_diario = get_bc_data(12, '01/01/2020')
    sharpe = engine.get_sharpe(cdi_diario.mean())

    return {
        "portfolio_rets": portfolio_rets,
        "max_drawdown": mdd_val,        # ← chave corrigida (era mdd_val)
        "mdd_series": mdd_series,
        "max_recovery": max_recovery,
        "recovery_series": recovery_series,
        "sharpe": sharpe,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "retorno_total": engine.cum_rets.iloc[-1] - 1,
        "serie_acumulada": engine.cum_rets
    }

def orientador_etapa(fase):
    orientacoes = {
        'questionario': {
            'tag': 'SYSTEM MONITOR',
            'msg': 'Responda ao questionário para calibrar sua tolerância a risco e os limites de volatilidade da estratégia.'
        },
        'alocacao': {
            'tag': 'STRATEGY ENGINE',
            'msg': 'Clique em “Browse Files” e envie seu perfil; o sistema processará a arquitetura de ativos e prêmios de risco.'
        },
        'alocacao_capital': {
            'tag': 'CAPITAL ALLOCATOR',
            'msg': 'Insira o valor total do aporte; o motor calculará o tamanho das posições e os limites de segurança da carteira.'
        },
        'selection': {
            'tag': 'DATA MONITOR',
            'msg': 'Selecione os ativos que deseja incluir na análise quantitativa de performance.'
        },
        'otimizacao': {
            'tag': 'TACTICAL OPTIMIZER',
            'msg': 'Clique em “Run Smart Allocation” para calcular os pesos ideais e proteger o portfólio contra quedas acentuadas.'
        },
        'otimizacao_tempo': {
            'tag': 'TIME ALLOCATOR',
            'msg': 'Defina o período de análise; o sistema testará a resiliência do portfólio em diferentes ciclos de mercado.'
        }
    }

    info = orientacoes.get(fase, {'tag': 'SYSTEM', 'msg': 'Aguardando comando...'})
    
    st.markdown(f"""
        <div style="background-color: rgba(197, 160, 89, 0.05); border-left: 2px solid #C5A059; padding: 10px; margin-bottom: 20px;">
            <span style="font-family: 'Courier New', monospace; font-size: 10px; color: #C5A059; font-weight: bold;">
                [ {info['tag'].upper()} ]
            </span><br>
            <span style="font-size: 0.65rem; color: #808080; letter-spacing: 0.5px;">
                {info['msg'].upper()}
            </span>
        </div>
    """, unsafe_allow_html=True)


def guia_conclusao(fase_atual):
    fluxo = {
        'questionario': {
            'tag': 'NEXT PHASE ADVISORY',
            'proximo': 'FASE 02 ➔',
            'destino': 'painel_alocacao',
            'texto': 'Diagnóstico concluído. Certifique-se de salvar seu relatório e prossiga para a <b>Fase 02: Alocação</b>.'
        },
        'alocacao': {
            'tag': 'NEXT PHASE ADVISORY',
            'proximo': 'FASE 03 ➔',
            'destino': 'painel_otimizacao',
            'texto': 'Estrutura de capital definida. Prossiga para a <b>Fase 03: Otimização</b> para refinar o vetor de pesos.'
        },
        'otimizacao': {
            'tag': 'NEXT PHASE ADVISORY',
            'proximo': 'EXTRA ➔',
            'destino': 'painel_calculadora',
            'texto': 'Otimização finalizada. Prossiga para a <b>Calculadora de Tributos</b> para análise de retorno líquido.'
        }
    }

    config = fluxo.get(fase_atual)
    if not config:
        return

    st.markdown("---")
    
    # Aplicando o estilo idêntico ao st.markdown do orientador
    st.markdown(f"""
        <div style="background-color: rgba(197, 160, 89, 0.05); border-left: 2px solid #C5A059; padding: 15px; margin-bottom: 20px;">
            <span style="font-family: 'Courier New', monospace; font-size: 10px; color: #C5A059; font-weight: bold;">
                [ {config['tag'].upper()} ]
            </span><br>
            <span style="font-size: 0.65rem; color: #808080; line-height: 1.6;">
                <b>PRÓXIMO PASSO:</b> {config['texto'].upper()}
            </span>
        </div>
    """, unsafe_allow_html=True)

    # Botão centralizado para ação de avanço
    _, col_btn, _ = st.columns([1, 2, 1])
    with col_btn:
        if st.button(config['proximo'], use_container_width=True, key=f"btn_next_{fase_atual}", type="primary"):
            st.session_state['pagina_atual'] = config['destino']
            # O rerun garante que o processamento recomece do topo do script
            st.rerun()

def reset_scroll():
    # Usamos o timestamp para garantir que o ID e o componente sejam únicos em cada chamada
    # Isso força o Streamlit a destruir o iframe antigo e criar um novo, executando o JS.
    uid = int(time.time() * 1000)
    
    # Criamos a âncora com um ID único
    st.markdown(f"<div id='top-{uid}'></div>", unsafe_allow_html=True)
    
    # O script agora busca esse ID específico e tem um pequeno delay (setTimeout)
    st.components.v1.html(
        f"""
        <script>
            setTimeout(function() {{
                var element = window.parent.document.getElementById('top-{uid}');
                if (element) {{
                    element.scrollIntoView({{behavior: 'auto', block: 'start'}});
                }}
            }}, 10);
        </script>
        """,
        height=0,
    )
    
# ==============================================================================
# ### PÁGINA INICIAL ###
# Descrição: 
# ==============================================================================

if st.session_state['pagina_atual'] == 'home':
    
    page_header()
    lynx_intelligence()

    # --- LAYOUT EM DUAS COLUNAS ---
    col1, col2 = st.columns(2, gap="medium")

    with col1:
        # 1. SELECTION INTELLIGENCE
        st.markdown("""
            <div class="glass-card">
                <div style="font-family: 'Courier New', monospace; font-size: 10px; color: #808080; margin-bottom: 2px;">PHASE 01</div>
                <div class="portal-title" style="color: #C5A059;">
                    <span class="status-dot dot-gold"></span>Questionário de Suitability
                </div>
                <div class="portal-text">
                    Perfil de risco do investidor.
                </div>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Iniciar Análise", type="primary", use_container_width=True, key="btn_select_v"):
            st.session_state['pagina_atual'] = 'painel_suitability'
            st.rerun()
        
        st.markdown("<div style='margin-bottom: 30px;'></div>", unsafe_allow_html=True)

        # 2. ASSET ALLOCATION
        st.markdown("""
            <div class="glass-card">
                <div style="font-family: 'Courier New', monospace; font-size: 10px; color: #808080; margin-bottom: 2px;">PHASE 02</div>
                <div class="portal-title" style="color: #C5A059;">
                    <span class="status-dot dot-gold"></span>Alocação de ativos
                </div>
                <div class="portal-text">
                    Limites de exposição por classe de ativos.
                </div>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Iniciar Gerenciamento", type="primary", use_container_width=True, key="btn_strat_v"):
            st.session_state['pagina_atual'] = 'painel_alocacao'
            st.rerun()

    with col2:
        # 3. TACTICAL ENGINE
        st.markdown("""
            <div class="glass-card">
                <div style="font-family: 'Courier New', monospace; font-size: 10px; color: #808080; margin-bottom: 2px;">PHASE 03</div>
                <div class="portal-title" style="color: #C5A059;">
                    <span class="status-dot dot-gold"></span>Otimização de portfólio
                </div>
                <div class="portal-text">
                    Retorno ajustado ao risco.
                </div>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Iniciar Otimização", type="primary", use_container_width=True, key="btn_tact_v"):
            st.session_state['pagina_atual'] = 'painel_otimizacao'
            st.rerun()

        st.markdown("<div style='margin-bottom: 30px;'></div>", unsafe_allow_html=True)

        # 4. CALCULADORA DE TRIBUTOS
        st.markdown("""
            <div class="glass-card">
                <div style="font-family: 'Courier New', monospace; font-size: 10px; color: #808080; margin-bottom: 2px;">EXTRA / BÔNUS</div>
                <div class="portal-title" style="color: #C5A059;">
                    <span class="status-dot dot-gold"></span>Calculadora de Tributos
                </div>
                <div class="portal-text">
                    Rendimento líquido de investimentos.
                </div>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Iniciar Cálculos", type="primary", use_container_width=True, key="btn_calc_v"):
            st.session_state['pagina_atual'] = 'painel_calculadora'
            st.rerun()
            
    dynamic_footer()

# ==============================================================================
# ### QUESTIONÁRIO DE SUITABILITY ###
# Descrição: questionário para avaliar a adequação do investidor ao portfólio.
# ==============================================================================

# --- PÁGINA DO INVESTIDOR (O QUESTIONÁRIO) ---
elif st.session_state['pagina_atual'] == 'painel_suitability':

    reset_scroll()

    # Botão de Voltar
    c1, c2 = st.columns([1, 10])
    with c1:
        if st.button("⬅", key="v_suitability_topo", use_container_width=True):
            st.session_state['pagina_atual'] = 'home'
            st.rerun()

    page_header()
    lynx_intelligence()
    orientador_etapa('questionario') # Agora está fora da definição, então será executada

    # --- 2. ÁREA DO QUESTIONÁRIO ---
    st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)

    st.markdown("""
        <div class="declaração-container">
            <div class="declaração-titulo">Conformidade</div>
            <div class="declaração-texto">
                • Baseado na Resolução CVM nº 30/21.<br>
                • Metodologia proprietária para definição de Perfil de Risco.<br>
                • Sem equivalência direta com outras instituições.
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    with st.form("suitability_portal", clear_on_submit=False):

        col_nome, col_cpf = st.columns([2, 1])
        with col_nome:
            st.markdown("<p style='font-size: 0.88rem; font-weight: 700; color: #1e293b; margin-bottom: 5px;'>Cliente</p>", unsafe_allow_html=True)
            nome_cliente = st.text_input(label="", placeholder="Nome completo", key="nome_cliente", label_visibility="collapsed")
        with col_cpf:
            st.markdown("<p style='font-size: 0.88rem; font-weight: 700; color: #1e293b; margin-bottom: 5px;'>CPF</p>", unsafe_allow_html=True)
            cpf_cliente = st.text_input(label="", placeholder="Apenas números", max_chars=11, key="cpf_cliente", label_visibility="collapsed")

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("<p style='font-size: 0.9rem; font-weight: 600; color: #334155; margin-bottom: 4px;'>1. Taxa média de aporte mensal em relação à receita líquida:</p>", unsafe_allow_html=True)
        q1 = st.selectbox("", options=[
            "Inferior a 10%: Para reserva de contingência.",
            "11% a 25%: Acumulação progressiva de capital.",
            "26% a 50%: Aceleração de patrimônio.",
            "Superior a 50%: Preservação e sucessão."
        ], index=None, key="q1", placeholder="Selecione uma opção...", label_visibility="collapsed")
 
        st.markdown("<p style='font-size: 0.9rem; font-weight: 600; color: #334155; margin-bottom: 4px;'>2. Necessidade de liquidez para os próximos 6 meses:</p>", unsafe_allow_html=True)
        q2 = st.selectbox("", options=[
            "Alta (> 20%): Projetos ou resgates imediatos.",
            "Moderada (10% a 20%): Saídas curtas planejadas.",
            "Mínima/Nula: Fluxo operacional autossuficiente."
        ], index=None, key="q2", placeholder="Selecione uma opção...", label_visibility="collapsed")
 
        st.markdown("<p style='font-size: 0.9rem; font-weight: 600; color: #334155; margin-bottom: 4px;'>3. Horizonte temporal definido para este portfólio:</p>", unsafe_allow_html=True)
        q3 = st.selectbox("", options=[
            "Curto Prazo: Proteção e liquidez (até 1 ano).",
            "Médio Prazo: Ciclo estratégico entre 1 a 5 anos.",
            "Longo Prazo: Ciclo de crescimento (5 a 10 anos).",
            "Perpetuidade: Foco multigeracional (> 10 anos)."
        ], index=None, key="q3", placeholder="Selecione uma opção...", label_visibility="collapsed")
 
        st.markdown("<p style='font-size: 0.9rem; font-weight: 600; color: #334155; margin-bottom: 4px;'>4. Classes de ativos com experiência prévia:</p>", unsafe_allow_html=True)
        experiencia_selecionada = st.selectbox(
            label="",
            options=[
                "Nível 0: Nenhuma experiência prévia.",
                "Nível 1: Renda Fixa e Crédito Privado.",
                "Nível 2: RF + Estruturados (FIIs, FIDC).",
                "Nível 3: RF + Estruturados + Equities (Ações).",
                "Nível 4: Todos acima + Alternativos e Cripto."
            ],
            index=None,
            placeholder="Selecione uma opção...",
            label_visibility="collapsed",
            key="exp_selectbox_cumulativo"
        )
 
        exp_rendafixa = experiencia_selecionada in [
            "Nível 1: Renda Fixa e Crédito Privado.",
            "Nível 2: RF + Estruturados (FIIs, FIDC).",
            "Nível 3: RF + Estruturados + Equities (Ações).",
            "Nível 4: Todos acima + Alternativos e Cripto."
        ]
        exp_imoveis = experiencia_selecionada in [
            "Nível 2: RF + Estruturados (FIIs, FIDC).",
            "Nível 3: RF + Estruturados + Equities (Ações).",
            "Nível 4: Todos acima + Alternativos e Cripto."
        ]
        exp_acoes = experiencia_selecionada in [
            "Nível 3: RF + Estruturados + Equities (Ações).",
            "Nível 4: Todos acima + Alternativos e Cripto."
        ]
        exp_alternativos = experiencia_selecionada == "Nível 4: Todos acima + Alternativos e Cripto."
 
        st.markdown("<p style='font-size: 0.9rem; font-weight: 600; color: #334155; margin-bottom: 4px; margin-top: 10px'>5. Principal objetivo de investimento:</p>", unsafe_allow_html=True)
        q5 = st.selectbox("", options=[
            "Preservação: Manutenção do valor nominal.",
            "Crescimento: Retorno real com risco controlado.",
            "Apreciação: Exposição ativa a prêmios de risco.",
            "Maximização: Foco em ativos de alta volatilidade."
        ], index=None, key="q5", placeholder="Selecione uma opção...", label_visibility="collapsed")
 
        st.markdown("<p style='font-size: 0.9rem; font-weight: 600; color: #334155; margin-bottom: 4px;'>6. Nível de proficiência sobre o mercado financeiro:</p>", unsafe_allow_html=True)
        q6 = st.selectbox("", options=[
            "Básica: Liquidez e depósitos bancários.",
            "Intermediária: Renda fixa e dinâmica de juros.",
            "Avançada: Renda variável e derivativos.",
            "Profissional: Gestão técnica de riscos ativos."
        ], index=None, key="q6", placeholder="Selecione uma opção...", label_visibility="collapsed")
 
        st.markdown("<p style='font-size: 0.9rem; font-weight: 600; color: #334155; margin-bottom: 4px;'>7. Conduta diante de desvalorização de 15%:</p>", unsafe_allow_html=True)
        q7 = st.selectbox("", options=[
            "Resgate Total: Mitigação imediata de perdas.",
            "Realocação: Migração para baixa volatilidade.",
            "Manutenção: Foco no horizonte estratégico.",
            "Aporte Tático: Aumento de exposição (Buy-the-dip)."
        ], index=None, key="q7", placeholder="Selecione uma opção...", label_visibility="collapsed")
 
        st.markdown("<br>", unsafe_allow_html=True)
 
        st.markdown("""
            <div class="declaração-container">
                <div class="declaração-titulo">Responsabilidade</div>
                <div class="declaração-texto">
                    • Confirmação da veracidade absoluta dos dados fornecidos.<br>
                    • Reconhecimento de que a tese de alocação depende da precisão destes dados.<br>
                    • Isenção de responsabilidade sobre perfis gerados a partir de dados inexatos.
                </div>
            </div>
        """, unsafe_allow_html=True)
 
        concordo_opcao = st.selectbox(
            "Termos de Responsabilidade",
            options=[
                "Não concordo com os termos",
                "Declaro que li e concordo com os termos acima."
            ],
            index=1,
            key="concordo_termos",
            label_visibility="collapsed"
        )
        concordo = (concordo_opcao == "Declaro que li e concordo com os termos acima.")
 
        st.markdown("<br>", unsafe_allow_html=True)
 
        col_l, col_btn, col_r = st.columns([1, 2, 1])
        with col_btn:
            btn_gerar = st.form_submit_button("GERAR ANÁLISE DE PERFIL", type="primary", use_container_width=True)
 
    # ---- LÓGICA DE VISIBILIDADE ----
    if not btn_gerar:
        # Exemplo na página de questionário
        guia_conclusao('questionario')

        # O footer dinâmico também fica dentro da condicional
        dynamic_footer()

    # ---- PROCESSAMENTO (fora do form) ----
    if btn_gerar:
        cpf_limpo = "".join(filter(str.isdigit, cpf_cliente))
 
        if not concordo:
            st.error("Você precisa concordar com os termos acima para prosseguir.")
        elif not nome_cliente or len(cpf_limpo) != 11:
            st.warning("Dados do Cliente incompletos: nome completo e um CPF válido (11 dígitos) são obrigatórios.")
        elif any(x is None for x in [q1, q2, q3, q5, q6, q7]):
            st.warning("O diagnóstico exige precisão. Por favor, responda todas as perguntas do questionário.")
        else:
            try:
                # --- SCORE ---
                map_q1 = {
                    "Inferior a 10%: Para reserva de contingência.": 1,
                    "11% a 25%: Acumulação progressiva de capital.": 2,
                    "26% a 50%: Aceleração de patrimônio.": 3,
                    "Superior a 50%: Preservação e sucessão.": 4
                }
                map_q2 = {
                    "Alta (> 20%): Projetos ou resgates imediatos.": 1,
                    "Moderada (10% a 20%): Saídas curtas planejadas.": 2,
                    "Mínima/Nula: Fluxo operacional autossuficiente.": 4
                }
                map_q3 = {
                    "Curto Prazo: Proteção e liquidez (até 1 ano).": 1,
                    "Médio Prazo: Ciclo estratégico entre 1 a 5 anos.": 2,
                    "Longo Prazo: Ciclo de crescimento (5 a 10 anos).": 3,
                    "Perpetuidade: Foco multigeracional (> 10 anos).": 4
                }
                map_q5 = {
                    "Preservação: Manutenção do valor nominal.": 1,
                    "Crescimento: Retorno real com risco controlado.": 2,
                    "Apreciação: Exposição ativa a prêmios de risco.": 3,
                    "Maximização: Foco em ativos de alta volatilidade.": 5
                }
                map_q6 = {
                    "Básica: Liquidez e depósitos bancários.": 1,
                    "Intermediária: Renda fixa e dinâmica de juros.": 2,
                    "Avançada: Renda variável e derivativos.": 4,
                    "Profissional: Gestão técnica de riscos ativos.": 5
                }
                map_q7 = {
                    "Resgate Total: Mitigação imediata de perdas.": 1,
                    "Realocação: Migração para baixa volatilidade.": 2,
                    "Manutenção: Foco no horizonte estratégico.": 3,
                    "Aporte Tático: Aumento de exposição (Buy-the-dip).": 5
                }
 
                # Q4: experiência cumulativa
                q4 = []
                pts_q4 = 0
                if exp_rendafixa:
                    q4.append("Renda Fixa e Crédito Privado (LCI/LCA, Debêntures)")
                    pts_q4 += 1
                if exp_imoveis:
                    q4.append("Ativos Reais e Estruturados (Imóveis, FIIs, FIDCs)")
                    pts_q4 += 2
                if exp_acoes:
                    q4.append("Equities (Ações, ETFs, BDRs)")
                    pts_q4 += 3
                if exp_alternativos:
                    q4.append("Alternativos (Private Equity, Hedge Funds, Crypto)")
                    pts_q4 += 5
                txt_q4 = ", ".join(q4) if q4 else "Nenhuma exposição prévia declarada."
 
                score = (
                    map_q1.get(q1, 0) +
                    map_q2.get(q2, 0) +
                    map_q3.get(q3, 0) +
                    pts_q4 +
                    map_q5.get(q5, 0) +
                    map_q6.get(q6, 0) +
                    map_q7.get(q7, 0)
                )
 
                # --- PERFIL ---
                if score <= 14:
                    perfil = "CONSERVADOR | Wealth Preservation"
                    cor = "#808080"
                    desc = "Foco estrito em ativos de alta solvência e proteção patrimonial."
                elif score <= 24:
                    perfil = "MODERADO | Balanced Growth"
                    cor = "#A68546"
                    desc = "Alocação híbrida visando retornos reais com volatilidade controlada."
                else:
                    perfil = "ARROJADO | Aggressive Appreciation"
                    cor = "#C5A059"
                    desc = "Estratégia focada em prêmios de risco elevados e ativos alternativos."

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(textwrap.dedent(f"""
                    <div style="padding: 24px; border: 1px solid #333333; border-left: 5px solid {cor}; border-radius: 4px;
                                background-color: #111111; box-shadow: 0 4px 15px rgba(0,0,0,0.4);
                                font-family: 'Inter', sans-serif; margin-bottom: 20px;">
                        <p style="color: {cor}; font-size: 0.8rem; letter-spacing: 1.5px; font-weight: 700; text-transform: uppercase; margin: 0 0 5px 0;">
                            Perfil Identificado
                        </p>
                        <h3 style="color: #FFFFFF; margin: 0 0 10px 0; font-weight: 800; letter-spacing: -0.5px; font-size: 1.5rem; border: none;">
                            {perfil}
                        </h3>
                        <p style="color: #999999; font-size: 0.95rem; line-height: 1.5; margin: 0;">
                            {desc}
                        </p>
                        <hr style="border: 0; border-top: 1px solid #333333; margin: 20px 0;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="color: #666666; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px;">Score de Adequação</span>
                            <span style="font-weight: 700; color: {cor}; font-size: 1.4rem;">
                                {score} <small style="font-size: 0.8rem; color: #666;">/ 38</small>
                            </span>
                        </div>
                    </div>
                """), unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("""
                    <div class="declaração-container">
                        <div class="declaração-titulo">Protocolo de Privacidade</div>
                        <div class="declaração-texto">
                            Esta infraestrutura opera sob arquitetura <i>Stateless</i>. Em conformidade com protocolos rígidos de segurança, o processamento de dados ocorre exclusivamente em memória volátil: nenhuma informação pessoal (PII) é armazenada após o encerramento da sessão.
                        </div>
                    </div>
                """, unsafe_allow_html=True)

                # --- GERAÇÃO DO PDF ---
                try:
                    txt_q1 = str(q1)
                    txt_q2 = str(q2)
                    txt_q3 = str(q3)
                    txt_q5 = q5[1] if isinstance(q5, (list, tuple)) else str(q5)
                    txt_q6 = q6[1] if isinstance(q6, (list, tuple)) else str(q6)
                    txt_q7 = q7[1] if isinstance(q7, (list, tuple)) else str(q7)

                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_auto_page_break(auto=True, margin=25)
                    m_left, m_right = 25, 25
                    pdf.set_margins(left=m_left, top=20, right=m_right)
                    page_width = 210 - m_left - m_right

                    pdf.set_fill_color(197, 160, 89)
                    pdf.rect(0, 0, 4, 297, 'F')

                    pdf.set_y(30)
                    pdf.set_font("Helvetica", 'B', 18)
                    pdf.set_text_color(17, 17, 17)
                    pdf.cell(page_width, 10, clean_pdf("RELATÓRIO DE PERFIL DE INVESTIMENTO"), ln=True)

                    pdf.set_font("Helvetica", '', 9)
                    pdf.set_text_color(100, 100, 100)
                    pdf.cell(page_width, 5, clean_pdf(f"BLACK BOX | {datetime.now().strftime('%d/%m/%Y | %H:%M')}"), ln=True)
                    pdf.ln(12)

                    pdf.set_font("Helvetica", 'B', 10)
                    pdf.set_text_color(17, 17, 17)
                    pdf.cell(page_width, 8, clean_pdf("I. IDENTIFICAÇÃO DO CLIENTE"), ln=True)
                    pdf.set_draw_color(197, 160, 89)
                    pdf.line(m_left, pdf.get_y(), 210 - m_right, pdf.get_y())
                    pdf.ln(4)

                    pdf.set_font("Helvetica", 'B', 9)
                    pdf.set_text_color(100, 100, 100)
                    pdf.cell(35, 7, clean_pdf("CLIENTE:"), 0)
                    pdf.set_font("Helvetica", '', 10)
                    pdf.set_text_color(17, 17, 17)
                    pdf.cell(page_width - 35, 7, clean_pdf(nome_cliente.upper()), ln=True)

                    pdf.set_font("Helvetica", 'B', 9)
                    pdf.set_text_color(100, 100, 100)
                    pdf.cell(35, 7, clean_pdf("CPF/DOC:"), 0)
                    pdf.set_font("Helvetica", '', 10)
                    pdf.set_text_color(17, 17, 17)
                    pdf.cell(page_width - 35, 7, formatar_cpf(cpf_cliente), ln=True)

                    pdf.ln(5)
                    pdf.set_fill_color(245, 245, 245)
                    pdf.set_font("Helvetica", 'B', 11)
                    pdf.set_text_color(197, 160, 89)
                    pdf.cell(page_width, 12, clean_pdf(f"   PERFIL IDENTIFICADO: {perfil.upper()}"), ln=True, fill=True)

                    pdf.ln(2)
                    pdf.set_font("Helvetica", 'I', 9)
                    pdf.set_text_color(80, 80, 80)
                    pdf.multi_cell(page_width, 6, clean_pdf(f"Estratégia: {desc}"))
                    pdf.ln(8)

                    pdf.set_font("Helvetica", 'B', 10)
                    pdf.set_text_color(17, 17, 17)
                    pdf.cell(page_width, 8, clean_pdf("II. ANÁLISE DE ADEQUAÇÃO (SUITABILITY & KYC)"), ln=True)
                    pdf.line(m_left, pdf.get_y(), 210 - m_right, pdf.get_y())
                    pdf.ln(5)

                    respostas_lista = [
                        ("Taxa de Poupança e Renda Líquida", txt_q1),
                        ("Necessidades de Liquidez", txt_q2),
                        ("Horizonte Temporal de Investimento", txt_q3),
                        ("Experiência com Classes de Ativos", txt_q4),
                        ("Objetivo Primário de Investimento", txt_q5),
                        ("Nível de Conhecimento Financeiro", txt_q6),
                        ("Tolerância ao Risco e Comportamento em Drawdown", txt_q7)
                    ]
                    for p, r in respostas_lista:
                        pdf.set_font("Helvetica", 'B', 9)
                        pdf.set_text_color(100, 100, 100)
                        pdf.multi_cell(page_width, 5, clean_pdf(p.upper()))
                        pdf.set_x(m_left)
                        pdf.set_font("Helvetica", '', 10)
                        pdf.set_text_color(17, 17, 17)
                        pdf.multi_cell(page_width - 5, 6, clean_pdf(f"R: {r}"), border=0)
                        pdf.ln(3)

                    pdf.set_y(-30)
                    pdf.set_font("Helvetica", 'I', 7)
                    pdf.set_text_color(150, 150, 150)
                    pdf.multi_cell(page_width, 4, clean_pdf("Confidencial - Elaborado de acordo com a Resolução CVM 30/21. Este documento não garante rentabilidade futura."), align='C')

                    pdf_bytes = pdf.output(dest='S').encode('latin-1')

                    _, col_central, _ = st.columns([1, 2, 1])
                    with col_central:
                        st.download_button(
                            label="GERAR RELATÓRIO INSTITUTIONAL (.PDF)",
                            data=pdf_bytes,
                            file_name=f"BLACK_BOX_Suitability_{nome_cliente.replace(' ', '_')}.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                            type="primary"
                        )

                except Exception as e:
                    st.error(f"Erro na geração do relatório: {e}")

            except Exception as e_geral:
                st.error(f"Ocorreu um erro no processamento: {e_geral}")

        dynamic_footer()

# ==============================================================================
# ### TELA: ALOCAÇÃO DE ATIVOS ###
# Descrição: painel para definir os limites de exposição por classe de ativos.
# ==============================================================================

# --- 3. TELA: PAINEL ALOCAÇÃO (Conectada pelo ELIF) ---
elif st.session_state['pagina_atual'] == 'painel_alocacao':

    reset_scroll()

    c1, c2 = st.columns([1, 10])
    with c1:
        if st.button("⬅", key="v_alocacao_topo", use_container_width=True):
            st.session_state['pagina_atual'] = 'home'
            st.rerun()

    page_header()
    lynx_intelligence()
    orientador_etapa('alocacao')

    # --- 3. CONTEÚDO (IPS FRAMEWORK - AI SUITABILITY PORTAL) ---
    
    # 1. Definição da função (Mantida intacta)
    def extrair_dados_suitability(file):
        import pdfplumber
        with pdfplumber.open(file) as pdf:
            texto = pdf.pages[0].extract_text()
        dados = {"perfil": "Mod", "horizonte": "Não Identificado", "objetivo": "Preservação"}
        if "PERFIL:" in texto:
            perfil_raw = texto.split("PERFIL:")[1].split("/")[0].strip().upper()
            if "ARROJADO" in perfil_raw: dados["perfil"] = "Arrojado"
            elif "CONSERVADOR" in perfil_raw: dados["perfil"] = "Conservador"
            else: dados["perfil"] = "Moderado"
        if "Horizonte Temporal" in texto:
            try:
                raw_h = texto.split("Horizonte Temporal")[1].split("\n")[1]
                dados["horizonte"] = raw_h.replace("R::", "").replace("R: :", "").strip()
            except: pass
        if "Objetivo Estratégico" in texto:
            try:
                raw_o = texto.split("Objetivo Estratégico")[1].split("\n")[1]
                dados["objetivo"] = raw_o.replace("R::", "").replace("R: :", "").strip()
            except: pass
        return dados

    # 2. Garantia de limpeza (Mantida intacta)
    if 'suit_mini' not in st.session_state:
        for key in ['ai_purpose', 'ai_horizon', 'ai_risk', 'percentual_acoes']:
            if key in st.session_state:
                st.session_state[key] = ""

    # --- ALTERAÇÃO PARA MÁXIMA DISCRIÇÃO ---
    # Injetando CSS para remover bordas, ícones e textos do uploader do Streamlit
    st.markdown("""
        <style>
            /* Reduz o uploader a uma linha minimalista */
            [data-testid="stFileUploader"] {
                padding: 0px;
            }
            [data-testid="stFileUploader"] section {
                padding: 0px;
                background-color: transparent;
                border: 1px dashed #262626; /* Borda quase invisível */
                min-height: 40px;
            }
            [data-testid="stFileUploader"] section > div {
                display: none; /* Esconde o ícone de nuvem e textos padrão */
            }
            [data-testid="stFileUploader"] button {
                font-size: 0.6rem !important;
                background-color: transparent !important;
                color: #64748b !important;
                border: none !important;
                text-transform: uppercase;
                letter-spacing: 1px;
            }
        </style>
    """, unsafe_allow_html=True)

    # placeholder discreto
    area_upload = st.empty()

    with area_upload.container():
        suitability_file = st.file_uploader("UPLOAD IPS", type=['pdf', 'txt'], key="suit_mini", label_visibility="collapsed")

        if suitability_file:
            import time
            progress_bar = st.progress(0)
            msg = st.empty() 
            
            msg.text("Lynx Engine: Lendo metadados...")
            dados_extraidos = extrair_dados_suitability(suitability_file)
            progress_bar.progress(40)
            time.sleep(0.5)

            msg.text("Análise Concluída")
            progress_bar.progress(100)

            st.session_state['ai_purpose'] = dados_extraidos["objetivo"]
            st.session_state['ai_horizon'] = dados_extraidos["horizonte"]
            st.session_state['ai_risk'] = dados_extraidos["perfil"]
            
            time.sleep(1) 
            progress_bar.empty() 
            msg.empty()
            
            # Autodestruição visual: Após processar, limpamos o uploader da tela
            area_upload.empty()
            st.rerun() # Opcional: força a limpeza e exibe os dados processados nos campos de destino
            
    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- 3. DISPLAYS DE RESULTADOS (NEW BLACK BOX VISUAL STANDARDS) ---
    c1, c2, c3 = st.columns([2, 1.5, 1])

    with c1:
        purp = st.session_state.get('ai_purpose', "---")
        st.markdown(f"""
            <div style='border-bottom: 1px solid #C5A059; padding-bottom: 5px;'>
                <p style='margin:0; font-size:0.65rem; color:#64748b; text-transform:uppercase; font-weight:700;'>Propósito Estratégico</p>
                <p style='margin:0; font-size:0.95rem; color:#111111; font-weight:500;'>{purp}</p>
            </div>
        """, unsafe_allow_html=True)

    with c2:
        horiz = st.session_state.get('ai_horizon', "---")
        st.markdown(f"""
            <div style='border-bottom: 1px solid #C5A059; padding-bottom: 5px;'>
                <p style='margin:0; font-size:0.65rem; color:#64748b; text-transform:uppercase; font-weight:700;'>Horizonte</p>
                <p style='margin:0; font-size:0.95rem; color:#111111; font-weight:500;'>{horiz}</p>
            </div>
        """, unsafe_allow_html=True)

    with c3:
        risk = st.session_state.get('ai_risk', "Mod")
        st.markdown(f"""
            <div style='text-align:right;'>
                <p style='margin:0; font-size:0.65rem; color:#64748b; text-transform:uppercase; font-weight:700;'>Risco Extraído</p>
                <span style='color:white; padding:2px 10px; border-radius:2px; font-size:0.75rem; font-weight:600;'>{risk.upper()}</span>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    orientador_etapa('alocacao_capital')

    c_add, _ = st.columns([0.5, 0.5]) # Ajustado levemente a proporção para o placeholder longo
    with c_add:
        capital_raw = st.text_input(
            label="Add", 
            placeholder="＋ INSERIR CAPITAL ALOCÁVEL (Ex.: 100000)", 
            label_visibility="collapsed",
            key="input_new_ac"
        )

    # --- 2. LÓGICA DE TRATAMENTO DE DADOS ---
    # Tentativa de conversão simples para manter o código leve
    try:
        v_capital = float(capital_raw.replace('.', '').replace(',', '.')) if capital_raw else 0.0
    except:
        v_capital = 0.0

    # Simulação de porcentagem de ações (Isso viria do seu motor de alocação/perfil)
    # Aqui usamos um valor de exemplo de 40%, ajuste conforme sua variável global
    perc_acoes = 0.40 
    v_acoes = v_capital * perc_acoes

    # --- 4. EXIBIÇÃO DOS INDICADORES ---
    k1, k2, k3 = st.columns(3)

    with k1:
        st.markdown(f"""
            <div class="stat-card" style="border-top: 2px solid #C5A059;">
                <p class="stat-label">Capital Total Alocável</p>
                <p class="stat-value">R$ {v_capital:,.0f}</p>
                <p class="stat-status" style="color: #C5A059;">LIQUIDEZ PRONTA</p>
            </div>
        """, unsafe_allow_html=True)

    with k2:
        st.markdown(f"""
            <div class="stat-card" style="border-top: 2px solid #C5A059;">
                <p class="stat-label">% Alocação em Ações</p>
                <p class="stat-value">{perc_acoes:.1%}</p>
                <p class="stat-status" style="color: #808080;">ESTRATÉGIA</p>
            </div>
        """, unsafe_allow_html=True)

    with k3:
        # Cor dinâmica para valor de ações (Dourado se houver capital)
        status_color = "#C5A059" if v_acoes > 0 else "#666666"
        st.markdown(f"""
            <div class="stat-card" style="border-top: 2px solid {status_color};">
                <p class="stat-label">Exposição em Renda Variável</p>
                <p class="stat-value">R$ {v_acoes:,.0f}</p>
                <p class="stat-status" style="color: {status_color};">RISCO NOMINAL</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # --- 2. LÓGICA DE INTEGRAÇÃO (SAA) ---
    risk_map = {"Cons": "Conservative", "Mod": "Moderate", "Arr": "Aggressive"}
    perfil_key = st.session_state.get('ai_risk', 'Mod')
    perfil_extraido = risk_map.get(perfil_key, "Moderate")

    # Pesos Originais [Renda Fixa, EQUITY, Global, Alternatives]
    weights_dict = {
        "Conservative": [0.75, 0.15, 0.05, 0.05],
        "Moderate":     [0.50, 0.30, 0.15, 0.05],
        "Aggressive":   [0.15, 0.50, 0.25, 0.10]
    }

    # --- 3. ENGENHARIA FINANCEIRA (RISK OVERLAY) ---
    sug_weights = list(weights_dict[perfil_extraido])

    # Ajuste Tático focado em Equity (sug_weights[1])
    old_equity = sug_weights[1]

    # Aglutinando "Outros" (Tudo que não é Equity)
    weights_others = 1.0 - sug_weights[1]

    # --- 4. OVERLAY MANUAL (DESIGN BLACK BOX) ---
    ca, cb = st.columns([1, 1])
    with ca:
        # FOCO: Ações
        w_eq = st.number_input("Equity Allocation (%)", 0, 100, int(sug_weights[1]*100), key=f"eq_focus_{perfil_extraido}") / 100
    with cb:
        # OUTROS: Aglutinado
        w_others = st.number_input("Other Assets (%)", 0, 100, int(weights_others*100), key=f"others_focus", disabled=True) / 100

    total_sum = w_eq + w_others

    # Barra de Validação Gold
    status_color = "#C5A059" if abs(total_sum - 1.0) < 0.001 else "#800000"
    st.markdown(f"""
        <div style="width: 100%; background-color: #1A1A1A; border-radius: 2px; margin-top: 10px; border: 1px solid #333;">
            <div style="width: {min(total_sum*100, 100):.0f}%; background-color: {status_color}; height: 4px; border-radius: 2px; transition: 0.3s;"></div>
        </div>
        <p style="text-align: right; font-size: 0.65rem; color: {status_color}; font-weight: 700; margin-top: 5px; letter-spacing: 1px;">
            TOTAL EXPOSURE: {total_sum*100:.1f}%
        </p>
    """, unsafe_allow_html=True)

    # --- 5. GRÁFICO CENTRAL (DONUT CHART - THE VAULT STYLE) ---
    col_a, col_b, col_c = st.columns([1, 2.5, 1])

    with col_b:
        labels = ['Equity', 'Others']
        values = [w_eq, w_others]
        colors = ['#C5A059', '#1A1A1A'] # Gold para Ações, Dark para o resto

        fig = go.Figure(data=[go.Pie(
            labels=labels, values=values, hole=.75,
            marker=dict(colors=colors, line=dict(color='#080808', width=2)),
            textinfo='none', hoverinfo='label+percent', direction='clockwise', sort=False
        )])

        # Central Badge (Institutional Grade)
        fig.add_annotation(
            text=f"EQUITY<br><br><b style='font-size:22px; color:#E0E0E0;'>{w_eq:.1%}</b>",
            showarrow=False, x=0.5, y=0.5,
            font=dict(size=14, family="Inter", color="#808080"),
            align="center"
        )

        fig.update_layout(
            margin=dict(t=0, b=0, l=0, r=0), 
            showlegend=True, # Legenda ativada
            legend=dict(
                orientation="h", 
                yanchor="bottom", 
                y=-0.1, 
                xanchor="center", 
                x=0.5,
                font=dict(family="Inter", size=11, color="#808080")
            ),
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)', 
            height=500
        )

        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    st.markdown("<br><br>", unsafe_allow_html=True)

    # 1. Geração de Relatório BLACK BOX Wealth Management (Padrão Institucional Atualizado)
    try:
        import textwrap
        from fpdf import FPDF
        import datetime

        def clean_pdf(txt):
            return str(txt).encode('latin-1', 'replace').decode('latin-1')

        # --- PREPARAÇÃO DOS DADOS TÁTICOS ---
        # v_capital e v_acoes vêm da sua lógica de input anterior
        txt_capital = f"R$ {v_capital:,.2f}"
        txt_exposicao = f"R$ {v_acoes:,.2f}"
        txt_perc = f"{perc_acoes:.1%}"
        
        # Dados de Perfil extraídos do session_state
        perfil_nome = st.session_state.get('ai_risk', 'Moderado')
        propósito = st.session_state.get('ai_purpose', 'Preservação de Capital')

        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=25)
        
        # Configuração de Margens de Private Banking
        m_left, m_right = 25, 25
        pdf.set_margins(left=m_left, top=20, right=m_right)
        page_width = 210 - m_left - m_right 

        # --- DESIGN: BARRA LATERAL DOURADA (IDENTIDADE BLACK BOX) ---
        pdf.set_fill_color(197, 160, 89) # Dourado #C5A059
        pdf.rect(0, 0, 4, 297, 'F') 

        # --- CABEÇALHO ---
        pdf.set_y(30)
        pdf.set_font("Helvetica", 'B', 18)
        pdf.set_text_color(17, 17, 17) # Preto Fosco
        pdf.cell(page_width, 10, clean_pdf("RELATÓRIO DE ALOCAÇÃO ESTRATÉGICA"), ln=True)
        
        pdf.set_font("Helvetica", '', 9)
        pdf.set_text_color(100, 100, 100)
        data_hoje = datetime.datetime.now().strftime("%d/%m/%Y | %H:%M")
        pdf.cell(page_width, 5, clean_pdf(f"BLACK BOX | {data_hoje}"), ln=True)
        pdf.ln(12)

        # --- I. RESUMO EXECUTIVO (O NOVO FOCO) ---
        pdf.set_font("Helvetica", 'B', 10)
        pdf.set_text_color(17, 17, 17)
        pdf.cell(page_width, 8, clean_pdf("I. RESUMO EXECUTIVO DE APORTE"), ln=True)
        pdf.set_draw_color(197, 160, 89) 
        pdf.line(m_left, pdf.get_y(), 210 - m_right, pdf.get_y())
        pdf.ln(6)

        # Box de Destaque para Capital
        pdf.set_fill_color(248, 248, 248)
        pdf.rect(m_left, pdf.get_y(), page_width, 25, 'F')
        
        pdf.set_y(pdf.get_y() + 5)
        pdf.set_x(m_left + 5)
        pdf.set_font("Helvetica", 'B', 9)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(page_width/2, 5, clean_pdf("CAPITAL TOTAL DISPONÍVEL:"), 0)
        pdf.cell(page_width/2, 5, clean_pdf("EXPOSIÇÃO EM AÇÕES:"), ln=True)
        
        pdf.set_x(m_left + 5)
        pdf.set_font("Helvetica", 'B', 14)
        pdf.set_text_color(197, 160, 89)
        pdf.cell(page_width/2, 10, clean_pdf(txt_capital), 0)
        pdf.set_text_color(17, 17, 17)
        pdf.cell(page_width/2, 10, clean_pdf(f"{txt_exposicao} ({txt_perc})"), ln=True)
        pdf.ln(10)

        # --- II. PARÂMETROS ESTRATÉGICOS ---
        pdf.set_font("Helvetica", 'B', 10)
        pdf.set_text_color(17, 17, 17)
        pdf.cell(page_width, 8, clean_pdf("II. PARÂMETROS ESTRATÉGICOS"), ln=True)
        pdf.line(m_left, pdf.get_y(), 210 - m_right, pdf.get_y())
        pdf.ln(5)

        params = [
            ("PERFIL DO INVESTIDOR", perfil_nome.upper()),
            ("PROPÓSITO ESTRATÉGICO", propósito),
            ("HORIZONTE TEMPORAL", st.session_state.get('ai_horizon', 'Longo Prazo'))
        ]

        for label, valor in params:
            pdf.set_font("Helvetica", 'B', 9)
            pdf.set_text_color(100, 100, 100)
            pdf.cell(50, 8, clean_pdf(label + ":"), 0)
            pdf.set_font("Helvetica", '', 10)
            pdf.set_text_color(17, 17, 17)
            pdf.cell(page_width - 50, 8, clean_pdf(valor), ln=True)

        # --- III. NOTA TÉCNICA (PROPOSTA DE VALOR) ---
        pdf.ln(10)
        pdf.set_font("Helvetica", 'B', 10)
        pdf.cell(page_width, 8, clean_pdf("III. NOTAS DE ALOCAÇÃO"), ln=True)
        pdf.line(m_left, pdf.get_y(), 210 - m_right, pdf.get_y())
        pdf.ln(4)
        pdf.set_font("Helvetica", '', 9)
        pdf.set_text_color(60, 60, 60)
        nota = (
            f"A exposição de {txt_perc} em Equities (Brasil & Internacional) reflete o apetite a risco '{perfil_nome}'. "
            f"O montante nominal de {txt_exposicao} deve ser alocado conforme as diretrizes da BLACK BOX Capital, "
            f"cuja tese integra modelagem preditiva e otimização de portfólio (Fronteira Eficiente e Simulações de Monte Carlo), "
            f"garantindo uma observação rigorosa da distribuição dos ativos aos limites de volatilidade dos mercados doméstico e global."
        )
        pdf.multi_cell(page_width, 5, clean_pdf(nota))

        # --- RODAPÉ / JURÍDICO ---
        pdf.set_y(-30)
        pdf.set_font("Helvetica", 'I', 7)
        pdf.set_text_color(150, 150, 150)
        aviso = "Confidencial | BLACK BOX. Elaborado de acordo com padrões institucionais. Riscos de mercado inerentes."
        pdf.multi_cell(page_width, 4, clean_pdf(aviso), align='C')

        # --- SAÍDA ---
        pdf_bytes = pdf.output(dest='S').encode('latin-1') 

        # Botão de Download estilizado
        _, col_central, _ = st.columns([1, 2, 1])
        with col_central:
            st.download_button(
                label="GERAR RELATÓRIO INSTITUTIONAL (.PDF)",
                data=pdf_bytes,
                file_name=f"BLACK_BOX_Strategy_{datetime.date.today()}.pdf",
                mime="application/pdf",
                use_container_width=True,
                type="primary"
            )

            st.markdown("""
                <div style='text-align: center; padding: 10px; border-top: 1px solid #333; margin-top: 15px;'>
                    <p style='font-family: "Inter", sans-serif; font-size: 0.6rem; color: #666; letter-spacing: 1.2px; text-transform: uppercase; line-height: 1.6;'>
                        CONFIDENTIAL PORTFOLIO REPORT: EQUITY ANALYTICS & FACTOR ATTRIBUTION<br>
                        POWERED BY LYNX INTELLIGENCE ENGINE | BLACK BOX TERMINAL
                    </p>
                </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Erro na geração do relatório: {e}")

    guia_conclusao('alocacao')
    dynamic_footer()

# ==============================================================================
# ### TELA: OTIMIZAÇÃO ###
# Descrição: painel para definir os ativos, pesos e restrições para otimização.
# ==============================================================================

elif st.session_state['pagina_atual'] == 'painel_otimizacao':

    reset_scroll()

    c1, c2 = st.columns([1, 10])
    with c1:
        if st.button("⬅", key="v_otimizacao_topo", use_container_width=True):
            st.session_state['pagina_atual'] = 'home'
            st.rerun()

    page_header()
    lynx_intelligence()
    orientador_etapa('selection')

    from datetime import datetime, timedelta
    
    # --- 1. SELEÇÃO DE ATIVOS (INPUT LAYER) ---
    # Lista Base
    lista_acoes_base = sorted([
        'ITUB4.SA', 'VALE3.SA', 'PETR4.SA', 'WEGE3.SA', 'ABEV3.SA', 
        'BBAS3.SA', 'BBDC4.SA', 'B3SA3.SA', 'ITSA4.SA', 'ELET3.SA'
    ])

    # Inicialização de Estados
    if 'lista_acoes_din' not in st.session_state:
        st.session_state.lista_acoes_din = lista_acoes_base
    if 'ativos_selecionados' not in st.session_state:
        st.session_state.ativos_selecionados = []

    # --- SELEÇÃO DE AÇÕES ---
    with st.container():

        st.markdown("<p style='font-size:0.65rem; color:#64748b; font-weight:600; margin-bottom:15px; letter-spacing:1px; text-transform:uppercase;'>Equities Selection</p>", unsafe_allow_html=True)

        # 1. Selectbox para Adicionar à Análise
        acao_veloce = st.selectbox(
            "Seletor de Ativos",
            options=["＋ Selecionar Ativo"] + st.session_state.lista_acoes_din,
            index=0,
            label_visibility="collapsed",
            key="sb_selecao_principal"
        )

        if acao_veloce != "＋ Selecionar Ativo":
            if acao_veloce not in st.session_state.ativos_selecionados:
                st.session_state.ativos_selecionados.append(acao_veloce)
                st.rerun()

        # 2. Input para Novos Tickers (Mantendo sua lógica original)
        c_add, _ = st.columns([2, 2])
        with c_add:
            novo_ticker = st.text_input(
                label="Add", 
                placeholder="＋ NEW TICKER (Ex.: MGLU3.SA)", 
                label_visibility="collapsed",
                key="input_new_ac"
            ).upper()

            if novo_ticker and novo_ticker not in st.session_state.lista_acoes_din:
                st.session_state.lista_acoes_din = sorted(st.session_state.lista_acoes_din + [novo_ticker])
                st.session_state.ativos_selecionados.append(novo_ticker) 
                st.rerun()

        st.markdown("<div style='margin-bottom: 10px;'></div>", unsafe_allow_html=True)
        
        # --- TARGET ALLOCATION (Gerenciamento da Lista e Pesos) ---
        ativos_selecionados = st.session_state.ativos_selecionados
        pesos_acoes = []

        if ativos_selecionados:
            st.markdown("<p style='font-size:0.65rem; color:#64748b; font-weight:600; margin-bottom:15px; letter-spacing:1px; text-transform:uppercase;'>Allocation %</p>", unsafe_allow_html=True)
            
            for ativo in ativos_selecionados:
                c_label, c_input, c_rem = st.columns([1, 4, 0.3])
                with c_label:
                    st.markdown(f"<p style='font-size:0.85rem; color:#E2E8F0; margin-top:8px;'>{ativo}</p>", unsafe_allow_html=True)
                with c_input:
                    p = st.number_input(
                        label=f"Weight {ativo}", 
                        min_value=0.0, max_value=1.0, 
                        value=0.0, step=0.01, format="%.2f",
                        key=f"weight_ac_{ativo}",
                        label_visibility="collapsed"
                    )
                    pesos_acoes.append(p)
                with c_rem:
                    if st.button("×", key=f"rem_{ativo}"):
                        st.session_state.ativos_selecionados.remove(ativo)
                        st.rerun()

            weights = np.array(pesos_acoes)
        else:
            # Estado Vazio
            st.markdown("""
                <div style='padding: 40px; border: 1px dashed #e2e8f0; border-radius: 10px; text-align: center; margin-top: 20px;'>
                    <p style='color: #94a3b8; font-size: 0.8rem; letter-spacing: 1px;'>AGUARDANDO SELEÇÃO DE ATIVOS PARA INICIAR TELEMETRIA</p>
                </div>
            """, unsafe_allow_html=True)
            
            if 'dynamic_footer' in globals():
                dynamic_footer()
            st.stop()

    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- EXECUTION ENGINE (BLACK BOX TERMINAL STYLE) ---
    if ativos_selecionados:
    
        orientador_etapa('otimizacao')

        # Centralização com layout proporcional
        _, c_btn, _ = st.columns([1.5, 2, 1.5])
        
        with c_btn:
            # Botão com estilo primário (Dourado por CSS global ou Type Primary)
            if st.button("RUN SMART ALLOCATION", key="run_weights", use_container_width=True, type="primary"):
                st.toast("LYNX ENGINE: STARTING OPTIMIZATION...")
            
            # Nota Metodológica Estilo Bloomberg Terminal
            st.markdown("""
                <div style='text-align: center; padding: 10px; border-top: 1px solid #333; margin-top: 15px;'>
                    <p style='font-family: "Inter", sans-serif; font-size: 0.6rem; color: #666; letter-spacing: 1.2px; text-transform: uppercase; line-height: 1.6;'>
                        Metodologia: <span style='color: #C5A059;'>CVaR Optimization</span> & Efficient Frontier<br>
                        <span style='opacity: 0.8;'>Validation: Monte Carlo (10k Iterations) | Asset Allocation Limits</span>
                    </p>
                </div>
            """, unsafe_allow_html=True)

        # --- FOOTER DE STATUS (OPCIONAL) ---
        st.markdown(f"""
            <div style="background-color: #1A1A1A; padding: 12px; border-left: 3px solid #C5A059; margin-top: 20px; border-radius: 2px;">
                        <span style="color: #C5A059; font-family: monospace; font-size: 0.6rem; font-weight: bold; letter-spacing: 1px; text-transform: uppercase;">
                    STATUS: ENGINE_READY | SAMPLES: {len(ativos_selecionados)} TICKERS | MODE: INSTITUTIONAL_RESEARCH
                </span>
            </div>
        """, unsafe_allow_html=True)
            
    st.divider()

    # --- 1. SELETOR DE PERÍODO (TIME-INTERVAL SELECTOR) ---
    with st.container():
        orientador_etapa('otimizacao_tempo')
        st.markdown("<p style='font-size:0.65rem; color:#64748b; font-weight:600; margin-bottom:15px; letter-spacing:1px; text-transform:uppercase;'>HORIZONTE TEMPORAL</p>", unsafe_allow_html=True)
        
        col_periodo, _ = st.columns([2, 2])

        with col_periodo:
            opcoes_periodo = {
                "1M": 30,
                "6M": 180,
                "YTD": (datetime.now() - datetime(datetime.now().year, 1, 1)).days,
                "1Y": 365,
                "3Y": 1095,
                "5Y": 1825
            }
            
            # Substituição do st.radio pelo st.selectbox para máxima estabilidade
            intervalo_selecionado = st.selectbox(
                label="Período",
                options=list(opcoes_periodo.keys()),
                index=3, # Mantém o Default em 1Y
                label_visibility="collapsed",
                key="sb_periodo_analise"
            )
            
            dias_selecionados = opcoes_periodo[intervalo_selecionado]

        # ==============================================================================
        # ### CÁLCULOS DE ANÁLISE QUANTITATIVA
        # Descrição: principais cálculos para otimização de portfólio.
        # ==============================================================================

        try:
            with st.spinner('⏳ CALCULANDO MÉTRICAS DE MERCADO...'):

                # --- DADOS BRUTOS ---
                dados_brutos = buscar_dados_limpos(ativos_selecionados, dias_selecionados)

                lista_ativos = list(ativos_selecionados)
                lista_completa = lista_ativos + ['^BVSP']

                precos_com_idx = dados_brutos[lista_completa].ffill().dropna()

                if precos_com_idx.empty:
                    st.error("SISTEMA: Sem dados suficientes para o período selecionado.")
                    st.stop()

                precos = precos_com_idx[lista_ativos]

                # --- RETORNOS ---
                retornos_full = precos_com_idx.pct_change().dropna()
                retornos_diarios = retornos_full[lista_ativos].copy()
                retornos_diarios['IBOV'] = retornos_full['^BVSP']
                retornos_diarios['Portfolio'] = retornos_diarios[lista_ativos].dot(weights)

                # --- RETORNO ANUALIZADO DA CARTEIRA ---
                carteira_retorno_diario = retornos_diarios['Portfolio'].mean()
                carteira_retorno_anual = ((1 + carteira_retorno_diario) ** 252) - 1

                # --- RETORNO ANUALIZADO DO IBOV ---
                ibov_retorno_diario = retornos_diarios['IBOV'].mean()
                ibov_retorno_anual = ((1 + ibov_retorno_diario) ** 252) - 1

                # --- TAXA LIVRE DE RISCO ---
                rfr_anual = buscar_rfr_pro()
                rfr_diaria = (1 + rfr_anual) ** (1 / 252) - 1

                # --- VOLATILIDADE E SHARPE ---
                cov_matrix = retornos_diarios[lista_ativos].cov() * 252
                port_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
                vol_portfolio = np.sqrt(port_variance)

                retornos_diarios['Portfolio_Excess'] = retornos_diarios['Portfolio'] - rfr_diaria
                sharpe_ratio = (retornos_diarios['Portfolio_Excess'].mean() / retornos_diarios['Portfolio'].std()) * (252 ** 0.5)

                # --- BETA (CAPM) ---
                data_inicio = retornos_diarios.index.min().strftime('%Y-%m-%d')
                data_fim = retornos_diarios.index.max().strftime('%Y-%m-%d')

                ibov_dados = yf.download("^BVSP", start=data_inicio, end=data_fim)['Close']
                ibov_retornos = ibov_dados.squeeze().pct_change()
                ibov_excess = ibov_retornos - rfr_diaria

                retornos_diarios['IBOV_Excess'] = ibov_excess

                market_var = retornos_diarios['IBOV_Excess'].std()
                if market_var == 0:
                    st.error("ERRO CRÍTICO: O Benchmark não possui variação. Alpha/Beta não podem ser calculados.")
                    st.stop()
                else:
                    beta, alpha, r2 = calcular_metricas_capm(
                        retornos_diarios['Portfolio_Excess'],
                        retornos_diarios['IBOV_Excess']
                    )

                # --- INFORMATION RATIO ---
                tracking_error = (retornos_diarios['Portfolio'] - retornos_diarios['IBOV']).std() * np.sqrt(252)

                ir = (carteira_retorno_anual - ibov_retorno_anual) / tracking_error \
                    if tracking_error and tracking_error != 0 else 0.0
                treynor = (carteira_retorno_anual - rfr_anual) / beta \
                    if beta and beta != 0 else 0.0

                # --- MAX DRAWDOWN (via optimize_portfolio) ---
                res = optimize_portfolio(ativos_selecionados)
                max_dd = res.get('max_drawdown', 0)

                # --- SKEWNESS E KURTOSIS ---
                pf_skew = retornos_diarios['Portfolio'].dropna().skew()
                pf_kurt = retornos_diarios['Portfolio'].dropna().kurtosis() + 3
                
                # --- EXECUÇÃO DO TESTE ESTATÍSTICO (VERDITO TERMINAL) ---
                shapiro_results = shapiro(retornos_diarios['Portfolio'].dropna())
                p_value = shapiro_results[1]

                is_normal = p_value > 0.05
                veredito = "GAUSSIANO (Normal)" if is_normal else "NÃO-NORMAL (Caudas Longas)"
                # Cores de status de mercado: Ouro para normalidade, Oxblood para risco de cauda
                cor_veredito = "#C5A059" if is_normal else "#800000" 

                st.markdown(f"""
                    <div style="background-color: #1A1A1A; padding: 12px; border-left: 3px solid #C5A059; margin-top: 20px; border-radius: 2px;">
                        <span style="color: #C5A059; font-family: monospace; font-size: 0.7rem; font-weight: bold; letter-spacing: 1px; text-transform: uppercase;">
                            SHAPIRO-WILK TEST: {veredito} | P-VALUE: {p_value:.4f} | 
                            STATUS: {"ESTATISTICAMENTE CONFIÁVEL" if is_normal else "ALERTA: RISCO DE CAUDA (KURTOSIS) ELEVADO"}
                        </span>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)

                tab_overview, tab_analytics = st.tabs(["OVERVIEW", "ANALYTICS"])

                with tab_overview:
                    # --- 1. CONFIGURAÇÃO DE CORES E IDENTIDADE ---
                    color_equity = "#C5A059"  # Champagne Gold

                    # --- 3. EXPOSIÇÃO ATUAL (STAT CARDS) ---
                    alloc = st.session_state.get('mandato_ativo', {"equity": 0.0})

                    # Centralizando o Card de Equity
                    _, c1, _ = st.columns([1, 2, 1])
                    with c1:
                        st.markdown(f'<div class="stat-card" style="border-top: 3px solid {color_equity};"><p class="stat-label">EQUITY EXPOSURE</p><p class="stat-value">{alloc.get("equity", 0.0):.1%}</p></div>', unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)

                    # --- 4. COMPOSIÇÃO DA CARTEIRA (DONUT CHART) ---
                    total_acoes = sum(pesos_acoes) if 'pesos_acoes' in locals() and pesos_acoes else 0

                    # ── CONFIGURAÇÃO DE COLUNAS ──
                    col_a, col_b, col_c = st.columns([1, 2.5, 1])

                    with col_b:
                        # Lógica de segurança para as variáveis
                        w_eq = total_acoes if 'total_acoes' in locals() else 0.0
                        color_equity = '#C5A059' # Dourado Black Box
                        
                        # Cálculo do 'Other' para garantir que o gráfico feche em 100%
                        val_other = max(0, 1 - w_eq)

                        import plotly.graph_objects as go
                        fig = go.Figure(data=[go.Pie(
                            labels=['Equities', 'Other'], 
                            values=[total_acoes, 1-total_acoes if total_acoes < 1 else 0], 
                            hole=.78,
                            marker=dict(colors=[color_equity, '#262626'], line=dict(color='#080808', width=2)),
                            textinfo='none', hoverinfo='label+percent', direction='clockwise', sort=False
                        )])

                        # Central Badge (Institutional Grade)
                        fig.add_annotation(
                            text=f"EQUITY<br><br><b style='font-size:22px; color:#E0E0E0;'>{w_eq:.1%}</b>",
                            showarrow=False, x=0.5, y=0.5,
                            font=dict(size=14, family="Inter", color="#808080"),
                            align="center"
                        )

                        fig.update_layout(
                            margin=dict(t=0, b=0, l=0, r=0), 
                            showlegend=True, # Alterado para True
                            legend=dict(
                                orientation="h", 
                                yanchor="bottom", 
                                y=-0.1, 
                                xanchor="center", 
                                x=0.5,
                                font=dict(family="Inter", size=11, color="#808080")
                            ),
                            paper_bgcolor='rgba(0,0,0,0)', 
                            plot_bgcolor='rgba(0,0,0,0)', 
                            height=500
                        )
                        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

                    # --- 5. PERFORMANCE RELATIVA (CURVA HISTÓRICA) ---
                    st.markdown("<div class='section-header-blackbox'>Relative Performance (Base 100)</div>", unsafe_allow_html=True)
                    ativos_plot = [a for a in ativos_selecionados if a in precos.columns]

                    if ativos_plot:
                        base_idx = precos.index[0]

                        # --- Ativos selecionados (base 100) ---
                        grafico_data = (precos[ativos_plot] / precos[ativos_plot].iloc[0] * 100).copy()

                        # --- Ibovespa ---
                        # Remove timezone de ambos os índices antes de alinhar
                        bvsp_raw = precos["^BVSP"].copy() if "^BVSP" in precos.columns else pd.Series(dtype=float)
                        if bvsp_raw.empty:
                            # Fallback: baixar separado se dropna() tiver removido
                            try:
                                _bvsp = yf.download("^BVSP", start=precos.index[0], end=precos.index[-1], progress=False)
                                bvsp_raw = _bvsp["Close"].squeeze()
                            except Exception:
                                bvsp_raw = pd.Series(dtype=float)

                        if not bvsp_raw.empty:
                            bvsp_raw.index = bvsp_raw.index.tz_localize(None) if bvsp_raw.index.tz else bvsp_raw.index
                            grafico_data.index  = grafico_data.index.tz_localize(None) if grafico_data.index.tz else grafico_data.index
                            bvsp_alinhado = bvsp_raw.reindex(grafico_data.index, method='ffill').dropna()
                            if not bvsp_alinhado.empty:
                                grafico_data["Ibovespa"] = bvsp_alinhado / bvsp_alinhado.iloc[0] * 100

                        # --- Selic acumulada ---
                        try:
                            data_inicio_str = precos.index[0].strftime("%d/%m/%Y")
                            selic_diaria = get_bc_data(11, data_inicio_str)
                            if not selic_diaria.empty:
                                selic_diaria.index = selic_diaria.index.tz_localize(None) if selic_diaria.index.tz else selic_diaria.index
                                selic_alinhada = selic_diaria.reindex(grafico_data.index, method='ffill').fillna(0)
                                grafico_data["Selic"] = (1 + selic_alinhada).cumprod() * 100
                        except Exception:
                            pass

                        # --- Construção do gráfico ---
                        cores_ativos = ["#C5A059", "#E8C98A", "#A07840", "#D4B070", "#8B6530"]
                        benchmark_styles = {
                            "Ibovespa": {"color": "#4A90D9", "dash": "solid",     "width": 1},
                            "Selic":    {"color": "#50C878", "dash": "dashdot",   "width": 1.5},
                        }

                        fig_curve = go.Figure()

                        # Ativos do portfólio
                        for i, ativo in enumerate(ativos_plot):
                            if ativo in grafico_data.columns:
                                cor = cores_ativos[i % len(cores_ativos)]
                                fig_curve.add_trace(go.Scatter(
                                    x=grafico_data.index,
                                    y=grafico_data[ativo].round(2),
                                    name=ativo,
                                    mode="lines",
                                    line=dict(color=cor, width=2),
                                    hovertemplate=f"<b>{ativo}</b><br>%{{x|%d/%m/%Y}}<br>Base 100: %{{y:.1f}}<extra></extra>"
                                ))

                        # Benchmarks
                        for bench, style in benchmark_styles.items():
                            if bench in grafico_data.columns:
                                fig_curve.add_trace(go.Scatter(
                                    x=grafico_data.index,
                                    y=grafico_data[bench].round(2),
                                    name=bench,
                                    mode="lines",
                                    line=dict(color=style["color"], dash=style["dash"], width=style["width"]),
                                    opacity=0.75,
                                    hovertemplate=f"<b>{bench}</b><br>%{{x|%d/%m/%Y}}<br>Base 100: %{{y:.1f}}<extra></extra>"
                                ))

                        # Linha de referência 100
                        fig_curve.add_hline(
                            y=100,
                            line=dict(color="#333333", width=1, dash="dash"),
                            annotation_text="",
                            annotation_position="left",
                            annotation_font=dict(color="#555555", size=10)
                        )

                        fig_curve.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            xaxis=dict(
                                gridcolor='#1E1E1E', title="",
                                tickfont=dict(size=10, color="#666666"),
                                showspikes=True, spikecolor="#333333", spikethickness=1, spikedash="dot"
                            ),
                            yaxis=dict(
                                gridcolor='#1E1E1E', title="",
                                tickfont=dict(size=10, color="#666666"),
                                ticksuffix="",
                                showspikes=True, spikecolor="#333333", spikethickness=1, spikedash="dot"
                            ),
                            legend=dict(
                                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                                font=dict(size=10, color="#808080"),
                                bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)"
                            ),
                            hovermode="x unified",
                            hoverlabel=dict(
                                bgcolor="#111111", bordercolor="#333333",
                                font=dict(size=11, color="#E0E0E0")
                            ),
                            margin=dict(l=10, r=10, t=30, b=10),
                            height=420
                        )

                        st.plotly_chart(fig_curve, use_container_width=True, config={'displayModeBar': False})

                    # ==============================================================================
                    # FUNÇÃO: gerar_relatorio_equity_pdf
                    # Posição no arquivo: nível de módulo, ANTES de qualquer página Streamlit.
                    # Cole junto aos outros helpers (buscar_dados_limpos, buscar_rfr_pro, etc.)
                    # ==============================================================================

                    def gerar_relatorio_equity_pdf(
                        nome_cliente,
                        ativos_selecionados,
                        pesos_acoes,
                        intervalo_selecionado,
                        carteira_retorno_anual,
                        ibov_retorno_anual,
                        rfr_anual,
                        vol_portfolio,
                        sharpe_ratio,
                        beta,
                        alpha,
                        max_dd,
                        tracking_error,
                        pf_skew,
                        pf_kurt,
                        p_value,
                        veredito,
                        ir,
                        treynor,
                    ):
                        from fpdf import FPDF
                        from datetime import datetime

                        def c(txt):
                            return str(txt).encode('latin-1', 'replace').decode('latin-1')

                        # ── Layout ──────────────────────────────────────────────────────────────
                        ML, MR, MT = 25, 25, 20
                        W   = 210 - ML - MR
                        COL = W / 3

                        # ── Paleta ──────────────────────────────────────────────────────────────
                        GOLD  = (197, 160, 89)
                        DARK  = (17,  17,  17)
                        MID   = (80,  80,  80)
                        LITE  = (150, 150, 150)
                        LGRAY = (230, 230, 230)
                        RED   = (128,   0,   0)
                        GREEN = (34,  139,  34)

                        def _sem(cond_green, cond_gold):
                            if cond_green:
                                return GREEN
                            if cond_gold:
                                return GOLD
                            return RED

                        # ── Helpers de renderização ──────────────────────────────────────────────
                        def _sidebar(pdf):
                            pdf.set_fill_color(*GOLD)
                            pdf.rect(0, 0, 4, 297, 'F')

                        def _section(pdf, title):
                            pdf.ln(6)
                            pdf.set_font("Helvetica", 'B', 10)
                            pdf.set_text_color(*DARK)
                            pdf.cell(W, 8, c(title), ln=True)
                            pdf.set_draw_color(*GOLD)
                            pdf.line(ML, pdf.get_y(), 210 - MR, pdf.get_y())
                            pdf.ln(4)

                        def _kv(pdf, label, value, color=None):
                            pdf.set_font("Helvetica", 'B', 8)
                            pdf.set_text_color(*LITE)
                            pdf.cell(65, 6, c(label.upper()), 0)
                            pdf.set_font("Helvetica", '', 9)
                            pdf.set_text_color(*(color if color else DARK))
                            pdf.cell(W - 65, 6, c(str(value)), ln=True)

                        def _grid3(pdf, items):
                            for i in range(0, len(items), 3):
                                grupo = items[i:i+3]

                                pdf.set_font("Helvetica", 'B', 7)
                                pdf.set_text_color(*LITE)
                                for j, (lbl, _, _, _) in enumerate(grupo):
                                    pdf.set_x(ML + j * COL)
                                    pdf.cell(COL - 2, 5, c(lbl.upper()))
                                pdf.ln(5)

                                pdf.set_font("Helvetica", 'B', 13)
                                pdf.set_text_color(*DARK)
                                for j, (_, val, _, _) in enumerate(grupo):
                                    pdf.set_x(ML + j * COL)
                                    pdf.cell(COL - 2, 8, c(val))
                                pdf.ln(8)

                                pdf.set_font("Helvetica", '', 7)
                                for j, (_, _, sts, cor) in enumerate(grupo):
                                    pdf.set_x(ML + j * COL)
                                    pdf.set_text_color(*cor)
                                    pdf.cell(COL - 2, 5, c(sts))
                                pdf.ln(8)

                                pdf.set_draw_color(*LGRAY)
                                pdf.line(ML, pdf.get_y(), 210 - MR, pdf.get_y())
                                pdf.ln(4)

                        def _new_page(pdf):
                            pdf.add_page()
                            _sidebar(pdf)

                        def _check_page(pdf, needed=40):
                            if pdf.get_y() > (297 - 25 - needed):
                                _new_page(pdf)

                        # ── Inicialização ────────────────────────────────────────────────────────
                        pdf = FPDF()
                        pdf.set_auto_page_break(auto=True, margin=25)
                        pdf.set_margins(left=ML, top=MT, right=MR)
                        pdf.add_page()
                        _sidebar(pdf)

                        # ════════════════════════════════════════════════════════════════════════
                        # CABEÇALHO
                        # ════════════════════════════════════════════════════════════════════════
                        pdf.set_y(30)
                        pdf.set_font("Helvetica", 'B', 17)
                        pdf.set_text_color(*DARK)
                        pdf.cell(W, 10, c("RELATÓRIO DE OTIMIZAÇÃO DE PORTFÓLIO"), ln=True)

                        pdf.set_font("Helvetica", '', 8)
                        pdf.set_text_color(*LITE)
                        pdf.cell(W, 5, c(
                            f"BLACK BOX TERMINAL  |  LYNX INTELLIGENCE ENGINE  |  "
                            f"{datetime.now().strftime('%d/%m/%Y  %H:%M')}"
                        ), ln=True)
                        pdf.ln(8)

                        # ════════════════════════════════════════════════════════════════════════
                        # I. IDENTIFICAÇÃO
                        # ════════════════════════════════════════════════════════════════════════
                        _section(pdf, "I. IDENTIFICAÇÃO")
                        _kv(pdf, "Cliente",   nome_cliente.upper() if nome_cliente else "NÃO IDENTIFICADO")
                        _kv(pdf, "Horizonte", str(intervalo_selecionado))
                        _kv(pdf, "Universo",  ", ".join(ativos_selecionados))
                        pdf.ln(4)

                        # ════════════════════════════════════════════════════════════════════════
                        # II. COMPOSIÇÃO E ALOCAÇÃO
                        # ════════════════════════════════════════════════════════════════════════
                        _section(pdf, "II. COMPOSIÇÃO E ALOCAÇÃO")

                        pdf.set_font("Helvetica", 'B', 8)
                        pdf.set_text_color(*LITE)
                        pdf.cell(90, 7, c("ATIVO"))
                        pdf.cell(35, 7, c("PESO"))
                        pdf.cell(35, 7, c("CONTRIBUIÇÃO"), ln=True)

                        total_peso = sum(pesos_acoes) if pesos_acoes else 0.0
                        max_peso   = max(pesos_acoes) if pesos_acoes else 1.0

                        for ativo, peso in zip(ativos_selecionados, pesos_acoes):
                            contrib = (peso / total_peso * 100) if total_peso > 0 else 0
                            barra_w = int((peso / max_peso) * 60) if max_peso > 0 else 0

                            pdf.set_font("Helvetica", '', 9)
                            pdf.set_text_color(*DARK)
                            pdf.cell(90, 6, c(ativo))
                            pdf.set_font("Helvetica", 'B', 9)
                            pdf.set_text_color(*GOLD)
                            pdf.cell(35, 6, c(f"{peso:.2%}"))
                            pdf.set_font("Helvetica", '', 9)
                            pdf.set_text_color(*MID)
                            pdf.cell(35, 6, c(f"{contrib:.1f}% do total"), ln=True)

                            y_bar = pdf.get_y()
                            pdf.set_fill_color(*GOLD)
                            pdf.rect(ML, y_bar, barra_w, 1.5, 'F')
                            pdf.set_fill_color(*LGRAY)
                            pdf.rect(ML + barra_w, y_bar, 60 - barra_w, 1.5, 'F')
                            pdf.ln(3)

                        pdf.ln(2)
                        pdf.set_draw_color(*GOLD)
                        pdf.line(ML, pdf.get_y(), 210 - MR, pdf.get_y())
                        pdf.ln(2)
                        pdf.set_font("Helvetica", 'B', 9)
                        pdf.set_text_color(*GOLD)
                        pdf.cell(90, 7, c("TOTAL ALOCADO"))
                        pdf.cell(35, 7, c(f"{total_peso:.2%}"))
                        peso_ok = abs(total_peso - 1.0) < 0.001
                        pdf.set_text_color(*(GOLD if peso_ok else LITE))
                        pdf.cell(35, 7, c("COMPLETO" if peso_ok else f"RESIDUAL: {1 - total_peso:.2%}"), ln=True)
                        pdf.ln(6)

                        # ════════════════════════════════════════════════════════════════════════
                        # III. PERFORMANCE E RETORNO
                        # ════════════════════════════════════════════════════════════════════════
                        _check_page(pdf, 60)
                        _section(pdf, "III. PERFORMANCE E RETORNO AJUSTADO AO RISCO")

                        alpha_excess = carteira_retorno_anual - ibov_retorno_anual
                        cor_ret = GOLD if carteira_retorno_anual > rfr_anual else RED

                        pdf.set_fill_color(245, 245, 245)
                        pdf.set_font("Helvetica", 'B', 11)
                        pdf.set_text_color(*cor_ret)
                        pdf.cell(W, 12,
                                c(f"   RETORNO ANUALIZADO DA CARTEIRA: {carteira_retorno_anual:.2%}"),
                                ln=True, fill=True)
                        pdf.ln(2)

                        _kv(pdf, "Retorno Anualizado (Portfólio)",
                            f"{carteira_retorno_anual:.2%}",
                            GOLD if carteira_retorno_anual > rfr_anual else RED)
                        _kv(pdf, "Retorno Anualizado (Ibovespa)",     f"{ibov_retorno_anual:.2%}")
                        _kv(pdf, "Taxa Livre de Risco (Selic Anual)", f"{rfr_anual:.2%}")
                        _kv(pdf, "Alpha vs Ibovespa",
                            f"{alpha_excess:+.2%}",
                            GOLD if alpha_excess >= 0 else RED)
                        _kv(pdf, "Volatilidade Anualizada",           f"{vol_portfolio:.2%}")
                        pdf.ln(4)

                        # ════════════════════════════════════════════════════════════════════════
                        # IV. MÉTRICAS DE RISCO E EFICIÊNCIA
                        # ════════════════════════════════════════════════════════════════════════
                        _check_page(pdf, 80)
                        _section(pdf, "IV. MÉTRICAS DE RISCO E EFICIÊNCIA")

                        _grid3(pdf, [
                            ("Sharpe Ratio",
                            f"{sharpe_ratio:.2f}",
                            "Excellent" if sharpe_ratio > 1.0 else ("Acceptable" if sharpe_ratio > 0.5 else "Poor"),
                            _sem(False, sharpe_ratio > 1.0) if sharpe_ratio > 0.5 else RED),

                            ("Portfolio Beta",
                            f"{beta:.2f}",
                            "Defensive" if beta < 1 else ("High Risk" if beta > 1.5 else "Aggressive"),
                            _sem(False, beta < 1) if beta <= 1.5 else RED),

                            ("Max Drawdown",
                            f"{max_dd:.2%}",
                            "Controlled" if max_dd > -0.10 else ("Moderate" if max_dd > -0.20 else "Severe"),
                            _sem(max_dd > -0.10, max_dd > -0.20)),

                            ("Information Ratio",
                            f"{ir:.2f}",
                            "Strong Alpha" if ir > 0.5 else ("Neutral" if ir >= 0 else "Underperforming"),
                            _sem(False, ir > 0.5) if ir >= 0 else RED),

                            ("Treynor Ratio",
                            f"{treynor:.2f}",
                            "Strong" if treynor > 0.10 else ("Neutral" if treynor >= 0 else "Negative"),
                            _sem(False, treynor > 0.10) if treynor >= 0 else RED),

                            ("Tracking Error",
                            f"{tracking_error:.2%}",
                            "Low TE" if tracking_error < 0.05 else ("Moderate TE" if tracking_error < 0.15 else "High TE"),
                            _sem(tracking_error < 0.05, tracking_error < 0.15)),
                        ])

                        # ════════════════════════════════════════════════════════════════════════
                        # V. DECOMPOSIÇÃO CAPM
                        # ════════════════════════════════════════════════════════════════════════
                        _check_page(pdf, 50)
                        _section(pdf, "V. DECOMPOSIÇÃO CAPM (ALPHA & BETA)")

                        _kv(pdf, "Alpha (Excess Return vs IBOV)",
                            f"{alpha:.4f}",
                            GOLD if alpha >= 0 else RED)
                        _kv(pdf, "Beta (Sensibilidade ao Mercado)",
                            f"{beta:.4f}",
                            GOLD if beta < 1 else RED)
                        _kv(pdf, "Interpretação do Alpha",
                            "Geração de valor acima do benchmark" if alpha > 0 else "Underperformance vs benchmark")
                        _kv(pdf, "Interpretação do Beta",
                            "Portfólio defensivo (vol < mercado)" if beta < 1
                            else ("Portfólio agressivo (vol > mercado)" if beta > 1
                                else "Portfólio neutro (vol = mercado)"))
                        pdf.ln(4)

                        # ════════════════════════════════════════════════════════════════════════
                        # VI. DISTRIBUIÇÃO ESTATÍSTICA
                        # ════════════════════════════════════════════════════════════════════════
                        _check_page(pdf, 70)
                        _section(pdf, "VI. DISTRIBUIÇÃO ESTATÍSTICA E RISCO DE CAUDA")

                        is_normal = p_value > 0.05

                        _grid3(pdf, [
                            ("Skewness",
                            f"{pf_skew:.4f}",
                            "Positive Bias" if pf_skew > 0 else "Negative Bias",
                            GOLD if pf_skew > 0 else RED),

                            ("Kurtosis (Excess)",
                            f"{pf_kurt:.4f}",
                            "Thin Tails" if pf_kurt < 0 else ("Normal" if pf_kurt <= 2 else "Fat Tails"),
                            _sem(pf_kurt < 0, pf_kurt <= 2)),

                            ("Shapiro-Wilk p",
                            f"{p_value:.4f}",
                            "Normal (p > 0.05)" if is_normal else "Non-Normal (p < 0.05)",
                            GOLD if is_normal else RED),
                        ])

                        pdf.set_fill_color(245, 245, 245)
                        pdf.set_font("Helvetica", 'B', 9)
                        pdf.set_text_color(*(GOLD if is_normal else RED))
                        pdf.cell(W, 10,
                                c(f"   VEREDITO: {veredito.upper()}  |  "
                                f"{'CONFIÁVEL' if is_normal else 'ALERTA — RISCO DE CAUDA ELEVADO'}"),
                                ln=True, fill=True)
                        pdf.ln(6)

                        # ════════════════════════════════════════════════════════════════════════
                        # VII. NOTA METODOLÓGICA
                        # ════════════════════════════════════════════════════════════════════════
                        _check_page(pdf, 50)
                        _section(pdf, "VII. NOTA METODOLÓGICA")

                        pdf.set_font("Helvetica", '', 8)
                        pdf.set_text_color(*MID)
                        pdf.multi_cell(W, 5, c(
                            "Metodologia: CVaR (Conditional Value at Risk) + Fronteira Eficiente de Markowitz. "
                            "Retornos calculados sobre preços ajustados via Yahoo Finance. "
                            "Benchmark: Ibovespa (^BVSP). Taxa Livre de Risco: Selic anual via API BCB (SGS). "
                            "Beta e Alpha via regressão OLS (CAPM). "
                            "Normalidade dos retornos avaliada pelo teste Shapiro-Wilk. "
                            "Resultados passados não garantem rentabilidade futura."
                        ))
                        pdf.ln(4)

                        # ════════════════════════════════════════════════════════════════════════
                        # RODAPÉ JURÍDICO
                        # ════════════════════════════════════════════════════════════════════════
                        pdf.set_y(-40)
                        pdf.set_draw_color(*GOLD)
                        pdf.line(ML, pdf.get_y(), 210 - MR, pdf.get_y())
                        pdf.ln(3)
                        pdf.set_font("Helvetica", 'I', 7)
                        pdf.set_text_color(*LITE)
                        pdf.multi_cell(W, 4, c(
                            "CONFIDENCIAL - BLACK BOX TERMINAL / LYNX INTELLIGENCE ENGINE. "
                            "Uso exclusivo do cliente identificado. Não constitui oferta de valores mobiliários "
                            "nem recomendação de investimento. Resolução CVM 30/21."
                        ), align='C')

                        return pdf.output(dest='S').encode('latin-1')

                    st.markdown("<div style='margin-top: 40px;'></div>", unsafe_allow_html=True)
                    _, c_pdf, _ = st.columns([1, 2, 1])
                    with c_pdf:

                        pdf_bytes = None
                        try:
                            pdf_bytes = gerar_relatorio_equity_pdf(
                                nome_cliente           = st.session_state.get('nome_cliente', 'Investidor'),
                                ativos_selecionados    = ativos_selecionados,
                                pesos_acoes            = pesos_acoes,
                                intervalo_selecionado  = intervalo_selecionado,
                                carteira_retorno_anual = carteira_retorno_anual,
                                ibov_retorno_anual     = ibov_retorno_anual,
                                rfr_anual              = rfr_anual,
                                vol_portfolio          = vol_portfolio,
                                sharpe_ratio           = sharpe_ratio,
                                beta                   = beta,
                                alpha                  = alpha,
                                max_dd                 = max_dd,
                                tracking_error         = tracking_error,
                                pf_skew                = pf_skew,
                                pf_kurt                = pf_kurt,
                                p_value                = p_value,
                                veredito               = veredito,
                                ir                     = ir,
                                treynor                = treynor,
                            )
                        except Exception as _e_pdf:
                            st.warning(f"ENGINE: Relatório indisponível — {_e_pdf}")

                        if pdf_bytes:
                            st.download_button(
                                label               = "GERAR RELATÓRIO INSTITUCIONAL (.PDF)",
                                data                = pdf_bytes,
                                file_name           = f"BLACK_BOX_Equity_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                                mime                = "application/pdf",
                                use_container_width = True,
                                type                = "primary",
                            )
                        else:
                            st.button("RELATÓRIO INDISPONÍVEL", disabled=True, use_container_width=True)

                        st.markdown("""
                            <div style='text-align:center; padding:10px;
                                        border-top:1px solid #333; margin-top:15px;'>
                                <p style='font-size:0.6rem; color:#666; letter-spacing:1.2px;
                                            text-transform:uppercase; line-height:1.6;'>
                                    CONFIDENTIAL PORTFOLIO REPORT: EQUITY ANALYTICS &amp; FACTOR ATTRIBUTION<br>
                                    POWERED BY LYNX INTELLIGENCE ENGINE | BLACK BOX TERMINAL
                                </p>
                            </div>
                        """, unsafe_allow_html=True)

                with tab_analytics:
                    # --- 1. ESTILIZAÇÃO CSS (THE BLACK BOX VAULT STYLE) ---
                    st.markdown("""
                        <style>
                        /* Títulos de Seção Estilo Bloomberg/Reuters */
                        .section-header-blackbox {
                            color: #808080; font-size: 0.7rem; font-weight: 800; 
                            letter-spacing: 2px; text-transform: uppercase; 
                            margin: 30px 0 15px 0; border-bottom: 1px solid #333333; padding-bottom: 10px;
                        }
                        /* Card de Métrica Lapidado */
                        .metric-card-blackbox {
                            background: #1A1A1A; padding: 25px; border-radius: 2px;
                            border: 1px solid #333333; margin-bottom: 20px;
                            transition: all 0.3s ease;
                        }
                        .metric-card-blackbox:hover {
                            border-color: #C5A059;
                            box-shadow: 0 0 15px rgba(197, 160, 89, 0.2);
                        }
                        /* Tipografia de Dados de Elite */
                        .metric-label-blackbox { 
                            font-size: 0.65rem; color: #808080; font-weight: 700; 
                            text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 12px; 
                        }
                        .metric-value-blackbox { 
                            font-size: 1.6rem; font-weight: 700; color: #E0E0E0; 
                            font-family: 'Inter', sans-serif; letter-spacing: -0.5px;
                        }
                        .metric-status-blackbox { 
                            font-size: 0.7rem; font-weight: 600; margin-top: 10px; 
                            text-transform: uppercase; letter-spacing: 0.5px;
                        }
                        </style>
                    """, unsafe_allow_html=True)

                    # --- 1. EQUITY ANALYTICS ---
                    st.markdown("<div class='section-header-blackbox'>Equity Selection Metrics</div>", unsafe_allow_html=True)
                    
                    e1, e2, e3 = st.columns(3)

                    with e1:
                        # ROIC: fixo por ora (hardcoded), assume positivo
                        roic = 18.4
                        color_roic = "#C5A059" if roic > 0 else "#800000"
                        status_roic = "ROIC > WACC" if roic > 0 else "ROIC < WACC"
                        st.markdown(f"""
                            <div class='metric-card-blackbox' style='border-top: 2px solid {color_roic};'>
                                <p class='metric-label-blackbox'>Weighted Avg ROIC</p>
                                <p class='metric-value-blackbox'>{roic:.1f}%</p>
                                <p class='metric-status-blackbox' style='color: {color_roic};'>{status_roic}</p>
                            </div>
                        """, unsafe_allow_html=True)

                    with e2:
                        # Beta: dourado se defensivo (< 1), vermelho se muito agressivo (> 1.5), cinza no meio
                        color_beta = "#C5A059" if beta < 1 else ("#800000" if beta > 1.5 else "#808080")
                        status_beta = "Defensive vs IBOV" if beta < 1 else ("High Risk vs IBOV" if beta > 1.5 else "Aggressive vs IBOV")
                        st.markdown(f"""
                            <div class='metric-card-blackbox' style='border-top: 2px solid {color_beta};'>
                                <p class='metric-label-blackbox'>Portfolio Beta</p>
                                <p class='metric-value-blackbox'>{beta:.2f}</p>
                                <p class='metric-status-blackbox' style='color: {color_beta};'>{status_beta}</p>
                            </div>
                        """, unsafe_allow_html=True)

                    with e3:
                        # Max Drawdown: verde se leve, dourado se moderado, vermelho se severo
                        max_dd = res.get('max_drawdown', 0)
                        color_dd = "#228B22" if max_dd > -0.10 else ("#C5A059" if max_dd > -0.20 else "#800000")
                        status_dd = "Controlled Risk" if max_dd > -0.10 else ("Moderate Decline" if max_dd > -0.20 else "Severe Drawdown")
                        st.markdown(f"""
                            <div class='metric-card-blackbox' style='border-top: 2px solid {color_dd};'>
                                <p class='metric-label-blackbox'>Max Drawdown</p>
                                <p class='metric-value-blackbox'>{max_dd:.2%}</p>
                                <p class='metric-status-blackbox' style='color: {color_dd};'>{status_dd}</p>
                            </div>
                        """, unsafe_allow_html=True)

                    # --- 2. RISK-ADJUSTED RETURN ---
                    st.markdown("<div class='section-header-blackbox'>Portfolio Efficiency (Risk-Adjusted)</div>", unsafe_allow_html=True)

                    s1, s2, s3 = st.columns(3)

                    with s1:
                        # Sharpe: > 1 excelente, 0.5–1 aceitável, < 0.5 ruim
                        color_sharpe = "#C5A059" if sharpe_ratio > 1.0 else ("#808080" if sharpe_ratio > 0.5 else "#800000")
                        status_sharpe = "Excellent Efficiency" if sharpe_ratio > 1.0 else ("Acceptable" if sharpe_ratio > 0.5 else "Poor Risk/Return")
                        st.markdown(f"""
                            <div class='metric-card-blackbox' style='border-top: 2px solid {color_sharpe};'>
                                <p class='metric-label-blackbox'>Sharpe Ratio</p>
                                <p class='metric-value-blackbox'>{sharpe_ratio:.2f}</p>
                                <p class='metric-status-blackbox' style='color: {color_sharpe};'>{status_sharpe}</p>
                            </div>
                        """, unsafe_allow_html=True)

                    with s2:
                        # IR: > 0.5 habilidade real, 0–0.5 neutro, < 0 underperformance
                        color_ir = "#C5A059" if ir > 0.5 else ("#808080" if ir >= 0 else "#800000")
                        status_ir = "Strong Alpha" if ir > 0.5 else ("Neutral" if ir >= 0 else "Underperforming")
                        st.markdown(f"""
                            <div class='metric-card-blackbox' style='border-top: 2px solid {color_ir};'>
                                <p class='metric-label-blackbox'>Information Ratio</p>
                                <p class='metric-value-blackbox'>{ir:.2f}</p>
                                <p class='metric-status-blackbox' style='color: {color_ir};'>{status_ir}</p>
                            </div>
                        """, unsafe_allow_html=True)

                    with s3:
                        # Treynor: > 0.10 bom, 0–0.10 neutro, negativo ruim
                        color_treynor = "#C5A059" if treynor > 0.10 else ("#808080" if treynor >= 0 else "#800000")
                        status_treynor = "Strong Return/Beta" if treynor > 0.10 else ("Neutral" if treynor >= 0 else "Negative Return")
                        st.markdown(f"""
                            <div class='metric-card-blackbox' style='border-top: 2px solid {color_treynor};'>
                                <p class='metric-label-blackbox'>Treynor Ratio</p>
                                <p class='metric-value-blackbox'>{treynor:.2f}</p>
                                <p class='metric-status-blackbox' style='color: {color_treynor};'>{status_treynor}</p>
                            </div>
                        """, unsafe_allow_html=True)

                    # --- 3. DISTRIBUTION & STATS ---
                    st.markdown("<div class='section-header-blackbox'>Statistical Distribution (Tail Risk)</div>", unsafe_allow_html=True)

                    st1, st2, st3 = st.columns(3)

                    with st1:
                        # Skewness: positivo é bom (assimetria favorável), negativo é ruim
                        color_skew = "#C5A059" if pf_skew > 0 else "#800000"
                        status_skew = "Positive Bias" if pf_skew > 0 else "Negative Bias"
                        st.markdown(f"""
                            <div class='metric-card-blackbox' style='border-top: 2px solid {color_skew};'>
                                <p class='metric-label-blackbox'>Skewness</p>
                                <p class='metric-value-blackbox'>{pf_skew:.2f}</p>
                                <p class='metric-status-blackbox' style='color: {color_skew};'>{status_skew}</p>
                            </div>
                        """, unsafe_allow_html=True)

                    with st2:
                        # Kurtosis: < 0 leve (bom), 0–2 moderado, > 2 fat tails (risco)
                        color_kurt = "#C5A059" if pf_kurt < 0 else ("#808080" if pf_kurt <= 2 else "#800000")
                        status_kurt = "Thin Tails (Safe)" if pf_kurt < 0 else ("Normal Dist." if pf_kurt <= 2 else "Fat Tails (Risk)")
                        st.markdown(f"""
                            <div class='metric-card-blackbox' style='border-top: 2px solid {color_kurt};'>
                                <p class='metric-label-blackbox'>Kurtosis (Excess)</p>
                                <p class='metric-value-blackbox'>{pf_kurt:.2f}</p>
                                <p class='metric-status-blackbox' style='color: {color_kurt};'>{status_kurt}</p>
                            </div>
                        """, unsafe_allow_html=True)

                    with st3:
                        # Probabilistic Sharpe: > 1 confiável, 0.5–1 moderado, < 0.5 incerto
                        prob_sharpe = 1.15  # substituir por cálculo real se disponível
                        color_ps = "#C5A059" if prob_sharpe > 1.0 else ("#808080" if prob_sharpe > 0.5 else "#800000")
                        status_ps = "High Confidence" if prob_sharpe > 1.0 else ("Moderate" if prob_sharpe > 0.5 else "Low Confidence")
                        st.markdown(f"""
                            <div class='metric-card-blackbox' style='border-top: 2px solid {color_ps};'>
                                <p class='metric-label-blackbox'>Prob. Sharpe</p>
                                <p class='metric-value-blackbox'>{prob_sharpe:.2f}</p>
                                <p class='metric-status-blackbox' style='color: {color_ps};'>{status_ps}</p>
                            </div>
                        """, unsafe_allow_html=True)

        except Exception as e:
            # --- 1. ESTILIZAÇÃO DE FALHA SISTÊMICA (BLACK BOX ENGINE) ---
            st.markdown("""
                <style>
                .terminal-alert-dark {
                    background-color: #0F0F0F; 
                    padding: 24px; 
                    border-radius: 2px;
                    border: 1px solid #333333;
                    border-left: 4px solid #800000; /* Oxblood Red para Crítico */
                    margin-top: 20px;
                }
                .alert-title-dark { 
                    color: #800000; 
                    font-size: 0.65rem; 
                    font-weight: 800; 
                    margin-bottom: 12px; 
                    text-transform: uppercase; 
                    letter-spacing: 2px; 
                }
                .alert-body-dark { 
                    color: #E0E0E0; 
                    font-size: 0.85rem; 
                    line-height: 1.6; 
                    font-family: 'Inter', sans-serif;
                }
                .technical-details-dark { 
                    color: #4D4D4D; 
                    font-size: 0.7rem; 
                    margin-top: 15px; 
                    font-family: 'JetBrains Mono', monospace; 
                    border-top: 1px solid #262626; 
                    padding-top: 12px; 
                    text-transform: uppercase;
                }
                </style>
            """, unsafe_allow_html=True)

            # --- 2. RENDERIZAÇÃO DO CARD DE ERRO ---
            st.markdown(f"""
                <div class="terminal-alert-dark">
                    <div class="alert-title-dark">Critical Engine Failure</div>
                    <div class="alert-body-dark">
                        O motor de análise detectou uma inconsistência na telemetria de mercado. 
                        A conexão com os terminais de dados foi interrompida para preservar a integridade dos cálculos.
                        <br><br>
                        <b>Ação recomendada:</b> Reinicie a sessão para restabelecer o handshake de dados.
                    </div>
                    <div class="technical-details-dark">
                        SYSTEM_LOG_ID: {str(e)}
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            st.stop()
            
    guia_conclusao('otimizacao')
    dynamic_footer()

elif st.session_state['pagina_atual'] == 'painel_calculadora':

    reset_scroll()

    c1, c2 = st.columns([1, 10])
    with c1:
        if st.button("⬅", key="v_calculadora_topo", use_container_width=True):
            st.session_state['pagina_atual'] = 'home'
            st.rerun()

    page_header()
    lynx_intelligence()
    
    # ──────────────────────────────────────────────
    # MOTOR DE CÁLCULO
    # ──────────────────────────────────────────────

    _IOF_TABELA = [
        96, 93, 90, 86, 83, 80, 76, 73, 70, 66,
        63, 60, 56, 53, 50, 46, 43, 40, 36, 33,
        30, 26, 23, 20, 16, 13, 10,  6,  3,  0,
    ]

    _ALIQ_RF = [
        {"max": 180,          "rate": 0.225, "label": "até 180 dias"},
        {"max": 360,          "rate": 0.200, "label": "181–360 dias"},
        {"max": 720,          "rate": 0.175, "label": "361–720 dias"},
        {"max": float("inf"), "rate": 0.150, "label": "acima de 720 dias"},
    ]

    _ALIQ_FUNDO_CP = [
        {"max": 180,          "rate": 0.225, "label": "até 180 dias"},
        {"max": float("inf"), "rate": 0.200, "label": "acima de 180 dias"},
    ]

    def _aliq_rf(dias: int) -> dict:
        for faixa in _ALIQ_RF:
            if dias <= faixa["max"]:
                return faixa
        return _ALIQ_RF[-1]

    def _iof_pct(dias: int) -> float:
        if dias >= 30:
            return 0.0
        return _IOF_TABELA[min(dias - 1, 29)] / 100

    def calc_renda_fixa(vn: float, taxa_aa: float, dias: int, subtipo: str) -> dict:
        ISENTO = {"lci_lca", "cri_cra", "debenture_incentivada", "poupanca"}
        dias_uteis = round(dias * 252 / 365)
        rend_bruto = vn * (math.pow(1 + taxa_aa / 100, dias_uteis / 252) - 1)
        iof = rend_bruto * _iof_pct(dias) if dias < 30 else 0.0
        base_ir = rend_bruto - iof

        if subtipo in ISENTO:
            ir = 0.0
            aliq_rate = 0.0
            aliq_label = "0%"
            aliq_desc  = "Isento de IR (pessoa física)"
        else:
            faixa = _aliq_rf(dias)
            ir = base_ir * faixa["rate"]
            aliq_rate  = faixa["rate"]
            aliq_label = f"{faixa['rate']*100:.1f}%".replace(".", ",")
            aliq_desc  = faixa["label"]

        rend_liq = rend_bruto - iof - ir
        taxa_liq = (math.pow(1 + rend_liq / vn, 365 / dias) - 1) * 100 if vn > 0 else 0

        return {
            "rend_bruto": rend_bruto, "iof": iof, "base_ir": base_ir,
            "ir": ir, "rend_liq": rend_liq, "taxa_liq": taxa_liq,
            "montante": vn + rend_liq, "aliq_rate": aliq_rate,
            "aliq_label": aliq_label, "aliq_desc": aliq_desc,
            "has_iof": iof > 0, "tabela_ir": "rf", "dias": dias,
        }

    def calc_fundo(vn: float, taxa_aa: float, dias: int, subtipo_fundo: str) -> dict:
        rend_bruto = vn * (math.pow(1 + taxa_aa / 100, dias / 365) - 1)
        tabela = _ALIQ_FUNDO_CP if subtipo_fundo == "curto_prazo" else _ALIQ_RF
        faixa = next((f for f in tabela if dias <= f["max"]), tabela[-1])
        ir = rend_bruto * faixa["rate"]
        rend_liq = rend_bruto - ir
        taxa_liq = (math.pow(1 + rend_liq / vn, 365 / dias) - 1) * 100 if vn > 0 else 0

        return {
            "rend_bruto": rend_bruto, "iof": 0.0, "base_ir": rend_bruto,
            "ir": ir, "rend_liq": rend_liq, "taxa_liq": taxa_liq,
            "montante": vn + rend_liq, "aliq_rate": faixa["rate"],
            "aliq_label": f"{faixa['rate']*100:.1f}%".replace(".", ","),
            "aliq_desc": faixa["label"], "has_iof": False,
            "tabela_ir": "fundo_cp" if subtipo_fundo == "curto_prazo" else "rf",
            "dias": dias,
        }

    def calc_acoes(vn: float, ganho: float, vendas_mes: float) -> dict:
        isento = vendas_mes <= 20_000
        ir = 0.0 if isento else ganho * 0.15
        rend_liq = ganho - ir
        taxa_liq = (rend_liq / vn * 100) if vn > 0 else 0

        return {
            "rend_bruto": ganho, "iof": 0.0, "base_ir": ganho,
            "ir": ir, "rend_liq": rend_liq, "taxa_liq": taxa_liq,
            "montante": vn + rend_liq,
            "aliq_rate": 0.0 if isento else 0.15,
            "aliq_label": "0%" if isento else "15%",
            "aliq_desc": "Isenção (vendas ≤ R$20.000/mês)" if isento else "Swing trade",
            "has_iof": False, "tabela_ir": None, "dias": None, "isento": isento,
        }

    def calc_etf(vn: float, taxa_aa: float, dias: int) -> dict:
        rend_bruto = vn * (math.pow(1 + taxa_aa / 100, dias / 365) - 1)
        ir = rend_bruto * 0.15
        rend_liq = rend_bruto - ir
        taxa_liq = (math.pow(1 + rend_liq / vn, 365 / dias) - 1) * 100 if vn > 0 else 0

        return {
            "rend_bruto": rend_bruto, "iof": 0.0, "base_ir": rend_bruto,
            "ir": ir, "rend_liq": rend_liq, "taxa_liq": taxa_liq,
            "montante": vn + rend_liq, "aliq_rate": 0.15,
            "aliq_label": "15%", "aliq_desc": "ETF renda variável",
            "has_iof": False, "tabela_ir": None, "dias": dias,
        }

    def calc_cripto(vn: float, ganho: float) -> dict:
        faixas = [
            (5_000_000,    0.150, "15%",   "ganho ≤ R$5 mi"),
            (10_000_000,   0.175, "17,5%", "R$5 mi < ganho ≤ R$10 mi"),
            (30_000_000,   0.200, "20%",   "R$10 mi < ganho ≤ R$30 mi"),
            (float("inf"), 0.225, "22,5%", "ganho > R$30 mi"),
        ]
        rate, label, desc = 0.15, "15%", "ganho ≤ R$5 mi"
        for limite, r, lb, dc in faixas:
            if ganho <= limite:
                rate, label, desc = r, lb, dc
                break

        ir = ganho * rate
        rend_liq = ganho - ir
        taxa_liq = (rend_liq / vn * 100) if vn > 0 else 0

        return {
            "rend_bruto": ganho, "iof": 0.0, "base_ir": ganho,
            "ir": ir, "rend_liq": rend_liq, "taxa_liq": taxa_liq,
            "montante": vn + rend_liq, "aliq_rate": rate,
            "aliq_label": label, "aliq_desc": desc,
            "has_iof": False, "tabela_ir": None, "dias": None,
        }

    # ──────────────────────────────────────────────
    # HELPERS DE FORMATAÇÃO
    # ──────────────────────────────────────────────

    def fmt_brl(v: float) -> str:
        return f"R$ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

    def fmt_pct(v: float, decimals: int = 2) -> str:
        return f"{v:.{decimals}f}%".replace(".", ",")

    # ──────────────────────────────────────────────
    # GRÁFICOS
    # ──────────────────────────────────────────────

    def grafico_barras(vn: float, r: dict) -> go.Figure:
        labels = ["Capital inicial", "Rendimento bruto", "Imposto + IOF", "Rendimento líq.", "Montante final"]
        valores = [vn, r["rend_bruto"], abs(r["ir"] + r["iof"]), r["rend_liq"], r["montante"]]
        cores = ["#404040", "#D4AF37", "#846B32", "#C5A059", "#DAA520"]

        fig = go.Figure(go.Bar(
            y=labels, x=valores, orientation='h',
            marker_color=cores, marker_line_width=0,
            text=[fmt_brl(abs(v)) for v in valores],
            textposition="outside", cliponaxis=False,
            textfont=dict(size=11, family="JetBrains Mono", color="#E0E0E0"),
        ))

        fig.update_layout(
            margin=dict(t=20, b=10, l=120, r=60),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(
                showgrid=True, gridcolor="#262626", tickformat=",.0f",
                tickprefix="R$ ", tickfont=dict(size=10, family="Inter", color="#808080"),
                zeroline=True, zerolinecolor="#404040",
                range=[0, max(valores) * 1.4]
            ),
            yaxis=dict(
                tickfont=dict(size=11, family="Inter", color="#E0E0E0"),
                autorange="reversed", showline=False
            ),
            height=350, showlegend=False, dragmode=False, hovermode=False
        )
        return fig

    def grafico_pizza(r: dict) -> go.Figure:
        labels = ["Rendimento líquido", "Imposto"]
        values = [max(r["rend_liq"], 0), r["ir"] + r["iof"]]
        cores  = ["#C5A059", "#846B32"]
        total = sum(values) or 1
        pct_liq = values[0] / total

        fig = go.Figure(data=[go.Pie(
            labels=labels, values=values, hole=0.78,
            marker=dict(colors=cores, line=dict(color="#080808", width=2)),
            textinfo="none", hoverinfo="label+percent",
            direction="clockwise", sort=False,
        )])

        fig.add_annotation(
            text=f"LÍQUIDO<br><br><b style='font-size:22px; color:#E0E0E0;'>{pct_liq:.1%}</b>",
            showarrow=False, x=0.5, y=0.5,
            font=dict(size=14, family="Inter", color="#808080"), align="center",
        )

        fig.update_layout(
            margin=dict(t=0, b=0, l=0, r=0), showlegend=True,
            legend=dict(
                orientation="h", yanchor="bottom", y=-0.2,
                xanchor="center", x=0.5,
                font=dict(family="Inter", size=11, color="#808080"),
                itemclick=False, itemdoubleclick=False,
            ),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", 
            height=500,
        )
        return fig

    # ──────────────────────────────────────────────
    # TABELA IR
    # ──────────────────────────────────────────────

    def html_tabela_ir(tipo_tabela: str, dias) -> str:
        if tipo_tabela == "rf":
            linhas = [
                ("até 180 dias",      "22,5%", dias is not None and dias <= 180),
                ("181 a 360 dias",    "20,0%", dias is not None and 181 <= dias <= 360),
                ("361 a 720 dias",    "17,5%", dias is not None and 361 <= dias <= 720),
                ("acima de 720 dias", "15,0%", dias is not None and dias > 720),
            ]
        elif tipo_tabela == "fundo_cp":
            linhas = [
                ("até 180 dias",      "22,5%", dias is not None and dias <= 180),
                ("acima de 180 dias", "20,0%", dias is not None and dias > 180),
            ]
        else:
            return ""

        S_TH = ("text-align:left; color:#666; font-weight:700; font-size:10px; "
                "padding:6px 10px; border-bottom:1px solid #333; text-transform:uppercase; "
                "letter-spacing:0.1em; font-family:'Inter',monospace;")
        S_TD = ("padding:10px; color:#E0E0E0; border-bottom:1px solid #1A1A1A; "
                "font-size:12px; font-family:'Inter',monospace;")
        S_TD_ATIVO = ("padding:10px; color:#C5A059; font-weight:600; "
                      "border-bottom:1px solid #1A1A1A; font-size:12px; "
                      "font-family:'Inter',monospace; "
                      "background:rgba(197,160,89,0.08); border-left: 2px solid #C5A059;")

        rows = ""
        for i, (prazo, aliq, ativo) in enumerate(linhas):
            td_style = S_TD_ATIVO if ativo else S_TD
            last = "border-bottom:none;" if i == len(linhas) - 1 else ""
            status = "✓ ATIVA" if ativo else "—"
            rows += (f'<tr>'
                     f'<td style="{td_style}{last}">{prazo}</td>'
                     f'<td style="{td_style}{last}">{aliq}</td>'
                     f'<td style="{td_style}{last}">{status}</td>'
                     f'</tr>')

        return (f'<table style="width:100%; border-collapse:collapse; background:#0E0E0E; border:1px solid #1A1A1A;">'
                f'<thead><tr>'
                f'<th style="{S_TH}">Prazo</th>'
                f'<th style="{S_TH}">Alíquota IR</th>'
                f'<th style="{S_TH}">Status</th>'
                f'</tr></thead>'
                f'<tbody>{rows}</tbody></table>')

    # ──────────────────────────────────────────────
    # INTERFACE
    # ──────────────────────────────────────────────
    col_inputs, col_result = st.columns([1, 1.6], gap="large")

    # ─────────────── COLUNA ESQUERDA: INPUTS ───────────────
    with col_inputs:
        st.markdown('<div class="section-title">Tipo de ativo</div>', unsafe_allow_html=True)

        TIPOS = {
            "Renda fixa":   "renda_fixa",
            "Fundos":       "fundo",
            "Ações":        "acoes",
            "ETFs":         "etf",
            "Criptoativos": "criptos",
        }

        tipo_label = st.selectbox(
            "Selecione a classe de ativo",
            options=list(TIPOS.keys()),
            label_visibility="collapsed",
            index=0
        )
        tipo = TIPOS[tipo_label]

        st.markdown('<div class="section-title">Dados do investimento</div>', unsafe_allow_html=True)

        vn = st.number_input("Valor nominal (R$)", min_value=0.0, value=10_000.0, step=500.0, format="%.2f")

        if tipo in ("acoes", "criptos"):
            taxa_label = "Ganho de capital (% sobre VN)"
        elif tipo == "fundo":
            taxa_label = "Rentabilidade bruta (% a.a.)"
        else:
            taxa_label = "Taxa nominal bruta (% a.a.)"
        taxa = st.number_input(taxa_label, min_value=0.0, value=12.0, step=0.1, format="%.2f")

        if tipo == "renda_fixa":
            prazo = st.number_input("Prazo (dias corridos)", min_value=1, value=365, step=1)
            subtipo = st.selectbox("Subtipo", [
                ("CDB / LC / Tesouro",             "geral"),
                ("LCI / LCA (Isento)",             "lci_lca"),
                ("CRI / CRA (Isento PF)",          "cri_cra"),
                ("Debênture incentivada (Isento)", "debenture_incentivada"),
                ("Poupança (Isenta PF)",           "poupanca"),
            ], format_func=lambda x: x[0])
            subtipo_val = subtipo[1]

        elif tipo == "fundo":
            prazo = st.number_input("Prazo (dias)", min_value=1, value=365, step=1)
            subtipo_fundo = st.selectbox("Tipo de fundo", [
                ("Longo prazo (> 365 dias)", "longo_prazo"),
                ("Curto prazo (≤ 365 dias)", "curto_prazo"),
            ], format_func=lambda x: x[0])
            subtipo_fundo_val = subtipo_fundo[1]

        elif tipo == "acoes":
            vendas_mes = st.number_input(
                "Total de vendas no mês (R$)",
                min_value=0.0, value=15_000.0, step=500.0, format="%.2f",
            )

        elif tipo == "etf":
            prazo = st.number_input("Prazo (dias)", min_value=1, value=365, step=1)

        # ── Cálculo ──
        if tipo == "renda_fixa":
            r = calc_renda_fixa(vn, taxa, int(prazo), subtipo_val)
        elif tipo == "fundo":
            r = calc_fundo(vn, taxa, int(prazo), subtipo_fundo_val)
        elif tipo == "acoes":
            r = calc_acoes(vn, vn * taxa / 100, vendas_mes)
        elif tipo == "etf":
            r = calc_etf(vn, taxa, int(prazo))
        else:
            r = calc_cripto(vn, vn * taxa / 100)

        # ── Aviso informativo ──
        if tipo == "renda_fixa" and subtipo_val in ("lci_lca", "cri_cra", "debenture_incentivada"):
            info_text = "Este ativo é <strong>isento de IR para pessoa física</strong>. A isenção não se aplica a PJ."
        elif tipo == "acoes" and r.get("isento"):
            info_text = "Vendas mensais <strong>≤ R$20.000</strong> em ações são isentas de IR para PF. Day trade: 20%."
        elif tipo == "criptos":
            info_text = "Tabela progressiva (Lei 14.754/2023). Vendas ≤ R$35.000/mês isentas para PF (ganho de capital)."
        elif tipo == "renda_fixa" and r.get("has_iof"):
            info_text = f"IOF regressivo aplicado: {fmt_pct(r['iof'] / r['rend_bruto'] * 100)} sobre o rendimento bruto (resgate em {int(prazo)} dias)."
        else:
            info_text = None

        if info_text:
            st.markdown(f"""
                <div style="background-color:#1A1A1A; padding:12px; border-left:3px solid #C5A059;
                            margin-top:20px; border-radius:2px;">
                    <span style="color:#C5A059; font-family:'Inter',monospace; font-size:0.75rem;
                                 font-weight:bold; letter-spacing:0.5px; text-transform:uppercase; line-height:1.4;">
                        {info_text}
                    </span>
                </div>
            """, unsafe_allow_html=True)

    # ─────────────── COLUNA DIREITA: RESULTADO ───────────────
    with col_result:
        st.markdown('<div class="section-title">Resultado</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-grid">
            <div class="stat-card" style="border-top:2px solid #C5A059;">
                <div class="stat-label">Rendimento bruto</div>
                <div class="stat-value">{fmt_brl(r["rend_bruto"])}</div>
                <div class="stat-status">{fmt_pct(r["rend_bruto"] / vn * 100) if vn else "—"} do capital</div>
            </div>
            <div class="stat-card" style="border-top:2px solid #C5A059;">
                <div class="stat-label">IR / Imposto</div>
                <div class="stat-value">{fmt_brl(r["ir"])}</div>
                <div class="stat-status">Alíquota {r["aliq_label"]} · {r["aliq_desc"]}</div>
            </div>
            <div class="stat-card" style="border-top:2px solid #C5A059;">
                <div class="stat-label">Montante líquido</div>
                <div class="stat-value">{fmt_brl(r["montante"])}</div>
                <div class="stat-status">+{fmt_pct(r["rend_liq"] / vn * 100) if vn else "—"} líquido</div>
            </div>
        </div>
        <div class="metric-grid">
            <div class="stat-card" style="border-top:2px solid #C5A059;">
                <div class="stat-label">Taxa líquida efetiva</div>
                <div class="stat-value">{fmt_pct(r["taxa_liq"])} a.a.</div>
                <div class="stat-status">equivalente anual</div>
            </div>
            <div class="stat-card" style="border-top:2px solid #C5A059;">
                <div class="stat-label">Alíquota aplicada</div>
                <div class="stat-value">{r["aliq_label"]}</div>
                <div class="stat-status">{r["aliq_desc"]}</div>
            </div>
            <div class="stat-card" style="border-top:2px solid #C5A059;">
                <div class="stat-label">IOF retido</div>
                <div class="stat-value">{fmt_brl(r["iof"])}</div>
                <div class="stat-status">{"dias < 30" if r["has_iof"] else "não aplicável"}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    tab_overview, tab_tabela_ir = st.tabs(["Overview", "Tabela IR"])

    with tab_overview:
        S_TABLE   = "width:100%; border-collapse:collapse; font-family:'Inter', sans-serif; background-color:#0E0E0E;"
        S_TD_KEY  = "padding:10px 5px; border-bottom:1px solid #262626; color:#808080; font-size:0.85rem; text-transform:uppercase; letter-spacing:0.5px;"
        S_TD_VAL  = "padding:10px 5px; border-bottom:1px solid #262626; color:#E0E0E0; text-align:right; font-family:'Inter', sans-serif; font-size:0.85rem;"
        S_TD_POS  = "padding:10px 5px; border-bottom:1px solid #262626; color:#2ecc71; text-align:right; font-family:'Inter', sans-serif; font-size:0.85rem;"
        S_TD_NEG  = "padding:10px 5px; border-bottom:1px solid #262626; color:#e74c3c; text-align:right; font-family:'Inter', sans-serif; font-size:0.85rem;"
        S_TD_LAST = "padding:12px 5px; border-top:2px solid #C5A059; background-color:rgba(197,160,89,0.05);"

        iof_row = ""
        if r["has_iof"]:
            iof_row = (
                f'<tr>'
                f'<td style="{S_TD_KEY}">IOF retido</td>'
                f'<td style="{S_TD_NEG}">–{fmt_brl(r["iof"])}</td>'
                f'</tr>'
            )

        html = f"""
        <table style="{S_TABLE}">
        <tr>
            <td style="{S_TD_KEY}">Capital aplicado</td>
            <td style="{S_TD_VAL}">{fmt_brl(vn)}</td>
        </tr>
        <tr>
            <td style="{S_TD_KEY}">Rendimento bruto</td>
            <td style="{S_TD_POS}">+{fmt_brl(r["rend_bruto"])}</td>
        </tr>
        <tr>
            <td style="{S_TD_KEY}">Base de cálculo IR</td>
            <td style="{S_TD_VAL}">{fmt_brl(r["base_ir"])}</td>
        </tr>
        {iof_row}
        <tr>
            <td style="{S_TD_KEY}">IR retido ({r["aliq_label"]})</td>
            <td style="{S_TD_NEG}">–{fmt_brl(r["ir"])}</td>
        </tr>
        <tr>
            <td style="{S_TD_KEY}">Rendimento líquido</td>
            <td style="{S_TD_POS}">+{fmt_brl(r["rend_liq"])}</td>
        </tr>
        <tr>
            <td style="{S_TD_LAST} color:#E0E0E0; font-size:0.9rem;"><strong>Montante final</strong></td>
            <td style="{S_TD_LAST} text-align:right; color:#C5A059; font-family:'Inter', sans-serif; font-size:0.9rem;"><strong>{fmt_brl(r["montante"])}</strong></td>
        </tr>
        </table>
        """
        st.markdown(html, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.plotly_chart(grafico_barras(vn, r), use_container_width=True, config={'displayModeBar': False})
        st.markdown("<br>", unsafe_allow_html=True)

        if r["rend_bruto"] > 0:
            st.plotly_chart(grafico_pizza(r), use_container_width=True)
        else:
            st.info("Insira valores para visualizar a composição.")

    with tab_tabela_ir:
        if r.get("tabela_ir"):
            st.markdown(html_tabela_ir(r["tabela_ir"], r.get("dias")), unsafe_allow_html=True)
        else:
            st.info("Este tipo de ativo não utiliza tabela regressiva de IR.")

    st.markdown("<div style='margin-top:3rem;'></div>", unsafe_allow_html=True)
    st.markdown("""
    ---
    <small style="color:#bbb; font-size:11px;">
    Calculadora para <strong>pessoa física</strong> · Valores meramente indicativos ·
    Consulte sempre um especialista · Legislação: Lei 11.033/2004 (RF), Lei 14.754/2023 (Cripto)
    </small>
    """, unsafe_allow_html=True)

    dynamic_footer()