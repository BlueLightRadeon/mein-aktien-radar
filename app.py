import os
import sys
import time
import importlib.util
from datetime import datetime, timezone, timedelta
import pandas as pd
import streamlit as st

# Pfade absichern
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

TABS_DIR = os.path.join(ROOT_DIR, "tabs")

# 1. AUTONOME SELBSTREPARATUR: Unsichtbare Steuerzeichen (\u2060) automatisch entfernen
if os.path.exists(TABS_DIR):
    for fname in os.listdir(TABS_DIR):
        clean_fname = fname.replace("\u2060", "").strip()
        if clean_fname != fname:
            old_p = os.path.join(TABS_DIR, fname)
            new_p = os.path.join(TABS_DIR, clean_fname)
            try:
                os.rename(old_p, new_p)
            except Exception:
                pass

# Backend-Services importieren
from storage_service import load_saved_portfolio, save_saved_portfolio, delete_saved_portfolio
from data_service import (
    get_stock_data, search_ticker_candidates, clean_ticker, 
    parse_trade_republic_pdf, get_display_name
)
from ai_service import get_account_models

# 2. ISOLIERTER TAB-LOADER: Verhindert, dass ein einzelner Tab die ganze App lahmlegt
def safe_load_tab(tab_idx, tab_name):
    # Versuch A: Regulärer Import
    try:
        mod = importlib.import_module(f"tabs.{tab_name}")
        return mod, None
    except Exception as e:
        err_a = str(e)

    # Versuch B: Direkter Dateipfad-Import (falls Dateiname minimal abweicht)
    if os.path.exists(TABS_DIR):
        for fname in os.listdir(TABS_DIR):
            c_name = fname.replace("\u2060", "").lower().strip()
            if c_name.startswith(f"tab{tab_idx}") and c_name.endswith(".py"):
                try:
                    fpath = os.path.join(TABS_DIR, fname)
                    spec = importlib.util.spec_from_file_location(f"tab_{tab_idx}", fpath)
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    return mod, None
                except Exception as e:
                    return None, f"Fehler in {fname}: {e}"

    return None, f"Modul tabs.{tab_name} konnte nicht geladen werden ({err_a})"

tab_names = [
    (0, "tab0_konto"),
    (1, "tab1_empfehlungen"),
    (2, "tab2_nachrichten"),
    (3, "tab3_cashflow"),
    (4, "tab4_kaufideen"),
    (5, "tab5_charts"),
    (6, "tab6_streuung"),
    (7, "tab7_duell"),
    (8, "tab8_prognose")
]

tabs_dict = {}
for idx, name in tab_names:
    mod, err = safe_load_tab(idx, name)
    tabs_dict[idx] = (mod, err)

# Initialisierung
if "v_portfolio" not in st.session_state:
    saved_items, saved_cash = load_saved_portfolio()
    st.session_state["v_portfolio"] = saved_items
    st.session_state["v_cash"] = saved_cash

if "v_cash" not in st.session_state:
    st.session_state["v_cash"] = 0.0

if "last_auto_run_ts" not in st.session_state:
    st.session_state["last_auto_run_ts"] = 0.0

st.set_page_config(page_title="KI Markt- & Depot-Radar", page_icon="📈", layout="wide")
st.title("📈 KI Markt- & Depot-Radar")

def get_groq_key():
    for k in ["GROQ_API_KEY", "groq_api_key", "GROQ_KEY", "groq_key"]:
        if k in st.secrets:
            v = str(st.secrets[k]).strip()
            if v:
                return v
    return ""

GROQ_KEY = get_groq_key()

def get_berlin_time_str():
    tz_de = timezone(timedelta(hours=2))
    return datetime.now(tz_de).strftime("%H:%M:%S Uhr")

def fmt_eur(val):
    try:
        val_float = float(val)
        return f"{val_float:,.2f} €".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return "0,00 €"

# Fragment für ruckelfreien Live-Timer
def get_fragment_decorator(interval_sec=1):
    if hasattr(st, "fragment"):
        return st.fragment(run_every=interval_sec)
    elif hasattr(st, "experimental_fragment"):
        return st.experimental_fragment(run_every=interval_sec)
    return lambda f: f

@get_fragment_decorator(interval_sec=1)
def render_live_timer_panel(ten_mins, auto_refresh_active, has_portfolio):
    now_ts = time.time()
    last_ts = st.session_state.get("last_auto_run_ts", 0.0)
    last_time_str = st.session_state.get("last_news_time", st.session_state.get("last_depot_time", ""))

    if last_time_str:
        st.write(f"🕒 Stand: **{last_time_str}**")
        if auto_refresh_active:
            elapsed = now_ts - last_ts
            remaining = max(0, int(ten_mins - elapsed))
            st.caption(f"⏳ Nächstes Auto-Update in ca. **{remaining // 60}m {remaining % 60:02d}s**")
            if remaining <= 0 and has_portfolio:
                st.session_state["last_auto_run_ts"] = time.time()
                st.rerun()
        else:
            st.caption("⏸️ Auto-Update pausiert")
    else:
        st.caption("Nutze die KI-Buttons in den Tabs, um Analysen zu starten.")

# --- SEITENLEISTE ---
with st.sidebar:
    st.header("💼 Trade Republic Depot")
    with st.expander("📥 TR-Kontoauszug (PDF) einlesen", expanded=True):
        st.caption("Wähle deine PDF aus und klicke auf '📄 Einlesen'.")
        tr_pdf = st.file_uploader("PDF auswählen", type=["pdf"], key="tr_pdf_file_input")
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            if st.button("📄 Einlesen", width="stretch", type="primary", disabled=(tr_pdf is None)):
                imported_items, imported_cash = parse_trade_republic_pdf(tr_pdf)
                st.session_state["v_portfolio"] = imported_items
                st.session_state["v_cash"] = float(imported_cash)
                save_saved_portfolio(imported_items, float(imported_cash), sync_github=True)
                st.session_state["last_auto_run_ts"] = 0.0
                st.success(f"✅ {len(imported_items)} Positionen gesichert!")
                st.rerun()
        with col_p2:
            if st.button("🗑️ Löschen", width="stretch"):
                delete_saved_portfolio()
                st.session_state["v_portfolio"] = []
                st.session_state["v_cash"] = 0.0
                st.session_state.pop("ai_signals", None)
                st.session_state.pop("ai_market", None)
                st.session_state.pop("ai_depot", None)
                st.session_state.pop("ai_cluster", None)
                st.session_state["last_auto_run_ts"] = 0.0
                st.success("✅ Auszug gelöscht!")
                st.rerun()

    display_cash = float(st.session_state.get("v_cash", 0.0))
    st.info(f"💶 **Cash (aus Auszug):** `{fmt_eur(display_cash)}`")

    if len(st.session_state.get("v_portfolio", [])) > 0:
        if str(st.secrets.get("GITHUB_TOKEN", "")).strip() and str(st.secrets.get("GITHUB_REPO", "")).strip():
            st.caption("☁️ **Status:** Depot mit GitHub synchronisiert.")
        else:
            st.caption("💾 **Status:** Depot lokal gesichert.")

    st.divider()
    st.subheader("🔍 Aktie manuell hinzufügen:")
    search_query = st.text_input("Name oder Symbol:", placeholder="z. B. Apple, Tesla...")
    if search_query:
        results = search_ticker_candidates(search_query)
        if results:
            selected_cand = st.selectbox("Treffer:", results, key="side_search_select")
            in_money = st.number_input("Aktueller Wert (€):", min_value=1.0, value=50.0, step=5.0)
            if st.button("➕ Hinzufügen", width="stretch"):
                sym = clean_ticker(selected_cand)
                disp_name = get_display_name(sym)
                st.session_state["v_portfolio"].append({"ticker": sym, "name": disp_name, "shares": 1.0, "buy_price": float(in_money)})
                save_saved_portfolio(st.session_state["v_portfolio"], st.session_state["v_cash"], sync_github=True)
                st.session_state["last_auto_run_ts"] = 0.0
                st.success(f"✅ {disp_name} hinzugefügt!")
                st.rerun()

    st.divider()
    st.subheader("📋 Positionen & Werte:")
    portfolio_list = st.session_state.get("v_portfolio", [])
    if portfolio_list:
        for idx, item in enumerate(list(portfolio_list)):
            disp_name = get_display_name(item.get("ticker", ""), item.get("name"))
            col_pos_a, col_pos_b = st.columns([3, 1])
            with col_pos_a:
                st.write(f"• **{disp_name}**")
                current_val = float(item.get("buy_price", 0.0))
                new_val = st.number_input(f"Wert (€):", min_value=0.0, value=current_val, step=5.0, key=f"input_pos_{idx}_{item.get('ticker','')}", label_visibility="collapsed")
                if new_val != current_val:
                    st.session_state["v_portfolio"][idx]["buy_price"] = float(new_val)
                    save_saved_portfolio(st.session_state["v_portfolio"], st.session_state["v_cash"], sync_github=True)
                    st.rerun()
            with col_pos_b:
                if st.button("❌", key=f"del_item_{idx}_{item.get('ticker','')}"):
                    st.session_state["v_portfolio"].pop(idx)
                    save_saved_portfolio(st.session_state["v_portfolio"], st.session_state["v_cash"], sync_github=True)
                    st.rerun()
    else:
        st.info("Noch kein Auszug geladen (0 Positionen).")

    st.divider()
    st.header("🤖 KI-Einstellungen")
    auto_refresh_active = st.toggle("🔄 Auto-Update alle 10 Min.", value=True)
    available_models = get_account_models(GROQ_KEY)
    saved_model = st.session_state.get("selected_groq_model", available_models[0])
    model_idx = available_models.index(saved_model) if saved_model in available_models else 0
    selected_model = st.selectbox("KI-Modell:", available_models, index=model_idx)
    st.session_state["selected_groq_model"] = selected_model

# --- BERECHNUNGEN & METRIKEN ---
stock_df, ticker_news, resolved_tickers = get_stock_data(portfolio_list)

if not stock_df.empty and "_raw_val" in stock_df.columns:
    stock_val = stock_df["_raw_val"].sum()
    stock_pnl = stock_df["_raw_pnl"].sum()
else:
    stock_val = sum([float(x.get("buy_price", 0.0)) for x in portfolio_list])
    stock_pnl = 0.0

total_tr_account = stock_val + display_cash
stock_pnl_pct = (stock_pnl / (stock_val - stock_pnl) * 100.0) if (stock_val - stock_pnl) > 0 else 0.0

c_m1, c_m2, c_m3 = st.columns(3)
with c_m1:
    st.metric("TR Gesamtkonto", fmt_eur(total_tr_account), help="Gesamtwert deines Kontos: Brokerage + Cash")
with c_m2:
    st.metric("Aktueller Wert der Aktien", fmt_eur(stock_val), help="Gesamtwert aller deiner gehaltenen Wertpapiere")
with c_m3:
    st.metric("Gewinn / Verlust (Tag)", fmt_eur(stock_pnl), delta=f"{stock_pnl_pct:+.2f}%")

col_head_info, col_timer_info = st.columns([3, 2])
with col_head_info:
    st.caption("💡 Jeder Tab besitzt einen eigenen KI-Button für gezielte und vollständige Auswertungen.")
with col_timer_info:
    ten_mins = 600.0
    render_live_timer_panel(ten_mins, auto_refresh_active, bool(portfolio_list))

# --- TABS RENDERN (Mit Einzelfall-Absicherung) ---
tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "🏦 TR-Konto",
    "💼 Stimmung & Empfehlungen",
    "🌍 Nachrichten",
    "📅 Termine & Cashflow",
    "🎯 Top 5 Kaufempfehlungen",
    "📊 Charts",
    "🥧 Streuung",
    "⚔️ Duell",
    "🧠 KI-Lernlabor & Prognose"
])

def render_or_error(tab_idx, render_func):
    mod, err = tabs_dict.get(tab_idx, (None, "Nicht gefunden"))
    if mod and hasattr(mod, "render"):
        render_func(mod)
    else:
        st.error(f"⚠️ Tab {tab_idx} konnte nicht geladen werden:")
        st.code(err or "Unbekannter Fehler")

with tab0:
    render_or_error(0, lambda m: m.render(stock_df, display_cash, stock_val, fmt_eur))

with tab1:
    render_or_error(1, lambda m: m.render(stock_df, selected_model, GROQ_KEY, get_berlin_time_str))

with tab2:
    render_or_error(2, lambda m: m.render(selected_model, GROQ_KEY, get_berlin_time_str))

with tab3:
    render_or_error(3, lambda m: m.render(stock_df, stock_val))

with tab4:
    render_or_error(4, lambda m: m.render(portfolio_list, selected_model, GROQ_KEY, get_berlin_time_str))

with tab5:
    render_or_error(5, lambda m: m.render(portfolio_list))

with tab6:
    render_or_error(6, lambda m: m.render(stock_df, stock_val, fmt_eur, selected_model, GROQ_KEY, get_berlin_time_str))

with tab7:
    render_or_error(7, lambda m: m.render(stock_df, selected_model, GROQ_KEY))

with tab8:
    render_or_error(8, lambda m: m.render(portfolio_list, selected_model, GROQ_KEY))
