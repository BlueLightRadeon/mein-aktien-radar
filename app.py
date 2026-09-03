import os
import importlib
import time
from datetime import datetime, timezone, timedelta
import streamlit as st
from groq import Groq

# Backend-Services importieren
try:
    from storage_service import load_saved_portfolio, save_saved_portfolio, delete_saved_portfolio
    from data_service import (
        get_stock_data, fetch_all_headlines, search_ticker_candidates, 
        clean_ticker, parse_trade_republic_pdf, get_display_name
    )
    from ai_service import get_account_models, run_analysis
except Exception as e:
    st.error(f"❌ Fehler in den Service-Dateien (Hauptordner): {e}")
    st.stop()

# Fehlertoleranter Tab-Loader (findet tab2 auch bei Groß-/Kleinschreibung oder kleinen Abweichungen)
def load_tab_module(tab_idx, default_name):
    tabs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tabs")
    if os.path.exists(tabs_path):
        # 1. Exakter Treffer
        if f"{default_name}.py" in os.listdir(tabs_path):
            return importlib.import_module(f"tabs.{default_name}")
        
        # 2. Tolerante Suche nach Präfix (tab0, Tab0, tab_0 etc.)
        for fname in os.listdir(tabs_path):
            lower = fname.lower().strip()
            if fname.endswith(".py") and (
                lower.startswith(f"tab{tab_idx}") or 
                lower.startswith(f"tab_{tab_idx}") or 
                lower.startswith(f"{tab_idx}_") or 
                lower == f"{tab_idx}.py"
            ):
                return importlib.import_module(f"tabs.{fname[:-3]}")
                
    return importlib.import_module(f"tabs.{default_name}")

expected_tabs = [
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

tab_modules = {}
import_errors = []

for idx, def_name in expected_tabs:
    try:
        tab_modules[idx] = load_tab_module(idx, def_name)
    except Exception as e:
        import_errors.append(f"Tab {idx} ({def_name}): {e}")

if import_errors:
    st.error("❌ Folgende Tabs konnten im Ordner 'tabs/' nicht geladen werden:")
    for err in import_errors:
        st.write(f"• **{err}**")
    tabs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tabs")
    if os.path.exists(tabs_dir):
        st.info(f"📂 Tatsächlich gefundene Dateien in `tabs/`: `{os.listdir(tabs_dir)}`")
    else:
        st.warning("⚠️ Der Ordner 'tabs/' existiert nicht auf dem Server.")
    st.stop()

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

def trigger_ai_run(portfolio_items, current_stock_df, model_to_use):
    if not GROQ_KEY or not portfolio_items:
        return False
    try:
        client = Groq(api_key=GROQ_KEY)
        news_data = fetch_all_headlines()
        news_text = "\n".join(news_data) if news_data else "Aktuell keine Sondermeldungen."

        if not current_stock_df.empty and "Unternehmen" in current_stock_df.columns:
            summary_cols = [c for c in ["Unternehmen", "Aktuelle KI-Empfehlung", "Aktuelle Begründung", "Empfehlung (vor 3 Monaten)", "Begründung (vor 3 Monaten)", "Börsenkurs", "Aktueller Wert (TR)", "KGV (P/E)", "Fair Value", "Analysten-Kursziel"] if c in current_stock_df.columns]
            metrics_summary = current_stock_df[summary_cols].to_string(index=False)
            cluster_cols = [c for c in ["Unternehmen", "Sektor", "Land", "Rolle", "Aktueller Wert (TR)"] if c in current_stock_df.columns]
            cluster_context = current_stock_df[cluster_cols].to_string(index=False)
        else:
            metrics_summary = "Keine Einzelwerte hinterlegt."
            cluster_context = "Keine Sektoraufteilung vorhanden."

        out_m, out_d, out_s, out_c = run_analysis(client, model_to_use, news_text, metrics_summary, "", cluster_context)
        st.session_state["ai_market"] = out_m
        st.session_state["ai_depot"] = out_d
        st.session_state["ai_signals"] = out_s
        st.session_state["ai_cluster"] = out_c
        st.session_state["last_analysis_time"] = get_berlin_time_str()
        st.session_state["last_auto_run_ts"] = time.time()
        return True
    except Exception as e:
        st.error(f"⚠️ Groq API Fehler: {str(e)}")
        return False

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
    last_time_str = st.session_state.get("last_analysis_time", "")

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
        st.caption("Lade ein Depot hoch, um die Analyse zu starten.")

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
                st.success(f"✅ {len(imported_items)} Positionen dauerhaft gesichert!")
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

now_ts = time.time()
last_ts = st.session_state.get("last_auto_run_ts", 0.0)
ten_mins = 600.0

if portfolio_list and GROQ_KEY and auto_refresh_active:
    if (now_ts - last_ts) >= ten_mins:
        with st.spinner("🔄 KI-Radar analysiert Marktlage & Portfolio..."):
            trigger_ai_run(portfolio_list, stock_df, selected_model)
            now_ts = time.time()
            last_ts = st.session_state.get("last_auto_run_ts", now_ts)

col_btn, col_info = st.columns([3, 2])
with col_btn:
    if st.button("🚀 Jetzt sofort manuell aktualisieren", width="stretch", type="primary"):
        with st.spinner("Analysiere Markt & Portfolio mit Groq KI..."):
            if trigger_ai_run(portfolio_list, stock_df, selected_model):
                st.success("✅ Auswertung erfolgreich aktualisiert!")

with col_info:
    render_live_timer_panel(ten_mins, auto_refresh_active, bool(portfolio_list))

# --- TABS ---
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

with tab0:
    tab_modules[0].render(stock_df, display_cash, stock_val, fmt_eur)

with tab1:
    tab_modules[1].render(stock_df)

with tab2:
    tab_modules[2].render(st.session_state.get("last_analysis_time", ""))

with tab3:
    tab_modules[3].render(stock_df, stock_val)

with tab4:
    tab_modules[4].render(st.session_state.get("last_analysis_time", ""))

with tab5:
    tab_modules[5].render(portfolio_list)

with tab6:
    tab_modules[6].render(stock_df, stock_val, fmt_eur)

with tab7:
    tab_modules[7].render(stock_df, selected_model, GROQ_KEY)

with tab8:
    tab_modules[8].render(portfolio_list, selected_model, GROQ_KEY)
