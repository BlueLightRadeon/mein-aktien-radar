import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from groq import Groq
from datetime import datetime, timezone, timedelta
import time

if "v_portfolio" not in st.session_state:
    st.session_state["v_portfolio"] = []

if "v_cash" not in st.session_state:
    st.session_state["v_cash"] = 0.0

if "last_auto_run_ts" not in st.session_state:
    st.session_state["last_auto_run_ts"] = 0.0

try:
    from data_service import (
        get_stock_data, get_individual_series_dict, fetch_all_headlines, 
        search_ticker_candidates, clean_ticker, parse_trade_republic_pdf, 
        get_display_name
    )
    from ai_service import get_account_models, run_analysis, run_duel_analysis
except Exception as e:
    st.error(f"Import-Fehler: {e}")
    st.stop()

st.set_page_config(
    page_title="KI Markt- & Depot-Radar",
    page_icon="📈",
    layout="wide"
)

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
            summary_cols = [c for c in ["Unternehmen", "Handelsempfehlung", "Börsenkurs", "RSI (14D)", "KGV (P/E)", "Fair Value", "Analysten-Kursziel", "Dividendenrendite"] if c in current_stock_df.columns]
            metrics_summary = current_stock_df[summary_cols].to_string(index=False)
            cluster_cols = [c for c in ["Unternehmen", "Sektor", "Land", "Rolle", "Aktueller Wert (TR)"] if c in current_stock_df.columns]
            cluster_context = current_stock_df[cluster_cols].to_string(index=False)
        else:
            metrics_summary = "Keine Einzelwerte hinterlegt."
            cluster_context = "Keine Sektoraufteilung vorhanden."

        out_m, out_d, out_s, out_c = run_analysis(
            client, model_to_use, news_text, metrics_summary, "", cluster_context
        )
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

# --- SEITENLEISTE ---
with st.sidebar:
    st.header("💼 Trade Republic Depot")
    
    with st.expander("📥 TR-Kontoauszug (PDF) einlesen", expanded=True):
        st.caption("Wähle deine PDF aus und klicke auf '📄 Auszug jetzt einlesen'.")
        tr_pdf = st.file_uploader("PDF auswählen", type=["pdf"], key="tr_pdf_file_input")
        if tr_pdf is not None:
            if st.button("📄 Auszug jetzt einlesen", width="stretch"):
                imported_items, imported_cash = parse_trade_republic_pdf(tr_pdf)
                st.session_state["v_portfolio"] = imported_items
                st.session_state["v_cash"] = float(imported_cash)
                st.session_state["last_auto_run_ts"] = 0.0
                st.success(f"✅ {len(imported_items)} Positionen & Cash ({fmt_eur(st.session_state['v_cash'])}) eingelesen!")
                st.rerun()

    display_cash = float(st.session_state.get("v_cash", 0.0))
    st.info(f"💶 **Cash (aus Auszug):** `{fmt_eur(display_cash)}`")

    if st.button("🗑️ Depot & Cash leeren", width="stretch"):
        st.session_state["v_portfolio"] = []
        st.session_state["v_cash"] = 0.0
        st.session_state.pop("ai_signals", None)
        st.session_state.pop("ai_market", None)
        st.session_state.pop("ai_depot", None)
        st.session_state.pop("ai_cluster", None)
        st.session_state["last_auto_run_ts"] = 0.0
        st.rerun()

    st.divider()
    st.subheader("🔍 Aktie manuell hinzufügen:")
    search_query = st.text_input("Name oder Symbol:", placeholder="z. B. Apple, Tesla...")
    if search_query:
        results = search_ticker_candidates(search_query)
        if results:
            selected_cand = st.selectbox("Treffer:", results, key="side_search_select")
            in_money = st.number_input("Investierter Betrag (€):", min_value=1.0, value=50.0, step=10.0)
            
            if st.button("➕ Hinzufügen", width="stretch"):
                sym = clean_ticker(selected_cand)
                disp_name = get_display_name(sym)
                st.session_state["v_portfolio"].append({
                    "ticker": sym,
                    "name": disp_name,
                    "shares": 1.0,
                    "buy_price": float(in_money)
                })
                st.session_state["last_auto_run_ts"] = 0.0
                st.success(f"✅ {disp_name} hinzugefügt!")
                st.rerun()

    st.divider()
    st.subheader("📋 Eingelesene Positionen:")
    portfolio_list = st.session_state.get("v_portfolio", [])
    if portfolio_list:
        for idx, item in enumerate(list(portfolio_list)):
            disp_name = get_display_name(item.get("ticker", ""), item.get("name"))
            col_pos_a, col_pos_b = st.columns([3, 1])
            with col_pos_a:
                st.write(f"• **{disp_name}**")
                st.caption(f"Einsatz: {fmt_eur(float(item.get('buy_price', 0.0)))}")
            with col_pos_b:
                if st.button("❌", key=f"del_item_{idx}_{item.get('ticker','')}"):
                    st.session_state["v_portfolio"].pop(idx)
                    st.rerun()
    else:
        st.info("Noch kein Auszug geladen (0 Positionen).")

    st.divider()
    st.header("🤖 KI-Einstellungen")
    auto_refresh_active = st.toggle("🔄 Auto-Update alle 10 Min.", value=True)
    available_models = get_account_models(GROQ_KEY)
    selected_model = st.selectbox("KI-Modell:", available_models, index=0)

# BERECHNUNG DER DEPOT-DATEN (Exakt nach Realwerten)
stock_df, ticker_news, resolved_tickers = get_stock_data(portfolio_list)

if not stock_df.empty and "_raw_invested" in stock_df.columns:
    total_invested = stock_df["_raw_invested"].sum()
    stock_val = stock_df["_raw_val"].sum()
    stock_pnl = stock_df["_raw_pnl"].sum()
else:
    total_invested = sum([float(x.get("buy_price", 0.0)) for x in portfolio_list])
    stock_val = total_invested
    stock_pnl = 0.0

total_tr_account = stock_val + display_cash
stock_pnl_pct = (stock_pnl / total_invested * 100.0) if total_invested > 0 else 0.0

# METRIKEN OBEN
c_m1, c_m2, c_m3 = st.columns(3)
with c_m1:
    st.metric("TR Gesamtkonto", fmt_eur(total_tr_account), help="Bargeld (aus Auszug) + Gesamtwert deiner Aktien")
with c_m2:
    st.metric("Eingezahltes Geld", fmt_eur(total_invested), help="Dein tatsächlich eingesetztes Kapital")
with c_m3:
    st.metric("Gewinn / Verlust", fmt_eur(stock_pnl), delta=f"{stock_pnl_pct:+.2f}%")

# AUTO-REFRESH LOGIK
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
                st.rerun()

with col_info:
    if st.session_state.get("last_analysis_time"):
        st.write(f"🕒 Stand: **{st.session_state['last_analysis_time']}**")
        if auto_refresh_active:
            elapsed = now_ts - last_ts
            remaining = max(0, int(ten_mins - elapsed))
            st.caption(f"⏳ Nächstes Auto-Update in ca. **{remaining // 60}m {remaining % 60:02d}s**")
    else:
        st.caption("Lade ein Depot hoch, um die Analyse zu starten.")

# 8 TABS
tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🏦 TR-Konto",
    "🌍 Nachrichten",
    "💼 Stimmung & Empfehlungen",
    "📅 Termine & Cashflow",
    "🎯 Top 5 Kaufempfehlungen",
    "📊 Charts",
    "🥧 Streuung",
    "⚔️ Duell"
])

# TAB 0: TR KONTO
with tab0:
    st.info("ℹ️ **Kurzinfo:** Zeigt dein reales Trade Republic Depot – getrennt nach Bargeld (Cash aus Auszug) und dem aktuellen Wert deiner Wertpapiere.")
    col_tr1, col_tr2 = st.columns(2)
    with col_tr1:
        st.success(f"💶 **Bargeld (Cash aus Auszug):** {fmt_eur(display_cash)}")
    with col_tr2:
        st.success(f"📈 **Aktueller Wert deiner Aktien:** {fmt_eur(stock_val)}")
        
    st.subheader("Deine Positionen im Überblick:")
    if not stock_df.empty:
        disp_cols = [c for c in ["Unternehmen", "Handelsempfehlung", "Dein Geldeinsatz", "Börsenkurs", "Aktueller Wert (TR)", "Gewinn / Verlust"] if c in stock_df.columns]
        st.dataframe(stock_df[disp_cols], hide_index=True, width="stretch")
    else:
        st.info("📂 Lade deinen Trade Republic Kontoauszug (PDF) in der linken Seitenleiste hoch, um deine Werte hier zu sehen.")

# TAB 1: WELT-NACHRICHTEN
with tab1:
    st.info("ℹ️ **Kurzinfo:** Scannt Finanzquellen und fasst die wichtigsten Markt-Ereignisse zusammen.")
    if st.session_state.get("ai_market"):
        st.caption(f"🕒 Stand (deutsche Zeit): **{st.session_state.get('last_analysis_time', '')}**")
        st.markdown(st.session_state["ai_market"])
    else:
        st.info("Lade dein Depot hoch, um die Marktanalyse automatisch zu laden.")

# TAB 2: STIMMUNG & EMPFEHLUNGEN FÜR BESTEHENDE AKTIEN
with tab2:
    st.info("ℹ️ **Kurzinfo:** Einzelanalyse & konkrete Kauf-/Verkaufsempfehlungen für jede deiner bestehenden Aktien.")
    if not stock_df.empty:
        disp_cols = [c for c in ["Unternehmen", "Handelsempfehlung", "Dein Geldeinsatz", "Börsenkurs", "Aktueller Wert (TR)", "Gewinn / Verlust"] if c in stock_df.columns]
        st.dataframe(stock_df[disp_cols], hide_index=True, width="stretch")
    if st.session_state.get("ai_depot"):
        st.divider()
        st.subheader("🤖 KI-Handelsempfehlungen für dein Depot:")
        st.markdown(st.session_state["ai_depot"])
    else:
        st.info("Keine Daten geladen.")

# TAB 3: TERMINE & CASHFLOW
with tab3:
    st.info("ℹ️ **Kurzinfo:** Zeigt Termine für Quartalszahlen und dein passives Einkommen (Ausschüttungen in €).")
    if not stock_df.empty:
        total_annual_div = stock_df["_raw_cashflow"].sum()
        col_cf1, col_cf2 = st.columns(2)
        with col_cf1:
            st.success(f"💰 **Erwartete Ausschüttung:** `{total_annual_div:.2f} € / Jahr`")
        with col_cf2:
            st.info(f"📊 **Durchschnittliche Rendite:** `{(total_annual_div / stock_val * 100.0) if stock_val > 0 else 0.0:.2f} % p.a.`")

        st.subheader("📅 Terminkalender & Ausschüttungen:")
        disp_cols = [c for c in ["Unternehmen", "Nächste Quartalszahlen", "Dividendenrendite", "Ausschüttung pro Jahr", "Ausschüttungs-Monate"] if c in stock_df.columns]
        st.dataframe(stock_df[disp_cols], hide_index=True, width="stretch")
    else:
        st.info("Keine Positionen eingelesen.")

# TAB 4: TOP 5 KAUFEMPFEHLUNGEN
with tab4:
    st.info("ℹ️ **Kurzinfo:** KI-Ratgeber zur Portfolio-Erweiterung: 5 konkrete Top-Aktien basierend auf der aktuellen Welt- und Marktlage.")
    if st.session_state.get("ai_signals"):
        st.caption(f"🕒 Stand (deutsche Zeit): **{st.session_state.get('last_analysis_time', '')}**")
        st.markdown(st.session_state["ai_signals"])
    else:
        st.info("Lade ein Depot ein, um die Kaufempfehlungen zu sehen.")

# TAB 5: CHARTS
with tab5:
    st.info("ℹ️ **Kurzinfo:** Interaktive Performance-Verläufe deiner Aktien im gewählten Zeitraum.")
    if portfolio_list:
        col_chart_focus, col_chart_period = st.columns([3, 2])
        
        series_names = [get_display_name(x.get("ticker", ""), x.get("name")) for x in portfolio_list]
        with col_chart_focus:
            selected_view = st.selectbox("Fokus-Ansicht:", ["Alle Aktien gleichzeitig"] + series_names, index=0, key="chart_focus_view")
            
        with col_chart_period:
            selected_period = st.radio("Zeitraum:", ["1W", "1M", "6M", "1J", "Max"], index=1, horizontal=True, key="chart_period_radio")

        series_dict = get_individual_series_dict(portfolio_list, period=selected_period)
        if series_dict:
            available_names = list(series_dict.keys())
            palette = ["#00D084", "#0693E3", "#FCB900", "#EB144C", "#9B51E0", "#00ACC1", "#FF6900", "#D81B60", "#8E24AA", "#00E676"]
            fig = go.Figure()
            names_to_plot = available_names if selected_view == "Alle Aktien gleichzeitig" else [selected_view]

            for i, name in enumerate(names_to_plot):
                if name in series_dict:
                    s = series_dict[name]
                    fig.add_trace(go.Scatter(
                        x=s.index, 
                        y=s.values, 
                        mode="lines", 
                        name=name,
                        line=dict(width=2.5, color=palette[i % len(palette)]),
                        hovertemplate=f"<b>{name}</b>: %{{y:+.2f}}%<extra></extra>"
                    ))
            
            fig.update_layout(
                title=f"Performance-Entwicklung ({selected_period})",
                xaxis=dict(title="Datum", type="date", autorange=True),
                yaxis=dict(title="Rendite / Entwicklung (%)", ticksuffix="%", autorange=True),
                hovermode="x unified",
                margin=dict(l=5, r=5, t=40, b=5),
                height=420,
                uirevision=f"{selected_period}_{selected_view}"
            )
            st.plotly_chart(fig, width="stretch", key=f"plotly_chart_perf_{selected_period}_{selected_view}")
    else:
        st.info("Lade deinen TR-Kontoauszug hoch, um die Performance-Diagramme anzuzeigen.")

# TAB 6: RISIKOSTREUUNG (Mit eigenständigen Farbwelten)
with tab6:
    st.info("ℹ️ **Kurzinfo:** Prüft die Verteilung deines realen Geldes auf Rollen, Branchen und Länder.")
    if not stock_df.empty:
        c_pie1, c_pie2, c_pie3 = st.columns(3)
        
        # 3 VÖLLIG GETRENNTE FARBWELTEN
        colors_role = ["#2563EB", "#0284C7", "#0D9488", "#64748B", "#3B82F6"]  # Blau/Cyan/Grau
        colors_sector = ["#059669", "#10B981", "#34D399", "#14B8A6", "#047857", "#6EE7B7", "#065F46"]  # Smaragd/Grün
        colors_geo = ["#D97706", "#F59E0B", "#8B5CF6", "#EC4899", "#6366F1", "#FB923C", "#A855F7"]  # Amber/Violett/Pink
        
        pie_df = stock_df.copy()
        if pie_df["_raw_val"].sum() <= 0:
            pie_df["_raw_val"] = 1.0

        with c_pie1:
            fig_role = px.pie(
                pie_df, names="Rolle", values="_raw_val",
                title="1. Rollen im Depot", hole=0.45,
                color_discrete_sequence=colors_role
            )
            fig_role.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig_role, width="stretch", key="pie_role")

        with c_pie2:
            fig_sec = px.pie(
                pie_df, names="Sektor", values="_raw_val",
                title="2. Branchen-Aufteilung", hole=0.45,
                color_discrete_sequence=colors_sector
            )
            fig_sec.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig_sec, width="stretch", key="pie_sec")

        with c_pie3:
            fig_geo = px.pie(
                pie_df, names="Land", values="_raw_val",
                title="3. Länder-Aufteilung", hole=0.45,
                color_discrete_sequence=colors_geo
            )
            fig_geo.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig_geo, width="stretch", key="pie_geo")

        st.divider()
        st.subheader("🧩 Wie verteilen sich deine tatsächlichen Aktien?")
        
        unique_roles = stock_df["Rolle"].unique()
        r_cols = st.columns(2)
        
        for idx, r_name in enumerate(unique_roles):
            sub_df = stock_df[stock_df["Rolle"] == r_name]
            stock_names = ", ".join(sub_df["Unternehmen"].tolist())
            role_sum = sub_df["_raw_val"].sum()
            role_pct = (role_sum / stock_val * 100.0) if stock_val > 0 else (100.0 / len(unique_roles))
            
            with r_cols[idx % 2]:
                with st.container(border=True):
                    st.markdown(f"### {r_name}")
                    st.write(f"**Deine Aktien hier:** {stock_names}")
                    st.write(f"**Anteil am Depot:** `{fmt_eur(role_sum)}` ({role_pct:.1f} %)")

    if st.session_state.get("ai_cluster"):
        st.divider()
        st.subheader("🛡️ KI-Gutachten & Empfehlungen zur Absicherung:")
        st.markdown(st.session_state["ai_cluster"])

# TAB 7: AKTIEN-VERGLEICH
with tab7:
    st.info("ℹ️ **Kurzinfo:** Direktes 1-gegen-1-Duell zweier beliebiger Aktien aus deinem Depot.")
    if not stock_df.empty and len(stock_df) >= 2:
        cd1, cd2 = st.columns(2)
        names_list = stock_df["Unternehmen"].tolist()
        with cd1:
            duel_a = st.selectbox("Erste Aktie:", names_list, index=0, key="duel_sel_a")
        with cd2:
            duel_b = st.selectbox("Zweite Aktie:", names_list, index=1, key="duel_sel_b")
        
        if st.button("⚡ Duell auswerten", width="stretch", key="btn_duel_run"):
            if GROQ_KEY:
                cl = Groq(api_key=GROQ_KEY)
                row_a = stock_df[stock_df["Unternehmen"] == duel_a].iloc[0].to_dict()
                row_b = stock_df[stock_df["Unternehmen"] == duel_b].iloc[0].to_dict()
                with st.spinner("Analysiere Duell..."):
                    res_duel = run_duel_analysis(cl, selected_model, str(row_a), str(row_b))
                    st.markdown(res_duel)
    else:
        st.info("Mindestens 2 Positionen nötig, um ein Duell zu starten.")
