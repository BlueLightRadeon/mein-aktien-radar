import os
import importlib.util
import time
from datetime import datetime, timezone, timedelta
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from groq import Groq

# Backend-Services importieren
try:
    from storage_service import load_saved_portfolio, save_saved_portfolio, delete_saved_portfolio
    from data_service import (
        get_stock_data, get_individual_series_dict, fetch_all_headlines, search_ticker_candidates, 
        clean_ticker, parse_trade_republic_pdf, get_display_name
    )
    from ai_service import get_account_models, run_analysis, run_duel_analysis
    from learning_service import (
        load_memory, save_memory, fetch_365d_stats, 
        audit_and_update_learning, run_ai_learning_prediction, 
        generate_realistic_30d_forecast_path
    )
except Exception as e:
    st.error(f"❌ Fehler in den Service-Dateien (Hauptordner): {e}")
    st.stop()

# --- INTEGRIERTE FALLBACK-RENDERER (Verhindern jeden Absturz) ---
def fallback_tab0(stock_df, display_cash, stock_val, fmt_eur):
    st.info("ℹ️ **Kurzinfo:** Zeigt dein reales Trade Republic Depot – getrennt nach Bargeld und Wertpapieren.")
    col_tr1, col_tr2 = st.columns(2)
    with col_tr1:
        st.success(f"💶 **Bargeld (Cash aus Auszug):** {fmt_eur(display_cash)}")
    with col_tr2:
        st.success(f"📈 **Aktueller Wert deiner Aktien:** {fmt_eur(stock_val)}")
    st.subheader("Deine Positionen im Überblick:")
    if not stock_df.empty:
        disp_cols = [c for c in ["Unternehmen", "Aktueller Wert (TR)", "Börsenkurs", "Gewinn / Verlust"] if c in stock_df.columns]
        st.dataframe(stock_df[disp_cols], hide_index=True, width="stretch")
    else:
        st.info("📂 Lade deinen TR-Kontoauszug in der Seitenleiste hoch.")

def fallback_tab1(stock_df):
    st.info("ℹ️ **Kurzinfo:** Vollständige Gegenüberstellung: Aktuelle Empfehlungen vs. 3-Monats-Analyse.")
    if not stock_df.empty:
        disp_cols = [c for c in ["Unternehmen", "Aktuelle KI-Empfehlung", "Aktuelle Begründung", "Empfehlung (vor 3 Monaten)", "Begründung (vor 3 Monaten)", "Börsenkurs", "Fair Value", "Analysten-Kursziel", "Gewinn / Verlust"] if c in stock_df.columns]
        st.dataframe(stock_df[disp_cols], hide_index=True, width="stretch")
    depot_text = st.session_state.get("ai_depot", "")
    if depot_text and len(depot_text.strip()) > 15:
        st.divider()
        st.subheader("🤖 Detaillierter KI-Handelsbericht für deine Aktien:")
        st.markdown(depot_text)
    else:
        st.info("Klicke oben auf '🚀 Jetzt sofort manuell aktualisieren', um den Handelsbericht zu berechnen.")

def fallback_tab2(last_analysis_time):
    st.info("ℹ️ **Kurzinfo:** Scannt Finanzquellen und fasst die wichtigsten Markt-Ereignisse zusammen.")
    market_text = st.session_state.get("ai_market", "")
    if market_text and len(market_text.strip()) > 15:
        st.caption(f"🕒 Stand (deutsche Zeit): **{last_analysis_time}**")
        st.markdown(market_text)
    else:
        st.info("Klicke oben auf '🚀 Jetzt sofort manuell aktualisieren', um den Marktbericht zu laden.")

def fallback_tab3(stock_df, stock_val):
    st.info("ℹ️ **Kurzinfo:** Zeigt Termine für Quartalszahlen und passives Einkommen.")
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

def fallback_tab4(last_analysis_time):
    st.info("ℹ️ **Kurzinfo:** KI-Ratgeber zur Portfolio-Erweiterung: 5 konkrete Top-Aktien.")
    signals_text = st.session_state.get("ai_signals", "")
    if signals_text and len(signals_text.strip()) > 15:
        st.caption(f"🕒 Stand (deutsche Zeit): **{last_analysis_time}**")
        st.markdown(signals_text)
    else:
        st.info("Klicke oben auf '🚀 Jetzt sofort manuell aktualisieren', um die Kaufempfehlungen zu laden.")

def fallback_tab5(portfolio_list):
    st.info("ℹ️ **Kurzinfo:** Interaktive Performance-Verläufe deiner Aktien im gewählten Zeitraum.")
    if portfolio_list:
        col_chart_focus, col_chart_period = st.columns([3, 2])
        series_names = [get_display_name(x.get("ticker", ""), x.get("name")) for x in portfolio_list]
        options = ["Alle Aktien gleichzeitig"] + series_names
        with col_chart_focus:
            prev_view = st.session_state.get("chart_focus_view", options[0])
            sel_idx = options.index(prev_view) if prev_view in options else 0
            selected_view = st.selectbox("Fokus-Ansicht:", options, index=sel_idx, key="chart_focus_view")
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
                    fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name=name, line=dict(width=2.5, color=palette[i % len(palette)]), hovertemplate=f"<b>{name}</b>: %{{y:+.2f}}%<extra></extra>"))
            fig.update_layout(title=f"Performance-Entwicklung ({selected_period})", xaxis=dict(title="Datum", type="date", autorange=True), yaxis=dict(title="Rendite / Entwicklung (%)", ticksuffix="%", autorange=True), hovermode="x unified", margin=dict(l=5, r=5, t=40, b=5), height=420, uirevision=f"{selected_period}_{selected_view}")
            st.plotly_chart(fig, width="stretch", key=f"plotly_chart_perf_{selected_period}_{selected_view}")
    else:
        st.info("Lade deinen TR-Kontoauszug hoch, um die Diagramme anzuzeigen.")

def fallback_tab6(stock_df, stock_val, fmt_eur):
    st.info("ℹ️ **Kurzinfo:** Prüft die Verteilung deines realen Geldes auf Rollen, Branchen und Länder.")
    if not stock_df.empty:
        c_pie1, c_pie2, c_pie3 = st.columns(3)
        colors_role = ["#2563EB", "#0284C7", "#0D9488", "#64748B", "#3B82F6"]
        colors_sector = ["#059669", "#10B981", "#34D399", "#14B8A6", "#047857", "#6EE7B7", "#065F46"]
        colors_geo = ["#D97706", "#F59E0B", "#8B5CF6", "#EC4899", "#6366F1", "#FB923C", "#A855F7"]
        pie_df = stock_df.copy()
        if pie_df["_raw_val"].sum() <= 0:
            pie_df["_raw_val"] = 1.0
        with c_pie1:
            fig_role = px.pie(pie_df, names="Rolle", values="_raw_val", title="1. Rollen im Depot", hole=0.45, color_discrete_sequence=colors_role)
            fig_role.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig_role, width="stretch", key="pie_role")
        with c_pie2:
            fig_sec = px.pie(pie_df, names="Sektor", values="_raw_val", title="2. Branchen-Aufteilung", hole=0.45, color_discrete_sequence=colors_sector)
            fig_sec.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig_sec, width="stretch", key="pie_sec")
        with c_pie3:
            fig_geo = px.pie(pie_df, names="Land", values="_raw_val", title="3. Länder-Aufteilung", hole=0.45, color_discrete_sequence=colors_geo)
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
    cluster_text = st.session_state.get("ai_cluster", "")
    if cluster_text and len(cluster_text.strip()) > 15:
        st.divider()
        st.subheader("🛡️ KI-Gutachten & Empfehlungen zur Absicherung:")
        st.markdown(cluster_text)

def fallback_tab7(stock_df, selected_model, groq_key):
    st.info("ℹ️ **Kurzinfo:** Direktes 1-gegen-1-Duell zweier beliebiger Aktien aus deinem Depot.")
    if not stock_df.empty and len(stock_df) >= 2:
        cd1, cd2 = st.columns(2)
        names_list = stock_df["Unternehmen"].tolist()
        with cd1:
            duel_a = st.selectbox("Erste Aktie:", names_list, index=0, key="duel_sel_a")
        with cd2:
            duel_b = st.selectbox("Zweite Aktie:", names_list, index=1, key="duel_sel_b")
        if st.button("⚡ Duell auswerten", width="stretch", key="btn_duel_run"):
            if groq_key:
                cl = Groq(api_key=groq_key)
                row_a = stock_df[stock_df["Unternehmen"] == duel_a].iloc[0].to_dict()
                row_b = stock_df[stock_df["Unternehmen"] == duel_b].iloc[0].to_dict()
                with st.spinner("Analysiere Duell..."):
                    res_duel = run_duel_analysis(cl, selected_model, str(row_a), str(row_b))
                    st.markdown(res_duel)
    else:
        st.info("Mindestens 2 Positionen nötig, um ein Duell zu starten.")

def fallback_tab8(portfolio_list, selected_model, groq_key):
    st.info("ℹ️ **Kurzinfo:** 30-Tage-Hochpräzisionsprognose: Die KI vergleicht ihre Prognosen autonom mit Realkursen.")
    memory = load_memory()
    with st.expander("💾 Wissensspeicher dauerhaft sichern & wiederherstellen"):
        col_bk1, col_bk2 = st.columns(2)
        with col_bk1:
            mem_json_str = json.dumps(memory, indent=2, ensure_ascii=False)
            st.download_button(label="📥 Gespeicherten Lernstand herunterladen (JSON)", data=mem_json_str, file_name="stock_memory.json", mime="application/json", width="stretch")
        with col_bk2:
            uploaded_mem = st.file_uploader("📤 Gesichertes Gedächtnis einspielen", type=["json"], key="restore_mem_uploader")
            if uploaded_mem is not None:
                try:
                    loaded_data = json.load(uploaded_mem)
                    save_memory(loaded_data)
                    st.success("✅ Wissensstand erfolgreich wiederhergestellt!")
                    time.sleep(0.8)
                    st.rerun()
                except Exception as e:
                    st.error(f"Fehlerhafte Datei: {e}")

    if portfolio_list:
        selected_stock_name = st.selectbox("Wähle eine Aktie aus deinem Depot zum Lernen & Prognostizieren:", [get_display_name(x.get("ticker", ""), x.get("name")) for x in portfolio_list], key="learning_stock_picker")
        chosen_ticker = "NVDA"
        for item in portfolio_list:
            if get_display_name(item.get("ticker", ""), item.get("name")) == selected_stock_name:
                chosen_ticker = clean_ticker(item.get("ticker", ""))
                break
        current_profile = memory.get(chosen_ticker, {}).get("learning_profile", {})
        with st.container(border=True):
            st.markdown(f"#### 🧠 Autonomes Lern-Dashboard: **{selected_stock_name}**")
            l_col1, l_col2, l_col3, l_col4 = st.columns(4)
            with l_col1:
                st.metric("Geprüfte Prognosen", f"{current_profile.get('total_evaluations', 0)}")
            with l_col2:
                st.metric("Richtungstreffer (Auf/Ab)", f"{current_profile.get('direction_accuracy_pct', 100.0):.1f} %")
            with l_col3:
                st.metric("Mittlere Abweichung (Fehler)", f"{current_profile.get('avg_error_pct', 0.0):.1f} %")
            with l_col4:
                b_fac = current_profile.get('bias_factor', 1.0)
                st.metric("Aktiver Bias-Kompensator", f"{b_fac:.3f}x")

        col_learn1, col_learn2 = st.columns([2, 1])
        with col_learn1:
            st.write(f"Aktie im Fokus: **{selected_stock_name}** (`{chosen_ticker}`)")
        with col_learn2:
            start_learn = st.button("🧠 30-Tage Präzisionsanalyse starten", type="primary", width="stretch")

        if start_learn:
            with st.spinner(f"Führe Soll-Ist-Abgleich durch & berechne neue Prognose für {selected_stock_name}..."):
                stats_raw, history_series = fetch_365d_stats(chosen_ticker)
                if stats_raw and history_series is not None:
                    current_profile = audit_and_update_learning(chosen_ticker, stats_raw["current_price"], history_series, memory)
                    bias_factor = current_profile.get("bias_factor", 1.0)
                    stats, _ = fetch_365d_stats(chosen_ticker, bias_factor=bias_factor)
                    if not stats:
                        stats = stats_raw
                    client = Groq(api_key=groq_key)
                    news_data = fetch_all_headlines()
                    macro_news_str = "\n".join(news_data) if news_data else "Stabile Weltwirtschaftslage."
                    pred_res, targets_dict = run_ai_learning_prediction(client, selected_model, chosen_ticker, selected_stock_name, stats, memory, macro_news=macro_news_str)
                    st.session_state[f"pred_{chosen_ticker}"] = pred_res
                    st.session_state[f"stats_{chosen_ticker}"] = stats
                    st.session_state[f"targets_{chosen_ticker}"] = targets_dict
                    st.session_state[f"history_{chosen_ticker}"] = history_series
                    st.success("✅ 30-Tage-Präzisionsanalyse abgeschlossen, gelernt & gesichert!")
                else:
                    st.error("Konnte historische Börsendaten nicht vollständig laden.")

        if f"stats_{chosen_ticker}" in st.session_state and f"targets_{chosen_ticker}" in st.session_state:
            s = st.session_state[f"stats_{chosen_ticker}"]
            tg = st.session_state[f"targets_{chosen_ticker}"]
            curr_sym = tg.get("currency_symbol", s.get("currency_symbol", "€"))
            p_base = float(s.get("current_price", 1.0) or 1.0)
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            with m_col1:
                st.metric("365-Tage Rendite", f"{s['return_365d_pct']:+.2f} %")
            with m_col2:
                st.metric("Bollinger-Spanne (20T)", f"{s.get('bb_upper', 0)} {curr_sym}", delta=f"Unten: {s.get('bb_lower', 0)} {curr_sym}")
            with m_col3:
                st.metric("Tagesschwankung (ATR)", f"±{s.get('atr_14', 0)} {curr_sym}")
            with m_col4:
                st.metric("MACD / Trend", f"{s.get('trend_status', 'Neutral')[:18]}")

            val_7 = float(tg.get("t_7") or p_base)
            val_14 = float(tg.get("t_14") or tg.get("t_30") or p_base)
            val_30 = float(tg.get("t_30") or p_base)
            val_bear = float(tg.get("t_bear") or (p_base * 0.95))
            diff_7 = ((val_7 - p_base) / p_base) * 100.0 if p_base > 0 else 0.0
            diff_14 = ((val_14 - p_base) / p_base) * 100.0 if p_base > 0 else 0.0
            diff_30 = ((val_30 - p_base) / p_base) * 100.0 if p_base > 0 else 0.0

            p_col1, p_col2, p_col3, p_col4 = st.columns(4)
            with p_col1:
                st.metric("Ziel in 7 Tagen", f"{val_7:.2f} {curr_sym}", delta=f"{diff_7:+.2f} %")
            with p_col2:
                st.metric("Ziel in 14 Tagen", f"{val_14:.2f} {curr_sym}", delta=f"{diff_14:+.2f} %")
            with p_col3:
                st.metric("Ziel in 30 Tagen (Hauptpfad)", f"{val_30:.2f} {curr_sym}", delta=f"{diff_30:+.2f} %")
            with p_col4:
                st.metric("Absicherung (Stop-Loss)", f"{val_bear:.2f} {curr_sym}")

        if f"history_{chosen_ticker}" in st.session_state and f"targets_{chosen_ticker}" in st.session_state:
            h_series = st.session_state[f"history_{chosen_ticker}"]
            tg = st.session_state[f"targets_{chosen_ticker}"]
            s = st.session_state[f"stats_{chosen_ticker}"]
            curr_sym = tg.get("currency_symbol", "€")
            p_base = float(s.get("current_price", 1.0) or 1.0)
            zoom_choice = st.radio("Historischer Kontext im Chart:", ["Letzte 30 Tage + 30T Prognose (Fokus)", "Letzte 90 Tage + Prognose", "Volle 365 Tage"], horizontal=True, key=f"zoom_{chosen_ticker}")
            plot_history = h_series.tail(30) if "30 Tage +" in zoom_choice else (h_series.tail(90) if "90 Tage" in zoom_choice else h_series)
            last_date = h_series.index[-1]

            val_7 = float(tg.get("t_7") or p_base)
            val_14 = float(tg.get("t_14") or tg.get("t_30") or p_base)
            val_30 = float(tg.get("t_30") or p_base)
            val_bull = float(tg.get("t_bull") or (p_base * 1.05))
            val_bear = float(tg.get("t_bear") or (p_base * 0.95))

            future_dates, f_prices, bull_curve, bear_curve, milestones = generate_realistic_30d_forecast_path(last_date=last_date, p_curr=p_base, t_7=val_7, t_14=val_14, t_30=val_30, t_bull=val_bull, t_bear=val_bear, volatility_pct=s.get("volatility_pct", 35.0), ticker_seed=chosen_ticker)
            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(x=plot_history.index, y=plot_history.values, mode="lines", name=f"Reale Börsenkurse ({selected_stock_name})", line=dict(color="#0693E3", width=2.5)))
            fig_pred.add_trace(go.Scatter(x=future_dates, y=bull_curve, mode="lines", name="🟢 Best-Case Korridor (30T)", line=dict(color="rgba(0, 208, 132, 0.4)", width=1, dash="dot"), hoverinfo="skip"))
            fig_pred.add_trace(go.Scatter(x=future_dates, y=bear_curve, mode="lines", name="🔴 Absicherungs-Kanal (30T)", fill='tonexty', fillcolor='rgba(0, 208, 132, 0.10)', line=dict(color="rgba(235, 20, 76, 0.4)", width=1, dash="dot"), hoverinfo="skip"))
            fig_pred.add_trace(go.Scatter(x=future_dates, y=f_prices, mode="lines", name="🎯 30-Tage KI-Prognose (Tagespfad)", line=dict(color="#FCB900", width=2.5, dash="dash"), hovertemplate="Prognose (%{x|%d. %b}): <b>%{y:.2f} " + curr_sym + "</b><extra></extra>"))
            fig_pred.add_trace(go.Scatter(x=milestones["dates"], y=milestones["prices"], mode="markers+text", name="📍 Meilensteine", text=[f"{p:.1f}" for p in milestones["prices"]], textposition="top center", textfont=dict(color="#FCB900", size=10), marker=dict(size=8, color="#FCB900", symbol="diamond"), hovertemplate="%{text}: <b>%{y:.2f} " + curr_sym + "</b><extra></extra>"))
            fig_pred.update_layout(title=f"30-Tage Hochpräzisions-Kursprognose für {selected_stock_name}", xaxis=dict(title="Datum", type="date"), yaxis=dict(title=f"Kurs ({curr_sym})", ticksuffix=f" {curr_sym}"), hovermode="x unified", height=450, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_pred, width="stretch", key=f"plotly_pred_chart_{chosen_ticker}")

        if f"pred_{chosen_ticker}" in st.session_state:
            st.divider()
            st.markdown(st.session_state[f"pred_{chosen_ticker}"])

        if chosen_ticker in memory and "history" in memory[chosen_ticker]:
            with st.expander(f"📚 Gespeichertes Wissensgedächtnis für {selected_stock_name} ({len(memory[chosen_ticker]['history'])} Lernpunkte)"):
                for entry in reversed(memory[chosen_ticker]["history"]):
                    c_sym = entry.get("currency", "€")
                    status_badge = "✅ Richtung korrekt" if entry.get("direction_correct") else ("⚠️ Abweichung" if "direction_correct" in entry else "⏳ Läuft")
                    st.write(f"• **{entry.get('date')}** | Basis: `{entry.get('price')} {c_sym}` | Ziel 30T: `{entry.get('target_30d')} {c_sym}` | Status: **{status_badge}**")
    else:
        st.info("Lade ein Depot hoch, um das Lernlabor zu aktivieren.")

# --- DYNAMISCHER TAB-LOADER (Lädt Datei aus tabs/ oder nutzt Fallback) ---
fallbacks = {
    0: fallback_tab0, 1: fallback_tab1, 2: fallback_tab2,
    3: fallback_tab3, 4: fallback_tab4, 5: fallback_tab5,
    6: fallback_tab6, 7: fallback_tab7, 8: fallback_tab8
}

def get_tab_renderer(tab_idx):
    tabs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tabs")
    if os.path.exists(tabs_path):
        for fname in os.listdir(tabs_path):
            if fname.startswith((".", "__")):
                continue
            # Sucht fehlertolerant nach der Ziffer des Tabs im Namen
            clean = "".join(c for c in fname if c.isalnum() or c in "._-")
            if str(tab_idx) in clean:
                try:
                    full_path = os.path.join(tabs_path, fname)
                    spec = importlib.util.spec_from_file_location(f"tab_{tab_idx}", full_path)
                    if spec and spec.loader:
                        mod = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(mod)
                        if hasattr(mod, "render"):
                            return mod.render
                except Exception:
                    pass
    return fallbacks[tab_idx]

# --- INITIALISIERUNG ---
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

# --- TABS RENDERN (100% fehlertolerant) ---
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
    get_tab_renderer(0)(stock_df, display_cash, stock_val, fmt_eur)

with tab1:
    get_tab_renderer(1)(stock_df)

with tab2:
    get_tab_renderer(2)(st.session_state.get("last_analysis_time", ""))

with tab3:
    get_tab_renderer(3)(stock_df, stock_val)

with tab4:
    get_tab_renderer(4)(st.session_state.get("last_analysis_time", ""))

with tab5:
    get_tab_renderer(5)(portfolio_list)

with tab6:
    get_tab_renderer(6)(stock_df, stock_val, fmt_eur)

with tab7:
    get_tab_renderer(7)(stock_df, selected_model, GROQ_KEY)

with tab8:
    get_tab_renderer(8)(portfolio_list, selected_model, GROQ_KEY)
