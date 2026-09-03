import streamlit as st
import json
import time
import pandas as pd
import plotly.graph_objects as go
from groq import Groq

from data_service import clean_ticker, get_display_name, fetch_all_headlines
from learning_service import (
    load_memory, save_memory, fetch_365d_stats, 
    audit_and_update_learning, run_ai_learning_prediction, 
    generate_realistic_30d_forecast_path
)

def render(portfolio_list, selected_model, groq_key):
    st.info("ℹ️ **Kurzinfo:** 30-Tage-Hochpräzisionsprognose: Die KI vergleicht frühere Prognosen mit echten Börsenkursen und passt künftige Kurspfade autonom an.")
    
    memory = load_memory()

    if not portfolio_list:
        st.warning("📂 Lade zuerst dein Depot hoch, um das Lernlabor zu aktivieren.")
        return

    selected_stock_name = st.selectbox(
        "Wähle eine Aktie aus deinem Depot zum Lernen & Prognostizieren:",
        [get_display_name(x.get("ticker", ""), x.get("name")) for x in portfolio_list],
        key="learning_stock_picker"
    )
    
    chosen_ticker = "NVDA"
    for item in portfolio_list:
        if get_display_name(item.get("ticker", ""), item.get("name")) == selected_stock_name:
            chosen_ticker = clean_ticker(item.get("ticker", ""))
            break

    # 1. AKTION-BUTTON GANZ OBEN
    col_btn, col_info = st.columns([2, 2])
    with col_btn:
        start_learn = st.button("🧠 30-Tage Präzisionsanalyse jetzt berechnen", type="primary", width="stretch", key=f"btn_calc_learn_{chosen_ticker}")
    with col_info:
        st.caption(f"Fokus: **{selected_stock_name}** (`{chosen_ticker}`) | Modell: `{selected_model}`")

    # 2. BERECHNUNG DIREKT AN ORT UND STELLE
    if start_learn:
        if not groq_key:
            st.error("⚠️ Bitte hinterlege deinen GROQ_API_KEY in den Secrets.")
        else:
            with st.spinner(f"Führe Soll-Ist-Abgleich durch & erstelle 30-Tage-Prognose für {selected_stock_name}..."):
                stats_raw, history_series = fetch_365d_stats(chosen_ticker)
                if stats_raw and history_series is not None:
                    # Soll-Ist-Abgleich mit realen Börsenkursen
                    current_profile = audit_and_update_learning(chosen_ticker, stats_raw["current_price"], history_series, memory)
                    bias_factor = current_profile.get("bias_factor", 1.0)
                    
                    stats, _ = fetch_365d_stats(chosen_ticker, bias_factor=bias_factor)
                    if not stats:
                        stats = stats_raw
                    
                    client = Groq(api_key=groq_key)
                    news_data = fetch_all_headlines()
                    macro_news_str = "\n".join(news_data) if news_data else "Stabile Weltwirtschaftslage."
                    
                    pred_res, targets_dict = run_ai_learning_prediction(
                        client, selected_model, chosen_ticker, selected_stock_name, stats, memory, macro_news=macro_news_str
                    )
                    st.session_state[f"pred_{chosen_ticker}"] = pred_res
                    st.session_state[f"stats_{chosen_ticker}"] = stats
                    st.session_state[f"targets_{chosen_ticker}"] = targets_dict
                    st.session_state[f"history_{chosen_ticker}"] = history_series
                    st.success("✅ Analyse abgeschlossen, gelernt & gesichert!")
                else:
                    st.error("Historische Börsendaten konnten nicht geladen werden.")

    # 3. AUTONOMES LERN-DASHBOARD (KENNZAHLEN)
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
            st.metric("Aktiver Bias-Kompensator", f"{b_fac:.3f}x", help="Dämpfungsfaktor, der systematische Fehlprognosen korrigiert.")

    # 4. DAS NEUE TRANSPARENZ-PRÜFFENSTER (SOLL-IST-HISTORIE)
    stock_history_records = memory.get(chosen_ticker, {}).get("history", [])
    with st.expander(f"🔍 Reales Soll-Ist-Prüflabor & Treffer-Historie ({len(stock_history_records)} erfasste Scans)", expanded=False):
        st.markdown("""
        Hier prüft die KI ihre eigenen Vorhersagen gegen die echten Börsenkurse von Yahoo Finance:
        - Liegt der Kurs über/unter der Prognose, passt sich der **Bias-Kompensator** automatisch für den nächsten Lauf an.
        """)
        if stock_history_records:
            table_rows = []
            for h in reversed(stock_history_records):
                c_sym = h.get("currency", "€")
                is_checked = "error_pct" in h
                status_icon = "✅ Volltreffer" if h.get("direction_correct") else ("⚠️ Abweichung" if is_checked else "⏳ Läuft")
                table_rows.append({
                    "Datum": h.get("date", "k. A."),
                    "Startkurs": f"{h.get('price', 0.0):.2f} {c_sym}",
                    "Ziel (30T)": f"{h.get('target_30d', 0.0):.2f} {c_sym}",
                    "Realkurs heute": f"{h.get('actual_evaluated_price', '-')} {c_sym}" if is_checked else "In Beobachtung",
                    "Abweichung": f"{h.get('error_pct', '-')}%" if is_checked else "-",
                    "Ergebnis": status_icon
                })
            st.dataframe(pd.DataFrame(table_rows), hide_index=True, width="stretch")
        else:
            st.info("Noch keine historischen Scans vorhanden. Nach dem ersten Klick startet die Aufzeichnung.")

    # 5. KURSZIELE & PROGNOSE-CHART
    if f"stats_{chosen_ticker}" in st.session_state and f"targets_{chosen_ticker}" in st.session_state:
        s = st.session_state[f"stats_{chosen_ticker}"]
        tg = st.session_state[f"targets_{chosen_ticker}"]
        curr_sym = tg.get("currency_symbol", s.get("currency_symbol", "€"))
        p_base = float(s.get("current_price", 1.0) or 1.0)

        val_7 = float(tg.get("t_7") or p_base)
        val_14 = float(tg.get("t_14") or p_base)
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
        val_14 = float(tg.get("t_14") or p_base)
        val_30 = float(tg.get("t_30") or p_base)
        val_bull = float(tg.get("t_bull") or (p_base * 1.05))
        val_bear = float(tg.get("t_bear") or (p_base * 0.95))

        future_dates, f_prices, bull_curve, bear_curve, milestones = generate_realistic_30d_forecast_path(
            last_date=last_date, p_curr=p_base, t_7=val_7, t_14=val_14, t_30=val_30,
            t_bull=val_bull, t_bear=val_bear, volatility_pct=s.get("volatility_pct", 30.0), ticker_seed=chosen_ticker
        )

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=plot_history.index, y=plot_history.values, mode="lines", name=f"Reale Börsenkurse ({selected_stock_name})", line=dict(color="#0693E3", width=2.5)))
        fig_pred.add_trace(go.Scatter(x=future_dates, y=bull_curve, mode="lines", name="🟢 Best-Case Korridor (30T)", line=dict(color="rgba(0, 208, 132, 0.4)", width=1, dash="dot"), hoverinfo="skip"))
        fig_pred.add_trace(go.Scatter(x=future_dates, y=bear_curve, mode="lines", name="🔴 Absicherungs-Kanal (30T)", fill='tonexty', fillcolor='rgba(0, 208, 132, 0.10)', line=dict(color="rgba(235, 20, 76, 0.4)", width=1, dash="dot"), hoverinfo="skip"))
        fig_pred.add_trace(go.Scatter(x=future_dates, y=f_prices, mode="lines", name="🎯 30-Tage KI-Prognose", line=dict(color="#FCB900", width=2.5, dash="dash"), hovertemplate="Prognose (%{x|%d. %b}): <b>%{y:.2f} " + curr_sym + "</b><extra></extra>"))
        fig_pred.add_trace(go.Scatter(x=milestones["dates"], y=milestones["prices"], mode="markers+text", name="📍 Meilensteine", text=[f"{p:.1f}" for p in milestones["prices"]], textposition="top center", textfont=dict(color="#FCB900", size=10), marker=dict(size=8, color="#FCB900", symbol="diamond"), hovertemplate="%{text}: <b>%{y:.2f} " + curr_sym + "</b><extra></extra>"))

        fig_pred.update_layout(
            title=f"30-Tage Hochpräzisions-Kursprognose für {selected_stock_name}",
            xaxis=dict(title="Datum", type="date"),
            yaxis=dict(title=f"Kurs ({curr_sym})", ticksuffix=f" {curr_sym}"),
            hovermode="x unified", height=450, margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig_pred, width="stretch", key=f"plotly_pred_chart_{chosen_ticker}")

    # 6. AUSFÜHRLICHER KI-BERICHT
    if f"pred_{chosen_ticker}" in st.session_state:
        st.divider()
        st.markdown(st.session_state[f"pred_{chosen_ticker}"])

    # 7. BACKUP-BEREICH
    with st.expander("💾 Wissensspeicher dauerhaft sichern & wiederherstellen"):
        col_bk1, col_bk2 = st.columns(2)
        with col_bk1:
            mem_json_str = json.dumps(memory, indent=2, ensure_ascii=False)
            st.download_button(label="📥 Lernstand herunterladen (JSON)", data=mem_json_str, file_name="stock_memory.json", mime="application/json", width="stretch")
        with col_bk2:
            uploaded_mem = st.file_uploader("📤 Gedächtnis einspielen", type=["json"], key="restore_mem_uploader")
            if uploaded_mem is not None:
                try:
                    loaded_data = json.load(uploaded_mem)
                    save_memory(loaded_data)
                    st.success("✅ Wissensstand wiederhergestellt!")
                    time.sleep(0.5)
                except Exception as e:
                    st.error(f"Fehler: {e}")
