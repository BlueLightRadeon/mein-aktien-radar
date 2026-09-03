import streamlit as st
from groq import Groq
from ai_service import run_depot_analysis

def render(stock_df, selected_model, groq_key, get_berlin_time_str):
    st.info("ℹ️ **Kurzinfo:** Vollständige Gegenüberstellung: Aktuelle Empfehlungen & Begründungen vs. historische 3-Monats-Analyse.")
    
    # Button ganz oben im Blickfeld
    col_btn, col_time = st.columns([2, 2])
    with col_btn:
        run_btn = st.button("🤖 Depot-Bewertung & Empfehlungen jetzt analysieren", type="primary", key="btn_eval_depot", width="stretch")
    with col_time:
        depot_time = st.session_state.get("last_depot_time", "")
        if depot_time:
            st.caption(f"🕒 Stand: **{depot_time}**")

    if run_btn:
        if not groq_key:
            st.error("⚠️ Bitte hinterlege deinen GROQ_API_KEY in den Secrets.")
        elif stock_df.empty:
            st.warning("⚠️ Lade zuerst ein Depot hoch.")
        else:
            summary_cols = [
                c for c in [
                    "Unternehmen", "Aktuelle KI-Empfehlung", "Aktuelle Begründung", 
                    "Empfehlung (vor 3 Monaten)", "Begründung (vor 3 Monaten)", 
                    "Börsenkurs", "Aktueller Wert (TR)", "KGV (P/E)", "Fair Value", "Analysten-Kursziel"
                ] if c in stock_df.columns
            ]
            metrics_summary = stock_df[summary_cols].to_string(index=False)

            with st.spinner("Analysiere deine Depot-Werte mit Groq KI..."):
                try:
                    client = Groq(api_key=groq_key)
                    report = run_depot_analysis(client, selected_model, metrics_summary)
                    st.session_state["ai_depot"] = report
                    st.session_state["last_depot_time"] = get_berlin_time_str()
                    st.success("✅ Depot-Bewertung erfolgreich aktualisiert!")
                except Exception as e:
                    st.error(f"Fehler bei der Analyse: {e}")

    depot_text = st.session_state.get("ai_depot", "")
    if depot_text and len(depot_text.strip()) > 15:
        st.divider()
        st.subheader("🤖 Detaillierter KI-Handelsbericht für deine Aktien:")
        st.markdown(depot_text)

    st.divider()
    st.subheader("Übersicht aller Positionen:")
    if not stock_df.empty:
        disp_cols = [
            c for c in [
                "Unternehmen", "Aktuelle KI-Empfehlung", "Aktuelle Begründung", 
                "Empfehlung (vor 3 Monaten)", "Begründung (vor 3 Monaten)", 
                "Börsenkurs", "Fair Value", "Analysten-Kursziel", "Gewinn / Verlust"
            ] if c in stock_df.columns
        ]
        st.dataframe(stock_df[disp_cols], hide_index=True, width="stretch")
    else:
        st.info("📂 Lade zuerst dein Depot hoch, um die Positionen hier zu sehen.")
