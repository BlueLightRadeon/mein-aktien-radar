import streamlit as st

def render(stock_df):
    st.info("ℹ️ **Kurzinfo:** Vollständige Gegenüberstellung: Aktuelle Empfehlungen & Begründungen vs. historische 3-Monats-Analyse.")
    if not stock_df.empty:
        disp_cols = [
            c for c in [
                "Unternehmen", 
                "Aktuelle KI-Empfehlung", 
                "Aktuelle Begründung", 
                "Empfehlung (vor 3 Monaten)", 
                "Begründung (vor 3 Monaten)", 
                "Börsenkurs", 
                "Fair Value", 
                "Analysten-Kursziel", 
                "Gewinn / Verlust"
            ] if c in stock_df.columns
        ]
        st.dataframe(stock_df[disp_cols], hide_index=True, width="stretch")
    
    depot_text = st.session_state.get("ai_depot", "")
    if depot_text and len(depot_text.strip()) > 15:
        st.divider()
        st.subheader("🤖 Detaillierter KI-Handelsbericht für deine Aktien:")
        st.markdown(depot_text)
    else:
        st.info("Klicke oben auf '🚀 Jetzt sofort manuell aktualisieren', um den Handelsbericht zu berechnen.")
