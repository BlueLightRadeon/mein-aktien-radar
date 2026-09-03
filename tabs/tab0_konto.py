import streamlit as st

def render(stock_df, display_cash, stock_val, fmt_eur):
    st.info("ℹ️ **Kurzinfo:** Zeigt dein reales Trade Republic Depot – getrennt nach Bargeld (Cash aus Auszug) und dem aktuellen Wert deiner Wertpapiere.")
    
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
        st.info("📂 Lade deinen Trade Republic Kontoauszug (PDF) in der linken Seitenleiste hoch, um deine Werte hier zu sehen.")
