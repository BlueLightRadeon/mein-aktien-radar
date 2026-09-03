import streamlit as st
from groq import Groq
from ai_service import run_duel_analysis

def render(stock_df, selected_model, groq_key):
    st.info("ℹ️ **Kurzinfo:** Direktes 1-gegen-1-Duell zweier beliebiger Aktien aus deinem Depot.")
    if not stock_df.empty and len(stock_df) >= 2:
        cd1, cd2 = st.columns(2)
        names_list = stock_df["Unternehmen"].tolist()
        with cd1:
            duel_a = st.selectbox("Erste Aktie:", names_list, index=0, key="duel_sel_a")
        with cd2:
            duel_b = st.selectbox("Zweite Aktie:", names_list, index=1, key="duel_sel_b")
        
        if st.button("⚡ Duell auswerten", width="stretch", key="btn_duel_run", type="primary"):
            if groq_key:
                cl = Groq(api_key=groq_key)
                row_a = stock_df[stock_df["Unternehmen"] == duel_a].iloc[0].to_dict()
                row_b = stock_df[stock_df["Unternehmen"] == duel_b].iloc[0].to_dict()
                with st.spinner(f"Analysiere Duell zwischen {duel_a} und {duel_b}..."):
                    res_duel = run_duel_analysis(cl, selected_model, str(row_a), str(row_b))
                    st.session_state[f"duel_{duel_a}_{duel_b}"] = res_duel
            else:
                st.error("⚠️ Bitte GROQ_API_KEY hinterlegen.")

        duel_res = st.session_state.get(f"duel_{duel_a}_{duel_b}", "")
        if duel_res:
            st.divider()
            st.markdown(duel_res)
    else:
        st.info("Mindestens 2 Positionen im Depot nötig, um ein Duell zu starten.")
