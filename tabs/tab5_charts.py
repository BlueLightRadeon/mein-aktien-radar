import streamlit as st
import plotly.graph_objects as go
from data_service import get_display_name, get_individual_series_dict

def render(portfolio_list):
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
