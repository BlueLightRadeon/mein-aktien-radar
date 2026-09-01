from datetime import datetime
from ai_service import get_account_models, run_analysis
from data_service import clean_ticker, fetch_all_headlines, get_individual_series_dict, get_stock_data, load_saved_portfolio, save_portfolio_to_file, search_ticker_candidates
from groq import Groq
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="KI Markt- & Depot-Radar",
    page_icon="📈",
    layout="centered",
)

st.title("📈 KI Markt- & Depot-Radar")

GROQ_KEY = st.secrets.get("GROQ_API_KEY", "")

if "my_portfolio" not in st.session_state:
  st.session_state.my_portfolio = load_saved_portfolio()

# Seitenleiste
with st.sidebar:
  st.header("💼 Mein Depot")

  search_query = st.text_input(
      "🔍 Aktie/ETF suchen:", placeholder="z. B. Novo Nordisk, Broadcom..."
  )

  if search_query:
    results = search_ticker_candidates(search_query)
    if results:
      selected_candidate = st.selectbox(
          "Gefundene Treffer:", results, key="search_select"
      )
      if st.button("➕ Zum Depot hinzufügen", use_container_width=True):
        if selected_candidate not in st.session_state.my_portfolio:
          st.session_state.my_portfolio.append(selected_candidate)
          save_portfolio_to_file(st.session_state.my_portfolio)
          st.success("Hinzugefügt!")
          st.rerun()
    else:
      st.caption("Keine Treffer gefunden.")

  st.divider()

  st.subheader("Aktuell im Depot:")
  if st.session_state.my_portfolio:
    for idx, item in enumerate(list(st.session_state.my_portfolio)):
      col_a, col_b = st.columns([4, 1])
      with col_a:
        st.write(f"• **{item}**")
      with col_b:
        if st.button("❌", key=f"del_{idx}", help=f"{item} entfernen"):
          st.session_state.my_portfolio.remove(item)
          save_portfolio_to_file(st.session_state.my_portfolio)
          st.rerun()
  else:
    st.info("Noch keine Aktien im Depot.")

  st.divider()
  st.header("🤖 KI-Modell")
  if GROQ_KEY:
    available_models = get_account_models(GROQ_KEY)
    selected_model = st.selectbox(
        "Aktives Groq-Modell", available_models, index=0
    )
  else:
    selected_model = "llama-3.3-70b-versatile"

# 5 Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🌍 Markt Top 10",
    "💼 Sentiment & Depot",
    "📅 Quartalszahlen",
    "🎯 Kauf- / Verkauf-Signale",
    "📊 Live-Charts & Performance",
])

# TAB 5: Live-Charts
with tab5:
  st.subheader("📊 Kursentwicklung & Performance")

  c1, c2 = st.columns([1, 1])
  with c1:
    timeframe = st.selectbox(
        "Zeitraum:",
        options=["1d", "5d", "1mo", "6mo", "1y", "5y"],
        index=2,
        format_func=lambda x: {
            "1d": "1 Tag (Live Intraday)",
            "5d": "5 Tage",
            "1mo": "1 Monat",
            "6mo": "6 Monate",
            "1y": "1 Jahr",
            "5y": "5 Jahre",
        }[x],
    )
  with c2:
    chart_mode = st.radio(
        "Darstellung:",
        ["Performance in %", "Preis pro Aktie (in Geld)"],
        horizontal=True,
    )

  if st.session_state.my_portfolio:
    with st.spinner("Lade Chartdaten für alle ausgewählten Werte..."):
      series_dict = get_individual_series_dict(
          st.session_state.my_portfolio, period=timeframe
      )

    if series_dict:
      available_tickers = list(series_dict.keys())
      view_options = ["Alle Aktien gleichzeitig"] + available_tickers
      selected_view = st.selectbox("Fokus-Auswahl:", view_options, index=0)

      metric_cols = st.columns(min(len(available_tickers), 4))
      for idx, t in enumerate(available_tickers):
        s = series_dict[t]
        if len(s) >= 2:
          start_p = s.iloc[0]
          end_p = s.iloc[-1]
          diff_pct = ((end_p - start_p) / start_p) * 100
          with metric_cols[idx % len(metric_cols)]:
            st.metric(
                label=t,
                value=f"{end_p:.2f}",
                delta=f"{diff_pct:+.2f}% ({timeframe.upper()})",
            )

      palette = [
          "#00D084",
          "#0693E3",
          "#FCB900",
          "#EB144C",
          "#9B51E0",
          "#00ACC1",
          "#FF6900",
      ]
      fig = go.Figure()
      tickers_to_plot = (
          available_tickers
          if selected_view == "Alle Aktien gleichzeitig"
          else [selected_view]
      )

      if chart_mode == "Performance in %":
        for i, t in enumerate(tickers_to_plot):
          s = series_dict[t]
          if not s.empty and s.iloc[0] > 0:
            base_val = s.iloc[0]
            pct_series = ((s - base_val) / base_val) * 100
            color = palette[i % len(palette)]
            fig.add_trace(
                go.Scatter(
                    x=s.index,
                    y=pct_series,
                    mode="lines",
                    name=t,
                    line=dict(width=2.5, color=color),
                    hovertemplate=f"<b>{t}</b>: %{{y:+.2f}}%<extra></extra>",
                )
            )

        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="rgba(150,150,150,0.6)",
            annotation_text="Start (0%)",
            annotation_position="bottom right",
        )

        fig.update_layout(
            title=f"Wertentwicklung in Prozent seit Start ({timeframe.upper()})",
            xaxis=dict(
                title="Datum / Uhrzeit",
                showgrid=True,
                gridcolor="rgba(200,200,200,0.15)",
            ),
            yaxis=dict(
                title="Gewinn / Verlust (%)",
                ticksuffix="%",
                showgrid=True,
                gridcolor="rgba(200,200,200,0.15)",
                zeroline=False,
            ),
            hovermode="x unified",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0
            ),
            margin=dict(l=10, r=10, t=60, b=10),
            height=430,
        )
      else:
        for i, t in enumerate(tickers_to_plot):
          s = series_dict[t]
          color = palette[i % len(palette)]
          fig.add_trace(
              go.Scatter(
                  x=s.index,
                  y=s,
                  mode="lines",
                  name=t,
                  line=dict(width=2.5, color=color),
                  hovertemplate=(
                      f"<b>{t}</b>: %{{y:.2f}} (EUR/USD)<extra></extra>"
                  ),
              )
          )

        fig.update_layout(
            title=f"Preis pro Einzelaktie in Geld ({timeframe.upper()})",
            xaxis=dict(
                title="Datum / Uhrzeit",
                showgrid=True,
                gridcolor="rgba(200,200,200,0.15)",
            ),
            yaxis=dict(
                title="Preis pro Aktie (in EUR / USD)",
                showgrid=True,
                gridcolor="rgba(200,200,200,0.15)",
            ),
            hovermode="x unified",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0
            ),
            margin=dict(l=10, r=10, t=60, b=10),
            height=430,
        )

      st.plotly_chart(
          fig,
          use_container_width=True,
          config={"displayModeBar": False, "responsive": True},
      )
    else:
      st.info("Synchronisiere Kursdaten...")
  else:
    st.warning("Dein Depot ist leer. Füge Aktien über die Seitenleiste hinzu!")

# BUTTON FÜR KI-ANALYSE
if st.button("🚀 KI-Analyse starten", use_container_width=True):
  if not GROQ_KEY:
    st.error(
        "⚠️ Kein GROQ_API_KEY in den Streamlit Secrets gefunden! Bitte unter"
        " Settings -> Secrets eintragen."
    )
  elif not st.session_state.my_portfolio:
    st.error("⚠️ Bitte füge zuerst mindestens eine Aktie zu deinem Depot hinzu!")
  else:
    save_portfolio_to_file(st.session_state.my_portfolio)
    client = Groq(api_key=GROQ_KEY.strip())

    with st.spinner(
        f"Lade Live-Daten für alle {len(st.session_state.my_portfolio)} Aktien &"
        f" erstelle Analyse..."
    ):
      news_data = fetch_all_headlines()
      news_text = "\n".join(news_data)

      stock_df, ticker_news, resolved_tickers = get_stock_data(
          st.session_state.my_portfolio
      )
      ticker_news_text = (
          "\n".join(ticker_news)
          if ticker_news
          else "Keine direkten Ad-hocs gefunden."
      )

      metrics_summary = stock_df[[
          "Name / Aktie",
          "Ticker",
          "Kurs",
          "RSI (14D)",
          "KGV (P/E)",
          "Analysten-Kursziel",
          "Konsens-Rating",
      ]].to_string(index=False)

      try:
        out_market, out_depot, out_signals = run_analysis(
            client,
            selected_model,
            news_text,
            metrics_summary,
            ticker_news_text,
        )

        with tab1:
          st.caption(f"🤖 Verwendetes Modell: `{selected_model}`")
          st.markdown(out_market)

        with tab2:
          st.subheader("Depot-Kennzahlen & Kurse")
          st.dataframe(
              stock_df[[
                  "Name / Aktie",
                  "Ticker",
                  "Kurs",
                  "RSI (14D)",
                  "KGV (P/E)",
                  "Analysten-Kursziel",
              ]],
              hide_index=True,
          )
          st.divider()
          st.markdown(out_depot)

        with tab3:
          st.subheader("📅 Anstehende Quartalszahlen")
          st.dataframe(
              stock_df[[
                  "Name / Aktie",
                  "Ticker",
                  "Kurs",
                  "Nächste Quartalszahlen",
              ]],
              hide_index=True,
          )

        with tab4:
          st.subheader("🎯 Handlungsempfehlungen & Signale")
          st.dataframe(
              stock_df[[
                  "Name / Aktie",
                  "Ticker",
                  "Kurs",
                  "Konsens-Rating",
                  "Analysten-Kursziel",
              ]],
              hide_index=True,
          )
          st.divider()
          st.markdown(out_signals)

      except Exception as e:
        st.error(f"Fehler bei der Analyse: {str(e)}")
