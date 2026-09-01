from datetime import datetime
from ai_service import get_account_models, run_analysis
from data_service import fetch_all_headlines, get_historical_chart_data, get_stock_data, load_saved_portfolio, resolve_to_ticker, save_portfolio_to_file
from groq import Groq
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="KI Markt- & Depot-Radar",
    page_icon="📈",
    layout="centered",
)

st.title("📈 KI Markt- & Depot-Radar")

GROQ_KEY = st.secrets.get("GROQ_API_KEY", "")
default_tickers = load_saved_portfolio()

# Seitenleiste
with st.sidebar:
  st.header("💼 Mein Depot")
  portfolio_input = st.text_input(
      "Deine Aktien & ETFs (Name oder Ticker)", value=default_tickers
  )

  if st.button("💾 Depot dauerhaft speichern", use_container_width=True):
    if save_portfolio_to_file(portfolio_input):
      st.success("✅ Gespeichert! Bleibt beim nächsten Öffnen erhalten.")

  st.caption("Beispiele: Apple, Allianz, Mercedes, MSCI World, NVDA")

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
    "📅 Earnings-Kalender",
    "🎯 Kauf- / Verkauf-Signale",
    "📊 Live-Charts & Performance",
])

raw_tickers = [t.strip() for t in portfolio_input.split(",") if t.strip()]

# TAB 5: Live-Charts & Performance
with tab5:
  st.subheader("📊 Kursentwicklung & Performance-Vergleich")

  c1, c2 = st.columns([2, 1])
  with c1:
    timeframe = st.selectbox(
        "Zeitraum:",
        options=["1d", "5d", "1mo", "6mo", "1y", "5y"],
        index=2,
        format_func=lambda x: {
            "1d": "1 Tag (Intraday Live)",
            "5d": "5 Tage",
            "1mo": "1 Monat",
            "6mo": "6 Monate",
            "1y": "1 Jahr",
            "5y": "5 Jahre",
        }[x],
    )
  with c2:
    chart_mode = st.radio(
        "Modus:",
        ["Performance in %", "Absolute Kurse"],
        horizontal=True,
    )

  if raw_tickers:
    resolved_list = [resolve_to_ticker(x) for x in raw_tickers]

    with st.spinner("Lade Chartdaten für alle Aktien..."):
      chart_df = get_historical_chart_data(resolved_list, period=timeframe)

    if not chart_df.empty and len(chart_df) > 1:
      fig = go.Figure()

      if chart_mode == "Performance in %":
        # Erste gültige Basis pro Aktie finden und in % umrechnen
        for col in chart_df.columns:
          first_val = chart_df[col].dropna().iloc[0]
          if first_val > 0:
            pct_series = ((chart_df[col] - first_val) / first_val) * 100
            fig.add_trace(
                go.Scatter(
                    x=chart_df.index,
                    y=pct_series,
                    mode="lines",
                    name=col,
                    hovertemplate=f"<b>{col}</b>: %{{y:+.2f}}%<extra></extra>",
                )
            )

        fig.update_layout(
            title=f"Performance-Vergleich seit Periodenbeginn ({timeframe.upper()})",
            xaxis_title="Datum / Zeit",
            yaxis_title="Entwicklung (%)",
            yaxis_ticksuffix="%",
            hovermode="x unified",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
            margin=dict(l=10, r=10, t=50, b=10),
        )
      else:
        # Absolute Kurse
        for col in chart_df.columns:
          fig.add_trace(
              go.Scatter(
                  x=chart_df.index,
                  y=chart_df[col],
                  mode="lines",
                  name=col,
                  hovertemplate=f"<b>{col}</b>: %{{y:.2f}}<extra></extra>",
              )
          )

        fig.update_layout(
            title=f"Absoluter Kursverlauf ({timeframe.upper()})",
            xaxis_title="Datum / Zeit",
            yaxis_title="Kurs",
            hovermode="x unified",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
            margin=dict(l=10, r=10, t=50, b=10),
        )

      st.plotly_chart(fig, use_container_width=True)
    else:
      st.info(
          "Aktuell werden Kursdaten geladen. Bitte prüfe deine Aktienkürzel"
          " oder versuche einen anderen Zeitraum."
      )
  else:
    st.warning("Bitte gib mindestens eine Aktie in der Seitenleiste ein.")

# BUTTON FÜR KI-ANALYSE
if st.button("🚀 KI-Analyse starten", use_container_width=True):
  if not GROQ_KEY:
    st.error(
        "⚠️ Kein GROQ_API_KEY in den Streamlit Secrets gefunden! Bitte unter"
        " Settings -> Secrets eintragen."
    )
  else:
    save_portfolio_to_file(portfolio_input)
    client = Groq(api_key=GROQ_KEY.strip())

    with st.spinner(
        f"Löse Ticker auf, lade Daten & analysiere mit Modell"
        f" '{selected_model}'..."
    ):
      news_data = fetch_all_headlines()
      news_text = "\n".join(news_data)

      stock_df, ticker_news, resolved_tickers = get_stock_data(raw_tickers)
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
                  "Nächste Earnings",
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
        st.error(
            f"Fehler mit Modell '{selected_model}': {str(e)}\n\n👉 Wähle in der"
            " Seitenleiste einfach ein anderes Modell aus dem Dropdown aus."
        )
