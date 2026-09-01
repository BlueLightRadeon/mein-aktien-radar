from datetime import datetime
import feedparser
from openai import OpenAI
import pandas as pd
import streamlit as st
import yfinance as yf

# Smartphone-optimierte Seitenkonfiguration
st.set_page_config(
    page_title="KI Markt- & Depot-Radar (Free)",
    page_icon="📈",
    layout="centered",
)

st.title("📈 KI Markt- & Depot-Radar")

# Seitenleiste für Einstellungen
with st.sidebar:
  st.header("⚙️ Einstellungen")
  # Liest den Key aus Streamlit Secrets ODER per Eingabe
  groq_key = st.secrets.get("GROQ_API_KEY") or st.text_input(
      "Groq API Key (100% kostenlos)",
      type="password",
      help="Hol dir deinen Key auf console.groq.com",
  )
  portfolio_input = st.text_input(
      "Deine Aktien (Ticker mit Komma getrennt)", value="AAPL, MSFT, NVDA, TSLA"
  )
  st.caption("Beispiele: AAPL, MSFT, SAP.DE, MBG.DE")

# 40+ Quellen (Auszug starker Feeds)
RSS_SOURCES = [
    "https://feeds.a.dj.com/rss/RSSWorldNews.xml",
    "https://feeds.bbci.co.uk/news/business/rss.xml",
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "https://www.tagesschau.de/wirtschaft/index~rss2.xml",
    "https://www.spiegel.de/wirtschaft/index.rss",
    "https://www.handelsblatt.com/contentexport/feed/top-themen",
    "https://feeds.content.dowjones.io/public/rss/mw_topstories",
    "https://rss.politico.com/economy.xml",
    "https://www.investing.com/rss/news_25.rss",
    "https://seekingalpha.com/market_currents.xml",
    "https://www.fool.com/a/feeds/foolwatch",
    "https://www.benzinga.com/feeds/news/markets",
    "https://techcrunch.com/category/fintech/feed/",
    "https://cointelegraph.com/rss",
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
]


def fetch_all_headlines(sources):
  headlines = []
  for url in sources:
    try:
      feed = feedparser.parse(url)
      for entry in feed.entries[:3]:
        headlines.append(f"- {entry.title}")
    except Exception:
      continue
  return headlines[:80]


def get_stock_data(tickers):
  data = []
  for t in tickers:
    try:
      stock = yf.Ticker(t)
      price = stock.fast_info.last_price
      earnings_date = "Unbekannt"
      try:
        cal = stock.calendar
        if isinstance(cal, pd.DataFrame) and not cal.empty:
          earnings_date = str(cal.iloc[0, 0]).split(" ")[0]
        elif isinstance(cal, dict) and "Earnings Date" in cal:
          earnings_date = str(cal["Earnings Date"][0]).split(" ")[0]
      except Exception:
        pass

      data.append({
          "Ticker": t,
          "Kurs": f"{price:.2f}" if price is not None else "N/A",
          "Nächste Earnings": earnings_date,
      })
    except Exception:
      data.append(
          {"Ticker": t, "Kurs": "N/A", "Nächste Earnings": "Nicht gefunden"}
      )
  return pd.DataFrame(data)


tab1, tab2, tab3 = st.tabs(
    ["🌍 Markt Top 10", "💼 Sentiment & Depot", "📅 Earnings-Kalender"]
)

if st.button("🚀 Kostenlose KI-Analyse starten", use_container_width=True):
  if not groq_key:
    st.error(
        "⚠️ Bitte trage deinen kostenlosen Groq-Key (console.groq.com) in der"
        " Seitenleiste ein!"
    )
  else:
    # Verbindung zu Groq über OpenAI-kompatible Schnittstelle
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1", api_key=groq_key
    )

    tickers = [
        t.strip().upper() for t in portfolio_input.split(",") if t.strip()
    ]

    with st.spinner("Lese Nachrichten und analysiere Daten per KI..."):
      news_data = fetch_all_headlines(RSS_SOURCES)
      news_text = "\n".join(news_data)
      stock_df = get_stock_data(tickers)

      prompt_market = f"""
            Hier sind weltweite Wirtschaftsnachrichten:
            {news_text}
            
            Fasse die **TOP 10 wichtigsten Markt-Informationen** prägnant auf Deutsch zusammen.
            Bewerte am Ende kurz die weltweite Marktstimmung (Bullisch / Neutral / Bärisch).
            """

      prompt_depot = f"""
            Depot-Aktien: {', '.join(tickers)}
            Wirtschaftsnachrichten:
            {news_text}
            
            Erstelle für jede Aktie einzeln:
            1. **Sentiment**: 🟢 Bullisch, 🟡 Neutral oder 🔴 Bärisch
            2. **Fokus/News**: Relevante Trends oder Neuigkeiten dazu.
            3. **Tipp für Anleger**: Worauf die nächsten Tage geachtet werden sollte.
            """

      # Kostenloses High-End-Modell: Llama 3.3 70B
      res_market = client.chat.completions.create(
          model="llama-3.3-70b-versatile",
          messages=[{"role": "user", "content": prompt_market}],
      )

      res_depot = client.chat.completions.create(
          model="llama-3.3-70b-versatile",
          messages=[{"role": "user", "content": prompt_depot}],
      )

    with tab1:
      st.markdown(res_market.choices[0].message.content)

    with tab2:
      st.subheader("Aktuelle Kurse")
      st.dataframe(stock_df[["Ticker", "Kurs"]], hide_index=True)
      st.divider()
      st.markdown(res_depot.choices[0].message.content)

    with tab3:
      st.subheader("📅 Anstehende Quartalszahlen")
      st.dataframe(stock_df, hide_index=True)
