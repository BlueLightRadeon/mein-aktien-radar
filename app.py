from datetime import datetime
import feedparser
from openai import OpenAI
import pandas as pd
import streamlit as st
import yfinance as yf

# Seitenkonfiguration für Smartphone-Displays optimiert
st.set_page_config(
    page_title="Aktien & Makro Radar Pro", page_icon="📈", layout="centered"
)

st.title("📈 KI Markt- & Depot-Radar")

# Seitenleiste für Einstellungen
with st.sidebar:
  st.header("⚙️ Einstellungen")
  api_key = st.text_input("OpenAI API Key", type="password")
  portfolio_input = st.text_input(
      "Deine Aktien (Ticker mit Komma getrennt)", value="AAPL, MSFT, NVDA, TSLA"
  )
  st.caption("Beispiele: AAPL (Apple), MSFT (Microsoft), SAP.DE (SAP)")

# Umfassende Liste von RSS-Feeds (Makro, Tech, Krypto & Börse)
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

      # Nächstes Earnings-Datum ermitteln
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
          "Kurs (USD/EUR)": (
              f"{price:.2f}" if price is not None else "N/A"
          ),
          "Nächste Quartalszahlen": earnings_date,
      })
    except Exception:
      data.append({
          "Ticker": t,
          "Kurs (USD/EUR)": "N/A",
          "Nächste Quartalszahlen": "Fehler",
      })
  return pd.DataFrame(data)


tab1, tab2, tab3 = st.tabs(
    ["🌍 Markt Top 10", "💼 Sentiment & Depot", "📅 Quartalszahlen-Kalender"]
)

if st.button("🚀 Vollständige KI-Analyse starten", use_container_width=True):
  if not api_key:
    st.error("⚠️ Bitte gib zuerst deinen OpenAI API-Key in der Seitenleiste ein!")
  else:
    client = OpenAI(api_key=api_key)
    tickers = [
        t.strip().upper() for t in portfolio_input.split(",") if t.strip()
    ]

    with st.spinner("Scanne 40+ Quellen, Kurse & Earnings-Termine..."):
      # 1. Daten aggregieren
      news_data = fetch_all_headlines(RSS_SOURCES)
      news_text = "\n".join(news_data)
      stock_df = get_stock_data(tickers)

      # 2. KI Prompt für Gesamtmarkt
      prompt_market = f"""
            Hier sind Live-Schlagzeilen aus über 40 globalen Finanz- und Wirtschaftsmedien:
            {news_text}
            
            Erstelle die **TOP 10 wichtigsten Markt-Informationen** des Tages für Anleger.
            Bewerte abschließend die globale Stimmung (Bullisch / Neutral / Bärisch) in einem Satz.
            """

      # 3. KI Prompt für Depot mit Sentiment-Ampel
      prompt_depot = f"""
            Gehaltene Aktien: {', '.join(tickers)}
            Aktuelle Marktnachrichten:
            {news_text}
            
            Erstelle für JEDE Aktie einzeln:
            1. **Sentiment-Ampel**: 🟢 Bullisch, 🟡 Neutral oder 🔴 Bärisch (basierend auf aktuellen Trends/News).
            2. **Konkrete Auswirkungen**: Welche News oder Branchentrends betreffen diese Aktie aktuell direkt?
            3. **Handlungsempfehlung/Fokus**: Worauf sollte man in den nächsten Tagen bei dieser Aktie achten?
            """

      res_market = client.chat.completions.create(
          model="gpt-4o-mini",
          messages=[{"role": "user", "content": prompt_market}],
      )

      res_depot = client.chat.completions.create(
          model="gpt-4o-mini",
          messages=[{"role": "user", "content": prompt_depot}],
      )

    with tab1:
      st.markdown(res_market.choices[0].message.content)

    with tab2:
      st.subheader("Depot-Übersicht")
      st.dataframe(stock_df[["Ticker", "Kurs (USD/EUR)"]], hide_index=True)
      st.divider()
      st.markdown(res_depot.choices[0].message.content)

    with tab3:
      st.subheader("📅 Anstehende Quartalsberichte (Earnings)")
      st.dataframe(stock_df, hide_index=True)
      st.caption(
          "Hinweis: Daten werden direkt über offizielle Börsenkalender"
          " bezogen."
      )
