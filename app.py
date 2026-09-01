from datetime import datetime
import feedparser
from groq import Groq
import pandas as pd
import streamlit as st
import yfinance as yf

# Smartphone-optimierte Seitenkonfiguration
st.set_page_config(
    page_title="KI Markt- & Depot-Radar",
    page_icon="📈",
    layout="centered",
)

st.title("📈 KI Markt- & Depot-Radar")

# Key aus Streamlit Secrets laden
GROQ_KEY = st.secrets.get("GROQ_API_KEY", "")

with st.sidebar:
  st.header("💼 Mein Depot")
  portfolio_input = st.text_input(
      "Deine Aktien (Ticker)", value="AAPL, MSFT, NVDA, TSLA"
  )
  st.caption("Beispiele: AAPL, MSFT, SAP.DE, MBG.DE")

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
      for entry in feed.entries[:2]:
        if hasattr(entry, "title") and entry.title:
          title = entry.title.strip()
          if title and title not in headlines:
            headlines.append(f"- {title}")
    except Exception:
      continue
  # Strikt auf 25 Schlagzeilen begrenzen, um Groq-Limits sicher einzuhalten
  return headlines[:25]


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

if st.button("🚀 KI-Analyse starten", use_container_width=True):
  if not GROQ_KEY:
    st.error(
        "⚠️ Kein GROQ_API_KEY in den Streamlit Secrets gefunden! Bitte unter"
        " Settings -> Secrets eintragen."
    )
  else:
    client = Groq(api_key=GROQ_KEY.strip())
    tickers = [
        t.strip().upper() for t in portfolio_input.split(",") if t.strip()
    ]

    with st.spinner("Scanne 40+ Quellen und werte Daten per KI aus..."):
      news_data = fetch_all_headlines(RSS_SOURCES)
      news_text = "\n".join(news_data)
      stock_df = get_stock_data(tickers)

      prompt_market = f"""
Aktuelle globale Finanz- und Wirtschaftsnachrichten:
{news_text}

Fasse die **TOP 10 wichtigsten Markt-Informationen** prägnant auf Deutsch zusammen.
Bewerte am Ende kurz die Gesamt-Marktstimmung (Bullisch / Neutral / Bärisch).
"""

      prompt_depot = f"""
Depot-Aktien: {', '.join(tickers)}
Nachrichtenlage:
{news_text}

Erstelle für jede Aktie ({', '.join(tickers)}):
1. **Sentiment**: 🟢 Bullisch, 🟡 Neutral oder 🔴 Bärisch
2. **Fokus/News**: Aktuelle Trends oder Einflussfaktoren.
3. **Tipp für Anleger**: Worauf die nächsten Tage geachtet werden sollte.
"""

      # Wir fragen das offizielle, stabile Llama-Modell gezielt ab
      try:
        res_market = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt_market}],
            temperature=0.4,
            max_tokens=800,
        )
        out_market = res_market.choices[0].message.content

        res_depot = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt_depot}],
            temperature=0.4,
            max_tokens=800,
        )
        out_depot = res_depot.choices[0].message.content

        with tab1:
          st.markdown(out_market)

        with tab2:
          st.subheader("Aktuelle Kurse")
          st.dataframe(stock_df[["Ticker", "Kurs"]], hide_index=True)
          st.divider()
          st.markdown(out_depot)

        with tab3:
          st.subheader("📅 Anstehende Quartalszahlen")
          st.dataframe(stock_df, hide_index=True)

      except Exception as e:
        st.error(f"Fehler bei der Groq-Abfrage: {str(e)}")
