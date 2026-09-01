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

# In der Seitenleiste Aktien verwalten
with st.sidebar:
  st.header("💼 Mein Depot")
  portfolio_input = st.text_input(
      "Deine Aktien & ETFs (Ticker)", value="AAPL, MSFT, NVDA, TSLA"
  )
  st.caption(
      "Beispiele: AAPL, MSFT, SAP.DE, MBG.DE, CSPX.AS (S&P 500 ETF), EUNL.DE"
      " (MSCI World)"
  )

# 45+ globale Feeds für Makro, Aktien, ETFs & Branchen
RSS_SOURCES = [
    # Global Macro & Weltpolitik
    "https://feeds.a.dj.com/rss/RSSWorldNews.xml",
    "https://feeds.bbci.co.uk/news/business/rss.xml",
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "https://www.tagesschau.de/wirtschaft/index~rss2.xml",
    "https://www.spiegel.de/wirtschaft/index.rss",
    "https://www.handelsblatt.com/contentexport/feed/top-themen",
    "https://rss.politico.com/economy.xml",
    # Börsen, Märkte & Ad-hocs
    "https://feeds.content.dowjones.io/public/rss/mw_topstories",
    "https://www.investing.com/rss/news_25.rss",
    "https://seekingalpha.com/market_currents.xml",
    "https://www.fool.com/a/feeds/foolwatch",
    "https://www.benzinga.com/feeds/news/markets",
    "https://www.finanzen.net/rss/news",
    "https://www.deraktionaer.de/rss/feed",
    # ETFs & Fondswelt
    "https://etfdb.com/feed/",
    "https://www.etftrends.com/feed/",
    "https://www.justetf.com/de/news/feed.rss",
    # Tech, Rohstoffe & Krypto
    "https://techcrunch.com/category/fintech/feed/",
    "https://cointelegraph.com/rss",
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://oilprice.com/rss/main",
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
  return headlines[:35]


def calculate_rsi(series, period=14):
  """Berechnet den 14-Tage RSI (Relative Strength Index)."""
  try:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    val = rsi.iloc[-1]
    return f"{val:.1f}" if pd.notnull(val) else "N/A"
  except Exception:
    return "N/A"


def get_stock_data(tickers):
  """Holt Kurse, Fundamentaldaten, RSI, Kursziele und Ticker-News."""
  data = []
  direct_news = []

  for t in tickers:
    try:
      stock = yf.Ticker(t)
      info = stock.info if hasattr(stock, "info") else {}
      fast = stock.fast_info

      price = fast.last_price if hasattr(fast, "last_price") else None
      currency = fast.currency if hasattr(fast, "currency") else "USD"

      # RSI (14 Tage) über historische Daten
      hist = stock.history(period="1mo")
      rsi_val = (
          calculate_rsi(hist["Close"])
          if not hist.empty and len(hist) > 14
          else "N/A"
      )

      # Fundamentaldaten & Analysten-Ziele
      pe_ratio = info.get("trailingPE")
      pe_str = f"{pe_ratio:.1f}" if pe_ratio else "N/A"

      target_price = info.get("targetMeanPrice")
      if target_price and price:
        upside = ((target_price - price) / price) * 100
        target_str = f"{target_price:.2f} {currency} ({upside:+.1f}%)"
      else:
        target_str = "N/A"

      # Nächste Quartalszahlen
      earnings_date = "Unbekannt"
      try:
        cal = stock.calendar
        if isinstance(cal, pd.DataFrame) and not cal.empty:
          earnings_date = str(cal.iloc[0, 0]).split(" ")[0]
        elif isinstance(cal, dict) and "Earnings Date" in cal:
          earnings_date = str(cal["Earnings Date"][0]).split(" ")[0]
      except Exception:
        pass

      # Direkte Ticker-News
      try:
        if stock.news:
          for item in stock.news[:2]:
            if "title" in item:
              direct_news.append(f"[{t}] {item['title']}")
      except Exception:
        pass

      data.append({
          "Ticker": t,
          "Kurs": (
              f"{price:.2f} {currency}" if price is not None else "N/A"
          ),
          "RSI (14D)": rsi_val,
          "KGV (P/E)": pe_str,
          "Analysten-Kursziel": target_str,
          "Nächste Earnings": earnings_date,
      })
    except Exception:
      data.append({
          "Ticker": t,
          "Kurs": "N/A",
          "RSI (14D)": "N/A",
          "KGV (P/E)": "N/A",
          "Analysten-Kursziel": "N/A",
          "Nächste Earnings": "Nicht gefunden",
      })

  return pd.DataFrame(data), direct_news


def pick_valid_chat_model(client):
  """Liest die für deinen API-Key freigeschalteten Chat-Modelle aus."""
  try:
    available = [m.id for m in client.models.list().data]
    valid_text_models = [
        m
        for m in available
        if not any(x in m.lower() for x in ["whisper", "guard", "vision"])
    ]
    for pref in [
        "llama-3.3-70b-versatile",
        "llama3-70b-8192",
        "llama3-8b-8192",
        "mixtral-8x7b-32768",
    ]:
      if pref in valid_text_models:
        return pref
    if valid_text_models:
      return valid_text_models[0]
  except Exception:
    pass
  return "llama-3.3-70b-versatile"


# Tabs beibehalten
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

    with st.spinner("Scanne 45+ Quellen, Ticker-News, RSI & Fundamentaldaten..."):
      model_to_use = pick_valid_chat_model(client)

      # 1. Daten holen
      news_data = fetch_all_headlines(RSS_SOURCES)
      news_text = "\n".join(news_data)
      stock_df, ticker_news = get_stock_data(tickers)
      ticker_news_text = (
          "\n".join(ticker_news)
          if ticker_news
          else "Keine direkten Ad-hocs gefunden."
      )

      # Datenübersicht für die KI aufbereiten
      metrics_summary = stock_df[
          ["Ticker", "Kurs", "RSI (14D)", "KGV (P/E)", "Analysten-Kursziel"]
      ].to_string(index=False)

      # 2. Prompts
      prompt_market = f"""
Aktuelle weltweite Wirtschaftsnachrichten (aus 45+ Quellen & ETF-Feeds):
{news_text}

Fasse die **TOP 10 wichtigsten Markt-Informationen** prägnant auf Deutsch zusammen.
Bewerte am Ende kurz die Marktstimmung (Bullisch / Neutral / Bärisch).
"""

      prompt_depot = f"""
Depot-Werte & Kennzahlen:
{metrics_summary}

Spezifische News zu deinen Ticker-Symbolen:
{ticker_news_text}

Allgemeine Makro-Nachrichten:
{news_text}

Erstelle für jeden Wert ({', '.join(tickers)}) eine fundierte Analyse:
1. **Sentiment**: 🟢 Bullisch, 🟡 Neutral oder 🔴 Bärisch
2. **Technik & Bewertung**: Interpretiere kurz den RSI (unter 30 = überverkauft, über 70 = überkauft), das KGV und das Analysten-Kursziel.
3. **Fokus/News**: Relevante Meldungen oder Einflussfaktoren.
4. **Tipp für Anleger**: Konkrete Beobachtungspunkte für die kommenden Tage.
"""

      try:
        res_market = client.chat.completions.create(
            model=model_to_use,
            messages=[{"role": "user", "content": prompt_market}],
            temperature=0.3,
            max_tokens=900,
        )
        out_market = res_market.choices[0].message.content

        res_depot = client.chat.completions.create(
            model=model_to_use,
            messages=[{"role": "user", "content": prompt_depot}],
            temperature=0.3,
            max_tokens=1000,
        )
        out_depot = res_depot.choices[0].message.content

        with tab1:
          st.caption(f"🤖 Verwendetes Modell: `{model_to_use}`")
          st.markdown(out_market)

        with tab2:
          st.subheader("Depot-Kennzahlen & Kurse")
          st.dataframe(
              stock_df[[
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
              stock_df[["Ticker", "Kurs", "Nächste Earnings"]], hide_index=True
          )

      except Exception as e:
        st.error(f"Fehler bei Modell '{model_to_use}': {str(e)}")
