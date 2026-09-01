from datetime import datetime
import os
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

# --- FUNKTIONEN ZUM DAUERHAFTEN SPEICHERN ---
PORTFOLIO_FILE = "portfolio.txt"


def load_saved_portfolio():
  # Prüfe zuerst URL-Parameter, dann Textdatei, sonst Standard-Werte
  url_tickers = st.query_params.get("tickers", None)
  if url_tickers:
    return url_tickers
  if os.path.exists(PORTFOLIO_FILE):
    try:
      with open(PORTFOLIO_FILE, "r") as f:
        saved = f.read().strip()
        if saved:
          return saved
    except Exception:
      pass
  return "AAPL, MSFT, NVDA, TSLA"


def save_portfolio_to_file(tickers_str):
  try:
    with open(PORTFOLIO_FILE, "w") as f:
      f.write(tickers_str)
    # URL synchronisieren, damit der Home-Bildschirm-Link die Daten behält
    st.query_params["tickers"] = tickers_str
    return True
  except Exception:
    return False


# Standardmäßig deine gespeicherten Werte laden
default_tickers = load_saved_portfolio()


@st.cache_data(ttl=3600)
def get_account_models(api_key):
  """Liest live alle für deinen Key freigeschalteten Modelle aus."""
  try:
    c = Groq(api_key=api_key.strip())
    models_list = [m.id for m in c.models.list().data]
    text_models = [
        m
        for m in models_list
        if not any(
            x in m.lower()
            for x in ["whisper", "guard", "orpheus", "vision", "safeguard"]
        )
    ]
    return text_models if text_models else models_list
  except Exception:
    return ["openai/gpt-oss-120b", "llama-3.3-70b-versatile"]


# In der Seitenleiste Aktien & Modellauswahl
with st.sidebar:
  st.header("💼 Mein Depot")

  portfolio_input = st.text_input(
      "Deine Aktien & ETFs (Ticker)", value=default_tickers
  )

  # Speicher-Button für dein Depot
  if st.button("💾 Depot dauerhaft speichern"):
    if save_portfolio_to_file(portfolio_input):
      st.success("✅ Gespeichert! Bleibt beim nächsten Öffnen erhalten.")

  st.caption("Beispiele: AAPL, MSFT, SAP.DE, MBG.DE, CSPX.AS, EUNL.DE")

  st.divider()
  st.header("🤖 KI-Modell")
  if GROQ_KEY:
    available_models = get_account_models(GROQ_KEY)
    selected_model = st.selectbox(
        "Aktives Groq-Modell", available_models, index=0
    )
  else:
    selected_model = "llama-3.3-70b-versatile"

# 45+ globale Feeds
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
  return headlines[:30]


def calculate_rsi(series, period=14):
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
  data = []
  direct_news = []

  for t in tickers:
    try:
      stock = yf.Ticker(t)
      info = stock.info if hasattr(stock, "info") else {}
      fast = stock.fast_info

      price = fast.last_price if hasattr(fast, "last_price") else None
      currency = fast.currency if hasattr(fast, "currency") else "USD"

      hist = stock.history(period="1mo")
      rsi_val = (
          calculate_rsi(hist["Close"])
          if not hist.empty and len(hist) > 14
          else "N/A"
      )

      pe_ratio = info.get("trailingPE")
      pe_str = f"{pe_ratio:.1f}" if pe_ratio else "N/A"

      target_price = info.get("targetMeanPrice")
      if target_price and price:
        upside = ((target_price - price) / price) * 100
        target_str = f"{target_price:.2f} {currency} ({upside:+.1f}%)"
      else:
        target_str = "N/A"

      earnings_date = "Unbekannt"
      try:
        cal = stock.calendar
        if isinstance(cal, pd.DataFrame) and not cal.empty:
          earnings_date = str(cal.iloc[0, 0]).split(" ")[0]
        elif isinstance(cal, dict) and "Earnings Date" in cal:
          earnings_date = str(cal["Earnings Date"][0]).split(" ")[0]
      except Exception:
        pass

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
    # Automatisch im Hintergrund auch beim Klick auf Analyse speichern
    save_portfolio_to_file(portfolio_input)

    client = Groq(api_key=GROQ_KEY.strip())
    tickers = [
        t.strip().upper() for t in portfolio_input.split(",") if t.strip()
    ]

    with st.spinner(
        f"Lade Daten & analysiere mit Modell '{selected_model}'..."
    ):
      news_data = fetch_all_headlines(RSS_SOURCES)
      news_text = "\n".join(news_data)
      stock_df, ticker_news = get_stock_data(tickers)
      ticker_news_text = (
          "\n".join(ticker_news)
          if ticker_news
          else "Keine direkten Ad-hocs gefunden."
      )

      metrics_summary = stock_df[
          ["Ticker", "Kurs", "RSI (14D)", "KGV (P/E)", "Analysten-Kursziel"]
      ].to_string(index=False)

      prompt_market = f"""
Aktuelle weltweite Wirtschaftsnachrichten (45+ Quellen & ETF-Feeds):
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
            model=selected_model,
            messages=[{"role": "user", "content": prompt_market}],
            temperature=0.3,
            max_tokens=900,
        )
        out_market = res_market.choices[0].message.content

        res_depot = client.chat.completions.create(
            model=selected_model,
            messages=[{"role": "user", "content": prompt_depot}],
            temperature=0.3,
            max_tokens=1000,
        )
        out_depot = res_depot.choices[0].message.content

        with tab1:
          st.caption(f"🤖 Verwendetes Modell: `{selected_model}`")
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
        st.error(
            f"Fehler mit Modell '{selected_model}': {str(e)}\n\n👉 Wähle in der"
            " Seitenleiste einfach ein anderes Modell aus dem Dropdown aus."
        )
