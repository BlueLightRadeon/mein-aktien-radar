from datetime import datetime
import os
import feedparser
from groq import Groq
import pandas as pd
import requests
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

# --- PERSISTENTES SPEICHERN ---
PORTFOLIO_FILE = "portfolio.txt"


def load_saved_portfolio():
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
  return "Apple, Microsoft, Nvidia, Tesla"


def save_portfolio_to_file(tickers_str):
  try:
    with open(PORTFOLIO_FILE, "w") as f:
      f.write(tickers_str)
    st.query_params["tickers"] = tickers_str
    return True
  except Exception:
    return False


default_tickers = load_saved_portfolio()


@st.cache_data(ttl=3600)
def get_account_models(api_key):
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
      "Deine Aktien & ETFs (Name oder Ticker)", value=default_tickers
  )

  if st.button("💾 Depot dauerhaft speichern"):
    if save_portfolio_to_file(portfolio_input):
      st.success("✅ Gespeichert! Bleibt beim nächsten Öffnen erhalten.")

  st.caption(
      "Du kannst Firmennamen oder Ticker eingeben (z. B. Apple, Allianz,"
      " Mercedes, MSCI World, NVDA)"
  )

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


def resolve_to_ticker(query):
  """Wandelt Firmennamen/Umgangssprachliche Namen in Börsenkürzel um."""
  q = query.strip()
  alias_map = {
      "MERCEDES": "MBG.DE",
      "MERCEDES-BENZ": "MBG.DE",
      "MERCEDES BENZ": "MBG.DE",
      "BMW": "BMW.DE",
      "VOLKSWAGEN": "VOW3.DE",
      "VW": "VOW3.DE",
      "ALLIANZ": "ALV.DE",
      "SIEMENS": "SIE.DE",
      "TELEKOM": "DTE.DE",
      "DEUTSCHE TELEKOM": "DTE.DE",
      "SAP": "SAP.DE",
      "BASF": "BAS.DE",
      "BAYER": "BAYN.DE",
      "DEUTSCHE BANK": "DBK.DE",
      "COMMERZBANK": "CBK.DE",
      "LUFTHANSA": "LHA.DE",
      "RHEINMETALL": "RHM.DE",
      "AIRBUS": "AIR.DE",
      "MSCI WORLD": "EUNL.DE",
      "S&P 500": "VUAA.DE",
      "SP500": "VUAA.DE",
      "NASDAQ": "EQAC.DE",
      "APPLE": "AAPL",
      "MICROSOFT": "MSFT",
      "NVIDIA": "NVDA",
      "TESLA": "TSLA",
      "AMAZON": "AMZN",
      "ALPHABET": "GOOGL",
      "GOOGLE": "GOOGL",
      "META": "META",
      "NETFLIX": "NFLX",
  }

  upper_q = q.upper()
  if upper_q in alias_map:
    return alias_map[upper_q]

  if "." in q or len(q) <= 5 and q.isalpha() and q.isupper():
    try:
      test_stock = yf.Ticker(q)
      if test_stock.fast_info.last_price is not None:
        return q
    except Exception:
      pass

  try:
    headers = {"User-Agent": "Mozilla/5.0"}
    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={requests.utils.quote(q)}&quotesCount=3&newsCount=0"
    resp = requests.get(url, headers=headers, timeout=4)
    if resp.status_code == 200:
      data = resp.json()
      quotes = data.get("quotes", [])
      if quotes:
        for item in quotes:
          sym = item.get("symbol", "")
          if sym.endswith(".DE") or sym.endswith(".F"):
            return sym
        return quotes[0].get("symbol", q)
  except Exception:
    pass

  return q


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


def get_stock_data(user_inputs):
  data = []
  direct_news = []
  resolved_tickers_list = []

  for raw_input in user_inputs:
    t = resolve_to_ticker(raw_input)
    resolved_tickers_list.append(t)

    try:
      stock = yf.Ticker(t)
      fast = stock.fast_info

      price = fast.last_price
      currency = fast.currency if hasattr(fast, "currency") else "USD"

      hist = stock.history(period="1mo")
      if (price is None or pd.isna(price)) and not hist.empty:
        price = hist["Close"].iloc[-1]

      rsi_val = (
          calculate_rsi(hist["Close"])
          if not hist.empty and len(hist) > 14
          else "N/A"
      )

      info = {}
      try:
        info = stock.info or {}
      except Exception:
        pass

      company_name = (
          info.get("shortName")
          or info.get("longName")
          or raw_input.capitalize()
      )

      pe_ratio = info.get("trailingPE") or info.get("forwardPE")
      pe_str = f"{pe_ratio:.1f}" if pe_ratio else "N/A"

      target_price = info.get("targetMeanPrice")
      if target_price and price:
        upside = ((target_price - price) / price) * 100
        target_str = f"{target_price:.2f} {currency} ({upside:+.1f}%)"
      else:
        target_str = "N/A"

      # Offizielles Analysten-Rating abrufen (z. B. Buy, Hold, Sell)
      recommendation = (
          info.get("recommendationKey", "N/A").replace("_", " ").upper()
      )

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
              direct_news.append(f"[{company_name} / {t}] {item['title']}")
      except Exception:
        pass

      data.append({
          "Name / Aktie": company_name,
          "Ticker": t,
          "Kurs": (
              f"{price:.2f} {currency}"
              if (price is not None and not pd.isna(price))
              else "N/A"
          ),
          "RSI (14D)": rsi_val,
          "KGV (P/E)": pe_str,
          "Analysten-Kursziel": target_str,
          "Konsens-Rating": recommendation,
          "Nächste Earnings": earnings_date,
      })
    except Exception:
      data.append({
          "Name / Aktie": raw_input,
          "Ticker": t,
          "Kurs": "N/A",
          "RSI (14D)": "N/A",
          "KGV (P/E)": "N/A",
          "Analysten-Kursziel": "N/A",
          "Konsens-Rating": "N/A",
          "Nächste Earnings": "Nicht gefunden",
      })

  return pd.DataFrame(data), direct_news, resolved_tickers_list


# JETZT 4 TABS: Neuer Tab "🎯 Kauf- / Verkauf-Signale" ergänzt
tab1, tab2, tab3, tab4 = st.tabs([
    "🌍 Markt Top 10",
    "💼 Sentiment & Depot",
    "📅 Earnings-Kalender",
    "🎯 Kauf- / Verkauf-Signale",
])

if st.button("🚀 KI-Analyse starten", use_container_width=True):
  if not GROQ_KEY:
    st.error(
        "⚠️ Kein GROQ_API_KEY in den Streamlit Secrets gefunden! Bitte unter"
        " Settings -> Secrets eintragen."
    )
  else:
    save_portfolio_to_file(portfolio_input)

    client = Groq(api_key=GROQ_KEY.strip())
    raw_tickers = [t.strip() for t in portfolio_input.split(",") if t.strip()]

    with st.spinner(
        f"Löse Ticker auf, lade Daten & analysiere mit Modell"
        f" '{selected_model}'..."
    ):
      news_data = fetch_all_headlines(RSS_SOURCES)
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

      prompt_market = f"""
Aktuelle weltweite Wirtschaftsnachrichten (45+ Quellen & ETF-Feeds):
{news_text}

Fasse die **TOP 10 wichtigsten Markt-Informationen** prägnant auf Deutsch zusammen.
Bewerte am Ende kurz die Marktstimmung (Bullisch / Neutral / Bärisch).
"""

      prompt_depot = f"""
Depot-Werte & Kennzahlen:
{metrics_summary}

Spezifische News zu deinen Titeln:
{ticker_news_text}

Allgemeine Makro-Nachrichten:
{news_text}

Erstelle für jeden Wert eine fundierte Analyse:
1. **Sentiment**: 🟢 Bullisch, 🟡 Neutral oder 🔴 Bärisch
2. **Technik & Bewertung**: Interpretiere kurz den RSI (unter 30 = überverkauft, über 70 = überkauft), das KGV und das Analysten-Kursziel.
3. **Fokus/News**: Relevante Meldungen oder Einflussfaktoren.
4. **Tipp für Anleger**: Konkrete Beobachtungspunkte für die kommenden Tage.
"""

      # Spezieller Prompt für verlässliche Kauf-/Verkauf-Signale
      prompt_signals = f"""
Du bist ein quantitativer Portfolio-Analyst. Hier sind die Echtzeitdaten der Aktien:
{metrics_summary}

News & Makro-Umfeld:
{ticker_news_text}
{news_text}

Erstelle für JEDE Aktie einzeln eine strukturierte Handlungsempfehlung nach diesem Schema:

### [Name der Aktie] ([Ticker])
- **Signal**: 🟢 **KAUFEN** / 🟡 **HALTEN** / 🔴 **VERKAUFEN**
- **Begründung**: (Kombination aus RSI-Charttechnik, KGV-Bewertung und Analysten-Kursziel)
- **Risikolevel**: Niedrig / Mittel / Hoch
- **Empfohlener Anlagehorizont**: Kurzfristig (Trading) / Mittelfristig / Langfristig (Buy & Hold)

Wichtig: Begründe das Signal objektiv mit den harten Kennzahlen (z. B. RSI über 70 = Gewinnmitnahme/Halten statt Neukauf).
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

        res_signals = client.chat.completions.create(
            model=selected_model,
            messages=[{"role": "user", "content": prompt_signals}],
            temperature=0.2,
            max_tokens=1000,
        )
        out_signals = res_signals.choices[0].message.content

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
