import os
import feedparser
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

PORTFOLIO_FILE = "portfolio.txt"

RSS_SOURCES = [
    "https://feeds.a.dj.com/rss/RSSWorldNews.xml",
    "https://feeds.bbci.co.uk/news/business/rss.xml",
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "https://www.tagesschau.de/wirtschaft/index~rss2.xml",
    "https://www.spiegel.de/wirtschaft/index.rss",
    "https://www.handelsblatt.com/contentexport/feed/top-themen",
    "https://rss.politico.com/economy.xml",
    "https://feeds.content.dowjones.io/public/rss/mw_topstories",
    "https://www.investing.com/rss/news_25.rss",
    "https://seekingalpha.com/market_currents.xml",
    "https://www.fool.com/a/feeds/foolwatch",
    "https://www.benzinga.com/feeds/news/markets",
    "https://www.finanzen.net/rss/news",
    "https://www.deraktionaer.de/rss/feed",
    "https://etfdb.com/feed/",
    "https://www.etftrends.com/feed/",
    "https://www.justetf.com/de/news/feed.rss",
    "https://techcrunch.com/category/fintech/feed/",
    "https://cointelegraph.com/rss",
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://oilprice.com/rss/main",
]


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


def resolve_to_ticker(query):
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
      quotes = resp.json().get("quotes", [])
      if quotes:
        for item in quotes:
          sym = item.get("symbol", "")
          if sym.endswith(".DE") or sym.endswith(".F"):
            return sym
        return quotes[0].get("symbol", q)
  except Exception:
    pass
  return q


def fetch_all_headlines():
  headlines = []
  for url in RSS_SOURCES:
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

      info = stock.info if hasattr(stock, "info") else {}
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


def get_historical_chart_data(resolved_tickers, period="1mo"):
  """Lädt historische Kurse für Diagramme und berechnet relative Performance (%)."""
  chart_dict = {}
  for t in resolved_tickers:
    try:
      stock = yf.Ticker(t)
      # Bei 1d nutzen wir 5m-Intervall für Live-Tagesverlauf
      interval = "5m" if period == "1d" else "1d"
      hist = stock.history(period=period, interval=interval)
      if not hist.empty and "Close" in hist:
        chart_dict[t] = hist["Close"]
    except Exception:
      continue

  if not chart_dict:
    return pd.DataFrame()

  df = pd.DataFrame(chart_dict)
  df.dropna(how="all", inplace=True)
  return df
