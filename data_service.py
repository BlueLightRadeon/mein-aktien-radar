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
    return [t.strip() for t in url_tickers.split(",") if t.strip()]
  if os.path.exists(PORTFOLIO_FILE):
    try:
      with open(PORTFOLIO_FILE, "r") as f:
        saved = f.read().strip()
        if saved:
          return [t.strip() for t in saved.split(",") if t.strip()]
    except Exception:
      pass
  # Deine gewünschten 7 Werte als vorkonfigurierter Standard
  return [
      "PANW (Palo Alto Networks)",
      "AVGO (Broadcom)",
      "NVDA (NVIDIA)",
      "TSM (TSMC)",
      "FORA.TO (VerticalScope)",
      "NVO (Novo Nordisk)",
      "RHM.DE (Rheinmetall)",
  ]


def save_portfolio_to_file(tickers_list):
  try:
    text_data = ", ".join(tickers_list)
    with open(PORTFOLIO_FILE, "w") as f:
      f.write(text_data)
    st.query_params["tickers"] = text_data
    return True
  except Exception:
    return False


def search_ticker_candidates(query):
  """Sucht online nach allen passenden Aktien/ETFs und gibt eine Auswahlliste zurück."""
  q = query.strip()
  if not q:
    return []

  # Schnelle Direkt-Kürzel für bekannte Tippfehler
  quick_map = {
      "BROADCOMM": "AVGO",
      "BROADCOM": "AVGO",
      "PALO ALTO": "PANW",
      "TSMC": "TSM",
      "NOVO NORDDISK": "NVO",
      "NOVO NORDISK": "NVO",
      "RHEINMETALL": "RHM.DE",
      "VERTICALSCOPE": "FORA.TO",
  }
  if q.upper() in quick_map:
    target = quick_map[q.upper()]
    try:
      stock = yf.Ticker(target)
      name = stock.fast_info.get("shortName") or stock.info.get(
          "shortName", target
      )
      return [f"{target} ({name})"]
    except Exception:
      return [f"{target} ({q.title()})"]

  candidates = []
  try:
    headers = {"User-Agent": "Mozilla/5.0"}
    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={requests.utils.quote(q)}&quotesCount=6&newsCount=0"
    resp = requests.get(url, headers=headers, timeout=4)
    if resp.status_code == 200:
      quotes = resp.json().get("quotes", [])
      for item in quotes:
        sym = item.get("symbol", "")
        name = item.get("shortname") or item.get("longname") or sym
        exch = item.get("exchDisp") or item.get("exchange") or ""
        if sym:
          candidates.append(f"{sym} ({name} - {exch})")
  except Exception:
    pass

  return candidates


def clean_ticker(ticker_str):
  """Extrahiert das reine Börsenkürzel aus Strings wie 'PANW (Palo Alto Networks)'."""
  return ticker_str.split(" ")[0].strip().upper()


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


def get_stock_data(tickers_list):
  data = []
  direct_news = []
  clean_tickers = []

  for item_str in tickers_list:
    t = clean_ticker(item_str)
    clean_tickers.append(t)

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
          info.get("shortName") or info.get("longName") or t
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
          for news_item in stock.news[:2]:
            if "title" in news_item:
              direct_news.append(
                  f"[{company_name} / {t}] {news_item['title']}"
              )
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
          "Name / Aktie": t,
          "Ticker": t,
          "Kurs": "N/A",
          "RSI (14D)": "N/A",
          "KGV (P/E)": "N/A",
          "Analysten-Kursziel": "N/A",
          "Konsens-Rating": "N/A",
          "Nächste Earnings": "Nicht gefunden",
      })

  return pd.DataFrame(data), direct_news, clean_tickers


def get_historical_chart_data(tickers_list, period="1mo"):
  chart_dict = {}

  if period == "1d":
    interval = "5m"
  elif period == "5d":
    interval = "15m"
  elif period in ["1mo", "6mo"]:
    interval = "1d"
  else:
    interval = "1wk" if period == "5y" else "1d"

  for item_str in tickers_list:
    t = clean_ticker(item_str)
    try:
      stock = yf.Ticker(t)
      hist = stock.history(period=period, interval=interval)
      if not hist.empty and "Close" in hist:
        series = hist["Close"].copy()
        if series.index.tz is not None:
          series.index = series.index.tz_convert("Europe/Berlin").tz_localize(
              None
          )
        chart_dict[t] = series
    except Exception:
      continue

  if not chart_dict:
    return pd.DataFrame()

  df = pd.DataFrame(chart_dict)
  df.ffill(inplace=True)
  df.bfill(inplace=True)
  df.dropna(how="all", inplace=True)
  return df
