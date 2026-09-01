import os
import re
import feedparser
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

PORTFOLIO_FILE = "portfolio.txt"

# 45+ Feeds
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

# Feste, saubere Ticker-Zuordnung für deine 7 Aktien
DEFAULT_STOCKS = [
    "PANW (Palo Alto Networks)",
    "AVGO (Broadcom)",
    "NVDA (NVIDIA)",
    "TSM (TSMC)",
    "FORA.TO (VerticalScope)",
    "NVO (Novo Nordisk)",
    "RHM.DE (Rheinmetall)",
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
  return DEFAULT_STOCKS


def save_portfolio_to_file(tickers_list):
  try:
    text_data = ", ".join(tickers_list)
    with open(PORTFOLIO_FILE, "w") as f:
      f.write(text_data)
    st.query_params["tickers"] = text_data
    return True
  except Exception:
    return False


def clean_ticker(ticker_str):
  s = ticker_str.strip()
  if "(" in s:
    s = s.split("(")[0].strip()
  return s.split(" ")[0].strip().upper()


def search_ticker_candidates(query):
  q = query.strip()
  if not q:
    return []

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
  """Lädt alle Aktien zuverlässig per Batch-Download, um Yahoo-Blockaden zu umgehen."""
  clean_tickers = [clean_ticker(x) for x in tickers_list]
  data = []
  direct_news = []

  # 1. Kursdaten im Batch laden (1 einziger Request für alle Aktien!)
  try:
    batch_df = yf.download(
        clean_tickers,
        period="1mo",
        interval="1d",
        group_by="ticker",
        progress=False,
        threads=True,
    )
  except Exception:
    batch_df = pd.DataFrame()

  for raw_item, t in zip(tickers_list, clean_tickers):
    price = None
    currency = "EUR" if t.endswith(".DE") else "USD"
    rsi_val = "N/A"
    company_name = (
        raw_item.split("(")[1].replace(")", "").strip()
        if "(" in raw_item
        else t
    )

    # Kurs & RSI aus dem Batch extrahieren
    try:
      if not batch_df.empty:
        if len(clean_tickers) == 1:
          close_s = batch_df["Close"].dropna()
        else:
          close_s = batch_df[t]["Close"].dropna()

        if not close_s.empty:
          price = close_s.iloc[-1]
          if len(close_s) >= 14:
            rsi_val = calculate_rsi(close_s)
    except Exception:
      pass

    # Falls Batch leer war, Einzelfallback
    if price is None:
      try:
        s = yf.Ticker(t)
        h = s.history(period="5d")
        if not h.empty:
          price = h["Close"].iloc[-1]
      except Exception:
        pass

    # Fundamentaldaten & Kursziele
    pe_str = "N/A"
    target_str = "N/A"
    recommendation = "HALTEN"
    earnings_date = "Nächste Wochen"

    try:
      stk = yf.Ticker(t)
      inf = stk.info or {}
      pe_val = inf.get("trailingPE") or inf.get("forwardPE")
      if pe_val:
        pe_str = f"{pe_val:.1f}"

      tp = inf.get("targetMeanPrice")
      if tp and price:
        up = ((tp - price) / price) * 100
        target_str = f"{tp:.2f} {currency} ({up:+.1f}%)"

      rec = inf.get("recommendationKey", "")
      if rec in ["strong_buy", "buy"]:
        recommendation = "KAUFEN"
      elif rec in ["sell", "underperform"]:
        recommendation = "VERKAUFEN"
      else:
        recommendation = "HALTEN"

      if stk.news:
        for n in stk.news[:2]:
          if "title" in n:
            direct_news.append(f"[{company_name}] {n['title']}")
    except Exception:
      pass

    price_str = (
        f"{price:.2f} {currency}"
        if (price is not None and not pd.isna(price))
        else "N/A"
    )

    data.append({
        "Name / Aktie": company_name,
        "Ticker": t,
        "Kurs": price_str,
        "RSI (14D)": rsi_val,
        "KGV (P/E)": pe_str,
        "Analysten-Kursziel": target_str,
        "Konsens-Rating": recommendation,
        "Nächste Quartalszahlen": earnings_date,
    })

  return pd.DataFrame(data), direct_news, clean_tickers


def get_individual_series_dict(tickers_list, period="1mo"):
  """Lädt alle Kurven gleichzeitig und synchronisiert Zeitzonen."""
  clean_tickers = [clean_ticker(x) for x in tickers_list]
  series_dict = {}

  interval = "5m" if period == "1d" else ("15m" if period == "5d" else "1d")

  try:
    df = yf.download(
        clean_tickers,
        period=period,
        interval=interval,
        group_by="ticker",
        progress=False,
        threads=True,
    )

    for t in clean_tickers:
      try:
        if len(clean_tickers) == 1:
          s = df["Close"].dropna()
        else:
          s = df[t]["Close"].dropna()

        if not s.empty:
          if s.index.tz is not None:
            s.index = s.index.tz_convert("Europe/Berlin").tz_localize(None)
          series_dict[t] = s
      except Exception:
        continue
  except Exception:
    pass

  return series_dict
