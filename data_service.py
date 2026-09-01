import json
import os
import re
import feedparser
import pandas as pd
import pypdf
import requests
import streamlit as st
import yfinance as yf

PORTFOLIO_FILE = "portfolio.json"

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

DEFAULT_HOLDINGS = [
    {
        "ticker": "PANW",
        "name": "Palo Alto Networks",
        "shares": 10.0,
        "buy_price": 300.0,
    },
    {"ticker": "AVGO", "name": "Broadcom", "shares": 15.0, "buy_price": 140.0},
    {"ticker": "NVDA", "name": "NVIDIA", "shares": 25.0, "buy_price": 110.0},
    {"ticker": "TSM", "name": "TSMC", "shares": 20.0, "buy_price": 150.0},
    {
        "ticker": "FORA.TO",
        "name": "VerticalScope",
        "shares": 100.0,
        "buy_price": 8.5,
    },
    {
        "ticker": "NVO",
        "name": "Novo Nordisk",
        "shares": 12.0,
        "buy_price": 125.0,
    },
    {
        "ticker": "RHM.DE",
        "name": "Rheinmetall",
        "shares": 5.0,
        "buy_price": 480.0,
    },
]


def load_saved_portfolio():
  if os.path.exists(PORTFOLIO_FILE):
    try:
      with open(PORTFOLIO_FILE, "r") as f:
        data = json.load(f)
        if data:
          return data
    except Exception:
      pass
  return DEFAULT_HOLDINGS


def save_portfolio_to_file(portfolio_list):
  try:
    with open(PORTFOLIO_FILE, "w") as f:
      json.dump(portfolio_list, f, indent=2)
    return True
  except Exception:
    return False


def clean_ticker(ticker_str):
  s = str(ticker_str).strip()
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


def parse_trade_republic_pdf(uploaded_file):
  found_items = []
  try:
    reader = pypdf.PdfReader(uploaded_file)
    full_text = "\n".join([page.extract_text() or "" for page in reader.pages])
    isin_matches = re.findall(r"\b([A-Z]{2}[A-Z0-9]{9}\d)\b", full_text)
    for isin in set(isin_matches):
      cand = search_ticker_candidates(isin)
      if cand:
        sym = clean_ticker(cand[0])
        found_items.append(
            {"ticker": sym, "name": sym, "shares": 1.0, "buy_price": 0.0}
        )
  except Exception:
    pass
  return found_items


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


def get_stock_data(portfolio_list):
  clean_tickers = [clean_ticker(x["ticker"]) for x in portfolio_list]
  data = []
  direct_news = []

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

  for item in portfolio_list:
    t = clean_ticker(item["ticker"])
    shares = float(item.get("shares", 1.0))
    buy_price = float(item.get("buy_price", 0.0))

    price = None
    currency = "EUR" if t.endswith(".DE") else "USD"
    rsi_val = "N/A"
    company_name = item.get("name") or t

    try:
      if not batch_df.empty:
        close_s = (
            batch_df["Close"].dropna()
            if len(clean_tickers) == 1
            else batch_df[t]["Close"].dropna()
        )
        if not close_s.empty:
          price = close_s.iloc[-1]
          if len(close_s) >= 14:
            rsi_val = calculate_rsi(close_s)
    except Exception:
      pass

    if price is None:
      try:
        s = yf.Ticker(t)
        h = s.history(period="5d")
        if not h.empty:
          price = h["Close"].iloc[-1]
      except Exception:
        pass

    pe_str = "N/A"
    target_str = "N/A"
    recommendation = "HALTEN"
    earnings_date = "Nächste Wochen"
    dividend_yield_str = "0.0%"
    sector = "Technologie / Sonstiges"
    country = "Global"
    fair_value_str = "N/A"

    try:
      stk = yf.Ticker(t)
      inf = stk.info or {}

      if "shortName" in inf and not item.get("name"):
        company_name = inf["shortName"]

      sector = inf.get("sector") or inf.get("industry") or sector
      country = inf.get("country") or ("Deutschland" if ".DE" in t else "USA")

      pe_val = inf.get("trailingPE") or inf.get("forwardPE")
      if pe_val:
        pe_str = f"{pe_val:.1f}"

      book_val = inf.get("bookValue")
      eps_val = inf.get("trailingEps")
      if book_val and eps_val and eps_val > 0 and book_val > 0:
        fv = (22.5 * eps_val * book_val) ** 0.5
        fair_value_str = f"{fv:.2f} {currency}"
      elif target_price_raw := inf.get("targetMeanPrice"):
        fair_value_str = f"{target_price_raw:.2f} {currency}"

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

      div_rate = inf.get("dividendYield")
      if div_rate:
        dividend_yield_str = f"{div_rate * 100:.2f}%"

      if stk.news:
        for n in stk.news[:2]:
          if "title" in n:
            direct_news.append(f"[{company_name}] {n['title']}")
    except Exception:
      pass

    curr_val = (price * shares) if (price and not pd.isna(price)) else 0.0
    invested_val = buy_price * shares
    pnl_val = curr_val - invested_val if invested_val > 0 else 0.0
    pnl_pct = (
        ((curr_val - invested_val) / invested_val * 100)
        if invested_val > 0
        else 0.0
    )

    data.append({
        "Name / Aktie": company_name,
        "Ticker": t,
        "Stückzahl": shares,
        "Kaufkurs": (
            f"{buy_price:.2f} {currency}" if buy_price > 0 else "Keiner"
        ),
        "Aktueller Kurs": (
            f"{price:.2f} {currency}"
            if (price and not pd.isna(price))
            else "N/A"
        ),
        "Positionswert": f"{curr_val:.2f} {currency}",
        "Gewinn / Verlust": f"{pnl_val:+.2f} {currency} ({pnl_pct:+.2f}%)",
        "RSI (14D)": rsi_val,
        "KGV (P/E)": pe_str,
        "Fair Value": fair_value_str,
        "Analysten-Kursziel": target_str,
        "Konsens-Rating": recommendation,
        "Dividendenrendite": dividend_yield_str,
        "Sektor": sector,
        "Land": country,
        "Nächste Quartalszahlen": earnings_date,
        "_raw_val": curr_val,
        "_raw_invested": invested_val,
        "_raw_price": price or 0.0,
    })

  return pd.DataFrame(data), direct_news, clean_tickers


def get_individual_series_dict(portfolio_list, period="1mo"):
  clean_tickers = [clean_ticker(x["ticker"]) for x in portfolio_list]
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
        s = (
            df["Close"].dropna()
            if len(clean_tickers) == 1
            else df[t]["Close"].dropna()
        )
        if not s.empty:
          if s.index.tz is not None:
            s.index = s.index.tz_convert("Europe/Berlin").tz_localize(None)
          series_dict[t] = s
      except Exception:
        continue
  except Exception:
    pass

  return series_dict
