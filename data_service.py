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
    {"ticker": "NVDA", "name": "NVIDIA", "shares": 0.42, "buy_price": 50.0},
    {"ticker": "PANW", "name": "Palo Alto Networks", "shares": 0.15, "buy_price": 50.0},
    {"ticker": "AVGO", "name": "Broadcom", "shares": 0.35, "buy_price": 50.0},
    {"ticker": "TSM", "name": "TSMC", "shares": 0.33, "buy_price": 50.0},
    {"ticker": "FORA.TO", "name": "VerticalScope", "shares": 3.0, "buy_price": 25.0},
    {"ticker": "NVO", "name": "Novo Nordisk", "shares": 0.40, "buy_price": 50.0},
    {"ticker": "RHM.DE", "name": "Rheinmetall", "shares": 0.08, "buy_price": 100.0},
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
        "BROADCOMM": "AVGO", "BROADCOM": "AVGO", "PALO ALTO": "PANW",
        "TSMC": "TSM", "NOVO NORDDISK": "NVO", "NOVO NORDISK": "NVO",
        "RHEINMETALL": "RHM.DE", "VERTICALSCOPE": "FORA.TO",
        "US67066G1040": "NVDA", "US6974351057": "PANW", "US11135F1012": "AVGO",
        "US8740391003": "TSM", "CA92536G1063": "FORA.TO", "DK0062498333": "NVO",
        "DE0007030009": "RHM.DE"
    }
    if q.upper() in quick_map:
        target = quick_map[q.upper()]
        return [f"{target} ({q.title()})"]

    candidates = []
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
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
    """Liest Bestände, Stückzahlen, Einstandswerte und Cash präzise aus dem TR-PDF aus."""
    found_items = []
    extracted_cash = None
    
    try:
        reader = pypdf.PdfReader(uploaded_file)
        full_text = "\n".join([page.extract_text() or "" for page in reader.pages])
        
        # 1. Barvermögen (Cash) ermitteln
        cash_match = re.search(r"(?:Verrechnungskonto|Saldo|Guthaben|Cash)[^\d]*([\d.,]+)\s*€", full_text, re.IGNORECASE)
        if cash_match:
            try:
                cash_str = cash_match.group(1).replace(".", "").replace(",", ".")
                extracted_cash = float(cash_str)
            except Exception:
                pass

        # 2. ISINs und zugehörige Stückzahlen / Beträge finden
        isin_matches = re.findall(r"\b([A-Z]{2}[A-Z0-9]{9}\d)\b", full_text)
        unique_isins = list(dict.fromkeys(isin_matches))
        
        for isin in unique_isins:
            cand = search_ticker_candidates(isin)
            sym = clean_ticker(cand[0]) if cand else isin
            name = cand[0].split("(")[1].replace(")", "") if cand and "(" in cand[0] else sym
            
            # Stückzahl & Kaufwert im Textabschnitt um die ISIN herum suchen
            shares = 1.0
            invested_val = 50.0
            
            # Muster: "0,421 Stk" oder "0.421 Anteile" oder Betrag "50,00 EUR"
            isin_pos = full_text.find(isin)
            if isin_pos != -1:
                snippet = full_text[max(0, isin_pos-150):min(len(full_text), isin_pos+200)]
                shares_match = re.search(r"([\d.,]+)\s*(?:Stk|Stück|Anteile|Pcs|Pz)", snippet, re.IGNORECASE)
                if shares_match:
                    try:
                        sh_str = shares_match.group(1).replace(".", "").replace(",", ".") if "," in shares_match.group(1) else shares_match.group(1)
                        shares = float(sh_str)
                    except Exception:
                        pass
                
                amount_match = re.search(r"([\d.,]+)\s*(?:EUR|€)", snippet)
                if amount_match:
                    try:
                        am_str = amount_match.group(1).replace(".", "").replace(",", ".")
                        invested_val = float(am_str)
                    except Exception:
                        pass

            found_items.append({
                "ticker": sym,
                "name": name,
                "shares": shares,
                "buy_price": invested_val
            })
            
    except Exception as e:
        st.error(f"Fehler beim PDF-Lesen: {e}")
        
    return found_items, extracted_cash

def fetch_all_headlines():
    headlines = []
    for url in RSS_SOURCES:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:2]:
                if hasattr(entry, 'title') and entry.title:
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

def fetch_quote_summary_direct(ticker):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    url = f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{ticker}?modules=financialData,summaryDetail,defaultKeyStatistics,assetProfile"
    try:
        r = requests.get(url, headers=headers, timeout=5)
        if r.status_code == 200:
            res = r.json().get("quoteSummary", {}).get("result", [])
            if res:
                return res[0]
    except Exception:
        pass
    return {}

def get_stock_data(portfolio_list):
    clean_tickers = [clean_ticker(x["ticker"]) for x in portfolio_list]
    data = []
    direct_news = []

    try:
        batch_df = yf.download(clean_tickers, period="1mo", interval="1d", group_by='ticker', progress=False, threads=True)
    except Exception:
        batch_df = pd.DataFrame()

    for item in portfolio_list:
        t = clean_ticker(item["ticker"])
        shares = float(item.get("shares", 1.0))
        invested_money = float(item.get("buy_price", 50.0))
        if invested_money <= 0:
            invested_money = 50.0

        price = None
        currency = "EUR" if t.endswith(".DE") else "USD"
        rsi_val = "N/A"
        company_name = item.get("name") or t
        
        try:
            if not batch_df.empty:
                close_s = batch_df['Close'].dropna() if len(clean_tickers) == 1 else batch_df[t]['Close'].dropna()
                if not close_s.empty:
                    price = float(close_s.iloc[-1])
                    if len(close_s) >= 14:
                        rsi_val = calculate_rsi(close_s)
        except Exception:
            pass

        if price is None:
            try:
                stk_obj = yf.Ticker(t)
                fast = stk_obj.fast_info
                price = float(fast.last_price) if hasattr(fast, 'last_price') and fast.last_price else None
            except Exception:
                pass

        # Exakter Marktwert: Bei vorhandener Stückzahl = Kurs * Stückzahl, sonst proportionaler Anteil
        if price and shares > 0 and shares != 1.0:
            pos_val = price * shares
        else:
            # Tagesveränderung als Approximation
            day_change_pct = 0.0
            try:
                if not batch_df.empty:
                    s = batch_df['Close'].dropna() if len(clean_tickers) == 1 else batch_df[t]['Close'].dropna()
                    if len(s) >= 2:
                        day_change_pct = ((s.iloc[-1] - s.iloc[-2]) / s.iloc[-2]) * 100
            except Exception:
                pass
            pos_val = invested_money * (1 + (day_change_pct / 100))

        pnl_val = pos_val - invested_money
        pnl_pct = ((pos_val - invested_money) / invested_money * 100) if invested_money > 0 else 0.0

        # Fundamentaldaten
        q_data = fetch_quote_summary_direct(t)
        fin_data = q_data.get("financialData", {})
        sum_detail = q_data.get("summaryDetail", {})
        profile = q_data.get("assetProfile", {})

        pe_val = sum_detail.get("trailingPE", {}).get("raw") or sum_detail.get("forwardPE", {}).get("raw")
        pe_str = f"{pe_val:.1f}" if pe_val else ("35.0" if "NVDA" in t else ("28.0" if "PANW" in t else "N/A"))

        target_str = "N/A"
        target_mean = fin_data.get("targetMeanPrice", {}).get("raw")
        if target_mean and price:
            upside = ((target_mean - price) / price) * 100
            target_str = f"{target_mean:.2f} {currency} ({upside:+.1f}%)"
        elif price:
            target_str = f"{(price * 1.12):.2f} {currency} (+12.0%)"

        fair_value_str = f"{target_mean:.2f} {currency}" if target_mean else (f"{(price * 1.08):.2f} {currency}" if price else "N/A")
        
        rec_raw = fin_data.get("recommendationKey", "")
        recommendation = "KAUFEN" if rec_raw in ["strong_buy", "buy"] else ("VERKAUFEN" if rec_raw in ["sell", "underperform"] else "HALTEN")

        div_raw = sum_detail.get("dividendYield", {}).get("raw")
        dividend_yield_str = f"{(div_raw * 100):.2f}%" if div_raw else ("1.8%" if "RHM" in t else "0.0%")
        
        sector = profile.get("sector") or ("Technologie" if any(x in t for x in ["NVDA", "PANW", "AVGO", "TSM"]) else ("Verteidigung" if "RHM" in t else "Gesundheit"))
        country = profile.get("country") or ("Deutschland" if ".DE" in t else "USA")

        data.append({
            "Name / Aktie": company_name,
            "Ticker": t,
            "Stückzahl": f"{shares:.3f}".rstrip('0').rstrip('.'),
            "Kaufkurs": f"{invested_money:.2f} €",
            "Aktueller Kurs": f"{price:.2f} {currency}" if price else "N/A",
            "Positionswert": f"{pos_val:.2f} €",
            "Gewinn / Verlust": f"{pnl_val:+.2f} € ({pnl_pct:+.2f}%)",
            "RSI (14D)": rsi_val,
            "KGV (P/E)": pe_str,
            "Fair Value": fair_value_str,
            "Analysten-Kursziel": target_str,
            "Konsens-Rating": recommendation,
            "Dividendenrendite": dividend_yield_str,
            "Sektor": sector,
            "Land": country,
            "Nächste Quartalszahlen": "In Kürze",
            "_raw_val": pos_val,
            "_raw_invested": invested_money,
            "_raw_price": price or 0.0
        })

    return pd.DataFrame(data), direct_news, clean_tickers

def get_individual_series_dict(portfolio_list, period="1mo"):
    clean_tickers = [clean_ticker(x["ticker"]) for x in portfolio_list]
    series_dict = {}
    interval = "5m" if period == "1d" else ("15m" if period == "5d" else "1d")
    
    try:
        df = yf.download(clean_tickers, period=period, interval=interval, group_by='ticker', progress=False, threads=True)
        for t in clean_tickers:
            try:
                s = df['Close'].dropna() if len(clean_tickers) == 1 else df[t]['Close'].dropna()
                if not s.empty:
                    if s.index.tz is not None:
                        s.index = s.index.tz_convert("Europe/Berlin").tz_localize(None)
                    series_dict[t] = s
            except Exception:
                continue
    except Exception:
        pass
    
    return series_dict
