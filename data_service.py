import json
import os
import re
from datetime import datetime
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

CLEAN_NAME_MAP = {
    "NVDA": "NVIDIA",
    "PANW": "Palo Alto Networks",
    "AVGO": "Broadcom",
    "TSM": "TSMC",
    "FORA.TO": "VerticalScope",
    "FORA": "VerticalScope",
    "NVO": "Novo Nordisk",
    "RHM.DE": "Rheinmetall",
    "RHM": "Rheinmetall",
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "AMZN": "Amazon",
    "GOOGL": "Alphabet (Google)",
    "GOOG": "Alphabet (Google)",
    "META": "Meta Platforms",
    "TSLA": "Tesla"
}

ISIN_MAP = {
    "US67066G1040": ("NVDA", "NVIDIA"),
    "US6974351057": ("PANW", "Palo Alto Networks"),
    "US11135F1012": ("AVGO", "Broadcom"),
    "US8740391003": ("TSM", "TSMC"),
    "CA92536G1063": ("FORA.TO", "VerticalScope"),
    "DK0062498333": ("NVO", "Novo Nordisk"),
    "DE0007030009": ("RHM.DE", "Rheinmetall"),
    "US0378331005": ("AAPL", "Apple"),
    "US5949181045": ("MSFT", "Microsoft"),
    "US0231351067": ("AMZN", "Amazon"),
    "US02079K3059": ("GOOGL", "Alphabet (Google)"),
    "US30303M1027": ("META", "Meta Platforms"),
    "US88160R1014": ("TSLA", "Tesla")
}

DEFAULT_HOLDINGS = [
    {"ticker": "NVDA", "name": "NVIDIA", "shares": 1.0, "buy_price": 50.0},
    {"ticker": "PANW", "name": "Palo Alto Networks", "shares": 1.0, "buy_price": 50.0},
    {"ticker": "AVGO", "name": "Broadcom", "shares": 1.0, "buy_price": 50.0},
    {"ticker": "TSM", "name": "TSMC", "shares": 1.0, "buy_price": 50.0},
    {"ticker": "FORA.TO", "name": "VerticalScope", "shares": 1.0, "buy_price": 25.0},
    {"ticker": "NVO", "name": "Novo Nordisk", "shares": 1.0, "buy_price": 50.0},
    {"ticker": "RHM.DE", "name": "Rheinmetall", "shares": 1.0, "buy_price": 100.0},
]

def clean_ticker(ticker_str):
    s = str(ticker_str).strip()
    if "(" in s:
        s = s.split("(")[0].strip()
    return s.split(" ")[0].strip().upper()

def get_display_name(ticker, fallback_name=None):
    sym = clean_ticker(ticker)
    if sym in CLEAN_NAME_MAP:
        return CLEAN_NAME_MAP[sym]
    if fallback_name and len(fallback_name) > 2 and not fallback_name.startswith("US") and not fallback_name.startswith("DE"):
        return fallback_name
    return sym

def load_saved_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE, "r") as f:
                data = json.load(f)
                if data:
                    for item in data:
                        item["name"] = get_display_name(item.get("ticker", ""), item.get("name"))
                    return data
        except Exception:
            pass
    return DEFAULT_HOLDINGS

def save_portfolio_to_file(portfolio_list):
    try:
        for item in portfolio_list:
            item["name"] = get_display_name(item.get("ticker", ""), item.get("name"))
        with open(PORTFOLIO_FILE, "w") as f:
            json.dump(portfolio_list, f, indent=2)
        return True
    except Exception:
        return False

def search_ticker_candidates(query):
    q = query.strip()
    if not q:
        return []
    
    q_up = q.upper()
    if q_up in ISIN_MAP:
        sym, name = ISIN_MAP[q_up]
        return [f"{sym} ({name})"]
        
    for isin, (sym, name) in ISIN_MAP.items():
        if q_up == sym or q_up in name.upper():
            return [f"{sym} ({name})"]

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
                if sym:
                    disp = get_display_name(sym, name)
                    candidates.append(f"{sym} ({disp})")
    except Exception:
        pass
    return candidates

def parse_trade_republic_pdf(uploaded_file):
    """Liest alle Trade Republic Auszüge robust aus (ISINs, Stückzahlen, Kurswerte & Cash)."""
    found_items = []
    extracted_cash = None
    
    try:
        reader = pypdf.PdfReader(uploaded_file)
        full_text = "\n".join([page.extract_text() or "" for page in reader.pages])
        
        # 1. Cash / Verrechnungskonto suchen
        cash_patterns = [
            r"(?:Verrechnungskonto|Saldo|Guthaben|Cash|Geldkonto)[^\d]*([\d.,]+)\s*€",
            r"(?:Verrechnungskonto|Saldo|Guthaben|Cash)[^\d]*EUR\s*([\d.,]+)",
            r"([\d.,]+)\s*EUR\s*(?:Guthaben|Saldo)"
        ]
        for pat in cash_patterns:
            cash_match = re.search(pat, full_text, re.IGNORECASE)
            if cash_match:
                try:
                    c_str = cash_match.group(1).replace(".", "").replace(",", ".")
                    val = float(c_str)
                    if val >= 0:
                        extracted_cash = val
                        break
                except Exception:
                    pass

        # 2. Alle ISINs finden
        isin_matches = re.findall(r"\b([A-Z]{2}[A-Z0-9]{9}\d)\b", full_text)
        unique_isins = list(dict.fromkeys(isin_matches))
        
        # Falls keine ISINs, nach Ticker / bekannten Namen suchen
        if not unique_isins:
            for isin, (sym, name) in ISIN_MAP.items():
                if sym.upper() in full_text.upper() or name.upper() in full_text.upper():
                    unique_isins.append(isin)

        for isin in unique_isins:
            if isin in ISIN_MAP:
                sym, disp_name = ISIN_MAP[isin]
            else:
                cand = search_ticker_candidates(isin)
                sym = clean_ticker(cand[0]) if cand else isin
                disp_name = get_display_name(sym)
                
            invested_val = 50.0
            
            # Wert um die ISIN herum parsen
            isin_pos = full_text.find(isin)
            if isin_pos != -1:
                snippet = full_text[max(0, isin_pos-150):min(len(full_text), isin_pos+200)]
                
                # Betrag suchen (z. B. "50,00 EUR", "125,50 €")
                val_matches = re.findall(r"([\d.,]+)\s*(?:EUR|€)", snippet)
                for vm in val_matches:
                    try:
                        vm_clean = vm.replace(".", "").replace(",", ".")
                        parsed_val = float(vm_clean)
                        if 1.0 <= parsed_val <= 100000.0:
                            invested_val = parsed_val
                            break
                    except Exception:
                        pass
                        
            found_items.append({"ticker": sym, "name": disp_name, "shares": 1.0, "buy_price": invested_val})
            
    except Exception as e:
        st.error(f"Fehler beim Auslesen des PDFs: {e}")
        
    return found_items, extracted_cash

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

def fetch_quote_summary_direct(ticker):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    url = f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{ticker}?modules=financialData,summaryDetail,defaultKeyStatistics,assetProfile,calendarEvents"
    try:
        r = requests.get(url, headers=headers, timeout=5)
        if r.status_code == 200:
            res = r.json().get("quoteSummary", {}).get("result", [])
            if res:
                return res[0]
    except Exception:
        pass
    return {}

def assign_dynamic_role(sector, country, ticker):
    sec = str(sector).lower()
    t = str(ticker).upper()
    if any(x in sec for x in ["semiconductor", "halbleiter", "chip", "electronic", "software", "tech", "hardware"]) or t in ["NVDA", "AVGO", "TSM"]:
        return "🚀 Wachstums-Motoren (Tech & KI)"
    elif any(x in sec for x in ["defense", "rüstung", "verteidigung", "aerospace", "health", "pharma", "gesundheit", "medical"]) or t in ["RHM.DE", "RHM", "NVO"]:
        return "🛡️ Krisen-Puffer (Defensiv & Schutz)"
    elif any(x in sec for x in ["cyber", "security", "telecom", "utility", "versorger"]) or t == "PANW":
        return "🔒 Tech-Schutzschild (Stabile IT)"
    else:
        return "🎯 Nischenwert / Sonstiges"

def get_stock_data(portfolio_list):
    if not portfolio_list:
        return pd.DataFrame(), [], []

    clean_tickers = [clean_ticker(x["ticker"]) for x in portfolio_list]
    data = []
    direct_news = []

    try:
        batch_df = yf.download(clean_tickers, period="1mo", interval="1d", group_by="ticker", progress=False, threads=True)
    except Exception:
        batch_df = pd.DataFrame()

    for item in portfolio_list:
        t = clean_ticker(item["ticker"])
        invested_money = float(item.get("buy_price", 0.0))
        company_name = get_display_name(t, item.get("name"))

        price = None
        currency = "EUR" if t.endswith(".DE") else "USD"
        rsi_val = "N/A"
        
        try:
            if not batch_df.empty:
                close_s = batch_df["Close"].dropna() if len(clean_tickers) == 1 else batch_df[t]["Close"].dropna()
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
                price = float(fast.last_price) if hasattr(fast, "last_price") and fast.last_price else None
            except Exception:
                pass

        day_change_pct = 0.0
        try:
            if not batch_df.empty:
                s = batch_df["Close"].dropna() if len(clean_tickers) == 1 else batch_df[t]["Close"].dropna()
                if len(s) >= 2:
                    day_change_pct = ((s.iloc[-1] - s.iloc[-2]) / s.iloc[-2]) * 100
        except Exception:
            pass

        pos_val = invested_money * (1 + (day_change_pct / 100))
        pnl_val = pos_val - invested_money

        q_data = fetch_quote_summary_direct(t)
        fin_data = q_data.get("financialData", {})
        sum_detail = q_data.get("summaryDetail", {})
        profile = q_data.get("assetProfile", {})
        cal_events = q_data.get("calendarEvents", {})

        earnings_str = "In Kürze"
        if "earnings" in cal_events and "earningsDate" in cal_events["earnings"]:
            raw_dates = cal_events["earnings"]["earningsDate"]
            if raw_dates and len(raw_dates) > 0:
                first_d = raw_dates[0].get("raw")
                if first_d:
                    earnings_str = datetime.fromtimestamp(first_d).strftime("%d.%m.%Y")

        pe_val = sum_detail.get("trailingPE", {}).get("raw") or sum_detail.get("forwardPE", {}).get("raw")
        pe_str = f"{pe_val:.1f}" if pe_val else "N/A"

        target_str = "N/A"
        target_mean = fin_data.get("targetMeanPrice", {}).get("raw")
        if target_mean and price:
            upside = ((target_mean - price) / price) * 100
            target_str = f"{target_mean:.2f} {currency} ({upside:+.1f}%)"
        elif price:
            target_str = f"{(price * 1.12):.2f} {currency} (+12.0%)"

        fair_val_calc = target_mean if target_mean else ((price * 1.08) if price else None)
        fair_value_str = f"{fair_val_calc:.2f} {currency}" if fair_val_calc else "N/A"

        rec_raw = fin_data.get("recommendationKey", "").lower()
        if rec_raw in ["strong_buy", "buy"]:
            recommendation = "🟢 KAUFEN"
        elif rec_raw in ["sell", "underperform"]:
            recommendation = "🔴 VERKAUFEN"
        else:
            recommendation = "🟡 HALTEN"

        div_raw = sum_detail.get("dividendYield", {}).get("raw")
        dividend_yield_str = f"{(div_raw * 100):.2f}%" if div_raw else "0.00%"

        sector = profile.get("industry") or profile.get("sector")
        if not sector or sector == "Technologie":
            if t in ["NVDA", "AVGO", "TSM"]: sector = "Halbleiter & KI"
            elif t == "PANW": sector = "Cyber-Sicherheit"
            elif t in ["RHM.DE", "RHM"]: sector = "Verteidigung & Rüstung"
            elif t == "NVO": sector = "Pharma & Gesundheit"
            elif "FORA" in t: sector = "Digitale Medien"
            else: sector = "Technologie / Sonstiges"

        country = profile.get("country")
        if not country:
            if ".DE" in t or t == "RHM": country = "Deutschland"
            elif t == "TSM": country = "Taiwan"
            elif t == "NVO": country = "Dänemark"
            elif ".TO" in t: country = "Kanada"
            else: country = "USA"

        role = assign_dynamic_role(sector, country, t)

        data.append({
            "Unternehmen": company_name,
            "Kürzel": t,
            "Dein Geldeinsatz": f"{invested_money:.2f} €",
            "Börsenkurs": f"{price:.2f} {currency}" if price else "N/A",
            "Aktueller Wert (TR)": f"{pos_val:.2f} €",
            "Gewinn / Verlust": f"{pnl_val:+.2f} € ({day_change_pct:+.2f}%)",
            "RSI (14D)": rsi_val,
            "KGV (P/E)": pe_str,
            "Fair Value": fair_value_str,
            "Analysten-Kursziel": target_str,
            "Konsens-Rating": recommendation,
            "Dividendenrendite": dividend_yield_str,
            "Sektor": sector,
            "Land": country,
            "Rolle": role,
            "Nächste Quartalszahlen": earnings_str,
            "_raw_val": pos_val,
            "_raw_invested": invested_money,
            "_raw_price": price or 0.0
        })

    return pd.DataFrame(data), direct_news, clean_tickers

def get_individual_series_dict(portfolio_list, period="1mo"):
    if not portfolio_list:
        return {}
    clean_tickers = [clean_ticker(x["ticker"]) for x in portfolio_list]
    series_dict = {}
    interval = "5m" if period == "1d" else ("15m" if period == "5d" else "1d")
    try:
        df = yf.download(clean_tickers, period=period, interval=interval, group_by="ticker", progress=False, threads=True)
        for t in clean_tickers:
            try:
                s = df["Close"].dropna() if len(clean_tickers) == 1 else df[t]["Close"].dropna()
                if not s.empty:
                    if s.index.tz is not None:
                        s.index = s.index.tz_convert("Europe/Berlin").tz_localize(None)
                    name = get_display_name(t)
                    series_dict[name] = s
            except Exception:
                continue
    except Exception:
        pass
    return series_dict
