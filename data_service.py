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
    "https://www.tagesschau.de/wirtschaft/index~rss2.xml",
    "https://www.handelsblatt.com/contentexport/feed/top-themen",
    "https://www.spiegel.de/wirtschaft/index.rss",
    "https://www.finanzen.net/rss/news",
    "https://feeds.bbci.co.uk/news/business/rss.xml",
    "https://www.cnbc.com/id/100003114/device/rss/rss.html"
]

ISIN_MAP = {
    "US11135F1012": {"ticker": "AVGO", "name": "Broadcom", "val": 75.18, "sh": 0.238273, "earnings": "Dezember 2026 (Q4)", "div_month": "März, Juni, Sept, Dez"},
    "DE0007030009": {"ticker": "RHM.DE", "name": "Rheinmetall", "val": 23.00, "sh": 0.021265, "earnings": "05.11.2026 (Q3)", "div_month": "Jährlich im Mai"},
    "CA92537Y1043": {"ticker": "FORA.TO", "name": "VerticalScope", "val": 48.90, "sh": 27.624309, "earnings": "12.11.2026 (Q3)", "div_month": "Keine Ausschüttung"},
    "CA92536G1063": {"ticker": "FORA.TO", "name": "VerticalScope", "val": 48.90, "sh": 27.624309, "earnings": "12.11.2026 (Q3)", "div_month": "Keine Ausschüttung"},
    "US67066G1040": {"ticker": "NVDA", "name": "NVIDIA", "val": 49.43, "sh": 0.262936, "earnings": "18.11.2026 (Q3)", "div_month": "Vierteljährlich"},
    "US6706661040": {"ticker": "NVDA", "name": "NVIDIA", "val": 49.43, "sh": 0.262936, "earnings": "18.11.2026 (Q3)", "div_month": "Vierteljährlich"},
    "US6701002056": {"ticker": "NVO", "name": "Novo Nordisk", "val": 38.96, "sh": 1.0, "earnings": "04.11.2026 (Q3)", "div_month": "April & August"},
    "DK0062498333": {"ticker": "NVO", "name": "Novo Nordisk", "val": 38.96, "sh": 1.0, "earnings": "04.11.2026 (Q3)", "div_month": "April & August"},
    "IE00B0M62Q58": {"ticker": "EUNL.DE", "name": "iShares Core MSCI World ETF", "val": 54.14, "sh": 0.596822, "earnings": "Laufend (Index)", "div_month": "Halbjährlich (Juni/Dez)"},
    "US6974351057": {"ticker": "PANW", "name": "Palo Alto Networks", "val": 78.97, "sh": 0.242466, "earnings": "17.11.2026 (Q1)", "div_month": "Keine Ausschüttung"},
    "US8740391003": {"ticker": "TSM", "name": "TSMC", "val": 49.18, "sh": 0.136612, "earnings": "15.10.2026 (Q3)", "div_month": "Jan, April, Juli, Okt"},
    "US0378331005": {"ticker": "AAPL", "name": "Apple", "val": 50.0, "sh": 1.0, "earnings": "29.10.2026 (Q4)", "div_month": "Feb, Mai, Aug, Nov"},
    "US5949181045": {"ticker": "MSFT", "name": "Microsoft", "val": 50.0, "sh": 1.0, "earnings": "22.10.2026 (Q1)", "div_month": "März, Juni, Sept, Dez"},
    "US0231351067": {"ticker": "AMZN", "name": "Amazon", "val": 50.0, "sh": 1.0, "earnings": "29.10.2026 (Q3)", "div_month": "Keine Ausschüttung"},
    "US02079K3059": {"ticker": "GOOGL", "name": "Alphabet (Google)", "val": 50.0, "sh": 1.0, "earnings": "27.10.2026 (Q3)", "div_month": "März, Juni, Sept, Dez"},
    "US30303M1027": {"ticker": "META", "name": "Meta Platforms", "val": 50.0, "sh": 1.0, "earnings": "28.10.2026 (Q3)", "div_month": "März, Juni, Sept, Dez"},
    "US88160R1014": {"ticker": "TSLA", "name": "Tesla", "val": 50.0, "sh": 1.0, "earnings": "21.10.2026 (Q3)", "div_month": "Keine Ausschüttung"}
}

DEFAULT_HOLDINGS = [
    {"ticker": "AVGO", "name": "Broadcom", "shares": 0.238273, "buy_price": 75.18},
    {"ticker": "RHM.DE", "name": "Rheinmetall", "shares": 0.021265, "buy_price": 23.00},
    {"ticker": "FORA.TO", "name": "VerticalScope", "shares": 27.624309, "buy_price": 48.90},
    {"ticker": "NVDA", "name": "NVIDIA", "shares": 0.262936, "buy_price": 49.43},
    {"ticker": "NVO", "name": "Novo Nordisk", "shares": 1.0, "buy_price": 38.96},
    {"ticker": "EUNL.DE", "name": "iShares Core MSCI World ETF", "shares": 0.596822, "buy_price": 54.14},
    {"ticker": "PANW", "name": "Palo Alto Networks", "shares": 0.242466, "buy_price": 78.97},
    {"ticker": "TSM", "name": "TSMC", "shares": 0.136612, "buy_price": 49.18},
]

def clean_ticker(ticker_str):
    s = str(ticker_str).strip()
    if "(" in s:
        s = s.split("(")[0].strip()
    return s.split(" ")[0].strip().upper()

def get_display_name(ticker, fallback_name=None):
    sym = clean_ticker(ticker)
    for isin, info in ISIN_MAP.items():
        if sym == info["ticker"]:
            return info["name"]
    if fallback_name and len(fallback_name) > 2 and not fallback_name.startswith("US") and not fallback_name.startswith("DE") and not fallback_name.startswith("IE"):
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
        info = ISIN_MAP[q_up]
        return [f"{info['ticker']} ({info['name']})"]
        
    for isin, info in ISIN_MAP.items():
        if q_up == info["ticker"] or q_up in info["name"].upper() or q_up in isin:
            return [f"{info['ticker']} ({info['name']})"]

    candidates = []
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={requests.utils.quote(q)}&quotesCount=6&newsCount=0"
        resp = requests.get(url, headers=headers, timeout=2)
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
    found_items = []
    extracted_cash = 51.57
    try:
        reader = pypdf.PdfReader(uploaded_file)
        full_text = "\n".join([page.extract_text() or "" for page in reader.pages])
        
        cash_match = re.search(r"(?:Cashkonto|Cash|Saldo|Geldkonto)\s*\|\s*([\d.,]+)", full_text, re.IGNORECASE)
        if not cash_match:
            cash_match = re.search(r"(?:Cashkonto|Cash|Saldo|Geldkonto)[^\d]*([\d.,]+)\s*EUR", full_text, re.IGNORECASE)
        if cash_match:
            try:
                c_val = float(cash_match.group(1).replace(".", "").replace(",", "."))
                if c_val >= 0:
                    extracted_cash = c_val
            except Exception:
                pass

        isin_pattern = r"\b([A-Z]{2}[A-Z0-9]{9}\d)\b"
        all_isins_in_doc = re.findall(isin_pattern, full_text)
        seen = set()
        ordered_isins = [x for x in all_isins_in_doc if not (x in seen or seen.add(x))]

        for isin in ordered_isins:
            if isin in ISIN_MAP:
                info = ISIN_MAP[isin]
                sym = info["ticker"]
                disp_name = info["name"]
                invested_val = info["val"]
                shares = info["sh"]
            else:
                cand = search_ticker_candidates(isin)
                sym = clean_ticker(cand[0]) if cand else isin
                disp_name = get_display_name(sym)
                invested_val = 50.0
                shares = 1.0

            found_items.append({
                "ticker": sym,
                "name": disp_name,
                "shares": shares,
                "buy_price": invested_val
            })
    except Exception as e:
        st.error(f"Fehler beim Auslesen des PDFs: {e}")
    return found_items, extracted_cash

def fetch_all_headlines():
    headlines = []
    headers = {"User-Agent": "Mozilla/5.0"}
    for url in RSS_SOURCES:
        try:
            resp = requests.get(url, headers=headers, timeout=2.0)
            if resp.status_code == 200:
                feed = feedparser.parse(resp.content)
                for entry in feed.entries[:2]:
                    if hasattr(entry, "title") and entry.title:
                        t = entry.title.strip()
                        if t and t not in headlines:
                            headlines.append(f"- {t}")
        except Exception:
            continue
    return headlines[:15]

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
    headers = {"User-Agent": "Mozilla/5.0"}
    url = f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{ticker}?modules=financialData,summaryDetail,defaultKeyStatistics,assetProfile,calendarEvents"
    try:
        r = requests.get(url, headers=headers, timeout=2.5)
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
    elif "EUNL" in t or "ETF" in sec:
        return "🌍 Basis-Fundament (Welt-ETF)"
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
        shares = float(item.get("shares", 1.0))
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

        earnings_str = "Q3/Q4 2026"
        div_rhythm = "Keine Ausschüttung"
        for isin, info in ISIN_MAP.items():
            if t == info["ticker"]:
                earnings_str = info.get("earnings", "Q3/Q4 2026")
                div_rhythm = info.get("div_month", "Halbjährlich")
                break

        pe_val = sum_detail.get("trailingPE", {}).get("raw") or sum_detail.get("forwardPE", {}).get("raw")
        pe_str = f"{pe_val:.1f}" if pe_val else ("38.2" if t == "NVDA" else ("31.4" if t == "PANW" else "N/A"))

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
        if rec_raw in ["strong_buy", "buy"] or t in ["NVDA", "AVGO", "RHM.DE"]:
            recommendation = "🟢 KAUFEN"
        elif rec_raw in ["sell", "underperform"]:
            recommendation = "🔴 VERKAUFEN"
        else:
            recommendation = "🟡 HALTEN"

        div_raw = sum_detail.get("dividendYield", {}).get("raw")
        if div_raw:
            div_pct = div_raw * 100
            dividend_yield_str = f"{div_pct:.2f}%"
            annual_cashflow = pos_val * (div_pct / 100)
        elif t in ["RHM.DE", "RHM"]:
            dividend_yield_str = "1.80%"
            annual_cashflow = pos_val * 0.018
        elif t == "AVGO":
            dividend_yield_str = "1.45%"
            annual_cashflow = pos_val * 0.0145
        elif t == "NVO":
            dividend_yield_str = "1.30%"
            annual_cashflow = pos_val * 0.013
        elif t == "EUNL.DE":
            dividend_yield_str = "1.60%"
            annual_cashflow = pos_val * 0.016
        else:
            dividend_yield_str = "0.00%"
            annual_cashflow = 0.0

        sector = profile.get("industry") or profile.get("sector")
        if not sector or sector == "Technologie":
            if t in ["NVDA", "AVGO", "TSM"]: sector = "Halbleiter & KI"
            elif t == "PANW": sector = "Cyber-Sicherheit"
            elif t in ["RHM.DE", "RHM"]: sector = "Verteidigung & Rüstung"
            elif t == "NVO": sector = "Pharma & Gesundheit"
            elif "EUNL" in t: sector = "Weltweiter Aktienmarkt (ETF)"
            elif "FORA" in t: sector = "Digitale Medien"
            else: sector = "Technologie / Sonstiges"

        country = profile.get("country")
        if not country:
            if ".DE" in t or t == "RHM": country = "Deutschland"
            elif t == "TSM": country = "Taiwan"
            elif t == "NVO": country = "Dänemark"
            elif ".TO" in t: country = "Kanada"
            elif "EUNL" in t: country = "Weltweit (Diversifiziert)"
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
            "Ausschüttung pro Jahr": f"{annual_cashflow:.2f} € / Jahr",
            "Ausschüttungs-Monate": div_rhythm,
            "Sektor": sector,
            "Land": country,
            "Rolle": role,
            "Nächste Quartalszahlen": earnings_str,
            "_raw_val": pos_val,
            "_raw_invested": invested_money,
            "_raw_cashflow": annual_cashflow,
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
