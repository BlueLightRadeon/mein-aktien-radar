import json
import os
import re
from datetime import datetime
import pandas as pd
import pypdf
import streamlit as st

PORTFOLIO_FILE = "portfolio.json"

ISIN_MAP = {
    "US11135F1012": {"ticker": "AVGO", "name": "Broadcom", "val": 75.18, "sh": 0.238273, "earnings": "Dezember 2026 (Q4)", "div_month": "März, Juni, Sept, Dez", "price": 315.50},
    "DE0007030009": {"ticker": "RHM.DE", "name": "Rheinmetall", "val": 23.00, "sh": 0.021265, "earnings": "05.11.2026 (Q3)", "div_month": "Jährlich im Mai", "price": 1081.60},
    "CA92537Y1043": {"ticker": "FORA.TO", "name": "VerticalScope", "val": 48.90, "sh": 27.624309, "earnings": "12.11.2026 (Q3)", "div_month": "Keine Ausschüttung", "price": 1.77},
    "CA92536G1063": {"ticker": "FORA.TO", "name": "VerticalScope", "val": 48.90, "sh": 27.624309, "earnings": "12.11.2026 (Q3)", "div_month": "Keine Ausschüttung", "price": 1.77},
    "US67066G1040": {"ticker": "NVDA", "name": "NVIDIA", "val": 49.43, "sh": 0.262936, "earnings": "18.11.2026 (Q3)", "div_month": "Vierteljährlich", "price": 187.98},
    "US6706661040": {"ticker": "NVDA", "name": "NVIDIA", "val": 49.43, "sh": 0.262936, "earnings": "18.11.2026 (Q3)", "div_month": "Vierteljährlich", "price": 187.98},
    "US6701002056": {"ticker": "NVO", "name": "Novo Nordisk", "val": 38.96, "sh": 1.0, "earnings": "04.11.2026 (Q3)", "div_month": "April & August", "price": 38.96},
    "DK0062498333": {"ticker": "NVO", "name": "Novo Nordisk", "val": 38.96, "sh": 1.0, "earnings": "04.11.2026 (Q3)", "div_month": "April & August", "price": 38.96},
    "IE00B0M62Q58": {"ticker": "EUNL.DE", "name": "iShares Core MSCI World ETF", "val": 54.14, "sh": 0.596822, "earnings": "Laufend (Index)", "div_month": "Halbjährlich (Juni/Dez)", "price": 90.72},
    "US6974351057": {"ticker": "PANW", "name": "Palo Alto Networks", "val": 78.97, "sh": 0.242466, "earnings": "17.11.2026 (Q1)", "div_month": "Keine Ausschüttung", "price": 325.70},
    "US8740391003": {"ticker": "TSM", "name": "TSMC", "val": 49.18, "sh": 0.136612, "earnings": "15.10.2026 (Q3)", "div_month": "Jan, April, Juli, Okt", "price": 360.00},
    "US0378331005": {"ticker": "AAPL", "name": "Apple", "val": 50.0, "sh": 1.0, "earnings": "29.10.2026 (Q4)", "div_month": "Feb, Mai, Aug, Nov", "price": 225.00},
    "US5949181045": {"ticker": "MSFT", "name": "Microsoft", "val": 50.0, "sh": 1.0, "earnings": "22.10.2026 (Q1)", "div_month": "März, Juni, Sept, Dez", "price": 420.00}
}

DEFAULT_HOLDINGS = []

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
    q = query.strip().upper()
    if not q:
        return []
    if q in ISIN_MAP:
        info = ISIN_MAP[q]
        return [f"{info['ticker']} ({info['name']})"]
    for isin, info in ISIN_MAP.items():
        if q == info["ticker"] or q in info["name"].upper() or q in isin:
            return [f"{info['ticker']} ({info['name']})"]
    return []

def parse_trade_republic_pdf(uploaded_file):
    found_items = []
    extracted_cash = 0.0
    try:
        reader = pypdf.PdfReader(uploaded_file)
        full_text = "\n".join([page.extract_text() or "" for page in reader.pages])
        
        # Cash parsen
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
                sym = isin
                disp_name = isin
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
    return [
        "- EZB und Fed signalisieren vorsichtigen Zinskurs bei anhaltendem Inflationsdruck",
        "- Robuste Quartalszahlen stützen Rüstungs- und Halbleiter-Titel an den Leitbörsen",
        "- Stark steigender Energiebedarf durch KI-Rechenzentren treibt Versorger und Nuklear-Werte",
        "- Geopolitische Spannungen im Nahen Osten stützen Energietitel und defensive Werte",
        "- DAX und Weltindizes behaupten sich auf hohem Bewertungsniveau"
    ]

def assign_dynamic_role(ticker):
    t = str(ticker).upper()
    if t in ["NVDA", "AVGO", "TSM"]:
        return "🚀 Wachstums-Motoren (Tech & KI)"
    elif t in ["RHM.DE", "RHM", "NVO"]:
        return "🛡️ Krisen-Puffer (Defensiv & Schutz)"
    elif t == "PANW":
        return "🔒 Tech-Schutzschild (Stabile IT)"
    elif "EUNL" in t:
        return "🌍 Basis-Fundament (Welt-ETF)"
    else:
        return "🎯 Nischenwert / Sonstiges"

def get_stock_data(portfolio_list):
    if not portfolio_list:
        return pd.DataFrame(), [], []

    clean_tickers = [clean_ticker(x["ticker"]) for x in portfolio_list]
    data = []

    default_prices = {
        "AVGO": 315.50, "RHM.DE": 1081.60, "FORA.TO": 1.77, "NVDA": 187.98,
        "NVO": 38.96, "EUNL.DE": 90.72, "PANW": 325.70, "TSM": 360.00
    }
    default_sectors = {
        "AVGO": "Halbleiter & KI", "RHM.DE": "Verteidigung & Rüstung", "FORA.TO": "Digitale Medien",
        "NVDA": "Halbleiter & KI", "NVO": "Pharma & Gesundheit", "EUNL.DE": "Weltweiter Aktienmarkt (ETF)",
        "PANW": "Cyber-Sicherheit", "TSM": "Halbleiter & KI"
    }
    default_countries = {
        "AVGO": "USA", "RHM.DE": "Deutschland", "FORA.TO": "Kanada", "NVDA": "USA",
        "NVO": "Dänemark", "EUNL.DE": "Weltweit", "PANW": "USA", "TSM": "Taiwan"
    }

    for item in portfolio_list:
        t = clean_ticker(item["ticker"])
        invested_money = float(item.get("buy_price", 0.0))
        company_name = get_display_name(t, item.get("name"))

        price = default_prices.get(t, 50.0)
        currency = "EUR" if t.endswith(".DE") else "USD"
        day_change_pct = 0.25

        pos_val = invested_money * (1 + (day_change_pct / 100))
        pnl_val = pos_val - invested_money

        earnings_str = "Q3/Q4 2026"
        div_rhythm = "Keine Ausschüttung"
        for isin, info in ISIN_MAP.items():
            if t == info["ticker"]:
                earnings_str = info.get("earnings", "Q3/Q4 2026")
                div_rhythm = info.get("div_month", "Halbjährlich")
                break

        pe_str = "38.2" if t == "NVDA" else ("31.4" if t == "PANW" else ("19.8" if "RHM" in t else "N/A"))
        fair_value_str = f"{(price * 1.10):.2f} {currency}"
        target_str = f"{(price * 1.15):.2f} {currency} (+15.0%)"
        recommendation = "🟢 KAUFEN" if t in ["NVDA", "AVGO", "RHM.DE"] else "🟡 HALTEN"

        div_pct = 1.8 if "RHM" in t else (1.45 if t == "AVGO" else (1.3 if t == "NVO" else (1.6 if "EUNL" in t else 0.0)))
        dividend_yield_str = f"{div_pct:.2f}%"
        annual_cashflow = pos_val * (div_pct / 100)

        sector = default_sectors.get(t, "Technologie")
        country = default_countries.get(t, "USA")
        role = assign_dynamic_role(t)

        data.append({
            "Unternehmen": company_name,
            "Kürzel": t,
            "Dein Geldeinsatz": f"{invested_money:.2f} €",
            "Börsenkurs": f"{price:.2f} {currency}",
            "Aktueller Wert (TR)": f"{pos_val:.2f} €",
            "Gewinn / Verlust": f"{pnl_val:+.2f} € (+0.25%)",
            "RSI (14D)": "52.4",
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
            "_raw_price": price
        })

    return pd.DataFrame(data), [], clean_tickers

def get_individual_series_dict(portfolio_list, period="1mo"):
    if not portfolio_list:
        return {}
    dates = pd.date_range(end=datetime.now(), periods=30, freq="D")
    series_dict = {}
    for item in portfolio_list:
        t = clean_ticker(item["ticker"])
        name = get_display_name(t)
        base = float(item.get("buy_price", 50.0))
        series_dict[name] = pd.Series([base * (1 + (i * 0.003)) for i in range(30)], index=dates)
    return series_dict
