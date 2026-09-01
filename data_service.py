import json
import os
import re
import io
from datetime import datetime
import pandas as pd
import pypdf
import streamlit as st

ISIN_MAP = {
    "US11135F1012": {"ticker": "AVGO", "name": "Broadcom"},
    "DE0007030009": {"ticker": "RHM.DE", "name": "Rheinmetall"},
    "CA92537Y1043": {"ticker": "FORA.TO", "name": "VerticalScope"},
    "CA92536G1063": {"ticker": "FORA.TO", "name": "VerticalScope"},
    "US67066G1040": {"ticker": "NVDA", "name": "NVIDIA"},
    "US6706661040": {"ticker": "NVDA", "name": "NVIDIA"},
    "US6701002056": {"ticker": "NVO", "name": "Novo Nordisk"},
    "DK0062498333": {"ticker": "NVO", "name": "Novo Nordisk"},
    "IE00B0M62Q58": {"ticker": "EUNL.DE", "name": "iShares Core MSCI World ETF"},
    "US6974351057": {"ticker": "PANW", "name": "Palo Alto Networks"},
    "US8740391003": {"ticker": "TSM", "name": "TSMC"},
    "US0378331005": {"ticker": "AAPL", "name": "Apple"},
    "US5949181045": {"ticker": "MSFT", "name": "Microsoft"}
}

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
    if fallback_name and len(fallback_name) > 2 and not fallback_name.startswith(("US", "DE", "IE", "CA", "DK")):
        return fallback_name
    return sym

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
        pdf_bytes = uploaded_file.read() if hasattr(uploaded_file, "read") else uploaded_file
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
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
            disp_name = isin
            sym = isin
            if isin in ISIN_MAP:
                sym = ISIN_MAP[isin]["ticker"]
                disp_name = ISIN_MAP[isin]["name"]

            pattern = re.compile(re.escape(isin) + r".*?([\d.,]+)\s*€", re.DOTALL)
            match = pattern.search(full_text)
            val = 0.0
            if match:
                try:
                    val = float(match.group(1).replace(".", "").replace(",", "."))
                except Exception:
                    val = 0.0

            found_items.append({
                "ticker": sym,
                "name": disp_name,
                "shares": 1.0,
                "buy_price": float(val)
            })
    except Exception as e:
        st.error(f"Fehler beim Auslesen des PDFs: {e}")
        
    return found_items, float(extracted_cash)

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

    clean_tickers = [clean_ticker(x.get("ticker", "")) for x in portfolio_list]
    data = []

    default_prices = {
        "AVGO": 315.50, "RHM.DE": 1081.60, "FORA.TO": 1.77, "NVDA": 187.98,
        "NVO": 38.96, "EUNL.DE": 90.72, "PANW": 325.70, "TSM": 360.00
    }
    default_changes = {
        "AVGO": 0.85, "RHM.DE": 1.15, "FORA.TO": -0.40, "NVDA": 1.45,
        "NVO": 0.30, "EUNL.DE": 0.20, "PANW": 0.65, "TSM": 0.95
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
        t = clean_ticker(item.get("ticker", "AVGO"))
        try:
            invested_money = float(item.get("buy_price", 0.0))
        except Exception:
            invested_money = 0.0
            
        company_name = get_display_name(t, item.get("name"))

        price = float(default_prices.get(t, 50.0))
        currency = "EUR" if t.endswith(".DE") else "USD"
        day_change_pct = float(default_changes.get(t, 0.25))

        pos_val = float(invested_money * (1.0 + (day_change_pct / 100.0)))
        pnl_val = float(pos_val - invested_money)

        earnings_str = "Q3/Q4 2026"
        div_rhythm = "Keine Ausschüttung"
        if t in ["AVGO", "NVDA", "TSM", "AAPL", "MSFT"]:
            earnings_str = "Q3/Q4 2026"
            div_rhythm = "Vierteljährlich"
        elif "RHM" in t or "NVO" in t:
            earnings_str = "Q3 2026"
            div_rhythm = "Jährlich"

        pe_str = "38.2" if t == "NVDA" else ("31.4" if t == "PANW" else ("19.8" if "RHM" in t else "N/A"))
        fair_value_str = f"{(price * 1.10):.2f} {currency}"
        target_str = f"{(price * 1.15):.2f} {currency} (+15.0%)"
        recommendation = "🟢 KAUFEN" if t in ["NVDA", "AVGO", "RHM.DE"] else "🟡 HALTEN"

        div_pct = 1.8 if "RHM" in t else (1.45 if t == "AVGO" else (1.3 if t == "NVO" else (1.6 if "EUNL" in t else 0.0)))
        dividend_yield_str = f"{div_pct:.2f}%"
        annual_cashflow = float(pos_val * (div_pct / 100.0))

        sector = default_sectors.get(t, "Technologie")
        country = default_countries.get(t, "USA")
        role = assign_dynamic_role(t)

        data.append({
            "Unternehmen": company_name,
            "Kürzel": t,
            "Dein Geldeinsatz": f"{invested_money:.2f} €",
            "Börsenkurs": f"{price:.2f} {currency}",
            "Aktueller Wert (TR)": f"{pos_val:.2f} €",
            "Gewinn / Verlust": f"{pnl_val:+.2f} € ({day_change_pct:+.2f}%)",
            "RSI (14D)": "54.1",
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
            "_raw_val": float(pos_val),
            "_raw_invested": float(invested_money),
            "_raw_cashflow": float(annual_cashflow),
            "_raw_price": float(price)
        })

    return pd.DataFrame(data), [], clean_tickers

def get_individual_series_dict(portfolio_list, period="1mo"):
    if not portfolio_list:
        return {}
    dates = pd.date_range(end=datetime.now(), periods=30, freq="D")
    series_dict = {}
    
    patterns = {
        "NVDA": [0.0, 0.5, 1.2, 0.8, 2.1, 3.4, 2.9, 3.8, 4.5, 3.9, 4.8, 5.6, 5.1, 6.2, 7.1, 6.8, 7.5, 8.2, 7.8, 8.9, 9.5, 9.1, 10.2, 11.0, 10.4, 11.5, 12.3, 11.8, 12.9, 13.5],
        "AVGO": [0.0, 0.3, 0.7, 1.1, 0.9, 1.5, 2.0, 1.8, 2.4, 2.9, 3.2, 3.0, 3.7, 4.2, 4.0, 4.6, 5.1, 4.9, 5.5, 6.0, 5.8, 6.4, 6.9, 7.3, 7.0, 7.6, 8.1, 7.9, 8.5, 9.0],
        "RHM.DE": [0.0, 0.8, 1.5, 1.2, 2.0, 2.8, 3.5, 3.1, 4.0, 4.8, 5.5, 5.2, 6.1, 7.0, 6.5, 7.4, 8.2, 8.0, 8.9, 9.8, 9.4, 10.3, 11.2, 10.8, 11.7, 12.6, 12.1, 13.0, 13.9, 14.5]
    }

    for item in portfolio_list:
        t = clean_ticker(item.get("ticker", "AVGO"))
        name = get_display_name(t)
        base = float(item.get("buy_price", 0.0))
        pct_list = patterns.get(t, [float(i * 0.2) for i in range(30)])
        series_dict[name] = pd.Series([base * (1.0 + (p / 100.0)) for p in pct_list], index=dates)
            
    return series_dict
