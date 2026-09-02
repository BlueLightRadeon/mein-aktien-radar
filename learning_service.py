import os
import json
import re
import time
import base64
import urllib.request
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import streamlit as st

MEMORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stock_memory.json")

def sync_to_github(memory_data):
    token = st.secrets.get("GITHUB_TOKEN", "")
    repo = st.secrets.get("GITHUB_REPO", "")
    if not token or not repo:
        return
    try:
        url = f"https://api.github.com/repos/{repo}/contents/stock_memory.json"
        headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "Streamlit-Stock-Radar"
        }
        sha = None
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode())
                sha = data.get("sha")
        except Exception:
            pass

        content_str = json.dumps(memory_data, indent=2, ensure_ascii=False)
        content_b64 = base64.b64encode(content_str.encode()).decode()
        payload = {
            "message": "Update KI-Wissensspeicher [Automatischer Sync]",
            "content": content_b64
        }
        if sha:
            payload["sha"] = sha

        req_put = urllib.request.Request(
            url, 
            data=json.dumps(payload).encode(), 
            headers=headers, 
            method="PUT"
        )
        with urllib.request.urlopen(req_put, timeout=5.0) as resp:
            pass
    except Exception:
        pass

def load_memory():
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_memory(memory_data):
    try:
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(memory_data, f, indent=2, ensure_ascii=False)
        sync_to_github(memory_data)
    except Exception:
        pass

def run_monte_carlo(current_price, volatility_annual, return_365d, trend_status, days=30, simulations=1000):
    try:
        dt = 1.0 / 252.0
        sigma = max(volatility_annual / 100.0, 0.12)
        if "Starker Aufwärtstrend" in trend_status:
            mu = min(0.35, max(0.08, (return_365d / 100.0) * 0.35))
        elif "Aufwärtstrend" in trend_status:
            mu = 0.06
        elif "Abwärtstrend" in trend_status:
            mu = max(-0.35, min(-0.08, (return_365d / 100.0) * 0.35))
        else:
            mu = 0.01

        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * np.random.normal(size=(days, simulations))
        daily_multipliers = np.exp(drift + diffusion)
        price_paths = current_price * np.cumprod(daily_multipliers, axis=0)
        final_prices = price_paths[-1]

        p5 = float(np.percentile(final_prices, 5))
        p50 = float(np.percentile(final_prices, 50))
        p95 = float(np.percentile(final_prices, 95))
        return round(p5, 2), round(p50, 2), round(p95, 2)
    except Exception:
        factor = 1.04 if "Aufwärtstrend" in trend_status else 0.96
        return round(current_price * 0.90, 2), round(current_price * factor, 2), round(current_price * 1.12, 2)

def generate_realistic_30d_forecast_path(last_date, p_curr, t_7, t_14, t_30, t_bull, t_bear, volatility_pct, ticker_seed=""):
    """Erzeugt einen tagesgenauen Börsenverlauf über exakt 30 Tage mit realistischer Mikro-Volatilität."""
    end_date = last_date + timedelta(days=31)
    future_dates = pd.bdate_range(start=last_date, end=end_date)
    if len(future_dates) < 15:
        future_dates = pd.date_range(start=last_date, periods=22, freq="B")
        
    n_days = len(future_dates)
    idx_7 = min(5, n_days - 3)
    idx_14 = min(10, n_days - 2)
    idx_30 = n_days - 1

    seed_val = int(abs(hash(str(ticker_seed) + str(last_date))) % (2**31 - 1))
    rng = np.random.default_rng(seed_val)

    daily_vol = max(0.006, min(0.028, (volatility_pct / 100.0) / np.sqrt(252))) * 0.70

    def brownian_bridge(start_v, end_v, steps):
        if steps <= 1:
            return np.array([start_v, end_v]) if steps == 1 else np.array([start_v])
        shocks = rng.normal(0, daily_vol, size=steps)
        w = np.cumsum(shocks)
        bridge = w - (np.arange(1, steps + 1) / steps) * w[-1]
        t_linear = np.linspace(start_v, end_v, steps + 1)
        path = t_linear[1:] + start_v * bridge
        return np.insert(path, 0, start_v)

    seg1 = brownian_bridge(p_curr, t_7, idx_7)
    seg2 = brownian_bridge(t_7, t_14, idx_14 - idx_7)[1:]
    seg3 = brownian_bridge(t_14, t_30, idx_30 - idx_14)[1:]
    forecast_prices = np.concatenate([seg1, seg2, seg3])

    t_steps = np.arange(n_days)
    time_factor = np.sqrt(t_steps / max(idx_30, 1))

    bull_spread = max(0.0, t_bull - p_curr)
    bear_spread = max(0.0, p_curr - t_bear)

    bull_envelope = p_curr + bull_spread * time_factor
    bear_envelope = p_curr - bear_spread * time_factor

    noise_bull = rng.normal(0, daily_vol * 0.15, size=n_days).cumsum()
    noise_bear = rng.normal(0, daily_vol * 0.15, size=n_days).cumsum()
    
    bull_curve = bull_envelope + p_curr * (noise_bull - (t_steps / n_days) * noise_bull[-1])
    bear_curve = bear_envelope - p_curr * (noise_bear - (t_steps / n_days) * noise_bear[-1])

    bull_curve[0] = p_curr
    bear_curve[0] = p_curr

    bull_curve = np.maximum(bull_curve, forecast_prices * 1.002)
    bear_curve = np.minimum(bear_curve, forecast_prices * 0.998)

    milestones = {
        "dates": [future_dates[0], future_dates[idx_7], future_dates[idx_14], future_dates[idx_30]],
        "prices": [p_curr, forecast_prices[idx_7], forecast_prices[idx_14], forecast_prices[idx_30]],
        "labels": ["Heute", "Ziel 7T", "Ziel 14T", "Ziel 30T"]
    }

    return future_dates, forecast_prices, bull_curve, bear_curve, milestones

def fetch_fundamental_and_wallstreet(ticker_sym):
    fund = {
        "target_mean": None,
        "target_high": None,
        "target_low": None,
        "recommendation": "Keine Einstufung",
        "forward_pe": "k. A.",
        "currency_symbol": "€" if ticker_sym.endswith(".DE") else "$",
        "next_earnings": "Nicht terminiert"
    }
    try:
        t = yf.Ticker(ticker_sym)
        inf = getattr(t, "info", {})
        if inf:
            fund["target_mean"] = inf.get("targetMeanPrice")
            fund["target_high"] = inf.get("targetHighPrice")
            fund["target_low"] = inf.get("targetLowPrice")
            fund["recommendation"] = str(inf.get("recommendationKey", "Halten")).upper()
            if inf.get("forwardPE"):
                fund["forward_pe"] = f"{inf.get('forwardPE'):.1f}"
            cur = str(inf.get("currency", "")).upper()
            if cur in ["USD", "$"]:
                fund["currency_symbol"] = "$"
            elif cur in ["EUR", "€"]:
                fund["currency_symbol"] = "€"

        cal = getattr(t, "calendar", None)
        if cal is not None and not (isinstance(cal, pd.DataFrame) and cal.empty):
            if isinstance(cal, dict) and "Earnings Date" in cal:
                dates = cal["Earnings Date"]
                if dates:
                    fund["next_earnings"] = str(dates[0])
            elif isinstance(cal, pd.DataFrame) and "Earnings Date" in cal.index:
                fund["next_earnings"] = str(cal.loc["Earnings Date"].iloc[0])
    except Exception:
        pass
    return fund

def fetch_365d_stats(ticker_sym):
    try:
        data = yf.download(ticker_sym, period="1y", interval="1d", progress=False, auto_adjust=True)
        if data is None or data.empty:
            return None, None
        
        if isinstance(data.columns, pd.MultiIndex):
            close = data["Close"][ticker_sym] if ticker_sym in data["Close"] else data["Close"].iloc[:, 0]
            high = data["High"][ticker_sym] if ticker_sym in data["High"] else data["High"].iloc[:, 0]
            low = data["Low"][ticker_sym] if ticker_sym in data["Low"] else data["Low"].iloc[:, 0]
        else:
            close = data["Close"]
            high = data["High"]
            low = data["Low"]
            
        close = close.dropna()
        if len(close) < 30:
            return None, None

        current_p = float(close.iloc[-1])
        start_p = float(close.iloc[0])
        ret_365d = ((current_p - start_p) / start_p) * 100.0

        sma20 = float(close.tail(20).mean())
        sma50 = float(close.tail(50).mean())
        sma200 = float(close.tail(200).mean()) if len(close) >= 200 else float(close.mean())

        # RSI (14 Tage)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi_series = 100 - (100 / (1 + rs))
        current_rsi = float(rsi_series.dropna().iloc[-1]) if not rsi_series.dropna().empty else 50.0

        # Bollinger-Bänder (20 Tage, 2 Sigma)
        rolling_std_20 = float(close.tail(20).std())
        bb_upper = round(sma20 + (2.0 * rolling_std_20), 2)
        bb_lower = round(sma20 - (2.0 * rolling_std_20), 2)

        # MACD (12, 26, 9)
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        macd_val = round(float(macd_line.iloc[-1]), 2)
        signal_val = round(float(signal_line.iloc[-1]), 2)
        macd_status = "Bullisch (MACD über Signallinie)" if macd_val > signal_val else "Bärisch (MACD unter Signallinie)"

        # ATR (Average True Range - 14 Tage)
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        atr_14 = round(float(tr.tail(14).mean()), 2)

        # Unterstützungs- & Widerstandsniveaus der letzten 90 Tage
        recent_90 = close.tail(90)
        resistance_level = round(float(recent_90.max()), 2)
        support_level = round(float(recent_90.min()), 2)

        high_365 = float(close.max())
        low_365 = float(close.min())
        volatility = float(close.pct_change().std() * np.sqrt(252) * 100.0)

        recent_trend = [round(float(x), 2) for x in close.tail(5).tolist()]
        
        if current_p > sma20 > sma50:
            trend_status = "Starker Aufwärtstrend"
        elif current_p > sma50:
            trend_status = "Moderater Aufwärtstrend"
        elif current_p < sma50 and current_p < sma200:
            trend_status = "Klarer Abwärtstrend"
        else:
            trend_status = "Seitwärtsphase / Konsolidierung"

        mc_worst, mc_median, mc_best = run_monte_carlo(current_p, volatility, ret_365d, trend_status, days=30)
        currency_sym = "€" if ticker_sym.endswith(".DE") else "$"

        stats_dict = {
            "current_price": round(current_p, 2),
            "currency_symbol": currency_sym,
            "return_365d_pct": round(ret_365d, 2),
            "sma20": round(sma20, 2),
            "sma50": round(sma50, 2),
            "sma200": round(sma200, 2),
            "rsi_14": round(current_rsi, 1),
            "bb_upper": bb_upper,
            "bb_lower": bb_lower,
            "macd_status": macd_status,
            "atr_14": atr_14,
            "resistance_level": resistance_level,
            "support_level": support_level,
            "high_365d": round(high_365, 2),
            "low_365d": round(low_365, 2),
            "volatility_pct": round(volatility, 2),
            "trend_status": trend_status,
            "mc_worst_30d": mc_worst,
            "mc_median_30d": mc_median,
            "mc_best_30d": mc_best,
            "last_5_days": recent_trend,
            "data_points": len(close)
        }
        return stats_dict, close
    except Exception:
        return None, None

def fetch_company_specific_news(ticker_sym):
    try:
        t = yf.Ticker(ticker_sym)
        news_items = getattr(t, "news", [])
        headlines = []
        if news_items:
            for item in news_items[:3]:
                title = item.get("title", "")
                if title:
                    headlines.append(f"- {title}")
        return "\n".join(headlines) if headlines else "Keine aktuellen Unternehmensmeldungen vorhanden."
    except Exception:
        return "Keine aktuellen Unternehmensmeldungen vorhanden."

def parse_num_tolerant(key_name, text, fallback):
    pattern = rf'{key_name}\s*[:=]\s*([0-9.,]+)'
    m = re.search(pattern, text, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1).replace(",", "."))
        except Exception:
            pass
    return fallback

def run_ai_learning_prediction(client, model_name, ticker_sym, company_name, stats, memory, macro_news=""):
    t = ticker_sym.upper()
    past_stock_memory = memory.get(t, {})
    past_predictions = past_stock_memory.get("history", [])

    fund = fetch_fundamental_and_wallstreet(ticker_sym)
    curr_sym = fund.get("currency_symbol", stats.get("currency_symbol", "€"))

    accuracy_eval = "ERSTMALIGE ANALYSE: Noch kein Vorwissen für diesen Ticker gespeichert."
    if past_predictions:
        accuracy_eval = "HISTORISCHE ABWEICHUNGEN AUS DEINEM SPEICHER:\n"
        for entry in past_predictions[-4:]:
            prev_p = entry.get("price", 0.0)
            target = entry.get("target_30d", prev_p)
            date_str = entry.get("date", "Unbekannt")
            if target > 0:
                diff_pct = ((target - stats["current_price"]) / stats["current_price"]) * 100.0
                accuracy_eval += f"- {date_str}: Prognose 30T war {target:.2f} {curr_sym} -> Kurs heute: {stats['current_price']:.2f} {curr_sym} (Abweichung: {diff_pct:+.1f}%)\n"

    specific_news = fetch_company_specific_news(ticker_sym)

    prompt_lines = [
        "Du bist ein quantitativer Chef-Anlagestratege. Antworte zu 100 % AUF DEUTSCH.",
        "WICHTIG: Gib NIEMALS englische Denkprozesse (<think>) oder englische Wörter aus.",
        "",
        f"[UNTERNEHMEN: {company_name} ({t}) - FOKUS: DIE NÄCHSTEN 30 TAGE]",
        f"WÄHRUNG: {curr_sym}",
        "",
        "[1. ERWEITERTE QUANTITATIVE KENNZAHLEN (30-TAGE-HORIZONT)]",
        f"- Aktueller Börsenkurs: {stats['current_price']} {curr_sym}",
        f"- Bisherige 365-Tage Wertentwicklung: {stats['return_365d_pct']} %",
        f"- Aktueller Trend: {stats['trend_status']}",
        f"- Gleitende Durchschnitte: SMA20: {stats['sma20']} | SMA50: {stats['sma50']} | SMA200: {stats['sma200']}",
        f"- RSI (14 Tage): {stats['rsi_14']} (unter 30 = überverkauft, über 70 = überkauft)",
        f"- Bollinger-Bänder (20 Tage): Unteres Band {stats['bb_lower']} {curr_sym} | Oberes Band {stats['bb_upper']} {curr_sym}",
        f"- MACD-Signal: {stats['macd_status']}",
        f"- Reale Tagesschwankung (ATR 14T): ±{stats['atr_14']} {curr_sym} pro Handelstag",
        f"- Wichtigste Chartmarken (90T): Unterstützung bei {stats['support_level']} {curr_sym} | Widerstand bei {stats['resistance_level']} {curr_sym}",
        f"- Volatilität: {stats['volatility_pct']} % | Letzte 5 Tage: {stats['last_5_days']}",
        "",
        "[2. STATISTISCHE MONTE-CARLO-SIMULATION (1.000 Pfade über 30 Tage)]",
        f"- Statistischer Median in 30 Tagen: {stats['mc_median_30d']} {curr_sym}",
        f"- Statistischer 95%-Worst-Case: {stats['mc_worst_30d']} {curr_sym}",
        f"- Statistischer 95%-Best-Case: {stats['mc_best_30d']} {curr_sym}",
        "",
        "[3. WALL-STREET-KONSENS & GEWINNTERMINE]",
        f"- Offizielles Analystenziel: {fund['target_mean'] if fund['target_mean'] else 'Kein Konsens'} {curr_sym}",
        f"- Analysten-Spanne: Tief {fund['target_low']} bis Hoch {fund['target_high']} {curr_sym}",
        f"- Einstufung: {fund['recommendation']}",
        f"- Nächste Quartalszahlen: {fund['next_earnings']}",
        "",
        "[4. NACHRICHTENLAGE]",
        macro_news,
        specific_news,
        "",
        "[5. LERNGEDÄCHTNIS]",
        accuracy_eval,
        "",
        "STRIKTE ANWEISUNG FÜR DIE 30-TAGE-PROGNOSE:",
        "Berechne EXAKTE, UNABHÄNGIGE Zahlen für die nächsten 30 Tage auf Basis von MACD, Bollinger-Bändern und Trendrichtung!",
        "- Liegt die Aktie am unteren Bollinger-Band mit überverkauftem RSI? -> Erholung einpreisen.",
        "- Ist die Aktie in einem klaren Abwärtstrend unter SMA50? -> Konservative bzw. fallende Kursziele setzen.",
        "- Beachte die tägliche Schwankungsbreite (ATR), damit die Ziele in 7, 14 und 30 Tagen mathematisch plausibel bleiben.",
        "",
        "Gliedere deine Antwort zwingend in zwei Teile:",
        "",
        "TEIL 1: EXAKTE ZAHLEN",
        "Gib zuerst exakt folgenden Block mit deinen berechneten Werten aus:",
        "PROGNOSE_WERTE_START",
        "ziel_7d = [Kursziel in 7 Tagen]",
        "ziel_14d = [Kursziel in 14 Tagen]",
        "ziel_30d = [Hauptziel in 30 Tagen]",
        "best_case_30d = [Maximales Best-Case-Ziel in 30 Tagen]",
        "worst_case_30d = [Absicherungsmarke / Stop-Loss für 30 Tage]",
        "wahrscheinlichkeit = [Zahl zwischen 60 und 94]",
        "PROGNOSE_WERTE_ENDE",
        "",
        "TEIL 2: AUSFÜHRLICHER DEUTSCHER ANALYSEBERICHT",
        "### 1. 🌐 Synthese: Weltwirtschaft, Quartalstermine & Wall Street",
        "(Bewerte den Einfluss des Makroumfelds und eventueller Quartalszahlen auf die kommenden 30 Tage)",
        "",
        "### 2. 🧠 Technische Tiefenanalyse: Bollinger, MACD & ATR",
        "(Erkläre die Signale aus Bollinger-Bändern, MACD und der durchschnittlichen Tagesschwankung)",
        "",
        "### 3. 🎯 Begründung der 30-Tage-Kursprognose",
        "(Konkrete Begründung der Meilensteine bei Tag 7, Tag 14 und Tag 30 sowie Stop-Loss)",
        "",
        "### 4. 🧭 Konkreter 30-Tage-Handelsplan",
        "(Praxisnahe Handlungsanweisung: Limit-Kauf, Trailing Stop-Loss oder Halten)"
    ]
    prompt = "\n".join(prompt_lines)

    full_output = ""
    for attempt in range(2):
        try:
            res = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "Du bist ein führender deutscher quantitativer Börsenanalyst. Antworte ausschließlich auf Deutsch. Gib niemals englische Denkprozesse (<think>) aus."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.25,
                max_tokens=1500
            )
            full_output = res.choices[0].message.content
            break
        except Exception as e:
            if "rate_limit" in str(e).lower() and attempt == 0:
                time.sleep(2.0)
                continue
            full_output = f"⚠️ Fehler bei der Analyse: {e}"

    full_output = re.sub(r'<think>.*?</think>', '', full_output, flags=re.DOTALL).strip()
    full_output = re.sub(r'^.*?Here\'s a thinking process.*?\n\n', '', full_output, flags=re.DOTALL | re.IGNORECASE).strip()

    p_curr = stats['current_price']
    
    fallback_factor = 1.03 if "Aufwärtstrend" in stats['trend_status'] else (0.97 if "Abwärtstrend" in stats['trend_status'] else 1.0)
    fb_7 = round(p_curr * (1.0 + (fallback_factor - 1.0) * 0.35), 2)
    fb_14 = round(p_curr * (1.0 + (fallback_factor - 1.0) * 0.65), 2)
    fb_30 = round(p_curr * fallback_factor, 2)

    targets_dict = {
        "p_curr": p_curr,
        "currency_symbol": curr_sym,
        "t_7": parse_num_tolerant("ziel_7d", full_output, fb_7),
        "t_14": parse_num_tolerant("ziel_14d", full_output, fb_14),
        "t_30": parse_num_tolerant("ziel_30d", full_output, fb_30),
        "t_bull": parse_num_tolerant("best_case_30d", full_output, stats['mc_best_30d']),
        "t_bear": parse_num_tolerant("worst_case_30d", full_output, stats['mc_worst_30d']),
        "prob": f"{int(parse_num_tolerant('wahrscheinlichkeit', full_output, 80))} %"
    }

    clean_report = re.sub(r'PROGNOSE_WERTE_START.*?PROGNOSE_WERTE_ENDE', '', full_output, flags=re.DOTALL).strip()
    clean_report = re.sub(r'^TEIL\s*2:?[^\n]*\n', '', clean_report, flags=re.IGNORECASE).strip()

    if t not in memory:
        memory[t] = {"name": company_name, "history": []}

    memory[t]["history"].append({
        "date": datetime.now().strftime("%d.%m.%Y %H:%M Uhr"),
        "price": stats['current_price'],
        "currency": curr_sym,
        "target_14d": targets_dict["t_14"],
        "target_30d": targets_dict["t_30"],
        "rsi": stats['rsi_14'],
        "trend": stats['trend_status'],
        "analysis_summary": clean_report[:280] + "..."
    })
    
    if len(memory[t]["history"]) > 15:
        memory[t]["history"] = memory[t]["history"][-15:]
        
    save_memory(memory)

    return clean_report, targets_dict
