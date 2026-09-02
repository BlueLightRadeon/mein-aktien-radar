import os
import json
import re
import time
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import streamlit as st

MEMORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stock_memory.json")

def load_memory():
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_memory(memory_data):
    """Speichert den Wissensstand lokal ab, ohne GitHub-Reboot-Schleifen auszulösen."""
    try:
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(memory_data, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

def audit_and_update_learning(ticker_sym, current_price, history_series, memory):
    t = ticker_sym.upper()
    if t not in memory:
        memory[t] = {"name": t, "history": [], "learning_profile": {}}

    stock_data = memory[t]
    history = stock_data.get("history", [])

    if not history or history_series is None or history_series.empty:
        return stock_data.get("learning_profile", {})

    total_evals = 0
    direction_hits = 0
    abs_errors = []
    signed_errors = []

    for entry in history:
        entry_ts = entry.get("timestamp")
        entry_price = float(entry.get("price", current_price))
        t_30 = float(entry.get("target_30d", entry_price))

        if not entry_ts:
            continue

        days_passed = (time.time() - entry_ts) / 86400.0

        if days_passed >= 3.0:
            entry_date = datetime.fromtimestamp(entry_ts).date()
            sub_series = history_series[history_series.index.date >= entry_date]
            
            if not sub_series.empty:
                actual_price = float(sub_series.iloc[-1])
                pred_up = t_30 > entry_price
                actual_up = actual_price > entry_price

                if pred_up == actual_up:
                    direction_hits += 1

                err_signed = ((t_30 - actual_price) / actual_price) * 100.0
                err_abs = abs(err_signed)

                abs_errors.append(err_abs)
                signed_errors.append(err_signed)
                total_evals += 1

                entry["audit_status"] = "Geprüft"
                entry["actual_evaluated_price"] = round(actual_price, 2)
                entry["error_pct"] = round(err_abs, 2)
                entry["direction_correct"] = (pred_up == actual_up)

    if total_evals > 0:
        dir_accuracy = round((direction_hits / total_evals) * 100.0, 1)
        mean_error = round(sum(abs_errors) / total_evals, 2)
        mean_bias = round(sum(signed_errors) / total_evals, 2)

        bias_factor = round(1.0 - (mean_bias / 100.0), 3)
        bias_factor = max(0.85, min(1.15, bias_factor))

        if mean_bias > 1.5:
            tendency = f"Zu optimistisch (+{mean_bias} %)"
        elif mean_bias < -1.5:
            tendency = f"Zu vorsichtig ({mean_bias} %)"
        else:
            tendency = "Präzise kalibriert"

        profile = {
            "total_evaluations": total_evals,
            "direction_accuracy_pct": dir_accuracy,
            "avg_error_pct": mean_error,
            "bias_tendency": tendency,
            "bias_factor": bias_factor,
            "last_audit": datetime.now().strftime("%d.%m.%Y %H:%M Uhr")
        }
    else:
        profile = {
            "total_evaluations": 0,
            "direction_accuracy_pct": 100.0,
            "avg_error_pct": 0.0,
            "bias_tendency": "Lernphase (Erste Prognosen aktiv)",
            "bias_factor": 1.0,
            "last_audit": datetime.now().strftime("%d.%m.%Y %H:%M Uhr")
        }

    stock_data["learning_profile"] = profile
    save_memory(memory)
    return profile

def run_monte_carlo(current_price, volatility_annual, return_365d, trend_status, days=30, simulations=1000, bias_factor=1.0, ticker_sym=""):
    try:
        today_str = datetime.now().strftime("%Y-%m-%d")
        seed_val = int(abs(hash(f"{ticker_sym}_{today_str}_{round(current_price, 2)}")) % (2**31 - 1))
        rng = np.random.default_rng(seed_val)

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

        mu = mu * bias_factor

        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * rng.normal(size=(days, simulations))
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
    end_date = last_date + timedelta(days=31)
    future_dates = pd.bdate_range(start=last_date, end=end_date)
    if len(future_dates) < 15:
        future_dates = pd.date_range(start=last_date, periods=22, freq="B")
        
    n_days = len(future_dates)
    idx_7 = min(5, n_days - 3)
    idx_14 = min(10, n_days - 2)
    idx_30 = n_days - 1

    seed_val = int(abs(hash(str(ticker_seed) + str(last_date) + str(round(p_curr, 1)))) % (2**31 - 1))
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
    except Exception:
        pass
    return fund

@st.cache_data(ttl=900, show_spinner=False)
def fetch_365d_stats(ticker_sym, bias_factor=1.0):
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
        if len(close) < 20:
            return None, None

        current_p = float(close.iloc[-1])
        start_p = float(close.iloc[0])
        ret_365d = ((current_p - start_p) / start_p) * 100.0

        sma20 = float(close.tail(20).mean())
        sma50 = float(close.tail(50).mean())
        sma200 = float(close.tail(200).mean()) if len(close) >= 200 else float(close.mean())

        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi_series = 100 - (100 / (1 + rs))
        current_rsi = float(rsi_series.dropna().iloc[-1]) if not rsi_series.dropna().empty else 50.0

        rolling_std_20 = float(close.tail(20).std())
        bb_upper = round(sma20 + (2.0 * rolling_std_20), 2)
        bb_lower = round(sma20 - (2.0 * rolling_std_20), 2)

        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        macd_val = round(float(macd_line.iloc[-1]), 2)
        signal_val = round(float(signal_line.iloc[-1]), 2)
        macd_status = "Bullisch (MACD über Signallinie)" if macd_val > signal_val else "Bärisch (MACD unter Signallinie)"

        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        atr_14 = round(float(tr.tail(14).mean()), 2)

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

        mc_worst, mc_median, mc_best = run_monte_carlo(
            current_p, volatility, ret_365d, trend_status, days=30, bias_factor=bias_factor, ticker_sym=ticker_sym
        )
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
    profile = past_stock_memory.get("learning_profile", {})
    past_predictions = past_stock_memory.get("history", [])

    fund = fetch_fundamental_and_wallstreet(ticker_sym)
    curr_sym = fund.get("currency_symbol", stats.get("currency_symbol", "€"))

    accuracy_eval = f"""
AUTONOMES SELBSTLERN-FEEDBACK (Reale Soll-Ist-Auswertung):
- Durchgeführte Prüfungen: {profile.get('total_evaluations', 0)}
- Historische Richtungstrefferquote: {profile.get('direction_accuracy_pct', 100)} %
- Mittlere Abweichung: {profile.get('avg_error_pct', 0)} %
- Bisherige Verzerrung: {profile.get('bias_tendency', 'Neutral')}
- Aktiver Dämpfungsfaktor: {profile.get('bias_factor', 1.0)}x
"""

    if past_predictions:
        accuracy_eval += "\nLETZTE PROGNOSEN & REALE ABWEICHUNGEN:\n"
        seen_dates = set()
        for entry in reversed(past_predictions):
            p_date = entry.get("date", "Unbekannt").split(" ")[0]
            if p_date in seen_dates:
                continue
            seen_dates.add(p_date)
            p_price = entry.get("price", 0.0)
            p_t30 = entry.get("target_30d", p_price)
            p_act = entry.get("actual_evaluated_price", stats["current_price"])
            p_err = entry.get("error_pct", 0.0)
            accuracy_eval += f"- {entry.get('date')}: Kurs {p_price:.2f} {curr_sym} -> Ziel: {p_t30:.2f} {curr_sym} | Realkurs: {p_act:.2f} {curr_sym} (Fehler: {p_err}%)\n"
            if len(seen_dates) >= 3:
                break

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
        f"- RSI (14 Tage): {stats['rsi_14']}",
        f"- Bollinger-Bänder (20 Tage): Unteres Band {stats['bb_lower']} {curr_sym} | Oberes Band {stats['bb_upper']} {curr_sym}",
        f"- MACD-Signal: {stats['macd_status']}",
        f"- Reale Tagesschwankung (ATR 14T): ±{stats['atr_14']} {curr_sym} pro Handelstag",
        f"- Wichtigste Chartmarken: Unterstützung bei {stats['support_level']} {curr_sym} | Widerstand bei {stats['resistance_level']} {curr_sym}",
        f"- Volatilität: {stats['volatility_pct']} % | Letzte 5 Tage: {stats['last_5_days']}",
        "",
        "[2. STATISTISCHE MONTE-CARLO-SIMULATION (1.000 Pfade über 30 Tage)]",
        f"- Statistischer Median in 30 Tagen: {stats['mc_median_30d']} {curr_sym}",
        f"- Statistischer 95%-Worst-Case: {stats['mc_worst_30d']} {curr_sym}",
        f"- Statistischer 95%-Best-Case: {stats['mc_best_30d']} {curr_sym}",
        "",
        "[3. WALL-STREET-KONSENS & GEWINNTERMINE]",
        f"- Offizielles Analystenziel: {fund['target_mean'] if fund['target_mean'] else 'Kein Konsens'} {curr_sym}",
        f"- Spanne: Tief {fund['target_low']} bis Hoch {fund['target_high']} {curr_sym}",
        f"- Einstufung: {fund['recommendation']}",
        "",
        "[4. NACHRICHTENLAGE]",
        macro_news,
        specific_news,
        "",
        "[5. LERNGEDÄCHTNIS]",
        accuracy_eval,
        "",
        "STRIKTE ANWEISUNG FÜR DIE 30-TAGE-PROGNOSE:",
        "Berechne EXAKTE, MATHEMATISCH FUNDIERTE Zahlen auf Basis von MACD, Bollinger-Bändern und Trendrichtung!",
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
        "(Bewerte den Einfluss des Makroumfelds auf die kommenden 30 Tage)",
        "",
        "### 2. 🧠 Erkenntnisse aus bisherigen Fehlern & Indikatoren",
        "(Erkläre die Signale aus Bollinger-Bändern, MACD und ATR für die nächsten 30 Tage)",
        "",
        "### 3. 🎯 Begründung der 30-Tage-Kursprognose",
        "(Konkrete Begründung der Meilensteine bei Tag 7, Tag 14 und Tag 30 sowie Stop-Loss)",
        "",
        "### 4. 🧭 Konkreter 30-Tage-Handelsplan",
        "(Praxisnahe Handlungsanweisung: Limit-Kauf, Trailing Stop-Loss oder Halten)"
    ]
    prompt = "\n".join(prompt_lines)

    full_output = ""
    try:
        res = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "Du bist ein quantitativer deutscher Börsenanalyst. Antworte ausschließlich auf Deutsch. Gib niemals Denkprozesse (<think>) aus."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            seed=42,
            max_tokens=1500,
            timeout=25.0
        )
        full_output = res.choices[0].message.content
    except Exception as e:
        full_output = f"⚠️ Fehler bei der Analyse: {e}"

    full_output = re.sub(r'<think>.*?</think>', '', full_output, flags=re.DOTALL).strip()
    full_output = re.sub(r'^.*?Here\'s a thinking process.*?\n\n', '', full_output, flags=re.DOTALL | re.IGNORECASE).strip()

    p_curr = stats['current_price']
    
    fallback_factor = 1.03 if "Aufwärtstrend" in stats['trend_status'] else (0.97 if "Abwärtstrend" in stats['trend_status'] else 1.0)
    fallback_factor *= profile.get("bias_factor", 1.0)
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
        memory[t] = {"name": company_name, "history": [], "learning_profile": {}}

    today_prefix = datetime.now().strftime("%d.%m.%Y")
    existing_idx = None
    for idx, entry in enumerate(memory[t]["history"]):
        if entry.get("date", "").startswith(today_prefix):
            existing_idx = idx
            break

    new_entry = {
        "timestamp": time.time(),
        "date": datetime.now().strftime("%d.%m.%Y %H:%M Uhr"),
        "price": stats['current_price'],
        "currency": curr_sym,
        "target_7d": targets_dict["t_7"],
        "target_14d": targets_dict["t_14"],
        "target_30d": targets_dict["t_30"],
        "rsi": stats['rsi_14'],
        "trend": stats['trend_status'],
        "analysis_summary": clean_report[:280] + "..."
    }

    if existing_idx is not None:
        memory[t]["history"][existing_idx] = new_entry
    else:
        memory[t]["history"].append(new_entry)
    
    if len(memory[t]["history"]) > 20:
        memory[t]["history"] = memory[t]["history"][-20:]
        
    save_memory(memory)

    return clean_report, targets_dict
