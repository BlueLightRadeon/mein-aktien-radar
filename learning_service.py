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
    try:
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(memory_data, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

def audit_and_update_learning(ticker_sym, current_price, history_series, memory):
    """Vergleicht vergangene Prognosen mit echten Realkursen und berechnet den Bias-Dämpfer."""
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
        entry_price = float(entry.get("price") or current_price)
        t_30 = float(entry.get("target_30d") or entry_price)

        if not entry_ts:
            continue

        days_passed = (time.time() - entry_ts) / 86400.0

        # Prüfe, sobald mindestens 1 Tag vergangen ist
        if days_passed >= 1.0:
            entry_date = datetime.fromtimestamp(entry_ts).date()
            sub_series = history_series[history_series.index.date >= entry_date]
            
            if not sub_series.empty:
                actual_price = float(sub_series.iloc[-1])
                pred_up = t_30 > entry_price
                actual_up = actual_price > entry_price

                is_correct = (pred_up == actual_up)
                if is_correct:
                    direction_hits += 1

                err_signed = ((t_30 - actual_price) / actual_price) * 100.0 if actual_price > 0 else 0.0
                err_abs = abs(err_signed)

                abs_errors.append(err_abs)
                signed_errors.append(err_signed)
                total_evals += 1

                entry["audit_status"] = "Geprüft"
                entry["actual_evaluated_price"] = round(actual_price, 2)
                entry["error_pct"] = round(err_abs, 2)
                entry["direction_correct"] = is_correct

    if total_evals > 0:
        dir_accuracy = round((direction_hits / total_evals) * 100.0, 1)
        mean_error = round(sum(abs_errors) / total_evals, 2)
        mean_bias = round(sum(signed_errors) / total_evals, 2)

        # Bias-Faktor dämpft künftige Prognosen ab (z. B. 0.95x bei chronischer Übertreibung)
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

        p5 = max(float(np.percentile(final_prices, 5)), current_price * 0.6)
        p50 = float(np.percentile(final_prices, 50))
        p95 = float(np.percentile(final_prices, 95))
        return round(p5, 2), round(p50, 2), round(p95, 2)
    except Exception:
        factor = 1.04 if "Aufwärtstrend" in trend_status else 0.96
        return round(current_price * 0.92, 2), round(current_price * factor, 2), round(current_price * 1.10, 2)

def generate_realistic_30d_forecast_path(last_date, p_curr, t_7, t_14, t_30, t_bull, t_bear, volatility_pct, ticker_seed=""):
    """Generiert einen stabilen Kursverlauf, der NIEMALS ins Negative fallen kann."""
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

    daily_vol = max(0.006, min(0.025, (volatility_pct / 100.0) / np.sqrt(252))) * 0.40

    def brownian_bridge(start_v, end_v, steps):
        if steps <= 1:
            return np.array([start_v, end_v]) if steps == 1 else np.array([start_v])
        shocks = rng.normal(0, daily_vol, size=steps)
        w = np.cumsum(shocks)
        bridge = w - (np.arange(1, steps + 1) / steps) * w[-1]
        t_linear = np.linspace(start_v, end_v, steps + 1)
        path = t_linear[1:] + start_v * bridge
        # Absicherung: Kurs kann niemals unter 60% des Startwerts fallen
        return np.maximum(path, start_v * 0.6)

    seg1 = brownian_bridge(p_curr, t_7, idx_7)
    seg2 = brownian_bridge(t_7, t_14, idx_14 - idx_7)[1:]
    seg3 = brownian_bridge(t_14, t_30, idx_30 - idx_14)[1:]
    forecast_prices = np.concatenate([[p_curr], seg1[1:], seg2, seg3])
    if len(forecast_prices) != n_days:
        forecast_prices = np.interp(np.linspace(0, 1, n_days), np.linspace(0, 1, len(forecast_prices)), forecast_prices)

    t_steps = np.arange(n_days)
    time_factor = np.sqrt(t_steps / max(idx_30, 1))

    bull_spread = max(0.0, t_bull - p_curr)
    bear_spread = max(0.0, p_curr - t_bear)

    bull_curve = np.maximum(p_curr + bull_spread * time_factor, forecast_prices * 1.002)
    bear_curve = np.maximum(p_curr - bear_spread * time_factor, p_curr * 0.6)
    bear_curve = np.minimum(bear_curve, forecast_prices * 0.998)

    milestones = {
        "dates": [future_dates[0], future_dates[idx_7], future_dates[idx_14], future_dates[idx_30]],
        "prices": [round(p_curr, 2), round(forecast_prices[idx_7], 2), round(forecast_prices[idx_14], 2), round(forecast_prices[idx_30], 2)],
        "labels": ["Heute", "Ziel 7T", "Ziel 14T", "Ziel 30T"]
    }

    return future_dates, forecast_prices, bull_curve, bear_curve, milestones

def fetch_fundamental_and_wallstreet(ticker_sym):
    fund = {
        "target_mean": None, "target_high": None, "target_low": None,
        "recommendation": "Keine Einstufung", "forward_pe": "k. A.",
        "currency_symbol": "€" if ticker_sym.endswith(".DE") else "$"
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
        ret_365d = ((current_p - start_p) / start_p) * 100.0 if start_p > 0 else 0.0

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
        bb_lower = round(max(0.1, sma20 - (2.0 * rolling_std_20)), 2)

        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        macd_val = round(float(macd_line.iloc[-1]), 2)
        signal_val = round(float(signal_line.iloc[-1]), 2)
        macd_status = "Bullisch (MACD über Signallinie)" if macd_val > signal_val else "Bärisch (MACD unter Signallinie)"

        prev_close = close.shift(1)
        tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        atr_14 = round(float(tr.tail(14).mean()), 2)

        recent_90 = close.tail(90)
        resistance_level = round(float(recent_90.max()), 2)
        support_level = round(float(recent_90.min()), 2)

        volatility = float(close.pct_change().std() * np.sqrt(252) * 100.0)

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
            "sma20": round(sma20, 2), "sma50": round(sma50, 2), "sma200": round(sma200, 2),
            "rsi_14": round(current_rsi, 1),
            "bb_upper": bb_upper, "bb_lower": bb_lower,
            "macd_status": macd_status, "atr_14": atr_14,
            "resistance_level": resistance_level, "support_level": support_level,
            "volatility_pct": round(volatility, 2),
            "trend_status": trend_status,
            "mc_worst_30d": mc_worst, "mc_median_30d": mc_median, "mc_best_30d": mc_best,
            "data_points": len(close)
        }
        return stats_dict, close
    except Exception:
        return None, None

def parse_target_price_safe(key_name, text, current_price, default_val):
    """Verhindert Abstürze ins Negative: Fängt Prozentangaben ab und rechnet sie in reale Kurse um."""
    pattern = rf'{key_name}\s*[:=]\s*([+-]?[0-9.,]+)'
    m = re.search(pattern, text, re.IGNORECASE)
    if m:
        try:
            val = float(m.group(1).replace(",", "."))
            # Fall 1: KI hat versehentlich Prozent (z. B. 2.5 oder -3.0) statt absolutem Kurs ausgegeben
            if -30.0 <= val <= 50.0 and current_price > 70.0:
                val = current_price * (1.0 + (val / 100.0))
            # Fall 2: Negativer Wert ausgegeben -> Absichern
            if val <= 0:
                val = current_price * 0.95
            return round(val, 2)
        except Exception:
            pass
    return round(default_val, 2)

def run_ai_learning_prediction(client, model_name, ticker_sym, company_name, stats, memory, macro_news=""):
    t = ticker_sym.upper()
    past_stock_memory = memory.get(t, {})
    profile = past_stock_memory.get("learning_profile", {})
    past_predictions = past_stock_memory.get("history", [])

    fund = fetch_fundamental_and_wallstreet(ticker_sym)
    curr_sym = fund.get("currency_symbol", stats.get("currency_symbol", "€"))
    p_curr = stats['current_price']

    accuracy_eval = f"""
AUTONOMES SELBSTLERN-FEEDBACK:
- Durchgeführte Prüfungen: {profile.get('total_evaluations', 0)}
- Richtungstreffer: {profile.get('direction_accuracy_pct', 100)} %
- Mittlere Abweichung: {profile.get('avg_error_pct', 0)} %
- Bias-Dämpfungsfaktor: {profile.get('bias_factor', 1.0)}x
"""

    prompt = f"""
Du bist ein quantitativer Chef-Anlagestratege. Antworte zu 100% AUF DEUTSCH.
Gib NIEMALS Denkprozesse (<think>) aus.

[UNTERNEHMEN: {company_name} ({t})]
- Aktueller realer Börsenkurs: {p_curr} {curr_sym}
- 365-Tage Wertentwicklung: {stats['return_365d_pct']} %
- Trend: {stats['trend_status']} | RSI: {stats['rsi_14']} | MACD: {stats['macd_status']}
- Bollinger-Spanne: {stats['bb_lower']} bis {stats['bb_upper']} {curr_sym}
- Monte-Carlo Korridor: {stats['mc_worst_30d']} bis {stats['mc_best_30d']} {curr_sym}
- Analystenkonsens: {fund.get('target_mean', 'k. A.')} {curr_sym}

[MARKT- & NACHRICHTENLAGE]
{macro_news}

[LERNGEDÄCHTNIS DER KI]
{accuracy_eval}

WICHTIGE ANWEISUNG:
Gib absolute KURSZIELE IN {curr_sym} an (KEINE Prozentzahlen, keine negativen Werte)!
Beispiel bei aktuellem Kurs von {p_curr}: ziel_7d = {round(p_curr * 1.01, 2)}

PROGNOSE_WERTE_START
ziel_7d = [Absoluter Kurs in 7 Tagen in {curr_sym}]
ziel_14d = [Absoluter Kurs in 14 Tagen in {curr_sym}]
ziel_30d = [Hauptziel in 30 Tagen in {curr_sym}]
best_case_30d = [Maximaler Best-Case in 30 Tagen in {curr_sym}]
worst_case_30d = [Absicherungsmarke / Stop-Loss in {curr_sym}]
wahrscheinlichkeit = [Zahl zwischen 65 und 92]
PROGNOSE_WERTE_ENDE

### 1. 🌐 Synthese & Makroumfeld
### 2. 🧠 Indikatoren (MACD, Bollinger, ATR)
### 3. 🎯 Begründung der Meilensteine (Tag 7, 14, 30)
### 4. 🧭 Konkreter 30-Tage-Handelsplan
"""

    full_output = ""
    try:
        res = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": f"Du bist ein quantitativer Börsenanalyst. Antworte auf Deutsch. Berechne absolute Kursziele nahe dem aktuellen Kurs von {p_curr} {curr_sym}."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=1500,
            timeout=25.0
        )
        full_output = res.choices[0].message.content
    except Exception as e:
        full_output = f"⚠️ Fehler bei der Prognose: {e}"

    full_output = re.sub(r'<think>.*?</think>', '', full_output, flags=re.DOTALL).strip()

    # Intelligenter Parser mit Plausibilitätscheck
    b_fac = profile.get("bias_factor", 1.0)
    fb_trend = 1.02 if "Aufwärtstrend" in stats['trend_status'] else 0.98
    fb_30 = round(p_curr * (1.0 + (fb_trend - 1.0) * b_fac), 2)
    fb_7 = round(p_curr + (fb_30 - p_curr) * 0.35, 2)
    fb_14 = round(p_curr + (fb_30 - p_curr) * 0.65, 2)

    t_7 = parse_target_price_safe("ziel_7d", full_output, p_curr, fb_7)
    t_14 = parse_target_price_safe("ziel_14d", full_output, p_curr, fb_14)
    t_30 = parse_target_price_safe("ziel_30d", full_output, p_curr, fb_30)
    t_bull = parse_target_price_safe("best_case_30d", full_output, p_curr, stats['mc_best_30d'])
    t_bear = parse_target_price_safe("worst_case_30d", full_output, p_curr, stats['mc_worst_30d'])

    # Harte Obergrenzen/Untergrenzen gegen Fehlausgaben
    t_7 = max(p_curr * 0.7, min(p_curr * 1.4, t_7))
    t_14 = max(p_curr * 0.7, min(p_curr * 1.4, t_14))
    t_30 = max(p_curr * 0.7, min(p_curr * 1.4, t_30))
    t_bear = max(p_curr * 0.65, min(p_curr, t_bear))
    t_bull = max(p_curr, min(p_curr * 1.5, t_bull))

    prob_m = re.search(r'wahrscheinlichkeit\s*[:=]\s*([0-9]+)', full_output)
    prob_str = f"{prob_m.group(1)} %" if prob_m else "78 %"

    targets_dict = {
        "p_curr": p_curr, "currency_symbol": curr_sym,
        "t_7": t_7, "t_14": t_14, "t_30": t_30,
        "t_bull": t_bull, "t_bear": t_bear, "prob": prob_str
    }

    clean_report = re.sub(r'PROGNOSE_WERTE_START.*?PROGNOSE_WERTE_ENDE', '', full_output, flags=re.DOTALL).strip()

    # Speichere die Prognose im Wissensspeicher
    if t not in memory:
        memory[t] = {"name": company_name, "history": [], "learning_profile": {}}

    new_entry = {
        "timestamp": time.time(),
        "date": datetime.now().strftime("%d.%m.%Y %H:%M Uhr"),
        "price": p_curr,
        "currency": curr_sym,
        "target_7d": t_7,
        "target_14d": t_14,
        "target_30d": t_30,
        "trend": stats['trend_status'],
        "analysis_summary": clean_report[:280] + "..."
    }

    memory[t]["history"].append(new_entry)
    if len(memory[t]["history"]) > 25:
        memory[t]["history"] = memory[t]["history"][-25:]
        
    save_memory(memory)
    return clean_report, targets_dict
