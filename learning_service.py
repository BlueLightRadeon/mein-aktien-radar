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

# Absoluter Pfad im App-Verzeichnis
MEMORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stock_memory.json")

def sync_to_github(memory_data):
    """Sichert die JSON-Datei automatisch im GitHub-Repo, falls Token vorhanden."""
    token = st.secrets.get("GITHUB_TOKEN", "")
    repo = st.secrets.get("GITHUB_REPO", "")  # Format: "benutzername/repo-name"
    if not token or not repo:
        return
    try:
        url = f"https://api.github.com/repos/{repo}/contents/stock_memory.json"
        headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "Streamlit-Stock-Radar"
        }
        
        # SHA des bestehenden Files abrufen
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
    """Berechnet aktienspezifische Monte-Carlo-Pfade basierend auf echtem Trend & Volatilität."""
    try:
        dt = 1.0 / 252.0
        sigma = max(volatility_annual / 100.0, 0.12)
        
        # Trend-abhängiger dynamischer Drift statt statischem Festwert
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
        else:
            close = data["Close"]
            
        close = close.dropna()
        if len(close) < 30:
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

        mc_worst, mc_median, mc_best = run_monte_carlo(current_p, volatility, ret_365d, trend_status)
        currency_sym = "€" if ticker_sym.endswith(".DE") else "$"

        stats_dict = {
            "current_price": round(current_p, 2),
            "currency_symbol": currency_sym,
            "return_365d_pct": round(ret_365d, 2),
            "sma20": round(sma20, 2),
            "sma50": round(sma50, 2),
            "sma200": round(sma200, 2),
            "rsi_14": round(current_rsi, 1),
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
        f"[UNTERNEHMEN: {company_name} ({t})]",
        f"WÄHRUNG: {curr_sym}",
        "",
        "[1. TECHNISCHE INDIKATOREN & ECHTER TREND]",
        f"- Aktueller Börsenkurs: {stats['current_price']} {curr_sym}",
        f"- Bisherige 365-Tage Wertentwicklung: {stats['return_365d_pct']} %",
        f"- Aktueller Trend-Status: {stats['trend_status']}",
        f"- Gleitende Durchschnitte: SMA20: {stats['sma20']} | SMA50: {stats['sma50']} | SMA200: {stats['sma200']}",
        f"- RSI (14 Tage): {stats['rsi_14']}",
        f"- Volatilität: {stats['volatility_pct']} %",
        f"- Letzte 5 Tage: {stats['last_5_days']}",
        "",
        "[2. STATISTISCHE MONTE-CARLO-REFERENZ (1.000 Pfade)]",
        f"- Statistischer Median in 30 Tagen: {stats['mc_median_30d']} {curr_sym}",
        f"- Statistischer 95%-Worst-Case: {stats['mc_worst_30d']} {curr_sym}",
        f"- Statistischer 95%-Best-Case: {stats['mc_best_30d']} {curr_sym}",
        "",
        "[3. WALL-STREET-KONSENS]",
        f"- Offizielles Analystenziel: {fund['target_mean'] if fund['target_mean'] else 'Kein Konsens'} {curr_sym}",
        f"- Spanne: {fund['target_low']} bis {fund['target_high']} {curr_sym}",
        f"- Einstufung: {fund['recommendation']}",
        "",
        "[4. NACHRICHTENLAGE]",
        macro_news,
        specific_news,
        "",
        "[5. LERNGEDÄCHTNIS]",
        accuracy_eval,
        "",
        "STRIKTE ANWEISUNG FÜR DIE PROGNOSE:",
        "Berechne UNABHÄNGIGE, REALISTISCHE Zahlen, die EXAKT zu den Indikatoren DIESER EINEN AKTIE passen!",
        "- Wenn die Aktie in einem Abwärtstrend ist, MÜSSEN die Kursziele sinken oder vorsichtig sein.",
        "- Wenn die Aktie in einem starken Aufwärtstrend ist, MÜSSEN die Kursziele steigen.",
        "- Verwende NIEMALS starre Standardprozente (+1%, +4%, etc.) für verschiedene Aktien!",
        "",
        "Gliedere deine Antwort zwingend in zwei Teile:",
        "",
        "TEIL 1: EXAKTE ZAHLEN",
        "Gib zuerst exakt folgenden Block mit deinen berechneten Werten aus:",
        "PROGNOSE_WERTE_START",
        "ziel_7d = [Zahl für 7 Tage]",
        "ziel_30d = [Zahl für 30 Tage]",
        "ziel_90d = [Zahl für 90 Tage]",
        "best_case_30d = [Optimistischer Zielwert 30T]",
        "worst_case_30d = [Stop-Loss / Absicherungswert 30T]",
        "wahrscheinlichkeit = [Zahl zwischen 50 und 92]",
        "PROGNOSE_WERTE_ENDE",
        "",
        "TEIL 2: AUSFÜHRLICHER DEUTSCHER ANALYSEBERICHT",
        "### 1. 🌐 Synthese: Weltwirtschaft & Wall-Street-Konsens",
        "(Vergleiche das offizielle Analystenziel der Wall Street mit deiner Einschätzung)",
        "",
        "### 2. 🧠 Erkenntnisse aus 365-Tage-Historie & Trend-Dynamik",
        "(Erkläre das Zusammenspiel aus RSI, Gleitenden Durchschnitten und dem tatsächlichen Trend)",
        "",
        "### 3. 🎯 Begründung der Kursprognose",
        "(Konkrete Begründung deiner berechneten Ziele für 7, 30 und 90 Tage)",
        "",
        "### 4. 🧭 Konkreter Handlungsplan",
        "(Klare Handlungsanweisung: Einstiegs-Limit, Absichern mit Stop-Loss oder Abwarten)"
    ]
    prompt = "\n".join(prompt_lines)

    full_output = ""
    for attempt in range(2):
        try:
            res = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "Du bist ein führender deutscher Börsenanalyst. Antworte ausschließlich auf Deutsch. Gib niemals englische Denkprozesse (<think>) aus."},
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
    
    # Trend-basierte Fallbacks, falls Regex fehlschlägt
    fallback_trend_factor = 1.03 if "Aufwärtstrend" in stats['trend_status'] else (0.97 if "Abwärtstrend" in stats['trend_status'] else 1.0)
    fb_7 = round(p_curr * (1.0 + (fallback_trend_factor - 1.0) * 0.3), 2)
    fb_30 = round(p_curr * fallback_trend_factor, 2)
    fb_90 = round(p_curr * (1.0 + (fallback_trend_factor - 1.0) * 2.0), 2)

    targets_dict = {
        "p_curr": p_curr,
        "currency_symbol": curr_sym,
        "t_7": parse_num_tolerant("ziel_7d", full_output, fb_7),
        "t_30": parse_num_tolerant("ziel_30d", full_output, fb_30),
        "t_90": parse_num_tolerant("ziel_90d", full_output, fb_90),
        "t_bull": parse_num_tolerant("best_case_30d", full_output, stats['mc_best_30d']),
        "t_bear": parse_num_tolerant("worst_case_30d", full_output, stats['mc_worst_30d']),
        "prob": f"{int(parse_num_tolerant('wahrscheinlichkeit', full_output, 76))} %"
    }

    clean_report = re.sub(r'PROGNOSE_WERTE_START.*?PROGNOSE_WERTE_ENDE', '', full_output, flags=re.DOTALL).strip()
    clean_report = re.sub(r'^TEIL\s*2:?[^\n]*\n', '', clean_report, flags=re.IGNORECASE).strip()

    if t not in memory:
        memory[t] = {"name": company_name, "history": []}

    memory[t]["history"].append({
        "date": datetime.now().strftime("%d.%m.%Y %H:%M Uhr"),
        "price": stats['current_price'],
        "currency": curr_sym,
        "target_30d": targets_dict["t_30"],
        "target_90d": targets_dict["t_90"],
        "rsi": stats['rsi_14'],
        "trend": stats['trend_status'],
        "analysis_summary": clean_report[:280] + "..."
    })
    
    if len(memory[t]["history"]) > 15:
        memory[t]["history"] = memory[t]["history"][-15:]
        
    save_memory(memory)

    return clean_report, targets_dict
