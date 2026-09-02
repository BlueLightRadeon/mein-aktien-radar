import os
import json
import re
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

MEMORY_FILE = "stock_memory.json"

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

def run_monte_carlo(current_price, volatility_annual, days=30, simulations=1000):
    """Berechnet 1.000 Monte-Carlo-Zukunftspfade auf Basis statistischer Drift & Volatilitaet."""
    try:
        dt = 1.0 / 252.0
        sigma = max(volatility_annual / 100.0, 0.10)
        mu = 0.05  # Konservativer Basis-Drift p.a.
        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * np.random.normal(size=(days, simulations))
        daily_multipliers = np.exp(drift + diffusion)
        price_paths = current_price * np.cumprod(daily_multipliers, axis=0)
        final_prices = price_paths[-1]

        p5 = float(np.percentile(final_prices, 5))    # 95% Absicherungs-Level
        p50 = float(np.percentile(final_prices, 50))  # Statistischer Median
        p95 = float(np.percentile(final_prices, 95))  # 95% Best-Case
        return round(p5, 2), round(p50, 2), round(p95, 2)
    except Exception:
        return round(current_price * 0.93, 2), round(current_price * 1.03, 2), round(current_price * 1.10, 2)

def fetch_fundamental_and_wallstreet(ticker_sym):
    """Zieht offizielle Wall-Street-Kursziele und Quartalstermine ab."""
    fund = {
        "target_mean": None,
        "target_high": None,
        "target_low": None,
        "recommendation": "Keine Einstufung",
        "forward_pe": "k. A.",
        "peg_ratio": "k. A.",
        "next_earnings": "Nicht terminiert"
    }
    try:
        t = yf.Ticker(ticker_sym)
        inf = getattr(t, "info", {})
        if inf:
            fund["target_mean"] = inf.get("targetMeanPrice")
            fund["target_high"] = inf.get("targetHighPrice")
            fund["target_low"] = inf.get("targetLowPrice")
            fund["recommendation"] = inf.get("recommendationKey", "Halten").upper()
            if inf.get("forwardPE"):
                fund["forward_pe"] = f"{inf.get('forwardPE'):.1f}"
            if inf.get("pegRatio"):
                fund["peg_ratio"] = f"{inf.get('pegRatio'):.2f}"

        # Terminkalender pruefen
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
            volume = data["Volume"][ticker_sym] if ticker_sym in data["Volume"] else data["Volume"].iloc[:, 0]
        else:
            close = data["Close"]
            volume = data["Volume"]
            
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

        avg_vol_20 = float(volume.tail(20).mean()) if not volume.empty else 1.0
        last_vol = float(volume.iloc[-1]) if not volume.empty else 1.0
        vol_ratio = round(last_vol / avg_vol_20, 2) if avg_vol_20 > 0 else 1.0

        recent_trend = [round(float(x), 2) for x in close.tail(5).tolist()]
        trend_status = "Starker Aufwärtstrend" if current_p > sma20 > sma50 else ("Aufwärtstrend" if current_p > sma50 else ("Abwärtstrend" if current_p < sma50 else "Seitwärtsphase"))

        # Monte-Carlo Leitplanken berechnen
        mc_worst, mc_median, mc_best = run_monte_carlo(current_p, volatility, days=30, simulations=1000)

        stats_dict = {
            "current_price": round(current_p, 2),
            "return_365d_pct": round(ret_365d, 2),
            "sma20": round(sma20, 2),
            "sma50": round(sma50, 2),
            "sma200": round(sma200, 2),
            "rsi_14": round(current_rsi, 1),
            "high_365d": round(high_365, 2),
            "low_365d": round(low_365, 2),
            "volatility_pct": round(volatility, 2),
            "volume_ratio": vol_ratio,
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

def run_ai_learning_prediction(client, model_name, ticker_sym, company_name, stats, memory, macro_news=""):
    t = ticker_sym.upper()
    past_stock_memory = memory.get(t, {})
    past_predictions = past_stock_memory.get("history", [])

    # Fundamentaldaten & Wall-Street-Konsens holen
    fund = fetch_fundamental_and_wallstreet(ticker_sym)

    # 1. Mathematische Fehleranalyse & Bias-Korrektur
    accuracy_eval = "ERSTMALIGE ANALYSE: Noch kein Vorwissen für diesen Ticker gespeichert."
    bias_correction_instruction = ""
    if past_predictions:
        errors = []
        accuracy_eval = "HISTORISCHE ABWEICHUNGEN AUS DEINEM SPEICHER:\n"
        for entry in past_predictions[-4:]:
            prev_p = entry.get("price", 0.0)
            target = entry.get("target_30d", prev_p)
            date_str = entry.get("date", "Unbekannt")
            if target > 0:
                diff_pct = ((target - stats["current_price"]) / stats["current_price"]) * 100.0
                errors.append(diff_pct)
                accuracy_eval += f"- {date_str}: Prognostiziertes 30T-Ziel war {target:.2f} € -> Realkurs heute: {stats['current_price']:.2f} € (Abweichung: {diff_pct:+.1f}%)\n"
        
        if errors:
            avg_bias = sum(errors) / len(errors)
            if abs(avg_bias) > 2.0:
                bias_correction_instruction = f"WICHTIGE KALIBRIERUNG: Du warst in früheren Prognosen im Schnitt um {avg_bias:+.1f} % zu optimistisch/pessimistisch. Passe dein neues 30-Tage-Ziel rechnerisch um genau diesen Faktor an, um Übertreibungen zu eliminieren!"

    specific_news = fetch_company_specific_news(ticker_sym)

    prompt_lines = [
        "Du bist ein quantitativer Chef-Anlagestratege. Antworte zu 100 % AUF DEUTSCH.",
        "Verwende niemals englische Denkprozesse, Floskeln oder englische Abschnitte!",
        "",
        f"[UNTERNEHMEN: {company_name} ({t})]",
        "",
        "[1. INSTITUTIONELLE FUNDAMENTALDATEN & WALL-STREET-KONSENS]",
        f"- Offizielles Analysten-Konsenskursziel: {fund['target_mean'] if fund['target_mean'] else 'Kein Konsens'} €",
        f"- Analysten-Spanne: Niedrig {fund['target_low']} € | Hoch {fund['target_high']} €",
        f"- Wall-Street Gesamturteil: {fund['recommendation']}",
        f"- Vorwärts-KGV: {fund['forward_pe']} | PEG-Ratio: {fund['peg_ratio']}",
        f"- Nächste Quartalszahlen: {fund['next_earnings']}",
        "",
        "[2. STATISTISCHE MONTE-CARLO-LEITPLANKEN (1.000 SIMULATIONEN)]",
        f"- Statistischer Median in 30 Tagen: {stats['mc_median_30d']} €",
        f"- Statistischer 95%-Worst-Case: {stats['mc_worst_30d']} €",
        f"- Statistischer 95%-Best-Case: {stats['mc_best_30d']} €",
        "",
        "[3. 365-TAGE CHARTTECHNIK]",
        f"- Aktueller Börsenkurs: {stats['current_price']} €",
        f"- 365-Tage Wertentwicklung: {stats['return_365d_pct']} %",
        f"- 52-Wochen Tief / Hoch: {stats['low_365d']} € / {stats['high_365d']} €",
        f"- Gleitende Durchschnitte: SMA20: {stats['sma20']} € | SMA50: {stats['sma50']} € | SMA200: {stats['sma200']} €",
        f"- RSI (14 Tage): {stats['rsi_14']} (unter 30 = überverkauft, über 70 = überkauft)",
        f"- Volatilität: {stats['volatility_pct']} % | Trend: {stats['trend_status']}",
        f"- Letzte 5 Tage: {stats['last_5_days']}",
        "",
        "[4. WELTLAGE & UNTERNEHMENS-NEWS]",
        macro_news,
        specific_news,
        "",
        "[5. LERNGEDÄCHTNIS & DYNAMISCHE KALIBRIERUNG]",
        accuracy_eval,
        bias_correction_instruction,
        "",
        "Gliedere deine Antwort zwingend in zwei Teile:",
        "",
        "TEIL 1: EXAKTE ZAHLEN",
        "Gib zuerst exakt folgenden Block aus:",
        "PROGNOSE_WERTE_START",
        f"ziel_7d = {round(stats['current_price'] * 1.01, 2)}",
        f"ziel_30d = {stats['mc_median_30d']}",
        f"ziel_90d = {round(stats['mc_median_30d'] * 1.04, 2)}",
        f"best_case_30d = {stats['mc_best_30d']}",
        f"worst_case_30d = {stats['mc_worst_30d']}",
        "wahrscheinlichkeit = 78",
        "PROGNOSE_WERTE_ENDE",
        "(Passe die Zahlenwerte hinter dem Gleichheitszeichen exakt an deine tatsächliche Berechnung an!)",
        "",
        "TEIL 2: AUSFÜHRLICHER DEUTSCHER ANALYSEBERICHT",
        "### 1. 🌐 Synthese: Weltwirtschaft & Wall-Street-Konsens",
        "(Vergleiche das offizielle Analystenziel der Wall Street mit deiner Einschätzung und den Makrozinsen)",
        "",
        "### 2. 🧠 Erkenntnisse aus der 365-Tage-Historie & Monte-Carlo-Wahrscheinlichkeiten",
        "(Erkläre das Zusammenspiel aus RSI, Gleitenden Durchschnitten und den berechneten 1.000 Zukunftspfaden)",
        "",
        "### 3. 🎯 Begründung der Kursprognose & Fehler-Korrektur",
        "(Begründung für 7, 30 und 90 Tage sowie Erläuterung der Best-Case- und Absicherungsmarke)",
        "",
        "### 4. 🧭 Konkreter Handlungsplan",
        "(Klare deutsche Anweisung für Anleger: Einstiegs-Limit, Stop-Loss setzen oder abwarten)"
    ]
    prompt = "\n".join(prompt_lines)

    res = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "Du bist ein führender deutscher quantitativer Börsenanalyst. Antworte ausschließlich auf Deutsch. Gib unter keinen Umständen englische Denkprozesse (<think>) aus."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=1500
    )
    full_output = res.choices[0].message.content

    # Filtert unerwünschte englische Thinking-Tags rigoros heraus
    full_output = re.sub(r'<think>.*?</think>', '', full_output, flags=re.DOTALL).strip()
    full_output = re.sub(r'^.*?Here\'s a thinking process.*?\n\n', '', full_output, flags=re.DOTALL | re.IGNORECASE).strip()

    p_curr = stats['current_price']
    targets_dict = {
        "p_curr": p_curr,
        "t_7": round(p_curr * 1.01, 2),
        "t_30": stats['mc_median_30d'],
        "t_90": round(stats['mc_median_30d'] * 1.04, 2),
        "t_bull": stats['mc_best_30d'],
        "t_bear": stats['mc_worst_30d'],
        "prob": "78 %"
    }

    val_block = re.search(r'PROGNOSE_WERTE_START(.*?)PROGNOSE_WERTE_ENDE', full_output, re.DOTALL)
    if val_block:
        lines = val_block.group(1).split("\n")
        for line in lines:
            if "=" in line:
                key, val = line.split("=", 1)
                k = key.strip()
                v_num_match = re.search(r'([\d.,]+)', val.strip())
                if v_num_match:
                    num_val = float(v_num_match.group(1).replace(",", "."))
                    if k == "ziel_7d": targets_dict["t_7"] = num_val
                    elif k == "ziel_30d": targets_dict["t_30"] = num_val
                    elif k == "ziel_90d": targets_dict["t_90"] = num_val
                    elif k == "best_case_30d": targets_dict["t_bull"] = num_val
                    elif k == "worst_case_30d": targets_dict["t_bear"] = num_val
                    elif k == "wahrscheinlichkeit": targets_dict["prob"] = f"{int(num_val)} %"

    clean_report = re.sub(r'PROGNOSE_WERTE_START.*?PROGNOSE_WERTE_ENDE', '', full_output, flags=re.DOTALL).strip()
    clean_report = re.sub(r'^TEIL\s*2:?[^\n]*\n', '', clean_report, flags=re.IGNORECASE).strip()

    if t not in memory:
        memory[t] = {"name": company_name, "history": []}

    memory[t]["history"].append({
        "date": datetime.now().strftime("%d.%m.%Y %H:%M Uhr"),
        "price": stats['current_price'],
        "target_30d": targets_dict["t_30"],
        "rsi": stats['rsi_14'],
        "analysis_summary": clean_report[:280] + "..."
    })
    save_memory(memory)

    return clean_report, targets_dict
