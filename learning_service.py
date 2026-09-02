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

    accuracy_eval = "ERSTMALIGE ANALYSE: Noch kein Vorwissen für diesen Ticker gespeichert."
    if past_predictions:
        accuracy_eval = "HISTORISCHE ABWEICHUNGEN AUS DEINEM SPEICHER:\n"
        for entry in past_predictions[-3:]:
            prev_p = entry.get("price", 0.0)
            target = entry.get("target_30d", prev_p)
            date_str = entry.get("date", "Unbekannt")
            if target > 0:
                diff_pct = abs((stats["current_price"] - target) / target) * 100.0
                accuracy_eval += f"- {date_str}: Kurs damals {prev_p:.2f} € -> Ziel war {target:.2f} €. Realkurs heute: {stats['current_price']:.2f} € (Differenz: {diff_pct:.1f}%)\n"

    specific_news = fetch_company_specific_news(ticker_sym)

    prompt_lines = [
        "Du musst ausnahmslos auf DEUTSCH antworten. Kein einziges Wort auf Englisch!",
        "",
        f"[UNTERNEHMEN: {company_name} ({t})]",
        "[AKTUELLE WELTMARKTLAGE]",
        macro_news,
        "",
        "[AKTUELLE MELDUNGEN ZU DIESER AKTIE]",
        specific_news,
        "",
        "[365-TAGE DATEN & KENNZAHLEN]",
        f"- Aktueller Börsenkurs: {stats['current_price']} €",
        f"- 365-Tage Performance: {stats['return_365d_pct']} %",
        f"- 52-Wochen Bandbreite: Tief {stats['low_365d']} € bis Hoch {stats['high_365d']} €",
        f"- Gleitende Durchschnitte: SMA20: {stats['sma20']} € | SMA50: {stats['sma50']} € | SMA200: {stats['sma200']} €",
        f"- RSI (14 Tage): {stats['rsi_14']}",
        f"- Schwankungsbreite (Volatilität): {stats['volatility_pct']} %",
        f"- Trendrichtung: {stats['trend_status']}",
        f"- Kurse der letzten 5 Tage: {stats['last_5_days']}",
        "",
        "[LERNSTAND & FEHLERKORREKTUR]",
        accuracy_eval,
        "",
        "Gliedere deine Antwort zwingend in zwei Teile:",
        "",
        "TEIL 1: EXAKTE ZAHLEN",
        "Gib zuerst exakt folgenden Block aus:",
        "PROGNOSE_WERTE_START",
        f"ziel_7d = {round(stats['current_price'] * 1.01, 2)}",
        f"ziel_30d = {round(stats['current_price'] * 1.04, 2)}",
        f"ziel_90d = {round(stats['current_price'] * 1.09, 2)}",
        f"best_case_30d = {round(stats['current_price'] * 1.08, 2)}",
        f"worst_case_30d = {round(stats['current_price'] * 0.94, 2)}",
        "wahrscheinlichkeit = 75",
        "PROGNOSE_WERTE_ENDE",
        "(Passe die Zahlenwerte hinter dem Gleichheitszeichen exakt an deine tatsächliche Berechnung an!)",
        "",
        "TEIL 2: AUSFÜHRLICHER DEUTSCHER ANALYSEBERICHT",
        "### 1. Marktlage & Nachrichten-Synthese",
        "(Analysiere auf Deutsch, wie Weltgeschehen, Zinsen und News die Aktie beeinflussen)",
        "",
        "### 2. Erkenntnisse aus den 365-Tage-Mustern",
        "(Welche Chartmuster und Unterstützungen hat die KI gelernt?)",
        "",
        "### 3. Begründung der Kursprognose",
        "(Begründung für 7, 30 und 90 Tage sowie Best-Case und Absicherungsmarke)",
        "",
        "### 4. Konkrete Handlungsanweisung",
        "(Klare deutsche Anweisung: Kaufen, Halten, Zukauf-Limit setzen)"
    ]
    prompt = "\n".join(prompt_lines)

    res = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "Du bist ein führender deutscher Börsenanalyst. Antworte ausschließlich auf Deutsch. Schreibe unter keinen Umständen englische Denkprozesse oder Wörter in die Ausgabe."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=1400
    )
    full_output = res.choices[0].message.content

    # 1. Entfernt interne <think>...</think> Abschnitte der KI
    full_output = re.sub(r'<think>.*?</think>', '', full_output, flags=re.DOTALL).strip()
    full_output = re.sub(r'^.*?Here\'s a thinking process.*?\n\n', '', full_output, flags=re.DOTALL | re.IGNORECASE).strip()

    p_curr = stats['current_price']
    targets_dict = {
        "p_curr": p_curr,
        "t_7": round(p_curr * 1.01, 2),
        "t_30": round(p_curr * 1.04, 2),
        "t_90": round(p_curr * 1.08, 2),
        "t_bull": round(p_curr * 1.07, 2),
        "t_bear": round(p_curr * 0.95, 2),
        "prob": "75 %"
    }

    # 2. Zahlenblock parsen
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

    # 3. Bereinigung für die saubere deutsche Anzeige
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
