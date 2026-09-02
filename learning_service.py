import os
import json
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
            return None
        
        if isinstance(data.columns, pd.MultiIndex):
            close = data["Close"][ticker_sym] if ticker_sym in data["Close"] else data["Close"].iloc[:, 0]
        else:
            close = data["Close"]
        close = close.dropna()
        if len(close) < 30:
            return None

        current_p = float(close.iloc[-1])
        start_p = float(close.iloc[0])
        ret_365d = ((current_p - start_p) / start_p) * 100.0

        sma20 = float(close.tail(20).mean())
        sma50 = float(close.tail(50).mean())
        sma200 = float(close.tail(200).mean()) if len(close) >= 200 else float(close.mean())

        # 14-Tage RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi_series = 100 - (100 / (1 + rs))
        current_rsi = float(rsi_series.dropna().iloc[-1]) if not rsi_series.dropna().empty else 50.0

        high_365 = float(close.max())
        low_365 = float(close.min())
        volatility = float(close.pct_change().std() * np.sqrt(252) * 100.0)

        # Letzte 5 Tage
        recent_trend = [round(float(x), 2) for x in close.tail(5).tolist()]

        return {
            "current_price": round(current_p, 2),
            "return_365d_pct": round(ret_365d, 2),
            "sma20": round(sma20, 2),
            "sma50": round(sma50, 2),
            "sma200": round(sma200, 2),
            "rsi_14": round(current_rsi, 1),
            "high_365d": round(high_365, 2),
            "low_365d": round(low_365, 2),
            "volatility_pct": round(volatility, 2),
            "last_5_days": recent_trend,
            "data_points": len(close)
        }
    except Exception:
        return None

def run_ai_learning_prediction(client, model_name, ticker_sym, company_name, stats, memory):
    t = ticker_sym.upper()
    past_stock_memory = memory.get(t, {})
    past_predictions = past_stock_memory.get("history", [])

    history_context = ""
    if past_predictions:
        history_context = "FRÜHERE PROGNOSEN & DEIN LERNSTAND:\n"
        for entry in past_predictions[-3:]:
            history_context += f"- Stand {entry.get('date')}: Damaliger Kurs war {entry.get('price')} €. Deine 30-Tage-Prognose war {entry.get('target_30d')} €.\n"
    else:
        history_context = "Es liegen noch keine früheren Vorhersagen in deiner Wissensdatenbank vor. Dies ist dein Basis-Lernpunkt."

    prompt = f"""
Du bist ein quantitatives KI-Prognose-System. Analysiere das vollständige 365-Tage-Profil von {company_name} ({t}).
Lerne aus den historischen Kursen, den technischen Indikatoren und deinen früheren Vorhersagen.

TECHNISCHE 365-TAGE-HISTORIE:
- Aktueller Börsenkurs: {stats['current_price']}
- 365-Tage-Performance: {stats['return_365d_pct']} %
- 52-Wochen-Hoch: {stats['high_365d']} | 52-Wochen-Tief: {stats['low_365d']}
- Gleitende Durchschnitte: SMA20: {stats['sma20']} | SMA50: {stats['sma50']} | SMA200: {stats['sma200']}
- RSI (14 Tage): {stats['rsi_14']}
- Annualisierte Volatilität: {stats['volatility_pct']} %
- Letzte 5 Schlusskurse: {stats['last_5_days']}

{history_context}

AUFGABE:
Formuliere deine Antwort auf Deutsch strukturiert wie folgt:

### 1. 🧠 Was die KI aus den 365 Tagen gelernt hat
(Analysiere Trendmuster, Zyklus-Verhalten und Unterstützungszonen)

### 2. 🎯 Konkrete Kursprognose
- **Ziel in 7 Tagen:** [Zahl] (Prozentuale Veränderung)
- **Ziel in 30 Tagen:** [Zahl] (Prozentuale Veränderung)
- **Ziel in 90 Tagen:** [Zahl] (Prozentuale Veränderung)
- **Konfidenz:** [Hoch / Mittel / Spekulativ]

### 3. 🛡️ Risikofaktoren & Validierung
(Wann bricht das gelernte Muster?)
"""

    res = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "Du bist eine lernende quantitative KI für Finanzprognosen."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=900
    )
    analysis_text = res.choices[0].message.content

    # Vorhersage im Gedächtnis protokollieren
    import re
    t_30 = stats['current_price']
    match_30 = re.search(r'Ziel in 30 Tagen:[^\d]*([\d.,]+)', analysis_text)
    if match_30:
        try:
            t_30 = float(match_30.group(1).replace(",", "."))
        except Exception:
            pass

    if t not in memory:
        memory[t] = {"name": company_name, "history": []}

    memory[t]["history"].append({
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "price": stats['current_price'],
        "target_30d": t_30,
        "rsi": stats['rsi_14'],
        "analysis_summary": analysis_text[:250] + "..."
    })
    save_memory(memory)

    return analysis_text
