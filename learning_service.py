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
            return None
        
        if isinstance(data.columns, pd.MultiIndex):
            close = data["Close"][ticker_sym] if ticker_sym in data["Close"] else data["Close"].iloc[:, 0]
            volume = data["Volume"][ticker_sym] if ticker_sym in data["Volume"] else data["Volume"].iloc[:, 0]
        else:
            close = data["Close"]
            volume = data["Volume"]
            
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

        # Volumen-Trend
        avg_vol_20 = float(volume.tail(20).mean()) if not volume.empty else 1.0
        last_vol = float(volume.iloc[-1]) if not volume.empty else 1.0
        vol_ratio = round(last_vol / avg_vol_20, 2) if avg_vol_20 > 0 else 1.0

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
            "volume_ratio": vol_ratio,
            "last_5_days": recent_trend,
            "data_points": len(close)
        }
    except Exception:
        return None

def fetch_company_specific_news(ticker_sym):
    """Zieht aktuelle Unternehmensschlagzeilen direkt über yfinance."""
    try:
        t = yf.Ticker(ticker_sym)
        news_items = getattr(t, "news", [])
        headlines = []
        if news_items:
            for item in news_items[:4]:
                title = item.get("title", "")
                if title:
                    headlines.append(f"• {title}")
        return "\n".join(headlines) if headlines else "Keine spezifischen Ticker-Meldungen abrufbar."
    except Exception:
        return "Keine spezifischen Ticker-Meldungen abrufbar."

def run_ai_learning_prediction(client, model_name, ticker_sym, company_name, stats, memory, macro_news=""):
    t = ticker_sym.upper()
    past_stock_memory = memory.get(t, {})
    past_predictions = past_stock_memory.get("history", [])

    # Selbstkorrektur / Feedback-Schleife aus früheren Prognosen
    accuracy_eval = "ERST-ANALYSE: Noch keine historischen Vorhersagen im Speicher."
    if past_predictions:
        accuracy_eval = "HISTORISCHE LERN-BILANZ & FEHLERKORREKTUR:\n"
        for idx, entry in enumerate(past_predictions[-3:]):
            prev_price = entry.get("price", 0.0)
            target = entry.get("target_30d", prev_price)
            date_str = entry.get("date", "Unbekannt")
            # Wie weit wich die damalige Prognose vom jetzigen Realkurs ab?
            diff_pct = ((stats["current_price"] - target) / target) * 100.0 if target > 0 else 0.0
            accuracy_eval += f"- Prognose vom {date_str} (Kurs damals: {prev_price} €, 30T-Ziel: {target} €) -> Tatsächlicher Kurs heute: {stats['current_price']} € (Abweichung: {diff_pct:+.1f}%).\n"
        accuracy_eval += "LERN-AUFTRAG: Berücksichtige deine bisherigen Abweichungen. Passe deine Konfidenz und dein Kurspotenzial entsprechend an!"

    # Spezifische Unternehmensnachrichten abrufen
    specific_news = fetch_company_specific_news(ticker_sym)

    prompt = f"""
Du bist ein quantitatives KI-Prognose-System auf Hedgefonds-Niveau. 
Verbinde die quantitativen 365-Tage-Chartmuster mit dem aktuellen Weltgeschehen und den neuesten Unternehmensnachrichten von {company_name} ({t}), um eine fundierte Kursprognose zu erstellen.

[1. AKTUELLE NACHRICHTENLAGE & WELTGESCHEHEN]
Globale Makro-Trends:
{macro_news}

Spezifische Unternehmens-News zu {company_name}:
{specific_news}

[2. TECHNISCHE 365-TAGE-HISTORIE & KENNZAHLEN]
- Aktueller Börsenkurs: {stats['current_price']} €
- 365-Tage Gesamt-Rendite: {stats['return_365d_pct']} %
- 52-Wochen Spanne: Tief {stats['low_365d']} € | Hoch {stats['high_365d']} €
- Gleitende Durchschnitte: SMA20: {stats['sma20']} | SMA50: {stats['sma50']} | SMA200: {stats['sma200']}
- RSI (14 Tage Momentum): {stats['rsi_14']}
- Annualisierte Volatilität: {stats['volatility_pct']} %
- Letzte 5 Handelstage: {stats['last_5_days']}

[3. DEIN LERNGEDÄCHTNIS]
{accuracy_eval}

AUFGABE:
Analysiere objektiv: Stützen die Nachrichten den Chart-Trend oder widersprechen sie ihm?
Gliedere deine Antwort auf Deutsch strukturiert in diese 4 Abschnitte:

### 1. 🌐 Synthese: Weltgeschehen vs. Charttechnik
(Erläutere in 3-4 Sätzen, wie Notenbankzinsen, geopolitische Lage und Branchennachrichten auf den Kurs wirken)

### 2. 🧠 Erkenntnisse aus der 365-Tage-Historie & Fehlern
(Welche Zyklen und Kursmuster wurden gelernt? Welche Korrekturen wurden aus früheren Fehleinschätzungen gezogen?)

### 3. 🎯 Multi-Szenario Kursprognose
- **Ziel in 7 Tagen:** [Zahl in €] ([+/- %])
- **Ziel in 30 Tagen (Basis-Szenario):** [Zahl in €] ([+/- %])
- **Ziel in 90 Tagen:** [Zahl in €] ([+/- %])
- **🟢 Bullish Best-Case (30T):** [Zahl in €]
- **🔴 Bearish Worst-Case (30T / Stop-Loss):** [Zahl in €]
- **Prognose-Wahrscheinlichkeit:** [z. B. 75%]

### 4. 🧭 Konkrete Handlungsanweisung
(Klare Empfehlung: Jetzt Limit-Order setzen, Einstieg abwarten oder bestehende Gewinne sichern)
"""

    res = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "Du bist ein führender quantitativer Analyst. Antworte strukturiert und fundiert auf Deutsch."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=1100
    )
    analysis_text = res.choices[0].message.content

    # Erwartetes 30-Tage-Ziel extrahieren
    t_30 = stats['current_price']
    match_30 = re.search(r'Ziel in 30 Tagen[^\d]*([\d.,]+)', analysis_text)
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
        "analysis_summary": analysis_text[:280] + "..."
    })
    save_memory(memory)

    return analysis_text
