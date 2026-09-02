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

        # Volumen-Verhältnis
        avg_vol_20 = float(volume.tail(20).mean()) if not volume.empty else 1.0
        last_vol = float(volume.iloc[-1]) if not volume.empty else 1.0
        vol_ratio = round(last_vol / avg_vol_20, 2) if avg_vol_20 > 0 else 1.0

        recent_trend = [round(float(x), 2) for x in close.tail(5).tolist()]

        # Trend-Signal ableiten
        trend_status = "Aufwärtstrend" if current_p > sma50 else ("Abwärtstrend" if current_p < sma50 else "Seitwärtsphase")

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
            "trend_status": trend_status,
            "last_5_days": recent_trend,
            "data_points": len(close)
        }
    except Exception:
        return None

def fetch_company_specific_news(ticker_sym):
    try:
        t = yf.Ticker(ticker_sym)
        news_items = getattr(t, "news", [])
        headlines = []
        if news_items:
            for item in news_items[:4]:
                title = item.get("title", "")
                if title:
                    headlines.append(f"• {title}")
        return "\n".join(headlines) if headlines else "Keine aktuellen Unternehmensmeldungen vorhanden."
    except Exception:
        return "Keine aktuellen Unternehmensmeldungen vorhanden."

def run_ai_learning_prediction(client, model_name, ticker_sym, company_name, stats, memory, macro_news=""):
    t = ticker_sym.upper()
    past_stock_memory = memory.get(t, {})
    past_predictions = past_stock_memory.get("history", [])

    # Mathematische Fehler-Rückkopplung
    accuracy_eval = "ERSTMALIGE ANALYSE: Es liegt noch kein historischer Lernstand in der Datenbank vor."
    error_list = []
    if past_predictions:
        accuracy_eval = "HISTORISCHER LERNSTAND & ABWEICHUNGSANALYSE:\n"
        for entry in past_predictions[-4:]:
            prev_p = entry.get("price", 0.0)
            target = entry.get("target_30d", prev_p)
            date_str = entry.get("date", "Unbekannt")
            if target > 0:
                diff_pct = abs((stats["current_price"] - target) / target) * 100.0
                error_list.append(diff_pct)
                accuracy_eval += f"- Prognose vom {date_str}: Damals {prev_p:.2f} € -> Ziel war {target:.2f} €. Aktueller Realkurs: {stats['current_price']:.2f} € (Differenz: {diff_pct:.1f}%)\n"
        
        if error_list:
            avg_err = sum(error_list) / len(error_list)
            accuracy_eval += f"\n-> Bisherige durchschnittliche Abweichung: {avg_err:.1f}%. Nutze diesen Fehlerfaktor, um Übertreibungen im Kursziel zu dämpfen!"

    specific_news = fetch_company_specific_news(ticker_sym)

    prompt = f"""
Du bist ein quantitatives KI-Prognosemodell. Erstelle eine fundierte, professionelle Aktienprognose AUSSCHLIESSLICH AUF DEUTSCH.
Verwende kein einziges englisches Wort in den Überschriften und Floskeln.

[1. WELTWIRTSCHAFT & UNTERNEHMENSNACHRICHTEN]
Globale Leitnachrichten:
{macro_news}

Unternehmensspezifische Nachrichten zu {company_name}:
{specific_news}

[2. TECHNISCHE 365-TAGE-KENNZAHLEN]
- Aktueller Börsenkurs: {stats['current_price']} €
- 365-Tage Wertentwicklung: {stats['return_365d_pct']} %
- 52-Wochen Tief / Hoch: {stats['low_365d']} € / {stats['high_365d']} €
- Gleitende Durchschnitte: 20 Tage: {stats['sma20']} € | 50 Tage: {stats['sma50']} € | 200 Tage: {stats['sma200']} €
- RSI (14 Tage): {stats['rsi_14']} (Momentum-Status)
- Schwankungsbreite (Volatilität): {stats['volatility_pct']} %
- Trendrichtung: {stats['trend_status']}
- Schlusskurse der letzten 5 Handelstage: {stats['last_5_days']}

[3. LERNGEDÄCHTNIS & FEHLERKORREKTUR]
{accuracy_eval}

AUFGABE:
Formuliere deine Analyse komplett auf Deutsch mit folgenden 4 Abschnitten:

### 1. 🌐 Marktlage & Nachrichten-Synthese
Bewerte in 3-4 Sätzen, wie die Zinspolitik, Weltwirtschaft und aktuelle Unternehmensnachrichten die Aktie beeinflussen.

### 2. 🧠 Erkenntnisse aus dem 365-Tage-Lernprozess
Welche Chart-Muster, Unterstützungszonen und Widerstände hat die KI aus den Kursdaten der letzten 365 Tage gelernt? Wie wurde die frühere Prognoseabweichung korrigiert?

### 3. 🎯 Multi-Szenario Kursprognose
- **Ziel in 7 Tagen:** [Zahl in €] ([+/- %])
- **Ziel in 30 Tagen (Hauptszenario):** [Zahl in €] ([+/- %])
- **Ziel in 90 Tagen:** [Zahl in €] ([+/- %])
- **🟢 Optimistisches Best-Case-Ziel (30 Tage):** [Zahl in €]
- **🔴 Absicherungs-Marke / Stop-Loss:** [Zahl in €]
- **Treffer-Wahrscheinlichkeit:** [z. B. 78 %]

### 4. 🧭 Konkreter Handlungsplan
Klare Anweisung für Privatanleger: Abwarten, Limit-Order platzieren oder schrittweise aufstocken?
"""

    res = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "Du bist ein führender deutscher quantitativer Analyst. Antworte ausschließlich in verständlichem, professionellem Deutsch."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=1100
    )
    analysis_text = res.choices[0].message.content

    # 30-Tage Zielwert aus dem deutschen Text ziehen
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
        "date": datetime.now().strftime("%d.%m.%Y %H:%M Uhr"),
        "price": stats['current_price'],
        "target_30d": t_30,
        "rsi": stats['rsi_14'],
        "analysis_summary": analysis_text[:280] + "..."
    })
    save_memory(memory)

    return analysis_text
