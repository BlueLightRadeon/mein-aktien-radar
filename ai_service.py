import streamlit as st
from groq import Groq
import re

DEFAULT_MODEL = "llama-3.3-70b-versatile"

def get_account_models(api_key):
    if not api_key:
        return [DEFAULT_MODEL]
    try:
        client = Groq(api_key=api_key)
        models_data = client.models.list()
        
        valid_chat_models = []
        for m in models_data.data:
            m_id = str(m.id).lower()
            if any(bad in m_id for bad in ["whisper", "tts", "orpheus", "arabic", "vision", "guard", "embed", "canopylabs"]):
                continue
            if any(allowed in m_id for allowed in ["llama-3.3", "gemma2", "qwen", "deepseek", "versatile"]):
                valid_chat_models.append(m.id)
                
        if DEFAULT_MODEL in valid_chat_models:
            valid_chat_models.remove(DEFAULT_MODEL)
            valid_chat_models.insert(0, DEFAULT_MODEL)
            
        return valid_chat_models if valid_chat_models else [DEFAULT_MODEL]
    except Exception:
        return [DEFAULT_MODEL]

def run_analysis(client, model_name, news_text, metrics_summary, ticker_news_text="", cluster_context=""):
    prompt = f"""
Du bist ein führender quantitativer Chef-Aktienanalyst. Erstelle einen vollständigen Finanzbericht auf Deutsch.
Schreibe alle Punkte direkt aus und nutze exakt die vier Überschriften:

### 1. MARKTANALYSE
- **TOP 10 Marktnachrichten**: Schreibe genau 10 konkrete Stichpunkte zu globalen Zinsen, Notenbanken, Halbleitern, Energie/KI-Rechenzentren, Rüstung und Geopolitik.
- **Gesamtstimmung der Börse**: 🟢 Optimistisch, 🟡 Neutral oder 🔴 Vorsichtig mit 3 Sätzen Begründung.

### 2. DEPOT-BEWERTUNG
Analysiere alle im Depot gehaltenen Aktien einzeln mit aktuellem Status vs. 3-Monats-Rückblick:
{metrics_summary}

Für jedes Unternehmen:
- **🟢/🟡/🔴 Aktuelle Empfehlung & Begründung**: (KGV, Fair Value, Auftragslage, Kurspotenzial).
- **⏱️ Rückblick vor 3 Monaten**: Damalige Einstufung und damalige Ausgangslage.
- **📈 Trend & Fazit**: Entwicklung in den letzten 3 Monaten und nächster Schritt.

### 3. TOP 5 KAUFEMPFEHLUNGEN
Empfehle 5 konkrete neue Qualitätsaktien/ETFs (die NICHT im aktuellen Depot liegen) zur Portfolio-Erweiterung:
1. Aktie 1 (Ticker | Branche | Land) - Einstiegsgrund, Kurspotenzial, Risiko.
2. Aktie 2 (Ticker | Branche | Land) - Einstiegsgrund, Kurspotenzial, Risiko.
3. Aktie 3 (Ticker | Branche | Land) - Einstiegsgrund, Kurspotenzial, Risiko.
4. Aktie 4 (Ticker | Branche | Land) - Einstiegsgrund, Kurspotenzial, Risiko.
5. Aktie 5 (Ticker | Branche | Land) - Einstiegsgrund, Kurspotenzial, Risiko.
- **🔴 Aktuell meiden**: 3 Branchen mit erhöhtem Abwärtsrisiko.

### 4. RISIKOSTREUUNG
- **Risiko-Score**: 1 (Sehr konservativ) bis 10 (Sehr spekulativ).
- **Analyse der Branchen- & Länder-Gewichtung**.
- **Konkreter Absicherungs-Tipp**.
"""

    duel_model = model_name if model_name and "orpheus" not in model_name else DEFAULT_MODEL
    models_to_try = [duel_model, DEFAULT_MODEL, "llama-3.3-70b-specdec"]
    seen = set()
    models_to_try = [x for x in models_to_try if x and not (x in seen or seen.add(x))]

    full_text = ""
    last_err = None

    for m_id in models_to_try:
        try:
            res = client.chat.completions.create(
                model=m_id,
                messages=[
                    {"role": "system", "content": "Du bist ein präziser deutscher Börsenanalyst. Verwende exakt die Überschriften '### 1. MARKTANALYSE', '### 2. DEPOT-BEWERTUNG', '### 3. TOP 5 KAUFEMPFEHLUNGEN' und '### 4. RISIKOSTREUUNG'."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=3500,
                timeout=35.0
            )
            if res.choices and len(res.choices) > 0 and res.choices[0].message and res.choices[0].message.content:
                full_text = res.choices[0].message.content
                break
        except Exception as e:
            last_err = e
            continue

    if not full_text:
        err_msg = f"⚠️ Fehler bei Groq-Generierung: {str(last_err)}"
        return err_msg, err_msg, err_msg, err_msg

    # Sichere Zerlegung nach den 4 Hauptkapiteln
    m_match = re.search(r'###\s*1\.\s*MARKTANALYSE(.*?)(?=###\s*2\.\s*DEPOT-BEWERTUNG|$)', full_text, re.DOTALL | re.IGNORECASE)
    d_match = re.search(r'###\s*2\.\s*DEPOT-BEWERTUNG(.*?)(?=###\s*3\.\s*TOP\s*5\s*KAUFEMPFEHLUNGEN|$)', full_text, re.DOTALL | re.IGNORECASE)
    s_match = re.search(r'###\s*3\.\s*TOP\s*5\s*KAUFEMPFEHLUNGEN(.*?)(?=###\s*4\.\s*RISIKOSTREUUNG|$)', full_text, re.DOTALL | re.IGNORECASE)
    c_match = re.search(r'###\s*4\.\s*RISIKOSTREUUNG(.*?)$', full_text, re.DOTALL | re.IGNORECASE)

    out_market = m_match.group(1).strip() if m_match else "Marktanalyse geladen."
    out_depot = d_match.group(1).strip() if d_match else "Depotbewertung geladen."
    out_signals = s_match.group(1).strip() if s_match else "Top 5 Empfehlungen geladen."
    out_cluster = c_match.group(1).strip() if c_match else "Streuungsanalyse geladen."

    return out_market, out_depot, out_signals, out_cluster

def run_duel_analysis(client, model_name, stock_a_info, stock_b_info):
    prompt = f"""
Vergleiche als quantitativer Analyst diese beiden Aktien objektiv und detailliert auf Deutsch:
Aktie A: {stock_a_info}
Aktie B: {stock_b_info}

1. Fundamentaler Kennzahlenvergleich (KGV, Dividende, Fair Value, Burggraben)
2. Klares Fazit: Welche Aktie ist aktuell der bessere Kauf und warum?
"""
    duel_model = model_name if model_name and "orpheus" not in model_name else DEFAULT_MODEL
    
    try:
        res = client.chat.completions.create(
            model=duel_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=800,
            timeout=20.0
        )
        return res.choices[0].message.content
    except Exception:
        try:
            res_fallback = client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=800,
                timeout=20.0
            )
            return res_fallback.choices[0].message.content
        except Exception as e2:
            return f"⚠️ Duell-Analyse Fehler: {str(e2)}"
