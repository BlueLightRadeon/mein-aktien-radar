import streamlit as st
from groq import Groq
import re

DEFAULT_MODEL = "llama-3.3-70b-versatile"

def get_account_models(api_key):
    """Filtert die Modell-Liste streng nach reinen Text-/Chat-Modellen."""
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
    combined_prompt = f"""
Du bist ein erfahrener Chef-Aktienanalyst. Formuliere eine vollständige, tiefgehende Finanzanalyse auf Deutsch. 
WICHTIG: Verwende keine Platzhalter oder Mustertexte, sondern schreibe jeden einzelnen Punkt und jede Aktienanalyse vollständig und konkret aus!

AKTUELLE NACHRICHTENLAGE:
{news_text}

BESTEHENDE DEPOTWERTE DES NUTZERS:
{metrics_summary}

DEPOT-STRUKTUR:
{cluster_context}

Gliedere deine Antwort zwingend anhand der folgenden vier Trennmarkierungen:

===MARKT===
### 🌍 TOP 10 Marktnachrichten
Schreibe genau 10 konkrete, aussagekräftige Stichpunkte zu den aktuellen weltweiten Leitbörsen, Zinsentscheiden, Tech-Entwicklungen und makroökonomischen Trends.

### 🧭 Gesamtstimmung der Börse
Bewerte die Gesamtlage eindeutig mit 🟢 Optimistisch, 🟡 Neutral oder 🔴 Vorsichtig und begründe dies in 3-4 Sätzen.

===DEPOT===
### 💼 KAUF- & VERKAUFSEMPFEHLUNGEN FÜR DEINE BESTEHENDEN AKTIEN
Analysiere ausnahmslos JEDE im Depot vorhandene Aktie einzeln:
- Gib für jede Aktie eine klare Empfehlung (🟢 KAUFEN / AUFSTOCKEN, 🟡 HALTEN oder 🔴 GEWINNE MITNEHMEN / VERKAUFEN).
- Erläutere jeweils in 2-3 Sätzen die fundamentale Begründung (Chancen, Risiken, Kurspotenzial).

===SIGNALE===
### 🎯 TOP 5 NEUE KAUF-EMPFEHLUNGEN
Empfehle 5 konkrete, kaufenswerte Qualitätsaktien oder ETFs zur Portfolio-Ergänzung (keine Werte, die schon im Depot liegen).
Nenne jeweils:
1. Name und Ticker
2. Warum sich der Einstieg jetzt lohnt
3. Kurspotenzial und Risiko

#### 🔴 AKTUELL MEIDEN
Nenne 3 Branchen oder Anlagesegmente, die derzeit gemieden werden sollten.

===KLUMPEN===
### 🛡️ Risikostreuung & Depot-Optimierung
1. Risiko-Score von 1 (sehr sicher) bis 10 (sehr spekulativ).
2. Ausführliche Bewertung der aktuellen Länder- und Branchenstreuung.
3. Konkreter Ratschlag, wie das Portfolio noch krisenfester aufgestellt werden kann.
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
                    {"role": "system", "content": "Du bist ein führender deutscher Börsenanalyst. Antworte immer auf Deutsch und formuliere alle Analysen direkt und ausführlich aus."},
                    {"role": "user", "content": combined_prompt}
                ],
                temperature=0.3,
                max_tokens=3000,
                timeout=30.0
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

    normalized_text = full_text
    normalized_text = re.sub(r'(\*{0,2}={2,5}\s*MARKT\s*={2,5}\*{0,2}|#{1,4}\s*MARKT)', '<<<SECTION_MARKT>>>', normalized_text, flags=re.IGNORECASE)
    normalized_text = re.sub(r'(\*{0,2}={2,5}\s*DEPOT\s*={2,5}\*{0,2}|#{1,4}\s*DEPOT)', '<<<SECTION_DEPOT>>>', normalized_text, flags=re.IGNORECASE)
    normalized_text = re.sub(r'(\*{0,2}={2,5}\s*(?:SIGNALE|KAUF|EMPFEHLUNGEN)\s*={2,5}\*{0,2}|#{1,4}\s*(?:SIGNALE|KAUF|EMPFEHLUNGEN))', '<<<SECTION_SIGNALE>>>', normalized_text, flags=re.IGNORECASE)
    normalized_text = re.sub(r'(\*{0,2}={2,5}\s*(?:KLUMPEN|RISIKO|STREUUNG)\s*={2,5}\*{0,2}|#{1,4}\s*(?:KLUMPEN|RISIKO|STREUUNG))', '<<<SECTION_KLUMPEN>>>', normalized_text, flags=re.IGNORECASE)

    sections = {}
    parts = re.split(r'<<<(SECTION_[A-Z]+)>>>', normalized_text)
    if len(parts) >= 3:
        for i in range(1, len(parts), 2):
            sec_name = parts[i]
            sec_content = parts[i+1].strip() if i+1 < len(parts) else ""
            sections[sec_name] = sec_content

    out_market = sections.get("SECTION_MARKT", full_text)
    out_depot = sections.get("SECTION_DEPOT", "Handelsempfehlungen liegen im Tab '💼 Stimmung & Empfehlungen' vor.")
    out_signals = sections.get("SECTION_SIGNALE", full_text)
    out_cluster = sections.get("SECTION_KLUMPEN", "Streuungsanalyse erstellt.")

    return out_market.strip(), out_depot.strip(), out_signals.strip(), out_cluster.strip()

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
