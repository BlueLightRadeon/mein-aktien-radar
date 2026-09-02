import streamlit as st
from groq import Groq
import re
import time

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
            if any(bad in m_id for bad in ["whisper", "tts", "orpheus", "arabic", "vision", "guard", "embed", "canopylabs", "deepseek", "r1", "qwq"]):
                continue
            if any(allowed in m_id for allowed in ["llama-3.3", "llama-3.1", "gemma2", "qwen", "versatile"]):
                valid_chat_models.append(m.id)
                
        if DEFAULT_MODEL in valid_chat_models:
            valid_chat_models.remove(DEFAULT_MODEL)
            valid_chat_models.insert(0, DEFAULT_MODEL)
            
        return valid_chat_models if valid_chat_models else [DEFAULT_MODEL]
    except Exception:
        return [DEFAULT_MODEL]

def clean_ai_output(text):
    if not text:
        return ""
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    cleaned = re.sub(r'^<think>.*?\n', '', cleaned, flags=re.DOTALL).strip()
    cleaned = re.sub(r'^(?:Here\'s|Thinking Process).*?\n\n', '', cleaned, flags=re.DOTALL | re.IGNORECASE).strip()
    return cleaned.strip()

def extract_4_sections(text):
    cleaned = clean_ai_output(text)
    if not cleaned:
        return "", "", "", ""

    # Finde alle Abschnitte fehlertolerant (auch wenn Markdown-Überschriften mehrfach im Text vorkommen)
    pat_m = list(re.finditer(r'(?:#{1,4}\s*|\*{1,2})?1[\.\s\:\-]+MARKT[^\n]*', cleaned, re.IGNORECASE))
    pat_d = list(re.finditer(r'(?:#{1,4}\s*|\*{1,2})?2[\.\s\:\-]+DEPOT[^\n]*', cleaned, re.IGNORECASE))
    pat_s = list(re.finditer(r'(?:#{1,4}\s*|\*{1,2})?3[\.\s\:\-]+(?:TOP|KAUF)[^\n]*', cleaned, re.IGNORECASE))
    pat_c = list(re.finditer(r'(?:#{1,4}\s*|\*{1,2})?4[\.\s\:\-]+RISIKO[^\n]*', cleaned, re.IGNORECASE))

    # Nimm das jeweils letzte Vorkommen, um vorangestellte Planungsnotizen/Outlines zu überspringen
    idx_m = pat_m[-1].end() if pat_m else None
    idx_d_start = pat_d[-1].start() if pat_d else None
    idx_d_end = pat_d[-1].end() if pat_d else None
    idx_s_start = pat_s[-1].start() if pat_s else None
    idx_s_end = pat_s[-1].end() if pat_s else None
    idx_c_start = pat_c[-1].start() if pat_c else None
    idx_c_end = pat_c[-1].end() if pat_c else None

    out_market = ""
    out_depot = ""
    out_signals = ""
    out_cluster = ""

    if idx_m is not None and idx_d_start is not None:
        out_market = cleaned[idx_m:idx_d_start].strip()
    elif idx_m is not None:
        out_market = cleaned[idx_m:].strip()

    if idx_d_end is not None and idx_s_start is not None:
        out_depot = cleaned[idx_d_end:idx_s_start].strip()
    elif idx_d_end is not None:
        out_depot = cleaned[idx_d_end:].strip()

    if idx_s_end is not None and idx_c_start is not None:
        out_signals = cleaned[idx_s_end:idx_c_start].strip()
    elif idx_s_end is not None:
        out_signals = cleaned[idx_s_end:].strip()

    if idx_c_end is not None:
        out_cluster = cleaned[idx_c_end:].strip()

    # Falls Trenner komplett fehlten, Text nicht verwerfen
    if not out_market and not out_depot:
        out_market = cleaned

    return out_market, out_depot, out_signals, out_cluster

def run_analysis(client, model_name, news_text, metrics_summary, ticker_news_text="", cluster_context=""):
    prompt = f"""
Du bist ein führender quantitativer Chef-Aktienanalyst. Erstelle einen vollständigen Finanzbericht auf Deutsch.
Beginne direkt mit '### 1. MARKTANALYSE'. Schreibe keine englischen Entwürfe, keine Vorbemerkungen und kein Brainstorming.

[AKTUELLE LEIT-NACHRICHTEN & WELTMÄRKTE]
{news_text}

[DEPOT-KENNZAHLEN & AKTIEN]
{metrics_summary}

Gliedere deine Antwort zwingend in genau diese vier Abschnitte:

### 1. MARKTANALYSE
- **TOP 10 Marktnachrichten**: Genau 10 konkrete, nummerierte Stichpunkte zu globalen Zinsen, Notenbanken, Halbleitern, Energie/KI-Rechenzentren, Rüstung und Geopolitik.
- **Gesamtstimmung der Börse**: 🟢 Optimistisch, 🟡 Neutral oder 🔴 Vorsichtig mit 3 Sätzen Begründung.

### 2. DEPOT-BEWERTUNG
Analysiere alle im Depot gehaltenen Aktien einzeln mit aktuellem Status vs. 3-Monats-Rückblick:
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
- **Analyse der Branchen- & Länder-Gewichtung**: {cluster_context}
- **Konkreter Absicherungs-Tipp**: Klare Handlungsanweisung zur Portfolio-Absicherung.
"""

    target_model = model_name if model_name and not any(bad in model_name.lower() for bad in ["orpheus", "deepseek", "r1", "qwq"]) else DEFAULT_MODEL
    full_text = ""
    last_err = None

    for attempt in range(2):
        try:
            res = client.chat.completions.create(
                model=target_model,
                messages=[
                    {"role": "system", "content": "Du bist ein präziser deutscher Börsenanalyst. Antworte ausschließlich auf Deutsch. Beginne direkt mit '### 1. MARKTANALYSE'. Gib niemals Denkprozesse, Entwürfe oder englische Einleitungen aus."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=3200,
                timeout=40.0
            )
            if res.choices and len(res.choices) > 0 and res.choices[0].message and res.choices[0].message.content:
                full_text = res.choices[0].message.content
                break
        except Exception as e:
            last_err = e
            time.sleep(1.0)

    if not full_text:
        err_msg = f"⚠️ Fehler bei Groq-Generierung: {str(last_err)}"
        return err_msg, err_msg, err_msg, err_msg

    return extract_4_sections(full_text)

def run_duel_analysis(client, model_name, stock_a_info, stock_b_info):
    prompt = f"""
Vergleiche als quantitativer Analyst diese beiden Aktien objektiv und detailliert auf Deutsch:
Aktie A: {stock_a_info}
Aktie B: {stock_b_info}

1. Fundamentaler Kennzahlenvergleich (KGV, Dividende, Fair Value, Burggraben)
2. Klares Fazit: Welche Aktie ist aktuell der bessere Kauf und warum?
"""
    duel_model = model_name if model_name and not any(bad in model_name.lower() for bad in ["orpheus", "deepseek", "r1", "qwq"]) else DEFAULT_MODEL
    try:
        res = client.chat.completions.create(
            model=duel_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=800,
            timeout=20.0
        )
        return clean_ai_output(res.choices[0].message.content)
    except Exception as e:
        return f"⚠️ Duell-Analyse Fehler: {str(e)}"
