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

def clean_ai_output(text):
    if not text:
        return ""
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    cleaned = re.sub(r'^<think>.*?\n(?=(?:={2,}|#{1,4}))', '', cleaned, flags=re.DOTALL).strip()
    cleaned = re.sub(r'^(?:Here\'s|Thinking Process).*?\n\n(?=(?:={2,}|#{1,4}|\b1\.))', '', cleaned, flags=re.DOTALL | re.IGNORECASE).strip()
    return cleaned

def extract_4_sections(raw_text):
    cleaned = clean_ai_output(raw_text)
    if not cleaned:
        return "", "", "", ""

    patterns = [
        ("market", re.compile(r'(?:={2,}\s*(?:ABSCHNITT\s*1|MARKTANALYSE)[^=]*={2,}|#{1,4}\s*1[\.\s\:\-]+MARKT[^\n]*|###\s*MARKTANALYSE|\b1\.\s*MARKTANALYSE\b)', re.IGNORECASE)),
        ("depot", re.compile(r'(?:={2,}\s*(?:ABSCHNITT\s*2|DEPOT)[^=]*={2,}|#{1,4}\s*2[\.\s\:\-]+DEPOT[^\n]*|###\s*DEPOT-BEWERTUNG|\b2\.\s*DEPOT-BEWERTUNG\b)', re.IGNORECASE)),
        ("signals", re.compile(r'(?:={2,}\s*(?:ABSCHNITT\s*3|KAUFEMPFEHLUNGEN)[^=]*={2,}|#{1,4}\s*3[\.\s\:\-]+(?:TOP|KAUF)[^\n]*|###\s*(?:TOP\s*5|KAUFEMPFEHLUNGEN)|\b3\.\s*TOP\s*5\b)', re.IGNORECASE)),
        ("cluster", re.compile(r'(?:={2,}\s*(?:ABSCHNITT\s*4|RISIKO)[^=]*={2,}|#{1,4}\s*4[\.\s\:\-]+RISIKO[^\n]*|###\s*RISIKOSTREUUNG|\b4\.\s*RISIKOSTREUUNG\b)', re.IGNORECASE))
    ]

    matches = []
    for key, pat in patterns:
        m = pat.search(cleaned)
        if m:
            matches.append((m.start(), m.end(), key))

    matches.sort(key=lambda x: x[0])

    sections = {}
    for i, (start_idx, end_idx, key) in enumerate(matches):
        next_start = matches[i + 1][0] if i + 1 < len(matches) else len(cleaned)
        content = cleaned[end_idx:next_start].strip()
        content = re.sub(r'^[:\s\-=]+', '', content).strip()
        sections[key] = content

    out_market = sections.get("market", "")
    out_depot = sections.get("depot", "")
    out_signals = sections.get("signals", "")
    out_cluster = sections.get("cluster", "")

    if not out_market and not out_depot:
        out_market = cleaned

    return out_market, out_depot, out_signals, out_cluster

def run_analysis(client, model_name, news_text, metrics_summary, ticker_news_text="", cluster_context=""):
    prompt = f"""
Du bist ein führender quantitativer Chef-Aktienanalyst. Erstelle einen vollständigen Finanzbericht auf Deutsch.
WICHTIG: Antworte zu 100% auf DEUTSCH. Verwende niemals englische Denkprozesse (<think>).

[AKTUELLE LEIT-NACHRICHTEN & WELTMÄRKTE]
{news_text}

[DEPOT-KENNZAHLEN & AKTIEN]
{metrics_summary}

Gliedere deine Antwort zwingend mit genau diesen 4 Abschnitts-Überschriften:

===ABSCHNITT 1: MARKTANALYSE===
- **TOP 10 Marktnachrichten**: Genau 10 konkrete, ausführliche Stichpunkte zu globalen Zinsen, Notenbanken, Halbleitern, Energie/KI-Rechenzentren, Rüstung und Geopolitik.
- **Gesamtstimmung der Börse**: 🟢 Optimistisch, 🟡 Neutral oder 🔴 Vorsichtig mit 3 Sätzen Begründung.

===ABSCHNITT 2: DEPOT-BEWERTUNG===
Analysiere alle im Depot gehaltenen Aktien einzeln mit aktuellem Status vs. 3-Monats-Rückblick:
- **🟢/🟡/🔴 Aktuelle Empfehlung & Begründung**: (KGV, Fair Value, Auftragslage, Kurspotenzial).
- **⏱️ Rückblick vor 3 Monaten**: Damalige Einstufung und damalige Ausgangslage.
- **📈 Trend & Fazit**: Entwicklung in den letzten 3 Monaten und nächster Schritt.

===ABSCHNITT 3: TOP 5 KAUFEMPFEHLUNGEN===
Empfehle 5 konkrete neue Qualitätsaktien/ETFs (die NICHT im aktuellen Depot liegen) zur Portfolio-Erweiterung:
1. Aktie 1 (Ticker | Branche | Land) - Einstiegsgrund, Kurspotenzial, Risiko.
2. Aktie 2 (Ticker | Branche | Land) - Einstiegsgrund, Kurspotenzial, Risiko.
3. Aktie 3 (Ticker | Branche | Land) - Einstiegsgrund, Kurspotenzial, Risiko.
4. Aktie 4 (Ticker | Branche | Land) - Einstiegsgrund, Kurspotenzial, Risiko.
5. Aktie 5 (Ticker | Branche | Land) - Einstiegsgrund, Kurspotenzial, Risiko.
- **🔴 Aktuell meiden**: 3 Branchen mit erhöhtem Abwärtsrisiko.

===ABSCHNITT 4: RISIKOSTREUUNG===
- **Risiko-Score**: 1 (Sehr konservativ) bis 10 (Sehr spekulativ).
- **Analyse der Branchen- & Länder-Gewichtung**: {cluster_context}
- **Konkreter Absicherungs-Tipp**: Klare Handlungsanweisung zur Portfolio-Absicherung.
"""

    target_model = model_name if model_name and "orpheus" not in model_name else DEFAULT_MODEL

    try:
        res = client.chat.completions.create(
            model=target_model,
            messages=[
                {"role": "system", "content": "Du bist ein führender deutscher Börsenanalyst. Antworte ausschließlich auf Deutsch. Gib niemals Denkprozesse (<think>) aus."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=2800,
            timeout=25.0
        )
        if res.choices and len(res.choices) > 0 and res.choices[0].message and res.choices[0].message.content:
            return extract_4_sections(res.choices[0].message.content)
    except Exception as e:
        # Ein schneller Fallback auf das Standardmodell
        try:
            res_fb = client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2500,
                timeout=20.0
            )
            return extract_4_sections(res_fb.choices[0].message.content)
        except Exception as e2:
            err_msg = f"⚠️ Groq-Fehler: {str(e2)}"
            return err_msg, err_msg, err_msg, err_msg

    return "Keine Daten erhalten.", "", "", ""

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
            temperature=0.2,
            max_tokens=800,
            timeout=15.0
        )
        return clean_ai_output(res.choices[0].message.content)
    except Exception as e:
        return f"⚠️ Duell-Analyse Fehler: {str(e)}"
