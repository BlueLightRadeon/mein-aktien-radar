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
    """Entfernt alle internen Thinking-Blöcke (<think>...</think>) und Prefixes."""
    if not text:
        return ""
    # 1. Geschlossene <think>...</think> Blöcke entfernen
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    # 2. Falls der <think> Tag am Anfang nicht geschlossen wurde
    cleaned = re.sub(r'^<think>.*?\n(?=#{1,4}\s|\*\*)', '', cleaned, flags=re.DOTALL).strip()
    # 3. Englische Thinking-Floskeln am Anfang entfernen
    cleaned = re.sub(r'^.*?Here\'s a thinking process.*?\n\n', '', cleaned, flags=re.DOTALL | re.IGNORECASE).strip()
    return cleaned

def extract_section(text, start_pattern, end_pattern, fallback_text):
    """Extrahiert einen Abschnitt robust und prüft, ob echter Text (nicht nur Kommas) vorliegt."""
    match = re.search(f"{start_pattern}(.*?)(?={end_pattern}|$)", text, re.DOTALL | re.IGNORECASE)
    if match:
        content = match.group(1).strip()
        # Prüfen, ob sinnvoller Inhalt existiert (mehr als 20 Zeichen und nicht nur Sonderzeichen)
        clean_content = re.sub(r'[\s,.\-_#*:]+', '', content)
        if len(clean_content) >= 20:
            return content
    return fallback_text

def run_analysis(client, model_name, news_text, metrics_summary, ticker_news_text="", cluster_context=""):
    prompt = f"""
Du bist ein führender quantitativer Chef-Aktienanalyst. Erstelle einen vollständigen Finanzbericht auf Deutsch.
WICHTIG: Gib NIEMALS interne Denkprozesse (<think>) oder englische Wörter aus.

[AKTUELLE LEIT-NACHRICHTEN & WELTMÄRKTE]
{news_text}

[DEPOT-KENNZAHLEN & AKTIEN]
{metrics_summary}

Schreibe alle Punkte direkt und ausführlich auf Deutsch aus und nutze exakt diese vier Überschriften:

### 1. MARKTANALYSE
- **TOP 10 Marktnachrichten**: Schreibe genau 10 konkrete, ausführliche Stichpunkte zu globalen Zinsen, Notenbanken, Halbleitern, Energie/KI-Rechenzentren, Rüstung und Geopolitik.
- **Gesamtstimmung der Börse**: 🟢 Optimistisch, 🟡 Neutral oder 🔴 Vorsichtig mit 3 Sätzen Begründung.

### 2. DEPOT-BEWERTUNG
Analysiere alle im Depot gehaltenen Aktien einzeln mit aktuellem Status vs. 3-Monats-Rückblick:
Für jedes Unternehmen im Depot:
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

    target_model = model_name if model_name and "orpheus" not in model_name else DEFAULT_MODEL
    models_to_try = [target_model, DEFAULT_MODEL, "llama-3.3-70b-specdec"]
    seen = set()
    models_to_try = [x for x in models_to_try if x and not (x in seen or seen.add(x))]

    full_text = ""
    last_err = None

    for m_id in models_to_try:
        for attempt in range(2):
            try:
                res = client.chat.completions.create(
                    model=m_id,
                    messages=[
                        {"role": "system", "content": "Du bist ein führender deutscher Börsenanalyst. Antworte zu 100% auf Deutsch. Gib NIEMALS Denkprozesse (<think>) aus. Verwende exakt die vier Markdown-Überschriften '### 1. MARKTANALYSE', '### 2. DEPOT-BEWERTUNG', '### 3. TOP 5 KAUFEMPFEHLUNGEN' und '### 4. RISIKOSTREUUNG'."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=3500,
                    timeout=40.0
                )
                if res.choices and len(res.choices) > 0 and res.choices[0].message and res.choices[0].message.content:
                    full_text = res.choices[0].message.content
                    break
            except Exception as e:
                last_err = e
                if "rate_limit" in str(e).lower() and attempt == 0:
                    time.sleep(2.0)
                    continue
                break
        if full_text:
            break

    if not full_text:
        err_msg = f"⚠️ Fehler bei Groq-Generierung: {str(last_err)}"
        return err_msg, err_msg, err_msg, err_msg

    # 1. Denkblöcke sauber entfernen BEVOR RegEx sucht!
    cleaned_text = clean_ai_output(full_text)

    # 2. Flexible RegEx-Muster (akzeptiert 2 bis 4 '#' und optionale Doppelpunkte)
    p1 = r'#{2,4}\s*1\.\s*MARKTANALYSE[:\s]*'
    p2 = r'#{2,4}\s*2\.\s*DEPOT-BEWERTUNG[:\s]*'
    p3 = r'#{2,4}\s*3\.\s*TOP\s*5\s*KAUFEMPFEHLUNGEN[:\s]*'
    p4 = r'#{2,4}\s*4\.\s*RISIKOSTREUUNG[:\s]*'

    out_market = extract_section(cleaned_text, p1, p2, "Marktanalyse wird geladen...")
    out_depot = extract_section(cleaned_text, p2, p3, "Depot-Bewertung wird geladen...")
    out_signals = extract_section(cleaned_text, p3, p4, "Kaufempfehlungen werden geladen...")
    out_cluster = extract_section(cleaned_text, p4, r'$', "Risikostreuung wird geladen...")

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
            temperature=0.2,
            max_tokens=800,
            timeout=20.0
        )
        return clean_ai_output(res.choices[0].message.content)
    except Exception:
        try:
            res_fallback = client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=800,
                timeout=20.0
            )
            return clean_ai_output(res_fallback.choices[0].message.content)
        except Exception as e2:
            return f"⚠️ Duell-Analyse Fehler: {str(e2)}"
