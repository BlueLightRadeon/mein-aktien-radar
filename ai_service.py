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
    """Entfernt Thinking-Tags und englische Meta-Outlines vor dem Parsen."""
    if not text:
        return ""
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    cleaned = re.sub(r'^<think>.*?\n', '', cleaned, flags=re.DOTALL).strip()
    cleaned = re.sub(r'^(?:Here\'s|Thinking Process).*?\n\n', '', cleaned, flags=re.DOTALL | re.IGNORECASE).strip()
    # Entfernt vorangestellte 'Content Requirements' Blöcke
    cleaned = re.sub(r'^.*?Content Requirements:.*?\n(?=(?:<[A-Z_]+>|#{1,4}\s*1|\b1\.))', '', cleaned, flags=re.DOTALL | re.IGNORECASE).strip()
    return cleaned

def extract_section_by_tag(text, tag_name, fallback_regex):
    """
    Sucht zuerst nach eindeutigen XML-Tags (<TAG>...</TAG>).
    Falls nicht vorhanden, greift der Fallback-RegEx auf die längste echte Textpassage zu.
    """
    # 1. Priorität: XML-Tags
    tag_pattern = rf'<{tag_name}>(.*?)(?:</{tag_name}>|$)'
    matches = re.findall(tag_pattern, text, re.DOTALL | re.IGNORECASE)
    if matches:
        best = max(matches, key=len).strip()
        if len(best) > 40 and "Content Requirements:" not in best:
            return best

    # 2. Priorität: Fallback per Markdown-Überschrift (filtert Outlines aus)
    fb_matches = re.findall(fallback_regex, text, re.DOTALL | re.IGNORECASE)
    if fb_matches:
        valid_matches = [m.strip() for m in fb_matches if "Content Requirements:" not in m and len(m.strip()) > 50]
        if valid_matches:
            return max(valid_matches, key=len)

    return ""

def run_analysis(client, model_name, news_text, metrics_summary, ticker_news_text="", cluster_context=""):
    prompt = f"""
Du bist ein quantitativer Chef-Aktienanalyst. Erstelle einen vollständigen Finanzbericht auf Deutsch.
WICHTIG: Antworte zu 100% auf DEUTSCH. Schreibe KEINE englischen Denkprozesse (<think>), KEINE Einleitung und KEINE "Content Requirements" oder Gliederung vorab!
Beginne deine Ausgabe SOFORT mit dem Tag <MARKTANALYSE>.

[AKTUELLE LEIT-NACHRICHTEN & WELTMÄRKTE]
{news_text}

[DEPOT-KENNZAHLEN & AKTIEN]
{metrics_summary}

Gliedere deine Antwort zwingend in genau diese vier Tags:

<MARKTANALYSE>
### 1. MARKTANALYSE
- **TOP 10 Marktnachrichten**: Genau 10 konkrete, ausführliche Stichpunkte zu globalen Zinsen, Notenbanken, Halbleitern, Energie/KI-Rechenzentren, Rüstung und Geopolitik.
- **Gesamtstimmung der Börse**: 🟢 Optimistisch, 🟡 Neutral oder 🔴 Vorsichtig mit 3 Sätzen Begründung.
</MARKTANALYSE>

<DEPOTBEWERTUNG>
### 2. DEPOT-BEWERTUNG
Analysiere alle im Depot gehaltenen Aktien einzeln mit aktuellem Status vs. 3-Monats-Rückblick:
Für jedes Unternehmen:
- **🟢/🟡/🔴 Aktuelle Empfehlung & Begründung**: (KGV, Fair Value, Auftragslage, Kurspotenzial).
- **⏱️ Rückblick vor 3 Monaten**: Damalige Einstufung und damalige Ausgangslage.
- **📈 Trend & Fazit**: Entwicklung in den letzten 3 Monaten und nächster Schritt.
</DEPOTBEWERTUNG>

<KAUFEMPFEHLUNGEN>
### 3. TOP 5 KAUFEMPFEHLUNGEN
Empfehle 5 konkrete neue Qualitätsaktien/ETFs (die NICHT im aktuellen Depot liegen) zur Portfolio-Erweiterung:
1. Aktie 1 (Ticker | Branche | Land) - Einstiegsgrund, Kurspotenzial, Risiko.
2. Aktie 2 (Ticker | Branche | Land) - Einstiegsgrund, Kurspotenzial, Risiko.
3. Aktie 3 (Ticker | Branche | Land) - Einstiegsgrund, Kurspotenzial, Risiko.
4. Aktie 4 (Ticker | Branche | Land) - Einstiegsgrund, Kurspotenzial, Risiko.
5. Aktie 5 (Ticker | Branche | Land) - Einstiegsgrund, Kurspotenzial, Risiko.
- **🔴 Aktuell meiden**: 3 Branchen mit erhöhtem Abwärtsrisiko.
</KAUFEMPFEHLUNGEN>

<RISIKOSTREUUNG>
### 4. RISIKOSTREUUNG
- **Risiko-Score**: 1 (Sehr konservativ) bis 10 (Sehr spekulativ).
- **Analyse der Branchen- & Länder-Gewichtung**: {cluster_context}
- **Konkreter Absicherungs-Tipp**: Klare Handlungsanweisung zur Portfolio-Absicherung.
</RISIKOSTREUUNG>
"""

    target_model = model_name if model_name and "orpheus" not in model_name else DEFAULT_MODEL
    full_text = ""
    last_err = None

    for attempt in range(2):
        try:
            res = client.chat.completions.create(
                model=target_model,
                messages=[
                    {"role": "system", "content": "Du bist ein führender deutscher Börsenanalyst. Antworte ausschließlich auf Deutsch. Gib niemals Denkprozesse (<think>) oder Vorbemerkungen aus. Verwende die Tags <MARKTANALYSE>, <DEPOTBEWERTUNG>, <KAUFEMPFEHLUNGEN> und <RISIKOSTREUUNG>."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=3200,
                timeout=35.0
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

    cleaned = clean_ai_output(full_text)

    # 4 Abschnitte zielgenau extrahieren
    fb_m = r'#{1,4}\s*(?:1[\.\s\:\-]+)?MARKT[^\n]*\n(.*?)(?=#{1,4}\s*(?:2[\.\s\:\-]+)?DEPOT|$)'
    fb_d = r'#{1,4}\s*(?:2[\.\s\:\-]+)?DEPOT[^\n]*\n(.*?)(?=#{1,4}\s*(?:3[\.\s\:\-]+)?(?:TOP|KAUF)|$)'
    fb_s = r'#{1,4}\s*(?:3[\.\s\:\-]+)?(?:TOP|KAUF)[^\n]*\n(.*?)(?=#{1,4}\s*(?:4[\.\s\:\-]+)?RISIKO|$)'
    fb_c = r'#{1,4}\s*(?:4[\.\s\:\-]+)?RISIKO[^\n]*\n(.*?)$'

    out_market = extract_section_by_tag(cleaned, "MARKTANALYSE", fb_m)
    out_depot = extract_section_by_tag(cleaned, "DEPOTBEWERTUNG", fb_d)
    out_signals = extract_section_by_tag(cleaned, "KAUFEMPFEHLUNGEN", fb_s)
    out_cluster = extract_section_by_tag(cleaned, "RISIKOSTREUUNG", fb_c)

    if not out_market and not out_depot:
        out_market = cleaned

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
    except Exception as e:
        return f"⚠️ Duell-Analyse Fehler: {str(e)}"
