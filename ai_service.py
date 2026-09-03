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

# --- 1. SPEZIALISIERTE NACHRICHTEN-ANALYSE (TAB 2) ---
def run_market_news_analysis(client, model_name, news_text):
    prompt = f"""
Du bist ein führender deutscher Börsenanalyst. Analysiere die aktuellen Weltmärkte und Leitnachrichten zu 100% auf Deutsch.
Schreibe keine englischen Denkprozesse, keine Vorbemerkungen und kein Brainstorming.

[AKTUELLE LEIT-NACHRICHTEN & WELTMÄRKTE]
{news_text}

Erstelle deinen Bericht mit exakt dieser Struktur:

### 1. 🌍 Globale Top 10 Marktnachrichten
Erstelle genau 10 ausführliche, konkrete und nummerierte Punkte. Decke dabei zwingend ab:
1. Globale Zinsen & Leitzinsentscheidungen der Fed & EZB
2. Halbleiterbranche & Auslastung von Sub-3nm-Nodes
3. Energieversorgung & Strombedarf für KI-Rechenzentren
4. Rüstungssektor & NATO-Auftragsbestände
5. Geopolitische Spannungen & Rohstoffpreise
6. DAX & europäische Leitindizes
7. US-Tech-Giganten & Hyperscaler-Investitionen
8. Anleihemärkte & Renditekurven
9. Währungsmärkte (EUR/USD)
10. Typische Anleger-Fehler im aktuellen Marktumfeld

### 2. 🧭 Gesamtstimmung der Börse
- **Stimmung**: 🟢 Optimistisch, 🟡 Neutral oder 🔴 Vorsichtig
- **Begründung**: Genau 3 prägnante, fundierte Sätze zur aktuellen Marktlage.
"""
    target_model = model_name if model_name and not any(bad in model_name.lower() for bad in ["orpheus", "deepseek", "r1"]) else DEFAULT_MODEL
    try:
        res = client.chat.completions.create(
            model=target_model,
            messages=[
                {"role": "system", "content": "Du bist ein führender quantitativer Analyst. Antworte ausschließlich auf Deutsch. Gib alle 10 Punkte vollständig aus."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=2500,
            timeout=40.0
        )
        return clean_ai_output(res.choices[0].message.content)
    except Exception as e:
        return f"⚠️ Fehler bei der Marktanalyse: {e}"

# --- 2. SPEZIALISIERTE DEPOT-BEWERTUNG (TAB 1) ---
def run_depot_analysis(client, model_name, metrics_summary):
    prompt = f"""
Du bist ein quantitativer Chef-Aktienanalyst. Analysiere alle im Depot gehaltenen Aktien einzeln auf Deutsch.

[DEPOT-KENNZAHLEN & AKTIEN]
{metrics_summary}

Analysiere JEDES gehaltene Unternehmen mit folgender Gliederung:

### [Unternehmensname]
- **🟢/🟡/🔴 Aktuelle Einstufung & Begründung**: Konkrete Bewertung anhand von KGV, Fair Value, Auftragslage und Kurspotenzial.
- **⏱️ Rückblick vor 3 Monaten**: Damalige Einstufung und damalige Ausgangslage.
- **📈 3-Monats-Trend & Fazit**: Wie hat sich das Papier entwickelt und welcher Schritt ist jetzt ratsam (Kauf/Halten/Verkauf).
"""
    target_model = model_name if model_name and not any(bad in model_name.lower() for bad in ["orpheus", "deepseek", "r1"]) else DEFAULT_MODEL
    try:
        res = client.chat.completions.create(
            model=target_model,
            messages=[
                {"role": "system", "content": "Du bist ein präziser deutscher Portfoliomanager. Antworte ausschließlich auf Deutsch."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=3000,
            timeout=45.0
        )
        return clean_ai_output(res.choices[0].message.content)
    except Exception as e:
        return f"⚠️ Fehler bei der Depotanalyse: {e}"

# --- 3. SPEZIALISIERTE KAUFEMPFEHLUNGEN (TAB 4) ---
def run_buy_recommendations_analysis(client, model_name, existing_holdings, news_text):
    prompt = f"""
Du bist ein quantitativer Investmentstratege. Empfehle 5 neue Qualitätsaktien/ETFs zur Erweiterung des Portfolios.
WICHTIG: Die folgenden Aktien sind BEREITS im Depot und dürfen NICHT empfohlen werden:
{existing_holdings}

[MARKTKONTEXT]
{news_text}

Gliedere deine Antwort exakt so:

### 🎯 Top 5 Kaufempfehlungen (Neu im Depot)
Empfehle 5 konkrete, qualitativ hochwertige Aktien oder ETFs:
1. **Name** (Ticker | Branche | Land)
   - **Einstiegsgrund**: Warum jetzt kaufen?
   - **Kurspotenzial**: Erwartete Rendite auf 12 Monate.
   - **Risikofaktor**: Größte Gefahr für die These.
(Wiederhole das für 2., 3., 4. und 5.)

### 🔴 Aktuell meiden
Nenne 3 konkrete Branchen oder Marktsegmente mit erhöhtem Abwärtsrisiko und begründe dies kurz.
"""
    target_model = model_name if model_name and not any(bad in model_name.lower() for bad in ["orpheus", "deepseek", "r1"]) else DEFAULT_MODEL
    try:
        res = client.chat.completions.create(
            model=target_model,
            messages=[
                {"role": "system", "content": "Du bist ein führender deutscher Börsenstratege. Antworte auf Deutsch."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=2500,
            timeout=40.0
        )
        return clean_ai_output(res.choices[0].message.content)
    except Exception as e:
        return f"⚠️ Fehler bei Kaufempfehlungen: {e}"

# --- 4. SPEZIALISIERTES RISIKOGUTACHTEN (TAB 6) ---
def run_cluster_analysis(client, model_name, cluster_context):
    prompt = f"""
Du bist ein Experte für Portfolio-Risikomanagement. Erstelle ein Risikogutachten zur Diversifikation auf Deutsch.

[PORTFOLIO-STRUKTUR (Sektoren, Länder, Rollen, Werte)]
{cluster_context}

Gliedere deine Antwort wie folgt:

### 🛡️ KI-Risikogutachten & Klumpenrisiken
- **Risiko-Score**: Vergib eine Note von 1 (Sehr konservativ) bis 10 (Sehr spekulativ) mit kurzer Einordnung.
- **Klumpenrisiko-Analyse**: Wo ist das Portfolio über- oder untergewichtet (z. B. Tech-Übergewicht, USA-Dominanz)?
- **Konkreter Absicherungs-Tipp**: Welche konkrete defensive Maßnahme (z. B. Gold, Anleihen, Puts oder defensive Dividendenwerte) stabilisiert dieses Depot am besten?
"""
    target_model = model_name if model_name and not any(bad in model_name.lower() for bad in ["orpheus", "deepseek", "r1"]) else DEFAULT_MODEL
    try:
        res = client.chat.completions.create(
            model=target_model,
            messages=[
                {"role": "system", "content": "Du bist ein präziser deutscher Risikoanalyst. Antworte ausschließlich auf Deutsch."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1500,
            timeout=30.0
        )
        return clean_ai_output(res.choices[0].message.content)
    except Exception as e:
        return f"⚠️ Fehler beim Risikogutachten: {e}"

# --- 5. 1-VS-1 DUELL (TAB 7) ---
def run_duel_analysis(client, model_name, stock_a_info, stock_b_info):
    prompt = f"""
Vergleiche als quantitativer Analyst diese beiden Aktien objektiv und detailliert auf Deutsch:
Aktie A: {stock_a_info}
Aktie B: {stock_b_info}

1. Fundamentaler Kennzahlenvergleich (KGV, Dividende, Fair Value, Burggraben)
2. Klares Fazit: Welche Aktie ist aktuell der bessere Kauf und warum?
"""
    duel_model = model_name if model_name and not any(bad in model_name.lower() for bad in ["orpheus", "deepseek", "r1"]) else DEFAULT_MODEL
    try:
        res = client.chat.completions.create(
            model=duel_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1000,
            timeout=25.0
        )
        return clean_ai_output(res.choices[0].message.content)
    except Exception as e:
        return f"⚠️ Duell-Analyse Fehler: {str(e)}"

# Kompatibilität für ältere Gesamtaufrufe
def run_analysis(client, model_name, news_text, metrics_summary, ticker_news_text="", cluster_context=""):
    m = run_market_news_analysis(client, model_name, news_text)
    d = run_depot_analysis(client, model_name, metrics_summary)
    s = run_buy_recommendations_analysis(client, model_name, "", news_text)
    c = run_cluster_analysis(client, model_name, cluster_context)
    return m, d, s, c
