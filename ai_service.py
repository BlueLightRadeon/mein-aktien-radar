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
    combined_prompt = f"""
Du bist ein quantitativer Chef-Aktienanalyst und Portfoliomanager. Erstelle eine detaillierte, vollständige Analyse auf Deutsch. 
Formuliere alle Analysen direkt und ohne Platzhalter aus.

AKTUELLE NACHRICHTENLAGE:
{news_text}

BESTEHENDE DEPOTWERTE & HISTORISCHE DATEN DES NUTZERS:
{metrics_summary}

DEPOT-STRUKTUR:
{cluster_context}

Gliedere deine Antwort zwingend anhand der folgenden vier Trennmarkierungen:

===MARKT===
### 🌍 TOP 10 Marktnachrichten
Formuliere 10 prägnante Stichpunkte zu globalen Leitbörsen, Notenbank-Zinsentscheiden, Tech-Investitionen und geopolitischen Einflussfaktoren.

### 🧭 Gesamtstimmung der Börse
Bewerte die Lage eindeutig mit 🟢 Optimistisch, 🟡 Neutral oder 🔴 Vorsichtig und begründe dies in 3-4 Sätzen.

===DEPOT===
### 💼 KAUF- & VERKAUFSEMPFEHLUNGEN FÜR DEINE BESTEHENDEN AKTIEN (KOMBINIERTE GESAMTANALYSE)
Analysiere JEDE im Depot gehaltene Position ausführlich mit dem direkten Vorher-Nachher-Vergleich:

Für jedes Unternehmen folgendes Format anwenden:
---
#### 📌 [Unternehmensname] ([Ticker]) | Wert im Depot: [Wert] €
- **🟢/🟡/🔴 Aktuelle Empfehlung:** [KAUFEN / AUFSTOCKEN / HALTEN / VERKAUFEN]
  - **Aktuelle Begründung:** Fundamentale Einschätzung (KGV, Fair Value, Auftragslage, Kurspotenzial).
- **⏱️ Rückblick vor 3 Monaten:** [Damalige Empfehlung z. B. HALTEN]
  - **Damalige Ausgangslage:** Welche Faktoren damals maßgeblich waren.
- **📈 Trend & Fazit:** Wie sich die These in den letzten 3 Monaten entwickelt hat und welcher konkrete Schritt jetzt empfohlen wird.

===SIGNALE===
### 🎯 TOP 5 NEUE KAUF-EMPFEHLUNGEN (Zur Portfolio-Erweiterung)
Empfehle 5 konkrete, kaufenswerte Qualitätsaktien oder ETFs zur Diversifikation (keine Werte, die schon im Depot liegen):
1. **[Aktie 1]** (Ticker | Branche | Land)
   - **Warum JETZT kaufen?**
   - **Chance / Kurspotenzial:**
   - **Risiko:** Gering / Mittel / Hoch
2. **[Aktie 2]** (Ticker | Branche | Land)
   - **Warum JETZT kaufen?**
   - **Chance / Kurspotenzial:**
   - **Risiko:**
3. **[Aktie 3]** (Ticker | Branche | Land)
   - **Warum JETZT kaufen?**
   - **Chance / Kurspotenzial:**
   - **Risiko:**
4. **[Aktie 4]** (Ticker | Branche | Land)
   - **Warum JETZT kaufen?**
   - **Chance / Kurspotenzial:**
   - **Risiko:**
5. **[Aktie 5]** (Ticker | Branche | Land)
   - **Warum JETZT kaufen?**
   - **Chance / Kurspotenzial:**
   - **Risiko:**

#### 🔴 AKTUELL MEIDEN
Nenne 3 Branchen oder Nischenwerte mit erhöhtem Abwärtsrisiko.

===KLUMPEN===
### 🛡️ Risikostreuung & Depot-Optimierung
1. **Risiko-Score**: 1 (sehr defensiv) bis 10 (sehr spekulativ).
2. **Bewertung der aktuellen Schwerpunkte** (Tech, Rüstung, Healthcare etc.).
3. **Konkrete Portfolio-Empfehlung** zur weiteren Absicherung.
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
                max_tokens=3200,
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
