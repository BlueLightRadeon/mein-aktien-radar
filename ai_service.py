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
Du bist ein quantitativer Chef-Aktienanalyst und Portfoliomanager. Erstelle eine detaillierte, vollständige Finanzanalyse auf Deutsch. 
WICHTIG: Schreibe alle Punkte direkt aus und nutze exakt die vier Trennmarker.

AKTUELLE NACHRICHTENLAGE:
{news_text}

BESTEHENDE DEPOTWERTE & HISTORISCHE DATEN DES NUTZERS:
{metrics_summary}

DEPOT-STRUKTUR:
{cluster_context}

Gliedere deine Antwort zwingend mit diesen 4 exakten Markern:

===MARKT===
### 🌍 TOP 10 Marktnachrichten
Formuliere genau 10 konkrete, aktuelle Stichpunkte zur weltweiten Konjunktur, Zinspolitik von Fed & EZB, Rohstoffen, Halbleiter- und Tech-Märkten sowie geopolitischen Risiken.

### 🧭 Gesamtstimmung der Börse
Bewerte die Börsenlage mit 🟢 Optimistisch, 🟡 Neutral oder 🔴 Vorsichtig und begründe dies in 3-4 Sätzen.

===DEPOT===
### 💼 KAUF- & VERKAUFSEMPFEHLUNGEN FÜR DEINE BESTEHENDEN AKTIEN (KOMBINIERTE GESAMTANALYSE)
Analysiere JEDE im Depot gehaltene Aktie ausnahmslos einzeln nach diesem Schema:

#### 📌 [Unternehmensname] ([Ticker])
- **🟢/🟡/🔴 Aktuelle Empfehlung:** [KAUFEN / AUFSTOCKEN / HALTEN / VERKAUFEN]
  - **Aktuelle Begründung:** Fundamentale Kennzahlen (KGV, Fair Value, Auftragslage, Kurspotenzial).
- **⏱️ Rückblick vor 3 Monaten:** [Damalige Empfehlung]
  - **Damalige Ausgangslage:** Welche Faktoren damals die Bewertung bestimmt haben.
- **📈 Trend & Fazit:** Entwicklung der Anlagethese in den letzten 3 Monaten und konkreter nächster Handlungsschritt.

===SIGNALE===
### 🎯 TOP 5 NEUE KAUF-EMPFEHLUNGEN (Zur Portfolio-Erweiterung)
Empfehle genau 5 konkrete, kaufenswerte Qualitätsaktien oder ETFs zur Diversifikation (keine Werte, die schon im Depot liegen):

1. **[Aktie 1]** (Ticker | Branche | Land)
   - **Warum JETZT kaufen?**
   - **Chance / Kurspotenzial:**
   - **Risikobewertung:** Gering / Mittel / Hoch

2. **[Aktie 2]** (Ticker | Branche | Land)
   - **Warum JETZT kaufen?**
   - **Chance / Kurspotenzial:**
   - **Risikobewertung:**

3. **[Aktie 3]** (Ticker | Branche | Land)
   - **Warum JETZT kaufen?**
   - **Chance / Kurspotenzial:**
   - **Risikobewertung:**

4. **[Aktie 4]** (Ticker | Branche | Land)
   - **Warum JETZT kaufen?**
   - **Chance / Kurspotenzial:**
   - **Risikobewertung:**

5. **[Aktie 5]** (Ticker | Branche | Land)
   - **Warum JETZT kaufen?**
   - **Chance / Kurspotenzial:**
   - **Risikobewertung:**

#### 🔴 AKTUELL MEIDEN:
Nenne 3-4 Branchen oder Aktiengruppen mit erhöhtem Abwärtsrisiko.

===KLUMPEN===
### 🛡️ Risikostreuung & Depot-Optimierung
1. **Risiko-Score**: 1 (Sehr konservativ) bis 10 (Sehr spekulativ).
2. **Erläuterung der Streuung**: Wo liegen aktuell Übergewichtungen (z. B. Tech, Rüstung)?
3. **Erweiterungs-Tipp**: Welche der oben empfohlenen 5 neuen Aktien das Depot am besten absichert.
"""

    target_model = model_name if model_name and "orpheus" not in model_name else DEFAULT_MODEL
    models_to_try = [target_model, DEFAULT_MODEL, "llama-3.3-70b-specdec"]
    seen = set()
    models_to_try = [x for x in models_to_try if x and not (x in seen or seen.add(x))]

    full_text = ""
    last_err = None

    for m_id in models_to_try:
        try:
            res = client.chat.completions.create(
                model=m_id,
                messages=[
                    {"role": "system", "content": "Du bist ein führender deutscher Börsenanalyst. Antworte immer auf Deutsch und formuliere alle 4 Abschnitte vollständig und sauber getrennt aus."},
                    {"role": "user", "content": combined_prompt}
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

    # Robuste Trennung der 4 Abschnitte
    pattern_m = re.search(r'={2,5}\s*MARKT\s*={2,5}(.*?)(?=={2,5}\s*DEPOT\s*={2,5}|$)', full_text, re.DOTALL | re.IGNORECASE)
    pattern_d = re.search(r'={2,5}\s*DEPOT\s*={2,5}(.*?)(?=={2,5}\s*SIGNALE\s*={2,5}|$)', full_text, re.DOTALL | re.IGNORECASE)
    pattern_s = re.search(r'={2,5}\s*SIGNALE\s*={2,5}(.*?)(?=={2,5}\s*KLUMPEN\s*={2,5}|$)', full_text, re.DOTALL | re.IGNORECASE)
    pattern_c = re.search(r'={2,5}\s*KLUMPEN\s*={2,5}(.*?)$', full_text, re.DOTALL | re.IGNORECASE)

    out_market = pattern_m.group(1).strip() if pattern_m else full_text[:800]
    out_depot = pattern_d.group(1).strip() if pattern_d else "Kombinierte Empfehlungen geladen."
    out_signals = pattern_s.group(1).strip() if pattern_s else "Kaufempfehlungen werden berechnet..."
    out_cluster = pattern_c.group(1).strip() if pattern_c else "Risikostreuung erstellt."

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
