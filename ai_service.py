import streamlit as st
from groq import Groq
import re

DEFAULT_MODEL = "llama-3.3-70b-versatile"
STATIC_MODELS = [
    "llama-3.3-70b-versatile",
    "llama3-70b-8192",
    "llama3-8b-8192",
    "mixtral-8x7b-32768",
    "gemma2-9b-it"
]

def get_account_models(api_key):
    return STATIC_MODELS

def run_analysis(client, model_name, news_text, metrics_summary, ticker_news_text="", cluster_context=""):
    combined_prompt = f"""
Du bist ein quantitativer Chef-Anlagestratege. Analysiere das Depot und die aktuellen Marktnachrichten auf Deutsch.

[AKTUELLE WELT- & WIRTSCHAFTSNACHRICHTEN]
{news_text[:1000]}

[BESTEHENDE DEPOT-WERTE DES NUTZERS]
{metrics_summary}

[DEPOT-AUFTEILUNG]
{cluster_context}

Gliedere deine Antwort zwingend mit diesen 4 exakten Trennzeilen:

===MARKT===
### 🌍 TOP 10 Marktnachrichten
10 prägnante Stichpunkte zur aktuellen globalen Wirtschaftslage.

### 🧭 Gesamtstimmung der Börse
🟢 Optimistisch, 🟡 Neutral oder 🔴 Vorsichtig mit kurzer Begründung.

===DEPOT===
### 💼 KAUF- & VERKAUFSEMPFEHLUNGEN FÜR DEINE BESTEHENDEN AKTIEN
Analysiere JEDE bestehende Position des Nutzers mit klarer Handlungsanweisung:
- **[Unternehmen]**: 🟢 **KAUFEN / AUFSTOCKEN**, 🟡 **HALTEN** oder 🔴 **GEWINNE MITNEHMEN / VERKAUFEN**
  - *Begründung & Kursziel*: Warum diese Handlung jetzt sinnvoll ist und welches Potenzial besteht.

===SIGNALE===
### 🎯 TOP 5 NEUE KAUF-EMPFEHLUNGEN (Zur Portfolio-Erweiterung)
Wähle basierend auf der aktuellen Lage genau 5 konkrete Qualitätsaktien oder ETFs (NICHT aus dem aktuellen Bestand), die das Depot ideal ergänzen:

1. **[Aktie 1]** (Ticker | Branche | Land)
   - **Warum JETZT kaufen?**
   - **Chance / Kurspotenzial:** z. B. +15 % bis +25 %
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
Nenne 3-4 Branchen oder Aktienarten mit erhöhtem Risiko.

===KLUMPEN===
### 🛡️ Risikostreuung & Depot-Optimierung
1. **Risiko-Score**: 1 (Sehr gut gestreut) bis 10 (Hohes Risiko).
2. **Erläuterung der Streuung**: Wo liegen aktuell die Schwerpunkte?
3. **Erweiterungs-Tipp**: Welche der oben empfohlenen 5 Aktien das Depot am besten absichert.
"""

    models_to_try = [model_name, "llama-3.3-70b-versatile", "llama3-70b-8192", "llama3-8b-8192"]
    seen = set()
    models_to_try = [x for x in models_to_try if x and not (x in seen or seen.add(x))]

    full_text = ""
    last_err = None

    for target_model in models_to_try:
        try:
            res = client.chat.completions.create(
                model=target_model,
                messages=[
                    {"role": "system", "content": "Du bist ein führender Börsen- und Finanzanalyst. Antworte auf Deutsch und verwende exakt die Trennmarker ===MARKT===, ===DEPOT===, ===SIGNALE=== und ===KLUMPEN===."},
                    {"role": "user", "content": combined_prompt}
                ],
                temperature=0.2,
                max_tokens=2200,
                timeout=25.0
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
    out_depot = sections.get("SECTION_DEPOT", "Handelsempfehlungen liegen im Tab '💼 Stimmung' vor.")
    out_signals = sections.get("SECTION_SIGNALE", full_text)
    out_cluster = sections.get("SECTION_KLUMPEN", "Streuungsanalyse erstellt.")

    return out_market.strip(), out_depot.strip(), out_signals.strip(), out_cluster.strip()

def run_duel_analysis(client, model_name, stock_a_info, stock_b_info):
    prompt = f"""
Vergleiche als quantitativer Analyst diese beiden Aktien objektiv:
Aktie A: {stock_a_info}
Aktie B: {stock_b_info}

1. Kennzahlenvergleich (KGV, Dividende, Fair Value, Burggraben)
2. Klares Fazit: Welche Aktie ist aktuell der bessere Kauf?
"""
    try:
        res = client.chat.completions.create(
            model=model_name if model_name in STATIC_MODELS else DEFAULT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=600,
            timeout=15.0
        )
        return res.choices[0].message.content
    except Exception as e:
        return f"⚠️ Duell-Analyse Fehler: {str(e)}"
