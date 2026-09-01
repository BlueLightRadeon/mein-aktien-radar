import streamlit as st
from groq import Groq
import re

DEFAULT_MODEL = "openai/gpt-oss-120b"
STATIC_MODELS = ["openai/gpt-oss-120b", "qwen/qwen3.6-27b", "openai/gpt-oss-20b"]

def get_account_models(api_key):
    # Statische Liste: Verhindert jedes Blockieren beim Laden der Seite
    return STATIC_MODELS

def run_analysis(client, model_name, news_text, metrics_summary, ticker_news_text="", cluster_context=""):
    combined_prompt = f"""
Du bist ein renommierter quantitativer Chef-Anlagestratege. Analysiere die aktuellen Weltnachrichten sowie das bestehende Depot des Nutzers und erstelle fundierte Empfehlungen auf Deutsch.

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
### 💼 Statusbericht zu deinen Depot-Positionen
Analysiere die bestehenden 8 Aktien des Nutzers:
- **[Unternehmen]**: Aktuelle Bewertung, Trend und worauf man jetzt achten muss.

===SIGNALE===
### 🎯 TOP 5 KAUF-EMPFEHLUNGEN (Stand Jetzt zur Portfolio-Erweiterung)
Wähle basierend auf der aktuellen Lage genau 5 konkrete Qualitätsaktien oder ETFs (NICHT aus dem aktuellen Bestand), die das Depot ideal ergänzen:

1. **[Aktie 1]** (Ticker | Branche | Land)
   - **Warum JETZT kaufen?** (Konkreter Treiber: z. B. Energie-Infrastruktur, Zinsgewinner, Rüstung, Basiskonsum)
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

#### 🔴 AKTUELL MEIDEN (Verlierer der aktuellen Marktlage):
Nenne 3-4 Branchen oder Aktienarten mit erhöhtem Risiko.

===KLUMPEN===
### 🛡️ Risikostreuung & Depot-Optimierung
1. **Risiko-Score**: 1 (Sehr gut gestreut) bis 10 (Hohes Risiko).
2. **Erläuterung der Streuung**: Wo liegen aktuell die Schwerpunkte?
3. **Erweiterungs-Tipp**: Welche der oben empfohlenen 5 Aktien das Depot am besten absichert.
"""

    target = model_name if model_name in STATIC_MODELS else DEFAULT_MODEL
    full_text = ""
    last_err_msg = ""

    # Schneller Request mit festem Timeout (max. 10 Sek)
    for model_to_try in [target, "openai/gpt-oss-20b", "qwen/qwen3.6-27b"]:
        try:
            res = client.chat.completions.create(
                model=model_to_try,
                messages=[
                    {"role": "system", "content": "Du bist ein führender Börsen- und Finanzanalyst. Antworte auf Deutsch und verwende exakt die Trennmarker ===MARKT===, ===DEPOT===, ===SIGNALE=== und ===KLUMPEN===."},
                    {"role": "user", "content": combined_prompt}
                ],
                temperature=0.2,
                max_tokens=2000,
                timeout=10.0
            )
            if res.choices and res.choices[0].message and res.choices[0].message.content:
                full_text = res.choices[0].message.content
                break
        except Exception as e:
            last_err_msg = str(e)
            continue

    if not full_text:
        err_display = f"⚠️ **Groq API Fehler:** `{last_err_msg}`\n\nBitte prüfe deinen API-Key in den Streamlit Secrets."
        return err_display, err_display, err_display, err_display

    # Zerlegung via Regex
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

    out_market = sections.get("SECTION_MARKT", "")
    out_depot = sections.get("SECTION_DEPOT", "")
    out_signals = sections.get("SECTION_SIGNALE", "")
    out_cluster = sections.get("SECTION_KLUMPEN", "")

    if not out_market:
        out_market = full_text
    if not out_depot:
        out_depot = "Statusbericht liegt vor:\n\n" + full_text[:500]
    if not out_signals:
        out_signals = full_text
    if not out_cluster:
        out_cluster = "Die Risikobewertung findest du im Tab '🌍 Nachrichten'."

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
            timeout=8.0
        )
        return res.choices[0].message.content
    except Exception:
        return "Duell-Analyse konnte nicht geladen werden."
