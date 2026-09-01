import streamlit as st
from groq import Groq
import re

DEFAULT_MODEL = "openai/gpt-oss-120b"

@st.cache_data(ttl=3600)
def get_account_models(api_key):
    if not api_key:
        return [DEFAULT_MODEL]
    try:
        c = Groq(api_key=api_key.strip())
        models_data = c.models.list().data
        valid_models = [
            m.id for m in models_data 
            if not any(x in m.id.lower() for x in ["whisper", "guard", "vision", "safeguard", "orpheus", "tts"])
        ]
        preferred = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.6-27b"]
        sorted_models = [m for m in preferred if m in valid_models] + [m for m in valid_models if m not in preferred]
        return sorted_models if sorted_models else [DEFAULT_MODEL]
    except Exception:
        return [DEFAULT_MODEL, "openai/gpt-oss-20b", "qwen/qwen3.6-27b"]

def run_analysis(client, model_name, news_text, metrics_summary, ticker_news_text="", cluster_context=""):
    combined_prompt = f"""
Du bist ein renommierter quantitativer Chef-Anlagestratege. Analysiere die aktuellen Weltnachrichten sowie das bestehende Depot des Nutzers und erstelle fundierte Empfehlungen auf Deutsch.

[AKTUELLE WELT- & WIRTSCHAFTSNACHRICHTEN]
{news_text[:1200]}

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

    models_to_try = [model_name, "openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.6-27b"]
    seen = set()
    models_to_try = [x for x in models_to_try if not (x in seen or seen.add(x))]

    full_text = ""
    last_err_msg = ""

    for target_model in models_to_try:
        try:
            res = client.chat.completions.create(
                model=target_model,
                messages=[
                    {"role": "system", "content": "Du bist ein führender Börsen- und Finanzanalyst. Antworte auf Deutsch und verwende exakt die Trennmarker ===MARKT===, ===DEPOT===, ===SIGNALE=== und ===KLUMPEN===."},
                    {"role": "user", "content": combined_prompt}
                ],
                temperature=0.2,
                max_tokens=2500,
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

    # Unfehlbare Normalisierung: Bereinigt alle Varianten der Trennzeilen
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

    # Fallbacks: Niemals wieder "Keine Daten." anzeigen
    if not out_market:
        out_market = full_text
    if not out_depot:
        out_depot = "Statusbericht liegt vor:\n\n" + full_text[:600]
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
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=600,
        )
        return res.choices[0].message.content
    except Exception:
        res = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=600,
        )
        return res.choices[0].message.content
