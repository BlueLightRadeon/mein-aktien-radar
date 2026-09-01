import streamlit as st
from groq import Groq
import re

DEFAULT_MODEL = "llama-3.3-70b-versatile"

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
        preferred = ["llama-3.3-70b-versatile", "openai/gpt-oss-120b", "openai/gpt-oss-20b", "llama-3.1-8b-instant"]
        sorted_models = [m for m in preferred if m in valid_models] + [m for m in valid_models if m not in preferred]
        return sorted_models if sorted_models else [DEFAULT_MODEL]
    except Exception:
        return [DEFAULT_MODEL, "openai/gpt-oss-120b", "llama-3.1-8b-instant"]

def extract_section(text, tag, next_tags):
    pattern = rf"(?:={2,5}\s*{tag}\s*={2,5}|\*\*\s*={2,5}\s*{tag}\s*={2,5}\s*\*\*|###\s*{tag})"
    m = re.search(pattern, text, re.IGNORECASE)
    if not m:
        return None
    start = m.end()
    
    end = len(text)
    for nt in next_tags:
        n_pattern = rf"(?:={2,5}\s*{nt}\s*={2,5}|\*\*\s*={2,5}\s*{nt}\s*={2,5}\s*\*\*|###\s*{nt})"
        nm = re.search(n_pattern, text[start:], re.IGNORECASE)
        if nm:
            end = min(end, start + nm.start())
            
    res = text[start:end].strip()
    return res if res else None

def run_analysis(client, model_name, news_text, metrics_summary, ticker_news_text="", cluster_context=""):
    combined_prompt = f"""
Du bist ein quantitativer Chef-Anlagestratege. Analysiere die aktuellen Weltnachrichten sowie das bestehende Depot des Nutzers und erstelle fundierte Empfehlungen auf Deutsch.

[AKTUELLE WELT- & WIRTSCHAFTSNACHRICHTEN]
{news_text[:1200]}

[BESTEHENDE DEPOT-WERTE DES NUTZERS]
{metrics_summary}

[DEPOT-AUFTEILUNG]
{cluster_context}

Antworte strukturiert und verwende exakt diese Trennmarken:

===MARKT===
### 🌍 TOP 10 Marktnachrichten
10 prägnante Stichpunkte zur aktuellen weltweiten Wirtschaftslage.

### 🧭 Gesamtstimmung der Börse
🟢 Optimistisch, 🟡 Neutral oder 🔴 Vorsichtig mit kurzer Begründung.

===DEPOT===
Kurzer Statusbericht zu den bestehenden Aktien des Nutzers:
- **[Unternehmen]**: Bewertung, Trend und worauf man aktuell achten muss.

===SIGNALE===
### 🎯 TOP 5 KAUF-EMPFEHLUNGEN (Stand Jetzt zur Portfolio-Erweiterung)
Wähle basierend auf den Nachrichten genau 5 konkrete Qualitätsaktien/ETFs (NICHT aus dem aktuellen Bestand), die das Depot ideal ergänzen:

1. **[Aktie 1]** (Ticker | Branche | Land)
   - **Warum JETZT kaufen?** (Konkreter Treiber basierend auf Zinsen, Geopolitik oder Tech-Trends)
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
1. **Risiko-Score**: 1 (Sehr gut gestreut) bis 10 (Hohes Risiko).
2. **Erläuterung der Streuung**: Wo liegen aktuell die Schwerpunkte?
3. **Erweiterungs-Tipp**: Welche der oben empfohlenen 5 Aktien das Depot am besten absichert.
"""

    models_to_try = [model_name, "llama-3.3-70b-versatile", "openai/gpt-oss-120b", "llama-3.1-8b-instant"]
    seen = set()
    models_to_try = [x for x in models_to_try if not (x in seen or seen.add(x))]
    
    full_text = ""
    last_err = None

    for target_model in models_to_try:
        try:
            res = client.chat.completions.create(
                model=target_model,
                messages=[
                    {"role": "system", "content": "Du bist ein präziser Finanz- und Aktienanalyst. Halte dich exakt an die geforderten Trennmarken."},
                    {"role": "user", "content": combined_prompt}
                ],
                temperature=0.2,
                max_tokens=2200,
            )
            if res.choices and res.choices[0].message and res.choices[0].message.content:
                full_text = res.choices[0].message.content
                break
        except Exception as e:
            last_err = e
            continue

    if not full_text:
        err_msg = str(last_err) if last_err else "Unbekannter API-Fehler"
        return (
            f"⚠️ **Fehler bei Groq API:** `{err_msg}`\n\nBitte prüfe deinen `GROQ_API_KEY` in den Streamlit Secrets.",
            "Keine Daten erhalten.",
            "Keine Empfehlungen erhalten.",
            "Keine Risikobewertung erhalten."
        )

    out_market = extract_section(full_text, "MARKT", ["DEPOT", "SIGNALE", "KLUMPEN"]) or full_text[:600]
    out_depot = extract_section(full_text, "DEPOT", ["SIGNALE", "KLUMPEN"]) or "Aktienbewertung aktualisiert."
    out_signals = extract_section(full_text, "SIGNALE", ["KLUMPEN"]) or "Top-5-Kaufempfehlungen generiert."
    out_cluster = extract_section(full_text, "KLUMPEN", []) or "Risikobewertung berechnet."

    return out_market.strip(), out_depot.strip(), out_signals.strip(), out_cluster.strip()

def run_duel_analysis(client, model_name, stock_a_info, stock_b_info):
    prompt = f"""
Vergleiche als Analyst diese beiden Aktien:
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
