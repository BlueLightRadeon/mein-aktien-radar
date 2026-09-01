import streamlit as st
from groq import Groq

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
        preferred = ["llama-3.3-70b-versatile", "llama-3.1-70b-versatile", "mixtral-8x7b-32768", "gemma2-9b-it"]
        sorted_models = [m for m in preferred if m in valid_models] + [m for m in valid_models if m not in preferred]
        return sorted_models if sorted_models else [DEFAULT_MODEL]
    except Exception:
        return [DEFAULT_MODEL, "llama-3.1-70b-versatile"]

def run_analysis(client, model_name, news_text, metrics_summary, ticker_news_text="", cluster_context=""):
    combined_prompt = f"""
Du bist ein renommierter Chef-Anlagestratege und quantitativer Portfoliomanager. 
Analysiere die aktuellen Weltnachrichten sowie das bestehende Depot des Nutzers und erstelle fundierte, konkrete Kaufempfehlungen für NEUE Aktien zur Portfolio-Erweiterung.

[AKTUELLE WELT- & MARKTNACHRICHTEN]
{news_text[:1500]}

[BESTEHENDE DEPOT-WERTE DES NUTZERS]
{metrics_summary}

[AKTUELLE DEPOT-AUFTEILUNG]
{cluster_context}

Erstelle deine Antwort strikt getrennt nach diesen Markern:

===MARKT===
### 🌍 TOP 10 Marktnachrichten
10 prägnante Stichpunkte zur aktuellen weltweiten Wirtschaftslage.

### 🧭 Gesamtstimmung der Börse
🟢 Optimistisch, 🟡 Neutral oder 🔴 Vorsichtig mit kurzer Begründung.

===DEPOT===
Kurzer Statuscheck zu den bestehenden 8 Werten des Nutzers:
- **[Unternehmen]**: Aktuelle Bewertung, Trend und worauf man jetzt achten muss.

===SIGNALE===
### 🎯 KI-Empfehlungen: NEUE Aktien zur Portfolio-Erweiterung
Empfiehl mindestens 10 konkrete, starke NEUE Aktien oder ETFs (NICHT aus dem aktuellen Depot), die das Depot des Nutzers perfekt ergänzen und Risiken ausgleichen:

1. **[Aktienname / ETF]** (Ticker | Branche | Land)
   - **Warum JETZT kaufen?** (Konkreter Treiber basierend auf Nachrichten & Bewertung)
   - **Rolle im Portfolio:** (z. B. Defensiver Cashflow, Profiteur hoher Zinsen, Basiskonsum)
   - **Faires Kursziel / Potenzial:** z. B. +15 % bis +25 %
2. **[Aktienname / ETF]** (Ticker | Branche | Land)
   - **Warum JETZT kaufen?**
   - **Rolle im Portfolio:**
   - **Faires Kursziel / Potenzial:**
3. **[Aktienname / ETF]** (Ticker | Branche | Land)
   - **Warum JETZT kaufen?**
   - **Rolle im Portfolio:**
   - **Faires Kursziel / Potenzial:**
4. **[Aktienname / ETF]** (Ticker | Branche | Land)
   - **Warum JETZT kaufen?**
   - **Rolle im Portfolio:**
   - **Faires Kursziel / Potenzial:**
5. **[Aktienname / ETF]** (Ticker | Branche | Land)
   - **Warum JETZT kaufen?**
   - **Rolle im Portfolio:**
   - **Faires Kursziel / Potenzial:**
6. **[Aktienname / ETF]** (Ticker | Branche | Land)
   - **Warum JETZT kaufen?**
   - **Rolle im Portfolio:**
   - **Faires Kursziel / Potenzial:**
7. **[Aktienname / ETF]** (Ticker | Branche | Land)
   - **Warum JETZT kaufen?**
   - **Rolle im Portfolio:**
   - **Faires Kursziel / Potenzial:**
8. **[Aktienname / ETF]** (Ticker | Branche | Land)
   - **Warum JETZT kaufen?**
   - **Rolle im Portfolio:**
   - **Faires Kursziel / Potenzial:**
9. **[Aktienname / ETF]** (Ticker | Branche | Land)
   - **Warum JETZT kaufen?**
   - **Rolle im Portfolio:**
   - **Faires Kursziel / Potenzial:**
10. **[Aktienname / ETF]** (Ticker | Branche | Land)
    - **Warum JETZT kaufen?**
    - **Rolle im Portfolio:**
    - **Faires Kursziel / Potenzial:**

#### 🔴 AKTUELL MEIDEN (Branchen mit hohem Risiko):
Nenne 4 Branchen oder Aktienarten, die man aktuell meiden sollte.

===KLUMPEN===
1. **Risiko-Score**: 1 (Sehr gut gestreut) bis 10 (Hohes Klumpenrisiko).
2. **Erläuterung zur Streuung**: Wo liegen aktuell die Übergewichte?
3. **Konkreter Erweiterungsplan**: Welche 2-3 der oben empfohlenen neuen Aktien das Gesamtrisiko am schnellsten senken.
"""

    models_to_try = [model_name] if model_name == DEFAULT_MODEL else [model_name, DEFAULT_MODEL]
    full_text = ""
    last_err = None

    for target_model in models_to_try:
        try:
            res = client.chat.completions.create(
                model=target_model,
                messages=[{"role": "user", "content": combined_prompt}],
                temperature=0.2,
                max_tokens=2200,
            )
            full_text = res.choices[0].message.content
            if full_text:
                break
        except Exception as e:
            last_err = e
            continue

    if not full_text:
        raise last_err if last_err else RuntimeError("Keine Antwort von der KI-Schnittstelle erhalten.")

    out_market = "Keine Daten."
    out_depot = "Keine Daten."
    out_signals = "Keine Daten."
    out_cluster = "Keine Daten."

    if "===MARKT===" in full_text:
        parts = full_text.split("===MARKT===")[1]
        if "===DEPOT===" in parts:
            out_market, rest = parts.split("===DEPOT===", 1)
            if "===SIGNALE===" in rest:
                out_depot, rest2 = rest.split("===SIGNALE===", 1)
                if "===KLUMPEN===" in rest2:
                    out_signals, out_cluster = rest2.split("===KLUMPEN===", 1)
                else:
                    out_signals = rest2
            else:
                out_depot = rest
        else:
            out_market = parts

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
