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
        preferred = ["llama-3.3-70b-versatile", "llama-3.1-70b-versatile", "mixtral-8x7b-32768"]
        sorted_models = [m for m in preferred if m in valid_models] + [m for m in valid_models if m not in preferred]
        return sorted_models if sorted_models else [DEFAULT_MODEL]
    except Exception:
        return [DEFAULT_MODEL, "llama-3.1-70b-versatile"]

def run_analysis(client, model_name, news_text, metrics_summary, ticker_news_text="", cluster_context=""):
    combined_prompt = f"""
Du bist ein quantitativer Chef-Anlagestratege. Analysiere die aktuellen Weltnachrichten sowie das bestehende Depot des Nutzers und erstelle fundierte Empfehlungen auf Deutsch.

[AKTUELLE WELT- & WIRTSCHAFTSNACHRICHTEN]
{news_text[:1200]}

[BESTEHENDE DEPOT-WERTE DES NUTZERS]
{metrics_summary}

[DEPOT-AUFTEILUNG]
{cluster_context}

Erstelle deine Antwort strikt mit diesen 4 Abschnitten:

[MARKT_SECTION]
### 🌍 TOP 10 Marktnachrichten
10 prägnante Stichpunkte zur aktuellen globalen Wirtschaftslage.

### 🧭 Gesamtstimmung der Börse
🟢 Optimistisch, 🟡 Neutral oder 🔴 Vorsichtig mit kurzer Begründung.

[DEPOT_SECTION]
### 💼 Statusbericht zu deinen Depot-Positionen
Analysiere die bestehenden Aktien des Nutzers:
- **[Unternehmen]**: Aktuelle Bewertung, Trend und worauf man jetzt achten muss.

[SIGNALE_SECTION]
### 🎯 TOP 5 KAUF-EMPFEHLUNGEN (Stand Jetzt zur Portfolio-Erweiterung)
Wähle basierend auf der Marktlage genau 5 konkrete Qualitätsaktien/ETFs (NICHT aus dem aktuellen Bestand), die das Depot ideal ergänzen:

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

[KLUMPEN_SECTION]
### 🛡️ Risikostreuung & Depot-Optimierung
1. **Risiko-Score**: 1 (Sehr gut gestreut) bis 10 (Hohes Risiko).
2. **Erläuterung der Streuung**: Wo liegen aktuell die Schwerpunkte?
3. **Erweiterungs-Tipp**: Welche der oben empfohlenen 5 Aktien das Depot am besten absichert.
"""

    models_to_try = [model_name, "llama-3.3-70b-versatile", "llama-3.1-70b-versatile"]
    seen = set()
    models_to_try = [x for x in models_to_try if not (x in seen or seen.add(x))]
    
    full_text = ""
    for target_model in models_to_try:
        try:
            res = client.chat.completions.create(
                model=target_model,
                messages=[
                    {"role": "system", "content": "Du bist ein führender Börsen- und Finanzanalyst. Antworte immer auf Deutsch und strukturiert."},
                    {"role": "user", "content": combined_prompt}
                ],
                temperature=0.2,
                max_tokens=2500,
            )
            if res.choices and res.choices[0].message and res.choices[0].message.content:
                full_text = res.choices[0].message.content
                break
        except Exception:
            continue

    if not full_text:
        return (
            "⚠️ Keine Antwort von der Groq-API erhalten. Bitte prüfe deinen API-Key.",
            "Keine Depot-Daten verfügbar.",
            "Keine Kaufempfehlungen verfügbar.",
            "Keine Risikodaten verfügbar."
        )

    # Intelligenter Parser mit Fallbacks: Zerteilt nach den Tags oder teilt den Text sinnvoll auf
    out_market = ""
    out_depot = ""
    out_signals = ""
    out_cluster = ""

    if "[MARKT_SECTION]" in full_text:
        parts = full_text.split("[MARKT_SECTION]")
        rest1 = parts[1] if len(parts) > 1 else parts[0]
        
        if "[DEPOT_SECTION]" in rest1:
            out_market, rest2 = rest1.split("[DEPOT_SECTION]", 1)
            if "[SIGNALE_SECTION]" in rest2:
                out_depot, rest3 = rest2.split("[SIGNALE_SECTION]", 1)
                if "[KLUMPEN_SECTION]" in rest3:
                    out_signals, out_cluster = rest3.split("[KLUMPEN_SECTION]", 1)
                else:
                    out_signals = rest3
            else:
                out_depot = rest2
        else:
            out_market = rest1
    else:
        # Fallback: Falls die KI die Tags weggelassen hat, teilen wir den Text proportional auf
        chunks = full_text.split("\n\n")
        total_chunks = len(chunks)
        c1 = total_chunks // 4
        c2 = total_chunks // 2
        c3 = (total_chunks * 3) // 4
        out_market = "\n\n".join(chunks[:c1]) if c1 > 0 else full_text
        out_depot = "\n\n".join(chunks[c1:c2]) if c2 > c1 else "Depot-Auswertung liegt vor."
        out_signals = "\n\n".join(chunks[c2:c3]) if c3 > c2 else "Top-5-Kaufempfehlungen werden berechnet."
        out_cluster = "\n\n".join(chunks[c3:]) if total_chunks > c3 else "Risikoanalyse liegt vor."

    return (
        out_market.strip() or "Marktdaten geladen.",
        out_depot.strip() or "Depot-Bewertung geladen.",
        out_signals.strip() or "Kaufempfehlungen geladen.",
        out_cluster.strip() or "Risikostreuung berechnet."
    )

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
