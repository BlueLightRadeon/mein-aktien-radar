import streamlit as st
from groq import Groq

@st.cache_data(ttl=3600)
def get_account_models(api_key):
    try:
        c = Groq(api_key=api_key.strip())
        models_list = [m.id for m in c.models.list().data]
        text_models = [m for m in models_list if not any(x in m.lower() for x in ["whisper", "guard", "orpheus", "vision", "safeguard"])]
        return text_models if text_models else models_list
    except Exception:
        return ["llama-3.3-70b-versatile", "openai/gpt-oss-120b"]

def run_analysis(client, model_name, news_text, metrics_summary, ticker_news_text, cluster_context=""):
    """Führt alle 4 Auswertungen in einem einzigen, token-sparenden API-Aufruf durch."""
    
    combined_prompt = f"""
Du bist ein professioneller Chefanyst für Aktienmärkte und Portfolios. Antworte AUSSCHLIESSLICH auf Deutsch.

Hier sind die aktuellen Daten:
[GLOBALE NACHRICHTEN]
{news_text}

[DEPOT-KENNZAHLEN]
{metrics_summary}

[UNTERNEHMENS-NEWS]
{ticker_news_text}

[ALLOKATION & BRANCHEN]
{cluster_context}

Erstelle die Gesamtanalyse strikt aufgeteilt mit den folgenden 4 Trennmarkern:

===MARKT===
1. **TOP 10 Markt-Informationen**: Die 10 wichtigsten Makro- & Börsenfakten stichpunktartig.
2. **Gesamtstimmung**: 🟢 Bullisch, 🟡 Neutral oder 🔴 Bärisch (inkl. kurzer Begründung).

===DEPOT===
Analysiere JEDE Aktie aus dem Depot einzeln:
- **[Aktienname]**: Sentiment (🟢/🟡/🔴), Einordnung der Technik (RSI) und fundamentale Bewertung (KGV/Fair Value).
- **Ausblick**: Wichtige Treiber für die nächsten Tage.

===SIGNALE===
Erstelle für JEDE Aktie eine prägnante Handlungsempfehlung:
- **[Aktienname]**: 🟢 **KAUFEN** / 🟡 **HALTEN** / 🔴 **VERKAUFEN**
- **Begründung**: (Kombination aus RSI, KGV, Fair Value & Kursziel)
- **Risiko**: Gering / Mittel / Hoch | **Horizont**: Kurzfristig / Mittelfristig / Langfristig

===KLUMPEN===
1. **Risiko-Score**: 1 bis 10 (Klumpenrisiko-Bewertung).
2. **Kritische Übergewichte**: Wo bestehen einseitige Branchen- oder Länder-Abhängigkeiten?
3. **Absicherungs-Tipps**: 1-2 konkrete Vorschläge zur Diversifikation.
"""

    res = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": combined_prompt}],
        temperature=0.2,
        max_tokens=2200,
    )
    
    full_text = res.choices[0].message.content
    
    # Text sauber anhand der Marker auf die Tabs aufteilen
    out_market = "Keine Marktdaten."
    out_depot = "Keine Depotanalyse."
    out_signals = "Keine Signale."
    out_cluster = "Keine Klumpenanalyse."
    
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
Vergleiche diese beiden Aktien direkt auf Deutsch:
Aktie A: {stock_a_info}
Aktie B: {stock_b_info}

1. Stärken & Schwächen (KGV, RSI, Fair Value, Kursziel)
2. Klares Duell-Urteil: Welche Aktie ist aktuell der bessere Kauf?
"""
    res = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=600,
    )
    return res.choices[0].message.content
