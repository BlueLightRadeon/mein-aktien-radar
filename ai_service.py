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
    """Führt die Gesamtanalyse in einfacher, verständlicher Sprache durch."""
    
    combined_prompt = f"""
Du bist ein verständlicher, sympathischer Finanzberater. Erkläre alles so, dass es auch ein absoluter Börsen-Neuling (Laie) sofort versteht.
Verzichte auf unnötiges Fachchinesisch. Wenn du Fachbegriffe wie KGV oder RSI nennst, erkläre in einem kurzen Nebensatz, was das bedeutet (z. B. "KGV von 25 = mäßig teuer bewertet" oder "RSI über 70 = Aktie ist kurzfristig heißgelaufen").

Hier sind die aktuellen Daten:
[GLOBALE NACHRICHTEN]
{news_text}

[DEPOT-KENNZAHLEN]
{metrics_summary}

[UNTERNEHMENS-NEWS]
{ticker_news_text}

[ALLOKATION & BRANCHEN]
{cluster_context}

Erstelle die Analyse strikt aufgeteilt mit diesen 4 Trennmarkern:

===MARKT===
1. **Die TOP 10 wichtigsten Welt- und Wirtschaftsnachrichten**: Einfach und verständlich formuliert.
2. **Gesamt-Börsenstimmung**: 🟢 Optimistisch (Gute Laune an den Börsen), 🟡 Abwartend oder 🔴 Ängstlich/Vorsichtig mit kurzer Begründung.

===DEPOT===
Analysiere JEDE Aktie einzeln in einfacher Sprache:
- **[Aktienname]**: Stimmung (🟢/🟡/🔴), Preisschild-Einschätzung (Ist sie gerade günstig oder teuer?) und Chart-Zustand (Läuft sie gut oder schwächelt sie?).
- **Was jetzt wichtig ist**: Worauf du in den nächsten Tagen achten solltest.

===SIGNALE===
Erstelle für JEDE Aktie eine klare Empfehlung:
- **[Aktienname]**: 🟢 **KAUFEN** / 🟡 **HALTEN** / 🔴 **VERKAUFEN**
- **Einfache Begründung**: Warum diese Entscheidung Sinn macht.
- **Risiko**: Gering / Mittel / Hoch | **Empfohlene Anlagedauer**: Eher kurzfristig oder langfristig liegenlassen.

===KLUMPEN===
1. **Risiko-Note**: 1 (Sehr sicher und breit verteilt) bis 10 (Gefährlich einseitig).
2. **Einseitigkeiten**: Hängt das Depot zu stark an einem Land (z. B. nur USA) oder einer Branche (z. B. nur Technik)?
3. **Einfacher Tipp**: Welche Branche oder Absicherung würde dem Depot guttun?
"""

    res = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": combined_prompt}],
        temperature=0.2,
        max_tokens=2200,
    )
    
    full_text = res.choices[0].message.content
    
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
Vergleiche diese beiden Aktien in einfachen Worten für einen Laien:
Aktie A: {stock_a_info}
Aktie B: {stock_b_info}

1. Wo liegen die Stärken und Schwächen im Vergleich?
2. Klares Urteil: Welche Aktie ist aktuell die schlauere Wahl und warum?
"""
    res = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=600,
    )
    return res.choices[0].message.content
