import streamlit as st
from groq import Groq

@st.cache_data(ttl=3600)
def get_account_models(api_key):
    try:
        c = Groq(api_key=api_key.strip())
        models_list = [m.id for m in c.models.list().data]
        # Priorisiere starke Reasoning- und Analyse-Modelle
        preferred = [
            "deepseek-r1-distill-llama-70b",
            "llama-3.3-70b-versatile",
            "llama-3.1-70b-versatile",
            "mixtral-8x7b-32768"
        ]
        available_preferred = [m for m in preferred if m in models_list]
        other_text_models = [
            m for m in models_list
            if m not in available_preferred and not any(x in m.lower() for x in ["whisper", "guard", "vision", "safeguard"])
        ]
        return available_preferred + other_text_models if available_preferred else models_list
    except Exception:
        return ["deepseek-r1-distill-llama-70b", "llama-3.3-70b-versatile"]

def run_analysis(client, model_name, news_text, metrics_summary, ticker_news_text="", cluster_context=""):
    combined_prompt = f"""
Du bist ein führender quantitativer Analyst und Investment-Stratege. Nutze fundierte Finanzmathematik und Makro-Logik für Privatanleger.

DATENBASIS:
[AKTUELLE WELT- & MARKTNACHRICHTEN]
{news_text}

[DEPOT-KENNZAHLEN & FUNDAMENTALDATEN]
{metrics_summary}

[DEPOT-AUFTEILUNG & DIVERSIFIKATION]
{cluster_context}

Erstelle deine strukturierte Auswertung exakt getrennt nach diesen 4 Markern:

===MARKT===
### 🌍 TOP 10 Marktnachrichten & Makro-Lage
Fasse die 10 wichtigsten weltweiten Wirtschaftsmeldungen präzise zusammen.

### 🧭 Gesamtstimmung der Börse
🟢 **Optimistisch**, 🟡 **Neutral** oder 🔴 **Vorsichtig** mit kurzer Begründung.

---

### 💡 Markt-Ratgeber: Konkrete Kauf- & Verkaufsempfehlungen basierend auf den Nachrichten

#### 🟢 TOP 10 KAUF-EMPFEHLUNGEN (Profiteure der aktuellen Trends):
Nenne 10 konkrete, liquide Qualitätsaktien oder ETFs (mit Ticker), die vom aktuellen Zins-, Geopolitik- oder Technologietrend profitieren:
1. **[Aktie/ETF 1]** (Ticker): **Warum kaufen?** (Konkreter Treiber: Zinsen, Margen, Marktmacht)
2. **[Aktie/ETF 2]** (Ticker): **Warum kaufen?**
3. **[Aktie/ETF 3]** (Ticker): **Warum kaufen?**
4. **[Aktie/ETF 4]** (Ticker): **Warum kaufen?**
5. **[Aktie/ETF 5]** (Ticker): **Warum kaufen?**
6. **[Aktie/ETF 6]** (Ticker): **Warum kaufen?**
7. **[Aktie/ETF 7]** (Ticker): **Warum kaufen?**
8. **[Aktie/ETF 8]** (Ticker): **Warum kaufen?**
9. **[Aktie/ETF 9]** (Ticker): **Warum kaufen?**
10. **[Aktie/ETF 10]** (Ticker): **Warum kaufen?**

#### 🔴 AKTUELL MEIDEN / GEWINNE MITNEHMEN:
Nenne mindestens 4 Branchen oder konkrete Aktien mit erhöhtem Abwärtsrisiko:
* **[Aktie/Branche 1]**: **Risikofaktor** (z. B. Bewertungsblase, Zinslast, Margendruck)
* **[Aktie/Branche 2]**: **Risikofaktor**
* **[Aktie/Branche 3]**: **Risikofaktor**
* **[Aktie/Branche 4]**: **Risikofaktor**

===DEPOT===
Analysiere jede der Depot-Positionen des Nutzers:
- **[Unternehmen]**: Bewertung (KGV, Fair Value), technische Verfassung und Treiber der nächsten Monate.

===SIGNALE===
Konkrete Handlungsempfehlungen für die Depot-Positionen:
- **[Unternehmen]**: 🟢 **KAUFEN**, 🟡 **HALTEN** oder 🔴 **VERKAUFEN** mit klarer Begründung und Risikolevel.

===KLUMPEN===
1. **Risiko-Score**: 1 (Sehr robust gestreut) bis 10 (Gefährliches Klumpenrisiko).
2. **Begründung der Asset-Allokation**: Bewertung der Verteilung auf Wachstums-, Schutz- und Basiswerte.
3. **Konkrete Portfolio-Ergänzung**: Welche 1-2 Anlageklassen oder Regionen das Depot optimal krisenfest machen würden.
"""

    res = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": combined_prompt}],
        temperature=0.2,
        max_tokens=2800,
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
Vergleiche als Analyst diese beiden Aktien objektiv:
Aktie A: {stock_a_info}
Aktie B: {stock_b_info}

1. Direkter Kennzahlenvergleich (KGV, Dividende, Fair Value, Sektor-Burggraben)
2. Klares Fazit: Welche Aktie bietet aktuell das bessere Chance-Risiko-Verhältnis für Neuinvestitionen?
"""
    res = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=700,
    )
    return res.choices[0].message.content
