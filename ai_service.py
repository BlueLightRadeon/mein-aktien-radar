import streamlit as st
from groq import Groq

@st.cache_data(ttl=3600)
def get_account_models(api_key):
    try:
        c = Groq(api_key=api_key.strip())
        models_list = [m.id for m in c.models.list().data]
        text_models = [
            m for m in models_list
            if not any(x in m.lower() for x in ["whisper", "guard", "orpheus", "vision", "safeguard"])
        ]
        return text_models if text_models else models_list
    except Exception:
        return ["llama-3.3-70b-versatile", "openai/gpt-oss-120b"]

def run_analysis(client, model_name, news_text, metrics_summary, ticker_news_text="", cluster_context=""):
    combined_prompt = f"""
Du bist ein renommierter Chef-Anlagestratege. Analysiere die aktuellen Weltnachrichten und leite daraus konkrete Investment-Chancen und Risiken für Privatanleger ab.

DATENBASIS:
[AKTUELLE WELT-NACHRICHTEN]
{news_text}

[DEPOT-KENNZAHLEN DES NUTZERS]
{metrics_summary}

[DEPOT-AUFTEILUNG]
{cluster_context}

Erstelle die Analyse strikt getrennt nach diesen 4 Markern:

===MARKT===
### 🌍 TOP 10 Marktnachrichten & Makro-Lage
Fasse die 10 wichtigsten Wirtschaftsmeldungen zusammen.

### 🧭 Gesamtstimmung der Börse
🟢 **Optimistisch**, 🟡 **Neutral** oder 🔴 **Vorsichtig** mit kurzer Begründung.

---

### 💡 Markt-Ratgeber: Konkrete Kauf- & Verkaufsempfehlungen basierend auf den Nachrichten

#### 🟢 TOP 10 KAUF-EMPFEHLUNGEN (Profiteure der aktuellen Nachrichtenlage):
Nenne mindestens 10 konkrete, bekannte Qualitätsaktien/ETFs (aus deinem Wissen und passend zu den Meldungen wie Zinsen, Rüstung, Energie, Tech, Pharma, Konsum), die von den aktuellen Trends massiv profitieren:
1. **[Aktie/ETF 1]** (Ticker): **Warum kaufen?** (Konkreter Bezug zu den aktuellen Nachrichten, z.B. geopolitische Lage, Zinstrend, Preissetzungsmacht)
2. **[Aktie/ETF 2]** (Ticker): **Warum kaufen?**
3. **[Aktie/ETF 3]** (Ticker): **Warum kaufen?**
4. **[Aktie/ETF 4]** (Ticker): **Warum kaufen?**
5. **[Aktie/ETF 5]** (Ticker): **Warum kaufen?**
6. **[Aktie/ETF 6]** (Ticker): **Warum kaufen?**
7. **[Aktie/ETF 7]** (Ticker): **Warum kaufen?**
8. **[Aktie/ETF 8]** (Ticker): **Warum kaufen?**
9. **[Aktie/ETF 9]** (Ticker): **Warum kaufen?**
10. **[Aktie/ETF 10]** (Ticker): **Warum kaufen?**

#### 🔴 AKTUELL MEIDEN / VERKAUFEN (Verlierer der Nachrichtenlage):
Nenne mindestens 4-5 Branchen oder konkrete Aktien, bei denen man aktuell vorsichtig sein oder Gewinne mitnehmen sollte:
* **[Aktie/Branche 1]**: **Warum meiden?** (z.B. Zinsdruck, schrumpfende Margen, Nachfragerückgang)
* **[Aktie/Branche 2]**: **Warum meiden?**
* **[Aktie/Branche 3]**: **Warum meiden?**
* **[Aktie/Branche 4]**: **Warum meiden?**

===DEPOT===
Analysiere jede der aktuell 8 Depot-Positionen des Nutzers im Lichte der heutigen Nachrichten:
- **[Unternehmen]**: Aktuelle Lage, Bewertung und Ausblick.

===SIGNALE===
Handlungsempfehlungen für die 8 Depot-Positionen:
- **[Unternehmen]**: 🟢 **KAUFEN**, 🟡 **HALTEN** oder 🔴 **VERKAUFEN** mit klarer Begründung.

===KLUMPEN===
1. **Risiko-Score**: 1 (Perfekt gestreut) bis 10 (Hohes Klumpenrisiko).
2. **Erklärung zur Streuung**: Warum die Verteilung auf Wachstums-Motoren, Krisen-Puffer und Schutzschilder wichtig ist.
3. **Optimierung**: Welche Sektoren fehlen noch zur vollständigen Absicherung?
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
Vergleiche diese beiden Aktien in einfachen Worten:
Aktie A: {stock_a_info}
Aktie B: {stock_b_info}

1. Stärken & Schwächen (KGV, Fair Value, Wachstum, Dividende)
2. Klares Fazit: Welche Aktie ist aktuell der bessere Kauf und warum?
"""
    res = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=600,
    )
    return res.choices[0].message.content
