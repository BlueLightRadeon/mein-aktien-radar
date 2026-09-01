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

def run_analysis(client, model_name, news_text, metrics_summary, ticker_news_text, cluster_context=""):
  combined_prompt = f"""
Du bist ein erfahrener Anlageberater. Antworte auf Deutsch in einfacher, klarer Sprache für Privatanleger.

DATENBASIS:
[NACHRICHTEN]
{news_text}

[DEPOT-KENNZAHLEN & RATINGS]
{metrics_summary}

[SEKTOREN & LÄNDER]
{cluster_context}

Erstelle die Analyse exakt getrennt nach diesen 4 Markern:

===MARKT===
1. **TOP 10 Marktnachrichten**: Die 10 wichtigsten Fakten stichpunktartig.
2. **Gesamtstimmung**: 🟢 Optimistisch, 🟡 Neutral oder 🔴 Vorsichtig mit kurzer Begründung.

===DEPOT===
Analysiere jede Aktie:
- **[Aktienname]**: Stimmung (🟢/🟡/🔴), Bewertung (KGV & Fair Value verständlich eingeordnet) und Chart-Eindruck.
- **Worauf achten**: Wichtige Treiber für die nächsten Wochen.

===SIGNALE===
Erstelle für JEDE Aktie eine differenzierte Handlungsempfehlung:
- **[Aktienname]**: 🟢 **KAUFEN**, 🟡 **HALTEN** oder 🔴 **VERKAUFEN**
- **Konkrete Begründung**: Erkläre genau, WARUM diese Entscheidung getroffen wird (z. B. "Unterbewertet mit 20% Kurspotenzial" oder "KGV sehr hoch, daher erst Rücksetzer abwarten").
- **Risikoeinstufung**: Gering / Mittel / Hoch | **Empfohlene Anlagedauer**: Kurz- oder langfristig.

===KLUMPEN===
1. **Risiko-Score**: 1 (Sehr breit gestreut) bis 10 (Hohes Klumpenrisiko).
2. **Analyse der Kreisdiagramme**: Warum die Verteilung auf Halbleiter, Cybersecurity, Pharma, Rüstung und Länder so aussieht wie dargestellt.
3. **Konkrete Verbesserung**: Welche 1-2 Branchen oder Regionen fehlen, um das Depot noch krisenfester zu machen?
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
Vergleiche diese beiden Aktien in einfachen Worten:
Aktie A: {stock_a_info}
Aktie B: {stock_b_info}

1. Stärken & Schwächen (KGV, Fair Value, Wachstum)
2. Klares Fazit: Welche Aktie ist aktuell der bessere Kauf und warum?
"""
  res = client.chat.completions.create(
      model=model_name,
      messages=[{"role": "user", "content": prompt}],
      temperature=0.2,
      max_tokens=600,
  )
  return res.choices[0].message.content
