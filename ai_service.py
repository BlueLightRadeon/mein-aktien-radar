from groq import Groq
import streamlit as st


@st.cache_data(ttl=3600)
def get_account_models(api_key):
  try:
    c = Groq(api_key=api_key.strip())
    models_list = [m.id for m in c.models.list().data]
    text_models = [
        m
        for m in models_list
        if not any(
            x in m.lower()
            for x in ["whisper", "guard", "orpheus", "vision", "safeguard"]
        )
    ]
    return text_models if text_models else models_list
  except Exception:
    return ["openai/gpt-oss-120b", "llama-3.3-70b-versatile"]


def run_analysis(
    client,
    model_name,
    news_text,
    metrics_summary,
    ticker_news_text,
    cluster_context="",
):
  prompt_market = f"""
Du bist ein professioneller Finanzanalyst. Antworte AUSSCHLIESSLICH auf Deutsch.
Hier sind aktuelle Wirtschaftsnachrichten:
{news_text}

Erstelle:
1. Die **TOP 10 wichtigsten Markt- und Börsen-Informationen** prägnant und fundiert.
2. Eine Einschätzung der aktuellen Marktstimmung (Bullisch / Neutral / Bärisch) inkl. Fear & Greed Tendenz.
"""

  prompt_depot = f"""
Du bist ein Portfolio-Manager. Antworte AUSSCHLIESSLICH auf Deutsch.
Depot-Werte & Kennzahlen:
{metrics_summary}

Unternehmensnachrichten:
{ticker_news_text}

Erstelle für JEDE Aktie:
1. **Sentiment**: 🟢 Bullisch, 🟡 Neutral oder 🔴 Bärisch
2. **Technik & Bewertung**: RSI, KGV und Fair-Value Einordnung.
3. **Fokus/News**: Relevante Treiber und Katalysatoren.
4. **Ausblick**: Wichtige Marken für die kommenden Tage.
"""

  prompt_signals = f"""
Du bist ein quantitativer Analyst. Antworte AUSSCHLIESSLICH auf Deutsch.
Kennzahlen & Konsens:
{metrics_summary}

Erstelle für jede Aktie eine klare Handlungsempfehlung:
### [Name der Aktie] ([Ticker])
- **Signal**: 🟢 **KAUFEN** / 🟡 **HALTEN** / 🔴 **VERKAUFEN**
- **Begründung**: Analyse aus RSI, KGV, Fair Value und Nachrichten.
- **Risikostufe**: Gering / Mittel / Hoch
- **Anlagehorizont**: Kurzfristig / Mittelfristig / Langfristig
"""

  prompt_cluster = f"""
Du bist ein Risikomanager. Antworte AUSSCHLIESSLICH auf Deutsch.
Portfolio-Zusammensetzung nach Branchen und Regionen:
{cluster_context}

Bewerte das Klumpenrisiko des Depots:
1. **Risiko-Score**: 1 (Sehr diversifiziert) bis 10 (Extremes Klumpenrisiko).
2. **Kritische Übergewichtungen**: Wo liegen gefährliche Abhängigkeiten (z.B. US-Tech oder Halbleiter)?
3. **Absicherungs-Empfehlungen**: Welche 1-2 Anlageklassen oder Sektoren fehlen zur optimalen Balance?
"""

  res_market = client.chat.completions.create(
      model=model_name,
      messages=[{"role": "user", "content": prompt_market}],
      temperature=0.2,
      max_tokens=900,
  )
  res_depot = client.chat.completions.create(
      model=model_name,
      messages=[{"role": "user", "content": prompt_depot}],
      temperature=0.2,
      max_tokens=1000,
  )
  res_signals = client.chat.completions.create(
      model=model_name,
      messages=[{"role": "user", "content": prompt_signals}],
      temperature=0.2,
      max_tokens=1000,
  )
  res_cluster = client.chat.completions.create(
      model=model_name,
      messages=[{"role": "user", "content": prompt_cluster}],
      temperature=0.2,
      max_tokens=800,
  )

  return (
      res_market.choices[0].message.content,
      res_depot.choices[0].message.content,
      res_signals.choices[0].message.content,
      res_cluster.choices[0].message.content,
  )


def run_duel_analysis(client, model_name, stock_a_info, stock_b_info):
  prompt = f"""
Du bist ein neutraler Aktien-Analyst. Vergleiche diese beiden Aktien in einem direkten Duell auf Deutsch:
AKTE 1: {stock_a_info}
AKTE 2: {stock_b_info}

Erstelle:
1. **Stärken-Vergleich**: Bewertung (KGV/Fair Value), Charttechnik (RSI) und Wachstum.
2. **Klares Duell-Urteil**: Welche Aktie bietet aktuell das bessere Chance-Risiko-Verhältnis für die nächsten 6-12 Monate?
"""
  res = client.chat.completions.create(
      model=model_name,
      messages=[{"role": "user", "content": prompt}],
      temperature=0.2,
      max_tokens=800,
  )
  return res.choices[0].message.content
