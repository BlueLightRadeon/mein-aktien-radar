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
    client, model_name, news_text, metrics_summary, ticker_news_text
):
  prompt_market = f"""
Aktuelle weltweite Wirtschaftsnachrichten (45+ Quellen & ETF-Feeds):
{news_text}

Fasse die **TOP 10 wichtigsten Markt-Informationen** prägnant auf Deutsch zusammen.
Bewerte am Ende kurz die Marktstimmung (Bullisch / Neutral / Bärisch).
"""

  prompt_depot = f"""
Depot-Werte & Kennzahlen:
{metrics_summary}

Spezifische News zu deinen Titeln:
{ticker_news_text}

Allgemeine Makro-Nachrichten:
{news_text}

Erstelle für jeden Wert eine fundierte Analyse:
1. **Sentiment**: 🟢 Bullisch, 🟡 Neutral oder 🔴 Bärisch
2. **Technik & Bewertung**: Interpretiere kurz den RSI (unter 30 = überverkauft, über 70 = überkauft), das KGV und das Analysten-Kursziel.
3. **Fokus/News**: Relevante Meldungen oder Einflussfaktoren.
4. **Tipp für Anleger**: Konkrete Beobachtungspunkte für die kommenden Tage.
"""

  prompt_signals = f"""
Du bist ein quantitativer Portfolio-Analyst. Hier sind die Echtzeitdaten der Aktien:
{metrics_summary}

News & Makro-Umfeld:
{ticker_news_text}
{news_text}

Erstelle für JEDE Aktie einzeln eine strukturierte Handlungsempfehlung nach diesem Schema:

### [Name der Aktie] ([Ticker])
- **Signal**: 🟢 **KAUFEN** / 🟡 **HALTEN** / 🔴 **VERKAUFEN**
- **Begründung**: (Kombination aus RSI-Charttechnik, KGV-Bewertung und Analysten-Kursziel)
- **Risikolevel**: Niedrig / Mittel / Hoch
- **Empfohlener Anlagehorizont**: Kurzfristig (Trading) / Mittelfristig / Langfristig (Buy & Hold)

Wichtig: Begründe das Signal objektiv mit den Kennzahlen (z. B. RSI über 70 = Gewinnmitnahme/Halten statt Neukauf).
"""

  res_market = client.chat.completions.create(
      model=model_name,
      messages=[{"role": "user", "content": prompt_market}],
      temperature=0.3,
      max_tokens=900,
  )

  res_depot = client.chat.completions.create(
      model=model_name,
      messages=[{"role": "user", "content": prompt_depot}],
      temperature=0.3,
      max_tokens=1000,
  )

  res_signals = client.chat.completions.create(
      model=model_name,
      messages=[{"role": "user", "content": prompt_signals}],
      temperature=0.2,
      max_tokens=1000,
  )

  return (
      res_market.choices[0].message.content,
      res_depot.choices[0].message.content,
      res_signals.choices[0].message.content,
  )
