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
Du bist ein professioneller Finanzanalyst. Antworte AUSSCHLIESSLICH auf reinem Deutsch.

Hier sind weltweite Wirtschaftsnachrichten:
{news_text}

Aufgabe:
1. Erstelle die **TOP 10 wichtigsten Markt- und Börsen-Informationen** des Tages prägnant und übersichtlich auf Deutsch.
2. Gib am Ende eine Zusammenfassung der weltweiten Marktstimmung (Bullisch / Neutral / Bärisch).
"""

  prompt_depot = f"""
Du bist ein Portfolio-Experte. Antworte AUSSCHLIESSLICH auf Deutsch.

Hier sind die Kennzahlen der Depot-Aktien:
{metrics_summary}

Direkte Unternehmensnachrichten:
{ticker_news_text}

Allgemeine Marktnachrichten:
{news_text}

Erstelle für JEDE gelistete Aktie eine strukturierte deutsche Analyse:
1. **Sentiment**: 🟢 Bullisch, 🟡 Neutral oder 🔴 Bärisch
2. **Technik & Bewertung**: Interpretiere Kurs, RSI (überkauft/überverkauft) und KGV.
3. **Fokus/News**: Wichtigste Treiber und Ad-hocs.
4. **Ausblick**: Worauf Anleger in den nächsten Tagen achten sollten.
"""

  prompt_signals = f"""
Du bist ein quantitativer Analyst. Antworte AUSSCHLIESSLICH auf Deutsch.

Hier sind die Daten:
{metrics_summary}
{ticker_news_text}

Erstelle für JEDE Aktie eine klare Handlungsempfehlung:

### [Name der Aktie] ([Ticker])
- **Signal**: 🟢 **KAUFEN** / 🟡 **HALTEN** / 🔴 **VERKAUFEN**
- **Begründung**: Begründe die Entscheidung mit RSI, KGV, Analysten-Kursziel und Nachrichtenlage.
- **Risikostufe**: Gering / Mittel / Hoch
- **Anlagehorizont**: Kurzfristig / Mittelfristig / Langfristig
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

  return (
      res_market.choices[0].message.content,
      res_depot.choices[0].message.content,
      res_signals.choices[0].message.content,
  )
