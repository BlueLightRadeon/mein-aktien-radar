import streamlit as st
from groq import Groq
import json

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
        preferred = ["llama-3.3-70b-versatile", "llama-3.1-70b-versatile", "llama-3.1-8b-instant"]
        sorted_models = [m for m in preferred if m in valid_models] + [m for m in valid_models if m not in preferred]
        return sorted_models if sorted_models else [DEFAULT_MODEL]
    except Exception:
        return [DEFAULT_MODEL, "llama-3.1-70b-versatile", "llama-3.1-8b-instant"]

def run_analysis(client, model_name, news_text, metrics_summary, ticker_news_text="", cluster_context=""):
    system_prompt = (
        "Du bist ein quantitativer Chef-Anlagestratege. Analysiere das Depot und die aktuellen Marktnachrichten. "
        "Du MUSST deine Antwort ausschließlich als valides JSON-Objekt mit genau vier Schlüsseln formatieren:\n"
        "{\n"
        '  "markt": "Markdown-Text mit TOP 10 Marktnachrichten und Börsenstimmung",\n'
        '  "depot": "Markdown-Text mit Statusbericht zu jeder bestehenden Depot-Aktie",\n'
        '  "signale": "Markdown-Text mit TOP 5 Kaufempfehlungen (NEUE Aktien) und Risikowerten",\n'
        '  "klumpen": "Markdown-Text mit Risiko-Score (1-10) und Streuungs-Analyse"\n'
        "}"
    )

    user_prompt = f"""
[AKTUELLE WELT- & WIRTSCHAFTSNACHRICHTEN]
{news_text[:1200]}

[BESTEHENDE DEPOT-WERTE DES NUTZERS]
{metrics_summary}

[DEPOT-AUFTEILUNG]
{cluster_context}

Erstelle die strukturierte Auswertung für die vier JSON-Felder:
1. "markt": 10 prägnante Stichpunkte zur Wirtschaftslage + Gesamtstimmung (🟢/🟡/🔴).
2. "depot": Konkreter Statuscheck zu den bestehenden Aktien des Nutzers.
3. "signale": Genau 5 konkrete NEUE Qualitätsaktien/ETFs (nicht im Depot) zur Portfolio-Erweiterung (mit Ticker, Branche, Kurspotenzial und Grund) + Branchen die man meiden sollte.
4. "klumpen": Risiko-Score 1-10, Klumpenrisiko-Erklärung und welcher der 5 neuen Werte das Gesamtrisiko am schnellsten senkt.
"""

    models_to_try = [model_name, "llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
    seen = set()
    models_to_try = [x for x in models_to_try if not (x in seen or seen.add(x))]

    raw_response = ""
    last_err = None

    for target_model in models_to_try:
        try:
            res = client.chat.completions.create(
                model=target_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
                max_tokens=2500,
            )
            if res.choices and res.choices[0].message and res.choices[0].message.content:
                raw_response = res.choices[0].message.content
                break
        except Exception as e:
            last_err = e
            continue

    if not raw_response:
        err_msg = str(last_err) if last_err else "Keine Antwort erhalten."
        return (
            f"⚠️ **Fehler:** `{err_msg}`",
            "Keine Daten empfangen.",
            "Keine Daten empfangen.",
            "Keine Daten empfangen."
        )

    try:
        data = json.loads(raw_response)
        out_market = data.get("markt", "")
        out_depot = data.get("depot", "")
        out_signals = data.get("signale", "")
        out_cluster = data.get("klumpen", "")
    except Exception:
        out_market = raw_response
        out_depot = "Auswertung in Tab 1 enthalten."
        out_signals = raw_response
        out_cluster = "Auswertung in Tab 1 enthalten."

    return (
        out_market if out_market else "Marktdaten geladen.",
        out_depot if out_depot else "Depot-Stimmung geladen.",
        out_signals if out_signals else "Kaufempfehlungen geladen.",
        out_cluster if out_cluster else "Risikobewertung geladen."
    )

def run_duel_analysis(client, model_name, stock_a_info, stock_b_info):
    prompt = f"""
Vergleiche als Analyst diese beiden Aktien objektiv:
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
