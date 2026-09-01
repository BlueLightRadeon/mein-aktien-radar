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
        preferred = ["llama-3.3-70b-versatile", "llama-3.1-70b-versatile", "llama-3.1-8b-instant"]
        sorted_models = [m for m in preferred if m in valid_models] + [m for m in valid_models if m not in preferred]
        return sorted_models if sorted_models else [DEFAULT_MODEL]
    except Exception:
        return [DEFAULT_MODEL, "llama-3.1-70b-versatile", "llama-3.1-8b-instant"]

def run_analysis(client, model_name, news_text, metrics_summary, ticker_news_text="", cluster_context=""):
    combined_prompt = f"""
Du bist ein renommierter quantitativer Chef-Anlagestratege. Analysiere die aktuellen Weltnachrichten sowie das bestehende Depot des Nutzers und erstelle fundierte Empfehlungen auf Deutsch.

[AKTUELLE WELT- & WIRTSCHAFTSNACHRICHTEN]
{news_text[:1200]}

[BESTEHENDE DEPOT-WERTE DES NUTZERS]
{metrics_summary}

[DEPOT-AUFTEILUNG]
{cluster_context}

Gliedere deine Antwort zwingend mit diesen 4 Trennzeilen:

---ABSCHNITT_MARKT---
### 🌍 TOP 10 Marktnachrichten
10 prägnante Stichpunkte zur aktuellen globalen Wirtschaftslage.

### 🧭 Gesamtstimmung der Börse
🟢 Optimistisch, 🟡 Neutral oder 🔴 Vorsichtig mit kurzer Begründung.

---ABSCHNITT_DEPOT---
### 💼 Statusbericht zu deinen Depot-Positionen
Analysiere die bestehenden Aktien des Nutzers:
- **[Unternehmen]**: Aktuelle Bewertung, Trend und worauf man jetzt achten muss.

---ABSCHNITT_SIGNALE---
### 🎯 TOP 5 KAUF-EMPFEHLUNGEN (Stand Jetzt zur Portfolio-Erweiterung)
Wähle basierend auf der aktuellen Lage genau 5 konkrete Qualitätsaktien oder ETFs (NICHT aus dem aktuellen Bestand), die das Depot ideal ergänzen:

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

---ABSCHNITT_KLUMPEN---
### 🛡️ Risikostreuung & Depot-Optimierung
1. **Risiko-Score**: 1 (Sehr gut gestreut) bis 10 (Hohes Risiko).
2. **Erläuterung der Streuung**: Wo liegen aktuell die Schwerpunkte?
3. **Erweiterungs-Tipp**: Welche der oben empfohlenen 5 Aktien das Depot am besten absichert.
"""

    models_to_try = [model_name, "llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
    seen = set()
    models_to_try = [x for x in models_to_try if not (x in seen or seen.add(x))]

    full_text = ""
    last_err_msg = ""

    for target_model in models_to_try:
        try:
            res = client.chat.completions.create(
                model=target_model,
                messages=[
                    {"role": "system", "content": "Du bist ein führender Börsen- und Finanzanalyst. Antworte auf Deutsch und verwende exakt die Trennzeilen ---ABSCHNITT_MARKT---, ---ABSCHNITT_DEPOT---, ---ABSCHNITT_SIGNALE--- und ---ABSCHNITT_KLUMPEN---."},
                    {"role": "user", "content": combined_prompt}
                ],
                temperature=0.2,
                max_tokens=2500,
            )
            if res.choices and res.choices[0].message and res.choices[0].message.content:
                full_text = res.choices[0].message.content
                break
        except Exception as e:
            last_err_msg = str(e)
            continue

    if not full_text:
        err_display = f"⚠️ **Groq API Fehler:** `{last_err_msg}`\n\nBitte prüfe deinen API-Key in den Streamlit Secrets."
        return err_display, err_display, err_display, err_display

    # Robuste Zerlegung
    out_market = ""
    out_depot = ""
    out_signals = ""
    out_cluster = ""

    # 1. Versuch: Trennung über die Standard-Tags
    if "---ABSCHNITT_MARKT---" in full_text:
        p1 = full_text.split("---ABSCHNITT_MARKT---")[1]
        if "---ABSCHNITT_DEPOT---" in p1:
            out_market, p2 = p1.split("---ABSCHNITT_DEPOT---", 1)
            if "---ABSCHNITT_SIGNALE---" in p2:
                out_depot, p3 = p2.split("---ABSCHNITT_SIGNALE---", 1)
                if "---ABSCHNITT_KLUMPEN---" in p3:
                    out_signals, out_cluster = p3.split("---ABSCHNITT_KLUMPEN---", 1)
                else:
                    out_signals = p3
            else:
                out_depot = p2
        else:
            out_market = p1

    # 2. Versuch: Flexible Regex-Erkennung falls die KI alternative Überschriften verwendet hat
    if not out_market or not out_signals:
        def grab_section(text, start_pat, end_pats):
            m = re.search(start_pat, text, re.IGNORECASE)
            if not m:
                return ""
            start_pos = m.end()
            end_pos = len(text)
            for ep in end_pats:
                em = re.search(ep, text[start_pos:], re.IGNORECASE)
                if em:
                    end_pos = min(end_pos, start_pos + em.start())
            return text[start_pos:end_pos].strip()

        m_txt = grab_section(full_text, r"(?:MARKT|TOP 10 Marktnachrichten)", [r"(?:DEPOT|Statusbericht)", r"(?:SIGNALE|KAUF-EMPFEHLUNGEN)", r"(?:KLUMPEN|Risikostreuung)"])
        d_txt = grab_section(full_text, r"(?:DEPOT|Statusbericht)", [r"(?:SIGNALE|KAUF-EMPFEHLUNGEN)", r"(?:KLUMPEN|Risikostreuung)"])
        s_txt = grab_section(full_text, r"(?:SIGNALE|KAUF-EMPFEHLUNGEN|TOP 5)", [r"(?:KLUMPEN|Risikostreuung)"])
        c_txt = grab_section(full_text, r"(?:KLUMPEN|Risikostreuung)", [])

        if m_txt: out_market = m_txt
        if d_txt: out_depot = d_txt
        if s_txt: out_signals = s_txt
        if c_txt: out_cluster = c_txt

    # 3. Absoluter Fallback: Wenn alles fehlschlägt, den gesamten KI-Text ausgeben (Niemals "Keine Daten.")
    if not out_market.strip():
        out_market = full_text
    if not out_depot.strip():
        out_depot = "Die detaillierte Auswertung findest du im Tab '🌍 Nachrichten'."
    if not out_signals.strip():
        out_signals = full_text
    if not out_cluster.strip():
        out_cluster = "Die Risikobewertung findest du im Tab '🌍 Nachrichten'."

    return (
        out_market.strip(),
        out_depot.strip(),
        out_signals.strip(),
        out_cluster.strip()
    )

def run_duel_analysis(client, model_name, stock_a_info, stock_b_info):
    prompt = f"""
Vergleiche als quantitativer Analyst diese beiden Aktien:
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
