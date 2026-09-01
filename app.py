import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from groq import Groq
from datetime import datetime

try:
    from data_service import (
        load_saved_portfolio, save_portfolio_to_file, get_stock_data,
        get_individual_series_dict, fetch_all_headlines, search_ticker_candidates,
        clean_ticker, parse_trade_republic_pdf
    )
    from ai_service import get_account_models, run_analysis, run_duel_analysis
except Exception as e:
    st.error(f"Import-Fehler: Bitte prüfe die Dateien auf GitHub. Details: {e}")
    st.stop()

st.set_page_config(
    page_title="KI Markt- & Depot-Radar",
    page_icon="📈",
    layout="wide"
)

st.title("📈 KI Markt- & Depot-Radar")

GROQ_KEY = st.secrets.get("GROQ_API_KEY", "")

if "my_portfolio" not in st.session_state:
    st.session_state.my_portfolio = load_saved_portfolio()

if "tr_cash" not in st.session_state:
    st.session_state.tr_cash = 0.0

def fmt_eur(val):
    try:
        return f"{val:,.2f} €".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return f"{val} €"

# --- SEITENLEISTE: EINGABE & VERWALTUNG ---
with st.sidebar:
    st.header("💼 Depot & Trade Republic")
    
    with st.expander("📥 TR-Kontoauszug (PDF) hochladen"):
        st.caption("PDF-Depotauszug hochladen, um Bestände automatisch einzulesen.")
        tr_pdf = st.file_uploader("PDF auswählen", type=["pdf"])
        if tr_pdf:
            imported_items, imported_cash = parse_trade_republic_pdf(tr_pdf)
            if imported_items:
                st.session_state.my_portfolio = imported_items
                if imported_cash is not None:
                    st.session_state.tr_cash = imported_cash
                save_portfolio_to_file(st.session_state.my_portfolio)
                st.success(f"✅ {len(imported_items)} Positionen importiert!")
                st.rerun()

    st.session_state.tr_cash = st.number_input(
        "💶 TR Guthaben (Cash in €):",
        min_value=0.0,
        value=float(st.session_state.tr_cash),
        step=10.0
    )

    st.divider()
    st.subheader("🔍 Aktie hinzufügen:")
    search_query = st.text_input("Name oder Symbol eingeben:", placeholder="z. B. Nvidia, Apple...")
    if search_query:
        results = search_ticker_candidates(search_query)
        if results:
            selected_cand = st.selectbox("Ergebnis wählen:", results, key="side_search_select")
            in_money = st.number_input("Investierter Betrag (€):", min_value=1.0, value=50.0, step=10.0)
            
            if st.button("➕ Hinzufügen", use_container_width=True):
                sym = clean_ticker(selected_cand)
                name = selected_cand.split("(")[1].replace(")", "") if "(" in selected_cand else sym
                st.session_state.my_portfolio.append({
                    "ticker": sym,
                    "name": name,
                    "shares": 1.0,
                    "buy_price": float(in_money)
                })
                save_portfolio_to_file(st.session_state.my_portfolio)
                st.success(f"{sym} hinzugefügt!")
                st.rerun()

    st.divider()
    st.subheader("📝 Beträge anpassen:")
    portfolio_changed = False
    if st.session_state.my_portfolio:
        for idx, item in enumerate(st.session_state.my_portfolio):
            current_invested = float(item.get("buy_price", 50.0))
            col_a, col_b = st.columns([3, 1])
            with col_a:
                new_invested = st.number_input(
                    f"💶 {item.get('name', item['ticker'])}:",
                    min_value=0.0,
                    value=float(current_invested),
                    step=10.0,
                    key=f"money_{idx}_{item['ticker']}"
                )
            with col_b:
                st.write("")
                st.write("")
                if st.button("🗑️", key=f"del_{idx}_{item['ticker']}"):
                    st.session_state.my_portfolio.pop(idx)
                    save_portfolio_to_file(st.session_state.my_portfolio)
                    st.rerun()

            if new_invested != current_invested:
                item["buy_price"] = float(new_invested)
                portfolio_changed = True
        
        if portfolio_changed or st.button("💾 Speichern", use_container_width=True):
            save_portfolio_to_file(st.session_state.my_portfolio)
            st.success("✅ Gespeichert!")
            st.rerun()
    else:
        st.info("Noch keine Positionen hinterlegt.")

    st.divider()
    st.header("🤖 Modell")
    if GROQ_KEY:
        available_models = get_account_models(GROQ_KEY)
        selected_model = st.selectbox("Auswahl:", available_models, index=0)
    else:
        selected_model = "llama-3.3-70b-versatile"

# DATENBERECHNUNG
if st.session_state.my_portfolio:
    stock_df, ticker_news, resolved_tickers = get_stock_data(st.session_state.my_portfolio)
    total_invested = sum([float(x.get("buy_price", 0.0)) for x in st.session_state.my_portfolio])
    stock_val = stock_df["_raw_val"].sum() if stock_df["_raw_val"].sum() > 0 else total_invested
    total_tr_account = stock_val + st.session_state.tr_cash
    stock_pnl = stock_val - total_invested
    stock_pnl_pct = (stock_pnl / total_invested * 100) if total_invested > 0 else 0.0

    c_m1, c_m2, c_m3 = st.columns(3)
    with c_m1:
        st.metric("TR Gesamtkonto", fmt_eur(total_tr_account))
    with c_m2:
        st.metric("Eingezahltes Geld", fmt_eur(total_invested))
    with c_m3:
        st.metric("Gewinn / Verlust", fmt_eur(stock_pnl), delta=f"{stock_pnl_pct:+.2f}%")
else:
    stock_df = pd.DataFrame()
    ticker_news = []
    resolved_tickers = []
    stock_val = 0.0
    total_invested = 0.0
    total_tr_account = st.session_state.tr_cash

# BUTTON FÜR KI-ANALYSE
if st.button("🚀 Gesamte KI-Auswertung starten", use_container_width=True):
    if not GROQ_KEY:
        st.error("⚠️ Kein GROQ_API_KEY hinterlegt!")
    elif not st.session_state.my_portfolio:
        st.error("⚠️ Bitte füge zuerst Aktien zum Depot hinzu!")
    else:
        save_portfolio_to_file(st.session_state.my_portfolio)
        client = Groq(api_key=GROQ_KEY.strip())

        with st.spinner("Analysiere Nachrichten, Kurse und Trends..."):
            news_data = fetch_all_headlines()
            news_text = "\n".join(news_data)
            ticker_news_text = "\n".join(ticker_news) if ticker_news else "Keine aktuellen Sondermeldungen."

            metrics_summary = stock_df[[
                "Name / Aktie", "Ticker", "Aktueller Kurs", "RSI (14D)", 
                "KGV (P/E)", "Fair Value", "Analysten-Kursziel", "Konsens-Rating", "Dividendenrendite"
            ]].to_string(index=False)

            cluster_context = stock_df[["Name / Aktie", "Ticker", "Sektor", "Land", "Positionswert"]].to_string(index=False)

            try:
                out_m, out_d, out_s, out_c = run_analysis(
                    client, selected_model, news_text, metrics_summary, ticker_news_text, cluster_context
                )
                st.session_state["ai_market"] = out_m
                st.session_state["ai_depot"] = out_d
                st.session_state["ai_signals"] = out_s
                st.session_state["ai_cluster"] = out_c
                st.success("✅ Analyse abgeschlossen!")
            except Exception as e:
                st.error(f"Fehler bei der Analyse: {str(e)}")

# 8 TABS
tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🏦 TR-Konto",
    "🌍 Nachrichten",
    "💼 Stimmung",
    "📅 Termine",
    "🎯 Tipps",
    "📊 Charts",
    "🥧 Streuung",
    "⚔️ Duell"
])

# TAB 0: TR KONTO
with tab0:
    st.info("ℹ️ **Kurzinfo:** Zeigt dein reales Guthaben bei Trade Republic – aufgeteilt nach unverzinstem Bargeld (Cash) und aktuellem Gesamtwert deiner Wertpapiere.")
    col_tr1, col_tr2 = st.columns(2)
    with col_tr1:
        st.success(f"💶 **Bargeld (Cash):** {fmt_eur(st.session_state.tr_cash)}")
    with col_tr2:
        st.success(f"📈 **Aktien-Marktwert:** {fmt_eur(stock_val)}")
        
    st.subheader("Positionen im Überblick:")
    if not stock_df.empty:
        st.dataframe(
            stock_df[[
                "Name / Aktie", "Ticker", "Kaufkurs", 
                "Aktueller Kurs", "Positionswert", "Gewinn / Verlust"
            ]].rename(columns={
                "Kaufkurs": "Dein Geldeinsatz",
                "Positionswert": "Aktueller Wert"
            }),
            hide_index=True,
            use_container_width=True
        )

# TAB 1: WELT-NACHRICHTEN
with tab1:
    st.info("ℹ️ **Kurzinfo:** Scannt über 45 weltweite Finanzquellen und fasst die 10 wichtigsten Wirtschafts-Ereignisse sowie die Gesamtstimmung der Börse zusammen.")
    if "ai_market" in st.session_state:
        st.markdown(st.session_state["ai_market"])
    else:
        st.write("Klicke oben auf den Button **'🚀 Gesamte KI-Auswertung starten'**, um die aktuellen Meldungen abzurufen.")

# TAB 2: STIMMUNG & DEPOT
with tab2:
    st.info("ℹ️ **Kurzinfo:** Einzelanalyse für jede deiner Aktien. Prüft die Marktstimmung, ob das Preisschild günstig ist und worauf in den nächsten Tagen zu achten ist.")
    if not stock_df.empty:
        st.dataframe(
            stock_df[[
                "Name / Aktie", "Ticker", "Kaufkurs", "Aktueller Kurs", "Positionswert", "Gewinn / Verlust"
            ]].rename(columns={
                "Kaufkurs": "Dein Geldeinsatz",
                "Positionswert": "Aktueller Wert"
            }),
            hide_index=True,
            use_container_width=True
        )
    if "ai_depot" in st.session_state:
        st.divider()
        st.markdown(st.session_state["ai_depot"])

# TAB 3: TERMINE & DIVIDENDEN
with tab3:
    st.info("ℹ️ **Kurzinfo:** Zeigt an, wann Unternehmen ihre nächsten Geschäftszahlen vorlegen und wie viel passive Gewinnausschüttung (Dividende in % p.a.) auf dein Konto fließt.")
    if not stock_df.empty:
        st.dataframe(
            stock_df[[
                "Name / Aktie", "Ticker", "Dividendenrendite", "Nächste Quartalszahlen"
            ]].rename(columns={
                "Dividendenrendite": "Gewinnausschüttung (% p.a.)",
                "Nächste Quartalszahlen": "Nächster Geschäftsbericht"
            }),
            hide_index=True,
            use_container_width=True
        )

# TAB 4: KAUF- / VERKAUF-TIPPS
with tab4:
    st.info("ℹ️ **Kurzinfo:** Konkrete Empfehlungen (Kaufen, Halten, Verkaufen) der KI mit Begründung, Banken-Kurszielen und dem fairen rechnerischen Wert.")
    if not stock_df.empty:
        st.dataframe(
            stock_df[[
                "Name / Aktie", "Ticker", "Aktueller Kurs", "Fair Value", "Analysten-Kursziel", "Konsens-Rating"
            ]].rename(columns={
                "Fair Value": "Faire Wertschätzung",
                "Analysten-Kursziel": "Experten-Kursziel",
                "Konsens-Rating": "Experten-Empfehlung"
            }),
            hide_index=True,
            use_container_width=True
        )
    if "ai_signals" in st.session_state:
        st.divider()
        st.markdown(st.session_state["ai_signals"])

# TAB 5: CHARTS
with tab5:
    st.info("ℹ️ **Kurzinfo:** Live-Diagramme deiner Aktienkurse. Vergleiche die Wertentwicklung in Prozent oder sieh dir den absoluten Preisverlauf an.")
    c1, c2 = st.columns([1, 1])
    with c1:
        timeframe = st.selectbox(
            "Zeitraum:",
            options=["1d", "5d", "1mo", "6mo", "1y", "5y"],
            index=2,
            format_func=lambda x: {
                "1d": "1 Tag (Live)", "5d": "5 Tage", "1mo": "1 Monat",
                "6mo": "6 Monate", "1y": "1 Jahr", "5y": "5 Jahre"
            }[x]
        )
    with c2:
        chart_mode = st.radio("Ansicht:", ["Wertentwicklung in %", "Preis pro Aktie (in Geld)"], horizontal=True)

    if st.session_state.my_portfolio:
        series_dict = get_individual_series_dict(st.session_state.my_portfolio, period=timeframe)
        if series_dict:
            available_tickers = list(series_dict.keys())
            selected_view = st.selectbox("Fokus:", ["Alle Aktien gleichzeitig"] + available_tickers, index=0)
            
            palette = ["#00D084", "#0693E3", "#FCB900", "#EB144C", "#9B51E0", "#00ACC1", "#FF6900"]
            fig = go.Figure()
            tickers_to_plot = available_tickers if selected_view == "Alle Aktien gleichzeitig" else [selected_view]

            if chart_mode == "Wertentwicklung in %":
                for i, t in enumerate(tickers_to_plot):
                    s = series_dict[t]
                    if not s.empty and s.iloc[0] > 0:
                        base_val = s.iloc[0]
                        pct_series = ((s - base_val) / base_val) * 100
                        fig.add_trace(go.Scatter(
                            x=s.index, y=pct_series, mode='lines', name=t,
                            line=dict(width=2.5, color=palette[i % len(palette)]),
                            hovertemplate=f"<b>{t}</b>: %{{y:+.2f}}%<extra></extra>"
                        ))
                
                fig.add_hline(y=0, line_dash="dash", line_color="rgba(150,150,150,0.6)", annotation_text="0%")
                fig.update_layout(
                    title=f"Performance ({timeframe.upper()})",
                    xaxis_title="Datum",
                    yaxis_title="Gewinn / Verlust (%)",
                    yaxis_ticksuffix="%",
                    hovermode="x unified",
                    margin=dict(l=5, r=5, t=40, b=5),
                    height=380
                )
            else:
                for i, t in enumerate(tickers_to_plot):
                    s = series_dict[t]
                    fig.add_trace(go.Scatter(
                        x=s.index, y=s, mode='lines', name=t,
                        line=dict(width=2.5, color=palette[i % len(palette)]),
                        hovertemplate=f"<b>{t}</b>: %{{y:.2f}}<extra></extra>"
                    ))
                fig.update_layout(
                    title=f"Kursverlauf ({timeframe.upper()})",
                    xaxis_title="Datum",
                    yaxis_title="Preis in € / $",
                    hovermode="x unified",
                    margin=dict(l=5, r=5, t=40, b=5),
                    height=380
                )

            st.plotly_chart(fig, use_container_width=True)

# TAB 6: RISIKOSTREUUNG
with tab6:
    st.info("ℹ️ **Kurzinfo:** Prüft die Verteilung deines Geldes auf Branchen und Länder, warnt vor Klumpenrisiken und schlägt Absicherungen vor.")
    if not stock_df.empty:
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            fig_sec = px.pie(stock_df, names="Sektor", values="_raw_val", title="Branchen-Verteilung", hole=0.4)
            fig_sec.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_sec, use_container_width=True)
        with col_d2:
            fig_geo = px.pie(stock_df, names="Land", values="_raw_val", title="Länder-Verteilung", hole=0.4)
            fig_geo.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_geo, use_container_width=True)
    if "ai_cluster" in st.session_state:
        st.divider()
        st.subheader("🛡️ KI-Gutachten zur Risikostreuung")
        st.markdown(st.session_state["ai_cluster"])

# TAB 7: AKTIEN-VERGLEICH
with tab7:
    st.info("ℹ️ **Kurzinfo:** 1-gegen-1-Vergleich zweier Aktien. Die KI bewertet Kurspotenzial, Kennzahlen und kürt den besseren Kauf.")
    if not stock_df.empty and len(stock_df) >= 2:
        cd1, cd2 = st.columns(2)
        with cd1:
            duel_a = st.selectbox("Erste Aktie:", stock_df["Ticker"].tolist(), index=0)
        with cd2:
            duel_b = st.selectbox("Zweite Aktie:", stock_df["Ticker"].tolist(), index=1)
        
        if st.button("⚡ Duell auswerten", use_container_width=True):
            if GROQ_KEY:
                cl = Groq(api_key=GROQ_KEY.strip())
                row_a = stock_df[stock_df["Ticker"] == duel_a].iloc[0].to_dict()
                row_b = stock_df[stock_df["Ticker"] == duel_b].iloc[0].to_dict()
                with st.spinner("Analysiere Duell..."):
                    res_duel = run_duel_analysis(cl, selected_model, str(row_a), str(row_b))
                    st.markdown(res_duel)
