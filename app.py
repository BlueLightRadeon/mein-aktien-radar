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
    layout="centered"
)

st.title("📈 KI Markt- & Depot-Radar")

GROQ_KEY = st.secrets.get("GROQ_API_KEY", "")

if "my_portfolio" not in st.session_state:
    st.session_state.my_portfolio = load_saved_portfolio()

# --- SEITENLEISTE ---
with st.sidebar:
    st.header("💼 Mein Depot & Trade Republic")
    
    with st.expander("📥 Trade Republic Auszug importieren"):
        st.caption("Lade hier deinen PDF-Kontoauszug hoch, um deine Aktien automatisch einzulesen.")
        tr_pdf = st.file_uploader("PDF-Kontoauszug hochladen", type=["pdf"])
        if tr_pdf:
            imported = parse_trade_republic_pdf(tr_pdf)
            if imported:
                for it in imported:
                    if not any(x["ticker"] == it["ticker"] for x in st.session_state.my_portfolio):
                        st.session_state.my_portfolio.append(it)
                save_portfolio_to_file(st.session_state.my_portfolio)
                st.success(f"{len(imported)} Positionen übernommen!")
                st.rerun()

    search_query = st.text_input("🔍 Aktie suchen:", placeholder="z. B. Novo Nordisk, Rheinmetall...")
    if search_query:
        results = search_ticker_candidates(search_query)
        if results:
            selected_cand = st.selectbox("Gefundene Treffer:", results, key="side_search_select")
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                in_shares = st.number_input("Stückzahl", min_value=0.01, value=1.0, step=1.0, help="Wie viele Anteile dieser Aktie besitzt du?")
            with col_s2:
                in_buy = st.number_input("Dein Kaufkurs (€/$)", min_value=0.0, value=0.0, step=10.0, help="Zu welchem Preis pro Aktie hast du damals gekauft?")
            
            if st.button("➕ Zum Depot hinzufügen", use_container_width=True):
                sym = clean_ticker(selected_cand)
                name = selected_cand.split("(")[1].replace(")", "") if "(" in selected_cand else sym
                st.session_state.my_portfolio.append({
                    "ticker": sym,
                    "name": name,
                    "shares": float(in_shares),
                    "buy_price": float(in_buy)
                })
                save_portfolio_to_file(st.session_state.my_portfolio)
                st.success(f"{sym} hinzugefügt!")
                st.rerun()

    st.divider()
    st.subheader("Aktuell im Depot:")
    if st.session_state.my_portfolio:
        for idx, item in enumerate(list(st.session_state.my_portfolio)):
            with st.container():
                c_a, c_b = st.columns([4, 1])
                with c_a:
                    st.write(f"• **{item.get('name', item['ticker'])}** ({item['ticker']})")
                    st.caption(f"Anzahl: {item.get('shares', 1.0):.1f} Stk. | Kauf: {item.get('buy_price', 0.0):.2f} €")
                with c_b:
                    if st.button("❌", key=f"del_btn_{idx}", help="Diese Aktie aus deiner Liste löschen"):
                        st.session_state.my_portfolio.pop(idx)
                        save_portfolio_to_file(st.session_state.my_portfolio)
                        st.rerun()
    else:
        st.info("Dein Depot ist aktuell leer.")

    st.divider()
    st.header("🤖 KI-Gehirn")
    if GROQ_KEY:
        available_models = get_account_models(GROQ_KEY)
        selected_model = st.selectbox("Aktives KI-Modell", available_models, index=0, help="Das Modell, das die Auswertungen für dich schreibt.")
    else:
        selected_model = "llama-3.3-70b-versatile"

# 7 TABS
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🌍 Welt-Nachrichten",
    "💼 Stimmung & Depot",
    "📅 Termine & Dividenden",
    "🎯 Kauf- / Verkauf-Tipps",
    "📊 Kurs-Diagramme",
    "🥧 Risikostreuung",
    "⚔️ Aktien-Vergleich"
])

# Daten laden
if st.session_state.my_portfolio:
    stock_df, ticker_news, resolved_tickers = get_stock_data(st.session_state.my_portfolio)
    total_val = stock_df["_raw_val"].sum()
    total_invested = stock_df["_raw_invested"].sum()
    total_pnl = total_val - total_invested if total_invested > 0 else 0.0
    total_pnl_pct = (total_pnl / total_invested * 100) if total_invested > 0 else 0.0

    col_t1, col_t2, col_t3 = st.columns(3)
    with col_t1:
        st.metric("Gesamtwert Depot", f"{total_val:,.2f} €", help="So viel sind all deine Aktien aktuell zusammen wert.")
    with col_t2:
        st.metric("Eingezahltes Geld", f"{total_invested:,.2f} €", help="So viel eigenes Geld hast du insgesamt für die Käufe bezahlt.")
    with col_t3:
        st.metric("Dein Gewinn / Verlust", f"{total_pnl:+,.2f} €", delta=f"{total_pnl_pct:+.2f}%", help="Dein tatsächlicher Gewinn oder Verlust seit dem Kauf.")
else:
    stock_df = pd.DataFrame()
    ticker_news = []
    resolved_tickers = []

# TAB 5: Kurs-Diagramme
with tab5:
    st.subheader("📊 Kursentwicklung & Performance")
    
    with st.expander("ℹ️ Was bedeuten die Optionen hier?"):
        st.write("""
        - **Wertentwicklung in %**: Alle Aktien starten bei 0 %. Du siehst sofort, welche Aktie am stärksten gestiegen oder gefallen ist.
        - **Preis pro Aktie**: Zeigt den tatsächlichen Geld-Betrag (in € oder $), den eine einzelne Aktie kostet.
        """)

    c1, c2 = st.columns([1, 1])
    with c1:
        timeframe = st.selectbox(
            "Zeitraum auswählen:",
            options=["1d", "5d", "1mo", "6mo", "1y", "5y"],
            index=2,
            format_func=lambda x: {
                "1d": "1 Tag (Live heute)",
                "5d": "5 Tage",
                "1mo": "1 Monat",
                "6mo": "6 Monate",
                "1y": "1 Jahr",
                "5y": "5 Jahre"
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
                
                fig.add_hline(y=0, line_dash="dash", line_color="rgba(150,150,150,0.6)", annotation_text="Ausgangswert (0%)")
                fig.update_layout(
                    title=f"Wertentwicklung in Prozent seit Start ({timeframe.upper()})",
                    xaxis_title="Datum / Uhrzeit",
                    yaxis_title="Gewinn / Verlust (%)",
                    yaxis_ticksuffix="%",
                    hovermode="x unified",
                    margin=dict(l=10, r=10, t=50, b=10),
                    height=420
                )
            else:
                for i, t in enumerate(tickers_to_plot):
                    s = series_dict[t]
                    fig.add_trace(go.Scatter(
                        x=s.index, y=s, mode='lines', name=t,
                        line=dict(width=2.5, color=palette[i % len(palette)]),
                        hovertemplate=f"<b>{t}</b>: %{{y:.2f}} (EUR/USD)<extra></extra>"
                    ))
                fig.update_layout(
                    title=f"Preis pro Einzelaktie ({timeframe.upper()})",
                    xaxis_title="Datum / Uhrzeit",
                    yaxis_title="Preis pro Aktie in € / $",
                    hovermode="x unified",
                    margin=dict(l=10, r=10, t=50, b=10),
                    height=420
                )

            st.plotly_chart(fig, use_container_width=True)

# TAB 6: Risikostreuung
with tab6:
    st.subheader("🥧 Risikostreuung & Einseitigkeit (Klumpenrisiko)")
    st.caption("Ein gutes Depot verteilt das Geld auf verschiedene Branchen und Länder, damit ein Branchen-Crash dich nicht hart trifft.")
    if not stock_df.empty:
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            fig_sec = px.pie(stock_df, names="Sektor", values="_raw_val", title="Aufteilung nach Branchen", hole=0.4)
            fig_sec.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_sec, use_container_width=True)
        with col_d2:
            fig_geo = px.pie(stock_df, names="Land", values="_raw_val", title="Aufteilung nach Ländern", hole=0.4)
            fig_geo.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_geo, use_container_width=True)

# TAB 7: Aktien-Vergleich
with tab7:
    st.subheader("⚔️ Direktes Aktien-Duell (1 gegen 1)")
    st.caption("Vergleiche zwei deiner Aktien direkt miteinander. Die KI kürt die aktuell attraktivere Aktie.")
    if not stock_df.empty and len(stock_df) >= 2:
        cd1, cd2 = st.columns(2)
        with cd1:
            duel_a = st.selectbox("Erste Aktie:", stock_df["Ticker"].tolist(), index=0)
        with cd2:
            duel_b = st.selectbox("Zweite Aktie:", stock_df["Ticker"].tolist(), index=1)
        
        if st.button("⚡ Wer ist aktuell der bessere Kauf?", use_container_width=True):
            if GROQ_KEY:
                cl = Groq(api_key=GROQ_KEY.strip())
                row_a = stock_df[stock_df["Ticker"] == duel_a].iloc[0].to_dict()
                row_b = stock_df[stock_df["Ticker"] == duel_b].iloc[0].to_dict()
                with st.spinner("Prüfe Bewertung, Kurse und Trends..."):
                    res_duel = run_duel_analysis(cl, selected_model, str(row_a), str(row_b))
                    st.markdown(res_duel)

# GROSSER ANALYSE BUTTON
if st.button("🚀 Gesamte KI-Auswertung starten", use_container_width=True):
    if not GROQ_KEY:
        st.error("⚠️ Kein GROQ_API_KEY hinterlegt!")
    elif not st.session_state.my_portfolio:
        st.error("⚠️ Bitte füge zuerst Aktien zum Depot hinzu!")
    else:
        save_portfolio_to_file(st.session_state.my_portfolio)
        client = Groq(api_key=GROQ_KEY.strip())

        with st.spinner("Analysiere Nachrichten, Kurse, Preisschilder und Trends verständlich auf Deutsch..."):
            news_data = fetch_all_headlines()
            news_text = "\n".join(news_data)
            ticker_news_text = "\n".join(ticker_news) if ticker_news else "Keine aktuellen Sondermeldungen."

            metrics_summary = stock_df[[
                "Name / Aktie", "Ticker", "Aktueller Kurs", "RSI (14D)", 
                "KGV (P/E)", "Fair Value", "Analysten-Kursziel", "Konsens-Rating", "Dividendenrendite"
            ]].to_string(index=False)

            cluster_context = stock_df[["Name / Aktie", "Ticker", "Sektor", "Land", "Positionswert"]].to_string(index=False)

            try:
                out_market, out_depot, out_signals, out_cluster = run_analysis(
                    client, selected_model, news_text, metrics_summary, ticker_news_text, cluster_context
                )

                with tab1:
                    st.caption(f"🤖 Verwendetes KI-Modell: `{selected_model}`")
                    st.markdown(out_market)

                with tab2:
                    st.subheader("💼 Dein Depot im Überblick")
                    st.dataframe(
                        stock_df[["Name / Aktie", "Ticker", "Stückzahl", "Kaufkurs", "Aktueller Kurs", "Positionswert", "Gewinn / Verlust"]],
                        hide_index=True
                    )
                    st.divider()
                    st.markdown(out_depot)

                with tab3:
                    st.subheader("📅 Termine & Gewinnausschüttungen")
                    with st.expander("ℹ️ Was bedeutet Dividendenrendite?"):
                        st.write("Die **Dividende** ist der Gewinnanteil, den ein Unternehmen regelmäßig (meist quartalsweise oder jährlich) bar auf dein Verrechnungskonto überweist. 3 % bedeutet: Bei 1.000 € Anlage bekommst du 30 € pro Jahr.")
                    st.dataframe(
                        stock_df[["Name / Aktie", "Ticker", "Dividendenrendite", "Nächste Quartalszahlen"]].rename(columns={
                            "Dividendenrendite": "Gewinnausschüttung (% p.a.)",
                            "Nächste Quartalszahlen": "Nächster Geschäftsbericht"
                        }),
                        hide_index=True
                    )

                with tab4:
                    st.subheader("🎯 Kauf- / Verkauf-Tipps & Faire Bewertung")
                    with st.expander("ℹ️ Erklärung zu den Bewertungs-Kennzahlen"):
                        st.write("""
                        - **Aktueller Kurs**: Was 1 Aktie gerade an der Börse kostet.
                        - **Faire Wertschätzung (Fair Value)**: Was die Aktie rechnerisch anhand von Gewinn und Substanz wert sein sollte. Liegt der Kurs deutlich darunter, gilt sie als günstig.
                        - **Experten-Kursziel**: Wo Analysten großer Banken die Aktie in 12 Monaten sehen.
                        - **Experten-Empfehlung**: Die Mehrheitseinschätzung der Profis (Kaufen, Halten oder Verkaufen).
                        """)
                    st.dataframe(
                        stock_df[["Name / Aktie", "Ticker", "Aktueller Kurs", "Fair Value", "Analysten-Kursziel", "Konsens-Rating"]].rename(columns={
                            "Fair Value": "Faire Wertschätzung",
                            "Analysten-Kursziel": "Experten-Kursziel",
                            "Konsens-Rating": "Experten-Empfehlung"
                        }),
                        hide_index=True
                    )
                    st.divider()
                    st.markdown(out_signals)

                with tab6:
                    st.divider()
                    st.subheader("🛡️ KI-Gutachten zur Risikostreuung")
                    st.markdown(out_cluster)

            except Exception as e:
                st.error(f"Fehler bei der Analyse: {str(e)}")
