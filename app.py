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

# Initialisierung des Portfolios
if "my_portfolio" not in st.session_state:
    st.session_state.my_portfolio = load_saved_portfolio()

if "tr_cash" not in st.session_state:
    st.session_state.tr_cash = 0.0

# --- SEITENLEISTE: DIREKTE EINGABE & ANPASSUNG ---
with st.sidebar:
    st.header("💼 Mein Depot & Trade Republic")
    
    # 1. Trade Republic Cash
    new_cash = st.number_input(
        "💶 TR Verrechnungskonto (Cash in €):",
        min_value=0.0,
        value=float(st.session_state.tr_cash),
        step=50.0,
        help="Dein uninvestiertes Bargeld auf dem Verrechnungskonto."
    )
    if new_cash != st.session_state.tr_cash:
        st.session_state.tr_cash = new_cash

    # 2. PDF-Import (optional)
    with st.expander("📥 TR-Kontoauszug (PDF) importieren"):
        st.caption("Lade hier deinen PDF-Depotauszug hoch, um Werte automatisch einzulesen.")
        tr_pdf = st.file_uploader("PDF hochladen", type=["pdf"])
        if tr_pdf:
            imported = parse_trade_republic_pdf(tr_pdf)
            if imported:
                st.session_state.my_portfolio = imported
                save_portfolio_to_file(st.session_state.my_portfolio)
                st.success(f"{len(imported)} Positionen übernommen!")
                st.rerun()

    st.divider()
    
    # 3. Neue Aktie suchen und hinzufügen
    st.subheader("🔍 Neue Aktie hinzufügen:")
    search_query = st.text_input("Aktie/ETF suchen:", placeholder="z. B. Novo Nordisk, Rheinmetall...")
    if search_query:
        results = search_ticker_candidates(search_query)
        if results:
            selected_cand = st.selectbox("Treffer:", results, key="side_search_select")
            c_add1, c_add2 = st.columns(2)
            with c_add1:
                add_shares = st.number_input("Stückzahl", min_value=0.01, value=1.0, step=1.0, key="add_shares_input")
            with c_add2:
                add_buy = st.number_input("Kaufkurs (€)", min_value=0.0, value=0.0, step=10.0, key="add_buy_input")
            
            if st.button("➕ Zum Depot hinzufügen", use_container_width=True):
                sym = clean_ticker(selected_cand)
                name = selected_cand.split("(")[1].replace(")", "") if "(" in selected_cand else sym
                st.session_state.my_portfolio.append({
                    "ticker": sym,
                    "name": name,
                    "shares": float(add_shares),
                    "buy_price": float(add_buy)
                })
                save_portfolio_to_file(st.session_state.my_portfolio)
                st.success(f"{sym} hinzugefügt!")
                st.rerun()

    st.divider()
    
    # 4. DIREKTE ANPASSUNG ALLER BESTEHENDEN POSITIONEN
    st.subheader("📝 Bestände & Geldbeträge anpassen:")
    st.caption("Passe Stückzahlen und Kaufkurse jederzeit an. Die Gesamtsummen berechnen sich sofort neu.")
    
    portfolio_changed = False
    if st.session_state.my_portfolio:
        for idx, item in enumerate(st.session_state.my_portfolio):
            with st.expander(f"📌 {item.get('name', item['ticker'])} ({item['ticker']})", expanded=False):
                col_inp1, col_inp2 = st.columns(2)
                with col_inp1:
                    updated_shares = st.number_input(
                        "Anzahl Anteile:",
                        min_value=0.0,
                        value=float(item.get("shares", 1.0)),
                        step=0.5,
                        key=f"sh_{idx}_{item['ticker']}"
                    )
                with col_inp2:
                    updated_buy = st.number_input(
                        "Kaufkurs (€/Stk):",
                        min_value=0.0,
                        value=float(item.get("buy_price", 0.0)),
                        step=5.0,
                        key=f"bp_{idx}_{item['ticker']}"
                    )
                
                # Gesamt investierter Betrag als Info
                invested_calc = updated_shares * updated_buy
                st.write(f"👉 **Investiert:** `{invested_calc:,.2f} €`")
                
                # Prüfen auf Änderungen
                if updated_shares != item.get("shares") or updated_buy != item.get("buy_price"):
                    item["shares"] = float(updated_shares)
                    item["buy_price"] = float(updated_buy)
                    portfolio_changed = True
                
                if st.button("🗑️ Diese Aktie löschen", key=f"del_{idx}_{item['ticker']}"):
                    st.session_state.my_portfolio.pop(idx)
                    save_portfolio_to_file(st.session_state.my_portfolio)
                    st.rerun()
        
        if portfolio_changed or st.button("💾 Werte speichern", use_container_width=True):
            save_portfolio_to_file(st.session_state.my_portfolio)
            st.success("✅ Gespeichert & neu berechnet!")
            st.rerun()
    else:
        st.info("Noch keine Positionen im Depot.")

    st.divider()
    st.header("🤖 KI-Modell")
    if GROQ_KEY:
        available_models = get_account_models(GROQ_KEY)
        selected_model = st.selectbox("Modell", available_models, index=0)
    else:
        selected_model = "llama-3.3-70b-versatile"

# 8 TABS
tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🏦 Trade Republic Konto",
    "🌍 Welt-Nachrichten",
    "💼 Stimmung & Depot",
    "📅 Termine & Dividenden",
    "🎯 Kauf- / Verkauf-Tipps",
    "📊 Kurs-Diagramme",
    "🥧 Risikostreuung",
    "⚔️ Aktien-Vergleich"
])

# Daten laden & Gesamtrechnung auf Basis deiner echten Werte
if st.session_state.my_portfolio:
    stock_df, ticker_news, resolved_tickers = get_stock_data(st.session_state.my_portfolio)
    stock_val = stock_df["_raw_val"].sum()
    total_invested = stock_df["_raw_invested"].sum()
    total_tr_account = stock_val + st.session_state.tr_cash
    stock_pnl = stock_val - total_invested if total_invested > 0 else 0.0
    stock_pnl_pct = (stock_pnl / total_invested * 100) if total_invested > 0 else 0.0

    # Haupt-Banner
    c_m1, c_m2, c_m3 = st.columns(3)
    with c_m1:
        st.metric("TR Gesamtkonto", f"{total_tr_account:,.2f} €", help="Aktienwert + uninvestiertes Cash-Guthaben")
    with c_m2:
        st.metric("Tatsächlich investiert", f"{total_invested:,.2f} €", help="Summe des Geldes, das du für alle Aktienkäufe bezahlt hast")
    with c_m3:
        st.metric("Dein Reingewinn / Verlust", f"{stock_pnl:+,.2f} €", delta=f"{stock_pnl_pct:+.2f}%", help="Reiner Buchgewinn/-verlust gegenüber deinem Kaufpreis")
else:
    stock_df = pd.DataFrame()
    ticker_news = []
    resolved_tickers = []
    stock_val = 0.0
    total_tr_account = st.session_state.tr_cash

# TAB 0: SPEZIELLER TRADE REPUBLIC KONTO TAB
with tab0:
    st.subheader("🏦 Mein Trade Republic Depot-Spiegel")
    
    col_tr1, col_tr2 = st.columns(2)
    with col_tr1:
        st.info(f"💶 **Verrechnungskonto (Cash):** {st.session_state.tr_cash:,.2f} €")
    with col_tr2:
        st.success(f"📈 **Aktueller Aktienwert (Depot):** {stock_val:,.2f} €")
        
    st.write("### Deine echten Positionen im Überblick:")
    if not stock_df.empty:
        st.dataframe(
            stock_df[[
                "Name / Aktie", "Ticker", "Stückzahl", "Kaufkurs", 
                "Aktueller Kurs", "Positionswert", "Gewinn / Verlust"
            ]],
            hide_index=True
        )
    else:
        st.info("Passe deine Aktien und Stückzahlen in der linken Seitenleiste an.")

# TAB 5: Live-Charts
with tab5:
    st.subheader("📊 Kursentwicklung & Performance")
    c1, c2 = st.columns([1, 1])
    with c1:
        timeframe = st.selectbox(
            "Zeitraum auswählen:",
            options=["1d", "5d", "1mo", "6mo", "1y", "5y"],
            index=2,
            format_func=lambda x: {
                "1d": "1 Tag (Live heute)", "5d": "5 Tage", "1mo": "1 Monat",
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

        with st.spinner("Analysiere Nachrichten, Kurse und Trends verständlich auf Deutsch..."):
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
                    st.dataframe(
                        stock_df[["Name / Aktie", "Ticker", "Dividendenrendite", "Nächste Quartalszahlen"]].rename(columns={
                            "Dividendenrendite": "Gewinnausschüttung (% p.a.)",
                            "Nächste Quartalszahlen": "Nächster Geschäftsbericht"
                        }),
                        hide_index=True
                    )

                with tab4:
                    st.subheader("🎯 Kauf- / Verkauf-Tipps & Faire Bewertung")
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
