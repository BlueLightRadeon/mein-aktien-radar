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
        clean_ticker, parse_trade_republic_pdf, get_display_name
    )
    from ai_service import get_account_models, run_analysis, run_duel_analysis
except Exception as e:
    st.error(f"Import-Fehler: {e}")
    st.stop()

st.set_page_config(
    page_title="KI Markt- & Depot-Radar",
    page_icon="📈",
    layout="wide"
)

st.title("📈 KI Markt- & Depot-Radar")

GROQ_KEY = st.secrets.get("GROQ_API_KEY", "")

# Portfolio initialisieren
if "my_portfolio" not in st.session_state:
    st.session_state.my_portfolio = load_saved_portfolio()
else:
    for item in st.session_state.my_portfolio:
        item["name"] = get_display_name(item.get("ticker", ""), item.get("name"))

if "tr_cash" not in st.session_state:
    st.session_state.tr_cash = 194.02

def fmt_eur(val):
    try:
        return f"{val:,.2f} €".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return f"{val} €"

# --- SEITENLEISTE ---
with st.sidebar:
    st.header("💼 Trade Republic Depot")
    
    with st.expander("📥 TR-Kontoauszug (PDF) einlesen", expanded=False):
        st.caption("Lade deinen Auszug hoch, um Bestände und Cash automatisch zu aktualisieren.")
        tr_pdf = st.file_uploader("PDF auswählen", type=["pdf"], key="tr_pdf_uploader")
        if tr_pdf is not None:
            imported_items, imported_cash = parse_trade_republic_pdf(tr_pdf)
            if imported_items:
                st.session_state.my_portfolio = imported_items
                if imported_cash is not None:
                    st.session_state.tr_cash = imported_cash
                save_portfolio_to_file(st.session_state.my_portfolio)
                st.success(f"✅ {len(imported_items)} Positionen aus Auszug übernommen!")
                st.rerun()

    st.session_state.tr_cash = st.number_input(
        "💶 TR Bargeld-Guthaben (Cash in €):",
        min_value=0.0,
        value=float(st.session_state.tr_cash),
        step=10.0
    )

    st.divider()

    st.subheader("🔍 Aktie hinzufügen:")
    search_query = st.text_input("Name oder Symbol:", placeholder="z. B. Apple, Tesla...")
    if search_query:
        results = search_ticker_candidates(search_query)
        if results:
            selected_cand = st.selectbox("Treffer:", results, key="side_search_select")
            in_money = st.number_input("Investierter Betrag (€):", min_value=1.0, value=50.0, step=10.0)
            
            if st.button("➕ Hinzufügen", use_container_width=True):
                sym = clean_ticker(selected_cand)
                disp_name = get_display_name(sym)
                st.session_state.my_portfolio.append({
                    "ticker": sym,
                    "name": disp_name,
                    "shares": 1.0,
                    "buy_price": float(in_money)
                })
                save_portfolio_to_file(st.session_state.my_portfolio)
                st.success(f"✅ {disp_name} hinzugefügt!")
                st.rerun()

    st.divider()

    st.subheader("📋 Aktive Depot-Positionen:")
    if st.session_state.my_portfolio:
        for idx, item in enumerate(list(st.session_state.my_portfolio)):
            disp_name = get_display_name(item.get("ticker", ""), item.get("name"))
            col_pos_a, col_pos_b = st.columns([3, 1])
            with col_pos_a:
                st.write(f"• **{disp_name}**")
                st.caption(f"Einsatz: {fmt_eur(float(item.get('buy_price', 0.0)))}")
            with col_pos_b:
                if st.button("❌", key=f"del_item_{idx}_{item['ticker']}", help=f"{disp_name} entfernen"):
                    st.session_state.my_portfolio.pop(idx)
                    save_portfolio_to_file(st.session_state.my_portfolio)
                    st.rerun()
        
        st.write("")
        if st.button("🗑️ Alle Positionen leeren", use_container_width=True):
            st.session_state.my_portfolio = []
            save_portfolio_to_file(st.session_state.my_portfolio)
            st.rerun()
    else:
        st.info("Keine Positionen hinterlegt.")

    st.divider()
    st.header("🤖 KI-Modell")
    if GROQ_KEY:
        available_models = get_account_models(GROQ_KEY)
        selected_model = st.selectbox("Auswahl:", available_models, index=0)
    else:
        selected_model = "llama-3.3-70b-versatile"

# KORREKTE GESAMTBERECHNUNG
if st.session_state.my_portfolio:
    stock_df, ticker_news, resolved_tickers = get_stock_data(st.session_state.my_portfolio)
    total_invested = sum([float(x.get("buy_price", 0.0)) for x in st.session_state.my_portfolio])
    stock_val = stock_df["_raw_val"].sum() if not stock_df.empty and stock_df["_raw_val"].sum() > 0 else total_invested
    total_tr_account = stock_val + st.session_state.tr_cash
    stock_pnl = stock_val - total_invested
    stock_pnl_pct = (stock_pnl / total_invested * 100) if total_invested > 0 else 0.0

    c_m1, c_m2, c_m3 = st.columns(3)
    with c_m1:
        st.metric("TR Gesamtkonto", fmt_eur(total_tr_account), help="Bargeld + Gesamtwert deiner Aktien")
    with c_m2:
        st.metric("Eingezahltes Geld", fmt_eur(total_invested), help="Dein tatsächlich eingesetztes Kapital")
    with c_m3:
        st.metric("Gewinn / Verlust", fmt_eur(stock_pnl), delta=f"{stock_pnl_pct:+.2f}%")
else:
    stock_df = pd.DataFrame()
    ticker_news = []
    resolved_tickers = []
    stock_val = 0.0
    total_invested = 0.0
    total_tr_account = st.session_state.tr_cash

    c_m1, c_m2, c_m3 = st.columns(3)
    with c_m1:
        st.metric("TR Gesamtkonto", fmt_eur(st.session_state.tr_cash))
    with c_m2:
        st.metric("Eingezahltes Geld", "0,00 €")
    with c_m3:
        st.metric("Gewinn / Verlust", "0,00 €")

# GROSSER ANALYSE BUTTON (MIT AUTOMATISCHEM SPEICHERN & RERUN)
if st.button("🚀 Gesamte KI-Auswertung starten", use_container_width=True, type="primary"):
    if not GROQ_KEY:
        st.error("⚠️ Kein GROQ_API_KEY hinterlegt! Bitte trage deinen API-Key in den Streamlit Secrets ein.")
    elif not st.session_state.my_portfolio:
        st.error("⚠️ Bitte lade zuerst dein PDF hoch oder füge Aktien hinzu!")
    else:
        save_portfolio_to_file(st.session_state.my_portfolio)
        try:
            client = Groq(api_key=GROQ_KEY.strip())
            with st.spinner("Analysiere Markt, Stimmungen und deine 8 Positionen..."):
                news_data = fetch_all_headlines()
                news_text = "\n".join(news_data) if news_data else "Aktuell keine Sondermeldungen."
                ticker_news_text = "\n".join(ticker_news) if ticker_news else "Keine aktuellen Unternehmensmeldungen."

                metrics_summary = stock_df[[
                    "Unternehmen", "Börsenkurs", "RSI (14D)", 
                    "KGV (P/E)", "Fair Value", "Analysten-Kursziel", "Konsens-Rating", "Dividendenrendite"
                ]].to_string(index=False)

                cluster_context = stock_df[["Unternehmen", "Sektor", "Land", "Rolle", "Aktueller Wert (TR)"]].to_string(index=False)

                out_m, out_d, out_s, out_c = run_analysis(
                    client, selected_model, news_text, metrics_summary, ticker_news_text, cluster_context
                )
                
                # Ergebnisse persistent speichern
                st.session_state["ai_market"] = out_m
                st.session_state["ai_depot"] = out_d
                st.session_state["ai_signals"] = out_s
                st.session_state["ai_cluster"] = out_c
                st.session_state["last_analysis_time"] = datetime.now().strftime("%H:%M:%S Uhr")
                
                st.toast("✅ Auswertung abgeschlossen!")
                st.rerun()
        except Exception as e:
            st.error(f"Fehler bei der KI-Analyse: {str(e)}")

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
    st.info("ℹ️ **Kurzinfo:** Zeigt dein reales Trade Republic Depot – getrennt nach Bargeld (Cash) und dem aktuellen Wert deiner Wertpapiere.")
    col_tr1, col_tr2 = st.columns(2)
    with col_tr1:
        st.success(f"💶 **Bargeld (Cash):** {fmt_eur(st.session_state.tr_cash)}")
    with col_tr2:
        st.success(f"📈 **Aktueller Wert deiner Aktien:** {fmt_eur(stock_val)}")
        
    st.subheader("Deine Positionen im Überblick:")
    if not stock_df.empty:
        st.dataframe(
            stock_df[[
                "Unternehmen", "Dein Geldeinsatz", 
                "Börsenkurs", "Aktueller Wert (TR)", "Gewinn / Verlust"
            ]],
            hide_index=True,
            use_container_width=True
        )
    else:
        st.info("Noch keine Positionen vorhanden. Lade links deinen TR-Auszug hoch.")

# TAB 1: WELT-NACHRICHTEN
with tab1:
    st.info("ℹ️ **Kurzinfo:** Scannt weltweite Finanzquellen und fasst die 10 wichtigsten Markt-Ereignisse zusammen.")
    if "ai_market" in st.session_state and st.session_state["ai_market"]:
        st.caption(f"🕒 Letzte Aktualisierung: {st.session_state.get('last_analysis_time', '')}")
        st.markdown(st.session_state["ai_market"])
    else:
        st.info("Klicke oben auf den roten Button **'🚀 Gesamte KI-Auswertung starten'**, um die Meldungen abzurufen.")

# TAB 2: STIMMUNG & DEPOT
with tab2:
    st.info("ℹ️ **Kurzinfo:** Einzelanalyse für jede deiner aktuell hinterlegten Aktien.")
    if not stock_df.empty:
        st.dataframe(
            stock_df[[
                "Unternehmen", "Dein Geldeinsatz", "Börsenkurs", "Aktueller Wert (TR)", "Gewinn / Verlust"
            ]],
            hide_index=True,
            use_container_width=True
        )
    if "ai_depot" in st.session_state and st.session_state["ai_depot"]:
        st.divider()
        st.subheader("🤖 KI-Stimmungsbericht:")
        st.markdown(st.session_state["ai_depot"])
    else:
        st.caption("Klicke oben auf **'🚀 Gesamte KI-Auswertung starten'**, um den Stimmungsbericht zu laden.")

# TAB 3: TERMINE & DIVIDENDEN
with tab3:
    st.info("ℹ️ **Kurzinfo:** Zeigt Quartalstermine und jährliche Dividendenausschüttungen.")
    if not stock_df.empty:
        st.dataframe(
            stock_df[[
                "Unternehmen", "Dividendenrendite", "Nächste Quartalszahlen"
            ]].rename(columns={
                "Dividendenrendite": "Gewinnausschüttung (% p.a.)",
                "Nächste Quartalszahlen": "Nächster Quartalstermin"
            }),
            hide_index=True,
            use_container_width=True
        )

# TAB 4: KAUF- / VERKAUF-TIPPS
with tab4:
    st.info("ℹ️ **Kurzinfo:** Konkrete Handlungsempfehlungen der KI basierend auf aktuellen Kurszielen und Bewertungen.")
    if not stock_df.empty:
        st.dataframe(
            stock_df[[
                "Unternehmen", "Börsenkurs", "Fair Value", "Analysten-Kursziel", "Konsens-Rating"
            ]].rename(columns={
                "Fair Value": "Faire Wertschätzung",
                "Analysten-Kursziel": "Experten-Kursziel",
                "Konsens-Rating": "Experten-Empfehlung"
            }),
            hide_index=True,
            use_container_width=True
        )
    if "ai_signals" in st.session_state and st.session_state["ai_signals"]:
        st.divider()
        st.subheader("🤖 KI-Begründungen für deine Aktien:")
        st.markdown(st.session_state["ai_signals"])
    else:
        st.caption("Klicke oben auf **'🚀 Gesamte KI-Auswertung starten'**, um die KI-Begründungen zu laden.")

# TAB 5: CHARTS
with tab5:
    st.info("ℹ️ **Kurzinfo:** Interaktive Diagramme für alle Aktien aus deinem Portfolio.")
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
            available_names = list(series_dict.keys())
            selected_view = st.selectbox("Fokus:", ["Alle Aktien gleichzeitig"] + available_names, index=0)
            
            palette = ["#00D084", "#0693E3", "#FCB900", "#EB144C", "#9B51E0", "#00ACC1", "#FF6900", "#D81B60", "#8E24AA"]
            fig = go.Figure()
            names_to_plot = available_names if selected_view == "Alle Aktien gleichzeitig" else [selected_view]

            if chart_mode == "Wertentwicklung in %":
                for i, name in enumerate(names_to_plot):
                    s = series_dict[name]
                    if not s.empty and s.iloc[0] > 0:
                        base_val = s.iloc[0]
                        pct_series = ((s - base_val) / base_val) * 100
                        fig.add_trace(go.Scatter(
                            x=s.index, y=pct_series, mode="lines", name=name,
                            line=dict(width=2.5, color=palette[i % len(palette)]),
                            hovertemplate=f"<b>{name}</b>: %{{y:+.2f}}%<extra></extra>"
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
                for i, name in enumerate(names_to_plot):
                    s = series_dict[name]
                    fig.add_trace(go.Scatter(
                        x=s.index, y=s, mode="lines", name=name,
                        line=dict(width=2.5, color=palette[i % len(palette)]),
                        hovertemplate=f"<b>{name}</b>: %{{y:.2f}}<extra></extra>"
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
    st.info("ℹ️ **Kurzinfo:** Prüft die Verteilung deines Geldes auf Rollen, Branchen und Länder.")
    
    if not stock_df.empty:
        c_pie1, c_pie2, c_pie3 = st.columns(3)
        custom_colors = ["#2E93fA", "#66DA26", "#FF9800", "#E91E63", "#546E7A", "#9C27B0", "#00ACC1", "#F4511E"]
        
        with c_pie1:
            fig_role = px.pie(
                stock_df, names="Rolle", values="_raw_val",
                title="1. Rollen im Depot", hole=0.45,
                color_discrete_sequence=["#2E93fA", "#66DA26", "#FF9800", "#546E7A"]
            )
            fig_role.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig_role, use_container_width=True)

        with c_pie2:
            fig_sec = px.pie(
                stock_df, names="Sektor", values="_raw_val",
                title="2. Branchen-Aufteilung", hole=0.45,
                color_discrete_sequence=custom_colors
            )
            fig_sec.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig_sec, use_container_width=True)

        with c_pie3:
            fig_geo = px.pie(
                stock_df, names="Land", values="_raw_val",
                title="3. Länder-Aufteilung", hole=0.45,
                color_discrete_sequence=custom_colors
            )
            fig_geo.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig_geo, use_container_width=True)

        st.divider()
        st.subheader("🧩 Wie verteilen sich deine tatsächlichen Aktien?")
        
        unique_roles = stock_df["Rolle"].unique()
        r_cols = st.columns(2)
        
        for idx, r_name in enumerate(unique_roles):
            sub_df = stock_df[stock_df["Rolle"] == r_name]
            stock_names = ", ".join(sub_df["Unternehmen"].tolist())
            role_sum = sub_df["_raw_val"].sum()
            role_pct = (role_sum / stock_val * 100) if stock_val > 0 else 0.0
            
            with r_cols[idx % 2]:
                with st.container(border=True):
                    st.markdown(f"### {r_name}")
                    st.write(f"**Deine Aktien hier:** {stock_names}")
                    st.write(f"**Anteil am Depot:** `{fmt_eur(role_sum)}` ({role_pct:.1f} %)")

    if "ai_cluster" in st.session_state and st.session_state["ai_cluster"]:
        st.divider()
        st.subheader("🛡️ KI-Gutachten & Empfehlungen zur Absicherung:")
        st.markdown(st.session_state["ai_cluster"])
    else:
        st.caption("Klicke oben auf **'🚀 Gesamte KI-Auswertung starten'**, um das KI-Risikogutachten abzurufen.")

# TAB 7: AKTIEN-VERGLEICH
with tab7:
    st.info("ℹ️ **Kurzinfo:** Direktes 1-gegen-1-Duell zweier beliebiger Aktien aus deinem Depot.")
    if not stock_df.empty and len(stock_df) >= 2:
        cd1, cd2 = st.columns(2)
        names_list = stock_df["Unternehmen"].tolist()
        with cd1:
            duel_a = st.selectbox("Erste Aktie:", names_list, index=0)
        with cd2:
            duel_b = st.selectbox("Zweite Aktie:", names_list, index=1)
        
        if st.button("⚡ Duell auswerten", use_container_width=True):
            if GROQ_KEY:
                cl = Groq(api_key=GROQ_KEY.strip())
                row_a = stock_df[stock_df["Unternehmen"] == duel_a].iloc[0].to_dict()
                row_b = stock_df[stock_df["Unternehmen"] == duel_b].iloc[0].to_dict()
                with st.spinner("Analysiere Duell..."):
                    res_duel = run_duel_analysis(cl, selected_model, str(row_a), str(row_b))
                    st.markdown(res_duel)
