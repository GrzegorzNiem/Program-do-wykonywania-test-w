import streamlit as st
import pandas as pd
import os
import time
import numpy as np

from models import ModelBenchmarker
from scraper import RealEstateScraper
from cleaner import DataCleaner

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

st.set_page_config(page_title="Praca dyplomowa WSIZ", layout="wide")

if 'models_trained' not in st.session_state:
    st.session_state['models_trained'] = False
    st.session_state['trained_models'] = {}
    st.session_state['results_df'] = None

st.title("Zintegrowany System Analizy Nieruchomości WSIZ PRACA MAGISTERSKA")
st.markdown("---")

tab_scraper, tab_cleaner, tab_ml, tab_predict = st.tabs([
    "Moduł Pobierania danych z witryny internetowej",
    "Moduł Transformacji danych",
    "Eksperyment Nauczania maszynowego",
    "Prototyp wyceny mieszkań na podstawie różnych modeli"
])

with tab_scraper:
    st.header("Pobieranie surowych danych")

    col1, col2 = st.columns([1, 2])
    with col1:
        pages_to_scrape = st.number_input("Ile stron ogłoszeń przejrzeć w poszukiwaniu linków?", min_value=1,
                                          max_value=50, value=2)
        if st.button("Start", use_container_width=True):
            with st.spinner(
                    "Trwa pobieranie danych"):
                try:
                    base_url = "https://www.nieruchomosci-online.pl/szukaj.html?3,mieszkanie,sprzedaz,,Rzeszów"
                    scraper = RealEstateScraper(base_url=base_url, pages_to_scrape=pages_to_scrape)
                    df_scraped = scraper.scrape()

                    if df_scraped is not None:
                        st.success("Zapisano dane szczegółowe.")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.warning("Nie znaleziono danych szczegółowych ofert.")
                except Exception as e:
                    st.error(f"Błąd scrapowania: {e}")
    with col2:
        st.write("**Podgląd pobranego zbioru szczegółowego:**")
        try:
            if not os.path.exists(DATA_DIR):
                os.makedirs(DATA_DIR)
            raw_files = [f for f in os.listdir(DATA_DIR) if f.startswith('surowe_oferty')]
            if raw_files:
                latest_raw = max(raw_files)
                df_raw = pd.read_csv(os.path.join(DATA_DIR, latest_raw))
                st.dataframe(df_raw.head(10), use_container_width=True)
            else:
                st.warning("Brak surowych danych. Uruchom scraper.")
        except Exception as e:
            st.error(f"Błąd: {e}")


with tab_cleaner:
    st.header("Zaawansowane czyszczenie i inżynieria cech")

    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("Oczyść dane wejściowe", use_container_width=True):
            with st.spinner("Trwa oczyszczanie danych"):
                try:
                    cleaner = DataCleaner(data_folder=DATA_DIR)
                    cleaned_df = cleaner.clean()
                    if cleaned_df is not None:
                        st.success("Wyczyszczono i utworzono dane_czyste_ml.csv")
                        time.sleep(1)
                        st.rerun()
                except Exception as e:
                    st.error(f"Błąd czyszczenia: {e}")
    with col2:
        st.write("**Macierz przygotowana do modeli matematycznych (Wszystkie cechy numeryczne):**")
        clean_file_path = os.path.join(DATA_DIR, 'dane_czyste_ml.csv')
        if os.path.exists(clean_file_path):
            df_clean = pd.read_csv(clean_file_path)
            st.dataframe(df_clean.head(10), use_container_width=True)
        else:
            st.warning("Brak pliku 'dane_czyste_ml.csv'.")


with tab_ml:
    st.header("Uczenie  modeli ML")
    clean_file_path = os.path.join(DATA_DIR, 'dane_czyste_ml.csv')

    if st.button("Trenuj modele", type="primary", use_container_width=True):
        with st.spinner("Uczenie algorytmów i wyliczanie wag cech..."):
            benchmarker = ModelBenchmarker(data_path=clean_file_path)
            X_train, X_test, y_train, y_test = benchmarker.load_and_split_data()

            if X_train is not None:
                results = {}
                for name, model in benchmarker.models.items():
                    results[name] = benchmarker.benchmark(model, X_train, X_test, y_train, y_test)
                    st.session_state['trained_models'][name] = model

                st.session_state['results_df'] = pd.DataFrame(results).T
                st.session_state['models_trained'] = True
                st.success("Modele wytrenowane")
            else:
                st.error("Brak wyczyszczonych danych.")

    if st.session_state['models_trained']:
        st.subheader("Wyniki modeli")
        st.dataframe(st.session_state['results_df'].style.highlight_min(subset=['MAE [PLN]'], color='#a8e6cf'),
                     use_container_width=True)

        st.markdown("---")
        st.subheader("Wagi cech")
        st.write(
            "Wybierz model, aby zobaczyć, którym parametrom przypisuje największe znaczenie podczas wyceny mieszkania.")

        # Wybór modelu z listy
        model_names = list(st.session_state['trained_models'].keys())
        selected_model = st.selectbox("Wybierz model do analizy:", model_names)

        if selected_model:
            model = st.session_state['trained_models'][selected_model]
            features = ['Powierzchnia', 'Pokoje', 'Piętro', 'Rok budowy', 'Parking', 'Stan']
            importances = []

            if "Regresja Liniowa" in selected_model:
                lr_model = model[1]
                coefs = np.abs(lr_model.coef_)
                importances = coefs / np.sum(coefs)
            else:
                importances = model.feature_importances_

            df_imp = pd.DataFrame({'Wpływ na cenę': importances}, index=features)
            df_imp = df_imp.sort_values(by='Wpływ na cenę', ascending=False)

            chart_color = "#3498db" if "Regresja" in selected_model else "#2ecc71"
            st.bar_chart(df_imp, color=chart_color)


with tab_predict:
    st.header("Estymator wartości nieruchomości online")

    if not st.session_state['models_trained']:
        st.warning("Najpierw wykonaj trening modeli")
    else:
        st.write("Skonfiguruj pełną charakterystykę mieszkania:")

        c1, c2, c3 = st.columns(3)
        with c1:
            user_area = st.number_input("Powierzchnia [m²]", min_value=15.0, max_value=200.0, value=50.0, step=0.5)
            user_rooms = st.slider("Liczba pokoi", min_value=1, max_value=6, value=2)
        with c2:
            user_floor = st.number_input("Piętro", min_value=0, max_value=20, value=1, step=1, help="0 oznacza Parter")
            user_year = st.number_input("Rok budowy", min_value=1900, max_value=2026, value=2015, step=1)
        with c3:
            parking_choice = st.radio("Miejsce parkingowe / Garaż", ["Brak", "Tak (Posiada)"])
            user_parking = 1 if parking_choice == "Tak (Posiada)" else 0

            stan_choice = st.radio("Stan nieruchomości",
                                   ["Surowy / Deweloperski / Do remontu", "Wykończony / Gotowy do zamieszkania"])
            user_stan = 0 if "surowy" in stan_choice.lower() else 1

        input_data = pd.DataFrame([{
            'Powierzchnia': user_area,
            'Pokoje': user_rooms,
            'Pietro': user_floor,
            'Rok_budowy': user_year,
            'Miejsce_parkingowe': user_parking,
            'Stan': user_stan
        }])

        st.markdown("<br><hr>", unsafe_allow_html=True)

        cols = st.columns(3)
        for idx, (name, model) in enumerate(st.session_state['trained_models'].items()):
            pred_price = model.predict(input_data)[0]
            price_formatted = f"{pred_price:,.0f} zł".replace(',', ' ')
            price_per_m = f"{(pred_price / user_area):,.0f} zł/m²".replace(',', ' ')

            col_idx = idx % 3
            cols[col_idx].metric(label=f" {name}", value=price_formatted, delta=price_per_m, delta_color="off")