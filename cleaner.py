import pandas as pd
import os
import glob
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(BASE_DIR, 'data')

class DataCleaner:
    def __init__(self, data_folder=DEFAULT_DATA_DIR):
        self.data_folder = data_folder

    def _get_latest_file(self):
        list_of_files = glob.glob(os.path.join(self.data_folder, 'surowe_oferty_*.csv'))
        if not list_of_files:
            return None
        return max(list_of_files, key=os.path.getctime)

    def clean(self):
        latest_file = self._get_latest_file()
        if not latest_file:
            return None

        df = pd.read_csv(latest_file)

        df['price_clean'] = df['Cena_surowa'].astype(str).str.replace(r'\s+', '', regex=True).str.replace(' ', '')
        df['Cena'] = df['price_clean'].str.extract(r'(\d+)')[0]
        df['Cena'] = pd.to_numeric(df['Cena'], errors='coerce')

        df['area_clean'] = df['Powierzchnia_surowa'].astype(str).str.replace(r'\s+', '', regex=True).str.replace(',', '.')
        df['Powierzchnia'] = df['area_clean'].str.extract(r'(\d+\.?\d*)')[0]
        df['Powierzchnia'] = pd.to_numeric(df['Powierzchnia'], errors='coerce')

        df['Pokoje'] = df['Pokoje_surowe'].astype(str).str.extract(r'(\d+)')
        df['Pokoje'] = pd.to_numeric(df['Pokoje'], errors='coerce')
        df['Pokoje'] = df['Pokoje'].fillna((df['Powierzchnia'] / 25).round())


        def czysc_pietro(val):
            val = str(val).lower()
            if 'parter' in val:
                return 0
            if 'suterena' in val:
                return -1

            import re
            match = re.search(r'(\d+)', val.split('/')[0] if '/' in val else val)
            if match:
                return int(match.group(1))
            return 1

        df['Pietro'] = df['Pietro_surowe'].apply(czysc_pietro)

        df['Rok_budowy'] = df['Rok_surowy'].astype(str).str.extract(r'(\d{4})')
        df['Rok_budowy'] = pd.to_numeric(df['Rok_budowy'], errors='coerce')
        df['Rok_budowy'] = df['Rok_budowy'].fillna(df['Rok_budowy'].median())


        df['Miejsce_parkingowe'] = df['Parking_surowy'].astype(str).str.lower().apply(
            lambda x: 0 if '-' in x or 'brak' in x or 'nie' in x or 'nan' in x else 1
        )

        def ocen_stan(tekst):
            t = str(tekst).lower()
            if 'remont' in t or 'surowy' in t or 'deweloperski' in t or 'odświeżenia' in t:
                return 0
            else:
                return 1
        df['Stan'] = df['Stan_surowy'].apply(ocen_stan)

        df_clean = df.dropna(subset=['Cena', 'Powierzchnia'])
        df_clean = df_clean[
            (df_clean['Cena'] > 100000) & (df_clean['Cena'] < 3000000) &
            (df_clean['Powierzchnia'] > 15) & (df_clean['Powierzchnia'] < 200)
            ]

        final_columns = ['Cena', 'Powierzchnia', 'Pokoje', 'Pietro', 'Rok_budowy', 'Miejsce_parkingowe', 'Stan', 'Link']
        df_final = df_clean[final_columns].copy()

        output_filename = os.path.join(self.data_folder, "dane_czyste_ml.csv")
        df_final.to_csv(output_filename, index=False, encoding='utf-8')
        return df_final