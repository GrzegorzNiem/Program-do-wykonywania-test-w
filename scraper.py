import time
import random
import logging
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')


class RealEstateScraper:
    def __init__(self, base_url, pages_to_scrape=1):
        self.base_url = base_url
        self.pages_to_scrape = pages_to_scrape
        self.data = []

        options = Options()
        options.add_argument("--headless")
        options.add_argument("--disable-gpu")
        options.add_argument("window-size=1920,1080")
        options.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

        self.driver = webdriver.Chrome(options=options)

    def scrape(self, progress_callback=None):
        logging.info(f"Rozpoczynam zbieranie linków z: {self.base_url}")
        detail_links = []

        for page in range(1, self.pages_to_scrape + 1):
            separator = "&" if "?" in self.base_url else "?"
            url = f"{self.base_url}{separator}page={page}"
            try:
                self.driver.get(url)
                time.sleep(random.uniform(2, 4))
                soup = BeautifulSoup(self.driver.page_source, 'html.parser')

                tiles = soup.find_all('div', class_=lambda c: c and 'tile' in c.lower())
                for tile in tiles:
                    a_tag = tile.find('a', href=True)
                    if a_tag and 'nieruchomosci-online.pl' in a_tag['href']:
                        detail_links.append(a_tag['href'])
                    elif a_tag and a_tag['href'].startswith('/'):
                        detail_links.append("https://www.nieruchomosci-online.pl" + a_tag['href'])
            except Exception as e:
                logging.error(f"Błąd zbierania linków na stronie {page}: {e}")

        detail_links = list(set(detail_links))
        logging.info(f"Łącznie znaleziono {len(detail_links)} unikalnych ofert do zbadania.")

        # KROK 2: Pobieranie z metryczki Z AKTUALIZACJĄ INTERFEJSU
        for idx, link in enumerate(detail_links):
            # ---------------------------------------------------------
            # WYSYŁANIE STATUSU DO STREAMLIT
            # ---------------------------------------------------------
            if progress_callback:
                progress_callback(idx + 1, len(detail_links), link)

            logging.info(f"Przetwarzanie ogłoszenia {idx + 1}/{len(detail_links)}...")
            try:
                self.driver.get(link)
                time.sleep(random.uniform(1.5, 3.5))
                page_soup = BeautifulSoup(self.driver.page_source, 'html.parser')

                self._parse_detail_page(page_soup, link)
            except Exception as e:
                logging.error(f"Błąd otwierania podstrony ogłoszenia: {e}")

        self.driver.quit()
        return self._save_to_csv()

    def _parse_detail_page(self, soup, link):
        try:
            # 1. Pobieranie ceny z głównego nagłówka
            price_box = soup.find(class_=lambda c: c and 'price' in str(c).lower())
            price = price_box.text.strip() if price_box else None

            # Domyślne puste wartości
            area, rooms, floor, year, parking, stan = None, None, None, None, None, None

            for box in soup.find_all(['div', 'li', 'span']):
                text = box.text.strip().lower()

                if len(text) > 60:
                    continue


                if 'm²' in text and not area:
                    if 'zł' in text or 'pln' in text:
                        continue
                    area = box.text.strip()

                if 'liczba pokoi:' in text:
                    rooms = text.replace('liczba pokoi:', '').strip()
                elif 'piętro:' in text:
                    floor = text.replace('piętro:', '').strip()
                elif 'rok budowy:' in text:
                    year = text.replace('rok budowy:', '').strip()
                elif 'miejsce parkingowe:' in text:
                    parking = text.replace('miejsce parkingowe:', '').strip()
                elif 'stan mieszkania:' in text:
                    stan = text.replace('stan mieszkania:', '').strip()

            if price:
                self.data.append({
                    'Cena_surowa': price,
                    'Powierzchnia_surowa': area,
                    'Pokoje_surowe': rooms,
                    'Pietro_surowe': floor,
                    'Rok_surowy': year,
                    'Parking_surowy': parking,
                    'Stan_surowy': stan,
                    'Link': link
                })
        except Exception as e:
            logging.error(f"Błąd parsowania szczegółów: {e}")

    def _save_to_csv(self):
        if not self.data:
            return None
        df = pd.DataFrame(self.data)
        os.makedirs(DATA_DIR, exist_ok=True)
        filename = os.path.join(DATA_DIR, f"surowe_oferty_{int(time.time())}.csv")
        df.to_csv(filename, index=False, encoding='utf-8')
        return df