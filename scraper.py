import os
import time

import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import csv


def scrape_images(url, csv_file_path):
    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)

    chrome_options = Options()
    chrome_options.add_argument("--disable-cookies")
    chrome_options.add_argument("--headless")

    chromedriver_path = "C:\\Users\\lucho\\OneDrive\\Documents\\cours\\Esgi3\\projet_annuel\\chrome driver\\chromedriver.exe"
    driver = webdriver.Chrome()
    driver.maximize_window()

    page_number = 1
    with open(csv_file_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Image URL"])
        while True:
            url_base = f"{url}?pagi={page_number}"
            driver.get(url_base)
            time.sleep(5)

            for _ in range(10):
                driver.execute_script("window.scrollBy(0, 1000);")
                time.sleep(0.5)

            image_elements = driver.find_elements(By.CSS_SELECTOR, 'a.link--WHWzm img')
            for i, image_element in enumerate(image_elements, start=1):
                image_url = image_element.get_attribute('src')
                print(image_url)
                writer.writerow([image_url])

            try:
                page_number += 1
            except:
                break

        driver.quit()


def download_image(image_url, download_path, image_number):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    response = requests.get(image_url, headers=headers)

    # Obtenez l'extension de fichier d'origine
    file_extension = os.path.splitext(image_url)[1]
    image_filename = f"vache-{image_number}{file_extension}"

    with open(os.path.join(download_path, image_filename), 'wb') as out_file:
        out_file.write(response.content)


base_url = "https://pixabay.com/fr/images/search/vache/"
download_path = "C:\\Users\\lucho\\OneDrive\\Documents\\cours\\Esgi3\\projet_annuel\\DataSet\\vache"
csv_path = "C:\\Users\\lucho\\OneDrive\\Documents\\cours\\Esgi3\\projet_annuel\\DataSet\\vache\\vache_link.csv"

scrape_images(base_url, csv_path)
