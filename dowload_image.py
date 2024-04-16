import csv

import os
import requests
from urllib.parse import urljoin

classe = "chevre"


def download_images(csv_path, download_path):
    with open(csv_path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        next(reader)  # Skip the header row
        for row in reader:
            url = row[0]  # Assuming the URL is in the first column of the CSV
            response = requests.get(url)
            filename = os.path.join(download_path, url.split("/")[-1])

            with open(filename, 'wb') as out_file:
                out_file.write(response.content)


download_path = f"C:\\Users\\Hugo HOUNTONDJI\\OneDrive\\Documents\\Projet Annuel\\PA3-BIGDATA-HLE\\DataSet\\{classe}"
csv_path = f"C:\\Users\\Hugo HOUNTONDJI\\OneDrive\\Documents\\Projet Annuel\\PA3-BIGDATA-HLE\\DataSet\\{classe}\\{classe}_link.csv"

# Créez le répertoire s'il n'existe pas
os.makedirs(download_path, exist_ok=True)

download_images(csv_path, download_path)
