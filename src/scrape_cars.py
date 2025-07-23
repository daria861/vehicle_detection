import requests
from bs4 import BeautifulSoup
import os

def download_image(url, folder, img_name):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(os.path.join(folder, img_name), 'wb') as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)

def scrape_car_images(car_model, search_url, output_folder):
    response = requests.get(search_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    # Find image tags (replace with site's image tag selector)
    img_tags = soup.find_all('img')
    os.makedirs(output_folder, exist_ok=True)
    for i, img in enumerate(img_tags):
        src = img.get('src')
        if src and src.startswith('http'):
            download_image(src, output_folder, f"{car_model}_{i}.jpg")

# Example usage https://www.carzone.ie/cars/toyota/corolla
scrape_car_images(
    car_model='Toyota C-HR',
    search_url='https://www.carzone.ie/cars/toyota/c-hr',
    output_folder='./dataset/Toyota_C-HR'
)