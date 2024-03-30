import requests
import json
import time

# Replace YOUR_API_KEY with your actual API key
API_KEY = ''

place_id = 'ChIJX9SSyAirQjQR9B9f7PKLqJo'
    # URL for Place Reviews API
reviews_url = f'https://maps.googleapis.com/maps/api/place/details/json?place_id={place_id}&fields=reviews&key={API_KEY}'
reviews_new_url =  f'https://maps.googleapis.com/maps/api/place/details/json?place_id={place_id}&fields=reviews&reviews_sort=newest&key={API_KEY}'
    # Get place reviews
reviews_response = requests.get(reviews_url)
reviews_data = reviews_response.json()
print(reviews_data)
time.sleep(2)
print()
reviews_response = requests.get(reviews_new_url)
reviews_data = reviews_response.json()
print(reviews_data)
