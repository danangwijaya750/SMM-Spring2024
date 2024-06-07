import requests
import json
import time

# Replace YOUR_API_KEY with your actual API key
API_KEY = ''

# Define the keywords for the types of places you're looking for
keywords = ['hiking', 'peak trails', 'hiking trails', 'mountain hiking', 'trailhead', 'mountain trails','mountain peak','main trails']

# Define the country (Taiwan) to restrict the search
country = 'Taiwan'

# Initialize an empty array to store all place details
all_places = []

# Function to fetch results for a given URL and append place details to the list
def fetch_results(url):
    response = requests.get(url)
    data = response.json()
    if 'results' in data:
        for result in data['results']:
            place_id = result['place_id']
            if place_id not in [place['place_id'] for place in all_places]:
                all_places.append({'place_id': place_id, 'name': result['name'], 'details': None, 'reviews': []})
    if 'next_page_token' in data:
        next_page_token = data['next_page_token']
        time.sleep(2)  # Adding a delay to wait for next page token to become valid
        next_page_url = f'https://maps.googleapis.com/maps/api/place/textsearch/json?pagetoken={next_page_token}&key={API_KEY}'
        fetch_results(next_page_url)

# Function to get place details
def get_details(place):
    place_id = place['place_id']
    # URL for Place Details API
    details_url = f'https://maps.googleapis.com/maps/api/place/details/json?place_id={place_id}&key={API_KEY}'
    
    # Get place details
    details_response = requests.get(details_url)
    details_data = details_response.json()

    # Update place details
    place['details'] = details_data['result']

# Function to get place reviews
def get_reviews(place):
    place_id = place['place_id']

    # URL for Relevance and Newest Place Reviews API
    relevance_reviews_url = f'https://maps.googleapis.com/maps/api/place/details/json?place_id={place_id}&fields=reviews&key={API_KEY}'
    newest_reviews_url = f'https://maps.googleapis.com/maps/api/place/details/json?place_id={place_id}&fields=reviews&reviews_sort=newest&key={API_KEY}'
    
    # Get place reviews
    relevance_reviews_response = requests.get(relevance_reviews_url)
    time.sleep(2)
    newest_reviews_response = requests.get(newest_reviews_url)
    time.sleep(2)

    # Parse JSON responses
    relevance_reviews_data = relevance_reviews_response.json()
    newest_reviews_data = newest_reviews_response.json()
    
    # Extract reviews and merge them
    all_reviews = []
    if 'reviews' in relevance_reviews_data['result']:
        all_reviews.extend(relevance_reviews_data['result']['reviews'])
    
    if 'reviews' in newest_reviews_data['result']:
        all_reviews.extend(newest_reviews_data['result']['reviews'])
    
    # Check uniqueness based on author_url
    unique_reviews = {}
    for review in all_reviews:
        author_url = review.get('author_url')
        if author_url not in unique_reviews:
            unique_reviews[author_url] = review
    
    # Update place with merged and unique reviews
    place['reviews'] = list(unique_reviews.values())


# Iterate over each keyword
for keyword in keywords:
    # URL for Text Search API
    text_search_url = f'https://maps.googleapis.com/maps/api/place/textsearch/json?query={keyword}+in+{country}&key={API_KEY}'
    
    print("search keyword :" + keyword)

    # Fetch results for the initial URL
    fetch_results(text_search_url)

# Get details and all reviews for each place
for place in all_places:
    get_details(place)
    get_reviews(place)

# Write data to a JSON file
with open('places_data_new.json', 'w') as json_file:
    json.dump(all_places, json_file, indent=4)

print("Crawled "+str(len(all_places))+" Places")
print('Data has been written to places_data.json')