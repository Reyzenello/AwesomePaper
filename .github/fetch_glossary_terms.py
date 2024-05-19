import requests
from bs4 import BeautifulSoup
import os

# Define the URL to fetch new terms and definitions
TERMS_API_URL = "https://developers.google.com/machine-learning/glossary"

# Define the path to the glossary file
GLOSSARY_FILE_PATH = "glossary.md"

def fetch_terms():
    try:
        response = requests.get(TERMS_API_URL)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.text
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except requests.exceptions.ConnectionError as conn_err:
        print(f"Connection error occurred: {conn_err}")
    except requests.exceptions.Timeout as timeout_err:
        print(f"Timeout error occurred: {timeout_err}")
    except requests.exceptions.RequestException as req_err:
        print(f"An error occurred: {req_err}")
    return ""

def parse_terms(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    glossary = {}

    glossary_items = soup.find_all('h2', class_='hide-from-toc')
    for item in glossary_items:
        term = item.get_text(strip=True)
        definition = []
        
        # Iterate over the next siblings until we find a new term or no more siblings
        for sibling in item.find_next_siblings():
            if sibling.name == 'h2':  # Stop if we reach the next term
                break
            if sibling.name in ['p', 'div']:
                definition.append(sibling.get_text(strip=True))
        
        # Join all parts of the definition
        definition_text = ' '.join(definition)
        glossary[term] = definition_text

    return glossary

def update_glossary(terms):
    if not os.path.exists(GLOSSARY_FILE_PATH):
        with open(GLOSSARY_FILE_PATH, 'w') as f:
            f.write("# Machine Learning Glossary\n\n## Terms and Definitions\n\n")

    with open(GLOSSARY_FILE_PATH, 'a') as f:
        for term, definition in terms.items():
            f.write(f"### {term}\n\n{definition}\n\n")

def main():
    html_content = fetch_terms()
    if html_content:
        terms = parse_terms(html_content)
        if terms:
            update_glossary(terms)
        else:
            print("No terms found in the glossary.")
    else:
        print("Failed to fetch the glossary page. Check the URL and try again.")

if __name__ == "__main__":
    main()
