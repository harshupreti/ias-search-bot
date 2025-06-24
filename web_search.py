import os
import requests
from dotenv import load_dotenv

load_dotenv(dotenv_path="SERPAPI_KEY.env")

SERPAPI_KEY = os.getenv("SERPAPI_KEY")  # Set this in your environment

def search_web(query, num_results=3):
    """
    Perform a web search using SerpAPI and return a dict with:
    - 'officer_names': inferred from titles
    - 'snippets': formatted "Title: snippet" strings
    - 'urls': associated result URLs
    """
    if not SERPAPI_KEY:
        raise ValueError("❌ SERPAPI_KEY not set in environment")

    params = {
        "engine": "google",
        "q": query,
        "num": num_results,
        "api_key": SERPAPI_KEY
    }

    try:
        response = requests.get("https://serpapi.com/search", params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"🌐 Web request failed: {e}")
        return {
            "officer_names": [],
            "snippets": [],
            "urls": []
        }

    results = data.get("organic_results", [])[:num_results]

    officer_names = []
    snippets = []
    urls = []

    for item in results:
        title = item.get("title", "").strip()
        snippet = item.get("snippet", "").strip()
        link = item.get("link", "").strip()

        formatted_snippet = f"{title}: {snippet}" if title and snippet else title or snippet or "No snippet available"
        name = title.split(" - ")[0].split(":")[0].strip() if title else "Unknown"

        officer_names.append(name)
        snippets.append(formatted_snippet)
        urls.append(link)

    return {
        "officer_names": officer_names,
        "snippets": snippets,
        "urls": urls
    }

def get_web_snippets(query: str, num_results=3) -> list:
    """
    Return a list of formatted "Title: snippet" results for a given query.
    """
    try:
        results = search_web(query, num_results=num_results)
        return results.get("snippets", [])
    except Exception as e:
        print(f"🌐 Web search failed for query: {query}\nError: {e}")
        return []
