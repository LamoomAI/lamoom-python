from dataclasses import dataclass, field
import typing as t
import os
from bs4 import BeautifulSoup
import requests
from dotenv import load_dotenv

load_dotenv()

@dataclass
class ToolParameter:
    name: str
    type: str
    description: str
    required: bool = True

@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: t.List[ToolParameter]
    execution_function: t.Callable


API_KEY = os.getenv("GOOGLE_API_KEY")
SEARCH_ID = os.getenv("SEARCH_ENGINE_ID")

def scrape_webpage(url: str):
    """
    Scrapes the content of a webpage and returns the text.
    """
    if not url.startswith(("https://", "http://")):
        url = "https://" + url
    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text()
        clean_text = text.splitlines()
        clean_text = [element.strip()
                      for element in clean_text if element.strip()]
        clean_text = '\n'.join(clean_text)
        return clean_text

    else:
        return "Failed to retrieve the website content."


def perform_web_search(query: str):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        'q': query,
        'key': API_KEY,
        'cx': SEARCH_ID,
        'num': 3
    }

    response = requests.get(url, params=params)
    results = response.json()

    search_result = ""
    
    if 'items' in results:
        for result in results['items']:
            search_result += scrape_webpage(result['link']) + '\n'
    
    return search_result

web_call_tool = ToolDefinition(
    name="web_call",
    description="Performs a web search using a search engine to find up-to-date information or details not present in the internal knowledge.",
    parameters=[
        ToolParameter(name="query", type="string", description="The search query to use.", required=True)
    ],
    execution_function=perform_web_search
)

AVAILABLE_TOOLS = [web_call_tool]