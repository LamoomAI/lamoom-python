
import os
import requests
from bs4 import BeautifulSoup
from dataclasses import dataclass
import typing as t
from dotenv import load_dotenv
import logging

from lamoom.ai_models.tools.base_tool import ToolDefinition, ToolParameter
from lamoom.settings import LAMOOM_GOOGLE_SEARCH_RESULTS_COUNT

load_dotenv()

logger = logging.getLogger(__name__)

API_KEY = os.getenv("GOOGLE_API_KEY")
SEARCH_ID = os.getenv("SEARCH_ENGINE_ID")

@dataclass
class WebSearchResult:
    url: str
    title: str
    snippet: str
    content: str

class WebCall:
    @staticmethod
    def scrape_webpage(url: str) -> str:
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

    @staticmethod
    def perform_web_search(query: str) -> t.List[WebSearchResult]:
        """
        Performs a web search and returns a list of search results.
        """
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            'q': query,
            'key': API_KEY,
            'cx': SEARCH_ID,
            'num': LAMOOM_GOOGLE_SEARCH_RESULTS_COUNT
        }

        response = requests.get(url, params=params)
        results = response.json()

        search_results = []
        
        if 'items' in results:
            for result in results['items']:
                content = WebCall.scrape_webpage(result['link'])
                search_results.append(WebSearchResult(
                    url=result['link'],
                    title=result['title'],
                    snippet=result['snippet'],
                    content=content
                ))
        logger.debug(f"Retrieved search_results {search_results}")
        return search_results

    @staticmethod
    def format_search_results(results: t.List[WebSearchResult]) -> str:
        """
        Formats search results into a readable string.
        """
        formatted = ""
        for i, result in enumerate(results, 1):
            formatted += f"<result_{i}>\n"
            formatted += f"Title: {result.title}\n"
            formatted += f"URL: {result.url}\n"
            formatted += f"Snippet: {result.snippet}\n"
            formatted += f"Content: {result.content[:500]}...\n"
            formatted += f'</result_{i}>'
        return formatted
    
    @staticmethod
    def execute(query: str) -> str:
        return WebCall.format_search_results(
            WebCall.perform_web_search(query)
        )


def perform_web_search(query: str) -> str:
    """
    Performs a web search and returns formatted results.
    """
    results = WebCall.execute(query)
    return results


WEB_SEARCH_TOOL = ToolDefinition(
    name="web_call",
    description="Performs a web search using a search engine to find up-to-date information or details not present in the internal knowledge. Today is {current_datetime_strftime} {timezone}.",
    parameters=[
        ToolParameter(name="query", type="string", description="The search query to use.", required=True)
    ],
    execution_function=perform_web_search
)
