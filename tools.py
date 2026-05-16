import os
import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchResults
from dotenv import load_dotenv

load_dotenv()

ddg_search_tool = DuckDuckGoSearchResults()

@tool
def web_search(query: str) -> str:
    """Search the web for recent and reliable information on a topic."""
    try:
        return ddg_search_tool.run(query)
    except Exception as e:
        return f"Search failed: {str(e)}"

@tool
def scrape_url(url: str) -> str:
    """Scrape and return clean text content from a given URL."""
    try:
        resp = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        return text[:3000] 
    except Exception as e:
        return f"Could not scrape URL: {str(e)}"