from tavily import TavilyClient
import config


def web_search(query: str) -> str:
    """Search the web and return a summarized answer with sources."""
    if not config.TAVILY_API_KEY:
        return "Web search is unavailable: TAVILY_API_KEY is not set."

    client = TavilyClient(api_key=config.TAVILY_API_KEY)
    result = client.search(query=query, max_results=5, include_answer=True)

    parts = []
    if result.get("answer"):
        parts.append(result["answer"])
    for item in result.get("results", []):
        parts.append(f"- {item['title']}: {item['url']}")
    return "\n".join(parts) if parts else "No results found."
