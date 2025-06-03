import os
import logging
from typing import List, Dict, Any, Optional
from fastmcp import FastMCP
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Basic logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("DXC_Document_Search")

# Initialize Azure Search Client
def get_search_client() -> Optional[SearchClient]:
    """Initialize Azure Search Client."""
    endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
    api_key = os.getenv("AZURE_SEARCH_API_KEY")
    index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")
    
    if not all([endpoint, api_key, index_name]):
        logger.error("Missing Azure Search environment variables")
        return None
    
    try:
        return SearchClient(
            endpoint=endpoint,
            index_name=index_name,
            credential=AzureKeyCredential(api_key)
        )
    except Exception as e:
        logger.error(f"Failed to initialize search client: {e}")
        return None

search_client = get_search_client()

@mcp.tool()
def search_documents(query: str) -> Dict[str, Any]:
    """
    Search Azure AI Search index and return top 5 relevant documents.
    
    Args:
        query: Search query string
        
    Returns:
        Dictionary with search results or error message
    """
    if not search_client:
        return {"error": "Azure Search not configured"}
    
    if not query or not query.strip():
        return {"error": "Query cannot be empty"}
    
    try:
        # Core business fields to return
        select_fields = [
            'id', 'title', 'description', 'chunk', 'contenttype',
            'industry', 'solutioncomponent', 'projectname', 'customername'
        ]
        
        results = search_client.search(
            search_text=query.strip(),
            top=10,
            select=select_fields,
            include_total_count=True
        )
        
        # Process results
        documents = []
        for result in results:
            doc = {"score": result.get("@search.score", 0.0)}
            for field in select_fields:
                value = result.get(field)
                # Truncate long text fields
                if isinstance(value, str) and len(value) > 1000:
                    doc[field] = value[:1000] + "..."
                else:
                    doc[field] = value
            documents.append(doc)
        
        return {
            "documents": documents,
            "total_found": results.get_count() if hasattr(results, 'get_count') else len(documents),
            "query": query
        }
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        return {"error": f"Search failed: {str(e)}"}

if __name__ == "__main__":
    logger.info("Starting simplified MCP Server...")
    mcp.run(transport="stdio")