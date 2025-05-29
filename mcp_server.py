import os
import logging
from typing import List, Dict, Any, Optional
from fastmcp import FastMCP
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import AzureError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("Math and Document Search")

# Azure AI Search configuration
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")

# Initialize Azure Search Client
search_client: Optional[SearchClient] = None

def initialize_search_client() -> Optional[SearchClient]:
    """
    Initialize Azure Search Client with proper error handling.
    
    Returns:
        SearchClient: Configured search client or None if configuration is missing
    """
    global search_client
    
    if not all([AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_API_KEY, AZURE_SEARCH_INDEX_NAME]):
        logger.warning("Azure Search configuration incomplete. Document search will be unavailable.")
        return None
    
    try:
        search_client = SearchClient(
            endpoint=AZURE_SEARCH_ENDPOINT,
            index_name=AZURE_SEARCH_INDEX_NAME,
            credential=AzureKeyCredential(AZURE_SEARCH_API_KEY)
        )
        logger.info("Azure Search Client initialized successfully")
        return search_client
    except Exception as e:
        logger.error(f"Failed to initialize Azure Search Client: {str(e)}")
        return None

# Initialize the search client
initialize_search_client()


@mcp.tool()
def search_documents(
    query: str, 
    top_k: int = 5, 
    include_highlights: bool = True
) -> Dict[str, Any]:
    """
    Search through Azure AI Search index for relevant documents.
    
    Args:
        query: The search query string
        top_k: Maximum number of results to return (default: 5)
        include_highlights: Whether to include search highlights (default: True)
        
    Returns:
        Dictionary containing search results, metadata, and any errors
    """
    if not search_client:
        return {
            "error": "Azure Search not configured. Please check environment variables.",
            "results": [],
            "total_count": 0
        }
    
    if not query or not query.strip():
        return {
            "error": "Query cannot be empty",
            "results": [],
            "total_count": 0
        }
    
    try:
        # Sanitize input parameters
        top_k = max(1, min(top_k, 50))  # Limit between 1 and 50
        query = query.strip()[:500]  # Limit query length
        
        # Configure search parameters
        search_params = {
            "search_text": query,
            "top": top_k,
            "include_total_count": True,
        }
        
        if include_highlights:
            search_params.update({
                "highlight_fields": "content,title",
                "highlight_pre_tag": "<mark>",
                "highlight_post_tag": "</mark>"
            })
        
        # Execute search
        logger.info(f"Searching for: '{query}' (top {top_k} results)")
        results = search_client.search(**search_params)
        
        # Process results
        processed_results = []
        total_count = 0
        
        for result in results:
            # Extract highlights if available
            highlights = []
            if hasattr(result, '@search.highlights') and result.get('@search.highlights'):
                for field, highlight_list in result['@search.highlights'].items():
                    highlights.extend(highlight_list)
            
            processed_result = {
                "id": result.get("id", ""),
                "title": result.get("title", "Untitled"),
                "content": result.get("content", "")[:1000],  # Limit content length
                "score": result.get("@search.score", 0.0),
                "highlights": highlights if include_highlights else []
            }
            
            # Include any additional metadata fields
            for key, value in result.items():
                if key not in ["id", "title", "content", "@search.score", "@search.highlights"] and not key.startswith("@"):
                    processed_result[key] = value
            
            processed_results.append(processed_result)
        
        # Get total count if available
        if hasattr(results, 'get_count'):
            try:
                total_count = results.get_count()
            except Exception:
                total_count = len(processed_results)
        else:
            total_count = len(processed_results)
        
        logger.info(f"Found {len(processed_results)} results for query: '{query}'")
        
        return {
            "results": processed_results,
            "total_count": total_count,
            "query": query,
            "returned_count": len(processed_results)
        }
        
    except AzureError as e:
        logger.error(f"Azure Search error: {str(e)}")
        return {
            "error": f"Search service error: {str(e)}",
            "results": [],
            "total_count": 0
        }
    except Exception as e:
        logger.error(f"Unexpected error during search: {str(e)}")
        return {
            "error": f"Unexpected error: {str(e)}",
            "results": [],
            "total_count": 0
        }

@mcp.tool()
def get_document_by_id(document_id: str) -> Dict[str, Any]:
    """
    Retrieve a specific document by its ID from the Azure AI Search index.
    
    Args:
        document_id: The unique identifier of the document
        
    Returns:
        Dictionary containing the document data or error information
    """
    if not search_client:
        return {
            "error": "Azure Search not configured. Please check environment variables.",
            "document": None
        }
    
    if not document_id or not document_id.strip():
        return {
            "error": "Document ID cannot be empty",
            "document": None
        }
    
    try:
        # Sanitize document ID
        document_id = document_id.strip()
        
        logger.info(f"Retrieving document with ID: '{document_id}'")
        document = search_client.get_document(key=document_id)
        
        return {
            "document": document,
            "document_id": document_id
        }
        
    except AzureError as e:
        logger.error(f"Azure Search error retrieving document {document_id}: {str(e)}")
        return {
            "error": f"Document retrieval error: {str(e)}",
            "document": None
        }
    except Exception as e:
        logger.error(f"Unexpected error retrieving document {document_id}: {str(e)}")
        return {
            "error": f"Unexpected error: {str(e)}",
            "document": None
        }

if __name__ == "__main__":
    mcp.run()