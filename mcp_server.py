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

mcp = FastMCP("DXC Document Search")

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
    include_highlights: bool = False,
    select_fields: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Search through Azure AI Search index for relevant documents with specific field selection.
    
    Args:
        query: The search query string
        top_k: Maximum number of results to return (default: 5)
        include_highlights: Whether to include search highlights (default: False)
        select_fields: Specific fields to return in results (default: predefined business fields)
        
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
    
    # Define specific business-relevant fields to retrieve
    if select_fields is None:
        select_fields = [
            'projectdate',
            'solutioncomponent', 
            'fyqwon',
            'businessfunction',
            'contenttype',
            'levelofreference',
            'projecttype',
            'usefullink',
            'metadata_storage_path',
            'id',
            'industry',
            'descriptionusefullink',
            'partner',
            'text',
            'metadata_storage_name',
            'title',
            'comments',
            'description',
            'region',
            'chunk_id',
            'oppid',
            'enablingtools',
            'chunk',
            'customername',
            'luxoftcontact',
            'projectname'
        ]
    
    try:
        # Sanitize input parameters
        top_k = max(1, min(top_k, 50))  # Limit between 1 and 50
        query = query.strip()[:500]  # Limit query length to prevent injection
        
        # Configure search parameters with field selection
        search_params = {
            "search_text": query,
            "top": top_k,
            "include_total_count": True,
            "select": select_fields  # Only return specified fields
        }
        
        # Add highlighting only if explicitly requested and for known text fields
        if include_highlights:
            # Define known searchable text fields for highlighting
            highlight_fields = ['title', 'description', 'chunk', 'text']
            search_params.update({
                "highlight_fields": ",".join(highlight_fields),
                "highlight_pre_tag": "<mark>",
                "highlight_post_tag": "</mark>"
            })
            logger.info(f"Search with highlights enabled for fields: {highlight_fields}")
        
        # Execute search with field selection
        logger.info(f"Searching for: '{query}' (top {top_k} results, {len(select_fields)} fields)")
        results = search_client.search(**search_params)
        
        # Process results with efficient field handling
        processed_results = []
        
        for result in results:
            # Extract highlights if available
            highlights = []
            if include_highlights and hasattr(result, '@search.highlights') and result.get('@search.highlights'):
                for field, highlight_list in result['@search.highlights'].items():
                    highlights.extend(highlight_list)
            
            # Build result with only selected fields
            processed_result = {
                "score": result.get("@search.score", 0.0)
            }
            
            # Add highlights only if requested
            if include_highlights:
                processed_result["highlights"] = highlights
            
            # Include only the selected business fields
            for field in select_fields:
                value = result.get(field)
                
                # Handle long text fields efficiently
                if isinstance(value, str) and len(value) > 2000:
                    processed_result[field] = value[:2000] + "..."
                else:
                    processed_result[field] = value
            
            processed_results.append(processed_result)
        
        # Get total count efficiently
        try:
            total_count = results.get_count() if hasattr(results, 'get_count') else len(processed_results)
        except Exception:
            total_count = len(processed_results)
        
        logger.info(f"Retrieved {len(processed_results)} results for query: '{query}' with {len(select_fields)} fields each")
        
        return {
            "results": processed_results,
            "total_count": total_count,
            "query": query,
            "returned_count": len(processed_results),
            "selected_fields": select_fields
        }
        
    except AzureError as azure_error:
        error_msg = f"Azure Search service error: {str(azure_error)}"
        logger.error(error_msg)
        return {
            "error": error_msg,
            "results": [],
            "total_count": 0
        }
    except Exception as unexpected_error:
        error_msg = f"Unexpected search error: {str(unexpected_error)}"
        logger.error(error_msg)
        return {
            "error": error_msg,
            "results": [],
            "total_count": 0
        }


@mcp.tool()
def search_documents_minimal(query: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Simplified search function returning only essential business fields for performance.
    
    Args:
        query: The search query string
        top_k: Maximum number of results to return
        
    Returns:
        Dictionary containing search results with core business fields only
    """
    # Core business fields for lightweight searches
    essential_fields = [
        'id',
        'title', 
        'description',
        'contenttype',
        'industry',
        'solutioncomponent',
        'metadata_storage_name',
        'chunk'
    ]
    
    return search_documents(
        query=query,
        top_k=top_k,
        include_highlights=False,
        select_fields=essential_fields
    )

if __name__ == "__main__":
    mcp.run()