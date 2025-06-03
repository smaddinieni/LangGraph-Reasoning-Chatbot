import os
import logging
from dotenv import load_dotenv
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langchain_openai import AzureChatOpenAI

# Configure logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "gpt-4.1-mini")
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION", "2024-02-01")

client = MultiServerMCPClient(
    {
        "DXC_Document_Search": {
            "command": "python",
            "args": ["mcp_server01.py"],
            "transport": "stdio",
        }
    }
)

async def main() -> None:
    """
    Main entry point for the MCP client.
    Loads tools, initializes the chat model, and sends a test message.
    """
    try:
        logger.info("Fetching tools from MCP server...")
        tools = await client.get_tools()
        logger.info(f"Loaded tools: {tools}")

        model = AzureChatOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            azure_deployment=AZURE_OPENAI_CHAT_DEPLOYMENT_NAME,
            api_version=OPENAI_API_VERSION,  # Fixed bug here
            api_key=AZURE_OPENAI_API_KEY,
            temperature=0.1,
            max_tokens=5000,
        )
        logger.info("Creating ReAct agent...")
        mcp_agent = create_react_agent(model, tools)
        logger.info("Sending test message to agent...")
        math_response = await mcp_agent.ainvoke({"messages": "Tell me all info about sabarnath Maddinieni?"})
        print("Agent response:", math_response["messages"][-1].content)
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())