import os
import logging
from typing import List, AsyncGenerator, Optional, Dict, Any
import pyodbc
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import BaseTool
from langchain_openai import AzureChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_mcp_adapters.client import MultiServerMCPClient # Assuming this is your actual working import
conn_str = os.getenv("SQL_CONN_STR")

logging.basicConfig(
    level=logging.ERROR, # Set to ERROR or higher for production
    format="%(asctime)s [%(levelname)s] %(name)s (%(module)s.%(funcName)s): %(message)s"
)
logger = logging.getLogger(__name__)

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "gpt-4o-mini")
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION", "2024-02-01")

def log_to_sql(session_id, user_id, prompt, response, is_error=False, metadata=None):
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO chainlit_logs (session_id, user_id, prompt, response, is_error, metadata)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        session_id, user_id, prompt, response, int(is_error), metadata
    )

    conn.commit()
    cursor.close()
    conn.close()


class LangGraphChatbot:
    """
    LangGraph-based chatbot using a ReAct agent with Azure OpenAI and MCP tool integration.
    This class manages the agent, tool loading, and streaming responses for Chainlit.
    """

    def __init__(self) -> None:
        """
        Initialize the chatbot with Azure OpenAI model and in-memory conversation memory.
        Raises:
            ValueError: If required Azure OpenAI environment variables are not set.
        """
        if not all([AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY]):
            logger.error("Azure OpenAI environment variables (AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY) not set.")
            raise ValueError(
                "Required Azure OpenAI environment variables not set. "
                "Please ensure AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY are configured."
            )

        try:
            self.model: AzureChatOpenAI = AzureChatOpenAI(
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                api_key=AZURE_OPENAI_API_KEY,
                azure_deployment=AZURE_OPENAI_CHAT_DEPLOYMENT_NAME,
                api_version=OPENAI_API_VERSION,
                max_tokens=1024,
                temperature=0.7,
                streaming=True
            )
        except Exception as e:
            logger.error(f"Failed to initialize AzureChatOpenAI model: {e}", exc_info=True)
            raise RuntimeError(f"Could not initialize Azure LLM: {e}")

        self.tools: List[BaseTool] = []
        self.agent_executor = None
        self.checkpointer = InMemorySaver()
        self._initialized = False


    async def initialize(self) -> None:
        """
        Initialize MCP client, load tools, and create the ReAct agent with memory.
        This method can be called multiple times but will only run full initialization once.
        """
        if self._initialized:
            return

        try:
            mcp_client = MultiServerMCPClient({
                "DXC_Document_Search": {
                    "command": "python",
                    "args": ["mcp_server01.py"],
                    "transport": "stdio",
                },
                "SQL_Server": {
                    "command": "python",
                    "args": ["mcp_server02.py"],
                    "transport": "stdio",
                },
            })
            self.tools = await mcp_client.get_tools()
        except Exception as e:
            logger.error(f"Failed to initialize/load MCP tools: {e}", exc_info=True)
            self.tools = []

        if not self.model:
            logger.error("self.model is None before creating ReAct agent. Azure LLM failed to init.")
            self._initialized = False
            raise RuntimeError("AzureChatOpenAI model (self.model) is not initialized properly.")

        try:
            self.agent_executor = create_react_agent(
                model=self.model,
                tools=self.tools,
                checkpointer=self.checkpointer,
            )
        except Exception as e:
            logger.error(f"Failed to create ReAct agent executor: {e}", exc_info=True)
            self._initialized = False
            self.agent_executor = None
            raise

        self._initialized = True


    async def get_response_stream(self, user_input: str, thread_id: str, user_id: Optional[str] = "anonymous") -> AsyncGenerator[str, None]:
        """
        Generate streaming response for user input with conversation memory and log interaction to SQL.
        Args:
            user_input: The user's message.
            thread_id: Unique conversation thread identifier for memory persistence.
            user_id: The user's identifier (default: "anonymous").
        Yields:
            str: Response content chunks as they are generated.
        """
        if not self._initialized:
            try:
                await self.initialize()
            except Exception as e:
                logger.error(f"Error during self.initialize() in get_response_stream: {e}", exc_info=True)
                # Log error to SQL
                log_to_sql(
                    session_id=thread_id,
                    user_id=user_id,
                    prompt=user_input,
                    response=f"Error: Chatbot failed to initialize during request: {e}",
                    is_error=True,
                    metadata=None
                )
                yield f"Error: Chatbot failed to initialize during request: {e}"
                return

        if not self.agent_executor:
            logger.error("self.agent_executor is None after initialization attempt in get_response_stream.")
            # Log error to SQL
            log_to_sql(
                session_id=thread_id,
                user_id=user_id,
                prompt=user_input,
                response="Error: Agent executor not available after initialization.",
                is_error=True,
                metadata=None
            )
            yield "Error: Agent executor not available after initialization."
            return

        config = {"configurable": {"thread_id": thread_id}}
        messages_for_agent: List[BaseMessage] = [HumanMessage(content=user_input)]
        current_ai_response_content = ""
        final_ai_response = ""

        try:
            async for chunk_event in self.agent_executor.astream({"messages": messages_for_agent}, config=config):
                messages_in_chunk: Optional[List[BaseMessage]] = None
                for key, value in chunk_event.items():
                    if isinstance(value, dict) and "messages" in value and isinstance(value["messages"], list):
                        if all(isinstance(m, BaseMessage) for m in value["messages"]):
                            messages_in_chunk = value["messages"]
                            break
                    elif key == "messages" and isinstance(value, list):
                        if all(isinstance(m, BaseMessage) for m in value):
                            messages_in_chunk = value
                            break

                if messages_in_chunk:
                    last_message = messages_in_chunk[-1]
                    if isinstance(last_message, AIMessage) and hasattr(last_message, 'content'):
                        full_content_of_last_ai_message = last_message.content
                        if full_content_of_last_ai_message.startswith(current_ai_response_content):
                            new_token_piece = full_content_of_last_ai_message[len(current_ai_response_content):]
                            if new_token_piece:
                                yield new_token_piece
                                current_ai_response_content = full_content_of_last_ai_message
                                final_ai_response = full_content_of_last_ai_message
                        elif not current_ai_response_content and full_content_of_last_ai_message:
                            yield full_content_of_last_ai_message
                            current_ai_response_content = full_content_of_last_ai_message
                            final_ai_response = full_content_of_last_ai_message
            # Log successful interaction to SQL
            log_to_sql(
                session_id=thread_id,
                user_id=user_id,
                prompt=user_input,
                response=final_ai_response,
                is_error=False,
                metadata=None
            )
        except Exception as e:
            logger.error(f"Error during agent_executor.astream: {e}", exc_info=True)
            # Log error to SQL
            log_to_sql(
                session_id=thread_id,
                user_id=user_id,
                prompt=user_input,
                response=f"Error: Agent streaming failed unexpectedly: {e}",
                is_error=True,
                metadata=None
            )
            yield f"Error: Agent streaming failed unexpectedly: {e}"
            return
