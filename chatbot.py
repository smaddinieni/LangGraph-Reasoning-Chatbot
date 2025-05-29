import os
from typing import Annotated, List, AsyncGenerator, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import AzureChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "gpt41")
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION", "2024-02-01")

class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

class LangGraphChatbot:
    def __init__(self):
        if not all([AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_CHAT_DEPLOYMENT_NAME, OPENAI_API_VERSION]):
            raise ValueError(
                "Azure OpenAI environment variables not set. "
                "Please ensure AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, "
                "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME, and OPENAI_API_VERSION are in your .env file."
            )

        self.model = AzureChatOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            azure_deployment=AZURE_OPENAI_CHAT_DEPLOYMENT_NAME,
            api_version=OPENAI_API_VERSION,
            max_tokens=2048,
            temperature=0.1,
            streaming=True
        )
        self.memory = MemorySaver()

        try:
            self.mcp_client = MultiServerMCPClient({
                "math": {
                    "command": "python",
                    # Replace with absolute path to your math_server.py file
                    "args": ["../mcp_server.py"],
                    "transport": "stdio",
                },
            })
            self.tools = self.mcp_client.get_tools()
            if self.tools:
                 print(f"Successfully loaded tools from MCP server: {[tool.name for tool in self.tools]}")
            else:
                print("Warning: No tools loaded from MCP server, but client initialized.")
                self.tools = [] 
        except Exception as e:
            print(f"Failed to initialize MCP client or get tools: {e}")
            self.tools = []

        if self.tools:
            self.graph = create_react_agent(self.model, self.tools, checkpointer=self.memory)
            print("Compiled graph with tools using create_react_agent.")
        else:
            graph_builder = StateGraph(State)
            graph_builder.add_node("chatbot", self._direct_llm_node)
            graph_builder.add_edge(START, "chatbot")
            graph_builder.add_edge("chatbot", END)
            self.graph = graph_builder.compile(checkpointer=self.memory)
            print("Warning: MCP tools not loaded. Compiled graph without tools (fallback to direct LLM).")

    def _direct_llm_node(self, state: State) -> dict:
        response_message = self.model.invoke(state["messages"])
        return {"messages": [response_message]}

    async def get_response_stream(self, user_input: str, thread_id: str) -> AsyncGenerator[str, None]:
        config = {"configurable": {"thread_id": thread_id}}
        graph_input = {"messages": [HumanMessage(content=user_input)]}

        async for event in self.graph.astream_events(graph_input, config=config, version="v2"):
            kind = event["event"]
            if kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if isinstance(chunk, AIMessage) and chunk.content:
                    yield chunk.content
            elif kind == "on_tool_end":
                 pass