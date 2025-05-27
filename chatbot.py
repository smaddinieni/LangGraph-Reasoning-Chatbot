import os
from typing import Annotated, List, AsyncGenerator
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import AzureChatOpenAI

# Load environment variables
# These should be in your .env file
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "gpt41") # Default to "gpt41" if not set
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION", "2024-02-01") # Use a recent, valid API version

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
            streaming=True  # Crucial for Chainlit streaming
        )
        self.memory = MemorySaver()
        
        graph_builder = StateGraph(State)
        graph_builder.add_node("chatbot", self._chatbot_node)
        graph_builder.add_edge(START, "chatbot")

        self.graph = graph_builder.compile(checkpointer=self.memory)

    def _chatbot_node(self, state: State) -> dict:
        """
        The core chatbot node. It takes the current conversation state,
        invokes the model, and returns the AI's response.
        """
        response_message = self.model.invoke(state["messages"])
        return {"messages": [response_message]}

    async def get_response_stream(self, user_input: str, thread_id: str) -> AsyncGenerator[str, None]:
        """
        Processes user input and streams the chatbot's response.
        """
        config = {"configurable": {"thread_id": thread_id}}
        graph_input = {"messages": [HumanMessage(content=user_input)]}

        async for event in self.graph.astream_events(
            graph_input,
            config=config,
            version="v2"  # Use "v1" or "v2" depending on LangGraph version features. "v2" is more structured.
        ):
            kind = event["event"]
            if kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if isinstance(chunk, AIMessage) and chunk.content:
                    yield chunk.content
