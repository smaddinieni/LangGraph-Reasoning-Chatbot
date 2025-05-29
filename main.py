import chainlit as cl
from chatbot import LangGraphChatbot

chatbot_instance = None

@cl.on_chat_start
async def start():
    """Initialize the chatbot when a new chat session starts."""
    global chatbot_instance
    chatbot_instance = LangGraphChatbot()
    
    # Properly initialize the async components
    await chatbot_instance.initialize()
    
    await cl.Message(
        content="Hello! I'm your LangGraph Reasoning Chatbot. How can I help you today?"
    ).send()

@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages and stream responses."""
    if not chatbot_instance:
        await cl.Message(content="Chatbot not initialized. Please refresh the page.").send()
        return
    
    # Generate a simple thread ID (in production, use proper session management)
    thread_id = cl.user_session.get("thread_id", "default_thread")
    cl.user_session.set("thread_id", thread_id)
    
    response_content = ""
    msg = cl.Message(content="")
    
    async for chunk in chatbot_instance.get_response_stream(message.content, thread_id):
        response_content += chunk
        await msg.stream_token(chunk)
    
    await msg.send()