import chainlit as cl
from chatbot import LangGraphChatbot # Import your chatbot class
import os # For getting thread_id for instance

# Optional: Configure Chainlit further if needed, e.g., project name
# from chainlit.server import app
# app.title = "My LangGraph Chatbot"

@cl.on_chat_start
async def start_chat():
    """
    Initializes the chatbot when a new chat session starts.
    """
    try:

        chatbot_instance = LangGraphChatbot()
        cl.user_session.set("chatbot", chatbot_instance)

        thread_id = cl.user_session.get("id")
        if not thread_id:
            # Fallback, though Chainlit should always provide a session ID.
            import uuid
            thread_id = str(uuid.uuid4())
            cl.user_session.set("id", thread_id) # Store it back if generated
        
        cl.user_session.set("thread_id", thread_id)

        await cl.Message(
            # content=f"Hello! I'm your LangGraph assistant. How can I help you today?\n(Session ID: {thread_id})"
            content=f"Hello! I'm your LangGraph MCP assistant. How can I help you today?"
        ).send()
    except ValueError as e:
        # Handle case where environment variables might be missing
        await cl.ErrorMessage(
            content=f"Failed to initialize chatbot: {str(e)}\nPlease check server logs and .env configuration."
        ).send()
        print(f"Error during on_chat_start: {e}") # Log to server console
    except Exception as e:
        await cl.ErrorMessage(
            content=f"An unexpected error occurred during startup: {str(e)}"
        ).send()
        print(f"Unexpected error during on_chat_start: {e}") # Log to server console


@cl.on_message
async def handle_message(message: cl.Message):
    """
    Processes incoming user messages and streams the chatbot's response.
    """
    chatbot_instance = cl.user_session.get("chatbot")  # type: LangGraphChatbot
    thread_id = cl.user_session.get("thread_id")

    if not chatbot_instance or not thread_id:
        await cl.ErrorMessage(
            content="Chat session not properly initialized. Please try refreshing the page."
        ).send()
        return

    user_input = message.content
    msg_ui = cl.Message(content="")  # Create an empty message in the UI to stream into
    await msg_ui.send()

    full_response = ""
    try:
        async for token in chatbot_instance.get_response_stream(user_input, thread_id):
            await msg_ui.stream_token(token)
            full_response += token
        
        # Once streaming is done, update the message if you need to store the full response
        # or if stream_token doesn't finalize it as desired.
        # msg_ui.content = full_response # Usually not needed as stream_token updates UI
        await msg_ui.update()

    except Exception as e:
        error_message = f"Sorry, an error occurred while processing your request: {str(e)}"
        await cl.ErrorMessage(content=error_message).send()
        print(f"Error during handle_message: {e}") # Log to server console
        # Optionally, update the message UI with an error state
        msg_ui.content = "An error occurred. Please check the logs."
        await msg_ui.update()