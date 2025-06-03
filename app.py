import chainlit as cl
from chatbot import LangGraphChatbot # Ensure chatbot.py is in the same directory or PYTHONPATH
import logging
import uuid # For more unique thread IDs if user_session.get('id') is not available
from typing import Optional


logging.basicConfig(
    level=logging.ERROR, # Set to ERROR or higher for production
    format="%(asctime)s [%(levelname)s] %(name)s (%(module)s.%(funcName)s): %(message)s"
)
logger = logging.getLogger(__name__)

chatbot_instance: Optional[LangGraphChatbot] = None

@cl.on_chat_start
async def start_chat_session():
    """Initialize the chatbot when a new chat session starts."""
    global chatbot_instance
    try:
        chatbot_instance = LangGraphChatbot()
        await chatbot_instance.initialize()
        await cl.Message(
            content="Hello! I'm your AI Chatbot. How can I help you today?"
        ).send()
    except ValueError as ve:
        logger.error(f"CONFIG ERROR during LangGraphChatbot instantiation in on_chat_start: {ve}", exc_info=True)
        chatbot_instance = None
        await cl.Message(content=f"ERROR: Failed to initialize chatbot due to a configuration issue: {ve}. Please check server logs.").send()
    except RuntimeError as re:
        logger.error(f"RUNTIME ERROR during chatbot_instance.initialize() in on_chat_start: {re}", exc_info=True)
        chatbot_instance = None
        await cl.Message(content=f"ERROR: Chatbot failed to initialize its components: {re}. Please check server logs.").send()
    except Exception as e:
        logger.error(f"UNEXPECTED ERROR during @cl.on_chat_start: {e}", exc_info=True)
        chatbot_instance = None
        await cl.Message(content=f"ERROR: Chatbot failed to start due to an unexpected error: {e}. Please check server logs.").send()


@cl.on_message
async def handle_message(message: cl.Message):
    """Handle incoming messages and stream responses."""
    if chatbot_instance is None:
        logger.error("Chatbot_instance is None in on_message.")
        await cl.Message(content="Chatbot is not available. It may have failed to start. Please refresh or contact support if the issue persists.").send()
        return
    
    if not chatbot_instance._initialized:
        logger.error("Chatbot instance exists but its internal _initialized flag is False.")
        await cl.Message(content="Chatbot is in an uninitialized state. Please try refreshing. If the problem continues, check server logs.").send()
        return

    thread_id = cl.user_session.get("thread_id")
    if not thread_id:
        session_specific_part = cl.user_session.get('id', str(uuid.uuid4()))
        thread_id = f"thread_{session_specific_part}"
        cl.user_session.set("thread_id", thread_id)
    
    response_ui_message = cl.Message(content="")
    streamed_anything = False
    try:
        async for chunk_content in chatbot_instance.get_response_stream(message.content, thread_id):
            if not chunk_content:
                continue
            await response_ui_message.stream_token(chunk_content)
            streamed_anything = True
        
        if streamed_anything:
            await response_ui_message.update()
        else:
            if not response_ui_message.id:
                 await cl.Message(content="Sorry, I was unable to generate a response for your query.").send()
            else:
                 response_ui_message.content = "Sorry, I received an empty response."
                 await response_ui_message.update()
    except Exception as e:
        logger.error(f"Error in @cl.on_message while streaming/handling response: {e}", exc_info=True)
        error_report_msg = f"An error occurred while processing your message: {type(e).__name__}. Please check server logs."
        if response_ui_message.id and streamed_anything:
            response_ui_message.content = error_report_msg
            await response_ui_message.update()
        else:
            await cl.Message(content=error_report_msg).send()