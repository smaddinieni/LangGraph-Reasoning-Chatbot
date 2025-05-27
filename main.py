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
    Handles uploaded files as context.
    """
    chatbot_instance = cl.user_session.get("chatbot")  # type: LangGraphChatbot
    thread_id = cl.user_session.get("thread_id")

    if not chatbot_instance or not thread_id:
        await cl.ErrorMessage(
            content="Chat session not properly initialized. Please try refreshing the page."
        ).send()
        return

    # --- File Handling Logic ---
    uploaded_files = [element for element in message.elements if isinstance(element, cl.File)]
    file_context_for_llm = []
    processed_files_info = [] # To inform the user which files were processed

    if uploaded_files:
        await cl.Message(content=f"Received {len(uploaded_files)} file(s). Processing...").send()
        for file_element in uploaded_files:
            if "text" in file_element.mime or \
               any(file_element.name.endswith(ext) for ext in [".txt", ".md", ".py", ".csv", ".json"]):
                try:
                    # Read the file content from its temporary path
                    with open(file_element.path, "r", encoding="utf-8") as f:
                        text_content = f.read()

                    # Prepare context for the LLM
                    file_context_for_llm.append(
                        f"\n\n--- Content from uploaded file: {file_element.name} ---\n"
                        f"{text_content}"
                        f"\n--- End of content from: {file_element.name} ---\n"
                    )
                    processed_files_info.append(f"Successfully processed and included content from: {file_element.name}")
                except Exception as e:
                    error_msg = f"Error processing file {file_element.name}: {e}"
                    processed_files_info.append(error_msg)
                    print(error_msg) # Log to server
            else:
                processed_files_info.append(
                    f"Received file: {file_element.name} (MIME: {file_element.mime}). "
                    "This demo primarily processes text-based files for context."
                )

        # Inform the user about file processing results
        if processed_files_info:
            await cl.Message(content="\n".join(processed_files_info)).send()
    # --- End of File Handling Logic ---

    # Combine user's typed message with context from files
    final_user_input = message.content
    if file_context_for_llm:
        final_user_input = "".join(file_context_for_llm) + "\n\nUser's question or message: " + message.content

    # Send the combined input to the chatbot
    msg_ui = cl.Message(content="")  # Create an empty message in the UI to stream into
    await msg_ui.send()

    full_response = ""
    try:
        # The get_response_stream method in your chatbot will receive the combined input
        async for token in chatbot_instance.get_response_stream(final_user_input, thread_id):
            await msg_ui.stream_token(token)
            full_response += token
        
        await msg_ui.update()

    except Exception as e:
        error_message_for_ui = f"Sorry, an error occurred while processing your request: {str(e)}"
        await cl.ErrorMessage(content=error_message_for_ui).send()
        print(f"Error during handle_message: {e}")
        msg_ui.content = "An error occurred. Please check the logs."
        await msg_ui.update()