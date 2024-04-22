#app.py
import chainlit as cl
from workflows.main.main_workflow import MainWorkflow

ARCHIVE_MIN_CALLS = 3

@cl.on_chat_start
async def on_chat_start():
    """
    This code is executed when the chat starts.
    """
    workflow = MainWorkflow()
    cl.user_session.set("workflow", workflow)

@cl.on_message
async def main(message: cl.Message):
    """
    main is a function that processes the user input and returns the response.
    """
    # Set the msg variable that will be sent back to the client and get sellbotix
    msg = cl.Message(content='')
    workflow: MainWorkflow = cl.user_session.get('workflow')

    # Stream the response
    async for chunk in workflow.process_user_request(user_request=message.content):
        await msg.stream_token(chunk)
    await msg.send()
