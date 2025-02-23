import json
from functools import lru_cache
from typing import Any, Dict
from langchain_core.messages import AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.types import Command
from my_agent.utils.tools import Router, get_customer_info, update_customer_info, get_albums_by_artist, check_for_songs, get_tracks_by_artist, get_invoices_by_customer, get_purchased_albums_by_customer

class State(AgentState):
    # updated by greeting_agent
    steps: list[str]
    index: int = 0
    next: str = ""

def dispatcher(state: State):
    if state.get("steps") and len(state["steps"]) > state["index"]:
        state["next"] = state["steps"][state["index"]]
        state["index"] += 1
    else:
        state["index"] = 0
        state["next"] = "end"
    return state

def dispatcher_should_continue(state: State):
    return state["next"]

# Define the function that determines whether to continue or not
def should_continue(state: State):
    messages = state["messages"]
    last_message = messages[-1]

    if state.get("steps") and len(state["steps"]) > 0:
        return "continue"
    else:
        return "end"

def customer_should_continue(state: State):
    messages = state["messages"]
    last_message = messages[-1]
    # If there are no tool calls, then we finish
    if not last_message.tool_calls:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"

# Define the initial greeting agent
def agent(state: State, config):
    system_message = """Your job is to help as a customer service representative for a music store.

        You should interact politely with customers to try to figure out how you can help. You can help in a few ways:

        - Looking up or Updating user/customer information: if a customer wants to lookup or update the information in the user database. Call the router with ["customer"]
        - Recommending music: if a customer wants to find some music or information about music. Call the router with ["music"]
        - Music recommendations based on past purchases: if a customer wants music recommendations based on past purchases, call the router with ["customer, "music"].

        Do not call the router multiple times.
        
        Your only job is to route the user to the appropriate representative(s). You can ask the user for more information if you are unable to help them.
        """

    messages = state["messages"]
    messages = [{"role": "system", "content": system_message}] + messages
    model = ChatOpenAI(temperature=0, model_name="gpt-4o")
    model = model.bind_tools([Router])
    response = model.invoke(messages)
    state["messages"].append(response)

    if response.additional_kwargs and 'tool_calls' in response.additional_kwargs:
        tool_calls = response.additional_kwargs['tool_calls']
        for tool_call in tool_calls:
            tool_call_id = tool_call['id']
            arguments = json.loads(tool_call['function']['arguments'])
            choices = arguments['choices']
            tool_message = ToolMessage(content=f"Routing to {choices}", tool_call_id=tool_call_id)
            state["messages"].append(tool_message)
            # state["next"] = choices
            state["steps"] = choices
            state["index"] = 0
    else:
        if state.get("steps"):
            state["index"] = 0
    return state

# Define music agent
def music_agent(state: State, config):
    system_message = """
        Your job is to provide music recommendations based on supplied artists, albums, or past purchases. You can also check if a song exists in the database.
        You don't have the ability to help with anything else besides that.

        You only have certain tools you can use for recommending music. These tools require specific input. You must use these tools when recommending music.
        If a customer asks you to look something up that you don't know how, politely tell them what you can help with.

        When looking up artists and songs, sometimes the artist/song will not be found. In that case, the tools will return information \
        on simliar songs and artists. This is intentional, it is not the tool messing up.
    """
    messages = state["messages"]
    messages = [{"role": "system", "content": system_message}] + messages
    model = ChatOpenAI(temperature=0, model_name="gpt-4o")
    model = model.bind_tools([get_albums_by_artist, get_tracks_by_artist, check_for_songs])
    response = model.invoke(messages)
    response = model.invoke(messages)
    state["messages"].append(response)
    return state

# Define customer support agent
def customer_support_agent(state: State, config):
    system_message = """
        Your only job is to help a user update their profile, look up their information, and retrieve the names of purchased albums.
        You don't have the ability to help with anything else besides that.

        You only have certain tools you can use for helping the user. These tools require specific input. You must use these tools when helping the user.
        If the user is asking for music reommendations based on past purchases, then find the purchased albums and let the user know you cannot help with music recommendations.

        If you don't know the required input, then ask the user for it.
        If you are unable to help the user, you can ask the user for more information.
        """
    messages = state["messages"]
    messages = [{"role": "system", "content": system_message}] + messages
    model = ChatOpenAI(temperature=0, model_name="gpt-4o")
    model = model.bind_tools([get_customer_info, update_customer_info, get_invoices_by_customer, get_purchased_albums_by_customer])
    response = model.invoke(messages)
    state["messages"].append(response)
    return state

# Define the function to execute tools
customer_tool_node = ToolNode([get_customer_info, update_customer_info, get_invoices_by_customer, get_purchased_albums_by_customer])
music_tool_node = ToolNode([get_albums_by_artist, get_tracks_by_artist, check_for_songs])
