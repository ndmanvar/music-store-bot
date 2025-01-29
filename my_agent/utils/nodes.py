import json
from functools import lru_cache
from typing import Any, Dict

from langchain_core.messages import AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.types import Command

from my_agent.utils.tools import base_tools, Router, get_customer_info, get_music_recs


class State(AgentState):
    # updated by the tool
    user_choice: dict[str, Any]

@lru_cache(maxsize=4)
def _get_model(model_name: str):
    if model_name == "openai":
        model = ChatOpenAI(temperature=0, model_name="gpt-4o")
    else:
        raise ValueError(f"Unsupported model type: {model_name}")

    model = model.bind_tools(base_tools)
    return model

def call_model(state: State, config, **kwargs):
    messages = state["messages"]
    system_message = kwargs.get('system_message', None)
    if system_message:
        messages = [{"role": "system", "content": system_message}] + messages
    model_name = config.get('configurable', {}).get("model_name", "openai")
    model = _get_model(model_name)
    response = model.invoke(messages)
    return {"messages": [response]}

def _get_last_ai_message(messages):
    for m in messages[::-1]:
        if isinstance(m, AIMessage):
            return m
    return None

def _get_last_tool_call(messages):
    for m in messages[::-1]:
        if _is_tool_call(m):
            return m
    return None

def _get_last_router_tool_call(messages):
    for m in messages[::-1]:
        if _is_tool_call(m):
            tool_calls = m.additional_kwargs['tool_calls']
            tool_call = tool_calls[0]
            if tool_call['function']['name'] == "Router":
                return m
    return None

def _is_tool_call(msg):
    return hasattr(msg, "additional_kwargs") and 'tool_calls' in msg.additional_kwargs

def _route(messages):
    last_message = messages[-1]
    if isinstance(last_message, AIMessage):
        if not _is_tool_call(last_message):
            return END
        else:
            if last_message.additional_kwargs['tool_calls']:
                tool_calls = last_message.additional_kwargs['tool_calls']
                if len(tool_calls) > 1:
                    raise ValueError
                tool_call = tool_calls[0]
                choice = json.loads(tool_call['function']['arguments'])['choice']
                return choice, tool_call['id']
    else:
        last_m = _get_last_ai_message(messages)
        if last_m is None:
            return "agent"
        if last_m.name == "music":
            return "music"
        elif last_m.name == "customer":
            return "customer"
        else:
            return "agent"   

# Define the function that determines whether to continue or not
def should_continue(state: State):
    messages = state["messages"]
    last_message = messages[-1]

    
    if last_message.tool_calls:
        return "continue"
    else:
        tcall_message = _get_last_router_tool_call(messages)
        if not tcall_message:
            return "end"
        else:
            tool_calls = tcall_message.additional_kwargs['tool_calls']
            if len(tool_calls) > 1:
                raise ValueError
            tool_call = tool_calls[0]
            choice = json.loads(tool_call['function']['arguments'])['choice']
            return choice




    # if state.get("user_choice"):
    #     if state["user_choice"] == "customer":
    #         return "customer"
    # If there are no tool calls, then we finish

    if not last_message.tool_calls:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"
    # return "continue"

def customer_should_continue(state: State):
    messages = state["messages"]
    last_message = messages[-1]
    # If there are no tool calls, then we finish
    if not last_message.tool_calls:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"

# Define the function that determines which agent to hand off to
def should_handoff(state: State):
    user_choice = state["user_choice"]
    # If there are no tool calls, then we finish
    if user_choice == "music":
        return "music"
    # Otherwise if there is, we continue
    elif user_choice == "customer":
        return "customer"

# Define the initial greeting agent
def greeting_agent(state: State, config):
    system_message = """Your job is to help as a customer service representative for a music store.

        You should interact politely with customers to try to figure out how you can help. You can help in a few ways:

        - Updating user information: if a customer wants to update the information in the user database. Call the router with `customer`
        - Recomending music: if a customer wants to find some music or information about music. Call the router with `music`

        If the user is asking or wants to ask about updating or accessing their information, send them to that route.
        If the user is asking or wants to ask about music, send them to that route.
        Otherwise, respond."""
    messages = state["messages"]
    messages = [{"role": "system", "content": system_message}] + messages
    model = ChatOpenAI(temperature=0, model_name="gpt-4o")
    model = model.bind_tools([Router])
    response = model.invoke(messages)
    return {"messages": [response]}




# Define the routing agent
def route_agent(state: State):
    """Routes the request to the correct sub-agent."""
    messages = state["messages"]
    # last_message = messages[-1]

    # if isinstance(last_message, AIMessage):
    #     if _is_tool_call(last_message):
    #         tool_call_id = last_message.tool_calls[0]['id']
    #         print("\n\n\n\nRoute agent called", tool_call_id)
    #     return Command(
    #         update={
    #         "messages": [
    #                 ToolMessage(
    #                     "Successfully set user choice", tool_call_id=tool_call_id
    #                 )
    #             ],
    #         }
    #     )

    if state.get("user_choice"):
        return state["user_choice"]
    
    choice, tool_call_id = _route(messages)
    return Command(
        goto=choice,
        update={
            # update the state keys
            "user_choice": choice,
            # update the message history
            "messages": [
                ToolMessage(
                    "Successfully set user choice", tool_call_id=tool_call_id
                )
            ],
        }
    )

# Define the routing agent
# def route_agent(state, config):
    """Use the router tool to route the request to the correct sub-agent."""
    print("\n\n\n\nRoute agent called")
    messages = state["messages"]
    last_message = messages[-1]
    if isinstance(last_message, AIMessage):
        if _is_tool_call(last_message):
            tool_call_id = last_message.tool_calls[0]['id']
            print("\n\n\n\nRoute agent called", tool_call_id)
        return Command(
            update={
            "messages": [
                    ToolMessage(
                        "Successfully set user choice", tool_call_id=tool_call_id
                    )
                ],
            }
        )


# Define music agent
def music_agent(state: State, config):
    print("\n\n\n\nMusic agent called")
    # A node that runs the tools called in the last AIMessage.
    system_message = """Your name is Muzika. When a user asks for help, first tell the customer your name. Next, help a user find music recommendations. If you are unable to help the user, you can ask the user for more information."""
    messages = state["messages"]
    messages = [{"role": "system", "content": system_message}] + messages
    model = ChatOpenAI(temperature=0, model_name="gpt-4o")
    model = model.bind_tools([get_music_recs])
    response = model.invoke(messages)
    return {"messages": [response]}
    

    

# Define customer support agent
def customer_support_agent(state: State, config):
    system_message = """Your job is to help a user update their profile or look up their information.
        You only have certain tools you can use. These tools require specific input.
        If you don't know the required input, then ask the user for it.
        If you are unable to help the user, you can ask the user for more information.
        """

    print("\n\n\n\nCustomer agent called")
    # response = call_model(state, config, system_message=system_message)
    messages = state["messages"]
    messages = [{"role": "system", "content": system_message}] + messages
    model = ChatOpenAI(temperature=0, model_name="gpt-4o")
    model = model.bind_tools([get_customer_info])
    response = model.invoke(messages)
    return {"messages": [response]}


# Define the function to execute tools
customer_tool_node = ToolNode(base_tools)

router_tool_node = ToolNode([Router])

music_tool_node = ToolNode([get_music_recs])