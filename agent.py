from typing import TypedDict, Literal

from langgraph.graph import StateGraph, END

from nodes import agent, dispatcher, dispatcher_should_continue, agent_should_continue, music_agent, customer_support_agent, rep_should_continue, customer_tool_node, music_tool_node, other
from state import AgentState

from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()

# Define the config
class GraphConfig(TypedDict):
    model_name: Literal["openai"]

# Define a new graph
workflow = StateGraph(AgentState, config_schema=GraphConfig)

# Define the nodes we will cycle between
workflow.add_node("agent", agent)
workflow.add_node("other", other)

workflow.add_node("dispatcher", dispatcher)
workflow.add_node("music", music_agent)
workflow.add_node("customer", customer_support_agent)
workflow.add_node("customer_tool", customer_tool_node)
workflow.add_node("music_tool", music_tool_node)

# Set the entrypoint as `agent`
workflow.set_entry_point("agent")

# Add conditional edges
workflow.add_conditional_edges(
    "agent", 
    agent_should_continue, {
        "continue": "dispatcher",
        "end": END,
    },
)

workflow.add_conditional_edges(
    "dispatcher", 
    dispatcher_should_continue, {
        "customer": "customer",
        "other": "other",
        "music": "music",
        "end": END,
    },
)


workflow.add_conditional_edges(
    "customer", 
    rep_should_continue, {
        "continue": "customer_tool",
        "end": "dispatcher",
    },
)

workflow.add_conditional_edges(
    "music", 
    rep_should_continue, {
        "continue": "music_tool",
        "end": "dispatcher",
    },
)

workflow.add_edge("customer_tool", "customer")
workflow.add_edge("music_tool", "music")

workflow.add_edge("other", END)

# Compile the graph
graph = workflow.compile(checkpointer=memory)
