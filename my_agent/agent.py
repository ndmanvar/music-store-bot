from typing import TypedDict, Literal

from langgraph.graph import StateGraph, END

from my_agent.utils.nodes import greeting_agent, should_continue, music_agent, customer_support_agent, should_handoff, customer_should_continue, customer_tool_node, music_tool_node
from my_agent.utils.state import AgentState

from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()

# Define the config
class GraphConfig(TypedDict):
    model_name: Literal["openai"]

# Define a new graph
workflow = StateGraph(AgentState, config_schema=GraphConfig)

# Define the nodes we will cycle between
workflow.add_node("agent", greeting_agent)
workflow.add_node("music", music_agent)
workflow.add_node("customer", customer_support_agent)
workflow.add_node("customer_tool", customer_tool_node)
workflow.add_node("music_tool", music_tool_node)

# Set the entrypoint as `agent`
workflow.set_entry_point("agent")

# Add conditional edges
workflow.add_conditional_edges(
    "agent", 
    should_continue, {
        "customer": "customer",
        "music": "music",
        "end": END,
    },
)

workflow.add_conditional_edges(
    "customer", 
    customer_should_continue, {
        "continue": "customer_tool",
        "end": END,
    },
)

workflow.add_conditional_edges(
    "music", 
    customer_should_continue, {
        "continue": "music_tool",
        "end": END,
    },
)

workflow.add_edge("customer_tool", "customer")
workflow.add_edge("music_tool", "music")

# Compile the graph
graph = workflow.compile(checkpointer=memory)