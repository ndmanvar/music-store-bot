from langchain_community.utilities.sql_database import SQLDatabase

print("Start of output tablenames ==============")
# Connect to the database
db = SQLDatabase.from_uri("sqlite:///chinook.db")
print(db.run("SELECT name FROM sqlite_master WHERE type='table';"))
print("End of output tablenames ==============")

from typing import TypedDict, Literal

from langgraph.graph import StateGraph, END


import json
from functools import lru_cache
from typing import Any, Dict
from langchain_core.messages import AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt.chat_agent_executor import AgentState

from typing import List
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from langchain_community.vectorstores import SKLearnVectorStore
from langchain_openai import OpenAIEmbeddings

db = SQLDatabase.from_uri("sqlite:///chinook.db")

from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage
from typing import TypedDict, Annotated, Sequence

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


@tool
class Router(BaseModel):
    """Call this if you are able to route the user to the appropriate representative(s)."""
    choices: List[str] = Field(description="A list of choices, each should be one of: 'music', 'customer', 'other'")

# This tool is given to the agent to look up information about a customer
@tool
def get_customer_info(customer_id: int, first_name: str, last_name: str):
    """
    Retrieves customer information from the database based on Customer ID, First Name, and Last Name.
    If Customer ID, First Name, and Last Name is not supplied, then ask the user to provide them.
    Do not retrive customer information if Customer ID, First Name, and Last Name is not supplied.
    
    Args:
        customer_id (int): The unique ID of the customer.
        first_name (str): The first name of the customer.
        last_name (str): The last name of the customer.

    Returns:
        dict: A dictionary containing customer information, or an error message.
    """
    # Ensure all required parameters are provided
    if not customer_id or not first_name or not last_name:
        return {"error": "Customer ID, First Name, and Last Name are required."}

    query = """
        SELECT * FROM customers
        WHERE CustomerId = %s AND FirstName = '%s' AND LastName = '%s'
    """
    try:
        return db.run(query % (customer_id, first_name, last_name))
    except Exception as e:
        return {"error": "Could not find customer with  CustomerId = %s AND FirstName = '%s' AND LastName = '%s'" % (customer_id, first_name, last_name)}

# This tools is given to the agent to update information about a customer
@tool
def update_customer_info(customer_id: int, first_name: str, last_name: str, updates: dict):
    """
    Updates customer information in the database based on Customer ID, first name, and last name. ALWAYS retrieve the customer information by looking it up first before updating.
    ALWAYS make sure you have the updates dictionary before updating the customer information.

    Args:
        customer_id (int): The ID of the customer to update.
        updates (dict): A dictionary of column-value pairs to update. Valid columns are: FirstName, LastName, Company, Address, City, State, Country, PostalCode, Phone, Fax, Email.

    Returns:
        dict: A success or failure message.
    """
    # Validate input
    if not customer_id or not isinstance(updates, dict) or len(updates) == 0:
        return {"error": "Customer ID and at least one update field are required."}

    # Dynamically construct the SET clause for the SQL query
    set_clause = ", ".join(f"{col} = :{col}" for col in updates.keys())

    # Construct the SQL query
    query = f"""
        UPDATE customers
        SET {set_clause}
        WHERE CustomerId = :customer_id AND FirstName = :first_name AND LastName = :last_name
    """

    # Add the customer_id to the parameters
    parameters = {**updates, "customer_id": customer_id, "first_name": first_name, "last_name": last_name}

    # Simulate human approval process
    approval = input(f"Do you approve the following update to customerId {customer_id}? {updates} (yes/no): ")
    if approval.lower() != 'yes':
        return {"error": "Update not approved by human."}
    try:
        return db.run(query, parameters=parameters)
    except Exception as e:
        return {"error": f"Error updating customer info: {e}"}

@tool
def get_invoices_by_customer(customer_id):
    """
    Get invoices for a given a customer.
    ALWAYS retrieve the customer information by looking it up first before looking up invoices.
    """
    return db.run(f"SELECT * FROM invoices WHERE CustomerId = {customer_id};")

@tool
def get_purchased_albums_by_customer(customer_id):
    """
    Get albums purchased by a customer.
    ALWAYS retrieve the customer information by looking it up first before looking up invoices.
    """
    return db.run(f"SELECT * FROM albums WHERE AlbumId IN (SELECT AlbumId FROM invoice_items WHERE InvoiceId IN (SELECT InvoiceId FROM invoices WHERE CustomerId = {customer_id})) LIMIT 10;")

@tool
def get_top_purchased_artists_by_customer(customer_id):
    """
    Get the top 10 most purchased artists by a customer.
    ALWAYS retrieve the customer information by looking it up first before looking up invoices.
    """
    query = """
        SELECT artists.*, COUNT(invoice_items.TrackId) as PurchaseCount
        FROM artists
        JOIN albums ON artists.ArtistId = albums.ArtistId
        JOIN tracks ON albums.AlbumId = tracks.AlbumId
        JOIN invoice_items ON tracks.TrackId = invoice_items.TrackId
        JOIN invoices ON invoice_items.InvoiceId = invoices.InvoiceId
        WHERE invoices.CustomerId = :customer_id
        GROUP BY artists.ArtistId
        ORDER BY PurchaseCount DESC
        LIMIT 10;
    """
    return db.run(query, parameters={"customer_id": customer_id})

artists = db._execute("select * from artists")

artist_retriever = SKLearnVectorStore.from_texts(
        [a['Name'] for a in artists],
        OpenAIEmbeddings(), 
        metadatas=artists
    ).as_retriever()

songs = db._execute("select * from tracks")
song_retriever = SKLearnVectorStore.from_texts(
    [a['Name'] for a in songs],
    OpenAIEmbeddings(), 
    metadatas=songs
).as_retriever()

@tool
def get_albums_by_artist(artist):
    """Get albums by an artist (or similar artists)."""
    docs = artist_retriever.get_relevant_documents(artist)
    artist_ids = ", ".join([str(d.metadata['ArtistId']) for d in docs])
    print("docs is ", docs)
    print("artist_ids is ", artist_ids)
    
    return db.run(f"SELECT Title, Name FROM albums LEFT JOIN artists ON albums.ArtistId = artists.ArtistId WHERE albums.ArtistId in ({artist_ids});", include_columns=True)

@tool
def get_tracks_by_artist(artist):
    """Get songs by an artist (or similar artists)."""
    docs = artist_retriever.get_relevant_documents(artist)
    artist_ids = ", ".join([str(d.metadata['ArtistId']) for d in docs])
    return db.run(f"SELECT tracks.Name as SongName, artists.Name as ArtistName FROM albums LEFT JOIN artists ON albums.ArtistId = artists.ArtistId LEFT JOIN tracks ON tracks.AlbumId = albums.AlbumId WHERE albums.ArtistId in ({artist_ids});", include_columns=True)

@tool
def check_for_songs(song_title):
    """Check if a song exists by its name."""
    return song_retriever.get_relevant_documents(song_title)

from langchain.chains import OpenAIModerationChain

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

def other(state: State, config):
    system_message = """
        Always respond with "I'm sorry, I'm not able to help with that. Please ask me something else."
        """
    messages = state["messages"]
    messages = [{"role": "system", "content": system_message}] + messages
    model = ChatOpenAI(temperature=0, model_name="gpt-4o")
    response = model.invoke(messages)
    state["messages"].append(response)
    return state
    

# Define the initial greeting agent
def agent(state: State, config):
    system_message = """Your job is to help as a customer service representative for a music store.

        You should interact politely with customers to try to figure out how you can help. You can help in a few ways:

        - Looking up or Updating user/customer information: if a customer wants to lookup or update the information in the user database. Call the router with ["customer"]
        - Recommending music: if a customer wants to find some music or information about music. Call the router with ["music"]
        - Music recommendations based on past purchases: if a customer wants music recommendations based on past purchases, call the router with ["customer, "music"].
        - for all other inquiries, call the router with ["other"].

        Do not call the router multiple times.
        
        Your only job is to route the user to the appropriate representative(s). You can ask the user for more information if you are unable to help them.
        """

    messages = state["messages"]

    # Moderate the input
    last_message = messages[-1]
    if last_message.content:
        moderated_content = OpenAIModerationChain().invoke(last_message.content)
        last_message.content = moderated_content['output']

    messages = [{"role": "system", "content": system_message}] + messages
    model = ChatOpenAI(temperature=0, model_name="gpt-4o")
    model = model.bind_tools([Router])
    response = model.invoke(messages)

    # Moderate the output
    moderated_response = OpenAIModerationChain().invoke(response.content)
    response.content = moderated_response['output']

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

    # Moderate the output
    moderated_response = OpenAIModerationChain().invoke(response.content)
    response.content = moderated_response['output']

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

    # Moderate the output
    moderated_response = OpenAIModerationChain().invoke(response.content)
    response.content = moderated_response['output']
    
    state["messages"].append(response)
    return state

# Define the function to execute tools
customer_tool_node = ToolNode([get_customer_info, update_customer_info, get_invoices_by_customer, get_purchased_albums_by_customer])
music_tool_node = ToolNode([get_albums_by_artist, get_tracks_by_artist, check_for_songs])

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
    should_continue, {
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
    customer_should_continue, {
        "continue": "customer_tool",
        "end": "dispatcher",
    },
)

workflow.add_conditional_edges(
    "music", 
    customer_should_continue, {
        "continue": "music_tool",
        "end": "dispatcher",
    },
)

workflow.add_edge("customer_tool", "customer")
workflow.add_edge("music_tool", "music")

workflow.add_edge("other", END)

# Compile the graph
graph = workflow.compile(checkpointer=memory)
