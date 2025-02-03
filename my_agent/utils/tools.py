from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from langchain_community.vectorstores import SKLearnVectorStore
from langchain_openai import OpenAIEmbeddings

db = SQLDatabase.from_uri("sqlite:///chinook.db")

@tool
class Router(BaseModel):
    """Call this if you are able to route the user to the appropriate representative."""
    choice: str = Field(description="should be one of: music, customer")

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
    return db.run(f"SELECT * FROM albums WHERE AlbumId IN (SELECT AlbumId FROM invoice_items WHERE InvoiceId IN (SELECT InvoiceId FROM invoices WHERE CustomerId = {customer_id}));")


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