from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.tools import tool
from pydantic import BaseModel, Field

class Router(BaseModel):
    """Call this if you are able to route the user to the appropriate representative."""
    choice: str = Field(description="should be one of: music, customer")

@tool
def get_music_recs():
    """This returns some music recommendations"""
    return "[Here are some music recommendations!]"

@tool
def get_customer_info(customer_id: int):
    """Look up customer info given their ID. ALWAYS make sure you have the customer ID before invoking this."""
    db = SQLDatabase.from_uri("sqlite:///chinook.db")
    return db.run(f"SELECT * FROM customers WHERE CustomerID = {customer_id};")