from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.tools import tool
from pydantic import BaseModel, Field

@tool
class Router(BaseModel):
    """Call this if you are able to route the user to the appropriate representative."""
    choice: str = Field(description="should be one of: music, customer")

@tool
def get_music_recs():
    """This returns some music recommendations"""
    return "[Here are some music recommendations!]"

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
    # return db.run(f"SELECT * FROM customers WHERE CustomerID = {customer_id}")
    # print("\n")
    # print("query is %s" % query % (customer_id, first_name, last_name))
    # print("\n")
    try:
        db = SQLDatabase.from_uri("sqlite:///chinook.db")
        return db.run(query % (customer_id, first_name, last_name))
    except Exception as e:
        return {"error": "Could not find customer with  CustomerId = %s AND FirstName = '%s' AND LastName = '%s'" % (customer_id, first_name, last_name)}


# This tools is given to the agent to update information about a customer
@tool
def update_customer_info(customer_id: int, first_name: str, last_name: str, updates: dict):
    """
    Updates customer information in the database based on Customer ID, first name, and last name. ALWAYS retrieve the customer information by looking it up first before updating.

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
    
    try:
        db = SQLDatabase.from_uri("sqlite:///chinook.db")
        return db.run(query, parameters=parameters)
    except Exception as e:
        return {"error": f"Error updating customer info: {e}"}