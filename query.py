def build_msg_query(filename: str, user_query: str) -> str:
    return f"""
Dataset: 
{filename}

Information:
You can use a tool to retrieve information about the dataset or modify the dataset. 
The dataset itself has been provided by the runtime, so you just have to run a specific tool (based on the query) to get information on how the dataset has been handled. 
Please specify the tool or action required for your query.


User's query:
{user_query}
"""


def build_msg_system() -> str:
    return """You are a helpful assistant for data analysis. 
You can answer questions about the dataset, provide insights, and suggest modifications. 
Use the tools available to interact with the dataset and provide accurate responses to user queries.
"""
