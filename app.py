import streamlit as st
import pandas as pd
import time
import os

from langchain.tools import tool
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated
from dotenv import load_dotenv

from query import build_msg_query

load_dotenv()

if st.session_state.get("messages") is None:
    st.session_state["messages"] = []  # Initialize chat history
if st.session_state.get("dataframe") is None:
    st.session_state["dataframe"] = None  # Initialize DataFrame state


# --- Tools ---
@tool
def check_for_null_values(df_name: str) -> str:
    """
    Calculates the number of null (NaN) values for each column in the dataset.
    Use this tool to check for missing data.

    Args:
        df_name (str): The name of the DataFrame to check for null values.
    """
    try:
        df = st.session_state.get("dataframe")
        if df is None:
            return "No DataFrame is loaded."
        null_counts = df.isna().sum()
        nulls_found = null_counts[null_counts > 0]
        if not nulls_found.empty:
            response = f"Found null values in '{df_name}':\n"
            response += nulls_found.to_string()
        else:
            response = f"✅ Great news! No null values were found in '{df_name}'."
        return response
    except Exception as e:
        return f"An error occurred while checking for null values: {e}"


@tool
def stats_of_dataset(df_name: str) -> str:
    """
    Retrieve an information about statistics value of dataset.
    use this tool if you want to analyze or enriched your information about dataset

    Args:
        df_name (str): The name of the DataFrame to retrieve statistics for.
    """
    try:
        df = st.session_state.get("dataframe")
        if df is None:
            return "No DataFrame is loaded."
        describe_dict = df.describe().to_dict()
        if describe_dict:
            response = f"Here's the information in '{df_name}'\n\n{describe_dict}"
        else:
            response = "No information about statistics of your dataset."
        return response
    except Exception as e:
        return f"An error occurred while getting statistics of dataset: {e}"


all_tools = [
    check_for_null_values,
    stats_of_dataset,
]


model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash", temperature=0, api_key=os.environ["GEMINI_API_KEY"]
)

model_with_tools = model.bind_tools(all_tools)


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def assistant(state: AgentState):
    return {"messages": model_with_tools.invoke(state["messages"])}


config = {"configurable": {"thread_id": os.environ["THREAD_ID"]}}

builder = StateGraph(AgentState)

builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(all_tools))

builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")

memory = MemorySaver()
agent_app = builder.compile(checkpointer=memory)

# --- App Title and Description ---
st.set_page_config(page_title="CSV Analysis Chatbot", page_icon="📊")
st.title("📊 CSV Analysis Chatbot")
st.write(
    """
    Upload a CSV file and chat with an AI assistant to analyze your data!  
    Ask questions like "What is the average sales?" or "Show me summary statistics."
    """
)

# --- File Upload and Session State Management (in sidebar) ---
with st.sidebar:
    st.header("Upload CSV")
    uploaded_file = st.file_uploader(
        "Upload your CSV file", type=["csv"], key="file_uploader"
    )

    if uploaded_file is not None:
        try:
            dataframe = pd.read_csv(uploaded_file)
            st.session_state["dataframe"] = dataframe
            filename = uploaded_file.name
            st.success(f"'{filename}' loaded successfully!")
            # Initialize chat history with a welcome message if empty
            if not st.session_state["messages"]:
                st.session_state["messages"].append(
                    {
                        "role": "assistant",
                        "content": f"Hello! Your file '{filename}' is loaded. How can I help you analyze your data?",
                    }
                )
        except Exception as e:
            st.error(f"Error loading file: {e}")
            st.session_state["dataframe"] = None

# --- Chat Interface ---
if st.session_state["dataframe"] is not None:
    # Display chat history
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input for user
    user_query = st.chat_input("Ask a question about your data...")

    if user_query:
        st.session_state["messages"].append({"role": "user", "content": user_query})

        def run_agent(dataframe, messages, filename, user_query):
            try:
                with st.status("Processing your request...") as status:
                    status.update(label="Analyzing your query...")
                    time.sleep(1)
                    status.update(label="Querying the DataFrame...")
                    time.sleep(2)
                    status.update(label="Generating final response...")
                    time.sleep(1)
                    # Build the prompt
                    prompt = build_msg_query(filename, user_query)
                    agent_state = AgentState(
                        messages=[{"role": "user", "content": prompt}]
                    )
                    result = agent_app.invoke(agent_state, config=config)
                    assistant_msg = (
                        result["messages"][-1].content
                        if result["messages"]
                        else "(No response)"
                    )
                    status.update(label="Analysis complete!", state="complete")
                return assistant_msg
            except Exception as e:
                return f"Sorry, an error occurred while processing your request: {e}"

        response = run_agent(
            st.session_state["dataframe"],
            st.session_state["messages"],
            filename,
            user_query,
        )
        st.session_state["messages"].append({"role": "assistant", "content": response})
        st.rerun()
