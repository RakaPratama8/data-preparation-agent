"""
Requirements (add to requirements.txt):

streamlit
pandas
"""

import streamlit as st
import pandas as pd
import time

from agent import AgentState
from graph import app as agent_app
from graph import config
from query import msg_query

filename = ""

# --- App Title and Description ---
st.set_page_config(page_title="CSV Analysis Chatbot", page_icon="📊")
st.title("📊 CSV Analysis Chatbot")
st.write(
    """
    Upload a CSV file and chat with an AI assistant to analyze your data!  
    Ask questions like "What is the average sales?" or "Show me summary statistics."
    """
)

# --- File Upload and Session State Management ---
if "dataframe" not in st.session_state:
    st.session_state["dataframe"] = None
if "messages" not in st.session_state:
    st.session_state["messages"] = []

uploaded_file = st.file_uploader(
    "Upload your CSV file", type=["csv"], key="file_uploader"
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.session_state["dataframe"] = df
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
        # Append user message to history
        st.session_state["messages"].append({"role": "user", "content": user_query})

        def run_agent(dataframe, messages):
            """
            Run the real agent using LangGraph and return the assistant's response.
            """
            try:
                with st.status("Processing your request...") as status:
                    status.update(label="Analyzing your query...")
                    time.sleep(1)
                    status.update(label="Querying the DataFrame...")
                    time.sleep(2)
                    status.update(label="Generating final response...")
                    time.sleep(1)
                    # Prepare the agent state
                    messages = msg_query.join([f"\n{messages}"])
                    agent_state = AgentState(messages=messages)
                    # Run the agent (single step)
                    result = agent_app.invoke(
                        agent_state,
                        config=config
                    )
                    # Extract the latest assistant message
                    assistant_msg = (
                        result["messages"][-1].content
                        if result["messages"]
                        else "(No response)"
                    )
                    status.update(label="Analysis complete!", state="complete")
                return assistant_msg
            except Exception as e:
                return f"Sorry, an error occurred while processing your request: {e}"

        # Run the agent and get the response
        response = run_agent(
            st.session_state["dataframe"], st.session_state["messages"]
        )

        # Append agent response to history
        st.session_state["messages"].append({"role": "assistant", "content": response})

        # Rerun to display the updated chat
        st.rerun()
