import streamlit as st
import pandas as pd
import time

import dataframe_singleton
from agent import AgentState
from graph import app as agent_app
from graph import config
from query import build_msg_query

if st.session_state.get("messages") is None:
    st.session_state["messages"] = []  # Initialize chat history
if st.session_state.get("dataframe") is None:
    st.session_state["dataframe"] = None  # Initialize DataFrame state

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
            dataframe_singleton.dataframe = dataframe
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
