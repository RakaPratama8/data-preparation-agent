"""
Requirements (add to requirements.txt):

streamlit
pandas
"""

import streamlit as st
import pandas as pd
import time

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

        # --- Agent Simulation with Status Visualization ---
        def run_agent(dataframe, user_query):
            """
            Simulate an AI agent analyzing the DataFrame and responding to the user's query.
            This is a placeholder for real agent logic.
            """
            try:
                with st.status("Processing your request...") as status:
                    status.update(label="Analyzing your query...")
                    time.sleep(1)
                    status.update(label="Querying the DataFrame...")
                    time.sleep(2)
                    status.update(label="Generating final response...")
                    time.sleep(1)
                    # Placeholder response
                    answer = (
                        f"Simulated answer to: '{user_query}'\n\n"
                        "*(This is a placeholder. Integrate your AI agent here!)*"
                    )
                    status.update(label="Analysis complete!", state="complete")
                return answer
            except Exception as e:
                # Return error message to be displayed in chat
                return f"Sorry, an error occurred while processing your request: {e}"

        # Run the agent and get the response
        response = run_agent(st.session_state["dataframe"], user_query)

        # Append agent response to history
        st.session_state["messages"].append({"role": "assistant", "content": response})

        # Rerun to display the updated chat
        st.rerun()
