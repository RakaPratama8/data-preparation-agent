import streamlit as st
import pandas as pd
import time

import dataframe_singleton
from agent import AgentState
from graph import app as agent_app
from graph import config
from query import build_msg_query, build_msg_system

if st.session_state.get("messages") is None:
    st.session_state["messages"] = []
if st.session_state.get("dataframe") is None:
    st.session_state["dataframe"] = None

st.set_page_config(page_title="Dataframe Agent Chatbot", page_icon="📊")

with st.sidebar:
    st.markdown(
        "<h2 style='text-align: center; color: #4F8BF9;'>📤 Upload Your CSV File</h2>",
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(
        "Choose a CSV file to get started",
        type=["csv"],
        key="file_uploader",
        help="Only CSV files are supported.",
        label_visibility="collapsed",
    )
    st.markdown(
        "<div style='color: #888; font-size: 0.95em; text-align: center;'>"
        "Your data is processed locally and never leaves your device."
        "</div>",
        unsafe_allow_html=True,
    )

    if uploaded_file is not None:
        try:
            dataframe = pd.read_csv(uploaded_file)
            st.session_state["dataframe"] = dataframe
            dataframe_singleton.dataframe = dataframe
            dataframe_singleton.backup_dataframe = dataframe.copy()
            filename = uploaded_file.name
            st.success(f"'{filename}' loaded successfully!")
            if not st.session_state["messages"]:
                st.session_state["messages"].append(
                    {
                        "role": "assistant",
                        "content": f"Hello! Your file '{filename}' is loaded. How can I help you analyze or modify your data?",
                    }
                )
        except Exception as e:
            st.error(f"Error loading file: {e}")
            st.session_state["dataframe"] = None

if st.session_state["dataframe"] is not None:
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {display: none;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

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
                    prompt = build_msg_query(filename, user_query)
                    system_prompt = build_msg_system()
                    agent_state = AgentState(
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt},
                        ]
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

    st.download_button(
        "Download Modified CSV",
        data=st.session_state["dataframe"].to_csv(index=False, sep=";"),
        file_name="modified_data.csv",
        mime="text/csv",
        help="Download the modified DataFrame as a CSV file.",
        key="download_button",
    )
else:
    st.title("📊 DataFrame Agent Chatbot")
    st.markdown(
        """
        Welcome to the **DataFrame Agent Chatbot**!  
        Effortlessly analyze, explore, and modify your CSV data using natural language.

        ---
        """
    )
    st.info(
        """
        ### 🚀 Features

        - **Instant Data Analysis:**  
          Ask questions about your dataset and receive quick, insightful answers.

        - **Smart Data Modification:**  
          Clean, filter, or transform your data using simple, conversational commands.

        - **Easy Download:**  
          Download your modified dataset as a CSV file at any time.
        """
    )
    st.subheader("📝 How to Get Started")
    st.markdown(
        """
        1. **Upload your CSV file**  
           Use the sidebar on the left to upload your data.

        2. **Interact with the Chatbot**  
           Type your questions or commands in the chatbox below.

        3. **View Results Instantly**  
           See responses and data previews right on this page.

        4. **Download Your Data**  
           After making changes, download the updated CSV file from the sidebar.
        """
    )
    st.warning(
        "ℹ️ No CSV file uploaded yet. Please upload a file to start exploring your data."
    )
