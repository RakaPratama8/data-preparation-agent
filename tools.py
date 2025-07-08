import streamlit as st
import dataframe_singleton

from langchain.tools import tool

@tool
def check_for_null_values(df_name: str) -> str:
    """
    Calculates the number of null (NaN) values for each column in the dataset.
    Use this tool to check for missing data.
    
    Args:
        df_name (str): The name of the DataFrame to check for null values.
    """
    try:
        df = dataframe_singleton.dataframe
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
        df = dataframe_singleton.dataframe
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
