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


@tool
def get_dataset_info(df_name: str) -> str:
    """
    Retrieve basic information about the dataset.
    Use this tool to understand the structure and content of your dataset.

    Args:
        df_name (str): The name of the DataFrame to retrieve information for.
    """
    try:
        df = dataframe_singleton.dataframe
        if df is None:
            return "No DataFrame is loaded."
        info = df.info()
        return f"Dataset '{df_name}' information:\n{info}"
    except Exception as e:
        return f"An error occurred while retrieving dataset info: {e}"

@tool
def get_a_number_of_rows_and_columns(df_name: str) -> str:
    """
    Retrieve the number of rows and columns in the dataset.
    Use this tool to quickly understand the size of your dataset.

    Args:
        df_name (str): The name of the DataFrame to retrieve size information for.
    """
    try:
        df = dataframe_singleton.dataframe
        if df is None:
            return "No DataFrame is loaded."
        num_rows, num_columns = df.shape
        return f"Dataset '{df_name}' has {num_rows} rows and {num_columns} columns."
    except Exception as e:
        return f"An error occurred while retrieving dataset size: {e}"

@tool
def get_dataset_head(df_name: str, num_rows: int = 5) -> str:
    """
    Retrieve the first few rows of the dataset.
    Use this tool to quickly inspect the data in your dataset.

    Args:
        df_name (str): The name of the DataFrame to retrieve rows from.
        num_rows (int): The number of rows to retrieve (default is 5).
    """
    try:
        df = dataframe_singleton.dataframe
        if df is None:
            return "No DataFrame is loaded."
        head_data = df.head(num_rows)
        return f"First {num_rows} rows of '{df_name}':\n{head_data.to_string()}"
    except Exception as e:
        return f"An error occurred while retrieving dataset head: {e}"


@tool
def get_column_names(df_name: str) -> str:
    """
    Retrieve the names of the columns in the dataset.
    Use this tool to understand the structure of your dataset.

    Args:
        df_name (str): The name of the DataFrame to retrieve column names from.
    """
    try:
        df = dataframe_singleton.dataframe
        if df is None:
            return "No DataFrame is loaded."
        columns = df.columns.tolist()
        return f"Column names in '{df_name}': {', '.join(columns)}"
    except Exception as e:
        return f"An error occurred while retrieving column names: {e}"


# --- Tools for Dataframe Manipulation ---
@tool
def handle_null_values(df_name: str, action: str, strategy: str) -> str:
    """
    Handle null values in the dataset by either dropping or filling them.
    Use this tool to clean your dataset from missing values.

    Args:
        df_name (str): The name of the DataFrame to handle null values for.
        action (str): The action to perform ('drop' or 'fill').
        strategy (str): The strategy to use for filling null values (if applicable).
    """
    try:
        df = dataframe_singleton.dataframe
        if df is None:
            return "No DataFrame is loaded."
        if action == "drop":
            df_dropped = df.dropna()
            dataframe_singleton.dataframe = df_dropped
            return f"Null values dropped from '{df_name}'. Remaining rows: {len(df_dropped)}"
        elif action == "fill":
            if strategy == "mean":
                df_filled = df.fillna(df.mean())
            elif strategy == "median":
                df_filled = df.fillna(df.median())
            elif strategy == "mode":
                df_filled = df.fillna(df.mode().iloc[0])
            else:
                return "Invalid fill strategy. Use 'mean', 'median', or 'mode'."
            dataframe_singleton.dataframe = df_filled
            return f"Null values filled in '{df_name}' using '{strategy}' strategy."
        else:
            return "Invalid action. Use 'drop' or 'fill'."
    except Exception as e:
        return f"An error occurred while handling null values: {e}"


@tool
def filter_dataset(df_name: str, column: str, value: str) -> str:
    """
    Filter the dataset based on a specific column and value.
    Use this tool to narrow down your dataset to specific criteria.

    Args:
        df_name (str): The name of the DataFrame to filter.
        column (str): The column to filter by.
        value (str): The value to filter for in the specified column.
    """
    try:
        df = dataframe_singleton.dataframe
        if df is None:
            return "No DataFrame is loaded."
        filtered_df = df[df[column] == value]
        if not filtered_df.empty:
            dataframe_singleton.dataframe = filtered_df
            return f"Filtered dataset '{df_name}' where {column} equals {value}:\n{filtered_df.to_string()}"
        else:
            return f"No rows found in '{df_name}' where {column} equals {value}."
    except Exception as e:
        return f"An error occurred while filtering the dataset: {e}"

@tool
def change_value_in_column(df_name: str, column: str, old_value: str, new_value: str) -> str:
    """
    Change a specific value in a column of the dataset.
    Use this tool to update values in your dataset.

    Args:
        df_name (str): The name of the DataFrame to modify.
        column (str): The column where the value will be changed.
        old_value (str): The value to be replaced.
        new_value (str): The new value to replace the old value with.
    """
    try:
        df = dataframe_singleton.dataframe
        if df is None:
            return "No DataFrame is loaded."
        df[column] = df[column].replace(old_value, new_value)
        dataframe_singleton.dataframe = df
        return f"Changed '{old_value}' to '{new_value}' in column '{column}' of '{df_name}'."
    except Exception as e:
        return f"An error occurred while changing values in the dataset: {e}"



all_tools = [
    check_for_null_values,
    stats_of_dataset,
    get_dataset_info,
    get_dataset_head,
    get_column_names,
    get_a_number_of_rows_and_columns,
    handle_null_values,
    filter_dataset,
    change_value_in_column
]
