import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from tools import all_tools
from typing import TypedDict, Annotated


model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash", temperature=0, api_key="AIzaSyB__hbFsn9A9wxMrIDsoA-Sr8UOezz5zEg"
)

model_with_tools = model.bind_tools(all_tools)


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def assistant(state: AgentState):
    result = model_with_tools.invoke(state["messages"])
    print(result.tool_calls)
    return {"messages": result}
