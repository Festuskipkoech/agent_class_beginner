from typing import Annotated
from typing_extensions import TypedDict
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langchain_tavily import TavilySearch

class State(TypedDict):
    messages: Annotated[list, add_messages]

@tool
def get_current_datetime() -> str:
    """Returns the current date and time. Only call this when the user explicitly asks for the date or time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def estimate_reading_time(text: str) -> str:
    """Estimate reading time in minutes based on text length."""
    word_count = len(text.split())
    minutes = max(1, round(word_count / 200))
    return f"{minutes} minute(s)"

tavily = TavilySearch(max_results=5, topic="general", time_range="day")

@tool
def search_news(query: str) -> str:
    """Search for recent news articles. Only call this when the user asks about news or current events."""
    dated_query = f"{query} {datetime.now().strftime('%Y-%m-%d')}"
    results = tavily.invoke(dated_query)
    if not results:
        return "No results found!"
    articles = results.get("results", [])
    output = []
    for r in articles:
        output.append(f"Title: {r.get('title', 'N/A')}\nURL: {r['url']}\nSummary: {r.get('content', '')[:300]}")
    return "\n\n".join(output)

tools = [get_current_datetime, estimate_reading_time, search_news]

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
llm_with_tools = llm.bind_tools(tools, tool_choice="auto")

def agent(state: State):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

graph_builder = StateGraph(State)
graph_builder.add_node("agent", agent)
graph_builder.add_node("tools", ToolNode(tools))
graph_builder.add_edge(START, "agent")
graph_builder.add_conditional_edges("agent", tools_condition)
graph_builder.add_edge("tools", "agent")
graph = graph_builder.compile()

system_prompt = SystemMessage(
    content=(
        "You are a helpful news assistant. "
        "Only call a tool when the user explicitly needs it. "
        "For greetings and simple questions, respond directly in plain text without calling any tools."
    )
)

conversation_history = [system_prompt]
print("News Agent is running. Type 'exit' to stop.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Nice chat, bye!")
        break
    conversation_history.append({"role": "user", "content": user_input})
    result = graph.invoke({"messages": conversation_history})
    conversation_history = result["messages"]
    assistant_message = result["messages"][-1]
    print(f"Assistant: {assistant_message.content}\n")