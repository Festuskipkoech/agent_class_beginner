from typing import Annotated
from typing_extensions import TypedDict
from datetime import datetime

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage

# 1. Define the State (Same as before, with memory)
class State(TypedDict):
    messages: Annotated[list, add_messages]

# 2. Define our Tools (@tool decorator makes them visible to LangChain)
@tool
def get_current_datetime() -> str:
    """Returns the current date and time. Use this whenever the user asks for the date or time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def estimate_reading_time(text: str) -> str:
    """Estimate how long an article takes to read in minutes based on text length.
    Input should be the full text of the article.
    """
    word_count = len(text.split())
    # Assuming an average reading speed of 200 words per minute
    minutes = max(1, round(word_count / 200))
    return f"{minutes} minute(s)"

@tool
def categorize_topic(headline: str) -> str:
    """Categorize a news headline into one of four categories: Tech, Politics, Sports, or Business."""
    headline_lower = headline.lower()
    if any(word in headline_lower for word in ["tech", "ai", "model", "software", "openai"]):
        return "Tech"
    elif any(word in headline_lower for word in ["vote", "parliament", "bill", "election", "president"]):
        return "Politics"
    elif any(word in headline_lower for word in ["game", "sports", "match", "team", "score"]):
        return "Sports"
    else:
        return "Business"

# Group our tools into a list
tools = [get_current_datetime, estimate_reading_time, categorize_topic]

# 3. Initialize LLM and Bind Tools
llm = ChatOllama(model="llama3.2", temperature=0)

# Binding tells the LLM: "Here are the tools available to you."
llm_with_tools = llm.bind_tools(tools)

# 4. Define the Agent Node
def agent(state: State):
    # We use the tool-bound LLM here instead of the plain one
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# 5. Build the Graph
graph_builder = StateGraph(State)

# Add our nodes
graph_builder.add_node("agent", agent)
# ToolNode handles actually executing the Python function when the LLM asks for it
tool_node = ToolNode(tools) 
graph_builder.add_node("tools", tool_node)

# Set up the edges (The Flow)
graph_builder.add_edge(START, "agent")

# The Conditional Edge: 
# Did the agent decide to call a tool? 
# If YES -> go to 'tools' node. If NO -> go to END.
graph_builder.add_conditional_edges("agent", tools_condition)

# After a tool finishes running, ALWAYS loop back to the agent so it can read the result and reply
graph_builder.add_edge("tools", "agent")

# Compile the graph
graph = graph_builder.compile()

# 6. The Execution Loop
# We add a System Message nudge because local models like Llama 3.2 
# sometimes need a reminder to use their tools rather than answering from memory.
system_prompt = SystemMessage(
    content="You are a smart assistant. Always check your available tools before answering. If a tool can help answer the prompt, use it."
)
conversation_history = [system_prompt]

print("Tool Agent is running. Type 'exit' to stop.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Nice chat, goodbye!")
        break

    # Add user message to history
    conversation_history.append({"role": "user", "content": user_input})

    # Invoke the graph with the full history
    result = graph.invoke({"messages": conversation_history})

    # Get the final message from the LLM
    assistant_message = result["messages"][-1]

    # Add the assistant's final response back to the history
    conversation_history.append(assistant_message)

    print(f"Assistant: {assistant_message.content}\n")