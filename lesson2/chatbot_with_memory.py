from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama

# Annotated allows us to attach extra behaviour to a type
# here we use it to tell LangGraph HOW to handle updates to the messages field


# same State blueprint as before — but with one key difference
# messages is now Annotated with add_messages
# add_messages is a reducer — instead of replacing the messages list when a node returns new messages
# it APPENDS the new messages to the existing list
# this is what gives the chatbot memory — the list keeps growing with every exchange

class State(TypedDict):
    messages: Annotated[list, add_messages]

# initialize our LLM — same as before, Llama 3.2 running locally via Ollama
llm = ChatOllama(model="llama3.2", temperature=0)

# same chatbot node as before — nothing changes here
# the memory magic does not happen inside the node
# it happens in the State definition above via add_messages
def chatbot(state: State):
    response = llm.invoke(state["messages"])  # LLM reads the full conversation history from State
    return {"messages": [response]}
    # add_messages ensures this response is APPENDED to the messages list, not replacing it

# the graph structure is identical to the no memory version
# START ---------> chatbot ---------> END
# the difference is entirely in the State definition and how we invoke the graph below
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()

# conversation_history is a list that lives outside the graph
# it accumulates every message — user and assistant — across all graph runs
# this is what we pass into graph.invoke() each time so the LLM always sees the full history
conversation_history = []

print("Chatbot is running. Type 'exit' to stop.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Nice chat, goodbye!")
        break

    # before invoking the graph we add the user's message to the history
    # role "user" tells the LLM this message came from the human
    conversation_history.append({"role": "user", "content": user_input})

    # we pass the FULL conversation history into graph.invoke() every time
    # this is the key difference from chatbot_no_memory.py
    # there we only passed the current message — here we pass everything
    result = graph.invoke({"messages": conversation_history})

    # get the last message from the result — that is always the assistant's response
    assistant_message = result["messages"][-1]

    # add the assistant's response to the history as well
    # role "assistant" tells the LLM this message came from the AI
    # this way the next invoke will include both sides of the conversation
    conversation_history.append({"role": "assistant", "content": assistant_message.content})

    print(f"Assistant: {assistant_message.content}\n")