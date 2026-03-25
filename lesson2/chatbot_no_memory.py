from typing_extensions import TypedDict 
# 
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama

# LangGraph uses State to pass data between nodes during a graph run
# class State is the blueprint that defines what data our agent will track
class State(TypedDict):
    # TypedDict stores data in dictionary form with fixed, typed keys
    # messages has type list — so our State is a dictionary that contains a list
    # each item in that list is a message dictionary with role and content
    messages: list

# initialize our LLM — connects LangChain to Ollama running Llama 3.2 locally
# temperature controls creativity: 0 = focused and consistent, 1 = more creative
llm = ChatOllama(model="llama3.2", temperature=0)  

# each node is a Python function that receives the current State, does work, and returns updated State
# we declare the function here and pass it to add_node below
def chatbot(state: State):
    response = llm.invoke(state["messages"])  # LLM reads the messages list from State
    return {"messages": [response]}
    # we return a dictionary — LangGraph merges this back into the State
    # note: without add_messages, this REPLACES the messages list rather than appending to it
    # that is why this chatbot has no memory — every invoke starts with only the current message

# initialize the graph with our State definition
graph_builder = StateGraph(State)

# add our first and only node
# first argument is the name we give the node — used when wiring edges
# second argument is the Python function the node will execute
graph_builder.add_node("chatbot", chatbot)

# edges are connections between nodes — they define the flow of the graph
# this edge tells LangGraph: when the graph starts, go to the chatbot node
graph_builder.add_edge(START, "chatbot")

# this edge tells LangGraph: after the chatbot node finishes, end the run
graph_builder.add_edge("chatbot", END)

# so the full flow is: START ---------> chatbot ---------> END

# compile() locks the graph structure and makes it ready to run
# we assign it to 'graph' — this is the object we use to invoke the agent
graph = graph_builder.compile()     

print("Chatbot is running. Type 'exit' to stop.\n")

while True:
    user_input = input("You: ")
    # take the user's input

    if user_input.lower() == "exit":
        print("Nice chat, goodbye!")
        break
    # end ends the program
    # exit if the user types 'exit'

    result = graph.invoke(
        # we pass the initial State to the graph — messages is the key we defined in our State
        # the value is a list containing one dictionary — the user's current message
        # role tells the LLM who is speaking: "user" for human, "assistant" for AI
        # content is the actual text of the message
        {"messages": [{"role": "user", "content": user_input}]}
    )

    # graph.invoke() runs the full graph and returns the final State when it reaches END

    print(f"Assistant: {result['messages'][-1].content}\n")
    # result["messages"] is the list of messages in the final State
    # [-1] gets the last message in the list — which is always the assistant's response
    # .content reads the text from the message object