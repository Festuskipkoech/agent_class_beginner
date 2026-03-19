# Lesson 2 — Building Your First LangGraph Agent

---

## 3.0 What is LangGraph — A Deeper Look

In lesson 1 we talked about why LangGraph exists and the problems it solves. Today we stop talking and start building. By the end of this lesson you will have written your first LangGraph agent from scratch.

Before we write any code, let us understand the four things that every LangGraph agent is made of. These four concepts are the foundation of everything you will build in this course and beyond.

| Building Block | What it is |
|---|---|
| State | What the agent knows and remembers |
| Nodes | The steps the agent takes |
| Edges | The connections between steps |
| Graph | Everything wired together and compiled |

Every LangGraph agent you ever build — simple or complex — is just these four things combined.

---

## 3.1 State

### What is State?

Imagine you are working on a task and you have a notepad next to you. Every time you find out something new you write it down. Every time you need to remember what you did earlier you look at your notepad. That notepad is the State.

State is a shared data structure that every node in the graph can read from and write to. It holds everything the agent knows at any point during a single run — the conversation history, search results, decisions made, anything you want to track.

Important: State does NOT persist between runs. It only lives for the duration of one graph run. The moment the graph reaches END the State is gone. We will come back to this shortly.

In Python, State is defined as a TypedDict. A TypedDict is just a dictionary where you declare in advance what keys it will have and what type of data each key holds. This makes your code predictable and easy to debug.

### The imports

```python
from typing_extensions import TypedDict
```

- `TypedDict` — comes from typing_extensions and lets us define a dictionary with fixed, typed keys. LangGraph requires State to be a TypedDict so it knows what data to expect and how to manage it

### Defining the State

```python
class State(TypedDict):
    messages: list
```

- `messages` is the only field in our State for now. It holds a list of all the messages in the conversation
- `list` tells Python and LangGraph that this field will always be a list
- Each item in that list is a dictionary with two keys — `role` (who is speaking) and `content` (what they said)
- Every node in the graph will receive this State, do its work, and return an updated version of it

---

## 3.2 Nodes

### What is a Node?

A node is a single step in your agent's workflow. It is a plain Python function. What makes it a node is that it receives the current State as input, does some work, and returns a dictionary with the parts of State it wants to update.

Think of a node like a worker on an assembly line. The item (State) comes in, the worker does their job, makes some changes, and passes it on to the next worker.

### The imports

```python
from langchain_ollama import ChatOllama
```

- `ChatOllama` — LangChain's connector to Ollama. Ollama is the software running Llama 3.2 locally on your machine. ChatOllama wraps it so LangChain can talk to it the same way it talks to any other LLM. Your LLM runs entirely on your laptop — no internet required, no API costs

### Setting up the LLM

```python
llm = ChatOllama(model="llama3.2", temperature=0)
```

- `model="llama3.2"` — tells Ollama which model to load. This must match exactly the name you used when you pulled the model with `ollama pull llama3.2`
- `temperature=0` — controls how creative or focused the model is. At 0 the model is very consistent and predictable. Higher values like 0.7 make the model more creative but less consistent. For agents we want 0 so the reasoning is reliable

### Writing the Node

```python
def chatbot(state: State):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}
```

- `def chatbot(state: State)` — defines a function called chatbot that takes the current State as its only argument
- `state["messages"]` — reads the messages list from the current State and passes it to the LLM so it knows what was said
- `llm.invoke(...)` — sends the messages to Llama 3.2 and waits for a response. invoke is LangChain's standard method for sending a request to an LLM
- `return {"messages": [response]}` — returns a dictionary with the new message. LangGraph merges this back into the State. Note: because we are using a plain list with no reducer, this REPLACES the messages list. More on this shortly

---

## 3.3 Edges

### What are Edges?

If nodes are the steps, edges are the arrows between them. An edge tells LangGraph — after this node finishes, go to that node next. Without edges your nodes would have no idea what order to run in.

There are two types of edges:

**Simple edge** — always moves to the same next node no matter what. Use this when the flow is fixed and predictable.

**Conditional edge** — the agent looks at the current State and decides which node to go to next. Use this when the flow depends on what the agent found or decided. We will use conditional edges in a later lesson when we add tools.

### The imports

```python
from langgraph.graph import StateGraph, START, END
```

- `StateGraph` — the main class used to build a LangGraph agent. You pass it your State definition and it creates a graph that knows how to manage that State
- `START` — a special built-in node that represents the entry point of the graph. Every graph must have an edge from START to the first real node
- `END` — a special built-in node that represents the exit point of the graph. When the agent reaches END the run is complete and the final State is returned

### Adding Edges

```python
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
```

- `add_edge(START, "chatbot")` — when the graph starts it immediately goes to the chatbot node
- `add_edge("chatbot", END)` — after the chatbot node finishes the run is complete
- The string "chatbot" must match exactly the name used when the node was registered with add_node
- The full flow looks like this: START ---------> chatbot ---------> END

---

## 3.4 The Graph

### What is the Graph?

The graph is where everything comes together. You create a StateGraph, register your nodes, connect them with edges, and then compile it. Compiling locks the structure and prepares it to run.

```python
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()
```

- `StateGraph(State)` — creates a new graph builder and tells it to use our State TypedDict as the shared data structure
- `add_node("chatbot", chatbot)` — registers our chatbot function as a node. The first argument is the name we give the node used in edges, the second is the actual Python function
- `compile()` — finalises the graph. After this the structure is locked and the graph is ready to run. The result is the graph object we use to invoke the agent

### Running the Graph

```python
result = graph.invoke({"messages": [{"role": "user", "content": user_input}]})
```

- `graph.invoke(...)` — starts the graph with an initial State. The dictionary we pass in becomes the starting State
- `{"role": "user", "content": user_input}` — the format LangChain uses for messages. role tells the LLM who is speaking — "user" for the human, "assistant" for the AI. content is the actual text of the message
- The graph runs through all the nodes and returns the final State when it reaches END
- `result["messages"][-1].content` — gets the last message in the messages list which is the assistant's response and reads its text content

---

## 3.5 Demo 1 — Chatbot Without Memory

We now build the full chatbot. Open chatbot_no_memory.py and run it together.

**Try this during the demo:**
1. Run the file
2. Tell the chatbot your name — type "My name is John"
3. Ask it something — "What is the capital of France?"
4. Now ask — "What is my name?"

It will have no idea. Every time you send a message the graph starts with only that one message in State. Nothing from previous messages is remembered.

**This is the problem we are about to solve.**

---

## 3.6 Why the Chatbot Forgets

Each time you call graph.invoke() LangGraph creates a brand new State from scratch using only what you pass in at that moment. It has no knowledge of anything said before. The messages list starts fresh with every single message.

Two reasons this happens:

- **No history passed forward** — we only pass the current message into graph.invoke(), not the full conversation
- **No reducer on the State** — because messages is a plain list with no add_messages reducer, any return from a node replaces the list entirely rather than adding to it

To fix this we need to do two things — keep the conversation history alive between runs, and use a reducer that appends instead of replaces. That is exactly what we do next.

---

## 3.7 Introducing add_messages — The Memory Reducer

Before we build the memory version we need to understand two new things.

### The new imports

```python
from typing import Annotated
from langgraph.graph.message import add_messages
```

- `Annotated` — a Python tool that lets you attach extra behaviour to a type. We use it here to tell LangGraph not just what type messages is, but also HOW to handle updates to it
- `add_messages` — a special LangGraph function called a reducer. A reducer controls what happens when a node returns an update to a State field. Without a reducer the new value replaces the old one. add_messages changes this — instead of replacing the messages list it APPENDS new messages to it. This is the key to memory

### Updating the State

```python
class State(TypedDict):
    messages: Annotated[list, add_messages]
```

- Compare this to the no memory version: `messages: list`
- The only difference is `Annotated[list, add_messages]` instead of just `list`
- This one change tells LangGraph — whenever a node returns new messages, do not replace the list, append to it
- The node itself does not change at all — the memory behaviour lives entirely in the State definition

---

## 3.8 Demo 2 — Chatbot With Memory

Now we solve the problem. Open chatbot_with_memory.py and run it.

The fix has two parts working together:

**Part 1 — conversation_history list**
```python
conversation_history = []
```
A plain Python list that lives outside the graph and accumulates every message — both user and assistant — across all graph runs.

**Part 2 — passing full history into every invoke**
```python
conversation_history.append({"role": "user", "content": user_input})
result = graph.invoke({"messages": conversation_history})
assistant_message = result["messages"][-1]
conversation_history.append({"role": "assistant", "content": assistant_message.content})
```

- Before invoking the graph we add the user's message to conversation_history
- We pass the FULL history into graph.invoke() — not just the current message
- After the graph runs we take the assistant's response and add it to conversation_history too
- Next time the user sends a message the history already contains everything said so far

**Try the same test:**
1. Type "My name is John"
2. Ask it something else
3. Type "What is my name?"

This time it remembers. Type exit to stop the chatbot.

---

## 3.9 Key Takeaways

- Every LangGraph agent is built from State, Nodes, Edges, and a Graph
- State is a TypedDict — it holds everything the agent knows during a run. It does NOT persist between runs by default
- Nodes are plain Python functions — they receive State, do work, and return updates
- Edges connect nodes — simple edges always move forward, conditional edges branch based on State
- Without a reducer, a node's return value REPLACES the State field — not appends to it
- add_messages is a reducer that appends new messages instead of replacing — this is what enables memory
- Passing the full conversation_history into every graph.invoke() is what makes the LLM aware of the full conversation
- Memory in LangGraph is not magic — it is the conversation history being carried forward explicitly

---

## What is Coming Next

In the next lesson we add tools to this agent. A tool is any ability we give the agent beyond just talking — searching the web, reading a database, running calculations. We start by building a simple custom tool then plug in Tavily for real-time web search. The chatbot becomes a proper agent that can go out and find information on its own.

---
