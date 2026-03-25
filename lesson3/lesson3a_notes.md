# Lesson 3A — Tools, What They Are and Why They Matter

---

## 4.0 What is a Tool

So far our agent can only talk. It reasons from whatever it already knows — its training data. That is useful but limited. What if the agent needs to know what time it is right now? Or estimate how long an article takes to read? It cannot figure that out by reasoning alone.

A tool is a Python function we give the agent so it can do something beyond talking. When the agent decides it needs help answering a question, it calls the right tool, gets the result, and uses that result to respond.

This is the moment a chatbot becomes an agent. An agent does not just reply — it acts.

---

## 4.1 How the Agent Decides to Use a Tool

The agent does not call tools randomly. Every time you send a message the agent reasons first. It asks itself — can I answer this from what I already know, or do I need a tool?

If it decides a tool is needed it picks the right one, calls it with the right inputs, reads the output, and then forms its final response. This reasoning loop is what makes agents powerful.

This is also why we write a strong system prompt. With Llama 3.2 running locally we nudge the model to always check its available tools before answering from memory. Without that nudge it sometimes skips the tools entirely.

---

## 4.2 The @tool Decorator

LangChain gives us a simple way to turn any Python function into a tool — the `@tool` decorator.

```python
from langchain_core.tools import tool

@tool
def get_current_datetime() -> str:
    """Returns the current date and time."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
```

Three things to note:

- `@tool` sits directly above the function definition — that is all it takes to register a function as a tool
- The docstring is not optional — the agent reads it to understand what the tool does and when to use it. Write it clearly
- The return type hint tells LangChain what kind of value to expect back

---

## 4.3 Binding Tools to the LLM

Decorating a function makes it a tool. But the agent still does not know the tool exists until we bind it.

Binding is how we tell the LLM — here are the tools available to you.

```python
llm_with_tools = llm.bind_tools([get_current_datetime, estimate_reading_time, categorize_topic])
```

- `bind_tools` takes a list of tools and returns a new version of the LLM that is aware of them
- From this point on the agent can see the tools and reason about when to call them
- We use `llm_with_tools` inside our node instead of the plain `llm`

---

## 4.4 The ToolNode

When the agent decides to call a tool, something has to actually run it. That is the job of `ToolNode`.

```python
from langgraph.prebuilt import ToolNode

tool_node = ToolNode([get_current_datetime, estimate_reading_time, categorize_topic])
```

- `ToolNode` is a built-in LangGraph node that handles tool execution for us
- It receives the tool call from the agent, runs the matching function, and returns the result back into State
- We pass it the same list of tools we gave to `bind_tools`

---

## 4.5 Conditional Edges — The Agent Deciding What to Do Next

In lesson 2 our graph had simple edges — the flow was always START to chatbot to END. No decisions.

Now we need the graph to branch. After the agent responds we need to ask — did the agent call a tool or did it just reply?

- If it called a tool, go to the tool node to run it, then come back to the agent
- If it just replied, we are done — go to END

This is a conditional edge.

```python
from langgraph.prebuilt import tools_condition

graph_builder.add_conditional_edges("agent", tools_condition)
```

- `tools_condition` is a built-in LangGraph function that checks whether the last message contains a tool call
- If yes it routes to the tool node. If no it routes to END
- This creates the loop — agent thinks, calls tool if needed, gets result, thinks again, replies

The flow now looks like this:

```
START --> agent --> tool called? --> yes --> tool_node --> agent --> END
                                --> no  --> END
```

---

## 4.6 Where This is Going — The News Agent

The three tools we build today are simple on purpose. They teach you the structure — decorator, docstring, bind, ToolNode, conditional edge.

In part B we replace the custom tools with a real news API. The agent will fetch live headlines, and all the structure you learned today stays exactly the same. The tools just become more powerful.

Think of part A as building the engine. Part B adds the fuel.

---

## 4.7 Demo — News Tool Agent

Open `lesson_3a_agent.py` and run it.

We have three tools wired in:

- `get_current_datetime` — returns today's date and time. News is time-sensitive, this one earns its place
- `estimate_reading_time` — paste any text and get back an estimated reading time in minutes
- `categorize_topic` — give it a headline and it returns a category like Tech, Politics, Sports, Business

Try these prompts and watch the agent decide:

- "What is today's date?"
- "How long would it take to read this: Kenya's parliament passed a new finance bill today after weeks of debate"
- "What category does this headline fall under: OpenAI releases new model"
- "Tell me something about history" — watch it answer without touching any tool

---

## 4.8 Key Takeaways

- A tool is a Python function decorated with `@tool` that gives the agent an ability beyond talking
- The docstring is what the agent reads to decide when to use a tool — write it clearly
- `bind_tools` tells the LLM which tools exist
- `ToolNode` is the node that actually runs the tool when the agent calls it
- `tools_condition` is a conditional edge that checks whether a tool was called and routes accordingly
- The agent reasons before acting — it decides whether to call a tool or reply directly
- Part B builds on everything here — same structure, real news API

---

## What is Coming Next

In part B we wire in a real news API. The agent will search for live headlines and you will see everything from today applied at scale. The structure stays the same — we just swap the custom tools for something that reaches out to the real world.

---