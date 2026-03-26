# Lesson 3B — Real News Agent with Tavily and Groq

---

## 5.0 What Changes in Part B

In part A we built the engine. We had three simple tools, a local Llama model, and a working agent loop. Everything was self-contained — no external services, no live data.

Part B adds the fuel. We wire in a real search API so the agent can fetch live news from the internet. We also swap the local model for Groq — a cloud-based inference provider that runs the same open models but significantly faster.

The structure does not change. Same State, same agent node, same ToolNode, same conditional edges. What changes is what the agent can do.

---

## 5.1 Why We Drop categorize_topic

In part A we needed a `categorize_topic` tool because Llama 3.2 running locally sometimes struggles with reasoning tasks. We gave it a keyword-matching shortcut.

Groq runs a capable model that handles categorization naturally. When the agent fetches a news article and you ask what category it falls under, Groq reasons it out from the content — no hardcoded keywords needed.

This is an important principle. Tools should do things the LLM genuinely cannot do on its own — fetch live data, check the time, run calculations. If the LLM can reason through it, let it. Do not build a tool for something the model already does well.

So we drop `categorize_topic` and keep the tools that do real work the LLM cannot replicate:

- `get_current_datetime` — the model has no clock
- `estimate_reading_time` — a deterministic calculation
- `search_news` — the model has no internet access

---

## 5.2 Introducing Groq

Groq is a cloud inference provider. You send it a message, it runs the model and sends back a response — same as any LLM API. What makes it stand out is speed. Groq uses custom hardware that processes tokens much faster than standard GPU inference.

For us the switch is one line. LangChain supports Groq through `langchain_groq`.

```python
from langchain_groq import ChatGroq

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
```

Everything else stays the same. We still call `bind_tools`, we still pass messages in, we still get a response out. The agent does not know or care that the model changed — it just gets faster, smarter replies.

You will need a Groq API key. Get one free at console.groq.com and add it to your `.env` file:

```
GROQ_API_KEY=your_key_here
```

---

## 5.3 Introducing Tavily

Tavily is a search API built specifically for AI agents. Regular web search returns raw HTML — links, ads, navigation menus, all mixed together. Tavily returns clean, structured results that an agent can actually read and reason from.

When our agent calls `search_news`, Tavily goes out to the web, finds relevant articles, and returns summaries with titles, URLs, and content snippets. The agent reads those and forms its response.

You will need a Tavily API key. Get one free at app.tavily.com and add it to your `.env`:

```
TAVILY_API_KEY=your_key_here
```

LangChain has a built-in Tavily tool we can use directly:

```python
from langchain_community.tools.tavily_search import TavilySearchResults

tavily_tool = TavilySearchResults(max_results=5)
```

We wrap it with `@tool` so we control the docstring — which is what the agent reads to decide when to use it.

---

## 5.4 The search_news Tool

```python
from langchain_community.tools.tavily_search import TavilySearchResults

tavily = TavilySearchResults(max_results=5)

@tool
def search_news(query: str) -> str:
    """Search for recent news articles on any topic.
    Use this whenever the user asks about current events, news, or recent developments.
    Input should be a search query like 'AI news today' or 'Kenya elections 2025'.
    """
    results = tavily.invoke(query)
    if not results:
        return "No results found."
    output = []
    for r in results:
        output.append(f"Title: {r.get('title', 'N/A')}\nURL: {r['url']}\nSummary: {r.get('content', '')[:300]}")
    return "\n\n".join(output)
```

Three things to note:

- The docstring is specific — it tells the agent exactly when to reach for this tool and what kind of input to pass
- We cap Tavily results at 5 to keep the context manageable
- We trim the content to 300 characters per result — enough for the agent to reason from, not so much that it floods the context

---

## 5.5 The Full Tool List

```python
tools = [get_current_datetime, estimate_reading_time, search_news]
llm_with_tools = llm.bind_tools(tools)
tool_node = ToolNode(tools)
```

Three tools. The agent knows what each one does from its docstring. When a user asks about current events, it reaches for `search_news`. When asked about time, `get_current_datetime`. When given a long article, `estimate_reading_time`. For everything else — summarizing, categorizing, explaining — Groq reasons it out directly.

---

## 5.6 The Graph — Nothing Changes

```python
graph_builder = StateGraph(State)
graph_builder.add_node("agent", agent)
graph_builder.add_node("tools", tool_node)
graph_builder.add_edge(START, "agent")
graph_builder.add_conditional_edges("agent", tools_condition)
graph_builder.add_edge("tools", "agent")
graph = graph_builder.compile()
```

This is identical to part A. The loop is the same. The conditional edge is the same. Groq and Tavily slot right in because LangChain treats them the same way as any other LLM and tool. This is the value of building on a consistent structure — you can upgrade the components without rebuilding the plumbing.

---

## 5.7 Demo — News Agent in Action

Open `lesson_3b_agent.py` and run it. Make sure your `.env` has both API keys set.

Try these prompts:

- "What is happening in AI today?" — agent calls `search_news`, Groq reads the results and summarizes
- "What is the latest news from Kenya?" — same flow, different query
- "How long would it take to read this: [paste an article]" — agent calls `estimate_reading_time`
- "What is today's date?" — agent calls `get_current_datetime`
- "What category does that AI story fall under?" — Groq reasons it out without calling any tool
- "Tell me something about the Roman Empire" — agent replies from its training, no tool needed

Watch how the agent decides. News questions go to Tavily. Date questions go to the clock. Category questions Groq handles on its own. That reasoning loop — checking tools first, using them only when needed — is what makes this an agent and not just a chatbot.

---

## 5.8 Key Takeaways

- Part B extends part A — same structure, upgraded components
- Groq replaces Ollama for faster, more capable cloud inference. One line change
- Tavily gives the agent real internet access with clean, agent-friendly results
- We dropped `categorize_topic` because Groq handles reasoning tasks directly — tools should do what the LLM cannot
- The docstring on each tool is what the agent reads to decide when to call it — write it clearly
- The graph structure is unchanged — LangGraph lets you swap components without rebuilding

---

## What Is Coming Next

The agent can now fetch live news. In the next lesson we look at memory — giving the agent the ability to remember what was discussed earlier in a conversation and refer back to it. The tools stay, the graph grows.

---