# Session Notes — AI Agents Course Planning
> **Date:** March 18, 2026  
> **Purpose:** Course design discussion — decisions and rationale  

---

## What We Set Out to Do
Plan a course on AI agents using Python, LangChain, a vector database, and Llama (locally hosted). The goal was to discuss the course structure before producing any documents.

---

## Who Is This Course For?
- **Level:** Beginner with some intermediate exposure
- **What they already know:** Python, LangChain basics (chains, prompt templates), Ollama with Llama 3.2 installed and running locally
- **What they do NOT know yet:** Agents, LangGraph, vector databases, embeddings, memory types

---

## Key Decisions Made

### Scope — Single Agent Only
- This course focuses entirely on **single agents**
- Multi-agent systems will be a **separate advanced course** built later
- The advanced course will cover: supervisor agents, parallel agents, pipeline patterns, and deeper memory modules
- We will **mention** multi-agent and advanced topics so students know where they are headed

### LangGraph vs LangChain AgentExecutor
- Originally planned to use LangChain's AgentExecutor
- Decided to **introduce LangGraph** instead because:
  - AgentExecutor is considered legacy in 2026
  - LangGraph is the modern, production-recommended way to build agents
  - It also sets students up for the advanced multi-agent course where LangGraph really shines
- **Depth:** We are NOT going deep into LangGraph — just enough to understand why it exists and build with it

### Web Search — Tavily
- Chosen tool: **Tavily Search API**
- Reasons: officially recommended by LangChain, native `TavilySearchResults` tool built in, very easy to integrate
- Free tier: **1,000 searches/month** — sufficient for a 2-week course
- Students sign up at tavily.com and get a free API key
- Other options considered: SerpAPI (100/month free — too low), DuckDuckGo (no key needed but unreliable)

### Vector Database — Chroma
- Using **Chroma** for this course — local, free, no setup friction
- Perfect for learning the concepts without cloud complexity
- Advanced course can introduce Pinecone for production/cloud use

### Memory — Three Types Together
- The three memory types will be taught **in one module** in this course:
  - **Sensory memory** — current prompt window, what the agent sees right now
  - **Short-term memory** — conversation history within a session
  - **Long-term memory** — vector store, persisted across sessions
- In the **advanced course**, each memory type will get its own dedicated module

### Course Duration — 2 Weeks
- Hard constraint of 2 weeks
- Works because students already know Python and LangChain — no time wasted on prerequisites
- Ollama setup also skipped — they already have it running

---

## Today's Deliverables
Three files to push to GitHub:

| File | Description |
|------|-------------|
| `course_path.md` | Full 2-week course syllabus |
| `session_notes.md` | This file — planning decisions and rationale |
| `demo_agent.py` | Single file demo agent — Tavily + Ollama + LangChain, well commented |

### Demo Agent Spec
- Single `.py` file, no folders
- API key hardcoded with a clear placeholder — students do not have .env knowledge yet
- Ollama running Llama 3.2 locally
- Tavily for web search
- `create_agent` from `langchain.agents` — LangChain 1.0 modern approach
- Minimal comments in code — all detailed explanations live here in session notes

---

## Code Explanation Notes (for instructors and students)

### Why create_agent and not create_react_agent?
`create_react_agent` and `AgentExecutor` are officially deprecated as of LangChain 1.0 (released late 2025).
The replacement is `create_agent` from `langchain.agents`. It runs on LangGraph under the hood, uses a middleware system for customization, and is the long-term supported path. Students should learn this from day one rather than learn something deprecated.

### What is ReAct?
ReAct stands for Reasoning + Acting. It is a prompting pattern that teaches the LLM to think in a structured loop instead of immediately guessing an answer.

The loop looks like this:

```
Thought:     "The user wants current info. I should search the web."
Action:      "I will use tavily_search with query: AI agent frameworks 2026"
Observation: "Here are 3 search results..."
Thought:     "I have enough information to answer."
Final Answer: "The most popular frameworks are..."
```

This loop can repeat multiple times. If the first search is not enough the agent searches again with a different query. `create_agent` implements this loop automatically — the LLM decides when it has enough information to stop.

### What each import does
- `os` — built-in Python module used to set the Tavily API key as an environment variable
- `ChatOllama` — LangChain's connector to Ollama running locally; no internet or API costs for the LLM
- `TavilySearchResults` — a ready-made tool that calls the Tavily search API and returns web results
- `create_agent` — LangChain 1.0 function that wires the LLM and tools into an agent running on LangGraph

### What each parameter does
- `model="llama3.2"` — must match exactly what was pulled with `ollama pull llama3.2`
- `temperature=0` — makes the model focused and consistent; higher values make it more creative but less reliable for agent reasoning
- `max_results=3` — Tavily returns the 3 most relevant search results; increase for more coverage
- `system_prompt` — tells the agent what role it plays and how to behave; replaces the old hub.pull("hwchase17/react") prompt

### What langgraph does here
`create_agent` uses LangGraph under the hood to manage the agent loop. Students do not need to write any LangGraph code directly — they just use `create_agent`. This is intentional for this course. The advanced course will teach LangGraph explicitly.

---

## What Was Intentionally Left Out of This Course
These topics were discussed but deliberately scoped out — saved for the advanced course:
- Deep dive into LangGraph internals
- Multi-agent systems (supervisor, parallel, pipeline)
- Advanced chunking strategies
- Pinecone (cloud vector DB)
- Each memory type as its own deep module
- Agent evaluation and testing

---

## Open Items / To Revisit
- Final project domain not yet decided — to be confirmed before Week 2 content is written
- Advanced course outline — to be planned in a future session

---

