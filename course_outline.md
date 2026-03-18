# AI Agents with Python — Course Path
> **Level:** Beginner with intermediate Python & LangChain knowledge  
> **Duration:** 2 Weeks  
> **Stack:** Python · LangChain · LangGraph (intro) · Ollama (Llama 3.2) · Chroma · Tavily  

---

## Prerequisites
Students joining this course are expected to already know:
- Python fundamentals (functions, loops, classes)
- LangChain basics — chains and prompt templates
- Ollama installed locally with Llama 3.2 running

---

## Learning Outcomes
By the end of this course, students will be able to:
- Explain what an AI agent is and how it reasons using the ReAct loop
- Understand why LangGraph is preferred over LangChain AgentExecutor in production
- Build a single agent that can use multiple tools
- Give an agent the ability to search the web using Tavily
- Store and retrieve knowledge using a vector database (Chroma)
- Implement all three types of agent memory
- Build a complete single agent that combines web search, vector DB knowledge, and memory

---

## Week 1 — Building the Agent Foundation

### 1.1 What is an AI Agent?
- Agents vs. simple LLM calls vs. chains — what makes an agent different
- The ReAct reasoning loop — Reason, Act, Observe
- How an agent decides what tool to use and when to stop
- Overview of the course stack and how the pieces connect

### 1.2 LangGraph — Why and Just Enough to Build
- Why LangChain's AgentExecutor is being replaced in production
- What LangGraph brings — state, control flow, reliability
- Core concepts: nodes, edges, and state (conceptual, not deep dive)
- Building our first agent graph — just enough to get running

### 1.3 Tools — Giving the Agent Abilities
- What is a tool in the context of an agent
- Building a simple custom tool in Python
- Plugging in Tavily for real-time web search
- Testing the agent with web search — seeing ReAct in action

### 1.4 Vector Databases — The Agent's Knowledge Store
- What a vector database is and why agents need one
- What embeddings are — turning text into searchable numbers
- Introduction to Chroma — local, free, beginner friendly
- Chunking — why we split documents before storing them

---

## Week 2 — Knowledge, Memory & Bringing It Together

### 2.1 Chunking & Embeddings in Practice
- Chunking strategies — size, overlap, and why it matters
- Generating embeddings with a local model
- Storing documents in Chroma
- Querying Chroma — similarity search in action

### 2.2 Wiring the Vector DB into the Agent
- Creating a retriever tool from Chroma
- Giving the agent two tools — Tavily (web) + Chroma (local knowledge)
- Watching the agent decide which source to use based on the question

### 2.3 Agent Memory — All Three Types
- **Sensory memory** — what the agent sees right now (current prompt + tool results)
- **Short-term memory** — conversation history carried through a session (ConversationBufferMemory)
- **Long-term memory** — persisted knowledge in the vector store, available across sessions
- Wiring all three into the agent
- Seeing the difference — agent with vs. without memory

### 2.4 Final Project
Build a complete single agent that:
- Takes questions from the user in a loop
- Searches the web via Tavily when it needs current information
- Queries a local Chroma knowledge base for stored documents
- Remembers the conversation within a session (short-term)
- Persists knowledge across sessions (long-term via Chroma)
- Runs entirely locally on Llama 3.2 via Ollama

---

## Tools & Setup

| Tool | Purpose | Cost |
|------|---------|------|
| Python 3.11+ | Programming language | Free |
| Ollama + Llama 3.2 | Local LLM — no API costs | Free |
| LangChain | Chains, prompt templates, agent wiring | Free |
| LangGraph | Agent state graph — modern agent runtime | Free |
| Chroma | Local vector database | Free |
| Tavily | Web search API for agents | Free (1,000 searches/month) |

---

## What Comes Next — Advanced Course (Preview)
This course covers single agents. The advanced course will go deeper into:
- Multi-agent systems — supervisor, parallel, and pipeline patterns
- Each memory type as its own dedicated module
- Advanced chunking and embedding strategies
- Agents that collaborate across specialized roles

---

*Course designed for the 2026 AI engineering cohort.*