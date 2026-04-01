# Lesson 5: Agent Memory — Remembering Users Across Sessions
### AI Agents with Python — Course Notes

---

## What This Lesson Covers

Every agent we have built so far has the same fundamental flaw: the moment you close the terminal, it forgets everything. It does not know your name. It does not know what you asked it last week. It starts completely fresh every single time.

This lesson fixes that. By the end, your agent will remember who you are across sessions — storing facts about you in a persistent vector database and retrieving them the next time you speak.

We will look at three types of memory. We will build Type 2 in depth, because it is the most practically powerful and because it connects directly back to everything we built in Lesson 4.

---

## The Three Types of Agent Memory

### Type 1 — In-Context Memory (Short-Term)

This is memory that lives inside the conversation history list — the `conversation_history` variable you have been passing into the graph since Lesson 1. Every message, every tool result, every agent response is stored there and passed back into the model on every turn.

You have already been using this. We are just naming it now.

```python
conversation_history = [system_prompt]

# Every turn, the full history goes in and comes back out
result = graph.invoke({"messages": conversation_history})
conversation_history = result["messages"]
```

The strength of in-context memory is that the agent can see the entire conversation and reason over it naturally. The weaknesses are:

- It disappears when the session ends
- It has a hard ceiling — the model's context window (usually 128K–200K tokens)
- Long conversations become expensive to process because every token is sent every turn

In-context memory is ideal for reasoning within a single session. It cannot remember anything about you tomorrow.

---

### Type 2 — External Persistent Memory (Long-Term) — Built Today

This is memory that lives in a database on disk. Facts about the user are saved as text, embedded as vectors, and stored in Chroma — the same tool we used in Lesson 4 for the news knowledge base. The critical difference is that this collection stores facts *about the user*, not documents about the news.

Because Chroma persists to disk, these memories survive between sessions. The next time the agent starts, it loads the same database and the memories are still there.

This is what we build today.

---

### Type 3 — Summary Memory (Compressed Long-Term)

Imagine a user who has had fifty conversations with the agent, each an hour long. Keeping every single message in context would be impossible — the token count would be enormous. Summary memory solves this by periodically asking the model to compress what has been discussed into a short paragraph, then discarding the original messages.

```
Full conversation history (10,000 tokens)
        |
        v
LLM summarises: "The user discussed Kenya floods, asked about SHA, 
                 expressed concern about recruitment of Kenyans to Ukraine."
        |
        v
Summary stored (200 tokens) — original discarded
```

This is how production agents handle memory at scale. It is closer to how human memory actually works — we do not replay every word we have ever heard; we retain the gist of important experiences.

We will not build this today, but you now know it exists and why it matters.

---

## Why Chroma Again?

In Lesson 4 we used Chroma to store news chunks and retrieve the most relevant ones for a query. The mechanism is identical here — we are just changing what we store.

```
Lesson 4 Chroma store:  news chunks about Kenya floods, SHA, Russia
Lesson 5 Chroma store:  facts about the user — name, interests, preferences
```

The same embedding model converts text to vectors. The same similarity search retrieves the most relevant memories. The only difference is the content and purpose of what is stored.

This is intentional. Once you understand vector search, you can use it for anything that benefits from meaning-based retrieval. Documents, memories, code snippets, emails — the pattern is identical.

---

## The Two New Tools

We add two tools to the agent today: `save_memory` and `search_memory`. The agent decides when to call them based on their docstrings, exactly as it decides when to use any other tool.

### save_memory

```python
@tool
def save_memory(memory: str) -> str:
    """Save an important fact about the user to long-term memory.
    Use this whenever the user shares personal information such as their name,
    location, profession, interests, preferences, or any detail worth remembering
    for future conversations. Write the memory as a clear, standalone sentence.
    Example: 'The user's name is Amina.' or 'The user is interested in Kenyan politics.'
    """
    doc_id = str(uuid.uuid4())
    memory_vectorstore.add_texts(
        texts=[memory],
        ids=[doc_id],
        metadatas=[{"timestamp": datetime.now().isoformat()}]
    )
    return f"Memory saved: {memory}"
```

Notice a few things:

- We generate a unique `doc_id` with `uuid.uuid4()` so each memory is stored as a separate document
- We attach a `timestamp` in the metadata so we know when the memory was created
- The memory itself is a plain English sentence — human readable, not a schema or JSON object
- The docstring tells the agent exactly what kind of information to save and how to phrase it

### search_memory

```python
@tool
def search_memory(query: str) -> str:
    """Search long-term memory for facts previously saved about the user.
    Call this at the start of every conversation to recall who you are speaking with
    and what you already know about them. Also call this any time the user references
    something personal that you should already know.
    """
    results = memory_retriever.invoke(query)
    if not results:
        return "No relevant memories found."
    memories = [doc.page_content for doc in results]
    return "What I remember:\n" + "\n".join(f"- {m}" for m in memories)
```

The docstring instructs the agent to call this tool at the start of every session — before it knows who it is talking to — by searching with a broad query like "user profile". This is how the agent greets returning users by name without being told explicitly.

---

## The Two Chroma Stores

The agent now has two separate Chroma databases on disk:

```
./chroma_db/     — Kenya news knowledge base (from Lesson 4)
./memory_db/     — User memories (new in Lesson 5)
```

They use the same embedding model and the same Chroma library but are completely independent. This separation matters — you do not want a query about the user's name returning news article chunks, and you do not want a query about Kenya floods returning personal facts.

```python
# News store — retrieves up to 3 relevant chunks
news_retriever = news_vectorstore.as_retriever(search_kwargs={"k": 3})

# Memory store — retrieves up to 5 relevant memories
memory_retriever = memory_vectorstore.as_retriever(search_kwargs={"k": 5})
```

We set `k=5` for memory retrieval because a user might have many saved facts and we want a broader recall. For news chunks we stay at `k=3` to keep the context focused.

---

## The System Prompt

The system prompt is where we give the agent its instructions for when and how to use memory. This is just as important as the tool docstrings.

```python
system_prompt = SystemMessage(
    content=(
        "You are a helpful, personable news assistant with both memory and knowledge tools.\n\n"
        "MEMORY RULES:\n"
        "- At the start of every conversation, call search_memory with a broad query like 'user profile' "
        "to recall who you are speaking with.\n"
        "- Whenever the user tells you something personal (name, job, location, interests, preferences), "
        "call save_memory immediately to store it.\n"
        "- When the user references something you should know, call search_memory to retrieve it.\n"
        "- Greet returning users by name if you remember them.\n\n"
        "KNOWLEDGE RULES:\n"
        "- Use search_knowledge_base for questions about Kenya floods, Russia recruitment, SHA, politics.\n"
        "- Use search_news for today's breaking news not covered in stored documents.\n"
        "- For simple greetings, respond directly without calling any tools."
    )
)
```

The explicit `MEMORY RULES` section is what makes the agent proactively search memory at the start of each session rather than waiting to be asked. Without this, the agent would only use `search_memory` when the user explicitly referenced something personal.

---

## What the Agent Can Now Do

| Situation | What Happens |
|-----------|--------------|
| Session starts | Agent calls `search_memory("user profile")` — greets you by name if you are a returning user |
| User says "My name is James" | Agent calls `save_memory("The user's name is James.")` |
| User says "I work in agriculture" | Agent calls `save_memory("The user works in agriculture.")` |
| User asks about Kenya floods | Agent calls `search_knowledge_base` |
| User asks about today's news | Agent calls `search_news` |
| Session ends, new session starts | Agent loads from disk — all memories intact |

---

## The Full Architecture

```
Session 1                         Session 2
----------                        ----------
User: "My name is James"          Session starts
Agent: save_memory()              Agent: search_memory("user profile")
  -> ./memory_db/                   -> "The user's name is James."
     "The user's name is James."  Agent: "Welcome back, James!"
```

```
All Tools Available to the Agent
----------------------------------
get_current_datetime     — current date/time
search_news              — Tavily live web search
search_knowledge_base    — ./chroma_db/ (Kenya news)
save_memory              — writes to ./memory_db/
search_memory            — reads from ./memory_db/
```

---

## How This Connects to the Full Course

Look at the arc:

- Lesson 1 — Basic agent, no tools, no memory
- Lesson 2 — Agent with tools (datetime, reading time)
- Lesson 3 — Agent with live web search (Tavily)
- Lesson 4 — Agent with local knowledge base (Chroma, embeddings, chunking)
- Lesson 5 — Agent with persistent memory (second Chroma store, user facts across sessions)

Every lesson has built on the previous one. Chroma appeared in Lesson 4 to store documents. In Lesson 5 you discover that the same tool, used differently, can give your agent a memory. The architecture is identical — only the purpose of what is stored changes.

This is the core insight of the course: these are composable building blocks. You now have everything you need to build agents that are genuinely useful over time.

---

## Package Requirements

No new packages are needed for this lesson. Everything was installed in Lesson 4:

```bash
pip install chromadb langchain-google-genai langchain-text-splitters
pip install langchain langchain-community langgraph langchain-tavily python-dotenv
pip install pillow
```

Your `.env` file still needs:

```
GOOGLE_API_KEY=your_key_here
TAVILY_API_KEY=your_key_here
```

---

## Exercises to Try

1. Start the agent. Tell it your name, your job, and one topic you care about. Exit.
2. Start the agent again. See whether it greets you by name without being told.
3. Tell the agent a second fact about yourself. Exit and restart. Ask it what it knows about you.
4. Try asking the agent about the Kenya floods while it also knows your name — watch it use both stores in one session.
5. Delete `./memory_db/` and start fresh. Notice what the agent says when there are no memories.

---

## Key Takeaways

- **In-context memory** is the conversation history — powerful within a session, gone when it ends
- **External persistent memory** uses a vector database to store facts across sessions — the agent remembers you tomorrow
- **Summary memory** compresses long histories to fit within token limits — the production-scale solution
- Two Chroma stores solve two different problems: one stores knowledge, one stores memory — same tool, different purpose
- The system prompt is what teaches the agent when to proactively search and save memory
- The docstring on each tool is what guides the agent to use it correctly and at the right moment
- You now have a complete, memorable agent — and a complete course

---

## What Comes Next (Beyond This Course)

If you continue building from here, the natural next areas are:

- **LangSmith** — observability and debugging for your agent's reasoning steps
- **Human-in-the-loop** — pausing the graph for human approval before the agent acts
- **Multi-agent systems** — multiple specialised agents working together, passing tasks between each other
- **Structured memory schemas** — storing memories as typed objects (user profiles, preferences) rather than plain sentences

The foundations you have now — LangGraph, tools, embeddings, vector search, memory — are the same foundations used in production agent systems. The path forward is clear.

---