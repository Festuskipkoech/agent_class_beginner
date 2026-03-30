# Lesson: Vector Databases, Embeddings & Chunking
### AI Agents with Python — Course Notes

---

## What This Lesson Covers

In the previous lesson we built an agent that can search the live web via Tavily. Today we give the agent a second type of knowledge — a **local knowledge base** it can search through. To do that we need three things to work together: chunking, embeddings, and a vector database.

By the end of this lesson your agent will have two tools:
- `search_news` — searches the live internet (Tavily, already built)
- `search_knowledge_base` — searches documents you have stored locally (Chroma, built today)

---

## Concept 1 — What is a Vector Database?

A regular database stores data in rows and columns. You search it by matching exact values: `WHERE name = "Kenya"`. This works well for structured data, but it breaks down when you want to search by **meaning**.

A vector database stores data as **vectors** — lists of numbers that represent the meaning of a piece of text. When you search it, you don't look for an exact keyword match. You look for vectors that are mathematically close to your query vector. This is called **similarity search** or **semantic search**.

Example: If your knowledge base contains the sentence "The floods in Nairobi killed 42 people", and you ask "How many died in the Kenya disaster?", a keyword search would likely fail — none of those words match. A vector search would succeed because both sentences carry similar meaning, and their vectors will be close together in the vector space.

**Chroma** is the vector database we use in this course. It is:
- Free and open source
- Runs entirely on your local machine — no external service needed
- Beginner friendly — minimal setup
- Persistent — data survives between sessions when you point it at a folder

---

## Concept 2 — What are Embeddings?

An embedding is what converts text into a vector. You pass a piece of text to an embedding model, and it returns a list of numbers (for example, 768 numbers). That list is the mathematical representation of the text's meaning.

The key property is that **similar meanings produce similar vectors**. This is not keyword matching — it is meaning matching. The embedding model has learned, from massive amounts of text, what concepts are related to each other.

```
"Kenya floods 2026"           → [0.12, -0.45, 0.88, 0.03, ...]
"Nairobi rain disaster"       → [0.14, -0.42, 0.91, 0.01, ...]  <- very close
"Python programming language" → [-0.67, 0.22, -0.11, 0.54, ...] <- very far
```

When you query the database, Chroma embeds your query using the same model, then finds the stored vectors that are closest to it. Those are your results.

### The Embedding Model We Use

We use **Google's Gemini embedding model** via the `langchain-google-genai` package. Specifically:

```
model="models/gemini-embedding-001"
```

This model is:
- Free on Google's AI Studio free tier
- High quality — optimised for retrieval and document search
- Easy to wire into LangChain with one import

---

## Concept 3 — What is Chunking?

You cannot embed an entire document as one vector. A 10-page article would produce one vector, and that single vector would be a blurry average of everything in the document. When you searched for "Kenya floods", it might match poorly even if the article has an entire section about floods.

**Chunking** solves this by splitting your document into smaller pieces before embedding. Each chunk gets its own vector. Now when you search, Chroma can return the specific chunk that is relevant to your question — not the entire document.

### The Two Key Parameters

**`chunk_size`** — how many characters each chunk should be (roughly). A common starting point is 500–1000 characters.

**`chunk_overlap`** — how many characters of overlap to include between consecutive chunks. For example with `chunk_size=500` and `chunk_overlap=100`, chunk 2 will begin 100 characters before chunk 1 ends. This prevents important sentences from being cut in half and losing their context.

### Visualising Chunking

```
Document:  [---paragraph 1---][---paragraph 2---][---paragraph 3---]

Chunk 1:   [---paragraph 1---][--start of para 2--]
Chunk 2:            [--end of para 1--][---paragraph 2---][--start--]
Chunk 3:                              [--end of para 2--][para 3---]
```

The overlap ensures that information at the boundary of one chunk is also present in the next.

### The Splitter We Use

LangChain's `RecursiveCharacterTextSplitter` is the recommended splitter for plain text. It works by trying to split at natural boundaries in order — paragraphs first (`\n\n`), then lines (`\n`), then sentences (`.`), then words (` `). Only if none of those work does it split mid-word. This preserves meaning as much as possible.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
```

---

## API Keys Setup

### 1. Google AI Studio (for Gemini Embeddings and LLM) — Free

1. Go to [https://aistudio.google.com](https://aistudio.google.com)
2. Sign in with your Google account
3. Click **Get API key** in the left sidebar
4. Click **Create API key**
5. Copy the key — it starts with `AIza...`
6. Add it to your `.env` file:

```
GOOGLE_API_KEY=your_key_here
```

**Free tier limits:** 1,500 embedding requests per day, 1 million tokens per minute. More than enough for this course.

> **Note:** In this lesson we switch the agent LLM from Groq to **Gemini**. Both the embeddings and the agent brain now use the same `GOOGLE_API_KEY`. You no longer need a separate `GROQ_API_KEY` for this lesson.

### 2. Tavily (already set up from Lesson 3A/3B)

```
TAVILY_API_KEY=your_key_here
```

Get one free at [https://app.tavily.com](https://app.tavily.com) if you have not already.

---

## Package Installation

```bash
pip install chromadb
pip install langchain-google-genai
pip install langchain-text-splitters
```

You should already have these from previous lessons:

```bash
pip install langchain langchain-community langgraph langchain-tavily python-dotenv
```

---

## How It All Connects

Here is the full picture of what we are building today and how the pieces connect:

```
Your Text File (news_article.txt)
        |
        v
RecursiveCharacterTextSplitter
        |  splits into chunks of ~500 chars
        v
GoogleGenerativeAIEmbeddings (gemini-embedding-001)
        |  converts each chunk into a vector
        v
Chroma Vector Database (stored locally in ./chroma_db/)
        |
        v
Retriever Tool (@tool: search_knowledge_base)
        |
        v
Agent Tools List  <--- also has search_news (Tavily)
        |
        v
LangGraph Agent (Gemini 2.0 Flash)
        |
        v
ReAct Loop decides: web search? or local knowledge?
```

---

## Why We Switched from Groq to Gemini

In previous lessons the agent used **Groq** (`llama-3.3-70b-versatile`) as its LLM. In this lesson we switch to **Gemini 2.0 Flash** (`gemini-2.0-flash`).

The reason is tool calling reliability. Groq's general-purpose models occasionally generate malformed tool call syntax, which causes a `400 BadRequestError` from the API. This becomes a real problem once the agent has multiple tools to choose between. Gemini's tool calling is consistent and well-tested with LangChain, making it the right choice as our agent grows more complex.

The switch is a one-line change in the code:

```python
# Before (Groq)
from langchain_groq import ChatGroq
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# After (Gemini)
from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
```

Everything else — the graph, the tools, the ReAct loop — stays exactly the same.

---

## The New Tool: search_knowledge_base

The tool itself is simple — it takes a query, searches Chroma, and returns the relevant chunks as text. The agent reads those chunks and forms its answer.

```python
@tool
def search_knowledge_base(query: str) -> str:
    """Search the local knowledge base for information from stored documents.
    Use this when the user asks about topics that may be covered in our stored articles,
    such as Kenya news, local events, or documents we have saved.
    Do NOT use this for general web searches or breaking news — use search_news for that.
    """
    results = retriever.invoke(query)
    if not results:
        return "No relevant information found in the knowledge base."
    return "\n\n".join([doc.page_content for doc in results])
```

Notice the docstring carefully. It tells the agent:
- When to use this tool (local stored documents)
- When NOT to use it (live web search)

The agent reads docstrings to decide which tool to call. Clear, specific docstrings are what make the routing work correctly.

---

## What the Agent Can Now Do

After today's lesson, when a user asks a question the agent will reason through its options:

| Question | Agent Decision |
|----------|----------------|
| "What happened in the Kenya floods?" | Uses `search_knowledge_base` — this is in our stored article |
| "What is the news in Kenya today?" | Uses `search_news` — needs live, current web results |
| "How many died in the Nairobi disaster?" | Uses `search_knowledge_base` — stored article has this |
| "What time is it?" | Uses `get_current_datetime` |
| "Summarise what you know about SHA" | Uses `search_knowledge_base` — SHA content is in our stored file |

The same ReAct loop from previous lessons handles all of this automatically. We are simply adding a new tool — the loop decides when to use it.

---

## Key Takeaways

- A **vector database** stores text as numbers (vectors) and searches by meaning, not keywords
- **Embeddings** are the conversion step — text goes in, a vector comes out
- **Chunking** splits documents before embedding so retrieval is specific, not blurry
- `chunk_size` controls how big each piece is; `chunk_overlap` prevents information loss at boundaries
- We use **Chroma** as our local vector database — free, local, persistent
- We use **Google Gemini embeddings** — free API, high quality, one import
- We use **Gemini 2.0 Flash** as the agent LLM — reliable tool calling, same API key as embeddings
- The agent gains a `search_knowledge_base` tool that sits alongside `search_news`
- The agent decides between tools automatically based on the question and the tool docstrings

---

## Coming Up Next

In Lesson 2.3 we go deeper into **agent memory** — giving the agent the ability to remember what was discussed earlier in a session (short-term memory) and persist knowledge across sessions (long-term memory via Chroma). The tools we build today feed directly into that lesson.

---