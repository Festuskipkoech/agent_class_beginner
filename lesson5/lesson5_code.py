# Lesson 5: Agent Memory — Remembering Users Across Sessions
# This lesson introduces three types of memory:
#   - In-context memory    (Type 1) — already built, just named
#   - Persistent memory    (Type 2) — built today using Chroma
#   - Summary memory       (Type 3) — discussed conceptually
#
# The agent now:
#   - Remembers facts about the user across sessions (name, interests, preferences)
#   - Searches the Kenya news knowledge base (from Lesson 4)
#   - Searches the live web via Tavily
#   - Saves and retrieves memories using two dedicated tools

# pip install pillow
import os
import uuid
from typing import Annotated
from typing_extensions import TypedDict
from datetime import datetime
from dotenv import load_dotenv
from PIL import Image
import io

load_dotenv()

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langchain_tavily import TavilySearch

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter


# STEP 1 — Load the Kenya news knowledge base (same as Lesson 4)

print("Loading Kenya news knowledge base...")

with open("news_article.txt", "r") as f:
    raw_text = f.read()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.create_documents([raw_text])
print(f"News article split into {len(chunks)} chunks.")


# STEP 2 — Embeddings (same model, same key, used for both stores)

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")


# STEP 3 — News knowledge base vector store (from Lesson 4)

news_db_path = "./chroma_db"

if not os.path.exists(news_db_path):
    print("Creating news knowledge base...")
    news_vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=news_db_path
    )
else:
    print("News knowledge base found. Loading from disk.")
    news_vectorstore = Chroma(
        persist_directory=news_db_path,
        embedding_function=embeddings
    )

news_retriever = news_vectorstore.as_retriever(search_kwargs={"k": 3})


# STEP 4 — Memory vector store (NEW in Lesson 5)
# This is a separate Chroma collection that stores facts about the user.
# Each memory is a short string: "The user's name is James."
# It lives in ./memory_db/ so it persists between sessions.

memory_db_path = "./memory_db"

if not os.path.exists(memory_db_path):
    print("Creating memory store (first run — no memories yet).")
    memory_vectorstore = Chroma(
        collection_name="user_memory",
        embedding_function=embeddings,
        persist_directory=memory_db_path
    )
else:
    print("Memory store found. Loading existing memories.")
    memory_vectorstore = Chroma(
        collection_name="user_memory",
        embedding_function=embeddings,
        persist_directory=memory_db_path
    )

memory_retriever = memory_vectorstore.as_retriever(search_kwargs={"k": 5})

print("All stores ready.\n")


# STEP 5 — Tool definitions

@tool
def get_current_datetime() -> str:
    """Returns the current date and time. Only call this when the user explicitly asks for the date or time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def search_news(query: str) -> str:
    """Search for recent news articles on the internet. Only call this when the user asks
    about current events, breaking news, or information that may have changed today.
    Do NOT use this for questions that can be answered from our stored local documents.
    """
    tavily = TavilySearch(max_results=5, topic="general", time_range="day")
    dated_query = f"{query} {datetime.now().strftime('%Y-%m-%d')}"
    results = tavily.invoke(dated_query)
    if not results:
        return "No results found."
    articles = results.get("results", [])
    output = []
    for r in articles:
        output.append(
            f"Title: {r.get('title', 'N/A')}\n"
            f"URL: {r['url']}\n"
            f"Summary: {r.get('content', '')[:300]}"
        )
    return "\n\n".join(output)


@tool
def search_knowledge_base(query: str) -> str:
    """Search the local knowledge base for information from stored documents.
    Use this for questions about Kenya news, floods, politics, SHA, Russia recruitment.
    Do NOT use this for general web searches or breaking news.
    """
    results = news_retriever.invoke(query)
    if not results:
        return "No relevant information found in the knowledge base."
    return "\n\n".join([doc.page_content for doc in results])


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


@tool
def search_memory(query: str) -> str:
    """Search long-term memory for facts previously saved about the user.
    Call this at the start of every conversation to recall who you are speaking with
    and what you already know about them. Also call this any time the user references
    something personal that you should already know — name, preferences, past topics.
    """
    results = memory_retriever.invoke(query)
    if not results:
        return "No relevant memories found."
    memories = [doc.page_content for doc in results]
    return "What I remember:\n" + "\n".join(f"- {m}" for m in memories)


# STEP 6 — LLM and graph setup

tools = [get_current_datetime, search_news, search_knowledge_base, save_memory, search_memory]
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
llm_with_tools = llm.bind_tools(tools)


class State(TypedDict):
    messages: Annotated[list, add_messages]


def agent(state: State):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


graph_builder = StateGraph(State)
graph_builder.add_node("agent", agent)
graph_builder.add_node("tools", ToolNode(tools))
graph_builder.add_edge(START, "agent")
graph_builder.add_conditional_edges("agent", tools_condition)
graph_builder.add_edge("tools", "agent")
graph = graph_builder.compile()

image_data = graph.get_graph().draw_mermaid_png()
image = Image.open(io.BytesIO(image_data))
image.show()

# STEP 7 — Conversation loop

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

conversation_history = [system_prompt]

print("Agent is running. Type 'exit' to quit.")
print("The agent will remember you across sessions.")
print("Try: telling it your name, asking about Kenya floods, then restarting and seeing if it remembers you.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    conversation_history.append({"role": "user", "content": user_input})
    result = graph.invoke({"messages": conversation_history})
    conversation_history = result["messages"]

    assistant_message = result["messages"][-1]
    print(f"Assistant: {assistant_message.content}\n")