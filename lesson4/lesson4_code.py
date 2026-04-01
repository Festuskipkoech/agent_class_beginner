# Lesson: Vector Databases, Embeddings & Chunking
# Agent with both live web search (Tavily) and local knowledge base (Chroma).
#
# Tools available:
#   - get_current_datetime
#   - estimate_reading_time
#   - search_news           (Tavily — live web)
#   - search_knowledge_base (Chroma — local documents)

import os
from typing import Annotated
from typing_extensions import TypedDict
from datetime import datetime
from dotenv import load_dotenv

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


# STEP 1 — Load and chunk the knowledge base document
# chunk_size=500 : each chunk is roughly 500 characters
# chunk_overlap=100 : overlap so sentences at boundaries keep context

print("Loading knowledge base...")

with open("news_article.txt", "r") as f:
    raw_text = f.read()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.create_documents([raw_text])
print(f"Document split into {len(chunks)} chunks.")


# STEP 2 — Google Gemini Embeddings
# Reads GOOGLE_API_KEY from .env automatically.

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)


# STEP 3 — Create or load the Chroma vector store
# First run: embeds all chunks and saves to ./chroma_db/
# Subsequent runs: loads directly from disk.

persist_directory = "./chroma_db"

if not os.path.exists(persist_directory):
    print("Folder './chroma_db' not found. Creating and saving vectors...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
else:
    print("Chroma DB already exists. Loading from disk.")
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

# k=3: return the 3 most relevant chunks per query
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

print("Knowledge base ready.\n")


# STEP 4 — Tool definitions

@tool
def get_current_datetime() -> str:
    """Returns the current date and time. Only call this when the user explicitly asks for the date or time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def estimate_reading_time(text: str) -> str:
    """Estimate reading time in minutes based on the length of a piece of text."""
    word_count = len(text.split())
    minutes = max(1, round(word_count / 200))
    return f"{minutes} minute(s)"


tavily = TavilySearch(max_results=5, topic="general", time_range="day")

@tool
def search_news(query: str) -> str:
    """Search for recent news articles on the internet. Only call this when the user asks
    about current events, breaking news, or information that may have changed today.
    Do NOT use this for questions that can be answered from our stored local documents.
    """
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
    Use this when the user asks about topics that may be covered in our stored articles,
    such as Kenya news, local events, or documents we have saved.
    Do NOT use this for general web searches or breaking news — use search_news for that.
    """
    results = retriever.invoke(query)
    if not results:
        return "No relevant information found in the knowledge base."
    return "\n\n".join([doc.page_content for doc in results])


# STEP 5 — LLM setup
# We use Gemini 2.5 Flash as the agent brain.
# Groq's general-purpose models have a known issue generating malformed tool call
# syntax which causes 400 errors. Gemini's tool calling is reliable and uses the
# same GOOGLE_API_KEY already in your .env for embeddings — no extra key needed.

tools = [get_current_datetime, estimate_reading_time, search_news, search_knowledge_base]
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
llm_with_tools = llm.bind_tools(tools)


# STEP 6 — Graph definition

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


# STEP 7 — Conversation loop

system_prompt = SystemMessage(
    content=(
        "You are a helpful news assistant with access to both a local knowledge base and live web search. "
        "Use search_knowledge_base for questions about stored articles (Kenya news, floods, politics, SHA, Russia recruitment). "
        "Use search_news for questions about today's breaking news or events not covered in the knowledge base. "
        "For greetings and simple questions, respond directly without calling any tools."
    )
)

conversation_history = [system_prompt]

print("Agent is running. Type 'exit' to stop.")
print("Try asking about the Kenya floods, Russia recruitment, or today's news.\n")

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