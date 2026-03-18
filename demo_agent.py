# What this demo does:
#   - Creates a simple AI agent that can search the web to answer questions
#   - Uses Llama 3.2 running locally via Ollama (no OpenAI API costs!)
#   - Uses Tavily to search the web in real time
#   - Uses the ReAct pattern: the agent Reasons, Acts (uses a tool), then Observes the result
#
# Prerequisites:
#   1. Ollama running locally with Llama 3.2 pulled:
#        ollama pull llama3.2
#   2. A free Tavily API key from https://tavily.com
#   3. A .env file in the same folder with:
#        TAVILY_API_KEY=your_key_here
#   4. Install dependencies:
#        pip install langchain langchain-community langchain-ollama tavily-python python-dotenv
#

import os

from langchain_ollama import ChatOllama                                    # connects LangChain to our local Ollama model
from langchain_community.tools.tavily_search import TavilySearchResults   # web search tool
from langchain.agents import create_react_agent, AgentExecutor             # agent builder + runner
from langchain import hub                                                   # pulls pre-built prompts from LangChain hub

# --- API Key ------------------------------------------------------------------
#
# Replace PASTE_YOUR_TAVILY_KEY_HERE with your actual Tavily API key
# Get a free key at https://tavily.com — free tier gives 1,000 searches/month
#
# NOTE: Never share this file publicly with your real key inside it!

TAVILY_API_KEY = "PASTE_YOUR_TAVILY_KEY_HERE"


# We use ChatOllama to connect to our locally running Llama 3.2 model.
# This means NO API costs and NO internet required for the LLM itself.
#
# model="llama3.2" must match the model name you pulled with: ollama pull llama3.2
# temperature=0 means the model will be more focused and consistent (less random)
# — good for agents that need to reason clearly and follow instructions

llm = ChatOllama(
    model="llama3.2",
    temperature=0,
)

# Tools are what give the agent abilities beyond just generating text.
# Without tools, the agent can only use what it already knows (its training data).
# With Tavily, the agent can search the web for real-time, up-to-date information.
#
# max_results=3 means Tavily returns the top 3 search results.
# The agent reads those results and uses them to form its answer.

search_tool = TavilySearchResults(max_results=3)

# Put all tools into a list — later we can add more tools here (e.g. a calculator,
# a database retriever, a file reader) and the agent will know how to use them all
tools = [search_tool]

# The ReAct prompt is a special system prompt that teaches the agent HOW to think.
# It instructs the agent to follow this loop:
#
#   Thought:  "What do I need to do?"
#   Action:   "I will use [tool] with [input]"
#   Observation: [tool result]
#   Thought:  "What did I learn? Do I have enough to answer?"
#   ... repeat until ready ...
#   Final Answer: "Here is the answer"
#
# We pull a pre-built ReAct prompt from LangChain's hub so we don't have to write it from scratch.
# This is the standard prompt used for ReAct agents — battle-tested and reliable.

prompt = hub.pull("hwchase17/react")

#
# create_react_agent() wires together three things:
#   - the LLM (Llama 3.2) — the brain that reasons and decides
#   - the tools (Tavily search) — the abilities the agent can use
#   - the prompt (ReAct) — the instructions that teach it HOW to reason
#
# Think of this like hiring someone (LLM), giving them tools (Tavily),
# and handing them a process manual (ReAct prompt) to follow.

agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)


# The AgentExecutor is the runtime that actually RUNS the agent loop.
# It handles:
#   - calling the LLM to get the next thought/action
#   - executing the chosen tool
#   - feeding the tool result back to the LLM
#   - repeating until the agent reaches a Final Answer
#
# verbose=True prints each step of the reasoning so students can SEE the agent thinking.
# This is really valuable for learning — turn it off in production.
#
# max_iterations=5 is a safety limit — prevents the agent from looping forever
# if it gets confused. In production you'd tune this based on task complexity.

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,       # show the agent's thinking step by step
    max_iterations=5,   # safety limit on reasoning steps
)


# We pass a question to the agent and it will:
#   1. Decide it needs to search the web
#   2. Call Tavily with a search query
#   3. Read the results
#   4. Form a final answer
#
# Try changing the question to anything current — the agent will search for it!

question = "What are the most exciting AI agent frameworks developers are using in 2026?"

print("\n" + "="*60)
print(f"Question: {question}")
print("="*60 + "\n")

# invoke() runs the agent and returns the final answer
# The input dict matches what the ReAct prompt expects: {"input": "your question"}
response = agent_executor.invoke({"input": question})

print("\n" + "="*60)
print("Final Answer:")
print(response["output"])
print("="*60 + "\n")

# What to try next:
#   - Change the question above to something else and observe the agent's reasoning
#   - Add a second tool (e.g. a calculator) and watch the agent choose between them
#   - Set verbose=False to see only the final answer (cleaner output)
#   - In the next lesson we will add a vector database as a second tool so the
#     agent can choose between searching the web OR querying a local knowledge base
