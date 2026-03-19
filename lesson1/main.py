# Read documentation carefully before running this code
# If possible try not to copy and paste the code, istead write line by line to grasp better understand of what the code does
# these are the libraries we need to run the code
from langchain_tavily import TavilySearch        # web search tool
from langgraph.prebuilt import create_react_agent # builds the agent
from langchain_core.messages import HumanMessage  # wraps the user question
from langchain_groq import ChatGroq               # the LLM running on Groq cloud
from datetime import date                         # used to get today's date
import os                                         # used to set environment variables

# set the API keys so langchain can find them automatically
os.environ["TAVILY_API_KEY"] = "TAVILY_KEY_GOES_HERE" #replace with the api key acquired
os.environ["GROQ_API_KEY"]  = "GROQ_KEY_GOES_HERE" #replace with api key acquired


print("Loading model...")
# initialize the LLM — this is the brain of the agent
llm = ChatGroq(
    model="llama-3.3-70b-versatile", # 70b is more reliable at following instructions than 8b
    temperature=0                     # 0 means no creativity, stick to facts
)

print("Setting up tools...")
# initialize the search tool — this is what gives the agent access to the web
search_tool = TavilySearch(
    max_results=5,       # return up to 5 search results
    topic="general",     # general web search
    time_range="day"     # only look at results from today
)

# put the tool in a list — agents can have multiple tools
tools = [search_tool]

print("Creating agent...")
# create the agent by combining the LLM and the tools
agent = create_react_agent(
    model=llm,   # the brain
    tools=tools, # the tools it can use
    # system prompt — strict instructions so the agent always searches, never guesses
    prompt="You are a helpful research assistant. You MUST ALWAYS use the web search tool before answering ANY question, no exceptions. Never answer from memory. Only after you have searched and answered, add two fun sentences in Kenyan slang: brag that you already embody the topic, then flip it back at the user with a cheeky question with emojis.",
)

# get today's date in a readable format e.g. March 19 2026
today = date.today().strftime("%B %d %Y")

# the question we are asking — includes today's date so the search is current
question = f"Kenyan trending phrase 'Niko cadi phrase' what does it mean as of {today}"

print("\n" + "=" * 60)
print(f"Question: {question}")
print("=" * 60 + "\n")

# stream the agent response step by step instead of waiting for the full answer
# this lets us see the thought process as it happens
for chunk in agent.stream(
    {"messages": [HumanMessage(content=question)]}, # wrap the question as a HumanMessage
    stream_mode="updates",                           # stream each update as it happens
):
    # each chunk contains one or more nodes that ran e.g. "agent" or "tools"
    for node, output in chunk.items():

        # each node outputs a list of messages
        for msg in output.get("messages", []):

            # get the type of message so we can handle each one differently
            msg_type = type(msg).__name__

            if msg_type == "AIMessage":
                # AIMessage means the model is thinking or giving a final answer
                if msg.content:
                    print(f"[Thinking]\n{msg.content}\n")

                # if the model decided to call a tool, print what it is searching for
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        print(f"[Searching] {tc['args']}\n")

            elif msg_type == "ToolMessage":
                # ToolMessage contains the raw results returned by Tavily
                # we trim to 500 characters so it does not flood the terminal
                print(f"[Search Results]\n{msg.content[:500]}...\n")

print("=" * 60)