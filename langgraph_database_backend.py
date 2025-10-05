from langchain_openai import ChatOpenAI  # official replacement
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated
from openai import OpenAI
from dotenv import load_dotenv
import sqlite3
import os

load_dotenv()

# Initialize LLM
llm = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENAI_API_KEY"]  # Must match secret name
)

class ChatState(TypedDict):
    messages: Annotated[list, add_messages]  # List of BaseMessage

def chat_node(state: ChatState):
    messages = state['messages']
    response = llm(messages)  # use __call__ to get response
    return {"messages": messages + [response]}

# SQLite connection & checkpointer
conn = sqlite3.connect('chatbot.db', check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# Graph setup
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

# Retrieve all threads
def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])
    return list(all_threads)
