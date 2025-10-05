from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize OpenRouter client
llm = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state['messages']
    
    # Convert to OpenAI-style messages
    chat_messages = [
        {"role": "user", "content": m.content} for m in messages if hasattr(m, "content")
    ]
    
    response = llm.chat.completions.create(
        model="gpt-4o-mini",  # or any model you want from OpenRouter
        messages=chat_messages
    )
    
    # Extract text from response
    reply_text = response.choices[0].message.content
    return {"messages": messages + [HumanMessage(content=reply_text)]}

# Checkpointer
checkpointer = InMemorySaver()

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)
