from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# ---------------------------
# 1. Initialize OpenRouter LLM
# ---------------------------
llm = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"]  # Must match secret name in Streamlit secrets
)

# ---------------------------
# 2. Chat State
# ---------------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# ---------------------------
# 3. Chat Node
# ---------------------------
def chat_node(state: ChatState):
    messages = state['messages']
    
    # Convert to OpenAI/OpenRouter format
    chat_messages = []
    for m in messages:
        if hasattr(m, "content"):
            role = "user" if isinstance(m, HumanMessage) else "assistant"
            chat_messages.append({"role": role, "content": m.content})
    
    # Call OpenRouter
    response = llm.chat.completions.create(
        model="gpt-4o-mini",
        messages=chat_messages
    )
    
    # Extract assistant reply
    reply_text = response.choices[0].message.content
    
    # Append as AIMessage
    messages.append(AIMessage(content=reply_text))
    
    return {"messages": messages}

# ---------------------------
# 4. Checkpointer
# ---------------------------
checkpointer = InMemorySaver()

# ---------------------------
# 5. Graph
# ---------------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)
