import os
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.document_loaders import ArxivLoader
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.messages import HumanMessage, AnyMessage, SystemMessage
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langchain.tools.retriever import create_retriever_tool
from supabase.client import Client, create_client
from typing import TypedDict, List, Dict, Any, Optional, Annotated

@tool
def add_numbers(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b

class AgentState(TypedDict):
    input_file: Optional[str]
    messages: Annotated[list[AnyMessage], add_messages]

sys_prompt = """You are a helpful assistant tasked with answering questions using a set of tools.
Now, I will ask you a question. Report your thoughts, and finish your answer with the following template:
FINAL ANSWER: [YOUR FINAL ANSWER].
YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string.
Your answer should only start with "FINAL ANSWER: ", then follows with the answer. """

sys_msg = SystemMessage(content=sys_prompt)

seed_state: AgentState = {
    "input_file": None,
    "messages": [sys_msg]
}


tools = [
    add_numbers
]

def build_graph():
    llm = ChatOpenAI(model_name="gpt-4o")
    llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)

    def assistant(state: AgentState):
        return {
            "messages": [llm_with_tools.invoke(state["messages"])]
        }

    builder = StateGraph(AgentState)

    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")

    return builder.compile()


def main():
    agent = build_graph()
    state = seed_state.copy()
    state["messages"].append(HumanMessage(content="What is 5.5 + 3.2?"))
    result = agent.invoke(state)
    print("\nAll messages in the conversation:")
    for msg in result["messages"]:
        print(f"\n{msg.type}: {msg.content}")


if __name__ == "__main__":
    main()



