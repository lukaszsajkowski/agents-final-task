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

@tool
def multiply_numbers(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b

@tool
def subtract_numbers(a: float, b: float) -> float:
    """Subtract second number from first number."""
    return a - b

@tool
def divide_numbers(a: float, b: float) -> float:
    """Divide first number by second number."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

@tool
def modulus_numbers(a: float, b: float) -> float:
    """Get the remainder when first number is divided by second number."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a % b

@tool
def search_wikipedia(query: str) -> str:
    """Search for information on Wikipedia. Input should be a search query."""
    try:
        loader = WikipediaLoader(query=query, load_max_docs=1)
        docs = loader.load()
        if not docs:
            return "No information found on Wikipedia for this query."
        return docs[0].page_content[:1000]  # Return first 1000 chars to keep response concise
    except Exception as e:
        return f"Error searching Wikipedia: {str(e)}"

@tool
def search_web(query: str) -> str:
    """Search the web for current information. Input should be a search query."""
    try:
        search = TavilySearchResults(api_key=os.getenv("TAVILY_API_KEY"))
        results = search.invoke(query)
        if not results:
            return "No results found for this query."
        
        # Format the top 3 results
        formatted_results = []
        for i, result in enumerate(results[:3], 1):
            formatted_results.append(f"Result {i}:\nTitle: {result['title']}\nContent: {result['content']}\nURL: {result['url']}\n")
        
        return "\n".join(formatted_results)
    except Exception as e:
        return f"Error searching the web: {str(e)}"

@tool
def search_arxiv(query: str) -> str:
    """Search for academic papers on arXiv. Input should be a search query."""
    try:
        loader = ArxivLoader(query=query, load_max_docs=2)
        docs = loader.load()
        if not docs:
            return "No papers found on arXiv for this query."
        
        # Format the results
        formatted_results = []
        for i, doc in enumerate(docs, 1):
            metadata = doc.metadata
            formatted_results.append(
                f"Paper {i}:\n"
                f"Title: {metadata.get('title', 'N/A')}\n"
                f"Authors: {metadata.get('authors', 'N/A')}\n"
                f"Published: {metadata.get('published', 'N/A')}\n"
                f"Abstract: {doc.page_content[:500]}...\n"  # First 500 chars of abstract
                f"URL: {metadata.get('entry_id', 'N/A')}\n"
            )
        
        return "\n".join(formatted_results)
    except Exception as e:
        return f"Error searching arXiv: {str(e)}"

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
    add_numbers,
    multiply_numbers,
    subtract_numbers,
    divide_numbers,
    modulus_numbers,
    search_wikipedia,
    search_web,
    search_arxiv
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
    state["messages"].append(HumanMessage(content="Find recent papers about large language models"))
    result = agent.invoke(state)
    print("\nAll messages in the conversation:")
    for msg in result["messages"]:
        print(f"\n{msg.type}: {msg.content}")


if __name__ == "__main__":
    main()



