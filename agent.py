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
import requests
import os.path
import pandas as pd
import json
import openpyxl

@tool
def add_numbers(a: float, b: float) -> float:
    """Add two numbers together.
    Args:
        a (float): First number to add
        b (float): Second number to add
    Returns:
        float: The sum of a and b
    """
    return a + b

@tool
def multiply_numbers(a: float, b: float) -> float:
    """Multiply two numbers together.
    Args:
        a (float): First number to multiply
        b (float): Second number to multiply
    Returns:
        float: The product of a and b
    """
    return a * b

@tool
def subtract_numbers(a: float, b: float) -> float:
    """Subtract second number from first number.
    Args:
        a (float): Number to subtract from
        b (float): Number to subtract
    Returns:
        float: The difference between a and b
    """
    return a - b

@tool
def divide_numbers(a: float, b: float) -> float:
    """Divide first number by second number.
    Args:
        a (float): Number to divide
        b (float): Number to divide by
    Returns:
        float: The quotient of a divided by b
    Raises:
        ValueError: If b is zero
    """
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

@tool
def modulus_numbers(a: float, b: float) -> float:
    """Get the remainder when first number is divided by second number.
    Args:
        a (float): Number to divide
        b (float): Number to divide by
    Returns:
        float: The remainder when a is divided by b
    Raises:
        ValueError: If b is zero
    """
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a % b

@tool
def search_wikipedia(query: str) -> str:
    """Search for information on Wikipedia.
    Args:
        query (str): The search query to look up on Wikipedia
    Returns:
        str: The first 1000 characters of the Wikipedia article content, or an error message if not found
    """
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
    """Search the web for current information.
    Args:
        query (str): The search query to look up on the web
    Returns:
        str: Formatted results from the top 3 search results, including title, content, and URL
    """
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
    """Search for academic papers on arXiv.
    Args:
        query (str): The search query to look up on arXiv
    Returns:
        str: Formatted results from up to 2 papers, including title, authors, publication date, abstract, and URL
    """
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

@tool
def download_file(url: str, save_path: Optional[str] = None) -> str:
    """Download a file from a URL.
    Args:
        url (str): The URL of the file to download
        save_path (str, optional): Where to save the file. If not provided, uses the filename from the URL
    Returns:
        str: Success message with the save location or an error message if something goes wrong
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # If no save_path is provided, use the filename from the URL
        if save_path is None:
            save_path = os.path.basename(url)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        
        # Download the file
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        return f"File successfully downloaded to: {save_path}"
    except requests.exceptions.RequestException as e:
        return f"Error downloading file: {str(e)}"
    except Exception as e:
        return f"Error saving file: {str(e)}"

@tool
def analyze_csv(file_path: str) -> str:
    """Analyze a CSV file and return basic statistics and information.
    Args:
        file_path (str): Path to the CSV file to analyze
    Returns:
        str: A JSON string containing file information including:
            - Number of rows and columns
            - Column names and their data types
            - Basic statistics for numeric columns
            - Sample of first 5 rows
            - Missing value counts per column
    """
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Basic information
        info = {
            "rows": len(df),
            "columns": len(df.columns),
            "column_info": {},
            "sample_data": df.head().to_dict(orient='records'),
            "missing_values": df.isnull().sum().to_dict()
        }
        
        # Get detailed column information
        for column in df.columns:
            col_info = {
                "dtype": str(df[column].dtype),
            }
            
            # Add statistics for numeric columns
            if pd.api.types.is_numeric_dtype(df[column]):
                col_info.update({
                    "mean": float(df[column].mean()),
                    "std": float(df[column].std()),
                    "min": float(df[column].min()),
                    "max": float(df[column].max()),
                    "median": float(df[column].median())
                })
            # Add value counts for categorical columns
            elif pd.api.types.is_categorical_dtype(df[column]) or pd.api.types.is_object_dtype(df[column]):
                col_info["unique_values"] = int(df[column].nunique())
                col_info["most_common"] = df[column].value_counts().head(3).to_dict()
            
            info["column_info"][column] = col_info
        
        return json.dumps(info, indent=2)
    except Exception as e:
        return f"Error analyzing CSV file: {str(e)}"

@tool
def analyze_excel(file_path: str, sheet_name: Optional[str] = None) -> str:
    """Analyze an Excel file and return basic statistics and information.
    Args:
        file_path (str): Path to the Excel file to analyze
        sheet_name (str, optional): Name of the sheet to analyze. If not provided, analyzes the first sheet
    Returns:
        str: A JSON string containing file information including:
            - List of all sheets
            - For the analyzed sheet:
                - Number of rows and columns
                - Column names and their data types
                - Basic statistics for numeric columns
                - Sample of first 5 rows
                - Missing value counts per column
                - Excel-specific information (merged cells, formulas, etc.)
    """
    try:
        # Get Excel file information
        wb = openpyxl.load_workbook(file_path, data_only=True)
        sheet_names = wb.sheetnames
        
        # If no sheet specified, use the first one
        if sheet_name is None:
            sheet_name = sheet_names[0]
        elif sheet_name not in sheet_names:
            return f"Error: Sheet '{sheet_name}' not found. Available sheets: {', '.join(sheet_names)}"
        
        # Read the specified sheet
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        
        # Get Excel-specific information
        ws = wb[sheet_name]
        merged_cells = [str(cell) for cell in ws.merged_cells.ranges]
        
        # Basic information
        info = {
            "file_info": {
                "total_sheets": len(sheet_names),
                "sheet_names": sheet_names,
                "current_sheet": sheet_name
            },
            "sheet_info": {
                "rows": len(df),
                "columns": len(df.columns),
                "column_info": {},
                "sample_data": df.head().to_dict(orient='records'),
                "missing_values": df.isnull().sum().to_dict(),
                "excel_specific": {
                    "merged_cells": merged_cells,
                    "has_formulas": any(cell.data_type == 'f' for row in ws.rows for cell in row)
                }
            }
        }
        
        # Get detailed column information
        for column in df.columns:
            col_info = {
                "dtype": str(df[column].dtype),
            }
            
            # Add statistics for numeric columns
            if pd.api.types.is_numeric_dtype(df[column]):
                col_info.update({
                    "mean": float(df[column].mean()),
                    "std": float(df[column].std()),
                    "min": float(df[column].min()),
                    "max": float(df[column].max()),
                    "median": float(df[column].median())
                })
            # Add value counts for categorical columns
            elif pd.api.types.is_categorical_dtype(df[column]) or pd.api.types.is_object_dtype(df[column]):
                col_info["unique_values"] = int(df[column].nunique())
                col_info["most_common"] = df[column].value_counts().head(3).to_dict()
            
            info["sheet_info"]["column_info"][column] = col_info
        
        return json.dumps(info, indent=2)
    except Exception as e:
        return f"Error analyzing Excel file: {str(e)}"

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
    search_arxiv,
    download_file,
    analyze_csv,
    analyze_excel
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



