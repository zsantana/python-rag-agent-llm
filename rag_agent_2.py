import os
from typing import List
from dotenv import load_dotenv

import streamlit as st
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langgraph.graph import START, END, StateGraph
from langgraph.types import Send
from tavily import TavilyClient

from schemas import ReportState, QueryResult
from prompts import build_queries, resume_search, build_final_response

# Load environment variables
load_dotenv()

# Initialize language models
llm = ChatOllama(model="llama3.1:8b")
reasoning_llm = ChatOllama(model="deepseek-r1:8b")

# Get API Key
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

def build_first_queries(state: ReportState):
    """Generates initial search queries based on user input."""
    class QueryList(BaseModel):
        queries: List[str]

    prompt = build_queries.format(user_input=state.user_input)
    query_llm = llm.with_structured_output(QueryList)
    result = query_llm.invoke(prompt)

    if not result:
        print("Error: LLM returned None")
        return {"queries": []}

    return {"queries": result.queries}

def spawn_researchers(state: ReportState):
    """Creates research tasks for each query."""
    return [Send("single_search", query) for query in state.queries]

def single_search(query: str):
    """Performs a search using Tavily API and summarizes results."""
    if not TAVILY_API_KEY:
        print("Error: Tavily API key is missing or invalid.")
        return None

    tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
    try:
        results = tavily_client.search(query=query, max_results=1, include_answer=True)
        if not results.get("results"):
            print("Error: No results found.")
            return None

        url = results["results"][0]["url"]
        extraction = tavily_client.extract(url)
        raw_content = extraction["results"][0]["raw_content"] if extraction["results"] else ""

        prompt = resume_search.format(user_input=query, search_results=raw_content)
        llm_result = llm.invoke(prompt)
        query_result = QueryResult(title=results["results"][0]["title"], url=url, resume=llm_result.content)

        return {"query_results": [query_result]}
    except Exception as e:
        print(f"Error during search: {e}")
        return None

def final_writer(state: ReportState):
    """Generates the final response based on search results."""
    search_results = "\n".join(
        f"[{i+1}]\nTitle: {res.title}\nURL: {res.url}\nContent: {res.resume}\n====================="
        for i, res in enumerate(state.queries_results)
    )
    references = "\n".join(f"[{i+1}] - [{res.title}]({res.url})" for i, res in enumerate(state.queries_results))

    prompt = build_final_response.format(user_input=state.user_input, search_results=search_results)
    llm_result = reasoning_llm.invoke(prompt)
    return {"final_response": f"{llm_result.content}\n\nReferences:\n{references}"}

# Build StateGraph
builder = StateGraph(ReportState)
builder.add_node("build_first_queries", build_first_queries)
builder.add_node("single_search", single_search)
builder.add_node("final_writer", final_writer)

builder.add_edge(START, "build_first_queries")
builder.add_conditional_edges("build_first_queries", spawn_researchers, ["single_search"])
builder.add_edge("single_search", "final_writer")
builder.add_edge("final_writer", END)

graph = builder.compile()

if __name__ == "__main__":
    st.title("Custom Local Perplexity")
    user_input = st.text_input("What do you want to know?", value="What is the process of building an LLM?")

    if st.button("Search"):
        with st.status("Generating answer"):
            for output in graph.stream({"user_input": user_input}, stream_mode="debug"):
                if output["type"] == "task_result":
                    st.write(f"Running {output['payload']['name']}")
                    st.write(output)
        
        response = output["payload"]["result"][0][1]
        think_str, final_response = response.split("</think>")

        with st.expander("🧠 Reflection", expanded=False):
            st.write(think_str)
        st.write(final_response)
    
    # Run initial query search and print debugging info
    result = graph.invoke({"user_input": user_input})
    queries = result.get("queries", [])

    for idx in [1, 2, 6]:
        if idx < len(queries):
            print(f"Query at index [{idx}]: {queries[idx]}")
        else:
            print(f"Not enough queries to access index [{idx}]")

    print("Number of queries:", len(queries))

    if queries:
        search_results = single_search(queries[0])
        print("Final Results:", search_results)
