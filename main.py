from dotenv import load_dotenv
from typing import Annotated, List
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from operation import serp_search,reddit_search_api,reddit_post_retrieval
from prompts import (
    get_google_analysis_messages, 
    get_reddit_analysis_messages, 
    get_synthesis_messages,
    create_message_pair,
    get_reddit_url_analysis_messages)

load_dotenv()
chat_model = init_chat_model("gpt-4o-mini")

class State(TypedDict):
    messages: Annotated[list, add_messages]
    user_input: str | None
    google_results: str | None
    reddit_results: str | None
    selected_reddit_urls: list[str] | None
    reddit_post_data: List| None
    google_analysis: str | None
    reddit_analysis: str | None
    final_analysis: str | None

class RedditUrlAnalysis(BaseModel):
    selected_urls: List[str] =Field(description="List of selected Reddit URLs for further analysis")

def google_search(state:State):
    user_question = state.get("user_input", "")
    print("Searching Google")
    google_results = serp_search(user_question, engine="google")
    print(google_results)
    return {"google_results": google_results}

def reddit_search(state:State):
    user_question = state.get("user_input", "")
    print("Searching Reddit")
    reddit_results = reddit_search_api(user_question)
    print(reddit_results)
    return {"reddit_results": reddit_results}

def analyze_reddit_posts(state:State):
    user_question = state.get("user_input", "")
    reddit_results = state.get("reddit_results","")
    if not reddit_results:
        return {"reddit_analysis": []}
    
    structured_data = chat_model.with_structured_output(RedditUrlAnalysis)
    messages = get_reddit_url_analysis_messages(user_question,reddit_results)

    try:
        analysis = structured_data.invoke(messages)
        selected_urls = analysis.selected_urls
        print("Selected URLs for further analysis:")
        for i, url in enumerate(selected_urls,1):
            print(f"{i}. {url}")
    except Exception as e:
        print(e)
        selected_urls = []
    return  {"selected_reddit_urls": selected_urls}

def retreive_reddit_posts(state:State):
    print("Retrieving Reddit posts")
    selected_urls = state.get("selected_reddit_urls",[])

    if not selected_urls:
        return {"reddit_post_data": []}
    print(f"Processing {len(selected_urls)} URLs")
    print(selected_urls)
    reddit_post_data = reddit_post_retrieval(selected_urls)

    if not reddit_post_data:
        print("No Reddit post data retrieved")
        reddit_post_data = []
    print(reddit_post_data)
    return {"reddit_post_data": reddit_post_data}


def analyze_google(state:State):
    print("Analyzing Google results")
    user_question = state.get("user_input","")
    google_results = state.get("google_results","")

    messages = get_google_analysis_messages(user_question,google_results)
    reply = chat_model.invoke(messages)
    return  {"google_analysis": reply.content}

def analyze_reddit(state:State):
    print("Analyzing reddit search results")

    user_question = state.get("user_question", "")
    reddit_results = state.get("reddit_results", "")
    reddit_post_data = state.get("reddit_post_data", "")

    messages = get_reddit_analysis_messages(user_question, reddit_results, reddit_post_data)
    reply = chat_model.invoke(messages)

    return {"reddit_analysis": reply.content}

def synthesize(state:State):
    print("All results together")

    user_question = state.get("user_question", "")
    google_analysis = state.get("google_analysis", "")
    reddit_analysis = state.get("reddit_analysis", "")

    messages = get_synthesis_messages(
        user_question, google_analysis,reddit_analysis
    )

    reply = chat_model.invoke(messages)
    final_answer = reply.content

    return {"final_analysis": final_answer, "messages": [{"role": "assistant", "content": final_answer}]}


graph_builder = StateGraph(State)

graph_builder.add_node("google_search",google_search)
graph_builder.add_node("reddit_search",reddit_search)
graph_builder.add_node("analyze_reddit_posts",analyze_reddit_posts)
graph_builder.add_node("retreive_reddit_posts",retreive_reddit_posts)
graph_builder.add_node("analyze_google",analyze_google)
graph_builder.add_node("analyze_reddit",analyze_reddit)
graph_builder.add_node("synthesize_analysis",synthesize)

graph_builder.add_edge(START,"google_search")
graph_builder.add_edge(START,"reddit_search")

graph_builder.add_edge("google_search","analyze_reddit_posts")
graph_builder.add_edge("reddit_search","analyze_reddit_posts")
graph_builder.add_edge("analyze_reddit_posts","retreive_reddit_posts")

graph_builder.add_edge("retreive_reddit_posts","analyze_google")
graph_builder.add_edge("retreive_reddit_posts","analyze_reddit")

graph_builder.add_edge("analyze_google","synthesize_analysis")
graph_builder.add_edge("analyze_reddit","synthesize_analysis")

graph_builder.add_edge("synthesize_analysis",END)

graph = graph_builder.compile()

def run_chatbot():
    print("Welcome to the Chatbot! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        state: State = {
            "messages": [{"role": "user", "content": user_input}],
            "user_input": user_input,
            "google_results": None,
            "reddit_results": None,
            "selected_reddit_urls": None,
            "reddit_post_data": None,
            "google_analysis": None,
            "reddit_analysis": None,
            "final_analysis": None
            }

        print("Chatbot is processing your input...")
        final_state = graph.invoke(state)
        if final_state.get("final_analysis"):
            print("Chatbot:", final_state.get("final_analysis"))


if __name__ == "__main__":
    run_chatbot()