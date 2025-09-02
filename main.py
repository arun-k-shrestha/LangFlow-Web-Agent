from dotenv import load_dotenv
from typing import Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from operation import serp_search,reddit_search_api

load_dotenv()
chat_model = init_chat_model("gpt-4o-mini")

class State(TypedDict):
    messages: Annotated[list, add_messages]
    user_input: str | None
    google_results: str | None
    reddit_results: str | None
    google_analysis: str | None
    reddit_analysis: str | None
    final_analysis: str | None

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

def analyze_google(state:State):
    return  {"google_analysis": ""}

def analyze_reddit(state:State):
    return  {"reddit_analysis": ""}

def final_analysis(state:State):
    return  {"final_analysis": ""}

graph_builder = StateGraph(State)

graph_builder.add_node("google_search",google_search)
graph_builder.add_node("reddit_search",reddit_search)
graph_builder.add_node("analyze_google",analyze_google)
graph_builder.add_node("analyze_reddit",analyze_reddit)
graph_builder.add_node("final_analysis",final_analysis)

graph_builder.add_edge(START,"google_search")
graph_builder.add_edge(START,"reddit_search")

graph_builder.add_edge("google_search","analyze_google")
graph_builder.add_edge("reddit_search","analyze_reddit")

graph_builder.add_edge("analyze_google","final_analysis")
graph_builder.add_edge("analyze_reddit","final_analysis")

graph_builder.add_edge("final_analysis",END)

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