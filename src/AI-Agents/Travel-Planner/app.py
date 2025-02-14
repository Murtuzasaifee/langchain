import streamlit as st
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import requests
import json
import os

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"]="Travel-Planner"


## Fetches flight prices from a free mock API.
def search_flights(destination):
    api_url = f"https://api.instantwebtools.net/v1/airlines"
    response = requests.get(api_url)
    if response.status_code == 200:
        data = response.json()
        return data[:5]
    return {"error": "Unable to fetch flight data."}

flight_tool = Tool(
    name="Flight Search Tool",
    func=lambda x: search_flights(x),
    description="Search for flights and prices to a given destination."
)

## Fetch travel recommendations from a free travel API.
def retrieve_travel_info(destination):
    api_url = f"https://api.api-ninjas.com/v1/city?name={destination}"
    headers = {"X-Api-Key": "uloTSrLHpM5azCYy2Qkv+Q==SGYCZC9R7HIDoMme"}
    response = requests.get(api_url, headers=headers)
    if response.status_code == 200:
        return response.json()
    return {"error": "Unable to fetch travel data."}

travel_tool = Tool(
    name="Travel Information Tool",
    func=lambda x: retrieve_travel_info(x),
    description="Retrieve travel recommendations for a given destination."
)

# Initializing LangChain Agent
llm = ChatOpenAI(model='gpt-4o',temperature=0.7, max_tokens=1024)
agent = initialize_agent(
    tools=[flight_tool, travel_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

st.title("Travel Itinerary Planner")
destination = st.text_input("Enter your destination:")
budget = st.selectbox("Choose your budget:", ["Low", "Moderate", "High"])

if st.button("Plan My Trip"):
    with st.spinner("Generating itinerary..."):
        response = agent.run(f"Plan a detailed 5-day trip to {destination} on a {budget} budget. Include daily activities, recommended hotels, food options, and travel tips.")
        st.success("Here's your travel itinerary:")
        st.write(response)
