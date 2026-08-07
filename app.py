from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from tavily import TavilyClient
import streamlit as st
import os
import warnings

warnings.filterwarnings("ignore")

# -----------------------------
# API KEYS & ENVIRONMENT
# -----------------------------
# Warning: It is highly recommended to use st.secrets for API keys in production
GOOGLE_API_KEY = 'YOUR_GOOGLE_API_KEY'
TAVILY_API_KEY = 'YOUR_TAVILY_API_KEY'

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

# -----------------------------
# LLM SETUP
# -----------------------------
# Replaced 'gemini-3.5-flash-lite' with a verified model name
model = ChatGoogleGenerativeAI(
    model='gemini-1.5-flash',
    temperature=0.7
)

# -----------------------------
# TOOL CREATION
# -----------------------------
# The @tool decorator and type hints are required for LangChain/Gemini tool binding
@tool
def search_career_info(search_query: str) -> str:
    """
    Fetch latest career information, certifications and job trends.
    """
    client = TavilyClient(api_key=TAVILY_API_KEY)
    response = client.search(search_query)
    return str(response)

# -----------------------------
# AGENT CREATION
# -----------------------------
# Using LangGraph's prebuilt React Agent
agent = create_react_agent(
    model=model,
    tools=[search_career_info]
)

# -----------------------------
# MAIN AGENT FUNCTION
# -----------------------------
def main_agent(agent, _query):
    prompt = f"""
You are an Expert Career Counselor.

Generate a professional roadmap with emojis, headings, tables and bullet points.

Student Details:
{_query}

Provide the following sections:
🎯 Career Goal
🛠 Required Skills
📚 Certifications
📖 Learning Resources
💻 Recommended Projects
📅 3 Month Plan
📅 6 Month Plan
📅 1 Year Plan
💰 Expected Salary
🚀 Future Scope

Format the response professionally using markdown.
"""

    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
    )

    # Extract the final output string from the AI's last message
    roadmap_content = response["messages"][-1].content
    return roadmap_content

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(
    page_title="AI Career Roadmap Planner",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 AI Career Roadmap Planner")
st.markdown("Generate a personalized career roadmap using Agentic AI.")

name = st.text_input("👤 Name")

education = st.selectbox(
    "🎓 Education",
    ["BCA", "B.Tech", "B.Sc", "MCA", "MBA", "Other"]
)

skills = st.text_area(
    "🛠 Current Skills",
    placeholder="Python, HTML, CSS..."
)

career_goal = st.text_input(
    "🎯 Career Goal",
    placeholder="AI Engineer"
)

hours = st.slider(
    "📚 Study Hours Per Week",
    1, 40, 10
)

if st.button("Generate Roadmap"):
    
    query_details = f"""
Name: {name}
Education: {education}
Current Skills: {skills}
Career Goal: {career_goal}
Weekly Study Hours: {hours}
"""

    with st.spinner("Generating Career Roadmap..."):
        try:
            roadmap = main_agent(agent, query_details)
            st.success("Roadmap Generated Successfully!")
            st.markdown(roadmap)
        except Exception as e:
            st.error(f"An error occurred: {e}")
