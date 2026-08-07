from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
import langchain
from langchain.agents import create_agent
from tavily import TavilyClient
import streamlit as st
import os
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# Step 3 API Keys
GOOGLE_API_KEY = 'AQ.Ab8RN6IP2UDeVTDB9670xfJFYrdj6mQSwENvJidevDuWhtseXg'
GROQ_API_KEY = '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'
TAVILY_API_KEY = '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'

model = ChatGoogleGenerativeAI(
    model = 'gemini-3.5-flash-lite',
    google_api_key = GOOGLE_API_KEY
)

response = model.invoke("Hello Buddy!")
response.content[-1]['text']
# -----------------------------
# TOOL CREATION
# -----------------------------

def search_career_info(query):
    """
    Fetch latest career information,
    certifications and job trends.
    """

    client = TavilyClient(
        api_key=TAVILY_API_KEY
    )

    response = client.search(query)

    return str(response)

# -----------------------------
# AGENT CREATION
# -----------------------------

agent = create_agent(
    model=model,
    tools=[search_career_info]
)

# -----------------------------
# MAIN AGENT
# -----------------------------

def main_agent(agent, query):

    prompt = f"""
You are an Expert Career Counselor.

Generate a professional roadmap with emojis,
headings, tables and bullet points.

Student Details:
{query}

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

    roadmap = response["messages"][-1].content

    try:
        return roadmap[0]["text"]
    except:
        return str(roadmap)

# -----------------------------
# STREAMLIT UI
# -----------------------------

st.set_page_config(
    page_title="AI Career Roadmap Planner",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 AI Career Roadmap Planner")

st.markdown(
    "Generate a personalized career roadmap using Agentic AI."
)

name = st.text_input("👤 Name")

education = st.selectbox(
    "🎓 Education",
    [
        "BCA",
        "B.Tech",
        "B.Sc",
        "MCA",
        "MBA",
        "Other"
    ]
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
    1,
    40,
    10
)

if st.button("Generate Roadmap"):

    query = f"""
Name: {name}

Education: {education}

Current Skills: {skills}

Career Goal: {career_goal}

Weekly Study Hours: {hours}
"""

    with st.spinner("Generating Career Roadmap..."):

        roadmap = main_agent(
            agent,
            query
        )

    st.success("Roadmap Generated Successfully!")

    st.markdown(roadmap)
