import os
import warnings
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from tavily import TavilyClient

warnings.filterwarnings("ignore")

# -----------------------------
# STREAMLIT UI CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="AI Career Roadmap Planner",
    page_icon="🚀",
    layout="wide"
)

# -----------------------------
# SIDEBAR: API KEY INPUTS
# -----------------------------
with st.sidebar:
    st.header("🔑 API Configuration")
    
    google_api_key = st.text_input(
        "Google Gemini API Key",
        type="password",
        placeholder="AIzaSy...",
        help="Get your key at https://aistudio.google.com/app/apikey"
    )
    
    tavily_api_key = st.text_input(
        "Tavily API Key",
        type="password",
        placeholder="tvly-...",
        help="Get your key at https://tavily.com/"
    )
    
    st.markdown("---")
    st.markdown(
        "ℹ️ **Note:** Your API keys are used only for your current session "
        "and are never saved or logged."
    )

st.title("🚀 AI Career Roadmap Planner")
st.markdown("Generate a personalized career roadmap using Agentic AI.")

# -----------------------------
# MAIN APP INPUTS
# -----------------------------
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

# -----------------------------
# GENERATION LOGIC
# -----------------------------
if st.button("Generate Roadmap"):
    # Validate API Keys before running
    if not google_api_key:
        st.warning("⚠️ Please enter your **Google Gemini API Key** in the sidebar to proceed.")
        st.stop()

    if not tavily_api_key:
        st.warning("⚠️ Please enter your **Tavily API Key** in the sidebar to proceed.")
        st.stop()

    # Set environment variables for current execution run
    os.environ["GOOGLE_API_KEY"] = google_api_key
    os.environ["TAVILY_API_KEY"] = tavily_api_key

    with st.spinner("Initializing AI Agent & Generating Roadmap..."):
        try:
            # 1. Instantiate Gemini Model
            model = ChatGoogleGenerativeAI(
                model='gemini-1.5-flash',
                google_api_key=google_api_key,
                temperature=0.7
            )

            # 2. Define Custom Tool with User's Tavily Key
            @tool
            def search_career_info(search_query: str) -> str:
                """Fetch latest career information, certifications, and job trends."""
                client = TavilyClient(api_key=tavily_api_key)
                response = client.search(search_query)
                return str(response)

            # 3. Create Agent
            agent = create_react_agent(
                model=model,
                tools=[search_career_info]
            )

            # 4. Construct Prompt & Run
            prompt_query = f"""
You are an Expert Career Counselor.

Generate a professional roadmap with emojis, headings, tables, and bullet points.

Student Details:
Name: {name}
Education: {education}
Current Skills: {skills}
Career Goal: {career_goal}
Weekly Study Hours: {hours}

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
                {"messages": [{"role": "user", "content": prompt_query}]}
            )

            roadmap_result = response["messages"][-1].content

            # 5. Render Output
            st.success("Roadmap Generated Successfully!")
            st.markdown(roadmap_result)

        except Exception as e:
            st.error(f"❌ An error occurred: {e}")
