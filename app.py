import os
import traceback
import warnings
import streamlit as st
from google import genai
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
        placeholder="AIza... or AQ...",
        help="Get your key at https://aistudio.google.com/app/apikey"
    )
    
    tavily_api_key = st.text_input(
        "Tavily Key (Optional for Web Search)",
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
st.markdown("Generate a personalized career roadmap using Google Gemini 3.5 Flash-Lite.")

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
    
    clean_google_key = google_api_key.strip().strip("'").strip('"')
    clean_tavily_key = tavily_api_key.strip().strip("'").strip('"')

    if not clean_google_key:
        st.warning("⚠️ Please enter your **Google Gemini API Key** in the sidebar to proceed.")
        st.stop()

    # Clear conflicting Cloud environment variables
    for env_var in ["GOOGLE_GENAI_USE_VERTEXAI", "GOOGLE_CLOUD_PROJECT", "GOOGLE_APPLICATION_CREDENTIALS"]:
        if env_var in os.environ:
            del os.environ[env_var]

    with st.spinner("Searching trends & generating custom career roadmap..."):
        try:
            # 1. Initialize Google's official native client directly
            client = genai.Client(api_key=clean_google_key)

            # 2. Optional Tavily live search context injection
            extra_search_context = ""
            if clean_tavily_key:
                try:
                    tavily = TavilyClient(api_key=clean_tavily_key)
                    search_response = tavily.search(query=f"latest industry requirements and roadmap for {career_goal}")
                    extra_search_context = f"\n\nLive Web Search Data:\n{str(search_response)}"
                except Exception:
                    pass

            # 3. Construct Detailed Professional Prompt
            prompt_query = f"""
You are an Expert Career Counselor.

Generate a comprehensive, professional career roadmap with emojis, clear headings, tables, and structured bullet points.

Student Details:
- Name: {name}
- Education: {education}
- Current Skills: {skills}
- Career Goal: {career_goal}
- Weekly Study Hours: {hours} hours/week

{extra_search_context}

Please provide the following exact sections:
1. 🎯 Career Goal Overview & Feasibility
2. 🛠 Required Technical & Soft Skills
3. 📚 Recommended Certifications
4. 📖 Curated Learning Resources (Free & Paid)
5. 💻 Hands-on Recommended Projects
6. 📅 3-Month Action Plan
7. 📅 6-Month Milestone Plan
8. 📅 1-Year Mastery Plan
9. 💰 Expected Salary Range & Job Roles
10. 🚀 Future Industry Scope & Growth

Format the response cleanly using markdown.
"""

            # 4. Directly invoke gemini-3.5-flash-lite
            response = client.models.generate_content(
                model='gemini-3.5-flash-lite',
                contents=prompt_query
            )

            roadmap_result = response.text

            # 5. Render Output cleanly
            st.success("Roadmap Generated Successfully!")
            st.markdown(roadmap_result)

        except Exception as e:
            st.error("❌ An error occurred:")
            st.code(traceback.format_exc())
