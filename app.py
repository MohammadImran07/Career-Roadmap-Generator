import os
import warnings
import streamlit as st
from google import genai
from google.genai import types
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
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
    
    clean_google_key = google_api_key.strip().strip("'").strip('"')
    clean_tavily_key = tavily_api_key.strip().strip("'").strip('"')

    if not clean_google_key:
        st.warning("⚠️ Please enter your **Google Gemini API Key** in the sidebar to proceed.")
        st.stop()
        
    if not clean_tavily_key:
        st.warning("⚠️ Please enter your **Tavily API Key** in the sidebar to proceed.")
        st.stop()

    os.environ["TAVILY_API_KEY"] = clean_tavily_key

    with st.spinner("Initializing AI Agent & Generating Roadmap..."):
        try:
            # Initialize Google's official modern client (Guaranteed to accept AI Studio API keys natively)
            client = genai.Client(api_key=clean_google_key)

            # Define Custom Tool with User's Tavily Key
            @tool
            def search_career_info(search_query: str) -> str:
                """Fetch latest career information, certifications, and job trends."""
                tavily_client = TavilyClient(api_key=clean_tavily_key)
                response = tavily_client.search(search_query)
                return str(response)

            # Custom lightweight wrapper to bridge Google's native client with LangGraph ReAct agent
            class NativeGeminiChat(BaseChatModel):
                model_name: str = "gemini-2.5-flash"
                api_key_val: str = clean_google_key

                def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                    native_client = genai.Client(api_key=self.api_key_val)
                    
                    # Convert LangChain message history to text/contents for the native client
                    formatted_contents = ""
                    for m in messages:
                        if isinstance(m, HumanMessage):
                            formatted_contents += f"\n{m.content}"
                        elif isinstance(m, AIMessage):
                            formatted_contents += f"\n{m.content}"
                        elif isinstance(m, str):
                            formatted_contents += f"\n{m}"

                    response = native_client.models.generate_content(
                        model=self.model_name,
                        contents=formatted_contents,
                    )
                    
                    from langchain_core.outputs import ChatResult, ChatGeneration
                    message = AIMessage(content=response.text)
                    generation = ChatGeneration(message=message)
                    return ChatResult(generations=[generation])

                @property
                def _llm_type(self) -> str:
                    return "native-gemini"

            # Instantiate model wrapper using a verified stable model
            model = NativeGeminiChat(model_name="gemini-2.5-flash")

            # Create Agent
            agent = create_react_agent(
                model=model,
                tools=[search_career_info]
            )

            # Construct Prompt & Run
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

            st.success("Roadmap Generated Successfully!")
            st.markdown(roadmap_result)

        except Exception as e:
            st.error(f"❌ An error occurred: {e}")
