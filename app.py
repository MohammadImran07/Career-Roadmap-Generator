import os
import traceback
import warnings
import streamlit as st
from google import genai
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatResult, ChatGeneration
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

    # Clear conflicting Cloud environment variables
    for env_var in ["GOOGLE_GENAI_USE_VERTEXAI", "GOOGLE_CLOUD_PROJECT", "GOOGLE_APPLICATION_CREDENTIALS"]:
        if env_var in os.environ:
            del os.environ[env_var]

    os.environ["TAVILY_API_KEY"] = clean_tavily_key

    with st.spinner("Initializing AI Agent & Generating Roadmap..."):
        try:
            # 1. Define Custom Tool with User's Tavily Key
            @tool
            def search_career_info(search_query: str) -> str:
                """Fetch latest career information, certifications, and job trends."""
                tavily_client = TavilyClient(api_key=clean_tavily_key)
                response = tavily_client.search(search_query)
                return str(response)

            # 2. Native Direct-Client Chat Model Wrapper for LangGraph ReAct Agent
            # This completely avoids langchain_google_genai credential routing errors.
            class DirectGenAIModel(BaseChatModel):
                model_name: str = "gemini-2.5-flash"
                api_key: str = clean_google_key

                def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                    # Use the official Google GenAI SDK client directly
                    client = genai.Client(api_key=self.api_key)
                    
                    # Format messages for the native client
                    contents_list = []
                    for m in messages:
                        if isinstance(m, HumanMessage):
                            contents_list.append(m.content)
                        elif isinstance(m, SystemMessage):
                            contents_list.append(f"System: {m.content}")
                        elif isinstance(m, AIMessage):
                            contents_list.append(m.content)
                        elif isinstance(m, str):
                            contents_list.append(m)

                    # Call Google AI Studio API natively via HTTP/REST
                    response = client.models.generate_content(
                        model=self.model_name,
                        contents=contents_list,
                    )
                    
                    message = AIMessage(content=response.text)
                    generation = ChatGeneration(message=message)
                    return ChatResult(generations=[generation])

                @property
                def _llm_type(self) -> str:
                    return "direct-genai"

            # 3. Instantiate Model Wrapper
            model = DirectGenAIModel()

            # 4. Create Agent
            agent = create_react_agent(
                model=model,
                tools=[search_career_info]
            )

            # 5. Construct Prompt & Run
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

            # 6. Render Output
            st.success("Roadmap Generated Successfully!")
            st.markdown(roadmap_result)

        except Exception as e:
            st.error("❌ An error occurred:")
            st.code(traceback.format_exc())
