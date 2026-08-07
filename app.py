# -----------------------------
# GENERATION LOGIC
# -----------------------------
if st.button("Generate Roadmap"):
    
    # 1. Clean the keys (remove accidental spaces or quotes)
    clean_google_key = google_api_key.strip().strip("'").strip('"')
    clean_tavily_key = tavily_api_key.strip().strip("'").strip('"')

    # 2. Validate Keys before running
    if not clean_google_key:
        st.warning("⚠️ Please enter your **Google Gemini API Key** in the sidebar to proceed.")
        st.stop()
        
    if not clean_google_key.startswith("AIzaSy"):
        st.error("❌ Invalid Google API Key format. Gemini API keys must start with 'AIzaSy'. Please check your key at https://aistudio.google.com/app/apikey")
        st.stop()

    if not clean_tavily_key:
        st.warning("⚠️ Please enter your **Tavily API Key** in the sidebar to proceed.")
        st.stop()

    # Set environment variables for current execution run
    os.environ["GOOGLE_API_KEY"] = clean_google_key
    os.environ["TAVILY_API_KEY"] = clean_tavily_key

    with st.spinner("Initializing AI Agent & Generating Roadmap..."):
        try:
            # 3. Instantiate Gemini Model
            # Note: Using 'api_key' parameter explicitly instead of 'google_api_key'
            model = ChatGoogleGenerativeAI(
                model='gemini-1.5-flash',
                api_key=clean_google_key, 
                temperature=0.7
            )

            # 4. Define Custom Tool with User's Tavily Key
            @tool
            def search_career_info(search_query: str) -> str:
                """Fetch latest career information, certifications, and job trends."""
                client = TavilyClient(api_key=clean_tavily_key)
                response = client.search(search_query)
                return str(response)

            # 5. Create Agent
            agent = create_react_agent(
                model=model,
                tools=[search_career_info]
            )

            # 6. Construct Prompt & Run
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

            # 7. Render Output
            st.success("Roadmap Generated Successfully!")
            st.markdown(roadmap_result)

        except Exception as e:
            st.error(f"❌ An error occurred: {e}")
