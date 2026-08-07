import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# 1. Provide your API Key
API_KEY = "your_actual_api_key_here"

# 2. Set BOTH environment variables (Recent SDK versions check these differently)
os.environ["GOOGLE_API_KEY"] = API_KEY
os.environ["GEMINI_API_KEY"] = API_KEY

# 3. Explicitly disable Vertex AI to prevent the OAuth Bearer token bug
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "false"

def main():
    # Initialize using a different model (gemini-2.5-flash)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        api_key=API_KEY,
        vertexai=False,  # <-- CRITICAL FIX: Forces API key auth instead of OAuth
        temperature=0.7
        # NOTE: Do NOT add a `project="..."` parameter here, as it will trigger OAuth.
    )

    print("Sending request to Gemini...")
    
    try:
        # 4. Test the invocation
        response = llm.invoke([
            HumanMessage(content="Hello! Please reply with 'Authentication successful!'")
        ])
        print("\nSuccess! Response:")
        print(response.content)
        
    except Exception as e:
        print(f"\nFailed. Error details:\n{e}")

if __name__ == "__main__":
    main()
