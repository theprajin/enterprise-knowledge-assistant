import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

# Resolve potential case sensitivity issues in environment variables
google_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("Google_API_KEY")

if google_api_key:
    os.environ["GOOGLE_API_KEY"] = google_api_key

def get_embeddings():
    """
    Get the Google Generative AI embeddings model instance.
    Uses 'text-embedding-004' by default.
    """
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("Google API key is not set. Please set GOOGLE_API_KEY or Google_API_KEY in your environment.")
    
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
