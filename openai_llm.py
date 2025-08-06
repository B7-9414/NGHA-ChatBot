from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# 🔐 Load environment variables (from .env file)
load_dotenv()

def get_llm():
    return ChatOpenAI(
        model="gpt-3.5-turbo",  # ✅ using 3.5 instead of gpt-4
        temperature=0.1,
        max_tokens=1024
    )
