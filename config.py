import os
from pathlib import Path
from dotenv import load_dotenv

# Load env variables from .env if present
load_dotenv()

# Project directory configuration
BASE_DIR = Path(__file__).resolve().parent
ARCHIVE_DIR = BASE_DIR / "archive"
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

# API Keys and URLs
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Defaults
DEFAULT_PROVIDER = os.getenv("DEFAULT_PROVIDER", "groq")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "llama-3.1-70b-versatile")

def get_llm(provider: str = None, model: str = None, api_key: str = None):
    """
    Returns a configured LangChain Chat Model based on the provider and model names.
    Allows overriding api_key at runtime.
    """
    provider = (provider or DEFAULT_PROVIDER).lower()
    
    if provider == "groq":
        from langchain_openai import ChatOpenAI
        key = api_key or GROQ_API_KEY
        if not key:
            raise ValueError("Groq API Key is required but not provided.")
        model_name = model or "llama-3.1-70b-versatile"
        return ChatOpenAI(
            openai_api_base="https://api.groq.com/openai/v1",
            openai_api_key=key,
            model_name=model_name,
            temperature=0.2
        )
        
    elif provider == "grok":
        from langchain_openai import ChatOpenAI
        key = api_key or os.getenv("GROK_API_KEY", "") or GROQ_API_KEY
        if not key:
            raise ValueError("Grok (xAI) API Key is required but not provided.")
        model_name = model or "grok-beta"
        return ChatOpenAI(
            openai_api_base="https://api.x.ai/v1",
            openai_api_key=key,
            model_name=model_name,
            temperature=0.2
        )
        
    elif provider == "gemini":
        from langchain_openai import ChatOpenAI
        key = api_key or os.getenv("GEMINI_API_KEY", "")
        if not key:
            raise ValueError("Gemini API Key is required but not provided.")
        model_name = model or "gemini-1.5-flash"
        # Gemini provides an OpenAI-compatible endpoint at https://generativelanguage.googleapis.com/v1beta/openai/
        return ChatOpenAI(
            openai_api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
            openai_api_key=key,
            model_name=model_name,
            temperature=0.2
        )
        
    elif provider == "openrouter":
        from langchain_openai import ChatOpenAI
        key = api_key or OPENROUTER_API_KEY
        if not key:
            raise ValueError("OpenRouter API Key is required but not provided.")
        model_name = model or "meta-llama/llama-3.1-70b-instruct"
        return ChatOpenAI(
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=key,
            model_name=model_name,
            temperature=0.2,
            default_headers={
                "HTTP-Referer": "http://localhost:8000",
                "X-Title": "ArchiveGuru Multi-Agent Research Generator"
            }
        )
        
    elif provider == "ollama":
        from langchain_community.chat_models import ChatOllama
        model_name = model or "llama3"
        return ChatOllama(
            base_url=OLLAMA_BASE_URL,
            model=model_name,
            temperature=0.2
        )
        
    else:
        # Fallback/Default using OpenAI compatible endpoint if provider name matches a direct url or is 'openai'
        from langchain_openai import ChatOpenAI
        key = api_key or os.getenv("OPENAI_API_KEY", "")
        model_name = model or "gpt-4o-mini"
        if not key:
            raise ValueError(f"Provider '{provider}' require an API key.")
        return ChatOpenAI(
            openai_api_key=key,
            model_name=model_name,
            temperature=0.2
        )
