import sys
print("Python version:", sys.version)

try:
    import fastapi
    print("fastapi imported successfully:", fastapi.__version__)
except ImportError as e:
    print("FAILED to import fastapi:", e)

try:
    import crewai
    print("crewai imported successfully")
except ImportError as e:
    print("FAILED to import crewai:", e)

try:
    import langgraph
    print("langgraph imported successfully")
except ImportError as e:
    print("FAILED to import langgraph:", e)

try:
    import langchain
    print("langchain imported successfully:", langchain.__version__)
except ImportError as e:
    print("FAILED to import langchain:", e)

try:
    from duckduckgo_search import DDGS
    print("duckduckgo-search imported successfully")
except ImportError as e:
    print("FAILED to import duckduckgo-search:", e)

try:
    import arxiv
    print("arxiv imported successfully")
except ImportError as e:
    print("FAILED to import arxiv:", e)

print("Environment verification complete.")
