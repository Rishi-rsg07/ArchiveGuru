import re
import sys
import queue
import threading
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from crewai import Crew

import config
import agents

# ----------------------------------------------------
# Thread-Safe Stdout Redirection for Streaming Logs
# ----------------------------------------------------

class ThreadSafeStdoutRedirector:
    def __init__(self):
        self.terminal = sys.stdout
        self.thread_local = threading.local()

    def register_callback(self, callback):
        self.thread_local.callback = callback

    def unregister_callback(self):
        if hasattr(self.thread_local, "callback"):
            del self.thread_local.callback

    def write(self, message):
        self.terminal.write(message)
        if hasattr(self.thread_local, "callback") and message.strip():
            self.thread_local.callback(message)

    def flush(self):
        self.terminal.flush()

    def isatty(self):
        return hasattr(self.terminal, "isatty") and self.terminal.isatty()

# Set sys.stdout to our redirector if it isn't already
if not isinstance(sys.stdout, ThreadSafeStdoutRedirector):
    sys.stdout = ThreadSafeStdoutRedirector()

# ----------------------------------------------------
# Active Jobs Cache
# ----------------------------------------------------

class JobStatus:
    def __init__(self):
        self.logs_queue = queue.Queue()
        self.state = {}
        self.completed = False
        self.error = None

active_jobs: Dict[str, JobStatus] = {}

# ----------------------------------------------------
# LangGraph State Definition
# ----------------------------------------------------

class ResearchState(TypedDict):
    topic: str
    word_count: int
    research_notes: str
    draft: str
    validation_report: str
    validation_score: int
    iterations: int
    max_iterations: int
    status: str
    job_id: str
    provider: str
    model: str
    api_key: Optional[str]

# Helper to log status updates specifically
def log_status(job_id: str, status_msg: str):
    print(f"\n[STATUS_UPDATE] {status_msg}\n")

# ----------------------------------------------------
# Graph Nodes
# ----------------------------------------------------

def research_node(state: ResearchState) -> ResearchState:
    job_id = state["job_id"]
    log_status(job_id, f"Research Node Started. Searching for sources on: {state['topic']}")
    
    # Get custom LLM configured for this job
    llm = config.get_llm(state["provider"], state["model"], state["api_key"])
    
    # Assemble CrewAI Researcher Agent and Task
    researcher = agents.get_researcher(llm)
    task = agents.get_research_task(researcher, state["topic"])
    
    crew = Crew(
        agents=[researcher],
        tasks=[task],
        verbose=True
    )
    
    result = crew.kickoff()
    
    state["research_notes"] = str(result)
    state["status"] = "Drafting initial version"
    log_status(job_id, "Research Node Completed. Sources compiled.")
    return state

def draft_node(state: ResearchState) -> ResearchState:
    job_id = state["job_id"]
    log_status(job_id, "Drafting Node Started. Writing first draft...")
    
    llm = config.get_llm(state["provider"], state["model"], state["api_key"])
    
    writer = agents.get_writer(llm)
    task = agents.get_drafting_task(writer, state["topic"])
    
    # We pass the research notes as context to the task inputs
    crew = Crew(
        agents=[writer],
        tasks=[task],
        verbose=True
    )
    
    result = crew.kickoff(inputs={
        "topic": state["topic"],
        "research_notes": state["research_notes"]
    })
    
    state["draft"] = str(result)
    state["status"] = "Validating draft"
    log_status(job_id, "Drafting Node Completed. First version written.")
    return state

def validate_node(state: ResearchState) -> ResearchState:
    job_id = state["job_id"]
    log_status(job_id, f"Validation Node Started (Iteration {state['iterations'] + 1}). Reviewing draft...")
    
    llm = config.get_llm(state["provider"], state["model"], state["api_key"])
    
    reviewer = agents.get_reviewer(llm)
    task = agents.get_validation_task(reviewer, state["topic"])
    
    crew = Crew(
        agents=[reviewer],
        tasks=[task],
        verbose=True
    )
    
    result = str(crew.kickoff(inputs={
        "topic": state["topic"],
        "research_notes": state["research_notes"],
        "draft": state["draft"]
    }))
    
    # Parse score
    score = 70  # default
    score_match = re.search(r"SCORE:\s*(\d+)", result, re.IGNORECASE)
    if score_match:
        try:
            score = int(score_match.group(1))
        except ValueError:
            pass
            
    state["validation_report"] = result
    state["validation_score"] = score
    state["status"] = f"Validated (Score: {score}/100)"
    log_status(job_id, f"Validation Node Completed. Peer-Review Score: {score}/100")
    return state

def refine_node(state: ResearchState) -> ResearchState:
    job_id = state["job_id"]
    state["iterations"] += 1
    log_status(job_id, f"Refinement Node Started (Iteration {state['iterations']}). Re-writing draft based on review comments...")
    
    llm = config.get_llm(state["provider"], state["model"], state["api_key"])
    
    refiner = agents.get_refiner(llm)
    task = agents.get_refinement_task(refiner, state["topic"])
    
    crew = Crew(
        agents=[refiner],
        tasks=[task],
        verbose=True
    )
    
    result = crew.kickoff(inputs={
        "topic": state["topic"],
        "draft": state["draft"],
        "validation_report": state["validation_report"]
    })
    
    state["draft"] = str(result)
    state["status"] = "Re-validating revised draft"
    log_status(job_id, f"Refinement Node Completed. Version {state['iterations'] + 1} generated.")
    return state

def finalize_node(state: ResearchState) -> ResearchState:
    job_id = state["job_id"]
    log_status(job_id, "Finalizing Paper Node. Formatting output and exporting to archive...")
    state["status"] = "Completed"
    log_status(job_id, "Workflow Complete. Paper written successfully!")
    return state

# ----------------------------------------------------
# Routing Logic
# ----------------------------------------------------

def validation_router(state: ResearchState):
    if state["validation_score"] >= 80 or state["iterations"] >= state["max_iterations"]:
        return "finalize"
    return "refine"

# ----------------------------------------------------
# Graph Compilation
# ----------------------------------------------------

workflow = StateGraph(ResearchState)

# Add Nodes
workflow.add_node("research", research_node)
workflow.add_node("draft", draft_node)
workflow.add_node("validate", validate_node)
workflow.add_node("refine", refine_node)
workflow.add_node("finalize", finalize_node)

# Set Entry Point and Edges
workflow.set_entry_point("research")
workflow.add_edge("research", "draft")
workflow.add_edge("draft", "validate")

workflow.add_conditional_edges(
    "validate",
    validation_router,
    {
        "finalize": "finalize",
        "refine": "refine"
    }
)

workflow.add_edge("refine", "validate")
workflow.add_edge("finalize", END)

# Compiled App
app = workflow.compile()
