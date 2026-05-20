import os
import uuid
import datetime
import json
import threading
import sys
import asyncio
from typing import Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

import config
import workflow

app = FastAPI(title="ArchiveGuru Multi-Agent Research Generator & Validator")

# Enable CORS for local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------
# Request and Response Models
# ----------------------------------------------------

class GenerateRequest(BaseModel):
    topic: str
    max_iterations: int = 2
    provider: str = "groq"
    model: str = "llama-3.1-70b-versatile"
    api_key: Optional[str] = None

class PaperMetadata(BaseModel):
    job_id: str
    topic: str
    date: str
    score: int
    provider: str
    model: str
    iterations: int

# ----------------------------------------------------
# Background Executor
# ----------------------------------------------------

def execute_workflow(job_id: str, req: GenerateRequest):
    job = workflow.active_jobs[job_id]
    
    # Configure logger callback
    def log_callback(message: str):
        # Push message to queue
        job.logs_queue.put(message)
        
    # Register callback on ThreadSafeStdoutRedirector
    if isinstance(sys.stdout, workflow.ThreadSafeStdoutRedirector):
        sys.stdout.register_callback(log_callback)
        
    try:
        # Initial State
        initial_state = {
            "topic": req.topic,
            "word_count": 1000,
            "research_notes": "",
            "draft": "",
            "validation_report": "",
            "validation_score": 0,
            "iterations": 0,
            "max_iterations": req.max_iterations,
            "status": "Starting Research",
            "job_id": job_id,
            "provider": req.provider,
            "model": req.model,
            "api_key": req.api_key
        }
        
        # Run LangGraph App
        result_state = workflow.app.invoke(initial_state)
        job.state = result_state
        job.completed = True
        
        # Save to local archive directory
        timestamp = datetime.datetime.now().isoformat()
        
        # Write metadata JSON
        meta_filename = config.ARCHIVE_DIR / f"{job_id}.json"
        metadata = {
            "job_id": job_id,
            "topic": req.topic,
            "date": timestamp,
            "score": result_state.get("validation_score", 0),
            "provider": req.provider,
            "model": req.model,
            "iterations": result_state.get("iterations", 0),
            "validation_report": result_state.get("validation_report", ""),
            "research_notes": result_state.get("research_notes", ""),
            "draft": result_state.get("draft", ""),
            "status": "Completed"
        }
        with open(meta_filename, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)
            
        # Write Markdown draft
        md_filename = config.ARCHIVE_DIR / f"{job_id}.md"
        with open(md_filename, "w", encoding="utf-8") as f:
            f.write(result_state.get("draft", ""))
            
        job.logs_queue.put("\n[STATUS_UPDATE] Job completed and saved to Archive.\n")
        
    except Exception as e:
        import traceback
        error_tb = traceback.format_exc()
        job.error = str(e)
        job.completed = True
        job.logs_queue.put(f"\n[ERROR] Workflow failed: {str(e)}\n{error_tb}\n")
        
        # Save failed job info to archive too so user knows what went wrong
        try:
            timestamp = datetime.datetime.now().isoformat()
            metadata = {
                "job_id": job_id,
                "topic": req.topic,
                "date": timestamp,
                "score": 0,
                "provider": req.provider,
                "model": req.model,
                "iterations": 0,
                "validation_report": "",
                "research_notes": "",
                "draft": f"# Generation Failed\n\nError:\n```\n{str(e)}\n```\n\nStacktrace:\n```\n{error_tb}\n```",
                "status": "Failed"
            }
            meta_filename = config.ARCHIVE_DIR / f"{job_id}.json"
            with open(meta_filename, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=4)
        except Exception:
            pass
            
    finally:
        # Always unregister callback
        if isinstance(sys.stdout, workflow.ThreadSafeStdoutRedirector):
            sys.stdout.unregister_callback()

# ----------------------------------------------------
# API Routes
# ----------------------------------------------------

@app.post("/api/generate")
def generate_paper(req: GenerateRequest, background_tasks: BackgroundTasks):
    # Check if API key is present for cloud APIs
    if req.provider in ["groq", "openrouter"]:
        key_exists = (req.provider == "groq" and (req.api_key or config.GROQ_API_KEY)) or \
                     (req.provider == "openrouter" and (req.api_key or config.OPENROUTER_API_KEY))
        if not key_exists:
            raise HTTPException(
                status_code=400, 
                detail=f"API Key for {req.provider.upper()} is missing. Please provide it or configure the server `.env` file."
            )
            
    job_id = uuid.uuid4().hex[:8]
    workflow.active_jobs[job_id] = workflow.JobStatus()
    
    # Run in background
    background_tasks.add_task(execute_workflow, job_id, req)
    
    return {"job_id": job_id}

@app.get("/api/stream/{job_id}")
async def stream_job_logs(job_id: str):
    if job_id not in workflow.active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
        
    job = workflow.active_jobs[job_id]
    
    async def event_generator():
        # Send initial status
        yield {"data": json.dumps({"type": "status", "message": "Connection established. Starting stream..."})}
        
        while True:
            # Yield any pending logs
            while not job.logs_queue.empty():
                try:
                    log_line = job.logs_queue.get_nowait()
                    # Check for our special status message format
                    if log_line.startswith("\n[STATUS_UPDATE] ") or log_line.startswith("[STATUS_UPDATE] "):
                        clean_msg = log_line.replace("\n[STATUS_UPDATE] ", "").replace("[STATUS_UPDATE] ", "").strip()
                        yield {"data": json.dumps({"type": "status", "message": clean_msg})}
                    elif log_line.startswith("\n[ERROR] ") or log_line.startswith("[ERROR] "):
                        clean_msg = log_line.replace("\n[ERROR] ", "").replace("[ERROR] ", "").strip()
                        yield {"data": json.dumps({"type": "error", "message": clean_msg})}
                    else:
                        yield {"data": json.dumps({"type": "log", "message": log_line})}
                except queue.Empty:
                    break
                    
            # Check if job is completed and all logs yielded
            if job.completed and job.logs_queue.empty():
                if job.error:
                    yield {"data": json.dumps({"type": "complete", "status": "failed", "error": job.error})}
                else:
                    yield {"data": json.dumps({"type": "complete", "status": "success", "result": job.state.get("status")})}
                break
                
            await asyncio.sleep(0.2)
            
    return EventSourceResponse(event_generator())

@app.get("/api/papers")
def list_papers():
    papers = []
    for filepath in config.ARCHIVE_DIR.glob("*.json"):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                papers.append({
                    "job_id": data.get("job_id"),
                    "topic": data.get("topic"),
                    "date": data.get("date"),
                    "score": data.get("score"),
                    "provider": data.get("provider"),
                    "model": data.get("model"),
                    "iterations": data.get("iterations"),
                    "status": data.get("status", "Completed")
                })
        except Exception:
            pass
            
    # Sort papers by date descending
    papers.sort(key=lambda x: x.get("date", ""), reverse=True)
    return papers

@app.get("/api/papers/{job_id}")
def get_paper(job_id: str):
    json_path = config.ARCHIVE_DIR / f"{job_id}.json"
    if not json_path.exists():
        raise HTTPException(status_code=404, detail="Paper not found")
        
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read paper data: {str(e)}")

@app.delete("/api/papers/{job_id}")
def delete_paper(job_id: str):
    json_path = config.ARCHIVE_DIR / f"{job_id}.json"
    md_path = config.ARCHIVE_DIR / f"{job_id}.md"
    
    if not json_path.exists():
        raise HTTPException(status_code=404, detail="Paper not found")
        
    try:
        if json_path.exists():
            os.remove(json_path)
        if md_path.exists():
            os.remove(md_path)
        return {"status": "success", "message": "Paper removed from archive"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete paper: {str(e)}")

# Mount the static files directory containing frontend files
static_dir = config.BASE_DIR / "static"
static_dir.mkdir(exist_ok=True)

app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    # Start server
    print("Starting server on http://localhost:8000")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
