from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import os
import subprocess

app = FastAPI(title="AI Lead Scraper API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all domains
    allow_credentials=False,
    allow_methods=["*"],  # Allow POST, GET, OPTIONS, etc.
    allow_headers=["*"],  # Allow all headers
)

class ScrapeRequest(BaseModel):
    query: str
    count: int = 5
    custom_instruction: str = ""
    api_key: str

@app.get("/")
def health_check():
    return {"status": "online", "message": "FastAPI engine is ready."}

@app.post("/api/scrape")
def trigger_scraper(req: ScrapeRequest):
    if not req.api_key or not req.api_key.startswith("gsk_"):
        raise HTTPException(status_code=400, detail="Invalid Groq API Key provided.")
    
    try:
        print(f"\n🚀 Starting isolated scraping process for: {req.query}")
        
        command = [
            "python", "agent.py",
            "--query", req.query,
            "--count", str(req.count),
            "--apikey", req.api_key
        ]
        
        if req.custom_instruction:
            command.extend(["--instruction", req.custom_instruction])

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        result = subprocess.run(
            command, 
            capture_output=True, 
            text=True, 
            env=env, 
            encoding="utf-8"
        )
        
        # --- THE FIX: PRINT THE AGENT'S INTERNAL LOGS ---
        print(f"--- AGENT CONSOLE LOG ---\n{result.stdout}\n-------------------------")
        
        if result.returncode != 0:
            print(f"Agent Error Log:\n{result.stderr}")
            raise Exception("Agent script crashed.")

        csv_path = "cleaning_leads.csv"
        if not os.path.exists(csv_path):
            raise HTTPException(status_code=500, detail="CSV file was not generated.")

        # --- THE FIX: SAFELY HANDLE EMPTY CSV FILES ---
        try:
            import pandas.errors
            df = pd.read_csv(csv_path)
            
            if df.empty:
                leads_data = []
            else:
                df = df.fillna("None") 
                leads_data = df.to_dict(orient="records")
        except pandas.errors.EmptyDataError:
            # If the file is 0 bytes, just return an empty list
            leads_data = []
        # ----------------------------------------------
        
        return {
            "status": "success", 
            "message": f"Successfully scraped {len(leads_data)} leads.",
            "leads": leads_data
        }
        
    except Exception as e:
        print(f"❌ API Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))