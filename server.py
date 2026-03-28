from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import os
import subprocess

app = FastAPI(title="AI Lead Scraper API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
    """The main endpoint that Next.js will call to start the AI agent."""
    
    if not req.api_key or not req.api_key.startswith("gsk_"):
        raise HTTPException(status_code=400, detail="Invalid Groq API Key provided.")
    
    try:
        print(f"\n🚀 Starting isolated scraping process for: {req.query}")
        
        # We define the command just like we would type it in the terminal
        command = [
            "python", "agent.py",
            "--query", req.query,
            "--count", str(req.count),
            "--apikey", req.api_key
        ]
        
        # Add instruction if provided
        if req.custom_instruction:
            command.extend(["--instruction", req.custom_instruction])

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        # Run the command and wait for it to finish. 
        # This completely isolates Playwright from Uvicorn's event loop.
        result = subprocess.run(
            command, 
            capture_output=True, 
            text=True, 
            env=env, 
            encoding="utf-8"
        )
        
        # If the script failed, print the error log
        if result.returncode != 0:
            print(f"Agent Error Log:\n{result.stderr}")
            raise Exception("Agent script crashed.")

        # Check if the CSV was actually generated
        csv_path = "cleaning_leads.csv"
        if not os.path.exists(csv_path):
            raise HTTPException(status_code=500, detail="CSV file was not generated.")

        # Read the CSV and convert it to a dictionary for Next.js
        df = pd.read_csv(csv_path)
        df = df.fillna("None") 
        leads_data = df.to_dict(orient="records")
        
        return {
            "status": "success", 
            "message": f"Successfully scraped {req.count} leads.",
            "leads": leads_data
        }
        
    except Exception as e:
        print(f"❌ API Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))