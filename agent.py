import os
import asyncio
import json
import pandas as pd
from typing import TypedDict, List, Dict, Any
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import urllib.parse
import re
# Langchain & LangGraph
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
import sys

# Playwright & Search
from playwright.async_api import async_playwright
from ddgs import DDGS

# Load environment variables (Groq API Key)
load_dotenv()

# ==========================================
# 1. DEFINE THE STATE
# ==========================================
# This defines the data that passes between our nodes.
class AgentState(TypedDict):
    search_query: str
    target_count: int
    custom_instruction: str
    groq_api_key: str
    leads: List[Dict[str, Any]]
    final_csv_path: str

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
async def scrape_website_text(url: str) -> str:
    """Visits homepage, hunts for a contact page, and extracts data from both (RAM Optimized)."""
    
    if sys.platform == "win32":
        import asyncio
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    try:
        async with async_playwright() as p:
            # --- 
            browser = await p.chromium.launch(
                headless=True,
                args=[
                    "--disable-dev-shm-usage", # Crucial for Docker: stops Chrome from using shared memory
                    "--no-sandbox",            # Required for running as root in Docker
                    "--disable-gpu",           # We don't need graphics
                    "--disable-extensions",    # Block all extensions
                    "--disable-background-networking",
                    "--disable-default-apps",
                    "--mute-audio",
                    "--no-zygote",             # Reduces process memory overhead
                    "--single-process"         # Forces Chrome to run in one process instead of spawning many
                ]
            )
            # ---------------------------------------
            
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            page = await context.new_page()

            # Block heavy resources
            async def intercept_route(route):
                if route.request.resource_type in ["image", "media", "font", "stylesheet", "websocket"]:
                    await route.abort()
                else:
                    await route.continue_()
            
            await page.route("**/*", intercept_route)
            
            # --- 1. SCRAPE THE HOMEPAGE ---
            await page.goto(url, timeout=15000, wait_until="domcontentloaded")
            html = await page.content()
            soup = BeautifulSoup(html, 'html.parser')
            
            text_content = soup.get_text(separator=' ', strip=True)[:2500]
            links = [a.get('href') for a in soup.find_all('a', href=True) if a.get('href')]
            
            social_links = [l for l in links if 'facebook.com' in l or 'instagram.com' in l or 'linkedin.com' in l]
            email_links = [l.replace('mailto:', '').split('?')[0] for l in links if l.startswith('mailto:')]
            
            # --- 2. HUNT FOR THE CONTACT PAGE ---
            contact_url = None
            for link in links:
                if 'contact' in link.lower() or 'about' in link.lower():
                    contact_url = urllib.parse.urljoin(url, link)
                    break 
            
            # --- 3. SCRAPE THE CONTACT PAGE ---
            if contact_url and contact_url != url:
                try:
                    print(f"      -> 🕵️ Found Contact Page: {contact_url}")
                    await page.goto(contact_url, timeout=10000, wait_until="domcontentloaded")
                    contact_html = await page.content()
                    contact_soup = BeautifulSoup(contact_html, 'html.parser')
                    
                    text_content += "\n\n--- CONTACT PAGE TEXT ---\n" + contact_soup.get_text(separator=' ', strip=True)[:2000]
                    
                    contact_links = [a.get('href') for a in contact_soup.find_all('a', href=True) if a.get('href')]
                    email_links.extend([l.replace('mailto:', '').split('?')[0] for l in contact_links if l.startswith('mailto:')])
                except Exception as ce:
                    print(f"      -> ⚠️ Couldn't load contact page: {ce}")

            await page.close()
            await context.close()
            await browser.close()
            # -----------------------------------

            email_links = list(set(email_links))
            text_emails = re.findall(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', text_content)
            all_found_emails = list(set(email_links + text_emails))
            
            clean_data = f"TEXT: {text_content} \n\n SOCIAL LINKS: {social_links} \n\n HIDDEN EMAIL LINKS: {all_found_emails}"
            return clean_data
            
    except Exception as e:
        print(f"Failed to scrape {url}: {e}")
        return ""

# ==========================================
# 3. GRAPH NODES
# ==========================================
def search_node(state: AgentState):
    """Searches the web for cleaning companies."""
    print(f"🔍 Searching the web for: {state['search_query']}")
    
    leads = []
    
    # NEW: Fetch a large buffer of links (e.g., if target is 5, fetch 25)
    buffer_limit = state["target_count"] * 5 
    
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(
                state["search_query"], 
                max_results=buffer_limit, # <--- Use the buffer here
                backend="lite" 
            ))
            
            for r in results:
                leads.append({
                    "name": r.get("title", "Unknown"),
                    "website": r.get("href", ""),
                    "status": "Found"
                })
    except Exception as e:
        print(f"⚠️ Search error: {e}")
        
    return {"leads": leads}

async def process_node(state: AgentState):
    """Visits each site and uses Groq to analyze it."""
    print(f"🤖 Processing {len(state['leads'])} leads...")
    
    # NEW: Initialize Groq using the API key from the state
    llm = ChatGroq(
        model="llama-3.3-70b-versatile", 
        temperature=0, 
        api_key=state["groq_api_key"] 
    )
    
    # Prompt instructing the LLM how to analyze the data
    prompt = PromptTemplate.from_template("""
    You are an expert web analyst looking at scraped text from a cleaning company website.
    
    Website Data:
    {website_data}
    
    Special Focus Instruction from the User:
    {user_instruction}
    
    Task:
    Extract the contact information and write a 1-sentence sales critique. 
    If a "Special Focus Instruction" is provided above, make sure your critique specifically addresses it based on the website text.
    
    Return ONLY a JSON object. Do not include markdown formatting or explanation.
    
    JSON Schema required:
    {{
        "email": "Any email address found (or 'None')",
        "social_link": "Any Facebook or Instagram link found (or 'None')",
        "booking_form_present": "Yes or No",
        "critique": "A 1-sentence sales hook on how to improve the site."
    }}
    """)

    chain = prompt | llm

    updated_leads = []
    emails_found = 0 # NEW: Keep track of our goal
    target = state["target_count"]

    for lead in state["leads"]:
        # NEW: If we hit our goal, stop scraping and break the loop!
        if emails_found >= target:
            print(f"🎯 Goal reached! Found {target} leads with emails.")
            break
            
        url = lead["website"]
        if not url:
            continue
            
        print(f"   -> Scraping: {url}")
        site_data = await scrape_website_text(url)
        
        if not site_data:
            continue
            
        try:
            response = await chain.ainvoke({
                "website_data": site_data,
                "user_instruction": state.get("custom_instruction", "")
            })
            
            json_str = response.content.replace("```json", "").replace("```", "").strip()
            analysis = json.loads(json_str)
            lead.update(analysis)
            
            # NEW: Check if the AI actually found an email
            extracted_email = lead.get("email", "None")
            if extracted_email != "None" and "@" in extracted_email:
                emails_found += 1
                lead["status"] = "Valid Email Found"
                updated_leads.append(lead) # Only save it if it has an email
                print(f"      ✅ Email found! ({emails_found}/{target})")
            else:
                print("      ❌ No email found, moving to next site...")
                # We do NOT append it to updated_leads, and we do NOT increase the count
            
        except Exception as e:
            print(f"   -> LLM parsing failed for {url}: {e}")
            
    return {"leads": updated_leads}

def save_node(state: AgentState):
    """Saves the final data to an Excel/CSV file."""
    print("💾 Saving results...")
    df = pd.DataFrame(state["leads"])
    filename = "cleaning_leads.csv"
    df.to_csv(filename, index=False)
    print(f"✅ Success! Saved to {filename}")
    return {"final_csv_path": filename}

# ==========================================
# 4. BUILD AND RUN THE GRAPH
# ==========================================
def build_graph():
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("search", search_node)
    workflow.add_node("process", process_node)
    workflow.add_node("save", save_node)
    
    # Define the flow
    workflow.set_entry_point("search")
    workflow.add_edge("search", "process")
    workflow.add_edge("process", "save")
    workflow.add_edge("save", END)
    
    return workflow.compile()

async def run_agent(query: str, count: int, custom_instruction: str, user_api_key: str):
    """This function runs the agent."""
    # Force the correct event loop policy for Windows right at the start
    if sys.platform == "win32":
        import asyncio
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    graph = build_graph()
    
    initial_state = {
        "search_query": query,
        "target_count": count,
        "custom_instruction": custom_instruction,
        "groq_api_key": user_api_key,
        "leads": [],
        "final_csv_path": ""
    }
    
    result = await graph.ainvoke(initial_state)
    return result["final_csv_path"]

# This allows us to call agent.py from the command line with arguments
if __name__ == "__main__":
    import argparse
    import asyncio
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--instruction", default="")
    parser.add_argument("--apikey", required=True)
    args = parser.parse_args()
    
    # Run the async function
    asyncio.run(run_agent(args.query, args.count, args.instruction, args.apikey))