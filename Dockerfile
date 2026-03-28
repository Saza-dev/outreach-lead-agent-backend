# 1. Use Microsoft's official Playwright Python image 
# (This automatically includes Chromium and ALL required Linux dependencies!)
FROM mcr.microsoft.com/playwright/python:v1.42.0-jammy

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy your requirements file and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy all your actual code (server.py, agent.py, etc.)
COPY . .

# 5. Tell the container to start your FastAPI server. 
# It listens on Render's assigned PORT, or defaults to 8000.
CMD uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}