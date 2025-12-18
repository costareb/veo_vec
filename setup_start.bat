@echo off
echo Starting VeoVec App - if you do this for the first time this might take a while, as everything needs to be setup ...

REM 1) Start containers in background
docker compose up -d --build

REM 2) Wait until Ollama is ready (no curl needed)
echo Waiting for Ollama to start (this may take a few seconds or minutes, be patient)...
:wait_ollama
docker compose exec ollama ollama list >nul 2>&1
if errorlevel 1 (
    echo Ollama not ready yet... retrying in 3 seconds
    timeout /t 3 >nul
    goto wait_ollama
)

echo Ollama is running. Pulling models (first run only)...

REM 3) Pull models (these are cached in ollama_data volume)
docker compose exec ollama ollama pull nomic-embed-text
docker compose exec ollama ollama pull mistral-nemo

echo Models ready.

REM 4) Open browser
start http://localhost:8501

REM 5) Show app logs so errors are visible
docker compose logs -f app

pause
