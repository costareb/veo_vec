@echo off
echo Starting VeoVec App...

REM 1) Start containers 
docker compose up -d

REM 2) Open browser
start http://localhost:8501

REM 3) Show app logs
docker compose logs -f app
