#!/usr/bin/env bash
set -e

python -m http.server 8000 --directory /app/pdf_data &
streamlit run /app/app.py --server.address=0.0.0.0 --server.port=8501
