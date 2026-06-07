import os
from fastapi import FastAPI
from dotenv import load_dotenv

# Load environment variables if .env file exists
load_dotenv()

app = FastAPI(
    title="Academic Policy Graph RAG API",
    description="A Graph RAG-based Academic Policy Assistant for student guidance.",
    version="1.0.0"
)

@app.get("/api/v1/graph-rag/health")
async def health_check():
    return {"status": "ok"}
