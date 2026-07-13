"""
===========================================================
FastAPI Backend

Purpose
-------
Expose the US Tax & Legal RAG Pipeline as a REST API.

Workflow
--------
Client (Browser / Streamlit)
        │
        ▼
POST /ask
        │
        ▼
RAGPipeline.answer_question()
        │
        ▼
Return Answer

Author : Shyam Suman
Project : US Tax & Legal RAG System
===========================================================
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    QuestionRequest,
    AnswerResponse,
)

from rag.rag_pipeline import RAGPipeline


# ==========================================================
# Create FastAPI App
# ==========================================================

app = FastAPI(
    title="US Tax & Legal RAG API",
    description="Backend API for the US Tax & Legal RAG System",
    version="1.0.0",
)


# ==========================================================
# Enable CORS
# ==========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Allow all origins during development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================================
# Load RAG Pipeline Once
# ==========================================================

print("\nLoading RAG Pipeline...\n")

pipeline = RAGPipeline(debug=False)

print("\nRAG Pipeline Loaded Successfully.\n")


# ==========================================================
# Root Endpoint
# ==========================================================

@app.get("/")
def root():
    """
    Root endpoint.
    """

    return {
        "message": "US Tax & Legal RAG API is running successfully."
    }


# ==========================================================
# Health Check
# ==========================================================

@app.get("/health")
def health():
    """
    Health check endpoint.
    """

    return {
        "status": "healthy"
    }

@app.get("/stats")
def stats():
    """
    Return corpus statistics for the frontend.
    """

    return {
        "document_count": 100,
        "chunk_count": 5428
    }


# ==========================================================
# Ask Question
# ==========================================================

@app.post(
    "/ask",
    response_model=AnswerResponse,
)
def ask_question(request: QuestionRequest):
    """
    Ask a legal question.

    Receives
    --------
    {
        "question": "..."
    }

    Returns
    -------
    {
        "answer": "..."
    }
    """

    response = pipeline.answer_question(
        request.question
    )

    return AnswerResponse(
        answer=response["answer"],
        latency=response["latency_seconds"],
        model=response["model"],
        references=response.get("references", [])
    )