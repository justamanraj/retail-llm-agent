# retail-llm-agent
LLM-powered retail customer support agent with RAG, conversation memory, and FastAPI deployment
# Retail LLM Agent

An LLM-powered retail customer support agent built with Python, 
Gemini API, ChromaDB, and FastAPI.

## Features
- Intent classification across 5 categories using LLM
- Modular prompt routing pipeline per intent
- RAG pipeline with ChromaDB + Sentence Transformers (1000+ product catalog)
- Stateful conversation memory across multi-turn dialogue
- REST API with FastAPI (5 endpoints, Swagger UI)
- LLM-as-judge evaluation framework (4.5/5 average score)

## Architecture
Customer Query
      ↓
Intent Classifier (LLM) → 5 categories
      ↓
Prompt Router
      ↓
RAG Search (ChromaDB) → top 3 products injected
      ↓
Conversation Memory (last 6 turns injected)
      ↓
Response Generator (LLM)
      ↓
FastAPI → JSON Response + Session Logger

## Tech Stack
- Python 3.13
- Google Gemini API (gemini-2.0-flash-lite)
- ChromaDB (vector database)
- Sentence Transformers (all-MiniLM-L6-v2)
- FastAPI + Uvicorn
- pandas

## API Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /chat | Send query, get agent response |
| GET | /history/{session_id} | Get conversation history |
| DELETE | /history/{session_id} | Clear session history |
| GET | /health | API health check |
| GET | / | Root endpoint |

## How To Run
1. Clone the repo
2. Install dependencies:
   pip install google-generativeai chromadb sentence-transformers fastapi uvicorn pandas
3. Add your Gemini API key in main.py
4. Run the API:
   python main.py
5. Open http://localhost:8000/docs for Swagger UI

## Evaluation Results
| Metric | Score |
|--------|-------|
| Relevance | 5.0/5 |
| Accuracy | 5.0/5 |
| Helpfulness | 4.5/5 |

## Project Structure
day1.py  → First LLM API call
day2.py  → Intent classifier
day3.py  → Prompt routing pipeline
day4.py  → Conversation loop + logging
day5.py  → Evaluation framework
day7.py  → Conversation memory
day8.py  → RAG pipeline (ChromaDB + embeddings)
main.py  → FastAPI deployment
