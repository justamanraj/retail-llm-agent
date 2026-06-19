import google.generativeai as genai
import chromadb
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time
import pandas as pd
from datetime import datetime
import uvicorn

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────

import os
from dotenv import load_dotenv
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")

embedder = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection("products")

app = FastAPI(
    title="Retail LLM Agent API",
    description="LLM-powered retail customer support agent with RAG and conversation memory",
    version="1.0.0"
)

# Session memory store — keyed by session_id
session_memory = {}

log = []

# ─────────────────────────────────────────────
# PRODUCT CATALOG
# ─────────────────────────────────────────────

df = pd.read_csv("amazon.csv")
df = df[
    [
        "product_id",
        "product_name",
        "category",
        "actual_price",
        "discounted_price",
        "rating",
        "about_product"
    ]
]

# Take 1466 products
df = df.sample(1000, random_state=42)

products = []

for idx, row in df.iterrows():

    products.append({
        "id": str(idx),
        "name": str(row["product_name"]),
        "category": str(row["category"]),
        "actualprice": str(row["actual_price"]),
        "discountedprice": str(row["discounted_price"]),
        "rating": str(row["rating"]),
        "description": str(row["about_product"])
        })
    

documents = [    f"""
    Product: {p['name']}
    Category: {p['category']}
    Price: {p['actualprice']}
    Description: {p['description']}
    """ for p in products]
ids = [p["id"] for p in products]
metadatas = [{"name": p["name"], "price": p["actualprice"], "category": p["category"]} for p in products]
embeddings = embedder.encode(documents).tolist()

collection.add(documents=documents, embeddings=embeddings, ids=ids, metadatas=metadatas)
print(f"{len(products)} products loaded\n")


# ─────────────────────────────────────────────
# REQUEST / RESPONSE MODELS
# ─────────────────────────────────────────────

class QueryRequest(BaseModel):
    session_id: str
    query: str

class QueryResponse(BaseModel):
    session_id: str
    query: str
    intent: str
    response: str
    memory_turns: int
    timestamp: str


# ─────────────────────────────────────────────
# CORE AGENT FUNCTIONS
# ─────────────────────────────────────────────

def ask_llm(prompt):
    time.sleep(35)
    response = model.generate_content(prompt)
    return response.text.strip()


def classify_intent(query):
    prompt = f"""
You are an intent classifier for a retail customer support system.

Classify the following customer query into EXACTLY ONE of these categories:
- order_tracking
- product_recommendation
- return_refund
- price_inquiry
- general_inquiry

Rules:
- Respond with ONLY the category name
- No explanation, no punctuation, nothing else

Customer query: {query}
"""
    return ask_llm(prompt)


PROMPTS = {
    "order_tracking":       "You are a helpful retail support agent specializing in order tracking. Ask for their order ID if not provided. Give a realistic simulated update. Keep response under 3 sentences.",
    "return_refund":        "You are a customer service agent handling returns and refunds. Be empathetic. Mention refunds are processed within 5-7 business days. Keep response under 4 sentences.",
    "price_inquiry":        "You are a retail pricing assistant. Give realistic price ranges and suggest checking the website. Keep response under 3 sentences.",
    "general_inquiry":      "You are a friendly retail assistant. Answer helpfully and professionally. Keep response under 3 sentences.",
}


def retrieve_products(query, top_k=3):
    query_embedding = embedder.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)
    retrieved = []
    for i in range(len(results["ids"][0])):
        retrieved.append({
            "name":        results["metadatas"][0][i]["name"],
            "price":       results["metadatas"][0][i]["price"],
            "category":    results["metadatas"][0][i]["category"],
            "description": results["documents"][0][i]
        })
    return retrieved


def build_prompt(query, intent, history):
    history_text = ""
    if history:
        history_text = "\n\nPrevious conversation:\n"
        for turn in history[-6:]:
            history_text += f"Customer: {turn['query']}\n"
            history_text += f"Agent: {turn['response']}\n"

    if intent in ["product_recommendation", "price_inquiry"]:
        retrieved = retrieve_products(query)
        catalog_text = "\n\nAvailable products from our catalog:\n"
        for p in retrieved:
            catalog_text += f"- {p['name']} | ₹{p['price']} | {p['description']}\n"

        system_prompt = """You are a knowledgeable retail product expert.
Recommend products ONLY from the catalog provided below.
Do not suggest products outside this catalog.
Mention the exact price from the catalog.
Keep response friendly and concise."""

        return f"{system_prompt}{catalog_text}{history_text}\n\nCustomer: {query}"

    else:
        system_prompt = PROMPTS.get(intent, PROMPTS["general_inquiry"])
        return f"{system_prompt}{history_text}\n\nCustomer: {query}"


def route_and_respond(query, history):
    intent = classify_intent(query)
    time.sleep(2)
    prompt = build_prompt(query, intent, history)
    response = ask_llm(prompt)
    return intent, response


# ─────────────────────────────────────────────
# API ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "message": "Retail LLM Agent API is running",
        "version": "1.0.0",
        "endpoints": {
            "chat":         "POST /chat",
            "history":      "GET  /history/{session_id}",
            "clear":        "DELETE /history/{session_id}",
            "health":       "GET  /health",
            "docs":         "GET  /docs"
        }
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "products_loaded": len(products),
        "active_sessions": len(session_memory)
    }


@app.post("/chat", response_model=QueryResponse)
def chat(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Get or create session memory
    if request.session_id not in session_memory:
        session_memory[request.session_id] = []

    history = session_memory[request.session_id]

    # Get response from agent
    intent, response = route_and_respond(request.query, history)

    # Update memory
    history.append({"query": request.query, "response": response})

    # Log interaction
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log.append({
        "timestamp":    timestamp,
        "session_id":   request.session_id,
        "query":        request.query,
        "intent":       intent,
        "response":     response,
        "memory_turns": len(history)
    })

    return QueryResponse(
        session_id=request.session_id,
        query=request.query,
        intent=intent,
        response=response,
        memory_turns=len(history),
        timestamp=timestamp
    )


@app.get("/history/{session_id}")
def get_history(session_id: str):
    if session_id not in session_memory:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "session_id": session_id,
        "turns": len(session_memory[session_id]),
        "history": session_memory[session_id]
    }


@app.delete("/history/{session_id}")
def clear_history(session_id: str):
    if session_id in session_memory:
        del session_memory[session_id]
    return {"message": f"Session {session_id} cleared"}


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)