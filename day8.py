import chromadb
from distro import name
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import time
import pandas as pd
from datetime import datetime

genai.configure(api_key="YOUR_API_KEY_HERE")
model = genai.GenerativeModel("gemini-2.5-flash")

embedder = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.create_collection("products")

log = []
conversation_history = []

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

collection.add(
    documents=documents,
    embeddings=embeddings,
    ids=ids,
    metadatas=metadatas
)

print(f"{len(products)} products loaded into ChromaDB")
def retrieve_products(query, top_k=3):
    query_embedding = embedder.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )

    retrieved = []
    for i in range(len(results["ids"][0])):
        retrieved.append({
            "name": results["metadatas"][0][i]["name"],
            "price": results["metadatas"][0][i]["price"],
            "category": results["metadatas"][0][i]["category"],
            "description": results["documents"][0][i]
        })
    return retrieved


# ─────────────────────────────────────────────
# STEP 4: LLM Call
# ─────────────────────────────────────────────

def ask_llm(prompt):
    time.sleep(35)
    response = model.generate_content(prompt)
    return response.text.strip()


# ─────────────────────────────────────────────
# STEP 5: Intent Classifier
# ─────────────────────────────────────────────

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


# ─────────────────────────────────────────────
# STEP 6: Prompt Library
# ─────────────────────────────────────────────

PROMPTS = {
    "order_tracking": "You are a helpful retail support agent specializing in order tracking. Ask for their order ID if not provided. Give a realistic simulated update. Keep response under 3 sentences.",
    "return_refund": "You are a customer service agent handling returns and refunds. Be empathetic. Mention refunds are processed within 5-7 business days. Keep response under 4 sentences.",
    "price_inquiry": "You are a retail pricing assistant. Give realistic price ranges and suggest checking the website. Keep response under 3 sentences.",
    "general_inquiry": "You are a friendly retail assistant. Answer helpfully and professionally. Keep response under 3 sentences."
}


# ─────────────────────────────────────────────
# STEP 7: Build Prompt with Memory + RAG
# ─────────────────────────────────────────────

def build_prompt(query, intent):

    # Add memory
    history_text = ""
    if conversation_history:
        history_text = "\n\nPrevious conversation:\n"
        for turn in conversation_history[-6:]:
            history_text += f"Customer: {turn['query']}\n"
            history_text += f"Agent: {turn['response']}\n"

    # If product recommendation or price inquiry, add RAG
    if intent in ["product_recommendation", "price_inquiry"]:
        retrieved = retrieve_products(query)

        catalog_text = "\n\nAvailable products from our catalog:\n"
        for p in retrieved:
            catalog_text += f"- {p['name']} | ₹{p['price']} | {p['description']}\n"

        system_prompt = """You are a knowledgeable retail product expert.
Rules:
1. Answer ONLY using retrieved products.
2. Never invent products.
3. Never invent prices.
4. If information is unavailable, say:
   'I could not find a matching product in the catalog.'
5. Mention exact prices from retrieved context.
6. Keep response concise.
"""

        final_prompt = f"""
{system_prompt}
{catalog_text}
{history_text}

Current customer message: {query}
"""
    else:
        system_prompt = PROMPTS.get(intent, PROMPTS["general_inquiry"])
        final_prompt = f"""
{system_prompt}
{history_text}

Current customer message: {query}
"""

    return final_prompt


# ─────────────────────────────────────────────
# STEP 8: Route and Respond
# ─────────────────────────────────────────────

def route_and_respond(query):
    intent = classify_intent(query)
    time.sleep(35)
    final_prompt = build_prompt(query, intent)
    response = ask_llm(final_prompt)
    return intent, response


def save_log():
    if log:
        df = pd.DataFrame(log)
        df.to_csv("agent_log_rag.csv", index=False)
        print(f"\nSession saved to agent_log_rag.csv ({len(log)} interactions logged)")
    else:
        print("\nNo interactions to save.")


# ─────────────────────────────────────────────
# STEP 9: Main Chat Loop
# ─────────────────────────────────────────────

print("=" * 60)
print("   Retail Agent with Memory + RAG")
print("   Type 'quit' to exit and save session")
print("=" * 60)

while True:
    user_input = input("\nYou: ").strip()

    if user_input.lower() == "quit":
        save_log()
        print("Goodbye!")
        break

    if not user_input:
        print("Please type something.")
        continue

    intent, reply = route_and_respond(user_input)

    print(f"Intent : {intent}")
    print(f"Agent  : {reply}")

    conversation_history.append({
        "query": user_input,
        "response": reply
    })

    log.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "query": user_input,
        "intent": intent,
        "response": reply,
        "memory_turns": len(conversation_history)
    })

    time.sleep(2)