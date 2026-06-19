
import google.generativeai as genai
import time

genai.configure(api_key="YOUR_API_KEY_HERE")
model = genai.GenerativeModel("gemini-2.5-flash")
# Step 1: Intent Classifier (carried over from Day 2)
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
    response = model.generate_content(prompt)
    return response.text.strip()


# Step 2: Prompt library — different prompt for each intent
PROMPTS = {
    "order_tracking": """
You are a helpful retail support agent specializing in order tracking.
The customer wants to know about their order status.
Ask for their order ID if not provided.
Give a realistic simulated update (e.g. "Your order is out for delivery").
Keep response under 3 sentences.
""",
    "product_recommendation": """
You are a knowledgeable retail product expert.
Help the customer find the right product based on their needs and budget.
Ask one clarifying question if needed.
Recommend 2-3 specific products with brief reasons.
Keep response friendly and concise.
""",
    "return_refund": """
You are a customer service agent handling returns and refunds.
Help the customer understand the return process clearly.
Be empathetic and solution-focused.
Mention that refunds are processed within 5-7 business days.
Keep response under 4 sentences.
""",
    "price_inquiry": """
You are a retail pricing assistant.
Help the customer with pricing information.
If you don't have exact pricing, give a realistic price range and suggest checking the website.
Mention any ongoing sale if relevant.
Keep response under 3 sentences.
""",
    "general_inquiry": """
You are a friendly retail assistant.
Answer the customer's question helpfully and professionally.
If you don't know the answer, offer to connect them with a human agent.
Keep response under 3 sentences.
"""
}


# Step 3: Route query to correct prompt and get response
def route_and_respond(query):
    # First classify
    intent = classify_intent(query)
    time.sleep(3)

    # Pick the right prompt
    system_prompt = PROMPTS.get(intent, PROMPTS["general_inquiry"])

    # Build final prompt
    final_prompt = f"""
{system_prompt}

Customer query: {query}
"""
    response = model.generate_content(final_prompt)
    return intent, response.text.strip()


# Step 4: Test the full pipeline
test_queries = [
    "Where is my order #1234?",
    "Suggest me a good phone under 15000",
    "I want to return my shoes",
    "How much does the Samsung TV cost?",
    "What are your store timings?"
]

print("Retail Agent — Full Routing Pipeline")
print("=" * 60)

for query in test_queries:
    print(f"\nCustomer : {query}")
    intent, reply = route_and_respond(query)
    print(f"Intent   : {intent}")
    print(f"Agent    : {reply}")
    print("-" * 60)
    time.sleep(5)
    