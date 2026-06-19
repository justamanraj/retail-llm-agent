import google.generativeai as genai
import time

genai.configure(api_key="YOUR_API_KEY_HERE")
model = genai.GenerativeModel("gemini-3.5-flash")

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

# Test queries
test_queries = [
    "Where is my order #1234?",
    "Suggest me a good phone under 15000",
    "I want to return my shoes",
    "How much does the Samsung TV cost?",
    "What are your store timings?",
    "My package hasn't arrived yet",
    "Can I get a refund?",
    "What laptop should I buy for coding?",
    "Is there a discount on iPhones?",
    "Hello, I need some help"
]

print("Testing Intent Classifier")
print("=" * 50)

for query in test_queries:
    intent = classify_intent(query)
    print(f"Query   : {query}")
    print(f"Intent  : {intent}")
    print("-" * 50)
    time.sleep(15)