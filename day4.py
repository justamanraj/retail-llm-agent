import google.generativeai as genai
import time
import pandas as pd
from datetime import datetime

genai.configure(api_key="YOUR_API_KEY_HERE")
model = genai.GenerativeModel("gemini-3.5-flash")

log = []

def ask_llm(prompt):
    time.sleep(15)  # stay within free tier rate limit
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
    "order_tracking": """
You are a helpful retail support agent specializing in order tracking.
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
Be empathetic and solution-focused.
Mention that refunds are processed within 5-7 business days.
Keep response under 4 sentences.
""",
    "price_inquiry": """
You are a retail pricing assistant.
If you don't have exact pricing, give a realistic price range and suggest checking the website.
Keep response under 3 sentences.
""",
    "general_inquiry": """
You are a friendly retail assistant.
Answer helpfully and professionally.
If you don't know, offer to connect them with a human agent.
Keep response under 3 sentences.
"""
}


def route_and_respond(query):
    intent = classify_intent(query)
    system_prompt = PROMPTS.get(intent, PROMPTS["general_inquiry"])
    final_prompt = f"{system_prompt}\n\nCustomer query: {query}"
    response = ask_llm(final_prompt)
    return intent, response


def save_log():
    if log:
        df = pd.DataFrame(log)
        df.to_csv("agent_log.csv", index=False)
        print(f"\nSession saved to agent_log.csv ({len(log)} interactions logged)")
    else:
        print("\nNo interactions to save.")


# Main chat loop
print("=" * 60)
print("   Retail Customer Support Agent")
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

    log.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "query": user_input,
        "intent": intent,
        "response": reply
    })