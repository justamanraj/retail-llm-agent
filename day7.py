import google.generativeai as genai
import time
import pandas as pd
from datetime import datetime

genai.configure(api_key="YOUR_API_KEY_HERE")
model = genai.GenerativeModel("gemini-3.5-flash")

log = []
conversation_history = []  # This is the memory

def ask_llm(prompt):
    time.sleep(15)
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
    "order_tracking": "You are a helpful retail support agent specializing in order tracking. Ask for their order ID if not provided. Give a realistic simulated update. Keep response under 3 sentences.",
    "product_recommendation": "You are a knowledgeable retail product expert. Recommend 2-3 specific products with brief reasons. Keep response friendly and concise.",
    "return_refund": "You are a customer service agent handling returns and refunds. Be empathetic. Mention refunds are processed within 5-7 business days. Keep response under 4 sentences.",
    "price_inquiry": "You are a retail pricing assistant. Give realistic price ranges and suggest checking the website. Keep response under 3 sentences.",
    "general_inquiry": "You are a friendly retail assistant. Answer helpfully and professionally. Keep response under 3 sentences."
}


def build_prompt_with_memory(query, intent):
    system_prompt = PROMPTS.get(intent, PROMPTS["general_inquiry"])

    # Build conversation history string
    history_text = ""
    if conversation_history:
        history_text = "\n\nPrevious conversation:\n"
        for turn in conversation_history[-6:]:  # last 6 turns only
            history_text += f"Customer: {turn['query']}\n"
            history_text += f"Agent: {turn['response']}\n"

    final_prompt = f"""
{system_prompt}
{history_text}

Current customer message: {query}

Important: Use the conversation history above to maintain context.
If the customer refers to something mentioned earlier, use that context in your response.
"""
    return final_prompt


def route_and_respond(query):
    intent = classify_intent(query)
    final_prompt = build_prompt_with_memory(query, intent)
    time.sleep(2)
    response = ask_llm(final_prompt)
    return intent, response


def save_log():
    if log:
        df = pd.DataFrame(log)
        df.to_csv("agent_log_memory.csv", index=False)
        print(f"\nSession saved to agent_log_memory.csv ({len(log)} interactions logged)")
    else:
        print("\nNo interactions to save.")


# Main chat loop
print("=" * 60)
print("   Retail Agent with Memory")
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

    # Add to memory
    conversation_history.append({
        "query": user_input,
        "response": reply
    })

    # Add to log
    log.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "query": user_input,
        "intent": intent,
        "response": reply,
        "memory_turns": len(conversation_history)
    })

    time.sleep(1)