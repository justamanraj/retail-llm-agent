import google.generativeai as genai
import pandas as pd
import time

genai.configure(api_key="YOUR_API_KEY_HERE")
model = genai.GenerativeModel("gemini-3.5-flash")

def ask_llm(prompt):
    time.sleep(15)
    response = model.generate_content(prompt)
    return response.text.strip()


def evaluate_response(query, intent, response):
    prompt = f"""
You are a strict quality evaluator for a retail customer support AI agent.
You are NOT the agent. You are an independent auditor.

Be critical and realistic. Most responses have flaws.
A score of 5 means absolutely perfect — rare.
A score of 3 means acceptable but has clear room for improvement.
A score of 1 or 2 means the response failed the customer.
Evaluate the following agent response on 3 criteria.
Score each from 1 to 5.

Query: {query}
Detected Intent: {intent}
Agent Response: {response}

Criteria:
1. Relevance - Did the response directly address the customer query?
2. Accuracy - Is the information provided correct and realistic?
3. Helpfulness - Would a real customer be satisfied with this response?

Respond in EXACTLY this format, nothing else:
relevance: X
accuracy: X
helpfulness: X
"""
    result = ask_llm(prompt)
    return result


def parse_scores(eval_text):
    scores = {"relevance": 0, "accuracy": 0, "helpfulness": 0}
    try:
        for line in eval_text.strip().split("\n"):
            if "relevance:" in line:
                scores["relevance"] = int(line.split(":")[1].strip())
            elif "accuracy:" in line:
                scores["accuracy"] = int(line.split(":")[1].strip())
            elif "helpfulness:" in line:
                scores["helpfulness"] = int(line.split(":")[1].strip())
    except:
        pass
    return scores


# Load your saved log
df = pd.read_csv("agent_log.csv")

print("Running Evaluation Framework")
print("=" * 60)

results = []

for _, row in df.iterrows():
    print(f"\nEvaluating: {row['query']}")

    eval_text = evaluate_response(row['query'], row['intent'], row['response'])
    scores = parse_scores(eval_text)

    print(f"Relevance  : {scores['relevance']}/5")
    print(f"Accuracy   : {scores['accuracy']}/5")
    print(f"Helpfulness: {scores['helpfulness']}/5")
    print("-" * 60)

    results.append({
        "query": row["query"],
        "intent": row["intent"],
        "relevance": scores["relevance"],
        "accuracy": scores["accuracy"],
        "helpfulness": scores["helpfulness"]
    })

    time.sleep(4)

# Save evaluated results
eval_df = pd.DataFrame(results)
eval_df.to_csv("agent_log_evaluated.csv", index=False)

# Print summary
print("\nSummary by Intent")
print("=" * 60)
summary = eval_df.groupby("intent")[["relevance", "accuracy", "helpfulness"]].mean().round(2)
print(summary)

print("\nOverall Averages")
print(f"Relevance  : {eval_df['relevance'].mean():.2f}/5")
print(f"Accuracy   : {eval_df['accuracy'].mean():.2f}/5")
print(f"Helpfulness: {eval_df['helpfulness'].mean():.2f}/5")

print("\nEvaluation saved to agent_log_evaluated.csv")