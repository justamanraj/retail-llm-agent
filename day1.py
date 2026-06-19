import google.generativeai as genai
import time
# Configure your API key
genai.configure(api_key="YOUR_API_KEY_HERE")

# Load the model
model = genai.GenerativeModel("gemini-3.5-flash")

# Function to ask the model anything
def ask(query):
    response = model.generate_content(query)
    print("Query:", query)
    print("Response:", response.text)
    print("-" * 50)
    time.sleep(15)
# Test with 5 retail queries
ask("What is a good laptop under 30000 rupees?")
ask("How do I track my online order?")
ask("What is the return policy for electronics?")
ask("Suggest a good smartphone for a student")
ask("What are the best deals on shoes right now?")