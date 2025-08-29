import google.generativeai as genai
import os
from dotenv import load_dotenv, find_dotenv

# Load .env
load_dotenv(find_dotenv(), override=True)

api_key = os.getenv("GEMINI_API_KEY")
model_name = os.getenv("GEMINI_MODEL_NAME")

genai.configure(api_key=api_key)
model = genai.GenerativeModel(model_name)

def main():
    msg = input("Enter your question: ").strip()
    try:
        response = model.generate_content(msg)
        print(f"\nGemini Response: {response.text}")
    except Exception as ex:
        print(f"\nError: {ex}")

if __name__ == "__main__":
    main()
