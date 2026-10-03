import os

from google import genai
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
# Initialize Gemini client

if os.environ.get("GEMINI_API_KEY"):
    print("GEMINI_API_KEY is set in the environment.")



client = genai.Client()
# Generate content using Gemini 3 Flash Preview
response = client.models.generate_content(
   model="gemini-3-flash-preview",
   contents="Explain how AI works in a few words"
)
print(response.text)

#https://platform.openai.com/tokenizer