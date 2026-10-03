import os

from google import genai

# Initialize client

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file so that the GEMINI_API_KEY cannot be hardcoded in the code
# Initialize Gemini client

if os.environ.get("GEMINI_API_KEY"):
    print("GEMINI_API_KEY is set in the environment.")

client = genai.Client()

# Generate embedding
response = client.models.embed_content(
    model="gemini-embedding-2",
    contents="What is the meaning of life?"
)

# Embedding vector
embedding = response.embeddings[0].values

print(f"Dimension: {len(embedding)}")
print(embedding[:10])  # first 10 values