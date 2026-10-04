
# PDF without IMAGE--------------------->
# PDF
#  ↓
# PyPDFLoader
#  ↓
# documents
#  ↓
# RecursiveCharacterTextSplitter
#  ↓
# splits (small chunks)
#  ↓
# Gemini Embeddings
#  ↓
# Vector Database (FAISS/Chroma/Pinecone)
#  ↓
# Similarity Search
#  ↓
# Gemini Answer


# For modern enterprise RAG, a common pattern is:
# PDF
#  ↓
# Convert PDF Pages → Images
#  ↓
# Gemini Vision OCR
#  ↓
# Chunking
#  ↓
# Gemini Embeddings
#  ↓
# Chroma
#  ↓
# Retriever
#  ↓
# Gemini Answer

#library used to call the Gemini API
from langchain_google_genai import ChatGoogleGenerativeAI
#library used for pdf loading
from langchain_community.document_loaders import PyPDFLoader
#library used to split the loaded pdf into smaller chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter
#library used to create embeddings from the smaller chunks
from langchain_google_genai import GoogleGenerativeAIEmbeddings
#library used to store the embeddings in a vector database
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma

import os

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# os.environ["GOOGLE_API_KEY"] = "YOUR_GEMINI_API_KEY"

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

# LLM
# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash"
# )



# Load single PDF
# loader = PyPDFLoader("document.pdf")
# documents = loader.load()
# print(documents[0].page_content)

#load multiple PDFs from a folder

documents = []

for file in os.listdir("./PDFS"):
    if file.endswith(".pdf"):
        path = os.path.join("./PDFS", file)

        loader = PyPDFLoader(path)
        documents.extend(loader.load())

print(f"Total pages: {len(documents)}")


#split loaded documents into smaller chunks
splitter = RecursiveCharacterTextSplitter(
    #for testing purposes, we can set a small chunk size and overlap
    chunk_size=40,
    chunk_overlap=30
    #we can also set a larger chunk size and overlap for production use
    # chunk_size=1000,
    # chunk_overlap=200

)
splits = splitter.split_documents(documents)

#check the number of splits and the content of the first split
print(f"Number of splits: {len(splits)}")
print(f"Content of first split: {splits[0].page_content}")

# Embedding model calling
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview"
) 
#start embeding of the splits
vector_store = embeddings.embed_documents([split.page_content for split in splits])

print(f"Number of embeddings: {len(vector_store)}")
#Number of embeddings: 18

print(vector_store[0][:10])  # first 10 values of the first embedding
#[0.0010302109, 0.01982421, 0.0093623325, -0.041581795, -0.019270731, -0.024562607, -0.00908324, 0.0127046835, 0.014576435, -0.03970208]

#length of first vector in the vector store
print(f"Length of first vector in the vector store: {len(vector_store[0])}")
#Length of first vector in the vector store: 3072
# Why create 3072 numbers?
# The model tries to capture:
# Meaning
# Context
# Topic
# Relationships between words
# Semantic similarity


#create a vector database using Chroma to store the embeddings
db = Chroma.from_documents(
    splits,
    embeddings,
    persist_directory="./chroma_db"
)
print("Vector database created and embeddings stored in Chroma.")
#now call the "call_rag.py" file to fetch the embeddings from the database and perform similarity search

# #chroma db connection to fetch the embeddings from the database
# chroma_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings) 

# #chroma similarity search to find the most relevant chunks for a given query
# #k is the number of chunks to return
# chroma_results = chroma_db.similarity_search("What is the main topic of the document?", k=3)
# print(chroma_results[0].page_content)  # print the content of the most relevant chunk


