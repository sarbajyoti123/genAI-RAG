#library used to store the embeddings in a vector database
from langchain_chroma import Chroma

#library used to call the embedding model
from langchain_google_genai import GoogleGenerativeAIEmbeddings

#library used to call the LLM model
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv
load_dotenv()  #call the load_dotenv() function to load the environment variables from the .env file

# Embedding model calling
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview"
) 

# chroma db connection to fetch the embeddings from the database
chroma_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings) 

# chroma similarity search to find the most relevant chunks for a given query
# k is the number of chunks to return
chroma_results = chroma_db.similarity_search("What is the main topics ?", k=3)
# print(chroma_results[0].page_content)  # print the content of the most relevant chunk
# print(chroma_results[1].page_content)  # print the content of the most relevant chunk
# print(chroma_results[2].page_content)  # print the content of the most relevant chunk

for i, result in enumerate(chroma_results):
    print(f"Result {i+1}:")
    print(result.page_content)
    print("-" * 50)  # separator for better readability

#call langchain google geai to process the results and generate a response

def ask_rag(question):
    # Retrieve relevant chunks
    docs = chroma_db.similarity_search(question, k=3)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
    You are a sarbajyoti's assistant.

    Use only the context below to answer.

    Context:
    {context}

    Question:
    {question}
    """
    # LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview"
    )
    response = llm.invoke(prompt)

    return response.content[0]["text"]  # return the text of the response


answer = ask_rag(
    "What are the main topics of the document?"
)

print(f"Answer: {answer}")  # print the content of the most relevant chunk