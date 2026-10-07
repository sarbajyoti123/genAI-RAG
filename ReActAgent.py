#ReactAgent:
#use llm to bind and call custom tools like internet search, wiki search or other stuff
# then it response back with the tool name
# then again using langchain create_agent it use that returned tool to search the query
# and use that returned query in llm to fine tune the reurned data


#library used to call the LLM model
from langchain_google_genai import ChatGoogleGenerativeAI

#library used to perform an internet search using DuckDuckGo and return the results
from langchain_community.tools import DuckDuckGoSearchResults

#library to create custom tools that can be used by the LLM to perform specific tasks
from langchain.tools import tool

from dotenv import load_dotenv
load_dotenv()  #call the load_dotenv() function to load the environment variables from the .env file

# import langchain
# print(langchain.__version__) #1.4.3

# LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview", temperature = 0
)
# response = llm.invoke(prompt)

#tool to perform an internet search using DuckDuckGo and return the results
# def tool_internet_search(query: str) -> str:
#     """Tool to perform an internet search using DuckDuckGo and return the results."""
#     tool = DuckDuckGoSearchResults()
#     return tool.invoke(query)

# print(tool_internet_search("What is the latest news on AI?"))  # Print the text of the response


@tool
def tool_internet_search(query: str) -> str:
    """Tool to perform an internet search using DuckDuckGo and return the latest results."""
    tool = DuckDuckGoSearchResults(num_results=2)
    return tool.invoke(query)

@tool
def tool_call_my_rag(query: str) -> str:
    #library used to store the embeddings in a vector database
    from langchain_chroma import Chroma

    #library used to call the embedding model
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    # Embedding model calling
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2-preview"
    ) 

    # chroma db connection to fetch the embeddings from the database
    chroma_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings) 

    # chroma similarity search to find the most relevant chunks for a given query
    # k is the number of chunks to return
    chroma_results = chroma_db.similarity_search(query, k=3)
    context = "\n\n".join(
            [doc.page_content for doc in chroma_results]
        )
    return context

tool_list = [tool_call_my_rag,tool_internet_search]

llm_bind =  llm.bind_tools(tool_list)  # Bind the tools to the LLM

#from this reponse, you can see that the llm tried to use the given tools and got a tool
# where the agent will call next to get the result
# data =  llm_bind.invoke("What is the latest news on AI?")  # Invoke the LLM with the tools bound
# print(data)  # Print the text of the response)
# print(data.content[0]["text"])  # Print the text of the response)

from langchain.agents import create_agent
react_agent = create_agent(llm_bind, tool_list)  # Create a ReAct agent with the LLM and tools bound

response = react_agent.invoke(
    {

        "messages":[
            {
                "role": "user",
                "content": "Give AI news in 3 bullet points only. only in 30 words."
            }
        ]
    }


)  # Invoke the ReAct agent with a query
# Print the exact answer of the response
print(response["messages"][-1].content)  # Print the text of the response

