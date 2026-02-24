
from langchain_huggingface import HuggingFaceEmbeddings
import os

# from langchain_classic.document_loaders import WebBaseLoader
from langchain_classic.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
# from langchain_classic.chains import RetrievalQA
from langchain.tools import tool

from langchain_tavily import TavilySearch

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_classic.agents import create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.agents import AgentExecutor
from pydantic import BaseModel

from fastapi import FastAPI,Depends

load_dotenv()

app=FastAPI()


class UserModel(BaseModel):
    user : str
    

# Load document using WebBaseloader document loader

#loader = WebBaseLoader("https://www.icici.bank.in/personal-banking/help?ITM=nli_imobileFaqs_waysToBank_mobileBanking_imobileFaqs_headercomponent_222_CMS_help_informationCenter_NLI")

from langchain_community.document_loaders import UnstructuredURLLoader

urls = [
    "https://www.icici.bank.in/personal-banking/help?ITM=nli_imobileFaqs_waysToBank_mobileBanking_imobileFaqs_headercomponent_222_CMS_help_informationCenter_NLI"
]

loader = UnstructuredURLLoader(urls=urls)

documents = loader.load()

# Split document in chunks
text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=200, separator="\n")

docs = text_splitter.split_documents(documents=documents)
embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create vectors
vectorstore = FAISS.from_documents(documents=docs, embedding=embeddings)
retriever=vectorstore.as_retriever()

def user_query_chat(query):
    @tool
    def rag_tool(query: str) -> str:
        """_summary_
        
        Args:
            query (str): This is a query from user
            
        Returns:
            str: It must be string 
        """
        # print("Rag Tool invoking with query:- ", query)
        docs=retriever.invoke(query)
        
        if not docs:
            return "Not found"
        # retrived_text = "\n".join([docs[i].page_content for i in range(len(docs))])
        # print("retrived text:- ", retrived_text)
        
        return "\n".join([docs[i].page_content for i in range(len(docs))])
    
    web_tool= TavilySearch(max_results=3,topic='finance')
    
    @tool
    def web_search(query: str) -> str:
        """_summary_

        Args:
            query (str): This is a query

        Returns:
            str: must be string format
        """
        
        result=web_tool.invoke(query)['results'][0]['content']
        
        
        if not result:
            return "No websearch results"
        
        return str(result)

    my_prompt="""
        YOUR ARE THE ICICI BANK CUSTOMER SUPPORT AGENT.
        FOLLOW THE BELOW RULES STRICTLY :
        1. If the user ask greetings related questions than RESPONDE. DONT call tools for greetings.
        Example : 
        User : Hi/Hello
        Agent : Hello! How can I assist you with ICICI Bank services today?
        2. Search ICICI BANK Related content in the docs.DONT HALLUCINATE.
        3. IF YOUR QUESTION IS NOT RELATED TO THE BANK/ACCOUNT THAN SIMPLY RESPONDE "I can help with ICICI Bank services and banking queries. Please ask a related question."

        4. First Go to user query to rag_tool if found the answer than GET THAT DOC INFO ONLY,DONT GIVE AGENT/LLM BASED ANSWER AND DONT BE HALLUCINATE.
        5. If The information from docs not AVAILABLE then Fetch the external information USING web_search tool
          BELOW STRICT RULES FOLLOW FOR EXTERNAL INFORMATION:
          ** GET ONLY ICICI BANK RELATED KNOWLEDGE
          ** RBI BASED KNOWLEDGE FOR ICICI 
          ** UPDATED BANKING REGULATIONS FOR ICICI
          ** RECENT ICICI BANK SERVICE OUTAGES
        6. If No websearch results → provide guidance & suggest contacting bank support
        7. FINALLY DONT BE HALLUCINATE.
    """
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", my_prompt),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", f"{query}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )
    tools=[rag_tool,web_search]

    llm=ChatGroq(
        groq_api_key=os.environ['GROQ_API_KEY'],
        model="moonshotai/kimi-k2-instruct-0905",
        )
    
    agent=create_openai_tools_agent(llm,tools,prompt)
    
    agent_executor=AgentExecutor(agent=agent,tools=tools,verbose=True)
    
    # print("Query that passed to Agent:- ", {f"{query}":f"{query}"})
    result=agent_executor.invoke({f"{query}":f"{query}"})
    
    
    return (result['output'])

@app.post('/chatwithAgent/')
def user_question(query: str) -> str:
    return user_query_chat(query)
