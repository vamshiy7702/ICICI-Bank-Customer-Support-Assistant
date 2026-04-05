import os
import re
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

# 📝 Optimized System Prompt
SYSTEM_PROMPT = """You are an expert ICICI Bank Customer Support Assistant. Your goal is to provide accurate, concise, and helpful responses based strictly on the provided tools.

STRICT OPERATING RULES:
1. GREETINGS: If the user says "Hi", "Hello", "Hey", etc., respond directly with: "Hello! How can I assist you with ICICI Bank services today?" DO NOT call any tools.
2. DOMAIN VALIDATION: Only answer ICICI Bank and general banking/finance queries. If unrelated (e.g., coding, movies, shopping), respond: "I can help with ICICI Bank services and banking queries. Please ask a related question." DO NOT call tools.
3. TOOL EXECUTION ORDER: ALWAYS use `rag_tool` FIRST. Only if it returns "Not found", insufficient data, or lacks the specific answer, then use `web_search`.
4. RAG RESPONSE: If `rag_tool` provides relevant information, use ONLY that information to answer. Do not add external knowledge or hallucinate. Use bullet points for steps.
5. WEB SEARCH RESPONSE: If using `web_search`, only extract ICICI-related info, RBI guidelines affecting ICICI, recent outages, or updated banking regulations. Summarize clearly and cite the source if possible.
6. FALLBACK: If neither tool provides a valid answer, respond: "I couldn't find specific information on this query. For accurate assistance, please contact ICICI Bank Customer Care at 1860 120 7777 or visit your nearest branch."
7. TONE: Professional, secure, and customer-focused."""

@st.cache_resource
def initialize_icici_agent():
    """Loads knowledge base, creates vector store, defines tools, and initializes the LangChain agent."""
    try:
        # 1️⃣ Load & Split Knowledge Base
        kb_url = "https://www.icici.bank.in/personal-banking/help?ITM=nli_imobileFaqs_waysToBank_mobileBanking_imobileFaqs_headercomponent_222_CMS_help_informationCenter_NLI"
        loader = WebBaseLoader(
            web_paths=(kb_url,),
            header_template={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        )
        documents = loader.load()
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200, separators=["\n\n", "\n", " ", ""])
        chunks = splitter.split_documents(documents)

        # 2️⃣ Vector Store
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3, "search_type": "similarity_score_threshold", "score_threshold": 0.5})

        # 3️⃣ Define Tools
        @tool
        def rag_tool(query: str) -> str:
            """Search ICICI Bank internal knowledge base for FAQs, account services, net/mobile banking instructions, charges, and policies."""
            results = retriever.invoke(query)
            if not results:
                return "Not found"
            return "\n\n".join([doc.page_content for doc in results])

        tavily_search = TavilySearchResults(max_results=3, topic="finance")

        @tool
        def web_search(query: str) -> str:
            """Search the web for latest RBI guidelines, recent ICICI service outages, new feature announcements, or updated banking regulations when internal KB lacks answers."""
            try:
                results = tavily_search.invoke(query)
                if not results:
                    return "No websearch results"
                contents = [r.get("content", "") for r in results if r.get("content")]
                return "\n\n".join(contents) if contents else "No relevant web results found."
            except Exception:
                return "Web search failed."

        # 4️⃣ LLM & Agent Setup
        # Note: Replace model string if your Groq account uses a different model name
        llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model="qwen/qwen3-32b",
            temperature=0.7
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ])

        tools = [rag_tool, web_search]
        agent = create_tool_calling_agent(llm, tools, prompt)
        
        return AgentExecutor(agent=agent, 
                             tools=tools,
                               verbose=False, 
                               handle_parsing_errors=True,
                               max_iterations=3,
                               max_execution_time=30)
    except Exception as e:
        st.error(f"⚠️ Failed to initialize Agent: {e}")
        return None

# 🖥️ Streamlit UI
st.set_page_config(page_title="ICICI Bank Support Agent", page_icon="🏦", layout="centered")
st.title("🏦 ICICI Bank Customer Support Assistant")
st.caption("Powered by LangChain • RAG Knowledge Base • Tavily Web Fallback • Groq LLM")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 💬 Chat Input & Decision Logic
if user_input := st.chat_input("Ask about ICICI Bank services, accounts, cards, or banking guidelines..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Step 1: Greeting Detection (No Tool Call)
    greeting_pattern = re.compile(r"^(hi|hello|hey|good\s*(morning|afternoon|evening)|namaste|howdy)$", re.IGNORECASE)
    is_greeting = bool(greeting_pattern.match(user_input.strip()))

    # Step 2: Domain Validation
    banking_keywords = [
        "icici", "bank", "account", "card", "loan", "net banking", "mobile app", 
        "upi", "balance", "transaction", "atm", "rbi", "fee", "charge", "interest", 
        "imobile", "money", "statement", "pin", "kyc", "branch", "deposit", "withdraw"
    ]
    is_banking_related = any(kw in user_input.lower() for kw in banking_keywords)

    if is_greeting:
        response = "Hello! How can I assist you with ICICI Bank services today?"
    elif not is_banking_related:
        response = "I can help with ICICI Bank services and banking queries. Please ask a related question."
    else:
        # Step 3 & 4: Agent Execution (RAG → Web Search Fallback)
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching ICICI Knowledge Base & Web..."):
                try:
                    executor = initialize_icici_agent()
                    
                    # Format chat history for LangChain
                    chat_history = []
                    for m in st.session_state.messages[:-1]:
                        chat_history.append(HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]))
                        
                    result = executor.invoke({"input": user_input, "chat_history": chat_history})
                    response = result.get("output", "I couldn't retrieve the information. Please contact ICICI Customer Care: 1860 120 7777")
                except Exception as e:
                    response = f"⚠️ An error occurred while processing your query. Please try again.\n`{str(e)}`"

    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)