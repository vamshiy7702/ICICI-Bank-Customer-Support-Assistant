import os
import re
from typing import List, Tuple 
import faiss
import numpy as np
import requests
import streamlit as st
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain_text_splitters import RecursiveCharacterTextSplitter 

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY is missing from .env file.")
    st.stop()

if not TAVILY_API_KEY:
    st.error("TAVILY_API_KEY is missing from .env file.")
    st.stop()


SYSTEM_PROMPT = """
You are an expert ICICI Bank Customer Support Assistant.

Your goal is to provide accurate, concise, secure and helpful
responses for ICICI Bank and general banking/finance queries.

STRICT RULES:

1. GREETINGS
If the user says Hi, Hello, Hey, Good Morning, Good Afternoon,
Good Evening, Namaste, etc., respond directly:

"Hello! How can I assist you with ICICI Bank services today?"

Do not use any tool.

2. DOMAIN
Only answer questions related to:
- ICICI Bank
- Banking
- Accounts
- Cards
- Loans
- UPI
- Net Banking
- Mobile Banking
- iMobile
- ATM
- Transactions
- RBI banking regulations
- Banking charges
- Interest
- KYC
- Deposits
- Withdrawals
- NEFT
- RTGS
- IMPS
- Bank statements
- PIN
- Branch services

For unrelated questions such as coding, movies, shopping,
sports, entertainment, etc., respond:

"I can help with ICICI Bank services and banking queries.
Please ask a related question."

3. KNOWLEDGE SEARCH
For banking questions, use the knowledge_search tool.

The tool internally follows this order:

RAG Knowledge Base
        ↓
If sufficient information exists
        ↓
Return RAG information

If RAG does not contain enough information
        ↓
Tavily Web Search
        ↓
Return recent/relevant information

4. RAG ANSWERS
When the knowledge base provides the answer:
- Use only the retrieved information.
- Do not add outside knowledge.
- Do not hallucinate.
- Use bullet points for procedures or steps.

5. WEB SEARCH ANSWERS
When web search information is returned:
- Prefer ICICI Bank official information.
- Prefer RBI official information for regulations.
- Recent service outages or banking updates may use
  reliable external sources.
- Mention the source URL when available.
- Do not invent citations.

6. FALLBACK
If the knowledge search tool cannot find valid information,
say:

"I couldn't find specific information on this query.
For accurate assistance, please contact ICICI Bank Customer
Care at 1860 120 7777 or visit your nearest branch."

7. SECURITY
Never ask the user for:
- OTP
- CVV
- Full card number
- PIN
- Internet banking password
- UPI PIN

8. TONE
Professional, secure, concise and customer-focused.
"""

KB_URL = (
    "https://www.icici.bank.in/personal-banking/help"
    "?ITM=nli_imobileFaqs_waysToBank_mobileBanking_imobileFaqs_"
    "headercomponent_222_CMS_help_informationCenter_NLI")

# ============================================================
# 4. SIMPLE FAISS RETRIEVER
#    Uses FAISS directly instead of langchain-community
# ============================================================

class SimpleFAISSRetriever:
    def __init__(
        self,
        documents: List[Document],
        embeddings: HuggingFaceEmbeddings,
        k: int = 3,
        score_threshold: float = 0.50,):

        self.documents = documents
        self.embeddings = embeddings
        self.k = k
        self.score_threshold = score_threshold

        # Create document embeddings
        texts = [doc.page_content for doc in documents]

        vectors = embeddings.embed_documents(texts)

        vectors = np.array(vectors, dtype=np.float32)

        # Normalize vectors for cosine similarity
        faiss.normalize_L2(vectors)

        dimension = vectors.shape[1]

        # Inner Product on normalized vectors = cosine similarity
        self.index = faiss.IndexFlatIP(dimension)

        self.index.add(vectors)

    def invoke(self, query: str) -> List[Tuple[Document, float]]:
        query_vector = self.embeddings.embed_query(query)

        query_vector = np.array(
            [query_vector],
            dtype=np.float32,
        )

        faiss.normalize_L2(query_vector)

        scores, indices = self.index.search(
            query_vector,
            self.k,
        )

        results = []

        for score, index in zip(scores[0], indices[0]):

            if index == -1:
                continue

            if float(score) >= self.score_threshold:

                results.append(
                    (
                        self.documents[index],
                        float(score),
                    )
                )

        return results


# ============================================================
# 5. LOAD KNOWLEDGE BASE + EMBEDDINGS + FAISS
# ============================================================

@st.cache_resource
def initialize_resources():

    try:

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 "
                "(KHTML, like Gecko) "
                "Chrome/153.0.0.0 Safari/537.36"
            )
        }

        response = requests.get(
            KB_URL,
            headers=headers,
            timeout=30,
        )

        response.raise_for_status()

        # ----------------------------------------------------
        # Parse HTML
        # ----------------------------------------------------

        soup = BeautifulSoup(
            response.text,
            "html.parser",
        )

        for element in soup(
            ["script", "style", "noscript"]
        ):
            element.decompose()

        text = soup.get_text(
            separator="\n",
            strip=True,
        )

        if not text:
            raise ValueError(
                "No text could be extracted from ICICI website."
            )

        # ----------------------------------------------------
        # Convert to LangChain Document
        # ----------------------------------------------------

        documents = [
            Document(
                page_content=text,
                metadata={
                    "source": KB_URL
                },
            )
        ]

        # ----------------------------------------------------
        # Split documents into chunks
        # ----------------------------------------------------

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
            separators=[
                "\n\n",
                "\n",
                " ",
                "",
            ],
        )

        chunks = splitter.split_documents(
            documents
        )

        # ----------------------------------------------------
        # HuggingFace Embeddings
        # ----------------------------------------------------

        embeddings = HuggingFaceEmbeddings(
            model_name=(
                "sentence-transformers/"
                "all-MiniLM-L6-v2"
            )
        )

        # ----------------------------------------------------
        # FAISS
        # ----------------------------------------------------

        retriever = SimpleFAISSRetriever(
            documents=chunks,
            embeddings=embeddings,
            k=3,
            score_threshold=0.50,
        )

        return retriever

    except Exception as e:

        st.error(
            f"Failed to initialize knowledge base: {e}"
        )

        return None

@st.cache_resource
def initialize_tavily():

    return TavilySearch(
        max_results=3,
        topic="general",
        search_depth="basic",
        include_answer=False,
        include_raw_content=False,
    )
# ============================================================
# 7. RAG SEARCH FUNCTION
# ============================================================

def run_rag_search(
    query: str,
    retriever: SimpleFAISSRetriever,
) -> str:

    try:

        results = retriever.invoke(query)

        if not results:
            return "RAG_NOT_FOUND"

        formatted_results = []

        for document, score in results:

            formatted_results.append(
                f"""
SOURCE: ICICI Bank Knowledge Base
SIMILARITY SCORE: {score:.3f}

CONTENT:
{document.page_content}
"""
            )

        return "\n\n".join(
            formatted_results
        )

    except Exception as e:

        return (
            f"RAG_SEARCH_ERROR: {str(e)}"
        )


# ============================================================
# 8. WEB SEARCH FUNCTION
# ============================================================

def run_web_search(query: str,tavily_search: TavilySearch) -> str:

    try:

        enhanced_query = f"""
ICICI Bank OR RBI banking information only.

User query:
{query}

Search for:
- ICICI Bank official information
- RBI official regulations
- Recent ICICI Bank service updates
- Relevant banking rules
- Recent outages or service issues
"""

        result = tavily_search.invoke(
            {
                "query": enhanced_query
            }
        )

        if not result:
            return "WEB_SEARCH_NOT_FOUND"

        search_results = result.get(
            "results",
            [],
        )

        if not search_results:
            return "WEB_SEARCH_NOT_FOUND"

        formatted_results = []

        for item in search_results:

            title = item.get(
                "title",
                "Unknown source",
            )

            url = item.get(
                "url",
                "",
            )

            content = item.get(
                "content",
                "",
            )

            formatted_results.append(
                f"""
TITLE:
{title}

URL:
{url}

CONTENT:
{content}
"""
            )

        return "\n\n".join(
            formatted_results
        )

    except Exception as e:

        return (
            f"WEB_SEARCH_ERROR: {str(e)}"
        )


# ============================================================
# 9. COMBINED KNOWLEDGE SEARCH TOOL
#
# Important:
# Agent gets ONE tool.
# This guarantees:
#
# RAG FIRST
#    ↓
# insufficient?
#    ↓
# Tavily
#
# ============================================================

def build_knowledge_search_tool(
    retriever: SimpleFAISSRetriever,
    tavily_search: TavilySearch,
):

    @tool
    def knowledge_search(query: str) -> str:
        """
        Search ICICI Bank knowledge.

        The tool ALWAYS searches the internal ICICI
        knowledge base first. If the internal knowledge
        base does not contain sufficient information,
        the tool performs a Tavily web search.
        """

        # ----------------------------------------------------
        # FIRST: RAG
        # ----------------------------------------------------

        rag_result = run_rag_search(
            query=query,
            retriever=retriever,
        )

        if (
            rag_result
            and not rag_result.startswith(
                "RAG_NOT_FOUND"
            )
            and not rag_result.startswith(
                "RAG_SEARCH_ERROR"
            )
        ):

            return f"""
SEARCH_SOURCE: INTERNAL_RAG

Use ONLY the following ICICI Bank knowledge:

{rag_result}
"""

        # ----------------------------------------------------
        # SECOND: WEB SEARCH
        # ----------------------------------------------------

        web_result = run_web_search(
            query=query,
            tavily_search=tavily_search,
        )

        if (
            web_result
            and not web_result.startswith(
                "WEB_SEARCH_NOT_FOUND"
            )
            and not web_result.startswith(
                "WEB_SEARCH_ERROR"
            )
        ):

            return f"""
SEARCH_SOURCE: WEB

The internal ICICI knowledge base did not contain
sufficient information.

Use the following web results:

{web_result}
"""

        # ----------------------------------------------------
        # FALLBACK
        # ----------------------------------------------------

        return """
NO_VALID_INFORMATION_FOUND

I couldn't find specific information on this query.
For accurate assistance, please contact ICICI Bank
Customer Care at 1860 120 7777 or visit your nearest branch.
"""

    return knowledge_search


# ============================================================
# 10. INITIALIZE GROQ + LANGCHAIN AGENT
# ============================================================

def initialize_agent(
    retriever: SimpleFAISSRetriever,
    tavily_search: TavilySearch,
):

    knowledge_search = build_knowledge_search_tool(
        retriever=retriever,
        tavily_search=tavily_search,
    )
    # Current Groq model
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model="openai/gpt-oss-120b",
        temperature=0.2,
    )

    # Current LangChain agent API
    agent = create_agent(
        model=llm,
        tools=[
            knowledge_search
        ],
        system_prompt=SYSTEM_PROMPT,
    )

    return agent


# ============================================================
# 11. EXTRACT FINAL AI TEXT
# ============================================================

def extract_final_response(result) -> str:

    messages = result.get(
        "messages",
        [],
    )

    if not messages:
        return (
            "I couldn't retrieve the information. "
            "Please contact ICICI Customer Care."
        )

    # Search backwards for final AI response
    for message in reversed(messages):

        if isinstance(
            message,
            AIMessage,
        ):

            content = message.content

            if isinstance(
                content,
                str,
            ):
                return content.strip()

            # Handle structured content blocks
            if isinstance(
                content,
                list,
            ):

                text_parts = []

                for block in content:

                    if isinstance(
                        block,
                        dict,
                    ):

                        if block.get(
                            "type"
                        ) == "text":

                            text_parts.append(
                                block.get(
                                    "text",
                                    "",
                                )
                            )

                if text_parts:
                    return "\n".join(
                        text_parts
                    ).strip()

    return (
        "I couldn't retrieve the information. "
        "Please contact ICICI Customer Care."
    )


# ============================================================
# 12. GREETING DETECTION
# ============================================================

def is_greeting(
    user_input: str,
) -> bool:

    greeting_pattern = re.compile(
        r"""
        ^
        (
            hi
            |hello
            |hey
            |howdy
            |namaste
            |good\s+morning
            |good\s+afternoon
            |good\s+evening
        )
        [!,. ]*
        $
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    return bool(
        greeting_pattern.match(
            user_input.strip()
        )
    )


# ============================================================
# 13. DOMAIN VALIDATION
# ============================================================

def is_banking_related(
    user_input: str,
) -> bool:

    banking_keywords = [

        # Bank
        "icici",
        "bank",
        "branch",
        "banking",

        # Accounts
        "account",
        "savings",
        "current account",
        "balance",
        "statement",
        "deposit",
        "withdraw",
        "withdrawal",

        # Cards
        "credit card",
        "debit card",
        "card",
        "cvv",
        "pin",

        # Banking channels
        "net banking",
        "internet banking",
        "mobile banking",
        "imobile",
        "upi",

        # Transactions
        "transaction",
        "transfer",
        "neft",
        "rtgs",
        "imps",

        # Loans
        "loan",
        "emi",
        "interest rate",

        # Regulations
        "rbi",
        "kyc",
        "regulation",

        # Charges
        "fee",
        "fees",
        "charge",
        "charges",

        # Money
        "money",
        "payment",
        "interest",
        "refund",
        "cash",

        # ATM
        "atm",
    ]

    query = user_input.lower()

    return any(
        keyword in query
        for keyword in banking_keywords
    )


# ============================================================
# 14. STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="ICICI Bank Support Agent",
    page_icon="🏦",
    layout="centered",
)

st.title(
    "🏦 ICICI Bank Customer Support Assistant"
)

st.caption(
    "Powered by LangChain • RAG • FAISS • Tavily • Groq"
)


# ============================================================
# 15. SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# 16. DISPLAY OLD CHAT
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# 17. USER INPUT
# ============================================================

if user_input := st.chat_input(
    "Ask about ICICI Bank services, accounts, cards, loans, UPI..."):

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input,
        }
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    if is_greeting(
        user_input
    ):

        response = (
            "Hello! How can I assist you with "
            "ICICI Bank services today?"
        )

    # --------------------------------------------------------
    # Domain validation
    # --------------------------------------------------------

    elif not is_banking_related(
        user_input
    ):

        response = (
            "I can help with ICICI Bank services "
            "and banking queries. Please ask a related question."
        )

    # --------------------------------------------------------
    # Agent
    # --------------------------------------------------------

    else:

        with st.chat_message(
            "assistant"
        ):

            with st.spinner(
                "🔍 Searching ICICI Knowledge Base..."
            ):

                try:
                    retriever = initialize_resources()

                    if retriever is None:

                        raise RuntimeError(
                            "Knowledge base initialization failed."
                        )

                    tavily_search = (
                        initialize_tavily()
                    )

                    agent = initialize_agent(
                        retriever=retriever,
                        tavily_search=tavily_search,
                    )

                    recent_messages = (
                        st.session_state.messages[-12:]
                    )

                    chat_messages = []

                    for message in recent_messages:

                        if message["role"] == "user":

                            chat_messages.append(
                                HumanMessage(
                                    content=message[
                                        "content"
                                    ]
                                )
                            )

                        elif message["role"] == "assistant":

                            chat_messages.append(
                                AIMessage(
                                    content=message[
                                        "content"
                                    ]
                                )
                            )

                    # ----------------------------------------
                    # Run current LangChain agent
                    # ----------------------------------------

                    result = agent.invoke(
                        {
                            "messages": chat_messages
                        }
                    )

                    # ----------------------------------------
                    # Extract final answer
                    # ----------------------------------------

                    response = extract_final_response(
                        result
                    )

                except Exception as e:

                    response = (
                        "⚠️ An error occurred while processing "
                        "your query. Please try again.\n\n"
                        f"`{str(e)}`"
                    )

            st.markdown(response)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response,
        }
    )
