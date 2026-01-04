import streamlit as st
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # Same imports!
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from mfg_ai_agent import model as InfoPal
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
#from mfg_ai_agent import embedds as InfoPalEmbeddings
from pathlib import Path

# --- CONFIGURATION ---
st.set_page_config(page_title="InfoPal", page_icon="🤖")
st.title("🤖 InfoPal")
st.write("Making onboarding easier for everyone.")
FAISS_INDEX_PATH = "faiss_index.faiss"
FALLBACK_RESPONSE = "I am sorry, I don't know how to answer that specific question just yet, as i could not find that specific fact in my knowledge base."

# LM Studio config - NO API KEY NEEDED!

#@st.cache_resource
# def setup_rag_pipeline():
#     """Loads documents and tests LM Studio connection."""
#     # try:
#     #     # Test embeddings endpoint
#     #     embedding_model = OpenAIEmbeddings(
#     #         check_embedding_ctx_length=False,
#     #         base_url="http://127.0.0.1:1234/v1",
#     #         api_key="lm-studio",  # Dummy key - LM Studio ignores it
#     #         model="text-embedding-nomic-embed-text-v1.5"
#     #     )
#     #     # Test will fail gracefully if LM Studio not running
#     #     st.sidebar.success("✅ LM Studio connected!")
#     #     return embedding_model
#     # except:
#     #     st.sidebar.error("❌ LM Studio not running at 1234")
#     #     st.error("Start LM Studio server first!")
#     #     return None

#     embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
#     return embedding_model

# Load documents (same as before)
@st.cache_resource
def setup_rag_pipeline():
    """Loads the FAISS index from the file system."""
    
    # 1. Initialize the embedding model (needed for loading)
    try:
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    except Exception as e:
        st.error(f"❌ Failed to load embedding model: {e}")
        return None

    # 2. Check and Load the existing index
    if not os.path.exists(FAISS_INDEX_PATH):
        st.error(f"❌ FAISS index not found at '{FAISS_INDEX_PATH}'!")
        st.info("Please run `python create_database.py` first to generate the index.")
        st.stop() # Stop the app gracefully
        return None
    
    try:
        vectorstore = FAISS.load_local(
            FAISS_INDEX_PATH, 
            embedding_model, 
            allow_dangerous_deserialization=True
        )
        st.sidebar.success("✅ Loaded FAISS index from disk!")
        
        # We can't know the sources used, so we'll just display the folder contents
        documents_dir = Path("./documents")
        txt_files = list(documents_dir.glob("*.txt"))
        st.sidebar.caption("📁 Sources:")
        for f in txt_files:
            st.sidebar.caption(f"• {f.name}")
            
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        return retriever
        
    except Exception as e:
        st.error(f"❌ RAG loading error. Index might be corrupted. Delete '{FAISS_INDEX_PATH}' and re-run indexer. Error: {e}")
        return None

retriever = setup_rag_pipeline()

def create_rag_chain(_retriever):
    """LCEL chain with LM Studio."""
    if not _retriever:
        return None
    
    # LM Studio ChatOpenAI 
    # llm = init_chat_model(
    #     base_url="http://127.0.0.1:1234/v1",
    #     api_key="lm-studio",  # Dummy key
    #     model_provider="openai",
    #     model="qwen3-vl-4b",  # Your loaded model
    #     temperature=0.3
    # )

    llm = InfoPal
    
    RAG_PROMPT = """
You are InfoPal, a highly knowledgeable and polite chatbot that works for Tata Consultancy Services.
Your primary goal is to answer the user's questions based *only* on the provided context.

Context:
{context}

If the answer cannot be found in the context, you must politely respond: 
"I am sorry, I don't know how to answer that specific question just yet, as i could not find that specific fact in my knowledge base."

Question: {input}
    """
    
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
        {"context": _retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

rag_chain = create_rag_chain(retriever)

# Rest of your code stays EXACTLY the same...
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Whether you have questions about TCS LATAM policies, work-life balance, disconnection guidelines, or anything else, I’m here to help. 😊"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about TCS processes..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if rag_chain:
        with st.chat_message("assistant"):
            with st.spinner("thinking..."):
                response_text = rag_chain.invoke(prompt)
                st.markdown(response_text)
            
            st.session_state.messages.append({"role": "assistant", "content": response_text})
