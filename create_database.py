import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path

# --- CONFIGURATION ---
DOCUMENTS_DIR = Path("./documents")
FAISS_INDEX_PATH = "faiss_index.faiss" 
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

def create_and_save_index():
    """Loads documents, embeds them, and saves the FAISS index to disk."""
    print("--- RAG Indexing Process Started ---")
    
    if not DOCUMENTS_DIR.exists():
        print(f"ERROR: '{DOCUMENTS_DIR}' folder not found!")
        return
        
    txt_files = list(DOCUMENTS_DIR.glob("*.txt"))
    if not txt_files:
        print("ERROR: No .txt files in documents/!")
        return

    print(f"✅ Found {len(txt_files)} files. Starting loading and splitting...")
    
    # 1. Load and Split Documents
    docs = []
    for txt_file in txt_files:
        loader = TextLoader(str(txt_file), encoding='utf-8')
        file_docs = loader.load()
        for doc in file_docs:
            doc.metadata['source'] = str(txt_file.name)
        docs.extend(file_docs)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    splits = text_splitter.split_documents(docs)
    print(f"Split documents into {len(splits)} chunks.")

    # 2. Initialize and Embed
    print(f"⏳ Initializing HuggingFace Embeddings: {EMBEDDING_MODEL_NAME}...")
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    print("⏳ Creating FAISS Vector Store (Embedding Chunks)...")
    vectorstore = FAISS.from_documents(splits, embedding_model)
    
    # 3. Save the Index
    vectorstore.save_local(FAISS_INDEX_PATH)
    print(f"💾 SUCCESS: Index created and saved to {FAISS_INDEX_PATH}!")
    print("--- RAG Indexing Process Complete ---")

if __name__ == "__main__":
    create_and_save_index()