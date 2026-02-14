# rag_components.py
import streamlit as st
import os

# Flag to track if RAG is available
RAG_AVAILABLE = False

try:
    # Try importing the necessary libraries
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.docstore.document import Document
    RAG_AVAILABLE = True
except ImportError as e:
    # Detailed error logging
    print(f"RAG Import Error: {e}")
    HuggingFaceEmbeddings = None
    FAISS = None
    RecursiveCharacterTextSplitter = None
    Document = None

from medical_reference import MEDICAL_KNOWLEDGE_BASE, REFERENCE_RANGES

class MedLabRAG:
    def __init__(self):
        """Initialize the RAG system"""
        self.vector_store = None
        
        if not RAG_AVAILABLE:
            print("RAG Dependencies not met. Running in passthrough mode.")
            return

        self.setup_knowledge_base()

    def setup_knowledge_base(self):
        """Create vector store from the knowledge base string"""
        try:
            # 1. Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )
            docs = [Document(page_content=x) for x in text_splitter.split_text(MEDICAL_KNOWLEDGE_BASE)]
            
            # 2. Initialize Embeddings 
            # We use a try/except here specifically for the model download
            try:
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            except Exception as e:
                st.error(f"Error downloading AI model: {e}. Check internet connection.")
                return

            # 3. Create Vector Store
            self.vector_store = FAISS.from_documents(docs, embeddings)
            print("✅ RAG System Initialized Successfully")
            
        except Exception as e:
            print(f"❌ Failed to initialize RAG: {e}")

    def enhance_analysis(self, current_data: dict) -> str:
        """Generate RAG-based insights for abnormal values"""
        # If RAG is not loaded, return a default message
        if not RAG_AVAILABLE or not self.vector_store:
            return "Medical context unavailable (RAG libraries missing or initialization failed)."

        # Identify abnormalities
        abnormal_params = []
        for param, value in current_data.items():
            if param in REFERENCE_RANGES:
                ref = REFERENCE_RANGES[param]
                
                # standardized range check
                low, high = 0, 0
                if 'range' in ref:
                    low, high = ref['range']
                elif 'male' in ref:
                    low, high = ref['male'] # Default to male for generic check
                
                if low != 0 and high != 0:
                    if value < low:
                        abnormal_params.append(f"Low {param}")
                    elif value > high:
                        abnormal_params.append(f"High {param}")

        if not abnormal_params:
            return "Values appear within standard reference ranges. No specific medical context alerts."

        # Query RAG for the top abnormalities
        context_text = "### 📚 Medical Reference Context:\n"
        
        # Limit to top 3 to avoid token overflow
        for issue in abnormal_params[:3]:
            # Query the vector database
            docs = self.vector_store.similarity_search(f"What causes {issue}?", k=1)
            if docs:
                context_text += f"**{issue}:** {docs[0].page_content}\n\n"

        return context_text
