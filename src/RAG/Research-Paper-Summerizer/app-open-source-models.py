import os
import tempfile
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from io import BytesIO
from langchain.document_loaders import PyPDFLoader
from transformers import pipeline
from langchain.schema import Document
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

# Load environment variables from Hugging Face Secrets
load_dotenv()

os.environ['HUGGINGFACE_API_KEY'] = os.getenv("HF_TOKEN")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"]="Research-Paper-Summarizer"

# Streamlit Page Config
st.set_page_config(
    page_title="Research Paper Summarizer",
    layout="centered"
)

st.title("📚 Research Paper Summarizer - Using Open Source Models")

# File Uploader
uploaded_files = st.file_uploader(
    "Upload one or more research PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

# A placeholder to store vector database (FAISS)
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Hugging Face LLM Model Pipeline
def get_huggingface_pipeline():
    
    model_name = "meta-llama/Llama-3.2-1B"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    st.info("Loading Hugging Face Model... Please wait.")
    
    return transformers.pipeline(
        "text-generation",
        model=model_name,
        tokenizer=tokenizer,
        max_new_tokens=256, 
        torch_dtype=torch.bfloat16
)


# Process the PDFs, Create/Update the Vector Store
if st.button("Process PDFs") and uploaded_files:
    all_documents = []

    for file in uploaded_files:
        # Save the file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file.getvalue())
            temp_file_path = temp_file.name

        # Load the PDF using PyPDFLoader
        loader = PyPDFLoader(temp_file_path)
        pdf_docs = loader.load()

        # Split text into manageable chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=300,
            separators=["\n\n", "\n", " ", ""]
        )

        for doc in pdf_docs:
            chunks = text_splitter.split_text(doc.page_content)
            for chunk in chunks:
                # Create Document object for each chunk
                all_documents.append(Document(page_content=chunk, metadata=doc.metadata))

    # Create embeddings with Hugging Face
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    st.session_state.vector_store = FAISS.from_documents(
        documents=all_documents,
        embedding=embeddings
    )

    st.success("PDFs processed and vector store created!")

# Query + Summarize
query = st.text_input("Enter your question or summary request:")

if st.button("Get Summary/Answer"):
    if st.session_state.vector_store is None:
        st.warning("Please upload and process PDFs first.")
    else:
        retriever = st.session_state.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )

        # Use Hugging Face LLM
        hf_pipeline = get_huggingface_pipeline()

        # Retrieve documents and generate response
        relevant_docs = retriever.get_relevant_documents(query)
        context_text = "\n".join([doc.page_content for doc in relevant_docs])

        # Generate answer using Hugging Face model
        response = hf_pipeline(f"Context: {context_text}\nQuestion: {query}", num_return_sequences=1)

        st.markdown("### Answer:")
        st.write(response[0]['generated_text'])

        with st.expander("Show source documents"):
            for i, doc in enumerate(relevant_docs):
                st.markdown(f"**Source Document {i + 1}:**")
                st.write(doc.page_content)
                st.write("---")