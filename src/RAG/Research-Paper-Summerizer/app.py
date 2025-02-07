import os
import tempfile
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.schema import Document
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Streamlit Page Config
st.set_page_config(
    page_title="Research Paper Summarizer",
    layout="centered"
)

st.title("📚 Research Paper Summarizer")

# File Uploader
uploaded_files = st.file_uploader(
    "Upload one or more research PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

# Initialize vector store in session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Process PDFs and create/update the vector store
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
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""]
        )

        for doc in pdf_docs:
            chunks = text_splitter.split_text(doc.page_content)
            for chunk in chunks:
                # Create Document object for each chunk
                all_documents.append(Document(page_content=chunk, metadata=doc.metadata))

    # Create vector store from documents
    embeddings = OpenAIEmbeddings()
    st.session_state.vector_store = FAISS.from_documents(
        documents=all_documents,
        embedding=embeddings
    )

    st.success("PDFs processed and vector store created! ✅")

# Query + Summarize
query = st.text_input("Enter your question or summary request:")

if st.button("Get Summary/Answer"):
    if st.session_state.vector_store is None:
        st.warning("Please upload and process PDFs first.")
    else:
        # Create retriever and chain
        retriever = st.session_state.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        llm = OpenAI(temperature=0.0)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )

        # Execute query
        result = qa_chain({"query": query})

        # Display the result
        st.markdown("### Answer:")
        st.write(result["result"])

        with st.expander("Show source documents"):
            source_docs = result["source_documents"]
            for i, doc in enumerate(source_docs):
                st.markdown(f"**Source Document {i+1}:**")
                st.write(doc.page_content)
                st.write("---")