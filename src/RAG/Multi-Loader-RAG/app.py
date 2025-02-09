import streamlit as st
from langchain_community.document_loaders import TextLoader, PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
import os
import tempfile
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"]="Multi Loader RAG"


# Streamlit app title
st.title("Multi Loader RAG")

# File upload and web link input
st.header("Upload Documents")
text_file = st.file_uploader("Upload a Text File", type=["txt"])
pdf_file = st.file_uploader("Upload a PDF File", type=["pdf"])
web_link = st.text_input("Enter a Web URL")

# Load documents function
def load_documents(text_file, pdf_file, web_link):
    docs = []
    
    # Load text file
    if text_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
            tmp_file.write(text_file.getvalue())
            tmp_file_path = tmp_file.name
        text_loader = TextLoader(tmp_file_path)
        docs.extend(text_loader.load())
        os.remove(tmp_file_path)
    
    # Load PDF file
    if pdf_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            tmp_file_path = tmp_file.name
        pdf_loader = PyPDFLoader(tmp_file_path)
        docs.extend(pdf_loader.load())
        os.remove(tmp_file_path)
    
    # Load web content
    if web_link:
        web_loader = WebBaseLoader([web_link])
        docs.extend(web_loader.load())
    
    return docs

# Split documents function
def split_documents(docs, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(docs)

# Create FAISS vector store function
def create_vector_store(splits):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(splits, embeddings)
    vectorstore.save_local("faiss_index")
    return vectorstore

# Main app logic
if st.button("Process Documents"):
    if not (text_file or pdf_file or web_link):
        st.error("Please upload at least one document or provide a web link.")
    else:
        with st.spinner("Processing documents..."):
            # Load documents
            documents = load_documents(text_file, pdf_file, web_link)
            
            # Split documents
            splits = split_documents(documents, 1000, 300)
            
            # Create FAISS vector store and save locally
            create_vector_store(splits)
            
            st.success("Documents processed and FAISS vector store created!")

# Query the vector store
st.header("Query the Vector Store")
query = st.text_input("Enter your query")

if st.button("Search"):
    if not os.path.exists("faiss_index"):
        st.error("Please process documents first.")
    elif not query:
        st.error("Please enter a query.")
    else:
        with st.spinner("Searching..."):
            # Load the FAISS vector store from local
            embeddings = OpenAIEmbeddings()
            vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
             
            # Create retriever and chain
            retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
            llm = OpenAI(temperature=0.6)
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