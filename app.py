import os
import tempfile
import hashlib
import uuid
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from langchain_classic.chains import (
    create_retrieval_chain,
    create_history_aware_retriever
)
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory


# Setup
load_dotenv()
st.set_page_config(page_title="Conversational PDF RAG")
st.title("Conversational PDF Chatbot")


# API Key
try:
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
except Exception:
    GROQ_API_KEY = None

GROQ_API_KEY = GROQ_API_KEY or os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("Please provide your GROQ API key.")
    st.stop()


# Models
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=GROQ_API_KEY,
    temperature=0
)


@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


embeddings = load_embeddings()


# Session State
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
    st.session_state.file_hash = None
    st.session_state.store = {}
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.conversational_chain = None


# Conversation History
def get_session_history(session: str) -> BaseChatMessageHistory:
    if session not in st.session_state.store:
        st.session_state.store[session] = ChatMessageHistory()

    return st.session_state.store[session]


# Upload PDF
uploaded_file = st.file_uploader("Upload PDF", type="pdf")


if uploaded_file:

    pdf_bytes = uploaded_file.getvalue()
    file_hash = hashlib.sha256(pdf_bytes).hexdigest()

    if st.button("Process PDF"):

        with st.spinner("Processing PDF..."):

            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    pdf_path = os.path.join(temp_dir, "document.pdf")

                    with open(pdf_path, "wb") as f:
                        f.write(pdf_bytes)

                    docs = PyPDFLoader(pdf_path).load()

                docs = [
                    doc for doc in docs
                    if doc.page_content.strip()
                ]

                if not docs:
                    st.error("No readable text found in PDF.")
                    st.stop()

                # Split Documents
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )

                chunks = splitter.split_documents(docs)

                # Create Vector Database
                st.session_state.vectorstore = FAISS.from_documents(
                    chunks, embeddings
                )

                st.session_state.file_hash = file_hash

                # Reset Conversation for New PDF
                st.session_state.store = {}
                st.session_state.session_id = str(uuid.uuid4())

                # Create Retriever
                retriever = st.session_state.vectorstore.as_retriever(
                    search_kwargs={"k": 4}
                )

                # History-Aware Retriever
                contextualize_prompt = ChatPromptTemplate.from_messages([
                    (
                        "system",
                        "Given the conversation history and latest question, "
                        "rewrite the question as a standalone question if needed. "
                        "Do not answer the question."
                    ),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}")
                ])

                history_aware_retriever = create_history_aware_retriever(
                    llm,
                    retriever,
                    contextualize_prompt
                )

                # Answer Prompt
                qa_prompt = ChatPromptTemplate.from_messages([
                    (
                        "system",
                        "Answer using ONLY the following retrieved context. "
                        "If the answer is unavailable, say: "
                        "\"I couldn't find this information in the uploaded PDF.\"\n\n"
                        "Context:\n{context}"
                    ),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}")
                ])

                document_chain = create_stuff_documents_chain(
                    llm, qa_prompt
                )

                # RAG Chain
                rag_chain = create_retrieval_chain(
                    history_aware_retriever,
                    document_chain
                )

                # Conversational RAG Chain
                st.session_state.conversational_chain = RunnableWithMessageHistory(
                    rag_chain,
                    get_session_history,
                    input_messages_key="input",
                    history_messages_key="chat_history",
                    output_messages_key="answer"
                )

                st.success(f"PDF processed! {len(chunks)} chunks created.")

            except Exception as e:
                st.error(f"Processing error: {e}")


# Question Answering
if (
    st.session_state.vectorstore is not None
    and st.session_state.file_hash == (
        hashlib.sha256(uploaded_file.getvalue()).hexdigest()
        if uploaded_file else None
    )
    and st.session_state.conversational_chain is not None
):

    st.divider()

    history = get_session_history(st.session_state.session_id)

    # Display Chat History
    for message in history.messages:

        role = "user" if message.type == "human" else "assistant"

        with st.chat_message(role):
            st.write(message.content)

    # User Input
    question = st.chat_input("Ask a question about your PDF...")

    if question:

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):

            try:
                with st.spinner("Generating answer..."):

                    response = st.session_state.conversational_chain.invoke(
                        {"input": question},
                        config={
                            "configurable": {
                                "session_id": st.session_state.session_id
                            }
                        }
                    )

                st.write(response["answer"])

                # Display Sources
                with st.expander("Source Passages"):

                    for i, doc in enumerate(response["context"], 1):

                        page = doc.metadata.get("page")

                        st.markdown(
                            f"**Source {i} | Page "
                            f"{page + 1 if isinstance(page, int) else 'N/A'}**"
                        )

                        st.write(doc.page_content)

            except Exception as e:
                st.error(f"Error generating answer: {e}")
