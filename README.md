# Conversational PDF RAG

A conversational PDF question-answering application built using Retrieval-Augmented Generation (RAG). Upload a PDF, ask questions based on its contents, and continue the conversation with context-aware follow-up questions.

### Live Demo

[**Try the Conversational PDF Chatbot**](https://conversational-pdf-chatbot-rag-mazjru3nvj4fz3as6pu2wj.streamlit.app/)

### Features

- Upload and process PDF documents.
- Ask questions based on document contents.
- Retrieve relevant passages using semantic search.
- Understand follow-up questions using conversational history.
- Maintain chat history during the active session.
- View source passages and page numbers.
- Generate answers using a Groq-hosted LLM.

### Tech Stack

Python · Streamlit · LangChain · FAISS · Hugging Face · Groq

### Run Locally

**1. Clone the repository**

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Conversational-PDF-RAG
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Configure your API key**

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
```

**4. Run the application**

```bash
streamlit run app.py
```

### Deployment

Deploy the repository using [Streamlit Community Cloud](https://share.streamlit.io/) and add `GROQ_API_KEY` to your application's Secrets settings.

### Note

The chatbot uses retrieved passages from the uploaded PDF to generate answers. Conversational history helps understand follow-up questions but is maintained only during the active session. Responses may not always be accurate, so verify important information against the original document. Scanned PDFs may require OCR.

---

Developed using Python, LangChain, and Streamlit.