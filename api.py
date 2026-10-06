import os
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from openai import OpenAI

# Load API key from .env file
load_dotenv()

UPLOAD_FOLDER = "pdfs"
CHROMA_PATH = "chroma_db"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Fetch secure Grok API Key
GROK_API_KEY = os.getenv("GROK_API_KEY")

# Local Fast Embeddings (No Hugging Face overhead)
embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

# Vector Database (ChromaDB)
vector_store = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

# Grok Client Setup
grok_client = OpenAI(
    api_key=GROK_API_KEY,
    base_url="https://api.x.ai/v1",
)

def process_and_store_pdfs(files):
    all_text = ""
    for file in files:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(file_path, "wb") as f:
            f.write(file.file.read())

        reader = PdfReader(file_path)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                all_text += text + "\n"

    # Text Splitting
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(all_text)

    # Save to ChromaDB
    vector_store.add_texts(chunks)
    return len(files), len(chunks)

def query_rag_and_grok(question: str):
    # Retrieve top relevant context chunks
    docs = vector_store.similarity_search(question, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are an AI Study Assistant for Python programming.
Answer the question based ONLY on the following context retrieved from the PDF chapters:

Context:
{context}

Question: {question}

Answer clearly and concisely:
"""

    # Call Grok LLM
    response = grok_client.chat.completions.create(
        model="grok-beta",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content, context