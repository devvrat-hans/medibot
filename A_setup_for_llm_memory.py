# Step 01: Load Raw Pdf

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

DATA_PATH="data/"

def load_pdf(data):
    """Load a PDF file and return its content."""
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader,
    )
    documents = loader.load()
    return documents

documents = load_pdf(data=DATA_PATH)
print(f"Loaded {len(documents)} documents from {DATA_PATH}")

# Step 02: Create Chunks

from langchain_text_splitters import RecursiveCharacterTextSplitter

def create_chunks(documents):
    """Split documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.create_documents([doc.page_content for doc in documents])
    return texts

text_chunks = create_chunks(documents)
print(f"Created {len(text_chunks)} text chunks from the documents.")

# Step 03: Create Vector Embeddings

from langchain_huggingface import HuggingFaceEmbeddings

def get_embedding_model():
    """Generate embeddings for the text chunks."""
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                            model_kwargs = {'device': 'cpu'},
                                            encode_kwargs = {'normalize_embeddings': False})
    return embedding_model

embedding_model = get_embedding_model()

# Step 04: Store embeddings in FAISS

from langchain_community.vectorstores import FAISS

DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = get_embedding_model()

db = FAISS.from_documents(text_chunks,embedding_model)
db.save_local(DB_FAISS_PATH)

print(f"Stored embeddings in FAISS at {DB_FAISS_PATH}")