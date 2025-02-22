import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from uuid import uuid4
from models import Models


load_dotenv()

models = Models()
embeddings = models.embeddings_ollama
llm = models.model_ollama

data_folder = "./data"
chunk_size = 1000
chunk_overlap = 50
check_interval = 10

vector_store = Chroma(
    collection_name="documents",
    embedding_function=embeddings,
    persist_directory=".db/chroma_langchain_db"
)

def ingest_file(file_path):
    if not file_path.lower().endswith(".pdf"):
        return

    print(f"Ingesting file: {file_path}")
    loader = PyPDFLoader(file_path)
    loader_document = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n", " ", ""]
    )

    documents = text_splitter.split_documents(loader_document)
    uuids = [str(uuid4()) for _ in range(len(documents))]
    print(f"Adding {len(documents)} documents to the vector store")
    vector_store.add_documents(documents=documents, ids=uuids)
    print(f"Finished ingesting file: {file_path}")

if __name__ == "__main__":
    while True:
        for filename in os.listdir(data_folder):
            if not filename.startswith("_"):
                file_path = os.path.join(data_folder, filename)
                ingest_file(file_path)
                new_filename = "_" + filename
                new_file_path = os.path.join(data_folder, new_filename)
                os.rename(file_path, new_file_path)
            time.sleep(check_interval)