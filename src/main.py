import argparse
import glob
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock
from typing import List, Optional

from dotenv import load_dotenv

from azure.cosmos import CosmosClient, PartitionKey
from langchain_azure_ai.vectorstores.azure_cosmos_db_no_sql import (
    AzureCosmosDBNoSqlVectorSearch,
)
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ==============================
# CONFIGURACIÓN GENERAL
# ==============================

load_dotenv()

DOCS_DIR = os.getenv("DOCS_DIR", "./src/documentos_subidos")

AZURE_COSMOS_DB_ENDPOINT = os.getenv("AZURE_COSMOS_DB_ENDPOINT", "")
AZURE_COSMOS_DB_KEY = os.getenv("AZURE_COSMOS_DB_KEY", "")
COSMOS_DB_NAME = os.getenv("COSMOS_DB_NAME", "vector1")
COSMOS_CONTAINER_NAME = os.getenv("COSMOS_CONTAINER_NAME", "rag_local_container")

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))


# ==============================
# FUNCIONES DE INICIALIZACIÓN
# ==============================

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")


def get_llm() -> ChatGoogleGenerativeAI:
    if not GEMINI_API_KEY:
        raise ValueError("❌ Falta GEMINI_API_KEY en tu .env")
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GEMINI_API_KEY,
        temperature=0.1,
    )

def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_cosmos_client() -> CosmosClient:
    if not AZURE_COSMOS_DB_ENDPOINT or not AZURE_COSMOS_DB_KEY:
        print("[ERROR] Configure las credenciales de Azure en el archivo .env")
        sys.exit(1)
    return CosmosClient(AZURE_COSMOS_DB_ENDPOINT, AZURE_COSMOS_DB_KEY)


# ==============================
# CARGA Y CHUNKING
# ==============================

def load_pdfs_from_folder(folder_path: str) -> List[Document]:
    pdf_paths = glob.glob(os.path.join(folder_path, "*.pdf"))
    if not pdf_paths:
        return []
    docs = []
    for path in pdf_paths:
        print(f"[INFO] Cargando PDF: {path}")
        loader = PyPDFLoader(path)
        docs.extend(loader.load())
    return docs


def split_documents(docs: List[Document], chunk_size: int = 1000, chunk_overlap: int = 100) -> List[Document]:
    if not docs: return []
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)


# ==============================
# COSMOS VECTOR STORE
# ==============================

def get_policies() -> tuple:
    indexing_policy = {
        "indexingMode": "consistent",
        "includedPaths": [{"path": "/*"}],
        "excludedPaths": [{"path": '/"_etag"/?'}],
        "vectorIndexes": [{"path": "/vector1", "type": "diskANN"}],
        "fullTextIndexes": [{"path": "/text"}],
    }

    vector_embedding_policy = {
        "vectorEmbeddings": [
            {
                "path": "/vector1",
                "dataType": "float32",
                "distanceFunction": "cosine",
                "dimensions": EMBEDDING_DIM,
            }
        ]
    }

    # CORRECCIÓN: Se eliminó el ":" al final de "en-US"
    full_text_policy = {
        "defaultLanguage": "en-US",
        "fullTextPaths": [{"path": "/text", "language": "en-US"}],
    }

    return indexing_policy, vector_embedding_policy, full_text_policy


def build_vector_store_in_cosmos(docs: List[Document],
                                 embeddings: HuggingFaceEmbeddings) -> AzureCosmosDBNoSqlVectorSearch:
    cosmos_client = get_cosmos_client()
    indexing_policy, vector_embedding_policy, full_text_policy = get_policies()

    # CORRECCIÓN: Sincronización de nombres de campos con Azure
    vector_search_fields = {
        "text_field": "text",
        "embedding_field": "vector1",
        "fields": [
            {
                "path": "/vector1",
                "type": "vector",
                "similarity": "cosine",
                "dimensions": EMBEDDING_DIM,
                "dataType": "float32",
            }
        ]
    }

    print("[INFO] Creando vector store en Azure Cosmos DB...")
    return AzureCosmosDBNoSqlVectorSearch.from_documents(
        documents=docs,
        embedding=embeddings,
        cosmos_client=cosmos_client,
        database_name=COSMOS_DB_NAME,
        container_name=COSMOS_CONTAINER_NAME,
        vector_embedding_policy=vector_embedding_policy,
        full_text_policy=full_text_policy,
        indexing_policy=indexing_policy,
        cosmos_container_properties={"partition_key": PartitionKey(path="/id")},
        cosmos_database_properties={},
        vector_search_fields=vector_search_fields,
        full_text_search_enabled=True,
    )


def get_vector_store_for_existing_data(embeddings: HuggingFaceEmbeddings) -> AzureCosmosDBNoSqlVectorSearch:
    cosmos_client = get_cosmos_client()
    indexing_policy, vector_embedding_policy, full_text_policy = get_policies()

    vector_search_fields = {
        "text_field": "text",
        "embedding_field": "vector1",  # Cambiado de "embedding" a "vector1"
        "fields": [
            {
                "path": "/vector1",
                "type": "vector",
                "similarity": "cosine",
                "dimensions": EMBEDDING_DIM,
                "dataType": "float32",
            }
        ]
    }

    return AzureCosmosDBNoSqlVectorSearch(
        cosmos_client=cosmos_client,
        embedding=embeddings,
        vector_embedding_policy=vector_embedding_policy,
        indexing_policy=indexing_policy,
        cosmos_container_properties={"partition_key": PartitionKey(path="/id")},
        cosmos_database_properties={},
        full_text_policy=full_text_policy,
        database_name=COSMOS_DB_NAME,
        container_name=COSMOS_CONTAINER_NAME,
        create_container=False,
        full_text_search_enabled=True,
        vector_search_fields=vector_search_fields,
    )


# ==============================
# RAG CHAIN & CLI
# ==============================

def build_rag_chain(vector_store: AzureCosmosDBNoSqlVectorSearch, llm: ChatGoogleGenerativeAI):
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    system_prompt = (
        "You are an expert assistant that responds ONLY using the information "
        "contained in the provided documents.\n"
        "1. Use only the provided context.\n"
        "2. If unknown, say: \"I do not know with the information available.\"\n"
        "3. Always respond in English.\n"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "User Question: {input}\n\nContext:\n{context}\n\nAnswer:"),
    ])

    def format_docs(docs: List[Document]) -> str:
        return "\n\n".join(d.page_content for d in docs)

    answer_chain = (
            {"context": retriever | format_docs, "input": RunnablePassthrough()}
            | prompt | llm | StrOutputParser()
    )

    return RunnableParallel(answer=answer_chain, context_docs=retriever)


def print_answer_with_sources(result: dict):
    print("\nRESPONSE:\n" + "-" * 40 + f"\n{result['answer']}\n" + "-" * 40)
    for i, doc in enumerate(result.get("context_docs", []), 1):
        print(f"[{i}] {doc.metadata.get('source')} (Page {doc.metadata.get('page')})")


class RAGService:
    def __init__(self):
        self._llm: Optional[ChatGoogleGenerativeAI] = None
        self._embeddings: Optional[HuggingFaceEmbeddings] = None
        self._rag_chain = None
        self._lock = Lock()

    def _get_llm(self) -> ChatGoogleGenerativeAI:
        if self._llm is None:
            self._llm = get_llm()
        return self._llm

    def _get_embeddings(self) -> HuggingFaceEmbeddings:
        if self._embeddings is None:
            self._embeddings = get_embeddings()
        return self._embeddings

    def connect_existing_index(self):
        with self._lock:
            vector_store = get_vector_store_for_existing_data(self._get_embeddings())
            self._rag_chain = build_rag_chain(vector_store, self._get_llm())

    def reindex_documents(self):
        with self._lock:
            docs = load_pdfs_from_folder(DOCS_DIR)
            split_docs = split_documents(docs)
            if not split_docs:
                raise ValueError(f"No se encontraron documentos PDF en: {DOCS_DIR}")
            vector_store = build_vector_store_in_cosmos(split_docs, self._get_embeddings())
            self._rag_chain = build_rag_chain(vector_store, self._get_llm())

    def is_ready(self) -> bool:
        return self._rag_chain is not None

    def ask(self, message: str) -> dict:
        if not self._rag_chain:
            raise RuntimeError("El agente no está inicializado. Use índice existente o reindexe.")

        result = self._rag_chain.invoke(message)
        sources = [
            f"{doc.metadata.get('source')} (Page {doc.metadata.get('page')})"
            for doc in result.get("context_docs", [])
        ]
        return {"answer": result["answer"], "sources": sources}


rag_service = RAGService()


class AgentRequestHandler(BaseHTTPRequestHandler):
    cors_origins = os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    )

    def _set_headers(self, status_code: int = 200):
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _write_json(self, payload: dict, status_code: int = 200):
        self._set_headers(status_code)
        self.wfile.write(json.dumps(payload).encode("utf-8"))

    def do_OPTIONS(self):
        self._set_headers(200)

    def do_GET(self):
        if self.path == "/health":
            self._write_json({"status": "ok"})
            return
        self._write_json({"detail": "Not found"}, status_code=404)

    def do_POST(self):
        try:
            if self.path == "/connect":
                rag_service.connect_existing_index()
                self._write_json({"status": "connected"})
                return

            if self.path == "/reindex":
                rag_service.reindex_documents()
                self._write_json({"status": "reindexed", "docs_dir": DOCS_DIR})
                return

            if self.path == "/chat":
                content_length = int(self.headers.get("Content-Length", "0"))
                body_raw = self.rfile.read(content_length) if content_length > 0 else b"{}"
                body = json.loads(body_raw.decode("utf-8"))
                message = str(body.get("message", "")).strip()
                if not message:
                    self._write_json({"detail": "message is required"}, status_code=400)
                    return

                if not rag_service.is_ready():
                    rag_service.connect_existing_index()

                answer = rag_service.ask(message)
                self._write_json(answer)
                return

            self._write_json({"detail": "Not found"}, status_code=404)
        except Exception as exc:
            self._write_json({"detail": str(exc)}, status_code=500)


def run_api_server(host: str, port: int):
    print(f"[INFO] API server running on http://{host}:{port}")
    server = ThreadingHTTPServer((host, port), AgentRequestHandler)
    server.serve_forever()


def run_cli():
    print("[INFO] Modo CLI iniciado.")
    while True:
        user_input = input("\nPregunta (o :reindex / :use / :salir): ").strip()
        if user_input.lower() in (":salir", ":q"):
            break

        try:
            if user_input.lower() == ":reindex":
                rag_service.reindex_documents()
                print("[INFO] Indexación lista.")
            elif user_input.lower() == ":use":
                rag_service.connect_existing_index()
                print("[INFO] Usando índice existente.")
            else:
                result = rag_service.ask(user_input)
                print("\nRESPONSE:\n" + "-" * 40 + f"\n{result['answer']}\n" + "-" * 40)
                for i, source in enumerate(result["sources"], 1):
                    print(f"[{i}] {source}")
        except Exception as exc:
            print(f"[ERROR] {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG agent in CLI or API mode.")
    parser.add_argument("--mode", choices=["cli", "api"], default="cli")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.mode == "api":
        run_api_server(args.host, args.port)
    else:
        run_cli()
