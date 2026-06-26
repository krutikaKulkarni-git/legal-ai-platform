import os
import shutil
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from pydantic import BaseModel
from pinecone import Pinecone
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import redis
import json

load_dotenv()

app = FastAPI(
    title="LegalAI Platform API",
    description="Enterprise-grade internal RAG API engine for multi-document contractual extraction.",
    version="1.0.0"
)

# Shared Component Configurations
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "legal-ai-index"

if not PINECONE_API_KEY:
    raise ValueError("❌ PINECONE_API_KEY missing from system configurations.")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# --- Pydantic Data Validation schemas ---
class QueryRequest(BaseModel):
    prompt: str

class QueryResponse(BaseModel):
    query: str
    response: str
    sources: list

# --- FIX 1: Use REDIS_HOST environment variable globally ---
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
try:
    # Changed 'localhost' here to the dynamic REDIS_HOST variable
    redis_client = redis.Redis(host=REDIS_HOST, port=6379, db=0, decode_responses=True)
    redis_client.ping()
    print("🔴 Connected to Redis Cache successfully.")
except Exception as e:
    print(f"⚠️ Redis connection failed. Proceeding without caching: {e}")
    redis_client = None

# --- FIX 2: Define global OLLAMA_URL configuration fallback ---
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


# --- Asynchronous Background Worker Logic ---
def async_process_and_upsert(file_path: str, filename: str):
    """Processes newly uploaded text document into Pinecone without breaking main UI main thread."""
    try:
        # FIX 3: Added base_url=OLLAMA_URL to the worker's embedding generator
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        text_splitter = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")
        
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            contract_text = f.read()

        raw_sections = [s.strip() for s in contract_text.split("\n\n") if s.strip()]
        chunks = []
        section_batch = ""
        
        for section in raw_sections:
            if len(section_batch) + len(section) < 10000:
                section_batch += section + "\n\n"
            else:
                if section_batch.strip():
                    chunks.extend(text_splitter.create_documents([section_batch]))
                section_batch = section + "\n\n"
        if section_batch.strip():
            chunks.extend(text_splitter.create_documents([section_batch]))

        vectors_to_upsert = []
        for i, chunk in enumerate(chunks):
            vector = embeddings.embed_query(chunk.page_content)
            vectors_to_upsert.append((
                f"{filename}_chunk_{i}",
                vector,
                {"filename": filename, "text": chunk.page_content, "chunk_index": i}
            ))

        batch_size = 50
        for b in range(0, len(vectors_to_upsert), batch_size):
            index.upsert(vectors=vectors_to_upsert[b:b+batch_size])
            
        print(f"⚡ [Worker] Successfully background-indexed {len(chunks)} vectors for {filename}")
        
    except Exception as e:
        print(f"❌ [Worker Error] Failed processing file {filename}: {str(e)}")


# --- API Routes ---

@app.get("/")
def read_root():
    return {"status": "online", "system": "LegalAI Engine v1"}

@app.post("/api/v1/upload", status_code=202)
async def upload_contract(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Accepts document uploads, saves to disk, and pushes execution stack out to background threads."""
    if not file.filename.endswith('.txt'):
        raise HTTPException(status_code=400, detail="Only plain text files (.txt) are supported currently.")
        
    upload_dir = "./data/raw_contracts"
    os.makedirs(upload_dir, exist_ok=True)
    
    saved_file_path = os.path.join(upload_dir, file.filename)
    
    with open(saved_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    background_tasks.add_task(async_process_and_upsert, saved_file_path, file.filename)
    
    return {
        "status": "accepted",
        "file_name": file.filename,
        "detail": "Document upload confirmed. Indexing process running asynchronously."
    }

@app.post("/api/v1/query", response_model=QueryResponse)
async def query_rag_engine(payload: QueryRequest):
    """Executes RAG query with an integrated Redis caching layer for performance optimization."""
    try:
        # 1. Check Redis Cache First
        cache_key = f"rag_cache:{payload.prompt.strip().lower()}"
        if redis_client:
            cached_data = redis_client.get(cache_key)
            if cached_data:
                print("⚡ [Cache Hit] Serving response directly from Redis!")
                return QueryResponse(**json.loads(cached_data))
        
        print("🧊 [Cache Miss] Running full vector extraction pipeline...")
        
        # 2. Run normal RAG pipeline if it's a miss
        # FIX 4: Added base_url=OLLAMA_URL to the query embeddings generator
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        query_vector = embeddings.embed_query(payload.prompt)
        
        search_results = index.query(vector=query_vector, top_k=4, include_metadata=True)
        
        context_chunks = []
        sources = []
        for match in search_results['matches']:
            if 'text' in match['metadata']:
                context_chunks.append(match['metadata']['text'])
                sources.append(f"{match['metadata']['filename']} (Chunk {int(match['metadata']['chunk_index'])})")
                
        if not context_chunks:
            return {"query": payload.prompt, "response": "No relevant context found.", "sources": []}
            
        full_context = "\n\n---\n\n".join(context_chunks)
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an expert legal AI assistant. Analyze the provided contract context "
                "and accurately answer the user's inquiry.\n\n"
                "Rely strictly on the text provided. Do not invent details.\n\n"
                "--- PROVIDED LEGAL CONTEXT ---\n{context}"
            )),
            ("human", "{question}")
        ])
        
        # FIX 5: Added base_url=OLLAMA_URL to the ChatOllama LLM initializer
        llm = ChatOllama(model="llama3.2", temperature=0.0)
        chain = prompt_template | llm | StrOutputParser()
        
        ai_response = await chain.ainvoke({"context": full_context, "question": payload.prompt})
        
        response_payload = {
            "query": payload.prompt,
            "response": ai_response,
            "sources": list(set(sources))
        }
        
        # 3. Store fresh result in Redis with an explicit Expiration (TTL) of 1 Hour (3600 seconds)
        if redis_client:
            redis_client.setex(
                cache_key,
                3600,
                json.dumps(response_payload)
            )
            print("💾 Saved fresh pipeline generation to Redis cache.")
        
        return response_payload
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference operational crash: {str(e)}")