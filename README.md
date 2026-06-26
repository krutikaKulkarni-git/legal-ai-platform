# Enterprise LegalAI Platform API ⚖️🤖

A production-ready internal RAG (Retrieval-Augmented Generation) API engine optimized for multi-document contractual extraction using FastAPI, LangChain, Pinecone, and an integrated Redis caching layer.

## Architecture Pipeline
1. **Asynchronous Ingestion**: Documents are streamed to disk, segmented using a `SemanticChunker`, embedded via `nomic-embed-text`, and bulk-upserted (batch size: 50) to Pinecone via FastAPI background worker threads.
2. **Optimized Query Loop**: Incoming user prompts are normalized and checked against a local **Redis Cache** (1-Hour TTL) for sub-millisecond responses on repeated queries.
3. **Inference**: Cache misses trigger a vector search across Pinecone (Top K: 4) and context is passed into a local **Llama 3.2** model via Ollama with strict system-prompt constraints.

---

## Local Setup Instructions

### 1. Clone the Repository
```bash
git clone <YOUR_REPOSITORY_URL>
cd legal-ai-platform
2. Install Python Dependencies
Bash
pip install -r requirements.txt
3. Spin Up Background Infrastructure (Ollama & Redis)
Ensure your local machine has Redis and Ollama installed and running:

Bash
# Start Redis Daemon
brew services start redis

# Ensure embedding and LLM models are pulled locally
ollama pull nomic-embed-text
ollama pull llama3.2
4. Environment Configuration
Create a .env file in the root directory:

Code snippet
PINECONE_API_KEY=your_pinecone_api_key_here
5. Ingest the Dataset (Optional)
To build your own vector index using the CUAD contract dataset, place your .txt files in ./data/raw_contracts/ and run:

Bash
python src/bulk_ingest.py
6. Boot the API Engine
Bash
uvicorn src.main:app --reload
Once started, navigate to http://127.0.0.1:8000/docs to explore the fully interactive Swagger API documentation panel!


---

### Step 3: Push the Blueprint Live!

Once you save those two files in your folder, run these three lines in your terminal to update your GitHub repo:

```bash
git add requirements.txt README.md
git commit -m "docs: add requirements and production installation guide to README"
git push