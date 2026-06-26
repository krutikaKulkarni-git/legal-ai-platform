Markdown
# Enterprise LegalAI Platform API ⚖️🤖

A production-ready internal RAG (Retrieval-Augmented Generation) API engine optimized for multi-document contractual extraction using FastAPI, LangChain, Pinecone, and an integrated Redis caching layer—fully containerized with Docker.

## System Architecture & Pipeline
1. **Asynchronous Ingestion**: Documents are streamed to disk, segmented using a `SemanticChunker`, embedded via `nomic-embed-text`, and bulk-upserted (batch size: 50) to a cloud Pinecone index via background worker threads.
2. **Optimized Query Loop**: Incoming user prompts are normalized and checked against an isolated **Redis Cache container** (1-Hour TTL) for sub-millisecond responses on repeated queries.
3. **Hybrid Inference Engine**: Cache misses trigger a vector search across Pinecone (Top K: 4). Context is passed out of the isolated Docker environment via a network bridge (`host.docker.internal`) to a local **Llama 3.2** model running natively via Ollama to maximize hardware acceleration on the host GPU.

---

## Technical Prerequisites

Before spinning up the container stack, ensure you have the following installed on your host machine:
* [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Ensure the daemon is running)
* [Ollama](https://ollama.com/)

### 1. Configure the Host LLM Engine
To leverage local hardware acceleration, pull the required embedding and LLM weights natively:
```bash
ollama pull nomic-embed-text
ollama pull llama3.2
Because the application runs inside an isolated container, configure your native Ollama daemon to listen to incoming container traffic by running this in a separate terminal window:

Bash
OLLAMA_HOST=0.0.0.0:11434 OLLAMA_ORIGINS="*" ollama serve
Local Deployment Instructions
1. Clone & Configure Environment
Clone the repository and create a .env file in the root directory:

Bash
git clone <YOUR_REPOSITORY_URL>
cd legal-ai-platform
Add your cloud vector database credentials to the .env file:

Code snippet
PINECONE_API_KEY=your_actual_pinecone_api_key_here
2. Launch the Containerized Stack
Spin up the FastAPI web backend and the Redis caching layer simultaneously using Docker Compose:

Bash
docker compose up --build
Docker will automatically resolve package dependencies, wire up the internal virtual network, and link the application to your Redis cache instance.

3. Verify and Explore
Once the terminal outputs successful startup logs, navigate to the fully interactive Swagger API documentation panel to run queries and test endpoints:

Interactive API Dashboard: http://localhost:8000/docs

Production Ingestion Workflow (Optional)
To ingest your own raw text files into the cloud vector index, place your raw text agreements inside the ./data/raw_contracts/ folder and trigger the asynchronous ingestion pipeline by uploading via the /api/v1/upload endpoint in the Swagger UI.


---

### Sync the Blueprint to GitHub

Once you save the file, run this final sequence in your terminal to push your pristine documentation live:

```bash
git add README.md
git commit -m "docs: update README to reflect hybrid docker-compose orchestration architecture"
git push