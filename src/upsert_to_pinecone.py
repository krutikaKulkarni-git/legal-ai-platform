import os
import glob
from pinecone import Pinecone
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Initialize Pinecone and check API Key
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("❌ PINECONE_API_KEY not found in .env file.")

pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "legal-ai-index"

# Connect to our existing index
index = pc.Index(INDEX_NAME)

def get_all_txt_files(base_dir):
    return glob.glob(os.path.join(base_dir, "**/*.txt"), recursive=True)

def upload_to_vector_db():
    input_dir = "./data/raw_contracts/CUAD_v1/full_contract_txt"
    
    print("🧠 Initializing local Ollama Embeddings (nomic-embed-text)...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    text_splitter = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")

    files = get_all_txt_files(input_dir)
    print(f"📂 Found {len(files)} contracts to index.")

    # Let's index the first 3 contracts to test our Pinecone connection safely
    test_limit = 3
    
    for idx, file_path in enumerate(files[:test_limit]):
        filename = os.path.basename(file_path)
        print(f"\n🔄 Processing & Embedding [{idx+1}/{test_limit}]: {filename}...")
        
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            contract_text = f.read()

        # Break the massive contract into smaller sections (e.g., by double newlines)
        # to prevent overwhelming local Ollama's tokenizer batch size
        raw_sections = [s.strip() for s in contract_text.split("\n\n") if s.strip()]
        
        chunks = []
        print(f"📄 Processing contract in smaller text sections...")
        
        # Process sections in smaller batches to keep Ollama happy
        section_batch = ""
        for section in raw_sections:
            # Group sections into roughly 10,000 character pieces
            if len(section_batch) + len(section) < 10000:
                section_batch += section + "\n\n"
            else:
                if section_batch.strip():
                    # Safely run semantic chunking on this smaller piece
                    chunks.extend(text_splitter.create_documents([section_batch]))
                section_batch = section + "\n\n"
        
        # Don't forget the last batch!
        if section_batch.strip():
            chunks.extend(text_splitter.create_documents([section_batch]))

        print(f"🧩 Split into {len(chunks)} total semantic chunks. Generating vectors...")

        # Prepare payload for Pinecone batch upsert
        vectors_to_upsert = []
        
        for i, chunk in enumerate(chunks):
            # Generate the vector embedding for this specific chunk text using Ollama
            vector = embeddings.embed_query(chunk.page_content)
            
            # Create a unique ID for this vector chunk
            chunk_id = f"{filename}_chunk_{i}"
            
            # Store the text inside metadata so the LLM can read it later
            metadata = {
                "filename": filename,
                "text": chunk.page_content,
                "chunk_index": i
            }
            
            vectors_to_upsert.append((chunk_id, vector, metadata))

        # Pinecone upserts work best in smaller batches (e.g., 50 vectors at a time)
        batch_size = 50
        for b in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[b:b+batch_size]
            index.upsert(vectors=batch)
            
        print(f"✅ Successfully upserted {len(chunks)} vectors to Pinecone for {filename}!")
if __name__ == "__main__":
    upload_to_vector_db()