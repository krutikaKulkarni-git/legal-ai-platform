import os
import glob
import time
from pinecone import Pinecone
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "legal-ai-index"

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

def get_all_txt_files(base_dir):
    return glob.glob(os.path.join(base_dir, "**/*.txt"), recursive=True)

def bulk_upload():
    input_dir = "./data/raw_contracts/CUAD_v1/full_contract_txt"
    
    print("🧠 Initializing local Ollama Embeddings...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    text_splitter = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")

    files = get_all_txt_files(input_dir)
    total_files = len(files)
    print(f"📂 Found {total_files} total contracts to index.")

    for idx, file_path in enumerate(files):
        filename = os.path.basename(file_path)
        print(f"\n🔄 Progress: [{idx+1}/{total_files}] | Processing: {filename}...")
        
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                contract_text = f.read()

            if not contract_text.strip():
                continue

            # Prevent memory overflow: break down into ~10k character segments
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

            # Generate and stage vectors
            vectors_to_upsert = []
            for i, chunk in enumerate(chunks):
                vector = embeddings.embed_query(chunk.page_content)
                chunk_id = f"{filename}_chunk_{i}"
                metadata = {
                    "filename": filename,
                    "text": chunk.page_content,
                    "chunk_index": i
                }
                vectors_to_upsert.append((chunk_id, vector, metadata))

            # Batch upload to Pinecone (50 at a time)
            batch_size = 50
            for b in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[b:b+batch_size]
                index.upsert(vectors=batch)
                
            print(f"✅ Indexed {len(chunks)} chunks for {filename}")
            
            # Rate limiting buffer to let your local machine cool down slightly between files
            time.sleep(0.5)
            
        except Exception as e:
            print(f"❌ Failed to process {filename}: {str(e)}")
            continue

if __name__ == "__main__":
    bulk_upload()