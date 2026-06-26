import os
import glob
from langchain_experimental.text_splitter import SemanticChunker
# 1. CHANGE THIS IMPORT: Swap OpenAI for Ollama
from langchain_ollama import OllamaEmbeddings 
from dotenv import load_dotenv

# Load environment variables (You don't strictly need this for Ollama anymore, 
# but it's fine to keep if you use other API keys later)
load_dotenv()

def get_all_txt_files(base_dir):
    """Finds all contract text files in Part_I and Part_II folders."""
    return glob.glob(os.path.join(base_dir, "**/*.txt"), recursive=True)

def run_semantic_chunking():
    input_dir = "./data/raw_contracts/CUAD_v1/full_contract_txt"
    output_dir = "./data/processed_chunks"
    
    # 2. CHANGE THIS LINE: Initialize Ollama with nomic-embed-text
    print("Initializing Semantic Chunker with Local Ollama Embeddings...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    # The rest of your logic stays exactly the same!
    text_splitter = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")

    # 3. Collect all text files
    files = get_all_txt_files(input_dir)
    print(f"Found {len(files)} contracts to process.")

    # Process just the first 5 contracts to test the pipeline locally
    test_limit = 5
    for idx, file_path in enumerate(files[:test_limit]):
        filename = os.path.basename(file_path)
        print(f"Processing [{idx+1}/{test_limit}]: {filename}...")
        
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            contract_text = f.read()

        # Create Chunks split by topic drift
        chunks = text_splitter.create_documents([contract_text])
        
        # Save chunks locally
        output_file_path = os.path.join(output_dir, f"chunks_{filename}")
        
        # Ensure output directory exists so it doesn't crash
        os.makedirs(output_dir, exist_ok=True)
        
        with open(output_file_path, "w", encoding="utf-8") as out_f:
            for i, chunk in enumerate(chunks):
                out_f.write(f"--- CHUNK {i+1} ---\n")
                out_f.write(chunk.page_content + "\n\n")
                
        print(f" Saved {len(chunks)} semantic chunks to {output_file_path}")

if __name__ == "__main__":
    run_semantic_chunking()