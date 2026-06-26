import os
from pinecone import Pinecone
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# 1. Initialize Pinecone connection
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("❌ PINECONE_API_KEY not found in .env file.")

pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "legal-ai-index"
index = pc.Index(INDEX_NAME)

def retrieve_and_generate(query: str):
    print(f"\n🔍 User Query: '{query}'")
    
    # 2. Convert query to vector embedding using our local nomic model
    print("🧠 Embedding query with nomic-embed-text...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    query_vector = embeddings.embed_query(query)
    
    # 3. Search Pinecone for top 4 matches
    print("📡 Querying Pinecone Vector DB...")
    search_results = index.query(
        vector=query_vector,
        top_k=4,
        include_metadata=True
    )
    
    # Extract text from the retrieved metadata matches
    context_chunks = []
    retrieved_docs_info = []
    
    for match in search_results['matches']:
        if 'text' in match['metadata']:
            context_chunks.append(match['metadata']['text'])
            retrieved_docs_info.append(f"- {match['metadata']['filename']} (Chunk {int(match['metadata']['chunk_index'])}) [Score: {round(match['score'], 3)}]")
            
    if not context_chunks:
        print("⚠️ No relevant legal context found in database.")
        return
        
    print("\n📄 Retrieved Context Sources:")
    print("\n".join(retrieved_docs_info))
    
    # Join the retrieved chunks into a single string block
    full_context = "\n\n---\n\n".join(context_chunks)
    
    # 4. Construct System Prompt Template
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert legal AI assistant. Your job is to strictly analyze the provided contract context "
            "and accurately answer the user's inquiry.\n\n"
            "CRITICAL RULES:\n"
            "1. ONLY base your response on the provided context below. Do not assume or make things up.\n"
            "2. Cite the filename of the source document you derived the answer from.\n"
            "3. If the context does not contain the answer, explicitly state: 'Based on the loaded documentation, "
            "I could not locate this information.'\n\n"
            "--- PROVIDED LEGAL CONTEXT ---\n"
            "{context}"
        )),
        ("human", "{question}")
    ])
    
    # 5. Initialize the Generator LLM (Local Llama 3.2 via Ollama)
    print("\n🤖 Invoking local Llama 3.2 for analysis...")
    llm = ChatOllama(model="llama3.2", temperature=0.0) # 0.0 temperature enforces precise extraction over creativity
    
    # Chain components together using LCEL (LangChain Expression Language)
    chain = prompt_template | llm | StrOutputParser()
    
    # Stream the output token-by-token directly to the terminal
    print("\n⚖️ LegalAI Response:")
    for chunk in chain.stream({"context": full_context, "question": query}):
        print(chunk, end="", flush=True)
    print("\n")

if __name__ == "__main__":
    # Test Question targeting common contract details
    sample_query = "What are the rules regarding confidentiality, non-disclosure, or governing law mentioned in these agreements?"
    retrieve_and_generate(sample_query)