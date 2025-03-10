from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.vectordb.lancedb import LanceDb, SearchType
from agno.embedder.openai import OpenAIEmbedder
import os
from dotenv import load_dotenv
from agno.knowledge.pdf import PDFKnowledgeBase, PDFReader

load_dotenv(override=True)

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set. Please add it to your .env file.")

# Paths
# Absolute paths for safety
DB_PATH = os.path.abspath("../knowledge/processed/knowledge_base")
PDF_DIR = os.path.abspath("../knowledge/raw/")

def build_knowledge_base():
    os.makedirs(DB_PATH, exist_ok=True)

    knowledge_base = PDFKnowledgeBase(
        path=PDF_DIR,
        vector_db=LanceDb(
            uri=DB_PATH,
            table_name="documents",
            embedder=OpenAIEmbedder(id="text-embedding-ada-002", api_key=api_key),
        ),
    )
    

    # Force rebuild and embedding
    knowledge_base.load(recreate=True)

    print(f"Knowledge base built and stored at {DB_PATH}")
    print(f"Knowledge base built and stored at {DB_PATH}")

if __name__ == "__main__":
    build_knowledge_base()