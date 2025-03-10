from agno.agent import Agent
from agno.knowledge.pdf import PDFKnowledgeBase
from agno.vectordb.lancedb import LanceDb
from agno.embedder.openai import OpenAIEmbedder
import os

# Paths
DB_PATH = "../knowledge/processed/knowledge_base"
PDF_DIR = "../knowledge/raw/"

def create_agent():
    # Initialize the knowledge base with the local PDF directory
    print("Initializing knowledge base...")
    knowledge_base = PDFKnowledgeBase(
        path=PDF_DIR,
        vector_db=LanceDb(
            uri=DB_PATH,
            table_name="documents",
            embedder=OpenAIEmbedder(id="text-embedding-ada-002"),
        ),
    )

    # Load the knowledge base
    print(f"Loading knowledge base from {PDF_DIR}...")
    knowledge_base.load(recreate=False)

    # Debug: Check how many documents are loaded
    print(f"Documents loaded: {(knowledge_base.num_documents)}")

    # Initialize the agent with the knowledge base
    agent = Agent(knowledge=knowledge_base, search_knowledge=True)
    return agent

def ask_question(question: str) -> str:
    # Create agent
    agent = create_agent()
    
    # Debug: Check if agent is initialized
    print("Agent initialized. Asking question...")

    # # Ask the question and get the response
    run_response = agent.run(question)
    answer = run_response.content
    agent.print_response(question)
    print(f"Answer: {answer}")
    return answer