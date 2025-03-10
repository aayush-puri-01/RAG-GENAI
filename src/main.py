from agno.agent import Agent
from dotenv import load_dotenv

load_dotenv(override=True)
from query_system import ask_question

def main():
    print("Welcome to the Knowledge Base Query System!")
    while True:
        question = input("\nEnter your question (or type 'exit' to quit): ")
        if question.lower() == "exit":
            break
        answer = ask_question(question)
        ask_question(question)
        print(f"\nAnswer: {answer}")

if __name__ == "__main__":
    main()