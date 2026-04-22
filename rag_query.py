from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from dotenv import load_dotenv
load_dotenv()


def answer_question(question: str):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectorstore = Chroma(
        persist_directory="vectorstore",
        embedding_function=embeddings
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    retrieved_docs = retriever.invoke(question)

    print("\nTop retrieved chunks:\n")
    for i, doc in enumerate(retrieved_docs, start=1):
        print(f"Chunk {i}:")
        print(doc.page_content)
        print("-" * 50)

    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    prompt = f"""
You are an internal company knowledge assistant.
Answer the user's question only from the provided context.
If the answer is not available in the context, say: "I could not find that in the company documents."

Context:
{context}

Question:
{question}

Answer:
"""

    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0
    )

    response = llm.invoke(prompt)

    print("\nFinal Answer:\n")
    print(response.content)


def main():
    print("RAG Query Demo")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()

        if question.lower() in ["exit", "quit"]:
            print("Exiting RAG demo.")
            break

        answer_question(question)


if __name__ == "__main__":
    main()
