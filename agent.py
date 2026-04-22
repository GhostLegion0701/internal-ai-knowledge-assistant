import math
import wikipedia
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.tools import Tool
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.agents import initialize_agent, AgentType
from langchain_chroma import Chroma


# ----------------------------
# Tool 1: Wikipedia Search
# ----------------------------
def wikipedia_search(query: str) -> str:
    """Search Wikipedia and return a short summary."""
    try:
        return wikipedia.summary(query, sentences=3)
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Your query is ambiguous. Try one of these: {e.options[:5]}"
    except wikipedia.exceptions.PageError:
        return "No Wikipedia page found for that query."
    except Exception as e:
        return f"Wikipedia tool error: {str(e)}"


# ----------------------------
# Tool 2: Calculator
# ----------------------------
def calculator_tool(expression: str) -> str:
    """Basic calculator for arithmetic expressions."""
    try:
        allowed_names = {
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "pow": pow,
            "sqrt": math.sqrt,
        }

        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"Calculator error: {str(e)}"


# ----------------------------
# Tool 3: Company Knowledge Base (RAG)
# ----------------------------
def company_knowledge_base(query: str) -> str:
    """Retrieve relevant company policy information from the vector database."""
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        vectorstore = Chroma(
            persist_directory="vectorstore",
            embedding_function=embeddings
        )

        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        retrieved_docs = retriever.invoke(query)

        if not retrieved_docs:
            return "I could not find that in the company documents."

        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0
        )

        prompt = f"""
You are an internal company knowledge assistant.
Answer the user's question only from the provided context.
If the answer is not available in the context, say: "I could not find that in the company documents."

Context:
{context}

Question:
{query}

Answer:
"""

        response = llm.invoke(prompt)
        return str(response.content)

    except Exception as e:
        return f"Knowledge base tool error: {str(e)}"


# ----------------------------
# Main LLM for Agent
# ----------------------------
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0
)


# ----------------------------
# Tools Registration
# ----------------------------
tools = [
    Tool(
        name="CompanyKnowledgeBase",
        func=company_knowledge_base,
        description=(
            "Use this for internal company policy, HR rules, leave policy, IT support, "
            "reimbursements, travel approvals, employee procedures, and company handbook questions."
        )
    ),
    Tool(
        name="Wikipedia",
        func=wikipedia_search,
        description=(
            "Use this for general world knowledge questions about people, places, companies, "
            "history, science, and concepts not related to internal company documents."
        )
    ),
    Tool(
        name="Calculator",
        func=calculator_tool,
        description=(
            "Use this for mathematical calculations. Input should be a valid expression like "
            "25 * 8 or sqrt(144)."
        )
    )
]


# ----------------------------
# Memory
# ----------------------------
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)


# ----------------------------
# Agent Setup
# ----------------------------
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
    agent_kwargs={
        "system_message": """You are an internal AI knowledge assistant.

Use conversation history first for personal follow-up questions such as the user's name, preferences, or anything the user already stated in the same session.

Do NOT use any tool for questions that can be answered directly from conversation memory.

Use the CompanyKnowledgeBase tool only for internal company policy, HR, leave, IT support, reimbursement, travel, and employee procedure questions.

Use Wikipedia only for general world knowledge unrelated to internal company documents.

Use Calculator only for math expressions.

If the user already told you their name earlier in the session, answer from memory directly."""
    }
)


# ----------------------------
# Reusable Assistant Function
# ----------------------------
def ask_assistant(question: str) -> str:
    """Send a question to the agent and return the final answer."""
    response = agent.run(question)
    return str(response)


# ----------------------------
# Terminal App
# ----------------------------
def main():
    print("\nInternal AI Knowledge Assistant")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ["exit", "quit"]:
            print("Exiting assistant.")
            break

        try:
            response = ask_assistant(user_input)
            print(f"Assistant: {response}\n")
        except Exception as e:
            print(f"Agent error: {str(e)}\n")


if __name__ == "__main__":
    main()