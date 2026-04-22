# Internal AI Knowledge Assistant

A Python-based enterprise-style AI assistant that combines **Retrieval-Augmented Generation (RAG)**, **tool-based reasoning**, **short-term conversational memory**, and a **FastAPI backend** to answer employee-style questions from internal company documents and external knowledge sources.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Solution Overview](#solution-overview)
- [Architecture](#architecture)
- [Layer-wise Explanation](#layer-wise-explanation)
  - [Layer 1: Data Layer](#layer-1-data-layer)
  - [Layer 2: Ingestion Layer](#layer-2-ingestion-layer)
  - [Layer 3: Retrieval and Generation Layer](#layer-3-retrieval-and-generation-layer)
  - [Layer 4: Agent and Tool Routing Layer](#layer-4-agent-and-tool-routing-layer)
  - [Layer 5: Memory Layer](#layer-5-memory-layer)
  - [Layer 6: API Layer](#layer-6-api-layer)
- [Project Workflow](#project-workflow)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [How the System Works End-to-End](#how-the-system-works-end-to-end)
- [Setup Instructions](#setup-instructions)
- [Running the Project](#running-the-project)
- [API Usage](#api-usage)
- [Sample Questions](#sample-questions)
- [What I Learned](#what-i-learned)
- [Current Limitations](#current-limitations)
- [Future Improvements](#future-improvements)
- [Why This Project Matters](#why-this-project-matters)

---

## Project Overview

This project simulates an **internal enterprise AI assistant** that can answer employee questions by combining:

- internal document retrieval using RAG
- tool-based reasoning using an AI agent
- short-term conversation memory
- external factual lookup
- mathematical calculation support
- API-based deployment using FastAPI

The assistant is designed to behave like a simplified internal company assistant that could support teams such as **HR, IT Support, Finance, or Operations**.

It can:

- answer internal policy questions from company documents
- answer general knowledge questions using Wikipedia
- solve mathematical expressions using a calculator tool
- remember user-provided context during the active session
- expose its functionality through a clean API endpoint

---

## Problem Statement

Large language models are useful, but by themselves they have major limitations in enterprise settings:

- they may hallucinate answers
- they do not automatically know internal company policies
- they do not have access to private internal documents unless explicitly connected
- they may not know when to use external tools
- they do not naturally behave like integrated enterprise assistants without orchestration

In a real company environment, employees ask questions such as:

- How many paid leave days do I get?
- How soon are password reset requests resolved?
- Do I need receipts for reimbursement claims?
- Who approves international travel?
- Can the assistant remember my earlier instructions in the same conversation?

A plain chatbot is not enough for this.  
A more useful solution needs:

1. access to internal company knowledge
2. controlled answering based on retrieved evidence
3. external tool use where appropriate
4. memory for follow-up questions
5. an API layer for integration with applications

---

## Solution Overview

To solve this, this project implements a **multi-layer AI assistant architecture**.

The solution combines:

- a **RAG pipeline** to retrieve internal policy information from company documents
- a **LangChain-based agent** to choose the right action or tool
- a **toolset** containing:
  - Company Knowledge Base
  - Wikipedia
  - Calculator
- a **conversation memory layer** for short-term context retention
- a **FastAPI backend** exposing the assistant via `POST /ask`

This design makes the system more realistic for enterprise AI assistant use cases.

---

## Architecture

```text
User Question
     │
     ▼
FastAPI Endpoint (/ask)
     │
     ▼
Agent Layer
 ├── Conversation Memory Check
 ├── Tool Selection Logic
 │    ├── CompanyKnowledgeBase (RAG)
 │    ├── Wikipedia
 │    └── Calculator
 │
 ▼
Final Answer

Internal RAG Flow
Internal Documents
     │
     ▼
Document Loader
     │
     ▼
Text Chunking
     │
     ▼
Embeddings
     │
     ▼
Chroma Vector Store
     │
     ▼
Retriever
     │
     ▼
Relevant Chunks
     │
     ▼
LLM Grounded Answer

Project Workflow

The complete project workflow looks like this:

Phase 1: Prepare the internal knowledge base
create internal company policy documents
load them using LangChain loaders
chunk them into overlapping segments
embed them with OpenAI embeddings
store them in ChromaDB

Phase 2: Build and test the retrieval pipeline
open the vector store
retrieve top relevant chunks for a question
build a grounded prompt
generate a context-based answer

Phase 3: Build the agent
define tools
configure the LLM
add memory
initialize the conversational ReAct-style agent

Phase 4: Integrate all tools into one assistant
route internal questions to CompanyKnowledgeBase
route general factual questions to Wikipedia
route math expressions to Calculator
answer follow-up memory questions directly from session history

Phase 5: Expose as an API
define FastAPI app
create request and response models
expose /ask
test using Swagger UI at /docs

Tech Stack

Core
Python
LLM / AI
OpenAI API
LangChain
LangChain OpenAI
LangChain Community
LangChain Classic
Retrieval
OpenAI Embeddings
ChromaDB
LangChain Chroma
LangChain Text Splitters
API
FastAPI
Uvicorn
Pydantic
Utilities
python-dotenv
wikipedia

ZUORA_PREP/
├── documents/                  # Fake internal company policy documents
│   ├── hr_policy.txt
│   ├── leave_policy.txt
│   ├── it_support.txt
│   ├── finance_policy.txt
│   └── travel_policy.txt
│
├── vectorstore/                # Persisted ChromaDB vector database
│
├── agent.py                    # Main multi-tool memory-aware assistant
├── ingest.py                   # Document ingestion and vector store creation
├── rag_query.py                # Standalone RAG testing script
├── api.py                      # FastAPI backend
├── requirements.txt            # Project dependencies
├── README.md                   # Project documentation
├── .env                        # OpenAI API key (not pushed to GitHub)
└── venv/                       # Local virtual environment (not pushed to GitHub)

How the System Works End-to-End: 

Case 1: Internal company question

Example:

How many paid leave days do full-time employees get?

Workflow:

User sends the question
Agent recognizes it as an internal policy question
Agent calls CompanyKnowledgeBase
Vector store retrieves the most relevant chunks
LLM answers only from that retrieved context
Final grounded answer is returned

Case 2: General knowledge question

Example:

Who is Satya Nadella?

Workflow:

User sends the question
Agent sees it is not an internal policy question
Agent chooses the Wikipedia tool
Wikipedia summary is fetched
Final answer is returned

Case 3: Mathematical question

Example:

What is 25 * 48?

Workflow:

User sends the question
Agent chooses Calculator
Calculator evaluates the expression
Result is returned

Case 4: Memory-based follow-up question

Example:

My name is Sivaji.
What is my name?

Workflow:

User provides name in the conversation
Memory stores that interaction
User later asks follow-up question
Agent answers directly from memory without using a tool