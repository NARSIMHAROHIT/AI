# My AI engineering learning goals

I'm working through a six-project AI engineering roadmap in my AI repo.

Completed so far: a simple tool-calling agent built from scratch (no framework magic — I wrote the agent loop by hand so I understand how tool calling works), and a Q&A chatbot with conversation memory that trims history to the last 10 exchanges.

Currently building: a RAG personal assistant. Stage 1 is the retrieval core with ChromaDB and Hugging Face sentence-transformers embeddings. Stage 2 will expose it through an MCP server with tools. Stage 3 adds a chat UI.

After that: fine-tuning a model on a specific dataset for a specific use case, LLMOps to measure and improve performance across all projects, and finally a multi-agent system built on LangGraph.

Everything runs on free Groq-hosted LLMs (llama-3.1-8b-instant for speed, llama-3.3-70b-versatile for hard reasoning) with a tier system to switch models based on task complexity.

Longer-term, I want to turn both GitHub repos into a portfolio website.
