"""Project 2: a Q&A chatbot with conversation memory.

Project 1's agent forgot everything between runs. This chatbot keeps the
conversation history and sends it back to the LLM on every turn — that's
all "memory" is: the full transcript, replayed.

Because history grows every turn (and models have context limits), the
history gets trimmed to the most recent turns before each call.

Usage:
    python chatbot.py                 # default: fast tier
    python chatbot.py --tier smart

Commands inside the chat:
    /tier fast|balanced|smart   switch models mid-conversation
    /clear                      wipe the memory
    /exit                       quit
"""

import argparse

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq

load_dotenv()

MODELS = {
    "fast": "llama-3.1-8b-instant",
    "balanced": "openai/gpt-oss-20b",
    "smart": "llama-3.3-70b-versatile",
}

SYSTEM_PROMPT = (
    "You are a friendly, concise Q&A assistant. Use the conversation history "
    "to resolve references like 'it', 'that', or 'the one I mentioned'. "
    "If you don't know something, say so."
)

MAX_TURNS = 10  # keep the last N user+assistant exchanges


def trim_history(history: list) -> list:
    """Keep memory from growing forever: retain only the last MAX_TURNS exchanges."""
    return history[-(MAX_TURNS * 2):]


def chat(tier: str):
    history = []  # list of HumanMessage / AIMessage — this IS the memory
    print(f"Chatbot ready (model: {MODELS[tier]}). Type /exit to quit.\n")

    while True:
        try:
            user_input = input("you: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye!")
            break

        if not user_input:
            continue

        # --- commands ---
        if user_input == "/exit":
            print("bye!")
            break
        if user_input == "/clear":
            history = []
            print("(memory cleared)\n")
            continue
        if user_input.startswith("/tier"):
            _, _, new_tier = user_input.partition(" ")
            if new_tier in MODELS:
                tier = new_tier
                print(f"(switched to {MODELS[tier]})\n")
            else:
                print(f"(unknown tier — pick from: {', '.join(MODELS)})\n")
            continue

        # --- normal turn: system prompt + trimmed history + new question ---
        history.append(HumanMessage(user_input))
        llm = ChatGroq(model=MODELS[tier], temperature=0.7)
        messages = [SystemMessage(SYSTEM_PROMPT)] + trim_history(history)

        response = llm.invoke(messages)
        history.append(AIMessage(response.content))
        print(f"\nbot: {response.content}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Q&A chatbot with memory")
    parser.add_argument("--tier", choices=MODELS, default="fast")
    chat(parser.parse_args().tier)
