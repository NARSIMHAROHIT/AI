"""Project 1: a simple tool-calling agent, built from scratch with LangChain + Groq.

No agent framework magic here — this implements the agent loop by hand so you
can see exactly how agents work:

    1. Send the user's question + tool definitions to the LLM
    2. If the LLM asks to call a tool, run it and send back the result
    3. Repeat until the LLM answers in plain text

Usage:
    python agent.py "What is 15% of 2847?"
    python agent.py --tier smart "Explain step by step: (23*4) + sqrt(144)"
"""

import argparse
import math
from datetime import datetime

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq

load_dotenv()  # reads GROQ_API_KEY from .env

# --- Model switching: pick a tier based on task complexity ---
MODELS = {
    "fast": "llama-3.1-8b-instant",        # quick, cheap — simple queries
    "balanced": "openai/gpt-oss-20b",      # good middle ground
    "smart": "llama-3.3-70b-versatile",    # complex reasoning
}


# --- Tools the agent can use ---
@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression, e.g. '2 * (3 + 4)' or 'sqrt(144)'."""
    allowed = {"sqrt": math.sqrt, "pi": math.pi, "e": math.e, "pow": pow, "abs": abs}
    try:
        return str(eval(expression, {"__builtins__": {}}, allowed))
    except Exception as exc:
        return f"Error: {exc}"


@tool
def current_datetime(dummy: str = "") -> str:
    """Get the current date and time."""
    return datetime.now().strftime("%A, %B %d, %Y at %H:%M:%S")


TOOLS = [calculator, current_datetime]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}


def run_agent(question: str, tier: str = "fast", max_steps: int = 5) -> str:
    """The agent loop: think → call tools if needed → answer."""
    llm = ChatGroq(model=MODELS[tier], temperature=0).bind_tools(TOOLS)

    messages = [
        SystemMessage("You are a helpful assistant. Use tools when they help. "
                      "Answer concisely."),
        HumanMessage(question),
    ]

    for _ in range(max_steps):
        response = llm.invoke(messages)
        messages.append(response)

        if not response.tool_calls:          # plain answer -> we're done
            return response.content

        for call in response.tool_calls:     # run each requested tool
            result = TOOLS_BY_NAME[call["name"]].invoke(call["args"])
            print(f"  [tool] {call['name']}({call['args']}) -> {result}")
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    return "Stopped: too many tool-call steps."


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple tool-calling agent")
    parser.add_argument("question", help="What to ask the agent")
    parser.add_argument("--tier", choices=MODELS, default="fast",
                        help="Model tier: fast, balanced, or smart")
    args = parser.parse_args()

    print(f"Model: {MODELS[args.tier]}\n")
    print(run_agent(args.question, args.tier))
