# Project 1: Simple Agent

My first AI agent. The goal here was to understand what an "agent" actually is before reaching for frameworks that hide the details.

## What makes this an agent (and not just a chatbot)

A chatbot takes your text and returns text. An agent can **decide to act** — it looks at your question, realizes it needs a tool (like a calculator), calls that tool, reads the result, and only then answers. The loop in `agent.py` is the whole trick:

1. Send the question + a list of available tools to the LLM
2. The LLM either answers directly, or asks to call a tool
3. If it asked for a tool, run it, feed the result back, and go to step 2

That loop is the foundation of every agent framework. I wrote it by hand here (~20 lines) so the later projects that use LangGraph won't feel like magic.

## Model switching

One of the goals of this whole repo is routing between models based on task complexity. `agent.py` has three tiers, all running on Groq's free API:

| Tier | Model | When to use |
|------|-------|-------------|
| `fast` | llama-3.1-8b-instant | simple questions, default |
| `balanced` | openai/gpt-oss-20b | mid-complexity |
| `smart` | llama-3.3-70b-versatile | multi-step reasoning |

## Setup

```bash
pip install -r ../requirements.txt
cp ../.env.example ../.env    # then paste your GROQ_API_KEY into .env
```

## Run

```bash
python agent.py "What is 15% of 2847?"
python agent.py --tier smart "What day is it today, and what is sqrt(144) * 3?"
```

You'll see the tool calls printed as they happen:

```
Model: llama-3.1-8b-instant

  [tool] calculator({'expression': '2847 * 0.15'}) -> 427.05
15% of 2847 is 427.05.
```

## Tools available

- `calculator` — evaluates math expressions safely (no access to Python builtins)
- `current_datetime` — returns the current date/time, something no LLM can know on its own

## What I learned

The LLM never runs anything itself — it only *asks* for tool calls as structured JSON, and my code decides whether to execute them. That separation is what makes agents controllable and safe.
