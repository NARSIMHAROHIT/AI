"""An agent that discovers and uses MCP tools.

Project 1's agent had tools hard-coded in the same file. This agent gets
its tools from MCP servers listed in mcp_config.json — it launches each
server, asks "what tools do you have?", hands the tool list to the Groq
LLM, and executes whatever the LLM decides to call.

Adding a new capability (Gmail, maps, filesystem...) = adding one entry
to mcp_config.json. No code changes. Servers that fail to start (missing
API key, not set up yet) are skipped with a warning.

Usage:
    python agent_mcp.py --list                          # show discovered tools
    python agent_mcp.py "What projects has Rohit built?"
    python agent_mcp.py --tier smart "..."
"""

import argparse
import asyncio
import json
import os
from contextlib import AsyncExitStack
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

HERE = Path(__file__).parent
CONFIG_FILE = HERE / "mcp_config.json"

MODELS = {
    "fast": "llama-3.1-8b-instant",
    "balanced": "openai/gpt-oss-20b",
    "smart": "llama-3.3-70b-versatile",
    "max": "openai/gpt-oss-120b",
}

SYSTEM_PROMPT = (
    "You are Rohit's personal assistant with access to tools: his document "
    "search, and possibly Gmail, Google Maps, and others. Use tools whenever "
    "they help. Never invent information a tool could verify. Answer concisely."
)


async def connect_servers(stack: AsyncExitStack, extra_env: dict | None = None) -> dict:
    """Launch every server in mcp_config.json and discover its tools.

    `extra_env`: session-only secrets (e.g. a visitor's own API key) merged
    into each server's environment — never written to disk.
    Returns {tool_name: {"session": ..., "schema": ...}}.
    Servers that fail to start are skipped with a warning.
    """
    config = json.loads(CONFIG_FILE.read_text())
    registry = {}

    for name, cfg in config["servers"].items():
        try:
            env = {**os.environ, **cfg.get("env", {}), **(extra_env or {})}
            params = StdioServerParameters(
                command=cfg["command"],
                args=[str(HERE / a) if a.endswith(".py") else a for a in cfg["args"]],
                env=env,
            )
            read, write = await stack.enter_async_context(stdio_client(params))
            session = await stack.enter_async_context(ClientSession(read, write))
            await asyncio.wait_for(session.initialize(), timeout=30)

            tools = (await session.list_tools()).tools
            for tool in tools:
                registry[tool.name] = {
                    "server": name,
                    "session": session,
                    "schema": {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description or "",
                            "parameters": tool.inputSchema,
                        },
                    },
                }
            print(f"[connected] {name}: {[t.name for t in tools]}")
        except Exception as exc:
            print(f"[skipped] {name}: {type(exc).__name__} — check setup ({exc})")

    return registry


async def run_agent(question: str, tier: str = "fast", history: list | None = None,
                    trace: list | None = None, extra_env: dict | None = None,
                    max_steps: int = 6) -> str:
    """Run the agent loop. `history` = prior chat messages for memory.
    `trace` (optional list) collects tool-call descriptions for UIs.
    `extra_env` = session-only secrets passed to MCP servers."""
    async with AsyncExitStack() as stack:
        registry = await connect_servers(stack, extra_env)
        if not registry:
            return "No MCP tools available — check mcp_config.json setup."

        llm = ChatGroq(model=MODELS[tier], temperature=0).bind_tools(
            [t["schema"] for t in registry.values()]
        )
        messages = ([SystemMessage(SYSTEM_PROMPT)] + (history or [])
                    + [HumanMessage(question)])

        for _ in range(max_steps):
            response = await llm.ainvoke(messages)
            messages.append(response)

            if not response.tool_calls:
                return response.content

            for call in response.tool_calls:
                if call["name"] not in registry:
                    text = f"Unknown tool: {call['name']}"
                else:
                    entry = registry[call["name"]]
                    result = await entry["session"].call_tool(call["name"], call["args"])
                    text = "\n".join(c.text for c in result.content if c.type == "text")
                line = f"{call['name']}({call['args']})"
                print(f"  [tool] {line} -> {text[:120]}...")
                if trace is not None:
                    trace.append({"call": line, "result": text[:400]})
                messages.append(ToolMessage(content=text, tool_call_id=call["id"]))

        return "Stopped: too many tool-call steps."


async def list_tools():
    async with AsyncExitStack() as stack:
        registry = await connect_servers(stack)
        print(f"\n{len(registry)} tools available:")
        for name, entry in registry.items():
            desc = entry["schema"]["function"]["description"].split("\n")[0]
            print(f"  [{entry['server']}] {name}: {desc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCP-powered agent")
    parser.add_argument("question", nargs="?", help="What to ask")
    parser.add_argument("--tier", choices=MODELS, default="fast")
    parser.add_argument("--list", action="store_true", help="List discovered tools")
    args = parser.parse_args()

    if args.list:
        asyncio.run(list_tools())
    elif args.question:
        print(asyncio.run(run_agent(args.question, args.tier)))
    else:
        parser.print_help()
