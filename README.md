# 🧠 LangMCP

**LangMCP** is a modular, asynchronous AI agent framework that connects **Claude (Anthropic)** to multiple **MCP (Model Context Protocol)** servers through a dynamic **LangGraph** workflow.  
It can process **multiple tool calls and LLM requests concurrently**, coordinating everything through an event-driven graph of four connected nodes.

---

## ⚙️ Architecture Overview

LangMCP is powered by **LangGraph** and uses a **4-node architecture**:

| Node | Role |
|------|------|
| 🧭 **Router Node** | Classifies each user request as `tool_use`, `conversational`, or `handle_mixed`. |
| 🧰 **Tool Node** | Uses the `create_react_agent` interface to invoke the proper MCP tool(s). |
| 💬 **Conversational Node** | Handles pure dialogue and free-form responses through Claude. |
| 🔄 **Mixed Node** | Orchestrates requests that combine tool actions and conversational replies. |

All nodes run asynchronously, enabling **parallel tool execution and LLM inference**.

---

## 🧩 Components

| File | Description |
|------|--------------|
| `agent_loop.py` | 🧠 Main entry point — builds and runs the 4-node LangGraph agent. |
| `mathserver.py` | MCP server for math operations (stdio-based). |
| `weather.py` | MCP server for weather queries. |
| `client.py` | Multi-server client handling async tool discovery and communication. |
| `pyproject.toml` / `uv.lock` | Dependency and environment management (via [uv](https://docs.astral.sh/uv/)). |
| `.env.example` | Template for environment variables (`ANTHROPIC_API_KEY`, etc.). |

---

## 🚀 Running the Agent

### 1. Clone and set up
```bash
git clone https://github.com//billlly4/LangMCP.git
cd LangMCP
uv sync
```

### 2. Configure environment
```bash
cp .env.example .env
```
Add your **Anthropic API key** and any server settings.

### 3. Launch the agent
```bash
uv run python agent_loop.py
```

Then start chatting:
```
You: add 3 to 7
AI: The sum of 3 and 7 is 10.
You: tell me a joke and multiply 4 by 5
AI: 4 × 5 = 20. Also, why did the AI cross the road? To optimize both sides.
```

Type `exit` to quit.

---

## 🧠 Features

- ⚡ **Concurrent LLM & tool execution** using async LangGraph  
- 🤖 **Claude integration** via `langchain_anthropic`  
- 🔗 **Multiple MCP servers** connected through `MultiServerMCPClient`  
- 🧩 **Dynamic routing** between conversational and tool-based responses  
- 🧰 **Composable 4-node graph** for clean, extensible logic

---

## 🧾 License
MIT License © 2025 Arash Khosropour
