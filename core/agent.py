"""LangGraph-based ReAct agent for LocalAgent."""
from typing import Annotated, Any, Dict, List, Sequence, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages

from core.config_loader import get_config
from core.skill_base import get_registry


# ── Graph state ────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


# ── System prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are LocalAgent, a powerful AI assistant built on Ollama and LangGraph.
You have access to a variety of tools (skills) that you can use to help users.

Available capabilities:
- Browse the web and read web pages
- Execute Python code and scripts
- Search the internet
- Read and write local files
- Any custom skills that have been configured

Guidelines:
- Always think step by step before using tools
- When writing Python code, include clear print() statements so output is visible
- For web tasks, first search for information, then fetch specific pages if needed
- Be concise and helpful in your responses
- If a task requires multiple steps, break it down and work through each step

You are running locally via Ollama. Your responses are private and secure."""


# ── Agent factory ──────────────────────────────────────────────────────────────

def create_agent(extra_tools: List[Any] | None = None):
    """Build the LangGraph ReAct agent with all registered tools."""
    cfg = get_config()
    registry = get_registry()

    # Gather tools from all loaded skills
    tools = registry.get_all_tools()
    if extra_tools:
        tools.extend(extra_tools)

    # Create Ollama LLM
    llm = ChatOllama(
        base_url=cfg.ollama.base_url,
        model=cfg.ollama.model,
        temperature=cfg.ollama.temperature,
        num_ctx=cfg.ollama.num_ctx,
    )

    if tools:
        llm_with_tools = llm.bind_tools(tools)
    else:
        llm_with_tools = llm

    # ── Node: call the model ──────────────────────────────────────────────────

    def call_model(state: AgentState) -> AgentState:
        messages = state["messages"]
        # Inject system prompt if not present
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    # ── Build graph ───────────────────────────────────────────────────────────

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)

    if tools:
        tool_node = ToolNode(tools)
        workflow.add_node("tools", tool_node)
        workflow.add_conditional_edges("agent", tools_condition)
        workflow.add_edge("tools", "agent")

    workflow.set_entry_point("agent")
    graph = workflow.compile()

    return graph, llm


def run_agent(graph, user_input: str, history: List[BaseMessage] | None = None) -> str:
    """Run the agent with a user message and return the final response."""
    cfg = get_config()
    messages = list(history or [])
    messages.append(HumanMessage(content=user_input))

    final_state = graph.invoke(
        {"messages": messages},
        config={"recursion_limit": cfg.agent.max_iterations * 2},
    )

    # Return the last AI message content
    for msg in reversed(final_state["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content

    return "(no response)"
