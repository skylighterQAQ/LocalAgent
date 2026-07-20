"""
Memory Tools - Save and retrieve long-term memories
"""
from local_agent.core.tools import tool


@tool
def memory_save(content: str, category: str = "general", tags: str = "") -> str:
    """
    Save a piece of information to long-term memory.
    Args:
        content: The information to remember
        category: Category for organizing memories (e.g., 'fact', 'preference', 'task')
        tags: Comma-separated tags for easier retrieval
    """
    try:
        from local_agent.core.memory import LongTermMemory
        mem = LongTermMemory()
        tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
        mem_id = mem.save(content, category=category, tags=tag_list)
        return f"Memory saved (ID: {mem_id})\nCategory: {category}\nContent: {content[:100]}"
    except Exception as e:
        return f"Error saving memory: {e}"


@tool
def memory_search(query: str, max_results: int = 5) -> str:
    """
    Search long-term memory for relevant information.
    Args:
        query: What to search for
        max_results: Maximum number of results to return
    """
    try:
        from local_agent.core.memory import LongTermMemory
        mem = LongTermMemory()
        results = mem.search(query, n_results=max_results)
        if not results:
            return f"No memories found for: {query}"
        lines = [f"Memory search results for '{query}':"]
        for i, r in enumerate(results, 1):
            lines.append(f"\n[{i}] Category: {r.get('category', 'general')}")
            lines.append(f"    Content: {r.get('content', '')[:200]}")
            lines.append(f"    Score: {r.get('score', 0):.3f}")
        return "\n".join(lines)
    except Exception as e:
        return f"Error searching memory: {e}"


@tool
def memory_list(category: str = "", limit: int = 20) -> str:
    """
    List saved memories.
    Args:
        category: Filter by category (empty = show all)
        limit: Maximum number of memories to show
    """
    try:
        from local_agent.core.memory import LongTermMemory
        mem = LongTermMemory()
        memories = mem.list_all(category=category, limit=limit)
        if not memories:
            return "No memories found" + (f" in category '{category}'" if category else "")
        lines = [f"Memories ({len(memories)} found):"]
        for m in memories:
            lines.append(f"\nID: {m.get('id', 'N/A')} | Category: {m.get('category', 'general')}")
            lines.append(f"  {m.get('content', '')[:150]}")
        return "\n".join(lines)
    except Exception as e:
        return f"Error listing memories: {e}"


@tool
def memory_delete(memory_id: str) -> str:
    """Delete a memory by its ID."""
    try:
        from local_agent.core.memory import LongTermMemory
        mem = LongTermMemory()
        success = mem.delete(memory_id)
        return f"Memory {memory_id} deleted" if success else f"Memory {memory_id} not found"
    except Exception as e:
        return f"Error deleting memory: {e}"


memory_save.metadata = memory_save.metadata or {}
memory_save.metadata["category"] = "memory"
memory_search.metadata = memory_search.metadata or {}
memory_search.metadata["category"] = "memory"
memory_list.metadata = memory_list.metadata or {}
memory_list.metadata["category"] = "memory"
memory_delete.metadata = memory_delete.metadata or {}
memory_delete.metadata["category"] = "memory"

TOOLS = [memory_save, memory_search, memory_list, memory_delete]
