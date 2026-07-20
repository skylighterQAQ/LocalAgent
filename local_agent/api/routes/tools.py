"""Tools API Routes"""
from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def list_tools():
    """List all registered tools"""
    from local_agent.tools.registry import ToolRegistry
    reg = ToolRegistry()
    return {"tools": reg.get_tool_info(), "total": len(reg.get_all())}


@router.get("/categories")
async def list_categories():
    """List all tool categories"""
    from local_agent.tools.registry import ToolRegistry
    reg = ToolRegistry()
    result = {}
    for cat in reg.get_categories():
        result[cat] = [t.name for t in reg.get_by_category(cat)]
    return {"categories": result}
