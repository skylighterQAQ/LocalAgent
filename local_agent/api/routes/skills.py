"""Skills API Routes"""
from fastapi import APIRouter, HTTPException
from typing import List

router = APIRouter()


@router.get("/")
async def list_skills():
    """List all available skills"""
    from local_agent.skills.registry import SkillRegistry
    reg = SkillRegistry()
    return {"skills": reg.get_all_info()}


@router.get("/{name}")
async def get_skill(name: str):
    """Get info about a specific skill"""
    from local_agent.skills.registry import SkillRegistry
    reg = SkillRegistry()
    skill = reg.get(name)
    if not skill:
        raise HTTPException(status_code=404, detail=f"Skill '{name}' not found")
    return skill.get_info()
