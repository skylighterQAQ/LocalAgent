"""
FastAPI Application – LocalAgent Web API.

Uses the modern ``lifespan`` context manager (FastAPI ≥ 0.95) instead of
the deprecated ``@app.on_event("startup")`` decorator.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from local_agent.api.routes import chat, models, skills, tools, mcp as mcp_routes


# ── Lifespan (startup / shutdown) ────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load tools, skills, and MCP servers at startup; clean up on shutdown."""
    from local_agent.core.config import get_settings
    from local_agent.skills.loader import SkillLoader
    from local_agent.tools.builtin import load_all_builtin_tools
    from local_agent.mcp import MCPManager

    settings = get_settings()
    n_tools = load_all_builtin_tools()
    loader = SkillLoader()
    n_skills = loader.load_builtin_skills()

    # Ensure workspace context is set at startup so fs_* tools always have a
    # valid base directory even in API mode (where no CLI workspace init runs).
    from local_agent.api.routes.chat import _ensure_workspace_context
    _ensure_workspace_context()

    # Load MCP servers and store manager in app state for route reuse
    mcp_manager = MCPManager.from_config_path(settings.mcp_config_path)
    if settings.mcp_enabled:
        mcp_tools = mcp_manager.load_all()
        if mcp_tools:
            from local_agent.tools.registry import ToolRegistry
            reg = ToolRegistry()
            for tool in mcp_tools:
                reg.register(tool)
            n_tools += len(mcp_tools)

    app.state.mcp_manager = mcp_manager

    import logging
    logging.getLogger(__name__).info(
        "Startup complete: %d tools, %d skills loaded", n_tools, n_skills
    )

    yield  # application runs here

    # Shutdown: close MCP connections
    mcp_manager.stop_all()


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="LocalAgent API",
    description="Local AI Agent powered by Ollama and LangGraph",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS – allow all origins for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────

app.include_router(chat.router,        prefix="/api/chat",   tags=["chat"])
app.include_router(skills.router,      prefix="/api/skills", tags=["skills"])
app.include_router(tools.router,       prefix="/api/tools",  tags=["tools"])
app.include_router(models.router,      prefix="/api/models", tags=["models"])
app.include_router(mcp_routes.router,  prefix="/api/mcp",    tags=["mcp"])


@app.get("/api/health", tags=["meta"])
async def health():
    """Health check endpoint."""
    return {"status": "ok", "version": "0.1.0"}


# ── SPA static file serving ───────────────────────────────────────────────────

_frontend_dist = Path(__file__).parent.parent.parent / "frontend" / "dist"

if _frontend_dist.exists():
    _assets = _frontend_dist / "assets"
    if _assets.exists():
        app.mount("/assets", StaticFiles(directory=str(_assets)), name="assets")

    @app.get("/", include_in_schema=False)
    async def serve_root():
        return FileResponse(str(_frontend_dist / "index.html"))

    @app.get("/{path:path}", include_in_schema=False)
    async def serve_spa(path: str):
        file = _frontend_dist / path
        if file.exists() and file.is_file():
            return FileResponse(str(file))
        return FileResponse(str(_frontend_dist / "index.html"))
