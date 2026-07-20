"""
Chat API Routes – HTTP (SSE) + WebSocket streaming.

Session lifecycle:
  - Sessions are keyed by session_id.
  - The agent is recreated only when model or skill changes.
  - Sessions are never auto-expired here (use an LRU cache in production).
"""
from __future__ import annotations

import asyncio
import json
import uuid
from typing import Dict, Optional, Tuple

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

router = APIRouter()

# (session_id) → (LocalAgent, model, skill, provider)
_sessions: Dict[str, Tuple] = {}


# ── Pydantic models ───────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    model: Optional[str] = None
    provider: Optional[str] = None  # "ollama" or "openai"
    skill: Optional[str] = None
    stream: bool = False


class ChatResponse(BaseModel):
    response: str
    session_id: str
    model: str


# ── Session helpers ───────────────────────────────────────────────────────────

def _ensure_workspace_context() -> None:
    """
    Ensure a workspace context is set so that relative paths in fs_* tools
    resolve to a predictable location.

    Priority:
      1. Already set (skip)
      2. LOCALAGENT_WORKSPACE/ directory next to the project root
      3. Current working directory as fallback
    """
    from local_agent.cli.workspace.context import get_active_workspace, set_active_workspace
    if get_active_workspace() is not None:
        return

    from pathlib import Path
    from local_agent.core.config import PROJECT_ROOT
    from local_agent.cli.workspace.manager import WorkspaceManager, find_workspace
    from local_agent.cli.workspace.config import WorkspaceConfig

    # Try auto-discovering a workspace.yaml from the cwd upward
    ws = find_workspace()
    if ws is None:
        # Fall back to LOCALAGENT_WORKSPACE next to project root
        candidate = PROJECT_ROOT / "LOCALAGENT_WORKSPACE"
        candidate.mkdir(parents=True, exist_ok=True)
        ws = WorkspaceManager.from_config(
            WorkspaceConfig(
                name="default",
                default_dir=str(candidate),
                terminal_dir=str(candidate),
                description="Auto-created workspace for API mode",
            )
        )
    set_active_workspace(ws)


def _get_agent(session_id: str, model: Optional[str], skill: Optional[str], provider: Optional[str] = None):
    """
    Return the agent for *session_id*, creating or re-creating it when
    the model, skill, or provider has changed since the session was last used.
    """
    from local_agent.core.agent import LocalAgent

    # Always make sure file tools know where to write
    _ensure_workspace_context()

    existing = _sessions.get(session_id)
    if existing:
        agent, stored_model, stored_skill, stored_provider = existing
        # Re-create only when model/skill/provider actually changed
        if stored_model == model and stored_skill == skill and stored_provider == provider:
            return agent

    agent = LocalAgent.create(model=model, skill=skill, provider=provider)
    _sessions[session_id] = (agent, model, skill, provider)
    return agent


# ── HTTP endpoints ────────────────────────────────────────────────────────────

@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a message; returns full response or SSE stream."""
    session_id = request.session_id or str(uuid.uuid4())
    agent = _get_agent(session_id, request.model, request.skill, request.provider)

    if request.stream:
        def _generate():
            for chunk in agent.stream(request.message):
                yield f"data: {json.dumps({'chunk': chunk, 'session_id': session_id})}\n\n"
            yield f"data: {json.dumps({'done': True, 'session_id': session_id})}\n\n"

        return StreamingResponse(_generate(), media_type="text/event-stream")

    response = agent.chat(request.message)
    return ChatResponse(response=response, session_id=session_id, model=agent.model)


@router.delete("/{session_id}")
async def clear_session(session_id: str):
    """Clear the conversation history of a session (keeps the agent alive)."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    agent, model, skill, provider = _sessions[session_id]
    agent.reset_conversation()
    return {"status": "cleared", "session_id": session_id}


@router.get("/sessions")
async def list_sessions():
    """List active session IDs."""
    return {"sessions": list(_sessions.keys()), "count": len(_sessions)}


# ── WebSocket ─────────────────────────────────────────────────────────────────

@router.websocket("/ws/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time streaming chat.

    Protocol (JSON):
      Client → Server: {"message": "...", "model": "...", "provider": "...", "skill": "..."}
      Server → Client: {"type": "start"|"chunk"|"done"|"error", ...}
    """
    await websocket.accept()
    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)

            message: str = msg.get("message", "").strip()
            if not message:
                await websocket.send_text(json.dumps({"type": "error", "message": "Empty message"}))
                continue

            model = msg.get("model")
            skill = msg.get("skill")
            provider = msg.get("provider")
            agent = _get_agent(session_id, model, skill, provider)

            await websocket.send_text(json.dumps({"type": "start", "session_id": session_id}))

            full_response = ""
            for chunk in agent.stream(message):
                full_response += chunk
                await websocket.send_text(
                    json.dumps({"type": "chunk", "content": chunk, "session_id": session_id})
                )
                await asyncio.sleep(0)  # yield to event loop

            await websocket.send_text(
                json.dumps({"type": "done", "full_response": full_response, "session_id": session_id})
            )

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await websocket.send_text(json.dumps({"type": "error", "message": str(exc)}))
        except Exception:
            pass
