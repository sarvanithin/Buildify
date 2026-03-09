import asyncio
import io
import json

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Literal, Optional

from generator import generate_floor_plan
from exporter import export_to_dxf
from cost import estimate_cost, REGION_MULTIPLIERS
from scoring import score_design

app = FastAPI(title="Buildify API")


@app.on_event("startup")
async def startup_event():
    from rag import rag
    try:
        await rag.initialize()
    except Exception as e:
        print(f"[RAG] Init warning: {e} — generation will work without RAG context.")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request models ────────────────────────────────────────────────────────────

class Constraints(BaseModel):
    bedrooms: int = 3
    bathrooms: int = 2
    sqft: int = 1800
    stories: int = 1
    style: str = "modern"
    openPlan: bool = False
    primarySuite: bool = True
    homeOffice: bool = False
    formalDining: bool = False
    garage: str = "2car"
    laundry: str = "room"
    outdoor: str = "patio"
    ceilingHeight: str = "standard"


class ExportRequest(BaseModel):
    floor_plan: dict


class CostRequest(BaseModel):
    floor_plan: dict
    region: str = "National Average"


class ScoreRequest(BaseModel):
    floor_plan: dict


class ChatMessage(BaseModel):
    role: str   # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    floor_plan: dict
    messages: list[ChatMessage]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.post("/api/generate")
async def generate(constraints: Constraints):
    try:
        c = constraints.model_dump()
        plans = await asyncio.gather(
            generate_floor_plan(c, 0),
            generate_floor_plan(c, 1),
            generate_floor_plan(c, 2),
        )
        return {"plans": list(plans)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/export/dxf")
async def export_dxf(request: ExportRequest):
    try:
        dxf_bytes = export_to_dxf(request.floor_plan)
        name = request.floor_plan.get("name", "floor_plan").replace(" ", "_")
        return StreamingResponse(
            io.BytesIO(dxf_bytes),
            media_type="application/octet-stream",
            headers={"Content-Disposition": f'attachment; filename="{name}.dxf"'},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/cost/regions")
async def cost_regions():
    return {"regions": list(REGION_MULTIPLIERS.keys())}


@app.post("/api/cost")
async def cost(request: CostRequest):
    try:
        return estimate_cost(request.floor_plan, request.region)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/score")
async def score(request: ScoreRequest):
    try:
        return score_design(request.floor_plan)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


CHAT_SYSTEM = """You are Buildify AI, an expert residential architect assistant.
You help users refine their floor plans. The user will share their current floor plan data
and ask questions or request modifications.

When asked to modify a plan, respond with:
1. A brief explanation of your suggested changes (2-3 sentences)
2. A JSON block inside ```json ... ``` with the COMPLETE updated floor plan (same structure, all rooms)

When answering questions (not modifications), just respond with helpful architectural advice.
Keep answers concise and practical. Focus on US residential standards.
"""


@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        plan_summary = _summarize_plan(request.floor_plan)
        system_context = f"{CHAT_SYSTEM}\n\nCurrent floor plan:\n{plan_summary}"

        messages = [{"role": "system", "content": system_context}]
        for m in request.messages:
            messages.append({"role": m.role, "content": m.content})

        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                "http://localhost:11434/api/chat",
                json={"model": "llama3.2", "messages": messages, "stream": False},
            )
            resp.raise_for_status()
            data = resp.json()

        reply = data.get("message", {}).get("content", "Sorry, no response.")
        updated_plan = _extract_plan_from_reply(reply, request.floor_plan)

        return {"reply": reply, "updated_plan": updated_plan}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _summarize_plan(plan: dict) -> str:
    rooms = plan.get("rooms", [])
    lines = [
        f"Name: {plan.get('name', 'Plan')}",
        f"Footprint: {plan.get('totalWidth', 0)}ft × {plan.get('totalHeight', 0)}ft",
        f"Ceiling height: {plan.get('ceilingHeight', 9)}ft",
        f"Rooms ({len(rooms)}):",
    ]
    for r in rooms:
        lines.append(f"  - {r['name']} ({r.get('type','')}) {r['width']}×{r['height']}ft at ({r['x']},{r['y']})")
    return "\n".join(lines)


def _extract_plan_from_reply(reply: str, original: dict) -> Optional[dict]:
    import re
    m = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", reply)
    if not m:
        return None
    try:
        plan = json.loads(m.group(1))
        if "rooms" in plan:
            return plan
    except Exception:
        pass
    return None


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
