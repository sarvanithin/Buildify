import asyncio
import io
import json

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Literal, Optional

from generator import generate_floor_plan
from exporter import export_to_dxf
from cost import estimate_cost, REGION_MULTIPLIERS
from scoring import score_design
from moe.inference import predict_floor_plan, load_model
from moe.api_auth import key_store, get_api_key
from moe.config import MOEConfig
from moe.experts import EXPERT_NAMES

app = FastAPI(title="Buildify API")


@app.on_event("startup")
async def startup_event():
    from rag import rag
    try:
        await rag.initialize()
    except Exception as e:
        print(f"[RAG] Init warning: {e} — generation will work without RAG context.")
    # Pre-load MOE model
    try:
        load_model()
    except Exception as e:
        print(f"[MOE] Init warning: {e} — MOE generation may be unavailable.")


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


class AuthRequest(BaseModel):
    email: str = ""
    tier: str = "free"


class UpgradeRequest(BaseModel):
    api_key: str
    tier: str


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


# ── MOE Endpoints ─────────────────────────────────────────────────────────────

@app.post("/api/generate/moe")
async def generate_moe(constraints: Constraints, request: Request):
    """Generate floor plans using the MOE AI model."""
    try:
        # Check API key for tier limits
        api_key = get_api_key(request)
        config = MOEConfig()
        num_variants = 3  # default

        if api_key:
            record = key_store.validate_key(api_key)
            if record:
                if not key_store.check_limit(api_key):
                    raise HTTPException(
                        status_code=429,
                        detail="Daily generation limit reached. Upgrade to Pro for unlimited."
                    )
                num_variants = config.TIER_VARIANTS.get(record["tier"], 3)
                key_store.record_usage(api_key, "generation")

        c = constraints.model_dump()
        result = predict_floor_plan(c, num_variants=num_variants)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/moe/experts")
async def moe_experts(constraints: Constraints):
    """Get expert activation weights for given constraints."""
    try:
        c = constraints.model_dump()
        result = predict_floor_plan(c, num_variants=1)
        return {
            "expert_weights": result["expert_weights"],
            "expert_names": EXPERT_NAMES,
            "confidence": result["confidence"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Auth Endpoints ────────────────────────────────────────────────────────────

@app.post("/api/auth/register")
async def auth_register(req: AuthRequest):
    """Register a new API key."""
    try:
        record = key_store.create_key(tier=req.tier, email=req.email)
        return {
            "api_key": record["key"],
            "tier": record["tier"],
            "message": f"API key created. Tier: {record['tier']}",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/auth/usage")
async def auth_usage(request: Request):
    """Get usage stats for the current API key."""
    api_key = get_api_key(request)
    if not api_key:
        raise HTTPException(status_code=401, detail="API key required in X-API-Key header.")
    usage = key_store.get_usage(api_key)
    if not usage:
        raise HTTPException(status_code=401, detail="Invalid API key.")
    return usage


@app.post("/api/auth/upgrade")
async def auth_upgrade(req: UpgradeRequest):
    """Upgrade an API key to a higher tier."""
    result = key_store.upgrade_key(req.api_key, req.tier)
    if not result:
        raise HTTPException(status_code=404, detail="API key not found.")
    return {"tier": req.tier, "message": f"Upgraded to {req.tier} tier."}


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
