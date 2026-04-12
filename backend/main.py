import asyncio
import io
import json
from pathlib import Path

# Load .env if present (for local dev)
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            import os as _os
            if not _os.environ.get(_k.strip()):
                _os.environ[_k.strip()] = _v.strip()

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


import os

ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://localhost:3000"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
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


def validate_constraints_feasibility(c: dict) -> list:
    """
    Check whether the requested constraints are physically feasible.
    Returns a list of issues (may be empty). Each issue:
      {"field": str, "severity": "error"|"warning", "message": str, "detail": str}
    Errors block generation; warnings are informational only.
    """
    issues = []
    sqft         = c.get("sqft", 1800)
    bedrooms     = c.get("bedrooms", 3)       # total incl. primary
    bathrooms    = c.get("bathrooms", 2)
    primary_suite = c.get("primarySuite", True)
    home_office  = c.get("homeOffice", False)
    formal_dining = c.get("formalDining", False)
    laundry      = c.get("laundry", "room")
    garage       = c.get("garage", "2car")
    stories      = c.get("stories", 1)

    secondary = max(0, bedrooms - 1)
    shared_baths = max(0, bathrooms - 1)

    # ── Base overhead (kitchen + living + hallway + foyer) ────────────────
    min_sqft = 168 + 120 + 200 + 40   # = 528

    # ── Primary bedroom cluster ───────────────────────────────────────────
    min_sqft += 240 if primary_suite else 168   # bed + ensuite + closet

    # ── Secondary bedrooms ────────────────────────────────────────────────
    min_sqft += secondary * 100

    # ── Shared bathrooms ──────────────────────────────────────────────────
    min_sqft += shared_baths * 40

    # ── Optional rooms ────────────────────────────────────────────────────
    if home_office:    min_sqft += 90
    if formal_dining:  min_sqft += 121
    if laundry == "room": min_sqft += 30

    # ── ERROR: total sqft below minimum ───────────────────────────────────
    if sqft < min_sqft:
        parts = []
        if secondary: parts.append(f"{secondary} secondary bedroom{'s' if secondary != 1 else ''}")
        parts.append(f"{bathrooms} bathroom{'s' if bathrooms != 1 else ''}")
        if home_office: parts.append("home office")
        if formal_dining: parts.append("formal dining")
        issues.append({
            "field": "sqft",
            "severity": "error",
            "message": "Not enough space for this configuration.",
            "detail": (
                f"Your selections ({', '.join(parts)}) require at least {min_sqft:,} sqft of "
                f"living space. You set {sqft:,} sqft. "
                f"Increase the size to {min_sqft:,}+ sqft, or remove bedrooms/rooms."
            ),
        })

    # ── ERROR: too many bedrooms for sqft ────────────────────────────────
    base_overhead = 528 + (240 if primary_suite else 168)
    max_secondary = max(0, (sqft - base_overhead) // 100)
    if secondary > max_secondary and sqft >= min_sqft:
        issues.append({
            "field": "bedrooms",
            "severity": "error",
            "message": f"{bedrooms} bedrooms is not feasible in {sqft:,} sqft.",
            "detail": (
                f"After essential rooms, only {int(sqft - base_overhead):,} sqft remains for "
                f"secondary bedrooms ({int(max_secondary)} max at 100 sqft each). "
                f"Use {int(max_secondary) + 1} total bedrooms or increase to "
                f"{int(base_overhead + secondary * 100):,}+ sqft."
            ),
        })

    # ── WARNING: bathrooms > bedrooms + 1 ────────────────────────────────
    if bathrooms > bedrooms + 1:
        issues.append({
            "field": "bathrooms",
            "severity": "warning",
            "message": f"{bathrooms} bathrooms for {bedrooms} bedrooms is unusual.",
            "detail": (
                f"Standard practice is 1 bathroom per bedroom or 1 shared bathroom per "
                f"2 bedrooms. Consider {min(bathrooms, bedrooms)} bathrooms."
            ),
        })

    # ── WARNING: 2-story with tiny footprint ─────────────────────────────
    if stories == 2 and sqft < 1200:
        issues.append({
            "field": "stories",
            "severity": "warning",
            "message": "Two-story layout under 1,200 sqft is cramped.",
            "detail": "Staircase overhead is significant in small homes. Consider single-story or increase to 1,200+ sqft.",
        })

    # ── WARNING: 3-car garage on small home ──────────────────────────────
    if garage == "3car" and sqft < 1800:
        issues.append({
            "field": "garage",
            "severity": "warning",
            "message": "A 3-car garage is disproportionate for this home size.",
            "detail": f"3-car garages suit homes 1,800+ sqft. With {sqft:,} sqft, a 1 or 2-car garage is more appropriate.",
        })

    return issues


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

        # Feasibility check — block generation for impossible configurations
        issues = validate_constraints_feasibility(c)
        hard_errors = [i for i in issues if i["severity"] == "error"]
        if hard_errors:
            raise HTTPException(
                status_code=422,
                detail={"validation_errors": issues},
            )

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


class RenderRequest(BaseModel):
    room_type: str
    style: str = "modern"
    width_ft: float = 12.0
    depth_ft: float = 12.0
    ceiling_height: int = 9


@app.post("/api/render/room")
async def render_room(request: RenderRequest):
    """Generate photorealistic AI room render via fal.ai Flux Schnell."""
    from render import generate_render
    try:
        result = await generate_render(
            request.room_type,
            request.style,
            request.width_ft,
            request.depth_ft,
            request.ceiling_height,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"Render API error: {e.response.status_code}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Render failed: {str(e)}")


@app.post("/api/export/pdf")
async def export_pdf(request: ExportRequest):
    try:
        from pdf_export import export_to_pdf
        pdf_bytes = export_to_pdf(request.floor_plan)
        name = request.floor_plan.get("name", "floor_plan").replace(" ", "_")
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{name}.pdf"'},
        )
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
