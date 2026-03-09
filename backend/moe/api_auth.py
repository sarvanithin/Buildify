"""
API key management and tier-based access control for monetization.

Tiers:
  - free:       5 generations/day, 1 variant, no export, basic scoring
  - pro:        unlimited, 3 variants, full export, full scoring
  - enterprise: unlimited, 5 variants, API access, custom training
"""
import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Optional, Dict
from functools import wraps
from fastapi import HTTPException, Request

from .config import MOEConfig

# ─────────────────────────────────────────────────────────────────────────────
# In-memory store (production: swap with Redis/Postgres)
# ─────────────────────────────────────────────────────────────────────────────

_KEYS_PATH = Path(__file__).parent / "keys.json"


class KeyStore:
    """Simple persistent key store."""

    def __init__(self):
        self._keys: Dict[str, dict] = {}
        self._load()

    def _load(self):
        if _KEYS_PATH.exists():
            try:
                self._keys = json.loads(_KEYS_PATH.read_text())
            except Exception:
                self._keys = {}

    def _save(self):
        _KEYS_PATH.write_text(json.dumps(self._keys, indent=2))

    def create_key(self, tier: str = "free", email: str = "") -> dict:
        """Generate a new API key."""
        raw = f"{uuid.uuid4().hex}{time.time()}{email}"
        api_key = f"bfy_{hashlib.sha256(raw.encode()).hexdigest()[:32]}"

        record = {
            "key": api_key,
            "tier": tier,
            "email": email,
            "created_at": time.time(),
            "usage": {
                "generations_today": 0,
                "total_generations": 0,
                "total_exports": 0,
                "total_chats": 0,
                "last_reset": time.time(),
            },
        }
        self._keys[api_key] = record
        self._save()
        return record

    def validate_key(self, api_key: str) -> Optional[dict]:
        """Validate an API key and return its record."""
        record = self._keys.get(api_key)
        if not record:
            return None

        # Reset daily counter if new day
        usage = record["usage"]
        now = time.time()
        if now - usage.get("last_reset", 0) > 86400:
            usage["generations_today"] = 0
            usage["last_reset"] = now
            self._save()

        return record

    def check_limit(self, api_key: str, action: str = "generation") -> bool:
        """Check if the key is within its tier limits."""
        record = self.validate_key(api_key)
        if not record:
            return False

        tier = record["tier"]
        config = MOEConfig()
        limit = config.TIER_LIMITS.get(tier, 5)

        if limit == -1:
            return True  # unlimited

        return record["usage"]["generations_today"] < limit

    def record_usage(self, api_key: str, action: str = "generation"):
        """Record a usage event."""
        record = self._keys.get(api_key)
        if not record:
            return

        usage = record["usage"]
        if action == "generation":
            usage["generations_today"] = usage.get("generations_today", 0) + 1
            usage["total_generations"] = usage.get("total_generations", 0) + 1
        elif action == "export":
            usage["total_exports"] = usage.get("total_exports", 0) + 1
        elif action == "chat":
            usage["total_chats"] = usage.get("total_chats", 0) + 1

        self._save()

    def get_usage(self, api_key: str) -> Optional[dict]:
        """Get usage stats for a key."""
        record = self.validate_key(api_key)
        if not record:
            return None

        config = MOEConfig()
        tier = record["tier"]
        limit = config.TIER_LIMITS.get(tier, 5)

        return {
            "tier": tier,
            "generations_today": record["usage"]["generations_today"],
            "daily_limit": limit if limit > 0 else "unlimited",
            "total_generations": record["usage"]["total_generations"],
            "total_exports": record["usage"]["total_exports"],
            "total_chats": record["usage"]["total_chats"],
            "variants_per_request": config.TIER_VARIANTS.get(tier, 1),
        }

    def upgrade_key(self, api_key: str, new_tier: str) -> Optional[dict]:
        """Upgrade a key to a higher tier."""
        record = self._keys.get(api_key)
        if not record:
            return None
        record["tier"] = new_tier
        self._save()
        return record


# Singleton
key_store = KeyStore()


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI middleware / dependency
# ─────────────────────────────────────────────────────────────────────────────

def get_api_key(request: Request) -> Optional[str]:
    """Extract API key from request headers or query params."""
    key = request.headers.get("X-API-Key")
    if not key:
        key = request.query_params.get("api_key")
    return key


def require_tier(min_tier: str = "free"):
    """Decorator to require a minimum tier for an endpoint."""
    tier_levels = {"free": 0, "pro": 1, "enterprise": 2}
    min_level = tier_levels.get(min_tier, 0)

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, request: Request = None, **kwargs):
            # Allow unauthenticated access for free-tier endpoints
            api_key = None
            if request:
                api_key = get_api_key(request)

            if min_level > 0 and not api_key:
                raise HTTPException(
                    status_code=401,
                    detail=f"API key required. This endpoint requires '{min_tier}' tier or higher."
                )

            if api_key:
                record = key_store.validate_key(api_key)
                if not record:
                    raise HTTPException(status_code=401, detail="Invalid API key.")

                user_level = tier_levels.get(record["tier"], 0)
                if user_level < min_level:
                    raise HTTPException(
                        status_code=403,
                        detail=f"Your tier '{record['tier']}' does not have access. "
                               f"Requires '{min_tier}' tier or higher."
                    )

                if not key_store.check_limit(api_key):
                    raise HTTPException(
                        status_code=429,
                        detail="Daily generation limit reached. Upgrade to Pro for unlimited."
                    )

            return await func(*args, **kwargs)
        return wrapper
    return decorator
