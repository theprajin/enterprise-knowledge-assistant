"""
API endpoints for prompt management and versioning.

Exposes the prompt registry for transparency and debugging.
"""

from fastapi import APIRouter, HTTPException
from app.prompts.registry import get_prompt_registry

router = APIRouter(
    prefix="/prompts",
    tags=["prompts"],
)


@router.get("/")
async def list_prompts():
    """List all registered prompt templates with version info."""
    registry = get_prompt_registry()
    return {
        "prompts": registry.list_prompts(),
        "total": len(registry.list_prompts()),
    }


@router.get("/{name}")
async def get_prompt(name: str, version: int | None = None):
    """
    Get a specific prompt template by name.
    
    Optionally specify a version number; defaults to the latest version.
    """
    registry = get_prompt_registry()
    try:
        prompt = registry.get(name, version)
        return {
            "prompt": prompt.to_dict(),
            "all_versions": registry.list_versions(name),
        }
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
